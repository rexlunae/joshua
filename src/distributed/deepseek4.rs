//! CPU tensor-parallel DeepSeek V4 inference with replicated attention and KV.
//!
//! This wrapper deliberately does not expose the model's session-cloning APIs:
//! one fixed cluster runs one ordered request, with identical inputs on all ranks.

use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
    sync::Arc,
};

use anyhow::{ensure, Context, Result};
use candle_core::{Device, Tensor};
use sha2::{Digest, Sha256};

use super::session::ClusterSession;
use crate::{gguf_ext, quantized_deepseek4::ModelWeights};

pub struct DeepSeekCluster {
    model: ModelWeights,
    cluster: Arc<ClusterSession>,
    context: usize,
    vocab: usize,
    position: usize,
}

impl DeepSeekCluster {
    /// Encoded local and unpartitioned routed-expert weight bytes.
    /// Excludes replicated dense/shared weights, KV, activations and page overhead.
    pub fn expert_shard_bytes(&self) -> (usize, usize) {
        self.model.distributed_expert_bytes()
    }

    /// Load an immutable local GGUF. Every rank must use identical file contents.
    ///
    /// Startup compares the header, file length and context limit, not all weight
    /// bytes (reading the entire file would defeat lazy shard loading). Verify a
    /// full-file checksum out of band before launch. No device expert cache,
    /// whole-file prefetch, or concurrent requests are enabled by this API.
    pub fn load(
        path: impl AsRef<Path>,
        n_ctx: usize,
        cluster: Arc<ClusterSession>,
    ) -> Result<Self> {
        cluster.claim_model()?;
        let result = (|| {
            let file = File::open(path.as_ref()).context("opening local cluster GGUF")?;
            let mut reader = BufReader::new(&file);
            let header = gguf_ext::read_header(&mut reader)?;
            ensure!(
                header.architecture().as_deref() == Some("deepseek4"),
                "cluster inference currently supports only DeepSeek V4"
            );
            let model_context = header
                .metadata
                .get("deepseek4.context_length")
                .context("missing DeepSeek context length")?
                .to_u32()? as usize;
            ensure!(
                n_ctx > 0 && n_ctx <= model_context.min(262_144),
                "cluster context must be positive and no larger than the model limit or 262144"
            );
            let embedding = header
                .tensors
                .get("token_embd.weight")
                .context("missing token embedding")?;
            ensure!(embedding.dims.len() == 2, "invalid token embedding shape");
            let vocab = embedding.dims[0];
            ensure!(vocab > 0, "empty model vocabulary");
            let file_len = file.metadata()?.len();
            ensure!(
                header.tensor_data_offset <= file_len,
                "truncated GGUF header"
            );
            reader.seek(SeekFrom::Start(0))?;
            let mut hasher = Sha256::new();
            hasher.update(b"joshua-deepseek4-cluster-v1");
            hasher.update(env!("CARGO_PKG_VERSION"));
            hasher.update((n_ctx as u64).to_le_bytes());
            hasher.update(file_len.to_le_bytes());
            let mut remaining = header.tensor_data_offset;
            let mut buffer = [0u8; 8192];
            while remaining > 0 {
                let len = remaining.min(buffer.len() as u64) as usize;
                reader.read_exact(&mut buffer[..len])?;
                hasher.update(&buffer[..len]);
                remaining -= len as u64;
            }
            cluster.agree(&hasher.finalize())?;
            // SAFETY: the caller must keep the local GGUF immutable for the
            // lifetime of the job, as with Joshua's other mmap loaders.
            let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file) }?);
            let model = ModelWeights::from_gguf_distributed(
                header.to_candle_content()?,
                &header,
                &mut reader,
                mmap,
                n_ctx,
                Arc::clone(&cluster),
            )?;
            Ok(Self {
                model,
                cluster: Arc::clone(&cluster),
                context: n_ctx,
                vocab,
                position: 0,
            })
        })();
        if result.is_err() {
            cluster.abort();
        }
        result
    }

    /// Prefill a chunk or decode a token, returning last-position logits.
    /// Offsets must be contiguous. Any failure requires a new job on every rank.
    pub fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>> {
        let result = (|| {
            ensure!(!tokens.is_empty(), "cluster input must not be empty");
            ensure!(
                offset == self.position,
                "cluster input offset is not contiguous"
            );
            let end = offset
                .checked_add(tokens.len())
                .context("context offset overflow")?;
            ensure!(end <= self.context, "cluster context exhausted");
            ensure!(
                tokens.iter().all(|&id| (id as usize) < self.vocab),
                "cluster input contains an out-of-vocabulary token"
            );
            let mut digest = Sha256::new();
            digest.update(b"joshua-deepseek4-forward-v1");
            digest.update((offset as u64).to_le_bytes());
            digest.update((tokens.len() as u64).to_le_bytes());
            for token in tokens {
                digest.update(token.to_le_bytes());
            }
            self.cluster.agree(&digest.finalize())?;
            let input = Tensor::new(tokens, &Device::Cpu)?.unsqueeze(0)?;
            let logits = self
                .model
                .forward(&input, offset)?
                .squeeze(0)?
                .to_vec1::<f32>()?;
            ensure!(
                logits.iter().all(|v| v.is_finite()),
                "nonfinite cluster logits"
            );
            self.position = end;
            Ok(logits)
        })();
        if result.is_err() {
            self.cluster.abort();
        }
        result
    }
}
