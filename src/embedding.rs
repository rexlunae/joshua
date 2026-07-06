//! Dense text embeddings from GGUF models.
//!
//! candle's quantized generation models only expose LM-head logits, so this
//! module implements its own single-pass decoder forward that stops at the
//! final hidden states — which is what sentence embeddings pool from.  It
//! supports the decoder families used by the popular GGUF embedding models:
//!
//! | `general.architecture` | example models |
//! |------------------------|----------------|
//! | `llama`                | e5-mistral-7b-instruct, SFR-Embedding |
//! | `qwen2`                | gte-Qwen2-1.5B/7B-instruct |
//! | `qwen3`                | Qwen3-Embedding-0.6B/4B/8B |
//!
//! The forward pass mirrors candle's `quantized_llama` / `quantized_qwen2` /
//! `quantized_qwen3` op-for-op (verified in the test suite by reproducing
//! their LM-head logits exactly), differing only in that no KV cache is kept
//! — embedding extraction is a single full-sequence pass.
//!
//! Pooling follows the GGUF `{arch}.pooling_type` metadata written by
//! llama.cpp's converters (mean / CLS / last-token), defaulting to mean, and
//! embeddings are L2-normalised.

use std::io::{Read, Seek};

use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};

// ─── Configuration ──────────────────────────────────────────────────────────

/// How token-level hidden states are pooled into one sentence vector.
///
/// Values match llama.cpp's `llama_pooling_type` metadata encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Average over all positions (`pooling_type = 1`, also the default).
    Mean,
    /// First position — CLS-style (`pooling_type = 2`).
    Cls,
    /// Final position — the convention for causal embedding models
    /// (`pooling_type = 3`).
    Last,
}

struct EmbedConfig {
    /// NEOX (non-interleaved) RoPE instead of interleaved.
    rope_is_neox: bool,
    /// Q/K/V projections carry bias vectors (Qwen2).
    qkv_bias: bool,
    /// Per-head RMS norm on Q/K before RoPE (Qwen3).
    qk_norm: bool,
}

impl EmbedConfig {
    fn for_arch(arch: &str) -> Option<Self> {
        Some(match arch {
            "llama" => Self {
                rope_is_neox: false,
                qkv_bias: false,
                qk_norm: false,
            },
            "qwen2" => Self {
                rope_is_neox: true,
                qkv_bias: true,
                qk_norm: false,
            },
            "qwen3" => Self {
                rope_is_neox: true,
                qkv_bias: false,
                qk_norm: true,
            },
            _ => return None,
        })
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

struct Layer {
    attn_norm: RmsNorm,
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    bq: Option<Tensor>,
    bk: Option<Tensor>,
    bv: Option<Tensor>,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    ffn_norm: RmsNorm,
    ffn_gate: QMatMul,
    ffn_down: QMatMul,
    ffn_up: QMatMul,
}

/// A decoder model evaluated for hidden states rather than logits.
pub struct EmbeddingModel {
    tok_embeddings: Tensor,
    layers: Vec<Layer>,
    output_norm: RmsNorm,
    /// LM head, kept for logit-parity validation against candle's
    /// generation models (see [`EmbeddingModel::logits`]).
    output: QMatMul,
    cos: Tensor,
    sin: Tensor,
    rope_is_neox: bool,
    qk_norm_eps: f64,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    pooling: Pooling,
}

/// RMS norm with an owned weight vector.
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn from_qtensor(qt: QTensor, eps: f64, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: qt.dequantize(device)?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps as f32)
    }
}

impl EmbeddingModel {
    /// Architectures this embedding path supports.
    pub fn supported_archs() -> &'static str {
        "llama, qwen2, qwen3"
    }

    /// Build from GGUF content, dispatching on `general.architecture`.
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md = &ct.metadata;
        let arch = md
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        let config = EmbedConfig::for_arch(&arch).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "Embeddings are not supported for architecture '{arch}'. \
                 Supported embedding architectures: {}.",
                Self::supported_archs()
            ))
        })?;

        let get_u32 = |key: &str| -> Result<u32> {
            match md.get(&format!("{arch}.{key}")) {
                Some(v) => v.to_u32(),
                None => candle_core::bail!("cannot find {arch}.{key} in metadata"),
            }
        };
        let get_f32 = |key: &str, default: f32| -> f32 {
            md.get(&format!("{arch}.{key}"))
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(default)
        };

        let n_head = get_u32("attention.head_count")? as usize;
        let n_kv_head = get_u32("attention.head_count_kv")? as usize;
        let block_count = get_u32("block_count")? as usize;
        let embedding_length = get_u32("embedding_length")? as usize;
        let rms_eps = get_f32("attention.layer_norm_rms_epsilon", 1e-5) as f64;
        let rope_freq_base = get_f32("rope.freq_base", 10_000.0);

        // Head dimension: explicit metadata where present (Qwen3 decouples it
        // from the embedding width), otherwise embedding / heads.  For llama,
        // rope.dimension_count == head_dim in all GGUF conversions.
        let head_dim = get_u32("attention.key_length")
            .or_else(|_| get_u32("rope.dimension_count"))
            .map(|v| v as usize)
            .unwrap_or(embedding_length / n_head);

        // RoPE tables sized to the model's context window (bounded to keep
        // the table small; embedding inputs are short).
        let max_seq = get_u32("context_length")
            .map(|v| v as usize)
            .unwrap_or(4096)
            .min(32_768);
        let (cos, sin) = precompute_freqs_cis(head_dim, rope_freq_base, max_seq, device)?;

        let pooling = match md
            .get(&format!("{arch}.pooling_type"))
            .and_then(|v| v.to_u32().ok())
        {
            Some(2) => Pooling::Cls,
            Some(3) => Pooling::Last,
            Some(1) => Pooling::Mean,
            Some(other) => {
                tracing::warn!("Unknown pooling_type {other} in GGUF — using mean pooling");
                Pooling::Mean
            }
            // Absent: mean is llama.cpp's conventional default for embeddings.
            None => Pooling::Mean,
        };

        let tensor = |reader: &mut R, name: &str| ct.tensor(reader, name, device);

        let tok_embeddings_q = tensor(reader, "token_embd.weight")?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let output_norm =
            RmsNorm::from_qtensor(tensor(reader, "output_norm.weight")?, rms_eps, device)?;
        // Tied embeddings when no output.weight is present, mirroring candle.
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(t) => QMatMul::from_qtensor(t)?,
            Err(_) => QMatMul::from_qtensor(tok_embeddings_q)?,
        };

        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let p = format!("blk.{i}");
            let bias = |reader: &mut R, name: String| -> Result<Option<Tensor>> {
                if config.qkv_bias {
                    Ok(Some(ct.tensor(reader, &name, device)?.dequantize(device)?))
                } else {
                    Ok(None)
                }
            };
            let norm = |reader: &mut R, name: String| -> Result<Option<RmsNorm>> {
                if config.qk_norm {
                    Ok(Some(RmsNorm::from_qtensor(
                        ct.tensor(reader, &name, device)?,
                        rms_eps,
                        device,
                    )?))
                } else {
                    Ok(None)
                }
            };
            layers.push(Layer {
                attn_norm: RmsNorm::from_qtensor(
                    tensor(reader, &format!("{p}.attn_norm.weight"))?,
                    rms_eps,
                    device,
                )?,
                wq: QMatMul::from_qtensor(tensor(reader, &format!("{p}.attn_q.weight"))?)?,
                wk: QMatMul::from_qtensor(tensor(reader, &format!("{p}.attn_k.weight"))?)?,
                wv: QMatMul::from_qtensor(tensor(reader, &format!("{p}.attn_v.weight"))?)?,
                wo: QMatMul::from_qtensor(tensor(reader, &format!("{p}.attn_output.weight"))?)?,
                bq: bias(reader, format!("{p}.attn_q.bias"))?,
                bk: bias(reader, format!("{p}.attn_k.bias"))?,
                bv: bias(reader, format!("{p}.attn_v.bias"))?,
                q_norm: norm(reader, format!("{p}.attn_q_norm.weight"))?,
                k_norm: norm(reader, format!("{p}.attn_k_norm.weight"))?,
                ffn_norm: RmsNorm::from_qtensor(
                    tensor(reader, &format!("{p}.ffn_norm.weight"))?,
                    rms_eps,
                    device,
                )?,
                ffn_gate: QMatMul::from_qtensor(tensor(reader, &format!("{p}.ffn_gate.weight"))?)?,
                ffn_down: QMatMul::from_qtensor(tensor(reader, &format!("{p}.ffn_down.weight"))?)?,
                ffn_up: QMatMul::from_qtensor(tensor(reader, &format!("{p}.ffn_up.weight"))?)?,
            });
        }

        Ok(Self {
            tok_embeddings,
            layers,
            output_norm,
            output,
            cos,
            sin,
            rope_is_neox: config.rope_is_neox,
            qk_norm_eps: rms_eps,
            n_head,
            n_kv_head,
            head_dim,
            pooling,
        })
    }

    /// The pooling strategy taken from the GGUF metadata.
    pub fn pooling(&self) -> Pooling {
        self.pooling
    }

    /// Run the decoder over `tokens` and return the final hidden states,
    /// shape `[seq_len, hidden]`.
    fn hidden_states(&self, tokens: &[u32]) -> Result<Tensor> {
        let _ = self.qk_norm_eps; // eps lives inside each RmsNorm
        let seq_len = tokens.len();
        let device = self.tok_embeddings.device();
        let input = Tensor::new(tokens, device)?;
        let mut x = self.tok_embeddings.index_select(&input, 0)?.unsqueeze(0)?;

        let mask = if seq_len > 1 {
            Some(causal_mask(seq_len, device)?)
        } else {
            None
        };

        for layer in &self.layers {
            let residual = &x;
            let h = layer.attn_norm.forward(&x)?;
            let attn = self.attention(layer, &h, mask.as_ref(), seq_len)?;
            let x2 = (attn + residual)?;

            let residual = &x2;
            let h = layer.ffn_norm.forward(&x2)?;
            let gate = candle_nn::ops::silu(&layer.ffn_gate.forward(&h)?)?;
            let up = layer.ffn_up.forward(&h)?;
            let mlp = layer.ffn_down.forward(&(gate * up)?)?;
            x = (mlp + residual)?;
        }

        self.output_norm.forward(&x)?.squeeze(0)
    }

    /// Standard multi-head attention over the full sequence (no KV cache).
    fn attention(
        &self,
        layer: &Layer,
        x: &Tensor,
        mask: Option<&Tensor>,
        seq_len: usize,
    ) -> Result<Tensor> {
        let (b_sz, _, _) = x.dims3()?;

        let mut q = layer.wq.forward(x)?;
        let mut k = layer.wk.forward(x)?;
        let mut v = layer.wv.forward(x)?;
        if let (Some(bq), Some(bk), Some(bv)) = (&layer.bq, &layer.bk, &layer.bv) {
            q = q.broadcast_add(bq)?;
            k = k.broadcast_add(bk)?;
            v = v.broadcast_add(bv)?;
        }

        let mut q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let mut k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Per-head Q/K norm (Qwen3), applied before RoPE.
        if let (Some(qn), Some(kn)) = (&layer.q_norm, &layer.k_norm) {
            q = qn
                .forward(&q.flatten(0, 2)?)?
                .reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
            k = kn
                .forward(&k.flatten(0, 2)?)?
                .reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
        }

        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let (q, k) = if self.rope_is_neox {
            (
                candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?,
                candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?,
            )
        } else {
            (
                candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?,
                candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?,
            )
        };

        // Grouped-query attention: repeat KV heads to match Q heads.
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => att.broadcast_add(mask)?,
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;
        let y = y
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.n_head * self.head_dim))?;
        layer.wo.forward(&y)
    }

    /// Compute a pooled, L2-normalised sentence embedding.
    pub fn embed_tokens(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            candle_core::bail!("cannot embed an empty token sequence");
        }
        let hidden = self.hidden_states(tokens)?; // [seq, hidden]
        let pooled = match self.pooling {
            Pooling::Mean => hidden.mean(0)?,
            Pooling::Cls => hidden.get(0)?,
            Pooling::Last => hidden.get(tokens.len() - 1)?,
        };
        // L2 normalise.
        let norm = pooled.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let pooled = if norm > 0.0 {
            (pooled / norm as f64)?
        } else {
            pooled
        };
        pooled.to_vec1::<f32>()
    }

    /// Last-position LM-head logits for `tokens`.
    ///
    /// Exists to validate this forward pass against candle's generation
    /// models: on identical weights the logits must match exactly.
    pub fn logits(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.hidden_states(tokens)?;
        let last = hidden.get(tokens.len() - 1)?.unsqueeze(0)?;
        self.output.forward(&last)?.squeeze(0)?.to_vec1::<f32>()
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// RoPE cos/sin tables, matching candle's `precomput_freqs_cis`.
fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    max_seq: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, max_seq as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((max_seq, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((idx_theta.cos()?, idx_theta.sin()?))
}

/// Additive causal mask: 0 on/below the diagonal, −∞ above.
fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)
}

/// Repeat KV heads `n_rep` times along the head axis (GQA).
fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, n_kv, seq, hd) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv, n_rep, seq, hd))?
        .reshape((b, n_kv * n_rep, seq, hd))
}

