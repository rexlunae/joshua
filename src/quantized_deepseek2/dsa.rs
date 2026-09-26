//! DeepSeek Sparse Attention (DSA) for `glm-dsa` (GLM-5 / 5.1 / 5.2) and
//! `glm5next` (GLM-5.3-Flash): an indexer scores the cached keys for each
//! query and the MLA attention only sees the `top_k` best (llama.cpp
//! `src/models/glm-dsa.cpp` / `glm5-next.cpp`, HF `GlmMoeDsaIndexer` /
//! `Glm5NextTextIndexer`).
//!
//! * Queries come from the attention's Q latent (`q_a_norm(q_a(x))`) through
//!   `indexer.attn_q_b`, one key per position from `indexer.attn_k` and a
//!   LayerNorm; both are RoPE'd over their leading `n_rot` dims with the
//!   attention's own (interleaved) table.
//! * A key's score for a query is `Σ_h relu(q_h · k) · w_h`, where the
//!   per-head weights `w = indexer.proj(x) / sqrt(head_dim · n_head)`.
//! * GLM-5.2's IndexShare: only the "full" layers run an indexer; each
//!   "shared" layer reuses the selection of the full layer before it.
//! * GLM-5.3-Flash's k-pool variant scores *pools* of `kpool` consecutive
//!   keys instead of single keys: a complete pool's key is the per-channel
//!   softmax-weighted mix of its keys (weights `indexer_compressor_gate(x) +
//!   indexer_compressor_ape`), the best `top_k / kpool` pools are expanded
//!   back to their tokens, and the query's own incomplete pool (the "tail")
//!   is always visible.  Neither queries nor keys carry RoPE (the model is
//!   NoPE throughout).
//!
//! llama.cpp rotates the indexer queries and keys by an orthonormal
//! Walsh–Hadamard matrix before scoring (so their cache quantizes well);
//! an orthonormal rotation leaves every dot product unchanged, so it is
//! omitted here.

use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::quantized::QMatMul;
use candle_core::{Device, Module, Result, Tensor, D};

use crate::attention::Rope;
use crate::gguf_meta::Meta;

/// The `attention.indexer.*` settings.
#[derive(Debug, Clone)]
pub(super) struct DsaConfig {
    n_head: usize,
    head_dim: usize,
    top_k: usize,
    /// The indexer key LayerNorm's epsilon (HF: 1e-6).
    eps: f64,
    /// Keys per pool (`glm5next`); 0 for per-key scoring (`glm-dsa`).
    kpool: usize,
    /// Per layer: runs its own indexer (`true`) or reuses the selection of
    /// the previous full layer.
    pub(super) full: Vec<bool>,
}

impl DsaConfig {
    pub(super) fn from_meta(
        m: &Meta<'_>,
        n_layer: usize,
        n_rot: usize,
        context_length: usize,
    ) -> Result<Self> {
        let n_head = m.u32("attention.indexer.head_count")? as usize;
        let head_dim = m.u32("attention.indexer.key_length")? as usize;
        let top_k = m.u32("attention.indexer.top_k")? as usize;
        if n_head == 0 || top_k == 0 || n_rot > head_dim {
            candle_core::bail!(
                "{}: invalid indexer ({n_head} heads of {head_dim}, top-k {top_k}, {n_rot} rotary dims)",
                m.arch()
            );
        }
        let kpool = m.u32_or("attention.indexer.kpool", 0) as usize;
        if kpool > 0
            && (!top_k.is_multiple_of(kpool)
                || !m.bool_or("attention.indexer.kpool_select_tail", true))
        {
            candle_core::bail!(
                "{}: the k-pool indexer needs a top-k ({top_k}) that is a whole number of pools of \
                 {kpool} and the tail pool always selected",
                m.arch()
            );
        }
        // GGUFs without `indexer.types` follow llama.cpp: GLM-5.3 indexes on
        // every sparse layer, GLM-5 / 5.1 (under a 1M context) on every
        // layer, GLM-5.2 on layers 0, 1 and every fourth from 2.
        let full = m
            .array_bool("attention.indexer.types", n_layer)
            .unwrap_or_else(|| {
                (0..n_layer)
                    .map(|i| kpool > 0 || context_length < 1 << 20 || i < 2 || (i - 2) % 4 == 0)
                    .collect()
            });
        Ok(Self {
            n_head,
            head_dim,
            top_k,
            eps: m.f32_or("attention.layer_norm_epsilon", 1e-6) as f64,
            kpool,
            full,
        })
    }
}

/// The keys each query of one input chunk may attend, as published by a
/// full indexer layer for the shared layers after it.
#[derive(Debug, Clone)]
pub(super) struct Selection {
    /// Position of the chunk's first query.
    pub(super) offset: usize,
    /// Per query: the selected key positions, or empty when the query sees
    /// every earlier position (it has no more than `top_k` of them).
    rows: Vec<Vec<u32>>,
}

impl Selection {
    /// The additive attention mask `[1, 1, t, offset + t]` restricting each
    /// query to its selection, or `None` when no query is restricted (the
    /// caller's causal mask then applies unchanged).
    pub(super) fn mask(&self, dev: &Device) -> Result<Option<Tensor>> {
        if self.rows.iter().all(|r| r.is_empty()) {
            return Ok(None);
        }
        let t = self.rows.len();
        let len = self.offset + t;
        let mut mask = vec![f32::NEG_INFINITY; t * len];
        for (i, picks) in self.rows.iter().enumerate() {
            let row = &mut mask[i * len..(i + 1) * len];
            if picks.is_empty() {
                row[..=self.offset + i].fill(0.0);
            } else {
                for &s in picks {
                    row[s as usize] = 0.0;
                }
            }
        }
        Tensor::from_vec(mask, (1, 1, t, len), dev).map(Some)
    }
}

/// One layer's indexer state: the per-position keys (and, for the k-pool
/// indexer, pool gates and the pooled keys of every complete pool).
#[derive(Clone, Default)]
pub(super) struct IndexCache {
    keys: Option<Tensor>,   // [len, head_dim]
    gates: Option<Tensor>,  // [len, head_dim]
    pooled: Option<Tensor>, // [len / kpool, head_dim]
}

impl IndexCache {
    /// Keep the first `keep` positions.  The pooled keys are dropped and
    /// re-pooled from the kept keys on the next selection.
    pub(super) fn truncate(&mut self, keep: usize) -> Result<()> {
        let cut = |t: &mut Option<Tensor>| -> Result<()> {
            if let Some(x) = t.as_ref() {
                if keep < x.dim(0)? {
                    *t = (keep > 0).then(|| x.narrow(0, 0, keep)).transpose()?;
                }
            }
            Ok(())
        };
        cut(&mut self.keys)?;
        cut(&mut self.gates)?;
        self.pooled = None;
        Ok(())
    }
}

/// Append `new` rows to a cached `[len, d]` tensor.
fn append(cache: &mut Option<Tensor>, new: Tensor) -> Result<Tensor> {
    let all = match cache.take() {
        Some(prev) => Tensor::cat(&[&prev, &new], 0)?,
        None => new,
    };
    *cache = Some(all.clone());
    Ok(all)
}

/// How an indexer builds the keys it scores.
enum Keys {
    /// One RoPE'd key per position (`glm-dsa`).
    Lightning { rope: Arc<Rope> },
    /// One learned mix per pool of `size` positions (`glm5next`).
    KPool {
        size: usize,
        gate: QMatMul,
        ape: Tensor,
    },
}

/// One full layer's indexer.
pub(super) struct Indexer {
    q_b: QMatMul, // q_lora_rank → n_head·head_dim
    k: QMatMul,   // n_embd → head_dim
    k_norm_w: Tensor,
    k_norm_b: Option<Tensor>,
    proj: QMatMul, // n_embd → n_head
    keys: Keys,
    n_head: usize,
    head_dim: usize,
    top_k: usize,
    eps: f64,
}

impl Indexer {
    /// `rope` is the attention's table (unused by the NoPE k-pool indexer).
    pub(super) fn load<R: Read + Seek>(
        rd: &mut super::Reader<R>,
        p: &str,
        cfg: &DsaConfig,
        rope: Option<Arc<Rope>>,
    ) -> Result<Self> {
        let keys = if cfg.kpool > 0 {
            Keys::KPool {
                size: cfg.kpool,
                gate: rd.qmatmul(&format!("{p}.indexer_compressor_gate.weight"))?,
                ape: rd.f32_tensor(&format!("{p}.indexer_compressor_ape.weight"))?,
            }
        } else {
            let Some(rope) = rope else {
                candle_core::bail!("{}: the lightning indexer needs rotary dims", rd.arch);
            };
            Keys::Lightning { rope }
        };
        Ok(Self {
            q_b: rd.qmatmul(&format!("{p}.indexer.attn_q_b.weight"))?,
            k: rd.qmatmul(&format!("{p}.indexer.attn_k.weight"))?,
            k_norm_w: rd.f32_tensor(&format!("{p}.indexer.k_norm.weight"))?,
            k_norm_b: rd.f32_opt(&format!("{p}.indexer.k_norm.bias"))?,
            proj: rd.qmatmul(&format!("{p}.indexer.proj.weight"))?,
            keys,
            n_head: cfg.n_head,
            head_dim: cfg.head_dim,
            top_k: cfg.top_k,
            eps: cfg.eps,
        })
    }

    /// Append this chunk's indexer state to `cache` and select the keys
    /// each query attends.  `x` is the attention input `[1, t, n_embd]`,
    /// `q_latent` the Q latent `[1, t, q_lora_rank]`.
    pub(super) fn select(
        &self,
        cache: &mut IndexCache,
        x: &Tensor,
        q_latent: &Tensor,
        offset: usize,
    ) -> Result<Selection> {
        let (_, t, _) = x.dims3()?;
        let d = self.head_dim;
        let len = offset + t;

        // Keys: LayerNorm, then (lightning) RoPE over the leading dims.
        let k = self.k.forward(x)?.reshape((t, d))?;
        let mean = k.mean_keepdim(D::Minus1)?;
        let centred = k.broadcast_sub(&mean)?;
        let var = centred.sqr()?.mean_keepdim(D::Minus1)?;
        let mut k = centred
            .broadcast_div(&(var + self.eps)?.sqrt()?)?
            .broadcast_mul(&self.k_norm_w)?;
        if let Some(b) = &self.k_norm_b {
            k = k.broadcast_add(b)?;
        }
        // What each query scores (`[n, d]`), how many candidates the query at
        // `pos` may pick from, and how many of them it keeps.
        let (cands, per_query, keep): (Tensor, Box<dyn Fn(usize) -> usize>, usize) =
            match &self.keys {
                Keys::Lightning { rope } => {
                    let k = rope
                        .apply_leading(&k.reshape((1, 1, t, d))?, offset)?
                        .reshape((t, d))?;
                    (
                        append(&mut cache.keys, k)?,
                        Box::new(|pos| pos + 1),
                        self.top_k,
                    )
                }
                Keys::KPool { size, gate, ape } => {
                    let size = *size;
                    let keys = append(&mut cache.keys, k)?;
                    let gates = append(&mut cache.gates, gate.forward(x)?.reshape((t, d))?)?;
                    // Pool every block this chunk completed.
                    let have = cache.pooled.as_ref().map_or(Ok(0), |p| p.dim(0))?;
                    let n_pools = len / size;
                    if n_pools > have {
                        let n = n_pools - have;
                        let rows =
                            |c: &Tensor| c.narrow(0, have * size, n * size)?.reshape((n, size, d));
                        let w = candle_nn::ops::softmax(&rows(&gates)?.broadcast_add(ape)?, 1)?;
                        append(&mut cache.pooled, (w * rows(&keys)?)?.sum(1)?)?;
                    }
                    let Some(pooled) = cache.pooled.clone() else {
                        return Ok(Selection {
                            offset,
                            rows: vec![Vec::new(); t],
                        });
                    };
                    (
                        pooled,
                        Box::new(move |pos| (pos + 1) / size),
                        self.top_k / size,
                    )
                }
            };

        // A query with no more candidates than it keeps sees everything
        // (for k-pool, every complete pool plus its tail).
        if (offset..len).all(|pos| per_query(pos) <= keep) {
            return Ok(Selection {
                offset,
                rows: vec![Vec::new(); t],
            });
        }

        let q = self
            .q_b
            .forward(q_latent)?
            .reshape((1, t, self.n_head, d))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = match &self.keys {
            Keys::Lightning { rope } => rope.apply_leading(&q, offset)?,
            Keys::KPool { .. } => q,
        }
        .squeeze(0)?; // [heads, t, d]
        let w =
            (self.proj.forward(x)?.reshape((t, self.n_head))? / ((d * self.n_head) as f64).sqrt())?;
        let scores = q
            .broadcast_matmul(&cands.t()?.contiguous()?)? // [heads, t, n]
            .relu()?
            .broadcast_mul(&w.t()?.unsqueeze(2)?)?
            .sum(0)?
            .to_vec2::<f32>()?;

        let rows = scores
            .iter()
            .enumerate()
            .map(|(i, score)| {
                let pos = offset + i;
                let n = per_query(pos);
                if n <= keep {
                    return Vec::new();
                }
                // The best `keep` candidates (earlier first on a tie).
                let mut order: Vec<u32> = (0..n as u32).collect();
                let cmp = |a: &u32, b: &u32| {
                    score[*b as usize]
                        .total_cmp(&score[*a as usize])
                        .then(a.cmp(b))
                };
                order.select_nth_unstable_by(keep - 1, cmp);
                order.truncate(keep);
                match &self.keys {
                    Keys::Lightning { .. } => order,
                    // Expand the pools to their positions and add the tail.
                    Keys::KPool { size, .. } => {
                        let size = *size as u32;
                        let tail = (n as u32 * size)..=pos as u32;
                        order
                            .iter()
                            .flat_map(|&p| p * size..(p + 1) * size)
                            .chain(tail)
                            .collect()
                    }
                }
            })
            .collect();
        Ok(Selection { offset, rows })
    }
}
