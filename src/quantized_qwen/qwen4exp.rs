//! The three pieces `qwen4exp` (Qwen3.8-Flash-Next) adds on top of the
//! Qwen3.5-MoE decoder, following llama.cpp's `src/models/qwen4exp.cpp`:
//!
//! * [`HyperMix`] — **Hyper-Connections** replace every layer norm: the
//!   residual is `hc` parallel streams; before each block a mixer
//!   RMS-normalises each stream, gates it through a low-rank sigmoid and
//!   averages the streams into the block input, and after the block every
//!   stream adds the block output scaled by its own `2·sigmoid(inject)`.
//!   The final mixer doubles as the output norm.
//! * [`Qsa`] — **QSA block-sparse attention** on the full-attention layers:
//!   an indexer scores mean-pooled blocks of `compress_ratio` cached keys
//!   and each query attends only to its best `indexer.top_k` cells plus the
//!   incomplete tail block.
//! * [`Ple`] — **PLE n-gram hash embeddings** on one linear-attention layer:
//!   each token hashes its preceding n-grams into rows of a shared table,
//!   which are gated into every stream and fed through a dilated causal conv.

use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::ops::{sigmoid, silu};
use candle_transformers::quantized_nn::RmsNorm;

use super::{Reader, Weight};
use crate::attention::Rope;
use crate::gguf_meta::Meta;
use crate::token_embedding::TokenEmbedding;

/// `x / sqrt(mean(x²) + eps) · w` over the last dim, with a per-stream
/// weight `w` broadcast over `x`'s leading dims (llama.cpp's grouped
/// `ggml_rms_norm` + `ggml_mul` on `[n_embd, hc]` gammas).
fn grouped_rms(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    x.broadcast_div(&rms)?.broadcast_mul(w)
}

/// An `[n_embd, hc]`-shaped GGUF gamma as a `[hc, n_embd]` f32 tensor.
fn stream_gamma<R: Read + Seek>(
    rd: &mut Reader<R>,
    name: &str,
    hc: usize,
    n: usize,
) -> Result<Tensor> {
    rd.f32_tensor(name)?.reshape((hc, n))
}

// ─── Hyper-Connections ───────────────────────────────────────────────────────

/// One hyper-connection mixer (`hc_attn_*`, `hc_ffn_*` or `output_hc_*`).
pub(super) struct HyperMix {
    norm: Tensor, // [hc, n_embd]
    down: Weight, // hc·n_embd → low_rank
    up: Weight,   // low_rank → hc·n_embd
    /// The per-stream scatter weights; absent on the output mixer.
    inject: Option<Weight>, // hc·n_embd → hc
    hc: usize,
    eps: f64,
}

impl HyperMix {
    pub(super) fn load<R: Read + Seek>(
        rd: &mut Reader<R>,
        prefix: &str,
        hc: usize,
        n_embd: usize,
        eps: f64,
    ) -> Result<Self> {
        let inject = format!("{prefix}_inject.weight");
        Ok(Self {
            norm: stream_gamma(rd, &format!("{prefix}_norm.weight"), hc, n_embd)?,
            down: rd.qmatmul(&format!("{prefix}_down.weight"))?,
            up: rd.qmatmul(&format!("{prefix}_up.weight"))?,
            inject: if rd.has(&inject) {
                Some(rd.qmatmul(&inject)?)
            } else {
                None
            },
            hc,
            eps,
        })
    }

    /// Mix the streams `res` (`[1, t, hc·n]`) into one block input
    /// `[1, t, n]`, plus the scatter logits `[1, t, hc]` for [`Self::combine`].
    pub(super) fn mix(&self, res: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let (b, t, wide) = res.dims3()?;
        let n = wide / self.hc;
        let xn = grouped_rms(&res.reshape((b, t, self.hc, n))?, &self.norm, self.eps)?
            .reshape((b, t, wide))?;
        let lo = silu(&(self.down.forward(&xn)? / self.hc as f64)?)?;
        let gate = self.up.forward(&lo)?;
        let mixed = (xn.clone() * sigmoid(&gate)?)?
            .reshape((b, t, self.hc, n))?
            .mean(2)?;
        let inject = self.inject.as_ref().map(|w| w.forward(&xn)).transpose()?;
        Ok((mixed, inject))
    }

    /// Add the block output `out` (`[1, t, n]`) to every stream of `res`,
    /// scaled per stream by `2·sigmoid(inject / hc)` — a zero injection is a
    /// plain residual add.
    pub(super) fn combine(&self, res: &Tensor, out: &Tensor, inject: &Tensor) -> Result<Tensor> {
        let (b, t, wide) = res.dims3()?;
        let w = (sigmoid(&(inject / self.hc as f64)?)? * 2.0)?; // [b, t, hc]
        let add = out.unsqueeze(2)?.broadcast_mul(&w.unsqueeze(3)?)?; // [b, t, hc, n]
        res + add.reshape((b, t, wide))?
    }
}

// ─── PLE n-gram hash embeddings ──────────────────────────────────────────────

/// The n-gram hash that picks each token's PLE table rows.
#[derive(Debug, Clone)]
struct PleHash {
    ngram: usize,
    heads_per_ngram: usize,
    eos: u32,
    multipliers: Vec<u64>,
    head_offsets: Vec<u64>,
    head_vocab: Vec<u64>,
}

impl PleHash {
    fn n_heads(&self) -> usize {
        (self.ngram - 1) * self.heads_per_ngram
    }

    /// Table rows `[t × n_heads]` for the last `t` tokens of `history`.
    ///
    /// Token `i`'s context is itself plus its `ngram - 1` predecessors, read
    /// as EOS before the sequence start and at or before an EOS (a token's
    /// own EOS does not cut it).  For each n-gram size `n` the hash is
    /// `ctx[0]·m[0] ^ … ^ ctx[n-1]·m[n-1]` (wrapping u64), reduced modulo
    /// each head's vocabulary and offset into the shared table.
    fn rows(&self, history: &[u32], t: usize) -> Vec<u32> {
        let start = history.len() - t;
        let mut out = Vec::with_capacity(t * self.n_heads());
        let mut ctx = vec![0u64; self.ngram];
        for i in start..history.len() {
            ctx[0] = history[i] as u64;
            let mut cut = false;
            for (s, slot) in ctx.iter_mut().enumerate().skip(1) {
                let prev = i.checked_sub(s).map(|p| history[p]);
                cut = cut || prev.is_none_or(|tok| tok == self.eos);
                *slot = if cut {
                    self.eos as u64
                } else {
                    prev.unwrap() as u64
                };
            }
            for n in 2..=self.ngram {
                let mixed = (0..n).fold(0u64, |acc, j| {
                    acc ^ ctx[j].wrapping_mul(self.multipliers[j])
                });
                for g in 0..self.heads_per_ngram {
                    let h = (n - 2) * self.heads_per_ngram + g;
                    out.push((mixed % self.head_vocab[h] + self.head_offsets[h]) as u32);
                }
            }
        }
        out
    }
}

/// `ple.*` settings: which layer carries the PLE block and its hash.
#[derive(Debug, Clone)]
pub(super) struct PleConfig {
    pub(super) layer: usize,
    hash: PleHash,
    kernel: usize,
}

impl PleConfig {
    /// `None` when the file configures no PLE layer.
    pub(super) fn from_meta(m: &Meta<'_>, n_layer: usize) -> Result<Option<Self>> {
        let layers = m.array_u64("ple.layers").unwrap_or_default();
        let layer = match layers.as_slice() {
            [] => return Ok(None),
            &[l] if (l as usize) < n_layer => l as usize,
            other => candle_core::bail!(
                "{}: expected one PLE layer below {n_layer}, got {other:?}",
                m.arch()
            ),
        };
        let ngram = m.u32("ple.ngram_size")? as usize;
        let heads_per_ngram = m.u32("ple.heads_per_ngram")? as usize;
        let n_heads = ngram.saturating_sub(1) * heads_per_ngram;
        let need = |key: &str, n: usize| -> Result<Vec<u64>> {
            match m.array_u64(key) {
                Some(v) if v.len() >= n => Ok(v),
                v => candle_core::bail!(
                    "{}: `{}` needs {n} entries, got {:?}",
                    m.arch(),
                    m.key(key),
                    v.map(|v| v.len())
                ),
            }
        };
        let hash = PleHash {
            ngram,
            heads_per_ngram,
            eos: m.u32("ple.eos_token_id")?,
            multipliers: need("ple.layer_multipliers", ngram)?,
            head_offsets: need("ple.head_offsets", n_heads)?,
            head_vocab: need("ple.head_vocab_sizes", n_heads)?,
        };
        let kernel = m.u32("ple.conv_kernel")? as usize;
        if ngram < 2 || n_heads == 0 || kernel == 0 || hash.head_vocab[..n_heads].contains(&0) {
            candle_core::bail!("{}: invalid PLE n-gram configuration", m.arch());
        }
        Ok(Some(Self {
            layer,
            hash,
            kernel,
        }))
    }
}

/// The PLE block of one layer.
pub(super) struct Ple {
    hash: PleHash,
    table: TokenEmbedding,
    key: Weight,   // n_embd → hc·n_embd
    value: Weight, // n_embd → n_embd
    norm_key: Tensor,
    norm_query: Tensor,
    norm_conv: Tensor,
    conv: Tensor, // [hc·n_embd, kernel]
    kernel: usize,
    hc: usize,
    eps: f64,
}

impl Ple {
    pub(super) fn load<R: Read + Seek>(
        rd: &mut Reader<R>,
        cfg: &PleConfig,
        p: &str,
        hc: usize,
        n_embd: usize,
        eps: f64,
    ) -> Result<Self> {
        let wide = hc * n_embd;
        let device = rd.device.clone();
        Ok(Self {
            hash: cfg.hash.clone(),
            table: TokenEmbedding::load(rd.qtensor("per_layer_token_embd.weight")?, &device)?,
            key: rd.qmatmul(&format!("{p}.ple_key.weight"))?,
            value: rd.qmatmul(&format!("{p}.ple_value.weight"))?,
            norm_key: stream_gamma(rd, &format!("{p}.ple_norm_key.weight"), hc, n_embd)?,
            norm_query: stream_gamma(rd, &format!("{p}.ple_norm_query.weight"), hc, n_embd)?,
            norm_conv: stream_gamma(rd, &format!("{p}.ple_norm_conv.weight"), hc, n_embd)?,
            conv: rd
                .f32_tensor(&format!("{p}.ple_conv1d.weight"))?
                .reshape((wide, cfg.kernel))?,
            kernel: cfg.kernel,
            hc,
            eps,
        })
    }

    /// History the dilated conv needs: `(kernel - 1) · ngram` positions.
    fn history_len(&self) -> usize {
        (self.kernel - 1) * self.hash.ngram
    }

    /// Add the PLE contribution to the streams `res` (`[1, t, hc·n]`) for the
    /// last `t` tokens of `history`, continuing the conv from `conv_state`.
    pub(super) fn forward(
        &self,
        res: &Tensor,
        history: &[u32],
        conv_state: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let (b, t, wide) = res.dims3()?;
        let n = wide / self.hc;
        let dev = res.device();
        if history.len() < t {
            candle_core::bail!("PLE needs the token history of every input position");
        }
        // Gather the hashed rows and lay each token's heads side by side.
        let rows = Tensor::new(self.hash.rows(history, t), dev)?;
        let emb = self.table.forward(&rows)?.reshape((b, t, ()))?;

        // Per-stream gate from a key/query dot product with a signed sqrt.
        let key = grouped_rms(
            &self.key.forward(&emb)?.reshape((b, t, self.hc, n))?,
            &self.norm_key,
            self.eps,
        )?;
        let query = grouped_rms(
            &res.reshape((b, t, self.hc, n))?,
            &self.norm_query,
            self.eps,
        )?;
        let s = ((key * query)?.sum(D::Minus1)? / (n as f64).sqrt())?; // [b, t, hc]
        let gate = sigmoid(&(s.sign()? * s.abs()?.maximum(1e-6)?.sqrt()?)?)?;
        let gated = self
            .value
            .forward(&emb)?
            .unsqueeze(2)?
            .broadcast_mul(&gate.unsqueeze(3)?)?; // [b, t, hc, n]

        // Depthwise causal conv over time, dilated by the n-gram size.
        let normalized = grouped_rms(&gated, &self.norm_conv, self.eps)?.reshape((t, wide))?;
        let hist = self.history_len();
        let prev = match conv_state.take() {
            Some(p) => p,
            None => Tensor::zeros((hist, wide), DType::F32, dev)?,
        };
        let padded = Tensor::cat(&[&prev, &normalized], 0)?; // [hist + t, wide]
        let mut conv: Option<Tensor> = None;
        for k in 0..self.kernel {
            // Tap k reads (kernel - 1 - k) · ngram positions back.
            let start = hist - (self.kernel - 1 - k) * self.hash.ngram;
            let term = padded
                .narrow(0, start, t)?
                .broadcast_mul(&self.conv.narrow(1, k, 1)?.t()?)?;
            conv = Some(match conv {
                Some(c) => (c + term)?,
                None => term,
            });
        }
        *conv_state = Some(padded.narrow(0, t, hist)?.contiguous()?);
        let conv = silu(&conv.expect("kernel >= 1"))?.reshape((b, t, wide))?;
        res + (gated.reshape((b, t, wide))? + conv)?
    }
}

// ─── QSA block-sparse attention ──────────────────────────────────────────────

/// `attention.indexer.*` / `attention.compress_ratios` settings.
#[derive(Debug, Clone)]
pub(super) struct QsaConfig {
    n_heads: usize,
    dim: usize,
    top_k: usize,
    /// Per layer; 0 = dense attention.
    ratios: Vec<usize>,
}

impl QsaConfig {
    /// `None` when no layer has a compress ratio.
    pub(super) fn from_meta(m: &Meta<'_>, n_layer: usize) -> Result<Option<Self>> {
        let ratios = m.array_u32("attention.compress_ratios", n_layer);
        if ratios.iter().all(|&r| r == 0) {
            return Ok(None);
        }
        Ok(Some(Self {
            n_heads: m.u32("attention.indexer.head_count")? as usize,
            dim: m.u32("attention.indexer.key_length")? as usize,
            top_k: m.u32("attention.indexer.top_k")? as usize,
            ratios,
        }))
    }
}

/// The indexer of one full-attention layer.
pub(super) struct Qsa {
    q_proj: Weight, // n_embd → heads·dim
    k_proj: Weight, // n_embd → dim
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rope: Arc<Rope>,
    n_heads: usize,
    dim: usize,
    /// Positions per pooled block (`attention.compress_ratios[layer]`).
    ratio: usize,
    /// Cells of whole blocks each query may select (`indexer.top_k`).
    top_k: usize,
}

impl Qsa {
    /// Load layer `layer`'s indexer when it has a compress ratio (and the
    /// file ships indexer tensors; without them llama.cpp runs dense).
    pub(super) fn load<R: Read + Seek>(
        rd: &mut Reader<R>,
        cfg: &QsaConfig,
        layer: usize,
        p: &str,
        rope: Arc<Rope>,
        eps: f64,
    ) -> Result<Option<Self>> {
        let ratio = cfg.ratios[layer];
        if ratio == 0 || !rd.has(&format!("{p}.indexer.k_proj.weight")) {
            return Ok(None);
        }
        Ok(Some(Self {
            q_proj: rd.qmatmul(&format!("{p}.indexer.q_proj.weight"))?,
            k_proj: rd.qmatmul(&format!("{p}.indexer.k_proj.weight"))?,
            q_norm: rd.rms_norm(&format!("{p}.indexer.q_norm.weight"), eps)?,
            k_norm: rd.rms_norm(&format!("{p}.indexer.k_norm.weight"), eps)?,
            rope,
            n_heads: cfg.n_heads,
            dim: cfg.dim,
            ratio,
            top_k: cfg.top_k,
        }))
    }

    /// Append this input's raw indexer keys to `keys` and return the
    /// attention mask `[1, 1, t, offset + t]` restricting each query to its
    /// selected cells — or `None` while the context is short enough that
    /// every query sees all of it (the selection is then dense).
    ///
    /// Selection for a query at position `q` (llama.cpp
    /// `build_qsa_top_k`): the incomplete tail block `[(q+1)/r·r, q]` is
    /// always visible; every earlier (complete) block is scored by
    /// `Σ_heads relu(q_h · k_b)`, where `k_b` is the RMS-normed, RoPE'd mean
    /// of the block's raw keys; the best blocks fill the budget of
    /// `top_k + r − 1` cells.  llama.cpp resolves a partial block at the
    /// budget edge with an unspecified tie order; here its earliest
    /// positions are taken.
    pub(super) fn mask(
        &self,
        keys: &mut Option<Tensor>,
        xs: &Tensor,
        offset: usize,
    ) -> Result<Option<Tensor>> {
        let (_, t, _) = xs.dims3()?;
        let (r, d, dev) = (self.ratio, self.dim, xs.device());
        let new = self.k_proj.forward(xs)?.reshape((t, d))?;
        let all = match keys.take() {
            Some(k) => Tensor::cat(&[&k, &new], 0)?,
            None => new,
        };
        *keys = Some(all.clone());

        let len = offset + t;
        let width = self.top_k + r - 1;
        if len <= width {
            return Ok(None);
        }

        // Pooled block keys for every complete block.
        let n_blocks = len / r;
        let pooled = all
            .narrow(0, 0, n_blocks * r)?
            .reshape((n_blocks, r, d))?
            .mean(1)?;
        let pooled = self.k_norm.forward(&pooled)?;
        let starts: Vec<u32> = (0..n_blocks).map(|b| (b * r) as u32).collect();
        let pooled = self.rope.apply_at(&pooled, &Tensor::new(starts, dev)?)?; // [blocks, d]

        // Indexer queries at their own positions.
        let q = self.q_proj.forward(xs)?.reshape((1, t, self.n_heads, d))?;
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?.contiguous()?; // [1, heads, t, d]
        let q = self.rope.apply_leading(&q, offset)?.squeeze(0)?; // [heads, t, d]
        let scores = q
            .broadcast_matmul(&pooled.t()?.contiguous()?)? // [heads, t, blocks]
            .relu()?
            .sum(0)?
            .to_vec2::<f32>()?;

        let mut mask = vec![f32::NEG_INFINITY; t * len];
        let mut order: Vec<usize> = Vec::with_capacity(n_blocks);
        for (i, score) in scores.iter().enumerate() {
            let q_pos = offset + i;
            let row = &mut mask[i * len..(i + 1) * len];
            if q_pos < width {
                row[..=q_pos].fill(0.0);
                continue;
            }
            let tail = (q_pos + 1) / r * r;
            row[tail..=q_pos].fill(0.0);
            let mut budget = width - (q_pos + 1 - tail);
            order.clear();
            order.extend(0..tail / r);
            // Only the best `ceil(budget / r)` blocks can matter: select
            // them, then order just those (the comparator is total, so the
            // result equals a full sort's prefix).
            let cmp = |a: &usize, b: &usize| score[*b].total_cmp(&score[*a]).then(a.cmp(b));
            let need = budget.div_ceil(r).min(order.len());
            if need > 0 && need < order.len() {
                order.select_nth_unstable_by(need - 1, cmp);
                order.truncate(need);
            }
            order.sort_by(cmp);
            for &b in &order {
                if budget == 0 {
                    break;
                }
                let take = budget.min(r);
                row[b * r..b * r + take].fill(0.0);
                budget -= take;
            }
        }
        Ok(Some(Tensor::from_vec(mask, (1, 1, t, len), dev)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hash reads EOS before the sequence start and at/before an EOS in
    /// the window, never at the token itself, and wraps in u64.
    #[test]
    fn ple_hash_resets_on_eos_and_wraps() {
        let h = PleHash {
            ngram: 3,
            heads_per_ngram: 1,
            eos: 9,
            multipliers: vec![3, u64::MAX, 5],
            head_offsets: vec![0, 100],
            head_vocab: vec![7, 11],
        };
        let hash = |ctx: [u64; 3]| {
            let bi = (ctx[0] * 3) ^ ctx[1].wrapping_mul(u64::MAX);
            let tri = bi ^ (ctx[2] * 5);
            vec![(bi % 7) as u32, (tri % 11 + 100) as u32]
        };
        // [4, 9, 6]: 4 has no predecessors; 9 (EOS) keeps its own id; 6 sees
        // EOS right before it, so everything earlier reads as EOS too.
        let rows = h.rows(&[4, 9, 6], 3);
        let mut want = hash([4, 9, 9]);
        want.extend(hash([9, 4, 9]));
        want.extend(hash([6, 9, 9]));
        assert_eq!(rows, want);
        // Only the last `t` tokens get rows, still hashed against history.
        assert_eq!(h.rows(&[4, 9, 6], 1), hash([6, 9, 9]));
    }
}
