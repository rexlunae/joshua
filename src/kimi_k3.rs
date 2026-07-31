//! Kimi K3 (`kimi-k3`) — architecture primitives.
//!
//! K3 is a 2.8 T-parameter hybrid MoE and shares almost nothing with the
//! `deepseek2` family Joshua already implements.  The pieces that differ, and
//! that this module gets right and tests:
//!
//! * **Hybrid layer topology.** 93 layers, most running *Kimi Delta
//!   Attention* (a gated delta-net linear attention) and every fourth running
//!   MLA.  The GGUF carries no "pattern" key — the layer type is encoded in
//!   the per-layer `attention.head_count_kv` **array**, where `0` marks a
//!   recurrent KDA layer.
//! * **Kimi Delta Attention.** A delta-rule recurrence whose decay gate is
//!   *per channel* rather than one scalar per head, with a sigmoid gate
//!   bounded below by `kda.gate_lower_bound`.
//! * **Attention Residuals.** Periodic checkpoints of the residual stream that
//!   later layers mix by softmax attention, replacing the additive residual at
//!   each checkpoint boundary.
//! * **`situ` activation** in place of SwiGLU, and a **latent** MoE whose
//!   routed experts operate in a narrower width than the router sees.
//!
//! # Status
//!
//! The primitives below are implemented from the published specification and
//! unit-tested against the reference formulations.  A complete forward pass is
//! **not** wired up: K3 needs on the order of a terabyte of weights, no
//! released llama.cpp can run it (support is an open pull request), and no
//! weights were available here — so there is no oracle to check end-to-end
//! logits against, the way [`crate::quantized_deepseek2`] was checked.  What
//! is here is the correctness-critical maths, isolated so it can be verified
//! now and trusted later.
//!
//! Two details are called out because getting them wrong fails *silently*:
//! the layer-type lists in the reference config are **1-indexed**, and the
//! attention-residual mixer computes its softmax over RMS-normalised
//! candidates while averaging the **raw** ones.

use candle_core::{Result, Tensor, D};

/// GGUF `general.architecture` value.
pub const ARCH: &str = "kimi-k3";

/// Default period, in layers, between attention-residual checkpoints.
pub const DEFAULT_ATTN_RES_BLOCK: usize = 12;

/// `situ` β for the gate branch.
pub const SITU_BETA: f32 = 4.0;
/// `situ` β for the linear (up) branch.
pub const SITU_LINEAR_BETA: f32 = 25.0;

/// Which attention a layer runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Kimi Delta Attention — a recurrent, linear-attention layer.
    Kda,
    /// Multi-head Latent Attention — full quadratic attention.
    Mla,
}

/// The attention used by layer `il` of a 93-layer K3.
///
/// The reference config lists the layer types **1-indexed**; expressed over
/// 0-indexed layers the rule is "every fourth layer is MLA, and so is the
/// final one", which makes the last two layers (91, 92) both MLA.
///
/// Prefer [`layer_kinds_from_head_count_kv`] when a GGUF is in hand: the file
/// states the topology explicitly and need not match this closed form.
pub fn layer_kind(il: usize, n_layer: usize) -> LayerKind {
    if il + 1 == n_layer || il % 4 == 3 {
        LayerKind::Mla
    } else {
        LayerKind::Kda
    }
}

/// Decode the per-layer topology from `kimi-k3.attention.head_count_kv`.
///
/// The array holds one entry per layer: `0` for a recurrent KDA layer and a
/// non-zero KV-head count for an MLA layer. This is the authoritative source —
/// there is no separate key describing the interleave.
pub fn layer_kinds_from_head_count_kv(head_count_kv: &[u32]) -> Vec<LayerKind> {
    head_count_kv
        .iter()
        .map(|&n| {
            if n == 0 {
                LayerKind::Kda
            } else {
                LayerKind::Mla
            }
        })
        .collect()
}

/// The `situ` activation, K3's replacement for SwiGLU.
///
/// `situ(g, u) = β·tanh(g/β)·σ(g) · (β_lin·tanh(u/β_lin))`
///
/// Both branches are soft-clipped by a `tanh`, which is what keeps activations
/// bounded at this depth. A non-positive `linear_beta` leaves the up branch
/// untransformed.
pub fn situ(gate: &Tensor, up: &Tensor, beta: f32, linear_beta: f32) -> Result<Tensor> {
    let b = beta as f64;
    let soft_gate = ((gate / b)?.tanh()? * b)?;
    let gated = (soft_gate * candle_nn::ops::sigmoid(gate)?)?;
    if linear_beta <= 0.0 {
        return gated * up;
    }
    let lb = linear_beta as f64;
    let soft_up = ((up / lb)?.tanh()? * lb)?;
    gated * soft_up
}

/// The KDA decay gate, in log space.
///
/// `g = lower_bound · σ(exp(A_log)ₕ · (a + dt_bias))`
///
/// `a` is the low-rank projection output shaped `[.., n_head, head_dim]` and
/// `a_log_exp` holds `exp(A_log)` — **one scalar per head**, broadcast across
/// the head's channels. The result is negative and bounded below by
/// `lower_bound` (−5 in the released config), so the per-step retention
/// `exp(g)` stays inside `(e^lower_bound, 1)`.
///
/// The gate being *per channel* rather than per head is the substantive
/// difference between KDA and the scalar-gated delta nets it derives from:
/// each key channel of the recurrent state decays at its own rate.
pub fn kda_gate(a: &Tensor, a_log_exp: &Tensor, lower_bound: f32) -> Result<Tensor> {
    // a: [.., n_head, head_dim]; a_log_exp: [n_head] → broadcast over channels.
    let dims = a.dims();
    let n_head = dims[dims.len() - 2];
    let scaled = a.broadcast_mul(&a_log_exp.reshape((n_head, 1))?)?;
    candle_nn::ops::sigmoid(&scaled)? * lower_bound as f64
}

/// One head's gated delta-rule recurrence, in the reference (unchunked) form.
///
/// Updates the `[key_dim, value_dim]` state across `seq_len` steps and returns
/// the per-step outputs:
///
/// ```text
/// S  ← S · exp(g_t)        (row-wise: each key channel decays on its own)
/// δ  ← v_t − Sᵀ k_t        (what the state currently gets wrong)
/// S  ← S + (β_t · k_t) δᵀ  (correct it, scaled by the write strength)
/// o_t ← Sᵀ q_t
/// ```
///
/// `q`/`k` are expected already L2-normalised with `q` scaled by
/// `1/√key_dim`, matching the reference kernels, which fold those steps in.
///
/// Shapes: `q`,`k` `[seq, key_dim]`; `v` `[seq, value_dim]`; `g` `[seq,
/// key_dim]`; `beta` `[seq]`; `state` `[key_dim, value_dim]`. Returns `[seq,
/// value_dim]`.
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
pub fn kda_recurrent_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    seq_len: usize,
    key_dim: usize,
    value_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; seq_len * value_dim];
    let mut delta = vec![0f32; value_dim];
    for t in 0..seq_len {
        let (qt, kt) = (&q[t * key_dim..], &k[t * key_dim..]);
        let vt = &v[t * value_dim..];
        let gt = &g[t * key_dim..];

        // Per-channel decay: row i of the state scales by exp(g[i]).
        for i in 0..key_dim {
            let decay = gt[i].exp();
            let row = &mut state[i * value_dim..(i + 1) * value_dim];
            for s in row.iter_mut() {
                *s *= decay;
            }
        }

        // delta = v - Sᵀk  — the prediction error the write will correct.
        for j in 0..value_dim {
            let mut acc = 0f32;
            for i in 0..key_dim {
                acc += kt[i] * state[i * value_dim + j];
            }
            delta[j] = vt[j] - acc;
        }

        // S += (β·k) ⊗ delta
        let b = beta[t];
        for i in 0..key_dim {
            let bk = b * kt[i];
            if bk == 0.0 {
                continue;
            }
            let row = &mut state[i * value_dim..(i + 1) * value_dim];
            for (j, s) in row.iter_mut().enumerate() {
                *s += bk * delta[j];
            }
        }

        // o = Sᵀq
        for j in 0..value_dim {
            let mut acc = 0f32;
            for i in 0..key_dim {
                acc += qt[i] * state[i * value_dim + j];
            }
            out[t * value_dim + j] = acc;
        }
    }
    out
}

/// Mix the running residual stream with the banked attention-residual
/// checkpoints.
///
/// `candidates` is `[n_candidates, hidden]` — the bank followed by the current
/// stream — and `score` is the per-layer fused score vector `[hidden]`.
///
/// The subtlety, and the easy way to get this wrong: scores are computed
/// against **RMS-normalised** candidates, but the convex combination is taken
/// over the **raw** ones.
pub fn attn_res_mix(candidates: &Tensor, score: &Tensor, eps: f64) -> Result<Tensor> {
    let raw = candidates.to_dtype(candle_core::DType::F32)?;
    // RMS-normalise each candidate row for scoring only.
    let ms = raw.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = raw.broadcast_div(&(ms + eps)?.sqrt()?)?;
    let scores = normed
        .broadcast_mul(&score.to_dtype(candle_core::DType::F32)?)?
        .sum(D::Minus1)?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    // Weighted sum over the *raw* candidates.
    raw.broadcast_mul(&probs.unsqueeze(D::Minus1)?)?.sum(0)
}

/// Layers at which the residual stream is checkpointed.
///
/// At each of these the ordinary additive residual is severed: the layer's
/// input is banked and the attention output becomes the new stream. With the
/// released period of 12 over 93 layers this banks 8 checkpoints.
pub fn checkpoint_layers(n_layer: usize, block: usize) -> Vec<usize> {
    if block == 0 {
        return Vec::new();
    }
    (0..n_layer).filter(|il| il % block == 0).collect()
}

/// Select experts and compute their weights.
///
/// Returns `(indices, weights)`, each `[n_tokens, n_used]`.
///
/// Order matters and differs from the DeepSeek families: the router is
/// **sigmoid**, the `exp_probs_b` bias steers *selection only* while the
/// weights come from the **unbiased** scores, and the selected weights are
/// renormalised before the (unit) routed scale. K3 ships a single expert group,
/// so no group-limited masking applies.
pub fn route_experts(
    logits: &Tensor,
    bias: Option<&Tensor>,
    n_used: usize,
    renormalize: bool,
    scale: f64,
) -> Result<(Tensor, Tensor)> {
    let scores = candle_nn::ops::sigmoid(&logits.to_dtype(candle_core::DType::F32)?)?;
    let selection = match bias {
        Some(b) => scores.broadcast_add(&b.reshape((1, ()))?)?,
        None => scores.clone(),
    };
    let idx = selection
        .arg_sort_last_dim(false)?
        .narrow(D::Minus1, 0, n_used)?
        .contiguous()?;
    // Weights always come from the unbiased scores.
    let mut w = scores.gather(&idx, D::Minus1)?;
    if renormalize {
        let denom = (w.sum_keepdim(D::Minus1)? + 1e-20)?;
        w = w.broadcast_div(&denom)?;
    }
    if scale != 1.0 {
        w = (w * scale)?;
    }
    Ok((idx, w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn layer_topology_matches_the_reference_pattern() {
        let n = 93;
        // Reference lists 1-indexed full-attention layers 4,8,...,88,92,93.
        let mla: Vec<usize> = (0..n)
            .filter(|&il| layer_kind(il, n) == LayerKind::Mla)
            .collect();
        assert_eq!(mla.len(), 24, "24 full-attention layers");
        assert_eq!(&mla[..3], &[3, 7, 11]);
        // The last two layers are both MLA — the period is broken at the end.
        assert_eq!(&mla[mla.len() - 2..], &[91, 92]);
        assert_eq!(
            (0..n).filter(|&il| layer_kind(il, n) == LayerKind::Kda).count(),
            69,
            "69 KDA layers"
        );
        // Off-by-one canary: layer 0 must be KDA, not MLA.
        assert_eq!(layer_kind(0, n), LayerKind::Kda);
    }

    #[test]
    fn head_count_kv_array_drives_the_topology() {
        // A zero entry marks a recurrent KDA layer.
        let kinds = layer_kinds_from_head_count_kv(&[0, 0, 0, 1, 0, 0, 0, 1]);
        assert_eq!(kinds[0], LayerKind::Kda);
        assert_eq!(kinds[3], LayerKind::Mla);
        assert_eq!(kinds[7], LayerKind::Mla);
        // And it agrees with the closed form on the same prefix.
        for (il, kind) in kinds.iter().enumerate() {
            assert_eq!(*kind, layer_kind(il, 93), "layer {il}");
        }
    }

    #[test]
    fn situ_matches_its_definition() {
        let dev = Device::Cpu;
        let gate = Tensor::new(&[0.0f32, 1.0, -2.0, 8.0], &dev).unwrap();
        let up = Tensor::new(&[1.0f32, -1.0, 2.0, 50.0], &dev).unwrap();
        let got: Vec<f32> = situ(&gate, &up, SITU_BETA, SITU_LINEAR_BETA)
            .unwrap()
            .to_vec1()
            .unwrap();

        let (b, lb) = (SITU_BETA, SITU_LINEAR_BETA);
        for (i, (g, u)) in [(0.0f32, 1.0f32), (1.0, -1.0), (-2.0, 2.0), (8.0, 50.0)]
            .iter()
            .enumerate()
        {
            let expect = b * (g / b).tanh() * (1.0 / (1.0 + (-g).exp())) * (lb * (u / lb).tanh());
            assert!(
                (got[i] - expect).abs() < 1e-5,
                "situ[{i}] = {}, expected {expect}",
                got[i]
            );
        }
        // Soft clipping: a large gate saturates towards beta rather than growing.
        assert!(got[3].abs() < (SITU_BETA * SITU_LINEAR_BETA) + 1.0);
    }

    #[test]
    fn kda_gate_is_negative_and_respects_its_lower_bound() {
        let dev = Device::Cpu;
        // 2 heads, 3 channels each; extreme inputs to probe both saturations.
        let a = Tensor::new(&[[-50.0f32, 0.0, 50.0], [-50.0, 0.0, 50.0]], &dev).unwrap();
        let a_log_exp = Tensor::new(&[1.0f32, 1.0], &dev).unwrap();
        let g = kda_gate(&a, &a_log_exp, -5.0).unwrap();
        let v: Vec<f32> = g.flatten_all().unwrap().to_vec1().unwrap();

        for x in &v {
            assert!(*x <= 0.0 && *x >= -5.0, "gate {x} outside (lower_bound, 0]");
        }
        // Large positive input saturates the sigmoid → close to the bound.
        assert!((v[2] - -5.0).abs() < 1e-3);
        // Large negative input → decay ≈ 0, i.e. retention ≈ 1.
        assert!(v[0].abs() < 1e-3);
        // Midpoint is exactly half the bound.
        assert!((v[1] - -2.5).abs() < 1e-5);
    }

    #[test]
    fn kda_gate_uses_a_per_head_scalar_across_channels() {
        let dev = Device::Cpu;
        let a = Tensor::new(&[[1.0f32, 1.0], [1.0, 1.0]], &dev).unwrap();
        // Different exp(A_log) per head must give different gates per head,
        // but identical values across channels within a head.
        let a_log_exp = Tensor::new(&[0.1f32, 10.0], &dev).unwrap();
        let v: Vec<f32> = kda_gate(&a, &a_log_exp, -5.0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(v[0], v[1], "channels within a head share the scalar");
        assert_eq!(v[2], v[3]);
        assert!(v[3] < v[1], "a larger exp(A_log) decays harder");
    }

    #[test]
    fn delta_rule_writes_a_value_then_recalls_it() {
        // With no decay and full write strength, a single (k, v) write should
        // be recalled exactly by querying with the same unit key.
        let (kd, vd) = (4usize, 3usize);
        let k = vec![1.0f32, 0.0, 0.0, 0.0];
        let q = k.clone();
        let v = vec![0.5f32, -1.0, 2.0];
        let g = vec![0.0f32; kd]; // exp(0) = 1 → no decay
        let beta = vec![1.0f32];
        let mut state = vec![0f32; kd * vd];
        let out = kda_recurrent_head(&q, &k, &v, &g, &beta, &mut state, 1, kd, vd);
        for j in 0..vd {
            assert!((out[j] - v[j]).abs() < 1e-6, "recall {j}: {} vs {}", out[j], v[j]);
        }
    }

    #[test]
    fn delta_rule_decay_forgets_previous_writes() {
        let (kd, vd) = (2usize, 1usize);
        // Step 1 writes v=1 on key e0; step 2 writes nothing (beta=0) but
        // decays hard, so the recall must shrink.
        let k = vec![1.0f32, 0.0, 1.0, 0.0];
        let q = k.clone();
        let v = vec![1.0f32, 0.0];
        // Second step decays by exp(-2).
        let g = vec![0.0f32, 0.0, -2.0, -2.0];
        let beta = vec![1.0f32, 0.0];
        let mut state = vec![0f32; kd * vd];
        let out = kda_recurrent_head(&q, &k, &v, &g, &beta, &mut state, 2, kd, vd);
        assert!((out[0] - 1.0).abs() < 1e-6);
        let expect = (-2.0f32).exp();
        assert!(
            (out[1] - expect).abs() < 1e-6,
            "decayed recall {} vs {expect}",
            out[1]
        );
    }

    #[test]
    fn attn_res_mix_scores_normalised_but_averages_raw() {
        let dev = Device::Cpu;
        // Two candidates with the same direction but very different scale.
        // Scoring is scale-invariant (RMS-normalised), so both score equally
        // and the output is their raw mean — not dominated by the larger one.
        let candidates = Tensor::new(&[[1.0f32, 0.0], [100.0, 0.0]], &dev).unwrap();
        let score = Tensor::new(&[1.0f32, 0.0], &dev).unwrap();
        let out: Vec<f32> = attn_res_mix(&candidates, &score, 1e-6)
            .unwrap()
            .to_vec1()
            .unwrap();
        // Equal weights → mean of the raw values.
        assert!(
            (out[0] - 50.5).abs() < 1e-3,
            "expected the raw mean 50.5, got {}",
            out[0]
        );
    }

    #[test]
    fn routing_biases_selection_but_not_weights() {
        let dev = Device::Cpu;
        // Expert 0 has the higher score; the bias is large enough to make
        // expert 1 win selection. The returned weight must still be expert 1's
        // *unbiased* sigmoid score.
        let logits = Tensor::new(&[[1.0f32, 0.5, -5.0, -6.0]], &dev).unwrap();
        let bias = Tensor::new(&[0.0f32, 10.0, 0.0, 0.0], &dev).unwrap();
        let (idx, w) = route_experts(&logits, Some(&bias), 1, false, 1.0).unwrap();
        let idx: Vec<u32> = idx.flatten_all().unwrap().to_vec1().unwrap();
        let w: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(idx[0], 1, "bias must steer selection");
        let unbiased = 1.0f32 / (1.0 + (-0.5f32).exp());
        assert!(
            (w[0] - unbiased).abs() < 1e-6,
            "weight {} must be the unbiased score {unbiased}",
            w[0]
        );
    }

    #[test]
    fn routing_renormalises_selected_weights() {
        let dev = Device::Cpu;
        let logits = Tensor::new(&[[2.0f32, 1.0, 0.0, -1.0]], &dev).unwrap();
        let (_, w) = route_experts(&logits, None, 2, true, 1.0).unwrap();
        let w: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "renormalised weights sum to {sum}");
        assert!(w[0] > w[1], "the stronger expert keeps the larger share");
    }

    #[test]
    fn checkpoints_land_every_block_layers() {
        let cps = checkpoint_layers(93, DEFAULT_ATTN_RES_BLOCK);
        assert_eq!(cps, vec![0, 12, 24, 36, 48, 60, 72, 84]);
        assert_eq!(cps.len(), 8);
        // A zero period disables the mechanism entirely.
        assert!(checkpoint_layers(93, 0).is_empty());
    }
}
