//! Kimi Delta Attention (KDA), the linear-attention layers of Kimi-Linear
//! (`kimi-linear`), Kimi K3 (`kimi-k3`) and GLM-5.3-Flash (`glm5next`;
//! llama.cpp `kimi-linear.cpp` / `kimi-k3.cpp` / `build_kda_layer`).
//!
//! Per layer: Q, K and V (separate or fused projections) each go through
//! their own causal depthwise conv (then SiLU); Q and K are L2-normalised
//! per head; a per-*channel* decay gate and a per-head write strength drive
//! the gated delta rule ([`crate::kimi_k3::kda_recurrent_head`]); the
//! output is RMS-normalised per head, gated by `sigmoid(g(x))` — a low-rank
//! `g_b(g_a(x))`, or K3's full-rank `ssm_g` — and projected back.  The
//! recurrent state is `[head_dim, head_dim]` per head plus the last
//! `d_conv - 1` conv inputs, so the layer cannot rewind to an *arbitrary*
//! prefix — but a speculative verification pass it has just run can be rolled
//! back, because it snapshots that state and replays the accepted prefix into
//! it (see [`KdaSnapshot`]).

use std::io::{Read, Seek};

use candle_core::quantized::QMatMul;
use candle_core::{Module, Result, Tensor, D};
use candle_nn::ops::{sigmoid, silu};
use candle_transformers::quantized_nn::RmsNorm;

use crate::gguf_meta::Meta;

/// The `kda.*` / `ssm.*` settings.
#[derive(Debug, Clone)]
pub(super) struct KdaConfig {
    n_head: usize,
    head_dim: usize,
    d_conv: usize,
    /// `kda.gate_lower_bound`: the decay gate is `lb · σ(exp(A_log) · a)`;
    /// without it, `−exp(A_log) · softplus(a)`.
    lower_bound: Option<f32>,
    /// The Q/K L2 norm's eps: `x / √(Σx² + eps)`.
    l2_eps: f64,
}

impl KdaConfig {
    pub(super) fn from_meta(m: &Meta<'_>, n_head: usize, l2_eps: f64) -> Result<Self> {
        let cfg = Self {
            n_head,
            l2_eps,
            head_dim: m.u32("kda.head_dim")? as usize,
            d_conv: m.u32("ssm.conv_kernel")? as usize,
            lower_bound: m
                .contains("kda.gate_lower_bound")
                .then(|| m.f32_or("kda.gate_lower_bound", 0.0)),
        };
        if cfg.head_dim == 0 || cfg.d_conv == 0 {
            candle_core::bail!(
                "{}: KDA needs a positive head dim and conv kernel",
                m.arch()
            );
        }
        Ok(cfg)
    }

    fn inner(&self) -> usize {
        self.n_head * self.head_dim
    }
}

/// One KDA layer's recurrent state.
#[derive(Clone)]
pub(super) struct KdaState {
    /// The last `d_conv - 1` inputs of the Q, K and V convs, `[d_conv - 1, inner]`.
    conv: [Tensor; 3],
    /// Per head `[head_dim (key), head_dim (value)]`, head-major.
    s: Vec<f32>,
}

/// One verification pass's worth of KDA recurrence inputs: the per-token `q`,
/// `k`, `v`, decay `g` and write gate `beta` the gated delta rule consumed,
/// exactly as [`crate::kimi_k3::kda_recurrent_head_into`] read them, plus the
/// pass's *pre-conv* Q/K/V rows so the three conv tails can be rewound too.
///
/// Every field is token-major (`[t, head, head_dim]`, `[t, head]`), so the
/// accepted prefix of a pass is a contiguous slice.
#[derive(Clone, Default)]
struct KdaInputs {
    q: Vec<f32>,
    key: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    /// The pass's pre-conv Q, K and V rows, `[t, inner]` token-major — the rows
    /// each causal conv's window was continued by.
    conv: [Vec<f32>; 3],
    /// Tokens the pass fed.
    t: usize,
}

/// A KDA layer's snapshot of its recurrent state, taken before a
/// multi-token verification pass so a rejected draft can be rolled back.
///
/// The recurrence is *running* state — there is no prefix to rewind to the way
/// an append-only KV cache has one — so a pass that feeds `k + 1` tokens would
/// otherwise leave the delta-rule matrices and the three conv tails advanced
/// past the tokens the target model rejected.
///
/// Restoring the pre-pass state alone is only right when the rollback rejects
/// *every* draft: a partial accept has to keep the tokens that survived, so the
/// snapshot also carries the pass's own recurrence inputs and [`Self::replay`]
/// re-runs the accepted prefix into the restored state, exactly as
/// `quantized_qwen`'s `RecurrentSnapshot` does for its Gated DeltaNet layers.
#[derive(Clone, Default)]
pub struct KdaSnapshot {
    /// The recurrent state as it was *before* the pass.
    pre: Option<KdaState>,
    /// The pass's recurrence inputs, token-major over its tokens.
    inputs: Option<KdaInputs>,
    /// The layer's KDA geometry.  Carried so the restore path — a static method
    /// with no handle on the weights — can replay without one.
    cfg: Option<KdaConfig>,
}

/// Copy head `hh`'s rows of a token-major `[t, h, d]` buffer into `buf`, in
/// time order: the per-head slice the recurrence consumes.
fn gather_head(src: &[f32], hh: usize, t: usize, h: usize, d: usize, buf: &mut Vec<f32>) {
    buf.clear();
    for tok in 0..t {
        let base = (tok * h + hh) * d;
        buf.extend_from_slice(&src[base..base + d]);
    }
}

impl KdaSnapshot {
    /// Roll a KDA layer's recurrent state back to `retained` tokens into the
    /// pass that produced this snapshot.
    ///
    /// A no-op for an unarmed capture (an attention layer of the same model
    /// never fills one).
    pub(super) fn replay(&self, state: &mut Option<KdaState>, retained: usize) -> Result<()> {
        let Some(pre) = &self.pre else {
            return Ok(());
        };
        *state = Some(pre.clone());
        let (Some(inputs), Some(cfg)) = (&self.inputs, &self.cfg) else {
            return Ok(());
        };
        let kept = retained.min(inputs.t);
        if kept == 0 {
            return Ok(());
        }
        let (h, d) = (cfg.n_head, cfg.head_dim);
        let inner = cfg.inner();
        let slot = state.as_mut().expect("just restored");
        // Each conv tail is the last `d_conv - 1` rows of the pre-pass window
        // continued by the pass's rows, so rebuilding it for the kept prefix is
        // the same window a pass of that length would have built.  Reusing a
        // rejected row here would leak the draft into the next token's conv.
        let k1 = cfg.d_conv.saturating_sub(1);
        for (c, rows) in inputs.conv.iter().enumerate() {
            let new = Tensor::from_vec(
                rows[..kept * inner].to_vec(),
                (kept, inner),
                pre.conv[c].device(),
            )?;
            let hist = Tensor::cat(&[&pre.conv[c], &new], 0)?; // [k - 1 + kept, inner]
            slot.conv[c] = hist.narrow(0, kept, k1)?.contiguous()?;
        }
        // Replay the accepted prefix through the same per-head recurrence, in
        // the same order, so the restored state is bit-identical to plain
        // decoding (the outputs are discarded — only the matrices matter).
        let (mut hq, mut hk, mut hv, mut hg, mut hb) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut out = vec![0f32; kept * d];
        let mut delta = vec![0f32; d];
        type HeadSlices<'a> = (&'a [f32], &'a [f32], &'a [f32], &'a [f32], &'a [f32]);
        for hh in 0..h {
            let (qs, ks, vs, gs, bs): HeadSlices = if kept == 1 {
                // One token: head `hh` is the slice the forward borrowed.
                (
                    &inputs.q[hh * d..(hh + 1) * d],
                    &inputs.key[hh * d..(hh + 1) * d],
                    &inputs.v[hh * d..(hh + 1) * d],
                    &inputs.g[hh * d..(hh + 1) * d],
                    std::slice::from_ref(&inputs.beta[hh]),
                )
            } else {
                gather_head(&inputs.q, hh, kept, h, d, &mut hq);
                gather_head(&inputs.key, hh, kept, h, d, &mut hk);
                gather_head(&inputs.v, hh, kept, h, d, &mut hv);
                gather_head(&inputs.g, hh, kept, h, d, &mut hg);
                hb.clear();
                hb.extend((0..kept).map(|tok| inputs.beta[tok * h + hh]));
                (&hq, &hk, &hv, &hg, &hb)
            };
            crate::kimi_k3::kda_recurrent_head_into(
                qs,
                ks,
                vs,
                gs,
                bs,
                &mut slot.s[hh * d * d..(hh + 1) * d * d],
                kept,
                d,
                d,
                &mut out,
                &mut delta,
            );
        }
        Ok(())
    }
}

/// The Q/K/V projections: separate, or one fused `attn_qkv`.
enum Qkv {
    Split([QMatMul; 3]),
    Fused(QMatMul),
}

/// The output gate's pre-activation: low-rank `g_b(g_a(x))` (Kimi-Linear,
/// GLM-5.3-Flash) or full-rank `ssm_g` (K3).
enum OutGate {
    LowRank(QMatMul, QMatMul),
    Full(QMatMul),
}

/// One KDA layer's weights.
pub(super) struct Kda {
    qkv: Qkv,
    /// Depthwise conv taps per channel, `[inner, d_conv]` (oldest first).
    conv: [Tensor; 3],
    f_a: QMatMul,
    f_b: QMatMul,
    dt_bias: Tensor,
    /// `exp(A_log)` per head (the GGUF stores `−exp(A_log)`).
    a_exp: Tensor,
    beta: QMatMul,
    gate: OutGate,
    o_norm: RmsNorm,
    o: QMatMul,
    cfg: KdaConfig,
}

impl Kda {
    pub(super) fn load<R: Read + Seek>(
        rd: &mut super::Reader<R>,
        p: &str,
        cfg: &KdaConfig,
        rms_eps: f64,
    ) -> Result<Self> {
        let inner = cfg.inner();
        let mut conv = |c: &str| -> Result<Tensor> {
            rd.f32_tensor(&format!("{p}.ssm_conv1d_{c}.weight"))?
                .reshape((inner, cfg.d_conv))
        };
        let conv = [conv("q")?, conv("k")?, conv("v")?];
        let qkv = match rd.qmatmul_if(&format!("{p}.attn_qkv.weight"))? {
            Some(w) => Qkv::Fused(w),
            None => Qkv::Split([
                rd.qmatmul(&format!("{p}.attn_q.weight"))?,
                rd.qmatmul(&format!("{p}.attn_k.weight"))?,
                rd.qmatmul(&format!("{p}.attn_v.weight"))?,
            ]),
        };
        let gate = match rd.qmatmul_if(&format!("{p}.ssm_g.weight"))? {
            Some(g) => OutGate::Full(g),
            None => OutGate::LowRank(
                rd.qmatmul(&format!("{p}.ssm_g_a.weight"))?,
                rd.qmatmul(&format!("{p}.ssm_g_b.weight"))?,
            ),
        };
        Ok(Self {
            qkv,
            conv,
            f_a: rd.qmatmul(&format!("{p}.ssm_f_a.weight"))?,
            f_b: rd.qmatmul(&format!("{p}.ssm_f_b.weight"))?,
            dt_bias: rd.f32_tensor(&format!("{p}.ssm_dt.bias"))?,
            // Kimi-Linear pads it to [1, n_head, 1, 1].
            a_exp: rd.f32_tensor(&format!("{p}.ssm_a"))?.flatten_all()?.neg()?,
            beta: rd.qmatmul(&format!("{p}.ssm_beta.weight"))?,
            gate,
            o_norm: rd.rms_norm(&format!("{p}.ssm_norm.weight"), rms_eps)?,
            o: rd.qmatmul(&format!("{p}.attn_output.weight"))?,
            cfg: cfg.clone(),
        })
    }

    /// Run the layer over `x` (`[1, t, n_embd]`), advancing `state`.
    ///
    /// When `cap` is armed — a verification pass, see
    /// [`crate::native_session::LayerStack::begin_verify`] — the state is cloned
    /// before it advances and the pass's own recurrence inputs are recorded, so
    /// [`KdaSnapshot::replay`] can roll the pass back.
    pub(super) fn forward(
        &self,
        state: &mut Option<KdaState>,
        cap: &mut Option<KdaSnapshot>,
        x: &Tensor,
    ) -> Result<Tensor> {
        let (_, t, _) = x.dims3()?;
        let (h, d, k) = (self.cfg.n_head, self.cfg.head_dim, self.cfg.d_conv);
        let inner = self.cfg.inner();
        let st = match state {
            Some(s) => s,
            None => state.insert(KdaState {
                conv: std::array::from_fn(|_| {
                    Tensor::zeros((k - 1, inner), candle_core::DType::F32, x.device())
                })
                .map(|z| z.expect("zeros")),
                s: vec![0f32; h * d * d],
            }),
        };
        // An armed verification pass records what it is about to overwrite, so a
        // rejected draft can be rolled back.  Clone *before* the conv advances
        // the tails and the delta rule advances the matrices below: a snapshot
        // taken afterwards would carry the rejected tokens' convolution history
        // and re-seed it on rollback.  Unarmed (prefill, plain decode) the clone
        // is skipped entirely.
        let pre = cap.as_ref().map(|_| st.clone());
        // The pre-conv Q/K/V rows of this pass, token-major `[t, inner]`, kept
        // only while armed: a rollback rebuilds each conv tail from them.
        let mut conv_rows: [Vec<f32>; 3] = Default::default();

        // Causal depthwise conv over [history ‖ new], then SiLU.
        let proj: [Tensor; 3] = match &self.qkv {
            Qkv::Split(w) => [w[0].forward(x)?, w[1].forward(x)?, w[2].forward(x)?],
            Qkv::Fused(w) => {
                let y = w.forward(x)?;
                let part = |c: usize| y.narrow(D::Minus1, c * inner, inner);
                [part(0)?, part(1)?, part(2)?]
            }
        };
        let mut qkv = Vec::with_capacity(3);
        for (c, y) in proj.iter().enumerate() {
            let y = y.reshape((t, inner))?;
            if cap.is_some() {
                // Armed only — an ordinary decode and every prefill skip this.
                // It is still three device→host readbacks per KDA layer on an
                // accelerator, on top of the five the delta rule already reads
                // back below on *every* forward, so the extra is bounded by
                // the pre-existing host-bound path rather than introducing a new
                // one.  Unmeasured on a device; on CPU the tensors are already
                // host-resident.
                conv_rows[c] = y.flatten_all()?.to_vec1::<f32>()?;
            }
            let hist = Tensor::cat(&[&st.conv[c], &y], 0)?; // [k - 1 + t, inner]
            let mut out = hist
                .narrow(0, 0, t)?
                .broadcast_mul(&self.conv[c].narrow(1, 0, 1)?.t()?)?;
            for j in 1..k {
                out = (out
                    + hist
                        .narrow(0, j, t)?
                        .broadcast_mul(&self.conv[c].narrow(1, j, 1)?.t()?)?)?;
            }
            st.conv[c] = hist.narrow(0, t, k - 1)?.contiguous()?;
            qkv.push(silu(&out)?.reshape((t, h, d))?);
        }
        // L2-normalise Q and K per head (x / √(Σx² + eps)); fold the 1/√d
        // query scale in.
        let eps = self.cfg.l2_eps;
        let l2 = |x: &Tensor| x.broadcast_div(&(x.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?);
        let q = (l2(&qkv[0])? / (d as f64).sqrt())?;
        let key = l2(&qkv[1])?;
        let v = &qkv[2];

        // Per-channel decay gate (log space) and per-head write strength.
        let a = self
            .f_b
            .forward(&self.f_a.forward(x)?)?
            .reshape((t, inner))?
            .broadcast_add(&self.dt_bias)?;
        let a = a.reshape((t, h, d))?;
        let g = match self.cfg.lower_bound {
            Some(lb) => crate::kimi_k3::kda_gate(&a, &self.a_exp, lb)?,
            None => crate::moe::softplus(&a)?.broadcast_mul(&self.a_exp.neg()?.reshape((h, 1))?)?,
        };
        let beta = sigmoid(&self.beta.forward(x)?.reshape((t, h))?)?;

        // The delta rule, one head at a time on the host. Each tensor is
        // read once in its natural [t, h, d] layout: for a decode step
        // (t == 1) head `hh`'s rows are already contiguous there and the
        // per-head slices below borrow them, while a prefill chunk gathers
        // them into per-head scratch (the same bytes the old
        // transpose+contiguous pass copied, without the intermediate
        // [h, t, d] tensor). Outputs go straight into the [t, h, d] buffer,
        // so no transpose copy follows the loop.
        type HeadSlices<'a> =
            (&'a [f32], &'a [f32], &'a [f32], &'a [f32], &'a [f32], &'a mut [f32]);
        let flat = |x: &Tensor| -> Result<Vec<f32>> { x.flatten_all()?.to_vec1() };
        let (q, key, v, g) = (flat(&q)?, flat(&key)?, flat(v)?, flat(&g)?);
        let beta = flat(&beta)?; // flat [t, h]: one read, no per-row Vecs
        if let (Some(pre), Some(cap)) = (pre, cap.as_mut()) {
            // Exactly the values the rule below consumes, in the same
            // token-major layout, so a rollback replays them bit for bit.
            *cap = KdaSnapshot {
                pre: Some(pre),
                inputs: Some(KdaInputs {
                    q: q.clone(),
                    key: key.clone(),
                    v: v.clone(),
                    g: g.clone(),
                    beta: beta.clone(),
                    conv: std::mem::take(&mut conv_rows),
                    t,
                }),
                cfg: Some(self.cfg.clone()),
            };
        }
        let mut o = vec![0f32; t * h * d]; // [t, h, d]
        let mut delta = vec![0f32; d];
        let mut hq: Vec<f32> = Vec::new();
        let mut hk: Vec<f32> = Vec::new();
        let mut hv: Vec<f32> = Vec::new();
        let mut hg: Vec<f32> = Vec::new();
        let mut hb: Vec<f32> = Vec::new();
        let mut ho: Vec<f32> = Vec::new();
        for hh in 0..h {
            let (qs, ks, vs, gs, bs, os): HeadSlices =
                if t == 1 {
                    (
                        &q[hh * d..(hh + 1) * d],
                        &key[hh * d..(hh + 1) * d],
                        &v[hh * d..(hh + 1) * d],
                        &g[hh * d..(hh + 1) * d],
                        std::slice::from_ref(&beta[hh]),
                        &mut o[hh * d..(hh + 1) * d],
                    )
                } else {
                    gather_head(&q, hh, t, h, d, &mut hq);
                    gather_head(&key, hh, t, h, d, &mut hk);
                    gather_head(&v, hh, t, h, d, &mut hv);
                    gather_head(&g, hh, t, h, d, &mut hg);
                    hb.clear();
                    hb.extend((0..t).map(|tok| beta[tok * h + hh]));
                    ho.clear();
                    ho.resize(t * d, 0.0);
                    (&hq, &hk, &hv, &hg, &hb, &mut ho)
                };
            crate::kimi_k3::kda_recurrent_head_into(
                qs,
                ks,
                vs,
                gs,
                bs,
                &mut st.s[hh * d * d..(hh + 1) * d * d],
                t,
                d,
                d,
                os,
                &mut delta,
            );
            if t > 1 {
                for tok in 0..t {
                    let base = (tok * h + hh) * d;
                    o[base..base + d].copy_from_slice(&ho[tok * d..(tok + 1) * d]);
                }
            }
        }
        let o = Tensor::from_vec(o, (t, h, d), x.device())?; // [t, h, d]

        // RMSNorm(o) · σ(g(x)), then the output projection.
        let gate = match &self.gate {
            OutGate::LowRank(a, b) => b.forward(&a.forward(x)?)?,
            OutGate::Full(g) => g.forward(x)?,
        }
        .reshape((t, h, d))?;
        let o = (self.o_norm.forward(&o)? * sigmoid(&gate)?)?.reshape((1, t, inner))?;
        self.o.forward(&o)
    }
}
