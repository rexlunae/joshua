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
//! `d_conv - 1` conv inputs, so the layer cannot rewind to a prefix.

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
    pub(super) fn forward(&self, state: &mut Option<KdaState>, x: &Tensor) -> Result<Tensor> {
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

        // The delta rule, one head at a time on the host.
        let heads = |x: &Tensor| -> Result<Vec<f32>> {
            x.transpose(0, 1)?.contiguous()?.flatten_all()?.to_vec1()
        };
        let (q, key, v, g) = (heads(&q)?, heads(&key)?, heads(v)?, heads(&g)?);
        let beta = beta.t()?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let mut o = Vec::with_capacity(h * t * d);
        for hh in 0..h {
            let r = hh * t * d..(hh + 1) * t * d;
            o.extend(crate::kimi_k3::kda_recurrent_head(
                &q[r.clone()],
                &key[r.clone()],
                &v[r.clone()],
                &g[r],
                &beta[hh * t..(hh + 1) * t],
                &mut st.s[hh * d * d..(hh + 1) * d * d],
                t,
                d,
                d,
            ));
        }
        let o = Tensor::from_vec(o, (h, t, d), x.device())?
            .transpose(0, 1)?
            .contiguous()?; // [t, h, d]

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
