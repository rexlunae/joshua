//! Manifold-constrained hyper-connections (mHC), shared by DeepSeek-V4
//! (`quantized_deepseek4`) and GLM-5.3-Flash (`glm5next`, in
//! `quantized_deepseek2`).
//!
//! The residual stream is carried as `hc` parallel copies.  Each sublayer
//! derives three sets of weights from the RMS-normalised, flattened copies
//! through one small projection: `pre` collapses the copies into the
//! sublayer's input, `post` spreads its output back over them, and `comb` —
//! made doubly stochastic by a few Sinkhorn–Knopp rounds — mixes the input
//! copies into the output ones.

use candle_core::quantized::QMatMul;
use candle_core::{Module, Result, Tensor, D};
use candle_nn::ops::{sigmoid, softmax_last_dim};

/// One sublayer's mixing weights.
pub struct HcMix {
    pub pre: Tensor,  // [b, s, hc]
    pub post: Tensor, // [b, s, hc]
    pub comb: Tensor, // [b, s, hc, hc]
}

/// Split the mixes row `[hc, 2*hc, hc*hc]` into pre/post/comb and run the
/// Sinkhorn normalization on the comb block, matching the reference
/// `hc_split_sinkhorn` kernel:
///   softmax over k (last dim), column-normalize over j (dim 1),
///   then `iters - 1` rounds of row (dim 2) then column (dim 1) normalization.
pub fn split_sinkhorn(
    mixes: &Tensor,
    scale: &Tensor,
    base: &Tensor,
    hc: usize,
    iters: usize,
    eps: f64,
) -> Result<HcMix> {
    split_sinkhorn_with_scale(mixes, &read_scale(scale)?, base, hc, iters, eps)
}

fn read_scale(scale: &Tensor) -> Result<[f64; 3]> {
    let s: Vec<f32> = scale.flatten_all()?.to_vec1()?;
    if s.len() < 3 {
        candle_core::bail!(
            "hyper-connection scale tensor has {} entries, expected 3",
            s.len()
        );
    }

    Ok([s[0] as f64, s[1] as f64, s[2] as f64])
}

fn split_sinkhorn_with_scale(
    mixes: &Tensor,
    scale: &[f64; 3],
    base: &Tensor,
    hc: usize,
    iters: usize,
    eps: f64,
) -> Result<HcMix> {
    let shape = mixes.dims();
    let n = shape[..shape.len() - 1].iter().product::<usize>();
    let m = mixes.reshape((n, (2 + hc) * hc))?;
    let pre = sigmoid(
        &m.narrow(D::Minus1, 0, hc)?
            .affine(scale[0], 0.0)?
            .broadcast_add(&base.narrow(0, 0, hc)?)?,
    )?
    .affine(1.0, eps)?;
    let post = (sigmoid(
        &m.narrow(D::Minus1, hc, hc)?
            .affine(scale[1], 0.0)?
            .broadcast_add(&base.narrow(0, hc, hc)?)?,
    )? * 2.0)?;
    let comb0 = m
        .narrow(D::Minus1, 2 * hc, hc * hc)?
        .affine(scale[2], 0.0)?
        .broadcast_add(&base.narrow(0, 2 * hc, hc * hc)?)?;
    let comb0 = comb0.reshape((n, hc, hc))?;

    let mut comb = softmax_last_dim(&comb0)?.affine(1.0, eps)?;
    comb = comb.broadcast_div(&comb.sum_keepdim(1)?.affine(1.0, eps)?)?;
    for _ in 0..iters.saturating_sub(1) {
        comb = comb.broadcast_div(&comb.sum_keepdim(2)?.affine(1.0, eps)?)?;
        comb = comb.broadcast_div(&comb.sum_keepdim(1)?.affine(1.0, eps)?)?;
    }

    let lead = &shape[..shape.len() - 1];
    let pre = pre.reshape([lead, &[hc]].concat())?;
    let post = post.reshape([lead, &[hc]].concat())?;
    let comb = comb.reshape([lead, &[hc, hc]].concat())?;
    Ok(HcMix { pre, post, comb })
}

/// A sublayer's mixing coefficients, from the stream `x` (`[b, s, hc, d]`):
/// `hc_fn` (`[hc_dim, mix_hc]`) applied to the flattened copies RMS-normalized
/// with the model's norm eps, then split by [`split_sinkhorn`] (llama.cpp
/// `build_hc_mixes`).
pub fn mixes(
    x: &Tensor,
    hc_fn: &QMatMul,
    scale: &Tensor,
    base: &Tensor,
    rms_eps: f64,
    iters: usize,
    eps: f64,
) -> Result<HcMix> {
    let (mixes, hc) = project_mixes(x, hc_fn, rms_eps)?;
    split_sinkhorn(&mixes, scale, base, hc, iters, eps)
}

fn project_mixes(x: &Tensor, hc_fn: &QMatMul, rms_eps: f64) -> Result<(Tensor, usize)> {
    let (b, s, hc, d) = x.dims4()?;
    let flat = x.reshape((b * s, hc * d))?;
    let mixes = hc_fn.forward(&rms_rows(&flat, rms_eps)?)?; // [b*s, (2+hc)*hc]
    let mixes = mixes.reshape((b, s, (2 + hc) * hc))?;
    Ok((mixes, hc))
}

/// Unweighted RMS norm over the last dim.
pub fn rms_rows(x: &Tensor, eps: f64) -> Result<Tensor> {
    x.broadcast_mul(
        &x.sqr()?
            .mean_keepdim(D::Minus1)?
            .affine(1.0, eps)?
            .powf(-0.5)?,
    )
}

/// Collapse the `hc` copies of `x` (`[b, s, hc, d]`) to one stream with the
/// weights `pre` (`[b, s, hc]`).
pub fn collapse(x: &Tensor, pre: &Tensor) -> Result<Tensor> {
    pre.unsqueeze(D::Minus1)?
        .broadcast_as(x.shape())?
        .mul(x)?
        .sum(D::Minus2)
}

/// `hc_post`: expand one stream back to `hc` copies.
/// `x`: `[b, s, d]`, `residual`: `[b, s, hc, d]`, `post`: `[b, s, hc]`,
/// `comb`: `[b, s, hc(src), hc(dst)]` (Sinkhorn-normalized over `dst`
/// first).  Copy `dst` becomes `post[dst]·x + Σ_src comb[src, dst]·residual[src]`
/// (llama.cpp `build_hc_post`).
pub fn post(x: &Tensor, residual: &Tensor, post: &Tensor, comb: &Tensor) -> Result<Tensor> {
    let (b, s, hc, d) = residual.dims4()?;
    let post_t = post.unsqueeze(D::Minus1)?.broadcast_as((b, s, hc, d))?;
    let x_t = x.unsqueeze(2)?.broadcast_as((b, s, hc, d))?;
    // [b, s, dst, src] · [b, s, src, d] → [b, s, dst, d]
    let comb_src = comb
        .transpose(2, 3)?
        .contiguous()?
        .matmul(&residual.contiguous()?)?;
    (post_t * x_t)?.add(&comb_src)
}

/// One sublayer's immutable mixer weights (`{p}_fn`, `{p}_base`, `{p}_scale`).
/// The three scalar scales are read once at construction, avoiding a device
/// readback (and synchronization) at every sublayer on every forward pass.
pub struct HyperConnection {
    hc_fn: QMatMul,
    base: Tensor,
    scale: [f64; 3],
}

impl HyperConnection {
    pub fn new(hc_fn: QMatMul, base: Tensor, scale: &Tensor) -> Result<Self> {
        Ok(Self {
            hc_fn,
            base,
            scale: read_scale(scale)?,
        })
    }

    /// Compute this sublayer's mixes without reading constant weights back
    /// from the device. V4.1 carries `pre` to the next sublayer.
    pub fn mixes(&self, x: &Tensor, rms_eps: f64, iters: usize, eps: f64) -> Result<HcMix> {
        let (mixes, hc) = project_mixes(x, &self.hc_fn, rms_eps)?;
        split_sinkhorn_with_scale(&mixes, &self.scale, &self.base, hc, iters, eps)
    }

    /// Enter the sublayer: its mixes, and the copies `x` (`[b, s, hc, d]`)
    /// collapsed with `pre`.
    pub fn enter(
        &self,
        x: &Tensor,
        rms_eps: f64,
        iters: usize,
        eps: f64,
    ) -> Result<(Tensor, HcMix)> {
        let mix = self.mixes(x, rms_eps, iters, eps)?;
        Ok((collapse(x, &mix.pre)?, mix))
    }
}
