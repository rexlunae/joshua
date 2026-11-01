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
use crate::precise::{sigmoid, softmax_last_dim};

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
    #[cfg(feature = "vulkan")]
    if hc == 4
        && mixes.dtype() == candle_core::DType::F32
        && crate::precise::active(mixes)
        && matches!(mixes.device(), candle_core::Device::Vulkan(_))
        && matches!(mixes.device(),candle_core::Device::Vulkan(d) if candle_core::vulkan_backend::kernels::fused_mhc_enabled_on(d))
        && mixes.is_contiguous()
        && shape.last() == Some(&24)
        && base.dtype() == candle_core::DType::F32
        && base.is_contiguous()
        && base.dims() == [24]
        && mixes.device().same_device(base.device())
    {
        let packed = mixes.apply_op2_no_bwd(
            base,
            &FusedSinkhorn {
                rows: n,
                scale: scale.map(|v| v as f32),
                iters,
                eps: eps as f32,
            },
        )?;
        let lead = &shape[..shape.len() - 1];
        return Ok(HcMix {
            pre: packed.narrow(0, 0, n * 4)?.reshape([lead, &[4]].concat())?,
            post: packed
                .narrow(0, n * 4, n * 4)?
                .reshape([lead, &[4]].concat())?,
            comb: packed
                .narrow(0, n * 8, n * 16)?
                .reshape([lead, &[4, 4]].concat())?,
        });
    }
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

    let comb = normalize_comb(&softmax_last_dim(&comb0)?.affine(1.0, eps)?, hc, iters, eps)?;

    let lead = &shape[..shape.len() - 1];
    let pre = pre.reshape([lead, &[hc]].concat())?;
    let post = post.reshape([lead, &[hc]].concat())?;
    let comb = comb.reshape([lead, &[hc, hc]].concat())?;
    Ok(HcMix { pre, post, comb })
}

/// Sinkhorn-normalize `comb` (`[n, hc, hc]`): one column pass, then
/// `iters - 1` row-then-column rounds.
///
/// A small CPU f32 matrix is normalized in one owned buffer instead of
/// allocating a sum tensor and a quotient tensor per pass (twenty rounds is
/// the loader default).  Every reduction keeps the tensor path's order, so
/// the result is bit-identical: rows through the same `vec_reduce_sum`
/// candle's last-dim sum uses, columns accumulated top to bottom, and both
/// as a sequential f64 sum under the precise policy (`precise::Rows`).
/// `sum * 1.0 + eps` is `affine(1.0, eps)` written out.
fn normalize_comb(comb: &Tensor, hc: usize, iters: usize, eps: f64) -> Result<Tensor> {
    if matches!(hc, 1..=4)
        && comb.dtype() == candle_core::DType::F32
        && comb.device().is_cpu()
        && comb.is_contiguous()
    {
        use candle_core::cpu::kernels::VecOps;

        let (storage, layout) = comb.storage_and_layout();
        let candle_core::Storage::Cpu(storage) = &*storage else {
            unreachable!()
        };
        let source = storage.as_slice::<f32>()?;
        let mut output =
            source[layout.start_offset()..layout.start_offset() + comb.elem_count()].to_vec();
        let precise = crate::precise::active(comb);
        let eps = eps as f32;
        for matrix in output.chunks_exact_mut(hc * hc) {
            for iteration in 0..iters.max(1) {
                if iteration != 0 {
                    for row in matrix.chunks_exact_mut(hc) {
                        let mut sum = 0f32;
                        if precise {
                            let mut wide_sum = 0f64;
                            for &value in row.iter() {
                                wide_sum += value as f64;
                            }
                            sum = wide_sum as f32;
                        } else {
                            // SAFETY: `row` holds exactly `hc` initialized f32s.
                            unsafe { f32::vec_reduce_sum(row.as_ptr(), &mut sum, hc) };
                        }
                        let denominator = sum * 1.0 + eps;
                        for value in row {
                            *value /= denominator;
                        }
                    }
                }
                let mut denominators = [0f32; 4];
                for column in 0..hc {
                    let mut sum = 0f32;
                    if precise {
                        let mut wide_sum = 0f64;
                        for row in 0..hc {
                            wide_sum += matrix[row * hc + column] as f64;
                        }
                        sum = wide_sum as f32;
                    } else {
                        for row in 0..hc {
                            sum += matrix[row * hc + column];
                        }
                    }
                    denominators[column] = sum * 1.0 + eps;
                }
                for row in matrix.chunks_exact_mut(hc) {
                    for (value, denominator) in row.iter_mut().zip(denominators) {
                        *value /= denominator;
                    }
                }
            }
        }
        return Tensor::from_vec(output, comb.shape(), comb.device());
    }
    let mut comb = comb.broadcast_div(&crate::precise::sum_keepdim(comb, 1)?.affine(1.0, eps)?)?;
    for _ in 0..iters.saturating_sub(1) {
        comb = comb.broadcast_div(&crate::precise::sum_keepdim(&comb, 2)?.affine(1.0, eps)?)?;
        comb = comb.broadcast_div(&crate::precise::sum_keepdim(&comb, 1)?.affine(1.0, eps)?)?;
    }
    Ok(comb)
}

#[cfg(test)]
mod sinkhorn_tests {
    use super::*;

    fn reference(comb: &Tensor, iters: usize, eps: f64) -> Result<Tensor> {
        let mut comb =
            comb.broadcast_div(&crate::precise::sum_keepdim(comb, 1)?.affine(1.0, eps)?)?;
        for _ in 0..iters.saturating_sub(1) {
            comb = comb.broadcast_div(&crate::precise::sum_keepdim(&comb, 2)?.affine(1.0, eps)?)?;
            comb = comb.broadcast_div(&crate::precise::sum_keepdim(&comb, 1)?.affine(1.0, eps)?)?;
        }
        Ok(comb)
    }

    #[test]
    fn cpu_normalization_preserves_reduction_bits_and_offsets() -> Result<()> {
        for hc in 1..=5 {
            for rows in [0, 1, 7, 129] {
                let values = (0..(rows + 2) * hc * hc)
                    .map(|index| ((index * 73 % 101) as f32 + 0.125) / 103.0)
                    .collect::<Vec<_>>();
                let input = Tensor::from_vec(values, (rows + 2, hc, hc), &candle_core::Device::Cpu)?;
                let input = input.narrow(0, 1, rows)?;
                for input in [&input, &input.transpose(1, 2)?] {
                    for iters in [0, 1, 2, 20] {
                        for eps in [0.0, 1e-12, 1e-5] {
                            let actual = normalize_comb(input, hc, iters, eps)?;
                            let expected = reference(input, iters, eps)?;
                            assert_eq!(actual.dims(), expected.dims());
                            let actual = actual.flatten_all()?.to_vec1::<f32>()?;
                            let expected = expected.flatten_all()?.to_vec1::<f32>()?;
                            assert_eq!(
                                actual.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                                expected.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                                "hc={hc} rows={rows} iters={iters} eps={eps}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(feature = "vulkan")]
struct FusedSinkhorn {
    rows: usize,
    scale: [f32; 3],
    iters: usize,
    eps: f32,
}
#[cfg(feature = "vulkan")]
impl candle_core::CustomOp2 for FusedSinkhorn {
    fn name(&self) -> &'static str {
        "fused-sinkhorn"
    }
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("fused-sinkhorn requires native Vulkan")
    }
    fn vulkan_fwd(
        &self,
        x: &candle_core::VulkanStorage,
        lx: &candle_core::Layout,
        base: &candle_core::VulkanStorage,
        lb: &candle_core::Layout,
    ) -> Result<(candle_core::VulkanStorage, candle_core::Shape)> {
        let shape = candle_core::Shape::from(self.rows * 24);
        let out = x
            .device
            .alloc(candle_core::DType::F32, shape.elem_count())?;
        candle_core::vulkan_backend::kernels::run_fused_mhc(
            &x.device,
            x.buf(),
            base.buf(),
            out.buf(),
            self.rows,
            lx.start_offset(),
            lb.start_offset(),
            self.iters,
            self.eps,
            self.scale,
        )?;
        Ok((out, shape))
    }
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
    crate::activation_trace::record("hc-flat", &flat)?;
    let mixes = {
        let norm = rms_rows(&flat, rms_eps)?;
        crate::activation_trace::record("hc-rms", &norm)?;
        hc_fn.forward(&norm)? // [b*s, (2+hc)*hc]
    };
    crate::activation_trace::record("hc-projection", &mixes)?;
    let mixes = mixes.reshape((b, s, (2 + hc) * hc))?;
    Ok((mixes, hc))
}

#[cfg(feature = "vulkan")]
struct VulkanRmsRowsF64(f32);
#[cfg(feature = "vulkan")]
impl candle_core::CustomOp1 for VulkanRmsRowsF64 {
    fn name(&self) -> &'static str {
        "rms-rows-f64"
    }
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("rms-rows-f64 requires Vulkan")
    }
    fn vulkan_fwd(
        &self,
        x: &candle_core::VulkanStorage,
        l: &candle_core::Layout,
    ) -> Result<(candle_core::VulkanStorage, candle_core::Shape)> {
        let cols = l.dims().last().copied().unwrap_or(0);
        let rows = l.shape().elem_count() / cols;
        let out = x.device.alloc(candle_core::DType::F32, rows * cols)?;
        candle_core::vulkan_backend::kernels::run_rms_rows_f64(
            &x.device,
            x.buf(),
            out.buf(),
            rows,
            cols,
            l.start_offset(),
            self.0,
        )?;
        Ok((out, l.shape().clone()))
    }
}

/// Unweighted RMS norm over the last dim.
pub fn rms_rows(x: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    if matches!(x.device(), candle_core::Device::Vulkan(_))
        && x.dtype() == candle_core::DType::F32 && x.is_contiguous()
        && x.dim(D::Minus1)? != 0
        && matches!(x.device(),candle_core::Device::Vulkan(d) if candle_core::vulkan_backend::kernels::f64_rms_enabled_on(d))
    {
        return x.apply_op1_no_bwd(&VulkanRmsRowsF64(eps as f32));
    }
    // A flattened hyper-connection row can be 20,480 elements wide. The
    // generic CPU f32 sum loses enough low bits to move the normalized row
    // by >1e-5. Borrow the input and accumulate in f64, using only the output
    // allocation rather than materializing square/mean/power intermediates.
    if x.device().is_cpu() && x.dtype() == candle_core::DType::F32 && x.is_contiguous() {
        let cols = x.dim(D::Minus1)?;
        if cols != 0 {
            let (storage, layout) = x.storage_and_layout();
            let candle_core::Storage::Cpu(cpu) = &*storage else { unreachable!() };
            let values = cpu.as_slice::<f32>()?;
            let values = &values[layout.start_offset()..layout.start_offset() + x.elem_count()];
            let mut out = Vec::with_capacity(values.len());
            for row in values.chunks_exact(cols) {
                let mut sums = [0f64; 8];
                for chunk in row.chunks(8) {
                    for (sum, &v) in sums.iter_mut().zip(chunk) {
                        *sum += (v as f64) * (v as f64);
                    }
                }
                let inv = (sums.iter().sum::<f64>() / cols as f64 + eps).sqrt().recip();
                out.extend(row.iter().map(|&v| (v as f64 * inv) as f32));
            }
            return Tensor::from_vec(out, x.shape(), x.device());
        }
    }
    x.broadcast_mul(
        &x.sqr()?
            .mean_keepdim(D::Minus1)?
            .affine(1.0, eps)?
            .powf(-0.5)?,
    )
}

#[cfg(test)]
mod rms_tests {
    use super::*;

    #[test]
    fn wide_cpu_rows_match_analytic_f64_norm_with_storage_offset() -> Result<()> {
        // A closed-form sum checks the 20K-element cancellation regression
        // independently of either implementation's reduction order.
        let cols = 20481;
        let mut data = vec![0.1f32; 3 * cols];
        data[cols] = 0.6;
        data[2 * cols] = -0.6;
        let input = Tensor::from_vec(data, (3, cols), &candle_core::Device::Cpu)?;
        let x = input.narrow(0, 1, 2)?;
        for eps in [1e-20, 1e-6] {
            let y = rms_rows(&x, eps)?.to_vec2::<f32>()?;
            let a = 0.1f32 as f64;
            let b = 0.6f32 as f64;
            let denom = (((cols - 1) as f64 * a * a + b * b) / cols as f64 + eps).sqrt();
            for (row, sign) in y.iter().zip([1., -1.]) {
                assert!((row[0] as f64 - sign * b / denom).abs() < 1e-6);
                assert!(row[1..].iter().all(|&v| (v as f64 - a / denom).abs() < 1e-6));
            }
        }
        Ok(())
    }
}

/// Collapse the `hc` copies of `x` (`[b, s, hc, d]`) to one stream with the
/// weights `pre` (`[b, s, hc]`).
pub fn collapse(x: &Tensor, pre: &Tensor) -> Result<Tensor> {
    if let [b, seq, 4, width] = x.dims() {
        let (b, seq, width) = (*b, *seq, *width);
        let enabled = match x.device() {
            candle_core::Device::Cpu => crate::precise::active(x),
            #[cfg(feature = "vulkan")]
            candle_core::Device::Vulkan(d) => {
                candle_core::vulkan_backend::kernels::precise_prefill_optimized(d, b * seq)
            }
            _ => false,
        };
        if enabled
            && x.dtype() == candle_core::DType::F32
            && pre.dtype() == candle_core::DType::F32
            && x.is_contiguous()
            && pre.is_contiguous()
            && pre.dims() == [b, seq, 4]
            && x.device().same_device(pre.device())
        {
            return x
                .apply_op2_no_bwd(
                    pre,
                    &FusedCollapse {
                        rows: b * seq,
                        width,
                    },
                )?
                .reshape((b, seq, width));
        }
    }
    let weighted = pre.unsqueeze(D::Minus1)?.broadcast_as(x.shape())?.mul(x)?;
    crate::precise::sum(&weighted, D::Minus2)
}

struct FusedCollapse {
    rows: usize,
    width: usize,
}
impl candle_core::CustomOp2 for FusedCollapse {
    fn name(&self) -> &'static str {
        "mhc-collapse-f64"
    }
    fn cpu_fwd(
        &self,
        x: &candle_core::CpuStorage,
        lx: &candle_core::Layout,
        pre: &candle_core::CpuStorage,
        lp: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        let x = x.as_slice::<f32>()?;
        let pre = pre.as_slice::<f32>()?;
        let mut out = vec![0f32; self.rows * self.width];
        for (i, value) in out.iter_mut().enumerate() {
            let row = i / self.width;
            let col = i % self.width;
            let mut sum = 0f64;
            for h in 0..4 {
                let product = x[lx.start_offset() + (row * 4 + h) * self.width + col]
                    * pre[lp.start_offset() + row * 4 + h];
                sum += product as f64;
            }
            *value = sum as f32;
        }
        Ok((
            candle_core::CpuStorage::F32(out),
            candle_core::Shape::from((self.rows, self.width)),
        ))
    }
    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(
        &self,
        x: &candle_core::VulkanStorage,
        lx: &candle_core::Layout,
        pre: &candle_core::VulkanStorage,
        lp: &candle_core::Layout,
    ) -> Result<(candle_core::VulkanStorage, candle_core::Shape)> {
        let shape = candle_core::Shape::from((self.rows, self.width));
        let out = x
            .device
            .alloc(candle_core::DType::F32, shape.elem_count())?;
        candle_core::vulkan_backend::kernels::run_mhc_collapse(
            &x.device,
            x.buf(),
            pre.buf(),
            out.buf(),
            self.rows,
            self.width,
            lx.start_offset(),
            lp.start_offset(),
        )?;
        Ok((out, shape))
    }
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
    let comb_src = crate::precise::matmul(&comb.transpose(2,3)?.contiguous()?, &residual.contiguous()?)?;
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

#[cfg(test)]
mod cpu_collapse_tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    #[ignore = "requires JOSHUA_CPU_DOT_F64=1"]
    fn cpu_fused_collapse_preserves_products_reduction_and_offsets() -> Result<()> {
        assert!(crate::precise::cpu_enabled());
        for (rows, width) in [(1usize, 7usize), (2, 13), (25, 5120), (32, 5120)] {
            let values: Vec<f32> = (0..(rows + 1) * 4 * width)
                .map(|i| ((i * 17 % 251) as f32 - 125.) / 11.7)
                .collect();
            let coefficients: Vec<f32> = (0..(rows + 1) * 4)
                .map(|i| ((i * 31 % 53) as f32 - 26.) / 11.3)
                .collect();
            let mut expected = Vec::new();
            for row in 0..rows {
                for col in 0..width {
                    let mut sum = 0f64;
                    for h in 0..4 {
                        let product = values[((row + 1) * 4 + h) * width + col]
                            * coefficients[(row + 1) * 4 + h];
                        sum += product as f64;
                    }
                    expected.push(sum as f32);
                }
            }
            let x = Tensor::from_vec(values, (rows + 1, 4, width), &Device::Cpu)?
                .narrow(0, 1, rows)?
                .unsqueeze(0)?;
            let p = Tensor::from_vec(coefficients, (rows + 1, 4), &Device::Cpu)?
                .narrow(0, 1, rows)?
                .unsqueeze(0)?;
            let weighted = p.unsqueeze(D::Minus1)?.broadcast_as(x.shape())?.mul(&x)?;
            assert_eq!(
                crate::precise::sum(&weighted, D::Minus2)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                expected
            );
            assert_eq!(collapse(&x, &p)?.flatten_all()?.to_vec1::<f32>()?, expected);
            assert_eq!(
                collapse(&x.squeeze(0)?, &p.squeeze(0)?)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                expected
            );
        }
        let ordinary = Tensor::ones((1, 1, 3, 4), DType::F32, &Device::Cpu)?;
        assert_eq!(
            collapse(
                &ordinary,
                &Tensor::ones((1, 1, 3), DType::F32, &Device::Cpu)?
            )?
            .flatten_all()?
            .to_vec1::<f32>()?,
            vec![3.; 4]
        );
        Ok(())
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod fused_sinkhorn_tests {
    use super::*;
    use candle_core::{Device, Storage};

    #[test]
    #[ignore = "requires native Vulkan and JOSHUA_VULKAN_PREFILL_OPT=1"]
    fn fused_collapse_preserves_f32_products_and_ordered_f64_sum() -> Result<()> {
        let vk = candle_core::VulkanDevice::new(0)?.with_precise_math()?;
        assert!(candle_core::vulkan_backend::kernels::precise_prefill_optimized(&vk, 32));
        let dev = Device::Vulkan(vk.clone());
        let before = candle_core::vulkan_backend::kernels::fallback_count();
        for (b, seq, width) in [(1usize, 2usize, 7usize), (1, 25, 5120), (2, 16, 13)] {
            let rows = b * seq;
            let data: Vec<f32> = (0..(rows + 1) * 4 * width)
                .map(|i| ((i * 17 % 251) as f32 - 125.) / 11.7)
                .collect();
            let weights: Vec<f32> = (0..(rows + 1) * 4)
                .map(|i| ((i * 31 % 53) as f32 - 26.) / 11.3)
                .collect();
            let mut expected = Vec::new();
            for r in 0..rows {
                for col in 0..width {
                    let mut sum = 0f64;
                    for h in 0..4 {
                        let product =
                            data[((r + 1) * 4 + h) * width + col] * weights[(r + 1) * 4 + h];
                        sum += product as f64;
                    }
                    expected.push(sum as f32);
                }
            }
            let x = Tensor::from_vec(data, ((rows + 1) * 4 * width,), &dev)?
                .narrow(0, 4 * width, rows * 4 * width)?
                .reshape((b, seq, 4, width))?;
            let pre = Tensor::from_vec(weights, ((rows + 1) * 4,), &dev)?
                .narrow(0, 4, rows * 4)?
                .reshape((b, seq, 4))?;
            assert_eq!(
                collapse(&x, &pre)?.flatten_all()?.to_vec1::<f32>()?,
                expected
            );
            if b == 1 {
                // Non-4D layouts must retain the existing broadcast/reduce
                // fallback rather than failing the fused shape check.
                assert_eq!(
                    collapse(&x.squeeze(0)?, &pre.squeeze(0)?)?.flatten_all()?.to_vec1::<f32>()?,
                    expected
                );
            }
        }
        assert_eq!(
            before,
            candle_core::vulkan_backend::kernels::fallback_count()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires native Vulkan and CPU/Vulkan precision plus JOSHUA_VULKAN_FUSED_MHC=1"]
    fn fused_sinkhorn_matches_unfused_cpu_with_borrowed_outputs() -> Result<()> {
        assert!(crate::precise::cpu_enabled());
        assert!(candle_core::vulkan_backend::kernels::fused_mhc_enabled());
        let gpu = Device::new_vulkan(0)?;
        assert!(crate::precise::active(&Tensor::zeros(
            1,
            candle_core::DType::F32,
            &gpu
        )?));
        let before = candle_core::vulkan_backend::kernels::fallback_count();
        let scale = [0.173_f32 as f64, -0.37_f32 as f64, 0.61_f32 as f64];
        let buffer = |x: &Tensor| {
            let (s, _) = x.storage_and_layout();
            match &*s {
                Storage::Vulkan(s) => s.buffer,
                _ => unreachable!(),
            }
        };
        for rows in [0, 1, 2, 32, 33] {
            let data: Vec<f32> = (0..(rows + 2) * 24)
                .map(|i| ((i * 17 % 251) as f32 - 125.) / 11.7)
                .collect();
            let bias: Vec<f32> = (0..26)
                .map(|i| ((i * 31 % 53) as f32 - 26.) / 11.3)
                .collect();
            for batches in [1, 2] {
                if rows % batches != 0 {
                    continue;
                }
                for iters in [0, 1, 3, 20] {
                    for eps in [0., 1e-6, 1e-4] {
                        let mut values = Vec::new();
                        for dev in [&Device::Cpu, &gpu] {
                            let x = Tensor::from_vec(data.clone(), (rows + 2, 24), dev)?
                                .narrow(0, 1, rows)?
                                .reshape((batches, rows / batches, 24))?;
                            let b = Tensor::from_vec(bias.clone(), 26, dev)?.narrow(0, 1, 24)?;
                            let v = split_sinkhorn_with_scale(&x, &scale, &b, 4, iters, eps)?;
                            if matches!(dev, Device::Vulkan(_)) {
                                assert_eq!(buffer(&v.pre), buffer(&v.post));
                                assert_eq!(buffer(&v.pre), buffer(&v.comb));
                            }
                            values.push(
                                [v.pre, v.post, v.comb]
                                    .map(|x| x.flatten_all()?.to_vec1::<f32>())
                                    .into_iter()
                                    .collect::<Result<Vec<_>>>()?,
                            );
                        }
                        assert_eq!(
                            values[0], values[1],
                            "rows={rows} batches={batches} iters={iters} eps={eps}"
                        );
                    }
                }
            }
        }
        assert_eq!(
            candle_core::vulkan_backend::kernels::fallback_count(),
            before
        );
        Ok(())
    }
}
