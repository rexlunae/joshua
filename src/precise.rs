//! Opt-in accurate arithmetic for isolating cross-backend DeepSeek drift.
//! Activations and weights stay f32; only register/small row accumulators widen.
use candle_core::{CpuStorage, CustomOp1, CustomOp2, DType, Device, Layout, Result, Shape, Tensor};
use rayon::prelude::*;
use std::{cell::RefCell, sync::OnceLock};

pub(crate) fn cpu_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    #[cfg(target_arch = "x86_64")]
    {
        *ON.get_or_init(|| std::env::var("JOSHUA_CPU_DOT_F64").is_ok_and(|v| v == "1"))
            && crate::simd::avx2_fma_available()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = &ON;
        false
    }
}
pub(crate) fn active(x: &Tensor) -> bool {
    if x.dtype() != DType::F32 {
        return false;
    }
    match x.device() {
        Device::Cpu => cpu_enabled(),
        #[cfg(feature = "vulkan")]
        Device::Vulkan(_) => {
            candle_core::vulkan_backend::kernels::f64_dot_enabled()
                && candle_core::vulkan_backend::kernels::native_enabled()
        }
        _ => false,
    }
}

// FMA range reduction and degree-14 exp on [-ln(2)/2,ln(2)/2].
// Split constants are exact f32 values shared with the shader, not decimal
// literals rounded by different shader/front-end parsers.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn exp64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x < -208. {
        return 0.;
    }
    if x > 89. {
        return f64::INFINITY;
    }
    let inv = f32::from_bits(0x3fb8aa3b) as f64 + f32::from_bits(0x32a57060) as f64;
    let n = x.mul_add(inv, 0.5).floor() as i32;
    let r = (-(n as f64)).mul_add(f32::from_bits(0x3f317218) as f64, x);
    let r = (-(n as f64)).mul_add(f32::from_bits(0xb102e308) as f64, r);
    let r = (-(n as f64)).mul_add(f32::from_bits(0xa4ca86c4) as f64, r);
    const COEFF: [f64; 15] = [
        1.,
        1.,
        1. / 2.,
        1. / 6.,
        1. / 24.,
        1. / 120.,
        1. / 720.,
        1. / 5040.,
        1. / 40320.,
        1. / 362880.,
        1. / 3628800.,
        1. / 39916800.,
        1. / 479001600.,
        1. / 6227020800.,
        1. / 87178291200.,
    ];
    let mut p = COEFF[14];
    for i in (0..14).rev() {
        p = p.mul_add(r, COEFF[i]);
    }
    let h = n / 3;
    p * f32::from_bits(((h + 127) as u32) << 23) as f64
        * f32::from_bits(((h + 127) as u32) << 23) as f64
        * f32::from_bits(((n - 2 * h + 127) as u32) << 23) as f64
}
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn unary64(x: f32, op: u32) -> f32 {
    let v = x as f64;
    if op == 0 {
        return exp64(v) as f32;
    }
    let e = exp64(-v.abs());
    if op == 3 {
        let z = e / (2. + e);
        let zz = z * z;
        let mut p: f64 = 1. / 31.;
        for i in (0..15).rev() {
            p = p.mul_add(zz, 1. / (2 * i + 1) as f64);
        }
        return (v.max(0.) + 2. * z * p).sqrt() as f32;
    }
    let sigmoid = if v >= 0. { 1. / (1. + e) } else { e / (1. + e) };
    if op == 1 {
        sigmoid as f32
    } else {
        (v * sigmoid) as f32
    }
}
struct Unary(u32);
impl CustomOp1 for Unary {
    fn name(&self) -> &'static str {
        "precise-unary"
    }
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let data = s.as_slice::<f32>()?;
        let data = &data[l.start_offset()..l.start_offset() + l.shape().elem_count()];
        let out: Vec<f32> = if data.len() >= 2048 {
            data.par_iter()
                .map(|&x| unsafe { unary64(x, self.0) })
                .collect()
        } else {
            data.iter()
                .map(|&x| unsafe { unary64(x, self.0) })
                .collect()
        };
        Ok((CpuStorage::F32(out), l.shape().clone()))
    }
    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(
        &self,
        s: &candle_core::VulkanStorage,
        l: &Layout,
    ) -> Result<(candle_core::VulkanStorage, Shape)> {
        let n = l.shape().elem_count();
        let out = s.device.alloc(DType::F32, n)?;
        candle_core::vulkan_backend::kernels::run_precise_unary(
            &s.device,
            s.buf(),
            out.buf(),
            n,
            l.start_offset(),
            self.0,
        )?;
        Ok((out, l.shape().clone()))
    }
}
pub(crate) fn exp(x: &Tensor) -> Result<Tensor> {
    if active(x) && x.is_contiguous() {
        x.apply_op1_no_bwd(&Unary(0))
    } else {
        x.exp()
    }
}
pub(crate) fn sigmoid(x: &Tensor) -> Result<Tensor> {
    if active(x) && x.is_contiguous() {
        x.apply_op1_no_bwd(&Unary(1))
    } else {
        candle_nn::ops::sigmoid(x)
    }
}
pub(crate) fn silu(x: &Tensor) -> Result<Tensor> {
    if active(x) && x.is_contiguous() {
        x.apply_op1_no_bwd(&Unary(2))
    } else {
        candle_nn::ops::silu(x)
    }
}
pub(crate) fn sqrt_softplus(x: &Tensor) -> Result<Tensor> {
    if active(x) && x.is_contiguous() {
        x.apply_op1_no_bwd(&Unary(3))
    } else {
        crate::moe::softplus(x)?.sqrt()
    }
}
struct Rows {
    axis: usize,
    softmax: bool,
}
impl Rows {
    fn geometry(&self, l: &Layout) -> (usize, usize, usize, Shape) {
        let d = l.dims();
        let cols = d[self.axis];
        let inner = d[self.axis + 1..].iter().product::<usize>();
        let rows = l.shape().elem_count() / cols;
        let mut out = d.to_vec();
        if !self.softmax {
            out[self.axis] = 1;
        }
        (rows, cols, inner, Shape::from(out))
    }
}
impl CustomOp1 for Rows {
    fn name(&self) -> &'static str {
        "precise-row-reduction"
    }
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let (rows, cols, inner, shape) = self.geometry(l);
        let src = s.as_slice::<f32>()?;
        let mut out = vec![0f32; shape.elem_count()];
        for row in 0..rows {
            let base = row / inner * cols * inner + row % inner;
            let start = l.start_offset() + base;
            if self.softmax {
                let mut max = f32::NEG_INFINITY;
                for c in 0..cols {
                    max = max.max(src[start + c * inner]);
                }
                let mut sum = 0f64;
                for c in 0..cols {
                    sum += unsafe { exp64(src[start + c * inner] as f64 - max as f64) };
                }
                for c in 0..cols {
                    out[base + c * inner] =
                        (unsafe { exp64(src[start + c * inner] as f64 - max as f64) } / sum) as f32;
                }
            } else {
                let mut sum = 0f64;
                for c in 0..cols {
                    sum += src[start + c * inner] as f64;
                }
                out[row] = sum as f32;
            }
        }
        Ok((CpuStorage::F32(out), shape))
    }
    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(
        &self,
        s: &candle_core::VulkanStorage,
        l: &Layout,
    ) -> Result<(candle_core::VulkanStorage, Shape)> {
        let (rows, cols, inner, shape) = self.geometry(l);
        let out = s.device.alloc(DType::F32, shape.elem_count())?;
        candle_core::vulkan_backend::kernels::run_precise_rows(
            &s.device,
            s.buf(),
            out.buf(),
            rows,
            cols,
            inner,
            l.start_offset(),
            self.softmax,
        )?;
        Ok((out, shape))
    }
}
pub(crate) fn softmax<D: candle_core::shape::Dim>(x: &Tensor, dim: D) -> Result<Tensor> {
    let axis = dim.to_index(x.shape(), "precise-softmax")?;
    if active(x) && x.is_contiguous() && x.dims()[axis] > 0 {
        x.apply_op1_no_bwd(&Rows {
            axis,
            softmax: true,
        })
    } else {
        candle_nn::ops::softmax(x, axis)
    }
}
pub(crate) fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    softmax(x, candle_core::D::Minus1)
}
pub(crate) fn sum_keepdim<D: candle_core::shape::Dim>(x: &Tensor, dim: D) -> Result<Tensor> {
    let axis = dim.to_index(x.shape(), "precise-sum")?;
    if active(x) && x.is_contiguous() && x.dims()[axis] > 0 {
        x.apply_op1_no_bwd(&Rows {
            axis,
            softmax: false,
        })
    } else {
        x.sum_keepdim(axis)
    }
}
pub(crate) fn sum<D: candle_core::shape::Dim>(x: &Tensor, dim: D) -> Result<Tensor> {
    let axis = dim.to_index(x.shape(), "precise-sum")?;
    sum_keepdim(x, axis)?.squeeze(axis)
}
#[derive(Clone)]
struct Matrix {
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    ab: usize,
    bb: usize,
    ar: usize,
    ac: usize,
    br: usize,
    bc: usize,
    shape: Shape,
}
fn batch_stride(l: &Layout) -> Option<usize> {
    let d = l.dims();
    if d.len() == 2 {
        return Some(0);
    }
    let mut expected = (0..d.len() - 2)
        .rev()
        .find(|&i| d[i] > 1)
        .map(|i| l.stride()[i])
        .unwrap_or(0);
    let result = expected;
    for i in (0..d.len() - 2).rev() {
        if d[i] > 1 && l.stride()[i] != expected {
            return None;
        }
        expected = expected.checked_mul(d[i])?;
    }
    Some(result)
}
fn matrix(a: &Layout, b: &Layout) -> Option<Matrix> {
    let ad = a.dims();
    let bd = b.dims();
    let r = ad.len();
    if r < 2 || bd.len() != r || ad[..r - 2] != bd[..r - 2] || ad[r - 1] != bd[r - 2] {
        return None;
    }
    let (m, k, n) = (ad[r - 2], ad[r - 1], bd[r - 1]);
    if m == 0 || n == 0 || k == 0 {
        return None;
    }
    let batch = ad[..r - 2].iter().product();
    let mut out = ad.to_vec();
    out[r - 1] = n;
    Some(Matrix {
        batch,
        m,
        k,
        n,
        ab: batch_stride(a)?,
        bb: batch_stride(b)?,
        ar: a.stride()[r - 2],
        ac: a.stride()[r - 1],
        br: b.stride()[r - 2],
        bc: b.stride()[r - 1],
        shape: Shape::from(out),
    })
}
thread_local! {static MATRIX_ROW:RefCell<Vec<f64>>=const{RefCell::new(Vec::new())};}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matrix_row(a: &[f32], b: &[f32], g: &Matrix, ao: usize, bo: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    if g.br == 1 && g.ac == 1 && g.bc != 1 {
        // A transposed RHS stores each output's weights contiguously along
        // K. Block four outputs in registers instead of sweeping every
        // strided column for each K (which thrashes a few cache sets).
        // Each lane still performs precisely the original sequential FMA
        // order, so this changes locality without changing accumulation.
        let mut j = 0;
        while j + 4 <= g.n {
            let mut acc = _mm256_setzero_pd();
            for k in 0..g.k {
                let w = bo + k;
                let weights = _mm256_cvtps_pd(_mm_set_ps(
                    b[w + (j + 3) * g.bc],
                    b[w + (j + 2) * g.bc],
                    b[w + (j + 1) * g.bc],
                    b[w + j * g.bc],
                ));
                acc = _mm256_fmadd_pd(_mm256_set1_pd(a[ao + k] as f64), weights, acc);
            }
            _mm_storeu_ps(out.as_mut_ptr().add(j), _mm256_cvtpd_ps(acc));
            j += 4;
        }
        while j < g.n {
            let mut acc = 0f64;
            for k in 0..g.k {
                acc = (a[ao + k] as f64).mul_add(b[bo + k + j * g.bc] as f64, acc);
            }
            out[j] = acc as f32;
            j += 1;
        }
        return;
    }
    MATRIX_ROW.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.resize(g.n, 0.);
        scratch.fill(0.);
        for k in 0..g.k {
            let x = _mm256_set1_pd(a[ao + k * g.ac] as f64);
            let w = bo + k * g.br;
            let mut j = 0;
            while j + 4 <= g.n {
                let v = if g.bc == 1 {
                    _mm256_cvtps_pd(_mm_loadu_ps(b.as_ptr().add(w + j)))
                } else {
                    _mm256_set_pd(
                        b[w + (j + 3) * g.bc] as f64,
                        b[w + (j + 2) * g.bc] as f64,
                        b[w + (j + 1) * g.bc] as f64,
                        b[w + j * g.bc] as f64,
                    )
                };
                let p = scratch.as_mut_ptr().add(j);
                _mm256_storeu_pd(p, _mm256_fmadd_pd(x, v, _mm256_loadu_pd(p)));
                j += 4;
            }
            while j < g.n {
                scratch[j] = (a[ao + k * g.ac] as f64).mul_add(b[w + j * g.bc] as f64, scratch[j]);
                j += 1;
            }
        }
        for (d, &v) in out.iter_mut().zip(scratch.iter()) {
            *d = v as f32;
        }
    });
}
#[cfg(target_arch = "x86_64")]
fn cpu_matrix_into(a: &[f32], b: &[f32], g: &Matrix, ao: usize, bo: usize, out: &mut [f32]) {
    let work = |i: usize, row: &mut [f32]| {
        let bz = i / g.m;
        let m = i % g.m;
        unsafe { matrix_row(a, b, g, ao + bz * g.ab + m * g.ar, bo + bz * g.bb, row) }
    };
    if g.batch * g.m > 1 && g.m.saturating_mul(g.n).saturating_mul(g.k) >= 131072 {
        out.par_chunks_mut(g.n)
            .enumerate()
            .for_each(|(i, row)| work(i, row));
    } else {
        for (i, row) in out.chunks_mut(g.n).enumerate() {
            work(i, row);
        }
    }
}

impl CustomOp2 for Matrix {
    fn name(&self) -> &'static str {
        "precise-matmul"
    }
    fn cpu_fwd(
        &self,
        a: &CpuStorage,
        la: &Layout,
        b: &CpuStorage,
        lb: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (a, la, b, lb);
            candle_core::bail!("precise matmul requires AVX2/FMA");
        }
        #[cfg(target_arch = "x86_64")]
        {
            let a = a.as_slice::<f32>()?;
            let b = b.as_slice::<f32>()?;
            let mut out = vec![0f32; self.shape.elem_count()];
            cpu_matrix_into(a, b, self, la.start_offset(), lb.start_offset(), &mut out);
            Ok((CpuStorage::F32(out), self.shape.clone()))
        }
    }
    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(
        &self,
        a: &candle_core::VulkanStorage,
        la: &Layout,
        b: &candle_core::VulkanStorage,
        lb: &Layout,
    ) -> Result<(candle_core::VulkanStorage, Shape)> {
        use candle_core::vulkan_backend::kernels::{run_matmul_with_precision, MatStrides};
        let out = a.device.alloc(DType::F32, self.shape.elem_count())?;
        let sa = MatStrides {
            row: self.ar,
            col: self.ac,
            offset: la.start_offset(),
            batch: self.ab,
        };
        let sb = MatStrides {
            row: self.br,
            col: self.bc,
            offset: lb.start_offset(),
            batch: self.bb,
        };
        run_matmul_with_precision(
            &a.device,
            a.buf(),
            b.buf(),
            out.buf(),
            (self.batch, self.m, self.n, self.k),
            sa,
            sb,
            true,
        )?;
        Ok((out, self.shape.clone()))
    }
}
pub(crate) fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if active(a) && active(b) {
        let la = a.layout();
        let lb = b.layout();
        if let Some(g) = matrix(la, lb) {
            return a.apply_op2_no_bwd(b, &g);
        }
    }
    a.matmul(b)
}

// Own the original compact allocation/mmap handle; the adapter adds no weight
// copy. Routed expert slices do not pass through this dense-loader adapter.
struct WideCpu(candle_core::quantized::QTensor);
impl WideCpu {
    fn inner(&self) -> &dyn candle_core::quantized::QuantizedType {
        match self.0.storage() {
            candle_core::quantized::QStorage::Cpu(s) => s.as_ref(),
            _ => unreachable!(),
        }
    }
}
impl candle_core::quantized::QuantizedType for WideCpu {
    fn dtype(&self) -> candle_core::quantized::GgmlDType {
        self.inner().dtype()
    }
    fn supports_grouped_q8(&self) -> bool {
        self.inner().dtype() == candle_core::quantized::GgmlDType::Q8_0
    }
    fn grouped_q8_matmul(
        &self,
        g: usize,
        k: usize,
        n: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<()> {
        let s = self.inner();
        let bytes = unsafe { std::slice::from_raw_parts(s.as_ptr(), s.storage_size_in_bytes()) };
        #[cfg(target_arch = "x86_64")]
        if crate::kquant_dot::grouped_q8_f64(g, k, n, x, bytes, y) {
            return Ok(());
        }
        candle_core::bail!("invalid selected-group Q8 precision geometry")
    }

    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        use candle_core::quantized::GgmlDType;
        let inner = self.inner();
        if matches!(
            inner.dtype(),
            GgmlDType::Q4K | GgmlDType::Q8_0 | GgmlDType::Q2K
        ) {
            let bytes = unsafe {
                std::slice::from_raw_parts(inner.as_ptr(), inner.storage_size_in_bytes())
            };
            #[cfg(target_arch = "x86_64")]
            if crate::kquant_dot::try_matmul_dense_f64(inner.dtype(), mkn, lhs, bytes, dst) {
                return Ok(());
            }
            candle_core::bail!("precise dense Q4_K requires AVX512");
        }
        if inner.dtype() == GgmlDType::F32 {
            let (m, k, n) = mkn;
            // CPU F32 quantized storage is typed/aligned, immutable and owned
            // by self.0 for the entire borrowed multiplication.
            let weights =
                unsafe { std::slice::from_raw_parts(inner.as_ptr() as *const f32, n * k) };
            let g = Matrix {
                batch: 1,
                m,
                k,
                n,
                ab: 0,
                bb: 0,
                ar: k,
                ac: 1,
                br: 1,
                bc: k,
                shape: Shape::from((m, n)),
            };
            #[cfg(target_arch = "x86_64")]
            {
                cpu_matrix_into(lhs, weights, &g, 0, 0, dst);
                return Ok(());
            }
        }
        inner.matmul_t(mkn, lhs, dst)
    }
    fn matmul_t_f16(
        &self,
        d: (usize, usize, usize),
        x: &[half::f16],
        y: &mut [half::f16],
    ) -> Result<()> {
        self.inner().matmul_t_f16(d, x, y)
    }
    fn embedding(&self, ids: &[u32], r: usize, h: usize) -> Result<CpuStorage> {
        self.inner().embedding(ids, r, h)
    }
    fn dequantize(&self, n: usize) -> Result<CpuStorage> {
        self.inner().dequantize(n)
    }
    fn storage_size_in_bytes(&self) -> usize {
        self.inner().storage_size_in_bytes()
    }
    fn as_ptr(&self) -> *const u8 {
        self.inner().as_ptr()
    }
    fn block_size(&self) -> usize {
        self.inner().block_size()
    }
    fn size(&self) -> usize {
        self.inner().size()
    }
    fn from_float(&mut self, _: &[f32]) {
        panic!("precise dense model weights are read-only")
    }
    fn from_float_imatrix(&mut self, _: &[f32], _: &[f32], _: usize) {
        panic!("precise dense model weights are read-only")
    }
}
pub(crate) fn qmatmul(
    q: candle_core::quantized::QTensor,
) -> Result<candle_core::quantized::QMatMul> {
    use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
    let dtype = q.dtype();
    if q.device().is_cpu()
        && cpu_enabled()
        && matches!(
            dtype,
            GgmlDType::Q4K | GgmlDType::F32 | GgmlDType::Q8_0 | GgmlDType::Q2K
        )
    {
        let shape = q.shape().clone();
        let q = QTensor::new(QStorage::Cpu(Box::new(WideCpu(q))), shape)?;
        return Ok(QMatMul::QTensor(std::sync::Arc::new(q)));
    }
    #[cfg(feature = "vulkan")]
    if matches!(q.device(), Device::Vulkan(_))
        && candle_core::vulkan_backend::kernels::f64_dot_enabled()
        && dtype == GgmlDType::F32
    {
        return Ok(QMatMul::QTensor(std::sync::Arc::new(q)));
    }
    QMatMul::from_qtensor(q)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reference(x: f32, op: u32) -> f32 {
        let x = x as f64;
        let e = (-x.abs()).exp();
        let s = if x >= 0. { 1. / (1. + e) } else { e / (1. + e) };
        match op {
            0 => x.exp() as f32,
            1 => s as f32,
            2 => (x * s) as f32,
            _ => (x.max(0.) + e.ln_1p()).sqrt() as f32,
        }
    }
    fn rounded_reference(got: f32, want: f32) {
        assert!(
            got == want || got.to_bits().abs_diff(want.to_bits()) <= 1,
            "got {got:e} expected {want:e}"
        );
    }
    #[test]
    fn scalar_math_matches_independent_float64_reference() {
        #[cfg(target_arch = "x86_64")]
        if !crate::simd::avx2_fma_available() {
            return;
        }
        for i in -4096..=1424 {
            let x = i as f32 / 16.;
            for op in 0..4 {
                rounded_reference(unsafe { unary64(x, op) }, reference(x, op));
            }
        }
    }
    #[test]
    #[ignore = "requires JOSHUA_CPU_DOT_F64=1"]
    fn selected_group_q8_matches_generic_diagonal_without_copying() -> Result<()> {
        use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
        use candle_core::Module;
        assert!(cpu_enabled());
        for (groups, k, rank) in [(1, 32, 7), (3, 96, 17), (8, 4096, 17)] {
            let n = groups * rank;
            let w: Vec<f32> = (0..n * k)
                .map(|i| ((i * 31 % 251) as f32 - 125.) / 64.)
                .collect();
            let x: Vec<f32> = (0..(groups + 2) * k)
                .map(|i| ((i * 17 % 127) as f32 - 63.) / 32.)
                .collect();
            let qt =
                QTensor::quantize(&Tensor::from_vec(w, (n, k), &Device::Cpu)?, GgmlDType::Q8_0)?;
            let weight = qmatmul(qt)?;
            let input = Tensor::from_vec(x, (groups + 2, k), &Device::Cpu)?.narrow(0, 1, groups)?;
            let generic = weight.forward(&input)?.to_vec2::<f32>()?;
            let selected = weight
                .try_grouped_q8(&input, groups)?
                .expect("precision Q8 selected path")
                .flatten_all()?
                .to_vec1::<f32>()?;
            let want: Vec<f32> = (0..groups)
                .flat_map(|g| generic[g][g * rank..(g + 1) * rank].iter().copied())
                .collect();
            assert_eq!(selected, want);
            let projected = crate::attention::grouped_output_projection(
                &weight,
                &input.reshape((1, groups, k))?,
                groups,
                rank,
            )?;
            assert_eq!(projected.flatten_all()?.to_vec1::<f32>()?, want);
            let strided = input.t()?.contiguous()?.t()?;
            if groups > 1 {
                assert!(!strided.is_contiguous());
                assert!(weight.try_grouped_q8(&strided, groups)?.is_none());
            }
        }
        let q = QTensor::quantize(
            &Tensor::ones((6, 32), DType::F32, &Device::Cpu)?,
            GgmlDType::Q8_0,
        )?;
        let ordinary = QMatMul::from_qtensor(q)?;
        assert!(ordinary
            .try_grouped_q8(&Tensor::ones((3, 32), DType::F32, &Device::Cpu)?, 3)?
            .is_none());
        Ok(())
    }

    #[test]
    #[ignore = "requires JOSHUA_CPU_DOT_F64=1 and AVX512"]
    fn dense_adapter_borrows_storage_and_matches_float64() -> Result<()> {
        use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
        use candle_core::Module;
        assert!(cpu_enabled());
        assert_eq!(crate::simd::simd_level(), crate::simd::SimdLevel::Avx512);
        let (m, k, n) = (5, 1280, 17);
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 31 % 251) as f32 - 125.) / 64.)
            .collect();
        let input: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 % 127) as f32 - 63.) / 32.)
            .collect();
        for dtype in [GgmlDType::Q4K, GgmlDType::F32] {
            let q = QTensor::quantize(
                &Tensor::from_vec(weights.clone(), (n, k), &Device::Cpu)?,
                dtype,
            )?;
            let reference_weights = q
                .dequantize(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let (ptr, bytes) = match q.storage() {
                QStorage::Cpu(s) => (s.as_ptr(), s.storage_size_in_bytes()),
                _ => unreachable!(),
            };
            let q = qmatmul(q)?;
            match &q {
                QMatMul::QTensor(t) => match t.storage() {
                    QStorage::Cpu(s) => {
                        assert_eq!(s.as_ptr(), ptr);
                        assert_eq!(s.storage_size_in_bytes(), bytes);
                    }
                    _ => unreachable!(),
                },
                _ => panic!("dense adapter lost quantized storage"),
            }
            let got = q
                .forward(&Tensor::from_vec(input.clone(), (m, k), &Device::Cpu)?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for i in 0..m {
                for j in 0..n {
                    let want = input[i * k..(i + 1) * k]
                        .iter()
                        .zip(&reference_weights[j * k..(j + 1) * k])
                        .map(|(&x, &w)| x as f64 * w as f64)
                        .sum::<f64>() as f32;
                    assert_eq!(got[i * n + j], want, "{dtype:?} ({i},{j})");
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires native Vulkan shaderFloat64 and all three precision environment flags"]
    fn native_profile_matches_cpu_and_independent_float64() -> Result<()> {
        let cpu = Device::Cpu;
        let gpu = Device::new_vulkan(0)?;
        assert!(active(&Tensor::zeros(1, DType::F32, &cpu)?));
        assert!(active(&Tensor::zeros(1, DType::F32, &gpu)?));
        let before = candle_core::vulkan_backend::kernels::fallback_count();
        let data: Vec<f32> = (0..2048)
            .map(|i| ((i * 31 % 1021) as f32 - 510.) / 8.)
            .collect();
        for op in 0..4 {
            let mut values = Vec::new();
            for dev in [&cpu, &gpu] {
                let x = Tensor::from_vec(data.clone(), (32, 64), dev)?;
                let y = match op {
                    0 => exp(&x)?,
                    1 => sigmoid(&x)?,
                    2 => silu(&x)?,
                    _ => sqrt_softplus(&x)?,
                };
                let y = y.flatten_all()?.to_vec1::<f32>()?;
                for (&v, &x) in y.iter().zip(&data) {
                    rounded_reference(v, reference(x, op));
                }
                values.push(y);
            }
            assert_eq!(values[0], values[1], "unary {op}");
        }
        let dims = [2, 33, 17];
        let data: Vec<f32> = (0..dims.iter().product())
            .map(|i| ((i * 17 % 251) as f32 - 125.) / 8.)
            .collect();
        for axis in 0..3 {
            let cols = dims[axis];
            let inner = dims[axis + 1..].iter().product::<usize>();
            let rows = data.len() / cols;
            let mut want_sum = vec![0f32; rows];
            let mut want_soft = vec![0f32; data.len()];
            for r in 0..rows {
                let base = r / inner * cols * inner + r % inner;
                let vals: Vec<f64> = (0..cols).map(|c| data[base + c * inner] as f64).collect();
                want_sum[r] = vals.iter().sum::<f64>() as f32;
                let m = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = vals.iter().map(|v| (v - m).exp()).collect();
                let den = e.iter().sum::<f64>();
                for c in 0..cols {
                    want_soft[base + c * inner] = (e[c] / den) as f32;
                }
            }
            let mut outputs = Vec::new();
            for dev in [&cpu, &gpu] {
                let x = Tensor::from_vec(data.clone(), dims.as_slice(), dev)?;
                let sum = sum_keepdim(&x, axis)?.flatten_all()?.to_vec1::<f32>()?;
                let soft = softmax(&x, axis)?.flatten_all()?.to_vec1::<f32>()?;
                assert_eq!(sum, want_sum, "axis {axis}");
                for (&v, &w) in soft.iter().zip(&want_soft) {
                    rounded_reference(v, w);
                }
                outputs.push(soft);
            }
            assert_eq!(outputs[0], outputs[1], "softmax axis {axis}");
        }
        for m in [1, 5, 33] {
            let (batch, k, n) = (2, 257, 17);
            let av: Vec<f32> = (0..batch * (m + 2) * k)
                .map(|i| ((i * 31 % 127) as f32 - 63.) / 32.)
                .collect();
            let bv: Vec<f32> = (0..batch * (n + 2) * k)
                .map(|i| ((i * 17 % 251) as f32 - 125.) / 64.)
                .collect();
            let mut want = vec![0f32; batch * m * n];
            for bz in 0..batch {
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0f64;
                        for kk in 0..k {
                            acc += av[(bz * (m + 2) + i + 1) * k + kk] as f64
                                * bv[(bz * (n + 2) + j + 1) * k + kk] as f64;
                        }
                        want[(bz * m + i) * n + j] = acc as f32;
                    }
                }
            }
            for dev in [&cpu, &gpu] {
                let a = Tensor::from_vec(av.clone(), (batch, m + 2, k), dev)?.narrow(1, 1, m)?;
                let b = Tensor::from_vec(bv.clone(), (batch, n + 2, k), dev)?
                    .narrow(1, 1, n)?
                    .transpose(1, 2)?;
                let got = matmul(&a, &b)?.flatten_all()?.to_vec1::<f32>()?;
                assert_eq!(got, want, "{dev:?} M={m}");
            }
        }
        for cols in [512, 1280, 5120, 20480] {
            let rows = 33;
            let data: Vec<f32> = (0..(rows + 2) * cols)
                .map(|i| ((i * 31 % 251) as f32 - 125.) / 32.)
                .collect();
            let alpha: Vec<f32> = (0..cols).map(|i| ((i % 17 + 1) as f32) / 16.).collect();
            let mut got = Vec::new();
            for dev in [&cpu, &gpu] {
                let x =
                    Tensor::from_vec(data.clone(), (rows + 2, cols), dev)?.narrow(0, 1, rows)?;
                let unweighted = crate::mhc::rms_rows(&x, 1e-20)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let weighted = candle_nn::ops::rms_norm(
                    &x,
                    &Tensor::from_vec(alpha.clone(), cols, dev)?,
                    1e-20,
                )?
                .flatten_all()?
                .to_vec1::<f32>()?;
                for r in 0..rows {
                    let row = &data[(r + 1) * cols..(r + 2) * cols];
                    let inv = (row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()
                        / cols as f64
                        + 1e-20)
                        .sqrt()
                        .recip();
                    for c in 0..cols {
                        rounded_reference(unweighted[r * cols + c], (row[c] as f64 * inv) as f32);
                        rounded_reference(
                            weighted[r * cols + c],
                            (row[c] as f64 * inv * alpha[c] as f64) as f32,
                        );
                    }
                }
                got.push((unweighted, weighted));
            }
            assert_eq!(got[0], got[1], "RMS width {cols}");
        }
        {
            use candle_core::quantized::{GgmlDType, QStorage, QTensor};
            for (dtype, k, n) in [
                (GgmlDType::Q8_0, 1280, 17),
                (GgmlDType::Q4K, 5120, 17),
                (GgmlDType::Q2K, 20480, 24),
            ] {
                let data: Vec<f32> = (0..n * k)
                    .map(|i| ((i * 31 % 251) as f32 - 125.) / 64.)
                    .collect();
                let q = QTensor::quantize(&Tensor::from_vec(data, (n, k), &cpu)?, dtype)?;
                let weights = q.dequantize(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
                let v = QTensor::new(QStorage::from_data(q.data()?, &gpu, dtype)?, (n, k))?;
                let qs = [qmatmul(q)?, qmatmul(v)?];
                for m in [1, 2, 16, 32, 33] {
                    let input: Vec<f32> = (0..(m + 2) * k)
                        .map(|i| ((i * 17 % 127) as f32 - 63.) / 32.)
                        .collect();
                    let mut want = vec![0f32; m * n];
                    for i in 0..m {
                        for j in 0..n {
                            want[i * n + j] = input[(i + 1) * k..(i + 2) * k]
                                .iter()
                                .zip(&weights[j * k..(j + 1) * k])
                                .map(|(&x, &w)| x as f64 * w as f64)
                                .sum::<f64>() as f32;
                        }
                    }
                    for (q, dev) in qs.iter().zip([&cpu, &gpu]) {
                        use candle_core::Module;
                        let x =
                            Tensor::from_vec(input.clone(), (m + 2, k), dev)?.narrow(0, 1, m)?;
                        let got = q.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
                        assert_eq!(got, want, "{dev:?} {dtype:?} M={m}");
                    }
                }
            }
        }
        {
            use candle_core::quantized::{GgmlDType, QStorage, QTensor};
            use candle_core::Module;
            let (groups, k, n) = (8, 4096, 8192);
            let rank = n / groups;
            let mut bytes = Vec::with_capacity(n * (k / 32) * 34);
            for block in 0..n * (k / 32) {
                bytes.extend_from_slice(&[0, 0x30]);
                for j in 0..32 {
                    bytes.push((((block * 31 + j * 17) % 255) as u8).wrapping_sub(127));
                }
            }
            let cpu_q = QTensor::new(
                QStorage::from_data(std::borrow::Cow::Borrowed(&bytes), &cpu, GgmlDType::Q8_0)?,
                (n, k),
            )?;
            let gpu_q = QTensor::new(
                QStorage::from_data(std::borrow::Cow::Borrowed(&bytes), &gpu, GgmlDType::Q8_0)?,
                (n, k),
            )?;
            let cpu_q = qmatmul(cpu_q)?;
            let gpu_q = qmatmul(gpu_q)?;
            let input: Vec<f32> = (0..groups * k)
                .map(|i| ((i * 17 % 127) as f32 - 63.) / 32.)
                .collect();
            let x = Tensor::from_vec(input.clone(), (groups, k), &cpu)?;
            let full = cpu_q.forward(&x)?.to_vec2::<f32>()?;
            let want: Vec<f32> = (0..groups)
                .flat_map(|g| full[g][g * rank..(g + 1) * rank].iter().copied())
                .collect();
            for (q, dev) in [(&cpu_q, &cpu), (&gpu_q, &gpu)] {
                let x = Tensor::from_vec(input.clone(), (groups, k), dev)?;
                let selected = q
                    .try_grouped_q8(&x, groups)?
                    .expect("B50 production selected-group shape");
                assert_eq!(
                    selected.flatten_all()?.to_vec1::<f32>()?,
                    want,
                    "{dev:?} selected model shape"
                );
            }
        }
        assert_eq!(
            candle_core::vulkan_backend::kernels::fallback_count(),
            before
        );
        Ok(())
    }
}
