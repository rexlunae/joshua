//! Threaded + SIMD-accelerated matmuls for candle's k-quant block types
//! (Q8_0, Q2_K, Q4_K, …).
//!
//! Candle's own `k_quants::matmul` is a scalar, single-threaded kernel, and
//! its block fields (`BlockQ8_0.d`, `BlockQ2K.qs`, …) are `pub(crate)` — so
//! the per-block dequantization has to stay candle's `GgmlType::to_float`.
//! What this module adds on top is exactly what llama.cpp's `ggml-cpu`
//! backend adds over a naive loop:
//!
//!   * the dequantized weight row is dot-producted against the activations
//!     with SIMD FMA instructions instead of scalar multiply-adds (8-lane
//!     AVX2/FMA on x86_64, 4-lane NEON on aarch64),
//!   * independent output rows are spread across the rayon pool.
//!
//! `T::to_float` (dequant) is reused untouched, so the numerics differ from
//! candle's scalar path only by FMA rounding (≤ 1 ulp), which the engine's
//! tolerance-based parity tests absorb.  On CPUs without a SIMD path, or for
//! shapes the SIMD path does not accept, the call is forwarded to
//! `k_quants::matmul` unchanged — bit-identical to joshua's previous
//! behavior.

use candle_core::quantized::k_quants::{
    self, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ6K, BlockQ8_0, BlockQ8K, BlockQ8_1,
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, GgmlType,
};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{bail, Device, DType, Result, Tensor};
use half::f16;

/// Decode raw GGUF tensor bytes to f32, for the streamed (non-mmap) load
/// path.  The mmap path borrows the quantized blocks and dequantizes inside
/// its matmul kernel with f32 activations; decoding weights to f32 here gives
/// the streamed path the *same* f32-activation semantics (candle's own
/// `k_quants::matmul` quantizes activations to Q8_0 before the dot, which
/// would make the two load paths diverge by ~1% per layer).
///
/// Returns `Ok(None)` for dtypes that should keep candle's own reader (F32,
/// and anything without a decoder here — e.g. the I32 routed-id tables, which
/// the engine reads through a separate path).
pub fn decode_raw_to_f32(dtype: u32, bytes: &[u8], elems: usize) -> Result<Option<Vec<f32>>> {
    use candle_core::quantized::k_quants::{
        BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ5_0, BlockQ5_1, BlockQ6K,
        BlockQ8_0, BlockQ8_1, BlockQ8K,
    };

    macro_rules! kquant {
        ($ty:ty) => {{
            let block_bytes = std::mem::size_of::<$ty>();
            let blck = <$ty as GgmlType>::BLCK_SIZE;
            if elems % blck != 0 || bytes.len() != elems / blck * block_bytes {
                bail!(
                    "dtype {dtype}: {} bytes for {elems} elements ({} per {blck}-block)",
                    bytes.len(),
                    block_bytes
                );
            }
            // SAFETY: `bytes` comes from a fresh heap read (malloc-aligned),
            // `block_bytes` divides it exactly, and every byte pattern is a
            // valid block (the same reinterpretation candle's own reader does).
            let blocks = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const $ty, bytes.len() / block_bytes)
            };
            let mut out = vec![0f32; elems];
            <$ty as GgmlType>::to_float(blocks, &mut out);
            out
        }};
    }

    let out = match dtype {
        0 => return Ok(None), // F32: candle's reader keeps it exact.
        1 => {
            if bytes.len() != elems * 2 {
                bail!("dtype F16: {} bytes for {elems} elements", bytes.len());
            }
            bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        }
        crate::iq2xxs::GGML_TYPE_IQ2_XXS => {
            let blocks = crate::iq2xxs::blocks_from_bytes(bytes)?;
            let mut out = vec![0f32; elems];
            crate::iq2xxs::dequantize(blocks, &mut out)?;
            out
        }
        crate::mxfp4::GGML_TYPE_MXFP4 => {
            let blocks = crate::mxfp4::blocks_from_bytes(bytes)?;
            let mut out = vec![0f32; elems];
            crate::mxfp4::dequantize(blocks, &mut out)?;
            out
        }
        2 => kquant!(BlockQ4_0),
        3 => kquant!(BlockQ4_1),
        6 => kquant!(BlockQ5_0),
        7 => kquant!(BlockQ5_1),
        8 => kquant!(BlockQ8_0),
        9 => kquant!(BlockQ8_1),
        10 => kquant!(BlockQ2K),
        11 => kquant!(BlockQ3K),
        12 => kquant!(BlockQ4K),
        13 => kquant!(BlockQ5K),
        14 => kquant!(BlockQ6K),
        15 => kquant!(BlockQ8K),
        _ => return Ok(None),
    };
    Ok(Some(out))
}

/// Fast CPU forward for a quantized weight that candle would otherwise run
/// through its own (scalar, single-threaded) kernel.
///
/// `qt` is the untouched quantized tensor behind a `QMatMul::QTensor` weight
/// and `xs` the f32 activations.  Returns:
///
/// * `None` — this input has no fast path here (non-CPU device, non-f32
///   activations, non-contiguous activations, or a dtype without a kernel);
///   the caller should fall back to candle unchanged,
/// * `Some(Ok(t))` — the matmul ran through Joshua's SIMD path,
///   `t = xs · qtᵀ` with `xs`'s leading dims flattened to rows,
/// * `Some(Err(e))` — the input was accepted but something failed; surface
///   the error rather than silently retrying.
///
/// Dtype routing: Q8_0 / Q2_K / Q4_K go to the fused dequant-in-registers
/// kernels ([`crate::kquant_dot::try_matmul_fused`], AVX2 on x86_64, NEON on
/// aarch64); every other k-quant goes to [`matmul_kquant`] — candle's
/// `to_float` dequant per row plus the SIMD dot and rayon row parallelism.
/// Both paths dot **f32 activations**, matching candle's dequantize-then-gemm
/// semantics up to FMA rounding (the same contract as the rest of this
/// module), so results differ from candle by rounding only.
pub fn try_fast_cpu_qmatmul(qt: &QTensor, xs: &Tensor) -> Option<Result<Tensor>> {
    if !matches!(xs.device(), Device::Cpu) || xs.dtype() != DType::F32 {
        return None;
    }
    let [n, k] = qt.shape().dims() else {
        return None;
    };
    let (k, n) = (*k, *n);
    let xs_dims = xs.dims();
    if xs_dims.len() < 2 || xs_dims[xs_dims.len() - 1] != k || !xs.is_contiguous() {
        return None;
    }
    let dtype = qt.dtype();
    let covered = matches!(
        dtype,
        GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q4K
            | GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_1
            | GgmlDType::Q3K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::Q8K
    );
    if !covered || k == 0 || n == 0 {
        return None;
    }
    let m: usize = xs_dims[..xs_dims.len() - 1].iter().product();

    // ── Dispatch policy (measured, see `examples/bench_neon.rs`) ────────────
    //
    // aarch64, when this build has candle's dotprod support compiled in
    // (`target-cpu=native` or similar):
    //
    // * **Q4_K, n%8==0** — candle's repacked `BlockQ4Kx8` SDOT kernel wins at
    //   every batch size (≈5× our fused NEON kernels at m=1 and m=32);
    //   defer.
    // * **small batches (decode)** — candle's NEON vec_dots win for the
    //   remaining k-quants too; defer.
    // * **larger batches (prefill)** — Joshua's row-parallel dequant+dot
    //   spreads across all cores and overtakes candle's single-threaded
    //   kernel (measured crossover below m≈8); take over.
    //
    // On x86_64 the AVX2 fused kernels stay first-choice (candle has no
    // comparable repacked path there).
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    if dtype == GgmlDType::Q4K && n.is_multiple_of(8) {
        return None;
    }
    // Small batches: candle wins across the board here, but only in the
    // same dotprod builds where that was measured.
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    if m < 8 {
        return None;
    }

    // Committed from here on: errors are returned, not swallowed.  (`ok()?`
    // on the flatten: a non-materializable view simply has no fast path.)
    let lhs = match xs.flatten_all().ok()?.to_vec1::<f32>() {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    let bytes = match qt.data() {
        Ok(b) => b,
        Err(e) => return Some(Err(e.into())),
    };

    let mut dst = vec![0f32; m * n];
    let mkn = (m, k, n);
    let ran = if matches!(dtype, GgmlDType::Q8_0 | GgmlDType::Q2K | GgmlDType::Q4K) {
        crate::kquant_dot::try_matmul_fused(dtype, mkn, &lhs, &bytes, &mut dst, true)
    } else {
        // Generic path for the remaining k-quants: reinterpret the bytes as
        // candle blocks (same cast candle's own reader performs — every byte
        // pattern is a valid block) and run the dequant+SIMD-dot matmul.
        macro_rules! generic_kquant {
            ($ty:ty) => {{
                let block_bytes = std::mem::size_of::<$ty>();
                if !bytes.len().is_multiple_of(block_bytes) {
                    false
                } else {
                    let blocks = unsafe {
                        std::slice::from_raw_parts(
                            bytes.as_ptr() as *const $ty,
                            bytes.len() / block_bytes,
                        )
                    };
                    matmul_kquant::<$ty>(mkn, &lhs, blocks, &mut dst).is_ok()
                }
            }};
        }
        match dtype {
            GgmlDType::Q4_0 => generic_kquant!(BlockQ4_0),
            GgmlDType::Q4_1 => generic_kquant!(BlockQ4_1),
            GgmlDType::Q5_0 => generic_kquant!(BlockQ5_0),
            GgmlDType::Q5_1 => generic_kquant!(BlockQ5_1),
            GgmlDType::Q8_1 => generic_kquant!(BlockQ8_1),
            GgmlDType::Q3K => generic_kquant!(BlockQ3K),
            GgmlDType::Q5K => generic_kquant!(BlockQ5K),
            GgmlDType::Q6K => generic_kquant!(BlockQ6K),
            GgmlDType::Q8K => generic_kquant!(BlockQ8K),
            _ => false,
        }
    };
    if !ran {
        return None;
    }

    let mut out_shape = xs_dims[..xs_dims.len() - 1].to_vec();
    out_shape.push(n);
    Some(Tensor::from_vec(dst, out_shape, &Device::Cpu))
}

/// `dst[m, n] = lhs[m, k] · rhs[n, k]ᵀ` over candle k-quant blocks.
///
/// Contract and result layout match `k_quants::matmul` exactly: `rhs` holds
/// `n` weight rows of `k / T::BLCK_SIZE` blocks each, and `dst[i*n + j]` is
/// the dot of activation row `i` with weight row `j`.
pub fn matmul_kquant<T: GgmlType>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    blocks: &[T],
    dst: &mut [f32],
) -> Result<()> {
    let blocks_per_row = validate(m, k, n, lhs, blocks, dst)?;
    matmul_kquant_impl((m, k, n), blocks_per_row, lhs, blocks, dst, true)
}

/// Same as [`matmul_kquant`] with a serial row loop — used by the
/// determinism tests.
#[cfg(test)]
pub fn matmul_kquant_serial<T: GgmlType>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    blocks: &[T],
    dst: &mut [f32],
) -> Result<()> {
    let blocks_per_row = validate(m, k, n, lhs, blocks, dst)?;
    matmul_kquant_impl((m, k, n), blocks_per_row, lhs, blocks, dst, false)
}

fn validate<T: GgmlType>(
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f32],
    blocks: &[T],
    dst: &mut [f32],
) -> Result<usize> {
    if !k.is_multiple_of(T::BLCK_SIZE) {
        bail!(
            "quantized matmul: k={k} is not a multiple of the block size {}",
            T::BLCK_SIZE
        );
    }
    let blocks_per_row = k / T::BLCK_SIZE;
    if blocks.len() != n * blocks_per_row {
        bail!(
            "quantized matmul: rhs holds {} blocks, expected {}",
            blocks.len(),
            n * blocks_per_row
        );
    }
    if lhs.len() != m * k || dst.len() != m * n {
        bail!(
            "quantized matmul: lhs/dst sized {}/{} expected {}/{}",
            lhs.len(),
            dst.len(),
            m * k,
            m * n
        );
    }
    Ok(blocks_per_row)
}

fn matmul_kquant_impl<T: GgmlType>(
    mkn: (usize, usize, usize),
    blocks_per_row: usize,
    lhs: &[f32],
    blocks: &[T],
    dst: &mut [f32],
    parallel: bool,
) -> Result<()> {
    let (m, k, n) = mkn;
    if m == 0 || n == 0 {
        return Ok(());
    }
    if try_simd_kquant(mkn, blocks_per_row, lhs, blocks, dst, parallel) {
        return Ok(());
    }
    // Scalar fallback: candle's own kernel, exactly as before (bit-identical
    // to joshua's prior behavior on non-SIMD CPUs and unusual shapes).
    k_quants::matmul((m, k, n), lhs, blocks, dst)
}

/// SIMD fast path for [`matmul_kquant_impl`]: for the block types the
/// model actually uses (Q8_0, Q2_K, Q4_K) this delegates to the *fused*
/// kernels in [`crate::kquant_dot`] — the quantized blocks are decoded
/// inside the dot in SIMD registers, so no f32 weight row is ever
/// materialized (the same design as the IQ2_XXS path).  Other k-quant types
/// fall back to the generic path below: dequantize each weight row once
/// into an f32 scratch, then dot it against every activation row with
/// SIMD FMA.  Returns `true` if a kernel ran.
///
/// On x86_64 this uses AVX2/FMA when available; on aarch64 it uses NEON
/// (always available).  On other targets it declines.
fn try_simd_kquant<T: GgmlType>(
    mkn: (usize, usize, usize),
    blocks_per_row: usize,
    lhs: &[f32],
    blocks: &[T],
    dst: &mut [f32],
    parallel: bool,
) -> bool {
    let (m, k, n) = mkn;
    #[cfg(target_arch = "x86_64")]
    let fused = crate::simd::avx2_fma_available();
    #[cfg(target_arch = "aarch64")]
    let fused = crate::simd::neon_available();
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let fused = false;
    if !(fused && k.is_multiple_of(8)) {
        return false;
    }
    // SAFETY: `blocks` is `n * blocks_per_row` T-blocks; reinterpreting it
    // as bytes is always sound (the fused kernels re-cast to their own
    // packed block structs, whose sizes are asserted to match candle's).
    let block_bytes = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            std::mem::size_of_val(blocks),
        )
    };
    if crate::kquant_dot::try_matmul_fused(T::DTYPE, mkn, lhs, block_bytes, dst, parallel) {
        return true;
    }
    let dst_ptr = crate::simd::DstPtr::new(dst);
    let worker = move |rows: &[usize]| {
        // One dequantization scratch per task, reused across the task's
        // rows (the weight row is fully dequantized before the SIMD dot).
        let mut wrow = vec![0f32; k];
        for &row in rows {
            T::to_float(&blocks[row * blocks_per_row..(row + 1) * blocks_per_row], &mut wrow);
            // SAFETY: `fused` (the matching SIMD availability check) was
            // verified above, so the target_feature kernel may run on this
            // CPU; row `row` writes exactly dst[i*n + row] for i in 0..m,
            // disjoint from every other row (see `crate::simd`).
            unsafe { dot_row_simd(m, k, n, lhs, &wrow, row, &dst_ptr) }
        }
    };
    if parallel {
        crate::simd::for_each_row_chunks(n, worker);
    } else {
        let rows: Vec<usize> = (0..n).collect();
        worker(&rows);
    }
    true
}

/// SIMD dot of one dequantized weight row against all m activation rows,
/// 4 lhs rows at a time.  Dispatches to the CPU's SIMD ISA.
///
/// # Safety
/// The caller must have verified the matching SIMD availability check and
/// `k % 8 == 0` (the loop consumes `k` in 8-lane chunks with no tail), and
/// must only hand out disjoint rows per [`crate::simd::DstPtr`].
unsafe fn dot_row_simd(
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f32],
    wrow: &[f32],
    row: usize,
    dst: &crate::simd::DstPtr,
) {
    #[cfg(target_arch = "x86_64")]
    {
        dot_row_avx2(m, k, n, lhs, wrow, row, dst)
    }
    #[cfg(target_arch = "aarch64")]
    {
        dot_row_neon(m, k, n, lhs, wrow, row, dst)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (m, k, n, lhs, wrow, row, dst);
        unreachable!("dot_row_simd is only called after a SIMD availability check")
    }
}

/// AVX2/FMA dot of one dequantized weight row against all m activation rows,
/// 4 lhs rows at a time.
///
/// # Safety
/// The caller must have verified `avx2_fma_available()` and `k % 8 == 0`
/// (the loop consumes `k` in 8-lane chunks with no tail), and must only
/// hand out disjoint rows per [`crate::simd::DstPtr`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)] // (m, k, n) is the matmul shape; kept flat for the hot loop
unsafe fn dot_row_avx2(
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f32],
    wrow: &[f32],
    row: usize,
    dst: &crate::simd::DstPtr,
) {
    use std::arch::x86_64::*;

    const MTILE: usize = 4;
    let mut m0 = 0;
    while m0 < m {
        let mcnt = (m - m0).min(MTILE);
        let mut acc = [_mm256_setzero_ps(); MTILE];
        let mut c = 0;
        while c + 8 <= k {
            let wv = _mm256_loadu_ps(wrow[c..c + 8].as_ptr());
            for i in 0..mcnt {
                let a = _mm256_loadu_ps(lhs[(m0 + i) * k + c..].as_ptr());
                acc[i] = _mm256_fmadd_ps(a, wv, acc[i]);
            }
            c += 8;
        }
        // `mcnt` can be < MTILE at the tail; iterate only the live lanes.
        for (i, acc_i) in acc.iter().enumerate().take(mcnt) {
            // SAFETY: row `row` owns dst[i*n + row] for all i; disjoint per
            // row (see `crate::simd`).
            dst.write((m0 + i) * n + row, crate::simd::hsum256(*acc_i));
        }
        m0 += MTILE;
    }
}

/// NEON dot of one dequantized weight row against all m activation rows,
/// 4 lhs rows at a time.
///
/// # Safety
/// The caller must have verified `neon_available()` and `k % 8 == 0`
/// (the loop consumes `k` in 4-lane chunks with no tail), and must only
/// hand out disjoint rows per [`crate::simd::DstPtr`].
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)] // (m, k, n) is the matmul shape; kept flat for the hot loop
unsafe fn dot_row_neon(
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f32],
    wrow: &[f32],
    row: usize,
    dst: &crate::simd::DstPtr,
) {
    use std::arch::aarch64::*;

    const MTILE: usize = 4;
    let mut m0 = 0;
    while m0 < m {
        let mcnt = (m - m0).min(MTILE);
        let mut acc = [vdupq_n_f32(0.0); MTILE];
        let mut c = 0;
        while c + 8 <= k {
            let w0 = vld1q_f32(wrow[c..c + 4].as_ptr());
            let w1 = vld1q_f32(wrow[c + 4..c + 8].as_ptr());
            for i in 0..mcnt {
                let a = lhs[(m0 + i) * k + c..].as_ptr();
                acc[i] = vfmaq_f32(acc[i], vld1q_f32(a), w0);
                acc[i] = vfmaq_f32(acc[i], vld1q_f32(a.add(4)), w1);
            }
            c += 8;
        }
        // `mcnt` can be < MTILE at the tail; iterate only the live lanes.
        for (i, acc_i) in acc.iter().enumerate().take(mcnt) {
            // SAFETY: row `row` owns dst[i*n + row] for all i; disjoint per
            // row (see `crate::simd`).
            dst.write((m0 + i) * n + row, crate::simd::hsum128(*acc_i));
        }
        m0 += MTILE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::k_quants::{BlockQ2K, BlockQ4K, BlockQ8_0};
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Tensor};

    /// Quantize a random `[n, k]` f32 tensor with `dtype` and return a
    /// borrowed view of its blocks.
    fn quantized_blocks<T: GgmlType>(n: usize, k: usize, dtype: GgmlDType) -> Vec<T> {
        let data: Vec<f32> = (0..n * k)
            .map(|i| ((((i as u64) * 2654435761) % 100000) as f32 / 1000.0) - 50.0)
            .collect();
        let t = Tensor::from_vec(data, (n, k), &Device::Cpu).unwrap();
        let qt = QTensor::quantize(&t, dtype).unwrap();
        // `data()` exposes the raw block bytes of candle's quantized storage,
        // which is a `Vec<T>` allocation — T-aligned by construction, so the
        // reinterpret below is sound (every byte pattern is a valid block).
        let bytes = qt.data().unwrap();
        assert!(bytes.len().is_multiple_of(std::mem::size_of::<T>()));
        let ptr = bytes.as_ptr() as *const T;
        let len = bytes.len() / std::mem::size_of::<T>();
        let blocks = unsafe { std::slice::from_raw_parts(ptr, len) };
        blocks.to_vec()
    }

    fn run_case<T: GgmlType>(dtype: GgmlDType, m: usize, k: usize, n: usize) {
        let blocks = quantized_blocks::<T>(n, k, dtype);
        // Activation rows, overlapping ranges so no accidental zero rows.
        let lhs: Vec<f32> = (0..m * k)
            .map(|i| (((i * 40503) % 1000) as f32 / 100.0) - 5.0)
            .collect();

        let mut fast = vec![0f32; m * n];
        matmul_kquant::<T>((m, k, n), &lhs, &blocks, &mut fast).unwrap();

        // Reference: a true f32 GEMM over the dequantised weights.  (Candle's
        // own `k_quants::matmul` quantises the activations to Q8_0 before the
        // dot, so it is NOT the right reference for this kernel — joshua dots
        // f32 activations, the same semantics as its IQ2_XXS path.)
        //
        // Accumulation order: when a SIMD fused kernel is active, replicate
        // its exact lane order with `mul_add` (single rounding) so the
        // comparison is pinned at FMA-rounding level.  AVX2 kernels are
        // 8-lane; the NEON kernels accumulate in a float32x4_t (4 lanes) for
        // every dtype.  Without SIMD the scalar kernel accumulates
        // sequentially with two roundings per term.
        let mut w = vec![0f32; n * k];
        T::to_float(&blocks, &mut w);
        let lanes: usize = if crate::simd::avx2_fma_available() || crate::simd::neon_available()
        {
            if cfg!(target_arch = "aarch64") {
                4
            } else {
                8
            }
        } else {
            1
        };
        for (i, fastv) in fast.iter().enumerate() {
            let (ii, jj) = (i / n, i % n);
            let mut acc = 0f32;
            if lanes == 1 {
                for (kk, wv) in w[jj * k..(jj + 1) * k].iter().enumerate() {
                    acc += lhs[ii * k + kk] * wv;
                }
            } else {
                let mut accv = [0f32; 8];
                let mut c = 0;
                while c + lanes <= k {
                    for l in 0..lanes {
                        accv[l] = lhs[ii * k + c + l].mul_add(w[jj * k + c + l], accv[l]);
                    }
                    c += lanes;
                }
                acc = accv[..lanes].iter().sum::<f32>();
            }
            let tol = 1e-4 * acc.abs().max(1.0);
            assert!(
                (fastv - acc).abs() <= tol,
                "{dtype:?}: dst[{i}] fast={} f32-gemm={} (tol {tol})",
                fast[i],
                acc
            );
        }
    }

    /// Smoke test against candle's own kernel: both dequantise the same
    /// weights, so a decode bug would blow past this tolerance even though
    /// candle's activation quantisation (Q8_0) makes the two disagree by
    /// ~0.1-0.5%.
    #[test]
    fn still_close_to_candle_quantised_activation_path() {
        let (m, k, n) = (2, 256, 9);
        let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01 - 1.0).collect();
        macro_rules! check {
            ($ty:ty, $dt:expr) => {{
                let blocks = quantized_blocks::<$ty>(n, k, $dt);
                let mut fast = vec![0f32; m * n];
                matmul_kquant::<$ty>((m, k, n), &lhs, &blocks, &mut fast).unwrap();
                let mut candle = vec![0f32; m * n];
                k_quants::matmul((m, k, n), &lhs, &blocks, &mut candle).unwrap();
                for (i, v) in fast.iter().enumerate() {
                    // Candle's activation quantisation shifts individual dots
                    // by up to ~5% on adversarial data; 1e-1 still catches any
                    // decode/scale bug (those blow past 2x, like the original
                    // IQ2_XXS block-offset bug did).
                    let tol = 1e-1 * candle[i].abs().max(1.0);
                    assert!(
                        (v - candle[i]).abs() <= tol,
                        "{:?}: dst[{i}] fast={} candle={} (tol {tol})",
                        $dt,
                        fast[i],
                        candle[i]
                    );
                }
            }};
        }
        check!(BlockQ8_0, GgmlDType::Q8_0);
        check!(BlockQ2K, GgmlDType::Q2K);
        check!(BlockQ4K, GgmlDType::Q4K);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q8_0_matches_candle_scalar() {
        run_case::<BlockQ8_0>(GgmlDType::Q8_0, 3, 512, 17);
        run_case::<BlockQ8_0>(GgmlDType::Q8_0, 1, 256, 64);
        run_case::<BlockQ8_0>(GgmlDType::Q8_0, 5, 128, 9);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q2k_matches_candle_scalar() {
        run_case::<BlockQ2K>(GgmlDType::Q2K, 3, 512, 11);
        run_case::<BlockQ2K>(GgmlDType::Q2K, 1, 256, 40);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q4k_matches_candle_scalar() {
        run_case::<BlockQ4K>(GgmlDType::Q4K, 3, 512, 13);
        run_case::<BlockQ4K>(GgmlDType::Q4K, 4, 768, 25);
    }

    /// The same correctness checks through the NEON path (aarch64).
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_matches_candle_scalar() {
        run_case::<BlockQ8_0>(GgmlDType::Q8_0, 3, 512, 17);
        run_case::<BlockQ8_0>(GgmlDType::Q8_0, 1, 256, 64);
        run_case::<BlockQ8_0>(GgmlDType::Q8_0, 5, 128, 9);
        run_case::<BlockQ2K>(GgmlDType::Q2K, 3, 512, 11);
        run_case::<BlockQ2K>(GgmlDType::Q2K, 1, 256, 40);
        run_case::<BlockQ4K>(GgmlDType::Q4K, 3, 512, 13);
        run_case::<BlockQ4K>(GgmlDType::Q4K, 4, 768, 25);
        // The real model's shapes.
        run_case::<BlockQ2K>(GgmlDType::Q2K, 1, 7680, 2048);
        run_case::<BlockQ4K>(GgmlDType::Q4K, 1, 2048, 7680);
        run_case::<BlockQ8_0>(GgmlDType::Q8_0, 32, 4096, 4096);
    }

    /// `decode_raw_to_f32` must decode every K-quant GGUF id (10..=15) from
    /// raw file bytes: the streamed loader sizes reads through
    /// `gguf_ext::type_size_bytes`, and this decoder must agree with candle's
    /// block layouts element-for-element.
    #[test]
    fn decode_raw_to_f32_handles_all_k_quants() {
        use candle_core::quantized::k_quants::{
            BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K,
        };
        macro_rules! check {
            ($id:expr, $ty:ty) => {{
                let n = 3usize;
                let k = 256usize; // QK_K
                let blocks = quantized_blocks::<$ty>(n, k, <$ty as GgmlType>::DTYPE);
                let elems = n * k;
                let block_bytes = std::mem::size_of::<$ty>();
                let mut bytes = Vec::with_capacity(blocks.len() * block_bytes);
                for b in &blocks {
                    let ptr = b as *const $ty as *const u8;
                    bytes.extend_from_slice(unsafe {
                        std::slice::from_raw_parts(ptr, block_bytes)
                    });
                }
                let decoded = decode_raw_to_f32($id, &bytes, elems)
                    .unwrap()
                    .unwrap_or_else(|| panic!("dtype {} must decode", $id));
                let mut w = vec![0f32; elems];
                <$ty as GgmlType>::to_float(&blocks, &mut w);
                for (i, (x, y)) in decoded.iter().zip(w.iter()).enumerate() {
                    let tol = 1e-6 * y.abs().max(1.0);
                    assert!(
                        (x - y).abs() <= tol,
                        "dtype {} elem {i}: decoded={x} to_float={y} (tol {tol})",
                        $id
                    );
                }
            }};
        }
        check!(10, BlockQ2K);
        check!(11, BlockQ3K);
        check!(12, BlockQ4K);
        check!(13, BlockQ5K);
        check!(14, BlockQ6K);
        check!(15, BlockQ8K);
    }

    /// Parallel and serial execution must agree bit-for-bit (rows are
    /// computed independently with identical operations).
    #[test]
    fn threaded_matches_serial_bit_exact() {
        let blocks = quantized_blocks::<BlockQ8_0>(24, 256, GgmlDType::Q8_0);
        let lhs: Vec<f32> = (0..3 * 256).map(|i| (i as f32) * 0.01 - 1.0).collect();

        let mut par = vec![0f32; 3 * 24];
        matmul_kquant::<BlockQ8_0>((3, 256, 24), &lhs, &blocks, &mut par).unwrap();

        let mut ser = vec![0f32; 3 * 24];
        matmul_kquant_serial::<BlockQ8_0>((3, 256, 24), &lhs, &blocks, &mut ser).unwrap();

        assert_eq!(par, ser, "parallel and serial matmul must be bit-identical");
    }

    #[test]
    fn rejects_bad_shapes() {
        let blocks = quantized_blocks::<BlockQ8_0>(2, 256, GgmlDType::Q8_0);
        let lhs = vec![0f32; 256];
        let mut dst = vec![0f32; 2];
        // k not a multiple of the block size.
        assert!(matmul_kquant::<BlockQ8_0>((1, 100, 2), &lhs, &blocks, &mut dst).is_err());
        // Wrong block count for (m, k, n).
        assert!(matmul_kquant::<BlockQ8_0>((1, 256, 4), &lhs, &blocks, &mut dst).is_err());
        // Mismatched lhs/dst lengths.
        assert!(matmul_kquant::<BlockQ8_0>((1, 256, 2), &[0.0], &blocks, &mut dst).is_err());
    }

    // ── try_fast_cpu_qmatmul (the QMatMul::QTensor forward hook) ────────────

    /// Build a quantized `[n, k]` QTensor and matching activations.
    fn qtensor_and_xs(
        n: usize,
        k: usize,
        dtype: GgmlDType,
        m: usize,
    ) -> (candle_core::quantized::QTensor, Tensor) {
        let data: Vec<f32> = (0..n * k)
            .map(|i| ((((i as u64) * 2654435761) % 100000) as f32 / 1000.0) - 50.0)
            .collect();
        let t = Tensor::from_vec(data, (n, k), &Device::Cpu).unwrap();
        let qt = candle_core::quantized::QTensor::quantize(&t, dtype).unwrap();
        let xs: Vec<f32> = (0..m * k).map(|i| (((i * 40503) % 997) as f32 / 317.0) - 1.5).collect();
        let xs = Tensor::from_vec(xs, (m, k), &Device::Cpu).unwrap();
        (qt, xs)
    }

    /// The fast path must reproduce candle's dequantize-then-gemm for every
    /// dtype it accepts — including Q6_K/Q5K etc. that only have the generic
    /// dequant+NEON-dot route.  Tolerances absorb accumulation-order and FMA
    /// rounding only (same contract as `run_case` above).
    ///
    /// Policy-aware: under the measured dispatch policy some inputs are
    /// deferred to candle (`None`); when that happens we just assert the
    /// deferral is the documented one instead of checking numerics.
    #[test]
    fn fast_cpu_qmatmul_matches_candle_dequantized_gemm() {
        let dev = Device::Cpu;
        // On a dotprod build Q4K defers to candle at every m; other
        // k-quants defer below the parallel-amortization threshold.
        let deferred = |dtype: GgmlDType, m: usize, n: usize| -> bool {
            #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
            {
                if dtype == GgmlDType::Q4K && n.is_multiple_of(8) {
                    return true;
                }
                m < 8
            }
            #[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
            {
                let _ = (dtype, m, n);
                false
            }
        };
        for dtype in [
            GgmlDType::Q8_0,
            GgmlDType::Q2K,
            GgmlDType::Q4K,
            GgmlDType::Q6K,
            GgmlDType::Q5K,
            GgmlDType::Q4_0,
        ] {
            let (qt, xs) = qtensor_and_xs(48, 256, dtype, 3);
            match try_fast_cpu_qmatmul(&qt, &xs) {
                None => assert!(deferred(dtype, 3, 48), "{dtype:?}: unexpected deferral"),
                Some(out) => {
                    let out = out.unwrap();
                    assert_eq!(out.dims(), &[3, 48]);
                    let w = qt.dequantize(&dev).unwrap(); // [n, k] f32
                    let reference = xs.matmul(&w.t().unwrap()).unwrap();
                    let fast = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    let refr =
                        reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    for (i, (a, b)) in fast.iter().zip(refr.iter()).enumerate() {
                        let tol = 2e-2 * b.abs().max(1.0);
                        assert!(
                            (a - b).abs() <= tol,
                            "{dtype:?}: dst[{i}] fast={a} candle-gemm={b} (tol {tol})"
                        );
                    }
                }
            }
        }
    }

    /// Real model shapes from the Qwen3-Coder-30B-A3B Q4_K_M GGUF: expert
    /// gate/up [768, 2048] Q4_K, early-layer ffn_down/attn_v [2048, 768] /
    /// [512, 2048] Q6_K — in the prefill regime (m=16), where the parallel
    /// path is supposed to run.
    #[test]
    fn fast_cpu_qmatmul_at_model_shapes() {
        for (dtype, n, k) in [
            (GgmlDType::Q4K, 768usize, 2048usize),
            (GgmlDType::Q6K, 2048, 768),
            (GgmlDType::Q6K, 512, 2048),
            (GgmlDType::Q4K, 4096, 2048),
        ] {
            let (qt, xs) = qtensor_and_xs(n, k, dtype, 16);
            let res = try_fast_cpu_qmatmul(&qt, &xs);
            #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
            if dtype == GgmlDType::Q4K && n.is_multiple_of(8) {
                assert!(
                    res.is_none(),
                    "{dtype:?}: Q4K defers to candle's repacked path"
                );
                continue;
            }
            let out = res
                .unwrap_or_else(|| panic!("{dtype:?} must be covered"))
                .unwrap();
            assert_eq!(out.dims(), &[16, n]);
        }
    }

    /// Inputs without a fast path must return `None` (caller falls back to
    /// candle), and 3-D activations must flatten correctly.
    #[test]
    fn fast_cpu_qmatmul_fallbacks_and_flattening() {
        // F32 weights are never routed here (candle keeps them exact).
        let data = vec![0.5f32; 4 * 256];
        let bytes_f32: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let storage = candle_core::quantized::QStorage::from_data(
            std::borrow::Cow::Owned(bytes_f32),
            &Device::Cpu,
            GgmlDType::F32,
        )
        .unwrap();
        let qt_f32 = candle_core::quantized::QTensor::new(storage, 4 * 256).unwrap();
        let xs = Tensor::zeros((2, 1024), DType::F32, &Device::Cpu).unwrap();
        assert!(try_fast_cpu_qmatmul(&qt_f32, &xs).is_none());

        // Non-f32 activations fall back.
        let (qt, _) = qtensor_and_xs(4, 256, GgmlDType::Q4K, 2);
        let xs_h = Tensor::zeros((2, 256), DType::F16, &Device::Cpu).unwrap();
        assert!(try_fast_cpu_qmatmul(&qt, &xs_h).is_none());

        // Non-contiguous activations fall back.
        let wide = Tensor::from_vec(
            (0..2 * 512).map(|i| i as f32 * 0.01).collect::<Vec<_>>(),
            (2, 512),
            &Device::Cpu,
        )
        .unwrap();
        let xs_nc = wide.narrow(1, 128, 256).unwrap(); // strided view
        assert!(!xs_nc.is_contiguous());
        assert!(try_fast_cpu_qmatmul(&qt, &xs_nc).is_none());

        // Last-dim mismatch falls back.
        let xs_bad = Tensor::zeros((2, 128), DType::F32, &Device::Cpu).unwrap();
        assert!(try_fast_cpu_qmatmul(&qt, &xs_bad).is_none());

        // 3-D contiguous activations flatten to rows and reshape back
        // (m=12: above the parallel-amortization threshold; Q6_K because
        // Q4_K always defers to candle on dotprod builds).
        let (qt8, xs_flat) = qtensor_and_xs(8, 256, GgmlDType::Q6K, 12);
        let xs_3d = xs_flat.reshape((2, 6, 256)).unwrap();
        let out = try_fast_cpu_qmatmul(&qt8, &xs_3d)
            .expect("covered dtype")
            .unwrap();
        assert_eq!(out.dims(), &[2, 6, 8]);
    }
}
