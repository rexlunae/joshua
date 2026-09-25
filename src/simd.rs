//! SIMD-accelerated f32 dot kernels (x86_64 AVX-512 and AVX2/FMA, aarch64
//! NEON) with runtime dispatch.
//!
//! llama.cpp reaches ~10x the throughput of joshua's original scalar matmuls
//! on the same CPU; the two levers this module pulls are the same ones
//! llama.cpp's `ggml-cpu` backend uses:
//!
//!   1. **SIMD vectorization** — the inner dot product runs 8 f32 lanes per
//!      instruction on x86_64 (`vfmadd231ps`), or 4 f32 lanes on aarch64
//!      (`fmla`), so one FMA replaces 16 (8) scalar multiply-adds — 16 lanes
//!      with AVX-512 (`zmm` registers, masked loads for tails).  On x86_64
//!      runtime detection via `is_x86_feature_detected!` keeps the crate
//!      portable: the kernels are only called when the CPU actually has the
//!      ISA, and every path has a scalar fallback.  On aarch64 NEON is part
//!      of the base architecture, so the kernels are always eligible.  One
//!      [`SimdLevel`] is picked per process (see [`simd_level`]) and every
//!      dispatcher follows it.
//!   2. **Row-parallelism** — the output rows of a quantized matmul are
//!      independent, so the work is spread across the rayon global pool
//!      (which CachyOS-sized machines get from `available_parallelism`).
//!      Row-splitting is deterministic: each row is computed by exactly one
//!      task with identical per-element operations, so the result is
//!      bit-identical regardless of thread count.
//!
//! # Safety model for parallel writes
//!
//! The kernels write their output through [`DstPtr`], a `&mut [f32]`
//! downgraded to a raw pointer so the rayon closures (which only capture
//! shared references) can write disjoint elements.  Soundness rests on the
//! row-splitting contract: **row `r` of a `(m, k, n)` matmul writes exactly
//! `dst[i*n + r]` for `i in 0..m`** — a set that is disjoint between rows.
//! Each row is handled by a single task, so no element is ever written
//! twice or read-modify-written concurrently.  All callers of [`DstPtr`]
//! must uphold this contract.

use std::marker::PhantomData;
use std::sync::LazyLock;

use rayon::prelude::*;
use rayon::ThreadPool;

/// Whether the AVX2+FMA kernels should be used on this CPU.
///
/// All AVX2-capable x86_64 CPUs from Haswell onwards also implement FMA, but
/// the two are checked independently so the kernels are only ever reached
/// when both instruction sets are actually present.
pub fn avx2_fma_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Whether this CPU has the AVX-512 subset the 16-lane kernels use: F
/// (the 512-bit f32/i32 core), BW and VL (byte ops and 256-bit forms of
/// them), DQ (256-bit lane inserts), plus AVX2+FMA.  Every AVX-512 CPU
/// since Skylake-SP implements all of these.
pub fn avx512_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as has;
        avx2_fma_available()
            && has!("avx512f")
            && has!("avx512bw")
            && has!("avx512vl")
            && has!("avx512dq")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// A SIMD instruction set the kernels are written for, in increasing order
/// of width on x86_64.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    Scalar,
    /// aarch64, 4 f32 lanes.
    Neon,
    /// x86_64 AVX2 + FMA, 8 f32 lanes.
    Avx2,
    /// x86_64 AVX-512 (F/BW/VL/DQ), 16 f32 lanes.
    Avx512,
}

impl SimdLevel {
    /// The widest level this CPU supports.
    pub fn detect() -> Self {
        if avx512_available() {
            Self::Avx512
        } else if avx2_fma_available() {
            Self::Avx2
        } else if neon_available() {
            Self::Neon
        } else {
            Self::Scalar
        }
    }

    /// Whether this CPU can run the kernels of `self`.
    pub fn supported(self) -> bool {
        match self {
            Self::Scalar => true,
            Self::Neon => neon_available(),
            Self::Avx2 => avx2_fma_available(),
            Self::Avx512 => avx512_available(),
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "scalar" | "none" => Self::Scalar,
            "neon" => Self::Neon,
            "avx2" => Self::Avx2,
            "avx512" | "avx-512" => Self::Avx512,
            _ => return None,
        })
    }
}

static LEVEL: LazyLock<SimdLevel> = LazyLock::new(|| {
    let detected = SimdLevel::detect();
    match std::env::var("JOSHUA_SIMD") {
        Ok(v) => match SimdLevel::parse(&v) {
            Some(level) if level.supported() => level,
            Some(level) => {
                tracing::warn!("JOSHUA_SIMD={v}: {level:?} is not supported on this CPU; using {detected:?}");
                detected
            }
            None => {
                tracing::warn!("JOSHUA_SIMD={v} is not one of scalar/neon/avx2/avx512; using {detected:?}");
                detected
            }
        },
        Err(_) => detected,
    }
});

/// The SIMD level every kernel dispatcher uses: the widest one this CPU
/// supports, or the one `JOSHUA_SIMD` (`scalar` / `neon` / `avx2` /
/// `avx512`) names when the CPU supports it — for bisecting a numerical
/// difference or a slowdown to one kernel family.  Fixed for the process.
pub fn simd_level() -> SimdLevel {
    *LEVEL
}

/// Whether the NEON kernels should be used on this CPU.
///
/// NEON (AArch64 SIMD) is part of the base ARMv8 architecture, so on aarch64
/// this is always true and no runtime detection is needed.  Kept as a
/// function so the dispatch in `kquant_dot`/`quant_matmul` reads the same on
/// both architectures.
pub fn neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// The rayon pool used for row-parallel matmuls.
static POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("joshua-matmul-{i}"))
        .build()
        .expect("failed to build the joshua matmul thread pool")
});

/// Run `f(row)` for every row in `0..n`, in parallel across the global pool
/// when there is enough work to amortize the dispatch cost.
pub fn for_each_row(n: usize, f: impl Fn(usize) + Send + Sync) {
    let pool = &*POOL;
    let threads = pool.current_num_threads();
    run_chunks(pool, threads, n, |rows| {
        for &row in rows {
            f(row);
        }
    });
}

/// Run `f(rows)` for each contiguous run of row indices covering `0..n`.
///
/// Chunked variant for workers that want to allocate one scratch buffer per
/// task (e.g. a dequantized weight row) instead of per output row.
pub fn for_each_row_chunks(n: usize, f: impl Fn(&[usize]) + Send + Sync) {
    let pool = &*POOL;
    let threads = pool.current_num_threads();
    run_chunks(pool, threads, n, f);
}

/// Same as [`for_each_row`] but with an explicit thread count — used by the
/// determinism tests to prove the parallel path is bit-identical to serial.
#[cfg(test)]
pub fn for_each_row_with_threads(threads: usize, n: usize, f: impl Fn(usize) + Send + Sync) {
    if threads <= 1 || n < 8 {
        for row in 0..n {
            f(row);
        }
        return;
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("test thread pool");
    run_chunks(&pool, threads, n, |rows| {
        for &row in rows {
            f(row);
        }
    });
}

fn run_chunks(
    pool: &ThreadPool,
    threads: usize,
    n: usize,
    f: impl Fn(&[usize]) + Send + Sync,
) {
    if threads <= 1 || n < 8 {
        let rows: Vec<usize> = (0..n).collect();
        f(&rows);
        return;
    }
    // Up to ~4 chunks per thread keeps the pool busy while bounding the
    // per-task dispatch overhead for small n.
    let chunk = (n / (threads * 4)).max(1);
    pool.install(|| {
        (0..n).into_par_iter().chunks(chunk).for_each(|rows| f(&rows));
    });
}

/// A `&mut [f32]` downgraded to a raw pointer so parallel row workers can
/// write disjoint elements through shared references.
///
/// See the module-level safety model: callers must write only the elements
/// owned by the row they were assigned, and no element may be written by two
/// workers.
pub struct DstPtr<'a> {
    ptr: *mut f32,
    _marker: PhantomData<&'a mut [f32]>,
}

// SAFETY: `DstPtr` is only ever used to write disjoint elements (the
// row-splitting contract above), never to alias a `&mut [f32]` that is
// concurrently accessed through its original reference, and the lifetime
// guard prevents outliving the buffer it points into.
unsafe impl Send for DstPtr<'_> {}
unsafe impl Sync for DstPtr<'_> {}

impl<'a> DstPtr<'a> {
    pub fn new(dst: &'a mut [f32]) -> Self {
        Self {
            ptr: dst.as_mut_ptr(),
            _marker: PhantomData,
        }
    }

    /// Write `val` at `dst[idx]`.
    ///
    /// # Safety
    /// `idx` must be in bounds, and no other worker may be writing `idx`
    /// concurrently (the row-splitting contract).
    #[inline(always)]
    pub unsafe fn write(&self, idx: usize, val: f32) {
        *self.ptr.add(idx) = val;
    }

    /// Accumulate: `dst[idx] += val` (read-modify-write).
    ///
    /// # Safety
    /// Same as [`DstPtr::write`], plus: the element must already have been
    /// initialized (e.g. the buffer was zero-filled) and must not be touched
    /// by any other worker.
    #[inline(always)]
    pub unsafe fn add(&self, idx: usize, val: f32) {
        *self.ptr.add(idx) += val;
    }
}

/// Horizontal sum of an 8-lane f32 vector (x86_64).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ps(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

/// Horizontal sum of a 16-lane f32 vector (x86_64 AVX-512).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn hsum512(v: std::arch::x86_64::__m512) -> f32 {
    std::arch::x86_64::_mm512_reduce_add_ps(v)
}

// ─── AVX-512 fused-kernel scaffolding ─────────────────────────────────────
//
// Shared by the AVX-512 row kernels of `kquant_dot` and `iq2xxs`: a kernel
// decodes one block into 16-lane weight vectors and hands them to
// `fma_tile_avx512`; `row_tiles_avx512` walks the m-tiles and writes sums.

/// Activation rows per AVX-512 m-tile.
#[cfg(target_arch = "x86_64")]
pub(crate) const MTILE_AVX512: usize = 8;

/// Accumulate the dequantized weight vectors `w` (16 lanes each, covering
/// `w.len() * 16` consecutive columns from `col`) into each live row of the
/// m-tile.
///
/// # Safety
/// AVX-512 must be available; `lhs` must hold `col + 16 * w.len()` columns
/// for every row `m0..m0+mcnt`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn fma_tile_avx512(
    acc: &mut [std::arch::x86_64::__m512; MTILE_AVX512],
    lhs: &[f32],
    k: usize,
    m0: usize,
    mcnt: usize,
    col: usize,
    w: &[std::arch::x86_64::__m512],
) {
    use std::arch::x86_64::*;
    for (i, a) in acc.iter_mut().enumerate().take(mcnt) {
        let p = lhs.as_ptr().add((m0 + i) * k + col);
        for (j, wj) in w.iter().enumerate() {
            *a = _mm512_fmadd_ps(_mm512_loadu_ps(p.add(16 * j)), *wj, *a);
        }
    }
}

/// Run `per_block(block_index, block, acc, m0, mcnt)` over one weight row's
/// blocks for each m-tile, then write the tile's horizontal sums.
///
/// # Safety
/// AVX-512 must be available; row-disjointness as in the module's safety model.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn row_tiles_avx512<B>(
    m: usize,
    n: usize,
    blocks: &[B],
    blocks_per_row: usize,
    row: usize,
    dst: &DstPtr,
    mut per_block: impl FnMut(usize, &B, &mut [std::arch::x86_64::__m512; MTILE_AVX512], usize, usize),
) {
    use std::arch::x86_64::*;
    let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
    let mut m0 = 0;
    while m0 < m {
        let mcnt = (m - m0).min(MTILE_AVX512);
        let mut acc = [_mm512_setzero_ps(); MTILE_AVX512];
        for (b, block) in row_blocks.iter().enumerate() {
            per_block(b, block, &mut acc, m0, mcnt);
        }
        for (i, acc_i) in acc.iter().enumerate().take(mcnt) {
            // SAFETY: row `row` owns dst[i*n + row] for all i; disjoint per row.
            dst.write((m0 + i) * n + row, hsum512(*acc_i));
        }
        m0 += MTILE_AVX512;
    }
}

/// `a · b` over the common length, with the widest SIMD kernel
/// [`simd_level`] allows (FMA accumulation in lanes, then a horizontal sum).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    dot_fn()(a, b)
}

/// [`dot`]'s kernel for this process, resolved once so hot loops can hoist
/// the dispatch out of their inner iterations.
pub fn dot_fn() -> fn(&[f32], &[f32]) -> f32 {
    match simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => |a, b| {
            let n = a.len().min(b.len());
            // SAFETY: `simd_level` only reports levels this CPU supports.
            unsafe { dot_avx512(&a[..n], &b[..n]) }
        },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => |a, b| {
            let n = a.len().min(b.len());
            // SAFETY: as above.
            unsafe { dot_avx2(&a[..n], &b[..n]) }
        },
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => |a, b| {
            let n = a.len().min(b.len());
            // SAFETY: NEON is baseline on aarch64.
            unsafe { dot_neon(&a[..n], &b[..n]) }
        },
        _ => |a, b| a.iter().zip(b).map(|(x, y)| x * y).sum(),
    }
}

/// # Safety
/// The CPU must support AVX-512 (see [`avx512_available`]); `a.len() == b.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i)), _mm512_loadu_ps(b.as_ptr().add(i)), acc);
        i += 16;
    }
    if i < n {
        // Masked tail: the unused lanes load as zero.
        let mask: __mmask16 = (1u32 << (n - i)) as u16 - 1;
        let x = _mm512_maskz_loadu_ps(mask, a.as_ptr().add(i));
        let y = _mm512_maskz_loadu_ps(mask, b.as_ptr().add(i));
        acc = _mm512_fmadd_ps(x, y, acc);
    }
    _mm512_reduce_add_ps(acc)
}

/// # Safety
/// The CPU must support AVX2+FMA; `a.len() == b.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)), acc);
        i += 8;
    }
    let mut s = hsum256(acc);
    for j in i..n {
        s = a[j].mul_add(b[j], s);
    }
    s
}

/// # Safety
/// `a.len() == b.len()` (NEON is baseline on aarch64).
#[cfg(target_arch = "aarch64")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= n {
        acc = vfmaq_f32(acc, vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
        i += 4;
    }
    let mut s = hsum128(acc);
    for j in i..n {
        s = a[j].mul_add(b[j], s);
    }
    s
}

/// Horizontal sum of a 4-lane f32 vector (aarch64 NEON).
///
/// `vaddvq_f32` is a single A64 instruction (ADDV), so this needs no
/// shuffle chain like the AVX2 version.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn hsum128(v: std::arch::aarch64::float32x4_t) -> f32 {
    std::arch::aarch64::vaddvq_f32(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dot product agrees with a scalar f64 reference at every length
    /// (full vectors plus every tail size) on every level this CPU runs.
    #[test]
    fn dot_matches_scalar_reference_at_every_length() {
        for n in 0..70 {
            let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos() - 0.3).collect();
            let want: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let got = dot(&a, &b) as f64;
            assert!((got - want).abs() < 1e-4, "n={n}: {got} vs {want}");
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if avx512_available() {
                    assert!((dot_avx512(&a, &b) as f64 - want).abs() < 1e-4, "avx512 n={n}");
                }
                if avx2_fma_available() {
                    assert!((dot_avx2(&a, &b) as f64 - want).abs() < 1e-4, "avx2 n={n}");
                }
            }
        }
    }

    #[test]
    fn simd_level_is_supported_and_ordered() {
        let level = simd_level();
        assert!(level.supported());
        assert!(level <= SimdLevel::detect());
        assert_eq!(SimdLevel::parse("AVX512"), Some(SimdLevel::Avx512));
        assert_eq!(SimdLevel::parse("bogus"), None);
    }

    #[test]
    fn row_parallel_matches_serial_exactly() {
        let n = 1000;
        // Each row writes its own slot of a shared output, disjoint by row.
        let mut out = vec![0f32; n];
        let work = |dst: &DstPtr, row: usize| {
            let v = (row as f32) * 1.0000001 + 0.5;
            unsafe { dst.write(row, v) };
        };

        {
            let dst = DstPtr::new(&mut out);
            for_each_row_with_threads(1, n, |row| work(&dst, row));
        }
        let serial = out.clone();

        out.fill(0.0);
        let dst = DstPtr::new(&mut out);
        for_each_row_with_threads(8, n, |row| work(&dst, row));
        assert_eq!(serial, out, "parallel rows must match serial bit-for-bit");
    }
}
