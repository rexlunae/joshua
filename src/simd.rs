//! SIMD-accelerated f32 dot kernels (x86_64 AVX2/FMA, aarch64 NEON) with
//! runtime dispatch.
//!
//! llama.cpp reaches ~10x the throughput of joshua's original scalar matmuls
//! on the same CPU; the two levers this module pulls are the same ones
//! llama.cpp's `ggml-cpu` backend uses:
//!
//!   1. **SIMD vectorization** — the inner dot product runs 8 f32 lanes per
//!      instruction on x86_64 (`vfmadd231ps`), or 4 f32 lanes on aarch64
//!      (`fmla`), so one FMA replaces 16 (8) scalar multiply-adds.  On x86_64
//!      runtime detection via `is_x86_feature_detected!` keeps the crate
//!      portable: the kernels are only called when the CPU actually has
//!      AVX2+FMA, and every path has a scalar fallback.  On aarch64 NEON is
//!      part of the base architecture, so the kernels are always eligible.
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
