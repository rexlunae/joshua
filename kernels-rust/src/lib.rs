//! joshua's Rust-native CUDA kernels (via NVlabs/cuda-oxide).
//!
//! This is a **standalone, CUDA-13-only** crate: it defines custom CUDA kernels
//! in pure Rust (`#[kernel]`) compiled to PTX by cuda-oxide's rustc backend.
//! It is intentionally NOT wired into the main joshua crate build (which must
//! keep building against CUDA 12.4 for Maxwell/sm_52).  Build it only on a
//! CUDA-13 host with `cargo oxide`.
//!
//! STATUS: this is a scaffold/spec.  The public API surface follows the
//! cuda-oxide README (thread::index_1d, LaunchConfig::for_num_elems,
//! module.kernel::<T>(&stream, launch, ..., &input, &mut out), #[cuda_module],
//! #[kernel]); exact intrinsic names should be re-verified against
//! cuda-oxide-book on the CUDA-13 build host before use.

use cuda_device::{cuda_module, kernel, thread};
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};

// ---------------------------------------------------------------------------
// Kernel 1: elementwise dequant + accumulate — the shape joshua needs for its
// IQ2_XXS expert decode.  Rather than the C passed in candle-kernels, the
// dequant is written in pure Rust and lowered to PTX by cuda-oxide.
// ---------------------------------------------------------------------------
#[cuda_module]
mod iq2xxs {
    use super::*;

    /// 4-bit phase -> weight component grid (matches joshua's `iq2xxs` crate).
    // SAFETY: exactly 16 entries, indexed by a masked 0..15.
    pub const GRID: [i8; 16] = [0, 1, 14, 15, 1, 14, 15, 16, 7, 8, 9, 15, 1, 7, 14, 15];

    /// Dequantize one packed 2-bit weight and add it into `out`.
    ///
    /// One thread per weight.  Packed layout: 4 two-bit weights per byte.
    /// `scale` is the per-16-weight group scale; `base` is the offset into
    /// the caller's activation vector this weight lands at.
    #[kernel]
    pub fn dequant_add<T: Copy>(
        packed: &[u8],
        scales: &[f32],
        base: u32,
        mut out: &mut [f32],
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        // packed[i] holds weights 2i and 2i+1; pick i%4 within its byte.
        let byte = packed[i / 4];
        let nib = (byte >> ((i % 4) * 2)) & 0x03;
        let q = GRID[(i % 16)] as f32;
        let w = (q - 1.0) * scales[i / 16];
        if let Some(o) = out.get_mut(idx) {
            *o += w;
        }
    }
}

// ---------------------------------------------------------------------------
// Host driver: load the embedded PTX and launch `dequant_add`.
//
// `cargo oxide build` embeds the kernel and generates
// `iq2xxs::dequant_add::<f32>(&stream, launch, &packed, &scales, base, &mut
// out)`.  The raw LaunchConfig form below is unsafe by design; a
// `#[launch_contract(...)]` kernel would yield a checked safe launch.
// ---------------------------------------------------------------------------
pub fn run_dequant_demo() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // 16 weights (one group) for a trivial demonstration.
    let n_weights = 16u32;
    let packed = DeviceBuffer::<u8>::from_host(&stream, &[0u8; 4])?;
    let scales = DeviceBuffer::<f32>::from_host(&stream, &[1.0f32; 1])?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, n_weights)?;

    let module = iq2xxs::load(&ctx)?;
    unsafe {
        module.dequant_add::<f32>(
            &stream,
            LaunchConfig::for_num_elems(n_weights),
            &packed,
            &scales,
            0u32,
            &mut out,
        )?;
    }

    let result = out.to_host_vec(&stream)?;
    tracing::info!("iq2xxs dequant demo: out[0] = {}", result[0]);
    Ok(())
}

/// Full dequant+gemm of a full 8832-wide expert row would slot in here, using
/// the same `#[kernel]` shape — see the design note in README.md.
pub fn iq2xxs_decode<T: Copy>(x: &[f32], packed: &[u8], scales: &[f32], mut y: &mut [f32]) {
    // Placeholder for the fused kernel; wire-up happens when building on
    // CUDA-13 hardware.
    for i in 0..y.len() {
        y[i] = x[i] * scales[i / 16];
    }
}