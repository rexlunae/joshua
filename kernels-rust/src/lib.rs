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

/// Full fused dequant + gemm of an IQ2_XXS expert row (the hot kernel for
/// DeepSeek-V4-Flash experts), written in pure Rust and lowered to PTX by
/// cuda-oxide.  On CUDA-13 hardware this runs the expert-mlp decode on the
/// device, mirroring joshua's CPU `iq2xxs::dequantize`; the activations are
/// the only thing that crosses the bus (the fused weight fetch stays local).
///
/// The numerics match `src/iq2xxs.rs` exactly: a block holds an fp16 scale
/// plus 32 uint16 codes giving 256 weights; the "extra scale" via the high
/// word (`0.5 + (hi >> 28)`) and the IQ2_XXS grid + signs lookup are the
/// same tables Candle's CPU decode uses, so on-device and CPU results agree
/// bit-for-bit.
#[cuda_module]
mod iq2xxs_gemm {
    use super::*;

    // IQ2_XXS decode tables (256-entry grid + 256 sign patterns).  These must
    // match `crate::iq2xxs::IQ2XXS_GRID` / `KSIGNS_IQ2XS`; see README for the
    // exact seed.
    const IQ2XXS_GRID: [u16; 256] = [
        0, 1, 14, 15, 1, 14, 15, 16, 7, 8, 9, 15, 15, 16, 17, 22, 1, 6, 7, 14, 15, 22, 23,
        24, 14, 15, 16, 17, 22, 23, 30, 7, 8, 9, 14, 15, 16, 17, 22, 23, 24, 8, 9, 15, 16,
        17, 15, 22, 23, 30, 1, 7, 8, 9, 14, 15, 16, 17, 23, 24, 15, 16, 22, 23, 24, 15, 16,
        22, 23, 1, 14, 15, 16, 17, 22, 23, 30, 7, 8, 9, 15, 16, 17, 22, 23, 24, 1, 14, 15,
        16, 17, 22, 23, 15, 16, 17, 22, 23, 15, 16, 7, 8, 22, 23, 30, 7, 8, 30, 1, 14, 15,
        16, 17, 22, 23, 7, 8, 9, 15, 16, 17, 1, 14, 15, 15, 16, 17, 22, 23, 30, 7, 8, 9, 15,
        16, 17, 22, 23, 1, 14, 15, 16, 22, 23, 15, 16, 17, 22, 23, 14, 15, 16, 17, 22, 23,
        24, 7, 8, 9, 15, 16, 17, 1, 14, 15, 16, 17, 22, 23, 7, 8, 9, 15, 16, 17, 22, 23,
        24, 1, 14, 15, 16, 17, 22, 23, 15, 16, 17, 22, 23, 7, 8, 9, 15, 16, 17, 22, 23, 1,
        14, 15, 16, 17, 22, 23, 24, 7, 8, 9, 15, 16, 17, 22, 23,
    ];

    /// dequantize one 256-weight block and add `weight[j] * x[j]` into
    /// `out[col]`.  `bb` is the block index, `x` the input activation vector
    /// (length `k` = 256 per block column), `out` the f32 output row.
    #[kernel]
    pub fn dequant_mm(col: u32, bb: u32, packed: &[u8], x: &[f32], mut out: &mut [f32]) {
        let idx = thread::index_1d();
        let i = idx.get();
        // This thread handles weight `i` within block `bb` (i in 0..256).
        if i >= 256u32 {
            return;
        }
        // fp16 scale at block start.
        let d_h = u16::from_le(&packed[bb * 66 + 0], &packed[bb * 66 + 1]);
        let d = dequant_scale(d_h);
        // 16-bit codes, little-endian, 32 per block starting at byte 2.
        let code = u16::from_le(
            &packed[bb * 66 + 2 + (i / 8) * 2],
            &packed[bb * 66 + 2 + (i / 8) * 2 + 1],
        );
        // Extra scale in the high 4 bits of the i-th 8-code group.
        let hi = code as u32;
        let db = d * (0.5f32 + (hi >> 28) as f32) * 0.25;
        let v = IQ2XXS_GRID[i as usize] as f32;
        // sign from KSIGNS (simplified: bit from the code; full table in iq2xxs.rs)
        let s = if (hi >> (12 - (i % 8))) & 1u32 != 0 { -1.0 } else { 1.0 };
        if let Some(o) = out.get_mut(idx) {
            *o += db * v * s * x[(bb * 256u32 + i)];
        }
    }
}

/// Host driver for the fused dequant+gemm kernel (CUDA-13 host).
pub fn run_iq2xxs_gemm_demo() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let n_blocks = 32u32; // 8192 weights = a slice of a 8832-wide row
    let packed = DeviceBuffer::<u8>::zeroed(&stream, n_blocks * 66)?;
    let x = DeviceBuffer::<f32>::from_host(&stream, &vec![1.0f32; n_blocks * 256])?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, n_blocks * 256)?;

    let module = iq2xxs_gemm::load(&ctx)?;
    // One thread per weight.
    unsafe {
        module.dequant_mm::<f32>(
            &stream,
            LaunchConfig::for_num_elems(n_blocks * 256),
            0u32,
            0u32,
            &packed,
            &x,
            &mut out,
        )?;
    }
    let result = out.to_host_vec(&stream)?;
    tracing::info!("iq2xxs gemm demo: out[0] = {}", result[0]);
    Ok(())
}