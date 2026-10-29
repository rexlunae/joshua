//! Native NVFP4 (`GGML_TYPE_NVFP4`, id 40) device GEMV.
//!
//! Nemotron 3.5's quantized checkpoints are NVFP4, and the model's documented
//! throughput advantage comes from moving FP4 weight bytes to the accelerator
//! and decoding them *there*.  Before this module, [`crate::mxfp4::BlockNvfp4`]
//! was decoded on the host only: `grep -ri nvfp4 vendor/` was empty, so every
//! NVFP4 matmul either dequantized to f32/f16 at load (forfeiting the memory
//! win) or moved f32 weights across the bus on every step (forfeiting the
//! bandwidth win).  This module adds the missing device path, keeping the host
//! decoder as the reference it is checked against.
//!
//! # Why a `CustomOp2` and not a `GgmlDType`
//!
//! `candle_core::quantized::GgmlDType` stops at `Iq2Xxs`, and its `from_u32`
//! *rejects* type 40, so candle cannot represent an NVFP4 tensor as a
//! quantized storage.  Expanding the vendored enum would touch every existing
//! quantized path (`QMatMul`, the OpenCL/Vulkan `QStorage` impls, mmap
//! zero-copy) for one new format.  Instead the raw 36-byte blocks stay in a
//! `u8` tensor and the matmul is a [`candle_core::CustomOp2`] over
//! `(weights, activations)` — the same routing [`crate::raw_block`] already
//! uses on the host.  `CustomOp2` supplies `cpu_fwd` (both fallback and
//! reference) plus `opencl_fwd` / `vulkan_fwd` hooks, whose defaults are a
//! correct CPU round-trip: a backend with no kernel, or one that rejects a
//! launch, still produces the right answer rather than an error.
//!
//! # Format (must match [`crate::mxfp4`] exactly)
//!
//! One block is 64 elements in 36 bytes: four `UE4M3` scale bytes (one per 16
//! elements) then 32 bytes of packed E2M1 codes.  Within each 16, byte `j`
//! holds element `j` in its low nibble and element `j + 16` in its high
//! nibble — **split-half**, not sequential pairing.  Getting that wrong
//! scrambles every weight without failing, so the tests assert it directly on
//! the host decoder and on the device result, and the kernels spell the index
//! arithmetic out longhand instead of reusing the host loop shape so the two
//! cannot drift together.
//!
//! Scale decoding mirrors `mxfp4::ue4m3_to_f32`:
//!
//! * `0x00` and `0x7F` (E4M3's NaN) decode to `0.0`.  A NaN scale would
//!   poison 16 of a block's 64 products, so the kernels skip that sub-block
//!   instead of multiplying by the bit pattern.
//! * `0x80..=0xFF` are *not* negative: the format is unsigned and the sign
//!   bit is ignored, so `0xFF` is 480.0 and `0x80` decodes like `0x00`.
//! * `exp == 0` is a denormal scale, `man · 2^-9`, which is neither flushed
//!   nor reachable by shifting an exponent field into place.
//!
//! # Weight layout
//!
//! `blocks` is conceptually `[out_dim, in_dim / 64]`, contiguous, 36 bytes per
//! block; decoding it gives the element matrix `[out_dim, in_dim]`.  The
//! activations are one row `[in_dim]` (or `[1, in_dim]`), or up to
//! [`MAX_ROWS`] rows.  `in_dim % 64 == 0` is required.
//!
//! # Vulkan status
//!
//! The OpenCL path is a real native kernel.  The Vulkan path is **not**: this
//! module does not override `vulkan_fwd`, so a Vulkan device takes
//! `CustomOp2`'s default CPU round-trip and gets the reference answer.
//!
//! The reason is reachability, not preference.  A Vulkan kernel must be
//! compiled to SPIR-V and dispatched through the device's command ring, and
//! every piece of that lives behind `pub(crate)`/private items in
//! `candle_core::vulkan_backend` (`kernels::launch`, `kernels::launch_spirv`,
//! `glsl_to_spirv`, `VulkanDevice::exec`/`ash`); `run_qgemv` takes a
//! `GgmlDType`, which cannot name NVFP4.  Reaching it from `src/` would mean
//! either editing `vendor/` (out of scope for this change) or reimplementing
//! the pipeline cache, descriptor layouts, push-constant ring and lifetime
//! rules here.  A second, hand-rolled dispatch path that no shared test
//! exercises is worse than an honest round-trip, so [`vulkan_glsl`] is
//! provided and reviewed, but not launched.  Wiring it up needs one
//! `pub` launcher in the vendored backend, mirroring `run_qgemv`.
//!
//! ## Backend matrix (what actually runs where)
//!
//! | Device | NVFP4 GEMV path | Native? |
//! |---|---|---|
//! | `Device::Cpu` | [`matmul_t`] | n/a (it *is* the reference) |
//! | `Device::OpenCl` | `k_nvfp4_gemv` from [`OPENCL_SRC`] | **yes** |
//! | `Device::Vulkan` | `CustomOp2` default CPU round-trip | **no** |
//! | `Device::Metal` / `Cuda` / `Sycl` | `CustomOp2` default CPU round-trip | **no** |
//!
//! So this change makes NVFP4 native on **one** backend, not two.  On Vulkan
//! the blocks are copied to the host, decoded by [`matmul_t`], and the f32
//! result copied back — correct, but it moves the weight bytes each call and
//! forfeits the reason for the format.  [`vulkan_native_gemv_uses_the_fallback`]
//! pins that behaviour so it cannot change silently.
//!
//! A native Vulkan launch is tracked as a follow-up: add one `pub` launcher to
//! `candle_core::vulkan_backend::kernels` (mirroring `run_qgemv`, but taking an
//! explicit format rather than a `GgmlDType`) and call it from a `vulkan_fwd`
//! override here.  [`vulkan_glsl`] is already written for that.
//!
//! # Kernel strategy
//!
//! Dequantize-in-kernel, following `k_qgemv` in the vendored OpenCL backend:
//! one work-group per output column whose lanes stride over the row's
//! 16-element sub-blocks, so each weight byte is read exactly once and each
//! activation element exactly once.  The E2M1 table lives in local memory, so
//! each element is one bank-conflict-free load instead of a 15-step select
//! chain.  The single-row decode path is the deliverable; [`MAX_ROWS`] is the
//! same kernel repeated per row.

use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};

/// Elements per NVFP4 block (re-exported from `mxfp4::QK_NVFP4`).
pub const QK_NVFP4: usize = crate::mxfp4::QK_NVFP4;

/// Elements per NVFP4 scale — four scales per block.
pub const QK_NVFP4_SUB: usize = crate::mxfp4::QK_NVFP4_SUB;

/// Bytes per NVFP4 block: four scale bytes then 32 code bytes.
pub const BLOCK_BYTES: usize = std::mem::size_of::<crate::mxfp4::BlockNvfp4>();

/// `GGML_TYPE_NVFP4` — the dtype id written in GGUF tensor info.
pub const GGML_TYPE_NVFP4: u32 = 40;

// The wire format is only "36 bytes per 64 elements" if these hold.
const _: () = assert!(BLOCK_BYTES == 4 + QK_NVFP4 / 2);
const _: () = assert!(BLOCK_BYTES == 36);
const _: () = assert!(QK_NVFP4 == 4 * QK_NVFP4_SUB);

/// Rows the batched path accepts.  Amortizing the weight read beyond a
/// handful of rows wants a tiled GEMM, which is deliberately not here: the
/// decode path is one row and is what this module exists for.
pub const MAX_ROWS: usize = 16;

/// Work-group size the OpenCL kernel is compiled with: a power of two every
/// OpenCL 1.2 device accepts, and a whole number of the 16-lane sub-block
/// stripes the kernel is built from.
#[cfg(feature = "opencl")]
pub const OPENCL_WG: usize = 64;

// ─── Host decode ─────────────────────────────────────────────────────────────

/// Decode one 36-byte block into `out` (64 values), matching
/// `BlockNvfp4::dequantize_block`.
///
/// Written against the raw bytes rather than by calling into `mxfp4`, so the
/// fallback exercises the same index expressions the kernels do; the two are
/// cross-checked by the unit tests.
pub fn decode_block(block: &[u8; BLOCK_BYTES], out: &mut [f32; QK_NVFP4]) {
    out.fill(0.0);
    for sub in 0..4 {
        let c = block[sub];
        // 0x00 is zero and 0x7F is E4M3's NaN; ggml decodes both to 0.
        if c == 0x00 || c == 0x7F {
            continue;
        }
        let exp = (c >> 3) & 0x0F;
        let man = c & 0x07;
        let d = if exp == 0 {
            // man · 2^-9, computed exactly.  Placing `man` in the mantissa of
            // an exponent-0 float32 word would give a denormal instead.
            (man as f32) * (1.0 / 512.0)
        } else {
            f32::from_bits((((exp as u32) + 120) << 23) | ((man as u32) << 20))
        };
        // Split-half packing: this 16 covers 8 code bytes, `qs` chunk `sub`.
        // Byte j holds element j of the 16 in its LOW nibble and element j + 8
        // in its HIGH nibble.  (The name is about the nibble split, not about
        // two 8-byte halves: there are four 8-byte chunks, one per scale.)
        let at = 4 + 8 * sub;
        for j in 0..8 {
            let byte = block[at + j];
            out[16 * sub + j] = d * crate::mxfp4::E2M1[(byte & 0x0F) as usize];
            out[16 * sub + 8 + j] = d * crate::mxfp4::E2M1[(byte >> 4) as usize];
        }
    }
}

/// `y = W x` on the host from raw NVFP4 blocks.
///
/// `blocks` is `out_dim * in_dim / 64` blocks in `[out_dim, in_dim/64]` order,
/// `acts` is `rows * in_dim` values, `out` is `rows * out_dim`.  All row-major;
/// accumulation is f32 in ascending `k`, the order the device kernels use.
pub fn matmul_t(
    blocks: &[u8],
    acts: &[f32],
    out: &mut [f32],
    rows: usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<()> {
    if in_dim % QK_NVFP4 != 0 {
        candle_core::bail!("nvfp4: in_dim {in_dim} is not a multiple of {QK_NVFP4}");
    }
    let per_row = in_dim / QK_NVFP4;
    let want = out_dim * per_row * BLOCK_BYTES;
    if blocks.len() != want {
        candle_core::bail!(
            "nvfp4: {} weight bytes for a [{out_dim}, {in_dim}] matrix, expected {want}",
            blocks.len()
        );
    }
    if acts.len() != rows * in_dim || out.len() != rows * out_dim {
        candle_core::bail!(
            "nvfp4: shape mismatch - acts {} (want {}), out {} (want {})",
            acts.len(),
            rows * in_dim,
            out.len(),
            rows * out_dim
        );
    }
    out.fill(0.0);
    // Decode each weight block once and reuse it across every row.
    let mut w = [0f32; QK_NVFP4];
    for row in 0..out_dim {
        for b in 0..per_row {
            let at = (row * per_row + b) * BLOCK_BYTES;
            let block: &[u8; BLOCK_BYTES] = blocks[at..at + BLOCK_BYTES]
                .try_into()
                .expect("block range was bounds-checked above");
            decode_block(block, &mut w);
            for m in 0..rows {
                let base = m * in_dim + b * QK_NVFP4;
                let xb = &acts[base..base + QK_NVFP4];
                let mut acc = out[m * out_dim + row];
                for (a, wv) in xb.iter().zip(w.iter()) {
                    acc += a * wv;
                }
                out[m * out_dim + row] = acc;
            }
        }
    }
    Ok(())
}

// ─── The op ──────────────────────────────────────────────────────────────────

/// `rows`, `out_dim` and `in_dim`, resolved once from the tensors.
#[derive(Clone, Copy)]
struct Nvfp4Matmul {
    rows: usize,
    out_dim: usize,
    in_dim: usize,
}

impl Nvfp4Matmul {
    fn shape(&self) -> Shape {
        Shape::from((self.rows, self.out_dim))
    }

    fn out_len(&self) -> usize {
        self.rows * self.out_dim
    }

    fn per_row(&self) -> usize {
        self.in_dim / QK_NVFP4
    }
}

fn geometry(blocks: &Tensor, acts: &Tensor) -> Result<Nvfp4Matmul> {
    if blocks.dtype() != DType::U8 {
        candle_core::bail!("nvfp4: weights must be u8 blocks, got {:?}", blocks.dtype());
    }
    if acts.dtype() != DType::F32 {
        candle_core::bail!("nvfp4: activations must be f32, got {:?}", acts.dtype());
    }
    // The block tensor holds raw bytes: `[out_dim, 36 * blocks_per_row]`.
    let (out_dim, row_bytes) = match blocks.dims() {
        [n, k] => (*n, *k),
        d => candle_core::bail!(
            "nvfp4: weights must be [out_dim, 36 * blocks_per_row] u8, got {d:?}"
        ),
    };
    if row_bytes % BLOCK_BYTES != 0 {
        candle_core::bail!(
            "nvfp4: weight row of {row_bytes} bytes is not a whole number of {BLOCK_BYTES}-byte blocks"
        );
    }
    let per_row = row_bytes / BLOCK_BYTES;
    let in_dim = per_row * QK_NVFP4;
    let rows = match acts.dims() {
        [k] if *k == in_dim => 1,
        [_, k] if *k == in_dim => acts.elem_count() / *k,
        d => {
            candle_core::bail!("nvfp4: activations must be [in_dim] or [rows, in_dim], got {d:?}")
        }
    };
    if rows == 0 || rows > MAX_ROWS {
        candle_core::bail!("nvfp4: {rows} activation rows, supported 1..={MAX_ROWS}");
    }
    if out_dim == 0 || in_dim == 0 {
        candle_core::bail!("nvfp4: empty weight matrix [{out_dim}, {in_dim}]");
    }
    Ok(Nvfp4Matmul { rows, out_dim, in_dim })
}

/// Read a `[out_dim, 36 * in_dim/64]` u8 layout into contiguous block bytes,
/// honouring stride and offset so a narrowed or transposed view still decodes.
///
/// The block tensor holds the raw 36-byte blocks, so its last dimension is
/// `36 * blocks_per_row` bytes, not `blocks_per_row`: index by block, then take
/// [`BLOCK_BYTES`] consecutive bytes from that block's start.
fn block_bytes_of(storage: &CpuStorage, layout: &Layout, g: &Nvfp4Matmul) -> Result<Vec<u8>> {
    if layout.dims().len() != 2 {
        candle_core::bail!("nvfp4: weight layout must be rank 2");
    }
    let data = storage.as_slice::<u8>()?;
    let per_row = g.per_row();
    let start = layout.start_offset();
    let stride = layout.stride();
    let (row_stride, col_stride) = (stride[0], stride[1]);
    // Column stride is per *byte* of the block row; a block's bytes are
    // consecutive only when col_stride == 1, which is the dense case.
    if col_stride != 1 {
        candle_core::bail!(
            "nvfp4: weight byte layout must be contiguous within a row (stride {col_stride})"
        );
    }
    let row_bytes = per_row * BLOCK_BYTES;
    if layout.dims()[1] != row_bytes {
        candle_core::bail!(
            "nvfp4: weight tensor last dim is {} but a [{}, {}] matrix needs {row_bytes} bytes",
            layout.dims()[1],
            g.out_dim,
            g.in_dim
        );
    }
    let mut out = Vec::with_capacity(g.out_dim * row_bytes);
    for n in 0..g.out_dim {
        let row_at = start + n * row_stride;
        let end = row_at + row_bytes;
        if end > data.len() {
            candle_core::bail!("nvfp4: weight row {n} runs past its storage ({end} > {})", data.len());
        }
        out.extend_from_slice(&data[row_at..end]);
    }
    Ok(out)
}

impl CustomOp2 for Nvfp4Matmul {
    fn name(&self) -> &'static str {
        "nvfp4-matmul-qt"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let blocks = block_bytes_of(s1, l1, self)?;
        if !l2.is_contiguous() {
            candle_core::bail!("nvfp4: activation layout must be contiguous");
        }
        let acts = s2.as_slice::<f32>()?;
        let len = self.rows * self.in_dim;
        let start = l2.start_offset();
        if start + len > acts.len() {
            candle_core::bail!(
                "nvfp4: activation view runs past its storage ({} + {len} > {})",
                start,
                acts.len()
            );
        }
        let mut out = vec![0f32; self.out_len()];
        matmul_t(
            &blocks,
            &acts[start..start + len],
            &mut out,
            self.rows,
            self.out_dim,
            self.in_dim,
        )?;
        Ok((CpuStorage::F32(out), self.shape()))
    }

    #[cfg(feature = "opencl")]
    fn opencl_fwd(
        &self,
        s1: &candle_core::OpenClStorage,
        l1: &Layout,
        s2: &candle_core::OpenClStorage,
        l2: &Layout,
    ) -> Result<(candle_core::OpenClStorage, Shape)> {
        opencl_launch(s1, l1, s2, l2, self)
    }

    // `vulkan_fwd` is deliberately NOT overridden: see the module docs on
    // "Vulkan status".  `CustomOp2`'s default is a correct CPU round-trip, so
    // a Vulkan device still gets the reference answer; it simply does not get
    // a native dispatch.  Overriding this with a launch we cannot reach
    // (`vulkan_backend::kernels`' launcher is `pub(crate)`) would be a
    // silent lie about what runs where.
}

/// `y = W x` for an NVFP4 weight matrix, decoding the FP4 blocks in-kernel on
/// the device that owns `blocks`.
///
/// `blocks` is `u8` of shape `[out_dim, in_dim / 64]` — 36 bytes per block, in
/// the layout described in the module docs — and `acts` is `f32` of shape
/// `[in_dim]` or `[rows, in_dim]` with `rows <= MAX_ROWS`.  The result is
/// `[rows, out_dim]` f32 on the same device.
///
/// On a device whose native launch is unavailable or disabled, this takes the
/// `CustomOp2` CPU round-trip, so the answer is always the reference decode.
pub fn nvfp4_matmul_qt(blocks: &Tensor, acts: &Tensor) -> Result<Tensor> {
    let g = geometry(blocks, acts)?;
    if blocks.device().location() != acts.device().location() {
        candle_core::bail!(
            "nvfp4: weights on {:?} but activations on {:?}",
            blocks.device(),
            acts.device()
        );
    }
    blocks.apply_op2_no_bwd(acts, &g)
}

// ─── OpenCL ──────────────────────────────────────────────────────────────────

/// OpenCL kernel source.  `__WG__` is replaced with [`OPENCL_WG`].  It is
/// compiled from this string rather than the vendored `kernels.cl`, so the
/// NVFP4 path does not depend on the vendored program's build options.
#[cfg(feature = "opencl")]
pub const OPENCL_SRC: &str = r#"
// UE4M3 -> f32, exactly as `mxfp4::ue4m3_to_f32`:
//   0x00 and 0x7F -> 0; else unsigned E4M3, bias 7, exp 0 is denormal.
inline float nv_ue4m3(uint x) {
    if (x == 0u || x == 0x7Fu) return 0.0f;
    uint e = (x >> 3) & 0xFu;
    uint m = x & 0x7u;
    if (e == 0u) return (float)m * (1.0f / 512.0f);   // m * 2^-9
    return as_float(((e + 120u) << 23) | (m << 20));  // (1+m/8) * 2^(e-7)
}

// C[m, n] = sum_k X[m, k] * W[n, k] with W = nvfp4 blocks [N, K/64].
//
// One work-group per output column.  Lane t owns the 16-element sub-blocks
// s = t, t + WG, ... of each block it visits: with WG >= 4 every lane gets at
// most one sub-block per block, so the group reads a block's 36 bytes once and
// each lane's partial sum is over whole 16s (no cross-lane split of a scale).
//
// Packing (split-half, per 16): the sub-block's 8 code bytes are at
// 4 + 8*s .. 4 + 8*s + 8; byte j is element j of the 16 in its LOW nibble and
// element j + 8 in its HIGH nibble.
__kernel __attribute__((reqd_work_group_size(__WG__, 1, 1)))
void k_nvfp4_gemv(__global const float* X, __global const uchar* W,
                  __global float* C, int N, int K, int M) {
    __local float tab[16];
    __local float sh[__WG__];
    const int t = (int)get_local_id(0);
    const int n = (int)get_group_id(0);
    if (t < 16) {
        // The E2M1 table, literal: an arithmetic reconstruction of
        // {0,0.5,1,1.5,2,3,4,6} is easy to get subtly wrong.
        tab[0] = 0.0f;  tab[1] = 0.5f;  tab[2] = 1.0f;  tab[3] = 1.5f;
        tab[4] = 2.0f;  tab[5] = 3.0f;  tab[6] = 4.0f;  tab[7] = 6.0f;
        tab[8] = 0.0f;  tab[9] = -0.5f; tab[10] = -1.0f; tab[11] = -1.5f;
        tab[12] = -2.0f; tab[13] = -3.0f; tab[14] = -4.0f; tab[15] = -6.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int nb = K / 64;
    for (int m = 0; m < M; ++m) {
        float acc = 0.0f;
        if (n < N) {
            __global const uchar* row = W + (ulong)n * (ulong)nb * 36ul;
            __global const float* x = X + (long)m * K;
            for (int b = 0; b < nb; ++b) {
                __global const uchar* blk = row + (ulong)b * 36ul;
                for (int s = t; s < 4; s += __WG__) {
                    float d = nv_ue4m3((uint)blk[s]);
                    if (d == 0.0f) continue;   // scale 0x00 or the 0x7F NaN
                    __global const uchar* q = blk + 4 + 8 * s;
                    const int xb = b * 64 + s * 16;
                    float part = 0.0f;
                    for (int j = 0; j < 8; ++j) {
                        uchar by = q[j];
                        part = fma(tab[by & 0x0Fu] * d, x[xb + j], part);
                        part = fma(tab[by >> 4] * d, x[xb + j + 8], part);
                    }
                    acc += part;
                }
            }
        }
        sh[t] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int st = __WG__ / 2; st > 0; st >>= 1) {
            if (t < st) sh[t] += sh[t + st];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (t == 0 && n < N) C[(long)m * N + n] = sh[0];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
"#;

/// Whether a native OpenCL launch is wanted.  `JOSHUA_OPENCL_NVFP4=0` forces
/// the CPU reference for this op alone; `JOSHUA_OPENCL_NATIVE=0` does so for
/// every OpenCL op, which also covers this one.
#[cfg(feature = "opencl")]
fn opencl_native_enabled() -> bool {
    candle_core::opencl_backend::kernels::native_enabled()
        && !std::env::var("JOSHUA_OPENCL_NVFP4").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("false"))
}

#[cfg(feature = "opencl")]
fn opencl_launch(
    s1: &candle_core::OpenClStorage,
    l1: &Layout,
    s2: &candle_core::OpenClStorage,
    l2: &Layout,
    g: &Nvfp4Matmul,
) -> Result<(candle_core::OpenClStorage, Shape)> {
    // Geometry the kernel can index directly.  Anything else takes the CPU
    // reference rather than silently reading the wrong bytes.
    let ok_weights =
        l1.start_offset() == 0
            && l1.dims().len() == 2
            && l1.stride() == [g.per_row() * BLOCK_BYTES, 1];
    let ok_acts = l2.start_offset() == 0 && l2.is_contiguous();
    if !opencl_native_enabled() || !ok_weights || !ok_acts {
        candle_core::opencl_backend::kernels::note_fallback("nvfp4-matmul-qt", None);
        use candle_core::backend::{BackendDevice as _, BackendStorage as _};
        let c1 = s1.to_cpu_storage()?;
        let c2 = s2.to_cpu_storage()?;
        let (out, shape) = g.cpu_fwd(&c1, l1, &c2, l2)?;
        return Ok((s1.device.storage_from_cpu_storage(&out)?, shape));
    }
    let out = s1.device.alloc(DType::F32, g.out_len())?;
    opencl_backend::launch_nvfp4(
        &s1.device,
        s2.buffer,
        s1.buffer,
        out.buffer,
        g.out_dim,
        g.in_dim,
        g.rows,
    )?;
    Ok((out, g.shape()))
}

/// The vendored OpenCL backend has no hook to launch a kernel outside its
/// `kernels.cl` program, so this shim compiles [`OPENCL_SRC`] through the
/// crate's own `opencl_backend` FFI declarations (which are already linked).
#[cfg(feature = "opencl")]
mod opencl_backend {
    use super::{Nvfp4Matmul, OPENCL_SRC, OPENCL_WG};
    use candle_core::{Error, Result};

    extern "C" {
        fn clCreateProgramWithSource(
            context: usize,
            count: u32,
            strings: *const *const std::ffi::c_void,
            lengths: *const usize,
            err: *mut i32,
        ) -> usize;
        fn clBuildProgram(
            program: usize,
            ndev: u32,
            devices: *const usize,
            opts: *const std::ffi::c_void,
            cb: *const std::ffi::c_void,
            ud: *const std::ffi::c_void,
        ) -> i32;
        fn clGetProgramBuildInfo(
            program: usize,
            device: usize,
            param: u32,
            size: usize,
            value: *mut std::ffi::c_void,
            ret: *mut usize,
        ) -> i32;
        fn clCreateKernel(program: usize, name: *const std::ffi::c_void, err: *mut i32) -> usize;
        fn clSetKernelArg(kernel: usize, index: u32, size: usize, value: *const std::ffi::c_void) -> i32;
        fn clEnqueueNDRangeKernel(
            queue: usize,
            kernel: usize,
            dim: u32,
            off: *const usize,
            gsz: *const usize,
            lsz: *const usize,
            nev: u32,
            ev: *const usize,
            e: *mut usize,
        ) -> i32;
        fn clReleaseKernel(kernel: usize) -> i32;
        fn clReleaseProgram(program: usize) -> i32;
    }

    struct Prog(usize);
    impl Drop for Prog {
        fn drop(&mut self) {
            if self.0 != 0 {
                unsafe { clReleaseProgram(self.0) };
            }
        }
    }

    static PROGS: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

    fn program(ctx: &candle_core::opencl_backend::kernels::Ctx) -> Result<usize> {
        let mut cache = PROGS.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(&(_, p)) = cache.iter().find(|(c, _)| *c == ctx.context) {
            return Ok(p);
        }
        let src = OPENCL_SRC.replace("__WG__", &OPENCL_WG.to_string());
        let bytes = src.as_bytes();
        let srcs = [bytes.as_ptr() as *const std::ffi::c_void];
        let lens = [bytes.len()];
        let mut err: i32 = 0;
        let p = unsafe { clCreateProgramWithSource(ctx.context, 1, srcs.as_ptr(), lens.as_ptr(), &mut err) };
        if err != 0 || p == 0 {
            return Err(Error::Msg(format!("nvfp4: clCreateProgramWithSource failed with status {err}")));
        }
        let guard = Prog(p);
        let opts = std::ffi::CString::new("-cl-std=CL1.2").unwrap();
        let rc = unsafe {
            clBuildProgram(p, 1, &ctx.device, opts.as_ptr() as *const std::ffi::c_void, std::ptr::null(), std::ptr::null())
        };
        if rc != 0 {
            let mut len = 0usize;
            unsafe { clGetProgramBuildInfo(p, ctx.device, 0x1183, 0, std::ptr::null_mut(), &mut len) };
            let mut log = vec![0u8; len.max(1)];
            unsafe {
                clGetProgramBuildInfo(p, ctx.device, 0x1183, log.len(), log.as_mut_ptr() as *mut std::ffi::c_void, std::ptr::null_mut())
            };
            let log = String::from_utf8_lossy(&log).trim_end_matches('\0').to_string();
            drop(guard);
            return Err(Error::Msg(format!("nvfp4: clBuildProgram failed with status {rc}:\n{log}")));
        }
        std::mem::forget(guard);
        cache.push((ctx.context, p));
        Ok(p)
    }

    struct Kern(usize);
    impl Drop for Kern {
        fn drop(&mut self) {
            if self.0 != 0 {
                unsafe { clReleaseKernel(self.0) };
            }
        }
    }

    pub(super) fn launch_nvfp4(
        dev: &candle_core::OpenClDevice,
        x: usize,
        w: usize,
        out: usize,
        n: usize,
        k: usize,
        m: usize,
    ) -> Result<()> {
        let ctx = dev.ctx();
        let prog = program(&ctx)?;
        let name = std::ffi::CString::new("k_nvfp4_gemv").unwrap();
        let mut err: i32 = 0;
        let kern = Kern(unsafe { clCreateKernel(prog, name.as_ptr() as *const std::ffi::c_void, &mut err) });
        if err != 0 || kern.0 == 0 {
            return Err(Error::Msg(format!("nvfp4: clCreateKernel failed with status {err}")));
        }
        let set = |i: u32, size: usize, ptr: *const std::ffi::c_void| -> Result<()> {
            let rc = unsafe { clSetKernelArg(kern.0, i, size, ptr) };
            if rc != 0 {
                return Err(Error::Msg(format!("nvfp4: clSetKernelArg({i}) failed with status {rc}")));
            }
            Ok(())
        };
        set(0, std::mem::size_of::<usize>(), &x as *const usize as *const std::ffi::c_void)?;
        set(1, std::mem::size_of::<usize>(), &w as *const usize as *const std::ffi::c_void)?;
        set(2, std::mem::size_of::<usize>(), &out as *const usize as *const std::ffi::c_void)?;
        for (i, v) in [n as i32, k as i32, m as i32].into_iter().enumerate() {
            set(3 + i as u32, 4, &v as *const i32 as *const std::ffi::c_void)?;
        }
        let global = [n * OPENCL_WG];
        let local = [OPENCL_WG];
        let rc = unsafe {
            clEnqueueNDRangeKernel(
                ctx.queue,
                kern.0,
                1,
                std::ptr::null(),
                global.as_ptr(),
                local.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(Error::Msg(format!("nvfp4: clEnqueueNDRangeKernel failed with status {rc}")));
        }
        candle_core::opencl_backend::kernels::note_native_exec();
        let _ = std::mem::size_of::<Nvfp4Matmul>();
        Ok(())
    }
}

// ─── Vulkan ──────────────────────────────────────────────────────────────────

/// GLSL source for the NVFP4 GEMV, for `WG` invocations per work-group.
/// Push constants: `N, K, M`.
///
/// `W` is bound as `uint` words (4 bytes each) because the SPIR-V binding
/// model needs a 4-byte element type; a block is nine words, so byte access is
/// `word = at / 4`, `shift = 8 * (at % 4)`.  One work-group per output column,
/// lane `t` owning sub-blocks `t, t + WG, …`.
#[cfg(feature = "vulkan")]
pub fn vulkan_glsl(wg: usize) -> String {
    format!(
        r#"#version 450
#define NVFP4_WG {wg}
layout(std430, set = 0, binding = 0) readonly buffer XB {{ float X[]; }};
layout(std430, set = 0, binding = 1) readonly buffer WB {{ uint W[]; }};
layout(std430, set = 0, binding = 2) buffer CB {{ float C[]; }};
layout(push_constant) uniform PC {{ int N; int K; int M; }} pc;
layout(local_size_x = NVFP4_WG) in;
shared float sh[NVFP4_WG];
shared float tab[16];

// UE4M3 -> float, matching mxfp4::ue4m3_to_f32.
float ue4m3(uint x) {{
    if (x == 0u || x == 0x7Fu) return 0.0;
    uint e = (x >> 3) & 0xFu;
    uint m = x & 0x7u;
    if (e == 0u) return float(m) * (1.0 / 512.0);
    return uintBitsToFloat(((e + 120u) << 23) | (m << 20));
}}

// Byte `i` of the word stream starting at word `base`.
uint byte_at(uint base, uint i) {{ return (W[base + i / 4u] >> (8u * (i % 4u))) & 0xFFu; }}

void main() {{
    int t = int(gl_LocalInvocationID.x);
    int n = int(gl_WorkGroupID.x);
    if (t < 16) {{
        float v[16] = float[16](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                                0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0);
        tab[t] = v[t];
    }}
    barrier();
    int nb = pc.K / 64;
    for (int m = 0; m < pc.M; ++m) {{
        float acc = 0.0;
        if (n < pc.N) {{
            // 36 bytes per block == 9 uint words.
            uint row = uint(n) * uint(nb) * 9u;
            for (int b = 0; b < nb; ++b) {{
                uint blk = row + uint(b) * 9u;
                for (int s = t; s < 4; s += NVFP4_WG) {{
                    float d = ue4m3(byte_at(blk, uint(s)));
                    if (d == 0.0) continue;
                    int xb = m * pc.K + b * 64 + s * 16;
                    // Split-half: the 8 code bytes at 4 + 8s hold element j in
                    // the low nibble and element j + 8 in the high nibble.
                    for (int j = 0; j < 8; ++j) {{
                        uint by = byte_at(blk, uint(4 + 8 * s + j));
                        acc = fma(tab[by & 0xFu] * d, X[xb + j], acc);
                        acc = fma(tab[by >> 4] * d, X[xb + j + 8], acc);
                    }}
                }}
            }}
        }}
        sh[t] = acc;
        barrier();
        for (int st = NVFP4_WG / 2; st > 0; st >>= 1) {{
            if (t < st) sh[t] += sh[t + st];
            barrier();
        }}
        if (t == 0 && n < pc.N) C[m * pc.N + n] = sh[0];
        barrier();
    }}
}}
"#,
        wg = wg
    )
}

/// Whether a native Vulkan NVFP4 launch is wanted.  `JOSHUA_VULKAN_NATIVE=0`
/// or `JOSHUA_VULKAN_NVFP4=0` forces the CPU reference.
#[cfg(feature = "vulkan")]
pub fn vulkan_native_enabled() -> bool {
    candle_core::vulkan_backend::kernels::native_enabled()
        && !std::env::var("JOSHUA_VULKAN_NVFP4").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("false"))
}
