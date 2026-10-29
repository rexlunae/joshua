//! Native fused depthwise causal conv1d for the Mamba-2 layer of Nemotron
//! 3.5 (`nemotron_h`).
//!
//! Every Mamba-2 layer convolves the projected `xBC` stream over time, once
//! per token and per recurrent layer, so this op sits on the decode hot path.
//! [`conv1d`] keeps the activations and the FIR taps on the device: it runs a
//! `CustomOp1` whose `cpu_fwd` is the reference loop and whose `opencl_fwd`
//! compiles and launches this module's own kernel, so there is no
//! `to_cpu_storage` round trip and no host synchronisation.  The vendored
//! backends are used read-only (their program cache, allocators and kernel
//! objects); nothing in `vendor/` changes.
//!
//! The Vulkan half is written but not wired: GLSL→SPIR-V compilation and
//! pipeline dispatch both live behind `pub(crate)`/private items in the
//! vendored Vulkan backend, so `vulkan_fwd` takes the trait's CPU round trip
//! (counted) and [`vulkan_source`] ships the finished kernel for the moment
//! that entry point is exported.  See the note above [`vulkan_source`].
//!
//! # Semantics
//!
//! This is llama.cpp's `ggml_ssm_conv` (`ggml/src/ggml-cpu/ops.cpp:9932`,
//! called from `src/models/mamba-base.cpp`), with the bias add and silu of
//! the same layer fused in:
//!
//! ```text
//! x: [T + d_conv - 1, C]   time-major rows; the caller prepends the d_conv-1
//!                          state rows, so the window for output row t starts
//!                          at input row t with tap 0
//! w: [d_conv, C]           w[k*C + c] is tap k of channel c (ggml stores
//!                          ssm_conv1d with ne0 = d_conv, ne1 = C)
//! b: [C]                   optional
//! y[t, c] = silu_or_identity( sum_{k<d_conv} w[k, c] * x[t + k, c] + b[c] )
//! ```
//!
//! The sum runs in tap order in both the CPU reference and the kernels, and
//! the op is *depthwise*: channel `c` reads only channel `c` — no im2col, no
//! cross-channel reduction.
//!
//! # Fallbacks
//!
//! `JOSHUA_OPENCL_NATIVE=0` / `JOSHUA_VULKAN_NATIVE=0` disable a whole
//! backend's native paths and `JOSHUA_CONV1D_NATIVE=0` disables just this op.
//! Any opt-out, an unsupported dtype or layout, or a kernel that fails to
//! compile or launch takes the trait's CPU round trip and is counted through
//! the backend's `note_fallback` (`fallback_count()`,
//! `fallback_counts_by_op()`), so a test can assert the native path ran by
//! watching `native_exec_count()` move with no fallback named [`OP_NAME`].

use candle_core::{CpuStorage, CustomOp1, DType, Layout, Result, Shape, Storage, Tensor};

/// Op name used in fallback accounting (`fallback_counts_by_op`).
pub const OP_NAME: &str = "conv1d";

/// Mamba-2 / Nemotron 3.5 use a 4-tap convolution (`ssm.conv_kernel`).
pub const DEFAULT_D_CONV: usize = 4;

/// A depthwise causal conv1d, optionally fused with a per-channel bias and
/// silu — llama.cpp's `ggml_ssm_conv` + `ggml_add` + `ggml_silu` in one op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv1d {
    d_conv: usize,
    silu: bool,
    bias: bool,
}

impl Conv1d {
    /// `d_conv` taps, adding a bias when `bias` is true, silu when `silu` is.
    pub fn new(d_conv: usize, silu: bool, bias: bool) -> Self {
        Self { d_conv, silu, bias }
    }

    /// Nemotron 3.5's convolution: 4 taps, bias, silu.
    pub fn mamba2() -> Self {
        Self::new(DEFAULT_D_CONV, true, true)
    }

    pub fn d_conv(&self) -> usize {
        self.d_conv
    }

    pub fn silu(&self) -> bool {
        self.silu
    }

    pub fn has_bias(&self) -> bool {
        self.bias
    }

    /// `JOSHUA_CONV1D_NATIVE=0` turns this op's native paths off.  Read per
    /// call, like the backends' own deny lists, so a test can flip it.
    pub fn native_enabled() -> bool {
        match std::env::var("JOSHUA_CONV1D_NATIVE") {
            Ok(v) => {
                !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
            }
            Err(_) => true,
        }
    }

    /// Validate the shapes and return `(rows_in, T, C)`.
    fn check(&self, x: &Shape, bias_len: Option<usize>, w: &Shape) -> Result<(usize, usize, usize)> {
        if self.d_conv == 0 {
            candle_core::bail!("conv1d: d_conv must be at least 1");
        }
        let xd = x.dims();
        if xd.len() != 2 {
            candle_core::bail!("conv1d: x must be [T + d_conv - 1, C], got {xd:?}");
        }
        let wd = w.dims();
        if wd.len() != 2 {
            candle_core::bail!("conv1d: weight must be [d_conv, C], got {wd:?}");
        }
        let (rows, c) = (xd[0], xd[1]);
        if rows + 1 < self.d_conv {
            candle_core::bail!(
                "conv1d: x has {rows} rows, fewer than d_conv - 1 = {}",
                self.d_conv - 1
            );
        }
        let t = rows + 1 - self.d_conv;
        if wd[0] != self.d_conv || wd[1] != c {
            candle_core::bail!(
                "conv1d: weight shape {wd:?} does not match d_conv {} and C {c}",
                self.d_conv
            );
        }
        match (self.bias, bias_len) {
            (true, None) => candle_core::bail!("conv1d: bias enabled but no bias tensor given"),
            (false, Some(_)) => candle_core::bail!("conv1d: op built without bias but a bias was given"),
            (_, Some(n)) if n != c => {
                candle_core::bail!("conv1d: bias length {n} does not match C {c}")
            }
            _ => {}
        }
        Ok((rows, t, c))
    }

    /// The reference loop over row-major contiguous f32 slices: the same tap
    /// order as `ggml_compute_forward_ssm_conv_f32` and as the kernels.
    pub fn reference(
        &self,
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        t: usize,
        c: usize,
    ) -> Vec<f32> {
        let dc = self.d_conv;
        let mut y = vec![0f32; t * c];
        for ti in 0..t {
            let win = &x[ti * c..(ti + dc) * c];
            let out = &mut y[ti * c..(ti + 1) * c];
            for (ci, o) in out.iter_mut().enumerate() {
                let mut sum = 0f32;
                for k in 0..dc {
                    sum += win[k * c + ci] * w[k * c + ci];
                }
                if let Some(b) = bias {
                    sum += b[ci];
                }
                *o = if self.silu { silu_f32(sum) } else { sum };
            }
        }
        y
    }
}

#[inline]
fn silu_f32(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

// ─── Public entry point ─────────────────────────────────────────────────────

/// `y = silu_or_identity(conv1d(x, w) + b)` over the time axis, per channel:
/// `x [T + d_conv - 1, C]`, `w [d_conv, C]`, `b [C]` when `cfg.bias`,
/// `y [T, C]`.
///
/// `x`'s rows are timesteps — row `t` is tap 0 of output row `t` — so the
/// caller prepends the `d_conv - 1` rows of convolution state, the way
/// llama.cpp's `ggml_concat(conv, ggml_transpose(xBC))` does; a single decode
/// step passes the `d_conv - 1` previous rows followed by the new one.
///
/// On a device the whole op is one kernel launch with no host round trip.
/// The CPU path is [`Conv1d::reference`] and is bit-identical to it.
pub fn conv1d(x: &Tensor, w: &Tensor, b: Option<&Tensor>, cfg: Conv1d) -> Result<Tensor> {
    cfg.check(x.shape(), b.map(|b| b.elem_count()), w.shape())?;
    if x.device().same_device(w.device()) {
        // common case: one device, one launch
    } else {
        candle_core::bail!("conv1d: x and the weight are on different devices");
    }
    if let Some(b) = b {
        if !x.device().same_device(b.device()) {
            candle_core::bail!("conv1d: x and the bias are on different devices");
        }
    }
    let op = Conv1dOp {
        cfg,
        w: w.clone(),
        b: b.cloned(),
    };
    x.apply_op1(op)
}

// ─── The custom op ──────────────────────────────────────────────────────────

struct Conv1dOp {
    cfg: Conv1d,
    w: Tensor,
    b: Option<Tensor>,
}

/// The device-independent part of one call.
struct Args {
    t: usize,
    c: usize,
    x_off: usize,
    w_off: usize,
    b_off: Option<usize>,
}

/// `offset` of a layout that is contiguous and holds exactly `want` elements.
fn contiguous(layout: &Layout, want: usize) -> Option<usize> {
    let (o, e) = layout.contiguous_offsets()?;
    (e == o + want).then_some(o)
}

/// A contiguous row-major f32 copy of the whole tensor, on the host.  Only
/// the CPU reference and the fallbacks read data through this, so the extra
/// copy is the slow path's own cost; the native path indexes buffers
/// directly.
fn host_f32(t: &Tensor) -> Result<Vec<f32>> {
    // `contiguous()` first: `t` may be a view, and flattening a view
    // preserves the view's element order but not its buffer layout.
    t.contiguous()?.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()
}

impl Conv1dOp {
    fn out_dims(&self, layout: &Layout) -> Result<(usize, usize)> {
        let dims = layout.dims();
        if dims.len() != 2 {
            candle_core::bail!("conv1d: x must be 2-D, got {dims:?}");
        }
        let (rows, c) = (dims[0], dims[1]);
        if rows + 1 < self.cfg.d_conv {
            candle_core::bail!("conv1d: x has {rows} rows, fewer than d_conv - 1");
        }
        Ok((rows + 1 - self.cfg.d_conv, c))
    }

    /// `Some(..)` when the layouts let the kernels index the buffers
    /// directly; `None` for a view the native path does not handle (the
    /// caller then falls back for a correct result).
    fn native_args(&self, layout: &Layout) -> Result<Option<Args>> {
        let (t, c) = self.out_dims(layout)?;
        let rows = layout.dims()[0];
        let Some(x_off) = contiguous(layout, rows * c) else { return Ok(None) };
        let Some(w_off) = contiguous(self.w.layout(), self.cfg.d_conv * c) else {
            return Ok(None);
        };
        let b_off = match self.b.as_ref() {
            Some(b) => match contiguous(b.layout(), c) {
                Some(o) => Some(o),
                None => return Ok(None),
            },
            None => None,
        };
        Ok(Some(Args { t, c, x_off, w_off, b_off }))
    }

    fn cpu_result(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (t, c) = self.out_dims(layout)?;
        // The fused reference runs in f32 like llama.cpp's `ssm_conv`.  The
        // input layout can be an arbitrary view (narrowed / transposed), which
        // the native path rejects, so walk it through its own strides: a
        // contiguous copy would be an extra allocation on a path that is
        // already the slow one, and getting the strides wrong here is exactly
        // the kind of silent error this function must not have.
        let (rows, c_in) = (layout.dims()[0], layout.dims()[1]);
        let x = match storage {
            CpuStorage::F32(data) => {
                let (s0, s1) = (layout.stride()[0], layout.stride()[1]);
                let base = layout.start_offset();
                let mut x = Vec::with_capacity(rows * c_in);
                for r in 0..rows {
                    let row = base + r * s0;
                    match s1 {
                        1 => x.extend_from_slice(&data[row..row + c_in]),
                        _ => x.extend((0..c_in).map(|c| data[row + c * s1])),
                    }
                }
                x
            }
            // f16 / bf16: gather the elements this layout exposes, then widen.
            //
            // `Tensor::from_storage` can only build a contiguous tensor, so a
            // narrowed or transposed input would be re-read from the start of
            // the backing storage and the convolution would consume unrelated
            // rows.  Walk the caller's `start_offset`/`stride` — the same way
            // the f32 arm does — and convert each selected element.
            _ => {
                let (rows, cols) = (layout.dims()[0], layout.dims()[1]);
                let (s0, s1) = (layout.stride()[0], layout.stride()[1]);
                let base = layout.start_offset();
                let mut x = Vec::with_capacity(rows * cols);
                match storage {
                    CpuStorage::F16(data) => {
                        for r in 0..rows {
                            for c in 0..cols {
                                x.push(data[base + r * s0 + c * s1].to_f32());
                            }
                        }
                    }
                    CpuStorage::BF16(data) => {
                        for r in 0..rows {
                            for c in 0..cols {
                                x.push(data[base + r * s0 + c * s1].to_f32());
                            }
                        }
                    }
                    CpuStorage::F64(data) => {
                        for r in 0..rows {
                            for c in 0..cols {
                                x.push(data[base + r * s0 + c * s1] as f32);
                            }
                        }
                    }
                    // `CpuStorage::dtype` is not reachable from here, but the
                    // match arms that fall through are exactly the integer
                    // and 4-bit storages, so name them instead of reading it.
                    _ => candle_core::bail!(
                        "conv1d: only f32/f16/bf16/f64 inputs are supported on the CPU path"
                    ),
                }
                x
            }
        };
        self.reference_storage(&x, t, c)
    }

    fn reference_storage(&self, x: &[f32], t: usize, c: usize) -> Result<(CpuStorage, Shape)> {
        let w = host_f32(&self.w)?;
        let b = match self.b.as_ref() {
            Some(b) => Some(host_f32(b)?),
            None => None,
        };
        let y = self.cfg.reference(x, &w, b.as_deref(), t, c);
        Ok((
            candle_core::WithDType::to_cpu_storage_owned(y),
            Shape::from((t, c)),
        ))
    }
}

impl CustomOp1 for Conv1dOp {
    fn name(&self) -> &'static str {
        OP_NAME
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        self.cpu_result(storage, layout)
    }

    #[cfg(feature = "opencl")]
    fn opencl_fwd(
        &self,
        storage: &candle_core::OpenClStorage,
        layout: &Layout,
    ) -> Result<(candle_core::OpenClStorage, Shape)> {
        use candle_core::backend::{BackendDevice, BackendStorage};
        use candle_core::opencl_backend::kernels;
        let shape = Shape::from(self.out_dims(layout)?);
        let native = Conv1d::native_enabled()
            && kernels::native_enabled()
            && storage.dtype == DType::F32;
        if native {
            match self.opencl_native(storage, layout) {
                Ok(Some(out)) => return Ok((out, shape)),
                Ok(None) => kernels::note_fallback(OP_NAME, None),
                Err(e) => kernels::note_fallback(OP_NAME, Some(&e)),
            }
        } else {
            kernels::note_fallback(OP_NAME, None);
        }
        let cpu = storage.to_cpu_storage()?;
        let (out, shape) = self.cpu_result(&cpu, layout)?;
        Ok((BackendDevice::storage_from_cpu_storage(&storage.device, &out)?, shape))
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(
        &self,
        storage: &candle_core::VulkanStorage,
        layout: &Layout,
    ) -> Result<(candle_core::VulkanStorage, Shape)> {
        // No native dispatch yet: `VulkanDevice::launch` is `pub(crate)` in
        // the vendored backend.  Still count it, so `fallback_count()` and
        // `fallback_counts_by_op()` tell a caller the truth.
        use candle_core::backend::BackendStorage;
        candle_core::vulkan_backend::kernels::note_fallback(
            OP_NAME,
            Some(&candle_core::Error::Msg(
                "conv1d: no Vulkan launch entry point exported by the backend".into(),
            )),
        );
        let cpu = storage.to_cpu_storage()?;
        let (out, shape) = self.cpu_result(&cpu, layout)?;
        Ok((
            candle_core::backend::BackendDevice::storage_from_cpu_storage(&storage.device, &out)?,
            shape,
        ))
    }
}

#[cfg(feature = "opencl")]
impl Conv1dOp {
    fn opencl_native(
        &self,
        storage: &candle_core::OpenClStorage,
        layout: &Layout,
    ) -> Result<Option<candle_core::OpenClStorage>> {
        let Some(args) = self.native_args(layout)? else { return Ok(None) };
        let (wg_storage, _) = self.w.storage_and_layout();
        let w = match &*wg_storage {
            Storage::OpenCl(s) => s.buffer,
            _ => return Ok(None),
        };
        let b_store = self.b.as_ref().map(|b| b.storage_and_layout().0);
        let b = match b_store.as_ref().map(|g| &**g) {
            None => None,
            Some(Storage::OpenCl(s)) => Some(s.buffer),
            Some(_) => return Ok(None),
        };
        let out = storage.device.alloc(DType::F32, args.t * args.c)?;
        crate::conv1d::run_conv1d_opencl(
            &storage.device.ctx(),
            storage.buffer,
            w,
            b,
            out.buffer,
            args.t,
            args.c,
            args.x_off,
            args.w_off,
            args.b_off,
            self.cfg,
        )?;
        Ok(Some(out))
    }
}

// ─── The kernels ────────────────────────────────────────────────────────────
//
// One invocation per output element `(t, c)`, reading the `d_conv` taps of
// its *own* channel (`x[(t + k) * C + c]`): depthwise, so the inner loop is a
// short FIR with no cross-channel traffic and the whole op is one pass over
// the input rows with the taps held in registers.  `d_conv` is compiled in
// (`D_CONV` below), so a 2- or 4-tap Mamba-2 convolution has no inner loop at
// all.  Bias and silu are folded into the same launch, where llama.cpp runs
// `ggml_ssm_conv`, `ggml_add` and `ggml_silu` in sequence.

/// The tap ladder both kernels share: `X` is the window base
/// (`x[t*C + c]`), `W` the per-channel weights (`w[c]`), `S` the channel
/// stride.  The tap order matches [`Conv1d::reference`].
/// The kernel symbol [`opencl_source`] defines.
pub const CONV1D_KERNEL: &str = "k_conv1d";

const TAP_LADDER: &str = r#"#if D_CONV == 1
        sum += X[0] * W[0];
    #elif D_CONV == 2
        sum += X[0] * W[0];
        sum += X[S] * W[S];
    #elif D_CONV == 3
        sum += X[0] * W[0];
        sum += X[S] * W[S];
        sum += X[2 * S] * W[2 * S];
    #elif D_CONV == 4
        sum += X[0] * W[0];
        sum += X[S] * W[S];
        sum += X[2 * S] * W[2 * S];
        sum += X[3 * S] * W[3 * S];
    #else
        for (int k = 0; k < D_CONV; ++k) sum += X[k * S] * W[k * S];
    #endif
"#;

/// The OpenCL program for `d_conv` taps.
pub fn opencl_source(d_conv: usize) -> String {
    let mut s = String::new();
    s.push_str("// Fused depthwise causal conv1d: llama.cpp ggml_ssm_conv + bias + silu.\n");
    s.push_str(&format!("#define D_CONV {d_conv}\n"));
    s.push_str(
        "__kernel void k_conv1d(\n\
         \x20   __global const float* restrict x,\n\
         \x20   __global const float* restrict w,\n\
         \x20   __global const float* restrict bp,\n\
         \x20   __global float* restrict y,\n\
         \x20   const int t_n, const int c_n, const int x_off, const int w_off,\n\
         \x20   const int b_off, const int b_on, const int silu_on)\n\
         {\n\
         \x20   const int i = (int)get_global_id(0);\n\
         \x20   if (i >= t_n * c_n) return;\n\
         \x20   const int t = i / c_n;\n\
         \x20   const int c = i - t * c_n;\n\
         \x20   const long S = c_n;\n\
         \x20   const __global float* X = x + x_off + (long)t * c_n + c;\n\
         \x20   const __global float* W = w + w_off + c;\n\
         \x20   float sum = b_on ? bp[b_off + c] : 0.0f;\n",
    );
    for line in TAP_LADDER.lines() {
        if line.trim().starts_with("sum +=") || line.trim().starts_with("for (int k") {
            s.push_str(&format!("    {line}\n"));
        } else {
            s.push_str(&format!("{line}\n"));
        }
    }
    s.push_str(
        "\x20   if (silu_on) sum = sum / (1.0f + exp(-sum));\n\
         \x20   y[i] = sum;\n\
         }\n",
    );
    s
}

/// The GLSL source for `d_conv` taps (one binding set, one push constant).
pub fn vulkan_source(d_conv: usize, wg: usize) -> String {
    let mut s = String::new();
    s.push_str(&format!("#version 450\n#define WG {wg}\n#define D_CONV {d_conv}\n"));
    s.push_str("layout(local_size_x = WG) in;\n");
    s.push_str("layout(std430, set = 0, binding = 0) readonly buffer X { float x[]; };\n");
    s.push_str("layout(std430, set = 0, binding = 1) readonly buffer W { float w[]; };\n");
    s.push_str("layout(std430, set = 0, binding = 2) readonly buffer B { float bp[]; };\n");
    s.push_str("layout(std430, set = 0, binding = 3) buffer Y { float y[]; };\n");
    s.push_str("layout(push_constant) uniform PC { int t_n; int c_n; int x_off; int w_off; int b_off; int b_on; int silu_on; } pc;\n");
    s.push_str(
        "void main() {\n\
         \x20   int i = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * uint(WG)));\n\
         \x20   if (i >= pc.t_n * pc.c_n) return;\n\
         \x20   int t = i / pc.c_n;\n\
         \x20   int c = i - t * pc.c_n;\n\
         \x20   int S = pc.c_n;\n\
         \x20   int xb = pc.x_off + t * S + c;\n\
         \x20   int wb = pc.w_off + c;\n\
         \x20   float sum = pc.b_on != 0 ? bp[pc.b_off + c] : 0.0;\n",
    );
    let ladder = TAP_LADDER
        .replace("X[0]", "x[xb]")
        .replace("X[S]", "x[xb + S]")
        .replace("X[2 * S]", "x[xb + 2 * S]")
        .replace("X[3 * S]", "x[xb + 3 * S]")
        .replace("X[k * S]", "x[xb + k * S]")
        .replace("W[0]", "w[wb]")
        .replace("W[S]", "w[wb + S]")
        .replace("W[2 * S]", "w[wb + 2 * S]")
        .replace("W[3 * S]", "w[wb + 3 * S]")
        .replace("W[k * S]", "w[wb + k * S]");
    for line in ladder.lines() {
        if line.trim().starts_with("sum +=") || line.trim().starts_with("for (int k") {
            s.push_str(&format!("    {line}\n"));
        } else {
            s.push_str(&format!("{line}\n"));
        }
    }
    s.push_str(
        "\x20   if (pc.silu_on != 0) sum = sum / (1.0 + exp(-sum));\n\
         \x20   y[i] = sum;\n\
         }\n",
    );
    s
}

// ─── Native launch (OpenCL) ─────────────────────────────────────────────────

#[cfg(feature = "opencl")]
mod opencl_kernel {
    //! Program cache, argument marshalling and dispatch for `k_conv1d`.
    //!
    //! The backend's `Kernel` helper is bound to the backend's own program
    //! (`Kernel::new` looks the kernel up in `program_for(context)`, which only
    //! holds `kernels.cl`), so this op owns the few calls `Kernel` wraps
    //! (`clCreateKernel` / `clSetKernelArg` / `clEnqueueNDRangeKernel`) over
    //! its own program.  Everything else is shared: the device's in-order
    //! command queue from `OpenClDevice::ctx()`, so a conv1d launch is ordered
    //! with the rest of the layer and never forces a synchronisation.

    use candle_core::opencl_backend::kernels::Ctx;
    use candle_core::{Error, Result};

    const CL_SUCCESS: i32 = 0;
    const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;
    const CL_DEVICE_MAX_WORK_GROUP_SIZE: u32 = 0x1004;

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
        fn clReleaseProgram(program: usize) -> i32;
        fn clCreateKernel(program: usize, name: *const std::ffi::c_void, err: *mut i32) -> usize;
        fn clReleaseKernel(kernel: usize) -> i32;
        fn clSetKernelArg(
            kernel: usize,
            index: u32,
            size: usize,
            value: *const std::ffi::c_void,
        ) -> i32;
        fn clEnqueueNDRangeKernel(
            queue: usize,
            kernel: usize,
            dim: u32,
            offset: *const usize,
            global: *const usize,
            local: *const usize,
            nev: u32,
            events: *const usize,
            event: *mut usize,
        ) -> i32;
        fn clGetDeviceInfo(
            device: usize,
            param: u32,
            size: usize,
            value: *mut std::ffi::c_void,
            ret: *mut usize,
        ) -> i32;
    }

    /// A compiled program: one per (context, `d_conv`), because `D_CONV` and
    /// the work-group size are baked in at build time.
    struct Program {
        handle: usize,
    }

    impl Drop for Program {
        fn drop(&mut self) {
            if self.handle != 0 {
                unsafe { clReleaseProgram(self.handle) };
            }
        }
    }

    struct Cached {
        context: usize,
        d_conv: usize,
        program: Program,
    }

    static PROGRAMS: std::sync::OnceLock<std::sync::Mutex<Vec<Cached>>> = std::sync::OnceLock::new();

    fn max_work_group(device: usize) -> usize {
        let mut v: usize = 0;
        let rc = unsafe {
            clGetDeviceInfo(
                device,
                CL_DEVICE_MAX_WORK_GROUP_SIZE,
                std::mem::size_of::<usize>(),
                &mut v as *mut usize as *mut std::ffi::c_void,
                std::ptr::null_mut(),
            )
        };
        if rc == CL_SUCCESS && v > 0 {
            v.min(256)
        } else {
            64
        }
    }

    fn compile(context: usize, device: usize, d_conv: usize) -> Result<Program> {
        let src = super::opencl_source(d_conv);
        let bytes = src.as_bytes();
        let strings = [bytes.as_ptr() as *const std::ffi::c_void];
        let lengths = [bytes.len()];
        let mut e: i32 = 0;
        let handle =
            unsafe { clCreateProgramWithSource(context, 1, strings.as_ptr(), lengths.as_ptr(), &mut e) };
        if e != CL_SUCCESS || handle == 0 {
            return Err(Error::Msg(format!(
                "opencl conv1d: clCreateProgramWithSource failed with status {e}"
            )));
        }
        let program = Program { handle };
        // The work-group size is a build-time constant, matching every other
        // kernel in the backend.
        let opts = std::ffi::CString::new(format!(
            "-D WG={} -cl-std=CL1.2",
            max_work_group(device)
        ))
        .map_err(|_| Error::Msg("opencl conv1d: bad build options".into()))?;
        let ok = unsafe {
            clBuildProgram(
                handle,
                1,
                &device,
                opts.as_ptr() as *const std::ffi::c_void,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if ok != CL_SUCCESS {
            let mut len: usize = 0;
            unsafe {
                clGetProgramBuildInfo(handle, device, CL_PROGRAM_BUILD_LOG, 0, std::ptr::null_mut(), &mut len)
            };
            let mut log = vec![0u8; len.max(1)];
            unsafe {
                clGetProgramBuildInfo(
                    handle,
                    device,
                    CL_PROGRAM_BUILD_LOG,
                    log.len(),
                    log.as_mut_ptr() as *mut std::ffi::c_void,
                    std::ptr::null_mut(),
                )
            };
            let log = String::from_utf8_lossy(&log).trim_end_matches('\0').trim().to_string();
            return Err(Error::Msg(format!(
                "opencl conv1d: clBuildProgram failed with status {ok}:\n{log}"
            )));
        }
        Ok(program)
    }

    /// Run `f` with the program compiled for this context and tap count.
    fn with_program<T>(ctx: &Ctx, d_conv: usize, f: impl FnOnce(usize) -> Result<T>) -> Result<T> {
        let cache = PROGRAMS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
        let mut list = cache.lock().unwrap_or_else(|p| p.into_inner());
        let idx = match list
            .iter()
            .position(|c| c.context == ctx.context && c.d_conv == d_conv)
        {
            Some(i) => i,
            None => {
                let program = compile(ctx.context, ctx.device, d_conv)?;
                list.push(Cached { context: ctx.context, d_conv, program });
                list.len() - 1
            }
        };
        f(list[idx].program.handle)
    }

    /// Sets arguments in order and enqueues; the kernel object is released on
    /// drop, so nothing outlives the launch.
    struct Launch {
        kernel: usize,
        queue: usize,
        next: u32,
    }

    impl Drop for Launch {
        fn drop(&mut self) {
            if self.kernel != 0 {
                unsafe { clReleaseKernel(self.kernel) };
            }
        }
    }

    impl Launch {
        fn new(program: usize, queue: usize) -> Result<Self> {
            let name = std::ffi::CString::new(super::CONV1D_KERNEL).unwrap();
            let mut e: i32 = 0;
            let kernel = unsafe { clCreateKernel(program, name.as_ptr() as *const std::ffi::c_void, &mut e) };
            if e != CL_SUCCESS || kernel == 0 {
                return Err(Error::Msg(format!(
                    "opencl conv1d: clCreateKernel({}) failed with status {e}",
                    super::CONV1D_KERNEL
                )));
            }
            Ok(Self { kernel, queue, next: 0 })
        }

        fn arg<T: Copy>(&mut self, v: &T) -> Result<&mut Self> {
            let rc = unsafe {
                clSetKernelArg(
                    self.kernel,
                    self.next,
                    std::mem::size_of::<T>(),
                    v as *const T as *const std::ffi::c_void,
                )
            };
            if rc != CL_SUCCESS {
                return Err(Error::Msg(format!(
                    "opencl conv1d: clSetKernelArg(index {}) failed with status {rc}",
                    self.next
                )));
            }
            self.next += 1;
            Ok(self)
        }

        fn buffer(&mut self, b: usize) -> Result<&mut Self> {
            self.arg(&b)
        }

        fn enqueue(self, global: usize) -> Result<()> {
            if global == 0 {
                return Ok(());
            }
            let rc = unsafe {
                clEnqueueNDRangeKernel(
                    self.queue,
                    self.kernel,
                    1,
                    std::ptr::null(),
                    &global,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                )
            };
            if rc != CL_SUCCESS {
                return Err(Error::Msg(format!(
                    "opencl conv1d: clEnqueueNDRangeKernel failed with status {rc}"
                )));
            }
            // Counted in the backend's own metric, so a caller (and a test)
            // can prove the work really ran on the device.
            candle_core::opencl_backend::kernels::note_native_exec();
            Ok(())
        }
    }

    /// `y = silu?(conv1d(x, w) + b?)`; `y` receives `t * c` f32.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        ctx: &Ctx,
        x: usize,
        w: usize,
        b: Option<usize>,
        out: usize,
        t: usize,
        c: usize,
        x_off: usize,
        w_off: usize,
        b_off: Option<usize>,
        cfg: super::Conv1d,
    ) -> Result<()> {
        let n = match t.checked_mul(c) {
            Some(n) => n,
            None => return Err(Error::Msg("opencl conv1d: output size overflow".into())),
        };
        let (t, c, x_off, w_off) = (
            i32_of(t, "T")?,
            i32_of(c, "C")?,
            i32_of(x_off, "x offset")?,
            i32_of(w_off, "w offset")?,
        );
        let b_off = i32_of(b_off.unwrap_or(0), "b offset")?;
        let b_on = b.is_some() as i32;
        let silu = cfg.silu() as i32;
        // A missing bias binds `x`: the kernel only dereferences `bp` when
        // `b_on` is set, and `x` is read-only here.
        let b_buf = b.unwrap_or(x);
        with_program(ctx, cfg.d_conv(), |program| {
            let mut launch = Launch::new(program, ctx.queue)?;
            launch
                .buffer(x)?
                .buffer(w)?
                .buffer(b_buf)?
                .buffer(out)?
                .arg(&t)?
                .arg(&c)?
                .arg(&x_off)?
                .arg(&w_off)?
                .arg(&b_off)?
                .arg(&b_on)?
                .arg(&silu)?;
            launch.enqueue(n)
        })
    }

    fn i32_of(v: usize, what: &str) -> Result<i32> {
        i32::try_from(v).map_err(|_| {
            Error::Msg(format!(
                "opencl conv1d: {what} = {v} exceeds the 32-bit kernel index range"
            ))
        })
    }
}

#[cfg(feature = "opencl")]
pub use opencl_kernel::run as run_conv1d_opencl;

// ─── Native launch (Vulkan) ─────────────────────────────────────────────────

// ─── The Vulkan kernel (source shipped, launch pending a vendor API) ────────
//
// [`vulkan_source`] is the finished GLSL for this op.  It cannot be
// dispatched from `src/` today: the only way to launch a compute pipeline is
// `VulkanDevice::launch` / `launch_spirv`, which are `pub(crate)` in
// `vendor/candle-core/src/vulkan_backend/kernels.rs`, and the GLSL→SPIR-V
// entry point (`glsl_to_spirv`) is private to that module.  Re-exporting a
// kernel from `src/` therefore needs a one-word vendor change
// (`pub(crate) fn launch` → `pub fn launch`) that this task's scope excludes
// (`vendor/` is read-only here, and it is tracked, so the edit would ship).
//
// Until then `Conv1dOp::vulkan_fwd` deliberately takes the trait default: a
// single upload, the CPU reference, and a single download — correct on every
// Mamba-2 shape, counted in `fallback_count()`, and never silently wrong.
// The GLSL below is what `run_conv1d_vulkan` would submit: `Push` fields in
// `PC` order, one invocation per `(t, c)`, `D_CONV` compiled in, and
// `x`/`w`/`b`/`y` on bindings 0..3 exactly like `k_fused_mhc`.

/// Whether a Vulkan device can run [`vulkan_source`] natively.  Always false
/// in this tree: see the module comment above.
pub fn vulkan_native_available() -> bool {
    false
}

/// Group counts for `n` invocations of `WG` lanes, spread over x and y — the
/// 2-D grid `vulkan_source`'s `gid()` assumes (same shape as the backend's
/// own `row_grid`/`grid` helpers).
pub fn vulkan_groups(n: usize, wg: usize, max_groups_x: u32) -> [u32; 3] {
    let groups = n.div_ceil(wg.max(1)).max(1);
    // Clamp x first, then cover the rest on y: `gx * gy >= groups` with the
    // fewest idle lanes, which is what the backend's own `grid()` does.
    let gx = groups.min(max_groups_x.max(1) as usize).max(1);
    let gy = groups.div_ceil(gx);
    debug_assert!(gx * gy >= groups);
    [gx as u32, gy as u32, 1]
}
