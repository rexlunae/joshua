//! Rust-authored fused GGUF Q4_K/Q6_K/Q8_0/Q2_K decode GEMVs and native artifacts.
//!
//! One workgroup per output row, one lane per 32-weight sub-block. Packed
//! weights are read directly (144 bytes per 256 weights); no f32 weight matrix
//! or intermediate dequantization buffer is allocated. Reduction order mirrors
//! Candle's generic Vulkan QGEMV when `workgroup == 256`.
use cubecl::prelude::*;
use cubecl::server::Handle;

pub const BLOCK_BYTES: usize = 144;
pub const BLOCK_VALUES: usize = 256;

#[cube]
fn byte_at(w: &[u32], address: usize) -> u32 {
    (w[address / 4] >> ((address % 4) * 8) as u32) & 255
}

// Decode fp16 without requiring shader-f16 support. Preserve signed zero,
// subnormals and non-finite values as well as ordinary quantization scales.
#[cube]
fn half_to_float(bits: u32) -> f32 {
    let sign = (bits & 32768) << 16;
    let exponent = (bits >> 10) & 31;
    let mantissa = bits & 1023;
    if exponent == 0 {
        let value = mantissa as f32 * (1.0f32 / 16777216.0f32);
        if sign != 0 { -value } else { value }
    } else if exponent == 31 {
        f32::reinterpret(sign | 2139095040 | (mantissa << 13))
    } else {
        f32::reinterpret(sign | ((exponent + 112) << 23) | (mantissa << 13))
    }
}

#[cube(launch)]
fn q4k_gemv(
    w: &[u32],
    x: &[f32],
    y: &mut [f32],
    x_offset: u32,
    w_offset: u32,
    #[comptime] k: usize,
    #[comptime] workgroup: usize,
) {
    q4k_body(w, x, y, x_offset, w_offset, k, workgroup);
}

// Native ABI: these four words occupy Candle's existing parameter-ring slot.
// Unchecked accesses require the validated native launcher below the shader.
#[cube(launch_unchecked)]
fn q4k_native(w: &[u32], x: &[f32], y: &mut [f32], params: &[u32], #[comptime] workgroup: usize) {
    let k = params[0] as usize;
    q4k_body(w, x, y, params[2], params[3], k, workgroup);
}

#[cube]
fn q4k_body(
    w: &[u32],
    x: &[f32],
    y: &mut [f32],
    x_offset: u32,
    w_offset: u32,
    k: usize,
    #[comptime] workgroup: usize,
) {
    let row = CUBE_POS;
    let lane = UNIT_POS as usize;
    let mut partial = 0.0f32;
    let mut sub = lane;
    while sub < k / 32 {
        let block = w_offset as usize + (row * (k / 256) + sub / 8) * 144;
        let group = sub % 8;
        let header = w[block / 4];
        let d = half_to_float(header & 65535);
        let dmin = half_to_float(header >> 16);
        let scale = if group < 4 {
            byte_at(w, block + 4 + group) & 63
        } else {
            (byte_at(w, block + 8 + group) & 15) | ((byte_at(w, block + group) >> 6) << 4)
        };
        let min = if group < 4 {
            byte_at(w, block + 8 + group) & 63
        } else {
            (byte_at(w, block + 8 + group) >> 4) | ((byte_at(w, block + 4 + group) >> 6) << 4)
        };
        let dl = d * scale as f32;
        let ml = dmin * min as f32;
        let mut dot = 0.0f32;
        for i in 0..32usize {
            let q =
                (byte_at(w, block + 16 + (group / 2) * 32 + i) >> ((group % 2) * 4) as u32) & 15;
            let weight = dl * q as f32 - ml;
            dot += x[x_offset as usize + sub * 32 + i] * weight;
        }
        partial += dot;
        sub += workgroup;
    }
    let mut shared = Shared::<[f32]>::new_slice(workgroup);
    shared[lane] = partial;
    sync_cube();
    // Unroll this compile-time-sized tree. Mutating a value inferred as
    // comptime inside a runtime while loop produces invalid IR in CubeCL 0.11.
    #[unroll]
    for step in 0..8usize {
        let stride = workgroup >> (step + 1);
        if stride > 0 {
            if lane < stride {
                shared[lane] += shared[lane + stride];
            }
            sync_cube();
        }
    }
    if lane == 0 {
        y[row] = shared[0];
    }
}

// Q8_0 dense/grouped attention, Q6_K dense/head and Q2_K hyper-connection
// projections. Format 108 specializes Q8_0 to selected diagonal group outputs;
// params[3] is rows per group instead of a weight offset in that specialization.
// Read packed weights directly, including Q8_0's unaligned
// 34-byte blocks. The native ABI and reduction are shared with Q4_K.
#[cube(launch_unchecked)]
fn packed_native(
    w: &[u32],
    x: &[f32],
    y: &mut [f32],
    params: &[u32],
    #[comptime] format: u32,
    #[comptime] workgroup: usize,
) {
    let k = params[0] as usize;
    let row = CUBE_POS;
    if row < params[1] as usize {
        let input_offset = if format == 108 {
            params[2] as usize + (row / params[3] as usize) * k
        } else {
            params[2] as usize
        };
        let lane = UNIT_POS as usize;
        let mut sub = lane;
        let mut partial = 0.0f32;
        while sub < k / 32 {
            let mut dot = 0.0f32;
            if format == 8 || format == 108 {
                let mut weight_offset = params[3] as usize;
                if format == 108 {
                    weight_offset = 0;
                }
                let block = weight_offset + (row * (k / 32) + sub) * 34;
                let d = half_to_float(byte_at(w, block) | (byte_at(w, block + 1) << 8));
                for i in 0..32usize {
                    let byte = byte_at(w, block + 2 + i);
                    let signed = (byte as i32 ^ 128) - 128;
                    dot += x[input_offset + sub * 32 + i] * (d * signed as f32);
                }
            } else if format == 14 {
                let block = params[3] as usize + (row * (k / 256) + sub / 8) * 210;
                let group = sub % 8;
                let half = group / 4;
                let quarter = group % 4;
                let d = half_to_float(byte_at(w, block + 208) | (byte_at(w, block + 209) << 8));
                for i in 0..32usize {
                    let lo = byte_at(w, block + half * 64 + (quarter % 2) * 32 + i);
                    let hi = byte_at(w, block + 128 + half * 32 + i);
                    let q = ((lo >> ((quarter / 2) * 4) as u32) & 15)
                        | (((hi >> (quarter * 2) as u32) & 3) << 4);
                    let scale = byte_at(w, block + 192 + half * 8 + quarter * 2 + i / 16);
                    let signed = (scale as i32 ^ 128) - 128;
                    let weight = (d * signed as f32) * (q as i32 - 32) as f32;
                    dot += x[input_offset + sub * 32 + i] * weight;
                }
            } else {
                let block = params[3] as usize + (row * (k / 256) + sub / 8) * 84;
                let group = sub % 8;
                let d = half_to_float(byte_at(w, block + 80) | (byte_at(w, block + 81) << 8));
                let dmin = half_to_float(byte_at(w, block + 82) | (byte_at(w, block + 83) << 8));
                for i in 0..32usize {
                    let scale = byte_at(w, block + (group / 4) * 8 + (group % 4) * 2 + i / 16);
                    let dl = d * (scale & 15) as f32;
                    let ml = dmin * (scale >> 4) as f32;
                    let q = (byte_at(w, block + 16 + (group / 4) * 32 + i)
                        >> ((group % 4) * 2) as u32)
                        & 3;
                    dot += x[input_offset + sub * 32 + i] * (dl * q as f32 - ml);
                }
            }
            partial += dot;
            sub += workgroup;
        }
        let mut shared = Shared::<[f32]>::new_slice(workgroup);
        shared[lane] = partial;
        sync_cube();
        #[unroll]
        for step in 0..8usize {
            let stride = workgroup >> (step + 1);
            if stride > 0 {
                if lane < stride {
                    shared[lane] += shared[lane + stride];
                }
                sync_cube();
            }
        }
        if lane == 0 {
            y[row] = shared[0];
        }
    }
}

/// Validated single-row geometry. The launch allocates no device buffers and
/// performs no readback or synchronization. Handles must belong to `client`.
#[derive(Clone, Copy, Debug)]
pub struct GemvShape {
    n: usize,
    k: usize,
    workgroup: usize,
}

impl GemvShape {
    pub fn new(n: usize, k: usize, workgroup: usize) -> Result<Self, &'static str> {
        if n == 0 || n > 65535 || k == 0 || !k.is_multiple_of(BLOCK_VALUES) {
            return Err("require 1..=65535 output rows and a nonzero multiple-of-256 width");
        }
        if !matches!(workgroup, 32 | 64 | 128 | 256) {
            return Err("workgroup must be 32, 64, 128 or 256");
        }
        n.checked_mul(k / BLOCK_VALUES)
            .and_then(|v| v.checked_mul(BLOCK_BYTES))
            .filter(|v| *v <= u32::MAX as usize)
            .ok_or("weight addressing exceeds u32")?;
        Ok(Self { n, k, workgroup })
    }

    /// # Safety
    /// `weights`, `input`, and `output` must be disjoint allocations on this
    /// client, with at least `weight_offset + n*k/256*144`,
    /// `(input_offset + k)*4`, and `n*4` bytes respectively. Weight offset must
    /// be four-byte aligned; bytes must have the GGUF little-endian Q4_K layout.
    /// Offsets plus lengths must fit u32 addressing. Keep buffers alive until
    /// queued work completes and order writes through the same client.
    pub unsafe fn enqueue(
        self,
        client: &Client,
        weights: &Handle,
        input: &Handle,
        output: &Handle,
        weight_offset: u32,
        input_offset: u32,
    ) {
        assert_eq!(weight_offset % 4, 0);
        let weight_bytes = self.n * (self.k / 256) * 144;
        assert!(weight_bytes <= (u32::MAX - weight_offset) as usize);
        assert!(self.k <= (u32::MAX - input_offset) as usize);
        unsafe {
            q4k_gemv::launch(
                client,
                CubeCount::Static(self.n as u32, 1, 1),
                CubeDim::new_1d(self.workgroup as u32),
                BufferArg::from_raw_parts(
                    weights.clone(),
                    (weight_offset as usize + weight_bytes) / 4,
                ),
                BufferArg::from_raw_parts(input.clone(), input_offset as usize + self.k),
                BufferArg::from_raw_parts(output.clone(), self.n),
                input_offset,
                weight_offset,
                self.k,
                self.workgroup,
            );
        }
    }
}

/// Compile the native entry point once during development. Production consumes
/// the generated WGSL through Candle's Naga compiler and existing Vulkan queue.
pub fn native_wgsl(workgroup: usize) -> String {
    native_wgsl_format(workgroup, 12)
}

/// Generate a specialized packed-weight kernel (GGML dtype 8, 10, 12 or 14).
/// Selector 108 is grouped Q8_0: params = [K, N, input_offset, rows_per_group].
pub fn native_wgsl_format(workgroup: usize, format: u32) -> String {
    use cubecl::Compiler;
    assert!(matches!(format, 8 | 10 | 12 | 14 | 108));
    assert!(matches!(workgroup, 32 | 64 | 128 | 256));
    let client = cubecl::Device::default().client();
    let settings = KernelSettings::new(
        *CubeDim::new_1d(workgroup as u32),
        ExecutionMode::Unchecked,
        AddressType::U32,
    );
    let definition = if format == 12 {
        let kernel = q4k_native::Q4kNative::new(
            settings,
            client.properties_shared(),
            client.target_properties_shared(),
            BufferCompilationArg { inplace: None },
            BufferCompilationArg { inplace: None },
            BufferCompilationArg { inplace: None },
            BufferCompilationArg { inplace: None },
            workgroup,
        );
        kernel.define()
    } else {
        packed_native::PackedNative::new(
            settings,
            client.properties_shared(),
            client.target_properties_shared(),
            BufferCompilationArg { inplace: None },
            BufferCompilationArg { inplace: None },
            BufferCompilationArg { inplace: None },
            BufferCompilationArg { inplace: None },
            format,
            workgroup,
        )
        .define()
    };
    let entry = definition.settings.kernel_name.clone();
    let shader = cubecl::wgpu::WgslCompiler
        .compile(definition, &Default::default())
        .unwrap();
    let source = shader
        .to_string()
        .replace(&format!("fn {entry}("), "fn main(")
        // Local vars have implicit function address space. Naga 27 requires
        // this spelling; CubeCL's newer emitter uses explicit address space.
        .replace("var<function> ", "var ");
    assert!(source.contains("fn main("));
    // Metadata is declared by CubeCL, but unchecked fixed-buffer accesses do
    // not read it: only bindings 0..=3 form the native ABI.
    assert!(!source.contains("info_var."));
    format!(
        "// Generated by kernels-cubecl/src/bin/generate.rs with CubeCL 0.11.0.\n// Source: kernels-cubecl/src/lib.rs (native GEMV, GGML dtype {format}); do not edit.\n{source}"
    )
}
