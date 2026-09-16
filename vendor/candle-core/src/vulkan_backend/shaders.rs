//! Vulkan compute kernels for the [`super`] backend.
//!
//! GLSL kernel source is embedded here and compiled to SPIR-V at runtime with
//! `naga` (pure Rust), so no external `glslc`/`glslangValidator` is needed to
//! build or run. Each kernel argument that is a tensor lives in a
//! STORAGE_BUFFER SSBO; scalar dimensions (n / m·n·k) go in a small
//! uniform-aligned push range.
//!
//! Native execution is gated behind `JOSHUA_VULKAN_NATIVE=1` (mirroring
//! `JOSHUA_OPENCL_NATIVE`), so the default behaviour of every op is the CPU
//! fallback and these kernels only run when explicitly enabled on Vulkan
//! hardware.

use crate::{Error, Result};
use super::{VulkanDevice, VulkanStorage};
use ash::vk;

// ---------------------------------------------------------------------------
// Native-kernel gating (mirrors the OpenCL backend's opt-in).
// ---------------------------------------------------------------------------
static NATIVE_EXEC: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn env_flag() -> bool {
    std::env::var("JOSHUA_VULKAN_NATIVE").map(|v| v == "1").unwrap_or(false)
}

/// Whether native Vulkan kernels are enabled (`JOSHUA_VULKAN_NATIVE=1`).
pub fn native_enabled() -> bool {
    env_flag()
}

/// Number of native kernel dispatches so far (diagnostic counter).
pub fn native_exec_count() -> usize {
    NATIVE_EXEC.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn note_native_exec() {
    NATIVE_EXEC.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Which unary GLSL ops we can dispatch (name -> GLSL expression).
pub fn has_unary(name: &'static str) -> bool {
    matches!(name, "exp" | "sin" | "cos" | "tan" | "sqrt" | "abs" | "neg" | "ln")
}

/// Which binary GLSL ops we can dispatch (name -> GLSL expression).
pub fn has_binary(name: &'static str) -> bool {
    matches!(name, "add" | "mul" | "sub" | "div" | "pow")
}

// ---------------------------------------------------------------------------
// GLSL kernels (compute).
// ---------------------------------------------------------------------------

/// Elementwise: out[i] = f(buf[i]).
const GLSL_UNARY: &str = r#"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InBuf { float inp[]; };
layout(set = 0, binding = 1) buffer OutBuf { float outp[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    float x = inp[i];
    float r = {EXPR};
    outp[i] = r;
}
"#;

/// Affine: out[i] = a * buf[i] + b.
const GLSL_AFFINE: &str = r#"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InBuf { float inp[]; };
layout(set = 0, binding = 1) buffer OutBuf { float outp[]; };
layout(push_constant) uniform PC { float a; float b; } pc;
void main() {
    uint i = gl_GlobalInvocationID.x;
    outp[i] = pc.a * inp[i] + pc.b;
}
"#;

/// Binary: out[i] = f(lhs[i], rhs[i]).
const GLSL_BINARY: &str = r#"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InBuf0 { float lhs[]; };
layout(set = 0, binding = 1) readonly buffer InBuf1 { float rhs[]; };
layout(set = 0, binding = 2) buffer OutBuf { float outp[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    float a = lhs[i];
    float b = rhs[i];
    outp[i] = {EXPR};
}
"#;

/// Matmul (row-major, single non-batched tile): o[r*N+c] = sum_k a[r*K+k]*b[k*N+c].
const GLSL_MATMUL: &str = r#"
#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) readonly buffer InBufA { float a[]; };
layout(set = 0, binding = 1) readonly buffer InBufB { float b[]; };
layout(set = 0, binding = 2) buffer OutBuf { float o[]; };
layout(push_constant) uniform PC { uint M; uint N; uint K; } pc;
void main() {
    uint r = gl_GlobalInvocationID.y;
    uint c = gl_GlobalInvocationID.x;
    if (r >= pc.M || c >= pc.N) return;
    float acc = 0.0;
    for (uint k = 0u; k < pc.K; k++) {
        acc += a[r * pc.K + k] * b[k * pc.N + c];
    }
    o[r * pc.N + c] = acc;
}
"#;

// ---------------------------------------------------------------------------
// naga GLSL -> SPIR-V.
// ---------------------------------------------------------------------------

fn glsl_to_spirv(source: &str, entry: &str) -> Result<Vec<u32>> {
    use naga::back::spv;
    use naga::front::glsl::{Frontend, Options};
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    use naga::ShaderStage;

    let options = Options::from(ShaderStage::Compute);
    let mut frontend = Frontend::default();
    let module = frontend
        .parse(&options, source)
        .map_err(|e| Error::Msg(format!("vulkan shader parse failed: {e:?}")))?;

    let caps = Capabilities::all();
    let mut validator = Validator::new(ValidationFlags::all(), caps);
    let info = validator
        .validate(&module)
        .map_err(|e| Error::Msg(format!("vulkan shader validation failed: {e:?}")))?;

    let spv_options = spv::Options::default();
    let mut writer = spv::Writer::new(&spv_options)
        .map_err(|e| Error::Msg(format!("vulkan spv writer init failed: {e:?}")))?;
    let mut words: Vec<u32> = Vec::new();
    let pipeline_options = spv::PipelineOptions {
        shader_stage: ShaderStage::Compute,
        entry_point: entry.to_string(),
    };
    writer
        .write(&module, &info, Some(&pipeline_options), &None, &mut words)
        .map_err(|e| Error::Msg(format!("vulkan spv write failed: {e:?}")))?;
    Ok(words)
}

// ---------------------------------------------------------------------------
// One dispatch.
// ---------------------------------------------------------------------------

/// A compiled pipeline + its command pool/descriptor set, holding the rare
/// state that doesn't change per call. Rebuilt per dispatch for simplicity
/// (kernel count is tiny in a benchmark); the hot path remains the CPU fallback
/// unless native kernels are explicitly enabled.
struct Dispatch {
    device: ash::Device,
    _pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    _descriptor_set_layout: vk::DescriptorSetLayout,
    _descriptor_pool: vk::DescriptorPool,
    _command_pool: vk::CommandPool,
    _shader_module: vk::ShaderModule,
}

impl Dispatch {
    fn new(dev: &VulkanDevice, spirv: &[u32], buffer_count: u32) -> Result<Self> {
        let ash_dev = dev.device();

        let sm_ci = vk::ShaderModuleCreateInfo::default().code(spirv);
        let shader_module = unsafe { ash_dev.create_shader_module(&sm_ci, None) }
            .map_err(|e| Error::Msg(format!("vulkan create_shader_module failed: {e:?}")))?;

        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..buffer_count)
            .map(|bi| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(bi)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { ash_dev.create_descriptor_set_layout(&dsl_info, None) }
                .map_err(|e| {
                    Error::Msg(format!("vulkan create_descriptor_set_layout failed: {e:?}"))
                })?;

        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&[]);
        let pipeline_layout = unsafe { ash_dev.create_pipeline_layout(&pipeline_layout_info, None) }
            .map_err(|e| Error::Msg(format!("vulkan create_pipeline_layout failed: {e:?}")))?;

        let entry = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry);
        let compute_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        let pipeline_stack = [compute_info];
        let pipelines = unsafe {
            ash_dev.create_compute_pipelines(vk::PipelineCache::null(), &pipeline_stack, None)
        }
        .map_err(|e| Error::Msg(format!("vulkan create_compute_pipelines failed: {:?}", e.1)))?;
        let pipeline = pipelines[0];

        let pool_sizes: Vec<vk::DescriptorPoolSize> = (0..buffer_count)
            .map(|_| vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
            })
            .collect();
        let pool = unsafe {
            ash_dev.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(|e| Error::Msg(format!("vulkan create_descriptor_pool failed: {e:?}")))?;

        let cmd_pool = unsafe {
            ash_dev.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(dev.queue_family()),
                None,
            )
        }
        .map_err(|e| Error::Msg(format!("vulkan create_command_pool failed: {e:?}")))?;

        Ok(Dispatch {
            device: ash_dev.clone(),
            _pipeline_layout: pipeline_layout,
            pipeline,
            _descriptor_set_layout: descriptor_set_layout,
            _descriptor_pool: pool,
            _command_pool: cmd_pool,
            _shader_module: shader_module,
        })
    }

    fn run(
        &self,
        dev: &VulkanDevice,
        buffers: &[vk::Buffer],
        _local: (u32, u32, u32),
        global: (u32, u32, u32),
        push: &[u8],
    ) -> Result<()> {
        let ash_dev = &self.device;
        // Descriptor set (layout 0).
        let dls = &self._descriptor_set_layout;
        let mut desc_set_handle = vk::DescriptorSet::null();
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self._descriptor_pool)
            .set_layouts(std::slice::from_ref(dls));
        unsafe { ash_dev.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| Error::Msg(format!("vulkan allocate_descriptor_sets failed: {e:?}")))?
            .into_iter()
            .next()
            .map(|d| desc_set_handle = d);
        if desc_set_handle == vk::DescriptorSet::null() {
            return Err(Error::Msg("vulkan allocate_descriptor_sets empty".into()));
        }

        let buf_infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|&b| vk::DescriptorBufferInfo {
                buffer: b,
                offset: 0,
                range: vk::WHOLE_SIZE,
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, bi)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set_handle)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(bi))
            })
            .collect();
        unsafe { ash_dev.update_descriptor_sets(&writes, &[]) };

        let cmd_buf_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self._command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_bufs = unsafe { ash_dev.allocate_command_buffers(&cmd_buf_info) }
            .map_err(|e| Error::Msg(format!("vulkan allocate_command_buffers failed: {e:?}")))?;
        let cmd = cmd_bufs[0];

        unsafe {
            ash_dev.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .map_err(|e| Error::Msg(format!("vulkan begin_command_buffer failed: {e:?}")))?;

        unsafe { ash_dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline) };
        unsafe {
            ash_dev.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self._pipeline_layout,
                0,
                &[desc_set_handle],
                &[],
            )
        };
        if !push.is_empty() {
            unsafe {
                ash_dev.cmd_push_constants(
                    cmd,
                    self._pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                )
            };
        }
        unsafe { ash_dev.cmd_dispatch(cmd, global.0, global.1, global.2) };
        unsafe { ash_dev.end_command_buffer(cmd) }
            .map_err(|e| Error::Msg(format!("vulkan end_command_buffer failed: {e:?}")))?;

        let submit_cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&submit_cmds);
        unsafe { ash_dev.queue_submit(dev.queue(), &[submit], vk::Fence::null()).map_err(|e| Error::Msg(format!("vulkan queue_submit failed: {e:?}")))? };
        unsafe { ash_dev.queue_wait_idle(dev.queue()) }
            .map_err(|e| Error::Msg(format!("vulkan queue_wait_idle failed: {e:?}")))?;
        Ok(())
    }
}

impl Drop for Dispatch {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self._command_pool, None);
            self.device.destroy_descriptor_pool(self._descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self._descriptor_set_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self._pipeline_layout, None);
            self.device.destroy_shader_module(self._shader_module, None);
        }
    }
}

fn dispatch_1buf(
    dev: &VulkanDevice,
    spirv: &[u32],
    inp: &VulkanStorage,
    out: &VulkanStorage,
    n: usize,
    push: &[u8],
) -> Result<()> {
    let d = Dispatch::new(dev, spirv, 2)?;
    let n32 = n as u32;
    d.run(dev, &[inp.buffer, out.buffer], (256, 1, 1), (n32, 1, 1), push)
}

fn dispatch_2buf(
    dev: &VulkanDevice,
    spirv: &[u32],
    lhs: &VulkanStorage,
    rhs: &VulkanStorage,
    out: &VulkanStorage,
    n: usize,
    push: &[u8],
) -> Result<()> {
    let d = Dispatch::new(dev, spirv, 3)?;
    let n32 = n as u32;
    d.run(
        dev,
        &[lhs.buffer, rhs.buffer, out.buffer],
        (256, 1, 1),
        (n32, 1, 1),
        push,
    )
}

/// out[i] = a*inp[i] + b.
pub fn run_affine(
    dev: &VulkanDevice,
    inp: &VulkanStorage,
    n: usize,
    a: f32,
    b: f32,
) -> Result<VulkanStorage> {
    let spirv = glsl_to_spirv(GLSL_AFFINE, "main")?;
    let out = unsafe { dev.alloc_buffer(n * 4, crate::DType::F32, n) }?;
    let push = a
        .to_le_bytes()
        .iter()
        .chain(b.to_le_bytes().iter())
        .copied()
        .collect::<Vec<u8>>();
    dispatch_1buf(dev, &spirv, inp, &out, n, &push)?;
    Ok(out)
}

/// out[i] = op(inp[i]).
pub fn run_unary(dev: &VulkanDevice, name: &str, inp: &VulkanStorage, n: usize) -> Result<VulkanStorage> {
    let expr = match name {
        "exp" => "exp(x)",
        "sin" => "sin(x)",
        "cos" => "cos(x)",
        "tan" => "tan(x)",
        "sqrt" => "sqrt(x)",
        "abs" => "abs(x)",
        "neg" => "-x",
        "ln" => "log(x)",
        _ => return Err(Error::Msg(format!("vulkan unary {name} not wired"))),
    };
    let src = GLSL_UNARY.replace("{EXPR}", expr);
    let spirv = glsl_to_spirv(&src, "main")?;
    let out = unsafe { dev.alloc_buffer(n * 4, crate::DType::F32, n) }?;
    dispatch_1buf(dev, &spirv, inp, &out, n, &[])?;
    Ok(out)
}

/// out[i] = op(lhs[i], rhs[i]).
pub fn run_binary(
    dev: &VulkanDevice,
    name: &str,
    lhs: &VulkanStorage,
    rhs: &VulkanStorage,
    n: usize,
) -> Result<VulkanStorage> {
    let expr = match name {
        "add" => "a + b",
        "mul" => "a * b",
        "sub" => "a - b",
        "div" => "a / b",
        "pow" => "pow(a, b)",
        _ => return Err(Error::Msg(format!("vulkan binary {name} not wired"))),
    };
    let src = GLSL_BINARY.replace("{EXPR}", expr);
    let spirv = glsl_to_spirv(&src, "main")?;
    let out = unsafe { dev.alloc_buffer(n * 4, crate::DType::F32, n) }?;
    dispatch_2buf(dev, &spirv, lhs, rhs, &out, n, &[])?;
    Ok(out)
}

/// o(row-major m×n) = a(m×k) @ b(k×n), single non-batched tile.
pub fn run_matmul(
    dev: &VulkanDevice,
    a: &VulkanStorage,
    b: &VulkanStorage,
    (m, n, k): (usize, usize, usize),
) -> Result<VulkanStorage> {
    let spirv = glsl_to_spirv(GLSL_MATMUL, "main")?;
    let out = unsafe { dev.alloc_buffer(m * n * 4, crate::DType::F32, m * n) }?;
    let d = Dispatch::new(dev, &spirv, 3)?;
    let push = [m as u32, n as u32, k as u32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<u8>>();
    let gx = ((n as u32 + 15) / 16).max(1);
    let gy = ((m as u32 + 15) / 16).max(1);
    d.run(dev, &[a.buffer, b.buffer, out.buffer], (16, 16, 1), (gx, gy, 1), &push)?;
    Ok(out)
}