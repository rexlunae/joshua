//! Native Vulkan kernels: pipeline cache, command batching, argument
//! marshalling and one launch helper per kernel in [`super::glsl`].
//!
//! Every kernel is compiled once per device (GLSL → SPIR-V through `naga`,
//! then a compute pipeline) and cached by name.  Launches are *recorded*
//! into one open command buffer per device and submitted lazily: the only
//! synchronisation points are host read-backs (`to_cpu_storage`), an
//! explicit `synchronize`, and the batch filling up.  Each dispatch is
//! followed by a compute→compute memory barrier, so consecutive kernels see
//! each other's writes in order; device buffers dropped while a batch is
//! open are released only after that batch has completed.
//!
//! Native execution is on by default; `JOSHUA_VULKAN_NATIVE=0` forces every
//! op through the CPU round-trip (a correctness reference), and
//! `JOSHUA_VULKAN_TRACE=1` logs each op that still falls back.
// Kernel launchers mirror their kernel's argument list one-to-one.
#![allow(clippy::too_many_arguments)]

use super::glsl;
use super::{Alloc, VulkanDevice};
use crate::{Error, Layout, Result};
use ash::vk;
use std::collections::HashMap;

// ─── Gating / diagnostics ────────────────────────────────────────────────────

static NATIVE_EXEC: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FALLBACKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Whether native kernels run (default) or every op takes the CPU
/// round-trip (`JOSHUA_VULKAN_NATIVE=0`).
pub fn native_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("JOSHUA_VULKAN_NATIVE") {
        Ok(s) => !(s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off")),
        Err(_) => true,
    })
}

fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("JOSHUA_VULKAN_TRACE"), Ok(s) if s == "1"))
}

/// Number of native kernel dispatches so far.
pub fn native_exec_count() -> usize {
    NATIVE_EXEC.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn note_native_exec() {
    NATIVE_EXEC.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Count of ops that went through the CPU round-trip.
pub fn fallback_count() -> usize {
    FALLBACKS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Record (and, under `JOSHUA_VULKAN_TRACE`, log) an op that fell back to
/// the CPU path.  `why` is `None` when no native kernel exists for the case
/// and `Some(err)` when the kernel failed.
pub fn note_fallback(op: &str, why: Option<&Error>) {
    FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if trace_enabled() {
        match why {
            Some(e) => eprintln!("[vulkan] {op}: native kernel failed, using CPU round-trip: {e}"),
            None => eprintln!("[vulkan] {op}: no native kernel for this case, using CPU round-trip"),
        }
    }
}

// ─── Index descriptor ────────────────────────────────────────────────────────

pub const MAXD: usize = glsl::MAXD;

/// Output shape plus up to three input stride sets, written to the
/// parameter buffer of a launch.  Layout must match the `Idx` struct in the
/// GLSL prelude (std430: tightly packed ints).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Idx {
    pub nd: i32,
    pub dims: [i32; MAXD],
    pub s0: [i32; MAXD],
    pub s1: [i32; MAXD],
    pub s2: [i32; MAXD],
    pub o0: i32,
    pub o1: i32,
    pub o2: i32,
}

impl Idx {
    pub fn unit() -> Self {
        Idx { nd: 1, dims: [1; MAXD], s0: [0; MAXD], s1: [0; MAXD], s2: [0; MAXD], o0: 0, o1: 0, o2: 0 }
    }

    pub fn new(dims: &[usize]) -> Result<Self> {
        if dims.len() > MAXD {
            return Err(Error::Msg(format!("vulkan: rank {} exceeds the kernel limit of {MAXD}", dims.len())));
        }
        let mut ix = Idx { nd: dims.len().max(1) as i32, ..Idx::unit() };
        for (i, &d) in dims.iter().enumerate() {
            ix.dims[i] = to_i32(d)?;
        }
        Ok(ix)
    }

    /// Fill stride set `which` (0..3) from a layout whose shape is this
    /// descriptor's output shape.
    pub fn with_layout(self, which: usize, l: &Layout) -> Result<Self> {
        if l.dims().len() > MAXD {
            return Err(Error::Msg("vulkan: layout rank exceeds the kernel limit".into()));
        }
        self.with_strides(which, l.stride(), l.start_offset())
    }

    /// Stride set `which` from explicit strides and offset.
    pub fn with_strides(mut self, which: usize, strides: &[usize], offset: usize) -> Result<Self> {
        let (s, o) = match which {
            0 => (&mut self.s0, &mut self.o0),
            1 => (&mut self.s1, &mut self.o1),
            _ => (&mut self.s2, &mut self.o2),
        };
        for (i, &st) in strides.iter().enumerate() {
            s[i] = to_i32(st)?;
        }
        *o = to_i32(offset)?;
        Ok(self)
    }

    fn bytes(&self) -> &[u8] {
        // # Safety: repr(C) plain-old-data.
        unsafe { std::slice::from_raw_parts(self as *const Idx as *const u8, std::mem::size_of::<Idx>()) }
    }
}

pub fn to_i32(v: usize) -> Result<i32> {
    i32::try_from(v).map_err(|_| Error::Msg(format!("vulkan: dimension {v} exceeds the 32-bit kernel index range")))
}

// ─── Pipelines ───────────────────────────────────────────────────────────────

/// Push-constant range every pipeline layout declares (the guaranteed
/// minimum `maxPushConstantsSize`).
const PUSH_BYTES: u32 = 128;

/// A compiled compute pipeline.
pub struct Pipe {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub dsl: vk::DescriptorSetLayout,
    pub nbuf: u32,
}

fn glsl_to_spirv(source: &str) -> Result<Vec<u32>> {
    use naga::back::spv;
    use naga::front::glsl::{Frontend, Options};
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    use naga::ShaderStage;

    let options = Options::from(ShaderStage::Compute);
    let mut frontend = Frontend::default();
    let module = frontend.parse(&options, source).map_err(|e| {
        Error::Msg(format!("vulkan shader parse failed: {}\n--- source ---\n{source}", e.emit_to_string(source)))
    })?;
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator
        .validate(&module)
        .map_err(|e| Error::Msg(format!("vulkan shader validation failed: {e:?}\n--- source ---\n{source}")))?;
    let mut writer = spv::Writer::new(&spv::Options::default())
        .map_err(|e| Error::Msg(format!("vulkan spv writer init failed: {e:?}")))?;
    let mut words: Vec<u32> = Vec::new();
    let pipeline_options = spv::PipelineOptions { shader_stage: ShaderStage::Compute, entry_point: "main".to_string() };
    writer
        .write(&module, &info, Some(&pipeline_options), &None, &mut words)
        .map_err(|e| Error::Msg(format!("vulkan spv write failed: {e:?}")))?;
    Ok(words)
}

/// The descriptor-set + pipeline layouts for `nbuf` tensor bindings plus
/// the parameter buffer, shared by every pipeline with that binding count.
pub struct Layouts {
    pub dsl: vk::DescriptorSetLayout,
    pub layout: vk::PipelineLayout,
}

pub(super) fn create_layouts(device: &ash::Device, nbuf: u32) -> Result<Layouts> {
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..=nbuf)
        .map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let dsl = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }
        .map_err(|e| Error::Msg(format!("vulkan create_descriptor_set_layout failed: {e:?}")))?;
    let push = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: PUSH_BYTES }];
    let set_layouts = [dsl];
    let layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(&push),
            None,
        )
    }
    .map_err(|e| {
        unsafe { device.destroy_descriptor_set_layout(dsl, None) };
        Error::Msg(format!("vulkan create_pipeline_layout failed: {e:?}"))
    })?;
    Ok(Layouts { dsl, layout })
}

pub(super) fn create_pipe(device: &ash::Device, layouts: &Layouts, nbuf: u32, source: &str) -> Result<Pipe> {
    let spirv = glsl_to_spirv(source)?;
    let module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spirv), None) }
        .map_err(|e| Error::Msg(format!("vulkan create_shader_module failed: {e:?}")))?;
    let entry = std::ffi::CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&entry);
    let info = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(layouts.layout)];
    let pipelines = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &info, None) };
    unsafe { device.destroy_shader_module(module, None) };
    let pipeline = match pipelines {
        Ok(p) => p[0],
        Err((_, e)) => return Err(Error::Msg(format!("vulkan create_compute_pipelines failed: {e:?}"))),
    };
    Ok(Pipe { pipeline, layout: layouts.layout, dsl: layouts.dsl, nbuf })
}

// ─── Command batching ────────────────────────────────────────────────────────

/// Bytes of parameter-buffer ring per batch (each launch takes one
/// [`PARAM_SLOT`]-byte slot).
const PARAMS_BYTES: usize = 2 << 20;
/// Slot size: two `Idx` descriptors, rounded to the largest
/// `minStorageBufferOffsetAlignment` (256).
const PARAM_SLOT: usize = 512;
/// Launches recorded before a batch is submitted on its own.
pub(super) const MAX_LAUNCHES: usize = 1024;

/// A device buffer that must outlive the batch that references it.
pub struct Garbage {
    pub buffer: vk::Buffer,
    pub alloc: Alloc,
}

/// The recording state of one device: a command buffer, its descriptor
/// pool, the parameter ring and the buffers awaiting release.
pub struct Exec {
    device: ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    desc_pool: vk::DescriptorPool,
    params_buf: vk::Buffer,
    params_mem: vk::DeviceMemory,
    params_map: *mut u8,
    params_off: usize,
    /// The device's fault buffer (see `glsl.rs`, one word per thread slot
    /// of `crate::fault_slot`): indexing kernels set the calling thread's
    /// slot on an out-of-range id; [`Exec::take_fault`] reads and clears it.
    fault_buf: vk::Buffer,
    fault_mem: vk::DeviceMemory,
    fault_map: *mut u8,
    pub(super) recording: bool,
    launches: usize,
    pub(super) garbage: Vec<Garbage>,
}

unsafe impl Send for Exec {}

impl Exec {
    pub(super) fn new(device: &ash::Device, queue: vk::Queue, queue_family: u32, host_mem_type: u32) -> Result<Self> {
        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| Error::Msg(format!("vulkan create_command_pool failed: {e:?}")))?;
        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1),
            )
        }
        .map_err(|e| Error::Msg(format!("vulkan allocate_command_buffers failed: {e:?}")))?[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| Error::Msg(format!("vulkan create_fence failed: {e:?}")))?;
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: (MAX_LAUNCHES * 8) as u32 }];
        let desc_pool = unsafe {
            device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(MAX_LAUNCHES as u32).pool_sizes(&sizes), None)
        }
        .map_err(|e| Error::Msg(format!("vulkan create_descriptor_pool failed: {e:?}")))?;
        // The parameter ring: one host-visible buffer, bump-allocated.
        let params_buf = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(PARAMS_BYTES as u64)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(|e| Error::Msg(format!("vulkan create_buffer(params) failed: {e:?}")))?;
        let req = unsafe { device.get_buffer_memory_requirements(params_buf) };
        let params_mem = unsafe {
            device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(host_mem_type), None)
        }
        .map_err(|e| Error::Msg(format!("vulkan allocate_memory(params) failed: {e:?}")))?;
        unsafe { device.bind_buffer_memory(params_buf, params_mem, 0) }
            .map_err(|e| Error::Msg(format!("vulkan bind_buffer_memory(params) failed: {e:?}")))?;
        let params_map = unsafe { device.map_memory(params_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
            .map_err(|e| Error::Msg(format!("vulkan map_memory(params) failed: {e:?}")))? as *mut u8;
        let (fault_buf, fault_mem, fault_map) = host_buffer(device, FAULT_BYTES as u64, host_mem_type, "fault")?;
        unsafe { std::ptr::write_bytes(fault_map, 0, FAULT_BYTES) };
        Ok(Exec {
            device: device.clone(),
            queue,
            cmd_pool,
            cmd,
            fence,
            desc_pool,
            params_buf,
            params_mem,
            params_map,
            params_off: 0,
            fault_buf,
            fault_mem,
            fault_map,
            recording: false,
            launches: 0,
            garbage: Vec::new(),
        })
    }

    fn begin(&mut self) -> Result<()> {
        if self.recording {
            return Ok(());
        }
        unsafe {
            self.device
                .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| Error::Msg(format!("vulkan reset_command_buffer failed: {e:?}")))?;
            self.device
                .begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .map_err(|e| Error::Msg(format!("vulkan begin_command_buffer failed: {e:?}")))?;
        }
        self.recording = true;
        self.launches = 0;
        self.params_off = 0;
        Ok(())
    }

    /// Submit the open batch (if any), wait for it, and release everything
    /// it referenced.  Returns the buffers to free to the caller (which
    /// holds the allocator).
    pub(super) fn flush(&mut self) -> Result<Vec<Garbage>> {
        if self.recording {
            self.recording = false;
            unsafe {
                // Make the batch's writes visible to the host.
                let barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)];
                self.device.cmd_pipeline_barrier(
                    self.cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                );
                self.device.end_command_buffer(self.cmd).map_err(|e| Error::Msg(format!("vulkan end_command_buffer failed: {e:?}")))?;
                let cmds = [self.cmd];
                let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
                self.device
                    .queue_submit(self.queue, &submit, self.fence)
                    .map_err(|e| Error::Msg(format!("vulkan queue_submit failed: {e:?}")))?;
                self.device
                    .wait_for_fences(&[self.fence], true, u64::MAX)
                    .map_err(|e| Error::Msg(format!("vulkan wait_for_fences failed: {e:?}")))?;
                self.device.reset_fences(&[self.fence]).map_err(|e| Error::Msg(format!("vulkan reset_fences failed: {e:?}")))?;
                self.device
                    .reset_descriptor_pool(self.desc_pool, vk::DescriptorPoolResetFlags::empty())
                    .map_err(|e| Error::Msg(format!("vulkan reset_descriptor_pool failed: {e:?}")))?;
            }
        }
        Ok(std::mem::take(&mut self.garbage))
    }

    /// Record one dispatch.  `params` is written to the ring and bound last.
    fn record(&mut self, pipe: &Pipe, bufs: &[Buf], push: &[u8], params: &[u8], groups: [u32; 3]) -> Result<bool> {
        self.begin()?;
        let dev = &self.device;
        // Parameter slot.
        if self.params_off + PARAM_SLOT > PARAMS_BYTES || params.len() > PARAM_SLOT {
            return Err(Error::Msg("vulkan: parameter ring exhausted".into()));
        }
        let slot = self.params_off;
        unsafe { std::ptr::copy_nonoverlapping(params.as_ptr(), self.params_map.add(slot), params.len()) };
        self.params_off += PARAM_SLOT;
        // Descriptor set.
        let set_layouts = [pipe.dsl];
        let set = unsafe { dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.desc_pool).set_layouts(&set_layouts)) }
            .map_err(|e| Error::Msg(format!("vulkan allocate_descriptor_sets failed: {e:?}")))?[0];
        let mut infos: Vec<vk::DescriptorBufferInfo> =
            bufs.iter().map(|b| vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: b.bytes.max(4) }).collect();
        infos.push(vk::DescriptorBufferInfo { buffer: self.params_buf, offset: slot as u64, range: PARAM_SLOT as u64 });
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, bi)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(bi))
            })
            .collect();
        unsafe {
            dev.update_descriptor_sets(&writes, &[]);
            dev.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            dev.cmd_bind_descriptor_sets(self.cmd, vk::PipelineBindPoint::COMPUTE, pipe.layout, 0, &[set], &[]);
            let mut pc = [0u8; PUSH_BYTES as usize];
            pc[..push.len()].copy_from_slice(push);
            dev.cmd_push_constants(self.cmd, pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
            dev.cmd_dispatch(self.cmd, groups[0], groups[1], groups[2]);
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            dev.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
        }
        self.launches += 1;
        note_native_exec();
        Ok(self.launches >= MAX_LAUNCHES)
    }
}

impl Exec {
    /// The fault word as a kernel argument.
    pub(super) fn fault(&self) -> Buf {
        Buf { buffer: self.fault_buf, bytes: FAULT_BYTES as u64 }
    }

    /// Whether an indexing kernel launched by this thread flagged an
    /// out-of-range id since the last call (the flag is cleared).
    /// Meaningful after a flush.
    pub(super) fn take_fault(&mut self) -> bool {
        let p = unsafe { (self.fault_map as *mut u32).add(crate::fault_slot::current()) };
        let v = unsafe { std::ptr::read_volatile(p) };
        if v != 0 {
            unsafe { std::ptr::write_volatile(p, 0) };
        }
        v != 0
    }
}

/// Size of the fault buffer (one word per thread slot).
const FAULT_BYTES: usize = crate::fault_slot::BYTES;

/// A small host-visible buffer bound to its own allocation and mapped.
fn host_buffer(device: &ash::Device, size: u64, mem_type: u32, what: &str) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8)> {
    let buf = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::STORAGE_BUFFER).sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .map_err(|e| Error::Msg(format!("vulkan create_buffer({what}) failed: {e:?}")))?;
    let req = unsafe { device.get_buffer_memory_requirements(buf) };
    let mem = unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mem_type), None) }
        .map_err(|e| Error::Msg(format!("vulkan allocate_memory({what}) failed: {e:?}")))?;
    unsafe { device.bind_buffer_memory(buf, mem, 0) }.map_err(|e| Error::Msg(format!("vulkan bind_buffer_memory({what}) failed: {e:?}")))?;
    let map = unsafe { device.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
        .map_err(|e| Error::Msg(format!("vulkan map_memory({what}) failed: {e:?}")))? as *mut u8;
    Ok((buf, mem, map))
}

impl Drop for Exec {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.unmap_memory(self.fault_mem);
            self.device.destroy_buffer(self.fault_buf, None);
            self.device.free_memory(self.fault_mem, None);
            self.device.unmap_memory(self.params_mem);
            self.device.destroy_buffer(self.params_buf, None);
            self.device.free_memory(self.params_mem, None);
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
        }
    }
}

/// A buffer argument: the handle and the bytes the kernel may address.
#[derive(Clone, Copy)]
pub struct Buf {
    pub buffer: vk::Buffer,
    pub bytes: u64,
}

/// Push-constant builder.
#[derive(Default)]
pub struct Push(Vec<u8>);

impl Push {
    pub fn i(mut self, v: i32) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u(mut self, v: u32) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn f(mut self, v: f32) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn us(self, v: usize) -> Result<Self> {
        Ok(self.i(to_i32(v)?))
    }
}

/// Group counts for `n` invocations of 1-D groups of `wg`, spread over the
/// x and y grid axes (`z` is the caller's batch axis).
fn grid(n: usize, wg: usize, z: usize, max_x: u32) -> [u32; 3] {
    let groups = n.div_ceil(wg).max(1);
    let gx = groups.min(max_x as usize).max(1);
    let gy = groups.div_ceil(gx);
    [gx as u32, gy as u32, z.max(1) as u32]
}

/// Group counts for `rows` one-group-per-row launches.
fn row_grid(rows: usize, z: usize, max_x: u32) -> [u32; 3] {
    let gx = rows.min(max_x as usize).max(1);
    let gy = rows.div_ceil(gx);
    [gx as u32, gy as u32, z.max(1) as u32]
}

impl VulkanDevice {
    /// Launch `key` (compiling it with `src` on first use) over `groups`.
    pub(crate) fn launch(&self, key: &str, src: impl FnOnce(usize) -> String, bufs: &[Buf], push: Push, ix: &[Idx], groups: [u32; 3]) -> Result<()> {
        if groups.contains(&0) {
            return Ok(());
        }
        let max_range = self.limits().max_storage_range;
        for b in bufs {
            if b.bytes > max_range {
                return Err(Error::Msg(format!(
                    "vulkan: a {}-byte buffer exceeds the device's maxStorageBufferRange ({max_range})",
                    b.bytes
                )));
            }
        }
        let nbuf = bufs.len() as u32;
        let pipe = self.pipeline(key, nbuf, src)?;
        let mut params = Vec::with_capacity(2 * std::mem::size_of::<Idx>());
        let unit = Idx::unit();
        params.extend_from_slice(ix.first().unwrap_or(&unit).bytes());
        params.extend_from_slice(ix.get(1).unwrap_or(&unit).bytes());
        let mut exec = self.exec();
        let full = exec.record(&pipe, bufs, &push.0, &params, groups)?;
        if full {
            let garbage = exec.flush()?;
            self.free_garbage(garbage);
        }
        Ok(())
    }

    /// The cached pipeline for `key`, compiled with the device's `WG`.
    fn pipeline(&self, key: &str, nbuf: u32, src: impl FnOnce(usize) -> String) -> Result<std::sync::Arc<Pipe>> {
        let mut pipes = self.pipes();
        if let Some(p) = pipes.get(key) {
            return Ok(p.clone());
        }
        let wg = self.limits().wg;
        let layouts = self.layouts(nbuf)?;
        let pipe = create_pipe(self.ash(), &layouts, nbuf, &src(wg))?;
        let pipe = std::sync::Arc::new(pipe);
        pipes.insert(key.to_string(), pipe.clone());
        Ok(pipe)
    }
}

// ─── Op codes (must match glsl.rs) ───────────────────────────────────────────

pub fn unary_code(name: &str) -> Option<i32> {
    Some(match name {
        "exp" => 0,
        "log" => 1,
        "sin" => 2,
        "cos" => 3,
        "tanh" => 4,
        "neg" => 5,
        "recip" => 6,
        "sqr" => 7,
        "sqrt" => 8,
        "gelu" => 9,
        "gelu_erf" => 10,
        "erf" => 11,
        "silu" => 12,
        "abs" => 13,
        "ceil" => 14,
        "floor" => 15,
        "round" => 16,
        "relu" => 17,
        "sign" => 18,
        "sigmoid" => 19,
        _ => return None,
    })
}

const OP_AFFINE: i32 = 20;
const OP_POWF: i32 = 21;
const OP_ELU: i32 = 22;

pub fn binary_code(name: &str) -> Option<i32> {
    Some(match name {
        "add" => 0,
        "sub" => 1,
        "mul" => 2,
        "div" => 3,
        "minimum" => 4,
        "maximum" => 5,
        _ => return None,
    })
}

pub fn cmp_code(op: crate::op::CmpOp) -> i32 {
    use crate::op::CmpOp;
    match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Le => 2,
        CmpOp::Ge => 3,
        CmpOp::Lt => 4,
        CmpOp::Gt => 5,
    }
}

pub const RED_SUM: i32 = 0;
pub const RED_MAX: i32 = 1;
pub const RED_MIN: i32 = 2;

// ─── Elementwise ─────────────────────────────────────────────────────────────

fn unary_launch(d: &VulkanDevice, op: i32, f0: f32, f1: f32, x: Buf, out: Buf, n: usize, l: &Layout) -> Result<()> {
    let (contig, off, ix) = match l.contiguous_offsets() {
        Some((o1, _)) => (1, o1, Idx::unit()),
        None => (0, 0, Idx::new(l.dims())?.with_layout(0, l)?),
    };
    let push = Push::default().us(n)?.i(contig).us(off)?.i(op).f(f0).f(f1);
    let lim = d.limits();
    d.launch("unary", glsl::k_unary, &[x, out], push, &[ix], grid(n, lim.wg, 1, lim.max_groups_x))
}

pub fn run_unary(d: &VulkanDevice, op: i32, x: Buf, out: Buf, n: usize, l: &Layout) -> Result<()> {
    unary_launch(d, op, 0.0, 0.0, x, out, n, l)
}

pub fn run_affine(d: &VulkanDevice, x: Buf, out: Buf, n: usize, l: &Layout, mul: f32, add: f32) -> Result<()> {
    unary_launch(d, OP_AFFINE, mul, add, x, out, n, l)
}

pub fn run_powf(d: &VulkanDevice, x: Buf, out: Buf, n: usize, l: &Layout, e: f32) -> Result<()> {
    unary_launch(d, OP_POWF, e, 0.0, x, out, n, l)
}

pub fn run_elu(d: &VulkanDevice, x: Buf, out: Buf, n: usize, l: &Layout, alpha: f32) -> Result<()> {
    unary_launch(d, OP_ELU, alpha, 0.0, x, out, n, l)
}

/// f32 binary op; `u32` selects the integer kernel.
pub fn run_binary(d: &VulkanDevice, op: i32, a: Buf, b: Buf, out: Buf, n: usize, la: &Layout, lb: &Layout, u32: bool) -> Result<()> {
    let (contig, oa, ob, ix) = match (la.contiguous_offsets(), lb.contiguous_offsets()) {
        (Some((oa, _)), Some((ob, _))) => (1, oa, ob, Idx::unit()),
        _ => (0, 0, 0, Idx::new(la.dims())?.with_layout(0, la)?.with_layout(1, lb)?),
    };
    let push = Push::default().us(n)?.i(contig).us(oa)?.us(ob)?.i(op);
    let lim = d.limits();
    let key = if u32 { "binary_u32" } else { "binary_f32" };
    d.launch(key, |wg| glsl::k_binary(wg, u32), &[a, b, out], push, &[ix], grid(n, lim.wg, 1, lim.max_groups_x))
}

pub fn run_cmp(d: &VulkanDevice, op: i32, a: Buf, b: Buf, out: Buf, n: usize, la: &Layout, lb: &Layout, u32: bool) -> Result<()> {
    let ix = Idx::new(la.dims())?.with_layout(0, la)?.with_layout(1, lb)?;
    let push = Push::default().us(n)?.i(op);
    let lim = d.limits();
    let key = if u32 { "cmp_u32" } else { "cmp_f32" };
    d.launch(key, |wg| glsl::k_cmp(wg, u32), &[a, b, out], push, &[ix], grid(n.div_ceil(4), lim.wg, 1, lim.max_groups_x))
}

/// `where_cond` with a u8 or u32 condition over 4- or 8-byte payloads.
pub fn run_where(d: &VulkanDevice, cond: Buf, t: Buf, f: Buf, out: Buf, n: usize, lc: &Layout, lt: &Layout, lf: &Layout, cond_u32: bool, elem8: bool) -> Result<()> {
    let ix = Idx::new(lc.dims())?.with_layout(0, lc)?.with_layout(1, lt)?.with_layout(2, lf)?;
    let push = Push::default().us(n)?;
    let lim = d.limits();
    let key = format!("where_{}_{}", if cond_u32 { "u32" } else { "u8" }, if elem8 { 8 } else { 4 });
    d.launch(&key, |wg| glsl::k_where(wg, cond_u32, elem8), &[cond, t, f, out], push, &[ix], grid(n, lim.wg, 1, lim.max_groups_x))
}

/// Strided copy of `n` elements (`elem` bytes each) described by `l` into
/// `dst` at element offset `dst_off`, contiguous.
pub fn run_copy_strided(d: &VulkanDevice, elem: usize, src: Buf, dst: Buf, n: usize, l: &Layout, dst_off: usize) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let ix = Idx::new(l.dims())?.with_layout(0, l)?;
    let lim = d.limits();
    match elem {
        4 | 8 => {
            let elem8 = elem == 8;
            let push = Push::default().us(n)?.us(dst_off)?;
            let key = if elem8 { "copy_s8" } else { "copy_s4" };
            d.launch(key, |wg| glsl::k_copy_wide(wg, elem8), &[src, dst], push, &[ix], grid(n, lim.wg, 1, lim.max_groups_x))
        }
        1 | 2 => {
            let per = 4 / elem;
            let w0 = dst_off / per;
            let nw = (dst_off + n).div_ceil(per) - w0;
            let push = Push::default().us(n)?.us(dst_off)?.us(w0)?.us(nw)?;
            let key = if elem == 1 { "copy_s1" } else { "copy_s2" };
            d.launch(key, |wg| glsl::k_copy_packed(wg, elem), &[src, dst], push, &[ix], grid(nw, lim.wg, 1, lim.max_groups_x))
        }
        _ => Err(Error::Msg(format!("vulkan: no copy kernel for {elem}-byte elements"))),
    }
}

pub fn run_copy2d(d: &VulkanDevice, elem: usize, src: Buf, dst: Buf, d1: usize, d2: usize, src_s: usize, dst_s: usize, src_o: usize, dst_o: usize) -> Result<()> {
    if d1 == 0 || d2 == 0 {
        return Ok(());
    }
    let lim = d.limits();
    match elem {
        4 | 8 => {
            let elem8 = elem == 8;
            let push = Push::default().us(d1)?.us(d2)?.us(src_s)?.us(dst_s)?.us(src_o)?.us(dst_o)?;
            let key = if elem8 { "copy2d_8" } else { "copy2d_4" };
            d.launch(key, |wg| glsl::k_copy2d_wide(wg, elem8), &[src, dst], push, &[], grid(d1 * d2, lim.wg, 1, lim.max_groups_x))
        }
        1 | 2 => {
            let per = 4 / elem;
            let end = dst_o + (d1 - 1) * dst_s + d2;
            let w0 = dst_o / per;
            let nw = end.div_ceil(per) - w0;
            let push = Push::default().us(d1)?.us(d2)?.us(src_s)?.us(dst_s)?.us(src_o)?.us(dst_o)?.us(w0)?.us(nw)?;
            let key = if elem == 1 { "copy2d_1" } else { "copy2d_2" };
            d.launch(key, |wg| glsl::k_copy2d_packed(wg, elem), &[src, dst], push, &[], grid(nw, lim.wg, 1, lim.max_groups_x))
        }
        _ => Err(Error::Msg(format!("vulkan: no copy2d kernel for {elem}-byte elements"))),
    }
}

/// Fill the elements addressed by `l` with the bit pattern `bits` (element
/// size `elem`).  1- and 2-byte fills need a contiguous layout.
pub fn run_fill(d: &VulkanDevice, elem: usize, dst: Buf, n: usize, l: &Layout, bits: u64) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let lim = d.limits();
    match elem {
        4 | 8 => {
            let elem8 = elem == 8;
            let ix = Idx::new(l.dims())?.with_layout(0, l)?;
            let push = Push::default().us(n)?.u(bits as u32).u((bits >> 32) as u32);
            let key = if elem8 { "fill_8" } else { "fill_4" };
            d.launch(key, |wg| glsl::k_fill_wide(wg, elem8), &[dst], push, &[ix], grid(n, lim.wg, 1, lim.max_groups_x))
        }
        1 | 2 => {
            let Some((off, _)) = l.contiguous_offsets() else {
                return Err(Error::Msg("vulkan: strided fill of 1/2-byte elements has no kernel".into()));
            };
            let per = 4 / elem;
            let w0 = off / per;
            let nw = (off + n).div_ceil(per) - w0;
            let push = Push::default().us(n)?.us(off)?.us(w0)?.us(nw)?.u(bits as u32);
            let key = if elem == 1 { "fill_1" } else { "fill_2" };
            d.launch(key, |wg| glsl::k_fill_packed(wg, elem), &[dst], push, &[], grid(nw, lim.wg, 1, lim.max_groups_x))
        }
        _ => Err(Error::Msg(format!("vulkan: no fill kernel for {elem}-byte elements"))),
    }
}

/// The storage class of a dtype for the cast kernel.
pub fn cls_of(dtype: crate::DType) -> Option<glsl::Cls> {
    use crate::DType::*;
    Some(match dtype {
        F32 => glsl::Cls::F32,
        U32 => glsl::Cls::U32,
        I32 => glsl::Cls::I32,
        U8 => glsl::Cls::U8,
        F16 => glsl::Cls::F16,
        BF16 => glsl::Cls::BF16,
        I64 => glsl::Cls::I64,
        _ => return None,
    })
}

/// Whether a (from, to) cast has a kernel.
pub fn cast_supported(from: crate::DType, to: crate::DType) -> bool {
    match (cls_of(from), cls_of(to)) {
        (Some(f), Some(t)) => glsl::k_cast(64, f, t).is_some(),
        _ => false,
    }
}

pub fn run_cast(d: &VulkanDevice, from: crate::DType, to: crate::DType, x: Buf, out: Buf, n: usize, l: &Layout) -> Result<()> {
    let (Some(f), Some(t)) = (cls_of(from), cls_of(to)) else {
        return Err(Error::Msg(format!("vulkan: no cast kernel {from:?} -> {to:?}")));
    };
    let ix = Idx::new(l.dims())?.with_layout(0, l)?;
    let per = match to.size_in_bytes() {
        1 => 4,
        2 => 2,
        _ => 1,
    };
    let nw = n.div_ceil(per);
    let push = Push::default().us(n)?.us(nw)?;
    let lim = d.limits();
    let key = format!("cast_{from:?}_{to:?}");
    d.launch(
        &key,
        |wg| glsl::k_cast(wg, f, t).expect("cast pair checked"),
        &[x, out],
        push,
        &[ix],
        grid(nw, lim.wg, 1, lim.max_groups_x),
    )
}

// ─── Reductions ──────────────────────────────────────────────────────────────

pub fn run_reduce_last(d: &VulkanDevice, op: i32, x: Buf, out: Buf, rows: usize, cols: usize, off: usize) -> Result<()> {
    let push = Push::default().us(rows)?.us(cols)?.us(off)?.i(op);
    let lim = d.limits();
    d.launch("reduce_last", glsl::k_reduce_last, &[x, out], push, &[], row_grid(rows, 1, lim.max_groups_x))
}

pub fn run_arg_last(d: &VulkanDevice, is_max: bool, x: Buf, out: Buf, rows: usize, cols: usize, off: usize) -> Result<()> {
    let push = Push::default().us(rows)?.us(cols)?.us(off)?.i(is_max as i32);
    let lim = d.limits();
    d.launch("arg_last", glsl::k_arg_last, &[x, out], push, &[], row_grid(rows, 1, lim.max_groups_x))
}

pub fn run_reduce_generic(d: &VulkanDevice, op: i32, x: Buf, out: Buf, n: usize, ix: Idx, rd: Idx, count: usize) -> Result<()> {
    let push = Push::default().us(n)?.us(count)?.i(op);
    let lim = d.limits();
    d.launch("reduce_generic", glsl::k_reduce_generic, &[x, out], push, &[ix, rd], grid(n, lim.wg, 1, lim.max_groups_x))
}

// ─── Indexing ────────────────────────────────────────────────────────────────

/// The calling thread's fault slot, as a push constant.
fn fslot() -> i32 {
    crate::fault_slot::current() as i32
}

pub fn run_index_select(d: &VulkanDevice, elem8: bool, ids_i64: bool, src: Buf, ids: Buf, out: Buf, n: usize, left: usize, n_ids: usize, right: usize, dim_size: usize, src_off: usize, ids_off: usize) -> Result<()> {
    let push = Push::default().us(n)?.us(left)?.us(n_ids)?.us(right)?.us(dim_size)?.us(src_off)?.us(ids_off)?.i(fslot());
    let lim = d.limits();
    let key = format!("index_select_{}_{}", if ids_i64 { "i64" } else { "u32" }, if elem8 { 8 } else { 4 });
    d.launch(&key, |wg| glsl::k_index_select(wg, elem8, ids_i64), &[src, ids, out, d.fault_buf()], push, &[], grid(n, lim.wg, 1, lim.max_groups_x))
}

pub fn run_gather(d: &VulkanDevice, src: Buf, ids: Buf, out: Buf, n: usize, ix: Idx, src_dim_stride: usize, dim_size: usize) -> Result<()> {
    let push = Push::default().us(n)?.us(src_dim_stride)?.us(dim_size)?.i(fslot());
    let lim = d.limits();
    d.launch("gather_4", glsl::k_gather, &[src, ids, out, d.fault_buf()], push, &[ix], grid(n, lim.wg, 1, lim.max_groups_x))
}

/// scatter set / add: `ix` enumerates the ids space with `dim` collapsed
/// (s0 ids, s1 src, s2 dst with the dim stride zeroed); the `*_ds` are the
/// strides along `dim`.
pub fn run_scatter(d: &VulkanDevice, add: bool, dst: Buf, ids: Buf, src: Buf, n: usize, ix: Idx, n_j: usize, ids_ds: usize, src_ds: usize, dst_ds: usize, dim_size: usize) -> Result<()> {
    let push = Push::default().us(n)?.us(n_j)?.us(ids_ds)?.us(src_ds)?.us(dst_ds)?.us(dim_size)?.i(fslot());
    let lim = d.limits();
    let key = if add { "scatter_add_f32" } else { "scatter_set_4" };
    d.launch(key, |wg| glsl::k_scatter(wg, add), &[dst, ids, src, d.fault_buf()], push, &[ix], grid(n, lim.wg, 1, lim.max_groups_x))
}

pub fn run_index_add(d: &VulkanDevice, dst: Buf, ids: Buf, src: Buf, left: usize, n_ids: usize, right: usize, dim_size: usize, src_off: usize, ids_off: usize) -> Result<()> {
    let n_lr = left * right;
    let push = Push::default().us(n_lr)?.us(left)?.us(n_ids)?.us(right)?.us(dim_size)?.us(src_off)?.us(ids_off)?.i(fslot());
    let lim = d.limits();
    d.launch("index_add_f32", glsl::k_index_add, &[dst, ids, src, d.fault_buf()], push, &[], grid(n_lr, lim.wg, 1, lim.max_groups_x))
}

// ─── Dense GEMM / GEMV ───────────────────────────────────────────────────────

/// Strides of one operand of a (batched) matmul.
#[derive(Clone, Copy, Debug)]
pub struct MatStrides {
    pub row: usize,
    pub col: usize,
    pub offset: usize,
    pub batch: usize,
}

/// `C[bz] = A[bz] @ B[bz]` for `batch` matrices of `(m, k) @ (k, n)`; C is
/// written contiguous `[batch, m, n]`.  Picks the GEMV kernels for `m == 1`.
pub fn run_matmul(d: &VulkanDevice, a: Buf, b: Buf, out: Buf, (batch, m, n, k): (usize, usize, usize, usize), sa: MatStrides, sb: MatStrides) -> Result<()> {
    if batch == 0 || m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let lim = d.limits();
    if m == 1 && sa.col == 1 {
        let push = |sbx: usize| -> Result<Push> {
            Push::default().us(n)?.us(k)?.us(sbx)?.us(sa.offset)?.us(sb.offset)?.i(0).us(sa.batch)?.us(sb.batch)?.us(n)
        };
        if sb.row == 1 {
            return d.launch("gemv_nt", glsl::k_gemv_nt, &[a, b, out], push(sb.col)?, &[], row_grid(n, batch, lim.max_groups_x));
        }
        if sb.col == 1 {
            return d.launch("gemv_nn", glsl::k_gemv_nn, &[a, b, out], push(sb.row)?, &[], grid(n, lim.wg, batch, lim.max_groups_x));
        }
    }
    let tx = lim.gemm_tx;
    let tile = 4 * tx;
    let push = Push::default()
        .us(m)?
        .us(n)?
        .us(k)?
        .us(sa.row)?
        .us(sa.col)?
        .us(sb.row)?
        .us(sb.col)?
        .us(sa.offset)?
        .us(sb.offset)?
        .i(0)
        .us(sa.batch)?
        .us(sb.batch)?
        .us(m * n)?
        .i((sb.row == 1) as i32);
    let gx = n.div_ceil(tile) as u32;
    let gy = m.div_ceil(tile) as u32;
    d.launch("gemm", |wg| glsl::k_gemm(wg, tx), &[a, b, out], push, &[], [gx, gy, batch as u32])
}

// ─── Fused attention-path ops ────────────────────────────────────────────────

pub fn run_softmax_last(d: &VulkanDevice, x: Buf, out: Buf, rows: usize, cols: usize, off: usize) -> Result<()> {
    let push = Push::default().us(rows)?.us(cols)?.us(off)?;
    let lim = d.limits();
    d.launch("softmax_last", glsl::k_softmax_last, &[x, out], push, &[], row_grid(rows, 1, lim.max_groups_x))
}

pub fn run_rmsnorm(d: &VulkanDevice, x: Buf, alpha: Buf, out: Buf, rows: usize, cols: usize, off: usize, aoff: usize, eps: f32) -> Result<()> {
    let push = Push::default().us(rows)?.us(cols)?.us(off)?.us(aoff)?.f(eps);
    let lim = d.limits();
    d.launch("rmsnorm", glsl::k_rmsnorm, &[x, alpha, out], push, &[], row_grid(rows, 1, lim.max_groups_x))
}

/// RoPE over `x [b, h, t, d]` with `cos`/`sin` `[t, d/2]` (or `[b, t, d/2]`
/// when `cs_batched`); `interleaved` selects candle's `rope_i` pairing.
pub fn run_rope(d: &VulkanDevice, interleaved: bool, x: Buf, cos: Buf, sin: Buf, out: Buf, (b, h, t, dd): (usize, usize, usize, usize), xoff: usize, coff: usize, soff: usize, cs_batched: bool) -> Result<()> {
    let n_pairs = b * h * t * (dd / 2);
    let push = Push::default().us(n_pairs)?.us(h)?.us(t)?.us(dd)?.us(xoff)?.us(coff)?.us(soff)?.i(cs_batched as i32);
    let lim = d.limits();
    let key = if interleaved { "rope_i" } else { "rope" };
    d.launch(key, |wg| glsl::k_rope(wg, interleaved), &[x, cos, sin, out], push, &[], grid(n_pairs, lim.wg, 1, lim.max_groups_x))
}

// ─── Quantized weights ───────────────────────────────────────────────────────

/// GGUF dtype code the kernels switch on (the on-disk ggml ids).
pub fn qtype_code(dtype: crate::quantized::GgmlDType) -> i32 {
    use crate::quantized::GgmlDType::*;
    match dtype {
        F32 => 0,
        F16 => 1,
        Q4_0 => 2,
        Q4_1 => 3,
        Q5_0 => 6,
        Q5_1 => 7,
        Q8_0 => 8,
        Q8_1 => 9,
        Q2K => 10,
        Q3K => 11,
        Q4K => 12,
        Q5K => 13,
        Q6K => 14,
        Q8K => 15,
        BF16 => 30,
    }
}

/// `C[m, n] = sum_k X[m, k] * W[n, k]` over block-quantized `W` (`[N, K]`).
pub fn run_qgemv(d: &VulkanDevice, dtype: crate::quantized::GgmlDType, x: Buf, w: Buf, out: Buf, m: usize, n: usize, k: usize, xoff: usize) -> Result<()> {
    let push = Push::default().us(n)?.us(k)?.i(qtype_code(dtype)).us(dtype.block_size())?.us(dtype.type_size())?.us(xoff)?.i(0).us(m)?;
    let lim = d.limits();
    d.launch("qgemv", glsl::k_qgemv, &[x, w, out], push, &[], row_grid(n, m, lim.max_groups_x))
}

/// `C[m, n] = sum_k X[m, k] * W[n, k]` over f16 / bf16 `W`.
pub fn run_hgemv(d: &VulkanDevice, bf16: bool, x: Buf, w: Buf, out: Buf, m: usize, n: usize, k: usize, xoff: usize) -> Result<()> {
    let push = Push::default().us(n)?.us(k)?.i(bf16 as i32).us(xoff)?.i(0).us(m)?;
    let lim = d.limits();
    d.launch("hgemv", glsl::k_hgemv, &[x, w, out], push, &[], row_grid(n, m, lim.max_groups_x))
}

/// Dequantize `elem_count` elements of blocks to f32.
pub fn run_dequant(d: &VulkanDevice, dtype: crate::quantized::GgmlDType, w: Buf, out: Buf, elem_count: usize) -> Result<()> {
    use crate::quantized::GgmlDType::*;
    let lim = d.limits();
    match dtype {
        F16 | BF16 => {
            let push = Push::default().us(elem_count)?.i((dtype == BF16) as i32);
            d.launch("dequant_half", glsl::k_dequant_half, &[w, out], push, &[], grid(elem_count, lim.wg, 1, lim.max_groups_x))
        }
        F32 => Err(Error::Msg("vulkan: f32 weights need no dequantization".into())),
        _ => {
            let nsub = elem_count / 32;
            let push = Push::default().us(nsub)?.i(qtype_code(dtype)).us(dtype.block_size())?.us(dtype.type_size())?;
            d.launch("dequant", glsl::k_dequant, &[w, out], push, &[], grid(nsub, lim.wg, 1, lim.max_groups_x))
        }
    }
}

/// Gather rows of an f16 / bf16 `[vocab, K]` table into f32 `[n_ids, K]`.
pub fn run_hembed(d: &VulkanDevice, bf16: bool, w: Buf, ids: Buf, out: Buf, n_ids: usize, k: usize, vocab: usize, ids_off: usize) -> Result<()> {
    let n = n_ids * k;
    let push = Push::default().us(n)?.us(k)?.i(bf16 as i32).us(ids_off)?.us(vocab)?.i(fslot());
    let lim = d.limits();
    d.launch("hembed", glsl::k_hembed, &[w, ids, out, d.fault_buf()], push, &[], grid(n, lim.wg, 1, lim.max_groups_x))
}

/// Gather rows of a block-quantized `[vocab, K]` table into f32 `[n_ids, K]`.
pub fn run_qembed(d: &VulkanDevice, dtype: crate::quantized::GgmlDType, w: Buf, ids: Buf, out: Buf, n_ids: usize, k: usize, vocab: usize, ids_off: usize) -> Result<()> {
    let push = Push::default().us(n_ids)?.us(k)?.i(qtype_code(dtype)).us(dtype.block_size())?.us(dtype.type_size())?.us(ids_off)?.us(vocab)?.i(fslot());
    let lim = d.limits();
    d.launch("qembed", glsl::k_qembed, &[w, ids, out, d.fault_buf()], push, &[], row_grid(n_ids, 1, lim.max_groups_x))
}

/// Type alias for the pipeline cache map.
pub type PipeMap = HashMap<String, std::sync::Arc<Pipe>>;
