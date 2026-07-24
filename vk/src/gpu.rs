//! The audited Vulkan module — every raw `ash` call in the crate lives here, behind a
//! safe API (`Gpu`, `Renderer`), mirroring the discipline of `wayland/src/sys.rs` and
//! core's `memory`/`ring`: one `#![allow(unsafe_code)]` module, invariants documented
//! per block, nothing `unsafe` escapes the boundary.
//!
//! Design (see crate docs + REFERENCES.md): a **compute** pipeline, buffers only — no
//! images, no render pass, no WSI swapchain. The I420 planes are memcpy'd into a
//! host-visible storage buffer; one dispatch converts (integer BT.601, the shader in
//! `shaders/yuv2rgb.comp`) into a per-slot output storage buffer holding the linear
//! XRGB8888 framebuffer; that buffer's device memory is allocated exportable
//! (`VK_EXT_external_memory_dma_buf`) and its fd handed to the Wayland client for
//! `zwp_linux_dmabuf_v1` import as a LINEAR `wl_buffer`. A compositor consuming a
//! LINEAR dmabuf only cares about (fd, offset, stride, fourcc, modifier) — which
//! Vulkan object produced the bytes is irrelevant, so a plain storage buffer is the
//! least-machinery correct choice for v1.
//!
//! Synchronisation: each `render` submits one command buffer (dispatch + a
//! memory-visibility barrier) and **waits its fence** before the frame is presented —
//! CPU-throttled explicit sync, correct everywhere. The zero-stall path
//! (`VK_KHR_external_semaphore_fd` / linux-explicit-synchronization) is a documented
//! follow-up in REFERENCES.md.

#![allow(unsafe_code)]

use std::ffi::CStr;
use std::io::Cursor;
use std::os::fd::{FromRawFd, OwnedFd};

use ash::vk;

/// The compiled `shaders/yuv2rgb.comp` (SPIR-V). Regenerate with
/// `glslangValidator -V vk/shaders/yuv2rgb.comp -o vk/shaders/yuv2rgb.spv`
/// (glslang is in the devshell); the GLSL source is the reviewable artifact.
const YUV2RGB_SPV: &[u8] = include_bytes!("../shaders/yuv2rgb.spv");

/// Bytes of a packed I420 frame (`streamcraft-video` geometry: chroma rounds up).
pub fn i420_size(width: usize, height: usize) -> usize {
    let c = width.div_ceil(2) * height.div_ceil(2);
    width * height + 2 * c
}

fn err<T>(what: &str, e: impl std::fmt::Display) -> Result<T, String> {
    Err(format!("{what}: {e}"))
}

/// A Vulkan instance + logical device with a compute queue, optionally created with
/// the dma-buf export extensions. One per sink; cheap to keep alive.
pub struct Gpu {
    // Field order is drop order — later fields must outlive earlier ones is NOT
    // guaranteed by Rust drop order (it drops in declaration order), so `Drop` for
    // `Gpu` tears down explicitly instead of relying on field order.
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    memory_props: vk::PhysicalDeviceMemoryProperties,
    /// Both `VK_KHR_external_memory_fd` and `VK_EXT_external_memory_dma_buf` were
    /// enabled — [`Renderer`]s may allocate exportable memory.
    pub can_export: bool,
    external_fd: Option<ash::khr::external_memory_fd::Device>,
    /// Human-readable device name, for logs/diagnostics.
    pub device_name: String,
}

impl Gpu {
    /// Bring up Vulkan. With `require_export`, only devices offering the dma-buf
    /// export extensions qualify (the presentation path needs them); without it any
    /// compute-capable device does — including lavapipe, Mesa's software
    /// implementation, which is what keeps the conversion-parity tests runnable on
    /// headless CI.
    pub fn new(require_export: bool) -> Result<Gpu, String> {
        // SAFETY: `Entry::load` dlopens libvulkan.so.1 (the loader) and resolves core
        // entry points; no Vulkan objects exist yet. Failure (no loader) is an Err.
        let entry = match unsafe { ash::Entry::load() } {
            Ok(e) => e,
            Err(e) => return err("vulkan loader", e),
        };

        let app = vk::ApplicationInfo::default()
            .application_name(c"streamcraft")
            // 1.1 promotes VK_KHR_external_memory (the capability core the fd/dma-buf
            // extensions build on) to core, so we require it rather than more extensions.
            .api_version(vk::make_api_version(0, 1, 1, 0));
        let create = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: `create` and everything it points at (`app`) outlive the call.
        let instance = match unsafe { entry.create_instance(&create, None) } {
            Ok(i) => i,
            Err(e) => return err("create_instance", e),
        };

        // SAFETY: `instance` is a live instance handle created above.
        let phys = match unsafe { instance.enumerate_physical_devices() } {
            Ok(p) => p,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return err("enumerate_physical_devices", e);
            }
        };

        let wanted: [&CStr; 2] = [
            ash::khr::external_memory_fd::NAME,
            ash::ext::external_memory_dma_buf::NAME,
        ];

        // Pick the best candidate: a compute queue family is mandatory; export
        // support is mandatory only when required. Among qualifiers prefer real
        // hardware over CPU implementations (lavapipe reports CPU type).
        let mut best: Option<(vk::PhysicalDevice, u32, bool, u32, String)> = None;
        for pd in phys {
            // SAFETY: `pd` comes from the enumeration above on this instance.
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let qprops = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            let Some(qfi) = qprops
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            else {
                continue;
            };
            let exts = match unsafe { instance.enumerate_device_extension_properties(pd) } {
                Ok(e) => e,
                Err(_) => continue,
            };
            let has_export = wanted.iter().all(|w| {
                exts.iter().any(|e| {
                    e.extension_name_as_c_str().map(|n| n == *w).unwrap_or(false)
                })
            });
            if require_export && !has_export {
                continue;
            }
            // Rank: discrete > integrated > virtual > cpu > other.
            let rank = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 4,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 1,
                _ => 0,
            };
            let name = props
                .device_name_as_c_str()
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "unknown".into());
            if best.as_ref().map_or(true, |b| rank > b.3) {
                best = Some((pd, qfi as u32, has_export, rank, name));
            }
        }
        let Some((pd, queue_family, has_export, _rank, device_name)) = best else {
            unsafe { instance.destroy_instance(None) };
            return Err(if require_export {
                "no Vulkan device with compute + dma-buf export \
                 (VK_KHR_external_memory_fd + VK_EXT_external_memory_dma_buf)"
                    .into()
            } else {
                "no Vulkan device with a compute queue".into()
            });
        };

        let prio = [1.0f32];
        let qinfo = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&prio)];
        let ext_ptrs: Vec<*const i8> = if has_export {
            wanted.iter().map(|c| c.as_ptr()).collect()
        } else {
            Vec::new()
        };
        let dinfo = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qinfo)
            .enabled_extension_names(&ext_ptrs);
        // SAFETY: `pd` is a valid physical device; `dinfo` and its pointees live
        // across the call; the queue family index came from this device's properties.
        let device = match unsafe { instance.create_device(pd, &dinfo, None) } {
            Ok(d) => d,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return err("create_device", e);
            }
        };
        // SAFETY: queue family/index were used at device creation (one queue, index 0).
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let memory_props = unsafe { instance.get_physical_device_memory_properties(pd) };
        let external_fd =
            has_export.then(|| ash::khr::external_memory_fd::Device::new(&instance, &device));

        Ok(Gpu {
            entry,
            instance,
            device,
            queue,
            queue_family,
            memory_props,
            can_export: has_export,
            external_fd,
            device_name,
        })
    }

    /// The index of a memory type matching `type_bits` with all `required` flags,
    /// preferring one that also has `preferred`.
    fn find_memory_type(
        &self,
        type_bits: u32,
        required: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let types = &self.memory_props.memory_types[..self.memory_props.memory_type_count as usize];
        let matches = |i: usize, flags: vk::MemoryPropertyFlags| {
            (type_bits & (1 << i)) != 0 && types[i].property_flags.contains(flags)
        };
        (0..types.len())
            .find(|&i| matches(i, required | preferred))
            .or_else(|| (0..types.len()).find(|&i| matches(i, required)))
            .map(|i| i as u32)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: all child objects (`Renderer`s) hold `&Gpu`, so borrowck guarantees
        // they are gone; the device is idle-waited before destruction.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        let _ = &self.entry; // dropped last, unloading the loader
    }
}

/// One output slot: an exportable storage buffer holding the linear XRGB framebuffer.
struct Slot {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// Our end of the exported dma-buf. The compositor dups the fd on import, so this
    /// stays owned here and closes on drop. `None` in the non-export (test) mode.
    fd: Option<OwnedFd>,
    set: vk::DescriptorSet,
}

/// A fixed-geometry converter: I420 in (host memcpy), XRGB8888 out (per-slot
/// exportable buffers), one fence-synchronised compute dispatch per frame.
pub struct Renderer {
    width: usize,
    height: usize,
    export: bool,
    input_buffer: vk::Buffer,
    input_memory: vk::DeviceMemory,
    /// Persistent HOST_VISIBLE|COHERENT mapping of `input_memory` (never unmapped
    /// until drop). INVARIANT: valid for `input_size` bytes; written only between
    /// fence waits, so the GPU never reads while the CPU writes.
    input_ptr: *mut u8,
    input_size: usize,
    slots: Vec<Slot>,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader: vk::ShaderModule,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
}

impl Renderer {
    /// Framebuffer row stride in bytes (LINEAR, tightly packed).
    pub fn stride(&self) -> u32 {
        (self.width * 4) as u32
    }

    /// The exported dma-buf fd for `slot` (raw; ownership stays here — the Wayland
    /// import path's compositor dups it). `None` for non-export renderers.
    pub fn slot_fd(&self, slot: usize) -> Option<std::os::fd::RawFd> {
        use std::os::fd::AsRawFd;
        self.slots.get(slot).and_then(|s| s.fd.as_ref()).map(|f| f.as_raw_fd())
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Build a renderer for `width`x`height` with `nslots` output framebuffers. With
    /// `export`, output memory is allocated dma-buf-exportable (requires
    /// [`Gpu::can_export`]); without it, output memory is host-visible so tests can
    /// [`read_back`](Self::read_back).
    pub fn new(
        gpu: &Gpu,
        width: usize,
        height: usize,
        nslots: usize,
        export: bool,
    ) -> Result<Renderer, String> {
        if width == 0 || height == 0 || nslots == 0 {
            return Err("renderer: zero-sized geometry".into());
        }
        if export && !gpu.can_export {
            return Err("renderer: device cannot export dma-bufs".into());
        }
        let dev = &gpu.device;
        // Round the plane bytes up to a whole number of u32 words (the shader reads
        // the planes as a uint array).
        let input_size = i420_size(width, height).next_multiple_of(4);
        let fb_size = (width * height * 4) as vk::DeviceSize;

        // --- input buffer: host-visible, persistently mapped ---
        let binfo = vk::BufferCreateInfo::default()
            .size(input_size as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `binfo` lives across the call; device is live.
        let input_buffer = match unsafe { dev.create_buffer(&binfo, None) } {
            Ok(b) => b,
            Err(e) => return err("create input buffer", e),
        };
        // SAFETY: `input_buffer` was created on this device just above.
        let req = unsafe { dev.get_buffer_memory_requirements(input_buffer) };
        let Some(mem_type) = gpu.find_memory_type(
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::empty(),
        ) else {
            unsafe { dev.destroy_buffer(input_buffer, None) };
            return Err("no host-visible memory type for the input buffer".into());
        };
        let ainfo = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        // SAFETY: allocation info is valid; size/type from this device's requirements.
        let input_memory = match unsafe { dev.allocate_memory(&ainfo, None) } {
            Ok(m) => m,
            Err(e) => {
                unsafe { dev.destroy_buffer(input_buffer, None) };
                return err("allocate input memory", e);
            }
        };
        // SAFETY: fresh buffer + fresh memory of at least `req.size`, offset 0.
        if let Err(e) = unsafe { dev.bind_buffer_memory(input_buffer, input_memory, 0) } {
            unsafe {
                dev.destroy_buffer(input_buffer, None);
                dev.free_memory(input_memory, None);
            }
            return err("bind input memory", e);
        }
        // SAFETY: `input_memory` is HOST_VISIBLE and not already mapped; the mapping
        // stays valid until `free_memory` in Drop (persistent map).
        let input_ptr = match unsafe {
            dev.map_memory(input_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        } {
            Ok(p) => p as *mut u8,
            Err(e) => {
                unsafe {
                    dev.destroy_buffer(input_buffer, None);
                    dev.free_memory(input_memory, None);
                }
                return err("map input memory", e);
            }
        };

        // From here on, build into a partially-initialised struct so one Drop path
        // cleans up whatever exists on any later failure.
        let mut r = Renderer {
            width,
            height,
            export,
            input_buffer,
            input_memory,
            input_ptr,
            input_size,
            slots: Vec::new(),
            layout: vk::DescriptorSetLayout::null(),
            pool: vk::DescriptorPool::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            shader: vk::ShaderModule::null(),
            cmd_pool: vk::CommandPool::null(),
            cmd: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
        };
        r.init_rest(gpu, fb_size, nslots)?;
        Ok(r)
    }

    fn init_rest(&mut self, gpu: &Gpu, fb_size: vk::DeviceSize, nslots: usize) -> Result<(), String> {
        let dev = &gpu.device;

        // --- descriptor set layout: {0: planes RO, 1: pixels} storage buffers ---
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let linfo = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY (this and every create below): the info structs and their pointees
        // live across the call; handles are created and destroyed on `dev` only.
        self.layout = unsafe { dev.create_descriptor_set_layout(&linfo, None) }
            .map_err(|e| format!("descriptor layout: {e}"))?;

        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count((nslots * 2) as u32)];
        let pinfo = vk::DescriptorPoolCreateInfo::default()
            .max_sets(nslots as u32)
            .pool_sizes(&sizes);
        self.pool = unsafe { dev.create_descriptor_pool(&pinfo, None) }
            .map_err(|e| format!("descriptor pool: {e}"))?;

        // --- pipeline: push constants {width, height}, the committed SPIR-V ---
        let pc = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8)];
        let set_layouts = [self.layout];
        let plinfo = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&pc);
        self.pipeline_layout = unsafe { dev.create_pipeline_layout(&plinfo, None) }
            .map_err(|e| format!("pipeline layout: {e}"))?;

        let words = ash::util::read_spv(&mut Cursor::new(YUV2RGB_SPV))
            .map_err(|e| format!("read_spv: {e}"))?;
        let sinfo = vk::ShaderModuleCreateInfo::default().code(&words);
        self.shader = unsafe { dev.create_shader_module(&sinfo, None) }
            .map_err(|e| format!("shader module: {e}"))?;

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(self.shader)
            .name(c"main");
        let cinfo = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.pipeline_layout)];
        self.pipeline = unsafe {
            dev.create_compute_pipelines(vk::PipelineCache::null(), &cinfo, None)
        }
        .map_err(|(_, e)| format!("compute pipeline: {e}"))?[0];

        // --- per-slot output buffers (+ export) and descriptor sets ---
        for _ in 0..nslots {
            let mut ext = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let mut binfo = vk::BufferCreateInfo::default()
                .size(fb_size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            if self.export {
                binfo = binfo.push_next(&mut ext);
            }
            let buffer = unsafe { dev.create_buffer(&binfo, None) }
                .map_err(|e| format!("output buffer: {e}"))?;
            let req = unsafe { dev.get_buffer_memory_requirements(buffer) };
            let (required, preferred) = if self.export {
                // The exported buffer is GPU-written, compositor-read: device-local
                // preferred, no host access needed.
                (vk::MemoryPropertyFlags::empty(), vk::MemoryPropertyFlags::DEVICE_LOCAL)
            } else {
                // Test mode: host-readable for `read_back`.
                (
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    vk::MemoryPropertyFlags::empty(),
                )
            };
            let Some(mem_type) = gpu.find_memory_type(req.memory_type_bits, required, preferred)
            else {
                unsafe { dev.destroy_buffer(buffer, None) };
                return Err("no matching memory type for the output buffer".into());
            };
            let mut export_info = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let mut ainfo = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type);
            if self.export {
                ainfo = ainfo.push_next(&mut export_info);
            }
            let memory = match unsafe { dev.allocate_memory(&ainfo, None) } {
                Ok(m) => m,
                Err(e) => {
                    unsafe { dev.destroy_buffer(buffer, None) };
                    return err("allocate output memory", e);
                }
            };
            if let Err(e) = unsafe { dev.bind_buffer_memory(buffer, memory, 0) } {
                unsafe {
                    dev.destroy_buffer(buffer, None);
                    dev.free_memory(memory, None);
                }
                return err("bind output memory", e);
            }

            let fd = if self.export {
                let ext_dev = gpu.external_fd.as_ref().expect("can_export checked");
                let info = vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
                // SAFETY: `memory` was allocated with DMA_BUF export on this device.
                // On success the returned fd is ours to own and close.
                match unsafe { ext_dev.get_memory_fd(&info) } {
                    Ok(raw) => Some(unsafe { OwnedFd::from_raw_fd(raw) }),
                    Err(e) => {
                        unsafe {
                            dev.destroy_buffer(buffer, None);
                            dev.free_memory(memory, None);
                        }
                        return err("get_memory_fd", e);
                    }
                }
            } else {
                None
            };

            let alloc = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.pool)
                .set_layouts(&set_layouts);
            let set = unsafe { dev.allocate_descriptor_sets(&alloc) }
                .map_err(|e| format!("descriptor set: {e}"))?[0];
            let in_info = [vk::DescriptorBufferInfo::default()
                .buffer(self.input_buffer)
                .range(vk::WHOLE_SIZE)];
            let out_info = [vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .range(vk::WHOLE_SIZE)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&in_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&out_info),
            ];
            // SAFETY: sets/buffers are live handles on this device; infos outlive the call.
            unsafe { dev.update_descriptor_sets(&writes, &[]) };

            self.slots.push(Slot { buffer, memory, fd, set });
        }

        // --- command pool + one reusable command buffer + fence ---
        let cpinfo = vk::CommandPoolCreateInfo::default()
            .queue_family_index(gpu.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        self.cmd_pool = unsafe { dev.create_command_pool(&cpinfo, None) }
            .map_err(|e| format!("command pool: {e}"))?;
        let cbinfo = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        self.cmd = unsafe { dev.allocate_command_buffers(&cbinfo) }
            .map_err(|e| format!("command buffer: {e}"))?[0];
        self.fence = unsafe { dev.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| format!("fence: {e}"))?;
        Ok(())
    }

    /// Convert one packed-I420 frame into `slot`'s framebuffer. Copies the planes to
    /// the mapped input buffer, records dispatch + visibility barrier, submits, and
    /// waits the fence — on return the slot's memory holds the finished frame and is
    /// safe for the compositor (or [`read_back`](Self::read_back)) to read.
    pub fn render(&mut self, gpu: &Gpu, planes: &[u8], slot: usize) -> Result<(), String> {
        let need = i420_size(self.width, self.height);
        if planes.len() < need {
            return Err(format!("short frame: {} < {need} bytes", planes.len()));
        }
        debug_assert!(need <= self.input_size, "mapped input covers a whole frame");
        let s = self.slots.get(slot).ok_or("bad slot")?;
        let dev = &gpu.device;

        // SAFETY: `input_ptr` maps `input_size >= need` coherent bytes (invariant on
        // the field); the previous submission's fence has been waited (or none was
        // submitted yet), so the GPU is not reading the buffer concurrently.
        unsafe { std::ptr::copy_nonoverlapping(planes.as_ptr(), self.input_ptr, need) };

        // SAFETY (recording): `cmd` was allocated RESET-able from `cmd_pool`; no other
        // recording is in flight (single command buffer, fence-serialised renders).
        unsafe {
            dev.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("reset cmd: {e}"))?;
            dev.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| format!("begin cmd: {e}"))?;
            dev.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[s.set],
                &[],
            );
            let pc = [self.width as u32, self.height as u32];
            let bytes: [u8; 8] = std::mem::transmute(pc.map(u32::to_ne_bytes));
            dev.cmd_push_constants(
                self.cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &bytes,
            );
            dev.cmd_dispatch(
                self.cmd,
                (self.width as u32).div_ceil(8),
                (self.height as u32).div_ceil(8),
                1,
            );
            // Make the shader writes available to any consumer (host read-back or the
            // dma-buf importer) before the fence signals.
            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(s.buffer)
                .size(vk::WHOLE_SIZE);
            dev.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
            dev.end_command_buffer(self.cmd).map_err(|e| format!("end cmd: {e}"))?;

            let cmds = [self.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            dev.queue_submit(gpu.queue, &[submit], self.fence)
                .map_err(|e| format!("submit: {e}"))?;
            dev.wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| format!("fence wait: {e}"))?;
            dev.reset_fences(&[self.fence]).map_err(|e| format!("fence reset: {e}"))?;
        }
        Ok(())
    }

    /// Read `slot`'s framebuffer back to the host (test mode only — the slot memory
    /// must be host-visible, i.e. the renderer was built with `export == false`).
    pub fn read_back(&self, gpu: &Gpu, slot: usize) -> Result<Vec<u8>, String> {
        if self.export {
            return Err("read_back: renderer is in export mode (memory not host-visible)".into());
        }
        let s = self.slots.get(slot).ok_or("bad slot")?;
        let size = self.width * self.height * 4;
        let dev = &gpu.device;
        // SAFETY: non-export slots are HOST_VISIBLE|COHERENT and unmapped (only the
        // input buffer holds a persistent map); mapped here, copied, unmapped before
        // return — no aliasing with GPU work (render() fence-waits before returning).
        unsafe {
            let p = dev
                .map_memory(s.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("map output: {e}"))? as *const u8;
            let mut out = vec![0u8; size];
            std::ptr::copy_nonoverlapping(p, out.as_mut_ptr(), size);
            dev.unmap_memory(s.memory);
            Ok(out)
        }
    }

    /// Tear down against the creating `Gpu`. Must be called (the plain `Drop` cannot
    /// reach the device); the sink owns both and calls this from `stop`.
    pub fn destroy(mut self, gpu: &Gpu) {
        let dev = &gpu.device;
        // SAFETY: every handle below was created on `dev` by this renderer, and the
        // device is idled first so nothing is in flight.
        unsafe {
            let _ = dev.device_wait_idle();
            dev.destroy_fence(self.fence, None);
            dev.destroy_command_pool(self.cmd_pool, None);
            dev.destroy_pipeline(self.pipeline, None);
            dev.destroy_shader_module(self.shader, None);
            dev.destroy_pipeline_layout(self.pipeline_layout, None);
            dev.destroy_descriptor_pool(self.pool, None);
            dev.destroy_descriptor_set_layout(self.layout, None);
            for s in self.slots.drain(..) {
                dev.destroy_buffer(s.buffer, None);
                dev.free_memory(s.memory, None);
                // s.fd (our dup of the dma-buf) closes on drop.
            }
            dev.unmap_memory(self.input_memory);
            dev.destroy_buffer(self.input_buffer, None);
            dev.free_memory(self.input_memory, None);
        }
    }
}

// SAFETY: the raw `input_ptr` is only dereferenced from `&mut self` methods, and the
// renderer moves between threads whole (the sink owns it) — no shared aliasing.
unsafe impl Send for Renderer {}
