//! FFI bindings to libdrm_amdgpu.so via dlopen.
//! Struct layouts match /opt/amdgpu/include/libdrm/amdgpu.h exactly.

use crate::{RedlineError, Result};
use std::ffi::c_void;

// Opaque handles
pub type AmdgpuDeviceHandle = *mut c_void;
pub type AmdgpuBoHandle = *mut c_void;
pub type AmdgpuContext = *mut c_void;
pub type AmdgpuVaHandle = *mut c_void;

// Memory domains
pub const AMDGPU_GEM_DOMAIN_VRAM: u32 = 0x4;
#[allow(dead_code)]
pub const AMDGPU_GEM_DOMAIN_GTT: u32 = 0x2;
pub const AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED: u64 = 1 << 0;

// VA ops
pub const AMDGPU_VA_OP_MAP: u32 = 1;
pub const AMDGPU_VA_OP_UNMAP: u32 = 2;

// VA range type
#[repr(u32)]
#[allow(dead_code)]
pub enum AmdgpuGpuVaRange {
    General = 0,
}

// BO alloc request — matches amdgpu.h exactly
#[repr(C)]
pub struct AmdgpuBoAllocRequest {
    pub alloc_size: u64,
    pub phys_alignment: u64,
    pub preferred_heap: u32,
    pub flags: u64,
}

// GPU info — matches amdgpu.h (note: rb_pipes is u32, not padding)
#[repr(C)]
#[derive(Default)]
pub struct AmdgpuGpuInfo {
    pub asic_id: u32,
    pub chip_rev: u32,
    pub chip_external_rev: u32,
    pub family_id: u32,
    pub ids_flags: u64,
    pub max_engine_clk: u64,
    pub max_memory_clk: u64,
    pub num_shader_engines: u32,
    pub num_shader_arrays_per_engine: u32,
    pub avail_quad_shader_pipes: u32,
    pub max_quad_shader_pipes: u32,
    pub cache_entries_per_quad_pipe: u32,
    pub num_hw_gfx_contexts: u32,
    pub rb_pipes: u32,
    pub enabled_rb_pipes_mask: u32,
    pub gb_addr_cfg: u32,
    pub num_good_cu_per_sh: u32,
    pub cu_active_number: u32,
    pub cu_ao_mask: u32,
    pub cu_bitmap: [[u32; 4]; 4],
    pub vram_type: u32,
    pub vram_bit_width: u32,
    pub ce_ram_size: u32,
    pub vce_harvest_config: u32,
    pub gc_double_offchip_lds_buf: u32,
    pub prim_buf_gpu_addr: u64,
    pub pos_buf_gpu_addr: u64,
    pub cntl_sb_buf_gpu_addr: u64,
    pub param_buf_gpu_addr: u64,
    pub prim_buf_size: u32,
    pub pos_buf_size: u32,
    pub cntl_sb_buf_size: u32,
    pub param_buf_size: u32,
    pub wave_front_size: u32,
    pub num_shader_visible_vgprs: u32,
    pub num_cu_per_sh: u32,
    pub num_tcc_blocks: u32,
    pub gs_vgt_table_depth: u32,
    pub gs_prim_buffer_depth: u32,
    pub max_gs_waves_per_vgt: u32,
    pub _pad1: u32,
    pub cu_ao_bitmap: [[u32; 4]; 4],
    pub high_va_offset: u64,
    pub high_va_max: u64,
    pub pa_sc_tile_steering_override: u32,
    pub tcc_disabled_mask: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct HeapInfo {
    pub total_heap_size: u64,
    pub usable_heap_size: u64,
    pub heap_usage: u64,
    pub max_allocation: u64,
}

/// Dynamically loaded libdrm_amdgpu functions.
pub struct DrmLib {
    _lib: libloading::Library,
    // Device
    pub device_initialize: unsafe extern "C" fn(fd: i32, major: *mut u32, minor: *mut u32, device: *mut AmdgpuDeviceHandle) -> i32,
    pub device_deinitialize: unsafe extern "C" fn(device: AmdgpuDeviceHandle) -> i32,
    // Info
    pub query_gpu_info: unsafe extern "C" fn(device: AmdgpuDeviceHandle, info: *mut AmdgpuGpuInfo) -> i32,
    pub query_heap_info: unsafe extern "C" fn(device: AmdgpuDeviceHandle, heap: u32, flags: u32, info: *mut HeapInfo) -> i32,
    // Memory — proper flow: bo_alloc → va_range_alloc → bo_va_op(MAP)
    pub bo_alloc: unsafe extern "C" fn(device: AmdgpuDeviceHandle, req: *const AmdgpuBoAllocRequest, handle: *mut AmdgpuBoHandle) -> i32,
    pub bo_free: unsafe extern "C" fn(bo: AmdgpuBoHandle) -> i32,
    pub bo_cpu_map: unsafe extern "C" fn(bo: AmdgpuBoHandle, cpu: *mut *mut c_void) -> i32,
    pub bo_cpu_unmap: unsafe extern "C" fn(bo: AmdgpuBoHandle) -> i32,
    pub bo_va_op: unsafe extern "C" fn(bo: AmdgpuBoHandle, offset: u64, size: u64, addr: u64, flags: u64, ops: u32) -> i32,
    pub va_range_alloc: unsafe extern "C" fn(device: AmdgpuDeviceHandle, va_type: u32, size: u64, align: u64, base_required: u64, base_allocated: *mut u64, va_handle: *mut AmdgpuVaHandle, flags: u64) -> i32,
    pub va_range_free: unsafe extern "C" fn(va_handle: AmdgpuVaHandle) -> i32,
    // Context
    pub cs_ctx_create2: unsafe extern "C" fn(device: AmdgpuDeviceHandle, priority: u32, ctx: *mut AmdgpuContext) -> i32,
    pub cs_ctx_free: unsafe extern "C" fn(ctx: AmdgpuContext) -> i32,
}

impl DrmLib {
    pub fn load() -> Result<Self> {
        unsafe {
            let lib = libloading::Library::new("libdrm_amdgpu.so")
                .or_else(|_| libloading::Library::new("libdrm_amdgpu.so.1"))
                .map_err(|e| RedlineError {
                    code: -1,
                    message: format!("failed to load libdrm_amdgpu.so: {e}. Is the amdgpu driver installed?"),
                })?;

            macro_rules! sym {
                ($name:expr, $ty:ty) => {{
                    let s: libloading::Symbol<$ty> = lib.get(concat!("amdgpu_", $name, "\0").as_bytes())
                        .map_err(|e| RedlineError { code: -1, message: format!("missing symbol amdgpu_{}: {e}", $name) })?;
                    *s
                }};
            }

            Ok(Self {
                device_initialize: sym!("device_initialize", unsafe extern "C" fn(i32, *mut u32, *mut u32, *mut AmdgpuDeviceHandle) -> i32),
                device_deinitialize: sym!("device_deinitialize", unsafe extern "C" fn(AmdgpuDeviceHandle) -> i32),
                query_gpu_info: sym!("query_gpu_info", unsafe extern "C" fn(AmdgpuDeviceHandle, *mut AmdgpuGpuInfo) -> i32),
                query_heap_info: sym!("query_heap_info", unsafe extern "C" fn(AmdgpuDeviceHandle, u32, u32, *mut HeapInfo) -> i32),
                bo_alloc: sym!("bo_alloc", unsafe extern "C" fn(AmdgpuDeviceHandle, *const AmdgpuBoAllocRequest, *mut AmdgpuBoHandle) -> i32),
                bo_free: sym!("bo_free", unsafe extern "C" fn(AmdgpuBoHandle) -> i32),
                bo_cpu_map: sym!("bo_cpu_map", unsafe extern "C" fn(AmdgpuBoHandle, *mut *mut c_void) -> i32),
                bo_cpu_unmap: sym!("bo_cpu_unmap", unsafe extern "C" fn(AmdgpuBoHandle) -> i32),
                bo_va_op: sym!("bo_va_op", unsafe extern "C" fn(AmdgpuBoHandle, u64, u64, u64, u64, u32) -> i32),
                va_range_alloc: sym!("va_range_alloc", unsafe extern "C" fn(AmdgpuDeviceHandle, u32, u64, u64, u64, *mut u64, *mut AmdgpuVaHandle, u64) -> i32),
                va_range_free: sym!("va_range_free", unsafe extern "C" fn(AmdgpuVaHandle) -> i32),
                cs_ctx_create2: sym!("cs_ctx_create2", unsafe extern "C" fn(AmdgpuDeviceHandle, u32, *mut AmdgpuContext) -> i32),
                cs_ctx_free: sym!("cs_ctx_free", unsafe extern "C" fn(AmdgpuContext) -> i32),
                _lib: lib,
            })
        }
    }
}
