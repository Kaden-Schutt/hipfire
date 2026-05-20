// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

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

// GPU info — matches /opt/amdgpu/include/libdrm/amdgpu.h exactly (416 bytes)
// Verified with gcc offsetof: cu_active_number@324, vram_type@396, pci_rev_id@412
#[repr(C)]
#[derive(Default)]
pub struct AmdgpuGpuInfo {
    pub asic_id: u32,                          // 0
    pub chip_rev: u32,                         // 4
    pub chip_external_rev: u32,                // 8
    pub family_id: u32,                        // 12
    pub ids_flags: u64,                        // 16
    pub max_engine_clk: u64,                   // 24
    pub max_memory_clk: u64,                   // 32
    pub num_shader_engines: u32,               // 40
    pub num_shader_arrays_per_engine: u32,     // 44
    pub avail_quad_shader_pipes: u32,          // 48
    pub max_quad_shader_pipes: u32,            // 52
    pub cache_entries_per_quad_pipe: u32,      // 56
    pub num_hw_gfx_contexts: u32,             // 60
    pub rb_pipes: u32,                         // 64
    pub enabled_rb_pipes_mask: u32,            // 68
    pub gpu_counter_freq: u32,                 // 72
    pub backend_disable: [u32; 4],             // 76
    pub mc_arb_ramcfg: u32,                    // 92
    pub gb_addr_cfg: u32,                      // 96
    pub gb_tile_mode: [u32; 32],               // 100
    pub gb_macro_tile_mode: [u32; 16],         // 228
    pub pa_sc_raster_cfg: [u32; 4],            // 292
    pub pa_sc_raster_cfg1: [u32; 4],           // 308
    pub cu_active_number: u32,                 // 324
    pub cu_ao_mask: u32,                       // 328
    pub cu_bitmap: [[u32; 4]; 4],              // 332
    pub vram_type: u32,                        // 396
    pub vram_bit_width: u32,                   // 400
    pub ce_ram_size: u32,                      // 404
    pub vce_harvest_config: u32,               // 408
    pub pci_rev_id: u32,                       // 412
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
    // Context + submission
    pub cs_ctx_create2: unsafe extern "C" fn(device: AmdgpuDeviceHandle, priority: u32, ctx: *mut AmdgpuContext) -> i32,
    pub cs_ctx_free: unsafe extern "C" fn(ctx: AmdgpuContext) -> i32,
    pub cs_submit: unsafe extern "C" fn(ctx: AmdgpuContext, flags: u64, request: *mut CsRequest, num_requests: u32) -> i32,
    pub cs_query_fence_status: unsafe extern "C" fn(fence: *mut CsFence, timeout_ns: u64, flags: u64, expired: *mut u32) -> i32,
    // BO list
    pub bo_list_create: unsafe extern "C" fn(device: AmdgpuDeviceHandle, num: u32, resources: *const AmdgpuBoHandle, prios: *const u8, result: *mut AmdgpuBoListHandle) -> i32,
    pub bo_list_destroy: unsafe extern "C" fn(list: AmdgpuBoListHandle) -> i32,
}

pub type AmdgpuBoListHandle = *mut c_void;

/// amdgpu_cs_ib_info — matches amdgpu.h
#[repr(C)]
pub struct CsIbInfo {
    pub flags: u64,
    pub ib_mc_address: u64,
    pub size: u32,   // in dwords
    pub _pad: u32,
}

/// amdgpu_cs_request — matches amdgpu.h EXACTLY
/// Verified field order: flags, ip_type, ip_instance, ring, resources,
/// number_of_dependencies, dependencies, number_of_ibs, ibs, seq_no, fence_info
#[repr(C)]
pub struct CsRequest {
    pub flags: u64,                       // 0
    pub ip_type: u32,                     // 8   (unsigned)
    pub ip_instance: u32,                 // 12  (unsigned)
    pub ring: u32,                        // 16
    pub _pad0: u32,                       // 20  (padding for pointer alignment)
    pub resources: AmdgpuBoListHandle,    // 24  (pointer)
    pub number_of_dependencies: u32,      // 32
    pub _pad1: u32,                       // 36  (padding for pointer alignment)
    pub dependencies: *const CsFence,     // 40  (pointer)
    pub number_of_ibs: u32,              // 48
    pub _pad2: u32,                       // 52  (padding for pointer alignment)
    pub ibs: *mut CsIbInfo,              // 56  (pointer)
    pub seq_no: u64,                      // 64  (output)
    pub fence_info: CsFenceInfo,          // 72
}

/// amdgpu_cs_fence_info
#[repr(C)]
#[derive(Default)]
pub struct CsFenceInfo {
    pub handle: AmdgpuBoHandle,
    pub offset: u64,
}

/// amdgpu_cs_fence
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct CsFence {
    pub context: AmdgpuContext,
    pub ip_type: u32,
    pub ip_instance: u32,
    pub ring: u32,
    pub fence: u64,
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
                cs_submit: sym!("cs_submit", unsafe extern "C" fn(AmdgpuContext, u64, *mut CsRequest, u32) -> i32),
                cs_query_fence_status: sym!("cs_query_fence_status", unsafe extern "C" fn(*mut CsFence, u64, u64, *mut u32) -> i32),
                bo_list_create: sym!("bo_list_create", unsafe extern "C" fn(AmdgpuDeviceHandle, u32, *const AmdgpuBoHandle, *const u8, *mut AmdgpuBoListHandle) -> i32),
                bo_list_destroy: sym!("bo_list_destroy", unsafe extern "C" fn(AmdgpuBoListHandle) -> i32),
                _lib: lib,
            })
        }
    }
}
