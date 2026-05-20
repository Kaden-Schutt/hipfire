// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! FFI bindings to libhsa-runtime64.so via dlopen.
//!
//! Covers only the subset needed for user-mode AQL kernel dispatch:
//! runtime init, agent discovery, queue create, signal create/wait,
//! memory pool allocate (kernarg + device), code object loading,
//! and kernel-descriptor symbol lookup.
//!
//! All functions are resolved at runtime — no link-time dependency on
//! libhsa-runtime64.so. Mirrors the hip-bridge pattern.

use crate::error::HsaStatus;
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_void};

// ─── Opaque handles (all HSA handles are wrapped u64) ─────────────────────

pub type HsaAgentHandle = u64;
pub type HsaSignalHandle = u64;
pub type HsaMemoryPoolHandle = u64;
pub type HsaExecutableHandle = u64;
pub type HsaCodeObjectReaderHandle = u64;
pub type HsaExecutableSymbolHandle = u64;

// ─── Public struct layouts (must match hsa.h byte-for-byte) ──────────────

/// `hsa_queue_t` from hsa.h lines 2308-2363.
/// Laid out for HSA_LARGE_MODEL + HSA_LITTLE_ENDIAN (x86_64 Linux).
#[repr(C)]
pub struct HsaQueue {
    pub type_: u32,
    pub features: u32,
    pub base_address: *mut c_void,
    pub doorbell_signal: HsaSignalHandle,
    pub size: u32,
    pub reserved1: u32,
    pub id: u64,
}

/// `hsa_kernel_dispatch_packet_t` from hsa.h line 2959. 64 bytes, 64B aligned.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct HsaKernelDispatchPacket {
    pub header: u16,
    pub setup: u16,
    pub workgroup_size_x: u16,
    pub workgroup_size_y: u16,
    pub workgroup_size_z: u16,
    pub reserved0: u16,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,
    pub private_segment_size: u32,
    pub group_segment_size: u32,
    pub kernel_object: u64,
    pub kernarg_address: *mut c_void,
    pub reserved2: u64,
    pub completion_signal: HsaSignalHandle,
}

const _: () = assert!(std::mem::size_of::<HsaKernelDispatchPacket>() == 64);

// ─── Enum constants (values from hsa.h and hsa_ext_amd.h) ────────────────

pub const HSA_DEVICE_TYPE_CPU: u32 = 0;
pub const HSA_DEVICE_TYPE_GPU: u32 = 1;

pub const HSA_AGENT_INFO_NAME: u32 = 0;
pub const HSA_AGENT_INFO_DEVICE: u32 = 17;

pub const HSA_QUEUE_TYPE_MULTI: u32 = 0;
pub const HSA_QUEUE_TYPE_SINGLE: u32 = 1;

pub const HSA_PACKET_TYPE_INVALID: u16 = 1;
pub const HSA_PACKET_TYPE_KERNEL_DISPATCH: u16 = 2;

pub const HSA_FENCE_SCOPE_NONE: u16 = 0;
pub const HSA_FENCE_SCOPE_AGENT: u16 = 1;
pub const HSA_FENCE_SCOPE_SYSTEM: u16 = 2;

// Header bit offsets from hsa.h hsa_packet_header_t
pub const HSA_PACKET_HEADER_TYPE: u16 = 0;
pub const HSA_PACKET_HEADER_BARRIER: u16 = 8;
pub const HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE: u16 = 9;
pub const HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE: u16 = 11;

pub const HSA_SIGNAL_CONDITION_EQ: u32 = 0;
pub const HSA_SIGNAL_CONDITION_NE: u32 = 1;
pub const HSA_SIGNAL_CONDITION_LT: u32 = 2;
pub const HSA_SIGNAL_CONDITION_GTE: u32 = 3;

pub const HSA_WAIT_STATE_BLOCKED: u32 = 0;
pub const HSA_WAIT_STATE_ACTIVE: u32 = 1;

pub const HSA_PROFILE_BASE: u32 = 0;
pub const HSA_PROFILE_FULL: u32 = 1;

pub const HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT: u32 = 0;

pub const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
pub const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 11;
pub const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 13;
pub const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 14;

pub const HSA_AMD_SEGMENT_GLOBAL: u32 = 0;

pub const HSA_AMD_MEMORY_POOL_INFO_SEGMENT: u32 = 0;
pub const HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS: u32 = 1;
pub const HSA_AMD_MEMORY_POOL_INFO_SIZE: u32 = 2;
pub const HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED: u32 = 5;

pub const HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT: u32 = 1;
pub const HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED: u32 = 2;
pub const HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED: u32 = 4;

// ─── Loaded runtime: function pointers ───────────────────────────────────

pub struct HsaLib {
    _lib: Library,

    // Runtime init / shutdown
    pub fn_init: unsafe extern "C" fn() -> HsaStatus,
    pub fn_shut_down: unsafe extern "C" fn() -> HsaStatus,

    // Agents
    pub fn_iterate_agents: unsafe extern "C" fn(
        callback: unsafe extern "C" fn(HsaAgentHandle, *mut c_void) -> HsaStatus,
        data: *mut c_void,
    ) -> HsaStatus,
    pub fn_agent_get_info:
        unsafe extern "C" fn(HsaAgentHandle, u32, *mut c_void) -> HsaStatus,

    // Queues
    pub fn_queue_create: unsafe extern "C" fn(
        agent: HsaAgentHandle,
        size: u32,
        type_: u32,
        callback: *mut c_void,
        data: *mut c_void,
        private_segment_size: u32,
        group_segment_size: u32,
        queue: *mut *mut HsaQueue,
    ) -> HsaStatus,
    pub fn_queue_destroy: unsafe extern "C" fn(queue: *mut HsaQueue) -> HsaStatus,
    pub fn_queue_load_write_index_relaxed:
        unsafe extern "C" fn(queue: *const HsaQueue) -> u64,
    pub fn_queue_store_write_index_release:
        unsafe extern "C" fn(queue: *const HsaQueue, value: u64),

    // Signals
    pub fn_signal_create: unsafe extern "C" fn(
        initial_value: i64,
        num_consumers: u32,
        consumers: *const HsaAgentHandle,
        signal: *mut HsaSignalHandle,
    ) -> HsaStatus,
    pub fn_signal_destroy: unsafe extern "C" fn(signal: HsaSignalHandle) -> HsaStatus,
    pub fn_signal_store_relaxed: unsafe extern "C" fn(signal: HsaSignalHandle, value: i64),
    pub fn_signal_store_screlease:
        unsafe extern "C" fn(signal: HsaSignalHandle, value: i64),
    pub fn_signal_load_relaxed: unsafe extern "C" fn(signal: HsaSignalHandle) -> i64,
    pub fn_signal_wait_scacquire: unsafe extern "C" fn(
        signal: HsaSignalHandle,
        condition: u32,
        compare_value: i64,
        timeout_hint: u64,
        wait_state_hint: u32,
    ) -> i64,

    // Memory pools (AMD extension)
    pub fn_amd_agent_iterate_memory_pools: unsafe extern "C" fn(
        agent: HsaAgentHandle,
        callback: unsafe extern "C" fn(HsaMemoryPoolHandle, *mut c_void) -> HsaStatus,
        data: *mut c_void,
    ) -> HsaStatus,
    pub fn_amd_memory_pool_get_info: unsafe extern "C" fn(
        pool: HsaMemoryPoolHandle,
        attribute: u32,
        value: *mut c_void,
    ) -> HsaStatus,
    pub fn_amd_memory_pool_allocate: unsafe extern "C" fn(
        pool: HsaMemoryPoolHandle,
        size: usize,
        flags: u32,
        ptr: *mut *mut c_void,
    ) -> HsaStatus,
    pub fn_amd_memory_pool_free: unsafe extern "C" fn(ptr: *mut c_void) -> HsaStatus,
    pub fn_amd_agents_allow_access: unsafe extern "C" fn(
        num_agents: u32,
        agents: *const HsaAgentHandle,
        flags: *const u32,
        ptr: *const c_void,
    ) -> HsaStatus,

    // Code objects / executables
    pub fn_code_object_reader_create_from_memory: unsafe extern "C" fn(
        code_object: *const c_void,
        size: usize,
        code_object_reader: *mut HsaCodeObjectReaderHandle,
    ) -> HsaStatus,
    pub fn_code_object_reader_destroy:
        unsafe extern "C" fn(reader: HsaCodeObjectReaderHandle) -> HsaStatus,
    pub fn_executable_create_alt: unsafe extern "C" fn(
        profile: u32,
        default_float_rounding_mode: u32,
        options: *const c_char,
        executable: *mut HsaExecutableHandle,
    ) -> HsaStatus,
    pub fn_executable_load_agent_code_object: unsafe extern "C" fn(
        executable: HsaExecutableHandle,
        agent: HsaAgentHandle,
        code_object_reader: HsaCodeObjectReaderHandle,
        options: *const c_char,
        loaded_code_object: *mut u64,
    ) -> HsaStatus,
    pub fn_executable_freeze: unsafe extern "C" fn(
        executable: HsaExecutableHandle,
        options: *const c_char,
    ) -> HsaStatus,
    pub fn_executable_destroy:
        unsafe extern "C" fn(executable: HsaExecutableHandle) -> HsaStatus,
    pub fn_executable_get_symbol_by_name: unsafe extern "C" fn(
        executable: HsaExecutableHandle,
        symbol_name: *const c_char,
        agent: *const HsaAgentHandle,
        symbol: *mut HsaExecutableSymbolHandle,
    ) -> HsaStatus,
    pub fn_executable_symbol_get_info: unsafe extern "C" fn(
        symbol: HsaExecutableSymbolHandle,
        attribute: u32,
        value: *mut c_void,
    ) -> HsaStatus,
}

// HSA runtime is thread-safe for API calls, like HIP.
unsafe impl Send for HsaLib {}
unsafe impl Sync for HsaLib {}

macro_rules! load_fn {
    ($lib:expr, $name:expr, $ty:ty) => {{
        let sym: Symbol<'_, $ty> = $lib.get($name.as_bytes()).map_err(|e| {
            crate::error::HsaError::new(
                0,
                &format!("failed to load symbol {}: {e}", $name),
            )
        })?;
        *sym.into_raw()
    }};
}

impl HsaLib {
    /// Dlopen libhsa-runtime64.so and resolve all function pointers.
    /// Searches /opt/rocm/lib then the default loader path.
    pub fn load() -> crate::error::HsaResult<Self> {
        let lib = unsafe {
            Library::new("/opt/rocm/lib/libhsa-runtime64.so.1")
                .or_else(|_| Library::new("/opt/rocm/lib/libhsa-runtime64.so"))
                .or_else(|_| Library::new("libhsa-runtime64.so.1"))
                .or_else(|_| Library::new("libhsa-runtime64.so"))
                .map_err(|e| {
                    crate::error::HsaError::new(
                        0,
                        &format!(
                            "failed to dlopen libhsa-runtime64.so: {e}. Is ROCm installed?"
                        ),
                    )
                })?
        };

        unsafe {
            Ok(Self {
                fn_init: load_fn!(lib, "hsa_init", unsafe extern "C" fn() -> HsaStatus),
                fn_shut_down: load_fn!(
                    lib,
                    "hsa_shut_down",
                    unsafe extern "C" fn() -> HsaStatus
                ),

                fn_iterate_agents: load_fn!(
                    lib,
                    "hsa_iterate_agents",
                    unsafe extern "C" fn(
                        unsafe extern "C" fn(HsaAgentHandle, *mut c_void) -> HsaStatus,
                        *mut c_void,
                    ) -> HsaStatus
                ),
                fn_agent_get_info: load_fn!(
                    lib,
                    "hsa_agent_get_info",
                    unsafe extern "C" fn(HsaAgentHandle, u32, *mut c_void) -> HsaStatus
                ),

                fn_queue_create: load_fn!(
                    lib,
                    "hsa_queue_create",
                    unsafe extern "C" fn(
                        HsaAgentHandle,
                        u32,
                        u32,
                        *mut c_void,
                        *mut c_void,
                        u32,
                        u32,
                        *mut *mut HsaQueue,
                    ) -> HsaStatus
                ),
                fn_queue_destroy: load_fn!(
                    lib,
                    "hsa_queue_destroy",
                    unsafe extern "C" fn(*mut HsaQueue) -> HsaStatus
                ),
                fn_queue_load_write_index_relaxed: load_fn!(
                    lib,
                    "hsa_queue_load_write_index_relaxed",
                    unsafe extern "C" fn(*const HsaQueue) -> u64
                ),
                fn_queue_store_write_index_release: load_fn!(
                    lib,
                    "hsa_queue_store_write_index_release",
                    unsafe extern "C" fn(*const HsaQueue, u64)
                ),

                fn_signal_create: load_fn!(
                    lib,
                    "hsa_signal_create",
                    unsafe extern "C" fn(
                        i64,
                        u32,
                        *const HsaAgentHandle,
                        *mut HsaSignalHandle,
                    ) -> HsaStatus
                ),
                fn_signal_destroy: load_fn!(
                    lib,
                    "hsa_signal_destroy",
                    unsafe extern "C" fn(HsaSignalHandle) -> HsaStatus
                ),
                fn_signal_store_relaxed: load_fn!(
                    lib,
                    "hsa_signal_store_relaxed",
                    unsafe extern "C" fn(HsaSignalHandle, i64)
                ),
                fn_signal_store_screlease: load_fn!(
                    lib,
                    "hsa_signal_store_screlease",
                    unsafe extern "C" fn(HsaSignalHandle, i64)
                ),
                fn_signal_load_relaxed: load_fn!(
                    lib,
                    "hsa_signal_load_relaxed",
                    unsafe extern "C" fn(HsaSignalHandle) -> i64
                ),
                fn_signal_wait_scacquire: load_fn!(
                    lib,
                    "hsa_signal_wait_scacquire",
                    unsafe extern "C" fn(HsaSignalHandle, u32, i64, u64, u32) -> i64
                ),

                fn_amd_agent_iterate_memory_pools: load_fn!(
                    lib,
                    "hsa_amd_agent_iterate_memory_pools",
                    unsafe extern "C" fn(
                        HsaAgentHandle,
                        unsafe extern "C" fn(HsaMemoryPoolHandle, *mut c_void) -> HsaStatus,
                        *mut c_void,
                    ) -> HsaStatus
                ),
                fn_amd_memory_pool_get_info: load_fn!(
                    lib,
                    "hsa_amd_memory_pool_get_info",
                    unsafe extern "C" fn(HsaMemoryPoolHandle, u32, *mut c_void) -> HsaStatus
                ),
                fn_amd_memory_pool_allocate: load_fn!(
                    lib,
                    "hsa_amd_memory_pool_allocate",
                    unsafe extern "C" fn(
                        HsaMemoryPoolHandle,
                        usize,
                        u32,
                        *mut *mut c_void,
                    ) -> HsaStatus
                ),
                fn_amd_memory_pool_free: load_fn!(
                    lib,
                    "hsa_amd_memory_pool_free",
                    unsafe extern "C" fn(*mut c_void) -> HsaStatus
                ),
                fn_amd_agents_allow_access: load_fn!(
                    lib,
                    "hsa_amd_agents_allow_access",
                    unsafe extern "C" fn(
                        u32,
                        *const HsaAgentHandle,
                        *const u32,
                        *const c_void,
                    ) -> HsaStatus
                ),

                fn_code_object_reader_create_from_memory: load_fn!(
                    lib,
                    "hsa_code_object_reader_create_from_memory",
                    unsafe extern "C" fn(
                        *const c_void,
                        usize,
                        *mut HsaCodeObjectReaderHandle,
                    ) -> HsaStatus
                ),
                fn_code_object_reader_destroy: load_fn!(
                    lib,
                    "hsa_code_object_reader_destroy",
                    unsafe extern "C" fn(HsaCodeObjectReaderHandle) -> HsaStatus
                ),
                fn_executable_create_alt: load_fn!(
                    lib,
                    "hsa_executable_create_alt",
                    unsafe extern "C" fn(
                        u32,
                        u32,
                        *const c_char,
                        *mut HsaExecutableHandle,
                    ) -> HsaStatus
                ),
                fn_executable_load_agent_code_object: load_fn!(
                    lib,
                    "hsa_executable_load_agent_code_object",
                    unsafe extern "C" fn(
                        HsaExecutableHandle,
                        HsaAgentHandle,
                        HsaCodeObjectReaderHandle,
                        *const c_char,
                        *mut u64,
                    ) -> HsaStatus
                ),
                fn_executable_freeze: load_fn!(
                    lib,
                    "hsa_executable_freeze",
                    unsafe extern "C" fn(HsaExecutableHandle, *const c_char) -> HsaStatus
                ),
                fn_executable_destroy: load_fn!(
                    lib,
                    "hsa_executable_destroy",
                    unsafe extern "C" fn(HsaExecutableHandle) -> HsaStatus
                ),
                fn_executable_get_symbol_by_name: load_fn!(
                    lib,
                    "hsa_executable_get_symbol_by_name",
                    unsafe extern "C" fn(
                        HsaExecutableHandle,
                        *const c_char,
                        *const HsaAgentHandle,
                        *mut HsaExecutableSymbolHandle,
                    ) -> HsaStatus
                ),
                fn_executable_symbol_get_info: load_fn!(
                    lib,
                    "hsa_executable_symbol_get_info",
                    unsafe extern "C" fn(
                        HsaExecutableSymbolHandle,
                        u32,
                        *mut c_void,
                    ) -> HsaStatus
                ),
                _lib: lib,
            })
        }
    }
}
