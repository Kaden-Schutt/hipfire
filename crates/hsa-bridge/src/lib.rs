// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hsa-bridge: thin Rust wrapper around libhsa-runtime64.so.
//!
//! Purpose: bypass HIP's ~10 µs launch overhead on gfx1100 by submitting
//! AQL packets directly to a user-mode queue. HSA is the exact layer HIP
//! sits on — what we save is HIP's wrappers, signal-wait conservatism,
//! and cross-stream synchronization cost.
//!
//! Scope (Phase 2): just enough API surface to dispatch a single kernel
//! loaded from a `.hsaco` code object and compare per-dispatch latency
//! against HIP. Full kernel loader, lifecycle, and multi-queue support
//! will come later as Redline ports more of the forward pass.

mod error;
mod ffi;

pub use error::{HsaError, HsaResult, HsaStatus, HSA_STATUS_SUCCESS};
pub use ffi::{HsaKernelDispatchPacket, HsaQueue as HsaQueueRaw};

use ffi::*;
use std::ffi::{c_void, CString};
use std::ptr;
use std::sync::Arc;

// ─── Runtime ──────────────────────────────────────────────────────────────

/// Owns the dlopen'd libhsa-runtime64 and tracks init state.
pub struct HsaRuntime {
    pub(crate) lib: HsaLib,
}

impl HsaRuntime {
    /// Load libhsa-runtime64 and call `hsa_init()`.
    pub fn load() -> HsaResult<Arc<Self>> {
        let lib = HsaLib::load()?;
        let status = unsafe { (lib.fn_init)() };
        error::check(status, "hsa_init")?;
        Ok(Arc::new(Self { lib }))
    }

    /// Enumerate agents and return the first GPU whose name contains `gfx_arch`.
    /// Pass `None` to get the first GPU agent regardless of arch.
    pub fn find_gpu_agent(self: &Arc<Self>, gfx_arch: Option<&str>) -> HsaResult<HsaAgent> {
        self.find_agent(HSA_DEVICE_TYPE_GPU, gfx_arch)
    }

    /// Enumerate agents and return the first CPU agent.
    pub fn find_cpu_agent(self: &Arc<Self>) -> HsaResult<HsaAgent> {
        self.find_agent(HSA_DEVICE_TYPE_CPU, None)
    }

    fn find_agent(
        self: &Arc<Self>,
        device_type: u32,
        name_filter: Option<&str>,
    ) -> HsaResult<HsaAgent> {
        struct Ctx {
            runtime: *const HsaRuntime,
            wanted_type: u32,
            target_name: Option<String>,
            found: HsaAgentHandle,
        }

        unsafe extern "C" fn visit(agent: HsaAgentHandle, data: *mut c_void) -> HsaStatus {
            let ctx = &mut *(data as *mut Ctx);
            let rt = &*ctx.runtime;
            let mut dev_type: u32 = 0;
            let st = (rt.lib.fn_agent_get_info)(
                agent,
                HSA_AGENT_INFO_DEVICE,
                &mut dev_type as *mut _ as *mut c_void,
            );
            if st != HSA_STATUS_SUCCESS || dev_type != ctx.wanted_type {
                return HSA_STATUS_SUCCESS; // continue
            }
            if let Some(ref want) = ctx.target_name {
                let mut name_buf = [0i8; 64];
                let st = (rt.lib.fn_agent_get_info)(
                    agent,
                    HSA_AGENT_INFO_NAME,
                    name_buf.as_mut_ptr() as *mut c_void,
                );
                if st != HSA_STATUS_SUCCESS {
                    return HSA_STATUS_SUCCESS;
                }
                let end = name_buf.iter().position(|&c| c == 0).unwrap_or(64);
                let name_bytes: &[u8] =
                    std::slice::from_raw_parts(name_buf.as_ptr() as *const u8, end);
                let name = std::str::from_utf8(name_bytes).unwrap_or("");
                if !name.contains(want.as_str()) {
                    return HSA_STATUS_SUCCESS;
                }
            }
            ctx.found = agent;
            0x1000 // short-circuit
        }

        let mut ctx = Ctx {
            runtime: Arc::as_ptr(self),
            wanted_type: device_type,
            target_name: name_filter.map(|s| s.to_string()),
            found: 0,
        };
        let status =
            unsafe { (self.lib.fn_iterate_agents)(visit, &mut ctx as *mut _ as *mut c_void) };
        if ctx.found == 0 {
            return Err(HsaError::new(
                status,
                &format!("find_agent({device_type}) did not match"),
            ));
        }
        Ok(HsaAgent {
            runtime: self.clone(),
            handle: ctx.found,
        })
    }
}

impl Drop for HsaRuntime {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.lib.fn_shut_down)();
        }
    }
}

// ─── Agent ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct HsaAgent {
    pub(crate) runtime: Arc<HsaRuntime>,
    pub(crate) handle: HsaAgentHandle,
}

impl HsaAgent {
    pub fn raw_handle(&self) -> HsaAgentHandle {
        self.handle
    }

    pub fn name(&self) -> HsaResult<String> {
        let mut buf = [0i8; 64];
        let st = unsafe {
            (self.runtime.lib.fn_agent_get_info)(
                self.handle,
                HSA_AGENT_INFO_NAME,
                buf.as_mut_ptr() as *mut c_void,
            )
        };
        error::check(st, "agent_get_info(NAME)")?;
        let end = buf.iter().position(|&c| c == 0).unwrap_or(64);
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, end) };
        Ok(std::str::from_utf8(bytes).unwrap_or("").to_string())
    }

    /// Create a user-mode AQL queue on this agent. `size` must be a power of 2.
    pub fn create_queue(&self, size: u32) -> HsaResult<HsaQueue> {
        let mut queue: *mut ffi::HsaQueue = ptr::null_mut();
        let st = unsafe {
            (self.runtime.lib.fn_queue_create)(
                self.handle,
                size,
                HSA_QUEUE_TYPE_SINGLE,
                ptr::null_mut(),
                ptr::null_mut(),
                u32::MAX, // private_segment_size: let runtime decide
                u32::MAX, // group_segment_size: let runtime decide
                &mut queue,
            )
        };
        error::check(st, "hsa_queue_create")?;
        Ok(HsaQueue {
            runtime: self.runtime.clone(),
            raw: queue,
        })
    }

    /// Find the memory pool this agent should use for kernarg buffers:
    /// global + fine-grained + runtime-alloc-allowed + KERNARG_INIT flag.
    pub fn find_kernarg_pool(&self) -> HsaResult<HsaMemoryPool> {
        self.find_pool(PoolKind::Kernarg)
    }

    /// Find a CPU-writable, GPU-accessible fine-grained pool for small
    /// host-side buffers. Used for quick test data upload.
    pub fn find_fine_grained_pool(&self) -> HsaResult<HsaMemoryPool> {
        self.find_pool(PoolKind::FineGrained)
    }

    /// Find the device-local (VRAM) coarse-grained pool.
    pub fn find_coarse_grained_pool(&self) -> HsaResult<HsaMemoryPool> {
        self.find_pool(PoolKind::CoarseGrained)
    }

    fn find_pool(&self, kind: PoolKind) -> HsaResult<HsaMemoryPool> {
        struct Ctx {
            runtime: *const HsaRuntime,
            kind: PoolKind,
            found: HsaMemoryPoolHandle,
        }

        unsafe extern "C" fn visit(pool: HsaMemoryPoolHandle, data: *mut c_void) -> HsaStatus {
            let ctx = &mut *(data as *mut Ctx);
            let rt = &*ctx.runtime;
            let mut segment: u32 = 0;
            let st = (rt.lib.fn_amd_memory_pool_get_info)(
                pool,
                HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                &mut segment as *mut _ as *mut c_void,
            );
            if st != HSA_STATUS_SUCCESS || segment != HSA_AMD_SEGMENT_GLOBAL {
                return HSA_STATUS_SUCCESS;
            }
            let mut flags: u32 = 0;
            let st = (rt.lib.fn_amd_memory_pool_get_info)(
                pool,
                HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                &mut flags as *mut _ as *mut c_void,
            );
            if st != HSA_STATUS_SUCCESS {
                return HSA_STATUS_SUCCESS;
            }
            let mut alloc_allowed: bool = false;
            let st = (rt.lib.fn_amd_memory_pool_get_info)(
                pool,
                HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                &mut alloc_allowed as *mut _ as *mut c_void,
            );
            if st != HSA_STATUS_SUCCESS || !alloc_allowed {
                return HSA_STATUS_SUCCESS;
            }
            let matches = match ctx.kind {
                PoolKind::Kernarg => flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT != 0,
                PoolKind::FineGrained => {
                    flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED != 0
                        && flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT == 0
                }
                PoolKind::CoarseGrained => {
                    flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED != 0
                }
            };
            if !matches {
                return HSA_STATUS_SUCCESS;
            }
            ctx.found = pool;
            0x1000 // short-circuit: "found"
        }

        let mut ctx = Ctx {
            runtime: Arc::as_ptr(&self.runtime),
            kind,
            found: 0,
        };
        let st = unsafe {
            (self.runtime.lib.fn_amd_agent_iterate_memory_pools)(
                self.handle,
                visit,
                &mut ctx as *mut _ as *mut c_void,
            )
        };
        if ctx.found == 0 {
            return Err(HsaError::new(
                st,
                &format!(
                    "find_pool({:?}) did not match any pool on this agent",
                    ctx.kind
                ),
            ));
        }
        Ok(HsaMemoryPool {
            runtime: self.runtime.clone(),
            handle: ctx.found,
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum PoolKind {
    Kernarg,
    FineGrained,
    CoarseGrained,
}

// ─── Queue ────────────────────────────────────────────────────────────────

pub struct HsaQueue {
    pub(crate) runtime: Arc<HsaRuntime>,
    pub(crate) raw: *mut ffi::HsaQueue,
}

unsafe impl Send for HsaQueue {}
unsafe impl Sync for HsaQueue {}

impl HsaQueue {
    pub fn raw(&self) -> *mut ffi::HsaQueue {
        self.raw
    }

    pub fn size(&self) -> u32 {
        unsafe { (*self.raw).size }
    }

    pub fn doorbell(&self) -> HsaSignalHandle {
        unsafe { (*self.raw).doorbell_signal }
    }

    /// Load the current write index with relaxed ordering.
    pub fn load_write_index_relaxed(&self) -> u64 {
        unsafe { (self.runtime.lib.fn_queue_load_write_index_relaxed)(self.raw) }
    }

    /// Store the new write index with release ordering.
    pub fn store_write_index_release(&self, value: u64) {
        unsafe {
            (self.runtime.lib.fn_queue_store_write_index_release)(self.raw, value);
        }
    }

    /// Get a mutable pointer to the packet slot for this index
    /// (indices wrap modulo queue size).
    pub fn packet_slot(&self, index: u64) -> *mut HsaKernelDispatchPacket {
        let base = unsafe { (*self.raw).base_address as *mut HsaKernelDispatchPacket };
        let size = self.size() as u64;
        unsafe { base.add((index & (size - 1)) as usize) }
    }

    /// Ring the doorbell with the new write index.
    pub fn ring_doorbell(&self, value: u64) {
        let signal = self.doorbell();
        unsafe {
            (self.runtime.lib.fn_signal_store_screlease)(signal, value as i64);
        }
    }
}

impl Drop for HsaQueue {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.runtime.lib.fn_queue_destroy)(self.raw);
        }
    }
}

// ─── Signal ───────────────────────────────────────────────────────────────

pub struct HsaSignal {
    pub(crate) runtime: Arc<HsaRuntime>,
    pub(crate) handle: HsaSignalHandle,
}

unsafe impl Send for HsaSignal {}
unsafe impl Sync for HsaSignal {}

impl HsaSignal {
    pub fn create(runtime: &Arc<HsaRuntime>, initial: i64) -> HsaResult<Self> {
        let mut handle: HsaSignalHandle = 0;
        let st = unsafe { (runtime.lib.fn_signal_create)(initial, 0, ptr::null(), &mut handle) };
        error::check(st, "hsa_signal_create")?;
        Ok(Self {
            runtime: runtime.clone(),
            handle,
        })
    }

    pub fn raw_handle(&self) -> HsaSignalHandle {
        self.handle
    }

    pub fn store_relaxed(&self, value: i64) {
        unsafe {
            (self.runtime.lib.fn_signal_store_relaxed)(self.handle, value);
        }
    }

    pub fn store_screlease(&self, value: i64) {
        unsafe {
            (self.runtime.lib.fn_signal_store_screlease)(self.handle, value);
        }
    }

    pub fn load_relaxed(&self) -> i64 {
        unsafe { (self.runtime.lib.fn_signal_load_relaxed)(self.handle) }
    }

    /// Wait until the signal satisfies `< compare_value` (HSA `LT` condition).
    /// Returns the observed value.
    pub fn wait_lt_scacquire(&self, compare_value: i64, timeout_ns: u64) -> i64 {
        unsafe {
            (self.runtime.lib.fn_signal_wait_scacquire)(
                self.handle,
                HSA_SIGNAL_CONDITION_LT,
                compare_value,
                timeout_ns,
                HSA_WAIT_STATE_BLOCKED,
            )
        }
    }

    /// Active spin wait (HSA `LT` condition, HSA_WAIT_STATE_ACTIVE).
    /// Lower wakeup latency than blocked wait for sub-µs dispatches.
    pub fn wait_lt_active(&self, compare_value: i64, timeout_ns: u64) -> i64 {
        unsafe {
            (self.runtime.lib.fn_signal_wait_scacquire)(
                self.handle,
                HSA_SIGNAL_CONDITION_LT,
                compare_value,
                timeout_ns,
                HSA_WAIT_STATE_ACTIVE,
            )
        }
    }
}

impl Drop for HsaSignal {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.runtime.lib.fn_signal_destroy)(self.handle);
        }
    }
}

// ─── Memory pool ──────────────────────────────────────────────────────────

pub struct HsaMemoryPool {
    pub(crate) runtime: Arc<HsaRuntime>,
    pub(crate) handle: HsaMemoryPoolHandle,
}

impl HsaMemoryPool {
    pub fn raw_handle(&self) -> HsaMemoryPoolHandle {
        self.handle
    }

    /// Allocate `size` bytes from this pool. Returned pointer lives until
    /// `free` or the runtime shuts down.
    pub fn allocate(&self, size: usize) -> HsaResult<*mut u8> {
        let mut ptr: *mut c_void = ptr::null_mut();
        let st = unsafe {
            (self.runtime.lib.fn_amd_memory_pool_allocate)(self.handle, size, 0, &mut ptr)
        };
        error::check(st, "hsa_amd_memory_pool_allocate")?;
        Ok(ptr as *mut u8)
    }

    /// Free a pointer previously returned by `allocate`.
    pub fn free(&self, ptr: *mut u8) -> HsaResult<()> {
        let st = unsafe { (self.runtime.lib.fn_amd_memory_pool_free)(ptr as *mut c_void) };
        error::check(st, "hsa_amd_memory_pool_free")
    }

    /// Grant the given agents access to `ptr`. Required after allocating
    /// from a coarse-grained pool if another agent will read/write it.
    /// HSA requires the `flags` parameter to be NULL (reserved).
    pub fn allow_access(&self, agents: &[&HsaAgent], ptr: *mut u8) -> HsaResult<()> {
        let handles: Vec<HsaAgentHandle> = agents.iter().map(|a| a.handle).collect();
        let st = unsafe {
            (self.runtime.lib.fn_amd_agents_allow_access)(
                handles.len() as u32,
                handles.as_ptr(),
                ptr::null(),
                ptr as *const c_void,
            )
        };
        error::check(st, "hsa_amd_agents_allow_access")
    }
}

// ─── Executable / kernel loading ─────────────────────────────────────────

pub struct HsaExecutable {
    runtime: Arc<HsaRuntime>,
    handle: HsaExecutableHandle,
    reader: HsaCodeObjectReaderHandle,
    frozen: bool,
}

impl HsaExecutable {
    /// Parse a `.hsaco` code object from memory and bind it to `agent`.
    /// Must call `freeze()` before using kernels.
    pub fn from_code_object(agent: &HsaAgent, hsaco_bytes: &[u8]) -> HsaResult<Self> {
        let runtime = agent.runtime.clone();

        // Unwrap Clang offload bundle if present.
        let bytes: &[u8] =
            if hsaco_bytes.len() > 24 && &hsaco_bytes[0..24] == b"__CLANG_OFFLOAD_BUNDLE__" {
                const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
                if let Some(pos) = hsaco_bytes.windows(4).position(|w| w == ELF_MAGIC) {
                    &hsaco_bytes[pos..]
                } else {
                    return Err(HsaError::new(0, "clang offload bundle contains no ELF"));
                }
            } else {
                hsaco_bytes
            };

        // Create the code object reader.
        let mut reader: HsaCodeObjectReaderHandle = 0;
        let st = unsafe {
            (runtime.lib.fn_code_object_reader_create_from_memory)(
                bytes.as_ptr() as *const c_void,
                bytes.len(),
                &mut reader,
            )
        };
        error::check(st, "hsa_code_object_reader_create_from_memory")?;

        // Create the executable.
        let mut handle: HsaExecutableHandle = 0;
        let st = unsafe {
            (runtime.lib.fn_executable_create_alt)(
                HSA_PROFILE_FULL,
                HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                ptr::null(),
                &mut handle,
            )
        };
        if st != HSA_STATUS_SUCCESS {
            unsafe {
                let _ = (runtime.lib.fn_code_object_reader_destroy)(reader);
            }
            return Err(HsaError::new(st, "hsa_executable_create_alt"));
        }

        // Load the code object onto the agent.
        let st = unsafe {
            (runtime.lib.fn_executable_load_agent_code_object)(
                handle,
                agent.handle,
                reader,
                ptr::null(),
                ptr::null_mut(),
            )
        };
        if st != HSA_STATUS_SUCCESS {
            unsafe {
                let _ = (runtime.lib.fn_executable_destroy)(handle);
                let _ = (runtime.lib.fn_code_object_reader_destroy)(reader);
            }
            return Err(HsaError::new(st, "hsa_executable_load_agent_code_object"));
        }

        Ok(Self {
            runtime,
            handle,
            reader,
            frozen: false,
        })
    }

    /// Finalize the executable — required before kernel lookup.
    pub fn freeze(&mut self) -> HsaResult<()> {
        let st = unsafe { (self.runtime.lib.fn_executable_freeze)(self.handle, ptr::null()) };
        error::check(st, "hsa_executable_freeze")?;
        self.frozen = true;
        Ok(())
    }

    /// Look up a kernel by its symbol name and return the fields an AQL
    /// packet needs. `name` should NOT include the `.kd` suffix — this
    /// function appends it if missing (matches standard hipcc output).
    pub fn kernel(&self, agent: &HsaAgent, name: &str) -> HsaResult<HsaKernel> {
        if !self.frozen {
            return Err(HsaError::new(
                0,
                "executable must be frozen before kernel lookup",
            ));
        }
        let full_name = if name.ends_with(".kd") {
            name.to_string()
        } else {
            format!("{name}.kd")
        };
        let c_name = CString::new(full_name.clone())
            .map_err(|_| HsaError::new(0, "invalid kernel name (contains NUL)"))?;

        let mut symbol: HsaExecutableSymbolHandle = 0;
        let st = unsafe {
            (self.runtime.lib.fn_executable_get_symbol_by_name)(
                self.handle,
                c_name.as_ptr(),
                &agent.handle,
                &mut symbol,
            )
        };
        error::check(st, &format!("get_symbol_by_name({full_name})"))?;

        let mut kernel_object: u64 = 0;
        let st = unsafe {
            (self.runtime.lib.fn_executable_symbol_get_info)(
                symbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                &mut kernel_object as *mut _ as *mut c_void,
            )
        };
        error::check(st, "executable_symbol_get_info(KERNEL_OBJECT)")?;

        let mut kernarg_size: u32 = 0;
        let st = unsafe {
            (self.runtime.lib.fn_executable_symbol_get_info)(
                symbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                &mut kernarg_size as *mut _ as *mut c_void,
            )
        };
        error::check(st, "executable_symbol_get_info(KERNARG_SEGMENT_SIZE)")?;

        let mut group_segment_size: u32 = 0;
        let st = unsafe {
            (self.runtime.lib.fn_executable_symbol_get_info)(
                symbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                &mut group_segment_size as *mut _ as *mut c_void,
            )
        };
        error::check(st, "executable_symbol_get_info(GROUP_SEGMENT_SIZE)")?;

        let mut private_segment_size: u32 = 0;
        let st = unsafe {
            (self.runtime.lib.fn_executable_symbol_get_info)(
                symbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                &mut private_segment_size as *mut _ as *mut c_void,
            )
        };
        error::check(st, "executable_symbol_get_info(PRIVATE_SEGMENT_SIZE)")?;

        Ok(HsaKernel {
            name: name.to_string(),
            kernel_object,
            kernarg_size,
            group_segment_size,
            private_segment_size,
        })
    }
}

impl Drop for HsaExecutable {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.runtime.lib.fn_executable_destroy)(self.handle);
            let _ = (self.runtime.lib.fn_code_object_reader_destroy)(self.reader);
        }
    }
}

/// Metadata for a loaded kernel, ready to stuff into an AQL packet.
#[derive(Debug, Clone)]
pub struct HsaKernel {
    pub name: String,
    pub kernel_object: u64,
    pub kernarg_size: u32,
    pub group_segment_size: u32,
    pub private_segment_size: u32,
}

// ─── AQL dispatch helpers ─────────────────────────────────────────────────

/// Build the header word for a kernel dispatch packet with system-scope
/// acquire + release fences and the barrier bit set. This is the standard
/// HIP/HSA dispatch header.
#[inline]
pub fn dispatch_packet_header() -> u16 {
    (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE)
        | (1 << HSA_PACKET_HEADER_BARRIER)
        | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
        | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE)
}

/// Fill an AQL packet, leaving the header for the caller to store last.
///
/// Sets completion_signal to 0 by default; caller can overwrite it after
/// this returns.
///
/// # Safety
///
/// `slot` must be valid and uniquely writable for one
/// `HsaKernelDispatchPacket`. The caller is responsible for publishing the
/// packet header after all fields are written.
#[inline]
pub unsafe fn build_dispatch_packet(
    slot: *mut HsaKernelDispatchPacket,
    kernel: &HsaKernel,
    grid: [u32; 3],
    block: [u32; 3],
    kernarg_ptr: *mut u8,
    completion_signal: HsaSignalHandle,
) {
    let ndims = if grid[2] > 1 {
        3u16
    } else if grid[1] > 1 {
        2
    } else {
        1
    };
    let p = unsafe { &mut *slot };
    // Header is written LAST with release ordering; leave it whatever it is
    // for now (the HSA runtime pre-fills HSA_PACKET_TYPE_INVALID).
    p.setup = ndims;
    p.workgroup_size_x = block[0] as u16;
    p.workgroup_size_y = block[1] as u16;
    p.workgroup_size_z = block[2] as u16;
    p.reserved0 = 0;
    p.grid_size_x = grid[0].saturating_mul(block[0]);
    p.grid_size_y = grid[1].saturating_mul(block[1]);
    p.grid_size_z = grid[2].saturating_mul(block[2]);
    p.private_segment_size = kernel.private_segment_size;
    p.group_segment_size = kernel.group_segment_size;
    p.kernel_object = kernel.kernel_object;
    p.kernarg_address = kernarg_ptr as *mut c_void;
    p.reserved2 = 0;
    p.completion_signal = completion_signal;
}

/// Atomic release-store of the header word. Must be called after the rest
/// of the packet is filled in. Makes the packet visible to the AQL engine.
///
/// # Safety
///
/// `slot` must point to a valid AQL packet whose non-header fields are fully
/// initialized. Publishing the header makes the packet visible to the queue's
/// AQL engine, so callers must not mutate it after this call.
#[inline]
pub unsafe fn publish_dispatch_packet(slot: *mut HsaKernelDispatchPacket, header: u16) {
    use std::sync::atomic::{fence, AtomicU16, Ordering};
    unsafe {
        fence(Ordering::Release);
        let header_atomic = &*(slot as *const AtomicU16);
        header_atomic.store(header, Ordering::Release);
    }
}
