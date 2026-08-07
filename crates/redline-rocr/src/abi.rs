//! Minimal ROCr/HSA ABI used by the AQL backend.
//!
//! Values and layouts are copied from the MIT-licensed ROCm 7.2 public headers
//! `/opt/rocm/include/hsa/{hsa.h,hsa_ext_amd.h}`, cross-checked against
//! `rocm-systems` commit `c0430a50286200ab0562f4733445cdee6e48d416`.
//! Keeping this module small is intentional: it is an auditable dynamic-ABI
//! boundary, not a replacement for generated HSA bindings.

use std::ffi::{CStr, c_char, c_void};
use std::fmt;
use std::sync::Arc;

pub type Status = i32;
pub type SignalValue = i64;

pub const STATUS_SUCCESS: Status = 0;
pub const STATUS_INFO_BREAK: Status = 1;

pub const SYSTEM_INFO_TIMESTAMP_FREQUENCY: u32 = 3;

pub const DEVICE_TYPE_CPU: u32 = 0;
pub const DEVICE_TYPE_GPU: u32 = 1;

pub const AGENT_INFO_NAME: u32 = 0;
pub const AGENT_INFO_PROFILE: u32 = 4;
pub const AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE: u32 = 5;
pub const AGENT_INFO_WORKGROUP_MAX_DIM: u32 = 7;
pub const AGENT_INFO_WORKGROUP_MAX_SIZE: u32 = 8;
pub const AGENT_INFO_GRID_MAX_DIM: u32 = 9;
pub const AGENT_INFO_GRID_MAX_SIZE: u32 = 10;
pub const AGENT_INFO_QUEUE_MIN_SIZE: u32 = 13;
pub const AGENT_INFO_QUEUE_MAX_SIZE: u32 = 14;
pub const AGENT_INFO_QUEUE_TYPE: u32 = 15;
pub const AGENT_INFO_DEVICE: u32 = 17;
pub const AMD_AGENT_INFO_BDFID: u32 = 0xA006;
pub const AMD_AGENT_INFO_DOMAIN: u32 = 0xA00F;
pub const AMD_AGENT_INFO_TIMESTAMP_FREQUENCY: u32 = 0xA016;

pub const PROFILE_BASE: u32 = 0;
pub const PROFILE_FULL: u32 = 1;
pub const DEFAULT_FLOAT_ROUNDING_MODE_ZERO: u32 = 1;
pub const DEFAULT_FLOAT_ROUNDING_MODE_NEAR: u32 = 2;

pub const QUEUE_TYPE_MULTI: u32 = 0;
pub const QUEUE_TYPE_SINGLE: u32 = 1;
pub const QUEUE_FEATURE_KERNEL_DISPATCH: u32 = 1;

pub const PACKET_TYPE_INVALID: u16 = 1;
pub const PACKET_TYPE_KERNEL_DISPATCH: u16 = 2;
pub const PACKET_TYPE_BARRIER_AND: u16 = 3;

pub const PACKET_HEADER_TYPE: u16 = 0;
pub const PACKET_HEADER_BARRIER: u16 = 8;
pub const PACKET_HEADER_SCACQUIRE_FENCE_SCOPE: u16 = 9;
pub const PACKET_HEADER_SCRELEASE_FENCE_SCOPE: u16 = 11;

pub const FENCE_SCOPE_NONE: u16 = 0;
pub const FENCE_SCOPE_AGENT: u16 = 1;
pub const FENCE_SCOPE_SYSTEM: u16 = 2;

pub const KERNEL_DISPATCH_SETUP_DIMENSIONS: u16 = 0;

pub const AMD_SEGMENT_GLOBAL: u32 = 0;
pub const AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT: u32 = 1;
pub const AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED: u32 = 2;
/// `HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED` — device-local placement.
/// Diagnostic only: the retained IB pool is expected to be fine-grained.
pub const AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED: u32 = 4;
pub const AMD_MEMORY_POOL_INFO_SEGMENT: u32 = 0;
pub const AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS: u32 = 1;
/// `HSA_AMD_MEMORY_POOL_INFO_SIZE`.
pub const AMD_MEMORY_POOL_INFO_SIZE: u32 = 2;
pub const AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED: u32 = 5;
pub const AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE: u32 = 6;
pub const AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT: u32 = 7;
pub const AMD_MEMORY_POOL_STANDARD_FLAG: u32 = 0;
pub const AMD_MEMORY_POOL_EXECUTABLE_FLAG: u32 = 1 << 2;

pub const EXECUTABLE_SYMBOL_INFO_TYPE: u32 = 0;
pub const SYMBOL_KIND_KERNEL: u32 = 1;
pub const EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 11;
pub const EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT: u32 = 12;
pub const EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 13;
pub const EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 14;
pub const EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK: u32 = 15;
pub const EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Agent(pub u64);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Signal(pub u64);

/// Public `hsa_amd_ipc_signal_t` layout. ROCr aliases this to the 256-bit
/// `hsa_amd_ipc_memory_t` process-independent handle.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IpcSignalHandle {
    pub words: [u32; 8],
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MemoryPool(pub u64);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CodeObjectReader(pub u64);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Executable(pub u64);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LoadedCodeObject(pub u64);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExecutableSymbol(pub u64);

/// Public `hsa_queue_t` layout for the 64-bit HSA machine model.
#[repr(C)]
#[derive(Debug)]
pub struct Queue {
    pub queue_type: u32,
    pub features: u32,
    pub base_address: *mut c_void,
    pub doorbell_signal: Signal,
    pub size: u32,
    pub reserved1: u32,
    pub id: u64,
}

/// Public `hsa_amd_profiling_dispatch_time_t` layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProfilingDispatchTime {
    pub start: u64,
    pub end: u64,
}

/// Public `hsa_amd_profiling_async_copy_time_t` layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProfilingAsyncCopyTime {
    pub start: u64,
    pub end: u64,
}

pub type AgentCallback = unsafe extern "C" fn(Agent, *mut c_void) -> Status;
pub type MemoryPoolCallback = unsafe extern "C" fn(MemoryPool, *mut c_void) -> Status;
pub type QueueErrorCallback = unsafe extern "C" fn(Status, *mut Queue, *mut c_void);

pub type InitFn = unsafe extern "C" fn() -> Status;
pub type ShutDownFn = unsafe extern "C" fn() -> Status;
pub type SystemGetInfoFn = unsafe extern "C" fn(u32, *mut c_void) -> Status;
pub type StatusStringFn = unsafe extern "C" fn(Status, *mut *const c_char) -> Status;
pub type IterateAgentsFn = unsafe extern "C" fn(Option<AgentCallback>, *mut c_void) -> Status;
pub type AgentGetInfoFn = unsafe extern "C" fn(Agent, u32, *mut c_void) -> Status;

pub type SignalCreateFn =
    unsafe extern "C" fn(SignalValue, u32, *const Agent, *mut Signal) -> Status;
pub type SignalDestroyFn = unsafe extern "C" fn(Signal) -> Status;
pub type SignalLoadScAcquireFn = unsafe extern "C" fn(Signal) -> SignalValue;
pub type SignalStoreRelaxedFn = unsafe extern "C" fn(Signal, SignalValue);
pub type SignalStoreScReleaseFn = unsafe extern "C" fn(Signal, SignalValue);
pub type AmdSignalCreateFn =
    unsafe extern "C" fn(SignalValue, u32, *const Agent, u64, *mut Signal) -> Status;
pub type IpcSignalCreateFn = unsafe extern "C" fn(Signal, *mut IpcSignalHandle) -> Status;
pub type IpcSignalAttachFn = unsafe extern "C" fn(*const IpcSignalHandle, *mut Signal) -> Status;

pub type QueueCreateFn = unsafe extern "C" fn(
    Agent,
    u32,
    u32,
    Option<QueueErrorCallback>,
    *mut c_void,
    u32,
    u32,
    *mut *mut Queue,
) -> Status;
pub type QueueDestroyFn = unsafe extern "C" fn(*mut Queue) -> Status;
pub type QueueInactivateFn = unsafe extern "C" fn(*mut Queue) -> Status;
pub type QueueLoadReadIndexRelaxedFn = unsafe extern "C" fn(*const Queue) -> u64;
pub type QueueLoadReadIndexScAcquireFn = unsafe extern "C" fn(*const Queue) -> u64;
pub type QueueLoadWriteIndexRelaxedFn = unsafe extern "C" fn(*const Queue) -> u64;
pub type QueueAddWriteIndexRelaxedFn = unsafe extern "C" fn(*const Queue, u64) -> u64;
pub type QueueCuSetMaskFn = unsafe extern "C" fn(*const Queue, u32, *const u32) -> Status;
pub type ProfilingSetProfilerEnabledFn = unsafe extern "C" fn(*mut Queue, i32) -> Status;
pub type ProfilingGetDispatchTimeFn =
    unsafe extern "C" fn(Agent, Signal, *mut ProfilingDispatchTime) -> Status;
pub type ProfilingAsyncCopyEnableFn = unsafe extern "C" fn(bool) -> Status;
pub type ProfilingGetAsyncCopyTimeFn =
    unsafe extern "C" fn(Signal, *mut ProfilingAsyncCopyTime) -> Status;

pub type AgentIterateMemoryPoolsFn =
    unsafe extern "C" fn(Agent, Option<MemoryPoolCallback>, *mut c_void) -> Status;
pub type MemoryPoolGetInfoFn = unsafe extern "C" fn(MemoryPool, u32, *mut c_void) -> Status;
pub type MemoryPoolAllocateFn =
    unsafe extern "C" fn(MemoryPool, usize, u32, *mut *mut c_void) -> Status;
pub type MemoryPoolFreeFn = unsafe extern "C" fn(*mut c_void) -> Status;
pub type AgentsAllowAccessFn =
    unsafe extern "C" fn(u32, *const Agent, *const u32, *const c_void) -> Status;
pub type MemoryAsyncCopyFn = unsafe extern "C" fn(
    *mut c_void,
    Agent,
    *const c_void,
    Agent,
    usize,
    u32,
    *const Signal,
    Signal,
) -> Status;
pub type MemoryAsyncCopyOnEngineFn = unsafe extern "C" fn(
    *mut c_void,
    Agent,
    *const c_void,
    Agent,
    usize,
    u32,
    *const Signal,
    Signal,
    u32,
    bool,
) -> Status;
pub type MemoryCopyEngineStatusFn = unsafe extern "C" fn(Agent, Agent, *mut u32) -> Status;
pub type MemoryGetPreferredCopyEngineFn = unsafe extern "C" fn(Agent, Agent, *mut u32) -> Status;

pub type CodeObjectReaderCreateFromMemoryFn =
    unsafe extern "C" fn(*const c_void, usize, *mut CodeObjectReader) -> Status;
pub type CodeObjectReaderDestroyFn = unsafe extern "C" fn(CodeObjectReader) -> Status;
pub type ExecutableCreateAltFn =
    unsafe extern "C" fn(u32, u32, *const c_char, *mut Executable) -> Status;
pub type ExecutableDestroyFn = unsafe extern "C" fn(Executable) -> Status;
pub type ExecutableLoadAgentCodeObjectFn = unsafe extern "C" fn(
    Executable,
    Agent,
    CodeObjectReader,
    *const c_char,
    *mut LoadedCodeObject,
) -> Status;
pub type ExecutableFreezeFn = unsafe extern "C" fn(Executable, *const c_char) -> Status;
pub type ExecutableGetSymbolByNameFn =
    unsafe extern "C" fn(Executable, *const c_char, *const Agent, *mut ExecutableSymbol) -> Status;
pub type ExecutableSymbolGetInfoFn =
    unsafe extern "C" fn(ExecutableSymbol, u32, *mut c_void) -> Status;

/// Dynamically resolved public ROCr entry points.
///
/// The `keepalive` object normally contains an `Arc<libloading::Library>`.
/// Keeping it here guarantees that no stored function pointer outlives the
/// loaded `libhsa-runtime64.so` image.
pub struct Symbols {
    _keepalive: Arc<dyn Send + Sync>,
    pub init: InitFn,
    pub shut_down: ShutDownFn,
    pub system_get_info: SystemGetInfoFn,
    pub status_string: StatusStringFn,
    pub iterate_agents: IterateAgentsFn,
    pub agent_get_info: AgentGetInfoFn,
    pub signal_create: SignalCreateFn,
    pub signal_destroy: SignalDestroyFn,
    pub signal_load_scacquire: SignalLoadScAcquireFn,
    pub signal_store_relaxed: SignalStoreRelaxedFn,
    pub signal_store_screlease: SignalStoreScReleaseFn,
    pub amd_signal_create: Option<AmdSignalCreateFn>,
    pub ipc_signal_create: Option<IpcSignalCreateFn>,
    pub ipc_signal_attach: Option<IpcSignalAttachFn>,
    pub queue_create: QueueCreateFn,
    pub queue_destroy: QueueDestroyFn,
    pub queue_inactivate: QueueInactivateFn,
    pub queue_load_read_index_relaxed: QueueLoadReadIndexRelaxedFn,
    pub queue_load_read_index_scacquire: QueueLoadReadIndexScAcquireFn,
    pub queue_load_write_index_relaxed: QueueLoadWriteIndexRelaxedFn,
    pub queue_add_write_index_relaxed: QueueAddWriteIndexRelaxedFn,
    pub queue_cu_set_mask: QueueCuSetMaskFn,
    pub profiling_set_profiler_enabled: ProfilingSetProfilerEnabledFn,
    pub profiling_get_dispatch_time: ProfilingGetDispatchTimeFn,
    pub profiling_async_copy_enable: Option<ProfilingAsyncCopyEnableFn>,
    pub profiling_get_async_copy_time: Option<ProfilingGetAsyncCopyTimeFn>,
    pub agent_iterate_memory_pools: AgentIterateMemoryPoolsFn,
    pub memory_pool_get_info: MemoryPoolGetInfoFn,
    pub memory_pool_allocate: MemoryPoolAllocateFn,
    pub memory_pool_free: MemoryPoolFreeFn,
    pub agents_allow_access: AgentsAllowAccessFn,
    pub memory_async_copy: MemoryAsyncCopyFn,
    pub memory_async_copy_on_engine: MemoryAsyncCopyOnEngineFn,
    pub memory_copy_engine_status: MemoryCopyEngineStatusFn,
    pub memory_get_preferred_copy_engine: MemoryGetPreferredCopyEngineFn,
    pub code_object_reader_create_from_memory: CodeObjectReaderCreateFromMemoryFn,
    pub code_object_reader_destroy: CodeObjectReaderDestroyFn,
    pub executable_create_alt: ExecutableCreateAltFn,
    pub executable_destroy: ExecutableDestroyFn,
    pub executable_load_agent_code_object: ExecutableLoadAgentCodeObjectFn,
    pub executable_freeze: ExecutableFreezeFn,
    pub executable_get_symbol_by_name: ExecutableGetSymbolByNameFn,
    pub executable_symbol_get_info: ExecutableSymbolGetInfoFn,
}

impl fmt::Debug for Symbols {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Symbols").finish_non_exhaustive()
    }
}

impl Symbols {
    /// Resolve the public symbols used by this backend.
    ///
    /// # Safety
    ///
    /// Each non-null pointer returned by `resolve` must name the C function
    /// requested by the supplied symbol name, with the exact ROCm 7.2 ABI.
    /// `keepalive` must keep the containing shared object mapped for its own
    /// lifetime. Loading symbols from a different, ABI-incompatible runtime is
    /// undefined behavior.
    pub unsafe fn load(
        keepalive: Arc<dyn Send + Sync>,
        mut resolve: impl FnMut(&CStr) -> *const c_void,
    ) -> Result<Arc<Self>, MissingSymbol> {
        macro_rules! symbol {
            ($name:literal, $ty:ty) => {{
                let name = CStr::from_bytes_with_nul(concat!($name, "\0").as_bytes())
                    .expect("symbol name has one trailing NUL");
                let pointer = resolve(name);
                if pointer.is_null() {
                    return Err(MissingSymbol($name));
                }
                // SAFETY: guaranteed by this function's contract. The macro
                // names a concrete function-pointer type, which is pointer-sized.
                unsafe { std::mem::transmute::<*const c_void, $ty>(pointer) }
            }};
        }
        macro_rules! optional_symbol {
            ($name:literal, $ty:ty) => {{
                let name = CStr::from_bytes_with_nul(concat!($name, "\0").as_bytes())
                    .expect("symbol name has one trailing NUL");
                let pointer = resolve(name);
                if pointer.is_null() {
                    None
                } else {
                    // SAFETY: same contract as `symbol!`; absence remains a
                    // runtime capability error instead of breaking all ROCr
                    // users on older libraries.
                    Some(unsafe { std::mem::transmute::<*const c_void, $ty>(pointer) })
                }
            }};
        }

        Ok(Arc::new(Self {
            _keepalive: keepalive,
            init: symbol!("hsa_init", InitFn),
            shut_down: symbol!("hsa_shut_down", ShutDownFn),
            system_get_info: symbol!("hsa_system_get_info", SystemGetInfoFn),
            status_string: symbol!("hsa_status_string", StatusStringFn),
            iterate_agents: symbol!("hsa_iterate_agents", IterateAgentsFn),
            agent_get_info: symbol!("hsa_agent_get_info", AgentGetInfoFn),
            signal_create: symbol!("hsa_signal_create", SignalCreateFn),
            signal_destroy: symbol!("hsa_signal_destroy", SignalDestroyFn),
            signal_load_scacquire: symbol!("hsa_signal_load_scacquire", SignalLoadScAcquireFn),
            signal_store_relaxed: symbol!("hsa_signal_store_relaxed", SignalStoreRelaxedFn),
            signal_store_screlease: symbol!("hsa_signal_store_screlease", SignalStoreScReleaseFn),
            amd_signal_create: optional_symbol!("hsa_amd_signal_create", AmdSignalCreateFn),
            ipc_signal_create: optional_symbol!("hsa_amd_ipc_signal_create", IpcSignalCreateFn),
            ipc_signal_attach: optional_symbol!("hsa_amd_ipc_signal_attach", IpcSignalAttachFn),
            queue_create: symbol!("hsa_queue_create", QueueCreateFn),
            queue_destroy: symbol!("hsa_queue_destroy", QueueDestroyFn),
            queue_inactivate: symbol!("hsa_queue_inactivate", QueueInactivateFn),
            queue_load_read_index_relaxed: symbol!(
                "hsa_queue_load_read_index_relaxed",
                QueueLoadReadIndexRelaxedFn
            ),
            queue_load_read_index_scacquire: symbol!(
                "hsa_queue_load_read_index_scacquire",
                QueueLoadReadIndexScAcquireFn
            ),
            queue_load_write_index_relaxed: symbol!(
                "hsa_queue_load_write_index_relaxed",
                QueueLoadWriteIndexRelaxedFn
            ),
            queue_add_write_index_relaxed: symbol!(
                "hsa_queue_add_write_index_relaxed",
                QueueAddWriteIndexRelaxedFn
            ),
            queue_cu_set_mask: symbol!("hsa_amd_queue_cu_set_mask", QueueCuSetMaskFn),
            profiling_set_profiler_enabled: symbol!(
                "hsa_amd_profiling_set_profiler_enabled",
                ProfilingSetProfilerEnabledFn
            ),
            profiling_get_dispatch_time: symbol!(
                "hsa_amd_profiling_get_dispatch_time",
                ProfilingGetDispatchTimeFn
            ),
            profiling_async_copy_enable: optional_symbol!(
                "hsa_amd_profiling_async_copy_enable",
                ProfilingAsyncCopyEnableFn
            ),
            profiling_get_async_copy_time: optional_symbol!(
                "hsa_amd_profiling_get_async_copy_time",
                ProfilingGetAsyncCopyTimeFn
            ),
            agent_iterate_memory_pools: symbol!(
                "hsa_amd_agent_iterate_memory_pools",
                AgentIterateMemoryPoolsFn
            ),
            memory_pool_get_info: symbol!("hsa_amd_memory_pool_get_info", MemoryPoolGetInfoFn),
            memory_pool_allocate: symbol!("hsa_amd_memory_pool_allocate", MemoryPoolAllocateFn),
            memory_pool_free: symbol!("hsa_amd_memory_pool_free", MemoryPoolFreeFn),
            agents_allow_access: symbol!("hsa_amd_agents_allow_access", AgentsAllowAccessFn),
            memory_async_copy: symbol!("hsa_amd_memory_async_copy", MemoryAsyncCopyFn),
            memory_async_copy_on_engine: symbol!(
                "hsa_amd_memory_async_copy_on_engine",
                MemoryAsyncCopyOnEngineFn
            ),
            memory_copy_engine_status: symbol!(
                "hsa_amd_memory_copy_engine_status",
                MemoryCopyEngineStatusFn
            ),
            memory_get_preferred_copy_engine: symbol!(
                "hsa_amd_memory_get_preferred_copy_engine",
                MemoryGetPreferredCopyEngineFn
            ),
            code_object_reader_create_from_memory: symbol!(
                "hsa_code_object_reader_create_from_memory",
                CodeObjectReaderCreateFromMemoryFn
            ),
            code_object_reader_destroy: symbol!(
                "hsa_code_object_reader_destroy",
                CodeObjectReaderDestroyFn
            ),
            executable_create_alt: symbol!("hsa_executable_create_alt", ExecutableCreateAltFn),
            executable_destroy: symbol!("hsa_executable_destroy", ExecutableDestroyFn),
            executable_load_agent_code_object: symbol!(
                "hsa_executable_load_agent_code_object",
                ExecutableLoadAgentCodeObjectFn
            ),
            executable_freeze: symbol!("hsa_executable_freeze", ExecutableFreezeFn),
            executable_get_symbol_by_name: symbol!(
                "hsa_executable_get_symbol_by_name",
                ExecutableGetSymbolByNameFn
            ),
            executable_symbol_get_info: symbol!(
                "hsa_executable_symbol_get_info",
                ExecutableSymbolGetInfoFn
            ),
        }))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MissingSymbol(pub &'static str);

impl fmt::Display for MissingSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "libhsa-runtime64 is missing {}", self.0)
    }
}

impl std::error::Error for MissingSymbol {}

const _: () = {
    assert!(std::mem::size_of::<usize>() == 8);
    assert!(std::mem::size_of::<Agent>() == 8);
    assert!(std::mem::size_of::<Signal>() == 8);
    assert!(std::mem::size_of::<IpcSignalHandle>() == 32);
    assert!(std::mem::align_of::<IpcSignalHandle>() == 4);
    assert!(std::mem::size_of::<MemoryPool>() == 8);
    assert!(std::mem::size_of::<Queue>() == 40);
    assert!(std::mem::align_of::<Queue>() == 8);
    assert!(std::mem::size_of::<ProfilingDispatchTime>() == 16);
    assert!(std::mem::size_of::<ProfilingAsyncCopyTime>() == 16);
};
