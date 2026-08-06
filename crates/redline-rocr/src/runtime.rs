//! Public ROCr resource management and queue publication.

use std::ffi::{CStr, CString, c_void};
use std::fmt;
use std::ptr::{self, NonNull};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::abi;
use super::packet::{AQL_PACKET_BYTES, KernelMetadata, LaunchGeometry, PacketImage};

/// Default bound for host-side queue-capacity and completion polling.
///
/// Callers that need a different completion budget can use the replay ticket's
/// explicit timeout method. Queue capacity should normally be immediately
/// available because a graph permits only one replay in flight. This does not
/// bound the foreign queue inactivation/destruction calls, for which the public
/// HSA API exposes no timeout.
pub const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(5);

const QUEUE_FAULT_NONE: u64 = 0;

#[derive(Default)]
struct QueueFaultState {
    // Bit 63 distinguishes "no callback" from every possible 32-bit HSA status.
    encoded: AtomicU64,
}

impl QueueFaultState {
    fn record(&self, status: abi::Status) {
        let encoded = (1_u64 << 63) | u64::from(status as u32);
        let _ = self.encoded.compare_exchange(
            QUEUE_FAULT_NONE,
            encoded,
            Ordering::Release,
            Ordering::Relaxed,
        );
    }

    fn status(&self) -> Option<abi::Status> {
        let encoded = self.encoded.load(Ordering::Acquire);
        (encoded != QUEUE_FAULT_NONE).then_some(encoded as u32 as abi::Status)
    }
}

unsafe extern "C" fn queue_error_callback(
    status: abi::Status,
    _source: *mut abi::Queue,
    data: *mut c_void,
) {
    if data.is_null() {
        return;
    }
    // SAFETY: `data` points into the queue-owned Box passed to
    // `hsa_queue_create`. The Box is stable until after queue destruction.
    unsafe { &*data.cast::<QueueFaultState>() }.record(status);
}

#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

struct RuntimeInner {
    symbols: Arc<abi::Symbols>,
}

impl Runtime {
    pub fn initialize(symbols: Arc<abi::Symbols>) -> Result<Self, RuntimeError> {
        // SAFETY: `Symbols::load` established the function ABI and keeps the
        // dynamic library mapped.
        let status = unsafe { (symbols.init)() };
        check_status(&symbols, "hsa_init", status)?;
        Ok(Self {
            inner: Arc::new(RuntimeInner { symbols }),
        })
    }

    pub fn gpu_devices(&self) -> Result<Vec<GpuDevice>, RuntimeError> {
        let agents = self.agents()?;
        let cpu = agents
            .iter()
            .find(|agent| agent.device_type == abi::DEVICE_TYPE_CPU)
            .cloned();
        let devices = agents
            .into_iter()
            .filter(|agent| agent.device_type == abi::DEVICE_TYPE_GPU)
            .map(|gpu| GpuDevice {
                runtime: self.inner.clone(),
                gpu,
                cpu: cpu.clone(),
            })
            .collect::<Vec<_>>();
        if devices.is_empty() {
            return Err(RuntimeError::NoGpuAgent);
        }
        Ok(devices)
    }

    pub fn select_gpu(&self, selector: GpuSelector<'_>) -> Result<GpuDevice, RuntimeError> {
        let devices = self.gpu_devices()?;
        match selector {
            GpuSelector::Ordinal(ordinal) => devices
                .into_iter()
                .nth(ordinal)
                .ok_or(RuntimeError::GpuOrdinalOutOfRange { ordinal }),
            GpuSelector::NameContains(needle) => {
                let needle_lower = needle.to_ascii_lowercase();
                devices
                    .into_iter()
                    .find(|device| device.name().to_ascii_lowercase().contains(&needle_lower))
                    .ok_or_else(|| RuntimeError::GpuNameNotFound {
                        needle: needle.to_owned(),
                    })
            }
        }
    }

    /// HSA system-clock frequency used by ROCr dispatch profiling timestamps.
    pub fn timestamp_frequency_hz(&self) -> Result<u64, RuntimeError> {
        let mut frequency = 0_u64;
        // SAFETY: the selected attribute writes one `u64` to the supplied
        // output pointer, as specified by the public HSA header.
        let status = unsafe {
            (self.inner.symbols.system_get_info)(
                abi::SYSTEM_INFO_TIMESTAMP_FREQUENCY,
                (&mut frequency as *mut u64).cast(),
            )
        };
        check_status(
            &self.inner.symbols,
            "hsa_system_get_info(timestamp frequency)",
            status,
        )?;
        if frequency == 0 {
            return Err(RuntimeError::InvalidRuntimeObject(
                "HSA timestamp frequency is zero",
            ));
        }
        Ok(frequency)
    }

    fn agents(&self) -> Result<Vec<AgentInfo>, RuntimeError> {
        unsafe extern "C" fn collect(agent: abi::Agent, data: *mut c_void) -> abi::Status {
            // SAFETY: `data` points at the live vector below for the synchronous
            // duration of `hsa_iterate_agents`.
            let agents = unsafe { &mut *(data.cast::<Vec<abi::Agent>>()) };
            agents.push(agent);
            abi::STATUS_SUCCESS
        }

        let mut handles = Vec::new();
        // SAFETY: callback and context satisfy the synchronous HSA iteration
        // contract.
        let status = unsafe {
            (self.inner.symbols.iterate_agents)(
                Some(collect),
                (&mut handles as *mut Vec<abi::Agent>).cast(),
            )
        };
        check_status(&self.inner.symbols, "hsa_iterate_agents", status)?;
        handles
            .into_iter()
            .map(|agent| self.agent_info(agent))
            .collect()
    }

    fn agent_info(&self, agent: abi::Agent) -> Result<AgentInfo, RuntimeError> {
        let symbols = &self.inner.symbols;
        let mut name = [0_u8; 64];
        let mut device_type = 0_u32;
        let mut profile = 0_u32;
        let mut rounding = 0_u32;
        let mut queue_min_size = 0_u32;
        let mut queue_max_size = 0_u32;
        let mut queue_type = 0_u32;
        let mut workgroup_max_dim = [0_u16; 3];
        let mut workgroup_max_size = 0_u32;
        let mut grid_max_dim = [0_u32; 3];
        let mut grid_max_size = 0_u32;
        let mut pci_domain = 0_u32;
        let mut pci_bdfid = 0_u32;
        query_agent(
            symbols,
            agent,
            abi::AGENT_INFO_NAME,
            name.as_mut_ptr().cast(),
        )?;
        query_agent(
            symbols,
            agent,
            abi::AGENT_INFO_DEVICE,
            (&mut device_type as *mut u32).cast(),
        )?;
        if device_type == abi::DEVICE_TYPE_GPU {
            query_agent(
                symbols,
                agent,
                abi::AGENT_INFO_PROFILE,
                (&mut profile as *mut u32).cast(),
            )?;
            query_agent(
                symbols,
                agent,
                abi::AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE,
                (&mut rounding as *mut u32).cast(),
            )?;
            query_agent(
                symbols,
                agent,
                abi::AGENT_INFO_QUEUE_MIN_SIZE,
                (&mut queue_min_size as *mut u32).cast(),
            )?;
            query_agent(
                symbols,
                agent,
                abi::AGENT_INFO_QUEUE_MAX_SIZE,
                (&mut queue_max_size as *mut u32).cast(),
            )?;
            query_agent(
                symbols,
                agent,
                abi::AGENT_INFO_QUEUE_TYPE,
                (&mut queue_type as *mut u32).cast(),
            )?;
            query_agent(
                symbols,
                agent,
                abi::AGENT_INFO_WORKGROUP_MAX_DIM,
                workgroup_max_dim.as_mut_ptr().cast(),
            )?;
            query_agent(
                symbols,
                agent,
                abi::AGENT_INFO_WORKGROUP_MAX_SIZE,
                (&mut workgroup_max_size as *mut u32).cast(),
            )?;
            query_agent(
                symbols,
                agent,
                abi::AGENT_INFO_GRID_MAX_DIM,
                grid_max_dim.as_mut_ptr().cast(),
            )?;
            query_agent(
                symbols,
                agent,
                abi::AGENT_INFO_GRID_MAX_SIZE,
                (&mut grid_max_size as *mut u32).cast(),
            )?;
            query_agent(
                symbols,
                agent,
                abi::AMD_AGENT_INFO_DOMAIN,
                (&mut pci_domain as *mut u32).cast(),
            )?;
            query_agent(
                symbols,
                agent,
                abi::AMD_AGENT_INFO_BDFID,
                (&mut pci_bdfid as *mut u32).cast(),
            )?;
        }
        let name_end = name
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(name.len());
        let name = String::from_utf8_lossy(&name[..name_end]).into_owned();
        let pci_bus_id = if device_type == abi::DEVICE_TYPE_GPU {
            Some(pci_bus_id_from_hsa_location(pci_domain, pci_bdfid))
        } else {
            None
        };
        Ok(AgentInfo {
            handle: agent,
            name,
            device_type,
            profile,
            rounding,
            queue_min_size,
            queue_max_size,
            queue_type,
            workgroup_max_dim,
            workgroup_max_size,
            grid_max_dim,
            grid_max_size,
            pci_bus_id,
        })
    }
}

impl Drop for RuntimeInner {
    fn drop(&mut self) {
        // SAFETY: this is the matching final shutdown for this initialized
        // RuntimeInner. Every RAII child retains this Arc and therefore drops
        // before it.
        let _ = unsafe { (self.symbols.shut_down)() };
    }
}

fn query_agent(
    symbols: &abi::Symbols,
    agent: abi::Agent,
    attribute: u32,
    output: *mut c_void,
) -> Result<(), RuntimeError> {
    // SAFETY: output points to the exact public-header type for `attribute`.
    let status = unsafe { (symbols.agent_get_info)(agent, attribute, output) };
    check_status(symbols, "hsa_agent_get_info", status)
}

#[derive(Clone, Debug)]
struct AgentInfo {
    handle: abi::Agent,
    name: String,
    device_type: u32,
    profile: u32,
    rounding: u32,
    queue_min_size: u32,
    queue_max_size: u32,
    queue_type: u32,
    workgroup_max_dim: [u16; 3],
    workgroup_max_size: u32,
    grid_max_dim: [u32; 3],
    grid_max_size: u32,
    pci_bus_id: Option<PciBusId>,
}

/// A normalized ROCm/HIP PCI domain/bus/device/function identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PciBusId {
    domain: u32,
    bus: u8,
    device: u8,
    function: u8,
}

impl PciBusId {
    pub fn domain(self) -> u32 {
        self.domain
    }

    pub fn bus(self) -> u8 {
        self.bus
    }

    pub fn device(self) -> u8 {
        self.device
    }

    pub fn function(self) -> u8 {
        self.function
    }
}

fn pci_bus_id_from_hsa_location(domain: u32, bdfid: u32) -> PciBusId {
    // ROCr LocationId may carry a partition ID in bits 31:28. HIP's PCI
    // identity normalization deliberately consumes only the conventional low
    // 16-bit bus/device/function fields, so mirror it exactly here.
    PciBusId {
        domain,
        bus: ((bdfid >> 8) & 0xff) as u8,
        device: ((bdfid >> 3) & 0x1f) as u8,
        function: (bdfid & 0x7) as u8,
    }
}

impl fmt::Display for PciBusId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PciBusIdParseError {
    input: String,
}

impl fmt::Display for PciBusIdParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid PCI bus ID {:?}; expected dddd:bb:dd.f",
            self.input
        )
    }
}

impl std::error::Error for PciBusIdParseError {}

impl std::str::FromStr for PciBusId {
    type Err = PciBusIdParseError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let invalid = || PciBusIdParseError {
            input: input.to_owned(),
        };
        let (domain_bus_device, function) = input.rsplit_once('.').ok_or_else(invalid)?;
        let mut components = domain_bus_device.split(':');
        let domain = components.next().ok_or_else(invalid)?;
        let bus = components.next().ok_or_else(invalid)?;
        let device = components.next().ok_or_else(invalid)?;
        if components.next().is_some()
            || domain.is_empty()
            || bus.is_empty()
            || device.is_empty()
            || function.is_empty()
        {
            return Err(invalid());
        }
        let domain = u32::from_str_radix(domain, 16).map_err(|_| invalid())?;
        let bus = u32::from_str_radix(bus, 16).map_err(|_| invalid())?;
        let device = u32::from_str_radix(device, 16).map_err(|_| invalid())?;
        let function = u32::from_str_radix(function, 16).map_err(|_| invalid())?;
        if bus > u32::from(u8::MAX) || device > 0x1f || function > 0x7 {
            return Err(invalid());
        }
        Ok(Self {
            domain,
            bus: bus as u8,
            device: device as u8,
            function: function as u8,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub enum GpuSelector<'a> {
    Ordinal(usize),
    NameContains(&'a str),
}

#[derive(Clone)]
pub struct GpuDevice {
    runtime: Arc<RuntimeInner>,
    gpu: AgentInfo,
    cpu: Option<AgentInfo>,
}

impl fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuDevice")
            .field("name", &self.gpu.name)
            .field("agent", &self.gpu.handle)
            .field("pci_bus_id", &self.gpu.pci_bus_id)
            .field("queue_min_size", &self.gpu.queue_min_size)
            .field("queue_max_size", &self.gpu.queue_max_size)
            .field("queue_type", &self.gpu.queue_type)
            .finish()
    }
}

impl GpuDevice {
    pub fn name(&self) -> &str {
        &self.gpu.name
    }

    pub fn agent_handle(&self) -> u64 {
        self.gpu.handle.0
    }

    pub fn pci_bus_id(&self) -> PciBusId {
        self.gpu
            .pci_bus_id
            .expect("GPU agent construction requires a ROCm/HIP PCI identity")
    }

    pub fn queue_size_range(&self) -> std::ops::RangeInclusive<u32> {
        self.gpu.queue_min_size..=self.gpu.queue_max_size
    }

    pub fn dispatch_time(
        &self,
        signal: &CompletionSignal,
    ) -> Result<abi::ProfilingDispatchTime, RuntimeError> {
        if !Arc::ptr_eq(&self.runtime, &signal.runtime) {
            return Err(RuntimeError::InvalidRuntimeObject(
                "profiling signal belongs to another HSA runtime",
            ));
        }
        let mut time = abi::ProfilingDispatchTime { start: 0, end: 0 };
        // SAFETY: the queue was created on this agent, the owned completion
        // signal is still live and has completed, and `time` is valid output.
        let status = unsafe {
            (self.runtime.symbols.profiling_get_dispatch_time)(
                self.gpu_agent(),
                signal.raw(),
                &mut time,
            )
        };
        check_status(
            &self.runtime.symbols,
            "hsa_amd_profiling_get_dispatch_time",
            status,
        )?;
        if time.end < time.start {
            return Err(RuntimeError::InvalidRuntimeObject(
                "HSA dispatch profiling timestamp runs backward",
            ));
        }
        Ok(time)
    }

    pub fn timestamp_frequency_hz(&self) -> Result<u64, RuntimeError> {
        Runtime {
            inner: self.runtime.clone(),
        }
        .timestamp_frequency_hz()
    }

    /// GPU clock frequency used by PM4 timestamp packets. This differs from
    /// the HSA system-clock frequency used by ROCr dispatch profiling.
    pub fn gpu_timestamp_frequency_hz(&self) -> Result<u64, RuntimeError> {
        let mut frequency = 0_u64;
        query_agent(
            &self.runtime.symbols,
            self.gpu_agent(),
            abi::AMD_AGENT_INFO_TIMESTAMP_FREQUENCY,
            (&mut frequency as *mut u64).cast(),
        )?;
        if frequency == 0 {
            return Err(RuntimeError::InvalidRuntimeObject(
                "GPU timestamp frequency is zero",
            ));
        }
        Ok(frequency)
    }

    fn gpu_agent(&self) -> abi::Agent {
        self.gpu.handle
    }

    pub fn raw_gpu_agent(&self) -> abi::Agent {
        self.gpu.handle
    }

    /// Return the available SDMA-engine bit mask for copies from `source` to
    /// this destination agent.
    pub fn copy_engine_mask(&self, source: &GpuDevice) -> Result<u32, RuntimeError> {
        self.require_same_runtime(source)?;
        let mut mask = 0_u32;
        // SAFETY: both agents belong to this live runtime and `mask` is valid
        // output for the public ROCr ABI.
        let status = unsafe {
            (self.runtime.symbols.memory_copy_engine_status)(
                self.gpu_agent(),
                source.gpu_agent(),
                &mut mask,
            )
        };
        check_status(
            &self.runtime.symbols,
            "hsa_amd_memory_copy_engine_status",
            status,
        )?;
        Ok(mask)
    }

    /// Return ROCr's preferred SDMA-engine mask for copies from `source`.
    pub fn preferred_copy_engine_mask(&self, source: &GpuDevice) -> Result<u32, RuntimeError> {
        self.require_same_runtime(source)?;
        let mut mask = 0_u32;
        // SAFETY: both agents belong to this live runtime and `mask` is valid
        // output for the public ROCr ABI.
        let status = unsafe {
            (self.runtime.symbols.memory_get_preferred_copy_engine)(
                self.gpu_agent(),
                source.gpu_agent(),
                &mut mask,
            )
        };
        check_status(
            &self.runtime.symbols,
            "hsa_amd_memory_get_preferred_copy_engine",
            status,
        )?;
        Ok(mask)
    }

    /// Submit one public-ROCr asynchronous peer copy.
    ///
    /// `engine` is one single-bit engine ID returned by [`Self::copy_engine_mask`].
    /// `None` leaves engine selection to ROCr. Dependencies and completion are
    /// persistent HSA signals; the caller must not reset or destroy them while
    /// the copy is in flight.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must each name at least `size` accessible bytes owned by
    /// their corresponding agents, and must remain live until completion.
    pub unsafe fn memory_async_copy(
        &self,
        dst: *mut c_void,
        source: &GpuDevice,
        src: *const c_void,
        size: usize,
        dependencies: &[&CompletionSignal],
        completion: &CompletionSignal,
        engine: Option<u32>,
    ) -> Result<(), RuntimeError> {
        self.require_same_runtime(source)?;
        if !Arc::ptr_eq(&self.runtime, &completion.runtime)
            || dependencies
                .iter()
                .any(|signal| !Arc::ptr_eq(&self.runtime, &signal.runtime))
        {
            return Err(RuntimeError::InvalidRuntimeObject(
                "async-copy signal belongs to another HSA runtime",
            ));
        }
        if dst.is_null() || src.is_null() || size == 0 {
            return Err(RuntimeError::InvalidRuntimeObject(
                "async-copy pointers must be non-null and size must be nonzero",
            ));
        }
        let dep_signals = dependencies
            .iter()
            .map(|signal| signal.raw())
            .collect::<Vec<_>>();
        let dep_count = u32::try_from(dep_signals.len()).map_err(|_| {
            RuntimeError::InvalidRuntimeObject("async-copy dependency count exceeds u32")
        })?;
        let dep_pointer = if dep_signals.is_empty() {
            ptr::null()
        } else {
            dep_signals.as_ptr()
        };
        // SAFETY: upheld by this method's contract; all handles are owned by
        // this runtime and the dependency array lives through the synchronous
        // submission call.
        let status = unsafe {
            match engine {
                Some(engine) => (self.runtime.symbols.memory_async_copy_on_engine)(
                    dst,
                    self.gpu_agent(),
                    src,
                    source.gpu_agent(),
                    size,
                    dep_count,
                    dep_pointer,
                    completion.raw(),
                    engine,
                    true,
                ),
                None => (self.runtime.symbols.memory_async_copy)(
                    dst,
                    self.gpu_agent(),
                    src,
                    source.gpu_agent(),
                    size,
                    dep_count,
                    dep_pointer,
                    completion.raw(),
                ),
            }
        };
        check_status(&self.runtime.symbols, "hsa_amd_memory_async_copy", status)
    }

    fn require_same_runtime(&self, other: &GpuDevice) -> Result<(), RuntimeError> {
        if Arc::ptr_eq(&self.runtime, &other.runtime) {
            Ok(())
        } else {
            Err(RuntimeError::InvalidRuntimeObject(
                "GPU agents belong to different HSA runtimes",
            ))
        }
    }

    pub fn validate_geometry(&self, geometry: LaunchGeometry) -> Result<(), RuntimeError> {
        for axis in 0..3 {
            if geometry.workgroup[axis] > self.gpu.workgroup_max_dim[axis] {
                return Err(RuntimeError::WorkgroupLimit {
                    axis,
                    requested: u32::from(geometry.workgroup[axis]),
                    maximum: u32::from(self.gpu.workgroup_max_dim[axis]),
                });
            }
            if geometry.grid_workitems[axis] > self.gpu.grid_max_dim[axis] {
                return Err(RuntimeError::GridLimit {
                    axis,
                    requested: u64::from(geometry.grid_workitems[axis]),
                    maximum: u64::from(self.gpu.grid_max_dim[axis]),
                });
            }
        }
        let workgroup_total = geometry
            .workgroup
            .iter()
            .map(|value| u64::from(*value))
            .product::<u64>();
        if workgroup_total > u64::from(self.gpu.workgroup_max_size) {
            return Err(RuntimeError::WorkgroupLimit {
                axis: 3,
                requested: u32::try_from(workgroup_total).unwrap_or(u32::MAX),
                maximum: self.gpu.workgroup_max_size,
            });
        }
        let grid_total = geometry
            .grid_workitems
            .iter()
            .try_fold(1_u64, |total, value| total.checked_mul(u64::from(*value)));
        let Some(grid_total) = grid_total else {
            return Err(RuntimeError::GridLimit {
                axis: 3,
                requested: u64::MAX,
                maximum: u64::from(self.gpu.grid_max_size),
            });
        };
        if grid_total > u64::from(self.gpu.grid_max_size) {
            return Err(RuntimeError::GridLimit {
                axis: 3,
                requested: grid_total,
                maximum: u64::from(self.gpu.grid_max_size),
            });
        }
        Ok(())
    }
}

/// One completion signal. It starts at one; AQL completion decrements it to
/// zero. A signal must not be reset while any packet still references it.
pub struct CompletionSignal {
    runtime: Arc<RuntimeInner>,
    raw: abi::Signal,
}

impl fmt::Debug for CompletionSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CompletionSignal")
            .field(&self.raw.0)
            .finish()
    }
}

impl CompletionSignal {
    pub fn new(device: &GpuDevice) -> Result<Self, RuntimeError> {
        let mut raw = abi::Signal(0);
        // A zero consumer count permits both GPU barrier packets and the host
        // wait path to consume the signal. Supplying only the GPU here would
        // make the CPU wait undefined under the public HSA contract.
        // SAFETY: null consumers is required when the count is zero; output is
        // valid.
        let status = unsafe { (device.runtime.symbols.signal_create)(1, 0, ptr::null(), &mut raw) };
        check_status(&device.runtime.symbols, "hsa_signal_create", status)?;
        if raw.0 == 0 {
            return Err(RuntimeError::InvalidRuntimeObject(
                "hsa_signal_create returned handle zero",
            ));
        }
        Ok(Self {
            runtime: device.runtime.clone(),
            raw,
        })
    }

    pub fn raw(&self) -> abi::Signal {
        self.raw
    }

    /// Reset a signal after its previous use has completed.
    pub fn reset(&mut self) {
        // SAFETY: `&mut self` prevents concurrent safe reset/wait through this
        // owner. This operation is crate-private so only the replay state
        // machine can reset after observing completion.
        unsafe { (self.runtime.symbols.signal_store_screlease)(self.raw, 1) };
    }

    pub fn is_complete(&self) -> bool {
        // SAFETY: the owned signal remains valid for this load.
        unsafe { (self.runtime.symbols.signal_load_scacquire)(self.raw) < 1 }
    }

    /// System-scope acquire load of the current signal value. Async-copy
    /// callers use this to distinguish success (`0`) from a negative ROCr
    /// asynchronous error indication.
    pub fn value_scacquire(&self) -> i64 {
        // SAFETY: the owned signal remains valid for this load.
        unsafe { (self.runtime.symbols.signal_load_scacquire)(self.raw) }
    }

    /// Wait for completion using the default finite host-side bound.
    pub fn wait(&self) -> Result<(), RuntimeError> {
        self.wait_timeout(DEFAULT_WAIT_TIMEOUT)
    }

    /// Poll for completion for at most `timeout`.
    pub fn wait_timeout(&self, timeout: Duration) -> Result<(), RuntimeError> {
        let started = Instant::now();
        let mut polls = 0_u32;
        loop {
            if self.is_complete() {
                return Ok(());
            }
            if started.elapsed() >= timeout {
                return Err(RuntimeError::SignalTimeout {
                    signal: self.raw.0,
                    timeout,
                });
            }
            bounded_poll_pause(&mut polls);
        }
    }
}

impl Drop for CompletionSignal {
    fn drop(&mut self) {
        // Graph teardown inactivates its queues before signals are dropped;
        // Drop never performs an unbounded signal wait.
        // SAFETY: this object uniquely owns the nonzero signal handle.
        let _ = unsafe { (self.runtime.symbols.signal_destroy)(self.raw) };
    }
}

fn bounded_poll_pause(polls: &mut u32) {
    *polls = polls.wrapping_add(1);
    if *polls & 0x3f == 0 {
        std::thread::yield_now();
    } else {
        std::hint::spin_loop();
    }
}

/// One non-synchronizing observation of an HSA queue's absolute indices.
///
/// This is diagnostic data only. Both indices are loaded with the public HSA
/// relaxed accessors, so a sample must not be used to establish packet or
/// resource visibility. Redline owns the sole producer while a replay ticket
/// is live, which keeps the write index stable after publication and makes the
/// wrapping difference a useful instantaneous backlog estimate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueueDepthSample {
    queue_id: u64,
    read_index: u64,
    write_index: u64,
    depth: u64,
}

impl QueueDepthSample {
    pub fn queue_id(self) -> u64 {
        self.queue_id
    }

    pub fn read_index(self) -> u64 {
        self.read_index
    }

    pub fn write_index(self) -> u64 {
        self.write_index
    }

    pub fn depth(self) -> u64 {
        self.depth
    }
}

/// Aggregate observations for one queue during one diagnostic completion wait.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueueDepthStats {
    queue_id: u64,
    poll_count: u64,
    first_sample: Option<QueueDepthSample>,
    last_sample: Option<QueueDepthSample>,
    minimum_depth: u64,
    maximum_depth: u64,
    depth_sum: u128,
    empty_poll_count: u64,
    depth_le_one_poll_count: u64,
}

impl QueueDepthStats {
    fn new(queue_id: u64) -> Self {
        Self {
            queue_id,
            poll_count: 0,
            first_sample: None,
            last_sample: None,
            minimum_depth: u64::MAX,
            maximum_depth: 0,
            depth_sum: 0,
            empty_poll_count: 0,
            depth_le_one_poll_count: 0,
        }
    }

    fn observe(&mut self, sample: QueueDepthSample) {
        debug_assert_eq!(self.queue_id, sample.queue_id);
        self.poll_count += 1;
        self.first_sample.get_or_insert(sample);
        self.last_sample = Some(sample);
        self.minimum_depth = self.minimum_depth.min(sample.depth);
        self.maximum_depth = self.maximum_depth.max(sample.depth);
        self.depth_sum += u128::from(sample.depth);
        self.empty_poll_count += u64::from(sample.depth == 0);
        self.depth_le_one_poll_count += u64::from(sample.depth <= 1);
    }

    pub fn queue_id(&self) -> u64 {
        self.queue_id
    }

    pub fn poll_count(&self) -> u64 {
        self.poll_count
    }

    pub fn first_sample(&self) -> Option<QueueDepthSample> {
        self.first_sample
    }

    pub fn last_sample(&self) -> Option<QueueDepthSample> {
        self.last_sample
    }

    pub fn minimum_depth(&self) -> Option<u64> {
        self.first_sample.map(|_| self.minimum_depth)
    }

    pub fn maximum_depth(&self) -> Option<u64> {
        self.first_sample.map(|_| self.maximum_depth)
    }

    pub fn mean_depth(&self) -> Option<f64> {
        (self.poll_count != 0).then(|| self.depth_sum as f64 / self.poll_count as f64)
    }

    pub fn empty_poll_count(&self) -> u64 {
        self.empty_poll_count
    }

    pub fn depth_le_one_poll_count(&self) -> u64 {
        self.depth_le_one_poll_count
    }
}

/// Queue-backlog telemetry collected by an explicitly diagnostic wait.
///
/// `backlog_with_idle_peer_poll_count` is the direct two-queue discriminator:
/// at least one queue had depth greater than one while at least one peer was
/// empty. The other joint counters distinguish both queues staying shallow
/// from both remaining busy. Poll counts are observations, not GPU timestamps.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueueDepthReport {
    poll_count: u64,
    queues: Vec<QueueDepthStats>,
    all_empty_poll_count: u64,
    all_depth_le_one_poll_count: u64,
    exactly_one_nonempty_poll_count: u64,
    backlog_with_idle_peer_poll_count: u64,
}

impl QueueDepthReport {
    pub fn new(queue_ids: impl IntoIterator<Item = u64>) -> Self {
        Self {
            poll_count: 0,
            queues: queue_ids.into_iter().map(QueueDepthStats::new).collect(),
            all_empty_poll_count: 0,
            all_depth_le_one_poll_count: 0,
            exactly_one_nonempty_poll_count: 0,
            backlog_with_idle_peer_poll_count: 0,
        }
    }

    fn observe(&mut self, samples: impl IntoIterator<Item = QueueDepthSample>) {
        let mut samples = samples.into_iter();
        let mut nonempty_queues = 0_usize;
        let mut any_empty = false;
        let mut any_backlogged = false;
        let mut all_depth_le_one = true;
        for stats in &mut self.queues {
            let sample = samples
                .next()
                .expect("one queue-depth sample is required per queue");
            assert_eq!(
                stats.queue_id, sample.queue_id,
                "queue sample order changed"
            );
            stats.observe(sample);
            nonempty_queues += usize::from(sample.depth != 0);
            any_empty |= sample.depth == 0;
            any_backlogged |= sample.depth > 1;
            all_depth_le_one &= sample.depth <= 1;
        }
        assert!(samples.next().is_none(), "too many queue-depth samples");
        self.poll_count += 1;
        self.all_empty_poll_count += u64::from(nonempty_queues == 0);
        self.all_depth_le_one_poll_count += u64::from(all_depth_le_one);
        self.exactly_one_nonempty_poll_count += u64::from(nonempty_queues == 1);
        self.backlog_with_idle_peer_poll_count += u64::from(any_backlogged && any_empty);
    }

    pub fn poll_count(&self) -> u64 {
        self.poll_count
    }

    pub fn queues(&self) -> &[QueueDepthStats] {
        &self.queues
    }

    pub fn all_empty_poll_count(&self) -> u64 {
        self.all_empty_poll_count
    }

    pub fn all_depth_le_one_poll_count(&self) -> u64 {
        self.all_depth_le_one_poll_count
    }

    pub fn exactly_one_nonempty_poll_count(&self) -> u64 {
        self.exactly_one_nonempty_poll_count
    }

    pub fn backlog_with_idle_peer_poll_count(&self) -> u64 {
        self.backlog_with_idle_peer_poll_count
    }
}

/// One distinct ROCr user-mode queue.
pub struct AqlQueue {
    runtime: Option<Arc<RuntimeInner>>,
    raw: NonNull<abi::Queue>,
    fault: Option<Box<QueueFaultState>>,
    active: bool,
}

impl fmt::Debug for AqlQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AqlQueue")
            .field("id", &self.id())
            .field("size", &self.size())
            .finish()
    }
}

impl AqlQueue {
    pub fn create(device: &GpuDevice, requested_size: u32) -> Result<Self, RuntimeError> {
        if device.gpu.queue_type != abi::QUEUE_TYPE_MULTI {
            return Err(RuntimeError::UnsupportedQueueType(device.gpu.queue_type));
        }
        if !requested_size.is_power_of_two()
            || requested_size < device.gpu.queue_min_size
            || requested_size > device.gpu.queue_max_size
        {
            return Err(RuntimeError::InvalidQueueSize {
                requested: requested_size,
                min: device.gpu.queue_min_size,
                max: device.gpu.queue_max_size,
            });
        }
        let mut raw = ptr::null_mut();
        let fault = Box::<QueueFaultState>::default();
        let fault_context = (&*fault as *const QueueFaultState).cast_mut().cast();
        // A MULTI queue plus exclusive `&mut AqlQueue` publication is robust to
        // future producer sharing while keeping packet publication ordered.
        // SAFETY: agent and output pointer are valid. `fault_context` points at
        // stable boxed storage retained through queue destruction. UINT32_MAX
        // requests runtime-selected segment sizing.
        let status = unsafe {
            (device.runtime.symbols.queue_create)(
                device.gpu_agent(),
                requested_size,
                abi::QUEUE_TYPE_MULTI,
                Some(queue_error_callback),
                fault_context,
                u32::MAX,
                u32::MAX,
                &mut raw,
            )
        };
        check_status(&device.runtime.symbols, "hsa_queue_create", status)?;
        let raw = NonNull::new(raw).ok_or(RuntimeError::InvalidRuntimeObject(
            "hsa_queue_create returned null",
        ))?;
        let queue = Self {
            runtime: Some(device.runtime.clone()),
            raw,
            fault: Some(fault),
            active: true,
        };
        let descriptor = queue.descriptor();
        if descriptor.base_address.is_null()
            || descriptor.queue_type != abi::QUEUE_TYPE_MULTI
            || descriptor.size != requested_size
            || descriptor.features & abi::QUEUE_FEATURE_KERNEL_DISPATCH == 0
        {
            return Err(RuntimeError::InvalidRuntimeObject(
                "HSA queue descriptor has the wrong type/size or lacks kernel dispatch",
            ));
        }
        Ok(queue)
    }

    pub fn id(&self) -> u64 {
        self.descriptor().id
    }

    pub fn size(&self) -> u32 {
        self.descriptor().size
    }

    fn set_profiling(&self, enabled: bool) -> Result<(), RuntimeError> {
        // SAFETY: the queue descriptor is owned and live for this call.
        let status = unsafe {
            (self.runtime().symbols.profiling_set_profiler_enabled)(
                self.raw.as_ptr(),
                i32::from(enabled),
            )
        };
        check_status(
            &self.runtime().symbols,
            "hsa_amd_profiling_set_profiler_enabled",
            status,
        )
    }

    fn set_cu_mask(&self, mask_bits: u32, mask: &[u32]) -> Result<(), RuntimeError> {
        if mask_bits == 0 || !mask_bits.is_multiple_of(32) {
            return Err(RuntimeError::InvalidCuMask(
                "mask bit count must be a non-zero multiple of 32",
            ));
        }
        let required_words = usize::try_from(mask_bits / 32)
            .map_err(|_| RuntimeError::InvalidCuMask("mask bit count exceeds usize"))?;
        if mask.len() < required_words {
            return Err(RuntimeError::InvalidCuMask(
                "mask storage is shorter than its declared bit count",
            ));
        }
        // SAFETY: the queue is live, the bit count is ABI-valid, and `mask`
        // supplies at least the declared number of words for this call.
        let status = unsafe {
            (self.runtime().symbols.queue_cu_set_mask)(self.raw.as_ptr(), mask_bits, mask.as_ptr())
        };
        check_status(&self.runtime().symbols, "hsa_amd_queue_cu_set_mask", status)
    }

    fn depth_sample(&self) -> QueueDepthSample {
        let queue = self.raw.as_ptr();
        // These loads deliberately mirror the requested diagnostic: write
        // index minus read index, both relaxed. They do not participate in the
        // correctness path, which retains its acquire read in
        // `wait_for_capacity`.
        // SAFETY: the queue descriptor remains live for both index loads.
        let write_index = unsafe { (self.runtime().symbols.queue_load_write_index_relaxed)(queue) };
        // SAFETY: the queue descriptor remains live for this index load.
        let read_index = unsafe { (self.runtime().symbols.queue_load_read_index_relaxed)(queue) };
        QueueDepthSample {
            queue_id: self.id(),
            read_index,
            write_index,
            depth: write_index.wrapping_sub(read_index),
        }
    }

    /// Reserve, fill, and release-publish one contiguous recorded batch without
    /// ringing its doorbell yet.
    fn prepare_batch(
        &mut self,
        packets: &[PacketImage],
        timeout: Duration,
    ) -> Result<Option<abi::SignalValue>, RuntimeError> {
        assert!(
            packets.len() <= self.size() as usize,
            "recorded batch must fit in one queue generation"
        );
        self.ensure_active()?;
        self.check_fault()?;
        if packets.is_empty() {
            return Ok(None);
        }
        self.wait_for_capacity(packets.len() as u64, timeout)?;
        // SAFETY: packet construction validates every field and this mutable
        // borrow serializes reserve/fill/publish across the whole batch. The
        // capacity check runs before reservation so timeout never leaves an
        // unpublished hole in the ring.
        Ok(Some(unsafe { self.publish_batch(packets) }))
    }

    fn descriptor(&self) -> &abi::Queue {
        // SAFETY: the runtime owns the descriptor until queue_destroy, which
        // only runs from this object's Drop.
        unsafe { self.raw.as_ref() }
    }

    fn runtime(&self) -> &RuntimeInner {
        self.runtime
            .as_deref()
            .expect("queue runtime is retained until Drop finishes")
    }

    fn fault_state(&self) -> &QueueFaultState {
        self.fault
            .as_deref()
            .expect("queue callback state is retained until Drop finishes")
    }

    fn ensure_active(&self) -> Result<(), RuntimeError> {
        if self.active {
            Ok(())
        } else {
            Err(RuntimeError::QueueInactive {
                queue_id: self.id(),
            })
        }
    }

    fn fault_error(&self) -> Option<RuntimeError> {
        self.fault_state()
            .status()
            .map(|status| RuntimeError::QueueFault {
                queue_id: self.id(),
                status,
                message: status_message(&self.runtime().symbols, status),
            })
    }

    fn check_fault(&self) -> Result<(), RuntimeError> {
        match self.fault_error() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn wait_for_capacity(&self, count: u64, timeout: Duration) -> Result<(), RuntimeError> {
        let queue = self.raw.as_ptr();
        let started = Instant::now();
        let mut polls = 0_u32;
        loop {
            self.check_fault()?;
            // No write-index reservation occurs until the entire batch fits.
            // SAFETY: the queue descriptor remains live for both index loads.
            let write = unsafe { (self.runtime().symbols.queue_load_write_index_relaxed)(queue) };
            // Acquire pairs with packet-processor slot retirement.
            // SAFETY: the queue descriptor remains live for this index load.
            let read = unsafe { (self.runtime().symbols.queue_load_read_index_scacquire)(queue) };
            let last_packet_id = write.wrapping_add(count - 1);
            if last_packet_id.wrapping_sub(read) < u64::from(self.size()) {
                return Ok(());
            }
            if started.elapsed() >= timeout {
                return Err(RuntimeError::QueueCapacityTimeout {
                    queue_id: self.id(),
                    packets: count as usize,
                    timeout,
                });
            }
            bounded_poll_pause(&mut polls);
        }
    }

    /// Copy prepared packet tails and publish every header/setup word with
    /// release ordering. The returned final packet ID is rung later, after all
    /// queues in the replay have been prepared.
    unsafe fn publish_batch(&mut self, packets: &[PacketImage]) -> abi::SignalValue {
        let queue = self.raw.as_ptr();
        let count = packets.len() as u64;
        // SAFETY: queue is live. One atomic range reservation is the public HSA
        // multiproducer mechanism and this owner publishes the range in order.
        let first_packet_id =
            unsafe { (self.runtime().symbols.queue_add_write_index_relaxed)(queue, count) };
        let last_packet_id = first_packet_id.wrapping_add(count - 1);

        for (offset, packet) in packets.iter().enumerate() {
            let packet_id = first_packet_id.wrapping_add(offset as u64);
            let slot_index = packet_id & u64::from(self.descriptor().size - 1);
            // SAFETY: queue base has `size` 64-byte slots and the mask bounds
            // the selected slot. HSA guarantees packet-size alignment.
            let slot = unsafe {
                self.descriptor()
                    .base_address
                    .cast::<u8>()
                    .add(slot_index as usize * AQL_PACKET_BYTES)
            };
            // The first word remains INVALID until every payload byte is
            // visible. SAFETY: source and queue slot are non-overlapping valid
            // regions. Only bytes 4..64 are copied before publication.
            unsafe {
                ptr::copy_nonoverlapping(
                    packet.bytes.as_ptr().add(4),
                    slot.add(4),
                    AQL_PACKET_BYTES - 4,
                );
            }
            // ROCr and LLVM's AMDGPU plugin publish header+setup as one release
            // store. The queue slot is at least 64-byte aligned.
            // SAFETY: slot begins with a properly aligned u32 header word.
            unsafe { &*slot.cast::<AtomicU32>() }.store(packet.header_word, Ordering::Release);
        }
        last_packet_id as abi::SignalValue
    }

    fn ring(&self, final_packet_id: abi::SignalValue) {
        // The doorbell value is the final absolute packet ID, not a ring index.
        // One write makes the complete contiguous batch visible to the CP.
        // A relaxed signal store is sufficient after the release header store,
        // matching ROCr consumers and LLVM's public HSA queue implementation.
        // SAFETY: the runtime owns this queue's doorbell signal.
        unsafe {
            (self.runtime().symbols.signal_store_relaxed)(
                self.descriptor().doorbell_signal,
                final_packet_id,
            )
        };
    }

    fn inactivate(&mut self) -> Result<(), RuntimeError> {
        if !self.active {
            return Ok(());
        }
        // HSA specifies that inactivation aborts pending executions and causes
        // later packets to be ignored. No completion signal is awaited on a
        // fault/cancellation path, but this foreign call itself has no public
        // timeout and therefore is not a wall-clock bound.
        // SAFETY: this object uniquely owns a live queue descriptor.
        let status = unsafe { (self.runtime().symbols.queue_inactivate)(self.raw.as_ptr()) };
        check_status(&self.runtime().symbols, "hsa_queue_inactivate", status)?;
        self.active = false;
        Ok(())
    }
}

impl Drop for AqlQueue {
    fn drop(&mut self) {
        // Never wait for a completion signal from Drop. Inactivation is the HSA
        // operation that aborts pending work before owned packet pointees and
        // callback state are released. The foreign calls themselves have no
        // public timeout.
        let _ = self.inactivate();
        // SAFETY: this object uniquely owns the queue returned by queue_create.
        let status = unsafe { (self.runtime().symbols.queue_destroy)(self.raw.as_ptr()) };
        if queue_destroy_requires_leak(status) {
            // ROCr may still invoke the registered callback or touch runtime
            // state after refusing destruction. Leaking both owners is the only
            // safe non-fail-stop choice available from Drop.
            if let Some(fault) = self.fault.take() {
                std::mem::forget(fault);
            }
            if let Some(runtime) = self.runtime.take() {
                std::mem::forget(runtime);
            }
        }
    }
}

fn queue_destroy_requires_leak(status: abi::Status) -> bool {
    status != abi::STATUS_SUCCESS
}

pub struct QueueSet {
    queues: Vec<AqlQueue>,
    prepared_doorbells: Vec<Option<abi::SignalValue>>,
}

impl QueueSet {
    pub fn create(
        device: &GpuDevice,
        queue_count: usize,
        queue_size: u32,
    ) -> Result<Self, RuntimeError> {
        if queue_count == 0 {
            return Err(RuntimeError::ZeroQueues);
        }
        let mut queues = Vec::with_capacity(queue_count);
        for _ in 0..queue_count {
            queues.push(AqlQueue::create(device, queue_size)?);
        }
        let mut ids = queues.iter().map(AqlQueue::id).collect::<Vec<_>>();
        ids.sort_unstable();
        if ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(RuntimeError::DuplicateQueueId);
        }
        Ok(Self {
            prepared_doorbells: vec![None; queues.len()],
            queues,
        })
    }

    pub fn len(&self) -> usize {
        self.queues.len()
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn queue_ids(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.queues.iter().map(AqlQueue::id)
    }

    pub fn set_profiling(&self, enabled: bool) -> Result<(), RuntimeError> {
        for queue in &self.queues {
            queue.set_profiling(enabled)?;
        }
        Ok(())
    }

    pub fn set_cu_mask(&self, mask_bits: u32, mask: &[u32]) -> Result<(), RuntimeError> {
        for queue in &self.queues {
            queue.set_cu_mask(mask_bits, mask)?;
        }
        Ok(())
    }

    pub fn size(&self, lane: usize) -> Option<u32> {
        self.queues.get(lane).map(AqlQueue::size)
    }

    /// Prepare every queue without ringing any queue. This avoids a CPU packet
    /// fill/reservation skew between the two measured overlap lanes.
    pub fn prepare_batches(&mut self, batches: &[Vec<PacketImage>]) -> Result<(), RuntimeError> {
        assert_eq!(batches.len(), self.queues.len());
        assert!(self.prepared_doorbells.iter().all(Option::is_none));
        if let Err(error) = self.check_faults() {
            return Err(self.inactivate_after_error(error));
        }
        for (lane, batch) in batches.iter().enumerate() {
            let prepared = self.queues[lane].prepare_batch(batch, DEFAULT_WAIT_TIMEOUT);
            match prepared {
                Ok(doorbell) => self.prepared_doorbells[lane] = doorbell,
                Err(error) => {
                    self.prepared_doorbells.fill(None);
                    return Err(self.inactivate_after_error(error));
                }
            }
        }
        Ok(())
    }

    /// Ring each nonempty queue exactly once after all queue packet images have
    /// been release-published.
    pub fn ring_prepared(&mut self) -> Result<(), RuntimeError> {
        if let Err(error) = self.check_faults() {
            self.prepared_doorbells.fill(None);
            return Err(self.inactivate_after_error(error));
        }
        for (queue, doorbell) in self.queues.iter().zip(&mut self.prepared_doorbells) {
            if let Some(final_packet_id) = doorbell.take() {
                queue.ring(final_packet_id);
            }
        }
        Ok(())
    }

    pub fn wait_signal(
        &self,
        signal: &CompletionSignal,
        timeout: Duration,
    ) -> Result<(), RuntimeError> {
        let started = Instant::now();
        let mut polls = 0_u32;
        loop {
            self.check_faults()?;
            if signal.is_complete() {
                return Ok(());
            }
            if started.elapsed() >= timeout {
                return Err(RuntimeError::SignalTimeout {
                    signal: signal.raw().0,
                    timeout,
                });
            }
            bounded_poll_pause(&mut polls);
        }
    }

    /// Wait until every completion signal reaches zero under one shared
    /// timeout. Polling each signal with a fresh timeout could otherwise
    /// multiply the finite failure bound by the queue count.
    pub fn wait_signals(
        &self,
        signals: &[CompletionSignal],
        timeout: Duration,
    ) -> Result<(), RuntimeError> {
        let started = Instant::now();
        let mut polls = 0_u32;
        loop {
            self.check_faults()?;
            if signals.iter().all(CompletionSignal::is_complete) {
                return Ok(());
            }
            if started.elapsed() >= timeout {
                let signal = signals
                    .iter()
                    .find(|signal| !signal.is_complete())
                    .map_or(0, |signal| signal.raw().0);
                return Err(RuntimeError::SignalTimeout { signal, timeout });
            }
            bounded_poll_pause(&mut polls);
        }
    }

    /// Poll queue indices alongside the completion signal.
    ///
    /// This method is intentionally separate from `wait_signal`, so ordinary
    /// replay contains neither index loads nor a diagnostic branch.
    pub fn wait_signal_with_queue_depth(
        &self,
        signal: &CompletionSignal,
        timeout: Duration,
    ) -> Result<QueueDepthReport, RuntimeError> {
        let started = Instant::now();
        let mut polls = 0_u32;
        let mut report = QueueDepthReport::new(self.queue_ids());
        loop {
            self.check_faults()?;
            report.observe(self.queues.iter().map(AqlQueue::depth_sample));

            if signal.is_complete() {
                return Ok(report);
            }
            if started.elapsed() >= timeout {
                return Err(RuntimeError::SignalTimeout {
                    signal: signal.raw().0,
                    timeout,
                });
            }
            bounded_poll_pause(&mut polls);
        }
    }

    pub fn check_faults(&self) -> Result<(), RuntimeError> {
        for queue in &self.queues {
            queue.check_fault()?;
        }
        Ok(())
    }

    /// Abort pending work on every queue. All lanes are attempted even if an
    /// earlier inactivation reports an HSA error.
    pub fn inactivate_all(&mut self) -> Result<(), RuntimeError> {
        self.prepared_doorbells.fill(None);
        let mut first_error = None;
        for queue in &mut self.queues {
            if let Err(error) = queue.inactivate()
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn inactivate_after_error(&mut self, operation: RuntimeError) -> RuntimeError {
        match self.inactivate_all() {
            Ok(()) => operation,
            Err(teardown) => RuntimeError::OperationAndTeardown {
                operation: Box::new(operation),
                teardown: Box::new(teardown),
            },
        }
    }
}

#[derive(Clone)]
pub struct KernargPool {
    inner: Arc<KernargPoolInner>,
}

struct KernargPoolInner {
    runtime: Arc<RuntimeInner>,
    pool: abi::MemoryPool,
    gpu: abi::Agent,
    cpu: Option<abi::Agent>,
    granule: usize,
    alignment: usize,
    /// True for GPU-agent (VRAM) pools: buffer writes must go through
    /// `hsa_amd_memory_copy`, not host pointer stores.
    device_local: bool,
}

impl KernargPool {
    pub fn discover(device: &GpuDevice) -> Result<Self, RuntimeError> {
        let cpu = device.cpu.as_ref().ok_or(RuntimeError::NoCpuAgent)?;
        unsafe extern "C" fn collect(pool: abi::MemoryPool, data: *mut c_void) -> abi::Status {
            // SAFETY: context is a live vector for synchronous iteration.
            unsafe { &mut *data.cast::<Vec<abi::MemoryPool>>() }.push(pool);
            abi::STATUS_SUCCESS
        }
        let mut pools = Vec::new();
        // Fine-grained kernarg pools are associated with the host agent.
        // SAFETY: callback and context satisfy the iteration contract.
        let status = unsafe {
            (device.runtime.symbols.agent_iterate_memory_pools)(
                cpu.handle,
                Some(collect),
                (&mut pools as *mut Vec<abi::MemoryPool>).cast(),
            )
        };
        check_status(
            &device.runtime.symbols,
            "hsa_amd_agent_iterate_memory_pools",
            status,
        )?;

        for pool in pools {
            let mut segment = 0_u32;
            let mut flags = 0_u32;
            let mut allowed = false;
            query_pool(
                &device.runtime.symbols,
                pool,
                abi::AMD_MEMORY_POOL_INFO_SEGMENT,
                (&mut segment as *mut u32).cast(),
            )?;
            if segment != abi::AMD_SEGMENT_GLOBAL {
                continue;
            }
            query_pool(
                &device.runtime.symbols,
                pool,
                abi::AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                (&mut flags as *mut u32).cast(),
            )?;
            query_pool(
                &device.runtime.symbols,
                pool,
                abi::AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                (&mut allowed as *mut bool).cast(),
            )?;
            let required = abi::AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
                | abi::AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
            if !allowed || flags & required != required {
                continue;
            }
            let mut granule = 0_usize;
            let mut alignment = 0_usize;
            query_pool(
                &device.runtime.symbols,
                pool,
                abi::AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                (&mut granule as *mut usize).cast(),
            )?;
            query_pool(
                &device.runtime.symbols,
                pool,
                abi::AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT,
                (&mut alignment as *mut usize).cast(),
            )?;
            if granule == 0 || !alignment.is_power_of_two() {
                continue;
            }
            // The retained PM4 indirect buffer is allocated from THIS pool
            // (SingleQueuePm4Ib::indirect is a KernargBuffer), so the command
            // processor re-fetches the whole IB from host-agent memory on every
            // replay. Ordinary HIP dispatch does not do that, which makes the
            // retained path far more sensitive to host-memory and PCIe
            // behaviour than the HIP arm — and that behaviour is an amdgpu
            // decision. Set HIPFIRE_REDLINE_POOL_DEBUG=1 to report exactly
            // which pool a machine selected, so two boxes reporting different
            // retained throughput can be compared on placement rather than
            // guessed at.
            if hipfire_config::developer_var_os("HIPFIRE_REDLINE_POOL_DEBUG")
                .is_some_and(|v| v != "0" && !v.is_empty())
            {
                let mut size = 0_usize;
                let _ = query_pool(
                    &device.runtime.symbols,
                    pool,
                    abi::AMD_MEMORY_POOL_INFO_SIZE,
                    (&mut size as *mut usize).cast(),
                );
                let kind = if flags & abi::AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED != 0 {
                    "fine-grained"
                } else if flags & abi::AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED != 0 {
                    "coarse-grained"
                } else {
                    "unflagged"
                };
                eprintln!(
                    "[redline] retained-IB pool: handle=0x{:x} owner=CPU-agent segment=global \
                     kind={kind} kernarg_init={} flags=0x{flags:08x} size={size} \
                     granule={granule} alignment={alignment}",
                    pool.0,
                    flags & abi::AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT != 0,
                );
                eprintln!(
                    "[redline] the retained indirect buffer lives in this pool, so the command \
                     processor fetches it over the host interface on every replay"
                );
            }

            return Ok(Self {
                inner: Arc::new(KernargPoolInner {
                    runtime: device.runtime.clone(),
                    pool,
                    gpu: device.gpu_agent(),
                    cpu: device.cpu.as_ref().map(|c| c.handle),
                    granule,
                    alignment,
                    device_local: false,
                }),
            });
        }
        Err(RuntimeError::NoKernargPool)
    }

    /// Discover a GPU-agent (VRAM) pool for the retained indirect buffer.
    ///
    /// The default [`KernargPool::discover`] pool is CPU-agent global — the
    /// command processor re-fetches the retained PM4 IB over the host
    /// interface on EVERY replay. A device-local pool lets the CP fetch the
    /// tape from VRAM instead. Requires a GPU-agent GLOBAL pool with runtime
    /// allocation allowed; coarse-grained preferred (true VRAM), fine-grained
    /// accepted (UMA, e.g. Strix Halo). Buffers from this pool are written via
    /// `hsa_amd_memory_copy` — host pointer stores are invalid on dGPU VRAM.
    ///
    /// Only the IB belongs here: timestamps/semaphores read by the host stay
    /// on the CPU-agent pool.
    pub fn discover_device_local(device: &GpuDevice) -> Result<Self, RuntimeError> {
        unsafe extern "C" fn collect(pool: abi::MemoryPool, data: *mut c_void) -> abi::Status {
            // SAFETY: context is a live vector for synchronous iteration.
            unsafe { &mut *data.cast::<Vec<abi::MemoryPool>>() }.push(pool);
            abi::STATUS_SUCCESS
        }
        let mut pools = Vec::new();
        // SAFETY: callback and context satisfy the iteration contract.
        let status = unsafe {
            (device.runtime.symbols.agent_iterate_memory_pools)(
                device.gpu_agent(),
                Some(collect),
                (&mut pools as *mut Vec<abi::MemoryPool>).cast(),
            )
        };
        check_status(
            &device.runtime.symbols,
            "hsa_amd_agent_iterate_memory_pools",
            status,
        )?;

        let mut fine_grained_fallback: Option<(abi::MemoryPool, usize, usize)> = None;
        for pool in pools {
            let mut segment = 0_u32;
            let mut flags = 0_u32;
            let mut allowed = false;
            query_pool(
                &device.runtime.symbols,
                pool,
                abi::AMD_MEMORY_POOL_INFO_SEGMENT,
                (&mut segment as *mut u32).cast(),
            )?;
            if segment != abi::AMD_SEGMENT_GLOBAL {
                continue;
            }
            query_pool(
                &device.runtime.symbols,
                pool,
                abi::AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                (&mut flags as *mut u32).cast(),
            )?;
            query_pool(
                &device.runtime.symbols,
                pool,
                abi::AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                (&mut allowed as *mut bool).cast(),
            )?;
            if !allowed {
                continue;
            }
            let mut granule = 0_usize;
            let mut alignment = 0_usize;
            query_pool(
                &device.runtime.symbols,
                pool,
                abi::AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                (&mut granule as *mut usize).cast(),
            )?;
            query_pool(
                &device.runtime.symbols,
                pool,
                abi::AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT,
                (&mut alignment as *mut usize).cast(),
            )?;
            if granule == 0 || !alignment.is_power_of_two() {
                continue;
            }
            if flags & abi::AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED != 0 {
                if hipfire_config::developer_var_os("HIPFIRE_REDLINE_POOL_DEBUG")
                    .is_some_and(|v| v != "0" && !v.is_empty())
                {
                    eprintln!(
                        "[redline] retained-IB pool: handle=0x{:x} owner=GPU-agent \
                         segment=global kind=coarse-grained (device-local) granule={granule} \
                         alignment={alignment}",
                        pool.0,
                    );
                }
                return Ok(Self {
                    inner: Arc::new(KernargPoolInner {
                        runtime: device.runtime.clone(),
                        pool,
                        gpu: device.gpu_agent(),
                        cpu: device.cpu.as_ref().map(|c| c.handle),
                        granule,
                        alignment,
                        device_local: true,
                    }),
                });
            }
            if flags & abi::AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED != 0
                && fine_grained_fallback.is_none()
            {
                fine_grained_fallback = Some((pool, granule, alignment));
            }
        }
        // UMA (Strix Halo, APU): GPU-agent fine-grained is still device-local
        // memory by construction — VRAM and host DRAM are one.
        if let Some((pool, granule, alignment)) = fine_grained_fallback {
            return Ok(Self {
                inner: Arc::new(KernargPoolInner {
                    runtime: device.runtime.clone(),
                    pool,
                    gpu: device.gpu_agent(),
                    cpu: device.cpu.as_ref().map(|c| c.handle),
                    granule,
                    alignment,
                    device_local: true,
                }),
            });
        }
        Err(RuntimeError::NoKernargPool)
    }

    pub fn allocate_for(&self, metadata: KernelMetadata) -> Result<KernargBuffer, RuntimeError> {
        let length = metadata.kernarg_segment_size as usize;
        let required_alignment = (metadata.kernarg_segment_alignment as usize).max(16);
        self.allocate_bytes(
            length,
            required_alignment,
            abi::AMD_MEMORY_POOL_STANDARD_FLAG,
        )
    }

    /// Allocate CPU-writable, GPU-accessible command memory. The executable
    /// flag is required for MEC indirect-buffer fetches even though the PM4
    /// words are data rather than shader ISA.
    ///
    /// This is host-resident fine-grained memory (see `discover`). That is the
    /// correct trade and was measured, not assumed: placing the retained IB in
    /// a device-local coarse-grained pool instead is SLOWER on both reference
    /// boards — hiptrx 565.52 -> 560.84 tok/s (-0.83%), k9lin 548.02 -> 544.28
    /// (-0.68%), qwen3.5-0.8b.mq4, two alternating reps each. Retained replay
    /// patches kernargs into this buffer from the CPU between replays; host
    /// memory is cached and fast to write, whereas VRAM through the BAR is
    /// write-combined and uncached, so the CPU-write cost outweighs any saving
    /// on the command processor's read. Do not "optimise" this into VRAM.
    pub fn allocate_executable_bytes(&self, length: usize) -> Result<KernargBuffer, RuntimeError> {
        self.allocate_bytes(length, 16, abi::AMD_MEMORY_POOL_EXECUTABLE_FLAG)
    }

    fn allocate_bytes(
        &self,
        length: usize,
        required_alignment: usize,
        flags: u32,
    ) -> Result<KernargBuffer, RuntimeError> {
        if !required_alignment.is_power_of_two() {
            return Err(RuntimeError::InvalidKernargAlignment(required_alignment));
        }
        if length == 0 {
            return Ok(KernargBuffer {
                pool: self.inner.clone(),
                device_local: self.inner.device_local,
                pointer: None,
                length: 0,
                allocation_size: 0,
            });
        }
        let allocation_size = length
            .checked_add(self.inner.granule - 1)
            .map(|value| value / self.inner.granule * self.inner.granule)
            .ok_or(RuntimeError::KernargSizeOverflow(length))?;
        let mut pointer = ptr::null_mut();
        // SAFETY: selected pool allows runtime allocations and output is valid.
        let status = unsafe {
            (self.inner.runtime.symbols.memory_pool_allocate)(
                self.inner.pool,
                allocation_size,
                flags,
                &mut pointer,
            )
        };
        check_status(
            &self.inner.runtime.symbols,
            "hsa_amd_memory_pool_allocate",
            status,
        )?;
        let pointer = match NonNull::new(pointer.cast::<u8>()) {
            Some(pointer) => pointer,
            None => {
                return Err(RuntimeError::InvalidRuntimeObject(
                    "memory-pool allocation returned null",
                ));
            }
        };
        if pointer.as_ptr() as usize % required_alignment != 0
            || pointer.as_ptr() as usize % self.inner.alignment != 0
        {
            // SAFETY: pointer came from the matching allocation function.
            let _ =
                unsafe { (self.inner.runtime.symbols.memory_pool_free)(pointer.as_ptr().cast()) };
            return Err(RuntimeError::KernargAlignmentNotMet {
                required: required_alignment.max(self.inner.alignment),
                address: pointer.as_ptr() as usize,
            });
        }
        // SAFETY: the allocation came from this pool and the GPU agent is a
        // valid access target; flags are reserved and therefore null.
        let status = unsafe {
            (self.inner.runtime.symbols.agents_allow_access)(
                1,
                &self.inner.gpu,
                ptr::null(),
                pointer.as_ptr().cast(),
            )
        };
        if let Err(error) = check_status(
            &self.inner.runtime.symbols,
            "hsa_amd_agents_allow_access",
            status,
        ) {
            // SAFETY: pointer came from the matching allocation function.
            let _ =
                unsafe { (self.inner.runtime.symbols.memory_pool_free)(pointer.as_ptr().cast()) };
            return Err(error);
        }
        // Zero the entire allocation before callers populate the ABI payload,
        // including any trailing padding expected by kernel metadata.
        // Device-local (coarse VRAM pool) allocations are host-mapped on
        // ReBAR systems (verified by the device_local_pool_host_memcpy_probe
        // test), so the same store path is valid.
        // SAFETY: pointer owns allocation_size writable host bytes.
        unsafe { ptr::write_bytes(pointer.as_ptr(), 0, allocation_size) };
        Ok(KernargBuffer {
            pool: self.inner.clone(),
            device_local: self.inner.device_local,
            pointer: Some(pointer),
            length,
            allocation_size,
        })
    }
}

fn query_pool(
    symbols: &abi::Symbols,
    pool: abi::MemoryPool,
    attribute: u32,
    output: *mut c_void,
) -> Result<(), RuntimeError> {
    // SAFETY: output points at the public-header type for this attribute.
    let status = unsafe { (symbols.memory_pool_get_info)(pool, attribute, output) };
    check_status(symbols, "hsa_amd_memory_pool_get_info", status)
}

/// Fine-grained, GPU-accessible kernarg storage.
///
/// The buffer may be modified only before its dispatch is submitted or after
/// that dispatch's completion signal has fired. A recorded graph owns buffers
/// across every replay and never exposes mutable access while work is in flight.
pub struct KernargBuffer {
    pool: Arc<KernargPoolInner>,
    device_local: bool,
    pointer: Option<NonNull<u8>>,
    length: usize,
    allocation_size: usize,
}

impl fmt::Debug for KernargBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernargBuffer")
            .field("length", &self.length)
            .field("allocation_size", &self.allocation_size)
            .finish()
    }
}

impl KernargBuffer {
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        match self.pointer {
            Some(pointer) => {
                // SAFETY: unique borrow of an owned allocation for `length`.
                unsafe { std::slice::from_raw_parts_mut(pointer.as_ptr(), self.length) }
            }
            None => &mut [],
        }
    }

    pub fn write_exact(&mut self, bytes: &[u8]) -> Result<(), RuntimeError> {
        if bytes.len() != self.length {
            return Err(RuntimeError::KernargLengthMismatch {
                expected: self.length,
                actual: bytes.len(),
            });
        }
        // Device-local (VRAM pool) buffers are host-mapped on ReBAR systems
        // (probe-verified), so plain stores are valid here too.
        self.as_mut_bytes().copy_from_slice(bytes);
        Ok(())
    }

    pub fn address(&self) -> *mut c_void {
        self.pointer
            .map_or(ptr::null_mut(), |pointer| pointer.as_ptr().cast())
    }

    pub fn agent(&self) -> abi::Agent {
        self.pool.gpu
    }
}

impl Drop for KernargBuffer {
    fn drop(&mut self) {
        if let Some(pointer) = self.pointer {
            // The owning graph guarantees all dispatches have completed.
            // SAFETY: pointer was allocated from this runtime memory pool and
            // is uniquely owned here.
            let _ =
                unsafe { (self.pool.runtime.symbols.memory_pool_free)(pointer.as_ptr().cast()) };
        }
    }
}

#[derive(Clone)]
pub struct Executable {
    inner: Arc<ExecutableInner>,
}

struct ExecutableInner {
    runtime: Arc<RuntimeInner>,
    agent: abi::Agent,
    executable: abi::Executable,
    reader: abi::CodeObjectReader,
    // `hsa_code_object_reader_create_from_memory` does not copy this storage.
    _code_object: Arc<[u8]>,
}

impl Executable {
    pub fn load(device: &GpuDevice, code_object: Arc<[u8]>) -> Result<Self, RuntimeError> {
        let code_object = unwrap_clang_offload_bundle(code_object)?;
        if code_object.is_empty() {
            return Err(RuntimeError::EmptyCodeObject);
        }
        let symbols = &device.runtime.symbols;
        let mut reader = abi::CodeObjectReader(0);
        // SAFETY: Arc bytes remain alive in ExecutableInner through reader use.
        let status = unsafe {
            (symbols.code_object_reader_create_from_memory)(
                code_object.as_ptr().cast(),
                code_object.len(),
                &mut reader,
            )
        };
        check_status(symbols, "hsa_code_object_reader_create_from_memory", status)?;
        let mut executable = abi::Executable(0);
        // SAFETY: queried profile/rounding values come from this GPU agent.
        let status = unsafe {
            (symbols.executable_create_alt)(
                device.gpu.profile,
                device.gpu.rounding,
                ptr::null(),
                &mut executable,
            )
        };
        if let Err(error) = check_status(symbols, "hsa_executable_create_alt", status) {
            // SAFETY: reader creation succeeded and is not associated with an
            // executable on this failure path.
            let _ = unsafe { (symbols.code_object_reader_destroy)(reader) };
            return Err(error);
        }
        let mut loaded = abi::LoadedCodeObject(0);
        // SAFETY: handles all belong to this initialized runtime and GPU.
        let status = unsafe {
            (symbols.executable_load_agent_code_object)(
                executable,
                device.gpu_agent(),
                reader,
                ptr::null(),
                &mut loaded,
            )
        };
        if let Err(error) = check_status(symbols, "hsa_executable_load_agent_code_object", status) {
            // SAFETY: both handles were successfully created.
            let _ = unsafe { (symbols.executable_destroy)(executable) };
            let _ = unsafe { (symbols.code_object_reader_destroy)(reader) };
            return Err(error);
        }
        // SAFETY: all code objects are loaded and options may be null.
        let status = unsafe { (symbols.executable_freeze)(executable, ptr::null()) };
        if let Err(error) = check_status(symbols, "hsa_executable_freeze", status) {
            // SAFETY: both handles were successfully created.
            let _ = unsafe { (symbols.executable_destroy)(executable) };
            let _ = unsafe { (symbols.code_object_reader_destroy)(reader) };
            return Err(error);
        }
        Ok(Self {
            inner: Arc::new(ExecutableInner {
                runtime: device.runtime.clone(),
                agent: device.gpu_agent(),
                executable,
                reader,
                _code_object: code_object,
            }),
        })
    }

    /// Resolve a loader symbol such as `mq4gemv_exact.kd` from the frozen
    /// executable. AMDGPU code-object metadata names the source kernel
    /// separately, but the public HSA executable API consumes the `.kd` symbol.
    pub fn kernel(&self, symbol_name: &str) -> Result<Kernel, RuntimeError> {
        let name = CString::new(symbol_name).map_err(|_| RuntimeError::SymbolContainsNul)?;
        let mut symbol = abi::ExecutableSymbol(0);
        // SAFETY: executable is frozen/live, name is NUL-terminated, and agent
        // is the one the code object was loaded for.
        let status = unsafe {
            (self.inner.runtime.symbols.executable_get_symbol_by_name)(
                self.inner.executable,
                name.as_ptr(),
                &self.inner.agent,
                &mut symbol,
            )
        };
        check_status(
            &self.inner.runtime.symbols,
            "hsa_executable_get_symbol_by_name",
            status,
        )?;
        let mut kind = 0_u32;
        symbol_info(
            &self.inner.runtime.symbols,
            symbol,
            abi::EXECUTABLE_SYMBOL_INFO_TYPE,
            (&mut kind as *mut u32).cast(),
        )?;
        if kind != abi::SYMBOL_KIND_KERNEL {
            return Err(RuntimeError::SymbolIsNotKernel(symbol_name.to_owned()));
        }
        let mut metadata = KernelMetadata {
            kernel_object: 0,
            kernarg_segment_size: 0,
            kernarg_segment_alignment: 0,
            group_segment_size: 0,
            private_segment_size: 0,
            dynamic_callstack: false,
        };
        symbol_info(
            &self.inner.runtime.symbols,
            symbol,
            abi::EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
            (&mut metadata.kernel_object as *mut u64).cast(),
        )?;
        symbol_info(
            &self.inner.runtime.symbols,
            symbol,
            abi::EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
            (&mut metadata.kernarg_segment_size as *mut u32).cast(),
        )?;
        symbol_info(
            &self.inner.runtime.symbols,
            symbol,
            abi::EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
            (&mut metadata.kernarg_segment_alignment as *mut u32).cast(),
        )?;
        symbol_info(
            &self.inner.runtime.symbols,
            symbol,
            abi::EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
            (&mut metadata.group_segment_size as *mut u32).cast(),
        )?;
        symbol_info(
            &self.inner.runtime.symbols,
            symbol,
            abi::EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
            (&mut metadata.private_segment_size as *mut u32).cast(),
        )?;
        symbol_info(
            &self.inner.runtime.symbols,
            symbol,
            abi::EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK,
            (&mut metadata.dynamic_callstack as *mut bool).cast(),
        )?;
        let pm4 = parse_kernel_pm4_metadata(
            &self.inner._code_object,
            symbol_name,
            metadata.kernel_object,
        );
        Ok(Kernel {
            executable: self.inner.clone(),
            name: symbol_name.to_owned(),
            metadata,
            pm4,
        })
    }
}

const CLANG_OFFLOAD_BUNDLE_MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";

/// HIP accepts a clang offload bundle at `hipModuleLoad`, while the public HSA
/// code-object reader accepts only the embedded AMDGPU ELF. Unwrap exactly one
/// AMDGPU entry in-process so HIP and AQL consume byte-identical device code.
fn unwrap_clang_offload_bundle(code: Arc<[u8]>) -> Result<Arc<[u8]>, RuntimeError> {
    if !code.starts_with(CLANG_OFFLOAD_BUNDLE_MAGIC) {
        return Ok(code);
    }
    let mut cursor = CLANG_OFFLOAD_BUNDLE_MAGIC.len();
    let bundle_count = read_bundle_u64(&code, &mut cursor)?;
    let mut amdgpu = None;
    for _ in 0..bundle_count {
        let offset = usize::try_from(read_bundle_u64(&code, &mut cursor)?)
            .map_err(|_| RuntimeError::InvalidOffloadBundle("entry offset overflows usize"))?;
        let size = usize::try_from(read_bundle_u64(&code, &mut cursor)?)
            .map_err(|_| RuntimeError::InvalidOffloadBundle("entry size overflows usize"))?;
        let id_len = usize::try_from(read_bundle_u64(&code, &mut cursor)?)
            .map_err(|_| RuntimeError::InvalidOffloadBundle("entry ID length overflows usize"))?;
        let id_end = cursor
            .checked_add(id_len)
            .ok_or(RuntimeError::InvalidOffloadBundle(
                "entry ID range overflows",
            ))?;
        let id = code
            .get(cursor..id_end)
            .ok_or(RuntimeError::InvalidOffloadBundle("truncated entry ID"))?;
        cursor = id_end;
        let end = offset
            .checked_add(size)
            .ok_or(RuntimeError::InvalidOffloadBundle(
                "entry payload range overflows",
            ))?;
        if end > code.len() {
            return Err(RuntimeError::InvalidOffloadBundle(
                "entry payload exceeds bundle",
            ));
        }
        if id
            .windows(b"amdgcn-amd-amdhsa".len())
            .any(|window| window == b"amdgcn-amd-amdhsa")
        {
            if amdgpu.replace((offset, end)).is_some() {
                return Err(RuntimeError::InvalidOffloadBundle(
                    "multiple AMDGPU code objects require an explicit target",
                ));
            }
        }
    }
    let (start, end) = amdgpu.ok_or(RuntimeError::InvalidOffloadBundle(
        "bundle contains no AMDGPU code object",
    ))?;
    Ok(Arc::from(&code[start..end]))
}

fn read_bundle_u64(code: &[u8], cursor: &mut usize) -> Result<u64, RuntimeError> {
    let end = cursor
        .checked_add(8)
        .ok_or(RuntimeError::InvalidOffloadBundle(
            "bundle header overflows",
        ))?;
    let bytes: [u8; 8] = code
        .get(*cursor..end)
        .ok_or(RuntimeError::InvalidOffloadBundle(
            "truncated bundle header",
        ))?
        .try_into()
        .expect("slice length checked");
    *cursor = end;
    Ok(u64::from_le_bytes(bytes))
}

impl Drop for ExecutableInner {
    fn drop(&mut self) {
        // No Kernel can remain: each retains this Arc. Recorded graphs either
        // observe completion or inactivate their queues before releasing
        // Kernel values.
        // SAFETY: executable is live and must be destroyed before its reader.
        let _ = unsafe { (self.runtime.symbols.executable_destroy)(self.executable) };
        // SAFETY: every associated executable has now been destroyed.
        let _ = unsafe { (self.runtime.symbols.code_object_reader_destroy)(self.reader) };
    }
}

fn symbol_info(
    symbols: &abi::Symbols,
    symbol: abi::ExecutableSymbol,
    attribute: u32,
    output: *mut c_void,
) -> Result<(), RuntimeError> {
    // SAFETY: output points to the exact public-header type for `attribute`.
    let status = unsafe { (symbols.executable_symbol_get_info)(symbol, attribute, output) };
    check_status(symbols, "hsa_executable_symbol_get_info", status)
}

#[derive(Clone)]
pub struct Kernel {
    executable: Arc<ExecutableInner>,
    name: String,
    metadata: KernelMetadata,
    pm4: Option<KernelPm4Metadata>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelPm4Metadata {
    pub code_entry: u64,
    pub compute_pgm_rsrc1: u32,
    pub compute_pgm_rsrc2: u32,
    pub compute_pgm_rsrc3: u32,
    pub kernel_code_properties: u16,
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Kernel")
            .field("name", &self.name)
            .field("metadata", &self.metadata)
            .finish()
    }
}

impl Kernel {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn metadata(&self) -> KernelMetadata {
        self.metadata
    }

    pub fn pm4_metadata(&self) -> Option<KernelPm4Metadata> {
        self.pm4
    }

    pub fn agent(&self) -> abi::Agent {
        self.executable.agent
    }
}

fn parse_kernel_pm4_metadata(
    elf: &[u8],
    symbol_name: &str,
    loaded_descriptor: u64,
) -> Option<KernelPm4Metadata> {
    const ELF64_HEADER_BYTES: usize = 64;
    const ELF64_SYMBOL_BYTES: usize = 24;
    const PT_LOAD: u32 = 1;
    const SHT_SYMTAB: u32 = 2;
    const SHT_DYNSYM: u32 = 11;
    if elf.len() < ELF64_HEADER_BYTES || elf.get(0..4)? != b"\x7fELF" {
        return None;
    }
    let phoff = read_elf_u64(elf, 32)? as usize;
    let shoff = read_elf_u64(elf, 40)? as usize;
    let phentsize = read_elf_u16(elf, 54)? as usize;
    let phnum = read_elf_u16(elf, 56)? as usize;
    let shentsize = read_elf_u16(elf, 58)? as usize;
    let shnum = read_elf_u16(elf, 60)? as usize;
    if phentsize < 56 || shentsize < 64 {
        return None;
    }
    let va_to_offset = |va: u64| -> Option<usize> {
        for index in 0..phnum {
            let base = phoff.checked_add(index.checked_mul(phentsize)?)?;
            if read_elf_u32(elf, base)? != PT_LOAD {
                continue;
            }
            let offset = read_elf_u64(elf, base + 8)?;
            let vaddr = read_elf_u64(elf, base + 16)?;
            let filesz = read_elf_u64(elf, base + 32)?;
            if va >= vaddr && va < vaddr.checked_add(filesz)? {
                return usize::try_from(offset.checked_add(va - vaddr)?).ok();
            }
        }
        None
    };
    let mut descriptor_va = None;
    for section in 0..shnum {
        let base = shoff.checked_add(section.checked_mul(shentsize)?)?;
        let section_type = read_elf_u32(elf, base + 4)?;
        if section_type != SHT_SYMTAB && section_type != SHT_DYNSYM {
            continue;
        }
        let symbols_offset = read_elf_u64(elf, base + 24)? as usize;
        let symbols_size = read_elf_u64(elf, base + 32)? as usize;
        let string_section = read_elf_u32(elf, base + 40)? as usize;
        let entry_size = read_elf_u64(elf, base + 56)? as usize;
        if entry_size < ELF64_SYMBOL_BYTES || string_section >= shnum {
            continue;
        }
        let string_base = shoff.checked_add(string_section.checked_mul(shentsize)?)?;
        let strings_offset = read_elf_u64(elf, string_base + 24)? as usize;
        let strings_size = read_elf_u64(elf, string_base + 32)? as usize;
        let strings = elf.get(strings_offset..strings_offset.checked_add(strings_size)?)?;
        for index in 0..(symbols_size / entry_size) {
            let symbol = symbols_offset.checked_add(index.checked_mul(entry_size)?)?;
            let name_offset = read_elf_u32(elf, symbol)? as usize;
            let name_tail = strings.get(name_offset..)?;
            let name_end = name_tail.iter().position(|byte| *byte == 0)?;
            if name_tail.get(..name_end)? == symbol_name.as_bytes() {
                descriptor_va = Some(read_elf_u64(elf, symbol + 8)?);
                break;
            }
        }
        if descriptor_va.is_some() {
            break;
        }
    }
    let descriptor_offset = va_to_offset(descriptor_va?)?;
    let descriptor = elf.get(descriptor_offset..descriptor_offset.checked_add(64)?)?;
    let entry_offset = i64::from_le_bytes(descriptor.get(16..24)?.try_into().ok()?);
    let code_entry = if entry_offset >= 0 {
        loaded_descriptor.checked_add(entry_offset as u64)?
    } else {
        loaded_descriptor.checked_sub(entry_offset.unsigned_abs())?
    };
    Some(KernelPm4Metadata {
        code_entry,
        compute_pgm_rsrc3: read_elf_u32(descriptor, 44)?,
        compute_pgm_rsrc1: read_elf_u32(descriptor, 48)?,
        compute_pgm_rsrc2: read_elf_u32(descriptor, 52)?,
        kernel_code_properties: read_elf_u16(descriptor, 56)?,
    })
}

fn read_elf_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        bytes.get(offset..offset.checked_add(2)?)?.try_into().ok()?,
    ))
}

fn read_elf_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn read_elf_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(offset..offset.checked_add(8)?)?.try_into().ok()?,
    ))
}

fn check_status(
    symbols: &abi::Symbols,
    operation: &'static str,
    status: abi::Status,
) -> Result<(), RuntimeError> {
    if status == abi::STATUS_SUCCESS {
        return Ok(());
    }
    Err(RuntimeError::Hsa {
        operation,
        status,
        message: status_message(symbols, status),
    })
}

fn status_message(symbols: &abi::Symbols, status: abi::Status) -> String {
    let mut pointer = ptr::null();
    // SAFETY: output pointer is valid; failure merely leaves message absent.
    let string_status = unsafe { (symbols.status_string)(status, &mut pointer) };
    if string_status == abi::STATUS_SUCCESS && !pointer.is_null() {
        // SAFETY: successful hsa_status_string returns a runtime-owned
        // NUL-terminated string.
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    } else {
        "unknown HSA status".to_owned()
    }
}

#[derive(Debug)]
pub enum RuntimeError {
    Hsa {
        operation: &'static str,
        status: abi::Status,
        message: String,
    },
    NoGpuAgent,
    NoCpuAgent,
    GpuOrdinalOutOfRange {
        ordinal: usize,
    },
    GpuNameNotFound {
        needle: String,
    },
    InvalidQueueSize {
        requested: u32,
        min: u32,
        max: u32,
    },
    UnsupportedQueueType(u32),
    QueueFault {
        queue_id: u64,
        status: abi::Status,
        message: String,
    },
    QueueInactive {
        queue_id: u64,
    },
    QueueCapacityTimeout {
        queue_id: u64,
        packets: usize,
        timeout: Duration,
    },
    SignalTimeout {
        signal: u64,
        timeout: Duration,
    },
    OperationAndTeardown {
        operation: Box<RuntimeError>,
        teardown: Box<RuntimeError>,
    },
    ZeroQueues,
    DuplicateQueueId,
    InvalidCuMask(&'static str),
    InvalidRuntimeObject(&'static str),
    NoKernargPool,
    InvalidKernargAlignment(usize),
    KernargAlignmentNotMet {
        required: usize,
        address: usize,
    },
    KernargSizeOverflow(usize),
    KernargLengthMismatch {
        expected: usize,
        actual: usize,
    },
    WorkgroupLimit {
        axis: usize,
        requested: u32,
        maximum: u32,
    },
    GridLimit {
        axis: usize,
        requested: u64,
        maximum: u64,
    },
    EmptyCodeObject,
    InvalidOffloadBundle(&'static str),
    SymbolContainsNul,
    SymbolIsNotKernel(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hsa {
                operation,
                status,
                message,
            } => write!(f, "{operation} failed with HSA {status:#x}: {message}"),
            Self::NoGpuAgent => write!(f, "HSA runtime exposed no GPU agent"),
            Self::NoCpuAgent => write!(f, "HSA runtime exposed no CPU agent for kernarg memory"),
            Self::GpuOrdinalOutOfRange { ordinal } => {
                write!(f, "GPU ordinal {ordinal} is out of range")
            }
            Self::GpuNameNotFound { needle } => {
                write!(f, "no HSA GPU name contains {needle:?}")
            }
            Self::InvalidQueueSize {
                requested,
                min,
                max,
            } => write!(
                f,
                "queue size {requested} is not a power of two in {min}..={max}"
            ),
            Self::UnsupportedQueueType(queue_type) => write!(
                f,
                "HSA agent queue type {queue_type} is unsupported; this prototype requires MULTI ({})",
                abi::QUEUE_TYPE_MULTI
            ),
            Self::QueueFault {
                queue_id,
                status,
                message,
            } => write!(
                f,
                "HSA queue {queue_id} reported asynchronous status {status:#x}: {message}"
            ),
            Self::QueueInactive { queue_id } => {
                write!(
                    f,
                    "HSA queue {queue_id} is inactive after cancellation or fault"
                )
            }
            Self::QueueCapacityTimeout {
                queue_id,
                packets,
                timeout,
            } => write!(
                f,
                "HSA queue {queue_id} did not free capacity for {packets} packets within {timeout:?}"
            ),
            Self::SignalTimeout { signal, timeout } => write!(
                f,
                "HSA signal {signal:#x} did not complete within {timeout:?}"
            ),
            Self::OperationAndTeardown {
                operation,
                teardown,
            } => write!(
                f,
                "AQL operation failed ({operation}); queue inactivation also failed ({teardown})"
            ),
            Self::ZeroQueues => write!(f, "at least one HSA queue is required"),
            Self::DuplicateQueueId => write!(f, "ROCr returned duplicate IDs for distinct queues"),
            Self::InvalidCuMask(message) => write!(f, "invalid HSA CU mask: {message}"),
            Self::InvalidRuntimeObject(message) => write!(f, "invalid ROCr object: {message}"),
            Self::NoKernargPool => write!(
                f,
                "no allocatable fine-grained host pool with KERNARG_INIT was found"
            ),
            Self::InvalidKernargAlignment(alignment) => {
                write!(f, "kernel reported invalid kernarg alignment {alignment}")
            }
            Self::KernargAlignmentNotMet { required, address } => write!(
                f,
                "kernarg allocation at {address:#x} does not meet {required}-byte alignment"
            ),
            Self::KernargSizeOverflow(size) => {
                write!(f, "kernarg allocation size overflows while rounding {size}")
            }
            Self::KernargLengthMismatch { expected, actual } => write!(
                f,
                "kernarg payload is {actual} bytes; loader metadata requires {expected}"
            ),
            Self::WorkgroupLimit {
                axis,
                requested,
                maximum,
            } => write!(
                f,
                "work-group axis/total {axis} is {requested}, agent maximum is {maximum}"
            ),
            Self::GridLimit {
                axis,
                requested,
                maximum,
            } => write!(
                f,
                "grid axis/total {axis} is {requested}, agent maximum is {maximum}"
            ),
            Self::EmptyCodeObject => write!(f, "HSACO code object is empty"),
            Self::InvalidOffloadBundle(reason) => {
                write!(f, "invalid clang offload bundle: {reason}")
            }
            Self::SymbolContainsNul => write!(f, "kernel symbol contains an interior NUL"),
            Self::SymbolIsNotKernel(name) => {
                write!(f, "executable symbol {name:?} is not a kernel")
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        #[test]
        fn arbitrary_elf_bytes_never_panic(
            elf in proptest::collection::vec(any::<u8>(), 0..4_096),
            loaded_descriptor in any::<u64>(),
        ) {
            let _ = parse_kernel_pm4_metadata(&elf, "kernel.kd", loaded_descriptor);
        }
    }

    #[test]
    fn queue_fault_callback_keeps_the_first_status_in_stable_state() {
        let state = Box::<QueueFaultState>::default();
        let context = (&*state as *const QueueFaultState).cast_mut().cast();
        // SAFETY: context points at stable live state for both callback calls;
        // the queue pointer is deliberately unused by the callback.
        unsafe { queue_error_callback(0x1001, ptr::null_mut(), context) };
        // SAFETY: same callback contract as above.
        unsafe { queue_error_callback(0x1002, ptr::null_mut(), context) };
        assert_eq!(state.status(), Some(0x1001));
    }

    #[test]
    fn queue_fault_state_can_represent_status_zero() {
        let state = QueueFaultState::default();
        state.record(abi::STATUS_SUCCESS);
        assert_eq!(state.status(), Some(abi::STATUS_SUCCESS));
    }

    #[test]
    fn parses_pm4_resources_and_rebases_descriptor_entry() {
        const PHOFF: usize = 64;
        const SHOFF: usize = 120;
        const SYMTAB: usize = 312;
        const STRTAB: usize = 336;
        const DESCRIPTOR: usize = 384;
        let mut elf = vec![0_u8; 512];
        elf[0..4].copy_from_slice(b"\x7fELF");
        elf[4] = 2; // ELFCLASS64
        elf[5] = 1; // little endian
        elf[32..40].copy_from_slice(&(PHOFF as u64).to_le_bytes());
        elf[40..48].copy_from_slice(&(SHOFF as u64).to_le_bytes());
        elf[54..56].copy_from_slice(&56_u16.to_le_bytes());
        elf[56..58].copy_from_slice(&1_u16.to_le_bytes());
        elf[58..60].copy_from_slice(&64_u16.to_le_bytes());
        elf[60..62].copy_from_slice(&3_u16.to_le_bytes());

        // One PT_LOAD maps ELF virtual addresses directly to file offsets.
        elf[PHOFF..PHOFF + 4].copy_from_slice(&1_u32.to_le_bytes());
        let elf_len = elf.len() as u64;
        elf[PHOFF + 32..PHOFF + 40].copy_from_slice(&elf_len.to_le_bytes());

        // Section 1: one ELF64 symbol; section 2: its string table.
        let sym_section = SHOFF + 64;
        elf[sym_section + 4..sym_section + 8].copy_from_slice(&2_u32.to_le_bytes());
        elf[sym_section + 24..sym_section + 32].copy_from_slice(&(SYMTAB as u64).to_le_bytes());
        elf[sym_section + 32..sym_section + 40].copy_from_slice(&24_u64.to_le_bytes());
        elf[sym_section + 40..sym_section + 44].copy_from_slice(&2_u32.to_le_bytes());
        elf[sym_section + 56..sym_section + 64].copy_from_slice(&24_u64.to_le_bytes());
        let str_section = SHOFF + 128;
        elf[str_section + 4..str_section + 8].copy_from_slice(&3_u32.to_le_bytes());
        elf[str_section + 24..str_section + 32].copy_from_slice(&(STRTAB as u64).to_le_bytes());
        let symbol_name = b"\0kernel.kd\0";
        elf[str_section + 32..str_section + 40]
            .copy_from_slice(&(symbol_name.len() as u64).to_le_bytes());
        elf[STRTAB..STRTAB + symbol_name.len()].copy_from_slice(symbol_name);
        elf[SYMTAB..SYMTAB + 4].copy_from_slice(&1_u32.to_le_bytes());
        elf[SYMTAB + 8..SYMTAB + 16].copy_from_slice(&(DESCRIPTOR as u64).to_le_bytes());

        elf[DESCRIPTOR + 16..DESCRIPTOR + 24].copy_from_slice(&0x120_i64.to_le_bytes());
        elf[DESCRIPTOR + 44..DESCRIPTOR + 48].copy_from_slice(&0x150_u32.to_le_bytes());
        elf[DESCRIPTOR + 48..DESCRIPTOR + 52].copy_from_slice(&0xe00f_0004_u32.to_le_bytes());
        elf[DESCRIPTOR + 52..DESCRIPTOR + 56].copy_from_slice(&0x84_u32.to_le_bytes());
        elf[DESCRIPTOR + 56..DESCRIPTOR + 58].copy_from_slice(&0x408_u16.to_le_bytes());

        let parsed = parse_kernel_pm4_metadata(&elf, "kernel.kd", 0x7f00_0000).unwrap();
        assert_eq!(
            parsed,
            KernelPm4Metadata {
                code_entry: 0x7f00_0120,
                compute_pgm_rsrc1: 0xe00f_0004,
                compute_pgm_rsrc2: 0x84,
                compute_pgm_rsrc3: 0x150,
                kernel_code_properties: 0x408,
            }
        );
    }

    fn queue_depth_sample(queue_id: u64, read_index: u64, write_index: u64) -> QueueDepthSample {
        QueueDepthSample {
            queue_id,
            read_index,
            write_index,
            depth: write_index.wrapping_sub(read_index),
        }
    }

    #[test]
    fn queue_depth_report_separates_shallow_queues_from_an_idle_peer_backlog() {
        let mut report = QueueDepthReport::new([11, 22]);
        report.observe([
            queue_depth_sample(11, 100, 101),
            queue_depth_sample(22, 200, 201),
        ]);
        report.observe([
            queue_depth_sample(11, 101, 101),
            queue_depth_sample(22, 201, 205),
        ]);
        report.observe([
            queue_depth_sample(11, 101, 101),
            queue_depth_sample(22, 205, 205),
        ]);

        assert_eq!(report.poll_count(), 3);
        assert_eq!(report.all_depth_le_one_poll_count(), 2);
        assert_eq!(report.all_empty_poll_count(), 1);
        assert_eq!(report.exactly_one_nonempty_poll_count(), 1);
        assert_eq!(report.backlog_with_idle_peer_poll_count(), 1);

        let first = &report.queues()[0];
        assert_eq!(first.minimum_depth(), Some(0));
        assert_eq!(first.maximum_depth(), Some(1));
        assert_eq!(first.empty_poll_count(), 2);
        assert_eq!(first.depth_le_one_poll_count(), 3);
        assert_eq!(first.mean_depth(), Some(1.0 / 3.0));
        assert_eq!(first.first_sample().unwrap().read_index(), 100);
        assert_eq!(first.last_sample().unwrap().write_index(), 101);

        let second = &report.queues()[1];
        assert_eq!(second.minimum_depth(), Some(0));
        assert_eq!(second.maximum_depth(), Some(4));
        assert_eq!(second.mean_depth(), Some(5.0 / 3.0));
    }

    #[test]
    fn failed_queue_destroy_requires_leaking_callback_and_runtime_owners() {
        assert!(!queue_destroy_requires_leak(abi::STATUS_SUCCESS));
        assert!(queue_destroy_requires_leak(0x1000));
    }

    #[test]
    fn pci_bus_id_parser_normalizes_case_and_width() {
        let parsed = "A:0b:1f.7".parse::<PciBusId>().unwrap();
        assert_eq!(parsed.domain(), 0x0a);
        assert_eq!(parsed.bus(), 0x0b);
        assert_eq!(parsed.device(), 0x1f);
        assert_eq!(parsed.function(), 7);
        assert_eq!(parsed.to_string(), "000a:0b:1f.7");
        assert_eq!(parsed, "000A:0B:1F.7".parse().unwrap());
    }

    #[test]
    fn pci_bus_id_parser_rejects_malformed_or_out_of_range_components() {
        for invalid in [
            "0000:01:00",
            "0000:01:00.8",
            "0000:100:00.0",
            "0000:01:20.0",
            "zzzz:01:00.0",
        ] {
            assert!(invalid.parse::<PciBusId>().is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn hsa_partition_nibble_does_not_change_hip_normalized_pci_identity() {
        let plain = pci_bus_id_from_hsa_location(0x1234, 0x0000_abee);
        let partitioned = pci_bus_id_from_hsa_location(0x1234, 0xf000_abee);
        assert_eq!(plain, partitioned);
        assert_eq!(plain.to_string(), "1234:ab:1d.6");
    }
}

#[cfg(test)]
mod device_local_pool_tests {
    use super::*;

    /// Probe: can the host write/read a GPU-agent coarse-grained (VRAM) pool
    /// allocation directly after agents_allow_access includes the CPU agent?
    /// Gated: only runs with HIPFIRE_REDLINE_PROBE_VMEMAP=1 (needs a GPU).
    #[test]
    fn device_local_pool_host_memcpy_probe() {
        if hipfire_config::developer_var_os("HIPFIRE_REDLINE_PROBE_VMEMAP")
            .is_none_or(|v| v == "0" || v.is_empty())
        {
            eprintln!("probe skipped (set HIPFIRE_REDLINE_PROBE_VMEMAP=1)");
            return;
        }
        let runtime =
            Runtime::initialize(crate::load_symbols().expect("symbols")).expect("runtime init");
        let device = runtime
            .select_gpu(GpuSelector::Ordinal(0))
            .expect("gpu select");
        let pool = match KernargPool::discover_device_local(&device) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("no device-local pool: {e:?}");
                return;
            }
        };
        let mut buf = pool
            .allocate_executable_bytes(4096)
            .expect("device-local alloc");
        // If coarse VRAM is host-mappable after CPU access grant, this writes
        // and reads back without faulting.
        let pattern: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        buf.as_mut_bytes().copy_from_slice(&pattern);
        let readback: Vec<u8> = buf.as_mut_bytes().to_vec();
        assert_eq!(
            pattern, readback,
            "host memcpy round-trip mismatch on device-local pool"
        );
        eprintln!("device-local pool host memcpy probe: OK");
    }
}
