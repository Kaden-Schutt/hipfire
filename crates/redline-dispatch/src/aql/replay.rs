//! Multi-queue AQL record/replay.

use std::collections::BTreeSet;
use std::fmt;
use std::time::Duration;

use redline_rocr::abi;
use redline_rocr::packet::{
    BARRIER_DEPENDENCY_CAPACITY, BarrierAndPacket, KernelDispatchPacket, LaunchGeometry,
    PacketError, PacketImage,
};
use redline_rocr::{
    CompletionSignal, DEFAULT_WAIT_TIMEOUT, FenceScope, Gfx10Pm4CommandBuffer,
    Gfx12Pm4CommandBuffer, GpuDevice, HeaderPolicy, KernargBuffer, KernargPool, Kernel,
    Pm4BuildError, QueueDepthReport, QueueSet, RuntimeError,
};

fn retained_queue_size(
    required_packets: usize,
    minimum: u32,
    maximum: u32,
) -> Result<u32, ReplayError> {
    let queue_size = required_packets
        .checked_next_power_of_two()
        .ok_or_else(|| ReplayError::PolicyShapeMismatch {
            detail: format!("retained packet count {required_packets} overflows queue sizing"),
        })?
        .max(minimum as usize);
    let queue_size = u32::try_from(queue_size).map_err(|_| ReplayError::PolicyShapeMismatch {
        detail: format!("retained queue size {queue_size} exceeds u32"),
    })?;
    if queue_size > maximum {
        return Err(ReplayError::PolicyShapeMismatch {
            detail: format!("retained queue size {queue_size} outside {minimum}..={maximum}"),
        });
    }
    Ok(queue_size)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QuiescenceTransition {
    Completed,
    Cancelled,
    Failed,
}

fn apply_quiescence_transition(
    in_flight: &mut bool,
    usable: &mut bool,
    transition: QuiescenceTransition,
) {
    match transition {
        QuiescenceTransition::Completed => {
            *in_flight = false;
        }
        QuiescenceTransition::Cancelled => {
            *in_flight = false;
            *usable = false;
        }
        QuiescenceTransition::Failed => {
            // The runtime did not prove that packet execution stopped. Retain
            // in-flight state so graph Drop retries queue inactivation.
            *in_flight = true;
            *usable = false;
        }
    }
}

/// One retained GFX12 PM4 indirect buffer submitted through AMD's
/// vendor-specific AQL packet. One queue publication replaces every dispatch
/// packet in the command stream while preserving public ROCr ownership and
/// completion handling.
pub struct SingleQueuePm4Ib {
    queues: QueueSet,
    completion: CompletionSignal,
    indirect: KernargBuffer,
    batch: Vec<PacketImage>,
    timestamps: Option<KernargBuffer>,
    timestamp_frequency_hz: Option<u64>,
    dispatch_workgroup_offsets: Vec<usize>,
    usable: bool,
}

impl SingleQueuePm4Ib {
    pub fn create(
        device: &GpuDevice,
        pool: &KernargPool,
        commands: &Gfx12Pm4CommandBuffer,
    ) -> Result<Self, ReplayError> {
        Self::create_encoded(
            device,
            pool,
            None,
            &commands.as_bytes(),
            commands.len_dwords(),
            None,
            None,
        )
    }

    /// [`Self::create`] with the indirect buffer allocated from a separate
    /// (device-local/VRAM) pool; everything host-read stays on `pool`.
    pub fn create_with_ib_pool(
        device: &GpuDevice,
        pool: &KernargPool,
        ib_pool: &KernargPool,
        commands: &Gfx12Pm4CommandBuffer,
    ) -> Result<Self, ReplayError> {
        Self::create_encoded(
            device,
            pool,
            Some(ib_pool),
            &commands.as_bytes(),
            commands.len_dwords(),
            None,
            None,
        )
    }

    /// [`Self::create_with_ib_pool`] with GPU-clock timestamps (the shadow
    /// harness's AQL batch profiling contract): timestamps stay on the
    /// host pool, IB on the device-local pool.
    pub fn create_profiled_with_ib_pool(
        device: &GpuDevice,
        pool: &KernargPool,
        ib_pool: &KernargPool,
        commands: &Gfx12Pm4CommandBuffer,
    ) -> Result<Self, ReplayError> {
        if commands.is_empty() {
            return Err(ReplayError::EmptyGraph);
        }
        let mut timestamps = pool.allocate_executable_bytes(16)?;
        timestamps.as_mut_bytes().fill(0);
        let start = timestamps.address() as usize as u64;
        let timed = commands.with_gpu_timestamps(start, start + 8);
        let frequency_hz = device.gpu_timestamp_frequency_hz()?;
        Self::create_encoded(
            device,
            pool,
            Some(ib_pool),
            &timed.as_bytes(),
            timed.len_dwords(),
            Some(timestamps),
            Some(frequency_hz),
        )
    }

    /// Create a retained GFX10 PM4 IB using the legacy compute-register map.
    pub fn create_gfx10(
        device: &GpuDevice,
        pool: &KernargPool,
        commands: &Gfx10Pm4CommandBuffer,
    ) -> Result<Self, ReplayError> {
        Self::create_encoded(
            device,
            pool,
            None,
            &commands.as_bytes(),
            commands.len_dwords(),
            None,
            None,
        )
    }

    /// GFX11 shares the GFX10 compute-register map but remains an explicit
    /// constructor so callers cannot silently submit a legacy IB on GFX12.
    pub fn create_gfx11(
        device: &GpuDevice,
        pool: &KernargPool,
        commands: &Gfx10Pm4CommandBuffer,
    ) -> Result<Self, ReplayError> {
        Self::create_encoded(
            device,
            pool,
            None,
            &commands.as_bytes(),
            commands.len_dwords(),
            None,
            None,
        )
    }

    /// Build a retained tape instrumented with one GPU-clock write after each
    /// dispatch plus a baseline before the stream, for attribution rather than
    /// certification. Later spans therefore include intervening boundary
    /// packets between dispatches; span 0 includes the entry acquire.
    ///
    /// The resulting tape has a different dword count and sequence hash than
    /// the certified route, so it cannot satisfy a golden fixture — that is
    /// inherent, not an oversight. Pair with [`replay_and_wait_dispatch_spans`].
    pub fn create_dispatch_profiled(
        device: &GpuDevice,
        pool: &KernargPool,
        commands: &Gfx12Pm4CommandBuffer,
    ) -> Result<Self, ReplayError> {
        if commands.is_empty() {
            return Err(ReplayError::EmptyGraph);
        }
        let slots = commands
            .timestamp_slot_count()
            .map_err(ReplayError::Pm4Build)?;
        let mut timestamps = pool.allocate_executable_bytes(slots * 8)?;
        timestamps.as_mut_bytes().fill(0);
        let base = timestamps.address() as usize as u64;
        let timed = commands
            .with_per_dispatch_timestamps(base)
            .map_err(ReplayError::Pm4Build)?;
        let frequency_hz = device.gpu_timestamp_frequency_hz()?;
        Self::create_encoded(
            device,
            pool,
            None,
            &timed.as_bytes(),
            timed.len_dwords(),
            Some(timestamps),
            Some(frequency_hz),
        )
    }

    pub fn create_profiled(
        device: &GpuDevice,
        pool: &KernargPool,
        commands: &Gfx12Pm4CommandBuffer,
    ) -> Result<Self, ReplayError> {
        if commands.is_empty() {
            return Err(ReplayError::EmptyGraph);
        }
        let mut timestamps = pool.allocate_executable_bytes(16)?;
        timestamps.as_mut_bytes().fill(0);
        let start = timestamps.address() as usize as u64;
        let timed = commands.with_gpu_timestamps(start, start + 8);
        let frequency_hz = device.gpu_timestamp_frequency_hz()?;
        Self::create_encoded(
            device,
            pool,
            None,
            &timed.as_bytes(),
            timed.len_dwords(),
            Some(timestamps),
            Some(frequency_hz),
        )
    }

    pub fn create_profiled_gfx10(
        device: &GpuDevice,
        pool: &KernargPool,
        commands: &Gfx10Pm4CommandBuffer,
    ) -> Result<Self, ReplayError> {
        Self::create_profiled_legacy(device, pool, commands)
    }

    pub fn create_profiled_gfx11(
        device: &GpuDevice,
        pool: &KernargPool,
        commands: &Gfx10Pm4CommandBuffer,
    ) -> Result<Self, ReplayError> {
        Self::create_profiled_legacy(device, pool, commands)
    }

    fn create_profiled_legacy(
        device: &GpuDevice,
        pool: &KernargPool,
        commands: &Gfx10Pm4CommandBuffer,
    ) -> Result<Self, ReplayError> {
        if commands.is_empty() {
            return Err(ReplayError::EmptyGraph);
        }
        let mut timestamps = pool.allocate_executable_bytes(16)?;
        timestamps.as_mut_bytes().fill(0);
        let start = timestamps.address() as usize as u64;
        let timed = commands.with_gpu_timestamps(start, start + 8);
        let frequency_hz = device.gpu_timestamp_frequency_hz()?;
        Self::create_encoded(
            device,
            pool,
            None,
            &timed.as_bytes(),
            timed.len_dwords(),
            Some(timestamps),
            Some(frequency_hz),
        )
    }

    fn create_encoded(
        device: &GpuDevice,
        pool: &KernargPool,
        ib_pool: Option<&KernargPool>,
        bytes: &[u8],
        dwords: u32,
        timestamps: Option<KernargBuffer>,
        timestamp_frequency_hz: Option<u64>,
    ) -> Result<Self, ReplayError> {
        if dwords == 0 {
            return Err(ReplayError::EmptyGraph);
        }
        let dispatch_workgroup_offsets = pm4_dispatch_workgroup_offsets(bytes)?;
        // The IB may live in a device-local (VRAM) pool so the command
        // processor fetches the tape from VRAM instead of re-reading it over
        // the host interface on every replay; everything host-read
        // (timestamps, completion) stays on the CPU-agent pool.
        let mut indirect = ib_pool
            .unwrap_or(pool)
            .allocate_executable_bytes(bytes.len())?;
        // Device-local (VRAM) IBs are host-mapped on ReBAR systems, so the
        // same store path works for both pools.
        indirect.write_exact(bytes)?;

        let completion = CompletionSignal::new(device)?;
        let packet =
            PacketImage::pm4_indirect_buffer(indirect.address(), dwords, completion.raw())?;
        let queue_size = *device.queue_size_range().start();
        let queues = QueueSet::create(device, 1, queue_size)?;
        Ok(Self {
            queues,
            completion,
            indirect,
            batch: vec![packet],
            timestamps,
            timestamp_frequency_hz,
            dispatch_workgroup_offsets,
            usable: true,
        })
    }

    pub fn queue_id(&self) -> u64 {
        self.queues
            .queue_ids()
            .next()
            .expect("single-queue PM4 replay owns one queue")
    }

    /// Restrict the retained replay queue to an ABI-valid WGP-paired CU mask.
    /// This must be called before the first submission.
    pub fn set_cu_mask(&self, mask_bits: u32, mask: &[u32]) -> Result<(), ReplayError> {
        self.queues.set_cu_mask(mask_bits, mask).map_err(Into::into)
    }

    pub fn indirect_address(&self) -> usize {
        self.indirect.address() as usize
    }

    /// Patch one quiescent `DISPATCH_DIRECT` packet's three raw dimensions.
    ///
    /// GFX10/GFX11 encode work-group counts while GFX12's
    /// `USE_THREAD_DIMENSIONS` encoding uses total work-items. The
    /// architecture-aware caller must convert into the command buffer's native
    /// unit before using this low-level patch surface.
    ///
    /// Retained PM4 replay is synchronous, so a successful prior replay has
    /// proved that the indirect buffer is no longer in flight. This permits a
    /// small dynamic-shape binding to update the next submission without
    /// rebuilding the command stream or weakening its dependency boundaries.
    pub fn patch_dispatch_dimensions(
        &mut self,
        dispatch: usize,
        dimensions: [u32; 3],
    ) -> Result<(), ReplayError> {
        if !self.usable {
            return Err(ReplayError::GraphInactive);
        }
        if dimensions.contains(&0) {
            return Err(ReplayError::Pm4ZeroWorkgroup {
                dispatch,
                workgroups: dimensions,
            });
        }
        let offset = *self.dispatch_workgroup_offsets.get(dispatch).ok_or(
            ReplayError::Pm4DispatchOutOfBounds {
                dispatch,
                dispatches: self.dispatch_workgroup_offsets.len(),
            },
        )?;
        let bytes = self.indirect.as_mut_bytes();
        let end = offset + 12;
        let destination = bytes
            .get_mut(offset..end)
            .ok_or(ReplayError::MalformedPm4IndirectBuffer { dword: offset / 4 })?;
        for (axis, value) in dimensions.into_iter().enumerate() {
            destination[axis * 4..axis * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }

    /// Submit and synchronously prove completion with a finite timeout.
    ///
    /// # Safety
    ///
    /// All code, kernarg, and pointee addresses encoded in the retained IB
    /// must remain live and GPU-accessible until this returns `Ok`. After an
    /// error they must remain live through this object's destruction.
    pub unsafe fn replay_and_wait(&mut self) -> Result<(), ReplayError> {
        // SAFETY: forwarded from this method's caller.
        unsafe { self.replay_and_wait_inner() }
    }

    /// Replay once and return the GPU-clock span bracketed inside the retained
    /// indirect buffer, excluding host queue publication and signal polling.
    pub unsafe fn replay_and_wait_profiled(&mut self) -> Result<GpuMultiQueueTiming, ReplayError> {
        let frequency_hz = self
            .timestamp_frequency_hz
            .ok_or(ReplayError::ProfilingUnavailable)?;
        // SAFETY: forwarded from this method's caller.
        unsafe { self.replay_and_wait_inner()? };
        let bytes = self
            .timestamps
            .as_mut()
            .ok_or(ReplayError::ProfilingUnavailable)?
            .as_mut_bytes();
        let start = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let end = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        if start == 0 || end < start {
            return Err(ReplayError::InvalidGpuTimestamp { start, end });
        }
        Ok(GpuMultiQueueTiming {
            first_start: start,
            last_end: end,
            frequency_hz,
        })
    }

    /// Replay once and return per-dispatch GPU-clock spans in nanoseconds.
    ///
    /// Instrumentation writes one baseline timestamp before the retained stream
    /// and one timestamp after each `DISPATCH_DIRECT`. `spans[i]` is therefore
    /// the interval from timestamp *i* to timestamp *i+1*: all PM4 commands
    /// after timestamp *i* through dispatch *i*, including any preceding
    /// wait/acquire boundary packets (and the entry acquire for span 0). The
    /// sum is the whole retained span; the distribution shows whether a deficit
    /// is spread evenly or concentrated.
    ///
    /// Requires a tape built by [`create_dispatch_profiled`]. A tape with only
    /// the two bracketing timestamps yields a single span, which is what
    /// [`replay_and_wait_profiled`] already reports.
    pub unsafe fn replay_and_wait_dispatch_spans(&mut self) -> Result<Vec<u64>, ReplayError> {
        let frequency_hz = self
            .timestamp_frequency_hz
            .ok_or(ReplayError::ProfilingUnavailable)?;
        // SAFETY: forwarded from this method's caller.
        unsafe { self.replay_and_wait_inner()? };
        let bytes = self
            .timestamps
            .as_mut()
            .ok_or(ReplayError::ProfilingUnavailable)?
            .as_mut_bytes();
        let ticks: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|word| u64::from_le_bytes(word.try_into().unwrap()))
            .collect();
        if ticks.len() < 2 {
            return Err(ReplayError::ProfilingUnavailable);
        }
        let mut spans = Vec::with_capacity(ticks.len() - 1);
        for pair in ticks.windows(2) {
            let (start, end) = (pair[0], pair[1]);
            // A zero or non-monotonic pair means the write did not land; report
            // it rather than silently producing a plausible-looking number.
            if start == 0 || end < start {
                return Err(ReplayError::InvalidGpuTimestamp { start, end });
            }
            spans.push((end - start).saturating_mul(1_000_000_000) / frequency_hz);
        }
        Ok(spans)
    }

    /// Replay exactly once and return both whole-tape and per-dispatch timing.
    ///
    /// Requires a tape built by [`create_dispatch_profiled`].
    pub unsafe fn replay_and_wait_dispatch_profiled(
        &mut self,
    ) -> Result<(GpuMultiQueueTiming, Vec<u64>), ReplayError> {
        // SAFETY: forwarded from this method's caller.
        unsafe { self.replay_and_wait_inner()? };
        self.read_dispatch_profile()
    }

    /// Decode the timestamps written by the most recent instrumented replay.
    pub fn read_dispatch_profile(
        &mut self,
    ) -> Result<(GpuMultiQueueTiming, Vec<u64>), ReplayError> {
        let frequency_hz = self
            .timestamp_frequency_hz
            .ok_or(ReplayError::ProfilingUnavailable)?;
        let bytes = self
            .timestamps
            .as_mut()
            .ok_or(ReplayError::ProfilingUnavailable)?
            .as_mut_bytes();
        let ticks = bytes
            .chunks_exact(8)
            .map(|word| u64::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        if ticks.len() < 2 {
            return Err(ReplayError::ProfilingUnavailable);
        }
        let mut spans_nanoseconds = Vec::with_capacity(ticks.len() - 1);
        for pair in ticks.windows(2) {
            let (start, end) = (pair[0], pair[1]);
            if start == 0 || end < start {
                return Err(ReplayError::InvalidGpuTimestamp { start, end });
            }
            spans_nanoseconds.push((end - start).saturating_mul(1_000_000_000) / frequency_hz);
        }
        Ok((
            GpuMultiQueueTiming {
                first_start: ticks[0],
                last_end: *ticks.last().unwrap(),
                frequency_hz,
            },
            spans_nanoseconds,
        ))
    }

    unsafe fn replay_and_wait_inner(&mut self) -> Result<(), ReplayError> {
        if !self.usable {
            return Err(ReplayError::GraphInactive);
        }
        self.completion.reset();
        if let Err(error) = self
            .queues
            .prepare_batches(std::slice::from_ref(&self.batch))
        {
            self.usable = false;
            return Err(error.into());
        }
        if let Err(error) = self.queues.ring_prepared() {
            self.usable = false;
            return Err(error.into());
        }
        match self
            .queues
            .wait_signal(&self.completion, DEFAULT_WAIT_TIMEOUT)
        {
            Ok(()) => Ok(()),
            Err(operation) => {
                self.usable = false;
                match self.queues.inactivate_all() {
                    Ok(()) => Err(operation.into()),
                    Err(teardown) => Err(RuntimeError::OperationAndTeardown {
                        operation: Box::new(operation),
                        teardown: Box::new(teardown),
                    }
                    .into()),
                }
            }
        }
    }
}

fn pm4_dispatch_workgroup_offsets(bytes: &[u8]) -> Result<Vec<usize>, ReplayError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(ReplayError::MalformedPm4IndirectBuffer {
            dword: bytes.len() / 4,
        });
    }
    let dwords = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mut offsets = Vec::new();
    let mut cursor = 0_usize;
    while cursor < dwords.len() {
        let header = dwords[cursor];
        if header >> 30 != 3 {
            return Err(ReplayError::MalformedPm4IndirectBuffer { dword: cursor });
        }
        let body_dwords = ((header >> 16) & 0x3fff) as usize + 1;
        let next = cursor
            .checked_add(1 + body_dwords)
            .ok_or(ReplayError::MalformedPm4IndirectBuffer { dword: cursor })?;
        if next > dwords.len() {
            return Err(ReplayError::MalformedPm4IndirectBuffer { dword: cursor });
        }
        let opcode = (header >> 8) & 0xff;
        if opcode == 0x15 {
            if body_dwords != 4 {
                return Err(ReplayError::MalformedPm4IndirectBuffer { dword: cursor });
            }
            offsets.push((cursor + 1) * 4);
        }
        cursor = next;
    }
    Ok(offsets)
}

impl Drop for SingleQueuePm4Ib {
    fn drop(&mut self) {
        if !self.usable {
            let _ = self.queues.inactivate_all();
        }
    }
}

/// Ordered retained-PM4 phases over one reusable public queue set.
///
/// Commands within a phase must be pairwise independent. AQL barrier packets
/// make every active lane in phase N wait for all active lanes in phase N-1,
/// so the full graph still costs one publication and one terminal host wait.
/// Each IB carries a GPU-clock bracket; profiled replay reports the makespan
/// from the earliest phase start through the latest phase end.
pub struct PhasedMultiQueuePm4Ib {
    queues: QueueSet,
    phase_completions: Vec<Vec<CompletionSignal>>,
    logical_phase_count: usize,
    indirects: Vec<KernargBuffer>,
    batches: Vec<Vec<PacketImage>>,
    timestamps: Option<Vec<KernargBuffer>>,
    _native_semaphores: Option<KernargBuffer>,
    timestamp_frequency_hz: Option<u64>,
    usable: bool,
}

impl PhasedMultiQueuePm4Ib {
    pub fn create_profiled(
        device: &GpuDevice,
        pool: &KernargPool,
        phases: &[Vec<Gfx12Pm4CommandBuffer>],
    ) -> Result<Self, ReplayError> {
        let frequency_hz = device.gpu_timestamp_frequency_hz()?;
        Self::create_encoded(
            device,
            pool,
            phases,
            |commands, timestamps| {
                let commands = match timestamps {
                    Some((start, end)) => commands.with_gpu_timestamps(start, end),
                    None => commands.clone(),
                };
                (commands.as_bytes(), commands.len_dwords())
            },
            Some(frequency_hz),
        )
    }

    pub fn create_profiled_gfx10(
        device: &GpuDevice,
        pool: &KernargPool,
        phases: &[Vec<Gfx10Pm4CommandBuffer>],
    ) -> Result<Self, ReplayError> {
        Self::create_profiled_legacy(device, pool, phases)
    }

    pub fn create_profiled_gfx11(
        device: &GpuDevice,
        pool: &KernargPool,
        phases: &[Vec<Gfx10Pm4CommandBuffer>],
    ) -> Result<Self, ReplayError> {
        Self::create_profiled_legacy(device, pool, phases)
    }

    /// Create one retained PM4 IB per queue and synchronize phases inside the
    /// IBs with GPU memory semaphores. This replaces every per-phase AQL
    /// completion signal and barrier packet with `RELEASE_MEM`/`WAIT_REG_MEM`,
    /// leaving only one terminal completion signal per queue.
    pub fn create_profiled_native_gfx10(
        device: &GpuDevice,
        pool: &KernargPool,
        phases: &[Vec<Gfx10Pm4CommandBuffer>],
    ) -> Result<Self, ReplayError> {
        Self::create_profiled_native_legacy(device, pool, phases)
    }

    /// GFX11 uses the same compute-ring semaphore packet encodings as GFX10.
    pub fn create_profiled_native_gfx11(
        device: &GpuDevice,
        pool: &KernargPool,
        phases: &[Vec<Gfx10Pm4CommandBuffer>],
    ) -> Result<Self, ReplayError> {
        Self::create_profiled_native_legacy(device, pool, phases)
    }

    fn create_profiled_legacy(
        device: &GpuDevice,
        pool: &KernargPool,
        phases: &[Vec<Gfx10Pm4CommandBuffer>],
    ) -> Result<Self, ReplayError> {
        let frequency_hz = device.gpu_timestamp_frequency_hz()?;
        Self::create_encoded(
            device,
            pool,
            phases,
            |commands, timestamps| {
                let commands = match timestamps {
                    Some((start, end)) => commands.with_gpu_timestamps(start, end),
                    None => commands.clone(),
                };
                (commands.as_bytes(), commands.len_dwords())
            },
            Some(frequency_hz),
        )
    }

    fn create_profiled_native_legacy(
        device: &GpuDevice,
        pool: &KernargPool,
        phases: &[Vec<Gfx10Pm4CommandBuffer>],
    ) -> Result<Self, ReplayError> {
        if phases.is_empty() || phases.iter().any(Vec::is_empty) {
            return Err(ReplayError::EmptyGraph);
        }
        if phases
            .iter()
            .flatten()
            .any(|commands| !commands.ends_with_compute_idle())
        {
            return Err(ReplayError::PolicyShapeMismatch {
                detail:
                    "native PM4 semaphore publication requires every phase lane to end compute-idle"
                        .to_owned(),
            });
        }
        let queue_count = phases.iter().map(Vec::len).max().unwrap();
        if queue_count < 2 {
            return Err(ReplayError::PolicyShapeMismatch {
                detail: "native PM4 phase synchronization requires at least two queues".to_owned(),
            });
        }
        let parallel_phase_count = phases.iter().filter(|phase| phase.len() > 1).count();
        if parallel_phase_count == 1 {
            return Err(ReplayError::PolicyShapeMismatch {
                detail:
                    "native PM4 retained epochs require either zero or at least two parallel phases"
                        .to_owned(),
            });
        }
        let semaphore_bytes =
            (queue_count - 1)
                .checked_mul(8)
                .ok_or_else(|| ReplayError::PolicyShapeMismatch {
                    detail: format!(
                        "native PM4 semaphore count for {queue_count} queues overflowed"
                    ),
                })?;
        let mut semaphores = pool.allocate_executable_bytes(semaphore_bytes)?;
        semaphores.as_mut_bytes().fill(0);
        let semaphore_base = semaphores.address() as usize as u64;
        let semaphore_addresses = |lane: usize| {
            let start = semaphore_base + ((lane - 1) * 8) as u64;
            (start, start + 4)
        };

        let mut streams = vec![Gfx10Pm4CommandBuffer::new(); queue_count];
        let mut parallel_epoch = 0_u32;
        for phase in phases {
            if phase.len() == 1 {
                streams[0].append_stream(&phase[0]);
                continue;
            }
            parallel_epoch =
                parallel_epoch
                    .checked_add(1)
                    .ok_or_else(|| ReplayError::PolicyShapeMismatch {
                        detail: "native PM4 parallel phase count exceeds u32".to_owned(),
                    })?;
            for lane in 1..phase.len() {
                let (start, _) = semaphore_addresses(lane);
                streams[lane].wait_memory_value(start, parallel_epoch);
            }
            for lane in 1..phase.len() {
                let (start, _) = semaphore_addresses(lane);
                if streams[0].ends_with_compute_idle() {
                    streams[0].write_memory_value_after_idle(start, parallel_epoch);
                } else {
                    streams[0].release_memory_value(start, parallel_epoch);
                }
            }
            streams[0].append_stream(&phase[0]);
            for lane in 1..phase.len() {
                let (_, done) = semaphore_addresses(lane);
                streams[lane].append_stream(&phase[lane]);
                streams[lane].write_memory_value_after_idle(done, parallel_epoch);
            }
            for lane in 1..phase.len() {
                let (_, done) = semaphore_addresses(lane);
                streams[0].wait_memory_value(done, parallel_epoch);
            }
        }
        if parallel_epoch == 0 {
            return Err(ReplayError::PolicyShapeMismatch {
                detail: "native PM4 phase synchronization found no parallel phase".to_owned(),
            });
        }

        let frequency_hz = device.gpu_timestamp_frequency_hz()?;
        let mut completions = Vec::with_capacity(queue_count);
        let mut indirects = Vec::with_capacity(queue_count);
        let mut batches = Vec::with_capacity(queue_count);
        let mut timestamps = Vec::with_capacity(queue_count);
        for commands in streams {
            let mut timestamp = pool.allocate_executable_bytes(16)?;
            timestamp.as_mut_bytes().fill(0);
            let start = timestamp.address() as usize as u64;
            let timed = commands.with_gpu_timestamps(start, start + 8);
            let bytes = timed.as_bytes();
            let dwords = timed.len_dwords();
            if dwords == 0 {
                return Err(ReplayError::EmptyGraph);
            }
            let mut indirect = pool.allocate_executable_bytes(bytes.len())?;
            indirect.write_exact(&bytes)?;
            let completion = CompletionSignal::new(device)?;
            let packet =
                PacketImage::pm4_indirect_buffer(indirect.address(), dwords, completion.raw())?;
            timestamps.push(timestamp);
            indirects.push(indirect);
            completions.push(completion);
            batches.push(vec![packet]);
        }

        let queue_size = retained_queue_size(
            1,
            *device.queue_size_range().start(),
            *device.queue_size_range().end(),
        )?;
        let queues = QueueSet::create(device, queue_count, queue_size)?;
        Ok(Self {
            queues,
            phase_completions: vec![completions],
            logical_phase_count: phases.len(),
            indirects,
            batches,
            timestamps: Some(timestamps),
            _native_semaphores: Some(semaphores),
            timestamp_frequency_hz: Some(frequency_hz),
            usable: true,
        })
    }

    fn create_encoded<C>(
        device: &GpuDevice,
        pool: &KernargPool,
        phases: &[Vec<C>],
        encode: impl Fn(&C, Option<(u64, u64)>) -> (Vec<u8>, u32),
        timestamp_frequency_hz: Option<u64>,
    ) -> Result<Self, ReplayError> {
        if phases.is_empty() || phases.iter().any(Vec::is_empty) {
            return Err(ReplayError::EmptyGraph);
        }
        let queue_count = phases.iter().map(Vec::len).max().unwrap();
        let mut batches = vec![Vec::new(); queue_count];
        let mut indirects = Vec::new();
        let mut timestamps = timestamp_frequency_hz.map(|_| Vec::new());
        let mut phase_completions = Vec::<Vec<CompletionSignal>>::with_capacity(phases.len());
        let mut prior = Vec::new();

        for phase in phases {
            let mut completions = Vec::with_capacity(phase.len());
            for _ in phase {
                completions.push(CompletionSignal::new(device)?);
            }
            for (lane, commands) in phase.iter().enumerate() {
                for dependencies in prior.chunks(BARRIER_DEPENDENCY_CAPACITY) {
                    let barrier = BarrierAndPacket::new(dependencies, abi::Signal(0))?;
                    batches[lane].push(PacketImage::barrier(&barrier));
                }
                let timestamp_pair = if let Some(timestamps) = timestamps.as_mut() {
                    let mut timestamp = pool.allocate_executable_bytes(16)?;
                    timestamp.as_mut_bytes().fill(0);
                    let start = timestamp.address() as usize as u64;
                    timestamps.push(timestamp);
                    Some((start, start + 8))
                } else {
                    None
                };
                let (bytes, dwords) = encode(commands, timestamp_pair);
                if dwords == 0 {
                    return Err(ReplayError::EmptyGraph);
                }
                let mut indirect = pool.allocate_executable_bytes(bytes.len())?;
                indirect.write_exact(&bytes)?;
                let packet = PacketImage::pm4_indirect_buffer(
                    indirect.address(),
                    dwords,
                    completions[lane].raw(),
                )?;
                indirects.push(indirect);
                batches[lane].push(packet);
            }
            prior = completions.iter().map(CompletionSignal::raw).collect();
            phase_completions.push(completions);
        }

        let required_packets = batches.iter().map(Vec::len).max().unwrap();
        let queue_size = retained_queue_size(
            required_packets,
            *device.queue_size_range().start(),
            *device.queue_size_range().end(),
        )?;
        let queues = QueueSet::create(device, queue_count, queue_size)?;
        for (lane, batch) in batches.iter().enumerate() {
            let capacity = queues.size(lane).expect("queue count is fixed") as usize;
            if batch.len() > capacity {
                return Err(ReplayError::BatchExceedsQueue {
                    lane,
                    packets: batch.len(),
                    capacity,
                });
            }
        }

        Ok(Self {
            queues,
            phase_completions,
            logical_phase_count: phases.len(),
            indirects,
            batches,
            timestamps,
            _native_semaphores: None,
            timestamp_frequency_hz,
            usable: true,
        })
    }

    pub fn queue_count(&self) -> usize {
        self.queues.len()
    }

    pub fn queue_ids(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.queues.queue_ids()
    }

    pub fn phase_count(&self) -> usize {
        self.logical_phase_count
    }

    pub fn indirect_addresses(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.indirects
            .iter()
            .map(|indirect| indirect.address() as usize)
    }

    /// Replay every ordered phase and return its cross-queue GPU makespan.
    ///
    /// # Safety
    ///
    /// Commands within each phase must be pairwise memory-independent. Every
    /// encoded pointer must remain live and GPU-accessible until completion.
    pub unsafe fn replay_and_wait_profiled(&mut self) -> Result<GpuMultiQueueTiming, ReplayError> {
        let frequency_hz = self
            .timestamp_frequency_hz
            .ok_or(ReplayError::ProfilingUnavailable)?;
        // SAFETY: forwarded from this method's caller.
        unsafe { self.replay_and_wait_inner()? };
        let timestamps = self
            .timestamps
            .as_mut()
            .ok_or(ReplayError::ProfilingUnavailable)?;
        let mut first_start = u64::MAX;
        let mut last_end = 0_u64;
        for timestamp in timestamps {
            let bytes = timestamp.as_mut_bytes();
            let start = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            let end = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
            if start == 0 || end < start {
                return Err(ReplayError::InvalidGpuTimestamp { start, end });
            }
            first_start = first_start.min(start);
            last_end = last_end.max(end);
        }
        Ok(GpuMultiQueueTiming {
            first_start,
            last_end,
            frequency_hz,
        })
    }

    unsafe fn replay_and_wait_inner(&mut self) -> Result<(), ReplayError> {
        if !self.usable {
            return Err(ReplayError::GraphInactive);
        }
        for completions in &mut self.phase_completions {
            for completion in completions {
                completion.reset();
            }
        }
        if let Err(error) = self.queues.prepare_batches(&self.batches) {
            self.usable = false;
            return Err(error.into());
        }
        if let Err(error) = self.queues.ring_prepared() {
            self.usable = false;
            return Err(error.into());
        }
        let terminal = self
            .phase_completions
            .last()
            .expect("nonempty phase list has terminal completions");
        if let Err(operation) = self.queues.wait_signals(terminal, DEFAULT_WAIT_TIMEOUT) {
            self.usable = false;
            return match self.queues.inactivate_all() {
                Ok(()) => Err(operation.into()),
                Err(teardown) => Err(RuntimeError::OperationAndTeardown {
                    operation: Box::new(operation),
                    teardown: Box::new(teardown),
                }
                .into()),
            };
        }
        Ok(())
    }
}

impl Drop for PhasedMultiQueuePm4Ib {
    fn drop(&mut self) {
        if !self.usable {
            let _ = self.queues.inactivate_all();
        }
    }
}

/// One immutable recorded dispatch.
///
/// Dependencies are indices of earlier `RecordedDispatch` values in the graph.
/// The `KernargBuffer` and `Kernel` keep argument memory and the executable
/// alive through every replay.
#[derive(Debug)]
pub struct RecordedDispatch {
    lane: usize,
    kernel: Kernel,
    geometry: LaunchGeometry,
    kernarg: KernargBuffer,
    dynamic_group_bytes: u32,
    dependencies: Vec<usize>,
}

impl RecordedDispatch {
    pub fn new(
        lane: usize,
        kernel: Kernel,
        geometry: LaunchGeometry,
        kernarg: KernargBuffer,
    ) -> Result<Self, ReplayError> {
        if kernarg.len() != kernel.metadata().kernarg_segment_size as usize {
            return Err(ReplayError::KernargMetadataMismatch {
                kernel: kernel.name().to_owned(),
                metadata_bytes: kernel.metadata().kernarg_segment_size as usize,
                buffer_bytes: kernarg.len(),
            });
        }
        if kernarg.agent() != kernel.agent() {
            return Err(ReplayError::MixedGpuObjects);
        }
        Ok(Self {
            lane,
            kernel,
            geometry,
            kernarg,
            dynamic_group_bytes: 0,
            dependencies: Vec::new(),
        })
    }

    /// Add the launch's dynamic LDS request to the loader-derived static LDS.
    /// Packet construction checks arithmetic; queue execution remains the
    /// authoritative hardware limit, as with HIP module launch.
    pub fn with_dynamic_group_bytes(mut self, bytes: u32) -> Result<Self, ReplayError> {
        self.dynamic_group_bytes = bytes;
        Ok(self)
    }

    pub fn with_dependencies(mut self, dependencies: impl IntoIterator<Item = usize>) -> Self {
        self.dependencies.extend(dependencies);
        self
    }

    pub fn lane(&self) -> usize {
        self.lane
    }

    pub fn dependencies(&self) -> &[usize] {
        &self.dependencies
    }

    pub(crate) fn kernarg_mut(&mut self) -> &mut KernargBuffer {
        &mut self.kernarg
    }
}

/// An immutable DAG bound to N distinct public HSA queues.
///
/// Each dispatch has a reusable completion signal. Cross-queue dependencies
/// become barrier-AND packets on the consumer lane. More than five inputs are
/// emitted as a sequence of barrier packets; because each barrier has its
/// header barrier bit set, the sequence is a logical AND over every chunk.
/// Terminal signals are similarly folded on queue zero into one replay signal.
pub struct RecordedGraph {
    queues: QueueSet,
    nodes: Vec<RecordedDispatch>,
    node_signals: Vec<CompletionSignal>,
    final_signal: CompletionSignal,
    terminal_nodes: Vec<usize>,
    batches: Vec<Vec<PacketImage>>,
    in_flight: bool,
    usable: bool,
}

impl fmt::Debug for RecordedGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecordedGraph")
            .field("queues", &self.queues.queue_ids().collect::<Vec<_>>())
            .field("node_count", &self.nodes.len())
            .field("terminal_nodes", &self.terminal_nodes)
            .field("in_flight", &self.in_flight)
            .field("usable", &self.usable)
            .finish()
    }
}

impl RecordedGraph {
    pub fn create(
        device: &GpuDevice,
        queue_count: usize,
        queue_size: u32,
        nodes: Vec<RecordedDispatch>,
    ) -> Result<Self, ReplayError> {
        if nodes.is_empty() {
            return Err(ReplayError::EmptyGraph);
        }
        validate_nodes(device, queue_count, &nodes)?;
        let queues = QueueSet::create(device, queue_count, queue_size)?;
        let mut node_signals = Vec::with_capacity(nodes.len());
        for _ in 0..nodes.len() {
            node_signals.push(CompletionSignal::new(device)?);
        }
        let final_signal = CompletionSignal::new(device)?;

        let mut depended_on = vec![false; nodes.len()];
        for node in &nodes {
            for dependency in &node.dependencies {
                depended_on[*dependency] = true;
            }
        }
        let terminal_nodes = depended_on
            .iter()
            .enumerate()
            .filter_map(|(index, used)| (!used).then_some(index))
            .collect::<Vec<_>>();

        // Build every potentially failing packet and queue-local batch before
        // anything can reach a doorbell. Replay reserves each batch in one
        // operation and rings each queue exactly once.
        let mut batches = vec![Vec::new(); queue_count];
        for (index, recorded) in nodes.iter().enumerate() {
            let dependencies = recorded
                .dependencies
                .iter()
                .map(|dependency| node_signals[*dependency].raw())
                .collect::<Vec<_>>();
            for chunk in dependencies.chunks(BARRIER_DEPENDENCY_CAPACITY) {
                let barrier = BarrierAndPacket::new(chunk, abi::Signal(0))?;
                batches[recorded.lane].push(PacketImage::barrier(&barrier));
            }
            let packet = KernelDispatchPacket::new(
                recorded.kernel.metadata(),
                recorded.geometry,
                recorded.dynamic_group_bytes,
                recorded.kernarg.address(),
                node_signals[index].raw(),
            )?;
            batches[recorded.lane].push(PacketImage::kernel(&packet));
        }
        let terminal_signals = terminal_nodes
            .iter()
            .map(|index| node_signals[*index].raw())
            .collect::<Vec<_>>();
        for chunk in terminal_signals.chunks(BARRIER_DEPENDENCY_CAPACITY) {
            let barrier = BarrierAndPacket::new(chunk, abi::Signal(0))?;
            batches[0].push(PacketImage::barrier(&barrier));
        }
        let completion = BarrierAndPacket::new(&[], final_signal.raw())?;
        batches[0].push(PacketImage::barrier(&completion));
        for (lane, batch) in batches.iter().enumerate() {
            let capacity = queues.size(lane).expect("queue count is fixed") as usize;
            if batch.len() > capacity {
                return Err(ReplayError::BatchExceedsQueue {
                    lane,
                    packets: batch.len(),
                    capacity,
                });
            }
        }

        Ok(Self {
            queues,
            nodes,
            node_signals,
            final_signal,
            terminal_nodes,
            batches,
            in_flight: false,
            usable: true,
        })
    }

    pub fn queue_ids(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.queues.queue_ids()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_in_flight(&self) -> bool {
        self.in_flight
    }

    pub fn doorbell_writes_per_replay(&self) -> usize {
        self.batches
            .iter()
            .filter(|batch| !batch.is_empty())
            .count()
    }

    /// Submit one replay and return a completion/cancellation ticket.
    ///
    /// A second token cannot be submitted until the first ticket completes.
    /// This is the token-latency core; a throughput implementation needs a ring
    /// of signal/kernarg generations and is intentionally left as integration
    /// work rather than unsafely resetting in-flight signals.
    ///
    /// # Safety
    ///
    /// Kernarg bytes can contain arbitrary device pointers which this crate
    /// cannot inspect. Every pointee must remain allocated, accessible to this
    /// graph's GPU, and free of incompatible host/agent mutation until an
    /// explicit `wait` or `cancel` returns `Ok`. After any `Err`, or after
    /// dropping a submission ticket, every pointee must instead remain valid
    /// through destruction of the graph itself. Only `Ok` proves quiescence.
    pub unsafe fn submit(&mut self) -> Result<ReplaySubmission<'_>, ReplayError> {
        if self.in_flight {
            return Err(ReplayError::AlreadyInFlight);
        }
        if !self.usable {
            return Err(ReplayError::GraphInactive);
        }
        for signal in &mut self.node_signals {
            signal.reset();
        }
        self.final_signal.reset();
        if let Err(error) = self.queues.prepare_batches(&self.batches) {
            self.usable = false;
            return Err(error.into());
        }
        self.in_flight = true;
        if let Err(error) = self.queues.ring_prepared() {
            self.in_flight = false;
            self.usable = false;
            return Err(error.into());
        }

        Ok(ReplaySubmission {
            graph: self,
            completed: false,
        })
    }

    /// Submit and wait using the default finite timeout.
    ///
    /// # Safety
    ///
    /// The pointer and access contract is identical to [`Self::submit`].
    pub unsafe fn replay_and_wait(&mut self) -> Result<(), ReplayError> {
        // SAFETY: forwarded from this method's caller.
        unsafe { self.submit()? }.wait()
    }

    fn wait_internal(&mut self, timeout: Duration) -> Result<(), ReplayError> {
        if !self.in_flight {
            return Ok(());
        }
        match self.queues.wait_signal(&self.final_signal, timeout) {
            Ok(()) => {
                apply_quiescence_transition(
                    &mut self.in_flight,
                    &mut self.usable,
                    QuiescenceTransition::Completed,
                );
                Ok(())
            }
            Err(wait_error) => match self.queues.inactivate_all() {
                Ok(()) => {
                    apply_quiescence_transition(
                        &mut self.in_flight,
                        &mut self.usable,
                        QuiescenceTransition::Cancelled,
                    );
                    Err(wait_error.into())
                }
                Err(teardown_error) => {
                    apply_quiescence_transition(
                        &mut self.in_flight,
                        &mut self.usable,
                        QuiescenceTransition::Failed,
                    );
                    Err(RuntimeError::OperationAndTeardown {
                        operation: Box::new(wait_error),
                        teardown: Box::new(teardown_error),
                    }
                    .into())
                }
            },
        }
    }

    fn cancel_internal(&mut self) -> Result<(), ReplayError> {
        if !self.in_flight {
            return Ok(());
        }
        match self.queues.inactivate_all() {
            Ok(()) => {
                apply_quiescence_transition(
                    &mut self.in_flight,
                    &mut self.usable,
                    QuiescenceTransition::Cancelled,
                );
                Ok(())
            }
            Err(error) => {
                apply_quiescence_transition(
                    &mut self.in_flight,
                    &mut self.usable,
                    QuiescenceTransition::Failed,
                );
                Err(error.into())
            }
        }
    }
}

impl Drop for RecordedGraph {
    fn drop(&mut self) {
        // Drop never waits on a completion signal. It retries queue
        // inactivation when an earlier ticket operation could not prove
        // quiescence; Queue Drop performs the final destroy attempt.
        let _ = self.cancel_internal();
    }
}

#[must_use = "wait or cancel the AQL submission to observe completion/teardown errors"]
pub struct ReplaySubmission<'a> {
    graph: &'a mut RecordedGraph,
    completed: bool,
}

impl ReplaySubmission<'_> {
    /// Wait with the default finite host-polling timeout.
    ///
    /// `Ok` proves quiescence. On `Err`, the external-pointee contract from
    /// [`RecordedGraph::submit`] remains in force through graph destruction.
    pub fn wait(self) -> Result<(), ReplayError> {
        self.wait_timeout(DEFAULT_WAIT_TIMEOUT)
    }

    /// Wait with an explicit finite host-polling timeout.
    ///
    /// `Ok` proves quiescence. On `Err`, the external-pointee contract from
    /// [`RecordedGraph::submit`] remains in force through graph destruction.
    pub fn wait_timeout(mut self, timeout: Duration) -> Result<(), ReplayError> {
        let result = self.graph.wait_internal(timeout);
        self.completed = true;
        result
    }

    /// Abort pending queue execution without waiting for a completion signal.
    /// `Ok` proves quiescence; `Err` requires retaining external pointees
    /// through graph destruction.
    pub fn cancel(mut self) -> Result<(), ReplayError> {
        let result = self.graph.cancel_internal();
        self.completed = true;
        result
    }
}

impl Drop for ReplaySubmission<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.graph.cancel_internal();
        }
    }
}

/// Packet-header policy for the single-queue fence attribution experiment.
///
/// `BoundarySerialized` is the cache/fence-only control: it changes fence
/// scopes but retains a barrier between every dispatch. `BoundaryIndependent`
/// additionally clears those dispatch barrier bits and is valid only when the
/// caller has independently proved the dispatches do not conflict.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchFencePolicy {
    SystemEveryDispatch,
    SystemAcquireAgentRelease,
    AgentEveryInternalDispatch,
    BoundarySerialized,
    BoundaryIndependent,
}

impl BatchFencePolicy {
    fn dispatch_header(self, index: usize) -> redline_rocr::HeaderPolicy {
        use redline_rocr::HeaderPolicy;
        match self {
            Self::SystemEveryDispatch => HeaderPolicy::RECORDED_DISPATCH,
            Self::SystemAcquireAgentRelease => HeaderPolicy::TWO_QUEUE_DISPATCH,
            Self::AgentEveryInternalDispatch if index == 0 => HeaderPolicy::TWO_QUEUE_DISPATCH,
            Self::AgentEveryInternalDispatch => HeaderPolicy::SAME_AGENT_DISPATCH,
            Self::BoundarySerialized if index == 0 => HeaderPolicy::BATCH_BOUNDARY_FIRST_SERIAL,
            Self::BoundarySerialized => HeaderPolicy::BATCH_BOUNDARY_INTERNAL_SERIAL,
            Self::BoundaryIndependent if index == 0 => {
                HeaderPolicy::BATCH_BOUNDARY_FIRST_INDEPENDENT
            }
            Self::BoundaryIndependent => HeaderPolicy::BATCH_BOUNDARY_INTERNAL_INDEPENDENT,
        }
    }
}

/// GPU timestamps spanning the first through last dispatch of one recorded
/// batch. Terminal host-export barrier time is intentionally excluded, matching
/// compute-stage timestamp brackets in the Vulkan and HIP controls.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuBatchTiming {
    pub first_start: u64,
    pub first_end: u64,
    pub last_start: u64,
    pub last_end: u64,
    pub frequency_hz: u64,
}

/// GPU timestamp span across the first and last profiled dispatches on every
/// active queue of a derived batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMultiQueueTiming {
    pub first_start: u64,
    pub last_end: u64,
    pub frequency_hz: u64,
}

impl GpuMultiQueueTiming {
    pub fn span_microseconds(self) -> f64 {
        self.last_end.saturating_sub(self.first_start) as f64 * 1_000_000.0
            / self.frequency_hz as f64
    }
}

impl GpuBatchTiming {
    pub fn span_microseconds(self) -> f64 {
        self.last_end.saturating_sub(self.first_start) as f64 * 1_000_000.0
            / self.frequency_hz as f64
    }

    pub fn dispatch_span_microseconds(self) -> f64 {
        self.last_end.saturating_sub(self.first_start) as f64 * 1_000_000.0
            / self.frequency_hz as f64
    }
}

/// One prebuilt single-queue dispatch batch used to attribute packet fences.
///
/// The graph owns every kernel, kernarg buffer, profiling signal, and queue.
/// Exactly the first and last dispatch carry profiling signals; the terminal
/// barrier carries the host completion signal. This avoids attaching a signal
/// to every dispatch and perturbing the policy under test.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostGate {
    /// Dispatch whose System-release completion publishes route data to the
    /// host. Dispatches after this point may continue until `resume_before`.
    pub checkpoint_dispatch: usize,
    /// Insert an owner-local barrier immediately before this dispatch. The
    /// barrier resumes only after the host publishes the corresponding result.
    pub resume_before: usize,
}

fn validate_host_gates(dispatch_count: usize, gates: &[HostGate]) -> Result<(), ReplayError> {
    let mut previous: Option<HostGate> = None;
    for (index, gate) in gates.iter().enumerate() {
        if gate.checkpoint_dispatch >= dispatch_count || gate.resume_before >= dispatch_count {
            return Err(ReplayError::PolicyShapeMismatch {
                detail: format!(
                    "host gate {index} ({}, {}) exceeds {dispatch_count} dispatches",
                    gate.checkpoint_dispatch, gate.resume_before
                ),
            });
        }
        if gate.checkpoint_dispatch >= gate.resume_before {
            return Err(ReplayError::PolicyShapeMismatch {
                detail: format!(
                    "host gate {index} checkpoint {} must precede resume dispatch {}",
                    gate.checkpoint_dispatch, gate.resume_before
                ),
            });
        }
        if let Some(previous) = previous {
            if previous.resume_before > gate.checkpoint_dispatch {
                return Err(ReplayError::PolicyShapeMismatch {
                    detail: format!(
                        "host gate {index} checkpoint {} overlaps prior resume dispatch {}",
                        gate.checkpoint_dispatch, previous.resume_before
                    ),
                });
            }
        }
        previous = Some(*gate);
    }
    Ok(())
}

pub struct SingleQueueBatchGraph {
    device: GpuDevice,
    queues: QueueSet,
    dispatches: Vec<RecordedDispatch>,
    first_signal: CompletionSignal,
    last_signal: CompletionSignal,
    checkpoint_signals: Vec<CompletionSignal>,
    resume_signals: Vec<CompletionSignal>,
    host_gates: Vec<HostGate>,
    final_signal: CompletionSignal,
    batch: Vec<PacketImage>,
    policy: BatchFencePolicy,
    timestamp_frequency_hz: u64,
    profiling: bool,
    in_flight: bool,
    usable: bool,
}

impl fmt::Debug for SingleQueueBatchGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SingleQueueBatchGraph")
            .field("queue_ids", &self.queues.queue_ids().collect::<Vec<_>>())
            .field("dispatch_count", &self.dispatches.len())
            .field("policy", &self.policy)
            .field("in_flight", &self.in_flight)
            .field("usable", &self.usable)
            .finish()
    }
}

impl SingleQueueBatchGraph {
    pub fn create(
        device: &GpuDevice,
        queue_size: u32,
        dispatches: Vec<RecordedDispatch>,
        policy: BatchFencePolicy,
    ) -> Result<Self, ReplayError> {
        let headers = (0..dispatches.len())
            .map(|index| policy.dispatch_header(index))
            .collect();
        Self::create_with_dispatch_headers(device, queue_size, dispatches, policy, headers)
    }

    pub fn create_with_dispatch_headers(
        device: &GpuDevice,
        queue_size: u32,
        dispatches: Vec<RecordedDispatch>,
        policy: BatchFencePolicy,
        headers: Vec<redline_rocr::HeaderPolicy>,
    ) -> Result<Self, ReplayError> {
        Self::create_with_dispatch_headers_mode(
            device,
            queue_size,
            dispatches,
            policy,
            headers,
            true,
            None,
            Vec::new(),
        )
    }

    pub fn create_unprofiled_with_dispatch_headers(
        device: &GpuDevice,
        queue_size: u32,
        dispatches: Vec<RecordedDispatch>,
        policy: BatchFencePolicy,
        headers: Vec<redline_rocr::HeaderPolicy>,
    ) -> Result<Self, ReplayError> {
        Self::create_with_dispatch_headers_mode(
            device,
            queue_size,
            dispatches,
            policy,
            headers,
            false,
            None,
            Vec::new(),
        )
    }

    /// Build one retained owner-local batch with a host-observable completion
    /// checkpoint after `checkpoint_dispatch`. The queue continues consuming
    /// later packets while the host handles the checkpoint; submission still
    /// uses one batch publication and one doorbell write.
    ///
    /// The checkpoint dispatch must carry a System release because the host
    /// consumes data it published before observing the completion signal.
    pub fn create_checkpointed_unprofiled_with_dispatch_headers(
        device: &GpuDevice,
        queue_size: u32,
        dispatches: Vec<RecordedDispatch>,
        policy: BatchFencePolicy,
        headers: Vec<redline_rocr::HeaderPolicy>,
        checkpoint_dispatch: usize,
    ) -> Result<Self, ReplayError> {
        Self::create_with_dispatch_headers_mode(
            device,
            queue_size,
            dispatches,
            policy,
            headers,
            false,
            Some(checkpoint_dispatch),
            Vec::new(),
        )
    }

    /// Build one retained owner-local queue containing multiple finite
    /// host-supervised checkpoints. Each checkpoint publishes preceding route
    /// data, permits owner-local tail work to continue, and then stops at a
    /// barrier until the host releases the corresponding owner-local signal.
    ///
    /// Queue publication and the doorbell happen once for the complete batch.
    /// No signal in this graph is owned by another GPU agent.
    pub fn create_host_gated_unprofiled_with_dispatch_headers(
        device: &GpuDevice,
        queue_size: u32,
        dispatches: Vec<RecordedDispatch>,
        policy: BatchFencePolicy,
        headers: Vec<redline_rocr::HeaderPolicy>,
        host_gates: Vec<HostGate>,
    ) -> Result<Self, ReplayError> {
        Self::create_with_dispatch_headers_mode(
            device, queue_size, dispatches, policy, headers, false, None, host_gates,
        )
    }

    fn create_with_dispatch_headers_mode(
        device: &GpuDevice,
        queue_size: u32,
        dispatches: Vec<RecordedDispatch>,
        policy: BatchFencePolicy,
        headers: Vec<redline_rocr::HeaderPolicy>,
        profiling: bool,
        checkpoint_dispatch: Option<usize>,
        host_gates: Vec<HostGate>,
    ) -> Result<Self, ReplayError> {
        if dispatches.len() < 2 {
            return Err(ReplayError::InvalidBatchShape(
                "single-queue profiling batch requires at least two dispatches",
            ));
        }
        if headers.len() != dispatches.len() {
            return Err(ReplayError::PolicyShapeMismatch {
                detail: format!(
                    "single-queue batch has {} dispatches but {} packet headers",
                    dispatches.len(),
                    headers.len()
                ),
            });
        }
        if checkpoint_dispatch.is_some() && !host_gates.is_empty() {
            return Err(ReplayError::PolicyShapeMismatch {
                detail: "single checkpoint and host gates are mutually exclusive".to_owned(),
            });
        }
        validate_host_gates(dispatches.len(), &host_gates)?;
        let checkpoint_dispatches = checkpoint_dispatch
            .into_iter()
            .chain(host_gates.iter().map(|gate| gate.checkpoint_dispatch))
            .collect::<Vec<_>>();
        for checkpoint in &checkpoint_dispatches {
            let Some(header) = headers.get(*checkpoint) else {
                return Err(ReplayError::PolicyShapeMismatch {
                    detail: format!(
                        "checkpoint dispatch {checkpoint} exceeds {} dispatches",
                        dispatches.len()
                    ),
                });
            };
            if header.release != FenceScope::System {
                return Err(ReplayError::PolicyShapeMismatch {
                    detail: format!(
                        "checkpoint dispatch {checkpoint} requires a System release header"
                    ),
                });
            }
        }
        validate_nodes(device, 1, &dispatches)?;
        for dispatch in &dispatches {
            if dispatch.lane != 0 {
                return Err(ReplayError::PhaseLaneMismatch {
                    expected: 0,
                    actual: dispatch.lane,
                });
            }
            if !dispatch.dependencies.is_empty() {
                return Err(ReplayError::PhaseHasExplicitDependencies { lane: 0 });
            }
        }

        let queues = QueueSet::create(device, 1, queue_size)?;
        if profiling {
            queues.set_profiling(true)?;
        }
        let first_signal = CompletionSignal::new(device)?;
        let last_signal = CompletionSignal::new(device)?;
        let mut checkpoint_signals = Vec::with_capacity(checkpoint_dispatches.len());
        for _ in &checkpoint_dispatches {
            checkpoint_signals.push(CompletionSignal::new(device)?);
        }
        let mut resume_signals = Vec::with_capacity(host_gates.len());
        for _ in &host_gates {
            resume_signals.push(CompletionSignal::new(device)?);
        }
        let final_signal = CompletionSignal::new(device)?;
        let mut batch = Vec::with_capacity(dispatches.len() + host_gates.len() + 1);
        for (index, (dispatch, header)) in dispatches.iter().zip(&headers).enumerate() {
            if let Some((gate_index, _)) = host_gates
                .iter()
                .enumerate()
                .find(|(_, gate)| gate.resume_before == index)
            {
                let barrier = BarrierAndPacket::new_with_policy(
                    &[resume_signals[gate_index].raw()],
                    abi::Signal(0),
                    HeaderPolicy::BATCH_INTERNAL_ACQUIRE_SYSTEM,
                )?;
                batch.push(PacketImage::barrier(&barrier));
            }
            let completion = if let Some(checkpoint_index) = checkpoint_dispatches
                .iter()
                .position(|checkpoint| *checkpoint == index)
            {
                checkpoint_signals[checkpoint_index].raw()
            } else if profiling && index == 0 {
                first_signal.raw()
            } else if profiling && index + 1 == dispatches.len() {
                last_signal.raw()
            } else {
                abi::Signal(0)
            };
            let packet = KernelDispatchPacket::new_with_policy(
                dispatch.kernel.metadata(),
                dispatch.geometry,
                dispatch.dynamic_group_bytes,
                dispatch.kernarg.address(),
                completion,
                *header,
            )?;
            batch.push(PacketImage::kernel(&packet));
        }
        // The host consumes only the completion signal; every payload buffer
        // is consumed next by another queue on the same GPU agent. Agent
        // release is therefore sufficient here and avoids a whole-system
        // writeback at every token boundary. The following HIP submission is
        // issued only after the host observes this terminal signal.
        let terminal = BarrierAndPacket::new_with_policy(
            &[],
            final_signal.raw(),
            HeaderPolicy::BATCH_INTERNAL_RELEASE_AGENT,
        )?;
        batch.push(PacketImage::barrier(&terminal));
        let capacity = queues.size(0).expect("one queue was created") as usize;
        if batch.len() > capacity {
            return Err(ReplayError::BatchExceedsQueue {
                lane: 0,
                packets: batch.len(),
                capacity,
            });
        }
        let timestamp_frequency_hz = if profiling {
            device.timestamp_frequency_hz()?
        } else {
            1
        };

        Ok(Self {
            device: device.clone(),
            queues,
            dispatches,
            first_signal,
            last_signal,
            checkpoint_signals,
            resume_signals,
            host_gates,
            final_signal,
            batch,
            policy,
            timestamp_frequency_hz,
            profiling,
            in_flight: false,
            usable: true,
        })
    }

    pub fn policy(&self) -> BatchFencePolicy {
        self.policy
    }

    pub fn dispatch_count(&self) -> usize {
        self.dispatches.len()
    }

    pub fn packet_count(&self) -> usize {
        self.batch.len()
    }

    pub fn queue_id(&self) -> u64 {
        self.queues.queue_ids().next().expect("one queue exists")
    }

    pub fn has_checkpoint(&self) -> bool {
        !self.checkpoint_signals.is_empty()
    }

    pub fn host_gate_count(&self) -> usize {
        self.host_gates.len()
    }

    /// Patch one explicitly bounded field in retained kernarg storage between
    /// replays. Packet images retain the same kernarg address; only the owned
    /// bytes behind that address change before queue publication.
    pub fn patch_kernarg_bytes(
        &mut self,
        dispatch: usize,
        offset: usize,
        value: &[u8],
    ) -> Result<(), ReplayError> {
        if self.in_flight {
            return Err(ReplayError::KernargPatchWhileInFlight);
        }
        let recorded =
            self.dispatches
                .get_mut(dispatch)
                .ok_or(ReplayError::KernargPatchOutOfBounds {
                    dispatch,
                    offset,
                    bytes: value.len(),
                    kernarg_bytes: 0,
                })?;
        let kernarg_bytes = recorded.kernarg.len();
        let end = offset
            .checked_add(value.len())
            .ok_or(ReplayError::KernargPatchOutOfBounds {
                dispatch,
                offset,
                bytes: value.len(),
                kernarg_bytes,
            })?;
        let destination = recorded.kernarg.as_mut_bytes().get_mut(offset..end).ok_or(
            ReplayError::KernargPatchOutOfBounds {
                dispatch,
                offset,
                bytes: value.len(),
                kernarg_bytes,
            },
        )?;
        destination.copy_from_slice(value);
        Ok(())
    }

    /// Patch one dynamic scalar in retained kernarg storage between replays.
    pub fn patch_kernarg_u32(
        &mut self,
        dispatch: usize,
        offset: usize,
        value: u32,
    ) -> Result<(), ReplayError> {
        self.patch_kernarg_bytes(dispatch, offset, &value.to_le_bytes())
    }

    /// # Safety
    ///
    /// Every external pointer embedded in the recorded kernargs must remain
    /// valid and free of incompatible mutation until the ticket returns `Ok`.
    pub unsafe fn submit(&mut self) -> Result<SingleQueueBatchSubmission<'_>, ReplayError> {
        if self.in_flight {
            return Err(ReplayError::AlreadyInFlight);
        }
        if !self.usable {
            return Err(ReplayError::GraphInactive);
        }
        self.first_signal.reset();
        self.last_signal.reset();
        for signal in &mut self.checkpoint_signals {
            signal.reset();
        }
        for signal in &mut self.resume_signals {
            signal.reset();
        }
        self.final_signal.reset();
        if let Err(error) = self
            .queues
            .prepare_batches(std::slice::from_ref(&self.batch))
        {
            self.usable = false;
            return Err(error.into());
        }
        self.in_flight = true;
        if let Err(error) = self.queues.ring_prepared() {
            self.in_flight = false;
            self.usable = false;
            return Err(error.into());
        }
        Ok(SingleQueueBatchSubmission {
            graph: self,
            completed: false,
            next_host_gate: 0,
            ready_host_gate: None,
        })
    }

    /// # Safety
    ///
    /// The pointer contract is identical to [`Self::submit`].
    pub unsafe fn replay_and_wait(&mut self) -> Result<GpuBatchTiming, ReplayError> {
        // SAFETY: forwarded from this method's caller.
        unsafe { self.submit()? }.wait()
    }

    fn wait_internal(&mut self, timeout: Duration) -> Result<GpuBatchTiming, ReplayError> {
        if !self.in_flight {
            return Err(ReplayError::InvalidBatchShape(
                "single-queue batch has no active submission",
            ));
        }
        match self.queues.wait_signal(&self.final_signal, timeout) {
            Ok(()) => {
                self.in_flight = false;
                if !self.profiling {
                    return Ok(GpuBatchTiming {
                        first_start: 0,
                        first_end: 0,
                        last_start: 0,
                        last_end: 0,
                        frequency_hz: 1,
                    });
                }
                let first = self.device.dispatch_time(&self.first_signal);
                let last = self.device.dispatch_time(&self.last_signal);
                match (first, last) {
                    (Ok(first), Ok(last)) => Ok(GpuBatchTiming {
                        first_start: first.start,
                        first_end: first.end,
                        last_start: last.start,
                        last_end: last.end,
                        frequency_hz: self.timestamp_frequency_hz,
                    }),
                    (Err(error), _) | (_, Err(error)) => {
                        self.usable = false;
                        Err(error.into())
                    }
                }
            }
            Err(wait_error) => match self.queues.inactivate_all() {
                Ok(()) => {
                    apply_quiescence_transition(
                        &mut self.in_flight,
                        &mut self.usable,
                        QuiescenceTransition::Cancelled,
                    );
                    Err(wait_error.into())
                }
                Err(teardown_error) => {
                    apply_quiescence_transition(
                        &mut self.in_flight,
                        &mut self.usable,
                        QuiescenceTransition::Failed,
                    );
                    Err(RuntimeError::OperationAndTeardown {
                        operation: Box::new(wait_error),
                        teardown: Box::new(teardown_error),
                    }
                    .into())
                }
            },
        }
    }

    fn wait_checkpoint_internal(
        &mut self,
        checkpoint_index: usize,
        timeout: Duration,
    ) -> Result<(), ReplayError> {
        if !self.in_flight {
            return Err(ReplayError::InvalidBatchShape(
                "single-queue batch has no active submission",
            ));
        }
        let signal =
            self.checkpoint_signals
                .get(checkpoint_index)
                .ok_or(ReplayError::InvalidBatchShape(
                    "checkpoint index is invalid",
                ))?;
        match self.queues.wait_signal(signal, timeout) {
            Ok(()) => Ok(()),
            Err(wait_error) => match self.queues.inactivate_all() {
                Ok(()) => {
                    apply_quiescence_transition(
                        &mut self.in_flight,
                        &mut self.usable,
                        QuiescenceTransition::Cancelled,
                    );
                    Err(wait_error.into())
                }
                Err(teardown_error) => {
                    apply_quiescence_transition(
                        &mut self.in_flight,
                        &mut self.usable,
                        QuiescenceTransition::Failed,
                    );
                    Err(RuntimeError::OperationAndTeardown {
                        operation: Box::new(wait_error),
                        teardown: Box::new(teardown_error),
                    }
                    .into())
                }
            },
        }
    }

    fn release_host_gate_internal(&mut self, gate_index: usize) -> Result<(), ReplayError> {
        if !self.in_flight {
            return Err(ReplayError::InvalidBatchShape(
                "single-queue batch has no active submission",
            ));
        }
        let signal = self
            .resume_signals
            .get_mut(gate_index)
            .ok_or(ReplayError::InvalidBatchShape("host gate index is invalid"))?;
        signal.complete_from_host();
        Ok(())
    }

    fn cancel_internal(&mut self) -> Result<(), ReplayError> {
        if !self.in_flight {
            return Ok(());
        }
        match self.queues.inactivate_all() {
            Ok(()) => {
                apply_quiescence_transition(
                    &mut self.in_flight,
                    &mut self.usable,
                    QuiescenceTransition::Cancelled,
                );
                Ok(())
            }
            Err(error) => {
                apply_quiescence_transition(
                    &mut self.in_flight,
                    &mut self.usable,
                    QuiescenceTransition::Failed,
                );
                Err(error.into())
            }
        }
    }
}

impl Drop for SingleQueueBatchGraph {
    fn drop(&mut self) {
        let _ = self.cancel_internal();
    }
}

#[must_use = "wait or cancel the AQL submission"]
pub struct SingleQueueBatchSubmission<'a> {
    graph: &'a mut SingleQueueBatchGraph,
    completed: bool,
    next_host_gate: usize,
    ready_host_gate: Option<usize>,
}

impl SingleQueueBatchSubmission<'_> {
    /// Wait until the retained batch's owner-local checkpoint completes while
    /// leaving later packets in flight on the same queue.
    pub fn wait_checkpoint(&mut self) -> Result<(), ReplayError> {
        self.wait_checkpoint_timeout(DEFAULT_WAIT_TIMEOUT)
    }

    pub fn wait_checkpoint_timeout(&mut self, timeout: Duration) -> Result<(), ReplayError> {
        if self.graph.host_gate_count() != 0 {
            return Err(ReplayError::InvalidBatchShape(
                "host-gated batches must use wait_next_host_gate",
            ));
        }
        self.graph.wait_checkpoint_internal(0, timeout)
    }

    /// Wait for the next host gate's route checkpoint. Checkpoints and resumes
    /// must be consumed in capture order so a stale host response cannot open a
    /// later layer gate.
    pub fn wait_next_host_gate(&mut self, timeout: Duration) -> Result<usize, ReplayError> {
        if self.graph.host_gate_count() == 0 {
            return Err(ReplayError::InvalidBatchShape(
                "single-queue batch has no host gates",
            ));
        }
        if self.ready_host_gate.is_some() {
            return Err(ReplayError::InvalidBatchShape(
                "the current host gate must resume before waiting for another checkpoint",
            ));
        }
        if self.next_host_gate >= self.graph.host_gate_count() {
            return Err(ReplayError::InvalidBatchShape(
                "all host gates have already resumed",
            ));
        }
        self.graph
            .wait_checkpoint_internal(self.next_host_gate, timeout)?;
        self.ready_host_gate = Some(self.next_host_gate);
        Ok(self.next_host_gate)
    }

    /// Publish the current owner-local result and release its queue barrier.
    pub fn resume_host_gate(&mut self, gate_index: usize) -> Result<(), ReplayError> {
        if self.ready_host_gate != Some(gate_index) {
            return Err(ReplayError::InvalidBatchShape(
                "host gates must resume in checkpoint order",
            ));
        }
        self.graph.release_host_gate_internal(gate_index)?;
        self.ready_host_gate = None;
        self.next_host_gate += 1;
        Ok(())
    }

    pub fn wait(self) -> Result<GpuBatchTiming, ReplayError> {
        self.wait_timeout(DEFAULT_WAIT_TIMEOUT)
    }

    pub fn wait_timeout(mut self, timeout: Duration) -> Result<GpuBatchTiming, ReplayError> {
        if self.ready_host_gate.is_some() || self.next_host_gate < self.graph.host_gate_count() {
            let cancel = self.graph.cancel_internal();
            self.completed = true;
            if let Err(error) = cancel {
                return Err(error);
            }
            return Err(ReplayError::InvalidBatchShape(
                "cannot wait for terminal completion before every host gate resumes",
            ));
        }
        let result = self.graph.wait_internal(timeout);
        self.completed = true;
        result
    }

    pub fn cancel(mut self) -> Result<(), ReplayError> {
        let result = self.graph.cancel_internal();
        self.completed = true;
        result
    }
}

impl Drop for SingleQueueBatchSubmission<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.graph.cancel_internal();
        }
    }
}

/// One globally synchronized phase for the measured two-queue sweet spot.
///
/// Dispatches within each lane are FIFO. The phase boundary is represented by
/// one tail completion signal per lane, not one signal/event per node.
pub struct TwoQueuePhase {
    lanes: [Vec<RecordedDispatch>; 2],
}

impl TwoQueuePhase {
    pub fn new(
        lane_zero: Vec<RecordedDispatch>,
        lane_one: Vec<RecordedDispatch>,
    ) -> Result<Self, ReplayError> {
        let lanes = [lane_zero, lane_one];
        for (lane, dispatches) in lanes.iter().enumerate() {
            if dispatches.is_empty() {
                return Err(ReplayError::EmptyPhaseLane { lane });
            }
            for dispatch in dispatches {
                if dispatch.lane != lane {
                    return Err(ReplayError::PhaseLaneMismatch {
                        expected: lane,
                        actual: dispatch.lane,
                    });
                }
                if !dispatch.dependencies.is_empty() {
                    return Err(ReplayError::PhaseHasExplicitDependencies { lane });
                }
            }
        }
        Ok(Self { lanes })
    }

    pub fn dispatch_count(&self, lane: usize) -> Option<usize> {
        self.lanes.get(lane).map(Vec::len)
    }
}

/// Specialized two-queue phase replay for the 4-root -> fan-in -> 4-child
/// token DAG shape.
///
/// For two phases with two roots and two children per queue, the emitted shape
/// is:
///
/// - queue 0: root, root(tail signal), one cross barrier, child,
///   child(tail signal), terminal fan-in;
/// - queue 1: root, root(tail signal), one cross barrier, child,
///   child(tail signal).
///
/// The complete queue-local sequence is filled and release-published before a
/// single doorbell write per queue (two doorbells per replay total).
///
/// This specialization uses Agent-release dispatch fences and fence-free
/// device dependency barriers. Every kernel and kernarg is validated against
/// the same `GpuDevice`; each following dispatch retains a System acquire, and
/// the final host-observed fan-in performs the System release.
pub struct TwoQueuePhasedGraph {
    queues: QueueSet,
    phases: Vec<TwoQueuePhase>,
    phase_tail_signals: Vec<[CompletionSignal; 2]>,
    final_signal: CompletionSignal,
    batches: [Vec<PacketImage>; 2],
    in_flight: bool,
    usable: bool,
}

impl fmt::Debug for TwoQueuePhasedGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TwoQueuePhasedGraph")
            .field("queues", &self.queues.queue_ids().collect::<Vec<_>>())
            .field("phase_count", &self.phases.len())
            .field(
                "batch_packets",
                &[self.batches[0].len(), self.batches[1].len()],
            )
            .field("in_flight", &self.in_flight)
            .field("usable", &self.usable)
            .finish()
    }
}

impl TwoQueuePhasedGraph {
    pub fn create(
        device: &GpuDevice,
        queue_size: u32,
        phases: Vec<TwoQueuePhase>,
    ) -> Result<Self, ReplayError> {
        if phases.is_empty() {
            return Err(ReplayError::EmptyPhaseSet);
        }
        for phase in &phases {
            for dispatches in &phase.lanes {
                for dispatch in dispatches {
                    if dispatch.kernel.agent() != device.raw_gpu_agent()
                        || dispatch.kernarg.agent() != device.raw_gpu_agent()
                    {
                        return Err(ReplayError::MixedGpuObjects);
                    }
                    device.validate_geometry(dispatch.geometry)?;
                }
            }
        }

        let queues = QueueSet::create(device, 2, queue_size)?;
        let mut phase_tail_signals = Vec::with_capacity(phases.len());
        for _ in 0..phases.len() {
            phase_tail_signals.push([
                CompletionSignal::new(device)?,
                CompletionSignal::new(device)?,
            ]);
        }
        let final_signal = CompletionSignal::new(device)?;
        let lane_counts = phases
            .iter()
            .map(|phase| [phase.lanes[0].len(), phase.lanes[1].len()])
            .collect::<Vec<_>>();
        let packet_counts = two_queue_batch_counts(&lane_counts);
        let mut batches = [
            Vec::with_capacity(packet_counts[0]),
            Vec::with_capacity(packet_counts[1]),
        ];

        for (phase_index, phase) in phases.iter().enumerate() {
            if phase_index != 0 {
                let previous = &phase_tail_signals[phase_index - 1];
                let dependencies = [previous[0].raw(), previous[1].raw()];
                for batch in &mut batches {
                    let barrier = BarrierAndPacket::new_two_queue_dependency(&dependencies)?;
                    batch.push(PacketImage::barrier(&barrier));
                }
            }
            for (lane, dispatches) in phase.lanes.iter().enumerate() {
                for (dispatch_index, dispatch) in dispatches.iter().enumerate() {
                    let is_tail = dispatch_index + 1 == dispatches.len();
                    let completion = if is_tail {
                        phase_tail_signals[phase_index][lane].raw()
                    } else {
                        abi::Signal(0)
                    };
                    let packet = KernelDispatchPacket::new_two_queue(
                        dispatch.kernel.metadata(),
                        dispatch.geometry,
                        dispatch.dynamic_group_bytes,
                        dispatch.kernarg.address(),
                        completion,
                    )?;
                    batches[lane].push(PacketImage::kernel(&packet));
                }
            }
        }

        let final_phase = phase_tail_signals.last().expect("phases is nonempty");
        let completion = BarrierAndPacket::new_two_queue_host_terminal(
            &[final_phase[0].raw(), final_phase[1].raw()],
            final_signal.raw(),
        )?;
        batches[0].push(PacketImage::barrier(&completion));
        for (lane, batch) in batches.iter().enumerate() {
            let capacity = queues.size(lane).expect("two queues were created") as usize;
            if batch.len() > capacity {
                return Err(ReplayError::BatchExceedsQueue {
                    lane,
                    packets: batch.len(),
                    capacity,
                });
            }
        }

        Ok(Self {
            queues,
            phases,
            phase_tail_signals,
            final_signal,
            batches,
            in_flight: false,
            usable: true,
        })
    }

    pub fn queue_ids(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.queues.queue_ids()
    }

    pub fn batch_packet_counts(&self) -> [usize; 2] {
        [self.batches[0].len(), self.batches[1].len()]
    }

    pub fn completion_signal_count(&self) -> usize {
        self.phase_tail_signals.len() * 2 + 1
    }

    pub fn doorbell_writes_per_replay(&self) -> usize {
        2
    }

    /// Submit one two-lane phase replay.
    ///
    /// # Safety
    ///
    /// Kernarg bytes may encode arbitrary device pointers. Every pointee must
    /// remain allocated, GPU-accessible, and free of incompatible host/agent
    /// mutation until an explicit `wait` or `cancel` returns `Ok`. After any
    /// `Err`, or after dropping a submission ticket, every pointee must instead
    /// remain valid through destruction of this graph. Only `Ok` proves
    /// quiescence.
    pub unsafe fn submit(&mut self) -> Result<TwoQueueSubmission<'_>, ReplayError> {
        if self.in_flight {
            return Err(ReplayError::AlreadyInFlight);
        }
        if !self.usable {
            return Err(ReplayError::GraphInactive);
        }
        for phase in &mut self.phase_tail_signals {
            for signal in phase {
                signal.reset();
            }
        }
        self.final_signal.reset();
        if let Err(error) = self.queues.prepare_batches(&self.batches) {
            self.usable = false;
            return Err(error.into());
        }
        self.in_flight = true;
        if let Err(error) = self.queues.ring_prepared() {
            self.in_flight = false;
            self.usable = false;
            return Err(error.into());
        }
        Ok(TwoQueueSubmission {
            graph: self,
            completed: false,
        })
    }

    /// Submit and wait using the default finite timeout.
    ///
    /// # Safety
    ///
    /// The pointer and access contract is identical to [`Self::submit`].
    pub unsafe fn replay_and_wait(&mut self) -> Result<(), ReplayError> {
        // SAFETY: forwarded from this method's caller.
        unsafe { self.submit()? }.wait()
    }

    fn wait_internal(&mut self, timeout: Duration) -> Result<(), ReplayError> {
        if !self.in_flight {
            return Ok(());
        }
        match self.queues.wait_signal(&self.final_signal, timeout) {
            Ok(()) => {
                apply_quiescence_transition(
                    &mut self.in_flight,
                    &mut self.usable,
                    QuiescenceTransition::Completed,
                );
                Ok(())
            }
            Err(wait_error) => match self.queues.inactivate_all() {
                Ok(()) => {
                    apply_quiescence_transition(
                        &mut self.in_flight,
                        &mut self.usable,
                        QuiescenceTransition::Cancelled,
                    );
                    Err(wait_error.into())
                }
                Err(teardown_error) => {
                    apply_quiescence_transition(
                        &mut self.in_flight,
                        &mut self.usable,
                        QuiescenceTransition::Failed,
                    );
                    Err(RuntimeError::OperationAndTeardown {
                        operation: Box::new(wait_error),
                        teardown: Box::new(teardown_error),
                    }
                    .into())
                }
            },
        }
    }

    fn cancel_internal(&mut self) -> Result<(), ReplayError> {
        if !self.in_flight {
            return Ok(());
        }
        match self.queues.inactivate_all() {
            Ok(()) => {
                apply_quiescence_transition(
                    &mut self.in_flight,
                    &mut self.usable,
                    QuiescenceTransition::Cancelled,
                );
                Ok(())
            }
            Err(error) => {
                apply_quiescence_transition(
                    &mut self.in_flight,
                    &mut self.usable,
                    QuiescenceTransition::Failed,
                );
                Err(error.into())
            }
        }
    }
}

impl Drop for TwoQueuePhasedGraph {
    fn drop(&mut self) {
        // Retained in-flight state makes this retry a prior failed ticket
        // cancellation before Queue Drop performs its destroy attempt.
        let _ = self.cancel_internal();
    }
}

#[must_use = "wait or cancel the AQL submission to observe completion/teardown errors"]
pub struct TwoQueueSubmission<'a> {
    graph: &'a mut TwoQueuePhasedGraph,
    completed: bool,
}

impl TwoQueueSubmission<'_> {
    /// Wait with the default finite host-polling timeout.
    ///
    /// `Ok` proves quiescence. On `Err`, the external-pointee contract from
    /// [`TwoQueuePhasedGraph::submit`] remains in force through graph
    /// destruction.
    pub fn wait(self) -> Result<(), ReplayError> {
        self.wait_timeout(DEFAULT_WAIT_TIMEOUT)
    }

    /// Wait with an explicit finite host-polling timeout.
    ///
    /// `Ok` proves quiescence. On `Err`, the external-pointee contract from
    /// [`TwoQueuePhasedGraph::submit`] remains in force through graph
    /// destruction.
    pub fn wait_timeout(mut self, timeout: Duration) -> Result<(), ReplayError> {
        let result = self.graph.wait_internal(timeout);
        self.completed = true;
        result
    }

    /// Abort pending execution. `Ok` proves quiescence; `Err` requires
    /// retaining external pointees through graph destruction.
    pub fn cancel(mut self) -> Result<(), ReplayError> {
        let result = self.graph.cancel_internal();
        self.completed = true;
        result
    }
}

impl Drop for TwoQueueSubmission<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.graph.cancel_internal();
        }
    }
}

/// A fixed batch of serialized two-queue tokens submitted with one doorbell
/// write per queue and one final completion wait.
///
/// Every token owns a distinct pair of tail signals for every phase. The first
/// phase of token `n + 1` waits on both final-phase tails from token `n`, so
/// tokens cannot overlap even though the entire batch is published at once.
/// Within each token, phase boundaries have the same all-lane fan-in semantics
/// as [`TwoQueuePhasedGraph`].
///
/// The reduced release scopes assume both queues and all dispatch resources
/// belong to one HSA GPU agent. [`Self::create`] enforces that through the same
/// mixed-agent validation used by the single-token specialization.
pub struct TwoQueueSerializedBatchGraph {
    queues: QueueSet,
    phases: Vec<TwoQueuePhase>,
    token_phase_tail_signals: Vec<Vec<[CompletionSignal; 2]>>,
    final_signal: CompletionSignal,
    batches: [Vec<PacketImage>; 2],
    token_count: usize,
    in_flight: bool,
    usable: bool,
    derived_profiling: Option<DerivedProfiling>,
}

struct DerivedProfiling {
    device: GpuDevice,
    signals: Vec<CompletionSignal>,
    mode: DerivedProfilingMode,
    frequency_hz: u64,
}

#[derive(Clone, Copy)]
enum DerivedProfilingMode {
    QueueEdges,
    AllDispatches,
}

/// Packet policies for generic two-queue phase lowering. The first token may
/// carry API-entry acquire fences; repeated tokens use device-only scopes.
pub(crate) struct TwoQueueDerivedPolicies {
    pub first_dispatches: Vec<[Vec<redline_rocr::HeaderPolicy>; 2]>,
    pub repeated_dispatches: Vec<[Vec<redline_rocr::HeaderPolicy>; 2]>,
    pub consolidation: redline_rocr::HeaderPolicy,
    pub dependency: redline_rocr::HeaderPolicy,
    pub terminal: redline_rocr::HeaderPolicy,
    pub profile_all_dispatches: bool,
}

impl fmt::Debug for TwoQueueSerializedBatchGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TwoQueueSerializedBatchGraph")
            .field("queues", &self.queues.queue_ids().collect::<Vec<_>>())
            .field("phase_count", &self.phases.len())
            .field("token_count", &self.token_count)
            .field(
                "batch_packets",
                &[self.batches[0].len(), self.batches[1].len()],
            )
            .field("in_flight", &self.in_flight)
            .field("usable", &self.usable)
            .finish()
    }
}

impl TwoQueueSerializedBatchGraph {
    /// Return the queue-local packet counts required to publish `token_count`
    /// serialized copies of `phases` as one batch.
    pub fn required_packet_counts(
        phases: &[TwoQueuePhase],
        token_count: usize,
    ) -> Result<[usize; 2], ReplayError> {
        if phases.is_empty() {
            return Err(ReplayError::EmptyPhaseSet);
        }
        let lane_counts = phases
            .iter()
            .map(|phase| [phase.lanes[0].len(), phase.lanes[1].len()])
            .collect::<Vec<_>>();
        two_queue_serialized_batch_counts(&lane_counts, token_count)
    }

    pub fn create(
        device: &GpuDevice,
        queue_size: u32,
        token_count: usize,
        phases: Vec<TwoQueuePhase>,
    ) -> Result<Self, ReplayError> {
        if phases.is_empty() {
            return Err(ReplayError::EmptyPhaseSet);
        }
        validate_two_queue_phases(device, &phases)?;
        let packet_counts = Self::required_packet_counts(&phases, token_count)?;
        let signal_count = two_queue_serialized_signal_count(phases.len(), token_count)?;

        let queues = QueueSet::create(device, 2, queue_size)?;
        let mut token_phase_tail_signals = Vec::with_capacity(token_count);
        for _ in 0..token_count {
            let mut phase_signals = Vec::with_capacity(phases.len());
            for _ in 0..phases.len() {
                phase_signals.push([
                    CompletionSignal::new(device)?,
                    CompletionSignal::new(device)?,
                ]);
            }
            token_phase_tail_signals.push(phase_signals);
        }
        debug_assert_eq!(
            token_phase_tail_signals
                .iter()
                .map(|token| token.len() * 2)
                .sum::<usize>()
                + 1,
            signal_count
        );
        let final_signal = CompletionSignal::new(device)?;
        let mut batches = [
            Vec::with_capacity(packet_counts[0]),
            Vec::with_capacity(packet_counts[1]),
        ];

        for token_index in 0..token_count {
            if token_index != 0 {
                let previous = token_phase_tail_signals[token_index - 1]
                    .last()
                    .expect("phases is nonempty");
                let dependencies = [previous[0].raw(), previous[1].raw()];
                for batch in &mut batches {
                    let barrier = BarrierAndPacket::new_two_queue_dependency(&dependencies)?;
                    batch.push(PacketImage::barrier(&barrier));
                }
            }

            for (phase_index, phase) in phases.iter().enumerate() {
                if phase_index != 0 {
                    let previous = &token_phase_tail_signals[token_index][phase_index - 1];
                    let dependencies = [previous[0].raw(), previous[1].raw()];
                    for batch in &mut batches {
                        let barrier = BarrierAndPacket::new_two_queue_dependency(&dependencies)?;
                        batch.push(PacketImage::barrier(&barrier));
                    }
                }
                for (lane, dispatches) in phase.lanes.iter().enumerate() {
                    for (dispatch_index, dispatch) in dispatches.iter().enumerate() {
                        let is_tail = dispatch_index + 1 == dispatches.len();
                        let completion = if is_tail {
                            token_phase_tail_signals[token_index][phase_index][lane].raw()
                        } else {
                            abi::Signal(0)
                        };
                        let packet = KernelDispatchPacket::new_two_queue(
                            dispatch.kernel.metadata(),
                            dispatch.geometry,
                            dispatch.dynamic_group_bytes,
                            dispatch.kernarg.address(),
                            completion,
                        )?;
                        batches[lane].push(PacketImage::kernel(&packet));
                    }
                }
            }
        }

        let final_phase = token_phase_tail_signals
            .last()
            .and_then(|token| token.last())
            .expect("tokens and phases are nonempty");
        let completion = BarrierAndPacket::new_two_queue_host_terminal(
            &[final_phase[0].raw(), final_phase[1].raw()],
            final_signal.raw(),
        )?;
        batches[0].push(PacketImage::barrier(&completion));
        debug_assert_eq!([batches[0].len(), batches[1].len()], packet_counts);
        for (lane, batch) in batches.iter().enumerate() {
            let capacity = queues.size(lane).expect("two queues were created") as usize;
            if batch.len() > capacity {
                return Err(ReplayError::BatchExceedsQueue {
                    lane,
                    packets: batch.len(),
                    capacity,
                });
            }
        }

        Ok(Self {
            queues,
            phases,
            token_phase_tail_signals,
            final_signal,
            batches,
            token_count,
            in_flight: false,
            usable: true,
            derived_profiling: None,
        })
    }

    /// Build the generic hazard-derived form. Unlike the historical
    /// specialization, each phase ends with a queue-local consolidation
    /// barrier: a tail dispatch signal cannot cover older barrier-free packets.
    pub(crate) fn create_derived(
        device: &GpuDevice,
        queue_size: u32,
        token_count: usize,
        phases: Vec<TwoQueuePhase>,
        policies: TwoQueueDerivedPolicies,
    ) -> Result<Self, ReplayError> {
        if phases.is_empty() {
            return Err(ReplayError::EmptyPhaseSet);
        }
        if token_count == 0 {
            return Err(ReplayError::EmptyTokenBatch);
        }
        validate_two_queue_phases(device, &phases)?;
        validate_derived_policy_shape(&phases, &policies)?;
        let base_counts = Self::required_packet_counts(&phases, token_count)?;
        let extra_consolidations = phases
            .len()
            .checked_mul(token_count)
            .ok_or(ReplayError::BatchShapeOverflow)?;
        let packet_counts = [
            base_counts[0]
                .checked_add(extra_consolidations)
                .ok_or(ReplayError::BatchShapeOverflow)?,
            base_counts[1]
                .checked_add(extra_consolidations)
                .ok_or(ReplayError::BatchShapeOverflow)?,
        ];
        let signal_count = two_queue_serialized_signal_count(phases.len(), token_count)?;

        let covers_batch = policies
            .repeated_dispatches
            .last()
            .is_some_and(|last_phase| {
                (0..2).all(|lane| last_phase[lane].last().is_some_and(|policy| policy.barrier))
            });
        let queues = QueueSet::create(device, 2, queue_size)?;
        let mut profiling_signals = Vec::new();
        let profiling_mode = if policies.profile_all_dispatches {
            Some(DerivedProfilingMode::AllDispatches)
        } else if covers_batch {
            Some(DerivedProfilingMode::QueueEdges)
        } else {
            None
        };
        let timestamp_frequency_hz = if let Some(mode) = profiling_mode {
            queues.set_profiling(true)?;
            let count = match mode {
                DerivedProfilingMode::QueueEdges => 4,
                DerivedProfilingMode::AllDispatches => phases
                    .iter()
                    .map(|phase| phase.lanes[0].len() + phase.lanes[1].len())
                    .sum::<usize>()
                    .checked_mul(token_count)
                    .ok_or(ReplayError::BatchShapeOverflow)?,
            };
            profiling_signals.reserve(count);
            for _ in 0..count {
                profiling_signals.push(CompletionSignal::new(device)?);
            }
            Some(device.timestamp_frequency_hz()?)
        } else {
            None
        };
        let mut token_phase_tail_signals = Vec::with_capacity(token_count);
        for _ in 0..token_count {
            let mut phase_signals = Vec::with_capacity(phases.len());
            for _ in 0..phases.len() {
                phase_signals.push([
                    CompletionSignal::new(device)?,
                    CompletionSignal::new(device)?,
                ]);
            }
            token_phase_tail_signals.push(phase_signals);
        }
        debug_assert_eq!(
            token_phase_tail_signals
                .iter()
                .map(|token| token.len() * 2)
                .sum::<usize>()
                + 1,
            signal_count
        );
        let final_signal = CompletionSignal::new(device)?;
        let mut batches = [
            Vec::with_capacity(packet_counts[0]),
            Vec::with_capacity(packet_counts[1]),
        ];
        let mut full_profile_index = 0_usize;

        for token_index in 0..token_count {
            if token_index != 0 {
                let previous = token_phase_tail_signals[token_index - 1]
                    .last()
                    .expect("phases is nonempty");
                let dependencies = [previous[0].raw(), previous[1].raw()];
                for batch in &mut batches {
                    let barrier = BarrierAndPacket::new_with_policy(
                        &dependencies,
                        abi::Signal(0),
                        policies.dependency,
                    )?;
                    batch.push(PacketImage::barrier(&barrier));
                }
            }

            let dispatch_policies = if token_index == 0 {
                &policies.first_dispatches
            } else {
                &policies.repeated_dispatches
            };
            for (phase_index, phase) in phases.iter().enumerate() {
                if phase_index != 0 {
                    let previous = &token_phase_tail_signals[token_index][phase_index - 1];
                    let dependencies = [previous[0].raw(), previous[1].raw()];
                    for batch in &mut batches {
                        let barrier = BarrierAndPacket::new_with_policy(
                            &dependencies,
                            abi::Signal(0),
                            policies.dependency,
                        )?;
                        batch.push(PacketImage::barrier(&barrier));
                    }
                }
                for (lane, dispatches) in phase.lanes.iter().enumerate() {
                    for (dispatch_index, dispatch) in dispatches.iter().enumerate() {
                        let is_first_profile =
                            token_index == 0 && phase_index == 0 && dispatch_index == 0;
                        let is_last_profile = token_index + 1 == token_count
                            && phase_index + 1 == phases.len()
                            && dispatch_index + 1 == dispatches.len();
                        if is_first_profile && is_last_profile {
                            return Err(ReplayError::InvalidBatchShape(
                                "derived profiling needs distinct first and last dispatches",
                            ));
                        }
                        let completion_signal = if policies.profile_all_dispatches {
                            let signal = profiling_signals[full_profile_index].raw();
                            full_profile_index += 1;
                            signal
                        } else if covers_batch && is_first_profile {
                            profiling_signals[lane].raw()
                        } else if covers_batch && is_last_profile {
                            profiling_signals[2 + lane].raw()
                        } else {
                            abi::Signal(0)
                        };
                        let packet = KernelDispatchPacket::new_with_policy(
                            dispatch.kernel.metadata(),
                            dispatch.geometry,
                            dispatch.dynamic_group_bytes,
                            dispatch.kernarg.address(),
                            completion_signal,
                            dispatch_policies[phase_index][lane][dispatch_index],
                        )?;
                        batches[lane].push(PacketImage::kernel(&packet));
                    }
                    let consolidation = BarrierAndPacket::new_with_policy(
                        &[],
                        token_phase_tail_signals[token_index][phase_index][lane].raw(),
                        policies.consolidation,
                    )?;
                    batches[lane].push(PacketImage::barrier(&consolidation));
                }
            }
        }

        let final_phase = token_phase_tail_signals
            .last()
            .and_then(|token| token.last())
            .expect("tokens and phases are nonempty");
        if policies.profile_all_dispatches {
            debug_assert_eq!(full_profile_index, profiling_signals.len());
        }
        let completion = BarrierAndPacket::new_with_policy(
            &[final_phase[0].raw(), final_phase[1].raw()],
            final_signal.raw(),
            policies.terminal,
        )?;
        batches[0].push(PacketImage::barrier(&completion));
        debug_assert_eq!([batches[0].len(), batches[1].len()], packet_counts);
        for (lane, batch) in batches.iter().enumerate() {
            let capacity = queues.size(lane).expect("two queues were created") as usize;
            if batch.len() > capacity {
                return Err(ReplayError::BatchExceedsQueue {
                    lane,
                    packets: batch.len(),
                    capacity,
                });
            }
        }

        Ok(Self {
            queues,
            phases,
            token_phase_tail_signals,
            final_signal,
            batches,
            token_count,
            in_flight: false,
            usable: true,
            derived_profiling: timestamp_frequency_hz.map(|frequency_hz| DerivedProfiling {
                device: device.clone(),
                signals: profiling_signals,
                mode: profiling_mode.expect("frequency implies profiling mode"),
                frequency_hz,
            }),
        })
    }

    pub fn queue_ids(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.queues.queue_ids()
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Patch stable kernarg allocations while no packet batch is in flight.
    /// Packet images retain the same kernarg addresses, so replay-time scalar
    /// values and resource pointers do not require packet reconstruction.
    pub(crate) fn patch_kernargs<E>(
        &mut self,
        mut patch: impl FnMut(usize, usize, usize, &mut KernargBuffer) -> Result<(), E>,
    ) -> Result<(), E> {
        assert!(
            !self.in_flight,
            "cannot patch kernargs while AQL work is in flight"
        );
        for (phase_index, phase) in self.phases.iter_mut().enumerate() {
            for (lane, dispatches) in phase.lanes.iter_mut().enumerate() {
                for (dispatch_index, dispatch) in dispatches.iter_mut().enumerate() {
                    patch(phase_index, lane, dispatch_index, dispatch.kernarg_mut())?;
                }
            }
        }
        Ok(())
    }

    pub fn batch_packet_counts(&self) -> [usize; 2] {
        [self.batches[0].len(), self.batches[1].len()]
    }

    pub fn completion_signal_count(&self) -> usize {
        self.token_phase_tail_signals
            .iter()
            .map(|token| token.len() * 2)
            .sum::<usize>()
            + 1
    }

    pub fn doorbell_writes_per_replay(&self) -> usize {
        2
    }

    pub(crate) fn gpu_batch_timing(&self) -> Result<GpuMultiQueueTiming, ReplayError> {
        if self.in_flight {
            return Err(ReplayError::AlreadyInFlight);
        }
        let profiling = self
            .derived_profiling
            .as_ref()
            .ok_or(ReplayError::ProfilingUnavailable)?;
        let mut first_start = u64::MAX;
        let mut last_end = 0_u64;
        match profiling.mode {
            DerivedProfilingMode::QueueEdges => {
                for signal in &profiling.signals[..2] {
                    first_start = first_start.min(profiling.device.dispatch_time(signal)?.start);
                }
                for signal in &profiling.signals[2..] {
                    last_end = last_end.max(profiling.device.dispatch_time(signal)?.end);
                }
            }
            DerivedProfilingMode::AllDispatches => {
                for signal in &profiling.signals {
                    let time = profiling.device.dispatch_time(signal)?;
                    first_start = first_start.min(time.start);
                    last_end = last_end.max(time.end);
                }
            }
        }
        Ok(GpuMultiQueueTiming {
            first_start,
            last_end,
            frequency_hz: profiling.frequency_hz,
        })
    }

    /// Submit the complete serialized token batch.
    ///
    /// # Safety
    ///
    /// Kernarg bytes may encode arbitrary device pointers. Every pointee must
    /// remain allocated, GPU-accessible, and free of incompatible host/agent
    /// mutation until an explicit `wait` or `cancel` returns `Ok`. After any
    /// `Err`, or after dropping a submission ticket, every pointee must instead
    /// remain valid through destruction of this graph. Only `Ok` proves
    /// quiescence.
    pub unsafe fn submit(&mut self) -> Result<TwoQueueBatchSubmission<'_>, ReplayError> {
        if self.in_flight {
            return Err(ReplayError::AlreadyInFlight);
        }
        if !self.usable {
            return Err(ReplayError::GraphInactive);
        }
        for token in &mut self.token_phase_tail_signals {
            for phase in token {
                for signal in phase {
                    signal.reset();
                }
            }
        }
        if let Some(profiling) = &mut self.derived_profiling {
            for signal in &mut profiling.signals {
                signal.reset();
            }
        }
        self.final_signal.reset();
        if let Err(error) = self.queues.prepare_batches(&self.batches) {
            self.usable = false;
            return Err(error.into());
        }
        self.in_flight = true;
        if let Err(error) = self.queues.ring_prepared() {
            self.in_flight = false;
            self.usable = false;
            return Err(error.into());
        }
        Ok(TwoQueueBatchSubmission {
            graph: self,
            completed: false,
        })
    }

    /// Submit and wait for the complete token batch using the default finite
    /// timeout.
    ///
    /// # Safety
    ///
    /// The pointer and access contract is identical to [`Self::submit`].
    pub unsafe fn replay_and_wait(&mut self) -> Result<(), ReplayError> {
        // SAFETY: forwarded from this method's caller.
        unsafe { self.submit()? }.wait()
    }

    fn wait_internal(&mut self, timeout: Duration) -> Result<(), ReplayError> {
        if !self.in_flight {
            return Ok(());
        }
        match self.queues.wait_signal(&self.final_signal, timeout) {
            Ok(()) => {
                apply_quiescence_transition(
                    &mut self.in_flight,
                    &mut self.usable,
                    QuiescenceTransition::Completed,
                );
                Ok(())
            }
            Err(wait_error) => match self.queues.inactivate_all() {
                Ok(()) => {
                    apply_quiescence_transition(
                        &mut self.in_flight,
                        &mut self.usable,
                        QuiescenceTransition::Cancelled,
                    );
                    Err(wait_error.into())
                }
                Err(teardown_error) => {
                    apply_quiescence_transition(
                        &mut self.in_flight,
                        &mut self.usable,
                        QuiescenceTransition::Failed,
                    );
                    Err(RuntimeError::OperationAndTeardown {
                        operation: Box::new(wait_error),
                        teardown: Box::new(teardown_error),
                    }
                    .into())
                }
            },
        }
    }

    fn wait_with_queue_depth_internal(
        &mut self,
        timeout: Duration,
    ) -> Result<QueueDepthReport, ReplayError> {
        if !self.in_flight {
            return Ok(QueueDepthReport::new(self.queues.queue_ids()));
        }
        match self
            .queues
            .wait_signal_with_queue_depth(&self.final_signal, timeout)
        {
            Ok(report) => {
                apply_quiescence_transition(
                    &mut self.in_flight,
                    &mut self.usable,
                    QuiescenceTransition::Completed,
                );
                Ok(report)
            }
            Err(wait_error) => match self.queues.inactivate_all() {
                Ok(()) => {
                    apply_quiescence_transition(
                        &mut self.in_flight,
                        &mut self.usable,
                        QuiescenceTransition::Cancelled,
                    );
                    Err(wait_error.into())
                }
                Err(teardown_error) => {
                    apply_quiescence_transition(
                        &mut self.in_flight,
                        &mut self.usable,
                        QuiescenceTransition::Failed,
                    );
                    Err(RuntimeError::OperationAndTeardown {
                        operation: Box::new(wait_error),
                        teardown: Box::new(teardown_error),
                    }
                    .into())
                }
            },
        }
    }

    fn cancel_internal(&mut self) -> Result<(), ReplayError> {
        if !self.in_flight {
            return Ok(());
        }
        match self.queues.inactivate_all() {
            Ok(()) => {
                apply_quiescence_transition(
                    &mut self.in_flight,
                    &mut self.usable,
                    QuiescenceTransition::Cancelled,
                );
                Ok(())
            }
            Err(error) => {
                apply_quiescence_transition(
                    &mut self.in_flight,
                    &mut self.usable,
                    QuiescenceTransition::Failed,
                );
                Err(error.into())
            }
        }
    }
}

impl Drop for TwoQueueSerializedBatchGraph {
    fn drop(&mut self) {
        let _ = self.cancel_internal();
    }
}

#[must_use = "wait or cancel the AQL batch submission to observe completion/teardown errors"]
pub struct TwoQueueBatchSubmission<'a> {
    graph: &'a mut TwoQueueSerializedBatchGraph,
    completed: bool,
}

impl TwoQueueBatchSubmission<'_> {
    pub fn wait(self) -> Result<(), ReplayError> {
        self.wait_timeout(DEFAULT_WAIT_TIMEOUT)
    }

    pub fn wait_timeout(mut self, timeout: Duration) -> Result<(), ReplayError> {
        let result = self.graph.wait_internal(timeout);
        self.completed = true;
        result
    }

    /// Wait while polling both queues' relaxed read/write indices.
    ///
    /// This is a diagnostic path and perturbs host polling. Ordinary `wait`
    /// and `wait_timeout` do not perform queue-index loads.
    pub fn wait_with_queue_depth(self) -> Result<QueueDepthReport, ReplayError> {
        self.wait_timeout_with_queue_depth(DEFAULT_WAIT_TIMEOUT)
    }

    /// Diagnostic queue-depth wait with an explicit finite timeout.
    pub fn wait_timeout_with_queue_depth(
        mut self,
        timeout: Duration,
    ) -> Result<QueueDepthReport, ReplayError> {
        let result = self.graph.wait_with_queue_depth_internal(timeout);
        self.completed = true;
        result
    }

    pub fn cancel(mut self) -> Result<(), ReplayError> {
        let result = self.graph.cancel_internal();
        self.completed = true;
        result
    }
}

impl Drop for TwoQueueBatchSubmission<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.graph.cancel_internal();
        }
    }
}

fn two_queue_batch_counts(phases: &[[usize; 2]]) -> [usize; 2] {
    if phases.is_empty() {
        return [0, 0];
    }
    let boundary_barriers = phases.len() - 1;
    let mut packets = [boundary_barriers + 1, boundary_barriers];
    for phase in phases {
        packets[0] += phase[0];
        packets[1] += phase[1];
    }
    packets
}

fn validate_derived_policy_shape(
    phases: &[TwoQueuePhase],
    policies: &TwoQueueDerivedPolicies,
) -> Result<(), ReplayError> {
    for (label, dispatches) in [
        ("first-token", &policies.first_dispatches),
        ("repeated-token", &policies.repeated_dispatches),
    ] {
        if dispatches.len() != phases.len() {
            return Err(ReplayError::PolicyShapeMismatch {
                detail: format!(
                    "{label} policy has {} phases, graph has {}",
                    dispatches.len(),
                    phases.len()
                ),
            });
        }
        for (phase_index, (phase, phase_policies)) in phases.iter().zip(dispatches).enumerate() {
            for lane in 0..2 {
                if phase_policies[lane].len() != phase.lanes[lane].len() {
                    return Err(ReplayError::PolicyShapeMismatch {
                        detail: format!(
                            "{label} phase {phase_index} lane {lane} has {} policies, {} dispatches",
                            phase_policies[lane].len(),
                            phase.lanes[lane].len()
                        ),
                    });
                }
            }
        }
    }
    Ok(())
}

fn two_queue_serialized_batch_counts(
    phases: &[[usize; 2]],
    token_count: usize,
) -> Result<[usize; 2], ReplayError> {
    if token_count == 0 {
        return Err(ReplayError::EmptyTokenBatch);
    }
    if phases.is_empty() {
        return Err(ReplayError::EmptyPhaseSet);
    }
    let phase_barriers = (phases.len() - 1)
        .checked_mul(token_count)
        .ok_or(ReplayError::BatchShapeOverflow)?;
    let token_barriers = token_count - 1;
    let mut packets = [0_usize; 2];
    for lane in 0..2 {
        let dispatches_per_token = phases.iter().try_fold(0_usize, |sum, phase| {
            sum.checked_add(phase[lane])
                .ok_or(ReplayError::BatchShapeOverflow)
        })?;
        packets[lane] = dispatches_per_token
            .checked_mul(token_count)
            .and_then(|count| count.checked_add(phase_barriers))
            .and_then(|count| count.checked_add(token_barriers))
            .and_then(|count| count.checked_add(usize::from(lane == 0)))
            .ok_or(ReplayError::BatchShapeOverflow)?;
    }
    Ok(packets)
}

fn two_queue_serialized_signal_count(
    phase_count: usize,
    token_count: usize,
) -> Result<usize, ReplayError> {
    if token_count == 0 {
        return Err(ReplayError::EmptyTokenBatch);
    }
    phase_count
        .checked_mul(token_count)
        .and_then(|count| count.checked_mul(2))
        .and_then(|count| count.checked_add(1))
        .ok_or(ReplayError::BatchShapeOverflow)
}

fn validate_two_queue_phases(
    device: &GpuDevice,
    phases: &[TwoQueuePhase],
) -> Result<(), ReplayError> {
    for phase in phases {
        for dispatches in &phase.lanes {
            for dispatch in dispatches {
                if dispatch.kernel.agent() != device.raw_gpu_agent()
                    || dispatch.kernarg.agent() != device.raw_gpu_agent()
                {
                    return Err(ReplayError::MixedGpuObjects);
                }
                device.validate_geometry(dispatch.geometry)?;
            }
        }
    }
    Ok(())
}

fn validate_nodes(
    device: &GpuDevice,
    queue_count: usize,
    nodes: &[RecordedDispatch],
) -> Result<(), ReplayError> {
    if queue_count == 0 {
        return Err(ReplayError::Runtime(RuntimeError::ZeroQueues));
    }
    for (index, node) in nodes.iter().enumerate() {
        if node.lane >= queue_count {
            return Err(ReplayError::LaneOutOfRange {
                node: index,
                lane: node.lane,
                queue_count,
            });
        }
        if node.kernel.agent() != device.raw_gpu_agent()
            || node.kernarg.agent() != device.raw_gpu_agent()
        {
            return Err(ReplayError::MixedGpuObjects);
        }
        device.validate_geometry(node.geometry)?;
        let mut unique = BTreeSet::new();
        for dependency in &node.dependencies {
            if *dependency >= index {
                return Err(ReplayError::DependencyNotEarlier {
                    node: index,
                    dependency: *dependency,
                });
            }
            if !unique.insert(*dependency) {
                return Err(ReplayError::DuplicateDependency {
                    node: index,
                    dependency: *dependency,
                });
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
pub enum ReplayError {
    Runtime(RuntimeError),
    Packet(PacketError),
    /// PM4 stream construction failed. Only reachable from the diagnostic
    /// per-dispatch timestamp path.
    Pm4Build(Pm4BuildError),
    EmptyGraph,
    EmptyPhaseSet,
    EmptyTokenBatch,
    InvalidBatchShape(&'static str),
    BatchShapeOverflow,
    EmptyPhaseLane {
        lane: usize,
    },
    PhaseLaneMismatch {
        expected: usize,
        actual: usize,
    },
    PhaseHasExplicitDependencies {
        lane: usize,
    },
    AlreadyInFlight,
    GraphInactive,
    LaneOutOfRange {
        node: usize,
        lane: usize,
        queue_count: usize,
    },
    DependencyNotEarlier {
        node: usize,
        dependency: usize,
    },
    DuplicateDependency {
        node: usize,
        dependency: usize,
    },
    BatchExceedsQueue {
        lane: usize,
        packets: usize,
        capacity: usize,
    },
    MixedGpuObjects,
    KernargMetadataMismatch {
        kernel: String,
        metadata_bytes: usize,
        buffer_bytes: usize,
    },
    KernargPatchWhileInFlight,
    KernargPatchOutOfBounds {
        dispatch: usize,
        offset: usize,
        bytes: usize,
        kernarg_bytes: usize,
    },
    PolicyShapeMismatch {
        detail: String,
    },
    ProfilingUnavailable,
    InvalidGpuTimestamp {
        start: u64,
        end: u64,
    },
    MalformedPm4IndirectBuffer {
        dword: usize,
    },
    Pm4DispatchOutOfBounds {
        dispatch: usize,
        dispatches: usize,
    },
    Pm4ZeroWorkgroup {
        dispatch: usize,
        workgroups: [u32; 3],
    },
    ProfilingDoesNotCoverBatch,
}

impl From<RuntimeError> for ReplayError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

impl From<PacketError> for ReplayError {
    fn from(value: PacketError) -> Self {
        Self::Packet(value)
    }
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(f),
            Self::Packet(error) => error.fmt(f),
            Self::Pm4Build(error) => error.fmt(f),
            Self::EmptyGraph => write!(f, "cannot record an empty AQL graph"),
            Self::EmptyPhaseSet => write!(f, "two-queue replay requires at least one phase"),
            Self::EmptyTokenBatch => {
                write!(f, "serialized two-queue replay requires at least one token")
            }
            Self::InvalidBatchShape(detail) => write!(f, "invalid AQL batch shape: {detail}"),
            Self::BatchShapeOverflow => {
                write!(f, "serialized two-queue packet or signal count overflowed")
            }
            Self::EmptyPhaseLane { lane } => {
                write!(f, "two-queue phase lane {lane} has no dispatches")
            }
            Self::PhaseLaneMismatch { expected, actual } => write!(
                f,
                "two-queue phase expected lane {expected}, dispatch uses lane {actual}"
            ),
            Self::PhaseHasExplicitDependencies { lane } => write!(
                f,
                "two-queue phase lane {lane} has per-node dependencies; use phase boundaries instead"
            ),
            Self::AlreadyInFlight => write!(f, "the previous AQL replay is still in flight"),
            Self::GraphInactive => write!(
                f,
                "the AQL graph was inactivated after cancellation, timeout, or queue fault"
            ),
            Self::LaneOutOfRange {
                node,
                lane,
                queue_count,
            } => write!(
                f,
                "node {node} uses lane {lane}, but only {queue_count} queues exist"
            ),
            Self::DependencyNotEarlier { node, dependency } => write!(
                f,
                "node {node} dependency {dependency} is not earlier in record order"
            ),
            Self::DuplicateDependency { node, dependency } => {
                write!(f, "node {node} repeats dependency {dependency}")
            }
            Self::BatchExceedsQueue {
                lane,
                packets,
                capacity,
            } => write!(
                f,
                "lane {lane} recorded batch has {packets} packets; one-doorbell queue capacity is {capacity}"
            ),
            Self::MixedGpuObjects => write!(
                f,
                "kernel, kernarg allocation, and queue device must use one HSA GPU agent"
            ),
            Self::KernargMetadataMismatch {
                kernel,
                metadata_bytes,
                buffer_bytes,
            } => write!(
                f,
                "kernel {kernel:?} metadata requires {metadata_bytes} kernarg bytes, buffer has {buffer_bytes}"
            ),
            Self::KernargPatchWhileInFlight => {
                write!(f, "cannot patch kernargs while an AQL replay is in flight")
            }
            Self::KernargPatchOutOfBounds {
                dispatch,
                offset,
                bytes,
                kernarg_bytes,
            } => write!(
                f,
                "dispatch {dispatch} kernarg patch {offset}..{} exceeds {kernarg_bytes} bytes",
                offset.saturating_add(*bytes)
            ),
            Self::PolicyShapeMismatch { detail } => {
                write!(f, "derived packet policy shape mismatch: {detail}")
            }
            Self::ProfilingUnavailable => write!(f, "AQL batch profiling is not enabled"),
            Self::InvalidGpuTimestamp { start, end } => write!(
                f,
                "retained PM4 GPU timestamp bracket is invalid: start={start}, end={end}"
            ),
            Self::MalformedPm4IndirectBuffer { dword } => write!(
                f,
                "retained PM4 indirect buffer is malformed at dword {dword}"
            ),
            Self::Pm4DispatchOutOfBounds {
                dispatch,
                dispatches,
            } => write!(
                f,
                "retained PM4 dispatch {dispatch} is outside the {dispatches}-dispatch indirect buffer"
            ),
            Self::Pm4ZeroWorkgroup {
                dispatch,
                workgroups,
            } => write!(
                f,
                "retained PM4 dispatch {dispatch} cannot use zero workgroups: {workgroups:?}"
            ),
            Self::ProfilingDoesNotCoverBatch => write!(
                f,
                "last dispatches are barrier-free, so first/last queue timestamps do not cover the batch"
            ),
        }
    }
}

impl std::error::Error for ReplayError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pm4_dispatch_offsets_follow_packet_boundaries() {
        let words = [
            0xc001_7602_u32,
            0x20c,
            0x1234,
            0xc003_1502,
            8,
            4,
            1,
            0x800d,
            0xc000_4600,
            0x407,
        ];
        let bytes = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(pm4_dispatch_workgroup_offsets(&bytes).unwrap(), vec![16]);
    }

    #[test]
    fn gpu_batch_span_uses_first_and_last_kernel_timestamps() {
        let timing = GpuBatchTiming {
            first_start: 100,
            first_end: 140,
            last_start: 900,
            last_end: 1_100,
            frequency_hz: 1_000_000,
        };
        assert_eq!(timing.span_microseconds(), 1_000.0);
        assert_eq!(timing.dispatch_span_microseconds(), 1_000.0);
    }

    #[test]
    fn arbitrary_fan_in_is_chunked_into_five_signal_packets() {
        let dependencies = (1_u64..=12).map(abi::Signal).collect::<Vec<_>>();
        let packets = dependencies
            .chunks(BARRIER_DEPENDENCY_CAPACITY)
            .map(|chunk| BarrierAndPacket::new(chunk, abi::Signal(0)).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(packets.len(), 3);
        assert_eq!(packets[0].dependency_signals[4], abi::Signal(5));
        assert_eq!(packets[2].dependency_signals[0], abi::Signal(11));
        assert_eq!(packets[2].dependency_signals[1], abi::Signal(12));
        assert_eq!(packets[2].dependency_signals[2], abi::Signal(0));
    }

    #[test]
    fn measured_two_phase_shape_is_six_and_five_packets_with_two_doorbells() {
        // Two roots per queue, one cross-queue phase barrier per queue, two
        // children per queue, and one final fan-in on queue zero.
        assert_eq!(two_queue_batch_counts(&[[2, 2], [2, 2]]), [6, 5]);
    }

    #[test]
    fn serialized_two_queue_batch_counts_match_one_and_one_hundred_tokens() {
        let phases = [[2, 2], [2, 2]];
        assert_eq!(
            two_queue_serialized_batch_counts(&phases, 1).unwrap(),
            [6, 5]
        );
        assert_eq!(
            two_queue_serialized_batch_counts(&phases, 100).unwrap(),
            [600, 599]
        );
        assert_eq!(two_queue_serialized_signal_count(2, 1).unwrap(), 5);
        assert_eq!(two_queue_serialized_signal_count(2, 100).unwrap(), 401);
    }

    #[test]
    fn serialized_two_queue_batch_rejects_empty_and_overflowing_shapes() {
        assert!(matches!(
            two_queue_serialized_batch_counts(&[[2, 2]], 0),
            Err(ReplayError::EmptyTokenBatch)
        ));
        assert!(matches!(
            two_queue_serialized_batch_counts(&[[usize::MAX, 1]], 2),
            Err(ReplayError::BatchShapeOverflow)
        ));
        assert!(matches!(
            two_queue_serialized_signal_count(usize::MAX, 2),
            Err(ReplayError::BatchShapeOverflow)
        ));
    }

    #[test]
    fn failed_quiescence_retains_in_flight_state_for_drop_retry() {
        let mut in_flight = true;
        let mut usable = true;
        apply_quiescence_transition(&mut in_flight, &mut usable, QuiescenceTransition::Failed);
        assert!(in_flight);
        assert!(!usable);
    }

    #[test]
    fn only_successful_completion_or_cancellation_clears_in_flight() {
        let mut in_flight = true;
        let mut usable = true;
        apply_quiescence_transition(&mut in_flight, &mut usable, QuiescenceTransition::Completed);
        assert!(!in_flight);
        assert!(usable);

        in_flight = true;
        apply_quiescence_transition(&mut in_flight, &mut usable, QuiescenceTransition::Cancelled);
        assert!(!in_flight);
        assert!(!usable);
    }

    #[test]
    fn host_gates_require_ordered_nonoverlapping_dispatch_cuts() {
        let valid = [
            HostGate {
                checkpoint_dispatch: 1,
                resume_before: 3,
            },
            HostGate {
                checkpoint_dispatch: 4,
                resume_before: 6,
            },
        ];
        validate_host_gates(8, &valid).unwrap();

        let reversed = [HostGate {
            checkpoint_dispatch: 3,
            resume_before: 3,
        }];
        assert!(matches!(
            validate_host_gates(8, &reversed),
            Err(ReplayError::PolicyShapeMismatch { .. })
        ));

        let overlapping = [
            HostGate {
                checkpoint_dispatch: 1,
                resume_before: 5,
            },
            HostGate {
                checkpoint_dispatch: 4,
                resume_before: 6,
            },
        ];
        assert!(matches!(
            validate_host_gates(8, &overlapping),
            Err(ReplayError::PolicyShapeMismatch { .. })
        ));
    }
}
