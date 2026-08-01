//! GFX12 compute command construction for AMD's vendor-specific AQL PM4-IB packet.
//!
//! This is deliberately narrower than a general PM4 library. It lowers a
//! loader-resolved, zero-scratch HSA kernel into the register writes and
//! `DISPATCH_DIRECT` packet used by ROCr's own command builder. Unsupported
//! implicit-SGPR contracts fail closed instead of guessing queue internals.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt;

use crate::{Kernel, LaunchGeometry};

const PACKET3_SET_SH_REG: u32 = 0x76;
const PACKET3_DISPATCH_DIRECT: u32 = 0x15;
const PACKET3_COPY_DATA: u32 = 0x40;
const PACKET3_RELEASE_MEM: u32 = 0x49;
const PACKET3_EVENT_WRITE: u32 = 0x46;
const PACKET3_ACQUIRE_MEM: u32 = 0x58;

// GFX12 SET_SH_REG offsets. The gfx12 register headers number COMPUTE
// registers from regCOMPUTE_DISPATCH_INITIATOR=0x1ba0; SET_SH_REG retains the
// architectural 0x200 COMPUTE window used by ROCr's PM4 builders.
const COMPUTE_NUM_THREAD_X: u32 = 0x207;
const COMPUTE_PGM_LO: u32 = 0x20c;
const COMPUTE_PGM_RSRC1: u32 = 0x212;
const COMPUTE_RESOURCE_LIMITS: u32 = 0x215;
const COMPUTE_TMPRING_SIZE: u32 = 0x216;
const COMPUTE_PGM_RSRC3_GFX12: u32 = 0x223;
const COMPUTE_STATIC_THREAD_MGMT_SE0: u32 = 0x230;
const COMPUTE_USER_DATA_0: u32 = 0x240;

const LDS_SIZE_MASK: u32 = 0x00ff_8000;
const LDS_SIZE_SHIFT: u32 = 15;
const GFX12_LDS_GRANULE: u32 = 512;

const ENABLE_SGPR_KERNARG_SEGMENT_PTR: u16 = 1 << 3;
const ENABLE_WAVEFRONT_SIZE32: u16 = 1 << 10;
const SUPPORTED_KERNEL_PROPERTIES: u16 = ENABLE_SGPR_KERNARG_SEGMENT_PTR | ENABLE_WAVEFRONT_SIZE32;
const DISPATCH_INITIATOR_BASE: u32 = (1 << 0) | (1 << 2) | (1 << 5);
const DISPATCH_INITIATOR_CS_W32_EN: u32 = 1 << 15;

/// Retained GFX12 PM4 command words suitable for one PM4 indirect buffer.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Gfx12Pm4CommandBuffer {
    dwords: Vec<u32>,
    register_state: Option<BTreeMap<u32, u32>>,
    cache_dynamic_registers: bool,
}

impl Gfx12Pm4CommandBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a command buffer which omits writes to SH registers whose
    /// values are already live earlier in this same retained indirect buffer.
    /// The first write to every register is always emitted.
    pub fn new_stateful() -> Self {
        Self {
            dwords: Vec::new(),
            register_state: Some(BTreeMap::new()),
            cache_dynamic_registers: true,
        }
    }

    /// Retain only queue-global invariant register values. Program, resource,
    /// workgroup, user-data, and dispatch state are still written exactly as
    /// in the legacy encoder.
    pub fn new_static_stateful() -> Self {
        Self {
            dwords: Vec::new(),
            register_state: Some(BTreeMap::new()),
            cache_dynamic_registers: false,
        }
    }

    /// Invalidate the agent caches at the HIP/HSA-to-PM4 ownership boundary.
    /// Encoding matches ROCr's gfx10+ `AcquireMemTemplate`, which remains the
    /// command shape used on gfx12.
    pub fn acquire_system(&mut self) {
        self.dwords.extend_from_slice(&[
            packet3(PACKET3_ACQUIRE_MEM, 7, false),
            0,
            u32::MAX,
            0xff,
            0,
            0,
            4,
            (1 << 16)
                | (1 << 15)
                | (1 << 14)
                | (1 << 9)
                | (1 << 8)
                | (1 << 7)
                | (1 << 6)
                | (1 << 5)
                | (1 << 4)
                | 1,
        ]);
    }

    /// GFX12 ownership-boundary acquire derived from the gfx12 GCR fields.
    /// This preserves system-scope L2 writeback/invalidate plus instruction,
    /// scalar, and vector cache visibility without carrying removed gfx11
    /// GL1/metadata bits into the merged RDNA4 hierarchy.
    pub fn acquire_system_gfx12(&mut self) {
        self.emit_acquire_gcr(0x1c1d1);
    }

    /// Return a copy bracketed by GPU-clock writes. The end timestamp follows
    /// all earlier compute work; the start uses RADV's top-of-pipe COPY_DATA
    /// timestamp form.
    pub fn with_gpu_timestamps(&self, start_address: u64, end_address: u64) -> Self {
        let mut timed = Self::new();
        timed.copy_gpu_timestamp(start_address);
        timed.dwords.extend_from_slice(&self.dwords);
        timed.release_gpu_timestamp(end_address);
        timed
    }

    /// Return a copy with a GPU-clock write after every `DISPATCH_DIRECT`,
    /// plus one before the first, so `N` dispatches yield `N + 1` timestamps
    /// and `N` per-dispatch deltas at `base_address + slot * 8`.
    ///
    /// Diagnostic only, and deliberately not on any certified path:
    ///
    ///   * it changes the tape — dword count and sequence hash both move — so
    ///     an instrumented run can never satisfy a golden fixture identity;
    ///   * it is an observer effect. On the 0.8b route (338 dispatches, ~5 us
    ///     each) a sub-microsecond COPY_DATA per dispatch is a measurable
    ///     fraction of the thing being measured, and it grows the IB ~21%.
    ///
    /// It answers a question end-to-end throughput cannot: whether a deficit is
    /// spread evenly across dispatches (per-dispatch overhead) or concentrated
    /// in a few (a barrier or cache-release stall).
    pub fn with_per_dispatch_timestamps(&self, base_address: u64) -> Result<Self, Pm4BuildError> {
        let mut timed = Self::new();
        let mut slot = 0_u64;
        timed.copy_gpu_timestamp(base_address);
        slot += 1;

        let mut cursor = 0_usize;
        while cursor < self.dwords.len() {
            let header = self.dwords[cursor];
            if header >> 30 != 3 {
                return Err(Pm4BuildError::MalformedStream { dword: cursor });
            }
            let body = ((header >> 16) & 0x3fff) as usize + 1;
            let next = cursor
                .checked_add(1 + body)
                .filter(|end| *end <= self.dwords.len())
                .ok_or(Pm4BuildError::MalformedStream { dword: cursor })?;
            timed.dwords.extend_from_slice(&self.dwords[cursor..next]);
            if (header >> 8) & 0xff == PACKET3_DISPATCH_DIRECT {
                timed.copy_gpu_timestamp(base_address + slot * 8);
                slot += 1;
            }
            cursor = next;
        }
        Ok(timed)
    }

    /// Timestamp slots [`with_per_dispatch_timestamps`] would write: one per
    /// dispatch plus a leading baseline. Callers size the buffer with this.
    pub fn timestamp_slot_count(&self) -> Result<usize, Pm4BuildError> {
        let mut slots = 1_usize;
        let mut cursor = 0_usize;
        while cursor < self.dwords.len() {
            let header = self.dwords[cursor];
            if header >> 30 != 3 {
                return Err(Pm4BuildError::MalformedStream { dword: cursor });
            }
            let body = ((header >> 16) & 0x3fff) as usize + 1;
            let next = cursor
                .checked_add(1 + body)
                .filter(|end| *end <= self.dwords.len())
                .ok_or(Pm4BuildError::MalformedStream { dword: cursor })?;
            if (header >> 8) & 0xff == PACKET3_DISPATCH_DIRECT {
                slots += 1;
            }
            cursor = next;
        }
        Ok(slots)
    }

    /// Attribute preceding boundary commands to each dispatch span.
    ///
    /// Span *i* is every PM4 packet after timestamp *i* through dispatch *i*.
    /// The entry ownership acquire is therefore part of span 0 metadata and is
    /// never reported as a mid-tape `acquire_inter_node`.
    pub fn dispatch_span_attributions(
        &self,
    ) -> Result<Vec<Pm4DispatchSpanAttribution>, Pm4BuildError> {
        let mut attributions = Vec::new();
        let mut pending = Pm4DispatchSpanAttribution::default();
        let mut saw_dispatch = false;
        let mut cursor = 0_usize;
        while cursor < self.dwords.len() {
            let header = self.dwords[cursor];
            if header >> 30 != 3 {
                return Err(Pm4BuildError::MalformedStream { dword: cursor });
            }
            let body = ((header >> 16) & 0x3fff) as usize + 1;
            let next = cursor
                .checked_add(1 + body)
                .filter(|end| *end <= self.dwords.len())
                .ok_or(Pm4BuildError::MalformedStream { dword: cursor })?;
            match (header >> 8) & 0xff {
                PACKET3_DISPATCH_DIRECT => {
                    attributions.push(pending);
                    pending = Pm4DispatchSpanAttribution::default();
                    saw_dispatch = true;
                }
                PACKET3_EVENT_WRITE => pending.wait_compute_idle = true,
                PACKET3_ACQUIRE_MEM if !saw_dispatch => pending.entry_acquire = true,
                PACKET3_ACQUIRE_MEM => pending.acquire_inter_node = true,
                _ => {}
            }
            cursor = next;
        }
        Ok(attributions)
    }

    fn copy_gpu_timestamp(&mut self, address: u64) {
        const COPY_DATA_TIMESTAMP_TO_MEMORY_64: u32 = 9 | (5 << 8) | (1 << 16) | (1 << 20);
        self.dwords.extend_from_slice(&[
            packet3(PACKET3_COPY_DATA, 5, false),
            COPY_DATA_TIMESTAMP_TO_MEMORY_64,
            0,
            0,
            address as u32,
            (address >> 32) as u32,
        ]);
    }

    fn release_gpu_timestamp(&mut self, address: u64) {
        const BOTTOM_OF_PIPE_TS_EVENT: u32 = 40 | (5 << 8);
        const TIMESTAMP_AFTER_WRITE_CONFIRM: u32 = (3 << 24) | (3 << 29);
        self.dwords.extend_from_slice(&[
            packet3(PACKET3_RELEASE_MEM, 7, false),
            BOTTOM_OF_PIPE_TS_EVENT,
            TIMESTAMP_AFTER_WRITE_CONFIRM,
            address as u32,
            (address >> 32) as u32,
            0,
            0,
            0,
        ]);
    }

    /// Same-agent inter-node acquire for one retained gfx12 tape. Kernel code
    /// is immutable and L2/MALL remains coherent, so only scalar/vector read
    /// caches plus forward sequencing are invalidated.
    pub fn acquire_inter_node_gfx12(&mut self) {
        self.emit_acquire_gcr(0x10180);
    }

    fn emit_acquire_gcr(&mut self, gcr_cntl: u32) {
        self.dwords.extend_from_slice(&[
            packet3(PACKET3_ACQUIRE_MEM, 7, false),
            0,
            u32::MAX,
            0xff,
            0,
            0,
            4,
            gcr_cntl,
        ]);
    }

    /// Append one zero-scratch dispatch using the exact loaded code
    /// entry and descriptor resources reported by the HSA loader.
    pub fn dispatch(
        &mut self,
        kernel: &Kernel,
        geometry: LaunchGeometry,
        dynamic_group_bytes: u32,
        kernarg_address: *mut c_void,
    ) -> Result<(), Pm4BuildError> {
        let loader = kernel.metadata();
        if loader.private_segment_size != 0 || loader.dynamic_callstack {
            return Err(Pm4BuildError::ScratchUnsupported {
                private_bytes: loader.private_segment_size,
                dynamic_callstack: loader.dynamic_callstack,
            });
        }
        let pm4 = kernel
            .pm4_metadata()
            .ok_or(Pm4BuildError::MissingKernelDescriptor)?;
        let unsupported = pm4.kernel_code_properties & !SUPPORTED_KERNEL_PROPERTIES;
        if unsupported != 0 {
            return Err(Pm4BuildError::UnsupportedKernelProperties(unsupported));
        }
        let wave32 = pm4.kernel_code_properties & ENABLE_WAVEFRONT_SIZE32 != 0;
        let needs_kernarg = pm4.kernel_code_properties & ENABLE_SGPR_KERNARG_SEGMENT_PTR != 0;
        if needs_kernarg && kernarg_address.is_null() {
            return Err(Pm4BuildError::NullKernarg);
        }

        let total_group_bytes = loader
            .group_segment_size
            .checked_add(dynamic_group_bytes)
            .ok_or(Pm4BuildError::GroupSegmentOverflow)?;
        let lds_blocks = total_group_bytes.div_ceil(GFX12_LDS_GRANULE);
        if lds_blocks > LDS_SIZE_MASK >> LDS_SIZE_SHIFT {
            return Err(Pm4BuildError::GroupSegmentTooLarge(total_group_bytes));
        }
        let rsrc2 = (pm4.compute_pgm_rsrc2 & !LDS_SIZE_MASK) | (lds_blocks << LDS_SIZE_SHIFT);

        self.set_sh_regs(
            COMPUTE_PGM_LO,
            &[(pm4.code_entry >> 8) as u32, (pm4.code_entry >> 40) as u32],
        );
        self.set_sh_regs(COMPUTE_PGM_RSRC1, &[pm4.compute_pgm_rsrc1, rsrc2]);
        self.set_sh_regs(COMPUTE_PGM_RSRC3_GFX12, &[pm4.compute_pgm_rsrc3]);
        self.set_sh_regs(COMPUTE_TMPRING_SIZE, &[0]);
        self.set_sh_regs(
            COMPUTE_NUM_THREAD_X,
            &[
                u32::from(geometry.workgroup[0]),
                u32::from(geometry.workgroup[1]),
                u32::from(geometry.workgroup[2]),
            ],
        );
        // Match ROCr's direct-dispatch template: all waves per SH are allowed
        // and every shader engine remains eligible.
        self.set_sh_regs(COMPUTE_RESOURCE_LIMITS, &[0x3ff]);
        self.set_sh_regs(COMPUTE_STATIC_THREAD_MGMT_SE0, &[u32::MAX; 4]);
        if needs_kernarg {
            let address = kernarg_address as usize as u64;
            self.set_sh_regs(
                COMPUTE_USER_DATA_0,
                &[address as u32, (address >> 32) as u32],
            );
        }

        self.dwords.push(packet3(PACKET3_DISPATCH_DIRECT, 4, true));
        self.dwords.extend_from_slice(&geometry.grid_workitems);
        // COMPUTE_SHADER_EN | FORCE_START_AT_000 | USE_THREAD_DIMENSIONS,
        // with CS_W32_EN derived from the kernel descriptor. A mixed-wave
        // retained tape must never inherit this bit from the preceding node.
        self.dwords.push(dispatch_initiator(wave32));
        Ok(())
    }

    /// Wait until all earlier compute waves have finished before the PM4 IB
    /// itself completes and its enclosing AQL packet publishes its signal.
    pub fn wait_compute_idle(&mut self) {
        self.dwords.push(packet3(PACKET3_EVENT_WRITE, 1, false));
        self.dwords.push(0x407); // CS_PARTIAL_FLUSH, event-index 4.
    }

    pub fn len_dwords(&self) -> u32 {
        self.dwords.len() as u32
    }

    pub fn is_empty(&self) -> bool {
        self.dwords.is_empty()
    }

    pub fn as_bytes(&self) -> Vec<u8> {
        self.dwords
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }

    pub fn dwords(&self) -> &[u32] {
        &self.dwords
    }

    fn set_sh_regs(&mut self, first: u32, values: &[u32]) {
        debug_assert!(!values.is_empty());
        let static_registers = matches!(
            first,
            COMPUTE_TMPRING_SIZE | COMPUTE_RESOURCE_LIMITS | COMPUTE_STATIC_THREAD_MGMT_SE0
        );
        if !self.cache_dynamic_registers && !static_registers {
            self.emit_set_sh_regs(first, values);
            return;
        }
        let Some(register_state) = self.register_state.as_mut() else {
            self.emit_set_sh_regs(first, values);
            return;
        };

        let mut changed_runs = Vec::<(u32, Vec<u32>)>::new();
        let mut run_first = None;
        let mut run_values = Vec::new();
        for (offset, value) in values.iter().copied().enumerate() {
            let register = first + offset as u32;
            if register_state.get(&register).copied() == Some(value) {
                if let Some(run_first) = run_first.take() {
                    changed_runs.push((run_first, std::mem::take(&mut run_values)));
                }
                continue;
            }
            register_state.insert(register, value);
            run_first.get_or_insert(register);
            run_values.push(value);
        }
        if let Some(run_first) = run_first {
            changed_runs.push((run_first, run_values));
        }

        for (run_first, run_values) in changed_runs {
            self.emit_set_sh_regs(run_first, &run_values);
        }
    }

    fn emit_set_sh_regs(&mut self, first: u32, values: &[u32]) {
        self.dwords
            .push(packet3(PACKET3_SET_SH_REG, 1 + values.len() as u32, true));
        self.dwords.push(first);
        self.dwords.extend_from_slice(values);
    }
}

fn packet3(opcode: u32, body_dwords: u32, compute: bool) -> u32 {
    debug_assert!(body_dwords > 0);
    (3 << 30) | ((body_dwords - 1) << 16) | (opcode << 8) | if compute { 1 << 1 } else { 0 }
}

fn dispatch_initiator(wave32: bool) -> u32 {
    DISPATCH_INITIATOR_BASE
        | if wave32 {
            DISPATCH_INITIATOR_CS_W32_EN
        } else {
            0
        }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Pm4BuildError {
    MissingKernelDescriptor,
    ScratchUnsupported {
        private_bytes: u32,
        dynamic_callstack: bool,
    },
    UnsupportedKernelProperties(u16),
    NullKernarg,
    GroupSegmentOverflow,
    GroupSegmentTooLarge(u32),
    /// Packet walk hit a non-PACKET3 header or a length running past the end.
    /// Only reachable from the diagnostic timestamp path.
    MalformedStream {
        dword: usize,
    },
}

/// Boundary commands attributed to one per-dispatch timestamp span.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Pm4DispatchSpanAttribution {
    /// Ownership acquire emitted before the first dispatch (span 0 only).
    pub entry_acquire: bool,
    pub wait_compute_idle: bool,
    /// Mid-tape acquire between dispatches (never the entry acquire).
    pub acquire_inter_node: bool,
}

impl fmt::Display for Pm4BuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingKernelDescriptor => {
                write!(formatter, "kernel descriptor PM4 metadata is unavailable")
            }
            Self::ScratchUnsupported {
                private_bytes,
                dynamic_callstack,
            } => write!(
                formatter,
                "PM4 dispatch does not support scratch (private={private_bytes}, dynamic_callstack={dynamic_callstack})"
            ),
            Self::UnsupportedKernelProperties(bits) => write!(
                formatter,
                "kernel requires unsupported implicit SGPR properties 0x{bits:04x}"
            ),
            Self::NullKernarg => write!(formatter, "kernel requires a non-null kernarg pointer"),
            Self::GroupSegmentOverflow => {
                write!(formatter, "static plus dynamic group segment overflowed")
            }
            Self::GroupSegmentTooLarge(bytes) => write!(
                formatter,
                "group segment size {bytes} cannot be encoded in GFX12 COMPUTE_PGM_RSRC2"
            ),
            Self::MalformedStream { dword } => {
                write!(formatter, "malformed PM4 stream at dword {dword}")
            }
        }
    }
}

impl std::error::Error for Pm4BuildError {}

#[cfg(test)]
mod tests {

    /// Build a stream of `n` DISPATCH_DIRECT packets separated by a filler
    /// packet, so the walk has to respect packet lengths rather than scanning
    /// for opcodes.
    fn synthetic_stream(n: usize) -> Gfx12Pm4CommandBuffer {
        let mut buffer = Gfx12Pm4CommandBuffer::new();
        for _ in 0..n {
            // filler: SET_SH_REG-shaped, 2 body dwords
            buffer.dwords.push(packet3(PACKET3_SET_SH_REG, 2, false));
            buffer.dwords.extend_from_slice(&[0xdead, 0xbeef]);
            // dispatch: 4 body dwords, matching the real emitter
            buffer
                .dwords
                .push(packet3(PACKET3_DISPATCH_DIRECT, 4, true));
            buffer.dwords.extend_from_slice(&[1, 1, 1, 0]);
        }
        buffer
    }

    #[test]
    fn per_dispatch_timestamps_emit_one_slot_per_dispatch_plus_baseline() {
        for n in [1_usize, 3, 8] {
            let plain = synthetic_stream(n);
            assert_eq!(plain.timestamp_slot_count().unwrap(), n + 1);
            let timed = plain.with_per_dispatch_timestamps(0x1000).unwrap();

            // Every original dword survives, in order, as a subsequence.
            let mut it = timed.dwords().iter();
            for want in plain.dwords() {
                assert!(it.any(|got| got == want), "original dword {want:#x} lost");
            }

            // One COPY_DATA per slot, addresses stepping by 8 from the base.
            let copies: Vec<usize> = timed
                .dwords()
                .iter()
                .enumerate()
                .filter(|(_, word)| **word == packet3(PACKET3_COPY_DATA, 5, false))
                .map(|(index, _)| index)
                .collect();
            assert_eq!(copies.len(), n + 1, "expected {} timestamp writes", n + 1);
            for (slot, index) in copies.iter().enumerate() {
                let want = 0x1000_u64 + slot as u64 * 8;
                let lo = timed.dwords()[index + 4] as u64;
                let hi = timed.dwords()[index + 5] as u64;
                assert_eq!(lo | (hi << 32), want, "slot {slot} address");
            }
        }
    }

    #[test]
    fn per_dispatch_timestamps_reject_a_malformed_stream() {
        let mut bad = Gfx12Pm4CommandBuffer::new();
        bad.dwords.push(0x0000_0000); // not a PACKET3 header
        assert!(matches!(
            bad.with_per_dispatch_timestamps(0x1000),
            Err(Pm4BuildError::MalformedStream { dword: 0 })
        ));
        // A length running past the end must be rejected, not truncated.
        let mut overrun = Gfx12Pm4CommandBuffer::new();
        overrun
            .dwords
            .push(packet3(PACKET3_DISPATCH_DIRECT, 4, true));
        assert!(overrun.with_per_dispatch_timestamps(0x1000).is_err());
    }
    use super::*;

    /// Span i owns every packet after timestamp i through dispatch i.
    /// Entry acquire is span-0 metadata; mid-tape waits/acquires attach to the
    /// following dispatch — never rewritten as entry_acquire.
    #[test]
    fn dispatch_span_attribution_follows_generated_packet_order() {
        let mut plain = Gfx12Pm4CommandBuffer::new();
        plain.acquire_system_gfx12();
        // dispatch 0 — entry acquire already above
        plain.dwords.push(packet3(PACKET3_DISPATCH_DIRECT, 4, true));
        plain.dwords.extend_from_slice(&[1, 1, 1, 0]);
        // boundary before dispatch 1
        plain.wait_compute_idle();
        plain.acquire_inter_node_gfx12();
        plain.dwords.push(packet3(PACKET3_DISPATCH_DIRECT, 4, true));
        plain.dwords.extend_from_slice(&[2, 1, 1, 0]);
        // wait-only boundary before dispatch 2
        plain.wait_compute_idle();
        plain.dwords.push(packet3(PACKET3_DISPATCH_DIRECT, 4, true));
        plain.dwords.extend_from_slice(&[3, 1, 1, 0]);
        plain.wait_compute_idle();

        let attributions = plain.dispatch_span_attributions().unwrap();
        assert_eq!(attributions.len(), 3);
        assert_eq!(
            attributions[0],
            Pm4DispatchSpanAttribution {
                entry_acquire: true,
                wait_compute_idle: false,
                acquire_inter_node: false,
            }
        );
        assert_eq!(
            attributions[1],
            Pm4DispatchSpanAttribution {
                entry_acquire: false,
                wait_compute_idle: true,
                acquire_inter_node: true,
            }
        );
        assert_eq!(
            attributions[2],
            Pm4DispatchSpanAttribution {
                entry_acquire: false,
                wait_compute_idle: true,
                acquire_inter_node: false,
            }
        );

        // Timestamp walk must place the baseline before the entry acquire and
        // each subsequent stamp immediately after its DISPATCH_DIRECT so span 0
        // covers entry acquire → dispatch 0, span 1 covers wait+acquire →
        // dispatch 1, etc.
        let timed = plain.with_per_dispatch_timestamps(0x2000).unwrap();
        let copies: Vec<usize> = timed
            .dwords()
            .iter()
            .enumerate()
            .filter(|(_, word)| **word == packet3(PACKET3_COPY_DATA, 5, false))
            .map(|(index, _)| index)
            .collect();
        assert_eq!(copies.len(), 4);

        let opcode_at = |dword: usize| (timed.dwords()[dword] >> 8) & 0xff;
        // baseline timestamp is the first packet
        assert_eq!(copies[0], 0);
        // first real command after baseline is the entry ACQUIRE_MEM
        assert_eq!(opcode_at(copies[0] + 6), PACKET3_ACQUIRE_MEM);
        // each post-dispatch stamp sits immediately after DISPATCH_DIRECT body
        for &stamp in &copies[1..] {
            assert_eq!(opcode_at(stamp - 5), PACKET3_DISPATCH_DIRECT);
        }

        // Reconstruct span contents from timestamp indices and prove flags.
        let mut cursor = copies[0] + 6; // first dword after baseline timestamp packet
        let mut reconstructed = Vec::new();
        for &end in &copies[1..] {
            let mut attr = Pm4DispatchSpanAttribution::default();
            let mut saw_dispatch = false;
            while cursor < end {
                let header = timed.dwords()[cursor];
                let body = ((header >> 16) & 0x3fff) as usize + 1;
                let next = cursor + 1 + body;
                match (header >> 8) & 0xff {
                    PACKET3_DISPATCH_DIRECT => saw_dispatch = true,
                    PACKET3_EVENT_WRITE => attr.wait_compute_idle = true,
                    PACKET3_ACQUIRE_MEM if reconstructed.is_empty() && !saw_dispatch => {
                        attr.entry_acquire = true;
                    }
                    PACKET3_ACQUIRE_MEM => attr.acquire_inter_node = true,
                    _ => {}
                }
                cursor = next;
            }
            assert!(saw_dispatch, "each span must include its dispatch");
            // skip the timestamp packet itself
            cursor = end + 6;
            reconstructed.push(attr);
        }
        assert_eq!(reconstructed, attributions);
    }

    #[test]
    fn packet3_count_and_shader_type_match_gfx12_headers() {
        assert_eq!(packet3(PACKET3_SET_SH_REG, 3, true), 0xc002_7602);
        assert_eq!(packet3(PACKET3_DISPATCH_DIRECT, 4, true), 0xc003_1502);
        assert_eq!(packet3(PACKET3_EVENT_WRITE, 1, false), 0xc000_4600);
        assert_eq!(packet3(PACKET3_ACQUIRE_MEM, 7, false), 0xc006_5800);
    }

    #[test]
    fn dispatch_initiator_tracks_kernel_descriptor_wave_size() {
        assert_eq!(dispatch_initiator(false), 0x25);
        assert_eq!(dispatch_initiator(true), 0x8025);
    }

    #[test]
    fn acquire_and_compute_idle_have_stable_rocr_encodings() {
        let mut commands = Gfx12Pm4CommandBuffer::new();
        commands.acquire_system();
        commands.acquire_system_gfx12();
        commands.acquire_inter_node_gfx12();
        commands.wait_compute_idle();
        assert_eq!(commands.dwords()[0], 0xc006_5800);
        assert_eq!(commands.dwords()[7], 0x1c3f1);
        assert_eq!(commands.dwords()[8], 0xc006_5800);
        assert_eq!(commands.dwords()[15], 0x1c1d1);
        assert_eq!(commands.dwords()[16], 0xc006_5800);
        assert_eq!(commands.dwords()[23], 0x10180);
        assert_eq!(&commands.dwords()[24..], &[0xc000_4600, 0x407]);
    }

    #[test]
    fn stateful_register_writes_emit_only_changed_contiguous_runs() {
        let mut commands = Gfx12Pm4CommandBuffer::new_stateful();
        commands.set_sh_regs(0x210, &[1, 2, 3, 4]);
        let first_len = commands.len_dwords();
        commands.set_sh_regs(0x210, &[1, 2, 3, 4]);
        assert_eq!(commands.len_dwords(), first_len);

        commands.set_sh_regs(0x210, &[5, 2, 6, 4]);
        assert_eq!(
            &commands.dwords()[first_len as usize..],
            &[
                packet3(PACKET3_SET_SH_REG, 2, true),
                0x210,
                5,
                packet3(PACKET3_SET_SH_REG, 2, true),
                0x212,
                6,
            ]
        );
    }

    #[test]
    fn legacy_register_writes_remain_byte_stable() {
        let mut commands = Gfx12Pm4CommandBuffer::new();
        commands.set_sh_regs(0x210, &[1, 2]);
        let once = commands.dwords().to_vec();
        commands.set_sh_regs(0x210, &[1, 2]);
        assert_eq!(commands.dwords().len(), once.len() * 2);
        assert_eq!(&commands.dwords()[once.len()..], once);
    }

    #[test]
    fn static_stateful_caches_only_queue_global_registers() {
        let mut commands = Gfx12Pm4CommandBuffer::new_static_stateful();
        commands.set_sh_regs(COMPUTE_RESOURCE_LIMITS, &[0x3ff]);
        let static_len = commands.len_dwords();
        commands.set_sh_regs(COMPUTE_RESOURCE_LIMITS, &[0x3ff]);
        assert_eq!(commands.len_dwords(), static_len);

        commands.set_sh_regs(COMPUTE_PGM_LO, &[1, 2]);
        let dynamic_len = commands.len_dwords();
        commands.set_sh_regs(COMPUTE_PGM_LO, &[1, 2]);
        assert_eq!(
            commands.len_dwords() - dynamic_len,
            dynamic_len - static_len
        );
    }
}
