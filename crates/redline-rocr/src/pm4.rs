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
const PACKET3_WRITE_DATA: u32 = 0x37;
const PACKET3_WAIT_REG_MEM: u32 = 0x3c;
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

    /// Append a complete independently-built command stream.
    ///
    /// The appended stream materializes its own register state. Clear this
    /// builder's cache so any later dispatch also rematerializes state rather
    /// than relying on values that the appended stream may have changed.
    pub fn append_stream(&mut self, commands: &Self) {
        self.dwords.extend_from_slice(&commands.dwords);
        if let Some(register_state) = self.register_state.as_mut() {
            register_state.clear();
        }
    }

    /// Publish a 32-bit value only after all preceding compute work completes.
    /// GFX12 retains the gfx9+ compute-ring `RELEASE_MEM` packet contract used
    /// by ROCr and Mesa: bottom-of-pipe, memory destination, immediate 32-bit
    /// data, and write confirmation.
    pub fn release_memory_value(&mut self, address: u64, value: u32) {
        debug_assert_ne!(address, 0);
        debug_assert_eq!(address & 3, 0);
        const BOTTOM_OF_PIPE_TS_EVENT: u32 = 40 | (5 << 8);
        const VALUE_32_AFTER_WRITE_CONFIRM: u32 = (3 << 24) | (1 << 29);
        self.dwords.extend_from_slice(&[
            packet3(PACKET3_RELEASE_MEM, 7, false),
            BOTTOM_OF_PIPE_TS_EVENT,
            VALUE_32_AFTER_WRITE_CONFIRM,
            address as u32,
            (address >> 32) as u32,
            value,
            0,
            0,
        ]);
    }

    /// Publish a 32-bit value after an explicit compute-idle packet already
    /// present in this stream, avoiding a second bottom-of-pipe event.
    pub fn write_memory_value_after_idle(&mut self, address: u64, value: u32) {
        debug_assert_ne!(address, 0);
        debug_assert_eq!(address & 3, 0);
        debug_assert!(self.ends_with_compute_idle());
        const MEMORY_WRITE_CONFIRMED: u32 = (5 << 8) | (1 << 20);
        self.dwords.extend_from_slice(&[
            packet3(PACKET3_WRITE_DATA, 4, false),
            MEMORY_WRITE_CONFIRMED,
            address as u32,
            (address >> 32) as u32,
            value,
        ]);
    }

    /// Stall this queue's command processor until a GPU-visible word equals
    /// `value`. Peer queues remain schedulable while this queue is parked.
    pub fn wait_memory_value(&mut self, address: u64, value: u32) {
        debug_assert_ne!(address, 0);
        debug_assert_eq!(address & 3, 0);
        const MEMORY_SPACE_EQUAL: u32 = (1 << 4) | 3;
        self.dwords.extend_from_slice(&[
            packet3(PACKET3_WAIT_REG_MEM, 6, false),
            MEMORY_SPACE_EQUAL,
            address as u32,
            (address >> 32) as u32,
            value,
            u32::MAX,
            4,
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

    pub fn ends_with_compute_idle(&self) -> bool {
        self.dwords.ends_with(&[
            packet3(PACKET3_EVENT_WRITE, 1, false),
            0x407,
        ])
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
        }
    }
}

impl std::error::Error for Pm4BuildError {}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(commands.ends_with_compute_idle());
    }

    #[test]
    fn native_queue_semaphore_packets_match_gfx9_plus_compute_encodings() {
        let address = 0x1234_5678_9abc_def0;
        let mut commands = Gfx12Pm4CommandBuffer::new();
        commands.release_memory_value(address, 7);
        commands.wait_memory_value(address + 4, 7);
        assert_eq!(
            commands.dwords(),
            &[
                0xc006_4900,
                0x528,
                0x2300_0000,
                0x9abc_def0,
                0x1234_5678,
                7,
                0,
                0,
                0xc005_3c00,
                0x13,
                0x9abc_def4,
                0x1234_5678,
                7,
                u32::MAX,
                4,
            ]
        );
    }

    #[test]
    fn confirmed_write_reuses_gfx12_compute_idle_boundary() {
        let address = 0x1234_5678_9abc_def0;
        let mut commands = Gfx12Pm4CommandBuffer::new();
        commands.wait_compute_idle();
        commands.write_memory_value_after_idle(address, 9);
        assert_eq!(
            &commands.dwords()[2..],
            &[
                0xc003_3700,
                0x0010_0500,
                0x9abc_def0,
                0x1234_5678,
                9,
            ]
        );
        assert!(!commands.ends_with_compute_idle());
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
