// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Narrow GFX10/GFX11 compute command construction for AMD's vendor AQL PM4-IB packet.
//!
//! This is intentionally separate from the GFX12 encoder: several compute
//! register offsets and the HSA implicit user-SGPR layout differ. GFX10 and
//! GFX11 share the legacy register map used here. The lowering supports
//! zero-scratch HSA kernels whose only implicit inputs are the optional
//! private-segment buffer and kernarg pointer. Static and dynamic LDS are
//! rounded to the hardware's 512-byte allocation granule. Other ABI contracts
//! fail closed.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt;

use crate::{Kernel, LaunchGeometry};

const PACKET3_SET_SH_REG: u32 = 0x76;
const PACKET3_DISPATCH_DIRECT: u32 = 0x15;
const PACKET3_WRITE_DATA: u32 = 0x37;
const PACKET3_COPY_DATA: u32 = 0x40;
const PACKET3_RELEASE_MEM: u32 = 0x49;
const PACKET3_EVENT_WRITE: u32 = 0x46;
const PACKET3_ACQUIRE_MEM: u32 = 0x58;
const PACKET3_WAIT_REG_MEM: u32 = 0x3c;

// GFX10/GFX11 SET_SH_REG offsets, matching ROCr's legacy compute command
// builder and Hipfire's independently exercised gfx1010 implementation.
const COMPUTE_NUM_THREAD_X: u32 = 0x207;
const COMPUTE_PGM_LO: u32 = 0x20c;
const COMPUTE_PGM_RSRC1: u32 = 0x212;
const COMPUTE_RESOURCE_LIMITS: u32 = 0x215;
const COMPUTE_TMPRING_SIZE_GFX10: u32 = 0x218;
const COMPUTE_PGM_RSRC3_GFX10: u32 = 0x228;
const COMPUTE_DISPATCH_INTERLEAVE: u32 = 0x22f;
const COMPUTE_USER_DATA_0: u32 = 0x240;

const LDS_SIZE_MASK: u32 = 0x00ff_8000;
const LDS_SIZE_SHIFT: u32 = 15;
const LEGACY_LDS_GRANULE: u32 = 512;

const ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER: u16 = 1 << 0;
const ENABLE_SGPR_DISPATCH_PTR: u16 = 1 << 1;
const ENABLE_SGPR_QUEUE_PTR: u16 = 1 << 2;
const ENABLE_SGPR_KERNARG_SEGMENT_PTR: u16 = 1 << 3;
const ENABLE_SGPR_DISPATCH_ID: u16 = 1 << 4;
const ENABLE_SGPR_FLAT_SCRATCH_INIT: u16 = 1 << 5;
const ENABLE_SGPR_PRIVATE_SEGMENT_SIZE: u16 = 1 << 6;
const ENABLE_WAVEFRONT_SIZE32: u16 = 1 << 10;
const IMPLICIT_SGPR_PROPERTIES: u16 = ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER
    | ENABLE_SGPR_DISPATCH_PTR
    | ENABLE_SGPR_QUEUE_PTR
    | ENABLE_SGPR_KERNARG_SEGMENT_PTR
    | ENABLE_SGPR_DISPATCH_ID
    | ENABLE_SGPR_FLAT_SCRATCH_INIT
    | ENABLE_SGPR_PRIVATE_SEGMENT_SIZE;
const SUPPORTED_KERNEL_PROPERTIES: u16 =
    ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER | ENABLE_SGPR_KERNARG_SEGMENT_PTR | ENABLE_WAVEFRONT_SIZE32;

/// Loader-resolved program state for the narrow GFX10/GFX11 PM4 lowering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gfx10KernelImage {
    pub code_entry: u64,
    pub compute_pgm_rsrc1: u32,
    pub compute_pgm_rsrc2: u32,
    pub compute_pgm_rsrc3: u32,
    pub group_segment_size: u32,
    pub private_segment_size: u32,
    pub dynamic_callstack: bool,
    pub wave32: bool,
    pub kernel_code_properties: u16,
}

impl Gfx10KernelImage {
    /// Adapt a public-HSA kernel while preserving its loader-resolved entry and
    /// descriptor resources.
    pub fn from_hsa(kernel: &Kernel) -> Result<Self, Gfx10Pm4BuildError> {
        let loader = kernel.metadata();
        let pm4 = kernel
            .pm4_metadata()
            .ok_or(Gfx10Pm4BuildError::MissingKernelDescriptor)?;
        let unsupported = pm4.kernel_code_properties & !SUPPORTED_KERNEL_PROPERTIES;
        if unsupported != 0 {
            return Err(Gfx10Pm4BuildError::UnsupportedKernelProperties(unsupported));
        }
        Ok(Self {
            code_entry: pm4.code_entry,
            compute_pgm_rsrc1: pm4.compute_pgm_rsrc1,
            compute_pgm_rsrc2: pm4.compute_pgm_rsrc2,
            compute_pgm_rsrc3: pm4.compute_pgm_rsrc3,
            group_segment_size: loader.group_segment_size,
            private_segment_size: loader.private_segment_size,
            dynamic_callstack: loader.dynamic_callstack,
            wave32: pm4.kernel_code_properties & ENABLE_WAVEFRONT_SIZE32 != 0,
            kernel_code_properties: pm4.kernel_code_properties,
        })
    }
}

/// `COMPUTE_DISPATCH_INITIATOR` policy for the shared GFX10/GFX11 register map.
///
/// `Legacy` preserves Redline's first validated encoding. The other policies
/// expose the architected `ORDER_MODE` and RADV-like `TUNNEL_ENABLE` bits as
/// explicit experiment arms instead of silently changing every legacy tape.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Gfx10DispatchInitiatorPolicy {
    #[default]
    Legacy,
    OrderMode,
    Radv,
}

impl Gfx10DispatchInitiatorPolicy {
    fn bits(self, wave32: bool) -> u32 {
        const COMPUTE_SHADER_EN: u32 = 1 << 0;
        const FORCE_START_AT_000: u32 = 1 << 2;
        const ORDERED_APPEND_ENBL: u32 = 1 << 3;
        const ORDER_MODE: u32 = 1 << 6;
        const TUNNEL_ENABLE: u32 = 1 << 13;
        const CS_W32_EN: u32 = 1 << 15;

        let policy = match self {
            Self::Legacy => ORDERED_APPEND_ENBL,
            Self::OrderMode => ORDER_MODE,
            Self::Radv => ORDER_MODE | TUNNEL_ENABLE,
        };
        COMPUTE_SHADER_EN | FORCE_START_AT_000 | policy | if wave32 { CS_W32_EN } else { 0 }
    }
}

/// Architected GFX11 `COMPUTE_DISPATCH_INTERLEAVE` values.
///
/// The register controls how many threads are sent to one shader engine before
/// dispatch advances to the next. `Disabled` writes zero; inheriting the queue
/// value is represented by `None` at the command-buffer API boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gfx11DispatchInterleave {
    Disabled,
    Threads64,
    Threads128,
    Threads256,
    Threads512,
}

impl Gfx11DispatchInterleave {
    pub const fn threads(self) -> u32 {
        match self {
            Self::Disabled => 0,
            Self::Threads64 => 64,
            Self::Threads128 => 128,
            Self::Threads256 => 256,
            Self::Threads512 => 512,
        }
    }
}

/// Per-dispatch `COMPUTE_RESOURCE_LIMITS` distribution policy.
///
/// Both policies leave the occupancy and threadgroup caps at zero. `Radv`
/// mirrors Mesa's topology-independent `SIMD_DEST_CNTL` rule and optionally
/// enables `FORCE_SIMD_DIST` for one-wave workgroups when the caller has
/// established that the target has a non-multiple-of-four CU count per SE.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Gfx11ComputeResourceLimitsPolicy {
    #[default]
    Legacy,
    SimdDestAlways,
    Radv {
        force_simd_dist_for_single_wave: bool,
    },
}

impl Gfx11ComputeResourceLimitsPolicy {
    fn bits(self, workgroup: [u16; 3], wave32: bool) -> u32 {
        const SIMD_DEST_CNTL: u32 = 1 << 22;
        const FORCE_SIMD_DIST: u32 = 1 << 23;

        let force_simd_dist_for_single_wave = match self {
            Self::Legacy => return 0,
            Self::SimdDestAlways => return SIMD_DEST_CNTL,
            Self::Radv {
                force_simd_dist_for_single_wave,
            } => force_simd_dist_for_single_wave,
        };
        let threads = workgroup.into_iter().map(u64::from).product::<u64>();
        let wave_size = if wave32 { 32 } else { 64 };
        let waves_per_threadgroup = threads.div_ceil(wave_size);
        let simd_dest = if waves_per_threadgroup.is_multiple_of(4) {
            SIMD_DEST_CNTL
        } else {
            0
        };
        let force_simd = if force_simd_dist_for_single_wave && waves_per_threadgroup == 1 {
            FORCE_SIMD_DIST
        } else {
            0
        };
        simd_dest | force_simd
    }
}

/// Retained GFX10/GFX11 PM4 words suitable for one vendor-AQL indirect buffer.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Gfx10Pm4CommandBuffer {
    dwords: Vec<u32>,
    register_state: Option<BTreeMap<u32, u32>>,
    dispatch_initiator_policy: Gfx10DispatchInitiatorPolicy,
    resource_limits_policy: Gfx11ComputeResourceLimitsPolicy,
}

/// GFX11 uses the same legacy compute-register map as GFX10. The replay API
/// still selects the device family explicitly before submission.
pub type Gfx11Pm4CommandBuffer = Gfx10Pm4CommandBuffer;

impl Gfx10Pm4CommandBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Elide repeated register values inside this retained command stream.
    /// The first value for every register is always emitted.
    pub fn new_stateful() -> Self {
        Self {
            dwords: Vec::new(),
            register_state: Some(BTreeMap::new()),
            dispatch_initiator_policy: Gfx10DispatchInitiatorPolicy::Legacy,
            resource_limits_policy: Gfx11ComputeResourceLimitsPolicy::Legacy,
        }
    }

    /// Select the initiator bits used by subsequently encoded dispatches.
    pub fn with_dispatch_initiator_policy(mut self, policy: Gfx10DispatchInitiatorPolicy) -> Self {
        self.dispatch_initiator_policy = policy;
        self
    }

    /// Select the resource-distribution bits used by subsequently encoded
    /// dispatches. Occupancy limits remain unlimited for every policy.
    pub fn with_resource_limits_policy(
        mut self,
        policy: Gfx11ComputeResourceLimitsPolicy,
    ) -> Self {
        self.resource_limits_policy = policy;
        self
    }

    /// Program a GFX11 dispatch-interleave queue preamble when requested.
    /// `None` preserves the queue/firmware value byte-for-byte.
    pub fn with_dispatch_interleave(
        mut self,
        interleave: Option<Gfx11DispatchInterleave>,
    ) -> Self {
        if let Some(interleave) = interleave {
            self.set_sh_regs(COMPUTE_DISPATCH_INTERLEAVE, &[interleave.threads()]);
        }
        self
    }

    /// Invalidate agent caches at the ROCr/AQL-to-PM4 ownership boundary.
    /// This is ROCr's gfx10+ `ACQUIRE_MEM` command shape.
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

    /// Same-agent shader-write to shader-read/write boundary for an unknown or
    /// scalar-reading consumer.
    pub fn dependency_rmw_same_agent(&mut self) {
        self.wait_compute_idle();
        self.acquire_inter_node_same_agent();
    }

    /// Invalidate the legacy scalar/vector read caches after the caller has
    /// already emitted the required dependency wait. Keeping the acquire and
    /// wait independently expressible lets the replay planner avoid duplicate
    /// CS_PARTIAL_FLUSH packets at real RMW boundaries.
    pub fn acquire_inter_node_same_agent(&mut self) {
        self.emit_acquire_radv(0x00380);
    }

    /// Same-agent boundary for a Radiowave-certified VMEM-only consumer.
    pub fn dependency_rmw_vmem(&mut self) {
        self.wait_compute_idle();
        self.acquire_inter_node_vmem();
    }

    /// Invalidate only vector/global read caches for a consumer whose loaded
    /// code object is Radiowave-certified to have no mutable scalar reads.
    pub fn acquire_inter_node_vmem(&mut self) {
        self.emit_acquire_radv(0x00300);
    }

    /// Historical conservative boundary with global L2 writeback/invalidate.
    pub fn dependency_rmw_global(&mut self) {
        self.wait_compute_idle();
        self.emit_acquire_radv(0x0c380);
    }

    fn emit_acquire_radv(&mut self, gcr_cntl: u32) {
        self.dwords.extend_from_slice(&[
            packet3(PACKET3_ACQUIRE_MEM, 7, false),
            0,
            u32::MAX,
            0x00ff_ffff,
            0,
            0,
            10,
            gcr_cntl,
        ]);
    }

    /// Return a copy bracketed by the same gfx9+ GPU-clock commands used by
    /// the GFX12 path.
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
    /// This is Mesa's compute-ring `RELEASE_MEM` fence shape: bottom-of-pipe,
    /// memory destination, immediate 32-bit data, and write confirmation.
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
    /// present in this stream. `WRITE_DATA` avoids a second bottom-of-pipe
    /// event while retaining write confirmation before a peer polls the word.
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
    /// `value`. Other queues remain schedulable, which permits a retained
    /// multi-queue graph to synchronize without per-phase AQL signals.
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

    /// Append one HSA-ABI dispatch, explicitly materializing gfx10's enabled
    /// implicit user SGPRs.
    pub fn dispatch(
        &mut self,
        kernel: &Kernel,
        geometry: LaunchGeometry,
        dynamic_group_bytes: u32,
        kernarg_address: *mut c_void,
    ) -> Result<(), Gfx10Pm4BuildError> {
        let image = Gfx10KernelImage::from_hsa(kernel)?;
        let user_sgprs = hsa_user_sgprs(image.kernel_code_properties, kernarg_address)?;
        self.dispatch_image(&image, geometry, dynamic_group_bytes, &user_sgprs)
    }

    /// Append one GFX10/GFX11 dispatch using work-group counts, matching the legacy
    /// compute command format used by ROCr and the working Hipfire path.
    pub fn dispatch_image(
        &mut self,
        image: &Gfx10KernelImage,
        geometry: LaunchGeometry,
        dynamic_group_bytes: u32,
        user_sgprs: &[u32],
    ) -> Result<(), Gfx10Pm4BuildError> {
        if image.private_segment_size != 0 || image.dynamic_callstack {
            return Err(Gfx10Pm4BuildError::ScratchUnsupported {
                private_bytes: image.private_segment_size,
                dynamic_callstack: image.dynamic_callstack,
            });
        }
        let total_group_bytes = image
            .group_segment_size
            .checked_add(dynamic_group_bytes)
            .ok_or(Gfx10Pm4BuildError::GroupSegmentOverflow)?;
        let lds_blocks = total_group_bytes.div_ceil(LEGACY_LDS_GRANULE);
        if lds_blocks > LDS_SIZE_MASK >> LDS_SIZE_SHIFT {
            return Err(Gfx10Pm4BuildError::GroupSegmentTooLarge(total_group_bytes));
        }
        let rsrc2 = (image.compute_pgm_rsrc2 & !LDS_SIZE_MASK) | (lds_blocks << LDS_SIZE_SHIFT);
        if image.code_entry == 0 || image.code_entry & 0xff != 0 {
            return Err(Gfx10Pm4BuildError::InvalidCodeEntry(image.code_entry));
        }
        if user_sgprs.len() > 16 {
            return Err(Gfx10Pm4BuildError::TooManyUserSgprs(user_sgprs.len()));
        }

        let mut workgroups = [0_u32; 3];
        for (axis, count) in workgroups.iter_mut().enumerate() {
            let workgroup = u32::from(geometry.workgroup[axis]);
            if !geometry.grid_workitems[axis].is_multiple_of(workgroup) {
                return Err(Gfx10Pm4BuildError::PartialWorkgroupUnsupported { axis });
            }
            *count = geometry.grid_workitems[axis] / workgroup;
        }

        self.set_sh_regs(
            COMPUTE_PGM_LO,
            &[
                (image.code_entry >> 8) as u32,
                (image.code_entry >> 40) as u32,
            ],
        );
        self.set_sh_regs(COMPUTE_PGM_RSRC1, &[image.compute_pgm_rsrc1, rsrc2]);
        self.set_sh_regs(COMPUTE_PGM_RSRC3_GFX10, &[image.compute_pgm_rsrc3]);
        self.set_sh_regs(COMPUTE_TMPRING_SIZE_GFX10, &[0]);
        self.set_sh_regs(
            COMPUTE_NUM_THREAD_X,
            &[
                u32::from(geometry.workgroup[0]),
                u32::from(geometry.workgroup[1]),
                u32::from(geometry.workgroup[2]),
            ],
        );
        let resource_limits = self
            .resource_limits_policy
            .bits(geometry.workgroup, image.wave32);
        self.set_sh_regs(COMPUTE_RESOURCE_LIMITS, &[resource_limits]);
        if !user_sgprs.is_empty() {
            self.set_sh_regs(COMPUTE_USER_DATA_0, user_sgprs);
        }

        let initiator = self.dispatch_initiator_policy.bits(image.wave32);
        self.dwords.push(packet3(PACKET3_DISPATCH_DIRECT, 4, true));
        self.dwords.extend_from_slice(&workgroups);
        self.dwords.push(initiator);
        Ok(())
    }

    /// Wait for all earlier compute waves before the enclosing vendor packet
    /// publishes completion.
    pub fn wait_compute_idle(&mut self) {
        self.dwords.push(packet3(PACKET3_EVENT_WRITE, 1, false));
        self.dwords.push(0x407); // CS_PARTIAL_FLUSH, event-index 4.
    }

    pub fn ends_with_compute_idle(&self) -> bool {
        self.dwords
            .ends_with(&[packet3(PACKET3_EVENT_WRITE, 1, false), 0x407])
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

fn hsa_user_sgprs(
    kernel_code_properties: u16,
    kernarg_address: *mut c_void,
) -> Result<Vec<u32>, Gfx10Pm4BuildError> {
    let unsupported = kernel_code_properties & !SUPPORTED_KERNEL_PROPERTIES;
    if unsupported != 0 {
        return Err(Gfx10Pm4BuildError::UnsupportedKernelProperties(unsupported));
    }
    let mut values = Vec::with_capacity(16);
    if kernel_code_properties & ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER != 0 {
        values.extend_from_slice(&[0; 4]);
    }
    if kernel_code_properties & ENABLE_SGPR_KERNARG_SEGMENT_PTR != 0 {
        if kernarg_address.is_null() {
            return Err(Gfx10Pm4BuildError::NullKernarg);
        }
        let address = kernarg_address as usize as u64;
        values.push(address as u32);
        values.push((address >> 32) as u32);
    }
    debug_assert_eq!(
        kernel_code_properties & IMPLICIT_SGPR_PROPERTIES,
        kernel_code_properties
            & (ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER | ENABLE_SGPR_KERNARG_SEGMENT_PTR)
    );
    Ok(values)
}

fn packet3(opcode: u32, body_dwords: u32, compute: bool) -> u32 {
    debug_assert!(body_dwords > 0);
    (3 << 30) | ((body_dwords - 1) << 16) | (opcode << 8) | if compute { 1 << 1 } else { 0 }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gfx10Pm4BuildError {
    MissingKernelDescriptor,
    InvalidCodeEntry(u64),
    ScratchUnsupported {
        private_bytes: u32,
        dynamic_callstack: bool,
    },
    UnsupportedKernelProperties(u16),
    NullKernarg,
    TooManyUserSgprs(usize),
    PartialWorkgroupUnsupported {
        axis: usize,
    },
    GroupSegmentOverflow,
    GroupSegmentTooLarge(u32),
}

impl fmt::Display for Gfx10Pm4BuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingKernelDescriptor => {
                write!(formatter, "kernel descriptor PM4 metadata is unavailable")
            }
            Self::InvalidCodeEntry(address) => write!(
                formatter,
                "kernel code entry {address:#x} is null or not 256-byte aligned"
            ),
            Self::ScratchUnsupported {
                private_bytes,
                dynamic_callstack,
            } => write!(
                formatter,
                "GFX10/GFX11 PM4 dispatch does not yet support scratch (private={private_bytes}, dynamic_callstack={dynamic_callstack})"
            ),
            Self::UnsupportedKernelProperties(bits) => write!(
                formatter,
                "kernel requires unsupported GFX10/GFX11 implicit SGPR properties 0x{bits:04x}"
            ),
            Self::NullKernarg => write!(formatter, "kernel requires a non-null kernarg pointer"),
            Self::TooManyUserSgprs(count) => write!(
                formatter,
                "kernel image requests {count} user-SGPR dwords; GFX10/GFX11 expose at most 16"
            ),
            Self::PartialWorkgroupUnsupported { axis } => write!(
                formatter,
                "GFX10/GFX11 direct dispatch requires an integral grid on axis {axis}"
            ),
            Self::GroupSegmentOverflow => {
                write!(formatter, "static plus dynamic group segment overflowed")
            }
            Self::GroupSegmentTooLarge(bytes) => write!(
                formatter,
                "group segment size {bytes} cannot be encoded in GFX10/GFX11 COMPUTE_PGM_RSRC2"
            ),
        }
    }
}

impl std::error::Error for Gfx10Pm4BuildError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(properties: u16) -> Gfx10KernelImage {
        Gfx10KernelImage {
            code_entry: 0x1_0000,
            compute_pgm_rsrc1: 0x11,
            compute_pgm_rsrc2: 0x22,
            compute_pgm_rsrc3: 0x33,
            group_segment_size: 0,
            private_segment_size: 0,
            dynamic_callstack: false,
            wave32: true,
            kernel_code_properties: properties,
        }
    }

    #[test]
    fn private_buffer_precedes_kernarg_in_user_sgprs() {
        let values = hsa_user_sgprs(
            ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER | ENABLE_SGPR_KERNARG_SEGMENT_PTR,
            0x1234_5678_9abc_def0_u64 as usize as *mut c_void,
        )
        .unwrap();
        assert_eq!(values, [0, 0, 0, 0, 0x9abc_def0, 0x1234_5678]);
    }

    #[test]
    fn gfx10_register_offsets_and_dispatch_words_are_stable() {
        let geometry = LaunchGeometry::new([256, 1, 1], [256, 1, 1]).unwrap();
        let mut commands = Gfx10Pm4CommandBuffer::new();
        commands
            .dispatch_image(&image(0x409), geometry, 0, &[0, 0, 0, 0, 4, 5])
            .unwrap();
        assert!(commands.dwords().windows(2).any(|words| words[1] == 0x228));
        assert!(commands.dwords().windows(2).any(|words| words[1] == 0x218));
        let dispatch = commands
            .dwords()
            .iter()
            .position(|word| *word == packet3(PACKET3_DISPATCH_DIRECT, 4, true))
            .unwrap();
        assert_eq!(
            &commands.dwords()[dispatch + 1..dispatch + 5],
            &[1, 1, 1, 0x800d]
        );
    }

    #[test]
    fn gfx11_dispatch_initiator_experiment_arms_use_architected_bits() {
        let geometry = LaunchGeometry::new([256, 1, 1], [256, 1, 1]).unwrap();
        let dispatch_word = |policy| {
            let mut commands = Gfx10Pm4CommandBuffer::new().with_dispatch_initiator_policy(policy);
            commands
                .dispatch_image(&image(0x409), geometry, 0, &[0, 0, 0, 0, 4, 5])
                .unwrap();
            let dispatch = commands
                .dwords()
                .iter()
                .position(|word| *word == packet3(PACKET3_DISPATCH_DIRECT, 4, true))
                .unwrap();
            commands.dwords()[dispatch + 4]
        };

        assert_eq!(dispatch_word(Gfx10DispatchInitiatorPolicy::Legacy), 0x800d);
        assert_eq!(
            dispatch_word(Gfx10DispatchInitiatorPolicy::OrderMode),
            0x8045
        );
        assert_eq!(dispatch_word(Gfx10DispatchInitiatorPolicy::Radv), 0xa045);
    }

    #[test]
    fn gfx11_dispatch_interleave_uses_queue_preamble_register() {
        let inherited = Gfx10Pm4CommandBuffer::new().with_dispatch_interleave(None);
        assert!(inherited.is_empty());

        for (policy, threads) in [
            (Gfx11DispatchInterleave::Disabled, 0),
            (Gfx11DispatchInterleave::Threads64, 64),
            (Gfx11DispatchInterleave::Threads128, 128),
            (Gfx11DispatchInterleave::Threads256, 256),
            (Gfx11DispatchInterleave::Threads512, 512),
        ] {
            let commands =
                Gfx10Pm4CommandBuffer::new().with_dispatch_interleave(Some(policy));
            assert_eq!(
                commands.dwords(),
                &[
                    packet3(PACKET3_SET_SH_REG, 2, true),
                    COMPUTE_DISPATCH_INTERLEAVE,
                    threads,
                ]
            );
        }
    }

    #[test]
    fn gfx11_radv_resource_limits_follow_waves_per_threadgroup() {
        let resource_word = |block: u16, wave32: bool, force_single_wave: bool| {
            let geometry = LaunchGeometry::new([u32::from(block), 1, 1], [block, 1, 1]).unwrap();
            let mut kernel = image(0x409);
            kernel.wave32 = wave32;
            let mut commands = Gfx10Pm4CommandBuffer::new().with_resource_limits_policy(
                Gfx11ComputeResourceLimitsPolicy::Radv {
                    force_simd_dist_for_single_wave: force_single_wave,
                },
            );
            commands
                .dispatch_image(&kernel, geometry, 0, &[0, 0, 0, 0, 4, 5])
                .unwrap();
            let register = commands
                .dwords()
                .windows(2)
                .position(|words| words[1] == COMPUTE_RESOURCE_LIMITS)
                .unwrap();
            commands.dwords()[register + 2]
        };

        assert_eq!(resource_word(32, true, false), 0);
        assert_eq!(resource_word(64, true, false), 0);
        assert_eq!(resource_word(128, true, false), 1 << 22);
        assert_eq!(resource_word(256, true, false), 1 << 22);
        assert_eq!(resource_word(64, false, true), 1 << 23);
        assert_eq!(resource_word(256, false, false), 1 << 22);
        assert_eq!(
            Gfx11ComputeResourceLimitsPolicy::SimdDestAlways.bits([32, 1, 1], true),
            1 << 22
        );
    }

    #[test]
    fn unsupported_implicit_state_and_lds_encoding_are_checked() {
        assert_eq!(
            hsa_user_sgprs(ENABLE_SGPR_QUEUE_PTR, std::ptr::null_mut()),
            Err(Gfx10Pm4BuildError::UnsupportedKernelProperties(
                ENABLE_SGPR_QUEUE_PTR
            ))
        );
        let mut lds_image = image(0x409);
        lds_image.group_segment_size = 1024;
        let geometry = LaunchGeometry::new([1, 1, 1], [1, 1, 1]).unwrap();
        Gfx10Pm4CommandBuffer::new()
            .dispatch_image(&lds_image, geometry, 0, &[])
            .unwrap();
        let mut commands = Gfx10Pm4CommandBuffer::new();
        commands
            .dispatch_image(&lds_image, geometry, 512, &[])
            .unwrap();
        let rsrc = commands
            .dwords()
            .windows(4)
            .find(|words| words[1] == COMPUTE_PGM_RSRC1)
            .unwrap();
        assert_eq!(rsrc[3] & LDS_SIZE_MASK, 3 << LDS_SIZE_SHIFT);

        lds_image.group_segment_size = u32::MAX;
        assert_eq!(
            Gfx10Pm4CommandBuffer::new().dispatch_image(&lds_image, geometry, 4, &[]),
            Err(Gfx10Pm4BuildError::GroupSegmentOverflow)
        );
    }

    #[test]
    fn stateful_stream_elides_repeated_register_values() {
        let geometry = LaunchGeometry::new([1, 1, 1], [1, 1, 1]).unwrap();
        let image = image(0x409);
        let mut commands = Gfx10Pm4CommandBuffer::new_stateful();
        commands
            .dispatch_image(&image, geometry, 0, &[0, 0, 0, 0, 4, 5])
            .unwrap();
        let first = commands.len_dwords();
        commands
            .dispatch_image(&image, geometry, 0, &[0, 0, 0, 0, 4, 5])
            .unwrap();
        assert_eq!(commands.len_dwords() - first, 5);
    }

    #[test]
    fn radiowave_cache_classes_keep_distinct_legacy_gcr_bits() {
        let mut scalar = Gfx10Pm4CommandBuffer::new();
        scalar.acquire_inter_node_same_agent();
        let mut vmem = Gfx10Pm4CommandBuffer::new();
        vmem.acquire_inter_node_vmem();
        assert_eq!(scalar.dwords().last(), Some(&0x00380));
        assert_eq!(vmem.dwords().last(), Some(&0x00300));
    }

    #[test]
    fn native_queue_semaphore_packets_match_mesa_compute_encodings() {
        let address = 0x1234_5678_9abc_def0;
        let mut commands = Gfx10Pm4CommandBuffer::new();
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
    fn confirmed_write_reuses_an_explicit_compute_idle_boundary() {
        let address = 0x1234_5678_9abc_def0;
        let mut commands = Gfx10Pm4CommandBuffer::new();
        commands.wait_compute_idle();
        assert!(commands.ends_with_compute_idle());
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
    fn appended_stream_invalidates_stateful_register_cache() {
        let geometry = LaunchGeometry::new([1, 1, 1], [1, 1, 1]).unwrap();
        let image = image(0x409);
        let mut commands = Gfx10Pm4CommandBuffer::new_stateful();
        commands
            .dispatch_image(&image, geometry, 0, &[0, 0, 0, 0, 4, 5])
            .unwrap();
        let first = commands.len_dwords();
        commands.append_stream(&Gfx10Pm4CommandBuffer::new());
        commands
            .dispatch_image(&image, geometry, 0, &[0, 0, 0, 0, 4, 5])
            .unwrap();
        assert_eq!(commands.len_dwords(), first * 2);
    }
}
