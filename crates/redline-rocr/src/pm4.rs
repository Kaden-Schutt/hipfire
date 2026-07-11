//! GFX12 compute command construction for AMD's vendor-specific AQL PM4-IB packet.
//!
//! This is deliberately narrower than a general PM4 library. It lowers a
//! loader-resolved, zero-scratch HSA kernel into the register writes and
//! `DISPATCH_DIRECT` packet used by ROCr's own command builder. Unsupported
//! implicit-SGPR contracts fail closed instead of guessing queue internals.

use std::ffi::c_void;
use std::fmt;

use crate::{Kernel, LaunchGeometry};

const PACKET3_SET_SH_REG: u32 = 0x76;
const PACKET3_DISPATCH_DIRECT: u32 = 0x15;
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

/// Retained GFX12 PM4 command words suitable for one PM4 indirect buffer.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Gfx12Pm4CommandBuffer {
    dwords: Vec<u32>,
}

impl Gfx12Pm4CommandBuffer {
    pub fn new() -> Self {
        Self::default()
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

    /// Append one zero-scratch wave32 dispatch using the exact loaded code
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
        if pm4.kernel_code_properties & ENABLE_WAVEFRONT_SIZE32 == 0 {
            return Err(Pm4BuildError::Wave64Unsupported);
        }
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
        // COMPUTE_SHADER_EN | FORCE_START_AT_000 | USE_THREAD_DIMENSIONS |
        // CS_W32_EN. These are the bits ROCr programs when lowering an AQL
        // wave32 dispatch to PM4.
        self.dwords.push((1 << 0) | (1 << 2) | (1 << 5) | (1 << 15));
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Pm4BuildError {
    MissingKernelDescriptor,
    ScratchUnsupported {
        private_bytes: u32,
        dynamic_callstack: bool,
    },
    UnsupportedKernelProperties(u16),
    Wave64Unsupported,
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
            Self::Wave64Unsupported => {
                write!(formatter, "PM4 dispatch currently requires a wave32 kernel")
            }
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
    fn acquire_and_compute_idle_have_stable_rocr_encodings() {
        let mut commands = Gfx12Pm4CommandBuffer::new();
        commands.acquire_system();
        commands.wait_compute_idle();
        assert_eq!(commands.dwords()[0], 0xc006_5800);
        assert_eq!(commands.dwords()[7], 0x1c3f1);
        assert_eq!(&commands.dwords()[8..], &[0xc000_4600, 0x407]);
    }
}
