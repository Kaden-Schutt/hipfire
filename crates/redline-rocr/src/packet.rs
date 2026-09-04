//! Architected Queuing Language packet construction.

use std::ffi::c_void;
use std::fmt;

use super::abi;

pub const AQL_PACKET_BYTES: usize = 64;
pub const BARRIER_DEPENDENCY_CAPACITY: usize = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum FenceScope {
    None = abi::FENCE_SCOPE_NONE,
    Agent = abi::FENCE_SCOPE_AGENT,
    System = abi::FENCE_SCOPE_SYSTEM,
}

/// Header policy for a packet.
///
/// `barrier = true` makes the packet wait for all earlier packets in the same
/// queue to complete. Redline uses this for each recorded dispatch so a logical
/// lane has stream-like completion order while different HSA queues can run
/// concurrently.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeaderPolicy {
    pub barrier: bool,
    pub acquire: FenceScope,
    pub release: FenceScope,
}

impl HeaderPolicy {
    pub const RECORDED_DISPATCH: Self = Self {
        barrier: true,
        acquire: FenceScope::System,
        release: FenceScope::System,
    };

    pub const CROSS_QUEUE_BARRIER: Self = Self {
        barrier: true,
        acquire: FenceScope::System,
        release: FenceScope::System,
    };

    /// Specialized same-agent dispatch policy matching gfx12 HIP packets.
    pub const TWO_QUEUE_DISPATCH: Self = Self {
        barrier: true,
        acquire: FenceScope::System,
        release: FenceScope::Agent,
    };

    /// Same-agent continuation after a batch-entry System acquire.
    pub const SAME_AGENT_DISPATCH: Self = Self {
        barrier: true,
        acquire: FenceScope::Agent,
        release: FenceScope::Agent,
    };

    /// Device-only phase/token dependency. The following dispatch performs the
    /// System acquire before it consumes data produced by the dependency.
    pub const TWO_QUEUE_DEPENDENCY: Self = Self {
        barrier: true,
        acquire: FenceScope::None,
        release: FenceScope::None,
    };

    /// Terminal fan-in publishing completion to a host-polled signal.
    pub const TWO_QUEUE_HOST_TERMINAL: Self = Self {
        barrier: true,
        acquire: FenceScope::None,
        release: FenceScope::System,
    };

    pub const BATCH_BOUNDARY_FIRST_SERIAL: Self = Self {
        barrier: true,
        acquire: FenceScope::System,
        release: FenceScope::None,
    };

    pub const BATCH_BOUNDARY_INTERNAL_SERIAL: Self = Self {
        barrier: true,
        acquire: FenceScope::None,
        release: FenceScope::None,
    };

    pub const BATCH_INTERNAL_RELEASE_AGENT: Self = Self {
        barrier: true,
        acquire: FenceScope::None,
        release: FenceScope::Agent,
    };

    pub const BATCH_INTERNAL_ACQUIRE_AGENT: Self = Self {
        barrier: true,
        acquire: FenceScope::Agent,
        release: FenceScope::None,
    };

    pub const BATCH_INTERNAL_RELEASE_SYSTEM: Self = Self {
        barrier: true,
        acquire: FenceScope::None,
        release: FenceScope::System,
    };

    pub const BATCH_INTERNAL_ACQUIRE_SYSTEM: Self = Self {
        barrier: true,
        acquire: FenceScope::System,
        release: FenceScope::None,
    };

    pub const BATCH_BOUNDARY_FIRST_INDEPENDENT: Self = Self {
        barrier: false,
        acquire: FenceScope::System,
        release: FenceScope::None,
    };

    pub const BATCH_BOUNDARY_INTERNAL_INDEPENDENT: Self = Self {
        barrier: false,
        acquire: FenceScope::None,
        release: FenceScope::None,
    };
}

fn packet_header(packet_type: u16, policy: HeaderPolicy) -> u16 {
    let mut header = packet_type << abi::PACKET_HEADER_TYPE;
    if policy.barrier {
        header |= 1 << abi::PACKET_HEADER_BARRIER;
    }
    header |= (policy.acquire as u16) << abi::PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    header |= (policy.release as u16) << abi::PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    header
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchGeometry {
    /// Total grid size in work-items, not work-group count.
    pub grid_workitems: [u32; 3],
    pub workgroup: [u16; 3],
    dimensions: u16,
}

impl LaunchGeometry {
    pub fn new(grid_workitems: [u32; 3], workgroup: [u16; 3]) -> Result<Self, PacketError> {
        for axis in 0..3 {
            if grid_workitems[axis] == 0 || workgroup[axis] == 0 {
                return Err(PacketError::ZeroDimension { axis });
            }
            if u32::from(workgroup[axis]) > grid_workitems[axis] {
                return Err(PacketError::WorkgroupExceedsGrid {
                    axis,
                    workgroup: workgroup[axis],
                    grid: grid_workitems[axis],
                });
            }
        }
        let dimensions = if grid_workitems[2] != 1 || workgroup[2] != 1 {
            3
        } else if grid_workitems[1] != 1 || workgroup[1] != 1 {
            2
        } else {
            1
        };
        if dimensions < 3 && (grid_workitems[2] != 1 || workgroup[2] != 1) {
            return Err(PacketError::InactiveDimensionNotOne { axis: 2 });
        }
        if dimensions < 2 && (grid_workitems[1] != 1 || workgroup[1] != 1) {
            return Err(PacketError::InactiveDimensionNotOne { axis: 1 });
        }
        Ok(Self {
            grid_workitems,
            workgroup,
            dimensions,
        })
    }

    pub fn dimensions(self) -> u16 {
        self.dimensions
    }

    /// Construct from work-group counts, matching HIP's grid convention while
    /// making AQL's total-work-item convention explicit.
    pub fn from_workgroups(workgroups: [u32; 3], workgroup: [u16; 3]) -> Result<Self, PacketError> {
        let mut grid_workitems = [0_u32; 3];
        for axis in 0..3 {
            grid_workitems[axis] = workgroups[axis]
                .checked_mul(u32::from(workgroup[axis]))
                .ok_or(PacketError::GridDimensionOverflow {
                    axis,
                    workgroups: workgroups[axis],
                    workgroup: workgroup[axis],
                })?;
        }
        Self::new(grid_workitems, workgroup)
    }

    /// Construct geometry for a HIP `dim3` launch.
    ///
    /// HIP supplies all three work-group IDs even when Y and Z have extent one.
    /// Keeping the AQL packet three-dimensional is required for kernels which
    /// read `blockIdx.y` while using a decode-time `grid.y == 1`; minimizing the
    /// setup dimension would otherwise leave that ID unspecified.
    pub fn from_hip_workgroups(
        workgroups: [u32; 3],
        workgroup: [u16; 3],
    ) -> Result<Self, PacketError> {
        let mut geometry = Self::from_workgroups(workgroups, workgroup)?;
        geometry.dimensions = 3;
        Ok(geometry)
    }
}

/// Loader-derived fields consumed by an AQL kernel-dispatch packet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelMetadata {
    pub kernel_object: u64,
    pub kernarg_segment_size: u32,
    pub kernarg_segment_alignment: u32,
    pub group_segment_size: u32,
    pub private_segment_size: u32,
    pub dynamic_callstack: bool,
}

/// ABI-identical to `hsa_kernel_dispatch_packet_t` on a 64-bit HSA host.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KernelDispatchPacket {
    full_header: u32,
    pub workgroup_size_x: u16,
    pub workgroup_size_y: u16,
    pub workgroup_size_z: u16,
    reserved0: u16,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,
    pub private_segment_size: u32,
    pub group_segment_size: u32,
    pub kernel_object: u64,
    pub kernarg_address: *mut c_void,
    reserved2: u64,
    pub completion_signal: abi::Signal,
}

impl KernelDispatchPacket {
    pub fn new(
        metadata: KernelMetadata,
        geometry: LaunchGeometry,
        dynamic_group_bytes: u32,
        kernarg_address: *mut c_void,
        completion_signal: abi::Signal,
    ) -> Result<Self, PacketError> {
        Self::new_with_policy(
            metadata,
            geometry,
            dynamic_group_bytes,
            kernarg_address,
            completion_signal,
            HeaderPolicy::RECORDED_DISPATCH,
        )
    }

    pub fn new_two_queue(
        metadata: KernelMetadata,
        geometry: LaunchGeometry,
        dynamic_group_bytes: u32,
        kernarg_address: *mut c_void,
        completion_signal: abi::Signal,
    ) -> Result<Self, PacketError> {
        Self::new_with_policy(
            metadata,
            geometry,
            dynamic_group_bytes,
            kernarg_address,
            completion_signal,
            HeaderPolicy::TWO_QUEUE_DISPATCH,
        )
    }

    pub fn new_with_policy(
        metadata: KernelMetadata,
        geometry: LaunchGeometry,
        dynamic_group_bytes: u32,
        kernarg_address: *mut c_void,
        completion_signal: abi::Signal,
        policy: HeaderPolicy,
    ) -> Result<Self, PacketError> {
        if metadata.kernel_object == 0 {
            return Err(PacketError::NullKernelObject);
        }
        if metadata.dynamic_callstack {
            return Err(PacketError::DynamicCallstackNeedsPrivateSize);
        }
        if metadata.kernarg_segment_size != 0 && kernarg_address.is_null() {
            return Err(PacketError::NullKernarg {
                required: metadata.kernarg_segment_size,
            });
        }
        let group_segment_size = metadata
            .group_segment_size
            .checked_add(dynamic_group_bytes)
            .ok_or(PacketError::DynamicGroupSegmentOverflow {
                static_bytes: metadata.group_segment_size,
                dynamic_bytes: dynamic_group_bytes,
            })?;
        let setup = geometry.dimensions() << abi::KERNEL_DISPATCH_SETUP_DIMENSIONS;
        let header = packet_header(abi::PACKET_TYPE_KERNEL_DISPATCH, policy);
        Ok(Self {
            full_header: u32::from(header) | (u32::from(setup) << 16),
            workgroup_size_x: geometry.workgroup[0],
            workgroup_size_y: geometry.workgroup[1],
            workgroup_size_z: geometry.workgroup[2],
            reserved0: 0,
            grid_size_x: geometry.grid_workitems[0],
            grid_size_y: geometry.grid_workitems[1],
            grid_size_z: geometry.grid_workitems[2],
            private_segment_size: metadata.private_segment_size,
            group_segment_size,
            kernel_object: metadata.kernel_object,
            kernarg_address,
            reserved2: 0,
            completion_signal,
        })
    }

    pub fn published_header_word(&self) -> u32 {
        self.full_header
    }
}

/// ABI-identical to `hsa_barrier_and_packet_t`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BarrierAndPacket {
    full_header: u32,
    reserved1: u32,
    pub dependency_signals: [abi::Signal; BARRIER_DEPENDENCY_CAPACITY],
    reserved2: u64,
    pub completion_signal: abi::Signal,
}

impl BarrierAndPacket {
    pub fn new(
        dependencies: &[abi::Signal],
        completion_signal: abi::Signal,
    ) -> Result<Self, PacketError> {
        Self::new_with_policy(
            dependencies,
            completion_signal,
            HeaderPolicy::CROSS_QUEUE_BARRIER,
        )
    }

    pub fn new_two_queue_dependency(dependencies: &[abi::Signal]) -> Result<Self, PacketError> {
        Self::new_with_policy(
            dependencies,
            abi::Signal(0),
            HeaderPolicy::TWO_QUEUE_DEPENDENCY,
        )
    }

    pub fn new_two_queue_host_terminal(
        dependencies: &[abi::Signal],
        completion_signal: abi::Signal,
    ) -> Result<Self, PacketError> {
        Self::new_with_policy(
            dependencies,
            completion_signal,
            HeaderPolicy::TWO_QUEUE_HOST_TERMINAL,
        )
    }

    pub fn new_with_policy(
        dependencies: &[abi::Signal],
        completion_signal: abi::Signal,
        policy: HeaderPolicy,
    ) -> Result<Self, PacketError> {
        if dependencies.len() > BARRIER_DEPENDENCY_CAPACITY {
            return Err(PacketError::TooManyBarrierDependencies {
                count: dependencies.len(),
                capacity: BARRIER_DEPENDENCY_CAPACITY,
            });
        }
        let mut dependency_signals = [abi::Signal(0); BARRIER_DEPENDENCY_CAPACITY];
        dependency_signals[..dependencies.len()].copy_from_slice(dependencies);
        let header = packet_header(abi::PACKET_TYPE_BARRIER_AND, policy);
        Ok(Self {
            full_header: u32::from(header),
            reserved1: 0,
            dependency_signals,
            reserved2: 0,
            completion_signal,
        })
    }

    pub fn published_header_word(&self) -> u32 {
        self.full_header
    }
}

/// Fully prepared, reusable packet bytes with the publication word kept out of
/// band. Queue code copies bytes 4..64 first and publishes `header_word` last.
#[derive(Clone, Debug)]
pub struct PacketImage {
    pub bytes: [u8; AQL_PACKET_BYTES],
    pub header_word: u32,
}

impl PacketImage {
    pub fn kernel(packet: &KernelDispatchPacket) -> Self {
        Self::copy(packet, packet.published_header_word())
    }

    pub fn barrier(packet: &BarrierAndPacket) -> Self {
        Self::copy(packet, packet.published_header_word())
    }

    /// Build AMD's vendor-specific AQL packet carrying one PM4
    /// INDIRECT_BUFFER command. Layout and constants match rocm-systems
    /// `aqlprofile/src/core/amd_aql_pm4_ib_packet.h`.
    pub fn pm4_indirect_buffer(
        address: *mut c_void,
        dwords: u32,
        completion_signal: abi::Signal,
    ) -> Result<Self, PacketError> {
        if address.is_null() || (address as usize) & 3 != 0 {
            return Err(PacketError::InvalidIndirectBufferAddress(
                address as usize as u64,
            ));
        }
        if dwords == 0 || dwords > 0x000f_ffff {
            return Err(PacketError::InvalidIndirectBufferSize(dwords));
        }
        const PACKET3_INDIRECT_BUFFER: u32 = 0x3f;
        const IB_VALID: u32 = 1 << 23;
        const IB_TEMPORAL_LU: u32 = 3 << 28;
        let address = address as usize as u64;
        let pm4_header = (3_u32 << 30) | (2 << 16) | (PACKET3_INDIRECT_BUFFER << 8);
        // Vendor-specific is packet type zero.  The retained IB crosses from
        // HIP-owned allocations into a distinct ROCr queue and back again, so
        // its completion must carry the same system-scope ownership contract
        // as an ordinary terminal AQL dispatch.  Barrier alone orders packets
        // but does not make shader writes available to host/HIP consumers.
        let aql_header = packet_header(
            0,
            HeaderPolicy {
                barrier: true,
                acquire: FenceScope::System,
                release: FenceScope::System,
            },
        );
        let mut bytes = [0_u8; AQL_PACKET_BYTES];
        bytes[0..2].copy_from_slice(&aql_header.to_le_bytes());
        bytes[2..4].copy_from_slice(&1_u16.to_le_bytes());
        bytes[4..8].copy_from_slice(&pm4_header.to_le_bytes());
        bytes[8..12].copy_from_slice(&(address as u32 & 0xffff_fffc).to_le_bytes());
        bytes[12..16].copy_from_slice(&((address >> 32) as u32).to_le_bytes());
        bytes[16..20].copy_from_slice(&(dwords | IB_VALID | IB_TEMPORAL_LU).to_le_bytes());
        bytes[20..24].copy_from_slice(&10_u32.to_le_bytes());
        bytes[56..64].copy_from_slice(&completion_signal.0.to_le_bytes());
        Ok(Self {
            bytes,
            header_word: u32::from(aql_header) | (1_u32 << 16),
        })
    }

    fn copy<T>(packet: &T, header_word: u32) -> Self {
        debug_assert_eq!(std::mem::size_of::<T>(), AQL_PACKET_BYTES);
        let mut bytes = [0_u8; AQL_PACKET_BYTES];
        // Both packet structs have fully initialized, padding-free 64-byte C
        // layouts, asserted below. The resulting image never exposes a Rust
        // reference to reinterpreted bytes.
        // SAFETY: source and destination are valid, non-overlapping 64-byte
        // regions.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (packet as *const T).cast::<u8>(),
                bytes.as_mut_ptr(),
                AQL_PACKET_BYTES,
            );
        }
        Self { bytes, header_word }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PacketError {
    ZeroDimension {
        axis: usize,
    },
    WorkgroupExceedsGrid {
        axis: usize,
        workgroup: u16,
        grid: u32,
    },
    InactiveDimensionNotOne {
        axis: usize,
    },
    GridDimensionOverflow {
        axis: usize,
        workgroups: u32,
        workgroup: u16,
    },
    NullKernelObject,
    NullKernarg {
        required: u32,
    },
    DynamicCallstackNeedsPrivateSize,
    DynamicGroupSegmentOverflow {
        static_bytes: u32,
        dynamic_bytes: u32,
    },
    TooManyBarrierDependencies {
        count: usize,
        capacity: usize,
    },
    InvalidIndirectBufferAddress(u64),
    InvalidIndirectBufferSize(u32),
}

impl fmt::Display for PacketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension { axis } => write!(f, "launch axis {axis} is zero"),
            Self::WorkgroupExceedsGrid {
                axis,
                workgroup,
                grid,
            } => write!(
                f,
                "work-group axis {axis} ({workgroup}) exceeds grid work-items ({grid})"
            ),
            Self::InactiveDimensionNotOne { axis } => {
                write!(f, "inactive launch axis {axis} is not one")
            }
            Self::GridDimensionOverflow {
                axis,
                workgroups,
                workgroup,
            } => write!(
                f,
                "grid axis {axis} overflows: {workgroups} groups * {workgroup} work-items"
            ),
            Self::NullKernelObject => write!(f, "kernel object handle is zero"),
            Self::NullKernarg { required } => {
                write!(
                    f,
                    "kernel requires {required} kernarg bytes but address is null"
                )
            }
            Self::DynamicCallstackNeedsPrivateSize => write!(
                f,
                "dynamic-callstack kernel needs an explicit private-segment policy"
            ),
            Self::DynamicGroupSegmentOverflow {
                static_bytes,
                dynamic_bytes,
            } => write!(
                f,
                "group segment size overflows: {static_bytes} static bytes + {dynamic_bytes} dynamic bytes"
            ),
            Self::TooManyBarrierDependencies { count, capacity } => write!(
                f,
                "barrier has {count} dependencies; one AQL packet holds {capacity}"
            ),
            Self::InvalidIndirectBufferAddress(address) => write!(
                f,
                "PM4 indirect buffer address 0x{address:x} is null or unaligned"
            ),
            Self::InvalidIndirectBufferSize(dwords) => write!(
                f,
                "PM4 indirect buffer size {dwords} dwords is outside 1..=0xfffff"
            ),
        }
    }
}

impl std::error::Error for PacketError {}

const _: () = {
    assert!(std::mem::size_of::<KernelDispatchPacket>() == AQL_PACKET_BYTES);
    assert!(std::mem::align_of::<KernelDispatchPacket>() == 8);
    assert!(std::mem::size_of::<BarrierAndPacket>() == AQL_PACKET_BYTES);
    assert!(std::mem::align_of::<BarrierAndPacket>() == 8);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_abi_and_offsets_match_hsa_headers() {
        assert_eq!(std::mem::size_of::<KernelDispatchPacket>(), 64);
        assert_eq!(std::mem::size_of::<BarrierAndPacket>(), 64);
        assert_eq!(std::mem::align_of::<KernelDispatchPacket>(), 8);
        assert_eq!(std::mem::align_of::<BarrierAndPacket>(), 8);

        let kernel = std::mem::MaybeUninit::<KernelDispatchPacket>::uninit();
        let base = kernel.as_ptr() as usize;
        // SAFETY: only addresses are formed; no uninitialized field is read.
        unsafe {
            assert_eq!(
                std::ptr::addr_of!((*kernel.as_ptr()).kernel_object) as usize - base,
                32
            );
            assert_eq!(
                std::ptr::addr_of!((*kernel.as_ptr()).kernarg_address) as usize - base,
                40
            );
            assert_eq!(
                std::ptr::addr_of!((*kernel.as_ptr()).completion_signal) as usize - base,
                56
            );
        }

        let barrier = std::mem::MaybeUninit::<BarrierAndPacket>::uninit();
        let base = barrier.as_ptr() as usize;
        // SAFETY: only addresses are formed; no uninitialized field is read.
        unsafe {
            assert_eq!(
                std::ptr::addr_of!((*barrier.as_ptr()).dependency_signals) as usize - base,
                8
            );
            assert_eq!(
                std::ptr::addr_of!((*barrier.as_ptr()).completion_signal) as usize - base,
                56
            );
        }
    }

    #[test]
    fn kernel_header_is_barriered_system_scope_and_carries_dimensions() {
        let geometry = LaunchGeometry::new([1024, 1, 1], [64, 1, 1]).unwrap();
        let metadata = KernelMetadata {
            kernel_object: 7,
            kernarg_segment_size: 0,
            kernarg_segment_alignment: 16,
            group_segment_size: 32,
            private_segment_size: 64,
            dynamic_callstack: false,
        };
        let packet =
            KernelDispatchPacket::new(metadata, geometry, 0, std::ptr::null_mut(), abi::Signal(9))
                .unwrap();
        assert_eq!(packet.published_header_word() & 0xffff, 0x1502);
        assert_eq!(packet.published_header_word() >> 16, 1);
        assert_eq!(packet.group_segment_size, 32);
        let dynamic =
            KernelDispatchPacket::new(metadata, geometry, 16, std::ptr::null_mut(), abi::Signal(9))
                .unwrap();
        assert_eq!(dynamic.group_segment_size, 48);
    }

    #[test]
    fn barrier_zero_fills_unused_dependencies() {
        let packet =
            BarrierAndPacket::new(&[abi::Signal(11), abi::Signal(12)], abi::Signal(13)).unwrap();
        assert_eq!(packet.published_header_word(), 0x1503);
        assert_eq!(
            packet.dependency_signals,
            [
                abi::Signal(11),
                abi::Signal(12),
                abi::Signal(0),
                abi::Signal(0),
                abi::Signal(0)
            ]
        );
    }

    #[test]
    fn pm4_ib_packet_matches_aqlprofile_vendor_layout() {
        let packet = PacketImage::pm4_indirect_buffer(
            0x1234_5678_9000_usize as *mut c_void,
            0x321,
            abi::Signal(0x5566_7788_99aa_bbcc),
        )
        .unwrap();
        assert_eq!(packet.header_word, 0x0001_1500);
        assert_eq!(&packet.bytes[0..4], &0x0001_1500_u32.to_le_bytes());
        assert_eq!(&packet.bytes[4..8], &0xc002_3f00_u32.to_le_bytes());
        assert_eq!(&packet.bytes[8..12], &0x5678_9000_u32.to_le_bytes());
        assert_eq!(&packet.bytes[12..16], &0x0000_1234_u32.to_le_bytes());
        assert_eq!(&packet.bytes[16..20], &0x3080_0321_u32.to_le_bytes());
        assert_eq!(&packet.bytes[20..24], &10_u32.to_le_bytes());
        assert!(packet.bytes[24..56].iter().all(|byte| *byte == 0));
        assert_eq!(
            &packet.bytes[56..64],
            &0x5566_7788_99aa_bbcc_u64.to_le_bytes()
        );
    }

    #[test]
    fn specialized_two_queue_headers_pin_exact_fence_and_barrier_bits() {
        let geometry = LaunchGeometry::new([32, 1, 1], [32, 1, 1]).unwrap();
        let metadata = KernelMetadata {
            kernel_object: 7,
            kernarg_segment_size: 0,
            kernarg_segment_alignment: 16,
            group_segment_size: 0,
            private_segment_size: 0,
            dynamic_callstack: false,
        };
        let dispatch = KernelDispatchPacket::new_two_queue(
            metadata,
            geometry,
            0,
            std::ptr::null_mut(),
            abi::Signal(9),
        )
        .unwrap();
        let dependency = BarrierAndPacket::new_two_queue_dependency(&[abi::Signal(11)]).unwrap();
        let terminal =
            BarrierAndPacket::new_two_queue_host_terminal(&[abi::Signal(12)], abi::Signal(13))
                .unwrap();

        assert_eq!(dispatch.published_header_word() & 0xffff, 0x0d02);
        assert_eq!(dependency.published_header_word(), 0x0103);
        assert_eq!(terminal.published_header_word(), 0x1103);
        assert_header_policy(
            dispatch.published_header_word() as u16,
            true,
            FenceScope::System,
            FenceScope::Agent,
        );
        assert_header_policy(
            dependency.published_header_word() as u16,
            true,
            FenceScope::None,
            FenceScope::None,
        );
        assert_header_policy(
            terminal.published_header_word() as u16,
            true,
            FenceScope::None,
            FenceScope::System,
        );
    }

    #[test]
    fn fence_ab_headers_pin_scope_and_ordering_axes() {
        let geometry = LaunchGeometry::new([32, 1, 1], [32, 1, 1]).unwrap();
        let metadata = KernelMetadata {
            kernel_object: 7,
            kernarg_segment_size: 0,
            kernarg_segment_alignment: 16,
            group_segment_size: 0,
            private_segment_size: 0,
            dynamic_callstack: false,
        };
        let header = |policy| {
            KernelDispatchPacket::new_with_policy(
                metadata,
                geometry,
                0,
                std::ptr::null_mut(),
                abi::Signal(9),
                policy,
            )
            .unwrap()
            .published_header_word()
                & 0xffff
        };
        assert_eq!(header(HeaderPolicy::RECORDED_DISPATCH), 0x1502);
        assert_eq!(header(HeaderPolicy::TWO_QUEUE_DISPATCH), 0x0d02);
        assert_eq!(header(HeaderPolicy::BATCH_BOUNDARY_FIRST_SERIAL), 0x0502);
        assert_eq!(header(HeaderPolicy::BATCH_BOUNDARY_INTERNAL_SERIAL), 0x0102);
        assert_eq!(
            header(HeaderPolicy::BATCH_BOUNDARY_FIRST_INDEPENDENT),
            0x0402
        );
        assert_eq!(
            header(HeaderPolicy::BATCH_BOUNDARY_INTERNAL_INDEPENDENT),
            0x0002
        );
    }

    fn assert_header_policy(header: u16, barrier: bool, acquire: FenceScope, release: FenceScope) {
        assert_eq!((header >> abi::PACKET_HEADER_BARRIER) & 1, barrier as u16);
        assert_eq!(
            (header >> abi::PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) & 0x3,
            acquire as u16
        );
        assert_eq!(
            (header >> abi::PACKET_HEADER_SCRELEASE_FENCE_SCOPE) & 0x3,
            release as u16
        );
    }
}
