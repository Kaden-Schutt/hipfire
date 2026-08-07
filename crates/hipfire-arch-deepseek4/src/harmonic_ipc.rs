// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Persistent CPU shared-memory ring for the DeepSeek V4 harmonic protocol.
//!
//! Every payload field is atomic. A source publishes with a release transition
//! only after the complete packet has been written; the destination consumes
//! after an acquire transition. No method waits or polls. Process supervision,
//! deadlines, and bounded worker teardown live above this transport.

use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::fmt;
use std::fs::File;
use std::io;
use std::mem;
use std::ptr;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use hip_bridge::{DeviceBuffer, HipRuntime};
use memmap2::{MmapMut, MmapOptions};

use crate::harmonic::{
    HarmonicCompletion, HarmonicContract, HarmonicOwner, HarmonicProtocolError,
    HarmonicRoutePacket, HarmonicSlotState, DS4_MQ2R_0731_IDENTITY, HARMONIC_ACTIVATION_EXTENT,
    HARMONIC_ACTIVATION_RESERVED_OFFSET, HARMONIC_EXPERT_IDS_OFFSET, HARMONIC_RESULT_EXTENT,
    HARMONIC_ROUTE_IDENTITY, HARMONIC_ROUTE_WEIGHTS_OFFSET, HARMONIC_SLOT_COUNT, HARMONIC_TOP_K,
};

const HARMONIC_WIRE_MAGIC: u64 = 0x4453_3448_4950_4331; // DS4HIPC1
const HARMONIC_WIRE_VERSION: u32 = 2;
const FLAG_SOURCE_OBSERVED: u32 = 1 << 0;
const FLAG_DESTINATION_QUIESCED: u32 = 1 << 1;
const OWNER_DENSE_MASK: u32 = 1 << 0;
const OWNER_EXPERT_MASK: u32 = 1 << 1;
const ACTIVATION_WORDS: usize = HARMONIC_ACTIVATION_EXTENT as usize / mem::size_of::<u64>();
const RESULT_WORDS: usize = HARMONIC_RESULT_EXTENT as usize / mem::size_of::<u64>();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum HarmonicWireState {
    Vacant = 0,
    Publishing = 1,
    Published = 2,
    Running = 3,
    Completing = 4,
    Completed = 5,
    Cancelled = 6,
    TimedOut = 7,
    FailedDense = 8,
    FailedExpert = 9,
}

impl HarmonicWireState {
    fn decode(raw: u32) -> HarmonicIpcResult<Self> {
        match raw {
            0 => Ok(Self::Vacant),
            1 => Ok(Self::Publishing),
            2 => Ok(Self::Published),
            3 => Ok(Self::Running),
            4 => Ok(Self::Completing),
            5 => Ok(Self::Completed),
            6 => Ok(Self::Cancelled),
            7 => Ok(Self::TimedOut),
            8 => Ok(Self::FailedDense),
            9 => Ok(Self::FailedExpert),
            _ => Err(HarmonicIpcError::InvalidLayout("wire state")),
        }
    }

    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed
                | Self::Cancelled
                | Self::TimedOut
                | Self::FailedDense
                | Self::FailedExpert
        )
    }

    pub const fn logical(self) -> HarmonicSlotState {
        match self {
            Self::Vacant => HarmonicSlotState::Vacant,
            Self::Publishing | Self::Published => HarmonicSlotState::Published,
            Self::Running | Self::Completing => HarmonicSlotState::Running,
            Self::Completed => HarmonicSlotState::Completed,
            Self::Cancelled => HarmonicSlotState::Cancelled,
            Self::TimedOut => HarmonicSlotState::TimedOut,
            Self::FailedDense => HarmonicSlotState::Failed(HarmonicOwner::DenseGfx1100),
            Self::FailedExpert => HarmonicSlotState::Failed(HarmonicOwner::ExpertGfx1151),
        }
    }
}

#[derive(Debug)]
pub enum HarmonicIpcError {
    Io(io::Error),
    Protocol(HarmonicProtocolError),
    InvalidLayout(&'static str),
    CompareExchange {
        operation: &'static str,
        expected: HarmonicWireState,
        actual: HarmonicWireState,
    },
}

impl fmt::Display for HarmonicIpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "deepseek4 harmonic IPC I/O: {error}"),
            Self::Protocol(error) => error.fmt(f),
            Self::InvalidLayout(field) => {
                write!(f, "deepseek4 harmonic IPC invalid shared layout: {field}")
            }
            Self::CompareExchange {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "deepseek4 harmonic IPC {operation}: expected {expected:?}, found {actual:?}"
            ),
        }
    }
}

impl std::error::Error for HarmonicIpcError {}

impl From<io::Error> for HarmonicIpcError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<HarmonicProtocolError> for HarmonicIpcError {
    fn from(value: HarmonicProtocolError) -> Self {
        Self::Protocol(value)
    }
}

pub type HarmonicIpcResult<T> = Result<T, HarmonicIpcError>;

#[derive(Clone, Debug, PartialEq)]
pub struct HarmonicWorkItem {
    pub packet: HarmonicRoutePacket,
    pub activation_payload: Vec<u8>,
    pub integrity_mode: HarmonicIntegrityMode,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarmonicResolved {
    pub state: HarmonicSlotState,
    pub completion: Option<HarmonicCompletion>,
    pub result_payload: Option<Vec<u8>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarmonicMappedResolved {
    pub state: HarmonicSlotState,
    pub completion: Option<HarmonicCompletion>,
}

/// One nonblocking observation by the persistent expert worker.
///
/// `Pending` is the normal idle result. It is intentionally not an error and
/// does not require a control-socket command from the dense process.
#[derive(Clone, Debug, PartialEq)]
pub enum HarmonicExpertPoll {
    Pending,
    Work(HarmonicWorkItem),
    Terminal(HarmonicWireState),
}

/// Nonblocking expert observation for a HIP-registered ring. Payload bytes
/// remain in the mapped slot and are consumed by the local GPU alias.
#[derive(Clone, Debug, PartialEq)]
pub enum HarmonicExpertMappedPoll {
    Pending,
    Work(HarmonicRoutePacket),
    Terminal(HarmonicWireState),
}

/// Runtime payload validation level for the shared ring.
///
/// The control fields, epoch, generations, owners, extents, and state machine
/// are checked in both modes. `Fingerprint` additionally re-hashes each 16 KiB
/// payload and is reserved for the CPU correctness oracle. `ReleaseAcquire`
/// is the shipping-shaped data plane: publication ordering protects one bulk
/// copy without a byte-wise checksum on every layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum HarmonicIntegrityMode {
    ReleaseAcquire = 0,
    Fingerprint = 1,
}

impl HarmonicIntegrityMode {
    fn decode(raw: u32) -> HarmonicIpcResult<Self> {
        match raw {
            0 => Ok(Self::ReleaseAcquire),
            1 => Ok(Self::Fingerprint),
            _ => Err(HarmonicIpcError::InvalidLayout("integrity mode")),
        }
    }
}

/// Cross-process monotonic nanoseconds used by packet deadlines.
///
/// Both worker processes call the same kernel clock; unlike `Instant`, the
/// value can safely be carried through the shared-memory protocol.
#[cfg(target_os = "linux")]
pub fn harmonic_monotonic_tick() -> HarmonicIpcResult<u64> {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `value` is a valid writable timespec and CLOCK_MONOTONIC does
    // not require any additional lifetime or ownership contract.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut value) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    let seconds = u64::try_from(value.tv_sec)
        .map_err(|_| HarmonicIpcError::InvalidLayout("monotonic seconds"))?;
    let nanos = u64::try_from(value.tv_nsec)
        .map_err(|_| HarmonicIpcError::InvalidLayout("monotonic nanoseconds"))?;
    Ok(seconds.saturating_mul(1_000_000_000).saturating_add(nanos))
}

#[repr(C, align(64))]
struct HarmonicWireHeader {
    magic: AtomicU64,
    version: AtomicU32,
    slot_count: AtomicU32,
    integrity_mode: AtomicU32,
    route_identity: AtomicU64,
    model_identity: [AtomicU64; 4],
    source_generation: AtomicU64,
    destination_generation: AtomicU64,
    isolated_owners: AtomicU32,
}

impl HarmonicWireHeader {
    fn new(contract: HarmonicContract, integrity_mode: HarmonicIntegrityMode) -> Self {
        let model_words = identity_to_words(contract.model_identity);
        Self {
            magic: AtomicU64::new(HARMONIC_WIRE_MAGIC),
            version: AtomicU32::new(HARMONIC_WIRE_VERSION),
            slot_count: AtomicU32::new(HARMONIC_SLOT_COUNT as u32),
            integrity_mode: AtomicU32::new(integrity_mode as u32),
            route_identity: AtomicU64::new(contract.route_identity),
            model_identity: model_words.map(AtomicU64::new),
            source_generation: AtomicU64::new(contract.source_allocation_generation),
            destination_generation: AtomicU64::new(contract.destination_allocation_generation),
            isolated_owners: AtomicU32::new(0),
        }
    }

    fn contract(&self) -> HarmonicContract {
        HarmonicContract {
            route_identity: self.route_identity.load(Ordering::Acquire),
            model_identity: words_to_identity(
                self.model_identity
                    .each_ref()
                    .map(|word| word.load(Ordering::Acquire)),
            ),
            source_allocation_generation: self.source_generation.load(Ordering::Acquire),
            destination_allocation_generation: self.destination_generation.load(Ordering::Acquire),
        }
    }
}

/// SPSC payload storage synchronized exclusively by the slot-state release and
/// acquire transitions. The source process is the sole activation writer and
/// the expert process is the sole result writer. Neither buffer is reused
/// until both owners have observed a terminal epoch.
#[repr(transparent)]
struct HarmonicPayload<const N: usize> {
    bytes: UnsafeCell<[u8; N]>,
}

impl<const N: usize> HarmonicPayload<N> {
    const fn new() -> Self {
        Self {
            bytes: UnsafeCell::new([0; N]),
        }
    }
}

// SAFETY: sharing is governed by the atomic slot state. Each payload has one
// fixed writer; readers acquire Published/Completed before copying, and slot
// recycling requires both owners to report quiescence.
unsafe impl<const N: usize> Sync for HarmonicPayload<N> {}

#[repr(C, align(64))]
struct HarmonicWireSlot {
    state: AtomicU32,
    flags: AtomicU32,
    epoch: AtomicU64,
    route_identity: AtomicU64,
    model_identity: [AtomicU64; 4],
    layer: AtomicU32,
    slot: AtomicU32,
    source_owner: AtomicU32,
    destination_owner: AtomicU32,
    source_generation: AtomicU64,
    destination_generation: AtomicU64,
    expert_ids: [AtomicU32; HARMONIC_TOP_K],
    route_weight_bits: [AtomicU32; HARMONIC_TOP_K],
    activation_extent: AtomicU32,
    result_extent: AtomicU32,
    deadline_tick: AtomicU64,
    activation_fingerprint: AtomicU64,
    result_fingerprint: AtomicU64,
    activation_payload: HarmonicPayload<{ ACTIVATION_WORDS * mem::size_of::<u64>() }>,
    result_payload: HarmonicPayload<{ RESULT_WORDS * mem::size_of::<u64>() }>,
}

// `deepseek4_harmonic_mailbox.gfx1100.hip` consumes only this control prefix.
// Keep these assertions beside the authority type so an innocent Rust layout
// edit fails the build instead of silently corrupting a cross-process packet.
const _: () = {
    assert!(HARMONIC_SLOT_COUNT == 2);
    assert!(mem::offset_of!(HarmonicWireHeader, route_identity) == 24);
    assert!(mem::offset_of!(HarmonicWireHeader, model_identity) == 32);
    assert!(mem::offset_of!(HarmonicWireHeader, source_generation) == 64);
    assert!(mem::offset_of!(HarmonicWireHeader, destination_generation) == 72);
    assert!(mem::offset_of!(HarmonicWireHeader, isolated_owners) == 80);
    assert!(mem::offset_of!(HarmonicWireSlot, state) == 0);
    assert!(mem::offset_of!(HarmonicWireSlot, flags) == 4);
    assert!(mem::offset_of!(HarmonicWireSlot, epoch) == 8);
    assert!(mem::offset_of!(HarmonicWireSlot, route_identity) == 16);
    assert!(mem::offset_of!(HarmonicWireSlot, model_identity) == 24);
    assert!(mem::offset_of!(HarmonicWireSlot, layer) == 56);
    assert!(mem::offset_of!(HarmonicWireSlot, slot) == 60);
    assert!(mem::offset_of!(HarmonicWireSlot, source_owner) == 64);
    assert!(mem::offset_of!(HarmonicWireSlot, destination_owner) == 68);
    assert!(mem::offset_of!(HarmonicWireSlot, source_generation) == 72);
    assert!(mem::offset_of!(HarmonicWireSlot, destination_generation) == 80);
    assert!(mem::offset_of!(HarmonicWireSlot, expert_ids) == 88);
    assert!(mem::offset_of!(HarmonicWireSlot, route_weight_bits) == 112);
    assert!(mem::offset_of!(HarmonicWireSlot, activation_extent) == 136);
    assert!(mem::offset_of!(HarmonicWireSlot, result_extent) == 140);
    assert!(mem::offset_of!(HarmonicWireSlot, deadline_tick) == 144);
    assert!(mem::offset_of!(HarmonicWireSlot, activation_fingerprint) == 152);
    assert!(mem::offset_of!(HarmonicWireSlot, result_fingerprint) == 160);
    assert!(mem::offset_of!(HarmonicWireSlot, activation_payload) == 168);
};

impl HarmonicWireSlot {
    fn new() -> Self {
        Self {
            state: AtomicU32::new(HarmonicWireState::Vacant as u32),
            flags: AtomicU32::new(0),
            epoch: AtomicU64::new(0),
            route_identity: AtomicU64::new(0),
            model_identity: [const { AtomicU64::new(0) }; 4],
            layer: AtomicU32::new(0),
            slot: AtomicU32::new(0),
            source_owner: AtomicU32::new(0),
            destination_owner: AtomicU32::new(0),
            source_generation: AtomicU64::new(0),
            destination_generation: AtomicU64::new(0),
            expert_ids: [const { AtomicU32::new(0) }; HARMONIC_TOP_K],
            route_weight_bits: [const { AtomicU32::new(0) }; HARMONIC_TOP_K],
            activation_extent: AtomicU32::new(0),
            result_extent: AtomicU32::new(0),
            deadline_tick: AtomicU64::new(0),
            activation_fingerprint: AtomicU64::new(0),
            result_fingerprint: AtomicU64::new(0),
            activation_payload: HarmonicPayload::new(),
            result_payload: HarmonicPayload::new(),
        }
    }

    fn state(&self) -> HarmonicIpcResult<HarmonicWireState> {
        HarmonicWireState::decode(self.state.load(Ordering::Acquire))
    }

    fn packet(&self) -> HarmonicIpcResult<HarmonicRoutePacket> {
        Ok(HarmonicRoutePacket {
            route_identity: self.route_identity.load(Ordering::Relaxed),
            model_identity: words_to_identity(
                self.model_identity
                    .each_ref()
                    .map(|word| word.load(Ordering::Relaxed)),
            ),
            epoch: self.epoch.load(Ordering::Relaxed),
            layer: u16::try_from(self.layer.load(Ordering::Relaxed))
                .map_err(|_| HarmonicIpcError::InvalidLayout("layer"))?,
            slot: u8::try_from(self.slot.load(Ordering::Relaxed))
                .map_err(|_| HarmonicIpcError::InvalidLayout("slot"))?,
            source_owner: decode_owner(self.source_owner.load(Ordering::Relaxed))?,
            destination_owner: decode_owner(self.destination_owner.load(Ordering::Relaxed))?,
            source_allocation_generation: self.source_generation.load(Ordering::Relaxed),
            destination_allocation_generation: self.destination_generation.load(Ordering::Relaxed),
            expert_ids: self
                .expert_ids
                .each_ref()
                .map(|expert| expert.load(Ordering::Relaxed)),
            route_weight_bits: self
                .route_weight_bits
                .each_ref()
                .map(|weight| weight.load(Ordering::Relaxed)),
            activation_extent: self.activation_extent.load(Ordering::Relaxed),
            result_extent: self.result_extent.load(Ordering::Relaxed),
            deadline_tick: self.deadline_tick.load(Ordering::Relaxed),
            activation_fingerprint: self.activation_fingerprint.load(Ordering::Relaxed),
        })
    }

    fn check_epoch(&self, got: u64) -> HarmonicIpcResult<()> {
        let expected = self.epoch.load(Ordering::Acquire);
        if expected != got || expected == 0 {
            return Err(HarmonicProtocolError::StaleEpoch { expected, got }.into());
        }
        Ok(())
    }

    fn transition(
        &self,
        operation: &'static str,
        expected: HarmonicWireState,
        next: HarmonicWireState,
    ) -> HarmonicIpcResult<()> {
        self.state
            .compare_exchange(
                expected as u32,
                next as u32,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|actual| match HarmonicWireState::decode(actual) {
                Ok(actual) => HarmonicIpcError::CompareExchange {
                    operation,
                    expected,
                    actual,
                },
                Err(error) => error,
            })
    }
}

#[repr(C, align(64))]
struct HarmonicWireLayout {
    header: HarmonicWireHeader,
    slots: [HarmonicWireSlot; HARMONIC_SLOT_COUNT],
}

impl HarmonicWireLayout {
    fn new(contract: HarmonicContract, integrity_mode: HarmonicIntegrityMode) -> Self {
        Self {
            header: HarmonicWireHeader::new(contract, integrity_mode),
            slots: std::array::from_fn(|_| HarmonicWireSlot::new()),
        }
    }
}

pub struct HarmonicSharedRing {
    mmap: MmapMut,
}

/// One process-local HIP registration of the shared ring's file-backed pages.
///
/// Each harmonic owner constructs its own instance after binding its exact
/// device. The host mmap remains the protocol authority; `device_base` is only
/// that process's address-space alias for asynchronous payload transfer. No
/// pointer is sent to the peer process or shared between HIP contexts.
pub struct HarmonicGpuMapping {
    host_base: *mut c_void,
    mapping_bytes: usize,
    device_base: DeviceBuffer,
    slot_offsets: [usize; HARMONIC_SLOT_COUNT],
    activation_offsets: [usize; HARMONIC_SLOT_COUNT],
    result_offsets: [usize; HARMONIC_SLOT_COUNT],
    registered: bool,
}

impl fmt::Debug for HarmonicGpuMapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarmonicGpuMapping")
            .field("host_base", &self.host_base)
            .field("device_base", &self.device_base.as_ptr())
            .field("mapping_bytes", &self.mapping_bytes)
            .field("registered", &self.registered)
            .finish()
    }
}

impl HarmonicGpuMapping {
    pub fn register(
        ring: &mut HarmonicSharedRing,
        hip: &HipRuntime,
    ) -> hip_bridge::HipResult<Self> {
        let host_base = ring.mmap.as_mut_ptr().cast::<c_void>();
        let mapping_bytes = ring.mmap.len();
        let base = host_base as usize;
        let slot_offsets = std::array::from_fn(|slot| {
            let slot = &ring.layout().slots[slot] as *const HarmonicWireSlot as usize;
            slot.checked_sub(base)
                .expect("harmonic slot precedes ring base")
        });
        let activation_offsets = std::array::from_fn(|slot| {
            let payload = ring.layout().slots[slot]
                .activation_payload
                .bytes
                .get()
                .cast::<u8>() as usize;
            payload
                .checked_sub(base)
                .expect("activation payload precedes ring base")
        });
        let result_offsets = std::array::from_fn(|slot| {
            let payload = ring.layout().slots[slot]
                .result_payload
                .bytes
                .get()
                .cast::<u8>() as usize;
            payload
                .checked_sub(base)
                .expect("result payload precedes ring base")
        });
        for (offset, extent) in activation_offsets
            .iter()
            .map(|offset| (*offset, HARMONIC_ACTIVATION_EXTENT as usize))
            .chain(
                result_offsets
                    .iter()
                    .map(|offset| (*offset, HARMONIC_RESULT_EXTENT as usize)),
            )
        {
            assert!(
                offset.saturating_add(extent) <= mapping_bytes,
                "harmonic payload view exceeds registered mapping"
            );
        }

        // SAFETY: `ring` owns this page-aligned live mmap. Product ownership
        // keeps it mapped until `unregister` has synchronized and released the
        // process-local HIP alias.
        unsafe { hip.host_register_mapped(host_base, mapping_bytes)? };
        let device_base = match unsafe { hip.host_get_device_buffer(host_base, mapping_bytes) } {
            Ok(device_base) => device_base,
            Err(error) => {
                // SAFETY: registration succeeded above and no device alias was
                // published, so no GPU operation can be using the mapping.
                let _ = unsafe { hip.host_unregister(host_base) };
                return Err(error);
            }
        };
        Ok(Self {
            host_base,
            mapping_bytes,
            device_base,
            slot_offsets,
            activation_offsets,
            result_offsets,
            registered: true,
        })
    }

    pub fn activation_buffer(&self, epoch: u64) -> DeviceBuffer {
        self.payload_buffer(
            self.activation_offsets[epoch as usize % HARMONIC_SLOT_COUNT],
            HARMONIC_ACTIVATION_EXTENT as usize,
        )
    }

    pub fn header_buffer(&self) -> DeviceBuffer {
        self.payload_buffer(0, mem::size_of::<HarmonicWireHeader>())
    }

    pub fn slot_control_buffer(&self, slot: usize) -> DeviceBuffer {
        assert!(slot < HARMONIC_SLOT_COUNT);
        self.payload_buffer(
            self.slot_offsets[slot],
            mem::offset_of!(HarmonicWireSlot, activation_payload),
        )
    }

    pub fn result_buffer(&self, epoch: u64) -> DeviceBuffer {
        self.payload_buffer(
            self.result_offsets[epoch as usize % HARMONIC_SLOT_COUNT],
            HARMONIC_RESULT_EXTENT as usize,
        )
    }

    pub fn unregister(&mut self, hip: &HipRuntime) -> hip_bridge::HipResult<()> {
        if !self.registered {
            return Ok(());
        }
        // SAFETY: the owning execution object synchronizes its local streams
        // before calling this method and the exact registration base is kept in
        // this object.
        unsafe { hip.host_unregister(self.host_base)? };
        self.registered = false;
        Ok(())
    }

    fn payload_buffer(&self, offset: usize, extent: usize) -> DeviceBuffer {
        assert!(
            self.registered,
            "harmonic GPU mapping used after unregister"
        );
        assert!(offset.saturating_add(extent) <= self.mapping_bytes);
        // SAFETY: registration covers the entire device alias and `offset` plus
        // `extent` was range-checked against that allocation at construction.
        unsafe {
            DeviceBuffer::from_raw(
                self.device_base.as_ptr().cast::<u8>().add(offset).cast(),
                extent,
            )
        }
    }
}

impl Drop for HarmonicGpuMapping {
    fn drop(&mut self) {
        if self.registered {
            eprintln!(
                "deepseek4 harmonic GPU mapping dropped while still registered; process teardown will reclaim it"
            );
        }
    }
}

impl fmt::Debug for HarmonicSharedRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarmonicSharedRing")
            .field("bytes", &self.mmap.len())
            .field("contract", &self.contract())
            .finish()
    }
}

impl HarmonicSharedRing {
    pub fn create(file: &File, contract: HarmonicContract) -> HarmonicIpcResult<Self> {
        Self::create_with_integrity(file, contract, HarmonicIntegrityMode::Fingerprint)
    }

    /// Create the zero-checksum, release/acquire data plane used for
    /// performance composition. Exact payload comparison remains a promotion
    /// test; it is not repeated byte-by-byte on every model layer.
    pub fn create_data_plane(file: &File, contract: HarmonicContract) -> HarmonicIpcResult<Self> {
        Self::create_with_integrity(file, contract, HarmonicIntegrityMode::ReleaseAcquire)
    }

    fn create_with_integrity(
        file: &File,
        contract: HarmonicContract,
        integrity_mode: HarmonicIntegrityMode,
    ) -> HarmonicIpcResult<Self> {
        validate_frozen_contract(contract)?;
        let bytes = mem::size_of::<HarmonicWireLayout>();
        file.set_len(bytes as u64)?;
        // SAFETY: the file has just been resized to the exact mapping length and
        // remains open for the lifetime of map creation. The returned mapping
        // owns its VM reference independently of `file`.
        let mut mmap = unsafe { MmapOptions::new().len(bytes).map_mut(file)? };
        if !(mmap.as_ptr() as usize).is_multiple_of(mem::align_of::<HarmonicWireLayout>()) {
            return Err(HarmonicIpcError::InvalidLayout("mapping alignment"));
        }
        // SAFETY: the mapping is writable, correctly aligned, exactly large
        // enough, and has no typed aliases. Every field installed here is an
        // atomic, so later cross-process access never races through plain Rust
        // references.
        unsafe {
            (mmap.as_mut_ptr() as *mut HarmonicWireLayout)
                .write(HarmonicWireLayout::new(contract, integrity_mode));
        }
        mmap.flush()?;
        let ring = Self { mmap };
        ring.validate_layout()?;
        Ok(ring)
    }

    pub fn open(file: &File) -> HarmonicIpcResult<Self> {
        let bytes = mem::size_of::<HarmonicWireLayout>();
        if file.metadata()?.len() != bytes as u64 {
            return Err(HarmonicIpcError::InvalidLayout("file length"));
        }
        // SAFETY: the creator fixed the file to the exact shared-layout size.
        // Validation below rejects an uninitialized or foreign mapping before
        // any packet operation is admitted.
        let mmap = unsafe { MmapOptions::new().len(bytes).map_mut(file)? };
        if !(mmap.as_ptr() as usize).is_multiple_of(mem::align_of::<HarmonicWireLayout>()) {
            return Err(HarmonicIpcError::InvalidLayout("mapping alignment"));
        }
        let ring = Self { mmap };
        ring.validate_layout()?;
        Ok(ring)
    }

    pub fn contract(&self) -> HarmonicContract {
        self.layout().header.contract()
    }

    pub fn integrity_mode(&self) -> HarmonicIpcResult<HarmonicIntegrityMode> {
        HarmonicIntegrityMode::decode(self.layout().header.integrity_mode.load(Ordering::Acquire))
    }

    fn require_release_acquire_data_plane(&self) -> HarmonicIpcResult<()> {
        if self.integrity_mode()? != HarmonicIntegrityMode::ReleaseAcquire {
            return Err(HarmonicIpcError::InvalidLayout(
                "mapped payload requires release/acquire integrity mode",
            ));
        }
        Ok(())
    }

    pub fn publish(
        &self,
        packet: HarmonicRoutePacket,
        endpoint_generation: u64,
        now: u64,
        activation_payload: &[u8],
    ) -> HarmonicIpcResult<()> {
        validate_payload(
            self.integrity_mode()?,
            activation_payload,
            HARMONIC_ACTIVATION_EXTENT as usize,
            packet.activation_fingerprint,
            "activation payload",
        )?;
        let slot = self.prepare_publish(packet, endpoint_generation, now)?;
        write_payload(&slot.activation_payload, activation_payload);
        slot.state
            .store(HarmonicWireState::Published as u32, Ordering::Release);
        Ok(())
    }

    /// Publish control metadata after the exact activation extent has already
    /// been transferred into this epoch's HIP-registered payload view.
    pub fn publish_mapped(
        &self,
        packet: HarmonicRoutePacket,
        endpoint_generation: u64,
        now: u64,
    ) -> HarmonicIpcResult<()> {
        self.require_release_acquire_data_plane()?;
        let slot = self.prepare_publish(packet, endpoint_generation, now)?;
        slot.state
            .store(HarmonicWireState::Published as u32, Ordering::Release);
        Ok(())
    }

    /// Mirror the route metadata into the mapped activation payload after the
    /// source GPU has finished writing `x_rot`, but before release-publication.
    ///
    /// The packet atomics remain authoritative. These bytes preserve the fixed
    /// wire layout for exactness/debug tooling without making the gfx1151 hot
    /// path copy the 16 KiB activation through the CPU.
    pub fn write_mapped_activation_metadata(
        &self,
        epoch: u64,
        expert_ids: [u32; HARMONIC_TOP_K],
        route_weight_bits: [u32; HARMONIC_TOP_K],
    ) -> HarmonicIpcResult<()> {
        self.require_release_acquire_data_plane()?;
        let slot = self.slot(epoch);
        let base = slot.activation_payload.bytes.get().cast::<u8>();
        // SAFETY: only the dense source writes the activation payload. The
        // caller invokes this before `publish_mapped`, so the expert cannot
        // have acquired this epoch. All ranges are fixed constants within the
        // payload extent and do not overlap the GPU-written x_rot prefix.
        unsafe {
            for (index, value) in expert_ids.into_iter().enumerate() {
                ptr::copy_nonoverlapping(
                    value.to_le_bytes().as_ptr(),
                    base.add(HARMONIC_EXPERT_IDS_OFFSET + index * mem::size_of::<u32>()),
                    mem::size_of::<u32>(),
                );
            }
            for (index, value) in route_weight_bits.into_iter().enumerate() {
                ptr::copy_nonoverlapping(
                    value.to_le_bytes().as_ptr(),
                    base.add(HARMONIC_ROUTE_WEIGHTS_OFFSET + index * mem::size_of::<u32>()),
                    mem::size_of::<u32>(),
                );
            }
            ptr::write_bytes(
                base.add(HARMONIC_ACTIVATION_RESERVED_OFFSET),
                0,
                HARMONIC_ACTIVATION_EXTENT as usize - HARMONIC_ACTIVATION_RESERVED_OFFSET,
            );
        }
        Ok(())
    }

    /// Reserve a slot and write only a prefix of its activation payload.
    ///
    /// This deliberately leaves the slot in `Publishing`. It exists solely so
    /// the CPU process-isolation probe can kill the source between reservation
    /// and release-publication, then prove that confirmed process isolation can
    /// reclaim the abandoned slot. Product code cannot compile this method.
    #[cfg(feature = "harmonic-fault-injection")]
    #[doc(hidden)]
    pub fn fault_inject_partial_publish_for_probe(
        &self,
        packet: HarmonicRoutePacket,
        endpoint_generation: u64,
        now: u64,
        activation_payload: &[u8],
        payload_words: usize,
    ) -> HarmonicIpcResult<()> {
        validate_payload(
            self.integrity_mode()?,
            activation_payload,
            HARMONIC_ACTIVATION_EXTENT as usize,
            packet.activation_fingerprint,
            "activation payload",
        )?;
        let slot = self.prepare_publish(packet, endpoint_generation, now)?;
        write_payload_prefix(
            &slot.activation_payload,
            activation_payload,
            payload_words.min(ACTIVATION_WORDS),
        );
        Ok(())
    }

    fn prepare_publish(
        &self,
        packet: HarmonicRoutePacket,
        endpoint_generation: u64,
        now: u64,
    ) -> HarmonicIpcResult<&HarmonicWireSlot> {
        self.check_endpoint(HarmonicOwner::DenseGfx1100, endpoint_generation)?;
        let contract = self.contract();
        contract.validate(&packet, now)?;
        let slot = self.slot(packet.epoch);
        slot.transition(
            "reserve publish slot",
            HarmonicWireState::Vacant,
            HarmonicWireState::Publishing,
        )?;
        slot.flags.store(0, Ordering::Relaxed);
        // The expert polls epoch before state. Publishing the epoch with
        // release ordering guarantees that a peer which acquires the new
        // epoch cannot subsequently observe the terminal state left by the
        // prior occupant of this physical slot. The later Published release
        // remains the packet/payload visibility gate.
        slot.epoch.store(packet.epoch, Ordering::Release);
        slot.route_identity
            .store(packet.route_identity, Ordering::Relaxed);
        for (wire, value) in slot
            .model_identity
            .iter()
            .zip(identity_to_words(packet.model_identity))
        {
            wire.store(value, Ordering::Relaxed);
        }
        slot.layer.store(packet.layer.into(), Ordering::Relaxed);
        slot.slot.store(packet.slot.into(), Ordering::Relaxed);
        slot.source_owner
            .store(packet.source_owner as u32, Ordering::Relaxed);
        slot.destination_owner
            .store(packet.destination_owner as u32, Ordering::Relaxed);
        slot.source_generation
            .store(packet.source_allocation_generation, Ordering::Relaxed);
        slot.destination_generation
            .store(packet.destination_allocation_generation, Ordering::Relaxed);
        for (wire, value) in slot.expert_ids.iter().zip(packet.expert_ids) {
            wire.store(value, Ordering::Relaxed);
        }
        for (wire, value) in slot.route_weight_bits.iter().zip(packet.route_weight_bits) {
            wire.store(value, Ordering::Relaxed);
        }
        slot.activation_extent
            .store(packet.activation_extent, Ordering::Relaxed);
        slot.result_extent
            .store(packet.result_extent, Ordering::Relaxed);
        slot.deadline_tick
            .store(packet.deadline_tick, Ordering::Relaxed);
        slot.activation_fingerprint
            .store(packet.activation_fingerprint, Ordering::Relaxed);
        slot.result_fingerprint.store(0, Ordering::Relaxed);
        Ok(slot)
    }

    pub fn expert_begin(
        &self,
        epoch: u64,
        endpoint_generation: u64,
        now: u64,
    ) -> HarmonicIpcResult<HarmonicWorkItem> {
        self.check_endpoint(HarmonicOwner::ExpertGfx1151, endpoint_generation)?;
        let slot = self.slot(epoch);
        slot.check_epoch(epoch)?;
        let state = slot.state()?;
        if state != HarmonicWireState::Published {
            return Err(HarmonicIpcError::CompareExchange {
                operation: "expert begin",
                expected: HarmonicWireState::Published,
                actual: state,
            });
        }
        // The acquire state load above pairs with source publication before
        // any relaxed packet or payload word is consumed.
        let packet = slot.packet()?;
        self.contract().validate(&packet, now)?;
        slot.transition(
            "expert begin",
            HarmonicWireState::Published,
            HarmonicWireState::Running,
        )?;
        let activation_payload = read_payload(&slot.activation_payload);
        let integrity_mode = self.integrity_mode()?;
        validate_payload(
            integrity_mode,
            &activation_payload,
            HARMONIC_ACTIVATION_EXTENT as usize,
            packet.activation_fingerprint,
            "activation payload",
        )?;
        Ok(HarmonicWorkItem {
            packet,
            activation_payload,
            integrity_mode,
        })
    }

    /// Acquire one published packet without copying its registered activation
    /// payload through the CPU. The caller consumes [`HarmonicGpuMapping`]'s
    /// activation view only after this acquire transition succeeds.
    pub fn expert_begin_mapped(
        &self,
        epoch: u64,
        endpoint_generation: u64,
        now: u64,
    ) -> HarmonicIpcResult<HarmonicRoutePacket> {
        self.require_release_acquire_data_plane()?;
        self.check_endpoint(HarmonicOwner::ExpertGfx1151, endpoint_generation)?;
        let slot = self.slot(epoch);
        slot.check_epoch(epoch)?;
        let state = slot.state()?;
        if state != HarmonicWireState::Published {
            return Err(HarmonicIpcError::CompareExchange {
                operation: "mapped expert begin",
                expected: HarmonicWireState::Published,
                actual: state,
            });
        }
        let packet = slot.packet()?;
        self.contract().validate(&packet, now)?;
        slot.transition(
            "mapped expert begin",
            HarmonicWireState::Published,
            HarmonicWireState::Running,
        )?;
        Ok(packet)
    }

    /// Discover the next expert epoch directly from the shared ring.
    ///
    /// This is the product-shaped data-plane seam: no Unix-socket message,
    /// serialization, process lookup, or GPU operation occurs while the slot
    /// is idle. The worker may spin briefly and then apply its own idle
    /// backoff around this nonblocking probe.
    pub fn expert_poll(
        &self,
        epoch: u64,
        endpoint_generation: u64,
    ) -> HarmonicIpcResult<HarmonicExpertPoll> {
        self.check_endpoint(HarmonicOwner::ExpertGfx1151, endpoint_generation)?;
        let slot = self.slot(epoch);
        let observed_epoch = slot.epoch.load(Ordering::Acquire);
        // Epoch must be sampled before state. A state-first poll can read the
        // old occupant's Completed state, race a recycle+republish, then read
        // the requested epoch and misclassify the new Published packet as an
        // old terminal packet.
        let state = slot.state()?;
        if observed_epoch != epoch {
            // The persistent worker is allowed to advance to the next logical
            // epoch before the source has resolved and recycled the older
            // epoch occupying that physical slot. An older occupant is
            // therefore backpressure, not a stale-worker fault. Seeing a
            // newer epoch means this worker really did fall behind and must
            // fail closed rather than execute a skipped request.
            if state == HarmonicWireState::Vacant || observed_epoch < epoch {
                return Ok(HarmonicExpertPoll::Pending);
            }
            return Err(HarmonicProtocolError::StaleEpoch {
                expected: observed_epoch,
                got: epoch,
            }
            .into());
        }
        match state {
            HarmonicWireState::Published => self
                .expert_begin(epoch, endpoint_generation, harmonic_monotonic_tick()?)
                .map(HarmonicExpertPoll::Work),
            HarmonicWireState::Completed
            | HarmonicWireState::Cancelled
            | HarmonicWireState::TimedOut
            | HarmonicWireState::FailedDense
            | HarmonicWireState::FailedExpert => Ok(HarmonicExpertPoll::Terminal(state)),
            HarmonicWireState::Vacant
            | HarmonicWireState::Publishing
            | HarmonicWireState::Running
            | HarmonicWireState::Completing => Ok(HarmonicExpertPoll::Pending),
        }
    }

    pub fn expert_poll_mapped(
        &self,
        epoch: u64,
        endpoint_generation: u64,
    ) -> HarmonicIpcResult<HarmonicExpertMappedPoll> {
        self.require_release_acquire_data_plane()?;
        self.check_endpoint(HarmonicOwner::ExpertGfx1151, endpoint_generation)?;
        let slot = self.slot(epoch);
        let observed_epoch = slot.epoch.load(Ordering::Acquire);
        // Keep this paired with expert_poll above; mapped and copied payload
        // modes share the same double-buffer publication protocol.
        let state = slot.state()?;
        if observed_epoch != epoch {
            if state == HarmonicWireState::Vacant || observed_epoch < epoch {
                return Ok(HarmonicExpertMappedPoll::Pending);
            }
            return Err(HarmonicProtocolError::StaleEpoch {
                expected: observed_epoch,
                got: epoch,
            }
            .into());
        }
        match state {
            HarmonicWireState::Published => self
                .expert_begin_mapped(epoch, endpoint_generation, harmonic_monotonic_tick()?)
                .map(HarmonicExpertMappedPoll::Work),
            HarmonicWireState::Completed
            | HarmonicWireState::Cancelled
            | HarmonicWireState::TimedOut
            | HarmonicWireState::FailedDense
            | HarmonicWireState::FailedExpert => Ok(HarmonicExpertMappedPoll::Terminal(state)),
            HarmonicWireState::Vacant
            | HarmonicWireState::Publishing
            | HarmonicWireState::Running
            | HarmonicWireState::Completing => Ok(HarmonicExpertMappedPoll::Pending),
        }
    }

    pub fn expert_complete(
        &self,
        epoch: u64,
        endpoint_generation: u64,
        completion: HarmonicCompletion,
        result_payload: &[u8],
    ) -> HarmonicIpcResult<()> {
        self.check_endpoint(HarmonicOwner::ExpertGfx1151, endpoint_generation)?;
        let slot = self.slot(epoch);
        slot.check_epoch(epoch)?;
        let state = slot.state()?;
        if state != HarmonicWireState::Running {
            return Err(HarmonicIpcError::CompareExchange {
                operation: "reserve completion slot",
                expected: HarmonicWireState::Running,
                actual: state,
            });
        }
        let packet = slot.packet()?;
        if completion.result_extent != packet.result_extent {
            return Err(HarmonicProtocolError::InvalidPacket("completion extent").into());
        }
        validate_payload(
            self.integrity_mode()?,
            result_payload,
            HARMONIC_RESULT_EXTENT as usize,
            completion.result_fingerprint,
            "result payload",
        )?;
        slot.transition(
            "reserve completion slot",
            HarmonicWireState::Running,
            HarmonicWireState::Completing,
        )?;
        slot.result_fingerprint
            .store(completion.result_fingerprint, Ordering::Relaxed);
        write_payload(&slot.result_payload, result_payload);
        slot.flags
            .fetch_or(FLAG_DESTINATION_QUIESCED, Ordering::Release);
        slot.state
            .store(HarmonicWireState::Completed as u32, Ordering::Release);
        Ok(())
    }

    /// Release-complete one packet whose exact result extent has already been
    /// written through this process's HIP-registered result view.
    pub fn expert_complete_mapped(
        &self,
        epoch: u64,
        endpoint_generation: u64,
        completion: HarmonicCompletion,
    ) -> HarmonicIpcResult<()> {
        self.require_release_acquire_data_plane()?;
        self.check_endpoint(HarmonicOwner::ExpertGfx1151, endpoint_generation)?;
        let slot = self.slot(epoch);
        slot.check_epoch(epoch)?;
        let state = slot.state()?;
        if state != HarmonicWireState::Running {
            return Err(HarmonicIpcError::CompareExchange {
                operation: "reserve mapped completion slot",
                expected: HarmonicWireState::Running,
                actual: state,
            });
        }
        let packet = slot.packet()?;
        if completion.result_extent != packet.result_extent {
            return Err(HarmonicProtocolError::InvalidPacket("completion extent").into());
        }
        slot.transition(
            "reserve mapped completion slot",
            HarmonicWireState::Running,
            HarmonicWireState::Completing,
        )?;
        slot.result_fingerprint
            .store(completion.result_fingerprint, Ordering::Relaxed);
        slot.flags
            .fetch_or(FLAG_DESTINATION_QUIESCED, Ordering::Release);
        slot.state
            .store(HarmonicWireState::Completed as u32, Ordering::Release);
        Ok(())
    }

    pub fn source_resolve(
        &self,
        epoch: u64,
        endpoint_generation: u64,
    ) -> HarmonicIpcResult<HarmonicResolved> {
        self.check_endpoint(HarmonicOwner::DenseGfx1100, endpoint_generation)?;
        let slot = self.slot(epoch);
        slot.check_epoch(epoch)?;
        let state = slot.state()?;
        if !state.is_terminal() {
            return Err(HarmonicProtocolError::SlotNotTerminal {
                slot: (epoch as usize % HARMONIC_SLOT_COUNT) as u8,
                state: state.logical(),
            }
            .into());
        }
        let (completion, result_payload) = if state == HarmonicWireState::Completed {
            let completion = HarmonicCompletion {
                result_extent: slot.result_extent.load(Ordering::Relaxed),
                result_fingerprint: slot.result_fingerprint.load(Ordering::Relaxed),
            };
            let result_payload = read_payload(&slot.result_payload);
            validate_payload(
                self.integrity_mode()?,
                &result_payload,
                HARMONIC_RESULT_EXTENT as usize,
                completion.result_fingerprint,
                "result payload",
            )?;
            (Some(completion), Some(result_payload))
        } else {
            (None, None)
        };
        slot.flags.fetch_or(FLAG_SOURCE_OBSERVED, Ordering::Release);
        Ok(HarmonicResolved {
            state: state.logical(),
            completion,
            result_payload,
        })
    }

    /// Resolve terminal control state while leaving the registered result
    /// payload in place for an owner-local asynchronous GPU copy.
    pub fn source_resolve_mapped(
        &self,
        epoch: u64,
        endpoint_generation: u64,
    ) -> HarmonicIpcResult<HarmonicMappedResolved> {
        self.require_release_acquire_data_plane()?;
        self.check_endpoint(HarmonicOwner::DenseGfx1100, endpoint_generation)?;
        let slot = self.slot(epoch);
        slot.check_epoch(epoch)?;
        let state = slot.state()?;
        if !state.is_terminal() {
            return Err(HarmonicProtocolError::SlotNotTerminal {
                slot: (epoch as usize % HARMONIC_SLOT_COUNT) as u8,
                state: state.logical(),
            }
            .into());
        }
        let completion = (state == HarmonicWireState::Completed).then(|| HarmonicCompletion {
            result_extent: slot.result_extent.load(Ordering::Relaxed),
            result_fingerprint: slot.result_fingerprint.load(Ordering::Relaxed),
        });
        slot.flags.fetch_or(FLAG_SOURCE_OBSERVED, Ordering::Release);
        Ok(HarmonicMappedResolved {
            state: state.logical(),
            completion,
        })
    }

    pub fn source_cancel(&self, epoch: u64, endpoint_generation: u64) -> HarmonicIpcResult<()> {
        self.check_endpoint(HarmonicOwner::DenseGfx1100, endpoint_generation)?;
        let slot = self.slot(epoch);
        slot.check_epoch(epoch)?;
        let state = slot.state()?;
        if !matches!(
            state,
            HarmonicWireState::Published | HarmonicWireState::Running
        ) {
            return Err(HarmonicProtocolError::InvalidTransition {
                state: state.logical(),
                operation: "source cancel",
            }
            .into());
        }
        slot.transition("source cancel", state, HarmonicWireState::Cancelled)?;
        slot.flags.fetch_or(FLAG_SOURCE_OBSERVED, Ordering::Release);
        Ok(())
    }

    pub fn expert_acknowledge_terminal(
        &self,
        epoch: u64,
        endpoint_generation: u64,
    ) -> HarmonicIpcResult<()> {
        self.check_endpoint(HarmonicOwner::ExpertGfx1151, endpoint_generation)?;
        let slot = self.slot(epoch);
        slot.check_epoch(epoch)?;
        let state = slot.state()?;
        if !state.is_terminal() {
            return Err(HarmonicProtocolError::SlotNotTerminal {
                slot: (epoch as usize % HARMONIC_SLOT_COUNT) as u8,
                state: state.logical(),
            }
            .into());
        }
        slot.flags
            .fetch_or(FLAG_DESTINATION_QUIESCED, Ordering::Release);
        Ok(())
    }

    pub fn expire(&self, epoch: u64, now: u64) -> HarmonicIpcResult<bool> {
        let slot = self.slot(epoch);
        slot.check_epoch(epoch)?;
        let deadline = slot.deadline_tick.load(Ordering::Acquire);
        if now < deadline {
            return Ok(false);
        }
        let state = slot.state()?;
        if !matches!(
            state,
            HarmonicWireState::Published | HarmonicWireState::Running
        ) {
            return Ok(false);
        }
        slot.transition("expire", state, HarmonicWireState::TimedOut)?;
        Ok(true)
    }

    /// Record a confirmed process-isolation boundary. The caller must invoke
    /// this only after the worker has exited and can no longer issue memory or
    /// device writes for `endpoint_generation`.
    pub fn isolate_owner(
        &self,
        owner: HarmonicOwner,
        endpoint_generation: u64,
    ) -> HarmonicIpcResult<usize> {
        self.check_endpoint(owner, endpoint_generation)?;
        let mut failed = 0;
        for slot in &self.layout().slots {
            let state = slot.state()?;
            if state == HarmonicWireState::Vacant {
                continue;
            }
            match owner {
                HarmonicOwner::DenseGfx1100 => {
                    slot.flags.fetch_or(FLAG_SOURCE_OBSERVED, Ordering::Release);
                    if state == HarmonicWireState::Publishing {
                        // The release publication has not happened, so the
                        // destination cannot have acquired this slot.
                        slot.flags
                            .fetch_or(FLAG_DESTINATION_QUIESCED, Ordering::Release);
                    }
                }
                HarmonicOwner::ExpertGfx1151 => {
                    slot.flags
                        .fetch_or(FLAG_DESTINATION_QUIESCED, Ordering::Release);
                }
            }
            if matches!(
                state,
                HarmonicWireState::Publishing
                    | HarmonicWireState::Published
                    | HarmonicWireState::Running
                    | HarmonicWireState::Completing
            ) {
                let failed_state = match owner {
                    HarmonicOwner::DenseGfx1100 => HarmonicWireState::FailedDense,
                    HarmonicOwner::ExpertGfx1151 => HarmonicWireState::FailedExpert,
                };
                if slot
                    .transition("isolate owner", state, failed_state)
                    .is_ok()
                {
                    failed += 1;
                }
            }
        }
        self.layout()
            .header
            .isolated_owners
            .fetch_or(owner_mask(owner), Ordering::Release);
        Ok(failed)
    }

    pub fn advance_generation(
        &self,
        owner: HarmonicOwner,
        current_generation: u64,
        next_generation: u64,
    ) -> HarmonicIpcResult<()> {
        let isolated = self.layout().header.isolated_owners.load(Ordering::Acquire);
        if isolated & owner_mask(owner) == 0 {
            return Err(HarmonicProtocolError::WorkerAlreadyAvailable(owner).into());
        }
        let actual_generation = self.generation(owner).load(Ordering::Acquire);
        if actual_generation != current_generation {
            return Err(HarmonicProtocolError::StaleGeneration {
                owner,
                expected: actual_generation,
                got: current_generation,
            }
            .into());
        }
        if next_generation <= current_generation {
            return Err(HarmonicProtocolError::StaleGeneration {
                owner,
                expected: current_generation.saturating_add(1),
                got: next_generation,
            }
            .into());
        }
        self.generation(owner)
            .compare_exchange(
                current_generation,
                next_generation,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|actual| {
                HarmonicIpcError::Protocol(HarmonicProtocolError::StaleGeneration {
                    owner,
                    expected: actual,
                    got: current_generation,
                })
            })?;
        self.layout()
            .header
            .isolated_owners
            .fetch_and(!owner_mask(owner), Ordering::Release);
        Ok(())
    }

    pub fn recycle(&self, epoch: u64) -> HarmonicIpcResult<()> {
        let slot = self.slot(epoch);
        slot.check_epoch(epoch)?;
        let state = slot.state()?;
        if !state.is_terminal() {
            return Err(HarmonicProtocolError::SlotNotTerminal {
                slot: (epoch as usize % HARMONIC_SLOT_COUNT) as u8,
                state: state.logical(),
            }
            .into());
        }
        let flags = slot.flags.load(Ordering::Acquire);
        let source_observed = flags & FLAG_SOURCE_OBSERVED != 0;
        let destination_quiesced = flags & FLAG_DESTINATION_QUIESCED != 0;
        if !source_observed || !destination_quiesced {
            return Err(HarmonicProtocolError::TerminalNotQuiesced {
                slot: (epoch as usize % HARMONIC_SLOT_COUNT) as u8,
                epoch,
                state: state.logical(),
                source_observed,
                destination_quiesced,
            }
            .into());
        }
        slot.transition("recycle", state, HarmonicWireState::Vacant)?;
        Ok(())
    }

    pub fn state(&self, epoch: u64) -> HarmonicIpcResult<HarmonicWireState> {
        let slot = self.slot(epoch);
        slot.check_epoch(epoch)?;
        slot.state()
    }

    fn validate_layout(&self) -> HarmonicIpcResult<()> {
        let header = &self.layout().header;
        if header.magic.load(Ordering::Acquire) != HARMONIC_WIRE_MAGIC {
            return Err(HarmonicIpcError::InvalidLayout("magic"));
        }
        if header.version.load(Ordering::Acquire) != HARMONIC_WIRE_VERSION {
            return Err(HarmonicIpcError::InvalidLayout("version"));
        }
        if header.slot_count.load(Ordering::Acquire) != HARMONIC_SLOT_COUNT as u32 {
            return Err(HarmonicIpcError::InvalidLayout("slot count"));
        }
        HarmonicIntegrityMode::decode(header.integrity_mode.load(Ordering::Acquire))?;
        validate_frozen_contract(header.contract())
    }

    fn check_endpoint(
        &self,
        owner: HarmonicOwner,
        endpoint_generation: u64,
    ) -> HarmonicIpcResult<()> {
        let expected = self.generation(owner).load(Ordering::Acquire);
        if expected != endpoint_generation {
            return Err(HarmonicProtocolError::StaleGeneration {
                owner,
                expected,
                got: endpoint_generation,
            }
            .into());
        }
        if self.layout().header.isolated_owners.load(Ordering::Acquire) & owner_mask(owner) != 0 {
            return Err(HarmonicProtocolError::WorkerUnavailable(owner).into());
        }
        Ok(())
    }

    fn generation(&self, owner: HarmonicOwner) -> &AtomicU64 {
        match owner {
            HarmonicOwner::DenseGfx1100 => &self.layout().header.source_generation,
            HarmonicOwner::ExpertGfx1151 => &self.layout().header.destination_generation,
        }
    }

    fn slot(&self, epoch: u64) -> &HarmonicWireSlot {
        &self.layout().slots[epoch as usize % HARMONIC_SLOT_COUNT]
    }

    fn layout(&self) -> &HarmonicWireLayout {
        // SAFETY: `create` initializes, and `open` validates, an exactly sized
        // aligned mapping. The layout contains only atomics; no mutable Rust
        // references to mapped fields are ever created.
        unsafe { &*(self.mmap.as_ptr() as *const HarmonicWireLayout) }
    }
}

fn validate_frozen_contract(contract: HarmonicContract) -> HarmonicIpcResult<()> {
    if contract.route_identity != HARMONIC_ROUTE_IDENTITY {
        return Err(HarmonicIpcError::InvalidLayout("route identity"));
    }
    if contract.model_identity != DS4_MQ2R_0731_IDENTITY {
        return Err(HarmonicIpcError::InvalidLayout("model identity"));
    }
    if contract.source_allocation_generation == 0 || contract.destination_allocation_generation == 0
    {
        return Err(HarmonicIpcError::InvalidLayout("zero generation"));
    }
    Ok(())
}

fn decode_owner(raw: u32) -> HarmonicIpcResult<HarmonicOwner> {
    match raw {
        value if value == HarmonicOwner::DenseGfx1100 as u32 => Ok(HarmonicOwner::DenseGfx1100),
        value if value == HarmonicOwner::ExpertGfx1151 as u32 => Ok(HarmonicOwner::ExpertGfx1151),
        _ => Err(HarmonicIpcError::InvalidLayout("owner")),
    }
}

const fn owner_mask(owner: HarmonicOwner) -> u32 {
    match owner {
        HarmonicOwner::DenseGfx1100 => OWNER_DENSE_MASK,
        HarmonicOwner::ExpertGfx1151 => OWNER_EXPERT_MASK,
    }
}

pub fn harmonic_payload_fingerprint(payload: &[u8]) -> u64 {
    // Stable FNV-1a integrity tag for the CPU oracle. Product GPU transport may
    // replace this with a cheaper guard once exact byte parity is established.
    payload.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn validate_payload(
    integrity_mode: HarmonicIntegrityMode,
    payload: &[u8],
    expected_len: usize,
    expected_fingerprint: u64,
    field: &'static str,
) -> HarmonicIpcResult<()> {
    if payload.len() != expected_len {
        return Err(HarmonicProtocolError::InvalidPacket(field).into());
    }
    if integrity_mode == HarmonicIntegrityMode::Fingerprint
        && harmonic_payload_fingerprint(payload) != expected_fingerprint
    {
        return Err(HarmonicProtocolError::InvalidPacket(field).into());
    }
    Ok(())
}

fn write_payload<const N: usize>(wire: &HarmonicPayload<N>, payload: &[u8]) {
    debug_assert_eq!(payload.len(), N);
    write_payload_prefix(wire, payload, N);
}

fn write_payload_prefix<const N: usize>(wire: &HarmonicPayload<N>, payload: &[u8], words: usize) {
    debug_assert_eq!(payload.len(), N);
    let bytes = words.saturating_mul(mem::size_of::<u64>()).min(N);
    // SAFETY: the protocol grants this endpoint exclusive write ownership of
    // the payload until release-publication. Source and destination payloads
    // have distinct fixed writers, and `payload` has at least `bytes` bytes.
    unsafe {
        ptr::copy_nonoverlapping(payload.as_ptr(), wire.bytes.get().cast::<u8>(), bytes);
    }
}

fn read_payload<const N: usize>(wire: &HarmonicPayload<N>) -> Vec<u8> {
    let mut payload = vec![0_u8; N];
    // SAFETY: the caller first acquired Published or Completed, pairing with
    // the sole writer's release transition. The destination vector is exactly
    // N bytes and does not overlap the mapped payload.
    unsafe {
        ptr::copy_nonoverlapping(
            wire.bytes.get().cast::<u8>(),
            payload.as_mut_ptr(),
            payload.len(),
        );
    }
    payload
}

fn identity_to_words(identity: [u8; 32]) -> [u64; 4] {
    std::array::from_fn(|index| {
        let offset = index * 8;
        u64::from_le_bytes(identity[offset..offset + 8].try_into().unwrap())
    })
}

fn words_to_identity(words: [u64; 4]) -> [u8; 32] {
    let mut identity = [0_u8; 32];
    for (index, word) in words.into_iter().enumerate() {
        let offset = index * 8;
        identity[offset..offset + 8].copy_from_slice(&word.to_le_bytes());
    }
    identity
}

#[cfg(test)]
mod tests {
    use std::fs::{self, OpenOptions};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::harmonic::{HARMONIC_ACTIVATION_EXTENT, HARMONIC_RESULT_EXTENT};

    static TEST_NONCE: AtomicU64 = AtomicU64::new(0);

    struct TestRing {
        path: PathBuf,
        ring: HarmonicSharedRing,
    }

    impl TestRing {
        fn new(source_generation: u64, destination_generation: u64) -> Self {
            Self::new_with_integrity(source_generation, destination_generation, false)
        }

        fn new_data_plane(source_generation: u64, destination_generation: u64) -> Self {
            Self::new_with_integrity(source_generation, destination_generation, true)
        }

        fn new_with_integrity(
            source_generation: u64,
            destination_generation: u64,
            data_plane: bool,
        ) -> Self {
            let nonce = TEST_NONCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "hipfire-harmonic-ipc-{}-{nonce}",
                std::process::id()
            ));
            let file = OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            let contract = HarmonicContract::frozen(source_generation, destination_generation);
            let ring = if data_plane {
                HarmonicSharedRing::create_data_plane(&file, contract)
            } else {
                HarmonicSharedRing::create(&file, contract)
            }
            .unwrap();
            Self { path, ring }
        }
    }

    impl Drop for TestRing {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    fn payload(len: usize, seed: u64) -> Vec<u8> {
        (0..len)
            .map(|index| seed.wrapping_add(index as u64).rotate_left(7) as u8)
            .collect()
    }

    fn request(
        contract: HarmonicContract,
        epoch: u64,
        deadline: u64,
    ) -> (HarmonicRoutePacket, Vec<u8>) {
        let activation = payload(HARMONIC_ACTIVATION_EXTENT as usize, epoch);
        let packet = contract.packet(
            epoch,
            epoch as u16 % 43,
            [0, 1, 2, 3, 4, 5],
            [0.5f32, 0.4, 0.3, 0.2, 0.1, 0.05].map(f32::to_bits),
            deadline,
            harmonic_payload_fingerprint(&activation),
        );
        (packet, activation)
    }

    #[test]
    fn independent_mappings_complete_and_recycle_exact_packet() {
        let owner = TestRing::new(7, 11);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&owner.path)
            .unwrap();
        let peer = HarmonicSharedRing::open(&file).unwrap();
        let contract = owner.ring.contract();
        let (request, activation) = request(contract, 1, 100);
        owner.ring.publish(request, 7, 0, &activation).unwrap();
        let observed = peer.expert_begin(1, 11, 1).unwrap();
        assert_eq!(observed.packet, request);
        assert_eq!(observed.activation_payload, activation);
        let result = payload(HARMONIC_RESULT_EXTENT as usize, 99);
        peer.expert_complete(
            1,
            11,
            HarmonicCompletion {
                result_extent: HARMONIC_RESULT_EXTENT,
                result_fingerprint: harmonic_payload_fingerprint(&result),
            },
            &result,
        )
        .unwrap();
        let resolved = owner.ring.source_resolve(1, 7).unwrap();
        assert_eq!(resolved.state, HarmonicSlotState::Completed);
        assert_eq!(resolved.result_payload.unwrap(), result);
        owner.ring.recycle(1).unwrap();
        assert_eq!(owner.ring.state(1).unwrap(), HarmonicWireState::Vacant);
    }

    #[test]
    fn mapped_control_plane_completes_without_copying_payload_through_cpu() {
        let owner = TestRing::new_data_plane(7, 11);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&owner.path)
            .unwrap();
        let peer = HarmonicSharedRing::open(&file).unwrap();
        let (request, _) = request(owner.ring.contract(), 1, u64::MAX);

        owner
            .ring
            .write_mapped_activation_metadata(1, request.expert_ids, request.route_weight_bits)
            .unwrap();
        owner.ring.publish_mapped(request, 7, 0).unwrap();
        let mirrored = read_payload(&owner.ring.slot(1).activation_payload);
        for (index, expected) in request.expert_ids.into_iter().enumerate() {
            let offset = HARMONIC_EXPERT_IDS_OFFSET + index * mem::size_of::<u32>();
            assert_eq!(
                u32::from_le_bytes(mirrored[offset..offset + 4].try_into().unwrap()),
                expected
            );
        }
        for (index, expected) in request.route_weight_bits.into_iter().enumerate() {
            let offset = HARMONIC_ROUTE_WEIGHTS_OFFSET + index * mem::size_of::<u32>();
            assert_eq!(
                u32::from_le_bytes(mirrored[offset..offset + 4].try_into().unwrap()),
                expected
            );
        }
        let observed = match peer.expert_poll_mapped(1, 11).unwrap() {
            HarmonicExpertMappedPoll::Work(packet) => packet,
            state => panic!("expected mapped work, got {state:?}"),
        };
        assert_eq!(observed, request);

        let completion = HarmonicCompletion {
            result_extent: HARMONIC_RESULT_EXTENT,
            result_fingerprint: 0xfeed_face_dead_beef,
        };
        peer.expert_complete_mapped(1, 11, completion).unwrap();
        let resolved = owner.ring.source_resolve_mapped(1, 7).unwrap();
        assert_eq!(resolved.state, HarmonicSlotState::Completed);
        assert_eq!(resolved.completion, Some(completion));
        owner.ring.recycle(1).unwrap();
        assert_eq!(owner.ring.state(1).unwrap(), HarmonicWireState::Vacant);
    }

    #[test]
    fn expert_poll_discovers_work_without_a_control_message() {
        let owner = TestRing::new(7, 11);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&owner.path)
            .unwrap();
        let peer = HarmonicSharedRing::open(&file).unwrap();
        assert_eq!(
            peer.expert_poll(1, 11).unwrap(),
            HarmonicExpertPoll::Pending
        );

        let (request, activation) = request(owner.ring.contract(), 1, u64::MAX);
        owner.ring.publish(request, 7, 0, &activation).unwrap();
        let work = match peer.expert_poll(1, 11).unwrap() {
            HarmonicExpertPoll::Work(work) => work,
            state => panic!("expected work, got {state:?}"),
        };
        assert_eq!(work.packet, request);
        assert_eq!(work.activation_payload, activation);
    }

    #[test]
    fn expert_poll_observes_cancelled_epoch_for_local_acknowledgement() {
        let owner = TestRing::new(7, 11);
        let (request, activation) = request(owner.ring.contract(), 1, u64::MAX);
        owner.ring.publish(request, 7, 0, &activation).unwrap();
        owner.ring.source_cancel(1, 7).unwrap();
        assert_eq!(
            owner.ring.expert_poll(1, 11).unwrap(),
            HarmonicExpertPoll::Terminal(HarmonicWireState::Cancelled)
        );
        owner.ring.expert_acknowledge_terminal(1, 11).unwrap();
        owner.ring.recycle(1).unwrap();
    }

    #[test]
    fn expert_poll_treats_older_slot_occupant_as_backpressure() {
        let owner = TestRing::new(7, 11);
        let (request, activation) = request(owner.ring.contract(), 1, u64::MAX);
        owner.ring.publish(request, 7, 0, &activation).unwrap();
        owner.ring.source_cancel(1, 7).unwrap();

        // Epoch 3 maps to the same physical slot as epoch 1. The worker can
        // reach this poll while the source is still resolving epoch 1.
        assert_eq!(
            owner.ring.expert_poll(3, 11).unwrap(),
            HarmonicExpertPoll::Pending
        );
    }

    #[test]
    fn mapped_poll_acquires_republished_epoch_instead_of_old_terminal_state() {
        let owner = TestRing::new_data_plane(7, 11);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&owner.path)
            .unwrap();
        let peer = HarmonicSharedRing::open(&file).unwrap();

        let (first, _) = request(owner.ring.contract(), 1, u64::MAX);
        owner.ring.publish_mapped(first, 7, 0).unwrap();
        assert!(matches!(
            peer.expert_poll_mapped(1, 11).unwrap(),
            HarmonicExpertMappedPoll::Work(_)
        ));
        peer.expert_complete_mapped(
            1,
            11,
            HarmonicCompletion {
                result_extent: HARMONIC_RESULT_EXTENT,
                result_fingerprint: 0,
            },
        )
        .unwrap();
        owner.ring.source_resolve_mapped(1, 7).unwrap();
        owner.ring.recycle(1).unwrap();

        // Epoch 3 reuses epoch 1's physical slot. Once its release-publish is
        // visible, the peer must acquire Work for epoch 3 and must never
        // report epoch 1's terminal state under epoch 3's identity.
        let (next, _) = request(owner.ring.contract(), 3, u64::MAX);
        owner.ring.publish_mapped(next, 7, 0).unwrap();
        match peer.expert_poll_mapped(3, 11).unwrap() {
            HarmonicExpertMappedPoll::Work(packet) => assert_eq!(packet, next),
            state => panic!("expected republished work, got {state:?}"),
        }
    }

    #[test]
    fn monotonic_ticks_are_process_comparable_and_non_decreasing() {
        let first = harmonic_monotonic_tick().unwrap();
        let second = harmonic_monotonic_tick().unwrap();
        assert!(second >= first);
    }

    #[test]
    fn cancelled_running_slot_needs_destination_ack() {
        let owner = TestRing::new(7, 11);
        let contract = owner.ring.contract();
        let (request, activation) = request(contract, 1, 100);
        owner.ring.publish(request, 7, 0, &activation).unwrap();
        owner.ring.expert_begin(1, 11, 1).unwrap();
        owner.ring.source_cancel(1, 7).unwrap();
        assert!(owner.ring.recycle(1).is_err());
        owner.ring.expert_acknowledge_terminal(1, 11).unwrap();
        owner.ring.recycle(1).unwrap();
    }

    #[test]
    fn isolated_expert_and_source_observation_reclaim_failed_slot() {
        let owner = TestRing::new(7, 11);
        let contract = owner.ring.contract();
        let (request, activation) = request(contract, 1, 100);
        owner.ring.publish(request, 7, 0, &activation).unwrap();
        owner.ring.expert_begin(1, 11, 1).unwrap();
        assert_eq!(
            owner
                .ring
                .isolate_owner(HarmonicOwner::ExpertGfx1151, 11)
                .unwrap(),
            1
        );
        assert!(owner.ring.recycle(1).is_err());
        let resolved = owner.ring.source_resolve(1, 7).unwrap();
        assert_eq!(
            resolved.state,
            HarmonicSlotState::Failed(HarmonicOwner::ExpertGfx1151)
        );
        assert!(resolved.completion.is_none());
        owner.ring.recycle(1).unwrap();
    }

    #[test]
    fn generations_reject_stale_endpoints_and_invalid_payloads() {
        let owner = TestRing::new(7, 11);
        assert!(owner
            .ring
            .advance_generation(HarmonicOwner::ExpertGfx1151, 11, 12)
            .is_err());
        owner
            .ring
            .isolate_owner(HarmonicOwner::ExpertGfx1151, 11)
            .unwrap();
        owner
            .ring
            .advance_generation(HarmonicOwner::ExpertGfx1151, 11, 12)
            .unwrap();
        assert!(owner.ring.expert_begin(1, 11, 0).is_err());
        assert!(owner
            .ring
            .advance_generation(HarmonicOwner::ExpertGfx1151, 11, 13)
            .is_err());
        let (mut bad, activation) = request(owner.ring.contract(), 1, 100);
        bad.activation_extent = HARMONIC_ACTIVATION_EXTENT - 1;
        assert!(owner.ring.publish(bad, 7, 0, &activation).is_err());
    }

    #[test]
    fn timeout_cannot_be_recycled_until_both_sides_are_safe() {
        let owner = TestRing::new(7, 11);
        let contract = owner.ring.contract();
        let (request, activation) = request(contract, 1, 10);
        owner.ring.publish(request, 7, 0, &activation).unwrap();
        owner.ring.expert_begin(1, 11, 1).unwrap();
        assert!(owner.ring.expire(1, 10).unwrap());
        assert!(owner.ring.recycle(1).is_err());
        owner.ring.source_resolve(1, 7).unwrap();
        assert!(owner.ring.recycle(1).is_err());
        owner.ring.expert_acknowledge_terminal(1, 11).unwrap();
        owner.ring.recycle(1).unwrap();
    }
}
