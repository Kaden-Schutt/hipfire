// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-first protocol for fault-contained DeepSeek V4 harmonic execution.
//!
//! This module does not create a HIP context, queue, allocation, or stream.
//! It defines the owner, generation, epoch, slot, deadline, and terminal-state
//! contract which H2 must satisfy before any GPU transport can be admitted.

use std::fmt;

pub const HARMONIC_ROUTE_IDENTITY: u64 = 0x4453_3448_4152_4d32; // DS4HARM2
pub const HARMONIC_SLOT_COUNT: usize = 2;
pub const HARMONIC_LAYER_COUNT: u16 = 43;
pub const HARMONIC_EXPERT_COUNT: u32 = 256;
pub const HARMONIC_TOP_K: usize = 6;
pub const HARMONIC_HIDDEN_SIZE: usize = 4_096;
pub const HARMONIC_MOE_INTERMEDIATE_SIZE: usize = 2_048;
pub const HARMONIC_X_ROT_BYTES: usize = HARMONIC_HIDDEN_SIZE * std::mem::size_of::<f32>();
pub const HARMONIC_EXPERT_IDS_OFFSET: usize = HARMONIC_X_ROT_BYTES;
pub const HARMONIC_ROUTE_WEIGHTS_OFFSET: usize =
    HARMONIC_EXPERT_IDS_OFFSET + HARMONIC_TOP_K * std::mem::size_of::<u32>();
pub const HARMONIC_ACTIVATION_RESERVED_OFFSET: usize =
    HARMONIC_ROUTE_WEIGHTS_OFFSET + HARMONIC_TOP_K * std::mem::size_of::<u32>();
pub const HARMONIC_ACTIVATION_EXTENT: u32 = (HARMONIC_ACTIVATION_RESERVED_OFFSET + 16) as u32;
pub const HARMONIC_RESULT_EXTENT: u32 = HARMONIC_X_ROT_BYTES as u32;
pub const DS4_MQ2R_0731_IDENTITY: [u8; 32] = [
    0xcb, 0xf2, 0xbb, 0xcf, 0xa3, 0xf4, 0x7b, 0x17, 0x12, 0xa0, 0x71, 0x83, 0x6b, 0x2c, 0x48, 0x23,
    0x2d, 0xad, 0x7d, 0xfb, 0x76, 0x38, 0x13, 0xa7, 0x20, 0xf7, 0xd3, 0x48, 0xa9, 0x31, 0x8c, 0xce,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum HarmonicOwner {
    DenseGfx1100 = 1,
    ExpertGfx1151 = 2,
}

impl HarmonicOwner {
    const fn index(self) -> usize {
        match self {
            Self::DenseGfx1100 => 0,
            Self::ExpertGfx1151 => 1,
        }
    }

    /// Exact architecture identity required for this owner. A broad gfx11
    /// family match is never sufficient for harmonic admission.
    pub const fn expected_arch(self) -> &'static str {
        match self {
            Self::DenseGfx1100 => "gfx1100",
            Self::ExpertGfx1151 => "gfx1151",
        }
    }

    pub fn validate_arch(self, actual: &str) -> HarmonicResult<()> {
        if actual.eq_ignore_ascii_case(self.expected_arch()) {
            Ok(())
        } else {
            Err(HarmonicProtocolError::WrongArchitecture {
                owner: self,
                expected: self.expected_arch(),
                actual: actual.to_owned(),
            })
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HarmonicSlotState {
    Vacant,
    Published,
    Running,
    Completed,
    Cancelled,
    TimedOut,
    Failed(HarmonicOwner),
}

impl HarmonicSlotState {
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Cancelled | Self::TimedOut | Self::Failed(_)
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HarmonicCompletion {
    pub result_extent: u32,
    pub result_fingerprint: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarmonicRoutePacket {
    pub route_identity: u64,
    pub model_identity: [u8; 32],
    pub epoch: u64,
    pub layer: u16,
    pub slot: u8,
    pub source_owner: HarmonicOwner,
    pub destination_owner: HarmonicOwner,
    pub source_allocation_generation: u64,
    pub destination_allocation_generation: u64,
    pub expert_ids: [u32; HARMONIC_TOP_K],
    /// IEEE-754 bits keep the protocol equality check raw-bit exact.
    pub route_weight_bits: [u32; HARMONIC_TOP_K],
    pub activation_extent: u32,
    pub result_extent: u32,
    /// Tick from the supervisor's monotonic clock domain. It is never wall time.
    pub deadline_tick: u64,
    /// CPU-oracle integrity tag. Product transport may replace this with guards.
    pub activation_fingerprint: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HarmonicContract {
    pub route_identity: u64,
    pub model_identity: [u8; 32],
    pub source_allocation_generation: u64,
    pub destination_allocation_generation: u64,
}

impl HarmonicContract {
    pub const fn frozen(source_generation: u64, destination_generation: u64) -> Self {
        Self {
            route_identity: HARMONIC_ROUTE_IDENTITY,
            model_identity: DS4_MQ2R_0731_IDENTITY,
            source_allocation_generation: source_generation,
            destination_allocation_generation: destination_generation,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn packet(
        self,
        epoch: u64,
        layer: u16,
        expert_ids: [u32; HARMONIC_TOP_K],
        route_weight_bits: [u32; HARMONIC_TOP_K],
        deadline_tick: u64,
        activation_fingerprint: u64,
    ) -> HarmonicRoutePacket {
        HarmonicRoutePacket {
            route_identity: self.route_identity,
            model_identity: self.model_identity,
            epoch,
            layer,
            slot: (epoch as usize % HARMONIC_SLOT_COUNT) as u8,
            source_owner: HarmonicOwner::DenseGfx1100,
            destination_owner: HarmonicOwner::ExpertGfx1151,
            source_allocation_generation: self.source_allocation_generation,
            destination_allocation_generation: self.destination_allocation_generation,
            expert_ids,
            route_weight_bits,
            activation_extent: HARMONIC_ACTIVATION_EXTENT,
            result_extent: HARMONIC_RESULT_EXTENT,
            deadline_tick,
            activation_fingerprint,
        }
    }

    pub fn validate(self, packet: &HarmonicRoutePacket, now: u64) -> HarmonicResult<()> {
        if packet.route_identity != self.route_identity {
            return Err(HarmonicProtocolError::InvalidPacket("route identity"));
        }
        if packet.model_identity != self.model_identity {
            return Err(HarmonicProtocolError::InvalidPacket("model identity"));
        }
        if packet.epoch == 0 {
            return Err(HarmonicProtocolError::InvalidPacket("zero epoch"));
        }
        if packet.layer >= HARMONIC_LAYER_COUNT {
            return Err(HarmonicProtocolError::InvalidPacket("layer out of range"));
        }
        if packet.slot as usize != packet.epoch as usize % HARMONIC_SLOT_COUNT {
            return Err(HarmonicProtocolError::InvalidPacket("epoch/slot mismatch"));
        }
        if packet.source_owner != HarmonicOwner::DenseGfx1100
            || packet.destination_owner != HarmonicOwner::ExpertGfx1151
        {
            return Err(HarmonicProtocolError::InvalidPacket("owner direction"));
        }
        if packet.source_allocation_generation != self.source_allocation_generation {
            return Err(HarmonicProtocolError::StaleGeneration {
                owner: HarmonicOwner::DenseGfx1100,
                expected: self.source_allocation_generation,
                got: packet.source_allocation_generation,
            });
        }
        if packet.destination_allocation_generation != self.destination_allocation_generation {
            return Err(HarmonicProtocolError::StaleGeneration {
                owner: HarmonicOwner::ExpertGfx1151,
                expected: self.destination_allocation_generation,
                got: packet.destination_allocation_generation,
            });
        }
        if packet.activation_extent != HARMONIC_ACTIVATION_EXTENT
            || packet.result_extent != HARMONIC_RESULT_EXTENT
        {
            return Err(HarmonicProtocolError::InvalidPacket("payload extent"));
        }
        if packet.deadline_tick <= now {
            return Err(HarmonicProtocolError::DeadlineExceeded {
                deadline: packet.deadline_tick,
                now,
            });
        }
        for (index, expert) in packet.expert_ids.iter().copied().enumerate() {
            if expert >= HARMONIC_EXPERT_COUNT {
                return Err(HarmonicProtocolError::InvalidPacket("expert out of range"));
            }
            if packet.expert_ids[..index].contains(&expert) {
                return Err(HarmonicProtocolError::InvalidPacket("duplicate expert"));
            }
        }
        let mut weight_sum = 0.0f32;
        for bits in packet.route_weight_bits {
            let weight = f32::from_bits(bits);
            if !weight.is_finite() || weight < 0.0 {
                return Err(HarmonicProtocolError::InvalidPacket("route weight"));
            }
            weight_sum += weight;
        }
        if !weight_sum.is_finite() || weight_sum <= 0.0 {
            return Err(HarmonicProtocolError::InvalidPacket("route weight sum"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HarmonicProtocolError {
    InvalidPacket(&'static str),
    WorkerUnavailable(HarmonicOwner),
    SlotBusy {
        slot: u8,
        state: HarmonicSlotState,
    },
    SlotNotTerminal {
        slot: u8,
        state: HarmonicSlotState,
    },
    TerminalNotQuiesced {
        slot: u8,
        epoch: u64,
        state: HarmonicSlotState,
        source_observed: bool,
        destination_quiesced: bool,
    },
    WorkerAlreadyAvailable(HarmonicOwner),
    StaleEpoch {
        expected: u64,
        got: u64,
    },
    StaleGeneration {
        owner: HarmonicOwner,
        expected: u64,
        got: u64,
    },
    WrongOwner {
        expected: HarmonicOwner,
        got: HarmonicOwner,
    },
    WrongArchitecture {
        owner: HarmonicOwner,
        expected: &'static str,
        actual: String,
    },
    DeadlineExceeded {
        deadline: u64,
        now: u64,
    },
    InvalidTransition {
        state: HarmonicSlotState,
        operation: &'static str,
    },
    EpochExhausted,
}

impl fmt::Display for HarmonicProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "deepseek4 harmonic protocol: {self:?}")
    }
}

impl std::error::Error for HarmonicProtocolError {}

pub type HarmonicResult<T> = Result<T, HarmonicProtocolError>;

/// Build the fixed harmonic activation payload without exposing a device
/// pointer. The first 16 KiB is the exact F32 FWHT-rotated activation, followed
/// by raw-bit mirrors of the six expert IDs and route weights. The final 16
/// bytes stay zero for forward-compatible guards.
pub fn pack_harmonic_activation_payload(
    x_rot_bytes: &[u8],
    expert_ids: [u32; HARMONIC_TOP_K],
    route_weight_bits: [u32; HARMONIC_TOP_K],
) -> HarmonicResult<Vec<u8>> {
    if x_rot_bytes.len() != HARMONIC_X_ROT_BYTES {
        return Err(HarmonicProtocolError::InvalidPacket(
            "activation x_rot extent",
        ));
    }
    let mut payload = vec![0_u8; HARMONIC_ACTIVATION_EXTENT as usize];
    payload[..HARMONIC_X_ROT_BYTES].copy_from_slice(x_rot_bytes);
    for (index, value) in expert_ids.into_iter().enumerate() {
        let offset = HARMONIC_EXPERT_IDS_OFFSET + index * std::mem::size_of::<u32>();
        payload[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (index, value) in route_weight_bits.into_iter().enumerate() {
        let offset = HARMONIC_ROUTE_WEIGHTS_OFFSET + index * std::mem::size_of::<u32>();
        payload[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    Ok(payload)
}

/// Validate the activation payload's self-description against the already
/// validated route packet and return only the F32 activation bytes. This makes
/// torn or mismatched control/payload publication fail before any H2D copy.
pub fn unpack_harmonic_x_rot<'a>(
    packet: &HarmonicRoutePacket,
    payload: &'a [u8],
) -> HarmonicResult<&'a [u8]> {
    if payload.len() != HARMONIC_ACTIVATION_EXTENT as usize {
        return Err(HarmonicProtocolError::InvalidPacket(
            "activation payload extent",
        ));
    }
    for (index, expected) in packet.expert_ids.into_iter().enumerate() {
        let offset = HARMONIC_EXPERT_IDS_OFFSET + index * std::mem::size_of::<u32>();
        let actual = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
        if actual != expected {
            return Err(HarmonicProtocolError::InvalidPacket(
                "activation expert mirror",
            ));
        }
    }
    for (index, expected) in packet.route_weight_bits.into_iter().enumerate() {
        let offset = HARMONIC_ROUTE_WEIGHTS_OFFSET + index * std::mem::size_of::<u32>();
        let actual = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
        if actual != expected {
            return Err(HarmonicProtocolError::InvalidPacket(
                "activation route-weight mirror",
            ));
        }
    }
    if payload[HARMONIC_ACTIVATION_RESERVED_OFFSET..]
        .iter()
        .any(|byte| *byte != 0)
    {
        return Err(HarmonicProtocolError::InvalidPacket(
            "activation reserved bytes",
        ));
    }
    Ok(&payload[..HARMONIC_X_ROT_BYTES])
}

#[derive(Clone, Debug)]
pub struct HarmonicSlot {
    index: u8,
    state: HarmonicSlotState,
    packet: Option<HarmonicRoutePacket>,
    completion: Option<HarmonicCompletion>,
    source_observed: bool,
    destination_quiesced: bool,
}

impl HarmonicSlot {
    pub const fn new(index: u8) -> Self {
        Self {
            index,
            state: HarmonicSlotState::Vacant,
            packet: None,
            completion: None,
            source_observed: false,
            destination_quiesced: false,
        }
    }

    pub const fn state(&self) -> HarmonicSlotState {
        self.state
    }

    pub fn packet(&self) -> Option<&HarmonicRoutePacket> {
        self.packet.as_ref()
    }

    fn epoch(&self) -> u64 {
        self.packet.as_ref().map_or(0, |packet| packet.epoch)
    }

    pub fn publish(
        &mut self,
        contract: HarmonicContract,
        packet: HarmonicRoutePacket,
        now: u64,
    ) -> HarmonicResult<()> {
        if self.state != HarmonicSlotState::Vacant {
            return Err(HarmonicProtocolError::SlotBusy {
                slot: self.index,
                state: self.state,
            });
        }
        contract.validate(&packet, now)?;
        if packet.slot != self.index {
            return Err(HarmonicProtocolError::InvalidPacket(
                "physical slot mismatch",
            ));
        }
        self.packet = Some(packet);
        self.completion = None;
        self.source_observed = false;
        self.destination_quiesced = false;
        self.state = HarmonicSlotState::Published;
        Ok(())
    }

    pub fn begin(&mut self, owner: HarmonicOwner, epoch: u64, now: u64) -> HarmonicResult<()> {
        self.check_epoch(epoch)?;
        if self.expire(now) {
            return Err(self.deadline_error(now));
        }
        let packet = self.packet.as_ref().unwrap();
        if owner != packet.destination_owner {
            return Err(HarmonicProtocolError::WrongOwner {
                expected: packet.destination_owner,
                got: owner,
            });
        }
        if self.state != HarmonicSlotState::Published {
            return Err(HarmonicProtocolError::InvalidTransition {
                state: self.state,
                operation: "begin",
            });
        }
        self.state = HarmonicSlotState::Running;
        Ok(())
    }

    pub fn complete(
        &mut self,
        owner: HarmonicOwner,
        epoch: u64,
        completion: HarmonicCompletion,
        now: u64,
    ) -> HarmonicResult<()> {
        self.check_epoch(epoch)?;
        if self.expire(now) {
            return Err(self.deadline_error(now));
        }
        let packet = self.packet.as_ref().unwrap();
        if owner != packet.destination_owner {
            return Err(HarmonicProtocolError::WrongOwner {
                expected: packet.destination_owner,
                got: owner,
            });
        }
        if self.state != HarmonicSlotState::Running {
            return Err(HarmonicProtocolError::InvalidTransition {
                state: self.state,
                operation: "complete",
            });
        }
        if completion.result_extent != packet.result_extent {
            return Err(HarmonicProtocolError::InvalidPacket("completion extent"));
        }
        self.completion = Some(completion);
        self.destination_quiesced = true;
        self.state = HarmonicSlotState::Completed;
        Ok(())
    }

    pub fn resolve(
        &mut self,
        owner: HarmonicOwner,
        epoch: u64,
        now: u64,
    ) -> HarmonicResult<(HarmonicSlotState, Option<HarmonicCompletion>)> {
        self.check_epoch(epoch)?;
        self.expire(now);
        let packet = self.packet.as_ref().unwrap();
        if owner != packet.source_owner {
            return Err(HarmonicProtocolError::WrongOwner {
                expected: packet.source_owner,
                got: owner,
            });
        }
        if self.state.is_terminal() {
            self.source_observed = true;
        }
        Ok((self.state, self.completion))
    }

    pub fn acknowledge_terminal(&mut self, owner: HarmonicOwner, epoch: u64) -> HarmonicResult<()> {
        self.check_epoch(epoch)?;
        let packet = self.packet.as_ref().unwrap();
        if owner != packet.destination_owner {
            return Err(HarmonicProtocolError::WrongOwner {
                expected: packet.destination_owner,
                got: owner,
            });
        }
        if !self.state.is_terminal() {
            return Err(HarmonicProtocolError::SlotNotTerminal {
                slot: self.index,
                state: self.state,
            });
        }
        self.destination_quiesced = true;
        Ok(())
    }

    pub fn cancel(&mut self, epoch: u64) -> HarmonicResult<()> {
        self.check_epoch(epoch)?;
        if !matches!(
            self.state,
            HarmonicSlotState::Published | HarmonicSlotState::Running
        ) {
            return Err(HarmonicProtocolError::InvalidTransition {
                state: self.state,
                operation: "cancel",
            });
        }
        self.state = HarmonicSlotState::Cancelled;
        // Cancellation is issued by the source-side supervisor. The
        // destination must still acknowledge quiescence (or be isolated).
        self.source_observed = true;
        Ok(())
    }

    pub fn expire(&mut self, now: u64) -> bool {
        if matches!(
            self.state,
            HarmonicSlotState::Published | HarmonicSlotState::Running
        ) && self
            .packet
            .as_ref()
            .is_some_and(|packet| now >= packet.deadline_tick)
        {
            self.state = HarmonicSlotState::TimedOut;
            return true;
        }
        false
    }

    pub fn isolate_owner(&mut self, owner: HarmonicOwner) -> bool {
        let Some(packet) = self.packet.as_ref() else {
            return false;
        };
        if packet.source_owner == owner {
            self.source_observed = true;
        } else if packet.destination_owner == owner {
            self.destination_quiesced = true;
        } else {
            return false;
        }
        if matches!(
            self.state,
            HarmonicSlotState::Published | HarmonicSlotState::Running
        ) {
            self.state = HarmonicSlotState::Failed(owner);
            return true;
        }
        false
    }

    pub fn recycle(&mut self, epoch: u64) -> HarmonicResult<()> {
        self.check_epoch(epoch)?;
        if !self.state.is_terminal() {
            return Err(HarmonicProtocolError::SlotNotTerminal {
                slot: self.index,
                state: self.state,
            });
        }
        if !self.source_observed || !self.destination_quiesced {
            return Err(HarmonicProtocolError::TerminalNotQuiesced {
                slot: self.index,
                epoch,
                state: self.state,
                source_observed: self.source_observed,
                destination_quiesced: self.destination_quiesced,
            });
        }
        self.state = HarmonicSlotState::Vacant;
        self.packet = None;
        self.completion = None;
        self.source_observed = false;
        self.destination_quiesced = false;
        Ok(())
    }

    fn check_epoch(&self, got: u64) -> HarmonicResult<()> {
        if self.packet.is_none() {
            return Err(HarmonicProtocolError::SlotBusy {
                slot: self.index,
                state: HarmonicSlotState::Vacant,
            });
        }
        let expected = self.epoch();
        if expected != got {
            return Err(HarmonicProtocolError::StaleEpoch { expected, got });
        }
        Ok(())
    }

    fn deadline_error(&self, now: u64) -> HarmonicProtocolError {
        HarmonicProtocolError::DeadlineExceeded {
            deadline: self
                .packet
                .as_ref()
                .map_or(now, |packet| packet.deadline_tick),
            now,
        }
    }
}

#[derive(Debug)]
pub struct HarmonicSupervisor {
    contract: HarmonicContract,
    slots: [HarmonicSlot; HARMONIC_SLOT_COUNT],
    worker_alive: [bool; 2],
    next_epoch: u64,
}

impl HarmonicSupervisor {
    pub const fn new(source_generation: u64, destination_generation: u64) -> Self {
        Self {
            contract: HarmonicContract::frozen(source_generation, destination_generation),
            slots: [HarmonicSlot::new(0), HarmonicSlot::new(1)],
            worker_alive: [true, true],
            next_epoch: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn publish(
        &mut self,
        layer: u16,
        expert_ids: [u32; HARMONIC_TOP_K],
        route_weight_bits: [u32; HARMONIC_TOP_K],
        activation_fingerprint: u64,
        now: u64,
        deadline_tick: u64,
    ) -> HarmonicResult<u64> {
        for owner in [HarmonicOwner::DenseGfx1100, HarmonicOwner::ExpertGfx1151] {
            if !self.worker_alive[owner.index()] {
                return Err(HarmonicProtocolError::WorkerUnavailable(owner));
            }
        }
        let epoch = self
            .next_epoch
            .checked_add(1)
            .ok_or(HarmonicProtocolError::EpochExhausted)?;
        let index = epoch as usize % HARMONIC_SLOT_COUNT;
        if self.slots[index].state().is_terminal() {
            let old_epoch = self.slots[index].epoch();
            self.slots[index].recycle(old_epoch)?;
        }
        let packet = self.contract.packet(
            epoch,
            layer,
            expert_ids,
            route_weight_bits,
            deadline_tick,
            activation_fingerprint,
        );
        self.slots[index].publish(self.contract, packet, now)?;
        self.next_epoch = epoch;
        Ok(epoch)
    }

    pub fn packet(&self, epoch: u64) -> HarmonicResult<&HarmonicRoutePacket> {
        let slot = &self.slots[epoch as usize % HARMONIC_SLOT_COUNT];
        slot.check_epoch(epoch)?;
        Ok(slot.packet().unwrap())
    }

    pub fn expert_begin(&mut self, epoch: u64, now: u64) -> HarmonicResult<()> {
        self.slot_mut(epoch)
            .begin(HarmonicOwner::ExpertGfx1151, epoch, now)
    }

    pub fn expert_complete(
        &mut self,
        epoch: u64,
        result_fingerprint: u64,
        now: u64,
    ) -> HarmonicResult<()> {
        self.slot_mut(epoch).complete(
            HarmonicOwner::ExpertGfx1151,
            epoch,
            HarmonicCompletion {
                result_extent: HARMONIC_RESULT_EXTENT,
                result_fingerprint,
            },
            now,
        )
    }

    pub fn dense_resolve(
        &mut self,
        epoch: u64,
        now: u64,
    ) -> HarmonicResult<(HarmonicSlotState, Option<HarmonicCompletion>)> {
        self.slot_mut(epoch)
            .resolve(HarmonicOwner::DenseGfx1100, epoch, now)
    }

    pub fn cancel(&mut self, epoch: u64) -> HarmonicResult<()> {
        self.slot_mut(epoch).cancel(epoch)
    }

    pub fn expert_acknowledge_terminal(&mut self, epoch: u64) -> HarmonicResult<()> {
        self.slot_mut(epoch)
            .acknowledge_terminal(HarmonicOwner::ExpertGfx1151, epoch)
    }

    pub fn expire(&mut self, now: u64) -> usize {
        let mut expired = 0;
        for slot in &mut self.slots {
            expired += usize::from(slot.expire(now));
        }
        expired
    }

    pub fn worker_exit(&mut self, owner: HarmonicOwner) -> usize {
        self.worker_alive[owner.index()] = false;
        let mut failed = 0;
        for slot in &mut self.slots {
            failed += usize::from(slot.isolate_owner(owner));
        }
        failed
    }

    pub fn worker_restart(
        &mut self,
        owner: HarmonicOwner,
        allocation_generation: u64,
    ) -> HarmonicResult<()> {
        if self.worker_alive[owner.index()] {
            return Err(HarmonicProtocolError::WorkerAlreadyAvailable(owner));
        }
        let current = match owner {
            HarmonicOwner::DenseGfx1100 => self.contract.source_allocation_generation,
            HarmonicOwner::ExpertGfx1151 => self.contract.destination_allocation_generation,
        };
        if allocation_generation <= current {
            return Err(HarmonicProtocolError::StaleGeneration {
                owner,
                expected: current.saturating_add(1),
                got: allocation_generation,
            });
        }
        match owner {
            HarmonicOwner::DenseGfx1100 => {
                self.contract.source_allocation_generation = allocation_generation;
            }
            HarmonicOwner::ExpertGfx1151 => {
                self.contract.destination_allocation_generation = allocation_generation;
            }
        }
        self.worker_alive[owner.index()] = true;
        Ok(())
    }

    pub fn state(&self, epoch: u64) -> HarmonicResult<HarmonicSlotState> {
        let slot = &self.slots[epoch as usize % HARMONIC_SLOT_COUNT];
        slot.check_epoch(epoch)?;
        Ok(slot.state())
    }

    fn slot_mut(&mut self, epoch: u64) -> &mut HarmonicSlot {
        &mut self.slots[epoch as usize % HARMONIC_SLOT_COUNT]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route_weights() -> [u32; HARMONIC_TOP_K] {
        [0.50f32, 0.40, 0.30, 0.25, 0.20, 0.15].map(f32::to_bits)
    }

    fn experts(seed: u32) -> [u32; HARMONIC_TOP_K] {
        std::array::from_fn(|index| (seed + index as u32) % HARMONIC_EXPERT_COUNT)
    }

    fn fingerprint(value: u64) -> u64 {
        value.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(17)
    }

    #[test]
    fn activation_payload_round_trips_and_mirrors_the_route() {
        let expert_ids = experts(17);
        let route_weight_bits = route_weights();
        let x_rot = (0..HARMONIC_X_ROT_BYTES)
            .map(|index| index.wrapping_mul(17) as u8)
            .collect::<Vec<_>>();
        let payload =
            pack_harmonic_activation_payload(&x_rot, expert_ids, route_weight_bits).unwrap();
        let packet =
            HarmonicContract::frozen(7, 11).packet(1, 0, expert_ids, route_weight_bits, 100, 42);
        assert_eq!(unpack_harmonic_x_rot(&packet, &payload).unwrap(), x_rot);
        assert!(payload[HARMONIC_ACTIVATION_RESERVED_OFFSET..]
            .iter()
            .all(|byte| *byte == 0));
    }

    #[test]
    fn activation_payload_rejects_control_mismatch_and_reserved_data() {
        let expert_ids = experts(17);
        let route_weight_bits = route_weights();
        let x_rot = vec![0_u8; HARMONIC_X_ROT_BYTES];
        let packet =
            HarmonicContract::frozen(7, 11).packet(1, 0, expert_ids, route_weight_bits, 100, 42);
        let mut mismatched =
            pack_harmonic_activation_payload(&x_rot, expert_ids, route_weight_bits).unwrap();
        mismatched[HARMONIC_EXPERT_IDS_OFFSET] ^= 1;
        assert!(matches!(
            unpack_harmonic_x_rot(&packet, &mismatched),
            Err(HarmonicProtocolError::InvalidPacket(
                "activation expert mirror"
            ))
        ));
        let mut reserved =
            pack_harmonic_activation_payload(&x_rot, expert_ids, route_weight_bits).unwrap();
        reserved[HARMONIC_ACTIVATION_RESERVED_OFFSET] = 1;
        assert!(matches!(
            unpack_harmonic_x_rot(&packet, &reserved),
            Err(HarmonicProtocolError::InvalidPacket(
                "activation reserved bytes"
            ))
        ));
    }

    #[test]
    fn ten_thousand_exact_cpu_chains_reuse_double_buffer_safely() {
        let mut supervisor = HarmonicSupervisor::new(7, 11);
        for chain in 0..10_000u64 {
            let now = chain * 10;
            let activation = fingerprint(chain);
            let epoch = supervisor
                .publish(
                    chain as u16 % HARMONIC_LAYER_COUNT,
                    experts(chain as u32),
                    route_weights(),
                    activation,
                    now,
                    now + 9,
                )
                .unwrap();
            assert_eq!(
                supervisor.packet(epoch).unwrap().activation_fingerprint,
                activation
            );
            supervisor.expert_begin(epoch, now + 1).unwrap();
            let result = fingerprint(activation);
            supervisor.expert_complete(epoch, result, now + 2).unwrap();
            let (state, completion) = supervisor.dense_resolve(epoch, now + 3).unwrap();
            assert_eq!(state, HarmonicSlotState::Completed);
            assert_eq!(completion.unwrap().result_fingerprint, result);
        }
    }

    #[test]
    fn packet_validation_rejects_owner_generation_extent_and_route_corruption() {
        let contract = HarmonicContract::frozen(7, 11);
        let base = contract.packet(1, 0, experts(0), route_weights(), 100, 42);
        let cases = [
            {
                let mut packet = base;
                packet.destination_owner = HarmonicOwner::DenseGfx1100;
                packet
            },
            {
                let mut packet = base;
                packet.destination_allocation_generation = 10;
                packet
            },
            {
                let mut packet = base;
                packet.activation_extent -= 1;
                packet
            },
            {
                let mut packet = base;
                packet.route_identity ^= 1;
                packet
            },
        ];
        for packet in cases {
            assert!(contract.validate(&packet, 0).is_err());
        }
    }

    #[test]
    fn owner_architecture_admission_is_exact_and_role_specific() {
        assert!(HarmonicOwner::DenseGfx1100.validate_arch("gfx1100").is_ok());
        assert!(HarmonicOwner::ExpertGfx1151
            .validate_arch("GFX1151")
            .is_ok());
        assert!(matches!(
            HarmonicOwner::DenseGfx1100.validate_arch("gfx1151"),
            Err(HarmonicProtocolError::WrongArchitecture {
                owner: HarmonicOwner::DenseGfx1100,
                expected: "gfx1100",
                ..
            })
        ));
        assert!(HarmonicOwner::ExpertGfx1151
            .validate_arch("gfx1101")
            .is_err());
    }

    #[test]
    fn timeout_is_terminal_and_late_completion_cannot_resurrect_it() {
        let mut supervisor = HarmonicSupervisor::new(1, 1);
        let epoch = supervisor
            .publish(0, experts(0), route_weights(), 7, 0, 10)
            .unwrap();
        supervisor.expert_begin(epoch, 1).unwrap();
        assert_eq!(supervisor.expire(10), 1);
        assert_eq!(
            supervisor.state(epoch).unwrap(),
            HarmonicSlotState::TimedOut
        );
        assert!(supervisor.expert_complete(epoch, 9, 11).is_err());
        assert_eq!(
            supervisor.state(epoch).unwrap(),
            HarmonicSlotState::TimedOut
        );
    }

    #[test]
    fn mid_copy_cancel_is_terminal_and_rejects_stale_completion() {
        let mut supervisor = HarmonicSupervisor::new(1, 1);
        let epoch = supervisor
            .publish(0, experts(0), route_weights(), 7, 0, 10)
            .unwrap();
        supervisor.expert_begin(epoch, 1).unwrap();
        supervisor.cancel(epoch).unwrap();
        assert!(supervisor.expert_complete(epoch, 9, 2).is_err());
        assert_eq!(
            supervisor.state(epoch).unwrap(),
            HarmonicSlotState::Cancelled
        );
    }

    #[test]
    fn either_worker_exit_resolves_inflight_epoch_and_requires_new_generation() {
        for owner in [HarmonicOwner::DenseGfx1100, HarmonicOwner::ExpertGfx1151] {
            let mut supervisor = HarmonicSupervisor::new(7, 11);
            let epoch = supervisor
                .publish(0, experts(0), route_weights(), 7, 0, 10)
                .unwrap();
            supervisor.expert_begin(epoch, 1).unwrap();
            assert_eq!(supervisor.worker_exit(owner), 1);
            assert_eq!(
                supervisor.state(epoch).unwrap(),
                HarmonicSlotState::Failed(owner)
            );
            assert!(supervisor
                .publish(1, experts(1), route_weights(), 8, 2, 12)
                .is_err());
            let next_generation = match owner {
                HarmonicOwner::DenseGfx1100 => 8,
                HarmonicOwner::ExpertGfx1151 => 12,
            };
            supervisor.worker_restart(owner, next_generation).unwrap();
            let next = supervisor
                .publish(1, experts(1), route_weights(), 8, 2, 12)
                .unwrap();
            let packet = supervisor.packet(next).unwrap();
            match owner {
                HarmonicOwner::DenseGfx1100 => {
                    assert_eq!(packet.source_allocation_generation, next_generation)
                }
                HarmonicOwner::ExpertGfx1151 => {
                    assert_eq!(packet.destination_allocation_generation, next_generation)
                }
            }
        }
    }

    #[test]
    fn stale_epoch_cannot_address_a_reused_physical_slot() {
        let mut supervisor = HarmonicSupervisor::new(1, 1);
        let first = supervisor
            .publish(0, experts(0), route_weights(), 1, 0, 10)
            .unwrap();
        supervisor.expert_begin(first, 1).unwrap();
        supervisor.expert_complete(first, 2, 2).unwrap();
        supervisor.dense_resolve(first, 2).unwrap();
        let second = supervisor
            .publish(1, experts(1), route_weights(), 3, 3, 13)
            .unwrap();
        supervisor.expert_begin(second, 4).unwrap();
        supervisor.expert_complete(second, 4, 5).unwrap();
        supervisor.dense_resolve(second, 5).unwrap();
        let third = supervisor
            .publish(2, experts(2), route_weights(), 5, 6, 16)
            .unwrap();
        assert_eq!(
            first as usize % HARMONIC_SLOT_COUNT,
            third as usize % HARMONIC_SLOT_COUNT
        );
        assert!(supervisor.packet(first).is_err());
        assert_eq!(
            supervisor.state(third).unwrap(),
            HarmonicSlotState::Published
        );
    }

    #[test]
    fn completed_slot_cannot_be_reused_until_source_observes_terminal_state() {
        let mut supervisor = HarmonicSupervisor::new(1, 1);
        let first = supervisor
            .publish(0, experts(0), route_weights(), 1, 0, 10)
            .unwrap();
        supervisor.expert_begin(first, 1).unwrap();
        supervisor.expert_complete(first, 2, 2).unwrap();

        let second = supervisor
            .publish(1, experts(1), route_weights(), 3, 3, 13)
            .unwrap();
        supervisor.expert_begin(second, 4).unwrap();
        supervisor.expert_complete(second, 4, 5).unwrap();
        supervisor.dense_resolve(second, 5).unwrap();

        assert!(matches!(
            supervisor.publish(2, experts(2), route_weights(), 5, 6, 16),
            Err(HarmonicProtocolError::TerminalNotQuiesced { epoch, .. }) if epoch == first
        ));
        let (state, completion) = supervisor.dense_resolve(first, 6).unwrap();
        assert_eq!(state, HarmonicSlotState::Completed);
        assert_eq!(completion.unwrap().result_fingerprint, 2);
        assert!(supervisor
            .publish(2, experts(2), route_weights(), 5, 6, 16)
            .is_ok());
    }

    #[test]
    fn failed_expert_requires_source_observation_before_reclaim() {
        let mut supervisor = HarmonicSupervisor::new(7, 11);
        let first = supervisor
            .publish(0, experts(0), route_weights(), 1, 0, 10)
            .unwrap();
        supervisor.expert_begin(first, 1).unwrap();
        assert_eq!(supervisor.worker_exit(HarmonicOwner::ExpertGfx1151), 1);
        supervisor
            .worker_restart(HarmonicOwner::ExpertGfx1151, 12)
            .unwrap();

        let second = supervisor
            .publish(1, experts(1), route_weights(), 3, 2, 12)
            .unwrap();
        supervisor.expert_begin(second, 3).unwrap();
        supervisor.expert_complete(second, 4, 4).unwrap();
        supervisor.dense_resolve(second, 4).unwrap();

        assert!(matches!(
            supervisor.publish(2, experts(2), route_weights(), 5, 5, 15),
            Err(HarmonicProtocolError::TerminalNotQuiesced {
                source_observed: false,
                destination_quiesced: true,
                ..
            })
        ));
        assert!(matches!(
            supervisor.dense_resolve(first, 5),
            Ok((
                HarmonicSlotState::Failed(HarmonicOwner::ExpertGfx1151),
                None
            ))
        ));

        let third = supervisor
            .publish(2, experts(2), route_weights(), 5, 5, 15)
            .unwrap();
        assert_eq!(
            first as usize % HARMONIC_SLOT_COUNT,
            third as usize % HARMONIC_SLOT_COUNT
        );
        assert_eq!(
            supervisor
                .packet(third)
                .unwrap()
                .destination_allocation_generation,
            12
        );
    }

    #[test]
    fn failed_source_requires_destination_ack_before_reclaim() {
        let mut supervisor = HarmonicSupervisor::new(7, 11);
        let first = supervisor
            .publish(0, experts(0), route_weights(), 1, 0, 10)
            .unwrap();
        supervisor.expert_begin(first, 1).unwrap();
        assert_eq!(supervisor.worker_exit(HarmonicOwner::DenseGfx1100), 1);
        supervisor
            .worker_restart(HarmonicOwner::DenseGfx1100, 8)
            .unwrap();

        let second = supervisor
            .publish(1, experts(1), route_weights(), 3, 2, 12)
            .unwrap();
        supervisor.expert_begin(second, 3).unwrap();
        supervisor.expert_complete(second, 4, 4).unwrap();
        supervisor.dense_resolve(second, 4).unwrap();

        assert!(matches!(
            supervisor.publish(2, experts(2), route_weights(), 5, 5, 15),
            Err(HarmonicProtocolError::TerminalNotQuiesced {
                source_observed: true,
                destination_quiesced: false,
                ..
            })
        ));
        supervisor.expert_acknowledge_terminal(first).unwrap();
        assert!(supervisor
            .publish(2, experts(2), route_weights(), 5, 5, 15)
            .is_ok());
    }

    #[test]
    fn restart_requires_a_prior_worker_exit_and_vacant_epoch_is_rejected() {
        let mut supervisor = HarmonicSupervisor::new(7, 11);
        assert!(matches!(
            supervisor.worker_restart(HarmonicOwner::DenseGfx1100, 8),
            Err(HarmonicProtocolError::WorkerAlreadyAvailable(
                HarmonicOwner::DenseGfx1100
            ))
        ));
        assert!(supervisor.packet(0).is_err());
        assert!(supervisor.state(0).is_err());
    }
}
