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
pub const HARMONIC_EXPERT_BITMAP_WORDS: usize = HARMONIC_EXPERT_COUNT as usize / u64::BITS as usize;
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
pub const HARMONIC_HOTSET_ROUTE_IDENTITY: u64 = 0x4453_3448_4152_4d33; // DS4HARM3
pub const HARMONIC_ROUTE_SLOT_RESULT_BYTES: usize = HARMONIC_X_ROT_BYTES;
pub const HARMONIC_SPLIT_RESULT_EXTENT: usize = HARMONIC_TOP_K * HARMONIC_ROUTE_SLOT_RESULT_BYTES;
pub const HARMONIC_REMOTE_SOURCE_BIT: u32 = 1_u32 << 31;
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

/// One exact routed-expert payload in the frozen DeepSeek4 artifact.
///
/// A slot includes that expert's w1, w2, and w3 payloads. It is deliberately
/// layer-qualified: expert 17 in layer 4 and expert 17 in layer 5 are distinct
/// allocations and may have different residency decisions.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct HarmonicExpertSlot {
    pub layer: u16,
    pub expert: u16,
}

impl HarmonicExpertSlot {
    pub fn checked(layer: usize, expert: usize) -> Result<Self, HarmonicResidencyError> {
        if layer >= HARMONIC_LAYER_COUNT as usize || expert >= HARMONIC_EXPERT_COUNT as usize {
            return Err(HarmonicResidencyError::SlotOutOfRange { layer, expert });
        }
        Ok(Self {
            layer: layer as u16,
            expert: expert as u16,
        })
    }
}

/// Owner selected for one canonical top-k route slot.
///
/// The full routed model always remains resident on gfx1151. `DenseGfx1100`
/// means an exact byte-for-byte replica is also resident on gfx1100 and may be
/// executed there; it never transfers expert weights in the token loop.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HarmonicExpertExecutionOwner {
    DenseGfx1100,
    ExpertGfx1151,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HarmonicSplitResultError {
    InvalidMaskBits(u8),
    OverlappingOwners(u8),
    MissingOwners(u8),
    InvalidOutputExtent {
        owner: HarmonicExpertExecutionOwner,
        got: usize,
        expected: usize,
    },
    InvalidResidualExtent {
        got: usize,
        expected: usize,
    },
    InvalidRouteWeight(usize),
}

impl fmt::Display for HarmonicSplitResultError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "deepseek4 harmonic split result: {self:?}")
    }
}

impl std::error::Error for HarmonicSplitResultError {}

/// Canonical top-k slot ownership for the versioned DS4HARM3 result layout.
/// Each owner exposes a full six-row view so buffer addresses stay persistent;
/// the masks decide which rows are live and stale rows are never consumed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HarmonicSplitResultLayout {
    local_mask: u8,
    remote_mask: u8,
}

impl HarmonicSplitResultLayout {
    const VALID_MASK: u8 = (1_u8 << HARMONIC_TOP_K) - 1;

    pub fn checked(local_mask: u8, remote_mask: u8) -> Result<Self, HarmonicSplitResultError> {
        let unknown = (local_mask | remote_mask) & !Self::VALID_MASK;
        if unknown != 0 {
            return Err(HarmonicSplitResultError::InvalidMaskBits(unknown));
        }
        let overlap = local_mask & remote_mask;
        if overlap != 0 {
            return Err(HarmonicSplitResultError::OverlappingOwners(overlap));
        }
        let missing = Self::VALID_MASK & !(local_mask | remote_mask);
        if missing != 0 {
            return Err(HarmonicSplitResultError::MissingOwners(missing));
        }
        Ok(Self {
            local_mask,
            remote_mask,
        })
    }

    pub fn from_owners(
        owners: [HarmonicExpertExecutionOwner; HARMONIC_TOP_K],
    ) -> Result<Self, HarmonicSplitResultError> {
        let mut local_mask = 0_u8;
        let mut remote_mask = 0_u8;
        for (slot, owner) in owners.into_iter().enumerate() {
            let bit = 1_u8 << slot;
            match owner {
                HarmonicExpertExecutionOwner::DenseGfx1100 => local_mask |= bit,
                HarmonicExpertExecutionOwner::ExpertGfx1151 => remote_mask |= bit,
            }
        }
        Self::checked(local_mask, remote_mask)
    }

    pub const fn local_mask(self) -> u8 {
        self.local_mask
    }

    pub const fn remote_mask(self) -> u8 {
        self.remote_mask
    }

    pub const fn owner(self, slot: usize) -> Option<HarmonicExpertExecutionOwner> {
        if slot >= HARMONIC_TOP_K {
            return None;
        }
        if self.local_mask & (1_u8 << slot) != 0 {
            Some(HarmonicExpertExecutionOwner::DenseGfx1100)
        } else {
            Some(HarmonicExpertExecutionOwner::ExpertGfx1151)
        }
    }

    pub const fn local_count(self) -> usize {
        self.local_mask.count_ones() as usize
    }

    pub const fn remote_count(self) -> usize {
        self.remote_mask.count_ones() as usize
    }

    /// Pack selected expert IDs independently for each owner while preserving
    /// canonical route-slot order. `slot_sources` is consumed directly by the
    /// exact gfx1100 split-combine kernel: bit 31 selects the mapped remote
    /// rows and the low bits select the packed row within that owner.
    pub fn pack_route(self, expert_ids: [u32; HARMONIC_TOP_K]) -> HarmonicPackedExpertRoute {
        let mut local_expert_ids = [0_u32; HARMONIC_TOP_K];
        let mut remote_expert_ids = [0_u32; HARMONIC_TOP_K];
        let mut slot_sources = [0_u32; HARMONIC_TOP_K];
        let mut local_count = 0_usize;
        let mut remote_count = 0_usize;
        for (slot, expert) in expert_ids.into_iter().enumerate() {
            match self.owner(slot).expect("validated split-result slot") {
                HarmonicExpertExecutionOwner::DenseGfx1100 => {
                    local_expert_ids[local_count] = expert;
                    slot_sources[slot] = local_count as u32;
                    local_count += 1;
                }
                HarmonicExpertExecutionOwner::ExpertGfx1151 => {
                    remote_expert_ids[remote_count] = expert;
                    slot_sources[slot] = HARMONIC_REMOTE_SOURCE_BIT | remote_count as u32;
                    remote_count += 1;
                }
            }
        }
        HarmonicPackedExpertRoute {
            layout: self,
            local_expert_ids,
            remote_expert_ids,
            slot_sources,
            local_count: local_count as u8,
            remote_count: remote_count as u8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HarmonicPackedExpertRoute {
    pub layout: HarmonicSplitResultLayout,
    pub local_expert_ids: [u32; HARMONIC_TOP_K],
    pub remote_expert_ids: [u32; HARMONIC_TOP_K],
    pub slot_sources: [u32; HARMONIC_TOP_K],
    pub local_count: u8,
    pub remote_count: u8,
}

impl HarmonicPackedExpertRoute {
    pub const fn remote_result_extent(self) -> u32 {
        self.remote_count as u32 * HARMONIC_ROUTE_SLOT_RESULT_BYTES as u32
    }
}

/// CPU oracle for the existing deterministic `moe_down_combine_k8_batched`
/// arithmetic. The two owner buffers retain canonical route-slot positions;
/// this function chooses exactly one owner per slot and folds slots 0..5 with
/// fused multiply-add before performing the existing residual addition.
///
/// Product execution must prove its GPU join raw-bit identical to this oracle
/// and the unsplit kernel before DS4HARM3 admission.
pub fn combine_harmonic_split_route_results(
    layout: HarmonicSplitResultLayout,
    route_weight_bits: [u32; HARMONIC_TOP_K],
    local_outputs: &[f32],
    remote_outputs: &[f32],
    residual: &mut [f32],
) -> Result<(), HarmonicSplitResultError> {
    for (owner, outputs) in [
        (HarmonicExpertExecutionOwner::DenseGfx1100, local_outputs),
        (HarmonicExpertExecutionOwner::ExpertGfx1151, remote_outputs),
    ] {
        if outputs.len() != HARMONIC_TOP_K * HARMONIC_HIDDEN_SIZE {
            return Err(HarmonicSplitResultError::InvalidOutputExtent {
                owner,
                got: outputs.len(),
                expected: HARMONIC_TOP_K * HARMONIC_HIDDEN_SIZE,
            });
        }
    }
    if residual.len() != HARMONIC_HIDDEN_SIZE {
        return Err(HarmonicSplitResultError::InvalidResidualExtent {
            got: residual.len(),
            expected: HARMONIC_HIDDEN_SIZE,
        });
    }
    let weights = route_weight_bits.map(f32::from_bits);
    for (slot, weight) in weights.iter().copied().enumerate() {
        if !weight.is_finite() || weight < 0.0 {
            return Err(HarmonicSplitResultError::InvalidRouteWeight(slot));
        }
    }

    for (column, destination) in residual.iter_mut().enumerate() {
        let mut accumulator = 0.0_f32;
        for (slot, weight) in weights.iter().copied().enumerate() {
            let outputs = match layout.owner(slot).expect("validated split-result slot") {
                HarmonicExpertExecutionOwner::DenseGfx1100 => local_outputs,
                HarmonicExpertExecutionOwner::ExpertGfx1151 => remote_outputs,
            };
            accumulator =
                weight.mul_add(outputs[slot * HARMONIC_HIDDEN_SIZE + column], accumulator);
        }
        *destination += accumulator;
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HarmonicResidencyError {
    ZeroSlotBytes,
    EmptyRanking,
    BudgetTooSmall { budget_bytes: u64, slot_bytes: u64 },
    SlotOutOfRange { layer: usize, expert: usize },
    DuplicateSlot(HarmonicExpertSlot),
    RequiredBytesOverflow,
    InvalidManifest(String),
}

impl fmt::Display for HarmonicResidencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "deepseek4 harmonic residency: {self:?}")
    }
}

impl std::error::Error for HarmonicResidencyError {}

/// Frozen-model capacity plan for exact expert replicas on gfx1100.
///
/// The input is occurrence-ranked, but the selected slots are stored in
/// canonical `(layer, expert)` order and the identity is derived from the
/// bitmap. Therefore equal residency sets have equal identities even if a
/// profiler reports ties in a different order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarmonicExpertResidencyPlan {
    slot_bytes: u64,
    budget_bytes: u64,
    required_bytes: u64,
    identity: u64,
    slots: Vec<HarmonicExpertSlot>,
    by_layer: [[u64; HARMONIC_EXPERT_BITMAP_WORDS]; HARMONIC_LAYER_COUNT as usize],
}

impl HarmonicExpertResidencyPlan {
    pub fn from_manifest(text: &str) -> Result<Self, HarmonicResidencyError> {
        let mut lines = text.lines();
        if lines.next() != Some("DS4HOT01") {
            return Err(HarmonicResidencyError::InvalidManifest(
                "missing DS4HOT01 magic".to_owned(),
            ));
        }
        let parse_header = |line: Option<&str>, key: &str| -> Result<u64, HarmonicResidencyError> {
            let value = line
                .and_then(|line| line.strip_prefix(key))
                .ok_or_else(|| {
                    HarmonicResidencyError::InvalidManifest(format!("missing {key} header"))
                })?;
            value.parse::<u64>().map_err(|error| {
                HarmonicResidencyError::InvalidManifest(format!("invalid {key}: {error}"))
            })
        };
        let slot_bytes = parse_header(lines.next(), "slot_bytes=")?;
        let budget_bytes = parse_header(lines.next(), "budget_bytes=")?;
        let declared_slots =
            usize::try_from(parse_header(lines.next(), "slots=")?).map_err(|_| {
                HarmonicResidencyError::InvalidManifest("slot count overflow".to_owned())
            })?;
        let mut ranked = Vec::with_capacity(declared_slots);
        for (line_index, line) in lines.enumerate() {
            if line.is_empty() {
                continue;
            }
            let mut fields = line.split_ascii_whitespace();
            let layer = fields
                .next()
                .ok_or_else(|| {
                    HarmonicResidencyError::InvalidManifest(format!(
                        "line {} missing layer",
                        line_index + 5
                    ))
                })?
                .parse::<usize>()
                .map_err(|error| {
                    HarmonicResidencyError::InvalidManifest(format!(
                        "line {} invalid layer: {error}",
                        line_index + 5
                    ))
                })?;
            let expert = fields
                .next()
                .ok_or_else(|| {
                    HarmonicResidencyError::InvalidManifest(format!(
                        "line {} missing expert",
                        line_index + 5
                    ))
                })?
                .parse::<usize>()
                .map_err(|error| {
                    HarmonicResidencyError::InvalidManifest(format!(
                        "line {} invalid expert: {error}",
                        line_index + 5
                    ))
                })?;
            if fields.next().is_some() {
                return Err(HarmonicResidencyError::InvalidManifest(format!(
                    "line {} has extra fields",
                    line_index + 5
                )));
            }
            ranked.push(HarmonicExpertSlot::checked(layer, expert)?);
        }
        if ranked.len() != declared_slots {
            return Err(HarmonicResidencyError::InvalidManifest(format!(
                "manifest contains {} slots, declares {declared_slots}",
                ranked.len()
            )));
        }
        let plan = Self::from_ranked(slot_bytes, budget_bytes, ranked)?;
        if plan.slots().len() != declared_slots {
            return Err(HarmonicResidencyError::InvalidManifest(
                "budget truncates declared slots".to_owned(),
            ));
        }
        Ok(plan)
    }

    pub fn from_ranked<I>(
        slot_bytes: u64,
        budget_bytes: u64,
        ranked: I,
    ) -> Result<Self, HarmonicResidencyError>
    where
        I: IntoIterator<Item = HarmonicExpertSlot>,
    {
        if slot_bytes == 0 {
            return Err(HarmonicResidencyError::ZeroSlotBytes);
        }
        let capacity = usize::try_from(budget_bytes / slot_bytes)
            .unwrap_or(usize::MAX)
            .min(HARMONIC_LAYER_COUNT as usize * HARMONIC_EXPERT_COUNT as usize);
        if capacity == 0 {
            return Err(HarmonicResidencyError::BudgetTooSmall {
                budget_bytes,
                slot_bytes,
            });
        }

        let mut seen = [[0_u64; HARMONIC_EXPERT_BITMAP_WORDS]; HARMONIC_LAYER_COUNT as usize];
        let mut selected = Vec::with_capacity(capacity);
        let mut saw_any = false;
        for slot in ranked {
            saw_any = true;
            let checked = HarmonicExpertSlot::checked(slot.layer as usize, slot.expert as usize)?;
            let word = checked.expert as usize / u64::BITS as usize;
            let bit = 1_u64 << (checked.expert as usize % u64::BITS as usize);
            let entry = &mut seen[checked.layer as usize][word];
            if *entry & bit != 0 {
                return Err(HarmonicResidencyError::DuplicateSlot(checked));
            }
            *entry |= bit;
            if selected.len() < capacity {
                selected.push(checked);
            }
        }
        if !saw_any {
            return Err(HarmonicResidencyError::EmptyRanking);
        }

        selected.sort_unstable();
        let mut by_layer = [[0_u64; HARMONIC_EXPERT_BITMAP_WORDS]; HARMONIC_LAYER_COUNT as usize];
        for slot in &selected {
            let word = slot.expert as usize / u64::BITS as usize;
            by_layer[slot.layer as usize][word] |=
                1_u64 << (slot.expert as usize % u64::BITS as usize);
        }
        let required_bytes = slot_bytes
            .checked_mul(selected.len() as u64)
            .ok_or(HarmonicResidencyError::RequiredBytesOverflow)?;
        let identity = Self::fingerprint(slot_bytes, &by_layer);
        Ok(Self {
            slot_bytes,
            budget_bytes,
            required_bytes,
            identity,
            slots: selected,
            by_layer,
        })
    }

    fn fingerprint(
        slot_bytes: u64,
        by_layer: &[[u64; HARMONIC_EXPERT_BITMAP_WORDS]; HARMONIC_LAYER_COUNT as usize],
    ) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for byte in HARMONIC_HOTSET_ROUTE_IDENTITY
            .to_le_bytes()
            .into_iter()
            .chain(DS4_MQ2R_0731_IDENTITY)
            .chain(slot_bytes.to_le_bytes())
            .chain(
                by_layer
                    .iter()
                    .flatten()
                    .flat_map(|word| word.to_le_bytes()),
            )
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    pub fn contains(&self, layer: usize, expert: usize) -> bool {
        if layer >= HARMONIC_LAYER_COUNT as usize || expert >= HARMONIC_EXPERT_COUNT as usize {
            return false;
        }
        let word = expert / u64::BITS as usize;
        self.by_layer[layer][word] & (1_u64 << (expert % u64::BITS as usize)) != 0
    }

    pub fn partition_route(
        &self,
        layer: usize,
        expert_ids: [u32; HARMONIC_TOP_K],
    ) -> [HarmonicExpertExecutionOwner; HARMONIC_TOP_K] {
        expert_ids.map(|expert| {
            if self.contains(layer, expert as usize) {
                HarmonicExpertExecutionOwner::DenseGfx1100
            } else {
                HarmonicExpertExecutionOwner::ExpertGfx1151
            }
        })
    }

    pub fn split_result_layout(
        &self,
        layer: usize,
        expert_ids: [u32; HARMONIC_TOP_K],
    ) -> HarmonicSplitResultLayout {
        HarmonicSplitResultLayout::from_owners(self.partition_route(layer, expert_ids))
            .expect("a residency plan assigns exactly one owner to every route slot")
    }

    pub fn slots(&self) -> &[HarmonicExpertSlot] {
        &self.slots
    }

    pub fn experts_in_layer(&self, layer: usize) -> impl Iterator<Item = u32> + '_ {
        self.slots
            .iter()
            .filter(move |slot| slot.layer as usize == layer)
            .map(|slot| u32::from(slot.expert))
    }

    /// Compact pointer-table index used by the gfx1100 replica allocation.
    /// The plan's canonical `(layer, expert)` sort is also the upload order.
    pub fn compact_expert_index(&self, layer: usize, expert: u32) -> Option<u32> {
        self.experts_in_layer(layer)
            .position(|candidate| candidate == expert)
            .map(|index| index as u32)
    }

    pub const fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    pub const fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    pub const fn required_bytes(&self) -> u64 {
        self.required_bytes
    }

    pub const fn identity(&self) -> u64 {
        self.identity
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
    /// Zero for DS4HARM2. DS4HARM3 binds every packet to the exact immutable
    /// replica bitmap used by the dense owner.
    pub residency_identity: u64,
    /// Number of leading `expert_ids` entries executed by the destination.
    pub active_experts: u8,
    /// Canonical route slots executed locally on gfx1100 (DS4HARM3 only).
    pub local_mask: u8,
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
    pub residency_identity: u64,
    pub source_allocation_generation: u64,
    pub destination_allocation_generation: u64,
}

impl HarmonicContract {
    pub const fn frozen(source_generation: u64, destination_generation: u64) -> Self {
        Self {
            route_identity: HARMONIC_ROUTE_IDENTITY,
            model_identity: DS4_MQ2R_0731_IDENTITY,
            residency_identity: 0,
            source_allocation_generation: source_generation,
            destination_allocation_generation: destination_generation,
        }
    }

    pub const fn hotset(
        source_generation: u64,
        destination_generation: u64,
        residency_identity: u64,
    ) -> Self {
        Self {
            route_identity: HARMONIC_HOTSET_ROUTE_IDENTITY,
            model_identity: DS4_MQ2R_0731_IDENTITY,
            residency_identity,
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
            residency_identity: self.residency_identity,
            active_experts: HARMONIC_TOP_K as u8,
            local_mask: 0,
            expert_ids,
            route_weight_bits,
            activation_extent: HARMONIC_ACTIVATION_EXTENT,
            result_extent: HARMONIC_RESULT_EXTENT,
            deadline_tick,
            activation_fingerprint,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn split_packet(
        self,
        epoch: u64,
        layer: u16,
        packed: HarmonicPackedExpertRoute,
        route_weight_bits: [u32; HARMONIC_TOP_K],
        deadline_tick: u64,
    ) -> HarmonicRoutePacket {
        debug_assert_eq!(self.route_identity, HARMONIC_HOTSET_ROUTE_IDENTITY);
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
            residency_identity: self.residency_identity,
            active_experts: packed.remote_count,
            local_mask: packed.layout.local_mask(),
            expert_ids: packed.remote_expert_ids,
            route_weight_bits,
            activation_extent: HARMONIC_ACTIVATION_EXTENT,
            result_extent: packed.remote_result_extent(),
            deadline_tick,
            activation_fingerprint: 0,
        }
    }

    pub fn validate(self, packet: &HarmonicRoutePacket, now: u64) -> HarmonicResult<()> {
        if packet.route_identity != self.route_identity {
            return Err(HarmonicProtocolError::InvalidPacket("route identity"));
        }
        if packet.model_identity != self.model_identity {
            return Err(HarmonicProtocolError::InvalidPacket("model identity"));
        }
        if packet.residency_identity != self.residency_identity {
            return Err(HarmonicProtocolError::InvalidPacket("residency identity"));
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
        if packet.activation_extent != HARMONIC_ACTIVATION_EXTENT {
            return Err(HarmonicProtocolError::InvalidPacket("payload extent"));
        }
        if self.route_identity == HARMONIC_ROUTE_IDENTITY {
            if packet.active_experts as usize != HARMONIC_TOP_K
                || packet.local_mask != 0
                || packet.result_extent != HARMONIC_RESULT_EXTENT
            {
                return Err(HarmonicProtocolError::InvalidPacket("DS4HARM2 payload"));
            }
        } else if self.route_identity == HARMONIC_HOTSET_ROUTE_IDENTITY {
            let valid_mask = (1_u8 << HARMONIC_TOP_K) - 1;
            if packet.local_mask & !valid_mask != 0
                || packet.active_experts as usize
                    != HARMONIC_TOP_K - packet.local_mask.count_ones() as usize
                || packet.result_extent
                    != packet.active_experts as u32 * HARMONIC_ROUTE_SLOT_RESULT_BYTES as u32
            {
                return Err(HarmonicProtocolError::InvalidPacket("DS4HARM3 payload"));
            }
        } else {
            return Err(HarmonicProtocolError::InvalidPacket(
                "unsupported route identity",
            ));
        }
        if packet.deadline_tick <= now {
            return Err(HarmonicProtocolError::DeadlineExceeded {
                deadline: packet.deadline_tick,
                now,
            });
        }
        for (index, expert) in packet
            .expert_ids
            .iter()
            .copied()
            .take(packet.active_experts as usize)
            .enumerate()
        {
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
    fn residency_plan_is_capacity_bounded_canonical_and_route_typed() {
        let ranked = [
            HarmonicExpertSlot::checked(29, 70).unwrap(),
            HarmonicExpertSlot::checked(5, 225).unwrap(),
            HarmonicExpertSlot::checked(13, 23).unwrap(),
            HarmonicExpertSlot::checked(32, 72).unwrap(),
        ];
        let plan = HarmonicExpertResidencyPlan::from_ranked(100, 250, ranked).unwrap();
        assert_eq!(plan.required_bytes(), 200);
        assert_eq!(plan.budget_bytes(), 250);
        assert_eq!(
            plan.slots(),
            &[
                HarmonicExpertSlot::checked(5, 225).unwrap(),
                HarmonicExpertSlot::checked(29, 70).unwrap(),
            ]
        );
        assert!(plan.contains(5, 225));
        assert!(!plan.contains(13, 23));
        assert_eq!(
            plan.partition_route(5, [225, 23, 1, 2, 3, 4]),
            [
                HarmonicExpertExecutionOwner::DenseGfx1100,
                HarmonicExpertExecutionOwner::ExpertGfx1151,
                HarmonicExpertExecutionOwner::ExpertGfx1151,
                HarmonicExpertExecutionOwner::ExpertGfx1151,
                HarmonicExpertExecutionOwner::ExpertGfx1151,
                HarmonicExpertExecutionOwner::ExpertGfx1151,
            ]
        );
    }

    #[test]
    fn residency_identity_depends_on_the_selected_set_not_tie_order() {
        let a = HarmonicExpertSlot::checked(2, 9).unwrap();
        let b = HarmonicExpertSlot::checked(7, 11).unwrap();
        let first = HarmonicExpertResidencyPlan::from_ranked(10, 20, [a, b]).unwrap();
        let second = HarmonicExpertResidencyPlan::from_ranked(10, 20, [b, a]).unwrap();
        assert_eq!(first.identity(), second.identity());
        assert_eq!(first.slots(), second.slots());
    }

    #[test]
    fn residency_plan_rejects_undersized_empty_and_duplicate_inputs() {
        let slot = HarmonicExpertSlot::checked(0, 0).unwrap();
        assert!(matches!(
            HarmonicExpertResidencyPlan::from_ranked(10, 9, [slot]),
            Err(HarmonicResidencyError::BudgetTooSmall { .. })
        ));
        assert!(matches!(
            HarmonicExpertResidencyPlan::from_ranked(10, 10, []),
            Err(HarmonicResidencyError::EmptyRanking)
        ));
        assert!(matches!(
            HarmonicExpertResidencyPlan::from_ranked(10, 20, [slot, slot]),
            Err(HarmonicResidencyError::DuplicateSlot(duplicate)) if duplicate == slot
        ));
    }

    #[test]
    fn split_result_layout_rejects_overlap_gap_and_unknown_bits() {
        assert!(matches!(
            HarmonicSplitResultLayout::checked(0b00_0011, 0b00_0010),
            Err(HarmonicSplitResultError::OverlappingOwners(0b00_0010))
        ));
        assert!(matches!(
            HarmonicSplitResultLayout::checked(0b00_0011, 0b11_0000),
            Err(HarmonicSplitResultError::MissingOwners(0b00_1100))
        ));
        assert!(matches!(
            HarmonicSplitResultLayout::checked(0b100_0000, 0b11_1111),
            Err(HarmonicSplitResultError::InvalidMaskBits(0b100_0000))
        ));
    }

    #[test]
    fn split_result_oracle_preserves_monolithic_slot_order_raw_bits() {
        let layout = HarmonicSplitResultLayout::checked(0b01_0101, 0b10_1010).unwrap();
        let weights = [0.37_f32, 0.23, 0.19, 0.11, 0.07, 0.03];
        let weight_bits = weights.map(f32::to_bits);
        let mut all = vec![0.0_f32; HARMONIC_TOP_K * HARMONIC_HIDDEN_SIZE];
        let mut local = vec![f32::NAN; all.len()];
        let mut remote = vec![f32::NAN; all.len()];
        for slot in 0..HARMONIC_TOP_K {
            for column in 0..HARMONIC_HIDDEN_SIZE {
                let index = slot * HARMONIC_HIDDEN_SIZE + column;
                let value = ((slot * 17 + column * 13) as f32 - 4096.0) * 0.000_031_25;
                all[index] = value;
                match layout.owner(slot).unwrap() {
                    HarmonicExpertExecutionOwner::DenseGfx1100 => local[index] = value,
                    HarmonicExpertExecutionOwner::ExpertGfx1151 => remote[index] = value,
                }
            }
        }
        let initial = (0..HARMONIC_HIDDEN_SIZE)
            .map(|column| (column as f32 - 2048.0) * 0.000_122_070_31)
            .collect::<Vec<_>>();
        let mut expected = initial.clone();
        for (column, destination) in expected.iter_mut().enumerate() {
            let mut accumulator = 0.0_f32;
            for (slot, weight) in weights.iter().copied().enumerate() {
                accumulator =
                    weight.mul_add(all[slot * HARMONIC_HIDDEN_SIZE + column], accumulator);
            }
            *destination += accumulator;
        }
        let mut observed = initial;
        combine_harmonic_split_route_results(layout, weight_bits, &local, &remote, &mut observed)
            .unwrap();
        assert_eq!(
            observed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn residency_plan_emits_a_complete_split_result_layout() {
        let plan = HarmonicExpertResidencyPlan::from_ranked(
            10,
            20,
            [
                HarmonicExpertSlot::checked(5, 225).unwrap(),
                HarmonicExpertSlot::checked(5, 23).unwrap(),
            ],
        )
        .unwrap();
        let layout = plan.split_result_layout(5, [225, 11, 23, 12, 13, 14]);
        assert_eq!(layout.local_mask(), 0b00_0101);
        assert_eq!(layout.remote_mask(), 0b11_1010);
    }

    #[test]
    fn split_layout_packs_each_owner_without_changing_route_order() {
        let layout = HarmonicSplitResultLayout::checked(0b01_0101, 0b10_1010).unwrap();
        let packed = layout.pack_route([10, 11, 12, 13, 14, 15]);
        assert_eq!(packed.local_count, 3);
        assert_eq!(packed.remote_count, 3);
        assert_eq!(&packed.local_expert_ids[..3], &[10, 12, 14]);
        assert_eq!(&packed.remote_expert_ids[..3], &[11, 13, 15]);
        assert_eq!(
            packed.slot_sources,
            [
                0,
                HARMONIC_REMOTE_SOURCE_BIT,
                1,
                HARMONIC_REMOTE_SOURCE_BIT | 1,
                2,
                HARMONIC_REMOTE_SOURCE_BIT | 2,
            ]
        );
        assert_eq!(packed.remote_result_extent(), 3 * 16_384);
    }

    #[test]
    fn hotset_manifest_and_packet_bind_the_exact_residency_identity() {
        let manifest = "DS4HOT01\nslot_bytes=10\nbudget_bytes=20\nslots=2\n5 225\n5 23\n";
        let plan = HarmonicExpertResidencyPlan::from_manifest(manifest).unwrap();
        assert_eq!(plan.compact_expert_index(5, 23), Some(0));
        assert_eq!(plan.compact_expert_index(5, 225), Some(1));
        let packed = plan
            .split_result_layout(5, [225, 11, 23, 12, 13, 14])
            .pack_route([225, 11, 23, 12, 13, 14]);
        let contract = HarmonicContract::hotset(7, 11, plan.identity());
        let packet = contract.split_packet(1, 5, packed, route_weights(), 100);
        contract.validate(&packet, 1).unwrap();
        assert_eq!(packet.local_mask, 0b00_0101);
        assert_eq!(packet.active_experts, 4);
        assert_eq!(packet.result_extent, 4 * 16_384);
        let mut wrong = packet;
        wrong.residency_identity ^= 1;
        assert!(contract.validate(&wrong, 1).is_err());
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
