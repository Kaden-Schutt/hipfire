// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Emulated EP2 harness for the Qwen3.5 Frozen MoE path (STEP-002 Task 8).
//!
//! **Test-only.** Compiled only under the non-default `emulated-ep2-harness`
//! feature.  Production Qwen35 EP remains `Planned { owner: "AXIS-002" }`;
//! this module never constructs `EpArch::Qwen35`, never touches the daemon
//! admission, and does not weaken the Frozen multi-device refusal.
//!
//! The harness runs two logical expert-ownership partitions over ONE GPU by
//! staging, inside the existing single [`crate::store::Qwen35MoeResident`]
//! owner (one [`SingleWeightStoreBuilder`], one freeze):
//!
//! * a deterministic stride-2 [`EmulatedExpertPartitionPlan`] (expert `i` →
//!   rank `i % 2`, compact local slot `i / 2`);
//! * per layer, both rank-masked gate-up pointer tables (owned experts keep
//!   their canonical pointer; non-owned experts point at a zero dummy);
//! * one zero gate-up dummy per distinct routed gate-up dtype (a zero
//!   activation makes the real unmasked down path contribute zero, so the
//!   canonical down / down-AWQ / dtype-tag / Paro resources stay borrowed
//!   and unmasked).
//!
//! [`Qwen35MoeResident::bind_layer_ep2`] returns the existing borrowed
//! [`MoeFfnBindings`] with only the gate-up pointer-table cell ID overridden.
//! No raw tensors, no second ownership domain, no per-rank residents.

use super::*;

// ── Staging switch ──────────────────────────────────────────────────

/// Test-only EP2 staging switch for [`crate::store::build_frozen_moe_resident`].
///
/// The production wrapper passes [`Ep2Staging::NONE`] and is byte-identical
/// to the pre-harness behavior; the harness builder passes the partition
/// plan so the shared inner builder stages the rank tables and dummies
/// before the SAME freeze.
#[derive(Clone, Copy)]
pub(super) struct Ep2Staging<'a>(pub(super) Option<&'a EmulatedExpertPartitionPlan>);

impl<'a> Ep2Staging<'a> {
    pub(super) const NONE: Ep2Staging<'a> = Ep2Staging(None);

    pub(super) fn with_plan(plan: &'a EmulatedExpertPartitionPlan) -> Ep2Staging<'a> {
        Ep2Staging(Some(plan))
    }
}

// ── Partition plan ──────────────────────────────────────────────────

/// Deterministic two-rank expert partition for the emulated EP2 harness.
///
/// Every expert `i` in `0..num_experts` is owned by exactly one rank
/// (`0` or `1`) and has a **compact local slot** on that rank (dense
/// 0-based, in global expert order).  Validation enforces:
///
/// * exactly two ranks — both must own at least one expert;
/// * disjoint + complete ownership — every expert has exactly one owner
///   and no expert is assigned an unknown rank;
/// * compact slots — `local_slot_of` is dense per rank.
///
/// `stride2` is the canonical harness partition: `owner = i % 2`,
/// `local_slot = i / 2`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EmulatedExpertPartitionPlan {
    num_experts: usize,
    /// Global expert id → owning rank (0 or 1).
    owner: Vec<u8>,
    /// Global expert id → compact local slot on its rank.
    local_slot: Vec<usize>,
}

impl EmulatedExpertPartitionPlan {
    /// Deterministic stride-2 ownership: expert `i` → rank `i % 2`.
    pub(crate) fn stride2(num_experts: usize) -> Result<Self, String> {
        Self::from_assignment((0..num_experts).map(|i| (i % 2) as u8).collect())
    }

    /// Build a plan from an explicit per-expert rank assignment.
    ///
    /// `owner.len()` IS the expert count; every entry must be `0` or `1`
    /// and both ranks must own at least one expert.
    pub(crate) fn from_assignment(owner: Vec<u8>) -> Result<Self, String> {
        let num_experts = owner.len();
        if num_experts == 0 {
            return Err("EP2 partition plan requires at least one expert".into());
        }
        let mut counts = [0usize; 2];
        for (i, &rank) in owner.iter().enumerate() {
            if rank > 1 {
                return Err(format!(
                    "EP2 partition plan assigns expert {i} to unknown rank {rank}"
                ));
            }
            counts[rank as usize] += 1;
        }
        if counts[0] == 0 || counts[1] == 0 {
            return Err(format!(
                "EP2 partition plan must cover exactly two ranks, got rank0={} rank1={}",
                counts[0], counts[1]
            ));
        }
        // Compact local slots: dense 0-based per rank, in global expert order.
        let mut next = [0usize; 2];
        let mut local_slot = vec![0usize; num_experts];
        for (i, &rank) in owner.iter().enumerate() {
            local_slot[i] = next[rank as usize];
            next[rank as usize] += 1;
        }
        Ok(Self {
            num_experts,
            owner,
            local_slot,
        })
    }

    pub(crate) fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Owning rank of the global expert id, if in range.
    pub(crate) fn owner_of(&self, expert: usize) -> Option<u8> {
        self.owner.get(expert).copied()
    }

    /// Compact local slot of the global expert id on its rank, if in range.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "compact local-slot seam exercised by the emulated EP2 tests; consumed by the Phase 2B driver"
        )
    )]
    pub(crate) fn local_slot_of(&self, expert: usize) -> Option<usize> {
        self.local_slot.get(expert).copied()
    }

    /// Global expert ids owned by `rank`, in ascending order.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "rank expert enumeration exercised by the emulated EP2 tests; consumed by the Phase 2B driver"
        )
    )]
    pub(crate) fn rank_experts(&self, rank: u8) -> Vec<usize> {
        (0..self.num_experts)
            .filter(|&i| self.owner[i] == rank)
            .collect()
    }

    /// Number of experts owned by `rank`.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "rank local count exercised by the emulated EP2 tests; consumed by the Phase 2B driver"
        )
    )]
    pub(crate) fn rank_local_count(&self, rank: u8) -> usize {
        self.owner.iter().filter(|&&o| o == rank).count()
    }
}

// ── Masked pointer-table construction (pure, CPU-testable) ──────────

/// ID-only descriptor for one staged zero gate-up dummy.
///
/// Carries the cell ID plus exactly the metadata needed to validate the
/// dummy against its canonical same-dtype representative: the canonical
/// gate-up dtype, the representative's exact shape, and the
/// representative's exact allocation byte length.  No tensor, no raw
/// pointer, no ownership — all ownership stays in the single Frozen
/// store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Ep2DummyDescriptor<K> {
    pub(crate) key: K,
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<usize>,
    pub(crate) byte_len: usize,
}

/// Select the canonical representative layout per distinct gate-up dtype
/// (first-seen order) and reject any same-dtype layout disagreement.
///
/// Each input entry is `(shape, allocation byte length, dtype)` of one
/// expert's canonical gate-up tensor.  The returned per-dtype layout is
/// taken EXACTLY from the representative tensor — no quant layout is
/// inferred from shapes or dtypes.  A same-dtype tensor whose shape or
/// byte length differs from the representative is refused BEFORE any
/// dummy is shared, so a dummy can never cover two different encodings
/// of one dtype.
pub(super) fn ep2_representative_layouts(
    per_expert: &[(Vec<usize>, usize, DType)],
) -> Result<Vec<(DType, Vec<usize>, usize)>, String> {
    let mut reps: Vec<(DType, Vec<usize>, usize)> = Vec::new();
    for (i, (shape, byte_len, dtype)) in per_expert.iter().enumerate() {
        match reps.iter_mut().find(|(rdt, _, _)| rdt == dtype) {
            Some((_, rep_shape, rep_len)) => {
                if rep_shape != shape {
                    return Err(format!(
                        "same-dtype gate-up layout mismatch for {dtype:?}: expert {i} has \
                         shape {shape:?} but the representative has {rep_shape:?}"
                    ));
                }
                if *rep_len != *byte_len {
                    return Err(format!(
                        "same-dtype gate-up layout mismatch for {dtype:?}: expert {i} has \
                         {byte_len} allocated bytes but the representative has {rep_len}"
                    ));
                }
            }
            None => reps.push((*dtype, shape.clone(), *byte_len)),
        }
    }
    Ok(reps)
}

/// Build both rank-masked gate-up pointer tables as raw `u64` byte payloads.
///
/// For rank `r`, expert `i` keeps its canonical `gate_up_addrs[i]` when the
/// plan assigns it to `r`; otherwise the slot points at the zero dummy for
/// `gate_up_dtypes[i]` (resolved through `dummy_addr`).  A masked expert
/// whose dtype has no staged dummy is an error — never a fabricated pointer.
pub(super) fn ep2_masked_gate_up_table_bytes(
    gate_up_addrs: &[u64],
    gate_up_dtypes: &[DType],
    plan: &EmulatedExpertPartitionPlan,
    dummy_addr: impl Fn(DType) -> Option<u64>,
) -> Result<[Vec<u8>; 2], String> {
    let n = gate_up_addrs.len();
    if gate_up_dtypes.len() != n {
        return Err(format!(
            "EP2 gate-up dtype vector length {} != expert count {n}",
            gate_up_dtypes.len()
        ));
    }
    if plan.num_experts() != n {
        return Err(format!(
            "EP2 partition plan covers {} experts but layer has {n}",
            plan.num_experts()
        ));
    }
    let mut tables: [Vec<u8>; 2] = [Vec::with_capacity(n * 8), Vec::with_capacity(n * 8)];
    for rank in 0..2u8 {
        for i in 0..n {
            let addr = if plan.owner_of(i) == Some(rank) {
                gate_up_addrs[i]
            } else {
                dummy_addr(gate_up_dtypes[i]).ok_or_else(|| {
                    format!(
                        "no EP2 zero dummy staged for gate-up dtype {:?} \
                         (expert {i}, rank {rank})",
                        gate_up_dtypes[i]
                    )
                })?
            };
            tables[rank as usize].extend_from_slice(&addr.to_ne_bytes());
        }
    }
    Ok(tables)
}

/// Cell-ID override for one rank's gate-up pointer table, or `None` when the
/// projection carries no EP2 tables for that rank (out of range or absent).
pub(super) fn ep2_rank_gate_up_ptrs_key<K: Copy>(
    proj: &Qwen35MoeLayerProjection<K>,
    rank: usize,
) -> Option<K> {
    proj.ep2_gate_up_ptrs
        .get(rank)
        .and_then(|d| d.as_ref().map(|d| d.key))
}

// ── Per-layer staged EP2 cells ──────────────────────────────────────

/// Cell IDs staged for one MoE layer's emulated EP2 partitions.
pub(super) struct Ep2LayerStaged {
    pub(super) rank_gate_up_ptrs: [WeightCellId; 2],
    /// One ID-only dummy descriptor per distinct canonical gate-up dtype,
    /// in first-seen dtype order.  Published onto the layer projection so
    /// `try_new_with_ep2` can validate every dummy against its canonical
    /// representative after freeze.
    pub(super) dummies: Vec<Ep2DummyDescriptor<WeightCellId>>,
}

/// Stage one layer's EP2 resources into the shared builder: one zero dummy
/// per distinct gate-up dtype, then both rank-masked gate-up pointer tables.
///
/// Each dummy is an EXACT clone of its canonical same-dtype representative
/// tensor: `representative.buf.size()` zero bytes, the representative's
/// exact shape, and the representative's dtype.  Same-dtype gate-up tensors
/// whose shape or byte length differs from the representative are refused
/// before any dummy is shared — a dummy never covers two different
/// encodings of one dtype.  No quant layout is inferred; all sizes come
/// from the representative tensor.
pub(super) fn stage_ep2_layer(
    builder: &mut SingleWeightStoreBuilder<'_>,
    cells: &FrozenMoeLayerCells,
    plan: &EmulatedExpertPartitionPlan,
    n: usize,
) -> Result<Ep2LayerStaged, String> {
    if plan.num_experts() != n {
        return Err(format!(
            "EP2 partition plan covers {} experts but layer {} has {n}",
            plan.num_experts(),
            cells.model_layer
        ));
    }

    // Per-expert canonical gate-up addresses, dtypes, and EXACT encoded
    // layouts (read-only builder borrows).
    let mut addrs = Vec::with_capacity(n);
    let mut dtypes = Vec::with_capacity(n);
    let mut layouts = Vec::with_capacity(n);
    for i in 0..n {
        let t = builder.tensor(cells.expert_gate_up[i]).map_err(|e| {
            format!(
                "layer {} expert.{i}.gate_up: builder.tensor failed: {e:?}",
                cells.model_layer
            )
        })?;
        addrs.push(t.buf.as_ptr() as u64);
        dtypes.push(t.dtype);
        layouts.push((t.shape.clone(), t.buf.size(), t.dtype));
    }

    // One zero dummy per distinct gate-up dtype, sized exactly like the
    // canonical representative.  DType is not Hash, so the small distinct
    // set (1-3 entries per layer) is kept as a Vec and scanned.
    let reps = ep2_representative_layouts(&layouts)?;
    let mut dummy_ids: Vec<(DType, WeightCellId)> = Vec::new();
    let mut dummies: Vec<Ep2DummyDescriptor<WeightCellId>> = Vec::new();
    for (dt, shape, byte_len) in reps {
        let zero = vec![0u8; byte_len];
        let id = builder
            .stage_derived(
                format!("layer_{}.ep2_dummy_{dt:?}", cells.model_layer),
                None,
                &zero,
                &shape,
                dt,
                WeightProjection::default(),
            )
            .map_err(|e| {
                format!(
                    "layer {} EP2 dummy staging for {dt:?} failed: {e}",
                    cells.model_layer
                )
            })?;
        dummy_ids.push((dt, id));
        dummies.push(Ep2DummyDescriptor {
            key: id,
            dtype: dt,
            shape,
            byte_len,
        });
    }

    // Masked rank tables.
    let dummy_addr = |dt: DType| -> Option<u64> {
        dummy_ids
            .iter()
            .find(|(staged_dt, _)| *staged_dt == dt)
            .and_then(|(_, id)| builder.tensor(*id).ok())
            .map(|t| t.buf.as_ptr() as u64)
    };
    let tables = ep2_masked_gate_up_table_bytes(&addrs, &dtypes, plan, dummy_addr)?;

    // Stage both tables.
    let mut rank_gate_up_ptrs = Vec::with_capacity(2);
    for (rank, bytes) in tables.into_iter().enumerate() {
        let id = builder
            .stage_derived(
                format!("layer_{}.ep2_gu_ptrs_rank{rank}", cells.model_layer),
                None,
                &bytes,
                &[n * 8],
                DType::Raw,
                WeightProjection::default(),
            )
            .map_err(|e| {
                format!(
                    "layer {} EP2 rank{rank} gate_up_ptrs staging failed: {e}",
                    cells.model_layer
                )
            })?;
        rank_gate_up_ptrs.push(id);
    }

    Ok(Ep2LayerStaged {
        rank_gate_up_ptrs: [rank_gate_up_ptrs[0], rank_gate_up_ptrs[1]],
        dummies,
    })
}

// ── Resident-side EP2 surface ───────────────────────────────────────

/// Pure EP2 projection validation (GPU-free, CPU-testable).
///
/// `resolve(key)` returns `(dtype, shape, allocation byte length)` for the
/// tensor named by `key`, or `None` for an invalid/foreign key.  Checks:
///
/// 1. Both rank gate-up pointer tables are present, resolve to Raw
///    `[num_experts * 8]` store cells, have a live allocation byte length of
///    EXACTLY `num_experts * 8`, and do not alias each other or the
///    canonical `proj.gate_up_ptrs.key` cell.
/// 2. The canonical gate-up tensors' distinct dtypes each have EXACTLY ONE
///    staged dummy descriptor (missing → [`Qwen35MoeValidationError::Ep2DummyMissing`],
///    duplicate/stray dtype → [`Qwen35MoeValidationError::Ep2DummyDuplicate`]).
/// 3. Every dummy descriptor's cell ID resolves (invalid/foreign →
///    [`Qwen35MoeValidationError::Ep2DummyInvalidId`]) and its store tensor
///    matches the descriptor's dtype / shape / byte length, which in turn
///    must match the canonical same-dtype representative exactly.
/// 4. Same-dtype canonical gate-up tensors agree on shape and byte length
///    (defense in depth — staging already enforced this).
pub(super) fn validate_ep2_projection<K: PartialEq>(
    proj: &Qwen35MoeLayerProjection<K>,
    shape_cfg: &MoeLayerShapeConfig,
    resolve: &impl Fn(&K) -> Option<(DType, Vec<usize>, usize)>,
) -> Result<(), Vec<Qwen35MoeValidationError>> {
    let mut errors = Vec::new();
    let layer = proj.layer_idx;

    // 1. Rank gate-up pointer tables: present, Raw, [num_experts * 8],
    //    EXACT live allocation byte length num_experts * 8, and no cell-ID
    //    aliasing (rank0 != rank1, and neither rank aliases the canonical
    //    gate-up pointer table).
    let expected = shape_cfg.num_experts * 8;
    for (rank, desc) in proj.ep2_gate_up_ptrs.iter().enumerate() {
        let label = format!("layer {layer} EP2 rank {rank} gate_up_ptrs");
        let Some(desc) = desc else {
            errors.push(Qwen35MoeValidationError::MissingCell(label));
            continue;
        };
        match resolve(&desc.key) {
            None => errors.push(Qwen35MoeValidationError::MissingCell(format!(
                "{label}: key not found"
            ))),
            Some((dt, sh, blen)) => {
                if dt != DType::Raw {
                    errors.push(Qwen35MoeValidationError::PointerTableDtype(format!(
                        "{label}: expected Raw, got {dt:?}"
                    )));
                }
                if sh.len() != 1 || sh[0] != expected {
                    errors.push(Qwen35MoeValidationError::PointerTableShape(format!(
                        "{label}: expected [{expected}], got {sh:?}"
                    )));
                }
                if blen != expected {
                    errors.push(Qwen35MoeValidationError::Ep2RankTableByteLen(format!(
                        "{label}: expected exactly {expected} allocated bytes, got {blen}"
                    )));
                }
            }
        }
        // Cell-ID aliasing is a key-level property, checked even when the
        // key fails to resolve (the alias itself is the violation).
        if desc.key == proj.gate_up_ptrs.key {
            errors.push(Qwen35MoeValidationError::Ep2RankTableAlias(format!(
                "{label}: rank table aliases the canonical gate-up pointer table"
            )));
        }
        if rank == 0 {
            if let Some(other) = proj.ep2_gate_up_ptrs[1].as_ref() {
                if desc.key == other.key {
                    errors.push(Qwen35MoeValidationError::Ep2RankTableAlias(format!(
                        "{label}: rank 0 and rank 1 share the same pointer-table cell ID"
                    )));
                }
            }
        }
    }

    // 2. Distinct canonical gate-up dtypes + per-dtype representative
    //    layout; refuse same-dtype canonical layout disagreement.
    let mut reps: Vec<(DType, Vec<usize>, usize)> = Vec::new();
    for (i, gu) in proj.expert_gate_up.iter().enumerate() {
        let Some((dt, sh, blen)) = resolve(&gu.key) else {
            // Canonical missing-cell is reported by the standard validator
            // (try_new runs after this); skip to avoid duplicate noise.
            continue;
        };
        match reps.iter_mut().find(|(rdt, _, _)| *rdt == dt) {
            Some((_, rep_shape, rep_len)) => {
                if *rep_shape != sh {
                    errors.push(Qwen35MoeValidationError::Ep2DummyShape(format!(
                        "layer {layer} canonical gate-up expert {i} dtype {dt:?} has shape \
                         {sh:?}, representative has {rep_shape:?}"
                    )));
                }
                if *rep_len != blen {
                    errors.push(Qwen35MoeValidationError::Ep2DummyByteLen(format!(
                        "layer {layer} canonical gate-up expert {i} dtype {dt:?} has \
                         {blen} allocated bytes, representative has {rep_len}"
                    )));
                }
            }
            None => reps.push((dt, sh, blen)),
        }
    }

    // 3. Exactly one dummy descriptor per distinct canonical dtype, and
    //    every dummy validates against both its descriptor metadata and
    //    the canonical representative.
    let mut seen: Vec<DType> = Vec::new();
    for (i, d) in proj.ep2_dummies.iter().enumerate() {
        let label = format!("layer {layer} EP2 dummy {i} ({:?})", d.dtype);
        if seen.contains(&d.dtype) {
            errors.push(Qwen35MoeValidationError::Ep2DummyDuplicate(format!(
                "{label}: dtype {:?} already covered by another dummy descriptor",
                d.dtype
            )));
            continue;
        }
        seen.push(d.dtype);
        let Some((_, rep_shape, rep_len)) = reps.iter().find(|(rdt, _, _)| *rdt == d.dtype) else {
            errors.push(Qwen35MoeValidationError::Ep2DummyDuplicate(format!(
                "{label}: stray dummy — no canonical gate-up tensor has dtype {:?}",
                d.dtype
            )));
            continue;
        };
        if d.shape != *rep_shape {
            errors.push(Qwen35MoeValidationError::Ep2DummyShape(format!(
                "{label}: descriptor shape {:?} != representative {rep_shape:?}",
                d.shape
            )));
        }
        if d.byte_len != *rep_len {
            errors.push(Qwen35MoeValidationError::Ep2DummyByteLen(format!(
                "{label}: descriptor byte length {} != representative {rep_len}",
                d.byte_len
            )));
        }
        match resolve(&d.key) {
            None => errors.push(Qwen35MoeValidationError::Ep2DummyInvalidId(format!(
                "{label}: cell ID does not resolve in the frozen store"
            ))),
            Some((dt, sh, blen)) => {
                if dt != d.dtype {
                    errors.push(Qwen35MoeValidationError::Ep2DummyDtype(format!(
                        "{label}: store tensor dtype {dt:?} != descriptor dtype {:?}",
                        d.dtype
                    )));
                }
                if sh != d.shape {
                    errors.push(Qwen35MoeValidationError::Ep2DummyShape(format!(
                        "{label}: store tensor shape {sh:?} != descriptor shape {:?}",
                        d.shape
                    )));
                }
                if blen != d.byte_len {
                    errors.push(Qwen35MoeValidationError::Ep2DummyByteLen(format!(
                        "{label}: store tensor byte length {blen} != descriptor byte length {}",
                        d.byte_len
                    )));
                }
            }
        }
    }

    // 4. Every distinct canonical dtype is covered by a dummy.
    for (dt, _, _) in &reps {
        if !seen.contains(dt) {
            errors.push(Qwen35MoeValidationError::Ep2DummyMissing(format!(
                "layer {layer}: no dummy descriptor for canonical gate-up dtype {dt:?}"
            )));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

impl Qwen35MoeResident {
    /// Publish a resident whose projections carry cfg-gated EP2 rank table
    /// IDs, validating that every layer/rank table resolves to a Raw
    /// `[num_experts * 8]` store cell before delegating to [`Self::try_new`].
    #[expect(
        clippy::result_large_err,
        reason = "Err returns every staged owner (validation errors + store + projections) so the caller retries or frees without losing any"
    )]
    #[expect(
        clippy::type_complexity,
        reason = "the tuple preserves each staged owner (errors, store, projections) for exact rollback — same shape as Qwen35MoeResident::try_new"
    )]
    pub(crate) fn try_new_with_ep2(
        store: SingleFrozenWeightStore,
        layers: Vec<Qwen35MoeLayerProjection<WeightCellId>>,
        shape_cfg: &MoeLayerShapeConfig,
    ) -> Result<
        Self,
        (
            Vec<Qwen35MoeValidationError>,
            SingleFrozenWeightStore,
            Vec<Qwen35MoeLayerProjection<WeightCellId>>,
        ),
    > {
        // Pure validation against the live store metadata: both rank
        // gate-up pointer tables (Raw, [num_experts * 8]) and exactly one
        // dummy descriptor per distinct canonical gate-up dtype, each
        // matching its representative's dtype/shape/byte length exactly.
        let resolve = |key: &WeightCellId| -> Option<(DType, Vec<usize>, usize)> {
            let tensor = store.tensor(*key)?;
            Some((tensor.dtype, tensor.shape.clone(), tensor.buf.size()))
        };
        let mut errors = Vec::new();
        for proj in &layers {
            if let Err(mut e) = validate_ep2_projection(proj, shape_cfg, &resolve) {
                errors.append(&mut e);
            }
        }
        if !errors.is_empty() {
            return Err((errors, store, layers));
        }
        Self::try_new(store, layers, shape_cfg)
    }

    /// Test-only borrowed seam: the staged zero dummy tensor for `dtype` on
    /// `layer`, or `None` when the layer stages no dummy for that dtype.
    ///
    /// Borrows from the single Frozen store — no ownership transfer, no raw
    /// pointer, no second resident.  Lets the harness verify the dummy's
    /// exact bytes/shape/dtype/byte length against its canonical
    /// representative without Phase 2B kernel execution.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "borrowed dummy seam exercised by the emulated EP2 GPU-ignored structural test; consumed by the Phase 2B driver"
        )
    )]
    pub(crate) fn ep2_dummy_tensor(&self, layer: usize, dtype: DType) -> Option<&GpuTensor> {
        let proj = self.layers.get(layer)?;
        let desc = proj.ep2_dummies.iter().find(|d| d.dtype == dtype)?;
        self.store.tensor(desc.key)
    }

    /// O(1) EP2 bind: like [`Self::bind_layer`] but with the gate-up pointer
    /// table cell ID overridden to the rank-masked table.  Canonical down /
    /// down-AWQ / dtype-tag / Paro resources stay borrowed and unmasked.
    /// Crate-private: the emulated EP2 harness surface is test-only.
    pub(crate) fn bind_layer_ep2(
        &self,
        layer: usize,
        rank: usize,
    ) -> Result<MoeFfnBindings<'_>, Qwen35MoeBindError> {
        if rank >= 2 {
            return Err(Qwen35MoeBindError::Ep2RankOutOfRange {
                requested: rank,
                count: 2,
            });
        }
        let proj = self
            .layers
            .get(layer)
            .ok_or(Qwen35MoeBindError::LayerOutOfRange {
                requested: layer,
                count: self.layers.len(),
            })?;
        let table = ep2_rank_gate_up_ptrs_key(proj, rank).ok_or_else(|| {
            Qwen35MoeBindError::TensorLookup(
                format!("layer {layer} EP2 rank {rank} gate_up_ptrs"),
                WeightCellLookupError::InvalidSlot,
            )
        })?;
        Ok(MoeFfnBindings {
            store: &self.store,
            proj,
            ep2_gate_up_ptrs: Some(table),
        })
    }
}
