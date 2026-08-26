// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! CAP-001 parallel capability resolver.
//!
//! Pure table-driven admission control for parallelism axes. No GPU, mesh,
//! stream, allocation, peer, or collective imports — only domain types.
//!
//! # Resolution order (see [`resolve`])
//!
//! 1. **Degree-zero reject** — any axis with value `0` is rejected.
//! 2. **Composition reject** — TP×EP or PP×{TP,EP} refused before remap or
//!    normalisation (COMP-001).
//! 3. **Legacy remap** — DeepSeek4 / MiniMax TP→EP (preserves degrees).
//! 4. **Policy lookup** — `(variant, effective_axis)` checked against the
//!    architecture-id matrix (`docs/architecture-ids.md` lines 14–30). Cells
//!    marked `NormalizeToSingle` canonicalise effective degrees to `(1,1,1)`
//!    and re-evaluate the `Single` cell — no separate dense-normalisation step.

/// Parallelism axis after resolution.
///
/// `#[non_exhaustive]` — new axes may be added in minor releases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ParallelAxis {
    Single,
    Pp,
    Tp,
    Ep,
}

impl ParallelAxis {
    /// Stable short name for diagnostic messages.
    pub fn name(&self) -> &'static str {
        match self {
            ParallelAxis::Single => "Single",
            ParallelAxis::Pp => "PP",
            ParallelAxis::Tp => "TP",
            ParallelAxis::Ep => "EP",
        }
    }
}

/// A raw parallelism request — degree for each axis.
///
/// All fields are `usize`. Any axis with value `0` is rejected by [`resolve`]
/// before any further processing (degree-zero reject).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawParallelRequest {
    pub pp: usize,
    pub tp: usize,
    pub ep: usize,
}

impl RawParallelRequest {
    pub const fn new(pp: usize, tp: usize, ep: usize) -> Self {
        Self { pp, tp, ep }
    }

    /// Dominant axis (first with degree > 1), or [`ParallelAxis::Single`]
    /// when all axes are ≤ 1.
    pub const fn axis(&self) -> ParallelAxis {
        if self.pp > 1 {
            ParallelAxis::Pp
        } else if self.tp > 1 {
            ParallelAxis::Tp
        } else if self.ep > 1 {
            ParallelAxis::Ep
        } else {
            ParallelAxis::Single
        }
    }
}

/// Model-variant classification for policy lookup.
///
/// Every variant maps to a unique row in the architecture-id registry
/// (`docs/architecture-ids.md`).  `#[non_exhaustive]` — new variants
/// may be added in minor releases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelVariant {
    /// LLaMA / Mistral with QK-norm enabled (arch_id=0, `has_qk_norm=true`).
    LlamaQkNorm,
    /// LLaMA / Mistral without QK-norm (arch_id=0, `has_qk_norm=false`).
    LlamaNoQkNorm,
    /// Plain Qwen3 (arch_id=1).
    PlainQwen3,
    /// Qwen3.5 dense (arch_id=5, non-VL).
    Qwen35Dense,
    /// Qwen3.5 / 3.6 MoE / A3B (arch_id=6).
    Qwen35Moe,
    /// Qwen3.5-VL (arch_id=5 with vision tower).
    Qwen35Vl,
    /// Qwen2 dense standalone (arch_id=7).
    Qwen2,
    /// Qwen2-VL / dots.ocr (arch_id=8).
    DotsOcr,
    /// DeepSeek V4 Flash (arch_id=9).
    Deepseek4,
    /// MiniMax-M2 (arch_id=10).
    Minimax,
    /// LFM2.5 dense (arch_id=11, dense variant).
    Lfm2Dense,
    /// LFM2.5-MoE (arch_id=11, MoE variant).
    Lfm2Moe,
    /// Cohere2-MoE / North-Mini-Code (arch_id=12).
    Cohere2Moe,
    /// Gemma 4 text, dense or MoE (arch_id=13).
    Gemma4,
    /// Muse Glimmer dense text (arch_id=14).
    MuseGlimmer,
}

/// Per-cell policy outcome returned by the policy table.
///
/// Every cell in the architecture-id matrix (`docs/architecture-ids.md`
/// lines 14–30) maps to exactly one variant. See that table for the
/// canonical mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellPolicy {
    /// Cell is open — implementation exists; HW-pending does not block.
    Admitted,
    /// Planned under a tracking issue; not yet implemented.
    Planned {
        /// The owning issue or task reference (e.g. `"AXIS-002"`, `"GEN-001"`).
        owner: &'static str,
        /// Human-readable explanation.
        reason: &'static str,
    },
    /// Architecturally unsupported for this variant–axis combination.
    /// Includes a technical reason for the diagnostic.
    Unsupported {
        /// Human-readable explanation.
        reason: &'static str,
    },
    /// Variant–axis combination is resolved by normalising the request
    /// to Single (dense EP cells). The cell itself is not directly
    /// admitted or refused — the effective axis after normalisation is
    /// evaluated instead.
    NormalizeToSingle,
}

/// Successful admission: the resolved parallelism is accepted.
///
/// Fields are private — constructors are resolver-only. Read accessors
/// are public.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParallelAdmission {
    /// The model variant that was resolved.
    variant: ModelVariant,
    /// The original raw request as supplied by the caller.
    requested: RawParallelRequest,
    /// The effective degrees after legacy remap and dense EP
    /// normalisation.
    effective: RawParallelRequest,
}

impl ParallelAdmission {
    /// The model variant that was resolved.
    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    /// The original raw request as supplied by the caller.
    pub fn requested(&self) -> RawParallelRequest {
        self.requested
    }

    /// The effective degrees after legacy remap and normalisation.
    pub fn effective(&self) -> RawParallelRequest {
        self.effective
    }

    /// Whether the effective degrees differ from the requested degrees
    /// (i.e., a legacy TP→EP remap or dense EP normalisation applied).
    pub fn was_normalized(&self) -> bool {
        self.requested != self.effective
    }

    /// Resolver-only constructor for an admission on any effective axis.
    pub(crate) fn new(
        variant: ModelVariant,
        requested: RawParallelRequest,
        effective: RawParallelRequest,
    ) -> Self {
        Self {
            variant,
            requested,
            effective,
        }
    }
}

/// Admission failure — an enum over every rejection category in the
/// CAP-001 resolution order.
///
/// Each variant carries the fields relevant to its category. Use
/// [`AdmissionError::code`] for the stable diagnostic tag
/// (`"CAP-001"` or `"COMP-001"`) and
/// [`AdmissionError::effective_axis`] for the resolved parallelism
/// axis at the point of rejection (returns `None` for
/// `InvalidDegree` and `Composition`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    /// A zero degree was found (step 1). Carries only the
    /// offending axis and degree value.
    InvalidDegree {
        /// The axis with degree zero.
        axis: ParallelAxis,
        /// The offending degree value (always 0 at step 1).
        degree: usize,
    },
    /// TP×EP or PP×{TP,EP} composition (step 2).
    Composition {
        /// The model variant at the point of rejection.
        variant: ModelVariant,
        /// The original raw request.
        requested: RawParallelRequest,
        /// The effective degrees at the point of rejection.
        effective: RawParallelRequest,
        /// The owning tracking issue (`"COMP-001"` or `"CAP-001"`).
        owner: &'static str,
        /// Human-readable reason.
        reason: &'static str,
    },
    /// Planned cell — implementation tracked by an issue.
    Planned {
        /// The model variant at the point of rejection.
        variant: ModelVariant,
        /// The original raw request.
        requested: RawParallelRequest,
        /// The effective degrees at the point of rejection.
        effective: RawParallelRequest,
        /// The owning tracking issue (e.g. `"AXIS-002"`).
        owner: &'static str,
        /// Human-readable reason.
        reason: &'static str,
    },
    /// Architecturally unsupported cell.
    Unsupported {
        /// The model variant at the point of rejection.
        variant: ModelVariant,
        /// The original raw request.
        requested: RawParallelRequest,
        /// The effective degrees at the point of rejection.
        effective: RawParallelRequest,
        /// The owning tracking issue (typically `"CAP-001"`).
        owner: &'static str,
        /// Technical reason for the refusal.
        reason: &'static str,
    },
}

impl AdmissionError {
    /// Stable diagnostic tag — returns `"CAP-001"` for all
    /// error categories except `Composition` which returns its
    /// owner code (`"COMP-001"` or `"CAP-001"`).
    pub fn code(&self) -> &'static str {
        match self {
            AdmissionError::InvalidDegree { .. } => "CAP-001",
            AdmissionError::Composition { owner, .. } => owner,
            AdmissionError::Planned { .. } => "CAP-001",
            AdmissionError::Unsupported { .. } => "CAP-001",
        }
    }

    /// The parallelism axis at the point of rejection.
    ///
    /// Returns `None` for [`InvalidDegree`](AdmissionError::InvalidDegree)
    /// (the axis is in the `axis` field) and
    /// [`Composition`](AdmissionError::Composition) (the composition
    /// involves multiple axes). For [`Planned`](AdmissionError::Planned)
    /// and [`Unsupported`](AdmissionError::Unsupported) returns
    /// `Some(axis)` derived from the effective degrees.
    pub fn effective_axis(&self) -> Option<ParallelAxis> {
        match self {
            AdmissionError::InvalidDegree { .. } => None,
            AdmissionError::Composition { .. } => None,
            AdmissionError::Planned { effective, .. } => Some(effective.axis()),
            AdmissionError::Unsupported { effective, .. } => Some(effective.axis()),
        }
    }
}

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdmissionError::InvalidDegree { axis, degree } => {
                write!(
                    f,
                    "[CAP-001] Invalid degree axis={:?} degree={}",
                    axis, degree,
                )
            }
            AdmissionError::Composition {
                variant,
                requested,
                effective,
                owner,
                reason,
            } => {
                write!(
                    f,
                    "[{owner}] Composition: {variant:?} request (pp={},tp={},ep={}) effective (pp={},tp={},ep={}) owner={owner} reason={reason}",
                    requested.pp,
                    requested.tp,
                    requested.ep,
                    effective.pp,
                    effective.tp,
                    effective.ep,
                )
            }
            AdmissionError::Planned {
                variant,
                requested,
                effective,
                owner,
                reason,
            } => {
                write!(
                    f,
                    "[CAP-001] Planned: {:?} request (pp={},tp={},ep={}) effective (pp={},tp={},ep={}) owner={} reason={}",
                    variant,
                    requested.pp,
                    requested.tp,
                    requested.ep,
                    effective.pp,
                    effective.tp,
                    effective.ep,
                    owner,
                    reason,
                )
            }
            AdmissionError::Unsupported {
                variant,
                requested,
                effective,
                owner,
                reason,
            } => {
                write!(
                    f,
                    "[CAP-001] Unsupported: {:?} request (pp={},tp={},ep={}) effective (pp={},tp={},ep={}) owner={} reason={}",
                    variant,
                    requested.pp,
                    requested.tp,
                    requested.ep,
                    effective.pp,
                    effective.tp,
                    effective.ep,
                    owner,
                    reason,
                )
            }
        }
    }
}

// ─── Resolution ──────────────────────────────────────────────────────

/// Resolve a raw parallelism request against a model variant.
///
/// Returns `Ok(ParallelAdmission)` with the normalised degrees when the
/// combination is admitted, or `Err(AdmissionError)` describing why the
/// request was refused.
///
/// # Resolution order
///
/// 1. **Degree-zero reject** — any axis with value `0` is rejected.
/// 2. **Composition reject** — TP and EP cannot both exceed one; PP
///    cannot be combined with TP or EP (COMP-001). Runs before legacy
///    remap or normalisation.
/// 3. **Legacy remap** — DeepSeek4 and MiniMax requests with `tp > 1`
///    and `ep <= 1` are remapped to EP, preserving the degree value.
/// 4. **Policy lookup** — `(variant, effective_axis)` checked against
///    the architecture-id matrix. Cells marked `NormalizeToSingle`
///    canonicalise effective degrees to `(1,1,1)` and re-evaluate the
///    `Single` cell in their place — no separate dense-normalisation step.
pub fn resolve(
    variant: ModelVariant,
    raw: RawParallelRequest,
) -> Result<ParallelAdmission, AdmissionError> {
    // Step 1: degree-zero reject — identify the exact zero axis
    if raw.pp == 0 || raw.tp == 0 || raw.ep == 0 {
        let axis = if raw.pp == 0 {
            ParallelAxis::Pp
        } else if raw.tp == 0 {
            ParallelAxis::Tp
        } else {
            ParallelAxis::Ep
        };
        return Err(AdmissionError::InvalidDegree { axis, degree: 0 });
    }

    // Step 2: composition reject (before legacy remap or normalisation)
    // TP×EP remains COMP-001; PP×{TP,EP} uses CAP-001 (per the approved contract).
    if raw.tp > 1 && raw.ep > 1 {
        return Err(AdmissionError::Composition {
            variant,
            requested: raw,
            effective: raw,
            owner: "COMP-001",
            reason: "TP and EP cannot both exceed one (COMP-001)",
        });
    }
    if raw.pp > 1 && (raw.tp > 1 || raw.ep > 1) {
        return Err(AdmissionError::Composition {
            variant,
            requested: raw,
            effective: raw,
            owner: "CAP-001",
            reason: "PP cannot be combined with TP or EP (CAP-001)",
        });
    }

    // Step 3: legacy DeepSeek4 / MiniMax TP→EP remap (preserves degrees)
    let pp = raw.pp;
    let mut tp = raw.tp;
    let mut ep = raw.ep;
    if matches!(variant, ModelVariant::Deepseek4 | ModelVariant::Minimax) && tp > 1 && ep <= 1 {
        ep = tp;
        tp = 1;
    }

    let effective = RawParallelRequest::new(pp, tp, ep);
    let axis = effective.axis();

    // Step 4: policy lookup on (variant, effective_axis)
    //
    // If the cell is `NormalizeToSingle` (dense EP), canonicalise the
    // effective degrees to `(1,1,1)` and re-evaluate the `Single` cell.
    // This makes `CellPolicy::NormalizeToSingle` the single source of
    // truth for dense-EP normalisation — no separate `is_dense()` check.
    resolve_cell(variant, raw, effective, axis)
}

/// Policy evaluation helper used by [`resolve`].
///
/// Looks up `(variant, axis)` in the policy table. If the cell is
/// [`NormalizeToSingle`](CellPolicy::NormalizeToSingle), canonicalises
/// the effective degrees to `(1,1,1)` and re-evaluates the `Single`
/// cell in its place.
fn resolve_cell(
    variant: ModelVariant,
    raw: RawParallelRequest,
    effective: RawParallelRequest,
    axis: ParallelAxis,
) -> Result<ParallelAdmission, AdmissionError> {
    let policy = cell_info(variant, axis);
    match policy {
        CellPolicy::NormalizeToSingle => {
            let normalized = RawParallelRequest::new(1, 1, 1);
            // Single cells are never NormalizeToSingle, so no infinite loop.
            let single_policy = cell_info(variant, ParallelAxis::Single);
            match single_policy {
                CellPolicy::Admitted => Ok(ParallelAdmission::new(variant, raw, normalized)),
                CellPolicy::Planned { owner, reason } => Err(AdmissionError::Planned {
                    variant,
                    requested: raw,
                    effective: normalized,
                    owner,
                    reason,
                }),
                CellPolicy::Unsupported { reason } => Err(AdmissionError::Unsupported {
                    variant,
                    requested: raw,
                    effective: normalized,
                    owner: "CAP-001",
                    reason,
                }),
                CellPolicy::NormalizeToSingle => {
                    unreachable!("Single cell cannot be NormalizeToSingle")
                }
            }
        }
        CellPolicy::Admitted => Ok(ParallelAdmission::new(variant, raw, effective)),
        CellPolicy::Planned { owner, reason } => Err(AdmissionError::Planned {
            variant,
            requested: raw,
            effective,
            owner,
            reason,
        }),
        CellPolicy::Unsupported { reason } => Err(AdmissionError::Unsupported {
            variant,
            requested: raw,
            effective,
            owner: "CAP-001",
            reason,
        }),
    }
}

// ─── Policy table ────────────────────────────────────────────────────

/// Table-driven policy lookup: `(variant, axis) → CellPolicy`.
///
/// Every cell in the arch‑id matrix has exactly one entry. The `owner`
/// and `reason` are carried inside the `Planned` / `Unsupported` variants
/// directly — no separate metadata tuple.
fn cell_info(variant: ModelVariant, axis: ParallelAxis) -> CellPolicy {
    use CellPolicy::*;
    use ModelVariant::*;
    use ParallelAxis::*;

    // Policy categories from docs/architecture-ids.md lines 14-30:
    //   "implemented" / "implemented code; HW-XXX pending"  → Admitted
    //   "partial (GEN-XXX; HW-XXX pending)"                 → Planned { owner: "GEN-XXX" }
    //   "planned (AXIS-XXX; HW-XXX pending)"                → Planned { owner: "AXIS-XXX" }
    //   "normalized-to-single(CAP-001)"                     → NormalizeToSingle
    //   architected refusal (e.g. "non-qk-norm refused")    → Unsupported { reason }

    match (variant, axis) {
        // ── LLaMA QK-norm (arch_id 0, has_qk_norm=true) ───────────
        (LlamaQkNorm, Single) => Admitted,
        (LlamaQkNorm, Pp) => Admitted,
        (LlamaQkNorm, Tp) => Admitted,
        (LlamaQkNorm, Ep) => NormalizeToSingle,

        // ── LLaMA no QK-norm (arch_id 0, has_qk_norm=false) ──────
        (LlamaNoQkNorm, Single) => Admitted,
        (LlamaNoQkNorm, Pp) => Admitted,
        (LlamaNoQkNorm, Tp) => Unsupported {
            reason: "non-QK-norm LLaMA/Mistral: TP not supported",
        },
        (LlamaNoQkNorm, Ep) => NormalizeToSingle,

        // ── Plain Qwen3 (arch_id 1) ───────────────────────────────
        (PlainQwen3, Single) => Admitted,
        (PlainQwen3, Pp) => Admitted,
        (PlainQwen3, Tp) => Admitted,
        (PlainQwen3, Ep) => NormalizeToSingle,

        // ── Qwen3.5 dense (arch_id 5) ─────────────────────────────
        (Qwen35Dense, Single) => Admitted,
        (Qwen35Dense, Pp) => Planned {
            owner: "GEN-001",
            reason: "Qwen3.5 dense PP: partial implementation; GEN-001 pending",
        },
        (Qwen35Dense, Tp) => Planned {
            owner: "AXIS-002",
            reason: "Qwen3.5 dense TP: planned; AXIS-002",
        },
        (Qwen35Dense, Ep) => NormalizeToSingle,

        // ── Qwen3.5 MoE / A3B (arch_id 6) ─────────────────────────
        (Qwen35Moe, Single) => Admitted,
        (Qwen35Moe, Pp) => Planned {
            owner: "GEN-001",
            reason: "Qwen3.5 MoE PP: partial implementation; GEN-001 pending",
        },
        (Qwen35Moe, Tp) => Planned {
            owner: "AXIS-002",
            reason: "Qwen3.5 MoE TP: planned; AXIS-002",
        },
        (Qwen35Moe, Ep) => Planned {
            owner: "AXIS-002",
            reason: "Qwen3.5 MoE EP: planned; AXIS-002",
        },

        // ── Qwen3.5-VL (arch_id 5 VL) ─────────────────────────────
        (Qwen35Vl, Single) => Admitted,
        (Qwen35Vl, Pp) => Planned {
            owner: "AXIS-004",
            reason: "Qwen3.5-VL PP: planned; AXIS-004",
        },
        (Qwen35Vl, Tp) => Planned {
            owner: "AXIS-004",
            reason: "Qwen3.5-VL TP: planned; AXIS-004",
        },
        (Qwen35Vl, Ep) => NormalizeToSingle,

        // ── Qwen2 dense (arch_id 7) ───────────────────────────────
        (Qwen2, Single) => Admitted,
        (Qwen2, Pp) => Planned {
            owner: "AXIS-001",
            reason: "Qwen2 PP: planned; AXIS-001",
        },
        (Qwen2, Tp) => Planned {
            owner: "AXIS-001",
            reason: "Qwen2 TP: planned; AXIS-001",
        },
        (Qwen2, Ep) => NormalizeToSingle,

        // ── dots.ocr (arch_id 8) ───────────────────────────────────
        (DotsOcr, Single) => Admitted,
        (DotsOcr, Pp) => Planned {
            owner: "AXIS-004",
            reason: "dots.ocr PP: planned; AXIS-004",
        },
        (DotsOcr, Tp) => Planned {
            owner: "AXIS-004",
            reason: "dots.ocr TP: planned; AXIS-004",
        },
        (DotsOcr, Ep) => NormalizeToSingle,

        // ── DeepSeek V4 Flash (arch_id 9) ──────────────────────────
        (Deepseek4, Single) => Admitted,
        (Deepseek4, Pp) => Planned {
            owner: "AXIS-003",
            reason: "DeepSeek V4 PP: planned; AXIS-003",
        },
        (Deepseek4, Tp) => Planned {
            owner: "AXIS-003",
            reason: "DeepSeek V4 TP: planned; AXIS-003",
        },
        (Deepseek4, Ep) => Admitted,

        // ── MiniMax (arch_id 10) ───────────────────────────────────
        (Minimax, Single) => Admitted,
        (Minimax, Pp) => Planned {
            owner: "AXIS-003",
            reason: "MiniMax PP: planned; AXIS-003",
        },
        (Minimax, Tp) => Planned {
            owner: "AXIS-003",
            reason: "MiniMax TP: planned; AXIS-003",
        },
        (Minimax, Ep) => Admitted,

        // ── LFM2 dense (arch_id 11, dense) ────────────────────────
        (Lfm2Dense, Single) => Planned {
            owner: "AXIS-003",
            reason: "LFM2 dense Single: planned admission; AXIS-003",
        },
        (Lfm2Dense, Pp) => Planned {
            owner: "AXIS-003",
            reason: "LFM2 dense PP: planned; AXIS-003",
        },
        (Lfm2Dense, Tp) => Planned {
            owner: "AXIS-003",
            reason: "LFM2 dense TP: planned; AXIS-003",
        },
        (Lfm2Dense, Ep) => NormalizeToSingle,

        // ── LFM2 MoE (arch_id 11, MoE) ────────────────────────────
        (Lfm2Moe, Single) => Admitted,
        (Lfm2Moe, Pp) => Planned {
            owner: "AXIS-003",
            reason: "LFM2 MoE PP: planned; AXIS-003",
        },
        (Lfm2Moe, Tp) => Planned {
            owner: "AXIS-003",
            reason: "LFM2 MoE TP: planned; AXIS-003",
        },
        (Lfm2Moe, Ep) => Planned {
            owner: "AXIS-003",
            reason: "LFM2 MoE EP: planned; AXIS-003",
        },

        // ── Cohere2-MoE (arch_id 12) ──────────────────────────────
        (Cohere2Moe, Single) => Admitted,
        (Cohere2Moe, Pp) => Planned {
            owner: "AXIS-003",
            reason: "Cohere2-MoE PP: planned; AXIS-003",
        },
        (Cohere2Moe, Tp) => Planned {
            owner: "AXIS-003",
            reason: "Cohere2-MoE TP: planned; AXIS-003",
        },
        (Cohere2Moe, Ep) => Planned {
            owner: "AXIS-003",
            reason: "Cohere2-MoE EP: planned; AXIS-003",
        },
        // ── Gemma 4 text, dense or MoE (arch_id 13) ───────────────
        (Gemma4, Single) => Admitted,
        (Gemma4, Pp) => Unsupported {
            reason: "Gemma4 PP: pp>1 unsupported",
        },
        (Gemma4, Tp) => Unsupported {
            reason: "Gemma4 TP: no tensor-parallel path; single-device decode only",
        },
        (Gemma4, Ep) => Unsupported {
            reason: "Gemma4 EP: no expert-parallel path",
        },

        // ── Muse Glimmer dense text (arch_id 14) ─────────────────
        (MuseGlimmer, Single) => Admitted,
        (MuseGlimmer, Pp) => Unsupported {
            reason: "Muse Glimmer PP: pp>1 unsupported",
        },
        (MuseGlimmer, Tp) => Unsupported {
            reason: "Muse Glimmer TP: no tensor-parallel path; single-device dense decode only",
        },
        (MuseGlimmer, Ep) => Unsupported {
            reason: "Muse Glimmer EP: no expert-parallel path; dense EP is not normalized",
        },
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────

    fn req(pp: usize, tp: usize, ep: usize) -> RawParallelRequest {
        RawParallelRequest::new(pp, tp, ep)
    }

    /// Extract the code string from any `AdmissionError` variant.
    fn err_code(e: &AdmissionError) -> &'static str {
        e.code()
    }

    /// Extract the reason string from any `AdmissionError` variant.
    fn err_reason(e: &AdmissionError) -> &'static str {
        match e {
            AdmissionError::InvalidDegree { .. } => "degree zero: every axis must be >= 1",
            AdmissionError::Composition { reason, .. } => reason,
            AdmissionError::Planned { reason, .. } => reason,
            AdmissionError::Unsupported { reason, .. } => reason,
        }
    }

    fn assert_refused(
        v: ModelVariant,
        pp: usize,
        tp: usize,
        ep: usize,
        expected_code: &str,
        expected_reason_substr: &str,
    ) -> AdmissionError {
        let result = resolve(v, req(pp, tp, ep));
        assert!(
            result.is_err(),
            "expected refusal for {v:?} (pp={pp},tp={tp},ep={ep}) but got Ok"
        );
        let err = result.unwrap_err();
        let code = err_code(&err);
        assert_eq!(
            code, expected_code,
            "code mismatch for {v:?} (pp={pp},tp={tp},ep={ep}): expected {expected_code}, got {code}",
        );
        let reason = err_reason(&err);
        assert!(
            reason.contains(expected_reason_substr),
            "reason for {v:?} (pp={pp},tp={tp},ep={ep}) should contain '{expected_reason_substr}': {reason}",
        );
        err
    }

    // ── Degree-zero reject ────────────────────────────────────────

    /// Rejects *any* zero degree — not merely `(0, 0, 0)`.
    #[test]
    fn rejects_any_zero_degree() {
        // All-zero — identifies Pp as the first zero axis
        let err = assert_refused(ModelVariant::LlamaQkNorm, 0, 0, 0, "CAP-001", "degree zero");
        assert_eq!(err.effective_axis(), None);
        if let AdmissionError::InvalidDegree { axis, degree } = &err {
            assert_eq!(*axis, ParallelAxis::Pp);
            assert_eq!(*degree, 0);
        } else {
            panic!("expected InvalidDegree variant");
        }

        // Individual zeros on each axis — identifies the exact zero axis
        let err = assert_refused(ModelVariant::LlamaQkNorm, 2, 1, 0, "CAP-001", "degree zero");
        assert_eq!(err.effective_axis(), None);
        if let AdmissionError::InvalidDegree { axis, degree } = &err {
            assert_eq!(*axis, ParallelAxis::Ep);
            assert_eq!(*degree, 0);
        } else {
            panic!("expected InvalidDegree variant");
        }

        let err = assert_refused(ModelVariant::LlamaQkNorm, 2, 0, 1, "CAP-001", "degree zero");
        assert_eq!(err.effective_axis(), None);
        if let AdmissionError::InvalidDegree { axis, degree } = &err {
            assert_eq!(*axis, ParallelAxis::Tp);
            assert_eq!(*degree, 0);
        } else {
            panic!("expected InvalidDegree variant");
        }

        let err = assert_refused(ModelVariant::LlamaQkNorm, 0, 2, 1, "CAP-001", "degree zero");
        assert_eq!(err.effective_axis(), None);
        if let AdmissionError::InvalidDegree { axis, degree } = &err {
            assert_eq!(*axis, ParallelAxis::Pp);
            assert_eq!(*degree, 0);
        } else {
            panic!("expected InvalidDegree variant");
        }
    }

    // ── Composition reject ────────────────────────────────────────

    #[test]
    fn rejects_tp_ep_composition() {
        for v in [ModelVariant::LlamaQkNorm, ModelVariant::Qwen35Moe] {
            let err = assert_refused(v, 1, 2, 2, "COMP-001", "TP and EP");
            assert_eq!(err.effective_axis(), None);
        }
    }

    /// PP×TP / PP×EP composition is CAP-001 per the approved contract.
    #[test]
    fn rejects_pp_tp_composition() {
        let err = assert_refused(ModelVariant::LlamaQkNorm, 2, 2, 1, "CAP-001", "PP cannot");
        assert_eq!(err.effective_axis(), None);
    }

    #[test]
    fn rejects_pp_ep_composition() {
        let err = assert_refused(ModelVariant::LlamaQkNorm, 2, 1, 2, "CAP-001", "PP cannot");
        assert_eq!(err.effective_axis(), None);
    }

    // ── Legacy DeepSeek4 TP→EP ────────────────────────────────────

    #[test]
    fn deepseek4_tp_remaps_to_ep() {
        let admission = resolve(ModelVariant::Deepseek4, req(1, 2, 1)).unwrap();
        assert_eq!(admission.effective().axis(), ParallelAxis::Ep);
        // Degrees preserved: ep = tp = 2
        assert_eq!(admission.effective().tp, 1);
        assert_eq!(admission.effective().ep, 2);
        // Legacy TP→EP remap makes requested != effective
        assert!(admission.was_normalized());
    }

    #[test]
    fn deepseek4_tp_does_not_remap_when_ep_already_set() {
        // tp=2, ep=2 → composition-rejected before remap
        let err = resolve(ModelVariant::Deepseek4, req(1, 2, 2)).unwrap_err();
        assert_eq!(err.code(), "COMP-001");
    }

    // ── Legacy MiniMax TP→EP ──────────────────────────────────────

    #[test]
    fn minimax_tp_remaps_to_ep() {
        let admission = resolve(ModelVariant::Minimax, req(1, 2, 1)).unwrap();
        assert_eq!(admission.effective().axis(), ParallelAxis::Ep);
        assert_eq!(admission.effective().ep, 2);
        // Legacy TP→EP remap makes requested != effective
        assert!(admission.was_normalized());
    }

    /// DeepSeek4 TP→EP preserves the degree value (tp=4 → ep=4).
    #[test]
    fn deepseek4_tp_remap_preserves_degree() {
        let admission = resolve(ModelVariant::Deepseek4, req(1, 4, 1)).unwrap();
        assert_eq!(admission.effective().axis(), ParallelAxis::Ep);
        assert_eq!(admission.effective().ep, 4);
    }

    /// MiniMax TP→EP preserves the degree value (tp=8 → ep=8).
    #[test]
    fn minimax_tp_remap_preserves_degree() {
        let admission = resolve(ModelVariant::Minimax, req(1, 8, 1)).unwrap();
        assert_eq!(admission.effective().axis(), ParallelAxis::Ep);
        assert_eq!(admission.effective().ep, 8);
    }

    // ── Dense EP normalisation (via NormalizeToSingle policy) ─────

    /// Dense EP normalisation canonicalises (1, 1, 5) to (1, 1, 1).
    /// (3, 4, 5) would be composition-rejected before normalisation.)
    #[test]
    fn dense_ep_normalises_any_degrees_to_111() {
        let admission = resolve(ModelVariant::LlamaQkNorm, req(1, 1, 5)).unwrap();
        assert_eq!(admission.effective(), req(1, 1, 1));
        assert!(admission.was_normalized());
        // Requested degrees are preserved for inspection
        assert_eq!(admission.requested(), req(1, 1, 5));
    }

    // ── LFM2 dense behaviour ──────────────────────────────────────

    #[test]
    fn lfm2_dense_single_is_refused() {
        let err = assert_refused(
            ModelVariant::Lfm2Dense,
            1,
            1,
            1,
            "CAP-001",
            "planned admission",
        );
        assert_eq!(err.effective_axis(), Some(ParallelAxis::Single));
    }

    #[test]
    fn lfm2_dense_ep_normalises_to_single_then_refused() {
        // EP=2 normalises to (1,1,1) via NormalizeToSingle, then policy
        // lookup on (Lfm2Dense, Single) is Planned with AXIS-003 owner.
        let err = assert_refused(
            ModelVariant::Lfm2Dense,
            1,
            1,
            2,
            "CAP-001",
            "planned admission",
        );
        // The effective after normalisation is (1,1,1) — extract from the variant
        match &err {
            AdmissionError::Planned { effective, .. } => {
                assert_eq!(*effective, req(1, 1, 1));
            }
            other => panic!("expected Planned variant, got {other:?}"),
        }
        assert_eq!(err.effective_axis(), Some(ParallelAxis::Single));
    }

    // ── RawParallelRequest helpers ─────────────────────────────────

    #[test]
    fn raw_request_new_and_axis() {
        assert_eq!(req(0, 0, 0).axis(), ParallelAxis::Single);
        assert_eq!(req(1, 1, 1).axis(), ParallelAxis::Single);
        assert_eq!(req(2, 1, 1).axis(), ParallelAxis::Pp);
        assert_eq!(req(1, 2, 1).axis(), ParallelAxis::Tp);
        assert_eq!(req(1, 1, 2).axis(), ParallelAxis::Ep);
        assert_eq!(req(2, 2, 1).axis(), ParallelAxis::Pp); // Pp dominant (first)
    }

    // ── AdmissionError Display ────────────────────────────────────

    #[test]
    fn admission_error_display_invalid_degree() {
        let err = AdmissionError::InvalidDegree {
            axis: ParallelAxis::Ep,
            degree: 0,
        };
        let msg = err.to_string();
        assert!(msg.contains("[CAP-001]"));
        assert!(msg.contains("axis=Ep"));
        assert!(msg.contains("degree=0"));
    }

    #[test]
    fn admission_error_display_composition() {
        let err = AdmissionError::Composition {
            variant: ModelVariant::LlamaQkNorm,
            requested: req(2, 2, 1),
            effective: req(2, 2, 1),
            owner: "CAP-001",
            reason: "PP cannot be combined with TP or EP",
        };
        let msg = err.to_string();
        assert!(msg.starts_with("[CAP-001]"), "msg={msg}");
        assert!(msg.contains("owner=CAP-001"), "msg={msg}");
        assert!(msg.contains("pp=2,tp=2,ep=1"), "msg={msg}");
        assert!(msg.contains("LlamaQkNorm"), "msg={msg}");
    }

    #[test]
    fn admission_error_display_composition_comp_001() {
        let err = AdmissionError::Composition {
            variant: ModelVariant::Qwen35Dense,
            requested: req(1, 2, 2),
            effective: req(1, 2, 2),
            owner: "COMP-001",
            reason: "TP and EP cannot both exceed one (COMP-001)",
        };
        let msg = err.to_string();
        assert!(msg.starts_with("[COMP-001]"), "msg={msg}");
        assert!(msg.contains("owner=COMP-001"), "msg={msg}");
        assert!(msg.contains("tp=2,ep=2"), "msg={msg}");
        assert!(msg.contains("Qwen35Dense"), "msg={msg}");
    }

    #[test]
    fn admission_error_display_planned() {
        let err = AdmissionError::Planned {
            variant: ModelVariant::Qwen35Dense,
            requested: req(2, 1, 1),
            effective: req(2, 1, 1),
            owner: "GEN-001",
            reason: "partial implementation; GEN-001 pending",
        };
        let msg = err.to_string();
        assert!(msg.starts_with("[CAP-001]"), "msg={msg}");
        assert!(msg.contains("owner=GEN-001"), "msg={msg}");
        assert!(msg.contains("pp=2,tp=1,ep=1"), "msg={msg}");
        assert!(msg.contains("Qwen35Dense"), "msg={msg}");
    }

    // ── Original 13×4 golden policy matrix ───────────────────────

    /// Original 13 variant × 4 axis matrix. Gemma4 and Muse Glimmer are
    /// exhaustively pinned by `full_policy_table_new_primary_variants`.
    /// Every cell asserts exact policy, owner, and reason.
    #[test]
    fn full_policy_table() {
        use CellPolicy::*;
        use ModelVariant::*;
        use ParallelAxis::*;

        fn check(v: ModelVariant, a: ParallelAxis, expected: CellPolicy) {
            let got = cell_info(v, a);
            assert_eq!(got, expected, "cell_info({v:?}, {a:?}) mismatch");
        }

        // ── LLaMA QK-norm (arch_id 0, has_qk_norm=true) ────────
        check(LlamaQkNorm, Single, Admitted);
        check(LlamaQkNorm, Pp, Admitted);
        check(LlamaQkNorm, Tp, Admitted);
        check(LlamaQkNorm, Ep, NormalizeToSingle);

        // ── LLaMA no QK-norm (arch_id 0, has_qk_norm=false) ────
        check(LlamaNoQkNorm, Single, Admitted);
        check(LlamaNoQkNorm, Pp, Admitted);
        check(
            LlamaNoQkNorm,
            Tp,
            Unsupported {
                reason: "non-QK-norm LLaMA/Mistral: TP not supported",
            },
        );
        check(LlamaNoQkNorm, Ep, NormalizeToSingle);

        // ── Plain Qwen3 (arch_id 1) ─────────────────────────────
        check(PlainQwen3, Single, Admitted);
        check(PlainQwen3, Pp, Admitted);
        check(PlainQwen3, Tp, Admitted);
        check(PlainQwen3, Ep, NormalizeToSingle);

        // ── Qwen3.5 dense (arch_id 5) ───────────────────────────
        check(Qwen35Dense, Single, Admitted);
        check(
            Qwen35Dense,
            Pp,
            Planned {
                owner: "GEN-001",
                reason: "Qwen3.5 dense PP: partial implementation; GEN-001 pending",
            },
        );
        check(
            Qwen35Dense,
            Tp,
            Planned {
                owner: "AXIS-002",
                reason: "Qwen3.5 dense TP: planned; AXIS-002",
            },
        );
        check(Qwen35Dense, Ep, NormalizeToSingle);

        // ── Qwen3.5 MoE / A3B (arch_id 6) ───────────────────────
        check(Qwen35Moe, Single, Admitted);
        check(
            Qwen35Moe,
            Pp,
            Planned {
                owner: "GEN-001",
                reason: "Qwen3.5 MoE PP: partial implementation; GEN-001 pending",
            },
        );
        check(
            Qwen35Moe,
            Tp,
            Planned {
                owner: "AXIS-002",
                reason: "Qwen3.5 MoE TP: planned; AXIS-002",
            },
        );
        check(
            Qwen35Moe,
            Ep,
            Planned {
                owner: "AXIS-002",
                reason: "Qwen3.5 MoE EP: planned; AXIS-002",
            },
        );

        // ── Qwen3.5-VL (arch_id 5 VL) ───────────────────────────
        check(Qwen35Vl, Single, Admitted);
        check(
            Qwen35Vl,
            Pp,
            Planned {
                owner: "AXIS-004",
                reason: "Qwen3.5-VL PP: planned; AXIS-004",
            },
        );
        check(
            Qwen35Vl,
            Tp,
            Planned {
                owner: "AXIS-004",
                reason: "Qwen3.5-VL TP: planned; AXIS-004",
            },
        );
        check(Qwen35Vl, Ep, NormalizeToSingle);

        // ── Qwen2 dense (arch_id 7) ─────────────────────────────
        check(Qwen2, Single, Admitted);
        check(
            Qwen2,
            Pp,
            Planned {
                owner: "AXIS-001",
                reason: "Qwen2 PP: planned; AXIS-001",
            },
        );
        check(
            Qwen2,
            Tp,
            Planned {
                owner: "AXIS-001",
                reason: "Qwen2 TP: planned; AXIS-001",
            },
        );
        check(Qwen2, Ep, NormalizeToSingle);

        // ── dots.ocr (arch_id 8) ─────────────────────────────────
        check(DotsOcr, Single, Admitted);
        check(
            DotsOcr,
            Pp,
            Planned {
                owner: "AXIS-004",
                reason: "dots.ocr PP: planned; AXIS-004",
            },
        );
        check(
            DotsOcr,
            Tp,
            Planned {
                owner: "AXIS-004",
                reason: "dots.ocr TP: planned; AXIS-004",
            },
        );
        check(DotsOcr, Ep, NormalizeToSingle);

        // ── DeepSeek V4 Flash (arch_id 9) ────────────────────────
        check(Deepseek4, Single, Admitted);
        check(
            Deepseek4,
            Pp,
            Planned {
                owner: "AXIS-003",
                reason: "DeepSeek V4 PP: planned; AXIS-003",
            },
        );
        check(
            Deepseek4,
            Tp,
            Planned {
                owner: "AXIS-003",
                reason: "DeepSeek V4 TP: planned; AXIS-003",
            },
        );
        check(Deepseek4, Ep, Admitted);

        // ── MiniMax (arch_id 10) ─────────────────────────────────
        check(Minimax, Single, Admitted);
        check(
            Minimax,
            Pp,
            Planned {
                owner: "AXIS-003",
                reason: "MiniMax PP: planned; AXIS-003",
            },
        );
        check(
            Minimax,
            Tp,
            Planned {
                owner: "AXIS-003",
                reason: "MiniMax TP: planned; AXIS-003",
            },
        );
        check(Minimax, Ep, Admitted);

        // ── LFM2 dense (arch_id 11, dense) ──────────────────────
        check(
            Lfm2Dense,
            Single,
            Planned {
                owner: "AXIS-003",
                reason: "LFM2 dense Single: planned admission; AXIS-003",
            },
        );
        check(
            Lfm2Dense,
            Pp,
            Planned {
                owner: "AXIS-003",
                reason: "LFM2 dense PP: planned; AXIS-003",
            },
        );
        check(
            Lfm2Dense,
            Tp,
            Planned {
                owner: "AXIS-003",
                reason: "LFM2 dense TP: planned; AXIS-003",
            },
        );
        check(Lfm2Dense, Ep, NormalizeToSingle);

        // ── LFM2 MoE (arch_id 11, MoE) ──────────────────────────
        check(Lfm2Moe, Single, Admitted);
        check(
            Lfm2Moe,
            Pp,
            Planned {
                owner: "AXIS-003",
                reason: "LFM2 MoE PP: planned; AXIS-003",
            },
        );
        check(
            Lfm2Moe,
            Tp,
            Planned {
                owner: "AXIS-003",
                reason: "LFM2 MoE TP: planned; AXIS-003",
            },
        );
        check(
            Lfm2Moe,
            Ep,
            Planned {
                owner: "AXIS-003",
                reason: "LFM2 MoE EP: planned; AXIS-003",
            },
        );

        // ── Cohere2-MoE (arch_id 12) ────────────────────────────
        check(Cohere2Moe, Single, Admitted);
        check(
            Cohere2Moe,
            Pp,
            Planned {
                owner: "AXIS-003",
                reason: "Cohere2-MoE PP: planned; AXIS-003",
            },
        );
        check(
            Cohere2Moe,
            Tp,
            Planned {
                owner: "AXIS-003",
                reason: "Cohere2-MoE TP: planned; AXIS-003",
            },
        );
        check(
            Cohere2Moe,
            Ep,
            Planned {
                owner: "AXIS-003",
                reason: "Cohere2-MoE EP: planned; AXIS-003",
            },
        );
    }

    // ── Gemma4 / Muse Glimmer primary rows (arch_ids 13 / 14) ──

    /// Full policy contract for the new-primary variants `Gemma4`
    /// (arch_id 13) and `MuseGlimmer` (arch_id 14), per
    /// `docs/architecture-ids.md` "Primary model ids".
    ///
    /// Gemma4 is a dense-or-MoE text family and Muse Glimmer is dense text.
    /// Both are single-device: `Single` is admitted; PP, TP, and EP are each
    /// an explicit `Unsupported` cell — never a wildcard fall-through, never
    /// a silent dense-EP normalisation, and never swept into the
    /// DeepSeek4/MiniMax TP→EP remap. Muse Glimmer's
    /// `pp > 1` refusal is production reality (loader carrier gate); Gemma4
    /// has the same carrier gate. Sidecar ids 22 (Gemma4 EAGLE) and 23
    /// (Glimmer DFlash draft) are NOT primary topology rows and have no
    /// variant here.
    #[test]
    fn full_policy_table_new_primary_variants() {
        use CellPolicy::*;
        use ModelVariant::*;
        use ParallelAxis::*;

        fn check(v: ModelVariant, a: ParallelAxis, expected: CellPolicy) {
            let got = cell_info(v, a);
            assert_eq!(got, expected, "cell_info({v:?}, {a:?}) mismatch");
        }

        // ── Gemma4 (arch_id 13, dense or MoE text) ───────────────
        check(Gemma4, Single, Admitted);
        check(
            Gemma4,
            Pp,
            Unsupported {
                reason: "Gemma4 PP: pp>1 unsupported",
            },
        );
        check(
            Gemma4,
            Tp,
            Unsupported {
                reason: "Gemma4 TP: no tensor-parallel path; single-device decode only",
            },
        );
        check(
            Gemma4,
            Ep,
            Unsupported {
                reason: "Gemma4 EP: no expert-parallel path",
            },
        );

        // ── Muse Glimmer (arch_id 14, dense text) ───────────────
        check(MuseGlimmer, Single, Admitted);
        check(
            MuseGlimmer,
            Pp,
            Unsupported {
                reason: "Muse Glimmer PP: pp>1 unsupported",
            },
        );
        check(
            MuseGlimmer,
            Tp,
            Unsupported {
                reason: "Muse Glimmer TP: no tensor-parallel path; single-device dense decode only",
            },
        );
        check(
            MuseGlimmer,
            Ep,
            Unsupported {
                reason: "Muse Glimmer EP: no expert-parallel path; dense EP is not normalized",
            },
        );
    }

    #[test]
    fn gemma4_single_admitted() {
        let admission = resolve(ModelVariant::Gemma4, req(1, 1, 1)).unwrap();
        assert_eq!(admission.variant(), ModelVariant::Gemma4);
        assert_eq!(admission.effective(), req(1, 1, 1));
        assert!(!admission.was_normalized());
    }

    /// Every non-Single Gemma4 axis is an explicit `Unsupported` cell:
    /// PP carries the carrier's `pp>1` refusal; TP must not be remapped to
    /// EP (the DeepSeek4/MiniMax legacy remap is variant-scoped); EP must
    /// not be normalised to (1,1,1) (Gemma4 has no CAP-001 dense-EP cell).
    #[test]
    fn gemma4_pp_tp_ep_explicitly_unsupported() {
        for (pp, tp, ep, reason, axis) in [
            (2, 1, 1, "pp>1 unsupported", ParallelAxis::Pp),
            (1, 2, 1, "no tensor-parallel path", ParallelAxis::Tp),
            (1, 1, 2, "no expert-parallel path", ParallelAxis::Ep),
        ] {
            let err = assert_refused(ModelVariant::Gemma4, pp, tp, ep, "CAP-001", reason);
            assert_eq!(err.effective_axis(), Some(axis));
            match &err {
                AdmissionError::Unsupported { effective, .. } => {
                    // Effective degrees are preserved verbatim: no TP→EP
                    // remap and no dense-EP normalisation for Gemma4.
                    assert_eq!(*effective, req(pp, tp, ep));
                }
                other => panic!("expected Unsupported variant, got {other:?}"),
            }
        }
    }

    #[test]
    fn museglimmer_single_admitted() {
        let admission = resolve(ModelVariant::MuseGlimmer, req(1, 1, 1)).unwrap();
        assert_eq!(admission.variant(), ModelVariant::MuseGlimmer);
        assert_eq!(admission.effective(), req(1, 1, 1));
        assert!(!admission.was_normalized());
    }

    /// Every non-Single Muse Glimmer axis is an explicit `Unsupported`
    /// cell: PP carries the arch's `pp>1` refusal; TP must not be remapped
    /// to EP; EP must not be normalised to (1,1,1) (Muse Glimmer has no
    /// CAP-001 dense-EP cell).
    #[test]
    fn museglimmer_pp_tp_ep_explicitly_unsupported() {
        for (pp, tp, ep, reason, axis) in [
            (2, 1, 1, "pp>1 unsupported", ParallelAxis::Pp),
            (1, 2, 1, "no tensor-parallel path", ParallelAxis::Tp),
            (1, 1, 2, "no expert-parallel path", ParallelAxis::Ep),
        ] {
            let err = assert_refused(ModelVariant::MuseGlimmer, pp, tp, ep, "CAP-001", reason);
            assert_eq!(err.effective_axis(), Some(axis));
            match &err {
                AdmissionError::Unsupported { effective, .. } => {
                    // Effective degrees are preserved verbatim: no TP→EP
                    // remap and no dense-EP normalisation for Muse Glimmer.
                    assert_eq!(*effective, req(pp, tp, ep));
                }
                other => panic!("expected Unsupported variant, got {other:?}"),
            }
        }
    }

    // ── Qwen3.8 family coverage (primary ids 5 / 6) ─────────────

    /// Qwen3.8 has no runtime id of its own: dense Qwen3.8 rides primary
    /// id 5 and MoE/A3B Qwen3.8 rides id 6 (`docs/architecture-ids.md`,
    /// "Qwen3.5 / 3.6 / 3.8 dense" and "Qwen3.5 / 3.6 / 3.8 MoE (A3B)").
    /// Its admission contract is therefore exactly the `Qwen35Dense` /
    /// `Qwen35Moe` rows — no new variant, no new policy row, no invented
    /// id. This pin keeps a future "Qwen3.8" variant (which would invent an
    /// id) from silently changing Qwen3.8 admission.
    #[test]
    fn qwen38_family_is_covered_by_qwen35_rows() {
        // Single: admitted on both dense (id 5) and MoE (id 6) rows.
        for v in [ModelVariant::Qwen35Dense, ModelVariant::Qwen35Moe] {
            let admission = resolve(v, req(1, 1, 1)).unwrap();
            assert_eq!(admission.variant(), v);
            assert_eq!(admission.effective(), req(1, 1, 1));
        }

        // Dense EP (id 5): CAP-001 normalisation to (1,1,1), no EP claim.
        let admission = resolve(ModelVariant::Qwen35Dense, req(1, 1, 2)).unwrap();
        assert_eq!(admission.effective(), req(1, 1, 1));
        assert!(admission.was_normalized());

        // MoE EP (id 6): planned under AXIS-002 — refused with owner+reason.
        let err = assert_refused(
            ModelVariant::Qwen35Moe,
            1,
            1,
            2,
            "CAP-001",
            "planned; AXIS-002",
        );
        match &err {
            AdmissionError::Planned {
                owner, effective, ..
            } => {
                assert_eq!(*owner, "AXIS-002");
                assert_eq!(*effective, req(1, 1, 2));
            }
            other => panic!("expected Planned variant, got {other:?}"),
        }

        // PP (id 5): partial under GEN-001 — refused with owner+reason.
        let err = assert_refused(
            ModelVariant::Qwen35Dense,
            2,
            1,
            1,
            "CAP-001",
            "GEN-001 pending",
        );
        match &err {
            AdmissionError::Planned { owner, .. } => assert_eq!(*owner, "GEN-001"),
            other => panic!("expected Planned variant, got {other:?}"),
        }

        // TP (id 5): planned under AXIS-002 — refused with owner+reason.
        let err = assert_refused(
            ModelVariant::Qwen35Dense,
            1,
            2,
            1,
            "CAP-001",
            "planned; AXIS-002",
        );
        match &err {
            AdmissionError::Planned { owner, .. } => assert_eq!(*owner, "AXIS-002"),
            other => panic!("expected Planned variant, got {other:?}"),
        }
    }
}
