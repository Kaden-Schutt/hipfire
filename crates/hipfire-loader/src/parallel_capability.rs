// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Loader-owned admission for the executable PP/TP/EP surface.
//!
//! This module is host-only apart from constructing the pure G1
//! [`hipfire_hardware::DeviceMesh`] returned on success. It reads no model
//! source, creates no devices, binds no mesh owner, and allocates no GPU state.
//! The loader classifies a concrete model source into [`ModelVariant`] and
//! [`SourceKind`], then calls [`resolve`] before entering a carrier or an
//! axis-specific constructor.
//!
//! The policy is conservative: a cell is admitted only when the current
//! upstream loader has an executable route. Physical-device checks (for
//! example peer access and exact GPU architecture) remain in that route; they
//! must not turn an unsupported cell into a fallback.
//!
//! Resolution order is part of the diagnostic contract:
//!
//! 1. reject the first zero degree (`CAP-001`);
//! 2. reject TP×EP, then PP×(TP|EP), before any remap (`COMP-001`/`CAP-001`);
//! 3. remap the legacy DeepSeek4/MiniMax `tp` spelling to EP;
//! 4. evaluate one source-aware policy cell, normalizing dense EP to Single;
//! 5. apply the few current-route degree bounds (Qwen dense TP and MoE EP).

use hipfire_hardware::{DeviceMesh, DimKind, MeshError};

/// Source namespace used by a model load.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SourceKind {
    /// Native `.hfq` source.
    Hfq,
    /// HuggingFace safetensors directory.
    SafetensorsDir,
}

impl SourceKind {
    /// Stable diagnostic name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Hfq => "HFQ",
            Self::SafetensorsDir => "safetensors-dir",
        }
    }
}

/// Parallelism axis selected by a degree request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ParallelAxis {
    /// All degrees are one.
    Single,
    /// Pipeline parallelism.
    Pp,
    /// Tensor parallelism.
    Tp,
    /// Expert parallelism.
    Ep,
}

impl ParallelAxis {
    /// Stable short name for diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Pp => "PP",
            Self::Tp => "TP",
            Self::Ep => "EP",
        }
    }
}

/// Raw requested degree for each parallelism axis.
///
/// All axes must be at least one. A zero is rejected before composition checks,
/// compatibility remapping, policy lookup, or any loader side effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RawParallelism {
    pub pp: usize,
    pub tp: usize,
    pub ep: usize,
}

impl RawParallelism {
    pub const fn new(pp: usize, tp: usize, ep: usize) -> Self {
        Self { pp, tp, ep }
    }

    /// Return the dominant requested axis. PP is checked first so a malformed
    /// composed request has a deterministic axis even before it is rejected.
    pub const fn axis(self) -> ParallelAxis {
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

/// Source-aware family/shape classification used by the policy table.
///
/// Variants are facts about the model and its executable carrier, not a
/// requested axis. Qwen3.5 and LFM2 variants are split by expert/vision
/// metadata rather than being inferred from `arch_id` alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelVariant {
    /// LLaMA/Mistral with QK-norm weights.
    LlamaQkNorm,
    /// LLaMA/Mistral without QK-norm weights.
    LlamaNoQkNorm,
    /// Plain Qwen3 (the LLaMA-family carrier, arch id 1).
    PlainQwen3,
    /// Qwen3.5 dense text.
    Qwen35Dense,
    /// Qwen3.5/3.6 MoE text.
    Qwen35Moe,
    /// Qwen3.5 dense vision-language model.
    Qwen35DenseVl,
    /// Qwen3.5 MoE vision-language model.
    Qwen35MoeVl,
    /// Standalone Qwen2 text.
    Qwen2,
    /// DeepSeek V4 Flash.
    Deepseek4,
    /// MiniMax-M2.
    Minimax,
    /// LFM2 dense text.
    Lfm2Dense,
    /// LFM2 MoE text.
    Lfm2Moe,
    /// LFM2-VL.
    Lfm2Vl,
    /// Standalone Dots.OCR vision/text model.
    DotsOcr,
    /// Cohere2-MoE/North-Mini-Code.
    Cohere2Moe,
    /// Maple native ternary model.
    Maple,
    /// Gemma4 text target.
    Gemma4,
    /// Muse Glimmer text target.
    MuseGlimmer,
}

/// One cell in the executable source-aware matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellPolicy {
    /// A current loader/executor path exists for this source and axis.
    Admitted,
    /// Dense EP is accepted as a request but canonicalized to Single before
    /// the loader is entered. The Single cell is then evaluated again.
    NormalizeToSingle,
    /// No current executable route exists. This is a hard refusal, not a
    /// signal to fall back to another axis or source implementation.
    Unsupported {
        /// Stable owner/category tag.
        owner: &'static str,
        /// Technical refusal reason.
        reason: &'static str,
    },
}

/// Typed refusal from the loader admission point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    /// A requested axis has degree zero. The first zero in PP, TP, EP order
    /// wins so diagnostics are deterministic for an all-zero request.
    InvalidDegree { axis: ParallelAxis, degree: usize },
    /// The effective parallel shape could not be represented by the device
    /// mesh without losing cardinality information.
    Topology {
        source: SourceKind,
        variant: ModelVariant,
        requested: RawParallelism,
        effective: RawParallelism,
        error: MeshError,
    },
    /// A forbidden multi-axis composition. Composition is checked against the
    /// raw request before compatibility remapping or normalization.
    Composition {
        source: SourceKind,
        variant: ModelVariant,
        requested: RawParallelism,
        owner: &'static str,
        reason: &'static str,
    },
    /// A policy cell or current-route degree bound refused the request.
    Unsupported {
        source: SourceKind,
        variant: ModelVariant,
        requested: RawParallelism,
        effective: RawParallelism,
        owner: &'static str,
        reason: &'static str,
    },
}

impl AdmissionError {
    /// Stable diagnostic owner/category.
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidDegree { .. } => "CAP-001",
            Self::Topology { .. } => "TOPO-001",
            Self::Composition { owner, .. } | Self::Unsupported { owner, .. } => owner,
        }
    }

    pub const fn source(&self) -> Option<SourceKind> {
        match self {
            Self::InvalidDegree { .. } => None,
            Self::Topology { source, .. }
            | Self::Composition { source, .. }
            | Self::Unsupported { source, .. } => Some(*source),
        }
    }

    pub const fn variant(&self) -> Option<ModelVariant> {
        match self {
            Self::InvalidDegree { .. } => None,
            Self::Topology { variant, .. }
            | Self::Composition { variant, .. }
            | Self::Unsupported { variant, .. } => Some(*variant),
        }
    }

    pub const fn effective(&self) -> Option<RawParallelism> {
        match self {
            Self::InvalidDegree { .. } | Self::Composition { .. } => None,
            Self::Topology { effective, .. } | Self::Unsupported { effective, .. } => {
                Some(*effective)
            }
        }
    }

    pub const fn reason(&self) -> &'static str {
        match self {
            Self::InvalidDegree { .. } => "every parallelism degree must be >= 1",
            Self::Topology { error, .. } => match error {
                MeshError::CardinalityOverflow => "device mesh cardinality overflow",
                MeshError::DuplicateAxis(_) => "device mesh axis repeated",
                MeshError::InvalidDevice { .. } => "device is not present in the mesh",
                MeshError::RankMismatch { .. } => "device mesh coordinate rank mismatch",
                MeshError::CoordinateOutOfBounds { .. } => {
                    "device mesh coordinate is out of bounds"
                }
            },
            Self::Composition { reason, .. } | Self::Unsupported { reason, .. } => reason,
        }
    }
}

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDegree { axis, degree } => {
                write!(f, "[CAP-001] invalid {} degree {}", axis.name(), degree)
            }
            Self::Topology {
                source,
                variant,
                requested,
                effective,
                error,
            } => write!(
                f,
                "[TOPO-001] {} {:?} topology refused (requested pp={},tp={},ep={}; effective pp={},tp={},ep={}): {error}",
                source.name(),
                variant,
                requested.pp,
                requested.tp,
                requested.ep,
                effective.pp,
                effective.tp,
                effective.ep,
            ),
            Self::Composition {
                source,
                variant,
                requested,
                owner,
                reason,
            } => write!(
                f,
                "[{owner}] {} {:?} composition refused (pp={},tp={},ep={}): {reason}",
                source.name(),
                variant,
                requested.pp,
                requested.tp,
                requested.ep,
            ),
            Self::Unsupported {
                source,
                variant,
                requested,
                effective,
                owner,
                reason,
            } => write!(
                f,
                "[{owner}] {} {:?} unsupported (requested pp={},tp={},ep={}; effective pp={},tp={},ep={}): {reason}",
                source.name(),
                variant,
                requested.pp,
                requested.tp,
                requested.ep,
                effective.pp,
                effective.tp,
                effective.ep,
            ),
        }
    }
}

/// Resolve one source-aware raw degree request to the effective G1 mesh.
///
/// This is the sole policy/admission operation. It performs no GPU or file
/// work. Composition rejection runs before legacy remapping and dense-EP
/// normalization. Dense normalization is performed at most once by the table
/// cell, then the Single cell is evaluated directly.
pub fn resolve(
    source: SourceKind,
    variant: ModelVariant,
    raw: RawParallelism,
) -> Result<DeviceMesh, AdmissionError> {
    // 1. Degree-zero refusal has precedence over every other diagnostic.
    let invalid_axis = if raw.pp == 0 {
        Some(ParallelAxis::Pp)
    } else if raw.tp == 0 {
        Some(ParallelAxis::Tp)
    } else if raw.ep == 0 {
        Some(ParallelAxis::Ep)
    } else {
        None
    };
    if let Some(axis) = invalid_axis {
        return Err(AdmissionError::InvalidDegree { axis, degree: 0 });
    }

    // 2. Composition refusal precedes both compatibility remapping and dense
    // EP normalization. TP×EP owns COMP-001; PP compositions own CAP-001.
    if raw.tp > 1 && raw.ep > 1 {
        return Err(AdmissionError::Composition {
            source,
            variant,
            requested: raw,
            owner: "COMP-001",
            reason: "TP and EP cannot both exceed one",
        });
    }
    if raw.pp > 1 && (raw.tp > 1 || raw.ep > 1) {
        return Err(AdmissionError::Composition {
            source,
            variant,
            requested: raw,
            owner: "CAP-001",
            reason: "PP cannot be combined with TP or EP",
        });
    }

    // 3. Legacy EP entrypoints historically called their degree `tp` for
    // DeepSeek4 and MiniMax. Preserve that one executable compatibility
    // mapping, but never remap a request that already carries EP.
    let mut effective = raw;
    if matches!(variant, ModelVariant::Deepseek4 | ModelVariant::Minimax)
        && effective.tp > 1
        && effective.ep == 1
    {
        effective.ep = effective.tp;
        effective.tp = 1;
    }

    // 4. One source-aware table lookup. Dense EP canonicalizes exactly once
    // and re-evaluates the Single cell, so no caller can allocate against the
    // requested EP degree.
    let axis = effective.axis();
    let policy = cell_info(source, variant, axis);
    let effective = match policy {
        CellPolicy::NormalizeToSingle => {
            let normalized = RawParallelism::new(1, 1, 1);
            match cell_info(source, variant, ParallelAxis::Single) {
                CellPolicy::Admitted => normalized,
                CellPolicy::NormalizeToSingle => unreachable!("Single policy cannot normalize"),
                CellPolicy::Unsupported { owner, reason } => {
                    return Err(AdmissionError::Unsupported {
                        source,
                        variant,
                        requested: raw,
                        effective: normalized,
                        owner,
                        reason,
                    });
                }
            }
        }
        CellPolicy::Admitted => effective,
        CellPolicy::Unsupported { owner, reason } => {
            return Err(AdmissionError::Unsupported {
                source,
                variant,
                requested: raw,
                effective,
                owner,
                reason,
            });
        }
    };

    // 5. Degree bounds are still host-only. They are kept here so a request
    // that the current route cannot execute is refused before Gpus::init_*.
    if let Some(reason) = current_degree_error(source, variant, effective) {
        return Err(AdmissionError::Unsupported {
            source,
            variant,
            requested: raw,
            effective,
            owner: "CAP-001",
            reason,
        });
    }

    mesh_for(effective).map_err(|error| AdmissionError::Topology {
        source,
        variant,
        requested: raw,
        effective,
        error,
    })
}

fn current_degree_error(
    source: SourceKind,
    variant: ModelVariant,
    effective: RawParallelism,
) -> Option<&'static str> {
    match (source, variant, effective.axis()) {
        (SourceKind::Hfq, ModelVariant::Qwen35Dense, ParallelAxis::Tp)
            if !(2..=5).contains(&effective.tp) =>
        {
            Some("Qwen3.5 dense TP currently supports degrees 2..=5")
        }
        (SourceKind::Hfq, ModelVariant::Qwen35Moe, ParallelAxis::Ep) if effective.ep != 4 => {
            Some("Qwen3.5 MoE EP currently requires degree 4")
        }
        _ => None,
    }
}

/// Build the effective rectangular G1 topology. Size-one axes are omitted;
/// [`DeviceMesh::single`] is the canonical one-device representation.
fn mesh_for(request: RawParallelism) -> Result<DeviceMesh, MeshError> {
    if request.axis() == ParallelAxis::Single {
        return DeviceMesh::single();
    }
    let mut axes = Vec::with_capacity(3);
    if request.pp > 1 {
        axes.push((DimKind::Pp, request.pp));
    }
    if request.tp > 1 {
        axes.push((DimKind::Tp, request.tp));
    }
    if request.ep > 1 {
        axes.push((DimKind::Ep, request.ep));
    }
    DeviceMesh::rect(&axes)
}

/// The one source-aware PP/TP/EP policy table.
///
/// Every registered family has a row for each axis. A source wildcard means
/// that both source kinds share the same executable route; source-specific rows
/// document current HFQ-only parallel constructors explicitly.
pub fn cell_info(source: SourceKind, variant: ModelVariant, axis: ParallelAxis) -> CellPolicy {
    use CellPolicy::{Admitted, NormalizeToSingle, Unsupported};
    use ModelVariant::*;
    use ParallelAxis::*;
    use SourceKind::*;

    match (source, variant, axis) {
        // LLaMA-family carriers are single-device in the current upstream
        // loader. Dense EP is a deliberate canonicalization to that route.
        (_, LlamaQkNorm, Single) => Admitted,
        (_, LlamaQkNorm, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "LLaMA PP has no current loader route",
        },
        (_, LlamaQkNorm, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "LLaMA TP has no current loader route",
        },
        (_, LlamaQkNorm, Ep) => NormalizeToSingle,
        (_, LlamaNoQkNorm, Single) => Admitted,
        (_, LlamaNoQkNorm, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "LLaMA PP has no current loader route",
        },
        (_, LlamaNoQkNorm, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "non-QK-norm LLaMA TP has no current loader route",
        },
        (_, LlamaNoQkNorm, Ep) => NormalizeToSingle,
        (_, PlainQwen3, Single) => Admitted,
        (_, PlainQwen3, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "plain Qwen3 PP has no current loader route",
        },
        (_, PlainQwen3, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "plain Qwen3 TP has no current loader route",
        },
        (_, PlainQwen3, Ep) => NormalizeToSingle,

        // Qwen3.5 PP is an HFQ-only current route. The carrier's PP branch
        // intentionally skips the vision tower, so VL must refuse here.
        (_, Qwen35Dense, Single) => Admitted,
        (Hfq, Qwen35Dense, Pp) => Admitted,
        (SafetensorsDir, Qwen35Dense, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "Qwen3.5 safetensors PP has no current loader route",
        },
        (Hfq, Qwen35Dense, Tp) => Admitted,
        (SafetensorsDir, Qwen35Dense, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "Qwen3.5 safetensors TP has no current loader route",
        },
        (_, Qwen35Dense, Ep) => NormalizeToSingle,
        (_, Qwen35Moe, Single) => Admitted,
        (Hfq, Qwen35Moe, Pp) => Admitted,
        (SafetensorsDir, Qwen35Moe, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "Qwen3.5 MoE safetensors PP has no current loader route",
        },
        (_, Qwen35Moe, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "Qwen3.5 MoE TP has no current loader route",
        },
        (Hfq, Qwen35Moe, Ep) => Admitted,
        (SafetensorsDir, Qwen35Moe, Ep) => Unsupported {
            owner: "CAP-001",
            reason: "Qwen3.5 MoE safetensors EP has no current loader route",
        },
        (Hfq, Qwen35DenseVl, Single) => Admitted,
        (SafetensorsDir, Qwen35DenseVl, Single) => Unsupported {
            owner: "CAP-001",
            reason: "Qwen3.5 dense-VL safetensors vision load has no current route",
        },
        (_, Qwen35DenseVl, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "Qwen3.5 dense-VL PP would skip the vision tower",
        },
        (_, Qwen35DenseVl, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "Qwen3.5 dense-VL TP has no current loader route",
        },
        (Hfq, Qwen35DenseVl, Ep) => NormalizeToSingle,
        (SafetensorsDir, Qwen35DenseVl, Ep) => Unsupported {
            owner: "CAP-001",
            reason: "Qwen3.5 dense-VL safetensors vision load has no current route",
        },
        (Hfq, Qwen35MoeVl, Single) => Admitted,
        (SafetensorsDir, Qwen35MoeVl, Single) => Unsupported {
            owner: "CAP-001",
            reason: "Qwen3.5 MoE-VL safetensors vision load has no current route",
        },
        (_, Qwen35MoeVl, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "Qwen3.5 MoE-VL PP would skip the vision tower",
        },
        (_, Qwen35MoeVl, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "Qwen3.5 MoE-VL TP has no current loader route",
        },
        (_, Qwen35MoeVl, Ep) => Unsupported {
            owner: "CAP-001",
            reason: "Qwen3.5 MoE-VL EP has no current loader route",
        },

        // Standalone dense/VL carriers have executable Single routes only.
        (_, Qwen2, Single) => Admitted,
        (_, Qwen2, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "Qwen2 PP has no current loader route",
        },
        (_, Qwen2, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "Qwen2 TP has no current loader route",
        },
        (_, Qwen2, Ep) => NormalizeToSingle,
        (_, DotsOcr, Single) => Admitted,
        (_, DotsOcr, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "dots.ocr PP has no current loader route",
        },
        (_, DotsOcr, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "dots.ocr TP has no current loader route",
        },
        (_, DotsOcr, Ep) => NormalizeToSingle,

        // DeepSeek4/MiniMax EP constructors reopen HFQ per rank. Their
        // compatibility spelling is handled above; directories refuse before
        // that constructor can bind devices.
        (_, Deepseek4, Single) => Admitted,
        (_, Deepseek4, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "DeepSeek4 PP has no current loader route",
        },
        (_, Deepseek4, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "DeepSeek4 TP has no current loader route",
        },
        (Hfq, Deepseek4, Ep) => Admitted,
        (SafetensorsDir, Deepseek4, Ep) => Unsupported {
            owner: "CAP-001",
            reason: "DeepSeek4 safetensors EP has no current loader route",
        },
        (_, Minimax, Single) => Admitted,
        (_, Minimax, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "MiniMax PP has no current loader route",
        },
        (_, Minimax, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "MiniMax TP has no current loader route",
        },
        (Hfq, Minimax, Ep) => Admitted,
        (SafetensorsDir, Minimax, Ep) => Unsupported {
            owner: "CAP-001",
            reason: "MiniMax safetensors EP has no current loader route",
        },

        // LFM2's current carrier executes dense and MoE Single. VL is HFQ
        // only because the directory branch currently loads text only.
        (_, Lfm2Dense, Single) => Admitted,
        (_, Lfm2Dense, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "LFM2 dense PP has no current loader route",
        },
        (_, Lfm2Dense, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "LFM2 dense TP has no current loader route",
        },
        (_, Lfm2Dense, Ep) => NormalizeToSingle,
        (_, Lfm2Moe, Single) => Admitted,
        (_, Lfm2Moe, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "LFM2 MoE PP has no current loader route",
        },
        (_, Lfm2Moe, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "LFM2 MoE TP has no current loader route",
        },
        (_, Lfm2Moe, Ep) => Unsupported {
            owner: "CAP-001",
            reason: "LFM2 MoE EP has no current loader route",
        },
        (Hfq, Lfm2Vl, Single) => Admitted,
        (SafetensorsDir, Lfm2Vl, Single) => Unsupported {
            owner: "CAP-001",
            reason: "LFM2-VL safetensors vision load has no current route",
        },
        (_, Lfm2Vl, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "LFM2-VL PP has no current loader route",
        },
        (_, Lfm2Vl, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "LFM2-VL TP has no current loader route",
        },
        (Hfq, Lfm2Vl, Ep) => NormalizeToSingle,
        (SafetensorsDir, Lfm2Vl, Ep) => Unsupported {
            owner: "CAP-001",
            reason: "LFM2-VL safetensors vision load has no current route",
        },

        (_, Cohere2Moe, Single) => Admitted,
        (_, Cohere2Moe, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "Cohere2-MoE PP has no current loader route",
        },
        (_, Cohere2Moe, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "Cohere2-MoE TP has no current loader route",
        },
        (_, Cohere2Moe, Ep) => Unsupported {
            owner: "CAP-001",
            reason: "Cohere2-MoE EP has no current loader route",
        },
        (Hfq, Maple, Single) => Admitted,
        (SafetensorsDir, Maple, Single) => Unsupported {
            owner: "CAP-001",
            reason: "Maple safetensors load is unsupported; convert to HFQ",
        },
        (_, Maple, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "Maple PP has no current loader route",
        },
        (_, Maple, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "Maple TP has no current loader route",
        },
        (_, Maple, Ep) => Unsupported {
            owner: "CAP-001",
            reason: "Maple EP has no current loader route",
        },
        (Hfq, Gemma4, Single) => Admitted,
        (SafetensorsDir, Gemma4, Single) => Unsupported {
            owner: "CAP-001",
            reason: "Gemma4 safetensors load is not wired",
        },
        (_, Gemma4, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "Gemma4 PP has no current loader route",
        },
        (_, Gemma4, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "Gemma4 TP has no current loader route",
        },
        (_, Gemma4, Ep) => Unsupported {
            owner: "CAP-001",
            reason: "Gemma4 EP has no current loader route",
        },
        (Hfq, MuseGlimmer, Single) => Admitted,
        (SafetensorsDir, MuseGlimmer, Single) => Unsupported {
            owner: "CAP-001",
            reason: "Muse Glimmer safetensors load is not wired",
        },
        (_, MuseGlimmer, Pp) => Unsupported {
            owner: "CAP-001",
            reason: "Muse Glimmer PP has no current loader route",
        },
        (_, MuseGlimmer, Tp) => Unsupported {
            owner: "CAP-001",
            reason: "Muse Glimmer TP has no current loader route",
        },
        (_, MuseGlimmer, Ep) => Unsupported {
            owner: "CAP-001",
            reason: "Muse Glimmer EP has no current loader route",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn req(pp: usize, tp: usize, ep: usize) -> RawParallelism {
        RawParallelism::new(pp, tp, ep)
    }

    #[test]
    fn policy_table_covers_current_executable_cells() {
        assert_eq!(
            cell_info(SourceKind::Hfq, ModelVariant::Qwen35Dense, ParallelAxis::Pp),
            CellPolicy::Admitted
        );
        assert!(matches!(
            cell_info(
                SourceKind::SafetensorsDir,
                ModelVariant::Qwen35Dense,
                ParallelAxis::Pp
            ),
            CellPolicy::Unsupported { .. }
        ));
        assert_eq!(
            cell_info(SourceKind::Hfq, ModelVariant::Qwen35Dense, ParallelAxis::Tp),
            CellPolicy::Admitted
        );
        assert_eq!(
            cell_info(SourceKind::Hfq, ModelVariant::Qwen35Moe, ParallelAxis::Ep),
            CellPolicy::Admitted
        );
        assert!(matches!(
            cell_info(
                SourceKind::SafetensorsDir,
                ModelVariant::Qwen35Moe,
                ParallelAxis::Ep
            ),
            CellPolicy::Unsupported { .. }
        ));
        assert_eq!(
            cell_info(SourceKind::Hfq, ModelVariant::Deepseek4, ParallelAxis::Ep),
            CellPolicy::Admitted
        );
        assert_eq!(
            cell_info(SourceKind::Hfq, ModelVariant::Minimax, ParallelAxis::Ep),
            CellPolicy::Admitted
        );
        assert_eq!(
            cell_info(SourceKind::Hfq, ModelVariant::LlamaQkNorm, ParallelAxis::Ep),
            CellPolicy::NormalizeToSingle
        );
        assert!(matches!(
            cell_info(SourceKind::Hfq, ModelVariant::Gemma4, ParallelAxis::Pp),
            CellPolicy::Unsupported { .. }
        ));
    }

    #[test]
    fn dots_ocr_policy_is_explicit_across_all_axes() {
        assert_eq!(
            cell_info(SourceKind::Hfq, ModelVariant::DotsOcr, ParallelAxis::Single),
            CellPolicy::Admitted
        );
        assert!(matches!(
            cell_info(SourceKind::Hfq, ModelVariant::DotsOcr, ParallelAxis::Pp),
            CellPolicy::Unsupported { .. }
        ));
        assert!(matches!(
            cell_info(SourceKind::Hfq, ModelVariant::DotsOcr, ParallelAxis::Tp),
            CellPolicy::Unsupported { .. }
        ));
        assert_eq!(
            cell_info(SourceKind::Hfq, ModelVariant::DotsOcr, ParallelAxis::Ep),
            CellPolicy::NormalizeToSingle
        );
        let mesh = resolve(SourceKind::Hfq, ModelVariant::DotsOcr, req(1, 1, 4)).unwrap();
        assert_eq!(mesh.n_devices(), 1);
        assert_eq!(mesh.axes(), &[]);
    }

    #[test]
    fn dense_and_moe_vl_have_disjoint_ep_policies() {
        assert_eq!(
            cell_info(
                SourceKind::Hfq,
                ModelVariant::Qwen35DenseVl,
                ParallelAxis::Ep
            ),
            CellPolicy::NormalizeToSingle
        );
        assert!(matches!(
            cell_info(SourceKind::Hfq, ModelVariant::Qwen35MoeVl, ParallelAxis::Ep),
            CellPolicy::Unsupported { .. }
        ));

        let dense = resolve(SourceKind::Hfq, ModelVariant::Qwen35DenseVl, req(1, 1, 4)).unwrap();
        assert_eq!(dense.n_devices(), 1);
        assert!(!dense.has_axis(DimKind::Ep));

        let moe = resolve(SourceKind::Hfq, ModelVariant::Qwen35MoeVl, req(1, 1, 4)).unwrap_err();
        assert!(moe.reason().contains("MoE-VL EP"));
    }

    #[test]
    fn zero_degree_wins_over_composition_and_policy() {
        let err = resolve(SourceKind::Hfq, ModelVariant::Qwen35Moe, req(0, 2, 2)).unwrap_err();
        assert_eq!(err.code(), "CAP-001");
        assert!(matches!(
            err,
            AdmissionError::InvalidDegree {
                axis: ParallelAxis::Pp,
                degree: 0
            }
        ));

        let err = resolve(SourceKind::Hfq, ModelVariant::Qwen35Moe, req(2, 0, 2)).unwrap_err();
        assert!(matches!(
            err,
            AdmissionError::InvalidDegree {
                axis: ParallelAxis::Tp,
                ..
            }
        ));
        let err = resolve(SourceKind::Hfq, ModelVariant::Qwen35Moe, req(2, 2, 0)).unwrap_err();
        assert!(matches!(
            err,
            AdmissionError::InvalidDegree {
                axis: ParallelAxis::Ep,
                ..
            }
        ));
    }

    #[test]
    fn composition_precedes_legacy_remap_and_dense_normalization() {
        let err = resolve(SourceKind::Hfq, ModelVariant::Deepseek4, req(1, 2, 2)).unwrap_err();
        assert_eq!(err.code(), "COMP-001");
        assert!(err.reason().contains("TP and EP"));

        let err = resolve(SourceKind::Hfq, ModelVariant::Qwen35Dense, req(2, 2, 1)).unwrap_err();
        assert_eq!(err.code(), "CAP-001");
        assert!(err.reason().contains("PP cannot"));
    }

    #[test]
    fn deepseek_and_minimax_legacy_tp_remap_preserves_degree() {
        for variant in [ModelVariant::Deepseek4, ModelVariant::Minimax] {
            let mesh = resolve(SourceKind::Hfq, variant, req(1, 4, 1)).unwrap();
            assert_eq!(mesh.size_of(DimKind::Tp), 1);
            assert_eq!(mesh.size_of(DimKind::Ep), 4);
            assert_eq!(mesh.n_devices(), 4);
        }
    }

    #[test]
    fn dense_ep_normalizes_once_to_single() {
        let mesh = resolve(SourceKind::Hfq, ModelVariant::Qwen35Dense, req(1, 1, 7)).unwrap();
        assert_eq!(mesh.n_devices(), 1);
        assert!(!mesh.has_axis(DimKind::Ep));
        assert_eq!(mesh.axes(), &[]);

        let mesh = resolve(
            SourceKind::SafetensorsDir,
            ModelVariant::Lfm2Dense,
            req(1, 1, 2),
        )
        .unwrap();
        assert_eq!(mesh.n_devices(), 1);
        assert_eq!(mesh.axes(), &[]);
    }

    #[test]
    fn current_route_degree_bounds_refuse_before_executor() {
        let err = resolve(SourceKind::Hfq, ModelVariant::Qwen35Dense, req(1, 6, 1)).unwrap_err();
        assert!(err.reason().contains("2..=5"));
        let err = resolve(SourceKind::Hfq, ModelVariant::Qwen35Moe, req(1, 1, 2)).unwrap_err();
        assert!(err.reason().contains("requires degree 4"));
        let mesh = resolve(SourceKind::Hfq, ModelVariant::Qwen35Moe, req(1, 1, 4)).unwrap();
        assert_eq!(mesh.size_of(DimKind::Ep), 4);
    }

    #[test]
    fn unsupported_source_refuses_without_mesh_or_executor() {
        let err = resolve(
            SourceKind::SafetensorsDir,
            ModelVariant::Deepseek4,
            req(1, 1, 2),
        )
        .unwrap_err();
        assert_eq!(err.code(), "CAP-001");
        assert_eq!(err.source(), Some(SourceKind::SafetensorsDir));
        assert_eq!(err.variant(), Some(ModelVariant::Deepseek4));
        assert!(err.reason().contains("safetensors EP"));
    }
    #[test]
    fn mesh_for_single_propagates_constructor_result() {
        let mesh = mesh_for(req(1, 1, 1)).expect("single-device mesh construction must succeed");
        assert_eq!(mesh.n_devices(), 1);
        assert_eq!(mesh.axes(), &[]);
    }

    #[test]
    fn mesh_for_rectangular_overflow_refuses_without_wrapping() {
        let error = mesh_for(req(usize::MAX, 2, 1))
            .expect_err("rectangular cardinality overflow must fail closed");
        assert_eq!(error, hipfire_hardware::MeshError::CardinalityOverflow);
    }
    #[test]
    fn resolver_refuses_composed_overflow_before_mesh_construction() {
        let err = resolve(
            SourceKind::Hfq,
            ModelVariant::Qwen35Dense,
            req(usize::MAX, 2, 1),
        )
        .unwrap_err();
        assert!(matches!(
            &err,
            AdmissionError::Composition {
                owner: "CAP-001",
                requested,
                ..
            } if requested.pp == usize::MAX && requested.tp == 2
        ));
        assert_ne!(err.code(), "TOPO-001");
    }
}
