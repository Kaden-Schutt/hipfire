// SPDX-License-Identifier: MIT OR Apache-2.0
use rdna_compute::DType;

// ── Pipeline composition ──────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum PipelineOp {
    RotateFwht,
    AwqDivide,
    Gemv,
    GemvResidual,
    SiluMul,
    SiluMulRotate,
    ResidualAdd,
    CopyD2D,
    GivensRotate,
}

// ── Variant enums ─────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum GemvVariant {
    Plain,
    Prerotated,
    WithResidual,
    WithSwiGLUResidual,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum FusedQkvVariant {
    Qkv,
    Qkvza,
    GateUp,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum AttentionVariant {
    Decode,
    Prefill,
    FlashDecode,
    FlashPrefill,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum MoeVariant {
    IndexedGateUp,
    IndexedDown,
    GroupedGemm,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum RotationVariant {
    Plain,
    PlainG128,
    Givens,
    WithRmsnorm,
    WithSwiGLU,
}

/// Sign-domain / scratch axis of rotation. Orthogonal to RotationVariant
/// (the fusion axis). Derivable purely from dtype.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum RotationPlan {
    None,
    FwhtG256,
    FwhtG128,
    Mq8Internal,
    Givens,
}

/// Sign-domain plan for a dtype. `None` <=> no activation rotation required.
pub fn dtype_rotation_plan(dtype: DType) -> RotationPlan {
    use DType::*;
    match dtype {
        MQ4G256 | MQ3G256 | MQ2G256 | MQ6G256
        | MQ2G256Lloyd | MQ3G256Lloyd | MQ4G256Lloyd
        | MFP4G32 => RotationPlan::FwhtG256,
        MQ4G128 => RotationPlan::FwhtG128,
        MQ8G256 => RotationPlan::Mq8Internal,
        ParoQ4G128 => RotationPlan::Givens,
        _ => RotationPlan::None,
    }
}

// ── Flat kernel key enum ──────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum KernelKey {
    // GEMV plain
    GemvF32,
    GemvF16,
    GemvQ8_0,
    GemvQ4K,
    GemvQ6K,
    GemvHfq4G256,
    GemvHfq4G128,
    GemvHfq3G256,
    GemvHfq3G128,
    GemvHfq2G256,
    GemvHfq2G128,
    GemvHfq6G256,
    GemvMq4G256,
    GemvMq4G128,
    GemvMq3G256,
    GemvMq2G256,
    GemvMq6G256,
    GemvMq8G256,
    GemvMq2G256Lloyd,
    GemvMq3G256Lloyd,
    GemvMq4G256Lloyd,
    GemvMfp4G32,
    GemvMfp4G32Fused,
    GemvHfp4G32,
    GemvParoQ4G128,
    GemvQ4F16G64,
    GemvQ4F16G32,
    // GEMV prerotated
    GemvMq4G256Prerotated,
    GemvMq3G256Prerotated,
    GemvMq2G256Prerotated,
    GemvMq6G256Prerotated,
    GemvMq8G256Prerotated,
    GemvMq2G256LloydPrerotated,
    GemvMq3G256LloydPrerotated,
    GemvMq4G256LloydPrerotated,
    GemvMfp4G32Prerotated,
    // GEMV residual
    GemvHfq4G256Residual,
    GemvHfq3G256Residual,
    GemvHfq6G256Residual,
    GemvMq4G256Residual,
    GemvMq3G256Residual,
    GemvMq6G256Residual,
    GemvMq3G256LloydResidual,
    GemvMq4G256LloydResidual,
    // GEMV SwiGLU + residual
    GemvHfq4G256SwiGLUResidual,
    GemvHfq3G256SwiGLUResidual,
    GemvHfq6G256SwiGLUResidual,
    GemvMq4G256SwiGLUResidual,
    GemvMq3G256SwiGLUResidual,
    GemvMq6G256SwiGLUResidual,
    GemvMq3G256LloydSwiGLUResidual,
    GemvMq4G256LloydSwiGLUResidual,
    // GEMM
    GemmHfq4G256,
    GemmHfq4G128,
    GemmQ8_0BatchedChunked,
    GemmQ8_0Wmma,
    GemmQ8_0Wmma4W,
    GemmHfq4G256Wmma,
    GemmF16XF16Wmma,
    GemmF32RegisterTiled,
    // Fused QKV
    FusedQkvHfq4G256,
    FusedQkvMq3G256Lloyd,
    FusedQkvMq4G256Lloyd,
    FusedQkvHfq6G256,
    FusedQkvParo4G128T,
    FusedQkvQ4K,
    // Fused QKVZA (linear attention)
    FusedQkvzaHfq4G256,
    FusedQkvzaMq3G256Lloyd,
    FusedQkvzaMq4G256Lloyd,
    FusedQkvzaHfq6G256,
    FusedQkvzaParo4G128T,
    // Fused Gate+Up
    FusedGateUpHfq4G256,
    FusedGateUpMq3G256Lloyd,
    FusedGateUpMq4G256Lloyd,
    FusedGateUpHfq6G256,
    FusedGateUpParo4G128T,
    FusedGateUpQ4K,
    // Rotation
    RotateMq,
    RotateMqG128,
    RotateMqAwq,
    RotateMqBatched,
    RotateMqAwqBatched,
    RmsnormRotateMq,
    RmsnormRotateMqAwq,
    RmsnormRotateMqBatched,
    RmsnormRotateMqAwqBatched,
    SiluMulRotateMq,
    SiluMulRotateMqAwq,
    RmsnormF32,
    // MoE
    MoeIndexedGateUpLloyd,
    MoeIndexedDownLloyd,
    MoeGroupedGemm,
    MoeGroupedI8,
    // Attention
    AttnFlashAsym4,
    AttnFlashAsym4Fwht,
    AttnFlashAsym3,
    AttnFlashAsym3Fwht,
    AttnFlashAsym2,
    AttnFlashAsym2Fwht,
    AttnFlashQ8_0,
    AttnGqaFused,
    AttnF32,
    // KV Cache Write
    KvWriteAsym4,
    KvWriteAsym4Fwht,
    KvWriteAsym3,
    KvWriteAsym3Fwht,
    KvWriteAsym2,
    KvWriteAsym2Fwht,
    KvWriteQ8_0,
    KvWriteF32,
}

// ── Shape context for predicate evaluation ───────────

/// Runtime tensor shape passed to `KernelRegistry::resolve` so that
/// `ShapePredicate` gates can evaluate against live dimensions.
///
/// Fields that are not relevant for a given call site can be left at 0
/// (they will only be checked if a registered `KernelVariant` carries a
/// `ShapePredicate` that references that field).  Pass `None` to
/// `resolve()` instead to skip all shape gating entirely.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShapeInfo {
    /// Token-batch size (number of rows being processed in parallel).
    pub batch_size: usize,
    /// Attention head dimension in elements.
    pub head_dim: usize,
    /// Output rows (M dimension of the weight matrix).
    pub m: usize,
}

// ── Arch gating ──────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum ArchPredicate {
    Always,
    HasWmmaW32,
    HasWmmaW32Gfx12,
    HasDp4a,
    HasSdot4,
    HasMmq,
    HasCdna3LdsGemv,
}

#[derive(Clone, Debug)]
pub enum ShapePredicate {
    BatchGt(usize),
    HeadDimEq(usize),
    MLt(usize),
}

// ── Registry entry ───────────────────────────────────

pub struct KernelVariant {
    pub key: KernelKey,
    pub arch_required: ArchPredicate,
    pub shape_gate: Option<ShapePredicate>,
    pub steps: &'static [PipelineOp],
    pub has_awq: bool,
}

// ── Error ────────────────────────────────────────────

#[derive(Debug)]
pub enum DispatchError {
    UnsupportedVariant {
        family: &'static str,
        variant: &'static str,
        arch: &'static str,
        quant: &'static str,
    },
    MissingImpl {
        key: KernelKey,
    },
    NotFound {
        key: KernelKey,
    },
    EmptyEntry {
        key: KernelKey,
    },
    Hip(String),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVariant { family, variant, arch, quant } => {
                write!(f, "unsupported {family}.{variant} for {arch}/{quant}")
            }
            Self::MissingImpl { key } => write!(f, "no implementation for {key:?}"),
            Self::NotFound { key } => write!(f, "kernel not registered: {key:?}"),
            Self::EmptyEntry { key } => write!(f, "kernel registry entry empty: {key:?}"),
            Self::Hip(msg) => write!(f, "HIP error: {msg}"),
        }
    }
}

impl std::error::Error for DispatchError {}

#[cfg(feature = "from-hip-error")]
impl From<DispatchError> for hip_bridge::HipError {
    fn from(e: DispatchError) -> Self {
        hip_bridge::HipError::new(0, &e.to_string())
    }
}



impl KernelKey {
    pub fn for_gemv(dtype: DType, variant: GemvVariant, _has_awq: bool) -> Result<Self, DispatchError> {
        use DType::*;
        use GemvVariant::*;
        match (dtype, variant) {
            (F32, Plain) => Ok(Self::GemvF32),
            (F16, Plain) => Ok(Self::GemvF16),
            (Q8_0, Plain) => Ok(Self::GemvQ8_0),
            (Q4K, Plain) => Ok(Self::GemvQ4K),
            (Q6K, Plain) => Ok(Self::GemvQ6K),
            (HFQ4G256, Plain) => Ok(Self::GemvHfq4G256),
            (HFQ4G128, Plain) => Ok(Self::GemvHfq4G128),
            (HFQ3G256, Plain) => Ok(Self::GemvHfq3G256),
            (HFQ3G128, Plain) => Ok(Self::GemvHfq3G128),
            (HFQ2G256, Plain) => Ok(Self::GemvHfq2G256),
            (HFQ2G128, Plain) => Ok(Self::GemvHfq2G128),
            (HFQ6G256, Plain) => Ok(Self::GemvHfq6G256),
            (MQ4G256, Plain) => Ok(Self::GemvMq4G256),
            (MQ4G128, Plain) => Ok(Self::GemvMq4G128),
            (MQ3G256, Plain) => Ok(Self::GemvMq3G256),
            (MQ2G256, Plain) => Ok(Self::GemvMq2G256),
            (MQ6G256, Plain) => Ok(Self::GemvMq6G256),
            (MQ8G256, Plain) => Ok(Self::GemvMq8G256),
            (MQ2G256Lloyd, Plain) => Ok(Self::GemvMq2G256Lloyd),
            (MQ3G256Lloyd, Plain) => Ok(Self::GemvMq3G256Lloyd),
            (MQ4G256Lloyd, Plain) => Ok(Self::GemvMq4G256Lloyd),
            (MFP4G32, Plain) => Ok(Self::GemvMfp4G32),
            (HFP4G32, Plain) => Ok(Self::GemvHfp4G32),
            (ParoQ4G128, Plain) => Ok(Self::GemvParoQ4G128),
            (Q4F16G64, Plain) => Ok(Self::GemvQ4F16G64),
            (Q4F16G32, Plain) => Ok(Self::GemvQ4F16G32),
            _ => Err(DispatchError::UnsupportedVariant {
                family: "gemv", variant: "unknown",
                arch: "", quant: "",
            }),
        }
    }

    pub fn for_gemv_prerotated(dtype: DType) -> Result<Self, DispatchError> {
        use DType::*;
        match dtype {
            MQ4G256 => Ok(Self::GemvMq4G256Prerotated),
            MQ3G256 => Ok(Self::GemvMq3G256Prerotated),
            MQ2G256 => Ok(Self::GemvMq2G256Prerotated),
            MQ6G256 => Ok(Self::GemvMq6G256Prerotated),
            MQ8G256 => Ok(Self::GemvMq8G256Prerotated),
            MQ2G256Lloyd => Ok(Self::GemvMq2G256LloydPrerotated),
            MQ3G256Lloyd => Ok(Self::GemvMq3G256LloydPrerotated),
            MQ4G256Lloyd => Ok(Self::GemvMq4G256LloydPrerotated),
            MFP4G32 => Ok(Self::GemvMfp4G32Prerotated),
            _ => Err(DispatchError::UnsupportedVariant {
                family: "gemv", variant: "prerotated",
                arch: "", quant: "",
            }),
        }
    }

    pub fn for_gemv_residual(dtype: DType) -> Result<Self, DispatchError> {
        use DType::*;
        match dtype {
            HFQ4G256 => Ok(Self::GemvHfq4G256Residual),
            HFQ3G256 => Ok(Self::GemvHfq3G256Residual),
            HFQ6G256 => Ok(Self::GemvHfq6G256Residual),
            MQ4G256 => Ok(Self::GemvMq4G256Residual),
            MQ3G256 => Ok(Self::GemvMq3G256Residual),
            MQ6G256 => Ok(Self::GemvMq6G256Residual),
            MQ3G256Lloyd => Ok(Self::GemvMq3G256LloydResidual),
            MQ4G256Lloyd => Ok(Self::GemvMq4G256LloydResidual),
            _ => Err(DispatchError::UnsupportedVariant {
                family: "gemv", variant: "residual",
                arch: "", quant: "",
            }),
        }
    }

    pub fn for_gemv_swiglu_residual(dtype: DType) -> Result<Self, DispatchError> {
        use DType::*;
        match dtype {
            HFQ4G256 => Ok(Self::GemvHfq4G256SwiGLUResidual),
            HFQ3G256 => Ok(Self::GemvHfq3G256SwiGLUResidual),
            HFQ6G256 => Ok(Self::GemvHfq6G256SwiGLUResidual),
            MQ4G256 => Ok(Self::GemvMq4G256SwiGLUResidual),
            MQ3G256 => Ok(Self::GemvMq3G256SwiGLUResidual),
            MQ6G256 => Ok(Self::GemvMq6G256SwiGLUResidual),
            MQ3G256Lloyd => Ok(Self::GemvMq3G256LloydSwiGLUResidual),
            MQ4G256Lloyd => Ok(Self::GemvMq4G256LloydSwiGLUResidual),
            _ => Err(DispatchError::UnsupportedVariant {
                family: "gemv", variant: "swiglu_residual",
                arch: "", quant: "",
            }),
        }
    }

    /// Architecture predicate required for a given DType's GEMV kernels.
    pub fn dtype_arch_predicate(dtype: DType) -> ArchPredicate {
        use DType::*;
        match dtype {
            F32 | F16 | Q8_0 | Q4K | Q6K | Q4F16G64 | Q4F16G32 => ArchPredicate::Always,
            HFQ4G256 | HFQ4G128 | HFQ2G256 | HFQ2G128
            | MQ4G256 | MQ4G128 | MQ2G256 | MQ8G256
            | HFP4G32 | MFP4G32
            | ParoQ4G128 => ArchPredicate::HasDp4a,
            HFQ3G256 | HFQ3G128 => ArchPredicate::HasSdot4,
            MQ3G256 => ArchPredicate::HasWmmaW32,
            MQ6G256 | HFQ6G256 => ArchPredicate::HasMmq,
            MQ2G256Lloyd | MQ3G256Lloyd | MQ4G256Lloyd => ArchPredicate::HasWmmaW32,
            Q8HFQ | Raw => ArchPredicate::Always,
        }
    }

    /// Pipeline steps required for a given (DType, GemvVariant) pair.
    pub fn gemv_steps(dtype: DType, variant: GemvVariant) -> &'static [PipelineOp] {
        use DType::*;
        use GemvVariant::*;
        match variant {
            Plain => {
                let steps: &[PipelineOp] = match dtype {
                    MQ4G256 | MQ4G128 | MQ3G256 | MQ2G256 | MQ6G256 | MQ8G256
                    | MQ2G256Lloyd | MQ3G256Lloyd | MQ4G256Lloyd
                    | MFP4G32 | ParoQ4G128 => {
                        &[PipelineOp::RotateFwht, PipelineOp::Gemv]
                    }
                    _ => &[PipelineOp::Gemv],
                };
                steps
            }
            Prerotated => {
                &[PipelineOp::Gemv]
            }
            WithResidual => {
                let steps: &[PipelineOp] = match dtype {
                    MQ4G256 | MQ3G256 | MQ6G256 | MQ3G256Lloyd | MQ4G256Lloyd => {
                        &[PipelineOp::RotateFwht, PipelineOp::Gemv, PipelineOp::ResidualAdd]
                    }
                    _ => &[PipelineOp::Gemv, PipelineOp::ResidualAdd],
                };
                steps
            }
            WithSwiGLUResidual => {
                let steps: &[PipelineOp] = match dtype {
                    MQ4G256 | MQ3G256 | MQ6G256 | MQ3G256Lloyd | MQ4G256Lloyd => {
                        &[PipelineOp::SiluMulRotate, PipelineOp::GemvResidual]
                    }
                    _ => &[PipelineOp::SiluMul, PipelineOp::Gemv, PipelineOp::ResidualAdd],
                };
                steps
            }
        }
    }
}

/// Whether a DType requires FWHT-rotated activations before GEMV.
/// Replaces per-model `needs_mq_rotation` / `weight_needs_fwht` helpers.
pub fn dtype_needs_fwht(dtype: DType) -> bool {
    use DType::*;
    matches!(
        dtype,
        MQ4G256 | MQ4G128 | MQ3G256 | MQ2G256 | MQ6G256 | MQ8G256
            | MQ2G256Lloyd | MQ3G256Lloyd | MQ4G256Lloyd
            | MFP4G32 | ParoQ4G128
    )
}
