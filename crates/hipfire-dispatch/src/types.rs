// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
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
    // MoE decode ops (Phase 1). TopKRenorm / MoeCombine fused impls are
    // k=8-only today; the variant names are k-agnostic so a future k=6
    // kernel family can reuse them.
    MoeGateSideProj,
    Softmax,
    TopKRenorm,
    SharedExpertDown,
    IndexedGateUp,
    IndexedDownExpanded,
    MoeCombine,
    /// Fused rmsnorm + optional rotation (MQ-weight producer step).
    /// rotation=FwhtG256 → rmsnorm + FWHT. rotation=None → rmsnorm only.
    RmsnormAutomatic,
    /// Paired KV-write + flash-attention (Phase 0.3). Not fusible —
    /// the two ops are inherently coupled via KvTierPlan.
    Attend,
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
    QkvParo,
    QkvzaParo,
    GateUpParo,
}

/// Implementation discriminator for attention tile variants. Attention-specific;
/// other families use `TileImpl::None` (the `#[default]`). Future families needing
/// multi-variant dispatch should consider a generic type parameter or opaque index
/// rather than extending this enum.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Default)]
pub enum TileImpl {
    #[default]
    None,
    // WMMA-FA (quantized causal prefill — asym4+Q8-V only)
    Asym4WmmaTile,
    Asym4WmmaTileGfx12,
    // Vision/dflash F16-K/V rungs
    DflashV5,
    DflashV5Gfx12,
    DflashN128,
    // Vision/dflash F32-K/V rungs
    DflashM32,
    DflashWmmaF32,
    // Causal (F16-K/V rungs)
    DflashV3Causal,
    DflashV3CausalGfx12,
    // Scalar floors — separate for causal vs non-causal
    DflashScalar,    // non-causal → gpu.attention_dflash_f32
    CausalScalar,     // causal → gpu.attention_causal_batched
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

/// GEMV variant to run AFTER the activation has been rotated.
/// ParoQ4G128 uses the Plain HFQ4G128 kernel post-Givens; the MQ family
/// uses Prerotated kernels; non-rotated dtypes are Plain.
pub fn dtype_post_rotation_variant(dtype: DType) -> GemvVariant {
    use DType::*;
    match dtype {
        ParoQ4G128 => GemvVariant::Plain,
        MQ4G256 | MQ3G256 | MQ2G256 | MQ6G256 | MQ8G256
        | MQ2G256Lloyd | MQ3G256Lloyd | MQ4G256Lloyd
        | MFP4G32 | MQ4G128 => GemvVariant::Prerotated,
        _ => GemvVariant::Plain,
    }
}

/// Fused-projection family for a kernel key: which of the QKV / QKVZA / Gate+Up
/// kernel groups a `KernelKey` belongs to. `None` for keys that are not fused
/// projections. The Paro4G128T keys carry the family in their name (Qkv/Qkvza/
/// GateUp) but the *Paro* discriminator variants (QkvParo/QkvzaParo/GateUpParo)
/// describe the rotation axis, not the projection shape — for error diagnostics
/// we report the projection family (Qkv/Qkvza/GateUp), which is what determines
/// the arity (3-way / 4-way / 2-way).
pub fn fused_qkv_variant_for_key(key: KernelKey) -> Option<FusedQkvVariant> {
    use KernelKey::*;
    match key {
        // 3-way Fused QKV (incl. Q4K, Q8_0/HFQ3/HFP4 prefill, and the Paro 4G128T QKV synthesis)
        FusedQkvHfq4G256 | FusedQkvMq3G256Lloyd | FusedQkvMq4G256Lloyd
        | FusedQkvHfq6G256 | FusedQkvQ4K | FusedQkvQ8_0 | FusedQkvHfq3G256
        | FusedQkvHfp4G32
        | FusedQkvParo4G128T => {
            Some(FusedQkvVariant::Qkv)
        }
        // 4-way Fused QKVZA (DeltaNet linear attention, incl. Q8_0/HFQ3/HFP4 prefill and Paro 4G128T)
        FusedQkvzaHfq4G256 | FusedQkvzaMq3G256Lloyd | FusedQkvzaMq4G256Lloyd
        | FusedQkvzaHfq6G256 | FusedQkvzaQ8_0 | FusedQkvzaHfq3G256
        | FusedQkvzaHfp4G32
        | FusedQkvzaParo4G128T => Some(FusedQkvVariant::Qkvza),
        // 2-way Fused Gate+Up (FFN, incl. Q8_0, HFQ3, HFP4 and Paro 4G128T)
        FusedGateUpHfq4G256 | FusedGateUpMq3G256Lloyd | FusedGateUpMq4G256Lloyd
        | FusedGateUpHfq6G256 | FusedGateUpQ4K | FusedGateUpQ8_0
        | FusedGateUpHfq3G256 | FusedGateUpHfp4G32
        | FusedGateUpParo4G128T => Some(FusedQkvVariant::GateUp),
        _ => None,
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
    GemvQ8HFQ,
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
    // GEMM — plain-family catalog (#397 Ship 5.1). These all take the
    // canonical plain-GEMM signature `(a, x, y, m, k, batch_size)` and are
    // dispatchable through GemmFamily::run. Methods whose signatures carry
    // extra params (bias / prequantized x ptr) or that belong to the fused /
    // residual / moe / lmhead families are NOT here — they are owned by their
    // own families (FusedQkv*, Moe*, etc.). See gemm_table.rs.
    GemmF16,
    GemmF16Tiled,
    GemmF16WmmaMb4,
    GemmF16WmmaMb8,
    GemmF32Batched,
    GemmQ8_0WmmaX64,
    GemmQ8_0ResidualWmma,
    GemmQ8_0ResidualWmmaGfx12,
    GemmHfq4G256Dp4a,
    GemmHfq4G256MmqSet,
    // GEMM — residual-fused catalog (#397 Ship 5.2 slice "5.2 FINAL").
    // These take the residual signature `(a, x, y, m, k, batch_size)` and
    // compute `y += a·x` (the add is internal to each kernel — the caller
    // passes the residual stream as `y` and the kernel never reuses it as
    // GEMV scratch). Dispatched through GemmFamily::run_key, NOT resolve().
    GemmHfq6G256Residual,
    GemmHfq4G256Residual,
    GemmHfq3G256Residual,
    GemmHfp4G32Residual,
    GemmMq3G256LloydResidual,
    // GEMM — spec-decode (DFlash) batched lm_head catalog (#397 Ship 5.3).
    // These take the canonical signature `(a, x, y, m, k, batch_size)` and
    // dispatch the batched verify/draft lm_head GEMM. Each `gpu.gemm_*` method
    // auto-routes its own arch ladder internally (WMMA for batch>1 on gfx11/12,
    // dp4a on gfx906, fp16/scalar fallback otherwise) and runs on EVERY arch,
    // so all are registered `ArchPredicate::Always`. Dispatched through
    // GemmFamily::run_key against the explicit key (NOT resolve()), so the
    // method selected is byte-identical to the prior direct spec-decode call.
    GemmQ8_0Batched,
    GemmHfq4G256BatchedLmhead,
    GemmHfq3G256BatchedLmhead,
    GemmHfq6G256BatchedLmhead,
    // Fused QKV
    FusedQkvHfq4G256,
    FusedQkvMq3G256Lloyd,
    FusedQkvMq4G256Lloyd,
    FusedQkvHfq6G256,
    FusedQkvQ4K,
    // Fused QKV — prefill dtypes (#397 Ship 5.2 slice 3)
    FusedQkvQ8_0,
    FusedQkvHfq3G256,
    FusedQkvHfp4G32,
    // Fused QKVZA (linear attention)
    FusedQkvzaHfq4G256,
    FusedQkvzaMq3G256Lloyd,
    FusedQkvzaMq4G256Lloyd,
    FusedQkvzaHfq6G256,
    // Fused QKVZA — prefill dtypes (#397 Ship 5.2 slice 3)
    FusedQkvzaQ8_0,
    FusedQkvzaHfq3G256,
    FusedQkvzaHfp4G32,
    // Fused Gate+Up
    FusedGateUpHfq4G256,
    FusedGateUpMq3G256Lloyd,
    FusedGateUpMq4G256Lloyd,
    FusedGateUpHfq6G256,
    FusedGateUpQ4K,
    FusedGateUpHfq3G256,
    FusedGateUpHfp4G32,
    // Fused Paro (4G128T)
    FusedGateUpParo4G128T,
    FusedQkvzaParo4G128T,
    FusedQkvParo4G128T,
    FusedGateUpQ8_0,
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
    AttnQ8_0Kv,    // non-flash short-context Q8_0 decode (ship 3.1 B0)
    AttnGqaFused,
    // Llama legacy quant KV (decode only — no batched variants)
    AttnHfq4Kv,       // HFQ4-quantized KV cache attention
    AttnQ4Kv,         // Q4-quantized KV cache attention
    // F32 KV (decode only — no batched variant)
    AttnF32,
    // Attention — batched prefill / tree-verify (ship 3.2)
    AttnFlashAsym4BatchedMasked,
    AttnFlashAsym4FwhtBatchedMasked,
    AttnFlashAsym3BatchedMasked,
    AttnFlashAsym3FwhtBatchedMasked,
    AttnFlashAsym2Batched,       // no _masked — 2-bit tree-verify gap
    AttnFlashAsym2FwhtBatched,   // no _masked — 2-bit tree-verify gap
    AttnQ8_0KvBatchedMasked,     // P-1 no-LDS-cap tiled kernel
    // TODO(3.3): F32-batched key for models with F32 KV + batchable weights
    // Full attention (no KV cache — vision / dflash cross-attention)
    AttnFullF16,         // F16 K/V, non-causal
    AttnFullF32,         // F32 K/V, non-causal
    AttnFullF16Causal,   // F16 K/V, causal
    AttnFullF32Causal,   // F32 K/V, causal
    // KV Cache Write
    KvWriteAsym4,
    KvWriteAsym4Fwht,
    KvWriteAsym3,
    KvWriteAsym3Fwht,
    KvWriteAsym2,
    KvWriteAsym2Fwht,
    KvWriteQ8_0,
    KvWriteHfq4,    // HFQ4-quantized KV write (llama legacy)
    KvWriteQ4,      // Q4-quantized KV write (llama legacy)
    KvWriteF32,
    // KV Cache Write — batched prefill (ship 3.2)
    KvWriteAsym4Batched,
    KvWriteAsym4FwhtBatched,
    KvWriteAsym3Batched,
    KvWriteAsym3FwhtBatched,
    KvWriteAsym2Batched,
    KvWriteAsym2FwhtBatched,
    KvWriteQ8_0Batched,
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
    /// For attention families, this is the number of query rows (n_patches for vision,
    /// n tokens for prefill, 1 for decode).
    pub batch_size: usize,
    /// Attention head dimension in elements.
    pub head_dim: usize,
    /// Output rows / sequence length (M dimension). For GEMV families, output rows.
    /// For attention families, seq_len.
    pub m: usize,
    /// Whether tree-verify is active (shape-level flag for IsTree predicates).
    pub is_tree: bool,
}

// ── Arch gating ──────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum ArchPredicate {
    Always,
    HasWmma,
    /// `has_wmma_w32()` — gfx11-family wave32 WMMA (RDNA3 + RDNA3.5), EXCLUDING
    /// gfx12/RDNA4. For WMMA kernels with NO gfx12 source sibling (e.g.
    /// gemm_f16_wmma_mb4, gemm_q8_0_wmma_x64); RDNA4 must fall through to a
    /// non-WMMA entry. Restores the arch gate Phase 0.4 collapsed into HasWmma.
    HasWmmaW32,
    HasWmmaGfx12,
    /// `has_dot2_f32_f16()` — RDNA1.1+ (gfx1011, gfx1030+, gfx1100+, gfx1200+).
    /// Gates the RDNA dot2-F16 codepath used by HFQ3 sdot4 kernels.
    /// Historically named "HasDp4a" — renamed because AMD "dp4a" in the ISA
    /// means `v_dot4_i32_i8` (gfx906-only), while this checks `v_dot2_f32_f16`
    /// (RDNA1.1+). The two are unrelated ISA features.
    HasDot2F32F16,
    HasSdot4,
    HasMmq,
    HasCdna3LdsGemv,
    /// `gemv_dp4a_enabled()` — gfx906-only by default (env-overridable).
    /// Gates the gfx906 wave64 `v_dot4_i32_i8` (sdot4) fused kernels (HFQ6/MQ6).
    /// This IS AMD "dp4a" — `v_dot4_i32_i8` INT8 dot4 accumulate, gfx906/gfx908.
    HasDp4a,
}

#[derive(Clone, Debug)]
pub enum ShapePredicate {
    BatchGt(usize),
    BatchGe(usize),
    BatchEq(usize),
    HeadDimEq(usize),
    HeadDimLe(usize),
    HeadDimMultipleOf(usize),
    HeadDimIn(&'static [usize]),
    MLt(usize),
    IsTree(bool),
    And(&'static [ShapePredicate]),
}

// ── Registry entry ───────────────────────────────────

#[derive(Debug)]
pub struct KernelVariant {
    pub key: KernelKey,
    pub arch_required: ArchPredicate,
    pub shape_gate: Option<ShapePredicate>,
    pub steps: &'static [PipelineOp],
    pub has_awq: bool,
    pub tile: TileImpl,
}

impl Default for KernelVariant {
    fn default() -> Self {
        Self {
            key: KernelKey::GemmF32RegisterTiled, // placeholder — must be overridden
            arch_required: ArchPredicate::Always,
            shape_gate: None,
            steps: &[],
            has_awq: false,
            tile: TileImpl::None,
        }
    }
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
            (Q8HFQ, Plain) => Ok(Self::GemvQ8HFQ),
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
            // Q8/Paro have no separate "prerotated" kernel: Q8 is not FWHT-rotated
            // (prerotated input == raw input → gemv_q8_0), and Paro's Givens-rotated
            // input feeds the same gemv_hfq4g128 kernel as its Plain path. launch()
            // dispatches GemvQ8_0 → gpu.gemv_q8_0 and GemvParoQ4G128 → gpu.gemv_hfq4g128.
            Q8_0 => Ok(Self::GemvQ8_0),
            ParoQ4G128 => Ok(Self::GemvParoQ4G128),
            // Any other dtype: if it is rotation-free (RotationPlan::None), its
            // "prerotated" input IS its plain input (no rotation was applied), so the
            // plain GEMV kernel is correct — route to for_gemv(Plain), the exact kernel
            // the legacy weight_gemv_prerotated→run_auto fallback used (e.g. F16/F32/
            // Q4K/Q6K/HFQ3G256/HFQ6G256/HFQ2G256/HFP4G32). Rotation-needing dtypes not
            // enumerated above (e.g. MQ4G128 = FwhtG128) MUST NOT fall through — the
            // plain path would re-rotate already-rotated input — so they stay an Err.
            _ => {
                if dtype_rotation_plan(dtype) == RotationPlan::None {
                    Self::for_gemv(dtype, GemvVariant::Plain, false)
                } else {
                    Err(DispatchError::UnsupportedVariant {
                        family: "gemv", variant: "prerotated",
                        arch: "", quant: "",
                    })
                }
            }
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
            // HFQ4/MQ4/HFQ2/MQ2/MQ8/HFP4/MFP4/Paro: all use generic wave32/wave64
            // kernels with no ISA-specific intrinsics. The underlying GEMV
            // functions (gemv_hfq4g256_for_arch, gemv_hfp4g32_for_arch, etc.)
            // have arch-specific *tuning* variants but a generic fallback that
            // runs on every arch including gfx906 and gfx1010.
            //
            // Previously gated on HasDp4a (has_dot2_f32_f16 = RDNA1.1+), which
            // excluded gfx906 where these kernels work fine via the generic path
            // (gfx906 uses v_dot4_i32_i8/sdot4 internally, NOT dot2_f32_f16).
            // See issue #397: the HasDp4a name is a misnomer — it maps to
            // has_dot2_f32_f16 (RDNA1.1+) while AMD "dp4a" = v_dot4_i32_i8
            // is gfx906-only. The two are unrelated ISA features.
            HFQ4G256 | HFQ4G128 | HFQ2G256 | HFQ2G128
            | MQ4G256 | MQ4G128 | MQ2G256 | MQ8G256
            | HFP4G32 | MFP4G32
            | ParoQ4G128 => ArchPredicate::Always,
            HFQ3G256 | HFQ3G128 => ArchPredicate::HasSdot4,
            MQ3G256 => ArchPredicate::HasWmma,
            MQ6G256 | HFQ6G256 => ArchPredicate::HasMmq,
            MQ2G256Lloyd | MQ3G256Lloyd | MQ4G256Lloyd => ArchPredicate::HasWmma,
            Q8HFQ | Raw => ArchPredicate::Always,
        }
    }

    /// Pipeline steps required for a given (DType, GemvVariant) pair.
    pub fn gemv_steps(dtype: DType, variant: GemvVariant) -> &'static [PipelineOp] {
        use DType::*;
        use GemvVariant::*;
        match variant {
            Plain => {
                match dtype_rotation_plan(dtype) {
                    RotationPlan::Givens => &[PipelineOp::GivensRotate, PipelineOp::Gemv],
                    RotationPlan::None => &[PipelineOp::Gemv],
                    _ => &[PipelineOp::RotateFwht, PipelineOp::Gemv],
                }
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

/// Whether a DType requires activation rotation (FWHT or Givens) before GEMV.
/// Replaces per-model `needs_mq_rotation` / `weight_needs_fwht` helpers.
pub fn dtype_needs_rotation(dtype: DType) -> bool {
    use DType::*;
    matches!(
        dtype,
        MQ4G256 | MQ4G128 | MQ3G256 | MQ2G256 | MQ6G256 | MQ8G256
            | MQ2G256Lloyd | MQ3G256Lloyd | MQ4G256Lloyd
            | MFP4G32 | ParoQ4G128
    )
}
