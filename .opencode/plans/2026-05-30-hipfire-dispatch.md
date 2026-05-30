# hipfire-dispatch — Kernel Dispatch Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify ~30 scattered kernel dispatch trees across 5 model implementations into a single `hipfire-dispatch` crate with 6 kernel families, eliminating DType matching at the model level.

**Architecture:** Flat `KernelKey` enum + runtime selection table per family. Pipeline-based composition where each kernel record declares its steps (`[RotateFwht, Gemv]`), and a best-fit dispatcher selects the longest-match (fused) kernel. Rotation is automatic — the model never calls it directly. Graph capture is handled by a `GraphMode` flag on `DispatchCtx`.

**Tech Stack:** Rust, `hipfire-runtime`, `rdna-compute`, existing HIP kernel symbols. No proc macros, no new build deps.

**Decisions (from design exploration):**
- Q1: Gpu-level `mq_x_rot` persistent across model loads (128 KB isn't worth abstracting)
- Q2: GraphMode enum inside Dispatch (model never touches HIP graph APIs)
- Q3: Migration starts with Rotation family (pure function, lowest risk)
- Q4: Model-specific kernels (DeltaNet state, DeepSeek compressor) in `model_ext/` with own traits
- Q5: 6 per-family feature flags, not 30 per-family×per-model flags

---

## File Structure: `crates/hipfire-dispatch/`

```
crates/hipfire-dispatch/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── context.rs          # DispatchCtx — ArchCaps + FeatureFlags + GraphMode
│   ├── traits.rs           # KernelFamily trait, entry-point traits
│   ├── types.rs            # KernelKey enum, KernelVariant, KernelRecord, PipelineOp
│   ├── families/
│   │   ├── mod.rs
│   │   ├── gemv.rs         # GemvFamily
│   │   ├── gemm.rs         # GemmFamily
│   │   ├── fused_qkv.rs    # FusedQkvFamily
│   │   ├── attention.rs    # AttentionFamily
│   │   ├── moe.rs          # MoeFamily
│   │   └── rotation.rs     # RotationFamily
│   ├── tables/
│   │   ├── mod.rs          # registry builder, validation
│   │   ├── gemv_table.rs   # GEMV kernel entries
│   │   ├── gemm_table.rs   # GEMM kernel entries
│   │   ├── fused_qkv_table.rs
│   │   ├── attention_table.rs
│   │   ├── moe_table.rs
│   │   └── rotation_table.rs
│   ├── pipeline/
│   │   ├── mod.rs          # Pipeline derivation + best-fit dispatcher
│   │   ├── steps.rs        # Step execution functions
│   │   └── decompose.rs    # Fallback: break needed steps into separate launches
│   ├── resource/
│   │   ├── mod.rs          # ResourceManager
│   │   └── model_resources.rs  # ModelResources builder
│   └── model_ext/
│       ├── mod.rs
│       └── deepseek4.rs    # Deepseek4ModelExt (compressor, joint K=V, etc.)
```

Models also change — each model crate (`hipfire-arch-qwen35`, `hipfire-arch-deepseek4`, `hipfire-arch-llama`, `hipfire-arch-qwen2`) receives modifications to use Dispatch. Those changes live in the model crate, not in hipfire-dispatch.

---

## Task Structure

### Phase 0: Foundation Crate (1 PR, 3-5 days)

#### Task 0.1: Create crate skeleton and core types

**Files:**
- Create: `crates/hipfire-dispatch/Cargo.toml`
- Create: `crates/hipfire-dispatch/src/lib.rs`
- Create: `crates/hipfire-dispatch/src/context.rs`
- Create: `crates/hipfire-dispatch/src/traits.rs`
- Create: `crates/hipfire-dispatch/src/types.rs`
- Create: `crates/hipfire-dispatch/src/families/mod.rs`
- Create: `crates/hipfire-dispatch/src/tables/mod.rs`
- Create: `crates/hipfire-dispatch/src/pipeline/mod.rs`
- Create: `crates/hipfire-dispatch/src/resource/mod.rs`
- Create: `crates/hipfire-dispatch/src/model_ext/mod.rs`

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "hipfire-dispatch"
version.workspace = true
edition.workspace = true

[dependencies]
hipfire-types = { path = "../hipfire-types" }
rdna-compute = { path = "../rdna-compute" }
hipfire-arch-traits = { path = "../hipfire-arch-traits" }

[features]
default = []
new-rotation = []
new-gemv = []
new-gemm = []
new-fused-qkv = []
new-attention = []
new-moe = []
```

- [ ] **Step 2: Define core types in `types.rs`**

```rust
use hipfire_types::DType;
use rdna_compute::arch_caps::ArchCaps;
use rdna_compute::dispatch::GraphMode;

/// A concrete kernel function handle.
pub type KernelFn = fn(&Gpu, &KernelArgs) -> HipResult<()>;

/// Steps a kernel performs internally.
/// Used for pipeline derivation and best-fit matching.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum PipelineOp {
    RotateFwht,
    AwqDivide,
    Gemv,
    GemvResidual,
    SiluMul,
    SiluMulRotate,
    ResidualAdd,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum GemvVariant {
    Plain,
    Prerotated,
    WithResidual,
    WithSwiGLUResidual,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum FusedQkvVariant {
    Qkv,     // 3 projections: q, k, v
    Qkvza,   // 4 projections: q, k, v, z (linear attention)
    GateUp,  // 2 projections: gate, up (FFN)
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
    WithRmsnorm,
    WithSwiGLU,
}

/// Flat enum — one variant per concrete kernel symbol.
/// Adding a new quant format = add variants here + add rows in table.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum KernelKey {
    // GEMV
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
    GemvHfp4G32,
    GemvParo4G128,
    GemvParo4G128T,
    GemvParoQ4G128,
    GemvQ4F16G64,
    GemvQ4F16G32,
    // Prerotated
    GemvMq4G256Prerotated,
    GemvMq3G256Prerotated,
    GemvMq2G256Prerotated,
    GemvMq6G256Prerotated,
    GemvMq8G256Prerotated,
    GemvMq2G256Lloyd,
    GemvMq3G256Lloyd,
    GemvMq4G256Lloyd,
    GemvMfp4G32Prerotated,
    // Residual
    GemvHfq4G256Residual,
    GemvHfq3G256Residual,
    GemvHfq6G256Residual,
    GemvMq4G256Residual,
    GemvMq3G256Residual,
    GemvMq6G256Residual,
    GemvMq3G256LloydResidual,
    GemvMq4G256LloydResidual,
    GemvParo4G128Residual,
    GemvParo4G128TResidual,
    // SwiGLU + Residual
    GemvHfq4G256SwiGLUResidual,
    GemvHfq3G256SwiGLUResidual,
    GemvHfq6G256SwiGLUResidual,
    GemvMq4G256SwiGLUResidual,
    GemvMq3G256SwiGLUResidual,
    GemvMq6G256SwiGLUResidual,
    GemvMq3G256LloydSwiGLUResidual,
    GemvMq4G256LloydSwiGLUResidual,
    GemvParo4G128SwiGLUResidual,
    GemvParo4G128TSwiGLUResidual,
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
    // Fused QKVZA (LA4)
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

pub struct KernelVariant {
    pub key: KernelKey,
    pub fn_ptr: KernelFn,
    pub arch_required: ArchPredicate,
    pub shape_gate: Option<ShapePredicate>,
    pub steps: &'static [PipelineOp],
    pub has_awq: bool,
}

pub enum ArchPredicate {
    Always,
    HasWmmaW32,
    HasWmmaW32Gfx12,
    HasDp4a,
    HasSdot4,
    HasMmq,
    HasCdna3LdsGemv,
}

pub enum ShapePredicate {
    BatchGt(usize),
    HeadDimEq(usize),
    MLt(usize),
}
```

- [ ] **Step 3: Define entry-point traits in `traits.rs`**

```rust
use crate::{types::*, context::DispatchCtx};

pub trait GemvFamily: Send + Sync {
    fn resolve(&self, dtype: DType, variant: GemvVariant, has_awq: bool)
        -> Result<&KernelVariant, DispatchError>;

    fn run(&self, ctx: &DispatchCtx, gpu: &Gpu, params: GemvParams) -> Result<()>;
}

pub trait RotationFamily: Send + Sync {
    fn resolve(&self, dtype: DType, variant: RotationVariant)
        -> Result<&KernelVariant, DispatchError>;

    fn run(&self, ctx: &DispatchCtx, gpu: &Gpu, params: RotationParams) -> Result<()>;
}

// Same pattern for GemmFamily, FusedQkvFamily, AttentionFamily, MoeFamily.
```

- [ ] **Step 4: Define DispatchCtx in `context.rs`**

```rust
use std::sync::Arc;
use rdna_compute::arch_caps::ArchCaps;
use rdna_compute::dispatch::{GraphMode, FeatureFlags};
use crate::resource::ResourceManager;

pub struct DispatchCtx {
    pub arch: ArchCaps,
    pub flags: Arc<FeatureFlags>,
    pub graph: GraphMode,
    pub resources: ResourceManager,
}

impl DispatchCtx {
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            arch: ArchCaps::detect(gpu),
            flags: Arc::new(FeatureFlags::from_env()),
            graph: GraphMode::Off,
            resources: ResourceManager::new(gpu),
        }
    }
}
```

- [ ] **Step 5: Verify compilation**

Run: `cargo build -p hipfire-dispatch 2>&1`
Expected: Compiles. No warnings.

- [ ] **Step 6: Commit**

```
git add crates/hipfire-dispatch/
git commit -m "feat: scaffold hipfire-dispatch crate with core types and traits

Phase 0 foundation. Defines KernelKey (flat enum, ~150 variants),
KernelVariant with step lists, entry-point traits per family,
and DispatchCtx (caps + flags + graph mode).

Assisted-by: OpenCode:deepseek/deepseek-v4-flash
```

---

### Task 0.2: Implement dispatch table infrastructure

**Files:**
- Modify: `crates/hipfire-dispatch/src/tables/mod.rs`

- [ ] **Step 1: Implement registry type and resolution logic**

```rust
use std::collections::HashMap;
use crate::types::*;
use crate::context::DispatchCtx;

pub struct KernelRegistry {
    table: HashMap<KernelKey, Vec<KernelVariant>>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        Self { table: HashMap::new() }
    }

    pub fn register(&mut self, entry: KernelVariant) {
        self.table.entry(entry.key).or_default().push(entry);
    }

    pub fn resolve(
        &self,
        key: KernelKey,
        ctx: &DispatchCtx,
    ) -> Result<&KernelVariant, DispatchError> {
        let variants = self.table.get(&key)
            .ok_or(DispatchError::UnsupportedKernel { key })?;

        for variant in variants {
            if !variant.arch_required.eval(&ctx.arch) {
                continue;
            }
            if let Some(ref gate) = variant.shape_gate {
                if !gate.eval(ctx) {
                    continue;
                }
            }
            return Ok(variant);
        }

        Err(DispatchError::NoMatchingVariant { key, arch: ctx.arch.clone() })
    }

    /// Validate all entries at startup.
    pub fn validate(&self) -> Result<(), DispatchError> {
        for (key, variants) in &self.table {
            if variants.is_empty() {
                return Err(DispatchError::EmptyEntry { key: *key });
            }
        }
        Ok(())
    }
}
```

- [ ] **Step 2: Implement ArchPredicate evaluation**

```rust
impl ArchPredicate {
    pub fn eval(&self, arch: &ArchCaps) -> bool {
        match self {
            Self::Always => true,
            Self::HasWmmaW32 => arch.has_wmma_w32(),
            Self::HasWmmaW32Gfx12 => arch.has_wmma_w32_gfx12(),
            Self::HasDp4a => arch.has_dot2_f32_f16(),
            Self::HasSdot4 => arch.has_hfq3_sdot4(),
            Self::HasMmq => arch.has_mmq(),
            Self::HasCdna3LdsGemv => arch.has_cdna3_lds_gemv(),
        }
    }
}
```

- [ ] **Step 3: Verify compilation**

Run: `cargo build -p hipfire-dispatch 2>&1`
Expected: Compiles.

- [ ] **Step 4: Commit**

```
git add crates/hipfire-dispatch/src/tables/mod.rs
git commit -m "feat: dispatch table registry with validation and arch gating

KernelRegistry supports register() + resolve() with ArchPredicate
and ShapePredicate filtering. validate() panics at init for missing
entries — catches unsupported (model, quant, arch) combos at load
time instead of first decode.

Assisted-by: OpenCode:deepseek/deepseek-v4-flash
```

---

### Phase 1: Rotation Family (2-3 PRs, ~1 week)

#### Task 1.1: Extract rotation dispatch tables

**Files:**
- Create: `crates/hipfire-dispatch/src/families/rotation.rs`
- Create: `crates/hipfire-dispatch/src/tables/rotation_table.rs`
- Modify: `crates/hipfire-dispatch/src/pipeline/mod.rs`

Extract the following kernel dispatch logic from `llama.rs` (lines 869-960, 971-1000, 1013-1101) and `qwen35.rs` (rotation prelude calls):
- `fused_rmsnorm_rotate_for_mq`
- `fused_rmsnorm_rotate_for_paro`
- `rotate_x_for_mq`
- `rotate_x_mq_for`
- `fused_silu_mul_rotate_mq_for`
- `fused_rmsnorm_rotate_mq_batched_for`

Into a single `RotationFamily` whose table maps `(DType, RotationVariant, has_awq)` → kernel.

**Verification:** Integration test that calls old and new rotation paths on random (arch, quant, shape) inputs and compares byte output.

- [ ] **Step 1: Create `rotation.rs` with `RotationFamily` impl**

```rust
use crate::{types::*, tables::*, context::DispatchCtx};
use rdna_compute::dispatch::Gpu;

pub struct RotationFamily {
    registry: KernelRegistry,
}

impl RotationFamily {
    pub fn new(gpu: &Gpu) -> Result<Self, DispatchError> {
        let mut registry = KernelRegistry::new();
        // Build table at init time
        rotation_table::populate(&mut registry, gpu)?;
        registry.validate()?;
        Ok(Self { registry })
    }

    pub fn resolve(
        &self,
        dtype: DType,
        variant: RotationVariant,
        has_awq: bool,
    ) -> Result<&KernelVariant, DispatchError> {
        let key = self.key_for(dtype, variant, has_awq)?;
        self.registry.resolve(key, ctx)
    }

    pub fn run(
        &self,
        ctx: &DispatchCtx,
        gpu: &Gpu,
        params: RotationParams,
    ) -> Result<(), DispatchError> {
        let variant = self.resolve(params.x.dtype(), params.target, params.awq_scale.is_some())?;
        // Execute pre-steps, launch kernel, execute post-steps
        // via pipeline runner
        run_pipeline(ctx, gpu, variant, params.into())?;
        Ok(())
    }

    fn key_for(&self, dtype: DType, variant: RotationVariant, has_awq: bool) -> Result<KernelKey, DispatchError> {
        // Map (dtype, variant, awq) → one of:
        // RotateMq, RotateMqAwq, RmsnormRotateMq, RmsnormRotateMqAwq,
        // SiluMulRotateMq, SiluMulRotateMqAwq, RmsnormF32, etc.
        todo!() // filled from table
    }
}
```

- [ ] **Step 2: Write integration test — old rotation vs new rotation byte-equality**

- [ ] **Step 3: Verify compilation and test pass**

- [ ] **Step 4: Commit**

---

### Phase 2: GEMV Family (4-6 PRs, ~3 weeks — HIGHEST RISK)

#### Task 2.1: Extract GEMV dispatch tables (all 26 quant formats × 4 variants)

**Files:**
- Create: `crates/hipfire-dispatch/src/families/gemv.rs`
- Create: `crates/hipfire-dispatch/src/tables/gemv_table.rs`

Port the `weight_gemv`, `weight_gemv_prerotated`, `weight_gemv_residual`, `weight_gemv_swiglu_residual` dispatch from `llama.rs` (lines 614-1468) into a single `GemvFamily` with a table keyed by `(DType, GemvVariant, has_awq)`.

The `best_fit` dispatcher in `pipeline/mod.rs` automatically selects fused variants when the requested variant matches multiple steps in a single kernel.

**Verification:** Per-model feature flag (`feature = "new-gemv"`). Coherence gate passes on both sides. Integration test: 500 random (arch, quant, shape) dispatch calls, old output == new output byte-for-byte.

#### Task 2.2: Port llama.rs to new GEMV dispatch

- [ ] Wire `feature = "new-gemv"` in hipfire-runtime
- [ ] Replace `weight_gemv` calls in `forward_scratch_layers` with `dispatch.gemv.run()`
- [ ] Coherence gate verification

#### Task 2.3: Port qwen35.rs to new GEMV dispatch

Same pattern — replace inline fused QKVZA/Gate+Up dispatch with `dispatch.fused_qkv.run()` and `dispatch.gemv.run()`.

#### Task 2.4: Port deepseek4.rs to new GEMV dispatch

Replace `gemv_auto` and `gemv_auto_batched_wmma` with `dispatch.gemv.run()`.

#### Task 2.5: Port qwen2.rs to new GEMV dispatch

---

### Phase 3: GEMM + FusedQKV (3-4 PRs, ~2 weeks)

#### Task 3.1: Extract GEMM dispatch tables

Port `weight_gemm` from `llama.rs` (lines 1472-1496) and prefill GEMM dispatches from `forward_prefill_chunk` (lines 2049-2373).

#### Task 3.2: Extract FusedQKV dispatch tables

Port the fused QKV/QKVZA/Gate+Up kernel selection from `qwen35.rs` and prefill fused paths from `llama.rs`.

---

### Phase 4: Attention + MoE (4-5 PRs, ~3 weeks)

#### Task 4.1: Extract KV cache write dispatch

Port `kv_cache_write_*` dispatch lines from `qwen35.rs` (9231-9338) and `llama.rs` (2655-2700).

#### Task 4.2: Extract attention flash dispatch

Port `attention_flash_*` and `attention_q8_0_kv` selection.

#### Task 4.3: Extract MoE dispatch

Port DeepSeek `ffn_routed` indexed blob dispatch and Qwen3.5 MoE grouped GEMM.

---

### Phase 5: Model-Specific Kernels (2-3 PRs, ~1 week)

#### Task 5.1: Extract DeepSeek compressor

**File:** Create: `crates/hipfire-dispatch/src/model_ext/deepseek4.rs`

Implement `Deepseek4ModelExt` trait with `run_compressor()`, `run_joint_kv()`, `run_q_lora()` — model-specific but uses `gemv_auto` internally.

#### Task 5.2: Extract Qwen3.5 DeltaNet state

DeltaNet state management (linear recurrence step, state quant) stays model-level. If reusable ops emerge, promote to a shared helper.

---

### Phase 6: Flag Cleanup (1 PR, ~1 week)

- Remove `feature = "new-rotation"`, `feature = "new-gemv"`, etc. from all Cargo.toml files
- Remove old dispatch trees from `llama.rs`, `qwen35.rs`, `deepseek4/forward.rs`, `qwen2.rs`
- Remove unused `fn weight_gemv`, `weight_gemv_prerotated`, `weight_gemv_residual`, `weight_gemv_swiglu_residual`, `gemv_auto`, `gemv_auto_batched_wmma`
- Full cleanup pass
- Final coherence gate

---

## Migration Strategy

| Phase | # PRs | Risk | Key verification |
|---|---|---|---|
| 0: Foundation crate | 1 | ✅ None | Compiles |
| 1: Rotation | 2-3 | ✅ Low | Byte-comparison integration test, coherence gate |
| 2: GEMV | 4-6 | ⚠️ **Highest** | Byte-comparison + kernel params audit + coherence gate |
| 3: GEMM + FusedQKV | 3-4 | 🔶 Medium | Byte-comparison, coherence gate |
| 4: Attention + MoE | 4-5 | 🔶 Medium-High | Coherence gate + eyeball decode output |
| 5: Model-specific | 2-3 | ✅ Low | Model functional tests |
| 6: Flag cleanup | 1 | 🔶 Medium | Full test suite passes, coherence gate |
| **Total** | **17-23** | | **8-12 weeks** |

### Key risk mitigations

**GEMV perf regression (highest risk):** every PR compares: (a) byte-level output equality, (b) selected kernel name match, (c) coherence gate tok/s within 2% of baseline. Feature flags stay for one release cycle so users can revert.

**Keep compilable at every commit:** default-features = old path only. CI compiles both `--release` (default) and `--release --features new-rotation,new-gemv,...` (all features on). The old path is removed only after the new path passes coherence on 3+ runs.

**Model-specific kernels in `model_ext/`:** prevents the dispatch abstraction from becoming a dumping ground. Only promote to a family when ≥2 models need it.
