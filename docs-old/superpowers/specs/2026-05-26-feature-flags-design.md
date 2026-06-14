# FeatureFlags struct — Design Spec

**Issue:** <https://github.com/Kaden-Schutt/hipfire/issues/328> (Task #1)
**Branch:** `feature/328-feature-flags`
**Date:** 2026-05-26

## Problem

`rdna-compute/src/dispatch.rs` (28,657 lines) has 76 raw `std::env::var("HIPFIRE_...")` calls scattered across hot-path dispatch methods. About half are cached via `OnceLock` free functions (lines 46–498); the other half are per-call reads that hit `env::var`'s global lock on every dispatch invocation. There is no single source of truth — arch-dependent defaults, env overrides, and cached state are split across free functions, `Gpu` fields, and inline `match` blocks.

`hipfire-runtime` has a similar problem (~52 env reads in `tokenizer.rs`, `dflash.rs`, `loop_guard.rs`, `multi_gpu.rs`, `llama.rs`, `arch.rs`), though less severe since the runtime is not on the GPU hot path.

## Scope

This PR consolidates **all** `HIPFIRE_` env-var reads in both `rdna-compute` and `hipfire-runtime` into typed, immutable structs built once at startup. It also removes the `OnceLock` free functions and `pub fn` exports for arch capability queries, replacing them with methods on the new structs.

### In scope

- New `FeatureFlags` struct in `rdna-compute/src/feature_flags.rs`
- New `RuntimeConfig` struct in `hipfire-runtime/src/config.rs`
- Migration of all ~76 env reads in `dispatch.rs` → `FeatureFlags` fields
- Migration of all ~22 `OnceLock` free functions (lines 46–498) → `FeatureFlags` methods
- Removal of the four `mmq_screen`, `mmq_screen_threshold`, `force_blob_path`, `mmq_diag_quantize_only` fields from `Gpu` → `FeatureFlags`
- `Arc<FeatureFlags>` field on `Gpu`
- Update all downstream crate call sites (`hipfire-arch-qwen35`, `hipfire-runtime`, examples)
- Migration of ~52 env reads in `hipfire-runtime/src/*.rs` → `RuntimeConfig`

### Out of scope (later tasks in #328)

- Arch routing centralization (task #2)
- Gpu God Object decomposition (task #3)
- Quant-format dispatch table (task #4)
- `kernels.rs` submodules (task #5)
- Relocating misplaced code (task #6)
- Graph-capture safety types (task #7)
- `profile.rs` split (task #8)

## Design

### 1. `FeatureFlags` struct (rdna-compute)

File: `crates/rdna-compute/src/feature_flags.rs`

```rust
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct FeatureFlags {
    // Arch identity (resolved at construction, not from env)
    pub arch: String,

    // ── GEMV tuning ───────────────────────────────────────────────
    pub gemv_rows: Option<u32>,
    pub gemv_dp4a_default_on: bool,
    pub gemv_prefetch_default_on: bool,
    pub gfx942_lds_gemv_default_on: bool,
    pub gemv_rows_default: u32,

    // ── Quant / format toggles ────────────────────────────────────
    pub hfq3_dp4a: Option<bool>,
    pub hfq3_mmq: Option<bool>,
    pub hfq4_mmq_rdna2: Option<bool>,
    pub fp8_wmma: bool,
    pub dot2_gemv: bool,
    pub gcn5_wave64_hybrid: Option<bool>,
    pub mmq_override: Option<bool>,
    pub mmq_min_batch: Option<usize>,
    pub fp16_disabled: bool,
    pub wo_mmq: bool,
    pub lm_head_wmma_disabled: bool,

    // ── MMQ screening ─────────────────────────────────────────────
    pub mmq_screen: bool,
    pub mmq_screen_threshold: f32,
    pub mmq_diag_quantize_only: bool,

    // ── Kernel variant overrides ─────────────────────────────────
    pub lloyd_mb4: Option<Mb4Mode>,
    pub hfq4g128_mmq: bool,
    pub gate_up_variant: Option<String>,
    pub gfx942_gemv_v2: Option<bool>,
    pub gfx942_gemv_v3: bool,
    pub gfx942_rmsnorm_split: bool,
    pub gfx942_mfma_prefill: Option<String>,
    pub moe_grouped_i8: Option<bool>,
    pub moe_grouped_i8_k8: bool,
    pub moe_grouped_i8_k4: bool,
    pub moe_grouped_i8_k4_gfx12: bool,
    pub moe_grouped_m2: bool,
    pub moe_hfq6_v2: bool,

    // ── Graph / capture / deterministic ─────────────────────────────
    pub force_blob_path: bool,
    pub gemm_dump: bool,
    pub deterministic: bool,
    pub mw16: bool,
    pub q8_batched_legacy: bool,
    pub rope_interleaved_legacy: bool,
    pub wo_wmma_variant: Option<String>,

    // ── rocBLAS ────────────────────────────────────────────────────
    pub rocblas_all_archs: bool,
    pub rocblas_off: bool,
    pub rocblas_min_batch: Option<usize>,

    // ── Kernels.rs / compiler.rs ───────────────────────────────────
    pub lloyd_force_baseline: Option<bool>,
    pub rdna2_variant: Option<String>,
    pub hipcc_extra_flags: Option<String>,

    // ── MQ3 variants of mb4 ───────────────────────────────────────
    // (MQ3_MB4 is read 4 times; consolidated here)
    pub mq3_mb4: Option<Mb4Mode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mb4Mode {
    Pack1,
    Pack2,
    Pack4,
}
```

#### Constructor

```rust
impl FeatureFlags {
    pub fn from_env(arch: &str) -> Self {
        // Read every HIPFIRE_ env var exactly once.
        // Arch-dependent defaults resolved here — no runtime arch checks elsewhere.
        Self {
            arch: arch.to_string(),
            gemv_rows: env_parse("HIPFIRE_GEMV_ROWS"),
            gemv_dp4a_default_on: arch == "gfx906",
            // ... etc
        }
    }
}
```

Every field is set from `std::env::var` exactly once, at `Gpu::init()` time. No field is mutated after construction. The `arch` string is stored so methods can combine arch-defaults with env-overrides:

```rust
impl FeatureFlags {
    pub fn gemv_dp4a_enabled(&self) -> bool {
        self.hfq3_dp4a.unwrap_or(self.gemv_dp4a_default_on)
    }

    pub fn has_wmma_f16(&self) -> bool {
        matches!(self.arch.as_str(),
            "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103"
          | "gfx1150" | "gfx1151" | "gfx1152"
          | "gfx1200" | "gfx1201"
        )
    }
}
```

### 2. Integration with `Gpu`

```diff
 pub struct Gpu {
+    pub flags: Arc<FeatureFlags>,
     pub hip: HipRuntime,
     pub arch: String,
     // ...
-    pub mmq_screen: bool,
-    pub mmq_screen_threshold: f32,
-    pub force_blob_path: bool,
-    pub mmq_diag_quantize_only: bool,
 }
```

`Gpu::init()`:

```rust
let flags = Arc::new(FeatureFlags::from_env(&arch));
// ... build Gpu with flags ...
Ok(Self { flags, hip, arch, ... })
```

Downstream code accesses flags via `gpu.flags.gemv_dp4a_enabled()` instead of the old `gemv_dp4a_enabled(&gpu.arch)` free function.

The `arch` field stays on `Gpu` for backward compatibility — it's still needed by methods that don't go through `FeatureFlags` (e.g., kernel source selection). `FeatureFlags.arch` mirrors it.

### 3. Free function removal

All free functions in `dispatch.rs` lines 46–498 are replaced:

| Old free function | New method on `FeatureFlags` |
|---|---|
| `gemv_rows_override()` | `flags.gemv_rows` (field) |
| `gemv_dp4a_enabled(arch)` | `flags.gemv_dp4a_enabled()` |
| `gemv_prefetch_enabled(arch)` | `flags.gemv_prefetch_enabled()` |
| `gfx942_lds_gemv_enabled(arch)` | `flags.gfx942_lds_gemv_enabled()` |
| `gemv_rows_default(arch)` | `flags.gemv_rows_default` (field) |
| `has_dot2_f32_f16(arch)` | `flags.has_dot2_f16()` |
| `hfq3_sdot4_gfx10_enabled(arch)` | `flags.hfq3_sdot4_gfx10_enabled()` |
| `hfq3_dp4a_enabled()` | `flags.hfq3_dp4a_enabled()` |
| `hfq3_mmq_rdna2_enabled()` | `flags.hfq3_mmq_rdna2_enabled()` |
| `hfq4_mmq_rdna2_enabled()` | `flags.hfq4_mmq_rdna2_enabled()` |
| `has_wmma_f16(arch)` | `flags.has_wmma_f16()` |
| `has_wmma_f16_gfx12(arch)` | `flags.has_wmma_f16_gfx12()` |
| `has_wmma_fp8_gfx12(arch)` | `flags.has_wmma_fp8_gfx12()` |
| `is_fp8_wmma_enabled()` | `flags.fp8_wmma` (field) |
| `is_dot2_gemv_enabled()` | `flags.dot2_gemv` (field) |
| `is_gcn5_wave64(arch)` | `flags.is_gcn5_wave64()` |
| `has_wave64_native(arch)` | `flags.has_wave64_native()` |
| `has_mmq_dp4a_or_wmma(arch)` | `flags.has_mmq_dp4a_or_wmma()` |
| `should_use_mmq(arch, batch_size)` | `flags.should_use_mmq(batch_size)` |
| `mmq_env_override()` | `flags.mmq_override` (field) |
| `mmq_min_batch_override()` | `flags.mmq_min_batch` (field) |
| `fp16_disabled()` | `flags.fp16_disabled` (field) |
| `wo_mmq_enabled()` | `flags.wo_mmq` (field) |
| `lm_head_wmma_disabled()` | `flags.lm_head_wmma_disabled` (field) |

Export changes in `lib.rs`:

```diff
 pub use dispatch::{
-    gen_fwht_signs, gemv_dp4a_enabled, has_wmma_f16, DType, Gpu, GpuTensor,
-    LLOYD_MQ4_GROUP_BYTES, MMQ_CURRENT_LAYER,
+    gen_fwht_signs, DType, Gpu, GpuTensor, LLOYD_MQ4_GROUP_BYTES, MMQ_CURRENT_LAYER,
 };
+pub use feature_flags::FeatureFlags;
```

### 4. `RuntimeConfig` struct (hipfire-runtime)

File: `crates/hipfire-runtime/src/config.rs`

```rust
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    // Prompt / tokenizer
    pub normalize_prompt: bool,           // HIPFIRE_NORMALIZE_PROMPT (default ON since 2026-04-26)
    pub prompt_token_heat: bool,           // HIPFIRE_PROMPT_TOKEN_HEAT
    pub prompt_heat_json: bool,           // HIPFIRE_PROMPT_HEAT_JSON
    pub prompt_heat_limit: usize,         // HIPFIRE_PROMPT_HEAT_LIMIT

    // KV / attention
    pub kv_mode: String,                  // HIPFIRE_KV_MODE
    pub kv_physical_cap: Option<usize>,   // HIPFIRE_KV_PHYSICAL_CAP
    pub attn_flash: Option<String>,       // HIPFIRE_ATTN_FLASH

    // DFlash / speculative decode
    pub dflash_draft: Option<String>,     // HIPFIRE_DFLASH_DRAFT
    pub dflash_mode: String,              // HIPFIRE_DFLASH_MODE (auto/on/off)

    // LM head
    pub lm_head_f16: String,              // HIPFIRE_LM_HEAD_F16 (auto/native/f32/legacy)

    // Loop guard / DFlash parameters
    pub adaptive_b_unsafe: bool,          // HIPFIRE_ADAPTIVE_B_UNSAFE
    pub adaptive_b_up: f64,               // HIPFIRE_ADAPTIVE_B_UP
    pub adaptive_b_down: f64,             // HIPFIRE_ADAPTIVE_B_DOWN
    pub ddtree_budget: usize,             // HIPFIRE_DDTREE_BUDGET
    pub ddtree_topk: usize,               // HIPFIRE_DDTREE_TOPK

    // Profiling / timing
    pub profile: bool,                    // HIPFIRE_PROFILE
    pub profile_cycles_target: usize,     // HIPFIRE_PROFILE_CYCLES
    pub host_timing: bool,                // HIPFIRE_HOST_TIMING
    pub dpm_warmup_secs: f32,             // HIPFIRE_DPM_WARMUP_SECS

    // Graph
    pub verify_graph: bool,               // HIPFIRE_VERIFY_GRAPH (default ON)

    // Execution
    pub local: bool,                       // HIPFIRE_LOCAL
    pub pp_layers: Option<usize>,          // HIPFIRE_PP_LAYERS (pipeline parallel)
    pub pp_dflash: bool,                   // HIPFIRE_PP_DFLASH
    pub pp_pflash: bool,                  // HIPFIRE_PP_PFLASH
    pub chat_template_file: Option<String>, // HIPFIRE_CHAT_TEMPLATE_FILE
    pub jinja_chat: bool,                 // HIPFIRE_JINJA_CHAT
    pub experimental_budget_alert: bool,  // HIPFIRE_EXPERIMENTAL_BUDGET_ALERT
    pub emit_token_ids: bool,             // HIPFIRE_EMIT_TOKEN_IDS

    // DDTree / DFlash loop break (diagnostic)
    pub dflash_loop_break: Option<f32>,           // HIPFIRE_DFLASH_LOOP_BREAK
    pub dflash_loop_break_temp: f32,              // HIPFIRE_DFLASH_LOOP_BREAK_TEMP
    pub dflash_loop_break_stop_after: usize,      // HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER
    pub dflash_loop_break_rp_step: f32,           // HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP
    pub dflash_loop_break_rp_max: f32,            // HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX
    pub dflash_loop_break_recovery: usize,        // HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY
    pub dflash_loop_break_max_escalations: usize, // HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS
    pub ddtree_logw_cutoff: String,              // HIPFIRE_DDTREE_LOGW_CUTOFF
}

impl RuntimeConfig {
    pub fn from_env() -> Self { /* read all env vars once */ }
}
```

Built once in `daemon.rs` at startup, threaded through to `dflash.rs`, `tokenizer.rs`, `loop_guard.rs`, `llama.rs`, `multi_gpu.rs`, `arch.rs` via a parameter or `Arc<RuntimeConfig>`.

### 5. Downstream impact

#### `hipfire-arch-qwen35`

Currently uses `rdna_compute::gemv_dp4a_enabled(arch)` and `rdna_compute::has_wmma_f16(arch)`. These calls change to `gpu.flags.gemv_dp4a_enabled()` and `gpu.flags.has_wmma_f16()`. Since `Gpu` already carries `flags: Arc<FeatureFlags>`, and the arch-qwen35 code already holds a `&mut Gpu`, this is a straightforward method-access change.

#### `hipfire-runtime/examples/`

The `daemon.rs` and `dflash_spec_demo.rs` examples read env vars directly. They'll construct a `RuntimeConfig` at startup and pass it down. The `Gpu` inside the daemon already gets `FeatureFlags` from `Gpu::init()`.

#### `hipfire-atlas`

`profile_report.rs` reads `HIPFIRE_` vars. Same pattern: construct `RuntimeConfig`, pass it.

### 6. Testing strategy

- Unit test `FeatureFlags::from_env()` with env vars set/cleared via `temp_env` or similar
- Existing smoke tests (`coherence-gate-dflash.sh`, `mq3-mq2-sweep.sh`) serve as integration validation
- The refactor is mechanical (env reads → struct fields). No algorithm changes. Behaviour must be byte-identical.
- **Hard rule from CLAUDE.md:** run `./scripts/coherence-gate-dflash.sh` after any change touching kernel dispatch.

### 7. Migration order

1. Create `feature_flags.rs` with `FeatureFlags` struct + `from_env()` constructor
2. Add `flags: Arc<FeatureFlags>` field to `Gpu`, populate in `Gpu::init()`
3. Remove the four `mmq_screen` / `mmq_screen_threshold` / `force_blob_path` / `mmq_diag_quantize_only` fields from `Gpu`, replace uses with `self.flags.*`
4. Convert each `OnceLock` free function to a `FeatureFlags` method, one at a time, updating call sites
5. Remove `pub use dispatch::{gemv_dp4a_enabled, has_wmma_f16, ...}` from `lib.rs`, add `pub use feature_flags::FeatureFlags`
6. Convert all remaining raw `std::env::var("HIPFIRE_...")` calls in `dispatch.rs` methods to `self.flags.*` field accesses
7. Update `hipfire-arch-qwen35` call sites
8. Create `RuntimeConfig` in `hipfire-runtime`
9. Migrate env reads in `hipfire-runtime/src/*.rs`
10. Update `hipfire-runtime/examples/` to construct `RuntimeConfig`
11. Run coherence gate + smoke tests

Each step should be a separate commit for bisectability.

### 8. Non-goals

- No arch-routing centralization (task #2)
- No Gpu decomposition (task #3)
- No dispatch table (task #4)
- No `kernels.rs` submodule split (task #5)
- No relocating `dpm_warmup`, `cross_entropy_loss`, etc. (task #6)
- No graph-capture type safety (task #7)
- No `profile.rs` split (task #8)
- No `ProfiledLaunch` builder (task #9)
- No `GpuPool` consistency (task #10)