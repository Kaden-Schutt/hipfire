# FeatureFlags Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate all HIPFIRE_ env-var reads in rdna-compute and hipfire-runtime into typed, immutable structs built once at startup.

**Architecture:** New `FeatureFlags` struct in `rdna-compute/src/feature_flags.rs` replaces all `OnceLock` free functions and raw `std::env::var` calls. `Arc<FeatureFlags>` stored on `Gpu`. New `RuntimeConfig` struct in `hipfire-runtime/src/config.rs` consolidates runtime env reads. Existing public free functions (`gemv_dp4a_enabled`, `has_wmma_f16`) removed; callers use `gpu.flags.method()`.

**Tech Stack:** Rust 2021, `std::sync::Arc`, `std::sync::OnceLock` (existing dependency). No new crates.

**Worktree:** `../hipfire-328-feature-flags` on branch `feature/328-feature-flags`

---

## File Structure

### New files
- `crates/rdna-compute/src/feature_flags.rs` — `FeatureFlags` struct, `Mb4Mode` enum, `from_env()` constructor, all methods replacing free functions
- `crates/hipfire-runtime/src/config.rs` — `RuntimeConfig` struct, `from_env()` constructor

### Modified files
- `crates/rdna-compute/src/lib.rs` — add `pub mod feature_flags;`, update `pub use`
- `crates/rdna-compute/src/dispatch.rs` — remove all free functions (lines 47–499), remove `mmq_screen`/`mmq_screen_threshold`/`force_blob_path`/`mmq_diag_quantize_only` fields from `Gpu`, add `flags: Arc<FeatureFlags>` field, replace all `std::env::var("HIPFIRE_...")` calls with `self.flags.*` accesses
- `crates/rdna-compute/src/kernels.rs` — replace `std::env::var("HIPFIRE_LLOYD_FORCE_BASELINE")` and `std::env::var("HIPFIRE_RDNA2_VARIANT")` with parameter passing (struct field or method arg)
- `crates/rdna-compute/src/compiler.rs` — replace `std::env::var("HIPFIRE_HIPCC_EXTRA_FLAGS")` with struct field
- `crates/rdna-compute/src/arch_caps.rs` — `paro_la_gates_mq4g128_default` stays (it's pure arch logic, no env var)
- `crates/hipfire-runtime/src/lib.rs` — add `pub mod config;`
- `crates/hipfire-runtime/src/tokenizer.rs` — replace env reads with `RuntimeConfig` fields
- `crates/hipfire-runtime/src/dflash.rs` — replace env reads with `RuntimeConfig` fields
- `crates/hipfire-runtime/src/llama.rs` — replace env reads with `RuntimeConfig` fields
- `crates/hipfire-runtime/src/loop_guard.rs` — replace env reads with `RuntimeConfig` fields
- `crates/hipfire-runtime/src/multi_gpu.rs` — replace env reads with `RuntimeConfig` fields
- `crates/hipfire-runtime/src/arch.rs` — replace env reads with `RuntimeConfig` fields
- `crates/hipfire-runtime/examples/daemon.rs` — construct `RuntimeConfig`, pass through
- `crates/hipfire-runtime/examples/dflash_spec_demo.rs` — construct `RuntimeConfig`, pass through
- `crates/hipfire-arch-qwen35/src/qwen35.rs` — replace `rdna_compute::gemv_dp4a_enabled(&gpu.arch)` with `gpu.flags.gemv_dp4a_enabled()`, same for `has_wmma_f16`
- `crates/hipfire-arch-qwen35/src/arch.rs` — import `FeatureFlags` if needed
- `crates/hipfire-arch-qwen35/src/paro_la_gates_codec.rs` — no change (uses `arch_caps::paro_la_gates_mq4g128_default`, not an env var)

---

## Task 1: Create `FeatureFlags` struct in rdna-compute

**Files:**
- Create: `crates/rdna-compute/src/feature_flags.rs`
- Modify: `crates/rdna-compute/src/lib.rs`

- [ ] **Step 1: Create `feature_flags.rs` with the struct definition and `from_env`**

Create `crates/rdna-compute/src/feature_flags.rs` with the complete `FeatureFlags` struct. Every env var that was previously read via `std::env::var("HIPFIRE_X")` in dispatch.rs, kernels.rs, or compiler.rs becomes a field. The `from_env(arch: &str)` constructor reads all env vars once at init time.

```rust
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Mb4Mode {
    Pack1,
    Pack2,
    Pack4,
}

#[derive(Debug, Clone)]
pub struct FeatureFlags {
    // ── Arch identity ──────────────────────────────────────────────
    pub arch: String,

    // ── GEMV tuning ────────────────────────────────────────────────
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
    pub mq3_mb4: Option<Mb4Mode>,
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

    // ── Kernels.rs env reads ───────────────────────────────────────
    pub lloyd_force_baseline: bool,
    pub rdna2_variant: Option<u32>,

    // ── Compiler.rs env reads ──────────────────────────────────────
    pub hipcc_extra_flags: String,
}

impl FeatureFlags {
    pub fn from_env(arch: &str) -> Self {
        let parse_bool = |name: &str| -> Option<bool> {
            match std::env::var(name).ok().as_deref() {
                Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON") => Some(true),
                Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => Some(false),
                _ => None,
            }
        };

        let parse_usize = |name: &str| -> Option<usize> {
            std::env::var(name).ok().and_then(|s| s.parse().ok())
        };

        let parse_mb4 = |name: &str| -> Option<Mb4Mode> {
            match std::env::var(name).ok().as_deref() {
                Some("1") => Some(Mb4Mode::Pack1),
                Some("2") => Some(Mb4Mode::Pack2),
                Some("4") => Some(Mb4Mode::Pack4),
                _ => None,
            }
        };

        // Per-arch defaults (resolved here, not at call sites)
        let is_gfx906 = arch == "gfx906";
        let is_gfx942_family = matches!(arch, "gfx940" | "gfx941" | "gfx942");

        // mmq_screen defaults: false on all arches as of 2026-05-18
        let mmq_screen_default = false;
        let mmq_screen_threshold_default: f32 = if is_gfx906 { 0.50 } else { 0.10 };

        // gemv_rows_default
        let gemv_rows_default: u32 = match arch {
            "gfx1100" | "gfx1101" | "gfx1102" => 1,
            "gfx1030" | "gfx1031" => 1,
            "gfx906" | "gfx908" | "gfx940" | "gfx941" | "gfx942" => 1,
            _ => 2,
        };

        Self {
            arch: arch.to_string(),

            // GEMV tuning
            gemv_rows: std::env::var("HIPFIRE_GEMV_ROWS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .map(|r| match r { 1 | 2 | 4 | 8 => r, _ => 1 }),
            gemv_dp4a_default_on: is_gfx906,
            gemv_prefetch_default_on: is_gfx906,
            gfx942_lds_gemv_default_on: false,
            gemv_rows_default,

            // Quant/format toggles
            hfq3_dp4a: parse_bool("HIPFIRE_HFQ3_DP4A"),
            hfq3_mmq: parse_bool("HIPFIRE_HFQ3_MMQ"),
            hfq4_mmq_rdna2: parse_bool("HIPFIRE_HFQ4_MMQ_RDNA2"),
            fp8_wmma: std::env::var("HIPFIRE_FP8_WMMA").map_or(false, |v| v == "1"),
            dot2_gemv: std::env::var("HIPFIRE_DOT2_GEMV").map_or(false, |v| v == "1"),
            gcn5_wave64_hybrid: parse_bool("HIPFIRE_GCN5_WAVE64_HYBRID"),
            mmq_override: match std::env::var("HIPFIRE_MMQ").ok().as_deref() {
                Some("0") | Some("off") => Some(false),
                Some("1") | Some("on") => Some(true),
                _ => None,
            },
            mmq_min_batch: parse_usize("HIPFIRE_MMQ_MIN_BATCH"),
            fp16_disabled: std::env::var("HIPFIRE_FP16").map_or(false, |v| v == "0"),
            wo_mmq: std::env::var("HIPFIRE_WO_MMQ").ok().as_deref() == Some("1"),
            lm_head_wmma_disabled: std::env::var("HIPFIRE_LM_HEAD_WMMA").map_or(false, |v| v == "0"),

            // MMQ screening
            mmq_screen: std::env::var("HIPFIRE_MMQ_SCREEN")
                .ok()
                .map(|v| v == "1")
                .unwrap_or(mmq_screen_default),
            mmq_screen_threshold: std::env::var("HIPFIRE_MMQ_SCREEN_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(mmq_screen_threshold_default),
            mmq_diag_quantize_only: std::env::var("HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY")
                .ok()
                .as_deref() == Some("1"),

            // Kernel variant overrides
            lloyd_mb4: parse_mb4("HIPFIRE_LLOYD_MB4"),
            mq3_mb4: parse_mb4("HIPFIRE_MQ3_MB4"),
            hfq4g128_mmq: std::env::var("HIPFIRE_HFQ4G128_MMQ").as_deref() != Ok("0"),
            gate_up_variant: std::env::var("HIPFIRE_GATE_UP_VARIANT").ok(),
            gfx942_gemv_v2: parse_bool("HIPFIRE_GFX942_GEMV_V2"),
            gfx942_gemv_v3: std::env::var("HIPFIRE_GFX942_GEMV_V3").map_or(false, |v| v == "1"),
            gfx942_rmsnorm_split: matches!(arch, "gfx940" | "gfx941" | "gfx942")
                && std::env::var("HIPFIRE_GFX942_RMSNORM_SPLIT").as_deref() != Ok("0"),
            gfx942_mfma_prefill: std::env::var("HIPFIRE_GFX942_MFMA_PREFILL").ok(),
            moe_grouped_i8: match std::env::var("HIPFIRE_MOE_GROUPED_I8").ok().as_deref() {
                Some("1") => Some(true),
                Some("0") => Some(false),
                _ => None,
            },
            moe_grouped_i8_k8: std::env::var("HIPFIRE_MOE_GROUPED_I8_K8").as_deref() == Ok("1"),
            moe_grouped_i8_k4: std::env::var("HIPFIRE_MOE_GROUPED_I8_K4").as_deref() == Ok("1"),
            moe_grouped_i8_k4_gfx12: std::env::var("HIPFIRE_MOE_GROUPED_I8_K4_GFX12").as_deref() == Ok("1"),
            moe_grouped_m2: std::env::var("HIPFIRE_MOE_GROUPED_M2").as_deref() == Ok("1"),
            moe_hfq6_v2: std::env::var("HIPFIRE_MOE_HFQ6_V2").as_deref() == Ok("1"),

            // Graph / capture / deterministic
            force_blob_path: std::env::var("HIPFIRE_BLOB_FORCE").ok().as_deref() == Some("1"),
            gemm_dump: std::env::var("HIPFIRE_GEMM_DUMP").ok().as_deref() == Some("1"),
            deterministic: std::env::var("HIPFIRE_DETERMINISTIC").ok().as_deref() == Some("1"),
            mw16: std::env::var("HIPFIRE_MW16").map_or(false, |v| v == "1"),
            q8_batched_legacy: std::env::var("HIPFIRE_Q8_BATCHED_LEGACY").as_deref() == Ok("1"),
            rope_interleaved_legacy: std::env::var("HIPFIRE_ROPE_INTERLEAVED_LEGACY").ok().as_deref() == Some("1"),
            wo_wmma_variant: std::env::var("HIPFIRE_WO_WMMA_VARIANT").ok(),

            // rocBLAS
            rocblas_all_archs: std::env::var("HIPFIRE_ROCBLAS_ALL_ARCHS").ok().as_deref() == Some("1"),
            rocblas_off: std::env::var("HIPFIRE_ROCBLAS_OFF").ok().as_deref() == Some("1"),
            rocblas_min_batch: parse_usize("HIPFIRE_ROCBLAS_MIN_BATCH"),

            // Kernels.rs
            lloyd_force_baseline: std::env::var("HIPFIRE_LLOYD_FORCE_BASELINE").ok().as_deref() == Some("1"),
            rdna2_variant: std::env::var("HIPFIRE_RDNA2_VARIANT")
                .ok()
                .and_then(|s| s.parse::<u32>().ok()),

            // Compiler.rs
            hipcc_extra_flags: std::env::var("HIPFIRE_HIPCC_EXTRA_FLAGS").unwrap_or_default(),
        }
    }

    // ── Methods replacing free functions ─────────────────────────────

    pub fn gemv_dp4a_enabled(&self) -> bool {
        self.hfq3_dp4a.unwrap_or(self.gemv_dp4a_default_on)
    }

    pub fn gemv_prefetch_enabled(&self) -> bool {
        self.gemv_prefetch.unwrap_or(self.gemv_prefetch_default_on)
    }

    pub fn has_wmma_f16(&self) -> bool {
        self.arch.starts_with("gfx11")
    }

    pub fn has_wmma_f16_gfx12(&self) -> bool {
        self.arch.starts_with("gfx12")
    }

    pub fn has_wmma_fp8_gfx12(&self) -> bool {
        self.arch.starts_with("gfx12")
    }

    pub fn has_dot2_f32_f16(&self) -> bool {
        matches!(self.arch.as_str(),
            "gfx1011" | "gfx1012"
            | "gfx1030" | "gfx1031" | "gfx1032"
            | "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103"
            | "gfx1150" | "gfx1151" | "gfx1152"
            | "gfx1200" | "gfx1201")
    }

    pub fn hfq3_sdot4_gfx10_enabled(&self) -> bool {
        matches!(self.arch.as_str(), "gfx1011" | "gfx1012" | "gfx1030" | "gfx1031" | "gfx1032")
    }

    pub fn has_mmq_dp4a_or_wmma(&self) -> bool {
        matches!(self.arch.as_str(),
            "gfx906"
            | "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103"
            | "gfx1150" | "gfx1151" | "gfx1152")
    }

    pub fn has_wave64_native(&self) -> bool {
        matches!(self.arch.as_str(), "gfx906" | "gfx908" | "gfx940" | "gfx941" | "gfx942")
    }

    pub fn is_gcn5_wave64(&self) -> bool {
        if self.arch == "gfx906" {
            return true;
        }
        self.arch == "gfx908" && self.gcn5_wave64_hybrid.unwrap_or(false)
    }

    pub fn should_use_mmq(&self, batch_size: usize) -> bool {
        if !self.has_mmq_dp4a_or_wmma() {
            return false;
        }
        match self.mmq_override {
            Some(false) => false,
            Some(true) => true,
            None => {
                let arch_min_batch: usize = if self.arch == "gfx906" { 8 } else { 256 };
                let min_batch = self.mmq_min_batch.unwrap_or(arch_min_batch);
                batch_size >= min_batch
            }
        }
    }

    pub fn hfq3_dp4a_enabled(&self) -> bool {
        self.hfq3_dp4a.unwrap_or(false) && self.hfq3_sdot4_gfx10_enabled()
    }

    pub fn hfq3_mmq_rdna2_enabled(&self) -> bool {
        self.hfq3_mmq.unwrap_or(false) && self.hfq3_sdot4_gfx10_enabled()
    }

    pub fn hfq4_mmq_rdna2_enabled(&self) -> bool {
        self.hfq4_mmq_rdna2.unwrap_or(false)
            && self.has_dot2_f32_f16()
    }

    pub fn gfx942_lds_gemv_enabled(&self) -> bool {
        self.gfx942_lds_gemv_default_on
    }
}
```

**Important:** The `gemv_prefetch_enabled` method above has a bug — I'm reusing `hfq3_dp4a` field instead of a dedicated `gemv_prefetch` field. Let me fix this: the `FeatureFlags` struct needs a `gemv_prefetch: Option<bool>` field and the method should combine it with the default.

Actually, re-reading the code: `gemv_prefetch_enabled(arch: &str)` uses the same boolean-parsing pattern as `gemv_dp4a_enabled`. It reads `HIPFIRE_GEMV_PREFETCH`. I need a separate field. Let me correct the struct — add `pub gemv_prefetch: Option<bool>` and fix the method:

```rust
    pub fn gemv_prefetch_enabled(&self) -> bool {
        self.gemv_prefetch.unwrap_or(self.gemv_prefetch_default_on)
    }
```

And add to `from_env`:

```rust
            gemv_prefetch: parse_bool("HIPFIRE_GEMV_PREFETCH"),
```

This correction applies to the struct definition above.

- [ ] **Step 2: Add `pub mod feature_flags` to `lib.rs`**

```rust
// In crates/rdna-compute/src/lib.rs:
mod compiler;
mod dispatch;
mod kernels;
pub mod arch_caps;
pub mod feature_flags;  // NEW
pub mod pool;
pub mod profile;
pub mod profile_rocprof;
pub mod profiler;
```

And add the re-export:

```rust
pub use feature_flags::FeatureFlags;
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p rdna-compute`

Expected: compiles with warnings about unused fields (that's OK — we'll wire them up in later tasks).

- [ ] **Step 4: Commit**

```bash
git add crates/rdna-compute/src/feature_flags.rs crates/rdna-compute/src/lib.rs
git commit -m "feat(rdna-compute): add FeatureFlags struct with from_env constructor

All HLSIMPL env vars enumerated as typed fields. from_env reads them once
at init time. Methods replace OnceLock free functions. Not yet wired into
Gpu — that's the next commit.

Assisted-by: OpenCode:anthropic/claude-sonnet-4-20250514"
```

---

## Task 2: Wire `Arc<FeatureFlags>` into `Gpu`, remove 4 migrated fields

**Files:**
- Modify: `crates/rdna-compute/src/dispatch.rs`

- [ ] **Step 1: Add `flags: Arc<FeatureFlags>` field to `Gpu` struct**

At line 690 area in `dispatch.rs`, add after `pub arch: String,`:

```rust
    pub flags: Arc<FeatureFlags>,
```

And add the import at the top:

```rust
use crate::feature_flags::FeatureFlags;
```

- [ ] **Step 2: Remove `mmq_screen`, `mmq_screen_threshold`, `force_blob_path`, `mmq_diag_quantize_only` fields from `Gpu`**

Remove these four fields from the `Gpu` struct definition (around lines 752–776):

```rust
// REMOVE these fields:
    pub mmq_screen: bool,
    pub mmq_screen_threshold: f32,
    pub force_blob_path: bool,
    pub mmq_diag_quantize_only: bool,
```

- [ ] **Step 3: Construct `FeatureFlags` in `Gpu::init()` and `Gpu::init_with_device()`**

In `Gpu::init_with_device()`, before the `Ok(Self { ... })` block, construct the flags:

```rust
let flags = Arc::new(FeatureFlags::from_env(&arch));
```

In the `Ok(Self { ... })` struct literal, add `flags` and remove the four fields:

```rust
Ok(Self {
    flags,
    hip,
    arch,
    device_id: id,
    compiler,
    // ...
    // REMOVED: mmq_screen, mmq_screen_threshold, force_blob_path, mmq_diag_quantize_only
    // (now on flags)
    capture_mode: false,
    // ...
})
```

- [ ] **Step 4: Replace all `self.mmq_screen` → `self.flags.mmq_screen`**

Every use of `self.mmq_screen` in dispatch.rs (lines 8474, 8508, 8526, 9187, 9203, 9220, 9920, 9932, 9951, 16356, 16364, 16398) changes to `self.flags.mmq_screen`.

Similarly:
- `self.mmq_screen_threshold` → `self.flags.mmq_screen_threshold` (line 1779)
- `self.force_blob_path` → `self.flags.force_blob_path` (line 1524)
- `self.mmq_diag_quantize_only` → `self.flags.mmq_diag_quantize_only` (line 17091)

- [ ] **Step 5: Update external consumers of `gpu.mmq_screen`**

In `hipfire-runtime/examples/daemon.rs`, lines ~658–659 and ~1714:

```rust
// BEFORE:
gpu.mmq_screen = v;
gpu.mmq_screen_threshold = v as f32;
if gpu.mmq_screen && matches!(...

// AFTER:
// FeatureFlags is immutable via Arc, so daemon can't set it after init.
// The mmq_screen flag from JSON-RPC needs a different mechanism.
// For now, the daemon passes mmq_screen at Gpu::init time via env var.
// If JSON-RPC override is needed, we'll need a separate mechanism
// (e.g., wrapping the bool in an AtomicBool inside FeatureFlags).
```

**WAIT — this is a breaking change.** The daemon currently sets `gpu.mmq_screen` from JSON-RPC params. Since `FeatureFlags` is immutable, we need either:
1. `mmq_screen` stays as a separate `AtomicBool` on `Gpu` (separate from FeatureFlags)
2. FeatureFlags uses `AtomicBool` for `mmq_screen`

**Decision:** Use `Arc<FeatureFlags>` for env-var-derived flags. `mmq_screen` and `mmq_screen_threshold` are special — they can also be set at runtime from JSON-RPC. Keep `mmq_screen: bool` and `mmq_screen_threshold: f32` as fields on `Gpu`, initialized from `FeatureFlags` defaults but mutable by the daemon. Don't move these three fields.

**Revised Step 2:** Only remove `force_blob_path` and `mmq_diag_quantize_only` from `Gpu`. Keep `mmq_screen` and `mmq_screen_threshold` as mutable `Gpu` fields, but initialize them from `flags.mmq_screen` and `flags.mmq_screen_threshold`:

```rust
Ok(Self {
    flags,
    // ...
    mmq_screen: flags.mmq_screen,
    mmq_screen_threshold: flags.mmq_screen_threshold,
    // REMOVED: force_blob_path, mmq_diag_quantize_only (now on flags)
    capture_mode: false,
})
```

- [ ] **Step 6: Verify it compiles**

Run: `cargo check -p rdna-compute`

- [ ] **Step 7: Commit**

```bash
git add crates/rdna-compute/src/dispatch.rs
git commit -m "feat(rdna-compute): wire Arc<FeatureFlags> into Gpu, migrate force_blob_path and mmq_diag_quantize_only

mmq_screen and mmq_screen_threshold stay as mutable Gpu fields (daemon
can override via JSON-RPC). force_blob_path and mmq_diag_quantize_only
move to immutable FeatureFlags.

Assisted-by: OpenCode:anthropic/claude-sonnet-4-20250514"
```

---

## Task 3: Remove OnceLock free functions, replace with FeatureFlags methods

**Files:**
- Modify: `crates/rdna-compute/src/dispatch.rs` (remove lines 47–499)
- Modify: `crates/rdna-compute/src/lib.rs` (remove `pub use` of free functions)

- [ ] **Step 1: Remove all free functions (lines 47–499)**

Delete from `fn gemv_rows_override()` through `fn lm_head_wmma_disabled()`. These 22 functions are now methods on `FeatureFlags`. Keep `MMQ_CURRENT_LAYER` (line 94) since it's a static, not an env var.

Also remove `thread_local! { static LAST_BOUND_DEVICE ... }`? No — that's a thread-local cache for `bind_thread`, not an env var. Keep it.

So the actual range to delete is lines 47–499 (the free functions + their doc comments), preserving:
- The `thread_local!` block at line 40–44
- `MMQ_CURRENT_LAYER` at line 94

Wait, `MMQ_CURRENT_LAYER` is at line 94 which is *inside* the free function block. Let me be precise:

Lines to **delete**:
- 47–60: `fn gemv_rows_override()`
- 62–87: `pub fn gemv_dp4a_enabled()`
- 96–99: (comment block about weight prefetch, this is part of a doc comment for the next fn)
- 113–123: `fn gemv_prefetch_enabled()`
- 125–147: `fn gfx942_lds_gemv_enabled()`
- 149–173: `fn gemv_rows_default()`
- 175–190: `fn has_dot2_f32_f16()`
- 192–199: `fn hfq3_sdot4_gfx10_enabled()`
- 201–223: `fn hfq3_dp4a_enabled()` (the local one, not the method)
- 225–242: `fn hfq3_mmq_rdna2_enabled()`
- 244–266: `fn hfq4_mmq_rdna2_enabled()`
- 268–282: `pub fn has_wmma_f16()`
- 284–298: `fn has_wmma_f16_gfx12()`, `fn has_wmma_fp8_gfx12()`
- 300–318: FP8 constants and `fn is_fp8_wmma_enabled()`
- 329–344: `fn is_dot2_gemv_enabled()`
- 346–363: `fn is_gcn5_wave64()`
- 365–373: `fn has_wave64_native()`
- 375–390: `fn has_mmq_dp4a_or_wmma()`
- 392–448: `fn should_use_mmq()`
- 450–461: `fn mmq_env_override()`
- 463–471: `fn mmq_min_batch_override()`
- 473–481: `fn fp16_disabled()`
- 483–490: `fn wo_mmq_enabled()`
- 492–499: `fn lm_head_wmma_disabled()`

Keep: `MMQ_CURRENT_LINE` AtomicUsize (line 94) — move it right after the `thread_local!` block.

- [ ] **Step 2: Update `lib.rs` exports**

Change `crates/rdna-compute/src/lib.rs`:

```rust
pub use dispatch::{
    gen_fwht_signs, DType, Gpu, GpuTensor, LLOYD_MQ4_GROUP_BYTES, MMQ_CURRENT_LAYER,
};
pub use feature_flags::FeatureFlags;
```

Remove `gemv_dp4a_enabled` and `has_wmma_f16` from the `pub use` — they're now methods on `FeatureFlags`.

- [ ] **Step 3: Replace all call sites in dispatch.rs**

Every call to the old free functions inside `impl Gpu` methods needs to change:

| Old call | New call |
|---|---|
| `gemv_dp4a_enabled(&self.arch)` | `self.flags.gemv_dp4a_enabled()` |
| `has_wmma_f16(&self.arch)` | `self.flags.has_wmma_f16()` |
| `has_wmma_f16_gfx12(&self.arch)` | `self.flags.has_wmma_f16_gfx12()` |
| `fp16_disabled()` | `self.flags.fp16_disabled` |
| `wo_mmq_enabled()` | `self.flags.wo_mmq` |
| `lm_head_wmma_disabled()` | `self.flags.lm_head_wmma_disabled` |
| `is_fp8_wmma_enabled()` | `self.flags.fp8_wmma` |
| `is_dot2_gemv_enabled()` | `self.flags.dot2_gemv` |
| `is_gcn5_wave64(&self.arch)` | `self.flags.is_gcn5_wave64()` |
| `has_wave64_native(&self.arch)` | `self.flags.has_wave64_native()` |
| `has_mmq_dp4a_or_wmma(&self.arch)` | `self.flags.has_mmq_dp4a_or_wmma()` |
| `should_use_mmq(&self.arch, batch_size)` | `self.flags.should_use_mmq(batch_size)` |
| `gemv_rows_override()` | `self.flags.gemv_rows` |
| `gemv_rows_default(&self.arch)` | `self.flags.gemv_rows_default` |
| `hfq3_dp4a_enabled(&self.arch)` | `self.flags.hfq3_dp4a_enabled()` |
| `hfq3_mmq_rdna2_enabled()` | `self.flags.hfq3_mmq_rdna2_enabled()` |
| `hfq4_mmq_rdna2_enabled()` | `self.flags.hfq4_mmq_rdna2_enabled()` |
| `gemv_prefetch_enabled(&self.arch)` | `self.flags.gemv_prefetch_enabled()` |
| `gfx942_lds_gemv_enabled(&self.arch)` | `self.flags.gfx942_lds_gemv_enabled()` |

For functions called outside `impl Gpu` (e.g., free-standing or in other modules), pass a `&FeatureFlags` reference or use the `Gpu.flags` accessor.

- [ ] **Step 4: Update hipfire-arch-qwen35**

In `crates/hipfire-arch-qwen35/src/qwen35.rs`, replace:

```rust
// BEFORE:
rdna_compute::gemv_dp4a_enabled(&gpu.arch)
rdna_compute::has_wmma_f16(gpu.arch.as_str())

// AFTER:
gpu.flags.gemv_dp4a_enabled()
gpu.flags.has_wmma_f16()
```

There are 7 call sites for `gemv_dp4a_enabled` and 3 for `has_wmma_f16` in qwen35.rs.

In `crates/hipfire-arch-qwen35/src/paro_la_gates_codec.rs`, `arch_caps::paro_la_gates_mq4g128_default` is not an env var function — keep it as-is.

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p rdna-compute -p hipfire-arch-qwen35`

- [ ] **Step 6: Commit**

```bash
git add crates/rdna-compute/src/dispatch.rs crates/rdna-compute/src/lib.rs crates/hipfire-arch-qwen35/src/
git commit -m "refactor(rdna-compute): remove OnceLock free functions, wire FeatureFlags methods

All 22 env-var-reading free functions replaced by FeatureFlags methods.
gemv_dp4a_enabled and has_wmma_f16 no longer exported as free fns.
Callers in hipfire-arch-qwen35 updated to use gpu.flags.method().

Assisted-by: OpenCode:anthropic/claude-sonnet-4-20250514"
```

---

## Task 4: Migrate remaining raw `std::env::var` reads in dispatch.rs methods

**Files:**
- Modify: `crates/rdna-compute/src/dispatch.rs`

This is the biggest task mechanically. Every `std::env::var("HIPFIRE_X")` inside `impl Gpu` methods changes to `self.flags.x`. Use rg to find all remaining reads:

```
rg -n 'std::env::var\("HIPFIRE_' crates/rdna-compute/src/dispatch.rs
```

Each one becomes a field access on `self.flags`. The mapping is:

| Env var | Flags field |
|---|---|
| `HIPFIRE_MQ3_MB4` | `self.flags.mq3_mb4` |
| `HIPFIRE_LLOYD_MB4` | `self.flags.lloyd_mb4` |
| `HIPFIRE_MOE_GROUPED_I8` | `self.flags.moe_grouped_i8` |
| `HIPFIRE_MOE_GROUPED_I8_K8` | `self.flags.moe_grouped_i8_k8` |
| `HIPFIRE_MOE_GROUPED_I8_K4` | `self.flags.moe_grouped_i8_k4` |
| `HIPFIRE_MOE_GROUPED_I8_K4_GFX12` | `self.flags.moe_grouped_i8_k4_gfx12` |
| `HIPFIRE_MOE_GROUPED_M2` | `self.flags.moe_grouped_m2` |
| `HIPFIRE_MOE_HFQ6_V2` | `self.flags.moe_hfq6_v2` |
| `HIPFIRE_GFX942_GEMV_V2` | `self.flags.gfx942_gemv_v2` |
| `HIPFIRE_GFX942_GEMV_V3` | `self.flags.gfx942_gemv_v3` |
| `HIPFIRE_GFX942_RMSNORM_SPLIT` | `self.flags.gfx942_rmsnorm_split` |
| `HIPFIRE_GFX942_MFMA_PREFILL` | `self.flags.gfx942_mfma_prefill` |
| `HIPFIRE_HFQ4G128_MMQ` | `self.flags.hfq4g128_mmq` |
| `HIPFIRE_GATE_UP_VARIANT` | `self.flags.gate_up_variant` |
| `HIPFIRE_DETERMINISTIC` | `self.flags.deterministic` |
| `HIPFIRE_GEMM_DUMP` | `self.flags.gemm_dump` |
| `HIPFIRE_MW16` | `self.flags.mw16` |
| `HIPFIRE_Q8_BATCHED_LEGACY` | `self.flags.q8_batched_legacy` |
| `HIPFIRE_ROPE_INTERLEAVED_LEGACY` | `self.flags.rope_interleaved_legacy` |
| `HIPFIRE_ROCBLAS_ALL_ARCHS` | `self.flags.rocblas_all_archs` |
| `HIPFIRE_ROCBLAS_OFF` | `self.flags.rocblas_off` |
| `HIPFIRE_ROCBLAS_MIN_BATCH` | `self.flags.rocblas_min_batch` |
| `HIPFIRE_WO_WMMA_VARIANT` | `self.flags.wo_wmma_variant` |
| `HIPFIRE_PARO_PACK1/PACK2/PACK4` | `self.flags.lloyd_mb4` (or `mq3_mb4`) |
| `HIPFIRE_PARO_SHARED_PAIRS` / `HIPFIRE_PARO_FUSED_PACK2` | need dedicated fields |

**Note on PARO vars:** The `HIPFIRE_PARO_PACK1/PACK2/PACK4` and `HIPFIRE_PARO_SHARED_PAIRS` / `HIPFIRE_PARO_FUSED_PACK2` reads are in the Lloyd MB4 dispatch arms. These map to `Mb4Mode` or need separate `bool` fields. Add:

```rust
pub paro_shared_pairs: bool,
pub paro_fused_pack2: bool,
```

to `FeatureFlags`, constructed from `HIPFIRE_PARO_SHARED_PAIRS` and `HIPFIRE_PARO_FUSED_PACK2`.

- [ ] **Step 1:** Add `paro_shared_pairs` and `paro_fused_pack2` fields to `FeatureFlags` and `from_env()`

- [ ] **Step 2:** Replace all `std::env::var("HIPFIRE_X")` in `impl Gpu` methods with `self.flags.x` (mechanical find-and-replace, ~40 call sites)

- [ ] **Step 3:** Verify compilation: `cargo check -p rdna-compute`

- [ ] **Step 4:** Commit

```bash
git add crates/rdna-compute/src/feature_flags.rs crates/rdna-compute/src/dispatch.rs
git commit -m "refactor(rdna-compute): migrate all remaining env reads to FeatureFlags

No more std::env::var calls in dispatch.rs impl Gpu methods. All feature
flags read once at init via FeatureFlags::from_env().

Assisted-by: OpenCode:anthropic/claude-sonnet-4-20250514"
```

---

## Task 5: Migrate `kernels.rs` and `compiler.rs` env reads

**Files:**
- Modify: `crates/rdna-compute/src/kernels.rs`
- Modify: `crates/rdna-compute/src/compiler.rs`
- Modify: `crates/rdna-compute/src/feature_flags.rs` (ensure `lloyd_force_baseline`, `rdna2_variant`, `hipcc_extra_flags` are there)

- [ ] **Step 1:** `kernels.rs` has `HIPFIRE_LLOYD_FORCE_BASELINE` (10 reads) and `HIPFIRE_RDNA2_VARIANT` (1 read). These are called from `Gpu` methods that have `&self`. Pass `&self.flags` or access the field directly.

The kernel-selection functions in `kernels.rs` are `pub const` strings or `pub fn` that take `arch: &str`. They don't take `&FeatureFlags`. Two approaches:

**Approach A (simpler):** The `HIPFIRE_LLOYD_FORCE_BASELINE` reads are in `*_for_arch()` functions that are called from dispatch. Pass the `bool` as a parameter:

```rust
// BEFORE:
pub fn gemm_hfq4g256_residual_lloyd_mb4_for_arch(arch: &str) -> &'static str { ... }

// AFTER:
pub fn gemm_hfq4g256_residual_lloyd_mb4_for_arch(arch: &str, force_baseline: bool) -> &'static str { ... }
```

And in dispatch.rs:
```rust
kernels::gemm_hfq4g256_residual_lloyd_mb4_for_arch(&self.arch, self.flags.lloyd_force_baseline)
```

**Approach B (full refactor):** Change all `*_for_arch()` functions to take `&FeatureFlags`. This is task #3 territory (centralize arch routing). Don't do it here.

Choose **Approach A** — add `bool` or `Option<u32>` params to the `*_for_arch()` functions that need env var info.

- [ ] **Step 2:** Replace `std::env::var("HIPFIRE_LLOYD_FORCE_BASELINE")` in `kernels.rs` with the passed `force_baseline: bool` param.

- [ ] **Step 3:** Replace `std::env::var("HIPFIRE_RDNA2_VARIANT")` in `kernels.rs` with `variant: Option<u32>` param.

- [ ] **Step 4:** Replace `std::env::var("HIPFIRE_HIPCC_EXTRA_FLAGS")` in `compiler.rs` with a field from the `KernelCompiler` struct. Since `KernelCompiler` already exists and is constructed in `Gpu::init()`, pass the extra flags through it:

```rust
// In compiler.rs:
pub struct KernelCompiler {
    // existing fields...
    pub extra_flags: String,  // was HIPFIRE_HIPCC_EXTRA_FLAGS
}
```

And in `Gpu::init()`, pass `flags.hipcc_extra_flags.clone()` to `KernelCompiler::new()`.

- [ ] **Step 5:** Verify compilation: `cargo check -p rdna-compute`

- [ ] **Step 6:** Commit

```bash
git add crates/rdna-compute/src/
git commit -m "refactor(rdna-compute): migrate kernels.rs and compiler.rs env reads to FeatureFlags

kernels.rs: HIPFIRE_LLOYD_FORCE_BASELINE and HIPFIRE_RDNA2_VARIANT now
passed as parameters. compiler.rs: HIPFIRE_HIPCC_EXTRA_FLAGS now a
KernelCompiler field. No more std::env::var in these files.

Assisted-by: OpenCode:anthropic/claude-sonnet-4-20250514"
```

---

## Task 6: Create `RuntimeConfig` in hipfire-runtime

**Files:**
- Create: `crates/hipfire-runtime/src/config.rs`
- Modify: `crates/hipfire-runtime/src/lib.rs`
- Modify: `crates/hipfire-runtime/src/tokenizer.rs`
- Modify: `crates/hipfire-runtime/src/dflash.rs`
- Modify: `crates/hipfire-runtime/src/llama.rs`
- Modify: `crates/hipfire-runtime/src/loop_guard.rs`
- Modify: `crates/hipfire-runtime/src/multi_gpu.rs`
- Modify: `crates/hipfire-runtime/src/arch.rs`

- [ ] **Step 1:** Create `config.rs` with `RuntimeConfig` struct and `from_env()` constructor, covering all env vars found in the runtime crate sources (see audit above — ~30 env vars across 6 source files).

The struct should mirror the design from the spec: all fields immutable after construction, `from_env()` reads everything once.

- [ ] **Step 2:** Add `pub mod config;` to `hipfire-runtime/src/lib.rs`

- [ ] **Step 3:** Migrate each source file:
  - `tokenizer.rs`: `HIPFIRE_NORMALIZE_PROMPT`, `HIPFIRE_PROMPT_TOKEN_HEAT`, `HIPFIRE_PROMPT_HEAT_JSON`, `HIPFIRE_PROMPT_HEAT_LIMIT`
  - `dflash.rs`: `HIPFIRE_DRAFT_F16`, `HIPFIRE_DRAFT_GEMM_DUMP`, `HIPFIRE_DRAFT_SUBPHASE`
  - `llama.rs`: `HIPFIRE_PARO_SMALL_DIRECT`, `HIPFIRE_PARO_PREROTATE`, `HIPFIRE_PARO_FUSE_RMSNORM`, `HIPFIRE_PARO_SWIGLU_FUSED`, `HIPFIRE_FLASH_PARTIALS_BATCH`, `HIPFIRE_PREFILL_BATCHED`
  - `loop_guard.rs`: `HIPFIRE_NGRAM_LOOP_THRESHOLD`, `HIPFIRE_NGRAM_WINDOW`
  - `multi_gpu.rs`: `HIPFIRE_DEVICES`, `HIPFIRE_ALLOW_MIXED_ARCH`, `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB`
  - `arch.rs`: `HIPFIRE_LM_HEAD_F16`

Each source file takes `&RuntimeConfig` as a parameter (or the config is threaded through existing constructor functions).

- [ ] **Step 4:** Verify compilation: `cargo check -p hipfire-runtime`

- [ ] **Step 5:** Commit

```bash
git add crates/hipfire-runtime/src/
git commit -m "feat(hipfire-runtime): add RuntimeConfig struct, consolidate env reads

All HIPFIRE_ env reads in tokenizer, dflash, llama, loop_guard,
multi_gpu, and arch now go through RuntimeConfig::from_env().
No more per-call std::env::var in runtime source files.

Assisted-by: OpenCode:anthropic/claude-sonnet-4-20250514"
```

---

## Task 7: Update `hipfire-runtime` examples to construct `RuntimeConfig`

**Files:**
- Modify: `crates/hipfire-runtime/examples/daemon.rs`
- Modify: `crates/hipfire-runtime/examples/dflash_spec_demo.rs`
- Modify: (other examples as needed)

- [ ] **Step 1:** In `daemon.rs`, construct `RuntimeConfig::from_env()` at startup and pass it through to the functions that need it. The daemon is the main entry point — it needs to pass the config down.

- [ ] **Step 2:** In `dflash_spec_demo.rs`, same pattern — construct `RuntimeConfig::from_env()` and thread it through.

- [ ] **Step 3:** Verify compilation: `cargo check -p hipfire-runtime --examples`

- [ ] **Step 4:** Commit

```bash
git add crates/hipfire-runtime/examples/
git commit -m "refactor(hipfire-runtime): update examples to use RuntimeConfig

daemon.rs and dflash_spec_demo.rs construct RuntimeConfig at startup
and pass it through instead of reading env vars inline.

Assisted-by: OpenCode:anthropic/claude-sonnet-4-20250514"
```

---

## Task 8: Update remaining examples that use `HIPFIRE_` env vars directly

**Files:**
- All `crates/hipfire-runtime/examples/*.rs` files that read HIPFIRE_ vars
- All `crates/rdna-compute/examples/*.rs` files that read HIPFIRE_ vars

- [ ] **Step 1:** Audit remaining examples with `rg -l 'HIPFIRE_' crates/*/examples/ --type rust`

- [ ] **Step 2:** For each example, either:
  a. Construct a `RuntimeConfig` / `FeatureFlags` at startup (for examples that construct a `Gpu`)
  b. Leave as-is if the example is a standalone test that deliberately reads env vars (e.g., `channel_test_mmq.rs` which sets `mmq_screen` from CLI flags, not env vars)

- [ ] **Step 3:** Verify compilation: `cargo check --workspace`

- [ ] **Step 4:** Commit

```bash
git add crates/hipfire-runtime/examples/ crates/rdna-compute/examples/
git commit -m "refactor: update remaining examples to use FeatureFlags/RuntimeConfig

Assisted-by: OpenCode:anthropic/claude-sonnet-4-20250514"
```

---

## Task 9: Full workspace compile check + coherence gate

**Files:** None (verification only)

- [ ] **Step 1:** `cargo check --workspace` — must pass with zero errors

- [ ] **Step 2:** `cargo build --release --features deltanet -p hipfire-runtime --example daemon` — must pass

- [ ] **Step 3:** Run `./scripts/coherence-gate-dflash.sh` (if GPU available) to verify no behavioral regression

- [ ] **Step 4:** Commit any fixups

---

## Task 10: Final cleanup — remove unused imports and verify no `std::env::var("HIPFIRE_")` remains in rdna-compute/src/

- [ ] **Step 1:** `rg -n 'std::env::var\("HIPFIRE_' crates/rdna-compute/src/` — should return zero results

- [ ] **Step 2:** `cargo clippy -p rdna-compute -- -D warnings` — clean up any dead code warnings

- [ ] **Step 3:** Commit

```bash
git add -A
git commit -m "chore: final cleanup, remove unused imports, zero env reads in rdna-compute/src

Assisted-by: OpenCode:anthropic/claude-sonnet-4-20250514"
```