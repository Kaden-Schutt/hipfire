# Arch Routing Centralization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize all GPU arch routing in `rdna-compute` into a single `ArchCaps` struct, eliminating 32 `*_for_arch()` panic bombs and ~72 scattered arch checks across `dispatch.rs` and `kernels.rs`.

**Architecture:** New `ArchCaps` struct in `arch_caps.rs` holds all arch-family predicates (computed once at construction) and kernel-symbol selection methods (replacing `*_for_arch`). `FeatureFlags` keeps env-var config but delegates arch predicates to `ArchCaps`. `Gpu` holds `arch_caps: ArchCaps` alongside `flags: Arc<FeatureFlags>`.

**Tech Stack:** Rust, `crates/rdna-compute`, no new dependencies.

**Base branch:** `feature/328-feature-flags-v3`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/rdna-compute/src/arch_caps.rs` | Modify (expand from 44 → ~550 lines) | `ArchCaps` struct: all arch-family predicates + kernel symbol selection methods |
| `crates/rdna-compute/src/dispatch.rs` | Modify (net -~250 lines Phase 1, ~35 changed Phase 2) | Replace all inline arch checks with `self.arch_caps.*` calls; replace `*_for_arch()` calls with `self.arch_caps.*` |
| `crates/rdna-compute/src/kernels.rs` | Modify (Phase 2: remove 32 `*_for_arch` functions; reorganize const strings) | Pure `pub const` symbol definitions; no routing logic |
| `crates/rdna-compute/src/feature_flags.rs` | Modify (Phase 1: move arch predicates out) | Keep only env-var config; delegate arch queries to `ArchCaps` |
| `crates/rdna-compute/src/lib.rs` | Modify (add `mod arch_caps` if not present) | Module declaration |
| `crates/rdna-compute/src/profiler.rs` | Modify (1 inline arch check) | Replace `arch.starts_with(...)` with `arch_caps.is_rdna_wave32()` |

---

## Phase 1: ArchCaps predicates + dispatch.rs inline migration

### Task 1: Create `ArchCaps` struct with family predicates

**Files:**
- Create/modify: `crates/rdna-compute/src/arch_caps.rs`
- Modify: `crates/rdna-compute/src/lib.rs`

- [ ] **Step 1: Add `mod arch_caps` to `lib.rs` if not present**

Check if `mod arch_caps;` exists in `crates/rdna-compute/src/lib.rs`. If not, add it. On the feature branch, it should already exist since `arch_caps.rs` is already a file.

- [ ] **Step 2: Write the `ArchCaps` struct definition**

Replace the entire `arch_caps.rs` file. Keep the existing `paro_la_gates_mq4g128_default` function and test module. Add the struct:

```rust
use crate::feature_flags::FeatureFlags;

/// Single source of truth for GPU architecture family membership and
/// hardware capability predicates. Computed once at `Gpu::init()` time;
/// immutable thereafter.
///
/// Named predicates replace the 13+ repeated arch-set literals that were
/// scattered across `dispatch.rs` and `kernels.rs`. Adding a new arch
/// (e.g. gfx1300) requires touching this struct's `new()` constructor and
/// exactly zero match arms elsewhere.
pub struct ArchCaps {
    arch: String,

    // ── Family membership ────────────────────────────────────────
    // gfx1100, gfx1101, gfx1102 (RDNA3 discrete GPUs, Navi 31/32/33)
    is_rdna3_dgpu: bool,
    // above + gfx1151 (RDNA3 dGPU + Strix Halo iGPU — MQ4 Lloyd fast-path)
    is_rdna3_dgpu_1151: bool,
    // above + gfx1150 (RDNA3 full family — MQ3 WMMA + HFP4 fast-path)
    is_rdna3_full: bool,
    // gfx1151 only (Strix Halo APU dGPU-class iGPU)
    is_strix_halo: bool,
    // gfx1150, gfx1151, gfx1152 (Strix Halo / Strix Point iGPU family)
    is_strix_halo_igpu: bool,
    // gfx1200, gfx1201 (RDNA4)
    is_rdna4: bool,
    // gfx940, gfx941, gfx942 (CDNA3 — MI300X/A/A8, MI325X, MI355X)
    is_cdna3: bool,
    // gfx1030, gfx1031 (RDNA2 dGPU — Navi 21/22)
    is_rdna2: bool,
    // gfx906 only (CDNA1 — MI50)
    is_gfx906: bool,
    // gfx908 only (CDNA1 — MI100)
    is_gfx908: bool,
    // gfx906, gfx908, gfx940, gfx941, gfx942 (native wave64 GPUs)
    is_wave64_native: bool,

    // ── Capability predicates (derived from family + FeatureFlags) ──
    has_wmma_f16: bool,
    has_wmma_f16_gfx12: bool,
    has_wmma_fp8_gfx12: bool,
    has_dot2_f32_f16: bool,
    has_mmq_dp4a_or_wmma: bool,
    is_gcn5_wave64: bool,
    should_use_mmq_cache: bool,
    gemv_dp4a: bool,
    gemv_prefetch: bool,
    gemv_rows_default: u32,
    hfq3_sdot4_gfx10: bool,
    hfq3_dp4a: bool,
    hfq3_mmq_rdna2: bool,
    hfq4_mmq_rdna2: bool,
    is_rdna_wave32: bool,
    gfx942_lds_gemv: bool,

    // ── Reference to FeatureFlags for env-var overrides ────────
    flags: std::sync::Arc<FeatureFlags>,
}
```

- [ ] **Step 3: Write `ArchCaps::new()` constructor**

```rust
impl ArchCaps {
    pub fn new(arch: &str, flags: std::sync::Arc<FeatureFlags>) -> Self {
        let is_rdna3_dgpu = matches!(arch, "gfx1100" | "gfx1101" | "gfx1102");
        let is_rdna3_dgpu_1151 = is_rdna3_dgpu || arch == "gfx1151";
        let is_rdna3_full = is_rdna3_dgpu_1151 || arch == "gfx1150";
        let is_strix_halo = arch == "gfx1151";
        let is_strix_halo_igpu = matches!(arch, "gfx1150" | "gfx1151" | "gfx1152");
        let is_rdna4 = matches!(arch, "gfx1200" | "gfx1201");
        let is_cdna3 = matches!(arch, "gfx940" | "gfx941" | "gfx942");
        let is_rdna2 = matches!(arch, "gfx1030" | "gfx1031");
        let is_gfx906 = arch == "gfx906";
        let is_gfx908 = arch == "gfx908";
        let is_wave64_native = matches!(arch, "gfx906" | "gfx908" | "gfx940" | "gfx941" | "gfx942");

        let has_wmma_f16 = arch.starts_with("gfx11");
        let has_wmma_f16_gfx12 = arch.starts_with("gfx12");
        let has_wmma_fp8_gfx12 = arch.starts_with("gfx12");
        let has_dot2_f32_f16 = matches!(arch,
            "gfx1011" | "gfx1012"
            | "gfx1030" | "gfx1031" | "gfx1032"
            | "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103"
            | "gfx1150" | "gfx1151" | "gfx1152"
            | "gfx1200" | "gfx1201"
        );
        let has_mmq_dp4a_or_wmma = matches!(arch,
            "gfx906"
            | "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103"
            | "gfx1150" | "gfx1151" | "gfx1152"
        );

        // Derived from family + FeatureFlags env overrides
        let is_gcn5_wave64 = is_gfx906
            || (is_gfx908 && flags.gcn5_wave64_hybrid.unwrap_or(false));
        let dp4a_default = is_gfx906;
        let gemv_dp4a = flags.gemv_dp4a.unwrap_or(dp4a_default);
        let gemv_prefetch = flags.gemv_prefetch.unwrap_or(is_gfx906);
        let gemv_rows_default = flags.gemv_rows.unwrap_or_else(|| {
            match arch {
                "gfx1100" | "gfx1101" | "gfx1102"
                | "gfx1030" | "gfx1031"
                | "gfx906" | "gfx908" | "gfx940" | "gfx941" | "gfx942" => 1,
                _ => 2,
            }
        });
        let hfq3_sdot4_gfx10 = matches!(arch,
            "gfx1011" | "gfx1012" | "gfx1030" | "gfx1031" | "gfx1032"
        );
        let hfq3_dp4a = flags.hfq3_dp4a.unwrap_or(false) && hfq3_sdot4_gfx10;
        let hfq3_mmq_rdna2 = flags.hfq3_mmq.unwrap_or(false) && hfq3_sdot4_gfx10;
        let hfq4_mmq_rdna2 = flags.hfq4_mmq.unwrap_or(false) && has_dot2_f32_f16;
        let should_use_mmq = |batch_size: usize| -> bool {
            if !has_mmq_dp4a_or_wmma { return false; }
            match flags.mmq_override {
                Some(false) => false,
                Some(true) => true,
                None => {
                    let arch_min_batch: usize = if is_gfx906 { 8 } else { 256 };
                    let min_batch = flags.mmq_min_batch.unwrap_or(arch_min_batch);
                    batch_size >= min_batch
                }
            }
        };
        let is_rdna_wave32 = arch.starts_with("gfx10")
            || arch.starts_with("gfx11")
            || arch.starts_with("gfx12");
        let gfx942_lds_gemv = flags.gfx942_lds_gemv.unwrap_or(false);

        Self {
            arch: arch.to_string(),
            is_rdna3_dgpu,
            is_rdna3_dgpu_1151,
            is_rdna3_full,
            is_strix_halo,
            is_strix_halo_igpu,
            is_rdna4,
            is_cdna3,
            is_rdna2,
            is_gfx906,
            is_gfx908,
            is_wave64_native,
            has_wmma_f16,
            has_wmma_f16_gfx12,
            has_wmma_fp8_gfx12,
            has_dot2_f32_f16,
            has_mmq_dp4a_or_wmma,
            is_gcn5_wave64,
            should_use_mmq_cache: false, // computed per-call via method
            gemv_dp4a,
            gemv_prefetch,
            gemv_rows_default,
            hfq3_sdot4_gfx10,
            hfq3_dp4a,
            hfq3_mmq_rdna2,
            hfq4_mmq_rdna2,
            is_rdna_wave32,
            gfx942_lds_gemv,
            flags,
        }
    }
```

Note: `should_use_mmq` can't be cached as a bool because it takes `batch_size`. It becomes a method instead:

```rust
    pub fn should_use_mmq(&self, batch_size: usize) -> bool {
        if !self.has_mmq_dp4a_or_wmma { return false; }
        match self.flags.mmq_override {
            Some(false) => false,
            Some(true) => true,
            None => {
                let arch_min_batch: usize = if self.is_gfx906 { 8 } else { 256 };
                let min_batch = self.flags.mmq_min_batch.unwrap_or(arch_min_batch);
                batch_size >= min_batch
            }
        }
    }
```

- [ ] **Step 4: Write accessor methods for all predicates**

Add public accessor methods (one per field):

```rust
    // ── Family membership ────────────────────────────────────────
    pub fn is_rdna3_dgpu(&self) -> bool { self.is_rdna3_dgpu }
    pub fn is_rdna3_dgpu_1151(&self) -> bool { self.is_rdna3_dgpu_1151 }
    pub fn is_rdna3_full(&self) -> bool { self.is_rdna3_full }
    pub fn is_strix_halo(&self) -> bool { self.is_strix_halo }
    pub fn is_strix_halo_igpu(&self) -> bool { self.is_strix_halo_igpu }
    pub fn is_rdna4(&self) -> bool { self.is_rdna4 }
    pub fn is_cdna3(&self) -> bool { self.is_cdna3 }
    pub fn is_rdna2(&self) -> bool { self.is_rdna2 }
    pub fn is_gfx906(&self) -> bool { self.is_gfx906 }
    pub fn is_gfx908(&self) -> bool { self.is_gfx908 }
    pub fn is_wave64_native(&self) -> bool { self.is_wave64_native }

    // ── Capability predicates ────────────────────────────────────
    pub fn has_wmma_f16(&self) -> bool { self.has_wmma_f16 }
    pub fn has_wmma_f16_gfx12(&self) -> bool { self.has_wmma_f16_gfx12 }
    pub fn has_wmma_fp8_gfx12(&self) -> bool { self.has_wmma_fp8_gfx12 }
    pub fn has_dot2_f32_f16(&self) -> bool { self.has_dot2_f32_f16 }
    pub fn has_mmq_dp4a_or_wmma(&self) -> bool { self.has_mmq_dp4a_or_wmma }
    pub fn is_gcn5_wave64(&self) -> bool { self.is_gcn5_wave64 }
    pub fn gemv_dp4a(&self) -> bool { self.gemv_dp4a }
    pub fn gemv_prefetch(&self) -> bool { self.gemv_prefetch }
    pub fn gemv_rows_default(&self) -> u32 { self.gemv_rows_default }
    pub fn hfq3_sdot4_gfx10(&self) -> bool { self.hfq3_sdot4_gfx10 }
    pub fn hfq3_dp4a(&self) -> bool { self.hfq3_dp4a }
    pub fn hfq3_mmq_rdna2(&self) -> bool { self.hfq3_mmq_rdna2 }
    pub fn hfq4_mmq_rdna2(&self) -> bool { self.hfq4_mmq_rdna2 }
    pub fn is_rdna_wave32(&self) -> bool { self.is_rdna_wave32 }
    pub fn gfx942_lds_gemv(&self) -> bool { self.gfx942_lds_gemv }
    pub fn arch(&self) -> &str { &self.arch }
```

- [ ] **Step 5: Run `cargo check -p rdna-compute`**

Expected: Passes (ArchCaps exists, compile-checks, but not yet wired into Gpu).

- [ ] **Step 6: Commit Phase 1 Task 1**

```
feat(rdna-compute): add ArchCaps struct with family predicates
```

---

### Task 2: Wire `ArchCaps` into `Gpu` and remove standalone helper functions from dispatch.rs

**Files:**
- Modify: `crates/rdna-compute/src/dispatch.rs`
- Modify: `crates/rdna-compute/src/feature_flags.rs` (remove arch predicates that move to ArchCaps)

- [ ] **Step 1: Add `arch_caps: ArchCaps` field to `Gpu` struct**

In `dispatch.rs`, add `arch_caps: crate::arch_caps::ArchCaps` field to the `Gpu` struct, next to `flags: Arc<FeatureFlags>`.

- [ ] **Step 2: Construct `ArchCaps` in `Gpu::init_with_device()`**

After the `FeatureFlags::from_env(&arch)` call, add:
```rust
let arch_caps = crate::arch_caps::ArchCaps::new(&arch, flags.clone());
```
And add `arch_caps` to the `Gpu { ... }` struct literal.

- [ ] **Step 3: Replace all 28 standalone helper functions' call sites**

Each call site that was `has_wmma_f16(&arch)` (and friends on the feature branch: `self.flags.has_wmma_f16()`) changes to `self.arch_caps.has_wmma_f16()`. This is a mechanical find-and-replace across dispatch.rs.

Mapping of old calls → new calls:

| Old call (feature branch) | New call |
|---|---|
| `self.flags.has_wmma_f16()` | `self.arch_caps.has_wmma_f16()` |
| `self.flags.has_wmma_f16_gfx12()` | `self.arch_caps.has_wmma_f16_gfx12()` |
| `self.flags.has_wmma_fp8_gfx12()` | `self.arch_caps.has_wmma_fp8_gfx12()` |
| `self.flags.has_dot2_f32_f16()` | `self.arch_caps.has_dot2_f32_f16()` |
| `self.flags.has_mmq_dp4a_or_wmma()` | `self.arch_caps.has_mmq_dp4a_or_wmma()` |
| `self.flags.is_gcn5_wave64()` | `self.arch_caps.is_gcn5_wave64()` |
| `self.flags.has_wave64_native()` | `self.arch_caps.has_wave64_native()` |
| `self.flags.should_use_mmq(batch_size)` | `self.arch_caps.should_use_mmq(batch_size)` |
| `self.flags.gemv_dp4a_enabled()` | `self.arch_caps.gemv_dp4a()` |
| `self.flags.gemv_prefetch_enabled()` | `self.arch_caps.gemv_prefetch()` |
| `self.flags.gemv_rows_default` | `self.arch_caps.gemv_rows_default()` |
| `self.flags.hfq3_sdot4_gfx10_enabled()` | `self.arch_caps.hfq3_sdot4_gfx10()` |
| `self.flags.hfq3_dp4a_enabled()` | `self.arch_caps.hfq3_dp4a()` |
| `self.flags.hfq3_mmq_rdna2_enabled()` | `self.arch_caps.hfq3_mmq_rdna2()` |
| `self.flags.hfq4_mmq_rdna2_enabled()` | `self.arch_caps.hfq4_mmq_rdna2()` |
| `self.flags.gfx942_lds_gemv_enabled()` | `self.arch_caps.gfx942_lds_gemv()` |

- [ ] **Step 4: Replace all inline `matches!(self.arch.as_str(), ...)` and `self.arch.starts_with(...)` in dispatch.rs**

The ~48 inline arch checks in dispatch.rs get replaced with `self.arch_caps.*()` calls. Complete mapping:

| Old inline pattern | New call |
|---|---|
| `matches!(self.arch.as_str(), "gfx1100"\|"gfx1101"\|"gfx1102")` (appears with or without `"gfx1151"`/`"gfx1150"`) | `self.arch_caps.is_rdna3_dgpu()` / `is_rdna3_dgpu_1151()` / `is_rdna3_full()` depending on the exact set |
| `matches!(self.arch.as_str(), "gfx940"\|"gfx941"\|"gfx942")` | `self.arch_caps.is_cdna3()` |
| `matches!(self.arch.as_str(), "gfx1150"\|"gfx1151"\|"gfx1152")` | `self.arch_caps.is_strix_halo_igpu()` |
| `self.arch.starts_with("gfx11")` | `self.arch_caps.has_wmma_f16()` |
| `self.arch.starts_with("gfx12")` | `self.arch_caps.is_rdna4()` |
| `self.arch == "gfx906"` | `self.arch_caps.is_gfx906()` |
| `matches!(self.arch.as_str(), "gfx1030"\|"gfx1031")` | `self.arch_caps.is_rdna2()` |

Note: Some inline checks don't map exactly to a single predicate. For example, `matches!(self.arch.as_str(), "gfx1100"\|"gfx1101"\|"gfx1102"\|"gfx1150"\|"gfx1151")` at the MQ3 mb4 sites maps to `self.arch_caps.is_rdna3_full()`.

- [ ] **Step 5: Remove arch predicates from `FeatureFlags`**

On the feature branch, these methods exist on `FeatureFlags`. Remove them and have `FeatureFlags` delegate to `ArchCaps` where needed, or simply remove them if all call sites now use `self.arch_caps.*()` directly.

The methods to remove from `FeatureFlags`:
- `has_wmma_f16()`
- `has_wmma_f16_gfx12()`
- `has_wmma_fp8_gfx12()`
- `has_dot2_f32_f16()`
- `hfq3_sdot4_gfx10_enabled()`
- `hfq3_dp4a_enabled()`
- `hfq3_mmq_rdna2_enabled()`
- `hfq4_mmq_rdna2_enabled()`
- `has_mmq_dp4a_or_wmma()`
- `has_wave64_native()`
- `is_gcn5_wave64()`
- `should_use_mmq(batch_size)`

Keep env-var fields on FeatureFlags (they're still needed by ArchCaps::new for the override logic).

- [ ] **Step 6: Update `profiler.rs`**

Replace `arch.starts_with("gfx10") || arch.starts_with("gfx11") || arch.starts_with("gfx12")` with `arch_caps.is_rdna_wave32()` once `profiler.rs` has access to ArchCaps (may need to pass `&ArchCaps` as parameter).

- [ ] **Step 7: Run `cargo check -p rdna-compute`**

Expected: Passes with zero errors.

- [ ] **Step 8: Run `cargo test -p rdna-compute`**

Expected: All existing tests pass.

- [ ] **Step 9: Commit Phase 1 Task 2**

```
refactor(rdna-compute): wire ArchCaps into Gpu, remove standalone arch helpers

Replace all inline arch checks and FeatureFlags arch predicates with
ArchCaps method calls. FeatureFlags retains env-var config only.
```

---

### Task 3: Add `ArchCaps` unit tests

**Files:**
- Modify: `crates/rdna-compute/src/arch_caps.rs` (test module)

- [ ] **Step 1: Write tests verifying family predicates for all known archs**

Add a test module to `arch_caps.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn default_flags() -> Arc<FeatureFlags> {
        Arc::new(FeatureFlags::from_env_for_test("gfx1100"))
    }

    fn make_caps(arch: &str) -> ArchCaps {
        ArchCaps::new(arch, default_flags())
    }

    #[test]
    fn rdna3_dgpu() {
        let caps = make_caps("gfx1100");
        assert!(caps.is_rdna3_dgpu());
        assert!(caps.is_rdna3_dgpu_1151());
        assert!(caps.is_rdna3_full());
        assert!(!caps.is_strix_halo());
        assert!(!caps.is_rdna4());
        assert!(!caps.is_cdna3());
    }

    #[test]
    fn strix_halo_1151() {
        let caps = make_caps("gfx1151");
        assert!(!caps.is_rdna3_dgpu());
        assert!(caps.is_rdna3_dgpu_1151());
        assert!(caps.is_rdna3_full());
        assert!(caps.is_strix_halo());
        assert!(caps.is_strix_halo_igpu());
    }

    #[test]
    fn strix_point_1150() {
        let caps = make_caps("gfx1150");
        assert!(!caps.is_rdna3_dgpu());
        assert!(!caps.is_rdna3_dgpu_1151());
        assert!(caps.is_rdna3_full()); // gfx1150 IS in the full family
        assert!(!caps.is_strix_halo());
        assert!(caps.is_strix_halo_igpu());
    }

    #[test]
    fn rdna4() {
        let caps = make_caps("gfx1200");
        assert!(caps.is_rdna4());
        assert!(caps.has_wmma_f16_gfx12());
        assert!(caps.has_wmma_fp8_gfx12());
        assert!(!caps.is_rdna3_dgpu());
    }

    #[test]
    fn cdna3_942() {
        let caps = make_caps("gfx942");
        assert!(caps.is_cdna3());
        assert!(caps.is_wave64_native());
        assert!(!caps.is_rdna3_dgpu());
    }

    #[test]
    fn gfx906_wave64() {
        let caps = make_caps("gfx906");
        assert!(caps.is_gfx906());
        assert!(caps.is_gcn5_wave64());
        assert!(caps.is_wave64_native());
        assert!(caps.has_mmq_dp4a_or_wmma());
    }

    #[test]
    fn rdna2() {
        let caps = make_caps("gfx1030");
        assert!(caps.is_rdna2());
        assert!(caps.has_dot2_f32_f16());
    }

    #[test]
    fn gfx1152_strix_point_igpu() {
        let caps = make_caps("gfx1152");
        assert!(caps.is_strix_halo_igpu());
        assert!(!caps.is_strix_halo());
        assert!(!caps.is_rdna3_dgpu_1151());
        assert!(caps.has_dot2_f32_f16());
    }

    #[test]
    fn dot2_coverage() {
        // All RDNA3+ archs have dot2
        for arch in &["gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1152", "gfx1200", "gfx1201"] {
            assert!(make_caps(arch).has_dot2_f32_f16(), "dot2 missing for {arch}");
        }
        // GCN5 does NOT have dot2
        assert!(!make_caps("gfx906").has_dot2_f32_f16());
    }
}
```

Note: `FeatureFlags::from_env_for_test()` may need to be added as a test-only constructor that reads no env vars and uses defaults. If it doesn't exist, create it as `pub fn from_env_for_test(arch: &str) -> Self { Self::from_env(arch) }` or similar, ensuring test isolation.

- [ ] **Step 2: Run `cargo test -p rdna-compute -- arch_caps`**

Expected: All 9 tests pass.

- [ ] **Step 3: Commit**

```
test(rdna-compute): add ArchCaps family predicate tests
```

---

### Task 4: Phase 1 validation — cargo check + coherence-gate

**Files:** None (validation only)

- [ ] **Step 1: Run `cargo check -p rdna-compute`**

Expected: Clean compilation, zero warnings about unused arch predicates.

- [ ] **Step 2: Run `cargo test -p rdna-compute`**

Expected: All tests pass.

- [ ] **Step 3: Run `./scripts/coherence-gate-dflash.sh`**

Expected: All 4 coherence-gate tests pass (per AGENTS.md hard rule #1). This requires a GPU on the machine.

- [ ] **Step 4: Verify no `matches!(self.arch, ...)` or `self.arch.starts_with(...)` remains in dispatch.rs**

```bash
rg 'matches!\(self\.arch|self\.arch\.starts_with' crates/rdna-compute/src/dispatch.rs | head
```

Expected: Zero results (all replaced by `self.arch_caps.*()` calls). The only acceptable `self.arch` references left are direct string comparisons that don't map to any named predicate (e.g., `self.arch == "gfx906"` inside method bodies that should become `self.arch_caps.is_gfx906()`).

---

## Phase 2: Eliminate `*_for_arch()` — kernel symbol selection moves to `ArchCaps`

### Task 5: Add kernel symbol selection methods to `ArchCaps`

**Files:**
- Modify: `crates/rdna-compute/src/arch_caps.rs`
- Modify: `crates/rdna-compute/src/dispatch.rs` (update call sites)

This is the largest single task. Each of the 32 `*_for_arch()` functions becomes an `ArchCaps` method. They're grouped by family (matching the audit above).

- [ ] **Step 1: Add MQ4-Lloyd WMMA methods (9 functions → 9 methods)**

For each WMMA method, the pattern is:
```rust
pub fn gemm_mq4g256_lloyd_residual_wmma(&self) -> (&'static str, &'static str) {
    if self.is_rdna4 {
        (kernels::GEMM_MQ4G256_LLOYD_RESIDUAL_WMMA_GFX12_SRC, "gemm_mq4g256_lloyd_residual_wmma_rdna4")
    } else if self.is_rdna3_dgpu_1151 {
        (kernels::GEMM_MQ4G256_LLOYD_RESIDUAL_WMMA_SRC, "gemm_mq4g256_lloyd_residual_wmma_rdna3")
    } else {
        panic!("MQ4 Lloyd residual WMMA: no kernel for {}", self.arch)
    }
}
```

Note: The original `*_for_arch()` functions return `(SRC_CONST, symbol_name_str)`. The `ArchCaps` methods should have the same return type for a drop-in replacement.

Add all 9 MQ4 WMMA methods following this pattern, using the appropriate predicates:
- `is_rdna4` + `is_rdna3_dgpu_1151` for the 5 dual-arch WMMA functions (residual, qkvza, qkv, gate_up for both rdna3 and rdna4)
- `is_rdna3_dgpu_1151` only for the 4 gfx11-only WMMA functions (residual_mb2, qkvza_mb4, qkv_mb4, gate_up_mb4)

- [ ] **Step 2: Add MQ4-Lloyd GEMV/fused methods (5 functions → 5 methods)**

Each has `HIPFIRE_LLOYD_FORCE_BASELINE` env override → `self.flags.lloyd_force_baseline`. Each returns baseline on `_` arm:

```rust
pub fn gemv_mq4g256_lloyd(&self) -> (&'static str, &'static str) {
    if self.flags.lloyd_force_baseline {
        return (kernels::GEMV_MQ4G256_LLOYD_SRC, "gemv_mq4g256_lloyd");
    }
    if self.is_rdna3_dgpu_1151 {
        (kernels::GEMV_MQ4G256_LLOYD_GFX1100_SRC, "gemv_mq4g256_lloyd_rdna3")
    } else {
        (kernels::GEMV_MQ4G256_LLOYD_SRC, "gemv_mq4g256_lloyd")
    }
}
```

Add all 5 MQ4 GEMV/fused methods.

- [ ] **Step 3: Add MQ3-Lloyd WMMA methods (8 functions → 8 methods)**

Same pattern as Step 1 but using `is_rdna3_full` (includes gfx1150) instead of `is_rdna3_dgpu_1151`:
- 5 dual-arch WMMA methods (residual, qkvza, qkv, gate_up for both rdna3 and rdna4)
- 3 gfx11-only WMMA methods (residual_mb4, qkvza_mb4, qkv_mb4, gate_up_mb4)

Wait, that's 4 mb4 methods. Add all 8.

- [ ] **Step 4: Add MQ3-Lloyd GEMV/fused methods (5 functions → 5 methods)**

Same pattern as Step 2 but using `is_rdna3_dgpu_1151` (NOT `is_rdna3_full` — gfx1150 is excluded from MQ3 GEMV per the documented PPL drift):

```rust
pub fn gemv_mq3g256_lloyd(&self) -> (&'static str, &'static str) {
    if self.flags.lloyd_force_baseline {
        return (kernels::GEMV_MQ3G256_LLOYD_SRC, "gemv_mq3g256_lloyd");
    }
    if self.is_rdna3_dgpu_1151 {
        (kernels::GEMV_MQ3G256_LLOYD_GFX1100_SRC, "gemv_mq3g256_lloyd_rdna3")
    } else {
        (kernels::GEMV_MQ3G256_LLOYD_SRC, "gemv_mq3g256_lloyd")
    }
}
```

- [ ] **Step 5: Add HFQ4/HFP4/HFQ3 GEMV methods (5 functions → 5 methods)**

Special case: `gemv_hfq4g256_for_arch` has RDNA2 variant selection based on env var `HIPFIRE_RDNA2_VARIANT` (which is now `self.flags.rdna2_variant` or similar on FeatureFlags). Check the exact FeatureFlags field name.

```rust
pub fn gemv_hfq4g256(&self) -> (&'static str, &'static str) {
    if self.is_rdna2 {
        match self.flags.rdna2_variant {
            1 => (kernels::GEMV_HFQ4G256_GFX1030_V1_SRC, "gemv_hfq4g256_rdna2v1"),
            2 => (kernels::GEMV_HFQ4G256_GFX1030_V2_SRC, "gemv_hfq4g256_rdna2v2"),
            3 => (kernels::GEMV_HFQ4G256_GFX1030_V3_SRC, "gemv_hfq4g256_rdna2v3"),
            4 => (kernels::GEMV_HFQ4G256_GFX1030_V4_SRC, "gemv_hfq4g256_rdna2v4"),
            5 => (kernels::GEMV_HFQ4G256_GFX1030_V5_SRC, "gemv_hfq4g256_rdna2v5"),
            _ => (kernels::GEMV_HFQ4G256_GFX1030_V1_SRC, "gemv_hfq4g256_rdna2v1"),
        }
    } else if self.is_rdna3_dgpu {
        (kernels::GEMV_HFQ4G256_GFX1100_SRC, "gemv_hfq4g256_rdna3")
    } else {
        (kernels::GEMV_HFQ4G256_SRC, "gemv_hfq4g256")
    }
}
```

Note: The `is_rdna3_dgpu` predicate here excludes gfx1150/1151 — they fall to baseline. This matches the original `for_arch` which only matches `"gfx1100" | "gfx1101" | "gfx1102"`.

For `gemv_hfp4g32`, `gemv_hfq4g256_residual`, `gemv_hfq3g256`, and `gemv_hfq3g256_residual` — use the appropriate predicates per the audit.

- [ ] **Step 6: Run `cargo check -p rdna-compute`**

Expected: Passes (all ArchCaps methods exist, but `*_for_arch` calls in dispatch.rs not yet replaced).

- [ ] **Step 7: Commit**

```
feat(rdna-compute): add kernel symbol selection methods to ArchCaps

32 methods replacing *_for_arch() functions in kernels.rs.
Each uses named predicates instead of raw arch matching.
```

---

### Task 6: Replace all `*_for_arch()` call sites in dispatch.rs

**Files:**
- Modify: `crates/rdna-compute/src/dispatch.rs`

- [ ] **Step 1: Find all `*_for_arch()` call sites**

```bash
rg '_for_arch\(' crates/rdna-compute/src/dispatch.rs
```

Each call site like `kernels::gemm_mq4g256_lloyd_residual_wmma_for_arch(&self.arch)` becomes `self.arch_caps.gemm_mq4g256_lloyd_residual_wmma()`.

The return type changes from `(&'static str, &'static str)` (what `*_for_arch` returned) to the same type (what the ArchCaps methods return). The call sites unpack the tuple as `(src, symbol_name)` so they should work without changes to the unpacking logic.

- [ ] **Step 2: Replace each call site mechanically**

All ~35 call sites. This is a mechanical find-and-replace, one per line. No logic changes.

- [ ] **Step 3: Run `cargo check -p rdna-compute`**

Expected: Passes.

- [ ] **Step 4: Commit**

```
refactor(rdna-compute): replace *_for_arch() calls with ArchCaps methods in dispatch

Mechanical replacement: kernels::X_for_arch(&self.arch) → self.arch_caps.X()
```

---

### Task 7: Remove `*_for_arch()` functions from `kernels.rs`

**Files:**
- Modify: `crates/rdna-compute/src/kernels.rs`

- [ ] **Step 1: Delete all 32 `*_for_arch()` function bodies**

Remove the `pub fn *_for_arch(arch: &str) -> ...` functions. Keep all `pub const` symbol strings — they're still referenced by `ArchCaps` methods.

- [ ] **Step 2: Verify no remaining references to `*_for_arch` in the crate**

```bash
rg '_for_arch' crates/rdna-compute/src/
```

Expected: Zero results in `kernels.rs`. Any remaining references are in ArchCaps method comments or test code.

- [ ] **Step 3: Run `cargo check -p rdna-compute`**

Expected: Passes with zero errors (all references now go through ArchCaps).

- [ ] **Step 4: Commit**

```
refactor(rdna-compute): remove *_for_arch() functions from kernels.rs

All kernel symbol selection now lives in ArchCaps. kernels.rs is
pure data (pub const symbol strings only).
```

---

### Task 8: Phase 2 validation — cargo check + tests + coherence-gate

**Files:** None (validation only)

- [ ] **Step 1: Run `cargo check -p rdna-compute`**

Expected: Clean compilation.

- [ ] **Step 2: Run `cargo test -p rdna-compute`**

Expected: All tests pass including the new ArchCaps tests.

- [ ] **Step 3: Run `./scripts/coherence-gate-dflash.sh`**

Expected: All 4 tests pass.

- [ ] **Step 4: Symbol-name correctness audit**

Run both before and after Phase 2 (or diff against the pre-refactor version):

```bash
rg 'pub const.*_SRC' crates/rdna-compute/src/kernels.rs | sort
```

Verify that every constant name referenced in the `*_for_arch` match arms is still present as a `pub const` in `kernels.rs`. No typos, no missing symbols.

- [ ] **Step 5: Verify arch-coverage parity**

For each `ArchCaps` method, confirm that the arch sets match the original `*_for_arch` exactly:
- MQ4 WMMA: `gfx1200|gfx1201` → rdna4, `gfx1100|gfx1101|gfx1102|gfx1151` → rdna3
- MQ4 GEMV/fused: `gfx1100|gfx1101|gfx1102|gfx1151` → rdna3, else → baseline
- MQ3 WMMA: `gfx1200|gfx1201` → rdna4, `gfx1100|gfx1101|gfx1102|gfx1150|gfx1151` → rdna3
- MQ3 GEMV/fused: `gfx1100|gfx1101|gfx1102|gfx1151` → rdna3, else → baseline
- HFQ4: `gfx1030|gfx1031` → rdna2 (variant), `gfx1100|gfx1101|gfx1102` → rdna3, else → baseline
- HFP4: `gfx1100|gfx1101|gfx1102|gfx1150|gfx1151` → rdna3, else → baseline
- HFQ4 residual: `gfx1100|gfx1101|gfx1102` → rdna3, else → baseline
- HFQ3: `gfx1100|gfx1101|gfx1102` → rdna3, else → baseline
- HFQ3 residual: `gfx1100|gfx1101|gfx1102` → rdna3, else → baseline

**If any arch set doesn't match, fix before proceeding.**