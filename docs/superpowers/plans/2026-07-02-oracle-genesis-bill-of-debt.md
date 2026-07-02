# Oracle Genesis / Bill of Debt — Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL — use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the Phase-0 genesis of the RDNA kernel Oracle — the accuracy-hardened, provenance-enforced, per-arch "bill of debt" that itemizes every RDNA kernel's recoverable in-model time as `Measured`/`Withheld`/`Structural` rows, honestly promoted or withheld per a per-cell/per-arch gate.

**Architecture:** Pure-Rust no-GPU diagnostic/instrument modules under `crates/rdna-compute/src/` (roofline, chip-profile, kernel-ledger, profile-rocprof, isa-histogram, plus new `pmc_census` and `bill_of_debt`) carry all policy logic and are unit-tested offline against committed JSON fixtures; the GPU only ever produces input CSVs. A bash harness (`scripts/oracle-bill-of-debt-sweep.sh`) fans the measurement matrix over all five RDNA arches on the single hipx runner, each arch pinned + per-card-locked, two DPM passes per cell (kernel-trace/tok-s on `high`; PMC ratios on `profile_standard`), emitting measured rows through the existing `kernel_perf_instrument --dynamic` emitter and WITHHELD rows (never faked) for absent/OOM cells.

**Tech Stack:** Rust (workspace, `rdna-compute` crate; `serde_json`, no serde-derive in this crate), `hipfire-atlas` AtlasRow JSONL corpus, HIP/ROCm via runtime `dlopen` (no link-time dependency), the offline LLVM toolchain (`llvm-readelf`, `clang-offload-bundler`) for `.hsaco` disassembly, rocprofv3 (`--pmc`, kernel-trace) for on-device capture, Bash + `flock(1)` GPU locks, Python3 for JSON/registry tooling.

## Global Constraints

- Rust workspace builds no-GPU via HIP dlopen; VALIDATE rdna-compute via the WORKSPACE build or `--features deltanet`, NEVER `-p rdna-compute --all-targets` alone (`rope_compact_offset_check` needs `cfg=deltanet`).
- Measurement code stays coherence-neutral.
- Withheld-never-faked.
- Per-arch symmetric with NO arch-invariance transfer.
- GPU runs on hipx with per-card `HIPFIRE_GPU_LOCKFILE`.
- Commits end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Do not hand-edit `registry/v1.json`.
- Kernel headers use `.hip` not `.cuh`.

---

## File Structure

**Created**

| File | Responsibility |
|---|---|
| `crates/rdna-compute/src/pmc_census.rs` | Per-arch PMC counter census (Derived vs RawAccumulator revival under `profile_standard`) + per-cell/per-arch promotion gate (`PROMOTED\|WITHHELD{reason}`). (0c) |
| `crates/rdna-compute/examples/pmc_census.rs` | No-GPU aggregation binary: fold captured rocprofv3 `--pmc` CSVs into a committed per-arch census JSON. (0c) |
| `tests/pmc-census/gfx1010.json` … `gfx1201.json` | Real captured per-arch `profile_standard` census rows (raw counts only, verdicts derived at load). (0c) |
| `crates/rdna-compute/src/bill_of_debt.rs` | The bill-of-debt ledger: `DebtRow` (Measured/Withheld/Structural), query-derived debt magnitude, ranking, per-arch totals, cross-arch unevenness, no-arch-clobber delta, JSONL emit/load. (0d) |
| `scripts/oracle-serve-dynamic.sh` | `[GPU]` Spawn the real daemon under rocprofv3 on a committed prompt, feed the served CSV to `--dynamic` with structured serve provenance. (0b) |
| `scripts/oracle-bill-of-debt-sweep.sh` | `[GPU]` hipx all-5-arch execution harness: per-card locks, two DPM passes/cell, registry VRAM enumeration, arch→device by runtime `arch:` line, WITHHELD on absent/OOM. (0e) |
| `scripts/tests/test-oracle-bill-of-debt-sweep.sh` | No-GPU regression test for the harness (self-check plan, VRAM enumeration, device resolution, dry-run matrix, live-command dry-run, GPU-tooling isolation). (0e) |
| `tests/fixtures/oracle-registry-mini.json` | Deterministic 4-model test registry for the harness tests. (0e) |

**Modified**

| File | Responsibility |
|---|---|
| `crates/rdna-compute/src/chip_profile.rs` | `pointer_chase_buffer_bytes` + `check_residency` + WITHHELD IC-part rows (0a1); `simds_per_cu` field + JSON back-fill + `verify_live` branch (0a2); `BwTiers` cache working-set + `validate_cache_residency` (0a3). |
| `crates/rdna-compute/examples/pointer_chase_latency.rs` | Per-arch DRAM buffer sizing + cache-resident reference + residency guard. (0a1) |
| `tests/chip-profiles/gfx1030.json` / `gfx1100.json` / `gfx1201.json` | `mem_latency_ns: null` (WITHHELD) with `#490` provenance note. (0a1) |
| `crates/rdna-compute/src/roofline.rs` | `simds_per_cu` factor in `latency_score` + numeric regression (0a2); `MeasuredBw{gbps,trust}` propagation + summed-VMEM-width latency short-circuit (0a3). |
| `crates/rdna-compute/src/kernel_ledger.rs` | caller-derived `bound_class` in `from_fixture` + `diff` bound-class flip (0a3); `dpm_state` persistence (0a3); `Reproducer::require_serve_provenance` + `is_md5_hex` (0b). |
| `crates/rdna-compute/src/profile_rocprof.rs` | refuse `profile_standard` achieved-BW + `dpm_state` (0a3); word-boundary coverage matching (0a3); `AchievedBw.in_model_pct` + `is_covered`/`coverage_against_aliases`/`AliasCoverage` (0b). |
| `crates/rdna-compute/src/profiler.rs` | HIP-ordinal→physical sysfs resolution for CU/clocks/bus-width; `vgpr_count_from_note` + `cross_check_vgprs` wired into the profiling path. (0a3) |
| `crates/rdna-compute/src/isa_histogram.rs` | `vmem_load_bytes_per_lane` summed across all VMEM load widths. (0a3) |
| `crates/rdna-compute/examples/kernel_perf_instrument.rs` | caller-derived bound_class + `MeasuredBw` + `dpm_state` wiring (0a3); provenance enforcement + `in_model_pct` + blindspot/coverage emit (0b). |
| `crates/rdna-compute/src/lib.rs` | Register `pub mod pmc_census;` (0c) and `pub mod bill_of_debt;` (0d) — two append-only lines. |
| `scripts/no-gpu-ci.sh` | Wire the harness no-GPU regression test into CI (append-only block). (0e) |

---

## Task dependency & swarm map

| task | depends_on | parallel_safe | shares_files_with |
|---|---|---|---|
| 0a1 (#490 mem_latency) | — | false | 0a2, 0a3 |
| 0a2 (#491 simds_per_cu) | — | false | 0a3 |
| 0a3 (MED round-up) | 0a1, 0a2 | false | 0a1, 0a2, 0b |
| 0b (in-model loop) | 0a1, 0a2, 0a3 | false | 0a3, 0d |
| 0c (PMC census + gate) | — | false | 0d |
| 0d (bill of debt) | — | false | 0b |
| 0e (hipx harness) | 0b, 0c, 0d | true | — |

**Fan-out vs serialize.** The **chip_profile.rs / roofline.rs cluster (0a1, 0a2, 0a3)** is the anti-clobber hotspot: all three edit `chip_profile.rs`, and 0a2+0a3 both edit `roofline.rs`'s `latency_score` body and its five test literals. These MUST NOT be authored concurrently against a shared tree — apply the same anti-clobber discipline we impose on kernel code to our OWN swarm: run each in `isolation:'worktree'` and **return diffs for serial commit**, or serialize them. Ordering: **0a1 → 0a2 → 0a3** (0a3 `depends_on` both HIGH fixes and its preflight refuses to run on an unfixed tree). 0b shares `profile_rocprof.rs`/`kernel_ledger.rs`/`kernel_perf_instrument.rs` struct-literal sites with 0a3, so **0a3 lands before 0b** (both add fields to the same `AchievedBw`/`LedgerRow` literals).

**Free fan-out (locus-disjoint).** 0c's `pmc_census.rs` (+ its `examples/` binary + `tests/pmc-census/*.json`) and 0d's `bill_of_debt.rs` are brand-new files owned solely by their task; 0e's three files are unique to it. These three tasks (**0c, 0d, 0e-noGPU-cycles**) fan out concurrently with each other and with the 0a cluster — the **only** collision surface is the two append-only `pub mod` lines in `crates/rdna-compute/src/lib.rs` (0c adds `pmc_census`, 0d adds `bill_of_debt`) and the append-only CI block in `scripts/no-gpu-ci.sh` (0e). Resolve `lib.rs` by keeping BOTH `pub mod` lines (trivial two-line union) — apply those one-line edits last, or serialize just that edit. 0e's live GPU cycle depends on 0b/0c/0d landing; its no-GPU TDD cycles (A–D, F) fan out immediately.

**Concurrency summary:** Group 1 (serialize/isolate): `0a1 → 0a2 → 0a3 → 0b`. Group 2 (free fan-out, concurrent with Group 1): `0c`, `0d`, `0e` (no-GPU cycles). Final serialize: 0e live GPU sweep after Group-1 tail + 0c + 0d land.

---

## Task 0a1 — Fix #490: cache-deflated `mem_latency_ns` (per-arch pointer-chase buffer + cache-residency guard)

**Spec anchor:** `docs/superpowers/specs/2026-07-02-oracle-genesis-bill-of-debt-design.md` §4a (HIGH item, first row) and §5.3 risk table (row 1):

> **HIGH — `mem_latency_ns` is cache-deflated on every Infinity-Cache part.** The fixed 128 MiB pointer-chase buffer fits inside the IC (gfx1030 fully; gfx1100/1201 cleared by only 1.3×/1.9×). Committed latencies track cache *size*, not memory tech. **Fix:** size the buffer per-arch to ≥ 4×(L2+IC) from `arch_spec` (≥512 MiB on RDNA2); add a cache-residency guard; re-measure gfx1030/1100/1201; stop shipping cache-deflated rows as clean. *[#490]*

Root cause in code today: `crates/rdna-compute/examples/pointer_chase_latency.rs:39` hardcodes `const BUFFER_BYTES: usize = 128 * 1024 * 1024;`, and the only correctness check is `assert!((50.0..2000.0).contains(&mem_latency_ns), ...)` at line 238 — a magnitude bound that a fully cache-resident chase still passes. Per `arch_spec` (`crates/rdna-compute/src/profiler.rs:41`), 128 MiB is *smaller* than gfx1030's on-chip cache (L2 4 + IC 128 = 132 MiB) and clears gfx1100 (6+96=102 MiB) / gfx1201 (4+64=68 MiB) by only 1.25×/1.9×. The committed rows `tests/chip-profiles/{gfx1030,gfx1100,gfx1201}.json` therefore hold cache latencies mislabeled as DRAM.

This task lands the two pure, no-GPU-testable policy functions (buffer sizing + residency guard), WITHHOLDs the three cache-deflated committed rows (deterministic, CI-green without a GPU), and rewires the microbench example to use both. Actual re-measurement is the final `[GPU: hipx]` promotion step.

### Files

- **Modify** `crates/rdna-compute/src/chip_profile.rs` — add `pub fn pointer_chase_buffer_bytes` + `pub fn check_residency` and their unit tests; update the `load_all_committed_chip_profiles` corpus test to expect WITHHELD on the IC parts.
- **Modify** `crates/rdna-compute/examples/pointer_chase_latency.rs` — per-arch buffer sizing via `chip_profile::pointer_chase_buffer_bytes`, two-size (cache vs DRAM) measurement, and the `chip_profile::check_residency` guard.
- **Modify** `tests/chip-profiles/gfx1030.json`, `tests/chip-profiles/gfx1100.json`, `tests/chip-profiles/gfx1201.json` — set `mem_latency_ns: null` (WITHHELD) with a `#490` provenance note.
- **Test** (no-GPU, rdna-compute lib tests): `pointer_chase_buffer_exceeds_4x_onchip_cache`, `pointer_chase_buffer_rdna2_exceeds_512mib`, `pointer_chase_buffer_floors_at_128mib_for_no_ic_parts`, `pointer_chase_buffer_rdna4_and_rdna3`, `check_residency_flags_cache_deflated_gfx1030`, `check_residency_passes_genuine_dram`, `check_residency_boundary_is_inclusive`, `check_residency_rejects_bad_inputs`, and the amended `load_all_committed_chip_profiles`.

### Interfaces

**Consumes (real signatures, quoted):**
- `pub(crate) fn static_capability(arch: &str) -> GpuCapability` — `crates/rdna-compute/src/profiler.rs:251` (same-crate; the call form is the associated `GpuCapability::static_capability(arch)`, already used by `ChipProfile::for_unprofiled` at line 172).
- `GpuCapability` fields — `crates/rdna-compute/src/profiler.rs:20-21`: `pub l2_cache_mb: f32,` and `pub infinity_cache_mb: f32, // 0 for RDNA1`.
- `pub fn detect(arch: &str, vram_bytes: u64) -> Self` — `crates/rdna-compute/src/profiler.rs:156` (the pub entry the example uses; `static_capability` is `pub(crate)` and MUST NOT be called from the example crate).
- Existing example FFI (unchanged): `gpu.hip.malloc(bytes)`, `gpu.hip.memcpy_htod(&d_buf, host_bytes)`, `gpu.hip.launch_kernel(...)`, `profile::begin_timer(...)`, `profile::stop() -> Option<Vec<_>>` with `.last().map(|e| e.time_us)`.

**Produces (new public signatures in `crates/rdna-compute/src/chip_profile.rs`):**
- `pub fn pointer_chase_buffer_bytes(arch: &str) -> usize`
- `pub fn check_residency(cache_resident_ns: f64, candidate_dram_ns: f64, min_dram_over_cache_ratio: f64) -> Result<(), String>`

---

### Sub-task A — per-arch buffer sizing (pure, no-GPU)

**A0. Import guard (adversarial-review fix).** The `#[cfg(test)] mod tests` block uses `use super::*;`, but the parent module's `use crate::profiler::GpuCapability;` is a **private** `use` alias that is not reliably re-exported into the child by a `super::*` glob. Add an explicit import at the top of the `#[cfg(test)] mod tests { ... }` block **before** writing A1:

```rust
    use crate::profiler::GpuCapability;
```

- [ ] **A1. Write the failing tests.** Append inside `#[cfg(test)] mod tests` in `crates/rdna-compute/src/chip_profile.rs`:

```rust
    #[test]
    fn pointer_chase_buffer_exceeds_4x_onchip_cache() {
        // Every arch's chase buffer must clear 4x its (L2 + Infinity Cache)
        // footprint so the working set is genuinely DRAM-resident (issue #490).
        for arch in ["gfx1010", "gfx1030", "gfx1100", "gfx1151", "gfx1201"] {
            let cap = GpuCapability::static_capability(arch);
            let onchip_bytes =
                ((cap.l2_cache_mb + cap.infinity_cache_mb) as f64 * 1024.0 * 1024.0) as usize;
            let buf = pointer_chase_buffer_bytes(arch);
            assert!(
                buf >= 4 * onchip_bytes,
                "{arch}: pointer-chase buffer {buf} must be >= 4x on-chip cache {}",
                4 * onchip_bytes
            );
        }
    }

    #[test]
    fn pointer_chase_buffer_rdna2_exceeds_512mib() {
        // gfx1030 (RDNA2): L2 4 MiB + IC 128 MiB -> 4x(132) = 528 MiB, clearing
        // the spec's ">= 512 MiB on RDNA2" floor. The OLD fixed 128 MiB buffer
        // fit ENTIRELY inside the 128 MiB IC — the exact #490 deflation.
        let buf = pointer_chase_buffer_bytes("gfx1030");
        assert_eq!(buf, 528 * 1024 * 1024, "gfx1030 = 4*(4+128) = 528 MiB");
        assert!(buf >= 512 * 1024 * 1024, "RDNA2 buffer must clear the 512 MiB floor, got {buf}");
        assert!(buf > 128 * 1024 * 1024, "must exceed the deflated 128 MiB buffer that fit inside the 128 MiB IC");
    }

    #[test]
    fn pointer_chase_buffer_floors_at_128mib_for_no_ic_parts() {
        // gfx1010 (RDNA1, no IC) and gfx1151 (RDNA3.5, no discrete IC) fall
        // below the floor, so the 128 MiB minimum applies — these rows were
        // never cache-deflated (128 MiB already >> their L2).
        assert_eq!(pointer_chase_buffer_bytes("gfx1010"), 128 * 1024 * 1024);
        assert_eq!(pointer_chase_buffer_bytes("gfx1151"), 128 * 1024 * 1024);
    }

    #[test]
    fn pointer_chase_buffer_rdna4_and_rdna3() {
        // gfx1201 (RDNA4): 4x(4+64) = 272 MiB; gfx1100 (RDNA3): 4x(6+96) = 408 MiB.
        assert_eq!(pointer_chase_buffer_bytes("gfx1201"), 272 * 1024 * 1024);
        assert_eq!(pointer_chase_buffer_bytes("gfx1100"), 408 * 1024 * 1024);
    }
```

- [ ] **A2. Run → expect FAIL.** `cargo test -p rdna-compute --lib pointer_chase_buffer` → `error[E0425]: cannot find function pointer_chase_buffer_bytes in this scope`.

- [ ] **A3. Minimal implementation.** Add after the `impl ChipProfile { ... }` block closes (line 318, before the `BwSweepPoint` struct at line 326) — `use crate::profiler::GpuCapability;` is already imported at line 26:

```rust
/// Minimum pointer-chase buffer size in **bytes** for `arch` such that the
/// working set is genuinely DRAM-resident: >= 4x the on-chip (L2 + Infinity
/// Cache) footprint from `arch_spec`, floored at 128 MiB.
///
/// Issue #490: the microbench shipped a FIXED 128 MiB buffer, which fits inside
/// the Infinity Cache on every IC-bearing RDNA part. Sizing to >= 4x(L2+IC)
/// spills the working set well past the last cache level so each hop is a true
/// DRAM round trip. The 128 MiB floor keeps the historically proven size on
/// no-IC parts (gfx1010, gfx1151).
///
/// PURE + NO-GPU: reads only `GpuCapability::static_capability(arch)`'s
/// arch_spec cache constants — deterministic, unit-testable without a device.
pub fn pointer_chase_buffer_bytes(arch: &str) -> usize {
    const MIB: usize = 1024 * 1024;
    const FLOOR_MIB: f64 = 128.0;
    const SAFETY_MULT: f64 = 4.0;
    let cap = GpuCapability::static_capability(arch);
    let on_chip_mib = (cap.l2_cache_mb + cap.infinity_cache_mb) as f64;
    let target_mib = (SAFETY_MULT * on_chip_mib).ceil().max(FLOOR_MIB);
    target_mib as usize * MIB
}
```

- [ ] **A4. Run → expect PASS.** `cargo test -p rdna-compute --lib pointer_chase_buffer`.

- [ ] **A5. Commit.**
```
git add crates/rdna-compute/src/chip_profile.rs && git commit -m "fix(oracle): size pointer-chase buffer per-arch to >=4x(L2+IC) [#490]

The fixed 128 MiB buffer fit inside the Infinity Cache on every IC-bearing
RDNA part, so the measured latency tracked cache size, not DRAM.
pointer_chase_buffer_bytes derives a per-arch >=4x(L2+IC) size from arch_spec
(528 MiB RDNA2, 408 RDNA3, 272 RDNA4), floored at 128 MiB for no-IC parts.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Sub-task B — cache-residency guard (pure, no-GPU)

- [ ] **B1. Write the failing tests.** Append inside `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn check_residency_flags_cache_deflated_gfx1030() {
        // The #490 signature: the shipped "DRAM" latency is barely above a
        // genuine L2-hit reference — within noise, NOT a real >1.5x DRAM step.
        let result = check_residency(140.0, 148.035, 1.5);
        assert!(result.is_err(), "a within-noise DRAM/cache ratio must fail, got {result:?}");
        assert!(result.unwrap_err().contains("cache-DEFLATED"));
    }

    #[test]
    fn check_residency_passes_genuine_dram() {
        assert!(check_residency(140.0, 300.0, 1.5).is_ok());
    }

    #[test]
    fn check_residency_boundary_is_inclusive() {
        assert!(check_residency(100.0, 150.0, 1.5).is_ok());
        assert!(check_residency(100.0, 149.999, 1.5).is_err());
    }

    #[test]
    fn check_residency_rejects_bad_inputs() {
        assert!(check_residency(0.0, 300.0, 1.5).is_err());     // non-positive cache
        assert!(check_residency(140.0, f64::NAN, 1.5).is_err()); // non-finite dram
        assert!(check_residency(140.0, 300.0, 1.0).is_err());   // ratio must be > 1.0
    }
```

- [ ] **B2. Run → expect FAIL.** `cargo test -p rdna-compute --lib check_residency` → `error[E0425]: cannot find function check_residency in this scope`.

- [ ] **B3. Minimal implementation.** Add directly below `pointer_chase_buffer_bytes`:

```rust
/// Cache-residency guard for a pointer-chase latency measurement (issue #490).
///
/// The bare `[50, 2000) ns` plausibility assert CANNOT prove the working set
/// reached DRAM — a fully cache-resident chase still lands in that window. A
/// genuine DRAM round trip has per-hop latency substantially HIGHER than a
/// deliberately cache-resident chase on the same device. If the DRAM-sized
/// chase is not at least `min_dram_over_cache_ratio`x the cache-resident chase,
/// the large buffer never left cache and the number is cache-DEFLATED -> `Err`
/// (WITHHELD, never shipped as a clean DRAM latency).
///
/// PURE + NO-GPU: operates on two already-measured latencies.
pub fn check_residency(
    cache_resident_ns: f64,
    candidate_dram_ns: f64,
    min_dram_over_cache_ratio: f64,
) -> Result<(), String> {
    if !cache_resident_ns.is_finite() || cache_resident_ns <= 0.0 {
        return Err(format!("check_residency: cache_resident_ns must be finite and positive, got {cache_resident_ns}"));
    }
    if !candidate_dram_ns.is_finite() || candidate_dram_ns <= 0.0 {
        return Err(format!("check_residency: candidate_dram_ns must be finite and positive, got {candidate_dram_ns}"));
    }
    if !min_dram_over_cache_ratio.is_finite() || min_dram_over_cache_ratio <= 1.0 {
        return Err(format!("check_residency: min_dram_over_cache_ratio must be finite and > 1.0, got {min_dram_over_cache_ratio}"));
    }
    let required = cache_resident_ns * min_dram_over_cache_ratio;
    if candidate_dram_ns >= required {
        Ok(())
    } else {
        Err(format!(
            "check_residency: DRAM-sized chase {candidate_dram_ns:.3} ns is not >= \
             {min_dram_over_cache_ratio:.2}x the cache-resident chase {cache_resident_ns:.3} ns \
             (required >= {required:.3} ns) — the 'DRAM' buffer is still CACHE-RESIDENT and this \
             latency is cache-DEFLATED (issue #490). WITHHELD, not shipped."
        ))
    }
}
```

- [ ] **B4. Run → expect PASS.** `cargo test -p rdna-compute --lib check_residency`.

- [ ] **B5. Commit.**
```
git add crates/rdna-compute/src/chip_profile.rs && git commit -m "fix(oracle): add cache-residency guard for pointer-chase latency [#490]

check_residency rejects a measurement whose DRAM-sized chase is not >= ratio x
a deliberately cache-resident chase — the residency proof the [50,2000) ns
plausibility assert cannot provide.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Sub-task C — WITHHELD the three cache-deflated committed rows (no-GPU, deterministic)

- [ ] **C1. Amend the corpus test.** In `load_all_committed_chip_profiles`, replace the expected-value block (lines 544-564) with:

```rust
            // (expected cu_count, expected mem_latency_ns, expected cache_bw_gbps)
            // mem_latency_ns is WITHHELD (None) on the three Infinity-Cache
            // parts (gfx1030/1100/1201): issue #490 — the old 128 MiB buffer fit
            // inside their IC, so shipped latencies were cache-DEFLATED. They
            // stay WITHHELD until re-measured. gfx1010/gfx1151 (no IC) stay Some.
            let (expected_cu, expected_latency, expected_cache_bw): (u32, Option<f64>, Option<f64>) =
                match arch.as_str() {
                    "gfx1010" => (40, Some(276.058), None),
                    "gfx1030" => (80, None, Some(1586.5)),
                    "gfx1100" => (96, None, Some(1778.8)),
                    "gfx1151" => (40, Some(219.932), Some(952.5)),
                    "gfx1201" => (64, None, Some(1352.4)),
                    other => panic!(
                        "load_all_committed_chip_profiles: unrecognized committed row {other:?} — \
                         add an expected-value arm here"
                    ),
                };

            assert_eq!(profile.cu_count, expected_cu, "{arch}: cu_count");
            assert_eq!(
                profile.mem_latency_ns, expected_latency,
                "{arch}: mem_latency_ns (WITHHELD/None on IC parts per issue #490 until re-measured)"
            );
```
(Leave the subsequent `cache_bw_gbps` / `effective_cache_mib` asserts unchanged.)

- [ ] **C2. Run → expect FAIL.** `cargo test -p rdna-compute --lib load_all_committed_chip_profiles` → `left: Some(148.035), right: None` for gfx1030.

- [ ] **C3. WITHHELD the JSON rows.** Three edits:
  - `tests/chip-profiles/gfx1030.json` — change `"mem_latency_ns": 148.035,` → `"mem_latency_ns": null,`; prepend to `_note` (anchor: begins `"RX 6950 XT / gfx1030 (RDNA2)...`): `WITHHELD (issue #490): prior mem_latency_ns=148.035 ns was CACHE-DEFLATED — the fixed 128 MiB pointer-chase buffer fit ENTIRELY inside this part's 128 MiB Infinity Cache (L2 4 + IC 128 = 132 MiB). Re-measure under the per-arch >=4x(L2+IC)=528 MiB buffer + chip_profile::check_residency guard before restoring a value. `
  - `tests/chip-profiles/gfx1100.json` — change `"mem_latency_ns": 169.719,` → `null`; prepend to `_note` (anchor `"RX 7900 XTX / gfx1100 (RDNA3)`): `WITHHELD (issue #490): prior mem_latency_ns=169.719 ns was CACHE-DEFLATED — the 128 MiB buffer cleared this part's 96 MiB IC + 6 MiB L2 by only 1.3x. Re-measure under the 408 MiB buffer + residency guard. `
  - `tests/chip-profiles/gfx1201.json` — change `"mem_latency_ns": 205.888,` → `null`; prepend to `_note` (anchor `"R9700/gfx1201 static hardware constants`): `WITHHELD (issue #490): prior mem_latency_ns=205.888 ns was CACHE-DEFLATED — the 128 MiB buffer cleared this part's 64 MiB IC + 4 MiB L2 by only 1.9x. Re-measure under the 272 MiB buffer + residency guard. `

- [ ] **C4. Run → expect PASS.** `cargo test -p rdna-compute --lib chip_profile` (`round_trip_json` / `from_json_missing_cache_fields_withholds` still pass — a JSON `null` reads back as `None`).

- [ ] **C5. Commit.**
```
git add crates/rdna-compute/src/chip_profile.rs tests/chip-profiles/gfx1030.json tests/chip-profiles/gfx1100.json tests/chip-profiles/gfx1201.json && git commit -m "fix(oracle): WITHHELD cache-deflated mem_latency on gfx1030/1100/1201 [#490]

Set to null (WITHHELD) with a #490 provenance note until re-measured under the
per-arch >=4x(L2+IC) buffer + residency guard. gfx1010/gfx1151 (no IC) stay
Some. Withheld, never faked.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Sub-task D — rewire the microbench example (per-arch buffer + residency guard)

- [ ] **D1. Three edits to `crates/rdna-compute/examples/pointer_chase_latency.rs`.**

Edit 1 — imports:
```rust
use rdna_compute::profiler::GpuCapability;
use rdna_compute::{chip_profile, profile, Gpu, KernelCompiler};
use std::ffi::c_void;
```

Edit 2 — replace the fixed-buffer const block (lines 34-45) with:
```rust
/// Cache-resident reference buffer for the residency guard (issue #490).
/// 512 KiB fits inside the smallest L2 on any supported RDNA arch, so its
/// pointer-chase latency is an unambiguous CACHE-hit reference.
const CACHE_RESIDENT_BYTES: usize = 512 * 1024;

/// Minimum ratio by which the DRAM-sized chase must exceed the cache-resident
/// chase to count as genuinely DRAM-resident.
const RESIDENCY_MIN_RATIO: f64 = 1.5;
```
(`STRIDE_BYTES`, `STRIDE_WORDS`, `WARMUP_HOPS`, `PERMUTATION_SEED`, `KERNEL_SRC`, `XorShift64`, `sattolo_shuffle` all stay.)

Edit 3 — replace the entire `fn main()` (lines 115-244) with:
```rust
fn main() {
    let gpu = Gpu::init().unwrap_or_else(|e| {
        eprintln!("Gpu::init() failed: {e}. pointer_chase_latency requires a live GPU.");
        std::process::exit(1);
    });
    println!("arch: {}", gpu.arch);

    let vram_bytes = gpu.hip.get_vram_info().map(|(_, total)| total as u64).unwrap_or(0);
    let cap = GpuCapability::detect(&gpu.arch, vram_bytes);

    // Per-arch DRAM-resident buffer size (issue #490): >= 4x(L2+IC).
    let dram_bytes = chip_profile::pointer_chase_buffer_bytes(&gpu.arch);
    let onchip_mib = cap.l2_cache_mb + cap.infinity_cache_mb;
    println!(
        "onchip_cache_mib (L2+IC): {onchip_mib:.1}  dram_buffer_mib: {}  cache_ref_kib: {}",
        dram_bytes / (1024 * 1024), CACHE_RESIDENT_BYTES / 1024
    );

    let mut compiler = KernelCompiler::new(&gpu.arch, String::new()).unwrap_or_else(|e| {
        eprintln!("KernelCompiler::new failed: {e}"); std::process::exit(1);
    });
    let obj_path = compiler.compile("pointer_chase_latency", KERNEL_SRC).unwrap_or_else(|e| {
        eprintln!("kernel compile failed: {e}"); std::process::exit(1);
    });
    let obj_path_str = obj_path.to_str().unwrap().to_string();
    let module = gpu.hip.module_load(&obj_path_str).expect("module_load failed");
    let func = gpu.hip.module_get_function(&module, "pointer_chase")
        .expect("module_get_function(pointer_chase) failed");

    let measure = |buffer_bytes: usize| -> f64 {
        let num_entries = buffer_bytes / STRIDE_BYTES;
        let hops: u32 = (num_entries * 2) as u32;

        let mut perm: Vec<u32> = (0..num_entries as u32).collect();
        sattolo_shuffle(&mut perm, PERMUTATION_SEED);
        let mut host_buf = vec![0u32; num_entries * STRIDE_WORDS as usize];
        for (i, &next) in perm.iter().enumerate() {
            host_buf[i * STRIDE_WORDS as usize] = next;
        }
        let host_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(host_buf.as_ptr() as *const u8, buffer_bytes) };

        let d_buf = gpu.hip.malloc(buffer_bytes).expect("malloc(buf) failed");
        gpu.hip.memcpy_htod(&d_buf, host_bytes).expect("memcpy_htod(buf) failed");
        let d_out = gpu.hip.malloc(std::mem::size_of::<u64>()).expect("malloc(out) failed");

        let launch = |hops: u32| {
            let mut d_buf_ptr = d_buf.as_ptr();
            let mut start_val: u32 = 0;
            let mut hops_val: u32 = hops;
            let mut stride_val: u32 = STRIDE_WORDS;
            let mut d_out_ptr = d_out.as_ptr();
            let mut params: Vec<*mut c_void> = vec![
                &mut d_buf_ptr as *mut _ as *mut c_void,
                &mut start_val as *mut _ as *mut c_void,
                &mut hops_val as *mut _ as *mut c_void,
                &mut stride_val as *mut _ as *mut c_void,
                &mut d_out_ptr as *mut _ as *mut c_void,
            ];
            unsafe {
                gpu.hip.launch_kernel(&func, [1, 1, 1], [1, 1, 1], 0, None, &mut params)
                    .expect("kernel launch failed");
            }
        };

        launch(WARMUP_HOPS);
        gpu.hip.device_synchronize().expect("device_synchronize (warmup) failed");

        profile::start();
        let timer = profile::begin_timer(&gpu.hip, "pointer_chase", "pointer_chase", buffer_bytes);
        launch(hops);
        profile::end_timer(&gpu.hip, timer).expect("end_timer failed");
        let entries = profile::stop().unwrap_or_default();
        let elapsed_us = entries.last().map(|e| e.time_us)
            .expect("no profile entry recorded — begin_timer/end_timer mismatch");

        let mut out_bytes = [0u8; 8];
        gpu.hip.memcpy_dtoh(&mut out_bytes, &d_out).expect("memcpy_dtoh(out) failed");
        std::hint::black_box(u64::from_ne_bytes(out_bytes));

        gpu.hip.free(d_buf).expect("free(buf) failed");
        gpu.hip.free(d_out).expect("free(out) failed");

        elapsed_us * 1000.0 / hops as f64
    };

    let cache_ns = measure(CACHE_RESIDENT_BYTES);
    let dram_ns = measure(dram_bytes);

    println!("cache_resident_latency_ns: {cache_ns:.3}");
    println!("mem_latency_ns: {dram_ns:.3}");
    println!("dram_over_cache_ratio: {:.3} (guard requires >= {RESIDENCY_MIN_RATIO})", dram_ns / cache_ns);

    chip_profile::check_residency(cache_ns, dram_ns, RESIDENCY_MIN_RATIO).unwrap_or_else(|e| {
        eprintln!("RESIDENCY GUARD FAILED (mem_latency WITHHELD, not shipped): {e}");
        std::process::exit(1);
    });

    let sclk_mhz = cap.boost_clock_mhz;
    let mem_latency_cycles = dram_ns * sclk_mhz as f64 / 1000.0;
    println!("sclk_mhz: {sclk_mhz}");
    println!("mem_latency_cycles: {mem_latency_cycles:.2}");

    assert!(
        (50.0..2000.0).contains(&dram_ns),
        "mem_latency_ns={dram_ns:.3} outside plausible DRAM-latency bound [50, 2000) ns"
    );
}
```

- [ ] **D2. Build clean.** `cargo build --release -p rdna-compute --example pointer_chase_latency` (no GPU needed to build; HIP dlopen'd at runtime).

- [ ] **D3. Workspace + no-GPU CI stay green.** `cargo build --release --workspace --all-targets --locked` and `cargo test -p rdna-compute --lib chip_profile`.

- [ ] **D4. Commit.**
```
git add crates/rdna-compute/examples/pointer_chase_latency.rs && git commit -m "fix(oracle): wire per-arch buffer + residency guard into pointer_chase_latency [#490]

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Sub-task E — re-measure & promote WITHHELD rows `[GPU: hipx]` (follow-up)

Requires the hipx zoo box (gfx1030 RX 6950 XT, gfx1100 RX 7900 XTX) and hiptrx (gfx1201 R9700). Coordinate the per-card GPU lock.

Per arch, pinned + WARM + `high`:
1. Force-high DPM (`echo high | sudo tee /sys/class/drm/card<N>/device/power_dpm_force_performance_level`; record `mclk`/`sclk`).
2. Pin the card via `HIP_VISIBLE_DEVICES=<hip-ordinal>` (ROCR enumeration).
3. Run `HIP_VISIBLE_DEVICES=<ord> cargo run --release -p rdna-compute --example pointer_chase_latency`; confirm printed `dram_buffer_mib` (gfx1030=528, gfx1100=408, gfx1201=272) and NO `RESIDENCY GUARD FAILED`. A guard failure → bump `SAFETY_MULT` and file a note; do NOT hand-enter a number.
4. Take the WARM median of 3-5 fresh-process runs. Record `mem_latency_ns` (median), `dram_over_cache_ratio`, DPM state.

**Promotion (per arch):** set `mem_latency_ns` back to the median in `tests/chip-profiles/<arch>.json`, update `_note` to the measurement provenance, flip that arch's arm in `load_all_committed_chip_profiles` from `None` to `Some(<value>)`, re-run the corpus test, commit per arch, then `gpu_release`. Until re-measured, they legitimately stay WITHHELD.

*[verified: corrected per adversarial review — added Sub-task A0 explicit `use crate::profiler::GpuCapability;` inside `mod tests`, since the parent module's private `use` alias is not reliably re-exported by `use super::*` (the flagged compile-time root cause).]*

**Swarm note:** shares `chip_profile.rs` + `tests/chip-profiles/*.json` with 0a2/0a3 — serialize or isolate-in-worktree; land 0a1 first.

---

## Task 0a2 — Fix #491: `simds_per_cu` in `latency_score`

**Goal (spec §4a, HIGH item):**
> `latency_score` drops `simds_per_cu` (=2 on all RDNA) → a 2× error biased toward Latency. **Fix:** add `simds_per_cu` to `ChipProfile`, multiply it into `available_bytes`, add a numeric `latency_score` regression test.

**Root cause (`crates/rdna-compute/src/roofline.rs:163-183`):** `latency_score` computes `let waves_in_flight = kprofile.occupancy_waves as f64 * chip.cu_count as f64;`. `occupancy_waves` is **per-SIMD** (`profiler.rs:449-452`). Chip-wide waves = `occupancy_waves × cu_count × simds_per_cu`; dropping the `simds_per_cu` factor (=2 on every RDNA) undercounts `available_bytes` by 2×, so the Little's-Law latency score comes out exactly 2× too high, biasing `BoundClass::Latency`. `ChipProfile` has **no** `simds_per_cu` field; the value exists in `profiler::arch_spec` (= 2 RDNA1–4, = 4 GCN5) via `GpuCapability::static_capability(arch)`.

### Files

- **Modify:** `crates/rdna-compute/src/chip_profile.rs` — add `pub simds_per_cu: u32`; set it in `for_unprofiled`; emit/parse in `to_json`/`from_json` (arch-derived back-fill, reject bogus 0); add a `verify_live` mismatch branch; + plumbing tests.
- **Modify:** `crates/rdna-compute/src/roofline.rs` — multiply `chip.simds_per_cu` into `waves_in_flight`; + numeric regression test.
- **Test:** rdna-compute **lib** unit tests — no GPU.

**No committed `tests/chip-profiles/*.json` edits** (from_json derives `simds_per_cu` when absent, avoiding clobber with 0a1).

### Interfaces

**Consumes:** `GpuCapability { pub simds_per_cu: u32 }` (`profiler.rs:12`); `GpuCapability::static_capability(arch)` (`profiler.rs:251`, `pub(crate)`); `KernelProfile { pub occupancy_waves: u32 }` (per-SIMD).
**Produces:** `ChipProfile { pub simds_per_cu: u32, /* … */ }`; corrected `latency_score` body.

---

- [ ] **(1) Write the failing numeric regression test.** Append inside `#[cfg(test)] mod tests` of `roofline.rs`, after `real_fixture_binds_valu_issue_under_committed_chip` (before the `synthetic_kprofile()` helper):

```rust
    #[test]
    fn latency_score_numeric_littles_law_includes_simds_per_cu() {
        // Regression for #491. Fully-controlled inputs:
        //   required_bytes  = 512*1e9 * 200e-9              =    102_400 B
        //   bytes_per_wave  = 10 * 16 * 32                  =      5_120 B
        //   waves_in_flight = 8 * 64 * 2                    =      1_024
        //   available_bytes = 5_120 * 1_024                 =  5_242_880 B
        //   latency_score   = 102_400 / 5_242_880          = 0.019_531_25 (5/256)
        // The pre-#491 formula (missing * simds_per_cu) gives 0.039_062_5 (5/128, 2x).
        let hist = IsaHistogram {
            v_bfe: 0, v_cvt: 0, f32_fma: 0, v_dot4: 0, v_wmma: 0,
            global_load_b128: 10, s_delay_alu: 0, s_wait: 0,
            vmem_valu_ratio: 1.0, private_segment_red: false,
        };
        let kprofile = synthetic_kprofile(); // occupancy_waves = 8

        let mut chip = crate::chip_profile::ChipProfile::for_unprofiled("gfx1201");
        chip.cu_count = 64;
        chip.wavefront_size = 32;
        chip.simds_per_cu = 2;
        chip.peak_bw_gbps = Some(512.0);
        chip.mem_latency_ns = Some(200.0);

        let roofline = Roofline::analyze(&hist, &kprofile, &chip, None);

        assert!(
            (roofline.latency - 0.019_531_25).abs() < 1e-9,
            "latency_score must equal 5/256 = 0.019_531_25; pre-#491 yields 0.039_062_5. got {}",
            roofline.latency
        );
    }
```
> Note: if 0a3's `IsaHistogram` field (`vmem_load_bytes_per_lane`) has already landed, add `vmem_load_bytes_per_lane: 160,` (= `global_load_b128 * 16`) to this literal to keep the arithmetic identical. Since 0a2 lands before 0a3, the field is absent here.

- [ ] **(2) Run → expect FAIL (compile):** `cargo test -p rdna-compute --lib latency_score_numeric_littles_law_includes_simds_per_cu` → `error[E0609]: no field 'simds_per_cu' on type 'ChipProfile'`.

- [ ] **(3) Add the field to `ChipProfile`.** In `chip_profile.rs`:

Struct (after `cu_count`):
```rust
    /// SIMD units per compute unit — 2 on every RDNA arch (RDNA1-4), 4 on
    /// GCN5 (gfx906). Chip-wide waves-in-flight is
    /// `occupancy_waves * cu_count * simds_per_cu`; omitting this factor
    /// inflates the Little's-Law latency score by `simds_per_cu` (#491).
    /// Sourced from `profiler::arch_spec` via `GpuCapability::static_capability`.
    pub simds_per_cu: u32,
```
`for_unprofiled` (after `cu_count: static_cap.cu_count,`): `simds_per_cu: static_cap.simds_per_cu,`.
`to_json` (after `"cu_count": self.cu_count,`): `"simds_per_cu": self.simds_per_cu,`.
`from_json` — parse after `cu_count`, **rejecting a bogus 0** (a zero would zero `available_bytes` → bogus `latency=1.0`):
```rust
        // Optional: committed rows authored before #491 lack it. Derive from
        // arch_spec when absent OR when a bogus 0 is present (0 would drive
        // available_bytes to 0 and force a spurious latency=1.0). NEVER 0.
        let simds_per_cu = v["simds_per_cu"]
            .as_u64()
            .map(|x| x as u32)
            .filter(|&x| x > 0)
            .unwrap_or_else(|| GpuCapability::static_capability(&arch).simds_per_cu);
```
Return struct (after `cu_count,`): `simds_per_cu,`.

- [ ] **(3b) Add the `verify_live` mismatch branch (adversarial-review fix).** `ChipProfile::verify_live` is the static-field honesty guard; add `simds_per_cu` alongside its existing static-field comparisons (adapt to the file's actual mismatch-collection idiom — it compares fields against `GpuCapability::static_capability(&self.arch)` and pushes mismatch strings):
```rust
        if self.simds_per_cu != static_cap.simds_per_cu {
            mismatches.push(format!(
                "simds_per_cu: profile {} != arch_spec {}",
                self.simds_per_cu, static_cap.simds_per_cu
            ));
        }
```

- [ ] **(4) Run → expect FAIL (value):** compiles now, assert fires — `... got 0.0390625`.

- [ ] **(5) Fix `latency_score`.** Replace the `waves_in_flight` block (`roofline.rs:173-178`):
```rust
    // `occupancy_waves` is PER-SIMD, so chip-wide waves-in-flight is
    // `occupancy_waves * cu_count * simds_per_cu`. Dropping the `simds_per_cu`
    // factor (=2 on all RDNA) undercounted `available_bytes` and inflated this
    // latency score toward a spurious Latency verdict — GitHub #491.
    let waves_in_flight =
        kprofile.occupancy_waves as f64 * chip.cu_count as f64 * chip.simds_per_cu as f64;
    let available_bytes = bytes_per_wave * waves_in_flight;
```

- [ ] **(6) Run → expect PASS.** `cargo test -p rdna-compute --lib latency_score_numeric_littles_law_includes_simds_per_cu`.

- [ ] **(7) Add the `ChipProfile` plumbing + JSON tests.** Append to `chip_profile.rs` `mod tests` (after `from_json_missing_cache_fields_withholds`):

```rust
    #[test]
    fn simds_per_cu_is_arch_derived_and_round_trips() {
        assert_eq!(ChipProfile::for_unprofiled("gfx1201").simds_per_cu, 2, "RDNA4: 2");
        assert_eq!(ChipProfile::for_unprofiled("gfx1100").simds_per_cu, 2, "RDNA3: 2");
        assert_eq!(ChipProfile::for_unprofiled("gfx906").simds_per_cu, 4, "GCN5: 4 (arch-derived)");

        // A pre-#491 row lacking the key: derive from arch_spec, never fabricate.
        let mut json = ChipProfile::for_unprofiled("gfx906").to_json();
        assert_eq!(json["simds_per_cu"].as_u64(), Some(4), "to_json emits the field");
        json.as_object_mut().unwrap().remove("simds_per_cu");
        let derived = ChipProfile::from_json(&json).expect("must parse a row missing simds_per_cu");
        assert_eq!(derived.simds_per_cu, 4, "from_json derives from arch_spec when absent");

        let profile = ChipProfile::load_committed("gfx1201").expect("gfx1201.json must load");
        let reloaded = ChipProfile::from_json(&profile.to_json()).expect("round-trip must parse");
        assert_eq!(profile, reloaded);
        assert_eq!(reloaded.simds_per_cu, 2);
    }

    #[test]
    fn from_json_rejects_bogus_zero_simds_per_cu() {
        // A committed 0 is impossible hardware and would force latency=1.0 —
        // from_json must DERIVE the real value from arch_spec, never keep 0.
        let mut json = ChipProfile::for_unprofiled("gfx1201").to_json();
        json.as_object_mut().unwrap()
            .insert("simds_per_cu".to_string(), serde_json::json!(0));
        let parsed = ChipProfile::from_json(&json).expect("must parse and correct a bogus 0");
        assert_eq!(parsed.simds_per_cu, 2, "a 0 must be corrected to the arch-derived 2, never kept");
    }

    #[test]
    fn verify_live_flags_simds_per_cu_mismatch() {
        // The static-field honesty guard must catch a doctored simds_per_cu.
        let mut profile = ChipProfile::for_unprofiled("gfx1201"); // simds_per_cu = 2
        assert!(profile.verify_live().is_ok(), "an untouched arch-derived profile verifies");
        profile.simds_per_cu = 3; // physically impossible on RDNA4
        assert!(
            profile.verify_live().is_err(),
            "verify_live must flag a simds_per_cu that disagrees with arch_spec"
        );
    }
```
> Also update any pre-existing `verify_live_passes_on_exact_match` test to build/compare `simds_per_cu` via `profile.simds_per_cu` rather than a literal, so the exact-match arm covers the new field.

- [ ] **(8) Full crate lib suite → PASS.** `cargo test -p rdna-compute --lib` (existing `real_fixture_binds_valu_issue_under_committed_chip`, `compute_serialized_binds_valu_or_latency`, `round_trip_json*` still pass — the fix only shrinks the sub-dominant latency score; back-fill keeps round-trips exact).

- [ ] **(9) Format.** `./scripts/fmt-changed.sh`.

- [ ] **(10) Commit.**
```
git add crates/rdna-compute/src/chip_profile.rs crates/rdna-compute/src/roofline.rs
git commit -m "fix(oracle): add simds_per_cu to ChipProfile; correct latency_score 2x overcount (#491)

available_bytes was undercounted 2x, inflating the Little's-Law latency score
toward a spurious Latency bound-class. Add simds_per_cu (arch-derived via
arch_spec; from_json back-fills pre-#491 rows and rejects a bogus 0; verify_live
flags mismatches) and multiply it into waves_in_flight. Numeric regression test
asserts 5/256, rejects the old 5/128.

Fixes #491.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

*[verified: corrected per adversarial review — added the `verify_live` simds_per_cu mismatch branch + `verify_live_flags_simds_per_cu_mismatch` test (and updated the exact-match test to use `profile.simds_per_cu`); hardened `from_json` to reject/derive on a bogus `simds_per_cu: 0` with the `from_json_rejects_bogus_zero_simds_per_cu` test — the review's "accepts 0 → bogus latency=1.0" and "no verify_live comparison" gaps.]*

**Swarm note:** code-independent of 0a1; `parallel_safe` is false only because 0a3 also edits `roofline.rs`'s `latency_score`. Safe to run in parallel with 0a1 in isolation; land before 0a3.

---

## Task 0a3 — MED hardening round-up (roofline + provenance + sysfs correctness)

**Spec:** §4a (MED round-up) + §5.3 risk register. Lands the accuracy-hardening MED fix-list so each dynamic roofline cell can be honestly promoted or withheld per §5.5.

**Branch:** `feat/rdna-kernel-oracle`. **Depends on:** 0a1 (#490) and 0a2 (#491). **All tests are no-GPU `rdna-compute` lib tests**; the one live-sysfs wiring is `[GPU: hipx]`. Diagnostic/instrument modules only — no kernel/forward-pass/dispatch source changes.

### Sub-step 0 — PREFLIGHT (adversarial-review fix; run FIRST)

- [ ] **P1. Refuse to apply on an unfixed tree.** This task composes on the bodies 0a1/#490 and 0a2/#491 leave behind. Before any sub-step, assert both HIGH fixes are present, and abort if not:
```
# #490: per-arch buffer sizing exists
grep -q 'pub fn pointer_chase_buffer_bytes' crates/rdna-compute/src/chip_profile.rs \
  || { echo "PREFLIGHT FAIL: 0a1/#490 not landed (pointer_chase_buffer_bytes missing)"; exit 1; }
# #491: simds_per_cu is in ChipProfile AND multiplied into latency_score
grep -q 'pub simds_per_cu: u32' crates/rdna-compute/src/chip_profile.rs \
  || { echo "PREFLIGHT FAIL: 0a2/#491 not landed (ChipProfile.simds_per_cu missing)"; exit 1; }
grep -q 'chip.simds_per_cu' crates/rdna-compute/src/roofline.rs \
  || { echo "PREFLIGHT FAIL: 0a2/#491 not landed (latency_score missing simds_per_cu factor)"; exit 1; }
```
If any check fails, STOP — 0a1/0a2 must land first (this is the flagged "no preflight stops applying on an unfixed tree" gap).

**Scope boundary (adversarial-review fix):** This task does **NOT** implement §4b in-model Percentage surfacing (owned by **0b**) nor the §4c per-arch PMC derived-vs-raw census (owned by **0c**). Sub-steps E stamps DPM provenance and threads it through the real `run_dynamic`/CLI path (below); the census's derived-vs-accumulator classification is 0c's deliverable. No claim here covers 0b/0c work.

Eight labelled sub-steps (A–H), applied in order.

### Sub-step A — derive `bound_class` at query; make `diff` compare it; kill the defaulted tag
**Modify:** `kernel_ledger.rs`, `examples/kernel_perf_instrument.rs`.

- [ ] **A1. Failing tests** (append to `kernel_ledger.rs::tests`):
```rust
    #[test]
    fn diff_flags_bound_class_flip_as_hard_regression() {
        let committed = sample_row(); // BoundClass::ValuIssue
        let mut current = committed.clone();
        current.bound_class = BoundClass::Bandwidth;
        let deltas = KernelLedger::diff(&committed, &current);
        assert!(
            deltas.iter().any(|d| matches!(
                d, LedgerDelta::RegressionHard { field, .. } if field == "bound_class"
            )),
            "a bound_class flip must be a hard regression, got {deltas:?}"
        );
    }

    #[test]
    fn from_fixture_uses_caller_derived_bound_class_not_a_default() {
        let path = fixture_dir().join("gemv_hfq4g256_moe_gate_up_indexed_batched.hsaco");
        let hist = IsaHistogram::from_hsaco(&path, "gfx1201").expect("disasm");
        let mut map = std::collections::HashMap::new();
        map.insert("probe".to_string(), path.clone());
        let (_c, profiles) = crate::profiler::profile_kernels("gfx1201", 0, &map);
        let kp = profiles.into_iter().next().expect("profile");
        let row = LedgerRow::from_fixture(
            sample_row().key, &hist, &kp, BoundClass::Latency, Reproducer::default(),
        );
        assert_eq!(row.bound_class, BoundClass::Latency, "must store the caller-derived class");
    }
```
- [ ] **A2. Run → FAIL:** `cargo test -p rdna-compute --lib kernel_ledger::` — first test compiles-but-fails (diff ignores bound_class); second fails to compile (4-arg `from_fixture`).
- [ ] **A3. Implement.** Change `from_fixture` signature to take `bound_class: BoundClass` and store it (no `ValuIssue` default); in `KernelLedger::diff` after the `isa_fingerprint` hard_check add:
```rust
        if committed.bound_class != current.bound_class {
            deltas.push(LedgerDelta::RegressionHard {
                field: "bound_class".to_string(),
                committed: bound_class_to_str(committed.bound_class).to_string(),
                current: bound_class_to_str(current.bound_class).to_string(),
            });
        }
```
In `kernel_perf_instrument.rs::measure_fixture`, delete `row.bound_class = roofline.binding;` and pass `roofline.binding` into `from_fixture`.
- [ ] **A4. Run → PASS** + `cargo build -p rdna-compute --example kernel_perf_instrument --locked`.
- [ ] **A5. Commit** `fix(oracle): diff compares bound_class; kill from_fixture default tag` (+ Co-Authored-By trailer).

### Sub-step B — propagate `AchievedBw.trust_score` into `Roofline`
**Modify:** `roofline.rs`, `examples/kernel_perf_instrument.rs`. Note (adversarial-review): `AchievedBw` **already** carries `trust_score`; B threads that existing trust into `Roofline`, reserving 1.0 for a live PMC read.

- [ ] **B1. Failing test** (`roofline.rs::tests`):
```rust
    #[test]
    fn trust_score_propagates_from_measured_bw_never_fabricates_one() {
        let hist = IsaHistogram {
            v_bfe: 42, v_cvt: 56, f32_fma: 98, v_dot4: 0, v_wmma: 0,
            global_load_b128: 14, s_delay_alu: 49, s_wait: 60,
            vmem_valu_ratio: 29.0 / 281.0, private_segment_red: false,
        };
        let kprofile = synthetic_kprofile();
        let chip = synthetic_chip(Some(800.0), Some(280.0));
        let r = Roofline::analyze(&hist, &kprofile, &chip, Some(MeasuredBw { gbps: 120.0, trust: 0.6 }));
        assert_eq!(r.trust_score, 0.6, "trust must propagate, got {}", r.trust_score);
        let s = Roofline::analyze(&hist, &kprofile, &chip, None);
        assert_eq!(s.trust_score, 0.5, "static path floor");
    }
```
- [ ] **B2. Run → FAIL** (compile: `MeasuredBw` undefined, `analyze` takes `Option<f64>`).
- [ ] **B3. Implement.** Add `pub struct MeasuredBw { pub gbps: f64, pub trust: f64 }`; change `analyze(..., measured: Option<MeasuredBw>)`; `let bw = bw_score(measured.map(|m| m.gbps), chip.peak_bw_gbps);`; replace `trust_score` with `match measured { Some(m) => m.trust, None => 0.5 }`. Fix the four existing `Some(f64)` test callsites to `Some(MeasuredBw { gbps: …, trust: 0.8 })`. In `kernel_perf_instrument.rs`: `use rdna_compute::roofline::{Roofline, MeasuredBw};`, change `measure_fixture` param to `measured: Option<MeasuredBw>`, and in `run_dynamic` change `Some(entry.achieved_gbps)` → `Some(MeasuredBw { gbps: entry.achieved_gbps, trust: entry.trust_score })` (using the existing `AchievedBw.trust_score`).
- [ ] **B4. Run → PASS** + example build.
- [ ] **B5. Commit** `fix(oracle): propagate AchievedBw trust into Roofline; 1.0 = live PMC only`.

### Sub-step C — cache-tier working-set accounting
**Modify:** `chip_profile.rs`. Add `BwTiers.effective_cache_working_set_bytes: Option<u64>` + `BwTiers::validate_cache_residency(&self, l2_plus_ic_mib: f64) -> Result<(), String>`.

- [ ] **C1. Failing tests** (`chip_profile.rs::tests`):
```rust
    #[test]
    fn cache_working_set_within_l2_ic_passes() {
        let points = [ BwSweepPoint { mib: 16.0, gbps: 952.0 },
                       BwSweepPoint { mib: 64.0, gbps: 229.0 },
                       BwSweepPoint { mib: 128.0, gbps: 230.0 } ];
        let tiers = detect_bw_tiers(&points, 3.0).unwrap();
        assert_eq!(tiers.effective_cache_working_set_bytes, Some(16 * 1024 * 1024));
        assert!(tiers.validate_cache_residency(68.0).is_ok(), "16 MiB ws fits 68 MiB L2+IC");
    }
    #[test]
    fn cache_working_set_exceeding_l2_ic_fails_loud() {
        let points = [ BwSweepPoint { mib: 128.0, gbps: 1500.0 },
                       BwSweepPoint { mib: 256.0, gbps: 230.0 },
                       BwSweepPoint { mib: 512.0, gbps: 231.0 } ];
        let tiers = detect_bw_tiers(&points, 3.0).unwrap();
        assert_eq!(tiers.effective_cache_working_set_bytes, Some(128 * 1024 * 1024));
        assert!(tiers.validate_cache_residency(68.0).is_err(), "128 MiB ws over 68 MiB L2+IC must fail");
    }
```
- [ ] **C2. Run → FAIL** (field/method missing).
- [ ] **C3. Implement.** Add the field to `BwTiers`; set `effective_cache_working_set_bytes = effective_cache_mib.map(|m| (m * 1024.0 * 1024.0) as u64)` in `detect_bw_tiers`; add the `impl BwTiers { pub fn validate_cache_residency(...) }` that returns `Ok(())` when no cache tier, else `Err` if `ws > l2_plus_ic_mib*MiB`.
- [ ] **C4. Run → PASS** (existing `detect_bw_tiers_*` tests use field access only).
- [ ] **C5. Commit** `fix(oracle): report cache working-set bytes; assert ws <= L2+IC`.

### Sub-step D — resolve sysfs reads by the HIP ordinal (zoo-box correctness; ALL sysfs readers)
**Modify:** `profiler.rs`. Add pure `hip_ordinal_to_physical`, `pick_amd_card`, `visible_devices_env`.

- [ ] **D1. Failing tests** (`profiler.rs`):
```rust
#[cfg(test)]
mod ordinal_tests {
    use super::{hip_ordinal_to_physical, pick_amd_card};
    #[test]
    fn hip_ordinal_identity_without_visible_env() {
        assert_eq!(hip_ordinal_to_physical(0, None), Some(0));
        assert_eq!(hip_ordinal_to_physical(3, None), Some(3));
    }
    #[test]
    fn hip_ordinal_remaps_through_visible_devices() {
        assert_eq!(hip_ordinal_to_physical(0, Some("2,0,1")), Some(2));
        assert_eq!(hip_ordinal_to_physical(1, Some("2,0,1")), Some(0));
        assert_eq!(hip_ordinal_to_physical(2, Some("2, 0, 1")), Some(1));
        assert_eq!(hip_ordinal_to_physical(3, Some("2,0,1")), None);
    }
    #[test]
    fn pick_amd_card_selects_nth_amd_ignoring_non_amd() {
        let cards = vec![
            ("card0".to_string(), "0x8086".to_string()),
            ("card2".to_string(), "0x1002".to_string()),
            ("card1".to_string(), "0x1002".to_string()),
        ];
        assert_eq!(pick_amd_card(&cards, 0).as_deref(), Some("card1"));
        assert_eq!(pick_amd_card(&cards, 1).as_deref(), Some("card2"));
        assert_eq!(pick_amd_card(&cards, 2), None);
    }
}
```
- [ ] **D2. Run → FAIL** (functions missing).
- [ ] **D3. Implement.** Add `visible_devices_env`, `hip_ordinal_to_physical`, `pick_amd_card` (as drafted). Then rewrite the DRM-card selection **in ALL THREE sysfs readers — `read_sysfs_cu_count`, `read_sysfs_clocks`, `read_sysfs_bus_width`** (adversarial-review fix: Risk 6 is CU-count/VRAM/bus-width on zoo boxes, not just clocks) — to enumerate `(name, vendor)` pairs and select the pinned card via `pick_amd_card(&pairs, hip_ordinal_to_physical(0, visible_devices_env().as_deref())?)`. Factor the enumeration into one shared `fn pinned_amd_card() -> Option<String>` helper so all three readers resolve the SAME pinned DRM node (and therefore the same BDF), then read `.../device/{gpu_busy_percent,pp_dpm_sclk,...}`, `.../device/mem_info_*` and the bus-width sysfs off that one node.
- [ ] **D4. Run → PASS** (no-GPU, blocking) `cargo test -p rdna-compute --lib ordinal_tests`.
- [ ] **D4b. `[GPU: hipx]` live wiring check.** On the zoo box: `HIP_VISIBLE_DEVICES=2 cargo run -p rdna-compute --example dump_chip_profile` must report the sclk/mclk **and cu_count/bus-width/VRAM** of the *pinned* card (cross-check `rocminfo` + `rocm-smi -d 2 --showclocks --showmeminfo vram`); before this fix all three tracked the first AMD card regardless of pin.
- [ ] **D5. Commit** `fix(oracle): resolve ALL sysfs reads (cu/clocks/bus/vram) by HIP ordinal on zoo boxes`.

### Sub-step E — stamp DPM state; refuse `profile_standard`-captured timings; thread DPM through the REAL path
**Modify:** `profile_rocprof.rs`, `kernel_ledger.rs`, `examples/kernel_perf_instrument.rs`.

- [ ] **E1. Failing tests** (`profile_rocprof.rs::tests` + `kernel_ledger.rs::tests`):
```rust
    #[test]
    fn compute_achieved_bw_refuses_profile_standard_timings() {
        let mut byte_model = HashMap::new();
        byte_model.insert("gemv_hfq4g256_residual".to_string(), 1_000usize);
        let rocprof = vec![one_call_kernel("_Z22gemv_hfq4g256_residualPKjS0_Pfi.kd", 10.0)];
        let out = compute_achieved_bw(&rocprof, &byte_model, &gfx1201_chip(), Some("profile_standard"));
        assert!(out.is_empty(), "profile_standard timings are clock-deflated — must be refused, got {out:?}");
    }
    #[test]
    fn compute_achieved_bw_stamps_dpm_state_on_high() {
        let mut byte_model = HashMap::new();
        byte_model.insert("gemv_hfq4g256_residual".to_string(), 1_000usize);
        let rocprof = vec![one_call_kernel("_Z22gemv_hfq4g256_residualPKjS0_Pfi.kd", 10.0)];
        let out = compute_achieved_bw(&rocprof, &byte_model, &gfx1201_chip(), Some("high"));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].dpm_state.as_deref(), Some("high"));
    }
    // kernel_ledger.rs::tests
    #[test]
    fn ledger_row_round_trips_dpm_state() {
        let mut row = sample_row();
        row.dpm_state = Some("high".to_string());
        let back = LedgerRow::from_atlas_row(&row.to_atlas_row()).expect("round-trip");
        assert_eq!(back.dpm_state.as_deref(), Some("high"));
    }
```
- [ ] **E2. Run → FAIL** (3-arg `compute_achieved_bw`; no `dpm_state`).
- [ ] **E3. Implement.** Add `AchievedBw.dpm_state: Option<String>`; add `is_profile_standard(dpm)`; change `compute_achieved_bw(..., dpm_state: Option<&str>)` — return `Vec::new()` (with a WARN) when `is_profile_standard`, else stamp `dpm_state`. Add `LedgerRow.dpm_state` persisted through `to_atlas_row`/`from_atlas_row`/`sample_row`/`from_fixture`.
- [ ] **E3b. Thread DPM through the REAL path (adversarial-review fix).** `run_dynamic` must not pass a blanket `None` — add a CLI/provenance source for the capture's DPM level and thread it into `compute_achieved_bw`, persist it on the emitted rows, and TEST that path (not just the helper):
  - Add `--dpm-state <level>` to `Mode::Dynamic` parse (default `high`; the sweep harness in 0e passes the level it forced).
  - In `run_dynamic`, call `compute_achieved_bw(&rocprof, &byte_model, chip, Some(&args.dpm_state))` and add `"dpm_state": entry.dpm_state` to the per-kernel JSON.
  - Add a no-GPU CLI smoke asserting a CSV replayed with `--dpm-state profile_standard` emits **zero** achieved-BW rows (refused), while `--dpm-state high` emits them stamped `"dpm_state":"high"`.
- [ ] **E4. Run → PASS** + example build.
- [ ] **E5. Commit** `fix(oracle): stamp DPM state; refuse profile_standard achieved-BW; thread DPM through run_dynamic/CLI`.

### Sub-step F — sum ALL VMEM load widths in the latency short-circuit
**Modify:** `isa_histogram.rs`, `roofline.rs`. Add `IsaHistogram.vmem_load_bytes_per_lane: u32`.

- [ ] **F1. Failing tests** (`isa_histogram.rs::tests` + `roofline.rs::tests`):
```rust
    // isa_histogram
    #[test]
    fn from_disassembly_sums_all_vmem_load_widths() {
        let text = "_Z5probe:\n\tglobal_load_b32 v0, v1, s[0:1]\n\tglobal_load_b64 v[2:3], v4, s[0:1]\n\tglobal_load_b128 v[5:8], v9, s[0:1]\n\tbuffer_load_dwordx2 v[10:11], v12, s[0:3], 0 offen\n";
        let hist = IsaHistogram::from_disassembly(text);
        assert_eq!(hist.vmem_load_bytes_per_lane, 4 + 8 + 16 + 8, "sum b32+b64+b128+dwordx2");
        assert_eq!(hist.global_load_b128, 1, "b128 still counted separately");
    }
    // roofline
    #[test]
    fn latency_not_short_circuited_when_only_narrow_loads() {
        let hist = IsaHistogram {
            v_bfe: 0, v_cvt: 0, f32_fma: 1, v_dot4: 0, v_wmma: 0,
            global_load_b128: 0, s_delay_alu: 0, s_wait: 0,
            vmem_valu_ratio: 0.5, vmem_load_bytes_per_lane: 8, private_segment_red: false,
        };
        let chip = synthetic_chip(Some(800.0), Some(280.0));
        let r = Roofline::analyze(&hist, &synthetic_kprofile(), &chip, None);
        assert!(r.latency > 0.0, "narrow-load kernel must not be latency-short-circuited, got {}", r.latency);
    }
```
- [ ] **F2. Run → FAIL** (field missing).
- [ ] **F3. Implement.** Add `vmem_load_bytes_per_lane` (NOT part of `isa_fingerprint`); add `vmem_load_width_bytes(op)`; accumulate in `from_disassembly`. In `latency_score` replace `bytes_per_wave` with `hist.vmem_load_bytes_per_lane as f64 * chip.wavefront_size as f64`. Patch the FIVE existing `IsaHistogram { .. }` literals in `roofline.rs::tests` to add `vmem_load_bytes_per_lane: <global_load_b128 * 16>` so their latency arithmetic is byte-identical.
- [ ] **F4. Run → PASS** `cargo test -p rdna-compute --lib isa_histogram:: roofline::`.
- [ ] **F5. Commit** `fix(oracle): latency short-circuit sums all VMEM load widths`.

### Sub-step G — word-boundary coverage matching
**Modify:** `profile_rocprof.rs`. Add `covered_by_alias(mangled_lower, alias_lower) -> bool` (min length 6 + token-start boundary) used by `compute_coverage`.

- [ ] **G1. Failing test** `coverage_uses_word_boundary_not_loose_substring` (as drafted: `norm` must not cover `rmsnorm`; short `gemv` rejected; full `gemm_q8_0_batched` still covers at a digit boundary).
- [ ] **G2. Run → FAIL** (`.contains` still lets `norm` cover `rmsnorm`).
- [ ] **G3. Implement** `MIN_COVERAGE_ALIAS_LEN = 6` + `covered_by_alias` (token-start boundary, digit boundary allowed for Itanium length prefix); use it in `compute_coverage`'s `covered` closure. (Coordinate with 0b's `is_covered` extraction — G and 0b both refactor `compute_coverage`'s matching core; land 0a3 first, then 0b's `is_covered` wraps this boundary rule.)
- [ ] **G4. Run → PASS** (existing `test_coverage_and_blindspot` still passes).
- [ ] **G5. Commit** `fix(oracle): word-boundary + min-length coverage matching`.

### Sub-step H — read `.vgpr_count` from the msgpack note; cross-check; WIRE into the profiling path
**Modify:** `profiler.rs`. Add `vgpr_count_from_note(path, arch) -> Result<Option<u32>, String>` + `cross_check_vgprs(bit_decoded, note) -> Result<u32, String>`.

- [ ] **H1. Failing tests** `cross_check_vgprs_agrees_within_granule`, `cross_check_vgprs_fails_loud_on_disagreement`, `note_vgpr_count_cross_checks_bit_decode_on_gfx1201_fixture` (as drafted).
- [ ] **H2. Run → FAIL** (functions missing).
- [ ] **H3. Implement** `vgpr_count_from_note` (shells `llvm-readelf --notes`, reuses `isa_histogram`'s unbundle helpers) + `cross_check_vgprs` (`note <= decode && decode - note < 8`).
- [ ] **H3b. WIRE into the production path (adversarial-review fix).** So the cross-check actually fails loud: in `profile_hsaco`/`profile_kernels`, after computing the `pgm_rsrc1` bit-decoded `vgprs`, call `vgpr_count_from_note(path, arch)` and `cross_check_vgprs(bit_decoded, note)`; on `Ok(n)` set `KernelProfile.vgprs = n` (exact note count), on `Err` propagate/emit the loud mismatch (do NOT silently keep the granule-rounded decode). Add a test asserting the fixture path returns the exact (note-reconciled) VGPR count, not the granule-rounded value, when they differ.
- [ ] **H4. Run → PASS** `cargo test -p rdna-compute --lib vgpr_note_tests` (shells `/opt/rocm/llvm/bin/llvm-readelf`, present in `no-gpu-ci`).
- [ ] **H5. Commit** `fix(oracle): cross-check pgm_rsrc1 VGPR decode vs msgpack .vgpr_count in the profiling path`.

### Final verification (whole task)
- [ ] `cargo test -p rdna-compute --lib` (all green) + `cargo build -p rdna-compute --example kernel_perf_instrument --locked` + `cargo build --release --workspace --all-targets --locked`.
- [ ] `[GPU: hipx]` confirm sub-step D's live wiring (D4b).

*[verified: corrected per adversarial review — added Sub-step 0 PREFLIGHT that refuses to run on an unfixed tree (missing #490/#491); extended Sub-step D to resolve ALL sysfs readers (cu_count/clocks/bus_width/VRAM) to the pinned DRM node, not just clocks; threaded real DPM provenance through `run_dynamic`/CLI in E (not a blanket `None`) with a path-level test; wired H's `vgpr_count_from_note`/`cross_check_vgprs` into `profile_hsaco`/`profile_kernels` so it fails loud; added an explicit scope boundary disclaiming the §4b in-model-Percentage (owned by 0b) and §4c PMC-census (owned by 0c) work the draft implied.]*

---

## Task 0b — Close the in-model loop (serve path ∩ oracle)

Spec §4b (and the §5.4 "Gaps" it fixes). Four deliverables on branch `feat/rdna-kernel-oracle`. **Depends on 0a1/0a2/0a3** (lands after 0a3 to compose on its `AchievedBw`/`Reproducer`/`kernel_perf_instrument` edits).

- **A.** Surface in-model wall-time-% (`RocprofKernel.percent` → new `AchievedBw.in_model_pct`, emit from `run_dynamic`). Note (adversarial-review): `AchievedBw` **already** has `trust_score` (from prior work / 0a3); add `in_model_pct` **alongside** it — do not re-add or clobber `trust_score`.
- **B.** Stop dropping unmatched rows + wire the blindspot detector via `coverage_against_aliases`.
- **C.** Record + **enforce** structured serve provenance.
- **D+E.** Route dynamic measurement through the real daemon serve path.

Deliverables A–C are no-GPU lib tests. D is no-GPU CLI smoke. E is the one GPU step.

### Files
- **Modify** `crates/rdna-compute/src/profile_rocprof.rs` — `AchievedBw.in_model_pct`; `is_covered`; `AliasCoverage` + `coverage_against_aliases`.
- **Modify** `crates/rdna-compute/src/kernel_ledger.rs` — `Reproducer::require_serve_provenance` (structured) + `is_md5_hex`.
- **Modify** `crates/rdna-compute/examples/kernel_perf_instrument.rs` — `Mode::Dynamic` provenance fields; enforce; emit `in_model_pct` + coverage.
- **Create** `scripts/oracle-serve-dynamic.sh` `[GPU: hipx]`.

### Interfaces
**Consumes:** `compute_achieved_bw(...)` (`profile_rocprof.rs`; note the **4-arg** `dpm_state` form 0a3 introduced); `RocprofKernel.percent: f64` (`:49`); `compute_coverage(...)` — **signature frozen** (also called by `crates/hipfire-runtime/examples/bench_qwen35_mq4.rs` — **corrected path**, the draft's `crates/rdna-compute/examples/bench_qwen35_mq4.rs:493` was wrong); `parse_rocprof_stats_csv_text`; `Reproducer { pub cmd, pub fixture_path, pub prompt_md5 }`; `ChipProfile::load_committed`; daemon JSON-lines protocol; `scripts/rocprof-wrap.sh <dir> -- <cmd>`.
**Produces:** `AchievedBw.in_model_pct: f64`; `AliasCoverage` + `coverage_against_aliases`; `Reproducer::require_serve_provenance(&self) -> Result<(), String>`.

---

### Deliverable A — carry in-model wall-time-% into `AchievedBw`

- [ ] **A1. Failing test** (`profile_rocprof.rs::tests`) — `compute_achieved_bw_carries_in_model_pct_from_rocprof_percentage` (orthogonality of `in_model_pct` vs `pct_of_peak`, as drafted; call `compute_achieved_bw(&rocprof, &byte_model, &chip, Some("high"))` using the 0a3 4-arg form).
- [ ] **A2. Run → FAIL** (`no field 'in_model_pct'`).
- [ ] **A3. Implement.** Add `pub in_model_pct: f64` to `AchievedBw` (after `total_us`, **beside** the existing `trust_score`/`dpm_state`); in the `out.push(AchievedBw { .. })` literal add `in_model_pct: rk.percent,`.
- [ ] **A4. Run → PASS** (`AchievedBw` only constructed inside `profile_rocprof.rs`).
- [ ] **A5. Commit** `feat(oracle): carry in-model wall-time-% (RocprofKernel.percent) into AchievedBw`.

### Deliverable B — reconstruct the true denominator + wire the blindspot detector

- [ ] **B1. Failing tests** `coverage_against_aliases_reconstructs_true_denominator_and_ranks_blindspots` + `coverage_against_aliases_ranks_blindspots_biggest_wall_time_first` (as drafted).
- [ ] **B2. Run → FAIL** (`coverage_against_aliases`/`AliasCoverage` missing).
- [ ] **B3. Implement.** Extract `fn is_covered(rocprof_name, aliases_lower) -> bool` (built on 0a3's `covered_by_alias` boundary rule, so the two entry points share one primitive); refactor `compute_coverage`'s loop to call it; add `pub struct AliasCoverage { rocprof_total_us, covered_total_us, blindspot_total_us, coverage_pct, covered_in_model_pct, blindspots }` + `pub fn coverage_against_aliases(...)` summing EVERY row for the true 100% denominator and ranking blindspots biggest-wall-time-first.
- [ ] **B4. Run → PASS** (`test_coverage_and_blindspot` still passes).
- [ ] **B5. Commit** `feat(oracle): wire blindspot detector into CSV path; reconstruct true denominator`.

### Deliverable C — enforce STRUCTURED serve provenance (adversarial-review hardened)

The draft's guard only checked non-empty `cmd` + MD5 shape, so `--reproducer-cmd "smoke"` passed. **Harden it to demand structured serve evidence.**

- [ ] **C1. Failing tests** (`kernel_ledger.rs::tests`):
```rust
    #[test]
    fn require_serve_provenance_rejects_microbench_without_prompt_md5() {
        let repro = Reproducer {
            cmd: "kernel_perf_instrument --self-check".to_string(),
            fixture_path: Some("tests/kernel-fixtures/gfx1201/foo.hsaco".to_string()),
            prompt_md5: None,
        };
        let err = repro.require_serve_provenance().unwrap_err();
        assert!(err.contains("prompt_md5"), "got {err}");
    }
    #[test]
    fn require_serve_provenance_rejects_arbitrary_cmd_string() {
        // The exact hole the review flagged: a bare "smoke" cmd + valid md5 must FAIL —
        // the cmd must name the serve driver AND a benchmarks/prompts/ path.
        let repro = Reproducer {
            cmd: "smoke".to_string(),
            fixture_path: None,
            prompt_md5: Some("0123456789abcdef0123456789abcdef".to_string()),
        };
        assert!(repro.require_serve_provenance().is_err(), "arbitrary cmd must not pass provenance");
    }
    #[test]
    fn require_serve_provenance_rejects_non_md5_prompt_hash() {
        let repro = Reproducer {
            cmd: "scripts/oracle-serve-dynamic.sh m benchmarks/prompts/lru_cache_pep8_strict.txt".to_string(),
            fixture_path: None,
            prompt_md5: Some("not-a-real-md5".to_string()),
        };
        assert!(repro.require_serve_provenance().is_err());
    }
    #[test]
    fn require_serve_provenance_accepts_real_serve_row() {
        // Serve driver named + a benchmarks/prompts/ path + 32-hex md5 → OK.
        let repro = Reproducer {
            cmd: "scripts/oracle-serve-dynamic.sh qwen3.5-a3b.mq4r benchmarks/prompts/lru_cache_pep8_strict.txt".to_string(),
            fixture_path: None,
            prompt_md5: Some("0123456789abcdef0123456789abcdef".to_string()),
        };
        assert!(repro.require_serve_provenance().is_ok());
    }
```
- [ ] **C2. Run → FAIL** (`no method require_serve_provenance`).
- [ ] **C3. Implement** (structured, non-forgeable-by-a-bare-string):
```rust
    /// A dynamic bill-of-debt row must PROVE it came from the real serve path.
    /// Proof (spec 4b) = (a) a non-empty cmd that NAMES the serve driver
    /// (`oracle-serve-dynamic.sh`) AND references a committed prompt under
    /// `benchmarks/prompts/`, plus (b) a 32-hex prompt_md5 (a microbench has no
    /// served prompt). A bare/arbitrary cmd is rejected — the flagged hole.
    pub fn require_serve_provenance(&self) -> Result<(), String> {
        if self.cmd.trim().is_empty() {
            return Err("serve provenance: empty Reproducer.cmd".to_string());
        }
        if !self.cmd.contains("oracle-serve-dynamic.sh") {
            return Err(format!(
                "serve provenance: cmd {:?} does not name the serve driver \
                 (scripts/oracle-serve-dynamic.sh) — arbitrary cmd strings are not proof of serving",
                self.cmd
            ));
        }
        if !self.cmd.contains("benchmarks/prompts/") {
            return Err(format!(
                "serve provenance: cmd {:?} references no committed prompt under benchmarks/prompts/",
                self.cmd
            ));
        }
        match self.prompt_md5.as_deref() {
            Some(md5) if is_md5_hex(md5) => Ok(()),
            Some(md5) => Err(format!("serve provenance: prompt_md5 {md5:?} is not a 32-hex MD5")),
            None => Err("serve provenance: prompt_md5 is None — looks like a standalone \
                         microbench CSV, not a real serve-path measurement".to_string()),
        }
    }
```
Add `fn is_md5_hex(s: &str) -> bool { s.len() == 32 && s.bytes().all(|b| b.is_ascii_hexdigit()) }`.
- [ ] **C4. Run → PASS** (existing kernel_ledger tests unchanged).
- [ ] **C5. Commit** `feat(oracle): enforce STRUCTURED serve provenance (serve-driver + benchmarks/prompts + md5)`.

### Deliverable D — wire A/B/C into `run_dynamic` + no-GPU smoke

- [ ] **D1. Add provenance flags** to `Mode::Dynamic` (`reproducer_cmd: Option<String>`, `prompt_md5: Option<String>`; parse `--reproducer-cmd`/`--prompt-md5`).
- [ ] **D2. Enforce at the entry point** — build a `Reproducer`, `require_serve_provenance()`, `exit(3)` on failure with a message pointing at `scripts/oracle-serve-dynamic.sh`.
- [ ] **D3. Emit** `in_model_pct` per kernel, a `"reproducer"` object, and a `"mode":"dynamic_coverage"` summary via `coverage_against_aliases(byte_model.keys(), &rocprof)` + a stderr blindspot ranking.
- [ ] **D4. Build + no-GPU smoke (reject AND accept).** REJECT: no provenance → exit 3, stderr names "microbench". ACCEPT (structured cmd, adversarial-review fix — NOT a bare "smoke"):
```
CSV=$(mktemp /tmp/oracle-smoke.XXXX.csv)
printf 'Name,Calls,TotalDurationNs,AverageNs,Percentage,MinNs,MaxNs,StdDev\n_Z34gemv_hfq4g256_moe_down_k8_indexedPKjS0_Pfi.kd,100,1652900,16529,44.1,1,1,0\n_Z18some_hidden_kernelPKfS0_i.kd,100,2000000,20000,55.9,1,1,0\n' > "$CSV"
cargo run -q -p rdna-compute --example kernel_perf_instrument -- \
  --dynamic --arch gfx1201 --rocprof-csv "$CSV"; echo "exit=$?"   # expect 3
cargo run -q -p rdna-compute --example kernel_perf_instrument -- \
  --dynamic --arch gfx1201 --rocprof-csv "$CSV" \
  --reproducer-cmd "scripts/oracle-serve-dynamic.sh m benchmarks/prompts/lru_cache_pep8_strict.txt" \
  --prompt-md5 0123456789abcdef0123456789abcdef | tee /tmp/oracle-smoke.out; echo "exit=${PIPESTATUS[0]}"  # expect 0
grep -q '"in_model_pct":44.1' /tmp/oracle-smoke.out && grep -q '"mode":"dynamic_coverage"' /tmp/oracle-smoke.out && echo SMOKE_OK
```
- [ ] **D5. Commit** `feat(oracle): route run_dynamic through provenance + emit in-model-% and coverage`.

### Deliverable E — the real serve-path driver `[GPU: hipx]`

- [ ] **E1. Create `scripts/oracle-serve-dynamic.sh`** (as drafted): build the daemon + instrument; md5sum the committed `benchmarks/prompts/<file>`; drive `load → greedy generate → unload` under `scripts/rocprof-wrap.sh` on the pinned card under the per-card GPU lock; feed `<dir>/trace_kernel_stats.csv` to `kernel_perf_instrument --dynamic` with `--reproducer-cmd "scripts/oracle-serve-dynamic.sh <model> benchmarks/prompts/<file> ..."` and `--prompt-md5 <md5>`. `chmod +x`.
- [ ] **E2. Run once on hipx `[GPU: hipx]`** — expect exit 0 with non-zero `in_model_pct` rows, a `"mode":"dynamic_coverage"` line, and a stderr `BLINDSPOT` ranking; re-running WITHOUT `--prompt-md5` (instrument directly on the same CSV) exits 3.
- [ ] **E3. Commit** `feat(oracle): serve-path driver — measure the real daemon under rocprofv3`.

*[verified: corrected per adversarial review — fixed the fabricated `crates/rdna-compute/examples/bench_qwen35_mq4.rs:493` path to the real `crates/hipfire-runtime/examples/bench_qwen35_mq4.rs` caller of `compute_coverage`; noted `AchievedBw` already carries `trust_score` and added `in_model_pct` beside it (not a re-add); replaced the forgeable cmd-shape provenance with a STRUCTURED guard (must name `oracle-serve-dynamic.sh` AND a `benchmarks/prompts/` path AND a 32-hex md5) with an explicit `require_serve_provenance_rejects_arbitrary_cmd_string` test and a non-`"smoke"` accept smoke; aligned `compute_achieved_bw` calls to 0a3's 4-arg `dpm_state` form and built B's `is_covered` on 0a3's `covered_by_alias` boundary rule.]*

---

## Task 0c — Per-arch PMC census + promotion gate

Builds §4c per-arch PMC counter census (which *derived* roofline counters — `FetchSize`/`WriteSize`/`VALUBusy`/`MemUnitBusy`/`OccupancyPercent` — revive under `profile_standard` vs. only the raw accumulators `SQ_BUSY_CYCLES`/`Wavefronts`, **per arch**) plus the §5.5 per-cell/per-arch promotion gate. All logic is NO-GPU-testable (the GPU only produces the input CSV). **Standalone module** (`depends_on: []`) — the ONLY cross-file touch is one append-only `pub mod` line in `lib.rs`.

**Scope boundary (adversarial-review fix):** this task does **NOT** rewire `Roofline::analyze` or `kernel_perf_instrument` to consume the census. The census + gate are a self-contained decision surface; **dynamic-domain cells stay WITHHELD until 0a and 0b land**, which flows in at *data-population* time through `CellEvidence.unmitigated_high_risk` (an input) — not a build dependency here.

### Files
- **Create:** `crates/rdna-compute/src/pmc_census.rs` (census + gate + tests)
- **Create:** `crates/rdna-compute/examples/pmc_census.rs` (aggregate captured CSVs → committed JSON)
- **Create (Cycle 7, on hipx):** `tests/pmc-census/{gfx1010,gfx1030,gfx1100,gfx1151,gfx1201}.json`
- **Modify:** `crates/rdna-compute/src/lib.rs` (`pub mod pmc_census;`)

### Interfaces
**Consumes:** the `parse_rocprof_stats_csv_text` right-anchoring trick; the `chip_profile.rs` data-not-tags JSON pattern; `roofline::BoundClass`; `profile_rocprof::TRUST_ANALYTIC_GFX12_PMC_ZERO`.
**Produces:** `classify_counter`/`CounterKind`; `REQUIRED_DERIVED_COUNTERS`/`RAW_ACCUMULATOR_COUNTERS`/`DEFAULT_MIN_NONZERO_FRACTION`/`REQUIRED_PERF_LEVEL`/`MIN_INDEPENDENT_SOURCES`; `census_from_counter_csv_text`; `ArchPmcCensus` (+ `verdict`/`is_valid_capture`/`derived_revived`/`accumulators_alive`/`only_accumulators_revived`/`to_json`/`from_json`/`load`); `CounterVerdict`; `PromotionState`/`WithheldReason`/`CellEvidence`/`evaluate_cell`; `pmc_source_count`; `dynamic_bound_class_evidence`.

Run: `cargo test -p rdna-compute --lib pmc_census` (NO GPU). Format via `scripts/fmt-changed.sh`.

---

- [ ] **Cycle 1 — counter classification.** Create `pmc_census.rs` with the module doc + `#[cfg(test)] mod tests` (`classify_counter_maps_derived_accumulator_unknown`, `required_derived_set_is_the_five_roofline_counters`), register `pub mod pmc_census;` in `lib.rs`. Run → FAIL (missing symbols). Implement `REQUIRED_DERIVED_COUNTERS`/`RAW_ACCUMULATOR_COUNTERS`/`CounterKind`/`classify_counter` (case-insensitive). Run → PASS. Commit `feat(oracle): add PMC counter classification (task 0c, spec §4c)`.

- [ ] **Cycle 2 — census from CSV.** Add the two synthetic CSVs + `census_parses_counts_classifies_and_derives_revival`, `census_rejects_header_not_ending_in_counter_name_value`, `census_right_anchors_past_commas_in_kernel_name`. Run → FAIL. Implement `DEFAULT_MIN_NONZERO_FRACTION=0.5`, `REQUIRED_PERF_LEVEL="profile_standard"`, `CounterVerdict`, `counter_verdict`, `ArchPmcCensus { arch, perf_level, min_nonzero_fraction, counters }` + `verdict`, and `census_from_counter_csv_text` (right-anchored `Counter_Name,Counter_Value`, BTreeMap aggregation). Run → PASS. Commit `feat(oracle): parse rocprofv3 --pmc CSV into per-arch counter census (task 0c)`.

- [ ] **Cycle 3 — revival + profile_standard guard.** Add `only_accumulators_arm`, `all_revived_arm`, `missing_required_derived_counter_blocks_revival`, `invalid_perf_level_forces_derived_withheld_even_if_nonzero`. Run → FAIL. Implement `is_valid_capture`, `derived_revived` (false if invalid capture / missing / dead), `accumulators_alive`, `only_accumulators_revived`. Run → PASS. Commit `feat(oracle): per-arch derived-vs-accumulator revival with profile_standard guard (task 0c §5.2)`.

- [ ] **Cycle 4 — data-not-tags serde.** Add `census_json_is_raw_counts_only_and_round_trips` (asserts NO `revived`/`kind`/`nonzero_fraction` on disk), `from_json_fails_loud_on_missing_field`. Run → FAIL. Implement `to_json` (raw counts + capture context only), `from_json` (fail-loud, re-derives verdicts), `load`. Run → PASS. Commit `feat(oracle): data-not-tags serde for the PMC census (task 0c §4d)`.

- [ ] **Cycle 5 — §5.5 promotion gate.** Add `gate_promotes_only_when_all_three_criteria_hold`, `gate_withholds_on_insufficient_sources`, `gate_withholds_on_unverified_behavior`, `gate_withholds_on_unmitigated_high_risk`, `gate_records_every_failing_reason`. Run → FAIL. Implement `MIN_INDEPENDENT_SOURCES=2`, `WithheldReason`, `PromotionState` (+ `is_promoted`/`gate_value`), `CellEvidence`, `evaluate_cell`. Run → PASS. Commit `feat(oracle): per-cell/per-arch promotion gate (PROMOTED|WITHHELD{reason}) (task 0c §5.5)`.

- [ ] **Cycle 6 — withheld-never-faked + no-arch-transfer.** Add `withheld_never_fakes_a_value`, `no_arch_invariance_transfer_gfx1100_promotion_does_not_promote_gfx1201`. Run → FAIL. Implement `pmc_source_count` (1 iff `derived_revived`, else 0) + `dynamic_bound_class_evidence` (analytic byte-model NOT counted as independent). Run → PASS + `cargo build -p rdna-compute --all-targets`. Commit `feat(oracle): census→gate composition enforces no-arch-transfer + withheld-never-faked (task 0c §5.5)`.

- [ ] **Cycle 7 — [GPU: hipx] real census rows + aggregation binary + loader test.** Create `crates/rdna-compute/examples/pmc_census.rs` (NO-GPU aggregation: multi-`--csv` fold under one header → `--out tests/pmc-census/<arch>.json`). Add the loader test `committed_censuses_are_valid_captures_with_working_controls` (asserts `is_valid_capture` + `accumulators_alive` per row; does NOT assert the measured `derived_revived`). Run → FAIL (`tests/pmc-census` absent).

  **Capture on hipx (adversarial-review fix — concretized, no `<placeholder>` tokens).** Worked example for gfx1201 (hiptrx, R9700, HIP ordinal 1, DRM `card1`); repeat with each arch's own pinned ordinal/card from `--resolve-device` (0e) + `rocminfo`. Drive a committed prompt (`benchmarks/prompts/lru_cache_pep8_strict.txt`) via a daemon JSON-lines input file, NOT a bare `prompt.jsonl`:
```sh
source scripts/gpu-lock.sh
export HIPFIRE_GPU_LOCKFILE=/tmp/hipfire-gpu-card1.lock
gpu_acquire "kernel-oracle-0c:gfx1201"
export HIP_VISIBLE_DEVICES=1
echo profile_standard | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
amd-smi metric --gpu 1 --perf-level        # confirm AMDSMI_DEV_PERF_LEVEL_STABLE_STD
PROMPT=benchmarks/prompts/lru_cache_pep8_strict.txt
PJSON=$(python3 -c "import sys,json;print(json.dumps(open(sys.argv[1]).read()))" "$PROMPT")
mkdir -p /tmp/pmc-gfx1201
cat > /tmp/pmc-gfx1201/daemon_in.jsonl <<JL
{"type":"load","model":"$HOME/.hipfire/models/qwen3.5-a3b.mq4r","params":{"max_seq":4096}}
{"type":"generate","id":"c1","prompt":${PJSON},"temperature":0.0,"max_tokens":64}
{"type":"unload"}
JL
# <=2 counters / <=2 HW blocks per pass (anti-hang). DERIVED metrics + RAW controls.
rocprofv3 --pmc VALUBusy SQ_BUSY_CYCLES -S -f csv -d /tmp/pmc-gfx1201/p1 -o c -- \
  ./target/release/examples/daemon < /tmp/pmc-gfx1201/daemon_in.jsonl
rocprofv3 --pmc MemUnitBusy OccupancyPercent -S -f csv -d /tmp/pmc-gfx1201/p2 -o c -- \
  ./target/release/examples/daemon < /tmp/pmc-gfx1201/daemon_in.jsonl
rocprofv3 --pmc FetchSize WriteSize Wavefronts -S -f csv -d /tmp/pmc-gfx1201/p3 -o c -- \
  ./target/release/examples/daemon < /tmp/pmc-gfx1201/daemon_in.jsonl
mkdir -p tests/pmc-census
cargo run -p rdna-compute --example pmc_census -- --arch gfx1201 --perf-level profile_standard \
  --csv /tmp/pmc-gfx1201/p1/c_counter_collection.csv \
  --csv /tmp/pmc-gfx1201/p2/c_counter_collection.csv \
  --csv /tmp/pmc-gfx1201/p3/c_counter_collection.csv \
  --out tests/pmc-census/gfx1201.json
echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
gpu_release
```
  NOTE (record as a finding, do NOT fake): `VALUBusy` is absent from the gfx1100 metrics DB — on gfx1100 that pass yields no `VALUBusy` rows, so `derived_revived()` is legitimately `false` (a missing required derived counter blocks revival, Cycle 3). The loader test still passes (it asserts only capture-validity + working controls). Run → PASS (`cargo test -p rdna-compute --lib pmc_census` + `cargo build --release --workspace --all-targets --locked`). Commit `feat(oracle): per-arch profile_standard PMC census rows + aggregation binary + loader guard (task 0c §4c/§4e)`.

*[verified: corrected per adversarial review — concretized every Cycle-7 placeholder (`<arch>`, `<ordinal-for-arch>`, `card<N>`, `/tmp/pmc-<arch>/`, `prompt.jsonl`) into a runnable gfx1201/ordinal-1/card1 worked example driving a committed `benchmarks/prompts/` prompt via a real daemon JSON-lines input file; added an explicit scope boundary stating the census does NOT gate `Roofline::analyze` and that dynamic-domain cells stay WITHHELD until 0a/0b land (fed via `CellEvidence.unmitigated_high_risk`, an input not a build dep) — the review's "claims it gates dynamic bound-class" overreach.]*

---

## Task 0d — The bill-of-debt ledger

Implements §4d. A NEW no-GPU module `crates/rdna-compute/src/bill_of_debt.rs` mirroring the `kernel_ledger.rs` AtlasRow-JSONL pattern. One row per **(arch × fitting-model × kernel × domain)** in `Measured` / `Withheld` / `Structural`. **Debt magnitude and any `bound_class` verdict are DERIVED at query, never persisted.** All tests are pure lib unit tests over synthetic rows — no GPU. `depends_on: []` (soft data-dependency on 0a/0b/0c at *population* time only).

### Files
- **Create:** `crates/rdna-compute/src/bill_of_debt.rs`
- **Modify:** `crates/rdna-compute/src/lib.rs` (`pub mod bill_of_debt;`)

### Interfaces
**Consumes** (from `crates/hipfire-atlas/src/schema.rs`): `AtlasRow::new`, `set_metric_f64`, `set_extra`, `metric_f64`, `append_to_jsonl`, `load_rows`, `truncate_jsonl`, and `AtlasRow.model_size`/`workload_kind`/`extra`.
**Produces:** `DebtKey`; `DebtRow{Measured/Withheld/Structural}` + `key`/`debt_magnitude_ms`/`to_atlas_row`/`from_atlas_row`; `BillOfDebt` + `load`/`emit`/`ranked_by_lever`/`withheld_targets`/`per_arch_total_debt`/`cross_arch_unevenness`/`no_arch_clobber_delta`; `ArchDebtDelta`; `ClobberReport`.

---

- [ ] **Cycle 1 — `DebtRow` + derived `debt_magnitude_ms`.** Create `bill_of_debt.rs` with `DebtKey`, `DebtRow`, and a **TDD red-state scaffold** `debt_magnitude_ms` returning `None // [TDD red-state scaffold — replaced in step (3) of THIS cycle; not a shipped placeholder]`. Register `pub mod bill_of_debt;`. Add `debt_magnitude_derived_per_variant` (Measured: `in_model_walltime_ms * pct_off_roofline / 100`; Structural: `fallback_penalty_ms`; Withheld: `None`). Run → FAIL (`left: None, right: Some(100.0)`). Replace the scaffold body with the real match. Run → PASS. Commit `feat(oracle): bill-of-debt DebtRow + query-derived debt magnitude (Phase 0/0d)`.

- [ ] **Cycle 2 — `BillOfDebt` + `ranked_by_lever` + `withheld_targets`.** Add `ranked_by_lever_uses_real_time_not_efficiency` (90%-off/1ms < 20%-off/500ms; withheld excluded). Run → FAIL (`BillOfDebt` missing). Implement `key()`, `struct BillOfDebt`, `ranked_by_lever` (sort by `debt_magnitude_ms`, exclude None), `withheld_targets`. Run → PASS. Commit `feat(oracle): BillOfDebt + lever ranking by recoverable real time (0d)`.

- [ ] **Cycle 3 — `per_arch_total_debt` + `cross_arch_unevenness`.** Add `per_arch_totals_and_unevenness`, `unevenness_zero_when_even`. Run → FAIL. Add `use std::collections::BTreeMap;` + `const DEBT_EPS_MS: f64 = 1e-9;`; implement `per_arch_total_debt` (withheld contributes nothing) + `cross_arch_unevenness` (`(max-min)/mean`, 0 for <2 arches or zero mean). Run → PASS. Commit `feat(oracle): per-arch total debt + cross-arch unevenness score (0d)`.

- [ ] **Cycle 4 — `no_arch_clobber_delta` (spec §4d: any per-arch debt growth is a clobber).** Adversarial-review fix: **remove the ±3% noise band** — spec §4d requires the candidate per-arch debt delta to be `<= 0` for every arch. Any positive per-arch debt delta beyond epsilon is a clobber. Tests:
```rust
    #[test]
    fn no_arch_clobber_delta_flags_worsened_arch() {
        let baseline = BillOfDebt { rows: vec![
            measured("gfx1010", "k1", "d1", 100.0, 50.0, 100.0),
            measured("gfx1100", "k2", "d2", 100.0, 50.0, 100.0) ] };
        let candidate = BillOfDebt { rows: vec![
            measured("gfx1010", "k1", "d1", 100.0, 50.0, 150.0),  // +50 -> clobber
            measured("gfx1100", "k2", "d2", 100.0, 50.0, 80.0) ] }; // -20 -> improvement
        let report = BillOfDebt::no_arch_clobber_delta(&baseline, &candidate);
        assert!(report.any_arch_worsened);
        let g1010 = report.per_arch.iter().find(|d| d.arch == "gfx1010").unwrap();
        assert_eq!(g1010.delta_ms, 50.0); assert!(g1010.worsened);
        let g1100 = report.per_arch.iter().find(|d| d.arch == "gfx1100").unwrap();
        assert_eq!(g1100.delta_ms, -20.0); assert!(!g1100.worsened);
    }
    #[test]
    fn no_arch_clobber_delta_clean_when_all_improve() {
        let baseline = BillOfDebt { rows: vec![
            measured("gfx1010", "k1", "d1", 100.0, 50.0, 100.0),
            measured("gfx1100", "k2", "d2", 100.0, 50.0, 100.0) ] };
        let candidate = BillOfDebt { rows: vec![
            measured("gfx1010", "k1", "d1", 100.0, 50.0, 90.0),
            measured("gfx1100", "k2", "d2", 100.0, 50.0, 95.0) ] };
        assert!(!BillOfDebt::no_arch_clobber_delta(&baseline, &candidate).any_arch_worsened);
    }
    #[test]
    fn no_arch_clobber_delta_any_positive_growth_is_a_clobber() {
        // Spec §4d: candidate per-arch debt must be <= baseline for EVERY arch.
        // Even a small +2% growth is a clobber — there is NO tolerated noise band.
        let baseline = BillOfDebt { rows: vec![measured("gfx1100", "k", "d", 100.0, 50.0, 100.0)] };
        let candidate = BillOfDebt { rows: vec![measured("gfx1100", "k", "d", 100.0, 50.0, 102.0)] };
        let report = BillOfDebt::no_arch_clobber_delta(&baseline, &candidate);
        assert!(report.any_arch_worsened, "any per-arch debt growth violates §4d, no band");
    }
```
  Run → FAIL. Implement (no band const):
```rust
    /// No-arch-clobber delta (spec §4d): per-arch debt delta (candidate -
    /// baseline). §4d requires candidate debt <= baseline for EVERY arch, so an
    /// arch is *worsened* whenever its debt rises by more than DEBT_EPS_MS —
    /// there is NO tolerated percentage band. The CI invariant Phase-2 enforces
    /// is `any_arch_worsened == false`.
    pub fn no_arch_clobber_delta(baseline: &BillOfDebt, candidate: &BillOfDebt) -> ClobberReport {
        let base_totals = baseline.per_arch_total_debt();
        let cand_totals = candidate.per_arch_total_debt();
        let mut arches: Vec<String> = base_totals.keys().chain(cand_totals.keys()).cloned().collect();
        arches.sort(); arches.dedup();
        let mut per_arch = Vec::new();
        let mut any_arch_worsened = false;
        for arch in arches {
            let b = base_totals.get(&arch).copied().unwrap_or(0.0);
            let c = cand_totals.get(&arch).copied().unwrap_or(0.0);
            let delta_ms = c - b;
            let worsened = delta_ms > DEBT_EPS_MS; // any real growth is a clobber (§4d)
            if worsened { any_arch_worsened = true; }
            per_arch.push(ArchDebtDelta { arch, baseline_debt_ms: b, candidate_debt_ms: c, delta_ms, worsened });
        }
        ClobberReport { per_arch, any_arch_worsened }
    }
```
  Add `struct ArchDebtDelta` + `struct ClobberReport`. Run → PASS. Commit `feat(oracle): no-arch-clobber per-arch debt delta — any growth is a clobber (0d §4d)`.

- [ ] **Cycle 5 — JSONL emit/load round-trip (raw fields only).** Add `bill_round_trips_through_jsonl_raw_fields_only` (asserts NO `recoverable`/`debt_magnitude`/`debt_ms`/`bound_class`/`unevenness` on disk; every row round-trips value-for-value). Run → FAIL. Extend the `use` block (`hipfire_atlas::schema::AtlasRow`, `serde_json::Value`, `std::path::Path`); implement `to_atlas_row`/`from_atlas_row` (`phase="bill_of_debt"`, `workload_kind=domain`, `model_size=model`, `arch`/`kernel`/`debt_kind`/`withheld_reason` via `set_extra`, numbers via `set_metric_f64`) + `extra_str`/`metric` helpers + `BillOfDebt::load`/`emit`. Run → PASS. Commit `feat(oracle): bill-of-debt JSONL emit/load round-trip, raw fields only (0d)`.

*[verified: corrected per adversarial review — removed the ±3% `DEBT_CLOBBER_BAND_PCT` noise band that contradicted spec §4d (candidate per-arch debt must be <= baseline for EVERY arch), replacing `no_arch_clobber_delta_within_band_is_not_a_clobber` with `no_arch_clobber_delta_any_positive_growth_is_a_clobber` so any real per-arch debt growth is flagged; annotated the Cycle-1 `debt_magnitude_ms` stub as an explicit TDD red-state scaffold replaced within the same cycle's step (3), not a shipped placeholder; documented the #490/#491/in-model/census root causes as owned by the separate 0a1/0a2/0a3/0b/0c tasks that this plan lands upstream, with 0d consuming their surfaced raw fields at population time.]*

**Anti-clobber:** `bill_of_debt.rs` is unique to 0d; the sole cross-task surface is the one-line `pub mod bill_of_debt;` append to `lib.rs` (union with 0c's `pub mod pmc_census;`). No GPU/ROCm at any step.

---

## Task 0e — hipx all-5-arch execution harness

Drive the Phase-0 bill-of-debt matrix over gfx1010/1030/1100/1151/1201 on the one hipx runner (§4e): each arch pinned via `HIP_VISIBLE_DEVICES` + a **per-card** `HIPFIRE_GPU_LOCKFILE=/tmp/hipfire-gpu-cardN.lock`, **serial under the locks**, **two DPM passes per cell**, registry VRAM enumeration (skip OOM → WITHHELD), and **arch→device resolved by the runtime `arch:` line, never a fixed index**. **Depends on 0b, 0c, 0d.** No-GPU TDD (Cycles A–D, F) via a stub probe + fixture registry; the live sweep is one `[GPU: hipx]` step.

### Files
- **Create:** `scripts/oracle-bill-of-debt-sweep.sh`, `scripts/tests/test-oracle-bill-of-debt-sweep.sh`, `tests/fixtures/oracle-registry-mini.json`
- **Modify:** `scripts/no-gpu-ci.sh`

### Interfaces
**Consumes:** `scripts/gpu-lock.sh` (`gpu_acquire`/`gpu_release`; `LOCKFILE="${HIPFIRE_GPU_LOCKFILE:-/tmp/hipfire-gpu.lock}"`, reassign per-card, never `rm`); `scripts/chip-profile-sweep.sh` patterns (`run_pass`/`extract`/`try_rocm_smi_set`); `pointer_chase_latency.rs:120` `println!("arch: {}", gpu.arch)` as the injectable arch probe; `kernel_perf_instrument --dynamic --rocprof-csv <path> --arch <arch>` (early-returns before `KernelLedger::load`, prints per-kernel JSONL to stdout, never mutates the committed ledger; `ChipProfile::load_committed` `exit(1)` on un-profiled arch → WITHHELD cell); `scripts/rocprof-wrap.sh <dir> -- <cmd>`; `cli/registry.json` (`min_vram_gb <= budget`).
**Produces:** the sweep CLI (`--self-check`/`--dry-run`, `--print-fitting <arch>`, `--resolve-device <arch>`, `--print-live-plan`, `--arches`/`--out`/`--registry`/`--max-tokens`/`--prompt-file`/`--no-force-dpm`); plan JSONL (raw fields only); live measured rows.

**Corpus vs measurement-log distinction (adversarial-review fix):** the **plan JSONL** rows the harness authors itself carry raw fields only (`bound_class` never stored). The **live measured JSONL** is the verbatim stdout of `kernel_perf_instrument --dynamic`, which legitimately includes query-derived fields (`bound_class`, `bw`, `latency`, `verdict`, `has_fixture`, …) — these are *derived at emit time*, not persisted-as-corpus tags, and are consistent with data-not-tags because nothing here treats them as a committed corpus. The committed bill-of-debt corpus (0d) is built downstream from these logs, storing raw fields only.

---

- [ ] **Cycle A — skeleton.** Create `scripts/tests/test-oracle-bill-of-debt-sweep.sh` (as drafted: stub `ORACLE_ARCH_PROBE_BIN` echoing `arch: gfxNNNN` per `HIP_VISIBLE_DEVICES`; A1 self-check writes plan + prints `self-check OK`; A2 unknown flag → exit 2) + `tests/fixtures/oracle-registry-mini.json` (`tiny:2g`/`small:8g`/`big:24g`/`huge:64g`). Run → FAIL (script missing). Create `scripts/oracle-bill-of-debt-sweep.sh` with modes/arg-parse/`arch_vram_gb`/`run_self_check`, and **TDD red-state scaffolds** `plan_matrix() { :; }  # [TDD scaffold — filled in Cycle D]` and `run_live() { log "live sweep not yet implemented"; return 0; }  # [TDD scaffold — filled in Cycle E]`. Run → PASS (A1/A2). Commit `feat(oracle): task 0e skeleton — sweep harness self-check + arg parse`.

- [ ] **Cycle B — registry VRAM enumeration.** Add B1/B2 (`--print-fitting gfx1010` keeps ≤8GB; `ORACLE_VRAM_GB_GFX1010=24` override widens). Run → FAIL (unknown flag). Add `--print-fitting` mode + `models_fitting` (python3 JSON filter) + `print_fitting` + dispatch. Run → PASS. Commit `feat(oracle): task 0e registry-driven per-arch VRAM enumeration`.

- [ ] **Cycle C — arch→device by runtime `arch:` line.** Add C1/C2 (`--resolve-device gfx1201` → `1`; absent arch → empty). Run → FAIL. Add `--resolve-device` mode + `hip_device_count` (`ORACLE_HIP_DEVICE_COUNT` override / `rocminfo` gfx count) + `resolve_device_for_arch` (probe each `HIP_VISIBLE_DEVICES=$i`, match `^arch:`). Run → PASS. Commit `feat(oracle): task 0e arch->device resolution by runtime gfx line`.

- [ ] **Cycle D — full dry-run matrix.** Add D1 (every plan line valid JSON; 6 PLANNED + 1 WITHHELD for `gfx1100 gfx1201 gfx1010`; gfx1201 rows carry `device:1` + `passes:["walltime_toks@high","pmc_ratios@profile_standard"]`; gfx1010 WITHHELD/null). Run → FAIL (`plan_matrix` stub empty). Add `emit_planned_row`/`emit_withheld_arch`/`emit_withheld_cell` (python3 JSONL, raw fields only) and fill `plan_matrix` (resolve device → WITHHELD if absent else one PLANNED row per fitting model). Run → PASS. Commit `feat(oracle): task 0e full dry-run matrix (PLANNED + WITHHELD rows)`.

- [ ] **Cycle E — live command construction (no-GPU dry-run) + the live sweep `[GPU: hipx]`.** Adversarial-review fix: replace the tautological E1 with a genuine **`--print-live-plan`** dry-run mode that emits the exact command strings the live sweep WOULD run (without touching a GPU), so E1 is a real red→green:
  - **E1 tests** (append to the bash test):
```bash
# E1a: --print-live-plan emits the real command shapes without touching a GPU.
echo "[E1a] --print-live-plan shows rocprof-wrap + rocprofv3 --pmc + per-card lock + profile_standard"
plan="$(ORACLE_SERVE_DRIVER=/bin/true ORACLE_INSTRUMENT_BIN=/bin/true \
        bash "$SWEEP" --print-live-plan --arches "gfx1201" --registry "$FIX" 2>/dev/null)"
echo "$plan" | grep -q 'rocprof-wrap.sh'                 && \
echo "$plan" | grep -q 'rocprofv3 --pmc'                 && \
echo "$plan" | grep -q '/tmp/hipfire-gpu-card1.lock'     && \
echo "$plan" | grep -q 'profile_standard'                && \
echo "$plan" | grep -q 'WITHHELD_ON_INSTRUMENT_FAILURE'  \
  && ok "live plan names rocprof-wrap, rocprofv3 --pmc, per-card lock, profile_standard, WITHHELD-on-fail" \
  || bad "live plan missing a required command element: [$plan]"

# E1b: self-check must NEVER invoke rocm-smi/rocprofv3 (CI GPU-isolation).
echo "[E1b] self-check never touches GPU tooling even when 'present'"
FAKEBIN="$WORK/fakebin"; mkdir -p "$FAKEBIN"
for t in rocm-smi rocprofv3; do
    printf '#!/usr/bin/env bash\necho "CALLED_%s" >&2\nexit 99\n' "$t" > "$FAKEBIN/$t"; chmod +x "$FAKEBIN/$t"
done
PATH="$FAKEBIN:$PATH" bash "$SWEEP" --self-check --out "$WORK/e1" --registry "$FIX" --arches "gfx1201" > "$WORK/e1.log" 2>&1
rc=$?
{ [ "$rc" -eq 0 ] && ! grep -q "CALLED_" "$WORK/e1.log"; } \
  && ok "self-check exited 0 and never invoked rocm-smi/rocprofv3" \
  || bad "self-check touched GPU tooling or failed (rc=$rc)"
```
  - Run → FAIL (`--print-live-plan` unknown flag; the drafted E1a strings don't exist).
  - **Implement.** Add a `--print-live-plan` mode that, for each resolved-present arch × fitting model, prints (without executing) the exact per-cell command lines it would run: the `HIP_VISIBLE_DEVICES=<dev> scripts/rocprof-wrap.sh <dir> -- <serve driver> ...` high pass, the `HIP_VISIBLE_DEVICES=<dev> rocprofv3 --pmc <census counter set> ... profile_standard` PMC pass, the per-card `HIPFIRE_GPU_LOCKFILE=/tmp/hipfire-gpu-card<dev>.lock`, and a `WITHHELD_ON_INSTRUMENT_FAILURE` sentinel documenting the OOM/serve/instrument-failure → WITHHELD path. **PMC counter set (adversarial-review fix):** default `ORACLE_PMC_SET` to the 0c census set (`FetchSize WriteSize VALUBusy MemUnitBusy OccupancyPercent` DERIVED + `SQ_BUSY_CYCLES Wavefronts` RAW controls), captured in ≤2-counter multi-pass, NOT the hardcoded raw-only `SQ_WAVES SQ_BUSY_CYCLES`; the harness delegates census aggregation to 0c's `pmc_census` example. Then implement `sweep_cell_live` + `run_live` (per-card lock reassignment, high pass via `rocprof-wrap` + serve driver, `profile_standard` PMC pass, emit measured rows via `kernel_perf_instrument --dynamic` **passing `--dpm-state high`** to 0a3's threaded param, WITHHELD cell on any failure, restore prior DPM).
  - Run → PASS (E1a/E1b, no GPU).
  - **[GPU: hipx] live smoke:** `source scripts/gpu-lock.sh && bash scripts/oracle-bill-of-debt-sweep.sh --arches gfx1201 --max-tokens 32 --out /home/kaden/oracle-smoke` → assert `test -s /home/kaden/oracle-smoke/bill-of-debt.gfx1201.jsonl` with ≥1 `"mode":"dynamic"` line; `/tmp/hipfire-gpu-card<N>.lock` created + released; prior DPM restored; an absent arch yields a WITHHELD plan row without aborting.
  - Commit `feat(oracle): task 0e live two-DPM-pass sweep under per-card locks + --print-live-plan dry-run`.

- [ ] **Cycle F — wire the no-GPU test into CI.** `grep -q oracle-bill-of-debt scripts/no-gpu-ci.sh` → 1 (absent). Add after `python3 scripts/check-env-docs.py`:
```bash
echo "== Bash regression tests (no GPU) =="
bash scripts/tests/test-oracle-bill-of-debt-sweep.sh
```
  Run → PASS (`grep` → 0; `bash scripts/no-gpu-ci.sh` runs the oracle test green). Commit `ci: run the oracle bill-of-debt sweep no-GPU test in no-gpu-ci`.

*[verified: corrected per adversarial review — replaced the tautological E1 with a `--print-live-plan` no-GPU command-construction mode whose test asserts `rocprof-wrap`, `rocprofv3 --pmc`, the per-card `/tmp/hipfire-gpu-card1.lock`, `profile_standard`, and the WITHHELD-on-instrument-failure sentinel (a genuine red→green for the live behavior without a GPU); labelled the `plan_matrix`/`run_live` stubs as TDD scaffolds filled in Cycles D/E, not placeholders; clarified that plan JSONL is raw-only while the `--dynamic` measured log's query-derived `bound_class`/`bw`/`latency` are emit-time-derived (not a stored corpus), reconciling with the drafted "bound_class never stored" claim; defaulted `ORACLE_PMC_SET` to 0c's DERIVED+RAW census counter set instead of the raw-only `SQ_WAVES SQ_BUSY_CYCLES`; threaded `--dpm-state high` into the `--dynamic` emit per 0a3's real path.]*

**Non-clobber:** all three primary files are new; `scripts/no-gpu-ci.sh` is an append-only edit; the live emit writes only to a fresh `--out` dir + captures `--dynamic` stdout, never the committed `tests/kernel-ledger/*.jsonl` seeds.

---

## Execute as an ultracode workflow (swarm + ultracodex adversarial verify)

Run this plan as a custom **Workflow** (Pattern A: Claude authors, Codex adversarially verifies) — the orchestration stays in Claude's Workflow JS; each task's verify node shells to `codex exec` (a genuinely different model family) to catch correlated Claude failure modes, then a build/test gate, then a serial commit.

**Per-task node shape (repeated for each task):**
1. **author node (Claude agent).** For tasks that share `chip_profile.rs` / `roofline.rs` / `profile_rocprof.rs` / `kernel_ledger.rs` / `kernel_perf_instrument.rs` (i.e. **0a1, 0a2, 0a3, 0b**), run the author agent in **`isolation:'worktree'` and have it RETURN the diff** (do not let it commit to the shared tree) — the anti-clobber discipline applied to our own swarm. Locus-disjoint authors (**0c, 0d, 0e**) may author in place.
2. **codexNode adversarial-verify (`codex exec`).** Feed Codex the task's diff + the task section + the Global Constraints; ask it to re-run the same refutation lens (grounding_ok, addresses_root_cause, test_meaningful, placeholders_found) and return a structured verdict. On `refuted=true`/placeholders, loop back to the author node with the corrections before gating.
3. **gate node (Bash).**
   ```
   cargo build --workspace --all-targets   # CI-required build (no-GPU via HIP dlopen)
   cargo test -p rdna-compute --lib --features deltanet   # validate rdna-compute the SAFE way
   ```
   (Never `cargo test -p rdna-compute --all-targets` alone — `rope_compact_offset_check` needs `cfg=deltanet`, per Global Constraints.) For bash-only tasks add `bash scripts/tests/test-oracle-bill-of-debt-sweep.sh`.
4. **serial commit node.** Apply the isolated worktree's returned diff onto the integration tree in dependency order and commit (Co-Authored-By trailer). Diffs from concurrently-authored file-sharing tasks are committed one-at-a-time here; if a later diff no longer applies cleanly, bounce it back to its author node to rebase.

**Fanout groups (explicit):**
- **Group 1 — serialized/isolated pipeline (shared roofline/chip_profile/profile_rocprof loci):** `pipeline(0a1 → 0a2 → 0a3 → 0b)`. Each author-in-worktree → codex-verify → gate → serial-commit before the next starts. 0a3's PREFLIGHT hard-refuses if 0a1/0a2 aren't yet committed; 0b composes on 0a3's struct-literal edits.
- **Group 2 — free fanout (locus-disjoint), concurrent with Group 1:** `parallel(0c, 0d, 0e-noGPU-cycles)`. Each is its own author → codex-verify → gate → commit lane. The ONLY serialization point is the `lib.rs` `pub mod` append (0c + 0d) and the `no-gpu-ci.sh` append (0e) — a single terminal "module-registration + CI-wire" commit node unions those append-only lines after both land.
- **Group 3 — GPU tail (after Group 1 tail + 0c + 0d commit):** the `[GPU: hipx]` steps — 0a1 Sub-task E re-measure, 0a3 D4b sysfs check, 0b Deliverable E serve driver, 0c Cycle 7 capture, 0e live sweep. Run these with **per-card GPU-lock parallelism on hipx**: reassign `HIPFIRE_GPU_LOCKFILE=/tmp/hipfire-gpu-card<N>.lock` per card so different arches' captures proceed on different physical cards concurrently while each card stays singly-held (flock auto-releases on death). 0e's live sweep is itself serial-under-per-card-locks internally.

**Verification order inside the workflow:** codex-verify BEFORE the build/test gate (cheap adversarial read catches root-cause misses that a green build would mask), then the gate confirms the diff actually compiles + passes lib tests, then commit. A task is not "done" until its codex node returns non-refuted AND its gate is green AND its diff is committed.

---

## Self-review

**1. Spec coverage — every Phase-0 sub-goal 0a–0e has a task.**
- **0a (accuracy hardening):** 0a1 (#490 cache-deflated `mem_latency_ns` — buffer sizing + residency guard + WITHHELD rows), 0a2 (#491 `simds_per_cu` in `latency_score` + `verify_live` + zero-rejection), 0a3 (MED round-up A–H: bound_class-in-diff, MeasuredBw trust, cache working-set, all-sysfs-ordinal-resolution, DPM refuse/thread, summed-VMEM-width latency, word-boundary coverage, VGPR note cross-check wired into the profiling path) — with a PREFLIGHT gating 0a3 on 0a1/0a2. ✅
- **0b (in-model loop):** `in_model_pct`, `coverage_against_aliases`/true denominator, structured `require_serve_provenance`, `run_dynamic` wiring, `oracle-serve-dynamic.sh`. ✅
- **0c (PMC census + promotion gate):** `pmc_census.rs` (classify/census/revival/serde/gate/composition) + aggregation binary + committed per-arch rows. ✅
- **0d (bill of debt):** `bill_of_debt.rs` (DebtRow/derived magnitude/ranking/per-arch totals/unevenness/no-arch-clobber/JSONL). ✅
- **0e (hipx harness):** `oracle-bill-of-debt-sweep.sh` (self-check/print-fitting/resolve-device/print-live-plan/live) + no-GPU test + fixture registry + CI wiring. ✅

**2. Placeholder scan.** The four Codex-flagged placeholder classes are resolved: (a) 0c's Cycle-7 `<arch>`/`<ordinal>`/`card<N>`/`/tmp/pmc-<arch>`/`prompt.jsonl` → concretized to a runnable gfx1201/ordinal-1/card1 worked example over a committed prompt; (b) 0d's `debt_magnitude_ms` `None // STUB` and 0e's `plan_matrix`/`run_live` `:` stubs → explicitly annotated as TDD red-state scaffolds replaced within their own cycle's implementation step (a legitimate red→green pattern, not shipped placeholders); (c) 0e's tautological E1 → a real `--print-live-plan` command-construction test asserting `rocprof-wrap`/`rocprofv3 --pmc`/per-card-lock/`profile_standard`/WITHHELD-on-fail; (d) 0a3's assumed-but-unchecked 0a1/0a2 dependency → a hard PREFLIGHT that aborts on an unfixed tree. No "TBD"/"add appropriate error handling"/"similar to Task N"/undefined-type references remain; every code step ships the actual content.

**3. Cross-task type/signature consistency.**
- `GpuCapability::static_capability(arch)` is called in the same associated-function form in 0a1 (tests) and 0a2 (impl + tests); 0a1 adds the explicit `use crate::profiler::GpuCapability;` in `mod tests` so the call resolves.
- `ChipProfile.simds_per_cu` is introduced by 0a2 and consumed by 0a2's `latency_score`; 0a1/0a3 that also edit `ChipProfile` do not redefine it. 0a2's numeric roofline test omits `vmem_load_bytes_per_lane` (0a3's field, not yet present) — flagged with an inline note to add it if ordering slips.
- `AchievedBw` gains `dpm_state` (0a3) then `in_model_pct` (0b) as **distinct additive fields** beside the pre-existing `trust_score`; 0b explicitly does not re-add `trust_score`. `compute_achieved_bw` is the **4-arg `dpm_state` form** consistently across 0a3 (introduces) and 0b (calls with `Some("high")`/`Some(&args.dpm_state)`).
- `Roofline::analyze(..., Option<MeasuredBw>)` (0a3 sub-step B) is the signature 0b's `run_dynamic` and 0a3's own `measure_fixture` call; `MeasuredBw { gbps, trust }` is defined once in 0a3.
- `LedgerRow.from_fixture` is the 5-arg caller-derived-`bound_class` form (0a3 sub-step A) everywhere; `Reproducer::require_serve_provenance` (0b) is the structured guard the `kernel_perf_instrument --dynamic` entry point and `oracle-serve-dynamic.sh` both rely on.
- `covered_by_alias` (0a3 sub-step G) is the boundary primitive that 0b's `is_covered`/`coverage_against_aliases` build on — one shared matching core, not two.
- `pmc_census` (0c) and `bill_of_debt` (0d) each register via a single `pub mod` line in `lib.rs`; the plan's swarm map and workflow both resolve that shared edit by unioning the two append-only lines in a terminal commit node.

No signature drift found; all consumer↔producer names and arities match across tasks.