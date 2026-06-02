# Codebase Quality Assessment Report

**Project**: HipFire — `feature/dispatch-unification`
**Date**: 2026-06-01
**Assessed by**: Nils Balkow-Tychsen (Senior QA Engineer Agent)
**Codebase**: /home/bjoern/hipfire
**Scope**: Branch diff — `feature/dispatch-unification` vs `master` (33 commits)

---

## Executive Summary

This branch introduces a new `hipfire-dispatch` crate that factors GPU kernel
selection into a clean family/table/model_ext architecture, migrates callers
toward it, makes `new-dispatch` a default feature, and removes a large pile of
env-var-gated experimental PARO code paths. The *library design* is genuinely
good: type-safe `KernelKey` enum, declarative dispatch tables, predicate-based
arch/shape gating, zero new external dependencies, and 97 CPU-only tests that
all pass.

However, the branch is **not mergeable in its current state**, for two
independent reasons. First, the actual integration into the model forward
passes — the entire purpose of "dispatch unification" — is **broken**: the
`#[cfg(feature = "new-dispatch")]` forward functions in the arch crates
construct `WeightRef` with only 4 of its 7 required fields, producing **13
`E0063` compile errors in `hipfire-arch-llama` alone** (plus matching breakage
in qwen35/deepseek4). This is masked today only because *nothing in the
workspace actually enables `hipfire-arch-*/new-dispatch`* — the runtime's
`new-dispatch` feature turns on the dispatch *crate*, not the arch crates'
integration feature. So the default build compiles and the old forward path
runs; the new path is dormant, unreachable, and broken-when-activated. Second,
a **206 MB, 611-file Cargo build tree (`target-baseline/`)** was committed
because root `.gitignore` only matches `/target`, not the renamed directory.
Merging this to `master` bloats every future clone's history permanently.

Beyond those, the migration is roughly a third done (4 of 6 kernel families are
table-only stubs, `ResourceManager` is a placeholder, `dispatch.rs` is still a
33,172-line god-file), the whole new crate fails `cargo fmt --check` (would be
rejected by CI), there is no evidence the mandatory `coherence-gate.sh` was run
for a dispatch change, and ~8 tests are empty-body stubs that always pass. The
foundation is solid and worth keeping; the branch needs another pass before it
lands.

Overall Health Score: **6.5 / 10 (Grade C)**

### Quality Scorecard

| Dimension | Score | Grade | Critical | Warnings | Suggestions | Notes |
|-----------|-------|-------|----------|----------|-------------|-------|
| Security Posture | 9/10 | A | 0 | 0 | 1 | 1 |
| Test Coverage & Quality | 6/10 | C | 0 | 3 | 3 | 1 |
| Architecture & Design | 5/10 | D | 1 | 3 | 3 | 1 |
| DevOps & CI/CD | 4/10 | D | 1 | 2 | 1 | 0 |
| Code Style & Consistency | 6/10 | C | 0 | 3 | 2 | 0 |
| Documentation | 7/10 | B | 0 | 0 | 2 | 1 |
| Dependency Health | 8/10 | B | 0 | 1 | 0 | 1 |
| **OVERALL** | **6.5/10** | **C** | **2** | **12** | **12** | **5** |

### Top 3 Strengths
1. **Clean, type-safe dispatch abstraction.** The `hipfire-dispatch` crate
   (`types.rs`, `families/`, `tables/`) centralizes kernel selection behind a
   flat `KernelKey` enum with init-time-validated tables — adding a quant format
   is a single-location change (`crates/hipfire-dispatch/src/types.rs`).
2. **Zero new external dependencies.** The full `Cargo.lock` diff is 20 lines,
   all internal path-dep edges; new crates use `version.workspace`/
   `edition.workspace` consistently.
3. **97 CPU-only tests, all passing, CI-runnable** (verified: 54 in
   `hipfire-dispatch-tests`, 43 in `hipfire-dispatch/src/tests.rs`, finished in
   0.00s with no GPU). The non-stub ones assert real routing decisions.

### Top 3 Concerns
1. **[CRITICAL] The new-dispatch arch integration does not compile** — 13×
   `E0063` in `hipfire-arch-llama` (verified), and the feature that would expose
   it is wired to nothing, so the "unification" is unreachable.
2. **[CRITICAL] 206 MB / 611-file `target-baseline/` build tree committed** —
   permanent history bloat; must be removed via branch history rewrite before
   merge.
3. **[WARNING] No evidence the mandatory `coherence-gate.sh` ran** for a change
   that touches dispatch/rotation — CLAUDE.md requires it; tests cover routing
   logic only, never numerical output.

---

## Codebase Profile

- **Language / build**: Rust workspace (Cargo), HIP/ROCm GPU backend
  (`rdna-compute`, `hip-bridge`), `.hip` kernels under `kernels/`.
- **Diff size**: 64 source files, **+8,530 / −3,574** (excluding artifacts);
  plus 611 committed build-artifact files (`target-baseline/`).
- **New crates**: `hipfire-dispatch` (library), `hipfire-dispatch-tests`
  (test-only).
- **Touched crates**: `hipfire-arch-{llama,qwen2,qwen35,deepseek4}`,
  `hipfire-runtime`, `rdna-compute`, `hipfire-quantize`.
- **Removed**: `paro_la_gates_codec.rs` (321), `test_gemv_paro4g128.rs` (665),
  `fused_rmsnorm_paro4g128t_rotate.hip` (186), and ~12 `HIPFIRE_PARO_*` env-var
  gates.
- **Design docs added**: `.opencode/plans/2026-05-30-hipfire-dispatch.md` (672),
  `.opencode/plan/arch-routing-centralization.md` (746).
- **Test framework**: native `cargo test`; pre-commit hook at
  `.githooks/pre-commit` (coherence gate) configured via `core.hooksPath`.

---

## Critical Findings (S1) — Immediate Action Required

### [ARCH-001] The new-dispatch arch integration does not compile

**Location**: `crates/hipfire-arch-llama/src/arch.rs:189-207` (and 328, 357-365,
…); same pattern in `crates/hipfire-arch-qwen35/src/qwen35.rs` and
`crates/hipfire-arch-deepseek4/src/forward.rs`.
**Category**: Architecture / Code Quality
**Time Horizon**: Short-term (now-problem)

**What**: The `#[cfg(feature = "new-dispatch")]` forward functions construct
`WeightRef` with only 4 fields:

```rust
gemv.run_auto(&ctx, gpu, &WeightRef {
    buf: &layer.wq.buf, dtype: layer.wq.gpu_dtype, m: layer.wq.m, k: layer.wq.k,
}, &scratch.x_rot, &scratch.q)?;
```

but `WeightRef` (`crates/hipfire-dispatch/src/families/gemv.rs:33-41`) has **7
required fields** (`buf, dtype, m, k, row_stride, rotation, awq_scale`), derives
no `Default`, and the literals use no `..Default::default()`.

**Why it matters**: This is the actual deliverable of the branch. It is broken.

**Evidence** (verified by running the compiler — not inferred):
```
$ cargo check -p hipfire-arch-llama --features new-dispatch
error[E0063]: missing fields `awq_scale`, `rotation` and `row_stride` in initializer of `WeightRef<'_>`
... (13 occurrences)
error: could not compile `hipfire-arch-llama` (lib) due to 13 previous errors
```
It is masked because **no Cargo manifest enables `hipfire-arch-*/new-dispatch`**.
`crates/hipfire-runtime/Cargo.toml:51` defines `new-dispatch = ["hipfire-dispatch"]`
(the crate, not the arch features), so the default workspace build compiles
(`cargo check -p hipfire-runtime` → Finished in 0.83s, verified) by selecting the
old `#[cfg(not(feature = "new-dispatch"))]` path. The new path is dead-compiled-out.

**Recommendation**: Replace every hand-written `WeightRef { … }` in the arch
crates with the existing adapter `WeightTensor::dispatch_ref()` (added in commit
`bb86cd49`, already used correctly in `hipfire-runtime/src/llama.rs:558`), or add
the three missing fields. Then add a CI job that builds with the arch features
on (`--features hipfire-arch-llama/new-dispatch`, etc.) so this can never regress
silently again. Until then, the branch title oversells what is reachable.

**Effort**: Moderate (mechanical fix + CI wiring).

---

### [DEVOPS-001] 206 MB / 611-file Cargo build tree committed to history

**Location**: `target-baseline/` (611 files added on this branch)
**Category**: DevOps / Git Hygiene
**Time Horizon**: Short-term (permanent if merged)

**What**: A full release build tree was committed: 58 `.rlib`, 58 `.rmeta`,
example ELF binaries (`bench_qwen35_mq4` 4.95 MB, `profile_qwen35_mq4` 4.55 MB),
`.d` depfiles, fingerprints, and timestamps. Largest blob:
`target-baseline/release/deps/librdna_compute-*.rlib` (11 MB). Total **206 MB**.

**Why it matters**: Root `.gitignore:2` ignores only `/target`; the renamed
`target-baseline/` dodges it. Build artifacts in history can never be cleaned by
a later delete commit — the blobs persist in every clone's `.git` forever.

**Evidence**: `git diff master...HEAD --diff-filter=A --name-only -- target-baseline | wc -l` → 611;
on-disk size 206 MB; `.gitignore` contains only `/target`.

**Recommendation**: This is **not yet on master**, so fix it on the branch
before merge: (1) add `target-baseline/` (or `**/target*/`) to `.gitignore`;
(2) `git rm -r --cached target-baseline`; (3) rewrite branch history
(`git rebase`/`git filter-repo`) to drop the blobs — a plain delete-and-recommit
leaves them in history. No master rewrite is needed since the blobs originate on
this branch.

**Effort**: Quick Win (the rewrite is branch-local).

---

## Warning Findings (S2) — Fix This Sprint

### [TEST-001] ~8 stub tests with empty bodies always pass
**Location**: `crates/hipfire-dispatch-tests/src/deepseek4.rs:20-50` (4 empty),
`qwen2.rs:19-30` (2 empty), `qwen35.rs:85-91` (1 empty).
**What**: Functions like `deepseek4_has_separate_mtp_layer()` contain only a
doc-comment and no assertions (verified: `deepseek4.rs` has 5 `#[test]` but only
3 `assert` lines; `qwen2.rs` 3 tests / 3 asserts). They pass unconditionally.
**Why**: They create false coverage confidence for exactly the per-model dispatch
paths most likely to regress. **Recommendation**: add assertions or mark
`#[ignore]` with a tracking ticket; do not ship green tests that assert nothing.
**Effort**: Moderate.

### [TEST-002] No coherence-gate evidence for a dispatch change
**Location**: branch-wide (rotation, GEMV, Q8HFQ, ParoQ4G128 routing changed).
**What**: CLAUDE.md: "Any change to kernels, quant formats, dispatch, fusion,
rotation, rmsnorm, or the forward pass MUST pass `./scripts/coherence-gate.sh`."
Tests validate *routing decisions* only; no test or artifact shows numerical
output was compared against the legacy path. **Why**: a dispatch refactor's
signature failure mode is silent mis-routing to a numerically-wrong-but-running
kernel, which routing tests cannot catch. **Recommendation**: run the gate once
the integration compiles; attach the report to the PR. **Effort**: Moderate
(needs GPU).

### [ARCH-002] `ResourceManager` is dead-code scaffolding
**Location**: `crates/hipfire-dispatch/src/resource/mod.rs:1-18` —
`pub struct ResourceManager { _priv: () }`, carried in `DispatchCtx.resources`
(`context.rs`) but never read. **Why**: the design doc promised persistent MQ
rotation-buffer management here; only a placeholder shipped. **Recommendation**:
implement it or delete the struct + field; don't merge an empty abstraction.
**Effort**: Moderate.

### [ARCH-003] 4 of 6 families and the pipeline executor are stubs
**Location**: `crates/hipfire-dispatch/src/families/{gemm,fused_qkv,attention,
moe}.rs` (tables exist, no working `run`); `pipeline/mod.rs:28-128` handles only
`RotateFwht` and `GemvMfp4G32Fused`, returns `UnsupportedVariant` for the rest of
the `PipelineOp` enum. **Why**: only Rotation + GEMV are functional. The crate
advertises a 6-family surface that's ~⅓ real. **Recommendation**: gate or
clearly mark the stub families as unimplemented so callers can't route into a
runtime error; track remaining families as explicit follow-ups. **Effort**:
Significant.

### [STYLE-001] Entire new crate fails `cargo fmt --check`
**Location**: `hipfire-dispatch` — 129 diff hunks across **15 of 16** source
files (verified; only `traits.rs` clean). **Why**: a CI/pre-commit `fmt --check`
rejects the branch. **Recommendation**: `cargo fmt -p hipfire-dispatch`.
**Effort**: Quick Win.

### [STYLE-002] Six dead feature flags
**Location**: `crates/hipfire-dispatch/Cargo.toml:12-17` —
`new-rotation/new-gemv/new-gemm/new-fused-qkv/new-attention/new-moe` have **zero
`#[cfg(feature=…)]` usages** (only `from-hip-error` is used). Leftovers from the
incremental-migration plan that `2e06affd` superseded. **Recommendation**: delete
them. **Effort**: Quick Win.

### [CORR-001] `row_stride` hard-coded to 0 in the compiled runtime path
**Location**: `crates/hipfire-runtime/src/llama.rs:651, 1014, 1021, 1028, 1128,
1266`. **What**: the *compiled* new-dispatch path (runtime's new-dispatch IS on
by default) builds `WeightRef { … row_stride: 0, rotation: None, awq_scale: None }`
by hand instead of using `dispatch_ref()`. **Why**: `dispatch_plain` keys
Q8HFQ off `row_stride` for its padded layout
(`hipfire-dispatch/src/families/gemv.rs`); a hard 0 silently mis-strides if a
Q8HFQ weight ever flows here. Latent today (llama doesn't emit Q8HFQ), but it's a
mis-route waiting to happen, and it duplicates the adapter that exists precisely
to avoid this. **Recommendation**: route all construction through
`WeightTensor::dispatch_ref()`. **Effort**: Quick Win.

### [DEVOPS-002] Stray npm lockfile in a bun tooling dir
**Location**: `.opencode/package-lock.json` (380 lines, npm v3). `.opencode/`
ignores `bun.lock`/`package.json`/`node_modules` (it's a bun dir), so this npm
lockfile slipped in uncovered. **Recommendation**: `git rm --cached` it and add
to `.opencode/.gitignore`. The two plan `.md` files are intentional — keep.
**Effort**: Quick Win.

### [DEPS-001] / [ARCH-004] `dispatch.rs` remains a 33,172-line god-file
**Location**: `crates/rdna-compute/src/dispatch.rs`. **What**: verified
34,284 → 33,172 lines — the refactor removed only ~1,100 net lines of a 33k-line
file. Dead `gemv_paro4g128*` / `fused_*_paro4g128t` methods remain after the
PARO4G128/T dtypes were dropped (`1a02aabe`). **Why**: the stated goal was to
*centralize* dispatch; the central file is barely dented and still mixes dozens
of concerns. **Recommendation**: schedule extraction of the now-orphaned PARO
methods and migrate remaining families out before claiming the file shrank.
**Effort**: Significant. *(Counted under Architecture; listed here for the
god-file metric.)*

---

## Detailed Analysis

### 1. Security Posture (Score: 9/10, A)

#### Strengths
Refactor with a small security surface: no new external dependencies, no secret
handling, no input-parsing changes, no network/IO surface added. `Cargo.lock`
delta is internal path edges only.

#### Findings
- **[S3] Panic-on-misroute in dispatch hot path** —
  `crates/hipfire-dispatch/src/families/attention.rs:91-108` calls `.unwrap()` on
  `Option` params (`givens_cos`/`givens_sin`) in live (non-test) code. For a
  crate whose own error type is `DispatchError`, a bad key should return an error,
  not panic. Robustness > security here, but worth fixing.
- **[S4]** No secrets, no injection vectors, no auth surface touched.

#### Security Posture Checklist
- Secrets management: PASS (none introduced)
- Authentication / Authorization: N/A
- Input validation: PARTIAL (`.unwrap()` on dispatch params)
- Dependency security: PASS (no new deps)
- Infrastructure hardening: N/A
- Security tooling: N/A (unchanged)

### 2. Test Coverage & Quality (Score: 6/10, C)

#### Summary Metrics
| Metric | Value |
|--------|-------|
| Total test functions | 97 (43 dispatch + 54 dispatch-tests) |
| Pass rate (verified) | 97/97, 0 failed, GPU-free |
| Stub tests (no assertions) | ~8 |
| End-to-end / coherence tests | 0 |

#### Strengths
Routing logic is well covered: `ShapePredicate`/`ArchPredicate` evaluation,
`KernelRegistry` resolve/fallback, `KernelKey::for_gemv*` dtype→key maps,
`dtype_rotation_plan`/`dtype_post_rotation_variant`, and a 20-arch capability
matrix. Non-stub tests carry real assertions (~220 total). All CPU-only — CI can
run them with no hardware.

#### Findings
See **TEST-001** (stub tests), **TEST-002** (no coherence validation). Plus:
- **[S3]** No test exercises `GemvFamily::run_auto()` / `RotationFamily::run()`
  end-to-end — only `resolve()`. A wrong-variant selection in `run_auto` ships
  undetected.
- **[S3]** Model-test arch lists omit `gfx1103` (RDNA3 iGPU) and CDNA3
  (`gfx94x`) — relevant for Qwen3.5 MoE on MI300X
  (`crates/hipfire-dispatch-tests/src/qwen35.rs:41`).
- **[S3]** DRY: arch lists hard-duplicated across `llama.rs`/`qwen35.rs`/
  `deepseek4.rs`; a new arch won't auto-appear in coverage.
- **[S4]** None of the tests would have caught ARCH-001 (they don't build the
  arch crates with their feature on) — a coverage blind spot, not a test bug.

### 3. Architecture & Design (Score: 5/10, D)

#### Overview
```
hipfire-dispatch/
├── types.rs          KernelKey enum + predicates  (single source of truth)
├── context.rs        DispatchCtx (carries dead ResourceManager)
├── families/         gemv ✓  rotation ✓  gemm ✗  fused_qkv ✗  attention ✗  moe ✗
├── tables/           declarative dtype→key registration  (good)
├── model_ext/        deepseek4 / qwen35 model-specific (partial scaffolding)
├── pipeline/         executor handles 2 of ~10 PipelineOps
└── resource/         ResourceManager — placeholder, never used
```
Dependency direction is clean and one-way: `hipfire-dispatch → rdna-compute →
hip-bridge`; arch crates depend on dispatch via an (orphaned) optional feature.
No circular deps.

#### Strengths
The family/table split is a sound abstraction that earns its complexity for the
two real families; `KernelKey` centralization genuinely removes per-model
dtype-matching boilerplate; init-time table validation fails fast.

#### Findings
**ARCH-001** (broken integration, Critical), **ARCH-002** (dead ResourceManager),
**ARCH-003** (stub families/pipeline), **ARCH-004** (33k-line god-file). Plus:
- **[S3] Inconsistent family ownership** — `deepseek4/forward.rs:139` caches
  `GemvFamily` in a `static OnceLock`; `llama/arch.rs` constructs
  `GemvFamily::new()` per call. Different perf characteristics; pick one.
- **[S3] Two parallel new-dispatch integrations** — one in
  `hipfire-runtime/src/llama.rs` (compiles, has CORR-001) and one in
  `hipfire-arch-llama/src/arch.rs` (doesn't compile). Unclear which is canonical;
  the duplication is itself a half-migrated smell.
- **[S3] Dead PARO4G128T `Gpu` methods** linger in `dispatch.rs` after the dtypes
  were removed.
- **[S4]** Design intent is well captured in `.opencode/plans/`.

### 4. DevOps & CI/CD (Score: 4/10, D)

#### Strengths
Pre-commit coherence hook exists (`.githooks/pre-commit`, `core.hooksPath` set);
clean source-file deletions verified (no dangling refs).

#### Findings
**DEVOPS-001** (206 MB artifact tree, Critical), **DEVOPS-002** (stray npm
lockfile). Plus:
- **[S2-implied] CI gap**: there is no CI configuration that builds the arch
  crates with `new-dispatch` on, which is why ARCH-001 went unnoticed; and
  `fmt --check` would currently fail the branch (STYLE-001). The deployment-safety
  posture of *this branch* is poor: it ships a non-building feature config and
  100+ MB of binaries.

#### Environment / artifacts table
| Item | Status |
|------|--------|
| `.gitignore` covers build dirs | FAIL (only `/target`, not `target-baseline/`) |
| Build artifacts excluded from VCS | FAIL (611 files / 206 MB committed) |
| Default build compiles | PASS (`cargo check -p hipfire-runtime`, 0.83s) |
| Feature-on build compiles | FAIL (`--features new-dispatch` → 13 E0063) |
| Formatting enforced/clean | FAIL (129 fmt hunks) |

### 5. Code Style & Consistency (Score: 6/10, C)

#### Strengths
No commented-out dead code in migrated files; naming is consistent with the
workspace; feature-flag flattening (`2e06affd`) reduced cfg noise.

#### Findings
**STYLE-001** (fmt fails, 15/16 files), **STYLE-002** (6 dead features),
**CORR-001** (hand-built WeightRef vs adapter). Plus:
- **[S3] ~20 dead-code warnings** in the new-dispatch arch path (unused
  `func`/`arch`/`name` vars, unused imports `HipResult`, `Path`, `HashMap`,
  `DType`) — surfaced when building with the feature on.
- **[S4]** `.opencode/plan/` vs `.opencode/plans/` — duplicate dir, likely typo.

#### Type-safety note
The `KernelKey`/predicate design is strongly typed and a real plus; the
`.unwrap()`-on-`Option` pattern (attention.rs) is the main hole.

### 6. Documentation (Score: 7/10, B)

#### Strengths
Two substantial, genuinely useful design docs
(`.opencode/plans/2026-05-30-hipfire-dispatch.md` 672 lines,
`.opencode/plan/arch-routing-centralization.md` 746 lines) capture the refactor
intent and phasing.

#### Findings
- **[S3] Public API under-documented** — core public enums in
  `crates/hipfire-dispatch/src/types.rs` (`PipelineOp`, `GemvVariant`,
  `FusedQkvVariant`, `AttentionVariant`, `MoeVariant`) carry no doc comments,
  while siblings (`RotationPlan`) do. Non-obvious variants (`KvWriteQ8_0` vs
  `KvWrite`) need rustdoc for a crate meant to be the canonical dispatch surface.
- **[S3]** The plan docs describe `ResourceManager` and a full 6-family pipeline
  as if shipped; reality is partial — docs should mark what's scaffolding.
- **[S4]** No CHANGELOG entry for the default-feature flip.

### 7. Dependency Health (Score: 8/10, B)

#### Strengths
**Zero new external crates** (verified `Cargo.lock` diff = internal path edges).
New crates inherit `version.workspace`/`edition.workspace`; lock file committed;
deletions reference-clean.

#### Findings
- **[S2]** Dead feature declarations (STYLE-002) are also a dependency-hygiene
  issue: they imply a granular opt-in surface that doesn't exist.
- **[S4]** Feature wiring is otherwise coherent — old path stays reachable via
  `--no-default-features`.

---

## Suggestion & Note Findings (S3/S4)

| ID | Severity | Category | Finding | Location |
|----|----------|----------|---------|----------|
| S3 | Suggestion | Security | `.unwrap()` on Option params in dispatch | `families/attention.rs:91-108` |
| S3 | Suggestion | Testing | No `run_auto()`/`run()` e2e test | dispatch-tests |
| S3 | Suggestion | Testing | Arch coverage gaps (gfx1103, CDNA3) | `dispatch-tests/src/qwen35.rs:41` |
| S3 | Suggestion | Testing | Arch lists duplicated (no DRY) | `dispatch-tests/src/*.rs` |
| S3 | Suggestion | Architecture | Family ownership inconsistent (OnceLock vs new()) | `deepseek4/forward.rs:139` vs `llama/arch.rs` |
| S3 | Suggestion | Architecture | Two parallel new-dispatch integrations | runtime/llama.rs vs arch crates |
| S3 | Suggestion | Architecture | Dead PARO4G128T Gpu methods linger | `rdna-compute/src/dispatch.rs` |
| S3 | Suggestion | Style | ~20 dead-code warnings (feature on) | arch new-dispatch path |
| S3 | Suggestion | Docs | Public dispatch enums lack rustdoc | `dispatch/src/types.rs` |
| S3 | Suggestion | Docs | Plan docs overstate shipped scope | `.opencode/plans/` |
| S4 | Note | Style | `.opencode/plan` vs `plans` dir typo | `.opencode/` |
| S4 | Note | Architecture | `HasCdna3LdsGemv` predicate unused | `dispatch/src/types.rs` |
| S4 | Note | Deps | No new external crates (positive) | `Cargo.lock` |
| S4 | Note | Docs | No CHANGELOG for default-feature flip | — |
| S4 | Note | Testing | Existing non-stub tests are high quality | `dispatch/src/tests.rs` |

---

## Action Plan

### P0 — Immediate (This Week, blocks merge)
| # | Action | Category | Effort | Finding |
|---|--------|----------|--------|---------|
| 1 | Fix `WeightRef` construction in arch crates (use `dispatch_ref()`); add CI build with arch `new-dispatch` on | Architecture | Moderate | ARCH-001 |
| 2 | `.gitignore` `target-baseline/`, `git rm --cached`, rewrite branch history to drop 206 MB | DevOps | Quick Win | DEVOPS-001 |
| 3 | `cargo fmt -p hipfire-dispatch` | Style | Quick Win | STYLE-001 |
| 4 | Run `coherence-gate.sh` once integration compiles; attach report | Testing | Moderate | TEST-002 |

### P1 — Short-term (This Sprint)
| # | Action | Category | Effort | Finding |
|---|--------|----------|--------|---------|
| 5 | Assert-or-`#[ignore]` the ~8 stub tests | Testing | Moderate | TEST-001 |
| 6 | Implement or delete `ResourceManager` | Architecture | Moderate | ARCH-002 |
| 7 | Route runtime `WeightRef` through `dispatch_ref()` (kill `row_stride:0`) | Correctness | Quick Win | CORR-001 |
| 8 | Delete 6 dead feature flags; `git rm` stray npm lockfile | Style/DevOps | Quick Win | STYLE-002 / DEVOPS-002 |

### P2 — Medium-term (Next 1-3 Sprints)
| # | Action | Category | Effort | Finding |
|---|--------|----------|--------|---------|
| 9 | Mark/gate the 4 stub families + pipeline ops as unimplemented | Architecture | Significant | ARCH-003 |
| 10 | Add `run_auto()`/`run()` e2e dispatch tests + gfx1103/CDNA3 coverage | Testing | Moderate | TEST e2e |
| 11 | Return `DispatchError` instead of `.unwrap()` on dispatch params | Security | Quick Win | attention.rs |
| 12 | rustdoc the public dispatch enums | Docs | Quick Win | types.rs |

### P3 — Long-term (Next Quarter)
| # | Action | Category | Effort | Finding |
|---|--------|----------|--------|---------|
| 13 | Extract orphaned PARO methods; continue shrinking `dispatch.rs` | Architecture | Significant | ARCH-004 |
| 14 | Unify the two new-dispatch integration sites | Architecture | Moderate | parallel paths |

### Quick Wins Summary
`.gitignore`+history rewrite (#2), `cargo fmt` (#3), `dispatch_ref()` for
runtime (#7), delete dead features + npm lockfile (#8), `DispatchError` instead
of unwrap (#11), rustdoc enums (#12).

---

## Maintainability Assessment by Time Horizon

### Short-term ("Does it work?")
- Default build compiles: **YES** (verified, 0.83s)
- Feature-on build compiles: **NO** (13 E0063 — ARCH-001)
- Tests pass: **YES** (97/97, CPU-only)
- Coherence validated: **NO** (TEST-002)
- Deployment safety: **NO** (206 MB artifacts; non-building feature config)

### Medium-term ("Can it be changed?")
- Readable by new developers: **PARTIALLY** (clean crate; god-file + stubs remain)
- Well-documented: **PARTIALLY** (design docs good; API rustdoc missing)
- Consistent patterns: **PARTIALLY** (family-init + WeightRef construction vary)
- Clear module boundaries: **YES** (one-way dep graph, no cycles)

### Long-term ("Will it last?")
- Architecture sustainable: **YES** (the family/table design is the right shape)
- Dependencies maintained: **YES** (no new external deps)
- Technical debt managed: **PARTIALLY** (debt documented in plans, but stubs +
  dead code shipped)
- Migration paths exist: **YES** (old path reachable via `--no-default-features`)

---

## Score History

| Date | Overall | Security | Testing | Architecture | DevOps | Style | Docs | Deps |
|------|---------|----------|---------|--------------|--------|-------|------|------|
| 2026-06-01 | 6.5 (C) | 9.0 | 6.0 | 5.0 | 4.0 | 6.0 | 7.0 | 8.0 |

---

## Methodology

Assessment performed using the Nils QA Agent framework, scoped to the
`feature/dispatch-unification` vs `master` diff.
- 4 parallel subagents (architecture, tests, correctness/mis-route, deps/style/
  git-hygiene) over the branch diff, plus direct verification by the lead
  reviewer.
- **Claims verified by running the compiler/tests**, not inferred: default build
  (`cargo check -p hipfire-runtime`, pass), feature-on build
  (`cargo check -p hipfire-arch-llama --features new-dispatch`, 13 E0063),
  test suites (`cargo test -p hipfire-dispatch -p hipfire-dispatch-tests`,
  97 pass), `dispatch.rs` line counts, `fmt --check`, and the 611-file/206 MB
  artifact measurement.
- Scoring: 1-10 per category, weighted overall. Severity: Critical (-2),
  Warning (-1), Suggestion (-0.25), Note (0).
- All findings reference specific files and line numbers.
