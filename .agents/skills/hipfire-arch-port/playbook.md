# hipfire arch-port playbook

Workflow for adding or repairing a **GPU ISA** path (RDNA/CDNA). Read end-to-end before editing dispatch or `.hip` sources.

Model-family work is out of scope here — see [`docs/architecture-ids.md`](../../../docs/architecture-ids.md) and `crates/hipfire-arch-toy/`.

## When to use

- Codegen / intrinsic select failure on a chip hipfire does not yet route correctly.
- New chip or family: `gfx1200` / `gfx1201` / `gfx115x` / `gfx94x` / etc.
- `ArchCaps` or GEMM/GEMV arch-branch refactors in `crates/rdna-compute/`.
- Mainstreaming an env-gated experimental path on hardware that already runs a fallback.

## Companion files (this skill)

| File | Use |
|---|---|
| `playbook.md` (this) | Sequence + traps |
| `wmma-matrix.md` | Operand / builtin / lane reference — **re-check ROCm headers** |
| `validation.md` | Local notes; route selection is **not** owned here |
| `contributor-onboarding.md` | Hardware owner → PR |
| `speculation.md` | Optional step after AR is correct |

## Canonical owners (do not fork inventories)

| Fact | Owner |
|---|---|
| Validation route for a claim class | [`docs/VALIDATION.md`](../../../docs/VALIDATION.md) |
| Arch-port correctness method | [`docs/methodology/arch-port-validation.md`](../../../docs/methodology/arch-port-validation.md) |
| Perf measurement protocol | [`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md) |
| Model `arch_id` ↔ crate | [`docs/architecture-ids.md`](../../../docs/architecture-ids.md) |
| Crate / lifecycle overview | [`docs/ARCHITECTURE.md`](../../../docs/ARCHITECTURE.md) |
| Route admission | [`docs/admissions.yml`](../../../docs/admissions.yml) (empty → fail closed) |
| Docs navigation | [`docs/INDEX.md`](../../../docs/INDEX.md) |

**No universal gate.** Implementation (source exists, branch compiles, one harness green) is never certification or admission.

---

## Modular surfaces (current tree)

Post-modular layout that matters for GPU ports:

```
crates/rdna-compute/src/
  arch_caps.rs     # ArchCaps atoms + has_wmma / has_wmma_w32 / has_wmma_w32_gfx12 / is_rdna4 / is_cdna3 / is_wave64_native / …
  kernels.rs       # include_str! registration of .hip sources
  gemm.rs / gemv.rs / moe.rs / attention.rs / …
  dispatch.rs      # public Gpu method surface (bind_thread, launch)
kernels/src/*.hip  # sources; chip/family tags
scripts/compile-kernels.sh
scripts/write-kernel-hashes.sh

crates/hipfire-dispatch/   # KernelKey tables, MoE Path-2 pipeline (uses ArchCaps)
crates/hipfire-arch-*/     # model forward; call Gpu methods — do not bury ISA tables here
crates/hipfire-runtime/examples/test_kernels.rs
```

**Rule:** ISA routing stays in `rdna-compute` (and shared dispatch tables that read `ArchCaps`). Do not scatter per-chip WMMA tables inside individual model crates unless the kernel is crate-local by design (rare; e.g. arch-tagged LFM scan kernels). Prefer shared `kernels/src` + `rdna-compute` registration.

---

## Workflow (6 load-bearing steps + optional 7)

### 1. Read WMMA/MFMA facts for the target family

Open [`wmma-matrix.md`](wmma-matrix.md), then verify builtins against the **local** ROCm install. Split by target family — do not run only the RDNA path when porting CDNA.

**RDNA (WMMA — e.g. gfx11 / gfx12):**

```bash
rg --no-heading -n 'wmma_f32_16x16x16_f16' /opt/rocm/include/ | head -20
```

Biggest pitfall: treating gfx12 as a gfx11 builtin rename. Operand vector length, K packing, and C-mapping change. Assume C-mapping is wrong until channel-tested on the target GPU.

**CDNA (MFMA — e.g. gfx94x / gfx942):**

```bash
rg --no-heading -n 'mfma_f32_16x16x16' /opt/rocm/include/ | head -20
# also inventory wave64 / MFMA siblings already in-tree
ls kernels/src/*mfma* kernels/src/*gfx942* 2>/dev/null | head
```

CDNA ports use MFMA builtins and wave64-native dispatch, not WMMA. Assume accumulator/lane layout wrong until channel-tested on the target CDNA GPU.

### 2. Map existing routing with `ArchCaps`

Read `crates/rdna-compute/src/arch_caps.rs`. Prefer capability predicates over raw `starts_with`:

| Predicate (examples) | Meaning |
|---|---|
| `has_wmma()` | RDNA3 or RDNA4 WMMA present |
| `has_wmma_w32()` | gfx11-family wave32 WMMA builtins |
| `has_wmma_w32_gfx12()` / `is_rdna4()` | gfx1200/gfx1201 gfx12 builtins |
| `is_cdna3()` | gfx940/gfx941/gfx942 CDNA3 MFMA family |
| `is_wave64_native()` | gfx906/gfx908/CDNA3 wave64-native paths |
| `has_dot2_f32_f16()` | broad fallback family |
| `is_gfx1201()` | chip-strict gates (only when product scope is chip-strict) |

Inspect call sites in `crates/rdna-compute/src/gemm.rs` (and related) that already branch. **RDNA WMMA cascade (example):**

```text
if arch_caps.has_wmma_w32_gfx12() { …_wmma_gfx12… }
else if arch_caps.has_wmma_w32() { …_wmma… }
else if arch_caps.has_dot2_f32_f16() { …_dot2… }
else { baseline }
```

**CDNA MFMA / wave64 cascade (example):**

```text
if arch_caps.is_cdna3() { …_mfma… / …_gfx942… }
else if arch_caps.is_wave64_native() { …_wave64… }
else { baseline / RDNA arm }
```

When adding a more specific arm:

1. Place it **above** broader arms that previously absorbed the chip.
2. Drop now-unreachable literal clauses in the **same** diff.
3. Broad helpers that intentionally cover families (`has_dot2_…`) usually stay wide; edit them only when the definition is wrong for the new chip.
4. Match surrounding style (predicate helper vs existing pattern).

Also check `crates/hipfire-dispatch/` tables/pipeline if the op goes through `KernelKey` / MoE Path-2 — those arms re-derive WMMA from `ArchCaps` and must stay consistent with `rdna-compute`.

### 3. Treat “should-be-no-op” dispatch refactors as measurement hazards

If speed numbers move after a pure routing tidy:

1. Delete the specific bench binary so the next run rebuilds it (speed-gate `ensure_build` is a no-op when the binary already exists).
2. `cargo clean -p rdna-compute` when dispatch artifacts may be stale.
3. Check DPM / thermal / firmware shadowing (`dmesg`, `/lib/firmware/updates/amdgpu`) before blaming codegen.
4. Re-measure with the protocol in [`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md).

Do not bypass hooks with `--no-verify` unless the maintainer authorized that exact change in writing.

### 4. Author kernels as tagged `.hip` files

Naming (unambiguous forms — one tag segment, no doubled dots):

| Form | Covers |
|---|---|
| `kernels/src/<base_name>.gfxNNNN.hip` | chip-only (e.g. `….gfx1201.hip`, `….gfx942.hip`) |
| `kernels/src/<base_name>.gfxNN.hip` | family (e.g. `….gfx12.hip` covers gfx1200+gfx1201 via compile-kernels resolution) |
| `kernels/src/<base_name>.hip` | default / older baseline (no chip/family tag) |

Resolution order in `scripts/compile-kernels.sh`: **chip → family → base**.

Prefer a separate tagged file when operand types or lane layout differ. Single-file `#ifdef` only when types, layout, and tuning are truly identical (rare for WMMA/MFMA).

**Reference patterns to read before forking (by family):**
- **RDNA / WMMA:** `kernels/src/gemm_qkv_hfq4g256_wmma.gfx12.hip` (and current WMMA siblings registered in `crates/rdna-compute/src/kernels.rs`).
- **CDNA / MFMA:** `kernels/src/gemm_hfq4g256_residual_mfma.gfx942.hip` (and current MFMA / wave64 siblings).

Do not assume a short “five remaining kernels” list — inventory the tree for the family you are porting.

Crate-local kernels (include from an arch crate) are exceptional; document why they are not in `kernels/src` if you add one.

### 5. Wire registration + launch path

1. `crates/rdna-compute/src/kernels.rs` — `include_str!` + public `const …_SRC`.
2. `crates/rdna-compute/src/gemm.rs` / `gemv.rs` / … — typed `Gpu` method that `ensure_kernel` + launches (always `bind_thread` on public entry; see `scripts/verify-bind-thread.sh`).
3. Public selector branch using `ArchCaps` (step 2) — WMMA predicates for RDNA, `is_cdna3` / `is_wave64_native` for CDNA.
4. Compile-check every arch you touch **first**, e.g.
   `./scripts/compile-kernels.sh gfx1100 gfx1200 gfx1201` (RDNA) or `./scripts/compile-kernels.sh gfx942` (CDNA) — adjust to the chips under test. All required compiles must succeed.
5. **Only after** successful compiles, when committing precompiled blobs: `./scripts/write-kernel-hashes.sh` so newly produced `.hsaco` blobs get matching trust sidecars. (The hash script documents “run after `compile-kernels.sh`”.)
6. Add or extend a **channel** case that forces the new symbol on the target arch (`test_kernels` and/or a focused example under `hipfire-runtime/examples/`). A port with no numeric coverage on its kernel is incomplete.

### 6. Validate by claim class (not a universal triple)

Select routes from [`docs/VALIDATION.md`](../../../docs/VALIDATION.md). For **arch port** claims the selector points at [`docs/methodology/arch-port-validation.md`](../../../docs/methodology/arch-port-validation.md) (channel + speed; **no** retired coherence battery as acceptance).

Minimum practical loop for a new/changed **numeric** ISA kernel path (GPU arch port):

| Layer | What | Notes |
|---|---|---|
| Tier C — Channel | `cargo build --release --features deltanet -p hipfire-runtime --example test_kernels` then run `test_kernels` **on target hardware** | Element-wise vs CPU reference. Catches bad C-mapping / MFMA layout that “looks fine” in smoke generation. Required on the **target** GPU. |
| Bind invariant | `scripts/verify-bind-thread.sh` (or pre-commit when hooks installed) | Public `dispatch.rs` entries must bind HIP thread. **Not** a numeric test. |
| Model-level manual route | After any new/changed numeric `.hip` path: the model-level manual route for the arch under test (VALIDATION “New/changed `.hip` kernel (numeric)”) | Required — channel alone is not the full GPU-port route. |
| Tier P — Path parity/oracle | Path-specific parity/state oracle whenever the GPU-port change **can alter forward/state** for a model path that has an oracle | Cosine / logit / KV/conv/`n_tokens` as the oracle defines. **Fail closed:** if no oracle exists, leave parity **blocked** — do not substitute serve smoke. |
| Tier S — Speed floor | `scripts/speed-gate.sh --fast` (or full) on **every baseline arch** whose shared edited path/predicate the diff can touch | When a committed `tests/speed-baselines/<arch>.txt` exists for that arch. Not optional when the path is shared. Force-rebuild bench binary before A/B. No matching baseline file → speed-floor claim for that arch is **blocked** until earned. |
| Method / plumbing | Tiny-oracle + per-layer cosine when porting a **model** forward onto existing kernels | See methodology owner — required for model-arch bring-up; also applies under Tier P when a GPU port can break numbers. |
| User-facing serve | `scripts/serve_harness.py` (or LFM harness when LFM-only) **after** numeric/state routes if numbers can break | Semantics only — not parity proof. |
| Perf win claim | Methodology owner + identity hashes | Measured ≠ admitted. |

**Retired:** `scripts/coherence-gate*.sh` (including missing `coherence-gate.sh`) are **historical reproduction only**. Never require them for merge, promotion, or benches. See VALIDATION “Retired coherence-gate scripts.”

**Fail closed:** if no oracle exists for a numerical/state claim, the route is **blocked** — do not substitute serve smoke.

**Capability ≠ certification:** green channel on one card does not flip product defaults or fill `admissions.yml`.

If you lack target hardware (or a required baseline-arch box for Tier S), you cannot complete that route — stop at fallback-safe routing, **record the route as blocked**, and hand off to a hardware holder via [`contributor-onboarding.md`](contributor-onboarding.md). Do not merge on “should be identical” alone.

### 7. (Optional) Speculative decode

Only after AR forward is correct. Follow [`speculation.md`](speculation.md): `impl SpecTarget` + registries in `hipfire-loader`. Under **greedy** verification with argmax fallback on miss, committed tokens match greedy AR for a correct `SpecTarget`. That is **not** a blanket claim for temperature sampling, sampled verification, or emitter/rendering behavior. Learned drafters are separate work.

---

## Known traps

| Trap | Symptom | Response |
|---|---|---|
| Wrong WMMA C-mapping | Garbage mats; may still “generate” | Channel-test; instrument `(tid, acc[j])`; see gfx11 fix class `b7ac66a` |
| Delete “unused” WMMA sources | Large silent slowdown | Grep `include_str!` / `KernelKey` before deleting |
| Stale bench binary A/B | Fake speed regression/gain | `rm` the bench exe; rebuild; re-run |
| Firmware shadowing | ~50% prefill drop after “no-op” | System path `/lib/firmware/updates/amdgpu`; not a code claim |
| Coherence script as merge bar | Policy violation / missing script | Use VALIDATION.md routes only |
| Env-gated path treated as default-on | False product claims | Read current selector + env flags in source |
| Chip-strict feature claimed for whole family | Wrong GPU runs or fails closed incorrectly | Prefer `is_rdna4` vs `is_gfx1201` intentionally; document scope |
| Serve harness as numeric proof | Missed state bugs | Parity/oracle first; serve is semantics |

## Quick links

- WMMA reference → `wmma-matrix.md` (verify ROCm)
- Validation routes → `docs/VALIDATION.md`
- Arch-port method → `docs/methodology/arch-port-validation.md`
- Contributor path → `contributor-onboarding.md`
- Spec-decode → `speculation.md`
- Perf protocol → `docs/methodology/perf-benchmarking.md`
