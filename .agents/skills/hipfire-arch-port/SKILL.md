---
name: hipfire-arch-port
description: Port hipfire compute kernels to a new RDNA / CDNA architecture (gfx1201/gfx1200/gfx94x/gfx1150/etc.). Use when adding support for a new GPU arch, fixing arch-specific kernel codegen failures (e.g. "Cannot select intrinsic %llvm.amdgcn.wmma..."), or refactoring rdna-compute arch-conditional dispatch. Captures the workflow for WMMA/MFMA ports, ArchCaps routing, channel/speed validation selection, contributor onboarding, and known correctness traps. Triggers on phrases like "port to gfx12", "9070 XT support", "R9700 support", "WMMA gfx12", "Cannot select intrinsic wmma", "amdgcn.wmma", "new arch port", "cross-arch kernel".
---

# hipfire-arch-port

Executable skill for adding a **GPU ISA arch** (RDNA/CDNA `gfx*`) to hipfire, or fixing arch-specific codegen and dispatch bugs.

This is **not** a model-family port. Model IDs and crates live under [`docs/architecture-ids.md`](../../../docs/architecture-ids.md). GPU routing lives in `crates/rdna-compute/` and is orthogonal to `hipfire-arch-*` crates.

## Scope boundary

| Concern | Owner / place |
|---|---|
| GPU ISA port workflow (this skill) | `.agents/skills/hipfire-arch-port/` (sole executable skill root) |
| Model `arch_id` table | [`docs/architecture-ids.md`](../../../docs/architecture-ids.md) |
| Crate layout / request lifecycle | [`docs/ARCHITECTURE.md`](../../../docs/ARCHITECTURE.md) |
| Claim → validation route selection | [`docs/VALIDATION.md`](../../../docs/VALIDATION.md) |
| Arch-port validation method (tiny oracle, channel, speed) | [`docs/methodology/arch-port-validation.md`](../../../docs/methodology/arch-port-validation.md) |
| Perf claim protocol | [`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md) |
| Product route admission | [`docs/admissions.yml`](../../../docs/admissions.yml) — empty / fail closed |
| Spec-decode after AR is correct | [`speculation.md`](speculation.md) + inventory owner in INDEX |

**Capability ≠ certification.** A kernel file, dispatch branch, or green harness proves implementation existence only. It does **not** admit a default product route, floor, or promotion. Admissions are machine rows in `admissions.yml` only (currently empty). There is **no universal gate**.

## When to use

- HIP codegen / select failure on a new chip (`Cannot select: intrinsic %llvm.amdgcn.wmma...`).
- Adding or finishing `gfx1200` / `gfx1201` / `gfx115x` / `gfx94x` / similar ISA support.
- Refactoring `ArchCaps` predicates or GEMM/GEMV arch branches in `rdna-compute`.
- Contributor has target hardware and wants a fork → port → PR path.

## Read order

1. [`playbook.md`](playbook.md) — load-bearing sequence (start here).
2. [`wmma-matrix.md`](wmma-matrix.md) — operand shapes / builtins / lane layout **reference** (re-verify against local ROCm before extending).
3. [`docs/methodology/arch-port-validation.md`](../../../docs/methodology/arch-port-validation.md) — correctness method (per-layer cosine / channel).
4. [`docs/VALIDATION.md`](../../../docs/VALIDATION.md) — which executable route covers your claim class.
5. [`contributor-onboarding.md`](contributor-onboarding.md) — hardware owner workflow + PR shape.
6. [`speculation.md`](speculation.md) — optional after AR forward is correct (n-gram `SpecTarget`).

Do **not** treat companion files in this directory as owners for mutable inventories (supported chips, default-on paths, admission lists). Those facts live in source + the docs owners above.

## Immediate facts (load-bearing)

- **WMMA gfx11 → gfx12 is not a `#ifdef` name swap.** A/B vectors, `kRepeat`, builtin suffix, and C-mapping differ. See `wmma-matrix.md`; prove C-mapping on hardware before trusting writeback. CDNA ports use MFMA + `is_cdna3` / `is_wave64_native`, not WMMA rename.
- **Route through `ArchCaps`, not ad-hoc string soup.** Predicates live in `crates/rdna-compute/src/arch_caps.rs` (`has_wmma`, `has_wmma_w32`, `has_wmma_w32_gfx12`, `is_rdna4`, `is_gfx1201`, `is_cdna3`, `is_wave64_native`, …). GEMM methods in `crates/rdna-compute/src/gemm.rs` / registration in `kernels.rs` / public dispatch surface in `dispatch.rs`.
- **Kernel naming:** `kernels/src/<base_name>.hip`, or tagged `kernels/src/<base_name>.gfxNNNN.hip` (chip) / `kernels/src/<base_name>.gfxNN.hip` (family). `scripts/compile-kernels.sh` resolves chip → family → base. Compile selected arches first; run `scripts/write-kernel-hashes.sh` only after successful compiles when committing precompiled blobs.
- **Hardware required** for target-arch channel proof. No emulator path for merge of a new ISA route. Required Tier S / model-level / Tier P routes that you cannot run are **blocked** (hand off), not optional.
- **Retired coherence batteries are not acceptance.** Do not require `scripts/coherence-gate*.sh`. Select routes via `docs/VALIDATION.md` (channel + model-level manual after numeric `.hip`; Tier P when forward/state can change; Tier S on every shared baseline arch touched).

## Directory map

| File | Role |
|---|---|
| `SKILL.md` | Entry + boundaries |
| `playbook.md` | Step sequence + traps |
| `wmma-matrix.md` | Local WMMA/MFMA reference (verify vs ROCm) |
| `validation.md` | Skill-local procedure notes; **defer route selection to VALIDATION.md** |
| `contributor-onboarding.md` | Hardware contributor path |
| `speculation.md` | Optional post-AR spec-decode wiring |
| `skill.json` | Manifest / triggers |

## Historical anchors (provenance only)

- gfx11 WMMA C-mapping fix class: commit `b7ac66a` — assume new-arch C-mapping wrong until channel-proven.
- Canonical early gfx12 WMMA pattern file: `kernels/src/gemm_qkv_hfq4g256_wmma.gfx12.hip` (inspect current tree; many gfx12 siblings now exist).
- Canonical early gfx942 MFMA pattern file: `kernels/src/gemm_hfq4g256_residual_mfma.gfx942.hip` (and wave64/MFMA siblings).
- Stale-binary speed-gate artifact (force-rebuild bench binary before A/B): see playbook traps + perf methodology.
- Firmware shadowing (`/lib/firmware/updates/amdgpu`) can fake ~50% prefill drops — system-side, not a code win.

Inspect **current** `kernels.rs` selectors, `gemm.rs` branches, and env gates (e.g. Lloyd / MMQ opt-ins) before claiming a path is default-on.
