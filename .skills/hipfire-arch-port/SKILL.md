---
name: hipfire-arch-port
description: Port hipfire compute kernels to a new RDNA / CDNA architecture (gfx1201/gfx1200/gfx94x/gfx1150/etc.). Use when adding support for a new GPU arch, fixing arch-specific kernel codegen failures (e.g. "Cannot select intrinsic %llvm.amdgcn.wmma..."), or refactoring dispatch.rs's arch-conditional branches. Captures the WMMA operand-shape matrix, builtin name table per arch, dispatch routing convention, validation procedure (channel-test / coherence-gate / speed-gate), contributor onboarding workflow, and known correctness traps. Triggers on phrases like "port to gfx12", "9070 XT support", "R9700 support", "WMMA gfx12", "Cannot select intrinsic wmma", "amdgcn.wmma", "new arch port", "cross-arch kernel".
---

# hipfire-arch-port

Skill for adding a new GPU arch to hipfire (or fixing arch-specific
codegen / dispatch issues). Most of the mistakes prior arch ports
have hit are documented here so you don't repeat them.

## When to use

- A user reports a HIP codegen / kernel-select failure on a new arch
  (e.g. issue #54: `Cannot select: intrinsic %llvm.amdgcn.wmma...`
  on gfx1201).
- You're adding `gfx1200` / `gfx1201` / `gfx1151` / `gfx1152` /
  `gfx94x` / `gfx950` to the supported list.
- You're refactoring `dispatch.rs`'s arch-conditional branches.

## Read these in order

1. `playbook.md` — top-level workflow (6 load-bearing steps from
   reading the matrix to running the three gates). Start here.
2. `wmma-matrix.md` — operand-shape × builtin × lane-layout table
   per arch. The single biggest pitfall: assuming a `#ifdef` macro
   swap of the WMMA builtin is enough. It isn't — operand vector
   lengths halve between gfx11 and gfx12.
3. `validation.md` — the three gates every port must pass:
   channel-test (correctness via `test_kernels`), coherence-gate
   (output sanity), speed-gate (no regression on the baseline arch).
4. `contributor-onboarding.md` — workflow for someone with hardware
   who wants to land a port. Designed for collaboration with an
   agent (Claude Code / Cursor / Codex); includes guardrails.

## Key facts to surface immediately

- **WMMA on gfx12 is not a `#ifdef` macro swap.** A and B vector
  lengths are `<8 x fp16>` (vs `<16 x fp16>` on gfx11), `kRepeat` is
  1 (vs 2), and the builtin name is `_w32_gfx12` (vs `_w32`). Per-
  lane K-packing differs. See `wmma-matrix.md`.
- **Run the speed-gate on every dispatch.rs change.** Do not bypass
  with `--no-verify` — the repo treats that as a contract violation
  unless explicitly authorized in writing by the maintainer.
- **No unreachable dispatch branches.** When you add a more specific
  check that absorbs an arch previously handled by a broader check,
  narrow the broader check in the same diff. Predicate helpers like
  `has_dot2_f32_f16` that legitimately cover broad families don't
  need narrowing — only literal `|| starts_with(...)` clauses that
  become unreachable.
- **Hardware required.** A new arch port cannot be merged without
  empirical channel-test on real hardware for the target arch.
  There is no emulator path. If you don't have hardware, find a
  contributor who does (see `contributor-onboarding.md`).

## What's in this directory

| File | Purpose |
|---|---|
| `SKILL.md` (this) | Entry point with frontmatter; loads the skill |
| `skill.json` | Framework-agnostic manifest (older convention, kept for compatibility with this repo's existing skill tooling) |
| `playbook.md` | Top-level workflow + when-to-use + known-traps table |
| `wmma-matrix.md` | Per-arch operand shape, builtin name, lane layout |
| `validation.md` | Three-gate procedure with troubleshooting |
| `contributor-onboarding.md` | Fork → port → PR workflow with agent-assist guidance |

## Cross-references in repo memory

- `memory/project_wmma_correctness_fix.md` — gfx11 C-mapping was
  silently wrong for 6 weeks; assume any new arch's C-mapping is
  wrong until proven by channel-test on hardware.
- `memory/feedback_quality_gate_baselines_degenerate.md` — greedy
  decode degenerates on Qwen3.5; relevant when validating a port
  with thinking-model coherence-gate prompts.
- `memory/feedback_firmware_shadowing_perf_trap.md` — `/lib/
  firmware/updates/amdgpu` shadowing causes ~50% prefill drop that
  looks like a code regression but is environmental.
