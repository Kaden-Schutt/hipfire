---
name: hipfire-arch-port
description: Port hipfire compute kernels to a new RDNA / CDNA architecture (gfx1201/gfx1200/gfx94x/gfx1150/etc.). Use when adding support for a new GPU arch, fixing arch-specific kernel codegen failures (e.g. "Cannot select intrinsic %llvm.amdgcn.wmma..."), or refactoring dispatch.rs's arch-conditional branches. Captures the WMMA operand-shape matrix, builtin name table per arch, dispatch routing convention, validation procedure (channel-test / coherence-gate / speed-gate), contributor onboarding workflow, and known correctness traps. Triggers on phrases like "port to gfx12", "9070 XT support", "R9700 support", "WMMA gfx12", "Cannot select intrinsic wmma", "amdgcn.wmma", "new arch port", "cross-arch kernel".
---

# hipfire-arch-port

Skill for adding a new **GPU arch** to hipfire (or fixing arch-specific
codegen / dispatch issues). Most of the mistakes prior arch ports
have hit are documented here so you don't repeat them.

> **Note (post-0.1.20):** This skill is about **RDNA/CDNA hardware
> archs** (gfx1100, gfx1201, gfx94x, etc.) — NOT model archs.
> If you want to add a new MODEL family (Llama-3, Mixtral, etc.) you
> implement the `Architecture` trait in your own crate. See
> `crates/hipfire-arch-toy/` as a reference template and
> `CONTRIBUTING.md`'s "Crate topology" section. RDNA arch routing
> stays in `crates/rdna-compute/src/dispatch.rs` regardless of model
> arch — they're orthogonal.

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

## Cross-references (commits + ops notes)

- **gfx12 bring-up history (issue #54):** commit `6e100c2` routed
  gfx1200/gfx1201 to a safe fallback to eliminate the original codegen
  crash on 9070 XT.
- **Canonical gfx12 WMMA pattern:** commit `6924f2a` added
  `kernels/src/gemm_qkv_hfq4g256_wmma.gfx12.hip` as the worked-out
  pattern reference. Current checkouts also contain additional gfx12
  WMMA sibling kernels and selectors; inspect `crates/rdna-compute/src/kernels.rs`,
  `crates/rdna-compute/src/dispatch.rs`, and env gates such as
  `HIPFIRE_LLOYD_GFX12` before claiming a path is enabled by default.
- **WMMA correctness fix (gfx11):** commit `b7ac66a` ("wmma
  correctness fix + MQ6 family + cross-arch prefill + gate
  framework"). The gfx11 C-mapping (`acc[j] = C[2*j + (tid>>4)]
  [tid & 15]`) was silently wrong for ~6 weeks before being caught.
  Assume any new arch's C-mapping is wrong until proven by
  channel-test on hardware.
- **Greedy decode degeneracy (Qwen3.5):** thinking-model prompts
  often emit empty `<think><|im_end|>` at `--temp 0` because the
  reasoning step exhausts max_tokens without closing. Use
  `--temp 0.3 --repeat-penalty 1.05` and `--max-tokens 1500+` for
  9b in coherence-gate-style validation.
- **Stale-binary speed-gate measurement (resolved 2026-04-27):**
  the speed-gate's `ensure_build()` is a no-op when the bench
  binary already exists, so "stash and re-bench" verification can
  measure the same code twice. Always `rm
  target/release/examples/bench_qwen35_mq4` before re-running the
  gate to compare diffs. See `validation.md` troubleshooting table.
- **Firmware shadowing (perf trap):** if the speed-gate flags a
  ~50% prefill drop after a "should-be-no-op" change AND the cause
  isn't the stale binary above, check `dmesg | tail` for SMU IF
  mismatch. The fix is system-side: `sudo mv
  /lib/firmware/updates/amdgpu /lib/firmware/updates/amdgpu.bak &&
  sudo reboot`. Documented operationally; no code commit.
