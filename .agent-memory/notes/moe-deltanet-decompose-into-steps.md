---
title: DECOMPOSE MoE/DeltaNet into separate Steps (split what from how/where) is the TARGET; P-D's ep_moe_allreduce primitive = DEFERRED decomposition, not a federation
date: 2026-07-07
tags: [device-mesh, moe, deltanet, step-ir, decomposition, transparency, what-how-where, p-d, p-e, ep, attention, execute_steps, grouped-gemm, correction, retraction]
---

Supersedes the earlier "Step::Moe is dead → federation" note (deleted). Direction AGREED with bjoern 2026-07-07.
Parent tracker: [[device-mesh-pivot-execute-steps-spine]] (`## COURSE CORRECTION 2026-07-07`).

## The target (agreed): compose ops out of SEPARATE Steps; split WHAT from HOW/WHERE

The `Step` IR's purpose is to separate **what** (the ordered ops + their logical operands) from
**how/where** (device, rank, shard, collective). MoE, DeltaNet, and attention are each meant to be
**composed out of several fine-grained steps**, NOT collapsed into one monolithic op. The executor,
given the flat step stream + the mesh + the manifest (shard/placement policy), places and
collectivizes transparently — the arch never names a device, rank, or collective.

**Attention already realizes this** (`Step::Attend`, PB-TP4b): rmsnorm → QKV `Gemv` → `Rope` →
`Attend` → `Wo` `Gemv`+reduce, sharded head-parallel with the KV cache placed by the manifest. It
touches persistent state (KV cache) and STILL decomposes cleanly. That is the template for MoE + DeltaNet.

## The adjudication (why the monolithic reading was a strawman)

Run the candidates through the what/how/where test:
- A monolithic **`Step::Moe` that "absorbs run_moe_ep"** hides how/where INSIDE a what-op → fails the split.
- The primitive **`ep_moe_allreduce(…, moe_rank: FnMut(&mut Gpu, rank, …))`** has the arch closure run
  the experts (*what*) AND receive `rank` (*where*) → re-couples the two → fails the purpose in the arch
  instead of in a variant.
- **Decomposition** — `Route` → grouped expert GEMM → `SiluMul` → grouped-down + combine, with expert
  placement from the MANIFEST and the `Ep` all-reduce from the EXECUTOR → HONORS the split.

So under the north star's own principle the composition was ALWAYS the target; a single `Step::Moe`
would have violated the goal even if it "fit." The P-D finding "MoE can't be one Step" is TRUE but
answered the strawman — it never evaluated the composition.

## ERROR LOG (mine, retracted 2026-07-07)

- **Conflation:** ds4 `run_moe_ep` (forward.rs:2114) "ignores the OpBinding, dispatches via `self.state`"
  is evidence that ds4's forward is a **monolithic hand-block never decomposed** — NOT that MoE is
  intrinsically un-step-able. My P-D conclusion + the prior note + the prior MEMORY.md line generalized
  "ds4's current code" into "MoE's nature." RETRACTED.
- **"Federation is the correct end-state / ratify it":** overstated — it's the CURRENT state produced by a
  shortcut, not an inevitability.
- **"DeltaNet resists like MoE / is even harder":** WRONG DIRECTION (see below). RETRACTED.
- **Root of the misread:** the plan's wording — P-D = "Add `Step::Moe` (absorb `run_moe_ep`)" — invited the
  monolithic reading. Corrected inline in `docs/superpowers/plans/2026-07-05-device-mesh-transparent-parallelism.md`.

## MoE decomposition is a WRAP + REFACTOR job — the kernels already exist (grounded 2026-07-07)

Not greenfield. The whole MoE sub-pipeline is already kernelized; it just isn't wrapped as Steps:
- **Route:** `kernels/src/deepseek4_moe_topk_bias_aware[_batched].hip` (router topk + bias).
- **Grouped expert gate/up + down:** `gemm_hfq4g256_moe_grouped_mmq*`, `gemm_paro_q4g128_moe_grouped_*`,
  `gemm_mq2g256_lloyd_moe_grouped_*`, … (per arch/dtype).
- **Combine:** `moe_down_combine_grouped_k8.hip`.

Grouped-GEMM takes a runtime **offsets/segment tensor** as an operand; the Step IR already passes tensor
operands and `Step::Attend` already carries a `positions` index-tensor operand → precedent for
`Step::GroupedGemm{ w_experts, offsets, x, y }`. Step COUNT stays fixed (one grouped-GEMM over a fixed
expert count, not one step/token) → no data-dependent control flow.

`ExpertSharded` (packed-blob + pointer-table + zeroed-dummy) is the sole genuinely arch-specific policy —
but it lives in **placement (how/where) = the manifest**, so a decomposed MoE step stream + an
`ExpertSharded` manifest policy IS the intended split, not a counterexample.

## DeltaNet re-forecast (corrected) — EASIER to decompose than MoE, not harder

On the IR axis DeltaNet is easier: **no data-dependent routing** (every token updates the S-matrix; no
gather/topk). The recurrent scan is a **fixed-shape big-block op**, structurally like `Step::Attend`, so
`Step::Recurrent`/`Step::Conv` carrying a state-plan (the analog of Attend's `KvTierPlan`) fit, with the
S-matrix / conv state co-sharded by head via a manifest `StateEntry` — same as the KV cache. Per-head
independence makes the head-parallel partition exact.

DeltaNet's real difficulty is NOT IR expressibility — it's the pre-existing **correctness debt**, which
you pay regardless of Step-vs-primitive:
1. Dropped `s_ef_residual` (qwen35.rs:1091) — a real single-vs-multi divergence on the stochastic path; fix FIRST.
2. Q8 DeltaNet-state stochastic rounding → pin FP32 + `HIPFIRE_DETERMINISTIC=1` for any parity ([[kv-site-a-asym-latent-bug]] rule).
3. Greenfield `dn_value_head_range` / `HeadSharded` (never drove a forward).
4. `la_to_device` compact-vs-global index reconciliation (StateStore keys by global layer).

## Calibration (the real remaining cost) + decision

- **Cost:** a few new step VARIANTS (`Route`, `GroupedGemm`, `Recurrent`, `Conv`) — variants, not new IR
  capabilities — + a decomposition refactor across the arches (the work P-D declined) + DeltaNet's
  correctness debt (owed anyway).
- **Buys:** MoE + DeltaNet get PP/TP/EP *for free* like dense — the actual transparency north star. The
  primitive gives working EP but leaves the arch hand-coding placement → never transparent.
- **DECISION (bjoern, 2026-07-07): decomposition into single Steps is the correct way forward.** The
  `ep_moe_allreduce` primitive stays as an INTERIM (it can drive the arch body inside a decomposed
  executor during migration, then retire — additive, not a dead end). Separately, the daemon god-struct →
  `ModelParallel`/`ArchDispatch` collapse remains the one structural must (the #462 state-bleed surface),
  and it is INDEPENDENT of the IR decomposition.
