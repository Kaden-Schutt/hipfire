---
title: Device-mesh PIVOT — the ONE executor is execute_steps(mesh, gpus), not run_layer_program (master merge collision)
date: 2026-07-06
tags: [device-mesh, parallel-expansion, execute_steps, dense_forward, run_layer_program, superop, forward_bindings, tp, pp, ep, pivot, phase2]
---

**Branch:** `feature/device-mesh` (worktree `.claude/worktrees/feature+device-mesh`), phase-2 branch off
`feature/parallel-expansion` (which carries `HIPFIRE_EMULATE_GPUS`). Plan doc:
`docs/superpowers/plans/2026-07-05-device-mesh-transparent-parallelism.md` — see the `## PIVOT` section
(authoritative; the original §1 + phase table are marked SUPERSEDED inline).

## What broke
The master merge (`5b95cbd3`, 519 commits) landed master's NEW dense spine and reverted our executor wiring:
- Master shipped `dense_forward` (`crates/hipfire-runtime/src/arch_spec.rs:131`, commit `2a41f98f`): builds a
  transient `Vec<Step>` per layer → feeds `execute_steps(gpu: &mut Gpu, ctx, steps)`
  (`crates/hipfire-dispatch/src/pipeline/steps.rs:600`). **llama** (`llama.rs:3460`) + **qwen2**
  (`qwen2.rs:1927`) now route through it. The merge REVERTED `3a3c60e5` (qwen2 → `run_layer_program_mesh`).
- **Two independent decode spines, no shared code** (parallel op vocabularies `Step` vs `SuperOp`):
  - Spine A (dense): `Step` IR, `dense_forward`→`execute_steps(&mut Gpu)`, llama+qwen2 (+qwen35/cohere2moe
    call execute_steps directly). **63 call sites.**
  - Spine B (superop): `SuperOp` IR, `run_layer_program[_mesh]` via `ForwardBindings`, deepseek4/minimax/lfm2moe.
    **7 call sites.**
- The device-mesh executor (`run_layer_program_mesh`, `superop.rs:457`) was built on Spine B. Master made
  Spine A the dense default. `execute_steps` is the true chokepoint (even Spine B prefill + run_layer_program
  bottom out in it). **The mesh belongs at execute_steps.**

## New thesis (decisions LOCKED by bjoern 2026-07-06)
ONE executor: **`execute_steps(mesh: &DeviceMesh, gpus: &mut Gpus, ctx, steps)`** — replace `gpu: &mut Gpu`
with `(mesh, gpus)`, fan out transparently inside. `Step` IR = single lowering for ALL arches + ALL axes.
Retire `SuperOp`/`ForwardBindings`/`run_layer_program*` for parallelism; migrate their one novel piece
(EP-MoE all-reduce) into a `Step`.
- **Grand-unify** — MoE/EP folds into execute_steps too. The "live" EP path (run_layer_program_mesh serving
  ds4/minimax multi-GPU) is OUR phase-2 feature work, not a production contract → free to retire.
- **Big-bang** — flip `execute_steps(&mut Gpu → mesh, gpus)` across all 63 sites at once (callers pass
  `DeviceMesh::single()`), then add sharding. Same for drivers holding `&mut Gpu` (`dense_forward`, qwen35
  `forward_from_x_gpu`, cohere2moe `decode_step_body`, `prefill_forward`).

## Three axes → three homes (conflating them was the original plan's error)
- **PP (inter-layer)** = driver level, ABOVE execute_steps. Already = `forward_scratch_band(layer_range)` +
  `Gpus::boundary_copy` (`llama.rs:4205`, bit-exact max|Δ|=0). Generalize into dense_forward; `mesh.stage_for_layer`.
- **TP (intra-op)** = INSIDE execute_steps. Per-`Step` shard by manifest `ShardPolicy` (Column/Row/HeadSharded/
  FusedQKV) + all-reduce over `Tp` group. Manifest IS the shard rule (single source of truth, already built).
- **EP (intra-MoE)** = `Step::Moe` (absorb `run_moe_ep`/`ep_add_into_residual` + `Ep` all-reduce). +
  `Step::Recurrent`/`Step::Conv` for qwen35 DeltaNet to join the one spine.

## Keep / rework / orphan
- KEEP (spine-agnostic, validated): hipfire-hardware (DeviceMesh/Gpus/collectives/boundary_copy),
  weight_manifest/plan_manifest, fulfill_manifest/WeightStore, tp_shard, forward_scratch_band.
- REWORK → Step::Moe: `run_moe_ep`/`ep_add_into_residual` + EP all-reduce (`superop.rs:352-533`).
- ORPHAN: `run_layer_program_mesh` top-level (1×1 arm already dead post-revert), `ep.rs` shim,
  ForwardBindings/LayerProgram/SuperOp for parallelism (SuperOp still underpins the separate
  HIPFIRE_FORWARD_LOWERED experiment — don't delete outright, just stop routing parallelism through it).

## 1×1 identity is free + byte-identical (safety anchor)
`execute_steps(DeviceMesh::single(), gpus)` → dispatch to `gpus.devices[0]`. `Gpus::single` (`lib.rs:195`)
moves Gpu in verbatim (active_stream:None); MUST NOT call `ensure_rank_streams` (`superop.rs:437`, the
None→Some flip that switches hot-path memset sync→async — CLAUDE.md trap); `group_along(kind,[])`=singleton,
`all_reduce_sum_f32*` short-circuit at len==1. Proven on qwen2 (md5 LOWERED=0==1, coherence 11/11 @3a3c60e5).

## Re-sequenced phases: P-A big-bang signature flip (byte-identical) → P-B TP-in-execute_steps → P-C PP-at-driver
→ P-D Step::Moe/EP fold (retire run_layer_program_mesh) → P-E Step::Recurrent/Conv + DeltaNet head-shard, then
5a/5b heterogeneity/ragged unchanged. ModelParallel/ArchDispatch daemon rehome carries forward (sits above executor).

## Lineage note
`feature/device-mesh` ⊇ `feature/parallel-expansion` (strict ancestor; parallel-expansion = HIPFIRE_EMULATE_GPUS
+ TP-default + PP plumbing, the enabling harness). The qwen35 per-device givens prefill fix also split out onto
current master as standalone branch `fix/qwen35-multigpu-prefill-givens` (worktree hipfire-fix-qwen35-givens).
