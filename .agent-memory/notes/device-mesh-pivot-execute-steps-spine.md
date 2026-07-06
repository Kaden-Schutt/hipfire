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
    **~4 direct call sites (9 incl. `_ep`/`_mesh`).**
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
  `forward_from_x_gpu`, cohere2moe `decode_step_body`, `prefill_forward` @ `crates/hipfire-runtime/src/llama.rs:1498`).
  NB: llama lives in `crates/hipfire-runtime/src/llama.rs`, NOT a `hipfire-arch-llama` crate (CLAUDE.md misleads).

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

## P-A DONE 2026-07-06 — mesh-only, NOT the "big-bang gpus flip" (bjoern chose it after evidence)
Implemented P-A as `execute_steps_mesh(mesh: &DeviceMesh, gpu: &mut Gpu, ctx, steps)` (steps.rs, delegates to
`execute_steps(gpu, ...)`; debug_asserts n_devices==1). NOT `&mut Gpus`. **Why the change from the plan's
"gpus: &mut Gpus":** the single-GPU serve path (daemon `Gpu::init` @737/768/1491 + `llama.rs:9588/9691`) owns a
BARE `Gpu`, never a `Gpus`, and its lifetime is decoupled from model load — so "every caller passes &mut Gpus"
secretly REQUIRES the daemon god-struct hoist that the handover itself defers to post-P-C. bjoern picked mesh-only
(threads the cheap mesh value, zero borrow rework, no hoist); the `gpu→gpus` promotion + daemon `Gpus` hoist move
to **P-B**, applied only to sharding paths. Style: each calling fn builds a local `DeviceMesh::single()` (or inlines
`&DeviceMesh::single()` in qwen35 — alloc-free empty Vec); `gpu` stays `&mut Gpu`; byte-identical by construction.

**Scope reality (measured, not the handover's "4 drivers"):** 48 real call sites; 41 THREADS across 19 fns + 7 OWNS.
**IN P-A (migrated, direct path):** cohere2moe (2), shared dense = arch_spec `dense_forward`(5) + arch-llama
`forward_scratch_layers`(5) + qwen2 `forward_step_after_x`(+_lowered)(6) + runtime `forward_scratch_layers_lowered`(1),
qwen35 (25 incl multi-GPU OWNS). **OUT (ForwardBindings lowered path, → P-D):** minimax `minimax_attn_block`, lfm2moe
`attn_mixer_block`, deepseek4 (no direct execute_steps), qwen35 `run_residual_gemv`(13558, kept bare `execute_steps`
in import). Commits: U0 shim `abb74fa9`, U2 cohere2moe, U3 dense, U5 qwen35 `8c024452`. Added `hipfire-hardware` dep to
cohere2moe/arch-llama/arch-qwen2/qwen35.

**Validated gfx1151, byte-identical:** cohere2moe gate 4/4 OK; coherence_probe qwen3-0.6b-llama + qwen25-0.5b OK
(0 hard/0 soft); coherence-gate.sh 11/11 OK (qwen35 0.8b/4b/9b/27b × mq4/mq3/mq3-lloyd/mq6). Multi-GPU OWNS sites
(forward_ep/_multi) are byte-identical drop-ins (execute_steps_mesh forwards the exact `&mut gpus.devices[i]`), not
exercised by single-GPU gates. **U6 (rename execute_steps_mesh→execute_steps) DEFERRED to P-B** — cosmetic, and P-B
reworks these signatures anyway. Local plan detail: `docs/superpowers/plans/2026-07-06-P-A-execute-steps-mesh-flip.md`.

**P-B START HERE:** promote the sharding paths `&mut Gpu`→`&mut Gpus` + implement TP INSIDE `execute_steps_mesh`
(per-Step ShardPolicy shard + Tp all-reduce); this is where the daemon `Gpus::single` hoist finally lands (the
mesh-only P-A deliberately left it undone). `execute_steps_mesh` currently degenerates to the single gpu — flip its
debug_assert to real multi-device handling there.

## P-B IN PROGRESS 2026-07-06 — dense-TP weight slicing DONE (PB-1a/1c); executor + serve remain
Plan: `docs/superpowers/plans/2026-07-06-P-B-tensor-parallel.md`. Full TP-infra inventory done (recall it: the 5
seams are fulfill_manifest slicing, execute_steps_mesh body, resolve_mesh Tp axis, store→forward bridge, daemon serve;
EP path = the working template — `run_layer_program_mesh` EP arm `superop.rs:457`, qwen35 `forward_prefill_batch_ep`).
- **PB-1a ColumnShard** (`58eecd64`) + **PB-1c RowShard** (`62c4f267`) LANDED in `weight_store.rs::fulfill_into`,
  byte-oracle-validated on emulated Tp-2 (`fulfill_manifest_probe`). Column = contiguous output-row split
  (format-agnostic, `m%tp==0`); Row = strided per-row k-gather (`rb%tp==0`, group-alignment is validate_manifest's).
  Covers the REAL llama/qwen dense manifest (separate Column wq/wk/wv + Row o_proj/down + Replicate/Pin/Tied).
- **PB-1b (FusedQkv/HeadSharded/VocabShard) DEFERRED** — emitted by NO current manifest (would be speculative);
  stays a clean `Err`. Implement when a manifest needs it.
- **PB-4 CORE VALIDATED** (`e513dc4f` + `62c5ee11`) — `tp_gemv_parity` example proves the TP compute+collective
  numerically on emulated Tp-2 (gfx1151), composing PB-1a/1c slicing + `all_reduce_sum_f32_peer`:
  (1) column-parallel `concat(W_r·x)==W·x` (2.4e-7), (2) row-parallel `all_reduce(W_r·x_r)==W·x` (4.8e-7),
  (3) **composed FFN block** `W2·(W1·x)` col→row with sharded on-rank intermediate + ONE end-of-block all-reduce
  == whole (1.2e-7). Every TP PRIMITIVE + the real block dataflow is proven correct. **GOTCHA found:** `gemv_f32`
  returns WRONG results for a non-64-aligned reduction dim (INTER/TP=48 → 0.04 err; =64 → 1e-7) — a real TP split
  must keep sharded reduction dims kernel-aligned (that's validate_manifest's group-alignment job).
- **FUNCTIONAL TP FORWARD VALIDATED** (`4b13f9dd` + `c8489ece`) — `hipfire_runtime::tp_forward::tp_ffn_forward`
  (reusable LIB fn, not just a demo) runs an n-layer FFN-residual stack tensor-parallel over the mesh's Tp group:
  per-rank sharded weights from `WeightStore`, on-device rank loop (rmsnorm → Column gemv → silu → Row gemv), ONE
  `all_reduce_sum_f32_peer` per row-parallel op, cross-layer residual, hidden kept replicated (no inter-layer
  broadcast). Example `tp_forward_parity`: 4-layer TP == host F32 ref, max|Δ|=1.2e-7, ranks bit-identical. **This is
  the production-callable TP executor pattern `dense_forward` adopts** (the FFN half). Preconds: caller sets
  per-device `active_stream` + `enable_peer_all`. **REMAINING to daemon-served TP:** attention head-parallel layer
  (Column head-split QKV + Row o_proj all-reduce — same machinery, `nh%tp==0`), wiring `tp_ffn_forward`'s pattern
  into the REAL llama `forward_scratch_layers` with `&mut Gpus`, + daemon `load_model_tp`/serve (PB-2/3/5).
- **Earlier REMAINING (production INTEGRATION):** PB-2 resolve_mesh real Tp axis —
  **FORK: `tp` knob maps to `Ep` (config.rs:155) for MoE; disentangle EP-vs-TP intent at the daemon load path
  (`load_model_ep` vs new `load_model_tp`)** — recommended default, ripples into daemon. PB-3 store→forward bridge
  (assemble per-rank sharded `LlamaWeights` from `WeightStore`). PB-4 FULL: wire the rank-loop + all-reduce into the
  REAL `dense_forward`/`forward_scratch_layers` with `&mut Gpus` (mirror `forward_prefill_batch_ep`); attention is
  head-parallel — Column-sharding qkv `[nh·hd,d]` by equal rows == head-split when `nh%tp==0`, then attention on
  owned heads + Row o_proj all-reduce. PB-5 daemon `load_model_tp` + serve + real-model `tp_decode_parity` (FNV vs
  single-GPU, FP32+DETERMINISTIC). This is the multi-session capstone; the hard primitives are done + validated.
