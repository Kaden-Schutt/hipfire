---
title: Device-mesh PIVOT — the ONE executor is execute_steps(mesh, gpus), not run_layer_program (master merge collision)
date: 2026-07-06
tags: [device-mesh, parallel-expansion, execute_steps, dense_forward, run_layer_program, superop, forward_bindings, tp, pp, ep, pivot, phase2]
---

> **Historical document.** This file preserves dated implementation and validation evidence. Current status and remaining work are tracked only in [device-mesh-refactor-tracker.md](../../.agent-progress/device-mesh-refactor-tracker.md).

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
  per-device `active_stream` + `enable_peer_all`.
- **PB-3 ATTENTION HEAD-PARALLEL VALIDATED** (`d5e63c8b`) — `tp_attn_parity` example proves a WHOLE TP transformer
  layer (attn+FFN) == single-device on emulated Tp-2 (gfx1151), max|Δ|=1.19e-7 (same level as FFN proof). New
  mechanism = the attention block: Column head-split QKV (`ShardPolicy::ColumnShard{axis:0}` on wq/wk/wv → rank owns
  q_head_range/kv_head_range), per-rank RoPE + `kv_cache_write` + `attention_f32` on owned heads, Row `wo`
  (`RowShard{axis:1}`) → partial → `all_reduce_sum_f32_peer` → attention residual, then the proven FFN block. KEY
  correctness fact: head-parallel is EXACT (not approximate) — RoPE is per-head, and clean GQA split keeps each rank's
  Q heads mapped entirely onto its OWN kv heads (n_heads/n_kv_heads ratio preserved per rank), so a rank's local
  `[max_seq, kv_dim/tp]` `attention_f32` == the head-slice of the full-cache attention (verified against attention.hip:
  `t*n_kv_heads*head_dim + kv_h*head_dim`, kv_h=h/(nh/nkv)). Cache layout confirmed `[max_seq, kv_dim]` position-major.
  Reference computed with the SAME GPU kernels (not host) so RoPE/softmax/GQA conventions match by construction. The
  single-device reference SKIP-guards on `init_uniform(TP,TP)` (n_layers>=n_devices). **REMAINING to daemon-served TP:**
  wire this proven pattern into the REAL llama `forward_scratch_layers` with `&mut Gpus` (PB-4-full), + daemon
  `load_model_tp`/serve (PB-2/5). llama layer op map (from Explore): separate wq/wk/wv (`LayerWeights` @ llama.rs:688,
  fusion is a kernel choice not storage), `weight_gemv_prerotated` for MQ quant (rotate_x_for_mq once + 3 prerotated
  GEMVs), `rope_f32`, `kv_cache_write` (7-tier quant ladder via `llama_kv_write_attend` @ llama.rs:3185), `attention_f32`,
  `weight_gemv_residual` (fused wo·attn + x). NO `forward_scratch_tp`/`load_weights_tp` exist yet (planned names only).
- **Earlier REMAINING (production INTEGRATION):** PB-2 resolve_mesh real Tp axis —
  **FORK: `tp` knob maps to `Ep` (config.rs:155) for MoE; disentangle EP-vs-TP intent at the daemon load path
  (`load_model_ep` vs new `load_model_tp`)** — recommended default, ripples into daemon. PB-3 store→forward bridge
  (assemble per-rank sharded `LlamaWeights` from `WeightStore`). PB-4 FULL: wire the rank-loop + all-reduce into the
  REAL `dense_forward`/`forward_scratch_layers` with `&mut Gpus` (mirror `forward_prefill_batch_ep`); attention is
  head-parallel — Column-sharding qkv `[nh·hd,d]` by equal rows == head-split when `nh%tp==0`, then attention on
  owned heads + Row o_proj all-reduce. PB-5 daemon `load_model_tp` + serve + real-model `tp_decode_parity` (FNV vs
  single-GPU, FP32+DETERMINISTIC). This is the multi-session capstone; the hard primitives are done + validated.

## DIRECTION LOCKED 2026-07-06 — bjoern chose GRAND-UNIFY (TP inside execute_steps), not a parallel forward
Asked bjoern how PB-4-full should reach real-model TP logit parity; offered (a) standalone F32 tp_llama_forward
parity, (b) mirror EP serve path into load_model_tp now, (c) TP inside execute_steps (Step IR) now. **He picked (c)**
— the pivot's endgame: the ONE dense executor (`execute_steps`) becomes TP; no parallel forward, no SuperOp spine.
Sub-plan (local, gitignored): `docs/superpowers/plans/2026-07-06-P-B-tp-in-execute-steps.md`. Renamed remaining P-B
work to **PB-TP1..PB-TP5**.

**Key IR facts (steps.rs):** `Step<'a>` carries whole-model borrows (`Gemv{w:&WeightRef, input, out:&GpuTensor}`,
`GemvResidual`, `RmsnormAutomatic`, `Attend`, `Rope`, `QkNorm`, `BiasAdd`) → a single `&[Step]` can't be sharded in
place; each rank needs its OWN sharded weight+buffers → the TP executor takes **per-rank Step lists** (lock-step).
`WeightRef{buf,dtype,m,k,row_stride,rotation,awq_scale}` is a plain borrow struct (build F32 directly, row_stride=0).
GEMV family `run_auto` dispatches F32 via `RotationPlan::None`→`gemv_f32`. **The Step IR has NO activation (silu) op**
— silu is fused into gate-up kernels, so a full FFN needs a new step or the fused path (later increment). hipfire-dispatch
already depends on hipfire-hardware and uses `Gpus`+`all_reduce_sum_f32_peer` (ep.rs/superop.rs) — the EP
`run_layer_program_mesh` (superop.rs:457) is the exact precedent (per-rank bindings + collective on `SuperOpKind::Moe`).

- **PB-TP1 DONE + validated** (`27002c55`) — `execute_steps_tp(mesh, gpus: &mut Gpus, per_rank_steps: &[Vec<Step>],
  collectives: &[TpCollective])` in steps.rs (re-exported from `pipeline`). Runs each Step on every rank of
  `group_along(Tp)` (bind_thread + `launch_op`, no fusion), then for `TpCollective::AllReduceOut{dim}` syncs each
  rank's stream + `all_reduce_sum_f32_peer` over the row-parallel step's `out` bufs (extracted via `tp_step_out_buf`).
  Column Gemv → sharded output feeds next step; Row Gemv → partial summed in place. Residual add must be a SEPARATE
  post-collective step (row-parallel GemvResidual would sum residual tp×). Example `tp_execute_steps_parity`: column→row
  GEMV pair routed THROUGH the executor == single-device on emulated Tp-2 (gfx1151), max|Δ|=1.21e-8; dispatch lib 172/0.
  NEW `execute_steps_tp` entry, NOT a signature change to `execute_steps_mesh` (P-A kept `&mut Gpu` for 40+ sites);
  unify once proven. Additive/off-path (only the example calls it). Uses `_peer` all-reduce; per-rank `DispatchCtx`.
- **PB-TP2 DONE + validated** (`5a97ca9e`) — `tp_execute_steps_layer_parity`: rmsnorm → col W1 → row W2 through
  `execute_steps_tp` == single-device on Tp-2, max|Δ|=5.59e-8. Two additions: (1) a REPLICATED non-Gemv step
  (`Step::RmsnormAutomatic`, `RotationPlan::None` → plain `gpu.rmsnorm_f32` into `out`, x_plain unused) flows through
  the TP executor unchanged (launch_op already handles every Step variant, so no executor change needed — just build
  the per-rank Steps + `TpCollective::None`); (2) the `collectives` list is DERIVED from each step's weight
  `ShardPolicy` via `collective_for_policy` (weight_manifest.rs:30; `RowShard`→`AllReduce{Tp}`→`AllReduceOut{dim}`,
  else None) — single source of truth, no hand-authored reduce. **DECISION:** `Step::Attend` deferred to PB-TP4 (its
  `AttnParams`+`KvTierPlan` surface is heavy — synthesizing it by hand is fragile; the REAL `attend_plan` builds it in
  the forward, and PB-3 already validated the attention math numerically). Downstream Gemv after RmsnormAutomatic uses
  `GemvInput::Raw(out)` (F32→no-op alias; equivalent to Prerotated for RotationPlan::None).
- **PB-TP3 DONE + validated** (`5c1274c8`) — added `Step::SiluMul { gate, up, out }` → `gpu.silu_mul_f32`
  (`PipelineOp::SiluMul` already existed in types.rs; just added the enum variant + its arm in the TWO total Step
  matches `op_kind`+`launch_op` — no other crate matches Step exhaustively, confirmed by full dispatch+runtime build).
  Closes the FFN silu gap: a whole SwiGLU FFN block is now one per-rank step list — rmsnorm → col W_gate + col W_up →
  `SiluMul`(on-rank inter/tp slice, no cross-rank dep) → row W_down + all-reduce. `tp_execute_steps_ffn_parity`: full
  FFN through `execute_steps_tp` == single-device on Tp-2, max|Δ|=2.79e-9 (+ in-example host-math cross-check). dispatch
  lib 172/0. `SiluMul` carries no weight → `TpCollective::None`; `tp_step_out_buf` returns None for it (never row-parallel).
- **PB-TP4a DONE + validated** (`dfc2f850`) — added `Step::ResidualAdd { x, y, dim }` → `gpu.add_f32(x,y,x)`
  (`PipelineOp::ResidualAdd` already existed). WHY: the real `dense_forward` fuses o_proj/down into
  `Step::GemvResidual` (`out = W·x + residual`), but a ROW-PARALLEL GemvResidual would all-reduce `(partial+residual)`
  → residual summed `tp×`. Under TP a row-parallel projection lowers to `Gemv (partial) → AllReduceOut → ResidualAdd`
  (residual added once, AFTER the collective). `tp_execute_steps_residual_parity`: full FFN block WITH residual
  (rmsnorm→col gate/up→SiluMul→row down+all-reduce→ResidualAdd) through `execute_steps_tp` == single-device on Tp-2,
  max|Δ|=2.98e-8. dispatch lib 172/0. **Executor op coverage for a dense layer is now COMPLETE except `Step::Attend`**
  (Gemv col/row ✓, RmsnormAutomatic ✓, SiluMul ✓, ResidualAdd ✓, derived collectives ✓; BiasAdd/QkNorm/Rope are
  replicated/on-owned-heads → launch_op handles them, no collective, and PB-3 validated the attention math on raw ops).
- **PB-TP4c PREREQ DONE + validated** (`827fac8f`) — `examples/llama_logit_dump.rs` drives the runtime
  `llama::forward_scratch` STANDALONE single-GPU (loads a llama-family HFQ via `load_weights_hfq`, prefill +
  greedy decode + per-step logit FNV). Validated gfx1151: `qwen3-0.6b-llama.mq4` (arch_id 1) → coherent
  ("Also, explain why the dog is not a complete combustion."). **GATING FINDING:** raw-F32-dir load is NOT
  wired for llama on this branch — llama carrier rejects `ModelSource::Dir`, no llama `ParoSource` (only qwen35
  has one), and `Qwen3-0.6B-PARO` is 4-bit paroquant anyway. Every small qwen3-0.6b on disk is 4-bit → NO native
  F32 checkpoint. **FP32 parity route = dequant HFQ→F32** (`weight_backend::dequant_f32` @575 exists): build F32
  `LlamaWeights` (reference) + F32 sharded per-rank buffers (TP), both via `dequant_f32`, so parity isolates
  sharding+collective (the note's chosen FP32 path; quant-GEMV-under-TP stays a later increment). NB `dense_forward`
  (arch_spec.rs:132) is the `dense_forward_tp` template: RmsnormAutomatic→3×Gemv(Prerotated)→[bias]→[qknorm]→Rope→
  `Step::Attend{plan,io}`(attend_plan Some)→o_proj `GemvResidual`; then FFN rmsnorm→gate/up→`silu_mul_f32`→down
  `GemvResidual`. Under TP the two `GemvResidual` (wo, w_down) split to Gemv→AllReduceOut→ResidualAdd (PB-TP4a).
- **DECISION 2026-07-06 — bjoern chose NATIVE-QUANT for the TP4c parity, NOT an F32-dequant detour.** The parity
  runs on the real mq4 weights through the production quant GEMV path (deterministic for llama, no DeltaNet). This
  reshaped the F32 question: the single-GPU reference is the existing quant `forward_scratch` (goal-1 harness),
  and TP shards the native-quant WeightStore buffers.
- **PB-TP4-quant DONE + validated** (`89f94d5b`) — `examples/tp_execute_steps_quant_ffn_parity.rs`: layer-0 FFN of
  a REAL `qwen3-0.6b-llama.mq4` (gate/up [3072,1024] Column, down [1024,3072] Row; inter/tp=1536) through
  `execute_steps_tp` == single-device, max|Δ|=**7.45e-9** on emulated Tp-2 (gfx1151). **KEY RESULT: the TP executor
  handles a ROTATED quant format (MQ4G256→FwhtG256) under column/row sharding with NO executor change.** Why: `launch_op`
  dispatches each dtype's rotation via `run_auto` (FWHT applied internally), and FWHT-G256 is block-diagonal per
  256-element k-group → commutes with a group-aligned k-split; in correct TP dataflow the row-parallel op gets its own
  on-rank k-slice (from column gate/up+silu), so each rank FWHTs exactly its groups and partials sum to the whole.
  Reuses `fulfill_manifest` (quant-aware Column contiguous-byte / Row group-aligned strided-k gather) + the F32 harness
  pattern. **Constraint: every sharded k-dim must stay %256==0** (validate_manifest's group-alignment job). No lib change.
  This is the linchpin: TP4c's bridge assembles per-rank quant `WeightRef`s from the store and runs them through the
  UNCHANGED executor; the single-GPU reference is the quant `forward_scratch`.
- **PB-TP4b DONE + validated** (`567a85d8`) — `examples/tp_execute_steps_attn_parity.rs`: the whole head-parallel
  attention block through `execute_steps_tp` via a first-class `Step::Attend` == single-device, max|Δ|=**7.45e-8** on
  emulated Tp-2 (gfx1151). Closes the executor gap PB-3 left (PB-3 validated the attention MATH on RAW ops). Per-rank
  Step list: RmsnormAutomatic[Replicate] → Wq/Wk/Wv[ColumnShard, rank owns heads] → Rope[per-head] →
  `Step::Attend{KvTierPlan, AttnParams}`[owned heads + per-rank KV cache] → Wo[RowShard]→AllReduceOut{D} →
  `Step::ResidualAdd`. **KEY: `Step::Attend` carries a REAL `KvTierPlan`+`AttnParams`** (the shape llama's
  `attend_plan` @llama.rs:3575 builds). Used the F32/`Simple` tier (`KvTierInputs` all-quant-false → `AttnF32`,
  same kernel PB-3 used raw) so parity is clean F32 — hand-built `KvTierInputs` since a plain F32 cache isn't a
  quantised `KvCache` (NB: the KV tier system has NO F32 KvMode; lowest is Q8 → a real-model TP4c uses Q8 KV, common-mode
  vs a Q8 single-GPU reference). Each rank's KV cache sized to its OWN kv heads (clean GQA split preserves nh/nkv per
  rank), seeded with its column-slice of F32 history; `Step::Attend` writes the current token per rank. `AttnParams`
  built inline in the Step (not Clone; `KvTierPlan` IS Clone). Reference = identical per-op kernels (same `run_attention`)
  on whole heads. flash_partials sized `n_heads*ceil(max_seq/128)*(2+head_dim)`. No lib change.
- **EXECUTOR IS NOW FEATURE-COMPLETE for a dense layer** — every op validated through `execute_steps_tp` on Tp-2:
  Gemv col/row (PB-TP1), RmsnormAutomatic + manifest-derived collectives (PB-TP2), SiluMul/full FFN (PB-TP3),
  ResidualAdd (PB-TP4a), **native-quant MQ4G256 FFN (PB-TP4-quant, no executor change)**, **head-parallel Step::Attend
  (PB-TP4b)**. A whole attn+FFN layer = the mechanical concatenation of the two proven per-rank Step lists.
- **PB-TP4c DONE + validated (THE CAPSTONE)** — full-model tensor-parallel forward == single-GPU, in two increments:
  - **(A) `874d6452`** `examples/tp_execute_steps_quant_layer_parity.rs`: a WHOLE real layer-0 (attn+FFN, qk-norm,
    Q8 KV, MQ4G256) through the store→forward bridge + `execute_steps_tp` == single-device, max|Δ|=**1.79e-7**.
    Proves the real bridge (`fulfill_manifest` shards real quant weights via an HFQ raw-bytes source closure) + the
    full 16-op per-rank layer Step list (mirrors `dense_forward`; row wo/down split to Gemv→AllReduceOut→ResidualAdd).
    Reference runs the SAME Steps one-at-a-time via `execute_steps(&[s])` (a lone step never fuses) → op-for-op match.
  - **(B) `c33bb926`** `examples/tp_full_model_parity.rs`: the WHOLE 28-layer qwen3-0.6b-llama.mq4 runs Tp-2 (embed
    rank0+broadcast → 28 sharded layers via `execute_steps_tp` → final norm + lm_head rank0 → logits) vs production
    `llama::forward_scratch`: **argmax IDENTICAL (33450)**, logit max|Δ|=**4.2e-4** on max|logit|=19.2. The unfused TP
    forward reproduces the fused production forward. `fulfill_manifest` shards ALL layers; replicated norms uploaded
    per rank; residual `x` stays replicated (all-reduce + replicated ResidualAdd/Rmsnorm keep it synced) so embed +
    final need no sharding. Single position (pos 0); multi-key attn under TP = PB-TP4b, q/k/o/ffn sharding = incr A.
  - **STATUS: the store→forward bridge + `dense_forward_tp` are PROVEN end-to-end on a real model.** Both live as
    examples (validated concepts); promoting the layer-step builder + Tp dispatch into a lib `dense_forward_tp<A>` +
    the bridge into a reusable `assemble_sharded_layers(store)` is a clean follow-up (the borrow pattern is worked out
    — per-rank WeightRefs built inline from `resident_l(store,name,l,dev)`; `WeightRef` isn't Clone; `leak` for the
    `&WeightRef` the Step holds, or keep a per-rank Vec that outlives the Step list). No lib change landed this session.
- **EP↔TP DISENTANGLED (`80d18401`) — the PB-TP5 prerequisite fork, RESOLVED (bjoern approved).** `resolve_mesh` was
  hard-wiring `tp`→`Ep`; now `resolve_mesh(pp, tp, ep, emulate)` maps each degree to its OWN axis (pp→Pp, ep→Ep,
  tp→Tp), precedence pp>ep>tp, emulate still defaults to EP. `resolve_parallelism` returns (pp,tp,ep). Daemon: parse an
  explicit `ep` knob; route `ep>1`→`load_model_ep`, dense `tp>1`→NEW `load_model_tp` (hipfire-loader). **Back-compat: a
  legacy `tp>1` on an EP-capable MoE arch (9/10) still means EP** (daemon peeks `HfqFile::arch_id`), so `--tp N` = "shard
  across N GPUs; arch picks the axis" (MoE→EP, dense→TP). `load_model_tp` is a RESERVED stub returning a clear "PB-TP5
  not yet wired" error (dense TP forward is validated by the examples; only the SERVE loop is unbuilt). CLI forwards
  `HIPFIRE_EP`→params.ep. No behavior change for existing EP/single-GPU. config tests 4/4.
- **PB-TP5 SERVE LOOP DONE + validated (`c2ae0b8f`)** — `examples/tp_decode_parity.rs`: dense-TP prefill + greedy
  decode == single-GPU `forward_scratch`, **argmax-exact**. prompt "The capital of France is" + 24 steps on emulated
  Tp-2 (gfx1151, HIPFIRE_DETERMINISTIC=1): ref_fnv==tp_fnv (`0a73e4975b94d4b7`), first_div=None, identical text. This
  is the REAL serve algorithm (growing per-rank KV → multi-key attention head-parallel under TP, not the pos-0 case).
  Per token: embed(rank0)+broadcast → 28 sharded layers via execute_steps_tp (KV write at pos) → final norm+lm_head
  (rank0) → argmax → feed back. Mirrors `ep_decode_parity` (which is ALSO a standalone example, separate from EP's
  daemon `generate_ep`). **The dense-TP forward + serve algorithm is fully proven; `build_layer_steps` is the reusable
  per-layer TP body a real `generate_tp` drives.**
- **PB-TP5 DAEMON INTEGRATION DONE + validated (`6b71b132`)** — dense-TP now SERVES through the daemon. New
  `hipfire_runtime::tp_serve::TpModel` (reusable form of tp_decode_parity: `forward_token(tok,pos)` + `logits()`;
  disjoint-field borrows split `self.gpus` mut from `self.ranks/store`). `LoadedModel.tp: Option<TpModel>` (a field
  distinct from `ep`; only `skeleton()` needed `tp:None` — the 15 other ctors spread `..skeleton()`; unload drops it).
  `load_model_tp` real (host tokenizer/chat-template/rec-sampling → `TpModel::load`; eos in the generic
  `deepseek4_eos_tok` slot). Daemon `generate_tp` (ChatFrame render → per-token prefill → `sampler::sample_cpu` decode →
  stream token/done events; eos/terminator/stop/max_tokens), dispatched `if m.tp.is_some()` before ep/arch. **Validated
  live gfx1151 emulated Tp-2:** `load {tp:2}` + generate → coherent stream + done event, and the tp=2 token stream is
  **BYTE-IDENTICAL to a tp=1 single-GPU serve** of the same prompt. Investigation used 3 parallel Explore subagents
  (LoadedModel ctors, generate protocol, load-path fields). **Lean scope:** llama-family qk-norm (arch 0/1), MQ4G256,
  Q8 KV, stateless per request (pos 0), per-token prefill; no spec/PFlash/eviction/grammar/tools, no multi-turn KV reuse.
  **P-B (tensor-parallel, TP1→TP5) is COMPLETE end-to-end: forward primitives → real-model parity → serve loop → daemon.**
## P-C STARTED 2026-07-06 — PP at the `dense_forward` driver (plan committed)
- Sub-plan: `docs/superpowers/plans/2026-07-06-P-C-pp-at-driver.md`. **Thesis:** PP lives ABOVE `execute_steps`, at the
  driver — run each layer on its `Pp` stage, `boundary_copy` the residual between stages. Generalize the pattern INTO
  `dense_forward` (mirror how P-B pulled TP into `execute_steps`). **NO executor change** (each stage runs its band via
  the same single-device `execute_steps`; PP = device selection + boundary copy; PP is EXACT → oracle bar max|Δ|=**0**).
- **REFRAME→REVIEW→REVERT 2026-07-06.** bjoern challenged "why above execute_steps? doesn't the manifest make PP
  transparent in dispatch?" → I reframed the plan to move PP INTO the executor (a whole-model `run_layer_program`). A
  **4-agent review team** (architecture/feasibility/simplicity/correctness) UNANIMOUSLY rejected it; bjoern reverted to
  the driver-owned loop. **The killer facts:** (1) dense llama attention is IMPERATIVE (`forward_scratch_band`), NOT
  Step-lowered — there is no whole-model Step program to feed a `run_layer_program` (`Step::` appears once in llama.rs).
  (2) `execute_steps_tp` rejects `tp<=1` (steps.rs:715) + `execute_steps_mesh` debug_asserts n_devices==1 (steps.rs:657)
  → a pure-Pp mesh can call NEITHER as the inner op; "PP wraps TP" is unreachable till N×M. (3) `run_layer_program` is
  the RETIRED Spine-B/ForwardBindings symbol (superop.rs:417). (4) contradicts the locked "three homes" + reverses
  P-A/P-B "defer the hoist" + is the N1 bounce (rejected 3 rounds). (5) whole-model program → self-referential
  WeightRef/Step lifetime (`WeightRef` not Clone). **RESOLUTION:** transparency = manifest placement + `DenseArch` trait
  boundary (arch never names a device) → INDEPENDENT of loop location; the shared generic `dense_forward` driver is the
  correct home (the locked altitude). Executor-transparent PP → P-5b, gated on real N×M + multi-GPU HW.
- **State (recon + review):** `dense_forward` (arch_spec.rs:131) is the arch-generic shared driver (llama+qwen2 route
  through it), single-GPU today. PP hand-coded ONLY in qwen35 (`forward_scratch_layers_multi` qwen35.rs:14367,
  `load_qwen35_pp` arch 5/6). Primitives proven bit-exact: `forward_scratch_band`/`_head`/`_embed` (llama.rs:4209/4525/
  3142), `Gpus::boundary_copy`/`wait_boundary`/`device_for_layer` (**hipfire-hardware/src/lib.rs:357/423/334** — NOT
  multi_gpu.rs, that's a re-export; active_stream NOT required, sync host-stage path), `mesh.stage_for_layer(l,n)`==
  `Gpus.device_for_layer` by construction (both uniform_split_counts), per-band KV `new_gpu_q8_multi`(llama.rs:7270)+
  `alloc_kv_per_layer_multi`(8246) ALREADY EXISTS, `fulfill_manifest` PP placement (`llama_store_pp` max|Δ|=**0**).
  s_ef_residual divergence is DeltaNet-Q8-state-only (qwen35.rs:5648), N/A to dense llama → =0 REACHABLE.
- **Increments (reverted, imperative driver loop):** **PC-0 DONE (`cc12222f`)** — un-broke the example gate:
  `llama_store_pp`+`llama_store_load` (added `LlamaWeights.lm_head_aliases_embd: false`) + a SEPARATE pre-existing break
  `ocr_e2e` (`qwen2::forward_step*(&mut gpus)`→`&mut gpus.devices[0]`, ×3, leftover from the master merge reverting
  3a3c60e5). `cargo check --workspace --examples` (the no-gpu-ci gate) now GREEN → catches future rot. Re-ran the PP
  oracle on gfx1151: banded PP forward (stage0 0..14 → boundary_copy → stage1 14..28+head) logit-IDENTICAL to bespoke,
  **max|Δ|=0** — the anchor PC-1 generalizes into `dense_forward`. NEXT: PC-1. PC-1 make the SHARED
  `dense_forward` PP-aware via the imperative `forward_scratch_band` stage loop + `boundary_copy` (mirror qwen35
  14388-14396); `DenseArch` gains a per-stage weights+scratch view (the one real trait change; model on
  `Qwen35ScratchSet`); size-1 group inner op = single-device `execute_steps` (NEVER `_tp`/`_mesh`); =0 oracle vs
  single-device on real qwen3-0.6b-llama emulated Pp-2. PC-2 decode + MULTI-TOKEN prefill (band copy `n_rows*dim*4`, the
  real gap — oracle only proves 1-tok pos0), banded `_multi` KV. PC-3 daemon serve (`PpModel`/`load_model_pp`/
  `generate_pp`, mirror PB-TP5). **Constraints:** Q8/FP32 KV only (NO asym); `active_stream=None` regime (debug_assert);
  =0 is SAME-ARCH scoped (emulation aliases to dev0 → proves banding logic NOT transport/residency; real 2-GPU same-arch
  =0 gate is a separate HW exit; mixed-arch is coherence-only); assert banding single-source-of-truth.
- **PC-1/2/3 ALL DONE + validated (emulated Pp-2, gfx1151).** `hipfire_runtime::pp_serve::PpModel` (the PP analog of
  TpModel; NO executor change — bands `forward_scratch_band` + `Gpus::boundary_copy`, `active_stream=None`):
  - **PC-1 (`6a7ac9dd`)** `pp_full_model_parity`: PpModel banded forward == single-device `forward_scratch`, **max|Δ|=0**
    (exact — reuses the identical forward kernels; only the F32 residual byte-copy differs).
  - **PC-2 (`a3e59d40`)** `pp_decode_parity`: prefill+decode token stream == single-GPU (FNV `0a73e497…`, first_div=None,
    == the TP FNV). **BUG the multi-position test caught (masked by the pos-0-only oracle):** `forward_scratch_band` reads
    `scratch.pos_buf` for RoPE+attention but only `forward_scratch_embed` (stage0) sets it → downstream stages RoPE'd at
    a STALE pos (0) → fine at pos0, garbage at pos>0. Fix: `forward_token` memcpy's pos into EVERY downstream stage's
    pos_buf. (Exactly the multi-token gap the review flagged.)
  - **PC-3 (`44edf2d6`)** daemon serve: `LoadedModel.pp_dense` + `load_model_pp` (dense llama arch 0/1, pp>1) +
    **`generate_tp` REFACTORED → generic `generate_dense<M: DenseServed>`** (one serve loop, `DenseServed` trait impl'd by
    TpModel + PpModel — both axes share it). Live `load {pp:2}` == pp=1 single-GPU byte-identical; TP=2 serve via
    generate_dense = no regression. **P-C (pipeline-parallel) SERVES end-to-end.**
- **P-C follow-ups (deferred):** real-HW per-stage weight banding (VRAM win; emulation loads whole weights on the output
  stage) + a real 2-GPU same-arch =0 gate (emulation can't prove transport/residency); batched prefill (per-token now);
  executor-transparent PP / N×M compose → P-5b. Then P-D (Step::Moe/EP fold), P-E (DeltaNet head-shard).

## P-D STARTED 2026-07-06 — fold EP-MoE into Step::Moe (plan; additive-first)
- Plan: `docs/superpowers/plans/2026-07-06-P-D-ep-fold-step-moe.md`. The locked grand-unify (bjoern 2026-07-06): retire
  `run_layer_program_ep` for parallelism, migrate the EP MoE all-reduce into a `Step::Moe` on a per-rank executor
  `execute_steps_ep` (sibling of P-B's `execute_steps_tp`). **Recon (subagent):** EP is LIVE (ds4/minimax multi-GPU serve
  via `run_layer_program_ep` ep.rs:73, default-on HIPFIRE_FORWARD_LOWERED). MoE forward is IMPERATIVE (`run_moe_ep`/
  `ep_add_into_residual` callbacks, ds4 forward.rs:2114 / minimax 825) → `Step::Moe` is a big-block variant like
  `Step::Attend` (design fields FROM ds4 run_moe_ep, don't invent). Per-rank sharded borrows = P-B's per-rank Step lists.
  **Verdict: legitimate (locked, proven P-B template), NOT N1 — BUT a 2-3 day rewrite of LIVE multi-GPU EP serve for
  maintainability (no new functionality), validatable only on emulated EP-2 (no 2-GPU HW).**
- **Increments (additive-first, live EP stays default until proven byte-identical):** P-D-0 `Step::Moe` variant +
  `execute_steps_ep` skeleton (zero-risk, not on any live path; build+dispatch-tests+dense-coherence green). P-D-1 ds4
  `forward_ep` builds per-rank `Step::Moe`, route via `execute_steps_ep` GATED behind `HIPFIRE_EP_STEP=1` (default OFF) →
  validate `ep_decode_parity` byte-identical vs `run_layer_program_ep`. P-D-2 minimax. P-D-3 flip default + delete
  `run_layer_program_ep` + the 2 ForwardBindings EP methods (keep the trait for the lowered single-GPU path).
- **Honest flag raised to bjoern:** P-C's follow-ups (batched prefill, real-HW banding) are lower-risk/higher-observable-
  value than a live-EP rewrite; offered the redirect. Awaiting "proceed all P-D items" vs redirect before the P-D-1 flip.

## P-D REFRAMED 2026-07-06 — SuperOp was a mistake: REMOVE it + redo EP clean (bjoern's call). Green baseline PASSED.
- **The Step::Moe fold is DEAD.** Reading ds4 `run_moe_ep` (forward.rs:2114): it IGNORES the OpBinding, dispatches via
  bespoke `self.state` (`ds4_moe_block_core` + deferred `hc_ffn_mix`); ds4 has NO fine-grained Steps (`run_proj`→"no Proj
  super-op"). EP arches are bespoke-callback-shaped (MLA + arch MoE), so a typed `Step::Moe` doesn't fit — it'd need a
  boxed closure (breaks POD Step) or a huge cross-arch plan. **bjoern: "SuperOp was a mistake from this branch, revert if
  needed, find a clean design" → "redo clean in the same pass."**
- **SuperOp topology:** #397 Ship 6 (9545e4cc/393d2a2b/eedf7ebe/55a539ce/fd879524) — IN origin/master (not branch-unique;
  the 519-commit merge pulled it in). It's the OLD parallelism substrate this pivot replaces with Step executors. Reverting
  it on device-mesh diverges from master + removes the shipped ds4/minimax EP multi-GPU serve → redo EP clean (no regression).
- **Footprint (full recon, subagent):** DELETE `superop.rs`(616L) + both `ep.rs` + per-arch `*_lowered`/`ForwardBindings`
  impl/`lower_program`/orphan-block-helpers/toggles across **4 arches** (ds4/minimax/lfm2moe/qwen35; qwen2 already OFF
  superop→dense_forward) + steps.rs `match_fused_prefix`/`step_op_kind` + the WHOLE daemon/loader EP serve (daemon: 8 fns
  generate_ep/ep_serve_ds4/_minimax/ep_emit_*+routing 6741-6774; loader: `ep` field/`EpState`/`EpArch`/`load_model_ep`(+_ds4/
  _minimax)/2 staging/unload branch/`ep-fault-inject` feature) + 3 EP example files (`ep_decode_parity`, `ep_deepseek4`,
  `ep_minimax`). **KEEP:** Step IR (`execute_steps`/`_mesh`/`_tp`, `match_prefix`/`FUSED_TABLE`/`op_kind`), `dense_forward`,
  TpModel/PpModel (all superop-INDEPENDENT — comments only), `all_reduce_sum_f32[_peer]` (shared w/ TP), single-GPU serve
  for ALL arches. **git-revert WON'T work** (arch adoption + EP wiring in separate commits + superop rewritten 4× + master
  merge) → manual surgical delete-forward, iterate on `cargo build --all-targets`.
- **GREEN BASELINE DONE (not committed — validation):** qwen3.5-4b `HIPFIRE_FORWARD_LOWERED=0` (hand path) == default
  (lowered), byte-identical + coherent. Confirms the single-GPU deletion is PURE (hand-written is a coherent sole-path).
- **SEQUENCE (green→change→build+test→commit; no broken EP window):** [1 DONE] green baseline. [2 NEXT — the big build]
  clean EP: a reusable `ep_moe_allreduce`(zero partials→run owned experts→all_reduce→add, extract from
  run_layer_program_mesh:479-525) + an `EpModel` (loader, analog TpModel/PpModel) driving the hand-written forward per-rank
  calling it at MoE layers — **minimax first, validate emulated EP-2 byte-identical to `run_layer_program_ep` WHILE IT STILL
  EXISTS**; ds4 mirrors (live check deferred to hipx — ds4 only fits hipx, NOT this gfx1151 box). [3] delete SuperOp + route
  EP through EpModel; make hand-written sole. [4] validate coherence gates (qwen35 dense+A3B, qwen2, minimax) + EP gate.
- **STATUS: multi-session pass. Step 1 (green baseline) validated; steps 2-3 (clean EpModel rebuild for ds4+minimax + the
  ~2000L deletion) are the large remaining work — handed off with the full footprint above.** Local plan (obsolete fold
  version, ignore): `docs/superpowers/plans/2026-07-06-P-D-ep-fold-step-moe.md`.
- **P-D STEP 2 DONE 2026-07-06 (commits 8783d35c ds4, 3cbbfc29 minimax).** DEVIATED from the note's "EpModel + delete the
  EP serve" plan: instead of a new loader `EpModel`, I made the arch `forward_ep` **clean in place** and KEPT the daemon/
  loader EP serve (generate_ep/ep_serve_ds4/_minimax + EpState/EpArch/load_model_ep) — it's EP-serve FEATURE wiring, NOT
  SuperOp, and "redo EP clean, no regression" wants it serving. New primitive: **`Gpus::ep_moe_allreduce(group, partials,
  residual_dim, moe_rank: FnMut(&mut Gpu, rank, &partial, EpMoePhase))`** + `pub enum EpMoePhase{RunOwned,AddResidual}` in
  `hipfire-hardware/src/lib.rs` (after `all_reduce_sum_f32_peer`). Owns zero→run-owned→all_reduce[_peer]→fold; the ONE
  phase-tagged closure sidesteps the two-closures-both-borrow-`&mut state` problem. Reads `HIPFIRE_EP_PEER_ALLREDUCE_DECODE`
  internally. ds4/minimax `forward_ep` now loop: per-rank `ds4_attn_block`/`minimax_attn_block` (replicated) then
  `ep_moe_allreduce` with a `{ds4,minimax}_ep_moe_rank` adapter (reuses the EXACT `ds4_moe_block_core`+`hc_ffn_mix` /
  `minimax_moe_block`+`add_inplace_f32(&state.h,·)` the retired trait methods called → byte-identical). Dropped per-forward
  `DeviceMesh::rect` + `run_layer_program_ep`. **VALIDATED gfx1151 emulated EP-2, peer all-reduce, DETERMINISTIC=1, byte-
  identical to pre-change baseline:** ds4 (deepseek-v4-flash.mq2lloyd 82GB) forward_ep FNV `0x0c04faf471f9c016` + MTP-EP
  draft "."==true next-next "." ACCEPT; minimax (MiniMax-M2.7.mq2 79GB) forward_ep FNV `0x31ede7c1d1cf140e`. Workspace
  `--all-targets --locked` clean. **GOTCHA: EP all-reduce needs librccl.so which is NOT installed on this box → MUST run
  EP with `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1` (peer boundary_copy path, RCCL-free); bare RCCL path panics at
  `RcclComms::init_all`.** **STEP 3 NEXT (the ~2000L deletion):** delete `superop.rs`+both `ep.rs`, per-arch lowered paths
  (ds4/minimax/lfm2moe/qwen35 Bindings/`*_lower_program`/`decode_step_body_lowered`/toggles), qwen35 `forward_ep` (EXAMPLE-
  ONLY EP — daemon EP dispatch is arch10→minimax else→ds4, NO ep_serve_qwen35) + `ep_decode_parity`, steps.rs
  `match_fused_prefix`/`step_op_kind`, pipeline/mod.rs superop decl, incidental import trims (loader/llama/pp_serve/
  arch_spec/mesh). KEEP daemon EP serve + clean forward_ep + ep_deepseek4/ep_minimax examples + Step IR.
- **P-D STEP 3 DONE 2026-07-07 — SuperOp SUBSTRATE FULLY DELETED (net −2958 LOC, 15 files).** Recon map via subagent
  (Explore); executed surgically (grep-verified boundaries, agent line numbers were approximate). **Relocated first (the
  lynchpin):** `ensure_rank_streams` (was `superop.rs`, needed by the KEPT loader EP serve + ep examples) → `Gpus::
  ensure_rank_streams()` method in hipfire-hardware; repointed loader×2 + both examples. **Deleted whole files:**
  `hipfire-dispatch/src/pipeline/superop.rs` (616L), `hipfire-dispatch/src/ep.rs` (69L shim), `hipfire-runtime/src/ep.rs`
  (5L re-export), `hipfire-runtime/examples/ep_decode_parity.rs` (450L, was the ONLY qwen35 EP caller). **Per-arch lowered
  deletion** (guard in decode_step_body + `*Bindings`+`impl ForwardBindings` + `*_superop`/`*_lower_program`/
  `decode_step_body_lowered`/`*_forward_lowered_enabled` + `ship6_lower_tests` + superop import trims): lfm2moe (−299),
  ds4 (−238, KEEP forward_ep/mtp_forward_ep/ds4_ep_moe_rank/ds4_attn_block/ds4_moe_block_core), minimax (−203, KEEP
  forward_ep/minimax_ep_moe_rank), qwen35 (−1094: op-code substrate + `forward_ep`+`forward_prefill_batch_ep`
  example-only EP + 4 `lowered_*` shape tests; KEPT the `*_via_execute_steps` helpers — those are Step IR, NOT superop;
  KEPT `set_ep_expert_shard`/`current_ep_expert_shard` EP-load helpers). qwen2 was already superop-free (dense_forward).
  Removed 3 mod decls (`pub mod superop;` pipeline/mod.rs, `pub mod ep;` dispatch+runtime lib.rs). **KEY: hand-written
  `decode_step_body` is now the SOLE single-GPU path (lowered was default-on; green baseline proved hand==lowered
  byte-identical so deletion is pure). Daemon/loader EP serve UNCHANGED (routes through the clean forward_ep).**
  **VALIDATED:** workspace `--all-targets --locked` clean (23.7s); ALL lib tests pass (0 failed); ds4 EP-2 FNV
  `0x0c04faf471f9c016`+MTP ACCEPT & minimax EP-2 FNV `0x31ede7c1d1cf140e` STILL byte-identical post-deletion; qwen35
  single-GPU `coherence-gate.sh` PASSED (no hard errors, all 8 rows 9b/27b/4b × mq4/mq3/mq3-lloyd/mq6 fluent on the
  hand-written sole path). pflash perf-stage FAILS but that's the KNOWN environmental gfx1100-baseline-on-gfx1151 mismatch
  (uniform ~2x drift across ALL 12 rows = wrong-GPU baseline; lowered→hand is perf-neutral/faster since it DROPS per-op
  dispatch overhead — NOT a regression). **No fmt: the remaining rustfmt hunks in touched files are PRE-EXISTING debt in
  untouched regions (my edits are fmt-clean); rustfmt would reformat them → violates surgical rule.** **P-D COMPLETE @
  commit 38def37a** (SuperOp gone, EP clean, single-GPU hand-sole). Commits: 8783d35c(ds4 EP)/3cbbfc29(minimax EP)/
  38def37a(deletion). Next: P-E (Step::Recurrent/Conv + DeltaNet head-shard) per the phase plan, OR the deferred TP/PP
  polish (PB-TP5 daemon `generate_tp`, batched prefill, real-HW multi-GPU checks).
- **Future TP polish — STATUS 2026-07-07 (P-C + P-D done; batched prefill done):**
  - ✅ **batched prefill** — DONE 2026-07-07, `feature/device-mesh` `10184aa0..fc5d846e` (6 commits, SDD), note
    [[tp-pp-batched-prefill]]. `DenseServed::prefill` seam (default per-token); PP banded `prefill_forward_band` (byte-exact
    parity); TP new `Step::Gemm` through `execute_steps_tp` (argmax parity, Q8-KV-vs-F32 Δ=0.85). Single-batch ≤256 +
    per-token fallback. Caught+fixed a latent `rotation.rs` batched-rotate bug (single-row on `n×k`).
  - ⏭ **NEXT deferred items (unblocked on this gfx1151 box), recommended order:**
    - ✅ **A** — DONE 2026-07-07 `feature/device-mesh` `91e99955`. `tp_prefill_parity` now asserts a numeric bound, not
      just greedy argmax. Swapped the reference `llama::prefill_forward` (F32 in-batch attn) → `forward_prefill_batch`
      (the Q8-KV `attention_flash_q8_0_batched_masked` path the TP side uses), so both read the same Q8 KV.
      **Triangulated (emulated Tp-2, gfx1151, DETERMINISTIC=1):** F32-in-batch-vs-Q8-flash single-GPU max|Δ|≈**0.88**
      (⇒ the old doc-comment's "a bit above 4.2e-4" was fiction — never asserted); Q8-flash single-GPU-vs-TP
      max|Δ|≈**0.20** (4× tighter). **TRAP: the note's "~1e-3" target was the *decode* number (`tp_full_model_parity`
      4.2e-4 = single pos, one-entry KV); prefill compounds Q8-KV rounding over 103 pos × 28 layers → 0.20 is the real
      floor, not a bug** (argmax matches " Also"; TP delta < attention-mode delta). Bound = `4.0e-1` (2× observed, churn
      headroom); a real sharding/all-reduce break lands near/above 0.88 and trips it.
    - ✅ **B** — DONE 2026-07-07 `feature/device-mesh` (`7b50839a` + fix `25f9046f`, SDD; primitive test `c573326f`). Dense
      TP/PP multi-turn KV reuse. **KEY FINDING: the single-GPU path ALREADY reused KV** (`conversation_tokens` +
      `plan_prompt_cache` LCP); the dense `generate_dense` path DROPPED `messages_history` entirely (single-shot, pos-0
      every request). Fix: both dense arms (`m.tp`/`m.pp_dense`) route through the SAME proven `plan_prompt_cache(&[], false)`
      (empty ckpts + resume off ⇒ single-GPU-**llama** behavior: strict-extension→reuse, divergence→cold-prefill);
      `generate_dense` is plan-driven (miss→batched `prefill`, hit→per-token `forward_token` suffix over the retained KV
      prefix) and returns `Option<Vec<u32>>` (`None`→caller clears `conversation_tokens`). **Pure-KV is positional ⇒ no
      #462 recurrent bleed; LCP reuses only the matching prefix.** **TRAP fixed in review (25f9046f): the decode loop
      breaks on max_tokens/stop AFTER `history.push` but BEFORE `forward_token`, so the last emitted token was baked into
      `conversation_tokens` without entering the KV → next-turn LCP over an unwritten slot (#462 mirror skew, the COMMON
      truncation case). Return `materialized` (pushed after each successful forward_token), not `history`.** Batched suffix
      prefill stays deferred to item **D** (needs cache-attention; B+D compose). Validated: `tp_multiturn_parity`
      (cache-hit suffix == full prefill, argmax-exact max|Δ|=2.5e-1) + `tp_prefill/decode_parity` (cold path unchanged) +
      **`coherence-gate.sh` PASS** (single-GPU serve fluent; pflash perf-stage fail = known gfx1100-baseline-vs-gfx1151 HW
      mismatch, untouched path) + **live dense-TP daemon multi-turn smoke PASS** (emulated tp=2: real cache HIT
      `lcp==prior_len`, turn-2 recalls a fact from history; tp=2 byte-identical to single-GPU on coherent prompts). **TRAP
      (cost ~real debugging): the daemon "os error 2 box-wide load failure" was a HARNESS BUG, not a daemon bug — the load
      message key is `model` NOT `path` (`daemon.rs:914` `msg.get("model").unwrap_or("")` → `HfqFile::open("")` → os error 2);
      the generate live-turn key is `prompt` NOT the `messages` array (`daemon.rs:1616` `unwrap_or("Hello")`, `messages` =
      PRIOR history only). Both my initial smoke AND the implementer's used the wrong keys ⇒ phantom "env blocker". Also: the
      0.6B model goes degenerate (Arabic attractor) on open-ended "remember my name" prompts — IDENTICAL on single-GPU
      (untouched) ⇒ model+prompt artifact, use factual/code prompts.** FOLLOWUP (checked): single-GPU dflash bake uses the
      spec-committed tail (KV-safe, can only undercount) so does NOT share the dense overcount skew — dense fix was a real
      new-gap close. Separate pre-existing latent: EP/minimax path (`daemon.rs:3725`) overcounts by one on stop-sequence
      exits (push-before-forward) — file separately. Spec/plan: local
      `docs/superpowers/{specs,plans}/2026-07-07-dense-multiturn-kv-reuse*.md`.
    - ✅ **C** — DONE 2026-07-08 `feature/device-mesh` `01234a93..df93a87a` (3 commits `c49de293`/`faf93ff3`/`df93a87a`, SDD).
      **FULL mesh-drive of qwen35-PP load** (routing + degree off the mesh; qwen35 was the last scalar-driven load path).
      **KEY DESIGN (revised by a 3-agent adversarial pre-review):** ragged `HIPFIRE_PP_LAYERS` bands are model-specific
      load-time data → carried on **`LoadCtx.pp_bands: Option<&'a [usize]>`, NOT on `DeviceMesh`** (mesh stays purely
      topological — `resolve_mesh`/`from_mesh`/`stage_for_layer`/mesh.rs + the 2 example callers all UNTOUCHED). The first
      draft put bands on the mesh; the review rejected it (topology smell: re-encodes `n_layers`, breaks
      `stage_for_layer(n_layers)` contract + `Eq`-means-placement; AND a real bug — the daemon builds one shared mesh
      before arch routing, so a banded mesh would flow into the dense arm where `from_mesh`/`PpModel` map uniformly →
      lying metadata). LoadCtx-carried bands dissolve all of it. **Shape:** pure `config::parse_pp_layers(env, pp)` helper
      (unit-tested: empty/None→uniform, len!=pp→Err, unparseable→Err); `load_model` takes `&DeviceMesh` (replaces `pp`
      scalar) + `pp_bands`, `ctx.pp=mesh.size_of(Pp)`; qwen35 carrier reads `ctx.pp_bands` (`Some`→`init_layers` VRAM-gate
      OFF / `None`→`init_uniform` gate ON — **preserved bit-for-bit**, the load-bearing constraint); daemon edge parses in
      the qwen35 arm ONLY (`pp>1 && !dense_llama_pp`), `dense_llama_pp` now on `mesh.has_axis(Pp)`. Validation split:
      `len==pp` at daemon edge (emit `{"type":"error"}`+`continue`, non-fatal), `sum==n_layers` in carrier (needs config).
      **Uniform path byte-identical; dense/EP/TP untouched.** Gates: build + no-gpu-ci PASS; coherence-gate no-hard-errors;
      emulated qwen35 PP-2 (qwen3.5-0.8b, 24L) B1 uniform / B2 ragged `10,14`→init_layers / B3a wrong-len→error-no-panic /
      B3b empty→uniform ALL PASS. Whole-branch review (opus) = **READY** (6/6 correctness props confirmed; no
      Critical/Important). **Follow-up (non-blocking):** `parse_pp_layers` accepts a `0` band (byte-identical to pre-change
      — no `>0` guard existed; harden with `count>0` if ragged PP graduates to real 2-GPU HW). Spec/plan (gitignored
      local): `docs/superpowers/{specs,plans}/2026-07-08-mesh-driven-qwen35-pp-load*.md`.
    - **D** — >256 cross-chunk batched prefill (needs the flash `forward_prefill_batch` cache-attention).
    - **E** *(small/low-value here)* — drop redundant rank0 whole-`LlamaWeights` (dead VRAM); revert/doc the `mq_x_rot`
      prefill VRAM growth; AWQ-batched rotate coverage.
  - 🚫 **HW-BLOCKED (need hipx/hiptrx 2-GPU):** real-hardware unload leak-check + a real 2-GPU same-arch =0 gate. Also the
    pre-existing `hipfire-loader/src/lib.rs:1702` pp>1 unload panic (orthogonal to dense PpModel/TpModel — file separately).
  - 🎯 **Big phase item after polish: P-E — Step::Recurrent/Conv + DeltaNet head-shard** (the last arch into the one spine).
- **(historical) PB-TP5 REMAINING (daemon productionization) — `load_model_tp` + `generate_tp` + `LoadedModel.tp`.** Large mechanical
  wiring (mirror the EP integration): (1) a `TpState` in hipfire-loader { `Gpus`, `WeightStore`, per-rank RankState
  (scratch+KV+norms), config, eos } ; (2) `LoadedModel.tp: Option<TpState>` — ripples `tp: None` to every LoadedModel
  constructor ; (3) `load_model_tp` builds the served model (tokenizer, chat-template/eos, recommended sampling, the
  fulfilled store) instead of the current stub ; (4) `generate_tp` in daemon.rs mirroring `generate_ep` (daemon.rs:2665,
  ~600L: ChatFrame render → prefill → decode loop [the tp_decode_parity algorithm] → stream JSON text events → sampling
  [temp/top_p, not just greedy] → stop/max_tokens/think-mode) ; (5) dispatch `if m.tp.is_some() { generate_tp(...);
  return; }` at daemon.rs:6552 (beside the `m.ep.is_some()` arm) ; (6) unload path for TpState. Validate with a LIVE
  daemon serve (not just a parity example): `hipfire serve --tp 2` on a dense llama HFQ under HIPFIRE_EMULATE_GPUS=2,
  then a chat request; token stream should match single-GPU serve. Leaner than generate_ep (standard llama chat, no MoE
  LCP/expert specifics). This is its own focused session — the daemon is a big critical file and needs live-serve testing.
- **(historical) PB-TP5 NEXT (the dense-TP serve loop) — fill in `load_model_tp`.** Now that the axis is disentangled: build a served
  `LoadedModel` for dense TP — per-rank sharded `LlamaWeights` from a `WeightStore` (the store→forward bridge, validated
  in `tp_full_model_parity`), per-rank scratch/KV, a `&mut Gpus`-threaded decode loop reusing the validated per-layer
  Step lists, embed(rank0+broadcast) + final norm/lm_head(rank0). Then `tp_decode_parity` (FNV token-stream vs
  single-GPU, mirror `ep_decode_parity`). The forward math is done; TP5 is the serve plumbing (daemon generate path +
  unload + the deferred-unload already handles the shard degree via `load_tp`=max(ep,tp)).
- **(historical) PB-TP4c REMAINING (the capstone — all primitives now PROVEN):** assemble `dense_forward_tp<A:DenseArch>(gpus,
  mesh, ...)` mirroring `dense_forward` (arch_spec.rs:132) but emitting per-rank Step lists (row `GemvResidual` → split
  Gemv→AllReduceOut→ResidualAdd per PB-TP4a) + the store→forward bridge: `fulfill_manifest(llama weight_manifest @
  arch.rs:98, Tp-2)` gives native-quant per-rank buffers in a `WeightStore`; assemble per-rank `DenseLayer{WeightRef}`
  via `WeightStore::take` + build `WeightRef` (bare GpuTensor + dtype/m/k, like the examples' `wref`; or
  `WeightTensor::dispatch_ref`). Per-rank `ForwardScratch`(n_heads=nh/tp)+`KvCache::new_gpu_q8(_,1,nkv/tp,hd,max_seq)`.
  Thread `&mut Gpus`. Single-GPU reference = the goal-1 harness (`llama::forward_scratch` on qwen3-0.6b-llama.mq4,
  Q8 KV). Validate full-model token/logit parity FP32+HIPFIRE_DETERMINISTIC=1 (Q8 KV common-mode). **Constraint: every
  sharded k-dim %256==0 (qwen3-0.6b Tp-2 OK: qkv k=1024, wo k=2048→1024, down k=3072→1536, gate/up k=1024).** Then TP5 =
  daemon `load_model_tp`+serve AFTER the EP-vs-TP fork (RAISE with bjoern: config.rs:155 `tp`→Ep).
- **(historical) PB-TP4b..5 REMAINING (real-model integration — the capstone):** TP4b = drive `Step::Attend` through the executor
  via the REAL llama `attend_plan` (builds `KvTierPlan{write_key,attend_key,...}` + `AttnParams` — needs a real model;
  synthesizing KvTierPlan by hand is fragile, do NOT). TP4c = the store→forward bridge (assemble per-rank sharded
  `LlamaWeights`/`DenseLayer` from a `WeightStore` via `take`) + a `dense_forward_tp<A:DenseArch>(gpus, mesh, ...)` that
  emits per-rank Step lists (mirror `dense_forward` @ arch_spec.rs:132 — note it uses whole scratch + `GemvResidual`
  for row ops, which TP must split per PB-TP4a) + thread `&mut Gpus`; full-model logit parity vs single-GPU (FP32 +
  `HIPFIRE_DETERMINISTIC=1`) on a small F32 llama (candidate `~/.hipfire/models/Qwen3-0.6B-PARO`, raw safetensors dir;
  NOTE: no existing example drives the runtime llama `forward_scratch` — needs a harness, and F32-raw-load on this
  branch is UNVERIFIED). **KEY UNKNOWN to resolve first in TP4c:** can a small F32 llama be driven standalone via
  `llama::load_weights`+`forward_scratch` (llama.rs:2794/3111) to produce a single-GPU logit reference? TP5 = daemon
  `load_model_tp`+serve AFTER the EP-vs-TP fork (config.rs:155 `tp`→Ep) + `tp_decode_parity`. **RAISE EP-vs-TP with
  bjoern before TP5.**
- **(superseded numbering) old PB-TP4 = wire
  `dense_forward` to emit per-rank sharded Steps when `mesh` Tp>1; thread `&mut Gpus`; real llama forward TP; full-model
  logit parity (FP32+DETERMINISTIC). TP5 = daemon `load_model_tp`+serve AFTER the EP-vs-TP intent fork (config.rs:155
  `tp`→Ep) + `tp_decode_parity`. **RAISE THE EP-vs-TP FORK with bjoern before TP5/daemon work** (his standing ask).

## COURSE CORRECTION + DIRECTION AGREED 2026-07-07 — DECOMPOSE into Steps is the target; P-D primitive = DEFERRED decomposition (NOT a "federation")
Triggered by a review of this branch vs the north star. bjoern challenged the "Step::Moe died → federation" read; on re-analysis he is right. Recorded here as the current direction + an error log. Carve-out note: [[moe-deltanet-decompose-into-steps]].

**AGREED DIRECTION:** compose MoE/DeltaNet/attention out of SEPARATE fine-grained Steps, splitting WHAT
(ordered ops + operands) from HOW/WHERE (device/rank/shard/collective — owned by mesh + manifest). The
arch never names a device/rank/collective. **`Step::Attend` (PB-TP4b) already proves the pattern**
(attention decomposed + head-parallel-transparent, KV placed by manifest). MoE + DeltaNet follow it.

**Why the monolithic reading was a strawman:** a single `Step::Moe` "absorbing run_moe_ep", AND the
`ep_moe_allreduce` primitive+closure, BOTH hide how/where in the arch → both FAIL the split's purpose.
Only the decomposition (Route → grouped-GEMM → SiluMul → grouped-down+combine, placement via manifest,
Ep all-reduce via executor) honors it. So under the north star's OWN principle the composition was
always the target; "Step::Moe doesn't fit" answered a strawman it never should have posed.

**ERROR LOG (mine, retracted):**
1. CONFLATION — ds4 `run_moe_ep` "ignores the OpBinding, dispatches via `self.state`" (forward.rs:2114)
   evidences a MONOLITHIC HAND-BLOCK never decomposed, NOT MoE being intrinsically un-step-able. The
   P-D "SuperOp mistake → primitive" record, the prior MEMORY.md line, and the (now-deleted) note all
   generalized "ds4's current code" into "MoE's nature." WRONG.
2. "Federation is the correct end-state / ratify it" — overstated; it's the CURRENT state from a
   shortcut, not an inevitability.
3. "DeltaNet resists Step-ification like MoE / is even harder" — WRONG DIRECTION. DeltaNet has NO
   data-dependent routing → its recurrent scan is a fixed-shape big-block op like `Step::Attend`;
   `Step::Recurrent`/`Conv` + manifest state co-shard fit. Its real cost is CORRECTNESS DEBT
   (s_ef_residual drop, Q8 stochastic determinism, greenfield dn_value_head_range, la_to_device), owed
   regardless of Step-vs-primitive.
4. Root: the plan wording ("P-D = Add `Step::Moe` (absorb run_moe_ep)") invited the monolithic read.
   Corrected inline in `docs/superpowers/plans/2026-07-05-device-mesh-transparent-parallelism.md`.

**GROUNDING (verified 2026-07-07):** MoE decomposition is WRAP+refactor, not greenfield — kernels exist:
`deepseek4_moe_topk_bias_aware[_batched]` (route), `gemm_*_moe_grouped_*` (gate/up + down, per
arch/dtype), `moe_down_combine_grouped_k8` (combine). `Step::Attend` already carries a `positions`
index-tensor operand → precedent for `Step::GroupedGemm{w_experts, offsets, x, y}`; step COUNT stays
fixed (one grouped-GEMM over a fixed expert count) → no data-dependent control flow. `ExpertSharded`
stays arch-specific but in the MANIFEST (how/where) — consistent with the split.

**P-D STATUS RE-LABELED:** the `ep_moe_allreduce` primitive is **DEFERRED decomposition**, kept as an
INTERIM (it can drive the arch body inside a decomposed executor during migration, then retire —
additive). NOT the intended end-state.

**P-E RE-SCOPED:** decompose qwen35 DeltaNet into `Step::Recurrent`/`Step::Conv` + projection Gemvs,
state co-sharded via a manifest `StateEntry` (mirror `Step::Attend`). Pay the correctness debt FIRST
(fix `s_ef_residual`; FP32+DETERMINISTIC harness). Do NOT build a DeltaNet primitive by analogy to EP —
that was the retracted misprediction.

**OUTSTANDING FACTUAL INCONSISTENCIES / TODOs surfaced by the review:**
- The verification subagent claimed `LoadedModel.tp`/`.pp_dense` are "currently unused in the daemon
  path" — **WRONG.** They ARE served: `generate_dense` dispatch at daemon.rs:6761 (`m.tp.is_some()`) /
  :6810 (`m.pp_dense.is_some()`), reset guard :2393. Validated live (PB-TP5 6b71b132, PC-3 44edf2d6).
- `tp_prefill_parity`'s old doc-comment ("~4.2e-4 / a bit above 1e-3") was FICTION (never asserted) — the
  real prefill floor is ~0.20 Q8-flash-vs-TP (item A, 91e99955). Doc/code drift; fixed in item A.
- `hipfire-loader/src/lib.rs:1708` dense-PP unload panic (`pp_gpus.expect`) — **FIXED 2026-07-07**
  [[dense-pp-unload-panic-pp-gpus-expect]]. Was NOT pre-existing/qwen35-only: the informational `pp`
  scalar (:1355, mesh-through-loader) + the non-returning `pp_dense` unload arm (:1633) fell through into
  the qwen35-PP teardown (:1707). Fix = `return` after the pp_dense drop (mirror EP arm); repro+regression
  `examples/pp_unload_reload.rs` (panic→PASS, emulated Pp-2).
- `.agent-progress/device-mesh-HANDOVER.md` is PRE-PIVOT (2026-07-05) and stale — THIS spine note supersedes it.
- The prior global MEMORY.md device-mesh line was stale ("P-A DONE / P-B START"); refreshed 2026-07-07.

**The one structural unification still worth doing (unchanged):** daemon god-struct →
`ModelParallel` + `ArchDispatch` (old Phase 3) — the ~20-`Option` `LoadedModel` + the #462 state-bleed
surface. NOT single-IR purity (decomposition delivers transparency without a god-object collapse; the
two are independent).

## UPSTREAM MERGE 2026-07-08 — merged upstream/master (DSpark + ddtree, 155 commits) @ merge commit `1df219dd`

Branch now **0 behind / 153 ahead** of `upstream/master` (Kaden-Schutt `45cb5abf`). **TRAP for next
time:** `origin` = fivetide is a STALE MIRROR (155 behind upstream). Fetch/merge against the `upstream`
remote, not `origin`. Merge-base was `d44f89e7`.

**What upstream added:** dominantly DSpark (DeepSeek-V4-style multi-stage draft-module spec-decode, ported
to deepseek4 + qwen3/qwen35) + ddtree GPU-residency/tree-verify (SWOR, temp>0 distribution-exact). New
modules `dspark_core`/`dspark_block_controller`/`dflash_generic`; new kernels (`dspark_*.hip`,
`ddtree_*.hip`, `chain_accept_spec.hip`, `batched_categorical_sample.hip`). ~+23k/−2k LOC. **Zero** of it
touches TP/PP/EP or the SuperOp subsystem.

**Resolution (3-agent static research, high confidence): KEEP our SuperOp/EP deletion.** No upstream
feature depends on it — every live `ep::`/`SuperOp` reference in upstream is pre-existing base-era EP
forward-path code (`ep.rs`/`superop.rs` byte-unchanged across all 155 commits); DSpark/ddtree is
orthogonal and references none of it.

Conflicts resolved (2 textual):
- `hipfire-runtime/src/lib.rs`: kept upstream's 3 dspark mods (ungated, matching upstream's own
  placement), dropped `pub mod ep;`.
- `hipfire-arch-qwen35/src/qwen35.rs`: took OUR deletion side — dropped upstream's ~1094-line SuperOp
  lowering block (`q35_superop`/`lower_variant`/`LayerProgram`).

**The load-bearing non-textual break (no conflict marker — would've silently shipped broken):**
`hipfire-runtime/src/llama.rs` auto-merge TANGLED our PP-band primitive `forward_scratch_band(layer_range)`
with upstream's new `forward_scratch_compute_capture` (DFlash hidden-capture). The two share a near-identical
per-layer loop, so git mis-aligned braces: `band` got a 1-line stub body (ignored `layer_range`, wrongly ran
the head), and `compute_capture` inherited an undefined `layer_range` ref AND lost its final norm+lm_head.
Reconstructed BOTH verbatim from their authoritative sides (band from HEAD — per-layer logic proven ==base
by diff; capture from upstream). `forward_scratch_compute` (= band+head) was intact.

`LoadCtx`: upstream folded dspark cfg into `SpecLoadCfg` (`ddtree_budget`/`ddtree_topk`/`dspark`/
`dspark_conf_threshold`) — auto-merged clean. Fixed 3 upstream examples missing our `pp_bands` field + 1
example calling `load_model` with the pre-mesh arg list (needs `&mesh` + `pp_bands`).

Cleanup (same session, folded into the follow-up commit): removed dead `q35_op`/`Q35Variant`/`variant_of`
SuperOp stub + its `#397 Ship 6` doc header in qwen35.rs; retargeted 5 stale
`hipfire_runtime::ep::{run_layer_program_ep, ensure_rank_streams}` doc links in deepseek4/minimax
`forward.rs` → `Gpus::{ep_moe_allreduce, ensure_rank_streams}`.

**Validation:** `cargo check --workspace --all-targets` (default incl. deltanet) 0 errors; daemon
`--features deltanet` clean; `cargo test --lib --workspace` 966 pass/0 fail; `coherence-gate.sh` coherence
PASS (no hard errors, qwen3.5 matrix); llama-arch `coherence_probe` OK 0/0 (covers the pure-llama
`forward_scratch_compute` path). **DFlash/spec-decode (the merge's dominant new surface + my capture-fn
reconstruction):** `coherence-gate-qwen3-dspark.sh` PASS — daemon-based, drafter ENGAGED
(`llama DSpark speculator enabled, block=7`, `dflash:true` τ=0.94–1.36), all 3 rows OK, exercises
`dspark_body`/`dspark_core` → arch-llama `forward_scratch_compute_capture` (my reconstructed fn);
`coherence-gate-dflash.sh` (qwen36-27b DeltaNet + ddtree) PASS — all 4 rows OK (t1/t2/t3 clean,
unique_ratio 0.68–0.75), τ healthy (dflash-code 7.8, prose 1.0–1.6, ddtree 1.58). pflash perf-stage
FAIL = known gfx1100-baseline-vs-gfx1151 artifact (uniform ~2× incl untouched AR baseline rows), NOT the
merge.

**Worktree gotcha:** `core.hooksPath` = `.git/hooks` (no pre-commit) → the pre-commit gate never fires
here; run gates manually.
