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
- **Future TP polish (not blocking):** batched prefill (currently per-token); multi-turn KV reuse (currently stateless);
  drop the redundant whole-`LlamaWeights` on rank0 (only embed/output_norm/lm_head + norms are used — the rank0 quant
  layers are dead VRAM); real-hardware unload leak-check (emulated drop is fine); `resolve_mesh` isn't yet called by the
  daemon (load routes on the raw `tp`/`ep` knobs) — wire it if mesh-driven load lands. Then P-C (PP-at-driver), P-D
  (Step::Moe/EP fold retiring run_layer_program_mesh), P-E (Step::Recurrent/Conv + DeltaNet head-shard).
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
