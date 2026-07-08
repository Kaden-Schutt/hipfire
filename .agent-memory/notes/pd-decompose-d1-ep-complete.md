---
title: P-D-decompose D1 COMPLETE — minimax+ds4 MoE Ep-collective+down decomposed into Steps, ep_moe_allreduce RETIRED, byte-identical emulated EP-2
date: 2026-07-08
tags: [device-mesh, moe, p-d, decompose, execute_steps_parallel, ep, deepseek4, minimax, step-ir, byte-identical, retire-primitive, d1, d2-next]
---

**Branch:** `feature/device-mesh` (worktree `.claude/worktrees/feature+device-mesh`). D1 = `73dfc583..8d25d6c8` (10 commits). Parent direction: [[moe-deltanet-decompose-into-steps]] + [[device-mesh-pivot-execute-steps-spine]] (`## COURSE CORRECTION 2026-07-07`). Spec: `docs/superpowers/specs/2026-07-08-pd-decompose-moe-steps-design.md`; plan: `docs/superpowers/plans/2026-07-08-pd-decompose-moe-d1-ep.md` (both gitignored-local per convention).

## What D1 delivered
The interim `Gpus::ep_moe_allreduce` primitive (a closure that ran experts AND received `rank` → coupled what/how/where) is **DELETED (−952 LOC)**. minimax + ds4 EP MoE now moves the **Ep collective + routed-down-projection** through a new axis-keyed executor `execute_steps_parallel`; the arch no longer names a rank or a collective. **Validated BYTE-IDENTICAL to the primitive on emulated EP-2** (same-process A/B, only the arm differs): minimax FNV `0x887c2e7717e9c3bf`, ds4 `0x6c0f2f000f1d398f` + MTP-EP draft ACCEPT.

## D1 is a NARROW (partial) decomposition — this was a DELIBERATE bjoern-approved call
Byte-identity FORCED it: only **down-projection + Ep collective** go through the executor. The pre-down MoE math (rmsnorm+rotate / bias-aware route / gate-up / fused `silu·mul+rotate` / sigmoid) stays as **direct arch kernels** — the arch fuses them (e.g. `fused_silu_mul_rotate_mq_batched_for`) in single kernels with NO bit-identical generic-`Step` twin; a `SiluMul`+separate-rotate would diverge. This is the pivot's explicit "deferred decomposition / interim" stance. Full fine-grained decomposition (route/gate-up/silu as byte-identical fused Step variants) is **D2's job**, where TP-of-experts actually needs those ops sharded.

## The executor (crates/hipfire-dispatch/src/pipeline/steps.rs)
`execute_steps_parallel(mesh, gpus, per_rank_steps: &[Vec<Step>], collectives: &[StepCollective], zero_before: &[bool])` — generalizes the proven `execute_steps_tp` (now a thin wrapper, TP path re-verified byte-identical: ffn 2.79e-9 / quant 7.45e-9 / full-model argmax 33450 logit 4.2e-4).
- `StepCollective::{None, AllReduce{kind: DimKind, dim}}`. Peer/RCCL choice is **axis-keyed**: `Tp → ALWAYS all_reduce_sum_f32_peer` (no env gate — preserves TP byte-identity; RCCL absent on box); `Ep → ep_peer_allreduce_decode()? peer : all_reduce_sum_f32` (matches the old primitive).
- `zero_before[i]=true` memsets the step's `tp_step_out_buf` before running it (mirrors the primitive's pre-RunOwned partial memset — accumulate semantics).
- **LIMITATION (safe D1, must fix before D2 2D):** the reduce group is resolved ONCE from the FIRST `AllReduce` collective's `kind`. Safe now (every D1 caller passes a homogeneous all-`Ep` slice). **D2's Tp+Ep 2D compose MUST add per-step group resolution OR a same-`kind` `debug_assert`** or it silently reduces over the wrong group.

## Step variants (Tasks 4/5) — which are LIVE vs BUILT-BUT-UNUSED
- **LIVE (production EP paths use):** `IndexedMoeGemv{which: MoeProj::DownExpanded | DownResidual}`, `MoeCombine{inverse_perm}`. `MoeExpertRef` operand (borrows the arch's bespoke device-ptr tables `expert_gate_up_ptrs`/`expert_down_ptrs`/`dummy_gate_up`). `launch_hc_ffn_mix` hook (ds4 tail).
- **BUILT-BUT-UNUSED (D2 scaffolding, compile+arg-tested only):** `Step::MoeRoute{gate_bias}`, `IndexedMoeGemv{GateUp}`, the new `SiluMul` use, and ALL prefill grouped variants `MoeScatter`/`GroupedMoeGemm`/`MoeUnscatter` + the grouped-combine path. **The prefill-grouped Steps are WHOLLY UNVALIDATED** (no `forward_batch_ep` primitive exists to A/B against). D2 followup: validate-or-strip them.

## The two down shapes (critical wiring — Task-3/4 finding)
Decode down has TWO kernel shapes, and DownResidual FOLDS the weighted combine (double-accumulate trap if you also emit MoeCombine):
- **Expanded** (MQ4/HFQ4/MQ6): `IndexedMoeGemv{DownExpanded}` (zero_before=false) → `MoeCombine{inverse_perm:None}` (zero_before=true, out=partial). Kernel `*_moe_down_expanded_k4` + `moe_down_combine_k8_batched`.
- **Residual-fused** (MQ2L/MQ3L): lone `IndexedMoeGemv{DownResidual{topk_weights}}` (zero_before=true, out=partial) — NO MoeCombine. Kernel `*_moe_down_residual_scaled_indexed`.

## Per-arch shape
- **minimax** (`crates/hipfire-arch-minimax/src/forward.rs` `minimax_ep_moe_step`): pre-down direct → **dtype-conditional** down (Lloyd→DownResidual / non-Lloyd→DownExpanded+MoeCombine — the Task-10 fix; shipped M2.7.mq2 down=MQ3L→DownResidual) → AllReduce{Ep} → `Step::ResidualAdd` tail (`state.h += partial`). gate/up experts = MQ2-Lloyd, down = MQ3-Lloyd.
- **ds4** (`crates/hipfire-arch-deepseek4/src/forward.rs` `ds4_ep_moe_step`, also `mtp_forward_ep`): shared expert `ffn_stub` FIRST into ffn_out (replicated, NOT in the EP partial) → routed pre-down direct → **hash(l<num_hash_layers=3)→DownResidual / non-hash→DownExpanded+MoeCombine** → AllReduce{Ep} → `launch_hc_ffn_mix` hook (post-reduce, `streams_out=state.q`, sub_offset(4,4)/(8,16)). routed experts = MQ2-Lloyd. `if routed { ffn_out += partial }` skips stale-partial add on shared-only/skip_ffn layers.

## Integration traps (any EP-through-executor path)
- Build an Ep-axis mesh: `DeviceMesh::rect(&[(DimKind::Ep, n)])` so `group_along(Ep)==group`.
- `Gpus::ensure_rank_streams()` before the executor (zero_before memset + all-reduce need each rank's `active_stream`). Single-GPU/PP must NOT.
- RCCL is NOT installed → EP runs need `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1` (bare RCCL panics at `RcclComms::init_all`).
- Emulated EP-2 (`HIPFIRE_EMULATE_GPUS=2`, both ranks = device 0, intra-device peer copy) is the validation CEILING on this box. **Live 2-GPU byte-identity = hipx exit** (ds4 88GB / minimax 79GB fit UMA for emulated only).
- MQ2-Lloyd GROUPED kernel variant must match production: helpers route through `select_grouped_lloyd_variant`/`dispatch_grouped_lloyd` (pub(crate)) → gfx1151 picks Lloyd4w not Base (Task-5 fix; Base≠Lloyd4w bit-wise).

## D2 = TP-of-experts (greenfield, the next plan)
NOT free — confirmed by the 4-agent adversarial review. Needs: composite `ShardPolicy` (`ExpertSharded × {Column,Row}` — flat enum can't compose today), per-expert TP sub-slice in `fulfill_manifest`'s `expert_compact_blob` (k%256-aligned; today slices only the outer expert dim), grouped/indexed kernel `tp_rank`/`tp_size` params (today `expert_weight_ptrs[E]`, no TP param), and a combine-time `Tp` all-reduce (today `moe_down_combine` has no reduce hook). Plus the full pre-down decomposition (fused Step variants matching the arch kernels) so route/gate-up/silu shard under TP. Validate emulated Tp-2 numeric parity (FP32+DETERMINISTIC). PP-for-MoE is ~free at the driver.

## Process gotcha (recorded so it doesn't recur)
A fix subagent ran a **bare `cargo fmt` workspace-wide** (20 unrelated files, forbidden by CLAUDE.md) then died at the session limit without committing. Recovered by `git checkout HEAD --` the churned files, keeping only the clean target-file fix. **Fix/impl subagent dispatches on this repo must explicitly forbid bare `cargo fmt` (use `scripts/fmt-changed.sh`) AND never wholesale-reformat the fmt-debt-heavy arch `forward.rs` files.**
