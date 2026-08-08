---
title: P-D-decompose D2b COMPLETE — minimax+ds4 TP-of-experts (column/row-sliced experts + inter_local + int64 AllReduce{Tp}), BIT-EXACT emulated Tp-2, EP regression holds
date: 2026-07-09
tags: [device-mesh, moe, p-d, decompose, d2b, tp-of-experts, tensor-parallel, deepseek4, minimax, step-ir, int64, bit-exact, partition-invariant, allreduce-tp, load_weights_tp]
---

**Branch:** `feature/device-mesh` (worktree `.claude/worktrees/feature+device-mesh`). Plan: `docs/superpowers/plans/2026-07-08-d2b-tp-of-experts.md` (Tasks 1–3 = shared slicer + `ExpertTensorSharded` policy + minimax arm; Task 4 = ds4 arm; Task 5 = this note). Parent: [[pd-decompose-d2a-decode-complete]] + [[reproducible-moe-down-kernel-complete]].

## What D2b delivered
**Tensor-parallelism of MoE experts** for minimax(10) + ds4(9): every rank owns ALL routed experts, but each expert's gate‖up is **column-split** and its down is **row-gathered** to `inter/tp`; the routed partial is reduced across the Tp group. Distinct from EP (which sub-sets WHICH experts a rank owns). Validated **argmax-exact AND logit max|Δ| == 0.0** (BIT-EXACT) on emulated Tp-2 vs tp=1, both minimax (M2.7.mq2) and ds4 (deepseek-v4-flash.mq2lloyd). **No .hip kernel changes** — kernels already take K/m as runtime params and 256-alignment makes K=inter/tp valid.

## The bit-exactness lever (why max|Δ|==0, not just "small")
The reproducible **int64 down kernel** ([[reproducible-moe-down-kernel-complete]]) is what makes TP bit-exact. `DownResidualI64` accumulates the weighted-combine in fixed-point i64. i64 addition is associative/exact → **partition-invariant**: `sum(i64_rank0 + i64_rank1) == i64_full`. So tp=1 (full-inter i64) and tp=2 (two inter/2 i64 halves, `AllReduceI64Tp`-summed) produce identical i64 → identical f32 after `ConvertI64ToF32`. The plan's original bar was argmax-exact + max|Δ|<1e-2 (f32-path); the int64 work upgraded it to == 0.0. (Earlier minimax MQ3L f32-split had max|Δ|~4.55 from atomicAdd K-split order — argmax-exact only; int64 removed even that.)

## Step ordering: TP vs EP branch (KEY)
`{minimax,ds4}_ep_moe_step` keys the down collective on `tp = mesh.size_of(Tp).max(1)`:
- **TP (tp>1):** steps `[..GateUp, MoeActivation, DownResidualI64, ConvertI64ToF32]`, collectives `[.., None, AllReduceI64Tp{hidden}, None]`, `zero_before[DownI64]=true`. Reduce i64 BEFORE convert → partition-invariant.
- **EP (tp==1, unchanged):** collectives `[.., None, ZeroI64Only{hidden}, AllReduce{Ep,hidden}]`. Each rank owns distinct experts → convert per-rank, then FP32 all-reduce.
`inter_local = inter/tp` threads into `MoeExpertRef.expert_m` (covers GateUp out=2·inter_local AND Down contraction=inter_local; kernel derives 2× internally) and `MoeActivation.inter`. tp==1 → inter_local==inter → EP path byte-identical.

## Loader (arch imperative path — the working GPU path)
- **`TpExpertSlice{tp, rank}`** (`tp_shard.rs`) with `inter_local()`. Shared CPU slicer in `weight_store.rs`: `expert_tp_column_pair` (gate‖up `[2·inter,hidden]`→`[2·inter/tp,hidden]`) + `expert_tp_row_gather` (down `[hidden,inter]`→`[hidden,inter/tp]`), pure byte moves on self-contained MQ-Lloyd blocks. Sliced per-expert stride == full `/tp` exactly (both cuts shrink 1/tp).
- **minimax** (Task 3, `bd5ddcbf`): `MiniMaxWeights::load(.., tp_slice)` 5th param; `gu_packed_stride`/`dn_packed_stride` separate from raw file strides for the ptr table.
- **ds4** (Task 4): new `DeepseekV4::load_weights_tp(hfq, cfg, gpu, TpExpertSlice)` → `load_weights_inner(.., None, Some(ts))`; sliced in `upload_layer_routed_experts` (threaded to BOTH `layers.{L}` AND `mtp.0` — MTP shares `ds4_ep_moe_step`; dspark stages get `None`). block_bytes from `HfqTensorInfo.quant_type` (19=MQ2L→72, 20=MQ3L→112). **RefCell trap**: column-pair needs w1‖w3 contiguous → copy each pread to an owned Vec (drop the `Ref`) before slicing; the un-sliced path keeps the two-block extend (no extra copy). EP `shard` + `tp_slice` mutually exclusive (guarded).

## Phase boundary — NO double-count (ds4 shared expert)
ds4 has a shared expert (minimax does not). Shared (`ffn_stub`) + `mhc_pre` + route stay Phase-1 direct/replicated → `state.ffn_out`. Only the routed `partials[r]` is Tp-reduced. Phase-3 `ffn_out += partials[r]` = shared(replicated, once) + routed(reduced), so shared is added exactly once per rank, NOT ×tp. `hc_ffn_mix` tail post-add. Confirmed by argmax-exact across hash layers 0-2 AND bias layers 3+ (43 layers, hidden=4096, inter=2048→1024/rank, 256 experts).

## New forwards (sibling-duplication pattern, like forward_ep/forward_tp)
- ds4 `forward_tp` (mirror of `forward_ep`, only diff = `DimKind::Tp` mesh) + `mtp_forward_tp` (mirror of `mtp_forward_ep`). Both drive `ds4_ep_moe_step` with a Tp mesh.
- Harness `crates/hipfire-arch-deepseek4/examples/tp_deepseek4.rs` (mirror of `ep_deepseek4.rs` + `tp_minimax.rs`): both runs via `forward_tp` (int64), asserts argmax-exact + max|Δ|==0.0, `--mtp` confirms MTP draft parity tp1==tp2, DSpark forced off, per-run free+drain_pool so 88GB fits twice under `HIPFIRE_EMULATE_GPUS=2`.

## Validation (this session, gfx1151, HIPFIRE_DETERMINISTIC=1 EMULATE_GPUS=2 EP_PEER_ALLREDUCE_DECODE=1)
- `tp_deepseek4 --max 32 --no-dspark`: **argmax-exact=true, logit max|Δ|=0.00e0**, coherent ("Paris. It is located in the north..."). Load: tp=1 81s, tp=2 177s (down row-gather included).
- `ep_deepseek4 --tp 2 --max 32 --no-dspark`: FNV **`0x6c0f2f000f1d398f`** — pinned EP-2 baseline unchanged (tp==1 path byte-identical).
- minimax pinned: `ep_minimax --tp 2` FNV `0x887c2e7717e9c3bf` (Task 3, held).
- `cargo build --release --workspace --all-targets --locked` green.

## fmt trap (re-confirmed)
`scripts/fmt-changed.sh` reformats EVERY file with pre-existing format-debt vs base (~15 files churned here, none mine) — revert the collateral and commit only intended files. Also fmt inserted `block_bytes_for_qt` such that it orphaned `upload_layer_routed_experts`' doc comment — hand-reordered. This worktree's `core.hooksPath` is unset → pre-commit gate doesn't auto-fire; gates run manually.

## Deferred (not in D2b)
- **2D EP×TP** (experts sub-set AND sliced) — needs real ≥2-GPU HW; single-axis only here. The D2a same-`kind` collective guard blocks a mis-reduce.
- **Manifest-transparent MoE loading**: `ShardPolicy::ExpertTensorSharded{n_experts, inner}` variant + fulfill arm exist (Task 2, `9131f131`) and are unit-tested, but the arch imperative loaders are still the GPU path.
- **D2c prefill** (>256 batched) TP-of-experts.
- ds4 TP is AR + MTP-draft only; dspark drafter stages not sliced (`--no-dspark`).
