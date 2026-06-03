# Design: MoE dispatch as a composable pipeline

**Date:** 2026-06-03
**Branch:** feature/dispatch-unification
**Tracks:** PR #393 item #2 (decide MoE dispatch routing API). Gates #6 (grouped-expert kernel) and #7 (`MoeFamily::run()`).
**Status:** approved, ready for implementation plan

## Problem

`MoeFamily::run()` returns `UnsupportedVariant` today — MoE expert compute lives
entirely in per-model paths (`crates/hipfire-arch-qwen35/src/qwen35.rs:
moe_ffn_decode_impl`). PR reviewer question 1 asks where the dispatch boundary
should sit: does `MoeFamily::run()` own top-k routing + scatter/gather, or is it
kernel-only with routing left to the model?

The codebase has two MoE archs with **divergent routing math** but a **shared
expert-compute substrate**:

- **qwen35**: softmax -> top-8 -> optional renorm (`norm_topk_prob`), k=8,
  shared expert with sigmoid-scaled gate.
- **deepseek4**: `topk_method` (string), host-cached `gate_bias` for top-k *with
  bias*, group-limited routing, k=6. (Currently a config skeleton — MoE forward
  not yet implemented.)

A single monolithic `run()` boundary forces either (a) routing math duplicated
per model, or (b) a config-`match` over every arch's routing algorithm buried
inside one function — fragile given the correctness-sensitive history (a 1-ULP
softmax/renorm error compounded into a structural attractor across A3B layers;
see `moe_topk_renorm_k8` comments and memory `moe-attractor.md`).

## Decision

Express the MoE forward as an **ordered `PipelineOp` list**, reusing the existing
GEMV/GEMM pipeline mechanism (`crates/hipfire-dispatch/src/pipeline/mod.rs`).
The two archs differ only in the **routing prefix**; the expert-compute tail is
shared ops. This dissolves the binary boundary question: routing divergence
becomes a *different op list*, not a config-switch inside `run()`.

Two architectural choices were settled during brainstorming:

1. **Params carrier = enum by op-family.** `PipelineParams` becomes
   `enum { Linear(LinearParams), Moe(MoeParams) }`. Most type-safe; a wrong
   variant for an op is a programming error guarded once per arm.
2. **Sequencing = reframe-first.** Phase 1 expresses the *existing* indexed
   decode path as a pipeline op-list with **no new kernel** (byte-parity
   refactor). Phase 2 (#6/#7) adds the grouped-expert kernel, resolve-selected
   by batch size.

## How the pipeline mechanism applies

The existing pipeline is three things:

1. A vocabulary of primitive `PipelineOp`s.
2. Each kernel-table entry declares the ordered op-sequence it satisfies.
3. `execute_pipeline` either finds **one fused kernel** covering a prefix
   (`find_fused`, fast path) or **falls back** to launching each op separately.

MoE fits because each MoE step already has a fused fast-path *and* a discrete
fallback in today's code — exactly the duality the pipeline rewards. The
GPU-top-k-vs-CPU-fallback split, and the fused-4-way-GEMV vs 4-separate-GEMV
split, both map directly onto fused-vs-fallback.

## Op vocabulary (new `PipelineOp` variants)

Each op carries one fused impl and one fallback. Fused impls are today's
fast-path kernels; fallbacks are today's slow paths.

| New op | Fused impl | Fallback |
|---|---|---|
| `MoeGateSideProj` | `fused_qkvza_hfq4g256` (router + shared_expert_gate + shared.gate + shared.up, 4-way) | 4x `weight_gemv` |
| `Softmax` | `gpu.softmax_f32` | — (shared math) |
| `TopKRenorm{k}` | `moe_topk_renorm_k8` (GPU, hipGraph-capture-safe) | CPU download + `select_nth_unstable` + renorm |
| `SharedExpertDown` | `gemv_hfq4g256_residual_sigmoid_scaled_gpu` (silu·mul·rotate + down + sigmoid + residual-add fused) | sigmoid + silu_mul + weight_gemv + scaled_add |
| `IndexedGateUp` | `gemv_{hfq4g256,hfq6g256,paro_q4g128}_moe_gate_up_k8_indexed` (dtype-resolved) | per-expert loop |
| `SiluMulRotate` | `fused_silu_mul_{rotate_mq,givens_rotate}` (batched) | silu_mul + rotate (2 launches) |
| `IndexedDownExpanded` | `gemv_*_moe_down_k8_indexed_batched_expanded` | per-expert loop |
| `MoeCombine` | `moe_down_combine_k8_batched` (atomic-free, deterministic) | — |

deepseek4's routing divergence is a **single substituted op** —
`SigmoidBiasGroupTopK{k}` in place of `Softmax` + `TopKRenorm` — not a config
`match` inside a monolith. The vocabulary leaves room for it; implementation is
out of scope here.

## qwen35 decode op-list

A faithful linearization of `moe_ffn_decode_impl` (decode path):

```
[ MoeGateSideProj, Softmax, TopKRenorm{8},
  SharedExpertDown,
  IndexedGateUp, SiluMulRotate, IndexedDownExpanded, MoeCombine ]
```

The shared-expert branch is **flattened in-line** rather than modeled as a
parallel sub-pipeline. It is a sequence of independent launches today, so
linearizing it changes nothing about execution order or numerics. (Approved
sanity-check (a).)

## Params carrier

```rust
pub enum PipelineParams<'a> {
    Linear(LinearParams<'a>),   // = today's struct, renamed; { x, y, buf, m, k }
    Moe(MoeParams<'a>),         // weights, x_rot, expert_gate_up_ptrs,
                                //   expert_down_ptrs, topk_indices, topk_weights,
                                //   gate_batch, up_batch, rot_batch,
                                //   down_expanded, k_top
}
```

- The existing `PipelineParams { x, y, buf, m, k }` struct is **renamed**
  `LinearParams`. Migration surface is one external construction site
  (`crates/hipfire-dispatch/src/families/gemv.rs:274`, wrap in `Linear(..)`)
  plus the two internal functions (`execute_pipeline`, `dispatch_fused`).
- `MoeParams` (already exists in `crates/hipfire-dispatch/src/families/moe.rs`)
  grows to carry the scratch refs the MoE ops consume.
- **dtype resolution moves per-op for the MoE arm.** `execute_pipeline`'s single
  `dtype` argument is GEMV-centric. MoE ops resolve dtype *per op* from the
  weights in `MoeParams`, because gate-side (MQ4) and routed (MQ6/Paro) families
  can differ within one layer. This is the one non-mechanical executor change.
  (Approved sanity-check (b).)

## Executor & determinism

- `execute_pipeline` matches `Linear` (unchanged behavior) vs `Moe` (new arm
  iterating the MoE op-list).
- `find_fused` gains MoE entries (prefix-capture of `MoeGateSideProj`,
  `SharedExpertDown`, the indexed gate_up/down kernels) — same hand-written
  `match` style as the existing `GemvMfp4G32Fused` entry.
- **Determinism is preserved by construction.** `TopKRenorm` keeps the
  split-softmax-then-renorm math (the documented 1-ULP attractor fix);
  `MoeCombine` stays the atomic-free expand->combine (avoids the wavefront-order
  FP32 non-determinism that diverges under hipGraph capture). The fused and
  fallback variants of each op must be numerically equivalent.

## Phasing

### Phase 1 — reframe (this item, #2)

Scope: **decode path only** (`moe_ffn_decode_impl`).

1. Rename `PipelineParams` struct -> `LinearParams`; introduce the
   `PipelineParams` enum. Update the one external caller + two internal fns.
2. Add the MoE `PipelineOp` variants.
3. Grow `MoeParams` to carry the MoE scratch refs.
4. Add the MoE arm to `execute_pipeline` with per-op dtype resolution and MoE
   `find_fused` entries.
5. Replace `moe_ffn_decode_impl`'s body with an `execute_pipeline(Moe(..))`
   call producing the qwen35 decode op-list.

**No new kernel in Phase 1.** This closes #2 and unblocks #9 (qwen35
dispatch-adjacent absorption).

### Phase 2 — grouped kernel (#6 / #7)

1. Add `GroupedGateUp` / `GroupedDown` ops + the new grouped HFQ4G256 GEMM
   kernel.
2. Resolve-select grouped (prefill, large batch) vs indexed (decode, batch=1)
   by `batch_size`, mirroring existing GEMV-vs-GEMM dispatch.
3. `MoeFamily::run()` (#7) becomes a thin `execute_pipeline(Moe(..))` wrapper.

## Verification

- Phase 1 success criterion is **byte-identical output**. The reframe must not
  move a single bit.
- Gate: `./scripts/coherence-gate.sh` plus a byte-parity A/B vs current `master`
  on an A3B MoE model (e.g. Qwen3.5-A3B at MQ4), with a byte-identical prompt
  (record prompt md5 per the CLAUDE.md bench rule).
- Coherence is mandatory for any dispatch/fusion change (CLAUDE.md coherence
  gate). The pre-commit hook runs it when dispatch files are staged.

## Non-goals

- No cross-family fusion beyond fused kernels that already exist; no new fused
  kernels in Phase 1.
- No deepseek4 routing implementation. The vocabulary accommodates
  `SigmoidBiasGroupTopK`; the impl is a separate later item.
- No change to routing math.
- **Not in Phase 1:** the batched-prefill path
  (`forward_prefill_batch_with_pbs`) and the PARO prefill echo bug (#1) — kept
  separate so byte-parity stays tractable. (#1 is a distinct Layer-3 item.)

## Open risks

- **Per-op dtype resolution** is the one place the executor stops being a dumb
  op-runner. If a future arch mixes dtypes in a way the current
  `routed_dtype_indexable_*` checks don't cover, the resolve logic needs
  extending. Acceptable: the existing code already encodes these same
  constraints; the reframe relocates them, it does not weaken them.
- **Flattening the shared-expert branch** assumes it stays a sequential launch
  block. If a future fused kernel wants to co-schedule shared + routed experts,
  the linear op-list would need a parallel-sub-pipeline construct. Out of scope;
  revisit only if such a kernel is built.
