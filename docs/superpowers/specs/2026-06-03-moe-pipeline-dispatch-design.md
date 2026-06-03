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
shared ops. Routing divergence becomes a *different op list* plus a *different
resolution pre-pass* — not a config-`match` inlined in the model's forward. The
config logic does not vanish (see "Resolution pre-pass"); it is relocated into
one typed, testable place. That is the honest framing of "dissolves the
boundary question."

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

**Caveat (adversarial review, finding #1):** unlike the GEMV pipeline, MoE
fused-vs-fallback eligibility is **not** independent per op. It is a coupled
lattice resolved jointly — see "Resolution pre-pass" below. The op-list models
*execution order*; a pre-pass resolves *which variant each op runs*. The clean
table is honest about which kernels exist, not about how independently they are
selected.

MoE fits because each MoE step already has a fused fast-path *and* a discrete
fallback in today's code — exactly the duality the pipeline rewards. The
GPU-top-k-vs-CPU-fallback split, and the fused-4-way-GEMV vs 4-separate-GEMV
split, both map directly onto fused-vs-fallback.

## Op vocabulary (new `PipelineOp` variants)

Each op carries one fused impl and one fallback. Fused impls are today's
fast-path kernels; fallbacks are today's slow paths. Two entries —
`MoeGateSideProj` and `SharedExpertDown` — are **composite macro-ops** (their
fused impl fuses several sub-steps; the prefix-matcher fuses them all-or-nothing,
it cannot fuse part of a macro-op and run the rest discretely). This is fine for
qwen35 but means the "composable primitive" framing is aspirational.

| New op | Fused impl | Fallback |
|---|---|---|
| `MoeGateSideProj` (macro) | `fused_qkvza_hfq4g256` (router + shared_expert_gate + shared.gate + shared.up, 4-way) | 4x `weight_gemv` |
| `Softmax` | `gpu.softmax_f32` | — (shared math) |
| `TopKRenorm{k=8 only}` | `moe_topk_renorm_k8` (GPU, hipGraph-capture-safe) | CPU download + `select_nth_unstable` + renorm |
| `SharedExpertDown` (macro) | `gemv_hfq4g256_residual_sigmoid_scaled_gpu` (silu·mul·rotate + down + sigmoid + residual-add fused) | sigmoid + silu_mul + weight_gemv + scaled_add |
| `IndexedGateUp` | `gemv_{hfq4g256,hfq6g256,paro_q4g128}_moe_gate_up_k8_indexed` (dtype-resolved) | per-expert loop |
| `SiluMulRotate` | `fused_silu_mul_{rotate_mq,givens_rotate}` (batched) | silu_mul + rotate (2 launches) |
| `IndexedDownExpanded` | `gemv_*_moe_down_k8_indexed_batched_expanded` | per-expert loop |
| `MoeCombine{k=8 only}` | `moe_down_combine_k8_batched` (atomic-free, deterministic) | — |

**The fused kernel family is k=8-hardcoded** (adversarial review, finding #2):
`moe_topk_renorm_k8.hip` has `#define K_TOP 8`; `moe_down_combine_k8_batched.hip`
and the indexed gate_up/down kernels are statically k=8. The parameterized `{k}`
notation is therefore misleading — only `{8}` has fused coverage today.

deepseek4's routing divergence is consequently **not** "a single substituted op."
It needs (a) a `SigmoidBiasGroupTopK` routing op AND (b) a **k=6 port of the
entire indexed-MoE kernel family** (topk_renorm, combine, gate_up, down). Until
that port lands, deepseek4 falls to the discrete fallback on every routed op.
The vocabulary leaves a *named slot* for deepseek4; it does not make the
implementation cheap. Out of scope here.

## qwen35 decode op-list

A faithful linearization of `moe_ffn_decode_impl` (decode path):

```
[ MoeGateSideProj, Softmax, TopKRenorm{8},
  SharedExpertDown,
  IndexedGateUp, SiluMulRotate, IndexedDownExpanded, MoeCombine ]
```

The shared-expert branch is **flattened in-line** rather than modeled as a
parallel sub-pipeline. Linearizing it is safe not because the launches are
"independent" (the weak argument) but because every cross-op data dependency is
buffer-mediated and already strictly ordered in the source: `x_rot_local`
(`s.x_rot_local`) and the shared-expert-down rotation scratch
(`gpu.scratch.mq_x_rot`) are **distinct buffers** (qwen35.rs:4674-4717 vs
4835-4848), so there is no aliasing hazard the op-list could reorder. (Approved
sanity-check (a).)

## Resolution pre-pass (adversarial review, finding #1)

MoE fused-eligibility is a **coupled lattice**, not a per-op flag. The op-list
gives execution *order*; a single pre-pass over `MoeParams` resolves *which
variant each op runs* before the executor iterates. The couplings, lifted
verbatim from `moe_ffn_decode_impl` (qwen35.rs:4598-4671):

- `gate_side_mq4` — all four gate-side weights MQ4G256 → `MoeGateSideProj` fused.
- `routed_dtype_indexable_{mq4,mq6,paro}` — gate_up AND down dtypes must *agree*;
  `IndexedGateUp` and `IndexedDownExpanded` cannot pick fused independently.
- `use_gpu_topk = (k == 8) && routed_dtype_indexable` — **`TopKRenorm`'s GPU
  fused path is gated by the routed-expert dtype**, i.e. one op's variant depends
  on another op's weights. This is the coupling a naive per-op table hides.
- `needs_x_rot_local` couples `MoeGateSideProj` and `IndexedGateUp`; the rotation
  *target weight* differs (`&ffn.router` vs `&ffn.experts[0].gate_up`,
  4708-4712) based on which fires.

Design consequence: `MoeFamily` computes a `MoeResolution` struct (the lattice
above) once, stamps each op's chosen variant into it, then `execute_pipeline`
consumes the stamped plan. This pre-pass IS the routing-config logic — it does
not vanish, it is *relocated* into one typed, testable place instead of being
inlined in the model. That is the honest version of "dissolves the config-match":
the match becomes one resolution struct, not zero logic.

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
  `LinearParams`. Migration surface: one external construction site
  (`crates/hipfire-dispatch/src/families/gemv.rs:274`, wrap in `Linear(..)`),
  plus **two** call sites that take the type by signature — `execute_pipeline`
  AND `dispatch_fused` (the latter is called *directly* from gemv.rs:278,
  bypassing `execute_pipeline`, so it needs the enum too — finding #4).
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
  FP32 non-determinism that diverges under hipGraph capture).
- **Equivalence invariant, corrected (finding #3).** Fused and fallback variants
  must produce identical *observable downstream state* — NOT identical state in
  every scratch buffer. `SharedExpertDown` is the case in point: its fused impl
  applies sigmoid internally and leaves `scalar_buf` holding the raw logit, while
  its fallback mutates `scalar_buf` in place (sigmoid eager, qwen35.rs:4859).
  Both yield the same `x_residual`; they differ in `scalar_buf`. This is safe
  only because nothing reads `scalar_buf` after `SharedExpertDown`. The pre-pass
  must therefore track buffer liveness, or the invariant must be stated as
  "equivalent on all *live* downstream buffers" — the naive "byte-identical
  scratch" version is false.

## Phasing

### Phase 1 — reframe (this item, #2)

Scope: **decode path only** (`moe_ffn_decode_impl`).

1. Rename `PipelineParams` struct -> `LinearParams`; introduce the
   `PipelineParams` enum. Update the external caller (gemv.rs:274) + both
   signature sites (`execute_pipeline`, `dispatch_fused`).
2. Add the MoE `PipelineOp` variants.
3. Grow `MoeParams` to carry the MoE scratch refs.
4. Implement the **`MoeResolution` pre-pass** (the eligibility lattice) and the
   MoE arm of `execute_pipeline`, consuming the stamped plan; per-op dtype
   resolution; MoE `find_fused` entries.
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
- No deepseek4 routing implementation. The vocabulary reserves a named slot
  (`SigmoidBiasGroupTopK`), but real deepseek4 support requires a **k=6 port of
  the entire indexed-MoE kernel family** (the current fused kernels are
  k=8-hardcoded — finding #2), which is a separate, non-trivial later item.
- No change to routing math.
- **Not in Phase 1:** the batched-prefill path
  (`forward_prefill_batch_with_pbs`) and the PARO prefill echo bug (#1) — kept
  separate so byte-parity stays tractable. (#1 is a distinct Layer-3 item.)

## Open risks

- **The eligibility lattice is the real complexity, not the op-list.** Finding #1
  showed fused-vs-fallback is cross-op coupled (`use_gpu_topk` gated by routed
  dtype, etc.). The `MoeResolution` pre-pass concentrates this; the risk is that
  it becomes a second home for the same combinatorial logic the model has today.
  Mitigation: it is one typed struct with unit tests, not inlined control flow —
  but it must be reviewed as the load-bearing piece, not an afterthought.
- **Per-op dtype resolution** is where the executor stops being a dumb op-runner.
  If a future arch mixes dtypes in a way the current `routed_dtype_indexable_*`
  checks don't cover, the resolve logic needs extending. Acceptable: the existing
  code already encodes these constraints; the reframe relocates them.
- **Flattening the shared-expert branch** assumes it stays a sequential launch
  block. If a future fused kernel wants to co-schedule shared + routed experts,
  the linear op-list would need a parallel-sub-pipeline construct. Out of scope;
  revisit only if such a kernel is built.
