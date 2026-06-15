# Ship 6 — Substrate Expert-Parallel (EP) for big MoE on hiptrx

**Goal:** run the two 80 GB MoE models — **MiniMax-M2** (arch 10) and
**DeepSeek-V4-Flash** (arch 9) — across **hiptrx (4× gfx1201, 32 GB each)**.
Neither fits one 32 GB card; both are MoE so their VRAM is *expert-weight
dominated* → sharding experts (EP) is what makes them fit.

This is an **extension of Ship 6** (the forward-as-pipeline lowering), not a new
ship. It builds directly on the fact that minimax + deepseek4 decode are
**already lowered** onto the super-op pipeline (the `Moe` super-op is the single
hook EP needs).

---

## The load-bearing decision: ALL-REDUCE EP (not dispatch/all-to-all EP)

Two ways to do expert parallelism:

| | **All-reduce EP** (chosen) | Dispatch EP (deferred) |
|---|---|---|
| Routing | **replicated** on every rank (all ranks compute the same top-k) | replicated top-k, then route tokens to owning rank |
| Expert compute | each rank runs its **owned** experts; non-owned slots read a **shared zero buffer** → contribute 0 | each rank runs only the tokens routed to it |
| Comms | **1 dense `all_reduce_sum_f32` of `[dim]`** per MoE layer | all-to-all dispatch + all-to-all combine |
| Determinism | **byte-deterministic** (fixed-order dense sum) | harder (variable token counts per rank) |
| Compute waste | each rank evaluates all *k* selected slots, most zeroed (decode: *k*=6–8 tiny GEMVs → negligible) | none |
| Complexity | **low** — proven in the prototype (Stage 3e, qwen3.5-A3B, validated TP=2≡TP=1) | high — new all-to-all kernels + token bookkeeping |
| VRAM | experts/N per card (non-owned freed; shared zero buffer ~1 expert) | same |

For the **decode** path (memory-bound, 1 token, tiny expert GEMVs) the
all-reduce design's compute waste is irrelevant and it is dramatically simpler
and already proven. Dispatch EP is the eventual perf play for prefill/throughput
— **deferred**. This dissolves Agent C's "needs all-to-all" wrinkles: minimax's
sigmoid+bias and deepseek4's hash-routing run **replicated**, no dispatch.

### Prefill staging + performance target (relaxed 2026-06-07)
Two phases: **(E6a) sequential token-by-token prefill first** for plumbing
/ correctness (run the EP decode path over the prompt to populate KV on all
ranks), then **(E6b) WMMA batched prefill EP immediately after** sequential is
validated. WMMA batched is required (token-by-token is NOT the end state) but
the **≥ single-card bar is a target, not a hard gate** — replicated attention
may keep EP prefill from beating single-card, which is acceptable. The
arithmetic for when batched EP *does* beat single-card (experts genuinely
sharded /4):

- EP prefill wall-clock ≈ `full_attn (replicated, runs in parallel → same
  wall-clock as 1 card) + experts/4 + small all-reduce`.
- Single-card ≈ `full_attn + full_experts`.
- EP < single-card **iff** experts are skip-computed /4 (the all-reduce is
  cheap). The decode **zero-dummy** trick (non-owned experts → shared zero
  buffer, GEMM still runs on zeros) keeps per-card expert compute = *full* →
  would FAIL the bar. **Prefill EP MUST skip non-owned experts** (grouped
  compute over owned experts only). Decode may keep zero-dummy (1 token, cheap);
  prefill may not.

Falsifiability:
- **qwen3.6-A3B** (fits 1 card): direct A/B — EP-prefill tok/s vs single-card
  batched prefill, byte-identical prompt. Must be ≥.
- **MiniMax / DeepSeek-V4** (don't fit 1 card → no single-card baseline):
  reference the **hipx gfx1151** batched-prefill numbers; hiptrx 4× gfx1201
  dGPU batched EP should **beat** the Strix Halo iGPU (higher compute + BW).

### Replicated attention + KV (v1)
Attention is replicated, so it is NOT parallelized — but it runs concurrently on
every rank, so its wall-clock contribution equals single-card (no slowdown, no
speedup). The EP win is entirely in the /4 expert sharding. If a future prompt
regime is so attention-bound that EP can't clear the bar, that is the signal to
pull attention-sharding (the dense-TP FaPhase work) back into scope — measure
first, don't pre-build it.

Attention/dense/norm/recurrent/conv super-ops run **replicated** on every rank
(full weights, full KV, identical input → bit-identical output). This **skips
the entire dense-TP attention-sharding effort** (FaPhase seam, wo col-gather,
AWQ replicate-vs-slice, DeltaNet head sharding — Stages 3a/3b/3c/3f of
`ship6-tp-port.md`, the "HIGHEST risk" work). Cost: KV is replicated N× (no KV
savings) and attention compute is replicated N× (wasteful but correct; fine for
decode). Sharding attention/KV is a later perf/long-context follow-on.

---

## Why it's a thin wrapper, not a rewrite

`run_layer_program` (`crates/hipfire-dispatch/src/pipeline/superop.rs:296`)
dispatches each super-op to `ForwardBindings`. **EP only diverges at the `Moe`
op.** Each arch's `run_moe` already implements its routing correctly
single-GPU:
- qwen35 `Qwen35Bindings::run_moe` (qwen35.rs:12546) → `run_moe_decode`
  (pipeline/mod.rs:207): softmax top-k, shared + routed both accumulate into
  `x_residual`.
- minimax `MinimaxBindings::run_moe` (minimax/forward.rs:664) →
  `minimax_moe_block`: sigmoid+bias top-k, routed combine into `state.h`,
  **no shared expert**.
- deepseek4 `Deepseek4Bindings::run_moe` (deepseek4/forward.rs:1930) →
  `ds4_moe_block`: shared expert seeds a **separate `ffn_out`**, routed
  (score-routed L≥`num_hash_layers`, or hash-routed L<that) accumulate into
  `ffn_out`, then `hc_ffn_mix` into the residual.

EP = (1) **load-time**: keep only `shard.owns_expert(rank,e)` experts, point
non-owned expert pointers at a shared zero buffer (prototype `shard_moe_experts`
generalizes — it just rewrites the `[2·n_exp]` device pointer table); (2)
**forward**: redirect the routed combine into a **zeroed `[dim]` partial**
(not the residual), `all_reduce_sum_f32` the partial, add it into the residual
once; (3) **shared expert** (if any) computed on **rank 0 only** (`skip_shared`
on rank>0) so it isn't summed N×; attention residual stays in the residual
buffer (replicated, never in the all-reduced partial → no double-count).

---

## Parallelism matrix (TP / PP / EP) per model

VRAM measured by tensor-byte summing on hipx; EP fit = experts/4 (sharded) +
shared + non-expert + KV (all replicated) per card, 4× gfx1201 32 GB.

| Model | On-disk | MoE shape | Routing | Per-card under EP/4 | Fits 4×32? | Needs | Notes |
|---|---|---|---|---|---|---|---|
| **qwen3.6-35B-A3B** (arch 6) | 17.4 GB | 256 exp, top-8, ~0 shared (tiny gate) | softmax + norm_topk | ~7 GB | n/a (fits **1 card**) | EP **optional** | Prototype-proven EP arch. Use as the **plumbing validation** (TP=2 on 2 cards, reference exists). |
| **MiniMax-M2** (arch 10) | 80.3 GB | 256 exp, top-8, **0 shared** | **sigmoid + per-expert bias[256]** | experts 76.3/4≈19 + non-exp 4.0 + KV ~4.4 ≈ **~28 GB** | **YES** | EP **required** | Cleanest EP shape (no shared → no skip_shared). Heaviest KV (62 full-attn layers, ~4.4 GB@32k, replicated). |
| **DeepSeek-V4-Flash** (arch 9) | 80.3 GB | 256 exp, top-6, **1 shared** | sqrtsoftplus + noaux bias; **L0-2 hash-routed** | experts 72.6/4≈18 + shared 1.1 + MLA/non-exp 6.6 ≈ **~26 GB** | **YES** | EP **required** | Hash layers trivial under all-reduce EP (replicated lookup). Shared+MLA replicated. MLA+SWA-128 → small KV. `routed_scaling_factor`. |

**Strategy summary:**
- **EP** = the primary lever for the two 80 GB models (expert-dominated VRAM). All-reduce variant.
- **PP** (layer split) already exists but is **qwen3.5-only** (daemon allowlist arch 5/6; minimax hard-rejected at daemon.rs:3786). An alternative/complement to EP (e.g. EP-within + PP-across) and the fallback if all-reduce bandwidth bites; generalizing PP = same "lower onto substrate" pattern (assign layer-bands to devices + boundary-copy between super-ops). Lower priority than EP for the goal.
- **TP (dense weight shard)** = the qwen35 `ship6-tp-port.md` Stages 3a–3f (FaPhase, weight slicing). **Not needed to fit these models.** Deferred to a perf/latency track.

---

## Concrete substrate-EP change list

Builds on Stage 1+2 (DONE, committed cf6ad952): `tp_shard.rs` (ExpertAssign /
owns_expert / experts_on_rank), `rccl.rs`, `Gpus::{init_tp, ensure_rccl,
all_reduce_sum_f32}`, `Stream::raw_ptr`, `config.tp_use_rccl`.

| # | Change | File(s) | Effort |
|---|--------|---------|--------|
| E1 | Multi-rank EP executor: `run_layer_program_ep(ranks, gpus, program, ...)` — step all ranks through each op; at `Moe` call each rank's EP partial, then `all_reduce_sum_f32`, then add-into-residual; all other ops run replicated per rank. `RankCtx { bindings, routed_partial, ctx }`. | `crates/hipfire-dispatch/src/pipeline/superop.rs` (+ runtime glue) | M |
| E2 | Per-rank zeroed routed-partial scratch (`[dim]` decode; `[N×dim]` later for batched). | per-arch State/Scratch | S |
| E3 | EP `run_moe` per arch: redirect routed combine → partial; `skip_shared` on rank>0; (deepseek4) skip all-reduce on hash-only-but-shared-only? no — hash layers still route experts, keep all-reduce; only pure-shared layers skip. | qwen35 `run_moe_decode`(+`MoeParams.routed_out`), minimax `minimax_moe_block`, deepseek4 `ds4_moe_block`/`ffn_routed` | M each |
| E4 | Load-time expert sharding (generalize prototype `shard_moe_experts`): keep owned experts, zero-buffer the rest, rebuild pointer tables, free non-owned (+AWQ sidecar). | runtime + per-arch weight load | M |
| E5 | Replicated full-model load across N ranks (`init_tp` + per-rank `load_weights` then E4 shard). | runtime weight-load path | M |
| E6a | **Sequential prefill EP (first, for plumbing/correctness).** Populate KV on all ranks by running the EP **decode** path token-by-token over the prompt. Slow, but proves the whole EP pipeline end-to-end (decode + KV + all-reduce) and unblocks a full working model on hiptrx. Validate, then E6b. | runtime driver | S |
| E6b | **WMMA batched prefill EP (immediately after E6a is validated).** Make the real batched WMMA/MMQ path (`forward_prefill_chunk` / `prefill_moe_ffn_body_batched`) EP-aware: each rank computes **only its owned experts** (genuine expert-skip, NOT decode's zero-dummy), routed combine → `[N×dim]` partial, `tp_allreduce_add_batched`. Reference: prototype `forward_prefill_chunk_tp` (Stage 3d). Perf target ≥ single-card where achievable (qwen3.6-A3B A/B); MM/DS vs hipx gfx1151. **Note (2026-06-07): the strict ≥single-card bar is relaxed — replicated attention may keep EP prefill from beating single-card; WMMA batched is still required (not token-by-token) but is the perf track, not a hard gate.** | runtime driver + per-arch batched MoE | L |
| E7 | Parity gate: TP=N decode ≡ TP=1 (argmax + max-abs-diff, gold 7.5e-7) on hiptrx devices 0+1; coherence-gate on the EP output. | examples / gate scripts | M |

**Explicitly NOT needed (vs dense-TP plan):** FaPhase seam, `run_fa_layer_body`
re-seam, wo col-gather, AWQ replicate-vs-slice, attention/DeltaNet weight
slicing, all-to-all dispatch kernels.

---

## Sequencing (de-risk plumbing → generalize → complexity)

1. **qwen3.6-A3B EP** — port the prototype's all-reduce EP onto the *current*
   substrate. Validates E1/E2/E4/E5/E7 (executor + RCCL + load-shard + parity)
   on the arch that **has a reference** and **fits one card** (so TP=2 is pure
   validation). The one new bit is redirecting current `run_moe_decode`'s
   shared+routed-in-`x_residual` to a partial (E3 qwen35).
2. **MiniMax EP** — cleanest MoE shape (no shared expert). First *goal* model
   on hiptrx. Generalizes E3 to a non-prototype arch (sigmoid+bias routing
   already in `minimax_moe_block`).
3. **DeepSeek-V4 EP** — most moving parts (shared + hash + score + MLA +
   `routed_scaling_factor`), but hash is trivial under all-reduce EP and routed
   is already isolated in `ffn_out`. Second goal model.

Each step gates on TP=N≡TP=1 parity + coherence on hiptrx 0+1 before the next.

**Validation boxes:** hiptrx devices 0+1 (gfx1201) for TP=2 parity; scale to
0,1,2,3 for the full 4-way fit of the 80 GB models. hipx cannot do TP=2
(single usable big-VRAM device). k9lin = single-GPU (TP=1 reference only).

---

## VALIDATED — qwen3.6-A3B EP plumbing (E1/E2/E3/E4/E5/E7), 2026-06-07

Step 1 of the sequencing above is **DONE and validated on RDNA4**. The generic
EP substrate (E1 `dispatch_super_op` + `run_layer_program_ep`, E2 routed-partial
param, E3 `MoeParams.routed_out`/`skip_shared` redirect, E4 `shard_moe_experts`,
E5 `forward_ep` N-rank driver + `shard_all_moe_layers`) is wired end-to-end and
proven byte-/argmax-exact.

**Harness:** `crates/hipfire-runtime/examples/ep_decode_parity.rs`
(`forward_ep` greedy decode; prints gen token-ids + FNV-1a hash; at tp=1 also
runs production `forward_scratch` on the unsharded rank-0 replica as an
in-process anchor).

**Results** (`qwen3.6-35b-a3b.mq4`, prompt "The capital of France is", 16 steps):

| box / arch | run | experts/rank | all-reduce | gen FNV | output |
|---|---|---|---|---|---|
| k9lin gfx1100 | tp=1 (anchor) | all (256) | identity | `0x6eb6f119212f3f68` | "…is Paris." (≡ production) |
| hiptrx gfx1201 | tp=1 (anchor) | all (256) | identity | `0xdf98c087d3de9725` | "…is Paris." (≡ production) |
| hiptrx gfx1201 ×2 | **tp=2** | 128 each (e%2==r), rest freed | **RCCL** | `0xdf98c087d3de9725` | "…is Paris." |

- **tp=1 anchor (both boxes):** EP argmax stream == production `forward_scratch`,
  byte-identical FNV. The EP machinery reproduces production on one rank.
- **tp=2 == tp=1 (hiptrx):** identical FNV with experts genuinely sharded across
  two gfx1201 (each rank frees its non-owned half) and summed via RCCL
  all-reduce (`peer_access_enabled=true`). The all-reduce-EP design (replicated
  attention/KV/DeltaNet, all-reduce only at MoE) is argmax-exact on RDNA4.

(FNV differs k9lin vs hiptrx — different arch/kernels — but tp=1≡tp=2 *within*
hiptrx is the sharding proof; tp=1≡production *within each box* is the
correctness anchor.)

**Next:** E6a sequential prefill (works already — `ep_decode_parity` prefills
token-by-token via `forward_ep`) → E6b WMMA batched prefill (genuine
expert-skip, not zero-dummy). Then MiniMax EP (step 2), DeepSeek-V4 EP (step 3).

---

## VALIDATED — E6b WMMA batched prefill EP (qwen3.6-A3B), 2026-06-07

Step "E6b" done: a genuine WMMA/grouped-GEMM batched prefill EP path
(`forward_prefill_batch_ep`), not token-by-token. Driven layer-granularly via
single-layer `forward_prefill_chunk` bands so each MoE layer gets its per-layer
all-reduce; routed combine → zeroed `[n×dim]` partial (owned experts only, Path
0/1/2 redirect via `MoePrefillParams.routed_out`), shared expert stays in
`x_batch` replicated, then `all_reduce_sum_f32([n×dim])` + add into each rank's
residual. `HIPFIRE_EP_PREFILL=batched` in `ep_decode_parity`.

**Results** (`qwen3.6-35b-a3b.mq4`, "The capital of France is", 16 steps):

| box / arch | run | gen FNV |
|---|---|---|
| k9lin gfx1100 | tp=1 sequential / batched | `0x6eb6f119212f3f68` (both, == production) |
| hiptrx gfx1201 ×2 | tp=2 sequential | `0xdf98c087d3de9725` |
| hiptrx gfx1201 ×2 | **tp=2 batched-WMMA** | `0xdf98c087d3de9725` (== sequential) |

Batched-WMMA prefill is argmax-identical to sequential at tp=2 → the prefill EP
path is correct with experts genuinely sharded across two gfx1201 + per-layer
RCCL all-reduce.

**BUG FOUND + FIXED (the slow part):** sharing ONE `[max_batch·dim]` routed
partial between the decode executor and prefill made decode's all-reduce a
`count=dim` in-place RCCL reduction over a `[max_batch·dim]` buffer (count <
buffer). On tp≥2 that **page-faults** (gfxhub ring fault, confirmed in dmesg);
tp=1 was immune because the `n==1` all-reduce short-circuits RCCL. Fix: separate
partials — decode `[dim]` (count==buffer, the validated decode config), prefill
`[max_batch·dim]` (count = n·dim == buffer). **Lesson: keep RCCL in-place
all-reduce count == allocated buffer size; a count<buffer in-place reduce faults
on multi-rank.** A faulting run can transiently stall the gfx12 ring (looks like
a hang); it self-recovers once the process is killed (no reset needed).

**Remaining for prefill EP (perf, not correctness):** v1 uses zero-dummy experts
(per-card expert compute still full); genuine expert-skip (1/N compute) +
chunked prompts (>max_batch) are follow-on perf work. Next ship-blocking step:
MiniMax-M2 EP (#15) then DeepSeek-V4 EP (#16).

---

## PERF REALITY CHECK — E6b prefill is CORRECT but SLOW (2026-06-07)

Measured on hiptrx gfx1201, qwen3.6-35b-a3b.mq4, q8 KV:

| metric | normal single-card | EP tp=2 | ratio |
|---|---|---|---|
| decode | 89.5 tok/s | 82.8 tok/s | **93%** ✓ |
| batched prefill @ B=14 | 749 tok/s (18.7 ms) | 8.3 tok/s (1677 ms) | **~1.1%** ✗ |
| batched prefill @ B=256 | ~2750 tok/s (93 ms) | 164 tok/s @ B=288 (1756 ms) | **~6%** ✗ |
| coherence | — | correct recursive fib() ✓ | — |

**Decode EP is production-quality** (7% all-reduce overhead). **Prefill EP is
correct but ~90× too slow** and does NOT meet the ≥single-card bar.

Root cause: `forward_prefill_batch_ep` drives prefill by calling
`forward_prefill_chunk` **once per layer** (single-layer band) so it can insert
the per-MoE-layer all-reduce. That fragments the fused full-stack batched prefill
(one call, ~40 layers pipelined, 18.7 ms) into 40 separate single-layer chunk
calls + 40 RCCL all-reduces with no cross-layer pipelining → ~42 ms/layer vs
~0.47 ms/layer fused. The per-layer-chunk dispatch overhead dominates at the
small batch.

**Diagnostic:** EP prefill wall time is ~CONSTANT vs batch — 1677 ms @ B=14 vs
1756 ms @ B=288 (only +79 ms for +274 tokens). So it is ~42 ms/layer of FIXED
per-layer overhead (40 standalone chunk calls + 40 all-reduce barriers, no
cross-layer pipelining), not token compute. Throughput just scales B against a
~1.7 s floor (8.3 tok/s @ B=14 → 164 tok/s @ B=288). The fused EP prefill below
removes that floor.

**Fix (next perf step):** a FUSED EP prefill — one prefill pass per rank that
runs all layers pipelined, with the cross-rank all-reduce interleaved INSIDE the
layer loop after each MoE layer (make the prefill layer loop EP-aware, rather
than 40 standalone chunk calls). Plus genuine expert-skip (1/N compute) and
chunked prompts. Until then EP prefill is usable-but-slow — which still ENABLES
the goal models (MiniMax/DeepSeek don't fit one card, so there is no single-card
prefill to beat — a correct slow prefill is strictly better than "cannot run").

---

## FIXED — EP prefill perf via peer-direct all-reduce (bypass RCCL), 2026-06-07

Root cause of the ~17–90× prefill slowdown: **`ncclAllReduce` costs ~40 ms/call**
on hiptrx (2× gfx1201, PCIe, no xGMI) for the routed-partial sum — *regardless*
of NCCL_PROTO / NCCL_MAX_NCHANNELS / NCCL_BUFFSIZE / NCCL_SOCKET_IFNAME=lo /
NCCL_PROXY_DISABLE (all tested, no change). The data path is already P2P/direct;
the cost is inside RCCL's collective machinery. Confirmed by isolation:
`HIPFIRE_EP_SKIP_ALLREDUCE` → prefill 1662 ms → 32 ms; `HIPFIRE_EP_PREFILL_TIMING`
→ `all_reduce=1637 ms, chunk=49 ms`. NOT DPM (clocks pinned high = no change).

**Fix:** `HIPFIRE_EP_PEER_ALLREDUCE` (DEFAULT ON) — N-rank peer-direct all-reduce
bypassing RCCL: phase 1 P2P-copies every other rank's ORIGINAL routed partial
into a local temp (`boundary_copy` = `hipMemcpyPeerAsync`); barrier; phase 2 adds
the peer temps into the local partial. All-reads-before-writes ⇒ race-free.

| metric (B=288, tp=2 gfx1201) | RCCL | peer-direct | single-card |
|---|---|---|---|
| all-reduce host time | 1637 ms | **1.0 ms** | — |
| prefill | 164 tok/s | **2355 tok/s** | 2750 tok/s |
| TTFT | 1756 ms | **122 ms** | 93 ms |

EP prefill is now **86% of single-card** + correct (FNV `0xdf98c087d3de9725`) +
coherent. EP decode = 85 tok/s (95% of single-card; still uses RCCL — decode's
small per-token all-reduce is ~0.3 ms, fine).

**Scope / generalization (answers "does this apply to TP/PP?"):**
- It is NOT a UDS/socket fix — `NCCL_SOCKET_IFNAME=lo` (UDS-class loopback) did
  nothing; the socket is bootstrap-only, the per-collective cost is in RCCL.
- **TP** uses the same cross-rank residual all-reduce → would hit the identical
  RCCL slowness → should use peer-direct too. Clean home = move peer-direct into
  the shared `multi_gpu::all_reduce_sum_f32` so EP-decode + EP-prefill + TP all
  get it (FOLLOW-UP; needs Gpus-owned peer temps + DeviceBuffer add wrapper).
- **PP** is already immune — it uses `boundary_copy` (the same P2P primitive),
  not RCCL all-reduce.
- Open curiosity: why is RCCL fast for decode's 8 KB all-reduce but 40 ms for
  prefill's 32 KB? Unresolved; peer-direct sidesteps it. Worth a standalone RCCL
  size-sweep microbench if RCCL is ever needed (e.g. N>4 / cross-node).

---

## VALIDATED — MiniMax-M2 EP on hiptrx (#15), 2026-06-08

The 86 GB `minimax-m2.mq2lloyd` tier (does NOT fit one 32 GB card) runs COHERENT
across **4× gfx1201** via EP:

```
prompt: "The capital of France is"
gen (tp=4 EP): " Paris. The capital of Germany is Berlin. The capital of Italy
is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon..."
decode 51.7 tok/s  |  shard-load ~18 s/rank (~24 GB/card)  |  peer-direct all-reduce
```

Port (`hipfire-arch-minimax`, commit a651f373; mirrors qwen35 EP):
- `minimax_moe_block(.., routed_out)` — no shared expert, so the whole MoE output
  (incl the MQ2/MQ3-Lloyd `*_residual_scaled_indexed` down) redirects to the partial.
- `MinimaxBindings::run_moe_ep` + `ep_add_into_residual` (adds the all-reduced
  partial into `state.h`).
- **`MiniMaxWeights::load(.., Some((shard, rank)))` — shard-aware load.** KEY
  difference from qwen35: MiniMax packs all experts into ONE blob per projection,
  so load-then-free is impossible on 32 GB. Each rank reads all experts but
  uploads ONLY its owned experts into a compact blob; non-owned ptrs → a zeroed
  gate_up dummy (Lloyd codebook centroids are zero inline ⇒ 0 output — validated
  coherent, no per-rank weight mask needed).
- `forward::forward_ep` N-rank decode driver; `examples/ep_minimax.rs`.

Model transfer: `hiptrx` can `ssh hipx`, so the 86 GB file was rsync-pulled
directly from hipx local disk (no HF re-download).

**Remaining for MiniMax:** batched WMMA prefill EP (E6b-equivalent; prefill is
currently per-token sequential — fine for short prompts). Then DeepSeek-V4 EP (#16).
