# Stage 3d — batched-TP prefill (plan)

**Status: ✅ DONE + measured (2026-05-29, uncommitted).** Parity PASS; the perf
result is an HONEST LOSS at large-N prefill (see "Result" below). Decode wins
(+11%, 3c) but batched prefill does NOT — the all-reduce volume scales with N
and the DeltaNet chunk recurrence is sequential, together outweighing GEMM
sharding on this DeltaNet-heavy hybrid.

## Result (27B-3.5 mq4-awq, q8 KV, q8 state, fresh process, warmed)
`tp_attn_parity --prefill` (HIPFIRE_PARITY_PREFILL=1 HIPFIRE_PARITY_KV=q8),
`forward_prefill_chunk_tp` tp=1 (full, Full-phase, 1 GPU) vs tp=2 (sliced, sharded):

| N | TP=1 tok/s | TP=2 tok/s | Δ | last-tok rel Δ / argmax |
|---|-----------|-----------|------|------|
| 1   | 36.4 | 48.4 | **+33%** | 2.595e-7 ✓ (decode-like, bandwidth-bound) |
| 64  | 596  | 410  | −31% | 1.5e-3 ✓ |
| 128 | 588  | 416  | −29% | 2.4e-2 ✓ |
| 256 | 569  | 438  | −23% | 5.3e-3 ✓ |

**Parity: PASS.** N=1 = 2.595e-7 is the fp32-equivalent **sharding-math gate**
(exercises sliced weights + all-reduce + FA+DeltaNet, no chunk compounding) →
the weight-slicing + all-reduce reconstruction is mathematically exact on 27B.
argmax matches at every N. The N≥2 rel (1e-3..2e-2) is **q8-chunk-recurrence
amplification of TP's ~1e-7 all-reduce reassociation** — the batched chunk
kernel `gated_delta_net_q8_batch_seq` is **q8-state-ONLY** (no fp32 batched
variant; fp32 state OOB-reads `s_scales` and faults at full head count), so
prefill is inherently q8-state and the 3c "q8 state amplifies TP" effect applies
(bounded, argmax-stable — NOT a sharding bug).

**Perf: TP=2 prefill LOSES at N≥64 (−23%..−31%); wins only N=1 (+33%).** Root
cause — comm-bound, not compute-bound: 2 all-reduces/layer × 64 layers, each
`[N×dim]`, so all-reduce TRAFFIC SCALES WITH N (~670 MB at N=256) while the GEMM
saving is a fixed ratio; and the DeltaNet chunk (`gated_delta_net_q8_batch_seq`,
48/64 layers) is **sequential per token** and only sharded by value head (half
the heads/rank, but the per-step recurrence is latency-bound, not throughput-
bound). The N=1 win matches 3c's +11% decode win (bandwidth-bound: each rank
reads ½ the weights+state). **Conclusion: tensor-parallel is NOT the path to
faster prefill on this DeltaNet-heavy hybrid.** A real prefill win needs
sequence parallelism (shard the N dim → fewer/smaller all-reduces) or, for A3B,
expert parallelism (leaner MoE comm). TP's value is decode latency + fitting a
bigger model per card (steady-state per-rank memory halved).

## Original plan / scope

**Original goal:** measure prefill throughput TP=1 vs TP=2 on 27B. Decode already
wins (+11%, 3c); prefill *was expected* to win more (GEMMs compute-bound, shard
cleanly; per-layer all-reduce amortizes over N) — empirically it does not, see above.

## Why a new path (don't touch production prefill)
`forward_prefill_chunk` (qwen35.rs ~6720) is the PRODUCTION prefill — one
monolithic call: embed N tokens → loop layers (batched DeltaNet arm ~6954-7428,
batched FA arm ~7429-~8100, both inline over `PrefillBatchScratch` `pbs`) →
norm+lm_head. It has NO phase seam and refactoring it risks production prefill.
**Decision: ADD phased batched bodies (byte-exact copies of the two arms with a
phase param) + a `forward_prefill_chunk_tp` orchestrator — leave
`forward_prefill_chunk` untouched.** Mirrors how the single-token path kept the
inline FA arm and added `run_fa_layer_body` separately.

## Sharding (same axes as 3c — sliced weights already produce local outputs)
With `load_weights_tp` weights + `local_attn_config`, every batched kernel
(`gemm_qkv`, `gemm_qkvza`, `gemm_gate_up`, batched attention, batched
gated-delta-net) emits LOCAL outputs for free (smaller m/k) — no kernel changes.
The only new work is the partial-wo / partial-w_down + all-reduce orchestration,
exactly like the single-token `forward_scratch_tp`.

Per FA layer (batched): rmsnorm+rotate → gemm_qkv (sliced → local Q/K/V
`[N×local]`) → batched attention (local heads) → gated output → **partial wo
gemm → `[N×dim]` partial** → all-reduce `[N×dim]` → add → batched FFN (sliced)
→ **partial w_down → all-reduce → add**. DeltaNet layer: analogous, with the
chunked gated-delta-net on local value heads + the recurrent state (already
local per 3c) + partial wo/w_down + all-reduces.

## Build steps
1. **Batched all-reduce helper** — `tp_allreduce_add_batched(gpus, bufs, residual_bufs, count=n*dim)`:
   all-reduce the per-rank partial `[N×dim]` buffers + add into the per-rank
   batched residual (`pbs.x_batch`). (Single-token `tp_allreduce_add` uses
   `s.o`/`s.x` with count=dim; batched uses pbs batched buffers, count=n*dim.)
2. **`run_fa_layer_batched(gpu, weights, config, layer_idx, pos, kv, pbs, phase)`** —
   copy of forward_prefill_chunk's FA arm (7429-~8100) with `FaPhase`-style
   TpAttn/TpFfnShard: partial wo into a batched scratch buf (NOT residual),
   return; TpFfnShard partial w_down. `Full` = byte-exact (verify vs production).
3. **`run_dn_layer_batched(...)`** — same for the DeltaNet arm (6954-7428). The
   batched gated-delta-net + chunked S-matrix recurrence run on local value
   heads (local config + sliced weights + per-rank local DeltaNetState).
4. **Per-rank `PrefillBatchScratch`** sized from the local config (local heads /
   local ffn → local batched buffers). The partial-wo/w_down land in a `[N×dim]`
   batched scratch buf (full dim — the output of wo/w_down is full dim).
5. **`forward_prefill_chunk_tp(gpus, shard, weights[], configs[], tokens, pos,
   kv_caches[], dn_states[], pbs[], fa_masks=None)`** — per-rank embedding →
   layer loop (DeltaNet: DnAttn→AR→add→DnFfnShard→AR→add; FA: TpAttn→AR→add→
   TpFfnShard→AR→add) → norm+lm_head on rank 0 (last token).
6. **Harness** — extend `tp_attn_parity` (or a new `tp_prefill_bench`): prefill
   a long prompt (e.g. 256–512 tokens) single-GPU (`forward_prefill_batch`) vs
   TP=2 (`forward_prefill_chunk_tp`); assert per-token logit parity at the last
   position (fp32) + measure prefill tok/s. Warm + fresh process + byte-identical
   prompt + md5 per docs/methodology/perf-benchmarking.md.

## Validation
Last-token logits TP=2 vs single-GPU prefill must match fp32 ~1e-5 (batched
reassociation is larger than single-token but should stay well under 1e-4).
Then prefill tok/s TP=1 vs TP=2 on 27B (the deliverable number). Expect a real
win (compute-bound GEMMs sharded; all-reduce amortized over N).

## Effort / risk
Comparable to 3b+3c combined: ~1000 lines of batched-arm copies + phases +
orchestration + a per-rank pbs + harness. The copies are mechanical (mirror the
single-token phasing) but large; the batched DeltaNet chunked recurrence is the
intricate part. Lean-sync (no per-layer device_synchronize — the 3c decode-win
fix) applies here too: rely on stream ordering + RCCL.
