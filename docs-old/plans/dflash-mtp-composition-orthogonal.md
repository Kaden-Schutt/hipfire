# Orthogonal DFlash + MTP composition (post-2026-05-15 K=5 result)

> **Status:** design doc, no implementation yet. Picks up where the v1
> composition attempts (Tasks 11 / 11b in `[[mtp-native-head-deferred-2026-05-15]]`)
> were falsified.
>
> **New empirical floor:** compressed-serial K=5 standalone = **49 tok/s**
> on 27B-3.5 LRU canonical (gfx1100). DFlash standalone (this branch) =
> **161.7 tok/s**. The interesting question: can we compose at >+20% over
> DFlash, given the K=5 result reframes the lever space.

## Why v1 composition failed (recap)

**Linear-chain (Task 11 ed1162de):** MTP candidates appended after dflash's
B=16 chain. Accept-prefix stops at first dflash miss → MTP candidates
beyond miss point waste their work. Measured 108.9 tok/s = -32.7% vs
DFlash baseline. Root cause: dflash's accept rate is high enough that
MTP has nothing useful to add in most cycles.

**Per-slot tree (Task 11b aec64dbb):** MTP K=2 children per accepted
dflash slot. Per-slot independent attention τ_mtp = 0.02-0.17 (zero
acceptance), AND τ_dflash collapsed from 8 to 1-2 due to FA tree-attn
sibling-position overload. -81% vs baseline. Root cause: tree-attn
isn't built for 30+ same-position siblings.

## What "orthogonal" can mean

Five mathematically-distinct compositions, each with different
per-cycle math:

| # | Pattern | Where MTP enters |
|---|---------|------------------|
| A | **Time-orthogonal** | Alternate dflash and MTP cycles |
| B | **Tail-amplifier** | After dflash cycle's bonus token, MTP K=1 looks ahead |
| C | **Phase-orthogonal** | DFlash for prompt phase, MTP for generation phase |
| D | **Drafter-replacement** | MTP head becomes dflash's drafter slot |
| E | **Speculative pipelining** | MTP runs async while trunk verifies dflash |

### A. Time-orthogonal (alternating cycles)

Run cycle N as dflash (B=16 → ~11 tokens), cycle N+1 as MTP K=5
(~6 tokens), repeat.

- Per-cycle: DFlash ~11 tok / 70 ms; MTP ~6 tok / 122 ms (49 tok/s × 6)
- Combined throughput: (11 + 6) / (0.070 + 0.122) = 17 / 0.192 = **88 tok/s**
- vs DFlash standalone 161: **-45%** ❌

Worse because MTP cycles are slower-throughput than DFlash. Skip.

### B. Tail-amplifier (MTP K=1 after dflash bonus)

Each dflash cycle commits N+1 tokens (N accepted MTP candidates + 1
bonus). After the bonus, run MTP K=1 to propose the next token (N+2);
verify with one trunk forward; commit if MTP matches.

- Per-cycle: dflash ~70 ms + MTP forward ~16 ms + trunk verify ~22 ms = ~108 ms
- Tokens: dflash gives N+1 ≈ 11; MTP adds 0.8 (if 80% accept) = ~11.8
- tok/s: 11.8 / 0.108 = **109 tok/s** vs 161 = **-32%** ❌

The extra trunk verify kills it. The bonus token is free; adding ANY
verified token is expensive in absolute terms.

### C. Phase-orthogonal (prompt vs generation)

DFlash optimizes prompt-conditioned generation (high τ on code/structured
text). MTP optimizes long-form generation where dflash drafter drifts
off-distribution (low τ). Switch modes at a signal — say, accept_count
< 3 for K consecutive cycles → switch to MTP.

- Best case: each mode runs at its own ceiling without paying the other
- Worst case: switching cost + KV reset
- Need empirical signal — when does dflash actually degrade enough to
  matter? Per CLAUDE.md the LRU bench gives τ_dflash ≈ 10. No degraded
  regime in our canonical benches.

Marginal. Defer until we have a bench that shows dflash degradation on
a real workload.

### D. Drafter-replacement (MTP IS the drafter)

Replace the dflash drafter with the MTP head. dflash's tree-verify
infrastructure is reused; the drafter slot proposes via MTP forward
chain instead of via the dflash drafter model.

- Per-cycle: MTP K=5 (250 ms) + dflash tree-verify (much heavier than
  K=5's K+1=6 verify because tree-verify uses B=16 slots).
- Wait — dflash's tree-verify cost is ~30-40 ms (n_verify=16 batched).
  MTP K=5 standalone uses n_verify=6.
- If we feed MTP's K=5 candidates into dflash tree-verify (n_verify=6
  flat, no tree branching since MTP is linear): we get the same K=5
  serial path with no win.
- If we make MTP propose B=16 candidates somehow (e.g., compressed
  K=16 or per-slot tree expansion of K=5 into B=16): we get the lossy
  chain regression we already measured.

Dead end without a different MTP topology than what we have.

### E. Speculative pipelining (async MTP)

While trunk verifies dflash's batch, the MTP head pre-computes its
K=5 chain off the same `last_committed`. When trunk verify finishes:

- If MTP's first proposal matches trunk's bonus: commit MTP's chain
  too (lossless via second verify pass on MTP candidates)
- If not: discard MTP work (wasted), proceed with dflash bonus only

Cost: MTP runs in parallel with trunk verify (different streams). On
gfx1100 with `gpu.active_stream` already used, this requires a second
HIP stream + careful memory management.

- Per-cycle wall (best case): max(dflash_verify=70ms, MTP_K5=250ms) +
  MTP_verify(if hit) ≈ 250 + 30 = 280 ms
- Tokens (if MTP all accept): 11 + 5 = 16
- tok/s: 16 / 0.280 = 57 tok/s ❌

The MTP chain at 250 ms is too slow even if we hide it under dflash.
Only viable if MTP K-step latency drops to ≤70 ms. Compressed K=2 might
fit (~60 ms based on K=2 = 37 tok/s = 27 ms/cycle ... wait that's per
token, cycle is τ_K2=2.77 / 37 = 75 ms — too slow still).

## What to actually try next

After the trained sidecar lands (hopefully ~60 tok/s standalone at K=5,
gate cleared), composition is moot for STANDALONE perf because MTP-K=5-
trained alone matches/beats AR by a clear margin on its own.

**The real composition lever is hipfire's untouched async/batching
slack:** the trunk-verify path uses `forward_prefill_batch_with_pbs`
which already amortizes well at n=6 (MTP K=5). DFlash uses the same
path at n=17 (B=16+1). Both are independent code paths; running
trunk-verify continuously across BOTH paths' candidates in a single
batched forward at n=22 might save one launch round.

That's not really "orthogonal stacking" though — it's a verify-batching
optimization that benefits both modes. Worth investigating after K=5
trained sidecar lands.

## Recommendation

Don't pursue any of A-E preemptively. Sequence:

1. **Ship trained sidecar (in flight)** → re-bench K=5 → if 60+ tok/s,
   call standalone done. MTP becomes a deployable mode.
2. **Bench DFlash + trained sidecar** by replacing DFlash drafter file
   with the trained-MTP-as-drafter via a thin shim — if the trained head
   has higher τ than the existing DFlash drafter chain on prose/non-code
   workloads, that's a real composition win on those regimes.
3. **If neither (1) nor (2) crosses gate** for the deployment workload,
   revisit B (tail-amplifier) — the trunk-verify cost projection assumes
   ~22 ms but if the trained sidecar bumps τ enough, the amortization
   math shifts.

Any further composition design WITHOUT a trained-sidecar baseline is
speculation on falsifiable numbers.
