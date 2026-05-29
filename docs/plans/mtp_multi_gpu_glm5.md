# MTP dual-GPU parallel decode: feasibility analysis

**Date:** 2026-05-26
**Context:** Can we speed up MTP spec-decode by running the MTP head on a
dedicated GPU while the trunk verifies on another?

## TL;DR

**Feasible but low ROI.** The MTP head forward is only ~12% of the per-cycle
wall clock (~9.5 ms of ~80 ms). Offloading it to a second GPU saves at most
9.5 ms/cycle = ~13% decode speedup. Pipeline-parallel (PP) on the trunk,
which already ships on `feat/multi-gpu-pp`, is the better multi-GPU lever
because it parallelizes the actual bottleneck (trunk verify + replay at ~70%
of cycle). The one scenario where MTP offload wins: an asymmetric rig where
GPU 1 is too small for half the trunk (~10 GB) but comfortably fits the MTP
head (~800 MB).

## Per-cycle anatomy (K=5, 27B MQ4, single GPU)

From `docs/plans/mtp-cycle-anatomy.md` and
`docs/plans/mtp-hiptrx-rocprof-2026-05-21.md`:

```
|<--------------- ~80 ms total cycle ---------------------------------->|
| 1ms | 6.5ms | 3ms |       30ms        |   15ms    | 0 | 15-20ms |0.5|
 snap  MTP     MTP    trunk verify       trunk       accept replay  prev
 chain  lm_head (6 tok, 64 layers) lm_head+argmax        (rollback
 (5 blocks)                                    +replay)
```

| Component | Time | % of cycle |
|---|---|---|
| Trunk batched verify forward (n=K+1, all layers) | ~30 ms | 37% |
| Trunk batched lm_head + argmax | ~15 ms | 19% |
| Trunk KV rollback + replay | ~15-20 ms | 20% |
| MTP chain forward (K=5 blocks) | ~6.5 ms | 8% |
| MTP lm_head + argmax | ~3 ms | 4% |
| DN snapshot + restore | ~5 ms | 6% |
| Misc (argmax D2H, bookkeeping) | ~3 ms | 4% |
| **Trunk total** | **~67 ms** | **~84%** |
| **MTP head total** | **~9.5 ms** | **~12%** |

Each extra MTP block costs only 1.3 ms. The trunk is the bottleneck.

## Proposed pipeline

```
Cycle N:
  GPU 0: [verify N][lm_head N][accept][replay N          ]
  GPU 1:                       ↓ prev_hidden [MTP N+1    ]
                                            ↓ candidates
Cycle N+1:
  GPU 0:                               [verify N+1][...][replay N+1]
  GPU 1:                                                      [MTP N+2]
```

After accept logic produces `last_committed` + `prev_hidden` (~45 ms into
the cycle), GPU 1 starts the MTP chain for the next cycle. This runs
during GPU 0's replay (~15-20 ms). The MTP chain (~9.5 ms) fits within
the replay window with margin, so GPU 0 never stalls waiting for
candidates.

### Data flow

1. GPU 0 → GPU 1: `prev_hidden` (dim × 4 = 20 KB) + `last_committed`
   (4 B). Transfered via `hipMemcpyPeerAsync`. Cost: < 0.01 ms.
2. GPU 1 runs K MTP block forwards + K lm_head dispatches (~9.5 ms).
3. GPU 1 → GPU 0: K candidate token IDs (K × 4 B). Cost: < 0.01 ms.
4. GPU 0 runs trunk verify on [last_committed, c₁, …, cₖ].

### Critical-path savings

The MTP chain (~9.5 ms) was previously on GPU 0's critical path (between
snapshot and verify). With the offload, it overlaps with the replay. The
replay is the longest post-accept GPU 0 operation (~15-20 ms) and the MTP
chain fits entirely within it.

**Savings: ~9.5 ms/cycle. At 80 ms/cycle → ~12% decode speedup.**

At K=5, τ≈3.8, single-GPU ~47 tok/s → dual-GPU projected ~54 tok/s.

### Why K scaling doesn't help

Increasing K to exploit GPU 1 more:

| K | MTP chain (GPU 1) | Trunk verify (GPU 0) | Cycle wall | Savings |
|---|---|---|---|---|
| 5 | ~9.5 ms | ~45 ms | ~80 ms | 9.5 ms (12%) |
| 10 | ~16 ms | ~90 ms | ~130 ms | 16 ms (12%) |

Trunk verify scales linearly with K (n=K+1 tokens, all 64 layers). The
MTP fraction stays ~12% regardless of K.

## GPU 1 resource requirements

GPU 1 holds only MTP head state — no trunk weights, no trunk KV cache:

| Component | Size | Notes |
|---|---|---|
| MTP head weights (1 transformer block) | ~300 MB | 1/64 of trunk |
| Trunk `token_embd` (step-0 embed lookup) | ~50 MB | MQ4 format |
| Compressed `lm_head_draft` sidecar | ~84 MB | 32K vocab slice |
| MTP KV cache (1 layer, asym3) | ~160 MB | max_seq=4096 |
| MTP head scratch (tmp, logits, etc.) | ~200 MB | |
| **Total** | **~800 MB** | |

This fits on any discrete GPU or even an APU iGPU with ≥ 2 GB VRAM.

The trunk (GPU 0) still needs its full allocation:
~20 GB for 27B MQ4 (weights + KV cache + scratch + DN state).

## Cross-GPU transfer costs

| Transfer | Size | PCIe Gen4 latency |
|---|---|---|
| GPU 0 → GPU 1 (prev_hidden + last_committed) | 20 KB | < 0.01 ms |
| GPU 1 → GPU 0 (K candidate token IDs) | K × 4 B | < 0.01 ms |

Negligible. Even on PCIe Gen3 x8 the transfers are sub-0.1 ms.

## Comparison: PP trunk split (existing infra)

hipfire already ships PP on `feat/multi-gpu-pp` (`multi_gpu.rs`,
`PrefillBandCtx`, `forward_scratch_multi`). Layers are sharded into
contiguous bands; the residual stream crosses device boundaries via
`hipMemcpyPeerAsync` (~20 KB per boundary).

| | MTP offload | PP trunk split |
|---|---|---|
| Cycle speedup | ~12% | ~50-78% projected |
| Infra needed | New pipeline sync, weight replication | Already shipped |
| GPU 1 VRAM | ~800 MB | ~half trunk (~10 GB for 27B) |
| PP+spec status | N/A (MTP only) | **Explicitly refused** at load time |
| Decode throughput (projected) | ~54 tok/s @ τ=3.8 | ~84 tok/s @ τ=3.8 |
| Asymmetric GPU support | **Yes** (weak GPU OK) | No (must hold half trunk) |

PP parallelizes the actual bottleneck (trunk verify + replay at ~70% of
cycle). Current PP decode runs at ~67% of single-GPU per-token throughput
due to sequential boundary copies; even with that overhead, PP dominates
the MTP-offload approach for raw throughput.

### Combined: PP + MTP offload

Both approaches compose:

```
GPU 0 (band 0, layers 0-31):  [verify band 0][replay band 0]
GPU 1 (band 1, layers 32-63):       [verify band 1][replay band 1]
GPU 2 (MTP head):            [MTP chain overlaps with replay on GPU 0/1]
```

Projected: PP=2 cuts verify+replay by ~40-50%; MTP offload saves an
additional 9.5 ms. Combined ~38 ms/cycle → ~95 tok/s @ τ=3.8.

Requires lifting the PP+spec refusal gate and the new 3-GPU pipeline
orchestration. Significant engineering effort for marginal gains beyond
PP alone.

## Implementation sketch

If pursued, the MTP offload requires:

1. **Weight replication**: Copy MTP head weights + trunk `token_embd` to
   GPU 1 during model load. ~350 MB one-time cost.

2. **MTP KV cache on GPU 1**: Allocate `Qwen35MtpHeadKvCache` on GPU 1
   instead of GPU 0. Already device-local by construction.

3. **Cross-GPU prev_hidden transfer**: After accept logic, memcpy
   `state.prev_hidden` from GPU 0 → GPU 1. 20 KB via
   `hipMemcpyPeerAsync`.

4. **Pipeline sync**: Two `hipEvent_t` barriers per cycle:
   - GPU 1 waits on GPU 0's accept-complete event before starting MTP
     chain.
   - GPU 0 waits on GPU 1's candidates-ready event before starting
     trunk verify.

5. **Candidate transfer**: GPU 1 → GPU 0 K×4B after MTP lm_head argmax.
   Could be `hipMemcpyPeerAsync` or even host-staged (tiny).

6. **Full-accept interaction**: When `advance == K+1`, replay is skipped.
   The MTP chain still starts from the verified prev_hidden. No special
   handling needed — the overlap window just shrinks.

## When MTP offload is the right call

The asymmetric-GPU case is the sweet spot:

- **GPU 0**: 24 GB discrete (7900 XTX, RX 9070, etc.) — full trunk
- **GPU 1**: Any GPU with ≥ 2 GB — MTP head only

This includes:
- APU iGPU (e.g., Strix Halo's gfx1151 integrated)
- Old/discrete GPU repurposed as MTP accelerator
- Laptop + eGPU setups where the eGPU runs trunk and the dGPU/iGPU runs
  MTP

In these configs, PP is impossible (GPU 1 can't hold half the trunk) but
MTP offload gives a free ~12% boost.

## Functional Parallelism via Async Speculation (Extension)

While the synchronous "Pipeline" proposed above yields ~12% ROI, a more aggressive **Async Speculation** strategy (Spec-on-Spec) can effectively hide 100% of MTP latency and potentially increase throughput by allowing deeper MTP chains without stalling the Trunk.

### The Async Overlap

Instead of waiting for GPU 0's accept logic to finish, GPU 1 can "blindly" speculate the next K tokens based on the *most likely* outcome of the current cycle.

```
Cycle N:
  GPU 0: [verify N    ][lm_head N][accept][replay N]
  GPU 1: [MTP N+1 (based on Draft N's last token) ]
```

1. **Cycle N start:** GPU 0 starts Trunk Verify for Draft N. Simultaneously, GPU 1 starts MTP Chain N+1, seeded by the *last predicted token* of Draft N and the `t_mtp_out` from GPU 1's own previous forward.
2. **Acceptance:** 
   - **Full Accept:** If GPU 0 accepts the entire Draft N, GPU 1's work was valid. GPU 0 proceeds to Verify N+1 immediately using the tokens GPU 1 just finished generating. **Result: 0 ms MTP stall.**
   - **Partial Accept/Reject:** GPU 0 sends a "Interrupt & Reset" signal to GPU 1. GPU 1 discards its current work and restarts MTP Chain N+1 using the correct `prev_hidden` and `bonus_token` from GPU 0. **Result: Falls back to Synchronous Pipeline (~80ms cycle).**

### Why MTP is uniquely suited for this

DFlash requires syncing a large, stateful KV cache with the Trunk. MTP's "native head" is nearly memoryless relative to the Trunk; it only requires:
- `t_mtp_out` (its own internal hidden state).
- `next_token` (embedding).

By decoupling the MTP GPU's forward from the Trunk GPU's verification, we can run **continuous speculation**. Even if GPU 1 only correctly predicts the first 2-3 tokens of a 5-token chain while "blind," the Trunk never waits for the drafting step to begin.

### Impact on τ (Speculative Efficiency)

Async Parallelism allows us to increase $K$ (draft length) "for free" on GPU 1.
- In a synchronous setup, $K=10$ adds ~16ms to the critical path.
- In an asynchronous setup, $K=10$ runs in the background. As long as MTP compute time < Trunk verify time (~30ms), the Trunk **never sees the MTP latency**.
- This enables deeper chains ($K=15+$) which can significantly raise τ on fluent text, whereas synchronous $K=15$ would be prohibitively slow.

### Summary: Corroboration & Extension

- **Synchronous Offload:** I corroborate the ~12% ROI estimate for a simple pipelined approach. The latency of the MTP head is too small relative to the Trunk to justify a synchronous cross-GPU wait.
- **Asymmetric Gains:** I extend the analysis to favor **Asymmetric Async Speculation**. This is the only path where MTP-offload becomes a "killer feature": using a secondary weak GPU to run a deep, continuous draft chain that stays 1 cycle ahead of the Trunk's validation.

## Conclusion (Final Assessment)

| Scenario | Recommendation | ROI (Cycle) | Complexity |
|---|---|---|---|
| **Two identical GPUs** | PP Trunk Split (Existing Infra) | **High (~50%)** | Low (Shipped) |
| **One Large + One Weak GPU** | **Asymmetric Async MTP** | **Medium (~20%+)** | High |
| **Three GPUs** | PP Trunk (2) + Async MTP (1) | **Extreme (~60%)** | Very High |

The MTP head is architecturally lightweight (1/64 of trunk layers, ~12%
of cycle wall). Parallelizing the lightweight component via simple pipelining
yields diminishing returns (~12%). However, **Async Speculation** allows
MTP to become a "latency-free" draft engine, enabling deep chains ($K=15$)
that would otherwise bottleneck a single GPU.

The real multi-GPU lever remains parallelizing the trunk itself via PP — 
lifting the PP+spec refusal gate is the highest priority for raw throughput.

