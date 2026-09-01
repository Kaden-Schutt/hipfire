# Ornith MQV2 V1/V2 — gfx1100 paired rocprof profile

**Date:** 2026-09-01 UTC  
**Lifecycle:** `historical`  
**Authority:** Dated, fixture-bound **Measured** evidence only.  
**Disposition:** **diagnostic evidence for PR #664**. Not a current product baseline, admission result, performance floor, SLA, or transferable claim. Newest file ≠ current baseline.

This record captures a paired V1/V2 kernel-time profile on gfx1100 for the exact fixture below. It does not authorize product-card claims or admit a transport route.

## Scope

A/B/B/A rocprofv3 captures of Qwen3.6-35B-A3B MQ V1 versus Ornith-1.5-35B-A3B MQ V2 (official zero-qt13) on a single RX 7900 XTX. The goal was to separate kernel-time deltas from product-transport outcomes at the same campaign commit, and to record the retained-PM4 sign reversal without treating command-dword growth as causal proof.

## Fixture

| field | value |
|---|---|
| Host | `hipx` |
| Worktree | `/home/kaden/mqv2-paired-profile` |
| Commit / binary source | `7d65cb243` |
| Binary | `/home/kaden/mqv2-paired-profile/target-profile/release/examples/bench_qwen35_mq4` |
| Binary md5 | `120d1a41eedf92a882e145398442b88b` |
| GPU route | `HIP_VISIBLE_DEVICES=0` |
| Device proof | every run logged `GPU dev 0: gfx1100` and `.hipfire_kernels/gfx1100` |
| Live VRAM | KFD node1 / RX 7900 XTX unique ID `0x43390a851e296ee5`; Strix gfx1151 / node2 remained at zero bytes |
| Profiler | rocprofv3 1.3.5 / ROCm 10 |
| V1 artifact | `/mnt/nas/kaden/hipfire/models/Qwen3.6-35b-a3b/qwen3.6-35b-a3b.mq4r` |
| V1 SHA256 | `4685c140c46b1a6f31a0fd9053bf09d5faf1d2529d715b84794249b66cde0428` |
| V2 artifact | `/mnt/nas/kaden/hipfire/models/Ornith-1.5-35B-A3B/ornith-1.5-35b-a3b.mq4r` (official zero-qt13) |
| V2 SHA256 | `84103fcc8ade42aa2ac8ec01176df7a4ead5e94810597c9fae2f6763152a3ac6` |

## Method

Four captures in A/B/B/A order: V1-A1, V2-B1, V2-B2, V1-A2.

Timed generation window anchored at the 100th-from-last `embedding_q8` dispatch. Each window contained:

- 100 embeddings
- 100 D2H copies
- 7,000 QKVZA dispatches
- 4,000 gate/up dispatches
- 60,500 dispatch records total

rocprof emitted a timestamp diagnostic and isolated multi-ms outliers. The comparison uses p99-winsorized per-kernel durations.

Raw captures remain on `hipx` at:

`/home/kaden/mqv2-paired-profile/profiles/{v1-a1,v2-b1,v2-b2,v1-a2}/hipx/*_results.json`

Durable machine-local summary:

`/home/kaden/mqv2-paired-profile/profiles/paired-summary.json`

These host-local paths are discovery pointers; the fixture identity and aggregates below are the durable in-tree record.

## Aggregate kernel results

Kernel sum per 100 generated tokens (p99-winsorized):

| capture | ms / 100 tokens |
|---|---:|
| V1-A1 | 364.769 |
| V1-A2 | 363.432 |
| V1 mean | 364.100 |
| V2-B1 | 354.306 |
| V2-B2 | 354.026 |
| V2 mean | 354.166 |

V2 advantage: **9.934 ms / 100 tokens** = **99.35 µs/token**.

Profiled wall p50:

| arm | p50 (ms) |
|---|---|
| V1 | 6.48 / 6.45 |
| V2 | 6.37 / 6.36 |

## Dominant V2 per-token kernel deltas versus V1

| kernel family | ∆ µs/token |
|---|---:|
| FA QKV | −32.61 |
| shared residual+gate | −18.53 |
| MoE gate/up | −16.16 |
| MoE down+postnorm | −14.08 |
| Delta QKVZA | −4.74 |
| residual | −2.85 |
| LM head | +0.53 |

## Product transports at the same campaign commit

| transport | V1 tok/s | V2 tok/s | V2 vs V1 |
|---|---:|---:|---|
| HIP | 219.649 | 221.874 | +1.013% (45.66 µs/token faster) |
| retained PM4 | 251.7978 | 249.095 | −1.073% (43.09 µs/token slower) |

Both retained sequences: **603 dispatches**.

Command dwords:

| arm | dwords |
|---|---:|
| V1 | 16,733 |
| V2 | 16,920 |
| V2 − V1 | **+187** (+1.118%) |

## Interpretation (not measurement)

- No remaining V2 MQV2 kernel deficit was observed on gfx1100 in this fixture.
- The HIP transport sign matches the kernel-time advantage; the retained-PM4 transport reverses it.
- The **+187 command dwords are a leading correlate, not causal proof.** Packet/register-class command-stream attribution is required before any compaction claim.
- Direct per-dispatch retained-PM4 timestamps remain code-gated to gfx12, so this gfx1100 run could not use that diagnostic.

## Explicit non-claims

- Not an admission decision.
- Not a current product performance baseline.
- Not transferable across model, quant, GPU, prompt, route, or method without a new record.
- Command-dword growth must not be cited as proven PM4 overhead without stream attribution.
