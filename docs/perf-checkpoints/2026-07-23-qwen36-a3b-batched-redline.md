# Qwen3.6 35B-A3B MQ4R batched Redline on gfx1201

Date: 2026-07-23

Host: `hiptrx`, four gfx1201 devices with 34.2 GB usable VRAM each

Model: `/home/kaden/.hipfire/models/qwen3.6-35b-a3b.mq4r`

Harness: `qwen35_batch_generate`, fixed-slot independent sampled AR, Q8 KV,
short uniform prompts, 64 output tokens, temperature 1.0, top-p 0.95, top-k
20, presence penalty 1.5. `decode_model` excludes prompt setup and output
serialization; `wall` includes both.

## Shadow admission

The retained route was not benchmarked as a product route until ordinary HIP,
captured-launch HIP, and retained PM4 produced identical logits, KV, and
recurrent-state bytes in two independent observations. The final B100 proof
advanced the same retained tape over eight positions per observation.

| Batch | Steps per observation | HIP/captured-HIP/PM4 exact | Dispatches | Kernels | PM4 dwords | Sequence hash |
|---:|---:|:---:|---:|---:|---:|:---|
| 63 | 8 | yes | 1,184 | 29 | 36,784 | `97f5a5b7c25141cc` |
| 64 | 8 | yes | 1,184 | 29 | 36,784 | `e831c4743b4c70fb` |
| 80 | 2 | yes | 1,184 | 29 | 36,784 | `f7a497383e4a3174` |
| 96 | 2 | yes | 1,184 | 29 | 36,784 | `dbfe9d3dc4e1b266` |
| 100 | 8 | yes | 1,184 | 29 | 36,784 | `23ad6c869ebab0c9` |

The B100 eight-step observations ended with distinct state hashes, proving
that the second observation was not a duplicate input:

| Observation | Logits | KV | Recurrent |
|---:|:---|:---|:---|
| 0 | `426f2b5e1e43413d` | `84e4f72508efb104` | `bf38646f1f089100` |
| 1 | `535607584d727aa0` | `1a799f2621b7c2c5` | `eb6c13ca7033ccc0` |

Each retained hash matched its ordinary-HIP hash exactly.

## Single-device throughput curve

The product profile captures the first model launch and routes the following
62 launches through one retained PM4 IB. The gfx12 adaptive bt4 family has a
large discontinuity at B64 for this exact model, so the certified profiles
retain bt1. This is a no-op below B64.

| Batch | Redline model tok/s | Change from prior point |
|---:|---:|---:|
| 63 | 1,538.3 | — |
| 64 | 1,542.5 | +0.3% |
| 80 | 1,651.3 | +7.1% |
| 96 | 1,684.6 | +2.0% |
| 100 | **1,703.8** | +1.1% |

B102 and B104 allocate most of the fixed-slot state but OOM on the first
prefill launch. B112 OOMs while allocating batch state. B100 is therefore the
usable 4K-context capacity edge without reducing state or model memory.

At B63, the matched sampled A/B was:

| Route | Model tok/s | Wall tok/s | Retained replays |
|:---|---:|---:|---:|
| HIP | 1,503.1 | 1,003.0 | 0 |
| Redline PM4 | 1,538.3 | 1,034.6 | 62 |

All 4,032 sampled completion tokens matched HIP exactly.

## Four-device simultaneous run

All four devices ran the B100 product profile concurrently with independent
per-device locks.

| Device | Model tok/s | Wall tok/s | Retained replays |
|---:|---:|---:|---:|
| 0 | 1,713.2 | 1,112.5 | 62 |
| 1 | 1,702.6 | 1,080.6 | 62 |
| 2 | 1,711.5 | 1,089.0 | 62 |
| 3 | 1,715.6 | 1,111.6 | 62 |
| **Aggregate** | **6,842.9** | **4,393.7** | **248** |

The four output-token matrices had the same SHA-256:
`3d967143074fa773609069e73f39c71f909ed93ddfe149ea14e7ed3ca943d389`.

## Current ceiling

For this short-context offline sampled workload, the demonstrated ceiling is
about **1.70K model tok/s per card** and **6.84K model tok/s across hiptrx**.
The end-to-end harness rate is about **1.10K wall tok/s per card** and **4.39K
wall tok/s aggregate**. Moving the model-rate ceiling materially higher now
requires reducing fixed-slot state memory to admit wider batches or improving
the batched compute kernels; submission tuning alone is no longer the primary
limiter at B100.
