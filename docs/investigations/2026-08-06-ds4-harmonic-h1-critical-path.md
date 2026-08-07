<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 harmonic H1: critical-path bill and native ownership map

Status: **complete from preserved evidence; no GPU execution performed**

Branch: `ds4-beta-staging`

Roadmap:
[`2026-08-06-deepseek4-harmonic-gfx1100-gfx1151.md`](../specs/2026-08-06-deepseek4-harmonic-gfx1100-gfx1151.md)

## Verdict

The accepted 32.002912953 tok/s direct-HIP route has real headroom, but the
old 62 tok/s first-order projection overestimated how much routed-expert work
the current layer DAG can hide.

The preserved post-grouped trace prices the route as:

```text
gfx1100 useful interval union                 17.014 ms/token
gfx1151 useful expert interval union           9.846 ms/token
measured cross-device useful overlap          -1.648 ms/token
                                               ----------------
global useful interval union                  25.212 ms/token
canonical product wall                        31.247 ms/token
explicit HIP/launch/gap accounting residual    6.036 ms/token
```

The accounting residual is not treated as zero and is not described as GPU
compute. It is the unprofiled product wall minus the measured useful GPU
interval union: host launch cost, queue gaps, dependency handling, terminal
copy/sampling work, and direct-HIP protocol overhead not represented by useful
kernel intervals. The profiler itself increased wall time from 31.247 to
35.819 ms/token, so its perturbed end-to-end rate is not a performance claim.

Eliminating the entire 6.036 ms residual would still leave 25.212 ms/token, or
39.66 tok/s. T1 therefore requires real kernel or graph work in addition to a
safe transport. Conversely, the dominant gfx1100 E8 tier is only at about 403
GB/s on a 960 GB/s-class card, so this is not a hardware ceiling.

## Identity and evidence

Canonical product result:

- implementation: grouped gfx1100 O-LoRA at `1f4f4c558`, recorded by
  `b2856ce8d`;
- prompt: `benchmarks/prompts/ds4_heterogeneous_code_2048.txt`;
- prompt MD5: `593234a767e71b97a3a4dad6431b47ce`;
- prompt/output: 2,048 / 512, batch 1, greedy, top-k 6, Q8 request mode;
- model SHA-256:
  `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`;
- output: 2,491 bytes, MD5 `ee05ab4f07393fb7d624d966a7dde4af`,
  SHA-256
  `3611840208334c77b3cfcf85984786920deabd550ba83311645f413d3ba6608b`;
- accepted median: 32.002912953 tok/s, 31.247155578 ms/token, three
  fresh-process samples.

Preserved selected-decode trace:

```text
/home/kaden/ds4-gfx1151-evidence/2026-08-06-ds4-heterogeneous-g5/
  post-grouped-device-timeline-selected/
```

| Artifact | SHA-256 |
|---|---|
| `run.log` | `56e98d5956a7601f4d0761e8239a1364f6bbf607fc2455d138c57c8d8e708876` |
| `rocprof/decode_results.db` | `c0fb0ed344b6f72509ea72bdb67ea14f67b20b639b2bfb2aae27be72d559687f` |
| `output.bin` | `3611840208334c77b3cfcf85984786920deabd550ba83311645f413d3ba6608b` |

The ROCTx-selected region contains 511 decode transitions from the 512-token
request. Every per-token trace value below divides occurrence and duration
totals by 511. The profiler run measured 27.965 tok/s and is used only for
device attribution.

## Dispatch and interval bill

The trace contains 3,165 GPU dispatch records per generated token:

| Owner / lane | Dispatches/token | Summed useful duration | Useful interval union | Role |
|---|---:|---:|---:|---|
| gfx1100 primary | 2,067 | 16.7239 ms | 15.6287 ms | serial dense, attention primary, shared expert, router, join |
| gfx1100 attention side | 750 | 3.0015 ms | 2.8352 ms | KV plus compressor branch |
| gfx1100 default | 4 | 0.0201 ms | 0.0201 ms | embedding and terminal work |
| gfx1151 expert | 344 | 9.8455 ms | 9.8455 ms | wait, routed gate/up/down, return |

The two gfx1100 compute lanes overlap for 1.4696 ms/token. The expert lane and
gfx1100 primary overlap for 1.6483 ms/token. The expert lane has zero measured
overlap with the attention side lane because the route is not available until
attention and FFN preparation have completed.

The current per-layer protocol itself is 344 operations/token:

- 172 peer copies: three outbound and one return for each of 43 layers;
- 86 device waits: one on each owner per layer;
- 86 device writes: one epoch publication in each direction per layer.

Those operations are 10.9% of all trace dispatch records before counting the
host HIP API calls. The waiting pseudo-kernels report 8.745 ms/token on
gfx1100 and 23.384 ms/token on gfx1151, but those are blocked queue intervals,
not additive useful compute. They prove the dependency structure and the
unsafe indefinite-wait mechanism; they are not added again to the union bill.

## Hot occurrence-weighted kernels

| Owner | Kernel class | Calls/token | Summed time/token | Bytes/token | Achieved rate | Implementation class |
|---|---|---:|---:|---:|---:|---|
| gfx1100 | generic `gemv_mfp4g32_e8_soa` | 511 | 7.586 ms | 2.858 GB | 376.7 GB/s | exact-compiled generic fallback; not gfx1100-tuned |
| gfx1100 | grouped O-LoRA E8 | 43 | 1.419 ms | 0.772 GB | 544.2 GB/s | exact-gfx1100 structure, accepted |
| gfx1100 | all dense E8 | 554 | 9.005 ms | 3.630 GB | 403.1 GB/s | mixed generic plus one native family |
| gfx1151 | MQ2-Lloyd gate/up plus down | 86 | 8.562 ms | 1.830 GB | 213.7 GB/s | certified exact-gfx1151 expert family |
| transport | compact route/result chain | 172 copies | 0.589 ms G0 oracle | 1.409 MB | 2.39 GB/s effective | public-ROCr SDMA oracle; latency dominated |

Dense E8 bytes use the measured 3.630 GB/token artifact tier. Grouped O-LoRA
is `43 * 8 * 1024` rows at 2,192 bytes/row, or 0.772145152 GB/token; the
remaining generic tier is the difference. Expert bytes are the measured six
selected experts' 1.83 GB/token.

The largest non-GEMV summed kernel durations on gfx1100 are:

| Kernel | Calls/token | Time/token |
|---|---:|---:|
| `deepseek4_attn_swa_topk_f32_buf` | 41 | 2.032 ms |
| `hc_compute_control` | 86 | 1.448 ms |
| `rope_tail_yarn_interleaved_f32` | 86 | 0.964 ms |
| `fused_rmsnorm_mq_rotate_plain` | 86 | 0.777 ms |
| `compressor_add_ape_f32_buf` | 62 | 0.554 ms |
| gfx1100 `copyBuffer` dispatches | 222 | 0.538 ms |
| `rmsnorm_f32` | 130 | 0.397 ms |
| `hc_sinkhorn_4x4` | 86 | 0.348 ms |
| `deepseek4_topk_kv_gather_f32_buf` | 21 | 0.318 ms |
| indexer chunk-sort + score + merge | 84 | 0.700 ms |

These context/state kernels do not have one constant weight-byte stream. Their
measured time remains fully charged in the lane union; absent source-byte
metadata in rocprof is not converted to zero bandwidth or zero cost.

## Native ownership map

### gfx1100 canonical owner

Serial, before the expert fork:

- embedding and residual initialization;
- attention RMS/FWHT preparation;
- Q-LoRA, joint KV, compressor and indexer projections;
- sparse attention/indexer selection and attention HC mix;
- FFN HC preparation, RMS/FWHT, router and top-k.

Overlap eligible before the route exists:

- Q-LoRA projection on the primary lane versus KV and both compressor branches
  on the attention-side lane. This already hides 1.4696 ms/token.

Overlap eligible after the route exists:

- the three shared-expert projections and their SwiGLU work while gfx1151
  evaluates routed experts. This is the source of essentially all 1.6483
  ms/token of measured cross-device useful overlap.

Serial, after the expert return:

- ordered shared-plus-routed add;
- FFN HC mix;
- the next layer and, ultimately, final norm/head and sampling.

The primary dense E8 kernel is only an exact-target compilation of the generic
source. The grouped O-LoRA family is the sole measured architecture-structured
gfx1100 E8 win. The gfx1100 two-stage indexer is exact-architecture admitted,
but it is a port of the existing wave32 structure rather than a complete
gfx1100-native indexer design. H3 therefore has a real, occurrence-weighted
native-kernel job; it is not recompiling already optimal kernels.

### gfx1151 expert owner

The owner retains 77.914 GB of routed payloads and executes, per layer:

- wait for the typed activation/route publication;
- zero routed partial;
- selected MQ2-Lloyd gate/up;
- selected SwiGLU/rotation;
- selected MQ2-Lloyd down and routed accumulation;
- one 16 KiB result return and completion publication.

The two projection kernels already sustain 213.7 GB/s against roughly 256
GB/s peak shared-memory bandwidth. This tier has less conventional kernel
headroom than the gfx1100 dense tier and must not be assigned an imagined 2.184x
Qwen architecture uplift.

### Required join and dependency limit

The next layer cannot begin until the routed partial has been added and
`hc_ffn_mix` has produced canonical state. The next autoregressive token cannot
begin until the final head selects the current token. Therefore neither
cross-layer nor cross-token pipelining can hide the remaining expert branch
without changing arithmetic/state ownership.

The current same-layer independent shared branch is only about 1.65 ms/token.
Scheduling it better cannot hide a 9.85 ms/token expert service. This is the
specific dependency that the earlier first-order estimate omitted.

## Target budgets

| Gate | Budget | Saving from 31.247 ms product | Saving still required after zeroing the 6.036 ms residual |
|---|---:|---:|---:|
| T1 50 tok/s | 20.000 ms | 11.247 ms | 5.212 ms of useful kernel/overlap work |
| T2 60 tok/s | 16.667 ms | 14.580 ms | 8.545 ms of useful kernel/overlap work |
| T3 62 tok/s | 16.129 ms | 15.118 ms | 9.082 ms of useful kernel/overlap work |

With current useful costs, required cross-device overlap would be:

```text
T1: 17.014 + 9.846 - overlap <= 20.000
    overlap >= 6.860 ms/token (69.7% of expert service)

T2: overlap >= 10.193 ms/token, greater than the entire expert service
T3: overlap >= 10.731 ms/token, greater than the entire expert service
```

Current overlap is 1.648 ms/token. Even perfect overlap of the entire expert
service leaves the current gfx1100 union at 17.014 ms/token: 58.78 tok/s before
transport and terminal work. T2 would then require at least 0.347 ms/token of
gfx1100 improvement, and T3 at least 0.885 ms/token. Full expert overlap is not
available in the current layer DAG, so those are mathematical lower bounds,
not projections.

## Consequences for the next gates

1. H2 remains mandatory because the current protocol is unsafe and contributes
   hundreds of per-token dispatch/API operations. Its target is bounded
   lifecycle plus the 0.589 ms G0 transport class, not a claimed 11 ms win.
2. H3 must first replace the 511-call, 376.7 GB/s generic gfx1100 E8 family.
   A 600 GB/s weighted tier saves about 2.82 ms/token; that alone is not T1.
3. The non-GEMV gfx1100 classes need graph/fusion work worth roughly another
   2-3 ms/token for T1 after transport/launch residue is removed.
4. T2/T3 are conditional under the frozen owner map. If H2+H3 cannot create
   substantially more independent same-layer work, 60-62 requires an explicit
   ownership revision such as a resident, capacity-bounded gfx1100 expert
   shard. Streaming expert weights is still prohibited and infeasible.
5. No performance ceiling is declared from this direct-HIP trace. The generic
   dense tier uses only about 39% of gfx1100 peak bandwidth, and retained
   launch overhead has not been composed.

## H1 exit

H1 exits with every current wall-time term charged to useful gfx1100 work,
useful gfx1151 work, measured overlap, or an explicit non-useful residual. The
ownership and dependency map identifies what can and cannot overlap. T1 has a
quantified 11.247 ms product gap and 5.212 ms useful-work gap after ideal
launch cleanup. T2 and T3 remain measured targets, but their current-owner
ceiling is explicitly conditional rather than silently assuming unavailable
expert overlap.
