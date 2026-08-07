<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 native gfx1100 owner throughput gate

Status: **accepted for harmonic composition; not a full-model throughput claim**

## Verdict

The exact-gfx1100 DS4 owner clears the 50 tok/s critical-path gate when its
model-shaped body is retained in one PM4 packet:

| Route | Samples | Median ms/token | Median tok/s | Verdict |
|---|---:|---:|---:|---|
| direct HIP | 12 | 21.318065 | 46.9086 | below gate |
| retained PM4 | 12 | 16.440211 | 60.8265 | **gate passed** |

The PM4 result was bit-identical to the captured HIP logits on all 12 replays.
The post-run health check reported no KFD processes on hipx.

This proves that the gfx1100 owner is capable of supporting a 50 tok/s
harmonic critical path. It does **not** prove 60.83 tok/s full-model decode:
the fixture executes route selection but intentionally omits routed
MQ2-Lloyd expert arithmetic, uses a fixed 2,048-depth position for replay, and
has no gfx1151 peer or transport.

## Exact fixture

- host: `hipx`;
- GPU: unique live gfx1100 resolved through HIP, then rebound by PCI identity
  `0000:66:00.0`;
- artifact: `/home/kaden/models/deepseek-v4-flash-0731.mq2r`;
- artifact SHA-256:
  `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`;
- context start: 2,048;
- warmups: 4;
- timed samples: 12 per arm;
- top-k route selection: 6;
- routed expert arithmetic: omitted after route selection;
- peer devices: zero;
- GPU lock: held;
- hard timeout: 600 seconds.

The owner body contains the production dense MQ2R weights, attention and
indexer state, router/top-k, shared expert, HC mixes, final norm, and output
head. The loaded model now owns an exact-gfx1100 backend for dense E8 and
grouped O-LoRA dispatch rather than entering through a portable RDNA branch.

## Retained identity

```text
capture launches      2469
unique symbols          34
AQL contracts         34/34
sequence hash         14208066552233626259
prepared dispatches   2469
packets                  1
queues                   1
phases                   1
command dwords        62235
replay bit exact        yes
```

The first PM4 sample was a cold outlier at 23.352320 ms. The remaining eleven
samples were 16.369394--16.765956 ms; the median above includes all twelve.

Direct HIP issued 2,470 launches/token and measured 1.745335 ms/token inside
the host launch calls. Retention saved 4.877855 ms/token end to end, showing
that host launch time alone understated the queue-gap and dependency cost of
the direct path.

## Provenance

- exact gfx1100 backend and owner fixture: `0a26ec722`;
- retained owner probe: `d61feaa32`;
- benchmark binary SHA-256:
  `48a27446295e8389a98246173926a3c2c5a5ba65f3be11cfd93bf454f9f2cdcd`;
- direct-HIP JSON SHA-256:
  `b844dd7b02eaec059f1ec8bcf280e9c0ce510ae6ee1dfc82c219ffe7ce91835c`;
- retained-PM4 JSON SHA-256:
  `b62d3312beabacc8d41582b1eb484f07150ba6406803a6fa865f005f383291f2`;
- raw evidence:
  `hipx:/home/kaden/ds4-gfx1151-evidence/2026-08-07-gfx1100-owner-gate/`;
- campaign ledger row:
  `2026-08-07-ds4-harmonic-gfx1100-owner-gate-v1`.

## Consequence

The next harmonic step may compose from this retained owner proof. It must
preserve the single-owner PM4 body while adding the finite gfx1151 expert
service; per-layer HIPGraph callbacks and indefinite cross-device GPU waits
remain rejected. The product T1 gate stays open until the canonical 2,048/512
full-model route itself reaches 50 tok/s with byte-identical decoded output.
