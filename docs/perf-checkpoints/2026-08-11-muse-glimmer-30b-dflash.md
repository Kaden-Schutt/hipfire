# Muse Glimmer 30B — DFlash speculative decode, first working measurement

**Lifecycle:** `historical`. Evidence under the exact fixture below. Not a current
default (DFlash remains opt-in via `HIPFIRE_DFLASH_DRAFT`), not an automatic
baseline, not an admission decision.

**Disposition:** first measurement in which Glimmer DFlash is both correct and
faster than AR. Companion to
[`2026-08-11-muse-glimmer-30b-bringup-baseline.md`](2026-08-11-muse-glimmer-30b-bringup-baseline.md),
which is the AR-only bring-up baseline for the same arch.

**Branch:** `muse-glimmer`, on `beta` = `b65f8159c`.

---

## Fixture

| field | value |
|---|---|
| target A | `muse-glimmer-30b.mq4` — MQ4 lm_head, 15.51 GB |
| target B | `muse-glimmer-30b-q8head.mq4` — Q8 lm_head, 16.26 GB, md5 `992375e7308e5d2c0dbe0a3852498915` |
| drafter | `muse-glimmer-30b-assistant.q8.hfq`, arch 23, md5 `9e170f3bdf53f1f22192e94c0aec0522` |
| prompt | `benchmarks/prompts/glimmer_bringup_merge.txt`, md5 `2ef49ee70df1483079b1f73c1f768339` |
| binary | md5 `0bd7bab6a1c485d960a8821abc9f8ba0` |
| GPU | hiptrx gfx1201 (R9700), device 0/1 |
| method | 64 tokens, greedy (temp 0), fresh daemon per run, 3 runs per arm |

## Result

| target | AR (3 runs) | DFlash (3 runs) | ratio | tau | byte-identical |
|---|---|---|---:|---:|---|
| MQ4 lm_head | 32.96 · 32.94 · 32.94 | 41.03 · 41.00 · 40.95 | **1.24x** | **8.333** | yes |
| Q8 lm_head | 31.73 | 29.32 | 0.92x | 4.200 | yes |

66 of 135 proposals accepted over 9 windows on target A. Byte-identical output at
temp 0 is the required contract, not a bonus: acceptance is greedy-argmax, so
divergence would be a correctness bug rather than an acceptance-rate matter.

## The tau gap is quantization noise, not a drafter difference

Target B has a strictly more faithful lm_head, yet tau HALVES. The obvious
explanation — that drafts and verify picks were flowing through different
numerical paths (per-row GEMV vs batched GEMM), so near-tie argmax flips became
spurious rejections — was tested directly with
`HIPFIRE_GLIMMER_BATCHED_LM_HEAD=0` and **killed**: tau is 4.200 with batching on
and 4.200 with it forced off.

The remaining reading is the uncomfortable one. MQ4's coarser output projection
collapses near-ties, so draft and target agree more often than the model really
warrants. Tau 8.333 on target A is partly quantization noise inflating agreement.
Tau 4.200 is the honest rate. Both artifacts still emit byte-identical output to
their own AR baseline, so this costs no correctness — but a high tau on a lossier
artifact should not be read as the drafter being better.

## Batched verify is the whole game

Verify is ONE batched forward over the block, not B sequential decodes. Same
drafter, same tau 8.333, sequential vs batched:

| verify | tok/s | vs AR |
|---|---:|---:|
| B sequential full forwards | 12.0 | 0.36x |
| one batched forward | 41.0 | 1.24x |

The sequential version streamed all 15.5 GB of weights 16 times per window. Tau
was already 8.333 there — a good acceptance rate bought nothing until the verify
economics were fixed.

## Per-window breakdown (`HIPFIRE_GLIMMER_TIMING=1`)

Target B, first window, 4.2 tokens emitted:

```
noise 0.5ms  drafter 7.7ms  draft_lm 69.4ms  verify 55.2ms  total 132.8ms
                                             (layers 46.9  norm 0.1  lm 7.6)
```

The 52-layer batched forward is 46.9 ms against 480 ms sequential — a 10x win,
and the reason this works at all.

## Open: the draft-side lm_head anomaly

`draft_lm` (69.4 ms, 15 rows) and verify's `lm` (7.6 ms, 16 rows) are the same
weight through the same `gemm_q8_0_batched_chunked` → `gemm_q8_0_wmma` dispatch,
and differ 15x. In isolation both measure 2.7–4.1 ms. As throughput that is
~20 GB/s versus ~312 GB/s on a ~640 GB/s part, so the slow one is at ~3% of
roofline. Neither call can be served from L2 (single-digit MB against a 1.34 GB
weight), so "first read misses, second read hits" does not explain it. The MQ4
`fp16_shadow_cache` is a different path and is not involved.

This is unresolved and is the reason target B sits at 0.92x. If `draft_lm` ran at
verify's rate the window would drop from ~133 ms to ~70 ms — about 16.6 ms/token
against AR's 31.7, i.e. comfortably past 1.0x on the compliant artifact. Treat the
15x as a defect to be found, not as a memory-hierarchy cost to be designed around.

## Reproduce

```bash
ssh hiptrx 'cd /tmp && HOME=/home/kaden/glimmer-home-1 \
  GLIMMER_DAEMON=/home/kaden/wt-glimmer/target/release/examples/daemon \
  GLIMMER_MODEL=/home/kaden/.hipfire/models/muse-glimmer-30b.mq4 \
  GLIMMER_PROMPT_FILE=/tmp/glimmer_prompt.txt HIP_VISIBLE_DEVICES=1 \
  HIPFIRE_DFLASH_DRAFT=/home/kaden/.hipfire/models/muse-glimmer-30b-assistant.q8.hfq \
  python3 /tmp/glimmer-smoke.py x 64'
```
Drop `HIPFIRE_DFLASH_DRAFT` for the AR arm.
