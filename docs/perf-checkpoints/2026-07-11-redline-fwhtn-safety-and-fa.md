# Redline FWHTN replay safety and flash-attention checkpoint

**Date:** 2026-07-11
**Host:** `hiptrx`, Radeon AI PRO R9700 (`gfx1201`), GPU 0
**Model:** `qwen3.6-35b-a3b.mq4r`
**Clock policy:** automatic; no clock or performance-level override

## Result

Redline retained PM4 replay is now safe for `fwht2`, `fwht3`, and `fwht4` KV.
The original FWHT3 capture omitted three dependent launches in each of the ten
full-attention layers: the FWHT K writer, attention tile, and attention reducer.
The resulting 803-launch PM4 tape matched its incomplete HIP-kernarg-blob oracle
but differed from the real 833-launch HIP forward in logits, KV, and recurrent
state.

All three launches now use the central kernarg-blob seam and the FWHT tile grid
uses the replay-stable maximum tile count while recording. Every FWHT mode now
captures 833 launches / 27 kernels and passes the 15-position byte gate against
both direct HIP and exact blob replay.

## Long-context flash-attention optimization

The Q8 flash tile already issued four independent token rows together. The FWHT
tiles serialized one Q·K row at a time. Four-row interleaving was ported to all
three FWHT tile kernels; FWHT3 also received four-row Q8-V accumulation.

Matched 8,192-context direct-HIP measurements use three runs of 20 decode
positions after resident prefill:

| KV mode | Before | After | Change |
| --- | ---: | ---: | ---: |
| FWHT2 | 122.0 tok/s | 127.2 tok/s | **+4.3%** |
| FWHT3 | 131.3 tok/s | 135.3 tok/s | **+3.0%** |
| FWHT4 | 121.0 tok/s | 126.2 tok/s | **+4.3%** |

The optimized FWHT3 kernel plus the retained PM4 product route measured:

| Route | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| HipGraph | 152.902 | 153.076 | 153.170 tok/s |
| Redline PM4 | 164.870 | 166.507 | 166.650 tok/s |

That is a **1.08774x** retained-replay gain at 8K context on top of the FWHT3
kernel improvement.

## Eight-turn sampled serve result

The real serve path was also exercised with the eight-turn coding/recall
session.  Both arms used the registry sampling recipe (`temperature=1.0`,
`top_p=0.95`, `top_k=20`, `presence_penalty=1.5`), seed `305419896`, medium
thinking effort, and `max_tokens=4096`.  Within each KV-mode pair, every prompt
length, generated-token count, and output byte matched between HipGraph and
Redline PM4.

| KV mode | HipGraph average | Redline average | Change | Final-turn context | HipGraph final | Redline final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q8 | 145.7 tok/s | 155.8 tok/s | **+6.9%** | 20,069 | 133.2 tok/s | 139.8 tok/s |
| FWHT3 | 148.6 tok/s | 159.3 tok/s | **+7.2%** | 17,633 | 138.5 tok/s | 147.5 tok/s |

The long-context tail (turns 6-8) improved from 135.8 to 143.4 tok/s for Q8
(**+5.6%**) and from 141.0 to 150.4 tok/s for FWHT3 (**+6.7%**).  Both modes had
zero attractor flags and preserved the expected 2/3 recall checks on turns 7
and 8.  FWHT3 turn 4 reached the 4096-token response cap in both arms, but its
output remained coherent and byte-identical.

Reverse-order HipGraph controls reproduced the original controls at 145.4
tok/s for Q8 and 149.1 tok/s for FWHT3.  This rules out the automatic-clock
warmup or arm order as the source of the Redline gain.

## Certification

- FWHT2/FWHT3/FWHT4: 15 consecutive positions bit-exact for logits, KV, and
  recurrent state against direct HIP and exact blob replay.
- Capture sequences are stable across repeated capture and remain 833 launches.
- DFlash fast coherence battery with `fwht3`: prose and code rows both `OK`, no
  hard or soft attractor flags.
- Redline replay unit suite: 6/6 passed.
- Automatic clocks throughout.

The harnesses now accept `--kv-mode q8|fwht2|fwht3|fwht4`, and the DFlash gate
accepts `HIPFIRE_GATE_KV_MODE` so this coverage is reproducible rather than a
one-off source edit.

## Reproduction

```bash
# Fifteen-position retained-PM4 gate (repeat for fwht2/fwht4).
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
python3 scripts/redline_daemon_harness.py \
  --model /home/kaden/.hipfire/models/qwen3.6-35b-a3b.mq4r \
  --daemon target/release/examples/daemon \
  --skip-prefill --kv-mode fwht3 --decode-context 128 \
  --decode-iterations 50 --capture-repeats 2 --measure-repeats 2 \
  --shadow-iterations 15 --max-seq 2048 --pm4 \
  --out .redline-work/fwhtn/fwht3-final-shadow15.json

# Matched product measurement at 8K context.
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
python3 -m tools.redline bench \
  --model /home/kaden/.hipfire/models/qwen3.6-35b-a3b.mq4r \
  --daemon target/release/examples/daemon --kv-mode fwht3 \
  --context 8192 --iterations 20 --warmups 1 --runs 5 \
  --transport pm4 --max-seq 16384 \
  --work-dir .redline-work/fwhtn/product-fwht3-long \
  --out .redline-work/fwhtn/product-fwht3-long.json

# Model-level FWHT3 DFlash gate.
HIPFIRE_GATE_KV_MODE=fwht3 \
HIPFIRE_COHERENCE_OUT=.redline-work/fwhtn/coherence-dflash-fwht3.md \
./scripts/coherence-gate-dflash.sh --fast

# Eight-turn sampled serve comparison. Repeat with --kv fwht3 and switch
# HIPFIRE_REPLAY_BACKEND between hip and auto; keep PM4 transport fixed.
HIPFIRE_REPLAY_BACKEND=auto HIPFIRE_REPLAY_TRANSPORT=pm4 \
HIPFIRE_AR_GRAPH=1 HIPFIRE_GRAPH=1 \
python3 scripts/serve_harness.py \
  --model /home/kaden/.hipfire/models/qwen3.6-35b-a3b.mq4r \
  --kv q8 --mtp off --thinking med --max-tokens 4096 --max-seq 32768 \
  --sampling registry --mode session \
  --session /home/kaden/mv/session_coding.json --seed 305419896
```
