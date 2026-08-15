# Retained PM4 full dirty-gate: small win on the gfx12 GCR base

## Verdict

Ship full SH-register dirty-gating as the retained-PM4 default. On top of the
gfx12 GCR trim it improves the 30-row A3B median in both A/B orders and is
neutral-to-positive across the sampled eight-turn session.

`HIPFIRE_REPLAY_PM4_STATEFUL=static` restores the prior static-only policy;
`legacy` retains full register emission.

## What changes

The stateful encoder keeps the last value of every programmed SH register.
Repeated program/resource/workgroup values are omitted; an exact repeated
kernel with the same kernarg GPU address emits only `DISPATCH_DIRECT` and its
grid/initiator body. A changed kernarg still rebinds `COMPUTE_USER_DATA_0`.

The measured A3B tape shrinks from 22,891 dwords under static-only retention to
18,090 dwords under the full dirty gate. Dispatch count and dependency waits do
not change.

## Gates

- `hiptrx`, gfx1201, automatic clocks.
- Qwen 3.6 35B A3B MQ4R, Q8 KV.
- Stable 833-launch capture `8ba4c8d66f32d116`.
- Fifteen-position logits/KV/recurrent/blob shadow: bit-exact.
- Context 128, 100 tokens, ten warmups, 30 rows in both A/B orders.
- Fixed-seed sampled eight-turn serve session through context 20,069.

## Results

| Order | Static-only median | Full dirty-gate median | Delta |
|---|---:|---:|---:|
| static then full | 192.474 tok/s | 193.777 tok/s | +0.68% |
| full then static | 194.036 tok/s | 194.588 tok/s | +0.28% |

| Eight-turn sampled session | Average decode | Final turn |
|---|---:|---:|
| static-only | 166.088 tok/s | 148.1 tok/s |
| full dirty-gate | **166.238 tok/s** | 148.0 tok/s |

The serve average is +0.09%; seven of eight turns improve and the final turn is
within 0.1 tok/s in the other direction. No empty, runaway, or attractor output
occurred.

Artifacts on `hiptrx`:

```text
/home/kaden/.redline-work/hipfire-pm4-lean/.redline-work/pm4-lean/stateful/
  shadow15.json
  control-30.json
  candidate-30.json
  reverse-candidate-30.json
  reverse-control-30.json
  serve-control.json
  serve-candidate.json
```

Only host-side PM4 encoding policy changed. Kernel sources and compiler inputs
are untouched, preserving gfx1100 kernel `.text` byte-for-byte.
