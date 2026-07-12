# Retained PM4 inter-node L2-skip: bit-exact, no product win

## Verdict

Reverted. Clearing only `GL2_INV` and `GL2_WB` from the 80 required
inter-node `ACQUIRE_MEM` packets passed bit-exact validation but did not beat
the single-stream retained-PM4 control.

## Method

- Host: `hiptrx`, gfx1201, automatic clocks.
- Model: Qwen 3.6 35B A3B MQ4R, Q8 KV.
- Stable capture: 833 launches, sequence hash `8ba4c8d66f32d116`.
- Candidate inter-node GCR word: `0x103f1`; token-entry system acquire stayed
  `0x1c3f1`.
- Same release binary for both arms; `HIPFIRE_REPLAY_PM4_L2_SKIP=0` selected
  the system-acquire control.
- Fifteen-position shadow compared logits, KV, and recurrent state.
- Product gate: context 128, 100 measured tokens, ten warmup rows, 30 rows.
- Serve gate: sampled, fixed seed, medium thinking, eight-turn session through
  context 20,069.

## Results

| Arm | tg100 PM4 | Eight-turn average | Parity |
|---|---:|---:|---|
| system inter-node acquire control | **191.016 tok/s** | 162.6 tok/s | bit-exact |
| L2-skip inter-node acquire | 190.096 tok/s | 162.6 tok/s | bit-exact |

The candidate was 0.48% slower at tg100 and exactly neutral in the long-session
average. It also missed the established 190.66 tok/s champion bar. The result
does not support the hypothesis that these mid-tape L2 actions are the limiting
weight-residency cost for this cache-resident A3B decode tape.

Artifacts on `hiptrx`:

```text
/home/kaden/.redline-work/hipfire-pm4-lean/.redline-work/pm4-lean/l2-skip/
  shadow15.json
  control-30.json
  candidate-30.json
  serve-control.json
  serve-candidate.json
```

No kernel source or compiler option changed, so gfx1100 kernel `.text` remains
byte-identical by construction.
