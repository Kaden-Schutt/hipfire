# GFX12 retained-PM4 GCR trim: +2.1–2.6% at tg100

## Verdict

Ship. Re-deriving the retained tape's `ACQUIRE_MEM` GCR words from gfx12
cache intent raises A3B single-stream retained-PM4 decode from about 189.08 to
193.07–193.93 tok/s in matched 30-row tests and from 163.0 to 165.2 tok/s in
the sampled eight-turn session.

This supersedes the historical 190.66 tok/s champion on the same gfx1201 host.

## Encoding

The legacy encoder reused ROCr's gfx10+ `0x1c3f1` word at every acquire. On
gfx12 that carries removed/merged hierarchy fields and requests a full L2
action at all 80 required inter-node boundaries.

The retained tape now emits:

- token entry: `0x1c1d1`, preserving the system ownership boundary, L2
  writeback/invalidate, and instruction/scalar/vector visibility;
- inter-node: `0x10180`, retaining forward sequencing plus scalar/vector
  invalidation while leaving immutable code and coherent L2/MALL resident.

`HIPFIRE_REPLAY_PM4_GCR_TRIM=0` is the same-binary fail-safe/control and emits
the prior `0x1c3f1` words.

## Gates

- `hiptrx`, gfx1201, automatic clocks.
- Qwen 3.6 35B A3B MQ4R, Q8 KV.
- Stable 833-launch capture `8ba4c8d66f32d116`.
- Fifteen-position shadow: logits, KV, recurrent state, and complete blob all
  bit-exact.
- Product A/B: context 128, 100 tokens, ten warmups, 30 rows, repeated in
  reversed order using one release binary.
- Serve A/B: sampled fixed-seed eight-turn session, medium thinking, context
  49 through 20,069; no empty, runaway, or attractor output.

## Results

| Order | Legacy control median | GFX12 GCR median | Delta |
|---|---:|---:|---:|
| control then candidate | 189.079 tok/s | 193.069 tok/s | +2.11% |
| candidate then control | 189.082 tok/s | 193.927 tok/s | +2.56% |

The candidate's 30-row minima were 192.546 and 193.802 tok/s, both above the
190.66 historical bar.

| Eight-turn sampled session | Average decode | Final turn at ctx 20,069 |
|---|---:|---:|
| legacy control | 163.0 tok/s | 145.6 tok/s |
| GFX12 GCR trim | **165.2 tok/s** | **147.4 tok/s** |

The long-session improvement is +1.39% on the unrounded averages
(162.975 to 165.238 tok/s), and every individual turn improved.

Artifacts on `hiptrx`:

```text
/home/kaden/.redline-work/hipfire-pm4-lean/.redline-work/pm4-lean/gcr-trim/
  shadow15.json
  control-30.json
  candidate-30.json
  reverse-candidate-30.json
  reverse-control-30.json
  serve-control.json
  serve-candidate.json
```

Only the Rust PM4 command encoder and replay policy changed. No HIP kernel
source, compilation option, or HSACO input changed; gfx1100 kernel `.text` is
therefore byte-identical by construction.
