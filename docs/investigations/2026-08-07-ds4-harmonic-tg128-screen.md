# DS4 harmonic TG128 composition screen

Date: 2026-08-07  
Branch: `ds4-beta-staging`  
Fixture commit: `8bbd8a8d7cb289bee835923a399e5bedd4e132ee`

## Verdict

Reject the current per-layer checkpointed retained-AQL composition. The
arithmetic is correct, but the route is structurally slower than direct HIP:
`12.5142 tok/s` versus `30.3318 tok/s` on the new TG128 screen. The decoded
outputs are byte-identical.

This is not a rejection of the gfx1100 kernels or the gfx1151 hot expert set.
The hotset reduces expert wait by `4.9554 ms/token`; the loss is the
`48.6755 ms/token` increase in route synchronization from synchronizing and
checkpointing 43 separately prepared layer queues.

The next composition must keep the proven one-owner gfx1100 retained body
coarse. It must not retry either per-layer checkpointed AQL on gfx1100 or the
already rejected synchronous per-layer PM4 chord on gfx1151.

## Fixture

- Model: `/home/kaden/models/deepseek-v4-flash-0731.mq2r`
- Model SHA-256:
  `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`
- Prompt: `benchmarks/prompts/ds4_heterogeneous_code_2048.txt`
- Prompt MD5: `593234a767e71b97a3a4dad6431b47ce`
- Context: first 128 tokens of the canonical 2,048-token prompt
- Generated tokens: 128
- Batch: 1
- Sampling: greedy, temperature 0
- Routed experts: top-k 6
- Dense device: gfx1100, `0000:66:00.0`
- Expert device: gfx1151, `0000:bf:00.0`
- GPU lock: held for each arm and released

TG128 is a screening fixture only. It does not satisfy a campaign throughput
target and cannot support a prefill claim. The promotion fixture remains
2,048 prompt tokens / 512 generated tokens.

## Binary identity

- Harness SHA-256:
  `13421af090f373f6469b6d3ed6229b568a489d399505ad058351b63ed58f9e21`
- Expert worker SHA-256:
  `ef6b4980805268cea026558fe9e54543c83d050685cc067da58fa3602cace33c`
- Hotset plan:
  `benchmarks/routes/ds4_0731_harmonic_hotset_1400.ds4hot`
- Hotset SHA-256:
  `af643539ff01acf706a14073dc4898c058bc1b8d241279b3cff7719151eca7b5`

## Results

| Metric | Checkpointed AQL + hotset | Direct-HIP control | Delta |
|---|---:|---:|---:|
| Decode tok/s | 12.5142 | 30.3318 | -58.74% |
| Layer wall, ms/token | 79.7399 | 32.5048 | +47.2351 |
| Route sync, ms/token | 66.5885 | 17.9129 | +48.6755 |
| Expert wait, ms/token | 7.1372 | 12.0926 | -4.9554 |
| Stream sync, ms/token | 14.9303 | 17.9110 | -2.9806 |
| D2H, ms/token | 0.6254 | 0.5575 | +0.0679 |
| Generated tokens | 128 | 128 | identical |
| Decoded bytes | 490 | 490 | identical |
| Output MD5 | `1e853189fd7dba9b03936590ac6b0b81` | `1e853189fd7dba9b03936590ac6b0b81` | identical |

The direct control intentionally omits the hotset plan and therefore sends all
routed experts to gfx1151. That difference biases the control against the
candidate: its expert wait is `4.9554 ms/token` worse. Even with that handicap,
the control is 2.42x faster. The component timings are therefore sufficient to
reject the checkpoint shape without spending a second product sample.

Historical same-hotset direct-HIP evidence at the longer diagnostic fixture is
consistent with the direction: `32.3171 tok/s`, `0.594 ms/token` route sync,
`7.236 ms/token` expert wait, and `30.431 ms/token` layer wall. It is supporting
diagnostic evidence only, not blended into the TG128 comparison.

## Mechanism

`harmonic_run_retained_dense_ffn` currently performs this sequence once per
layer:

1. synchronize the gfx1100 primary HIP stream;
2. submit a separately prepared checkpointed AQL queue;
3. wait for that layer's route checkpoint;
4. publish the route to the gfx1151 worker;
5. wait for the same layer queue's terminal completion.

The implementation is at
`crates/hipfire-arch-deepseek4/src/forward.rs:6747`, with the primary stream
synchronization at `:6762`, submit at `:6796`, checkpoint wait at `:6800`, and
terminal wait at `:6805`.

There are 43 prepared layer controllers, each with 18 dispatches. Candidate
route synchronization is `66.5885 ms/token`; only `14.9303 ms/token` is
reported as HIP stream synchronization. The remaining approximately
`51.6581 ms/token`, or about `1.20 ms/layer`, is checkpointed queue
submission/wakeup/wait overhead. That tax overwhelms both the hotset gain and
the proven gfx1100 kernel throughput.

The separate owner gate remains valid: the exact gfx1100 body measured
`46.9086 tok/s` under direct HIP and `60.8265 tok/s` when retained as one PM4
packet. The performance exists; fragmenting it into 43 checkpointed queue
transactions destroys it.

## Next cut

Design a coarse single-owner continuation protocol around the already proven
whole-token gfx1100 retained body:

- one persistent owner queue/tape rather than one queue per layer;
- owner-local, finite checkpoint and continuation signals;
- CPU publishes the typed route packet after the owner-local route checkpoint;
- gfx1151 remains direct HIP for routed experts;
- continuation is released only after the gfx1151 result is complete;
- no peer-owned GPU wait and no unbounded `WAIT_REG_MEM`;
- retain finite timeout, queue inactivation, and exact runtime device binding.

Before another model run, screen that protocol with a small multi-checkpoint
oracle. It must demonstrate that checkpoint/continuation overhead projects to
at least a 2% end-to-end win. The next model run remains TG128; 2,048/512 is
reserved for a passing promotion candidate.

## Evidence

- Candidate:
  `hipx:/home/kaden/ds4-gfx1151-evidence/2026-08-07-harmonic-tg128-screen-v1/`
- Direct-HIP control:
  `hipx:/home/kaden/ds4-gfx1151-evidence/2026-08-07-harmonic-tg128-direct-control-v1/`

## Skipped

- No repeat samples: the candidate regressed by 58.74%, far beyond the reject
  threshold.
- No 2,048/512 promotion run.
- No prefill claim; this example's prefill path iterates `decode_step`.
- No gfx1151 PM4 retry; H8 v3 already rejected synchronous per-layer PM4.
- No Qwen or gfx1100-shipping replay policy change.
- No weight, quant, sampling, KV, or model-format change.
