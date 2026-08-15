# DeepSeek V4 gfx1201 TP3 HC fusion checkpoint

Date: 2026-08-07  
Branch: `ds4-gfx1201-opt`  
Candidate: `607edb7ea`  
Parent promoted line: `252192fc6`

## Verdict

Promoted. Exact-gfx1201 DeepSeek V4 MQ2R now uses the existing one-launch HC
control/finalize kernel. The candidate composes two changes:

1. `hc_finalize_control` replaces `hc_apply_alpha`,
   `hc_pre_post_sigmoid_scale_f32`, and `hc_sinkhorn_4x4`.
2. `hc_compute_control_vec4_finalize` folds that finalizer into the control
   projection.

The route is selected automatically only for DeepSeek V4 MQ2R on exact
`gfx1201`. gfx1151 keeps its certified defaults; gfx942, gfx1100, Qwen, other
architectures, non-MQ2R DS4 routes, weights, sampling, expert count, KV mode,
and model arithmetic policy are unchanged.

## Fixture

- Model: `/home/kaden/models/deepseek-v4-flash-0731.mq2r`
- Model SHA-256: `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`
- Prompt: `benchmarks/prompts/ds4-gfx942-ar-2048.txt`
- Prompt MD5: `25e22faef15a20ae53501f1956e62b79`
- Effective context: 2,052 tokens
- Generation: 512 tokens, batch 1, greedy, thinking off, speculation off
- Experts per token: checkpoint default 6
- Requested KV: Q8; current DS4 implementation remains F32 contiguous
- Route: TP3 on three gfx1201 R9700s through `scripts/serve_harness.py`

## Mechanism gate

The finalizer channel ran two adversarial inputs on gfx1201. It compared the
three-launch incumbent to the fused finalizer and found:

- `hc_c`: 24/24 raw-bit identical in both cases
- `hc_x_in`: 4,096/4,096 raw-bit identical in both cases
- input and output guards intact; alpha, base, and stream inputs unchanged
- incumbent: 0.011058 ms/invocation
- fused finalizer: 0.005640 ms/invocation
- isolated speedup: 1.961x; two launches removed per invocation

Evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-hc-finalizer/channel.txt`

The finalizer-only product screen measured 48.2615 tok/s, +1.82% over the
47.3987 promoted median. It was kept in the composed checkpoint rather than
repeated alone. Its graph contained 8,317 blobs, down from 8,833.

## Product checkpoint

The composed candidate produced:

| Fresh process | Decode tok/s | Prefill tok/s | Graph blobs |
|---|---:|---:|---:|
| 1 | 50.860761 | 53.695960 | 8,059 |
| 2 | 50.912561 | 54.394561 | 8,059 |
| 3 | 50.601655 | 54.024984 | 8,059 |

- Decode median: **50.860761 tok/s**
- Prior promoted median: **47.398707 tok/s**
- Delta: **+3.462053 tok/s / +7.3041%**
- Decode range spread: **0.6113%**
- Prefill median: 54.024984 tok/s, +7.1932% diagnostic-only
- Route identity: three ranks, 86 peer barriers, 8,059 kernarg blobs
- Graph reduction versus prior route: 774 blobs/token

All three outputs were byte-identical to the promoted baseline:
`b6255240b6ccd34d621f152cc9898c71c340fa9393680ddb3d3efde2172b9c41`.
Every result generated 512 tokens, ended by length, contained 395 answer words,
and reported zero empty or attractor failures.

Candidate binary SHA-256:

- `hipfire`: `02856ba085eda24ba11eaf47f809e55735c764e5a25e3d0ab9ad8875d513c6ea`
- `daemon`: `769531502ac52fd0dedd0d02536b6f73a4d0a409dd98fa927ac0933b15b1e15e`

Product evidence:

- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-hc-control-finalize-product-screen/`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-hc-control-finalize-product-run2/`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-hc-control-finalize-product-run3/`

## Scope and next gate

No weight, format, top-k, sampling, KV, speculation, PM4/Redline, or long-context
change was included. No TP4 product run was included. Production TP prefill is
still token-serial and its 54 tok/s result is not a hardware ceiling.

The decode campaign remains active. Re-profile the promoted 50.86 tok/s route,
then admit only another candidate projecting at least 2% toward the 60–100
tok/s TP3 target. After decode, replace token-serial EP/TP prefill with a
batched sparse lowering before making any prefill performance claim.
