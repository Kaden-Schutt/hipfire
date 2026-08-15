# DS4 MQ2R automatic PM4 admission and DSpark thinking

Date: 2026-08-05 UTC  
Branch: `ds4-beta-staging`  
Admission commit: `fefaea3ec9c208a0190dca3441626666884968fb`

## Result

DeepSeek V4 Flash 0731 `.mq2r` now selects the certified retained-PM4 route
automatically on exact `gfx1151` when no drafter is installed. An installed
DSpark drafter leaves the request on the speculative direct-HIP path. The
policy is also limited to single-GPU `deepseek4`; the existing `.mq4r` policy
is unchanged.

The same commit passes a low-effort, uncapped DSpark-thinking request through
the user-facing serve path. The model emitted a 33-word reasoning section,
then the correct concise answer, and stopped normally.

## Automatic AR admission

The production proof deliberately removed `HIPFIRE_REPLAY_BACKEND` and
`HIPFIRE_REPLAY_MANUAL_CAPTURE` from the serve environment. With
`--speculation off`, the daemon reported:

```text
[redline] enabling fail-closed retained default on gfx1151 \
  (model_arch=deepseek4, drafter=off, transport=pm4)
HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4 position=35 \
  request_id=chatcmpl-936597-1 replays=43
```

The committed arithmetic fixture completed with `finish=stop`, 33 prompt
tokens, 45 generated tokens, the expected `$66`, and zero empty, runaway,
attractor, or retrieval failures. Its decode rate was 28.535 tok/s; that is a
one-request route diagnostic, not a competitive benchmark claim.

The required retained-replay correctness gate passed separately:

- 15 consecutive PM4-versus-HIP positions bit-exact;
- logits, KV cache, and recurrent state equal;
- 2,320 launches and 32 unique symbols;
- 32 AQL contracts;
- stable sequence hash `c11845041c3101e7`;
- harness verdict `pass=true`.

The admission predicate has focused unit coverage for all of its negative
edges: DSpark present, gfx1100, Qwen architecture, pipeline parallelism, and a
non-`.mq2r` artifact all remain disabled. The existing `.mq4r` default tests
continue to pass.

## DSpark thinking

The current-commit serve run used:

- committed fixture `benchmarks/prompts/ds4_0731_thinking_low.json`;
- prompt MD5 `37d74e9455aaebc9c490a4925eab8668`;
- `--speculation dspark`;
- parent-model `reasoning_effort=low`;
- uncapped thinking budget;
- registry `general` sampling, seed 1;
- Q8 request mode, contiguous DS4 cache path;
- `max_seq=32768`, with automatic compressed-cache VMM growth to the
  advertised 1,048,576-token context.

The result was:

- `finish=stop`;
- 33 prompt tokens, 112 generated tokens;
- 33 reasoning words and 25 answer words;
- expected `$66` recall 1/1;
- zero empty, runaway, attractor, or retrieval failures;
- DSpark confirmed by `drafter=dspark`, 40 speculative windows, 61%
  acceptance, and tau 1.825;
- no automatic-Redline admission marker.

The decoded output was:

```text
:The item costs $80 reduced by 25%:
$80 x 0.75 = $60

Then add 10% sales tax:
$60 x 1.10 = $66

**Final price: $66**
```

The exact current-commit row reported 34.616 tok/s. An earlier run on the
pre-admission binary produced byte-identical output at 25.096 tok/s. Those are
single-sample functional diagnostics with an unresolved performance spread;
neither is promoted as a DSpark-thinking throughput claim.

## Work not repeated

The 2,048/512 AR fixture already exists under
`/home/kaden/ds4-gfx1151-evidence/2026-08-04-ds4-ar-pm4-2048x512-g1-ab/`.
It is a one-run synthetic PP/TG matrix, not a multi-sample prompt acceptance
campaign:

| Arm | Prefill | Decode at 2,048/512 |
|---|---:|---:|
| default | 26.2 tok/s | 26.8781 tok/s |
| two-stage | 25.9 tok/s | 27.7206 tok/s |

The files are preserved as `default.json` and `twostage.json`, SHA-256
`c71e4a0624110272b4e4252a233321f0d4153c586aeab29644fa66f3608dd0b7`
and `070f7ae40f1f95b571d0e3ae7550763bbd28fa96596fbe78109ec821fa061919`.
They were not rerun.

Real long-context correctness was also not rerun. The committed 85,693-token
NIAH result already has recall 1/1, `finish=stop`, coherent byte-identical
output, and zero empty, runaway, or attractor failures. See
`docs/investigations/2026-08-05-ds4-gfx1151-prefill-e8-coop.md`.

## Identity and evidence

- daemon SHA-256:
  `ee7609bf08b01137332195a5119f74301e456d2e5960c0d8eef30b4995c30aa3`
- CLI SHA-256:
  `91aefe0e08507ffc0bacaf67cda417eb187e55d947a0bb707a9697ecbac063e4`
- target model SHA-256:
  `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`
- DSpark sidecar SHA-256:
  `bc695a000643801d26e5ae96c9f4ac4c222a36d9db40566f4cc1de0e9d3d5d2e`

Evidence roots on hipx:

- `/home/kaden/ds4-gfx1151-evidence/2026-08-05-ds4-mq2r-auto-pm4/`
- `/home/kaden/ds4-gfx1151-evidence/2026-08-05-ds4-dspark-thinking-low/current-fefaea3ec/`

Key evidence SHA-256 values:

- automatic AR result:
  `40551791bacb2403251a1a27bf9148d96d649355b0119e2dc008607575226acb`
- automatic AR serve log:
  `7f07ac9258a98f3966b3cf523eff6d601dac01d8225a19949907d8478d386f38`
- 15-position shadow report:
  `2a25524bd3020f73d33374e5dab30944620d5eda10fd8fdda6ff0eb3d71bb990`
- 15-position shadow log:
  `76f61a1cc9dcf450d724c6e1556d20db89766f3e3e3a207ab2eb7d8553edcec3`
- current DSpark-thinking result:
  `ad9ee3842b314fbff09707a1cc5880f40e2bc738c90c155a60129d2809586d1f`
- current DSpark-thinking serve log:
  `95542c350cfc0efefc7a2284528f9942810badddf6293ca1a46905854f5d0746`

