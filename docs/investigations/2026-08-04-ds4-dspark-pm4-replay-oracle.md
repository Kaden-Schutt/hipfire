# DS4 gfx1151 DSpark retained-PM4 replay oracle

Date: 2026-08-04
Branch: `ds4-cdna-test-fail`
Base commit: `7d0535aaa7b8919252c07ba48f34c239223f204c`

## Verdict

The DSpark-specific retained-PM4 verify route is bit-exact, but it is slower
than ordinary HIP and must not be enabled for production. The opt-in
`HIPFIRE_DEEPSEEK4_DSPARK_VERIFY_PM4=1` route remains a diagnostic path.

The replay oracle and canonical indexer-RoPE lowering are useful infrastructure
and are retained. Further DSpark performance work should continue on the
direct-HIP cooperative-dispatch path.

## Correctness oracle

Fixture: DeepSeek V4 Flash 0731 MQ2R target, P3-aligned E8 DSpark sidecar,
gfx1151, Q8 KV, verify batch 3, positions 128 through 170, 15 iterations.

All four arms matched exactly:

- ordinary HIP verify;
- capture-safe HIP verify;
- exact captured-HIP blob replay;
- prepared retained-PM4 replay.

The comparison covers target picks, logits, 216,530,944 bytes of KV state,
12,206,080 bytes of recurrent state, DSpark captures and streams, and the
inactive-row guard region. The prepared route recorded:

- 3,602 launches and 40 unique kernel symbols;
- sequence hash `b42954f3f1a3a49b`;
- 40/40 valid AQL contracts;
- one PM4 packet on queue 3 containing 87,006 dwords;
- 15 observed replays with no failure.

The initial oracle exposed a raw-bit mismatch in the ratio-4 indexer KV cache:
ordinary HIP used the plain RoPE symbol while capture-safe execution used the
at-slot YaRN symbol with the plain settings. The DS4 indexer path now uses the
same at-slot symbol in both modes. A dedicated `ext_factor == 0` branch retains
the plain arithmetic expression while leaving the normal YaRN path unchanged.

## Serving-path performance

Fixture: committed code prompt `d782138f5bc8bbbd234ca8e4b17cace9`, 25 prompt
tokens, 128 generated tokens, batch 1, temperature 0, adaptive DSpark block,
Q8 KV, exact target and E8 sidecar, fresh harness-owned daemon per arm.

First request:

| arm | decode tok/s | tau | result |
|---|---:|---:|---|
| ordinary HIP | 37.0574 | 2.0238 | coherent, length-capped |
| retained PM4 | 30.7105 | 2.0976 | byte-identical output, length-capped |

The PM4 log proves real retained replay rather than fallback. Adaptive routes
for batches 2 through 6 were captured, all symbols certified, and each route
reported an observed replay. The first request nevertheless includes those
five one-time route constructions.

To isolate steady-state replay, each arm then served the same committed prompt
twice in one daemon. The second request compared the same controller state and
the same `tau = 1.54`:

| arm | second-request decode tok/s | result |
|---|---:|---|
| ordinary HIP | 32.2583 | coherent |
| retained PM4 | 29.0610 | byte-identical output |

That is a 9.91% retained-PM4 regression after route warmup. The lower absolute
rate of both second requests is the known stateful adaptive-controller effect;
the paired second-request comparison is still valid because both arms start
from a fresh daemon and receive the same two requests in the same order.

## Evidence

- `/home/kaden/ds4-gfx1151-evidence/2026-08-04-ds4-dspark-replay-oracle/`
- `/home/kaden/ds4-gfx1151-evidence/2026-08-04-ds4-dspark-serving-pm4/`

The canonical oracle report is
`canonical-rope-b3-pos15.json` (SHA-256
`27a90edd2c14471bc68b0772d2b9bb471accbb7c2bb387c417def53af911aa73`).

## Skipped

- No promotion or default change for retained PM4.
- No long-context claim; this fixture is the 128-token code waterline.
- No weight, sidecar, quant, sampling, expert-count, or KV-format change.
- No arithmetic-changing kernel optimization.
