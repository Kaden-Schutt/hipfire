# DS4 beta staging golden-set recertification

Date: 2026-08-04  
Branch: `ds4-beta-staging`  
Corrected commit: `d97994f00548890a48af01ac2de6b23124cc1fa7`

## Result

The short code-fixture golden set reproduces on the reconciled staging branch.
Both serving arms decode byte-identical output with MD5
`e49b9893a207d8a698eb17fdca13db51`.

| Arm | Historical | Reproduced | Delta |
|---|---:|---:|---:|
| DSpark, direct HIP | 37.308 tok/s | 37.3264 tok/s | +0.049% |
| AR, retained PM4 | 28.8809 tok/s | 28.8678 tok/s | -0.045% |

Fixture: committed prompt
`benchmarks/prompts/ds4_dspark_genre_code.json`, prompt MD5
`d782138f5bc8bbbd234ca8e4b17cace9`, 25 prompt tokens, 128 generated
tokens, batch 1, greedy, Q8 request mode, six experts per token, reasoning
off. These are fresh-process diagnostic reproductions of the historical
golden rows, not 2,048/512 acceptance claims.

## Correctness and route proof

- AR Redline shadow: 15 positions, bit-exact logits, KV, and recurrent
  state; 2,320 launches, 32 symbols, 32/32 AQL contracts, zero fallbacks,
  sequence hash `c11845041c3101e7`.
- AR prepared route: one packet, queue 3, 57,663 dwords, 2,319/2,319
  boundaries covered, 126 retained replays in the serving run.
- DSpark verify oracle: 15 positions, bit-exact ordinary HIP,
  capture-safe HIP, captured blob, and retained PM4 state; 3,602 launches,
  40 symbols, 40/40 AQL contracts, zero fallbacks, sequence hash
  `b42954f3f1a3a49b`.
- The DSpark oracle fixes verify batch size at B=3 to certify one route.
  Product serving remains adaptive-B and produced the 37.3264 tok/s row.

## Regression found during recertification

The first staging AR request failed with
`duplicate gen_start after contract already latched`. Commit `e8d68c0dd`
had added an early DS4 AR `gen_start`, while the reconciled base already
emitted `gen_start` immediately before the first token. Commit `d97994f00`
removes the duplicate helper and returns the daemon source exactly to the
pre-`e8d68c0dd` implementation for that file. The focused runtime and CLI
stream-contract tests pass.

The first DSpark timing was also excluded: the new worktree had no warm
gfx1151 code-object cache and JIT-compiled 264 cache files. The preserved
cold-JIT row is 7.7825 tok/s and is not a product result.

## Frozen identities

- Target MQ2R SHA-256:
  `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`
- P3-aligned DSpark sidecar SHA-256:
  `bc695a000643801d26e5ae96c9f4ac4c222a36d9db40566f4cc1de0e9d3d5d2e`
- Corrected daemon SHA-256:
  `e1ecf75d7566d99b3a6e89da1059480e726d24b3d25a169a01fd914c7a554384`
- CLI SHA-256:
  `94470dace14fea040fd0d97e896ce7438996a7cc4955e197dc14f8a8f165893c`

## Evidence

Canonical evidence is preserved on hipx at:

`/home/kaden/ds4-gfx1151-evidence/2026-08-04-ds4-beta-staging-golden-recert/`

Key artifact SHA-256 values:

- `dspark-d979-result.json`: `60d19d1003e19cc589fd620f9d49d7624c9ea94f29e638694ea03e0079c07b30`
- `dspark-d979-serve.log`: `777e4545244ecd2883add84655839907892343424490255de8c63a5eef20b957`
- `ar-pm4-fixed-result.json`: `24731ee8a94f0ebf8bba8e8ac9094495d13db282b0770da1159994ba097f341d`
- `ar-pm4-fixed-serve.log`: `6171bd1153f74621286749fc6178fdb7ff4fa18a4e8c204308e608e59dae9312`
- `redline-shadow15.json`: `45a9052a9114f0f609569bc1d952bed2d2bfeb9b2052a8435fc9b35682082cc1`
- `dspark-shadow15.json`: `0988a8639d5c55f598e030fb66bbcfb8fbe0fb246a8dddfb531c5de3890eb2df`

## Remaining certification work

This checkpoint does not certify the 2,048/512 product fixture, long-context
growth, thinking/multi-turn behavior, or automatic `.mq2r` retained-route
admission. The AR result explicitly selected the Redline arm; product-default
automation must be wired and certified separately rather than hidden behind a
new environment gate.
