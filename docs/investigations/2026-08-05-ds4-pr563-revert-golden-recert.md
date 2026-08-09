# DS4 PR #563 revert golden recertification

Date: 2026-08-05  
Branch: `ds4-beta-staging`  
Restoration commit: `25111034e32b38f085bdd5e964a8d08de36f70e2`  
Restored tree: byte-identical to pre-merge parent
`6464d6081e08993161748d7b419edc6995473c35`

## Result

The short code-fixture performance and decoded-output golden set reproduces
after reverting PR #563. Both serving arms produced byte-identical output with
MD5 `e49b9893a207d8a698eb17fdca13db51`.

| Arm | Pre-merge golden | Post-revert | Delta |
|---|---:|---:|---:|
| DSpark, direct HIP, adaptive B | 37.3264 tok/s | 37.0646 tok/s | -0.701% |
| AR, retained PM4 | 28.8678 tok/s | 29.0118 tok/s | +0.499% |

Fixture: committed prompt
`benchmarks/prompts/ds4_dspark_genre_code.json`, prompt-text MD5
`d782138f5bc8bbbd234ca8e4b17cace9`, 25 prompt tokens, 128 generated
tokens, batch 1, greedy, Q8 request mode, six experts per token, reasoning
off. These are fresh-process diagnostic reproductions of the historical
golden rows, not 2,048/512 acceptance claims.

DSpark retained the historical `tau=2.0238095238095237`, 42 verify windows,
and 67% acceptance. The AR serving run recorded 126 retained PM4 replays.

## Correctness and route proof

- AR Redline shadow: 15 positions, bit-exact HIP/PM4 logits, KV and recurrent
  state; 2,320 launches, 32 symbols, 32/32 AQL contracts, sequence hash
  `c11845041c3101e7`.
- AR prepared route: one packet, queue 3, 57,663 dwords, 2,319/2,319
  boundaries covered, zero fallback launches.
- DSpark verify shadow: context 2,048, fixed B=3, 15 positions, bit-exact
  ordinary HIP, capture-safe HIP, captured blob and retained PM4 state;
  3,602 launches, 40 symbols, 40/40 AQL contracts, one packet, queue 3,
  87,006 dwords, positions 2,048 through 2,090.
- The current pre-#563 DSpark verify sequence hash is `336244b42222753a`.
  The earlier `d97994f00` report recorded `b42954f3f1a3a49b`; the intervening
  pre-merge branch work changed the sequence identity without changing launch
  count, symbol count, prepared geometry, covered positions, or exact state.
  This recertification does not attribute that pre-#563 route-version change.

## Startup timing observation

PR #563 introduced a real startup concern, but the post-revert evidence does
not support treating all load-time variance as caused by that merge. On the
restored binary, identical Redline harness loads were observed both around
90 seconds and beyond the 120-second response timeout. The completed serving
runs took 168.03 seconds wall for AR (including PM4 preparation and generation)
and 107.00 seconds for DSpark (including target/sidecar load and generation).

Startup therefore remains a separate measurement and loader-pipeline problem.
It is not part of the decode-throughput verdict above.

## Frozen identities

- Target MQ2R SHA-256:
  `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`
- P3-aligned DSpark sidecar SHA-256:
  `bc695a000643801d26e5ae96c9f4ac4c222a36d9db40566f4cc1de0e9d3d5d2e`
- Restored daemon SHA-256:
  `cd4ab2fb91abebfe44741c52b28c0f0ffc1f18d46d0df759f87549db3aa6a2c0`
- Restored CLI SHA-256:
  `256032e2fa160077804da1eb3da6a477c4ce9c1904eef222977de19f626b5a75`

## Evidence

Canonical evidence is preserved on hipx at:

`/home/kaden/ds4-gfx1151-evidence/2026-08-05-ds4-pr563-golden-recert/`

Key artifact SHA-256 values:

- `ar-pm4-pre563-restore-result.json`: `80c77de50c8c86e7c2838dc867f81e0faf00c1805364c432a41d8c5d6394f2cb`
- `ar-pm4-pre563-restore-serve.log`: `dfea29976d085fe335b24bba1d81446a816a99aeec68c660d0d449403ed817da`
- `dspark-pre563-restore-result.json`: `6b514af217e671dc2050c42fdf7259cc5126c099f85f6e60f0188ad5b069258c`
- `dspark-pre563-restore-serve.log`: `8e32af4d1f1a5e1a2ad63dad74387c8a8a466bfcd1fbed39216f3be2d883b38f`
- `redline-shadow15-ctx2048-pre563-restore-retry.json`: `af751471edb8dc74b921cb1c0b05bc5bb4d7c617338a94404f97b94ddc3ce783`
- `dspark-shadow15-ctx2048-pre563-restore.json`: `3ea8a7b95928b9ec7228264b6610c610c9114216bb38ee256e5359f3f05ee216`

The failed 120-second AR load attempt and the earlier post-#563 load failures
remain preserved in the same evidence directory; they are not golden rows.
