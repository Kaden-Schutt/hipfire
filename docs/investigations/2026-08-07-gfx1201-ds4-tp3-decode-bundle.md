# gfx1201 DS4 TP3 decode checkpoint bundle

Date: 2026-08-07

Campaign line: `54.903757 tok/s` with the selected T1024 HC-control
candidate (`HIPFIRE_HC_CTRL_T1024=1`).

## Verdict

Rejected at the product continuation gate.

The checkpoint composed three raw-bit-exact exact-gfx1201 mechanisms:

1. cooperative 64-byte LDS staging in owned MQ2-Lloyd down;
2. one shared-activation E8 launch for shared-expert w1/w3; and
3. one shared-activation E8 launch for main/indexer Q-LoRA B on ratio-4
   layers.

The occurrence-weighted micros projected about 2.4% end-to-end. The canonical
product screen instead measured **55.207285 tok/s**, only **+0.303528 tok/s /
+0.5528%** over the `54.903757 tok/s` line. This is a valid product win but
does not clear the campaign's 2% continuation threshold, so no second or third
product process was spent.

## Correctness and route proof

- Fixture: `benchmarks/prompts/ds4-gfx942-ar-2048.txt`
- Prompt MD5: `25e22faef15a20ae53501f1956e62b79`
- Effective context / generation: 2052 / 512
- Batch: 1
- Sampling: greedy, checkpoint-default top-k 6
- Thinking / speculation: off / off
- Device route: TP3 on devices 0,1,2, all exact gfx1201
- Output SHA-256:
  `b6255240b6ccd34d621f152cc9898c71c340fa9393680ddb3d3efde2172b9c41`
  (byte-identical to the campaign baseline)
- Answer: 395 words, non-empty, no attractor, finished at the fixed length
- Graph: 3 ranks, 86 barriers, **7,157 kernarg blobs**

The T1024-only graph carried 7,349 blobs. The 192-node reduction is exactly
the expected 129 shared-expert pair nodes plus 63 main/indexer-Q pair nodes,
proving the product daemon exercised both pair routes. The daemon was built
from candidate commit `1e4699c93`, contained both candidate kernel symbols,
and had SHA-256
`94c94fc29edfbc1b4875cad2e064b7da6aa64e0502102ea89643019913721b07`.

## MQ2 down mechanism

Three fresh-process micros compared the current exact-gfx1201 EP kernel with
the LDS sister at the production owned-2-of-6 shape:

| Metric | Incumbent | LDS |
|---|---:|---:|
| Median time | 21.016 us | 15.400 us |
| Speedup | 1.000x | 1.365x |
| VGPR / SGPR | 93 / 38 | 74 / 24 |
| LDS | 0 B | 64 B |
| Spills / private | 0 / 0 | 0 / 0 |

All 73,728 compared outputs matched at raw bits. The 43-layer standalone
projection is 0.24149 ms/token, or 1.326% at the campaign line. The micro
kernel and channel remain available as a parked bundle asset; only product
routing is reverted.

## Evidence and skipped work

Durable evidence:

- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-mq2-down-lds/`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-tp3-decode-bundle/`

Skipped: second/third product processes, broad serve battery, prefill tuning,
long-context, TP4, weight/format/top-k/sampling/KV/speculation changes,
Redline/PM4, and every adjacent architecture. The first coherent product
sample was already far below the 2% continuation gate.
