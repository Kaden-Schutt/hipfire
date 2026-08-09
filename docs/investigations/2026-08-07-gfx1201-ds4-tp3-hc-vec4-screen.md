# gfx1201 DS4 TP3 peer-HC vec4 screen

Date: 2026-08-07  
Candidate: `101a6120e`  
Route: DeepSeek V4 Flash 0731 MQ2R, TP3, exact gfx1201

## Hypothesis

The shipping `hc_mix_4stream_peer3_gfx1201` assigns one thread to each
`(stream_out, hidden_index)` pair, so all four output streams independently
reload the four input streams and three peer transforms. The candidate
`hc_mix_4stream_peer3_vec4_gfx1201` assigns one thread to a hidden index,
loads the shared inputs once, and computes all four output streams while
preserving the incumbent FP32 accumulation order.

The product route was not changed. The screen held the standalone producer
release and two-peer acquire nodes constant in both arms and captured the full
86-boundary TP3 graph shape.

## Result

| arm | median graph replay |
|---|---:|
| incumbent | 1.004012 ms |
| vec4 | 1.006298 ms |

- Speedup: `0.9977x`
- Saved time: `-0.002287 ms`
- Projected product effect at the 54.903757 tok/s T1024 line: `-0.013%`
- Raw-bit comparisons: `49,152`, all equal
- Graph nodes: `774` in each arm
- Trials: 9 ABBA, 16 replays per trial

The lower traffic does not offset the reduction from 64 to 16 HC blocks per
rank. This misses the required 2% projection gate by two orders of magnitude,
so no product dispatch, model load, or product benchmark was run. The
candidate is rejected and reverted.

Raw evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-tp3-hc-vec4/run.log`

## Scope skipped

No product route, product benchmark, second process, prefill, long-context,
TP4, weights, format, sampling, top-k, KV, speculation, Redline, or adjacent
architecture was changed or measured.
