# DeepSeek V4 gfx1201 TP3 graph barrier screen

Date: 2026-08-07  
Branch: `ds4-gfx1201-opt`  
Candidate checkpoint: `8eabe8ded`  
Campaign line: `54.903757` tok/s with the selected T1024 HC control candidate

## Verdict

Reject the graph-resident TP3 barrier/HC-consumer fusion before product
dispatch. It removed 516 graph nodes from the 86-boundary, three-rank micro
but made the captured sequence 4.14x slower:

| Arm | Nodes | Median/replay |
|---|---:|---:|
| incumbent store + wait2 + HC | 774 | 1.011195 ms |
| HC with embedded peer handshake | 258 | 4.188108 ms |

The candidate regressed by 3.176913 ms per replay, projecting -17.442% against
the current product token time. It therefore received no model-level product
benchmark.

## Mechanism

The experiment preserved the fixed-rank transform reduction and replaced each
standalone TP3 signal-store and wait2 pair with a system-scope release/acquire
handshake at the start of the HC consumer. Every output block had to perform
the handshake because a single designated block cannot make its acquire visible
to the rest of a non-cooperative grid.

That was the wrong composition point. Node count fell by two thirds, but each
of the 64 HC output blocks per rank executed peer-visible atomics and spin
loads. The generated kernel was not suffering from resource collapse:

| Resource | Incumbent HC | Fused HC |
|---|---:|---:|
| VGPR | 12 | 17 |
| SGPR | 22 | 26 |
| static instructions | 85 | 121 |
| wait instructions | 25 | 30 |
| spills/private segment | 0 | 0 |

The regression is structural rather than an occupancy accident. Future graph
work must keep the peer handshake single-workgroup or move only the producer
release into a producer completion point; it must not repeat system-scope
spinning across the HC output grid.

## Correctness and fixture

The micro captured 86 boundaries on each of three gfx1201 R9700 devices and
compared all 49,152 output floats at raw bits. The outputs were identical.
Nine ABBA trials with 16 graph replays per arm produced the medians above.

Evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-tp3-hc-graph-sync/`.
The bundle contains the complete micro log and both Radiowave reports.

## Next gate

Screen a single-workgroup, grid-stride HC consumer on the same 86-boundary
fixture before attempting any further barrier fusion. If its compute cost is
competitive, it provides a legal place for one peer acquire per rank and may
remove both graph nodes without a grid-wide synchronization primitive. If it
cannot project at least 2% end-to-end, close this graph-fusion family and move
to the next occurrence-weighted decode hotspot.
