# DeepSeek V4 gfx1201 TP3 E8 decode screen checkpoint

Date: 2026-08-07  
Branch: `ds4-gfx1201-opt`  
Candidate checkpoints: `75a75570c`, `cbfcc5392`, `58541b812`  
Parent product line: `425d3f1a9` (`53.376417` tok/s median)  
T1024 campaign line: `54.903757` tok/s median, pending default approval

## Verdict

Three occurrence-weighted candidates were screened against the corrected TP3
decode shape inventory. None independently cleared the campaign's 2% product
bench admission threshold:

| Candidate | Numerical result | Mean projected saving/rank/token | Disposition |
|---|---:|---:|---|
| Shared-activation pair | Raw-bit exact | 0.195744 ms | Park as a ~1.0% bundle ingredient |
| Four-group prefetch | Raw-bit exact | 0.010717 ms | Reject as flat |
| Late block scaling | FP32 grouping differs | -0.015555 ms | Reject as slower and non-identical |

No candidate was wired into DeepSeek dispatch, and no product benchmark was
spent on a sub-threshold micro result. All methods and kernels are exact
`gfx1201` experiments; other architectures and model routes are unchanged.

## Fixture and occurrence map

The microbench working set rotates up to 160 MiB of MFP4G32E8SOA weights per
case. Shapes and occurrences come from the corrected 2,052-token TP3 decode
trace:

- router: M=256, K=4096, 43 calls/rank/token
- shared up: M=768 on ranks 0/1 or M=512 on rank 2, K=4096, 86 calls
- shared down: M=4096, K=768 on ranks 0/1 or K=512 on rank 2, 43 calls
- O-LoRA down: M=4096, K=8192, 43 calls
- WQ/indexer: M=12288/8192, K=1024, 43/21 calls
- LM head: M=129280, K=4096, one call on rank 0

The production fixture remains
`benchmarks/prompts/ds4-gfx942-ar-2048.txt` (MD5
`25e22faef15a20ae53501f1956e62b79`), effective context 2,052, generation
512, batch 1, greedy, top-k 6, thinking/speculation off, TP3 on three gfx1201
R9700 devices. No product run was admitted by this checkpoint.

## Shared-activation pair

One wave computes two independent projections that share X while retaining
each matrix's incumbent two-chain accumulation and wave32 reduction.

| Pair | Serial | Paired | Speedup | Saved/rank/token |
|---|---:|---:|---:|---:|
| shared w1/w3, ranks 0/1 | 13.687409 us | 10.893122 us | 1.2565x | 0.120154 ms |
| shared w1/w3, rank 2 | 12.438446 us | 9.779500 us | 1.2719x | 0.114335 ms |
| main/indexer WQ, ranks 0/1 | 28.665858 us | 24.828715 us | 1.1545x | 0.080580 ms |
| main/indexer WQ, rank 2 | 24.971222 us | 21.575611 us | 1.1574x | 0.071308 ms |

All 38,400 compared outputs matched at raw bits. Evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-e8-shared-pair/`.

## Four-group prefetch

Four-group lookahead retained the incumbent even/odd accumulator chains but
won only selected small shapes. Regressions on shared-down and O-LoRA down
canceled those gains. The three-rank mean projection was only 0.010717 ms,
effectively flat. All 181,568 comparisons matched at raw bits. Evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-e8-prefetch4/`.

## Late block scaling

This candidate formed each eight-term E8 coordinate dot before applying its
block scale. It reduced static instructions from 344 to 318 and waits from 87
to 77 while retaining 48 VGPR, 18 SGPR, wave32, zero spills, and zero private
segment. That ISA simplification did not translate:

- only 44,018 of 181,568 outputs matched at raw bits;
- max absolute error reached 0.0078125 and max relative error 0.0120482;
- ranks 0/1 project -0.0121 ms/token before the rank-0 LM head;
- rank 2 projects -0.0228 ms/token;
- the three-rank mean projects -0.015555 ms/token.

The short-K shared-down cases alone regressed 0.0406-0.0423 ms/rank/token.
The candidate is rejected before product wiring. Evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-e8-late-scale/`.

## Next gate

The ungrouped E8 family remains hot, but these arithmetic variants do not
provide the required next step. The follow-up graph-resident barrier/HC
composition was screened and rejected: repeating peer-visible spin handshakes
across every HC output block made the 86-boundary sequence 4.14x slower despite
removing 516 graph nodes. See
`2026-08-07-gfx1201-ds4-tp3-graph-barrier-screen.md`. The parked
shared-activation pair remains only a bundle ingredient; projections are not
added arithmetically into a claim.
