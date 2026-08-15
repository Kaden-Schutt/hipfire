# DS4 gfx1151 cooperative indexer K staging

Date: 2026-08-05 UTC  
Branch: `ds4-beta-staging`  
Probe commit: `f30a487121daeb5fbf5922d917d9f987812ef010`  
Promoted route commit: `f09799c63f349d2a279e48c676094d0dabbd0cc1`

## Result

The gfx1151 DeepSeek V4 prefill indexer now stages each complete 16-slot by
128-dimension K tile once in LDS and shares it across the four existing WMMA
warps. The previous kernel loaded and converted the same K tile independently
in every warp.

The change is default-on only in the DeepSeek4 prefill path when the device is
exactly `gfx1151` and the canonical indexer WMMA shape is `H=64, D=128`. The
portable WMMA symbol remains selected on all other architectures. No weights,
quantization, sampling, expert count, KV policy, or arithmetic order changed.

| Fixture | Previous promoted | Cooperative K | Throughput delta |
|---|---:|---:|---:|
| 21,349-token NIAH | 115,326 ms / 185.12 tok/s | 110,182 ms / 193.76 tok/s | +4.67% |
| 85,693-token NIAH | 650,073 ms / 131.82 tok/s | 568,628 ms / 150.70 tok/s | +14.32% |

Both model runs were fresh-process, cold-prefill AR runs with greedy sampling,
thinking and speculation disabled, six experts per token, a Q8 request, and the
contiguous KV backend. Both reproduced the preserved control answer exactly,
passed NIAH recall 1/1, and reported zero empty, runaway, or attractor failures.
The prompt MD5s were `2e311623a082f6850a45b2ceefee9d9b` and
`1328229814512e36c4743aa3f9df0e33` respectively.

## Kernel mechanism and parity

The candidate keeps the existing four-warps-per-block mapping, Q conversion,
eight WMMA operations, accumulator layout, ReLU, per-head weights, and
fixed-order 64-head reduction. It adds a 4 KiB F16 LDS tile populated
cooperatively by the 128 threads, followed by one block barrier before WMMA.

The production-shape ABBA microbench compared every score-buffer slot as raw
F32 bits: 44,040,192 comparisons across the 2K, 8K, and 32K capacity shapes.
All live scores, causal `-inf` tails, and storage-capacity slots matched.

| Shape | Reference | Candidate | Kernel speedup |
|---|---:|---:|---:|
| capacity 2,048 / live 512 / B=1,024 | 3.481 ms | 1.850 ms | 1.882x |
| capacity 8,192 / live 5,338 / B=1,024 | 42.945 ms | 21.749 ms | 1.975x |
| capacity 32,768 / live 21,423 / B=1,024 | 175.500 ms | 87.767 ms | 2.000x |

Radiowave and the extracted code-object metadata show:

| Resource | Reference | Candidate |
|---|---:|---:|
| VGPR | 50 | 38 |
| SGPR | 21 | 22 |
| LDS | 4,096 B | 8,192 B |
| wave size | 32 | 32 |
| private/spill storage | 0 | 0 |

The candidate HSACO SHA-256 is
`dbbfeb70254af80c7e83bca8b76a9c39cb82abbdc7ece4291f877d74b0ab90aa`.

## Route certification

The supported Redline validation path passed at a 2,048-token decode context:

- stable repeated capture;
- 2,320 launches and 32 unique symbols;
- 32/32 AQL contracts;
- sequence hash `c11845041c3101e7`, unchanged from the reconciled golden route;
- 15 consecutive retained-PM4 versus HIP shadow positions bit-exact for
  logits, KV cache, and recurrent state.

The first attempt to request Redline prefill captures was preserved but is not
acceptance evidence: DS4's `bench_prefill` response omitted
`redline_capture`, so `redline_daemon_harness.py` raised `KeyError` before a
comparison. Prefill correctness is instead established by the direct all-slot
kernel parity gate and the two exact-retrieval serve-harness runs.

## Evidence

All artifacts are under:

`/home/kaden/ds4-gfx1151-evidence/2026-08-05-ds4-prefill-indexer-klds/`

Important files:

- `micro/micro-abba.txt`
- `micro/{reference,candidate}-radiowave.json`
- `micro/{reference,candidate}-readobj.txt`
- `model-screen-21k/{stdout,rows.json,serve.log}`
- `model-confirm-85k/{stdout,rows.json,serve.log}`
- `redline-shadow-supported/{stdout,report.json,daemon.log}`
- `redline-shadow/stdout` (preserved unsupported prefill-capture attempt)
- `candidate-bin/{hipfire,daemon}`

The exact promoted binaries are:

- CLI SHA-256 `a0dfc56a117da77c7313e105eea73c1db64a176f4b9b912ef78ccf5b287b7ae4`
- daemon SHA-256 `d609976fd0db1f3b4d5c490083939d49d2e0e750bf9bacb52dab2d9b979d18c1`

The central DS4 gfx1151 campaign ledger contains the complete promotion row,
including source and evidence hashes.

## Scope not claimed

This is a cold-prefill improvement. It does not claim a decode-speed change,
prefix-cache behavior, PM4 prefill acceleration, quantized-KV improvement, or
performance on Qwen, gfx1100, or any non-gfx1151 architecture. One fresh model
process was used at each long depth because the measured deltas were 4.67% and
14.32%, the isolated mechanism reproduced at all three shapes, and each 85K
confirmation costs about ten minutes.
