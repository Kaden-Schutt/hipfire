# gfx1201 DeepSeek V4 TP3 long-context bounded top-K

Date: 2026-08-08 UTC
Branch: `ds4-gfx1201-opt`
Promotion commit: `576069e872d0adb5003ab25ba1eef6fe6dca0bb8`

## Verdict

Promoted. The gfx1201 DeepSeek V4 TP3 prefill route was not merely paying for
replicated compressor state. Once compressed history exceeded K=512 it fell
through to `indexer_top_k_batched`, whose exact rank-count algorithm is
quadratic in compressed depth. The same production path on gfx1151 already
used a bounded exact merge.

An exact-gfx1201 sister symbol now performs the same stable score-descending,
source-index-ascending selection in `O(N log^2 K)`. The 21,349-token NIAH
fixture improves from 312.025 to **390.074 prefill tok/s** (+25.0%). The
previously failing 85,693-token fixture improves from 84.293 to **268.664
prefill tok/s** (+218.7%, 3.19x) and now retrieves the exact needle.

The new kernel has its own source, Rust method, module, symbol, and exact
`gfx1201` selector. The gfx1151 source/code object/selector and every Qwen,
gfx1100, gfx1151, gfx1200, gfx942, single-GPU, and non-DS4 route are unchanged.
There is no environment variable required to reach the good route.

## Baseline attribution

ROCm 7.14 `rocprofv3` observed the unmodified 21K product route through
`serve_harness.py`. The trace includes all three gfx1201 ranks.

| Kernel | Calls | Total device time | Share | Average |
|---|---:|---:|---:|---:|
| `indexer_top_k_batched` | 2,646 | 11.736 s | 9.12% | 4.435 ms |
| `indexer_relu_score_wmma_batched_f32_gfx12` | 2,646 | 4.393 s | 3.41% | 1.660 ms |

The source mechanism is stronger than the percentage alone: the portable
top-K computes every candidate's exact rank against every other candidate.
Compressed depth grows approximately with context/4, so this term grows
quadratically. That explains why a 4.0x context increase caused 14.9x prefill
wall time before this change.

A separate 85K baseline trace was skipped. The 21K trace, live source, and
shape-distributed micro made it decision-redundant; the real 85K candidate
fixture was retained as the more useful promotion gate.

## Exactness and micro screen

`test_indexer_top_k_batched_bounded` ran the portable kernel, the new gfx1201
kernel, and a host oracle in one process over the same frozen score buffers.
All 35,328 ordered i32 output slots matched. Cases include the identity
boundary, N=513, tied scores, real `-inf`, stride larger than the live bound,
and growing valid rows within a prefill chunk.

| Shape | Portable | gfx1201 bounded | Speedup | Verdict |
|---|---:|---:|---:|---|
| N=512, B=7 | 0.007161 ms | 0.007124 ms | 1.005x | pass |
| N=513, B=7 | 0.053190 ms | 0.029789 ms | 1.786x | pass |
| N=2,048, B=7 | 0.499669 ms | 0.060441 ms | 8.267x | pass |
| N=5,338, B=32 | 3.727152 ms | 0.149623 ms | 24.910x | pass |
| N=8,192, B=16 | 7.547132 ms | 0.204222 ms | 36.955x | pass |

## Product results

Both arms are real `scripts/serve_harness.py` requests using the committed
NIAH fixtures, TP3 on devices 0,1,2, `max_seq=1,048,576`, greedy sampling,
checkpoint-default top-k 6, speculation off, thinking off, and at most 64
generated tokens. Product runs used the direct daemon, not the profiler
wrapper.

| Fixture | Route | Prefill wall | Prefill | Decode | Recall |
|---|---|---:|---:|---:|---:|
| 21,349 tokens | baseline | 68,420.852 ms | 312.025 tok/s | 20.489 tok/s | 1/1 |
| 21,349 tokens | bounded | 54,730.628 ms | **390.074 tok/s** | 41.148 tok/s | **1/1** |
| 85,693 tokens | baseline | 1,016,614.521 ms | 84.293 tok/s | 30.652 tok/s | 0/1 |
| 85,693 tokens | bounded | 318,959.315 ms | **268.664 tok/s** | 30.767 tok/s | **1/1** |

The 21K decoded answer is byte-identical to the baseline. At 85K the old route
misspelled the needle as `mauve-velrapocior-7741`; the promoted route returns
`mauve-velociraptor-7741` exactly. Both promoted runs stopped normally with
zero empty responses, runaways, or attractor failures.

Decode changes are reported because the harness records them, but this is a
prefill-only promotion and no AR/decode gain is claimed.

## Mechanism confirmation

The matched post-change 21K profiler run observed the new symbol exactly 2,646
times:

| Route | Top-K total | Top-K share | Average/call | Profiled prefill |
|---|---:|---:|---:|---:|
| portable | 11.736 s | 9.12% | 4.435 ms | 274.202 tok/s |
| bounded | 0.657 s | 0.56% | 0.248 ms | 317.754 tok/s |

The top-K stage falls by **94.4%** in the full trace. The remaining dominant
context-growing kernels are the indexer-score WMMA (3.75%) and tiled top-K KV
gather (2.14%), but neither was changed in this checkpoint.

## Validation and identity

- `cargo check -p rdna-compute --example
  test_indexer_top_k_batched_bounded -p hipfire-arch-deepseek4`: pass;
- raw-order GPU parity and host oracle: pass, five shapes;
- real 21K NIAH generation: pass, recall 1/1;
- real 85K NIAH generation: pass, recall 1/1;
- daemon SHA-256:
  `580de5d549bef3ae26292e229f2c351a074d053a1005af234e565910683633f9`;
- model SHA-256:
  `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`.

## Evidence

- Baseline and post-change traces:
  `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-08-gfx1201-tp3-longctx-profile/`
- Raw-order micro:
  `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-08-gfx1201-bounded-topk-micro/`
- Product NIAH runs:
  `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-08-gfx1201-bounded-topk-product/`

Key SHA-256:

- micro log: `528fb4535e0e352222b2c6b268d22b15db2a020909a19431ad73498cf8f08d5d`;
- 21K product JSON: `38721a9281109739aef7f6fc186c7e5c8975c6848d3679523fa5461ec155779b`;
- 85K product JSON: `dc51f6be72b0ddf740fc6b8a33faddd896d83d88ad18e32b90fe53174ca56c74`;
- baseline stats CSV: `0d5478dc752b9b3147acffdb4b8665e79df7f1362fb3abf371b572314561648f`;
- bounded stats CSV: `064bf68cb9a0811445bf760f7f3ea89ca3045395c831463f1673cbd9e8aa9e47`.

## Skipped

No repeated product samples were collected because both effects are decisive
and the 85K request costs several minutes. No 85K baseline profiler trace,
1M-token prefill, quantized KV, TP4, DSpark, retained PM4, weight/format,
sampling, expert-count, or top-k change was attempted. No gfx1100, gfx1151,
gfx1200, gfx942, Qwen, or single-GPU runtime was re-run because the new source
and selector are exact-gfx1201-only and the gfx1151 code object is untouched.
