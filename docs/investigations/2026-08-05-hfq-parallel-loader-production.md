# HFQ parallel-loader production validation

Date: 2026-08-05  
Branch: `ds4-beta-staging`  
Loader commit: `c6c3c0ac393e00303d594e10c05b57db647964d4`  
Qwen first-request fix: `a2ab1607ce76d970f0709f2573decff36c6c42fb`

## Outcome

The bounded parallel HFQ reader is suitable as the default packed-expert load
path. It reduces full-model startup wall time while preserving canonical GPU
allocation and upload order. There is no environment gate on the successful
path. Overlays and shapes that cannot be planned safely retain the serial
fallback.

The initial Qwen serve measurement of 181--183 tok/s was a real user-visible
regression: `hipfire serve` called the load-only `ensure_model` path
"pre-warmed" even though the first request still had to initialize decode and
prepare retained PM4. The Qwen-only follow-up now runs a bounded resident decode
before the service reports the MQ4R model ready. This is not a kernel change and
does not affect DeepSeek, LFM, `.mq4`, or TP2 routes.

## Full-loader results

All measurements were made on `hipx`. Qwen ran on physical card 0, `gfx1100`;
DeepSeek ran on physical card 1, `gfx1151`. The source model files were on the
same local NVMe for each A/B.

| Model | Serial/default baseline | Parallel default | Wall change | Validation |
|---|---:|---:|---:|---|
| DeepSeek V4 Flash 0731 MQ2R, 82 GB | 102.30 / 100.27 s | 86.25 / 72.25 s | median 101.285 -> 79.25 s, -21.8% | decoded output exact; retained PM4 route and 15-position shadow pass |
| Qwen3.6 35B-A3B MQ4R, 18 GB | 29.33 s | 19.32 / 19.33 s | about -34.1% | decoded output exact; steady product route unchanged |
| LFM2.5 350M | 6.63 / 7.22 s | 4.47 / 4.31 s | median 6.925 -> 4.39 s, -36.6% | load-only; inference harness did not unwind cleanly, so no coherence claim |

The DeepSeek two-sample candidate spread is storage-state sensitive; the claim
is the measured median reduction, not a claim that every cold load completes in
72 seconds.

## Qwen regression closure

### Product route was not the regression

The exact historical product harness was run once with the preserved golden
binary and once with the current daemon on the same host and ROCm installation:

| Binary | HIP/HipGraph | retained PM4 | speedup |
|---|---:|---:|---:|
| sealed golden `76f2a5748` | 213.451 tok/s | 246.315 tok/s | 1.15397 |
| current loader daemon | 215.225 tok/s | 246.180 tok/s | 1.14383 |

Current retained-PM4 throughput is -0.055% from the preserved binary. The
current route is valid but not byte-identical to the old sealed tape: it has 603
dispatches and 22 symbols rather than 604 and 22. This comparison proves that
the kernels and warm product route did not lose 20--40%; it does not relabel the
current tape as the sealed historical identity.

### The first request was the regression

The same five built-in `serve_harness.py` prompts, registry sampling, Q8 state,
thinking off, and 128-token cap were run before and after the Qwen prewarm fix:

| Route | First request | Five-request average | Per-request decode tok/s |
|---|---:|---:|---|
| automatic PM4 before fix | 196.3 | 224.6 | 196.3, 228.9, 230.1, 234.3, 233.6 |
| automatic PM4 after fix | 236.7 | 238.2 | 236.7, 239.6, 237.3, 240.0, 237.4 |

The five decoded response bodies are byte-identical before and after. The first
request improves 20.6%; the battery average improves 6.1%.

The exact short fixture that produced the disputed result was also replayed:
`benchmarks/prompts/bare_factual.txt`, prompt MD5
`1d32df5f12c414d3e34c7b35b6611e6c`, 42-token context, 32 generated tokens.
It now measures 237.7 tok/s versus 183.0 before, a 29.9% recovery, and produces
a coherent Paris/Seine/Eiffel answer. This stochastic run is a direct
throughput reproduction, not the output-parity evidence; output parity comes
from the five-prompt battery above.

The explicit HIP fallback starts at 210.7 tok/s and averages 210.9 tok/s over
the same five prompts, matching the 215.225 tok/s product-control waterline
within fixture overhead.

Explicit retained AQL is not promoted: it generated incoherent text at about
205 tok/s. Repeating a committed prompt with the new prewarm deliberately
bypassed reproduced the same corruption, proving the AQL defect predates and is
independent of this fix. Automatic MQ4R serving remains retained PM4; the AQL
number is rejected correctness evidence, not a performance result.

## DeepSeek retained-route validation

The production loader completed the DS4 golden code fixture at 28.887 tok/s on
retained PM4. Route proof:

- `gfx1151`, Q8 request mode, speculation off;
- 2,320 launches, 32 unique symbols;
- 32/32 certified AQL contracts, zero fallback;
- one packet, queue 3, 57,663 command dwords;
- sequence hash `c11845041c3101e7`.

The claim-scoped `redline_daemon_harness.py` follow-up passed its 15-position
PM4/HIP shadow battery with exact output, stable 2,320-launch sequence, and all
32 AQL contracts. Its 128-context diagnostic median was 27.420 tok/s; this is
not substituted for the user-facing golden fixture.

## Implementation and blast radius

`crates/hipfire-runtime/src/hfq_parallel.rs` owns a bounded ordered reader
pipeline. Each job has its own file handle and uses positional reads directly
into the final packed host buffer. Completion may be out of order, but upload
and installation remain in canonical job order. At most the configured lane
count of host outputs is live. DeepSeek and Qwen use two independent jobs per
MoE layer; LFM uses four. Serial fallback remains available whenever the packed
plan is not applicable.

The Qwen prewarm predicate is fail-closed to loaded architecture IDs beginning
with `qwen3_5`, exact case-insensitive `.mq4r`, and single-GPU operation. It uses
the daemon's existing `bench_decode` request and fails model readiness if that
request fails. DS4 and LFM do not enter this branch.

## Evidence

Raw host evidence is preserved at:

`/home/kaden/ds4-gfx1151-evidence/2026-08-05-hfq-fast-loader-production/`

Key SHA-256 values:

- DS4 PM4 result: `ef5f7307a9600be3b60361c6d3fce50cc14d9a13a56ee5de75eb72d1338a0be2`
- DS4 PM4 serve log: `0f1d9afae3ec717f6ed0119675d4007d931da02fc9104c6d792f79b033e52ed7`
- DS4 shadow report: `52d30ff3ab28671e66a1e0f1673a7f29b0a32e126f1e0deb8d87b27ec1a4bce0`
- Qwen five-prompt before result: `9b1a6fda4bfc1f36276895bc5fc4acb602db27008d51a09b11c630f8f68bf185`
- Qwen five-prompt after result: `d6ca33f21d9a6512ccc1d0ab05eb039c72db1ae587f89b658a6de3f9a77f9752`
- Qwen exact short-fixture result: `d54dce0a1957d247a4ecfa6aa4bee36071aca6b3d13ba180f605de16bb16149c`
- current Qwen product report: `2b83947bf85e5966e39bff715054029af429ddc0a44fddf3fba23e3486b655a6`
- historical Qwen product report: `334408e7c93f886eae86f91a4f4b6488832ce810db6db1b5213a22f16e165ba9`
- validated CLI: `91aefe0e08507ffc0bacaf67cda417eb187e55d947a0bb707a9697ecbac063e4`
- validated daemon: `3c7185bd544b9192ecefa5af605c25b110abde217f71c04d5e4d932a17a5c61c`

`cargo test -p hipfire-cli` passes 125/125 tests. The first sandboxed run's 13
HTTP failures were exclusively denied ephemeral socket binds; rerunning outside
that restriction passed every test.
