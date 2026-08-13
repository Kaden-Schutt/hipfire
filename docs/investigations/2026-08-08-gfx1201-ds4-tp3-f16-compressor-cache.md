# DeepSeek V4 0731 MQ2R gfx1201 TP3 selectable F16 compressor cache

Date: 2026-08-08 UTC  
Branch: `ds4-gfx1201-opt`  
Model SHA-256: `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`

## Verdict

DeepSeek V4 now tells the truth about its cache selector. The registry default
is `f32`, matching the implementation that shipped throughout the campaign.
An explicit `f16` route is available for the exact gfx1201 MQ2R TP3/TP4 path.
DS4 `q8` remains unimplemented and fails closed instead of silently running an
F32 cache under a Q8 label. Qwen registry defaults and runtime paths are
unchanged.

The selectable F16 route is coherent at the canonical 2K fixture and at real
21K and 85K NIAH depth. It halves the logical size of the 62 replicated
compressor/indexer VMM tensors from 14,428,405,760 to 7,214,202,880 bytes per
rank. On three 34.2 GB R9700 ranks it physically admits 475,136 tokens and
atomically rejects 491,520. It therefore does **not** make replicated TP3 a 1M
route by itself. The matched F32 bracket passes 229,376 and rejects 245,760,
so F16 provides just over twice the measured physical context capacity.

This is a capacity route, not yet a speed promotion. At a declared 1M maximum,
the loader conservatively selects prefill B=128 and leaves much of the read path
widening stored halves to F32. The immediate optimization target is to restore
B=512 or higher for F16 and keep gathered compressed K/V in native F16 through
the gfx1201 WMMA consumers while preserving F32 commit, softmax, and
accumulation arithmetic.

## User-facing selection

- Omitted `--kv` resolves `deepseek-v4-flash:mq2r` to `f32` from the registry.
- `--kv f16` selects the exact-gfx1201 MQ2R TP3/TP4 F16 compressor cache.
- `--kv q8` rejects with an explicit unsupported-mode error for DS4.
- The harness no longer injects its historical `fwht3` default over the model
  registry; its preflight prints the selected value and source.

The long-lived F16 storage is limited to `main_kv_cache` and
`indexer_kv_cache`. Compressor rings, pooling state, RMSNorm, RoPE, score and
top-K scratch remain F32. Commit arithmetic is completed in F32 before the
single F32-to-F16 store. Read kernels widen to F32 where required, and the
batched gfx1201 indexer score route uses native F16 WMMA with F32 accumulation.

## Kernel exactness gate

The model-free gfx1201 battery exercised every new F16 kernel and compared it
against the existing F32 implementation consuming the exact widened stored-half
values:

| Surface | Raw-bit comparisons | Result |
|---|---:|---|
| staged commit | 128 | pass |
| decode score | 257 | pass |
| batched score | 2,056 | pass |
| batched WMMA score | 2,056 | pass |
| decode gather | 8,192 | pass |
| batched gather | 65,536 | pass |
| batched identity gather | 65,536 | pass |
| decode identity gather | 8,192 | pass |

This gate proves the F16 consumers are arithmetically identical conditional on
the stored half values. It does not claim F16 and F32 model generations must be
byte-identical; storage quantization legitimately changes the model state.

## Product results

All product rows used `scripts/serve_harness.py`, TP3 devices 0,1,2, exact
gfx1201, MQ2R, checkpoint-default six experts, greedy sampling, thinking off,
speculation off, and real decoded-output inspection.

| Cache | Fixture | Context | Generated | Prefill tok/s | Decode tok/s | Correctness |
|---|---|---:|---:|---:|---:|---|
| registry F32 | canonical 2K | 2,052 | 512 | 483.413 | 53.274 | coherent, no empty/attractor |
| explicit F16 | canonical 2K | 2,052 | 512 | 481.008 | 52.692 | coherent, no empty/attractor |
| explicit F16 | NIAH 32K | 21,349 | 19 | 312.682 | 40.333 | recall 1/1 |
| explicit F16 | NIAH 128K | 85,693 | 19 | 220.884 | 28.777 | recall 1/1 |

A same-session, same-schedule screen removed the declared-ceiling confound:

| Cache | Declared max | Prefill batch | Prefill tok/s | Decode tok/s | Recall |
|---|---:|---:|---:|---:|---|
| F32 | 131,072 | 512 | 385.664 | 41.011 | 1/1 |
| F16 | 131,072 | 512 | 382.760 | 40.113 | 1/1 |

At matched B=512, F16 is 0.75% slower in prefill and 2.19% slower in decode.
Halving storage therefore nearly reaches parity, but the current cast/widen
path consumes the bandwidth saving. A speed promotion requires native-F16
gather/attention composition rather than storage precision alone.

The canonical prompt MD5 is
`25e22faef15a20ae53501f1956e62b79`. The F32 and F16 512-token completions
both remained coherent but diverged slightly, as expected for a cache-precision
change. The harness marks any length-capped row as `runaway`; decoded inspection
showed a normal technical continuation with no repetition attractor.

The NIAH fixtures returned the exact answer
`mauve-velociraptor-7741`. Their prompt MD5 values remain
`2e311623a082f6850a45b2ceefee9d9b` (21K) and
`1328229814512e36c4743aa3f9df0e33` (85K).

For comparison, the established F32 B=512 bounded-top-K route measured
390.074/41.148 tok/s at 21K and 268.664/30.767 tok/s at 85K. The F16 runs used
B=128 because `max_seq=1,048,576`; that schedule mismatch accounts for the
large prefill deficit and is the first performance lever. No F16 speed win is
claimed from these rows.

## Physical capacity

The production loader and request-growth path were exercised with stable VMM
reservations and replicated cache placement.

| Cache | Highest measured pass | First measured rejection | Logical reserve/rank |
|---|---:|---:|---:|
| F32 | 229,376 | 245,760 | 14,428,405,760 B |
| F16 | 475,136 | 491,520 | 7,214,202,880 B |

The earlier informal estimate that F32 reached roughly 500K was not backed by
a capacity run and was wrong. The matched brackets show the expected 2.07x
increase from halving the dominant replicated cache storage.

Both brackets used B=128 so cache dtype, not prefill scratch, determined the
comparison. The F32 bracket was collected before `6e4a2dd8f` restored the
shipping F32 long-context schedule to B=512; it is therefore a
capacity-maximized F32 control, not a claim that the current default admits
229K with its larger scratch allocation. The restored default can only have
less physical cache headroom, not more.

| Requested tokens | Result | Prepared tokens | Mapped cache bytes/rank | Pointer identity |
|---:|---|---:|---:|---|
| 20,480 | pass | 28,671 | 262,144,000 | stable |
| 81,920 | pass | 90,111 | 658,505,728 | stable |
| 393,216 | pass | 401,407 | 2,814,377,984 | stable |
| 458,752 | pass | 466,943 | 3,254,779,904 | stable |
| 475,136 | pass | 483,327 | 3,342,860,288 | stable |
| 491,520 | rejected | unchanged | unchanged | stable |
| 1,048,576 | rejected | unchanged | unchanged | stable |

At 1M, rank 0 needed another 6.29 GiB plus the mandatory 0.50 GiB headroom
with only 3.09 GiB free. At 491,520 after the successful staged growth, rank 0
needed another 0.12 GiB plus headroom with 0.52 GiB free. Rank 0 owns one more
routed expert than ranks 1 and 2, so it is the TP3 admission limit.

The original probe printed identical VRAM totals for all ranks because it did
not rebind before `hipMemGetInfo`. Cache pointer and mapped-byte accounting was
correct, but the per-rank used/free labels reflected whichever device was last
bound. `0cb934112` fixes this diagnostic defect. The admission itself ran before
scratch/cache mutation and remained atomic.

## Commits

- `2830c5cd8`: selectable F16 compressor-cache implementation and registry
  truthfulness;
- `f6d9984d7`: gfx12-native `_Float16` spelling for ROCm 7.14 device compile;
- `d2a50f93f`: harness resolves omitted `--kv` from the registry;
- `97511ad83`: admission errors report the selected cache dtype;
- `0cb934112`: capacity reports bind the correct rank before VRAM queries;
- `6e4a2dd8f`: preserve the established F32 B=512 long-context schedule.

## Evidence

Raw evidence is preserved at:

`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-08-gfx1201-tp3-f16-cache/`

Key SHA-256 identities:

- `f32-short.json`: `efc47ad5a6945fa28c7111db6c84ceeb4b1e9c88725669ece6342ad1e104c9f9`;
- `f16-short.json`: `46927bbad47f15ae79d0ee7d4b8b408af1b2a1aad5223d90ca3e15f3a56995ca`;
- `f16-niah21k.json`: `d4676b811177f7b60c9e849fff71e3f2e5b2cceb71d3dd008eac4f65c1862f60`;
- `f16-niah85k.json`: `1e8d6cdef5241f1f0d20fcb682993bb0e6fc4d9229812951fb0fe797926d26a5`;
- `f32-b512-niah21k.json`: `a9794cb17aca3b61aae8c4a540e12f86b62e5b6a8d4700b7d643cc74726327a8`;
- `f16-b512-niah21k.json`: `24a0301491ecbc424ad4995e03e0aa87cab3634be2256b925345fb9ee685a12d`;
- `f16-capacity.log`: `7928757edf42790e3b250980905fc56a9647062b8c57444f8fc1497a6f18772f`;
- `f16-capacity-bracket.log`: `93e805bc67d965056f5944bfcab0f3ba4922bef14e1f6c2bffccabf4ae41a2a2`;
- `f32-capacity-bracket.log`: `bf3a99a282234a7cf7e2c3bcf234541986ad98b8ed3b41a78d675796c52a97e6`.

## Skipped

No 1M-token prefill, repeated product samples, TP4 capacity run, optimized F16
schedule, full native-F16 compressed attention, 8-bit DS4 cache, shared-cache
retry, DSpark, retained PM4, weight/format/sampling/top-k/expert-count change,
gfx1100/gfx1151/gfx1200/gfx942 runtime, or Qwen GPU runtime was attempted.
Registry and unit tests cover the non-DS4/default isolation; the F16 runtime
gate remains exact-gfx1201 MQ2R TP3/TP4.
