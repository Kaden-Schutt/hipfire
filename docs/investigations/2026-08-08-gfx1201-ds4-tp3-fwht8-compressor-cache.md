# DeepSeek V4 0731 MQ2R gfx1201 TP3 FWHT8 compressor-cache trial

Date: 2026-08-08 UTC

Branch under test: `ds4-gfx1201-opt`

Parent commit: `1019a0e568ef8ff21ff58e7e34354c9d00d17ec7`

Model SHA-256: `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`

## Verdict

Rejected. A selectable DeepSeek V4 compressor-cache tier storing FWHT-rotated
signed INT8 rows passed its model-free kernel oracle and a 21,349-token NIAH
smoke, but all three 85,693-token variants missed the exact needle. The final
G32 indexer variant returned the same corruption family as the first variant.
The failure is therefore not promoted as a user-selectable cache tier.

Only this report is landed. The full experimental patch is preserved with the
raw evidence and was removed from the product branch. F32 remains the default
DeepSeek V4 compressor cache and the separately certified F16 route is
unchanged.

NIAH was the requested initial quality gate. No KLD or perplexity claim is
made.

## Implemented experiment

The rejected implementation was route-strict to gfx1201 MQ2R TP3/TP4 and used
`DType::Raw` only as a DS4-owned wire type. It did not change Qwen dispatch or
any registry default.

Three variants were tested:

1. Indexer K and Q used `H * Dsign / sqrt(128)` with one F16 row scale; tied
   main K/V used rowwise symmetric INT8 in the model basis.
2. Both indexer and tied main K/V used the orthonormal transform; main rows
   were inverse-transformed during gather before the established F32 attention
   consumer.
3. Variant 2 plus four F16 G32 scales for the 128-wide indexer row, directly
   targeting top-K ranking noise. The dominant 512-wide main row remained one
   scale plus 512 signed INT8 values.

Pooling, RMSNorm, RoPE, recurrent rings, score reductions, and attention
accumulation remained F32 in all variants.

## Kernel oracle

The final gfx1201 oracle compared every product reader with an F32
reconstruction of the exact stored values.

| Surface | Gate | Result |
|---|---:|---:|
| indexer FWHT-G32 quantization | maximum error in local-scale steps | 0.499979 |
| main FWHT quantization | maximum error in stored-scale steps | 0.499989 |
| decode indexer score | raw F32 comparisons | 257/257 |
| batched scalar indexer score | raw F32 comparisons | 1,028/1,028 |
| batched WMMA indexer score | raw F32 comparisons | 1,028/1,028 |
| decode selected gather | raw F32 comparisons | 32,768/32,768 |
| batched selected gather | raw F32 comparisons | 131,072/131,072 |
| batched identity gather | raw F32 comparisons | 131,072/131,072 |
| decode identity gather | raw F32 comparisons | 32,768/32,768 |
| direct versus device-slot main-row write | raw bytes | 514/514 |

This rules out wire-reader and inverse-transform disagreement for the tested
values. It does not prove model-level quality.

## Capacity screen

The first single-scale wire format exercised the production loader and
request-owned VMM growth path with `--max-seq 1048576` and the established
B=128 extreme-context schedule.

| Requested tokens | Result | Prepared tokens | Mapped cache bytes/rank | Pointer identity |
|---:|---|---:|---:|---|
| 81,920 | pass | 90,111 | 394,264,576 | stable |
| 851,968 | pass | 860,159 | 3,034,578,944 | stable |
| 917,504 | pass | 925,695 | 3,254,779,904 | stable |
| 950,272 | rejected | unchanged | unchanged | stable |

The logical reservation was 3,629,449,216 bytes/rank. At 950,272, rank 0
needed another 0.08 GiB while only the mandatory 0.50 GiB headroom remained,
so admission failed atomically. This was a useful capacity result but did not
override the retrieval failure. Capacity was not repeated after adding the
six extra indexer-header bytes in the final G32 variant.

## NIAH results

All rows used `scripts/serve_harness.py`, TP3 devices 0,1,2, greedy sampling,
checkpoint-default six experts, thinking off, speculation off, and a declared
1,048,576-token maximum.

| Variant | Context | Generated | Prefill tok/s | Decode tok/s | Answer | Recall |
|---|---:|---:|---:|---:|---|---:|
| indexer FWHT, main rowwise INT8 | 21,349 | 19 | 282.323 | 40.191 | `mauve-velociraptor-7741` | 1/1 |
| indexer FWHT, main rowwise INT8 | 85,693 | 17 | 239.148 | 29.976 | `mauve-velrapoci-7741` | 0/1 |
| indexer and main FWHT | 85,693 | 18 | 202.368 | 29.967 | `mauve-velrapocior-7741` | 0/1 |
| indexer FWHT-G32 and main FWHT | 85,693 | 17 | 204.199 | 29.205 | `mauve-velrapoci-7741` | 0/1 |

The expected answer was `mauve-velociraptor-7741`. All failures had zero
empty responses, runaways, and repetition attractors; they were exact
retrieval misses. The 21K prompt MD5 was
`2e311623a082f6850a45b2ceefee9d9b`.

## Evidence

Raw logs, harness JSON, capacity output, and the complete rejected
implementation are preserved at:

`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-08-gfx1201-tp3-fwht8-cache/`

The source patch is `rejected-implementation.patch`.

## Skipped

No KLD/PPL, repeated throughput acceptance, 1M-token prefill, TP4 capacity,
DSpark, PM4, weight/format/sampling/top-k/expert-count change, cache sharding,
GTT spill, or non-gfx1201 runtime was attempted. NIAH establishes that this
specific FWHT8 construction is not acceptable at the measured 85K depth; it
does not rule out a future trained or higher-precision compressor-cache format.
