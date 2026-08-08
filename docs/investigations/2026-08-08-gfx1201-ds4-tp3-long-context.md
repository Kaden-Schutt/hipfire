# DeepSeek V4 0731 MQ2R gfx1201 TP3 long-context validation

Date: 2026-08-08 UTC
Branch: `ds4-gfx1201-opt`
Runtime commit: `3b8bf149d8c0a7bda24ece3038446afc321ab4cb`

## Verdict

The exact gfx1201 TP3 route now uses the same stable-address compressor-cache
VMM mechanism as gfx1151 and automatically selects the existing B=512 prefill
schedule when the declared `max_seq` exceeds 32K. Real generation passes the
committed 21,349-token NIAH fixture, and an allocation-only probe admits 81,920
tokens on all three ranks without changing any cache pointer.

One million tokens does **not** physically fit with the current replicated F32
compressor cache. TP3 has 96 GB of aggregate device memory, but every rank owns
its own complete compressor/indexer state and is individually limited to
34,208,743,424 addressable bytes. At the 1M admission, rank 0 needed another
12.77 GiB plus the mandatory 0.50 GiB headroom with only 1.43 GiB free. The
request was rejected atomically: prepared capacity, mapped bytes, score stride,
and all 62 cache pointers remained unchanged.

The real 85,693-token NIAH request allocated and ran, but failed retrieval and
is not certified. It produced `mauve-velrapocior-7741` instead of
`mauve-velociraptor-7741`. The failure is coherent text, not an empty response,
runaway, daemon crash, or allocation rejection, but recall 0/1 is a hard gate.

## Changes under test

- `d2225b4fc`: exact-gfx1201 admission for DS4 compressor-cache VMM;
- `c6c61138e`: production-loader TP long-context capacity probe;
- `ababa5937`: keep the unsupported DSpark sidecar out of TP loads;
- `3b8bf149d`: automatic B=512 for declared long context and immediate reclaim
  of the old batched score slab during request-boundary growth.

The gfx1201 gate is exact. Qwen, gfx1100, gfx1151, single-GPU DS4, MQ2RXT, and
non-TP routes are unchanged. The user-facing `kv_cache` selector has no `f32`
value; the runs use `--kv auto`. DS4's compressor/indexer cache itself remains
F32 and its VMM allocation is independent of the generic `--kv-backend`
selector, which remained `contiguous`.

## Capacity probe

Fixture: production `load_model_ep`, TP3 on gfx1201 devices 0,1,2, no prefill,
declared maximum 1,048,576 tokens. Each rank loaded with B=512 and 1.50 GiB of
prefill scratch.

| Requested tokens | Result | Prepared tokens | Ratio-4 rows | VMM tensors | Logical reserve/rank | Physically mapped/rank |
|---:|---|---:|---:|---:|---:|---:|
| 20,480 | pass, all ranks | 28,671 | 8,192 | 62/62 | 14,428,405,760 B | 524,288,000 B |
| 81,920 | pass, all ranks | 90,111 | 32,768 | 62/62 | 14,428,405,760 B | 1,317,011,456 B |
| 1,048,576 | rejected at admission | unchanged | unchanged | 62/62 | unchanged | unchanged |

The per-rank pointer hashes were stable from 20K through the rejected 1M
request:

- rank 0: `0x582552701ad96050`;
- rank 1: `0x92764f66d4627eb4`;
- rank 2: `0x2735956ea4e604d1`.

This proves virtual reservation and incremental physical mapping on gfx1201,
but falsifies the proposed 1M physical-F32 fit for replicated TP3.

## Real NIAH generation

Both requests used `scripts/serve_harness.py`, the same committed fixtures as
the gfx1151 campaign, TP3 devices 0,1,2, `max_seq=1,048,576`, greedy sampling,
checkpoint-default top-k 6, speculation off, thinking off, and at most 64
generated tokens.

| Fixture | Prompt MD5 | Prompt tokens | Prefill | Decode | Output gate |
|---|---|---:|---:|---:|---|
| `niah_32k.jsonl` | `2e311623a082f6850a45b2ceefee9d9b` | 21,349 | 68,420.852 ms / **312.025 tok/s** | 20.489 tok/s, 19 tokens | pass, recall 1/1 |
| `niah_128k.jsonl` | `1328229814512e36c4743aa3f9df0e33` | 85,693 | 1,016,614.521 ms / **84.293 tok/s** | 30.652 tok/s, 18 tokens | **fail, recall 0/1** |

The 21K answer was exactly `The secret pass code is
**mauve-velociraptor-7741**.` with `finish=stop` and zero empty, runaway, or
attractor failures. The 85K answer also stopped normally but misspelled the
needle, so neither its correctness nor its throughput is promoted.

For context, the certified gfx1151 cooperative-prefill route measured 202.08
tok/s at the same 21K fixture and 155.91 tok/s at the same 85K fixture. TP3 is
54.4% faster at 21K but 45.9% slower at 85K. The short-context gfx1201 prefill
gain therefore does not transfer as a rising tide at this depth. A likely
candidate is the gfx1201 fallback to the portable batched long-context indexer
path, but this run did not profile or prove that mechanism.

## Evidence

Raw evidence is preserved at:

`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-tp3-longctx/`

Authoritative files and SHA-256:

- `capacity-v3.log`: `fbb6fe9b7e61279edc5b82563783c9ea3b7e6efee48cf7a840aaecdcc3913b39`;
- `niah-21k-v2.json`: `190b5974ee74c2b6c38d7de0ecf8ae1d8f9e83c8b37951441ab9c72ec05cbebb`;
- `niah-21k-serve-v2.log`: `563d03c75f6a93493796d36a7093a7b46420c8444cd794d8adeabb1d7b5f3146`;
- `niah-85k.json`: `6a8a125fb81104fe251bb165bd766ca0e28b3b643bf136dd280e5b575848c92e`;
- `niah-85k-serve.log`: `2d4f4ea1b88071d9a6a47fd98cfdddf7972d6e38d6e3b0571c0d9fe336e7a8eb`.

`capacity.log` (sidecar OOM before the TP loader fix), `capacity-v2.log`
(B=1024 pre-fix), and the first `niah-21k-serve.log` (`--kv f32` rejected by
configuration validation) are retained diagnostic attempts and excluded from
the verdict.

Model SHA-256:
`cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`.

## Skipped

No repeated 20K/80K product samples, no 1M-token prefill, no profile of the
85K regression, no bisect of the retrieval miss, no quantized-KV work, no
DSpark, no TP4, no retained PM4, no weight/format/sampling/top-k change, and no
gfx1100/gfx1151/Qwen runtime were attempted. The 85K correctness failure and
the per-rank physical-memory rejection prevent a certification or 1M claim.
