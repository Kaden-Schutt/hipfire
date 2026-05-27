# gfx1100 text-prefill WMMA validation: 30× over scalar (2026-05-26)

Validates commit `3b3b0b64` ("causal WMMA flash attention + GQA for text
prefill") on gfx1100 (7900 XTX) — the primary deploy target. The commit
measured the win on gfx1151 only and left "gfx1100 perf TBD".

Branch `feat/dots-ocr-phase-3-daemon` @ `3b3b0b64`. System ROCm 7.2.3.
Bench: `ocr_e2e` full dots.ocr pipeline (vision encoder → text prefill →
greedy decode).

```
./target/release/examples/ocr_e2e \
  --hfq /data/hipfire/dots-ocr.q8.hfq \
  --image benchmarks/images/dots_ocr_smoke_001.jpg \
  --prompt-json benchmarks/references/dots_ocr_smoke_001.json \
  --prefill {seq|batch} --max-tokens {64|1}
```

Prompt = 5095 positions (4880 visual + 215 text), hd=128, GQA 12:2.

## Result: prefill win reproduces on gfx1100

| `--prefill` | path | 5095-tok prefill | tok/s |
|---|---|---|---|
| `seq`   | scalar `attention_causal_batched` (no GQA, 12× K/V reload) | 41.3 s | **123** |
| `batch` | `attention_dflash_wmma_m64_n128_f16kv_v3_causal` (causal WMMA + GQA + f16 KV) | 1.4 s | **3696** |

**~30× prefill speedup**, output correct: valid dots.ocr layout JSON
(`{"bbox": [628,172,1077,194], "category": "Page-header", "text":
"EXPOSURE TO MEAT AND RISK OF LYMPHOMA"}` …). The `batch` path engages the
WMMA causal+GQA kernel when `head_dim==128 && batch>=64`; scalar fallback
otherwise.

Single run per mode (not median-of-3), but a 30× gap is far outside the
±10–15 % session-noise band — the magnitude is unambiguous. The `seq` and
`batch` numbers came from separate fresh processes (each re-loads weights +
re-runs the vision encoder), so they are not within-session A/B.

## Decode (the slow axis, unchanged by this commit)

`batch` run, greedy: **2.7 tok/s** (64 tokens in 24.0 s) at context ~5100.
This is the full text-model forward per token — the path the GQA flash-decode
kernel work (`d8bfe006`, and the seq-crossover analysis in
`2026-05-26-gfx1100-decode-*`) targets. Independent of the prefill win.

Vision encoder: 40.2 s (separate concern, not touched here).

## Cross-machine: ratio smaller here, absolute batch lower (counter-intuitive)

| | gfx1100 (here) | gfx1151 (commit `3b3b0b64`) | note |
|---|---|---|---|
| seq baseline | 123 tok/s | ~59 tok/s | gfx1100 2.1× faster (more CUs/BW) |
| batch WMMA | 3696 tok/s | 4972 tok/s | gfx1100 **0.74× — slower** |
| speedup | **30×** | **84×** | smaller ratio here = higher baseline |

The 7900 XTX has far more compute + bandwidth than the Strix Halo APU, so a
WMMA-bound kernel running **slower** here is suspicious. Most likely the 1.4 s
is not dominated by the attention kernel at this size: the `batch` path builds
a `[5095, hidden]` embeds matrix on the CPU and issues 215 per-text-token
`embed_token_row` GPU round-trips before the single
`forward_prefill_batch_embeds` call. At only 5095 tokens that fixed overhead is
a large fraction of wall time, and it would not scale with GPU FLOPs/BW. Not a
regression — a clean 30× either way — but if we want to close the gap to the
APU's absolute number, instrument the embed-build vs attention split (kernel-
trace works on gfx1100; PMC does not — see
`2026-05-26-gfx1151-decode-attention-pmc.md`).

## Open follow-ups

- Attribute the 1.4 s `batch` prefill: CPU embed-matrix build + 215
  `embed_token_row` round-trips vs the WMMA attention itself (kernel-trace).
- Decode at long context (2.7 tok/s) is the dominant user-visible cost for OCR
  — see the decode seq-length crossover work (gqa vs flash, ~seq 11200).
