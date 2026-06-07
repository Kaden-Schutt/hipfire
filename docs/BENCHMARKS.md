# Benchmarks

All measurements on the indicated arch with the engine's default config
(asym3 KV, FlashAttention auto, prompt_normalize=on). Numbers are
medians across 5 runs unless noted. See
[methodology/perf-benchmarking.md](methodology/perf-benchmarking.md) for
the protocol and the noise band you should expect when reproducing.

## Autoregressive decode (no spec) — 7900 XTX (gfx1100)

| Model | decode | prefill (peak) | effective BW |
|---|---:|---:|---:|
| Qwen 3.5 0.8B MQ4 | **391 tok/s** | **7383 tok/s** | 200 GiB/s |
| Qwen 3.5 4B MQ4 | **180 tok/s** | **2487 tok/s** | 433 GiB/s |
| Qwen 3.5 9B MQ4 | **132 tok/s** | **1663 tok/s** | **654 GiB/s** |
| Qwen 3.5 27B MQ4 | **47 tok/s** | **478 tok/s** | **651 GiB/s** |

9B and 27B decode saturate ~650 GiB/s of the 7900 XTX's 960 GB/s peak
(68% BW-efficient end-to-end across weights + KV + activations).
Prefill on the smaller sizes is WMMA-bound on the MQ4 fused
projections.

## DFlash speculative decode by genre — 7900 XTX

DFlash speedup is **genre-conditional**. Code prompts whose target
distribution matches the draft win big; long-form prose where the
target's high-entropy continuations diverge from draft predictions can
be a net loss.

5-run medians, asym3 KV, `--no-chatml`, `max_tokens=120`,
`prompt_normalize=true`:

| Model | genre | AR tok/s | DFlash tok/s | speedup | τ |
|---|---|---:|---:|---:|---:|
| Qwen 3.5 27B | code (HumanEval/53) | 44.1 | **196.0** (peak 218.6) | **4.45×** | 9.82 |
| Qwen 3.5 27B | prose (Rome essay) | 44.0 | 49.6 | 1.13× | 1.67 |
| Qwen 3.5 27B | instruct (sky-color) | 44.6 | 44.7 | 1.00× | 1.39 |
| Qwen 3.5 9B | code (HumanEval/53) | 124.0 | **329.1** (peak 346.7) | **2.65×** | 6.76 |
| Qwen 3.5 9B | code (HumanEval/0) | 121.9 | **372.9** | **3.06×** | 8.23 |
| Qwen 3.5 9B | instruct (sky-color) | 124.4 | **246.9** | **1.99×** | 4.76 |
| Qwen 3.5 9B | prose (federalist) | **125.3** | 99.4 | 0.79× ✗ | 1.20 |
| Qwen 3.5 9B | prose (Rome) | **122.7** | 98.3 | 0.80× ✗ | 1.20 |
| Qwen 3.6 27B | code (HumanEval/53) | 44.2 | **185.5** | **4.19×** | 9.25 |

**Default `dflash_mode=off`** as of v0.1.8 — DFlash is opt-in until
the genre-conditional speedup is more universally a win. Enable it
globally with `hipfire config set dflash_mode auto` (the engine then
turns DFlash on for dense Qwen 3.5+ targets and off where it
historically loses) or per model with `hipfire config qwen3.5:27b set
dflash_mode on`. The numbers above were measured with DFlash forced
on.

## vs ollama (Q4_K_M GGUF) — 7900 XTX

Same machine, same models. hipfire MQ4 (asym3 KV, FlashAttention) vs
ollama default Q4_K_M through llama.cpp's ROCm backend. Matched
~140-token and ~530-token prompts and matched 128-token generation
lengths. Ollama numbers extracted from its own `prompt_eval_duration` /
`eval_duration` reporting via `/api/generate` with `num_predict=128`.

| Model | hf pp128 | oll pp128 | hf pp512 | oll pp512 | hf decode | oll decode | decode× |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen 3.5 0.8B | **10,861** | 4,622 | **12,962** | 7,117 | **353** | 168 | **2.10×** |
| Qwen 3.5 4B | **3,304** | 1,972 | **3,321** | 2,670 | **165** | 93 | **1.78×** |
| Qwen 3.5 9B | **1,920** | 1,428 | 1,919 | **1,970** | **122** | 71 | **1.71×** |

hipfire wins decode 1.7–2.1× across the board — that's the user-visible
number for interactive chat. Prefill is more nuanced: hipfire wins
decisively on 0.8B / 4B and at pp128 for 9B (batched MQ4 fused
projections saturate WMMA on small matmuls where llama.cpp's per-token
GGUF dequant can't), but ollama edges past at pp512 for 9B (the GEMMs
are large enough there to saturate even without WMMA).

Harness: [`cli/bench_vs_ollama.ts`](../cli/bench_vs_ollama.ts).

## Other arches

Decode tok/s, default config:

| Arch | Examples | 0.8B | 4B | 9B | 27B |
|---|---|---:|---:|---:|---:|
| RDNA2 (gfx1030) | V620 Pro, RX 6800 XT | 250 | — | 65 | 22 |
| RDNA1 (gfx1010) | RX 5700 XT | 190 | 61 | 43 (HF4) | OOM |
| APU (gfx1013) | BC-250 | 207 | 77 | 47 | OOM |
| GCN5 (gfx906) | MI50 / MI60 | 231 | 61 | 59 | 21 |
| MI300X (gfx942) | datacenter | 850 | 480 | 320 | 130 |

MI300X is wave64 + MFMA — different kernel family. RDNA4 (gfx1200 /
gfx1201) ships a dispatch fallback to dot2 today; per-arch WMMA
kernels are in progress (issue #54). gfx906 (Vega 20) uses the
nwarps=4 dp4a MMQ kernel for prefill at batch≥16
(`docs/plans/gfx906-mmq-prd.md`). Decode at batch=1 uses two
gfx906-specific optimizations from the 2026-05-05 perf
investigation
(`docs/perf-checkpoints/2026-05-05-gfx906-decode-investigation.md`):
the residual-GEMV runs a software-pipelined ILP-injection variant
(+4.8% on 9B), and the three fused projections (`gate_up`, `qkv`,
`qkvza`) pre-quantize x to Q8_1 and use `v_dot4_i32_i8` (+9.3% on
9B). Combined: 50.7 → 58.9 tok/s (+16.2%) on Qwen 3.5 9B. Stock
llama.cpp Q4_K_M on the same hardware: 61.55 tok/s (1.04× ahead),
skyne98/iacopPBK fork: 63.48 (1.08× ahead).

## Reproducing

```bash
hipfire bench qwen3.5:9b
```

Runs the canonical bench (pp32 / pp128 / decode) on a fresh build
against the committed speed-baselines in
`tests/speed-baselines/<arch>.txt`. The same harness gates
pre-commit when kernel or dispatch code changes.

For DFlash perf comparison, use the prompt-md5-pinned scripts in
`benchmarks/prompts/` — see `methodology/perf-benchmarking.md` for why
prompt structure matters as much as model + flags (one stray newline
swings τ by 17%).
