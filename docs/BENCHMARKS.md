# Benchmarks

Measured tok/s across the RDNA family. MQ4 weights throughout. Measured
2026-04-13 against hipfire v0.1.5 "redline".

Bench methodology:

```
bench_qwen35_mq4 <model> --prefill N --gen 30 --warmup 10
```

- First prefill excluded (kernel JIT); best-of-one reported.
- 7900 XTX numbers use Q8 KV + `HIPFIRE_GRAPH=1` (hipGraph decode, best case).
- V620 + BC-250 use asym3 KV (5.5× compression), no hipGraph (not tuned there).
- `BW` = model_bytes × gen_tok_s (effective weight-read bandwidth).

## RDNA3 — Radeon RX 7900 XTX (gfx1100, 24 GB, 960 GB/s peak)

Primary target. WMMA-accelerated MQ4 projections + hipGraph decode + Q8 KV cache.

| Model | pp32 | pp128 | pp512 | pp2048 | decode | BW | % of peak BW |
|---|---:|---:|---:|---:|---:|---:|:---:|
| 0.8B | 2072 | 4878 | 7059 | **7383** | **391** | 200 GiB/s | 22% |
| 4B   | 1041 | 2062 | 2487 | 2467 | **180** | 433 GiB/s | 48% |
| 9B   |  980 | 1509 | 1663 | 1624 | **132** | 654 GiB/s | **73%** |
| 27B  |  398 |  478 |  477 |  455 |  **47** | 651 GiB/s | **73%** |

- Decode on 9B/27B saturates 73% of 7900 XTX's 960 GB/s peak memory
  bandwidth — weight-read is the binding constraint.
- Prefill on 4B reaches 2487 tok/s at pp512 (WMMA GEMM ceiling), holds flat
  through pp2048.
- 0.8B prefill keeps rising to pp2048 (7383 tok/s) because launch overhead
  dominates at small prefill sizes; longer prompts amortize.
- 0.8B is launch-count-bound on decode; a proposed multi-row GEMV experiment
  pulled back on this arch (user memory: "DON'T multi-row GEMV on gfx1100").

## RDNA2 — V620 Pro (gfx1030, 32 GB, 512 GB/s peak)

Clean datacenter card. asym3 KV + batched flash attention. Full asym3 sweep
across prefill lengths:

| Model | pp32 | pp128 | pp512 | pp1024 | pp2048 | pp4096 | decode | BW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8B | 48 † | 3989 | 4303 | 4298 | 4179 | 3914 | 250 | 128 GiB/s |
| 4B   | 803  |  952 |  909 |  894 |  872 |  834 |  96 | 230 GiB/s |
| 9B   | 48 † |  547 |  527 |  520 |  512 |  498 |  65 | **322 GiB/s** |
| 27B  | 151  |  149 |  147 |  146 |  144 |  141 |  22 | 303 GiB/s |

† Includes kernel JIT cost; subsequent small prefills hit cache.

- 9B decode at 322 GiB/s is ~63% of V620's 512 GB/s peak.
- Prefill scales almost flat from pp128 → pp4096 (15-20% drop). asym3 + flash
  attention keeps long-context prefill efficient.
- 27B works out-of-the-box on the 32 GB card with plenty of VRAM headroom.

## APU — BC-250 (gfx1013 → gfx1010, 14 GB shared, DDR5)

Ryzen 5800X3D + 8-CU Navi 10 iGPU, shared DDR5 memory. 27B won't fit.

| Model | pp32 | pp128 | pp512 | pp1024 | pp2048 | pp4096 | decode | BW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8B | 685 | 764 | 1253 | 1366 | 1331 | 1282 | 207 | 106 GiB/s |
| 4B   | 151 | 238 |  237 |  241 |  242 |  233 |  77 | 187 GiB/s |
| 9B   |  30 | 104 |  144 |  154 |  154 |  150 |  47 | **236 GiB/s** |

- 9B decode 236 GiB/s on an **APU** running off DDR5. asym3's 5.5× compression
  is what makes this viable.
- Sub-100 tok/s prefill on 9B pp32 is APU launch-latency cost; amortizes by pp128.
- 0.8B is the only size that keeps decode > 200 tok/s; 4B+ are weight-BW bound.

## RDNA1 — RX 5700 XT (gfx1010, 8 GB, 448 GB/s peak)

Historical / smoke-test. Numbers from v0.1.3 era (HF4, pre-MQ4):

| Model | Quant | tok/s | Notes |
|---|---|---:|---|
| Qwen3.5-0.8B | HF4 | 190 | DeltaNet |
| Qwen3.5-4B  | HF4 |  61 | — |
| Qwen3.5-9B  | HF4 |  43 | Best quality in 8 GB |
| Qwen3-8B    | HF4 |  60 | Standard attention |

Side-by-side on the same hardware:

```
ollama + llama.cpp + ROCm (HSA_OVERRIDE):  4.93 tok/s  (Qwen3.5-9B)
hipfire HF4:                              43    tok/s  (same model)
                                          ────────────
                                          8.7× speedup
```

MQ4 + asym3 retest on gfx1010 is pending post-0.1.5 propagation.

## KV cache format comparison (9B, gfx1030, ctx=4096)

All three asym modes pass the multi-turn "Kaden" rare-token recall test.

| KV mode | K bits | V type | Bytes/head/pos | Compression | Decode tok/s | Quality |
|---|:---:|:---:|:---:|:---:|:---:|---|
| q8 | 8 | Q8 | 544 | 3.76× | baseline | reference |
| asym4 | 4 | Q8 | 404 | 5.1× | 116 | ✓ |
| **asym3** | **3** | **Q8** | **372** | **5.5×** | **120** | **✓ default** |
| asym2 | 2 | Q8 | 340 | 6.0× | 116 | ✓ |

asym3 is the sweet spot — best compression of the recall-safe options. See
[KV_CACHE.md](KV_CACHE.md) for the design rationale.

## Running benchmarks yourself

```bash
hipfire bench qwen3.5:4b                    # smoke test
hipfire bench qwen3.5:9b --runs 3           # best-of-3
./scripts/speed-gate.sh                     # full sweep (used in CI)
./scripts/speed-gate.sh --update-baselines  # record new ground-floor
```

`scripts/speed-gate.sh` values are the **permanent regression floor** —
commits may not ship numbers below them without a `--update-baselines`
re-record + justification.

## Comparison to alternatives

Sorted by "works today on consumer RDNA":

| Tool | RDNA1 | RDNA2 | RDNA3 | APU | Multi-turn recall | Setup |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **hipfire** | ✅ | ✅ | ✅ | ✅ | ✅ asym3 | one command |
| llama.cpp + ROCm | HSA hack | ✅ | ✅ | HSA hack | ✅ | fiddly |
| Ollama | via llama.cpp | via llama.cpp | via llama.cpp | ✗ | ✅ | mostly works |
| MLC | ✓ | ✓ | ✓ | ✗ | ✓ | Python stack |
| vLLM | ✗ | ✓ | ✓ | ✗ | ✓ | datacenter only |

The consistent hipfire win is **explicit RDNA-generation coverage** — per-arch
defaults and per-model overlays take the "which knobs do I set for this card"
problem off the user. Raw-speed deltas vary by card.
