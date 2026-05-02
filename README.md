# hipfire

LLM inference for AMD RDNA GPUs. Rust + HIP. Single binary. No Python
in the hot path. Ollama-style UX.

```bash
hipfire pull qwen3.5:9b
hipfire run  qwen3.5:9b "What is the capital of France?"
hipfire serve -d        # background daemon, OpenAI-compatible API on :11435
```

Current release: **v0.1.9-alpha**. See [CHANGELOG.md](CHANGELOG.md).

## Why

`llama.cpp + ROCm` works on RDNA but is painful: upstream ROCm
officially supports only a handful of datacenter cards; consumer RDNA
is a second-class citizen. hipfire targets the entire RDNA family
(RDNA1 → RDNA4, consumer + pro + APU) with a single Rust binary that
ships pre-compiled kernel blobs when possible and JIT-compiles the
rest through HIP. No Python, no PyTorch, no ROCm userspace stack at
runtime.

## Headline numbers — 7900 XTX (gfx1100)

Decode tok/s, default config (asym3 KV, FlashAttention auto):

| Model | hipfire decode | hipfire prefill (peak) | vs ollama Q4_K_M |
|---|---:|---:|---:|
| Qwen 3.5 0.8B | **391** | 7383 | **2.10×** decode |
| Qwen 3.5 4B | **180** | 2487 | **1.78×** decode |
| Qwen 3.5 9B | **132** | 1663 | **1.71×** decode |
| Qwen 3.5 27B | **47** | 478 | — |

DFlash speculative decode lifts code prompts further: **218 tok/s peak
on 27B HumanEval/53** (4.45× over AR), **372 tok/s peak on 9B**.
DFlash speedup is genre-conditional — see
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the full per-genre table
and the cross-arch matrix (RDNA1 / RDNA2 / APU / MI300X).

## Install

Linux with ROCm 6+:

```bash
curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.sh | bash
```

For Windows, source builds, and verifying the install:
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## Inspiration: Lucebox

hipfire's DFlash work was substantially shaped by Davide Ciffa's
[Lucebox DFlash on ggml](https://www.lucebox.com/blog/dflash27b) — a
standalone C++/ggml/CUDA DFlash for Qwen 3.5-27B on a single RTX 3090.
Different stack, different vendor — but Lucebox's blog gave us
concrete published numbers to target, n_gen-aware bench methodology,
and pointers at where the fat is. Cached snapshot at
`.research-cache/lucebox-dflash27b.html` for forensic reproducibility.

## Documentation

| Page | Topic |
|---|---|
| [GETTING_STARTED.md](docs/GETTING_STARTED.md) | Install, first run, what to read next |
| [CLI.md](docs/CLI.md) | Every subcommand, flags, file locations |
| [MODELS.md](docs/MODELS.md) | Curated tags, BYO models, file extensions |
| [QUANTIZE.md](docs/QUANTIZE.md) | `hipfire quantize` for HF / safetensors / GGUF |
| [CONFIG.md](docs/CONFIG.md) | Every config key, env overrides |
| [SERVE.md](docs/SERVE.md) | OpenAI-compatible HTTP API |
| [BENCHMARKS.md](docs/BENCHMARKS.md) | Measured perf per arch, vs ollama |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Engine layout, dispatch, two model paths |
| [QUANTIZATION.md](docs/QUANTIZATION.md) | MQ4 / HF4 design, asym KV cache, FWHT math |
| [methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md) | Bench protocol — read before claiming a perf win |

## License

MIT — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Any change to kernels, quant
formats, dispatch, fusion, rotation, rmsnorm, or the spec-decode path
must pass `./scripts/coherence-gate-dflash.sh` before commit. The
canonical correctness gate is per-arch channel-test; the speed-gate
catches regressions on the baseline arch. Don't bypass either with
`--no-verify` — see
[methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md).
