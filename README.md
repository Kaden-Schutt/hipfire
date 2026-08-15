# hipfire

LLM inference for AMD RDNA GPUs. Rust + HIP. Single binary. No Python
in the hot path. Ollama-style UX.

```bash
hipfire pull lfm2.5:1.2b   # 1.25 GB — small dense model, fast first token
hipfire serve -d           # background daemon, OpenAI-compatible API on 0.0.0.0:11435
hipfire chat lfm2.5:1.2b   # interactive chat (auto-starts serve if it isn't running)
```

One-shot prompts and bigger models work the same way:

```bash
hipfire pull qwen3.5:9b
hipfire run  qwen3.5:9b "What is the capital of France?"
```

Current release: **v0.2.1** — dispatch unification (#397). DeepSeek V4
Flash support landed in v0.2.0. See [CHANGELOG.md](CHANGELOG.md).
Curated weights live on Hugging Face at
[huggingface.co/hipfire-models](https://huggingface.co/hipfire-models)
(newer publishes) and the legacy per-model repos the registry still
points at.

Discord: <https://discord.gg/F3BaywB8Rs>

## v0.2.1 highlights

- **Dispatch unification (#397).** Every GEMV / GEMM / attention / MoE /
  rotation / fused projection across all seven architecture crates now
  resolves through typed kernel families with per-(arch × dtype) tables —
  adding a dense quant is a table entry plus a kernel file, no model
  code. Forward-as-pipeline lowered decode is default-on everywhere, and
  GPU-free coverage gates assert dispatch plans for the whole fleet
  including RDNA4.
- **Multi-GPU expert-parallel serving.** `hipfire serve --tp N` shards
  DeepSeek V4 Flash / MiniMax-M2 experts across N GPUs, with DSML
  tool-call render/parse on the EP serve path.
- **RDNA4 F16 fix.** A broken `gemm_f16_tiled` fallback was corrupting
  ds4's DSA compressor on gfx12 — F16 GEMVs now route through WMMA on
  RDNA4 (restoring ds4 EP tool-calling), and the rewritten fallback
  kernel is correct and up to 13.8× faster.
- **Jinja chat templates default-on** (HF-byte-exact tool rendering),
  plus q8 error-feedback DeltaNet state by default.

## Supported architectures

| Family | Pull tags | Notes |
|---|---|---|
| Qwen 3.5 / 3.6 dense | `qwen3.5:0.8b` … `qwen3.5:27b`, `qwen3.6:27b` | Hybrid DeltaNet + FullAttn; DFlash spec-decode draft tags for 9B/27B |
| Qwen 3.5 / 3.6 MoE | `qwen3.5:35b-a3b`, `qwen3.6:35b-a3b` | 35B total / 3B active |
| LFM2.5 family | `lfm2.5:350m`, `lfm2.5:1.2b`, `lfm2.5:1.2b-thinking`, `lfm2.5:8b-a1b` | LiquidAI hybrid conv + GQA, dense and MoE; published under [hipfire-models](https://huggingface.co/hipfire-models) |
| DeepSeek V4 Flash | `deepseek-v4-flash` | MQ2-Lloyd routed-expert MoE + MTP spec-decode; tool calls; multi-GPU `hipfire serve --tp 4` |
| MiniMax-M2 | BYO (`hipfire quantize`) | Interleaved thinking, prefix caching, EP serve arm |
| Qwen2 | BYO (`hipfire quantize`) | Plain Qwen2 decoder (e.g. 1.5B class) |
| dots.ocr | BYO (`hipfire quantize`) | Qwen2-VL-family layout-extraction VLM — image → structured OCR |
| LLaMA family | `qwen3:0.6b`, `qwen3:8b`, BYO GGUF | Standard-attention loader path |
| Gemma 4 | — | In integration (`feat/gemma4-integrate`) — not on master yet |

See [docs/MODELS.md](docs/MODELS.md) for the full curated registry
(MQ6 variants, drafts, fine-tunes) and bring-your-own-model flows.

## Why

`llama.cpp + ROCm` works on RDNA but is painful: upstream ROCm
officially supports only a handful of datacenter cards; consumer RDNA
is a second-class citizen. hipfire targets the entire RDNA family
(RDNA1 → RDNA4, consumer + pro + APU) with a single Rust binary that
ships pre-compiled kernel blobs when possible and JIT-compiles the
rest through HIP. No Python, no PyTorch, no ROCm userspace stack at
runtime.

## Headline numbers — 7900 XTX (gfx1100)

Decode tok/s, default config (measured at asym3 KV — perf-equivalent
to today's `fwht3` per-arch default; FlashAttention auto):

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

### RDNA4 (gfx1201, Radeon AI PRO R9700)

| Model | Config | Decode tok/s |
|---|---|---:|
| Qwen2 1.5B HFQ4 | single GPU | **266** |
| DeepSeek V4 Flash (82 GB MQ2-Lloyd) | 4× R9700, `hipfire serve --tp 4` (EP) | **25.6** |
| Gemma 4 12B MQ4 | single GPU (integration branch, pre-merge) | **~47** |

CASK-based KV cache eviction lets you run long-context prompts without
OOM: generate a sidecar with `hipfire sidecar-gen <model>` and enable
eviction with `hipfire config cask-profile balanced`. See
[CONFIG.md](docs/CONFIG.md) for details.

## Install

Linux with ROCm 6+:

```bash
curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.sh | bash
```

For Windows, source builds, and verifying the install:
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## NixOS

First-class support via Nix flake. See [docs/NIXOS.md](docs/NIXOS.md).

```bash
nix develop github:Kaden-Schutt/hipfire  # dev shell with Rust + ROCm + bun
nix build github:Kaden-Schutt/hipfire    # build package
```

NixOS module:

```nix
{
  inputs.hipfire.url = "github:Kaden-Schutt/hipfire";
  # then in configuration.nix:
  services.hipfire.enable = true;
  services.hipfire.gpuTargets = [ "gfx1100" ];
}
```

## Inspiration: Lucebox

hipfire's DFlash work was substantially shaped by Davide Ciffa's
[Lucebox DFlash on ggml](https://www.lucebox.com/blog/dflash27b) — a
standalone C++/ggml/CUDA DFlash for Qwen 3.5-27B on a single RTX 3090.
Different stack, different vendor — but Lucebox's blog gave us
concrete published numbers to target, n_gen-aware bench methodology,
and pointers at where the fat is. Cached snapshot at
`.research-cache/lucebox-dflash27b.html` for forensic reproducibility.

## Inspiration: gfx906 (MI50/MI60) optimizations

hipfire's gfx906 prefill MMQ kernel and AR-decode optimizations were
shaped by two community forks of `llama.cpp` that target Vega 20:

- **[iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906)**
  — the original fork that ported and tuned gfx906-specific code paths
  (warp-cooperative GEMV via half-wave split, Y-tile prefetch via
  inline-asm `global_load_dword`, `__builtin_amdgcn_readfirstlane`-based
  SGPR hoisting, separate HBM-load → register-cache → LDS-store
  pipelining in the MMQ body). The "2602.01 version" commit
  `eec153c086df6a9e7a69499bea3639597c085fff` was the canonical reference
  we audited against.
- **[skyne98/llama.cpp-gfx906](https://github.com/skyne98/llama.cpp-gfx906)**
  — fork-of-fork that propagates iacop's optimizations (commit
  `42c298c` "port iacop optimizations") and tracks upstream more
  aggressively. The accompanying
  [skyne98/wiki-gfx906](https://skyne98.github.io/wiki-gfx906/intro.html)
  is the best public reference for gfx906 ISA quirks (LDS bank-conflict
  patterns at stride 32, dp4a issue-rate ceiling, Q8_1 activation
  layout) — we used it as a sanity-check for several PMC-driven
  redesign decisions.

And of course an extra shout-out to `ggml-org/llama.cpp` itself: the
templated `mmq_x` body in `mul_mat_q.cu` was the architectural scaffold
we ported to gfx906 (templated mmq_x ladder, per-thread accumulator
layout, MMQ_TILE_NE_K=32 sub-block factoring, Q8_1 quantize math). The
inner loop is gfx906-specific; the outer shape is descendant.

A standalone gfx906 perf investigation log is at
[`docs/perf-checkpoints/2026-05-05-gfx906-decode-investigation.md`](docs/perf-checkpoints/2026-05-05-gfx906-decode-investigation.md);
the prefill MMQ redesign log is at
[`docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md`](docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md).

## Documentation

| Page | Topic |
|---|---|
| [GETTING_STARTED.md](docs/GETTING_STARTED.md) | Install, first run, what to read next |
| [NIXOS.md](docs/NIXOS.md) | NixOS flake, module, dev shell |
| [CLI.md](docs/CLI.md) | Every subcommand, flags, file locations |
| [MODELS.md](docs/MODELS.md) | Curated tags, BYO models, file extensions |
| [QUANTIZE.md](docs/QUANTIZE.md) | `hipfire quantize` for HF / safetensors / GGUF |
| [CONFIG.md](docs/CONFIG.md) | Every config key, CASK sidecar / KV eviction policies, env overrides |
| [SERVE.md](docs/SERVE.md) | OpenAI-compatible HTTP API |
| [BENCHMARKS.md](docs/BENCHMARKS.md) | Measured perf per arch, vs ollama |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Engine layout, dispatch, two model paths |
| [QUANTIZATION.md](docs/QUANTIZATION.md) | MQ4 / HF4 design, asym KV cache, FWHT math |
| [multi-gpu.md](docs/multi-gpu.md) | Pipeline-parallel (pp≥2) — memory budget, deployment, refusals |
| [methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md) | Bench protocol — read before claiming a perf win |
| [methodology/redline-rust-autoresearch.md](docs/methodology/redline-rust-autoresearch.md) | Rust Redline Loop 0–3 hooks and four-card mining |

## License

hipfire is dual-licensed under MIT or Apache-2.0 at your option. See
[LICENSE](LICENSE) (dual-license pointer), [LICENSE-MIT](LICENSE-MIT),
[LICENSE-APACHE](LICENSE-APACHE), and [NOTICE](NOTICE) for details.

New contributions default to Apache-2.0 via DCO sign-off; existing
contributors' MIT-licensed contributions remain MIT unless they opt
in. Each source file carries an `SPDX-License-Identifier` reflecting
actual authorship (MIT, Apache-2.0, or `MIT OR Apache-2.0`). See
[CONTRIBUTING.md](CONTRIBUTING.md) for the contributor side and
[docs/governance/relicense-2026-05.md](docs/governance/relicense-2026-05.md)
for the decision record (including the 2026-05-19 course correction
from a unilateral Apache-2.0 relicense to dual licensing).

Original architectural innovations originating in hipfire are
catalogued in [PRIOR-ART.md](PRIOR-ART.md); derivative works
(including reimplementations informed by hipfire's design) should
attribute the corresponding inventions per [AGENTS.md](AGENTS.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Install local hooks with
`./scripts/install-hooks.sh`. The no-GPU CI subset is
`./scripts/no-gpu-ci.sh`; it does not replace the hardware gates. Any
change to kernels, quant formats, dispatch, fusion, rotation, rmsnorm,
or the spec-decode path must pass `./scripts/coherence-gate-dflash.sh`
before commit. The canonical correctness gate is per-arch channel-test;
the speed-gate catches regressions on the baseline arch. Don't bypass
either with `--no-verify` — see
[methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md).
