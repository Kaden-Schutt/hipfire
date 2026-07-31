<p align="center">
  <img width="1774" height="887" alt="hipfire — local LLM inference for AMD GPUs" src="https://github.com/user-attachments/assets/2013c1ae-6011-477c-a8de-5e90751bed74" />
</p>

<h1 align="center">hipfire</h1>

<p align="center">
  <strong>Fast local LLM inference for AMD GPUs.</strong><br />
  Rust + HIP + Redline. No Python in the hot path. Ollama-style UX.
</p>

<p align="center">
  <a href="https://github.com/warpfront/hipfire/releases"><img alt="Stable release v0.2.1" src="https://img.shields.io/badge/stable-v0.2.1-24292f?style=flat-square" /></a>
  <a href="CHANGELOG.md"><img alt="Next release v0.3.0 beta" src="https://img.shields.io/badge/next-v0.3.0%20beta-f04b24?style=flat-square" /></a>
  <a href="docs/MODELS.md"><img alt="54 curated model entries" src="https://img.shields.io/badge/registry-54%20curated%20models-ff8a1f?style=flat-square" /></a>
  <a href="https://discord.gg/F3BaywB8Rs"><img alt="Join Discord" src="https://img.shields.io/badge/chat-Discord-5865F2?style=flat-square" /></a>
</p>

<p align="center">
  <a href="#mq4r--redline">MQ4R + Redline</a> ·
  <a href="#curated-model-registry">Models</a> ·
  <a href="#gpu-support">GPU support</a> ·
  <a href="#install">Install</a> ·
  <a href="#documentation">Docs</a>
</p>

```bash
hipfire pull qwen3.5:4b
hipfire serve qwen3.5:4b -d
hipfire chat qwen3.5:4b
```

One-shot inference uses the same model registry and serving stack:

```bash
hipfire run qwen3.5:4b "What is the capital of France?"
```

The daemon exposes an OpenAI-compatible API on `0.0.0.0:11435`.

Current stable release: **v0.2.1**. The next release is **v0.3.0**,
headlined by MQ4R and Redline across RDNA. See
[CHANGELOG.md](CHANGELOG.md).

Curated weights are published through
[huggingface.co/hipfire-models](https://huggingface.co/hipfire-models)
and the per-model repositories recorded in the dynamic registry.

Discord: <https://discord.gg/F3BaywB8Rs>

## MQ4R + Redline

MQ4R is the performance-oriented Qwen 3.6 35B-A3B SKU. It combines
uniform MQ4 attention and gate-side weights with graded routed experts
and the fused gate path.

```bash
hipfire pull qwen3.6:35b-a3b-mq4r
hipfire run qwen3.6:35b-a3b-mq4r \
  "Implement a bounded lock-free queue in Rust."
```

The model is 18.7 GB and requires approximately 22 GB of available
VRAM. MQ4R prioritizes throughput; use the default MQ4P model or MFP4,
MQ5, or MQ6 when quality matters more than maximum decode speed.

Redline is hipfire's in-tree dispatch and retained-replay substrate. It
records the actual kernel graph, derives resource dependencies, retains
invariant command state, and lowers validated paths through public ROCr
queue interfaces.

Redline supports the RDNA architecture family from RDNA1 through RDNA4.
Optimized routes remain architecture- and workload-specific. Unsupported
graphs, failed shadow validation, ABI mismatches, queue faults, and model
changes fail closed to ordinary HIP dispatch.

<p align="center">
  <img width="1200" alt="Hipfire runtime and Redline validated replay path, with fail-closed ordinary HIP fallback" src="docs/assets/readme/redline-runtime.svg" />
</p>

### Qwen 3.6 35B-A3B MQ4R performance

Ordinary autoregressive decode with Q8 KV. No MTP, DFlash, speculative
acceptance, reduced-output benchmark, or manual clock pinning.

<p align="center">
  <img width="1200" alt="MQ4R ordinary autoregressive decode performance: gfx1100 253.3 tokens per second, gfx1151 115.1 tokens per second, gfx1201 203.9 tokens per second" src="docs/assets/readme/mq4r-performance.svg" />
</p>

| GPU | Architecture | TG128 AR | 8-turn average | Final-turn context | Final-turn speed | Health |
|---|---:|---:|---:|---:|---:|---|
| Radeon RX 7900 XTX | gfx1100 | **253.3 tok/s** | **191.0 tok/s** | 18.2K | **160.3 tok/s** | 8/8 clean |
| Radeon 8060S / Strix Halo | gfx1151 | **115.1 tok/s** | **92.2 tok/s** | 21.3K | **82.5 tok/s** | 8/8 clean |
| Radeon AI PRO R9700 | gfx1201 | **203.9 tok/s** | **169.5 tok/s** | 22.2K | **146.7 tok/s** | 8/8 clean; both recall probes 3/3 |

Short-context measured ranges:

| Architecture | Minimum | Median | Maximum |
|---|---:|---:|---:|
| gfx1100 | 253.04 | **253.31** | 253.48 tok/s |
| gfx1151 | 115.02 | **115.10** | 115.18 tok/s |
| gfx1201 | 203.42 | **203.93** | 204.04 tok/s |

The eight-turn column is the mean decode speed across the user-facing
multi-turn serving run. Final-turn context is reported explicitly because
the three recorded sessions did not all terminate at exactly 22K.

The gfx1201 campaign raised ordinary MQ4R autoregressive decode from
approximately 110 tok/s to 203.9 tok/s. See the
[gfx1201 campaign report](docs/perf-checkpoints/2026-07-13-redline-mq4r-110-to-204.md)
and [Redline integration boundary](crates/redline-dispatch/HIPFIRE-GRAFT.md).
To reproduce the sealed TG128 route, model, and sampling fixture on a supported
GPU, run `python3 -m tools.redline golden`; the
[Golden Redline guide](docs/GOLDEN-REDLINE.md) also covers making the validated
fixture the serve default and connecting Hermes, Pi, or another
OpenAI-compatible client.

## Curated model registry

The registry currently contains 54 pullable model entries. Run
`hipfire list -r` to see the authoritative live list.

| Registry family | Pull tags and variants |
|---|---|
| Qwen 3.5 dense | Primary: `qwen3.5:0.8b`, `qwen3.5:2b`, `qwen3.5:4b`, `qwen3.5:9b`, `qwen3.5:27b`; MQ3: `qwen3.5:2b-mq3`, `qwen3.5:4b-mq3`, `qwen3.5:9b-mq3`, `qwen3.5:27b-mq3`; MQ6: `qwen3.5:0.8b-mq6`, `qwen3.5:2b-mq6`, `qwen3.5:4b-mq6`, `qwen3.5:9b-mq6`, `qwen3.5:27b-mq6`; legacy HF6: `qwen3.5:2b-hf6`; drafts: `qwen3.5:9b-draft`, `qwen3.5:27b-draft`, `qwen3.5:27b-draft-mq3` |
| Qwen 3.5 MoE | `qwen3.5:35b-a3b` |
| Qwen 3.6 dense | `qwen3.6:27b`, `qwen3.6:27b-mq3`, `qwen3.6:27b-draft`, `qwen3.6:27b-draft-mq3` |
| Qwen 3.6 35B-A3B | `qwen3.6:35b-a3b` (MQ4P default), `qwen3.6:35b-a3b-mq2`, `qwen3.6:35b-a3b-mq3p`, `qwen3.6:35b-a3b-mq4p`, `qwen3.6:35b-a3b-mfp4`, `qwen3.6:35b-a3b-mq4r`, `qwen3.6:35b-a3b-mq5`, `qwen3.6:35b-a3b-mq6` |
| DeepSeek V4 Flash | `deepseek-v4-flash` |
| MiniMax-M2.7 | `minimax-m2.7` |
| North-Mini-Code-1.0 | `north-mini-code` |
| Qwen3 standard attention | `qwen3:0.6b`, `qwen3:8b` |
| Carnice tool-use | `carnice:9b`, `carnice:27b`, `carnice:9b-mq6`, `carnice:27b-mq6` |
| Qwopus | `qwopus:4b`, `qwopus:9b`, `qwopus:27b`, `qwopus:4b-mq6`, `qwopus:9b-mq6`, `qwopus:27b-mq6`, `qwopus3.6:27b-coder` |
| LFM2.5 | `lfm2.5:350m`, `lfm2.5:1.2b`, `lfm2.5:1.2b-thinking`, `lfm2.5:8b-a1b` |
| NEX N2 Mini | `nex-n2:mini` |
| VibeThinker-3B | `vibethinker:3b`, `vibethinker:3b-mq6` |

Common aliases include `qwen3.5`, `qwen3.6`, `qwen3`, `carnice`,
`qwopus`, `deepseek4`, `deepseek-v4`, and `vibethinker`.

Carnice uses the Hermes tool-call format. Plain Qwen 3.5 and 3.6 use
their native Qwen XML tool-call format.

See [docs/MODELS.md](docs/MODELS.md) for sizes, minimum VRAM,
recommended sampling settings, sidecars, artifact provenance, and
bring-your-own-model flows through `hipfire quantize`.

## GPU support

| Family | Representative architectures | Notes |
|---|---|---|
| Vega/CDNA | `gfx906`, `gfx908`, `gfx940`-`gfx942` | Native wave64 HIP paths |
| RDNA1 | `gfx1010`-`gfx1013` | Portable HIP and Redline dispatch support |
| RDNA2 | `gfx1030`-`gfx1032` | Portable HIP and Redline dispatch support |
| RDNA3 | `gfx1100`-`gfx1103` | Architecture-tuned kernels and validated MQ4R route |
| RDNA3.5 | `gfx1150`-`gfx1152` | Architecture-tuned kernels and validated MQ4R route |
| RDNA4 | `gfx1200`, `gfx1201` | WMMA paths and validated retained-PM4 MQ4R route |

Architecture-specific kernels are selected through typed dispatch tables.
Unsupported specializations return to the correct portable or architecture
fallback instead of being applied to neighboring GPU families.

## Why

AMD GPUs are capable inference devices, but tuning and runtime support vary
widely across consumer, professional, APU, and datacenter products.

hipfire supplies its own Rust runtime, model implementations, quantization
formats, dispatch layer, and HIP kernels. ROCm is loaded dynamically; there
is no Python, PyTorch, CUDA translation layer, or third-party inference
engine in the hot path.

Redline removes launch overhead after the ordinary graph has been proven
safe. It does not replace the model implementation or bypass hipfire's
correctness gates.

## Performance snapshots

### Historical — 7900 XTX (gfx1100)

Historical decode snapshot measured with the then-default asym3 KV
configuration (FlashAttention auto):

| Model | hipfire decode | hipfire prefill (peak) | vs ollama Q4_K_M |
|---|---:|---:|---:|
| Qwen 3.5 0.8B | **391** | 7383 | **2.10×** decode |
| Qwen 3.5 4B | **180** | 2487 | **1.78×** decode |
| Qwen 3.5 9B | **132** | 1663 | **1.71×** decode |
| Qwen 3.5 27B | **47** | 478 | — |

Historical DFlash measurements using the legacy asym3 / max=120 method
reached **218 tok/s peak on 27B HumanEval/53** (4.45× over AR) and
**372 tok/s peak on 9B**. These numbers are retained for historical
context; they are not current performance baselines. Current DFlash
claims use q8 / max=256 with prompt and binary hashes recorded. See
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the full per-genre table
and methodology notice.

### RDNA4 (gfx1201, Radeon AI PRO R9700)

| Model | Config | Decode tok/s |
|---|---|---:|
| Qwen2 1.5B HFQ4 | single GPU | **266** |
| DeepSeek V4 Flash (82 GB MQ2-Lloyd) | 4× R9700, `hipfire serve --tp 4` (EP) | **25.6** |
| Gemma 4 12B MQ4 | single GPU (integration branch, pre-merge) | **~47** |

Experimental long-context compression and eviction are opt-in. PFlash is off
by default, TriAttention sidecars do not auto-attach, and CASK m-folding is
disabled. Generate a sidecar with `hipfire sidecar-gen <model>`, then set
`memory.cask.sidecar` to its exact path (or explicitly enable
`memory.cask.auto_attach`). Set `memory.cask.enabled=true` only when m-folding
is intended. See [CONFIG.md](docs/CONFIG.md) for details.

## Install

Linux with ROCm 6 or newer:

```bash
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.sh | bash

# Or install the integration branch for testing:
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.sh \
  | bash -s -- --branch beta
```

RDNA4 requires ROCm 6.4 or newer. gfx1151 requires ROCm 7.2 or newer.
Run `hipfire --version` for a concise build ID or `hipfire version` to compare
the installed binary, managed source checkout, and daemon. Managed Linux
installs can switch revisions with `hipfire update @beta`,
`hipfire update --tag v0.2.1`, or `hipfire update --commit <sha>`.

To uninstall a managed Linux install while keeping downloaded models and
settings:

```bash
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/uninstall.sh | bash
```

Add `--dry-run` to preview it. `--purge` also deletes all data under
`~/.hipfire` and requires explicit confirmation.

For Windows, source builds, and verifying the install:
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## NixOS

First-class support via Nix flake. See [docs/NIXOS.md](docs/NIXOS.md).

```bash
nix develop github:warpfront/hipfire  # dev shell with Rust + ROCm
nix build github:warpfront/hipfire    # build package
```

NixOS module:

```nix
{
  inputs.hipfire.url = "github:warpfront/hipfire";
  # then in configuration.nix:
  services.hipfire.enable = true;
  services.hipfire.gpuTargets = [ "gfx1100" ];
}
```

## Container (podman/docker)

A multi-stage `Containerfile` builds a slim deliverable inference image and a
full-toolchain GPU gate-runner for reproducible PR/dev-build validation. See
[docs/CONTAINER.md](docs/CONTAINER.md).

```bash
podman build -f Containerfile --target runtime -t hipfire .
podman run --rm -it --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined \
  -v hipfire-models:/root/.hipfire/models \
  -v hipfire-kcache:/var/cache/hipfire \
  hipfire run qwen3.5:4b "2+2="
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
| [CONTAINER.md](docs/CONTAINER.md) | Runtime and GPU gate-runner containers |
| [multi-gpu.md](docs/multi-gpu.md) | Pipeline-parallel (pp≥2) — memory budget, deployment, refusals |
| [methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md) | Bench protocol — read before claiming a perf win |
| [HIPFIRE-GRAFT.md](crates/redline-dispatch/HIPFIRE-GRAFT.md) | Redline integration and enablement boundary |

## License

hipfire is licensed under **Apache-2.0** as of v0.3.0. See
[LICENSE](LICENSE), [LICENSE-APACHE](LICENSE-APACHE), and
[NOTICE](NOTICE) for details. Releases up to and including v0.2.1 were
dual MIT/Apache-2.0; that grant is not revoked.

Individual files whose substantive authors have not elected Apache-2.0
remain MIT-licensed (see [LICENSE-MIT](LICENSE-MIT)) and are identified
by their per-file `SPDX-License-Identifier`. No contributor's file was
relicensed in absentia. New contributions default to Apache-2.0 via DCO
sign-off. See [CONTRIBUTING.md](CONTRIBUTING.md) for the contributor
side and
[docs/governance/relicense-2026-05.md](docs/governance/relicense-2026-05.md)
for the decision record (including the 2026-05-19 course correction and
the v0.3.0 move to outbound Apache-2.0).

Original architectural innovations originating in hipfire are
catalogued in [PRIOR-ART.md](PRIOR-ART.md); derivative works
(including reimplementations informed by hipfire's design) should
attribute the corresponding inventions per [AGENTS.md](AGENTS.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Install local hooks with
`./scripts/install-hooks.sh`. The no-GPU CI subset is
`./scripts/no-gpu-ci.sh`; it does not replace the hardware gates.
**There is no single canonical correctness gate** — the retired
`scripts/coherence-gate*.sh` batteries are not acceptance evidence and
`coherence-gate-dflash.sh` no longer exists. Select the route your change
owes from [docs/VALIDATION.md](docs/VALIDATION.md): `scripts/serve_harness.py`
for generation and state-lifecycle changes, `scripts/redline_daemon_harness.py`
for kernel, dispatch, graph, or Redline-replay changes. Architecture-specific
work must also pass its channel test, anti-bleed checks, and relevant speed
gate. Don't bypass the gates with `--no-verify` — see
[methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md).
