# hipfire

Fast local LLM inference for AMD GPUs. Rust + HIP + Redline. No Python
in the hot path. Ollama-style UX.

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

**Current stable release: v0.2.1.** The next release is **v0.3.0**
(MQ4R + Redline across RDNA). See [CHANGELOG.md](CHANGELOG.md).

Curated weights: [huggingface.co/hipfire-models](https://huggingface.co/hipfire-models)
and the per-model repositories recorded in the dynamic registry.

Discord: <https://discord.gg/F3BaywB8Rs>

## Quickstart

Linux with ROCm 6 or newer:

```bash
curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.sh | bash
hipfire diag
hipfire pull qwen3.5:4b
hipfire run qwen3.5:4b "Explain FFT in one line"
```

- RDNA4 (`gfx1200`/`gfx1201`): ROCm 6.4+
- Strix Halo / `gfx115x` (`gfx1150`–`gfx1152`): ROCm 7.2+
- Windows, source builds, verify steps: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
- NixOS flake/module: [docs/NIXOS.md](docs/NIXOS.md)
- Containers: [docs/CONTAINER.md](docs/CONTAINER.md)

## Models (dynamic registry)

The authoritative pull list is **live**, not this README:

```bash
hipfire list -r
```

Registry presence is not runtime admission. Sizes, VRAM floors, sampling
defaults, sidecars, provenance, and bring-your-own flows live in
[docs/MODELS.md](docs/MODELS.md). Operator surface:
[docs/CLI.md](docs/CLI.md), [docs/SERVE.md](docs/SERVE.md),
[docs/CHAT.md](docs/CHAT.md), [docs/CONFIG.md](docs/CONFIG.md),
[docs/env-vars.md](docs/env-vars.md).

## GPU support

hipfire targets AMD RDNA/CDNA GPUs with HIP/ROCm-direct kernels and typed
dispatch. Representative families include Vega/CDNA, RDNA1–RDNA4 (including
`gfx115x` and `gfx1200`/`gfx1201`). Capability on a chip is not blanket route
certification.

Architecture-specific kernels are selected through typed dispatch tables. A
portable path is used **only when one exists** for that specialization;
unsupported combinations fail closed (`UnsupportedVariant` / no silent
fallback). Install and host prerequisites:
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md). Multi-GPU ops:
[docs/multi-gpu.md](docs/multi-gpu.md). Retained-replay admission:
[docs/REDLINE.md](docs/REDLINE.md).

## MQ4R and Redline (next release)

**MQ4R** is the performance-oriented Qwen 3.6 35B-A3B SKU (uniform MQ4
attention/gate-side weights, graded routed experts, fused gate path).
Pull via the live registry (`hipfire list -r`); VRAM and sampling live
in [docs/MODELS.md](docs/MODELS.md). Prefer MQ4P / MFP4 / MQ5 / MQ6 when
quality matters more than maximum decode speed.

**Redline** is hipfire's in-tree dispatch and retained-replay substrate.
It records the kernel graph, derives resource dependencies, retains
invariant command state, and lowers validated paths through public ROCr
queue interfaces. Implementation capability across RDNA is **not** the
same as automatic product admission or timed-arm route proof. Normative
certification and evidence policy:
[docs/REDLINE.md](docs/REDLINE.md). Integration boundary:
[crates/redline-dispatch/HIPFIRE-GRAFT.md](crates/redline-dispatch/HIPFIRE-GRAFT.md).

## Performance (historical / measured)

Published tables and campaign checkpoints are **dated measurements**, not
live floors, defaults, or admissions:

- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — historical snapshots (incomplete evidence manifests)
- [docs/perf-checkpoints/](docs/perf-checkpoints/) — immutable dated campaigns
  (e.g. gfx1201 MQ4R campaign report)
- How to measure correctly:
  [docs/methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md)

Do not treat README snippets, registry tags, or harness exits as route
admission. Admissions are machine-recorded only in
[docs/admissions.yml](docs/admissions.yml) (fail closed when empty).

## Why

AMD GPUs are capable inference devices, but tuning and runtime support
vary across consumer, professional, APU, and datacenter products.

hipfire supplies its own Rust runtime, model implementations, quantization
formats, dispatch layer, and HIP kernels. ROCm is loaded dynamically;
there is no Python, PyTorch, CUDA translation layer, or third-party
inference engine in the hot path.

A retained Redline route changes how an eligible forward is submitted; it
does not replace the model implementation or turn capability into
certification.

## Documentation

Navigation, lifecycle labels, and ownership map (start here):

**[docs/INDEX.md](docs/INDEX.md)**

| Page | Topic |
|---|---|
| [GETTING_STARTED.md](docs/GETTING_STARTED.md) | Install, first run |
| [CLI.md](docs/CLI.md) | Subcommands and flags |
| [MODELS.md](docs/MODELS.md) | Registry, BYO, sidecars |
| [SERVE.md](docs/SERVE.md) / [CHAT.md](docs/CHAT.md) | HTTP API and chat UX |
| [CONFIG.md](docs/CONFIG.md) / [env-vars.md](docs/env-vars.md) | Config and environment |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Crate layout, request lifecycle |
| [QUANTIZATION.md](docs/QUANTIZATION.md) / [QUANTIZE.md](docs/QUANTIZE.md) | Formats and `hipfire quantize` |
| [VALIDATION.md](docs/VALIDATION.md) | Claim → validation route selector |
| [REDLINE.md](docs/REDLINE.md) | Retained-replay certification |
| [methodology/](docs/methodology/) | Perf, arch-port, Kernel Atlas protocols |
| [BENCHMARKS.md](docs/BENCHMARKS.md) | Historical bench tables |

Executable agent skills root: [`.agents/skills/`](.agents/skills/)
(`docs/skills/` is retired/removed — not a second root).

## Inspiration

hipfire's DFlash work was substantially shaped by Davide Ciffa's
[Lucebox DFlash on ggml](https://www.lucebox.com/blog/dflash27b).
Cached snapshot: `.research-cache/lucebox-dflash27b.html`.

gfx906 prefill MMQ and AR-decode optimizations were shaped by community
`llama.cpp` forks targeting Vega 20:

- [iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906)
- [skyne98/llama.cpp-gfx906](https://github.com/skyne98/llama.cpp-gfx906)
  and [skyne98/wiki-gfx906](https://skyne98.github.io/wiki-gfx906/intro.html)

plus the templated MMQ scaffold in `ggml-org/llama.cpp`. Investigation
logs: [docs/perf-checkpoints/](docs/perf-checkpoints/).

## License

hipfire is dual-licensed under MIT or Apache-2.0 at your option. See
[LICENSE](LICENSE), [LICENSE-MIT](LICENSE-MIT),
[LICENSE-APACHE](LICENSE-APACHE), and [NOTICE](NOTICE).

New contributions default to Apache-2.0 via DCO sign-off; existing
contributors' MIT-licensed contributions remain MIT unless they opt in.
Per-file `SPDX-License-Identifier` reflects actual authorship. Contributor
procedure: [CONTRIBUTING.md](CONTRIBUTING.md). Decision record:
[docs/governance/relicense-2026-05.md](docs/governance/relicense-2026-05.md).

Original architectural innovations are catalogued in
[PRIOR-ART.md](PRIOR-ART.md); derivative works (including reimplementations
informed by hipfire's design) should attribute the corresponding inventions
per [AGENTS.md](AGENTS.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Install local hooks with
`./scripts/install-hooks.sh`. The no-GPU CI subset is
`./scripts/no-gpu-ci.sh` — it is not a substitute for GPU/model evidence.
Pick validation routes from [docs/VALIDATION.md](docs/VALIDATION.md).
Do not treat retired `scripts/coherence-gate-*.sh` batteries as current
acceptance. Don't bypass hooks with `--no-verify`. Perf claims follow
[docs/methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md).
