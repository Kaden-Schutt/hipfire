# hipfire

LLM inference for AMD RDNA GPUs. Rust + HIP. Single binary. No Python in the hot path. Ollama-style UX.

```bash
hipfire pull qwen3.5:9b
hipfire run  qwen3.5:9b "What is the capital of France?"
hipfire serve -d       # background daemon on port 11435 (OpenAI API compatible)
```

Current release: **v0.1.7-alpha.2** — hotfix for `hipfire config` TUI crash + A3B DFlash default-off. FlashTriAttn long-ctx speculative decoding (+42% tok/s on 9B), CASK m-folding KV eviction, Qwen3.6-35B-A3B MoE support, MI300X wave64 port. See [CHANGELOG.md](CHANGELOG.md). Previous: **v0.1.6 "deltacut"**.

## Why

`llama.cpp + ROCm` works on RDNA but is painful: upstream ROCm officially supports
only a handful of datacenter cards, consumer RDNA is a second-class citizen, and
setup is an adventure. hipfire targets the entire RDNA family (RDNA1 through RDNA4,
consumer + pro + APU) with a single Rust binary that ships pre-compiled kernel
blobs when possible and JIT-compiles the rest through HIP. No Python, no PyTorch,
no ROCm userspace stack at runtime.

**RDNA3 — RX 7900 XTX** (gfx1100, 24 GB) — primary target. WMMA prefill + hipGraph decode + Q8 KV:

| Model | decode | prefill (peak) | effective BW |
|---|---:|---:|---:|
| Qwen 3.5 0.8B MQ4 | **391 tok/s** | **7383 tok/s** | 200 GiB/s |
| Qwen 3.5 4B MQ4   | **180 tok/s** | **2487 tok/s** | 433 GiB/s |
| Qwen 3.5 9B MQ4   | **132 tok/s** | **1663 tok/s** | **654 GiB/s** |
| Qwen 3.5 27B MQ4  | **47 tok/s**  | **478 tok/s**  | **651 GiB/s** |

9B and 27B decode saturate ~650 GiB/s of the 7900 XTX's 960 GB/s peak — 68%
BW efficiency end-to-end (weights + KV + activations). Prefill is WMMA-bound
on the MQ4 fused projections.

### vs ollama (Q4_K_M GGUF via llama.cpp/ROCm) — 7900 XTX

Same machine, same models. hipfire asym3 MQ4 vs ollama default Q4_K_M
(llama.cpp ROCm backend). Matched ~140-token and ~530-token prompts and
matched 128-token generation lengths fed to both (ollama via
`/api/generate` with `num_predict=128`, numbers from its own reported
`prompt_eval_duration` / `eval_duration`).

| Model | hf pp128 | oll pp128 | hf pp512 | oll pp512 | hf decode | oll decode | decode× |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen 3.5 0.8B | **10,861** | 4,622 | **12,962** | 7,117 | **353** | 168 | **2.10×** |
| Qwen 3.5 4B   | **3,304**  | 1,972 | **3,321**  | 2,670 | **165** | 93  | **1.78×** |
| Qwen 3.5 9B   | **1,920**  | 1,428 | 1,919      | **1,970** | **122** | 71  | **1.71×** |

Decode is the user-visible number for interactive chat and hipfire wins
1.7–2.1× across the board. Prefill is more nuanced: hipfire wins
decisively on 0.8B/4B and at pp128 for 9B (batched MQ4 fused projections
saturate WMMA on small matmuls where ollama's per-token GGUF dequant
can't), but ollama edges past hipfire at pp512 on 9B (1,970 vs 1,919
tok/s) — the GEMMs are large enough there to saturate even without WMMA.
Harness: [`cli/bench_vs_ollama.ts`](cli/bench_vs_ollama.ts).

Other arches:

- **RDNA2** (gfx1030 — V620 Pro, 32 GB): 250 / 65 / 22 tok/s decode (0.8B / 9B / 27B)
- **APU** (gfx1013 — BC-250, 14 GB shared): 207 / 77 / 47 tok/s decode (0.8B / 4B / 9B), 27B won't fit
- **RDNA1** (gfx1010 — RX 5700 XT, 8 GB): 190 / 61 / 43 tok/s (0.8B / 4B / 9B HF4 — MQ4 numbers pending 0.1.5 retest)

Full per-architecture numbers including long-context prefill sweeps: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Install

### Linux (any RDNA card with ROCm 6+ installed)

```bash
curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.sh | bash
```

This:
1. Detects your GPU arch (`gfx1010`/`gfx1030`/`gfx1100`/etc)
2. Downloads the matching pre-compiled kernel blobs
3. Installs the `daemon` + `hipfire-quantize` binaries to `~/.hipfire/bin/`
4. Drops a `hipfire` wrapper script into `~/.local/bin/` (add to `PATH`)

### Windows (WSL2 only)

Native Windows is not supported — hipfire needs `/dev/kfd`. Use WSL2:

```powershell
wsl --install -d Ubuntu
# then inside WSL2:
sudo amdgpu-install --usecase=wsl
curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.sh | bash
```

### From source

```bash
git clone https://github.com/Kaden-Schutt/hipfire
cd hipfire
cargo build --release --features deltanet --example daemon -p engine
cargo build --release -p hipfire-quantize
```

Run `hipfire diag` afterwards to verify ROCm/HIP/GPU detection and see a
targeted error message for anything that's off.

## Getting started

Four commands:

```bash
hipfire diag                                       # sanity check GPU + HIP + kernels
hipfire pull qwen3.5:4b                            # ~2.6 GB download
hipfire run  qwen3.5:4b "Explain FFT in one line"  # generate
hipfire config                                     # interactive TUI — kv_cache, temperature, etc.
```

Longer walk-through: **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)**.

## CLI reference

```
hipfire pull <model>                  Download model from HuggingFace
hipfire run <model> [prompt]          Generate text (auto-pulls; uses running serve if any)
hipfire serve [port] [-d]             Start OpenAI-compatible HTTP server (-d = background)
hipfire stop                          Stop the background daemon
hipfire quantize <hf-id|local-dir>    Quantize any Qwen 3.5 model to MQ4/MQ6 (CPU-only)
hipfire config                        Interactive settings TUI
hipfire config <tag>                  Per-model overlay (e.g. hipfire config qwen3.5:9b)
hipfire list [-r]                     Show local models (-r: show available too)
hipfire ps                            Running daemons / quantize jobs / HF uploads
hipfire bench <model>                 Benchmark prefill + decode tok/s
hipfire diag                          GPU, VRAM, HIP version, kernels, models
hipfire rm <model>                    Delete model
hipfire update                        Pull latest code, rebuild, update kernel blobs
```

## Model catalog

All models are MQ4 by default (FWHT-rotated 4-bit, quality-gated — matches Q8
output at ~Q4 bandwidth). MQ6 variants available with `:<size>-mq6` suffix.

| Tag | Size | VRAM floor | Notes |
|---|---|---|---|
| `qwen3.5:0.8b` | 0.55 GB | 1 GB | Tiny, DeltaNet |
| `qwen3.5:4b` | 2.6 GB | 4 GB | Best speed/quality balance |
| `qwen3.5:9b` | 5.3 GB | 6 GB | Default `serve` pre-warm |
| `qwen3.5:27b` | 15 GB | 16 GB | Needs 16 GB+ VRAM |
| `qwen3.5:{size}-mq6` | 1.47× | +2 GB | Higher quality, larger file |
| `qwopus:{4,9,27}b` | Qwen 3.5 arch | as above | Jackrong reasoning fine-tune |
| `carnice:{9,27}b` | Qwen 3.5 arch | as above | kai-os Hermes tool-use |

Full list: `hipfire list -r` or [docs/MODELS.md](docs/MODELS.md).

## Quantize your own

```bash
# From a HuggingFace model (auto-download + quantize + upload in one shot):
hipfire quantize Jackrong/Qwopus3.5-4B-v3 \
    --both \
    --upload schuttdev/hipfire-qwopus-4b \
    --create-repo --install \
    --register qwopus:4b

# From a local safetensors directory:
hipfire quantize ./my-finetune --format mq4 -o finetune.mq4
```

The quantizer is CPU-only (minutes to tens of minutes depending on model size).
It produces a single `.mq4` or `.mq6` file that the daemon loads directly.

## Configuration

`hipfire config` opens an interactive TUI. All keys are arrow-key + space/enter
driven; values persist in `~/.hipfire/config.json`.

```
▸ kv_cache         asym3          (default)  auto q8 asym4 asym3 asym2
  flash_mode       auto           (default)  auto always never
  default_model    qwen3.5:9b     (default)
  temperature      0.30           (default)  0.0–2
  top_p            0.80           (default)  0.0–1
  repeat_penalty   1.05           (default)  1.0–3
  max_tokens       512            (default)  1–131072
  max_seq          32768          (default)  512–524288
  thinking         on             (default)  on off
  max_think_tokens 0              (default)  0–32768
  port             11435          (default)  1–65535
  idle_timeout     300            (default)  0–86400

  per-model configs  no overrides   → enter to open model picker
```

Per-model overrides: `hipfire config qwen3.5:9b` — sparse JSON overlay at
`~/.hipfire/per_model_config.json`. Rows show `(inherited)` vs `(overridden)`
so you can see exactly what diverges from global.

Environment variables (override config for one invocation):

```
HIPFIRE_KV_MODE=asym3|q8|asym4|asym2
HIPFIRE_ATTN_FLASH=auto|always|never
HIPFIRE_LOCAL=1                     # force run to spawn its own daemon (skip serve HTTP)
HIPFIRE_HIPCC_EXTRA_FLAGS=...       # one-off JIT flags (e.g. -mcumode)
```

## Serve (OpenAI-compatible HTTP)

```bash
hipfire serve -d                 # background daemon, pre-warms default_model
hipfire ps                       # confirm it's running
curl -N http://localhost:11435/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"hi"}],"stream":true}'
hipfire stop                     # graceful shutdown
```

Idle eviction is on by default — `idle_timeout=300` seconds frees VRAM after
five minutes of no requests; next request reloads.

`hipfire run` automatically uses the running serve's HTTP surface when it's
up (skips the 2-5s cold-start cost). `HIPFIRE_LOCAL=1 hipfire run` forces
local-spawn instead.

## Architecture

- **`crates/engine`** — model loader, Qwen 3.5 + Qwen 3 forward passes, KV cache, DeltaNet state
- **`crates/rdna-compute`** — kernel dispatch, hipGraph capture, pre-compiled blob loader with hash-verified JIT fallback
- **`crates/hip-bridge`** — safe Rust FFI wrapping `libamdhip64.so`
- **`crates/hipfire-quantize`** — CPU-side safetensors → `.mq4`/`.mq6` encoder
- **`crates/redline`** — direct-KMD dispatch research (future, skips HIP runtime entirely)
- **`kernels/src/*.hip`** — HIP kernels (GEMV, fused projections, flash attention, asym K quant, etc)
- **`cli/index.ts`** — Bun/TypeScript CLI + OpenAI-compatible HTTP server

Deeper walkthrough: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Docs

- [GETTING_STARTED.md](docs/GETTING_STARTED.md) — install → first run
- [BENCHMARKS.md](docs/BENCHMARKS.md) — measured perf per GPU arch
- [QUANTIZATION.md](docs/QUANTIZATION.md) — MagnumQuant (MQ4/MQ6), asym KV, comparison to HF4/HF6 legacy
- [KV_CACHE.md](docs/KV_CACHE.md) — asym3/asym4/asym2 design (Lloyd-Max rotated K + Q8 V)
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — engine, dispatch, kernel layout
- [DELTANET.md](docs/DELTANET.md) — Qwen 3.5 linear attention path
- [MODELS.md](docs/MODELS.md) — supported models + HuggingFace repo layout
- [LEGACY_HFQ.md](docs/LEGACY_HFQ.md) — retired HF4/HF6 format (still loads)
- [CONTRIBUTING.md](CONTRIBUTING.md) — build, test, PR flow

## Supported hardware

Everything with an RDNA GPU and amdgpu kernel driver:

| Arch | Examples | Default KV | Status |
|---|---|---|---|
| gfx1010 | RX 5700 XT | asym2 | tested |
| gfx1013 | BC-250 APU | asym2 | tested |
| gfx1030 | V620 Pro, RX 6800 XT | asym3 | tested |
| gfx1031 | RX 6700 XT | asym3 | expected to work |
| gfx1032 | RX 6600 XT | asym2 | expected to work |
| gfx1100 | RX 7900 XTX | asym3 | primary target |
| gfx1101 | RX 7900 XT | asym3 | expected |
| gfx1102 | RX 7800 XT | asym3 | expected |
| gfx1151 | Strix Halo | asym2 | APU path |
| gfx1200 | RX 9070 XT (RDNA4) | asym3 | expected |

`hipfire diag` prints your arch + the auto-selected KV default.

## Troubleshooting

- **`/health` times out on first `serve -d`** — kernel JIT on slow hardware can
  take 30s-2min on a cold cache. Tail `~/.hipfire/serve.log` to watch layer
  loading progress.
- **`hipcc compilation failed: hip/hip_runtime.h not found`** — your ROCm install
  doesn't auto-inject `/opt/rocm/include`. Run `hipfire update` (fix lands in v0.1.5+).
- **Multi-turn says "Kendall" instead of "Kaden"** — you're on a pre-0.1.5 build
  with `givens4` KV. Update + set `hipfire config set kv_cache asym3`.
- **Multiple serves fighting port 11435** — `pkill -9 daemon bun; rm ~/.hipfire/serve.pid; hipfire serve -d`.

## License

MIT. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The quality gate (`./scripts/quality-gate.sh --fast`)
enforces byte-exact greedy output — any change to kernels, quant formats, dispatch,
fusion, rotation, rmsnorm, or the forward pass must pass before committing.
