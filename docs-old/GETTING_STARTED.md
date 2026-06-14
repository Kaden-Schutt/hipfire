# Getting started

## Install

Linux with ROCm 6+ installed and an AMD RDNA GPU:

```bash
curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.sh | bash
```

From a source checkout, install hipfire into `~/.hipfire/bin/`:

```bash
./install.sh
# or
make install
```

The installer detects your GPU arch (`gfx1010` / `gfx1030` / `gfx1100` / etc.),
fetches matching pre-compiled kernel blobs, drops the daemon and quantizer
binaries into `~/.hipfire/bin/`, and adds a wrapper to `~/.local/bin/`. Make
sure `~/.local/bin` is on your `PATH`.

For Windows (native, with the AMD HIP SDK):

```powershell
irm https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.ps1 | iex
```

The installer detects your AMD GPU via `Win32_VideoController`, downloads
the prebuilt `daemon.exe` from the latest GitHub release, sets up the
`bun`-based CLI, and runs `daemon.exe --precompile` to JIT-compile kernels
for your arch into `~\.hipfire\bin\kernels\compiled\<arch>\`. This requires
the [AMD HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
to be installed (provides `hipcc.bat` + `amdhip64.dll`).

If hipcc is not available, kernels can still load from any prebuilt blobs
in the repo. To force a fresh compile of the full kernel set:

```powershell
.\scripts\compile-kernels.ps1 gfx1100   # or your arch
```

For WSL2 (Linux paths, `/dev/kfd` available): inside Ubuntu under WSL2
run `sudo amdgpu-install --usecase=wsl` first, then the Linux installer
above.

For source builds:

```bash
git clone https://github.com/Kaden-Schutt/hipfire
cd hipfire
cargo build --release
cargo build --release -p hipfire-quantize
```

For normal source-checkout builds, `make build` is the short form for the
release build.

On machines with an AMD XDNA NPU (Ryzen AI), enable the NPU SwiGLU, RMSNorm,
and RoPE kernels by adding the `npu-kernels` feature. This requires XRT 2.x and
the `mlir_aie` Python package in `~/.venv`; the build scripts auto-detect the
NPU generation via pyxrt and write xclbin artifacts to `target/npu/`:

```bash
cargo build --release --features npu-kernels
```

To target specific generations or override the default sizes:

```bash
HIPFIRE_NPU_TARGETS=npu1,npu2 HIPFIRE_NPU_HIDDEN_SIZES=8960,18944 \
    HIPFIRE_NPU_RMSNORM_SIZES=1536,3584 \
    HIPFIRE_NPU_ROPE_CONFIGS=8:2:256:64 \
    cargo build --release --features npu-kernels
```

You can also build individual xclbins directly without Cargo:

```bash
python tools/npu/build_qwen35_swiglu.py --hidden-size 8960   # auto-detects NPU
python tools/npu/build_qwen35_rmsnorm.py --hidden-size 1536  # Qwen3.5-1.5B
python tools/npu/build_qwen35_rmsnorm.py --hidden-size 3584  # Qwen3.5-7B
# RoPE (Q + K xclbins, Qwen3.5-1.5B dense config):
python tools/npu/build_qwen35_rope.py --n-heads 8 --n-kv-heads 2 --head-dim 256 --n-rot 64
# QK head norm (Q + K xclbins):
python tools/npu/build_qwen35_headnorm.py --n-heads 8 --n-kv-heads 2 --head-dim 256
# Attention output gate (when config.attn_output_gate=true):
python tools/npu/build_qwen35_attn_gate.py --n-heads 8 --head-dim 256
# Attention score softmax (builds xclbins for ctx_len ∈ {64,128,256,512}):
python tools/npu/build_qwen35_softmax.py --n-heads 8 --ctx-lens 64,128,256,512
# Fused QK head norm + RoPE (replaces separate headnorm + rope dispatches; 4 → 2 per layer):
python tools/npu/build_qwen35_headnorm_rope.py --n-heads 8 --n-kv-heads 2 --head-dim 256
```

## Verify

```bash
hipfire diag
```

Confirms ROCm version, HIP runtime, GPU arch, VRAM, and that the kernel
blobs match. If anything is off it prints a targeted error rather than
failing later at first inference.

## First run

```bash
hipfire pull qwen3.5:4b                         # ~2.6 GB download
hipfire run  qwen3.5:4b "Explain FFT in one line"
```

Cold start is 2–5 s while weights upload to VRAM and the kernel cache
warms. After that decode is ~165 tok/s on a 7900 XTX.

## Background daemon

For repeated calls or programmatic use, run the daemon in the background
and hit it over HTTP:

```bash
hipfire serve -d                                 # detaches, pre-warms default_model
hipfire run qwen3.5:4b "..."                     # auto-routes through HTTP, skips cold-start
hipfire stop                                     # graceful shutdown
```

The daemon speaks an OpenAI-compatible API on `localhost:11435`. See
[SERVE.md](SERVE.md) for the HTTP surface.

## Configure

```bash
hipfire config                                   # interactive TUI for global keys
hipfire config qwen3.5:9b                        # per-model overlay
```

Common overrides: `temperature` (default 0.30), `kv_cache` (default
`asym3`), `dflash_mode` (default `off`). Full key list in
[CONFIG.md](CONFIG.md).

## Long context: KV cache eviction

For long-context prompts (16K+ tokens), CASK-based eviction prevents OOM by
capping physical VRAM regardless of the advertised `max_seq`. For HuggingFace
models pulled via `hipfire pull`, a sidecar is auto-attached — just enable
eviction with:

```bash
hipfire config cask-profile balanced   # or conservative / aggressive-vram
```

The daemon automatically discovers the shipped TriAttention sidecar beside
the weights and sets `cask_sidecar` for you. No manual path needed.

**Note:** eviction is disabled by default even when a sidecar exists —
you must set a CASK profile (`balanced`, `conservative`, or `aggressive-vram`) to activate it.

For custom or quantized models, generate the sidecar first:

```bash
hipfire sidecar-gen ~/models/my-finetune-mq4.hfq --corpus corpus.txt
hipfire config cask-profile balanced
```

See [CONFIG.md](CONFIG.md) for profiles, knobs, and safety constraints.

## What to read next

- [MODELS.md](MODELS.md) — supported model tags + how to bring your own
  (HuggingFace, local safetensors, GGUF).
- [CLI.md](CLI.md) — full subcommand reference.
- [QUANTIZE.md](QUANTIZE.md) — quantize a finetune or a GGUF you already
  have.
- [BENCHMARKS.md](BENCHMARKS.md) — measured tok/s per arch.
- [ARCHITECTURE.md](ARCHITECTURE.md) — high-level engine design if you
  want to contribute.
