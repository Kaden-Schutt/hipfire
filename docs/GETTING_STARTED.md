# Getting started

## Install

Linux with ROCm 6+ installed and an AMD RDNA GPU:

```bash
curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.sh | bash
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
cargo build --release --features deltanet --example daemon -p hipfire-runtime
cargo build --release -p hipfire-quantize
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
`auto` → per-arch: `fwht3` on most arches, `fwht2` on tight-memory
parts), `dflash_mode` (default `auto`). Full key list in
[CONFIG.md](CONFIG.md).

## Long context: KV cache eviction

For long-context prompts (16K+ tokens), CASK-based eviction prevents OOM by
capping physical VRAM regardless of the advertised `max_seq`. For HuggingFace
models pulled via `hipfire pull`, a sidecar is auto-attached — just enable
eviction with:

```bash
hipfire config cask-profile balanced   # or conservative / aggressive-vram
```

The daemon automatically discovers the shipped `.triattn.bin` beside the
weights and sets `cask_sidecar` for you. No manual path needed.

**Note:** eviction is disabled by default even when a sidecar exists —
you must set a CASK profile (`balanced`, `conservative`, or `aggressive-vram`) to activate it.

For custom or quantized models, generate the sidecar first:

```bash
hipfire sidecar-gen ~/models/my-finetune.mq4 --corpus corpus.txt
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
