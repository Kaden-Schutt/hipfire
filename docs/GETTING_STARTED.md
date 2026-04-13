# Getting started with hipfire

From zero to inferencing on your AMD RDNA GPU in ~10 minutes.

## 1. Check your hardware

Any AMD GPU with an RDNA die (RDNA 1 through RDNA 4) should work. That covers:

- **Consumer:** RX 5700/XT (RDNA1), RX 6x00/XT (RDNA2), RX 7x00/XT (RDNA3), RX 9x00 (RDNA4)
- **Pro:** V520/V620/W6800/W7900
- **APU:** Steam Deck, BC-250, Ryzen AI (Strix Halo)
- **Datacenter:** MI200/MI300 series (fp16 path — no specific tuning yet)

Not supported: NVIDIA GPUs, Apple Silicon, AMD GCN (pre-RDNA) cards.

## 2. Install ROCm (Linux side)

You need ROCm 6+ and the amdgpu kernel driver. On most distros:

```bash
# Ubuntu / Debian
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_*.deb
sudo apt install ./amdgpu-install_*.deb
sudo amdgpu-install --usecase=rocm

# Arch / CachyOS
sudo pacman -S rocm-hip-sdk rocm-opencl-sdk

# Verify
sudo usermod -aG render,video $USER
# log out + back in, then:
rocminfo | grep -i gfx
```

You should see your GPU architecture (`gfx1030`, `gfx1100`, etc).

For WSL2:

```bash
sudo amdgpu-install --usecase=wsl
```

## 3. Install hipfire

```bash
curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.sh | bash
```

Add `~/.local/bin` to your `PATH` if it isn't already:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 4. Verify the install

```bash
hipfire diag
```

Expected output includes (adapt to your GPU):

```
platform:      Linux (native)
PCI GPUs:
  03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] ...
/dev/dri/:     card1, renderD128
/dev/kfd:      present, readable

rocminfo GPUs:
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
hipcc:         HIP version: 7.2.x
amdgpu module: loaded

daemon:        found
kernels/gfx1100: 87 blobs, 87 hashes

Probing GPU via HIP runtime...
  GPU arch:    gfx1100
  HIP version: 7.2
  VRAM free:   24321 MB
  VRAM total:  24576 MB
  kv default:  asym3 (24 GB VRAM)
  WMMA:        yes (4.1x prefill)
```

If any line says `NOT FOUND` or `NOT LOADED`, `hipfire diag` prints a targeted
next step (install ROCm, add to render group, etc).

## 5. Pull a model

```bash
hipfire pull qwen3.5:4b      # ~2.6 GB, best speed/quality balance
```

Other options:

```bash
hipfire pull qwen3.5:0.8b    # 0.55 GB — tiny
hipfire pull qwen3.5:9b      # 5.3 GB — best quality on 8 GB cards
hipfire pull qwen3.5:27b     # 15 GB — needs 16 GB+ VRAM
hipfire pull qwen3.5:9b-mq6  # 7.3 GB — higher quality 9B
```

Models download from HuggingFace (`schuttdev/hipfire-{model}-{size}`).

## 6. Run your first prompt

```bash
hipfire run qwen3.5:4b "Explain the Fourier transform in one paragraph."
```

You'll see kernel compile logs on first run (cached in `/tmp/hipfire_kernels/`
afterwards), then the response streaming back.

## 7. Start the background daemon

The one-shot `run` command spawns + kills a daemon every call (2-5s cold start).
If you're going to send many prompts, run a persistent server:

```bash
hipfire serve -d                           # detaches, pre-warms qwen3.5:9b
hipfire ps                                 # shows daemon + serve state
hipfire run qwen3.5:4b "What's up?"        # automatically uses the running serve over HTTP
hipfire stop                               # shuts it down
```

Any OpenAI-compatible client works against `http://localhost:11435`:

```bash
curl -N http://localhost:11435/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
          "model": "qwen3.5:9b",
          "messages": [{"role":"user","content":"Hi"}],
          "stream": true
        }'
```

The daemon auto-evicts the model after 5 min of inactivity to free VRAM; the
next request reloads it. Tune via `hipfire config set idle_timeout 600` or
disable with `idle_timeout=0`.

## 8. Tweak settings

All settings live in `~/.hipfire/config.json`. Edit interactively:

```bash
hipfire config
```

Keys you'll actually touch:

- **`kv_cache`** — `asym3` (default, 5.5× compression vs fp32) / `q8` (baseline quality) / `asym2` (6× compression for 8 GB cards)
- **`temperature`** — 0 for deterministic, 0.7 for varied
- **`max_tokens`** — generation cap per response
- **`max_seq`** — KV cache capacity (context window allocated at load). Bigger = more VRAM.
- **`default_model`** — what `hipfire serve` pre-warms on start

Per-model overrides when you want different settings for different models:

```bash
hipfire config qwen3.5:9b set max_seq 65536
hipfire config qwen3.5:27b set max_seq 8192   # smaller for the big model
```

Or drill in via the TUI: open `hipfire config`, scroll to the
**`[per-model configs]`** row at the bottom, press Enter, pick a model.

## 9. Quantize your own models (optional)

Any Qwen 3.5 HuggingFace model (or a local safetensors dir) can be quantized:

```bash
# One-shot: download + quantize + upload + register
hipfire quantize Jackrong/Qwopus3.5-4B-v3 \
    --both \
    --upload schuttdev/hipfire-qwopus-4b \
    --create-repo --install \
    --register qwopus:4b

# Or local:
hipfire quantize ./my-finetune --format mq4 -o finetune.mq4
hipfire run ./finetune.mq4 "Hello"
```

The quantizer is CPU-only. Figure ~1 min/GB of weights.

## 10. What now?

- [BENCHMARKS.md](BENCHMARKS.md) — see how your card should perform
- [MODELS.md](MODELS.md) — full model catalog
- [KV_CACHE.md](KV_CACHE.md) — understand what asym3 actually does
- [QUANTIZATION.md](QUANTIZATION.md) — MQ4/MQ6 encoding + when to use which
- [ARCHITECTURE.md](ARCHITECTURE.md) — internals if you want to hack

## Common hiccups

**"no hipcc in PATH"** — either install `rocm-hip-sdk` or add `/opt/rocm/bin`
to your PATH. The pre-compiled kernels ship with hipfire, but JIT fallback
needs hipcc.

**"hipcc compilation failed: hip/hip_runtime.h not found"** — some distros
(CachyOS, niche ROCm builds) don't inject the default include path. 0.1.5+
handles this automatically; if you're on an older build, set
`HIPFIRE_HIPCC_EXTRA_FLAGS="-I/opt/rocm/include"`.

**"Serve started but /health did not respond within 5min"** — cold JIT on slow
hardware (gfx1013 APU, older gfx1030) for a 9B model. Just wait — tail
`~/.hipfire/serve.log` to watch layer-loading progress.

**Port 11435 already in use** — another serve crashed without cleanup. Nuke:

```bash
pkill -9 daemon bun
rm -f ~/.hipfire/serve.pid
hipfire serve -d
```

**Model says "Kendall" when asked what your name is** — you're on pre-0.1.5.
The K-kernel head_dim=256 fix + asym3 KV recall correctly. `hipfire update`.

**Still stuck?** Run `hipfire diag` + paste output at
https://github.com/Kaden-Schutt/hipfire/issues.
