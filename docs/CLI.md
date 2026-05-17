# CLI reference

Every subcommand of the `hipfire` wrapper. Run `hipfire <cmd> --help` for
flag-level detail; this page is the index.

## Model lifecycle

| Command | Purpose |
|---|---|
| `hipfire pull <tag>` | Download a model from HuggingFace into `~/.hipfire/models/`. |
| `hipfire list [-r]` | Show local models. `-r` adds remotely-available tags from the curated registry. |
| `hipfire ps` | Show running daemons, in-flight quantize jobs, and HuggingFace upload tasks. |
| `hipfire rm <tag>` | Delete a local model file. |

## Inference

| Command | Purpose |
|---|---|
| `hipfire run <tag\|path> [prompt...]` | Generate. Auto-pulls if missing. Routes through the running `serve` daemon if one is up; otherwise spawns a one-shot daemon. |
| `hipfire chat <tag>` | Interactive TUI chat with streaming, markdown, multi-line input. Reuses running serve or spawns a dedicated daemon. |
| `hipfire serve [host] [port] [-d]` | Start the OpenAI-compatible HTTP server. Accepts `host port` or `host:port` such as `hipfire serve 0.0.0.0:11435`. `-d` detaches into the background and writes a pid file. Defaults: host `0.0.0.0`, port `11435` (`hipfire config set host ...`, `hipfire config set port ...`). |
| `hipfire stop` | Graceful shutdown of the background daemon. |
| `hipfire bench <tag>` | Measure prefill + decode tok/s on a fixed prompt set. |

`hipfire run` accepts either a registry tag (`qwen3.5:9b`) or a literal
file path (`./my.mq4`). For a prompt with shell-special characters,
quote it: `hipfire run qwen3.5:9b "What's 2+2?"`.

## Configuration

| Command | Purpose |
|---|---|
| `hipfire config` | Interactive TUI for global config (`~/.hipfire/config.json`). |
| `hipfire config <tag>` | Per-model overlay (`~/.hipfire/per_model_config.json`). Rows show `(inherited)` vs `(overridden)`. |
| `hipfire config set <key> <val>` | Non-interactive set. |
| `hipfire config view` | Print effective config + all overlays. |

Full key list and tradeoffs in [CONFIG.md](CONFIG.md).

## Quantization

| Command | Purpose |
|---|---|
| `hipfire quantize <hf-id\|local-dir\|file.gguf>` | CPU-side quantize from safetensors or GGUF to MQ4 / MQ6 / HF4 / HF6. Optional `--install` puts the result in `~/.hipfire/models/` and `--register <tag>` adds an alias. |

The full quantize how-to (formats, when to pick which, GGUF caveats) is
in [QUANTIZE.md](QUANTIZE.md).

## Calibration

For models with custom or quantized weights that need CASK KV eviction,
generate the calibration sidecar:

| Command | Purpose |
|---|---|
| `hipfire sidecar-gen <model>` | Generate a `.triattn.bin` sidecar for the given model. The daemon auto-discovers it alongside the model file when CASK is enabled (`cask_sidecar on`). |

Usage:

```bash
hipfire sidecar-gen qwen35-27b-dflash-mq4 --corpus my-corpus.txt --max-tokens 8192 --chunk-len 1024 -o /path/to/output.triattn.bin
```

Flags for `sidecar-gen`:

| Flag | Purpose |
|---|---|
| `<model>` (positional) | Model tag or local file path. The sidecar is written next to the model file by default with the same base name: `my-finetune.mq4` → `my-finetune.triattn.bin`. See **Filename discovery** below for details. |
| `--corpus PATH` | Text corpus for calibration. If omitted, uses an internal default. |
| `--max-tokens N` | Maximum tokens of context to calibrate over (default: 8192). |
| `--chunk-len N` | Chunk size for KV cache statistics collection (default: 1024). |
| `--gpu-calib` | Run calibration on GPU instead of CPU. |
| `--cpu-calib` | Override to run on CPU even if a compatible GPU is available. |
| `-o PATH` | Output path for the `.triattn.bin` file (default: next to model). |
| `--skip-validation` | Skip post-generation KV statistics validation check. |

The generated sidecar contains per-position KV cache statistics that
calculate which key-value positions are most important for retention.
Without it, CASK eviction treats all positions equally and can discard
critical early tokens on long context prompts, causing quality drop-off.

**Quick setup after quantizing your own model:**

```bash
hipfire quantize ./my-model/ --format mq4 -o my-finetune.mq4
hipfire sidecar-gen my-finetune.mq4 --corpus /path/to/corpus.txt
# The daemon will auto-attach the sidecar when you run with cask_sidecar on
```

See [CONFIG.md](CONFIG.md) for CASK-related configuration keys.

> **Note:** `sidecar-gen` requires a model file to exist — it does not
> pull models from HuggingFace. First pull your target: `hipfire pull <tag>`
> or put your quantized weights alongside the expected filename pattern
> (`qwen3{ver}-{size}-dflash-{quant}.hfq`). The daemon auto-discovers
> `.triattn.bin` files next to their matching model.
>
> **Filename discovery:** When `cask_sidecar` is unset, the daemon looks for
> a sidecar in the same directory as the model file using `<basename>.triattn*.bin`
> (e.g., `my-finetune.mq4` → `my-finetune.triattn.bin`). If you specify a
> path with directories (`foo/bar/model.mq4`), it scans `foo/bar/` for the
> sidecar — not the current working directory.

## Diagnostics

| Command | Purpose |
|---|---|
| `hipfire diag` | GPU arch, VRAM, HIP version, ROCm version, kernel blob hashes, model directory. First place to check if anything misbehaves. |
| `hipfire update` | `git pull` + rebuild + refresh kernel blobs. Use when upstream pushes a fix. |

## Where files live

- Models: `~/.hipfire/models/`
- Config: `~/.hipfire/config.json`
- Per-model overlay: `~/.hipfire/per_model_config.json`
- Local model aliases: `~/.hipfire/models.json`
- Pre-compiled kernels: `~/.hipfire/bin/kernels/<arch>/`
- Daemon log: `~/.hipfire/serve.log`
- Daemon pid file: `~/.hipfire/serve.pid`

## Environment overrides

Single-invocation overrides bypass the config file:

| Variable | Effect |
|---|---|
| `HIPFIRE_KV_MODE=asym3\|q8\|asym4\|asym2` | Override KV cache layout. |
| `HIPFIRE_ATTN_FLASH=auto\|always\|never` | Force or disable FlashAttention. |
| `HIPFIRE_NORMALIZE_PROMPT=0` | Opt out of `\n{3,}` → `\n\n` prompt collapse (default ON). |
| `HIPFIRE_LOCAL=1` | `hipfire run` skips the HTTP daemon and spawns a fresh one-shot. |
| `HIPFIRE_HIPCC_EXTRA_FLAGS=...` | Append flags to JIT kernel compilations. |
| `HIPFIRE_PROMPT_TOKEN_HEAT=1` | Dump per-position BPE merge-rank heat to stderr. |
