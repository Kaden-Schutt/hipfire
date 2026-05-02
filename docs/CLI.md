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
| `hipfire serve [port] [-d]` | Start the OpenAI-compatible HTTP server. `-d` detaches into the background and writes a pid file. Default port `11435`. |
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
