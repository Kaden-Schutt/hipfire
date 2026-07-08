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

### `hipfire run` speculative-decode flags

One-shot overrides (override config for that invocation only). They follow
llama.cpp's `speculative` flag naming and resolve through the ladder
**env > flag > per-model > global**. See [CONFIG.md](CONFIG.md#speculative-decode)
for the persistent config keys.

| Flag | Purpose |
|---|---|
| `--spec <m>` | Mechanism selector: `off` / `auto` / `ngram` / `dflash` / `mtp` / `dspark`. |
| `-md, --model-draft <path>` | DFlash draft model path. Implies `--spec dflash` unless `--spec` is also given. |
| `--draft-max, --draft <N>` | Draft window of the active mechanism (n-gram K / MTP k). |
| `--dspark-conf-threshold <f>` | DSpark confidence-truncation cutoff `0..1` (qwen3 + deepseek4). Unset = per-arch default (qwen3 `0.1`, deepseek4 `0.3`). |

`dspark` uses the model's `<stem>-dspark.<ext>` draft-module sidecar (a small dense
drafter) if present. Its draft block size is picked per request by a default-on
τ-adaptive controller (disable with `HIPFIRE_DSPARK_ADAPTIVE_BLOCK=0`).

```bash
hipfire run qwen3.5:9b --spec ngram "Repeat verbatim: ..."   # model-free, byte-identical to AR
hipfire run qwen3.5:27b -md qwen3.5-27b-dflash-mq4.hfq "..."  # draft-model DFlash
hipfire run qwen3-8b --spec dspark "Write an LRU cache"       # DSpark draft-module sidecar
```

## Configuration

| Command | Purpose |
|---|---|
| `hipfire config` | Interactive TUI for global config (`~/.hipfire/config.json`). |
| `hipfire config <tag>` | Per-model overlay (`~/.hipfire/per_model_config.json`). Rows show `(inherited)` vs `(overridden)`. |
| `hipfire config set <key> <val>` | Non-interactive set. |
| `hipfire config list` | Print effective config. Prefix with a tag (`hipfire config <tag> list`) for the per-model overlay. |

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
| `hipfire sidecar-gen <model>` | Generate a `.triattn.bin` sidecar for the given model. The daemon auto-discovers it alongside the model file when a CASK profile is enabled. |

Usage:

```bash
hipfire sidecar-gen qwen35-27b-dflash-mq4 --corpus my-corpus.txt --max-tokens 8192 --chunk-len 1024 -o /path/to/output.triattn.bin
```

Flags for `sidecar-gen`:

| Flag | Purpose |
|---|---|
| `<model>` (positional) | Model tag or local file path. The sidecar is written next to the model file by default using the full model filename: `my-finetune.mq4` → `my-finetune.mq4.triattn.bin`. See **Filename discovery** below for details. |
| `--corpus PATH` | Text corpus for calibration. If omitted, uses an internal default. |
| `--max-tokens N` | Maximum tokens of context to calibrate over (default: 4000). |
| `--chunk-len N` | Chunk size for KV cache statistics collection (default: 256). |
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
hipfire config cask-profile balanced
# The daemon will auto-attach the sidecar on the next model load
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
> (e.g., `my-finetune.mq4` → `my-finetune.mq4.triattn.bin`). If you specify a
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
- Dynamic registry cache: `~/.hipfire/registry.cache.json`
- Pre-compiled kernels: `~/.hipfire/bin/kernels/<arch>/`
- Daemon log: `~/.hipfire/serve.log`
- Daemon pid file: `~/.hipfire/serve.pid`

## Dynamic model registry

Compiled `hipfire` binaries inline `cli/registry.json` at build time, so on
its own a shipped binary would never learn about new models. At startup the
CLI fetches `registry/v1.json` from this repo's `master` branch (regenerated
daily from the HF Hub by `.github/workflows/registry.yml` →
`scripts/registry_gen.py`) and caches it for 24h at
`~/.hipfire/registry.cache.json`. Fallback chain: fresh cache → network →
stale cache → bundled registry — offline use always works, and a registry
that fails validation is rejected wholesale in favor of the next fallback.
`hipfire diag` prints which source the current run used (`registry: ...`).
v1 entries additionally carry `sha256` (HF LFS oid), `size_bytes`, `arch_id`
(see `docs/architecture-ids.md`), and `quant` next to the legacy fields.

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
| `HIPFIRE_NO_REGISTRY_FETCH=1` | Pin the bundled model registry (skip dynamic fetch + cache). |
| `HIPFIRE_REGISTRY_URL=...` | Fetch the dynamic registry from a different URL. |
