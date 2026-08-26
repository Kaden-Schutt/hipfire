# CLI reference

Audience: operators driving the native Rust `hipfire` binary
(`crates/hipfire-cli`).
Run `hipfire <cmd> --help` (or `hipfire help <cmd>`) for the live flag text. This page indexes subcommands and file locations; it does **not** duplicate the full model registry or every config key.

Bare interactive `hipfire` launches the terminal UI when `hipfire-tui` is installed; otherwise it prints the command list. Config keys: [CONFIG.md](CONFIG.md). Env vars: [env-vars.md](env-vars.md). Models: [MODELS.md](MODELS.md).

## Model lifecycle

| Command | Purpose |
|---|---|
| `hipfire pull <tag>` | Download a registry model (and published sidecars when listed) into `~/.hipfire/models/`. |
| `hipfire list [-r\|--remote] [-j\|--json]` | Local models; `-r` also lists pullable registry tags; user aliases from `quantize --register` appear separately. |
| `hipfire rm <tag\|path> [-y\|--yes]` | Delete the weight file and sibling sidecars (`.triattn*.bin`, `*.mtp`). Confirms unless `-y`. |
| `hipfire ps [-j\|--json]` | Running daemon / quantize / upload processes and whether the configured serve port is busy (process scan is Linux-oriented). |

Tags resolve through the dynamic registry + aliases. Authoritative live list:
`hipfire list -r`. The native binary embeds `registry/v1.json`; the curated
editing surface is `registry/models.json`. At startup it may refresh the v1
payload into `~/.hipfire/registry.cache.json` (24h). Offline or invalid remote
payloads fall back to cache then the embedded registry. Pin the bundle with
`HIPFIRE_NO_REGISTRY_FETCH=1`.

## Inference

| Command | Purpose |
|---|---|
| `hipfire run <model> [flags] [prompt...]` | One-shot generate. Model = registry tag, alias, or path. Uses a healthy `serve` over HTTP when present; otherwise spawns a one-shot daemon. Forces local spawn when `HIPFIRE_LOCAL=1` or a load-time override such as `--kv-mode`/`--image` cannot safely reuse the resident model. Recognized missing registry tags auto-pull. |
| `hipfire chat <model> [--no-color]` | Interactive multi-turn TUI. See [CHAT.md](CHAT.md). |
| `hipfire serve [model] [host] [port] [flags]` | OpenAI-compatible HTTP server. See [SERVE.md](SERVE.md). |
| `hipfire restart [serve flags...]` | `stop --force` semantics then start with the same flags. |
| `hipfire stop [port] [--force] [--all]` | Stop the tracked background serve (`~/.hipfire/serve.pid`). `--force` reaps orphan daemons and frees the port; `--all` also reaps orphan quantize jobs. |
| `hipfire tui [flags...]` | Launch `hipfire-tui` (Home / Chat / Models / Settings / System). |

### `hipfire run` flags
Flags may appear before or after the model. CLI help and the native typed schema list stored globals; **effective `run` sampling** is not “global default unless overridden.” The Rust request resolver transmits only explicit CLI flags, per-model overlay values, or registry `recommended_settings`. Otherwise the field is **omitted** so daemon/HFQ/arch fallback applies. Global-only `temperature` / `top_p` / `repeat_penalty` are not sent. **Chat** is the exception (global session snapshot — [CHAT.md](CHAT.md)).

| Flag | Purpose |
|---|---|
| `--spec <m>` / `--speculation <m>` | Mechanism selector: `off` / `auto` / `ngram` / `dflash` / `mtp` / `dspark`. |
| `-md, --model-draft <path>` | DFlash draft model path. Implies `--spec dflash` unless `--spec` is also given. |
| `--draft-max, --draft <N>` | Draft window for an explicitly selected mechanism. Under `--spec auto`, only DeepSeek maps it to strict MTP K (`1..8`); Qwen/DFlash/n-gram auto routes keep their own configured windows. |
| `--dspark-conf-threshold <f>` | DSpark confidence-truncation cutoff `0..1` (qwen3 + deepseek4). Unset = per-arch default (qwen3 `0.1`, deepseek4 `0.3`). |
| `-t, --temp <float>` | Temperature when set on the CLI (`0` = greedy). Stored global default is `0.3` but is not auto-sent. |
| `--top-p <float>` | Nucleus sampling when set. Stored global default `0.8` is not auto-sent. |
| `--repeat-penalty <float>` | Repeat penalty when set. Stored global default `1.05` is not auto-sent. |
| `-n, --max-tokens <int>` | Generation cap (config default `4096`). |
| `--kv-mode <m>` | This-load KV mode: `auto`, `q8`, `fwht4`/`3`/`2`, `asym4`/`3`/`2`, `turbo`… |
| `--system <text>` | System prompt. |
| `--image <path>` | Vision input (when the model supports it). |
| `-j, --json` | Machine-readable output. |
| `--no-stream` | Buffer full response. |

Resolution ladder for speculation: **env > CLI flag > per-model > global**.
`dflash_mode` itself defaults to **`off`** — see [CONFIG.md](CONFIG.md#speculative-decode). Pulling a draft does not enable DFlash.

```bash
hipfire run qwen3.5:9b "What's 2+2?"
hipfire run qwen3.5:9b --spec ngram "Repeat verbatim: ..."
hipfire run qwen3.5:27b -md ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq "..."
HIPFIRE_LOCAL=1 hipfire run qwen3.5:4b "..."   # skip HTTP; always local spawn
```

Native MTP is currently served only for DeepSeek. Qwen `--spec mtp` / `mtp_mode=on`
rejects pending SPEC-003; Qwen `auto` and `off` use autoregressive decoding.

Local-forcing (skip a healthy serve): `HIPFIRE_LOCAL=1`, `--kv-mode`, or `--image`. JSON and non-streaming responses are supported by the native HTTP service and do not by themselves force a local daemon.

### `hipfire serve` flags

| Flag / arg | Purpose |
|---|---|
| `[host]` `[port]` or `host:port` | Bind (config defaults `0.0.0.0` and `11435`). **No authentication and no TLS** — prefer `127.0.0.1` for local-only; expose beyond localhost only on a trusted/firewalled network or behind an **authenticated TLS-terminating reverse proxy**. |
| `-d`, `--detach`, `--background` | Background; log `~/.hipfire/serve.log`, pid `~/.hipfire/serve.pid`. |
| `--kv-mode <m>` | KV mode for this process. |
| `--idle-timeout <s>` | Unload after idle seconds (`0` = never; max `86400`). |
| `--no-prewarm` | Lazy-load on first request. |
| `--tp N` | Forwards `params.tp` in the serve load message; admission resolves the axis — dense TP for admitted dense HFQ cells (LLaMA QK-norm / plain Qwen3) or EP for supported MoE cells (DeepSeek V4 Flash / MiniMax-M2). Range `1..64`. No `HIPFIRE_TP` env read (retired). |

## Configuration

| Command | Purpose |
|---|---|
| `hipfire config` | Global interactive TUI → `~/.hipfire/config.toml`. |
| `hipfire config <tag>` | Print resolved per-model policy and override provenance. |
| `hipfire config list\|get\|set\|reset ...` | Scriptable global ops (`--json` on list/get). |
| `hipfire config <tag> list\|get\|set\|reset ...` | Same, scoped to per-model keys. |

Do not inventory every key here — [CONFIG.md](CONFIG.md) owns defaults and ranges. Notable defaults from source: `dflash_mode=off`, `speculation=auto`, `thinking=on`, `reasoning_effort=auto`, schema default `thinking_budget=med` (legacy named-cap route only; dropped+warned on effort-native models), `max_tokens=4096`, `idle_timeout=300`. Qwen Jinja contracts additionally accept an explicit integer think cap; other families do not synthesize a force-close mechanism. Reasoning axes and family examples: [CONFIG.md](CONFIG.md), [SERVE.md](SERVE.md).

## Quantization and calibration

| Command | Purpose |
|---|---|
| `hipfire quantize <hf-id\|dir\|file.gguf> [flags]` | CPU quantize via `hipfire-quantize`. |
| `hipfire sidecar-gen <model> [flags]` | Build a `.triattn.bin` next to the model (does not pull). |

### `quantize` (summary)

| Flag | Purpose |
|---|---|
| `--format <fmt>` | Repeatable. Safetensors default **`mq4`**; GGUF default **`hf4`**. |
| `--both` | `mq4` + `mq6`. |
| `-o` / `--output`, `--output-dir`, `--stem` | Output naming. |
| `--install` | Copy into `~/.hipfire/models/`. |
| `--register <tag>` | Local alias in `~/.hipfire/models.toml`. |
| `--upload <owner/repo>`, `--create-repo` | Optional HF publish. |

Supported CLI formats include `mq4`, `mq6`, `q8`/`q8f16`, `hf4`/`hf6` and hfq aliases. Graded MoE recipes need the quantizer binary directly — [QUANTIZE.md](QUANTIZE.md).

### `sidecar-gen` (summary)

| Flag | Purpose |
|---|---|
| `--corpus PATH` | Calibration text (else builtin seeds). |
| `--max-tokens N` | Default `4000`. |
| `--chunk-len N` | Default `256`. |
| `--gpu-calib` / `--cpu-calib` | Device (GPU is the usual default path). |
| `-o PATH` | Output path (default `<model-path>.triattn.bin`). |
| `--skip-validation` | Skip post-gen validation. |

## Benchmarks and diagnostics

| Command | Purpose |
|---|---|
| `hipfire bench <model> [opts] [prompt]` | Prefill/decode timing. `--runs N` (default 5), `--json`, `--exp` (RDNA2 variant sweep). |
| `hipfire bench <model> --matrix ...` | Synthetic PP/context/TG matrix (`--pp`, `--ctx`, `--tg`, `--sustained-tg`, `--sustained-ctx`, `--warmups`, `--kv-mode`, `--redline`). |
| `hipfire profile [model] [--kernel substr] [--json]` | Live daemon roofline and compiled-kernel VGPR/SGPR/LDS/occupancy report. Use `hipfire-atlas` for measured ISA-fit and workload analysis. |
| `hipfire diag` | Static device/runtime checks plus a live HIP arch, version, and VRAM probe when the daemon is available. |
| `hipfire --version` | Concise semver + build commit + source ref identity. |
| `hipfire version [--json]` | Detailed build/source match, checkout state, target, config schema, and installed-daemon SHA-256. |
| `hipfire update [@REF]` | **Linux managed installs:** advance the current branch or auto-detect and install a branch/tag/commit. |
| `hipfire update --branch NAME` | Switch to and track a remote branch such as `master` or `beta`. |
| `hipfire update --tag TAG` / `--commit SHA` | Install an immutable detached revision. A later explicit selector moves away from the pin. |

Perf claim protocol (warmup, fresh-process, noise): [methodology/perf-benchmarking.md](methodology/perf-benchmarking.md). Published tables are measured/historical: [BENCHMARKS.md](BENCHMARKS.md).

## Where files live

| Path | Role |
|---|---|
| `~/.hipfire/bin/` | native `hipfire`, `daemon`, optional `hipfire-tui`, tools |
| `~/.hipfire/bin/kernels/compiled/<arch>/` | Precompiled / JIT kernels |
| `~/.hipfire/models/` | Weight files and co-located sidecars |
| `~/.hipfire/config.toml` | Sparse typed global config (`config.json` is migration input) |
| `~/.hipfire/models.toml` | Local aliases, paths, registry identities, and per-model overrides |
| `~/.hipfire/models.json` | Legacy catalog migration input |
| `~/.hipfire/per_model_config.json` | Legacy overlay; folded into the catalog on refresh |
| `~/.hipfire/registry.cache.json` | Dynamic registry cache |
| `~/.hipfire/serve.pid` / `serve.log` | Detached serve pid record and log |
| `~/.hipfire/hf-cache/` | HF download cache used by quantize |
| `~/.hipfire/src/` | Source tree used by `update` / some installs |

Override catalog path with `HIPFIRE_MODELS_CATALOG_PATH` if needed.

## Dynamic registry

1. Bundled `registry/v1.json` (embedded in the binary).
2. Optional network fetch of `registry/v1.json` → 24h cache.
3. Invalid remote payloads are rejected wholesale.
`hipfire diag` reports which source the process used. Entry metadata may include size, `arch_id` ([architecture-ids.md](architecture-ids.md)), quant, and hashes — registry presence is not a runtime admission ([INDEX.md](INDEX.md), [admissions.yml](admissions.yml)).

## Useful environment overrides

Single-invocation knobs (non-exhaustive; full list in [env-vars.md](env-vars.md)):

| Variable | Effect |
|---|---|
| `HIPFIRE_LOCAL=1` | `run` skips HTTP serve and spawns a local daemon (also forced by load-time overrides such as `--kv-mode` or `--image`). |
| `HIPFIRE_HOME=...` | Override the state/config root (default `~/.hipfire`). |
| `HIPFIRE_MODELS_DIR=...` | Override model discovery, pull, list, and TUI model paths. |
| `HIPFIRE_KV_MODE=...` | Override KV layout. |
| `HIPFIRE_SPECULATION=...` | Top of speculation ladder. |
| `HIPFIRE_DFLASH_DRAFT=...` | Explicit draft path. |
| `HIPFIRE_DFLASH_MODE=...` | Daemon-side mode (CLI default config is still `off`). |
| `HIPFIRE_NO_REGISTRY_FETCH=1` | Pin bundled registry. |
| `HIPFIRE_REGISTRY_URL=...` | Alternate registry URL. |
| `HIPFIRE_NORMALIZE_PROMPT=0` | Opt out of `\n{3,}` collapse (config `prompt_normalize` default on). |
| `HIPFIRE_DEBUG=1` | Full stacks on CLI failure. |
| `NO_COLOR` / `CLICOLOR=0` | Disable ANSI in chat (also `--no-color`). |

## Errors and next steps

| Problem | Action |
|---|---|
| Unknown command | `hipfire help` |
| Unknown config key | `hipfire config set` prints valid keys; see [CONFIG.md](CONFIG.md) |
| Model missing | `hipfire list` / `pull` |
| Serve already running | `hipfire stop` or `restart` |
| Stale pid / port busy | `hipfire stop --force`; inspect `serve.log` |
| `update` on non-Linux | Re-run `install.ps1 -Branch/-Tag/-Commit` or use a source build |
| `update` from a detached pin | Choose the next target explicitly, e.g. `hipfire update @beta` |
| Need HTTP API detail | [SERVE.md](SERVE.md) |
| Need chat UX | [CHAT.md](CHAT.md) |
| Need onboarding | [GETTING_STARTED.md](GETTING_STARTED.md) |
