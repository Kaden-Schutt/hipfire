# Getting started

Audience: first install on an AMD GPU host. Goal: install → verify → pull a model → run or chat.

## Prerequisites

- **Linux:** AMD GPU with `/dev/kfd`, HIP runtime (`libamdhip64.so`). RDNA4 (`gfx1200`/`gfx1201`) needs HIP/ROCm **6.4+**. Strix Halo / gfx115x needs **7.2+**.
- **Windows:** [AMD HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) (`hipcc` + `amdhip64.dll`).
- **WSL2:** install AMD WSL GPU support first (`sudo amdgpu-install --usecase=wsl`), then use the Linux installer inside the distro.
- Disk space for models under `~/.hipfire/models/` (a few GB for small tags; tens of GB for 27B+).

Live model tags, VRAM floors, and formats: [MODELS.md](MODELS.md). Full env list: [env-vars.md](env-vars.md).

## Install

### Linux (recommended) — pin, download, verify, execute

Do **not** pipe a mutable branch URL into a shell. Pin a release tag or full
commit, fetch the installer from that pin, inspect it, then run it against a
checkout of the **same** pin (local mode). The installer script hard-codes
`GITHUB_BRANCH=master` for remote clone mode, so a remote-only curl of a pinned
script still installs the moving `master` tip unless you use a local checkout.

```bash
# Replace PIN with a release tag (e.g. v0.2.1) or full commit SHA.
PIN=v0.2.1
git clone --depth 1 --branch "$PIN" https://github.com/Kaden-Schutt/hipfire.git
cd hipfire
# Optional integrity check: record/compare sha256 of scripts/install.sh yourself.
sha256sum scripts/install.sh
# Inspect, then execute from the pinned tree (local mode wires CLI + PATH):
less scripts/install.sh
./scripts/install.sh
```

Download-only variant (still requires a same-pin checkout before execute if you
want the installed tree pinned — do not `curl … | bash`):

```bash
PIN=v0.2.1
curl -fsSL "https://raw.githubusercontent.com/Kaden-Schutt/hipfire/${PIN}/scripts/install.sh" \
  -o /tmp/hipfire-install.sh
sha256sum /tmp/hipfire-install.sh   # compare to a value you trust
# Review the script, then either:
#   (A) preferred — clone PIN and run ./scripts/install.sh from that tree, or
#   (B) bash /tmp/hipfire-install.sh  # remote mode clones master tip (not PIN)
```

> **Unverified moving tip:** `curl -L …/master/scripts/install.sh | bash` follows
> the live `master` branch for both the script and the checkout it installs. Use
> only for throwaway experiments; it is not the ref-pinned default.

The installer detects GPU arch, ensures HIP and Rust build prerequisites, builds
or copies the daemon, installs the native `hipfire` binary under
`~/.hipfire/bin/`, and places kernels at
`~/.hipfire/bin/kernels/compiled/<arch>/`. It can add that bin directory to
`PATH`; reload the shell afterward if `hipfire` is not found.

### Windows — pin, download, verify, execute

```powershell
# Replace $Pin with a release tag (e.g. v0.2.1) or full commit SHA.
$Pin = "v0.2.1"
git clone --depth 1 --branch $Pin https://github.com/Kaden-Schutt/hipfire.git
cd hipfire
Get-FileHash .\scripts\install.ps1 -Algorithm SHA256   # record/compare yourself
# Inspect, then run from the pinned tree (local mode):
notepad .\scripts\install.ps1
.\scripts\install.ps1
```

Download-only (do not `irm … | iex` for production). Remote mode in the script
uses `$GithubBranch = "master"` for the checkout it installs:

```powershell
$Pin = "v0.2.1"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Kaden-Schutt/hipfire/$Pin/scripts/install.ps1" `
  -OutFile "$env:TEMP\hipfire-install.ps1"
Get-FileHash "$env:TEMP\hipfire-install.ps1" -Algorithm SHA256
# Review, then prefer running install.ps1 from a git clone of $Pin.
```

> **Unverified moving tip:** `irm …/master/scripts/install.ps1 | iex` is a
> mutable master path for both script and installed tree — not the pinned default.

Uses a GitHub release `daemon.exe` when available; otherwise builds from source
under `~\.hipfire\src`. The native CLI is built from the same checkout, and the
installer runs `daemon.exe --precompile` into
`~\.hipfire\bin\kernels\compiled\<arch>\`. To force a full kernel compile after install:

```powershell
cd ~\.hipfire\src
.\scripts\compile-kernels.ps1 gfx1100   # or your arch
# script writes to the checkout's kernels\compiled\<arch>\ — copy into the install cache (or re-run install.ps1):
Copy-Item .\kernels\compiled\<arch>\* $env:USERPROFILE\.hipfire\bin\kernels\compiled\<arch>\ -Force
```

### Source checkout

```bash
git clone https://github.com/Kaden-Schutt/hipfire
cd hipfire
cargo build --release --features deltanet --example daemon -p hipfire-runtime
cargo build --release -p hipfire-cli
cargo build --release -p hipfire-quantize
# optional TUI:
cargo build --release -p hipfire-tui
./scripts/install.sh   # from a checkout: local mode wires CLI + PATH
```

Other packaging: [NIXOS.md](NIXOS.md), [CONTAINER.md](CONTAINER.md).

## Verify

```bash
hipfire diag
```

Reports GPU arch, VRAM, HIP/ROCm, kernel locations, model dir, and config overrides. Prefer this over guessing when a later command fails.

## First inference

```bash
hipfire pull qwen3.5:4b
hipfire run  qwen3.5:4b "Explain FFT in one line"
```

- `pull` downloads the registry artifact into `~/.hipfire/models/` (and published sidecars when the registry entry lists them).
- `run` accepts a registry tag, alias, or path. **Recognized registry tags that are not local yet are auto-pulled** (can be multi-GB). Unresolved tags, aliases, or paths error with a `pull` / `list --remote` hint — they do not download.
- Cold start loads weights and may JIT kernels; later calls are faster if a daemon is already up.

Interactive multi-turn:

```bash
hipfire chat qwen3.5:4b
```

See [CHAT.md](CHAT.md).

## Keep a daemon warm

```bash
hipfire serve qwen3.5:4b -d    # background; OpenAI-compatible HTTP
hipfire run qwen3.5:4b "..."   # reuses serve when healthy
hipfire stop                   # graceful stop of the tracked daemon
```

Defaults (overridable in config): bind **`0.0.0.0:11435`**, pre-warm **`default_model`** (`qwen3.5:9b` unless you set another). HTTP surface: [SERVE.md](SERVE.md). Subcommand flags: [CLI.md](CLI.md).

> **No auth / no TLS:** the serve HTTP API has **neither authentication nor TLS**. The default `0.0.0.0` listens on all interfaces and exposes inference to any reachable network (including chat-spawned serves). For local-only use, bind loopback: `hipfire config set host 127.0.0.1` or `hipfire serve 127.0.0.1 11435`. Expose beyond localhost only on a trusted/firewalled network **or** behind an **authenticated TLS-terminating reverse proxy** you control — never publish the raw port to the internet.

Force a one-shot daemon and skip HTTP:

```bash
HIPFIRE_LOCAL=1 hipfire run qwen3.5:4b "..."
```

## Light configuration

```bash
hipfire config                                      # global TUI → ~/.hipfire/config.toml
hipfire config qwen3.5:9b                           # resolved per-model policy
hipfire config qwen3.5:9b set generation.temperature 0.7
hipfire config qwen3.5:9b list                      # overlay + provenance
```

Defaults that matter on day one (from the native schema; full table in [CONFIG.md](CONFIG.md)). **Sampling send path:** `run` / `serve` transmit explicit request/CLI values, per-model TOML overlays, or the complete registry `recommended_settings` recipe (`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repeat_penalty`, plus fallback `system_prompt`). Otherwise sampling fields are omitted for daemon/HFQ/arch fallback. Bare global sampling values alone are **not** effective `run`/`serve` defaults. **Chat** is the exception — it uses a global config snapshot for the session ([CHAT.md](CHAT.md)).

| Key | Default | Note |
|---|---|---|
| `temperature` | `0.3` | Stored global default only for run/serve send (see above); Chat session seed |
| `max_tokens` | `4096` | Per-request generation cap for `run` / API fallback |
| `kv_cache` | `auto` | Resolves via registry `default_kv_mode`, else `q8` |
| `dflash_mode` | **`off`** | DFlash is opt-in; pulling a draft does not enable it |
| `speculation` | `auto` | Mechanism selector; DFlash stays off when `dflash_mode=off`, but eligible **MTP / DSpark** paths may still activate under `auto`. Use `speculation=off` to force plain AR. |
| `thinking` | `on` | Reasoning models may emit `<think>`; display strip is CLI/API-side |
| `host` / `port` | `0.0.0.0` / `11435` | Serve bind (no auth, no TLS) |

Enable draft-model speculation only when you intend to:

```bash
hipfire pull qwen3.5:9b
hipfire pull qwen3.5:9b-draft
hipfire config set dflash_mode auto    # or on / per-model
```

## Long context (optional)

CASK/TriAttention eviction is experimental and disabled by default. It is
**not** required for short prompts. When deliberately testing it for long
context on limited VRAM:

1. Prefer models whose `pull` ships a `.triattn.bin` sidecar, **or** generate one: `hipfire sidecar-gen <model>`.
2. Set `cask_sidecar` to that exact path. Enable `cask` separately only when m-folding is intended.
3. Read constraints (A3B, DFlash + m-fold) in [CONFIG.md](CONFIG.md) before enabling on MoE or with DFlash. `cask_auto_attach` remains `false` unless explicitly opted in.

## If something fails

| Symptom | What to try |
|---|---|
| `hipfire: command not found` | Reload shell; ensure install dir is on `PATH` |
| HIP / `/dev/kfd` / arch errors | `hipfire diag`; match ROCm/HIP version to arch (above) |
| Model not found | `hipfire list` / `hipfire list -r`; `hipfire pull <tag>` |
| Port in use / stale serve | `hipfire ps`; `hipfire stop --force`; check `~/.hipfire/serve.pid` and (detached only) `serve.log` |
| Draft pulled but no speedup | Expected: `dflash_mode` defaults to **off** |
| Truncated answers on thinking models | Raise `max_tokens` / `thinking_budget`; see [MODELS.md](MODELS.md) thinking section |

```bash
hipfire diag
tail -f ~/.hipfire/serve.log
```

## What to read next

| Doc | When |
|---|---|
| [CLI.md](CLI.md) | Every subcommand and flag surface |
| [CHAT.md](CHAT.md) | Interactive chat, thinking display, daemon attach |
| [SERVE.md](SERVE.md) | OpenAI-compatible HTTP |
| [MODELS.md](MODELS.md) | Tags, VRAM, BYO quantize, thinking/templates |
| [CONFIG.md](CONFIG.md) | All config keys and experimental CASK/TriAttention opt-ins |
| [QUANTIZE.md](QUANTIZE.md) | `hipfire quantize` operator guide |
| [INDEX.md](INDEX.md) | Ownership map for the rest of `docs/` |
