# Getting started

Audience: first install on an AMD GPU host. Goal: install → verify → pull a model → run or chat.

## Prerequisites

- **Linux:** AMD GPU with `/dev/kfd` plus a ROCm HIP development stack.
  hipfire JIT-compiles kernels, so a runtime-only install is insufficient: the
  selected ROCm root must provide `lib/libamdhip64.so` (and
  `libhsa-runtime64.so`), `include/hip/hip_runtime.h`, and `bin/hipcc`.
  Install a supported AMD ROCm HIP runtime, development headers, and device
  compiler via
  [AMD's live install selector](https://rocm.docs.amd.com/en/latest/install/rocm.html)
  (choose packages for your GPU, OS, and ROCm version — package names drift;
  the selector is authoritative).
- **Supported ROCm range:** Linux with **ROCm 6 or newer** (project baseline from
  [README.md](../README.md)). **ROCm 6.4+** for RDNA4 (`gfx1200`/`gfx1201`);
  **ROCm 7.2+** for Strix Halo / gfx115x. hipfire's path resolver does not
  hardcode a required release — install a supported stack for your GPU.
- **Windows:** [AMD HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) (`hipcc` + `amdhip64.dll`).
- **WSL2:** install AMD WSL GPU support first (`sudo amdgpu-install --usecase=wsl`), then use the Linux installer inside the distro.
- Disk space for models under `~/.hipfire/models/` (a few GB for small tags; tens of GB for 27B+).

Live model tags, VRAM floors, and formats: [MODELS.md](MODELS.md). Full env list: [env-vars.md](env-vars.md).

For a non-default or side-by-side install, pin one coherent SDK root before
starting hipfire:

```bash
export HIPFIRE_ROCM_PATH=/absolute/path/to/rocm
# If HIPFIRE_ROCM_PATH is unset, ROCM_PATH then HIP_PATH are accepted:
# export ROCM_PATH=/absolute/path/to/rocm
# export HIP_PATH=/absolute/path/to/rocm   # or .../hip (normalized to the parent)
```

Priority is `HIPFIRE_ROCM_PATH` > `ROCM_PATH` > `HIP_PATH`. An explicit override
is authoritative: once a root is selected, HIP/HSA libraries, headers, and
`hipcc` stay in that root family — hipfire will not fall back to another install
or a bare soname. If several complete roots are equally eligible (for example
multiple `/opt/rocm-*` with no active `/opt/rocm`), discovery refuses to guess;
set `HIPFIRE_ROCM_PATH` to one absolute root.

## Install

### Linux — master or beta in one command

The revision selector controls the managed source checkout under
`~/.hipfire/src`; no installer editing is needed:

```bash
# Current master:
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.sh | bash

# Integration/testing branch:
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.sh \
  | bash -s -- --branch beta
```

Branch installs remain on that branch when `hipfire update` is run without a
selector. The equivalent generic form is `--ref beta`; it auto-detects a
branch, tag, or commit. `HIPFIRE_INSTALL_REF=beta` is available for automation.

Both `master` and `beta` are mutable. For a reproducible install, pin and
inspect the installer itself, then ask it to install the same tag or commit:

```bash
PIN=v0.2.1
curl -fsSL "https://raw.githubusercontent.com/warpfront/hipfire/${PIN}/scripts/install.sh" \
  -o /tmp/hipfire-install.sh
sha256sum /tmp/hipfire-install.sh
less /tmp/hipfire-install.sh
bash /tmp/hipfire-install.sh --ref "$PIN"
```

Use `--tag v0.2.1` when the kind is known, or `--commit <full-sha>` for an
immutable commit. Fetching a pinned script but omitting the selector installs
`master`, so keep the two pins together.

The installer detects GPU arch, ensures HIP and Rust build prerequisites, builds
or copies the daemon, installs the native `hipfire` binary under
`~/.hipfire/bin/`, and places kernels at
`~/.hipfire/bin/kernels/compiled/<arch>/`. It can add that bin directory to
`PATH`; reload the shell afterward if `hipfire` is not found.

### Windows — select a branch, tag, or commit

```powershell
# Current master:
iex (irm https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.ps1)

# Integration/testing branch:
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.ps1))) `
  -Branch beta
```

For a reviewed, pinned installation:

```powershell
$Pin = "v0.2.1"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/warpfront/hipfire/$Pin/scripts/install.ps1" `
  -OutFile "$env:TEMP\hipfire-install.ps1"
Get-FileHash "$env:TEMP\hipfire-install.ps1" -Algorithm SHA256
notepad "$env:TEMP\hipfire-install.ps1"
& "$env:TEMP\hipfire-install.ps1" -Ref $Pin
```

PowerShell also accepts `-Branch beta`, `-Tag v0.2.1`, and `-Commit <sha>`.
The native `hipfire update` command remains Linux-only because Windows cannot
atomically replace the running executable; re-run `install.ps1` with the
desired selector instead.

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
git clone https://github.com/warpfront/hipfire
cd hipfire
cargo build --release --features deltanet --example daemon -p hipfire-runtime
cargo build --release -p hipfire-cli
cargo build --release -p hipfire-quantize
# optional TUI:
cargo build --release -p hipfire-tui
./scripts/install.sh   # from a checkout: local mode wires CLI + PATH
```

Other packaging: [NIXOS.md](NIXOS.md), [CONTAINER.md](CONTAINER.md).

## Uninstall a managed Linux install

The default uninstall removes the installed binaries, kernels, clean managed
source checkout, runtime PID/log files, and the PATH entry created by the
installer. It preserves downloaded models and settings under `~/.hipfire`:

```bash
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/uninstall.sh | bash
```

Preview without changing anything:

```bash
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/uninstall.sh \
  | bash -s -- --dry-run
```

Use `--purge` only when models, configuration, and every other file under
`~/.hipfire` should also be deleted. The script asks for an explicit
confirmation; automation can add `--yes`. It does not remove ROCm, Rust, or
other shared system dependencies.

## Verify

```bash
hipfire --version
hipfire version
hipfire diag
```

`--version` prints the release, source commit, and source ref in one line.
`hipfire version` additionally reports whether the managed checkout still
matches the binary and hashes the installed daemon; add `--json` for support
reports. `diag` reports GPU arch, VRAM, HIP/ROCm, kernel locations, model dir,
and config overrides.

### Update or switch channels on Linux

```bash
hipfire update                  # advance the currently selected branch
hipfire update @beta            # auto-detect branch/tag/commit
hipfire update --branch master
hipfire update --tag v0.2.1
hipfire update --commit <sha>
```

Tags and commits are detached, immutable pins; a later selector is required to
move away from them. Before switching, the updater retains the previous commit
under `refs/hipfire/backups/` and stashes dirty files with a recoverable
`hipfire-update-*` label.

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

### Measured capacity (Qwen3.5/3.6 35B-A3B-class, 24GB)

On 24GB GPUs, Q8 KV is about 10,880 B/token across 10 full-attention layers,
plus O(N) flash partials (~2,064 B/token), ~25 MiB DeltaNet state, and multi-GiB
fixed HIP/graph overhead. **50K Q8 is tight but physically feasible**; **200K Q8
is not a 24GB-class configuration** — it needs >32GB-class VRAM or CASK/compressed
KV. Some historical 131K sidecar benches clamped physical capacity to ~2432
tokens and should not be read as full-context Q8 support. Long-context decode
slowdown is expected O(N) full-attention bandwidth, not by itself an admission
regression.

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
