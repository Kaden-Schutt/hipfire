# NixOS

First-class Nix flake support lives at the repo root (`flake.nix`) with
packages and the NixOS module under `nix/`.

| Output | Path / attribute | Role |
|---|---|---|
| Package | `packages.hipfire` (default) | Native `hipfire` CLI + wrapped `hipfire-daemon` |
| Kernels | `packages.hipfire-kernels` | Optional precompile via `scripts/compile-kernels.sh` |
| Dev shell | `devShells.default` | Rust and hipcc/ROCm tools when `rocmSupport` |
| Module | `nixosModules.default` | `services.hipfire` systemd unit(s) |
| Overlay | `overlays.default` | Pins `rocmPackages` from this flake’s nixpkgs + hipfire packages |

| Field | Value |
|---|---|
| Flake nixpkgs input | `github:NixOS/nixpkgs/nixos-unstable` |
| Module default port | `11435` |
| System model dir option default | `/var/lib/hipfire/models` (see [modelDir limitation](#modeldir-and-effective-paths)) |
| Kernel package default targets | `[]` (JIT only unless overridden) |

Truth state: **shipped / ref-pinned** for the flake/module surface described
here. Pin a release tag or commit for production hosts; building
`github:warpfront/hipfire` without a rev follows that input’s default
branch and is **not** a stable release pin. Branch-only inference features
are not implied by enabling the module.

## Prerequisites

- NixOS with flakes (examples below use `nixos-unstable` or a 25.11-class
  channel for **your** system flake; hipfire’s own flake input is unstable
  for ROCm 7.x — see [ROCm](#rocm-configuration)).
- AMD GPU with `amdgpu` loaded; `/dev/kfd` and `/dev/dri/renderD*` present.
- Interactive users in `video` and `render` when not using the system service
  user.

```bash
ls /dev/kfd
ls /dev/dri/
# architecture hints:
rocminfo 2>/dev/null | grep -oP 'amdgcn-amd-amdhsa--\K\S+' | sort -u
# or:
grep gfx_target_version /sys/class/kfd/kfd/topology/nodes/*/properties
```

## Quick start (flake packages)

### Dev shell

```bash
nix develop github:warpfront/hipfire
```

Tools only — not a checkout. From a local tree:

```bash
nix develop
cargo build --release --features deltanet --example daemon -p hipfire-runtime
```

### Build CLI + daemon

```bash
nix build github:warpfront/hipfire
./result/bin/hipfire run qwen3.5:9b "Hello"
```

Stable pin example (replace rev/hash with a real release or commit):

```nix
hipfire.url = "github:warpfront/hipfire/<tag-or-commit>";
```

### Precompiled kernels package

```bash
nix build github:warpfront/hipfire#hipfire-kernels
```

Default `gpuTargets = []` produces an empty/near-empty kernels tree; the
daemon JIT-compiles on first use. Override targets in the NixOS module or
your own package override — an empty default avoids baking the wrong arch.

## NixOS module

`nixosModules.default` → `services.hipfire`.

### Minimal system example

```nix
# flake.nix (host)
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    hipfire.url = "github:warpfront/hipfire"; # pin rev for production
  };

  outputs = { nixpkgs, hipfire, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        hipfire.nixosModules.default
        {
          nixpkgs.overlays = [ hipfire.overlays.default ];
          services.hipfire.enable = true;
          services.hipfire.gpuTargets = [ "gfx1100" ]; # required non-empty
        }
      ];
    };
  };
}
```

Assertions when `enable = true`:

- `gpuTargets` must be non-empty (Nix cannot probe the GPU at eval time).
- With `rocmSupport = true` (default), `hardware.graphics.enable` must be
  true (the module defaults it on and registers the ROCm ICD).

### Full options example

```nix
services.hipfire = {
  enable = true;
  gpuTargets = [ "gfx1100" ]; # required non-empty
  openFirewall = false;  # does NOT change listen address — only skips opening the port when the host firewall is enabled
  # openFirewall = true opens cfg.port with NO auth/TLS on the listener — see SERVE.md

  defaultModel = "qwen3.5:9b";
  # Module writes these into config.toml as stored globals. run/serve sampling
  # send still omits bare globals (CLI flags / per-model / registry card only);
  # see CONFIG.md. Chat uses the global snapshot exception.
  temperature = 0.3;
  topP = 0.8;
  maxTokens = 512;       # module default; CLI global default is 4096 if unset in config
  maxSeq = 32768;
  repeatPenalty = 1.05;
  kvCache = "auto";
  dflashMode = "off";
  idleTimeout = 300;

  # Writes native ~/.hipfire/models.toml per-model overrides.
  perModelSettings = {
    "qwen3.5:27b" = {
      max_seq = 16384;
      kv_cache = "q8";
      dflash_mode = "on";
    };
  };

  extraSettings = {
    # Canonical dotted keys are preferred; legacy snake_case aliases are
    # translated before writing config.toml.
    # Actual loopback-only listen (openFirewall=false alone is not enough):
    host = "127.0.0.1";
  };

  environment = {
    HIPFIRE_NORMALIZE_PROMPT = "1";
  };

  # Exported as HIPFIRE_MODELS_DIR and honored by the native CLI/TUI.
  modelDir = "/var/lib/hipfire/models";
};
```

Typed options are written to `config.toml` using the canonical dotted schema
(`serve.default_model`, `generation.top_p`, `serve.idle_timeout_seconds`, …). See [CONFIG.md](CONFIG.md)
for the full key set the CLI understands; module-typed keys are the subset
in `nix/module.nix`.

### modelDir and effective paths

`services.hipfire.modelDir` is exported as `HIPFIRE_MODELS_DIR`. The native
CLI and TUI use it for tag/list/pull/pre-warm discovery:

| Mode | `HOME` | Effective CLI models path |
|---|---|---|
| System service (`userService = false`) | `/var/lib/hipfire` | configured `services.hipfire.modelDir` |
| User service | the login user’s home | configured `services.hipfire.modelDir` |

`hipfire-setup` creates `modelDir`, and the system service includes it in
`ReadWritePaths`. Absolute model paths remain valid for ad-hoc runs.

### Desktop / user service

```nix
services.hipfire = {
  enable = true;
  userService = true;
  gpuTargets = [ "gfx1201" ];
  # Must be user-writable. Default /var/lib/hipfire/models is not; setup's
  # mkdir -p fails for an unprivileged user unit.
  modelDir = "/home/yourname/.hipfire/models";  # match effective CLI path
  defaultModel = "";                             # module starts --no-prewarm
};

users.users.yourname.extraGroups = [ "video" "render" ];
```

```bash
# hipfire.service only has After=hipfire-setup — it does NOT Wants/Requires setup.
# Start setup first (or ensure default.target already ran it after login):
systemctl --user start hipfire-setup.service
systemctl --user start hipfire.service
systemctl --user status hipfire.service
```

User mode skips the dedicated `hipfire` system user and the oneshot
precompile unit; setup still copies config and symlinks daemon/kernels under
`$HOME/.hipfire`. Models for tag discovery must live under
`$HOME/.hipfire/models` (see [modelDir limitation](#modeldir-and-effective-paths)).

### Pinning source from the module

Precedence: `services.hipfire.src` > `services.hipfire.github.rev` > flake
package default.

```nix
# branch / tag / commit via fetchFromGitHub
services.hipfire = {
  enable = true;
  gpuTargets = [ "gfx1100" ];
  github.rev = "v0.2.1";          # example pin — use a real tag/commit
  github.hash = "";               # first build prints the SRI hash
};

# fork
services.hipfire.github.owner = "my-username";
services.hipfire.github.repo = "hipfire";
services.hipfire.github.rev = "my-branch";
services.hipfire.github.hash = "sha256-…";

# Path source only (module type is nullOr path — not an arbitrary derivation).
# Use a checkout path or a path-coerced fetch result your Nix accepts as path:
services.hipfire.src = ./path/to/hipfire-checkout;
# Prefer github.* when you need fetchFromGitHub (typed owner/repo/rev/hash):
# services.hipfire.github.rev = "abc123…";
# services.hipfire.github.hash = "sha256-…";
```

Unreleased / working-branch checkouts are **branch-implemented** relative to
whatever you pin — do not describe them as a stable release without an
explicit tag or commit hash.

## What the system service does

When `userService = false` (default):

| Unit | Role |
|---|---|
| `hipfire-setup.service` | Oneshot: native `config.toml` + `models.toml`, `bin/daemon` symlink, kernels symlink under `/var/lib/hipfire/.hipfire` |
| `hipfire-precompile.service` | Oneshot: `hipfire-daemon --precompile` (failure is warned; JIT still works) |
| `hipfire.service` | `ExecStart = hipfire serve`, restart on failure |

Environment always includes `HIPFIRE_MODELS_DIR=<modelDir>`; the native CLI
and TUI honor it for discovery and model lifecycle. With `rocmSupport`,
`LD_LIBRARY_PATH` is set for nixpkgs ROCm. System mode also sets
`HOME=/var/lib/hipfire` and `HIPFIRE_KERNEL_CACHE=/var/cache/hipfire/kernels`.

**Security:** `hipfire serve` has **no authentication and no TLS**
([SERVE.md](SERVE.md)). Default listen remains **`0.0.0.0`** unless you set
`extraSettings.host = "127.0.0.1"` (or equivalent). `openFirewall = false` does
**not** change the listen address — it only avoids opening `cfg.port` when the
**host firewall is enabled**. Keep the firewall closed unless the host is on a
trusted network or an authenticated TLS reverse proxy terminates in front of
the port. Prefer probing on loopback (`127.0.0.1`).

Device policy allows `/dev/kfd`, DRM char devices, and `/dev/accel/accel0`.

HTTP contract after start: [SERVE.md](SERVE.md).

## GPU targets

Set only arches you need. Common labels (not an admission list):

| Arch | Typical hardware |
|---|---|
| gfx906 | Vega 20 / MI50 |
| gfx908 | MI100 |
| gfx1010 | RX 5700 XT |
| gfx1030 | RX 6800 XT class |
| gfx1100 | RX 7900 XTX class |
| gfx1151 | Strix Halo |
| gfx1200 | Radeon AI PRO R9700 class |
| gfx1201 | RX 9070 / 9070 XT class |

```nix
services.hipfire.gpuTargets = [ "gfx1100" "gfx1030" ];
```

## Module options reference

From `nix/module.nix` (defaults are module defaults, not every CLI default):

| Option | Type | Default | Description |
|---|---|---|---|
| `enable` | bool | `false` | Enable the service |
| `openFirewall` | bool | `false` | Open `port` when the host firewall is enabled; does **not** change listen address (default still `0.0.0.0`) |
| `package` | package | `pkgs.hipfire` | Package to run |
| `src` | path or null | `null` | Override source **path** (not a package/derivation) |
| `github.owner` | str |  `"warpfront"`    | fetch owner |
| `github.repo` | str | `"hipfire"` | fetch repo |
| `github.rev` | str or null | `null` | branch/tag/commit |
| `github.hash` | str | `""` | SRI hash for fetch |
| `kernelsPackage` | package | `pkgs.hipfire-kernels` | Precompiled kernels |
| `gpuTargets` | list of str | `[]` (**required non-empty when enabled**) | Kernel compile arches |
| `rocmSupport` | bool | `true` | Use nixpkgs ROCm + graphics ICD |
| `port` | port | `11435` | Serve port |
| `defaultModel` | str | `""` | Empty omits `serve.default_model` and makes the module start `hipfire serve --no-prewarm`; a non-empty value is pre-warmed |
| `temperature` | float | `0.3` | Stored in config.toml only; not an effective run/serve send default (see CONFIG.md sampling send) |
| `topP` | float | `0.8` | Stored global; same send caveat as temperature |
| `maxTokens` | int | `512` | Written to config.toml |
| `maxSeq` | int | `32768` | KV capacity |
| `repeatPenalty` | float | `1.05` | Stored global; same send caveat as temperature |
| `kvCache` | str | `"auto"` | KV mode string |
| `dflashMode` | enum | `"off"` | `on` / `off` / `auto` |
| `idleTimeout` | int | `300` | Idle unload seconds (`0` = never) |
| `extraSettings` | attrs | `{}` | Extra config.toml keys (e.g. `"serve.host" = "127.0.0.1"`; legacy `host` is translated) |
| `perModelSettings` | attrs of attrs | `{}` | Writes native `models.toml` per-model overrides |
| `environment` | attrs of str | `{}` | Extra env for the unit |
| `modelDir` | str | `"/var/lib/hipfire/models"` | Exported as `HIPFIRE_MODELS_DIR`; native CLI/TUI model discovery uses it |
| `userService` | bool | `false` | systemd `--user` mode |
| `user` / `group` | str | `"hipfire"` | System service identity |

## ROCm configuration

### Default: flake overlay + nixpkgs ROCm

`overlays.default` re-imports this flake’s `nixpkgs` (unstable) with
`rocmSupport` so `rocmPackages` tracks ROCm **7.x**. Comment in-tree:
nixos-25.11’s ROCm 6.4.3 `libamdhip64` can segfault on **gfx1151** during
weight upload — prefer the overlay on Strix Halo hosts.

`LD_LIBRARY_PATH` is injected on the daemon/CLI wrappers and the systemd
unit when `rocmSupport = true`.

### Bring your own ROCm

```nix
services.hipfire = {
  rocmSupport = false;
  environment = {
    LD_LIBRARY_PATH = "/opt/rocm/lib";
  };
};
```

You must still satisfy GPU device access and any ICD requirements yourself.

## Configuration precedence

Same layered model as the CLI:

1. Engine / daemon defaults
2. `config.toml` (module typed options + `extraSettings`)
3. Per-model overlay from `models.toml` (`perModelSettings`)
4. Environment (`services.hipfire.environment` and process env)

Interactive shells outside systemd only see variables you export yourself
([env-vars.md](env-vars.md)).

## Production smoke after deploy

Enabling the module starts `hipfire serve`; it does **not** by itself certify
model quality or GPU numerical correctness.

On a machine with models in the **effective** CLI path
(`/var/lib/hipfire/.hipfire/models` for the system service, or
`~/.hipfire/models` for user mode — not necessarily `modelDir`):

```bash
curl -s http://127.0.0.1:11435/health
# GPU serve semantics (manual); pass an absolute model file:
python3 scripts/serve_harness.py --model /var/lib/hipfire/.hipfire/models/<file> --tag qwen3.5:9b
```

Route selection: [VALIDATION.md](VALIDATION.md). No-GPU CI does not replace
this.

## Troubleshooting

### `libamdhip64.so` not found

With `rocmSupport = true`, confirm the overlay is applied and
`rocmPackages.clr` builds. With BYO ROCm, check `LD_LIBRARY_PATH` and the
`.so` path.

### Permission denied on `/dev/kfd`

Add the runtime user to `video` and `render`, rebuild, re-login (user
sessions). The system service user already gets those groups.

### No AMD GPU detected

```bash
lsmod | grep amdgpu
```

```nix
hardware.firmware = [ pkgs.linux-firmware ];
# and/or:
hardware.amdgpu.initrd.enable = true;
```

Ensure `hardware.graphics.enable = true` when using bundled ROCm.

### Kernel pre-compilation fails

`hipfire-precompile` warns and continues; first request JIT-compiles. Check
`hipcc --version` against the target arch. gfx1151 expects ROCm 7.2+ class
tooling (aligned with the container base and flake overlay intent).

## Related

- Serve HTTP: [SERVE.md](SERVE.md)
- Config keys: [CONFIG.md](CONFIG.md)
- Containers: [CONTAINER.md](CONTAINER.md)
- Package implementation: `nix/package.nix`, `nix/kernels.nix`, `nix/dev-shell.nix`, `nix/module.nix`
