# NixOS Packaging Design

**Date:** 2026-05-07
**Status:** Approved
**Scope:** Nix flake (dev shell + package + NixOS module) + documentation

## Overview

Provide first-class NixOS support for hipfire via a single `flake.nix` at the
repo root. Users get three entry points:

1. `nix develop` — full dev shell (Rust, bun, ROCm, hipcc)
2. `nix build` — produces the hipfire package (daemon + CLI + kernels)
3. `nixosModules.default` — declarative `services.hipfire` system configuration

## File Layout

```
flake.nix              # inputs, outputs, wiring
flake.lock
nix/
  package.nix          # hipfire binary derivation
  kernels.nix          # GPU kernel compilation derivation
  dev-shell.nix        # development shell
  module.nix           # NixOS module (services.hipfire)
docs/
  NIXOS.md             # user-facing installation & configuration guide
README.md              # add short NixOS section pointing to docs/NIXOS.md
```

## Flake Inputs

```nix
inputs = {
  nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  rust-overlay.url = "github:oxalica/rust-overlay";
  flake-utils.url = "github:numtide/flake-utils";
};
```

## Flake Outputs

- `packages.${system}.default` — hipfire package
- `packages.${system}.hipfire-kernels` — pre-compiled GPU kernels
- `devShells.${system}.default` — development shell
- `nixosModules.default` — NixOS module
- `overlays.default` — for overlay-based consumption

## Package Derivation (`nix/package.nix`)

Uses `rustPlatform.buildRustPackage`.

**Build inputs:**
- Rust toolchain (from nixpkgs, not overlay — overlay is dev-shell only)
- `makeWrapper` for LD_LIBRARY_PATH injection

**Parameters:**
- `rocmSupport ? true` — wire ROCm libs into wrapper's LD_LIBRARY_PATH
- `gpuTargets ? [ "gfx1100" ]` — passed to kernel derivation

**Install outputs:**
- `$out/bin/hipfire-daemon` — wrapped daemon binary
- `$out/bin/hipfire` — bun wrapper invoking CLI
- `$out/share/hipfire/cli/` — TypeScript CLI files

**Runtime dependency:** `libamdhip64.so` resolved via `makeWrapper --prefix
LD_LIBRARY_PATH` when `rocmSupport = true`. hipfire uses bare-soname dlopen
(`libamdhip64.so`, `.so.7`, `.so.6`, `.so.5`) — no env var override exists on
Linux, so the wrapper approach is required on NixOS.

**Cargo build flags:**
```
cargo build --release --features deltanet \
  --example daemon --example infer --example infer_hfq \
  -p hipfire-runtime
```

## Kernel Derivation (`nix/kernels.nix`)

Separate derivation — kernels need `hipcc` at build time but the Rust binary
does not.

**Build inputs:**
- `rocmPackages.clr` (provides hipcc)
- `rocmPackages.llvm.clang`

**Parameters:**
- `gpuTargets ? [ "gfx1100" ]` — architectures to compile for

**Build phase:** Invokes existing `scripts/compile-kernels.sh` with the target
list. The script handles variant-tag resolution, wave64/wave32 selection, and
parallel compilation.

**Output:** `$out/kernels/compiled/{arch}/*.hsaco` + `.hash` files.

## Dev Shell (`nix/dev-shell.nix`)

Provides the full development environment. Does NOT depend on the hipfire source
tree — works with any checkout/fork.

**Contents:**
- Rust stable (via rust-overlay) with rust-src + rust-analyzer
- Bun (CLI runtime)
- pkg-config
- When `rocmSupport = true`: rocmPackages.clr, rocm-smi, rocminfo
- `LD_LIBRARY_PATH` set for dlopen during `cargo run`

**Usage with different repos:**
```bash
nix develop github:Kaden-Schutt/hipfire
cd ~/my-hipfire-fork
cargo build --release
```

## NixOS Module (`nix/module.nix`)

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable` | bool | false | Enable hipfire service |
| `package` | package | pkgs.hipfire | Package to use |
| `kernelsPackage` | package | pkgs.hipfire-kernels | Pre-compiled kernels |
| `gpuTargets` | listOf str | ["gfx1100"] | GPU architectures |
| `rocmSupport` | bool | true | Use nixpkgs ROCm libs |
| `port` | port | 11435 | API listen port |
| `modelDir` | path | /var/lib/hipfire/models | Model directory |
| `settings` | attrs | {temperature=0.3, ...} | Global config.json |
| `perModelSettings` | attrsOf attrs | {} | Per-model overrides |
| `environment` | attrsOf str | {} | HIPFIRE_* env vars |
| `userService` | bool | false | User-level systemd unit |
| `user` | str | "hipfire" | Daemon user (system mode) |
| `group` | str | "hipfire" | Daemon group (system mode) |

### System Mode (default)

- Creates dedicated `hipfire` user in `video` + `render` groups
- `systemd.services.hipfire-precompile` — oneshot, runs `daemon --precompile`
  on activation (before main service)
- `systemd.services.hipfire` — long-running daemon
- Hardened: `ProtectSystem=strict`, `NoNewPrivileges=true`,
  `DeviceAllow=/dev/kfd rw, /dev/dri/* rw`
- `ReadWritePaths` includes `modelDir` and `/var/lib/hipfire`

### User Service Mode (`userService = true`)

- Creates `systemd.user.services.hipfire` (managed via `systemctl --user`)
- No dedicated user/group created
- User must be in `video` + `render` groups themselves
- Models default to `~/.hipfire/models`

### Config File Generation

- `settings` attrset -> `config.json` (via `pkgs.writeText` + `builtins.toJSON`)
- `perModelSettings` attrset -> `per_model_config.json`
- `environment` attrset -> systemd `Environment=` lines
- Precedence: engine defaults < config.json < per_model_config.json < env vars

### Daemon CLI Flags

**TODO (verify during implementation):** Confirm actual daemon flags for:
- `--port` (or is it positional / config-only?)
- `--models` / `--model-dir`
- `--config` (path to config.json)
- `--kernels` (path to compiled kernels)

If the daemon reads config.json from `~/.hipfire/` by default, the service may
need to set `HOME=/var/lib/hipfire` or use an env var to redirect.

## ROCm Dependency Handling

Three modes:

1. **Bundled (default):** `rocmSupport = true` — uses `rocmPackages.clr` from
   nixpkgs. `LD_LIBRARY_PATH` injected via wrapper/systemd env.

2. **Bring your own:** `rocmSupport = false` — no ROCm from nixpkgs. User sets
   `LD_LIBRARY_PATH` manually or via `services.hipfire.environment`.

3. **Custom ROCm overlay:** User overrides `rocmPackages` in their nixpkgs
   overlay to use a different ROCm version. The flake's package/module
   references `rocmPackages.clr` generically, so overlays just work.

## GPU Target Configuration

Supported architectures (from install.sh):
| Arch | Card | Generation |
|------|------|-----------|
| gfx906 | Vega 20 | GCN5 |
| gfx908 | MI100 | CDNA |
| gfx1010 | RX 5700 XT | RDNA1 |
| gfx1030 | RX 6800 XT | RDNA2 |
| gfx1100 | RX 7900 XTX | RDNA3 |
| gfx1151 | Strix Halo | RDNA3.5 |
| gfx1200 | Radeon R9700 | RDNA4 |
| gfx1201 | RX 9070 XT | RDNA4 |

`gpuTargets` accepts any subset. Kernels are compiled only for listed arches.
Detection command: `grep gfx_target_version /sys/class/kfd/kfd/topology/nodes/*/properties`

## Documentation (`docs/NIXOS.md`)

Covers:
- Prerequisites (NixOS version, AMD GPU, firmware, kernel module)
- Quick start (nix develop, nix build, flake input)
- NixOS module usage with examples (minimal, full, desktop mode)
- GPU target configuration with detection instructions
- Configuration (settings, perModelSettings, environment, precedence)
- ROCm dependency modes (bundled, BYOD, overlay)
- Dev shell usage (including different repo/fork workflow)
- Building from source inside dev shell
- Troubleshooting (libamdhip64 not found, /dev/kfd permissions, firmware, hipcc mismatch)
- Environment variables (document that interactive HIPFIRE_* vars go in shell profile)

## README.md Addition

Short section:
```markdown
## NixOS

hipfire has first-class NixOS support via a Nix flake. See [docs/NIXOS.md](docs/NIXOS.md).

Quick start:
```bash
nix develop github:Kaden-Schutt/hipfire  # dev shell
nix build github:Kaden-Schutt/hipfire    # build package
```

For NixOS module usage:
```nix
{
  inputs.hipfire.url = "github:Kaden-Schutt/hipfire";

  # In your configuration.nix:
  services.hipfire.enable = true;
  services.hipfire.gpuTargets = [ "gfx1100" ];
}
```
```

## Open Questions (resolve during implementation)

1. **Daemon CLI flags** — verify exact flag names for port, model dir, config
   path, and kernel path by reading the daemon example source.
2. **Home directory assumption** — the CLI reads `~/.hipfire/config.json`. In
   system service mode, need to either set `HOME=/var/lib/hipfire` or symlink.
3. **Bun in Nix** — verify `pkgs.bun` is available in nixos-unstable and works
   for the CLI. If not, bundle via fetchurl or use the bun flake.
4. **Cargo examples as binaries** — `rustPlatform.buildRustPackage` may need a
   custom `buildPhase`/`installPhase` since the main binaries are `[[example]]`
   targets, not `[[bin]]`.
