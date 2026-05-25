# NixOS Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide first-class NixOS support via a Nix flake with dev shell, package derivation, kernel compilation, NixOS module, and documentation.

**Architecture:** Single `flake.nix` at repo root wiring together four Nix files under `nix/`. The daemon has no CLI flags — all config flows through the Bun CLI (`hipfire serve`) which spawns the daemon and communicates via JSON IPC over stdin/stdout. Kernels live relative to the daemon binary at `<binary_dir>/kernels/compiled/<arch>/`. The NixOS module runs `hipfire serve` as a systemd service with `HOME` set to manage config/PID/model paths.

**Tech Stack:** Nix flakes, `rustPlatform.buildRustPackage`, `rust-overlay`, `rocmPackages`, `makeWrapper`, systemd

**Spec:** `docs/superpowers/specs/2026-05-07-nixos-packaging-design.md`

---

### File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `flake.nix` | Inputs, outputs, wiring |
| Create | `nix/package.nix` | Rust binary derivation |
| Create | `nix/kernels.nix` | GPU kernel compilation |
| Create | `nix/dev-shell.nix` | Development environment |
| Create | `nix/module.nix` | NixOS module (`services.hipfire`) |
| Create | `docs/NIXOS.md` | User-facing installation guide |
| Modify | `README.md:43-52` | Add NixOS section after Install |

---

### Task 1: Create `flake.nix`

**Files:**
- Create: `flake.nix`

- [ ] **Step 1: Write `flake.nix`**

```nix
{
  description = "hipfire — LLM inference for AMD RDNA GPUs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        hipfire = pkgs.callPackage ./nix/package.nix {
          rocmSupport = true;
        };

        hipfire-kernels = pkgs.callPackage ./nix/kernels.nix {
          gpuTargets = [ "gfx1100" ];
        };
      in
      {
        packages = {
          default = hipfire;
          inherit hipfire hipfire-kernels;
        };

        devShells.default = pkgs.callPackage ./nix/dev-shell.nix {
          rust-bin = pkgs.rust-bin;
          rocmSupport = true;
        };
      }
    ) // {
      nixosModules.default = import ./nix/module.nix;

      overlays.default = final: prev: {
        hipfire = final.callPackage ./nix/package.nix {
          rocmSupport = true;
        };
        hipfire-kernels = final.callPackage ./nix/kernels.nix {
          gpuTargets = [ "gfx1100" ];
        };
      };
    };
}
```

- [ ] **Step 2: Verify flake parses**

Run: `nix flake check --no-build 2>&1 | head -20`

Expected: No syntax errors. May warn about missing `nix/*.nix` files — that's fine, we create them next.

- [ ] **Step 3: Commit**

```bash
git add flake.nix
git commit -m "feat(nix): add flake.nix skeleton with inputs and outputs"
```

---

### Task 2: Create `nix/package.nix`

**Files:**
- Create: `nix/package.nix`

- [ ] **Step 1: Write `nix/package.nix`**

```nix
{ lib
, rustPlatform
, rocmPackages
, bun
, makeWrapper
, rocmSupport ? true
}:

rustPlatform.buildRustPackage {
  pname = "hipfire";
  version = "0.1.20";

  src = lib.cleanSource ./..;
  cargoLock.lockFile = ../Cargo.lock;

  # The main binaries are cargo [[example]] targets, not [[bin]].
  # We need a custom build/install phase.
  buildPhase = ''
    runHook preBuild
    cargo build --release --features deltanet \
      --example daemon --example infer --example infer_hfq \
      -p hipfire-runtime
    runHook postBuild
  '';

  # Skip default install (no [[bin]] targets exist)
  dontCargoInstall = true;

  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall

    # Install daemon and inference binaries
    mkdir -p $out/bin
    cp target/release/examples/daemon $out/bin/hipfire-daemon-unwrapped
    cp target/release/examples/infer $out/bin/hipfire-infer 2>/dev/null || true
    cp target/release/examples/infer_hfq $out/bin/hipfire-infer-hfq 2>/dev/null || true

    # Wrap daemon with LD_LIBRARY_PATH for libamdhip64.so dlopen
    makeWrapper $out/bin/hipfire-daemon-unwrapped $out/bin/hipfire-daemon \
      ${lib.optionalString rocmSupport
        "--prefix LD_LIBRARY_PATH : ${rocmPackages.clr}/lib"}

    # Install CLI (TypeScript, invoked via bun)
    mkdir -p $out/share/hipfire/cli
    cp -r cli/. $out/share/hipfire/cli/
    # Remove dev artifacts from CLI
    rm -rf $out/share/hipfire/cli/node_modules \
           $out/share/hipfire/cli/.gitignore \
           $out/share/hipfire/cli/tsconfig.json \
           $out/share/hipfire/cli/bun.lock
    find $out/share/hipfire/cli/ -maxdepth 1 -type f \
         \( -name '*.test.ts' -o -name 'test_*.ts' -o -name 'bench_*.ts' \) \
         -delete 2>/dev/null || true

    # Create hipfire CLI wrapper
    # The CLI spawns the daemon binary — it looks for it at:
    #   1. ~/.hipfire/bin/daemon  2. alongside the CLI  3. in PATH
    # We set HIPFIRE_DAEMON_BIN so the CLI finds our wrapped daemon.
    makeWrapper ${bun}/bin/bun $out/bin/hipfire \
      --add-flags "run $out/share/hipfire/cli/index.ts" \
      --set HIPFIRE_DAEMON_BIN $out/bin/hipfire-daemon \
      ${lib.optionalString rocmSupport
        "--prefix LD_LIBRARY_PATH : ${rocmPackages.clr}/lib"}

    runHook postInstall
  '';

  meta = with lib; {
    description = "LLM inference for AMD RDNA GPUs";
    homepage = "https://github.com/Kaden-Schutt/hipfire";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "hipfire";
  };
}
```

**Note:** The CLI discovers the daemon binary in several ways. We set
`HIPFIRE_DAEMON_BIN` as the env var — **this must be verified during
implementation**. If the CLI doesn't support this env var, we need to either:
(a) add it to `cli/index.ts` (small patch), or (b) symlink the daemon to
`$out/share/hipfire/cli/../bin/daemon` matching the expected relative layout.
Check `cli/index.ts` for how `bin` is resolved (look for `daemon` spawn path).

- [ ] **Step 2: Test that the derivation evaluates**

Run: `nix eval .#packages.x86_64-linux.default.name 2>&1`

Expected: `"hipfire-0.1.20"` (or eval error pointing to a fixable issue).

- [ ] **Step 3: Commit**

```bash
git add nix/package.nix
git commit -m "feat(nix): add package derivation for hipfire binary + CLI"
```

---

### Task 3: Create `nix/kernels.nix`

**Files:**
- Create: `nix/kernels.nix`

- [ ] **Step 1: Write `nix/kernels.nix`**

```nix
{ lib
, stdenv
, rocmPackages
, gpuTargets ? [ "gfx1100" ]
}:

stdenv.mkDerivation {
  pname = "hipfire-kernels";
  version = "0.1.20";

  src = lib.cleanSource ./..;

  nativeBuildInputs = [
    rocmPackages.clr           # provides hipcc
    rocmPackages.llvm.clang    # HIP compilation toolchain
  ];

  buildPhase = ''
    runHook preBuild
    export HOME=$TMPDIR
    bash scripts/compile-kernels.sh ${lib.concatStringsSep " " gpuTargets}
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out/kernels/compiled
    for arch in ${lib.concatStringsSep " " gpuTargets}; do
      if [ -d "kernels/compiled/$arch" ]; then
        cp -r "kernels/compiled/$arch" "$out/kernels/compiled/"
      fi
    done
    runHook postInstall
  '';

  meta = with lib; {
    description = "Pre-compiled GPU kernels for hipfire";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
```

- [ ] **Step 2: Test that the derivation evaluates**

Run: `nix eval .#packages.x86_64-linux.hipfire-kernels.name 2>&1`

Expected: `"hipfire-kernels-0.1.20"`

- [ ] **Step 3: Commit**

```bash
git add nix/kernels.nix
git commit -m "feat(nix): add kernel compilation derivation"
```

---

### Task 4: Create `nix/dev-shell.nix`

**Files:**
- Create: `nix/dev-shell.nix`

- [ ] **Step 1: Write `nix/dev-shell.nix`**

```nix
{ lib
, mkShell
, rust-bin
, rocmPackages
, bun
, pkg-config
, rocmSupport ? true
}:

mkShell {
  name = "hipfire-dev";

  nativeBuildInputs = [
    (rust-bin.stable.latest.default.override {
      extensions = [ "rust-src" "rust-analyzer" ];
    })
    bun
    pkg-config
  ] ++ lib.optionals rocmSupport [
    rocmPackages.clr
    rocmPackages.rocm-smi
    rocmPackages.rocminfo
  ];

  LD_LIBRARY_PATH = lib.optionalString rocmSupport
    "${rocmPackages.clr}/lib";

  shellHook = ''
    echo "hipfire dev shell"
    echo "  rust: $(rustc --version)"
    echo "  bun:  $(bun --version)"
    ${lib.optionalString rocmSupport ''
      echo "  hip:  $(hipcc --version 2>&1 | head -1)"
    ''}
  '';
}
```

- [ ] **Step 2: Test that the dev shell evaluates**

Run: `nix eval .#devShells.x86_64-linux.default.name 2>&1`

Expected: `"hipfire-dev"`

- [ ] **Step 3: Commit**

```bash
git add nix/dev-shell.nix
git commit -m "feat(nix): add development shell with Rust, bun, ROCm"
```

---

### Task 5: Create `nix/module.nix`

**Files:**
- Create: `nix/module.nix`

- [ ] **Step 1: Write `nix/module.nix`**

```nix
{ config, lib, pkgs, ... }:

let
  cfg = config.services.hipfire;
  configJson = pkgs.writeText "hipfire-config.json"
    (builtins.toJSON cfg.settings);
  perModelConfigJson = pkgs.writeText "hipfire-per-model-config.json"
    (builtins.toJSON cfg.perModelSettings);

  # Build the hipfire package with the user's rocmSupport choice
  hipfirePkg = cfg.package.override {
    rocmSupport = cfg.rocmSupport;
  };

  hipfireKernelsPkg = cfg.kernelsPackage.override {
    gpuTargets = cfg.gpuTargets;
  };

  # Home directory for the service
  homeDir = if cfg.userService then "$HOME" else "/var/lib/hipfire";

  # LD_LIBRARY_PATH for ROCm
  rocmLdPath = lib.optionalString cfg.rocmSupport
    "LD_LIBRARY_PATH=${pkgs.rocmPackages.clr}/lib";

  # Merged environment for systemd
  envList =
    (lib.mapAttrsToList (k: v: "${k}=${v}") cfg.environment)
    ++ lib.optionals cfg.rocmSupport [
      "LD_LIBRARY_PATH=${pkgs.rocmPackages.clr}/lib"
    ];
in
{
  options.services.hipfire = {

    enable = lib.mkEnableOption "hipfire inference daemon";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.hipfire;
      defaultText = lib.literalExpression "pkgs.hipfire";
      description = "The hipfire package to use.";
    };

    kernelsPackage = lib.mkOption {
      type = lib.types.package;
      default = pkgs.hipfire-kernels;
      defaultText = lib.literalExpression "pkgs.hipfire-kernels";
      description = "Pre-compiled GPU kernels package.";
    };

    gpuTargets = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ "gfx1100" ];
      example = [ "gfx1100" "gfx1030" ];
      description = ''
        GPU architectures to compile kernels for.
        Detect yours: grep gfx_target_version /sys/class/kfd/kfd/topology/nodes/*/properties
      '';
    };

    rocmSupport = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Use nixpkgs ROCm libraries (rocmPackages.clr).
        Set to false to provide your own libamdhip64.so via environment.
      '';
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 11435;
      description = "Port for the OpenAI-compatible API server.";
    };

    modelDir = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/hipfire/models";
      description = "Directory containing model files.";
    };

    settings = lib.mkOption {
      type = lib.types.attrs;
      default = {
        temperature = 0.3;
        top_p = 0.8;
        max_tokens = 512;
      };
      description = ''
        Global configuration written to config.json.
        See docs/CONFIG.md for all available keys.
      '';
    };

    perModelSettings = lib.mkOption {
      type = lib.types.attrsOf lib.types.attrs;
      default = { };
      example = lib.literalExpression ''
        {
          "qwen3.5:27b" = {
            max_seq = 16384;
            kv_cache = "q8";
          };
        }
      '';
      description = ''
        Per-model config overrides written to per_model_config.json.
        Keys are model tags, values are config attrsets.
      '';
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      example = lib.literalExpression ''
        { HIPFIRE_KV_MODE = "asym3"; }
      '';
      description = "Extra environment variables (HIPFIRE_*) for the daemon.";
    };

    userService = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Run as a user-level systemd service (systemctl --user)
        instead of a system service. No dedicated user is created.
      '';
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "hipfire";
      description = "User to run the daemon as (system mode only).";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "hipfire";
      description = "Group to run the daemon as (system mode only).";
    };
  };

  config = lib.mkIf cfg.enable (lib.mkMerge [

    # ── System service mode ──────────────────────────────────
    (lib.mkIf (!cfg.userService) {

      users.users.${cfg.user} = {
        isSystemUser = true;
        group = cfg.group;
        extraGroups = [ "video" "render" ];
        home = "/var/lib/hipfire";
        createHome = true;
      };
      users.groups.${cfg.group} = { };

      # Write config files into the service home on activation
      systemd.services.hipfire-setup = {
        description = "hipfire config setup";
        wantedBy = [ "multi-user.target" ];
        before = [ "hipfire-precompile.service" "hipfire.service" ];
        serviceConfig = {
          Type = "oneshot";
          RemainAfterExit = true;
          User = cfg.user;
          Group = cfg.group;
        };
        script = ''
          mkdir -p /var/lib/hipfire/.hipfire/bin
          mkdir -p ${cfg.modelDir}
          cp -f ${configJson} /var/lib/hipfire/.hipfire/config.json
          cp -f ${perModelConfigJson} /var/lib/hipfire/.hipfire/per_model_config.json

          # Symlink daemon binary so the CLI can find it at ~/.hipfire/bin/daemon
          ln -sf ${hipfirePkg}/bin/hipfire-daemon /var/lib/hipfire/.hipfire/bin/daemon

          # Symlink pre-compiled kernels next to the daemon
          ln -sfn ${hipfireKernelsPkg}/kernels /var/lib/hipfire/.hipfire/bin/kernels
        '';
      };

      # Pre-compile any missing kernels on activation
      systemd.services.hipfire-precompile = {
        description = "hipfire GPU kernel pre-compilation";
        wantedBy = [ "multi-user.target" ];
        after = [ "hipfire-setup.service" ];
        before = [ "hipfire.service" ];
        serviceConfig = {
          Type = "oneshot";
          RemainAfterExit = true;
          User = cfg.user;
          Group = cfg.group;
          Environment = envList;
        };
        script = ''
          export HOME=/var/lib/hipfire
          ${hipfirePkg}/bin/hipfire-daemon --precompile || true
        '';
      };

      # Main daemon service
      systemd.services.hipfire = {
        description = "hipfire inference daemon";
        after = [ "network.target" "hipfire-precompile.service" ];
        wantedBy = [ "multi-user.target" ];
        serviceConfig = {
          ExecStart = "${hipfirePkg}/bin/hipfire serve -d";
          Restart = "on-failure";
          RestartSec = 5;
          User = cfg.user;
          Group = cfg.group;
          Environment = envList ++ [
            "HOME=/var/lib/hipfire"
            "HIPFIRE_PORT=${toString cfg.port}"
          ];
          # Hardening
          ProtectSystem = "strict";
          ReadWritePaths = [ cfg.modelDir "/var/lib/hipfire" ];
          NoNewPrivileges = true;
          DeviceAllow = [ "/dev/kfd rw" "/dev/dri/renderD128 rw" ];
        };
      };
    })

    # ── User service mode ────────────────────────────────────
    (lib.mkIf cfg.userService {

      systemd.user.services.hipfire-setup = {
        description = "hipfire config setup (user)";
        wantedBy = [ "default.target" ];
        before = [ "hipfire.service" ];
        serviceConfig = {
          Type = "oneshot";
          RemainAfterExit = true;
        };
        script = ''
          mkdir -p $HOME/.hipfire/bin
          mkdir -p ${cfg.modelDir}
          cp -f ${configJson} $HOME/.hipfire/config.json
          cp -f ${perModelConfigJson} $HOME/.hipfire/per_model_config.json
          ln -sf ${hipfirePkg}/bin/hipfire-daemon $HOME/.hipfire/bin/daemon
          ln -sfn ${hipfireKernelsPkg}/kernels $HOME/.hipfire/bin/kernels
        '';
      };

      systemd.user.services.hipfire = {
        description = "hipfire inference daemon (user)";
        after = [ "hipfire-setup.service" ];
        wantedBy = [ "default.target" ];
        serviceConfig = {
          ExecStart = "${hipfirePkg}/bin/hipfire serve -d";
          Restart = "on-failure";
          RestartSec = 5;
          Environment = envList ++ [
            "HIPFIRE_PORT=${toString cfg.port}"
          ];
        };
      };
    })
  ]);
}
```

**Important implementation notes:**
- The daemon has NO CLI flags except `--precompile`. The CLI (`hipfire serve`)
  manages config, spawns the daemon, and handles HTTP.
- Config files are written to `~/.hipfire/` (the CLI reads from `HOME`).
- The daemon binary is symlinked to `~/.hipfire/bin/daemon` where the CLI
  expects to find it.
- Kernel directory is symlinked to `~/.hipfire/bin/kernels/` (the daemon looks
  for kernels relative to its binary at `<binary_dir>/kernels/compiled/<arch>/`).
- `HIPFIRE_PORT` env var needs verification — check if `cli/index.ts` reads it.
  If not, the port must be set in `config.json` under the `port` key (which the
  setup service already handles via `cfg.settings`).

- [ ] **Step 2: Test that the module evaluates**

Run: `nix eval .#nixosModules.default --apply 'x: builtins.typeOf x' 2>&1`

Expected: `"lambda"` (it's a function, as NixOS modules should be).

- [ ] **Step 3: Commit**

```bash
git add nix/module.nix
git commit -m "feat(nix): add NixOS module with system and user service modes"
```

---

### Task 6: Create `docs/NIXOS.md`

**Files:**
- Create: `docs/NIXOS.md`

- [ ] **Step 1: Write `docs/NIXOS.md`**

```markdown
# NixOS

hipfire has first-class NixOS support via a Nix flake.

## Prerequisites

- NixOS 24.05+ or nixos-unstable
- AMD GPU with the `amdgpu` kernel module loaded
- User in `video` and `render` groups (for non-service usage)

Verify your GPU is visible:

    ls /dev/kfd         # should exist
    ls /dev/dri/        # should show renderD128+

Detect your GPU architecture:

    grep gfx_target_version /sys/class/kfd/kfd/topology/nodes/*/properties

## Quick Start

### Development shell

Enter a shell with Rust, bun, hipcc, and ROCm tools:

    nix develop github:Kaden-Schutt/hipfire

This works with any hipfire checkout — the shell provides tools only,
not the source tree:

    nix develop github:Kaden-Schutt/hipfire
    cd ~/my-hipfire-fork
    cargo build --release --features deltanet --example daemon -p hipfire-runtime

### Build from source

    nix build github:Kaden-Schutt/hipfire
    ./result/bin/hipfire run qwen3.5:9b "Hello"

### Build kernels

    nix build github:Kaden-Schutt/hipfire#hipfire-kernels

## NixOS Module

Add hipfire to your flake inputs and enable the service.

### Minimal example

```nix
# flake.nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    hipfire.url = "github:Kaden-Schutt/hipfire";
  };

  outputs = { nixpkgs, hipfire, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        hipfire.nixosModules.default
        {
          nixpkgs.overlays = [ hipfire.overlays.default ];
          services.hipfire.enable = true;
          services.hipfire.gpuTargets = [ "gfx1100" ];
        }
      ];
    };
  };
}
```

### Full example

```nix
services.hipfire = {
  enable = true;
  gpuTargets = [ "gfx1100" ];

  # Global config (written to config.json)
  settings = {
    temperature = 0.3;
    top_p = 0.8;
    max_tokens = 512;
    kv_cache = "asym3";
    dflash_mode = "auto";
    port = 11435;
    idle_timeout = 300;
    default_model = "qwen3.5:9b";
  };

  # Per-model overrides (written to per_model_config.json)
  perModelSettings = {
    "qwen3.5:27b" = {
      max_seq = 16384;
      kv_cache = "q8";
      dflash_mode = "on";
    };
  };

  # Extra env vars (highest precedence)
  environment = {
    HIPFIRE_NORMALIZE_PROMPT = "1";
  };

  modelDir = "/var/lib/hipfire/models";
};
```

### Desktop / user-service mode

For single-user desktop setups, use a user-level systemd service:

```nix
services.hipfire = {
  enable = true;
  userService = true;
  gpuTargets = [ "gfx1100" ];
};
```

Manage with `systemctl --user start hipfire`, `systemctl --user status hipfire`.

Your user must be in `video` and `render` groups:

```nix
users.users.yourname.extraGroups = [ "video" "render" ];
```

## GPU Targets

| Arch | Card | Generation |
|------|------|-----------|
| gfx906 | Vega 20 / MI50 | GCN5 |
| gfx908 | MI100 | CDNA |
| gfx1010 | RX 5700 XT | RDNA1 |
| gfx1030 | RX 6800 XT | RDNA2 |
| gfx1100 | RX 7900 XTX | RDNA3 |
| gfx1151 | Strix Halo | RDNA3.5 |
| gfx1200 | Radeon R9700 | RDNA4 |
| gfx1201 | RX 9070 XT | RDNA4 |

Build kernels for multiple arches:

```nix
services.hipfire.gpuTargets = [ "gfx1100" "gfx1030" ];
```

## ROCm Configuration

### Default: bundled nixpkgs ROCm

hipfire uses `rocmPackages.clr` from nixpkgs by default.
`LD_LIBRARY_PATH` is injected automatically via wrapper scripts.

### Bring your own ROCm

```nix
services.hipfire = {
  rocmSupport = false;
  environment = {
    LD_LIBRARY_PATH = "/opt/rocm/lib";
  };
};
```

### Custom ROCm version via overlay

Override `rocmPackages` in your nixpkgs overlay. The hipfire package
and module reference `rocmPackages.clr` generically, so overlays
apply transparently.

## Configuration

hipfire uses a layered config system. Precedence (lowest to highest):

1. Engine defaults (hardcoded)
2. `config.json` — set via `services.hipfire.settings`
3. `per_model_config.json` — set via `services.hipfire.perModelSettings`
4. Environment variables — set via `services.hipfire.environment`

For the full list of config keys, see [CONFIG.md](CONFIG.md).

### Environment variables (interactive use)

For interactive usage outside the systemd service (`hipfire run`, `hipfire chat`),
set `HIPFIRE_*` variables in your shell profile:

```bash
# ~/.bashrc or equivalent
export HIPFIRE_KV_MODE=asym3
export HIPFIRE_NORMALIZE_PROMPT=1
```

These only affect the current user's shell sessions, not the systemd service.

## Troubleshooting

### "libamdhip64.so not found"

The HIP runtime library is missing. If using bundled ROCm (`rocmSupport = true`),
verify `rocmPackages.clr` is available:

    nix build nixpkgs#rocmPackages.clr

If using bring-your-own, check your `LD_LIBRARY_PATH`:

    ls -la /opt/rocm/lib/libamdhip64.so*

### "Permission denied" on /dev/kfd

Your user needs `video` and `render` group membership:

```nix
users.users.yourname.extraGroups = [ "video" "render" ];
```

Then rebuild and relogin.

### "No AMD GPU detected"

Check that the amdgpu kernel module is loaded:

    lsmod | grep amdgpu

On NixOS, ensure firmware is available:

```nix
hardware.firmware = [ pkgs.linux-firmware ];
# or for AMD specifically:
hardware.amdgpu.initrd.enable = true;  # NixOS 24.05+
```

### Kernel pre-compilation fails

Ensure hipcc version matches your GPU target. Check with:

    hipcc --version

gfx1200/gfx1201 requires ROCm 6.4+, gfx1151 requires ROCm 7.2+.
```

- [ ] **Step 2: Verify doc renders**

Run: `head -5 docs/NIXOS.md`

Expected: The title and first lines of the doc.

- [ ] **Step 3: Commit**

```bash
git add docs/NIXOS.md
git commit -m "docs: add NixOS installation and configuration guide"
```

---

### Task 7: Add NixOS section to README.md

**Files:**
- Modify: `README.md:52` (insert after the Install section's closing line)

- [ ] **Step 1: Add NixOS section to README.md**

Insert after line 52 (`[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).`) and before line 54 (`## Inspiration: Lucebox`):

```markdown

## NixOS

First-class support via Nix flake. See [docs/NIXOS.md](docs/NIXOS.md).

```bash
nix develop github:Kaden-Schutt/hipfire  # dev shell with Rust + ROCm + bun
nix build github:Kaden-Schutt/hipfire    # build package
```

NixOS module:

```nix
{
  inputs.hipfire.url = "github:Kaden-Schutt/hipfire";
  # then in configuration.nix:
  services.hipfire.enable = true;
  services.hipfire.gpuTargets = [ "gfx1100" ];
}
```

```

- [ ] **Step 2: Add NIXOS.md to the Documentation table**

In the Documentation table (around line 112 after insertion), add a row:

```markdown
| [NIXOS.md](docs/NIXOS.md) | NixOS flake, module, dev shell |
```

Insert after the `GETTING_STARTED.md` row.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add NixOS section to README"
```

---

### Task 8: Verify flake builds end-to-end

**Files:** None (verification only)

- [ ] **Step 1: Run `nix flake check`**

Run: `nix flake check 2>&1 | tail -20`

Expected: No errors. Warnings about unchecked packages are acceptable.

- [ ] **Step 2: Test dev shell enters**

Run: `nix develop --command bash -c 'echo "rust: $(rustc --version); bun: $(bun --version)"'`

Expected: Prints Rust and bun versions.

- [ ] **Step 3: Test package build (may take several minutes)**

Run: `nix build .#default 2>&1 | tail -10`

Expected: Builds successfully, creates `./result/bin/hipfire` and `./result/bin/hipfire-daemon`.

- [ ] **Step 4: Test kernel build**

Run: `nix build .#hipfire-kernels 2>&1 | tail -10`

Expected: Builds kernels for default arch, creates `./result/kernels/compiled/gfx1100/`.

- [ ] **Step 5: Verify binary runs**

Run: `./result/bin/hipfire --help 2>&1 | head -10`

Expected: hipfire CLI help output.

- [ ] **Step 6: Commit lock file**

```bash
git add flake.lock
git commit -m "chore(nix): add flake.lock"
```

---

### Task 9: Verify daemon binary discovery

**Files:** Potentially modify `cli/index.ts` if `HIPFIRE_DAEMON_BIN` env var is not supported.

- [ ] **Step 1: Check how CLI finds the daemon binary**

Read `cli/index.ts` and search for how the daemon binary path is resolved.
Look for references to `daemon`, `bin`, spawn/exec calls.

- [ ] **Step 2: If `HIPFIRE_DAEMON_BIN` is not supported, add it**

In `cli/index.ts`, find the daemon binary resolution logic and add:

```typescript
// Check HIPFIRE_DAEMON_BIN env var first (for Nix and other package managers)
const daemonBin = process.env.HIPFIRE_DAEMON_BIN
  || path.join(hipfireDir, "bin", "daemon");
```

Use the existing code style and insert at the appropriate location.

- [ ] **Step 3: If changes were made, test the CLI still works**

Run: `HIPFIRE_DAEMON_BIN=./target/release/examples/daemon bun run cli/index.ts --help 2>&1 | head -5`

Expected: CLI help output.

- [ ] **Step 4: Commit if changes were made**

```bash
git add cli/index.ts
git commit -m "feat(cli): support HIPFIRE_DAEMON_BIN env var for packaged installs"
```
