{ config, lib, pkgs, ... }:

let
  cfg = config.services.hipfire;


  # Build config.json from typed options — camelCase NixOS options → snake_case JSON keys
  configAttrs = {
    port = cfg.port;
    default_model = cfg.defaultModel;
    temperature = cfg.temperature;
    top_p = cfg.topP;
    max_tokens = cfg.maxTokens;
    max_seq = cfg.maxSeq;
    repeat_penalty = cfg.repeatPenalty;
    kv_cache = cfg.kvCache;
    dflash_mode = cfg.dflashMode;
    idle_timeout = cfg.idleTimeout;
  } // cfg.extraSettings;

  configJson = pkgs.writeText "hipfire-config.json"
    (builtins.toJSON configAttrs);
  perModelConfigJson = pkgs.writeText "hipfire-per-model-config.json"
    (builtins.toJSON cfg.perModelSettings);

  # Resolve source: explicit src > github.rev > package default
  effectiveSrc =
    if cfg.src != null then cfg.src
    else if cfg.github.rev != null then
      pkgs.fetchFromGitHub {
        owner = cfg.github.owner;
        repo = cfg.github.repo;
        rev = cfg.github.rev;
        hash = cfg.github.hash;
      }
    else null;

  hipfirePkg =
    if effectiveSrc != null then
      cfg.package.override { src = effectiveSrc; cargoLockFile = "${effectiveSrc}/Cargo.lock"; }
    else
      cfg.package;
  hipfireKernelsPkg =
    if cfg.kernelsPackage == pkgs.hipfire-kernels
    then cfg.kernelsPackage.override { gpuTargets = cfg.gpuTargets; }
    else cfg.kernelsPackage;

  envList =
    (lib.mapAttrsToList (k: v: "${k}=${v}") cfg.environment)
    ++ [ "HIPFIRE_MODELS_DIR=${cfg.modelDir}" ]
    ++ lib.optionals cfg.rocmSupport [
      "LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
        pkgs.rocmPackages.clr
        pkgs.rocmPackages.rocm-runtime
        pkgs.rocmPackages.rocm-comgr
        pkgs.rocmPackages.rocprofiler-register
      ]}"
      # JIT hipcc needs HIP_PATH (clr) and rocm-device-libs bitcode dir.
      # Without these the daemon panics on first kernel compile.
      "HIP_PATH=${pkgs.rocmPackages.clr}"
      "HIPFIRE_HIPCC_EXTRA_FLAGS=--rocm-device-lib-path=${pkgs.rocmPackages.rocm-device-libs}/amdgcn/bitcode"
    ];
in
{
  options.services.hipfire = {

    enable = lib.mkEnableOption "hipfire inference daemon";

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open the API port in the firewall.";
    };

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.hipfire;
      defaultText = lib.literalExpression "pkgs.hipfire";
      description = "The hipfire package to use.";
    };

    src = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = ''
        Custom source tree for building hipfire. Overrides the package's
        default source. Set this directly, or use the github.* options
        for convenience.
      '';
    };

    github = {
      owner = lib.mkOption {
        type = lib.types.str;
        default = "Kaden-Schutt";
        description = "GitHub repository owner.";
      };

      repo = lib.mkOption {
        type = lib.types.str;
        default = "hipfire";
        description = "GitHub repository name.";
      };

      rev = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "master";
        description = ''
          Git revision to build from (branch name, tag, or commit hash).
          When set, fetches the source from GitHub instead of using the
          local flake source. Takes precedence unless src is also set.
        '';
      };

      hash = lib.mkOption {
        type = lib.types.str;
        default = "";
        example = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
        description = ''
          SRI hash of the fetched source. Required when github.rev is set.
          Set to "" and build once to get the correct hash from the error.
        '';
      };
    };

    kernelsPackage = lib.mkOption {
      type = lib.types.package;
      default = pkgs.hipfire-kernels;
      defaultText = lib.literalExpression "pkgs.hipfire-kernels";
      description = "Pre-compiled GPU kernels package.";
    };

    gpuTargets = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [ "gfx1100" "gfx1030" ];
      description = ''
        GPU architectures to compile kernels for. Must be set explicitly —
        Nix cannot probe hardware at evaluation time.

        Detect yours:
          rocminfo 2>/dev/null | grep -oP 'amdgcn-amd-amdhsa--\K\S+' | sort -u
        or:
          grep gfx_target_version /sys/class/kfd/kfd/topology/nodes/*/properties
      '';
    };

    rocmSupport = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Use nixpkgs ROCm libraries (rocmPackages.clr).
        When true (the default), the module automatically enables
        hardware.graphics and registers the ROCm ICD so the HIP
        runtime can discover GPUs. Set to false only if you provide
        your own libamdhip64.so via services.hipfire.environment.
      '';
    };

    # ── Inference settings (written to config.json) ──────────

    port = lib.mkOption {
      type = lib.types.port;
      default = 11435;
      description = "Port for the OpenAI-compatible API server.";
    };

    defaultModel = lib.mkOption {
      type = lib.types.str;
      default = "";
      example = "qwen3.5:27b";
      description = "Model to pre-warm on startup. Empty = none.";
    };

    temperature = lib.mkOption {
      type = lib.types.float;
      default = 0.3;
      description = "Sampling temperature.";
    };

    topP = lib.mkOption {
      type = lib.types.float;
      default = 0.8;
      description = "Nucleus sampling threshold.";
    };

    maxTokens = lib.mkOption {
      type = lib.types.int;
      default = 512;
      description = "Per-request token cap.";
    };

    maxSeq = lib.mkOption {
      type = lib.types.int;
      default = 32768;
      description = "KV cache physical capacity (tokens).";
    };

    repeatPenalty = lib.mkOption {
      type = lib.types.float;
      default = 1.05;
      description = "Repetition penalty. Keep conservative — 1.3+ causes MQ4 gibberish at low temp.";
    };

    kvCache = lib.mkOption {
      type = lib.types.str;
      default = "auto";
      example = "asym3";
      description = "KV cache quantization mode: auto / q8 / asym4 / asym3 / asym2 / turbo / turbo4 / turbo3 / turbo2.";
    };

    dflashMode = lib.mkOption {
      type = lib.types.enum [ "on" "off" "auto" ];
      default = "off";
      description = "DFlash speculative decode: on / off / auto.";
    };

    idleTimeout = lib.mkOption {
      type = lib.types.int;
      default = 300;
      description = "Seconds before evicting loaded model from VRAM. 0 = never.";
    };

    extraSettings = lib.mkOption {
      type = lib.types.attrsOf lib.types.anything;
      default = { };
      description = ''
        Additional config.json keys (snake_case) not covered by dedicated options.
        These are merged last and can override typed options.
      '';
    };

    # ── Per-model overrides ──────────────────────────────────

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
        Keys are model tags, values are config attrsets (snake_case keys).
      '';
    };

    # ── Runtime / service options ────────────────────────────

    modelDir = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/hipfire/models";
      description = ''
        Directory containing model files (.mq4, .hfq6, etc.).
        The CLI and daemon resolve models from this path via HIPFIRE_MODELS_DIR.
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

    {
      assertions = [
        {
          assertion = cfg.gpuTargets != [ ];
          message = ''
            services.hipfire.gpuTargets is empty. Set it to your GPU architecture(s).
            Detect yours by running:
              rocminfo 2>/dev/null | grep -oP 'amdgcn-amd-amdhsa--\K\S+' | sort -u
            Example: services.hipfire.gpuTargets = [ "gfx1100" ];
          '';
        }
        {
          assertion = !cfg.rocmSupport || config.hardware.graphics.enable;
          message = ''
            services.hipfire requires hardware.graphics.enable = true so the
            HIP runtime can discover GPUs via the Mesa/DRI stack.
            Either set hardware.graphics.enable = true (recommended) or set
            services.hipfire.rocmSupport = false and provide your own
            libamdhip64.so via services.hipfire.environment.
          '';
        }
      ];
    }

    # GPU driver stack (ICD + DRI). The HIP runtime discovers GPUs via
    # the ROCm ICD registered through hardware.graphics.extraPackages.
    # Without this, the daemon loads libamdhip64.so but hipGetDeviceCount
    # returns 0 on hosts that don't manually configure the graphics stack.
    (lib.mkIf cfg.rocmSupport {
      hardware.graphics.enable = lib.mkDefault true;
      hardware.graphics.extraPackages = [
        pkgs.rocmPackages.clr.icd
        pkgs.rocmPackages.clr
      ];
    })

    # Expose the CLI globally so `hipfire pull`, `hipfire diag`, etc. work
    { environment.systemPackages = [ hipfirePkg ]; }

    (lib.mkIf cfg.openFirewall {
      networking.firewall.allowedTCPPorts = [ cfg.port ];
    })

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
          mkdir -p ${lib.escapeShellArg cfg.modelDir}
          cp -f ${configJson} /var/lib/hipfire/.hipfire/config.json
          cp -f ${perModelConfigJson} /var/lib/hipfire/.hipfire/per_model_config.json
          ln -sf ${hipfirePkg}/bin/hipfire-daemon /var/lib/hipfire/.hipfire/bin/daemon
          ln -sfn ${hipfireKernelsPkg}/kernels /var/lib/hipfire/.hipfire/bin/kernels
        '';
      };

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
          StateDirectory = "hipfire";
          CacheDirectory = "hipfire";
          WorkingDirectory = "/var/lib/hipfire";
          Environment = envList ++ [
            "HIPFIRE_KERNEL_CACHE=/var/cache/hipfire/kernels"
          ];
        };
        script = ''
          export HOME=/var/lib/hipfire
          if ! ${hipfirePkg}/bin/hipfire-daemon --precompile; then
            echo "WARNING: kernel pre-compilation failed (exit $?). Daemon will JIT-compile on first request." >&2
          fi
        '';
      };

      systemd.services.hipfire = {
        description = "hipfire inference daemon";
        after = [ "network.target" "hipfire-precompile.service" ];
        wantedBy = [ "multi-user.target" ];
        path = lib.optionals cfg.rocmSupport [ pkgs.rocmPackages.clr ];
        serviceConfig = {
          ExecStart = "${hipfirePkg}/bin/hipfire serve";
          Restart = "on-failure";
          RestartSec = 5;
          User = cfg.user;
          Group = cfg.group;
          # systemd creates these under /var/lib + /var/cache and chowns
          # them to User:Group with 0700, so the daemon can always write
          # its kernel cache + state without race or perm guesswork.
          StateDirectory = "hipfire";
          CacheDirectory = "hipfire";
          WorkingDirectory = "/var/lib/hipfire";
          Environment = envList ++ [
            "HOME=/var/lib/hipfire"
            "HIPFIRE_KERNEL_CACHE=/var/cache/hipfire/kernels"
          ];
          ProtectSystem = "strict";
          ReadWritePaths = [ cfg.modelDir "/var/lib/hipfire" "/var/cache/hipfire" ];
          PrivateTmp = true;
          NoNewPrivileges = true;
          DevicePolicy = "closed";
          DeviceAllow = [
            "/dev/kfd rw"
            "char-drm rw"      # /dev/dri/* — glob is silently ineffective
                               # under DevicePolicy=closed on systemd 258.5+
            "/dev/accel/accel0 rw"  # kernel 6.2+ DRM_ACCEL node
          ];
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
          mkdir -p ${lib.escapeShellArg cfg.modelDir}
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
        path = lib.optionals cfg.rocmSupport [ pkgs.rocmPackages.clr ];
        serviceConfig = {
          ExecStart = "${hipfirePkg}/bin/hipfire serve";
          Restart = "on-failure";
          RestartSec = 5;
          Environment = envList;
        };
      };
    })
  ]);
}
