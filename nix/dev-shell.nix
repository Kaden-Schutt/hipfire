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
    # rocprofv3 CLI — needed for kernel-time attribution via
    # scripts/rocprof-wrap.sh. Internal `begin_timer` wrappers miss
    # uninstrumented kernels (see crates/rdna-compute/src/profile_rocprof.rs
    # — the "hidden lever bug" comment). rocprof is the ground truth.
    rocmPackages.rocprofiler-sdk
  ];

  # Match package.nix + module.nix runtime closure: clr alone is not
  # enough — the daemon dlopens libamdhip64 / rocm-runtime / rocm-comgr /
  # rocprofiler-register at startup. Without the full set, `nix develop`
  # cannot run the daemon ("no ROCm-capable device detected" at
  # initialization). lib.makeLibraryPath stitches the lib/ subdirs.
  LD_LIBRARY_PATH = lib.optionalString rocmSupport
    (lib.makeLibraryPath [
      rocmPackages.clr
      rocmPackages.rocm-runtime
      rocmPackages.rocm-comgr
      rocmPackages.rocprofiler-register
    ]);

  shellHook = ''
    echo "hipfire dev shell"
    echo "  rust: $(rustc --version)"
    echo "  bun:  $(bun --version)"
    ${lib.optionalString rocmSupport ''
      echo "  hip:  $(hipcc --version 2>&1 | head -1)"
    ''}
  '';
}
