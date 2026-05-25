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
