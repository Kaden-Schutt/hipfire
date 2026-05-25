{ lib
, stdenv
, rocmPackages
, gpuTargets ? []
}:

let
  src = lib.cleanSource ./..;
  cargoToml = builtins.fromTOML (builtins.readFile (src + "/Cargo.toml"));
in
stdenv.mkDerivation {
  pname = "hipfire-kernels";
  version = cargoToml.workspace.package.version or cargoToml.package.version;

  inherit src;

  nativeBuildInputs = [
    rocmPackages.clr
    rocmPackages.llvm.clang
  ];

  buildPhase = ''
    runHook preBuild
    export HOME=$TMPDIR
    # Allow partial failures — some kernels are arch-specific and won't
    # compile for every target. The daemon JIT-compiles missing kernels.
    bash scripts/compile-kernels.sh ${lib.concatStringsSep " " gpuTargets} || {
      echo "WARNING: some kernels failed to compile (see above). Daemon will JIT-compile them on first use."
    }
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
