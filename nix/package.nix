{ lib
, rustPlatform
, rocmPackages
, bun
, makeWrapper
, rocmSupport ? true
, src ? lib.cleanSource ./..
, cargoLockFile ? ../Cargo.lock
}:

let
  cargoToml = builtins.fromTOML (builtins.readFile (src + "/Cargo.toml"));
in
rustPlatform.buildRustPackage {
  pname = "hipfire";
  version = cargoToml.workspace.package.version or cargoToml.package.version;

  inherit src;
  cargoLock.lockFile = cargoLockFile;
  doCheck = false;  # tests require GPU

  # The main binaries are cargo [[example]] targets, not [[bin]].
  buildPhase = ''
    runHook preBuild
    cargo build --release --features deltanet \
      --example daemon --example infer --example infer_hfq \
      -p hipfire-runtime
    runHook postBuild
  '';

  dontCargoInstall = true;

  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin

    # Install and wrap daemon binary with LD_LIBRARY_PATH for libamdhip64.so dlopen
    cp target/release/examples/daemon $out/bin/hipfire-daemon-unwrapped
    makeWrapper $out/bin/hipfire-daemon-unwrapped $out/bin/hipfire-daemon \
      ${lib.optionalString rocmSupport
        "--prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath [
          rocmPackages.clr
          rocmPackages.rocm-runtime
          rocmPackages.rocm-comgr
          rocmPackages.rocprofiler-register
        ]}"}

    # Install other binaries
    cp target/release/examples/infer $out/bin/hipfire-infer 2>/dev/null || true
    cp target/release/examples/infer_hfq $out/bin/hipfire-infer-hfq 2>/dev/null || true

    # Install CLI (TypeScript, invoked via bun)
    mkdir -p $out/share/hipfire/cli
    cp -r cli/. $out/share/hipfire/cli/
    # Remove dev artifacts
    rm -rf $out/share/hipfire/cli/node_modules \
           $out/share/hipfire/cli/.gitignore \
           $out/share/hipfire/cli/tsconfig.json \
           $out/share/hipfire/cli/bun.lock
    find $out/share/hipfire/cli/ -maxdepth 1 -type f \
         \( -name '*.test.ts' -o -name 'test_*.ts' -o -name 'bench_*.ts' \) \
         -delete 2>/dev/null || true

    # Create hipfire CLI wrapper
    # HIPFIRE_DAEMON_BIN tells the CLI where to find the wrapped daemon
    makeWrapper ${bun}/bin/bun $out/bin/hipfire \
      --add-flags "run $out/share/hipfire/cli/index.ts" \
      --set HIPFIRE_DAEMON_BIN $out/bin/hipfire-daemon \
      ${lib.optionalString rocmSupport
        "--prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath [
          rocmPackages.clr
          rocmPackages.rocm-runtime
          rocmPackages.rocm-comgr
          rocmPackages.rocprofiler-register
        ]}"}

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
