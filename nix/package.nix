{ lib
, rustPlatform
, rocmPackages
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
    cargo build --release -p hipfire-cli
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

    # Install the native Rust control plane. HIPFIRE_DAEMON_BIN points it at
    # the ROCm-wrapped daemon rather than relying on a source-tree layout.
    cp target/release/hipfire $out/bin/hipfire-unwrapped
    makeWrapper $out/bin/hipfire-unwrapped $out/bin/hipfire \
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
    homepage = "https://github.com/warpfront/hipfire";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "hipfire";
  };
}
