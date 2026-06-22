# Utility Scripts

This directory is for operator utilities and reusable helpers: diagnostics,
data conversion, corpus collection, profiling, installs, code generation,
and benchmark helpers. (GPU coordination now lives in the engine: daemons
self-lock, and non-daemon binaries use `hipfire gpu-lock`.)

Do not add new pass/fail test entrypoints here. Put those under `../tests/`
instead. A script is a test entrypoint when its main purpose is to validate a
behavior and communicate success or failure with its exit code.

When a pass/fail script moves to `../tests/`, update callers and documentation
to use the new path directly instead of adding compatibility symlinks or
wrappers here.
