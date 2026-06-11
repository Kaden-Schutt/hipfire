# Utility Scripts

This directory is for operator utilities and reusable helpers: diagnostics,
data conversion, corpus collection, profiling, installs, code generation,
benchmark helpers, and shared shell utilities such as `gpu-lock.sh`.

Do not add new pass/fail test entrypoints here. Put those under `../tests/`
instead. A script is a test entrypoint when its main purpose is to validate a
behavior and communicate success or failure with its exit code.

When a pass/fail script moves to `../tests/`, update callers and documentation
to use the new path directly instead of adding compatibility symlinks or
wrappers here.
