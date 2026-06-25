# AGENTS.md - hipfire-lock

This crate owns the shared `flock(2)` primitive and the single-GPU lockfile
contract used by `hipfire gpu-lock` and non-daemon GPU workflows.

## Lock Contract

- The single-GPU mutex path is `gpu_lock_path()`:
  `$HIPFIRE_GPU_LOCKFILE`, otherwise `/tmp/hipfire-gpu.lock`.
- Keep the lockfile inode stable. Do not unlink a flocked lockfile as a release
  mechanism.
- `flock` releases on process death, including crash or SIGKILL. Avoid stale-lock
  cleanup schemes that assume pid files are authoritative.
- Changes here must stay compatible with `crates/hipfire-cli` and tests/scripts
  that call `hipfire gpu-lock {acquire,release,status}`.
