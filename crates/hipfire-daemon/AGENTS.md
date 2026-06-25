# AGENTS.md - hipfire-daemon

The daemon owns runtime resource leases before HIP initialization. Keep resource
locking deterministic and compatible with CLI GPU-lock workflows.

## Resource Locks

- `hipfire-daemon` acquires `flock(2)` GPU/NPU/CPU leases before HIP init.
- Single-GPU coordination must share the same inode as the `hipfire gpu-lock`
  CLI: `gpu_lock_path()` (`$HIPFIRE_GPU_LOCKFILE`, else
  `/tmp/hipfire-gpu.lock`).
- Multi-GPU, NPU, and CPU leases use `/tmp/hipfire-resource-locks/<resource>.lock`
  unless overridden by documented env vars.
- `HIPFIRE_RESOURCE_LOCK_WAIT_MS>0` waits for a busy lease; `0` fails fast.
- Do not add stale-lock cleanup that fights `flock`; the kernel releases the
  lock on process death.
