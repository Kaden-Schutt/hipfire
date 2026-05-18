# Known Issues → Fix Mapping

## /dev/kfd missing
```bash
# Ubuntu/Debian
sudo apt install linux-firmware amdgpu-dkms
sudo reboot

# Arch Linux  
sudo pacman -S linux-firmware
sudo reboot

# Fedora
sudo dnf install kernel-modules-extra
sudo reboot
```

## ROCm runtime not installed
```bash
# Ubuntu 22.04/24.04
sudo apt install rocm-hip-runtime
# OR just the library:
sudo apt install libamdhip64-dev

# Arch Linux
yay -S rocm-hip-runtime
```

## Pre-compiled kernels missing
```bash
# If hipcc is available (ROCm SDK installed):
./scripts/compile-kernels.sh gfx1010  # for RX 5700 XT
./scripts/compile-kernels.sh gfx1030  # for RX 6800 XT
./scripts/compile-kernels.sh gfx1100  # for RX 7900 XTX
./scripts/compile-kernels.sh gfx1200  # for RX 9060/9070
./scripts/compile-kernels.sh gfx1201  # for RX 9070 XT

# After compiling, generate hash files:
./scripts/write-kernel-hashes.sh

# If hipcc is NOT available:
# Download from GitHub releases for your arch
```

## Kernel hash files missing (INCOMPLETE in hipfire diag)
Hash sidecar files validate pre-compiled kernels aren't stale.
Without them, the engine recompiles via hipcc on first run (slow).
```bash
./scripts/write-kernel-hashes.sh
# Or: cargo run --release -p rdna-compute --example gen_kernel_hashes
```

## All "!" output (especially gfx1100)
Pre-compiled kernel blobs are stale — delete and let the engine recompile:
```bash
rm -rf ~/.hipfire/bin/kernels/compiled/gfx1100
# Next run will recompile from correct embedded source
```
Or update: `hipfire update`

## Test binaries not built
```bash
cargo build --release --features deltanet \
  --example test_kernels \
  --example test_inference \
  --example infer \
  --example infer_hfq \
  -p hipfire-runtime
```

## OOM during inference
Try a smaller model:
- 4B fits in 4GB VRAM
- 9B needs 6GB+ VRAM
- 8B Qwen3 needs 5GB+ VRAM

Or reduce context with `--max-tokens` flag or `--turbo4` KV cache compression.

## VRAM leak (65MB per model load)
Update to latest version. Fixed in commit 0d1fca6:
- Call `kv_cache.free_gpu(&mut gpu)` before dropping
- Call `gpu.drain_pool()` after unloading

## Slow first inference (cold kernel compilation)
Pre-compiled kernels eliminate this. Verify:
- `kernels/compiled/{your_arch}/` has .hsaco files
- The binary prints "pre-compiled kernels: kernels/compiled/{arch}" at startup
