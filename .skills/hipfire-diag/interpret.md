# hipfire-diag: Interpretation Guide

You are interpreting output from `run-diagnostics.sh`. Walk the user through
what works, what doesn't, and offer to fix each issue. For complex / runtime
breakage, chain to the `hipfire-autoheal` skill.

## Reading the results

### gpu.kfd = false
**No AMD GPU detected.** The user needs:
- An AMD GPU with RDNA (RX 5000-series through RX 9000-series, Pro V620/W7900, APUs like BC-250 or Steam Deck)
- The `amdgpu` kernel driver loaded
- Check: `ls /dev/kfd` should exist, `lsmod | grep amdgpu` should show the module

Fix on Ubuntu:
```bash
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_*.deb
sudo apt install ./amdgpu-install_*.deb
sudo amdgpu-install --usecase=rocm
sudo usermod -aG render,video $USER
# reboot or log out + back in
```

### gpu.arch = "unknown"
The detector couldn't read the arch. Either:
- ROCm runtime isn't installed → `sudo apt install rocm-hip-sdk` (Ubuntu) / `sudo pacman -S rocm-hip-sdk` (Arch)
- User is in a container without `/dev/kfd` passed through
- WSL2 without `amdgpu-install --usecase=wsl`

### kernels.{arch} = 0
No pre-compiled kernel blobs for this arch. Options:
1. **Fastest:** `hipfire update` — re-downloads the matching release bundle
2. **Build locally:** `./scripts/compile-kernels.sh {arch}` (needs hipcc + 5-10 min)
3. **Accept JIT:** first run will compile every kernel on demand (2-5 min on slow hardware)

### kernel_tests.failed > 0
GPU kernel test failed. Look at the `failures` array:
- **"hipcc compilation failed"** — `hip_runtime.h not found` → chain to `hipfire-autoheal` Fix 2
- **"hipcc not in PATH"** → add `/opt/rocm/bin` to PATH
- **"FAIL: NaN"** → numerical issue in a kernel. File a GitHub issue with the kernel name + arch.
- **"PANIC"** — GPU hang. Likely VGPR overflow or infinite loop. Report with arch and kernel name.

### inference_tests.failed > 0

**Common causes:**
- **OOM** → reduce `max_seq` (see hipfire-autoheal Fix 8) or use a smaller model
- **"illegal memory access"** → max_tokens > max_seq (hipfire-autoheal Fix 6)
- **Model not found** → user didn't `hipfire pull <tag>` yet

### inference_tests.tok_s below expected

**Expected decode minimums (MQ4 + asym3, per-arch):**

| Arch | 0.8B | 4B | 9B | 27B |
|---|---:|---:|---:|---:|
| gfx1100 (7900 XTX) | >350 | >170 | >125 | >45 |
| gfx1030 (V620) | >240 | >90 | >60 | >20 |
| gfx1013 (BC-250) | >200 | >70 | >45 | n/a |
| gfx1010 (5700 XT) | >180 | >55 | >40 | n/a |

If numbers are *much* lower (< 50% of expected):
- Another process competing for GPU (ollama, llama.cpp, comfyui) — `sudo fuser -v /dev/kfd`
- Thermal throttling — check `rocm-smi --showtemp`
- Wrong kernel arch (pre-compiled blobs for different gfx target)
- asym KV with non-Qwen-3.5 model hitting the per-token fallback (hipfire-autoheal Fix 9)

### serve_tests.port_11435 = LISTENING + /health = OK
The background serve daemon is alive. Normal.

### serve_tests.port_11435 = LISTENING + /health = fail
Zombie serve or stuck daemon. Chain to `hipfire-autoheal` Fix 1.

## Offering fixes

After interpreting, tell the user what's broken and offer specific next
actions in order of impact:

1. **What works ✓**
2. **What doesn't ✗** — each item as its own bullet
3. **Recommended action for each** — phrased as "Want me to <action>?"

Example actions:
- "Install ROCm" (walk them through the amdgpu-install path)
- "Run `hipfire update` to re-fetch kernel blobs"
- "Kill zombie serves and restart" (chain to hipfire-autoheal)
- "Pull a smaller model that fits your VRAM"
- "File a GitHub issue with this diag output" (for real bugs)

Never chain-execute fixes without confirming — some actions (like
`hipfire update`) force a rebuild that takes minutes.

## When to chain to `hipfire-autoheal`

- Anything with "hang", "unresponsive", or "won't start"
- Multi-turn recall failures ("Kendall" / "Kade")
- Mid-generation HipError panics
- Port conflicts, serve readiness timeouts
- Anything needing bisection across KV modes / flash on-off / hipGraph

For fresh installs + basic inventory, stay in `hipfire-diag`.
For runtime breakage + fix catalog, hand off to `hipfire-autoheal`.
