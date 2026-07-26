# Known inventory issues → bounded suggestions

Commands below are **opt-in**. Present the relevant snippet; do not run
privileged, destructive, or long rebuild steps without explicit user
approval. Prefer read-only confirmation first.

For serve/runtime repair after inventory is healthy, use
[hipfire-autoheal](../hipfire-autoheal/) (`triage.sh`, `playbook.md`) —
not this file.

---

## `/dev/kfd` missing

**Confirm:**

```bash
ls -l /dev/kfd 2>&1
lsmod | grep -E 'amdgpu|amdkfd' || true
```

**Install paths differ by distro** (privileged; reboot often required).
Examples only — match the user’s OS docs and ROCm version:

```bash
# Debian/Ubuntu-family (example — verify package names for the release)
# sudo apt update
# sudo apt install linux-firmware
# # ROCm/amdgpu: follow https://rocm.docs.amd.com/ for the supported installer
# sudo reboot
```

WSL2: host must expose KFD; typical prep is `amdgpu-install --usecase=wsl`
then the Linux user-space stack. Containers need `/dev/kfd` and render
nodes passed through.

Product onboarding: [docs/GETTING_STARTED.md](../../../docs/GETTING_STARTED.md).

---

## ROCm / HIP user-space missing or incomplete

**Confirm:**

```bash
command -v rocm-smi hipcc rocminfo
ls /opt/rocm/bin/hipcc /opt/rocm*/bin/hipcc 2>/dev/null
```

**Suggestion (distro-specific; approve first):** install the ROCm HIP
runtime/SDK the user already standardizes on (e.g. distro `rocm-hip-runtime`
/ `rocm-hip-sdk`, or AMD’s install guide). Then re-run:

```bash
.agents/skills/hipfire-diag/run-diagnostics.sh
```

---

## `hipcc` not on PATH

```bash
export PATH="/opt/rocm/bin:$PATH"
# if versioned prefix only:
# export PATH="/opt/rocm-<ver>/bin:$PATH"
command -v hipcc && hipcc --version
```

Persist to shell rc only if the user asks. Runtime JIT and
`scripts/compile-kernels.sh` need a working `hipcc`.

Include-path failures (`hip/hip_runtime.h` not found) are runtime/JIT
repair — hand to autoheal Fix 2 (`HIPFIRE_HIPCC_EXTRA_FLAGS` / update),
do not mass-edit system headers from diag.

---

## Pre-compiled kernels missing for this arch

Script counts: `kernels/compiled/<arch>/*.{hsaco,hash}` for
`gfx1010|gfx1030|gfx1100|gfx1200|gfx1201`.

**Confirm arch first** (`rocminfo`, `hipfire diag`, or `test_kernels` GPU line).

**Option A — installed product assets (can rebuild; confirm):**

```bash
hipfire update
```

**Option B — source tree compile (needs hipcc; slow):**

```bash
./scripts/compile-kernels.sh <arch>    # e.g. gfx1201
./scripts/write-kernel-hashes.sh
```

Scripts: [scripts/compile-kernels.sh](../../../scripts/compile-kernels.sh),
[scripts/write-kernel-hashes.sh](../../../scripts/write-kernel-hashes.sh).

**Option C — diagnose with JIT only** if hipcc works and the user accepts
cold-start latency. Do **not** delete `kernels/compiled/**` unless the user
approves a cache reset and understands a full recompile follows.

Installed product blobs are a generated runtime location under the user
hipfire install prefix (see [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md));
this skill’s runner only inspects the repo-local generated compile-output tree
from `scripts/compile-kernels.sh` (generated runtime output, not a tracked
checkout path).

---

## Hash sidecars missing (`blobs` > 0, `hashes` = 0)

```bash
./scripts/write-kernel-hashes.sh
```

Regenerate sidecars; do not delete `.hsaco` files for this symptom alone.

---

## Test / example binaries not built

Match the runner’s error strings when present:

```bash
# Kernel channel binary (VALIDATION.md kernel route)
cargo build --release --features deltanet --example test_kernels -p hipfire-runtime

# Optional inference smoke used by run-diagnostics.sh
cargo build --release --features deltanet --example test_inference -p hipfire-runtime

# Presence flags in JSON "build"
cargo build --release --features deltanet --example infer -p hipfire-runtime
cargo build --release --example infer_hfq -p hipfire-runtime
```

Feature set: package default includes `deltanet` and several `arch-*`
features ([crates/hipfire-runtime/Cargo.toml](../../../crates/hipfire-runtime/Cargo.toml)).
Prefer the exact line from diagnostics JSON over improvised feature combos.

Daemon (serve) is separate from this skill’s inventory:

```bash
cargo build --release --features deltanet --example daemon -p hipfire-runtime
```

---

## Inference smoke OOM or won’t load

**Non-destructive first:**

- Confirm model path exists and format matches what `test_inference` expects
  (Qwen3.5 HFQ-oriented example).
- Try a smaller registry tag via `hipfire pull` — sizes/VRAM:
  [docs/MODELS.md](../../../docs/MODELS.md) (do not hardcode stale GB tables here).
- For **serve**/daemon OOM after inventory is fine → autoheal Fix 8
  (`max_seq`, `kv_cache`, smaller `default_model`) with user approval.

---

## Competing GPU process (suspected)

Read-only where possible:

```bash
# may require privileges for full fuser output
fuser -v /dev/kfd 2>&1 || true
rocm-smi 2>/dev/null | head -40
```

Do not kill other users’ jobs from diag. If hipfire serve is the zombie,
autoheal Fix 1 owns pid/port cleanup **after approval**.

---

## Slow first run / cold JIT

Expected when blobs are missing and hipcc compiles on demand. Fix the
blob inventory (above) rather than treating JIT latency as a kernel bug.
Do not claim a tok/s regression from a single cold `test_inference` sample.

---

## When suggestions are exhausted

1. Re-run `.agents/skills/hipfire-diag/run-diagnostics.sh` and keep full JSON.
2. If inventory is green → `.agents/skills/hipfire-autoheal/triage.sh`.
3. Still stuck → GitHub issue with diag JSON + `rocminfo` gfx name +
   repro steps: https://github.com/warpfront/hipfire/issues

Do not require retired `scripts/coherence-gate-*.sh` batteries as a
diagnostic exit criterion ([docs/VALIDATION.md](../../../docs/VALIDATION.md)).
