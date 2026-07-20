# hipfire-diag: Interpretation Guide

Interpret JSON from `.agents/skills/hipfire-diag/run-diagnostics.sh` only.
Walk subsystems in the order below. Stay inventory-scoped; do not start
repairs that autoheal owns.

## Reading order

1. `gpu`
2. `build` + binary presence implied by `kernel_tests` / `inference_tests`
3. `kernels` for the detected arch (or all zeros)
4. `kernel_tests`
5. `inference_tests` (if a model path was passed)
6. Summarize; offer one approved next step per failure

## Field map

### `gpu.kfd` = false

**Meaning:** `/dev/kfd` is missing. No usable AMD KFD device node for this
process.

**Safe checks (read-only):**

```bash
ls -l /dev/kfd /dev/dri 2>&1
lsmod | grep -E 'amdgpu|amdkfd' || true
```

**Typical causes (do not assert without evidence):** missing `amdgpu`
driver, container without device pass-through, non-AMD host, WSL without
WSL ROCm setup.

**Remediation:** driver/ROCm install is privileged and distro-specific —
see [fix-suggestions.md](fix-suggestions.md) § GPU / KFD. Get explicit
user approval before any `apt`/`pacman`/`dnf` or reboot.

**Product path:** installed users can also run `hipfire diag` for a broader
platform probe ([docs/CLI.md](../../../docs/CLI.md)).

---

### `gpu.kfd` = true, `gpu.arch` unknown or build hint

**Meaning:** KFD is present, but arch string was not parsed.

Script behavior:

- Tries `rocm-smi --showproductname` for `card` and VRAM.
- Sets `arch` from `target/release/examples/test_kernels` stdout line
  matching `GPU:` when that binary exists.
- Else emits `arch: "unknown (build test_kernels first)"`.

**Safe checks:**

```bash
command -v rocm-smi hipcc rocminfo
rocminfo 2>/dev/null | grep -E 'Name:.*gfx' || true
test -x target/release/examples/test_kernels && target/release/examples/test_kernels 2>&1 | head -20
```

**Do not claim** a specific gfx target until `rocminfo`, the test binary, or
`hipfire diag` prints one.

---

### `kernels.<arch>.blobs` = 0 (and/or `hashes` = 0)

**Meaning:** Under repo-relative `kernels/compiled/<arch>/`, no `.hsaco`
blobs and/or no `.hash` sidecars were found. Script enumerates only:
`gfx1010`, `gfx1030`, `gfx1100`, `gfx1200`, `gfx1201`.

**Implications:**

- Missing blobs → first run may JIT via `hipcc` (slow cold start) if the
  toolchain is present; otherwise kernels fail to load.
- Missing hashes with blobs present → integrity sidecars absent; runtime
  may treat cache as incomplete and recompile. Do not delete blobs to
  “fix” missing hashes without approval.

**Bounded options (user chooses):**

1. Re-fetch install assets: `hipfire update` (installed tree; can rebuild —
   confirm first).
2. Local compile for **one** known arch:
   `./scripts/compile-kernels.sh <arch>` then
   `./scripts/write-kernel-hashes.sh` (needs `hipcc`; minutes).
3. Accept JIT for diagnosis only if `hipcc` works.

Canonical compile entry: [scripts/compile-kernels.sh](../../../scripts/compile-kernels.sh).
Do not invent arch names outside what the machine reports.

---

### `kernel_tests.error` (no binary)

**Meaning:** Neither `target/release/examples/test_kernels` nor
`test_kernelsQA` exists.

Use the build line from the JSON `error` field when present. Script’s
documented rebuild:

```bash
cargo build --release --features deltanet --example test_kernels -p hipfire-runtime
```

(`deltanet` is in package defaults; the flag matches the script and
[docs/VALIDATION.md](../../../docs/VALIDATION.md) kernel-channel route.)

Also check `build.infer` / `build.infer_hfq` — false means those example
binaries were not built yet; same package, different examples.

---

### `kernel_tests.failed` > 0

**Meaning:** Channel tests ran and reported failures. Read `failures[]`
(script keeps up to five lines matching `FAIL` or `PANIC`).

| Failure text (substring) | Interpretation | Next |
|---|---|---|
| `hipcc compilation failed` / `hip_runtime.h` | HIP headers / include path | Prefer autoheal Fix 2 after user ok; or PATH/`-I` checks in fix-suggestions |
| `hipcc not in PATH` / failed to run hipcc | Toolchain not on PATH | `ls /opt/rocm*/bin/hipcc`; export PATH — see fix-suggestions |
| `FAIL` + `NaN` | Numeric mismatch in a named kernel | Capture arch + full failure line; file issue — do not “fix” by deleting caches |
| `PANIC` | Hard fault / hang during test | Stop further GPU stress; report arch + kernel name; autoheal/bisection if serve-related |

Kernel channel role: manual numeric check only — not dispatch bind coverage,
not serve semantics ([docs/VALIDATION.md](../../../docs/VALIDATION.md)).

---

### `inference_tests.skipped`

No model argument. Inventory-only run is valid. Offer optional second run
with an existing weight file if the user wants load/generate smoke.

---

### `inference_tests.error` — model path / missing binary

- Path string in error → file missing; pull or fix path (`hipfire pull` for
  registry tags — [docs/MODELS.md](../../../docs/MODELS.md)).
- Binary missing → build `test_inference` per JSON error /
  fix-suggestions. Note: `test_inference` is a Qwen3.5-oriented example
  (`arch-qwen35`); do not claim it covers every arch_id.

---

### `inference_tests.failed` > 0 or suspicious `tok_s`

**From this runner alone you can only say:** load/generate smoke failed or
produced a tok/s sample. Re-read the captured test stdout if still in the
shell history; the JSON does not embed full logs.

**Hypothesis → handoff (do not auto-apply):**

| Symptom class | Prefer |
|---|---|
| OOM / `hipErrorOutOfMemory` / code=2 | Smaller model, lower `max_seq` — autoheal Fix 8 after approval |
| Illegal memory access / code=700 mid-gen | autoheal Fix 6 / bisection — not diag |
| Model path / format errors | Pull or correct path; quant docs if format mismatch |
| Very low tok/s vs user expectation | Check competing GPU users (`fuser` on `/dev/kfd` — may need sudo), thermals (`rocm-smi`), wrong-arch blobs; **not** a certified floor |

Do **not** paste historical bench tables as “expected minimums” for this
smoke. Measured numbers live in dated owners; protocol in
[docs/methodology/perf-benchmarking.md](../../../docs/methodology/perf-benchmarking.md).

`vram_leak` is whatever substring the script scraped (`LEAK`/`leak` or
`none`). Treat non-`none` as a signal to inspect full test output — not as
proof of a specific historical bug.

---

### Healthy JSON, broken product serve

If `gpu.kfd`, kernels for the live arch, and `kernel_tests` look fine but
`hipfire serve` / chat / multi-turn fails:

1. State that **checkout inventory is OK**.
2. Chain to **hipfire-autoheal**:
   `.agents/skills/hipfire-autoheal/triage.sh`
   then playbook / known-issues / bisection as that skill directs.
3. Do not kill daemons or clear pidfiles from diag.

## Report shape

After interpreting:

1. **Works** — short bullets
2. **Fails** — one bullet per subsystem with the JSON evidence cited
3. **Next** — one optional action each, phrased as a question when destructive
   or privileged (`Want me to …?`)

Never chain-execute install, reboot, `hipfire update`, cache wipes, or
process kills without confirmation.

## Chain to hipfire-autoheal when

- Hang, unresponsive daemon, won’t start, port 11435 conflict
- Stale `~/.hipfire/serve.pid` / multiple daemon PIDs
- Multi-turn recall wrong
- Mid-generation HipError after kernels already pass
- Need env bisect (`HIPFIRE_KV_MODE`, flash, graph) on a running serve path

Stay in **hipfire-diag** for fresh installs, missing blobs, missing
binaries, and channel-test inventory.
