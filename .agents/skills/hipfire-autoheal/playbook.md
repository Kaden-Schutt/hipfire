# hipfire-autoheal playbook

Interpret `triage.sh` (and optional `hipfire diag`) output. Work the catalog
top-down unless evidence uniquely matches a later row.

## Operating principles

1. **Read the log.** `tail -n 80 ~/.hipfire/serve.log` usually names the error.
   Triage `likely_issues` only flags common patterns.
2. **Prefer CLI lifecycle over raw kills.** Order:
   plain `hipfire stop` (tracked ownership-validated pid only) →
   (approval + owner inspection)
   `hipfire stop --force` /
   `scripts/serve-restart.sh --kill-only` then
   `hipfire serve 127.0.0.1:<port> -d` for local recovery.
   `hipfire restart -d` and non-`--kill-only` `serve-restart.sh` are **not**
   safe-before-approval: force-reap / `fuser -k`, and they default to
   all-interface `0.0.0.0` bind (no auth/TLS — need explicit exposure approval).
3. **Never blind-kill.** Confirm ownership (`hipfire ps`, pidfile, port
   listener, `/health`) before `pkill`, `kill -9`, `fuser -k`, force, or
   restart. Plain `hipfire stop` will not kill a tracked pid that fails
   validation; force/restart reaping is broader — inspect owners first.
4. **Do not wipe kernel cache** unless a hash/arch mismatch is proven. Rebuild
   cost is minutes on slow hardware. Cache root: `HIPFIRE_KERNEL_CACHE`
   (default `.hipfire_kernels/<arch>/`).
5. **Prefer one-shot env overrides** while bisecting
   (`HIPFIRE_KV_MODE=q8`, `HIPFIRE_ATTN_FLASH=never`, `HIPFIRE_GRAPH=0`)
   over persistent `hipfire config set`. **Daemon-side env only takes effect
   on a process that inherits it:** use `HIPFIRE_LOCAL=1` (or the CLI local
   path) with **no resident serve/GPU lock**, or an explicitly approved stop
   and relaunch with the env applied to the daemon. Prefixing `hipfire run`
   alone attaches to an existing HTTP serve and does **not** reconfigure it.
6. **Do not rebuild the tree** unless code or install assets actually changed.
   End users run `~/.hipfire/bin/`; local `cargo build` only helps after an
   install/update path they use.

## Symptom → diagnosis map

| Observed symptom | First checks | Catalog |
|---|---|---|
| `serve` /health timeout, “port in use”, multiple daemons, hang after first HTTP request | `hipfire ps`; triage `running_state` / `likely_issues`; `ss`/`curl` :11435 | 1, 7 |
| `hip_runtime.h` not found / hipcc compile fail | `serve.log` / JIT error; `which hipcc`; ROCm include layout | 2, 3 |
| First run extremely slow; `kernels: 0 blobs` | triage `hipfire_install`; `hipfire diag` | 4, 7 |
| Multi-turn name/recall wrong on old installs | log / version; see **historical** known-issues | 5 (legacy only) |
| Mid-gen `HipError` 700 illegal memory access | Confirm whether path is legacy/direct without auto-bump; else kernel/KV bisection — **not** automatic `max_seq` repair | 6 (legacy only) or bisection C |
| `HipError` 2 OOM on load | `rocm-smi` VRAM; model size; idle other GPU holders | 8, 10 |
| Prefill ~1 tok/s on non-Qwen-3.5 with asym KV | model family + `kv_cache` | 9 |
| `HipError` 101 invalid device / 201 invalid context | Map code correctly; require independent process/VRAM/port evidence before foreign-holder path | 10 |
| Bench OK, HTTP/serve broken | bisection A/G | bisection.md |
| Quality/perf claim after a repair | [`docs/VALIDATION.md`](../../../docs/VALIDATION.md) route for that claim | — |

---

## Fix catalog

### 1. Contended or zombie serve / stale pidfile

**Symptom:** `/health did not respond…`, “port in use”, multi-turn HTTP hang,
`hipfire ps` shows unexpected daemons, triage: multiple daemon PIDs, stale
`serve.pid`, port listening but `/health` NOT responding.

**Diagnosis:** Two serves share :11435, a dead pid left a pidfile, or an orphan
daemon holds VRAM/GPU lock after crash.

**Minimal repair (safe → stronger):**

```bash
# 1) Inspect only — required before any force/restart
hipfire ps
curl -s --max-time 2 http://127.0.0.1:11435/health || true
ss -ltnp | grep 11435 || true
# 2) Tracked stop (ownership-validated pid only; no orphan/port reap)
hipfire stop
```

**If still stuck — destructive, needs approval + owner inspection first:**

```bash
hipfire stop --force          # tracked stop + orphan daemon reap + free port
# or restart (also force-reaps / fuser -k on stop leg) — still defaults to
# all-interface bind; prefer kill-only + loopback for local testing:
hipfire restart -d            # approval: force-reap stop leg + default 0.0.0.0 bind
# preferred local pattern after force teardown:
scripts/serve-restart.sh 11435 --kill-only
hipfire serve 127.0.0.1:11435 -d
# full scripted relaunch binds 0.0.0.0 — needs explicit all-interface approval
# (trusted firewall or authenticated reverse proxy; serve has no auth/TLS):
# scripts/serve-restart.sh 11435 -- -d
```

Do **not** start with a broad `pkill -9 daemon hipfire`. If you must free a port manually
after approval, prefer `fuser -k <port>/tcp` only when `hipfire stop --force`
is unavailable, and never delete `/tmp/hipfire-gpu.lock` (flock inode; kernel
releases on holder death — unlinking breaks exclusion).

**Bind security:** serve has **no authentication and no TLS**
([`docs/SERVE.md`](../../../docs/SERVE.md)). Local recovery/testing must use
`hipfire serve 127.0.0.1:<port> -d`. Default `hipfire serve -d`,
`hipfire restart -d`, and non-`--kill-only` `serve-restart.sh` expose
`0.0.0.0` — require explicit approval plus trusted firewall or authenticated
reverse proxy.

**Verify:** `hipfire ps` shows one serve; `curl -s http://127.0.0.1:11435/health`
OK; VRAM idle-ish via `rocm-smi --showmeminfo vram` before next large load.

---

### 2. Missing HIP include path (JIT)

**Symptom:**

```text
hipcc compilation failed …
fatal error: 'hip/hip_runtime.h' file not found
```

**Diagnosis:** Distro/ROCm layout where includes are not on the default hipcc
search path. Current hipfire often injects this; older installs and some ROCm
7.x packs still need an override.

**Minimal repair:**

```bash
# one-shot (safe)
export HIPFIRE_HIPCC_EXTRA_FLAGS="-I/opt/rocm/include"
# locate headers if needed:
ls /opt/rocm*/include/hip/hip_runtime.h 2>/dev/null
```

Persistent: only with approval — shell profile or upgrade via `hipfire update`.

**Verify:** retry the failing `hipfire run` / kernel JIT; log no longer shows
`hip_runtime.h`.

---

### 3. hipcc not on PATH

**Symptom:** `failed to run hipcc: No such file or directory`.

**Minimal repair:**

```bash
export PATH="/opt/rocm/bin:$PATH"
# or versioned:
ls /opt/rocm*/bin/hipcc
export PATH="/opt/rocm-7.0.0/bin:$PATH"   # example — use the path that exists
```

Persistent PATH edits need approval.

**Verify:** `command -v hipcc` and a short JIT or `hipfire diag`.

---

### 4. Precompiled kernel blobs missing

**Symptom:** triage `pre-compiled kernels: 0 blobs`; long first-run JIT;
`hipfire diag` kernels section empty for the arch.

**Diagnosis:** Release assets not installed for this gfx target, or install
incomplete. JIT still works (slow cold start).

**Minimal repair:**

```bash
hipfire update    # approval — network + asset replace
# optional local build from a checkout (slow):
# ./scripts/compile-kernels.sh <gfxarch>
```

**Verify:** blob count > 0 under `~/.hipfire/bin/kernels` (or install layout
triage reports); subsequent starts skip multi-minute cold JIT.

---

### 5. Multi-turn recall wrong (“Kendall” / wrong name) — **legacy path**

**Status:** **Historical** for current releases (see `known-issues.md`).
Only apply if the install is pre-fix and triage still flags givens/KV-era
symptoms.

**Minimal repair (legacy installs):**

Daemon-side `HIPFIRE_KV_MODE` only applies if the daemon process inherits it.
Use a confirmed local path (no resident serve), or approved stop + relaunch
with the env on the daemon. Client-only prefix on an attached serve is not a
valid experiment.

```bash
hipfire update                    # approval
# one-shot local bisect first (no resident serve / GPU lock):
HIPFIRE_LOCAL=1 HIPFIRE_KV_MODE=asym3 hipfire run <model> "…"
# persistent only if needed + approved:
hipfire config set kv_cache asym3
hipfire stop                      # then approved restart/serve if needed
# local testing: loopback only (serve has no auth/TLS)
hipfire serve 127.0.0.1:11435 -d  # approval if force-reap path was used instead
```

**Verify:** multi-turn name probe via `/v1/chat/completions` at temperature 0
returns the expected name. For serve semantics broadly, prefer
`scripts/serve_harness.py` when a model path is available
([`docs/VALIDATION.md`](../../../docs/VALIDATION.md)).

---

### 6. `max_tokens` / generation beyond KV capacity — **legacy / direct only**

**Symptom:** mid-generation panic,
`HipError { code: 700 … illegal memory access }` **and** a proven path where
effective prompt + generation exceeds the **loaded** `max_seq` without modern
CLI auto-bump / daemon capacity rejection (old install, direct daemon/bench,
or explicit undersized load params).

**Diagnosis (narrow):** On that legacy/direct path only, generation length
exceeded allocated KV/`max_seq`.

**Modern default:** On current CLI load/request paths, `max_seq` is often
auto-bumped and the daemon can reject over-capacity requests explicitly. A
generic mid-gen code **700 is illegal address / kernel failure**, **not**
evidence for a persistent `max_seq` config repair. Fail closed: route to
kernel/KV bisection (`bisection.md` C–E, playbook 2–4) and gather layer
evidence. Do **not** change `max_seq` unless the legacy/direct overflow is
proven.

**Minimal repair (legacy/direct overflow only, after proof):**

```bash
# one-shot / smaller request first
# then, with approval if persistent and overflow proven:
hipfire config set max_seq 32768
# or per-model:
hipfire config <tag> set max_seq 65536
```

**Verify:** same prompt + `max_tokens` completes without 700 **on the same
path class** that proved overflow. If 700 persists after capacity is adequate,
stop catalog-6 work and bisect kernels/KV.

---

### 7. Cold kernel JIT /health timeout

**Symptom:** `Serve started but /health did not respond within …`; slow APU;
cold kernel cache.

**Diagnosis:** Legitimate first-load compile/upload, not necessarily a hang.

**Minimal repair (non-destructive):**

```bash
tail -f ~/.hipfire/serve.log
# wait for layer load / "warm-up complete"
curl -s http://127.0.0.1:11435/health
```

Do not kill mid-JIT unless the log is wedged with no progress for far longer
than expected (many minutes on large models + cold cache). Subsequent starts
should be seconds if the cache retained blobs.

**Verify:** `/health` OK; a later start (without unapproved restart/force)
is fast if cache retained blobs. Do **not** use `hipfire restart -d` as an
unapproved verification step.

---

### 8. VRAM OOM on load

**Symptom:** `HipError: code=2 out of memory` during load.

**Diagnosis:** Model + KV footprint > free VRAM; or a zombie still holds memory
(catalog 1 + 10 first).

**Minimal repair:**

```bash
# free holders first (see fix 1) — approval for force/restart paths
rocm-smi --showmeminfo vram
# one-shot smaller footprint on a confirmed local path (no resident serve):
HIPFIRE_LOCAL=1 HIPFIRE_KV_MODE=asym2 hipfire run <smaller-tag> "hi"
# persistent (approval):
hipfire config set max_seq 4096
hipfire config set default_model qwen3.5:4b   # example — pick a tag that fits
```

Prefixing env on `hipfire run` while attached to a resident serve does **not**
reconfigure that daemon’s KV mode.

**Verify:** load succeeds; `rocm-smi` shows expected resident weights.

---

### 9. Asym KV on models without full flash path

**Symptom:** Prefill collapses (~1 tok/s); logs show per-token gather/scatter
fallback. Common on non-Qwen-3.5 attention shapes with asym modes.

**Diagnosis:** asym2/3/4 are flash-oriented; wrong pairing forces slow path.

**Minimal repair:**

```bash
# confirmed local path — daemon must inherit HIPFIRE_KV_MODE
HIPFIRE_LOCAL=1 HIPFIRE_KV_MODE=q8 hipfire run <tag> "hi"
# persistent (approval):
hipfire config set kv_cache q8
```

**Verify:** prefill returns to normal order-of-magnitude for that GPU/model
(measurement ≠ admission; no floor claimed here).

---

### 10. GPU / device open failure (foreign holder only with independent evidence)

**HIP codes (current ROCm headers):**

| Code | Enum | Alone proves foreign holder? |
|---:|---|---|
| 101 | `hipErrorInvalidDevice` | No |
| 201 | `hipErrorInvalidContext` | No |
| 2 | out of memory | No — see catalog 8 |

Neither 101 nor 201 alone proves another process owns the GPU. Enter the
foreign-process recovery path only with **independent** evidence: unexpected
PIDs on `/dev/kfd` or render nodes, VRAM held while `hipfire ps` shows no
serve, or a confirmed non-hipfire holder the user recognizes.

**Symptom:** `HipError` 101 / 201 at daemon start; or load OOM with “no” free
VRAM while no hipfire serve shows in `hipfire ps` **plus** process/VRAM/port
evidence above.

**Minimal repair:**

```bash
# read-only identify first
hipfire ps
rocm-smi --showmeminfo vram
ss -ltnp | grep 11435 || true
# sudo only with approval — still identify, do not kill yet:
sudo fuser -v /dev/kfd /dev/dri/renderD128
# stop *your* hipfire first (plain stop if tracked); only then ask to stop
# foreign ML workloads the user confirms
```

Do not kill unknown PIDs without user confirmation of what they are.

**Verify:** device opens; daemon starts; VRAM coherent with `rocm-smi`.

---

## HIP error quick reference

| Code | Meaning (ROCm) | First catalog |
|---:|---|---|
| 2 | out of memory | 8, 1, 10 |
| 101 | invalid device | 10 only with independent holder evidence; else fail closed / install path |
| 201 | invalid context | 10 only with independent holder evidence; else fail closed |
| 700 | illegal memory access (generic) | Kernel/KV bisection; catalog 6 **only** if legacy/direct `max_seq` overflow is proven |
| 999 | unknown (often JIT upstream) | 2, 3, 4 |

---

## After the catalog

1. `known-issues.md` — current arch-specific hangs and dated historical rows.
2. `bisection.md` — bench vs daemon vs CLI; KV/flash/graph isolation (local
   daemon required for env experiments).
3. Escalate to GitHub with full `triage.sh` output, `rocminfo` gfx line,
   repro steps, expected vs actual, and `tail -n 100 ~/.hipfire/serve.log`.
   Repo: https://github.com/warpfront/hipfire/issues

Do not claim kernel/forward correctness from serve success alone. Pick the
route in [`docs/VALIDATION.md`](../../../docs/VALIDATION.md).
