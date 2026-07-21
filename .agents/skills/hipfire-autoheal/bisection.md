# Bisection (after the playbook catalog)

Use when playbook fixes 1–10 do not localize the failure. Goal: isolate
**layer** (GPU/kernel vs daemon stdio vs native HTTP service) and **feature flag**
(KV mode, flash, hipGraph) with reversible one-shot env overrides.

Guardrails:

- Prefer env overrides over config writes.
- **Daemon-side env** (`HIPFIRE_KV_MODE`, `HIPFIRE_ATTN_FLASH`,
  `HIPFIRE_GRAPH`, …) only affects a process that **inherits** those vars.
  `hipfire run` attaches to an existing HTTP serve when one is up — prefixing
  the client alone does **not** reconfigure a resident daemon. For sections
  C–E use either:
  - `HIPFIRE_LOCAL=1` (or the CLI local-control path) with **no** resident
    serve / GPU lock, or
  - an explicitly approved stop, then relaunch serve/daemon with the env
    applied to that process.
- Do not kill processes while bisecting unless a wedged GPU blocks the next
  experiment — and then only with approval via `hipfire stop --force` /
  approved `scripts/serve-restart.sh` / approved `hipfire restart -d`, after
  port/process-owner inspection. Plain `hipfire stop` is tracked-pid only.
- Bench binaries and model paths vary by install; substitute the paths
  `hipfire diag` / `ls ~/.hipfire/models` show. Examples below use common
  layouts — **fail closed** if the binary or model is missing.
- Correctness/perf claims after isolation still follow
  [`docs/VALIDATION.md`](../../../docs/VALIDATION.md).

---

## A. Does a direct bench/daemon path work?

```bash
# Example shape — only if this bench binary exists on the install:
HIPFIRE_KV_MODE=q8 ~/.hipfire/bin/bench_qwen35_mq4 \
  ~/.hipfire/models/<model>.mq4 \
  --prefill 32 --gen 10 --warmup 5
```

If no dedicated bench binary, drive the daemon over stdio (section G) or use
`hipfire run` with `HIPFIRE_LOCAL=1` to skip HTTP attach (and ensure no
resident serve holds the GPU lock if the experiment needs a fresh daemon).

| Result | Meaning | Next |
|---|---|---|
| Bench/local OK, HTTP/serve hangs | Native serve / HTTP / pid lifecycle | Playbook 1; section G |
| Bench/local also fails | Load, kernel, or GPU stack | Playbook 2–4, 8, 10; sections C–E |
| Nothing runs | Install/ROCm | `hipfire diag` + hipfire-diag skill |

---

## B. First request vs subsequent

| Pattern | Suspect | Next |
|---|---|---|
| First hangs | Cold JIT, load OOM, readiness | Playbook 7, 8, 4 — tail `~/.hipfire/serve.log` |
| First OK, second hangs | Cross-request state, zombie child, pipe backpressure | Playbook 1; BC-250 historical/unknown if gfx1013; section F |
| Only large `max_tokens` fails | Capacity path or illegal-address — do not assume catalog 6 without legacy proof; bisection C + logs | Playbook 6 (legacy only) or kernel/KV |

---

## C. KV mode bisection

One-shot only; **stop when the first mode works** (conditional break). Requires
a confirmed local daemon path (see guardrails) — not a resident HTTP serve.

```bash
# Ensure no resident serve, or use approved stop first.
for m in q8 asym4 asym3 asym2; do
  echo "=== HIPFIRE_KV_MODE=$m ==="
  if HIPFIRE_LOCAL=1 HIPFIRE_KV_MODE=$m hipfire run <tag> "Say hi in five words."; then
    echo "first working mode: $m"
    break
  fi
done
```

Or the same loop on a bench binary if present (bench inherits env directly).

| Pattern | Suspect |
|---|---|
| Only `q8` works | Asym/flash K path |
| One asym width fails | That bit-width kernel |
| All fail | Load, GPU, or shared non-KV path |

---

## D. Flash vs non-flash (Q8)

Local path required (guardrails). Run modes individually:

```bash
HIPFIRE_LOCAL=1 HIPFIRE_KV_MODE=q8 HIPFIRE_ATTN_FLASH=always hipfire run <tag> "…"
HIPFIRE_LOCAL=1 HIPFIRE_KV_MODE=q8 HIPFIRE_ATTN_FLASH=never  hipfire run <tag> "…"
```

If Q8 flash works but asym flash fails, blame asym-specific kernels, not the
entire flash pipeline.

---

## E. hipGraph on/off (decode)

Local path required. Unset `HIPFIRE_GRAPH` is **not** proven “off” for all
arches/models — pin the value under test. 0.8B graph default is **unknown**
until ref-pinned (`known-issues.md`); prefer explicit `HIPFIRE_GRAPH=0` there.

```bash
HIPFIRE_LOCAL=1 HIPFIRE_GRAPH=0 hipfire run <tag> "…"
# only if intentionally testing capture (approval if it may panic):
HIPFIRE_LOCAL=1 HIPFIRE_GRAPH=1 hipfire run <tag> "…"
```

---

## F. Where is CPU time going? (optional, approval for attach)

If a daemon sits at ~90% CPU with no tokens:

```bash
# identify pid first
hipfire ps
# approval: bounded foreground strace (do not background + sudo kill %1)
sudo timeout 10 strace -c -p <daemon_pid> -o /tmp/daemon.strace
cat /tmp/daemon.strace
```

| Dominating syscalls | Hint |
|---|---|
| `ioctl` | HIP/GPU work (slow vs stuck — correlate with log) |
| `futex` | Lock contention |
| `read`/`write` on pipes | Stdio backpressure with native service parent |
| near-zero syscalls + high CPU | Userspace spin (see BC-250 historical notes) |

---

## G. Daemon stdio vs native HTTP

Minimal Python driver (no HTTP). Adjust daemon path to the install
(`~/.hipfire/bin/daemon` or a local `target/release/examples/daemon`).
Drain generation through a terminal `done`/`error` event inside a
timeout-protected `try`/`finally` before terminate/wait.

```python
import json, subprocess, time, os, select

daemon = os.path.expanduser("~/.hipfire/bin/daemon")
model = os.path.expanduser("~/.hipfire/models/<model>.mq4")
p = subprocess.Popen(
    [daemon],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1,
)

def send(obj):
    p.stdin.write(json.dumps(obj) + "\n")
    p.stdin.flush()

def readline_deadline(deadline):
    while time.time() < deadline:
        r, _, _ = select.select([p.stdout], [], [], max(0.0, deadline - time.time()))
        if not r:
            continue
        line = p.stdout.readline()
        if not line:
            return None
        return line.strip()
    return None

try:
    send({"type": "ping"})
    print(readline_deadline(time.time() + 30))
    send({"type": "load", "model": model, "params": {"max_seq": 8192}})
    deadline = time.time() + 600
    while True:
        line = readline_deadline(deadline)
        if line is None:
            raise TimeoutError("load timed out or daemon closed stdout")
        print(line)
        if '"type":"loaded"' in line or '"type":"error"' in line:
            break
    send({
        "type": "generate",
        "id": "t1",
        "prompt": "Hi",
        "temperature": 0,
        "max_tokens": 5,
        "repeat_penalty": 1.0,
        "top_p": 1.0,
    })
    deadline = time.time() + 300
    while True:
        line = readline_deadline(deadline)
        if line is None:
            raise TimeoutError("generate timed out or daemon closed stdout")
        print(line)
        if '"type":"done"' in line or '"type":"error"' in line:
            break
finally:
    try:
        p.terminate()
    except Exception:
        pass
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(timeout=5)
```

| Result | Meaning |
|---|---|
| Stdio OK, `hipfire serve` / HTTP fails | CLI serve layer, pid/port, or HTTP state |
| Stdio fails | Daemon/GPU/kernels — stay below the HTTP layer |

Also useful: `HIPFIRE_LOCAL=1 hipfire run …` forces one-shot local spawn and
skips attaching to a background serve ([`docs/SERVE.md`](../../../docs/SERVE.md)).

---

## Decision tree

```
hipfire misbehaves
│
├─ triage.sh + hipfire ps + serve.log
│  └─ match playbook 1–10 first
│
├─ direct bench / HIPFIRE_LOCAL / stdio daemon works?
│  ├─ YES → native serve/HTTP lifecycle (playbook 1; section G)
│  └─ NO  → GPU/JIT/KV (playbook 2–4, 8, 10; sections C–E)
│
├─ first request only?
│  └─ cold JIT / OOM / max_seq (playbook 7, 8, 6-legacy-only)
│
├─ subsequent HTTP only?
│  └─ restart hygiene (approval); BC-250 only if gfx1013 + fresh repro
│
└─ quality / wrong tokens on modern install?
   └─ not automatic “Kendall” legacy — bisect KV/flash on local path;
      serve_harness for semantics; path oracle per VALIDATION.md for numbers/state
```

---

## Recovery after bisection

Return the environment to a known lifecycle without leftover overrides:

```bash
hipfire stop          # tracked pid only; approved force/restart if still wedged
# unset experimental overrides in the shell
unset HIPFIRE_KV_MODE HIPFIRE_ATTN_FLASH HIPFIRE_GRAPH HIPFIRE_LOCAL
```

If the GPU is still wedged after experiments, approved force path only (owner
inspection first):

```bash
hipfire stop --force
# or: scripts/serve-restart.sh 11435 --kill-only
```
