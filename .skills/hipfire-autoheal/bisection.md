# Bisection patterns when the fix catalog doesn't apply

Use these when `playbook.md` fixes 1-10 all miss and you need to localize
where the problem lives.

## A. Does bench work?

```bash
HIPFIRE_KV_MODE=asym3 ~/.hipfire/bin/bench_qwen35_mq4 \
    ~/.hipfire/models/qwen3.5-4b.mq4 \
    --prefill 32 --gen 10 --warmup 5
```

- **Bench works, daemon hangs** → isolates to daemon / serve layer.
  - Drive the daemon directly, skipping Bun:
    ```bash
    (
      echo '{"type":"ping"}'
      sleep 1
      echo '{"type":"load","model":"'"$HOME/.hipfire/models/qwen3.5-4b.mq4"'","params":{"max_seq":16384}}'
      sleep 30
      echo '{"type":"generate","id":"t1","prompt":"Hi","temperature":0,"max_tokens":5,"repeat_penalty":1.0,"top_p":1.0}'
      sleep 20
    ) | timeout 120 ~/.hipfire/bin/daemon
    ```
  - Daemon prints JSON responses. If this works, issue is in `cli/index.ts`
    serve/HTTP layer, not Rust.
- **Bench also fails** → kernel-level issue. Try `HIPFIRE_KV_MODE=q8` to
  bypass asym kernels. If q8 works, the asym kernel is the suspect.

## B. Does it hang on first request or subsequent?

- **First works, second hangs** → state-across-requests bug. Check for:
  - Zombie subprocess from prior request
  - Unfinished drain (`e.generating = true` stuck)
  - Pipe backpressure between Bun and daemon
- **First hangs** → pre-warm or cold JIT. Consult Fix 7.

## C. Kernel bisection via KV mode

When a kernel-level fault is suspected:

```bash
# Try each mode in order — first one that works isolates the bug
HIPFIRE_KV_MODE=q8    ~/.hipfire/bin/bench_qwen35_mq4 <model> --prefill 32 --gen 5
HIPFIRE_KV_MODE=asym4 ~/.hipfire/bin/bench_qwen35_mq4 <model> --prefill 32 --gen 5
HIPFIRE_KV_MODE=asym3 ~/.hipfire/bin/bench_qwen35_mq4 <model> --prefill 32 --gen 5
HIPFIRE_KV_MODE=asym2 ~/.hipfire/bin/bench_qwen35_mq4 <model> --prefill 32 --gen 5
```

If q8 works but all asym fail → the flash-path rotation code (shared across
asym modes). If only one asym mode fails → that bit-width's K kernel.

## D. Flash vs non-flash bisection

Only Q8 has both paths. If Q8 works but asym doesn't, force Q8 flash to rule
out the flash pipeline in general:

```bash
HIPFIRE_KV_MODE=q8 HIPFIRE_ATTN_FLASH=always \
    ~/.hipfire/bin/bench_qwen35_mq4 <model> --prefill 32 --gen 5
```

If Q8 flash works but asym3 flash doesn't, asym3-specific kernel is at fault
(not the flash pipeline).

## E. hipGraph bisection (decode only)

```bash
# Default (capture after warmup):
~/.hipfire/bin/bench_qwen35_mq4 <model> --prefill 32 --gen 5
# Force no-graph:
HIPFIRE_GRAPH=0 ~/.hipfire/bin/bench_qwen35_mq4 <model> --prefill 32 --gen 5
```

0.8B models have a known hipGraph panic — use `HIPFIRE_GRAPH=0` there.

## F. Strace the daemon

If the daemon is at 90% CPU but making no progress, attach strace (install
with `apt install strace` if missing):

```bash
sudo strace -c -p $(pgrep -f 'bin/daemon') -o /tmp/daemon.strace &
sleep 10
kill %1
cat /tmp/daemon.strace
```

Dominant syscalls tell you where time is going:
- `ioctl` heavy → HIP runtime busy (probably fine, just slow)
- `futex` heavy → lock contention
- `read/write` on pipe heavy → stdio flush backpressure with Bun
- `mmap/munmap` heavy → allocator churn

## G. Isolate cli/index.ts vs daemon

Run the daemon with a Python driver (no Bun, no HTTP):

```python
import json, subprocess, time
p = subprocess.Popen(['~/.hipfire/bin/daemon'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
p.stdin.write(json.dumps({"type":"ping"}) + "\n"); p.stdin.flush()
print(p.stdout.readline())
p.stdin.write(json.dumps({
    "type":"load",
    "model":"/full/path/to.mq4",
    "params":{"max_seq":16384}
}) + "\n"); p.stdin.flush()
# read lines until "loaded" appears
while True:
    line = p.stdout.readline()
    if not line: break
    print(line.strip())
    if '"type":"loaded"' in line or '"type":"error"' in line: break
```

If this works but `hipfire run` / `hipfire serve` doesn't, the issue is in
the Bun CLI layer.

## Triage decision tree

```
hipfire misbehaves
│
├─ bench_qwen35_mq4 works?
│  ├─ YES → Problem is in cli/index.ts (Bun) or serve HTTP
│  │        → Run Fix 1 (kill zombies)
│  │        → Try direct daemon stdio (bisection A)
│  │        → Check ~/.hipfire/serve.log for stack traces
│  └─ NO  → Kernel or load issue
│           → Try HIPFIRE_KV_MODE=q8 (bisection C)
│           → Check hipcc / include path (Fix 2, 3)
│           → Check pre-compiled blob presence (Fix 4)
│
├─ model loads but first request hangs/fails?
│  → Fix 6 (max_tokens vs max_seq), Fix 7 (cold JIT), Fix 8 (OOM)
│
├─ subsequent requests hang?
│  → Known issue on BC-250 (see known-issues.md)
│  → Workaround: pkill + restart between requests
│
└─ quality / recall broken?
   → Fix 5 (pre-0.1.5 givens)
   → Verify with Kaden test
```
