# hipfire-autoheal Playbook

You are interpreting output from `triage.sh` and applying fixes. Work the
catalog in order — do not skip to later fixes until the earlier ones have
ruled themselves out.

## Operating principles

- **Read `serve.log` first.** `tail -50 ~/.hipfire/serve.log` shows the actual
  error most of the time. The `likely_issues` section of triage output flags
  the common patterns; consult the log for details.
- **Kill zombies before relaunching.** 80% of "won't start" reports are two
  serves fighting over port 11435. SIGKILL + stale pidfile removal first.
- **Never rm kernel cache unless diagnosing a hash mismatch.** Rebuilding
  `/tmp/hipfire_kernels/*.hsaco` from scratch is 30s-2min per kernel on slow
  hardware.
- **Prefer env-var override to editing config.** `HIPFIRE_KV_MODE=q8`,
  `HIPFIRE_ATTN_FLASH=never`, `HIPFIRE_LOCAL=1` override config for one
  invocation. Use them to bisect without committing a config change.
- **Don't rebuild unless code actually changed.** Users run binaries from
  `~/.hipfire/bin/`; rebuilding locally only helps them after
  `scripts/install.sh` re-pulls release assets.

## Fix catalog — in order of frequency

### Fix 1: kill zombie serves (80% of "won't start" reports)

**Symptom:** `hipfire serve -d` reports "/health did not respond within 5 min",
or multi-turn HTTP hangs while `hipfire ps` shows multiple daemon PIDs.

**Triage signal:** `likely_issues` section says `multiple daemon PIDs running`,
or `running_state` shows two daemon PIDs and port 11435 is listening but
`/health: NOT responding`.

**Fix:**
```bash
pkill -9 daemon bun 2>/dev/null
rm -f ~/.hipfire/serve.pid
sleep 2
ss -tln | grep 11435 || echo "port free"   # should say "port free"
hipfire serve -d
```

### Fix 2: missing HIP include path

**Symptom:**
```
hipcc compilation failed for <kernel>:
<file>.hip:1:10: fatal error: 'hip/hip_runtime.h' file not found
```

**Affects:** V620 on CachyOS, some other distro-packaged ROCm 7.x builds.

**Fix (hipfire v0.1.5+ does this automatically):**
```bash
export HIPFIRE_HIPCC_EXTRA_FLAGS="-I/opt/rocm/include"
# or upgrade:
hipfire update
```

### Fix 3: hipcc not in PATH

**Symptom:** `failed to run hipcc: No such file or directory`.

**Fix:**
```bash
export PATH="/opt/rocm/bin:$PATH"
echo 'export PATH="/opt/rocm/bin:$PATH"' >> ~/.bashrc
```

On some distros: `/opt/rocm-<version>/bin`. Check `ls /opt/rocm*/bin/hipcc`.

### Fix 4: pre-compiled kernel blobs missing

**Symptom:** `hipfire diag` shows `kernels/<arch>: 0 blobs`. Long JIT wait
on every kernel first-run.

**Triage signal:** `hipfire_install` section says `pre-compiled kernels: 0 blobs`.

**Fix:**
```bash
hipfire update                              # re-fetches release assets
```

If still empty, the release for your arch wasn't packaged. JIT fallback
works (slow first run only).

### Fix 5: multi-turn recall fails ("Kendall" instead of "Kaden")

**Symptom:** Multi-turn prompt about someone's name returns a different name.
Only visible on 9B+ Qwen 3.5 models.

**Triage signal:** `likely_issues` says `pre-0.1.5 givens4 KV`.

**Fix:**
```bash
hipfire update                              # pull v0.1.5+
hipfire config set kv_cache asym3
hipfire stop; hipfire serve -d
```

**Verify:**
```bash
curl -s http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5:9b","messages":[
    {"role":"user","content":"My name is Kaden."},
    {"role":"assistant","content":"Hello Kaden!"},
    {"role":"user","content":"What is my name? One word."}
  ],"max_tokens":30,"temperature":0}' | grep -oE '"content":"[^"]+"'
```

Expected: `"content":"Kaden"` (think block stripped).

### Fix 6: max_tokens exceeds KV cache capacity

**Symptom:** `panicked at 'Result::unwrap()': HipError { code: 700, message:
"hipMemcpy H2D: illegal memory access" }` — always mid-generation.

**Fix (v0.1.5 auto-bumps via `buildLoadMessage`):**
```bash
hipfire config set max_seq 32768
# or per-model:
hipfire config qwen3.5:9b set max_seq 65536
```

### Fix 7: serve /health timeout on cold kernel JIT

**Symptom:** `Serve started but /health did not respond within 5min`.

**Affects:** Slow APUs (gfx1013) with a cold `/tmp/hipfire_kernels/` cache
for a 9B model. Legitimate first-run wait of 2-5 minutes.

**Fix:** wait. Tail the log:
```bash
tail -f ~/.hipfire/serve.log
# watch "loading layer N/M" progression
# once "warm-up complete" appears:
curl -s http://localhost:11435/health
```

Subsequent starts hit the kernel cache and come up in seconds.

### Fix 8: VRAM OOM on model load

**Symptom:** `HipError: code=2 out of memory` during `load_model`.

**Triage signal:** `vram` section shows tight free VRAM vs model size.

**Fix:**
```bash
hipfire config set max_seq 4096             # from default 32768
hipfire config set kv_cache asym2           # 6× compression vs fp32
# or smaller model:
hipfire config set default_model qwen3.5:4b
```

### Fix 9: asym modes with non-Qwen-3.5 model

**Symptom:** Prefill drops to ~1 tok/s. Daemon logs show per-token fallback.

**Cause:** asym3/asym4/asym2 are flash-only. Qwen 3 (standard attention, no
DeltaNet) or models with non-MQ/HFQ weights fall back to per-token
gather/scatter.

**Fix:**
```bash
HIPFIRE_KV_MODE=q8 hipfire run qwen3:8b "hi"
# or persistently:
hipfire config set kv_cache q8
```

### Fix 10: GPU claimed by another process

**Symptom:** `HipError code=201 hipErrorInvalidDevice` at daemon startup.

**Fix:**
```bash
sudo fuser -v /dev/kfd /dev/dri/renderD128   # see who holds it
# kill other ML workloads (ollama, llama.cpp, pytorch notebook)
```

## HIP error code quick reference

```
code=2   hipErrorOutOfMemory        — reduce max_seq / smaller model
code=201 hipErrorInvalidDevice      — another process has the GPU
code=700 illegal memory access      — kernel bug or OOB (bisect KV mode)
code=999 hipErrorUnknown            — almost always JIT compile failure upstream
```

## When to escalate to GitHub issues

File at https://github.com/Kaden-Schutt/hipfire/issues with `triage.sh` output if:

- `hipfire diag` reports a healthy GPU but nothing works
- A specific model tag consistently breaks (say which)
- Bench works on some sizes but not others (kernel bug)
- Behavior regressed after `hipfire update`

See also: `bisection.md` for when the standard fixes don't apply, and
`known-issues.md` for tracked bugs.
