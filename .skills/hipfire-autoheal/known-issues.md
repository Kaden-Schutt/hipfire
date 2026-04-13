# Known issues — v0.1.5

Tracked bugs that don't yet have full fixes. Check here before filing a
GitHub issue.

## BC-250 (gfx1013) — daemon HTTP multi-turn hang under specific sequences

**Status:** known, reproducible, no fix yet
**Release:** since 0.1.5
**Affects:** BC-250 APU only (gfx1013 → gfx1010 emulated path)

**Symptoms:**
- `bench_qwen35_mq4` works fine (77 tok/s decode, 241 tok/s prefill on 4B).
- Driving the daemon directly via stdin works fine.
- First HTTP `/v1/chat/completions` request works (single-turn or multi-turn).
- Second consecutive HTTP request with larger `max_tokens` hangs at 90% CPU
  for 2+ minutes. Eventually times out or panics.

**Workaround:**
```bash
# Between sessions, nuke state:
pkill -9 daemon bun
rm -f ~/.hipfire/serve.pid
hipfire serve -d
```

Or keep requests small and serial — each fresh serve handles the first
request correctly.

**Investigation notes:**
- Not a port conflict (`/health` returns OK).
- Not kernel JIT (kernels cached after first run).
- `strace` shows Rust-level busy loop (90% CPU), no syscall activity, so it's
  spinning in Rust state management rather than blocked on GPU.
- Isolated to the Bun serve wrapper layer — direct daemon stdin path is fine.

See bisection.md sections A and G for the repro steps.

## 0.8B + hipGraph panic

**Status:** known, worked around in speed-gate.sh
**Affects:** Qwen 3.5 0.8B with `HIPFIRE_GRAPH=1`

**Symptom:** daemon panics during hipGraph capture on 0.8B.

**Workaround:** use `HIPFIRE_GRAPH=0` for 0.8B, or just use the default
(non-graph) path. Other sizes are fine with hipGraph.

`scripts/speed-gate.sh` already handles this — 0.8B rows skip the `HIPFIRE_GRAPH=1`
flag.

## gfx1010 (RX 5700 XT) full MQ4 retest pending

**Status:** numbers are from v0.1.3-era HF4; MQ4 + asym3 not re-measured
on this arch since 0.1.5 shipped.

**Expected behavior:** should work at comparable or better speeds. The asym3
K kernel has the head_dim=256 2-pass fix (same as every arch), so recall
should be correct.

**Action:** run the same sweep we ran on V620/BC-250 after the release
propagates through tester hardware.

## `hipfire config` TUI on tiny terminals

**Status:** cosmetic, affects ≤ 80-column terminals

**Symptom:** Long enum lists (default_model: 14 options) collapse to
`←→ cycle (N/M)` instead of showing all — by design. But very narrow
terminals can still wrap the row hint.

**Workaround:** resize terminal to ≥ 100 columns, or use
`hipfire config list` for a plain-text listing.

## Qwen 3 (non-3.5) on asym KV modes

**Status:** by design — asym modes are flash-only and Qwen 3 (standard
attention, no DeltaNet) doesn't have a batched flash path for all layer
types.

**Symptom:** decode works, prefill drops to ~1 tok/s due to per-token
gather/scatter fallback.

**Fix:** use `HIPFIRE_KV_MODE=q8` for non-Qwen-3.5 models, or set
`hipfire config set kv_cache q8` persistently.

## Reporting new issues

Run `.skills/hipfire-autoheal/triage.sh` and paste its output at the top of
the GitHub issue. Include:

1. `triage.sh` output (full)
2. `rocminfo | grep -E 'Name:.*gfx'` — your GPU arch
3. Steps to reproduce
4. Expected vs actual output
5. `tail -100 ~/.hipfire/serve.log` if serve-related

Repo: https://github.com/Kaden-Schutt/hipfire/issues.
