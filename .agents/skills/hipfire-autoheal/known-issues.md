# Known issues (dated)

Caveats that affect runtime triage. **Current** = still treat as live, with a
dated or ref-pinned reproduction. **Historical** = fixed, superseded,
retained measurement debt, or older observation without a current pinned
repro — do not present as today’s default failure mode. Prefer source +
triage over this file when they disagree. **Unknown** = status not verified
on a current binary; fail closed (do not assume default-on or default-off).

Before filing https://github.com/warpfront/hipfire/issues, run
`.agents/skills/hipfire-autoheal/triage.sh` and attach its output.

---

## Current

### Asym KV modes on non-Qwen-3.5 (standard attention) models

| Field | Value |
|---|---|
| **Status** | Current — by design / path limitation |
| **Scope** | Models without the flash-oriented paths asym modes expect |

**Symptom:** Decode may work; prefill drops to ~1 tok/s (per-token
gather/scatter fallback).

**Workaround (safe one-shot on a confirmed local path, then optional config
with approval):**

Daemon-side env must be inherited by the daemon. Use `HIPFIRE_LOCAL=1` with
no resident serve, or approved stop + relaunch. Client-only prefix while
attached to HTTP serve does not reconfigure KV mode.

```bash
HIPFIRE_LOCAL=1 HIPFIRE_KV_MODE=q8 hipfire run <tag> "…"
# hipfire config set kv_cache q8
```

Playbook fix 9.

---

### `hipfire config` TUI on very narrow terminals

| Field | Value |
|---|---|
| **Status** | Current — cosmetic |
| **Scope** | Terminals ≲ 80 columns |

**Symptom:** Long enum rows collapse to cycle hints; extreme widths wrap badly.

**Workaround:** Widen to ≥ ~100 columns, or `hipfire config list` for plain text.

---

## Unknown / needs ref-pinned verification

### Qwen 3.5 0.8B + hipGraph capture panic

| Field | Value |
|---|---|
| **Status** | **Unknown** default path — fail closed until a ref-pinned route trace |
| **Was reported** | Daemon panic during hipGraph capture on 0.8B (older skill notes) |
| **Scope** | Qwen 3.5 0.8B when graph capture is active |

**Do not claim** that normal CLI defaults keep graph off for 0.8B, or that
only forced `HIPFIRE_GRAPH=1` is in scope. Unset `HIPFIRE_GRAPH` may select
arch defaults (including gfx11/gfx12 paths), and related graph toggles can
remain enabled unless explicitly set to `0`. Omitting `HIPFIRE_GRAPH=1` in
`scripts/speed-gate.sh` does **not** prove the graph path is skipped.

**Safe workaround when graph is suspected (one-shot, local path):**

```bash
HIPFIRE_LOCAL=1 HIPFIRE_GRAPH=0 hipfire run qwen3.5:0.8b "…"
```

Until a ref-pinned reproduction and default-route trace exist, treat 0.8B
graph behavior as **unknown**: explicitly set `HIPFIRE_GRAPH=0` on the
executable path under test rather than assuming product defaults.

---

### BC-250 (gfx1013) — HTTP multi-turn hang (historical observation)

| Field | Value |
|---|---|
| **Status** | **Historical observation / unknown on current binaries** — not a current default without fresh repro |
| **Observed era** | ~0.1.5-era notes; no audited ref or refresh date for a current reproduction |
| **Scope** | BC-250 APU (gfx1013 → gfx1010-emulated path). Not a general RDNA default. |

**Symptoms (as originally recorded)**

- Direct model bench / daemon stdin can look fine.
- First HTTP `/v1/chat/completions` OK; a later request (often larger
  `max_tokens`) hangs at high CPU for minutes, then times out or panics.
- `/health` may still answer — not a simple port conflict; not cold JIT after
  kernels are cached.

**If a similar hang appears today:** re-verify on current binaries before
treating it as “the same bug.” Do not default-blame BC-250.

**Workaround if hang is confirmed (destructive — user approval + owner
inspection):**

Prefer CLI force paths over ad-hoc `pkill -9`. Serve has **no auth/TLS**
([`docs/SERVE.md`](../../../docs/SERVE.md)); local recovery uses loopback:

```bash
hipfire stop --force
# or:
scripts/serve-restart.sh 11435 --kill-only
hipfire serve 127.0.0.1:11435 -d
# Do not use scripts/serve-restart.sh without --kill-only unless the user
# explicitly approves all-interface (0.0.0.0) exposure behind a trusted
# firewall or authenticated reverse proxy.
```

Keep requests small and serial, or restart between heavy multi-turn sessions
(restart/force need approval). Bisection: `bisection.md` sections A and G.

**Notes:** Investigation historically pointed at Bun serve / cross-request state
rather than GPU JIT (strace: userspace spin, little syscall progress).

---

## Historical

### Pre-0.1.5 multi-turn recall (givens4 KV — “Kendall” vs “Kaden”)

| Field | Value |
|---|---|
| **Status** | **Historical** — addressed in the 0.1.5-era KV/flash work |
| **Was** | 9B-class Qwen 3.5 multi-turn name probe returned a wrong name under broken KV |

**Do not** treat random modern wrong answers as this bug without version
evidence. If an ancient install still reproduces: `hipfire update` + current
recommended `kv_cache` (playbook fix 5 legacy path), then verify with a fixed
temperature-0 multi-turn probe or `scripts/serve_harness.py`.

Triage still greps logs for `Kendall` as a **legacy marker** only.

---

### gfx1010 (RX 5700 XT) full MQ4 + asym3 retest debt

| Field | Value |
|---|---|
| **Status** | **Historical measurement debt** (called out post-0.1.5); not a runtime defect by itself |
| **Was** | Public numbers on this arch lagged the MQ4/asym3 stack; expected to work with the shared head_dim fixes |

**Action if validating that arch today:** run a current bench/serve matrix and
record identity hashes per [`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md).
Do not quote pre-retest figures as live floors.

---

### SERVE.md “pkill -9 daemon bun” multi-process recipe

| Field | Value |
|---|---|
| **Status** | **Historical pattern** in older prose; superseded by ownership-safe CLI |
| **Prefer** | `hipfire stop`, approved `hipfire stop --force`, approved `hipfire restart -d`, or approved `scripts/serve-restart.sh` |

Blind `pkill -9` is not an autoheal first action (pid reuse, wrong process name,
orphans reparented to PID 1). Plain tracked `hipfire stop` ownership rules:
`crates/hipfire-cli/src/main.rs`. Force/restart reaping is broader — inspect owners
first (playbook / SKILL destructive class).

---

## Reporting new issues

Attach:

1. Full `.agents/skills/hipfire-autoheal/triage.sh` output
2. `rocminfo` gfx `Name:` line (or `hipfire diag`)
3. Exact repro (commands + model tag/path)
4. Expected vs actual
5. `tail -n 100 ~/.hipfire/serve.log` when serve-related
6. hipfire / daemon identity if known (`hipfire` version, binary mtime/md5)

Label severity honestly: hang vs quality vs perf. Perf needs protocol +
identity — not a single tok/s line. Prefer dated/ref-pinned status over
“as of last skill refresh.”
