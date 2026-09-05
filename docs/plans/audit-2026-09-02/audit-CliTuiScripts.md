<!-- SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Kaden Schutt; hipfire — see LICENSE and NOTICE in the project root. -->

# Audit: CliTuiScripts

# Audit: CliTuiScripts

Scope: `crates/hipfire-cli/src/**`, `crates/hipfire-tui/**`, `scripts/**`, `tools/**`, `.github/workflows/{ci,no-gpu-ci,registry}.yml`. Read-only on master `/home/kaden/ClaudeCode/warpfront/hipfire`.

## Broken

### 1. TUI `/stats` uptime key mismatch — verified

Serve emits `uptime_sec`; TUI reads `uptime_s` and defaults missing numeric keys to `0`, so Live serve Uptime always shows ~0s while queue/requests still populate.

- Serve: `crates/hipfire-cli/src/serve/http.rs:469-482` — `"uptime_sec": meta.started.elapsed().as_secs()`.
- TUI: `crates/hipfire-tui/src/hipfire/dashboard.rs:455-470` — `uptime_s: as_u64("uptime_s")` with `unwrap_or(0)`.
- Render: `crates/hipfire-tui/src/ui.rs:468` — `fmt_uptime(s.uptime_s)`.
- How known: cross-read both sides. Valid object ⇒ `parse_stats` returns `Some` (not "stats unavailable"); only uptime is wrong.

### 2. `stop --force` / `restart` system-wide `pkill -x daemon` — verified

- `crates/hipfire-cli/src/serve/mod.rs:1399-1413`: `pkill -x daemon` (any exact-name daemon), optional `pkill -f target/release/hipfire-quantize`, `fuser -k {port}/tcp`.
- `crates/hipfire-cli/src/main.rs:660-678`: restart always `StopArgs { force: true }`.
- Docs warn (`docs/SERVE.md:47-58`) but restart defaults to force. Status discarded (`let _ = ...status()`).
- How known: read stop + restart + SERVE.md.

### 3. Windows stop/ps/pid ownership is `/proc`-only — verified

- `proc_start_time` `serve/mod.rs:1248-1251`; `pid_owns_listen_port` `:1258-1282`; `validate_serve_pid` `:1305-1316`; stop wait `:1385`; `kill -TERM` `:1376`.
- `ps` + aux: `main.rs:3437-3503` via `/proc`.
- On Windows plain stop: validate fails → deletes `serve.pid` without signaling → orphan serve. Force needs pkill/fuser. Additive to open #669/#643/#655.

### 4. setup + launch_tui extensionless binaries only — verified

- `setup.rs:285-305` installs `daemon`/`hipfire`/`hipfire-tui` without `.exe`.
- `launch_tui` `main.rs:3402-3414` probes `bin/hipfire-tui` only.
- Contrast: `find_daemon` `:5765-5769` prefers `daemon.exe`; `install.ps1` installs `*.exe`.
- Linux setup backup/rollback ordering is solid (stage→backup live→rename; reverse rollback; cleanup after install.json) — not broken.

### 5. Dual automatic CI docs contradiction — verified

| Source | Claim |
|---|---|
| `docs/VALIDATION.md:42-51` | Automatic = only `no-gpu-ci.yml` → `no-gpu-ci.sh` |
| `ci.yml:49-102` | Also on PR: workspace build, `cargo test --lib --workspace`, leanup-ratchets, crate-maps, ratchet-diff, cargo-deny |
| `CLAUDE.md:94-97` | Describes `ci.yml` |
| CONTRIBUTING + PR template | Push `no-gpu-ci.sh` |

`no-gpu-ci.sh` tests a crate subset + change_gate/autoresearch Python; `ci.yml` is full workspace. Both fire on PRs. `registry.yml` is daily `registry_gen.py` only.

### 6. PR template claims coherence-gate gone; still in tree — verified

- PR template `:39-42`: "no longer exist in-tree".
- Present: `coherence-gate-{minimax,ornith15,qwen3-dspark,qwen35-dspark,cohere2moe,deepseek4-mtp,deepseek4-recall}.sh` + `_coherence_runner.py`.
- `gates.sh:7-9` correctly declines to call them. VALIDATION retired section accurate; PR template claim false.

### 7. `bench --concurrency` batch backend always fails — verified

- `bench_concurrency.rs:404-414`: `DaemonDriver::start` always `bail!`s (no multi-inflight Engine API). CLI still admits `--backend batch|both`.

## Missing

### 1. change_gate still wired while slated for deletion — verified

- `no-gpu-ci.sh:25` unittest `tools.change_gate.tests.*`
- PR template `:24-34` required plan/run paste
- `leanup-thresholds.txt:33`, `docs/governance/2026-08-16-phase3-scope.md:75-89`
- Package: `tools/change_gate/**`
- VALIDATION does not list change_gate as automatic. Deletion without CI/PR-template cutover will red no-gpu-ci.

### 2. hw-gate / hardware-evidence CI absent on master — verified

- No `scripts/hw-gate/` (PR #679 path). `ci.yml:9-14` says GPU workflows removed.

### 3. Large script corpus unreachable from CI/docs — structural

- CI reaches ~8 scripts. VALIDATION/gates.sh maintain a small set.
- Orphan/historical clusters: coherence-gate-*, mi300x_*, dflash_diag_*, mtp_train/, reap/, campaign one-offs. No inventory of maintained vs historical.
- verified:false for per-file reachability; structural from entrypoint census.

### 4. Windows lifecycle beyond install.ps1 — verified

- install.ps1 parallel path (no setup atomic rollback/install.json writer).
- update Linux-only (`CLI.md:189`).
- stop/ps/restart Linux tooling (finding 3).

### 5. autoresearch/ar/review still in no-gpu pytest — verified

- `no-gpu-ci.sh:17-20` pytest `autoresearch/ar/tests`. Full `ar/review/**` remains. Same cutover need as change_gate if deletion planned.

## Would change

1. **hours** — Align `/stats` uptime keys (`http.rs:475`, `dashboard.rs:465`) + fixture test. Highest user-visible TUI bug.
2. **days** — Scope stop --force to hipfire-owned PIDs (`serve/mod.rs:1399-1413`); restart plain-stop first.
3. **days** — Windows process control + `.exe` install parity (setup, stop, launch_tui, ps). Coordinates with #669/#643/#655.
4. **days** — Unify VALIDATION/CLAUDE/PR-template with real CI; atomic cut of change_gate + fix coherence-gate in-tree claim; quarantine `coherence-gate-*.sh` under `scripts/historical/`.
5. **hours** — Refuse or HTTP-route `bench --backend batch` (`bench_concurrency.rs:404-414`).

## Confidence

Did: CLI Commands/setup/serve stop+http, TUI dashboard/ui/serve_ctrl, three workflows + no-gpu-ci/gates/install.ps1, VALIDATION/CLI/SERVE/PR template/CLAUDE, change_gate entry, coherence-gate samples, bench_concurrency dead path, script inventory vs CI refs.

Did not: run tests/GPU; full 330-script callgraph; open gh issue bodies; full serve/complete.rs; every TUI chat SSE edge; deep tools/redline beyond CI unittests; autoresearch gate product semantics.

## Parent JSON contract

```json
{
  "slice": "CliTuiScripts",
  "broken": [
    {"title": "TUI /stats uptime key mismatch", "path_line": "crates/hipfire-tui/src/hipfire/dashboard.rs:460-470; crates/hipfire-cli/src/serve/http.rs:469-482", "verified": true, "summary": "Serve emits uptime_sec; TUI reads uptime_s→0."},
    {"title": "stop --force / restart pkill -x daemon", "path_line": "crates/hipfire-cli/src/serve/mod.rs:1399-1413; main.rs:660-678", "verified": true, "summary": "System-wide pkill -x daemon + fuser -k; restart always force."},
    {"title": "Windows stop/ps is /proc-only", "path_line": "serve/mod.rs:1248-1385; main.rs:3437-3503", "verified": true, "summary": "Plain stop deletes pidfile without signaling on Windows."},
    {"title": "setup + launch_tui extensionless only", "path_line": "setup.rs:285-305; main.rs:3402-3414", "verified": true, "summary": "No .exe; diverges from install.ps1 and find_daemon."},
    {"title": "Dual automatic CI docs contradiction", "path_line": "docs/VALIDATION.md:42; ci.yml; no-gpu-ci.yml; CLAUDE.md:94", "verified": true, "summary": "VALIDATION names only no-gpu-ci; ci.yml also gates every PR."},
    {"title": "PR template says coherence-gate gone", "path_line": "PULL_REQUEST_TEMPLATE.md:39-42; scripts/coherence-gate-*.sh", "verified": true, "summary": "≥7 coherence-gate scripts still in tree."},
    {"title": "bench batch backend always bails", "path_line": "bench_concurrency.rs:404-414", "verified": true, "summary": "CLI admits batch/both; DaemonDriver::start always errors."}
  ],
  "missing": [
    {"title": "change_gate still wired", "path_line": "no-gpu-ci.sh:25; PULL_REQUEST_TEMPLATE.md:24-34; tools/change_gate/**", "verified": true, "summary": "CI+PR template require package slated for deletion."},
    {"title": "hw-gate absent on master", "path_line": "scripts/ (no hw-gate/); ci.yml:9-14", "verified": true, "summary": "PR #679 path not present."},
    {"title": "Script orphans", "path_line": "scripts/* vs workflows + VALIDATION", "verified": false, "summary": "CI ~8 scripts; large historical clusters."},
    {"title": "Windows lifecycle gaps", "path_line": "install.ps1; CLI.md:189; serve/mod.rs", "verified": true, "summary": "Parallel install; update Linux-only; stop Linux tooling."},
    {"title": "autoresearch/review in no-gpu pytest", "path_line": "no-gpu-ci.sh:17-20; autoresearch/ar/review/**", "verified": true, "summary": "Still load-bearing for CPU CI."}
  ],
  "changes": [
    {"title": "Align /stats uptime keys", "path_line": "http.rs:475; dashboard.rs:465", "cost": "hours", "summary": "Fix key + fixture test."},
    {"title": "Scope stop --force reaping", "path_line": "serve/mod.rs:1399-1413", "cost": "days", "summary": "Replace pkill -x; restart not always force."},
    {"title": "Windows pid/kill/.exe parity", "path_line": "setup.rs; serve/mod.rs; main.rs", "cost": "days", "summary": "taskkill + .exe names."},
    {"title": "Unify CI docs; cut change_gate wiring", "path_line": "VALIDATION.md; PR template; no-gpu-ci.sh", "cost": "days", "summary": "One merge-bar story; atomic deletion."},
    {"title": "Refuse or wire bench batch", "path_line": "bench_concurrency.rs:404-414", "cost": "hours", "summary": "Do not advertise dead arm."}
  ]
}
```
