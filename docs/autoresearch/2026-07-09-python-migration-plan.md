# Autoresearch Loop Python Migration — Implementation Plan

> **For agentic workers:** This plan is executed by a **multi-agent Workflow** (per the approved design, `docs/autoresearch/2026-07-09-python-migration-design.md`): one workflow phase per plan phase, modules within a phase fan out to parallel agents, and Codex nodes adversarially verify the two correctness seams (certify-verdict parity, git-CAS/fold). If executed by hand instead, use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the 38-script bash + embedded-python autoresearch harness with one configurable Python package (`autoresearch/ar/`), a guardrailed watcher daemon, and an agent-facing `ar` CLI + skill — reusing the ~40%-built Python track — with the first live run being the Sol/Terra/Luna lever-finding eval on mq4r/gfx1201.

**Architecture:** One package with thin, single-responsibility modules and dependency-injected GPU seams so all decision logic is no-GPU-unit-testable (the existing pattern). Git is piloted via `subprocess` to preserve bash's death-safe `flock` + `update-ref` CAS semantics. Config is per-arch TOML. Migration is strangler-lite: each module is validated against the bash it replaces (fixture parity + one live A/B) *before* the bash is deleted.

**Tech Stack:** Python 3.11+ stdlib only (the decision arms are pure-stdlib — no scipy; `tomllib` for TOML), `subprocess` for git/cargo/daemon, `pytest`, `flock`/`fcntl`, rocprof (`profile_standard`), `scripts/serve_harness.py`, the `hipfire serve` daemon.

## Global Constraints

*(Every task implicitly includes these — exact values from the spec.)*

- **Source of truth:** `loop/gfx1201 @ fd6deaa9`. gfx1201 naming throughout. **Nothing hardcodes arch/model/card** — all config-driven.
- **mqN quant only** — never import ggml/Q4_K formats or code.
- **Python is tooling only** — never the inference hot path.
- **Git via `subprocess`.** GPU lock = `flock` held on an open fd (kernel releases on holder death). **Never unlink the lockfile.**
- **Baseline advance = `git update-ref <ref> <new> <expected>`** (compare-and-swap) under a per-SHA build `flock`.
- **Ledger rows are self-describing:** every row carries `gpu_arch, model, base_sha, variant_sha, prompt_md5, kv, maxtok` and `measurement_hash = sha256("|".join([gpu_arch, model, base_sha, var_sha, prompt_md5, kv, maxtok]))[:16]`.
- **Perf gate is conjunctive:** a WIN requires `kernel_decode_tok_s` UP **and** rocprof kernel-`duration` DOWN (each Mann-Whitney U); either alone ⇒ `DEAD`. Both recorded on every row.
- **Coherence via the real `hipfire serve` path; perf via `serve_harness`.** Raw-daemon voodoo only on the sanctioned short-greedy parity path.
- **Auto-enforced actions (watcher) are dry-run-logged + git-reversible + leashed** (budget/TTL/exhaustion). **master-push and default-flips are never automated** — staged + notified only.
- **Measure at AUTO clock** (`power_dpm_force_performance_level=auto`; `high` underclocks R9700 ~13%). **Byte-identical prompts**, record md5.
- **No-GPU CI must stay green:** `./scripts/no-gpu-ci.sh`. GPU validation: k9lin/gfx1100 + hiptrx/gfx1201 **blocking**, hipx/gfx1151 async.
- **New files** get `# Copyright (c) Kaden Schutt`. **Commits** end `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. **Doc-only commits** use `--no-verify`.

## File Structure (decomposition lock-in)

```
autoresearch/ar/
  __init__.py          # package marker
  config.py            # LoopConfig/WorkerCfg/Bounds dataclasses + load_config(path)->LoopConfig
  db.py                # connect(path); ingest(conn,ledger_dir,bod_glob)->int; wins/best/history/kernel_stats
  gitpilot.py          # gpu_lock(ctx); update_ref_cas; worktree_*; current_sha; show_file; fold; rollover
  census.py            # run_census(...)->bod dict; write_bod(bod,path)
  candidates.py        # Candidate; select(...); gen_digest(...); update_exhaustion(...); is_exhausted(...)
  agent_exec.py        # run_round(harness,model,effort,prompt,cwd,max_turns)->int
  driver.py            # run_loop(cfg,worker,safety_cap)->None
  swarm.py             # launch(cfg,repo)->list[int]
  watcher.py           # Watcher (track/reap/enforce)
  cli.py               # main(argv)  — `ar` entrypoint, role-scoped
  probes.py            # run_probe(name,...)  — thin runner over probes/*.hip
  certify/
    __init__.py
    verdict.py         # Verdict consts; parity_result; perf_result (conjunctive); coherence_result
    perf.py            # mwu(a,b); (helpers)
    coherence.py       # detect_attractor; run_validators; mcnemar_worse
    serve_runner.py    # LiveServeRunner(ServeRunner)  — GPU adapter
    orchestrator.py    # ServeRunner (abc); certify(runner,...)->row
    resolve.py         # resolve_kernel_file; DeadFile/NoOp exceptions
    cross_arch.py      # check_cross_arch(kernel_file,arch,other_archs,repo)->list[str]
  tests/               # test_*.py (ported + new), fixtures/
autoresearch/config/   # loop_gfx1201.toml, prompt_gfx1201.md, ...
autoresearch/db/       # ar.db (gitignored) + schema.sql
autoresearch/ledger/   # *.jsonl (unchanged)
autoresearch/levers/   # gfx1100.md gfx1151.md gfx1201.md (unchanged)
autoresearch/probes/   # *.hip (moved from harness/)
docs/autoresearch/     # architecture.md operations.md config-reference.md
docs/skills/autoresearch-loop.md
```

**Canonical signatures** (referenced across tasks — keep names exact):
- `LoopConfig(arch, baseline_ref, model, kv_mode, max_tokens, prompt_md5, cand_wall, k_exhaust, agent_harness, workers: list[WorkerCfg], bounds: Bounds)`
- `WorkerCfg(card:int, dev:int, model:str, effort:str)` · `Bounds(call_budget:int, wall_ttl_s:int)`
- `perf_result(base_tok, var_tok, base_dur, var_dur, base_clk=None, var_clk=None, alpha=0.05) -> dict` with keys `verdict, tok_delta_pct, dur_delta_pct, tok_p, dur_p`
- `certify(runner, *, arch, kernel, lever, base_daemon, var_daemon, base_ref, model, kv, maxtok, prompt_md5) -> dict` (the ledger row)
- `Candidate(kernel, wall_pct, mem_busy, occ, vgpr, l2_hit, tried, wins, best_win_pct, state)` where `state ∈ {"OPEN","EXHAUSTED"}`
- `update_ref_cas(ref, new_sha, expected, repo) -> bool`

---

## Phase 1 — Foundation (config, db, gitpilot)

### Task 1.1: `ar/config.py` — TOML config loader

**Files:**
- Create: `autoresearch/ar/__init__.py` (empty), `autoresearch/ar/config.py`
- Create: `autoresearch/config/loop_gfx1201.toml` (from spec §6, model = `qwen3.6-35b-a3b.mq4r`; 3 workers Luna@max/Terra@xhigh/Sol@medium on cards 1/2/3)
- Test: `autoresearch/ar/tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path:str)->LoopConfig`; dataclasses `LoopConfig/WorkerCfg/Bounds` (signatures above).

- [ ] **Step 1: Write the failing test**
```python
from autoresearch.ar.config import load_config
def test_loads_workers_and_bounds(tmp_path):
    cfg = load_config("autoresearch/config/loop_gfx1201.toml")
    assert cfg.arch == "gfx1201"
    assert cfg.model.endswith("mq4r")
    assert cfg.k_exhaust == 5
    assert [w.model for w in cfg.workers] == ["gpt-5.6-luna","gpt-5.6-terra","gpt-5.6-sol"]
    assert [w.effort for w in cfg.workers] == ["max","xhigh","medium"]
    assert cfg.bounds.call_budget == 400
```
- [ ] **Step 2: Run** `pytest autoresearch/ar/tests/test_config.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** `config.py`: three `@dataclass` types; `load_config` uses `tomllib.load`, maps `[[workers]]`→`list[WorkerCfg]`, `[bounds]`→`Bounds`, defaults per spec §6. Write `loop_gfx1201.toml` verbatim from spec §6.
- [ ] **Step 4: Run** the test → PASS.
- [ ] **Step 5: Commit** `feat(ar): config.py TOML loader + loop_gfx1201.toml`.

### Task 1.2: `ar/db.py` — durable store + idempotent ingest

**Files:**
- Create: `autoresearch/ar/db.py`, `autoresearch/db/schema.sql`
- Port from: `autoresearch/oracle_db.py` (query fns: `wins/best/history/kernel`), `autoresearch/ar/hipfire_ar.py:50-108` (`db()` schema + `ingest()`)
- Test: `autoresearch/ar/tests/test_db.py`, fixture `autoresearch/ar/tests/fixtures/mini_ledger/*.jsonl` (3 rows: 1 WIN, 1 DEAD, 1 noise) + `bod_gfx1201.json`

**Interfaces:**
- Produces: `connect(path)->sqlite3.Connection`; `ingest(conn, ledger_dir, bod_glob)->int` (rows ingested, idempotent); `wins(conn)`, `best(conn,arch,kernel)`, `history(conn,arch,kernel)`, `kernel_stats(conn,arch,kernel,k)->dict{tried,wins,best_win_pct,consecutive_dead}`.
- Consumes: nothing.

- [ ] **Step 1: Write the failing test**
```python
from autoresearch.ar.db import connect, ingest, kernel_stats
def test_ingest_idempotent(tmp_path):
    db = tmp_path/"ar.db"
    c = connect(str(db))
    n1 = ingest(c, "autoresearch/ar/tests/fixtures/mini_ledger", "autoresearch/ar/tests/fixtures/bod_gfx1201.json")
    n2 = ingest(c, "autoresearch/ar/tests/fixtures/mini_ledger", "autoresearch/ar/tests/fixtures/bod_gfx1201.json")
    assert n1 > 0 and n2 == n1                      # second ingest adds nothing new
    assert c.execute("SELECT count(*) FROM attempts").fetchone()[0] == n1
def test_kernel_stats_counts(tmp_path):
    c = connect(str(tmp_path/"ar.db"))
    ingest(c, "autoresearch/ar/tests/fixtures/mini_ledger", "autoresearch/ar/tests/fixtures/bod_gfx1201.json")
    s = kernel_stats(c, "gfx1201", "fused_qkvza_hfq4g256", k=5)
    assert s["tried"] >= 1 and "consecutive_dead" in s
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `db.py`: `schema.sql` with `attempts(arch,kernel,lever,verdict,tok_delta,dur_delta,profile,base_sha,var_sha,measurement_hash UNIQUE,ts)`, `bod(arch,kernel,wall_pct,l2_hit,mem_busy,occ,vgpr)`, `runs(id,arch,model,card,status,budget,calls,ttl,pid,ts)`. `ingest` upserts on `measurement_hash` (idempotency key) + refreshes `bod` from the json. Port query bodies from `oracle_db.py`.
- [ ] **Step 4: Run** → PASS. Also run `python3 -c "import sqlite3"` sanity.
- [ ] **Step 5: Commit** `feat(ar): db.py durable store + idempotent ledger ingest`.

### Task 1.3: `ar/gitpilot.py` — subprocess git (flock, CAS, worktree)

**Files:**
- Create: `autoresearch/ar/gitpilot.py`
- Port from: `ab_certify_v2p.sh` (the `flock`/`update-ref`/`git show SHA:path` bits), `scripts/gpu-lock.sh` (flock-on-fd semantics)
- Test: `autoresearch/ar/tests/test_gitpilot.py`

**Interfaces:**
- Produces: `gpu_lock(lockfile:str)` (contextmanager, `fcntl.flock` on an held-open fd); `update_ref_cas(ref, new_sha, expected, repo)->bool`; `current_sha(repo, ref)->str`; `show_file(repo, sha, path)->bytes`; `worktree_add(repo, path, ref)`, `worktree_list(repo)`, `worktree_remove(repo, path)`.

- [ ] **Step 1: Write the failing test**
```python
import subprocess, os
from autoresearch.ar.gitpilot import update_ref_cas, current_sha
def _git(repo,*a): return subprocess.run(["git","-C",repo,*a],capture_output=True,text=True).stdout.strip()
def _mkrepo(tmp):
    r=str(tmp); _git(r,"init","-q"); open(f"{r}/x","w").write("1")
    _git(r,"add","x"); subprocess.run(["git","-C",r,"commit","-qm","a"],env={**os.environ,"GIT_AUTHOR_NAME":"t","GIT_AUTHOR_EMAIL":"t@t","GIT_COMMITTER_NAME":"t","GIT_COMMITTER_EMAIL":"t@t"})
    return r
def test_cas_succeeds_when_expected_matches(tmp_path):
    r=_mkrepo(tmp_path); a=current_sha(r,"HEAD")
    open(f"{r}/x","w").write("2"); subprocess.run(["git","-C",r,"commit","-aqm","b"],env={**os.environ,"GIT_AUTHOR_NAME":"t","GIT_AUTHOR_EMAIL":"t@t","GIT_COMMITTER_NAME":"t","GIT_COMMITTER_EMAIL":"t@t"})
    b=current_sha(r,"HEAD"); _git(r,"branch","base",a)
    assert update_ref_cas("refs/heads/base", b, a, r) is True
    assert current_sha(r,"refs/heads/base")==b
def test_cas_fails_when_stale(tmp_path):
    r=_mkrepo(tmp_path); a=current_sha(r,"HEAD"); _git(r,"branch","base",a)
    assert update_ref_cas("refs/heads/base", a, "0"*40, r) is False
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `gitpilot.py`: `update_ref_cas` = `git -C repo update-ref ref new expected` (rc 0 → True). `gpu_lock` opens the lockfile, `fcntl.flock(fd, LOCK_EX)`, yields, releases on exit (never unlinks). Others are thin `subprocess` wrappers.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(ar): gitpilot.py — subprocess git (flock-on-death, update-ref CAS, worktree)`.

---

## Phase 2 — Certify (three-arm gate + guards)

### Task 2.1: `ar/certify/{verdict,perf,coherence}.py` — decision arms (conjunctive perf)

**Files:**
- Create: `autoresearch/ar/certify/__init__.py`, `verdict.py`, `perf.py`, `coherence.py`
- Port from: `certify_verdict.py`, `perf.py`, `coherence_arm.py`. **Extend perf to conjunctive** (currently duration-only).
- Test: `autoresearch/ar/tests/test_certify_arms.py` (port `test_certify_verdict.py` + `test_coherence_arm.py`; add conjunctive-perf cases)

**Interfaces:**
- Produces: `parity_result(base_gens, var_gens)->dict`; `perf_result(base_tok, var_tok, base_dur, var_dur, base_clk=None, var_clk=None, alpha=0.05)->dict{verdict,tok_delta_pct,dur_delta_pct,tok_p,dur_p}`; `coherence_result(base_gens, var_gens, expects=None, alpha=0.05)->dict`; `detect_attractor`, `run_validators`, `mcnemar_worse` (unchanged from `coherence_arm.py`). `Verdict = {"WIN","DEAD","PARITY_FAIL","INCONCLUSIVE"}`.

- [ ] **Step 1: Write the failing test** (the conjunctive gate is the new contract):
```python
from autoresearch.ar.certify.verdict import perf_result
def test_perf_win_requires_both():
    r = perf_result(base_tok=[150,151,149,150,150], var_tok=[160,161,159,160,160],
                    base_dur=[10.0,10.1,9.9,10.0,10.0], var_dur=[9.0,9.1,8.9,9.0,9.0])
    assert r["verdict"] == "WIN" and r["tok_delta_pct"] > 0 and r["dur_delta_pct"] < 0
def test_perf_tok_up_dur_flat_is_dead():
    r = perf_result([150]*5,[160]*5, base_dur=[10.0]*5, var_dur=[10.0]*5)
    assert r["verdict"] == "DEAD"
def test_perf_dur_down_tok_flat_is_dead():
    r = perf_result([150]*5,[150]*5, base_dur=[10.0]*5, var_dur=[9.0]*5)
    assert r["verdict"] == "DEAD"
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**: port the three modules; `perf_result` runs Mann-Whitney U on tok (one-sided, var>base) AND on duration (one-sided, var<base); `verdict="WIN"` iff both p<alpha with correct direction, else `"DEAD"`. Return both deltas + p-values. Port `coherence.py` and `verdict.py` parity logic verbatim.
- [ ] **Step 4: Run** the full ported suite → PASS.
- [ ] **Step 5: Commit** `feat(ar): certify decision arms; perf gate now conjunctive (tok/s up AND duration down)`.

### Task 2.2: `ar/certify/orchestrator.py` — 3-arm gate + ledger row

**Files:**
- Create: `autoresearch/ar/certify/orchestrator.py`
- Port from: `ab_certify_serve.py` (the `ServeRunner` abc + `certify()`), adapting the perf call to the conjunctive `perf_result`.
- Test: `autoresearch/ar/tests/test_orchestrator.py` (port `test_ab_certify_serve.py`; mock runner)

**Interfaces:**
- Consumes: `perf_result`, `coherence_result`, `parity_result` (Task 2.1); `measurement_hash` recipe (Global Constraints).
- Produces: `class ServeRunner` (abc: `parity_gens`, `coherence_gens`, `perf_measure(daemon)->(tok:list,dur:list)`, `clocks`); `certify(runner, *, arch, kernel, lever, base_daemon, var_daemon, base_ref, model, kv, maxtok, prompt_md5)->dict` (the ledger row incl. `measurement_hash`, `tok_delta_pct`, `dur_delta_pct`, `verdict`).

- [ ] **Step 1: Write the failing test**
```python
from autoresearch.ar.certify.orchestrator import certify, ServeRunner
class MockRunner(ServeRunner):
    def __init__(self,**k): self.k=k
    def parity_gens(self,d): return [{"token_ids":[1,2,3],"text":"x"}]      # identical base==var
    def perf_measure(self,d): return (self.k["tok"][d], self.k["dur"][d])
    def coherence_gens(self,d,seeds): return [{"text":"ok fine","token_ids":[5,6,7,8]}]
    def clocks(self,d): return [3200]
def test_win_row_is_self_describing():
    r=MockRunner(tok={"base":[150]*5,"var":[160]*5}, dur={"base":[10.0]*5,"var":[9.0]*5})
    row=certify(r,arch="gfx1201",kernel="k1",lever="L",base_daemon="base",var_daemon="var",
                base_ref="loop/gfx1201",model="qwen3.6-35b-a3b.mq4r",kv="q8",maxtok=128,prompt_md5="d97ec9d3")
    assert row["verdict"]=="WIN"
    assert set(["gpu_arch","model","base_sha","variant_sha","prompt_md5","kv","maxtok","measurement_hash","tok_delta_pct","dur_delta_pct"]) <= set(row)
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**: port `certify()`; order parity→perf→coherence; short-circuit `PARITY_FAIL`; perf via conjunctive `perf_result`; emit the row with `measurement_hash`. Runner supplies `base_daemon`/`var_daemon` SHAs via `perf_measure` keying (mock uses the daemon label).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(ar): certify orchestrator (parity→perf→coherence) + self-describing ledger row`.

### Task 2.3: `ar/certify/serve_runner.py` — live GPU adapter

**Files:**
- Create: `autoresearch/ar/certify/serve_runner.py`
- Port from: `autoresearch/harness/serve_runner.py` (the `LiveServeRunner`), adapting imports to the package + adding `perf_measure` returning `(tok_list, dur_list)`.
- Test: `autoresearch/ar/tests/test_serve_runner_wiring.py` (no-GPU: assert it builds the serve_harness cfg + parses; GPU path is exercised live in Phase 5)

**Interfaces:**
- Consumes: `ServeRunner` abc (Task 2.2); `scripts/serve_harness.py`.
- Produces: `LiveServeRunner(ServeRunner)` with `parity_gens/coherence_gens/perf_measure/clocks` reading through `serve_harness`; captures BOTH `kernel_decode_tok_s` and rocprof kernel-duration in `perf_measure`.

- [ ] **Step 1: Write the failing test** (wiring only — mock `serve_harness`):
```python
def test_perf_measure_returns_tok_and_dur(monkeypatch):
    import autoresearch.ar.certify.serve_runner as sr
    monkeypatch.setattr(sr, "_run_rocprof", lambda *a,**k: ([9.0,9.1,8.9], [160,161,159]))  # (dur, tok)
    r = sr.LiveServeRunner(model="m", arch="gfx1201", dev=0)
    dur, tok = None, None
    tok_list, dur_list = r.perf_measure("some_daemon_bin")
    assert len(tok_list)==3 and len(dur_list)==3
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**: port `LiveServeRunner`; `perf_measure` runs the `profile_standard` rocprof through `serve_harness` and returns `(tok_list, dur_list)`; `coherence_gens` uses the real serve path; `parity_gens` uses the sanctioned raw-daemon short greedy. Keep the `sys.path` shim to `scripts/serve_harness`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(ar): LiveServeRunner GPU adapter (serve_harness tok/s + rocprof duration)`.

### Task 2.4: `ar/certify/{resolve,cross_arch}.py` — target guards

**Files:**
- Create: `autoresearch/ar/certify/resolve.py`, `autoresearch/ar/certify/cross_arch.py`
- Port from: `ab_certify_v2p.sh` (symbol→file + DEAD_FILE/NO_OP), `cross_arch_guard.sh`
- Test: `autoresearch/ar/tests/test_guards.py`

**Interfaces:**
- Produces: `resolve_kernel_file(kernel, repo)->str` (raises `DeadFile`/`NoOp`); `check_cross_arch(kernel_file, arch, other_archs, repo)->list[str]` (archs whose device-TU changed).

- [ ] **Step 1: Write the failing test**
```python
from autoresearch.ar.certify.resolve import resolve_kernel_file, DeadFile
def test_resolve_uncompiled_symbol_raises(tmp_path):
    # a kernel symbol with no include_str!'d .hip => DeadFile
    import pytest
    with pytest.raises(DeadFile):
        resolve_kernel_file("definitely_not_a_kernel_xyz", ".")
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**: `resolve_kernel_file` = `grep -rlE "__global__.*\bKERNEL[[:space:]]*\("` over `kernels/src/*.hip`, cross-checked against `crates/*/src` `include_str!`; `DeadFile` when 0/ambiguous. `check_cross_arch` runs `hipcc --cuda-device-only -E --offload-arch=<other>` for each other arch, normalizes (strip `#line`, blanks, trailing ws), diffs vs baseline; returns changed archs. Skip `.gfxNNNN.hip` arch-suffixed files.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(ar): certify target guards (symbol→file resolve, cross-arch preprocessor-invariance)`.

### Task 2.5 (Codex-verify seam): certify parity vs bash

**Files:** Test: `autoresearch/ar/tests/test_certify_parity.py` (data-driven from a captured ledger case)

- [ ] **Step 1:** Capture one WIN and one DEAD `ab_certify_v2p.sh` ledger row (arch/kernel/deltas) into `fixtures/certify_cases.json`.
- [ ] **Step 2: Write the test**: feed the same base/var stats through `certify()` with a mock runner replaying those measurements → assert identical verdict + delta signs.
- [ ] **Step 3: Run** → PASS. (This is the seam a Codex node adversarially re-checks in the workflow before Phase 2's bash is deleted.)
- [ ] **Step 4: Commit** `test(ar): certify verdict parity vs ab_certify_v2p on captured cases`.

---

## Phase 3 — Census + candidates

### Task 3.1: `ar/census.py` — BOD census

**Files:**
- Create: `autoresearch/ar/census.py`
- Port from: `autoresearch/harness/oracle_profile.sh`
- Test: `autoresearch/ar/tests/test_census.py` (parse a captured rocprof CSV fixture → bod rows; GPU run is live-only)

**Interfaces:**
- Consumes: `gitpilot` (worktree for the baseline), `serve_harness`.
- Produces: `run_census(arch, dev, drm, model, layers, repo)->dict` (`{arch,model,rows:[{kernel,wall_pct,mem_busy,occ,vgpr,l2_hit}]}`); `write_bod(bod, path)`.

- [ ] **Step 1: Write the failing test**
```python
from autoresearch.ar.census import parse_rocprof
def test_parse_rocprof_to_bod_rows():
    rows = parse_rocprof("autoresearch/ar/tests/fixtures/rocprof_gfx1201.csv")
    top = max(rows, key=lambda r: r["wall_pct"])
    assert top["kernel"].startswith("fused_qkvza") and top["wall_pct"] > 10 and 0 <= top["occ"] <= 100
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**: split `oracle_profile.sh` into `run_census` (spawns daemon under `profile_standard`, runs the census generation, invokes rocprof) and `parse_rocprof` (pure CSV→rows, unit-testable). `write_bod` dumps json matching the current `bod_gfx1201.json` shape.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(ar): census.py (oracle_profile port) with unit-testable rocprof parse`.

### Task 3.2: `ar/candidates.py` — candidate selection + exhaustion + digest

**Files:**
- Create: `autoresearch/ar/candidates.py`
- Port from: `exhaustion.py`, `v2/check_exhausted.py`, `v2/gen_digest.py`, `v2/update_exhaustion.py`
- Test: `autoresearch/ar/tests/test_candidates.py` (port `test_exhaustion.py`; fixtures bod + exhaustion + ledger)

**Interfaces:**
- Consumes: BOD dict (Task 3.1), `db` (Task 1.2).
- Produces: `Candidate` dataclass; `select(bod, exhaustion, cand_wall, k)->list[Candidate]`; `is_exhausted(exhaustion, bod, cand_wall, k, folded)->bool`; `gen_digest(exhaustion, bod, cand_wall, k, folded)->str`; `update_exhaustion(exhaustion_path, round, repo, arch)->None`.

- [ ] **Step 1: Write the failing test**
```python
from autoresearch.ar.candidates import select, is_exhausted
BOD={"rows":[{"kernel":"k1","wall_pct":5.0,"mem_busy":50,"occ":40,"vgpr":88,"l2_hit":60},
             {"kernel":"low","wall_pct":1.0,"mem_busy":0,"occ":0,"vgpr":0,"l2_hit":0}]}
def test_below_cand_wall_excluded():
    cands=select(BOD, exhaustion={}, cand_wall=3.0, k=5)
    assert [c.kernel for c in cands]==["k1"]        # 'low' below 3.0 wall dropped
def test_exhausted_after_k_deads():
    assert is_exhausted({"k1":{"consecutive_dead":5}}, BOD, 3.0, 5, folded=[]) is True
def test_open_below_k():
    assert is_exhausted({"k1":{"consecutive_dead":4}}, BOD, 3.0, 5, folded=[]) is False
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**: port the trio; `select` filters `wall_pct>=cand_wall`, joins tried/win counts from `db`, marks `state`; `is_exhausted` = every candidate at `k` consecutive deads (excluding `folded`); `gen_digest` renders the tried-lever text codex reads.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(ar): candidates.py — selection + exhaustion + tried-lever digest`.

---

## Phase 4 — Driver

### Task 4.1: `ar/agent_exec.py` — round dispatch (codex/grok, per-worker model+effort)

**Files:**
- Create: `autoresearch/ar/agent_exec.py`
- Port from: `autoresearch/harness/agent_exec.sh`
- Test: `autoresearch/ar/tests/test_agent_exec.py`

**Interfaces:**
- Produces: `build_argv(harness, model, effort, prompt_file, cwd, max_turns)->list[str]`; `run_round(harness, model, effort, prompt, cwd, max_turns)->int`.

- [ ] **Step 1: Write the failing test**
```python
from autoresearch.ar.agent_exec import build_argv
def test_codex_argv_has_model_and_effort():
    argv = build_argv("codex", "gpt-5.6-luna", "max", "/tmp/p.md", "/repo", 100)
    s=" ".join(argv)
    assert "codex" in argv[0] and "-m" in argv and "gpt-5.6-luna" in argv
    assert 'model_reasoning_effort="max"' in s or "model_reasoning_effort=max" in s
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**: `build_argv` constructs `codex exec --skip-git-repo-check -m <model> -c model_reasoning_effort=<effort> -C <cwd> - ` (prompt via stdin) or the grok equivalent; `run_round` pipes the prompt on stdin, returns rc.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(ar): agent_exec.py — per-worker codex/grok round dispatch (model+effort)`.

### Task 4.2: `ar/driver.py` — self-exhausting per-worker loop

**Files:**
- Create: `autoresearch/ar/driver.py`
- Port from: `autoresearch/harness/v2/driver_v3.sh`
- Test: `autoresearch/ar/tests/test_driver.py` (mock agent_exec + certify + candidates)

**Interfaces:**
- Consumes: `load_config`, `candidates.{is_exhausted,gen_digest,update_exhaustion}`, `agent_exec.run_round`, `gitpilot.update_ref_cas`.
- Produces: `run_loop(cfg, worker, safety_cap, *, hooks=None)->int` (rounds run). `hooks` injects agent/certify/git for testing.

- [ ] **Step 1: Write the failing test**
```python
from autoresearch.ar.driver import run_loop
from autoresearch.ar.config import LoopConfig, WorkerCfg, Bounds
def test_loop_terminates_on_global_exhaustion():
    cfg = LoopConfig("gfx1201","loop/gfx1201","m","q8",128,"md5",3.0,5,"codex",
                     [WorkerCfg(1,1,"gpt-5.6-luna","max")], Bounds(400,43200))
    calls={"n":0}
    hooks={"is_exhausted": lambda *a,**k: calls["n"]>=2,   # exhausted after 2 rounds
           "run_round": lambda *a,**k: calls.__setitem__("n",calls["n"]+1) or 0,
           "update_exhaustion": lambda *a,**k: None, "advance": lambda *a,**k: True}
    rounds = run_loop(cfg, cfg.workers[0], safety_cap=100, hooks=hooks)
    assert rounds == 2                      # stops when is_exhausted flips true
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**: `run_loop` = the `driver_v3` loop: while `not is_exhausted` and `round<safety_cap`: build round prompt (`gen_digest`+prompt), `run_round`, `update_exhaustion`, advance baseline on WIN via `update_ref_cas`. `hooks` override each seam for tests.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(ar): driver.py — self-exhausting per-worker loop (driver_v3 port)`.

---

## Phase 5 — Swarm (+ live acceptance)

### Task 5.1: `ar/swarm.py` — config-driven worker launcher

**Files:**
- Create: `autoresearch/ar/swarm.py`
- Port from: `autoresearch/harness/swarm_explore.sh` (replacing the `sed`-prompt-munge with the config workers list)
- Test: `autoresearch/ar/tests/test_swarm.py`

**Interfaces:**
- Consumes: `LoopConfig` (with `workers`), `driver.run_loop`, `gitpilot.worktree_*`.
- Produces: `plan_workers(cfg, repo)->list[dict]` (per-worker `{card,dev,model,effort,worktree,anchor,lockfile}`); `launch(cfg, repo, *, spawn=None)->list[int]` (pids; `spawn` injectable).

- [ ] **Step 1: Write the failing test**
```python
from autoresearch.ar.swarm import plan_workers
from autoresearch.ar.config import LoopConfig, WorkerCfg, Bounds
def test_per_worker_heterogeneity_no_sed():
    cfg = LoopConfig("gfx1201","loop/gfx1201","m","q8",128,"md5",3.0,5,"codex",
        [WorkerCfg(1,1,"gpt-5.6-luna","max"), WorkerCfg(2,2,"gpt-5.6-terra","xhigh"), WorkerCfg(3,3,"gpt-5.6-sol","medium")],
        Bounds(400,43200))
    plans = plan_workers(cfg, "/repo")
    assert [p["model"] for p in plans]==["gpt-5.6-luna","gpt-5.6-terra","gpt-5.6-sol"]
    assert [p["effort"] for p in plans]==["max","xhigh","medium"]
    assert plans[0]["lockfile"].endswith("gfx1201-dev1.lock") and plans[2]["worktree"].endswith("sw_card3")
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**: `plan_workers` maps each `WorkerCfg`→ its worktree `.aw/sw_card<card>`, anchor branch `loop/<arch>_w<i>`, per-dev lockfile; `launch` spawns `run_loop` per worker (`setsid`/detached), returns pids. No prompt `sed`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(ar): swarm.py — config-driven worker launcher (kills the sed-munge)`.

### Task 5.2 (LIVE acceptance): Sol/Terra/Luna eval on mq4r/gfx1201

**Files:** Runbook: `docs/autoresearch/operations.md` (§ "first run"); no new code.

- [ ] **Step 1:** On hiptrx, drive the Phase-1/3 modules directly (the `ar` CLI wrapping them lands in Phase 7): ingest the ledger into `ar.db` and confirm `bod_gfx1201.json` (reuse the already-regenerated fd6deaa9 BOD). Verify worker card→dev mapping per-fleet first (rocminfo lies; confirm via the daemon-reported arch), and set all 4 R9700 to AUTO clock.
- [ ] **Step 2:** Launch via the swarm module directly: `python -c "from autoresearch.ar.config import load_config; from autoresearch.ar.swarm import launch; launch(load_config('autoresearch/config/loop_gfx1201.toml'), '.')"` (3 workers: Luna@max card1, Terra@xhigh card2, Sol@medium card3; `model=mq4r`; `k_exhaust=5`).
- [ ] **Step 3: Verify (acceptance):** all 3 workers produce ledger rows that are self-describing (`measurement_hash` present, `gpu_arch=gfx1201`, `model=…mq4r`) with BOTH `tok_delta_pct` and `dur_delta_pct`; any WIN folds into the shared baseline via CAS; no worker corrupts another arch (cross-arch guard clean).
- [ ] **Step 4: Commit** the run's ledger rows + a short results note in `operations.md`.

---

## Phase 6 — Watcher daemon

### Task 6.1: `ar/watcher.py` — run tracking + guardrailed auto-enforce

**Files:**
- Create: `autoresearch/ar/watcher.py`
- Port from: `autoresearch/ar/hipfire_ar.py` (`cmd_start/stop/status`, run table, budget/TTL), `v2/rollover_v2.sh` (fold/rollover)
- Test: `autoresearch/ar/tests/test_watcher.py`

**Interfaces:**
- Consumes: `db.runs`, `gitpilot.{update_ref_cas,fold,rollover}`, `Bounds`.
- Produces: `class Watcher(db, repo)` with `track(run)`, `reap()`, `enforce()`; `enforce()` auto-folds WINs + auto-rolls-over on advance/exhaustion, each **dry-run-logged + prior-SHA-recorded**; auto-stops over-budget/TTL runs. Never master-pushes.

- [ ] **Step 1: Write the failing test**
```python
from autoresearch.ar.watcher import Watcher
def test_over_budget_run_autostops(fake_db, tmp_repo):
    w=Watcher(fake_db, tmp_repo)
    fake_db.add_run(id="r1", calls=401, budget=400, ttl=99999, pid=None, status="running")
    w.enforce(); assert fake_db.get_run("r1")["status"]=="stopped"
def test_win_autofold_is_dryrun_logged_and_reversible(fake_db, tmp_repo):
    w=Watcher(fake_db, tmp_repo); prior=w.enforce_fold("loop/gfx1201","<win_sha>", dry_run=True)
    assert prior["prior_sha"] and prior["dry_run"] is True   # records reversibility, no mutation
def test_master_push_never_automated(fake_db, tmp_repo):
    w=Watcher(fake_db, tmp_repo)
    assert not hasattr(w, "push_master")   # capability absent by construction
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**: `Watcher` reaps dead pids, auto-stops runs past `call_budget`/`wall_ttl_s`, folds WINs into the shared baseline (CAS, prior-SHA recorded, dry-run first), triggers re-census+re-ingest on advance/exhaustion. No master-push/default-flip methods exist.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(ar): watcher.py — run tracking + guardrailed auto-fold/rollover + leashes`.

---

## Phase 7 — CLI + skill

### Task 7.1: `ar/cli.py` — role-scoped `ar` entrypoint

**Files:**
- Create: `autoresearch/ar/cli.py`, `autoresearch/ar/__main__.py` (`from .cli import main; main()`)
- Port from: `autoresearch/ar/hipfire_ar.py` (`main` + `cmd_*`)
- Test: `autoresearch/ar/tests/test_cli.py`

**Interfaces:**
- Consumes: all modules above.
- Produces: `main(argv)->int`; operator verbs `start/stop/status/why/bod/ingest/fold/rollover/config`; agent verbs `why/status/bod/certify`. Agent `certify` on EXHAUSTED/off-target/over-budget → exit 3.

- [ ] **Step 1: Write the failing test**
```python
from autoresearch.ar.cli import main
def test_agent_certify_on_exhausted_exits_3(monkeypatch, tmp_ar):
    # kernel marked EXHAUSTED in fixture db
    rc = main(["--role","agent","certify","--arch","gfx1201","--kernel","gate_up_exhausted","--lever","L","--variant","/tmp/v.hip"])
    assert rc == 3
def test_agent_cannot_start():
    rc = main(["--role","agent","start","--config","x.toml"])
    assert rc == 3        # start is operator-only
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**: argparse with `--role {operator,agent}`; dispatch to `cmd_*`; mechanical bounds refusal (exit 3) for agent verbs on exhausted/off-target/over-budget (reuse `candidates`/`db`). Port `hipfire_ar.py:main`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(ar): ar CLI — role-scoped operator/agent entrypoint with mechanical bounds`.

### Task 7.2: agent-facing skill doc

**Files:** Create: `docs/skills/autoresearch-loop.md`; Modify: `CLAUDE.md` (skills index — one line).

- [ ] **Step 1:** Write `docs/skills/autoresearch-loop.md`: when to reach for it (ssh-in / on-device loop interaction), the `ar` verbs, the contract "read `ar bod`/`ar why` before authoring a lever; submit via `ar certify`; never touch a raw script", and the role model.
- [ ] **Step 2:** Add the one-line index entry to CLAUDE.md's skills list.
- [ ] **Step 3: Commit** `--no-verify` (doc-only) `docs(skills): autoresearch-loop agent skill + CLAUDE.md index`.

---

## Phase 8 — Cleanup + docs

### Task 8.1: delete bash, move to target layout

**Files:** Delete the 38 `.sh` under `autoresearch/`; move probe `.hip` → `autoresearch/probes/`; move loose design `.md` → `docs/autoresearch/`; ensure `autoresearch/db/` (gitignore `ar.db`), `autoresearch/config/`.

- [ ] **Step 1:** `git rm` the 38 `.sh` (list from spec §3) + `autoresearch/harness/` remnants; `git mv` probes + loose docs.
- [ ] **Step 2: Verify**: `rg -n '\.sh' autoresearch/` returns only references inside probe sources/docs; `./scripts/no-gpu-ci.sh` green (the new `ar/tests/` run under it).
- [ ] **Step 3: Commit** `refactor(ar): delete 38 bash scripts; move to clean autoresearch/ layout`.

### Task 8.2: rigorous docs

**Files:** Create: `docs/autoresearch/architecture.md`, `operations.md`, `config-reference.md`.

- [ ] **Step 1:** `architecture.md` (the module map + data flow), `operations.md` (start/stop/status/fold/rollover runbook + the Phase-5 first-run results), `config-reference.md` (every TOML key + default).
- [ ] **Step 2: Commit** `--no-verify` `docs(autoresearch): architecture + operations + config reference`.

---

## Coverage note

Spec §-by-§ → task: §3 reuse-map → Tasks 1.2/2.1-2.4/3.x/4.x/6.1/7.1; §4 architecture → all; §5 layout → 8.1; §6 config → 1.1; §7 gitpilot → 1.3; §8 certify (conjunctive perf) → 2.1-2.3; §9 watcher → 6.1; §10 CLI+skill → 7.1/7.2; §11 strangler → per-task "port then delete" + 2.5; §12 phases → Phases 1-8; §13 workflow → the multi-agent Workflow that executes this plan (header note; authored after plan approval, one workflow phase per plan phase); §14 testing → every task's test + 5.2 live. No orphaned spec requirements.
