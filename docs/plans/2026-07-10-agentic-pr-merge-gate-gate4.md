# Agentic PR Merge-Gate — Gate 4 (non-clobber merge) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the no-GPU-testable core of **Gate 4** (spec §10): a trial-merge of
the PR into the staging tip, a re-gate of the *merged* tree, a codex merge-fix
retry, and a Bill of Debt (BOD) when it can't be made clean.

**Architecture:** A new `autoresearch/ar/gate/merge.py` module. `trial_merge`
uses `git merge-tree --write-tree` (no working-tree touch) to detect
conflicts/produce a merged tree. `gate4` orchestrates merge → (optional
codex merge-fix) → re-gate on the merged tree → (optional merge-fix) → PASS/BOD,
with every side-effecting piece (git, the merged-tree gate, the codex fixer)
injected as a seam so the whole flow is unit-testable with no GPU and no real git.

**Tech Stack:** Python 3.11+ (stdlib `subprocess`); pytest; no third-party deps.
Reuses the Phase-1 `run_gate` (as an injected `run_merged_gate` thunk) and the
`agent_exec` codex seam (as an injected `merge_fix`).

## Global Constraints

- **No-GPU unit-testable** — every side-effect (git, the merged-tree gate, the
  codex fixer) is an injected seam; tests use mocks. Runs under
  `scripts/no-gpu-ci.sh`.
- **Trial merge never touches the working tree** — use `git merge-tree
  --write-tree <base> <head>` (git ≥ 2.38; the repo has 2.43): exit 0 = clean +
  merged-tree OID on stdout line 1; exit 1 = conflicts (conflicted paths in the
  informational output).
- **Merge-fix is in-repo only** — spec §10: codex merge-fix pushes only to
  in-repo PR branches; a fork clobber goes straight to BOD. This plan's `gate4`
  takes `merge_fix=None` to model "no fixer available" (fork) → BOD.
- **BOD = the itemized blockers** — `{blockers: [{kind, detail}], summary}` with
  `kind ∈ {"merge_conflict", "perf_regression", "coherence"}`.
- **New files** carry `# Copyright (c) Kaden Schutt` as the first line.
- Commit messages end with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **Out of scope (later phases):** the real `run_merged_gate` wiring over
  `LiveServeRunner` + daemon builds (Phase 3), the actual codex `sol xhigh`
  merge-fix invocation via `agent_exec` (Phase 3), the staging-train fold/land
  (Phase 5). This plan stops at the injected-seam flow + its BOD.

## File Structure

- Create `autoresearch/ar/gate/merge.py` — `trial_merge`, `assemble_bod`, `gate4`.
- Modify `autoresearch/ar/gate/__init__.py` — re-export the three.
- Create `autoresearch/ar/tests/test_gate_merge.py` — the unit tests.

---

### Task 1: `trial_merge` (merge-tree conflict detection, injectable git)

**Files:**
- Create: `autoresearch/ar/gate/merge.py`
- Test: `autoresearch/ar/tests/test_gate_merge.py`

**Interfaces:**
- Consumes: stdlib `subprocess`.
- Produces:
  - `default_run_git(repo: str, *args: str) -> tuple[int, str]` — runs `git -C repo <args>`, returns `(returncode, stdout)`.
  - `trial_merge(base_ref: str, head_ref: str, repo: str, run_git=None) -> dict`
    returning `{"clean": bool, "merged_tree": str | None, "conflicts": list[str]}`.
    Uses `git merge-tree --write-tree --name-only <base_ref> <head_ref>`: on a
    clean merge (rc 0) stdout line 1 is the tree OID; on conflicts (rc 1) the
    lines after the first blank line are the conflicted paths.

- [ ] **Step 1: Write the failing test**

Create `autoresearch/ar/tests/test_gate_merge.py`:

```python
# Copyright (c) Kaden Schutt
from autoresearch.ar.gate.merge import trial_merge


def _git_clean(repo, *args):
    # `git merge-tree --write-tree ...` clean: rc 0, tree OID on line 1
    return (0, "a1b2c3d4e5f6\n")


def _git_conflict(repo, *args):
    # rc 1, tree OID line, blank line, then conflicted paths (--name-only)
    return (1, "deadbeef\n\ncrates/hipfire-runtime/examples/daemon.rs\n")


def test_clean_merge():
    r = trial_merge("staging", "pr", "/repo", run_git=_git_clean)
    assert r["clean"] is True
    assert r["merged_tree"] == "a1b2c3d4e5f6"
    assert r["conflicts"] == []


def test_conflicted_merge_lists_paths():
    r = trial_merge("staging", "pr", "/repo", run_git=_git_conflict)
    assert r["clean"] is False
    assert r["conflicts"] == ["crates/hipfire-runtime/examples/daemon.rs"]


def test_passes_refs_to_git():
    seen = {}

    def spy(repo, *args):
        seen["repo"] = repo
        seen["args"] = args
        return (0, "abc\n")

    trial_merge("staging", "pr", "/repo", run_git=spy)
    assert seen["repo"] == "/repo"
    assert "merge-tree" in seen["args"]
    assert seen["args"][-2:] == ("staging", "pr")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_merge.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'autoresearch.ar.gate.merge'`.
(Note: system `python`/`python3` lack pytest; use the project venv
`/home/kaden/.venvs/hipfire-pytest/bin/python`, the same interpreter
`scripts/no-gpu-ci.sh` drives.)

- [ ] **Step 3: Write `trial_merge`**

Create `autoresearch/ar/gate/merge.py`:

```python
# Copyright (c) Kaden Schutt
"""ar.gate.merge — Gate 4: non-clobber merge test (spec §10).

trial_merge does a `git merge-tree --write-tree` merge of the PR into the staging
tip WITHOUT touching the working tree, reporting conflicts. gate4 orchestrates
merge -> (codex merge-fix) -> re-gate the merged tree -> (merge-fix) -> PASS/BOD.
Every side-effect (git, the merged-tree gate, the codex fixer) is an injected
seam, so the flow is unit-testable with no GPU and no real git.
"""
from __future__ import annotations

import subprocess


def default_run_git(repo: str, *args: str) -> tuple[int, str]:
    """Run `git -C repo <args>`; return (returncode, stdout). stderr is folded in
    so a caller can surface it, but the parse only needs stdout."""
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return p.returncode, p.stdout


def trial_merge(base_ref: str, head_ref: str, repo: str, run_git=None) -> dict:
    """Trial-merge head_ref into base_ref via `git merge-tree --write-tree`
    (no working-tree touch). Returns {clean, merged_tree, conflicts}."""
    run = run_git or default_run_git
    rc, out = run(repo, "merge-tree", "--write-tree", "--name-only", base_ref, head_ref)
    lines = out.splitlines()
    merged_tree = lines[0].strip() if lines else None
    if rc == 0:
        return {"clean": True, "merged_tree": merged_tree, "conflicts": []}
    # rc != 0: conflicts. Format is <tree>\n\n<conflicted path>*  — take the
    # non-empty lines after the first blank separator.
    conflicts: list[str] = []
    seen_blank = False
    for ln in lines[1:]:
        if not ln.strip():
            seen_blank = True
            continue
        if seen_blank:
            conflicts.append(ln.strip())
    return {"clean": False, "merged_tree": merged_tree, "conflicts": conflicts}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_merge.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add autoresearch/ar/gate/merge.py autoresearch/ar/tests/test_gate_merge.py
git commit -m "feat(ar-gate): trial_merge — merge-tree conflict detection (Gate 4)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `assemble_bod` (the Bill of Debt)

**Files:**
- Modify: `autoresearch/ar/gate/merge.py` (append `assemble_bod`)
- Test: `autoresearch/ar/tests/test_gate_merge.py` (append)

**Interfaces:**
- Produces:
  - `assemble_bod(*, conflicts=None, perf_regressions=None, coherence_fails=None) -> dict`
    returning `{"blockers": [{"kind": str, "detail": str}], "summary": str}` with
    `kind ∈ {"merge_conflict", "perf_regression", "coherence"}`. Empty inputs →
    `{"blockers": [], "summary": "no blockers"}`.

- [ ] **Step 1: Write the failing test (append to `test_gate_merge.py`)**

```python
from autoresearch.ar.gate.merge import assemble_bod


def test_bod_collects_all_kinds():
    bod = assemble_bod(
        conflicts=["daemon.rs"],
        perf_regressions=["perf_regression"],
        coherence_fails=["coherence"],
    )
    kinds = [b["kind"] for b in bod["blockers"]]
    assert kinds == ["merge_conflict", "perf_regression", "coherence"]
    assert bod["blockers"][0]["detail"] == "daemon.rs"
    assert "3" in bod["summary"]


def test_bod_empty_is_clean():
    bod = assemble_bod()
    assert bod["blockers"] == []
    assert bod["summary"] == "no blockers"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_merge.py -q`
Expected: FAIL — `ImportError: cannot import name 'assemble_bod'`.

- [ ] **Step 3: Append `assemble_bod` to `merge.py`**

```python
def assemble_bod(*, conflicts=None, perf_regressions=None, coherence_fails=None) -> dict:
    """Assemble the Bill of Debt — the itemized blockers a PR must clear (spec §10)."""
    blockers: list[dict] = []
    for c in conflicts or []:
        blockers.append({"kind": "merge_conflict", "detail": c})
    for r in perf_regressions or []:
        blockers.append({"kind": "perf_regression", "detail": r})
    for c in coherence_fails or []:
        blockers.append({"kind": "coherence", "detail": c})
    summary = f"{len(blockers)} blocker(s)" if blockers else "no blockers"
    return {"blockers": blockers, "summary": summary}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_merge.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add autoresearch/ar/gate/merge.py autoresearch/ar/tests/test_gate_merge.py
git commit -m "feat(ar-gate): assemble_bod — itemized Bill of Debt (Gate 4)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `gate4` — the decision flow (merge → fix → re-gate → fix → PASS/BOD)

**Files:**
- Modify: `autoresearch/ar/gate/merge.py` (append `gate4`)
- Modify: `autoresearch/ar/gate/__init__.py` (re-export `trial_merge`, `assemble_bod`, `gate4`)
- Test: `autoresearch/ar/tests/test_gate_merge.py` (append)

**Interfaces:**
- Consumes: `trial_merge` (Task 1), `assemble_bod` (Task 2).
- Produces:
  - `gate4(*, base_ref, head_ref, staging_ref, repo, run_merged_gate, merge_fix=None, trial_merge_fn=None) -> dict`
    returning `{"verdict": "PASS" | "BOD", "bod": dict | None, "merged": dict, "gate": dict | None}`.
  - `run_merged_gate()` — a zero-arg thunk the caller binds to a Phase-1
    `run_gate` over the *merged* tree vs staging; returns a run_gate-style dict
    `{"verdict": "PASS"|"REJECT", "reasons": [str], ...}`. Injected (may be
    stateful so a post-fix call can differ).
  - `merge_fix(kind: str, detail) -> dict` — the codex seam; returns
    `{"fixed": bool}`. `None` = no fixer available (fork PR) → any blocker is BOD.
  - `trial_merge_fn(staging_ref, head_ref, repo) -> dict` — injected trial-merge
    (defaults to `trial_merge`).

Flow: (1) trial-merge the PR into the staging tip; on conflicts, try `merge_fix`
once and re-trial — still conflicted → BOD. (2) re-gate the merged tree; PASS →
done. (3) on a post-merge clobber (REJECT), try `merge_fix` once and re-gate —
still REJECT → BOD, partitioning the reasons into perf vs coherence blockers.

- [ ] **Step 1: Write the failing test (append to `test_gate_merge.py`)**

```python
from autoresearch.ar.gate.merge import gate4


def _clean_tm(*a, **k):
    return {"clean": True, "merged_tree": "t", "conflicts": []}


def _conflict_tm(*a, **k):
    return {"clean": False, "merged_tree": "t", "conflicts": ["daemon.rs"]}


def _gate(verdict, reasons=()):
    return lambda: {"verdict": verdict, "reasons": list(reasons)}


def test_clean_merge_clean_gate_passes():
    r = gate4(base_ref="m", head_ref="pr", staging_ref="staging", repo="/repo",
              run_merged_gate=_gate("PASS"), trial_merge_fn=_clean_tm)
    assert r["verdict"] == "PASS" and r["bod"] is None


def test_conflict_no_fixer_is_bod():
    r = gate4(base_ref="m", head_ref="pr", staging_ref="staging", repo="/repo",
              run_merged_gate=_gate("PASS"), merge_fix=None, trial_merge_fn=_conflict_tm)
    assert r["verdict"] == "BOD"
    assert r["bod"]["blockers"][0]["kind"] == "merge_conflict"


def test_conflict_fixed_then_passes():
    # trial-merge conflicts first, then (after fix) is clean
    seq = [_conflict_tm(), _clean_tm()]
    tm = lambda *a, **k: seq.pop(0)
    r = gate4(base_ref="m", head_ref="pr", staging_ref="staging", repo="/repo",
              run_merged_gate=_gate("PASS"),
              merge_fix=lambda kind, detail: {"fixed": True}, trial_merge_fn=tm)
    assert r["verdict"] == "PASS"


def test_post_merge_clobber_no_fixer_is_bod_partitioned():
    r = gate4(base_ref="m", head_ref="pr", staging_ref="staging", repo="/repo",
              run_merged_gate=_gate("REJECT", ["perf_regression", "coherence"]),
              merge_fix=None, trial_merge_fn=_clean_tm)
    assert r["verdict"] == "BOD"
    kinds = sorted(b["kind"] for b in r["bod"]["blockers"])
    assert kinds == ["coherence", "perf_regression"]


def test_post_merge_clobber_fixed_then_passes():
    gates = [{"verdict": "REJECT", "reasons": ["perf_regression"]},
             {"verdict": "PASS", "reasons": []}]
    r = gate4(base_ref="m", head_ref="pr", staging_ref="staging", repo="/repo",
              run_merged_gate=lambda: gates.pop(0),
              merge_fix=lambda kind, detail: {"fixed": True}, trial_merge_fn=_clean_tm)
    assert r["verdict"] == "PASS"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_merge.py -q`
Expected: FAIL — `ImportError: cannot import name 'gate4'`.

- [ ] **Step 3: Append `gate4` to `merge.py`**

```python
def gate4(*, base_ref, head_ref, staging_ref, repo, run_merged_gate,
          merge_fix=None, trial_merge_fn=None) -> dict:
    """Gate 4 (spec §10): non-clobber merge test with a codex merge-fix retry.

    merge -> (fix conflict, re-trial) -> re-gate merged tree -> (fix clobber,
    re-gate) -> PASS or BOD. All side-effects are injected seams."""
    tmf = trial_merge_fn or (lambda b, h, r: trial_merge(b, h, r))

    # 1. Trial-merge the PR into the staging tip.
    tm = tmf(staging_ref, head_ref, repo)
    if not tm["clean"]:
        if merge_fix is None or not merge_fix("merge_conflict", tm["conflicts"]).get("fixed"):
            return {"verdict": "BOD", "gate": None, "merged": tm,
                    "bod": assemble_bod(conflicts=tm["conflicts"])}
        tm = tmf(staging_ref, head_ref, repo)                       # re-trial after fix
        if not tm["clean"]:
            return {"verdict": "BOD", "gate": None, "merged": tm,
                    "bod": assemble_bod(conflicts=tm["conflicts"])}

    # 2. Re-gate the MERGED tree (does the merge clobber?).
    g = run_merged_gate()
    if g["verdict"] == "PASS":
        return {"verdict": "PASS", "bod": None, "merged": tm, "gate": g}

    # 3. Post-merge clobber: try one codex merge-fix, then re-gate.
    if merge_fix is not None and merge_fix("clobber", g.get("reasons", [])).get("fixed"):
        g = run_merged_gate()
        if g["verdict"] == "PASS":
            return {"verdict": "PASS", "bod": None, "merged": tm, "gate": g}

    reasons = g.get("reasons", [])
    bod = assemble_bod(
        perf_regressions=[r for r in reasons if "perf" in r],
        coherence_fails=[r for r in reasons if "coher" in r],
    )
    return {"verdict": "BOD", "bod": bod, "merged": tm, "gate": g}
```

Update `autoresearch/ar/gate/__init__.py` to re-export:

```python
# Copyright (c) Kaden Schutt
"""ar.gate — the Tier-3 PR merge-gate engine (no-GPU-testable core)."""
from .config import GateConfig, load_gate_config
from .engine import gate_cell, run_gate
from .merge import assemble_bod, gate4, trial_merge

__all__ = [
    "GateConfig", "load_gate_config", "gate_cell", "run_gate",
    "trial_merge", "assemble_bod", "gate4",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_merge.py -q`
Expected: PASS (10 tests: 3 + 2 + 5).

- [ ] **Step 5: Full gate suite (no regressions)**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/ -q -k "gate"`
Expected: PASS (32 = the 22 Phase-1 gate tests + 10 Gate-4 tests), no regressions.
(The 2 pre-existing `test_config`/`test_cli` worker-roster failures are unrelated
to `-k "gate"` and are out of scope for this plan.)

- [ ] **Step 6: Commit**

```bash
git add autoresearch/ar/gate/merge.py autoresearch/ar/gate/__init__.py \
        autoresearch/ar/tests/test_gate_merge.py
git commit -m "feat(ar-gate): gate4 — non-clobber merge flow + merge-fix retry + BOD (Gate 4)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (this plan = Phase 2 / spec §10):**
- Trial merge into the staging tip, no working-tree touch → Task 1 (`trial_merge`
  via `merge-tree --write-tree`). ✔
- Re-gate the merged tree (post-merge clobber detection) → Task 3 (`gate4` step 2,
  `run_merged_gate` seam bound to Phase-1 `run_gate`). ✔
- codex merge-fix retry → Task 3 (`merge_fix` seam, one retry for conflict + one
  for clobber). ✔
- Fork PR (no fixer) → BOD → Task 3 (`merge_fix=None` path). ✔
- Bill of Debt itemization → Task 2 (`assemble_bod`, three kinds). ✔
- **Deferred (later phases):** real `run_merged_gate`/`merge_fix` wiring over GPU +
  codex `sol xhigh` (Phase 3), staging fold/land (Phase 5).

**Placeholder scan:** no TBD/TODO; every code step is complete; every test has real
assertions + exact venv run command. ✔

**Type consistency:** `trial_merge` returns `{clean, merged_tree, conflicts}` used
verbatim by `gate4`; `assemble_bod`'s `{blockers:[{kind,detail}], summary}` is what
`gate4` returns under `bod`; `run_merged_gate` returns the Phase-1 `run_gate`
`{verdict, reasons}` shape; `merge_fix` returns `{fixed: bool}` consistently. ✔

## Next plans

- **Phase 3** — `gpu-gates.yml` + `claude-review.yml` dispatch/interpret/merge +
  §8.1 model routing + `LiveServeRunner` wiring for `run_merged_gate`/`merge_fix`.
- **Phase 4** — perf governance (high-water B + drift guard + ledger).
- **Phase 5** — staging merge-train + freshness sync.
