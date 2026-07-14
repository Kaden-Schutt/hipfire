# Agentic PR Merge-Gate — Staging Merge-Train (Phase 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the no-GPU-testable core of the **staging merge-train** (spec §11 +
the §10 post-merge recall-reproduce): stack gate-approved PRs onto a derived
`staging`, resolving conflicts via Gate-4 merge-fix instead of punting, validating
each fold by *recalling recorded behaviors* (not re-running), and landing the whole
train to master in one non-clobber merge with close-behind bookkeeping.

**Architecture:** A new `autoresearch/ar/gate/staging.py`. Every side-effecting piece
— git, the codex merge-fix, the recall-reproduce check — is an **injected seam**, so
the whole train logic is unit-testable with no GPU and no real git. It reuses
`merge.trial_merge` (the non-clobber check) and mirrors `gate4`'s resolve→re-validate
→BOD flow at the stack level.

**Tech Stack:** Python 3.11+ (stdlib); pytest via `/home/kaden/.venvs/hipfire-pytest/bin/python`.
Reuses `autoresearch/ar/gate/merge.py` (`trial_merge`, `default_run_git`).

> **Correction (post-adversarial-review, fix commit `7db68173`).** Task 3's `fold_pr`
> code below is WRONG: it dispatches `merge_fix_fn` only for a *textual* conflict, so a
> **semantic clobber** (clean merge, but a recorded behavior fails to reproduce — the
> exact §10 case) BODs immediately with no fix attempt. Correct form: a single `_try(stg)`
> returns `("FOLDED", tree) | ("clobber", detail) | ("conflict", None)`; **both** `clobber`
> and `conflict` attempt `merge_fix_fn` then re-`_try`; BOD only if still unresolved
> (`reason="clobber"` for a semantic clobber, else `classify_conflict`). Add a test where a
> semantic clobber + a resolving fixer → FOLDED (the `merge_fix_fn=None` test can't catch it).

## Global Constraints

- **No-GPU unit-testable** — git / codex merge-fix / recall-reproduce are injected
  seams; tests use mocks. Runs under `scripts/no-gpu-ci.sh`.
- **Resolve, don't punt (spec §11):** a clobbering fold goes through the merge-fix
  seam (resolve on the agent-owned staging) and re-validates; BOD only if the fix
  fails. Debt is split: `"stale"` (conflicts vs master → rebase) vs `"stack"` (clean
  vs master, conflicts with an already-folded PR).
- **Post-merge check is RECALL-based (spec §10):** the validation seam re-confirms
  the PR's *already-recorded* behaviors REPRODUCE on the merged tree — it does NOT
  re-run the full PR gate or re-measure master. A recorded behavior that fails to
  reproduce = a semantic clobber.
- **New files** carry `# Copyright (c) Kaden Schutt` as line 1.
- Commit messages end with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **Out of scope (prod wiring, not this plan):** the real git rebase/merge, the real
  codex merge-fix, the real GPU recall-reproduce, and the GitHub close-behind API —
  all injected here as seams and validated on the first live run.

## File Structure

- Create `autoresearch/ar/gate/staging.py` — `classify_conflict`, `recall_reproduce`,
  `fold_pr`, `stack_train`, `land_train`.
- Modify `autoresearch/ar/gate/__init__.py` — re-export them.
- Create `autoresearch/ar/tests/test_gate_staging.py` — the unit tests.

Shared seam signatures (used across tasks — keep them exact):
- `git(repo, *args) -> tuple[int, str]` — like `merge.default_run_git`.
- `trial_merge_fn(base_ref, head_ref, repo) -> {"clean": bool, "merged_tree": str, "conflicts": [str]}` — `merge.trial_merge`.
- `merge_fix_fn(pr_ref, staging_ref, repo) -> {"resolved": bool, "staging_ref": str}` — codex resolves on staging; `staging_ref` is the new tip after a successful fix.
- `reproduce_fn(pr_ref, merged_ref, recorded, repo) -> {"reproduced": bool, "failures": [str]}` — recall-reproduce the PR's recorded behaviors on the merged tree.

---

### Task 1: `classify_conflict` — the debt split (stale vs stack)

**Files:** Create `autoresearch/ar/gate/staging.py`; Test `autoresearch/ar/tests/test_gate_staging.py`.

**Interfaces:**
- Produces `classify_conflict(pr_ref, master_ref, repo, *, trial_merge_fn) -> str`
  returning `"stale"` (the PR conflicts with master itself → rebase) or `"stack"`
  (clean vs master, so the conflict is with an already-folded PR).

- [ ] **Step 1: Write the failing test**

```python
# Copyright (c) Kaden Schutt
from autoresearch.ar.gate.staging import classify_conflict


def _tm(clean):
    return lambda base, head, repo: {"clean": clean, "merged_tree": "t", "conflicts": [] if clean else ["f"]}


def test_conflict_vs_master_is_stale():
    assert classify_conflict("pr", "master", "/r", trial_merge_fn=_tm(False)) == "stale"


def test_clean_vs_master_is_stack_conflict():
    assert classify_conflict("pr", "master", "/r", trial_merge_fn=_tm(True)) == "stack"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_staging.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'autoresearch.ar.gate.staging'`.

- [ ] **Step 3: Write `classify_conflict`**

```python
# Copyright (c) Kaden Schutt
"""ar.gate.staging — the staging merge-train (spec §11 + §10 recall-reproduce).

Stack gate-approved PRs onto a derived `staging`, resolving conflicts via the
Gate-4 merge-fix instead of punting, validating each fold by RECALLING the PR's
already-recorded behaviors (not re-running), and landing the whole train to master
in one non-clobber merge. Every side-effect (git, codex merge-fix, recall-reproduce)
is an injected seam, so this is unit-testable with no GPU and no real git.
"""
from __future__ import annotations

from .merge import trial_merge as _trial_merge

__all__ = ["classify_conflict", "recall_reproduce", "fold_pr", "stack_train", "land_train"]


def classify_conflict(pr_ref, master_ref, repo, *, trial_merge_fn=None) -> str:
    """Split a fold conflict: 'stale' (conflicts with master itself -> rebase) vs
    'stack' (clean vs master, so it conflicts with an already-folded PR)."""
    tm = trial_merge_fn or (lambda b, h, r: _trial_merge(b, h, r))
    return "stack" if tm(master_ref, pr_ref, repo)["clean"] else "stale"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_staging.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add autoresearch/ar/gate/staging.py autoresearch/ar/tests/test_gate_staging.py
git commit -m "feat(ar-gate): classify_conflict — staging debt split (stale vs stack)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `recall_reproduce` — post-merge recorded-behavior reproduction (§10)

**Files:** Modify `staging.py`; Test append.

**Interfaces:**
- Produces `recall_reproduce(pr_ref, merged_ref, recorded, repo, *, reproduce_fn) -> dict`
  returning `{"reproduced": bool, "failures": [str]}`. It does NOT re-run the PR gate
  or re-measure master — it delegates to `reproduce_fn` (which re-runs only the PR's
  `recorded` behaviors on the merged tree) and returns its result verbatim. An empty
  `recorded` list is a trivial reproduce (`reproduced=True`).

- [ ] **Step 1: Write the failing test (append)**

```python
from autoresearch.ar.gate.staging import recall_reproduce


def test_recall_reproduce_delegates_and_passes():
    rf = lambda pr, merged, rec, repo: {"reproduced": True, "failures": []}
    out = recall_reproduce("pr", "merged", ["parity", "coh"], "/r", reproduce_fn=rf)
    assert out["reproduced"] is True


def test_recall_reproduce_reports_failures():
    rf = lambda pr, merged, rec, repo: {"reproduced": False, "failures": ["behavior:cli"]}
    out = recall_reproduce("pr", "merged", ["cli"], "/r", reproduce_fn=rf)
    assert out["reproduced"] is False and out["failures"] == ["behavior:cli"]


def test_recall_reproduce_empty_recorded_is_trivially_reproduced():
    called = {"n": 0}

    def rf(pr, merged, rec, repo):
        called["n"] += 1
        return {"reproduced": True, "failures": []}

    out = recall_reproduce("pr", "merged", [], "/r", reproduce_fn=rf)
    assert out["reproduced"] is True and called["n"] == 0   # nothing to reproduce -> no call
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_staging.py -q`
Expected: FAIL — `ImportError: cannot import name 'recall_reproduce'`.

- [ ] **Step 3: Append `recall_reproduce` to `staging.py`**

```python
def recall_reproduce(pr_ref, merged_ref, recorded, repo, *, reproduce_fn) -> dict:
    """Confirm the PR's ALREADY-RECORDED behaviors REPRODUCE on the merged tree
    (spec §10). Does not re-run the full PR gate or re-measure master — delegates to
    reproduce_fn, which re-runs only ``recorded`` on ``merged_ref``. Empty recorded
    -> trivially reproduced (no call)."""
    if not recorded:
        return {"reproduced": True, "failures": []}
    r = reproduce_fn(pr_ref, merged_ref, recorded, repo)
    return {"reproduced": bool(r.get("reproduced")), "failures": list(r.get("failures", []))}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_staging.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add autoresearch/ar/gate/staging.py autoresearch/ar/tests/test_gate_staging.py
git commit -m "feat(ar-gate): recall_reproduce — post-merge recorded-behavior check (spec 10)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `fold_pr` — fold one PR onto staging (trial → resolve → recall-reproduce → FOLDED/BOD)

**Files:** Modify `staging.py`; Test append.

**Interfaces:**
- Produces `fold_pr(*, pr_ref, staging_ref, master_ref, repo, recorded, trial_merge_fn, merge_fix_fn, reproduce_fn) -> dict`
  returning `{"pr": pr_ref, "verdict": "FOLDED"|"BOD", "staging_ref": str, "reason": str, "detail": str}`.

Flow (mirrors `gate4` at the fold level): (1) trial-merge `pr_ref` onto `staging_ref`.
If clean → recall-reproduce on the merged tree; reproduced → **FOLDED** (advance
`staging_ref` to the merged tree), else **BOD** `reason="clobber"`. (2) If the trial
conflicts → `merge_fix_fn` (resolve on staging). If `resolved` → re-trial on the new
`staging_ref`, then recall-reproduce → FOLDED/BOD. (3) If the fix fails → **BOD** with
`reason=classify_conflict(...)` (`"stale"`|`"stack"`) so the message is actionable.

- [ ] **Step 1: Write the failing test (append)**

```python
from autoresearch.ar.gate.staging import fold_pr


def _clean_tm(base, head, repo):
    return {"clean": True, "merged_tree": "merged-" + head, "conflicts": []}


def _conflict_tm(base, head, repo):
    return {"clean": False, "merged_tree": "t", "conflicts": ["f.rs"]}


def _repro(ok):
    return lambda pr, merged, rec, repo: {"reproduced": ok, "failures": [] if ok else ["behavior:x"]}


_K = dict(master_ref="master", repo="/r", recorded=["parity"])


def test_clean_fold_reproduces_is_folded():
    r = fold_pr(pr_ref="pr1", staging_ref="stg", **_K,
                trial_merge_fn=_clean_tm, merge_fix_fn=None, reproduce_fn=_repro(True))
    assert r["verdict"] == "FOLDED" and r["staging_ref"] == "merged-pr1"


def test_clean_fold_but_behavior_broken_is_bod_clobber():
    r = fold_pr(pr_ref="pr1", staging_ref="stg", **_K,
                trial_merge_fn=_clean_tm, merge_fix_fn=None, reproduce_fn=_repro(False))
    assert r["verdict"] == "BOD" and r["reason"] == "clobber"


def test_conflict_no_fixer_is_bod_with_split_reason():
    # conflicts on staging; clean vs master -> 'stack'
    def tm(base, head, repo):
        return {"clean": base == "master", "merged_tree": "t", "conflicts": ["f"]}
    r = fold_pr(pr_ref="pr1", staging_ref="stg", **_K,
                trial_merge_fn=tm, merge_fix_fn=None, reproduce_fn=_repro(True))
    assert r["verdict"] == "BOD" and r["reason"] == "stack"


def test_conflict_fixed_then_reproduces_is_folded():
    calls = {"n": 0}

    def tm(base, head, repo):
        calls["n"] += 1
        # first trial (on 'stg') conflicts; after fix, trial on 'fixed' is clean
        return {"clean": base == "fixed", "merged_tree": "merged", "conflicts": ["f"]}

    fix = lambda pr, stg, repo: {"resolved": True, "staging_ref": "fixed"}
    r = fold_pr(pr_ref="pr1", staging_ref="stg", **_K,
                trial_merge_fn=tm, merge_fix_fn=fix, reproduce_fn=_repro(True))
    assert r["verdict"] == "FOLDED" and r["staging_ref"] == "merged"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_staging.py -q`
Expected: FAIL — `ImportError: cannot import name 'fold_pr'`.

- [ ] **Step 3: Append `fold_pr` to `staging.py`**

```python
def fold_pr(*, pr_ref, staging_ref, master_ref, repo, recorded,
            trial_merge_fn, merge_fix_fn=None, reproduce_fn) -> dict:
    """Fold one PR onto staging: trial-merge -> (resolve) -> recall-reproduce -> FOLDED/BOD."""
    def _validate(stg):
        tm = trial_merge_fn(stg, pr_ref, repo)
        if not tm["clean"]:
            return None
        rr = recall_reproduce(pr_ref, tm["merged_tree"], recorded, repo, reproduce_fn=reproduce_fn)
        if not rr["reproduced"]:
            return {"pr": pr_ref, "verdict": "BOD", "staging_ref": stg,
                    "reason": "clobber", "detail": ", ".join(rr["failures"])}
        return {"pr": pr_ref, "verdict": "FOLDED", "staging_ref": tm["merged_tree"],
                "reason": "folded", "detail": ""}

    res = _validate(staging_ref)
    if res is not None:
        return res

    # trial conflicted: try the codex merge-fix (resolve on staging), then re-validate.
    if merge_fix_fn is not None:
        fix = merge_fix_fn(pr_ref, staging_ref, repo)
        if fix.get("resolved"):
            res = _validate(fix["staging_ref"])
            if res is not None:
                return res

    # unresolved -> BOD, split reason for an actionable message.
    reason = classify_conflict(pr_ref, master_ref, repo, trial_merge_fn=trial_merge_fn)
    return {"pr": pr_ref, "verdict": "BOD", "staging_ref": staging_ref,
            "reason": reason, "detail": "rebase on master" if reason == "stale"
            else "conflicts with an already-approved PR on the stack"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_staging.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add autoresearch/ar/gate/staging.py autoresearch/ar/tests/test_gate_staging.py
git commit -m "feat(ar-gate): fold_pr — trial -> merge-fix resolve -> recall-reproduce -> FOLDED/BOD

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `stack_train` — greedily stack the approved set, resolve-or-debt

**Files:** Modify `staging.py` + `__init__.py`; Test append.

**Interfaces:**
- Produces `stack_train(*, approved_prs, master_ref, repo, fold_fn) -> dict` returning
  `{"train": [pr,...], "debt": [{"pr", "reason", "detail"}], "staging_ref": str}`.
  Starts `staging_ref = master_ref`; folds each approved PR in order via `fold_fn`
  (which wraps `fold_pr` with the seams bound). A FOLDED result advances `staging_ref`
  and appends to `train`; a BOD appends to `debt` (staging tip unchanged) and continues.

- [ ] **Step 1: Write the failing test (append)**

```python
from autoresearch.ar.gate.staging import stack_train


def test_stack_train_folds_clean_and_collects_debt():
    # fold_fn: pr2 BODs (stale); others FOLD, advancing the tip.
    def fold_fn(pr, staging_ref):
        if pr == "pr2":
            return {"pr": pr, "verdict": "BOD", "staging_ref": staging_ref,
                    "reason": "stale", "detail": "rebase on master"}
        return {"pr": pr, "verdict": "FOLDED", "staging_ref": staging_ref + "+" + pr,
                "reason": "folded", "detail": ""}

    out = stack_train(approved_prs=["pr1", "pr2", "pr3"], master_ref="M", repo="/r", fold_fn=fold_fn)
    assert out["train"] == ["pr1", "pr3"]
    assert [d["pr"] for d in out["debt"]] == ["pr2"] and out["debt"][0]["reason"] == "stale"
    assert out["staging_ref"] == "M+pr1+pr3"     # tip advanced only by folded PRs


def test_stack_train_empty_is_master():
    out = stack_train(approved_prs=[], master_ref="M", repo="/r", fold_fn=lambda p, s: None)
    assert out["train"] == [] and out["staging_ref"] == "M" and out["debt"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_staging.py -q`
Expected: FAIL — `ImportError: cannot import name 'stack_train'`.

- [ ] **Step 3: Append `stack_train` to `staging.py`; re-export**

```python
def stack_train(*, approved_prs, master_ref, repo, fold_fn) -> dict:
    """Greedily stack the approved PRs onto a derived staging (starting at master).
    fold_fn(pr, staging_ref) -> a fold_pr-style result. FOLDED advances the tip and
    joins the train; BOD is collected as debt (tip unchanged) and stacking continues."""
    staging_ref = master_ref
    train, debt = [], []
    for pr in approved_prs:
        res = fold_fn(pr, staging_ref)
        if res["verdict"] == "FOLDED":
            staging_ref = res["staging_ref"]
            train.append(pr)
        else:
            debt.append({"pr": pr, "reason": res["reason"], "detail": res.get("detail", "")})
    return {"train": train, "debt": debt, "staging_ref": staging_ref}
```

Update `autoresearch/ar/gate/__init__.py` to add:
```python
from .staging import classify_conflict, fold_pr, land_train, recall_reproduce, stack_train
```
and extend `__all__` with `"classify_conflict", "recall_reproduce", "fold_pr", "stack_train", "land_train"`.
(`land_train` is Task 5 — import it now so the re-export is complete after Task 5; if Task 5
is not yet done, import only the four that exist and add `land_train` in Task 5.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_staging.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add autoresearch/ar/gate/staging.py autoresearch/ar/gate/__init__.py autoresearch/ar/tests/test_gate_staging.py
git commit -m "feat(ar-gate): stack_train — greedy stack, resolve-or-debt

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `land_train` — flush the train to master, re-validate, close-behind

**Files:** Modify `staging.py`; ensure `__init__.py` re-exports `land_train`; Test append.

**Interfaces:**
- Produces `land_train(*, train, staging_ref, master_ref, repo, git, land_reproduce_fn) -> dict`
  returning `{"landed": bool, "master_sha": str|None, "closed": [pr,...], "reason": str}`.
  Flow: (1) if `train` is empty → `{"landed": False, "reason": "empty-train", "closed": []}`.
  (2) re-validate the LANDED master before finalizing: `land_reproduce_fn(staging_ref,
  master_ref, repo) -> {"reproduced": bool, "failures": [str]}` (the whole stack's
  recorded behaviors reproduce on the landed master). If not reproduced →
  `{"landed": False, "reason": "landing-clobber", "closed": []}`. (3) else land:
  `git(repo, "merge", "--no-ff", staging_ref)` (non-squash so commits become master
  ancestors); on rc 0 → `{"landed": True, "master_sha": <new head>, "closed": train,
  "reason": "landed"}`; on rc != 0 → `{"landed": False, "reason": "merge-failed",
  "closed": []}`.

- [ ] **Step 1: Write the failing test (append)**

```python
from autoresearch.ar.gate.staging import land_train


def _git_ok(repo, *args):
    if args[:1] == ("merge",):
        return (0, "")
    if args[:2] == ("rev-parse", "HEAD"):
        return (0, "landedsha\n")
    return (0, "")


def _repro_ok(stg, master, repo):
    return {"reproduced": True, "failures": []}


def test_land_flushes_train_and_closes_behind():
    out = land_train(train=["pr1", "pr3"], staging_ref="stg", master_ref="M", repo="/r",
                     git=_git_ok, land_reproduce_fn=_repro_ok)
    assert out["landed"] is True and out["closed"] == ["pr1", "pr3"]
    assert out["master_sha"] == "landedsha"


def test_land_empty_train_is_noop():
    out = land_train(train=[], staging_ref="stg", master_ref="M", repo="/r",
                     git=_git_ok, land_reproduce_fn=_repro_ok)
    assert out["landed"] is False and out["reason"] == "empty-train"


def test_land_reclobber_blocks_landing():
    bad = lambda stg, master, repo: {"reproduced": False, "failures": ["behavior:x"]}
    out = land_train(train=["pr1"], staging_ref="stg", master_ref="M", repo="/r",
                     git=_git_ok, land_reproduce_fn=bad)
    assert out["landed"] is False and out["reason"] == "landing-clobber" and out["closed"] == []


def test_land_merge_failure_does_not_close():
    def git_fail(repo, *args):
        return (1, "conflict") if args[:1] == ("merge",) else (0, "")
    out = land_train(train=["pr1"], staging_ref="stg", master_ref="M", repo="/r",
                     git=git_fail, land_reproduce_fn=_repro_ok)
    assert out["landed"] is False and out["reason"] == "merge-failed" and out["closed"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_staging.py -q`
Expected: FAIL — `ImportError: cannot import name 'land_train'`.

- [ ] **Step 3: Append `land_train` to `staging.py`**

```python
def land_train(*, train, staging_ref, master_ref, repo, git, land_reproduce_fn) -> dict:
    """Flush the whole approved train to master in one non-clobber merge (spec §11).
    Re-validates the LANDED result (recall-reproduce the stack's behaviors) BEFORE
    finalizing, then merges non-squash so folded commits become master ancestors; the
    folded PRs are returned in ``closed`` for close-behind bookkeeping."""
    if not train:
        return {"landed": False, "master_sha": None, "closed": [], "reason": "empty-train"}

    rr = land_reproduce_fn(staging_ref, master_ref, repo)
    if not rr.get("reproduced"):
        return {"landed": False, "master_sha": None, "closed": [], "reason": "landing-clobber"}

    rc, _ = git(repo, "merge", "--no-ff", staging_ref)
    if rc != 0:
        return {"landed": False, "master_sha": None, "closed": [], "reason": "merge-failed"}
    _, head = git(repo, "rev-parse", "HEAD")
    return {"landed": True, "master_sha": head.strip(), "closed": list(train), "reason": "landed"}
```

Ensure `autoresearch/ar/gate/__init__.py` re-exports `land_train` (add it if Task 4
imported only the four).

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_staging.py -q`
Expected: PASS (15 tests).

- [ ] **Step 5: Full gate suite (no regressions)**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/ -q`
Expected: PASS, 0 failed (all prior tests + the new staging tests).

- [ ] **Step 6: Commit**

```bash
git add autoresearch/ar/gate/staging.py autoresearch/ar/gate/__init__.py autoresearch/ar/tests/test_gate_staging.py
git commit -m "feat(ar-gate): land_train — flush train, re-validate landed master, close-behind

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (Phase 5):**
- §11 derive/stack (staging = master + approved) → Task 4 (`stack_train`). ✔
- §11 resolve-not-punt (merge-fix before BOD) → Task 3 (`fold_pr`). ✔
- §11 debt split (stale vs stack) → Task 1 (`classify_conflict`) + Task 3. ✔
- §10 post-merge RECALL-reproduce (no re-run) → Task 2 (`recall_reproduce`) + Task 3 fold + Task 5 landing. ✔
- §11 land-the-whole-train + close-behind + landing re-validation → Task 5 (`land_train`). ✔
- **Deferred (prod wiring):** real git rebase/merge, real codex merge-fix, real GPU
  recall-reproduce, GitHub close-behind API, `ar gate --stage/--land` CLI — the seams
  are here; the live bindings are the next (authored) step.

**Placeholder scan:** no TBD/TODO; complete code + real assertions + exact venv commands. ✔

**Type consistency:** `fold_pr`/`stack_train` share the `{verdict, staging_ref, reason,
detail}` shape; `classify_conflict` returns the `"stale"|"stack"` strings `fold_pr`
consumes; `recall_reproduce`/`reproduce_fn` return `{reproduced, failures}`;
`land_train`'s `git` seam matches `merge.default_run_git`'s `(rc, out)`. ✔

## Next (prod wiring — authored, not this plan)

Bind the seams: `trial_merge` (real) · `merge_fix_fn` = codex-on-staging · `reproduce_fn`
= re-run the PR's recorded behaviors on the merged tree · `git` = real · close-behind =
GitHub API. Add `ar gate --stage`/`--land` + a workflow staging job triggered on a
landing event.
