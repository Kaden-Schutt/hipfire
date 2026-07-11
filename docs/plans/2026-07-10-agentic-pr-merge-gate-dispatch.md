# Agentic PR Merge-Gate — Dispatch Core (Phase 3a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the no-GPU-testable **dispatch + interpret core** the Phase-3
workflow consumes (spec §8.1 + §12): a PR risk classifier, the executor routing
table, and the deterministic PR-outcome decision (auto-merge / tag-maintainer /
BOD / neutral / draft-report).

**Architecture:** Two new modules in `autoresearch/ar/gate/` plus a `[routing]`
extension to `GateConfig`. `classify_pr` maps a changed-file list to a risk class
using the coherence-gate trigger taxonomy `claude-review.yml` already encodes.
`GateConfig.route` turns a class into `(harness, model, effort)` for the
`agent_exec` executor. `decide_pr` reduces the per-arch gate verdicts + author +
draft-state + Claude's helpfulness judgment into a single PR action — the
deterministic skeleton Claude wraps with prose. All pure functions, no GPU.

**Tech Stack:** Python 3.11+ (stdlib); pytest; no third-party deps. Consumes the
Phase-1 `run_gate` / Phase-2 `gate4` result shapes; the real GPU execution +
GitHub wiring is Phase 3b.

## Global Constraints

- **No-GPU unit-testable** — pure functions over lists/dicts; runs under
  `scripts/no-gpu-ci.sh`. Use the venv `/home/kaden/.venvs/hipfire-pytest/bin/python`
  for pytest (system python lacks it).
- **Risk taxonomy = the coherence-gate trigger set** — high-risk iff the diff
  touches kernels / dispatch / forward-pass / quant (the set CLAUDE.md's coherence
  gate guards); trivial iff only docs/CI/`.md`; else low (small) / moderate.
  Conservative on ambiguity (err toward *more* testing).
- **Executor routing (spec §8.1), config-driven**: trivial→none · low→codex luna
  high · moderate→codex terra high · high-risk→**codex sol xhigh** (+ grok gfx1201
  diversity, wired in 3b). Table lives in `pr_gate.toml`.
- **Auto-merge authority (spec §12)**: only `auto_merge_authors` (config;
  `["Kaden-Schutt"]`) auto-merge; other `maintainers` get tagged to `@claude
  /merge`; non-maintainers → tag the invoking maintainer; Draft → report only;
  not-helpful → neutral; any REJECT/BOD → BOD.
- **New files** carry `# Copyright (c) Kaden Schutt` as line 1.
- Commit messages end with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **Out of scope (Phase 3b):** `gpu-gates.yml`, `claude-review.yml` dispatch/
  interpret wiring, `LiveServeRunner` binding of `run_gate`/`gate4`/`merge_fix`,
  the grok diversity second-opinion, GitHub API calls.

## File Structure

- Create `autoresearch/ar/gate/routing.py` — `classify_pr` + the taxonomy markers.
- Modify `autoresearch/config/pr_gate.toml` — `auto_merge_authors` + `[routing]`.
- Modify `autoresearch/ar/gate/config.py` — `GateConfig.routing`/`auto_merge_authors`
  + `route` / `is_auto_merge_author`.
- Create `autoresearch/ar/gate/outcome.py` — `decide_pr` + `format_pr_comment`.
- Modify `autoresearch/ar/gate/__init__.py` — re-export the new symbols.
- Tests: `autoresearch/ar/tests/test_gate_routing.py`,
  `test_gate_outcome.py`; extend `test_gate_config.py`.

---

### Task 1: `classify_pr` — PR risk classifier

**Files:**
- Create: `autoresearch/ar/gate/routing.py`
- Test: `autoresearch/ar/tests/test_gate_routing.py`

**Interfaces:**
- Produces:
  - `classify_pr(changed_files: list[str], lines_changed: int | None = None, small_threshold: int = 40) -> str`
    returning one of `"trivial"`, `"low"`, `"moderate"`, `"high-risk"`.

Rules, in order: any high-risk path → `"high-risk"`; no files or all trivial
paths → `"trivial"`; else `"low"` if `lines_changed` is known and
`≤ small_threshold`, otherwise `"moderate"`.

- [ ] **Step 1: Write the failing test**

Create `autoresearch/ar/tests/test_gate_routing.py`:

```python
# Copyright (c) Kaden Schutt
from autoresearch.ar.gate.routing import classify_pr


def test_kernel_change_is_high_risk():
    assert classify_pr(["kernels/src/gemv_hfq4g256_moe_down.hip"]) == "high-risk"


def test_dispatch_and_forward_are_high_risk():
    assert classify_pr(["crates/rdna-compute/src/dispatch.rs"]) == "high-risk"
    assert classify_pr(["crates/hipfire-arch-qwen35/src/forward.rs"]) == "high-risk"
    assert classify_pr(["crates/hipfire-quantize/src/hfq.rs"]) == "high-risk"


def test_high_risk_wins_even_mixed_with_docs():
    assert classify_pr(["docs/x.md", "kernels/src/a.hip"]) == "high-risk"


def test_docs_only_is_trivial():
    assert classify_pr(["docs/specs/x.md", ".github/workflows/ci.yml", "README.md"]) == "trivial"


def test_empty_diff_is_trivial():
    assert classify_pr([]) == "trivial"


def test_small_nonkernel_rust_is_low():
    assert classify_pr(["crates/hipfire-loader/src/lib.rs"], lines_changed=12) == "low"


def test_large_nonkernel_rust_is_moderate():
    assert classify_pr(["crates/hipfire-runtime/src/daemon_util.rs"], lines_changed=300) == "moderate"


def test_unknown_size_nonkernel_defaults_moderate():
    # conservative: if we can't size it, assume moderate (more coverage)
    assert classify_pr(["cli/index.ts"]) == "moderate"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_routing.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'autoresearch.ar.gate.routing'`.

- [ ] **Step 3: Write `classify_pr`**

Create `autoresearch/ar/gate/routing.py`:

```python
# Copyright (c) Kaden Schutt
"""ar.gate.routing — PR risk classification (spec §8.1).

Maps a changed-file list to a risk class the dispatcher uses to pick the on-box
executor tier. The high-risk set is exactly the coherence-gate trigger taxonomy
CLAUDE.md guards (kernels / dispatch / forward-pass / quant): a change there can
induce attractors / perf regressions and warrants the strongest executor + full
behavior coverage. Conservative on ambiguity — err toward more testing.
"""
from __future__ import annotations

# A path is high-risk if it contains ANY of these markers (substring match).
HIGH_RISK_MARKERS = (
    "kernels/",                       # all HIP kernel source
    "crates/rdna-compute/",           # dispatch + kernel launch + JIT
    "crates/hipfire-dispatch/",       # unified per-family dispatch
    "crates/hipfire-arch-",           # forward passes (all arch crates)
    "crates/hipfire-quantize/",       # quant encoders / formats
    "/sampler",                       # sampling path
    "dispatch.rs",                    # the most-reverted file
)

# A path is trivial if it matches ANY of these (prefix or suffix).
_TRIVIAL_PREFIXES = ("docs/", ".github/")
_TRIVIAL_SUFFIXES = (".md",)


def _is_high_risk(path: str) -> bool:
    return any(m in path for m in HIGH_RISK_MARKERS)


def _is_trivial(path: str) -> bool:
    return path.startswith(_TRIVIAL_PREFIXES) or path.endswith(_TRIVIAL_SUFFIXES)


def classify_pr(changed_files, lines_changed=None, small_threshold=40) -> str:
    """Classify a PR into 'trivial' | 'low' | 'moderate' | 'high-risk'."""
    if any(_is_high_risk(f) for f in changed_files):
        return "high-risk"
    if not changed_files or all(_is_trivial(f) for f in changed_files):
        return "trivial"
    if lines_changed is not None and lines_changed <= small_threshold:
        return "low"
    return "moderate"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_routing.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add autoresearch/ar/gate/routing.py autoresearch/ar/tests/test_gate_routing.py
git commit -m "feat(ar-gate): classify_pr — PR risk classifier (spec 8.1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Routing table + auto-merge authors in `GateConfig`

**Files:**
- Modify: `autoresearch/config/pr_gate.toml`
- Modify: `autoresearch/ar/gate/config.py`
- Test: `autoresearch/ar/tests/test_gate_config.py` (append)

**Interfaces:**
- Consumes: the Task-1 class strings.
- Produces (added to `GateConfig`):
  - fields `routing: dict[str, dict]`, `auto_merge_authors: list[str]`.
  - `GateConfig.route(pr_class: str) -> dict` — `{"harness","model","effort"}` for
    the class (unknown class → the `high-risk` row, fail-safe strongest).
  - `GateConfig.is_auto_merge_author(author: str) -> bool`.

- [ ] **Step 1: Write the failing test (append to `test_gate_config.py`)**

First extend the `_TOML` string in `test_gate_config.py` — add these lines before
the `[fit]` table:

```python
auto_merge_authors = ["Kaden-Schutt"]

[routing]
trivial = { harness = "none", model = "", effort = "" }
low = { harness = "codex", model = "gpt-5.6-luna", effort = "high" }
moderate = { harness = "codex", model = "gpt-5.6-terra", effort = "high" }
"high-risk" = { harness = "codex", model = "gpt-5.6-sol", effort = "xhigh" }
```

Then append these tests:

```python
def test_route_returns_executor_tier():
    import os, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_gate_config(_write(tmp))
    assert cfg.route("high-risk") == {"harness": "codex", "model": "gpt-5.6-sol", "effort": "xhigh"}
    assert cfg.route("low")["model"] == "gpt-5.6-luna"
    assert cfg.route("trivial")["harness"] == "none"


def test_route_unknown_class_is_failsafe_high_risk():
    import os, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_gate_config(_write(tmp))
    assert cfg.route("bogus") == cfg.route("high-risk")


def test_auto_merge_author():
    import os, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_gate_config(_write(tmp))
    assert cfg.is_auto_merge_author("Kaden-Schutt") is True
    assert cfg.is_auto_merge_author("fivetide") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_config.py -q`
Expected: FAIL — `AttributeError: 'GateConfig' object has no attribute 'route'`.

- [ ] **Step 3: Extend `pr_gate.toml` and `GateConfig`**

Append to `autoresearch/config/pr_gate.toml`:

```toml

# Only these authors auto-merge on a clean pass (spec §12); other maintainers are
# tagged to comment `@claude /merge`. The auto-merge kill-switch = this list.
auto_merge_authors = ["Kaden-Schutt"]

# Executor tier per PR risk class (spec §8.1). harness="none" => deterministic
# gate only (no agentic executor). Re-tuned from the merge ledger's per-tier
# false-pass / false-reject rates.
[routing]
trivial = { harness = "none", model = "", effort = "" }
low = { harness = "codex", model = "gpt-5.6-luna", effort = "high" }
moderate = { harness = "codex", model = "gpt-5.6-terra", effort = "high" }
"high-risk" = { harness = "codex", model = "gpt-5.6-sol", effort = "xhigh" }
```

In `autoresearch/ar/gate/config.py`, add the two fields to the `GateConfig`
dataclass (after `alpha`):

```python
    routing: dict = field(default_factory=dict)
    auto_merge_authors: list[str] = field(default_factory=list)
```

Add `from dataclasses import dataclass, field` is already imported in config.py
(Task-1 version imports `dataclass, field`) — no import change needed.

Add these two methods to `GateConfig`:

```python
    def route(self, pr_class: str) -> dict:
        """Executor {harness, model, effort} for a PR risk class. Unknown class ->
        the high-risk row (fail-safe strongest)."""
        return self.routing.get(pr_class, self.routing.get("high-risk", {}))

    def is_auto_merge_author(self, author: str) -> bool:
        return author in self.auto_merge_authors
```

Extend `load_gate_config` to read them — add these two kwargs to the
`GateConfig(...)` constructor call:

```python
        routing={str(k): dict(v) for k, v in data.get("routing", {}).items()},
        auto_merge_authors=[str(a) for a in data.get("auto_merge_authors", [])],
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_config.py -q`
Expected: PASS (the 4 original + 3 new = 7).

- [ ] **Step 5: Commit**

```bash
git add autoresearch/config/pr_gate.toml autoresearch/ar/gate/config.py \
        autoresearch/ar/tests/test_gate_config.py
git commit -m "feat(ar-gate): GateConfig routing table + auto_merge_authors (spec 8.1/12)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `decide_pr` + `format_pr_comment` — the interpreter's deterministic core

**Files:**
- Create: `autoresearch/ar/gate/outcome.py`
- Modify: `autoresearch/ar/gate/__init__.py`
- Test: `autoresearch/ar/tests/test_gate_outcome.py`

**Interfaces:**
- Consumes: a `GateConfig` (Task 2), the per-arch result shape
  `{"arch": str, "verdict": "PASS"|"REJECT"|"BOD", "reasons": [str], "bod": dict | None}`.
- Produces:
  - `decide_pr(*, arch_results, author, is_draft, helpful, cfg) -> dict` returning
    `{"action", "status", "reasons", "bod"}` where
    `action ∈ {"auto_merge","tag_maintainer","bod","neutral","draft_report"}` and
    `status ∈ {"success","failure","neutral"}`.
  - `format_pr_comment(outcome: dict, arch_results: list) -> str` — the markdown
    body (verdict line + per-arch table + BOD list or merge/tag note).

Decision order: (1) any arch REJECT/BOD → `bod`/failure (aggregate blockers).
(2) all PASS + draft → `draft_report`/success. (3) all PASS + not helpful →
`neutral`/success. (4) all PASS + helpful + `is_auto_merge_author` →
`auto_merge`/success. (5) all PASS + helpful + maintainer → `tag_maintainer`/
success. (6) all PASS + helpful + non-maintainer → `tag_maintainer`/success (a
maintainer must `@claude /merge`).

- [ ] **Step 1: Write the failing test**

Create `autoresearch/ar/tests/test_gate_outcome.py`:

```python
# Copyright (c) Kaden Schutt
import os
import tempfile

from autoresearch.ar.gate.config import load_gate_config
from autoresearch.ar.gate.outcome import decide_pr, format_pr_comment

_TOML = """
archs = ["gfx1100", "gfx1151", "gfx1201"]
canonical_models = ["qwen3.6-27b", "qwen3.6-a3b"]
maintainers = ["fivetide", "unverbraucht", "nwoolmer", "Kaden-Schutt"]
auto_merge_authors = ["Kaden-Schutt"]
floor = 0.15
drift_pct = 3.0
alpha = 0.05

[fit]
"qwen3.6-27b" = ["gfx1100", "gfx1151", "gfx1201"]

[routing]
"high-risk" = { harness = "codex", model = "gpt-5.6-sol", effort = "xhigh" }
"""


def _cfg():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "pr_gate.toml")
    with open(p, "w") as fh:
        fh.write(_TOML)
    return load_gate_config(p)


_PASS = [{"arch": "gfx1201", "verdict": "PASS", "reasons": [], "bod": None},
         {"arch": "gfx1100", "verdict": "PASS", "reasons": [], "bod": None}]


def test_kaden_clean_helpful_auto_merges():
    o = decide_pr(arch_results=_PASS, author="Kaden-Schutt", is_draft=False, helpful=True, cfg=_cfg())
    assert o["action"] == "auto_merge" and o["status"] == "success"


def test_other_maintainer_clean_helpful_is_tagged():
    o = decide_pr(arch_results=_PASS, author="fivetide", is_draft=False, helpful=True, cfg=_cfg())
    assert o["action"] == "tag_maintainer" and o["status"] == "success"


def test_non_maintainer_clean_is_tagged_for_maintainer_merge():
    o = decide_pr(arch_results=_PASS, author="randocontrib", is_draft=False, helpful=True, cfg=_cfg())
    assert o["action"] == "tag_maintainer"


def test_draft_never_merges():
    o = decide_pr(arch_results=_PASS, author="Kaden-Schutt", is_draft=True, helpful=True, cfg=_cfg())
    assert o["action"] == "draft_report" and o["status"] == "success"


def test_not_helpful_is_neutral():
    o = decide_pr(arch_results=_PASS, author="Kaden-Schutt", is_draft=False, helpful=False, cfg=_cfg())
    assert o["action"] == "neutral"


def test_any_arch_reject_is_bod_failure():
    mixed = _PASS + [{"arch": "gfx1151", "verdict": "REJECT", "reasons": ["perf_regression"], "bod": None}]
    o = decide_pr(arch_results=mixed, author="Kaden-Schutt", is_draft=False, helpful=True, cfg=_cfg())
    assert o["action"] == "bod" and o["status"] == "failure"
    assert any(b["detail"] == "perf_regression" for b in o["bod"]["blockers"])


def test_arch_bod_blockers_are_aggregated():
    b = {"blockers": [{"kind": "merge_conflict", "detail": "daemon.rs"}], "summary": "1 blocker(s)"}
    mixed = _PASS + [{"arch": "gfx1201", "verdict": "BOD", "reasons": [], "bod": b}]
    o = decide_pr(arch_results=mixed, author="Kaden-Schutt", is_draft=False, helpful=True, cfg=_cfg())
    assert o["action"] == "bod"
    assert {"kind": "merge_conflict", "detail": "daemon.rs", "arch": "gfx1201"} in o["bod"]["blockers"]


def test_comment_renders_verdict_and_table():
    o = decide_pr(arch_results=_PASS, author="Kaden-Schutt", is_draft=False, helpful=True, cfg=_cfg())
    md = format_pr_comment(o, _PASS)
    assert "auto_merge" in md.lower() or "merge" in md.lower()
    assert "gfx1201" in md and "gfx1100" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_outcome.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'autoresearch.ar.gate.outcome'`.

- [ ] **Step 3: Write `outcome.py`**

Create `autoresearch/ar/gate/outcome.py`:

```python
# Copyright (c) Kaden Schutt
"""ar.gate.outcome — the interpreter's deterministic PR decision (spec §12).

decide_pr reduces the per-arch gate verdicts + author + draft-state + Claude's
helpfulness judgment into a single PR action. This is the mechanical skeleton;
Claude (Tier 2) wraps it with the human-facing prose + the actual merge/tag call.
"""
from __future__ import annotations


def _aggregate_bod(arch_results) -> dict:
    """Merge every failing arch's blockers into one BOD, tagging each by arch."""
    blockers: list[dict] = []
    for r in arch_results:
        if r["verdict"] == "BOD" and r.get("bod"):
            for b in r["bod"].get("blockers", []):
                blockers.append({**b, "arch": r["arch"]})
        elif r["verdict"] == "REJECT":
            for reason in r.get("reasons", []):
                blockers.append({"kind": reason, "detail": reason, "arch": r["arch"]})
    summary = f"{len(blockers)} blocker(s) across "
    summary += ", ".join(sorted({r["arch"] for r in arch_results
                                 if r["verdict"] in ("REJECT", "BOD")}))
    return {"blockers": blockers, "summary": summary}


def decide_pr(*, arch_results, author, is_draft, helpful, cfg) -> dict:
    """Decide the PR action from the per-arch verdicts + authority + helpfulness."""
    failed = [r for r in arch_results if r["verdict"] in ("REJECT", "BOD")]
    if failed:
        return {"action": "bod", "status": "failure",
                "reasons": sorted({r["arch"] for r in failed}),
                "bod": _aggregate_bod(arch_results)}

    # All arches PASS from here.
    if is_draft:
        return {"action": "draft_report", "status": "success", "reasons": [], "bod": None}
    if not helpful:
        return {"action": "neutral", "status": "success", "reasons": ["not-helpful"], "bod": None}
    if cfg.is_auto_merge_author(author):
        return {"action": "auto_merge", "status": "success", "reasons": [], "bod": None}
    # Any other author (maintainer or non-maintainer) is tagged: a maintainer must
    # comment `@claude /merge` — the agent never merges another author's PR unasked.
    return {"action": "tag_maintainer", "status": "success", "reasons": [], "bod": None}


_ACTION_LINE = {
    "auto_merge": "✅ All gates green — **auto-merging** (flushes the staging train).",
    "tag_maintainer": "✅ All gates green — comment `@claude /merge` to land it.",
    "neutral": "🟡 All gates green, but no measurable improvement — clarify intent.",
    "draft_report": "📋 Draft — verdict + BOD only, no merge.",
    "bod": "❌ Blocked — see the Bill of Debt below.",
}


def format_pr_comment(outcome: dict, arch_results) -> str:
    """Render the PR comment markdown: verdict line + per-arch table + BOD/notes."""
    lines = [_ACTION_LINE.get(outcome["action"], outcome["action"]), "",
             "| arch | verdict |", "|---|---|"]
    for r in arch_results:
        lines.append(f"| {r['arch']} | {r['verdict']} |")
    if outcome.get("bod") and outcome["bod"]["blockers"]:
        lines += ["", "**Bill of Debt:**"]
        for b in outcome["bod"]["blockers"]:
            arch = b.get("arch", "")
            lines.append(f"- `{arch}` **{b['kind']}**: {b['detail']}")
    return "\n".join(lines)
```

Update `autoresearch/ar/gate/__init__.py` to re-export the new symbols:

```python
# Copyright (c) Kaden Schutt
"""ar.gate — the Tier-3 PR merge-gate engine (no-GPU-testable core)."""
from .config import GateConfig, load_gate_config
from .engine import gate_cell, run_gate
from .merge import assemble_bod, gate4, trial_merge
from .outcome import decide_pr, format_pr_comment
from .routing import classify_pr

__all__ = [
    "GateConfig", "load_gate_config", "gate_cell", "run_gate",
    "trial_merge", "assemble_bod", "gate4",
    "classify_pr", "decide_pr", "format_pr_comment",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/test_gate_outcome.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Full gate suite (no regressions)**

Run: `/home/kaden/.venvs/hipfire-pytest/bin/python -m pytest autoresearch/ar/tests/ -q`
Expected: PASS, 0 failed (all prior gate + ar tests plus the new routing/config/outcome tests).

- [ ] **Step 6: Commit**

```bash
git add autoresearch/ar/gate/outcome.py autoresearch/ar/gate/__init__.py \
        autoresearch/ar/tests/test_gate_outcome.py
git commit -m "feat(ar-gate): decide_pr + format_pr_comment — interpreter core (spec 12)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (Phase 3a):**
- §8.1 risk classifier → Task 1 (`classify_pr`, coherence-gate taxonomy). ✔
- §8.1 executor routing table → Task 2 (`route`, config `[routing]`). ✔
- §12 authority (Kaden auto / maintainer tag / draft report / not-helpful neutral
  / any-fail BOD) → Task 3 (`decide_pr`). ✔
- §11/§10 BOD aggregation across arches → Task 3 (`_aggregate_bod`). ✔
- **Deferred to Phase 3b:** `gpu-gates.yml`, `claude-review.yml` dispatch/interpret,
  `LiveServeRunner` wiring, grok diversity, GitHub API calls.

**Placeholder scan:** no TBD/TODO; complete code + real assertions + exact venv run
commands. ✔

**Type consistency:** `classify_pr` returns the four class strings `route` keys on;
`decide_pr` consumes the Phase-1/2 `{arch, verdict, reasons, bod}` shape and emits
`{action, status, reasons, bod}` that `format_pr_comment` reads; `GateConfig.route`/
`is_auto_merge_author` signatures match their call sites. ✔

## Next (Phase 3b — authored directly, not via this plan)

`gpu-gates.yml` (matrix + triggers + gpu-lock + aggregator) · `claude-review.yml`
elevation (dispatch → interpret → merge/tag/BOD, wiring `classify_pr`/`route`/
`decide_pr`) · `LiveServeRunner` binding of `run_gate`/`gate4`/`merge_fix` · grok
gfx1201 diversity check. Verified by actionlint + review, not pytest.
