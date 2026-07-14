# Copyright (c) Kaden Schutt
"""ar.gate.dispatch — the codex/dispatch layer (spec §8).

Claude reads the diff and emits a plan: a risk class, a serve_harness FLOOR spec,
and a list of bespoke behavior tests (each a prompt + which codex tier runs it).
This module is the deterministic glue around that plan:

  * parse_plan   — parse Claude's plan and apply the classify_pr FLOOR to its risk
                   (escalate-only: Claude can go higher than the path floor, never
                   lower — a kernel PR can't be de-classified to trivial).
  * run_behavior_test(s) — execute each bespoke test via the injected bounded
                   gate-local Codex seam, reading a structured verdict Codex writes. This is
                   the "codex tests beyond serve_harness" piece — codex runs the
                   test GENERALLY on-box (build / run the new path / verify), not
                   bound to serve_harness.
  * aggregate    — combine the deterministic serve_harness floor verdict with the
                   agentic behavior-test results: PASS iff floor PASS AND every
                   behavior test passed.

Everything here is pure/no-GPU with an injected agent_exec_fn; the only real GPU/
Codex touch is the gate-local executor itself in production.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .routing import classify_pr

# Ascending risk; the executor tier grows with it. Unknown -> treated as the
# extreme (claude unknown -> trivial index; floor unknown -> max, the safe side).
RISK_ORDER = ("trivial", "low", "moderate", "high-risk")

# Physical runner ownership is deterministic. Claude decides which boxes need to
# run; this table prevents a plan from inventing labels or sending an arch to the
# wrong host. ``labels`` are serialized into the dynamic Actions matrix.
BOXES = {
    "hipx": {
        "archs": ("gfx1100", "gfx1151"),
        "labels": ("self-hosted", "gfx1100", "gfx1151"),
    },
    "hiptrx": {
        "archs": ("gfx1201",),
        "labels": ("self-hosted", "gfx1201"),
    },
}

__all__ = [
    "BOXES",
    "floor_risk",
    "parse_plan",
    "validate_dispatch_plan",
    "validate_interpret_action",
    "verify_evidence",
    "run_behavior_test",
    "run_behavior_tests",
    "aggregate",
]


def floor_risk(claude_risk: str, floor: str) -> str:
    """Escalate-only: the HIGHER of Claude's semantic read and the classify_pr floor."""
    ci = RISK_ORDER.index(claude_risk) if claude_risk in RISK_ORDER else 0
    fi = RISK_ORDER.index(floor) if floor in RISK_ORDER else len(RISK_ORDER) - 1
    return RISK_ORDER[max(ci, fi)]


def parse_plan(plan, changed_files, lines_changed=None) -> dict:
    """Parse Claude's dispatch plan (dict or JSON string), floor its risk with
    classify_pr, and normalize. Returns {risk, floor_risk, serve_floor, behavior_tests, reason}."""
    if isinstance(plan, str):
        plan = json.loads(plan)
    floor = classify_pr(changed_files, lines_changed=lines_changed)
    return {
        "risk": floor_risk(plan.get("risk", "trivial"), floor),
        "floor_risk": floor,
        "serve_floor": dict(plan.get("serve_floor", {}) or {}),
        "behavior_tests": list(plan.get("behavior_tests", []) or []),
        "reason": str(plan.get("reason", "")),
    }


def _strings(value, field: str, errors: list[str]) -> list[str]:
    """Return a de-duplicated list of strings or record a schema error."""
    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
        errors.append(f"{field} must be a list of non-empty strings")
        return []
    return list(dict.fromkeys(value))


def _normalize_behaviors(value, *, route: dict, field: str, errors: list[str]) -> list[dict]:
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")
        return []
    out = []
    seen = set()
    for i, item in enumerate(value):
        where = f"{field}[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{where} must be an object")
            continue
        test_id = item.get("id")
        if not isinstance(test_id, str) or not test_id:
            errors.append(f"{where}.id must be a non-empty string")
            continue
        if test_id in seen:
            errors.append(f"duplicate behavior id: {test_id}")
            continue
        seen.add(test_id)
        missing = [k for k in ("what", "prompt", "expect")
                   if not isinstance(item.get(k), str) or not item.get(k)]
        if missing:
            errors.append(f"{where} missing non-empty fields: {','.join(missing)}")
            continue
        normalized = dict(item)
        normalized.setdefault("harness", route.get("harness") or "codex")
        normalized.setdefault("model", route.get("model") or None)
        normalized.setdefault("effort", route.get("effort") or "high")
        out.append(normalized)
    return out


def validate_dispatch_plan(plan, changed_files, cfg, lines_changed=None) -> dict:
    """Validate Claude's semantic dispatch and produce the Actions matrix.

    A valid plan is the *only* release signal for self-hosted jobs. Deterministic
    path/arch rules are coverage floors: Claude may add boxes/archs/models/tests,
    but it cannot omit an affected arch or invent runner ownership. The returned
    matrix has exactly the boxes Claude selected after those checks.
    """
    errors: list[str] = []
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except json.JSONDecodeError as e:
            return {"valid": False, "errors": [f"invalid JSON: {e}"],
                    "matrix": {"include": []}}
    if not isinstance(plan, dict):
        return {"valid": False, "errors": ["plan must be a JSON object"],
                "matrix": {"include": []}}

    if plan.get("schema") != 1:
        errors.append("schema must equal 1")
    decision = plan.get("decision")
    if decision not in ("run", "skip"):
        errors.append("decision must be 'run' or 'skip'")
    claude_risk = plan.get("risk")
    if claude_risk not in RISK_ORDER:
        errors.append(f"risk must be one of {RISK_ORDER}")
        claude_risk = "trivial"
    floor = classify_pr(changed_files, lines_changed=lines_changed)
    risk = floor_risk(claude_risk, floor)
    route = cfg.route(risk)
    reason = plan.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("reason must be a non-empty string")
        reason = ""
    raw_boxes = plan.get("boxes")
    if not isinstance(raw_boxes, dict):
        errors.append("boxes must be an object keyed by runner name")
        raw_boxes = {}
    unknown = sorted(set(raw_boxes) - set(BOXES))
    if unknown:
        errors.append("unknown boxes: " + ",".join(unknown))

    # Import lazily: run.py does not import dispatch at module load, so this does
    # not create a cycle and keeps arch ownership in one source of truth.
    from .run import affected_archs

    required_archs = affected_archs(changed_files, cfg)
    normalized_boxes = {}
    matrix = []
    covered_archs: set[str] = set()
    any_run = False
    for box, meta in BOXES.items():
        raw = raw_boxes.get(box, {})
        if not isinstance(raw, dict):
            errors.append(f"boxes.{box} must be an object")
            raw = {}
        run = raw.get("run") is True
        if "run" in raw and not isinstance(raw.get("run"), bool):
            errors.append(f"boxes.{box}.run must be boolean")
        archs = _strings(raw.get("archs", []), f"boxes.{box}.archs", errors)
        models = _strings(raw.get("models", []), f"boxes.{box}.models", errors)
        behaviors = _normalize_behaviors(
            raw.get("behavior_tests", []), route=route,
            field=f"boxes.{box}.behavior_tests", errors=errors,
        )
        invalid_archs = [a for a in archs if a not in meta["archs"]]
        if invalid_archs:
            errors.append(f"boxes.{box} does not own archs: {','.join(invalid_archs)}")
        for arch in archs:
            for model in models:
                if not cfg.fits(model, arch):
                    errors.append(f"model {model} does not fit {arch}")
        if not run and (archs or models or behaviors):
            errors.append(f"boxes.{box} has work but run=false")
        if run and not archs and not behaviors:
            errors.append(f"boxes.{box} run=true but has no archs or behavior tests")
        if run and archs and not models:
            errors.append(f"boxes.{box} must select at least one model for GPU arch work")
        if run:
            any_run = True
            covered_archs.update(archs)
        normalized_boxes[box] = {
            "run": run,
            "archs": archs,
            "models": models,
            "behavior_tests": behaviors,
        }
        if run:
            matrix.append({
                "box": box,
                "archs": " ".join(archs),
                "models": ",".join(models),
                "labels": json.dumps(list(meta["labels"]), separators=(",", ":")),
            })

    missing_archs = [a for a in required_archs if a not in covered_archs]
    if missing_archs:
        errors.append("Claude dispatch omitted affected archs: " + ",".join(missing_archs))
    if decision == "skip":
        if floor != "trivial":
            errors.append(f"only trivial diffs may be skipped (floor={floor})")
        if any_run:
            errors.append("decision='skip' cannot schedule runners")
    elif decision == "run" and not any_run:
        errors.append("decision='run' must schedule at least one runner")

    valid = not errors
    return {
        "valid": valid,
        "errors": errors,
        "schema": 1,
        "decision": decision,
        "risk": risk,
        "floor_risk": floor,
        "reason": reason,
        "required_archs": required_archs,
        "boxes": normalized_boxes,
        "matrix": {"include": matrix if valid and decision == "run" else []},
    }


def validate_interpret_action(interp: dict, action: dict) -> dict:
    """Mechanically constrain Claude's final proposal to the gate outcome."""
    errors = []
    if not isinstance(interp, dict) or not isinstance(action, dict):
        return {"valid": False, "errors": ["interp and action must be objects"]}
    outcome = interp.get("outcome", {})
    expected = outcome.get("action")
    proposed = action.get("action")
    allowed = {
        "bod": {"bod"},
        "auto_merge": {"auto_merge", "hold"},
        "tag_maintainer": {"tag_maintainer"},
        "draft_report": {"draft_report"},
        "neutral": {"hold"},
    }.get(expected, set())
    if proposed not in allowed:
        errors.append(f"action {proposed!r} is not allowed for outcome {expected!r}")
    comment = action.get("comment_markdown")
    if not isinstance(comment, str) or not comment.strip():
        errors.append("comment_markdown must be non-empty")
    if outcome.get("status") != "success" and proposed in ("auto_merge", "tag_maintainer"):
        errors.append("failed deterministic outcome cannot propose a green action")
    return {"valid": not errors, "errors": errors, "action": proposed,
            "comment_markdown": comment if isinstance(comment, str) else ""}


def verify_evidence(plan: dict, *, results_dir, behavior_dir, pr, base_sha, head_sha) -> dict:
    """Join the dynamic matrix's evidence and reject missing/stale artifacts.

    Artifact download success is not enough: Actions pattern downloads can
    succeed with only one box present. This verifies every selected box, arch,
    behavior id, and immutable identity before interpretation is allowed.
    """
    errors: list[str] = []
    behavior_results: list[dict] = []
    results_root = Path(results_dir)
    behavior_root = Path(behavior_dir)
    boxes = plan.get("boxes", {}) if isinstance(plan, dict) else {}
    for box, assignment in boxes.items():
        if not isinstance(assignment, dict) or not assignment.get("run"):
            continue
        archs = assignment.get("archs", [])
        expected_results = ([f"{a}.json" for a in archs]
                            if archs else [f"behavior-only-{box}.json"])
        for name in expected_results:
            path = results_root / name
            if not path.exists():
                errors.append(f"missing result {path}")
                continue
            try:
                row = json.loads(path.read_text())
            except Exception as e:
                errors.append(f"malformed result {path}: {e}")
                continue
            for key, want in (("pr", pr), ("base_sha", base_sha),
                              ("head_sha", head_sha), ("box", box)):
                if str(row.get(key)) != str(want):
                    errors.append(f"{path} {key}={row.get(key)!r}, expected {want!r}")
            expected_arch = name.removesuffix(".json")
            if row.get("arch") != expected_arch:
                errors.append(f"{path} arch={row.get('arch')!r}, expected {expected_arch!r}")
            if row.get("verdict") not in ("PASS", "REJECT", "BOD"):
                errors.append(f"{path} invalid verdict {row.get('verdict')!r}")

        bpath = behavior_root / f"behavior-{box}.json"
        if not bpath.exists():
            errors.append(f"missing behavior result {bpath}")
            continue
        try:
            box_rows = json.loads(bpath.read_text())
            if not isinstance(box_rows, list):
                raise TypeError("not a list")
        except Exception as e:
            errors.append(f"malformed behavior result {bpath}: {e}")
            continue
        expected_ids = {b["id"] for b in assignment.get("behavior_tests", [])}
        observed_ids = {b.get("id") for b in box_rows}
        if observed_ids != expected_ids:
            errors.append(
                f"{bpath} ids={sorted(map(str, observed_ids))}, "
                f"expected={sorted(map(str, expected_ids))}"
            )
        for row in box_rows:
            if not isinstance(row.get("passed"), bool):
                errors.append(f"{bpath}:{row.get('id')} passed must be boolean")
            for key, want in (("base_sha", base_sha), ("head_sha", head_sha), ("box", box)):
                if str(row.get(key)) != str(want):
                    errors.append(f"{bpath}:{row.get('id')} {key} mismatch")
        behavior_results.extend(box_rows)
    return {"valid": not errors, "errors": errors, "behavior_results": behavior_results}


def run_behavior_test(bt, *, agent_exec_fn, cwd, verdict_path) -> dict:
    """Run ONE bespoke behavior test via codex. ``agent_exec_fn(harness, model, effort,
    prompt, cwd, verdict_path) -> int`` is the injected seam. codex is
    told to write a structured verdict to ``verdict_path``; a missing/unreadable verdict
    is a FAIL (never a silent pass). Returns {what, passed, harness, model, detail}."""
    prompt = (
        (bt.get("prompt") or "")
        + f"\n\nExpected: {bt.get('expect', 'the behavior works correctly')}."
        + f"\nBuild/run whatever is needed to verify this on-box (you are NOT limited to"
        + f" serve_harness). When done, write your verdict as JSON to {verdict_path}:"
        + ' {"passed": <true|false>, "detail": "<one-line reason>"}. Then STOP immediately.'
    )
    exec_error = None
    try:
        rc = agent_exec_fn(
            harness=bt.get("harness", "codex"), model=bt.get("model"),
            effort=bt.get("effort", "high"), prompt=prompt, cwd=cwd,
            verdict_path=verdict_path,
        )
    except Exception as e:
        # Executor discovery/auth/process errors are structured behavior failures,
        # not uncaught tracebacks that leave an empty, unjoinable artifact.
        rc = 127
        exec_error = f"executor error: {type(e).__name__}: {e}"
    passed, detail = False, exec_error or f"no verdict written (exec rc={rc})"
    try:
        with open(verdict_path) as fh:
            v = json.load(fh)
        passed = bool(v.get("passed"))
        detail = str(v.get("detail", ""))
    except Exception as e:  # missing / malformed verdict -> FAIL, do not silently pass
        if exec_error is None:
            detail = f"verdict unreadable ({e}); exec rc={rc}"
    return {"id": bt.get("id"), "what": bt.get("what", "behavior"), "passed": passed,
            "harness": bt.get("harness", "codex"), "model": bt.get("model"), "detail": detail}


def run_behavior_tests(behavior_tests, *, agent_exec_fn, cwd, verdict_dir) -> list[dict]:
    """Run every bespoke behavior test, each with its own verdict file."""
    os.makedirs(verdict_dir, exist_ok=True)
    out = []
    for i, bt in enumerate(behavior_tests):
        vp = os.path.join(verdict_dir, f"behavior_{i}.json")
        out.append(run_behavior_test(bt, agent_exec_fn=agent_exec_fn, cwd=cwd, verdict_path=vp))
    return out


def aggregate(floor_verdict, behavior_results) -> dict:
    """Combine the deterministic serve_harness FLOOR (a run_gate-style {verdict, reasons})
    with the agentic behavior-test results. PASS iff floor PASS AND every behavior test
    passed; otherwise REJECT, itemizing the failed floor reasons + failed behaviors."""
    floor_ok = floor_verdict.get("verdict") == "PASS"
    failed = [b for b in behavior_results if not b["passed"]]
    reasons = list(floor_verdict.get("reasons", []))
    if not floor_ok and not reasons:
        reasons.append(str(floor_verdict.get("verdict", "floor_fail")).lower())
    reasons += [f"behavior:{b['what']}" for b in failed]
    return {
        "verdict": "PASS" if (floor_ok and not failed) else "REJECT",
        "floor": floor_verdict,
        "behavior_results": behavior_results,
        "reasons": reasons,
    }
