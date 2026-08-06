# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Change-gate report builder and PR-ready markdown renderer."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from tools.change_gate.model import SCHEMA_ID, RouteResult, Selection

_BLOCKED_SELECTION = frozenset(
    {
        "blocked_model",
        "blocked_arch",
        "trimmed_budget",
        "excluded_heavy",
    }
)


def _selection_dict(item: Selection | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(item, Selection):
        return {
            "route_id": item.route_id,
            "matched_paths": list(item.matched_paths),
            "rule_reason": item.rule_reason,
            "status": item.status,
            "detail": item.detail,
        }
    return {
        "route_id": str(item.get("route_id", "")),
        "matched_paths": list(item.get("matched_paths") or ()),
        "rule_reason": str(item.get("rule_reason", "")),
        "status": str(item.get("status", "")),
        "detail": str(item.get("detail", "")),
    }


def _result_dict(item: RouteResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(item, RouteResult):
        verdict = item.verdict if isinstance(item.verdict, dict) else {"raw": item.verdict}
        return {
            "route_id": item.route_id,
            "status": item.status,
            "duration_s": float(item.duration_s),
            "verdict": verdict,
            "artifacts": list(item.artifacts),
        }
    verdict = item.get("verdict")
    if not isinstance(verdict, dict):
        verdict = {"raw": verdict}
    return {
        "route_id": str(item.get("route_id", "")),
        "status": str(item.get("status", "")),
        "duration_s": float(item.get("duration_s") or 0.0),
        "verdict": verdict,
        "artifacts": list(item.get("artifacts") or ()),
    }


def _status_of(item: Any) -> str:
    if isinstance(item, Selection):
        return item.status
    if isinstance(item, RouteResult):
        return item.status
    if isinstance(item, Mapping):
        return str(item.get("status", ""))
    return ""


def compute_verdict(
    selected: Sequence[Selection | Mapping[str, Any]] = (),
    not_run: Sequence[Selection | Mapping[str, Any]] = (),
    results: Sequence[RouteResult | Mapping[str, Any]] = (),
) -> str:
    """Precedence: incomplete (any blocked) > fail (any failure) > pass."""
    for item in (*selected, *not_run):
        status = _status_of(item)
        if status in _BLOCKED_SELECTION or status.startswith("blocked"):
            return "incomplete"
    for item in results:
        status = _status_of(item)
        if status == "blocked" or status.startswith("blocked"):
            return "incomplete"
    for item in results:
        if _status_of(item) == "fail":
            return "fail"
    return "pass"


def build_report(
    *,
    base: str,
    head: str,
    dirty: bool,
    host: Mapping[str, Any],
    changed_files: Sequence[str],
    selected: Sequence[Selection | Mapping[str, Any]],
    not_run: Sequence[Selection | Mapping[str, Any]],
    results: Sequence[RouteResult | Mapping[str, Any]] = (),
    est_minutes: float = 0.0,
) -> dict[str, Any]:
    """Build the batch-contract JSON object (`schema` = SCHEMA_ID)."""
    selected_dicts = [_selection_dict(s) for s in selected]
    not_run_dicts = [_selection_dict(s) for s in not_run]
    result_dicts = [_result_dict(r) for r in results]

    routes_selected = sum(1 for s in selected_dicts if s.get("status") == "selected")
    if routes_selected == 0 and selected_dicts:
        # Caller may pass only the to_run list (all status=selected).
        routes_selected = sum(
            1 for s in selected_dicts if s.get("status") in {"selected", ""}
        )

    routes_blocked = sum(
        1
        for s in (*selected_dicts, *not_run_dicts)
        if s.get("status") in _BLOCKED_SELECTION
        or str(s.get("status", "")).startswith("blocked")
    )

    actual_s = sum(float(r.get("duration_s") or 0.0) for r in result_dicts)

    host_out = {
        "gfx": str(host.get("gfx") or host.get("gfx_arch") or ""),
        "rocm": str(host.get("rocm") or host.get("rocm_version") or ""),
        "models_dir": str(host.get("models_dir") or ""),
    }

    return {
        "schema": SCHEMA_ID,
        "base": base,
        "head": head,
        "dirty": bool(dirty),
        "host": host_out,
        "changed_files": list(changed_files),
        "selected": selected_dicts,
        "not_run": not_run_dicts,
        "results": result_dicts,
        "totals": {
            "est_minutes": float(est_minutes),
            "actual_s": float(actual_s),
            "routes_selected": int(routes_selected),
            "routes_blocked": int(routes_blocked),
        },
        "verdict": compute_verdict(selected_dicts, not_run_dicts, result_dicts),
    }


def to_json(report: Mapping[str, Any]) -> str:
    """Serialize report JSON with a trailing newline."""
    return json.dumps(report, indent=2, sort_keys=False) + "\n"


def _fmt_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes}m{rem:04.1f}s"


def _pad(cell: str, width: int) -> str:
    if len(cell) >= width:
        return cell
    return cell + (" " * (width - len(cell)))


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [len(h) for h in headers]
    str_rows = [[str(c) for c in row] for row in rows]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    lines = [
        "| " + " | ".join(_pad(h, widths[i]) for i, h in enumerate(headers)) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    if not str_rows:
        # Keep the table structurally present even when empty.
        lines.append(
            "| " + " | ".join(_pad("—", widths[i]) for i in range(len(headers))) + " |"
        )
    else:
        for row in str_rows:
            padded = [
                _pad(row[i] if i < len(row) else "", widths[i])
                for i in range(len(headers))
            ]
            lines.append("| " + " | ".join(padded) + " |")
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    """PR-ready telemetry block. Always includes the NOT RUN table."""
    verdict = str(report.get("verdict") or "unknown").upper()
    badge = {
        "PASS": "PASS",
        "FAIL": "FAIL",
        "INCOMPLETE": "INCOMPLETE",
    }.get(verdict, verdict)

    host = report.get("host") or {}
    gfx = host.get("gfx") or "?"
    rocm = host.get("rocm") or "?"
    models_dir = host.get("models_dir") or "?"
    base = report.get("base") or "?"
    head = report.get("head") or "?"
    dirty_flag = " dirty" if report.get("dirty") else ""

    totals = report.get("totals") or {}
    est = totals.get("est_minutes", 0.0)
    actual_s = float(totals.get("actual_s") or 0.0)

    lines: list[str] = []
    lines.append(f"**change_gate: {badge}**")
    lines.append("")
    lines.append(
        f"host gfx=`{gfx}` rocm=`{rocm}` models_dir=`{models_dir}` · "
        f"`{base}`..`{head}`{dirty_flag} · "
        f"est={est}min actual={_fmt_duration(actual_s)}"
    )
    lines.append("")

    lines.append("### Routes RUN")
    run_rows: list[list[str]] = []
    results = report.get("results") or []
    if results:
        for r in results:
            run_rows.append(
                [
                    str(r.get("route_id") or ""),
                    str(r.get("status") or ""),
                    _fmt_duration(float(r.get("duration_s") or 0.0)),
                ]
            )
    else:
        # plan mode: selected routes not yet executed
        for s in report.get("selected") or []:
            if str(s.get("status") or "selected") == "selected":
                run_rows.append([str(s.get("route_id") or ""), "planned", "—"])
    lines.extend(_md_table(("route", "status", "duration"), run_rows))
    lines.append("")

    lines.append("### Routes NOT RUN")
    not_run_rows: list[list[str]] = []
    for s in report.get("not_run") or []:
        status = str(s.get("status") or "")
        detail = str(s.get("detail") or "")
        if status and detail and detail != status:
            reason = f"{status}: {detail}"
        else:
            reason = status or detail or "—"
        not_run_rows.append([str(s.get("route_id") or ""), reason])
    lines.extend(_md_table(("route", "reason"), not_run_rows))
    lines.append("")

    lines.append(
        "_Blocked or excluded routes mean coverage is incomplete — "
        "this report is not an admission that unrun surfaces are safe._"
    )
    lines.append("")
    return "\n".join(lines)
