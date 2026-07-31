# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""CLI entry: python3 -m tools.change_gate {plan|run|routes} ..."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[2]

def _silence_broken_pipe() -> None:
    """Avoid traceback when stdout is a closed pipe (e.g. ``... | head``)."""
    try:
        import signal

        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError):
        pass



def _usage() -> None:
    print(
        "usage: python3 -m tools.change_gate {plan|run|routes} ...",
        file=sys.stderr,
    )


def _exit_for_verdict(verdict: str) -> int:
    v = (verdict or "").lower()
    if v == "pass":
        return 0
    if v == "fail":
        return 1
    if v == "incomplete":
        return 2
    return 1


def _routes_by_id() -> dict[str, Any]:
    from tools.change_gate import routes as routes_mod

    if hasattr(routes_mod, "routes_by_id"):
        by_id = routes_mod.routes_by_id()
        if callable(by_id):
            by_id = by_id()
        return dict(by_id)
    if hasattr(routes_mod, "ROUTES"):
        routes = routes_mod.ROUTES
        if isinstance(routes, dict):
            return dict(routes)
        return {r.id: r for r in routes}
    raise RuntimeError("tools.change_gate.routes has no ROUTES/routes_by_id")


def _rules():
    from tools.change_gate import routes as routes_mod

    if hasattr(routes_mod, "rules"):
        out = routes_mod.rules()
        if callable(out):
            out = out()
        return tuple(out)
    if hasattr(routes_mod, "RULES"):
        return tuple(routes_mod.RULES)
    raise RuntimeError("tools.change_gate.routes has no RULES/rules")


def _host_dict() -> dict[str, Any]:
    from tools.change_gate.hostinfo import gfx_arch, models_dir, rocm_version

    return {
        "gfx": gfx_arch() or "",
        "rocm": rocm_version() or "",
        "models_dir": str(models_dir()),
    }


def _est_minutes(selected, by_id) -> float:
    total = 0.0
    for s in selected:
        if getattr(s, "status", "") != "selected":
            continue
        route = by_id.get(getattr(s, "route_id", ""))
        if route is not None:
            total += float(route.est_minutes)
    return total


def _write_text(path: str | None, text: str) -> None:
    """Write ``text`` to ``path``.

    ``-`` means stdout (the conventional CLI idiom, and what the PR template
    tells contributors to use to paste telemetry inline). Both ``plan`` and
    ``run`` already echo the markdown, so a ``-`` target is a no-op here rather
    than a duplicate dump.
    """
    if not path or path == "-":
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _build_common_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog)
    p.add_argument(
        "--base",
        default="origin/beta",
        help="git base ref (default: origin/beta)",
    )
    p.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="budget cap in minutes (selector may trim)",
    )
    p.add_argument(
        "--include-heavy",
        action="store_true",
        help="allow heavy-tier routes",
    )
    p.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="write report JSON to path",
    )
    p.add_argument(
        "--md",
        dest="md_out",
        default=None,
        help="write markdown report to path",
    )
    return p


def cmd_routes(_argv: Sequence[str]) -> int:
    by_id = _routes_by_id()
    routes = sorted(by_id.values(), key=lambda r: r.id)
    print(f"{'ID':<42} {'KIND':<10} {'TIER':<10} {'EST':>6}  WHY")
    print("-" * 100)
    for r in routes:
        print(
            f"{r.id:<42} {r.kind:<10} {r.tier:<10} {r.est_minutes:>6.1f}  {r.why}"
        )
    print(f"\n{len(routes)} routes")
    return 0


def cmd_plan(argv: Sequence[str]) -> int:
    parser = _build_common_parser("python3 -m tools.change_gate plan")
    args = parser.parse_args(list(argv))

    from tools.change_gate.hostinfo import gfx_arch, models_dir
    from tools.change_gate.report import build_report, render_markdown, to_json
    from tools.change_gate.selector import changed_files, select

    paths, base_sha, head_sha, dirty = changed_files(args.base)
    by_id = _routes_by_id()
    rules = _rules()
    host = _host_dict()
    selected, not_run = select(
        paths,
        by_id,
        rules,
        gfx=gfx_arch(),
        models_dir=models_dir(),
        max_minutes=args.max_minutes,
        include_heavy=bool(args.include_heavy),
    )
    est = _est_minutes(selected, by_id)
    report = build_report(
        base=base_sha,
        head=head_sha,
        dirty=dirty,
        host=host,
        changed_files=paths,
        selected=selected,
        not_run=not_run,
        results=(),
        est_minutes=est,
    )
    md = render_markdown(report)
    if args.json_out:
        _write_text(args.json_out, to_json(report))
    if args.md_out:
        _write_text(args.md_out, md)
    print(md)
    print(
        f"plan: {len(selected)} selected, {len(not_run)} not_run, "
        f"est_minutes={est:.1f}, verdict={report['verdict']}"
    )
    # plan never executes and always exits 0 when the plan itself succeeded
    return 0


def cmd_run(argv: Sequence[str]) -> int:
    parser = _build_common_parser("python3 -m tools.change_gate run")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="artifact directory (default: temp change_gate-*)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and record argv without executing",
    )
    args = parser.parse_args(list(argv))

    from tools.change_gate.hostinfo import gfx_arch, models_dir
    from tools.change_gate.report import build_report, render_markdown, to_json
    from tools.change_gate.runner import run_routes
    from tools.change_gate.selector import changed_files, select

    paths, base_sha, head_sha, dirty = changed_files(args.base)
    by_id = _routes_by_id()
    rules = _rules()
    host = _host_dict()
    selected, not_run = select(
        paths,
        by_id,
        rules,
        gfx=gfx_arch(),
        models_dir=models_dir(),
        max_minutes=args.max_minutes,
        include_heavy=bool(args.include_heavy),
    )
    est = _est_minutes(selected, by_id)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(tempfile.mkdtemp(prefix="change_gate-"))
    )

    t0 = time.monotonic()
    results = run_routes(
        selected,
        by_id,
        out_dir=out_dir,
        dry_run=bool(args.dry_run),
        env={"HIPFIRE_MODELS_DIR": str(host.get("models_dir") or "")},
    )
    wall = time.monotonic() - t0

    report = build_report(
        base=base_sha,
        head=head_sha,
        dirty=dirty,
        host=host,
        changed_files=paths,
        selected=selected,
        not_run=not_run,
        results=results,
        est_minutes=est,
    )
    report["totals"]["actual_s"] = max(
        float(report["totals"].get("actual_s") or 0.0), wall
    )

    md = render_markdown(report)
    json_path = args.json_out or str(out_dir / "report.json")
    md_path = args.md_out or str(out_dir / "report.md")
    _write_text(json_path, to_json(report))
    _write_text(md_path, md)
    print(md)
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    print(f"verdict={report['verdict']}")
    return _exit_for_verdict(str(report["verdict"]))


def main(argv: list[str] | None = None) -> int:
    _silence_broken_pipe()
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _usage()
        return 0 if args and args[0] in {"-h", "--help"} else 3

    command, rest = args[0], args[1:]
    try:
        if command == "plan":
            return cmd_plan(rest)
        if command == "run":
            return cmd_run(rest)
        if command == "routes":
            return cmd_routes(rest)
    except SystemExit as exc:
        code = exc.code
        if code in (None, 0):
            return 0
        if isinstance(code, int):
            # argparse uses 2 for usage errors; contract wants 3
            return 3 if code == 2 else code
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"tools.change_gate: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    print(
        f"tools.change_gate: unknown subcommand {command!r} "
        f"(expected plan, run, or routes)",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
