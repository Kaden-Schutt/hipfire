# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Execute selected change-gate routes and collect RouteResult rows."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.change_gate.model import Route, RouteResult, Selection

_DETECT_KINDS = frozenset({"serve", "detect"})
_JSON_HINT_RE = re.compile(r"(serve_harness|redline_daemon_harness)")


def _safe_name(route_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", route_id)


def _timeout_s(est_minutes: float) -> float:
    """Generous multiplier: 3× estimate with a 5-minute floor."""
    try:
        est = float(est_minutes)
    except (TypeError, ValueError):
        est = 1.0
    if est < 0:
        est = 0.0
    return max(5.0 * 60.0, est * 60.0 * 3.0)


def _first_model(route: Route, env: Mapping[str, str] | None) -> str:
    models = tuple(route.models or ())
    if not models:
        return ""
    if env and env.get("HIPFIRE_MODELS_DIR"):
        models_dir = env["HIPFIRE_MODELS_DIR"]
    elif env and env.get("MODELS_DIR"):
        models_dir = env["MODELS_DIR"]
    else:
        try:
            from tools.change_gate.hostinfo import models_dir as _models_dir

            models_dir = str(_models_dir())
        except Exception:  # noqa: BLE001
            models_dir = os.environ.get(
                "HIPFIRE_MODELS_DIR",
                str(Path.home() / ".hipfire" / "models"),
            )
    name = models[0]
    candidate = Path(models_dir) / name
    if candidate.exists():
        return str(candidate)
    return name


def _substitute(argv: Sequence[str], *, model: str, out: str) -> list[str]:
    return [
        str(part).replace("{model}", model).replace("{out}", out) for part in argv
    ]


def _load_json_file(path: Path) -> tuple[Any | None, str | None]:
    if not path.is_file():
        return None, f"missing json out file: {path}"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, f"read failed: {exc}"
    text_stripped = text.strip()
    if not text_stripped:
        return None, "empty json out file"
    try:
        return json.loads(text_stripped), None
    except json.JSONDecodeError as exc:
        tail = text_stripped[-2000:] if len(text_stripped) > 2000 else text_stripped
        return None, f"json parse error: {exc}; tail={tail!r}"


def _extract_generation_text(payload: Any) -> str:
    """Pull assistant-facing text from harness JSON for the detector bridge."""
    chunks: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in (
                "assistant_content",
                "content",
                "text",
                "output",
                "ans_preview",
                "completion",
            ):
                val = node.get(key)
                if isinstance(val, str) and val.strip():
                    chunks.append(val)
            for val in node.values():
                walk(val)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    uniq: list[str] = []
    seen: set[str] = set()
    for c in sorted(chunks, key=len, reverse=True):
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
        if len(uniq) >= 8:
            break
    return "\n\n".join(uniq)


def _harness_pass(payload: Any) -> bool | None:
    """Interpret harness JSON pass/fail when present. None = unknown."""
    if isinstance(payload, dict):
        if "pass" in payload:
            return bool(payload.get("pass"))
        if "verdict" in payload:
            v = str(payload.get("verdict")).lower()
            if v in {"pass", "ok", "passed"}:
                return True
            if v in {"fail", "failed", "error"}:
                return False
        return None
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            if row.get("empty") or row.get("attractor") or row.get("runaway"):
                return False
        return None
    return None


def _should_detect(route: Route, argv: Sequence[str]) -> bool:
    if route.kind in _DETECT_KINDS:
        return True
    return bool(_JSON_HINT_RE.search(" ".join(argv)))


def _run_one(
    selection: Selection,
    route: Route,
    *,
    out_dir: Path,
    dry_run: bool,
    env: Mapping[str, str] | None,
) -> RouteResult:
    route_id = selection.route_id or route.id
    safe = _safe_name(route_id)
    route_dir = out_dir / safe
    try:
        route_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        route_dir = out_dir

    log_path = route_dir / "route.log"
    json_out = route_dir / "out.json"
    model = _first_model(route, env)
    argv = _substitute(route.argv, model=model, out=str(json_out))
    timeout = _timeout_s(route.est_minutes)
    artifacts: list[str] = [str(log_path)]

    if dry_run:
        return RouteResult(
            route_id=route_id,
            status="skipped",
            duration_s=0.0,
            verdict={"dry_run": True, "argv": argv},
            artifacts=tuple(artifacts),
        )

    if selection.status != "selected":
        return RouteResult(
            route_id=route_id,
            status="skipped",
            duration_s=0.0,
            verdict={
                "skipped": True,
                "selection_status": selection.status,
                "detail": selection.detail,
            },
            artifacts=tuple(artifacts),
        )

    run_env = os.environ.copy()
    if env:
        run_env.update({str(k): str(v) for k, v in env.items()})

    t0 = time.monotonic()
    log_fp = None
    try:
        log_fp = open(log_path, "w", encoding="utf-8", errors="replace")
        log_fp.write(f"$ {' '.join(argv)}\n")
        log_fp.flush()
        proc = subprocess.run(
            argv,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env=run_env,
            timeout=timeout,
            check=False,
        )
        duration = time.monotonic() - t0
        rc = int(proc.returncode)
        log_fp.write(f"\n# exit={rc} duration_s={duration:.3f}\n")
        log_fp.flush()
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - t0
        if log_fp is not None:
            try:
                log_fp.write(
                    f"\n# TIMEOUT after {duration:.3f}s (limit={timeout:.1f}s)\n"
                )
                log_fp.flush()
            except OSError:
                pass
        return RouteResult(
            route_id=route_id,
            status="fail",
            duration_s=duration,
            verdict={
                "error": "timeout",
                "timeout_s": timeout,
                "detail": str(exc),
                "argv": argv,
            },
            artifacts=tuple(artifacts),
        )
    except FileNotFoundError as exc:
        duration = time.monotonic() - t0
        return RouteResult(
            route_id=route_id,
            status="fail",
            duration_s=duration,
            verdict={
                "error": "executable_not_found",
                "detail": str(exc),
                "argv": argv,
            },
            artifacts=tuple(artifacts),
        )
    except Exception as exc:  # noqa: BLE001 — never abort the batch
        duration = time.monotonic() - t0
        return RouteResult(
            route_id=route_id,
            status="fail",
            duration_s=duration,
            verdict={
                "error": "runner_exception",
                "detail": f"{type(exc).__name__}: {exc}",
                "argv": argv,
            },
            artifacts=tuple(artifacts),
        )
    finally:
        if log_fp is not None:
            try:
                log_fp.close()
            except OSError:
                pass

    verdict: dict[str, Any] = {"returncode": rc, "argv": argv}
    status = "pass" if rc == 0 else "fail"

    expects_json = "{out}" in " ".join(route.argv) or json_out.is_file()
    payload = None
    if expects_json and json_out.is_file():
        artifacts.append(str(json_out))
        payload, parse_err = _load_json_file(json_out)
        if parse_err is not None:
            try:
                raw_tail = json_out.read_text(encoding="utf-8", errors="replace")[-2000:]
            except OSError:
                raw_tail = ""
            verdict["json_error"] = parse_err
            verdict["raw_tail"] = raw_tail
            return RouteResult(
                route_id=route_id,
                status="fail",
                duration_s=duration,
                verdict=verdict,
                artifacts=tuple(artifacts),
            )
        verdict["harness"] = payload
        harness_ok = _harness_pass(payload)
        if harness_ok is False:
            status = "fail"
            verdict["harness_pass"] = False
        elif harness_ok is True:
            verdict["harness_pass"] = True

    if _should_detect(route, argv):
        try:
            from tools.change_gate.detect import analyse
        except Exception as exc:  # noqa: BLE001
            verdict["detect"] = {
                "available": False,
                "verdict": "unknown",
                "error": f"import_failed: {type(exc).__name__}: {exc}",
            }
            if status == "pass":
                status = "blocked"
                verdict["error"] = "detect_unavailable"
        else:
            text = ""
            if payload is not None:
                text = _extract_generation_text(payload)
            if not text:
                try:
                    text = log_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    text = ""
            try:
                det = analyse(text or None)
            except Exception as exc:  # noqa: BLE001
                det = {
                    "available": False,
                    "verdict": "unknown",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            verdict["detect"] = det
            det_verdict = str((det or {}).get("verdict") or "unknown").lower()
            if det_verdict == "fail":
                status = "fail"
            elif det_verdict == "unknown" and not (det or {}).get("available", True):
                if status == "pass":
                    status = "blocked"
                    verdict["error"] = "detect_unavailable"

    if rc != 0:
        status = "fail"

    # de-dupe artifacts, preserve order
    art = tuple(dict.fromkeys(artifacts))
    return RouteResult(
        route_id=route_id,
        status=status,
        duration_s=duration,
        verdict=verdict,
        artifacts=art,
    )


def run_routes(
    selections: Sequence[Selection],
    routes_by_id: Mapping[str, Route],
    *,
    out_dir: str | Path,
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
) -> list[RouteResult]:
    """Execute each *selected* route; never raise for a single route failure."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: list[RouteResult] = []
    for sel in selections:
        if sel.status != "selected":
            results.append(
                RouteResult(
                    route_id=sel.route_id,
                    status="skipped",
                    duration_s=0.0,
                    verdict={
                        "skipped": True,
                        "selection_status": sel.status,
                        "detail": sel.detail,
                    },
                    artifacts=(),
                )
            )
            continue

        route = routes_by_id.get(sel.route_id)
        if route is None:
            results.append(
                RouteResult(
                    route_id=sel.route_id,
                    status="fail",
                    duration_s=0.0,
                    verdict={
                        "error": "unknown_route",
                        "detail": f"route id not in manifest: {sel.route_id}",
                    },
                    artifacts=(),
                )
            )
            continue

        try:
            result = _run_one(
                sel,
                route,
                out_dir=out_path,
                dry_run=dry_run,
                env=env,
            )
        except Exception as exc:  # noqa: BLE001 — batch must complete
            result = RouteResult(
                route_id=sel.route_id,
                status="fail",
                duration_s=0.0,
                verdict={
                    "error": "runner_exception",
                    "detail": f"{type(exc).__name__}: {exc}",
                },
                artifacts=(),
            )
        results.append(result)
    return results
