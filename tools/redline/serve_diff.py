# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tools.redline.product_bench import (
    CERTIFIED_PM4_POLICY,
    REPO,
    _allocate_loopback_port,
    _kill_serve_process_group,
    _unique_coherence_daemon,
    _unique_smoke_dir,
    backend_config_value,
    collect_route_proof_evidence,
    load_pm4_multiturn_session,
    validate_coherence_route_evidence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m tools.redline serve-diff",
        description="Compare one sampled multi-turn serve session over HIP and retained PM4.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--daemon", default=str(REPO / "target/release/examples/daemon"))
    parser.add_argument("--cli", default=str(REPO / "target/release/hipfire"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--kv", choices=("q8", "fwht2", "fwht3", "fwht4"), default="q8")
    parser.add_argument("--thinking", default="med")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--sampling", default="registry")
    parser.add_argument("--seed", type=int, default=305419896)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--work-dir", default=str(REPO / ".redline-work/serve-diff"))
    parser.add_argument("--out", required=True)
    return parser


def _validate_arm(turns: list[dict], rows: object, backend: str, route: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(rows, list):
        return [f"{backend}: harness output must be a list"]
    if len(rows) != len(turns):
        errors.append(f"{backend}: expected {len(turns)} turns, got {len(rows)}")
        return errors
    successful_rows: list[dict] = []
    for index, (turn, row) in enumerate(zip(turns, rows), 1):
        label = f"{backend} turn {index}"
        if not isinstance(row, dict):
            errors.append(f"{label}: harness row must be an object")
            continue
        if row.get("finish") != "stop":
            errors.append(f"{label}: finish must be 'stop', got {row.get('finish')!r}")
        content = row.get("assistant_content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"{label}: assistant_content must be nonempty")
            content = ""
        for flag in ("empty", "runaway", "attractor"):
            if row.get(flag):
                errors.append(f"{label}: {flag} generation")
        for needle in turn.get("expect", []):
            if needle.lower() not in content.lower():
                errors.append(f"{label}: answer missing expected substring {needle!r}")
        successful_rows.append(row)
    route_result = validate_coherence_route_evidence(
        backend, "pm4", route, rows=successful_rows
    )
    errors.extend(f"{backend}: {error}" for error in route_result["errors"])
    return errors


def validate_comparison(
    turns: list[dict],
    hip_rows: object,
    pm4_rows: object,
    hip_route: dict,
    pm4_route: dict,
) -> dict:
    errors: list[str] = []
    if len(turns) != 8:
        errors.append(f"session must contain exactly 8 turns, got {len(turns)}")
    errors.extend(_validate_arm(turns, hip_rows, "hip", hip_route))
    errors.extend(_validate_arm(turns, pm4_rows, "auto", pm4_route))
    matched = 0
    if isinstance(hip_rows, list) and isinstance(pm4_rows, list):
        for index, (hip, pm4) in enumerate(zip(hip_rows, pm4_rows), 1):
            if not isinstance(hip, dict) or not isinstance(pm4, dict):
                continue
            turn_matches = True
            if hip.get("assistant_content") != pm4.get("assistant_content"):
                errors.append(f"turn {index}: sampled output differs between HIP and PM4")
                turn_matches = False
            if hip.get("ctx") != pm4.get("ctx"):
                errors.append(f"turn {index}: prompt-token count differs between HIP and PM4")
                turn_matches = False
            if hip.get("gen") != pm4.get("gen"):
                errors.append(f"turn {index}: generated-token count differs between HIP and PM4")
                turn_matches = False
            matched += int(turn_matches)
    return {
        "valid": not errors,
        "errors": errors,
        "turns": len(turns),
        "matched_turns": matched,
    }


def _run_arm(args: argparse.Namespace, backend: str) -> dict:
    work_dir = _unique_smoke_dir(Path(args.work_dir), f"serve-diff-{backend}")
    daemon = _unique_coherence_daemon(Path(args.daemon).resolve(), work_dir, backend)
    out_path = work_dir / "harness.json"
    serve_log = work_dir / "serve.log"
    home = work_dir / "home"
    pid_path = work_dir / "serve.pid"
    port = _allocate_loopback_port()
    argv = [
        sys.executable,
        "-m",
        "tools.serve_harness",
        "--model",
        str(Path(args.model).expanduser().resolve()),
        "--kv",
        args.kv,
        "--mtp",
        "off",
        "--thinking",
        args.thinking,
        "--max-tokens",
        str(args.max_tokens),
        "--max-seq",
        str(args.max_seq),
        "--sampling",
        args.sampling,
        "--mode",
        "session",
        "--session",
        str(Path(args.session).expanduser().resolve()),
        "--seed",
        str(args.seed),
        "--port",
        str(port),
        "--home",
        str(home),
        "--serve-log",
        str(serve_log),
        "--out",
        str(out_path),
        "--replay-route-proof-log",
    ]
    env = dict(os.environ)
    env.update(CERTIFIED_PM4_POLICY)
    env.update(
        HIP_VISIBLE_DEVICES=str(args.device),
        ROCR_VISIBLE_DEVICES=str(args.device),
        HIPFIRE_CLI_BIN=str(Path(args.cli).resolve()),
        HIPFIRE_DAEMON_BIN=str(daemon),
        HIPFIRE_REPLAY_BACKEND=backend_config_value(backend),
        HIPFIRE_REPLAY_TRANSPORT="pm4",
        HIPFIRE_KV_MODE=args.kv,
        HIPFIRE_CASK_OFF="1",
        HIPFIRE_AR_GRAPH="1",
        HIPFIRE_GRAPH="1",
        HIPFIRE_SERVE_HARNESS_PID_FILE=str(pid_path),
    )
    env.pop("HIPFIRE_REPLAY_MANUAL_CAPTURE", None)
    env.pop("HIPFIRE_HOME", None)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as error:
        cleanup = _kill_serve_process_group(pid_path)
        detail = f"{backend} serve session timed out after {args.timeout}s"
        if cleanup:
            detail += f"; cleanup failed: {'; '.join(cleanup)}"
        raise RuntimeError(detail) from error
    finally:
        try:
            daemon.unlink()
        except OSError:
            pass
    rows = None
    if out_path.is_file():
        rows = json.loads(out_path.read_text(encoding="utf-8"))
    route = collect_route_proof_evidence(serve_log, proc.stdout, proc.stderr)
    return {
        "backend": backend,
        "seconds": time.monotonic() - started,
        "returncode": proc.returncode,
        "rows": rows,
        "route_proof": route,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": argv,
        "work_dir": str(work_dir),
        "serve_log": str(serve_log),
    }


def _average_decode(rows: object) -> float | None:
    if not isinstance(rows, list):
        return None
    values = [row.get("decode_tok_s") for row in rows if isinstance(row, dict)]
    values = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session_path, turns = load_pm4_multiturn_session(args.session)
    if len(turns) != 8:
        raise SystemExit(f"serve-diff requires exactly 8 session turns, got {len(turns)}")
    if args.thinking == "med" and args.max_tokens <= 2048:
        raise SystemExit("--max-tokens must exceed the 2048-token med thinking cap")
    for path, label in ((args.model, "model"), (args.daemon, "daemon"), (args.cli, "CLI")):
        if not Path(path).expanduser().is_file():
            raise SystemExit(f"{label} not found: {path}")

    print(
        f"serve-diff: 8 turns, sampling={args.sampling}, seed={args.seed}, "
        f"thinking={args.thinking}, max_tokens={args.max_tokens}, max_seq={args.max_seq}",
        flush=True,
    )
    hip = _run_arm(args, "hip")
    print(f"serve-diff: HIP arm finished in {hip['seconds']:.1f}s", flush=True)
    pm4 = _run_arm(args, "auto")
    print(f"serve-diff: PM4 arm finished in {pm4['seconds']:.1f}s", flush=True)

    comparison = validate_comparison(
        turns,
        hip["rows"],
        pm4["rows"],
        hip["route_proof"],
        pm4["route_proof"],
    )
    if hip["returncode"] != 0:
        comparison["errors"].append(f"HIP serve harness exited {hip['returncode']}")
    if pm4["returncode"] != 0:
        comparison["errors"].append(f"PM4 serve harness exited {pm4['returncode']}")
    comparison["valid"] = not comparison["errors"]

    report = {
        "valid": comparison["valid"],
        "config": {
            "model": str(Path(args.model).expanduser().resolve()),
            "session": str(session_path),
            "turns": len(turns),
            "device": str(args.device),
            "kv": args.kv,
            "thinking": args.thinking,
            "max_tokens": args.max_tokens,
            "max_seq": args.max_seq,
            "sampling": args.sampling,
            "seed": args.seed,
        },
        "comparison": comparison,
        "hip": hip,
        "pm4": pm4,
        "average_decode_tok_s": {
            "hip": _average_decode(hip["rows"]),
            "pm4": _average_decode(pm4["rows"]),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    hip_rows = hip["rows"] if isinstance(hip["rows"], list) else []
    pm4_rows = pm4["rows"] if isinstance(pm4["rows"], list) else []
    for index, (hip_row, pm4_row) in enumerate(zip(hip_rows, pm4_rows), 1):
        print(
            f"  t{index}: ctx={hip_row.get('ctx')} gen={hip_row.get('gen')} "
            f"HIP={hip_row.get('decode_tok_s')} PM4={pm4_row.get('decode_tok_s')} "
            f"match={hip_row.get('assistant_content') == pm4_row.get('assistant_content')}",
            flush=True,
        )
    if comparison["valid"]:
        print(
            f"serve-diff: PASS — {comparison['matched_turns']}/8 sampled turns exact; "
            f"report={out_path}",
            flush=True,
        )
        return 0
    for error in comparison["errors"]:
        print(f"serve-diff: ERROR: {error}", file=sys.stderr)
    print(f"serve-diff: FAIL — report={out_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
