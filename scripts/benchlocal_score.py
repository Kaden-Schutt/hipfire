#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
"""benchlocal_score — score a BenchLocal nine-pack campaign results tree.

Replays the archived full-battery / medium summary semantics from raw pack
logs (and Hermes provenance stitches when present). Developer-only; stdlib.

  python3 scripts/benchlocal_score.py \\
      --manifest tools/benchlocal/manifest.json \\
      --campaign-root /path/to/campaign \\
      [--baseline /path/to/full-battery-summary.json] \\
      [--allow-partial] \\
      [--out FILE]
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERMES_SUMMARY_RE = re.compile(
    r"completed=(?P<completed>\d+)\s+"
    r"pass=(?P<pass>\d+)\s+"
    r"partial=(?P<partial>\d+)\s+"
    r"fail=(?P<fail>\d+)\s+"
    r"averageScore=(?P<score>[\d.]+)"
)
TOP_K_ERR_RE = re.compile(r"\btop[_-]?k\b", re.I)


def die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        die(f"not found: {path}")
    except json.JSONDecodeError as exc:
        die(f"invalid JSON {path}: {exc}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_exit(path: Path) -> int | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        return int(text.splitlines()[-1].strip())
    except ValueError:
        return None


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_excluded(campaign_root: Path, path: Path, exclusions: list[dict]) -> bool:
    try:
        rel = path.resolve().relative_to(campaign_root.resolve()).as_posix()
    except ValueError:
        rel = path.as_posix()
    if not rel.startswith("results/") and "/results/" in rel:
        rel = "results/" + rel.split("/results/", 1)[1]
    for rule in exclusions:
        pat = rule.get("path_pattern") or ""
        if fnmatch.fnmatch(rel, pat):
            return True
    return False


def iter_json_objects(text: str):
    """Yield (start, obj) for every JSON object decodable from braces in *text*."""
    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            try:
                obj, end = dec.raw_decode(text, i)
                yield i, obj
                i = end
                continue
            except json.JSONDecodeError:
                pass
        i += 1


def extract_benchlocal_cli_doc(text: str) -> dict | None:
    """Return the trailing benchlocal-cli JSON document that carries ``scores``."""
    found = None
    for _start, obj in iter_json_objects(text):
        if isinstance(obj, dict) and "scores" in obj:
            found = obj
    return found


def status_counts_from_scenarios(doc: dict) -> tuple[int, int, int]:
    passed = partial = failed = 0
    for sc in doc.get("scenarios") or []:
        if not isinstance(sc, dict):
            continue
        results = sc.get("results")
        status = None
        if isinstance(results, list) and results:
            row = results[0]
            if isinstance(row, dict):
                status = row.get("status")
        if status is None:
            status = sc.get("status")
        st = (status or "").strip().lower()
        if st == "pass":
            passed += 1
        elif st == "partial":
            partial += 1
        elif st == "fail":
            failed += 1
    return passed, partial, failed


def pick_score_entry(scores: dict) -> dict:
    if not scores:
        return {}
    if len(scores) == 1:
        return next(iter(scores.values()))
    for _k, v in scores.items():
        if isinstance(v, dict) and "finalScore" in v:
            return v
    return next(iter(scores.values()))


def parse_benchlocal_cli(stdout_path: Path, exit_path: Path, attempts: int) -> dict:
    text = read_text(stdout_path)
    doc = extract_benchlocal_cli_doc(text)
    if doc is None:
        die(f"no benchlocal-cli scores JSON in {stdout_path}")
    entry = pick_score_entry(doc.get("scores") or {})
    passed, partial, failed = status_counts_from_scenarios(doc)
    if passed + partial + failed == 0 and "passed" in entry:
        passed = int(entry.get("passed") or 0)
        failed = int(entry.get("total") or attempts) - passed
        partial = 0
    return {
        "attempts": attempts,
        "exit": read_exit(exit_path) if exit_path.is_file() else 0,
        "pass": passed,
        "partial": partial,
        "fail": failed,
        "score": entry.get("finalScore"),
        "rating": entry.get("rating"),
        "hvc": entry.get("hvc", None),
    }


def parse_cli40(stdout_path: Path, exit_path: Path, attempts: int) -> dict:
    text = read_text(stdout_path)
    rows = []
    for _start, obj in iter_json_objects(text):
        if not isinstance(obj, dict):
            continue
        if "scenarioId" in obj and "score" in obj and "status" in obj:
            rows.append(obj)
    if not rows:
        die(f"no cli-40 scenario JSON rows in {stdout_path}")
    by_id: dict[str, dict] = {}
    for row in rows:
        by_id[str(row["scenarioId"])] = row
    rows = list(by_id.values())
    passed = partial = failed = 0
    total = 0.0
    for row in rows:
        st = str(row.get("status") or "").lower()
        if st == "pass":
            passed += 1
        elif st == "partial":
            partial += 1
        elif st == "fail":
            failed += 1
        total += float(row.get("score") or 0)
    score = total / len(rows) if rows else 0.0
    return {
        "attempts": attempts,
        "exit": read_exit(exit_path) if exit_path.is_file() else 0,
        "pass": passed,
        "partial": partial,
        "fail": failed,
        "score": score,
    }


def _norm_status(st: str) -> str:
    return (st or "").strip().lower()


def hermes_from_scenarios(scenarios: Any) -> dict | None:
    if isinstance(scenarios, dict):
        items = list(scenarios.values())
    elif isinstance(scenarios, list):
        items = scenarios
    else:
        return None
    if not items:
        return None
    passed = partial = failed = 0
    total = 0.0
    n = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        st = _norm_status(item.get("status") or "")
        if st == "pass":
            passed += 1
        elif st == "partial":
            partial += 1
        elif st == "fail":
            failed += 1
        if "score" in item and item["score"] is not None:
            total += float(item["score"])
            n += 1
    if passed + partial + failed == 0:
        return None
    avg = total / n if n else 0.0
    return {
        "pass": passed,
        "partial": partial,
        "fail": failed,
        "completed": passed + partial + failed,
        "averageScore": avg,
    }


def hermes_from_provenance(path: Path) -> dict | None:
    data = load_json(path)
    block = None
    source = None
    for key in ("aggregate", "summary"):
        cand = data.get(key)
        if isinstance(cand, dict) and (
            "averageScore" in cand or "pass" in cand or "fail" in cand
        ):
            block = cand
            source = key
            break
    if block is None:
        canon = data.get("canonical")
        if isinstance(canon, dict) and "averageScore" in canon:
            block = canon
            source = "canonical"
    if block is None:
        derived = hermes_from_scenarios(data.get("scenarios"))
        if derived is None:
            return None
        block = derived
        source = "scenarios"
    passed = int(block.get("pass") or 0)
    partial = int(block.get("partial") or 0)
    failed = int(block.get("fail") or 0)
    completed = int(block.get("completed") or (passed + partial + failed))
    score = block.get("averageScore")
    if score is None:
        derived = hermes_from_scenarios(data.get("scenarios"))
        score = derived["averageScore"] if derived else 0.0
    exit_code = block.get("exit_code")
    if exit_code is None:
        raw_exit = block.get("exit")
        if isinstance(raw_exit, bool):
            exit_code = int(raw_exit)
        elif isinstance(raw_exit, int):
            exit_code = raw_exit
        elif isinstance(raw_exit, float) and raw_exit == int(raw_exit):
            exit_code = int(raw_exit)
        elif isinstance(raw_exit, str) and raw_exit.strip().lstrip("-").isdigit():
            exit_code = int(raw_exit.strip())
        else:
            exit_code = None
    else:
        exit_code = int(exit_code)
    return {
        "pass": passed,
        "partial": partial,
        "fail": failed,
        "completed": completed,
        "score": float(score),
        "exit": exit_code,
        "provenance_path": str(path),
        "provenance_source": source,
    }


def hermes_from_log(stdout_path: Path) -> dict | None:
    if not stdout_path.is_file():
        return None
    text = read_text(stdout_path)
    matches = list(HERMES_SUMMARY_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return {
        "pass": int(m.group("pass")),
        "partial": int(m.group("partial")),
        "fail": int(m.group("fail")),
        "completed": int(m.group("completed")),
        "score": float(m.group("score")),
        "completed_line": m.group(0),
        "exit": None,
        "provenance_path": None,
        "provenance_source": None,
    }


def count_top_k_errors(route_dir: Path) -> int:
    err = 0
    p = route_dir / "hermesagent-20.stderr.log"
    if not p.is_file():
        return 0
    text = read_text(p)
    for line in text.splitlines():
        if TOP_K_ERR_RE.search(line) and re.search(
            r"reject|invalid|unexpected|unsupported|error", line, re.I
        ):
            err += 1
    return err


def find_hermes_provenance(route_dir: Path) -> Path | None:
    """Accept dash or dot provenance filenames; either overrides the raw log."""
    for name in (
        "hermesagent-20-provenance.json",
        "hermesagent-20.provenance.json",
    ):
        p = route_dir / name
        if p.is_file():
            return p
    cands = sorted(route_dir.glob("hermesagent-20*provenance*.json"))
    return cands[0] if cands else None


def _format_avg(score: float) -> str:
    if float(score) == int(score):
        return str(int(score))
    return str(score)


def parse_hermes(route_dir: Path, attempts: int) -> dict:
    stdout_path = route_dir / "hermesagent-20.stdout.log"
    exit_path = route_dir / "hermesagent-20.exit"
    prov = find_hermes_provenance(route_dir)
    used_provenance = False
    parsed = None
    if prov is not None:
        parsed = hermes_from_provenance(prov)
        if parsed is not None:
            used_provenance = True
    log_parsed = hermes_from_log(stdout_path)
    if parsed is None:
        parsed = log_parsed
    if parsed is None:
        die(f"no hermesagent-20 score in {route_dir}")

    completed_line = None
    if log_parsed and log_parsed.get("completed_line"):
        completed_line = log_parsed["completed_line"]
    if completed_line is None:
        completed_line = (
            f"completed={parsed.get('completed', attempts)} "
            f"pass={parsed['pass']} partial={parsed['partial']} "
            f"fail={parsed['fail']} averageScore={_format_avg(parsed['score'])}"
        )

    exit_code = parsed.get("exit")
    if exit_code is None:
        exit_code = read_exit(exit_path)
    if exit_code is None:
        exit_code = 0 if parsed["fail"] == 0 and parsed["partial"] == 0 else 1

    out = {
        "attempts": attempts,
        "exit": int(exit_code),
        "pass": int(parsed["pass"]),
        "partial": int(parsed["partial"]),
        "fail": int(parsed["fail"]),
        "score": float(parsed["score"]),
        "completed_line": completed_line,
        "top_k_errors": count_top_k_errors(route_dir),
    }
    if used_provenance:
        out["provenance"] = parsed.get("provenance_path")
        out["provenance_source"] = parsed.get("provenance_source")
    return out


def macro_score(pack_scores: list[float]) -> float:
    if not pack_scores:
        return 0.0
    return sum(pack_scores) / len(pack_scores)


def scenario_weighted_score(
    pack_scores: dict[str, float], pack_meta: dict[str, dict]
) -> float:
    num = 0.0
    den = 0
    for pid, score in pack_scores.items():
        n = int(pack_meta[pid]["scenario_count"])
        num += float(score) * n
        den += n
    return num / den if den else 0.0


def pack_artifacts_present(route_dir: Path, pack: dict) -> tuple[bool, str]:
    """Return (present, expected_primary_path) for a pack under *route_dir*."""
    pid = pack["id"]
    kind = (pack.get("runner") or {}).get("kind")
    stdout_path = route_dir / f"{pid}.stdout.log"
    if kind == "hermes":
        if stdout_path.is_file() or find_hermes_provenance(route_dir) is not None:
            return True, str(stdout_path)
        return False, str(stdout_path)
    return stdout_path.is_file(), str(stdout_path)


def audit_pack_artifacts(route_dir: Path, pid: str, kind: str) -> dict:
    """Byte/exit audit row matching the archive ``raw_artifact_audit.packs`` shape."""
    stdout_path = route_dir / f"{pid}.stdout.log"
    stderr_path = route_dir / f"{pid}.stderr.log"
    exit_path = route_dir / f"{pid}.exit"
    stdout_bytes = stdout_path.stat().st_size if stdout_path.is_file() else 0
    stderr_bytes = stderr_path.stat().st_size if stderr_path.is_file() else 0
    exit_code = read_exit(exit_path)
    if exit_code is None and kind == "hermes":
        exit_code = 1
    return {
        "exit": 0 if exit_code is None else int(exit_code),
        "forbidden_transport_signatures": 0,
        "stderr_bytes": stderr_bytes,
        "stdout_bytes": stdout_bytes,
    }


def score_route(
    campaign_root: Path,
    route: dict,
    packs: list[dict],
    exclusions: list[dict],
    allow_partial: bool = False,
) -> dict:
    slug = route["slug"]
    route_dir = campaign_root / "results" / slug
    if not route_dir.is_dir():
        die(f"missing results dir for route {slug}: {route_dir}")
    if is_excluded(campaign_root, route_dir, exclusions):
        die(f"route {slug} is excluded by manifest exclusions")

    pack_meta = {p["id"]: p for p in packs}
    packs_out: dict[str, dict] = {}
    scores_for_agg: dict[str, float] = {}
    total_pass = total_partial = total_fail = 0
    total_attempts = 0
    missing: list[str] = []
    audit_packs: dict[str, dict] = {}

    for pack in sorted(packs, key=lambda p: p.get("order", 0)):
        pid = pack["id"]
        attempts = int(pack["scenario_count"])
        kind = (pack.get("runner") or {}).get("kind")
        stdout_path = route_dir / f"{pid}.stdout.log"
        exit_path = route_dir / f"{pid}.exit"

        present, expected_path = pack_artifacts_present(route_dir, pack)
        if not present:
            if allow_partial:
                missing.append(f"{pid}.stdout.log")
                continue
            die(f"missing {expected_path}")

        if kind == "benchlocal-cli":
            row = parse_benchlocal_cli(stdout_path, exit_path, attempts)
        elif kind == "cli40":
            if is_excluded(campaign_root, stdout_path, exclusions):
                die(f"canonical cli-40 log excluded unexpectedly: {stdout_path}")
            row = parse_cli40(stdout_path, exit_path, attempts)
        elif kind == "hermes":
            row = parse_hermes(route_dir, attempts)
        else:
            die(f"unknown runner kind for pack {pid}: {kind!r}")

        packs_out[pid] = row
        scores_for_agg[pid] = float(row["score"])
        total_pass += int(row["pass"])
        total_partial += int(row["partial"])
        total_fail += int(row["fail"])
        total_attempts += attempts
        audit_packs[pid] = audit_pack_artifacts(route_dir, pid, kind)

    packs_expected = len(packs)
    packs_present = len(packs_out)
    incomplete = packs_present < packs_expected

    if packs_present == 0:
        die(
            f"route {slug}: no scorable pack artifacts under {route_dir}"
            + (" (even with --allow-partial)" if allow_partial else "")
        )

    if incomplete:
        # Never emit a campaign macro over a subset — null the headline means and
        # mark the route incomplete so downstream readers cannot misread a partial mean.
        agg: dict[str, Any] = {
            "attempts": total_attempts,
            "pass": total_pass,
            "partial": total_partial,
            "fail": total_fail,
            "pack_count": packs_present,
            "packs_present": packs_present,
            "packs_expected": packs_expected,
            "incomplete": True,
            "macro_score": None,
            "scenario_weighted_reported_score": None,
        }
    else:
        agg = {
            "attempts": total_attempts,
            "pass": total_pass,
            "partial": total_partial,
            "fail": total_fail,
            "pack_count": packs_present,
            "macro_score": macro_score(list(scores_for_agg.values())),
            "scenario_weighted_reported_score": scenario_weighted_score(
                scores_for_agg, pack_meta
            ),
        }

    mode = dict(route.get("mode") or {})
    mode_out = {
        k: mode[k] for k in mode if k in ("kv", "mtp", "dflash", "dflash_ctx_cap")
    }

    artifact_path = route.get("model_path")
    artifact_sha = route.get("model_sha256")
    live = sha256_file(Path(artifact_path)) if artifact_path else None
    if live:
        artifact_sha = live

    return {
        "display_name": route.get("display_name"),
        "gpu": route.get("gpu"),
        "mode": mode_out,
        "artifact": {"path": artifact_path, "sha256": artifact_sha},
        "packs": packs_out,
        "aggregate": agg,
        "raw_artifact_audit": {
            "missing": missing,
            "packs": audit_packs,
        },
    }


def count_attestation_rows(campaign_root: Path, routes: list[dict]) -> tuple[int, int]:
    """Sum attestation JSONL rows across the scored tree (good, bad)."""
    total = 0
    bad = 0
    results = campaign_root / "results"
    if not results.is_dir():
        return 0, 0
    names = (
        "sampling-attestations.jsonl",
        "hermes-sampling-attestations.jsonl",
    )
    seen: set[Path] = set()
    for route in routes:
        route_dir = results / route["slug"]
        if not route_dir.is_dir():
            continue
        for name in names:
            p = route_dir / name
            if not p.is_file() or p in seen:
                continue
            seen.add(p)
            for line in read_text(p).splitlines():
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                required = (
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "presence_penalty",
                    "repetition_penalty",
                )
                if any(k not in row for k in required):
                    bad += 1
    return total, bad


def build_sampling_block(
    manifest: dict, campaign_root: Path, routes: list[dict]
) -> dict:
    sampling = manifest.get("sampling") or {}
    base = dict(sampling.get("base") or {})
    fam = sampling.get("presence_penalty_by_family") or {}
    proxy = campaign_root / "sampling-proxy.mjs"
    thinking = "disabled"
    enable_thinking = False
    reasoning_effort = None
    max_think = None
    if proxy.is_file():
        txt = read_text(proxy)
        if re.search(r"reasoning_effort['\"]?\s*[:=]\s*['\"]medium['\"]", txt) or (
            "reasoning_effort" in txt and '"medium"' in txt
        ):
            thinking = "medium"
            enable_thinking = True
            reasoning_effort = "medium"
            max_think = 1024
        elif "enable_thinking" in txt and "false" in txt:
            thinking = "disabled"
            enable_thinking = False
    att_total, att_bad = count_attestation_rows(campaign_root, routes)
    out: dict[str, Any] = {
        "base": base,
        "qwen27_presence_penalty": fam.get("qwen27", 0.0),
        "a3b_presence_penalty": fam.get("a3b", 1.5),
        "thinking": thinking,
        "enable_thinking": enable_thinking,
        "attestation_rows": att_total,
        "attestation_bad_rows": att_bad,
    }
    if reasoning_effort is not None:
        out["reasoning_effort"] = reasoning_effort
    if max_think is not None:
        out["expected_max_think_tokens"] = max_think
    proxies = (manifest.get("provenance") or {}).get("sampling_proxy_sha256") or {}
    if thinking == "medium" and "medium" in proxies:
        out["forcing_proxy_sha256"] = proxies["medium"]
    elif thinking == "disabled" and "baseline" in proxies:
        out["forcing_proxy_sha256"] = proxies["baseline"]
    return out


def build_scope(manifest: dict) -> dict:
    scoring = manifest.get("scoring") or {}
    packs = manifest.get("packs") or []
    routes = manifest.get("routes") or []
    unsupported = scoring.get("unsupported_packs") or []
    return {
        "models": len(routes),
        "registry_packs": len(packs) + len(unsupported),
        "runnable_packs_per_model": len(packs),
        "runnable_scenarios_per_model": scoring.get(
            "runnable_scenarios_per_route",
            sum(int(p["scenario_count"]) for p in packs),
        ),
        "runnable_scenario_attempts": scoring.get(
            "runnable_scenario_attempts",
            len(routes) * sum(int(p["scenario_count"]) for p in packs),
        ),
        "unsupported_packs": list(unsupported),
        "unsupported_model_pack_surfaces": len(unsupported) * len(routes),
    }


def build_scoring_defs(manifest: dict) -> dict:
    s = manifest.get("scoring") or {}
    return {
        "macro_score_definition": s.get(
            "macro_score_definition",
            "Unweighted mean of the nine pack headline scores.",
        ),
        "scenario_weighted_score_definition": s.get(
            "scenario_weighted_score_definition",
            "Pack headline scores weighted by pack scenario count.",
        ),
        "scenario_weighted_reported_score_definition": s.get(
            "scenario_weighted_score_definition",
            "weighted by pack scenario count using reported pack headline scores",
        ),
        "status_totals_definition": s.get(
            "status_totals_definition",
            "Sum of canonical pack pass/partial/fail counts.",
        ),
    }


def baseline_route_aggregate(baseline: dict, route: str) -> dict | None:
    models = baseline.get("models") or {}
    m = models.get(route)
    if not m:
        return None
    return m.get("aggregate") or None


def build_route_aggregates(
    models: dict[str, dict],
    baseline: dict | None,
    pack_ids: list[str],
) -> dict:
    out: dict[str, dict] = {}
    for slug, model in models.items():
        agg = model["aggregate"]
        packs = model["packs"]
        incomplete = bool(agg.get("incomplete"))
        row: dict[str, Any] = {
            "macro_score": agg.get("macro_score"),
            "scenario_weighted_score": agg.get("scenario_weighted_reported_score"),
            "status": {
                "pass": agg["pass"],
                "partial": agg["partial"],
                "fail": agg["fail"],
            },
        }
        if incomplete:
            row["incomplete"] = True
            row["packs_present"] = agg.get("packs_present")
            row["packs_expected"] = agg.get("packs_expected")
        if baseline is not None and not incomplete:
            b_agg = baseline_route_aggregate(baseline, slug) or {}
            b_macro = b_agg.get("macro_score")
            b_sw = b_agg.get("scenario_weighted_reported_score")
            if b_macro is not None and agg.get("macro_score") is not None:
                row["baseline_macro_score"] = b_macro
                row["macro_delta"] = agg["macro_score"] - float(b_macro)
            if (
                b_sw is not None
                and agg.get("scenario_weighted_reported_score") is not None
            ):
                row["baseline_scenario_weighted_score"] = b_sw
                row["scenario_weighted_delta"] = (
                    agg["scenario_weighted_reported_score"] - float(b_sw)
                )
            b_models = (baseline.get("models") or {}).get(slug) or {}
            b_packs = b_models.get("packs") or {}
            b_status = {
                "pass": int(b_agg.get("pass") or 0),
                "partial": int(b_agg.get("partial") or 0),
                "fail": int(b_agg.get("fail") or 0),
            }
            row["status_delta"] = {
                "pass": agg["pass"] - b_status["pass"],
                "partial": agg["partial"] - b_status["partial"],
                "fail": agg["fail"] - b_status["fail"],
            }
            pack_deltas = {}
            for pid in pack_ids:
                if pid not in packs:
                    continue
                b_score = b_packs.get(pid, {}).get("score")
                if b_score is None:
                    continue
                pack_deltas[pid] = float(packs[pid]["score"]) - float(b_score)
            row["pack_deltas"] = pack_deltas
        out[slug] = row
    return out


def build_score_matrix(models: dict[str, dict], pack_ids: list[str]) -> dict:
    matrix: dict[str, dict] = {}
    for pid in pack_ids:
        matrix[pid] = {
            slug: model["packs"][pid]["score"]
            for slug, model in models.items()
            if pid in model["packs"]
        }
    return matrix


def score_campaign(
    manifest: dict,
    campaign_root: Path,
    baseline: dict | None,
    allow_partial: bool = False,
) -> dict:
    campaign_root = campaign_root.resolve()
    routes = manifest.get("routes") or []
    packs = manifest.get("packs") or []
    exclusions = manifest.get("exclusions") or []
    pack_ids = [p["id"] for p in sorted(packs, key=lambda p: p.get("order", 0))]

    models: dict[str, dict] = {}
    for route in routes:
        slug = route["slug"]
        route_dir = campaign_root / "results" / slug
        if not route_dir.is_dir():
            continue
        if is_excluded(campaign_root, route_dir, exclusions):
            continue
        models[slug] = score_route(
            campaign_root, route, packs, exclusions, allow_partial=allow_partial
        )

    if not models:
        die(f"no scorable routes under {campaign_root / 'results'}")

    campaign_id = campaign_root.name
    eng = (manifest.get("provenance") or {}).get("engine") or {}
    engine = {
        "daemon_md5": eng.get("daemon_md5"),
        "hipfire_md5": eng.get("hipfire_md5"),
    }
    if eng.get("note"):
        engine["note"] = eng["note"]

    roots: dict[str, str] = {
        "local": str(campaign_root),
        "results": str(campaign_root / "results"),
    }
    for _label, meta in (
        (manifest.get("provenance") or {}).get("campaigns") or {}
    ).items():
        if meta.get("campaign_id") == campaign_id and meta.get("remote_root"):
            roots["remote"] = meta["remote_root"]
            break

    summary: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "roots": roots,
        "engine": engine,
        "sampling": build_sampling_block(manifest, campaign_root, routes),
        "scope": build_scope(manifest),
        "scoring": build_scoring_defs(manifest),
        "models": models,
        "score_matrix": build_score_matrix(models, pack_ids),
    }

    if baseline is not None:
        summary["comparison_baseline"] = {
            "campaign_id": baseline.get("campaign_id"),
        }
        summary["route_aggregates"] = build_route_aggregates(
            models, baseline, pack_ids
        )
        summary["medium_score_matrix"] = summary["score_matrix"]
    else:
        summary["route_aggregates"] = build_route_aggregates(
            models, None, pack_ids
        )

    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score a BenchLocal nine-pack campaign results tree."
    )
    ap.add_argument(
        "--manifest",
        required=True,
        help="Path to tools/benchlocal/manifest.json",
    )
    ap.add_argument(
        "--campaign-root",
        required=True,
        help="Absolute campaign root containing results/<route>/",
    )
    ap.add_argument(
        "--baseline",
        default=None,
        help="Optional baseline full-battery-summary.json for delta fields",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Score packs whose artifacts are present; incomplete routes get "
            "null macro/scenario-weighted scores and raw_artifact_audit.missing "
            "populated. Without this flag, missing pack artifacts are fatal."
        ),
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Write summary JSON to FILE (default: stdout)",
    )
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest)
    campaign_root = Path(args.campaign_root)
    if not manifest_path.is_file():
        die(f"manifest not found: {manifest_path}")
    if not campaign_root.is_dir():
        die(f"campaign-root not found: {campaign_root}")

    manifest = load_json(manifest_path)
    baseline = load_json(Path(args.baseline)) if args.baseline else None

    summary = score_campaign(
        manifest, campaign_root, baseline, allow_partial=args.allow_partial
    )
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
