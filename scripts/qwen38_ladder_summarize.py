#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""
qwen38_ladder_summarize — deterministic summarizer for the hiptrx qwen38 ladder.

Produces 15-cell KLD/AR/DFlash matrix + per-bit draft choice with full audit trail.
Single Python file, stdlib only. Never invent unavailable metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_QCAL = "/home/kaden/qcal/ladder-v2"

N_VALUES = [2, 3, 4, 5, 6]
TIER_ORDER = ["xt", "base", "pro"]
EXPECTED_CELLS = [f"mq{n}-{tier}" for n in N_VALUES for tier in TIER_ORDER]
EXPECTED_KLD_REFS = ["wt2", "v6sel"]
CONTROL_DRAFT = "mq4v2"
KLD_RE = re.compile(
    r"slice-mean KLD\s*=\s*([0-9eE+\-\.]+)\s+mean NLL\s*=\s*([0-9eE+\-\.]+)\s+PPL\s*=\s*([0-9eE+\-\.]+)"
)
# DFlash measured line: [req ...] drafter=dflash tau=X tok/s=Y decode
DFLASH_MEASURED_RE = re.compile(
    r"\[req[^\]]*\][^\n]*drafter\s*=\s*dflash[^\n]*?tau\s*=\s*([0-9]*\.?[0-9]+)[^\n]*?tok/s\s*=\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)
# try to extract generated token count from the same line to filter exact 128-token runs
DFLASH_TOK_COUNT_RE = re.compile(r"\(\s*(\d+)\s*tok", re.IGNORECASE)


def eprint(*a, **kw):
    import sys
    print(*a, file=sys.stderr, **kw)


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_bytes(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size
    except Exception:
        return None


def compute_bpw(num_bytes: Optional[int], params: Optional[int]) -> Optional[float]:
    if num_bytes is None or params is None:
        return None
    try:
        return (num_bytes * 8.0) / float(params)
    except Exception:
        return None


def parse_json_robust_bounded(mixed: str, limit: int = 1 << 20) -> List[Any]:
    """Extract JSON objects from mixed log output, bounded.

    Tries line-wise JSON first, then brace-balanced fallback only if nothing found.
    Input truncated to `limit` to keep parser bounded.
    """
    if len(mixed) > limit:
        mixed = mixed[-limit:]
    objs: List[Any] = []
    for line in mixed.splitlines():
        s = line.strip()
        if not s or (not s.startswith("{") and not s.startswith("[")):
            continue
        # bounded line length
        if len(s) > 100_000:
            continue
        try:
            objs.append(json.loads(s))
            continue
        except Exception:
            pass
    if not objs:
        depth = 0
        start = -1
        in_str = False
        esc = False
        # bounded scan
        scan = mixed[-limit:] if len(mixed) > limit else mixed
        for i, ch in enumerate(scan):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                    if depth > 500:  # unbounded nesting guard
                        depth = 0
                        start = -1
                elif ch == "}":
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start != -1:
                            # bound object size
                            if i - start < 200_000:
                                cand = scan[start : i + 1]
                                try:
                                    objs.append(json.loads(cand))
                                except Exception:
                                    pass
                            start = -1
                            if len(objs) >= 20:
                                break
    return objs[:20]


def extract_medians_from_objs(objs: List[Any]) -> Tuple[Optional[float], Optional[float], Optional[List[float]], Optional[List[float]], Optional[Any]]:
    """From parsed_json objects, extract decode_tok_s.median, prefill_tok_s.median and raw samples.

    Returns (decode_median, prefill_median, decode_samples, prefill_samples, raw_obj)
    """
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        # bench JSON report has top-level decode_tok_s / prefill_tok_s dicts with median
        d = obj.get("decode_tok_s")
        p = obj.get("prefill_tok_s")
        # also support nested done/completion wrappers
        if isinstance(d, dict) and "median" in d:
            dec_med = d.get("median")
        elif isinstance(obj.get("done"), dict):
            done = obj["done"]
            d2 = done.get("decode_tok_s")
            dec_med = d2 if isinstance(d2, (int, float)) else (d2.get("median") if isinstance(d2, dict) else None)
        else:
            dec_med = obj.get("decode_tok_s") if isinstance(obj.get("decode_tok_s"), (int, float)) else None

        if isinstance(p, dict) and "median" in p:
            pre_med = p.get("median")
        elif isinstance(obj.get("done"), dict):
            done = obj["done"]
            p2 = done.get("prefill_tok_s")
            pre_med = p2 if isinstance(p2, (int, float)) else (p2.get("median") if isinstance(p2, dict) else None)
        else:
            pre_med = obj.get("prefill_tok_s") if isinstance(obj.get("prefill_tok_s"), (int, float)) else None

        # samples
        samples = obj.get("samples", {})
        dec_samples = None
        pre_samples = None
        if isinstance(samples, dict):
            if "decode" in samples and isinstance(samples["decode"], list):
                dec_samples = samples["decode"]
            if "prefill" in samples and isinstance(samples["prefill"], list):
                pre_samples = samples["prefill"]

        if dec_med is not None or pre_med is not None:
            try:
                dm = float(dec_med) if dec_med is not None else None
            except Exception:
                dm = None
            try:
                pm = float(pre_med) if pre_med is not None else None
            except Exception:
                pm = None
            return dm, pm, dec_samples, pre_samples, obj
    return None, None, None, None, None


def find_acceptance(obj: Any) -> Tuple[Optional[float], Optional[str]]:
    """Return (acceptance_value or None, reason). Never derive.

    If a real field exists in JSON, return it; otherwise return (None, unavailable_in_native_generate_v1).
    """
    if not isinstance(obj, dict):
        return None, "unavailable_in_native_generate_v1"
    # explicit fields that count as acceptance
    for key in ("acceptance_rate", "acceptance", "accept_rate", "accepted_rate", "spec_acceptance_rate"):
        if key in obj:
            try:
                return float(obj[key]), None
            except Exception:
                return None, "malformed_acceptance"
        # nested done
        if isinstance(obj.get("done"), dict) and key in obj["done"]:
            try:
                return float(obj["done"][key]), None
            except Exception:
                return None, "malformed_acceptance"
    return None, "unavailable_in_native_generate_v1"


def parse_kld_from_text(text: str) -> Optional[Tuple[float, float, float, str]]:
    m = KLD_RE.search(text)
    if not m:
        return None
    try:
        kld = float(m.group(1))
        nll = float(m.group(2))
        ppl = float(m.group(3))
        return kld, nll, ppl, m.group(0)
    except Exception:
        return None


def parse_dflash_measured(text: str, max_tokens: int = 128, runs: int = 5) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
    """Parse [req ...] drafter=dflash tau=X tok/s=Y lines.

    Drops warmup/short-output by using final `runs` lines or exact 128-token lines.
    Returns (taus, tok_s, rows_with_meta)
    """
    taus: List[float] = []
    toks: List[float] = []
    rows: List[Dict[str, Any]] = []
    # bound text
    if len(text) > (1 << 20):
        text = text[-(1 << 20):]
    for m in DFLASH_MEASURED_RE.finditer(text):
        try:
            tau = float(m.group(1))
            tok_s = float(m.group(2))
        except Exception:
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        if line_end == -1:
            line_end = len(text)
        line = text[line_start:line_end]
        # try to extract token count
        tok_c = None
        mc = DFLASH_TOK_COUNT_RE.search(line)
        if mc:
            try:
                tok_c = int(mc.group(1))
            except Exception:
                tok_c = None
        rows.append({"tau": tau, "tok_s": tok_s, "tok_count": tok_c, "line": line.strip()[:500]})
        taus.append(tau)
        toks.append(tok_s)
    if not rows:
        return [], [], []
    # Filter to exact 128-token lines if any such filter yields at least `runs` rows
    filtered = [r for r in rows if r.get("tok_count") == max_tokens]
    if len(filtered) >= runs:
        # use exact 128-token lines, take final `runs`
        rows = filtered[-runs:]
    elif len(rows) > runs:
        rows = rows[-runs:]
    taus = [r["tau"] for r in rows]
    toks = [r["tok_s"] for r in rows]
    return taus, toks, rows


def load_json_strict(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8", errors="replace")
    # bound size for safety
    if len(raw) > (10 << 20):
        raise ValueError(f"{path} too large")
    return json.loads(raw)


def gather_cells(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    cells = manifest.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError("manifest cells not a list")
    return cells


def gather_drafts(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    drafts = manifest.get("drafts", [])
    if not isinstance(drafts, list):
        raise ValueError("manifest drafts not a list")
    return drafts


class SummarizeError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Deterministic summarizer for the hiptrx qwen38 ladder (15 cells, dual KLD, AR/DFlash, per-bit draft choice).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--qcal-dir", default=DEFAULT_QCAL, help="qcal output root (contains manifest.json/results.json/state.json)")
    p.add_argument("--dry-run", action="store_true", help="validate and print what would be produced, without writing outputs")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    qcal = Path(args.qcal_dir)
    manifest_path = qcal / "manifest.json"
    results_path = qcal / "results.json"
    state_path = qcal / "state.json"

    errors: List[str] = []

    if not manifest_path.is_file():
        eprint(f"missing manifest {manifest_path}")
        return 2
    if not results_path.is_file():
        eprint(f"missing results {results_path}")
        return 2

    try:
        manifest = load_json_strict(manifest_path)
    except Exception as e:
        eprint(f"malformed manifest {manifest_path}: {e}")
        return 2
    try:
        results = load_json_strict(results_path)
    except Exception as e:
        eprint(f"malformed results {results_path}: {e}")
        return 2
    state: Dict[str, Any] = {}
    if state_path.is_file():
        try:
            state = load_json_strict(state_path)
        except Exception as e:
            errors.append(f"state.json malformed: {e}")

    cells = gather_cells(manifest)
    drafts = gather_drafts(manifest)

    # deterministic ordering
    def tier_key(t: str) -> int:
        try:
            return TIER_ORDER.index(t)
        except ValueError:
            return 99
    cells_sorted = sorted(cells, key=lambda c: (N_VALUES.index(c["n"]) if c["n"] in N_VALUES else 99, tier_key(c["tier"])))
    # validation of expected shape
    cell_ids = [c.get("cell_id") for c in cells_sorted]
    if len(cell_ids) != 15:
        errors.append(f"expected 15 cells, got {len(cell_ids)}")
    if len(set(cell_ids)) != len(cell_ids):
        errors.append(f"duplicate cell_id: {cell_ids}")
    missing_cells = [cid for cid in EXPECTED_CELLS if cid not in cell_ids]
    extra_cells = [cid for cid in cell_ids if cid not in EXPECTED_CELLS]
    if missing_cells:
        errors.append(f"missing expected cells: {missing_cells}")
    if extra_cells:
        errors.append(f"extra unexpected cells: {extra_cells}")

    # index cells by id
    cell_by_id = {c["cell_id"]: c for c in cells_sorted}
    draft_by_id = {d["draft_id"]: d for d in drafts}

    params = manifest.get("params")
    # fallback params may be in manifest or default
    if params is None:
        params = 26895998464

    # binary/ref/prompt digests from manifest
    binaries = manifest.get("binaries", {})
    prompt_digest = manifest.get("prompt_digest", {})
    refs = manifest.get("refs", [])

    rows = results.get("rows", [])
    if not isinstance(rows, list):
        errors.append("results rows not a list")
        rows = []

    # Group rows by phase
    # For robust fallback, also scan logs directory for per-rep files if rows missing.
    logs_dir = qcal / "logs"
    kld_dir = qcal / "kld"

    # --- KLD arms: 30 expected ---
    kld_arms_expected = 30
    kld_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    kld_duplicates: List[str] = []
    for r in rows:
        if r.get("phase") != "kld":
            continue
        cid = r.get("cell_id")
        rk = r.get("ref_kind")
        if cid is None or rk is None:
            # try task_id fallback "cell:ref"
            tid = r.get("task_id", "")
            if ":" in tid:
                cid, rk = tid.split(":", 1)
        key = (cid, rk)
        if key in kld_by_key:
            kld_duplicates.append(f"{cid}:{rk}")
        kld_by_key[key] = r

    if kld_duplicates:
        errors.append(f"duplicate KLD arms: {sorted(set(kld_duplicates))}")

    # Parse KLD metrics
    kld_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for cid in cell_ids:
        for ref_kind in EXPECTED_KLD_REFS:
            key = (cid, ref_kind)
            r = kld_by_key.get(key)
            source_paths: List[str] = []
            raw_text = ""
            if r is not None:
                # prefer stdout+stderr from row, bounded
                raw_text = (r.get("stdout") or "") + "\n" + (r.get("stderr") or "") + "\n" + (r.get("raw_stdout") or "") + "\n" + (r.get("raw_stderr") or "")
                if r.get("log"):
                    source_paths.append(str(r["log"]))
                if r.get("output"):
                    source_paths.append(str(r["output"]))
            # fallback to per-rep logs if raw_text empty or parse fails
            parsed = parse_kld_from_text(raw_text) if raw_text else None
            if parsed is None:
                # search log files: logs/kld_{cell}_{ref}.log — exact name only
                candidates = []
                if logs_dir.is_dir():
                    pat1 = logs_dir / f"kld_{cid}_{ref_kind}.log"
                    pat2 = logs_dir / f"kld_{cid.replace('-','_')}_{ref_kind}.log"
                    for cand in [pat1, pat2]:
                        if cand.is_file() and str(cand) not in source_paths:
                            candidates.append(cand)
                for cand in candidates:
                    try:
                        txt = cand.read_text(encoding="utf-8", errors="replace")
                        if len(txt) > (1 << 20):
                            txt = txt[-(1 << 20):]
                        raw_text += "\n" + txt
                        source_paths.append(str(cand))
                    except Exception:
                        continue
                parsed = parse_kld_from_text(raw_text) if raw_text else None
            if parsed is None:
                errors.append(f"missing/malformed KLD metric for {cid}:{ref_kind} source={source_paths[:2]}")
                kld_results[key] = {
                    "cell_id": cid,
                    "ref_kind": ref_kind,
                    "kld": None,
                    "nll": None,
                    "ppl": None,
                    "error": "missing slice-mean KLD line",
                    "sources": source_paths,
                    "raw_snippet": raw_text[-500:] if raw_text else "",
                }
            else:
                kld_val, nll_val, ppl_val, line = parsed
                kld_results[key] = {
                    "cell_id": cid,
                    "ref_kind": ref_kind,
                    "kld": kld_val,
                    "nll": nll_val,
                    "ppl": ppl_val,
                    "raw_line": line,
                    "sources": source_paths,
                }

    # --- AR rows: 15 expected ---
    ar_by_cell: Dict[str, Dict[str, Any]] = {}
    ar_duplicates: List[str] = []
    for r in rows:
        if r.get("phase") != "bench-ar":
            continue
        cid = r.get("cell_id")
        if cid is None:
            continue
        if cid in ar_by_cell:
            ar_duplicates.append(cid)
        ar_by_cell[cid] = r
    if ar_duplicates:
        errors.append(f"duplicate AR rows: {sorted(set(ar_duplicates))}")

    ar_results: Dict[str, Dict[str, Any]] = {}
    for cid in cell_ids:
        r = ar_by_cell.get(cid)
        rep_medians_decode: List[float] = []
        rep_medians_prefill: List[float] = []
        rep_samples_decode: List[Any] = []
        rep_samples_prefill: List[Any] = []
        acceptance_val = None
        acceptance_reason = None
        sources: List[str] = []
        reps_raw = None
        if r is not None:
            reps_raw = r.get("reps")
            sources = []
            if r.get("bench_bin"):
                sources.append(str(r.get("bench_bin")))
            # also raw result path
            sources.append(f"results.json:bench-ar:{cid}")
            # if reps present
            if isinstance(reps_raw, list) and len(reps_raw) == 3:
                for rep in reps_raw:
                    # bounded mixed json
                    mixed = ""
                    if rep.get("stdout"):
                        mixed += rep["stdout"] + "\n"
                    if rep.get("stderr"):
                        mixed += rep["stderr"] + "\n"
                    if rep.get("raw_stdout"):
                        mixed += rep["raw_stdout"] + "\n"
                    # fallback to log file
                    if not mixed.strip() and rep.get("log"):
                        lp = Path(rep["log"])
                        if lp.is_file():
                            try:
                                mixed = lp.read_text(encoding="utf-8", errors="replace")
                                sources.append(str(lp))
                            except Exception:
                                mixed = ""
                        else:
                            sources.append(str(lp) + " (missing)")
                    else:
                        if rep.get("log"):
                            sources.append(str(rep["log"]))
                    objs = parse_json_robust_bounded(mixed)
                    if not objs and rep.get("parsed_json"):
                        objs = rep["parsed_json"] if isinstance(rep["parsed_json"], list) else [rep["parsed_json"]]
                    dm, pm, ds, ps, raw_obj = extract_medians_from_objs(objs)
                    if dm is None:
                        errors.append(f"AR {cid} rep {rep.get('rep')} missing decode_tok_s.median sources={sources[-1:]}")
                    else:
                        rep_medians_decode.append(dm)
                    if pm is None:
                        errors.append(f"AR {cid} rep {rep.get('rep')} missing prefill_tok_s.median")
                    else:
                        rep_medians_prefill.append(pm)
                    rep_samples_decode.append(ds)
                    rep_samples_prefill.append(ps)
                    # acceptance: check raw_obj
                    av, ar_reason = find_acceptance(raw_obj if raw_obj is not None else {})
                    # Only use first non-null? But spec says acceptance is null unless real field exists
                    # Preserve first; if any rep has real acceptance, use it, else unavailable
                    if acceptance_val is None and av is not None:
                        acceptance_val = av
                        acceptance_reason = ar_reason
                    elif acceptance_reason is None and ar_reason is not None:
                        acceptance_reason = ar_reason
            else:
                # fallback: scan logs directory for bench_ar logs
                candidates = []
                if logs_dir.is_dir():
                    for idx in range(1, 4):
                        cand = logs_dir / f"bench_ar_{cid}_rep{idx}.log"
                        if cand.is_file():
                            candidates.append(cand)
                    if not candidates:
                        # generic glob bounded
                        for lf in sorted(logs_dir.glob(f"bench_ar_{cid}*"))[:6]:
                            candidates.append(lf)
                if len(candidates) == 3:
                    for cand in candidates:
                        try:
                            txt = cand.read_text(encoding="utf-8", errors="replace")
                            sources.append(str(cand))
                            objs = parse_json_robust_bounded(txt)
                            dm, pm, ds, ps, raw_obj = extract_medians_from_objs(objs)
                            if dm is not None:
                                rep_medians_decode.append(dm)
                            if pm is not None:
                                rep_medians_prefill.append(pm)
                            rep_samples_decode.append(ds)
                            rep_samples_prefill.append(ps)
                            av, ar_reason = find_acceptance(raw_obj if raw_obj is not None else {})
                            if acceptance_val is None and av is not None:
                                acceptance_val = av
                            if acceptance_reason is None:
                                acceptance_reason = ar_reason
                        except Exception as e:
                            errors.append(f"AR fallback log {cand} failed: {e}")
                else:
                    errors.append(f"AR {cid} missing 3 fresh reps (found {len(candidates) if candidates else 0} logs, results reps={type(reps_raw).__name__})")
        else:
            # no row at all: try fallback logs
            candidates = []
            if logs_dir.is_dir():
                for idx in range(1, 4):
                    cand = logs_dir / f"bench_ar_{cid}_rep{idx}.log"
                    if cand.is_file():
                        candidates.append(cand)
            if candidates:
                sources = [str(c) for c in candidates]
                for cand in candidates:
                    try:
                        txt = cand.read_text(encoding="utf-8", errors="replace")
                        objs = parse_json_robust_bounded(txt)
                        dm, pm, ds, ps, raw_obj = extract_medians_from_objs(objs)
                        if dm is not None:
                            rep_medians_decode.append(dm)
                        if pm is not None:
                            rep_medians_prefill.append(pm)
                        rep_samples_decode.append(ds)
                        rep_samples_prefill.append(ps)
                    except Exception as e:
                        errors.append(f"AR fallback {cid} {cand} failed: {e}")
            else:
                errors.append(f"AR {cid} missing (no results row and no fallback logs)")

        # Now we have rep arrays; compute medians if we have 3 values
        if len(rep_medians_decode) != 3:
            errors.append(f"AR {cid} expected 3 decode rep medians, got {len(rep_medians_decode)}: {rep_medians_decode}")
        if len(rep_medians_prefill) != 3:
            errors.append(f"AR {cid} expected 3 prefill rep medians, got {len(rep_medians_prefill)}: {rep_medians_prefill}")

        decode_median = None
        prefill_median = None
        if len(rep_medians_decode) == 3:
            decode_median = statistics.median(rep_medians_decode)
        elif rep_medians_decode:
            decode_median = statistics.median(rep_medians_decode)
        if len(rep_medians_prefill) == 3:
            prefill_median = statistics.median(rep_medians_prefill)
        elif rep_medians_prefill:
            prefill_median = statistics.median(rep_medians_prefill)

        if acceptance_reason is None:
            acceptance_reason = "unavailable_in_native_generate_v1"
        # Never report acceptance if absent
        if acceptance_val is not None and acceptance_reason is None:
            acceptance_reason = None

        ar_results[cid] = {
            "cell_id": cid,
            "decode_tok_s": {
                "median": decode_median,
                "rep_medians": rep_medians_decode,
                "rep_samples": rep_samples_decode,
            },
            "prefill_tok_s": {
                "median": prefill_median,
                "rep_medians": rep_medians_prefill,
                "rep_samples": rep_samples_prefill,
            },
            "acceptance": acceptance_val,
            "acceptance_reason": acceptance_reason if acceptance_val is None else None,
            "sources": sources,
        }

    # --- DFlash rows: 15 same-bit + 12 control (non-mq4) = 27 total, keyed (cell_id,draft_id) ---
    # The runner now produces 27 rows keyed (cell_id,draft_id) via results.json
    dflash_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    dflash_duplicates: List[str] = []
    for r in rows:
        if r.get("phase") != "bench-dflash":
            continue
        cid = r.get("cell_id")
        did = r.get("draft_id")
        if cid is None or did is None:
            # try task_id
            tid = r.get("task_id") or r.get("task_id", "")
            if ":" in str(tid):
                cid, did = str(tid).split(":", 1)
            else:
                # draft_id may be in r["draft_id"] or missing
                continue
        key = (cid, did)
        if key in dflash_by_key:
            dflash_duplicates.append(f"{cid}:{did}")
        dflash_by_key[key] = r
    if dflash_duplicates:
        errors.append(f"duplicate DFlash rows: {sorted(set(dflash_duplicates))}")

    # Validate expected counts
    expected_same = set((cid, cell_by_id[cid]["codec"]) for cid in cell_ids)
    expected_control = set((cid, CONTROL_DRAFT) for cid in cell_ids if cell_by_id[cid]["codec"] != CONTROL_DRAFT)
    expected_all = expected_same | expected_control
    missing_dflash = [f"{a}:{b}" for (a, b) in sorted(expected_all) if (a, b) not in dflash_by_key]
    extra_dflash = [f"{a}:{b}" for (a, b) in dflash_by_key if (a, b) not in expected_all]
    if missing_dflash:
        errors.append(f"missing DFlash arms: {missing_dflash}")
    if extra_dflash:
        errors.append(f"extra DFlash arms: {extra_dflash}")

    dflash_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    # Also need draft byte sizes for tie-break: smaller draft wins
    draft_bytes_map: Dict[str, Optional[int]] = {}
    for did, dinfo in draft_by_id.items():
        p = Path(dinfo.get("draft_path", ""))
        draft_bytes_map[did] = file_bytes(p) if p.is_file() else dinfo.get("bytes")

    # For each expected key, parse metrics
    for (cid, did) in sorted(expected_all):
        r = dflash_by_key.get((cid, did))
        rep_medians_decode: List[float] = []
        rep_medians_prefill: List[float] = []
        rep_taus: List[float] = []
        rep_tok_s_measured: List[float] = []
        taus_all_reps: List[List[float]] = []
        toks_all_reps: List[List[float]] = []
        # per-rep tau/tok_s arrays
        sources: List[str] = []
        acceptance_val = None
        acceptance_reason: Optional[str] = None
        per_rep_details: List[Dict[str, Any]] = []
        if r is None:
            # fallback to logs?
            # try logs/bench_dflash_{cid}_rep*.log but disambiguate draft
            candidates = []
            if logs_dir.is_dir():
                # draft-specific logs may be named bench_dflash_{cid}_rep{idx}.log without draft suffix;
                # but runner should have included draft in task_id, fallback ambiguous
                for idx in range(1, 4):
                    cand = logs_dir / f"bench_dflash_{cid}_rep{idx}.log"
                    if cand.is_file():
                        candidates.append(cand)
                # also draft-specific names
                for idx in range(1, 4):
                    cand2 = logs_dir / f"bench_dflash_{cid}_{did}_rep{idx}.log"
                    if cand2.is_file():
                        candidates.append(cand2)
            if candidates:
                sources = [str(c) for c in candidates[:3]]
                # not enough to disambiguate; mark missing
                errors.append(f"DFlash {cid}:{did} fallback logs ambiguous (need draft-qualified logs)")
            else:
                errors.append(f"DFlash {cid}:{did} missing result row and no fallback logs")
            dflash_results[(cid, did)] = {
                "cell_id": cid,
                "draft_id": did,
                "error": "missing",
                "sources": sources,
            }
            continue

        reps_raw = r.get("reps")
        if r.get("draft_path"):
            sources.append(str(r["draft_path"]))
        sources.append(f"results.json:bench-dflash:{cid}:{did}")
        # draft artifact bytes
        draft_path = r.get("draft_path") or draft_by_id.get(did, {}).get("draft_path")
        draft_bytes = r.get("draft_bytes")
        if draft_bytes is None and draft_path:
            draft_bytes = file_bytes(Path(draft_path)) if Path(draft_path).is_file() else draft_bytes_map.get(did)

        if isinstance(reps_raw, list) and len(reps_raw) == 3:
            for rep in reps_raw:
                mixed = ""
                if rep.get("stdout"):
                    mixed += rep["stdout"] + "\n"
                if rep.get("stderr"):
                    mixed += rep["stderr"] + "\n"
                if rep.get("raw_stdout"):
                    mixed += rep["raw_stdout"] + "\n"
                if rep.get("raw_stderr"):
                    mixed += rep["raw_stderr"] + "\n"
                # include log file content if mixed empty
                if not mixed.strip() and rep.get("log"):
                    lp = Path(rep["log"])
                    if lp.is_file():
                        try:
                            mixed = lp.read_text(encoding="utf-8", errors="replace")
                        except Exception:
                            mixed = ""
                    sources.append(str(lp) if lp else "")
                else:
                    if rep.get("log"):
                        sources.append(str(rep["log"]))
                objs = parse_json_robust_bounded(mixed)
                if not objs and rep.get("parsed_json"):
                    objs = rep["parsed_json"] if isinstance(rep["parsed_json"], list) else [rep["parsed_json"]]
                dm, pm, ds, ps, raw_obj = extract_medians_from_objs(objs)
                if dm is not None:
                    rep_medians_decode.append(dm)
                else:
                    errors.append(f"DFlash {cid}:{did} rep {rep.get('rep')} missing decode_tok_s.median")
                if pm is not None:
                    rep_medians_prefill.append(pm)
                # measured tau/tok_s from stderr/log (bounded)
                stderr_text = rep.get("stderr") or rep.get("raw_stderr") or ""
                if rep.get("log") and Path(rep["log"]).is_file() and not stderr_text.strip():
                    try:
                        stderr_text = Path(rep["log"]).read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        stderr_text = ""
                # also try stdout+stderr combined bounded
                combined = (rep.get("stdout") or "") + "\n" + stderr_text
                taus, toks, rows_meta = parse_dflash_measured(combined, max_tokens=128, runs=5)
                if not taus:
                    # try fallback to reading log file directly bounded
                    if rep.get("log") and Path(rep["log"]).is_file():
                        try:
                            txt = Path(rep["log"]).read_text(encoding="utf-8", errors="replace")
                            taus, toks, rows_meta = parse_dflash_measured(txt, max_tokens=128, runs=5)
                        except Exception:
                            pass
                if taus:
                    # per-rep median tau/tok_s across the 5 runs? spec says measured [req ...] lines; drop warmup.
                    # We'll take median across runs within rep, then retain arrays.
                    # For overall, we will median across reps' medians.
                    rep_tau_median = statistics.median(taus)
                    rep_tok_median = statistics.median(toks)
                    rep_taus.append(rep_tau_median)
                    rep_tok_s_measured.append(rep_tok_median)
                    taus_all_reps.append(taus)
                    toks_all_reps.append(toks)
                    per_rep_details.append({"rep": rep.get("rep"), "taus": taus, "tok_s": toks, "median_tau": rep_tau_median, "median_tok_s": rep_tok_median})
                else:
                    errors.append(f"DFlash {cid}:{did} rep {rep.get('rep')} missing drafter=dflash measured lines")
                    per_rep_details.append({"rep": rep.get("rep"), "error": "missing drafter=dflash line"})
                    rep_taus.append(None)
                    rep_tok_s_measured.append(None)

                av, ar_reason = find_acceptance(raw_obj if raw_obj is not None else {})
                if acceptance_val is None and av is not None:
                    acceptance_val = av
                    acceptance_reason = ar_reason
                elif acceptance_reason is None and ar_reason is not None:
                    acceptance_reason = ar_reason
        else:
            # fallback: scan logs for 3 reps
            candidates = []
            if logs_dir.is_dir():
                for idx in range(1, 4):
                    cand = logs_dir / f"bench_dflash_{cid}_{did}_rep{idx}.log"
                    if cand.is_file():
                        candidates.append(cand)
                if not candidates:
                    for idx in range(1, 4):
                        cand = logs_dir / f"bench_dflash_{cid}_rep{idx}.log"
                        if cand.is_file():
                            candidates.append(cand)
            if len(candidates) == 3:
                for cand in candidates:
                    try:
                        txt = cand.read_text(encoding="utf-8", errors="replace")
                        sources.append(str(cand))
                        objs = parse_json_robust_bounded(txt)
                        dm, pm, _, _, raw_obj = extract_medians_from_objs(objs)
                        if dm is not None:
                            rep_medians_decode.append(dm)
                        if pm is not None:
                            rep_medians_prefill.append(pm)
                        taus, toks, _ = parse_dflash_measured(txt, max_tokens=128, runs=5)
                        if taus:
                            rep_taus.append(statistics.median(taus))
                            rep_tok_s_measured.append(statistics.median(toks))
                            taus_all_reps.append(taus)
                            toks_all_reps.append(toks)
                    except Exception as e:
                        errors.append(f"DFlash fallback {cid}:{did} {cand} failed: {e}")
            else:
                errors.append(f"DFlash {cid}:{did} expected 3 reps, got {type(reps_raw).__name__} / fallback {len(candidates)} logs")

        # per-draft median across reps (fresh-process median)
        decode_median = statistics.median([x for x in rep_medians_decode if x is not None]) if rep_medians_decode else None
        prefill_median = statistics.median([x for x in rep_medians_prefill if x is not None]) if rep_medians_prefill else None
        # filter None
        clean_taus = [x for x in rep_taus if x is not None]
        tau_median = statistics.median(clean_taus) if clean_taus else None
        clean_toks = [x for x in rep_tok_s_measured if x is not None]
        tok_s_median = statistics.median(clean_toks) if clean_toks else None

        if acceptance_reason is None:
            acceptance_reason = "unavailable_in_native_generate_v1"

        dflash_results[(cid, did)] = {
            "cell_id": cid,
            "draft_id": did,
            "draft_path": draft_path,
            "draft_bytes": draft_bytes,
            "draft_sha256": row.get("draft_sha256")
            or (sha256_file(Path(draft_path)) if draft_path and Path(draft_path).is_file() else None),
            "decode_tok_s": {"median": decode_median, "rep_medians": rep_medians_decode},
            "prefill_tok_s": {"median": prefill_median, "rep_medians": rep_medians_prefill},
            "tau": {"median": tau_median, "rep_medians": rep_taus, "rep_arrays": taus_all_reps},
            "measured_tok_s": {"median": tok_s_median, "rep_medians": rep_tok_s_measured, "rep_arrays": toks_all_reps},
            "acceptance": acceptance_val,
            "acceptance_reason": acceptance_reason if acceptance_val is None else None,
            "sources": list(dict.fromkeys(sources))[:10],
            "per_rep": per_rep_details,
        }

    # --- Draft selection per cell (evidence, not assume same-bit) ---
    choices: Dict[str, Dict[str, Any]] = {}
    for cid in cell_ids:
        codec = cell_by_id[cid]["codec"]
        same_key = (cid, codec)
        ctrl_key = (cid, CONTROL_DRAFT)
        same = dflash_results.get(same_key)
        ctrl = dflash_results.get(ctrl_key) if codec != CONTROL_DRAFT else None
        same_med = None
        ctrl_med = None
        if same and isinstance(same.get("decode_tok_s", {}).get("median"), (int, float)):
            same_med = float(same["decode_tok_s"]["median"])
        if ctrl and isinstance(ctrl.get("decode_tok_s", {}).get("median"), (int, float)):
            ctrl_med = float(ctrl["decode_tok_s"]["median"])
        if codec == CONTROL_DRAFT:
            chosen = CONTROL_DRAFT
            reason = "controller is same-bit (mq4)"
        elif same_med is None and ctrl_med is None:
            chosen = None
            reason = "both missing"
        elif same_med is None:
            chosen = CONTROL_DRAFT
            reason = "same-bit missing"
        elif ctrl_med is None:
            chosen = codec
            reason = "control missing"
        elif same_med > ctrl_med:
            chosen = codec
            reason = f"same-bit {same_med:.1f} > control {ctrl_med:.1f}"
        elif ctrl_med > same_med:
            chosen = CONTROL_DRAFT
            reason = f"control {ctrl_med:.1f} > same-bit {same_med:.1f}"
        else:  # tie
            # smaller draft wins
            same_bytes = same.get("draft_bytes") if same else None
            ctrl_bytes = ctrl.get("draft_bytes") if ctrl else None
            if isinstance(same_bytes, int) and isinstance(ctrl_bytes, int):
                if ctrl_bytes < same_bytes:
                    chosen = CONTROL_DRAFT
                    reason = f"tie {same_med:.1f}; smaller draft {CONTROL_DRAFT} ({ctrl_bytes} < {same_bytes})"
                elif same_bytes < ctrl_bytes:
                    chosen = codec
                    reason = f"tie {same_med:.1f}; smaller draft {codec} ({same_bytes} < {ctrl_bytes})"
                else:
                    chosen = codec
                    reason = f"tie {same_med:.1f}; equal bytes, prefer same-bit"
            else:
                chosen = codec
                reason = f"tie {same_med:.1f}; bytes unavailable, prefer same-bit"
        choices[cid] = {"chosen_draft": chosen, "reason": reason, "same_median": same_med, "control_median": ctrl_med}

    # --- Build product rows (15) ---
    products: List[Dict[str, Any]] = []
    for cid in cell_ids:
        cell = cell_by_id[cid]
        artifact_path = Path(cell.get("artifact", ""))
        art_bytes = file_bytes(artifact_path) if artifact_path.is_file() else cell.get("bytes")
        # try to find quant row for bytes/sha if not on disk
        quant_row = next((r for r in rows if r.get("phase") == "quantize" and r.get("cell_id") == cid), None)
        if art_bytes is None and quant_row:
            art_bytes = quant_row.get("bytes") or quant_row.get("artifact_bytes")
        art_sha = (quant_row.get("sha256") if quant_row else None) or (
            sha256_file(artifact_path) if artifact_path.is_file() else None
        )
        art_md5 = (quant_row.get("md5") if quant_row else None) or (
            md5_file(artifact_path) if artifact_path.is_file() else None
        )
        bpw = compute_bpw(art_bytes, params) if art_bytes else cell.get("bpw")
        # digests
        codec = cell.get("codec")
        tier = cell.get("tier")
        fixed_tier = cell.get("fixed_tier")
        n = cell.get("n")
        # raw paths
        raw_paths: Dict[str, Any] = {
            "artifact": str(artifact_path),
            "quant_log": quant_row.get("log") if quant_row else None,
            "kld_wt2_sources": kld_results.get((cid, "wt2"), {}).get("sources", []),
            "kld_v6sel_sources": kld_results.get((cid, "v6sel"), {}).get("sources", []),
            "ar_sources": ar_results.get(cid, {}).get("sources", []),
            "dflash_same_sources": dflash_results.get((cid, codec), {}).get("sources", []) if (cid, codec) in dflash_results else [],
            "dflash_control_sources": dflash_results.get((cid, CONTROL_DRAFT), {}).get("sources", []) if codec != CONTROL_DRAFT else [],
        }
        prod = {
            "cell_id": cid,
            "n": n,
            "tier": tier,
            "codec": codec,
            "format": cell.get("format", codec),
            "fixed_tier": fixed_tier,
            "artifact": str(artifact_path),
            "artifact_bytes": art_bytes,
            "artifact_sha256": art_sha,
            "artifact_md5": art_md5,
            "bpw": bpw,
            "commit": manifest.get("commit") or cell.get("commit") or (quant_row.get("commit") if quant_row else None),
            "prompt_digest": prompt_digest,
            "binaries": binaries,
            "refs": refs,
            "raw_paths": raw_paths,
            "kld": {
                "wt2": kld_results.get((cid, "wt2")),
                "v6sel": kld_results.get((cid, "v6sel")),
            },
            "ar": ar_results.get(cid),
            "dflash": {
                "same_bit": dflash_results.get((cid, codec)),
                "control": dflash_results.get((cid, CONTROL_DRAFT)) if codec != CONTROL_DRAFT else None,
                "choice": choices.get(cid),
            },
        }
        products.append(prod)

    # --- Validation final counts ---
    # Already tracked errors, but ensure fail nonzero on missing/duplicate/malformed
    # Check every product has required fields source-linked
    for p in products:
        if p["artifact_bytes"] is None:
            errors.append(f"{p['cell_id']} missing artifact bytes")
        if p["bpw"] is None:
            errors.append(f"{p['cell_id']} missing bpw")
        # each value should be source-linked: check raw_paths present
        if not p["raw_paths"].get("kld_wt2_sources"):
            errors.append(f"{p['cell_id']} KLD wt2 missing source link")
        # ensure KLD values present
        for ref in EXPECTED_KLD_REFS:
            kv = p["kld"].get(ref, {})
            if kv is None or kv.get("kld") is None:
                errors.append(f"{p['cell_id']} KLD {ref} missing metric")
        ar = p["ar"]
        if ar is None or ar.get("decode_tok_s", {}).get("median") is None:
            errors.append(f"{p['cell_id']} AR decode median missing")
        d_same = p["dflash"]["same_bit"]
        if d_same is None or d_same.get("decode_tok_s", {}).get("median") is None:
            errors.append(f"{p['cell_id']} DFlash same-bit decode median missing")

    # Sort errors for determinism
    errors = sorted(set(errors))

    # --- Prepare outputs ---
    summary: Dict[str, Any] = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "qcal_dir": str(qcal),
        "manifest": {
            "path": str(manifest_path),
            "commit": manifest.get("commit"),
            "diff_md5": manifest.get("diff_md5"),
            "params": params,
            "version": manifest.get("version"),
        },
        "prompt": prompt_digest,
        "binaries": binaries,
        "refs": refs,
        "state": {"path": str(state_path), "exists": state_path.is_file()},
        "counts": {
            "cells": len(products),
            "kld_arms": len(kld_results),
            "ar_rows": len(ar_results),
            "dflash_same": len([k for k in dflash_results if k[1] == cell_by_id[k[0]]["codec"]]) if products else 0,
            "dflash_control": len([k for k in dflash_results if k[1] == CONTROL_DRAFT and cell_by_id[k[0]]["codec"] != CONTROL_DRAFT]) if products else 0,
            "dflash_total": len(dflash_results),
        },
        "expected": {
            "cells": 15,
            "kld_arms": 30,
            "ar_rows": 15,
            "dflash_same": 15,
            "dflash_control_non_mq4": 12,
            "dflash_total": 27,
        },
        "errors": errors,
        "products": products,
    }

    # --- Dry-run handling ---
    if args.dry_run:
        eprint(f"[dry-run] qcal={qcal} cells={len(products)} kld={len(kld_results)} ar={len(ar_results)} dflash={len(dflash_results)} errors={len(errors)}")
        for err in errors:
            eprint(f"[dry-run] error: {err}")
        # Print concise product choice lines
        for cid in cell_ids:
            ch = choices.get(cid, {})
            eprint(f"[dry-run] {cid} -> chosen {ch.get('chosen_draft')} reason={ch.get('reason')} same={ch.get('same_median')} ctrl={ch.get('control_median')}")
        # Do not write files
        if errors:
            eprint(f"[dry-run] validation FAILED with {len(errors)} errors")
            return 1
        eprint("[dry-run] validation OK, would write summary.json/csv/md")
        return 0

    # --- Write summary.json ---
    out_json = qcal / "summary.json"
    out_csv = qcal / "summary.csv"
    out_md = qcal / "summary.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    # atomic write via temp
    import tempfile, os

    def atomic_write(path: Path, data: str):
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".tmp.")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            try:
                if Path(tmp_path).exists():
                    Path(tmp_path).unlink()
            except Exception:
                pass

    atomic_write(out_json, json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # --- CSV ---
    csv_fields = [
        "cell_id", "n", "tier", "codec", "fixed_tier", "artifact_bytes", "bpw",
        "kld_wt2", "nll_wt2", "ppl_wt2", "kld_v6sel", "nll_v6sel", "ppl_v6sel",
        "ar_decode_median", "ar_prefill_median", "ar_decode_rep_medians", "ar_prefill_rep_medians",
        "dflash_same_decode_median", "dflash_same_tau_median", "dflash_same_measured_tok_s_median",
        "dflash_control_decode_median", "dflash_control_tau_median", "dflash_control_measured_tok_s_median",
        "chosen_draft", "choice_reason",
        "artifact_sha256", "commit",
    ]
    csv_buf = []
    import io

    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=csv_fields, lineterminator="\n")
    w.writeheader()
    for p in products:
        cid = p["cell_id"]
        k_wt2 = p["kld"]["wt2"] or {}
        k_v6 = p["kld"]["v6sel"] or {}
        ar = p["ar"] or {}
        d_same = p["dflash"]["same_bit"] or {}
        d_ctrl = p["dflash"]["control"] or {}
        choice = p["dflash"]["choice"] or {}
        row = {
            "cell_id": cid,
            "n": p["n"],
            "tier": p["tier"],
            "codec": p["codec"],
            "fixed_tier": p["fixed_tier"] or "",
            "artifact_bytes": p["artifact_bytes"] if p["artifact_bytes"] is not None else "",
            "bpw": f"{p['bpw']:.6f}" if isinstance(p["bpw"], (int, float)) else "",
            "kld_wt2": k_wt2.get("kld", ""),
            "nll_wt2": k_wt2.get("nll", ""),
            "ppl_wt2": k_wt2.get("ppl", ""),
            "kld_v6sel": k_v6.get("kld", ""),
            "nll_v6sel": k_v6.get("nll", ""),
            "ppl_v6sel": k_v6.get("ppl", ""),
            "ar_decode_median": ar.get("decode_tok_s", {}).get("median", "") if isinstance(ar.get("decode_tok_s"), dict) else "",
            "ar_prefill_median": ar.get("prefill_tok_s", {}).get("median", "") if isinstance(ar.get("prefill_tok_s"), dict) else "",
            "ar_decode_rep_medians": json.dumps(ar.get("decode_tok_s", {}).get("rep_medians", [])) if isinstance(ar.get("decode_tok_s"), dict) else "[]",
            "ar_prefill_rep_medians": json.dumps(ar.get("prefill_tok_s", {}).get("rep_medians", [])) if isinstance(ar.get("prefill_tok_s"), dict) else "[]",
            "dflash_same_decode_median": d_same.get("decode_tok_s", {}).get("median", "") if isinstance(d_same.get("decode_tok_s"), dict) else "",
            "dflash_same_tau_median": d_same.get("tau", {}).get("median", "") if isinstance(d_same.get("tau"), dict) else "",
            "dflash_same_measured_tok_s_median": d_same.get("measured_tok_s", {}).get("median", "") if isinstance(d_same.get("measured_tok_s"), dict) else "",
            "dflash_control_decode_median": d_ctrl.get("decode_tok_s", {}).get("median", "") if isinstance(d_ctrl.get("decode_tok_s"), dict) else "",
            "dflash_control_tau_median": d_ctrl.get("tau", {}).get("median", "") if isinstance(d_ctrl.get("tau"), dict) else "",
            "dflash_control_measured_tok_s_median": d_ctrl.get("measured_tok_s", {}).get("median", "") if isinstance(d_ctrl.get("measured_tok_s"), dict) else "",
            "chosen_draft": choice.get("chosen_draft", ""),
            "choice_reason": choice.get("reason", ""),
            "artifact_sha256": p["artifact_sha256"] or "",
            "commit": p["commit"] or "",
        }
        # ensure json-serialized rep medians are bounded
        w.writerow(row)
    csv_text = sio.getvalue()
    atomic_write(out_csv, csv_text)

    # --- Markdown ---
    md_lines: List[str] = []
    md_lines.append("# qwen38 ladder summary")
    md_lines.append("")
    md_lines.append(f"Generated: `{summary['generated_at']}` | qcal: `{qcal}` | commit: `{summary['manifest']['commit']}`")
    md_lines.append("")
    if errors:
        md_lines.append("## errors")
        md_lines.append("")
        for e in errors:
            md_lines.append(f"- {e}")
        md_lines.append("")
    md_lines.append(f"Cells: {len(products)} (expected 15) | KLD arms: {len(kld_results)} (expected 30) | AR rows: {len(ar_results)} (expected 15) | DFlash same-bit: {summary['counts']['dflash_same']} (15) + control: {summary['counts']['dflash_control']} (12) = {summary['counts']['dflash_total']} (27)")
    md_lines.append("")
    md_lines.append("| cell | bpw | KLD wt2 | KLD v6sel | AR dec tok/s | AR pre tok/s | DFlash same dec | DFlash ctrl dec | tau same | tau ctrl | chosen |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for p in products:
        cid = p["cell_id"]
        bpw = f"{p['bpw']:.3f}" if isinstance(p["bpw"], (int, float)) else "—"
        k_wt2 = p["kld"]["wt2"] or {}
        k_v6 = p["kld"]["v6sel"] or {}
        k1 = f"{k_wt2.get('kld', 0):.6f}" if isinstance(k_wt2.get("kld"), (int, float)) else "—"
        k2 = f"{k_v6.get('kld', 0):.6f}" if isinstance(k_v6.get("kld"), (int, float)) else "—"
        ar = p["ar"] or {}
        ar_d = ar.get("decode_tok_s", {}).get("median")
        ar_p = ar.get("prefill_tok_s", {}).get("median")
        ar_ds = f"{ar_d:.1f}" if isinstance(ar_d, (int, float)) else "—"
        ar_ps = f"{ar_p:.1f}" if isinstance(ar_p, (int, float)) else "—"
        d_same = p["dflash"]["same_bit"] or {}
        d_ctrl = p["dflash"]["control"] or {}
        sd = d_same.get("decode_tok_s", {}).get("median") if isinstance(d_same.get("decode_tok_s"), dict) else None
        cd = d_ctrl.get("decode_tok_s", {}).get("median") if isinstance(d_ctrl.get("decode_tok_s"), dict) else None
        st = d_same.get("tau", {}).get("median") if isinstance(d_same.get("tau"), dict) else None
        ct = d_ctrl.get("tau", {}).get("median") if isinstance(d_ctrl.get("tau"), dict) else None
        sd_s = f"{sd:.1f}" if isinstance(sd, (int, float)) else ("—" if p["tier"] else "—")
        cd_s = f"{cd:.1f}" if isinstance(cd, (int, float)) else ("—" if cid.startswith("mq4") else "—")
        st_s = f"{st:.2f}" if isinstance(st, (int, float)) else "—"
        ct_s = f"{ct:.2f}" if isinstance(ct, (int, float)) else ("—" if cid.startswith("mq4") else "—")
        choice = p["dflash"]["choice"] or {}
        chosen = choice.get("chosen_draft", "—")
        md_lines.append(f"| {cid} | {bpw} | {k1} | {k2} | {ar_ds} | {ar_ps} | {sd_s} | {cd_s} | {st_s} | {ct_s} | {chosen} |")
    md_lines.append("")
    md_lines.append("Acceptance: `null` with `unavailable_in_native_generate_v1` (native-generate-v1 never reports it; never derived).")
    md_lines.append("")
    md_lines.append(f"Artifacts: `{manifest.get('parent', '')}` → `artifacts/` (bytes/bpw/codec/tier/fixed-tier/commit tracked per row). Drafts: `{draft_by_id.get(CONTROL_DRAFT, {}).get('draft_path','')}` control vs same-bit (choose higher fresh-process median DFlash decode tok/s, tie → smaller draft bytes, both preserved). All values source-linked via `raw_paths`/per-rep `sources` in `summary.json`.")
    md_lines.append("")
    md_lines.append(f"Binaries: `hipfire-quantize` {binaries.get('hipfire_quantize', {}).get('sha256','')[:12]}…, `dflash_convert` {binaries.get('dflash_convert', {}).get('sha256','')[:12]}…, `hipfire` {binaries.get('hipfire_bench', {}).get('sha256','')[:12]}…, `eval_hipfire` {binaries.get('eval_hipfire', {}).get('sha256','')[:12]}…. Prompt: `{prompt_digest.get('path','')} sha256 {str(prompt_digest.get('sha256',''))[:12]}…`.")
    md_lines.append("")
    md = "\n".join(md_lines) + "\n"
    atomic_write(out_md, md)

    if errors:
        eprint(f"summarize FAILED with {len(errors)} errors; wrote {out_json}, {out_csv}, {out_md}")
        for e in errors:
            eprint(f"  error: {e}")
        return 1
    eprint(f"summarize OK: wrote {out_json}, {out_csv}, {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
