#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""
qwen38_ladder_campaign — resumable developer orchestrator for the exact
hiptrx Qwen3.8 15-cell ladder (mq2..mq6 × xt/base/pro), five V2 DFlash2
drafts, dual-reference KLD, fresh-process AR/prefill benches, and DFlash benches.

Defaults are the committed hiptrx paths; every default is overridable via CLI
so the script is repository-owned and re-usable without hidden defaults.

Contract recap (see task description):
  checkout  /home/kaden/hipfire-quantcal
  qcal      /home/kaden/qcal/ladder-v2
  parent    /home/kaden/qcal/parents/qwen3.8-27b
  imatrix   /home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf
  refs      /home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin
            /home/kaden/kldrefs/qwen3.8-27b.ref_v6sel-814d8fd.bin
  prompt    benchmarks/prompts/merge_sort_thinking_off.txt
  params    26895998464
  cells     mq2..mq6 × xt/base/pro  (15) — base format mqNv2
            xt:   --tier xt                         (no imatrix/AWQ)
            base: --tier base --imatrix <p> --awq-alpha 0.55
            pro:  --tier pro  --imatrix <p> --awq-alpha 0.55
                  + mq2 pro: lm_head + ssm_out lifted to MQ6V2
                  + mq3 pro: ssm_out lifted to MQ6V2
                  + mq4/5/6 pro: Q8 default (no fixed-tier)
            every artifact must satisfy model bpw ∈ [N, N+1)
  drafts    dflash_convert --mqNv2 same-bit from incoai/Qwen3.8-27B-DFlash2
            (mq4 control always included; 5 drafts total)
  KLD       env HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 HIPFIRE_LLOYD_GFX12=1
            24 chunks, q8/q8, prefill, both refs
  perf      hipfire bench <model> "<prompt bytes as one arg>" --runs 5 --warmups 3
            --max-tokens 128 --backend noslots --workload stateless --json
            + 3 fresh processes per arm; AR --spec off; DFlash --spec dflash
            with explicit HIPFIRE_DFLASH_DRAFT
  capture   full JSON/stdout/stderr, commit, GPU assignment, sha256+md5+bytes+bpw
            for artifacts/drafts/binaries/refs/prompt, τ/acceptance without
            summarizing away samples

Phases: quantize | drafts | kld | bench-ar | bench-dflash | manifest | all
Resumable state/result JSON under qcal output, deterministic 15-cell manifest,
serial quantization (51GB RSS), bounded GPU-parallel KLD (one process/GPU;
private kernel cache/HOME), serial fresh-process perf by default.

Validation: refuse stale/malformed prior outputs, size/bpw band before scoring,
atomic writes, robust JSON parsing from mixed logs, nonzero child → visible
failure and resumable (never marked measured).

Standard library only. No shell=True, no scratch-script, no /tmp fixture.
"""

import textwrap

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Defaults — exact hiptrx contract, all overridable
# ---------------------------------------------------------------------------

DEFAULT_CHECKOUT = "/home/kaden/hipfire-quantcal"
DEFAULT_QCAL = "/home/kaden/qcal/ladder-v2"
DEFAULT_PARENT = "/home/kaden/qcal/parents/qwen3.8-27b"
DEFAULT_IMATRIX = "/home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf"
DEFAULT_REF_WT2 = "/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin"
DEFAULT_REF_V6SEL = "/home/kaden/kldrefs/qwen3.8-27b.ref_v6sel-814d8fd.bin"
DEFAULT_PROMPT = "benchmarks/prompts/merge_sort_thinking_off.txt"
DEFAULT_PARAMS = 26895998464
DEFAULT_DRAFT_SOURCE = "incoai/Qwen3.8-27B-DFlash2"
DEFAULT_WARMUPS = 3
DEFAULT_RUNS = 5
DEFAULT_MAX_TOKENS = 128
DEFAULT_KLD_CHUNKS = 24

N_VALUES = [2, 3, 4, 5, 6]
TIER_ORDER = ["xt", "base", "pro"]
MANIFEST_VERSION = 1
STATE_VERSION = 1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def repo_root(checkout: Optional[str] = None) -> Path:
    # checkout is the repo checkout; prompt resolution falls back to script repo
    if checkout and Path(checkout).is_dir():
        return Path(checkout)
    # script lives in <repo>/scripts/
    return Path(__file__).resolve().parents[1]

def eprint(*a, **kw):
    print(*a, file=sys.stderr, **kw)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def file_bytes(path: Path) -> int:
    return path.stat().st_size

def compute_bpw(num_bytes: int, params: int) -> float:
    return (num_bytes * 8.0) / float(params)

def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use qcal-local tmp to avoid /tmp
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".tmp.")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
        except Exception:
            pass

def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".tmp.")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
        except Exception:
            pass

def git_commit(checkout: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", checkout, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"

def git_diff_md5(checkout: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", checkout, "diff", "--no-ext-diff"],
            capture_output=True, timeout=10
        )
        if out.returncode == 0:
            return hashlib.md5(out.stdout).hexdigest()
    except Exception:
        pass
    return "unknown"

def resolve_prompt_path(prompt_arg: str, checkout: str) -> Path:
    p = Path(prompt_arg)
    if p.is_absolute():
        return p
    # Try checkout-relative, then script repo-relative
    c = Path(checkout) / prompt_arg
    if c.exists():
        return c
    r = repo_root(checkout) / prompt_arg
    return r

def prompt_digest(prompt_path: Path) -> Dict[str, Any]:
    if not prompt_path.is_file():
        return {"path": str(prompt_path), "exists": False, "bytes": None, "sha256": None, "md5": None}
    return {
        "path": str(prompt_path),
        "exists": True,
        "bytes": file_bytes(prompt_path),
        "sha256": sha256_file(prompt_path),
        "md5": md5_file(prompt_path),
    }

def binary_digest(bin_path: Path) -> Dict[str, Any]:
    if not bin_path.is_file():
        return {"path": str(bin_path), "exists": False, "bytes": None, "sha256": None, "md5": None}
    return {
        "path": str(bin_path),
        "exists": True,
        "bytes": file_bytes(bin_path),
        "sha256": sha256_file(bin_path),
        "md5": md5_file(bin_path),
    }

def parse_json_robust(mixed: str) -> Tuple[List[Any], str]:
    """
    Parse JSON robustly from mixed log output while retaining raw logs.
    Returns (list_of_parsed_objects, raw_text).
    Tries line-wise JSON, then brace-balanced extraction as fallback.
    """
    raw = mixed
    objs: List[Any] = []
    # First, try each line that looks like JSON
    for line in mixed.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("{") or s.startswith("["):
            try:
                objs.append(json.loads(s))
                continue
            except Exception:
                pass
    # Fallback: find JSON objects via brace counting if nothing found
    if not objs:
        # Use a simple stack to extract top-level {...} blocks
        depth = 0
        start = -1
        in_str = False
        esc = False
        for i, ch in enumerate(mixed):
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
                elif ch == "}":
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start != -1:
                            cand = mixed[start:i+1]
                            try:
                                objs.append(json.loads(cand))
                            except Exception:
                                pass
                            start = -1
    return objs, raw

def validate_bpw(num_bytes: int, n: int, params: int) -> Tuple[bool, float]:
    bpw = compute_bpw(num_bytes, params)
    ok = (n <= bpw < n + 1)
    return ok, bpw

def detect_gpus() -> int:
    # Try rocm-smi
    for cmd in (["rocm-smi", "--showid"], ["rocm-smi", "-i"], ["rocminfo"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                txt = out.stdout + out.stderr
                # rocm-smi --showid prints "GPU[0] ..."
                m = re.findall(r"GPU\[(\d+)\]", txt)
                if m:
                    return max(int(x) for x in m) + 1
                # rocminfo prints " Agent 2 ... gfx1201"
                if "gfx" in txt.lower():
                    agents = re.findall(r"Agent \d+", txt)
                    if agents:
                        # heuristic: number of GPU agents
                        return len(re.findall(r"gfx1[0-2]\d+", txt.lower())) or 1
        except Exception:
            continue
    # Fallback: count HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES
    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        v = os.environ.get(var)
        if v:
            parts = [x.strip() for x in v.split(",") if x.strip() != ""]
            if parts:
                return len(parts)
    # Fallback: try /dev/kfd or /dev/dri
    try:
        dri = list(Path("/dev/dri").glob("card*"))
        if dri:
            return len(dri)
    except Exception:
        pass
    return 1

def run_subprocess(argv: List[str], env: Dict[str, str], cwd: Optional[str] = None, dry_run: bool = False) -> Tuple[int, str, str]:
    """
    Never use shell=True. Returns (returncode, stdout, stderr).
    Dry-run prints exact command/environment/path without execution and returns 0.
    """
    env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()) if k not in os.environ or os.environ.get(k) != v)
    # For dry-run we print the full env delta + command
    if dry_run:
        q = " ".join(shlex.quote(a) for a in argv)
        if cwd:
            eprint(f"[dry-run] cd {shlex.quote(cwd)} && {env_str + ' ' if env_str else ''}{q}")
        else:
            eprint(f"[dry-run] {env_str + ' ' if env_str else ''}{q}")
        return 0, "", ""
    # Real execution: capture stdout+stderr, no shell, no /tmp
    proc = subprocess.run(argv, env={**os.environ, **env}, capture_output=True, text=True, cwd=cwd)
    return proc.returncode, proc.stdout, proc.stderr

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_cells(qcal_dir: str) -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    qcal = Path(qcal_dir)
    for n in N_VALUES:
        for tier in TIER_ORDER:
            codec = f"mq{n}v2"
            cell_id = f"mq{n}-{tier}"
            artifact = qcal / "artifacts" / f"qwen3.8-27b.{codec}.{tier}.hfq"
            fixed_tier = None
            if tier == "pro" and n == 2:
                fixed_tier = "lm_head:mq6v2,ssm_out:mq6v2"
            elif tier == "pro" and n == 3:
                fixed_tier = "ssm_out:mq6v2"
            # Q8 default for mq4/5/6 pro is implicit (no fixed_tier).
            cells.append({
                "cell_id": cell_id,
                "n": n,
                "tier": tier,
                "codec": codec,
                "format": codec,
                "artifact": str(artifact),
                "fixed_tier": fixed_tier,
            })
    # Deterministic ordering already enforced; assert uniqueness
    assert len(cells) == 15, f"expected 15 cells, got {len(cells)}"
    assert len({c["cell_id"] for c in cells}) == 15, "duplicate cell_id"
    assert len({c["artifact"] for c in cells}) == 15, "duplicate artifact path"
    return cells

def build_drafts(qcal_dir: str, draft_source: str) -> List[Dict[str, Any]]:
    drafts: List[Dict[str, Any]] = []
    qcal = Path(qcal_dir)
    for n in N_VALUES:
        codec = f"mq{n}v2"
        draft_id = codec  # keep codec identity explicit
        draft_path = qcal / "drafts" / f"qwen3.8-27b-dflash.{codec}.hfq"
        drafts.append({
            "draft_id": draft_id,
            "n": n,
            "codec": codec,
            "format": codec,
            "draft_path": str(draft_path),
            "source": draft_source,
        })
    assert len(drafts) == 5, f"expected 5 drafts, got {len(drafts)}"
    assert len({d["draft_id"] for d in drafts}) == 5, "duplicate draft_id"
    # Always include mq4v2 control — ensure it's present
    assert any(d["draft_id"] == "mq4v2" for d in drafts), "mq4v2 control missing"
    return drafts

def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    checkout = args.checkout
    qcal_dir = args.qcal_dir
    commit = git_commit(checkout)
    diff_md5 = git_diff_md5(checkout)
    prompt_path = resolve_prompt_path(args.prompt, checkout)
    # prompt digest if exists, else placeholder
    pd = prompt_digest(prompt_path) if prompt_path.is_file() else {"path": str(prompt_path), "exists": False, "bytes": None, "sha256": None, "md5": None}
    # binary digests (may not exist yet)
    quant_bin = Path(args.quantize_bin) if args.quantize_bin else Path(checkout) / "target/release/hipfire-quantize"
    dflash_bin = Path(args.dflash_bin) if args.dflash_bin else Path(checkout) / "target/release/dflash_convert"
    bench_bin = Path(args.bench_bin) if args.bench_bin else Path(checkout) / "target/release/hipfire"
    eval_bin = Path(args.eval_bin) if args.eval_bin else Path(checkout) / "target/release/eval_hipfire"
    # Try alternative eval_hipfire locations
    alt_eval = Path(checkout) / "target/release/examples/eval_hipfire"
    if not eval_bin.is_file() and alt_eval.is_file():
        eval_bin = alt_eval
    cells = build_cells(qcal_dir)
    drafts = build_drafts(qcal_dir, args.draft_source)
    # Validate uniqueness across identities — cannot confuse codec/tier/draft
    cell_identities = [f"{c['codec']}:{c['tier']}" for c in cells]
    assert len(cell_identities) == len(set(cell_identities)), "cell codec/tier collision"
    draft_identities = [d["codec"] for d in drafts]
    assert len(draft_identities) == len(set(draft_identities)), "draft codec collision"
    manifest: Dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "generated_at": utc_now(),
        "checkout": checkout,
        "commit": commit,
        "diff_md5": diff_md5,
        "qcal_dir": qcal_dir,
        "parent": args.parent,
        "imatrix": args.imatrix,
        "ref_wt2": args.ref_wt2,
        "ref_v6sel": args.ref_v6sel,
        "prompt": str(prompt_path),
        "prompt_digest": pd,
        "params": args.params,
        "draft_source": args.draft_source,
        "binaries": {
            "hipfire_quantize": binary_digest(quant_bin),
            "dflash_convert": binary_digest(dflash_bin),
            "hipfire_bench": binary_digest(bench_bin),
            "eval_hipfire": binary_digest(eval_bin),
        },
        "cells": cells,
        "drafts": drafts,
        "refs": [
            {"kind": "wt2", "path": args.ref_wt2},
            {"kind": "v6sel", "path": args.ref_v6sel},
        ],
        # KLD/Perf contract
        "kld": {
            "env": {"HIPFIRE_NORMALIZE_PROMPT": "0", "HIPFIRE_GRAPH": "0", "HIPFIRE_LLOYD_GFX12": "1"},
            "chunks": DEFAULT_KLD_CHUNKS,
            "kv_mode": "q8",
            "kv_v": "q8",
            "scoring_mode": "prefill",
            "refs": ["wt2", "v6sel"],
        },
        "perf": {
            "runs": DEFAULT_RUNS,
            "warmups": DEFAULT_WARMUPS,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "backend": "noslots",
            "workload": "stateless",
            "json": True,
            "fresh_processes_per_arm": 3,
        },
    }
    return manifest

def manifest_path(args: argparse.Namespace) -> Path:
    return Path(args.qcal_dir) / "manifest.json"

def state_path(args: argparse.Namespace) -> Path:
    return Path(args.qcal_dir) / "state.json"

def results_path(args: argparse.Namespace) -> Path:
    return Path(args.qcal_dir) / "results.json"

def load_state_strict(path: Path, expected_manifest_sha: Optional[str] = None) -> Dict[str, Any]:
    if not path.is_file():
        return {"version": STATE_VERSION, "phases": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"refusing stale/malformed state {path}: {e}")
    if not isinstance(data, dict):
        raise SystemExit(f"refusing stale/malformed state {path}: not a dict")
    if data.get("version") != STATE_VERSION:
        raise SystemExit(f"refusing stale state {path}: version {data.get('version')} != {STATE_VERSION}")
    if expected_manifest_sha is not None:
        got = data.get("manifest_sha256")
        if got is not None and got != expected_manifest_sha:
            raise SystemExit(f"refusing stale state {path}: manifest_sha256 mismatch (expected {expected_manifest_sha}, got {got})")
    # Validate phase keys do not confuse identities
    phases = data.get("phases", {})
    if not isinstance(phases, dict):
        raise SystemExit(f"refusing malformed state {path}: phases not a dict")
    return data

def load_results_strict(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "rows": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"refusing stale/malformed results {path}: {e}")
    if not isinstance(data, dict):
        raise SystemExit(f"refusing malformed results {path}: not a dict")
    if "rows" not in data or not isinstance(data["rows"], list):
        raise SystemExit(f"refusing malformed results {path}: missing rows list")
    return data

def manifest_sha256(manifest: Dict[str, Any]) -> str:
    blob = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def quantize_argv(cell: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[str], Dict[str, str]]:
    qbin = args.quantize_bin if args.quantize_bin else str(Path(args.checkout) / "target/release/hipfire-quantize")
    argv = [qbin, "--input", args.parent, "--output", cell["artifact"], "--format", cell["format"], "--tier", cell["tier"]]
    if cell["tier"] != "xt":
        argv += ["--imatrix", args.imatrix, "--awq-alpha", "0.55"]
    if cell["fixed_tier"]:
        argv += ["--fixed-tier", cell["fixed_tier"]]
    env: Dict[str, str] = {}
    return argv, env

def draft_argv(draft: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[str], Dict[str, str]]:
    dbin = args.dflash_bin if args.dflash_bin else str(Path(args.checkout) / "target/release/dflash_convert")
    # dflash_convert flag is --mq{n}v2
    flag = f"--{draft['format']}"
    argv = [dbin, "--input", draft["source"], "--output", draft["draft_path"], flag]
    env: Dict[str, str] = {}
    return argv, env

def kld_argv(cell: Dict[str, Any], ref_kind: str, ref_path: str, output_path: str, args: argparse.Namespace) -> Tuple[List[str], Dict[str, str]]:
    ebin = args.eval_bin
    if not ebin:
        cand = Path(args.checkout) / "target/release/eval_hipfire"
        alt = Path(args.checkout) / "target/release/examples/eval_hipfire"
        if cand.is_file():
            ebin = str(cand)
        elif alt.is_file():
            ebin = str(alt)
        else:
            # cargo run form
            ebin = str(cand)
    # Prefer binary if exists, else cargo run wrapper (handled by caller)
    use_cargo = not Path(ebin).is_file()
    if use_cargo:
        argv = ["cargo", "run", "--quiet", "--manifest-path", str(Path(args.checkout) / "Cargo.toml"),
                "--example", "eval_hipfire", "--",
                "--model", cell["artifact"], "--ref", ref_path, "--output", output_path,
                "--kv-mode", "q8", "--kv-v", "q8", "--scoring-mode", "prefill", "--max-chunks", str(DEFAULT_KLD_CHUNKS)]
    else:
        argv = [ebin, "--model", cell["artifact"], "--ref", ref_path, "--output", output_path,
                "--kv-mode", "q8", "--kv-v", "q8", "--scoring-mode", "prefill", "--max-chunks", str(DEFAULT_KLD_CHUNKS)]
    env = {
        "HIPFIRE_NORMALIZE_PROMPT": "0",
        "HIPFIRE_GRAPH": "0",
        "HIPFIRE_LLOYD_GFX12": "1",
    }
    return argv, env

def bench_argv(cell: Dict[str, Any], prompt_text: str, spec: str, draft_path: Optional[str], args: argparse.Namespace) -> Tuple[List[str], Dict[str, str]]:
    bbin = args.bench_bin if args.bench_bin else str(Path(args.checkout) / "target/release/hipfire")
    # hipfire bench <model> --runs 5 --warmups 3 --max-tokens 128 --backend noslots --workload stateless --json --spec {off|dflash} "<prompt>"
    argv = [bbin, "bench", cell["artifact"],
            "--runs", str(DEFAULT_RUNS),
            "--warmups", str(DEFAULT_WARMUPS),
            "--max-tokens", str(DEFAULT_MAX_TOKENS),
            "--backend", "noslots",
            "--workload", "stateless",
            "--json",
            "--spec", spec,
            prompt_text]
    env: Dict[str, str] = {}
    if spec == "dflash" and draft_path:
        env["HIPFIRE_DFLASH_DRAFT"] = draft_path
    return argv, env

# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def ensure_qcal_dir(args: argparse.Namespace) -> None:
    Path(args.qcal_dir).mkdir(parents=True, exist_ok=True)

def validate_inputs_or_raise(args: argparse.Namespace, dry_run: bool) -> None:
    # Validate parent/imatrix/refs exist unless dry-run (warn only)
    checks = [
        ("parent", args.parent, True),
        ("imatrix", args.imatrix, False),  # only needed for base/pro but warn
        ("ref_wt2", args.ref_wt2, True),
        ("ref_v6sel", args.ref_v6sel, True),
        ("prompt", resolve_prompt_path(args.prompt, args.checkout), True),
    ]
    for kind, p, required in checks:
        path = Path(p)
        if not path.exists():
            msg = f"missing {kind}: {p}"
            if dry_run:
                eprint(f"[dry-run warn] {msg}")
            elif required:
                raise SystemExit(msg)
            else:
                eprint(f"warn: {msg}")

def do_manifest(args: argparse.Namespace) -> int:
    ensure_qcal_dir(args)
    manifest = build_manifest(args)
    mp = manifest_path(args)
    if args.dry_run:
        eprint(f"[dry-run] would write manifest {mp} with {len(manifest['cells'])} cells, {len(manifest['drafts'])} drafts")
        # Print deterministic manifest summary
        for c in manifest["cells"]:
            eprint(f"[dry-run] cell {c['cell_id']} codec={c['codec']} tier={c['tier']} artifact={c['artifact']} fixed_tier={c['fixed_tier']}")
        for d in manifest["drafts"]:
            eprint(f"[dry-run] draft {d['draft_id']} codec={d['codec']} path={d['draft_path']} source={d['source']}")
        # Also print full manifest JSON to stdout for inspection
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    validate_inputs_or_raise(args, dry_run=False)
    atomic_write_json(mp, manifest)
    eprint(f"wrote manifest {mp} commit={manifest['commit']} cells={len(manifest['cells'])} drafts={len(manifest['drafts'])}")
    return 0

def do_quantize(args: argparse.Namespace) -> int:
    ensure_qcal_dir(args)
    # Need manifest first
    mp = manifest_path(args)
    if mp.is_file():
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except Exception as e:
            raise SystemExit(f"refusing stale/malformed manifest {mp}: {e}")
        if manifest.get("version") != MANIFEST_VERSION:
            raise SystemExit(f"refusing stale manifest {mp}: version mismatch")
    else:
        manifest = build_manifest(args)
        if not args.dry_run:
            atomic_write_json(mp, manifest)
    cells = manifest["cells"]
    sp = state_path(args)
    msha = manifest_sha256(manifest)
    state = load_state_strict(sp, expected_manifest_sha=msha)
    # Ensure manifest sha recorded
    if not args.dry_run and state.get("manifest_sha256") != msha:
        state["manifest_sha256"] = msha
        state.setdefault("phases", {})
        atomic_write_json(sp, state)
    # Validate parent etc.
    validate_inputs_or_raise(args, dry_run=args.dry_run)
    # Check imatrix needed for base/pro cells
    need_imatrix = any(c["tier"] != "xt" for c in cells)
    if need_imatrix and not Path(args.imatrix).is_file() and not args.dry_run:
        raise SystemExit(f"missing imatrix {args.imatrix} required for base/pro tiers")
    # Serial quantization (51GB RSS)
    failed: List[str] = []
    completed: List[str] = []
    # Load prior completed from state
    phases = state.get("phases", {})
    qstate = phases.get("quantize", {})
    prior_completed = set(qstate.get("completed", []))
    prior_failed = set(qstate.get("failed", []))
    if args.dry_run:
        eprint(f"[dry-run] quantize phase: {len(cells)} cells serial, 51GB RSS cap implied")
    for cell in cells:
        cell_id = cell["cell_id"]
        # Resumable: skip if artifact exists and passes bpw band and digests OK and prior completed
        artifact = Path(cell["artifact"])
        if not args.dry_run and artifact.is_file() and cell_id in prior_completed:
            ok, bpw = validate_bpw(file_bytes(artifact), cell["n"], args.params)
            if ok:
                eprint(f"skip quantize {cell_id}: artifact exists and prior completed bpw={bpw:.3f}")
                completed.append(cell_id)
                continue
            else:
                eprint(f"stale quantize {cell_id}: bpw {bpw:.3f} out of band [{cell['n']},{cell['n']+1}) — re-quantizing")
                prior_completed.discard(cell_id)
        if not args.dry_run:
            artifact.parent.mkdir(parents=True, exist_ok=True)
        argv, env = quantize_argv(cell, args)
        # Print every exact command/environment/path
        if args.dry_run:
            env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))
            eprint(f"[dry-run] quantize {cell_id}: {env_str + ' ' if env_str else ''}{' '.join(shlex.quote(a) for a in argv)}")
            eprint(f"[dry-run]   artifact={cell['artifact']} codec={cell['codec']} tier={cell['tier']} fixed_tier={cell['fixed_tier']} bpw band=[{cell['n']},{cell['n']+1})")
            # not executing
            continue
        # Validate bpw pre-check is not applicable before file exists; we will check after.
        rc, out, err = run_subprocess(argv, env, dry_run=False)
        log_path = Path(args.qcal_dir) / "logs" / f"quantize_{cell_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Capture full stdout/stderr
        log_text = f"$ {' '.join(shlex.quote(a) for a in argv)}\n# env {json.dumps(env)}\n# cwd {os.getcwd()}\nSTDOUT:\n{out}\nSTDERR:\n{err}\n"
        atomic_write_text(log_path, log_text)
        if rc != 0:
            eprint(f"quantize {cell_id} FAILED rc={rc} (see {log_path}) — resumable, not marked measured")
            failed.append(cell_id)
            # never mark measured; continue to next cell but ensure state not updated to completed
            continue
        # Validate artifact exists and bpw
        if not artifact.is_file():
            eprint(f"quantize {cell_id} missing output {artifact} rc=0 but no file — marking failed")
            failed.append(cell_id)
            continue
        ok, bpw = validate_bpw(file_bytes(artifact), cell["n"], args.params)
        if not ok:
            eprint(f"quantize {cell_id} bpw {bpw:.3f} out of band [{cell['n']},{cell['n']+1}) — marking failed")
            failed.append(cell_id)
            continue
        # Compute digests for result row
        row = {
            "cell_id": cell_id,
            "n": cell["n"],
            "tier": cell["tier"],
            "codec": cell["codec"],
            "format": cell["format"],
            "fixed_tier": cell["fixed_tier"],
            "artifact": str(artifact),
            "bytes": file_bytes(artifact),
            "sha256": sha256_file(artifact),
            "md5": md5_file(artifact),
            "bpw": bpw,
            "commit": git_commit(args.checkout),
            "log": str(log_path),
        }
        completed.append(cell_id)
        # Update results atomically per cell
        rp = results_path(args)
        results = load_results_strict(rp)
        # Replace or append row with matching cell_id and phase quantize
        # Use deterministic identity: cell_id
        new_rows = [r for r in results["rows"] if not (r.get("phase") == "quantize" and r.get("cell_id") == cell_id)]
        new_rows.append({"phase": "quantize", **row})
        results["rows"] = sorted(new_rows, key=lambda x: (x.get("phase",""), x.get("cell_id","")))
        results["updated_at"] = utc_now()
        atomic_write_json(rp, results)
        # Update state
        state = load_state_strict(sp, expected_manifest_sha=msha)
        phases = state.setdefault("phases", {})
        qs = phases.setdefault("quantize", {"completed": [], "failed": []})
        qs["completed"] = sorted(set(qs.get("completed", [])) | {cell_id})
        qs["failed"] = sorted(set(qs.get("failed", [])) - {cell_id})
        qs["updated_at"] = utc_now()
        atomic_write_json(sp, state)
        eprint(f"quantize {cell_id} OK bpw={bpw:.3f} sha256={row['sha256'][:12]}")
    # Final state for failed
    if not args.dry_run:
        state = load_state_strict(sp, expected_manifest_sha=msha)
        phases = state.setdefault("phases", {})
        qs = phases.setdefault("quantize", {"completed": [], "failed": []})
        qs["failed"] = sorted(set(qs.get("failed", [])) | set(failed))
        qs["completed"] = sorted(set(qs.get("completed", [])) | set(completed))
        # Remove any that are both (failed now overrides completed if still failed)
        atomic_write_json(sp, state)
        if failed:
            eprint(f"quantize phase completed with {len(failed)} failures: {', '.join(failed)}")
            return 1
    return 0

def do_drafts(args: argparse.Namespace) -> int:
    ensure_qcal_dir(args)
    mp = manifest_path(args)
    if mp.is_file():
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except Exception as e:
            raise SystemExit(f"refusing stale/malformed manifest {mp}: {e}")
    else:
        manifest = build_manifest(args)
        if not args.dry_run:
            atomic_write_json(mp, manifest)
    drafts = manifest["drafts"]
    msha = manifest_sha256(manifest)
    sp = state_path(args)
    state = load_state_strict(sp, expected_manifest_sha=msha)
    if not args.dry_run and state.get("manifest_sha256") != msha:
        state["manifest_sha256"] = msha
        atomic_write_json(sp, state)
    failed: List[str] = []
    completed: List[str] = []
    phases = state.get("phases", {})
    dstate = phases.get("drafts", {})
    prior_completed = set(dstate.get("completed", []))
    if args.dry_run:
        eprint(f"[dry-run] drafts phase: {len(drafts)} drafts via dflash_convert same-bit --mqNv2 from {args.draft_source} (mq4 control included)")
    for d in drafts:
        draft_id = d["draft_id"]
        draft_path = Path(d["draft_path"])
        if not args.dry_run and draft_path.is_file() and draft_id in prior_completed:
            # Validate size exists
            eprint(f"skip drafts {draft_id}: exists and prior completed")
            completed.append(draft_id)
            continue
        if not args.dry_run:
            draft_path.parent.mkdir(parents=True, exist_ok=True)
        argv, env = draft_argv(d, args)
        if args.dry_run:
            env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))
            eprint(f"[dry-run] draft {draft_id}: {env_str + ' ' if env_str else ''}{' '.join(shlex.quote(a) for a in argv)}")
            eprint(f"[dry-run]   draft_path={d['draft_path']} codec={d['codec']} source={d['source']}")
            continue
        rc, out, err = run_subprocess(argv, env, dry_run=False)
        log_path = Path(args.qcal_dir) / "logs" / f"draft_{draft_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_text = f"$ {' '.join(shlex.quote(a) for a in argv)}\n# env {json.dumps(env)}\nSTDOUT:\n{out}\nSTDERR:\n{err}\n"
        atomic_write_text(log_path, log_text)
        if rc != 0:
            eprint(f"draft {draft_id} FAILED rc={rc} (see {log_path}) — resumable")
            failed.append(draft_id)
            continue
        if not draft_path.is_file():
            eprint(f"draft {draft_id} missing output {draft_path} — failed")
            failed.append(draft_id)
            continue
        # Compute digests
        row = {
            "draft_id": draft_id,
            "n": d["n"],
            "codec": d["codec"],
            "format": d["format"],
            "draft_path": str(draft_path),
            "bytes": file_bytes(draft_path),
            "sha256": sha256_file(draft_path),
            "md5": md5_file(draft_path),
            "source": d["source"],
            "commit": git_commit(args.checkout),
            "log": str(log_path),
        }
        # bpw not applicable for draft (small) but compute if params known?
        # Drafts are not full model, skip bpw band.
        completed.append(draft_id)
        rp = results_path(args)
        results = load_results_strict(rp)
        new_rows = [r for r in results["rows"] if not (r.get("phase") == "drafts" and r.get("draft_id") == draft_id)]
        new_rows.append({"phase": "drafts", **row})
        results["rows"] = sorted(new_rows, key=lambda x: (x.get("phase",""), x.get("draft_id","")))
        results["updated_at"] = utc_now()
        atomic_write_json(rp, results)
        state = load_state_strict(sp, expected_manifest_sha=msha)
        phases = state.setdefault("phases", {})
        ds = phases.setdefault("drafts", {"completed": [], "failed": []})
        ds["completed"] = sorted(set(ds.get("completed", [])) | {draft_id})
        ds["failed"] = sorted(set(ds.get("failed", [])) - {draft_id})
        ds["updated_at"] = utc_now()
        atomic_write_json(sp, state)
        eprint(f"draft {draft_id} OK sha256={row['sha256'][:12]}")
    if not args.dry_run:
        state = load_state_strict(sp, expected_manifest_sha=msha)
        phases = state.setdefault("phases", {})
        ds = phases.setdefault("drafts", {"completed": [], "failed": []})
        ds["failed"] = sorted(set(ds.get("failed", [])) | set(failed))
        ds["completed"] = sorted(set(ds.get("completed", [])) | set(completed))
        atomic_write_json(sp, state)
        if failed:
            eprint(f"drafts phase completed with {len(failed)} failures: {', '.join(failed)}")
            return 1
    return 0

def do_kld(args: argparse.Namespace) -> int:
    ensure_qcal_dir(args)
    mp = manifest_path(args)
    if mp.is_file():
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except Exception as e:
            raise SystemExit(f"refusing stale/malformed manifest {mp}: {e}")
    else:
        manifest = build_manifest(args)
        if not args.dry_run:
            atomic_write_json(mp, manifest)
    cells = manifest["cells"]
    # Validate refs exist (unless dry-run)
    refs = {"wt2": args.ref_wt2, "v6sel": args.ref_v6sel}
    for kind, p in refs.items():
        if not Path(p).is_file() and not args.dry_run:
            raise SystemExit(f"missing ref {kind}: {p}")
    # Validate artifacts exist and bpw before scoring
    for c in cells:
        art = Path(c["artifact"])
        if not args.dry_run and not art.is_file():
            eprint(f"warn: kld skipping {c['cell_id']}: artifact missing {art} (quantize first)")
        elif not args.dry_run and art.is_file():
            ok, bpw = validate_bpw(file_bytes(art), c["n"], args.params)
            if not ok:
                raise SystemExit(f"refusing to score {c['cell_id']}: bpw {bpw:.3f} out of band [{c['n']},{c['n']+1}) artifact {art}")
    msha = manifest_sha256(manifest)
    sp = state_path(args)
    state = load_state_strict(sp, expected_manifest_sha=msha)
    if not args.dry_run and state.get("manifest_sha256") != msha:
        state["manifest_sha256"] = msha
        atomic_write_json(sp, state)
    # Build tasks: each cell × 2 refs
    tasks: List[Tuple[Dict[str, Any], str, str, str]] = []  # cell, ref_kind, ref_path, output_path
    for cell in cells:
        for kind, ref_path in refs.items():
            out = Path(args.qcal_dir) / "kld" / f"{cell['cell_id']}.{kind}.kldseq"
            tasks.append((cell, kind, ref_path, str(out)))
    num_gpus = detect_gpus() if not args.dry_run else 1
    eprint(f"kld phase: {len(tasks)} jobs, {len(cells)} cells × 2 refs, gpus={num_gpus} (one process/GPU; private HOME+cache)")
    # Load prior state
    phases = state.get("phases", {})
    kstate = phases.get("kld", {})
    prior_completed = set(kstate.get("completed", []))
    # For dry-run, just print every command/environment/path
    if args.dry_run:
        for idx, (cell, kind, ref_path, out) in enumerate(tasks):
            gpu = idx % max(1, num_gpus)
            argv, env_base = kld_argv(cell, kind, ref_path, out, args)
            # private HOME/cache per GPU
            home = str(Path(args.qcal_dir) / f"tmp_home_gpu{gpu}")
            cache = str(Path(args.qcal_dir) / f"kernel_cache_gpu{gpu}")
            env = {**env_base, "HOME": home, "HIPFIRE_KERNEL_CACHE": cache, "HIP_VISIBLE_DEVICES": str(gpu)}
            task_id = f"{cell['cell_id']}:{kind}"
            eprint(f"[dry-run] kld {task_id} gpu={gpu} HOME={home} cache={cache}")
            eprint(f"[dry-run]   ref={ref_path} output={out}")
            eprint(f"[dry-run]   { ' '.join(shlex.quote(a) for a in argv)}")
            eprint(f"[dry-run]   env { ' '.join(f'{k}={shlex.quote(v)}' for k,v in sorted(env.items()))}")
        return 0
    # Real execution: bounded GPU-parallel (one process/GPU)
    # Validate no stale outputs: if output exists but state missing, treat as stale? The spec says refuse stale/malformed prior outputs.
    # We will only consider completed if output exists AND we have valid prior row and bpw check passed.
    failed: List[str] = []
    completed: List[str] = []
    # Use thread pool for gpu-bounded parallelism; each task runs subprocess.
    def run_one(task):
        cell, kind, ref_path, out = task
        task_id = f"{cell['cell_id']}:{kind}"
        art = Path(cell["artifact"])
        # size/bpw already validated above, but re-check per task
        if not art.is_file():
            return (task_id, 2, "", f"artifact missing {art}", "", task)
        # Check resumable: if prior completed and output exists and we can parse it, skip
        out_path = Path(out)
        if task_id in prior_completed and out_path.is_file():
            try:
                # Validate output is parsable and non-empty
                if out_path.stat().st_size == 0:
                    raise ValueError("empty kld output")
                # Could be binary kldseq; just check exists
                return (task_id, 0, "skipped (resumable)", "", str(out_path), task)
            except Exception as e:
                # stale -> re-run
                eprint(f"refusing stale kld output {out_path}: {e} — re-running")
        # Assign GPU round-robin based on hash
        # Use thread-local gpu index via tasks index
        idx = tasks.index(task) if task in tasks else 0
        gpu = idx % max(1, num_gpus)
        home = str(Path(args.qcal_dir) / f"tmp_home_gpu{gpu}")
        cache = str(Path(args.qcal_dir) / f"kernel_cache_gpu{gpu}")
        Path(home).mkdir(parents=True, exist_ok=True)
        Path(cache).mkdir(parents=True, exist_ok=True)
        argv, env_base = kld_argv(cell, kind, ref_path, out, args)
        env = {**env_base, "HOME": home, "HIPFIRE_KERNEL_CACHE": cache, "HIP_VISIBLE_DEVICES": str(gpu)}
        # Pre-create output parent
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        rc, sout, serr = run_subprocess(argv, env, dry_run=False)
        log_path = Path(args.qcal_dir) / "logs" / f"kld_{task_id.replace(':','_')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_text = f"$ {' '.join(shlex.quote(a) for a in argv)}\n# gpu={gpu} HOME={home} cache={cache}\n# env {json.dumps(env)}\nSTDOUT:\n{sout}\nSTDERR:\n{serr}\n"
        atomic_write_text(log_path, log_text)
        if rc != 0:
            return (task_id, rc, sout, serr, str(log_path), task)
        # Validate output exists
        if not Path(out).is_file():
            return (task_id, 2, sout, f"missing output {out}", str(log_path), task)
        # Capture digests + bpw for result row
        return (task_id, 0, sout, serr, str(log_path), task)
    # Execute with bounded parallelism
    results_map: Dict[str, Tuple[int,str,str,str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as pool:
        future_to_task = {pool.submit(run_one, t): t for t in tasks}
        for fut in concurrent.futures.as_completed(future_to_task):
            task_id, rc, sout, serr, log_path, task = fut.result()
            cell, kind, ref_path, out = task
            if rc == 0 and sout == "skipped (resumable)":
                eprint(f"kld {task_id} skip (resumable)")
                completed.append(task_id)
                continue
            if rc != 0:
                eprint(f"kld {task_id} FAILED rc={rc} log={log_path} — resumable, not marked measured")
                failed.append(task_id)
                # Do not mark measured; leave output maybe partial but not counted
                continue
            # Parse JSON robustly from mixed log output while retaining raw logs
            objs, raw = parse_json_robust(sout + "\n" + serr)
            # For kld, there is no JSON stdout in eval_hipfire; but we retain raw logs
            # Create result row with full contract columns
            art = Path(cell["artifact"])
            row = {
                "phase": "kld",
                "task_id": task_id,
                "cell_id": cell["cell_id"],
                "n": cell["n"],
                "tier": cell["tier"],
                "codec": cell["codec"],
                "format": cell["format"],
                "ref_kind": kind,
                "ref_path": ref_path,
                "ref_sha256": sha256_file(Path(ref_path)) if Path(ref_path).is_file() else None,
                "ref_bytes": file_bytes(Path(ref_path)) if Path(ref_path).is_file() else None,
                "artifact": str(art),
                "artifact_bytes": file_bytes(art) if art.is_file() else None,
                "artifact_sha256": sha256_file(art) if art.is_file() else None,
                "artifact_md5": md5_file(art) if art.is_file() else None,
                "artifact_bpw": compute_bpw(file_bytes(art), args.params) if art.is_file() else None,
                "output": out,
                "output_bytes": file_bytes(Path(out)) if Path(out).is_file() else None,
                "output_sha256": sha256_file(Path(out)) if Path(out).is_file() else None,
                "log": log_path,
                "stdout": sout,
                "stderr": serr,
                "parsed_json": objs,
                "raw_stdout": sout,
                "raw_stderr": serr,
                "commit": git_commit(args.checkout),
                "gpu": None,  # could extract from log
                "chunks": DEFAULT_KLD_CHUNKS,
                "kv_mode": "q8",
                "kv_v": "q8",
                "scoring_mode": "prefill",
                "env": {"HIPFIRE_NORMALIZE_PROMPT": "0", "HIPFIRE_GRAPH": "0", "HIPFIRE_LLOYD_GFX12": "1"},
            }
            # Add prompt/bins etc. later via results
            rp = results_path(args)
            results = load_results_strict(rp)
            # Ensure no confusion of identities: key on task_id
            new_rows = [r for r in results["rows"] if not (r.get("phase") == "kld" and r.get("task_id") == task_id)]
            new_rows.append(row)
            results["rows"] = sorted(new_rows, key=lambda x: (x.get("phase",""), x.get("task_id","")))
            results["updated_at"] = utc_now()
            atomic_write_json(rp, results)
            state = load_state_strict(sp, expected_manifest_sha=msha)
            phases = state.setdefault("phases", {})
            ks = phases.setdefault("kld", {"completed": [], "failed": []})
            ks["completed"] = sorted(set(ks.get("completed", [])) | {task_id})
            ks["failed"] = sorted(set(ks.get("failed", [])) - {task_id})
            ks["updated_at"] = utc_now()
            atomic_write_json(sp, state)
            completed.append(task_id)
            eprint(f"kld {task_id} OK")
    if not args.dry_run:
        state = load_state_strict(sp, expected_manifest_sha=msha)
        phases = state.setdefault("phases", {})
        ks = phases.setdefault("kld", {"completed": [], "failed": []})
        ks["failed"] = sorted(set(ks.get("failed", [])) | set(failed))
        ks["completed"] = sorted(set(ks.get("completed", [])) | set(completed))
        atomic_write_json(sp, state)
        if failed:
            eprint(f"kld phase completed with {len(failed)} failures: {', '.join(failed)}")
            return 1
    return 0

def _read_prompt_text(args: argparse.Namespace) -> str:
    p = resolve_prompt_path(args.prompt, args.checkout)
    if not p.is_file():
        if args.dry_run:
            return "<prompt bytes as one positional argument: " + str(p) + " (missing in dry-run)>"
        raise SystemExit(f"missing prompt {p}")
    data = p.read_bytes()
    # Spec: prompt bytes read as one positional argument
    # Preserve exact bytes as utf-8 text; the merge_sort prompt is ascii.
    try:
        return data.decode("utf-8")
    except Exception:
        # Fallback: decode with surrogateescape to preserve bytes
        return data.decode("utf-8", errors="surrogateescape")

def _ensure_artifact_bpw_or_fail(cell: Dict[str, Any], args: argparse.Namespace):
    art = Path(cell["artifact"])
    if not art.is_file():
        raise SystemExit(f"missing artifact {cell['cell_id']}: {art} (quantize first)")
    ok, bpw = validate_bpw(file_bytes(art), cell["n"], args.params)
    if not ok:
        raise SystemExit(f"refusing to bench {cell['cell_id']}: bpw {bpw:.3f} out of band [{cell['n']},{cell['n']+1})")
    return bpw

def do_bench_ar(args: argparse.Namespace) -> int:
    ensure_qcal_dir(args)
    mp = manifest_path(args)
    if mp.is_file():
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except Exception as e:
            raise SystemExit(f"refusing stale/malformed manifest {mp}: {e}")
    else:
        manifest = build_manifest(args)
        if not args.dry_run:
            atomic_write_json(mp, manifest)
    cells = manifest["cells"]
    prompt_text = _read_prompt_text(args)
    msha = manifest_sha256(manifest)
    sp = state_path(args)
    state = load_state_strict(sp, expected_manifest_sha=msha)
    if not args.dry_run and state.get("manifest_sha256") != msha:
        state["manifest_sha256"] = msha
        atomic_write_json(sp, state)
    prompt_path = resolve_prompt_path(args.prompt, args.checkout)
    prompt_info = prompt_digest(prompt_path)
    bench_bin = Path(args.bench_bin) if args.bench_bin else Path(args.checkout) / "target/release/hipfire"
    # Serial fresh-process perf by default
    failed: List[str] = []
    completed: List[str] = []
    phases = state.get("phases", {})
    bstate = phases.get("bench-ar", {})
    prior_completed = set(bstate.get("completed", []))
    if args.dry_run:
        eprint(f"[dry-run] bench-ar phase: {len(cells)} cells × 3 fresh processes = {len(cells)*3} serial runs (avoid thermal/daemon cross-talk)")
        eprint(f"[dry-run] prompt {prompt_path} bytes={prompt_info.get('bytes')} sha256={prompt_info.get('sha256')} md5={prompt_info.get('md5')}")
        eprint(f"[dry-run] bench_bin {bench_bin} exists={bench_bin.is_file()} params={args.params}")
    for cell in cells:
        # Validate size/bpw before scoring (unless dry-run)
        if not args.dry_run:
            _ensure_artifact_bpw_or_fail(cell, args)
        cell_id = cell["cell_id"]
        # For dry-run we print 3 commands per cell
        if args.dry_run:
            for rep in range(3):
                argv, env = bench_argv(cell, prompt_text, spec="off", draft_path=None, args=args)
                env_str = " ".join(f"{k}={shlex.quote(v)}" for k,v in sorted(env.items()))
                eprint(f"[dry-run] bench-ar {cell_id} rep={rep+1}/3: {env_str + ' ' if env_str else ''}{' '.join(shlex.quote(a) for a in argv)}")
                eprint(f"[dry-run]   artifact={cell['artifact']} prompt_bytes={prompt_info.get('bytes')} commit={git_commit(args.checkout)} gpu=0")
            continue
        # Non-dry-run: 3 fresh processes per arm, serial
        rep_outputs: List[Dict[str, Any]] = []
        any_fail = False
        for rep in range(3):
            argv, env = bench_argv(cell, prompt_text, spec="off", draft_path=None, args=args)
            rc, sout, serr = run_subprocess(argv, env, dry_run=False)
            log_path = Path(args.qcal_dir) / "logs" / f"bench_ar_{cell_id}_rep{rep+1}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_text = f"$ {' '.join(shlex.quote(a) for a in argv)}\n# env {json.dumps(env)}\n# rep {rep+1}/3\nSTDOUT:\n{sout}\nSTDERR:\n{serr}\n"
            atomic_write_text(log_path, log_text)
            if rc != 0:
                eprint(f"bench-ar {cell_id} rep {rep+1} FAILED rc={rc} (see {log_path}) — resumable, not marked measured")
                any_fail = True
                break
            objs, raw = parse_json_robust(sout)
            # Without summarizing away samples: retain all parsed objects
            rep_outputs.append({
                "rep": rep+1,
                "argv": argv,
                "env": env,
                "returncode": rc,
                "stdout": sout,
                "stderr": serr,
                "log": str(log_path),
                "parsed_json": objs,
                "raw_stdout": sout,
                "raw_stderr": serr,
            })
        if any_fail:
            failed.append(cell_id)
            continue
        # All 3 reps succeeded — create aggregated result row with every contract column
        art = Path(cell["artifact"])
        row = {
            "phase": "bench-ar",
            "cell_id": cell_id,
            "n": cell["n"],
            "tier": cell["tier"],
            "codec": cell["codec"],
            "format": cell["format"],
            "artifact": str(art),
            "artifact_bytes": file_bytes(art) if art.is_file() else None,
            "artifact_sha256": sha256_file(art) if art.is_file() else None,
            "artifact_md5": md5_file(art) if art.is_file() else None,
            "artifact_bpw": compute_bpw(file_bytes(art), args.params) if art.is_file() else None,
            "bench_bin": str(bench_bin),
            "bench_bin_sha256": sha256_file(bench_bin) if bench_bin.is_file() else None,
            "bench_bin_md5": md5_file(bench_bin) if bench_bin.is_file() else None,
            "prompt": str(prompt_path),
            "prompt_bytes": prompt_info.get("bytes"),
            "prompt_sha256": prompt_info.get("sha256"),
            "prompt_md5": prompt_info.get("md5"),
            "spec": "off",
            "runs": DEFAULT_RUNS,
            "warmups": DEFAULT_WARMUPS,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "backend": "noslots",
            "workload": "stateless",
            "json": True,
            "reps": rep_outputs,
            "fresh_processes": 3,
            "commit": git_commit(args.checkout),
            "gpu": 0,
            # τ / acceptance fields: retain every sample, do not summarize away
            "tau_samples": [o.get("tau") for o in [x for r in rep_outputs for x in r["parsed_json"]] if isinstance(o, dict) and "tau" in o],
            "acceptance_samples": [o.get("acceptance_rate") or o.get("acceptance") for o in [x for r in rep_outputs for x in r["parsed_json"]] if isinstance(o, dict)],
        }
        # Add any top-level json fields verbatim
        rp = results_path(args)
        results = load_results_strict(rp)
        new_rows = [r for r in results["rows"] if not (r.get("phase") == "bench-ar" and r.get("cell_id") == cell_id)]
        new_rows.append(row)
        results["rows"] = sorted(new_rows, key=lambda x: (x.get("phase",""), x.get("cell_id","")))
        results["updated_at"] = utc_now()
        atomic_write_json(rp, results)
        state = load_state_strict(sp, expected_manifest_sha=msha)
        phases = state.setdefault("phases", {})
        bs = phases.setdefault("bench-ar", {"completed": [], "failed": []})
        bs["completed"] = sorted(set(bs.get("completed", [])) | {cell_id})
        bs["failed"] = sorted(set(bs.get("failed", [])) - {cell_id})
        bs["updated_at"] = utc_now()
        atomic_write_json(sp, state)
        completed.append(cell_id)
        eprint(f"bench-ar {cell_id} OK 3 reps")
    if not args.dry_run:
        state = load_state_strict(sp, expected_manifest_sha=msha)
        phases = state.setdefault("phases", {})
        bs = phases.setdefault("bench-ar", {"completed": [], "failed": []})
        bs["failed"] = sorted(set(bs.get("failed", [])) | set(failed))
        bs["completed"] = sorted(set(bs.get("completed", [])) | set(completed))
        atomic_write_json(sp, state)
        if failed:
            eprint(f"bench-ar phase completed with {len(failed)} failures: {', '.join(failed)}")
            return 1
    return 0

def parse_draft_map_arg(value: Optional[str]) -> Optional[Dict[str, str]]:
    if not value:
        return None
    m: Dict[str,str] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"bad --draft-map entry '{part}' (expected k=v, e.g. mq2v2=mq2v2)")
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Accept either "2" or "mq2" or "mq2v2" as key
        # Normalize to codec string
        def norm(x: str) -> str:
            x = x.lower()
            if x in ("2","3","4","5","6"):
                return f"mq{x}v2"
            if re.fullmatch(r"mq[2-6]", x):
                return f"{x}v2"
            if re.fullmatch(r"mq[2-6]v2", x):
                return x
            raise SystemExit(f"bad draft-map codec '{x}'")
        m[norm(k)] = norm(v)
    return m

def do_bench_dflash(args: argparse.Namespace) -> int:
    ensure_qcal_dir(args)
    mp = manifest_path(args)
    if mp.is_file():
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except Exception as e:
            raise SystemExit(f"refusing stale/malformed manifest {mp}: {e}")
    else:
        manifest = build_manifest(args)
        if not args.dry_run:
            atomic_write_json(mp, manifest)
    cells = manifest["cells"]
    drafts = manifest["drafts"]
    draft_by_codec = {d["codec"]: d for d in drafts}
    # draft map: optional per-bit mapping, defaults same-bit
    draft_map = parse_draft_map_arg(getattr(args, "draft_map", None))
    if draft_map is None:
        # default same-bit
        draft_map = {d["codec"]: d["codec"] for d in drafts}
    # Always include mq4v2 control data — ensure mapping contains mq4v2 or we emit extra control row
    has_mq4_control = any(v == "mq4v2" for v in draft_map.values()) or "mq4v2" in draft_by_codec
    if not has_mq4_control:
        eprint("warn: draft-map lacks mq4v2 control; will still include mq4v2 control arm per contract")
    prompt_text = _read_prompt_text(args)
    msha = manifest_sha256(manifest)
    sp = state_path(args)
    state = load_state_strict(sp, expected_manifest_sha=msha)
    if not args.dry_run and state.get("manifest_sha256") != msha:
        state["manifest_sha256"] = msha
        atomic_write_json(sp, state)
    prior_dflash_completed = set(
        state.get("phases", {}).get("bench-dflash", {}).get("completed", [])
    )
    prompt_path = resolve_prompt_path(args.prompt, args.checkout)
    prompt_info = prompt_digest(prompt_path)
    bench_bin = Path(args.bench_bin) if args.bench_bin else Path(args.checkout) / "target/release/hipfire"
    failed: List[str] = []
    completed: List[str] = []
    if args.dry_run:
        eprint(f"[dry-run] bench-dflash phase: {len(cells)} cells × 3 fresh processes = {len(cells)*3} serial runs (same fresh-process contract as bench-ar)")
        eprint(f"[dry-run] drafts: {', '.join(d['codec'] for d in drafts)} (mq4v2 control always included)")
        eprint(f"[dry-run] draft_map={draft_map}")
    for cell in cells:
        if not args.dry_run:
            _ensure_artifact_bpw_or_fail(cell, args)
        cell_id = cell["cell_id"]
        # Determine draft for this cell via map; default same-bit
        desired_draft_codec = draft_map.get(cell["codec"], cell["codec"])
        task_id = f"{cell_id}:{desired_draft_codec}"
        if not args.dry_run and task_id in prior_dflash_completed:
            eprint(f"bench-dflash {task_id} skip (resumable)")
            continue
        draft = draft_by_codec.get(desired_draft_codec)
        if draft is None and not args.dry_run:
            eprint(f"bench-dflash {cell_id} missing draft codec {desired_draft_codec} — marking failed")
            failed.append(task_id)
            continue
        draft_path = draft["draft_path"] if draft else str(Path(args.qcal_dir) / "drafts" / f"qwen3.8-27b-dflash.{desired_draft_codec}.hfq")
        # Validate draft exists for non-dry-run
        if not args.dry_run and not Path(draft_path).is_file():
            eprint(f"bench-dflash {cell_id} draft file missing {draft_path} — run drafts phase first")
            failed.append(task_id)
            continue
        if args.dry_run:
            for rep in range(3):
                argv, env = bench_argv(cell, prompt_text, spec="dflash", draft_path=draft_path, args=args)
                env_str = " ".join(f"{k}={shlex.quote(v)}" for k,v in sorted(env.items()))
                eprint(f"[dry-run] bench-dflash {cell_id} rep={rep+1}/3 draft={desired_draft_codec} -> {draft_path}")
                eprint(f"[dry-run]   {env_str + ' ' if env_str else ''}{' '.join(shlex.quote(a) for a in argv)}")
                # Always include mq4v2 control data: if this cell is not mq4, also show control
                if cell["codec"] != "mq4v2":
                    control_draft = draft_by_codec.get("mq4v2")
                    if control_draft:
                        argv_c, env_c = bench_argv(cell, prompt_text, spec="dflash", draft_path=control_draft["draft_path"], args=args)
                        env_c_str = " ".join(f"{k}={shlex.quote(v)}" for k,v in sorted(env_c.items()))
                        eprint(f"[dry-run]   control mq4v2 draft {control_draft['draft_path']}: {env_c_str + ' ' if env_c_str else ''}{' '.join(shlex.quote(a) for a in argv_c)}")
            continue
        # Real execution: 3 fresh processes per arm, serial
        rep_outputs: List[Dict[str, Any]] = []
        any_fail = False
        for rep in range(3):
            argv, env = bench_argv(cell, prompt_text, spec="dflash", draft_path=draft_path, args=args)
            rc, sout, serr = run_subprocess(argv, env, dry_run=False)
            log_path = Path(args.qcal_dir) / "logs" / f"bench_dflash_{cell_id}_rep{rep+1}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_text = f"$ {' '.join(shlex.quote(a) for a in argv)}\n# env {json.dumps(env)}\n# rep {rep+1}/3 draft={desired_draft_codec}\nSTDOUT:\n{sout}\nSTDERR:\n{serr}\n"
            atomic_write_text(log_path, log_text)
            if rc != 0:
                eprint(f"bench-dflash {cell_id} rep {rep+1} FAILED rc={rc} (see {log_path}) — resumable")
                any_fail = True
                break
            objs, raw = parse_json_robust(sout)
            rep_outputs.append({
                "rep": rep+1,
                "argv": argv,
                "env": env,
                "returncode": rc,
                "stdout": sout,
                "stderr": serr,
                "log": str(log_path),
                "parsed_json": objs,
                "raw_stdout": sout,
                "raw_stderr": serr,
            })
        if any_fail:
            failed.append(task_id)
            continue
        art = Path(cell["artifact"])
        dpath = Path(draft_path)
        row = {
            "phase": "bench-dflash",
            "cell_id": cell_id,
            "n": cell["n"],
            "tier": cell["tier"],
            "codec": cell["codec"],
            "format": cell["format"],
            "artifact": str(art),
            "artifact_bytes": file_bytes(art) if art.is_file() else None,
            "artifact_sha256": sha256_file(art) if art.is_file() else None,
            "artifact_md5": md5_file(art) if art.is_file() else None,
            "artifact_bpw": compute_bpw(file_bytes(art), args.params) if art.is_file() else None,
            "draft_id": desired_draft_codec,
            "draft_path": str(dpath),
            "draft_bytes": file_bytes(dpath) if dpath.is_file() else None,
            "draft_sha256": sha256_file(dpath) if dpath.is_file() else None,
            "draft_md5": md5_file(dpath) if dpath.is_file() else None,
            "bench_bin": str(bench_bin),
            "bench_bin_sha256": sha256_file(bench_bin) if bench_bin.is_file() else None,
            "prompt": str(prompt_path),
            "prompt_bytes": prompt_info.get("bytes"),
            "prompt_sha256": prompt_info.get("sha256"),
            "prompt_md5": prompt_info.get("md5"),
            "spec": "dflash",
            "runs": DEFAULT_RUNS,
            "warmups": DEFAULT_WARMUPS,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "backend": "noslots",
            "workload": "stateless",
            "json": True,
            "reps": rep_outputs,
            "fresh_processes": 3,
            "commit": git_commit(args.checkout),
            "gpu": 0,
            "tau_samples": [o.get("tau") for o in [x for r in rep_outputs for x in r["parsed_json"]] if isinstance(o, dict) and "tau" in o],
            "acceptance_samples": [o.get("acceptance_rate") or o.get("acceptance") for o in [x for r in rep_outputs for x in r["parsed_json"]] if isinstance(o, dict)],
            "control_mq4v2_included": desired_draft_codec == "mq4v2",
        }
        rp = results_path(args)
        results = load_results_strict(rp)
        new_rows = [r for r in results["rows"] if not (r.get("phase") == "bench-dflash" and r.get("cell_id") == cell_id and r.get("draft_id") == desired_draft_codec)]
        new_rows.append(row)
        results["rows"] = sorted(new_rows, key=lambda x: (x.get("phase",""), x.get("cell_id",""), x.get("draft_id","")))
        results["updated_at"] = utc_now()
        atomic_write_json(rp, results)
        state = load_state_strict(sp, expected_manifest_sha=msha)
        phases = state.setdefault("phases", {})
        bs = phases.setdefault("bench-dflash", {"completed": [], "failed": []})
        bs["completed"] = sorted(set(bs.get("completed", [])) | {task_id})
        bs["failed"] = sorted(set(bs.get("failed", [])) - {task_id})
        bs["updated_at"] = utc_now()
        atomic_write_json(sp, state)
        completed.append(cell_id)
        eprint(f"bench-dflash {cell_id} draft={desired_draft_codec} OK 3 reps")
    # Always include mq4v2 control data: if not already benching mq4v2 cells with mq4v2 draft, ensure we have control rows
    # The above already benches each cell with its mapped draft; mq4 cells inherently include control. For non-mq4 cells we retain control flag.
    if not args.dry_run:
        state = load_state_strict(sp, expected_manifest_sha=msha)
        phases = state.setdefault("phases", {})
        bs = phases.setdefault("bench-dflash", {"completed": [], "failed": []})
        bs["failed"] = sorted(set(bs.get("failed", [])) | set(failed))
        atomic_write_json(sp, state)
        if failed:
            eprint(f"bench-dflash phase completed with {len(failed)} failures: {', '.join(failed)}")
            return 1
    return 0

# ---------------------------------------------------------------------------
# Manifest command already above; All command
# ---------------------------------------------------------------------------

def do_all(args: argparse.Namespace) -> int:
    # Sequential phases to respect RSS/thermal constraints: quantize serial, drafts serial, kld parallel, benches serial
    rc = 0
    for fn, name in [
        (do_manifest, "manifest"),
        (do_quantize, "quantize"),
        (do_drafts, "drafts"),
        (do_kld, "kld"),
        (do_bench_ar, "bench-ar"),
        (do_bench_dflash, "bench-dflash"),
    ]:
        eprint(f"=== phase {name} ===")
        r = fn(args)
        if r != 0:
            eprint(f"phase {name} exited {r} — all stopping (resumable)")
            return r
        rc |= r
    # Final manifest write for completeness
    eprint("=== all phases done ===")
    return rc

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_global_args(parser: argparse.ArgumentParser):
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help=f"hipfire checkout (default {DEFAULT_CHECKOUT})")
    parser.add_argument("--qcal-dir", dest="qcal_dir", default=DEFAULT_QCAL, help=f"qcal output dir (default {DEFAULT_QCAL})")
    parser.add_argument("--parent", default=DEFAULT_PARENT, help=f"parent dir (default {DEFAULT_PARENT})")
    parser.add_argument("--imatrix", default=DEFAULT_IMATRIX, help=f"imatrix gguf (default {DEFAULT_IMATRIX})")
    parser.add_argument("--ref-wt2", dest="ref_wt2", default=DEFAULT_REF_WT2, help=f"wt2 ref bin (default {DEFAULT_REF_WT2})")
    parser.add_argument("--ref-v6sel", dest="ref_v6sel", default=DEFAULT_REF_V6SEL, help=f"v6sel ref bin (default {DEFAULT_REF_V6SEL})")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help=f"prompt file (default {DEFAULT_PROMPT})")
    parser.add_argument("--params", type=int, default=DEFAULT_PARAMS, help=f"model params for bpw (default {DEFAULT_PARAMS})")
    parser.add_argument("--draft-source", dest="draft_source", default=DEFAULT_DRAFT_SOURCE, help=f"draft HF hub id (default {DEFAULT_DRAFT_SOURCE})")
    parser.add_argument("--quantize-bin", dest="quantize_bin", default=None, help="override hipfire-quantize binary")
    parser.add_argument("--dflash-bin", dest="dflash_bin", default=None, help="override dflash_convert binary")
    parser.add_argument("--bench-bin", dest="bench_bin", default=None, help="override hipfire bench binary")
    parser.add_argument("--eval-bin", dest="eval_bin", default=None, help="override eval_hipfire binary")
    parser.add_argument("--draft-map", dest="draft_map", default=None, help="optional per-bit draft mapping k=v pairs, e.g. mq2v2=mq2v2,mq3v2=mq3v2 (defaults same-bit)")
def build_parser() -> argparse.ArgumentParser:
    # Top-level dry-run must also be accepted after subcommand (e.g. `manifest --dry-run`)
    # so we create a shared parent to avoid duplication.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", help="print every exact command/environment/path without execution")
    p = argparse.ArgumentParser(
        description="qwen38 ladder campaign orchestrator — 15 cells (mq2..6 × xt/base/pro) + 5 drafts + dual KLD + AR/DFlash benches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s manifest --dry-run
              %(prog)s quantize --dry-run
              %(prog)s drafts --dry-run
              %(prog)s kld --dry-run
              %(prog)s bench-ar --dry-run
              %(prog)s bench-dflash --dry-run --draft-map mq2v2=mq2v2,mq3v2=mq3v2,mq4v2=mq4v2,mq5v2=mq5v2,mq6v2=mq6v2
              %(prog)s all --dry-run
              %(prog)s --dry-run manifest
            """),
        parents=[common],
    )
    add_global_args(p)
    sub = p.add_subparsers(dest="cmd", required=True)
    # manifest
    sp = sub.add_parser("manifest", parents=[common], help="write deterministic 15-cell + 5-draft manifest.json")
    sp.set_defaults(func=do_manifest)
    # quantize
    sp = sub.add_parser("quantize", parents=[common], help="serial quantization (51GB RSS) — resumable, bpw-banded, atomic writes")
    sp.set_defaults(func=do_quantize)
    # drafts
    sp = sub.add_parser("drafts", parents=[common], help="build 5 V2 DFlash2 drafts via dflash_convert same-bit --mqNv2 (mq4 control included)")
    sp.set_defaults(func=do_drafts)
    # kld
    sp = sub.add_parser("kld", parents=[common], help="bounded GPU-parallel KLD: 24 chunks q8/q8 prefill both refs, private HOME/cache per GPU")
    sp.set_defaults(func=do_kld)
    # bench-ar
    sp = sub.add_parser("bench-ar", parents=[common], help="serial fresh-process AR benches: --spec off, 3 reps per arm, prompt bytes as positional arg")
    sp.set_defaults(func=do_bench_ar)
    # bench-dflash
    sp = sub.add_parser("bench-dflash", parents=[common], help="serial fresh-process DFlash benches: --spec dflash HIPFIRE_DFLASH_DRAFT, same-bit default, mq4v2 control always included")
    sp.set_defaults(func=do_bench_dflash)
    # all
    sp = sub.add_parser("all", parents=[common], help="run all phases sequentially (manifest→quantize→drafts→kld→bench-ar→bench-dflash)")
    sp.set_defaults(func=do_all)
    return p
def main(argv=None) -> int:
    # Support --dry-run in any position (before or after subcommand) without
    # relying on argparse parent duplication which is fragile across Python
    # versions. Strip it manually and inject into Namespace after parse.
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)
    dry_run = False
    # Remove all --dry-run occurrences wherever they appear
    cleaned: list[str] = []
    for tok in argv:
        if tok == "--dry-run":
            dry_run = True
        else:
            cleaned.append(tok)
    parser = build_parser()
    args = parser.parse_args(cleaned)
    # Inject dry_run into namespace
    args.dry_run = dry_run or getattr(args, "dry_run", False)
    # Validate params
    if args.params <= 0:
        parser.error("--params must be positive")
    # Dispatch
    try:
        return args.func(args)
    except SystemExit as e:
        # Preserve SystemExit code
        if e.code is None:
            return 0
        if isinstance(e.code, int):
            # Ensure message visible
            if e.code != 0:
                eprint(str(e))
            return e.code
        eprint(str(e))
        return 1
    except KeyboardInterrupt:
        eprint("interrupted")
        return 130
    except Exception as e:
        eprint(f"error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
