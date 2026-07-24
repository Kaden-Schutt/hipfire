#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Harvest the redline A/B capture corpus across the fleet.

Companion to ``harvest_ledgers.py``, deliberately separate: redline measurement
answers a different question (dispatch-stream parity and per-launch replay, not
lever-vs-baseline perf) and has its own identity model (``sequence_hash`` over
the whole dispatch stream, not a per-experiment measurement hash).

**Discovery is content-driven, not path-driven.** Every path/filename-pattern
sweep written against this fleet missed this corpus entirely: the files are
plain ``.json`` in directories named after ad-hoc experiments
(``gfx1100-conv-qknor``, ``on-exact``, ``gfx1151-cumode-probe``) matching no
convention. Files are classified by their JSON key signature instead. If you
extend this script, keep it that way.

What it collects, by shape:

  capture     decode[].captures[].redline_capture -- per-launch dispatch trace
              {kernel, artifact(.hsaco), grid, block, shared_mem,
               kernarg_bytes, kernarg_hex, kernarg_hash} + sequence_hash
  product     daemon_sha256 / git_commit / dpm_warmup_secs provenance runs
  speedup     A/B arm pairs carrying an explicit `speedup`
  inspection  hipcc compiler probes (arch, hipcc_version, output_sha256)
  bench       cert measurement rows (backend, correctness, classification, isa)
  aql         aql_shadow (PM4-vs-HIP parity verdict: backend, dispatches,
              command_dwords, bit_exact/logits/kv/recurrent/oracle flags) and
              aql_contract_probe (per-kernel captured-vs-loader kernarg bytes,
              alignment, static/dynamic LDS)

Outputs (git-tracked JSONL + regenerable index):

  redline/corpus/captures.jsonl       one row per capture (seq_hash keyed)
  redline/corpus/launch_shapes.jsonl  DISTINCT launch shapes + occurrence counts
  redline/corpus/bench.jsonl          bench/cert measurement rows
  redline/corpus/aql_contracts.jsonl  per-kernel AQL kernarg/LDS ABI contracts
  redline/corpus/manifest.json        harvest metadata (timestamps live here)
  redline/db/redline.db               SQLite index (gitignored, regenerable)

The 1.1M raw launches collapse to a few thousand distinct shapes; the corpus
keeps the distinct shapes with counts, not the repetitions.

Usage:
    scripts/harvest_redline.py --dry-run
    scripts/harvest_redline.py --ingest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(REPO, "redline", "corpus")
DB_PATH = os.path.join(REPO, "redline", "db", "redline.db")

DEFAULT_BOXES = ["k9lin", "hipx", "hiptrx"]
LOCAL_BOX = "k9lin"
LOCAL_ROOTS = ["/home/kaden/ClaudeCode/autorocm", "/tmp"]

# ── remote extractor ─────────────────────────────────────────────────────────
# Streams candidate JSON as envelopes. Classification happens locally so the
# remote side stays trivial and dependency-free.
REMOTE = r'''
import json, os, re, sys

roots = sys.argv[1:] or [os.path.expanduser("~"), "/tmp"]
seen, R = set(), []
for r in roots:
    rp = os.path.realpath(r)
    if rp not in seen and os.path.isdir(rp):
        seen.add(rp); R.append(rp)

EXCL = re.compile(
    r"/(\.git|node_modules|\.cargo|site-packages|__pycache__|\.rustup|\.bun|\.npm"
    r"|llvm|mesa|rocm-systems|googletest|_deps)/"
    r"|/target/(debug|release)/(deps|build|incremental)/|/\.fingerprint/"
    r"|/pytest-of-|/niah/|/prompts/"
)

# Key signatures that mark a redline artifact. Content-based on purpose.
MARK = (b'"redline_capture"', b'"claim_gate"', b'"daemon_sha256"',
        b'"aql_contract_probe"', b'"hipcc_version"', b'"speedup"',
        b'"environment_ref"', b'"timing_mode"')

out = sys.stdout
for root in R:
    for dp, dn, fn in os.walk(root, onerror=lambda e: None):
        if EXCL.search(dp + "/"):
            dn[:] = []
            continue
        for f in fn:
            if not f.endswith(".json"):
                continue
            p = os.path.join(dp, f)
            try:
                sz = os.path.getsize(p)
                if sz > 80_000_000:
                    continue
                with open(p, "rb") as fh:
                    head = fh.read(4096)
                    if not any(m in head for m in MARK):
                        fh.seek(0)
                        if not any(m in fh.read() for m in MARK):
                            continue
                with open(p, errors="ignore") as fh:
                    body = fh.read()
                out.write(json.dumps({"p": p, "m": int(os.path.getmtime(p)), "b": body}) + "\n")
            except OSError:
                continue
'''


def _sha16(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


_ARCHES = ("gfx1010", "gfx1030", "gfx1100", "gfx1101", "gfx1102",
           "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200", "gfx1201")


def _arch_from_artifacts(seq) -> str | None:
    """Arch from a capture's kernel-cache artifact paths.

    This is the reliable source: capture docs on some boxes carry no arch field
    and sit under paths with no gfx component, but every launch names its
    artifact as `.../kernel-cache/gfxNNNN/<kernel>.hsaco`.
    """
    for s in seq or []:
        art = s.get("artifact") if isinstance(s, dict) else None
        if not art:
            continue
        for a in _ARCHES:
            if f"/{a}/" in art or art.endswith(f"_{a}.hsaco"):
                return a
    return None


def _arch_of(path: str, doc: dict) -> str | None:
    for a in _ARCHES:
        if a in path:
            return a
    hw = doc.get("hardware")
    if isinstance(hw, dict):
        for k in ("arch", "gfx", "target", "gcn_arch_name"):
            v = hw.get(k)
            if isinstance(v, str):
                for a in ("gfx1010", "gfx1030", "gfx1100", "gfx1151", "gfx1201"):
                    if a in v:
                        return a
    for k in ("arch", "target"):
        v = doc.get(k)
        if isinstance(v, str) and v.startswith("gfx"):
            return v
    return None


def _walk_captures(obj):
    """Yield dicts containing a redline_capture, from dict- or list-rooted docs."""
    if isinstance(obj, list):
        for x in obj:
            yield from _walk_captures(x)
        return
    if not isinstance(obj, dict):
        return
    if "redline_capture" in obj:
        yield obj
    for v in obj.values():
        if isinstance(v, (dict, list)):
            yield from _walk_captures(v)


def classify(doc: dict) -> str:
    """Shape of a redline artifact, by key signature."""
    if not isinstance(doc, dict):
        return "other"
    k = set(doc)
    if "decode" in k or "redline_capture" in json.dumps(doc)[:400]:
        return "capture"
    if "claim_gate" in k or "comparison_groups" in k:
        return "claim"
    if "hipcc_version" in k or "inspection" in k:
        return "inspection"
    if "speedup" in k:
        return "speedup"
    if "daemon_sha256" in k or "dpm_warmup_secs" in k:
        return "product"
    if "measurements" in k or "timing_mode" in k or "environment_ref" in k:
        return "bench"
    return "other"


_SHADOW_FLAGS = ("bit_exact", "pre_bit_exact", "pre_logits_equal", "pre_kv_equal",
                 "pre_recurrent_equal", "hip_oracles_equal", "blob_bit_exact",
                 "logits_equal", "force_blob_path")


def parse_aql(doc: dict, path: str, box: str, mtime: int, arch):
    """→ (shadow summary dict, per-kernel AQL contract rows).

    `aql_shadow` is the PM4-vs-HIP parity verdict and `aql_contract_probe` is
    the per-kernel ABI check -- the two things CLAUDE.md requires as redline
    acceptance evidence. An earlier version of this script stored both as a
    bare bool, discarding the parity flags and every kernarg contract.
    """
    sh = doc.get("aql_shadow")
    summary = {}
    if isinstance(sh, dict):
        summary = {
            "shadow_backend": sh.get("backend"),
            "shadow_dispatches": sh.get("dispatches"),
            "shadow_packets": sh.get("packets"),
            "shadow_command_dwords": sh.get("command_dwords"),
            "shadow_queue_id": sh.get("queue_id"),
            "shadow_context_tokens": sh.get("context_tokens"),
            "shadow_iterations": sh.get("iterations"),
            "shadow_frame_checkpoint": sh.get("frame_checkpoint"),
        }
        for f in _SHADOW_FLAGS:
            if f in sh:
                summary["shadow_" + f] = bool(sh[f])

    probe = doc.get("aql_contract_probe")
    contracts = []
    if isinstance(probe, dict):
        for c in probe.get("contracts", []):
            if not isinstance(c, dict):
                continue
            cap_b, ld_b = c.get("captured_kernarg_bytes"), c.get("loader_kernarg_bytes")
            contracts.append({
                "arch": arch,
                "kernel": c.get("kernel"),
                "captured_kernarg_bytes": cap_b,
                "loader_kernarg_bytes": ld_b,
                "loader_kernarg_alignment": c.get("loader_kernarg_alignment"),
                "static_group_bytes": c.get("static_group_bytes"),
                "dynamic_group_bytes": c.get("dynamic_group_bytes"),
                # Derived, not persisted upstream. NOTE: this is a DISCREPANCY
                # INDICATOR, not a defect count. It fires on 54/66 contracts on
                # one box alone, and the direction (captured 32 vs loader 288)
                # says the two fields most likely measure different things --
                # bytes the packet actually used vs bytes the loader declared or
                # allocated. Treat a hit as "worth a look", never as a bug, until
                # the loader-side semantics are confirmed. `captured > loader` is
                # the genuinely alarming direction and is flagged separately.
                "kernarg_differs": (cap_b is not None and ld_b is not None
                                    and cap_b != ld_b),
                "kernarg_captured_exceeds_loader": (
                    cap_b is not None and ld_b is not None and cap_b > ld_b),
                "_prov": {"box": box, "path": path, "mtime": mtime},
            })
    return summary, contracts


def parse_capture(doc: dict, path: str, box: str, mtime: int):
    """→ (capture rows, launch-shape rows). Launch shapes are deduped by caller."""
    caps, shapes = [], []
    model = os.path.basename(str(doc.get("model", "") or ""))
    arch = _arch_of(path, doc)
    kv = doc.get("kv_mode")
    daemon = doc.get("daemon")
    _aql_summary, _ = parse_aql(doc, path, box, mtime, arch)
    contracts = []
    for holder in _walk_captures(doc):
        rc = holder.get("redline_capture") or {}
        if not rc:
            continue
        seq = rc.get("sequence_hash")
        seq_list = rc.get("sequence") or []
        arch = arch or _arch_from_artifacts(seq_list)
        if not contracts:
            _s2, contracts = parse_aql(doc, path, box, mtime, arch)
        caps.append({
            "sequence_hash": seq,
            "arch": arch,
            "model": model,
            "kv_mode": kv,
            "launches": rc.get("launches"),
            "unique_kernels": rc.get("unique_kernels"),
            "tok_s": _num(holder.get("tok_s")),
            "us_per_token": _num(holder.get("us_per_token")),
            "ms": _num(holder.get("ms")),
            "context_tokens": holder.get("context_tokens"),
            "iterations": holder.get("iterations"),
            "daemon": daemon,
            "aql_contract_probe": bool(doc.get("aql_contract_probe")),
            "aql_shadow": bool(doc.get("aql_shadow")),
            **_aql_summary,
            "arm": os.path.splitext(os.path.basename(path))[0],
            "experiment": os.path.basename(os.path.dirname(path)),
            "_prov": {"box": box, "path": path, "mtime": mtime},
        })
        for i, s in enumerate(seq_list):
            if not isinstance(s, dict):
                continue
            art = s.get("artifact")
            shapes.append({
                "arch": arch,
                "kernel": s.get("kernel"),
                "artifact": os.path.basename(art) if art else None,
                "artifact_path": art,
                "grid": s.get("grid"),
                "block": s.get("block"),
                "shared_mem": s.get("shared_mem"),
                "kernarg_bytes": s.get("kernarg_bytes"),
                "kernarg_hash": s.get("kernarg_hash"),
                "position": i,
                "sequence_hash": seq,
                "_prov": {"box": box, "path": path, "mtime": mtime},
            })
    return caps, shapes, contracts


def _scalar(v, limit: int = 400):
    """SQLite-bindable scalar. Redline docs are not shape-stable across runs --
    `classification`, `correctness` and `status` appear as str, dict AND list
    depending on the emitter, so anything non-scalar is JSON-flattened."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return json.dumps(v, default=str)[:limit]


def parse_bench(doc: dict, path: str, box: str, mtime: int, kind: str):
    corr = doc.get("correctness")
    if isinstance(corr, dict):
        corr = corr.get("status") or corr.get("ok") or json.dumps(corr)[:80]
    corr = _scalar(corr)
    row = {
        "kind": kind,
        "backend": _scalar(doc.get("backend")),
        "bench": _scalar(doc.get("bench")),
        "family": _scalar(doc.get("family")),
        "arch": _arch_of(path, doc),
        "timing_mode": _scalar(doc.get("timing_mode")),
        "classification": _scalar(doc.get("classification")),
        "correctness": corr,
        "status": _scalar(doc.get("status")),
        "speedup": _num(doc.get("speedup")),
        "claim_gate": json.dumps(doc["claim_gate"])[:400] if doc.get("claim_gate") else None,
        "hipcc_version": _scalar(doc.get("hipcc_version")),
        "output_sha256": _scalar(doc.get("output_sha256")),
        "daemon_sha256": _scalar(doc.get("daemon_sha256")),
        "git_commit": _scalar(doc.get("git_commit")),
        "dpm_warmup_secs": doc.get("dpm_warmup_secs"),
        "run_tag": _scalar(doc.get("run_tag")),
        "experiment": os.path.basename(os.path.dirname(path)),
        "arm": os.path.splitext(os.path.basename(path))[0],
        "_prov": {"box": box, "path": path, "mtime": mtime},
    }
    n = doc.get("measurements") or doc.get("runs") or doc.get("rows")
    if isinstance(n, list):
        row["n_measurements"] = len(n)
    return row


def fetch(box, roots, timeout):
    cmd = ([sys.executable, "-", *roots] if box == LOCAL_BOX
           else ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", box,
                 "python3 -", *roots])
    try:
        p = subprocess.run(cmd, input=REMOTE, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return [], f"timeout after {timeout}s"
    except OSError as e:
        return [], str(e)
    if p.returncode != 0:
        return [], (p.stderr.strip().splitlines() or [f"exit {p.returncode}"])[-1]
    envs, bad = [], 0
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        try:
            envs.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"  [{box}] {bad} unreadable envelopes", file=sys.stderr)
    return envs, None


def dedup_shapes(shapes):
    """Collapse the ~1.1M raw launches to distinct shapes with counts.

    Identity is (arch, kernel, artifact, grid, block, shared_mem, kernarg_bytes)
    -- deliberately NOT kernarg_hash or position, which vary per token and would
    defeat the collapse. Position/kernarg variety is summarized instead.
    """
    agg = {}
    for s in shapes:
        k = _sha16(s["arch"], s["kernel"], s["artifact"],
                   s["grid"], s["block"], s["shared_mem"], s["kernarg_bytes"])
        e = agg.get(k)
        if e is None:
            e = dict(s)
            e["shape_key"] = k
            e["occurrences"] = 0
            e["kernarg_hashes"] = set()
            e["sequence_hashes"] = set()
            e["positions"] = []
            e.pop("position", None)
            e.pop("kernarg_hash", None)
            agg[k] = e
        e["occurrences"] += 1
        if s.get("kernarg_hash"):
            e["kernarg_hashes"].add(s["kernarg_hash"])
        if s.get("sequence_hash"):
            e["sequence_hashes"].add(s["sequence_hash"])
        if len(e["positions"]) < 8:
            e["positions"].append(s.get("position"))
        if s["_prov"]["mtime"] > e["_prov"]["mtime"]:
            e["_prov"] = s["_prov"]
    out = []
    for e in agg.values():
        e["distinct_kernargs"] = len(e.pop("kernarg_hashes"))
        e["distinct_sequences"] = len(e.pop("sequence_hashes"))
        out.append(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", nargs="*", default=DEFAULT_BOXES)
    ap.add_argument("--roots", nargs="*", default=[])
    ap.add_argument("--out", default=CORPUS_DIR)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--ingest", action="store_true")
    args = ap.parse_args()

    harvest_ts = int(time.time())
    captures, raw_shapes, benches, contracts = [], [], [], []
    kinds, failures, malformed = Counter(), {}, Counter()

    for box in args.boxes:
        roots = args.roots or (LOCAL_ROOTS if box == LOCAL_BOX else [])
        print(f"[{box}] harvesting...", file=sys.stderr)
        envs, err = fetch(box, roots, args.timeout)
        if err:
            failures[box] = err
            print(f"[{box}] FAILED: {err}", file=sys.stderr)
            continue
        n0 = len(captures)
        for e in envs:
            path, mtime = e["p"], e["m"]
            try:
                doc = json.loads(e["b"])
            except json.JSONDecodeError:
                malformed[box] += 1
                continue
            root = doc if isinstance(doc, dict) else {}
            kind = classify(root) if root else "capture"
            kinds[kind] += 1
            if kind == "capture" or "redline_capture" in e["b"][:100000]:
                c, s, ct = parse_capture(root or {}, path, box, mtime)
                contracts.extend(ct)
                if not c and isinstance(doc, list):
                    for sub in doc:
                        if isinstance(sub, dict):
                            c2, s2, ct2 = parse_capture(sub, path, box, mtime)
                            contracts.extend(ct2)
                            c += c2
                            s += s2
                captures.extend(c)
                raw_shapes.extend(s)
                if c:
                    continue
            if root:
                benches.append(parse_bench(root, path, box, mtime, kind))
        print(f"[{box}] {len(envs)} files, {len(captures)-n0} captures",
              file=sys.stderr)

    if failures and not args.allow_partial:
        print("\nERROR: unreachable: " +
              ", ".join(f"{b} ({e})" for b, e in failures.items()), file=sys.stderr)
        print("Refusing to write a partial corpus (--allow-partial to override).",
              file=sys.stderr)
        return 2

    # Dedup captures on (sequence_hash, arm, experiment, tok_s) so re-harvesting
    # the same file from several worktrees collapses.
    seen, caps = {}, []
    for c in captures:
        k = _sha16(c["sequence_hash"], c["experiment"], c["arm"],
                   c["tok_s"], c["launches"], c["model"])
        if k in seen:
            seen[k]["_prov"].setdefault("also_seen", 0)
            seen[k]["_prov"]["also_seen"] += 1
            if (c["_prov"]["box"], c["_prov"]["path"]) < (
                    seen[k]["_prov"]["box"], seen[k]["_prov"]["path"]):
                seen[k]["_prov"].update(box=c["_prov"]["box"], path=c["_prov"]["path"],
                                        mtime=c["_prov"]["mtime"])
            continue
        c["capture_key"] = k
        seen[k] = c
        caps.append(c)

    shapes = dedup_shapes(raw_shapes)

    cseen, contract_rows = {}, []
    for ct in contracts:
        k = _sha16(ct["arch"], ct["kernel"], ct["captured_kernarg_bytes"],
                   ct["loader_kernarg_bytes"], ct["static_group_bytes"],
                   ct["dynamic_group_bytes"])
        if k in cseen:
            cseen[k]["occurrences"] += 1
            continue
        ct["contract_key"] = k
        ct["occurrences"] = 1
        cseen[k] = ct
        contract_rows.append(ct)
    contract_rows.sort(key=lambda r: (r.get("arch") or "", r.get("kernel") or "",
                                      r["contract_key"]))

    bseen, bench = set(), []
    for b in benches:
        k = _sha16(b["kind"], b["backend"], b["bench"], b["arch"], b["arm"],
                   b["experiment"], b["speedup"], b["output_sha256"])
        if k in bseen:
            continue
        bseen.add(k)
        b["bench_key"] = k
        bench.append(b)

    caps.sort(key=lambda r: (r.get("arch") or "", r.get("experiment") or "",
                             r.get("arm") or "", r["capture_key"]))
    shapes.sort(key=lambda r: (r.get("arch") or "", r.get("kernel") or "",
                               r["shape_key"]))
    bench.sort(key=lambda r: (r.get("arch") or "", r.get("kind") or "",
                              r.get("backend") or "", r["bench_key"]))

    tot_launch = sum(c["launches"] or 0 for c in caps)
    archs = sorted({c["arch"] for c in caps if c.get("arch")})
    kern = {s["kernel"] for s in shapes if s.get("kernel")}
    arts = {s["artifact"] for s in shapes if s.get("artifact")}
    seqs = {c["sequence_hash"] for c in caps if c.get("sequence_hash")}

    print(f"\n{'='*68}")
    print(f"captures      : {len(caps):6d}  ({tot_launch:,} launches)")
    print(f"launch shapes : {len(shapes):6d} distinct (from {len(raw_shapes):,} raw)")
    print(f"bench rows    : {len(bench):6d}")
    mism = sum(1 for r in contract_rows if r["kernarg_differs"])
    over = sum(1 for r in contract_rows if r["kernarg_captured_exceeds_loader"])
    print(f"aql contracts : {len(contract_rows):6d}  ({mism} differ, {over} captured>loader)")
    shadow = [c for c in caps if c.get("shadow_backend")]
    bx = sum(1 for c in shadow if c.get("shadow_bit_exact"))
    print(f"pm4 shadows   : {len(shadow):6d}  ({bx} bit_exact)")
    print(f"kernels       : {len(kern):6d}   .hsaco artifacts: {len(arts)}")
    print(f"sequence hashes:{len(seqs):6d}")
    print(f"archs         : {', '.join(archs)}")
    print(f"shapes seen   : {dict(kinds.most_common())}")
    if malformed:
        print(f"malformed     : {dict(malformed)}")
    if failures:
        print(f"PARTIAL       : missing {', '.join(failures)}")
    print("=" * 68)

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 1 if failures else 0

    os.makedirs(args.out, exist_ok=True)
    for name, rows in (("captures.jsonl", caps),
                       ("launch_shapes.jsonl", shapes),
                       ("bench.jsonl", bench),
                       ("aql_contracts.jsonl", contract_rows)):
        p = os.path.join(args.out, name)
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
        print(f"wrote {p} ({len(rows)} rows)")

    manifest = {
        "harvest_ts": harvest_ts,
        "harvest_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(harvest_ts)),
        "boxes": list(args.boxes),
        "unreachable": failures,
        "captures": len(caps),
        "total_launches": tot_launch,
        "launch_shapes": len(shapes),
        "raw_launches": len(raw_shapes),
        "bench_rows": len(bench),
        "aql_contracts": len(contract_rows),
        "kernarg_differs": mism,
        "kernarg_captured_exceeds_loader": over,
        "pm4_shadows": len(shadow),
        "pm4_shadow_bit_exact": bx,
        "distinct_kernels": len(kern),
        "distinct_artifacts": len(arts),
        "sequence_hashes": len(seqs),
        "archs": archs,
        "shapes_seen": dict(kinds.most_common()),
        "malformed": dict(malformed),
    }
    mp = os.path.join(args.out, "manifest.json")
    with open(mp, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {mp}")

    if args.ingest:
        print(f"ingested -> {ingest(caps, shapes, bench, contract_rows)}")
    return 1 if failures else 0


SCHEMA = """
CREATE TABLE IF NOT EXISTS captures(
  capture_key TEXT PRIMARY KEY, sequence_hash TEXT, arch TEXT, model TEXT,
  kv_mode TEXT, launches INTEGER, unique_kernels INTEGER, tok_s REAL,
  us_per_token REAL, ms REAL, context_tokens INTEGER, iterations INTEGER,
  aql_contract_probe INTEGER, aql_shadow INTEGER,
  experiment TEXT, arm TEXT, box TEXT, path TEXT, mtime INTEGER,
  shadow_backend TEXT, shadow_dispatches INTEGER, shadow_packets INTEGER,
  shadow_command_dwords INTEGER, shadow_queue_id INTEGER,
  shadow_context_tokens INTEGER, shadow_iterations INTEGER,
  shadow_frame_checkpoint INTEGER,
  shadow_bit_exact INTEGER, shadow_pre_bit_exact INTEGER,
  shadow_pre_logits_equal INTEGER, shadow_pre_kv_equal INTEGER,
  shadow_pre_recurrent_equal INTEGER, shadow_hip_oracles_equal INTEGER,
  shadow_blob_bit_exact INTEGER, shadow_logits_equal INTEGER,
  shadow_force_blob_path INTEGER);
CREATE TABLE IF NOT EXISTS launch_shapes(
  shape_key TEXT PRIMARY KEY, arch TEXT, kernel TEXT, artifact TEXT,
  artifact_path TEXT, grid TEXT, block TEXT, shared_mem INTEGER,
  kernarg_bytes INTEGER, occurrences INTEGER, distinct_kernargs INTEGER,
  distinct_sequences INTEGER);
CREATE TABLE IF NOT EXISTS bench(
  bench_key TEXT PRIMARY KEY, kind TEXT, backend TEXT, bench TEXT, family TEXT,
  arch TEXT, timing_mode TEXT, classification TEXT, correctness TEXT,
  status TEXT, speedup REAL, hipcc_version TEXT, output_sha256 TEXT,
  daemon_sha256 TEXT, git_commit TEXT, run_tag TEXT, experiment TEXT, arm TEXT,
  n_measurements INTEGER, box TEXT, path TEXT);
CREATE TABLE IF NOT EXISTS aql_contracts(
  contract_key TEXT PRIMARY KEY, arch TEXT, kernel TEXT,
  captured_kernarg_bytes INTEGER, loader_kernarg_bytes INTEGER,
  loader_kernarg_alignment INTEGER, static_group_bytes INTEGER,
  dynamic_group_bytes INTEGER, kernarg_differs INTEGER,
  kernarg_captured_exceeds_loader INTEGER, occurrences INTEGER);
CREATE INDEX IF NOT EXISTS ix_aqlc ON aql_contracts(arch, kernel);
CREATE INDEX IF NOT EXISTS ix_cap   ON captures(arch, experiment, sequence_hash);
CREATE INDEX IF NOT EXISTS ix_shape ON launch_shapes(arch, kernel);
CREATE INDEX IF NOT EXISTS ix_bench ON bench(arch, backend, bench);
"""


def ingest(caps, shapes, bench, contract_rows) -> str:
    import sqlite3
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.executescript(SCHEMA)
    for t in ("captures", "launch_shapes", "bench", "aql_contracts"):
        c.execute(f"DELETE FROM {t}")
    CAP_COLS = ["capture_key","sequence_hash","arch","model","kv_mode","launches",
                "unique_kernels","tok_s","us_per_token","ms","context_tokens",
                "iterations","aql_contract_probe","aql_shadow","experiment","arm",
                "box","path","mtime","shadow_backend","shadow_dispatches",
                "shadow_packets","shadow_command_dwords","shadow_queue_id",
                "shadow_context_tokens","shadow_iterations","shadow_frame_checkpoint",
                "shadow_bit_exact","shadow_pre_bit_exact","shadow_pre_logits_equal",
                "shadow_pre_kv_equal","shadow_pre_recurrent_equal",
                "shadow_hip_oracles_equal","shadow_blob_bit_exact",
                "shadow_logits_equal","shadow_force_blob_path"]
    # Named binding, not positional: the shadow_* fields reached the JSONL but not
    # the DB when this was a fixed 19-slot positional INSERT, and nothing failed.
    ins = ("INSERT OR IGNORE INTO captures(" + ",".join(CAP_COLS) + ") VALUES(" +
           ",".join(":" + c for c in CAP_COLS) + ")")
    for r in caps:
        p = r["_prov"]
        row = {c: r.get(c) for c in CAP_COLS}
        row.update(box=p["box"], path=p["path"], mtime=p["mtime"])
        for k, v in row.items():
            if isinstance(v, bool):
                row[k] = int(v)
        c.execute(ins, row)
    for r in shapes:
        c.execute("INSERT OR IGNORE INTO launch_shapes VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                  (r["shape_key"], r["arch"], r["kernel"], r["artifact"],
                   r["artifact_path"], json.dumps(r["grid"]), json.dumps(r["block"]),
                   r["shared_mem"], r["kernarg_bytes"], r["occurrences"],
                   r["distinct_kernargs"], r["distinct_sequences"]))
    for r in bench:
        p = r["_prov"]
        c.execute("INSERT OR IGNORE INTO bench VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (r["bench_key"], r["kind"], r["backend"], r["bench"], r["family"],
                   r["arch"], r["timing_mode"], r["classification"], r["correctness"],
                   r["status"], r["speedup"], r["hipcc_version"], r["output_sha256"],
                   r["daemon_sha256"], r["git_commit"], r["run_tag"], r["experiment"],
                   r["arm"], r.get("n_measurements"), p["box"], p["path"]))
    for r in contract_rows:
        c.execute("INSERT OR IGNORE INTO aql_contracts VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                  (r["contract_key"], r["arch"], r["kernel"],
                   r["captured_kernarg_bytes"], r["loader_kernarg_bytes"],
                   r["loader_kernarg_alignment"], r["static_group_bytes"],
                   r["dynamic_group_bytes"], int(bool(r["kernarg_differs"])),
                   int(bool(r["kernarg_captured_exceeds_loader"])),
                   r["occurrences"]))
    c.commit()
    c.close()
    return DB_PATH


if __name__ == "__main__":
    sys.exit(main())
