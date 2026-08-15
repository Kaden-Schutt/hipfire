#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Harvest scattered autoresearch ledgers across the fleet into one corpus.

The autoresearch loop writes one self-describing JSON line per A/B (win, loss,
or noise) into ``autoresearch/ledger/*.jsonl``. Those files accumulate in every
worktree on every box -- including two disposable ones -- and have never been
read in aggregate. This script pulls them together.

Pull-based, read-only at the source, idempotent. Run it from anywhere; it
ssh's to each configured box, streams the ledger / kernel-ledger / atlas / BOD
files, normalizes them into two flat corpora, dedups, and writes:

    autoresearch/corpus/attempts.jsonl   outcome facts   (git-tracked)
    autoresearch/corpus/kernels.jsonl    static ISA facts (git-tracked)
    autoresearch/corpus/bod.jsonl        BOD snapshots   (git-tracked)

``attempts.jsonl`` is deliberately ledger-shaped, so the existing
``ar.db.ingest()`` indexes the corpus directory with no changes.

Identity follows autoresearch/db/schema.sql: ``measurement_hash`` is the
canonical ``sha256(gpu_arch|model|base_sha|var_sha|prompt_md5|kv|maxtok)[:16]``
and is left NULL when any component is missing -- which is the common case
(~97% of historical rows). A separate ``fallback_key`` carries the weaker
composite so those rows still dedup, without minting look-alike hashes under
the canonical column name.

Scope is autoresearch only. Redline artifacts are excluded by construction:
they answer a different question (retained-PM4 replay parity, not
lever-vs-baseline perf) and have a different identity model.

Usage:
    scripts/harvest_ledgers.py                      # all boxes -> corpus
    scripts/harvest_ledgers.py --dry-run            # report, write nothing
    scripts/harvest_ledgers.py --boxes k9lin hipx   # subset
    scripts/harvest_ledgers.py --allow-partial      # tolerate a dead box
    scripts/harvest_ledgers.py --ingest             # also rebuild ar.db
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
CORPUS_DIR = os.path.join(REPO, "autoresearch", "corpus")

# Local box first: harvested without ssh.
DEFAULT_BOXES = ["k9lin", "hipx", "hiptrx"]
LOCAL_BOX = "k9lin"

# ── remote extractor ─────────────────────────────────────────────────────────
# Runs on each box via `ssh <box> python3 -`. Read-only: walks the tree and
# emits one envelope per source line. Kept dependency-free and stdlib-only so
# it runs on whatever Python the box happens to have.
REMOTE = r'''
import json, os, re, sys

# /tmp is included deliberately: hiptrx keeps 23 ledger files / 531 rows there,
# and tmp data is the most at-risk on the fleet -- exactly what a durable corpus
# is for. Duplicate roots are de-duplicated below so ~ and /tmp can overlap.
roots = sys.argv[1:] or [os.path.expanduser("~"), "/tmp"]
seen_roots, _r = set(), []
for r in roots:
    rp = os.path.realpath(r)
    if rp not in seen_roots and os.path.isdir(rp):
        seen_roots.add(rp)
        _r.append(rp)
roots = _r

# Prune only toolchains and non-outcome data. niah/prompts/fixtures are eval and
# test data; pytest-of-* are throwaway fixture DBs.
#
# Redline is deliberately NOT pruned here. Redline worktrees contain full hipfire
# checkouts whose autoresearch loop wrote real ledgers -- e.g.
# ~/.redline-work/hipfire-redline-kernel-oracle/autoresearch/ledger/. Pruning the
# path would silently drop ~1.3k autoresearch rows on hiptrx alone. Redline's OWN
# artifacts are excluded by OWNED below instead, which keeps the two corpora
# separate without losing autoresearch data that happens to sit under one.
EXCL = re.compile(
    r"/(\.git|node_modules|\.cargo|site-packages|__pycache__|\.rustup|\.bun|\.npm"
    r"|llvm|mesa|rocm-systems|target)/"
    r"|/pytest-of-|/niah/|/prompts/|/fixtures/"
)

# A ledger/kernel-ledger/bod file counts as autoresearch only if it sits under
# one of these. Atlas emissions are exempt: they are self-identifying via
# schema=hipfire.kernel_atlas.v0 and land in ad-hoc experiment dirs (e.g.
# .worktrees/atlas-mq4-publish-sweep/experiments/), so a path rule would drop them.
OWNED = re.compile(r"/autoresearch/|/kernel-ledger/|/tests/")

# Redline's OWN artifact roots. Everything here belongs to the redline corpus,
# which has a different identity model and is out of scope. Note this does NOT
# match a hipfire checkout nested under a redline worktree -- those carry genuine
# autoresearch ledgers and are kept.
REDLINE_OWN = re.compile(r"/redline-(results|runs|cert)")

PATS = [
    ("ledger",        re.compile(r"/autoresearch/ledger/[^/]+\.jsonl$")),
    ("kernel_ledger", re.compile(r"/kernel-ledger/[^/]+\.jsonl$")),
    ("atlas",         re.compile(r"/atlas[^/]*\.jsonl$")),
    ("bod",           re.compile(r"/bod_[^/]*\.json$")),
]

out = sys.stdout
nf = 0
for root in roots:
    for dp, dn, fn in os.walk(root, onerror=lambda e: None):
        if EXCL.search(dp + "/"):
            dn[:] = []
            continue
        for f in fn:
            p = os.path.join(dp, f)
            kind = None
            for k, rx in PATS:
                if rx.search(p):
                    kind = k
                    break
            if kind is None:
                continue
            if REDLINE_OWN.search(p):
                continue
            if kind != "atlas" and not OWNED.search(p):
                continue
            try:
                mtime = int(os.path.getmtime(p))
            except OSError:
                continue
            nf += 1
            try:
                if kind == "bod":
                    with open(p, errors="ignore") as fh:
                        body = fh.read()
                    out.write(json.dumps({"k": kind, "p": p, "m": mtime, "b": body}) + "\n")
                else:
                    with open(p, errors="ignore") as fh:
                        for i, line in enumerate(fh, 1):
                            line = line.strip()
                            if line:
                                out.write(json.dumps(
                                    {"k": kind, "p": p, "m": mtime, "n": i, "l": line}) + "\n")
            except OSError:
                continue
sys.stderr.write("files=%d\n" % nf)
'''


# ── helpers ──────────────────────────────────────────────────────────────────

def _sha16(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _ts(x) -> int:
    """Epoch seconds from an int, a float, or an ISO-8601 string.

    The ledger writers disagree: k9lin/hipx emit epoch ints, hiptrx emits
    ``2026-07-09T23:47:48Z``. Both are normalized here so ``ts`` sorts.
    """
    if x is None or x == "":
        return 0
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x).strip()
    try:
        return int(float(s))
    except ValueError:
        pass
    try:
        from datetime import datetime
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def _parse_name(basename: str) -> tuple[str, str]:
    """Fallback (arch, kernel) from a ``swarm_<arch>_<kernel>.jsonl`` filename.

    Mirrors ar.db._parse_name -- only used when a row omits its own fields.
    """
    stem = basename[: -len(".jsonl")] if basename.endswith(".jsonl") else basename
    for pfx in ("swarm_", "fleet_"):
        if stem.startswith(pfx):
            stem = stem[len(pfx):]
    parts = stem.split("_", 1)
    return parts[0], (parts[1] if len(parts) > 1 else stem)


# Raw source keys that normalize_attempt actually consumes. Anything NOT listed
# here survives verbatim in `extra` -- including base_runs/var_runs (the raw
# repeat measurements), base_sclk/var_sclk, confirmed, and cross_arch.
#
# This list must stay exactly the consumed set. An earlier version also listed
# fields that were never carried, which excluded them from `extra` and dropped
# them outright -- collapsing distinct experiments that differed only in their
# raw repeats into a single row.
_CONSUMED = {
    "gpu_arch", "arch", "kernel", "lever", "label", "variant",
    "verdict", "WIN", "tok_delta_pct", "delta_pct", "dur_delta_pct",
    "mwu_dominance", "rounds", "base_decode", "var_decode",
    "base_coh", "var_coh", "win_commit", "roofline",
    "profile", "profile_feedback", "base_sha", "variant_sha", "var_sha",
    "model", "prompt_md5", "kv", "maxtok", "ts", "measurement_hash",
}


def normalize_attempt(row: dict, path: str) -> dict:
    farch, fkern = _parse_name(os.path.basename(path))
    arch = row.get("gpu_arch") or row.get("arch") or farch
    kernel = row.get("kernel") or fkern
    label = row.get("label") or row.get("lever")
    variant = row.get("variant")
    var_sha = row.get("variant_sha") or row.get("var_sha")

    out = {
        "arch": arch,
        "gpu_arch": row.get("gpu_arch") or arch,
        "kernel": kernel,
        "lever": row.get("lever") or label or "?",
        "label": label,
        "variant": variant,
        "verdict": (row.get("verdict") or "").strip().upper() or None,
        "WIN": row.get("WIN"),
        "tok_delta_pct": _num(row.get("tok_delta_pct", row.get("delta_pct"))),
        "dur_delta_pct": _num(row.get("dur_delta_pct")),
        "delta_pct": _num(row.get("delta_pct")),
        "mwu_dominance": _num(row.get("mwu_dominance")),
        "rounds": row.get("rounds"),
        "base_decode": _num(row.get("base_decode")),
        "var_decode": _num(row.get("var_decode")),
        "base_coh": row.get("base_coh"),
        "var_coh": row.get("var_coh"),
        "win_commit": row.get("win_commit"),
        "roofline": row.get("roofline"),
        "profile": row.get("profile") or row.get("profile_feedback") or "",
        "base_sha": row.get("base_sha"),
        "var_sha": var_sha,
        "model": row.get("model"),
        "prompt_md5": row.get("prompt_md5"),
        "kv": row.get("kv"),
        "maxtok": row.get("maxtok"),
        "ts": _ts(row.get("ts")),
    }

    # Canonical identity, per autoresearch/db/schema.sql. Stays NULL unless every
    # component is present -- a look-alike hash here would silently fail to match
    # rows the loop emits later.
    if row.get("measurement_hash"):
        out["measurement_hash"] = str(row["measurement_hash"])
        conf = "canonical"
    else:
        comps = [out["gpu_arch"], out["model"], out["base_sha"], out["var_sha"],
                 out["prompt_md5"], out["kv"], out["maxtok"]]
        if all(c is not None for c in comps):
            out["measurement_hash"] = _sha16(*comps)
            conf = "canonical"
        else:
            out["measurement_hash"] = None
            conf = None

    fb = [arch, kernel, label, out["base_sha"], variant, out["prompt_md5"]]
    out["fallback_key"] = _sha16(*["" if c is None else c for c in fb])
    if conf is None:
        conf = "strong" if all(c is not None for c in fb) else "weak"
    out["key_confidence"] = conf

    extra = {k: v for k, v in row.items() if k not in _CONSUMED}
    if extra:
        out["extra"] = extra
    return out


def normalize_kernel(row: dict, path: str) -> dict:
    m = row.get("metrics") or {}
    repro = row.get("reproducer") or {}
    out = {
        "schema": row.get("schema"),
        "arch": row.get("arch"),
        "kernel": row.get("kernel"),
        "phase": row.get("phase"),
        "workload_kind": row.get("workload_kind"),
        "quant": row.get("quant"),
        "shape_bucket": row.get("shape_bucket"),
        "bound_class": row.get("bound_class"),
        "isa_fingerprint": m.get("isa_fingerprint"),
        "vgpr": m.get("vgpr"),
        "sgpr": m.get("sgpr"),
        "lds": m.get("lds"),
        "scratch": m.get("scratch"),
        "fixture_path": repro.get("fixture_path"),
        "repro_cmd": repro.get("cmd"),
        "metrics": m,
    }
    known = set(out) | {"metrics", "reproducer", "artifacts"}
    extra = {k: v for k, v in row.items() if k not in known}
    if extra:
        out["extra"] = extra

    # Static identity is the compiled kernel itself -- isa_fingerprint already
    # hashes the ISA. Atlas rows (AtlasRow schema) often omit it, so fall back to
    # a content hash of the discriminating fields; keying those on
    # (arch, kernel, shape_bucket) alone would collapse every distinct atlas row
    # for a kernel into one bogus collision pile.
    if out["isa_fingerprint"] is not None:
        out["static_key"] = _sha16(out["arch"], out["kernel"],
                                   out["isa_fingerprint"], out["shape_bucket"] or "")
        out["static_key_kind"] = "isa"
    else:
        body = json.dumps({k: v for k, v in out.items() if k != "static_key"},
                          sort_keys=True, default=str)
        out["static_key"] = _sha16(out["arch"], out["kernel"], body)
        out["static_key_kind"] = "content"
    return out


def _populated(d: dict) -> int:
    """How many non-empty fields a row carries -- the dedup precedence signal."""
    return sum(1 for k, v in d.items()
               if not k.startswith("_") and v not in (None, "", [], {}))


def dedup(rows: list[dict], keyfn) -> tuple[list[dict], int, int]:
    """Collapse byte-identical rows; resolve same-key conflicts by richness.

    Precedence is (populated fields desc, source mtime desc). Genuinely distinct
    rows sharing a key are all retained -- losers flagged ``collision`` -- because
    silently dropping a divergent historical result would misrepresent coverage.
    """
    by_key: dict[str, dict] = defaultdict(dict)
    identical = 0
    for r in rows:
        k = keyfn(r)
        body = json.dumps({kk: vv for kk, vv in r.items() if not kk.startswith("_")},
                          sort_keys=True, default=str)
        ch = hashlib.sha256(body.encode()).hexdigest()[:16]
        if ch in by_key[k]:
            identical += 1
            prev = by_key[k][ch]
            prev["_prov"]["also_seen"] = prev["_prov"].get("also_seen", 0) + 1
            prev["_prov"]["boxes"] = sorted(
                set(prev["_prov"].get("boxes", [prev["_prov"]["box"]])) | {r["_prov"]["box"]})
            # Canonical provenance is the lexicographically smallest (box, path),
            # NOT "latest mtime wins" -- mtime ties would then be resolved by
            # os.walk order, which is not guaranteed stable across runs and made
            # the output non-idempotent.
            if (r["_prov"]["box"], r["_prov"]["path"]) < (prev["_prov"]["box"], prev["_prov"]["path"]):
                prev["_prov"]["box"] = r["_prov"]["box"]
                prev["_prov"]["path"] = r["_prov"]["path"]
                prev["_prov"]["mtime"] = r["_prov"]["mtime"]
            continue
        by_key[k][ch] = r

    out, collisions = [], 0
    for k in sorted(by_key):
        # Deterministic precedence: richer row wins, then newer, then a stable
        # content-hash tiebreak so equal rows never reorder between runs.
        vs = sorted(by_key[k].items(),
                    key=lambda kv: (-_populated(kv[1]), -kv[1]["_prov"]["mtime"], kv[0]))
        out.append(vs[0][1])
        for _, loser in vs[1:]:
            loser["key_confidence"] = "collision"
            collisions += 1
            out.append(loser)
    return out, identical, collisions


# ── harvest ──────────────────────────────────────────────────────────────────

def fetch(box: str, roots: list[str], timeout: int) -> tuple[list[dict], str | None]:
    """Stream one box. Returns (envelopes, error). Never writes to the source."""
    if box == LOCAL_BOX:
        cmd = [sys.executable, "-", *roots]
    else:
        cmd = ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", box,
               "python3 -", *roots]
    try:
        p = subprocess.run(cmd, input=REMOTE, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return [], f"timeout after {timeout}s"
    except OSError as e:
        return [], str(e)
    if p.returncode != 0:
        return [], (p.stderr.strip().splitlines() or ["exit %d" % p.returncode])[-1]

    envs, bad = [], 0
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        try:
            envs.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"  [{box}] warning: {bad} unreadable envelope lines", file=sys.stderr)
    return envs, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", nargs="*", default=DEFAULT_BOXES)
    ap.add_argument("--roots", nargs="*", default=[],
                    help="remote roots to walk (default: the box's $HOME)")
    ap.add_argument("--out", default=CORPUS_DIR)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write the corpus even if a box was unreachable")
    ap.add_argument("--ingest", action="store_true", help="rebuild ar.db from the corpus")
    args = ap.parse_args()

    harvest_ts = int(time.time())
    attempts, kernels, bods = [], [], []
    malformed = Counter()
    failures: dict[str, str] = {}
    per_box = Counter()

    for box in args.boxes:
        print(f"[{box}] harvesting...", file=sys.stderr)
        envs, err = fetch(box, args.roots, args.timeout)
        if err:
            failures[box] = err
            print(f"[{box}] FAILED: {err}", file=sys.stderr)
            continue
        for e in envs:
            kind, path, mtime = e["k"], e["p"], e["m"]
            # harvest_ts deliberately NOT stamped per row -- it changes every run
            # and would make every line of a git-tracked corpus diff on every
            # harvest. It lives once in manifest.json instead.
            prov = {"box": box, "path": path, "mtime": mtime}
            if kind == "bod":
                try:
                    data = json.loads(e["b"])
                except json.JSONDecodeError:
                    malformed[f"{box}:bod"] += 1
                    continue
                bods.append({"arch": data.get("arch"), "rows": data.get("rows", []),
                             "_prov": prov})
                per_box[box] += 1
                continue
            try:
                row = json.loads(e["l"])
            except json.JSONDecodeError:
                malformed[f"{box}:{kind}"] += 1
                print(f"  malformed {path}:{e.get('n')}", file=sys.stderr)
                continue
            if kind == "ledger":
                r = normalize_attempt(row, path)
                r["_prov"] = prov
                attempts.append(r)
            else:  # kernel_ledger | atlas
                r = normalize_kernel(row, path)
                r["_prov"] = prov
                kernels.append(r)
            per_box[box] += 1
        print(f"[{box}] {per_box[box]} rows", file=sys.stderr)

    if failures and not args.allow_partial:
        print("\nERROR: unreachable boxes: " +
              ", ".join(f"{b} ({e})" for b, e in failures.items()), file=sys.stderr)
        print("Refusing to write a partial corpus. Re-run with --allow-partial "
              "to accept it.", file=sys.stderr)
        return 2

    raw_a, raw_k = len(attempts), len(kernels)
    attempts, ident_a, coll_a = dedup(
        attempts, lambda r: r.get("measurement_hash") or r["fallback_key"])
    kernels, ident_k, coll_k = dedup(kernels, lambda r: r["static_key"])

    # Stable order so re-harvesting is a minimal diff, not a reshuffle. COALESCE
    # matters: measurement_hash is NULL on the large majority of historical rows.
    attempts.sort(key=lambda r: (r.get("arch") or "", r.get("kernel") or "",
                                 r.get("ts") or 0,
                                 r.get("measurement_hash") or r["fallback_key"]))
    kernels.sort(key=lambda r: (r.get("arch") or "", r.get("kernel") or "",
                                r["static_key"]))
    bods.sort(key=lambda r: (r.get("arch") or "", r["_prov"]["mtime"]))

    verd = Counter(r["verdict"] for r in attempts if r.get("verdict"))
    conf = Counter(r["key_confidence"] for r in attempts)
    archs = sorted({r["arch"] for r in attempts if r.get("arch")})
    kerns = {r["kernel"] for r in attempts if r.get("kernel")}

    print(f"\n{'='*66}")
    print(f"attempts : {raw_a:6d} raw -> {len(attempts):6d} unique "
          f"({ident_a} identical collapsed, {coll_a} collisions kept)")
    print(f"kernels  : {raw_k:6d} raw -> {len(kernels):6d} unique "
          f"({ident_k} identical collapsed, {coll_k} collisions kept)")
    print(f"bod      : {len(bods):6d} snapshots")
    print(f"archs    : {', '.join(archs)}")
    print(f"kernels  : {len(kerns)} distinct")
    print(f"verdicts : {dict(verd.most_common())}")
    print(f"key conf : {dict(conf.most_common())}")
    if malformed:
        print(f"malformed: {dict(malformed)}")
    if failures:
        print(f"PARTIAL  : missing {', '.join(failures)}")
    print("=" * 66)

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 1 if failures else 0

    os.makedirs(args.out, exist_ok=True)
    for name, rows in (("attempts.jsonl", attempts),
                       ("kernels.jsonl", kernels),
                       ("bod.jsonl", bods)):
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
        "raw": {"attempts": raw_a, "kernels": raw_k},
        "unique": {"attempts": len(attempts), "kernels": len(kernels), "bod": len(bods)},
        "collapsed": {"attempts": ident_a, "kernels": ident_k},
        "collisions": {"attempts": coll_a, "kernels": coll_k},
        "malformed": dict(malformed),
        "archs": archs,
        "distinct_kernels": len(kerns),
        "verdicts": dict(verd.most_common()),
        "key_confidence": dict(conf.most_common()),
    }
    mp = os.path.join(args.out, "manifest.json")
    with open(mp, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {mp}")

    if args.ingest:
        dbp = ingest_corpus(attempts, kernels, bods)
        print(f"ingested -> {dbp}")

    return 1 if failures else 0


def ingest_corpus(attempts: list[dict], kernels: list[dict], bods: list[dict]) -> str:
    """Index the corpus into ar.db (corpus_attempts / kernel_static / provenance).

    Deliberately does NOT reuse ``ar.db.ingest()``: that globs ``*.jsonl`` in the
    directory it is given, which would slurp kernels.jsonl and bod.jsonl in as
    bogus attempt rows. It also keys on canonical measurement_hash, which ~97% of
    harvested rows lack.
    """
    sys.path.insert(0, REPO)
    from autoresearch.ar import db as ardb  # noqa: E402

    dbp = os.path.join(REPO, "autoresearch", "db", "ar.db")
    conn = ardb.connect(dbp)          # applies schema.sql, creating the new tables
    conn.execute("DELETE FROM corpus_attempts")
    conn.execute("DELETE FROM kernel_static")
    conn.execute("DELETE FROM provenance")

    for r in attempts:
        ck = r.get("measurement_hash") or r["fallback_key"]
        conn.execute(
            "INSERT OR IGNORE INTO corpus_attempts(corpus_key,measurement_hash,"
            "fallback_key,key_confidence,arch,kernel,lever,label,variant,verdict,"
            "tok_delta,dur_delta,mwu_dominance,rounds,base_decode,var_decode,"
            "win_commit,profile,base_sha,var_sha,model,prompt_md5,ts)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ck, r.get("measurement_hash"), r["fallback_key"], r.get("key_confidence"),
             r.get("arch"), r.get("kernel"), r.get("lever"), r.get("label"),
             r.get("variant"), r.get("verdict"), r.get("tok_delta_pct"),
             r.get("dur_delta_pct"), r.get("mwu_dominance"), r.get("rounds"),
             r.get("base_decode"), r.get("var_decode"), r.get("win_commit"),
             r.get("profile"), r.get("base_sha"), r.get("var_sha"), r.get("model"),
             r.get("prompt_md5"), r.get("ts")))
        p = r["_prov"]
        conn.execute(
            "INSERT INTO provenance(corpus_key,table_name,box,path,mtime,also_seen,boxes)"
            " VALUES(?,?,?,?,?,?,?)",
            (ck, "corpus_attempts", p.get("box"), p.get("path"), p.get("mtime"),
             p.get("also_seen", 0), ",".join(p.get("boxes", [p.get("box")]))))

    for k in kernels:
        conn.execute(
            "INSERT OR IGNORE INTO kernel_static(static_key,static_key_kind,arch,kernel,"
            "phase,workload_kind,quant,shape_bucket,bound_class,isa_fingerprint,"
            "vgpr,sgpr,lds,scratch,fixture_path)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (k["static_key"], k.get("static_key_kind"), k.get("arch"), k.get("kernel"),
             k.get("phase"), k.get("workload_kind"), k.get("quant"),
             k.get("shape_bucket"), k.get("bound_class"),
             str(k.get("isa_fingerprint")) if k.get("isa_fingerprint") is not None else None,
             k.get("vgpr"), k.get("sgpr"), k.get("lds"), k.get("scratch"),
             k.get("fixture_path")))

    now = int(time.time())
    conn.execute("DELETE FROM bod")
    for b in bods:
        for row in b.get("rows", []):
            conn.execute(
                "INSERT INTO bod(arch,kernel,wall_pct,l2_hit,mem_busy,occ,vgpr,snap_ts)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (b.get("arch"), row.get("kernel"), _num(row.get("wall_pct")),
                 _num(row.get("l2_hit_pct", row.get("l2_hit"))),
                 _num(row.get("mem_busy")), _num(row.get("occ")),
                 row.get("vgpr"), now))
    conn.commit()
    conn.close()
    return dbp


if __name__ == "__main__":
    sys.exit(main())
