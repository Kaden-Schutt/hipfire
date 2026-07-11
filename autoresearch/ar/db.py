# Copyright (c) Kaden Schutt
"""ar.db — durable SQLite store + idempotent ledger ingest for the loop.

The git-tracked ``autoresearch/ledger/*.jsonl`` corpus is the source of truth
(every A/B — win, loss, or noise — is one append-only self-describing JSON
line). ``ar.db`` is a queryable index rebuilt from it. Ingest is idempotent on
``measurement_hash`` (``sha256(gpu_arch|model|base_sha|var_sha|prompt_md5|kv|
maxtok)[:16]``) so ``ar ingest`` can be re-run any time without double-counting.

Ports the query surface of ``autoresearch/oracle_db.py`` (wins/best/history/
kernel roll-ups) and the schema + ``ingest()`` of ``autoresearch/ar/
hipfire_ar.py`` onto the durable store, extending the row to the conjunctive
perf gate (both ``tok_delta`` and ``dur_delta`` recorded).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sqlite3
import time

# WIN resets the streak; these count toward EXHAUSTED (matches driver_v3 /
# hipfire_ar update_exhaustion semantics).
DEAD_VERDICTS = {"DEAD", "INCONCLUSIVE", "LOSS", "NOISE"}

_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db", "schema.sql")


def _schema_sql() -> str:
    with open(_SCHEMA_PATH, "r") as fh:
        return fh.read()


def connect(path: str) -> sqlite3.Connection:
    """Open (creating if needed) the ar.db store with the schema applied.

    ``path`` is the sqlite file path (``":memory:"`` works for tests). Rows come
    back as :class:`sqlite3.Row` (dict-like access by column name).
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_schema_sql())
    conn.commit()
    return conn


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_name(basename: str) -> tuple[str, str]:
    """Fallback arch/kernel from a ``swarm_<arch>_<kernel>.jsonl`` filename.

    Only used when a row omits its own ``arch``/``kernel`` — real ledger rows
    carry both, so this is a legacy safety net.
    """
    stem = basename[: -len(".jsonl")] if basename.endswith(".jsonl") else basename
    if stem.startswith("swarm_"):
        stem = stem[len("swarm_") :]
    parts = stem.split("_", 1)
    arch = parts[0]
    kernel = parts[1] if len(parts) > 1 else stem
    return arch, kernel


def _measurement_hash(row: dict, arch: str) -> str:
    """Row identity: prefer an explicit ``measurement_hash``; else the canonical
    self-describing recipe; else a deterministic hash of the whole row (which
    still dedups byte-identical duplicate lines on re-ingest)."""
    mh = row.get("measurement_hash")
    if mh:
        return str(mh)
    fields = [
        row.get("gpu_arch") or arch,
        row.get("model"),
        row.get("base_sha"),
        row.get("variant_sha") or row.get("var_sha"),
        row.get("prompt_md5"),
        row.get("kv"),
        row.get("maxtok"),
    ]
    if all(f is not None for f in fields):
        payload = "|".join(str(f) for f in fields)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
    payload = json.dumps(row, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _refresh_bod(conn: sqlite3.Connection, path: str) -> None:
    """Replace the BOD snapshot for one arch from a ``bod_<arch>.json`` file."""
    try:
        data = json.load(open(path))
    except Exception:
        return
    base = os.path.basename(path)
    arch = base
    if arch.startswith("bod_"):
        arch = arch[len("bod_") :]
    if arch.endswith(".json"):
        arch = arch[: -len(".json")]
    arch = data.get("arch") or arch
    now = int(time.time())
    conn.execute("DELETE FROM bod WHERE arch=?", (arch,))
    for r in data.get("rows", []):
        conn.execute(
            "INSERT INTO bod(arch,kernel,wall_pct,l2_hit,mem_busy,occ,vgpr,snap_ts) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                arch,
                r.get("kernel"),
                _num(r.get("wall_pct")),
                _num(r.get("l2_hit_pct", r.get("l2_hit"))),
                _num(r.get("mem_busy")),
                _num(r.get("occ")),
                r.get("vgpr"),
                now,
            ),
        )


def ingest(conn: sqlite3.Connection, ledger_dir: str, bod_glob: str) -> int:
    """Index every ledger row + refresh the BOD snapshots. Idempotent.

    Returns the number of distinct ledger rows (by ``measurement_hash``) seen in
    ``ledger_dir`` — stable across re-ingest, so a second call returns the same
    count and inserts nothing new (``INSERT OR IGNORE`` on the unique hash).
    """
    seen: set[str] = set()
    for f in sorted(glob.glob(os.path.join(ledger_dir, "*.jsonl"))):
        farch, fkern = _parse_name(os.path.basename(f))
        with open(f, errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                arch = row.get("gpu_arch") or row.get("arch") or farch
                kernel = row.get("kernel") or fkern
                mh = _measurement_hash(row, arch)
                if mh in seen:
                    continue
                seen.add(mh)
                lever = row.get("lever") or row.get("label") or "?"
                conn.execute(
                    "INSERT OR IGNORE INTO attempts"
                    "(arch,kernel,lever,verdict,tok_delta,dur_delta,profile,base_sha,var_sha,measurement_hash,ts)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        arch,
                        kernel,
                        lever,
                        row.get("verdict"),
                        _num(row.get("tok_delta_pct", row.get("delta_pct"))),
                        _num(row.get("dur_delta_pct")),
                        row.get("profile") or row.get("profile_feedback") or "",
                        row.get("base_sha"),
                        row.get("variant_sha") or row.get("var_sha"),
                        mh,
                        int(row.get("ts") or 0),
                    ),
                )
    for bf in sorted(glob.glob(bod_glob)):
        _refresh_bod(conn, bf)
    conn.commit()
    return len(seen)


# ── query surface (ported from oracle_db.py, onto the durable store) ──────────


def wins(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every certified WIN, best (tok/s delta) first."""
    return conn.execute(
        "SELECT * FROM attempts WHERE verdict='WIN' ORDER BY tok_delta DESC"
    ).fetchall()


def best(conn: sqlite3.Connection, arch: str, kernel: str):
    """The best certified WIN variant for a kernel (or ``None``)."""
    return conn.execute(
        "SELECT * FROM attempts WHERE arch=? AND kernel=? AND verdict='WIN' "
        "ORDER BY tok_delta DESC LIMIT 1",
        (arch, kernel),
    ).fetchone()


def history(conn: sqlite3.Connection, arch: str, kernel: str) -> list[sqlite3.Row]:
    """Full A/B history for a kernel, oldest first."""
    return conn.execute(
        "SELECT * FROM attempts WHERE arch=? AND kernel=? ORDER BY ts, id",
        (arch, kernel),
    ).fetchall()


def kernel_stats(conn: sqlite3.Connection, arch: str, kernel: str, k: int) -> dict:
    """Roll-up for a kernel: ``{tried, wins, best_win_pct, consecutive_dead}``
    (plus ``exhausted``). The consecutive-dead streak counts from the tail; a
    WIN resets it (matches update_exhaustion)."""
    rows = conn.execute(
        "SELECT verdict, tok_delta FROM attempts WHERE arch=? AND kernel=? ORDER BY ts, id",
        (arch, kernel),
    ).fetchall()
    win_rows = [r for r in rows if r["verdict"] == "WIN"]
    streak = 0
    for r in reversed(rows):
        if r["verdict"] == "WIN":
            break
        if r["verdict"] in DEAD_VERDICTS:
            streak += 1
    best_win = max((r["tok_delta"] for r in win_rows if r["tok_delta"] is not None), default=None)
    return {
        "tried": len(rows),
        "wins": len(win_rows),
        "best_win_pct": best_win,
        "consecutive_dead": streak,
        "exhausted": streak >= k and not win_rows,
    }
