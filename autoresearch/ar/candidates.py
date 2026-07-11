# Copyright (c) Kaden Schutt
"""ar.candidates — candidate selection + exhaustion + tried-lever digest.

Converges four predecessors onto one module:

* ``harness/exhaustion.py`` — the ONE per-kernel exhaustion definition (the
  per-round primitives :func:`apply_round` / :func:`kernel_exhausted` /
  :func:`all_exhausted` / :func:`dead_streak`).
* ``harness/v2/check_exhausted.py`` — the GLOBAL :func:`is_exhausted` predicate
  (every wall>=cand_wall candidate at K consecutive deads ⇒ stop).
* ``harness/v2/gen_digest.py`` — :func:`gen_digest`, the codex-facing coverage
  text (candidate kernels, wall%, N/K exhaustion, levers already tried).
* ``harness/v2/update_exhaustion.py`` — :func:`update_exhaustion`, which folds a
  round's ledger verdicts into the per-kernel consecutive-dead counters.

Candidate ``state ∈ {"OPEN","EXHAUSTED"}``. The exhaustion store is per-kernel
stats (``{"consecutive_dead", "tried", "wins", "best_win_pct", "levers"}``,
typically populated from :func:`ar.db.kernel_stats` + :func:`ar.db.history`); the
readers also accept the legacy int form (bare consecutive-dead count) and the
:func:`apply_round` state form (``{"dead", "infra", "parity"}``).

Exhaustion semantics (from ``exhaustion.py``): a WIN resets the streak;
DEAD/COHERENCE_FAIL/LOSS/NOISE count; PARITY_FAIL counts only once REPEATED;
INCONCLUSIVE does NOT count (routed to a re-measure queue); BUILD_FAIL/VOID use a
SEPARATE infra cap. The dead streak advances by AT MOST 1 per round, so K counts
consecutive ROUNDS, not attempts.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

# ── verdict taxonomy (verbatim from exhaustion.py) ────────────────────────────
WIN = "WIN"
DEAD_VERDICTS = frozenset({"DEAD", "COHERENCE_FAIL", "LOSS", "NOISE"})
INFRA_VERDICTS = frozenset({"BUILD_FAIL", "VARIANT_BUILD_FAIL", "BASELINE_BUILD_FAIL", "VOID"})
INCONCLUSIVE = "INCONCLUSIVE"
PARITY_FAIL = "PARITY_FAIL"

# update_exhaustion (ledger-driven) also counts the certify's target-guard
# rejections — a dead/no-op swap is exhausting, not codex-fixable.
_LEDGER_DEAD = frozenset({"DEAD", "COHERENCE_FAIL", "LOSS", "NOISE", "DEAD_FILE", "NO_OP"})


@dataclass
class Candidate:
    """One selectable BOD kernel with its roofline lens + A/B history roll-up.

    ``state`` is ``"EXHAUSTED"`` once the consecutive-dead streak reaches K,
    else ``"OPEN"``.
    """

    kernel: str
    wall_pct: float
    mem_busy: float | None = None
    occ: float | None = None
    vgpr: int | None = None
    l2_hit: float | None = None
    tried: int = 0
    wins: int = 0
    best_win_pct: float | None = None
    state: str = "OPEN"


# ── exhaustion store readers (accept dict / apply_round-state / legacy int) ────


def _dead_count(entry) -> int:
    """Consecutive-dead count from any accepted exhaustion-store entry form."""
    if isinstance(entry, dict):
        v = entry.get("consecutive_dead", entry.get("dead", 0))
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0
    try:
        return int(entry)
    except (TypeError, ValueError):
        return 0


def _tried_levers(entry) -> list[tuple[str, str]]:
    """Normalize an entry's tried-lever list to ``[(lever, verdict), ...]``."""
    if not isinstance(entry, dict):
        return []
    raw = entry.get("levers") or entry.get("tried_levers") or []
    out: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append((str(item.get("lever", "?")), str(item.get("verdict", "?"))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((str(item[0]), str(item[1])))
    return out


def _live_candidates(bod: dict, cand_wall: float, folded) -> list[tuple[str, float]]:
    fold = set(folded or [])
    return [
        (r["kernel"], float(r.get("wall_pct", 0) or 0))
        for r in bod.get("rows", [])
        if float(r.get("wall_pct", 0) or 0) >= cand_wall and r.get("kernel") not in fold
    ]


# ── plan canonical API ────────────────────────────────────────────────────────


def select(bod: dict, exhaustion: dict, cand_wall: float, k: int) -> list[Candidate]:
    """Rank the wall>=cand_wall BOD kernels into :class:`Candidate`s (wall-desc).

    Joins each kernel's tried/win counts + consecutive-dead from ``exhaustion``
    and marks ``state`` OPEN/EXHAUSTED. Does NOT drop EXHAUSTED kernels (it marks
    them); does NOT drop folded ones (that's :func:`is_exhausted`/:func:`gen_digest`).
    """
    out: list[Candidate] = []
    for r in bod.get("rows", []):
        if float(r.get("wall_pct", 0) or 0) < cand_wall:
            continue
        st = exhaustion.get(r["kernel"]) if exhaustion else None
        stats = st if isinstance(st, dict) else {}
        out.append(
            Candidate(
                kernel=r["kernel"],
                wall_pct=float(r.get("wall_pct", 0) or 0),
                mem_busy=r.get("mem_busy"),
                occ=r.get("occ"),
                vgpr=r.get("vgpr"),
                l2_hit=r.get("l2_hit", r.get("l2_hit_pct")),
                tried=int(stats.get("tried", 0) or 0),
                wins=int(stats.get("wins", 0) or 0),
                best_win_pct=stats.get("best_win_pct"),
                state="EXHAUSTED" if _dead_count(st) >= k else "OPEN",
            )
        )
    out.sort(key=lambda c: -c.wall_pct)
    return out


def is_exhausted(exhaustion: dict, bod: dict, cand_wall: float, k: int, folded) -> bool:
    """GLOBAL stop: there ARE live candidates and every one is at K deads.

    Ports ``v2/check_exhausted.py``'s ``allx``. Folded (already-banked) kernels
    are excluded from the candidate set. No candidates ⇒ ``False`` (not "done").
    """
    cands = _live_candidates(bod, cand_wall, folded)
    return bool(cands) and all(_dead_count(exhaustion.get(kern)) >= k for kern, _ in cands)


def gen_digest(exhaustion: dict, bod: dict, cand_wall: float, k: int, folded, seed: str = "") -> str:
    """Render the codex-facing COVERAGE DIGEST (ports ``v2/gen_digest.py``).

    Active kernels (fewest deads) first, then exhausted; each line carries wall%,
    ``N/K`` dead, and the last-6 tried ``lever->verdict`` pairs (from the
    exhaustion store's ``levers``). ``seed`` (opt-in, arch-CORRECT) is prepended.
    """
    exhaustion = exhaustion or {}
    cands = _live_candidates(bod, cand_wall, folded)
    lines = [
        "COVERAGE DIGEST -- pick candidates this round, preferring those with the FEWEST attempts; "
        "NEVER re-try a lever already marked DEAD/INCONCLUSIVE on its kernel; SKIP every [EXHAUSTED] kernel:"
    ]
    # active first (fewest dead), then exhausted; ties broken by wall descending.
    cands.sort(key=lambda x: (_dead_count(exhaustion.get(x[0])) >= k, _dead_count(exhaustion.get(x[0])), -x[1]))
    for kern, wall in cands:
        n = _dead_count(exhaustion.get(kern))
        status = "EXHAUSTED" if n >= k else "ACTIVE"
        levers = _tried_levers(exhaustion.get(kern))
        tl = ", ".join(f"{lv}->{vd}" for lv, vd in levers[-6:]) or "(none tried)"
        lines.append(f"  [{status}] {kern} ({wall:.1f}% wall, {n}/{k} dead) tried: {tl}")
    body = "\n".join(lines)
    return (seed.strip() + "\n\n" + body) if seed.strip() else body


def update_exhaustion(exhaustion_path: str, round: int, repo: str, arch: str = "gfx1201") -> None:
    """Fold a round's ledger verdicts into per-kernel consecutive-dead counters.

    Ports ``v2/update_exhaustion.py``. Scans ``<repo>/autoresearch/ledger/
    swarm_<arch>_*.jsonl`` for rows whose ``label`` starts with ``R<round>c``
    (so a per-arch loop only counts its OWN round's verdicts), then per kernel:
    a WIN resets to 0; a DEAD/COHERENCE_FAIL/LOSS/NOISE/DEAD_FILE/NO_OP advances
    by 1 (round-capped); INCONCLUSIVE / infra fails do not count. Writes the
    dict-form store (``{kernel: {"consecutive_dead": n, ...}}``), preserving any
    other per-kernel fields already present.
    """
    exh = _load_store(exhaustion_path)
    prefix = f"R{round}c"
    byk: dict[str, list] = {}
    for f in sorted(glob.glob(os.path.join(repo, "autoresearch", "ledger", f"swarm_{arch}_*.jsonl"))):
        with open(f, errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if str(d.get("label", "")).startswith(prefix):
                    byk.setdefault(d.get("kernel"), []).append(d.get("verdict"))
    for kern, verds in byk.items():
        if not kern:
            continue
        entry = exh.get(kern)
        entry = dict(entry) if isinstance(entry, dict) else {"consecutive_dead": _dead_count(entry)}
        if WIN in verds:
            entry["consecutive_dead"] = 0
        elif any(v in _LEDGER_DEAD for v in verds):
            entry["consecutive_dead"] = _dead_count(entry) + 1  # round-capped at +1
        exh[kern] = entry
    with open(exhaustion_path, "w") as fh:
        json.dump(exh, fh)


def _load_store(path: str) -> dict:
    if path and os.path.exists(path):
        try:
            data = json.load(open(path))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


# ── ported per-round exhaustion primitives (from exhaustion.py) ───────────────
# State form: {kernel: {"dead": int, "infra": int, "parity": int}}.


def _st(state, kernel):
    return state.setdefault(kernel, {"dead": 0, "infra": 0, "parity": 0})


def apply_round(state, round_verdicts, parity_repeat=3):
    """Advance exhaustion state by one round.

    Returns ``(state, needs_confirmation)`` — kernels with an INCONCLUSIVE this
    round and no dead-progress, to route to a higher-sample re-measure. The dead
    streak advances by AT MOST 1 per round (K = consecutive ROUNDS).
    """
    needs = set()
    for kernel, verdicts in round_verdicts.items():
        if not kernel:
            continue
        st = _st(state, kernel)
        if WIN in verdicts:
            st["dead"] = 0
            st["parity"] = 0
            continue
        st["infra"] += sum(1 for v in verdicts if v in INFRA_VERDICTS)
        st["parity"] += sum(1 for v in verdicts if v == PARITY_FAIL)
        hard_dead = any(v in DEAD_VERDICTS for v in verdicts)
        parity_dead = (PARITY_FAIL in verdicts) and st["parity"] >= parity_repeat
        if hard_dead or parity_dead:
            st["dead"] += 1  # round-capped at +1
        elif INCONCLUSIVE in verdicts:
            needs.add(kernel)  # real-but-small -> re-measure, do NOT count dead
    return state, needs


def kernel_exhausted(state, kernel, K, infra_cap=8):
    """Per-kernel exhaustion (dead streak >= K OR infra fails >= cap)."""
    st = state.get(kernel, {})
    return st.get("dead", 0) >= K or st.get("infra", 0) >= infra_cap


def all_exhausted(state, candidates, K, infra_cap=8):
    """True iff there ARE candidates and every one is exhausted (global stop)."""
    return bool(candidates) and all(kernel_exhausted(state, k, K, infra_cap) for k in candidates)


def dead_streak(state, kernel):
    return state.get(kernel, {}).get("dead", 0)
