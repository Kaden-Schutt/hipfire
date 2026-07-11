# Copyright (c) Kaden Schutt
"""No-GPU unit tests for ar.candidates — selection + exhaustion + digest.

Ports ``harness/exhaustion.py`` + the v2 trio (``check_exhausted`` /
``gen_digest`` / ``update_exhaustion``) onto one module. Covers the plan's
canonical API (``select`` / global ``is_exhausted`` / ``gen_digest`` /
``update_exhaustion``) AND the ported per-round exhaustion primitives
(``apply_round`` / ``kernel_exhausted`` / ``all_exhausted`` / ``dead_streak``).
"""
import json

from autoresearch.ar import candidates as C
from autoresearch.ar.candidates import (
    Candidate,
    gen_digest,
    is_exhausted,
    select,
    update_exhaustion,
)

BOD = {
    "rows": [
        {"kernel": "k1", "wall_pct": 5.0, "mem_busy": 50, "occ": 40, "vgpr": 88, "l2_hit": 60},
        {"kernel": "low", "wall_pct": 1.0, "mem_busy": 0, "occ": 0, "vgpr": 0, "l2_hit": 0},
    ]
}


# ── plan Task 3.2 Step 1 (the new contract) ───────────────────────────────────
def test_below_cand_wall_excluded():
    cands = select(BOD, exhaustion={}, cand_wall=3.0, k=5)
    assert [c.kernel for c in cands] == ["k1"]  # 'low' below 3.0 wall dropped


def test_exhausted_after_k_deads():
    assert is_exhausted({"k1": {"consecutive_dead": 5}}, BOD, 3.0, 5, folded=[]) is True


def test_open_below_k():
    assert is_exhausted({"k1": {"consecutive_dead": 4}}, BOD, 3.0, 5, folded=[]) is False


# ── select marks state + carries roofline/tried/win counts ────────────────────
def test_select_marks_state_and_carries_stats():
    exh = {"k1": {"consecutive_dead": 5, "tried": 7, "wins": 1, "best_win_pct": 2.1}}
    (c,) = select(BOD, exh, cand_wall=3.0, k=5)
    assert isinstance(c, Candidate)
    assert c.state == "EXHAUSTED"
    assert c.tried == 7 and c.wins == 1 and c.best_win_pct == 2.1
    assert c.mem_busy == 50 and c.occ == 40 and c.vgpr == 88 and c.l2_hit == 60


def test_select_open_when_below_k():
    (c,) = select(BOD, {"k1": {"consecutive_dead": 4}}, cand_wall=3.0, k=5)
    assert c.state == "OPEN"


def test_select_accepts_int_exhaustion_and_l2_hit_pct_alias():
    bod = {"rows": [{"kernel": "k1", "wall_pct": 9.0, "l2_hit_pct": 77.6}]}
    (c,) = select(bod, {"k1": 5}, cand_wall=3.0, k=5)  # legacy int form
    assert c.state == "EXHAUSTED" and c.l2_hit == 77.6


def test_is_exhausted_folded_excluded_and_empty_is_false():
    # k1 exhausted but folded out => no live candidates => not "all exhausted"
    assert is_exhausted({"k1": {"consecutive_dead": 5}}, BOD, 3.0, 5, folded=["k1"]) is False
    assert is_exhausted({}, {"rows": []}, 3.0, 5, folded=[]) is False  # no candidates


# ── gen_digest (the codex-facing tried-lever text) ────────────────────────────
def test_gen_digest_marks_active_and_exhausted():
    bod = {"rows": [{"kernel": "hot", "wall_pct": 10.0}, {"kernel": "done", "wall_pct": 5.0}]}
    exh = {"done": {"consecutive_dead": 5, "levers": [("lds_stage2", "DEAD")]}}
    txt = gen_digest(exh, bod, cand_wall=3.0, k=5, folded=[])
    assert "[EXHAUSTED] done" in txt
    assert "[ACTIVE] hot" in txt
    assert "lds_stage2->DEAD" in txt
    assert "(none tried)" in txt  # 'hot' has no tried levers
    # active kernels come before exhausted ones
    assert txt.index("[ACTIVE] hot") < txt.index("[EXHAUSTED] done")


# ── update_exhaustion (ledger -> per-kernel consecutive-dead counters) ─────────
def test_update_exhaustion_counts_dead_win_resets_and_round_scopes(tmp_path):
    ledger = tmp_path / "autoresearch" / "ledger"
    ledger.mkdir(parents=True)
    (ledger / "swarm_gfx1201_x.jsonl").write_text(
        '{"label":"R1c0_a","kernel":"k1","verdict":"DEAD"}\n'
        '{"label":"R1c0_b","kernel":"k1","verdict":"DEAD"}\n'  # 2 deads, ONE round -> +1
        '{"label":"R1c1_c","kernel":"k2","verdict":"WIN"}\n'
        '{"label":"R1c2_d","kernel":"k3","verdict":"INCONCLUSIVE"}\n'  # does NOT count
        '{"label":"R9c9_z","kernel":"k1","verdict":"DEAD"}\n'  # other round -> ignored
    )
    exh_path = str(tmp_path / "exh.json")
    json.dump({"k1": {"consecutive_dead": 2}, "k2": {"consecutive_dead": 3}}, open(exh_path, "w"))
    update_exhaustion(exh_path, 1, str(tmp_path), "gfx1201")
    got = json.load(open(exh_path))
    assert got["k1"]["consecutive_dead"] == 3  # 2 + 1 (round-capped), R9 ignored
    assert got["k2"]["consecutive_dead"] == 0  # WIN resets
    assert got.get("k3", {}).get("consecutive_dead", 0) == 0  # INCONCLUSIVE not counted


# ── ported per-round exhaustion primitives (from harness/test_exhaustion.py) ───
def test_coherence_fail_counts_toward_exhaustion():
    state = {}
    for _ in range(5):
        C.apply_round(state, {"k": ["COHERENCE_FAIL"]})
    assert C.kernel_exhausted(state, "k", K=5)


def test_win_resets_streak():
    state = {}
    for _ in range(4):
        C.apply_round(state, {"k": ["DEAD"]})
    C.apply_round(state, {"k": ["WIN"]})
    assert C.dead_streak(state, "k") == 0
    assert not C.kernel_exhausted(state, "k", K=5)


def test_inconclusive_does_not_increment_but_queues():
    state = {}
    needs = set()
    for _ in range(10):
        _, n = C.apply_round(state, {"k": ["INCONCLUSIVE"]})
        needs |= n
    assert C.dead_streak(state, "k") == 0
    assert not C.kernel_exhausted(state, "k", K=5)
    assert "k" in needs


def test_per_round_cap_is_one():
    state = {}
    C.apply_round(state, {"k": ["DEAD", "DEAD", "DEAD"]})
    assert C.dead_streak(state, "k") == 1


def test_parity_fail_counts_only_when_repeated():
    state = {}
    for _ in range(2):
        C.apply_round(state, {"k": ["PARITY_FAIL"]})
    assert C.dead_streak(state, "k") == 0
    C.apply_round(state, {"k": ["PARITY_FAIL"]})
    assert C.dead_streak(state, "k") == 1


def test_infra_uses_separate_cap_not_dead_streak():
    state = {}
    for _ in range(8):
        C.apply_round(state, {"k": ["VARIANT_BUILD_FAIL"]})
    assert C.dead_streak(state, "k") == 0
    assert C.kernel_exhausted(state, "k", K=5, infra_cap=8)


def test_all_exhausted_global_stop():
    state = {}
    for _ in range(5):
        C.apply_round(state, {"a": ["DEAD"], "b": ["DEAD"]})
    assert C.all_exhausted(state, ["a", "b"], K=5)
    assert not C.all_exhausted(state, ["a", "b", "c"], K=5)  # c untouched
    assert not C.all_exhausted(state, [], K=5)  # no candidates
