# Copyright (c) Kaden Schutt
"""ar CLI — role-scoped operator/agent entrypoint with mechanical bounds.

No-GPU unit tests. The CLI is the ONLY surface an agent touches (ssh-in →
``ar``, never a raw script). Two contracts are enforced here mechanically, not
by prompt-discipline:

1. **Role scoping.** The ``agent`` role may ONLY call ``why/status/bod/certify``;
   any operator-only verb (``start/stop/ingest/fold/rollover/config``) is refused
   with exit code 3.
2. **Certify bounds.** An agent ``certify`` on an EXHAUSTED / off-target
   (below-cand-wall / non-census) / over-budget kernel is refused with exit 3 —
   the leash is in the tool, so a runaway agent cannot re-burn a dead kernel.

Fixtures are seeded from the real ledger/BOD shape (see ``bod_gfx1201.json`` +
``mini_ledger``); ``tmp_ar`` builds a throwaway ``ar.db`` and points the CLI at
it via ``AR_DB``.
"""
import json

import pytest

from autoresearch.ar.cli import main
from autoresearch.ar.db import connect


@pytest.fixture
def tmp_ar(tmp_path, monkeypatch):
    """A throwaway ``ar.db`` with one EXHAUSTED candidate, one OPEN candidate, and
    an empty runs table; ``AR_DB`` points the CLI at it."""
    db_path = tmp_path / "ar.db"
    conn = connect(str(db_path))
    # An EXHAUSTED census candidate: 5 consecutive DEAD, no WIN (streak >= k).
    for i in range(5):
        conn.execute(
            "INSERT INTO attempts(arch,kernel,lever,verdict,tok_delta,dur_delta,measurement_hash,ts) "
            "VALUES(?,?,?,?,?,?,?,?)",
            ("gfx1201", "gate_up_exhausted", f"L{i}", "DEAD", -0.5, 0.1, f"exh{i:013d}", 1720500000 + i),
        )
    # Both kernels are BOD census candidates (wall% >= cand_wall).
    conn.execute(
        "INSERT INTO bod(arch,kernel,wall_pct,l2_hit,mem_busy,occ,vgpr,snap_ts) VALUES(?,?,?,?,?,?,?,?)",
        ("gfx1201", "gate_up_exhausted", 12.8, 64.4, 54.7, 51.8, 96, 1720500000),
    )
    conn.execute(
        "INSERT INTO bod(arch,kernel,wall_pct,l2_hit,mem_busy,occ,vgpr,snap_ts) VALUES(?,?,?,?,?,?,?,?)",
        ("gfx1201", "gate_up_open", 14.7, 77.6, 41.4, 34.9, 72, 1720500000),
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("AR_DB", str(db_path))
    return str(db_path)


# ── plan Task 7.1 canonical tests ─────────────────────────────────────────────


def test_agent_certify_on_exhausted_exits_3(monkeypatch, tmp_ar):
    # kernel marked EXHAUSTED in fixture db
    rc = main(
        ["--role", "agent", "certify", "--arch", "gfx1201", "--kernel", "gate_up_exhausted",
         "--lever", "L", "--variant", "/tmp/v.hip"]
    )
    assert rc == 3


def test_agent_cannot_start():
    rc = main(["--role", "agent", "start", "--config", "x.toml"])
    assert rc == 3  # start is operator-only


# ── mechanical-bounds coverage (the leash) ────────────────────────────────────


def test_agent_certify_off_target_exits_3(tmp_ar):
    # a kernel that is NOT a census candidate (absent from BOD) => OFF_TARGET.
    rc = main(
        ["--role", "agent", "certify", "--arch", "gfx1201", "--kernel", "not_a_candidate_xyz",
         "--lever", "L", "--variant", "/tmp/v.hip"]
    )
    assert rc == 3


def test_agent_certify_open_candidate_accepted(tmp_ar, capsys):
    # an OPEN candidate (in BOD, wall>=cand_wall, no dead streak, no over-budget run) => accepted.
    rc = main(
        ["--role", "agent", "certify", "--arch", "gfx1201", "--kernel", "gate_up_open",
         "--lever", "reg_kpack", "--variant", "/tmp/v.hip"]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["accepted"] is True


def test_agent_certify_over_budget_exits_3(tmp_ar, capsys):
    import sqlite3

    conn = sqlite3.connect(tmp_ar)
    conn.execute(
        "INSERT INTO runs(id,arch,model,card,status,budget,calls,ttl,pid,ts) "
        "VALUES('r1','gfx1201','m',1,'running',400,400,99999,NULL,1720500000)"
    )
    conn.commit()
    conn.close()
    rc = main(
        ["--role", "agent", "certify", "--arch", "gfx1201", "--kernel", "gate_up_open",
         "--lever", "L", "--variant", "/tmp/v.hip"]
    )
    assert rc == 3
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["reason"] == "BUDGET_SPENT"


# ── role scoping (agent allowed verbs; operator-only refusals) ─────────────────


def test_agent_bod_allowed(tmp_ar):
    rc = main(["--role", "agent", "bod", "--arch", "gfx1201", "--json"])
    assert rc == 0


def test_agent_why_allowed(tmp_ar):
    rc = main(["--role", "agent", "why", "gate_up_exhausted", "--arch", "gfx1201", "--json"])
    assert rc == 0


def test_agent_status_allowed(tmp_ar):
    rc = main(["--role", "agent", "status", "--json"])
    assert rc == 0


@pytest.mark.parametrize("verb_argv", [
    ["ingest"],
    ["fold", "--ref", "refs/heads/loop/gfx1201", "--sha", "deadbeef"],
    ["rollover"],
    ["stop"],
    ["config", "--arch", "gfx1201"],
])
def test_agent_operator_verbs_refused(tmp_ar, verb_argv, capsys):
    rc = main(["--role", "agent", *verb_argv])
    assert rc == 3
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["reason"] == "ROLE_FORBIDDEN"


# ── operator role: the no-GPU verbs work end-to-end ───────────────────────────


def test_operator_ingest_rebuilds_db(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "ar.db"
    monkeypatch.setenv("AR_DB", str(db_path))
    rc = main([
        "--role", "operator", "ingest",
        "--ledger", "autoresearch/ar/tests/fixtures/mini_ledger",
        "--bod", "autoresearch/ar/tests/fixtures/bod_gfx1201.json",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["ingested"] >= 1
    # the db really has the ingested attempts
    conn = connect(str(db_path))
    assert conn.execute("SELECT count(*) FROM attempts").fetchone()[0] == out["ingested"]


def test_operator_config_prints_resolved(capsys, loop_toml):
    rc = main(["--role", "operator", "config", "--config", loop_toml, "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())  # config --json is indented multi-line
    assert out["arch"] == "gfx1201" and out["model"].endswith("mq4r")
    assert [w["model"] for w in out["workers"]] == ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]


def test_operator_certify_bounds_apply_regardless_of_role(tmp_ar):
    # the leash is mechanical: even an operator cannot certify an EXHAUSTED kernel.
    rc = main(
        ["--role", "operator", "certify", "--arch", "gfx1201", "--kernel", "gate_up_exhausted",
         "--lever", "L", "--variant", "/tmp/v.hip"]
    )
    assert rc == 3


def test_default_role_is_operator(tmp_ar):
    # no --role => operator (can call an operator-only verb without refusal).
    rc = main(["config", "--config", "autoresearch/config/loop_gfx1201.toml", "--json"])
    assert rc == 0
