# Copyright (c) Kaden Schutt
from autoresearch.ar.db import connect, ingest, kernel_stats


def test_ingest_idempotent(tmp_path):
    db = tmp_path / "ar.db"
    c = connect(str(db))
    n1 = ingest(c, "autoresearch/ar/tests/fixtures/mini_ledger", "autoresearch/ar/tests/fixtures/bod_gfx1201.json")
    n2 = ingest(c, "autoresearch/ar/tests/fixtures/mini_ledger", "autoresearch/ar/tests/fixtures/bod_gfx1201.json")
    assert n1 > 0 and n2 == n1  # second ingest adds nothing new
    assert c.execute("SELECT count(*) FROM attempts").fetchone()[0] == n1


def test_kernel_stats_counts(tmp_path):
    c = connect(str(tmp_path / "ar.db"))
    ingest(c, "autoresearch/ar/tests/fixtures/mini_ledger", "autoresearch/ar/tests/fixtures/bod_gfx1201.json")
    s = kernel_stats(c, "gfx1201", "fused_qkvza_hfq4g256", k=5)
    assert s["tried"] >= 1 and "consecutive_dead" in s
