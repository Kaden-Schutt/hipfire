# Copyright (c) Kaden Schutt
"""No-GPU unit tests for ar.census — the PURE rocprof CSV parse.

The GPU seam (``run_census`` spawning the daemon under ``profile_standard`` +
``rocprofv3``) is exercised live only; here we pin the pure ``parse_rocprof``
against a captured rocprofv3 counter_collection CSV fixture (derived from the
real ``state/bod_gfx1201.json`` census).
"""
import os

from autoresearch.ar.census import parse_rocprof, write_bod

_CSV = "autoresearch/ar/tests/fixtures/rocprof_gfx1201.csv"


def test_parse_rocprof_to_bod_rows():
    rows = parse_rocprof(_CSV)
    top = max(rows, key=lambda r: r["wall_pct"])
    assert top["kernel"].startswith("fused_qkvza") and top["wall_pct"] > 10 and 0 <= top["occ"] <= 100


def test_parse_strips_signature_and_counts_dispatches():
    rows = parse_rocprof(_CSV)
    top = next(r for r in rows if r["kernel"].startswith("fused_qkvza"))
    # the "(...)" signature is stripped off the kernel_name
    assert "(" not in top["kernel"]
    # fused_qkvza has TWO dispatches in the fixture -> n == 2 (dedup by dispatch_id)
    assert top["n"] == 2


def test_parse_l2_ratio_and_pct_averaging():
    rows = parse_rocprof(_CSV)
    top = next(r for r in rows if r["kernel"].startswith("fused_qkvza"))
    # GL2C_HIT/(HIT+MISS) summed across both dispatches -> 77.6%
    assert abs(top["l2_hit_pct"] - 77.6) < 0.2
    # MemUnitBusy / OccupancyPercent are per-dispatch percentages -> averaged, not summed
    assert abs(top["mem_busy"] - 41.4) < 0.2
    assert 0 <= top["occ"] <= 100


def test_parse_derives_roofline_lens():
    rows = parse_rocprof(_CSV)
    lens = {r["kernel"]: r["roofline"] for r in rows}
    # occ 5.7 < 15 -> latency/occ-starved
    assert lens["gemv_hfq4g256_residual_sigmoid_scaled_gpu"] == "latency/occ-starved"
    # l2>=70 & mem>40 -> L2-resident/mem-busy
    assert lens["fused_qkvza_hfq4g256"] == "L2-resident/mem-busy"


def test_write_bod_round_trips_shape(tmp_path):
    import json

    rows = parse_rocprof(_CSV)
    bod = {"arch": "gfx1201", "model": "qwen3.6-35b-a3b.mq4r", "rows": rows}
    out = tmp_path / "bod_gfx1201.json"
    write_bod(bod, str(out))
    got = json.load(open(out))
    assert got["arch"] == "gfx1201"
    assert got["rows"][0]["kernel"] == max(rows, key=lambda r: r["wall_pct"])["kernel"]
