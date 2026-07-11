# Copyright (c) Kaden Schutt
"""ar.census — the BOD (Book-of-Decode) census for the autoresearch loop.

Ports ``autoresearch/harness/oracle_profile.sh`` and splits it into two seams:

* :func:`run_census` — the **GPU** seam (spawns the baseline daemon under
  ``profile_standard`` and runs ``rocprofv3 --pmc``). Not exercised in no-GPU
  CI; driven live on-device.
* :func:`parse_rocprof` — the **pure** CSV→rows parse. Unit-tested against a
  captured rocprofv3 counter_collection CSV fixture, so all the attribution
  logic (per-dispatch wall%, PCT-counter averaging, GL2C L2-hit ratio, the
  derived roofline lens) stays no-GPU-testable.

The census counter set (``GL2C_HIT``/``GL2C_MISS``/``MemUnitBusy``/
``OccupancyPercent``/``SQ_BUSY_CYCLES`` universal; ``SQ_INST_CYCLES_VALU`` +
``GL2C_EA_RDREQ`` gfx1201-only) revives under ``profile_standard`` on all five
RDNA gens (the RDNA-PMC gating rule). rocprofv3's counter_collection CSV carries
BOTH per-dispatch ``Start_Timestamp``/``End_Timestamp`` AND the counter rows, so
a single CSV yields wall% + census ratios — no separate kernel-trace pass.

``bod`` shape (matches ``state/bod_<arch>.json``)::

    {"arch", "model", "kt_dispatches", "census_kernels",
     "rows": [{"kernel","wall_pct","n","l2_hit_pct","mem_busy","occ",
               "dram_miss","valu_cyc","vgpr","accum_vgpr","sgpr","lds",
               "scratch","roofline"}, ...]}
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import os
import re
import subprocess

# MemUnitBusy / OccupancyPercent are per-dispatch percentages -> AVERAGE across
# dispatches; GL2C_*/SQ_* are raw accumulators -> SUM. (Verbatim from
# oracle_profile.sh.)
_PCT = {"MemUnitBusy", "OccupancyPercent"}
_TOP_N = 12


def _col(header, want: str):
    """Case-insensitive column lookup (rocprofv3 capitalizes header names)."""
    for k in header:
        if k and k.strip().lower() == want:
            return k
    return None


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _i(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _clean_kernel(name: str) -> str:
    """Strip the ``(args...)`` signature + surrounding whitespace off a kernel
    name (``fused_qkvza_hfq4g256(float*, int)`` -> ``fused_qkvza_hfq4g256``)."""
    return re.sub(r"\(.*", "", name or "").strip()


def _roofline(l2, mem, occ):
    """Derived roofline lens (query-time hint only — data-not-tags).

    Priority: occupancy-starved > DRAM-traffic > L2-resident > mem-active. Ported
    verbatim from oracle_profile.sh's ``lens()``.
    """
    if mem is None and occ is None:
        return "unmeasured"
    if occ is not None and occ < 15:
        return "latency/occ-starved"
    if l2 is not None and l2 < 45 and (mem or 0) > 25:
        return "DRAM-traffic-bound"
    if l2 is not None and l2 >= 70 and (mem or 0) > 40:
        return "L2-resident/mem-busy"
    if (mem or 0) > 25:
        return "mem-active(mixed)"
    return "compute/other"


def parse_rocprof(csv_path: str) -> list[dict]:
    """Parse a rocprofv3 counter_collection CSV into ranked BOD rows (pure).

    Per-dispatch wall time is summed per kernel (deduped by ``Dispatch_Id`` so
    the per-counter row fan-out doesn't multi-count), then normalized to
    ``wall_pct``. Counters are averaged (percentages) or summed (accumulators)
    per kernel; ``vgpr``/``sgpr``/``lds``/``scratch`` are taken from the first
    dispatch seen. Returns the top ``12`` kernels by wall%, wall-descending.
    """
    try:
        with open(csv_path, newline="", errors="ignore") as fh:
            reader = csv.DictReader(fh)
            data = list(reader)
    except OSError:
        return []
    if not data:
        return []

    header = list(data[0].keys())
    c_kn = _col(header, "kernel_name")
    c_did = _col(header, "dispatch_id")
    c_cn = _col(header, "counter_name")
    c_cv = _col(header, "counter_value")
    c_st = _col(header, "start_timestamp")
    c_en = _col(header, "end_timestamp")
    c_vg = _col(header, "vgpr_count")
    c_av = _col(header, "accum_vgpr_count")
    c_sg = _col(header, "sgpr_count")
    c_ld = _col(header, "lds_block_size")
    c_sc = _col(header, "scratch_size")

    wall = collections.defaultdict(float)
    n_disp = collections.defaultdict(int)
    # kernel -> counter_name -> [sum, count]
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
    meta: dict[str, dict] = {}
    seen_disp: set = set()

    for row_i, r in enumerate(data):
        n = _clean_kernel(r.get(c_kn, "") if c_kn else "")
        if not n:
            continue
        # duration: count each dispatch ONCE (the counter rows repeat it).
        did = r.get(c_did) if c_did else None
        disp_key = did if did not in (None, "") else f"__row{row_i}"
        if disp_key not in seen_disp:
            seen_disp.add(disp_key)
            dur = 0.0
            if c_st and c_en:
                dur = _f(r.get(c_en)) - _f(r.get(c_st))
                if dur < 0:
                    dur = 0.0
            wall[n] += dur
            n_disp[n] += 1
        # counters accumulate across every row.
        if c_cn and c_cv:
            cn = r.get(c_cn)
            if cn:
                acc[n][cn][0] += _f(r.get(c_cv))
                acc[n][cn][1] += 1
        # static resources: first dispatch of the kernel wins.
        if c_vg and n not in meta:
            meta[n] = {
                "vgpr": _i(r.get(c_vg)),
                "accum_vgpr": _i(r.get(c_av)) if c_av else None,
                "sgpr": _i(r.get(c_sg)) if c_sg else None,
                "lds": _i(r.get(c_ld)) if c_ld else None,
                "scratch": _i(r.get(c_sc)) if c_sc else None,
            }

    tot = sum(wall.values())

    def g(kernel, counter):
        e = acc[kernel].get(counter)
        if not e:
            return None
        return (e[0] / e[1]) if (counter in _PCT and e[1]) else e[0]

    rows: list[dict] = []
    for n, d in sorted(wall.items(), key=lambda x: -x[1])[:_TOP_N]:
        hit = g(n, "GL2C_HIT")
        miss = g(n, "GL2C_MISS")
        l2 = (100.0 * hit / (hit + miss)) if (hit is not None and miss is not None and (hit + miss) > 0) else None
        mem = g(n, "MemUnitBusy")
        occ = g(n, "OccupancyPercent")
        m = meta.get(n, {})
        rows.append(
            {
                "kernel": n[:60],
                "wall_pct": round(100 * d / tot, 1) if tot else 0.0,
                "n": n_disp[n],
                "l2_hit_pct": round(l2, 1) if l2 is not None else None,
                "mem_busy": round(mem, 1) if mem is not None else None,
                "occ": round(occ, 1) if occ is not None else None,
                "dram_miss": miss,
                "valu_cyc": g(n, "SQ_INST_CYCLES_VALU"),
                "vgpr": m.get("vgpr"),
                "accum_vgpr": m.get("accum_vgpr"),
                "sgpr": m.get("sgpr"),
                "lds": m.get("lds"),
                "scratch": m.get("scratch"),
                "roofline": _roofline(l2, mem, occ),
            }
        )
    return rows


def write_bod(bod: dict, path: str) -> None:
    """Serialize a census dict to ``bod_<arch>.json`` (matches the on-disk shape)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(bod, fh)


# ── GPU seam (live-only; not exercised in no-GPU CI) ──────────────────────────

# universal counters (revive on all gens) + gfx1201-only VALU cycles / EA read
# requests. rocprofv3 errors on an unknown counter, so the extras go on RDNA4 only.
_PMC_UNIVERSAL = ["GL2C_HIT", "GL2C_MISS", "MemUnitBusy", "OccupancyPercent", "SQ_BUSY_CYCLES"]
_PMC_GFX1201 = ["SQ_INST_CYCLES_VALU", "GL2C_EA_RDREQ"]

_CENSUS_REQ = (
    '{{"type":"load","model":"{model}","params":{{"max_seq":2048,"kv_mode":"q8"}}}}\n'
    '{{"type":"generate","id":"r","prompt":"Explain how a hash map resolves collisions, '
    'in two sentences.","temperature":0.0,"max_tokens":{maxg}}}\n'
    '{{"type":"unload"}}\n'
)


def _pmc_for(arch: str) -> list[str]:
    pmc = list(_PMC_UNIVERSAL)
    if arch == "gfx1201":
        pmc += _PMC_GFX1201
    return pmc


def _find_census_csv(out_dir: str) -> str | None:
    for pat in (
        os.path.join(out_dir, "**", "*counter_collection*.csv"),
        os.path.join(out_dir, "**", "*.csv"),
    ):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def run_census(
    arch: str,
    dev: int,
    drm: int,
    model: str,
    layers: int,
    repo: str,
    *,
    rocprof: str = "/opt/rocm/bin/rocprofv3",
    daemon_bin: str = "./target/release/examples/daemon",
    out_dir: str | None = None,
    runner=subprocess.run,
) -> dict:
    """Run the GPU census for ``arch`` and return the BOD dict.

    Spawns the baseline daemon under ``profile_standard`` on DRM ``drm`` / HIP
    ``dev`` and drives ``rocprofv3 --pmc`` over a short fixed decode of ``layers``
    tokens, then hands the resulting counter CSV to :func:`parse_rocprof`. This is
    the GPU seam (``runner`` is injectable for tests; the perfmon pins clocks low
    per the RDNA-PMC gating rule, so tok/s is not measured here). Returns
    ``{"arch","model","kt_dispatches","census_kernels","rows",...}``.
    """
    model_path = model
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        # accept a bare SKU name; the caller resolves the real path via MODELS_DIR
        model_path = os.environ.get("HIPFIRE_MODEL_PATH", model)
    if not os.path.exists(model_path):
        return {"arch": arch, "model": os.path.basename(model_path), "error": f"model absent: {model_path}", "rows": []}

    out_dir = out_dir or f"/tmp/pmc-{arch}"
    os.makedirs(out_dir, exist_ok=True)
    req = _CENSUS_REQ.format(model=model_path, maxg=int(layers))
    req_file = os.path.join(out_dir, f"req-{arch}.jsonl")
    with open(req_file, "w") as fh:
        fh.write(req)

    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(dev))
    pl = f"/sys/class/drm/card{drm}/device/power_dpm_force_performance_level"

    def _set_dpm(level: str) -> None:
        try:
            runner(f"echo {level} | sudo -n tee {pl} >/dev/null", shell=True, cwd=repo, env=env, timeout=30)
        except Exception:
            pass

    # warm the JIT cache (throwaway), then census under profile_standard.
    _set_dpm("profile_standard")
    pmc = " ".join(_pmc_for(arch))
    runner(
        [
            rocprof, "-f", "csv", "--pmc", *_pmc_for(arch),
            "-d", out_dir, "--",
            "bash", "-c", f"{daemon_bin} < {req_file} >/dev/null 2>&1",
        ],
        cwd=repo,
        env=env,
        timeout=900,
        check=False,
    )
    _set_dpm("auto")

    csv_path = _find_census_csv(out_dir)
    rows = parse_rocprof(csv_path) if csv_path else []
    return {
        "arch": arch,
        "model": os.path.basename(model_path),
        "kt_dispatches": sum(r.get("n", 0) for r in rows),
        "census_kernels": len(rows),
        "rows": rows,
    }
