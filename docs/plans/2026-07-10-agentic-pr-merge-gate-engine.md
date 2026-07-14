# Agentic PR Merge-Gate — `ar gate` Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the no-GPU-unit-testable core of the Tier-3 PR gate — a new
`autoresearch.ar.gate` package + `ar gate` CLI verb that, given base/head daemon
identities and an injected `ServeRunner`, renders a per-arch PASS/REJECT verdict
(fit → cross-arch → parity → perf → coherence) by reusing the existing
`certify/` arms.

**Architecture:** A sibling package to `autoresearch/ar/certify/`. It reuses the
pure arms (`verdict.parity_result` / `perf_result` / `coherence_result`,
`perf.dominance_f`, `cross_arch.check_cross_arch`) and the `ServeRunner` seam, but
adds gate-specific orchestration: unlike the loop's `certify()` (which wants a
WIN and short-circuits on perf-neutral), the gate wants **not-a-regression** and
must run coherence even on a perf-neutral PR. The perf rule is the exact mirror
of the loop's WIN gate. Everything is dependency-injected (runner factory,
cross-arch fn) so the whole engine is testable with no GPU.

**Tech Stack:** Python 3.11+ (stdlib `tomllib`, `dataclasses`, `statistics`,
`argparse`); pytest; no third-party deps. Reuses `autoresearch/ar/certify/`.

## Global Constraints

- **No-GPU unit-testable** — the entire engine runs under `scripts/no-gpu-ci.sh`
  with a mock `ServeRunner` and an injected cross-arch fn; no GPU, no ROCm, no
  `hipcc` in the unit path.
- **Reuse the arms, don't reimplement** — parity/perf/coherence decisions come
  from `autoresearch/ar/certify/{verdict,perf}.py`; the gate adds orchestration
  only.
- **Perf rule = mirror of the loop's WIN gate** — constants `perf.FLOOR = 0.15`,
  `perf.WIN_F = 0.90`, `perf.DEAD_F = 0.65`, `alpha = 0.05`. REJECT a regression
  only when it is significant (dominance ≤ `DEAD_F`) on **both** axes and beyond
  `FLOOR`; NEUTRAL and IMPROVEMENT both PASS.
- **Confirmation rerun** — a REGRESSION cell is re-run once (fresh runner from the
  factory); only a replicated regression rejects (noise → PASS).
- **New files** carry `# Copyright (c) Kaden Schutt` as the first line.
- **Maintainers** (config, verbatim): `fivetide, unverbraucht, nwoolmer, Kaden-Schutt`.
- **Canonical models** (config, verbatim): `qwen3.6-27b` (dense), `qwen3.6-a3b`
  (MoE); `deepseek4` fits **gfx1151 only**.
- **Archs**: `gfx1100, gfx1151, gfx1201`.
- Commit messages end with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- The GPU execution path (`LiveServeRunner`, daemon builds, rerun on real
  hardware) and Gate 4 / workflow / perf-governance / staging-train are **later
  plans** — this plan stops at the no-GPU engine + the `--plan` CLI view.

## File Structure

- Create `autoresearch/config/pr_gate.toml` — the gate config (fit map,
  maintainers, thresholds).
- Create `autoresearch/ar/gate/__init__.py` — package marker + public re-exports.
- Create `autoresearch/ar/gate/config.py` — `GateConfig` + `load_gate_config`.
- Create `autoresearch/ar/gate/perf_policy.py` — `classify_perf` (the mirror rule).
- Create `autoresearch/ar/gate/engine.py` — `gate_cell` + `run_gate`.
- Modify `autoresearch/ar/cli.py` — add the `gate` subcommand + `cmd_gate`.
- Create tests: `autoresearch/ar/tests/test_gate_config.py`,
  `test_gate_perf_policy.py`, `test_gate_engine.py`, `test_gate_cli.py`.

---

### Task 1: Gate config (`pr_gate.toml` + `GateConfig` loader + fit map)

**Files:**
- Create: `autoresearch/config/pr_gate.toml`
- Create: `autoresearch/ar/gate/__init__.py`
- Create: `autoresearch/ar/gate/config.py`
- Test: `autoresearch/ar/tests/test_gate_config.py`

**Interfaces:**
- Consumes: stdlib `tomllib`, `dataclasses` (mirrors `autoresearch/ar/config.py`).
- Produces:
  - `GateConfig(archs: list[str], canonical_models: list[str], fit: dict[str, list[str]], maintainers: list[str], floor: float, drift_pct: float, alpha: float)`
  - `GateConfig.fits(model: str, arch: str) -> bool`
  - `GateConfig.other_archs(arch: str) -> list[str]`
  - `GateConfig.models_for(arch: str, extra: tuple[str, ...] = ()) -> list[str]`
  - `load_gate_config(path: str) -> GateConfig`

- [ ] **Step 1: Write the failing test**

Create `autoresearch/ar/tests/test_gate_config.py`:

```python
# Copyright (c) Kaden Schutt
import os
import tempfile

from autoresearch.ar.gate.config import GateConfig, load_gate_config

_TOML = """
archs = ["gfx1100", "gfx1151", "gfx1201"]
canonical_models = ["qwen3.6-27b", "qwen3.6-a3b"]
maintainers = ["fivetide", "unverbraucht", "nwoolmer", "Kaden-Schutt"]
floor = 0.15
drift_pct = 3.0
alpha = 0.05

[fit]
"qwen3.6-27b" = ["gfx1100", "gfx1151", "gfx1201"]
"qwen3.6-a3b" = ["gfx1100", "gfx1151", "gfx1201"]
"deepseek4" = ["gfx1151"]
"""


def _write(tmp):
    p = os.path.join(tmp, "pr_gate.toml")
    with open(p, "w") as fh:
        fh.write(_TOML)
    return p


def test_load_and_fields():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_gate_config(_write(tmp))
    assert isinstance(cfg, GateConfig)
    assert cfg.canonical_models == ["qwen3.6-27b", "qwen3.6-a3b"]
    assert cfg.maintainers == ["fivetide", "unverbraucht", "nwoolmer", "Kaden-Schutt"]
    assert cfg.floor == 0.15 and cfg.drift_pct == 3.0 and cfg.alpha == 0.05


def test_fits_respects_map():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_gate_config(_write(tmp))
    assert cfg.fits("qwen3.6-27b", "gfx1100") is True
    assert cfg.fits("deepseek4", "gfx1151") is True
    assert cfg.fits("deepseek4", "gfx1100") is False        # DS4 does not fit 24 GB
    assert cfg.fits("unknown-sku", "gfx1201") is False       # unknown model -> not fit


def test_other_archs_excludes_self():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_gate_config(_write(tmp))
    assert cfg.other_archs("gfx1201") == ["gfx1100", "gfx1151"]


def test_models_for_filters_by_fit_and_adds_extra():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_gate_config(_write(tmp))
    # canonical only, gfx1100: DS4 excluded automatically
    assert cfg.models_for("gfx1100") == ["qwen3.6-27b", "qwen3.6-a3b"]
    # extra DS4 requested on gfx1151 -> included (fits); on gfx1100 -> dropped (no fit)
    assert cfg.models_for("gfx1151", extra=("deepseek4",)) == [
        "qwen3.6-27b", "qwen3.6-a3b", "deepseek4"]
    assert cfg.models_for("gfx1100", extra=("deepseek4",)) == [
        "qwen3.6-27b", "qwen3.6-a3b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest autoresearch/ar/tests/test_gate_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'autoresearch.ar.gate'`.

- [ ] **Step 3: Write the config file + package marker**

Create `autoresearch/config/pr_gate.toml`:

```toml
# Copyright (c) Kaden Schutt
# autoresearch/config/pr_gate.toml — the Tier-3 PR merge-gate config.
# Model x arch fit map, maintainer allowlist, and perf thresholds (mirror of the
# loop's WIN-gate constants). Nothing hardcodes these elsewhere; the gate reads
# it all from here.
archs = ["gfx1100", "gfx1151", "gfx1201"]
canonical_models = ["qwen3.6-27b", "qwen3.6-a3b"]
maintainers = ["fivetide", "unverbraucht", "nwoolmer", "Kaden-Schutt"]
floor = 0.15       # min |delta%| above clock noise (= perf.FLOOR)
drift_pct = 3.0    # cumulative master drift vs high-water B that fires investigation
alpha = 0.05       # Mann-Whitney significance for the regression test

# Which archs each SKU fits (VRAM). A (model, arch) cell runs only if it fits.
[fit]
"qwen3.6-27b" = ["gfx1100", "gfx1151", "gfx1201"]
"qwen3.6-a3b" = ["gfx1100", "gfx1151", "gfx1201"]
"deepseek4" = ["gfx1151"]
```

Create `autoresearch/ar/gate/__init__.py`:

```python
# Copyright (c) Kaden Schutt
"""ar.gate — the Tier-3 PR merge-gate engine (no-GPU-testable core).

Reuses the certify arms (parity/perf/coherence) + the ServeRunner seam, adding
gate-specific orchestration: reject a significant *regression* (mirror of the
loop's WIN gate), but PASS perf-neutral and improvement PRs — and run coherence
even on a neutral PR (unlike the loop's certify, which short-circuits).
"""
from .config import GateConfig, load_gate_config

__all__ = ["GateConfig", "load_gate_config"]
```

- [ ] **Step 4: Write the loader**

Create `autoresearch/ar/gate/config.py`:

```python
# Copyright (c) Kaden Schutt
"""ar.gate.config — TOML config for the Tier-3 PR merge-gate.

Mirrors ar.config's stdlib-tomllib pattern. Holds the model x arch fit map, the
maintainer allowlist, and the perf thresholds (the loop's WIN-gate constants,
mirrored for the gate's regression test).
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field


@dataclass
class GateConfig:
    archs: list[str]
    canonical_models: list[str]
    fit: dict[str, list[str]]
    maintainers: list[str]
    floor: float = 0.15
    drift_pct: float = 3.0
    alpha: float = 0.05

    def fits(self, model: str, arch: str) -> bool:
        """True iff SKU ``model`` fits ``arch`` per the [fit] map (unknown -> False)."""
        return arch in self.fit.get(model, [])

    def other_archs(self, arch: str) -> list[str]:
        """The configured archs except ``arch`` (the cross-arch isolation targets)."""
        return [a for a in self.archs if a != arch]

    def models_for(self, arch: str, extra: tuple[str, ...] = ()) -> list[str]:
        """Canonical models that fit ``arch`` plus any fitting ``extra`` (change-specific),
        de-duplicated, order-preserving."""
        out: list[str] = []
        for m in list(self.canonical_models) + list(extra):
            if m not in out and self.fits(m, arch):
                out.append(m)
        return out


def load_gate_config(path: str) -> GateConfig:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return GateConfig(
        archs=[str(a) for a in data["archs"]],
        canonical_models=[str(m) for m in data["canonical_models"]],
        fit={str(k): [str(a) for a in v] for k, v in data.get("fit", {}).items()},
        maintainers=[str(m) for m in data.get("maintainers", [])],
        floor=float(data.get("floor", 0.15)),
        drift_pct=float(data.get("drift_pct", 3.0)),
        alpha=float(data.get("alpha", 0.05)),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest autoresearch/ar/tests/test_gate_config.py -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add autoresearch/config/pr_gate.toml autoresearch/ar/gate/__init__.py \
        autoresearch/ar/gate/config.py autoresearch/ar/tests/test_gate_config.py
git commit -m "feat(ar-gate): pr_gate.toml + GateConfig loader with fit map

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Perf policy — `classify_perf` (mirror of the loop's WIN gate)

**Files:**
- Create: `autoresearch/ar/gate/perf_policy.py`
- Test: `autoresearch/ar/tests/test_gate_perf_policy.py`

**Interfaces:**
- Consumes: `autoresearch.ar.certify.perf` (`dominance_f`, `FLOOR`, `WIN_F`,
  `DEAD_F`), stdlib `statistics`.
- Produces:
  - `classify_perf(base_tok, var_tok, base_dur, var_dur, floor=perf.FLOOR, win_f=perf.WIN_F, dead_f=perf.DEAD_F) -> str` returning one of `"IMPROVEMENT"`, `"REGRESSION"`, `"NEUTRAL"`.

Semantics: `dominance_f(base, var)` is the fraction of pairs with `var > base`.
For tok/s, high dominance ⇒ variant faster; for duration, high dominance ⇒
variant *slower*. IMPROVEMENT = the loop's WIN (var faster on both axes,
dominance ≥ `win_f`, delta beyond `floor`). REGRESSION = the mirror (var slower
on both axes, dominance ≤ `dead_f`, delta beyond `floor`). Anything else NEUTRAL.

- [ ] **Step 1: Write the failing test**

Create `autoresearch/ar/tests/test_gate_perf_policy.py`:

```python
# Copyright (c) Kaden Schutt
from autoresearch.ar.gate.perf_policy import classify_perf


def test_clear_regression_var_slower_both_axes():
    # variant tok/s down (150 -> 140) AND duration up (10.0 -> 10.8), tight -> significant
    assert classify_perf(
        base_tok=[150] * 8, var_tok=[140] * 8,
        base_dur=[10.0] * 8, var_dur=[10.8] * 8,
    ) == "REGRESSION"


def test_clear_improvement_var_faster_both_axes():
    assert classify_perf(
        base_tok=[150] * 8, var_tok=[162] * 8,
        base_dur=[10.0] * 8, var_dur=[9.2] * 8,
    ) == "IMPROVEMENT"


def test_neutral_within_floor():
    # 0.05% moves, well under FLOOR=0.15 -> NEUTRAL even though tight
    assert classify_perf(
        base_tok=[1000.0] * 8, var_tok=[999.5] * 8,
        base_dur=[10.0] * 8, var_dur=[10.005] * 8,
    ) == "NEUTRAL"


def test_one_sided_move_is_neutral_not_regression():
    # tok/s down but duration FLAT -> not a conjunctive regression -> NEUTRAL
    assert classify_perf(
        base_tok=[150] * 8, var_tok=[140] * 8,
        base_dur=[10.0] * 8, var_dur=[10.0] * 8,
    ) == "NEUTRAL"


def test_noisy_overlap_is_neutral():
    # large overlap between arms -> dominance in the mushy middle -> NEUTRAL
    assert classify_perf(
        base_tok=[150, 149, 151, 148, 152, 150, 149, 151],
        var_tok=[149, 151, 148, 152, 150, 149, 151, 150],
        base_dur=[10.0, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.1],
        var_dur=[10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.1, 9.9],
    ) == "NEUTRAL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest autoresearch/ar/tests/test_gate_perf_policy.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'autoresearch.ar.gate.perf_policy'`.

- [ ] **Step 3: Write the classifier**

Create `autoresearch/ar/gate/perf_policy.py`:

```python
# Copyright (c) Kaden Schutt
"""ar.gate.perf_policy — the gate's perf decision, mirror of the loop's WIN gate.

The loop declares a WIN iff tok/s UP and duration DOWN, both dominant (>= WIN_F)
and beyond FLOOR. The gate declares a REGRESSION iff tok/s DOWN and duration UP,
both dominant against the variant (<= DEAD_F) and beyond FLOOR. Everything in
between — including any one-sided or sub-FLOOR move — is NEUTRAL and PASSes.
"""
from __future__ import annotations

import statistics as _st

from ..certify import perf


def _delta_pct(base, var) -> float:
    """Median %% change of var vs base (+ = var larger)."""
    mb = _st.median(base)
    return (_st.median(var) - mb) / mb * 100.0 if mb else 0.0


def classify_perf(base_tok, var_tok, base_dur, var_dur,
                  floor=perf.FLOOR, win_f=perf.WIN_F, dead_f=perf.DEAD_F) -> str:
    """Return "IMPROVEMENT" | "REGRESSION" | "NEUTRAL" (conjunctive, both axes)."""
    tok_d = _delta_pct(base_tok, var_tok)     # + = variant faster
    dur_d = _delta_pct(base_dur, var_dur)     # + = variant slower (worse)
    f_tok = perf.dominance_f(base_tok, var_tok)   # P(var_tok > base_tok): high => faster
    f_dur = perf.dominance_f(base_dur, var_dur)   # P(var_dur > base_dur): high => slower

    improvement = (f_tok >= win_f and tok_d > floor and f_dur <= dead_f and dur_d < -floor)
    if improvement:
        return "IMPROVEMENT"
    regression = (f_tok <= dead_f and tok_d < -floor and f_dur >= win_f and dur_d > floor)
    if regression:
        return "REGRESSION"
    return "NEUTRAL"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest autoresearch/ar/tests/test_gate_perf_policy.py -q`
Expected: PASS (5 tests). If `test_noisy_overlap_is_neutral` fails, the arms are
too tight/separated — the given sample vectors overlap enough that
`dominance_f` lands between `DEAD_F` and `WIN_F`; do not weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add autoresearch/ar/gate/perf_policy.py autoresearch/ar/tests/test_gate_perf_policy.py
git commit -m "feat(ar-gate): classify_perf — regression = mirror of the WIN gate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `gate_cell` — single (model,arch) orchestration

**Files:**
- Create: `autoresearch/ar/gate/engine.py`
- Test: `autoresearch/ar/tests/test_gate_engine.py`

**Interfaces:**
- Consumes: `certify.verdict` (`parity_result`, `coherence_result`),
  `certify.orchestrator` (`ServeRunner`, `DEFAULT_SEEDS`, `measurement_hash`),
  `gate.perf_policy.classify_perf`.
- Produces:
  - `gate_cell(runner, *, base_daemon, var_daemon, arch, model, base_ref, kv, maxtok, prompt_md5, floor=perf.FLOOR, seeds=None, expects=None) -> dict`
    with keys: `gate_verdict` (`"PASS"`|`"REJECT"`), `reason`
    (`"parity"`|`"coherence"`|`"perf_regression"`|`"neutral"`|`"improvement"`),
    `perf_class`, `model`, `gpu_arch`, `base_sha`, `variant_sha`,
    `measurement_hash`, `tok_delta_pct`, `dur_delta_pct`.

Orchestration differs from the loop's `certify()`: parity fail short-circuits
(a value change is a hard reject), but perf-neutral does **not** short-circuit —
coherence runs regardless, because a perf-neutral PR must still be coherent.

- [ ] **Step 1: Write the failing test**

Create `autoresearch/ar/tests/test_gate_engine.py`:

```python
# Copyright (c) Kaden Schutt
from autoresearch.ar.certify.orchestrator import ServeRunner
from autoresearch.ar.gate.engine import gate_cell

_CELL = dict(arch="gfx1201", model="qwen3.6-a3b", base_ref="master",
             kv="q8", maxtok=128, prompt_md5="abc123")


def _cg(genre="prose", text="fine", toks=None):
    return {"prompt_id": genre, "genre": genre, "text": text,
            "token_ids": toks if toks is not None else list(range(1000, 1060)),
            "tool_calls": []}


class Runner(ServeRunner):
    """Configurable mock: parity ids, perf sample maps, coherence gens per daemon."""

    def __init__(self, *, parity=None, tok=None, dur=None, coh=None):
        self._parity = parity or {"base": [1, 2, 3], "var": [1, 2, 3]}
        self._tok = tok or {"base": [150] * 8, "var": [150] * 8}
        self._dur = dur or {"base": [10.0] * 8, "var": [10.0] * 8}
        self._coh = coh or {"base": [_cg()], "var": [_cg()]}

    def parity_gens(self, d):
        return [{"prompt_id": "p1", "token_ids": self._parity[d]}]

    def perf_measure(self, d):
        return (self._tok[d], self._dur[d])

    def coherence_gens(self, d, seeds):
        return [dict(g, seed=s) for s in seeds for g in self._coh[d]]

    def clocks(self, d):
        return []


def test_neutral_passes_and_is_self_describing():
    r = Runner()  # identical everything
    row = gate_cell(r, base_daemon="base", var_daemon="var", **_CELL)
    assert row["gate_verdict"] == "PASS"
    assert row["perf_class"] == "NEUTRAL"
    assert row["measurement_hash"] and len(row["measurement_hash"]) == 16
    assert row["gpu_arch"] == "gfx1201" and row["model"] == "qwen3.6-a3b"


def test_parity_fail_rejects_and_short_circuits_perf():
    class NoPerf(Runner):
        def perf_measure(self, d):
            raise AssertionError("perf must not run after parity fail")

    r = NoPerf(parity={"base": [1, 2, 3], "var": [1, 9, 3]})
    row = gate_cell(r, base_daemon="base", var_daemon="var", **_CELL)
    assert row["gate_verdict"] == "REJECT" and row["reason"] == "parity"


def test_significant_regression_rejects():
    r = Runner(tok={"base": [150] * 8, "var": [140] * 8},
               dur={"base": [10.0] * 8, "var": [10.8] * 8})
    row = gate_cell(r, base_daemon="base", var_daemon="var", **_CELL)
    assert row["gate_verdict"] == "REJECT" and row["reason"] == "perf_regression"


def test_coherence_runs_even_when_perf_neutral():
    # perf identical (neutral) but variant attractors -> must still REJECT on coherence
    r = Runner(coh={"base": [_cg()], "var": [_cg(toks=[7] * 60)]})
    row = gate_cell(r, base_daemon="base", var_daemon="var", **_CELL)
    assert row["gate_verdict"] == "REJECT" and row["reason"] == "coherence"


def test_improvement_passes():
    r = Runner(tok={"base": [150] * 8, "var": [162] * 8},
               dur={"base": [10.0] * 8, "var": [9.2] * 8})
    row = gate_cell(r, base_daemon="base", var_daemon="var", **_CELL)
    assert row["gate_verdict"] == "PASS" and row["reason"] == "improvement"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest autoresearch/ar/tests/test_gate_engine.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'autoresearch.ar.gate.engine'`.

- [ ] **Step 3: Write `gate_cell`**

Create `autoresearch/ar/gate/engine.py`:

```python
# Copyright (c) Kaden Schutt
"""ar.gate.engine — gate orchestration over the reused certify arms.

gate_cell: one (model, arch) certification. Order: parity -> perf -> coherence,
but — unlike the loop's certify() — perf-neutral does NOT short-circuit;
coherence always runs (a perf-neutral PR must still be coherent). The verdict is
gate-shaped (PASS / REJECT), reusing verdict.parity_result / coherence_result and
gate.perf_policy.classify_perf.
"""
from __future__ import annotations

from ..certify import perf
from ..certify import verdict as V
from ..certify.orchestrator import DEFAULT_SEEDS, ServeRunner, measurement_hash
from .perf_policy import _delta_pct, classify_perf

__all__ = ["gate_cell", "ServeRunner"]


def _row(*, gate_verdict, reason, perf_class, arch, model, base_daemon, var_daemon,
         kv, maxtok, prompt_md5, tok_delta_pct=None, dur_delta_pct=None):
    return {
        "gate_verdict": gate_verdict, "reason": reason, "perf_class": perf_class,
        "gpu_arch": arch, "model": model, "base_sha": base_daemon, "variant_sha": var_daemon,
        "kv": kv, "maxtok": maxtok, "prompt_md5": prompt_md5,
        "measurement_hash": measurement_hash(arch, model, base_daemon, var_daemon,
                                             prompt_md5, kv, maxtok),
        "tok_delta_pct": tok_delta_pct, "dur_delta_pct": dur_delta_pct,
    }


def gate_cell(runner, *, base_daemon, var_daemon, arch, model, base_ref, kv, maxtok,
              prompt_md5, floor=perf.FLOOR, seeds=None, expects=None) -> dict:
    """Certify one (model, arch) cell: PASS unless parity/coherence fail or a
    significant perf regression. Neutral and improvement both PASS."""
    seeds = DEFAULT_SEEDS if seeds is None else seeds
    common = dict(perf_class=None, arch=arch, model=model, base_daemon=base_daemon,
                  var_daemon=var_daemon, kv=kv, maxtok=maxtok, prompt_md5=prompt_md5)

    # 1. PARITY — a value change is a hard reject; short-circuit.
    p_ok, _ = V.parity_result(runner.parity_gens(base_daemon), runner.parity_gens(var_daemon))
    if not p_ok:
        return _row(gate_verdict="REJECT", reason="parity", **common)

    # 2. PERF — classify, but do NOT short-circuit on neutral.
    base_tok, base_dur = runner.perf_measure(base_daemon)
    var_tok, var_dur = runner.perf_measure(var_daemon)
    pclass = classify_perf(base_tok, var_tok, base_dur, var_dur, floor=floor)
    tok_d, dur_d = _delta_pct(base_tok, var_tok), _delta_pct(base_dur, var_dur)
    common.update(perf_class=pclass, tok_delta_pct=tok_d, dur_delta_pct=dur_d)

    # 3. COHERENCE — always (a neutral PR must still be coherent).
    c_ok, _ = V.coherence_result(runner.coherence_gens(base_daemon, seeds),
                                 runner.coherence_gens(var_daemon, seeds), expects=expects)
    if not c_ok:
        return _row(gate_verdict="REJECT", reason="coherence", **common)

    if pclass == "REGRESSION":
        return _row(gate_verdict="REJECT", reason="perf_regression", **common)
    return _row(gate_verdict="PASS", reason=pclass.lower(), **common)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest autoresearch/ar/tests/test_gate_engine.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add autoresearch/ar/gate/engine.py autoresearch/ar/tests/test_gate_engine.py
git commit -m "feat(ar-gate): gate_cell — parity/perf/coherence, no neutral short-circuit

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `run_gate` — per-arch engine (fit filter + cross-arch + confirm-rerun + reduce)

**Files:**
- Modify: `autoresearch/ar/gate/engine.py` (append `run_gate`)
- Modify: `autoresearch/ar/gate/__init__.py` (re-export `run_gate`, `gate_cell`)
- Test: `autoresearch/ar/tests/test_gate_engine.py` (append `run_gate` tests)

**Interfaces:**
- Consumes: `gate_cell` (Task 3), `certify.cross_arch.check_cross_arch`
  (signature `check_cross_arch(kernel_file, arch, other_archs, repo, base_sha=None, preprocess=None) -> list[str]`),
  a `runner_factory(model: str) -> ServeRunner`, a `GateConfig`.
- Produces:
  - `run_gate(*, arch, changed_kernel_files, models, base_ref, head_ref, repo, cfg, runner_factory, cross_arch_fn=None, kv="q8", maxtok=128, prompt_md5="", rerun_on_regression=True) -> dict`
    with keys: `arch`, `verdict` (`"PASS"`|`"REJECT"`), `cells` (list of cell rows),
    `cross_arch_leaks` (list of `{"file","leaks"}`), `reasons` (list of str).

Rules: an empty `models` list (nothing fits this arch) → `verdict="PASS"` with a
`"no-fitting-model"` reason (N/A, not a failure). Any cross-arch leak → REJECT. A
cell that REJECTs for `perf_regression` is re-run once via a fresh
`runner_factory(model)`; only a replicated regression keeps the REJECT (noise →
PASS). Any parity/coherence/replicated-regression cell REJECT → arch REJECT.

- [ ] **Step 1: Write the failing test (append to `test_gate_engine.py`)**

```python
from autoresearch.ar.gate.config import GateConfig
from autoresearch.ar.gate.engine import run_gate

_CFG = GateConfig(
    archs=["gfx1100", "gfx1151", "gfx1201"],
    canonical_models=["qwen3.6-27b", "qwen3.6-a3b"],
    fit={"qwen3.6-27b": ["gfx1100", "gfx1151", "gfx1201"],
         "qwen3.6-a3b": ["gfx1100", "gfx1151", "gfx1201"],
         "deepseek4": ["gfx1151"]},
    maintainers=["Kaden-Schutt"],
)


def _no_leak(*a, **k):
    return []


def _factory(**per_model):
    """runner_factory returning a fresh Runner per model (per_model overrides)."""
    def make(model):
        return per_model.get(model, Runner())
    return make


def test_run_gate_all_neutral_passes():
    res = run_gate(arch="gfx1201", changed_kernel_files=[],
                   models=["qwen3.6-27b", "qwen3.6-a3b"], base_ref="master",
                   head_ref="pr", repo="/repo", cfg=_CFG,
                   runner_factory=_factory(), cross_arch_fn=_no_leak)
    assert res["verdict"] == "PASS"
    assert len(res["cells"]) == 2


def test_run_gate_cross_arch_leak_rejects():
    def leak(kernel_file, arch, other_archs, repo, base_sha=None, preprocess=None):
        return ["gfx1100"]                       # this file perturbs gfx1100 codegen
    res = run_gate(arch="gfx1201", changed_kernel_files=["kernels/src/x.hip"],
                   models=["qwen3.6-a3b"], base_ref="master", head_ref="pr",
                   repo="/repo", cfg=_CFG, runner_factory=_factory(), cross_arch_fn=leak)
    assert res["verdict"] == "REJECT"
    assert res["cross_arch_leaks"] == [{"file": "kernels/src/x.hip", "leaks": ["gfx1100"]}]


def test_run_gate_no_fitting_model_is_pass_na():
    res = run_gate(arch="gfx1100", changed_kernel_files=[], models=[],
                   base_ref="master", head_ref="pr", repo="/repo", cfg=_CFG,
                   runner_factory=_factory(), cross_arch_fn=_no_leak)
    assert res["verdict"] == "PASS" and "no-fitting-model" in res["reasons"]


def test_run_gate_confirm_rerun_flips_noise_to_pass():
    # first runner regresses, the rerun (fresh from factory) is neutral -> PASS
    calls = {"n": 0}
    regress = Runner(tok={"base": [150] * 8, "var": [140] * 8},
                     dur={"base": [10.0] * 8, "var": [10.8] * 8})

    def make(model):
        calls["n"] += 1
        return regress if calls["n"] == 1 else Runner()   # 2nd call = neutral
    res = run_gate(arch="gfx1201", changed_kernel_files=[], models=["qwen3.6-a3b"],
                   base_ref="master", head_ref="pr", repo="/repo", cfg=_CFG,
                   runner_factory=make, cross_arch_fn=_no_leak)
    assert res["verdict"] == "PASS" and calls["n"] == 2      # reran once


def test_run_gate_replicated_regression_rejects():
    def make(model):
        return Runner(tok={"base": [150] * 8, "var": [140] * 8},
                      dur={"base": [10.0] * 8, "var": [10.8] * 8})
    res = run_gate(arch="gfx1201", changed_kernel_files=[], models=["qwen3.6-a3b"],
                   base_ref="master", head_ref="pr", repo="/repo", cfg=_CFG,
                   runner_factory=make, cross_arch_fn=_no_leak)
    assert res["verdict"] == "REJECT" and "perf_regression" in res["reasons"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest autoresearch/ar/tests/test_gate_engine.py -q`
Expected: FAIL — `ImportError: cannot import name 'run_gate'`.

- [ ] **Step 3: Append `run_gate` to `engine.py`**

Add to the top imports of `autoresearch/ar/gate/engine.py`:

```python
from ..certify import cross_arch
```

Update `__all__` in `engine.py` to `["gate_cell", "run_gate", "ServeRunner"]`, and append:

```python
def run_gate(*, arch, changed_kernel_files, models, base_ref, head_ref, repo, cfg,
             runner_factory, cross_arch_fn=None, kv="q8", maxtok=128, prompt_md5="",
             rerun_on_regression=True) -> dict:
    """Certify a PR on one arch: cross-arch isolation + every fitting (model,arch)
    cell. REJECT on any cross-arch leak, parity/coherence fail, or replicated perf
    regression; PASS otherwise (empty models => N/A PASS)."""
    cross = cross_arch_fn or cross_arch.check_cross_arch
    reasons: list[str] = []
    leaks: list[dict] = []

    # Cross-arch isolation (cheapest; independent of GPU cells).
    for f in changed_kernel_files:
        got = cross(f, arch, cfg.other_archs(arch), repo, base_sha=base_ref)
        if got:
            leaks.append({"file": f, "leaks": list(got)})
    if leaks:
        reasons.append("cross_arch")

    cells: list[dict] = []
    if not models:
        reasons.append("no-fitting-model")

    for model in models:
        cell = gate_cell(runner_factory(model), base_daemon=head_ref, var_daemon=head_ref,
                         arch=arch, model=model, base_ref=base_ref, kv=kv, maxtok=maxtok,
                         prompt_md5=prompt_md5)
        # NOTE: base_daemon/var_daemon are the SHA identities the GPU runner maps to
        # base_ref vs head_ref builds; the mock ignores the values. Real (Phase-3)
        # LiveServeRunner is constructed by the factory bound to (base_ref, head_ref).
        if cell["gate_verdict"] == "REJECT" and cell["reason"] == "perf_regression" and rerun_on_regression:
            confirm = gate_cell(runner_factory(model), base_daemon=head_ref, var_daemon=head_ref,
                                arch=arch, model=model, base_ref=base_ref, kv=kv, maxtok=maxtok,
                                prompt_md5=prompt_md5)
            if not (confirm["gate_verdict"] == "REJECT" and confirm["reason"] == "perf_regression"):
                cell = confirm            # first was noise; keep the (passing) rerun
        cells.append(cell)
        if cell["gate_verdict"] == "REJECT":
            reasons.append(cell["reason"])

    verdict = "REJECT" if (leaks or any(c["gate_verdict"] == "REJECT" for c in cells)) else "PASS"
    return {"arch": arch, "verdict": verdict, "cells": cells,
            "cross_arch_leaks": leaks, "reasons": reasons}
```

Update `autoresearch/ar/gate/__init__.py` to also re-export the engine:

```python
# Copyright (c) Kaden Schutt
"""ar.gate — the Tier-3 PR merge-gate engine (no-GPU-testable core)."""
from .config import GateConfig, load_gate_config
from .engine import gate_cell, run_gate

__all__ = ["GateConfig", "load_gate_config", "gate_cell", "run_gate"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest autoresearch/ar/tests/test_gate_engine.py -q`
Expected: PASS (10 tests total — 5 from Task 3 + 5 here).

- [ ] **Step 5: Commit**

```bash
git add autoresearch/ar/gate/engine.py autoresearch/ar/gate/__init__.py \
        autoresearch/ar/tests/test_gate_engine.py
git commit -m "feat(ar-gate): run_gate — fit filter + cross-arch guard + confirm-rerun + reduce

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `ar gate` CLI verb (plan view, no-GPU)

**Files:**
- Modify: `autoresearch/ar/cli.py` (add `cmd_gate`, register `gate`, dispatch)
- Test: `autoresearch/ar/tests/test_gate_cli.py`

**Interfaces:**
- Consumes: `gate.config.load_gate_config`, the `cli._repo()` helper.
- Produces: `cmd_gate(a) -> int`; `gate` in the subparser + `_DISPATCH` +
  `OPERATOR_VERBS`. In this plan the verb resolves and prints the gate **plan**
  for an arch (`--plan`): the fitting models, the cross-arch targets, and the
  thresholds — the no-GPU slice. The GPU execution path (build daemons + real
  `run_gate` over `LiveServeRunner`) is wired in the Phase-3 workflow plan.

- [ ] **Step 1: Write the failing test**

Create `autoresearch/ar/tests/test_gate_cli.py`:

```python
# Copyright (c) Kaden Schutt
import io
import json
from contextlib import redirect_stdout

from autoresearch.ar.cli import main


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def test_gate_plan_lists_fitting_models_and_other_archs():
    rc, out = _run(["gate", "--arch", "gfx1201", "--plan"])
    assert rc == 0
    d = json.loads(out)
    assert d["arch"] == "gfx1201"
    assert d["models"] == ["qwen3.6-27b", "qwen3.6-a3b"]
    assert d["other_archs"] == ["gfx1100", "gfx1151"]
    assert d["floor"] == 0.15 and d["alpha"] == 0.05


def test_gate_plan_extra_model_included_only_where_it_fits():
    rc, out = _run(["gate", "--arch", "gfx1151", "--plan", "--models", "deepseek4"])
    assert json.loads(out)["models"] == ["qwen3.6-27b", "qwen3.6-a3b", "deepseek4"]
    rc, out = _run(["gate", "--arch", "gfx1100", "--plan", "--models", "deepseek4"])
    assert json.loads(out)["models"] == ["qwen3.6-27b", "qwen3.6-a3b"]


def test_gate_is_operator_only():
    rc, out = _run(["--role", "agent", "gate", "--arch", "gfx1201", "--plan"])
    assert rc == 3
    assert json.loads(out)["reason"] == "ROLE_FORBIDDEN"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest autoresearch/ar/tests/test_gate_cli.py -q`
Expected: FAIL — `argument cmd: invalid choice: 'gate'`.

- [ ] **Step 3: Add `cmd_gate` + register the verb**

In `autoresearch/ar/cli.py`, add after `cmd_config` (near the operator verbs):

```python
def cmd_gate(a) -> int:
    """Resolve + print the Tier-3 gate plan for an arch (the fitting models, the
    cross-arch targets, the thresholds). No GPU here — the GPU run is the Phase-3
    workflow. Reads autoresearch/config/pr_gate.toml."""
    from .gate.config import load_gate_config

    path = a.gate_config or os.path.join(_repo(), "autoresearch", "config", "pr_gate.toml")
    cfg = load_gate_config(path)
    extra = tuple(m for m in (a.models.split(",") if a.models else []) if m)
    out = {
        "arch": a.arch,
        "models": cfg.models_for(a.arch, extra=extra),
        "other_archs": cfg.other_archs(a.arch),
        "floor": cfg.floor,
        "alpha": cfg.alpha,
        "drift_pct": cfg.drift_pct,
    }
    print(json.dumps(out, indent=2) if a.json else json.dumps(out))
    return 0
```

Add `"gate"` to `OPERATOR_VERBS`:

```python
OPERATOR_VERBS = frozenset(
    {"start", "stop", "status", "why", "bod", "ingest", "fold", "rollover", "config", "certify", "gate"}
)
```

Register the subparser in `build_parser()` (after the `config` subparser):

```python
    s = sub.add_parser("gate", help="operator: resolve/run the Tier-3 PR gate for an arch")
    s.add_argument("--arch", required=True)
    s.add_argument("--plan", action="store_true", help="print the resolved gate plan (no GPU)")
    s.add_argument("--models", default=None, help="comma-separated change-specific SKUs to add")
    s.add_argument("--gate-config", dest="gate_config", default=None)
    s.add_argument("--json", action="store_true")
```

Add to `_DISPATCH`:

```python
    "gate": cmd_gate,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest autoresearch/ar/tests/test_gate_cli.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the whole gate + certify suite (no regressions)**

Run: `python -m pytest autoresearch/ar/tests/ -q -k "gate or certify or config or cli"`
Expected: PASS (all gate tests + the existing certify/config/cli tests untouched).

- [ ] **Step 6: Commit**

```bash
git add autoresearch/ar/cli.py autoresearch/ar/tests/test_gate_cli.py
git commit -m "feat(ar-gate): ar gate CLI verb (plan view, operator-only)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (this plan = Phase 1 of the spec):**
- §4 gate 1 (cross-arch) → Task 4 (`run_gate` cross-arch guard). ✔
- §4 gates 2–3b (parity/perf/coherence) → Task 3 (`gate_cell`) reusing the arms. ✔
- §5 model×arch fit map → Task 1 (`GateConfig.fits`/`models_for`). ✔
- §6.1 perf rule (reject significant replicated regression, mirror of WIN) →
  Task 2 (`classify_perf`) + Task 4 (confirm-rerun). ✔
- §13 `ar gate` subcommand, no-GPU-testable via mock runner → Task 5 + all tests. ✔
- **Deferred to later plans (correctly out of scope here):** §4 gate 0 fit/smoke
  on real GPU, §10 Gate 4 non-clobber merge, §12 authority/triggers, §14
  workflow, §6.2 drift guard + high-water B + ledger, §11 staging train. Each is
  its own plan (Phases 2–5).

**Placeholder scan:** no TBD/TODO; every code step shows complete code; every test
has real assertions and exact run commands. ✔

**Type consistency:** `gate_cell` and `run_gate` share the cell-row dict shape
(`gate_verdict`/`reason`/`perf_class`/`measurement_hash`); `classify_perf` returns
the three literals both consume; `GateConfig.models_for`/`other_archs` signatures
match their CLI + engine call sites; `cross_arch_fn` signature matches
`check_cross_arch`. ✔

## Next plans (not this one)

- **Phase 0** — self-hosted runners + the loop `gate-priority` hook.
- **Phase 2** — Gate 4 non-clobber merge + codex merge-fix + BOD.
- **Phase 3** — `gpu-gates.yml` + `claude-review.yml` dispatch/interpret/merge +
  triggers + `LiveServeRunner` daemon-build wiring into `run_gate`.
- **Phase 4** — perf governance (high-water B + drift guard + `pr_gate_merges.jsonl`).
- **Phase 5** — staging merge-train + freshness sync.
