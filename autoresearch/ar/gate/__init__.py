# Copyright (c) Kaden Schutt
"""ar.gate — the Tier-3 PR merge-gate engine (no-GPU-testable core).

Reuses the certify arms (parity/perf/coherence) + the ServeRunner seam, adding
gate-specific orchestration: reject a significant *regression* (mirror of the
loop's WIN gate), but PASS perf-neutral and improvement PRs — and run coherence
even on a neutral PR (unlike the loop's certify, which short-circuits).
"""
from .config import GateConfig, load_gate_config
from .engine import gate_cell, run_gate
from .merge import assemble_bod, gate4, trial_merge

__all__ = [
    "GateConfig", "load_gate_config", "gate_cell", "run_gate",
    "trial_merge", "assemble_bod", "gate4",
]
