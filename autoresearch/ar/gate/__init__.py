# Copyright (c) Kaden Schutt
"""ar.gate — the Tier-3 PR merge-gate engine (no-GPU-testable core)."""
from .config import GateConfig, load_gate_config
from .engine import gate_cell, run_gate
from .merge import assemble_bod, gate4, trial_merge
from .outcome import decide_pr, format_pr_comment
from .routing import classify_pr
from .staging import classify_conflict, fold_pr, land_train, recall_reproduce, stack_train

__all__ = [
    "GateConfig", "load_gate_config", "gate_cell", "run_gate",
    "trial_merge", "assemble_bod", "gate4",
    "classify_pr", "decide_pr", "format_pr_comment",
    "classify_conflict", "recall_reproduce", "fold_pr", "stack_train", "land_train",
]
