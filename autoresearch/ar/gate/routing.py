# Copyright (c) Kaden Schutt
"""ar.gate.routing — PR risk FLOOR (spec §8/§8.1).

This is a deterministic *floor*, NOT the primary classifier: Claude reads the diff
semantically and sets the authoritative risk class + behavior-test plan. classify_pr
only pins a MINIMUM tier from the file paths — the coherence-gate trigger taxonomy
CLAUDE.md guards (kernels / dispatch / forward-pass / quant) → high-risk — so Claude
can escalate above it but a kernel PR can never be de-classified to trivial (even if
Claude errs or is unavailable). Conservative on ambiguity — err toward more testing.
"""
from __future__ import annotations

# A path is high-risk if it contains ANY of these markers (substring match).
HIGH_RISK_MARKERS = (
    "kernels/",                       # all HIP kernel source
    "crates/rdna-compute/",           # dispatch + kernel launch + JIT
    "crates/hipfire-dispatch/",       # unified per-family dispatch
    "crates/hipfire-arch-",           # forward passes (all arch crates)
    "crates/hipfire-quantize/",       # quant encoders / formats
    "/sampler",                       # sampling path
    "dispatch.rs",                    # the most-reverted file
)

# A path is trivial if it matches ANY of these (prefix or suffix).
_TRIVIAL_PREFIXES = ("docs/", ".github/")
_TRIVIAL_SUFFIXES = (".md",)


def _is_high_risk(path: str) -> bool:
    return any(m in path for m in HIGH_RISK_MARKERS)


def _is_trivial(path: str) -> bool:
    return path.startswith(_TRIVIAL_PREFIXES) or path.endswith(_TRIVIAL_SUFFIXES)


def classify_pr(changed_files, lines_changed=None, small_threshold=40) -> str:
    """Classify a PR into 'trivial' | 'low' | 'moderate' | 'high-risk'."""
    if any(_is_high_risk(f) for f in changed_files):
        return "high-risk"
    if not changed_files or all(_is_trivial(f) for f in changed_files):
        return "trivial"
    if lines_changed is not None and lines_changed <= small_threshold:
        return "low"
    return "moderate"
