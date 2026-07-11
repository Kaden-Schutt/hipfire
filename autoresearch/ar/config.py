# Copyright (c) Kaden Schutt
"""ar.config — per-arch TOML config for the autoresearch loop.

One file per arch (``autoresearch/config/loop_<arch>.toml``) fully describes a
loop: the kernel baseline ref it advances, the measurement SKU/kv/maxtok, the
candidate-wall and exhaustion budget, the agent harness, the per-worker
``{card, dev, model, effort}`` heterogeneity (the ``sed``-munge replacement),
and the watcher leashes. Nothing hardcodes arch/model/card — it all comes from
here.

Parsed with the stdlib ``tomllib`` (Python 3.11+); no third-party deps.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field


@dataclass
class WorkerCfg:
    """One loop worker pinned to a GPU: DRM ``card``, HIP ``dev``, agent
    ``model`` + reasoning ``effort``."""

    card: int
    dev: int
    model: str
    effort: str


@dataclass
class Bounds:
    """Watcher leashes: max agent calls per run and wall-clock TTL (seconds)."""

    call_budget: int
    wall_ttl_s: int


@dataclass
class LoopConfig:
    """The full per-arch loop configuration (see ``config-reference.md``)."""

    arch: str
    baseline_ref: str
    model: str
    kv_mode: str
    max_tokens: int
    prompt_md5: str
    cand_wall: float
    k_exhaust: int
    agent_harness: str
    workers: list[WorkerCfg] = field(default_factory=list)
    bounds: Bounds = field(default_factory=lambda: Bounds(400, 43200))


# Defaults per spec §6 — applied when a key is absent from the TOML.
_DEFAULTS = {
    "kv_mode": "q8",
    "max_tokens": 128,
    "cand_wall": 3.0,
    "k_exhaust": 5,
    "agent_harness": "codex",
}
_BOUNDS_DEFAULTS = {"call_budget": 400, "wall_ttl_s": 43200}


def load_config(path: str) -> LoopConfig:
    """Load a per-arch loop TOML into a :class:`LoopConfig`.

    Maps ``[[workers]]`` → ``list[WorkerCfg]`` and ``[bounds]`` → :class:`Bounds`,
    filling spec §6 defaults for any omitted key.
    """
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    workers = [
        WorkerCfg(
            card=int(w["card"]),
            dev=int(w["dev"]),
            model=str(w["model"]),
            effort=str(w["effort"]),
        )
        for w in data.get("workers", [])
    ]

    b = data.get("bounds", {})
    bounds = Bounds(
        call_budget=int(b.get("call_budget", _BOUNDS_DEFAULTS["call_budget"])),
        wall_ttl_s=int(b.get("wall_ttl_s", _BOUNDS_DEFAULTS["wall_ttl_s"])),
    )

    return LoopConfig(
        arch=str(data["arch"]),
        baseline_ref=str(data["baseline_ref"]),
        model=str(data["model"]),
        kv_mode=str(data.get("kv_mode", _DEFAULTS["kv_mode"])),
        max_tokens=int(data.get("max_tokens", _DEFAULTS["max_tokens"])),
        prompt_md5=str(data.get("prompt_md5", "")),
        cand_wall=float(data.get("cand_wall", _DEFAULTS["cand_wall"])),
        k_exhaust=int(data.get("k_exhaust", _DEFAULTS["k_exhaust"])),
        agent_harness=str(data.get("agent_harness", _DEFAULTS["agent_harness"])),
        workers=workers,
        bounds=bounds,
    )
