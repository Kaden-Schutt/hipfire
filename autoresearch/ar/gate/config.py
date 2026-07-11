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
    routing: dict = field(default_factory=dict)
    auto_merge_authors: list[str] = field(default_factory=list)

    def route(self, pr_class: str) -> dict:
        """Executor {harness, model, effort} for a PR risk class. Unknown class ->
        the high-risk row (fail-safe strongest)."""
        return self.routing.get(pr_class, self.routing.get("high-risk", {}))

    def is_auto_merge_author(self, author: str) -> bool:
        return author in self.auto_merge_authors

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
        routing={str(k): dict(v) for k, v in data.get("routing", {}).items()},
        auto_merge_authors=[str(a) for a in data.get("auto_merge_authors", [])],
    )
