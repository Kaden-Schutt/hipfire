# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Shared dataclasses for the change-targeted validation gate."""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_ID = "hipfire.change_gate/1"


@dataclass(frozen=True)
class Route:
    id: str  # stable dotted id, e.g. "serve.battery.qwen35-27b"
    kind: str  # "serve" | "redline" | "speed" | "unit" | "detect" | "shell"
    argv: tuple[str, ...]  # executable command; {model} / {out} placeholders allowed
    est_minutes: float
    tier: str  # "cheap" (<2min) | "standard" (2-15min) | "heavy" (>15min)
    arches: tuple[str, ...]  # () means any arch
    models: tuple[str, ...]  # model basenames required under MODELS_DIR; () means none
    why: str  # one line: what regression class this route catches


@dataclass(frozen=True)
class Rule:
    surface: str  # repo-relative glob (fnmatch) OR "re:<regex>"
    route_ids: tuple[str, ...]
    reason: str  # why this surface owes these routes


@dataclass(frozen=True)
class Selection:
    route_id: str
    matched_paths: tuple[str, ...]
    rule_reason: str
    status: str  # "selected" | "blocked_model" | "blocked_arch" | "trimmed_budget" | "excluded_heavy"
    detail: str


@dataclass
class RouteResult:
    route_id: str
    status: str  # "pass" | "fail" | "blocked" | "skipped"
    duration_s: float
    verdict: dict  # detector/harness output, JSON-serialisable
    artifacts: tuple[str, ...]
