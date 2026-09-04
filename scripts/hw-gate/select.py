#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""hw-gate bucket selection: changed paths -> buckets, policy hits, needs_hw.

Pure and deterministic. No git, no network, no GPU. Reads changed paths from
stdin (one per line, as `git diff --name-only BASE...HEAD` prints them) and
classifies them by prefix/glob. The tables below are the whole policy; there
is no rule engine and no per-route table. Optionally parses an author's
`<!-- hw-gate-request -->` block from `--pr-body`.

CONTRACT
    stdin : changed paths, one per line
    stdout: JSON (also written to --json PATH when given)
        {
          "schema": "hipfire.hw-gate.select", "version": 1,
          "needs_hw": bool,                 # any bucket, OR any policy path (the seats must review those)
          "buckets": ["load", "serve", "kernel"],   # sorted, deduplicated, may be []
          "policy_paths": ["..."],          # touched paths matching POLICY (never bot-approvable)
          "exec_sensitive_paths": ["..."],  # informational input to the reviewer (not a gate)
          "request": {"routes":[{"mode":"battery"|"chain","tag":"..."}], "claim":"..."} | null,
          "request_error": "..." | null,   # set when marker present but body malformed
          "surfaces": {"load": ["path", ...], "serve": [...], "kernel": [...], "policy": [...], "other": [...]}
        }
    --pr-body FILE : parse the first hw-gate-request JSON fence from the PR body
    --github-output FILE : append `needs_hw=`, `buckets=`, `policy=`, `exec_sensitive=`,
                           `request_present=` (true|false) lines
    exit 0 always on well-formed input; exit 2 on usage error

BUCKET RULES (first match wins per path; a path may hit `policy` in addition)
    kernel : kernels/**, crates/rdna-compute/**, crates/hipfire-dispatch/**, crates/hip-bridge/**,
             crates/saddle-core/**
    serve  : crates/hipfire-engine/**, crates/hipfire-generate/**, crates/hipfire-daemon/src/slots.rs,
             crates/hipfire-daemon/src/serve*.rs, crates/hipfire-runtime/src/{emit_text,eos_filter,dflash,
             dflash_generic,dspark_core,spec,reset_core,triattn}.rs, crates/hipfire-arch-*/src/**/{serve,generate,spec}*.rs
    load   : crates/hipfire-loader/**, crates/hipfire-daemon/** (remaining), crates/hipfire-runtime/src/{model_load,
             hfq,loader_api,config,safetensors_source,weight_backend,multi_gpu,arch_model,arch}.rs,
             crates/hipfire-arch-*/src/**/load*.rs, crates/hipfire-arch-*/src/**/weights*.rs,
             crates/hipfire-arch-*/src/carrier.rs, crates/hipfire-config/**, crates/hipfire-registry/**,
             registry/**, Cargo.toml, Cargo.lock, crates/*/Cargo.toml
    none   : everything else (docs/**, benchmarks/**, scripts/** except hw-gate, tests/**, *.md, ...)

    `serve` and `kernel` imply `load` (the fixtures must still load through the user route).

EXEC-SENSITIVE (informational input to the reviewer seat; not a hardware gate —
    Sol decides run_hardware; a maintainer's `hw-run` label only forces a run)
    **/build.rs, .cargo/**, Cargo.toml, Cargo.lock, crates/*/Cargo.toml, rust-toolchain*, .github/**,
    scripts/**, tools/**, **/*.sh, **/*.py, Makefile, justfile, Dockerfile*, flake.nix, nix/**

POLICY (touching any of these => policy_paths non-empty => review.py may never greenlight)
    .github/workflows/**, .github/CODEOWNERS, scripts/hw-gate/**, scripts/leanup-thresholds.txt,
    scripts/layering.txt, scripts/ratchet-diff.sh, scripts/leanup-ratchets.sh, registry/**
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import posixpath
import sys

# -- pattern tables -----------------------------------------------------------

_KERNEL_PATTERNS: list[str] = [
    "kernels/**",
    "crates/rdna-compute/**",
    "crates/hipfire-dispatch/**",
    "crates/hip-bridge/**",
    "crates/saddle-core/**",
]

_SERVE_PATTERNS: list[str] = [
    "crates/hipfire-engine/**",
    "crates/hipfire-generate/**",
    "crates/hipfire-daemon/src/slots.rs",
    "crates/hipfire-daemon/src/serve*.rs",
    # crates/hipfire-runtime/src/{emit_text,...}.rs expanded
    "crates/hipfire-runtime/src/emit_text.rs",
    "crates/hipfire-runtime/src/eos_filter.rs",
    "crates/hipfire-runtime/src/dflash.rs",
    "crates/hipfire-runtime/src/dflash_generic.rs",
    "crates/hipfire-runtime/src/dspark_core.rs",
    "crates/hipfire-runtime/src/spec.rs",
    "crates/hipfire-runtime/src/reset_core.rs",
    "crates/hipfire-runtime/src/triattn.rs",
    # crates/hipfire-arch-*/src/**/{serve,generate,spec}*.rs expanded
    "crates/hipfire-arch-*/src/**/serve*.rs",
    "crates/hipfire-arch-*/src/**/generate*.rs",
    "crates/hipfire-arch-*/src/**/spec*.rs",
]

_LOAD_PATTERNS: list[str] = [
    "crates/hipfire-loader/**",
    "crates/hipfire-daemon/**",
    # crates/hipfire-runtime/src/{model_load,...}.rs expanded
    "crates/hipfire-runtime/src/model_load.rs",
    "crates/hipfire-runtime/src/hfq.rs",
    "crates/hipfire-runtime/src/loader_api.rs",
    "crates/hipfire-runtime/src/config.rs",
    "crates/hipfire-runtime/src/safetensors_source.rs",
    "crates/hipfire-runtime/src/weight_backend.rs",
    "crates/hipfire-runtime/src/multi_gpu.rs",
    "crates/hipfire-runtime/src/arch_model.rs",
    "crates/hipfire-runtime/src/arch.rs",
    "crates/hipfire-arch-*/src/**/load*.rs",
    "crates/hipfire-arch-*/src/**/weights*.rs",
    "crates/hipfire-arch-*/src/carrier.rs",
    "crates/hipfire-config/**",
    "crates/hipfire-registry/**",
    "registry/**",
    "Cargo.toml",
    "Cargo.lock",
    "crates/*/Cargo.toml",
]

# Paths whose change alters what the hardware job EXECUTES beyond the crate's
# own Rust/HIP: build scripts, dependency manifests, shell/python that the
# harnesses run, CI. Reported as informational input to the reviewer; not a
# hardware gate on their own.
_EXEC_SENSITIVE_PATTERNS: list[str] = [
    "**/build.rs",
    "build.rs",
    ".cargo/**",
    "Cargo.toml",
    "Cargo.lock",
    "crates/*/Cargo.toml",
    "rust-toolchain*",
    ".github/**",
    "scripts/**",
    "tools/**",
    "**/*.sh",
    "**/*.py",
    "Makefile",
    "justfile",
    "Dockerfile*",
    "flake.nix",
    "nix/**",
]

_POLICY_PATTERNS: list[str] = [
    ".github/workflows/**",
    ".github/CODEOWNERS",
    "scripts/hw-gate/**",
    "scripts/leanup-thresholds.txt",
    "scripts/layering.txt",
    "scripts/ratchet-diff.sh",
    "scripts/leanup-ratchets.sh",
    "registry/**",
]


def _normalize(p: str) -> str:
    p = p.strip()
    if not p:
        return ""
    p = p.replace("\\", "/")
    # collapse repeated slashes via normpath
    p = posixpath.normpath(p)
    if p == ".":
        return ""
    return p

_REQUEST_MARKER = "<!-- hw-gate-request -->"
_VALID_MODES = frozenset({"battery", "chain"})


def parse_pr_request(body: str) -> tuple[dict | None, str | None]:
    """Parse the first hw-gate-request block from a PR body.

    Returns (request, request_error). Never raises on body content.
    No marker => (None, None). Malformed => (None, reason).
    """
    if not body:
        return None, None
    # Normalize newlines so CRLF bodies match the same rules.
    text = body.replace("\r\n", "\n").replace("\r", "\n")
    idx = text.find(_REQUEST_MARKER)
    if idx < 0:
        return None, None
    rest = text[idx + len(_REQUEST_MARKER) :]
    # optional whitespace after the marker
    rest = rest.lstrip(" \t\n")
    fence = "```json"
    if not rest.startswith(fence):
        return None, "missing json fence after hw-gate-request marker"
    rest = rest[len(fence) :]
    # optional space on the opening fence line, then content until closing fence
    if rest.startswith("\n"):
        rest = rest[1:]
    elif rest[:1].isspace() and "\n" in rest:
        # ```json extra\n...
        rest = rest.split("\n", 1)[1]
    close = rest.find("```")
    if close < 0:
        return None, "unclosed json fence"
    raw = rest[:close].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"
    return _normalize_request(data)


def _normalize_request(data: object) -> tuple[dict | None, str | None]:
    if not isinstance(data, dict):
        return None, "request must be a JSON object"
    routes = data.get("routes")
    if not isinstance(routes, list):
        return None, "routes must be a list"
    normalized: list[dict] = []
    for i, item in enumerate(routes):
        if not isinstance(item, dict):
            return None, f"routes[{i}] must be an object"
        mode = item.get("mode")
        tag = item.get("tag")
        if mode not in _VALID_MODES:
            return None, f"routes[{i}].mode must be 'battery' or 'chain'"
        if not isinstance(tag, str) or not tag.strip():
            return None, f"routes[{i}].tag must be a non-empty string"
        normalized.append({"mode": mode, "tag": tag})
    if "claim" not in data:
        claim = ""
    else:
        claim = data["claim"]
        if not isinstance(claim, str):
            return None, "claim must be a string"
    return {"routes": normalized, "claim": claim}, None



def _matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def classify(paths: list[str]) -> dict:
    """Return the CONTRACT JSON object for `paths`. Implemented in this file."""
    buckets: set[str] = set()
    policy_paths: list[str] = []
    seen_policy: set[str] = set()
    exec_sensitive_paths: list[str] = []
    surfaces: dict[str, list[str]] = {
        "load": [],
        "serve": [],
        "kernel": [],
        "policy": [],
        "other": [],
    }
    # track seen per surface to avoid duplicates while preserving order
    seen_surfaces: dict[str, set[str]] = {k: set() for k in surfaces}

    for raw in paths:
        p = _normalize(raw)
        if not p:
            continue

        # policy is additive, checked independently
        is_policy = _matches(p, _POLICY_PATTERNS)
        if is_policy and p not in seen_policy:
            policy_paths.append(p)
            seen_policy.add(p)
        if is_policy and p not in seen_surfaces["policy"]:
            surfaces["policy"].append(p)
            seen_surfaces["policy"].add(p)

        if _matches(p, _EXEC_SENSITIVE_PATTERNS) and p not in exec_sensitive_paths:
            exec_sensitive_paths.append(p)

        # bucket: first match wins
        bucket: str | None = None
        if _matches(p, _KERNEL_PATTERNS):
            bucket = "kernel"
        elif _matches(p, _SERVE_PATTERNS):
            bucket = "serve"
        elif _matches(p, _LOAD_PATTERNS):
            bucket = "load"
        else:
            bucket = "other"

        if bucket == "kernel":
            if "kernel" not in buckets:
                buckets.add("kernel")
            if "load" not in buckets:
                buckets.add("load")
            if p not in seen_surfaces["kernel"]:
                surfaces["kernel"].append(p)
                seen_surfaces["kernel"].add(p)
            if p not in seen_surfaces["load"]:
                surfaces["load"].append(p)
                seen_surfaces["load"].add(p)
        elif bucket == "serve":
            if "serve" not in buckets:
                buckets.add("serve")
            if "load" not in buckets:
                buckets.add("load")
            if p not in seen_surfaces["serve"]:
                surfaces["serve"].append(p)
                seen_surfaces["serve"].add(p)
            if p not in seen_surfaces["load"]:
                surfaces["load"].append(p)
                seen_surfaces["load"].add(p)
        elif bucket == "load":
            if "load" not in buckets:
                buckets.add("load")
            if p not in seen_surfaces["load"]:
                surfaces["load"].append(p)
                seen_surfaces["load"].add(p)
        else:  # other
            # only record in other if not already accounted as policy-only ? policy additive
            # but policy paths that are other should be in other? Spec says policy only -> not in other
            # To satisfy "scripts/leanup-thresholds.txt -> policy only", we put policy-only
            # paths only in policy, not other.
            if not is_policy:
                if p not in seen_surfaces["other"]:
                    surfaces["other"].append(p)
                    seen_surfaces["other"].add(p)
            # if is_policy and bucket == other, we have already recorded in policy surfaces,
            # and we do NOT record in other (policy only semantics).

    sorted_buckets = sorted(buckets)
    # Policy-file changes have no fixtures of their own but must go through
    # the seats and the hard floor; a policy-only diff must never pass the
    # required status by skipping the gate.
    needs_hw = bool(sorted_buckets) or bool(policy_paths)

    return {
        "schema": "hipfire.hw-gate.select",
        "version": 1,
        "needs_hw": needs_hw,
        "buckets": sorted_buckets,
        "policy_paths": policy_paths,
        "exec_sensitive_paths": exec_sensitive_paths,
        "request": None,
        "request_error": None,
        "surfaces": surfaces,
    }

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--json", help="also write the result here")
    ap.add_argument("--github-output", help="append needs_hw=/buckets=/policy=/request_present= lines here")
    ap.add_argument("--pr-body", help="PR body file; parse first hw-gate-request JSON fence")
    args = ap.parse_args(argv)
    paths = [line.strip() for line in sys.stdin if line.strip()]
    result = classify(paths)
    if args.pr_body is not None:
        try:
            with open(args.pr_body, encoding="utf-8") as fh:
                body = fh.read()
        except OSError as exc:
            # File access is a usage/runtime issue; body content never raises.
            print(f"select.py: cannot read --pr-body: {exc}", file=sys.stderr)
            return 2
        request, request_error = parse_pr_request(body)
        result["request"] = request
        result["request_error"] = request_error
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as fh:
            fh.write(f"needs_hw={'true' if result['needs_hw'] else 'false'}\n")
            fh.write(f"buckets={','.join(result['buckets'])}\n")
            fh.write(f"policy={','.join(result['policy_paths'])}\n")
            fh.write(f"exec_sensitive={','.join(result['exec_sensitive_paths'])}\n")
            present = "true" if result.get("request") is not None else "false"
            fh.write(f"request_present={present}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
