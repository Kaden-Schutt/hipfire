#!/usr/bin/env python3
"""Generate registry/v1.json — the dynamic model registry (task #47).

Sources of truth:
  - registry/models.json : curated overlay — tags, repos, files, size_gb,
    min_vram_gb, desc, aliases. Hand-edited; stays the editing surface.
  - Hugging Face Hub API : per-file LFS sha256 + size_bytes (ground truth
    for what is actually downloadable), probed live on every run.

Output (registry/v1.json) is a STRICT SUPERSET of registry/models.json:
old CLIs that read {models, aliases} keep working unchanged; new fields are
purely additive:
  top-level : schema_version, generated_at
  per-entry : sha256 (HF LFS oid), size_bytes, arch_id, quant
  sidecars  : triattn/mtp gain sha256/size_bytes next to their `file`

Fail-closed: ANY problem — repo unreachable, file missing from the repo
tree, file not LFS (no sha256), size_bytes disagreeing with curated
size_gb, unmappable arch_id/quant, alias pointing at a missing tag, or a
superset violation — aborts with exit 1 and does NOT write output. A broken
run must never replace a good committed registry.

Namespace probe: every repo in the hipfire-models and schuttdev namespaces
is enumerated; repos that exist on HF but have no curated entry are listed
as warnings (discovery aid), never auto-added — the curated overlay is
authoritative for what the CLI offers.

Usage:
  python3 scripts/registry_gen.py                 # write registry/v1.json
  python3 scripts/registry_gen.py --check         # exit 1 if file is stale
  HF_TOKEN=hf_xxx python3 scripts/registry_gen.py # authenticated (rate limits)

stdlib only — no pip installs needed in CI.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HF_API = "https://huggingface.co"
PROBE_NAMESPACES = ("hipfire-models", "schuttdev")
SCHEMA_VERSION = 1
# Curated size_gb is a rounded decimal-GB figure; the HF byte count is ground
# truth. Disagreement beyond this fraction means the curated entry is stale
# (wrong file / re-quantized upload) — fail so a human reconciles it.
SIZE_TOLERANCE = 0.25
# Known weight-file quant suffixes (see docs/MODELS.md). Anything else is an
# error: a new format must be added here deliberately, not silently passed.
KNOWN_QUANTS = {
    "mq2lloyd",
    "mq2",
    "mq2r",
    "mq3",
    "mq3p",
    "mq4",
    "mq4p",
    "mq4r",
    "mq5",
    "mq6",
    "mfp4",
    "hf4",
    "hf6",
    "q8",
    "hfq",
}
# Allowlist for the optional per-entry `default_kv_mode` field (the registry is
# the per-model card). MUST stay in sync with the hipfire-config schema and
# hipfire-registry parser. A curated
# entry carrying an unknown value fails the run (fail-closed, like arch_id/quant).
KNOWN_KV_MODES = {
    "auto",
    "q8",
    "asym4",
    "asym3",
    "asym2",
    "fwht4",
    "fwht3",
    "fwht2",
    "turbo",
    "turbo4",
    "turbo3",
    "turbo2",
}
KNOWN_TOOL_FORMATS = {"hermes", "qwen_xml"}

# Bounds for the optional curated `recommended_settings` (author-recommended
# inference settings inherited from the parent model card). MUST stay in sync
# with hipfire-registry's recommended-settings validation. Each present numeric
# knob is range-checked; an out-of-range value fails the run (fail-closed).
#   (lo, hi, int_only)
RECOMMENDED_BOUNDS = {
    "temperature": (0.0, 2.0, False),
    "top_p": (0.0, 1.0, False),
    "top_k": (1.0, 100000.0, True),
    "min_p": (0.0, 1.0, False),
    "presence_penalty": (0.0, 2.0, False),
    "repeat_penalty": (0.5, 2.0, False),
}
REASONING_EFFORTS = {"auto", "none", "low", "high", "max"}
THINKING_BUDGETS = {"off", "low", "med", "high", "xhigh", "max", "uncapped"}


def validate_recommended_settings(tag: str, rs: object, errors: list) -> None:
    """Carry-through validator for `recommended_settings` (verbatim in v1.json).

    Mirrors hipfire-registry's recommended-settings bounds. Adds a
    descriptive error per offending key; does not mutate rs (deepcopy already
    carries it through)."""
    if rs is None:
        return
    if not isinstance(rs, dict):
        errors.append(f"{tag}: recommended_settings must be an object, got {type(rs).__name__}")
        return
    allowed = set(RECOMMENDED_BOUNDS) | {
        "system_prompt",
        "reasoning_effort",
        "thinking_budget",
    }
    for key, val in rs.items():
        if key not in allowed:
            errors.append(f"{tag}: recommended_settings has unknown key {key!r} (allowed: {sorted(allowed)})")
            continue
        if key == "system_prompt":
            if not isinstance(val, str):
                errors.append(f"{tag}: recommended_settings.system_prompt must be a string")
            continue
        if key == "reasoning_effort":
            if val not in REASONING_EFFORTS:
                errors.append(
                    f"{tag}: recommended_settings.reasoning_effort must be one of "
                    f"{sorted(REASONING_EFFORTS)}, got {val!r}"
                )
            continue
        if key == "thinking_budget":
            if val not in THINKING_BUDGETS:
                errors.append(
                    f"{tag}: recommended_settings.thinking_budget must be one of "
                    f"{sorted(THINKING_BUDGETS)}, got {val!r}"
                )
            continue
        lo, hi, int_only = RECOMMENDED_BOUNDS[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            errors.append(f"{tag}: recommended_settings.{key} must be a number, got {val!r}")
            continue
        if int_only and not float(val).is_integer():
            errors.append(f"{tag}: recommended_settings.{key} must be an integer, got {val!r}")
            continue
        if not (lo <= val <= hi):
            errors.append(f"{tag}: recommended_settings.{key}={val} out of range [{lo}, {hi}]")


REPO_ROOT = Path(__file__).resolve().parent.parent
CURATED_PATH = REPO_ROOT / "registry" / "models.json"
OUTPUT_PATH = REPO_ROOT / "registry" / "v1.json"


def log(msg: str) -> None:
    print(f"[registry_gen] {msg}", file=sys.stderr)


# ─── arch_id mapping (docs/architecture-ids.md) ──────────────────────────
#
# Derived from the tag family + file name. Unknown families return None and
# fail the run: every new model family must be mapped here explicitly.
#   1  = plain Qwen3 (llama-crate config_from_hfq branch)
#   5  = Qwen3.5/3.6 dense hybrid (incl. carnice / qwopus finetunes)
#   6  = Qwen3.5/3.6 MoE / A3B
#   9  = DeepSeek V4 Flash
#   11 = LFM2.5 family
#   12 = Cohere2-MoE / North-Mini-Code
#   20 = DFlash drafter sidecar (crates/hipfire-quantize/src/bin/dflash_convert.rs)
def arch_id_for(tag: str, entry: dict) -> int | None:
    file = entry.get("file", "")
    if "dflash" in file:
        return 20
    family = tag.split(":", 1)[0]
    if family in ("qwen3.5", "qwen3.6", "qwopus3.6", "carnice", "qwopus"):
        return 6 if "a3b" in tag else 5
    if family == "nex-n2":
        return 6  # Nex-N2-mini = Qwen3.5-35B-A3B MoE (a3b not in tag name)
    if family == "qwen3":
        return 1
    # Dated checkpoints (…-0731) are the same architecture, so match the prefix
    # rather than adding a new arm per release.
    if family == "deepseek-v4-flash" or family.startswith("deepseek-v4-flash-"):
        return 9
    if family == "minimax" or family.startswith("minimax-"):
        return 10
    if family == "lfm2.5":
        return 11
    if family == "north-mini-code":
        return 12
    if family == "vibethinker":
        return 7   # Qwen2 dense (WeiboAI/VibeThinker-3B base)
    return None


def quant_for(file: str) -> str | None:
    # DFlash drafts encode their quant in the stem: qwen35-9b-dflash-mq4.hfq
    m = re.search(r"[-.](mq\d)\.hfq$", file)
    if m:
        return m.group(1)
    ext = file.rsplit(".", 1)[-1]
    if ext in KNOWN_QUANTS:
        return ext
    return None


# ─── HF API ───────────────────────────────────────────────────────────────


def hf_get(url: str, token: str | None, retries: int = 3) -> tuple[object, dict]:
    """GET a HF API URL → (parsed JSON, response headers). Retries transient errors."""
    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": "hipfire-registry-gen/1"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp), dict(resp.headers)
        except urllib.error.HTTPError as e:
            # 4xx (other than 429) won't improve on retry.
            if e.code != 429 and 400 <= e.code < 500:
                raise
            last_err = e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
        time.sleep(2 ** attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")


def repo_tree(repo: str, token: str | None) -> dict[str, dict]:
    """Full recursive file listing of a model repo → {path: tree-entry}."""
    url = f"{HF_API}/api/models/{repo}/tree/main?recursive=true&limit=1000"
    files: dict[str, dict] = {}
    while url:
        items, headers = hf_get(url, token)
        for item in items:
            if item.get("type") == "file":
                files[item["path"]] = item
        # Cursor pagination via the Link header (large repos).
        link = headers.get("Link") or headers.get("link") or ""
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return files


def list_namespace_repos(namespace: str, token: str | None) -> list[str]:
    items, _ = hf_get(f"{HF_API}/api/models?author={namespace}&limit=1000", token)
    return [m["id"] for m in items]


# ─── validation helpers ───────────────────────────────────────────────────


def is_strict_superset(old: object, new: object, path: str, errors: list[str]) -> None:
    """Every key/value in `old` must appear identically in `new` (dicts recurse)."""
    if isinstance(old, dict):
        if not isinstance(new, dict):
            errors.append(f"superset violation at {path}: dict replaced by {type(new).__name__}")
            return
        for k, v in old.items():
            if k not in new:
                errors.append(f"superset violation at {path}.{k}: key dropped")
            else:
                is_strict_superset(v, new[k], f"{path}.{k}", errors)
    elif old != new:
        errors.append(f"superset violation at {path}: {old!r} != {new!r}")


def annotate_sidecar(
    sidecar: dict, tree: dict[str, dict], tag: str, kind: str, errors: list[str]
) -> dict:
    """triattn/mtp sub-object: require existence, add sha256/size_bytes if LFS."""
    out = dict(sidecar)
    fname = sidecar.get("file", "")
    item = tree.get(fname)
    if item is None:
        errors.append(f"{tag}: {kind} sidecar {fname!r} not found in repo tree")
        return out
    lfs = item.get("lfs")
    if lfs:
        out["sha256"] = lfs["oid"]
        out["size_bytes"] = lfs.get("size", item.get("size"))
    else:
        # Tiny non-LFS sidecars have no content sha256 on the HF API; record
        # size only. (All current sidecars are LFS — this is belt-and-braces.)
        out["size_bytes"] = item.get("size")
    return out


# ─── main build ───────────────────────────────────────────────────────────


def build_registry(curated: dict, token: str | None) -> tuple[dict | None, list[str]]:
    errors: list[str] = []
    models: dict = curated.get("models", {})
    aliases: dict = curated.get("aliases", {})

    # Alias integrity first — cheap and catches curated typos.
    for alias, target in aliases.items():
        if target not in models:
            errors.append(f"alias {alias!r} points at missing tag {target!r}")

    # One tree fetch per unique repo.
    repos = sorted({e["repo"] for e in models.values() if e.get("repo")})
    trees: dict[str, dict[str, dict]] = {}
    for repo in repos:
        try:
            trees[repo] = repo_tree(repo, token)
            log(f"probed {repo}: {len(trees[repo])} files")
        except Exception as e:  # noqa: BLE001 — collected, run fails closed
            errors.append(f"repo {repo}: tree probe failed: {e}")

    out_models: dict = {}
    for tag, entry in models.items():
        new_entry = copy.deepcopy(entry)

        arch_id = arch_id_for(tag, entry)
        if arch_id is None:
            errors.append(f"{tag}: no arch_id mapping — add its family to arch_id_for()")
        quant = quant_for(entry.get("file", ""))
        if quant is None:
            errors.append(f"{tag}: unknown quant for file {entry.get('file')!r}")

        # Optional per-model default_kv_mode (carried through verbatim by the
        # deepcopy above). Validate against the KV allowlist — fail-closed.
        kv_default = entry.get("default_kv_mode")
        if kv_default is not None and kv_default not in KNOWN_KV_MODES:
            errors.append(
                f"{tag}: invalid default_kv_mode {kv_default!r} "
                f"(allowed: {sorted(KNOWN_KV_MODES)})"
            )

        tool_format = entry.get("default_tool_format")
        if tool_format is not None and tool_format not in KNOWN_TOOL_FORMATS:
            errors.append(
                f"{tag}: invalid default_tool_format {tool_format!r} "
                f"(allowed: {sorted(KNOWN_TOOL_FORMATS)})"
            )

        # Optional curated recommended_settings (carried verbatim by deepcopy).
        # Validate bounds — fail-closed, matching hipfire-registry.
        validate_recommended_settings(tag, entry.get("recommended_settings"), errors)

        # Optional per-mode sampling_profiles (carried verbatim by deepcopy).
        # Validate each present profile with the same bounds — fail-closed,
        # matching hipfire-registry's SamplingProfiles parser.
        profiles = entry.get("sampling_profiles")
        if profiles is not None:
            if not isinstance(profiles, dict):
                errors.append(f"{tag}: sampling_profiles must be an object")
            else:
                for mode, rs in profiles.items():
                    if mode not in ("general", "coding", "instruct"):
                        errors.append(
                            f"{tag}: sampling_profiles has unknown mode {mode!r} "
                            f"(allowed: ['coding', 'general', 'instruct'])"
                        )
                        continue
                    validate_recommended_settings(f"{tag} sampling_profiles.{mode}", rs, errors)

        repo = entry.get("repo", "")
        if not repo:
            # Local-only entry (pull short-circuits). A campaign artifact may
            # still carry a curated content identity even though there is no
            # Hugging Face tree to probe. Preserve it verbatim and validate it
            # below through the generated registry parser.
            new_entry.update(
                {
                    "sha256": entry.get("sha256"),
                    "size_bytes": entry.get("size_bytes"),
                }
            )
        elif repo in trees:
            tree = trees[repo]
            item = tree.get(entry["file"])
            if item is None:
                errors.append(f"{tag}: file {entry['file']!r} not found in {repo}")
            else:
                lfs = item.get("lfs")
                if not lfs:
                    errors.append(f"{tag}: {entry['file']!r} in {repo} is not LFS — no sha256")
                else:
                    size_bytes = lfs.get("size", item.get("size"))
                    new_entry["sha256"] = lfs["oid"]
                    new_entry["size_bytes"] = size_bytes
                    curated_gb = entry.get("size_gb")
                    if isinstance(curated_gb, (int, float)) and curated_gb > 0:
                        drift = abs(size_bytes / 1e9 - curated_gb) / curated_gb
                        if drift > SIZE_TOLERANCE:
                            errors.append(
                                f"{tag}: size mismatch — curated {curated_gb} GB vs "
                                f"HF {size_bytes / 1e9:.2f} GB ({drift:.0%} drift); "
                                f"update registry/models.json"
                            )
            for kind in ("triattn", "mtp"):
                if isinstance(entry.get(kind), dict):
                    new_entry[kind] = annotate_sidecar(entry[kind], tree, tag, kind, errors)
        # repo probe already failed → error recorded above; entry still gets
        # arch_id/quant so the error list is the only blocker.

        new_entry["arch_id"] = arch_id
        new_entry["quant"] = quant
        out_models[tag] = new_entry

    registry = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "_comment": (
            "GENERATED by scripts/registry_gen.py — do not hand-edit. "
            "Edit registry/models.json (curated overlay) and re-run the generator. "
            "Strict superset of registry/models.json: models/aliases keep the curated "
            "shape; sha256/size_bytes come from the HF LFS API; arch_id per "
            "docs/architecture-ids.md; min_vram_gb gates pull/run on VRAM."
        ),
        "models": out_models,
        "aliases": dict(aliases),
    }

    # Strict-superset guarantee — the whole point of v1 back-compat.
    is_strict_superset(curated.get("models", {}), registry["models"], "models", errors)
    is_strict_superset(curated.get("aliases", {}), registry["aliases"], "aliases", errors)

    if errors:
        return None, errors
    return registry, []


def strip_generated_at(reg: dict) -> dict:
    out = dict(reg)
    out.pop("generated_at", None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUTPUT_PATH)
    ap.add_argument("--curated", type=Path, default=CURATED_PATH)
    ap.add_argument(
        "--check",
        action="store_true",
        help="don't write; exit 1 if the committed file differs from a fresh build",
    )
    args = ap.parse_args()

    import os

    token = os.environ.get("HF_TOKEN") or None

    curated = json.loads(args.curated.read_text())

    # Discovery aid: namespace repos with no curated entry (warn-only).
    curated_repos = {e["repo"] for e in curated.get("models", {}).values() if e.get("repo")}
    for ns in PROBE_NAMESPACES:
        try:
            for repo in list_namespace_repos(ns, token):
                if repo not in curated_repos:
                    log(f"note: {repo} exists on HF but has no curated entry (skipped)")
        except Exception as e:  # noqa: BLE001 — discovery is best-effort
            log(f"warning: could not enumerate namespace {ns}: {e}")

    registry, errors = build_registry(curated, token)
    if registry is None:
        log(f"FAILED — {len(errors)} error(s), output NOT written:")
        for e in errors:
            log(f"  - {e}")
        return 1

    # Keep generated_at stable when nothing else changed, so the cron
    # workflow's commit-on-diff stays quiet on no-op days.
    old: dict | None = None
    if args.out.exists():
        try:
            old = json.loads(args.out.read_text())
        except json.JSONDecodeError:
            old = None
    if old is not None and strip_generated_at(old) == strip_generated_at(registry):
        registry["generated_at"] = old.get("generated_at", registry["generated_at"])

    rendered = json.dumps(registry, indent=2) + "\n"

    if args.check:
        current = args.out.read_text() if args.out.exists() else ""
        if current != rendered:
            log("STALE — registry/v1.json differs from a fresh build")
            return 1
        log("up to date")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    log(f"wrote {args.out} ({len(registry['models'])} models, {len(registry['aliases'])} aliases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
