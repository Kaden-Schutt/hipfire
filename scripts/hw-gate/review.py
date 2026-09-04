#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""hw-gate reviewer driver: Sol prelim/verdict and Fable decide with staging merges.

Two seats, one hardware gate, one human owner.
- Sol (reviewer seat) reads diff and DECIDES whether PR code runs on hardware (run_hardware) and which routes run.
- Sol verdict after hardware: greenlight | needs-human | block (never merges).
- Fable (deciding seat) reads everything inc Sol verdict and returns merge-staging | hold | block, may veto/override Sol,
  and on merge-staging merges PR head into staging branch (beta) via GitHub merges API. master stays human-owned.

Hard floor (no seat overrides): failed fixture/harness, attractor, policy-file change, RATCHET-RAISE without label.
Soft floor (Fable may override): coverage gaps, confidence <0.8, Sol needs-human.

CONTRACT
    prelim: review.py --seat sol --phase prelim --repo R --pr N --base SHA --head SHA --checkout DIR --select select.json --fixtures fixtures.json --system-prompt sol.md --out prelim.json --routes routes.json
    verdict: review.py --seat sol --phase verdict --repo R --pr N --base SHA --head SHA --checkout DIR --select select.json --prelim prelim.json --evidence hw-gate.json --hw-run-result success|failure|... --system-prompt sol.md --out verdict.json
    decide: review.py --seat fable --phase decide --repo R --pr N --base SHA --head SHA --checkout DIR --select select.json --prelim prelim.json --evidence hw-gate.json --verdict verdict.json --hw-run-result ... --staging beta --system-prompt fable.md --out decision.json
    env: HW_GATE_REVIEW_MODEL (default gpt-5.6-sol) for sol, HW_GATE_DECIDE_MODEL (default claude-fable-5-1) for fable,
         HIPFIRE_MODELS_DIR, HW_GATE_OMP_BIN, HW_GATE_GH_BIN.
    prelim.json: {"schema":"hipfire.hw-gate.prelim","version":2,"seat":"sol","model":..,"prelim":{...}|null,"run_hardware":bool,"posted":{...}}
    routes.json: [{"mode":"battery"|"chain","tag":"registry:tag","source":"bucket"|"author"|"sol","why":".."}]
    verdict.json: {"schema":"hipfire.hw-gate.verdict","version":2,"seat":"sol","model":..,"verdict":{...}|null,"floor":{"hard":[..],"soft":[..],"model_decision":..,"final_decision":..},"posted":{...}}
    decision.json: {"schema":"hipfire.hw-gate.decision","version":1,"seat":"fable","model":..,"decision":{...}|null,"floor":{"hard":[..],"soft":[..]},"decision_final":"merge-staging|hold|block","override":null|{...},"merged":null|{...},"posted":{...}}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


class ReviewError(Exception):
    pass


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    """Return the last balanced top-level JSON object in assistant text, or None."""
    if not text:
        return None
    candidates: list[str] = []
    depth = 0
    current_start: int | None = None
    in_str = False
    esc = False
    for idx, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    current_start = idx
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and current_start is not None:
                        candidates.append(text[current_start: idx + 1])
                        current_start = None
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# hard_floor / soft_floor
# ---------------------------------------------------------------------------

def hard_floor(
    evidence: dict | None,
    select: dict,
    hw_run_result: str,
    commit_messages: list[str],
) -> list[str]:
    """Hard floor: non-overridable. Returns list of reason strings."""
    reasons: list[str] = []
    # hw_run_result
    if hw_run_result != "success":
        reasons.append(f"hw_run_result={hw_run_result}")
    # evidence missing or verdict != pass (failed fixture/harness)
    if evidence is None:
        reasons.append("evidence missing")
    elif evidence.get("verdict") != "pass":
        reasons.append(f"evidence verdict={evidence.get('verdict')!r}")
    # kernel bucket check
    buckets = select.get("buckets", []) if isinstance(select, dict) else []
    if "kernel" in buckets:
        kernel = None
        if isinstance(evidence, dict):
            kernel = evidence.get("kernel")
        if kernel is None or (isinstance(kernel, dict) and kernel.get("status") != "pass"):
            reasons.append("kernel status != pass")
    # attractor detection
    if isinstance(evidence, dict):
        fixtures = evidence.get("fixtures", [])
        if isinstance(fixtures, list):
            for fx in fixtures:
                if not isinstance(fx, dict):
                    continue
                modes = fx.get("modes", {})
                if not isinstance(modes, dict):
                    continue
                for mode_data in modes.values():
                    if not isinstance(mode_data, dict):
                        continue
                    rows = mode_data.get("rows", [])
                    if not isinstance(rows, list):
                        continue
                    for row in rows:
                        if isinstance(row, dict) and row.get("attractor"):
                            reasons.append("attractor detected")
                            break
                    if "attractor detected" in reasons:
                        break
                if "attractor detected" in reasons:
                    break
    # policy
    policy_paths = select.get("policy_paths", []) if isinstance(select, dict) else []
    if policy_paths:
        reasons.append(f"policy_paths: {','.join(policy_paths)}")
    # RATCHET-RAISE
    for msg in commit_messages or []:
        if re.match(r"^RATCHET-RAISE:", msg):
            reasons.append("RATCHET-RAISE without ratchet-raise label")
            break
    return reasons


def soft_floor(
    verdict: dict | None,
    model_decision: str | None,
) -> list[str]:
    """Soft floor: Fable may override."""
    reasons: list[str] = []
    if isinstance(verdict, dict):
        coverage = verdict.get("coverage", {})
        if isinstance(coverage, dict):
            gaps = coverage.get("gaps", [])
            if gaps:
                reasons.append(f"coverage_gaps: {gaps}")
        conf = verdict.get("confidence")
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) or conf < 0.8:
            reasons.append(f"confidence {conf!r} < 0.8")
    # model needs-human is soft
    if model_decision == "needs-human":
        reasons.append("model needs-human")
    # verdict parse failure counted elsewhere? but treat as soft
    if verdict is None and model_decision is None:
        # will be handled as needs-human
        pass
    return reasons


def apply_floor(
    model_decision: str | None,
    evidence: dict | None,
    select: dict,
    hw_run_result: str,
    commit_messages: list[str],
    verdict: dict | None,
) -> tuple[str, list[str]]:
    """Legacy wrapper: combines hard+soft for old tests. Returns (final, reasons)."""
    hard = hard_floor(evidence, select, hw_run_result, commit_messages)
    # Determine if hard contains evidence-type vs policy
    def _is_hold_reason(r: str) -> bool:
        return "policy_paths" in r or "RATCHET-RAISE" in r
    hard_has_block = any(not _is_hold_reason(r) for r in hard)
    hard_has_hold = any(_is_hold_reason(r) for r in hard)
    soft = soft_floor(verdict, model_decision)
    # verdict parse failure
    if verdict is None and model_decision is None:
        # treat as needs-human reason
        soft.append("verdict_parse_failed") if "verdict_parse_failed" not in soft else None
    if model_decision == "needs-human" and "model needs-human" not in soft and verdict is not None:
        # already covered
        pass
    if hard:
        if hard_has_block:
            return ("block", hard + soft)
        else:
            return ("needs-human", hard + soft)
    if soft:
        return ("needs-human", soft)
    if model_decision == "greenlight":
        return ("greenlight", [])
    if model_decision == "block":
        return ("block", ["model_block"])
    if verdict is None:
        return ("needs-human", ["verdict_parse_failed"])
    return ("needs-human", ["model_decision_not_greenlight"])


# ---------------------------------------------------------------------------
# helpers: gh seam and omp
# ---------------------------------------------------------------------------

def _gh(args: list[str]) -> str:
    gh_bin = os.environ.get("HW_GATE_GH_BIN", "gh")
    cmd = [gh_bin] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ReviewError(f"gh {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def omp_review(phase: str, prompt: str, system_prompt: str, checkout: str, model: str) -> dict:
    """Run omp for a phase and return the extracted JSON object."""
    omp_bin = os.environ.get("HW_GATE_OMP_BIN", "omp")
    last_error: str | None = None
    for attempt in range(2):
        cur_prompt = prompt if attempt == 0 else prompt + "\n\nReturn only the JSON object."
        with tempfile.NamedTemporaryFile("w", suffix=f"-hw-gate-{phase}.md", delete=False, encoding="utf-8") as fh:
            fh.write(cur_prompt)
            prompt_path = fh.name
        cmd = [
            omp_bin, "-p", "--mode", "json", "--auto-approve",
            "--tools=read,grep,glob",
            "--cwd", checkout,
            "--model", model,
            "--system-prompt", system_prompt,
            "--max-time", "15m",
            f"@{prompt_path}",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=checkout)
        finally:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass
        if result.returncode != 0:
            last_error = f"omp {phase} failed ({result.returncode}): {result.stderr.strip()}"
            if attempt == 0:
                continue
            raise ReviewError(last_error)
        stdout = result.stdout
        last_text: str | None = None
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except Exception:
                continue
            if evt.get("type") == "message_end":
                msg = evt.get("message", {})
                if msg.get("role") == "assistant":
                    parts = msg.get("content", [])
                    texts = []
                    for p in parts:
                        if isinstance(p, dict) and p.get("type") == "text":
                            texts.append(p.get("text", ""))
                    last_text = "".join(texts)
        if last_text is None:
            last_error = f"omp {phase}: no assistant message_end in output"
            if attempt == 0:
                continue
            raise ReviewError(last_error)
        obj = extract_json(last_text)
        if obj is None:
            last_error = f"omp {phase}: no JSON object in assistant text: {last_text[:500]!r}"
            if attempt == 0:
                continue
            raise ReviewError(last_error)
        return obj
    raise ReviewError(last_error or f"omp {phase} failed after retry")
def _is_credential_env_key(key: str) -> bool:
    """Whether env key looks like a credential and must be stripped."""
    if key in ("GH_TOKEN", "GITHUB_TOKEN"):
        return True
    if key.endswith("_TOKEN") or key.endswith("_KEY") or key.endswith("_SECRET"):
        return True
    # HW_GATE_*_PRIVATE_KEY and HW_GATE_*_APP_ID are already covered by _KEY / _APP_ID,
    # but be explicit for APP_ID which ends with _APP_ID not _ID.
    if key.startswith("HW_GATE_") and (key.endswith("_PRIVATE_KEY") or key.endswith("_APP_ID")):
        return True
    return False


def _build_investigate_env(args) -> tuple[dict, str]:
    """Build sandboxed env for omp investigate mode. Returns (env_dict, max_minutes_str)."""
    devices = getattr(args, "devices", None)
    if not devices:
        raise ReviewError("missing --devices for --investigate (HW_GATE_DEVICES)")
    home = getattr(args, "home", None)
    if not home:
        raise ReviewError("missing --home for --investigate (HIPFIRE_HOME)")
    evidence_dir = getattr(args, "evidence_dir", None)
    if not evidence_dir:
        raise ReviewError("missing --evidence-dir for --investigate (HW_GATE_EVIDENCE)")
    bin_path = getattr(args, "bin", None)
    if not bin_path:
        raise ReviewError("missing --bin for --investigate (HW_GATE_BIN)")
    models_dir = os.environ.get("HIPFIRE_MODELS_DIR")
    if not models_dir:
        raise ReviewError("missing HIPFIRE_MODELS_DIR for --investigate")
    base_bin = getattr(args, "base_bin", None)
    round_val = getattr(args, "round", 1)
    if round_val is None:
        round_val = 1
    max_minutes = os.environ.get("HW_GATE_MAX_MINUTES", "45")
    # create dirs
    try:
        Path(home).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ReviewError(f"failed to create HIPFIRE_HOME {home}: {e}")
    try:
        Path(evidence_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ReviewError(f"failed to create HW_GATE_EVIDENCE {evidence_dir}: {e}")
    child: dict[str, str] = {}
    # allow-list: PATH, HOME, LANG, TERM, USER, ROCM_PATH
    for k in ("PATH", "HOME", "LANG", "TERM", "USER", "ROCM_PATH"):
        if k in os.environ:
            child[k] = os.environ[k]
    # ROCm/HIP vars
    for k, v in os.environ.items():
        if k.startswith("HSA_") or k.startswith("HIP_"):
            child[k] = v
    # HW_GATE_* and HIPFIRE_*
    for k, v in os.environ.items():
        if k.startswith("HW_GATE_") or k.startswith("HIPFIRE_"):
            child[k] = v
    # Propagate FAKE_OMP_* / FAKE_GH_* for test harness (not part of allow-list but needed for tests)
    for k, v in os.environ.items():
        if k.startswith("FAKE_OMP_") or k.startswith("FAKE_GH_"):
            child[k] = v
    # required overrides
    child["HW_GATE_DEVICES"] = devices
    child["HIP_VISIBLE_DEVICES"] = devices
    child["HIPFIRE_MODELS_DIR"] = models_dir
    child["HIPFIRE_HOME"] = home
    child["HW_GATE_EVIDENCE"] = evidence_dir
    child["HW_GATE_BIN"] = bin_path
    if base_bin:
        child["HW_GATE_BASE_BIN"] = base_bin
    else:
        child.pop("HW_GATE_BASE_BIN", None)
    child["HW_GATE_BASE_SHA"] = args.base
    child["HW_GATE_ROUND"] = str(round_val)
    child["HW_GATE_MAX_MINUTES"] = str(max_minutes)
    # strip credentials
    for k in list(child.keys()):
        if _is_credential_env_key(k):
            child.pop(k, None)
    return child, str(max_minutes)


def omp_investigate(prompt: str, system_prompt: str, checkout: str, model: str, child_env: dict, max_minutes: str):
    """Run Fable's investigation session: full tool set (no --tools), thinking xhigh,
    a wall-clock budget, and ONLY the sandboxed env. Returns the CompletedProcess;
    raises subprocess.TimeoutExpired past the budget (plus a grace minute)."""
    omp_bin = os.environ.get("HW_GATE_OMP_BIN", "omp")
    with tempfile.NamedTemporaryFile("w", suffix="-hw-gate-decide.md", delete=False, encoding="utf-8") as fh:
        fh.write(prompt)
        prompt_path = fh.name
    cmd = [
        omp_bin, "-p", "--mode", "json", "--auto-approve",
        "--cwd", checkout,
        "--model", model,
        "--system-prompt", system_prompt,
        "--max-time", f"{max_minutes}m",
        "--thinking", "xhigh",
        f"@{prompt_path}",
    ]
    try:
        minutes = int(str(max_minutes).strip().rstrip("m"))
    except ValueError:
        minutes = 45
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=checkout, env=child_env, timeout=max(1, minutes + 1) * 60)
    finally:
        try:
            os.unlink(prompt_path)
        except OSError:
            pass


def _parse_omp_stdout(stdout: str) -> tuple[str | None, dict | None]:
    """Extract last assistant text and parsed JSON object from omp JSONL stdout."""
    last_text: str | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except Exception:
            continue
        if evt.get("type") == "message_end":
            msg = evt.get("message", {})
            if msg.get("role") == "assistant":
                parts = msg.get("content", [])
                texts: list[str] = []
                for p in parts:
                    if isinstance(p, dict) and p.get("type") == "text":
                        texts.append(p.get("text", ""))
                last_text = "".join(texts)
    if last_text is None:
        return None, None
    obj = extract_json(last_text)
    return last_text, obj


def _list_registry_fixtures(checkout: str, models_dir: str) -> list[tuple[str, str]]:
    """List (tag, file) for registry models present under models_dir."""
    registry_path = Path(checkout) / "registry" / "v1.json"
    if not registry_path.is_file():
        return []
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    models = data.get("models", {}) if isinstance(data, dict) else {}
    if not isinstance(models, dict):
        return []
    present: list[tuple[str, str]] = []
    base = Path(models_dir).expanduser()
    for tag, info in models.items():
        if not isinstance(info, dict):
            continue
        f = info.get("file")
        if not f:
            continue
        p = base / f
        try:
            if p.is_file():
                present.append((tag, f))
        except Exception:
            continue
    # also handle aliases? spec says tag -> file, so models keys already tags.
    present.sort(key=lambda x: x[0])
    return present


def _build_investigate_prompt_addon(checkout: str, child_env: dict, registry_present: list[tuple[str, str]]) -> str:
    """Build prompt addon that tells Fable sandbox values and available fixtures."""
    lines: list[str] = []
    lines.append("Sandbox for this investigation (exact env values for this session):")
    # order per spec
    for key in ("HW_GATE_DEVICES", "HIP_VISIBLE_DEVICES", "HIPFIRE_MODELS_DIR", "HIPFIRE_HOME", "HW_GATE_EVIDENCE", "HW_GATE_BIN", "HW_GATE_BASE_BIN", "HW_GATE_BASE_SHA", "HW_GATE_ROUND", "HW_GATE_MAX_MINUTES"):
        if key in child_env:
            lines.append(f"- {key}={child_env[key]}")
        elif key == "HW_GATE_BASE_BIN":
            lines.append(f"- {key}=(not set)")
        else:
            lines.append(f"- {key}=(missing)")
    lines.append(f"- checkout cwd={checkout}")
    lines.append("")
    lines.append("Available registry fixtures present under $HIPFIRE_MODELS_DIR (tag -> file):")
    if registry_present:
        for tag, f in registry_present:
            lines.append(f"- {tag} -> {f}")
    else:
        lines.append("(none present)")
    return "\n".join(lines)




def _git(args: list[str], checkout: str) -> str:
    result = subprocess.run(["git"] + args, capture_output=True, text=True, cwd=checkout)
    if result.returncode != 0:
        raise ReviewError(f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


# ---------------------------------------------------------------------------
# fixtures helpers
# ---------------------------------------------------------------------------

def _get_models_dir(fixtures_manifest: dict) -> Path:
    env_dir = os.environ.get("HIPFIRE_MODELS_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    manifest_dir = fixtures_manifest.get("models_dir", "~/.hipfire/models") if isinstance(fixtures_manifest, dict) else "~/.hipfire/models"
    return Path(manifest_dir).expanduser()


def _available_fixtures(fixtures_manifest: dict) -> tuple[dict, Path]:
    models_dir = _get_models_dir(fixtures_manifest)
    available: dict[str, dict] = {}
    for fx in fixtures_manifest.get("fixtures", []) if isinstance(fixtures_manifest.get("fixtures"), list) else []:
        tag = fx.get("tag")
        file = fx.get("file")
        if not tag or not file:
            continue
        p = models_dir / file
        if p.is_file():
            available[tag] = fx
    return available, models_dir


def _modes_for_buckets(buckets: list[str], manifest: dict) -> list[str]:
    modes_set: set[str] = set()
    for b in buckets:
        cfg = manifest.get("buckets", {}).get(b, {}) if isinstance(manifest.get("buckets"), dict) else {}
        if isinstance(cfg, dict):
            modes = cfg.get("modes", [])
            if isinstance(modes, list):
                for m in modes:
                    modes_set.add(m)
    ordered: list[str] = []
    for m in ["battery", "chain"]:
        if m in modes_set:
            ordered.append(m)
    for m in sorted(modes_set):
        if m not in ordered:
            ordered.append(m)
    return ordered


def _bucket_routes(manifest: dict, buckets: list[str]) -> list[dict]:
    modes = _modes_for_buckets(buckets, manifest)
    routes: list[dict] = []
    for fx in manifest.get("fixtures", []) if isinstance(manifest.get("fixtures"), list) else []:
        tag = fx.get("tag")
        if not tag:
            continue
        for mode in modes:
            routes.append({"mode": mode, "tag": tag, "source": "bucket", "why": f"bucket {','.join(buckets)}"})
    return routes


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def build_prelim_prompt(select: dict, checkout: str, base: str, head: str, repo: str, pr: int, fixtures_manifest: dict | None = None, available_tags: list[str] | None = None) -> str:
    pr_json_text = _gh(["pr", "view", str(pr), "--repo", repo, "--json", "title,body,author,url"])
    pr_info = json.loads(pr_json_text)
    title = pr_info.get("title", "")
    body = pr_info.get("body", "") or ""
    author = pr_info.get("author", {})
    url = pr_info.get("url", "")

    diff_stat = _git(["diff", "--stat", f"{base}...{head}"], checkout)
    surfaces = select.get("surfaces", {}) if isinstance(select, dict) else {}
    priority_paths: list[str] = []
    for bucket in ("load", "serve", "kernel", "policy"):
        paths = surfaces.get(bucket, []) if isinstance(surfaces, dict) else []
        priority_paths.extend(paths)

    diff_text = _git(["diff", f"{base}...{head}"], checkout)
    cap = 400 * 1024
    if len(diff_text.encode("utf-8")) > cap:
        included_parts: list[str] = []
        included_size = 0
        emitted_files: set[str] = set()
        for fpath in priority_paths:
            if fpath in emitted_files:
                continue
            try:
                part = _git(["diff", f"{base}...{head}", "--", fpath], checkout)
            except ReviewError:
                part = ""
            if not part:
                continue
            part_bytes = len(part.encode("utf-8"))
            if included_size + part_bytes > cap:
                continue
            included_parts.append(part)
            included_size += part_bytes
            emitted_files.add(fpath)
        if included_parts:
            truncated_note = f"\n\n[diff truncated: {len(diff_text.encode('utf-8'))} bytes total, showing {included_size} bytes prioritized for {len(emitted_files)} surface files; omitted rest]"
            if included_size < cap:
                needed = cap - included_size - len(truncated_note.encode("utf-8"))
                full_bytes = diff_text.encode("utf-8")
                filler = full_bytes[:needed].decode("utf-8", errors="ignore") if needed > 0 else ""
                combined = "\n".join(included_parts) + "\n" + filler if filler else "\n".join(included_parts)
                diff_text = combined + truncated_note
            else:
                diff_text = "\n".join(included_parts) + truncated_note
        else:
            eb = diff_text.encode("utf-8")
            truncated = eb[:cap].decode("utf-8", errors="ignore")
            omitted = len(eb) - cap
            diff_text = truncated + f"\n\n[diff truncated: {omitted} bytes omitted]"

    select_json = json.dumps(select, indent=2, sort_keys=True)
    parts: list[str] = []
    parts.append(f"PR #{pr} {title}")
    if url:
        parts.append(f"URL: {url}")
    if author:
        author_name = author.get("login", "") if isinstance(author, dict) else str(author)
        if author_name:
            parts.append(f"Author: {author_name}")
    parts.append("")
    parts.append("PR body (author's claims, not evidence)")
    parts.append(body)
    parts.append("")
    # select.request as claim
    req = select.get("request") if isinstance(select, dict) else None
    req_err = select.get("request_error") if isinstance(select, dict) else None
    if req is not None:
        parts.append("Author's hw-gate-request (quoted as a claim, not evidence):")
        parts.append(json.dumps(req, indent=2, sort_keys=True))
        parts.append("")
    if req_err:
        parts.append(f"hw-gate-request parse error (malformed block): {req_err}")
        parts.append("")
    buckets = select.get("buckets", []) if isinstance(select, dict) else []
    parts.append(f"Mandatory buckets: {', '.join(buckets) if buckets else '(none)'}")
    parts.append("")
    if fixtures_manifest is not None:
        fixtures_list = fixtures_manifest.get("fixtures", []) if isinstance(fixtures_manifest.get("fixtures"), list) else []
        all_tags = [f.get("tag") for f in fixtures_list if f.get("tag")]
        parts.append(f"All fixture tags: {', '.join(all_tags)}")
        if available_tags is not None:
            parts.append(f"Available fixture tags on runner: {', '.join(available_tags) if available_tags else '(none)'}")
        parts.append("")
    parts.append("select.json:")
    parts.append(select_json)
    parts.append("")
    parts.append("git diff --stat base...head:")
    parts.append(diff_stat)
    parts.append("")
    parts.append("Full diff base...head:")
    parts.append(diff_text)
    return "\n".join(parts)


def build_verdict_prompt(prelim_prompt: str, prelim: dict | None, evidence: dict | None, select: dict, hw_run_result: str) -> str:
    parts: list[str] = []
    parts.append(prelim_prompt)
    parts.append("")
    parts.append("Prelim JSON:")
    parts.append(json.dumps(prelim, indent=2, sort_keys=True) if prelim is not None else "null")
    parts.append("")
    parts.append(f"hw-run result: {hw_run_result}")
    parts.append("")
    if evidence is not None:
        parts.append("hw-gate.json:")
        parts.append(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        parts.append("hw-gate.json: (missing — hw-run did not produce evidence)")
    return "\n".join(parts)


def build_decide_prompt(select: dict, prelim: dict | None, evidence: dict | None, sol_verdict_obj: dict | None, hard_reasons: list[str], checkout: str, base: str, head: str, repo: str, pr: int) -> str:
    # PR metadata
    try:
        pr_json_text = _gh(["pr", "view", str(pr), "--repo", repo, "--json", "title,body,author,url"])
        pr_info = json.loads(pr_json_text)
        title = pr_info.get("title", "")
        body = pr_info.get("body", "") or ""
    except Exception:
        title = ""
        body = ""
    diff_stat = ""
    diff_text = ""
    try:
        diff_stat = _git(["diff", "--stat", f"{base}...{head}"], checkout)
        diff_text = _git(["diff", f"{base}...{head}"], checkout)
        cap = 400 * 1024
        if len(diff_text.encode("utf-8")) > cap:
            diff_text = diff_text.encode("utf-8")[:cap].decode("utf-8", errors="ignore") + f"\n\n[diff truncated]"
    except Exception:
        pass

    parts: list[str] = []
    parts.append(f"PR #{pr} {title}")
    parts.append(body)
    parts.append("")
    parts.append("Hard floor result (non-overridable):")
    if hard_reasons:
        parts.append("HARD FLOOR FIRED: " + "; ".join(hard_reasons) + " — you cannot override this; decision must be block (evidence failure) or hold (policy/ratchet)")
    else:
        parts.append("Hard floor: no hard reasons (you may decide merge-staging/hold/block on merits)")
    parts.append("")
    parts.append("select.json:")
    parts.append(json.dumps(select, indent=2, sort_keys=True))
    parts.append("")
    parts.append("Sol prelim (prelim.json prelim field):")
    parts.append(json.dumps(prelim, indent=2, sort_keys=True) if prelim is not None else "null")
    parts.append("")
    if evidence is not None:
        parts.append("hw-gate.json evidence:")
        parts.append(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        parts.append("hw-gate.json: missing")
    parts.append("")
    parts.append("Sol verdict (verdict.json verdict+floor):")
    parts.append(json.dumps(sol_verdict_obj, indent=2, sort_keys=True) if sol_verdict_obj is not None else "null")
    parts.append("")
    parts.append("git diff --stat:")
    parts.append(diff_stat)
    parts.append("")
    parts.append("Full diff:")
    parts.append(diff_text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# comments
# ---------------------------------------------------------------------------

def upsert_comment(repo: str, pr: int, marker: str, body: str) -> str:
    cap = 60 * 1024
    if len(body.encode("utf-8")) > cap:
        body = body.encode("utf-8")[:cap].decode("utf-8", errors="ignore") + "\n\n...[truncated; see artifact]..."
    stdout = _gh(["api", f"repos/{repo}/issues/{pr}/comments", "--paginate"])
    try:
        comments = json.loads(stdout) if stdout.strip() else []
        if isinstance(comments, dict):
            comments = comments.get("comments", [comments])
    except json.JSONDecodeError:
        comments = []
        for line in stdout.splitlines():
            line=line.strip()
            if not line:
                continue
            try:
                val = json.loads(line)
                if isinstance(val, list):
                    comments.extend(val)
                elif isinstance(val, dict):
                    comments.append(val)
            except Exception:
                continue
    if not isinstance(comments, list):
        comments = [comments] if isinstance(comments, dict) else []
    existing_id = None
    existing_url = None
    for c in comments:
        if not isinstance(c, dict):
            continue
        b = c.get("body", "")
        if marker in b:
            existing_id = c.get("id")
            existing_url = c.get("html_url", "")
            break
    payload_body = body
    if marker not in payload_body:
        payload_body = marker + "\n" + payload_body
    if existing_id is not None:
        out = _gh(["api", f"repos/{repo}/issues/comments/{existing_id}", "--method", "PATCH", "-f", f"body={payload_body}"])
        try:
            resp = json.loads(out)
            if isinstance(resp, dict) and "html_url" in resp:
                return resp["html_url"]
        except Exception:
            pass
        return existing_url or f"https://github.com/{repo}/pull/{pr}#issuecomment-{existing_id}"
    else:
        out = _gh(["api", f"repos/{repo}/issues/{pr}/comments", "--method", "POST", "-f", f"body={payload_body}"])
        try:
            resp = json.loads(out)
            if isinstance(resp, dict) and "html_url" in resp:
                return resp["html_url"]
        except Exception:
            pass
        return out.strip() or f"https://github.com/{repo}/pull/{pr}#issuecomment-new"


_LABEL_COLORS = {
    "agent-approved": "0e8a16",
    "needs-human": "fbca04",
    "hw-gate-blocked": "b60205",
    "hw-run": "5319e7",
    "ratchet-raise": "d93f0b",
    "merged-staging": "0e8a16",
}


def _load_commit_messages(checkout: str, base: str, head: str) -> list[str]:
    try:
        out2 = subprocess.run(
            ["git", "log", "--format=%s%n%b%x00", f"{base}..{head}"],
            capture_output=True, text=True, cwd=checkout
        )
        if out2.returncode != 0:
            return []
        msgs = [m.strip() for m in out2.stdout.split("\x00") if m.strip()]
        return msgs
    except Exception:
        return []


# ---------------------------------------------------------------------------
# prelim phase
# ---------------------------------------------------------------------------

def _run_prelim(args) -> int:
    # Load select
    try:
        with open(args.select, "r", encoding="utf-8") as f:
            select = json.load(f)
    except Exception as e:
        sys.stderr.write(f"failed to load select: {e}\n")
        return 1
    # Load fixtures manifest
    fixtures_manifest: dict = {}
    fixtures_path = args.fixtures
    if fixtures_path and Path(fixtures_path).is_file():
        try:
            with open(fixtures_path, "r", encoding="utf-8") as f:
                fixtures_manifest = json.load(f)
        except Exception as e:
            sys.stderr.write(f"failed to load fixtures: {e}\n")
            fixtures_manifest = {}
    else:
        # try default location relative to script? Not needed
        fixtures_manifest = {"fixtures": [], "buckets": {}}

    available_map, models_dir = _available_fixtures(fixtures_manifest)
    available_tags = sorted(available_map.keys())
    known_tags = set(f.get("tag") for f in fixtures_manifest.get("fixtures", []) if f.get("tag"))

    # Bucket routes (always)
    buckets = select.get("buckets", []) if isinstance(select, dict) else []
    bucket_routes = _bucket_routes(fixtures_manifest, buckets)

    model = os.environ.get("HW_GATE_REVIEW_MODEL", "gpt-5.6-sol")

    # Build prompt
    try:
        prelim_prompt = build_prelim_prompt(select, args.checkout, args.base, args.head, args.repo, args.pr, fixtures_manifest, available_tags)
    except Exception as e:
        # still need to write prelim with failure?
        prelim_prompt = f"failed to build prelim prompt: {e}"

    prelim = None
    run_hardware = False
    posted: dict = {}
    prelim_comment_url = None
    # Try omp
    try:
        prelim = omp_review("prelim", prelim_prompt, args.system_prompt, args.checkout, model)
        # Validate prelim has run_hardware
        if isinstance(prelim, dict) and "run_hardware" in prelim:
            run_hardware = bool(prelim.get("run_hardware"))
        else:
            # If missing, treat as false
            run_hardware = False
    except ReviewError as e:
        prelim = None
        run_hardware = False
        # comment says so
        try:
            body = f"<!-- hw-gate:sol-prelim -->\n# hw-gate sol prelim\n\nSol prelim unavailable: {e}\n\nrun_hardware: false (sol unavailable, label hw-run may force)\n"
            prelim_comment_url = upsert_comment(args.repo, args.pr, "<!-- hw-gate:sol-prelim -->", body)
            posted["prelim_comment"] = prelim_comment_url
        except Exception:
            posted["prelim_comment"] = None
        # Write prelim.json and routes.json
        out_obj = {
            "schema": "hipfire.hw-gate.prelim",
            "version": 2,
            "seat": "sol",
            "model": model,
            "prelim": None,
            "run_hardware": False,
            "posted": posted,
        }
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out_obj, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception:
            pass
        # routes.json = bucket routes only (unknown dropped, but no sol routes)
        routes_to_write = bucket_routes
        # Always write routes.json even on failure
        routes_path = args.routes
        if routes_path:
            try:
                Path(routes_path).parent.mkdir(parents=True, exist_ok=True)
                with open(routes_path, "w", encoding="utf-8") as f:
                    json.dump(routes_to_write, f, indent=2, sort_keys=True)
                    f.write("\n")
            except Exception as ee:
                sys.stderr.write(f"failed to write routes: {ee}\n")
        return 0

    # Success path: compute routes.json filtering
    sol_routes_raw = prelim.get("routes", []) if isinstance(prelim, dict) else []
    sol_filtered: list[dict] = []
    sol_unavailable: list[dict] = []
    # keep track of known for unavailable
    for r in sol_routes_raw if isinstance(sol_routes_raw, list) else []:
        if not isinstance(r, dict):
            continue
        tag = r.get("tag")
        mode = r.get("mode")
        if tag not in known_tags:
            continue  # dropped unknown
        if mode not in ("battery", "chain"):
            continue
        entry = {"mode": mode, "tag": tag, "source": r.get("source", "sol"), "why": r.get("why", "")}
        sol_filtered.append(entry)
        if tag not in available_map:
            sol_unavailable.append({"tag": tag, "why": entry.get("why","") + " (not present on runner)"})

    # Merge bucket + sol (dedup by tag+mode)
    route_map: dict[tuple[str,str], dict] = {}
    for r in bucket_routes:
        key = (r["tag"], r["mode"])
        route_map[key] = r
    for r in sol_filtered:
        key = (r["tag"], r["mode"])
        if key not in route_map:
            route_map[key] = r
    routes_combined = list(route_map.values())

    # Build comment body
    summary = prelim.get("summary", "") if isinstance(prelim, dict) else ""
    run_hw_reasons = prelim.get("run_hardware_reasons", []) if isinstance(prelim, dict) else []
    claim_assessment = prelim.get("claim_assessment", "") if isinstance(prelim, dict) else ""
    questions = prelim.get("questions_for_author", []) if isinstance(prelim, dict) else []
    # Also include unavailable_routes from prelim plus our computed unavailable
    prelim_unavail = prelim.get("unavailable_routes", []) if isinstance(prelim, dict) else []
    # Merge with computed unavailable that not already present
    all_unavail = list(prelim_unavail) if isinstance(prelim_unavail, list) else []
    existing_unavail_tags = set(u.get("tag") for u in all_unavail if isinstance(u, dict))
    for u in sol_unavailable:
        if u["tag"] not in existing_unavail_tags:
            all_unavail.append(u)

    lines: list[str] = []
    lines.append("<!-- hw-gate:sol-prelim -->")
    lines.append("# hw-gate sol prelim")
    lines.append("")
    if summary:
        lines.append(f"**summary:** {summary}")
        lines.append("")
    lines.append(f"**run_hardware:** {str(run_hardware).lower()}")
    if run_hw_reasons:
        lines.append(f"**run_hardware_reasons:** {'; '.join(str(x) for x in run_hw_reasons)}")
    lines.append("")
    # routes table
    lines.append("**routes:**")
    lines.append("")
    lines.append("| mode | tag | source | why |")
    lines.append("|---|---|---|---|")
    if routes_combined:
        for r in routes_combined:
            lines.append(f"| {r.get('mode','')} | {r.get('tag','')} | {r.get('source','')} | {r.get('why','').replace('|','\\|')} |")
    else:
        lines.append("| — | — | — | no routes |")
    lines.append("")
    lines.append("**unavailable_routes:**")
    lines.append("")
    if all_unavail:
        lines.append("| tag | why |")
        lines.append("|---|---|")
        for u in all_unavail:
            lines.append(f"| {u.get('tag','')} | {u.get('why','').replace('|','\\|')} |")
    else:
        lines.append("(none)")
    lines.append("")
    if claim_assessment:
        lines.append(f"**claim_assessment:** {claim_assessment}")
        lines.append("")
    if questions:
        lines.append("**questions_for_author:**")
        for q in questions:
            lines.append(f"- {q}")
        lines.append("")
    body = "\n".join(lines)
    try:
        prelim_comment_url = upsert_comment(args.repo, args.pr, "<!-- hw-gate:sol-prelim -->", body)
        posted["prelim_comment"] = prelim_comment_url
    except Exception as e:
        sys.stderr.write(f"prelim comment failed: {e}\n")
        posted["prelim_comment"] = None

    # Write prelim.json
    out_obj = {
        "schema": "hipfire.hw-gate.prelim",
        "version": 2,
        "seat": "sol",
        "model": model,
        "prelim": prelim,
        "run_hardware": run_hardware,
        "posted": posted,
    }
    try:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2, sort_keys=True)
            f.write("\n")
    except Exception as e:
        sys.stderr.write(f"failed to write prelim.json: {e}\n")
        return 1

    # Write routes.json
    routes_path = args.routes
    if not routes_path:
        # Derive from out's dir? For workflow, --routes routes.json is explicit. If missing, skip
        sys.stderr.write("no --routes path for prelim\n")
    else:
        try:
            Path(routes_path).parent.mkdir(parents=True, exist_ok=True)
            with open(routes_path, "w", encoding="utf-8") as f:
                json.dump(routes_combined, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception as e:
            sys.stderr.write(f"failed to write routes.json: {e}\n")
            return 1
    return 0


# ---------------------------------------------------------------------------
# verdict phase (sol)
# ---------------------------------------------------------------------------

def _run_verdict(args) -> int:
    # Load select
    try:
        with open(args.select, "r", encoding="utf-8") as f:
            select = json.load(f)
    except Exception as e:
        sys.stderr.write(f"failed to load select: {e}\n")
        return 1
    # Load prelim
    prelim_obj: dict | None = None
    prelim_data: dict | None = None
    if args.prelim and Path(args.prelim).is_file():
        try:
            with open(args.prelim, "r", encoding="utf-8") as f:
                prelim_obj = json.load(f)
                # prelim file is prelim.json with schema; its "prelim" field is the model JSON
                if isinstance(prelim_obj, dict) and "prelim" in prelim_obj:
                    prelim_data = prelim_obj.get("prelim")
                else:
                    prelim_data = prelim_obj
        except Exception:
            prelim_data = None
    # Load evidence
    evidence = None
    evidence_path = Path(args.evidence) if args.evidence else None
    if evidence_path and evidence_path.is_file():
        try:
            with open(evidence_path, "r", encoding="utf-8") as f:
                evidence = json.load(f)
        except Exception:
            evidence = None
    # evidence md
    evidence_md_path = None
    if evidence_path:
        candidate = evidence_path.parent / "hw-gate.md"
        if candidate.is_file():
            evidence_md_path = candidate
        else:
            # try alongside evidence path's dir with same name but .md
            alt = evidence_path.with_suffix(".md")
            if alt.is_file():
                evidence_md_path = alt
    # hw-run result and commit messages
    hw_run_result = args.hw_run_result if args.hw_run_result else "success"
    commit_messages = _load_commit_messages(args.checkout, args.base, args.head)

    model = os.environ.get("HW_GATE_REVIEW_MODEL", "gpt-5.6-sol")

    # Post evidence comment first (as per spec: verdict phase posts evidence comment)
    evidence_comment_url = None
    if evidence_md_path and evidence_md_path.is_file():
        try:
            ev_md = evidence_md_path.read_text(encoding="utf-8")
        except Exception as e:
            ev_md = f"(failed to read hw-gate.md: {e})"
        ev_body = f"<!-- hw-gate:evidence -->\n{ev_md}\n"
    else:
        ev_body = f"<!-- hw-gate:evidence -->\nNo hardware evidence found — hw-run did not produce hw-gate.md (hw-run result: {hw_run_result}).\n"
    try:
        evidence_comment_url = upsert_comment(args.repo, args.pr, "<!-- hw-gate:evidence -->", ev_body)
    except Exception as e:
        sys.stderr.write(f"evidence comment failed: {e}\n")
        evidence_comment_url = None

    # Build prelim_prompt for verdict prompt (reuse prelim prompt generation with fixtures?)
    # Need fixtures for prelim prompt generation; try to load via --fixtures if provided? Verdict doesn't have fixtures arg, so skip.
    try:
        prelim_prompt = build_prelim_prompt(select, args.checkout, args.base, args.head, args.repo, args.pr, None, None)
    except Exception as e:
        prelim_prompt = f"failed to build prelim prompt: {e}"

    verdict_prompt = build_verdict_prompt(prelim_prompt, prelim_data, evidence, select, hw_run_result)

    verdict = None
    verdict_parse_failed = False
    try:
        verdict = omp_review("verdict", verdict_prompt, args.system_prompt, args.checkout, model)
    except ReviewError as e:
        verdict_parse_failed = True
        verdict = None
        sys.stderr.write(f"verdict omp failed: {e}\n")

    # Compute floor
    model_decision = None
    if isinstance(verdict, dict):
        model_decision = verdict.get("decision")

    hard = hard_floor(evidence, select, hw_run_result, commit_messages)
    soft = soft_floor(verdict, model_decision)
    # If verdict parse failed, add reason to hard/soft? It should be soft/hard? We'll add to hard or soft via later logic.
    # For floor display, hard and soft as computed. Parse failure will be reflected in final decision.

    # Determine final decision per spec
    def _has_hold(r: str) -> bool:
        return "policy_paths" in r or "RATCHET" in r
    hard_has_block = any(not _has_hold(r) for r in hard) if hard else False
    hard_has_hold = any(_has_hold(r) for r in hard) if hard else False

    if hard:
        if hard_has_block:
            final_decision = "block"
        else:
            final_decision = "needs-human"
    elif soft:
        final_decision = "needs-human"
    else:
        if verdict_parse_failed:
            final_decision = "needs-human"
        elif model_decision == "greenlight":
            final_decision = "greenlight"
        elif model_decision == "block":
            final_decision = "block"
        elif model_decision == "needs-human":
            final_decision = "needs-human"
        else:
            final_decision = "needs-human"

    # If verdict parse failed, ensure soft includes note? For floor display, hard stays as is, soft may include parse failure
    if verdict_parse_failed:
        if "verdict_parse_failed" not in soft:
            # add to soft for visibility
            pass

    # Post verdict comment with marker sol-verdict
    # Include verdict JSON + floor lines
    verdict_comment_url = None
    # Build body per spec: verdict JSON + floor lines
    if not verdict_parse_failed:
        floor_text = f"Floor: hard={hard} soft={soft} model_decision={model_decision} final={final_decision}"
        verdict_body = f"<!-- hw-gate:sol-verdict -->\n# hw-gate sol verdict\n\n```json\n{json.dumps(verdict, indent=2, sort_keys=True)}\n```\n\n{floor_text}\n"
    else:
        floor_text = f"Floor: hard={hard} soft={soft} model_decision={model_decision} final={final_decision} (verdict parse failed)"
        verdict_body = f"<!-- hw-gate:sol-verdict -->\n# hw-gate sol verdict\n\nVerdict parsing failed; floor forces {final_decision}.\n\n{floor_text}\n```json\n{json.dumps(prelim_data, indent=2, sort_keys=True) if prelim_data else 'null'}\n```\n"
    try:
        verdict_comment_url = upsert_comment(args.repo, args.pr, "<!-- hw-gate:sol-verdict -->", verdict_body)
    except Exception as e:
        sys.stderr.write(f"verdict comment failed: {e}\n")
        verdict_comment_url = None

    # gh pr review --comment only (never approve)
    review_url = None
    # Build summary for review
    if isinstance(verdict, dict):
        rationale = verdict.get("rationale", "")
        summary = f"hw-gate sol verdict {final_decision}: {rationale}" if rationale else f"hw-gate sol verdict {final_decision} floor hard={hard} soft={soft}"
    else:
        summary = f"hw-gate sol verdict {final_decision}: verdict parse failed, fail-closed to needs-human"
    try:
        out = _gh(["pr", "review", str(args.pr), "--repo", args.repo, "--comment", "--body", summary])
        review_url = out.strip() or None
    except Exception as e:
        sys.stderr.write(f"review comment failed: {e}\n")
        review_url = None

    # Write verdict.json version 2
    out_obj = {
        "schema": "hipfire.hw-gate.verdict",
        "version": 2,
        "seat": "sol",
        "model": model,
        "verdict": verdict,
        "floor": {"hard": hard, "soft": soft, "model_decision": model_decision, "final_decision": final_decision},
        "posted": {"evidence_comment": evidence_comment_url, "verdict_comment": verdict_comment_url, "review": review_url},
    }
    try:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2, sort_keys=True)
            f.write("\n")
    except Exception as e:
        sys.stderr.write(f"failed to write verdict.json: {e}\n")
        return 1

    return 1 if verdict_parse_failed else 0


# ---------------------------------------------------------------------------
# decide phase (fable)
# ---------------------------------------------------------------------------

def _run_decide(args) -> int:
    # Load select
    try:
        with open(args.select, "r", encoding="utf-8") as f:
            select = json.load(f)
    except Exception as e:
        sys.stderr.write(f"failed to load select: {e}\n")
        return 1
    # Load prelim
    prelim_obj = None
    prelim_data = None
    if args.prelim and Path(args.prelim).is_file():
        try:
            with open(args.prelim, "r", encoding="utf-8") as f:
                prelim_obj = json.load(f)
                prelim_data = prelim_obj.get("prelim") if isinstance(prelim_obj, dict) and "prelim" in prelim_obj else prelim_obj
        except Exception:
            prelim_data = None
    else:
        prelim_data = None
    # Load evidence
    evidence = None
    if args.evidence and Path(args.evidence).is_file():
        try:
            with open(args.evidence, "r", encoding="utf-8") as f:
                evidence = json.load(f)
        except Exception:
            evidence = None
    # Load verdict (sol verdict json)
    sol_verdict_file = None
    sol_verdict_data = None
    sol_floor = None
    sol_final = None
    if args.verdict and Path(args.verdict).is_file():
        try:
            with open(args.verdict, "r", encoding="utf-8") as f:
                sol_verdict_file = json.load(f)
                sol_verdict_data = sol_verdict_file.get("verdict")
                sol_floor = sol_verdict_file.get("floor", {})
                sol_final = sol_floor.get("final_decision") if isinstance(sol_floor, dict) else None
                if sol_final is None and isinstance(sol_verdict_data, dict):
                    sol_final = sol_verdict_data.get("decision")
                    if sol_final == "greenlight":
                        sol_final = "greenlight"
                    elif sol_final == "needs-human":
                        sol_final = "needs-human"
                    elif sol_final == "block":
                        sol_final = "block"
        except Exception:
            sol_verdict_data = None
            sol_final = None
    hw_run_result = args.hw_run_result if args.hw_run_result else "success"
    staging = args.staging if args.staging else "beta"
    commit_messages = _load_commit_messages(args.checkout, args.base, args.head)
    model = os.environ.get("HW_GATE_DECIDE_MODEL", "claude-fable-5-1")
    hard = hard_floor(evidence, select, hw_run_result, commit_messages)
    soft_model_decision = None
    if isinstance(sol_verdict_data, dict):
        soft_model_decision = sol_verdict_data.get("decision")
    else:
        if isinstance(sol_floor, dict):
            soft_model_decision = sol_floor.get("model_decision")
    soft = soft_floor(sol_verdict_data, soft_model_decision)
    try:
        decide_prompt = build_decide_prompt(select, prelim_data, evidence, sol_verdict_file, hard, args.checkout, args.base, args.head, args.repo, args.pr)
    except Exception as e:
        decide_prompt = f"failed to build decide prompt: {e}"
    # --- investigate mode handling ---
    investigate = bool(getattr(args, "investigate", False))
    child_env: dict | None = None
    max_minutes_str = os.environ.get("HW_GATE_MAX_MINUTES", "45")
    evidence_dir_val: str | None = getattr(args, "evidence_dir", None)
    decision = None
    fable_unavailable = False
    fable_error_reason: str | None = None
    investigation: list = []
    unproven: list = []
    if investigate:
        # Validate and build sandboxed env
        try:
            child_env, max_minutes_str = _build_investigate_env(args)
            evidence_dir_val = child_env.get("HW_GATE_EVIDENCE")
        except ReviewError as e:
            sys.stderr.write(f"investigate env failed: {e}\n")
            fable_unavailable = True
            fable_error_reason = str(e)
            decision = None
            investigation = []
            unproven = []
            child_env = None
        if not fable_unavailable:
            # Augment prompt with sandbox values and registry listing
            try:
                registry_present = _list_registry_fixtures(args.checkout, child_env["HIPFIRE_MODELS_DIR"]) if child_env else []
                addon = _build_investigate_prompt_addon(args.checkout, child_env, registry_present) if child_env else ""
                decide_prompt = decide_prompt + "\n\n" + addon
            except Exception as e:
                sys.stderr.write(f"failed to build investigate prompt addon: {e}\n")
            stdout_text = ""
            stderr_text = ""
            try:
                result = omp_investigate(decide_prompt, args.system_prompt, args.checkout, model, child_env, max_minutes_str)
                stdout_text = result.stdout or ""
                stderr_text = result.stderr or ""
                if result.returncode != 0:
                    fable_unavailable = True
                    fable_error_reason = f"omp decide failed ({result.returncode}): {stderr_text.strip()}"
                    sys.stderr.write(f"fable omp failed: {fable_error_reason}\n")
                    # best effort parse investigation from stdout
                    last_text, obj = _parse_omp_stdout(stdout_text)
                    if obj is not None and isinstance(obj, dict):
                        decision = obj
                        inv = obj.get("investigation")
                        if isinstance(inv, list):
                            investigation = inv
                        else:
                            investigation = []
                        up = obj.get("unproven")
                        if isinstance(up, list):
                            unproven = up
                        else:
                            unproven = []
                    else:
                        # try to extract investigation even if not full json? parse last_text
                        investigation = []
                        unproven = []
                        decision = None
                else:
                    last_text, obj = _parse_omp_stdout(stdout_text)
                    if obj is None:
                        fable_unavailable = True
                        fable_error_reason = "omp decide: no JSON object in assistant text"
                        sys.stderr.write(f"fable omp failed: {fable_error_reason}\n")
                        investigation = []
                        unproven = []
                        decision = None
                    else:
                        decision = obj
                        inv = obj.get("investigation")
                        investigation = inv if isinstance(inv, list) else []
                        up = obj.get("unproven")
                        unproven = up if isinstance(up, list) else []
            except subprocess.TimeoutExpired as te:
                # capture stdout if any
                try:
                    if isinstance(te.stdout, str):
                        stdout_text = te.stdout
                    elif te.stdout:
                        stdout_text = te.stdout.decode("utf-8", errors="ignore")
                    else:
                        stdout_text = ""
                except Exception:
                    stdout_text = ""
                fable_unavailable = True
                fable_error_reason = f"omp decide timed out after {max_minutes_str}m"
                sys.stderr.write(f"fable omp timeout: {fable_error_reason}\n")
                last_text, obj = _parse_omp_stdout(stdout_text) if stdout_text else (None, None)
                if obj is not None and isinstance(obj, dict):
                    decision = obj
                    inv = obj.get("investigation")
                    investigation = inv if isinstance(inv, list) else []
                    up = obj.get("unproven")
                    unproven = up if isinstance(up, list) else []
                else:
                    investigation = []
                    unproven = []
                    decision = None
            except Exception as e:
                fable_unavailable = True
                fable_error_reason = str(e)
                sys.stderr.write(f"fable omp failed: {e}\n")
                investigation = []
                unproven = []
                decision = None
    else:
        # non-investigate: read-only behavior unchanged
        try:
            decision = omp_review("decide", decide_prompt, args.system_prompt, args.checkout, model)
            # extract investigation/unproven if present (for decision.json completeness)
            if isinstance(decision, dict):
                inv = decision.get("investigation")
                investigation = inv if isinstance(inv, list) else []
                up = decision.get("unproven")
                unproven = up if isinstance(up, list) else []
                # also keep evidence_dir for completeness: may be None
                if evidence_dir_val is None:
                    # no investigate, but try to get from decision or args?
                    evidence_dir_val = getattr(args, "evidence_dir", None)
            else:
                investigation = []
                unproven = []
        except ReviewError as e:
            decision = None
            fable_unavailable = True
            fable_error_reason = str(e)
            sys.stderr.write(f"fable omp failed: {e}\n")
            investigation = []
            unproven = []
            # evidence_dir_val remains as arg value if any
    # Ensure evidence_dir_val is set for decision.json even in non-investigate case
    if investigate and child_env and not evidence_dir_val:
        evidence_dir_val = child_env.get("HW_GATE_EVIDENCE")
    # Determine decision_final with hard floor precedence
    def _has_hold(r: str) -> bool:
        return "policy_paths" in r or "RATCHET" in r
    hard_has_block = any(not _has_hold(r) for r in hard) if hard else False
    if hard:
        if hard_has_block:
            decision_final = "block"
        else:
            decision_final = "hold"
    else:
        if fable_unavailable or decision is None:
            decision_final = "hold"
        else:
            fable_dec = decision.get("decision")
            if fable_dec in ("merge-staging", "hold", "block"):
                decision_final = fable_dec
            else:
                decision_final = "hold"
    # Override detection
    override = None
    sol_to_fable = {"greenlight": "merge-staging", "needs-human": "hold", "block": "block"}
    if sol_final and decision and not hard:
        sol_fable_equiv = sol_to_fable.get(sol_final)
        fable_dec = decision.get("decision")
        if sol_fable_equiv and fable_dec != sol_fable_equiv:
            why = None
            ov = decision.get("override")
            if isinstance(ov, dict) and ov.get("why"):
                why = ov.get("why")
            elif decision.get("rationale"):
                why = decision.get("rationale")
            else:
                why = f"Fable {fable_dec} overrides Sol {sol_final}"
            override = {"of": sol_final, "why": why}
    merged = None
    announcement_extra = ""
    pr_title = ""
    try:
        pr_json_text = _gh(["pr", "view", str(args.pr), "--repo", args.repo, "--json", "title"])
        pr_info = json.loads(pr_json_text)
        pr_title = pr_info.get("title", "")
    except Exception:
        pr_title = ""
    if decision_final == "merge-staging":
        commit_msg = f"hw-gate: merge PR #{args.pr} ({pr_title}) to staging"
        try:
            out = _gh(["api", f"repos/{args.repo}/merges", "-f", f"base={staging}", "-f", f"head={args.head}", "-f", f"commit_message={commit_msg}"])
            try:
                resp = json.loads(out) if out.strip() else {}
                merge_sha = resp.get("sha") if isinstance(resp, dict) else None
                if not merge_sha:
                    merge_sha = out.strip() or None
            except Exception:
                merge_sha = out.strip() or None
            merged = {"base": staging, "head": args.head, "merge_sha": merge_sha}
        except ReviewError as e:
            err_msg = str(e)
            merged = {"base": staging, "head": args.head, "merge_sha": None, "error": err_msg}
            is_409 = "409" in err_msg or "already" in err_msg.lower() or "conflict" in err_msg.lower()
            decision_final = "hold"
            hard.append("staging_merge_failed" if not is_409 else "staging_merge_conflict")
            announcement_extra = (
                f" Fable decided merge-staging, but merging into `{staging}` failed"
                f"{' with a conflict (409)' if is_409 else ''}: {err_msg}. Holding for a human."
            )
    announcement = ""
    if isinstance(decision, dict):
        announcement = decision.get("announcement", "")
    if not announcement:
        if fable_unavailable:
            # include reason if available
            reason = f" {fable_error_reason}" if fable_error_reason else ""
            announcement = f"Fable unavailable; holding for human review.{reason}".strip()
        elif decision_final == "merge-staging":
            announcement = "Fable merges to staging."
        elif decision_final == "hold":
            announcement = "Fable holds for human review."
        else:
            announcement = "Fable blocks."
    rationale = ""
    if isinstance(decision, dict):
        rationale = decision.get("rationale", "")
    override_note = ""
    if override:
        override_note = f"Override Sol {override['of']}: {override['why']}"
    comment_lines: list[str] = []
    comment_lines.append("<!-- hw-gate:fable-decision -->")
    comment_lines.append(f"**announcement:** {announcement}")
    if announcement_extra:
        comment_lines.append(announcement_extra)
    if override_note:
        comment_lines.append(f"**override:** {override_note}")
    # investigation table above rationale
    if investigation:
        comment_lines.append("**investigation:**")
        comment_lines.append("")
        comment_lines.append("| question | route | result | evidence |")
        comment_lines.append("|---|---|---|---|")
        for inv in investigation:
            if not isinstance(inv, dict):
                continue
            q = str(inv.get("question", ""))
            route = str(inv.get("route", ""))
            res = str(inv.get("result", ""))
            ev = str(inv.get("evidence", ""))
            # show evidence file name (basename) but keep full if not path
            ev_disp = Path(ev).name if ev else ev
            # escape pipes
            q = q.replace("|", "\\|").replace("\n", " ")
            route = route.replace("|", "\\|").replace("\n", " ")
            res = res.replace("|", "\\|").replace("\n", " ")
            ev_disp = ev_disp.replace("|", "\\|").replace("\n", " ")
            comment_lines.append(f"| {q} | {route} | {res} | {ev_disp} |")
        comment_lines.append("")
    if unproven:
        comment_lines.append("**unproven:**")
        for u in unproven:
            if isinstance(u, str):
                comment_lines.append(f"- {u}")
            else:
                comment_lines.append(f"- {json.dumps(u)}")
        comment_lines.append("")
    if rationale:
        comment_lines.append(f"**rationale:** {rationale}")
    if merged:
        if merged.get("merge_sha"):
            comment_lines.append(f"**merged:** {merged['base']} {merged['merge_sha']}")
        elif merged.get("error"):
            comment_lines.append(f"**merge error:** {merged['error']}")
    if hard:
        comment_lines.append(f"**hard floor:** {hard}")
    if soft:
        comment_lines.append(f"**soft floor:** {soft}")
    # include fable error reason if unavailable and not already in announcement
    if fable_unavailable and fable_error_reason and fable_error_reason not in announcement:
        comment_lines.append(f"**fable unavailable:** {fable_error_reason}")
    fable_body = "\n\n".join(comment_lines)
    fable_comment_url = None
    try:
        fable_comment_url = upsert_comment(args.repo, args.pr, "<!-- hw-gate:fable-decision -->", fable_body)
    except Exception as e:
        sys.stderr.write(f"fable comment failed: {e}\n")
        fable_comment_url = None
    if decision_final == "merge-staging" and merged and merged.get("error") and "409" in str(merged.get("error", "")):
        target_label = "needs-human"
        review_flag = "--comment"
        review_body = f"{announcement} {override_note} {rationale} {announcement_extra}".strip()
    elif decision_final == "merge-staging":
        target_label = "merged-staging"
        review_flag = "--approve"
        review_body = f"{announcement} {rationale} {override_note}".strip()
    elif decision_final == "hold":
        target_label = "needs-human"
        review_flag = "--comment"
        review_body = f"{announcement} {rationale} {override_note}".strip()
    else:
        target_label = "hw-gate-blocked"
        review_flag = "--request-changes"
        review_body = f"{announcement} {rationale} {override_note}".strip()
    labels_added: list[str] = []
    labels_removed: list[str] = []
    review_url = None
    color = _LABEL_COLORS.get(target_label, "ededed")
    try:
        _gh(["label", "create", target_label, "--repo", args.repo, "--color", color, "--force"])
    except ReviewError:
        pass
    try:
        out = _gh(["pr", "review", str(args.pr), "--repo", args.repo, review_flag, "--body", review_body])
        review_url = out.strip() or None
    except ReviewError as e:
        sys.stderr.write(f"pr review failed: {e}\n")
        review_url = None
    try:
        _gh(["api", f"repos/{args.repo}/issues/{args.pr}/labels", "--method", "POST", "-f", f"labels[]={target_label}"])
        labels_added.append(target_label)
    except ReviewError:
        pass
    for lbl in ["merged-staging", "needs-human", "hw-gate-blocked", "agent-approved"]:
        if lbl != target_label:
            try:
                _gh(["api", f"repos/{args.repo}/issues/{args.pr}/labels/{lbl}", "--method", "DELETE"])
                labels_removed.append(lbl)
            except ReviewError:
                pass
    try:
        _gh(["api", f"repos/{args.repo}/issues/{args.pr}/labels/hw-run", "--method", "DELETE"])
        labels_removed.append("hw-run")
    except ReviewError:
        pass
    posted = {
        "fable_comment": fable_comment_url,
        "review": review_url,
        "labels_added": labels_added,
        "labels_removed": labels_removed,
    }
    out_obj = {
        "schema": "hipfire.hw-gate.decision",
        "version": 1,
        "seat": "fable",
        "model": model,
        "decision": decision,
        "floor": {"hard": hard, "soft": soft},
        "decision_final": decision_final,
        "override": override,
        "merged": merged,
        "posted": posted,
        "investigation": investigation,
        "unproven": unproven,
        "evidence_dir": evidence_dir_val,
    }
    try:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2, sort_keys=True)
            f.write("\n")
    except Exception as e:
        sys.stderr.write(f"failed to write decision.json: {e}\n")
        return 1
    return 1 if fable_unavailable else 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="hw-gate reviewer driver: sol prelim/verdict and fable decide")
    ap.add_argument("--seat", choices=("sol", "fable"), required=True, help="reviewer seat")
    ap.add_argument("--phase", choices=("prelim", "verdict", "decide"), required=True, help="phase")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pr", required=True, type=int)
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--checkout", required=True)
    ap.add_argument("--select", required=True, help="select.json path")
    ap.add_argument("--fixtures", help="fixtures.json path (prelim only)")
    ap.add_argument("--prelim", help="prelim.json path (verdict/decide)")
    ap.add_argument("--evidence", help="hw-gate.json path (verdict/decide)")
    ap.add_argument("--verdict", help="verdict.json path (decide only)")
    ap.add_argument("--hw-run-result", help="hw-run result string")
    ap.add_argument("--staging", help="staging branch name (decide only)")
    ap.add_argument("--routes", help="routes.json output path (prelim only)")
    ap.add_argument("--system-prompt", required=True)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--investigate", action="store_true", help="run Fable with full tools (investigation mode)")
    ap.add_argument("--devices", help="HW_GATE_DEVICES for investigate")
    ap.add_argument("--home", help="HIPFIRE_HOME for investigate")
    ap.add_argument("--evidence-dir", dest="evidence_dir", help="HW_GATE_EVIDENCE directory for investigate")
    ap.add_argument("--bin", help="HW_GATE_BIN for investigate")
    ap.add_argument("--base-bin", dest="base_bin", help="HW_GATE_BASE_BIN for investigate (optional)")
    ap.add_argument("--round", type=int, default=1, help="HW_GATE_ROUND for investigate (default 1)")
    args = ap.parse_args(argv)
    # The workflow passes `--checkout pr` relative to the job workspace. Every seat
    # launch runs omp with cwd=checkout AND `--cwd checkout`; a relative path is
    # resolved twice (`pr/pr`) and omp exits 1 before the seat ever reads the diff.
    args.checkout = os.path.abspath(args.checkout)

    # Validate seat/phase combo per authority model
    if args.seat == "fable" and args.phase in ("prelim", "verdict"):
        sys.stderr.write("fable seat only supports decide phase\n")
        return 2
    if args.seat == "sol" and args.phase == "decide":
        sys.stderr.write("sol seat does not support decide\n")
        return 2

    try:
        if args.phase == "prelim":
            return _run_prelim(args)
        elif args.phase == "verdict":
            return _run_verdict(args)
        elif args.phase == "decide":
            return _run_decide(args)
        else:
            sys.stderr.write(f"unknown phase {args.phase}\n")
            return 2
    except ReviewError as e:
        sys.stderr.write(f"review error: {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
