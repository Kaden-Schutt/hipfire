#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Assert benchlocal_campaign plan reproduces archived 2026-07-31 runner commands.

Compares plan output to vendored provenance fixtures structurally: cwd/env/runner
tokens must match exactly; archived flags must be a subset of the plan; the plan
may only add ``--provider-model``. Flag order and ``--flag=value`` vs
``--flag value`` are irrelevant.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "scripts" / "benchlocal_campaign.py"
MANIFEST = REPO / "tools" / "benchlocal" / "manifest.json"
FIXTURES = REPO / "tests" / "fixtures" / "benchlocal"

PACK_DIR_BY_ID = {
    "dataextract-15": "DataExtract-15",
    "instructfollow-15": "InstructFollow-15",
    "reasonmath-15": "ReasonMath-15",
    "toolcall-15": "ToolCall-15",
    "promptauthority-15": "PromptAuthority-15",
    "structoutput-15": "StructOutput-15",
    "bugfind-15": "BugFind-15",
    "cli-40": "CLI-40",
    "hermesagent-20": "HermesAgent-20",
}
PACK_ID_BY_DIR = {v: k for k, v in PACK_DIR_BY_ID.items()}
TOOL_PACK_IDS = frozenset({"toolcall-15", "promptauthority-15"})
EXTRA_FLAG_ALLOWLIST = frozenset({"--provider-model"})

# Explicit pack entries recorded in the AR archive (not only via command_pattern).
AR_EXPLICIT_PACKS = ("structoutput-15", "bugfind-15", "cli-40", "hermesagent-20")
# Explicit pack entries recorded in the DFlash archive.
DFLASH_EXPLICIT_PACKS = ("cli-40", "hermesagent-20")


def _require_driver() -> None:
    if not DRIVER.is_file():
        pytest.skip(f"driver not landed yet: {DRIVER.relative_to(REPO)}")


def _load_fixture(slug: str) -> dict[str, Any]:
    path = FIXTURES / f"{slug}.runner-commands.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _campaign_root_from_fixture(fixture: dict[str, Any]) -> str:
    """Derive campaign root from fixture results_root (no disk dependency)."""
    results_root = Path(fixture["results_root"])
    # <campaign-root>/results/<slug>
    return str(results_root.parent.parent)


def _run_plan(
    campaign_root: str,
    route: str,
    *,
    thinking: str | None = None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(DRIVER),
        "plan",
        "--manifest",
        str(MANIFEST),
        "--campaign-root",
        campaign_root,
        "--route",
        route,
    ]
    if thinking is not None:
        cmd.extend(["--thinking", thinking])
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"plan failed (exit {proc.returncode}) for route={route!r} "
            f"thinking={thinking!r}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"plan stdout is not JSON for route={route!r}: {exc}\n{proc.stdout!r}"
        ) from exc


def _parse_flags(argv: list[str]) -> tuple[list[str], dict[str, str | bool]]:
    """Split argv into leading program tokens and a flag→value map."""
    prog: list[str] = []
    i = 0
    while i < len(argv) and not argv[i].startswith("-"):
        prog.append(argv[i])
        i += 1

    flags: dict[str, str | bool] = {}
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            if "=" in tok:
                key, _, val = tok.partition("=")
                flags[key] = val
                i += 1
            elif i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                flags[tok] = argv[i + 1]
                i += 2
            else:
                flags[tok] = True
                i += 1
        elif tok.startswith("-") and len(tok) > 1 and not tok[1:].startswith("-"):
            # short flag: -x [value]
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                flags[tok] = argv[i + 1]
                i += 2
            else:
                flags[tok] = True
                i += 1
        else:
            # unexpected positional; keep scanning
            i += 1
    return prog, flags


def parse_command(
    command: str,
) -> tuple[str, dict[str, str], list[str], dict[str, str | bool]]:
    """Parse ``cd <cwd> && K=V ... <argv>`` into (cwd, env, prog, flags)."""
    text = command.strip()
    if not text.startswith("cd "):
        raise AssertionError(f"command does not start with 'cd ': {command!r}")
    body = text[3:]
    if " && " not in body:
        raise AssertionError(f"command missing ' && ' separator: {command!r}")
    cwd_part, _, after = body.partition(" && ")
    cwd = cwd_part.strip()

    tokens = shlex.split(after)
    env: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-"):
            break
        if "=" not in tok:
            break
        key, _, val = tok.partition("=")
        if not key or not (key[0].isalpha() or key[0] == "_"):
            break
        env[key] = val
        i += 1

    argv = tokens[i:]
    prog, flags = _parse_flags(argv)
    return cwd, env, prog, flags


def _load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sidecar_allowance(pack_id: str, manifest: dict[str, Any]) -> tuple[str, str, str] | None:
    """Return (url_env, cli_flag, url) for a pack with a sidecar, else None."""
    for pack in manifest.get("packs") or []:
        if pack.get("id") != pack_id:
            continue
        sidecar = pack.get("sidecar")
        if not isinstance(sidecar, dict):
            return None
        url_env = sidecar.get("url_env")
        cli_flag = sidecar.get("cli_flag")
        host_port = sidecar.get("host_port")
        if not url_env or not cli_flag or host_port is None:
            return None
        url = f"http://127.0.0.1:{host_port}"
        return str(url_env), str(cli_flag), url
    return None


def assert_command_compatible(
    archive_cmd: str,
    plan_cmd: str,
    *,
    label: str,
    pattern_derived: bool = False,
    sidecar: tuple[str, str, str] | None = None,
) -> None:
    """Archive command must be structurally reproduced by plan (archive ⊆ plan).

    Explicit archive entries (pattern_derived=False): env equality is exact; plan
    may only add flags from EXTRA_FLAG_ALLOWLIST.

    Pattern-derived entries: env and flags are archive-subset. The plan may add
    only the pack's manifest sidecar url_env/cli_flag (when declared) and
    EXTRA_FLAG_ALLOWLIST flags. Sidecar wiring must be present iff declared.
    """
    a_cwd, a_env, a_prog, a_flags = parse_command(archive_cmd)
    p_cwd, p_env, p_prog, p_flags = parse_command(plan_cmd)

    assert a_cwd == p_cwd, f"{label}: cwd mismatch\n  archive: {a_cwd!r}\n  plan:    {p_cwd!r}"
    assert a_prog == p_prog, (
        f"{label}: runner program tokens mismatch\n"
        f"  archive: {a_prog!r}\n  plan:    {p_prog!r}"
    )

    for key, val in a_env.items():
        assert key in p_env, (
            f"{label}: plan missing archived env {key}={val!r}\n"
            f"  archive: {a_env!r}\n  plan:    {p_env!r}"
        )
        assert p_env[key] == val, (
            f"{label}: env {key} value mismatch\n"
            f"  archive: {val!r}\n  plan:    {p_env[key]!r}"
        )

    for key, val in a_flags.items():
        assert key in p_flags, (
            f"{label}: plan missing archived flag {key}={val!r}\n"
            f"  archive flags: {a_flags!r}\n  plan flags:    {p_flags!r}"
        )
        assert p_flags[key] == val, (
            f"{label}: flag {key} value mismatch\n"
            f"  archive: {val!r}\n  plan:    {p_flags[key]!r}"
        )

    if not pattern_derived:
        assert a_env == p_env, (
            f"{label}: env mismatch\n  archive: {a_env!r}\n  plan:    {p_env!r}"
        )
        extras = set(p_flags) - set(a_flags)
        unexpected = extras - EXTRA_FLAG_ALLOWLIST
        assert not unexpected, (
            f"{label}: plan has unexpected extra flags {sorted(unexpected)} "
            f"(only {sorted(EXTRA_FLAG_ALLOWLIST)} allowed)\n"
            f"  archive flags: {a_flags!r}\n  plan flags:    {p_flags!r}"
        )
        return

    # Pattern-derived: allow only manifest sidecar + --provider-model extras.
    allowed_env: set[str] = set()
    allowed_flags: set[str] = set(EXTRA_FLAG_ALLOWLIST)
    if sidecar is not None:
        url_env, cli_flag, url = sidecar
        allowed_env.add(url_env)
        allowed_flags.add(cli_flag)
        assert p_env.get(url_env) == url, (
            f"{label}: expected sidecar env {url_env}={url!r}, got {p_env.get(url_env)!r}"
        )
        assert p_flags.get(cli_flag) == url, (
            f"{label}: expected sidecar flag {cli_flag}={url!r}, got {p_flags.get(cli_flag)!r}"
        )
    else:
        # No sidecar declared — plan must not invent one-looking extras beyond
        # the archive baseline (caught by the unexpected checks below).
        pass

    env_extras = set(p_env) - set(a_env)
    unexpected_env = env_extras - allowed_env
    assert not unexpected_env, (
        f"{label}: plan has unexpected extra env {sorted(unexpected_env)} "
        f"(only sidecar {sorted(allowed_env) or '∅'} allowed beyond archive)\n"
        f"  archive: {a_env!r}\n  plan:    {p_env!r}"
    )

    flag_extras = set(p_flags) - set(a_flags)
    unexpected_flags = flag_extras - allowed_flags
    assert not unexpected_flags, (
        f"{label}: plan has unexpected extra flags {sorted(unexpected_flags)} "
        f"(only {sorted(allowed_flags)} allowed beyond archive)\n"
        f"  archive flags: {a_flags!r}\n  plan flags:    {p_flags!r}"
    )

    if sidecar is None:
        # Double-check no stray sidecar-shaped keys snuck in via the archive
        # baseline itself being empty for these.
        for key in p_env:
            if key.endswith("_URL") and key not in a_env:
                raise AssertionError(
                    f"{label}: non-sidecar pack has unexpected URL env {key}={p_env[key]!r}"
                )



def _strip_pattern_prose(pattern: str) -> str:
    """Drop bracketed human prose tacked onto archived command_pattern strings."""
    cut = pattern.find(" [")
    if cut != -1:
        return pattern[:cut].rstrip()
    return pattern.rstrip()


def _expand_wave1_command(
    pattern: str,
    pack_dir: str,
    *,
    pack_id: str,
    tool_extra: str | None,
) -> str:
    base = _strip_pattern_prose(pattern).replace("<Pack>", pack_dir)
    if pack_id in TOOL_PACK_IDS:
        extra = (tool_extra or "--repetition-penalty 1.0 --tools-format default").strip()
        if extra:
            # Insert tool-pack flags before trailing --show-raw --json when present.
            marker = " --show-raw --json"
            if marker in base:
                head, _, tail = base.partition(marker)
                return f"{head} {extra}{marker}{tail}"
            return f"{base} {extra}"
    return base


def _plan_pack_command(plan: dict[str, Any], pack_id: str) -> str:
    packs = plan.get("packs") or {}
    assert pack_id in packs, (
        f"plan missing pack {pack_id!r}; have {sorted(packs)}"
    )
    entry = packs[pack_id]
    assert "command" in entry, f"plan pack {pack_id!r} missing 'command': {entry!r}"
    return entry["command"]


def _assert_ports_in_text(text: str, ports: tuple[int, ...], *, label: str) -> None:
    for port in ports:
        assert str(port) in text, f"{label}: expected port {port} in plan output"


# ---------------------------------------------------------------------------
# qwen27-ar
# ---------------------------------------------------------------------------


def test_plan_qwen27_ar_matches_archive() -> None:
    _require_driver()
    fixture = _load_fixture("qwen27-ar")
    campaign_root = _campaign_root_from_fixture(fixture)
    plan = _run_plan(campaign_root, "qwen27-ar")

    assert plan.get("route") == "qwen27-ar"
    assert plan.get("model") == fixture["model"]
    assert plan.get("results_root") == fixture["results_root"]
    assert plan.get("shared_env_non_hermes") == fixture["shared_env_non_hermes"]

    # Ports from the AR archive topology.
    blob = json.dumps(plan)
    _assert_ports_in_text(blob, (12180, 12280, 13180), label="qwen27-ar plan")

    for pack_id in AR_EXPLICIT_PACKS:
        arch_entry = fixture["packs"][pack_id]
        assert_command_compatible(
            arch_entry["command"],
            _plan_pack_command(plan, pack_id),
            label=f"qwen27-ar/{pack_id}",
        )
        if "env" in arch_entry:
            plan_env = plan["packs"][pack_id].get("env") or {}
            assert plan_env == arch_entry["env"], (
                f"qwen27-ar/{pack_id}: env dict mismatch\n"
                f"  archive: {arch_entry['env']!r}\n  plan:    {plan_env!r}"
            )

    manifest = _load_manifest()
    wave = fixture["packs"]["wave1_already_complete"]
    pattern = wave["command_pattern"]
    for pack_id in wave["packs"]:
        pack_dir = PACK_DIR_BY_ID[pack_id]
        expected = _expand_wave1_command(
            pattern, pack_dir, pack_id=pack_id, tool_extra=None
        )
        plan_cmd = _plan_pack_command(plan, pack_id)
        assert_command_compatible(
            expected,
            plan_cmd,
            label=f"qwen27-ar/wave1/{pack_id}",
            pattern_derived=True,
            sidecar=_sidecar_allowance(pack_id, manifest),
        )
        _, _, _, flags = parse_command(plan_cmd)
        if pack_id in TOOL_PACK_IDS:
            assert flags.get("--repetition-penalty") == "1.0", pack_id
            assert flags.get("--tools-format") == "default", pack_id
        else:
            assert "--repetition-penalty" not in flags, pack_id
            assert "--tools-format" not in flags, pack_id



# ---------------------------------------------------------------------------
# qwen27-dflash
# ---------------------------------------------------------------------------


def test_plan_qwen27_dflash_matches_archive() -> None:
    _require_driver()
    fixture = _load_fixture("qwen27-dflash")
    campaign_root = _campaign_root_from_fixture(fixture)
    plan = _run_plan(campaign_root, "qwen27-dflash")

    assert plan.get("route") == "qwen27-dflash"
    assert plan.get("model") == fixture["model"]
    assert plan.get("results_root") == fixture["results_root"]
    assert plan.get("shared_env_non_hermes") == fixture["shared_env_non_hermes"]

    blob = json.dumps(plan)
    _assert_ports_in_text(blob, (12181, 12281), label="qwen27-dflash plan")

    for pack_id in DFLASH_EXPLICIT_PACKS:
        arch_entry = fixture["packs"][pack_id]
        assert_command_compatible(
            arch_entry["command"],
            _plan_pack_command(plan, pack_id),
            label=f"qwen27-dflash/{pack_id}",
        )

    manifest = _load_manifest()
    wave = fixture["packs"]["wave1_cli_pattern"]
    pattern = wave["command_pattern"]
    tool_extra = wave.get("tool_packs_extra_flags")
    for pack_dir in wave["packs"]:
        pack_id = PACK_ID_BY_DIR[pack_dir]
        expected = _expand_wave1_command(
            pattern, pack_dir, pack_id=pack_id, tool_extra=tool_extra
        )
        plan_cmd = _plan_pack_command(plan, pack_id)
        assert_command_compatible(
            expected,
            plan_cmd,
            label=f"qwen27-dflash/wave1/{pack_id}",
            pattern_derived=True,
            sidecar=_sidecar_allowance(pack_id, manifest),
        )
        _, _, _, flags = parse_command(plan_cmd)
        if pack_id in TOOL_PACK_IDS:
            assert flags.get("--repetition-penalty") == "1.0", pack_id
            assert flags.get("--tools-format") == "default", pack_id
        else:
            assert "--repetition-penalty" not in flags, pack_id
            assert "--tools-format" not in flags, pack_id



# ---------------------------------------------------------------------------
# thinking variant is proxy-side only
# ---------------------------------------------------------------------------


def test_thinking_medium_leaves_non_hermes_commands_unchanged() -> None:
    """``--thinking medium`` must not alter pack CLI flags (proxy injects it)."""
    _require_driver()
    fixture = _load_fixture("qwen27-ar")
    campaign_root = _campaign_root_from_fixture(fixture)

    disabled = _run_plan(campaign_root, "qwen27-ar", thinking="disabled")
    medium = _run_plan(campaign_root, "qwen27-ar", thinking="medium")

    d_packs = disabled.get("packs") or {}
    m_packs = medium.get("packs") or {}
    assert set(d_packs) == set(m_packs), (
        f"pack key set differs under thinking modes: "
        f"{sorted(d_packs)} vs {sorted(m_packs)}"
    )

    for pack_id, d_entry in d_packs.items():
        if pack_id == "hermesagent-20":
            continue
        m_entry = m_packs[pack_id]
        assert d_entry.get("command") == m_entry.get("command"), (
            f"{pack_id}: non-Hermes command changed under --thinking medium\n"
            f"  disabled: {d_entry.get('command')!r}\n"
            f"  medium:   {m_entry.get('command')!r}"
        )
        assert d_entry.get("env") == m_entry.get("env"), (
            f"{pack_id}: non-Hermes env changed under --thinking medium"
        )
        assert d_entry.get("argv") == m_entry.get("argv"), (
            f"{pack_id}: non-Hermes argv changed under --thinking medium"
        )

    # shared_env is also non-Hermes surface area
    assert disabled.get("shared_env_non_hermes") == medium.get("shared_env_non_hermes")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
