# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Declarative change→route manifest for ``tools.change_gate``.

This module is the **single place to add coverage**. Selection, execution, and
reporting live elsewhere; they only consume ``ROUTES`` / ``RULES``.

Adding a new arch crate
-----------------------
1. Add one or more ``Route`` entries whose ``models`` / ``argv`` exercise that
   arch only (never borrow another family's rows).
2. Add a ``Rule`` whose ``surface`` is ``crates/hipfire-arch-<name>/**`` (and any
   arch-private kernel globs) pointing at those route ids.
3. Put a concrete regression class in every new ``Route.why`` — preferably with
   a dead-gate row id, issue number, or commit hash. A route without a ``why``
   naming a concrete regression class should not be added.

Policy sources (do not invent wider coverage)
---------------------------------------------
- ``docs/VALIDATION.md`` — claim class → minimum route; fail closed.
- ``.githooks/pre-commit`` HOTSPOT / SERVE_HOTSPOT / PP_HOTSPOT — split into
  precise per-surface rules below (flat regexes were the bug).
- ``.research/dead-gates/coherence-gate*.sh`` — hard-won row comments preserved
  in each ``Route.why`` (e.g. ``a9e8dfda`` Q8_0-wo MoE residual aliasing,
  ``0912c73a`` Paro GemvResidual Givens skip, Path-A dflash attractor /
  ``6c84b13``, AWQ lm_head / MQ3 sidecar loader bugs, issue #87 / #462).

Cost discipline
---------------
- Docs-only or control-plane-only (``crates/hipfire-{cli,config,registry,client}``)
  changes must select **no GPU route**.
- An arch-crate change selects **only that arch's** routes.
- Anything ``tier="heavy"`` (est > 15 min), including the 128K/200K-class
  pflash NIAH run, is only owed when the pflash/long-context surface itself
  changes (selector enforces ``include_heavy`` / direct-match).
"""

from __future__ import annotations

from tools.change_gate.model import Route, Rule

# ---------------------------------------------------------------------------
# Helpers (local construction only — not part of the public contract)
# ---------------------------------------------------------------------------


def _R(
    id: str,
    kind: str,
    argv: tuple[str, ...],
    est_minutes: float,
    why: str,
    *,
    models: tuple[str, ...] = (),
    arches: tuple[str, ...] = (),
    tier: str | None = None,
) -> Route:
    if tier is None:
        if est_minutes < 2.0:
            tier = "cheap"
        elif est_minutes <= 15.0:
            tier = "standard"
        else:
            tier = "heavy"
    return Route(
        id=id,
        kind=kind,
        argv=argv,
        est_minutes=est_minutes,
        tier=tier,
        arches=arches,
        models=models,
        why=why,
    )


def _serve(
    *,
    mode: str = "battery",
    kv: str = "fwht3",
    dflash: str = "off",
    draft: str | None = None,
    thinking: str = "med",
    max_tokens: int = 512,
    sampling: str = "greedy",
    extra: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """``scripts/serve_harness.py`` argv; ``{model}`` / ``{out}`` filled by runner."""
    argv: list[str] = [
        "python3",
        "scripts/serve_harness.py",
        "--model",
        "{model}",
        "--mode",
        mode,
        "--kv",
        kv,
        "--dflash",
        dflash,
        "--thinking",
        thinking,
        "--max-tokens",
        str(max_tokens),
        "--sampling",
        sampling,
        "--out",
        "{out}",
    ]
    if draft is not None:
        argv.extend(["--draft", draft])
    argv.extend(extra)
    return tuple(argv)


# ===========================================================================
# ROUTES
# ===========================================================================

ROUTES: dict[str, Route] = {
    # ------------------------------------------------------------------
    # Cheap control-plane / unit / shell (no GPU required to *select*;
    # GPU routes below may still block at host check).
    # ------------------------------------------------------------------
    "unit.env-docs": _R(
        "unit.env-docs",
        "unit",
        ("python3", "scripts/check-env-docs.py"),
        0.2,
        "HIPFIRE_* env-name / config-ownership drift "
        "(docs/VALIDATION.md automatic docs check; scripts/check-env-docs.py).",
    ),
    "unit.diff-check": _R(
        "unit.diff-check",
        "shell",
        ("git", "diff", "--check"),
        0.05,
        "Whitespace/conflict-marker hygiene for docs-only edits "
        "(docs/VALIDATION.md documentation checks).",
    ),
    "unit.no-gpu-control": _R(
        "unit.no-gpu-control",
        "unit",
        (
            "cargo",
            "test",
            "-p",
            "hipfire-config",
            "-p",
            "hipfire-registry",
            "-p",
            "hipfire-client",
            "-p",
            "hipfire-cli",
            "-p",
            "hipfire-tui",
            "--",
            "--quiet",
        ),
        1.5,
        "No-GPU control-plane crate tests "
        "(scripts/no-gpu-ci.sh cargo test -p hipfire-config -p hipfire-registry …).",
    ),
    "unit.rdna-compute": _R(
        "unit.rdna-compute",
        "unit",
        ("cargo", "test", "-p", "rdna-compute", "--lib", "--", "--quiet"),
        1.0,
        "rdna-compute lib unit tests (dispatch tables, pool, compiler helpers) "
        "from scripts/no-gpu-ci.sh.",
    ),
    "unit.hipfire-detect": _R(
        "unit.hipfire-detect",
        "unit",
        ("cargo", "test", "-p", "hipfire-detect", "--", "--quiet"),
        0.5,
        "Attractor/ngram/special-leak detector crate "
        "(ported from coherence-gate-dflash.sh:191-243; do not reimplement in Python).",
    ),
    "unit.hipfire-dispatch": _R(
        "unit.hipfire-dispatch",
        "unit",
        ("cargo", "test", "-p", "hipfire-dispatch", "--", "--quiet"),
        0.8,
        "Dispatch-table / kernel-id unit coverage without GPU launch.",
    ),
    "unit.hipfire-quantize": _R(
        "unit.hipfire-quantize",
        "unit",
        ("cargo", "test", "-p", "hipfire-quantize", "--", "--quiet"),
        0.8,
        "Quantize-tooling unit tests (format tables, packing helpers).",
    ),
    "unit.redline-crates": _R(
        "unit.redline-crates",
        "unit",
        (
            "cargo",
            "test",
            "-p",
            "redline",
            "-p",
            "redline-dispatch",
            "-p",
            "redline-rocr",
            "--",
            "--quiet",
        ),
        1.0,
        "Redline crate unit tests (tape/PM4 lower helpers; not product route proof).",
    ),
    "unit.tools-redline": _R(
        "unit.tools-redline",
        "unit",
        ("python3", "-m", "unittest", "discover", "-s", "tools/redline/tests", "-q"),
        0.3,
        "tools.redline golden/bench/serve-diff unit suite "
        "(scripts/no-gpu-ci.sh python3 -m unittest discover).",
    ),
    "shell.bind-thread": _R(
        "shell.bind-thread",
        "shell",
        ("./scripts/verify-bind-thread.sh",),
        0.1,
        "Every public dispatch.rs Gpu entry must bind_thread "
        "(pre-commit + docs/VALIDATION.md; silent mis-bind → cross-device pointer corruption, issue #58).",
    ),
    "shell.agentic-self-check": _R(
        "shell.agentic-self-check",
        "shell",
        ("./scripts/agentic-gate.sh", "--self-check"),
        0.1,
        "Agentic tool-call detector rot guard "
        "(scripts/agentic-gate.sh --self-check; issue #87 class detectors).",
    ),
    "detect.kernels-channel": _R(
        "detect.kernels-channel",
        "detect",
        (
            "cargo",
            "build",
            "--release",
            "--features",
            "deltanet",
            "--example",
            "test_kernels",
            "-p",
            "hipfire-runtime",
        ),
        3.0,
        "Build test_kernels channel binary "
        "(docs/VALIDATION.md: new/changed .hip → test_kernels then model-level route). "
        "est includes release build; numeric run is host/arch gated by the runner.",
    ),
    # ------------------------------------------------------------------
    # Serve — Qwen3.5 dense short battery (dead coherence-gate.sh SHORT)
    # ------------------------------------------------------------------
    "serve.battery.qwen35-0.8b": _R(
        "serve.battery.qwen35-0.8b",
        "serve",
        _serve(max_tokens=80),
        2.0,
        "Qwen3.5-0.8B MQ4 capital/smoke row "
        "(coherence-gate.sh SHORT `cap` on qwen3.5-0.8b.mq4) — launch/overhead + basic AR coherence.",
        models=("qwen3.5-0.8b.mq4",),
    ),
    "serve.battery.qwen35-4b": _R(
        "serve.battery.qwen35-4b",
        "serve",
        _serve(max_tokens=180),
        3.0,
        "Qwen3.5-4B MQ4 code-shape row "
        "(coherence-gate.sh SHORT `code` on qwen3.5-4b.mq4).",
        models=("qwen3.5-4b.mq4",),
    ),
    "serve.battery.qwen35-9b": _R(
        "serve.battery.qwen35-9b",
        "serve",
        _serve(max_tokens=300),
        4.0,
        "Qwen3.5-9B MQ4 reason + tool-call shapes "
        "(coherence-gate.sh SHORT `reason`/`tool-call`; tool-call covers issue #87 auto-MMQ class on short system prompts).",
        models=("qwen3.5-9b.mq4",),
    ),
    "serve.battery.qwen35-9b-mq3": _R(
        "serve.battery.qwen35-9b-mq3",
        "serve",
        _serve(max_tokens=300),
        4.0,
        "MQ3 WMMA prefill + K4-unroll decode + fused residual coherence "
        "(coherence-gate.sh SHORT `reason-mq3`; gfx11+gfx12 only at load).",
        models=("qwen3.5-9b.mq3",),
        arches=("gfx1100", "gfx1101", "gfx1150", "gfx1151", "gfx1200", "gfx1201"),
    ),
    "serve.battery.qwen35-27b-mq3": _R(
        "serve.battery.qwen35-27b-mq3",
        "serve",
        _serve(max_tokens=80),
        5.0,
        "27B MQ3 capital smoke "
        "(coherence-gate.sh SHORT `cap-mq3-27b`) — large dense MQ3 load path.",
        models=("qwen3.5-27b.mq3",),
        arches=("gfx1100", "gfx1101", "gfx1150", "gfx1151", "gfx1200", "gfx1201"),
    ),
    "serve.battery.qwen35-mq3-lloyd": _R(
        "serve.battery.qwen35-mq3-lloyd",
        "serve",
        _serve(max_tokens=80),
        3.5,
        "MQ3-Lloyd K4 + fp32-LDS-codebook + tail-rotation "
        "(coherence-gate.sh SHORT `cap-mq3-lloyd-4b` / PR #115 research-gated format).",
        models=("qwen3.5-4b.mq3-lloyd",),
    ),
    "serve.battery.qwen35-mq3-lloyd-long": _R(
        "serve.battery.qwen35-mq3-lloyd-long",
        "serve",
        _serve(max_tokens=220),
        5.0,
        "MQ3-Lloyd batched-prefill WMMA fused kernels (qkv/qkvza/gate_up/residual) "
        "via ~180-tok prompt (coherence-gate.sh `long-prefill-mq3-lloyd-4b`; issue #116 Phase B2; "
        "prompt md5 f20bbc4f5b88ab5f7b44fe7c7da0e2e3).",
        models=("qwen3.5-4b.mq3-lloyd",),
    ),
    "serve.battery.qwen35-mq4-lloyd": _R(
        "serve.battery.qwen35-mq4-lloyd",
        "serve",
        _serve(max_tokens=300),
        5.0,
        "MQ4-Lloyd gemm_*_mq4g256_lloyd_wmma + nibble-pair decode + per-row LDS codebook "
        "(coherence-gate.sh `reason-mq4-lloyd-9b`; issue #182 Phase B3; gfx11+gfx12).",
        models=("qwen3.5-9b.mq4-lloyd",),
        arches=("gfx1100", "gfx1101", "gfx1150", "gfx1151", "gfx1200", "gfx1201"),
    ),
    "serve.battery.qwen35-q8-long": _R(
        "serve.battery.qwen35-q8-long",
        "serve",
        _serve(max_tokens=220),
        5.0,
        "Q8_0 batched-prefill Tier-2 arms "
        "(gemm_q8_0_batched_chunked at qkv/qkvza/gate_up/wo+residual/w_down+residual) "
        "(coherence-gate.sh `long-prefill-q8-9b`; docs/plans/q8-fused-prefill-kernels.md T3-0).",
        models=("qwen3.5-9b.q8f16",),
    ),
    "serve.battery.qwen35-mq6": _R(
        "serve.battery.qwen35-mq6",
        "serve",
        _serve(max_tokens=300),
        4.0,
        "MQ6/HFQ6-G256 dispatch routing safety "
        "(coherence-gate.sh `reason-mq6` — guards gfx906 HFQ4 dp4a defaults from stealing mq6 routes).",
        models=("qwen3.5-9b.mq6",),
    ),
    "serve.battery.qwen35-mq3-awq": _R(
        "serve.battery.qwen35-mq3-awq",
        "serve",
        _serve(max_tokens=80),
        3.0,
        "MQ3-AWQ sidecar attachment regression "
        "(coherence-gate.sh `mq3-awq-paris`; 2026-05-18 loader bug gated AWQ on DType::MQ4G256 only "
        "at qwen35.rs:907 and silently dropped MQ3G256 sidecars — fixed via DType::supports_awq_sidecar).",
        models=("qwen3.5-4b.mq3-awq-only",),
    ),
    "serve.battery.qwen35-lmhead-awq": _R(
        "serve.battery.qwen35-lmhead-awq",
        "serve",
        _serve(max_tokens=300),
        5.0,
        "AWQ-aware lm_head dispatch "
        "(coherence-gate.sh `lmhead-awq-paris`; lm-head-awq-runtime PR 2026-05-18 — without "
        "weight_gemv→rotate_x_mq_for / speculative.rs::rotate_x_mq_batched_for the lm_head computes "
        "(W·s)·x ≠ W·x → KLD 0.67→13.5 class, docs/plans/awq_fix_claude.md).",
        models=("qwen3.5-9b.mq4-awq-gptq-f2-lmhead",),
    ),
    # ------------------------------------------------------------------
    # Serve — Paro / A3B MoE (FULL_EXTRA + paro SHORT rows)
    # ------------------------------------------------------------------
    "serve.battery.paro-a3b": _R(
        "serve.battery.paro-a3b",
        "serve",
        _serve(max_tokens=80),
        6.0,
        "ParoQ4G128 GemvResidual Givens-rotation fix 0912c73a "
        "(coherence-gate.sh `paro-a3b-cap`/`paro-a3b-sheep`: steps.rs GemvResidual else-branch "
        "called gemv.run(Plain) and skipped Givens for Paro weights → wrong o_proj).",
        models=("qwen3.6-35b-a3b-paro.hfq",),
    ),
    "serve.battery.qwen35-a3b-mq4": _R(
        "serve.battery.qwen35-a3b-mq4",
        "serve",
        _serve(max_tokens=500),
        8.0,
        "Qwen3.5 35B-A3B MQ4 MoE sheep reasoning "
        "(coherence-gate.sh FULL `moe-sheep`) — router + expert path AR coherence.",
        models=("qwen3.5-35b-a3b.mq4",),
    ),
    "serve.battery.qwen35-a3b-q8-wo": _R(
        "serve.battery.qwen35-a3b-q8-wo",
        "serve",
        _serve(max_tokens=500),
        10.0,
        "gfx12/RDNA4 Q8_0-wo MoE residual-buffer aliasing a9e8dfda "
        "(coherence-gate.sh FULL `moe-q8-wo-sheep`: GemvResidual fallback `out` aliased onto residual; "
        "Q8_0 wo/dn_out on RDNA4 took that fallback → RAW same buffer → silent wrong MoE for ~100 "
        "commits until ae13aa75; MQ4-only MoE rows never caught it).",
        models=("qwen3.5-35b-a3b.q8f16",),
        arches=("gfx1200", "gfx1201"),
    ),
    "serve.battery.qwen36-a3b": _R(
        "serve.battery.qwen36-a3b",
        "serve",
        _serve(max_tokens=800),
        10.0,
        "Qwen3.6 35B-A3B MQ4 MoE sheep "
        "(coherence-gate.sh FULL `moe36-sheep`).",
        models=("qwen3.6-35b-a3b.mq4",),
    ),
    "serve.battery.qwen36-27b-tool": _R(
        "serve.battery.qwen36-27b-tool",
        "serve",
        _serve(max_tokens=220),
        6.0,
        "Qwen3.6-27B tool-call shape "
        "(coherence-gate.sh FULL `tool-call-27b` + agentic-gate dense-27B stand-in for #262).",
        models=("qwen3.6-27b.mq4",),
    ),
    # ------------------------------------------------------------------
    # Serve — multi-request / agentic / chain
    # ------------------------------------------------------------------
    "serve.loop.cross-request": _R(
        "serve.loop.cross-request",
        "shell",
        ("./scripts/serve-loop-gate.sh",),
        4.0,
        "Cross-request DeltaNet/KV state contamination issue #462 "
        "(serve-loop-gate.sh; PR #455 bundle migration left daemon reset reading dead fields → "
        "</think> attractor only visible on multi-request serve).",
        models=("qwen3.5-0.8b.mq4", "qwen3.5-4b.mq4", "qwen3.5-9b.mq4", "qwen3.6-27b.mq4"),
    ),
    "serve.agentic.a3b-fast": _R(
        "serve.agentic.a3b-fast",
        "shell",
        ("./scripts/agentic-gate.sh", "--fast"),
        2.5,
        "Agentic long-system-prompt tool-call JSON structural gate "
        "(agentic-gate.sh --fast; issue #87 auto-MMQ ChatML-leak into <tool_call> on 780–1300 tok "
        "Pi/Hermes system prompts — short coherence tool-call row missed it).",
        models=("qwen3.5-35b-a3b.mq4", "qwen3.6-27b.mq4"),
    ),
    "serve.chain.qwen35-9b": _R(
        "serve.chain.qwen35-9b",
        "serve",
        _serve(mode="chain", max_tokens=256),
        6.0,
        "Prefix-cache + cross-turn prefill/decode chain "
        "(serve_harness.py --mode chain; docs/VALIDATION.md serve semantics).",
        models=("qwen3.5-9b.mq4",),
    ),
    # ------------------------------------------------------------------
    # Serve — DFlash / DDTree speculative (coherence-gate-dflash.sh)
    # ------------------------------------------------------------------
    "serve.dflash.qwen35-27b-fast": _R(
        "serve.dflash.qwen35-27b-fast",
        "serve",
        _serve(
            dflash="on",
            draft="qwen35-27b-dflash-mq4.hfq",
            max_tokens=192,
            thinking="off",
        ),
        3.0,
        "DFlash Path-A single-token attractor class "
        "(coherence-gate-dflash.sh --fast / FAST_TESTS; Path A DDTree slow-path-kill 2026-04-23 "
        "reverted in 6c84b13 — pure-stat gates missed 'numbers(numbers(...' forever; "
        "three-tier detector now in crates/hipfire-detect).",
        models=("qwen3.5-27b.mq4", "qwen35-27b-dflash-mq4.hfq"),
    ),
    "serve.dflash.qwen35-27b-short": _R(
        "serve.dflash.qwen35-27b-short",
        "serve",
        _serve(
            dflash="on",
            draft="qwen35-27b-dflash-mq4.hfq",
            max_tokens=192,
            thinking="off",
        ),
        6.0,
        "DFlash + DDTree-b12 prose/code short battery "
        "(coherence-gate-dflash.sh SHORT_TESTS ~2-3 min; Tier1/2 hard attractor thresholds).",
        models=("qwen3.5-27b.mq4", "qwen35-27b-dflash-mq4.hfq"),
    ),
    # ------------------------------------------------------------------
    # Serve — DeepSeek V4 Flash
    # ------------------------------------------------------------------
    "serve.battery.deepseek4": _R(
        "serve.battery.deepseek4",
        "serve",
        _serve(max_tokens=80),
        5.0,
        "DeepSeek V4 Flash AR capital/reason/long-prefill "
        "(coherence-gate.sh FULL `deepseek4-*`; arch_id=9 hipfire-arch-deepseek4 + optional MTP addon).",
        models=("deepseek-v4-flash.mq2lloyd",),
    ),
    "serve.mtp.deepseek4": _R(
        "serve.mtp.deepseek4",
        "shell",
        ("./scripts/coherence-gate-deepseek4-mtp.sh", "--fast"),
        2.0,
        "DeepSeek V4 MTP spec-decode attractor battery "
        "(coherence-gate-deepseek4-mtp.sh; speculative_decode_step_with_pbs in "
        "hipfire-arch-deepseek4/src/spec_decode.rs — Path-A-class detector on MTP path).",
        models=("deepseek-v4-flash.mq2lloyd", "deepseek-v4-flash-mtp.mq2lloyd"),
    ),
    # ------------------------------------------------------------------
    # Serve — MiniMax / Cohere2 / LFM / Qwen2
    # ------------------------------------------------------------------
    "serve.battery.minimax": _R(
        "serve.battery.minimax",
        "shell",
        ("./scripts/coherence-gate-minimax.sh",),
        5.0,
        "MiniMax-M2 chat-templated MoE prefill coherence "
        "(coherence-gate-minimax.sh: short prompts → indexed MoE GEMV; long ≥256-row chunk → "
        "scatter-grouped MoE prefill; hard-fail on attractor/zero tokens).",
        models=("MiniMax-M2.7.mq2",),
    ),
    "serve.battery.cohere2moe": _R(
        "serve.battery.cohere2moe",
        "shell",
        ("./scripts/coherence-gate-cohere2moe.sh",),
        6.0,
        "Cohere2-MoE / North-Mini-Code marker-leak + SWA long-context "
        "(coherence-gate-cohere2moe.sh: <|MARKER|> visible-stream leak; long-context ~5.7k tok "
        "above 4096 window + KV-capacity OOB guard; md5 cohere2moe_long.txt).",
        models=("north-mini-code.mq4.hfq",),
    ),
    "serve.battery.lfm25": _R(
        "serve.battery.lfm25",
        "serve",
        _serve(
            sampling="recipe:nothink",
            thinking="off",
            max_tokens=128,
        ),
        2.5,
        "LFM2.5 chat framing / thinking-output smoke "
        "(docs/VALIDATION.md LFM route; registry tag lfm2.5:350m → lfm2.5-350m.q8).",
        models=("lfm2.5-350m.q8",),
    ),
    "serve.reset.qwen2": _R(
        "serve.reset.qwen2",
        "shell",
        ("./scripts/qwen2-reset-gate.sh",),
        2.0,
        "Qwen2 per-request reset no-op (#462 bundle class) "
        "(qwen2-reset-gate.sh: daemon reset rewound dead m.qwen2_state instead of "
        "ModelState::Qwen2 bundle → next_pos bled across requests).",
        models=(
            "qwen25-0.5b-instruct.mq4",
            "qwen25-0.5b-q2.mq4",
            "vibethinker-3b.mq4.hfq",
        ),
    ),
    "serve.dspark.qwen35": _R(
        "serve.dspark.qwen35",
        "shell",
        ("./scripts/coherence-gate-qwen35-dspark.sh", "--fast"),
        3.0,
        "Qwen3.5-MoE DSpark EAGLE-3 spec path "
        "(coherence-gate-qwen35-dspark.sh: silent AR fallback = false-green; "
        "ornith-35b-aeon.mq6 + dspark sidecar).",
        models=("ornith-35b-aeon.mq6",),
    ),
    # ------------------------------------------------------------------
    # Redline / retained replay
    # ------------------------------------------------------------------
    "redline.capture": _R(
        "redline.capture",
        "redline",
        (
            "python3",
            "scripts/redline_daemon_harness.py",
            "--model",
            "{model}",
            "--out",
            "{out}",
        ),
        8.0,
        "Resident-daemon Redline phase fingerprint + shadow/parity "
        "(scripts/redline_daemon_harness.py; docs/VALIDATION.md retained replay claim — "
        "discovery evidence, not product PM4/AQL route proof without REDLINE.md ladder).",
        models=("qwen3.5-4b.mq4",),
    ),
    "redline.golden": _R(
        "redline.golden",
        "redline",
        ("python3", "-m", "tools.redline", "golden"),
        10.0,
        "Sealed MQ4R TG128 golden fixture reproduction "
        "(tools.redline golden; gfx1100/gfx1151/gfx1201 only — exact identity + route proof).",
        arches=("gfx1100", "gfx1151", "gfx1201"),
    ),
    # ------------------------------------------------------------------
    # Speed
    # ------------------------------------------------------------------
    "speed.arch-fast": _R(
        "speed.arch-fast",
        "speed",
        ("./scripts/speed-gate.sh", "--fast"),
        1.5,
        "MQ4 prefill/decode floor vs tests/speed-baselines/<arch>.txt (4B only) "
        "(speed-gate.sh --fast; pre-commit-class perf signal).",
        models=("qwen3.5-4b.mq4",),
    ),
    "speed.arch": _R(
        "speed.arch",
        "speed",
        ("./scripts/speed-gate.sh",),
        8.0,
        "Full MQ4 size sweep vs committed arch baselines "
        "(speed-gate.sh 0.8B/4B/9B/27B; ANY metric below baseline×(1-tol) is a PERFORMANCE BUG).",
        models=("qwen3.5-0.8b.mq4", "qwen3.5-4b.mq4", "qwen3.5-9b.mq4", "qwen3.5-27b.mq4"),
    ),
    # ------------------------------------------------------------------
    # Multi-GPU PP
    # ------------------------------------------------------------------
    "shell.pp-gate": _R(
        "shell.pp-gate",
        "shell",
        ("./scripts/pp-gate.sh",),
        6.0,
        "Pipeline-parallel pp=1 vs pp=2 bit-equivalence + DFlash/CASK refusal "
        "(pp-gate.sh; pre-commit PP_HOTSPOT — skips when <2 usable GPUs).",
        models=("qwen3.5-0.8b.mq4",),
    ),
    # ------------------------------------------------------------------
    # PFlash / long-context (standard + heavy)
    # ------------------------------------------------------------------
    "shell.pflash-gate": _R(
        "shell.pflash-gate",
        "shell",
        ("./scripts/pflash-gate.sh",),
        12.0,
        "PFlash Phase-5 NIAH 8K/16K/multi-16K/longcode/longprose/32K verdict+wall regression "
        "(pflash-gate.sh vs scripts/pflash-baselines/*; hooked historically from coherence-gate.sh "
        "follow-up stage; target qwen3.5-27b.mq3 + drafter qwen3.5-0.8b.mq4).",
        models=("qwen3.5-27b.mq3", "qwen3.5-0.8b.mq4"),
    ),
    "shell.pflash-niah-128k": _R(
        "shell.pflash-niah-128k",
        "shell",
        (
            "./scripts/pflash-niah-bench.sh",
            "{model}",
            "benchmarks/longctx/niah/niah_128k.jsonl",
            "--drafter",
            "qwen3.5-0.8b.mq4",
            "--keep-ratio",
            "0.30",
            "--pretok",
            "--runs",
            "1",
            "--label",
            "pflash30-128k",
        ),
        45.0,
        "128K-context (131072-token fixture) PFlash NIAH heavy route "
        "(benchmarks/longctx/niah/niah_128k.jsonl; the >15min timesink that must NOT run for "
        "unrelated CLI/docs/arch edits — only pflash/long-context surfaces; est ~45 min single run, "
        "derived from 32K baseline compress+prefill ~25s scaled + load).",
        models=("qwen3.5-27b.mq3", "qwen3.5-0.8b.mq4"),
        tier="heavy",
    ),
    # ------------------------------------------------------------------
    # Per-arch unit crates (cheap; no model)
    # ------------------------------------------------------------------
    "unit.arch-qwen35": _R(
        "unit.arch-qwen35",
        "unit",
        (
            "cargo",
            "test",
            "-p",
            "hipfire-arch-qwen35",
            "--lib",
            "--",
            "--quiet",
        ),
        1.0,
        "hipfire-arch-qwen35 lib tests (incl. moe_prefill slice from no-gpu-ci.sh).",
    ),
    "unit.arch-qwen35-vl": _R(
        "unit.arch-qwen35-vl",
        "unit",
        ("cargo", "test", "-p", "hipfire-arch-qwen35-vl", "--lib", "--", "--quiet"),
        0.8,
        "hipfire-arch-qwen35-vl lib unit tests.",
    ),
    "unit.arch-qwen2": _R(
        "unit.arch-qwen2",
        "unit",
        ("cargo", "test", "-p", "hipfire-arch-qwen2", "--lib", "--", "--quiet"),
        0.8,
        "hipfire-arch-qwen2 lib unit tests.",
    ),
    "unit.arch-deepseek4": _R(
        "unit.arch-deepseek4",
        "unit",
        ("cargo", "test", "-p", "hipfire-arch-deepseek4", "--lib", "--", "--quiet"),
        0.8,
        "hipfire-arch-deepseek4 lib unit tests.",
    ),
    "unit.arch-minimax": _R(
        "unit.arch-minimax",
        "unit",
        ("cargo", "test", "-p", "hipfire-arch-minimax", "--lib", "--", "--quiet"),
        0.8,
        "hipfire-arch-minimax lib unit tests.",
    ),
    "unit.arch-cohere2moe": _R(
        "unit.arch-cohere2moe",
        "unit",
        ("cargo", "test", "-p", "hipfire-arch-cohere2moe", "--lib", "--", "--quiet"),
        0.8,
        "hipfire-arch-cohere2moe lib unit tests.",
    ),
    "unit.arch-lfm2moe": _R(
        "unit.arch-lfm2moe",
        "unit",
        ("cargo", "test", "-p", "hipfire-arch-lfm2moe", "--lib", "--", "--quiet"),
        0.8,
        "hipfire-arch-lfm2moe lib unit tests.",
    ),
    "unit.arch-llama": _R(
        "unit.arch-llama",
        "unit",
        ("cargo", "test", "-p", "hipfire-arch-llama", "--lib", "--", "--quiet"),
        0.8,
        "hipfire-arch-llama / DSpark body lib unit tests.",
    ),
    "unit.arch-dots-ocr": _R(
        "unit.arch-dots-ocr",
        "unit",
        ("cargo", "test", "-p", "hipfire-arch-dots-ocr", "--lib", "--", "--quiet"),
        0.8,
        "hipfire-arch-dots-ocr lib unit tests.",
    ),
    "unit.arch-toy": _R(
        "unit.arch-toy",
        "unit",
        ("cargo", "test", "-p", "hipfire-arch-toy", "--lib", "--", "--quiet"),
        0.5,
        "hipfire-arch-toy lib unit tests (synthetic arch).",
    ),
    "unit.hipfire-runtime": _R(
        "unit.hipfire-runtime",
        "unit",
        ("cargo", "test", "-p", "hipfire-runtime", "--lib", "--", "--quiet"),
        1.5,
        "hipfire-runtime lib unit tests (sampler/loop_guard/prompt_frame/eos_filter hot path).",
    ),
    "unit.hipfire-cli": _R(
        "unit.hipfire-cli",
        "unit",
        ("cargo", "test", "-p", "hipfire-cli", "--", "--quiet"),
        0.8,
        "Native hipfire-cli unit tests (control plane only).",
    ),
    "unit.hipfire-config": _R(
        "unit.hipfire-config",
        "unit",
        ("cargo", "test", "-p", "hipfire-config", "--", "--quiet"),
        0.5,
        "hipfire-config schema/unit tests.",
    ),
    "unit.hipfire-registry": _R(
        "unit.hipfire-registry",
        "unit",
        ("cargo", "test", "-p", "hipfire-registry", "--", "--quiet"),
        0.5,
        "hipfire-registry unit tests.",
    ),
    "unit.hipfire-loader": _R(
        "unit.hipfire-loader",
        "unit",
        ("cargo", "test", "-p", "hipfire-loader", "--", "--quiet"),
        0.8,
        "Model loader unit tests (AWQ sidecar attach paths among others).",
    ),
}

# Sanity: every route id is the dict key.
assert all(k == v.id for k, v in ROUTES.items()), "ROUTES key/id mismatch"


# ===========================================================================
# RULES — precise surfaces (split from pre-commit flat HOTSPOT regexes)
# ===========================================================================
# Order does not matter for selection (selector unions route ids). Prefer
# narrow globs. Docs/cli must not pull GPU routes.

RULES: tuple[Rule, ...] = (
    # ----- docs / env tables: cheap only -----
    Rule(
        surface="docs/**",
        route_ids=("unit.env-docs", "unit.diff-check"),
        reason="Docs-only changes: env/docs drift + whitespace check; no GPU "
        "(docs/VALIDATION.md documentation checks).",
    ),
    Rule(
        surface="README.md",
        route_ids=("unit.diff-check",),
        reason="Top-level readme: whitespace only.",
    ),
    Rule(
        surface="registry/**",
        route_ids=("unit.hipfire-registry", "unit.env-docs"),
        reason="Registry JSON/schema edits → registry unit + env-doc name coverage.",
    ),
    # ----- Control plane (Rust-only): NO GPU -----
    Rule(
        # `re:` because fnmatch has no brace expansion — "{a,b}" would never match.
        surface=r"re:^crates/hipfire-(config|registry|client)/",
        route_ids=("unit.no-gpu-control", "unit.hipfire-cli"),
        reason="Control-plane crates (config/registry/client): control-plane tests only. "
        "The control plane is Rust-only — there is no cli/ TypeScript surface in this "
        "codebase (the pre-commit SERVE_HOTSPOT's historical cli/index.ts alternative is "
        "dead), so a pure control-plane edit must never become a multi-minute GPU bill.",
    ),
    Rule(
        surface="crates/hipfire-cli/**",
        route_ids=("unit.hipfire-cli", "unit.no-gpu-control"),
        reason="Native CLI crate: unit/control-plane only.",
    ),
    Rule(
        surface="crates/hipfire-client/**",
        route_ids=("unit.no-gpu-control",),
        reason="HTTP client crate: no-GPU control-plane tests.",
    ),
    Rule(
        surface="crates/hipfire-tui/**",
        route_ids=("unit.no-gpu-control",),
        reason="TUI crate: no-GPU control-plane tests.",
    ),
    Rule(
        surface="crates/hipfire-config/**",
        route_ids=("unit.hipfire-config", "unit.env-docs"),
        reason="Config schema owns HIPFIRE_* production reads.",
    ),
    Rule(
        surface="crates/hipfire-registry/**",
        route_ids=("unit.hipfire-registry",),
        reason="Registry crate unit tests.",
    ),
    # ----- quantize / loader -----
    Rule(
        surface="crates/hipfire-quantize/**",
        route_ids=("unit.hipfire-quantize", "serve.battery.qwen35-4b"),
        reason="Quantize tooling can break pack/load shapes → unit + one dense MQ4 smoke.",
    ),
    Rule(
        surface="crates/hipfire-loader/**",
        route_ids=(
            "unit.hipfire-loader",
            "serve.battery.qwen35-mq3-awq",
            "serve.battery.qwen35-lmhead-awq",
        ),
        reason="Loader/AWQ sidecar attachment (mq3-awq-paris / lmhead-awq-paris classes).",
    ),
    # ----- detect crate -----
    Rule(
        surface="crates/hipfire-detect/**",
        route_ids=("unit.hipfire-detect",),
        reason="Detector port of coherence-gate-dflash three-tier attractor logic.",
    ),
    # ----- tools.redline / change_gate itself -----
    Rule(
        surface="tools/redline/**",
        route_ids=("unit.tools-redline",),
        reason="Python redline package unit suite only.",
    ),
    Rule(
        surface="tools/change_gate/**",
        route_ids=("unit.tools-redline",),
        reason="change_gate edits: keep no-GPU python discovery path green "
        "(GateTests wires unittest; avoid GPU self-selection).",
    ),
    Rule(
        surface="tools/serve_harness/**",
        route_ids=("unit.no-gpu-control",),
        reason="serve_harness package plumbing without forcing a model battery.",
    ),
    # ----- Redline crates + harness -----
    Rule(
        surface="crates/redline/**",
        route_ids=("unit.redline-crates", "redline.capture"),
        reason="Redline core → unit + daemon phase capture.",
    ),
    Rule(
        surface="crates/redline-dispatch/**",
        route_ids=("unit.redline-crates", "redline.capture"),
        reason="Redline dispatch/tape → unit + capture.",
    ),
    Rule(
        surface="crates/redline-rocr/**",
        route_ids=("unit.redline-crates", "redline.capture"),
        reason="ROCr bridge for retained replay.",
    ),
    Rule(
        surface="scripts/redline_daemon_harness.py",
        route_ids=("redline.capture", "unit.tools-redline"),
        reason="Harness script itself.",
    ),
    Rule(
        surface="scripts/redline_dispatch_profile.py",
        route_ids=("redline.capture",),
        reason="Attribution-only PM4 profile script shares capture surface.",
    ),
    # ----- dispatch bind_thread + hipfire-dispatch -----
    Rule(
        surface="crates/rdna-compute/src/dispatch.rs",
        route_ids=("shell.bind-thread", "unit.rdna-compute", "detect.kernels-channel"),
        reason="Public Gpu bind_thread invariant (pre-commit) + dispatch unit + kernel channel build.",
    ),
    Rule(
        surface="crates/rdna-compute/**",
        route_ids=(
            "unit.rdna-compute",
            "detect.kernels-channel",
            "speed.arch-fast",
            "redline.capture",
        ),
        reason="Compute runtime: unit, channel binary, fast speed floor, redline capture "
        "(kernel/dispatch/graph surface per VALIDATION.md).",
    ),
    Rule(
        surface="crates/hipfire-dispatch/**",
        route_ids=("unit.hipfire-dispatch", "detect.kernels-channel", "speed.arch-fast"),
        reason="Dispatch tables / kernel ids.",
    ),
    Rule(
        surface="crates/hip-bridge/**",
        route_ids=("unit.rdna-compute", "detect.kernels-channel"),
        reason="HIP launch primitives under channel tests.",
    ),
    Rule(
        surface="crates/hsa-bridge/**",
        route_ids=("unit.rdna-compute",),
        reason="HSA bridge unit coverage via rdna-compute tests.",
    ),
    # ----- kernels/.hip — numeric + arch-relevant serve/speed -----
    Rule(
        surface="kernels/src/**",
        route_ids=(
            "detect.kernels-channel",
            "speed.arch-fast",
            "serve.battery.qwen35-4b",
            "redline.capture",
        ),
        reason="Any .hip change: test_kernels build + fast speed + one dense serve smoke + redline capture "
        "(docs/VALIDATION.md new/changed .hip).",
    ),
    Rule(
        surface="re:kernels/src/.*residual",
        route_ids=(
            "detect.kernels-channel",
            "serve.battery.qwen35-a3b-q8-wo",
            "serve.battery.paro-a3b",
            "serve.battery.qwen35-q8-long",
        ),
        reason="Residual/GemvResidual kernels: a9e8dfda Q8 MoE alias + 0912c73a Paro Givens + Q8 long prefill.",
    ),
    Rule(
        surface="re:kernels/src/.*moe",
        route_ids=(
            "detect.kernels-channel",
            "serve.battery.qwen35-a3b-mq4",
            "serve.battery.qwen35-a3b-q8-wo",
        ),
        reason="MoE kernels → A3B MQ4 + Q8-wo MoE rows only (not unrelated dense families).",
    ),
    Rule(
        surface="re:kernels/src/.*pflash",
        route_ids=(
            "detect.kernels-channel",
            "shell.pflash-gate",
            "shell.pflash-niah-128k",
        ),
        reason="PFlash score kernels → pflash gate + heavy 128K NIAH (direct heavy surface).",
    ),
    Rule(
        surface="re:kernels/src/.*awq",
        route_ids=(
            "detect.kernels-channel",
            "serve.battery.qwen35-mq3-awq",
            "serve.battery.qwen35-lmhead-awq",
        ),
        reason="AWQ rotate/lm_head kernels → AWQ paris rows.",
    ),
    Rule(
        surface="re:kernels/src/.*lloyd",
        route_ids=(
            "detect.kernels-channel",
            "serve.battery.qwen35-mq3-lloyd",
            "serve.battery.qwen35-mq3-lloyd-long",
            "serve.battery.qwen35-mq4-lloyd",
        ),
        reason="Lloyd codebook kernels → MQ3/MQ4-Lloyd coherence rows.",
    ),
    Rule(
        surface="kernels/src/pflash/**",
        route_ids=(
            "detect.kernels-channel",
            "shell.pflash-gate",
            "shell.pflash-niah-128k",
        ),
        reason="kernels/src/pflash/ tree → pflash standard + heavy.",
    ),
    # ----- hipfire-runtime hot path (sampler, loop_guard, …) -----
    Rule(
        surface="crates/hipfire-runtime/src/sampler.rs",
        route_ids=("unit.hipfire-runtime", "serve.battery.qwen35-4b", "serve.battery.qwen35-9b"),
        reason="Sampler hot path (pre-commit HOTSPOT sampler.rs).",
    ),
    Rule(
        surface="crates/hipfire-runtime/src/loop_guard.rs",
        route_ids=("unit.hipfire-runtime", "serve.battery.qwen35-4b", "serve.loop.cross-request"),
        reason="loop_guard affects every model load (pre-commit HOTSPOT).",
    ),
    Rule(
        surface="crates/hipfire-runtime/src/prompt_frame.rs",
        route_ids=("unit.hipfire-runtime", "serve.battery.qwen35-9b", "serve.agentic.a3b-fast"),
        reason="prompt_frame / ChatML framing (tool-call + agentic shapes).",
    ),
    Rule(
        surface="crates/hipfire-runtime/src/eos_filter.rs",
        route_ids=("unit.hipfire-runtime", "serve.battery.qwen35-4b"),
        reason="EOS filter hot path (pre-commit HOTSPOT).",
    ),
    Rule(
        surface="crates/hipfire-runtime/src/arch.rs",
        route_ids=("unit.hipfire-runtime", "serve.battery.qwen35-4b"),
        reason="Architecture trait surface (pre-commit HOTSPOT arch.rs).",
    ),
    Rule(
        surface="crates/hipfire-runtime/src/multi_gpu.rs",
        route_ids=("unit.hipfire-runtime", "shell.pp-gate", "serve.battery.qwen35-0.8b"),
        reason="PP dispatch path (pre-commit HOTSPOT + PP_HOTSPOT multi_gpu.rs).",
    ),
    Rule(
        surface="crates/hipfire-daemon/src/main.rs",
        route_ids=(
            "serve.loop.cross-request",
            "serve.battery.qwen35-4b",
            "serve.reset.qwen2",
            "redline.capture",
        ),
        reason="Daemon binary is SERVE_HOTSPOT — multi-request contamination + smoke + redline.",
    ),
    Rule(
        surface="crates/hipfire-runtime/examples/**",
        route_ids=("unit.hipfire-runtime", "serve.battery.qwen35-4b"),
        reason="Runtime examples (benches/gates drivers).",
    ),
    Rule(
        surface="crates/hipfire-runtime/**",
        route_ids=(
            "unit.hipfire-runtime",
            "serve.battery.qwen35-4b",
            "speed.arch-fast",
        ),
        reason="Broad runtime touch: unit + one dense smoke + fast speed "
        "(narrower rules above add serve-loop/agentic/pp when those files match).",
    ),
    Rule(
        surface="re:crates/hipfire-runtime/.*(?:peer_access|pp_|pipeline|stages|forward_prefill_batch_multi|forward_scratch_multi)",
        route_ids=("shell.pp-gate", "unit.hipfire-runtime"),
        reason="PP_HOTSPOT regex split from pre-commit (peer_access/pp_/pipeline/…).",
    ),
    # ----- Qwen35 arch (precise; no other families) -----
    Rule(
        surface="crates/hipfire-arch-qwen35/**",
        route_ids=(
            "unit.arch-qwen35",
            "serve.battery.qwen35-4b",
            "serve.battery.qwen35-9b",
            "speed.arch-fast",
        ),
        reason="Qwen35 arch crate baseline: unit + dense 4B/9B smoke + fast speed "
        "(never selects gemma/deepseek/minimax/cohere rows).",
    ),
    Rule(
        surface="crates/hipfire-arch-qwen35/src/qwen35.rs",
        route_ids=(
            "unit.arch-qwen35",
            "serve.battery.qwen35-4b",
            "serve.battery.qwen35-9b",
            "serve.battery.qwen35-9b-mq3",
            "serve.battery.qwen35-mq3-awq",
            "serve.battery.qwen35-lmhead-awq",
            "serve.battery.qwen35-a3b-mq4",
            "serve.battery.qwen35-a3b-q8-wo",
            "speed.arch-fast",
        ),
        reason="Core qwen35 forward (pre-commit HOTSPOT qwen35.rs) — dense + MoE + AWQ rows for this arch only.",
    ),
    Rule(
        surface="crates/hipfire-arch-qwen35/src/pflash.rs",
        route_ids=(
            "unit.arch-qwen35",
            "shell.pflash-gate",
            "shell.pflash-niah-128k",
        ),
        reason="PFlash implementation — standard pflash-gate + heavy 128K NIAH only.",
    ),
    Rule(
        surface="crates/hipfire-arch-qwen35/src/dflash_spec.rs",
        route_ids=(
            "unit.arch-qwen35",
            "serve.dflash.qwen35-27b-fast",
            "serve.loop.cross-request",
        ),
        reason="DFlash spec path — Path-A attractor fast battery + cross-request loop.",
    ),
    Rule(
        surface="crates/hipfire-arch-qwen35/src/speculative.rs",
        route_ids=(
            "unit.arch-qwen35",
            "serve.dflash.qwen35-27b-fast",
            "serve.loop.cross-request",
            "serve.battery.qwen35-lmhead-awq",
        ),
        reason="speculative.rs SERVE_HOTSPOT + lm_head AWQ batched rotate path.",
    ),
    Rule(
        surface="crates/hipfire-arch-qwen35/src/spec_impl.rs",
        route_ids=("unit.arch-qwen35", "serve.dflash.qwen35-27b-fast", "serve.dspark.qwen35"),
        reason="Spec implementation shared by DFlash/DSpark.",
    ),
    Rule(
        surface="crates/hipfire-arch-qwen35/src/paro_moe.rs",
        route_ids=("unit.arch-qwen35", "serve.battery.paro-a3b"),
        reason="Paro MoE path → 0912c73a GemvResidual Givens row only.",
    ),
    Rule(
        surface="re:crates/hipfire-arch-qwen35/src/mtp_.*\\.rs$",
        route_ids=("unit.arch-qwen35", "serve.dflash.qwen35-27b-fast"),
        reason="MTP head/compose/probe/spec modules (pre-commit HOTSPOT mtp_*.rs).",
    ),
    Rule(
        surface="crates/hipfire-arch-qwen35/src/grammar.rs",
        route_ids=("unit.arch-qwen35", "serve.agentic.a3b-fast", "serve.battery.qwen35-9b"),
        reason="Grammar / tool-call emission shapes.",
    ),
    Rule(
        surface="crates/hipfire-arch-qwen35-vl/**",
        route_ids=("unit.arch-qwen35-vl", "serve.battery.qwen35-4b"),
        reason="Qwen35-VL arch: unit + shared dense smoke (pre-commit HOTSPOT qwen35_vl.rs).",
    ),
    # ----- other arch crates: own routes only -----
    Rule(
        surface="crates/hipfire-arch-deepseek4/**",
        route_ids=(
            "unit.arch-deepseek4",
            "serve.battery.deepseek4",
            "serve.mtp.deepseek4",
        ),
        reason="DeepSeek4 arch only — AR battery + MTP fast gate (no qwen35 rows).",
    ),
    Rule(
        surface="crates/hipfire-arch-deepseek4/src/spec_decode.rs",
        route_ids=("unit.arch-deepseek4", "serve.mtp.deepseek4"),
        reason="DeepSeek MTP spec_decode path (pre-commit HOTSPOT spec_decode.rs historically).",
    ),
    Rule(
        surface="crates/hipfire-arch-minimax/**",
        route_ids=("unit.arch-minimax", "serve.battery.minimax"),
        reason="MiniMax arch only.",
    ),
    Rule(
        surface="crates/hipfire-arch-cohere2moe/**",
        route_ids=("unit.arch-cohere2moe", "serve.battery.cohere2moe"),
        reason="Cohere2-MoE / North arch only.",
    ),
    Rule(
        surface="crates/hipfire-arch-lfm2moe/**",
        route_ids=("unit.arch-lfm2moe", "serve.battery.lfm25"),
        reason="LFM2 MoE arch — LFM framing serve smoke only.",
    ),
    Rule(
        surface="crates/hipfire-arch-llama/**",
        route_ids=("unit.arch-llama", "serve.dspark.qwen35"),
        reason="Llama/DSpark body used by qwen35-dspark sidecar path.",
    ),
    Rule(
        surface="crates/hipfire-arch-qwen2/**",
        route_ids=("unit.arch-qwen2", "serve.reset.qwen2"),
        reason="Qwen2 arch — reset no-op gate (#462 bundle class).",
    ),
    Rule(
        surface="crates/hipfire-arch-dots-ocr/**",
        route_ids=("unit.arch-dots-ocr", "serve.reset.qwen2"),
        reason="dots-ocr still owns legacy qwen2_state field; reset path adjacency.",
    ),
    Rule(
        surface="crates/hipfire-arch-toy/**",
        route_ids=("unit.arch-toy",),
        reason="Toy arch: unit only.",
    ),
    # ----- scripts / gates / harnesses -----
    Rule(
        surface="scripts/serve_harness.py",
        route_ids=("serve.battery.qwen35-4b",),
        reason="Harness changes validated by one dense battery.",
    ),
    Rule(
        surface="scripts/speed-gate.sh",
        route_ids=("speed.arch-fast",),
        reason="Speed gate script → fast arm.",
    ),
    Rule(
        surface="scripts/pflash-gate.sh",
        route_ids=("shell.pflash-gate",),
        reason="PFlash gate script.",
    ),
    Rule(
        surface="scripts/pflash-niah-bench.sh",
        route_ids=("shell.pflash-gate", "shell.pflash-niah-128k"),
        reason="NIAH wrapper → standard + heavy pflash routes.",
    ),
    Rule(
        surface="scripts/pflash-baselines/**",
        route_ids=("shell.pflash-gate",),
        reason="Committed pflash baselines.",
    ),
    Rule(
        surface="scripts/serve-loop-gate.sh",
        route_ids=("serve.loop.cross-request",),
        reason="Serve-loop gate script.",
    ),
    Rule(
        surface="scripts/agentic-gate.sh",
        route_ids=("shell.agentic-self-check", "serve.agentic.a3b-fast"),
        reason="Agentic gate script + detector self-check.",
    ),
    Rule(
        surface="scripts/pp-gate.sh",
        route_ids=("shell.pp-gate",),
        reason="PP gate script.",
    ),
    Rule(
        surface="scripts/verify-bind-thread.sh",
        route_ids=("shell.bind-thread",),
        reason="bind_thread verifier script.",
    ),
    Rule(
        surface="scripts/check-env-docs.py",
        route_ids=("unit.env-docs",),
        reason="Env-docs checker.",
    ),
    Rule(
        surface="scripts/no-gpu-ci.sh",
        route_ids=("unit.no-gpu-control", "unit.rdna-compute", "unit.tools-redline"),
        reason="No-GPU CI script body coverage.",
    ),
    Rule(
        surface="scripts/gates.sh",
        route_ids=("serve.battery.qwen35-4b", "redline.capture"),
        reason="Manual gates.sh wrapper touches serve + optional redline.",
    ),
    Rule(
        surface="scripts/coherence-gate*.sh",
        route_ids=("unit.hipfire-detect",),
        reason="Retired coherence scripts kept as historical — detector unit only "
        "(docs/VALIDATION.md retired gates; not acceptance).",
    ),
    Rule(
        surface="scripts/qwen2-reset-gate.sh",
        route_ids=("serve.reset.qwen2",),
        reason="Qwen2 reset gate script.",
    ),
    Rule(
        surface="scripts/coherence-gate-deepseek4*.sh",
        route_ids=("serve.mtp.deepseek4", "unit.arch-deepseek4"),
        reason="DeepSeek4 historical/MTP gate scripts.",
    ),
    Rule(
        surface="scripts/coherence-gate-minimax.sh",
        route_ids=("serve.battery.minimax",),
        reason="MiniMax gate script.",
    ),
    Rule(
        surface="scripts/coherence-gate-cohere2moe.sh",
        route_ids=("serve.battery.cohere2moe",),
        reason="Cohere2moe gate script.",
    ),
    Rule(
        surface="scripts/coherence-gate-qwen35-dspark.sh",
        route_ids=("serve.dspark.qwen35",),
        reason="Qwen35 DSpark gate script.",
    ),
    Rule(
        surface="scripts/gpu-lock.sh",
        route_ids=("unit.diff-check",),
        reason="GPU lock helper — no product matrix.",
    ),
    # ----- benchmarks / baselines / tests -----
    Rule(
        surface="tests/speed-baselines/**",
        route_ids=("speed.arch",),
        reason="Committed speed floors — full speed-gate when baselines change.",
    ),
    Rule(
        surface="benchmarks/longctx/**",
        route_ids=("shell.pflash-gate", "shell.pflash-niah-128k"),
        reason="Long-context / NIAH fixtures → pflash standard + heavy.",
    ),
    Rule(
        surface="benchmarks/prompts/**",
        route_ids=("serve.battery.qwen35-4b", "shell.agentic-self-check"),
        reason="Prompt fixtures used by batteries/agentic detectors.",
    ),
    Rule(
        surface="benchmarks/prompts/agentic_*",
        route_ids=("serve.agentic.a3b-fast", "shell.agentic-self-check"),
        reason="Agentic system/user prompts → agentic fast cell.",
    ),
    Rule(
        surface="crates/hipfire-runtime/examples/pflash_niah_bench.rs",
        route_ids=("shell.pflash-gate", "shell.pflash-niah-128k"),
        reason="PFlash NIAH bench example source.",
    ),
    # ----- .githooks / CI workflow (control plane) -----
    Rule(
        surface=".githooks/**",
        route_ids=("unit.diff-check", "shell.bind-thread"),
        reason="Hook changes: cheap checks only.",
    ),
    Rule(
        surface=".github/**",
        route_ids=("unit.no-gpu-control", "unit.env-docs"),
        reason="CI workflow: no-GPU control plane + env docs.",
    ),
)


def routes_by_id() -> dict[str, Route]:
    """Return the route manifest (id → Route)."""
    return ROUTES


def rules() -> tuple[Rule, ...]:
    """Return the surface → route selection rules."""
    return RULES


def _validate_manifest() -> None:
    unknown: list[str] = []
    for rule in RULES:
        for rid in rule.route_ids:
            if rid not in ROUTES:
                unknown.append(f"{rule.surface!r} → {rid}")
    if unknown:
        raise RuntimeError("RULES reference unknown route ids: " + "; ".join(unknown))


_validate_manifest()
