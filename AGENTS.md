# AGENTS.md - hipfire repo guidance

This file is the repo-wide contract for agents. Keep it small: put
subsystem-only rules in the nearest nested `AGENTS.md`, keep long procedures in
skills or docs, and use the task prompt for one-off constraints.

For project background, notices, and detailed playbooks, read `README.md` and
the relevant docs under `docs/`.

## Core Invariants

- hipfire is Rust + HIP/ROCm-direct inference. Do not put Python in the
  inference hot path; Python is allowed for tooling, benchmarks, and comparison
  baselines.
- Do not add Vulkan, wgpu, or a cross-vendor compute backend. The backend is
  HIP/ROCm-direct.
- Treat portability as a design constraint. When touching runtime, dispatch,
  kernels, or quant formats, ask whether the change works on RDNA2, RDNA3, and
  RDNA4.
- Document meaningful experiment results, including failures. Failed approaches
  are useful when they narrow the search space.
- Use one lock primitive: `hipfire-lock` `flock(2)`. Rust callers use
  `FlockGuard` and the shared path helpers. Shell, script, and external callers
  use `hipfire lock {acquire,release,status}`; `gpu-lock` is only an alias. Do
  not add sentinel files, pidfile liveness locks, `create_dir` mutexes,
  alternate lockfiles, or shell-only mutexes.
- Keep coexistence and compatibility tooling out of the inference binaries. Any
  import/export, format conversion, or interop tool (e.g. safetensors/GGUF
  import, model or LoRA conversion, adapter merge/bundle, external-format export)
  belongs in the `hipfire-coexistence` binary (or another dedicated tooling
  crate), not folded into the daemon, server, or runtime hot path. The inference
  path stays lean and HIP-direct; conversion and compatibility concerns are
  offline tooling.

## Branch And Git

- Use `chaingun` as the reference branch for further work. New work should
  happen directly on `chaingun` or be explicitly based on and compared against
  `chaingun`; do not treat `master` as the active baseline unless the user says
  so.
- Before meaningful changes, pull/rebase from the `chaingun` reference when the
  worktree state allows it.
- Preserve unrelated user changes. When committing or pushing, stage only files
  that belong to the current task and use descriptive messages.

## Verification

- Run `./tests/no-gpu-ci.sh` before handing off workflow-only changes.
- `./tests/coherence-gate-dflash.sh` is the canonical correctness gate after
  changes touching kernels, quant formats, dispatch, fusion, rotation, rmsnorm,
  or the spec-decode path.
- Model/runtime admission evidence belongs in `hipfire-eval` batteries or
  suites first. Shell gates should remain enforcement wrappers when they still
  provide baseline comparison or hook integration.
- Non-daemon GPU binaries do not self-lock. Coordinate GPU examples, benches,
  `hipfire eval`, and `hipfire-quantize` with
  `hipfire lock {acquire,release,status}` unless the called gate already does
  so.

## Artifact Names

Canonical artifact shape:

`<family>[-]<version>-<size[-effective/active]>[-tag1][-tag2...][.feature1[.feature2...]].<format>[.arch].hfq`

Periods are used to separate groups known to hipfire.
Examples:
- LFM2.5-1.2B-Thinking.bf16.hfq
- Qwen3.5-122B-A10B.mtp.vl.mq2l.hfq
- Gemma-4-8B-E4B-it-heretic-QAT.dflash.triattn.oq4++.gfx1151.hfq
- MedGemma-27B-it.triattn.hfq

This system allows machine parsing by working backwards:
- last field is always hfq
- dots separate machine-readable fields
- dashes separate human-readable fields, aside from size and effective/active size

Quant tokens use this shape:

`<family><bitwidth>[l][+][+]`

- `mq` / `MQ` is affine Magnum Quant.
- `oq` / `OQ` is symmetric Opus Quant. Do not use `op` for new artifacts.
- `l` after the bitwidth means Lloyd-Max/codebook MQ encoding, for example
  `mq4l`. Do not use `lloyd-mq4` for new artifacts.
- A first `+` means clip-search, SmoothQuant, AWQ, or a comparable
  activation-aware clipping/scaling pass.
- A second `+` means Hessian/LDLQ error feedback.
- Mixed-precision formats include a decimal place in the bitwidth, for example
  `mq4.5+` or `oq4.25++`.
- Examples: `mq4`, `mq4+`, `mq4++`, `mq4l`, `oq4`, `oq4+`, `oq4++`.

Notes:
- `family` and `version` may optionally include a dash. eg. Qwen3.5, Llama-3
- size with optional active/effective parameters. eg. 0.8b, 30B-A3B, 8B-E4B, 50M, 2.5T
- Use `.hfq` for all hipfire container artifacts.
- Use dotted model versions such as `Qwen3.5`. do not use qwen35 for example.
- Put calibration or transform modifiers that are not part of the quant token
  before it. Lloyd is part of the quant token: use `mq3l`, not `lloyd-mq3`.
- Do not use `+` for bundled roles or feature sidecars. Encode each feature as
  its own dot group before the quant token, for example `.mtp.vl.mq4.hfq` or
  `.dflash.triattn.oq4++.hfq`.
- Use role sidecars when loaded independently: `.mtp.hfq`, `.dflash.hfq`,
  `.jinja.`, `.hessian` and `.triattn.hfq`.
- The quant should detail the weight encoding. eg. Lloyd MQ2 uses `.mq2l.hfq`,
  Magnum uses `.mq4.hfq`
- `arch` must start with gfx followed by 3 or 4 numbers. eg. gfx906, gfx1103, gfx1151, gfx1201
- When a script, gate, registry, or doc uses an older format, update it to the
  canonical naming convention as part of the fix.
- Remove legacy-name fallback whenever you find it

## Local Routing

When a task primarily targets one of these subtrees, read that subtree's
`AGENTS.md` before editing even if Codex was started from the repo root.

- `.agents/` owns reusable agent skills and workflow instructions.
- `benchmarks/` owns prompt corpora, benchmark scripts, baselines, and results.
- `crates/hipfire-eval/` owns model/runtime evidence batteries and suites.
- `kernels/` and `crates/rdna-compute/` own HIP kernel and dispatch mechanics.
- `crates/hipfire-runtime/` owns the inference hot path, model execution, and
  runtime examples.
- `scripts/` owns reusable shell workflows and ad hoc benches outside the formal
  test gate directory.
- `tests/` owns enforcement wrappers and CI/smoke gates.

Add or tighten nested `AGENTS.md` files when a rule only applies to a subtree.

## Local Overlay

`AGENTS.local.md` is gitignored and may be absent. When present, it contains
machine-specific guidance such as hostnames, GPU fleet notes, SSH targets, and
lock pins.

@./AGENTS.local.md

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
