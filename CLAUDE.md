# hipfire: Rust-Native Inference Engine for AMD RDNA GPUs

## Mission

hipfire is a Rust-native ML inference engine (eventually training) for AMD RDNA
GPUs, built on a single HIP/ROCm-direct compute backend — no Python in the hot
path, no Vulkan/cross-vendor layer. It runs natively across RDNA generations
(RDNA1 gfx1010 → RDNA4 gfx1201) without lying about the hardware identity (no
`HSA_OVERRIDE_GFX_VERSION`). The project merges three efforts into one pipeline:

1. **autorocm** — map and unlock ROCm on consumer RDNA hardware
2. **autokernel** — optimize HIP/compute kernels per architecture
3. **hipfire** — the Rust-native inference engine itself

## Architecture

Full crate map, layering invariants, and request lifecycle live in
`docs/ARCHITECTURE.md` — this section is the short working summary.

The crate stack layers bottom-up with a strict one-way edge
`saddle-core -> hipfire-runtime -> hipfire-loader -> hipfire-engine ->
hipfire-generate -> hipfire-daemon` (`docs/ARCHITECTURE.md` § Layering).
User-facing entry is the native Rust CLI (`crates/hipfire-cli`),
backed by shared typed config, registry, and HTTP/JSONL client crates.
It posts to — or spawns — the inference daemon
(`crates/hipfire-daemon/src/main.rs`, ~3,879 lines, `[[bin]] name = "daemon"`,
JSON-lines stdio/HTTP server). Kernels are HIP source under `kernels/src/` —
see `docs/ARCHITECTURE.md` § Kernel build for precompiled `.hsaco` vs JIT
cache selection.

- **hip-bridge / hsa-bridge** — safe Rust FFI to the AMD HIP / HSA runtimes via
  `dlopen` (no link-time ROCm dependency).
- **rdna-compute** — kernel dispatch, hipGraph capture, JIT loader, arch-feature
  predicates.
- **hipfire-dispatch** — unified per-kernel-family dispatch; picks the kernel by
  quant format × arch capability × feature flags resolved at init.
- **saddle-core** — the substrate: grammar, KV, caps, sampling policy (depends
  only on `rdna-compute`, `hip-bridge`, `serde`).
- **hipfire-runtime** — arch-agnostic infra + `Architecture` trait; HFQ/safetensors,
  tokenizer, sampler, framing, spec primitives. LLaMA dense forward still lives
  here (`hipfire-arch-llama` is a facade; see `docs/ARCHITECTURE.md`).
- **hipfire-loader** — composition root: `Carrier` registry, single
  `load_model` dispatch, `LoadedModel`, continuous-batch staging. Depends on all
  arch crates; arch crates MUST NOT depend on it.
- **hipfire-engine** — arch-free serve engine: scheduler, terminal, emit, prompt.
  Declares no `hipfire-arch-*` dependency.
- **hipfire-generate** — generation bodies (`ar`, `qwen`, `dense`, `vision`,
  `batch`, Redline fixtures) that need both arch types and engine machinery;
  sits `hipfire-loader -> hipfire-engine -> hipfire-generate -> hipfire-daemon`.
- **hipfire-daemon** — the product binary. `[[bin]] name = "daemon"` in
  `crates/hipfire-daemon`; message dispatch and process lifetime only
  (`docs/ARCHITECTURE.md` § Serve / generate path).
- **hipfire-quantize** — CPU-side encoder (safetensors/GGUF → `.hfq`/`.mq4`/…);
  builds without a GPU.
- **hipfire-detect** — GPU-independent observability: attractor / special-token /
  n-gram / tool-call detectors (used by the gates and `coherence_probe`).
- **hipfire-config / hipfire-registry / hipfire-client / hipfire-cli** — typed
  TOML configuration, model catalog, dynamic registry, daemon/HTTP client, and
  the native operator/service surface.
- **hipfire-arch-\*** — one crate per model family's forward pass, keyed by
  `arch_id` (see `docs/architecture-ids.md` and `docs/ARCHITECTURE.md` §
  Architecture crates): llama (0/1), qwen35 (5/6), qwen2 (7), dots-ocr (8),
  deepseek4 (9, EP), minimax (10, EP), lfm2moe (11), cohere2moe (12), gemma4
  (13, 22), muse-glimmer (14). `toy` (0xFF) is the new-port template; the
  daemon refuses to dispatch it.
- Support crates: hipfire-atlas (perf corpus), hipfire-reap (MoE
  expert pruning), hipfire-tui (chat/settings TUI), redline (experimental
  direct-KMD compute, not wired into serving) — plus `hipfire-ds4-parent`
  (provisional) and `hipfire-pflash` (retained legacy, see below).

### PFlash status

PFlash is retained legacy research, not mainline or production functionality.
Prefix caching supersedes it for supported serving workloads. Its remaining
code and artifacts exist only for historical reference and reproduction; do
not treat PFlash as a production element, recommendation, acceptance route, or
basis for a current performance claim.

## Building, testing & gates

```
# Workspace build — no GPU/ROCm needed (HIP is dlopen'd at runtime); CI-required:
cargo build --release --workspace --all-targets --locked
# Product binaries — no --features required (required-features is 0):
cargo build --release
# → target/release/hipfire  (from crates/hipfire-cli, [[bin]] name = "hipfire")
# → target/release/daemon   (from crates/hipfire-daemon/src/main.rs, [[bin]] name = "daemon")
# No-GPU local check (cargo check + Rust tests + pytest):
./scripts/no-gpu-ci.sh
# Wire the pre-commit hook once per clone (sets core.hooksPath=.githooks):
./scripts/install-hooks.sh
```

CI (`.github/workflows/ci.yml`) runs `build` + `test --lib` (both required) plus
advisory `fmt`/`clippy` on every PR/push to `master` or `integration/**`; a daily
workflow refreshes `registry/v1.json`. There is no GPU CI runner yet — the
**GPU-bound correctness/perf gates run locally** (the pre-commit hook fires the
relevant ones when staged files match the kernel/serve/spec/pp hotspot globs).
The gate suite is documented in the sections below; `scripts/gates.sh` is the
unified wrapper. Conventions: don't `--no-verify`; don't bare `cargo fmt` (use
`scripts/fmt-changed.sh <master|beta>` — the base ref is mandatory, since a
branch cut from `beta` formatted against `master` reformats the whole
divergence); don't hand-edit `registry/v1.json` (it's generated by
`scripts/registry_gen.py` from the curated `registry/models.json`).

### The `cargo fmt` block (enforced, not advisory)

The repo carries historical rustfmt debt, so CI only checks CHANGED files. A
bare `cargo fmt` rewrites the whole workspace, buries the real change under a
200+ file diff, and makes review impossible. That rule used to be prose in this
file — prose an agent or a teammate can skip — so it is now enforced in three
independent layers:

1. **`.cargo/config.toml`** shadows the `fmt` subcommand, so `cargo fmt` exits
   101 with a self-documenting error. Covers humans, IDEs, and non-Claude
   agents; other cargo subcommands are unaffected.
2. **`scripts/check-fmt-bomb.sh`** (run by `.githooks/pre-commit`) blocks any
   commit where >25 staged `.rs` files changed *only* in whitespace — the
   signature of a workspace reformat, whatever produced it. Bypass a deliberate
   reviewed sweep with `SKIP_FMT_BOMB_CHECK=1`.
3. **`scripts/no-cargo-fmt-guard.py`** is a Claude Code `PreToolUse` hook that
   denies `cargo fmt` / unbounded `rustfmt` before the command runs, with a
   message pointing at `scripts/fmt-changed.sh`. It is self-scoping (no-ops in
   any repo lacking `scripts/fmt-changed.sh`) so it is safe to wire globally.
   Opt in per-machine by adding to `~/.claude/settings.json`:

   ```json
   { "hooks": { "PreToolUse": [ { "matcher": "Bash", "hooks": [
       { "type": "command",
         "command": "python3 '/abs/path/to/hipfire/scripts/no-cargo-fmt-guard.py'" } ] } ] } }
   ```

   Not shipped as a tracked `.claude/settings.json` on purpose: `.claude/` is
   gitignored here, and project hooks prompt every teammate for trust. Layers 1
   and 2 already cover a fresh clone.

Regression test for the hook: `python3 scripts/test-no-cargo-fmt-guard.py`
(42 command shapes — compound commands, subshells, `bash -lc '...'`, `ssh host
'...'`, find/xargs fan-out — no GPU, no deps). Layer 2 is only live once
`./scripts/install-hooks.sh` has been run in that clone.

## Reference architecture lineage

hipfire follows two patterns: ncdrone/rustane's Rust-native `dlopen` FFI to a
private hardware runtime (their `ane-bridge` → our `hip-bridge`/`hsa-bridge`),
and Karpathy's autoresearch single-file-experiment + fixed-eval loop (here the
"fixed eval" is the coherence/perf gate suite). Mesa radeonsi/RADV, the amdgpu
KMD ioctl surface, and the ROCm HSA runtime remain useful references for
kernel/driver work.

## Orchestration & experiment tracking

Heavy or parallelizable work is dispatched to subagents; the orchestrator
synthesizes findings and decides what to test in what order. **Git-commit every
meaningful state change** — every approach tested, including failed ones, gets a
commit with structured results. The history IS the research; document WHY a thing
failed so the search space narrows.

## Project history

Phases 0–4 (ROCm-unlock recon → approach bake-off → E2E validation → first
forward pass) bootstrapped the engine and are complete. For that archaeology see
`git log` and the `findings/` / `approaches/` history; day-to-day work now lives
in `crates/` and the gate suite below — there is no setup phase to re-run.

## Perf benchmarking (kernel perf changes)

Before claiming any kernel-level tok/s win: read
`docs/methodology/perf-benchmarking.md`. **Warm the kernel cache and
DPM state first** (a couple of throwaway forwards or
`HIPFIRE_DPM_WARMUP_SECS=10`); a cold first run is 3-7× slower and
NOT representative. Once warm, the within-session A/B noise band on
gfx1100 is **±1–3%** — anything bigger is a real signal, NOT
"DPM drift". Real regressions get hand-waived by inflated noise
claims; treat a 3%+ delta as something worth bisecting.

**JIT tax is per-(config × kernel-shape), and a slowdown that
SURVIVES a rerun is NOT JIT.** Kernels JIT-compile on first use *per
distinct shape* (each batch size / B-value / dtype / arch path caches
separately), so warming one cell of an A/B matrix does NOT warm the
others — warm EACH cell, and run the whole matrix a throwaway first
pass before measuring. The trap: seeing one cell slow on pass 1,
calling "JIT", and moving on. The discriminator is the rerun — if the
number snaps back toward its neighbour on pass 2 it was JIT; if it
holds (e.g. fp32-state DFlash stuck at 33 tok/s across both passes
while q8 went 105→151), it is a REAL kernel-perf gap (here: the
non-tree FP32 path reusing the single-token `gated_delta_net_f32`
instead of a batch-tiled `_batch_seq` kernel). Do not file a stable
cross-rerun slowdown under "JIT".

For cross-commit perf claims, verify across a fresh process with
`scripts/probe_commits.sh $(git rev-parse HEAD~1) HEAD` (it handles
warmup + multi-run aggregation correctly). The methodology doc also
keeps a negative-result log of attempts that looked like wins in
one-shell A/B but measured as no-op or regression on fresh probe —
check it before starting a new kernel experiment.

**Δ ≥ 5% investigation rule (mandatory).** Any perf delta whose
magnitude crosses ±5% warrants investigation. Do NOT shrug it off as
"within the ±10–15 % session noise band" — that band describes
worst-case spread, not the expected center, and a ±5% point estimate
is most likely real signal partly masked by noise. Walk the rule
cheapest-step first:

1. **Warming first (always cheapest, always required).** Re-run 3–5
   times with the established protocol — one `--max 16` warmup per
   cell, gpu-tcas-coordinated, fresh process per measure, byte-identical
   prompt (md5 recorded). Take the median of the 3–5 measures.
   - Median snaps back to baseline → thermal/DPM/cache noise. Record
     and close.
   - Median holds (still ≥5%) → the delta is real. Continue.
2. **If real LOSS: investigation rule activated.** Walk in order
   (cheapest diagnostic first): kernel occupancy (use the
   `gfx-kernel-metadata` skill — VGPR/SGPR/LDS/spill from `.hsaco`),
   rocprof attribution, env state (ROCm version, kernel cache,
   sccache, mold, DPM governor), flag state (`HIPFIRE_*` env vars,
   `--kv-mode`, `--no-chatml`, `prompt_normalize`, prompt md5), then
   code-change bisect via `scripts/probe_commits.sh`.
3. **If real GAIN: validate the actual runtime path before ANY claim.**
   Use `scripts/redline_daemon_harness.py` for retained-launch capture,
   contract checks, and multi-position HIP/PM4 output parity. Use
   `scripts/serve_harness.py` for sampled, chained, or session-level output
   checks through the user-facing serve path. The retired
   `scripts/coherence-gate*.sh` family is stale and MUST NOT be used as
   acceptance evidence.

**Diagnosing memset pressure:** run with `HIPFIRE_MEMSET_DUMP=1` — the
gpu layer's memset helper is `#[track_caller]` and prints `file:line`
per call. Grep the dump by source location, not by byte size. Note:
the `memset_async` helper is **gated by `active_stream` being `Some`**;
when the caller leaves `active_stream = None`, it silently falls
through to sync `hipMemset`. If you add new gated async memsets,
verify the caller actually sets a stream (fix pattern: create
`gpu.active_stream` at the top of the caller — see da2753e for
`spec_step_dflash`).

## Skills (`.agents/skills/`)

Reusable how-tos kept out of CLAUDE.md to avoid bloat. Each skill is a
self-contained reference; reach for it by name when the situation
matches. Index of currently-available skills:

- **`gfx-kernel-metadata`** — extract VGPR/SGPR/LDS/spill counts from
  a compiled `.hsaco` and compute theoretical occupancy. Covers all
  CDNA (gfx906/908/90a/942 wave64) and RDNA (gfx10xx through gfx1200+
  wave32) archs. **Reach for this when:** verifying zero spills after
  a kernel change, computing occupancy headroom, comparing register /
  LDS budgets across kernel variants, or interpreting
  `__launch_bounds__` tradeoffs. Manual disassembly via
  `clang-offload-bundler` + `llvm-readelf` is fiddly enough that the
  skill doc is faster to follow than to rederive.

- **`serve-restart`** — cleanly stop, free :11435, and restart
  `hipfire serve`. **Reach for this when:** serve "Failed to start
  (port in use)", a stale daemon holds VRAM, a pre-warm JSON-parse /
  os-error-2 crash left a zombie `daemon.pid` singleton, or you need a
  guaranteed-fresh daemon. Kills the native serve + spawned daemon, fuser-frees
  the port, reaps pid/lock files. `scripts/serve-restart.sh [port]`.

- **`agent-memory`** — git-tracked cross-session project memory as in-repo markdown
  notes (`.agent-memory/notes/*.md`, YAML frontmatter) with **lexical (ripgrep)
  recall** via `scripts/mem.sh` — no DB, no embeddings, no MCP process. **Reach for
  this when:** answering a project/architecture question, starting a task like past
  work, or you've learned something durable (finding / falsification / decision /
  gotcha). Recall before answering (`scripts/mem.sh recall <terms>`), remember after
  learning (`scripts/mem.sh remember <slug> "<title>" tags`). Project findings go
  here (shared, diffable, travel with the code); personal/fleet notes stay in global
  memory. See `.agent-memory/` and `scripts/mem.sh`.

When adding a new skill, give it a one-line index entry here so future
sessions find it without grepping.

## Runtime validation (mandatory)

Do not use `scripts/coherence-gate*.sh`. Those fixed, single-request batteries
are retired: their model matrix and prompt assumptions are stale, and they can
mislabel both regressions and valid numerical changes.

Choose validation by the path under test:

- Kernel, dispatch, graph, or Redline replay changes: run
  `scripts/redline_daemon_harness.py`. Require stable capture, valid AQL
  contracts, and multi-position HIP/PM4 output parity. Record the JSON report.
- User-facing generation or state-lifecycle changes: run
  `scripts/serve_harness.py` against the exact model and settings under test.
  Use `battery` for varied prompts, `chain` for related turns, and `session`
  for recurrent-state/reset behavior. Record the per-turn JSON and decoded
  text.
- Performance claims: combine the applicable harness above with the matched,
  fresh-process timing protocol in this file. A microbenchmark alone is not
  acceptance evidence.

First-time setup (once per clone):
```
git config core.hooksPath .githooks
```

## Prompt-structure τ sensitivity (mandatory bench rule)

**One newline character can swing τ by 17% on 27B DFlash.** Two prompts
that tokenize to the same number of tokens (e.g. both 232) but with
different whitespace patterns produce dramatically different draft-target
acceptance:

```
PEP-8 strict (\n\n\n between top-level defs):    27B-3.5 LRU max=120  → 161 tok/s τ=8.07 (deterministic ±2%)
Single-blank (\n\n between top-level defs):      27B-3.5 LRU max=120  → 184 tok/s τ=9.42 (range 173-204)
```

**Why:** identical token COUNT, different token SEQUENCE → different
prefix-conditioned distribution shape at each position → different
draft/target argmax alignment → different τ. Same model, same flags,
same kernels, same binary md5.

**How to apply:** ANY tok/s or τ comparison across sessions, agents, or
commits MUST use byte-identical prompts. Embed prompts as committed
files (not heredocs in scripts that get reformatted by editors), and
record the prompt md5 alongside the result. A 14% perf delta from a
whitespace cleanup is invisible in code review but catastrophic for
benchmarking. Discovery cost ~6 hours of phantom-regression chasing
on 2026-04-24 (rocBLAS, DKMS, firmware, kernel cache, mold/sccache,
DPM — all null) before isolating to a single newline.

**Corollary**: agent-to-agent perf claims that lack prompt md5 are
unverifiable. Don't accept "X agent got Y tok/s" without reproducing
on the exact prompt bytes they ran.

**Mitigation (Phase 1 implemented):** The engine collapses all 3+ consecutive
newlines to exactly 2 before tokenization. This eliminates the whitespace-
variance source entirely, making PEP-8 and single-blank prompts tokenize
identically.

**DEFAULT ON since 2026-04-26.** The original Phase 1 ship gated this behind
`HIPFIRE_NORMALIZE_PROMPT=1` opt-in, but empirical bench showed it's worth
+24% τ on PEP-8 code prompts (159 → 196 tok/s on 27B-3.5 LRU DFlash) without
correctness cost. Opt out with `HIPFIRE_NORMALIZE_PROMPT=0` (or
`prompt_normalize=false` in config) only when raw `\n{3,}` whitespace is
semantically load-bearing. See:
- `crates/hipfire-runtime/src/tokenizer.rs:maybe_normalize_prompt()` — engine impl
- `crates/hipfire-runtime/examples/encode_prompt.rs` — verification utility
- commit 9a2c667 — root cause + bench data behind the default flip

**Canonical bench config (post-2026-04-26) for 27B-3.5 LRU code DFlash:**
```
max=256 --no-chatml --kv-mode q8
PEP-8 strict prompt (\n\n\n between top-level defs)
prompt_normalize=true (default)
```
DFlash perf gates must use `q8` or an FWHT KV mode. Do not use `asym*` KV modes
for DFlash perf/gate claims; older pre-q8 DFlash perf numbers are historical
only. Drift >5% from the current q8/max256 baseline is a regression
— start with `git bisect` against this rule, not against session-recalled
"peak" numbers.

## Hard-won measurement & debugging rules

Cross-cutting rules distilled from the memory log — promoted here so they are not
re-learned per session.

- **Measure spec-decode durability on the DAEMON, not the demo harness.**
  `dflash_spec_demo` / `mtp_only_demo` run raw continuation and under-report by
  ~40% — they never trigger the chatml + structured-thinking serving behavior
  that makes even novel content predictable (τ jumps 1.05 → 2.2+). A
  "fails on prose/fiction" verdict from a demo is a harness artifact, not a
  product verdict; confirm τ/tok-s through the daemon (`serve`, the path users
  hit) before calling a genre drafter-bound.
- **Offline spec-decode proxies mis-rank drafters — always confirm online.** An
  offline argmax / CE-vs-KL ranking reversed on-device (the proxy rewards the
  training objective, not real acceptance τ). Never ship a drafter decision on
  an offline proxy alone — budget one online sweep first.
- **Garbage output → swap to a KNOWN-GOOD model FIRST, before debugging the
  engine.** A wrong-recipe model (e.g. dense-AWQ on an MoE) lobotomizes output
  and burns engine-debugging cycles. Coherence gates must include a bare factual
  prompt — strong/code prompts mask a lobotomy.
- **Byte-parity validation is meaningless under stochastic state — pin FP32 +
  `HIPFIRE_DETERMINISTIC=1`.** Q8 DeltaNet-state stochastic rounding makes an
  exact TP/seam path look non-bit-exact ("ULP cascade") and mis-attributes the
  bug. Use FP32 DeltaNet state for any byte-parity claim.

## GPU Lock Protocol (Multi-Agent)

When multiple Claude Code agents work in parallel (e.g. via worktrees), they coordinate
GPU access through `scripts/gpu-lock.sh`. **Coordination is currently MANUAL** — there is
no committed hook that auto-acquires the lock. (`.claude/settings.json` is not tracked in
the repo; if you want `cargo` commands to auto-acquire/release, wire a PreToolUse/PostToolUse
hook in your own local `.claude/settings.json` that sources `scripts/gpu-lock.sh`.)

- Lock file: `/tmp/hipfire-gpu.lock` (override with `HIPFIRE_GPU_LOCKFILE`)
- Backed by `flock(1)` held on an open fd, so the **kernel auto-releases the
  lock when the holder dies for any reason** (kill -9, crash, OOM, terminal
  close). Stale locks are structurally impossible — **never `rm` the lockfile**
  (unlinking an flock'd file lets a second acquirer lock a fresh inode → two
  holders). If validation seems "stuck on a stale lock", it isn't stale: a live
  holder is genuinely using the GPU — check `gpu_status`.
- Records `agent pid=… host=… acquired=…` in the file for diagnostics
- While a live holder is busy, waiters print status every 5s
  (`GPU_POLL_INTERVAL`) and give up after `GPU_LOCK_TIMEOUT` (default 1800s;
  `0` = wait forever), exiting non-zero instead of hanging indefinitely
- `gpu_acquire` is reentrant within a process tree (`HIPFIRE_GPU_LOCK_OWNER`),
  so nested gates (e.g. `pp-gate` under `speed-gate`) don't self-deadlock
- Manual usage: `source scripts/gpu-lock.sh && gpu_acquire "<branch>" && gpu_release`
- Check status: `source scripts/gpu-lock.sh && gpu_status`
- Regression test (no GPU): `bash scripts/test-gpu-lock.sh`

## Rules

1. **No Python in the inference hot path.** Python is fine for tooling,
   benchmarks, and comparison baselines — never in the engine itself.
2. **Git commit everything.** Every experiment, finding, and failed approach.
   The history IS the research.
3. **Document failures explicitly.** "Approach B failed because HSA_RUNTIME
   returns 0x1013 on init without override" beats "it didn't work."
4. **Portability matters.** Every decision should consider RDNA2/3/4. If it only
   works on one GPU arch, it's a hack, not a solution.
5. **No `HSA_OVERRIDE_GFX_VERSION` as a permanent solution.** Acceptable only as
   a throwaway local test; the engine must never depend on lying about the
   hardware identity.
6. **When blocked, search.** Use the internet aggressively — GitHub issues, AMD
   docs, Mesa source, forums.
7. **No Vulkan / wgpu / cross-vendor backend.** Out of scope (issue #44, closed
   2026-04-25). hipfire ships a single HIP/ROCm-direct backend; if a compute path
   stalls, pivot to another HIP-side approach (KMD-direct, ROCm patch, HSA FFI),
   not Vulkan.

## Local / personal config

Machine-specific context — your GPU fleet, ssh hosts, device pins, local paths —
does NOT belong in this committed file (it confuses contributors). Keep it in a
personal file that Claude Code auto-loads but git ignores. Recommended:
`~/.claude/hipfire-fleet.md` (loads in every checkout/worktree, pulled in by the
import below). A per-repo `CLAUDE.local.md` is also gitignored + auto-loaded, but
only in the single worktree it sits in. The import below is a no-op for anyone
who doesn't have the file.

@~/.claude/hipfire-fleet.md

# Code intelligence — CodeGraph

GitNexus is **removed from this project**. Do not run it, do not restore its
block into this file, and do not follow any `gitnexus://` resource, MCP tool,
or `.claude/skills/gitnexus/**` skill file: none of them are provisioned here.
Tooling that re-injects a GitNexus marker block into CLAUDE.md or AGENTS.md is
re-adding a dependency this project already dropped; delete it rather than
merging it.

The code-intelligence tool for hipfire is **CodeGraph** — a per-project SQLite
graph of symbols, edges, and call paths, queried through the
`codegraph_explore` MCP tool or the equivalent `codegraph` CLI. See
[AGENTS.md](AGENTS.md) § "Code intelligence — CodeGraph" for the full usage
contract; the short version:

- `codegraph_explore` FIRST, before a grep/read loop or an edit. Source it
  returns is already read — do not re-open those files.
- Point it at a checkout that actually has a `.codegraph/` index; fresh
  worktrees start without one (`codegraph init .`). The index is machine-local
  and gitignored.
- `lsp references` — not CodeGraph — is the authority on "did I get every
  callsite". CodeGraph shows call paths; the language server shows the set.
- NEVER rename across files with find-and-replace; use `lsp rename`.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **hipfire** (54797 symbols, 169392 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "master"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/hipfire/context` | Codebase overview, check index freshness |
| `gitnexus://repo/hipfire/clusters` | All functional areas |
| `gitnexus://repo/hipfire/processes` | All execution flows |
| `gitnexus://repo/hipfire/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
