# HipFire: RDNA GPU Unlock & Rust-Native Inference Engine

## Mission

Build a Rust-native ML inference (and eventually training) engine for AMD RDNA GPUs,
starting with the RX 5700 XT (gfx1010/RDNA1) on this machine (k9lin). The end goal is
a portable method that works across ANY RDNA generation (RDNA1→RDNA4), not just this card.

This project combines three efforts into one pipeline:
1. **autorocm** — Map and unlock ROCm on consumer RDNA hardware
2. **autokernel** — Optimize HIP/compute kernels for the specific hardware
3. **hipfire** — Rust-native inference engine (no Python in the hot path)

## Reference Projects (READ THESE FIRST)

Before writing any code or dispatching any agents, study these two projects deeply.
They define the methodology and architectural patterns we're following:

### 1. Karpathy's autoresearch
- https://github.com/karpathy/autoresearch
- Key pattern: `program.md` (strategy) → agent modifies single file → fixed eval → keep/discard → repeat
- We adapt this for hardware/driver exploration, not model training
- The "fixed eval" equivalent is our tiered ROCm validation harness (see harness.sh)

### 2. ncdrone/rustane
- https://github.com/ncdrone/rustane
- Key pattern: Rust-native FFI to private/undocumented hardware APIs via dlopen
- Their `ane-bridge` crate talks to Apple's Neural Engine through reverse-engineered private APIs
- We do the same thing but targeting AMD's ROCm/HIP/HSA runtime stack
- Study their architecture: ane-bridge (FFI layer) → metal-decode (GPU shaders) → engine (orchestrator)
- Our equivalent: hip-bridge (FFI layer) → rdna-compute (shader dispatch) → engine (orchestrator)

### 3. Also reference
- Mesa radeonsi/radv source — open AMD GPU driver, has gfx1010 support paths
- amdgpu kernel driver source — KMD ioctl surface, PM4 command buffer format
- ROCm source (especially the HSA runtime) — find the artificial gating checks

## Hardware Context

- **GPU:** AMD RX 5700 XT (Navi 10, gfx1010, RDNA 1)
- **GFX ID:** gfx1010
- **VRAM:** 8GB GDDR6
- **Status:** AMD officially refuses ROCm support for RDNA1. Consumer RDNA cards are artificially gated.
- **Known hack:** `HSA_OVERRIDE_GFX_VERSION=10.3.0` tricks ROCm into treating gfx1010 as gfx1030 (RDNA2). Unreliable, version-dependent, causes segfaults.

## Orchestration Model

You (Claude Code Opus) are the orchestrator. You make all architectural decisions.
You dispatch Sonnet subagents via the Task tool for parallel work.
You synthesize their findings and decide what to test and in what order.

**Reasoning budget:** You are running at max reasoning effort. Think hard at every
phase transition. The subagents are cheaper — dispatch them liberally for scoped tasks.

**Experiment tracking:** Git-commit every meaningful state change. Every approach tested
gets a commit with structured results. Failed approaches are just as valuable as
successful ones — document WHY they failed so the search space narrows.

```
git init (if not already)
git add -A && git commit -m "phase N: description of what changed and result"
```

## Phases

### Phase 0: Setup (~10 min)

1. Configure Serena plugin for this Rust project (you have the Serena plugin — figure out its init sequence for a new Rust workspace)
2. Verify Rust toolchain: `rustup default stable`, confirm 1.75.0+
3. Verify hardware visibility:
   - `lspci | grep -i amd` — confirm 5700 XT visible
   - `ls /dev/dri/` — confirm render nodes exist
   - `dmesg | grep -i amdgpu` — confirm kernel driver loaded
   - `cat /sys/class/drm/card*/device/vendor` — confirm AMD vendor ID
4. Check what's already installed: `dpkg -l | grep -i rocm`, `which hipcc`, `pip list | grep torch`
5. Initialize git repo, commit initial scaffold
6. Run `./harness.sh` to get baseline (expect most tiers to fail — that's the point)
7. Document starting state in `findings/phase0-baseline.md`

### Phase 1: Mapping (~2-4 hrs)

Dispatch 16 Sonnet subagents in parallel. Each agent gets a focused probe task.
They write structured findings to `findings/phase1-*.md`.

**Hardware probing agents (4):**
- Agent 1: Full hardware inventory — PCIe topology, IOMMU groups, power states, clock ranges, firmware versions. Dump everything from sysfs.
- Agent 2: KMD ioctl surface mapping — what ioctls does amdgpu expose? Which ones relate to compute dispatch? Read `/usr/include/drm/amdgpu_drm.h` or equivalent headers.
- Agent 3: Memory architecture — VRAM layout, GTT size, visible VRAM, doorbell pages. Map the memory hierarchy from sysfs + drm info ioctls.
- Agent 4: Current driver state — which amdgpu module params are loaded? What firmware blobs are present? What's in `/lib/firmware/amdgpu/navi10*`?

**ROCm compatibility agents (4):**
- Agent 5: ROCm version matrix — search online for every reported gfx1010 + ROCm version combination. Structure as: ROCm version → result (works/partial/fails) → failure mode → source URL.
- Agent 6: HSA runtime gating analysis — if ROCm source is available locally or online, find the exact checks that reject gfx1010. Is it a GFX ID allowlist? A feature capability check? Where in the code?
- Agent 7: HIP compilation path for gfx1010 — can hipcc target gfx1010 directly? What flags are needed? Does it need the GFX override or can it be told explicitly? Search ROCm issues and forums.
- Agent 8: rocBLAS/MIOpen gfx1010 status — these libraries ship precompiled kernels per GFX ID. Are gfx1010 kernels included in any version? If not, can they be compiled from source targeting gfx1010?

**Mesa/open-source path agents (3):**
- Agent 9: radeonsi OpenCL — does Mesa's rusticl or clover provide OpenCL on gfx1010? This could be an alternative compute path.
- Agent 10: Mesa's register headers for gfx10 — find `sid.h`, `gfx10_format_table.h`, etc. Map the compute-relevant registers (COMPUTE_DISPATCH_INITIATOR, shader resource descriptors, etc.)
- Agent 11: Compare gfx1010 vs gfx1030 ISA differences — what RDNA2 instructions are actually missing from RDNA1? This determines whether the HSA override hack is fundamentally sound or just lucky.

**Rust ecosystem agents (3):**
- Agent 12: Survey existing Rust AMD GPU crates — hip-rs, ocl (OpenCL), any direct amdgpu bindings. What's the state of the art?
- Agent 13: Study rustane's ane-bridge FFI pattern — how they dlopen private frameworks, wrap unsafe calls in safe Rust. Document the pattern for adaptation to HIP/HSA.
- Agent 14: Research candle-rs AMD support — candle has some ROCm support. What's the status? Could we build on it rather than from scratch?

**Note:** Vulkan/wgpu/RADV is explicitly **out of scope** as of 2026-04-25 (issue #44 closed). hipfire ships a single HIP/ROCm-direct backend; cross-vendor compute is not a goal.

**After all agents complete:** Synthesize findings into `findings/phase1-synthesis.md`.
Identify the actual blocking points (not folklore). Rank the viable paths forward.

### Phase 2: Theory & Competing Approaches (~1-2 hrs)

Based on Phase 1 synthesis, dispatch a SECOND wave of research agents.
These agents each advocate for a DIFFERENT approach. You want competition, not consensus.

Expected approach categories (adjust based on Phase 1 findings):

- **Approach A: Patch ROCm** — Find and bypass the gfx1010 gating. Compile ROCm components from source targeting gfx1010. Most direct path if feasible.
- **Approach B: Rust FFI to HIP/HSA directly** — Skip the ROCm userspace stack. dlopen libhsa-runtime64.so and libamdhip64.so directly, replicate the dispatch path in Rust. Like rustane does for ANE.
- **Approach D: Direct KMD dispatch** — Bypass all userspace. Talk to /dev/dri/renderD128 via amdgpu ioctls. Build command buffers (PM4 packets) in Rust. Maximum control, maximum effort.

**Note:** Vulkan-based approaches (former Approach C "compute baseline" and Approach E "hybrid") are out of scope as of 2026-04-25. We do not ship a second backend; cross-vendor compute is not a goal of this project.

Each approach gets a dedicated agent that writes a structured proposal to `approaches/approach-X.md`:
- Prerequisites and dependencies
- Estimated implementation effort
- Risk assessment (what could go wrong)
- Performance ceiling (theoretical max throughput)
- Portability to other RDNA generations
- Concrete first step to validate feasibility

**After all proposals:** You (Opus) rank them. Write `approaches/ranking.md` with your reasoning.
Pick the top 2-3 for Phase 3 validation.

### Phase 3: E2E Validation (~4-6 hrs)

Test approaches IN ORDER of your ranking. For each approach:

1. Implement the minimum viable version
2. Run `./harness.sh` — record which tiers pass
3. If it reaches Tier 4+ (actual compute works), keep going
4. If it fails below Tier 2, document why and move to next approach
5. Git commit results regardless

The harness tiers (see harness.sh for implementation):
- Tier 0: Does amdgpu kernel module load cleanly?
- Tier 1: Does the userspace runtime see the card?
- Tier 2: Can the compute runtime initialize?
- Tier 3: Can we allocate GPU memory and copy data?
- Tier 4: Can a simple compute kernel execute and return correct results?
- Tier 5: Can a matmul kernel run correctly?
- Tier 6: Performance — bandwidth and FLOPS relative to theoretical peak

**Key decision point:** After testing all ranked approaches, which path has the best
Tier reached + portability + Rust-native potential? That's your Phase 4 foundation.

Write decision to `experiments/phase3-decision.md`.

### Phase 4: Build the Engine (remaining time)

Using the validated approach from Phase 3, start building the actual Rust inference engine.

Target architecture (adapt based on what works):
```
hipfire/
├── crates/
│   ├── hip-bridge/      # (or kmd-bridge — whichever HIP path won)
│   │   └── src/lib.rs   # Safe Rust FFI to AMD compute runtime
│   ├── rdna-compute/    # Compute shader dispatch, kernel management
│   │   └── src/lib.rs   # Kernel compilation, buffer management, dispatch
│   └── engine/          # Inference orchestrator
│       └── src/lib.rs   # Model loading, tensor ops, inference loop
├── kernels/             # HIP compute shaders
│   ├── gemv.hip
│   ├── rmsnorm.hip
│   └── rope.hip
└── Cargo.toml
```

**Minimum Phase 4 deliverable:** Load a small model (e.g., TinyLlama 1.1B Q4),
run a single forward pass on the 5700 XT, get correct output tokens.
Performance doesn't matter yet — correctness first.

## Perf benchmarking (kernel perf changes)

Before claiming any kernel-level tok/s win: read
`docs/methodology/perf-benchmarking.md`. Within-session A/B is noisy on
gfx1100 (±10–15 % drift from DPM/thermal state); verify across a fresh
process with `scripts/probe_commits.sh $(git rev-parse HEAD~1) HEAD` and
confirm speed-gate passes before committing. The doc also keeps a
negative-result log of attempts that looked like wins in one-shell A/B
but measured as no-op or regression on fresh probe — check it before
starting a new kernel experiment.

**Diagnosing memset pressure:** run with `HIPFIRE_MEMSET_DUMP=1` — the
gpu layer's memset helper is `#[track_caller]` and prints `file:line`
per call. Grep the dump by source location, not by byte size. Note:
the `memset_async` helper is **gated by `active_stream` being `Some`**;
when the caller leaves `active_stream = None`, it silently falls
through to sync `hipMemset`. If you add new gated async memsets,
verify the caller actually sets a stream (fix pattern: create
`gpu.active_stream` at the top of the caller — see da2753e for
`spec_step_dflash`).

## Skills (`docs/skills/`)

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

When adding a new skill, give it a one-line index entry here so future
sessions find it without grepping.

## Coherence Gate (mandatory)

Any change to kernels, quant formats, dispatch, fusion, rotation, rmsnorm,
or the forward pass MUST pass `./scripts/coherence-gate.sh` before
committing. A pre-commit hook in `.githooks/pre-commit` runs it automatically
when relevant files are staged. Spec-decode changes also trigger
`./scripts/coherence-gate-dflash.sh` (see next section).

First-time setup (once per clone):
```
git config core.hooksPath .githooks
```

The coherence battery runs a small fixed matrix of prompts through the
daemon and writes a markdown report. It hard-fails only on panics, zero
tokens, or timeouts — soft output changes do NOT block, since legitimate
numerical-correctness fixes (e.g., norm convention) intentionally change
output. The committer reads the report and confirms each model is fluent,
on-topic, and not stuck in a verbatim loop before landing the commit.

This replaces the prior byte-exact `quality-gate.sh` barrier (removed),
which blocked legitimate forward-pass fixes by treating any token diff as
a regression.

## Coherence Probe (user-facing behavior debugger)

`coherence_probe` (in `crates/hipfire-runtime/examples/`) is the
user-facing version of the gate scripts: spawns the daemon, runs a
prompt, surfaces token attractors / special-token leaks / empty-think
halts / n-gram density spikes / tool-call malformations. Detector code
lives in `crates/hipfire-detect/`, a GPU-independent library crate that
the bash gates can also pipe into via a future thin CLI binary
(eliminates the inline-Python wart in
`coherence-gate-dflash.sh:191-243` and `agentic-gate.sh:72-144`).

Quick run:
```
cargo build --release --example coherence_probe
./target/release/examples/coherence_probe --self-check     # no GPU needed
./target/release/examples/coherence_probe \
    --model ~/.hipfire/models/qwen3.5-9b.mq4 \
    --prompt-file benchmarks/prompts/lru_cache_pep8_strict.txt \
    --max-tokens 200 --temperature 0.0
```

The probe sets `HIPFIRE_EMIT_TOKEN_IDS=1` on the daemon child it spawns;
the daemon then emits a parallel `{"type":"committed",...}` event
stream alongside the existing text events so the probe can run token-id
detectors (attractor windows, n-gram density, loop_guard mirror)
without re-tokenizing. The flag is off by default — existing JSONL
clients see no change. The 3-gram density detector promised below is
now implemented in `hipfire-detect::ngram` as a soft warn.

## DFlash Coherence Gate (spec-decode token-attractor guard)

Any DDTree / spec-decode / slow-path-kill change that claims a τ or tok/s
improvement MUST pass `scripts/coherence-gate-dflash.sh` (shipped 9883e98)
before commit. Enhanced three-tier thresholds (as of 2026-04-26):

**Tier 1 — First 128 tokens (hard fail, catches single-token attractors):**
- `unique_token_ratio < 0.15` OR `max_single_token_frequency > 0.50`

**Tier 2 — Last 128 tokens (hard fail, catches block-level attractors):**
- `unique_token_ratio < 0.30` OR `max_single_token_frequency > 0.50`

**Tier 3 — Full output (soft flag, requires human eyeball):**
- Consecutive 3gram repetition density > 50% in final half → structural loop signature
- Full-output unique-token ratio << 0.10 → structural code loop even if early tokens pass

**Why:** Attractors manifest in two forms: (1) single-token loops visible in first 128,
and (2) block-level structural loops (5+ token sequences repeating) that appear later.
CASK m-fold + DFlash 2026-04-26 example: τ=8.98 with tight stddev passed first-128 gate
but emitted 1500-token garbage (47-token vocabulary, 76+ reps of `[1734, 2357, 2733, 283, 869]`).
Root cause: m-fold hidden-state drift off draft distribution. Per `feedback_attention_precision.md`,
5% attention error cascades into attractor within ~10 tokens under greedy decode.

Bit DDTree Path A (fake +79% τ / +120% tok/s at 6c84b13) and Path B Variant B1 
(f9c920a, 2026-04-23) on identical `numbers(numbers(numbers(...` attractor were single-token.
Linearization-slot RoPE phase delta skew in tree-mode FA — not a bug, structural mismatch 
between tree-mode and committed-slot phase deltas.

**How to apply:** tight stddev on a spec-decode bench is actively
SUSPICIOUS, not reassuring. Real acceptance noise is wider. Any new
spec-decode bench script must include ALL of:
1. unique-token-ratio check on FIRST 128 (< 0.15 fail) AND LAST 128 (< 0.30 fail)
2. max-frequency check (> 50% fail) on both windows
3. decoded text printed for human eyeball (REQUIRED, not optional)
4. 3gram density check over second half of output (> 50% repetition → block-attractor flag)

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
- `crates/engine/src/tokenizer.rs:maybe_normalize_prompt()` — engine impl
- `crates/engine/examples/encode_prompt.rs` — verification utility
- commit 9a2c667 — root cause + bench data behind the default flip

**Canonical bench config (post-2026-04-26) for 27B-3.5 LRU code DFlash:**
```
max=120 --no-chatml --kv-mode asym3
PEP-8 strict prompt (\n\n\n between top-level defs)
prompt_normalize=true (default)
```
Expected: **199 tok/s τ=10.36** on 7900 XTX. ±2% deterministic. Drift >5% from
this is a regression — start with `git bisect` against this rule, not against
session-recalled "peak" numbers.

## GPU Lock Protocol (Multi-Agent)

When multiple Claude Code agents work in parallel (e.g. via worktrees), they coordinate
GPU access through `gpu-lock.sh`. **This is enforced automatically via hooks in
`.claude/settings.json`** — any `cargo` command triggers lock acquire before execution
and release after completion.

- Lock file: `/tmp/hipfire-gpu.lock`
- Contains a human-readable status like `model-ingestion agent is using the gpu`
- Agents poll every 5s (configurable via `GPU_POLL_INTERVAL`) when the GPU is busy
- Manual usage: `source gpu-lock.sh && gpu_acquire "<branch>" && gpu_release`
- Check status: `source gpu-lock.sh && gpu_status`

## Rules

1. **No Python in the inference hot path.** Python is allowed for tooling, benchmarks, comparison baselines. Never in the actual engine.
2. **Git commit everything.** Every experiment, every finding, every failed approach. The history IS the research.
3. **Document failures explicitly.** "Approach B failed because HSA_RUNTIME returns error code 0x1013 when initializing on gfx1010 without override" is more valuable than "it didn't work."
4. **Portability matters.** Every decision should consider: will this work on RDNA2? RDNA3? RDNA4? If it's 5700XT-only it's a hack, not a solution.
5. **No HSA_OVERRIDE_GFX_VERSION as a permanent solution.** It's acceptable as a temporary test during Phase 3, but the final engine must not depend on lying about the hardware identity.
6. **When blocked, search.** You have internet access. Use it aggressively — GitHub issues, AMD docs, Mesa source, phoronix forums, reddit r/ROCm, Tom's Hardware.
7. **No Vulkan / wgpu / cross-vendor compute backend.** Out of scope as of 2026-04-25 (issue #44 closed). hipfire ships a single HIP/ROCm-direct backend; cross-vendor coverage is not a goal of this project. If Phase 3 yields nothing, pivot to a different HIP-side approach (KMD direct, ROCm patch, HSA FFI), not to Vulkan.

## Success Criteria

- [ ] RX 5700 XT running compute workloads through a Rust-native path (no Python)
- [ ] At least one inference-relevant kernel (matmul/GEMV) executing correctly
- [ ] Documented method that generalizes to other RDNA generations
- [ ] All findings, approaches, and experiments committed to git with structured documentation
- [ ] Clear `NEXT-STEPS.md` for what to build next after this overnight session
