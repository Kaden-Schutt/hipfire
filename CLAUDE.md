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

**Mesa/open-source path agents (4):**
- Agent 9: RADV compute shader dispatch — Vulkan compute works on gfx1010 via RADV. Trace how RADV dispatches compute shaders. This is a known-working path to the hardware.
- Agent 10: radeonsi OpenCL — does Mesa's rusticl or clover provide OpenCL on gfx1010? This could be an alternative compute path.
- Agent 11: Mesa's register headers for gfx10 — find `sid.h`, `gfx10_format_table.h`, etc. Map the compute-relevant registers (COMPUTE_DISPATCH_INITIATOR, shader resource descriptors, etc.)
- Agent 12: Compare gfx1010 vs gfx1030 ISA differences — what RDNA2 instructions are actually missing from RDNA1? This determines whether the HSA override hack is fundamentally sound or just lucky.

**Rust ecosystem agents (4):**
- Agent 13: Survey existing Rust AMD GPU crates — ash (Vulkan), hip-rs, ocl (OpenCL), any direct amdgpu bindings. What's the state of the art?
- Agent 14: Study rustane's ane-bridge FFI pattern — how they dlopen private frameworks, wrap unsafe calls in safe Rust. Document the pattern for adaptation to HIP/HSA.
- Agent 15: Evaluate wgpu compute shaders as a path — wgpu targets Vulkan on AMD. Could this be a "it just works" baseline while we build the HIP path?
- Agent 16: Research candle-rs AMD support — candle has some ROCm support. What's the status? Could we build on it rather than from scratch?

**After all agents complete:** Synthesize findings into `findings/phase1-synthesis.md`.
Identify the actual blocking points (not folklore). Rank the viable paths forward.

### Phase 2: Theory & Competing Approaches (~1-2 hrs)

Based on Phase 1 synthesis, dispatch a SECOND wave of research agents.
These agents each advocate for a DIFFERENT approach. You want competition, not consensus.

Expected approach categories (adjust based on Phase 1 findings):

- **Approach A: Patch ROCm** — Find and bypass the gfx1010 gating. Compile ROCm components from source targeting gfx1010. Most direct path if feasible.
- **Approach B: Rust FFI to HIP/HSA directly** — Skip the ROCm userspace stack. dlopen libhsa-runtime64.so and libamdhip64.so directly, replicate the dispatch path in Rust. Like rustane does for ANE.
- **Approach C: Vulkan compute baseline** — Use RADV (which already works) as the compute backend. Write Rust inference engine on wgpu or ash. Less optimal but known-working.
- **Approach D: Direct KMD dispatch** — Bypass all userspace. Talk to /dev/dri/renderD128 via amdgpu ioctls. Build command buffers (PM4 packets) in Rust. Maximum control, maximum effort.
- **Approach E: Hybrid** — Vulkan for known-working baseline, HIP FFI for optimized paths, fallback gracefully.

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
│   ├── hip-bridge/      # (or vulkan-bridge, or kmd-bridge — whatever won)
│   │   └── src/lib.rs   # Safe Rust FFI to AMD compute runtime
│   ├── rdna-compute/    # Compute shader dispatch, kernel management
│   │   └── src/lib.rs   # Kernel compilation, buffer management, dispatch
│   └── engine/          # Inference orchestrator
│       └── src/lib.rs   # Model loading, tensor ops, inference loop
├── kernels/             # HIP/Vulkan compute shaders
│   ├── gemv.hip         # (or .comp for Vulkan SPIR-V)
│   ├── rmsnorm.hip
│   └── rope.hip
└── Cargo.toml
```

**Minimum Phase 4 deliverable:** Load a small model (e.g., TinyLlama 1.1B Q4),
run a single forward pass on the 5700 XT, get correct output tokens.
Performance doesn't matter yet — correctness first.

## Rules

1. **No Python in the inference hot path.** Python is allowed for tooling, benchmarks, comparison baselines. Never in the actual engine.
2. **Git commit everything.** Every experiment, every finding, every failed approach. The history IS the research.
3. **Document failures explicitly.** "Approach B failed because HSA_RUNTIME returns error code 0x1013 when initializing on gfx1010 without override" is more valuable than "it didn't work."
4. **Portability matters.** Every decision should consider: will this work on RDNA2? RDNA3? RDNA4? If it's 5700XT-only it's a hack, not a solution.
5. **No HSA_OVERRIDE_GFX_VERSION as a permanent solution.** It's acceptable as a temporary test during Phase 3, but the final engine must not depend on lying about the hardware identity.
6. **When blocked, search.** You have internet access. Use it aggressively — GitHub issues, AMD docs, Mesa source, phoronix forums, reddit r/ROCm, Tom's Hardware.
7. **If Phase 3 yields nothing by Tier 4, pivot.** Fall back to Vulkan compute (Approach C) which is known-working. A working Vulkan engine is infinitely better than a non-working HIP engine.

## Success Criteria

- [ ] RX 5700 XT running compute workloads through a Rust-native path (no Python)
- [ ] At least one inference-relevant kernel (matmul/GEMV) executing correctly
- [ ] Documented method that generalizes to other RDNA generations
- [ ] All findings, approaches, and experiments committed to git with structured documentation
- [ ] Clear `NEXT-STEPS.md` for what to build next after this overnight session
