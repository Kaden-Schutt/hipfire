# Autoresearch — program roadmap

## What it is
A reusable **generate → measure-on-real-daemon → gate → bank/compound** loop that drives
hipfire kernels (and eventually forward passes) toward their **per-arch roofline**, producing
a git-tracked, per-arch **evolutionary corpus** of optimal kernels. The moat is layered: the
loop *requires* hipfire's architecture to run at all (hot-swappable per-arch JIT *source*, no
ggml blob, no Python hot path, LLM-editable + re-measured in seconds), and it aims at consumer
RDNA where vendor kernels are weak and the headroom is large. A CUDA/ggml competitor would have
to rebuild into hipfire to run it — and would then be optimizing hardware cuBLAS already tuned
to death.

## Phases (in order)

### Phase 1 — Decode kernel-opt  [v1 DONE · v2 designed, pending build]
- **v1** (autonomous Codex on gfx1201): `baseline_v2` ≈150 tok/s, **+15.9%** a3b-mq4r decode,
  8 kernels folded, coherence-gated. Corpus + harness committed.
- **v2** (`harness-v2-design.md`): adaptive sampling (resolve `f`, not fixed floor), per-agent
  wins-only branches, per-kernel exhaustion + self-termination, computed-roofline stop metric.
- Models in scope: qwen3.6 **{a3b, 27b, 9b (gfx10)}**.

### Phase 2 — Prefill kernel-opt
Same harness-v2, pointed at the **prefill** workload. Prefill is **compute-bound** (GEMM/WMMA
hot-list) not memory-bound — so the compute levers (occupancy, WMMA tiling, dual-issue) that were
*neutral* on decode become the wins. Changes: metric (prefill throughput / TTFT), census target
(GEMMs), roofline lens (FLOPs). Loop structure unchanged.

### Phase 3 — Spec-decode autoresearch
Same measure-via-daemon loop over the spec-decode search space (drafter choice, tree shape,
accept/verify). Reuse the DFlash coherence/τ gates. Goal: autonomously tune τ / tok-s across
genres, per model × arch.

### Phase 4 (stretch) — Arch auto-implementation
Generate + integrate + gate a new model family's **Rust forward pass** (not just kernels).
Scaffold: the toy (0xFF) port template + coherence/KLD-vs-reference gates as the correctness
harness. Hardest layer — generation, integration, and correctness-gating all rise — but it
reuses the same generate→gate→bank engine. Self-building/self-optimizing, conditional on a model
smart enough for kernels **and** Rust.

## Cross-cutting: per-arch evolutionary trees
Kernel source is shared (one `.hip` → JIT per arch), so a gfx12-tuned win can clobber RDNA3/1
(precedent: r2lds coherent gfx1201/gfx1100, **incoherent gfx1151**). Each arch therefore grows
its **own** optimal-kernel lineage via the anti-clobber cross-arch gate + per-arch autoresearch
runs. `baseline_v2` is really `baseline_v2_gfx1200` until the gate maps each folded kernel
`{universal | gfx12-only}`.

**Fleet decode ceilings** (BW-scaled; realized only by per-arch runs, since decode is currently
*utilization*-bound not BW-bound): gfx1201 ~180 · gfx1100 (7900XTX, ~1.5× BW) **~200+** · gfx1151
· gfx1010/1030. Targets by SKU on gfx12: mq4r ~180, mq4 ~150, mq4p ~135.

## Near-term execution order
0. **Cross-arch transfer map** — `baseline_v2`'s 8 kernels → coherence+perf on hipx
   (gfx1100/gfx1151) → `{universal | gfx12-only}` routing map; seeds the per-arch trees.
1. **Build harness-v2** (decode) per spec + review (CAP, exhaustion-K, roofline-model).
2. **Validate v2 on hiptrx/gfx1201** vs `baseline_v2` (homogeneous, known-good) before deploying.
3. **Deploy v2 per-arch** — gfx1100 first (perf leader, ~200+ ceiling), then gfx1151, then gfx10 zoo.
4. Phase 2 (prefill) → Phase 3 (spec-decode) → Phase 4 (arch-autoimpl).
