# Devlog 2026-05-27 — MTP single-GPU baseline + gfx1031 placement scoping

Branch `fix/q8-batched-masked-no-lds-cap` (now merged with master +
cherry-picked the daemon MTP-serve wiring `dfed971b` from origin/mtp-kevin).

## Setup

MTP code was NOT on this branch (it predated the MTP merge). Brought it in:
1. `git merge upstream/master` (197 commits) — CLEAN, 0 conflicts. Master
   has MTP core (`mtp_head.rs`, `mtp_spec.rs`, `mtp_compose.rs`) + examples
   (`mtp_only_demo`, `dflash_mtp_demo`) but NOT the daemon serve wiring.
2. Cherry-picked `ab3acbdb` (corpus) + `dfed971b` (daemon mtp-serve:
   `generate_mtp`, dual-source head load, sampling, multi-turn KV reuse)
   from origin/mtp-kevin. Clean apart from 2 trivial post-merge fixups:
   - qwen2 `LoadedModel` site needed `mtp: None` (master added that path)
   - `LoopGuard::from_env` → `from_config(config::get())` (master renamed)
   All libs + demos + daemon build clean.

## Single-GPU MTP — WORKS (gfx906, device 0)

`mtp_only_demo --target qwen3.6-27b.mq4 (AWQ) --mtp-head
/data/hipfire/qwen3.6-27b-cvs16384.mtp --max 64 --temp 0`:

- Loads 27B AWQ trunk + cvs16384 MTP head. Prefill 1850 tok/s.
- 34 cycles, 64 committed, **τ=1.85** (29 MTP accepted, 34 bonus),
  coherent `<think>` output. **6.45 tok/s** decode.
- τ=1.85 is modest (plan anatomy assumed τ≈3.8 on gfx1100; varies by
  content/greedy + this is a reasoning preamble). 6.45 tok/s reflects
  gfx906 running 27B-MQ4 on the unoptimized attention path — this step
  validated FUNCTIONALITY (load/compose/accept/emit), not peak perf.

## "MTP on the same GPU as PFlash" (gfx1031) — scoping

**Constraint:** 27B trunk needs ~20 GB; gfx1031 = 12 GB → the trunk
CANNOT live on gfx1031. So "MTP on the pflash GPU" must mean the
**asymmetric split** (the plan `docs/plans/mtp_multi_gpu_glm5.md`'s
favored case): MTP head (~800 MB) on gfx1031, trunk on gfx906.

**Why it's a build, not a flag:**
- `mtp_only_demo` + `mtp_spec` are single-GPU (`Gpu::init()` = device 0).
  `Gpu::init_with_device(1)` exists but would place the WHOLE model on
  gfx1031 → trunk OOM.
- PFlash's cross-device handoff is trivial (compress output is a
  host-side `Vec<u32>`, no peer-copy — daemon.rs:563). MTP is NOT: its
  per-cycle loop needs `prev_hidden` (~20 KB) shuttled gfx906→gfx1031
  EVERY cycle + candidate tokens back — `hipMemcpyPeerAsync` + cross-GPU
  event sync. That orchestration does NOT exist yet (the plan is
  analysis; no multi-GPU MTP code).

**Plan's ROI estimate:** synchronous offload ~12% (MTP head is only ~12%
of cycle wall); the asymmetric-async variant is the one worth building
(hides MTP latency, enables deeper K). Both require new cross-device
pipeline code.

## Next (decision needed)

Single-GPU MTP confirmed. gfx1031 placement options (all need new code):
- (A) Synchronous offload: head on gfx1031, per-cycle peer copy + 2 event
  barriers. ~12% projected, simplest of the multi-GPU options.
- (B) Asymmetric-async (plan's recommended): GPU1 speculates 1 cycle
  ahead "blind", interrupt-on-reject. Hides MTP latency, deeper K. Higher
  complexity.
- Prereq either way: a multi-GPU MTP harness (extend mtp_only_demo with
  `init_with_device` for the head + peer-copy prev_hidden), since the
  daemon serve path is also single-GPU for MTP today.
