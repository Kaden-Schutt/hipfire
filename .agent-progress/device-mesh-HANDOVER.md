# Device-mesh — HANDOVER for the next session

> ⚠️ **PIVOT (2026-07-06) — the executor plan changed. Read the plan's `## PIVOT` section BEFORE
> setting a goal.** The master merge made `execute_steps` (63 call sites) the dense spine and reverted
> our qwen2→`run_layer_program_mesh` wiring. The new ONE executor is **`execute_steps(mesh, gpus)`** on
> the `Step` IR — NOT `run_layer_program` on `SuperOp`. The loader/placement half (below, "What's DONE")
> is all still valid and done; the *executor* half re-sequences to phases **P-A…P-E**. **Suggested next
> goal:** `/goal device-mesh P-A: big-bang execute_steps(&mut Gpu → mesh, gpus) signature flip, byte-identical`.
> Full reconciliation + keep/rework/orphan in the plan `## PIVOT` and the git-tracked note
> `.agent-memory/notes/device-mesh-pivot-execute-steps-spine.md` (`scripts/mem.sh recall execute_steps mesh pivot`).

**Set a *single-phase* goal** (not the whole roadmap — that's a multi-week, one-PR-per-phase effort by
design; a session-scoped `/goal` on all phases loops forever).

## Where things are
- **Branch:** `feature/device-mesh` (worktree `.claude/worktrees/feature+device-mesh`),
  off `feature/parallel-expansion` (which carries `HIPFIRE_EMULATE_GPUS` — the single-card
  multi-rank harness that P-B/P-C/5a validation depends on). Tree clean,
  workspace builds with 0 errors, all no-GPU tests green (0 failures).
- **Plan:** `docs/superpowers/plans/2026-07-05-device-mesh-transparent-parallelism.md`
  — the **`## PIVOT` section is authoritative** (the older "IMPLEMENTATION STATUS" + §1 + phase table
  are marked SUPERSEDED inline but kept for the mesh-tree / manifest / safety design record).
- **Commit map:** `.agent-progress/device-mesh-status.md`.

## What's DONE (pure layer + 2 GPU-validated integrations)
- `hipfire-hardware` leaf crate: `Gpus`+collectives (extracted from `multi_gpu`, config→`DeviceResolveOpts`), `DeviceMesh` (`mesh.rs`), `CollectiveHint`. Coherence-gate validated.
- EP executor relocated to `hipfire-dispatch` (`ep.rs`), now mesh-driven (`ep_decode_parity` tp=1 anchor PASS on qwen3.6-35b-a3b).
- `hipfire_runtime::config::resolve_mesh`, `hipfire_runtime::weight_manifest::*`:
  `ShardPolicy`/`WeightEntry`/`StateEntry`, `collective_for_policy`, `layer_collectives`,
  `placement_devices`, `validate_manifest`, **`plan_manifest` → `ManifestPlan`** (the full
  deterministic compile). `Architecture::{weight_manifest,state_manifest}` implemented for
  llama, qwen2, minimax, toy.

## DONE (last 2 sessions): `fulfill_manifest` whole-tensor + ExpertSharded (Phase 2 GPU exec)
`crates/hipfire-runtime/src/weight_store.rs` — `fulfill_manifest(weights, mesh, n_layers, gpus,
source) -> Result<WeightStore, FulfillError>`. GPU-validated on gfx1151 (`fulfill_manifest_probe`:
single-1×1 + emulated PP-2 + emulated EP-2; placement + `memcpy_dtoh` byte-oracle; the oracle
caught a real missing-`layer`-in-key bug + a rollback smoke). Implemented: whole-tensor upload
(single/PP/Replicate/Pin/Tied→Alias) + **`ExpertSharded`** (each rank = compact blob of its owned
experts, generic expert-outermost gather via `expert_compact_blob` + `ShardConfig`) + **§6
transactional guard** (mid-load failure → `free_all` the partial uploads, return `Err`; no VRAM
leak). **Dense-TP slice returns `Err`** (Phase 5). `source(entry)->bytes` closure keeps arch
on-disk HFQ naming out of the engine. Additive
— forward untouched (Tier-1; store-read is Phase 3). NOTE: the EP path produces the placed *bytes*;
the per-expert pointer-table + zeroed-dummy the deepseek4 kernel indexes through
(`crates/hipfire-arch-deepseek4/src/arch.rs:163-333`) is forward-consumption, wired in Phase 3.

**Also DONE (arch manifests):** qwen35 `state_manifest` (hybrid Kv+Recurrent+Conv by `layer_types`)
and **deepseek4 `weight_manifest` + `state_manifest`** (MLA replicated, routed experts
`ExpertSharded`, `num_hash_layers` gate-bias split; compressor/indexer/HC/MTP scoped out — all
`Replicate`/file-shaped; 2 unit tests). Phase-2 arch coverage now: llama, qwen2, minimax, toy
(weight+state); qwen35 (state only); deepseek4 (weight+state). Only **qwen35 `weight_manifest`**
(DeltaNet fused projections + MoE variants) is left in Phase-2 arch coverage.

**Also DONE (Phase 3 START): store-backed REAL llama load.** `source(entry)` now returns
`(bytes, DType)` (real quant type → forward-ready store tensor, not `Raw`). `examples/llama_store_load.rs`
loads `qwen3-0.6b-llama.mq4`'s quantized projections via generic `fulfill_manifest` + a llama
HFQ-backed source and byte+dtype-matches bespoke `Llama::load_weights` — **196 tensors/28 layers,
identical (MQ4G256), GPU-validated**. Loader name map: HF names `model.layers.{i}.self_attn.q_proj.weight`
via `hfq::load_weights_hfq` (NOT the GGUF `blk.*` path); quant projections upload raw/verbatim (match),
norms/embed/tied-lm_head do F16→F32 host dequant (scoped out).

**Also DONE (Phase 3, WHOLE-MODEL, bit-exact): store→forward.** llama `weight_manifest` gained
q_norm/k_norm; a universal `source` (quant_type 1→F16→F32, 2→F32, else raw+real dtype) covers the
whole model; `llama_store_load` fulfills the FULL manifest, assembles a complete `LlamaWeights` from
the store, and the forward is **logit-IDENTICAL to bespoke (max |Δ|=0, 311 tensors, gfx1151)** — a
drop-in bit-exact replacement for the bespoke llama loader on a single GPU.

**Also DONE (Phase 3, PP-2 PLACEMENT): mesh-driven pipeline load.** llama `output_norm`→`Pin(Output)`;
`llama_store_pp` fulfills the WHOLE manifest on a PP-2 emulated mesh, asserts mesh-correct banding
(311 tensors 155/156; embed→0, output_norm+lm_head→1, layers by `stage_for_layer`), and the gathered
forward is **logit-IDENTICAL to bespoke (max |Δ|=0, gfx1151)**. The load half of PP is validated.

**Also DONE (Phase 1c: PP-2 banded EXECUTION).** Refactored llama's forward: `forward_scratch_band(
gpu, w, cfg, layer_range, pos, kv, scratch)` (range-parameterized layer loop) + `forward_scratch_head`
(final norm + lm_head); `forward_scratch_compute` = band(0..n)+head (bit-exact). `llama_store_pp` runs
a REAL banded PP forward — stage0 embed+band(0..14)/dev0 → `boundary_copy` → stage1 band(14..28)+head
/dev1 — **logit-IDENTICAL to bespoke (max |Δ|=0)**. Full pipeline-parallel LOAD + EXECUTE, mesh-driven.

## The NEXT unit — the executor half now sequences P-A → P-E (plan `## PIVOT`)

**START HERE — P-A (mechanical, byte-identical):** big-bang flip
`execute_steps(gpu: &mut Gpu, …)` → `execute_steps(mesh: &DeviceMesh, gpus: &mut Gpus, …)`
(`crates/hipfire-dispatch/src/pipeline/steps.rs:600`) across **all 63 call sites** + the forward
drivers that hold `&mut Gpu` (`dense_forward` `arch_spec.rs:131`, qwen35 `forward_from_x_gpu`,
cohere2moe `decode_step_body`, `superop::prefill_forward`). Every caller passes `DeviceMesh::single()`.
Internally degenerate to `gpus.devices[0]`, **never call `ensure_rank_streams`** (the memset sync→async
trap). Validate: per-arch committed-token md5 A/B == pre-flip (`HIPFIRE_FORWARD_LOWERED`-style) +
`coherence-gate.sh`. This threads the mesh to the chokepoint with zero behavior change — the safe
foundation everything else builds on. Then: **P-B** TP-in-execute_steps (per-`Step` `ShardPolicy` shard +
`Tp` all-reduce), **P-C** PP-at-driver (generalize `forward_scratch_band`+`boundary_copy` into
`dense_forward`), **P-D** `Step::Moe` + EP fold (retire `run_layer_program_mesh`/`ep.rs`), **P-E**
`Step::Recurrent`/`Conv` + DeltaNet head-shard.

**Still-valid parallel/later units (loader half, unblocked, do anytime):**
- **qwen35 `weight_manifest`** — finishes Phase-2 arch coverage; pure-CPU DeltaNet loader study
  (`qwen35.rs:2876-2945`, per-`LayerType` weight sets: LinearAttention fused
  `in_proj_qkv`/`in_proj_z`/`in_proj_a`/`in_proj_b` + `A_log`/`dt_bias`/`conv1d`/`norm` vs FullAttention
  gated QKV; dense-vs-MoE variants). Feeds P-E.
- **real 2-GPU HW validation** — run `llama_store_pp` on hiptrx (4× gfx1201) / hipx (gfx1151+gfx1010)
  with distinct devices (emulation aliases device 0); confirms `boundary_copy` peer path. Validates P-C.
- **`ModelParallel`/`ArchDispatch` daemon hoist (serve-reach)** — sits *above* the executor, carries
  forward unchanged from the old Phase 3. The god-struct refactor (`EpArch`/`LoadedModel`/`load_model_pp`
  guard `daemon.rs:4843`) → `ModelParallel{gpus, mesh, weights: WeightStore, state}` + `Box<dyn ArchDispatch>`.
  Only needed once an axis is serve-reachable (after P-C/P-D land a real forward).

## GOTCHAS (bit me this session)
- **GPU lock goes stale** (`/tmp/hipfire-gpu.lock`, noclobber variant). Verify dead (dead pid
  + no `/proc/*/fd` holder + idle `rocm-smi`) then `rm -f` — happened twice this session.
- **NEVER `cargo fmt`**; per-file `rustfmt --edition 2021 --config skip_children=true <file>`.
  NEVER rustfmt the fmt-debt files: `daemon.rs`, `qwen35.rs`, `deepseek4/minimax forward.rs`.
- **Multi-invocation bash caches builds** — the first `cargo build` in a block compiles, the
  rest show 0.02s. Capture output in one invocation to read real compile/error counts.
- **deepseek4 EP is slow to cold-JIT** (35B MoE) — use `ep_decode_parity` (fast tp=1 anchor)
  or a small model, not the full daemon EP, for quick byte-identity checks.
- Base is `feature/parallel-expansion`, so the mesh branch's diff-vs-master includes the
  emulation feature until that lands; rebase once it merges.

## Engineering decisions recorded (don't re-litigate)
- ~~The literal single-GPU+EP one-signature merge was NOT done~~ **— SUPERSEDED by the 2026-07-06 pivot.**
  The one-executor merge IS the plan again, but onto `execute_steps(mesh, gpus)` (the 63-site `Step` spine),
  NOT `run_layer_program`/`SuperOp`. bjoern locked: grand-unify all arches+axes into `Step`, big-bang flip.
- `CollectiveHint` is DERIVED from `ShardPolicy` (single source of truth), not hand-written. **(Still holds —
  now emitted per-`Step` in `execute_steps`, keyed by the manifest policy.)**
- Mesh is named-axis-primary; the `Dimension` tree is the raggedness (mixed-arch) extension. **(Unchanged.)**
