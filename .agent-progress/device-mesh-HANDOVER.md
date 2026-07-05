# Device-mesh — HANDOVER for the next session

**Read this first, then set a *single-phase* goal** (not the whole 8-phase plan — that's a
multi-week, one-PR-per-phase effort by design; a session-scoped `/goal` on all 8 phases
loops forever). Suggested next goal: `/goal implement device-mesh Phase 2 fulfill_manifest`.

## Where things are
- **Branch:** `feature/device-mesh` (worktree `.claude/worktrees/feature+device-mesh`),
  off `feature/parallel-expansion` (which carries `HIPFIRE_EMULATE_GPUS` — the single-card
  multi-rank harness that Phase-1b/5a validation depends on). 26 commits, tree clean,
  workspace builds with 0 errors, all no-GPU tests green (0 failures).
- **Plan (updated with status):** `docs/superpowers/plans/2026-07-05-device-mesh-transparent-parallelism.md`
  — the top "IMPLEMENTATION STATUS" section is authoritative.
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

**Also DONE (Phase 3 consumption): store→forward.** `WeightStore::take` moves handles out;
`llama_store_load` now assembles a `LlamaWeights` whose projection `WeightTensor`s wrap the STORE
buffers and runs a REAL forward — **151936 finite logits, valid argmax, gfx1151**. Byte-identity +
consumption close the load→forward loop for the quantized projections.

## The NEXT unit — pick ONE (Phase 3 continuation):
**Option A (recommended): whole-model store load, then PP-2.** (1) Extend the llama `source` to cover
norms/embed/tied-lm_head — reproduce the F16→F32 host dequant (`llama::f16_to_f32` is public; return
`(f32_bytes, DType::F32)`; see `hfq.rs:551 load_f16_tensor` + the embed `EmbeddingFormat` branch,
`hfq.rs:792`), so the WHOLE model loads via the store (drop the projection-only filter in
`llama_store_load`). Then assemble the FULL `LlamaWeights` from the store (no bespoke fallback) and
assert a forward logit-matches bespoke. (2) Then `HIPFIRE_EMULATE_GPUS=2` PP-2 fulfill + banded
forward. This is the ModelParallel/ArchDispatch-hoist track.

**Option B: qwen35 `weight_manifest`** (finish Phase-2 arch coverage — DeltaNet loader study;
see `qwen35.rs:2876-2945` per-`LayerType` weight sets).

**Option B: qwen35 `weight_manifest`** (finish Phase-2 arch coverage). Pure-CPU like the ds4/qwen35
work just done, but needs DeltaNet loader study: per-`LayerType` weight sets (LinearAttention fused
`in_proj_qkv`/`in_proj_z`/`in_proj_a`/`in_proj_b` + `A_log`/`dt_bias`/`conv1d`/`norm` vs FullAttention
gated QKV), dense-vs-MoE variants. FusedQkv/HeadSharded policies for the DeltaNet head axis. Study
`qwen35.rs:2876-2945` (the per-layer-type loader) + `LayerType`.

**Option C: dense-TP slice (Phase 5 placement).** Greenfield quant-blob row-gather; only worth it
alongside the live-TP forward (Phase 5-dense).

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
- The literal single-GPU+EP one-signature merge was NOT done (N1-rejected "unified contract"
  shape; both executors mesh-aware in `hipfire-dispatch` is the unification). See status ledger.
- `CollectiveHint` is DERIVED from `ShardPolicy` (single source of truth), not hand-written.
- Mesh is named-axis-primary; the `Dimension` tree is the raggedness (mixed-arch) extension.
