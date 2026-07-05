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

## DONE this session: `fulfill_manifest` whole-tensor path (Phase 2 GPU exec)
`crates/hipfire-runtime/src/weight_store.rs` — `fulfill_manifest(weights, mesh, n_layers, gpus,
source) -> Result<WeightStore, FulfillError>`. GPU-validated on gfx1151 (`fulfill_manifest_probe`:
single-1×1 + emulated PP-2, placement + `memcpy_dtoh` byte-oracle; the oracle caught a real
missing-`layer`-in-key bug). Whole-tensor upload only; `ExpertSharded` + dense-TP slice return
`Err` (deferred). Takes a `source(entry)->bytes` closure (the arch owns on-disk HFQ naming; the
engine only does placement). Additive — forward path untouched (Tier-1; store-read is Phase 3).

## The NEXT unit — pick ONE (both build on `fulfill_manifest`):
**Option A (recommended — self-contained, byte-validatable like this session): `ExpertSharded`
upload.** Finish Phase-2 EP placement: host-pack owned experts + zeroed-dummy for non-owned,
reusing the deepseek4 convention (`crates/hipfire-arch-deepseek4/src/arch.rs:163-333`
`upload_layer_routed_experts` — compact blob via `gpu.upload_raw`, per-expert ptr table,
`gpu.zeros` dummy for `gate_up`, `ShardConfig::owns_expert`). The `source` closure yields the
experts tensor bytes; fulfill host-slices owned experts. Validate against `deepseek4
load_weights_sharded` bytes on a small MoE, or extend the probe with an ExpertSharded entry on Ep-2.

**Option B: wire `WeightStore` into a real load (Phase 3 start).** Give an arch (llama — has
`weight_manifest`/`state_manifest` but NO multi-loader) a `source` backed by its real HFQ + name
resolver (study `qwen35_tensor_data`, `qwen35.rs:1155`), fulfill on PP-2, forward reads the store
not arch fields. Higher risk (forward rewiring); pairs with the ModelParallel/ArchDispatch hoist.

Original PP validation target (for whenever the forward reads the store): `HIPFIRE_EMULATE_GPUS=2`
PP on qwen3.5-4b; compare to bespoke `load_model_pp`/`_ep` output (byte-identical).

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
