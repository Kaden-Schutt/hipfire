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

## The NEXT unit (recommended): `fulfill_manifest` (Phase 2 GPU execution)
`plan_manifest(weights, state, mesh, n_layers) -> ManifestPlan` already computes *where every
weight/state goes + the collective/band schedule*, validated. `fulfill_manifest` is now just
the **GPU execution** of that plan:
  1. `let plan = plan_manifest(&W::weight_manifest(cfg), &W::state_manifest(cfg), &mesh, n_layers)?;`
  2. for each `WeightPlacement`: read the HFQ tensor, **slice it per `ShardPolicy`**, upload
     the slice to each device in `placement.devices` (reuse the existing per-device upload
     primitive — study `qwen35 load_weights_multi` and `deepseek4 load_weights_sharded` +
     the zeroed-dummy for `ExpertSharded`).
  3. populate the arch's existing per-device weight/state fields (Tier-1: forward still reads
     arch fields — do NOT block on the Tier-2 slot-binding).
Validate: `HIPFIRE_EMULATE_GPUS=2` PP (`+HIPFIRE_PP=2` via the Bun CLI) on qwen3.5-4b + EP
default on a MoE; compare to the current bespoke `load_model_pp`/`_ep` output (byte-identical).

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
