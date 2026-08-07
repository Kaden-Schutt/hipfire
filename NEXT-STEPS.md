# NEXT-STEPS — after Adaptive KV (branch `feat/kv-vquant-fwht-lloyd-v`)

Adaptive KV shipped 2026-05-31: runtime VRAM-fit downshift of **both** K and V
precision as context grows, re-quantizing the live cache in place along a
configurable pattern. Validated on gfx1100 (qwen3.6-27b.mq4) + fleet-hardened on
gfx1201. This file lists the deliberate follow-ups.

## Immediate follow-ups (own PRs)

1. **DFlash / spec-decode hook.** v1 wires only the linear `generate` decode
   path (design §2 non-goal: DFlash a fast-follow). Add `maybe_downshift` at the
   `generate_dflash` committed-position site (near the `ev.maybe_evict` calls,
   `daemon.rs:3375`/`3558`), handling spec-decode position semantics (commit
   only on accepted positions, not tree branches). Validate via
   `scripts/serve_harness.py` (`battery` + `chain`) per
   [docs/VALIDATION.md](docs/VALIDATION.md) and eyeball the decoded output for
   attractors — the old `scripts/coherence-gate-dflash.sh` is retired and absent.
   DFlash perf gates use q8 or FWHT KV — never asym.

2. **Default-on decision.** Adaptive is opt-in (`HIPFIRE_KV_ADAPTIVE=off`
   default). Because it runs the fast high-precision tiers until the cap,
   enabling it has **zero perf cost at short context** and only helps at long
   context — so default-on is a strong candidate. Gate the flip on: (a) a formal
   short-ctx perf A/B confirming adaptive-on == static within ±2%, (b) the
   DFlash hook landed, (c) a broad coherence sweep across the model zoo. Keep
   Q8-V the static default (capacity-not-speed).

3. **Multi-GPU.** `set_v_mode_realloc` / `set_adaptive_floor_alloc` are
   single-GPU. The pp>1 (tensor-parallel) load path currently ignores the
   adaptive override. Thread the controller + transcodes through the multi-GPU
   `Gpus` path.

## Refinements

4. **Pattern-tuning KLD sweep.** The `balanced` interleave
   `[V→l4, V→l3, K→f2, V→l2]` is the reasoned default (keeps K/V bit-gap ≤1,
   per the validated "balanced beats lopsided" matrix finding) and is
   coherence-clean. A dedicated adaptive-pattern KLD sweep over alternative
   interleaves at each equal-byte budget could shave KLD further; wire it into
   `benchmarks/quality-baselines/results/.../adaptive-pattern-sweep.txt`.

5. **`Aggressive` preset differentiation.** Currently `Aggressive` == `Balanced`
   (same floors fwht2/lloyd2). Differentiate by interleaving K earlier (reach a
   given capacity sooner at a small quality cost) once the pattern sweep (4)
   informs the tradeoff.

6. **Recency-tiered precision (research-grade).** Instead of a uniform tier per
   step, keep recent tokens at higher precision and old tokens at the floor (à
   la PyramidKV/H2O but precision-tiered, not eviction). Composes with the
   existing per-position transcode.

7. **fleet: full 27B coherence + KLD on gfx1151.** gfx1201 fleet-hardened. hipx
   (Strix Halo gfx1151) verification — see the fleet report; complete if its
   ROCm/build was flaky at ship time.

## Hygiene (small, unrelated)

8. **`mtp_mode` / `mtp_k` config-meta gap.** `cli/config_meta.test.ts` fails
   (pre-existing, not adaptive-KV): these two keys lack `meta` entries and would
   crash the config TUI if navigated to. Two-line fix in `cli/index.ts`.

9. **Revert 2E (carried over from the V-quant line).** Commit 373d0f59
   (per-tile→reduce-kernel lloyd-V inverse) was a null result (no perf gain at any
   ctx, +0.9% KLD FP-reassociation). It is still in the branch; the documented
   KLD matrix reflects the per-tile version. Revert was deferred here to avoid
   re-validating the adaptive coherence runs (all validated against the current
   kernel state). Revert + re-run the lloyd-V matrix as its own change.

## Validation status at ship (gfx1100, qwen3.6-27b.mq4)

- All four transcodes (V q8→lloyd4, V lloyd-down, K fwht4→fwht2, K fwht4→fwht3)
  proven transcode≈direct (max diff = one quant-boundary step) via the synthetic
  GPU harness `crates/hipfire-runtime/examples/adaptive_kv_check`.
- Presets (conservative/balanced/aggressive) + advanced (k=fwht3,v=lloyd2)
  coherence-validated end-to-end: downshifts fire at the controller's predicted
  positions and output stays fluent through every transition (incl. the
  attractor-prone K transitions; last-128 unique-token ratio ≥ 0.59,
  max-token-freq ≤ 0.07 at every checkpoint).
- **KLD continuity** is implied by the buffer-level transcode≈direct proof: an
  adaptive cache at tier X is byte-equivalent (±1 quant step) to a static-tier-X
  cache, so an adaptive run's KLD equals the static end-tier KLD from the 12-cell
  matrix (prior session).
- **Perf** is zero-overhead pre-threshold by construction (one integer compare
  per token below the cap); transcode is a one-time O(ctx) pass per step.
  A formal short-ctx A/B is folded into the default-on decision (2).

## EP serving — constructor-mid-failure VRAM leak (scoped follow-up)

The daemon's EP-shard load path (`load_model_ep` / `load_model_ep_minimax` in
`crates/hipfire-runtime/examples/daemon.rs`) builds each rank's weights+state
into a staging guard (`Ds4EpStaging` / `MinimaxEpStaging`) whose `Drop` frees
every COMPLETED rank on any early return. That handles a failure BETWEEN ranks
and the `HIPFIRE_EP_FAIL_RANK` fault (which fires AFTER a rank's constructor
returns `Ok`) — the primary completed-rank cleanup path, which is fixed and
fault-injection-tested.

RESIDUAL (not fixed): a failure INSIDE a single rank's constructor —
`DeepseekV4::load_weights_sharded` / `DeepseekV4State::new` /
`MiniMaxWeights::load` / `MiniMaxState::new_with_max_seq` — that occurs after it
has uploaded some-but-not-all of that rank's tensors and before it returns `Ok`
leaks those partial allocations. The half-built weights/state value is dropped
on the `?` early-return, and `GpuTensor` has no `Drop`, so its already-uploaded
device buffers are never returned to the pool. This is a partial-load-only leak
(a clean load leaks nothing); a subsequent successful load reuses the pool, so
the practical exposure is a repeatedly-failing big-model load.

Proper fix: an unwind-safe allocation-tracking refactor of those four loaders —
build each rank's tensors into a scratch list whose `Drop` frees every tensor on
any early return, then commit the list into the weights/state struct only on
full success (or give `GpuTensor` itself a pool-returning `Drop`, the broader
change). Deferred. Documented inline at the constructor call sites and on the
`load_model_ep` doc-comment.

## STEP-002R — Qwen35 Frozen construction rollback + BundleTeardown pivot (DONE)

Owner-preserving teardown for the Qwen35 Frozen construction path (STEP-002R)
and its generalization: `hipfire_runtime::gpu_cleanup` (`RetainedGpuTensor`,
`GpuCleanupFailure` with a `RetryableOwner` category, `BundleTeardown`,
`retain_free!`), checked `free_checked` on every arch bundle (qwen2, qwen35,
llama, lfm2moe, minimax, cohere2moe, deepseek4, dots-ocr), `unload_model`
dispatching through `ModelState::free_checked` with retry-before-log,
env-var-driven fault injection (`frozen-fault-inject` feature,
`HIPFIRE_FROZEN_FAIL_STAGE` / `HIPFIRE_FROZEN_FAIL_FREE`), GPU fault battery +
per-arch load/unload VRAM verification + packed-MQ4 expert teardown test.
Forensic discoveries fixed en route (see commit message): dspark sidecar pread
`RefCell` borrow bug; packed-expert interior-view pooling hazard in the checked
MoE free; VMM arena release skipped by `free_tensor_checked` (now arena-aware).

RESIDUAL DEBT (tracked, do not forget):

1. **Qwen35 mid-constructor leaks (same class as the EP constructor leak
   above).** A failure partway through `DeltaNetState::new_with_quant`,
   `Qwen35Scratch::new_with_kv_max`, `PrefillBatchScratch::new_opt`, the
   `KvCache` constructors, or mid-iteration `fulfill_manifest_gpu` leaks the
   already-uploaded tensors (the half-built value drops on `?`; `GpuTensor`
   has no `Drop`). STEP-002R covers POST-construction rollback only — the
   fault-injection stages fire after each constructor returns `Ok`. Proper
   fix: the same allocation-tracking refactor as the EP loaders (scratch-list
   staging or a pool-returning `Drop`). Deferred.

2. **Qwen35 PP path (`load_qwen35_pp`, carriers.rs).** Load errors propagate
   with plain `?`, leaking any already-built weights/kv/dn/scratch — same
   unchecked-construction class as STEP-002R fixed for the single-GPU bundle
   path. Owned by GEN-001/HW-004 (PP is device-mesh "Planned"). NOTE: PP
   UNLOAD is verified safe — `free_gpu_multi` routes through
   `free_moe_storage`, which handles packed experts — only the load-error
   rollback is unchecked. No PP fault injection or tests were added (per
   STEP-002R scope).
