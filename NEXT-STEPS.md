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
   only on accepted positions, not tree branches). Must pass
   `scripts/coherence-gate-dflash.sh` (3-tier attractor thresholds). DFlash perf
   gates use q8 or FWHT KV — never asym.

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
