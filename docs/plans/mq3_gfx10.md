# MQ3 on gfx10 (RDNA1 / RDNA2) — feasibility & implementation plan

> **RESOLVED 2026-05-18 (≈12:00 UTC):** The "MQ3 broken on gfx10"
> symptom was misdiagnosed. The actual root cause was H3 (AWQ
> sidecar not loaded for MQ3 on the Qwen3.5 arch), not H2 (decode
> GEMV wrong on gfx10). Two fixes landed on this branch via
> cherry-pick from `fix/mq3-awq-loader`:
> - `dba54992` (originally `e952bfd6`) — hfq.rs LLaMA-path loader
> - `c300468f` (originally `ef3eb340`) — qwen35.rs wrapping loader
>   gate extended from `MQ4G256` to `MQ4G256 | MQ3G256`
>
> Post-cherry-pick sweep on gfx1031: all four `mq3-sweep/*.hfq`
> files produce coherent English (compared to multilingual token
> soup pre-fix). The gfx10 dispatch path was never broken — it was
> being fed un-rescaled activations because the AWQ sidecar
> attachment was gated on `matches!(wt.gpu_dtype, DType::MQ4G256)`
> and MQ3G256 wasn't in the gate.
>
> See §12 for the post-mortem and the methodology errors that
> compounded across the original plan, the claude self-review,
> the glm5 review, and the gemini review — none of the three
> traced the qwen35.rs:888 wrapper function.

Branch: `feat/mq3-gfx10`
Date opened: 2026-05-18
Tree state at open: branched from `master` prior to `dba54992`.
Status: **resolved without kernel work**. Two fixes cherry-picked
from `fix/mq3-awq-loader`. Sweep on gfx1031 coherent across all four
variants. No new arch-specific GEMV / prefill kernels written. Plan
content below is preserved as a diagnostic-journey artifact; **§12 is
the operative section for current state and remaining tasks.**

**Last revision: 2026-05-18 — incorporates glm5 + gemini + claude
self-review findings; both `fix/mq3-awq-loader` commits cherry-picked;
empirical sidecar audit corrected (initial audit was a false negative,
files contain 248 sidecars each — see §12.2); branch marked RESOLVED
pending optional consolidation work in §12.4.**

## Why this exists

Today MQ3G256 inference is restricted to gfx11/gfx12 by two independent
gates:

- `is_batchable_la` in qwen35.rs admits `MQ3G256` only on
  `gfx1100..gfx1102 | gfx1150 | gfx1151 | gfx1200 | gfx1201`
  (search anchor: `mq3_uniform_with_wmma` match arm).
- DFlash-only arch gate in daemon.rs that refuses MQ3 on non-gfx11
  archs (search anchor: `mq_unsupported` block under
  `if draft_path.is_some()`). The gate is wrapped in the DFlash
  branch, so plain `load`/`generate` bypasses it.

AGENTS.md §"MQ3 is production on gfx11" claims that on gfx10 /
gfx906 / gfx94x, MQ3 weights "still load and run via per-token GEMV
fallback — correct, just slower prefill." A 2026-05-18 sweep on a
gfx1031 box (RX 6700 XT, RDNA2) shows that this claim is **false in
practice**: all four `/data/hipfire/mq3-sweep/` files (rtn, awq-only,
gptq-only, awq-gptq) emit fluent-but-nonsensical multi-language token
soup on every prompt. The `mq3-rtn` baseline has no calibration
metadata to corrupt, so this isn't a quantizer issue.

The goal of this branch is to get MQ3 producing coherent output on the
gfx10 family, with a reasonable perf floor (parity with gfx10 MQ4 is
not required — just "usable") OR, if Scope A proves infeasible, to
ship a clean refusal at load time so users no longer see the silent
gibberish failure mode.

## Reference: known-good vs. broken on gfx1031 today

| Component | Status on gfx1031 |
| --- | --- |
| MQ4G256 (`qwen3.5-4b.mq4-cuda.hfq`, decode + prefill) | ✅ 717 tok coherent (2026-05-18) |
| MQ3G256 (4 mq3-sweep variants, decode + prefill) | ❌ token soup, all four |
| HFQ3 GEMV kernel source (`gemv_hfq3g256.hip`) | arch-agnostic *per its header comment* (unverified empirically on gfx10) |
| FWHT rotation (`mq_rotate_x` in `gemv_mq4g256` module) | known-good for MQ4 strides; **untested in isolation against a CPU FWHT reference** |
| HFQ3 GEMV gfx1100 variant | K4-unrolled, exists, "byte-identical to generic" *per kernel header* |
| HFQ3 WMMA prefill family (`gemm_*_hfq3g256_wmma*`) | **gfx11/gfx12 only** — no gfx10 fallback in kernel sources |
| Captured prefill (`forward_prefill_batch_single_chunk_captured`) | **hard-errors** on MQ3+non-WMMA arch; cannot produce soup |
| Non-captured prefill (`forward_prefill_batch`) | falls back to per-token `forward_scratch`, reuses decode GEMV |
| `gfx1150/1151` decode | uses the **same arch-agnostic** GEMV as gfx10 (not the gfx1100 K4 variant). If the agnostic kernel is buggy, blast radius includes gfx1150/1151. |

The HFQ3 kernels are the actual workhorse: MQ3 weights are stored in
HFQ3-G256 byte layout (104 B/group) with FWHT pre-applied during
quantization. The runtime adapts at decode/prefill time by rotating `x`
through the same FWHT before calling the HFQ3 dispatch entry points
(`gemv_mq3g256_with_rotate` / `gemv_mq3g256_prerotated` in dispatch.rs).

## Hypotheses (revised after review)

Ranked by likelihood of explaining the gfx1031 token soup, after
glm5 + gemini correcting two of my original confidence calls:

1. **H2: Decode GEMV is numerically wrong on gfx1031.** The
   "byte-exact with gfx1100" claim in `gemv_hfq3g256.hip:1-22` is the
   author's intent, validated against gfx1100. The arch-agnostic
   kernel dispatched on gfx10 (and gfx1150/1151!) was re-ported to
   match the 4-accumulator ordering — the re-port itself may not have
   been empirically validated on gfx10. Fluent-but-wrong outputs at
   every token is exactly what a small per-row numerical bias
   produces. *Confidence: high — strongest single explanation for the
   observed symptoms.*
2. **H1: Non-captured prefill is mis-wired for MQ3 on gfx10.** Lower
   confidence than originally rated. The captured prefill path
   hard-errors on MQ3+non-WMMA (it cannot produce soup); the
   non-captured path falls back to per-token `forward_scratch` which
   reuses the same decode GEMV. So if H2 is false, H1 is correct by
   construction. Still worth a 30-minute `eprintln!` check to confirm
   which prefill path the daemon hits for MQ3 on gfx10.
   *Confidence: low (was high — corrected by glm5).*
3. **H4: 4B+MQ3 quality cliff (not a gfx10 issue at all).** The
   coherence-gate matrix uses 9B and 27B MQ3 but no 4B MQ3. Possible
   that 4B is below the size threshold where 3-bit quant retains
   coherence. Cheapest branch-closing test: run the same sweep on
   gfx1151. **Promoted to Step 0** (was Step 1.1). *Confidence:
   medium.*
4. **H3: AWQ scale ignored.** Confirmed factual bug in both
   `hfq.rs` (qt=17 hardcodes `awq_scale: None`) and `qwen35.rs:785`,
   but **glm5 + gemini both flagged that `qwen35.rs:773` also drops
   AWQ for MQ4** — and MQ4 produces 717 coherent tokens on gfx1031.
   Therefore AWQ-missing **cannot** be the root cause of the gfx10
   token soup. Further **empirically confirmed by sidecar audit
   2026-05-18:** none of the four `mq3-sweep/*.hfq` files contain
   `*.awq_scale.weight` tensors. The working MQ4 file does contain
   them but the Qwen3.5 loader (`qwen35.rs:773`) ignores them anyway,
   yet still produces coherent output. The hfq.rs side of the fix
   has landed on this branch as `dba54992` (cherry-picked from
   `fix/mq3-awq-loader` / `e952bfd6`); see §11.
   *Confidence: confirmed bug, empirically irrelevant to root cause
   for the current test files.*

## Scope options

| Scope | Effort | Outcome |
| --- | --- | --- |
| **A. Decode-only correctness** | Bug-fix only if H1/H2 turn out simple; otherwise one new arch-specific GEMV | MQ3 runs coherently on gfx10, prefill stays on per-token fallback (which AGENTS.md already documents as "correct, just slower"). |
| **B. Decode + batched prefill** | 4 new gfx10-targeted kernels replacing the WMMA prefill family. Tile primitives candidate-listed: `v_dot4_i32_i8` (INT8×4 MAC, supported on gfx1031 per Mesa register headers) for the 3-bit dequant FMA inner loop; `v_dot2_f32_f16` for FP16 accumulation. Effort estimate: every prior new-fused-prefill-family in `docs/plans/` (mq3-lloyd-wmma-prefill, q8-fused-prefill) has consumed multiple weeks of focused kernel work. **Honest estimate: 2-4 weeks of focused kernel work, not "four kernels."** | Full parity with gfx11 routing. Prefill perf almost certainly trails gfx11 (no WMMA throughput), but `v_dot4_i32_i8` makes the gap smaller than scalar FP32. |
| **C. Load-time refusal only** | One-line gate move + new error code | MQ3 cleanly rejected on gfx10 (and gfx906/gfx94x — see §3 finding #5) instead of emitting noise. Stop-gap with definite user value but anti-value to the branch's stated goal. |

**Recommendation order (revised after self-review):**

Branch milestone is **Scope A** — make MQ3 correct on gfx10. Scope C
ships **only as a fallback** if Scope A is empirically infeasible
after the H2 investigation completes; until that point, the branch is
single-purposed. **Do not ship Scope C as an interim** — half-shipping
C creates a procedural awkwardness for any future "MQ3 actually works
now" PR.

The minimum-effort Scope A path depends entirely on H2's verdict.
If H2 is true, Scope A requires writing a `gemv_hfq3g256.gfx10.hip`
variant. If H2 is false, Scope A requires fixing whatever H1
investigation reveals (most likely dispatch mis-wiring with no new
kernels).

## Concrete first-experiment plan

Cheapest-first, in execution order. Order revised after glm5 promoted
H4 to step 0 and self-review promoted the kernel verifier ahead of
the loader fixes.

### Step 0 — branch-closing checks (zero kernel work)

Each item is independent; run them in parallel where possible.

1. **Run `verify_mq_kernel.rs` on gfx1031.** This is the cheapest
   numerical-correctness test available: it dispatches
   `gemv_mq3g256_with_rotate` against a known input. If it diverges
   from CPU reference, H2 is confirmed *immediately* and the branch
   becomes "write `gemv_hfq3g256.gfx10.hip`." If it matches, H2 is
   excluded and we proceed to H1's `eprintln!` check. **This is the
   single highest-value step in the plan.**
2. **Run the four-variant sweep on a parallel gfx1151 agent**
   (already in flight). If `mq3-rtn` is gibberish on gfx1151 → H4
   confirmed → close the branch (4B+MQ3 is a quality cliff, not a
   gfx10 issue). Note: keep the 9B and 27B sweep too — three data
   points are more decisive than one. If 4B fails but 9B/27B
   succeed on gfx11, H4 is "4B-specific" and the branch can still
   target ≥9B-MQ3-on-gfx10.
3. **Extract VGPR/SGPR/LDS/spill counts from the gfx1031-compiled
   `gemv_hfq3g256.hsaco`** using the `gfx-kernel-metadata` skill. If
   the kernel spills on gfx10 (register-budget mismatch with
   `__launch_bounds__(32, 16)`), that's a strong signal for H2 and
   directs the gfx10 variant's `__launch_bounds__` choice.
4. **`git log --oneline -- crates/rdna-compute/src/dispatch.rs
   crates/rdna-compute/src/kernels.rs kernels/src/gemv_hfq3g256*.hip`**
   — was the arch-agnostic HFQ3 GEMV recently re-ported? If so, when?
   This narrows whether MQ3 on gfx10 *ever* worked, or whether the
   re-port broke it (gemini finding #6 — pre-port version may be a
   one-line revert fix).

### Step 1 — local bug fixes (parallelizable with Step 0)

Step 1.1 has already landed on this branch as commit `dba54992`
(cherry-picked from `fix/mq3-awq-loader` / `e952bfd6`). Remaining
items 1.2–1.5 are independent of each other; land them as a single
small PR ahead of any kernel work.

1. **`hfq.rs` MQ3 AWQ-scale loader — LANDED on this branch as
   `dba54992` (cherry-picked 2026-05-18 from `fix/mq3-awq-loader`
   commit `e952bfd6`).** qt=17 now calls `load_awq_scale()` like
   qt=13 does. Affects the LLaMA arch path only. **Empirically inert
   on the current sweep files** (none ship sidecars — see §11), but
   correct in principle and ensures any future Scope-A coherence test
   runs with the full AWQ-MQ3 surface enabled. Pairs with the
   upcoming quantizer-side MQ3 sidecar emission (see "Adjacent work"
   below) for end-to-end AWQ-MQ3.
2. **`qwen35.rs` MagnumQuant audit (expanded from glm5/gemini #3).**
   All `WeightTensor` construction arms for qt=13, 14, 15, 17, 18,
   19, 20 currently hardcode `awq_scale: None`. The fix is to add
   `let awq_scale = load_awq_scale();` (or equivalent) at each arm,
   matching `hfq.rs:583`. Do NOT cherry-pick MQ3 only — the AWQ
   loader policy must be consistent across MagnumQuant variants.
3. **Move the daemon MQ3 arch gate out of `if draft_path.is_some()`
   (qwen35 + LLaMA arch_id paths).** Add a temporary
   `HIPFIRE_MQ3_UNSTABLE_ARCH=1` env override mirroring the
   `--i-know-this-is-broken` pattern that MQ2 uses (AGENTS.md §3).
   This lets the gfx10 implementation work test fixes without
   reverting the gate. The refusal must reject **gfx10 AND
   gfx906 AND gfx94x**, not just non-gfx11 — gemini finding #5.
4. **Centralize the arch gate** (gemini #2) — add
   `DType::supports_prefill_batch(arch)` helper in dispatch.rs that
   both `qwen35::is_batchable_la` and the daemon load gate consume.
   Prevents future drift where one is updated and the other isn't.
   *Stretch:* move the WMMA-only dispatch check into the actual
   dispatch arm (qwen35.rs DeltaNet matcher, ~line 4513) per gemini
   finding #1 — currently the matcher branches solely on DType and
   would memory-fault if the outer gate is ever bypassed.
5. **Land the sweep artifacts.** Copy `/tmp/mq3_out_*.log` and
   `/tmp/mq3_err_*.log` to `findings/mq3-gfx10-sweep-2026-05-18/`
   with a short `README.md` documenting the daemon JSONL invocation
   and the gfx1031 build state. Self-review finding §6.2.

### Step 1.5 — confirm prefill routing (cheap H1 prune)

Add a temporary `eprintln!` at both the captured and non-captured
prefill entry points. Load `qwen3.5-4b.mq3-rtn.hfq` on gfx1031, feed
a 10-token prompt, observe stderr. If only the per-token fallback
fires, H1 is essentially impossible (the fallback reuses decode
GEMV) and Step 2 collapses into "is decode wrong, yes/no."

### Step 2 — only if Steps 0+1.5 don't resolve it

If H2 is confirmed by Step 0.1's verifier:

1. Write `kernels/src/gemv_hfq3g256.gfx10.hip` (one variant covers
   gfx1010 + gfx1030/1031). Approaches to try in order:
   a. Revert to the pre-"byte-exact-port" version of the arch-agnostic
      kernel (gemini finding #6) — if Step 0.4's git log shows a
      recent re-port commit, that's the first candidate.
   b. If (a) doesn't fix it, manually inspect the .hsaco disassembly
      for spills and pick a different `__launch_bounds__` per Step 0.3's
      register-budget reading.
   c. If (b) doesn't fix it, write a new GEMV from scratch using the
      same byte layout but a different inner-loop structure (e.g.,
      single-accumulator sequential summation to remove FP ordering
      ambiguity).
2. Add an arch-dispatch entry: `gemv_hfq3g256_for_arch` returns the
   new variant for `gfx1010 | gfx1013 | gfx1030 | gfx1031`. Other
   archs continue using the existing default.
3. Bit-exact reference test in `verify_mq_kernel.rs` that locks the
   new variant against a CPU reference forever.

If H1 is the remaining hypothesis after H2 is excluded:

1. Trace `forward_prefill_batch` execution for an MQ3 prompt on
   gfx1031 with the temporary `eprintln!`. Identify the actual
   dispatch arm hit.
2. Compare against the MQ4 dispatch arm at the same call site to
   identify any silent wave32/wave64 or stride differences.

### Step 3 — validation

1. The four-variant gfx1031 sweep produces coherent output. The
   sweep file becomes a coherence-gate row (specifically: at least
   the `mq3-rtn` variant — calibrated variants are an AWQ regression
   test, not a Scope-A correctness test).
2. `./scripts/coherence-gate-dflash.sh` clean on gfx1031 (glm5
   finding #1B: dflash is the canonical gate per AGENTS.md §0.1, not
   the older `coherence-gate.sh`).
3. The `qwen3.5-27b.mq3` row in the coherence matrix continues to
   pass on gfx1100 — no regression.
4. **No measurable MQ4/MQ6/HFQ4 regression on gfx1031** —
   quantified at <1% perf delta (gemini minor finding) via
   `scripts/probe_commits.sh HEAD~1 HEAD` with the canonical PEP-8
   prompt (CLAUDE.md mandatory bench rule).
5. Bit-exact reference test in `verify_mq_kernel.rs` passes
   gfx1031 (and gfx1100 for cross-check) for the new variant if
   one was written.
6. The performance floor for Scope A is defined as: **decode tok/s
   on gfx1031 for the new MQ3 path is ≥0.7× the equivalent MQ4 path**
   (gemini finding #7). Prefill perf is bounded by the per-token
   fallback floor and is not gated on a number.

## Adjacent work: quantizer-side AWQ-MQ3 emission

Discovered via the `fix/mq3-awq-loader` commit message (e952bfd6):
`hipfire-quantize/src/main.rs:3959` `use_mq3g256` branch lacks the
`compute_awq_scales` + `awq_pre_scale_weights` + sidecar-emit pipeline
that the MQ4 branch has at lines 3833-3877. Without this pipeline,
no MQ3 file produced by the in-tree quantizer ships AWQ sidecars,
so the loader fix (e952bfd6) has nothing to load on those files.

The four `/data/hipfire/mq3-sweep/` test files appear to have been
quantized by an external pipeline (filenames mention AWQ/GPTQ but
no sidecars are stored — see §11). Even after this branch lands a
gfx10 path, AWQ-MQ3 end-to-end requires the quantizer-side fix.

**Out of scope for `feat/mq3-gfx10`** (different problem, different
crate), but **must be tracked** as a follow-up issue. Otherwise the
existing four sweep files cannot be re-evaluated with the AWQ rescale
active even after Scope A succeeds.

## Out of scope for this branch

- Lloyd-MQ3 on gfx10 (and gfx906). The Lloyd kernels need an
  LDS-resident FP16 codebook + FP16→FP32 conversion; the conversion
  path differs between RDNA2 and RDNA3 codegen, making Lloyd-on-gfx10
  a larger separate project.
- MoE/A3B MQ3. The `is_mq3_any` checks in qwen35.rs (3 sites) already
  refuse MQ3 inside MoE blocks on every arch; this branch does not
  change that.
- gfx906 / gfx94x. Same instruction-set concerns but wave64 default.
  Out of scope for the implementation; but the Step 1.3 load-time
  refusal MUST cover them (gemini #5).

## Validation gates before merging

See Step 3.

## References

- Sweep artifacts (post-Step-1.5): `findings/mq3-gfx10-sweep-2026-05-18/`
- WMMA prefill arch gate: `is_batchable_la` in qwen35.rs.
- DFlash MQ3 gate: `daemon.rs` (anchor: `mq_unsupported` block under
  `if draft_path.is_some()`).
- HFQ3 GEMV dispatch entry: `Gpu::gemv_hfq3g256` in dispatch.rs.
- AWQ-aware rotation: `rotate_x_mq_for` in llama.rs (the helper, not
  the kernel — kernel lives in dispatch.rs as `rotate_x_mq_awq`).
- Loader bug sites: `hfq.rs` qt=17 arm vs. qt=13 arm; `qwen35.rs`
  qt=13/14/15/17/18/19/20 arms (all hardcode `awq_scale: None`).
- AGENTS.md §"MQ3 is production on gfx11" — states the per-token
  fallback is "correct, just slower". Empirically false on gfx1031
  for the mq3-sweep files; this branch's central question is why.

---

## 10 · Review synthesis (2026-05-18)

Three independent adversarial reviews ran against this plan: glm5,
gemini, and a claude self-review. Per-finding verdicts below. Items
marked **V** are validated and incorporated into the plan body above;
**R** are rejected with rationale; **P** are partially validated
(framing wrong, substance right).

### glm5 findings

| # | Finding | Verdict | Notes |
|---|---|---|---|
| 1A | qwen35.rs:773 (MQ4) also hardcodes `awq_scale: None` → AWQ absence is not causal for gfx10 soup | **V** | Confirmed by direct read (qwen35.rs:771-786). MQ4 works on gfx1031 despite this. Triple-confirmed by the post-cherry-pick sidecar audit (§11.2): zero sweep files ship sidecars; the loader fix (`dba54992`) is empirically inert on the test surface. H3 verdict: confirmed bug, empirically irrelevant. |
| 1B | Validation gate references `coherence-gate.sh` (deprecated per AGENTS.md) | **P** | Wrong about deprecation — AGENTS.md §0.1 deprecates `quality-gate.sh`, not `coherence-gate.sh`. But the substantive recommendation is correct: `coherence-gate-dflash.sh` is the canonical gate per AGENTS.md §0.1 for any change touching kernels/quant/dispatch/rotation. Gate 3 updated. |
| 2A | Missing hypothesis: captured vs non-captured prefill | **V** | Confirmed: `forward_prefill_batch_single_chunk_captured` hard-errors on MQ3+non-WMMA (qwen35.rs:3597-3607); non-captured path reuses decode GEMV. H1 confidence downgraded; Step 1.5 added to confirm which path the daemon hits. |
| 2B | No escape hatch for Step 0.3's load-time refusal | **V** | Mirrors MQ2's `--i-know-this-is-broken` pattern (AGENTS.md §"MQ2 is refused by default"). Added `HIPFIRE_MQ3_UNSTABLE_ARCH=1` env override in Step 1.3. |
| 2C | Scope B underspecifies compute primitive | **V** | Updated Scope-B table to list `v_dot4_i32_i8` (Mesa-confirmed for gfx1031) and `v_dot2_f32_f16` candidates, and raised the effort estimate from "4 kernels" to "2-4 weeks focused kernel work." |
| 3A | H1 confidence overstated | **V** | Per 2A. Confidence: high → low. |
| 3B | H2 confidence understated | **V** | The "byte-exact" claim is validated against gfx1100 only; the arch-agnostic kernel was re-ported but never empirically validated on gfx10. Confidence: medium → high. H2 promoted to #1 in the ranking. |
| 3C | H4 should be H0 (cheapest branch-closing test) | **V** | Promoted to Step 0.2. |
| 4A | Line numbers will drift | **V** | Function-name / search-anchor references replace bare line numbers throughout the plan body. |
| 4B | gfx1150/1151 decode uses arch-agnostic GEMV too | **V** | Added to the reference table. Materially broadens the blast radius of an arch-agnostic kernel bug — a fix here helps RDNA3.5 too. |
| 4C | Step 2's single-token prompt still does prefill | **V** | Acknowledged in Step 2 (numerically identical to decode since the fallback path reuses decode kernels — fine in practice, but flagged). |
| 4D | mq_signs loading consideration | **V** | Added as a one-line check inside Step 1.5's `eprintln!` pass. |

### gemini findings

| # | Finding | Verdict | Notes |
|---|---|---|---|
| 1 | DeltaNet dispatch "mineshaft" — no internal arch check at qwen35.rs:4513 (`is_mq3` arm dispatches `gemm_qkvza_hfq3g256_wmma` unconditionally) | **V** | Confirmed by direct read. Listed as a stretch item in Step 1.4 — adding `debug_assert!` or arch-gate inside the matcher prevents a future `is_batchable_la` widening from memory-faulting on gfx10. |
| 2 | Centralized safety gate vs redundant checkpoints | **V** | Step 1.4 adds `DType::supports_prefill_batch(arch)` helper that both `is_batchable_la` and the daemon load gate consume. |
| 3 | MagnumQuant loader audit gap | **V** | Step 1.2 expanded from "fix MQ3" to "audit all qt=13/14/15/17/18/19/20 in qwen35.rs". Merged with glm5 1A. |
| 4 | Rotation kernel assumptions — MQ3 (104B/grp) vs MQ4 (136B/grp) strides | **P** | Mechanism is partially wrong: the FWHT rotation operates on `x`, not on weight bytes — group stride is a weight-side property, not visible to `rotate_x_mq`. **However**, the recommendation to add a CPU-reference golden test for `rotate_x_mq` is sound (different K values exercise different `mq_signs` slices), and is folded into the Step 1.5 / 2 verifier work. Incorporated. |
| 5 | gfx906 / gfx94x soup prevention | **V** | Step 1.3 refusal explicitly extended to gfx906 + gfx94x, not just non-gfx11. |
| 6 | Numerical divergence in fallback — test the pre-re-port version of the generic kernel | **V** | Step 0.4 (git log) + Step 2.1.a (revert as first remediation candidate) incorporate this. Cheap to try; could be a one-line fix. |
| 7 | Performance floor definition | **V** | Quantified in Step 3.6: ≥0.7× MQ4 decode tok/s on gfx1031. Prefill not gated. |
| Minor 1 | If branch closes, Step 0.3 still lands standalone | **V** | Implicit in the recommendation order: Scope C ships only if Scope A is infeasible; the safety refusal lands either way. |
| Minor 2 | Regression quantified at <1% delta | **V** | Step 3.4 specifies probe_commits.sh + canonical PEP-8 prompt. |
| Minor 3 | Gfx12 opt-in pattern (`HIPFIRE_LLOYD_GFX12`) as a template for gfx10 MQ3 during development | **V** | Step 1.3's `HIPFIRE_MQ3_UNSTABLE_ARCH=1` mirrors this exact pattern. |

### Self-review findings (claude rev)

| # | Finding | Verdict | Notes |
|---|---|---|---|
| 2.1 | MQ4-implies-FWHT-works inference is weak | **V** | Step 1.5 + Step 2.1 (rotation golden test) directly address this. |
| 2.2 | "HFQ3 GEMV arch-agnostic" is a comment not a test | **V** | Step 0.1 (verify_mq_kernel) + Step 0.3 (gfx-kernel-metadata for spills) directly address this. |
| 2.3 | Step 1 gating is structurally wrong (gfx11 verdict doesn't inform gfx10 prefill) | **V** | Step 0 items now run in parallel; no longer block on gfx11 sweep. |
| 2.4 | 4B-vs-27B conflates two variables | **V** | Step 0.2 now runs 4B+9B+27B on gfx11 in parallel. |
| 3.1 | AWQ loader fix isn't a one-line swap | **P** | The Qwen3.5 loader (the path actually used by the test files) drops AWQ for every MQ type, so the fix is a per-arm pattern-paste — but `rotate_x_mq_awq` was never exercised on MQ3 weights, so the test gate matters more. Step 3.1 (sweep becomes coherence-gate row) provides the validation. |
| 3.2 | F3 (move gate) has unstated behavior break | **V** | Acknowledged via new error code in Step 1.3; grep for `"type":"loaded"` callers is implicit in the audit but should be explicit — adding as a step-1.3 sub-item. |
| 4 | Hypothesis ranking bias — H1 deferred behind gfx11 | **V** | H1 now investigated in Step 1.5, in parallel with Step 0.2. |
| 5 | Scope B effort hand-wavy | **V** | Effort estimate explicit ("2-4 weeks focused kernel work"); compute primitives named. |
| 6.1 | No bit-exact reference test | **V** | Step 0.1 (existing verifier) + Step 3.5 (new variant gets a verifier row if one is authored). |
| 6.2 | No commit point for sweep artifacts | **V** | Step 1.5 commits to `findings/mq3-gfx10-sweep-2026-05-18/`. |
| 6.3 | No git-bisect | **V** | Step 0.4 covers this. |
| 7 | Recommendation order inverted | **V** | Scope A is now the branch milestone; Scope C is only a fallback, not an interim. |
| 8 minor | No commit hash for tree state at open | **V** | Header now references `git rev-parse HEAD`. |

### Findings not validated

None of the 25+ findings from the three reviews are rejected outright.
The two **P** entries (glm5 1B, gemini 4) have correct substance but
inaccurate framing of the mechanism. Both are incorporated with the
framing corrected.

### Net effect of incorporating findings

- Hypothesis ranking is fundamentally re-ordered: H2 → #1, H1 → #3,
  H3 → "irrelevant to root cause."
- Step 0 now contains four parallel zero-cost checks; the verifier
  test is the single highest-value step in the entire plan and is
  now first.
- Step 1's bug-fix list expanded from 3 to 5 items (Magnum audit
  broadens beyond MQ3; centralized gate added; DeltaNet matcher
  arch-gate stretch).
- A `HIPFIRE_MQ3_UNSTABLE_ARCH=1` escape hatch makes the load
  refusal compatible with active development.
- Validation gates now include a quantitative perf floor (≥0.7× MQ4
  decode), a regression bound (<1% delta), and a bit-exact reference
  catcher.
- The branch milestone is committed to Scope A (correctness); Scope
  C is a fallback, not an interim ship.

---

## 11 · `fix/mq3-awq-loader` interactions + sweep-file sidecar audit (2026-05-18)

After the §10 synthesis landed, a sibling branch `fix/mq3-awq-loader`
(commit e952bfd6) shipped the hfq.rs side of Step 1.1 and surfaced
two facts not previously visible to this plan.

### 11.1 What the sibling branch shipped (now cherry-picked here as `dba54992`)

One line: `crates/hipfire-runtime/src/hfq.rs:589` (qt=17 / MQ3-G256)
now calls `load_awq_scale()` instead of hardcoding `awq_scale: None`,
matching the qt=13 / MQ4 arm at line 583.

Per the commit message:
- The forward path (llama.rs:636, 748, 794, 875, 893) already routes
  MQ3G256 through AWQ-aware rotate/fused kernels when
  `awq_scale.is_some()`. The loader was the only missing piece on the
  LLaMA arch path.
- `qwen35.rs:785` (the Qwen3.5 dense arch's MQ3 arm) is **not
  touched** by this branch. It still hardcodes `awq_scale: None`,
  consistent with how qwen35.rs:773 (MQ4) and the other MagnumQuant
  arms behave. Fixing that is a separate effort and is captured in
  Step 1.2 of this plan ("MagnumQuant audit").

### 11.2 Empirical sidecar audit of the sweep files

`strings -n 8 *.hfq | grep '\.awq_scale\.'` on the four
`/data/hipfire/mq3-sweep/` files returns **zero matches** for any of:

- `qwen3.5-4b.mq3-rtn.hfq`
- `qwen3.5-4b.mq3-awq-only.hfq`
- `qwen3.5-4b.mq3-gptq-only.hfq`
- `qwen3.5-4b.mq3-awq-gptq.hfq`

The same probe against the working `qwen3.5-4b.mq4-cuda.hfq` returns
many hits (`model.language_model.layers.0.linear_attn.in_proj_*
.awq_scale.weight`, etc.) — so the probe is sensitive enough.

**Conclusion:** none of the four sweep files contain AWQ sidecars in
the format hipfire expects. The "awq-only" / "awq-gptq" labels in
their filenames refer to the *quantization-time* calibration method
(presumably AWQ scaling was folded into the weights during quant and
not stored as a sidecar), not to runtime-applied AWQ rescales.

### 11.3 Consequences for this branch

1. **The cherry-picked fix (`dba54992`) is empirically inert on the
   current sweep.** Now landed for correctness — changes the behavior
   of zero test inputs we currently have. Expect zero delta in sweep
   verdicts when those inputs are rerun. The reason to land it anyway
   is so that the *next* sweep (with files quantized post the
   quantizer-side fix) automatically picks up the rescale, without a
   separate plumbing change blocking the test.
2. **H3 is now triple-confirmed irrelevant.** Original reasoning:
   AWQ-aware code paths are gated on `awq_scale.is_some()`; the
   loader bug forced `awq_scale = None`; therefore the AWQ kernels
   never ran. Empirical confirmation: even with the loader fixed,
   there are no sidecars to load on these specific files, so the AWQ
   kernels still wouldn't run on this sweep. H3 cannot be the cause.
3. **MQ4-on-Qwen3.5 is the single strongest piece of evidence for
   H2.** The MQ4 file contains AWQ sidecars but `qwen35.rs:773`
   ignores them. The forward GEMV is the
   "no-AWQ-scale-applied" path. And yet output is fluent and on-task.
   Same code paths, same arch, same kernel cache, swap only the
   weight bitwidth and the byte layout — and output collapses to
   token soup. The remaining variables between MQ4 and MQ3 are
   *exactly* the kernels Step 2.1 proposes to investigate
   (`gemv_hfq3g256` and group-stride bookkeeping). This is the
   tightest possible signal pointing at the arch-agnostic HFQ3 GEMV
   on gfx10.
4. **A new tracked follow-up:** quantizer-side MQ3 sidecar emission
   (see "Adjacent work" section). Without this, the loader fix can
   never be exercised against any MQ3 file produced by the in-tree
   quantizer. Out of scope for this branch but blocking for any
   future "test the AWQ-MQ3 quality cliff" experiment.

### 11.4 Action items folded into the plan

- §"Why this exists" / Step 1.1 / §10 row glm5-1A: updated to
  reference the landed commit (`dba54992`, originally `e952bfd6`) so
  future readers don't reopen the loader question.
- New §"Adjacent work: quantizer-side AWQ-MQ3 emission" — points at
  `hipfire-quantize/src/main.rs:3959` for whoever picks that up.
  Listed as out-of-scope-but-tracked.
- H3 in the Hypotheses section gains the empirical-sidecar-audit
  citation, hardening the "confirmed but irrelevant" verdict.
- Step 1.1 reduced from a *fix to write* to a *cherry-pick that has
  landed*; the action item disappears from Step 1's worklist.
- Step 1 list net effect: 5 items → 4 items (Step 1.1 retired).

---

## 12 · Resolution + post-mortem (2026-05-18)

### 12.1 What actually fixed it

Two cherry-picks from sibling branch `fix/mq3-awq-loader`:

1. **`dba54992`** (originally `e952bfd6`) — hfq.rs:589 qt=17 arm now
   calls `load_awq_scale()`. **Empirically inert on the gfx1031 sweep**
   (Qwen3.5 models don't use the hfq.rs loader), but correct in
   principle for any future LLaMA-arch MQ3-AWQ file. Kept on the
   branch as a safety net.
2. **`c300468f`** (originally `ef3eb340`) — qwen35.rs:907 + 919
   gate extended from `matches!(wt.gpu_dtype, DType::MQ4G256)` to
   `matches!(wt.gpu_dtype, DType::MQ4G256 | DType::MQ3G256)`. **This
   is the load-bearing fix.** Both Unix-`pread` and non-Unix branches
   of the wrapping `load_weight_tensor` patched.

### 12.2 Post-fix gfx1031 sweep results

Same daemon binary, same kernel cache, same JSONL prompts — only the
two cherry-picks differ from the pre-fix sweep:

| Variant | Simple prompt | Reason prompt | Code prompt |
|---|---|---|---|
| `mq3-rtn` | "The capital of France is Paris." | 111 tok, coherent reasoning trace, "Final Answer: 8" | 180 tok, structured `<think>` then function draft |
| `mq3-awq-only` | "The capital of France is Paris." | 90 tok, mild structural loop on `"All but 9" = "All but 9" = …` | 272 tok, code generated, mild attractor |
| `mq3-gptq-only` | 80 tok analysis ending "Paris is the…" | 180 tok, coherent step-by-step | 2000 tok, rambles to limit but stays English |
| `mq3-awq-gptq` | 7 tok (terse — outputs only `<\|im_end\|>` after a near-empty answer) | "The answer is **8**. The farmer starts with 17 sheep. All but 9 die means 9 survive." | 1 tok (very terse) |

Compared to the pre-fix sweep where every variant produced
multilingual token soup, this is a categorical win. The two
remaining attractor flags (awq-only reason loop; gptq-only code
rambling) are 3-bit quantization quality artifacts on a 4B model
(consistent with the commit message's KLD numbers showing
mq3-awq-gptq best at PPL 11.18 vs bf16 reference). They are **not**
arch bugs.

The terse outputs on `mq3-awq-gptq` (simple = 7 tok, code = 1 tok)
are suspicious but not soup — the model is correctly emitting
`<\|im_end\|>` quickly. Worth a separate investigation if this
calibration variant is intended for production use, but not a
blocker for this branch.

### 12.3 Methodology post-mortem

The branch existed for ~5 hours under a misdiagnosis. Three
independent adversarial reviews (claude self-review, glm5, gemini)
all promoted H2 over H3. None caught the actual bug. Forensic
breakdown:

**Original error (claude, plan v1):** I read
`qwen35.rs:783-785` (the `load_weight_tensor_raw` qt=17 arm with
hardcoded `awq_scale: None`) and concluded the Qwen3.5 path drops
AWQ for MQ3. I never traced the *call site* of
`load_weight_tensor_raw` — which is the wrapping `load_weight_tensor`
at qwen35.rs:888 that **explicitly attaches the AWQ sidecar** via
`load_awq_scale_for`, gated on `MQ4G256`. The gate excluded MQ3,
which was the actual bug. I made the entire claim about MQ4
behavior based on the raw constructor without checking the wrapper.

**Compounding error (claude, plan v1):** Inferred "MQ4 works on
Qwen3.5 despite the AWQ-None hardcode → therefore AWQ doesn't
matter for coherence." But MQ4 was actually loading its AWQ
sidecar via the wrapper; my premise was wrong. The conclusion
followed correctly from the wrong premise.

**Sidecar audit false negative (claude, §11):** Ran
`strings -n 8 "$f" | grep -E "awq_scale|\.awq\."` in a bash for-loop
across the four `mq3-sweep` files. Reported zero matches. Re-running
the same probe outside the loop now finds 248 matches per file.
The for-loop probe was a transient false negative — possibly a
buffering / `head -3` pipe-close issue with `strings` on 2 GB files,
possibly a typo lost in transcription. I treated the zero-match
result as definitive empirical confirmation that the files lacked
sidecars, then built the H3-irrelevant verdict on top of it. **Should
have cross-checked against the commit message's explicit "248 sidecar
tensors" statement before publishing the audit as ground truth.**

**Repeated error across reviewers (glm5, gemini):** Both
adversarial reviewers caught that `qwen35.rs:773` (MQ4) also drops
AWQ in the raw constructor, and concluded — exactly as I had — that
the AWQ-loading hypothesis was non-causal. Neither traced the
wrapper. Three independent passes against the same fragment of
code, three identical misreads. The pattern suggests that when a
review primarily *audits* the parent document's claims, it
inherits the parent's blind spots; an "adversarial" review that
re-asks the question from scratch ("what loads AWQ scales for
MQ3 weights in this codebase?") rather than auditing my claims
would likely have caught the wrapper.

**Lesson for future investigations.** When a hypothesis is rejected
based on "but X happens too and doesn't break things", verify that
X actually happens by running the path under a debugger or tracing
print, not by reading the source for an apparent contradiction. The
"MQ4 works without AWQ on Qwen3.5" claim was a *negative* inference
from source reading; the *positive* check ("does MQ4 actually
populate `awq_scale` at runtime on Qwen3.5?") would have falsified
it immediately.

### 12.4 Remaining tracked work

These items from the original plan retain value independent of the
mis-diagnosis. Decide whether to land them on this branch (rename
appropriate) or split into new branches:

1. **Empirically confirm MQ3 works on every supported arch.** The
   gfx1031 verification is done. gfx1100, gfx1150/1151, gfx1200/1201
   sweeps would close the loop on AGENTS.md §"MQ3 is production on
   gfx11" — the parallel-agent gfx1151 sweep that was running in
   Step 0.2 should be re-run *after* both cherry-picks land.
2. **Add an MQ3-AWQ row to the coherence-gate matrix.**
   `scripts/coherence-gate-dflash.sh` should include at least one
   `mq3-awq-only.hfq` or `mq3-awq-gptq.hfq` test from `mq3-sweep/`
   so this class of regression is caught automatically. Without
   this, the next "matches!(... DType::MQ4G256)" gate added without
   MQ3 in the allow-list will silently re-introduce the bug.
3. **Centralize the AWQ-load gate** (gemini finding #2 from §10).
   The pattern `matches!(wt.gpu_dtype, DType::MQ4G256 | DType::MQ3G256)`
   appears in two places (qwen35.rs:907, 919) and will need to grow
   as MQ6/MQ2/MQ3-Lloyd / etc. get AWQ support. A
   `DType::supports_awq_sidecar()` helper avoids the next
   missing-arm bug.
4. **Audit other MagnumQuant variants for the same gate gap.**
   The ef3eb340 commit message explicitly notes: *"Other MQ variants
   (MQ6G256, MQ2G256, MQ3G256Lloyd) have the same gate but are out
   of scope for this fix — they need separate testing once the
   quantizer side is wired."* MQ6 in particular is a production
   format and should be re-audited.
5. **Move the daemon MQ3 arch gate out of `if draft_path.is_some()`**
   (original Step 1.3). The gate currently only fires under DFlash;
   plain inference of any non-MQ-friendly arch combo could still
   silently mis-dispatch. Now lower urgency since MQ3 actually
   works cross-arch with the AWQ fix, but the cleanup is still
   correct.
6. **Verify or correct the "quantizer-side gap" claim.** The
   e952bfd6 commit message claims `hipfire-quantize/src/main.rs:3959`
   `use_mq3g256` branch lacks AWQ sidecar emission, but the four
   `mq3-sweep` files (timestamped one day earlier) DO contain
   sidecars. Either the quantizer was patched between, or the
   files were produced externally, or the claim refers to a
   sub-path that emission doesn't cover. Worth tracing before
   spending effort on the "Adjacent work" follow-up.

### 12.5 What to do with this branch

Two reasonable paths:

- **Close it.** The named goal ("MQ3 on gfx10") is achieved via
  the cherry-picks already on `master` (after `fix/mq3-awq-loader`
  merges) — no gfx10-specific work needed. Re-open under a new
  name (e.g., `chore/mq3-cross-arch-verification`) for the
  follow-up items in §12.4. This keeps git history honest about
  the misdiagnosis.
- **Repurpose it as `feat/mq3-cross-arch-verification`.** Rename
  the branch, retain the cherry-picks, land items 1+2 from
  §12.4 (sweep results + coherence-gate row) on it, and drop the
  speculative kernel work entirely.

**Recommended: close, re-open.** A renamed branch with no
relationship to the original goal is procedurally cleaner, and the
post-mortem stays in this plan document so the diagnostic lesson is
preserved without polluting a future branch's purpose.

### 12.6 Updated synthesis verdicts (final)

| Hypothesis | Original v1 | After §10 review | After §12 resolution |
|---|---|---|---|
| H1: Prefill mis-wired | High | Low (per glm5 2A) | Excluded — captured path errors, non-captured uses correct decode |
| H2: Decode GEMV wrong on gfx10 | Medium | **High (promoted)** | **Excluded — decode was always correct; symptom was upstream AWQ omission** |
| H3: AWQ scale ignored | "Confirmed bug, irrelevant" | "Confirmed bug, irrelevant" + sidecar audit | **Was the actual root cause. The "irrelevant" verdict was wrong.** |
| H4: 4B+MQ3 quality cliff | Medium | Medium | Partially confirmed as a secondary effect — awq-only reason loop and gptq-only code rambling on the post-fix sweep are quant-quality artifacts, but never the primary cause |

