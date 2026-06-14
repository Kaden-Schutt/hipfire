# Signed-FWHT KV-cache family (Fwht{2,3,4}) for hipfire

> **Branch:** `feat/tbq` — plan revised 2026-05-15 after auditing existing infra.
> **Status:** OPEN. Pick this up cold; everything you need is in this doc.
> **Scope change vs seed commit:** plan originally targeted a single TBQ4 format
> as if it were a new orthogonal scheme. Audit found that hipfire's existing
> `asym{2,3,4}` family **already implements TurboQuant-style centroid quant on
> rotated K** — just with Givens instead of signed-FWHT. The work is therefore
> a rotation swap that fans out across three bit-widths, not a new family.

## What we're building

Three new KV-cache modes — `KvMode::Fwht2`, `Fwht3`, `Fwht4` — that mirror
the existing `Asym{2,3,4}` family bit-for-bit on storage and dispatch, but
replace the Givens K-rotation with the signed-FWHT primitive already shipped
for MQ4 weights.

## Existing infrastructure to reuse (no net-new work)

- **`kernels/src/turbo_common.h`**
  - `fwht_shfl_forward(a, b, c, d, signs1, signs2, tid)` — register-only
    128-element signed-FWHT via `__shfl_xor` / `ds_swizzle_b32`. Zero LDS,
    zero barriers, 7 butterfly stages. (turbo_common.h:81)
  - `fwht_shfl_forward_256(...)` — 256-element widened variant for hd=256.
    (turbo_common.h:168)
  - `fwht_shfl_inverse(...)` / `fwht_shfl_inverse_256(...)` — inverses.
  - `HADAMARD_BFLY(v, pattern, stride, tid)` macro — per-stage shuffle primitive.
- **`TURBO_C2/C3/C4[]` centroid LUTs** — Lloyd-Max-fit for `N(0, 1/128)`
  post-FWHT distribution per the `turbo_common.h:13` comment. Currently used
  by the asym path with Givens-rotated data (small distribution mismatch).
  After this work, they'll finally be used with the rotation they were
  calibrated for.
- **Storage layout** — identical to asym at every bit-width:
  - K: `4 B cnorm (FP32) + packed centroid indices` per head
  - V: Q8_0 via existing `kv_cache_write_q8_0.hip`
  - Per-head footprint at hd=256:
    - Fwht4: 132 B/head (matches asym4)
    - Fwht3: 100 B/head (matches asym3)
    - Fwht2: 68 B/head (matches asym2)
- **`cnorm = orig_norm / recon_norm`** per-head FP32 correction — same trick,
  unchanged. Carries the attention-sink fix regardless of rotation.
- **Scoring path** — `triattn_score_asym{2,3,4}.hip` and
  `attention_flash_asym{2,3,4}_tile{,_batched}.hip` reconstruct K via
  `cnorm * TURBO_Cn_256[idx]`. **Rotation-agnostic** — same scoring kernels
  serve both Givens and FWHT cache contents. No score-side work needed.

## What's actually new

Per bit-width (2/3/4), one K-write kernel pair:

- `kernels/src/kv_cache_write_asym_k_fwht{2,3,4}.hip`
- `kernels/src/kv_cache_write_asym_k_fwht{2,3,4}_batched.hip`

These are byte-by-byte copies of the existing `_givens{2,3,4}{,_batched}.hip`
files with two diffs:

1. Replace the `givens_forward(fa, fb, fc, fd, cos_theta, sin_theta, block_tid)`
   call with `fwht_shfl_forward(fa, fb, fc, fd, signs1, signs2, tid)`.
2. Replace the cos_theta/sin_theta kernel parameters with signs1/signs2.

Plus on the Rust side:

- Three new `KvCache::new_gpu_fwht{2,3,4}{,_capped,_multi}` constructors that
  mirror `new_gpu_asym{2,3,4}_*` but allocate a signs1/signs2 byte buffer
  (32 B per group, deterministic seed) instead of cos/sin FP32 LUT.
- `KvMode::{Fwht2, Fwht3, Fwht4}` enum variants in
  `crates/hipfire-arch-qwen35/src/speculative.rs`.
- Dispatch wrappers in `crates/rdna-compute/src/dispatch.rs`.
- CLI `--kv-mode` parsing in `crates/hipfire-runtime/examples/*` and the
  daemon `kv_mode` param. Accept `fwht2|fwht3|fwht4` as new values.

## Why this matters (audit findings)

1. **Existing cos/sin LUTs are fixed-seed random.** `KvCache::gen_givens_angles(42, n_blocks)`
   at `crates/hipfire-runtime/src/llama.rs:2976` generates rotation angles
   via linear-congruential PRNG seeded with the constant 42. No model
   parameter, no calibration. Every model loads the same LUT. The "Givens
   has potential for per-model calibration" hypothesis is falsified —
   nothing to give up by switching to fixed-seed random signed-FWHT.

2. **Centroid/rotation mismatch in current asym.** `TURBO_C2/3/4` are
   Lloyd-Max-fit for `N(0, 1/128)` post-FWHT distribution (per
   `turbo_common.h:13` comment). Used today with Givens output, which has
   higher per-element variance (Givens only mixes 4 elements; CLT effect
   is weaker). Small but consistent quality cost asym pays. Fwht mode
   eliminates it for free.

3. **Better outlier suppression at low bit-widths.** Givens mixes 4 dims
   per quad; an outlier in 1 of 4 dominates the quad post-rotation. Signed-FWHT
   mixes all 128 dims; the same outlier's amplitude per output drops by √32
   (~5.6×). Effect grows as bit-width shrinks: small for Fwht4, moderate for
   Fwht3, **substantial for Fwht2** (the regime where current asym2 is
   doc'd as "most lossy").

4. **Cheaper rotation on RDNA.** Signed-FWHT runs entirely through register
   ops + `__shfl_xor` / `ds_swizzle_b32` cross-lane primitives (1 cycle on
   wave32 RDNA3+). Givens loads a small cos/sin FP32 LUT into LDS. Per
   K-write the FWHT path is ~3-5× cheaper on cycle count, though the K-write
   is already a small fraction of decode time so this is a marginal win at
   best.

5. **Inherits zero of paroquant's trig pain.** Paroquant cost ~512
   transcendentals per group per gemv (8 rotation passes × 64 pairs). FWHT
   has zero transcendentals by construction. The "trig hurt us" objection
   to paroquant doesn't apply.

## Storage math (hd=256, identical to asym)

| Mode  | K bytes/head | V bytes/head | Total K+V | bpv (K) | bpv (K+V) |
|-------|--------------|--------------|-----------|---------|-----------|
| Fwht4 | 4 + 128 = 132 | 272 | 404 | 4.125 | 12.6 |
| Fwht3 | 4 + 96 = 100  | 272 | 372 | 3.125 | 11.6 |
| Fwht2 | 4 + 64 = 68   | 272 | 340 | 2.125 | 10.6 |

(V at Q8_0: `n_kv_heads × (head_dim / 32) × 34` bytes per position, identical
across all asym/fwht modes.)

## Execution plan

### Phase 1 — Fwht4 end-to-end (proof of pattern)
1. Copy `kernels/src/kv_cache_write_asym_k_givens4{,_batched}.hip` →
   `kv_cache_write_asym_k_fwht4{,_batched}.hip`. Replace the rotation call.
2. Add `KvCache::new_gpu_fwht4{,_capped,_multi}` mirroring the asym4
   constructors. Replace cos/sin allocation with signs1/signs2 byte buffer
   seeded by `gen_fwht_signs(42, head_dim)` (deterministic, same magic seed
   as gen_givens_angles for continuity).
3. Add `KvMode::Fwht4` variant + dispatch wrappers + CLI parsing.
4. **Coherence gate (mandatory per CLAUDE.md):**
   - `./scripts/coherence-gate.sh` full battery passes
   - Canonical 27B-3.5 LRU PEP-8 prompt with `--kv-mode fwht4 --max=120
     --temp 0.0` produces coherent code (compare side-by-side vs asym4
     baseline; both should be fluent, not byte-identical)
5. Bench: canonical decode tok/s gate — Fwht4 must land within ±5% of asym4
   (~199 tok/s on 7900 XTX). If <5% regress, accept; if >5% regress,
   investigate before proceeding.

### Phase 2 — Fan-out to Fwht3 + Fwht2
1. Mirror-edit the same diffs for asym3/asym2 → fwht3/fwht2. Same 2-line
   change per K-write kernel, same constructor pattern, same dispatch arm.
2. Each gets its own coherence-gate run with the canonical prompt.
3. Fwht3 is the **production-relevant tier** (matches asym3, the canonical
   default). Must land within ±5% tok/s of asym3.
4. Fwht2 is the **research-relevant tier** — strongest theoretical argument
   for quality improvement over asym2. Coherence-gate at greedy temp=0,
   then a longctx PPL probe (Phase 3).

### Phase 3 — Quality bench (the actual research question)
The performance gates in Phases 1-2 only protect against regression. The
quality question — does Fwht-rotated K preserve attention fidelity better
than Givens-rotated K — is answered here.

1. **Short-context coherence:** Canonical 27B-3.5 PEP-8 LRU prompt, all
   three Fwht modes vs all three asym modes. All six should produce
   coherent code at max=120, temp=0. Output need not be byte-identical;
   fluency is the bar.

2. **Long-context PPL on a small dataset (16k / 32k windows):**
   - Use `crates/hipfire-runtime/examples/perplexity.rs` with `--kv-mode
     fwht{2,3,4}` and `--kv-mode asym{2,3,4}`.
   - 6-row table at each context length: PPL delta vs Q8 (highest-fidelity
     KV baseline).
   - **Predicted outcome:**
     - Fwht4 ≈ Asym4 (both already near-optimal at 4-bit, rotation choice
       is in the noise)
     - Fwht3 slightly beats Asym3 at 32k (1-3% PPL recovery)
     - Fwht2 meaningfully beats Asym2 (5-15% PPL recovery — the headline
       result if it lands)
   - **Falsification trigger:** if Fwht2 doesn't beat Asym2 by at least 3%
     at 32k context, the rotation-quality hypothesis is wrong on Qwen3.5
     (likely because QK-norm already does the outlier suppression that
     rotation would do). Document and consider the family research-only.

3. **DFlash coherence-gate** (`./scripts/coherence-gate-dflash.sh`): if
   Fwht3 is to be considered as a production candidate, must pass the
   three-tier attractor gate per CLAUDE.md.

## Naming notes

- **`KvMode::Fwht{2,3,4}`** chosen over `Tq{2,3,4}` to avoid collision with
  llama.cpp's TQ ternary family and to be self-documenting.
- The seed branch name `feat/tbq` stays (history continuity). The reference
  fork (DrBearJew/llama.cpp@tbq4-rdna3-experiment) inspired the work
  direction but hipfire's implementation reuses local FWHT primitives, not
  the fork's ggml type machinery.
- Kernel files keep the `asym_k_` prefix (since K is rotated/quantized and
  V is Q8 — the asymmetric structure is unchanged); only the rotation suffix
  differs: `kv_cache_write_asym_k_fwht{2,3,4}.hip`.

## Constraints (CLAUDE.md + feedback memory)

- **PR gating policy (`feedback_pr_gating_policy.md`):** these are *additive*
  modes, gated behind explicit `--kv-mode fwht{2,3,4}` selection. Asym3 stays
  canonical default. Land freely once coherence + perf gates pass.
- **Coherence gate mandatory** per CLAUDE.md before commit. Both
  `coherence-gate.sh` and `coherence-gate-dflash.sh` (for DFlash-relevant
  changes).
- **Canonical bench config** per CLAUDE.md: 27B-3.5 LRU PEP-8 strict prompt,
  max=120, no-chatml, prompt_normalize=true (default). Expected:
  199 tok/s τ=10.36 on 7900 XTX for asym3 baseline.
- **No Python in inference hot path.** Centroid LUTs are baked into
  `turbo_common.h` at compile time (already are); sign vectors generated by
  Rust constructor at KvCache creation, uploaded to GPU once.

## Risks & open questions

1. **Falsification prior is non-trivial.** Per memory entries
   `*_falsified_*`, hipfire has had ≥4 raw-perf swaps falsify between
   microbench wins and production losses in 2026 (FP8 dot4, gfx11 fdot2,
   MFP4-v3-#1, v2 SGPR-LUT). Rotation swap is the same lever class. Default
   prior on perf: should land within ±2-3% of asym (not a meaningful gap
   either way). Default prior on quality: Fwht2 wins big, Fwht3/4 land
   within noise. The quality result is the only reason to ship this.

2. **QK-norm dampens the rotation lever.** Qwen3.5/3.6 (primary target),
   Gemma3/Gemma4 (planned target), Llama 3.2+ (potential) all use QK-norm.
   K-vectors are explicitly normalized per head before being written to
   cache, so the heavy-tail outlier problem rotation was designed to fix is
   partly pre-solved by the architecture. Expect smaller quality gains than
   on Llama 2-era arches.

3. **Hd=128 vs hd=256.** Existing kernels handle both. Verify the
   `fwht_shfl_forward_128` vs `fwht_shfl_forward_256` selection lines up
   correctly with the head_dim dispatch in the K-write code.

4. **`__shfl_xor` width assumption.** The FWHT shuffle path assumes wave32
   (RDNA1-4). gfx9xx/CDNA would need a wave64 variant — but hipfire is
   RDNA-only per CLAUDE.md, so this is not a constraint.

5. **What if Fwht2 doesn't win?** Then the rotation-quality hypothesis is
   falsified on QK-normed models, and the right disposition is "ship Fwht{4,3,2}
   as additive opt-in modes; keep asym3 as default; document the negative
   result for the rotation-on-K-vectors literature." Same disposition pattern
   as the MFP4 / FP8 / fdot2 falsified entries — no problem shipping the
   infra, only a problem shipping a new default.

## Bench gate (definition of done)

For each Fwht{4,3,2} to be considered shippable:
- [ ] Coherence-gate full battery passes
- [ ] Canonical 27B-3.5 LRU decode tok/s within ±5% of paired asym variant
- [ ] Coherent code on canonical PEP-8 prompt at max=120, temp=0
- [ ] At least one 16k+ PPL data point captured (compares to paired asym
  and to Q8 baseline)

For Fwht3 to replace Asym3 as canonical (separate, later decision):
- [ ] All four above
- [ ] τ ≥ 10.0 on canonical DFlash bench (current 10.36)
- [ ] coherence-gate-dflash passes
- [ ] ≥3% PPL improvement at 32k context (or matching at all contexts with
  no quality regression on the canonical PEP-8 prompt)

## Sister branch

`feat/rtq` ships RotorQuant — same KV-cache scope, different rotation
(2D-Givens / 4D-quaternion learned per tensor). That branch can land in
parallel; both branches stay independent and bench-comparable. RotorQuant
would be the calibration-freedom path that this plan deliberately doesn't
take.

## Reference

- `crates/hipfire-runtime/src/llama.rs:2976` — `gen_givens_angles` (the
  fixed-seed PRNG that established Givens has no calibration).
- `kernels/src/turbo_common.h:13,81,168` — Lloyd-Max comment + FWHT shuffle
  primitives.
- `kernels/src/kv_cache_write_asym_k_givens2.hip` — reference for the
  K-write structure to mirror.
- `kernels/src/triattn_score_asym3.hip` — scoring kernel (unchanged by this
  work).
- DrBearJew/llama.cpp@tbq4-rdna3-experiment — original external inspiration
  (different impl, same conceptual direction).
- Commit `b7e55f47 feat(kv-cache): asym{4,3,2} family replaces givens` —
  historical migration that established the current Givens-rotated +
  centroid-quantized + cnorm-corrected asym layout.
