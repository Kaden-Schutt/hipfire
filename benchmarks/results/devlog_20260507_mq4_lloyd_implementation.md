# Dev log 2026-05-07 — MQ4-Lloyd implementation: Phase 1 + Phase 2 + multi-acc bisect

**Branch:** `feat/issue-182-mq4-lloyd` (off `lloyd-max-mq3-spike` →
`upstream/master + Lloyd-MQ3 perf chain + this branch's MQ4-Lloyd work`).
**Target hardware (this session):** AMD Ryzen AI MAX+ 395 / Radeon 8060S
(gfx1151, RDNA3.5 Strix Halo APU, ROCm 7.12). gfx1100 (the MQ-Lloyd
calibrated arch) is deferred — definitive perf comparisons happen
there. This session is the conformance-and-quality cut.

**Continues from:**
[`devlog_20260506_lloyd_mq4_extension.md`](./devlog_20260506_lloyd_mq4_extension.md)
which proposed the extension and projected 9B PPL ~4.30 (full Lloyd
ratio) or ~6.0 (halved). This devlog captures the actual Phase 1 + 2
outcomes and the multi-accumulator bug bisect.

## TL;DR

- **Phase 1 (quantizer + slow GEMV + viability PPL): green.** Local
  9B-Qwen3.5 wikitext2-flavored corpus PPL **12.4759** vs uniform MQ4
  **14.6820** → MQ4-Lloyd is **0.850× MQ4 PPL = 15% better**.
  Projected onto issue #182's wikitext2-test baseline (MQ4=7.78):
  ~6.61. Lands between the issue's "abandon at ≥7.78" and "ship at
  ≤6.0" gates — clearly NOT abandon-territory; ship decision depends
  on Phase 2 perf delivery, not quality.

- **Phase 2 (5 fast kernels + dispatch + qwen35 wiring + parity):
  green.** All 5 gfx1100 fast kernels (basic GEMV, residual, fused
  gate+up, fused QKV, fused QKVZA) ship single-accumulator, byte-equal
  to slow generic at 10-decimal NLL/tok precision under full inference
  wiring. Standalone parity tests pass with max-abs ≤ 5e-3 fp32-reorder
  tolerance (actual measurements 7e-7 to 2e-6).

- **Multi-accumulator bug story: not what it looked like.** Initial
  framing was "MQ3-Lloyd's K4 multi-acc works, MQ4-Lloyd's doesn't —
  why the asymmetry?" After bisecting, the answer is that
  MQ4-Lloyd multi-acc was never structurally broken — both MQ3 and
  MQ4 multi-acc kernels produce fp32-reorder-noise drift of essentially
  identical per-call magnitude. The 1.7% PPL drift came from
  **multi-acc coverage**, not the kernel. With matched coverage MQ3
  also drifts (~0.9% PPL); MQ4 just drifts ~2× MQ3 under matched
  conditions, plausibly from LDS-layout / codebook-span effects.

- **Production decision:** ship MQ4-Lloyd with single-accumulator
  kernels because they match the slow generic accumulation order
  byte-equal regardless of coverage. MQ3-Lloyd's existing multi-acc
  kernels keep shipping unchanged.

## Phase 1 — quantizer + slow GEMV + viability PPL

(Detailed methodology: [`findings/mq4-lloyd-9b-ppl.md`](../../findings/mq4-lloyd-9b-ppl.md).)

Implemented in commit `7ef567d`:

- `crates/hipfire-quantize/src/main.rs`: `quantize_mq4g256_lloyd`
  (K=16 centroids, 32 B header + 128 B 4-bit nibble indices = 160 B/group,
  +17.6% over uniform MQ4). Direct port of `quantize_mq3g256_lloyd`'s
  Lloyd k-means with K=16 percentile init and deterministic centroid
  sort.
- `kernels/src/gemv_mq4g256_lloyd.hip`: chip-agnostic slow generic
  GEMV (16-way ternary lookup, 4-bit nibble unpack). 38 VGPR / 0 LDS /
  0 spills.
- `DType::MQ4G256Lloyd` (qt=21) plumbed through rdna-compute dispatch
  + bytes accounting + Rust bindings; loaders in `hfq.rs` and
  `qwen35.rs`; basic `weight_gemv` / `weight_gemv_prerotated` /
  `fused_rmsnorm_rotate_for_mq` / `rotate_x_for_mq` arms.
- `--format mq4-lloyd` CLI + `--allow-mq4-lloyd` research-only gate.

### 9B PPL on local wikitext-flavored corpus (`benchmarks/calib/calib-5m.txt`)

```
ctx=2048  warmup=8  offset=0  gfx1151
```

| format        | B/group | NLL/tok    | PPL    | ratio vs MQ4 |
|---------------|--------:|-----------:|-------:|-------------:|
| MQ4 (uniform) |     136 |   2.6866   | 14.68  |        1.000× |
| MQ3-Lloyd     |     112 |   3.2111   | 24.81  |        1.690× |
| **MQ4-Lloyd** | **160** | **2.5238** | **12.4759** | **0.850×** |

The MQ3-Lloyd / MQ4 PPL ratio matches issue #182's local wikitext2-test
ratio to 0.5 % (1.69× vs 1.68×), so the within-corpus comparison is
methodologically sound. Projecting onto issue #182's wikitext2-test
baseline:

| Source                                | Lloyd-vs-uniform ratio @ 4-bit | Projected 9B MQ4-Lloyd PPL (#182 corpus) |
|---------------------------------------|-------------------------------:|------------------------------------------:|
| Full extrapolation (1.81× from 3-bit) |                          0.553× |                                     ~4.30 |
| Half extrapolation                    |                          0.776× |                                     ~6.04 |
| **Observed (this devlog)**            |                      **0.850×** |                                  **~6.61** (projected) |

The observed Lloyd ratio (0.85×) is half-of-half of the 3-bit gain
(0.55×) — issue #182's open question 1 was *"Plausibly smaller, but
even halved is still a win"*; it landed at half-of-half but is still
a clear quality lift over uniform MQ4.

## Phase 2 — fast kernels + dispatch + wiring + parity

Phase 2 ran across commits `3a86201`, `a2ca634`, `641b865`, `ca4f7e5`.
Five kernel families, each with chip-agnostic .hip + gfx1100 .hip:

| Variant                  | Slow .hip resources         | gfx1100 .hip resources           |
|--------------------------|-----------------------------|-----------------------------------|
| `gemv_mq4g256_lloyd`     | 38 VGPR /  0 LDS / 0 spills | 71 VGPR / 256 B LDS / 0 spills    |
| `_residual`              | 38 VGPR /  0 LDS / 0 spills | 71 VGPR / 256 B LDS / 0 spills    |
| `fused_gate_up_*_lloyd`  | 38 VGPR /  0 LDS / 0 spills | 71 VGPR / 256 B LDS / 0 spills    |
| `fused_qkv_*_lloyd`      | 38 VGPR /  0 LDS / 0 spills | 71 VGPR / 256 B LDS / 0 spills    |
| `fused_qkvza_*_lloyd`    | 38 VGPR /  0 LDS / 0 spills | 71 VGPR / 256 B LDS / 0 spills    |

The gfx1100 fast variants use **K4 unroll + 64-slot LDS-resident
codebook (cooperative two-phase load) + SINGLE linear accumulator**.
The single-accumulator choice is non-obvious and fundamental — see
the multi-acc bisect section below.

`gfx1151` (Strix Halo APU, RDNA3.5) included in the fast-arm matcher
for on-host conformance testing. gfx1100 is the calibrated perf
target — definitive bench numbers there are deferred to a future
session.

### qwen35 + llama wiring

All MQ4-Lloyd routing arms added:

- `weight_gemv_residual` (llama.rs): MQ4G256Lloyd → `gemv_mq4g256_lloyd_residual`
- `weight_gemv_swiglu_residual` (llama.rs): MQ4G256Lloyd → `fused_silu_mul_rotate_mq + gemv_mq4g256_lloyd_residual`
- 14 sites in qwen35.rs (5 fused gate+up, 4 fused QKVZA-LA, 5 fused QKV-FA): MQ4G256Lloyd arms route to the new fused kernels

### End-to-end PPL byte-equality (the gold-standard conformance gate)

Qwen3.5-9B / calib-5m / ctx=2048 / warmup=8 / offset=0 / gfx1151:

```
fast (default):                       NLL/tok = 2.5237956800   PPL = 12.4759   32.5 tok/s
slow (HIPFIRE_LLOYD_FORCE_BASELINE=1): NLL/tok = 2.5237956800   PPL = 12.4759    9.6 tok/s
```

**Byte-equal at 10 decimal places.** The fast kernel family is correct
and is **3.4× faster** than slow generic on this gfx1151 host. gfx1100
expected to be higher (calibrated arch + GDDR6 vs LPDDR5x).

### Standalone parity tests

`crates/rdna-compute/examples/test_gemv_mq4g256_lloyd_tail.rs` (K-sweep
on basic GEMV) and `test_mq4g256_lloyd_fused_parity.rs` (residual + 3
fused variants at K=4096) both pass for slow and fast at max-abs
≤ 5e-3 fp32-reorder tolerance. Actual measurements: 5e-7 to 2e-6.

The standalone parity tests are necessary but **not sufficient** —
they passed for the buggy multi-acc kernel that produced 1.7% PPL
drift on real model. PPL byte-equality is the decisive gate.

## Multi-accumulator bisect — what we learned

(Full writeup: [`findings/mq4-lloyd-multiacc-investigation.md`](../../findings/mq4-lloyd-multiacc-investigation.md).)

**Initial framing:** during P2-B bring-up, the K4 multi-accumulator
pattern (4 separate `acc0..acc3` registers + final
`(acc0+acc1)+(acc2+acc3)` merge — verbatim port from
`gemv_mq3g256_lloyd.gfx1100.hip`) produced a 1.7% PPL drift on
Qwen3.5-9B vs the slow generic kernel, despite passing the synthetic
parity test at fp32-reorder tolerance. The drift went away when
switching to single-accumulator. MQ3-Lloyd uses the same multi-acc
pattern and was byte-equal, so the question was: why the asymmetry?

**Bisect result:** the asymmetry was a **coverage** artifact, not a
structural bug.

Per-call multi-acc-vs-CPU drift on real Qwen3.5-9B weights
(measured via `diag_mq{3,4}_lloyd_multiacc.rs` against a CPU reference
that matches the slow generic kernel byte-equal):

| Tensor (K)                          | MQ3 multi-acc max-abs | MQ4 multi-acc max-abs |
|-------------------------------------|---------------------:|---------------------:|
| `linear_attn.in_proj_qkv` (K=4096)  |               9.5e-7 |               1.1e-6 |
| `mlp.gate_proj`           (K=4096)  |               5.7e-7 |               5.4e-7 |
| `mlp.down_proj`           (K=12288) |               2.3e-6 |               1.8e-6 |

Per-GEMV magnitudes are essentially identical between MQ3 and MQ4 —
both pure fp32 reorder noise.

The 1.7% PPL drift came from the **fraction of GEMV calls** running
through multi-acc kernels at the time of measurement. The original
test was BEFORE residual/fused wiring landed, so:

- `weight_gemv_residual` had no MQ4-Lloyd arm in `llama.rs` → fell
  through to `weight_gemv + add_inplace_f32` → multi-acc fast GEMV
- `qwen35.rs` had no MQ4-Lloyd fused arms → fell through to
  per-projection `weight_gemv_prerotated` → multi-acc fast GEMV

Result: **~100% of inference GEMV calls ran through the multi-acc
kernel.** With ~1e-6 per-call drift × 200 GEMVs/token × 2K tokens ×
softmax non-linearity, drift compounds to 0.0166 NLL = 1.7% PPL.

After fused/residual wiring landed (as single-acc production kernels
in `a2ca634` + `641b865`), only the output `wo` projection still uses
standalone GEMV — about 10% coverage. Per-call drift × 0.1 coverage
no longer compounds enough to surface at 6-decimal NLL/tok precision.

The MQ3-Lloyd comparison happened to land in the partial-coverage
regime on gfx1151 the whole time (matcher only included gfx1151 for
basic GEMV; fused/residual stayed on slow generic). So MQ3 looked
byte-equal, MQ4 looked broken — but it was about coverage.

**Coverage repro:** swapping all 5 MQ4-Lloyd gfx1100 fast kernels to
multi-acc bodies and re-running PPL with the full qwen35 wiring
restores the original 1.7% drift exactly:

```
All 5 multi-acc:    NLL/tok = 2.5403703159   PPL = 12.6844
Slow generic:       NLL/tok = 2.5237956800   PPL = 12.4759
                                  Δ = 0.0166 = 1.7% PPL drift
```

Restoring single-acc kernels → byte-equal.

**MQ3-Lloyd under matched full coverage:** when all 5 MQ3-Lloyd
gfx1100 fast variants were enabled on gfx1151 (a brief test before
backing off to GEMV-only), MQ3 also drifted: ~0.9% PPL (24.81 → 25.03).
About half of MQ4's 1.7%. The 2× ratio could come from:

- **LDS layout.** MQ4's 64-slot codebook spans 2 LDS bank rows; MQ3's
  32-slot fits in 1. Different read scheduling could change fp-op
  issue ordering and per-call rounding profile.
- **Codebook value distribution.** Lloyd k-means at K=16 captures
  finer detail in the FWHT-rotated weight distribution — possibly
  with wider absolute magnitudes via the additional centroids,
  amplifying per-FMA reorder noise.
- **Variance.** A 2× ratio across a finite corpus could partially be
  statistical noise.

None of these point to a bug. Both kernels are algebraically correct;
they just emit fp32 reorder noise that compounds with coverage.

**Production decision:** single-accumulator for all 5 MQ4-Lloyd fast
kernels. Single-acc removes the merge step, matches slow's
accumulation order byte-equal, and is robust to any future coverage
changes (e.g. wiring more inference paths through the fast kernels).

The kernel header in `gemv_mq4g256_lloyd.gfx1100.hip` warns against
switching back to multi-acc without re-validating PPL byte-equality.

**MQ3-Lloyd:** existing multi-acc kernels remain. The follow-up
gfx1100 measurement (see findings update 2026-05-07) showed gfx1100
multi-acc drift is the *same magnitude* as gfx1151's but with
*opposite sign* (NLL-favorable on gfx1100, NLL-disfavorable on
gfx1151) — consistent with intrinsic fp32 reorder noise where the
direction is essentially random across the compiler's per-arch FMA
scheduling. Both arches carry the same drift envelope; coherence
gates clear it on either side. PR #189 enabled MQ3-Lloyd full
coverage on gfx1151 for deployment parity with the gfx1100 deployment
shape. A single-acc port for MQ3-Lloyd is tracked under issue #188 —
the trade-off is ~2% decode perf cost on gfx1100 (121.7 → 119.2 tok/s,
just below #181's ≥120 ship gate) for byte-equal PPL parity with
slow generic; pending maintainer call.

## Performance footnote (gfx1151, conformance-only)

Decode tok/s on Qwen3.5-9B / calib-5m PPL run:

```
slow generic kernel:       9.6 tok/s
fast (single-acc, all 5):  32.5 tok/s   (3.4× speedup)
```

This is **not** the headline perf number — gfx1100 is the calibrated
target and has higher memory bandwidth (GDDR6 vs LPDDR5x). The
gfx1151 measurement is here only as a sanity check that the fast
kernels actually run faster than slow, and they do. MQ3-Lloyd on
gfx1100 ships at ~120 tok/s on 9B per the 2026-05-06 devlog; MQ4-Lloyd
should land in a similar range, possibly slightly lower due to the
larger 160 B/group bandwidth. To be measured.

## Open follow-ups — status (2026-05-07 update)

1. **Validate on gfx1100** — *partially closed*. Multi-acc drift
   bisect on gfx1100 done (see #2 below); PPL byte-equality + decode
   tok/s + issue #182 wikitext2-test quality framing still pending the
   final maintainer pass on the calibrated arch.
2. **Bisect MQ3 vs MQ4 multi-acc drift on gfx1100** — **closed**.
   Drift is *universal* across gfx1100 + gfx1151 (same per-call
   magnitude ~5e-7 to 2e-6 on real Qwen3.5-9B weights). Full-coverage
   PPL drift on gfx1100 is the *same magnitude* as gfx1151 but
   *opposite sign* — fp32 reorder direction is essentially random per
   arch. See `findings/mq4-lloyd-multiacc-investigation.md` "Update
   2026-05-07" section for the data. Conclusion: single-acc is
   universally correct for MQ4-Lloyd (already shipping); MQ3-Lloyd
   would benefit too at ~2% decode cost (issue #188).
3. **MQ3-Lloyd → single-acc port** — **WIP under issue #188**.
   Branch `feat/188-mq3-lloyd-single-acc-gfx1100` has the 5-kernel
   port; gfx1100 cross-process A/B shows 121.7 → 119.2 tok/s on
   Qwen3.5-9B (−2.05%, misses #181's ≥120 ship gate by 0.8 tok/s).
   Maintainer decision pending: keep multi-acc + accept arch-dependent
   reorder drift, port single-acc + accept 2% perf hit, or
   arch-conditional dispatch. Out of scope for this MQ4-Lloyd PR.
4. **Smaller-model MQ4-Lloyd PPL** — *pending*. Issue #182 suggested
   0.8B + 4B first; landed on 9B directly. Quick re-run would confirm
   Lloyd ratio is stable across model scales.
5. **Cross-corpus calibration** — *pending*. Confirm the ~1.9×
   absolute-level gap between local calib-5m.txt and issue #182's
   wikitext2-test path is purely corpus, not implementation.

## Artifacts

Production kernels (single-acc):

- `kernels/src/gemv_mq4g256_lloyd.{,gfx1100.}hip`
- `kernels/src/gemv_mq4g256_lloyd_residual.{,gfx1100.}hip`
- `kernels/src/fused_gate_up_mq4g256_lloyd.{,gfx1100.}hip`
- `kernels/src/fused_qkv_mq4g256_lloyd.{,gfx1100.}hip`
- `kernels/src/fused_qkvza_mq4g256_lloyd.{,gfx1100.}hip`

Diagnostic infrastructure (kept for future bisects):

- `kernels/src/gemv_mq4g256_lloyd_multiacc_diag.gfx1100.hip` — the
  multi-acc body, exposed via `Gpu::gemv_mq4g256_lloyd_multiacc_diag`
- `crates/hipfire-runtime/examples/diag_mq4_lloyd_multiacc.rs`
- `crates/hipfire-runtime/examples/diag_mq3_lloyd_multiacc.rs`
- `crates/rdna-compute/examples/test_gemv_mq4g256_lloyd_tail.rs`
- `crates/rdna-compute/examples/test_mq4g256_lloyd_fused_parity.rs`

Findings docs:

- `findings/mq4-lloyd-9b-ppl.md` — Phase 1 viability + quality projection
- `findings/mq4-lloyd-multiacc-investigation.md` — multi-acc bisect

Sibling PR (out of MQ4-Lloyd scope):

- **PR #189 — `feat(mq3-lloyd): enable fast variants on gfx1151
  (Strix Halo APU) — parity with gfx1100`** — adds gfx1151 to all 5
  MQ3-Lloyd matchers, matching gfx1100's existing full-coverage
  deployment shape. ~2× decode speedup on Qwen3.5-9B (18.2 →
  46.4 tok/s) with the same documented multi-acc drift envelope as
  gfx1100. Cross-references this devlog and findings doc.
  See `benchmarks/results/devlog_20260507_mq3_lloyd_gfx1151.md` for
  the full perf + drift snapshot.

## Bench-host quirks

- gfx1151 (Strix Halo APU) consistently SIGSEGVs during ROCm teardown
  after metrics print. Doesn't affect bench numbers (printed before
  teardown). Exit codes are 139 even on otherwise-successful runs.
- `~/.hipfire/models/qwen36-27b-dflash-mq4.hfq` filename uses `qwen36`
  (no dot) but the model ID in code is `qwen3.6` (with dot). Cosmetic;
  noted here for grep-ability.
