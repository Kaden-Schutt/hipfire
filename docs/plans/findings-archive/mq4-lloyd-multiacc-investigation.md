# MQ3 vs MQ4 multi-accumulator drift — investigation

**Question (issue #182 follow-up):** Why does the K4 multi-accumulator merge
`(acc0+acc1)+(acc2+acc3)` work byte-equal at PPL on MQ3-Lloyd but produce a
1.7% PPL drift on MQ4-Lloyd? The kernels are structurally identical apart
from K=8 vs K=16 codebook.

**Hardware:** AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151, RDNA3.5
Strix Halo APU, ROCm 7.12). Investigation done on this host; gfx1100 not
available locally.

**Date:** 2026-05-07.

## TL;DR — the framing was misleading

Both MQ3-Lloyd and MQ4-Lloyd K4 multi-accumulator kernels produce
**fp32-reorder-noise drift of essentially the same per-call magnitude**
(~1e-6 max-abs at K=4096, ~2e-6 at K=12288, on real Qwen3.5-9B weights).
The "MQ3 works, MQ4 doesn't" framing was an artifact of **multi-acc
coverage**, not a structural bug in MQ4.

When *all* GEMV call sites in the Qwen3.5 forward pass route through a
multi-acc kernel, per-call drift compounds across ~200 GEMVs/token × 2K
tokens × softmax non-linearity into a measurable NLL drift:

| Coverage scenario | MQ3-Lloyd NLL drift | MQ4-Lloyd NLL drift |
|---|---:|---:|
| Partial multi-acc (only basic GEMV; fused/residual stay slow or single-acc) | 0 (byte-equal at 10 dp) | 0 (byte-equal at 10 dp) |
| Full multi-acc (basic GEMV + residual + fused gate+up + fused QKV + fused QKVZA) | +0.009 (24.81 → 25.03 PPL) | +0.0166 (12.4759 → 12.6844 PPL) |

The MQ4 fix (single-accumulator) eliminates the drift by removing the
`(acc0+acc1)+(acc2+acc3)` merge, which restores byte-equal PPL even
under full coverage. MQ3-Lloyd's existing kernels still ship multi-acc
because they were validated on gfx1100 (the calibrated arch) where the
drift may be even smaller — and on this host the partial-coverage
deployment (only basic-GEMV multi-acc on gfx1151) is byte-equal.

## How the original framing went wrong

The original 1.7% MQ4-Lloyd PPL drift was measured *before* the
residual/fused-kernel wiring landed. At that point:

- `weight_gemv_residual` had no MQ4-Lloyd arm in `llama.rs` → fell
  through to `weight_gemv + add_inplace_f32`, which routed through the
  multi-acc fast GEMV.
- `qwen35.rs` had no MQ4-Lloyd fused arms (gate+up / QKV / QKVZA) → fell
  through to per-projection `weight_gemv_prerotated`, also routing
  through the multi-acc fast GEMV.

Result: ~100% of GEMV calls during inference ran through the multi-acc
kernel. After the fused/residual variants were wired in (as single-acc
production kernels, post-fix), only the output `wo` projection still
used the standalone multi-acc kernel — about 10% coverage. Per-call
drift × 0.1 coverage no longer compounds enough to surface at 6-decimal
NLL/tok precision.

The MQ3-Lloyd comparison happened to land in the lower-coverage regime
because the gfx1151 arch matcher only included gfx1151 in the basic
GEMV arm at the time (fused/residual MQ3-Lloyd variants stayed on the
slow generic kernel). So MQ3 had partial multi-acc coverage and MQ4
had full — the apparent asymmetry was about coverage, not the kernel
itself.

## Confirming the coverage hypothesis

To verify the multi-acc kernel itself is not "broken" but just produces
fp-reorder noise that compounds with coverage, I:

1. Resurrected the multi-acc body as
   [`kernels/src/gemv_mq4g256_lloyd_multiacc_diag.gfx1100.hip`](../kernels/src/gemv_mq4g256_lloyd_multiacc_diag.gfx1100.hip)
   with explicit `gemv_mq4g256_lloyd_multiacc_diag` symbol so it could
   be invoked alongside the production single-acc kernel.

2. Wrote per-row diagnostic binaries:
   [`diag_mq4_lloyd_multiacc.rs`](../crates/hipfire-runtime/examples/diag_mq4_lloyd_multiacc.rs)
   and the MQ3-Lloyd sibling
   [`diag_mq3_lloyd_multiacc.rs`](../crates/hipfire-runtime/examples/diag_mq3_lloyd_multiacc.rs).
   Each loads one weight tensor from a real .hfq, runs the multi-acc
   kernel via the production dispatcher, and compares per-row against
   a CPU reference (which matches the slow generic kernel byte-equal).

3. Confirmed similar per-call magnitudes on real Qwen3.5-9B weights:

   | Tensor (K) | MQ3-Lloyd multi-acc max abs | MQ4-Lloyd multi-acc max abs |
   |---|---:|---:|
   | `linear_attn.in_proj_qkv` (K=4096)   | 9.54e-7 | 1.07e-6 |
   | `mlp.gate_proj`           (K=4096)   | 5.66e-7 | 5.36e-7 |
   | `mlp.down_proj`           (K=12288)  | 2.32e-6 | 1.76e-6 |

   Both are within fp32 reorder-noise bounds. MQ3 is sometimes higher,
   sometimes lower — on average, identical magnitude.

4. Reproduced the 1.7% PPL drift by swapping all five MQ4-Lloyd
   gfx1100 fast kernels (basic GEMV, residual, fused gate+up, fused
   QKV, fused QKVZA) to multi-acc bodies and re-running PPL on
   Qwen3.5-9B with full wiring:

   ```
   All 5 kernels multi-acc:  NLL/tok = 2.5403703159  PPL = 12.6844
   Slow generic baseline:    NLL/tok = 2.5237956800  PPL = 12.4759
   ```

   NLL drift = 0.0166 = 1.7% PPL drift, exactly matching the original
   pre-wiring measurement. With the production single-acc kernels back
   in place: NLL/tok = 2.5237956800 byte-equal to slow.

## Why MQ4 drift is ~2× MQ3 drift (under matched coverage)

Under matched full multi-acc coverage on gfx1151:

- MQ3-Lloyd: NLL drift 0.009 (24.81 → 25.03 PPL ≈ +0.9%)
- MQ4-Lloyd: NLL drift 0.0166 (12.48 → 12.68 PPL ≈ +1.7%)

The 2× ratio is real but the per-GEMV measurements show similar
magnitudes. Plausible contributors (not bisected — would need gfx1100
hardware to control for arch-specific LDS behavior):

- **LDS layout.** MQ4's 64-slot codebook spans 2 LDS bank rows; MQ3's
  32-slot fits in 1 row. The compiler may schedule LDS reads
  differently, producing different fp-op-issue ordering and therefore
  different per-call rounding.
- **Codebook value distribution.** Lloyd k-means with K=16 produces
  centroids covering finer detail in the FWHT-rotated weight
  distribution — possibly with wider absolute magnitudes captured
  by the additional centroids, which would make the per-FMA values
  larger and amplify reorder noise per call.
- **Variance.** A 2× ratio across PPL evaluations on a finite corpus
  could partially be statistical noise. Without multi-corpus repeats
  it's hard to separate signal from noise here.

None of these point to a bug. The multi-acc structure is algebraically
correct on both formats; it just compounds fp32 reorder noise more
visibly on MQ4 in the regime tested.

## Why single-accumulator is the right MQ4-Lloyd production choice

The single-acc structure removes the merge step entirely: each
quad-iter accumulates all 4 group contributions directly into one
linear `acc` register. This produces the **same fp32 op order as the
slow generic kernel**, so the production fast kernels are byte-equal
to slow at 10-decimal NLL precision regardless of coverage. The
production family (`gemv_mq4g256_lloyd`, `gemv_mq4g256_lloyd_residual`,
`fused_gate_up_mq4g256_lloyd`, `fused_qkv_mq4g256_lloyd`,
`fused_qkvza_mq4g256_lloyd`) all use this pattern and pass the PPL
byte-equality gate on gfx1151.

For MQ3-Lloyd, the existing kernels keep multi-acc — they were
validated on gfx1100 (the calibrated target arch) where the drift may
be smaller, and the production deployment on gfx1151 currently uses
partial coverage (only basic GEMV multi-acc; residual/fused stay slow)
which is byte-equal to slow. Porting MQ3-Lloyd to single-acc is a
follow-up that would close the residual full-coverage drift gap on
gfx1151 (and possibly other archs); not blocking, since current
deployments are clean.

## Open questions left for gfx1100 sessions

1. **Does the MQ3-Lloyd multi-acc full-coverage drift exist on gfx1100,
   or only on gfx1151?** If gfx1100 is byte-equal, the gfx1151 drift
   is an arch-specific LDS-scheduling artifact. If gfx1100 also drifts
   by ~0.9%, MQ3-Lloyd has the same latent bug as MQ4 had pre-fix
   (just smaller magnitude).
2. **Same question for MQ4-Lloyd multi-acc.** Was the original 1.7%
   drift gfx1151-specific, or would the same kernel drift on gfx1100
   too? If gfx1100 also drifts, single-acc is a universally correct
   choice. If only gfx1151 drifts, multi-acc could ship for gfx1100
   with arch-conditional dispatch.

The investigation infrastructure (multiacc_diag kernel + Rust binding +
diag binaries) is in this PR for future reuse on a gfx1100 host.

## Update 2026-05-07 — gfx1100 confirmation (closes Q1)

Open question 1 answered on gfx1100 (7900 XTX, ROCm 7.2). Per-call
multi-acc-vs-CPU drift on real Qwen3.5-9B weights is essentially the
same magnitude as gfx1151:

| Tensor (K) | MQ3-Lloyd gfx1100 | MQ3-Lloyd gfx1151 | MQ4-Lloyd gfx1100 | MQ4-Lloyd gfx1151 |
|---|---:|---:|---:|---:|
| `mlp.gate_proj` (K=4096)   | 5.96e-7 | 5.66e-7 | 5.36e-7 | 5.36e-7 |
| `mlp.down_proj` (K=12288)  | 2.35e-6 | 2.32e-6 | 1.76e-6 | 1.76e-6 |

So Q2 (does MQ4-Lloyd multi-acc drift on gfx1100 too) is also
answered yes — same per-call magnitude as gfx1151. Single-acc is the
universally correct choice; multi-acc reorder noise is intrinsic to
the kernel structure, not arch-specific.

Full-coverage PPL drift on gfx1100, `qwen3.5-9b.mq3-lloyd`,
ctx=2048 warmup=8 offset=0, calib-5m corpus:

| Variant | NLL/tok | PPL | Δ vs slow |
|---|---:|---:|---:|
| Multi-acc production (PR #181) | 3.170653 | 23.8230 | **−0.0093** |
| Slow generic (`HIPFIRE_LLOYD_FORCE_BASELINE=1`) | 3.179968 | 24.0460 | (canonical) |
| Single-acc port (issue #188 WIP) | 3.179968 | 24.0460 | **byte-equal** |

Same magnitude as gfx1151's drift (Δ NLL ±0.01 nats), but **opposite
sign** — gfx1100 multi-acc lands NLL-favorable, gfx1151 multi-acc
lands NLL-disfavorable. Both consistent with fp32 reorder noise; the
direction is essentially random across the compiler's per-arch FMA
scheduling. The "MQ3 fine, MQ4 broken" framing was always coverage,
never a structural MQ4 bug — re-confirmed.

Decode perf cost of porting MQ3-Lloyd to single-acc on gfx1100
(`probe_commits.sh` cross-process A/B, 3 samples each, gen=30
GRAPH=1 KV=asym3, qwen3.5-9b.mq3-lloyd):

| Variant | Mean | Δ vs multi-acc |
|---|---:|---:|
| Multi-acc (production) | 121.7 tok/s | — |
| Single-acc (5 kernels ported, K4+LDS preserved) | 119.2 tok/s | **−2.05%** |

Misses PR #181's ≥120 tok/s ship gate by 0.8 tok/s. The decision
(keep multi-acc + document arch-dependency vs port single-acc + accept
2% perf hit vs arch-conditional dispatch) is open for maintainer input
at issue #188. Single-acc implementation is on branch
`feat/188-mq3-lloyd-single-acc-gfx1100` (one commit, mechanical
5-file edit pattern).

## Repro

```sh
# Per-call multi-acc-vs-CPU drift (real Qwen3.5-9B weights):
./target/release/examples/diag_mq4_lloyd_multiacc \
  ~/.hipfire/models/qwen3.5-9b.mq4-lloyd \
  model.language_model.layers.0.mlp.down_proj.weight --rows 1024
./target/release/examples/diag_mq3_lloyd_multiacc \
  ~/.hipfire/models/qwen3.5-9b.mq3-lloyd \
  model.language_model.layers.0.mlp.down_proj.weight --rows 1024

# Full-coverage PPL drift reproduction:
#  - swap kernels/src/{gemv,residual,fused_gate_up,fused_qkv,fused_qkvza}_
#    mq4g256_lloyd.gfx1100.hip to multi-acc bodies (use multiacc_diag as
#    template; replace `float acc = ...` with 4 acc registers, change
#    DOG4_LDS to (a) += form, merge as `(acc0+acc1)+(acc2+acc3)` before
#    wave reduction)
#  - rebuild perplexity, run with --ctx 2048 --warmup 8 --offset 0 against
#    qwen3.5-9b.mq4-lloyd → drift surfaces in NLL/tok (0.0166 above slow).
#  - restoring single-acc returns to byte-equal.
```

## Bench-host quirks worth noting

- The Lloyd-MQ4 fast kernel produces SIGSEGV during ROCm teardown on
  gfx1151 after metrics print (same as MQ3-Lloyd). Doesn't affect
  bench numbers — they print before teardown — but causes exit code
  139 on otherwise-successful runs.
- gfx1151 sees ~3.4× speedup from multi-acc fast vs slow generic on
  Qwen3.5-9B; calibrated gfx1100 is expected to be higher.
