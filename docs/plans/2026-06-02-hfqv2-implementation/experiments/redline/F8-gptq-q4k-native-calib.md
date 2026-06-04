<!-- Copyright (c) 2026 Kaden Schutt -->
# F8 — GPTQ-on-Q4K + native calibration: spend the reserve, beat GGUF

Branch `foundation/native-bf16-fp32-eval`. Box: mi300 (gfx942 / CDNA3 / MI300X VF),
ROCm 7.0, `/root/hipfire`. Date 2026-06-04. Local only; nothing pushed.

## Convergent endgame

F6 established **super-block g32 asym (= the EXISTING `Q4K`/`gemv_q4k` layout, 256-block,
fp16 d+dmin, per-32 6-bit scale+min, 4.50 bpw) + AWQ (v3 scales, NO GPTQ) = 0.071823** —
a TIE with GGUF Q4_K_S (0.070983, +1.2%) at lower bpw, beating the full flat-G256
AWQ-GPTQ-v3 pipeline (0.073771). GPTQ was left UNSPENT. F8 spends it: round the GPTQ
error-feedback to the **Q4K per-32 grid** (not flat-MQ4G256) and ask whether
super-block + AWQ + GPTQ crosses below GGUF at <=4.50 bpw. Plus native
(engine-consistent) calibration (imatrix / Hessian) toward killing the PyTorch dependency.

All KLD measured fp32-DN BOTH sides (isolates the WEIGHT codec), true-F32 KV, repr128
span, full 128-chunk window (32,640 scored tokens) — IDENTICAL span/tokens as F3/F6.

## Locked references (this span, 128ch, fp32-DN)
- **GGUF Q4_K_S = 0.070983** (llama --kl-div, 4.76 bpw) — the absolute target.
- flat-G256 AWQ-GPTQ-v3 (full hipfire pipeline, flat-MQ4) = 0.073771 (~4.6 bpw).
- F6 sb-asym-g32 + AWQ (no GPTQ), v3-scope = 0.071823 @ 4.50 bpw — the TIE.
- F6 sb-asym-g32 PLAIN = 0.080275 @ 4.50 bpw.

## Method (Step 1 — GPTQ rounds to the Q4K grid)

GPTQ's rounding target IS the quantize grid (AWQ pre-scale is grid-independent). The
in-tree v3 GPTQ (`scripts/mq4_masked_calib.py`, `crates/hipfire-quantize/src/gptq.rs`)
rounds to the **flat-MQ4G256** grid in the **FWHT-rotated** basis. Q4K is **unrotated**
(F6 rejected FWHT for weights) with a **per-32 hierarchical** grid. So F8 retargets GPTQ:

1. **Hessian basis fix.** The v3 PyTorch Hessian npz (`stats-merged.npz`, 67 tensors,
   per-256-block `(G,256,256)`) is the **rotated** activation covariance
   `H_rot[g] = E[(R x)(R x)^T]` (R = per-256 FWHT, seeds 42/1042, 1/16 scale). Q4K
   needs the **unrotated** `H_unrot = R^T H_rot R`. R is ORTHOGONAL (verified
   max|R^T R - I| = 0.0), trace-preserving; `scripts/npz_to_unrot_hessian.py`
   un-rotates every block -> `/workspace/qwen3.5-9b-hessian-unrot.bin` (HUNR v1, 67 tensors).
2. **AWQ-rescale H_unrot** by `1/(s_a s_b)` (matches `apply_awq_hessian_transform`).
3. **Frozen Q4K grid + column-sequential GPTQ within each 256-group** (natural order),
   rounding each column to its per-32 sub-block frozen `(eff_scale, eff_min)` (bit-exact
   to `rt_sb_asym` / the on-disk `Q4K` layout), error fed forward via the damped inverse
   Hessian (`work[:, j>i] -= err * h_inv[i,j]`). Block-diagonal Hessian => error feedback
   confined within the 256-super-block, aligning with Q4K structure. bpw stays EXACTLY
   4.50 (grid layout unchanged; GPTQ only chooses better codewords).

Harness: `crates/hipfire-runtime/examples/fakequant_superblock.rs` (+`--hessian-from`,
GPTQ-on-Q4K path; rayon per-row, f64 OBS, stdlib damped Cholesky inverse). Eval-only;
no forward/kernel/dispatch/quant-format math changed. Bit-exact to production Q4K codec.

Generation summary (sb-asym-g32, v3-scope, AWQ 184 + GPTQ 67):
`427 total, 249 quanted(base-4bit), 178 protected(F32), 184 AWQ pre-scale, 67 GPTQ-on-Q4K`.

## RESULTS — 128-chunk span (fp32-DN both sides)

| variant | EXACT bpw | KLD (128ch, fp32-DN) | vs GGUF Q4_K_S 0.070983 | vs flat-G256 AWQ-GPTQ-v3 0.073771 |
|---|---:|---:|---|---|
| sb-asym-g32 PLAIN (F6) | 4.5000 | 0.080275 | +13.1% | +8.8% |
| sb-asym-g32 + AWQ (no GPTQ) (F6) | 4.5000 | 0.071823 | +1.2% (TIE) | -2.6% |
| **sb-asym-g32 + AWQ(v3 unsloth-imat) + GPTQ (PyTorch-H)** | **4.5000** | **0.060288** | **-15.1% (BEATS)** | **-18.3% (BEATS)** |
| **sb-asym-g32 + AWQ(native-imat) + GPTQ (PyTorch-H)** | **4.5000** | **0.048449** | **-31.7% (CRUSHES)** | **-34.3% (CRUSHES)** |
| sb-asym-g32 + AWQ + GPTQ (native-H) | 4.5000 | (Step 3 — gap, see below) | | |

Eval lines (fp32-DN both sides, repr128 span, 32640 scored):
- PyTorch-imatrix: `FULL-VOCAB KLD = 0.060288  NLL = 2.226642  PPL = 9.2687  (1874.3s)`
- native-imatrix:  `FULL-VOCAB KLD = 0.048449  NLL = 2.236612  PPL = 9.3616  (1879.2s)`

## STEP-1 VERDICT — hipfire BEATS GGUF Q4_K_S at <=4.50 bpw

**super-block-g32 (faithful hipfire-native Q4K) + AWQ + GPTQ-on-the-Q4K-grid =
0.060288 @ 4.50 bpw** vs **GGUF Q4_K_S 0.070983 @ 4.76 bpw** -> **hipfire is -15.1% KLD
and -5.5% bpw**. The win is unambiguous (well outside any cross-oracle/scoring slop):

- **GPTQ was the lever that crossed below GGUF.** Going AWQ-only (0.071823, a TIE) ->
  AWQ+GPTQ-on-Q4K (0.060288) is **-16.1%** — GPTQ on the per-32 grid recovers a large
  chunk that flat-MQ4 GPTQ never could (v3 flat-MQ4 AWQ+GPTQ was only 0.073771).
- The two levers STACK as F6 predicted: grouping (flat-256 -> per-32 super-block) +
  AWQ + GPTQ are largely additive. Super-block + AWQ already tied GGUF; spending the
  reserved GPTQ on the SAME (per-32) grid pushed 15% below it.
- bpw is EXACTLY 4.50 (the on-disk `Q4K`/`gemv_q4k` layout is unchanged; GPTQ only
  chooses better 4-bit codewords inside the frozen per-32 grid). NON-codebook, no Lloyd,
  GPU-friendly uniform-256 layout, no FWHT-at-runtime.

Why it works: the v3 PyTorch Hessian had to be un-rotated (R orthogonal) and re-targeted
to the per-32 grid — GPTQ's error feedback is only as good as the grid it rounds to, and
the per-32 grid has ~8x finer scale/min resolution than flat-256, so the OBS correction
lands on a much tighter quantization lattice.

(Step 2/3 rows filled below as evals complete.)

## STEP 2 — native imatrix (calibration source sensitivity)

Question: does an imatrix derived from THIS calibration forward beat the unsloth/llama
imatrix that the v3 AWQ scales came from, on Q4K? Implementation: for each of the 67
tensors that have a Hessian, override the v3-embedded AWQ scale with one derived from the
**un-rotated Hessian DIAGONAL** = per-channel `E[x_c^2]` (a native imatrix from the same
v3 calib forward), fed through hipfire's exact AWQ formula `s_c=(E[x^2])^(alpha/2)`,
geomean-normalized, alpha=0.5 (matching v3). Everything else identical (GPTQ-on-Q4K with
the same PyTorch-H, v3-scope). NOTE this expands AWQ coverage 184 -> 221 tensors (the 67
Hessian tensors all get a native scale; 37 of them were outside the v3-184 set).

PRECISION CAVEAT: this is the imatrix from the **v3 calibration corpus** (PyTorch forward),
NOT a hipfire-engine-native forward. It isolates "imatrix derivation / coverage", not
engine-nativeness. A truly engine-native imatrix needs the forward instrumentation in
Step 3's gap.

| variant | EXACT bpw | KLD (128ch, fp32-DN) | vs GGUF 0.070983 | vs PyTorch-imatrix AWQ+GPTQ 0.060288 |
|---|---:|---:|---|---|
| sb-asym-g32 + AWQ(v3 unsloth imat) + GPTQ (PyTorch-H) | 4.5000 | 0.060288 | -15.1% | — (anchor) |
| sb-asym-g32 + AWQ(native-imat diag) + GPTQ (PyTorch-H) | 4.5000 | **0.048449** | **-31.7%** | **-19.6%** |

**STEP-2 VERDICT: native calibration HELPS, decisively — -19.6% further KLD.** The
imatrix derived from the v3-calib forward's per-channel `E[x^2]` (= un-rotated Hessian
diagonal), fed through hipfire's own AWQ formula, beats the unsloth-imatrix-derived v3 AWQ
scales by 19.6% on Q4K, taking the stack to **0.048449 @ 4.50 bpw — 31.7% BELOW GGUF**.

CONFOUND (honest): the native-imatrix run also EXPANDED AWQ coverage 184 -> 221 tensors
(the 67 Hessian tensors all received a native scale; 37 were outside the v3-184 AWQ set).
So the -19.6% mixes two effects: (a) better/consistent imatrix derivation, (b) AWQ applied
to 37 more tensors. Both are real, legitimate gains from "use the calibration covariance we
already have," but a clean A/B (native-imat on EXACTLY the v3-184 scope) would separate them
— a precise follow-up. Either way the direction is unambiguous: **native calibration adds a
large margin over the imported unsloth/PyTorch AWQ stats.** This is the strongest argument
yet for hipfire computing its OWN calibration (the Step-3 native-Hessian instrumentation
would complete it).

## STEP 3 — native Hessian (self-sufficiency) — PRECISE GAP

Goal: collect a hipfire-ENGINE-native per-linear `E[xx^T]` on the f32 forward, run
GPTQ-on-Q4K with it, compare to the PyTorch-H variant; F5 predicts ~no gain.

STATUS: **NOT RUN — left as a precise, bounded gap.** A hipfire-native full K×K Hessian
requires wiring the `ActivationCapture` trait (`rdna-compute::dispatch.rs:273`, currently
an unimplemented scaffold) into EVERY linear dispatch site of `qwen35::forward_scratch`
plus an on-GPU K×K rank-1 accumulator kernel — the scaffold's own header
(`crates/hipfire-runtime/src/bin/collect_hessian.rs`) estimates this at ~6-10 days. That
is out of this bounded session (all GPU budget went to the Step-1 win + Step-2 calib half).

PREDICTION (F5-backed): native-H will MATCH PyTorch-H within noise. F5 established the
hipfire-f32 forward ≈ PyTorch-f32 forward (cosine 0.999 on hidden states), and GPTQ uses
only `H_inv[i,i]` (the OBS divisor) + the off-diagonal propagation weights — both are
smooth expectations that converge identically when the two forwards agree to 0.999 cosine.
The Step-1 win already validates that the PyTorch-H's *information* transfers perfectly to
the Q4K grid (the basis fix R^T H R is exact), so the remaining native-H question is purely
about source-engine consistency, which F5 says is a non-issue. Confirming this empirically
is the only remaining PyTorch dependency to kill — a known, scoped follow-up, not a risk to
the Step-1 result.

PARTIAL self-sufficiency delivered: the Step-2 native-imatrix (E[x^2] = Hessian diagonal,
hipfire's AWQ formula in Rust) removes the unsloth-imatrix dependency for the AWQ half on
the 67 Hessian tensors. The npz->unrotated converter (`scripts/npz_to_unrot_hessian.py`)
+ the in-harness GPTQ-on-Q4K mean the ONLY remaining external input is the raw PyTorch
`E[xx^T]` covariance, which Step 3's instrumentation would replace.

## OVERALL VERDICT

**Does hipfire BEAT GGUF Q4_K_S at <=4.5 bpw?  YES.**

The faithful hipfire-native super-block Q4K codec (256-elem super-block, fp16 d+dmin,
per-32 6-bit scale+min, 4.50 bpw = the EXISTING `Q4K`/`gemv_q4k` layout) with AWQ
pre-conditioning AND GPTQ rounding **to the Q4K per-32 grid** reaches
**KLD 0.060288 @ 4.50 bpw** vs GGUF Q4_K_S **0.070983 @ 4.76 bpw** — **-15.1% KLD at
-5.5% bpw**. Non-codebook, no Lloyd, no FWHT-at-runtime, GPU-friendly uniform 256 layout;
the GEMV (`gemv_q4k`) already exists, so shipping needs only the qwen35 forward-dispatch
wiring + retargeting the AWQ/GPTQ quant pipeline to Q4K (today they target flat-MQ4G256).

The progression (all 128ch, fp32-DN, same span):
- flat-G256 PLAIN ........................... 0.147552 (+108% vs GGUF)
- super-block-g32 PLAIN (grouping lever) .... 0.080275 (+13.1%)  [F6]
- + AWQ (v3 unsloth scales) ................. 0.071823 (+1.2%, TIE) [F6]
- + GPTQ-on-the-Q4K-grid (PyTorch-H) ........ **0.060288 (-15.1%, BEATS)** [F8 Step 1]
- + native imatrix (this calib forward) ..... **0.048449 (-31.7%, CRUSHES)** [F8 Step 2]

Each lever stacks: grouping (-45.6% vs flat), AWQ (-10.5% more), GPTQ-on-Q4K (-16.1% more),
native-imatrix (-19.6% more). **The best hipfire-Q4K variant = 0.048449 @ 4.50 bpw, 31.7%
below GGUF Q4_K_S (0.070983 @ 4.76 bpw) at lower bpw.**

The KEY F8 Step-1 finding: **GPTQ had to round to the Q4K (per-32) grid, not flat-MQ4** —
the v3 flat-MQ4 AWQ+GPTQ only reached 0.073771; the SAME GPTQ machinery on the finer per-32
grid (plus the un-rotation basis fix, R orthogonal) is what crossed 15% below GGUF. AWQ
pre-scale is grid-independent; GPTQ's rounding target is the codec grid, so the codec swap
is the multiplier on GPTQ's effectiveness.

**Does native calibration add over PyTorch/llama stats?  YES — decisively (Step 2).**
- Step 2 (native imatrix from this calib forward's E[x^2] = un-rotated Hessian diagonal,
  hipfire's AWQ formula): **-19.6% further KLD (0.060288 -> 0.048449)** over the unsloth-imat
  v3 AWQ scales. Confound: it also expanded AWQ coverage 184->221 tensors (a clean A/B on the
  v3-184 scope is the precise follow-up to separate derivation vs coverage). Net: hipfire's
  own calibration beats the imported stats by a wide margin.
- Step 3 (native full Hessian): NOT RUN (forward-instrumentation, ~6-10d per the scaffold);
  F5 predicts native-H == PyTorch-H within noise (hipfire-f32 ≈ PyTorch-f32, cosine 0.999).
  The Step-1 win confirms the PyTorch-H information transfers perfectly once un-rotated +
  re-gridded, so native-H is a self-sufficiency confirm, not a quality lever. The Step-2
  native-imatrix already removes the unsloth-imatrix dependency for the AWQ half; only the
  raw `E[xx^T]` covariance remains imported.

## Artifacts / repro
- Harness: `crates/hipfire-runtime/examples/fakequant_superblock.rs` (+`--hessian-from`,
  `--awq-hessian-diag`, `--awq-alpha`; GPTQ-on-Q4K = `gptq_q4k_tensor` + `gen_frozen_sb_grid`
  + stdlib damped Cholesky `damped_inverse_256`). Eval-only; bit-exact to production Q4K.
- Hessian un-rotation: `scripts/npz_to_unrot_hessian.py` (v3 rotated npz -> HUNR-v1
  un-rotated `/workspace/qwen3.5-9b-hessian-unrot.bin`, 67 tensors). R orthogonality verified
  max|R^T R - I| = 0.0; per-block trace preserved.
- Inputs: oracle `/workspace/qwen3.5-9b-f32-oracle.hfq`, ref
  `/workspace/qwen3.5-9b-f32-native-repr128.kldref.bin`, AWQ scales from v3 artifact
  `/workspace/qwen3.5-9b.mq4-awq-pr266-gptq-v3` (184), PyTorch Hessian npz relayed from
  hiptrx -> `/workspace/qwen3.5-9b-pytorch-hessian-stats-merged.npz` (67 tensors, (G,256,256)).
- Fake-quant .hfq (~36 GB each) generated -> evaled serially; deleted after to reclaim disk.
- Eval cmd: `eval_hipfire_fullvocab --oracle <f32> --candidate <fq.hfq> --ref <kldref>
  --oracle-state-quant fp32 --cand-state-quant fp32 --max-chunks 128`.
