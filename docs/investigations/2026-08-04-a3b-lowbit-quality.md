# Qwen3.6-35B-A3B low-bit quality — external reference point + MQ2-Lloyd-Redline spec

Date: 2026-08-04
Branch: `research/escha-w2-mining` (off `master` @ 5d3683a78)
Sources: public HF model cards (EschaLabs org, `Qwen3.6-35B-A3B-Escha-W2`,
`escha-runtime-qwen3moe`) + a contributor-supplied README describing a HIP port
of 8 PTX kernels extracted from their shipped CUDA cubins.

---

## 0. Provenance ruling — read first

A contributor ran `cuobjdump -ptx` over EschaLabs' shipped CUDA cubins,
recovered 8 kernels (`escham_*`), and hand-ported them to HIP. Their repo is
Apache-2.0, so that port is *legally* redistributable with attribution and a
NOTICE entry.

**Do not bring those `.hip` files into the hipfire tree, or into a hipfire
worktree.** Reasons, in order of weight:

1. **It buys nothing.** Every mechanism those 8 kernels implement already
   exists in `kernels/src/` (§3 table). The MQ2-Lloyd GEMV carries a
   `2026-05-18` provenance comment; the Escha repos are 6 days old.
2. **It costs the independent-invention posture.** hipfire currently has a
   clean story: FWHT-rotated MagnumQuant tiers built from published technique
   (Lloyd–Max quantization; Hadamard incoherence processing à la
   QuIP#/QTIP/QuaRot). Ingesting decompilation-derived, competitor-attributed
   source replaces that with an attribution trail, for zero capability gain.
3. **Existing hygiene rule already covers it** — never run a competitor engine,
   never import their source, lift arch *mechanisms* and drop tricks, stay
   mqN-only. Prior mining artifacts live in `.competitors/` (gitignored); if
   the port is retained at all, that is where it goes.

Reading the contributor's *README prose* for mechanism confirmation — which is
what produced §2 below — is within the rule. Reading or adapting the ported
kernel bodies is not. Nothing in this document is derived from their source.

Also note their runtime is **CUDA-only** (SGLang/Python + ZML/Zig, compute
capability 8.0–12.0, Linux x86-64). There is no HIP/ROCm path to lift even if
we wanted one, and per hygiene we do not run it.

## 1. What EschaLabs actually shipped

| | |
|---|---|
| Base | Qwen3.6-35B-A3B (256-expert MoE) — same base as our `qwen3.6:35b-a3b-*` SKUs |
| Format | `eschamoe`, mixed 2/3-bit per projection |
| gate_up | 2-bit ("code rate K=2") |
| down | 3-bit ("code rate K=3") |
| Dense + attention | int8, claimed lossless vs fp16; toggleable (`INT8=on` single-user, `off` for batched) |
| Rotation | trained scales folded into `escha_rin` / `escha_rout` at export |
| Size | **12.3 GB** |
| Quality claim | 76.06% vs FP8 75.10% on Commonsense-6 → **101.3% retention**; 100.2% on a 6-axis capability mean |
| Perf claim | 4090: 225 tok/s single-user, 1321 tok/s @ batch 32. 5090: 283 / ~2670 @ b32. 5060 Ti: 128 / 387 @ b16 |
| License | Apache-2.0 |

Caveats on their numbers, unverified by us: >100% retention on Commonsense-6 is
within the noise of that benchmark family and is not evidence of a lossless
quant; the HF "7B parameters" badge is HF's packed-int counter misreading the
safetensors, not a real parameter count. Their perf figures are NVIDIA-only and
not comparable to any hipfire measurement.

## 2. Mechanism decode (from the kernel README prose)

The 8 kernels: `had_in`, `had_epilogue`, `moe_had_in`, `moe_epi`,
`moe_epi_scatter`, `moe_scatter_combine`, `moe_build_chunks`, `moe_epi_swiglu`.

**Rotation is a normalized Hadamard of size 128, computed in-register.**
Derivable from the stated constants without touching their code:

- Thread map `col = (bid_x << 7) | (tid << 2)` with `maxntid 32,1,1` → one warp
  (32 lanes) × 4 elements = **128 columns per block**.
- "2×2 add/sub tree on 4 products" (2 in-register stages) + butterfly masks
  {1,2,4,8,16} (5 shuffle stages) = 7 stages = log2(128).
- Scale constant `sqrt(2)/16 ≈ 0.0884` = **1/sqrt(128)** — the orthonormal
  H₁₂₈ normalization.

So: H₁₂₈ via `shfl.bfly.b32`, never materialized. `escha_rin`/`escha_rout` are
the learned diagonal scalings on either side of it.

**The rotation is fused, not a separate pass.** `moe_had_in` does the input
rotation *with expert pointer-table indirection* (rotating the per-expert
activation gather), and `moe_epi_swiglu` folds the output rotation into SwiGLU
(with an `ex2.approx`-based sigmoid). The rotation rides inside kernels already
touching the data, so it costs close to nothing.

**Load-balanced expert chunking.** `moe_build_chunks` uses shared-memory prefix
sum + atomic counters to build chunks; `moe_scatter_combine` does table-based
routing with per-block-row gate multiplication. This is a *batching* mechanism —
it is what their batch-32 numbers ride on, and it is orthogonal to the quant
format.

## 3. Side-by-side: hipfire already has the quant stack

| Escha mechanism | hipfire equivalent | Status |
|---|---|---|
| H₁₂₈ incoherence rotation on weights | seeded FWHT-256, baked into weights; kernel rotates *x* instead of inverse-rotating *W* | **present** |
| Rotation fused into preceding op | `fused_rmsnorm_mq_rotate_wavegrid.gfx1100.hip`, `fused_rmsnorm_mq_rotate_vecsum.gfx1100.hip` | **present** |
| Rotation fused into SwiGLU epilogue | `fused_silu_mul_givens_rotate.hip` | **present** |
| 2-bit codebook experts | `gemv_mq2g256_lloyd*`, `gemm_mq2g256_lloyd_moe_grouped_*` (gfx1030/1151/12) | **present** |
| gate_up=2b / down=3b grading | `gemv_mq2g256_lloyd_moe_gate_up_indexed.hip` + `gemv_mq3g256_lloyd_moe_down_indexed.hip` | **present**, same pairing |
| int8-protected attention | Q8-protected attention (the mq4p tier) | **present** |
| Expert scatter offsets | `moe_scatter_offsets_k8.hip` | partial — offsets, not load-balanced chunking |
| **Trained** rin/rout scales | `gen_fwht_signs(seed, n)` — LCG pseudo-random signs, canonical seeds 42/1042 | **absent** |
| Importance-weighted codebook fit | `fit_*_lloyd_codebook` — unweighted k-means | **absent** |

hipfire's MQ2-Lloyd GEMV header states it directly: *"X must be FWHT-pre-rotated
by the caller (MQ-family kernels rotate X once per token, then re-use the
rotated vector across all top-k experts)."* Rotate-then-VQ is already the
architecture here.

**The convergence is expected, not suspicious.** Hadamard incoherence
processing + vector/codebook quantization + per-projection bit grading is where
the published literature points for sub-3-bit MoE. Two teams landing on H-rotate
+ 2-bit codebook + graded gate_up/down for the same base model is convergent
engineering on public technique. This document does not claim otherwise, and
§0 exists to keep our side of that clean.

## 4. The real gap — quality at equal size, not kernels

The uncomfortable comparison:

| | hipfire `qwen3.6-35b-a3b.mq2` | Escha W2 |
|---|---|---|
| Size | **11.6 GB** | 12.3 GB |
| Registry description | *"Floor SKU — smallest, coherent but degraded."* | 101.3% FP8 retention (claimed) |
| Encoder status | gated behind `--allow-mq2-lloyd` ("research-only") | shipped default |

Same base model, same mechanism family, ~same footprint (+0.7 GB on their
side) — and we shipped ours labeled *degraded* behind a research flag. This is
**not a kernel gap**. Three concrete, verifiable causes, all fixable with
machinery already in the tree:

**G1 — the rotation is untrained.**
`gen_fwht_signs` (`crates/hipfire-quantize/src/main.rs:845`) is an LCG emitting
±1 from a fixed seed. That is QuIP#-style *random* incoherence. Escha states
"trained scales are already folded into `escha_rin`/`escha_rout` at export" —
i.e. learned diagonal scaling around the transform (SpinQuant/QuaRot family).
At 4 bits the difference is small; at 2 bits it is most of the quality.

**G2 — the Lloyd codebook fit is unweighted.**
`fit_mfp4_lloyd_codebook` (`main.rs:2310`) does percentile init + 8 k-means
iterations with `sums[best] += w; counts[best] += 1` — every element weighted
equally. Codebook placement should minimize *output* error, not weight-space
error, which means weighting each element by its diagonal Hessian entry. We
already produce those Hessians: `bin/collect_e8_hessian.rs` + `hessian_io.rs` +
`e8_gptq.rs`. They are simply not wired to the Lloyd path.

**G3 — dense/attention floor. CLOSED, was not the problem.**
Audited via `cargo run --example hfq_dump -p hipfire-quantize` over all 21093
tensors of `qwen3.6-35b-a3b.mq2`:

| role | n | quant |
|---|---|---|
| experts gate_up | 10240 | MQ2G256Lloyd (qt=19) |
| experts down | 10240 | **MQ2G256Lloyd (qt=19)** |
| shared_expert | 40 / 120 | Q8F16 / MQ4G256 (qt=13) |
| full_attn | 40 / 20 | Q8F16 (int8) / F16 |
| linear_attn | 180 / 90 | Q8F16 (int8) / F16 |
| router | 40 | Q8F16 (int8) |
| embed + lm_head | 2 | Q8F16 (int8) |

The dense/attention floor is **already int8** — this SKU is already the Escha
recipe. It is not uniform MQ2 (qt=18); it is MQ2-Lloyd throughout.

**G3′ (the actual third gap) — routed experts are ungraded.**
`down_proj` sits at MQ2-Lloyd, identical to `gate_up`. Escha grades down to
3-bit ("code rate K=3") and only gate_up to 2-bit. We have the graded kernel
already: `gemv_mq3g256_lloyd_moe_down_indexed.hip` (+ `_batched_k4`,
+ `gemm_mq3g256_lloyd_moe_grouped_*` for gfx1151/gfx12). Nothing to write —
re-encode `down_proj` as `MQ3G256Lloyd` (qt=20) and dispatch already resolves.

Size cost: MQ2-Lloyd is 72 B/256 weights (2.25 bpw incl. the 4×fp16 per-block
codebook); MQ3-Lloyd is 112 B/256 (3.5 bpw). Over 10240 down_proj tensors of
[2048,512] that is +1.25 bpw × 10.74e9 weights ≈ **+1.68 GB → ~13.3 GB**, which
overshoots both Escha's 12.3 GB and the ≤12.5 GB target. Two ways to pay for it,
both worth pricing before committing: drop the per-block fp16 codebook in favour
of a tensor-global or learned codebook (the per-block codebook is 0.25 bpw of
pure overhead — 0.67 GB across all experts), or grade only the layers where
`down` actually matters rather than all 40.

## 4b. The existing a3b ladder — and the hole in it

| SKU | size | experts | dense/attn |
|---|---|---|---|
| `mq2` | 11.6 GB | uniform MQ2-Lloyd (qt=19) × 10240 per projection | Q8F16 int8 |
| **— nothing here —** | **12–17 GB** | | |
| `mq3p` | 17.2 GB | MQ6 ×2040 / MQ4 ×3080 / MQ2-Lloyd ×5120, per projection | Q8F16 int8 |
| `mq4r` | 18.7 GB | graded, uniform MQ4 attn + gate-side | — |
| `mq4p` | 19.8 GB | graded (default SKU) | — |
| `mfp4` | 20.2 GB | MFP4-E8 vector quant | — |
| `mq5` / `mq6` | 23.7 / 27.7 GB | quality tiers | — |

Two things fall out of the `mq3p` dump:

**hipfire and Escha grade orthogonal axes.** `mq3p` tiers by *which expert*
(hot→MQ6, mid→MQ4, cold→MQ2-Lloyd) and gives `gate_up` and `down` **identical**
treatment — 2040/3080/5120 on both. Escha tiers by *which projection*
(gate_up=2b, down=3b) uniformly across experts. Neither is a superset of the
other; a real `mq2r` should apply **both**, using the per-expert importance
machinery we already have plus the per-projection split we do not.

**`mq3p` contains no MQ3-Lloyd.** No tensor in it is qt=20. The name denotes a
size/quality tier, not the format — its cold tier is MQ2-Lloyd and its warm
tiers are plain FWHT-rotated affine MQ4/MQ6. So `MQ3G256Lloyd`, the format that
measured **0.0184 KLD on the 27B** — better than mq3 (0.0318) and 5× better than
mq2lloyd (0.0931) — is used by **no a3b SKU at all**, despite having full kernel
coverage (`gemv_mq3g256_lloyd_moe_{gate_up,down}_indexed*`,
`gemm_mq3g256_lloyd_moe_grouped_*` on gfx1151/gfx12). That is the single most
underused asset in the tree for this problem.

**The hole is 11.6 → 17.2 GB, and Escha's 12.3 GB sits squarely in it.** That is
the `mq2r` slot: a ~12–13.5 GB SKU with meaningfully better quality than the
floor, which today has no entry.

## 5. Spec — `qwen3.6:35b-a3b-mq2r` (MQ2-Lloyd-Redline)

Goal: promote MQ2-Lloyd from research-gated floor SKU to a shippable Redline
(speed/size) tier at ≤12.5 GB, with quality good enough to drop the "degraded"
label. Per the SKU naming rule the tier suffix stays short — `mq2r`; composition
lives in the registry description. `deepseek-v4-flash:mq2r` already establishes
the `mq2r` precedent (MFP4-E8 dense + MQ2-Lloyd routed experts).

Phased, cheapest-first, each phase independently falsifiable:

**Phase 0 — measure the actual baseline. The reference already exists.**
`~/.hipfire/models/q36a3b-wt2-f32.kldref.bin` — valid HFKLDR v1 (size matches
spec exactly), n_ctx=512, n_chunk=32, top_k=256, n_vocab=248320, 8160 scored
tokens, dated 2026-06-14. Note it is a *small* reference: the 27B MASTER ref
(`~/.hipfire/kldref/qwen3.6-27b-MASTER-small.kldref.bin`) carries 2048×97 =
99231 scored tokens. Fine for ranking variants, noisier for tight deltas — if a
Phase-1/2 delta lands inside the noise, regenerate a larger a3b ref before
concluding anything.

### Phase 0 RESULT — 2026-08-04, k9lin / gfx1201

```
eval_hipfire --model qwen3.6-35b-a3b.mq2 --ref q36a3b-wt2-f32.kldref.bin \
             --scoring-mode prefill --kv-mode q8
→ slice-mean KLD = 0.238060   mean NLL = 1.827463   PPL = 6.2181
  8160 tokens scored in 199.3s; HIPFIRE_LLOYD_GFX12=1 (auto-set on gfx12)
```

**The "coherent but degraded" label is accurate. Phase 0 does not close the
project — it confirms the gap.** 0.238 is well outside the sub-0.10 bar this
repo has used for shippable quality.

Cross-model context (different arch, different model — indicative only): the
27B measured 0.0931 at the same MQ2-Lloyd format. a3b is **~2.6× worse** at
identical format and identical dense floor. That direction is plausible on
first principles — A3B activates 3B of 35B with `moe_intermediate_size=512`,
so each expert weight carries more of the signal and there is less redundancy
for a 2-bit codebook to hide in than in a 27B dense-ish MLP. It also means the
27B's numbers are an *optimistic* guide for what calibration can recover here.

Caveats on this number, both of which matter for the *next* measurement rather
than this one:
- **gfx1201, not gfx1100.** Not strictly comparable to the 27B table above.
- **Small ref** (8160 scored tokens). Adequate as a point estimate against a
  bar that far away; **not** adequate to resolve a Phase-1/2/3 delta of a few
  percent. Regenerate a larger a3b ref before measuring improvements.
- No same-arch ladder reference: `mq3p`/`mq4r`/`mq4p` all exceed the 15.92 GiB
  card and could not be run locally. Run the ladder on hipx/gfx1100 for one
  internally consistent set.

Target for `mq2r`: land meaningfully below 0.238 at ≤~12.5 GB. The 27B data
says the levers are there — grading is worth ~5× and calibration ~42% on that
model — but neither figure transfers numerically.

### Prior art — MQ2-Lloyd is already measured on the 27B sibling

`~/.hipfire/awqtest/`, 2026-05-28, gfx1100, `qwen3.6-27b.*`, prefill scoring,
q8 KV, `HIPFIRE_NORMALIZE_PROMPT=0`:

| variant | slice-mean KLD | PPL |
|---|---|---|
| mq3lloyd-awqU | 0.0124 | 3.591 |
| mq3lloyd | 0.0184 | 3.610 |
| mq3-awqU | 0.0235 | 3.633 |
| mq3 | 0.0318 | 3.654 |
| **mq2lloyd-awqU** | **0.0540** | 3.716 |
| **mq2lloyd** | **0.0931** | 3.843 |
| mq2-awqU | 0.2365 | 4.460 |
| ternary | 0.3811 | 5.122 |
| **mq2 (uniform)** | **1.0315** | 9.823 |

Three things this settles, on our own numbers:

1. **The Lloyd codebook is what makes 2-bit viable at all** — 0.093 vs 1.032 for
   uniform MQ2, an 11× gap. Escha's "code rate K" framing is the same insight.
2. **Calibration recovers ~42% of the remaining KLD at 2-bit** — AWQ on top of
   mq2lloyd takes 0.0931 → 0.0540, *on this exact format*. G1/G2 are therefore
   not speculative; the lever is already demonstrated, just never applied to the
   a3b SKU.
3. **Grading down_proj 2→3 bit is worth a lot** — mq3lloyd 0.0184 vs mq2lloyd
   0.0931 is 5×. Even applying it to only one of the two expert projections
   should move a3b materially (G3′).

Caveat: these are 27B numbers on a different architecture slice, and KLD does
not transfer across models. They establish *which levers work on this quant
family*, not what a3b's number is. Phase 0 still has to be run.

**Phase 1 — Hessian-weighted Lloyd (G2).**
Highest value per line of code. Thread the existing per-column diagonal Hessian
into `fit_*_lloyd_codebook`: `sums[best] += h*w; counts[best] += h`. Keep the
percentile init and the 8-iteration cap. Wire behind an encoder flag so both
codebooks can be produced from one calibration run and compared. Pure CPU-side
encoder work — no GPU, no kernel change, wire format unchanged, so **every
existing MQ2-Lloyd kernel consumes the output as-is**.

**Phase 2 — learned rotation scales (G1).**
Replace the fixed LCG sign vector with a trained diagonal pre/post scale around
the same FWHT-256. Wire-format impact: the scales fold into the baked weights at
export exactly as Escha describes, so again **no kernel change** — the runtime
still calls `mq_rotate_x`. Cost is a calibration/optimization loop in the
encoder. Note the existing warning in `main.rs:52-59`: AWQ-style scaling must be
applied in the **unrotated** basis before FWHT bake-in, because rotation
flattens per-channel importance. Any learned-scale work must respect that
ordering or it will silently no-op.

**Phase 3 — two-axis expert grading (G3′). Probably do this FIRST.**
Cheapest real quality lever and it needs no new code. Compose the two orthogonal
axes:

- *per-expert* (existing `mq3p` machinery): hot/mid/cold tiering
- *per-projection* (new, Escha's insight): `down` one tier above `gate_up`

Concretely, a candidate recipe using only formats with existing kernel coverage:
`down` → MQ3-Lloyd (qt=20) on hot+mid experts, MQ2-Lloyd on cold; `gate_up` →
MQ2-Lloyd throughout. That reuses `gemv_mq3g256_lloyd_moe_down_indexed.hip` and
its batched/grouped siblings, so it is a re-encode plus a recipe entry.

Budget the size against the 11.6 GB floor: full `down`→MQ3-Lloyd is +1.68 GB
(→13.3 GB); restricting it to the 5120 hot+mid experts is roughly +0.84 GB
(→~12.4 GB), which lands on Escha's number. Sweep the split rather than
guessing.

**Phase 4 (separate track, optional) — load-balanced expert chunking.**
The one mechanism genuinely worth lifting, and it is a *batching* win, not a
quant win: shared-memory prefix sum + atomic counters to build load-balanced
expert chunks, replacing naive offset-based grouping. Belongs with the batched
serve work, not with this SKU. Only pursue if batch-N throughput is a current
priority — it does nothing for batch-1 decode, which is where our decode work
has been.

### Acceptance criteria

- **Quality**: KLD vs llama.cpp bf16 oracle, measured against the Phase-0
  baseline. Target: beat the current MQ2 SKU by a margin that survives the
  oracle's own run-to-run spread; stretch target is mq4r parity.
- **Behavioral**: `scripts/serve_harness.py` — `battery` for varied prompts,
  `chain` for related turns, `session` for state/reset. Record per-turn JSON and
  decoded text. Include a **bare factual prompt** — strong/code prompts mask a
  lobotomy.
- **Perf**: fresh-process protocol per CLAUDE.md — one `--max 16` warmup per
  cell, gpu-lock coordinated, fresh process per measure, byte-identical prompt
  with md5 recorded, median of 3–5. MQ2 should be *faster* than mq4r on decode
  (fewer bytes/weight); if it is not, that is its own finding.
- **Size**: ≤12.5 GB.
- **Arch coverage**: gfx1100 (hipx) + gfx1201 (hiptrx) blocking; gfx1151
  non-blocking async. MQ2-Lloyd kernels exist for gfx1030/1151/12 — confirm the
  gfx1100 path resolves before promising it.

### What would falsify this

- Phase 0 shows MQ2 KLD is fine → the gap was never real, it was a stale
  registry description. Close and relabel.
- Phase 1 Hessian weighting moves KLD by less than oracle noise → codebook
  placement was not the binding constraint; go straight to Phase 2 or reconsider
  whether 2-bit is viable for this model at all.
- MQ2 measures *slower* than mq4r on decode → the SKU is size-only, not a
  Redline tier, and should be named/positioned accordingly.

## 5b. The "MQ2-Lloyd loses ~12% activation energy" report

**Prior art supersedes this section's first draft.** See
`docs/plans/2026-08-02-lloyd-shrinkage-gain.md` and
`docs/plans/2026-08-02-quant-rate-distortion-headroom.md` on branch
`ds4-beta-staging` (2026-08-02, status: proposed/analysis, not started). Read
those first; what follows cross-validates them and answers two of their open
questions.

**Corrected conclusion.** The shrinkage is inherent to MMSE quantization — that
part is settled and both analyses agree. But "inherent" does **not** mean
"harmless" or "leave it alone", which is what an MSE-only reading suggests. The
decisive fact is in the prior-art doc and is not visible from the codec math:
DeepSeek V4 carries `route_scale`, a routed-branch-only multiplier that has been
**silently absorbing this since May** (shipped 1.8 on `.mq2r`, 2.2 on other DS4
builds, against a checkpoint value of 1.5 that the PyTorch reference confirms is
correct at PPL 4.693). **Every other model on an MQ\*-Lloyd tier has no such
knob, so those tensors are simply short — including
`qwen3.6-35b-a3b.mq2`.** The defect is invisible where it was found and live
everywhere else.

That makes this a live candidate contributor to the 0.2381 baseline in Phase 0,
not a closed item.

Replicated `quantize_mq2g256_lloyd` exactly (LCG sign gen seeds 42/1042,
`cpu_fwht_256`, percentile init, 8 Lloyd iters with the early-break, centroid
sort + index remap, fp16 codebook rounding) and measured reconstruction energy
against the Lloyd–Max theoretical distortion for a unit-Gaussian source:

| format | levels | energy kept | measured deficit | Lloyd–Max theory | LS gain α |
|---|---|---|---|---|---|
| MQ1L | 2 | 0.6423 | 35.77% | 36.34% | 1.0003 |
| **MQ2L** | **4** | **0.8881** | **11.19%** | **11.75%** | **0.9999** |
| MQ3L | 8 | 0.9656 | 3.44% | 3.45% | 1.0000 |
| MQ4L | 16 | 0.9879 | 1.21% | 0.95% | 0.9999 |

Two conclusions, both load-bearing:

**1. The ~12% *is* the price of 2 bits.** The deficit tracks D(b) for a Gaussian
source across the whole ladder. FWHT rotation is what makes the per-block
distribution Gaussian (CLT over 256 points) — that is the *point* of incoherence
processing — so scalar Lloyd on a rotated block lands exactly on the Gaussian
rate–distortion curve. A heavy-tailed input (Gaussian × Gamma) gave 11.06%, i.e.
the result is robust to the source distribution.

**2. No rescale can reduce MSE — but MSE is the wrong acceptance metric.** The
least-squares optimal rescale is **α = ⟨x,x̂⟩/⟨x̂,x̂⟩ = 0.9999** at every bit
width (orthogonality principle: ⟨x̂, e⟩ = 0, so ‖x‖² = ‖x̂‖² + ‖e‖² and the
missing energy is entirely orthogonal to the reconstruction). Any gain
correction therefore *increases* reconstruction MSE by construction. The prior-art
doc says the same thing and draws the right conclusion from it: **do not accept
or reject a codebook gain on reconstruction MSE — accept on end-to-end PPL/KLD
against the torch teacher.** The hypothesis worth testing is that a *systematic*
magnitude bias compounds across 40+ layers in a way random error does not; DS4's
empirically-tuned `route_scale` of 1.8–2.2 against a correct 1.5 is evidence for
it, and it is the only evidence either way right now.

### Answering prior-art open question #2 — per-format constants

The prior art measured only MQ2-Lloyd and asked for the other tiers. Full ladder,
independent codepath (a faithful Python port of the Rust encoder, synthetic
post-FWHT Gaussian, 4000 blocks):

| fmt | levels | retained *r* | ‖x̂‖/‖x‖ | **1/√r** (energy-preserving) | **1/r** (doc's constant) |
|---|---|---|---|---|---|
| MQ1L | 2 | 0.6423 | 0.8014 | 1.2478 | 1.5569 |
| **MQ2L** | **4** | **0.8881** | **0.9424** | **1.0611** | **1.1260** |
| MQ3L | 8 | 0.9656 | 0.9826 | 1.0177 | 1.0356 |
| MQ4L | 16 | 0.9879 | 0.9939 | 1.0061 | 1.0122 |

MQ2L cross-validates the prior art to three digits (0.8881 vs 0.8877 retained;
1.1260 vs 1.1265). Their guess that MQ4's correction "may be negligible" is
confirmed: 1.2% energy, 0.6% amplitude.

### ⚠ Flag on the proposed constant — objective/formula mismatch

The prior-art doc proposes folding **`1/retained` = 1.1265** into the codebook
*"so the reconstruction preserves energy"*. Those are two different things.
Applying 1.1265 yields energy `(1/r)²·r = 1/r = 1.1265×` the source — it
**overshoots energy restoration by 12.65%**. The energy-preserving constant is
`1/√r = 1.0614`.

Three defensible targets, and the choice must be explicit:

| constant | value (MQ2L) | restores |
|---|---|---|
| `1` | 1.0000 | MSE-optimal (orthogonality); no change |
| `1/√r` | 1.0614 | ‖x̂‖ = ‖x‖ — energy/magnitude |
| `1/r` | 1.1265 | E[x̂·x] = E[x²] — unbiased projection onto source |

The gap between the two candidates is **6.1%**, which is ~5× the per-group
spread (1.25% sd) the doc uses to justify a global constant over a per-group
one — so this choice matters more than the thing that was already ruled out. It
also changes the DS4 attribution: with 1.0614 rather than 1.1265, `.mq2r`'s
shipped 1.2000 leaves residual **1.1306** (not 1.065) and the 2.2 build's 1.4667
leaves **1.3819** (not 1.302), which enlarges their open question #4 rather than
shrinking it.

**Corollary — energy retention is the wrong *acceptance* metric** (though it is
a fine diagnostic for deriving the constant). It is blind to *where* the error
lands. Two codebooks with identical 11.2% deficit can differ enormously in KLD
depending on whether error falls along high- or low-curvature directions. Judge
changes on KLD/PPL against a teacher — for a3b, against the Phase 0 baseline of
0.2381.

Incidentally, the 8-vs-16 Lloyd-iteration question does **not** show up here
either (11.19% vs 11.15%). Whatever caused the 2026-05-20 DeepSeek V4 60×
wikitext2 PPL blowup at 16 iterations (758 vs 12) is not an MSE effect — most
likely pathological blocks falling into a bad local minimum that block-mean
distortion does not surface. That makes the iteration-count divergence below a
genuine open defect, not a stylistic one.

**Open defect (real, live, unrelated to the energy question):**
`quantize_mq2g256_lloyd_weighted` (`main.rs:3370`) and
`quantize_mq2g256_lloyd_gptq` (`main.rs:3550`) both run `max_iter = 16`, and the
weighted arm's comment claims *"16-iter cap matches the plain Lloyd path."* The
plain path (`main.rs:3709`) is `max_iter = 8`, deliberately reverted from 16 on
2026-05-20, with an explicit *"Do NOT raise this back to 16 … without running
wikitext2 PPL on a DeepSeek V4 build first."* The weighted arm is reachable
(`main.rs:9458`). Either the revert should propagate to all three arms or the
16-iter arms need the DS4 PPL run the comment demands.

Reproduction script (synthetic, CPU-only, no GPU):
`scratchpad/mq2l_energy.py` — regenerate before trusting these numbers; it is a
faithful port of the Rust encoder but it *is* a port, not the encoder itself.
The decisive version of this test would call `quantize_mq2g256_lloyd` directly
from a Rust unit test over real a3b expert tensors.

## 5c. Their quantizer vs ours — and what our g256 codebook actually costs

### Method comparison

| | external W2 build | hipfire MQ*-Lloyd |
|---|---|---|
| rotation | **H₁₂₈**, in-register via `shfl.bfly`, never materialized | **FWHT-256**, baked into weights offline |
| per-channel scale | **learned** diagonals `rin`/`rout`, trained, folded at export | none |
| codebook | **global/shared** (inferred, see below) | **per-block**, 4×fp16 fitted by Lloyd |
| grading axis | per-projection (gate_up 2b / down 3b) | per-expert (MQ6/MQ4/MQ2L tiers) |
| metadata overhead | **≈0.03 bpw** | **0.25 bpw** |

### The size arithmetic rules out small groups

a3b geometry: gate_up 21.47B params, down 10.74B, experts 32.21B total;
embed+lm_head 1.02B; attention/linear_attn ~1.35B.

At gate_up=2b, down=3b, int8 dense, **zero** group-scale overhead → **~11.76 GB**.
They ship **12.3 GB**, leaving only ~0.5 GB for all metadata. So their overhead is
very low, which excludes small-group-with-per-group-codebook entirely:

| group | 8 B codebook per group | total | overhead on payload |
|---|---|---|---|
| g32 | 2.000 bpw | 4.000 | 100% |
| g64 | 1.000 bpw | 3.000 | 50% |
| g128 | 0.500 bpw | 2.500 | 25% |
| **g256 (ours)** | **0.250 bpw** | **2.250** | **12.5%** |
| g512 | 0.125 bpw | 2.125 | 6.2% |
| *per-channel fp16 scale* | *0.031 bpw* | *2.031* | *1.6%* |

So the g32/g64 hypothesis is backwards — at g64 a per-group codebook alone
would cost 1.0 bpw and blow the budget. Their card's *"trained scales already
folded into `escha_rin`/`escha_rout` at export"* is the actual mechanism:
**per-channel learned scales (finest granularity, ~0.031 bpw) + a shared
codebook**, rather than per-block fitted codebooks.

**The g256 choice from March is not the problem, and it is what made a per-block
codebook affordable at all.** The problem is that the group size is doing two
jobs — payload tiling *and* codebook amortization — and only the first needs
g256.

### Measured: what the per-block codebook buys (synthetic, CPU)

| variant | bpw | energy kept | NRMSE (Gauss) | NRMSE (heavy-tail) |
|---|---|---|---|---|
| **A** per-block Lloyd cb — **current** | **2.250** | 0.8875 | **0.3354** | 0.3334 |
| **B** global fitted cb + per-block fp16 scale | 2.062 | 0.8835 | 0.3415 | 0.3395 |
| **C** *textbook* Lloyd–Max Gaussian cb + per-block scale | 2.062 | 0.8855 | 0.3416 | 0.3395 |
| **D** global cb + one scale per 1024 weights | 2.016 | 0.8817 | 0.3423 | 0.3431 |

**The per-block codebook costs 0.1875 bpw (8.3% of the format) and buys 1.8%
NRMSE.** For scale: one extra *bit* takes NRMSE 0.335 → 0.186, so at that slope
0.1875 bpw spent on rate would buy ~8%. The per-block codebook returns about a
quarter of that.

And variant **C uses no fitting at all** — the textbook Lloyd–Max Gaussian levels
(±0.4528, ±1.5104) — yet matches the globally-fitted codebook exactly and lands
within 1.8% of per-block fitting. On heavy-tailed input C retains 0.8888 vs the
current path's 0.8889.

**Why: the FWHT is doing its job too well to leave the codebook anything to
learn.** Rotation Gaussianizes every 256-block by CLT, so the optimal *level
shape* is the same everywhere — the per-block fit is re-deriving the same
Gaussian-optimal shape ~4000× per tensor, differing only by scale, and a scale
is one number instead of four. This is the same conclusion the external build
reached by construction.

Bonus: a shared codebook hoists out of the GEMV inner loop entirely. The current
kernel header notes the codebook was moved "from per-thread registers to LDS so
all 8 weights per group share one `ds_read_b32`" — a global codebook removes
that per-group load.

**Caveats.** Synthetic Gaussian / Gaussian×Gamma, not real expert weights, and
my blocks are drawn i.i.d. so they understate what per-block *scale* adaptation
is worth on a real tensor — though variants B/C/D all carry a per-block scale,
so only per-block *shape* adaptation is being given up, and shape is exactly
what the rotation normalizes. Confirm on real a3b expert tensors and accept on
KLD, not NRMSE.

## 6. Open item

The contributor's HIP port is unreviewed by us and stays out of the tree (§0).
If it is kept for reference it belongs in `.competitors/` (gitignored). No
hipfire commit should reference its contents.
