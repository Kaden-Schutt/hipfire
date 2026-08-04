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

**G3 — dense/attention floor may be set too low.**
Escha holds *all* dense layers at int8 and calls it lossless vs fp16; the +0.7 GB
is consistent with that. Our mq4p tier already does Q8-protected attention, so
the mechanism exists — the MQ2 SKU's tensor recipe needs auditing to see what it
actually drops.

## 5. Spec — `qwen3.6:35b-a3b-mq2r` (MQ2-Lloyd-Redline)

Goal: promote MQ2-Lloyd from research-gated floor SKU to a shippable Redline
(speed/size) tier at ≤12.5 GB, with quality good enough to drop the "degraded"
label. Per the SKU naming rule the tier suffix stays short — `mq2r`; composition
lives in the registry description. `deepseek-v4-flash:mq2r` already establishes
the `mq2r` precedent (MFP4-E8 dense + MQ2-Lloyd routed experts).

Phased, cheapest-first, each phase independently falsifiable:

**Phase 0 — measure the actual baseline (blocking, ~half a day).**
Nobody has a KLD number for `qwen3.6-35b-a3b.mq2`; "degraded" is prose. Run the
llama.cpp bf16 oracle (per the established KLD methodology) against the existing
MQ2 SKU, and against mq4r/mq4p for reference. **If MQ2 KLD is already
competitive, Phases 1–3 are unnecessary and this becomes a labeling +
registry-description task.** Do not skip this — it is the cheapest step and it
can close the whole project.

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

**Phase 3 — tensor-recipe audit (G3).**
Enumerate what the MQ2 recipe drops below int8 on the dense/attention side and
price restoring it. Budget: we have 0.7 GB of headroom against Escha's 12.3 GB
and still land under a 14 GB min-VRAM target.

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

## 6. Open item

The contributor's HIP port is unreviewed by us and stays out of the tree (§0).
If it is kept for reference it belongs in `.competitors/` (gitignored). No
hipfire commit should reference its contents.
