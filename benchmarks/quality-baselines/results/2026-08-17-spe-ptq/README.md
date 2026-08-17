# SP-E — runtime-free low-bit PTQ of Qwen3.6-27B vs PrismML Bonsai

**Date:** 2026-08-17 · **Host:** gfx1151 · **Slice:** wikitext2-1024s-2048ctx,
8 chunks × n_ctx 512 (2040 scored tokens/arm) · **KV:** asym3 ·
**Scoring:** per-token

**Teacher / reference:** `qwen3.6-27b.mq4` via `build_kld_ref_native`
(top-256, 32 chunks). Every arm is scored against this one reference and its
token stream. KLD is therefore **distance from the mq4 teacher**, not from
FP16 — an arm cannot score below the teacher's own quantization error.

## Results

| Variant | bits | Mean KLD ± 95% CI | PPL | Generates? |
|---|---|---|---:|---|
| bonsai-ternary (PrismML) | 2.125 | **0.5363** (0.4223–0.6715) | 16.69 | yes — coherent reasoning → "Paris" |
| bonsai-binary (PrismML) | 1.14 | **0.6292** (0.5476–0.7226) | 17.76 | yes (per SP-B) |
| spe-tq2-awqim (ours + AWQ imatrix) | 2.125 | 2.2418 (2.0376–2.3978) | 86.57 | **no** — emits `<think>` then EOS |
| spe-tq2-sweep (ours, uniform) | 2.125 | 5.1007 (4.9463–5.2548) | 1436.38 | **no** — multilingual token soup |
| spe-bq1-sweep (ours, uniform) | 1.14 | 7.9984 (7.7765–8.2272) | 30183.40 | no |
| spe-bq1-awqim (ours + AWQ imatrix) | 1.14 | 8.4162 (8.0920–8.7121) | 45482.57 | no |

Reference points: the mq4 teacher scores NLL ≈ 2.00 on this slice; uniform over
the 248320-token vocab is NLL 11.9 / KLD ≈ 12.

## Codebook controls — the decisive comparison

Added after the first pass, because the first pass drew the wrong conclusion.
Same source, same teacher, same slice; only the **codebook and rotation** vary
at a fixed ~2 bpw:

| target | bpw | codebook | rotated | KLD | PPL |
|---|---|---|---|---:|---:|
| spe-tq2-sweep | 2.125 | uniform 3-level | no | 5.1007 | 1436.38 |
| spe-mq2-uniform | 2.25 | uniform 4-level | yes | 3.9225 | 472.05 |
| spe-tq2-awqim | 2.125 | uniform 3-level + imatrix | no | 2.2418 | 86.57 |
| **spe-mq2-lloyd** | **2.25** | **Lloyd-Max non-uniform** | **yes** | **0.6125** | **17.04** |
| bonsai-ternary (PrismML) | 2.125 | uniform 3-level + *their transform* | no | 0.5363 | 16.69 |

**A plain PTQ with a non-uniform per-block codebook lands within noise of
PrismML's proprietary transform** (0.61 vs 0.54). No transform, no calibration
corpus, no GPU pass — just Lloyd-Max centroids plus the FWHT rotation that
hipfire's MQ formats already carry.

This overturns the first pass's reading. **~2 bpw is not the problem; the fixed
uniform level set is.** The Q2_0/Q1_0 wire format pins levels to `{-d, 0, +d}`,
leaving the encoder only `d` to choose — and no amount of scale search
(5.10) or importance weighting (2.24) recovers what a free codebook gets
(0.61). Rotation alone is worth little (5.10 → 3.92); the non-uniform codebook
is what matters.

It also puts a bound on how much of Bonsai's advantage is the proprietary
transform: on this evidence, most of it is reachable with a better codebook at
the same bit budget. Note the in-tree gate for `mq2-lloyd` records "still
collapse (9B ppl 2163)" — that verdict does not reproduce here at 27B, so it is
either model-size-specific or stale.

## Findings

**1. The AWQ sidecars are a usable imatrix, and worth 2.3× at 2 bits.**
`compute_awq_scales` emits `s = C·in_sum2^(α/2)`, so `in_sum2 ∝ s^(2/α)` and the
per-tensor constant cancels inside the packers' per-block argmin. Using it as
column importance moved ternary from KLD 5.10 → **2.24** with no GPU pass, no
llama.cpp, and no tensor-name mapping. (α is assumed 0.55 — it is not recorded
in any .hfq; a wrong α re-sharpens the weighting but preserves channel
ordering.)

**2. At 1 bit the same imatrix slightly HURTS** (7.998 → 8.416). Consistent
with the mechanism: with no zero level, column weighting can only move the
single magnitude `d`, whereas at 2 bits it decides *which columns get zeroed* —
which is where the leverage is.

**3. KLD 2.24 is still not a working model.** Both ternary arms fail
generation under the same command that produces a full coherent trace from
Bonsai. This is the headline caveat: a mid-range KLD does not imply usability,
and no arm here should be reported as "close to Bonsai" on the strength of the
number alone. **Always pair the KLD with a generation smoke.**

**4. PrismML's transform is doing the essential work.** Their *1-bit* model
(0.629) beats our best *2-bit* PTQ (2.242) by 3.6× while using half the bits,
and theirs generate while ours do not. The 27B whitepaper's claim that the
behaviour-preserving transform — not the packing — is the value is consistent
with everything measured here. SP-E's original premise (reproduce the transform)
is not reachable by scale-and-importance tuning alone.

## Provenance / validity

The harness was validated before any of these numbers were believed:

- **Identity control:** scoring `qwen3.6-27b.mq4` against its OWN reference
  (per-token, kv f32) gives `slice-mean KLD = 0.000000` exactly.
- **Reference soundness:** the oracle's own NLL recovered offline from the
  .kldref = 1.815 with a 96.9% top-256 hit rate and 2.1% residual mass.
- **Scoring mode:** prefill and per-token agree bit-for-bit (0.608557 both) —
  TQ2G128/BQ1G128 are not in `is_batchable_la`, so prefill takes an exact
  per-token fallback.

The superseded 2026-07-16 canary (bonsai-ternary KLD 6.15, spe-tq2-r0 KLD
13.30) is **void**: its Bonsai arm scored a .hfq built before that day's
norm-bias fix, and its PTQ arm predates the AWQ-fold, scale-sweep and
code-3 fixes. `.hfq` files now carry a `hipfire_provenance` stamp and
`spe_ablation.sh` prints it for every arm before scoring, so this class of
error is visible rather than silent.

## Reproduce

```
benchmarks/quality-baselines/harness/spe_ablation.sh full 8
```

Arm models are built with:

```
hipfire-quantize --input ~/.hipfire/models/qwen3.6-27b.mq4 \
    --output /data/hipfire-models/spe-tq2-awqim.hfq \
    --format ternary --awq-imatrix 0.55 --allow-lowbit-ptq
```

`--allow-lowbit-ptq` is required as of this run: on the strength of these
numbers, requantizing an ordinary checkpoint down to ternary/binary is now
gated by default (`lowbit_ptq_gate`), matching how `mq2` and `mq2-lloyd` are
handled. The GGUF byte-verbatim passthrough path used for Bonsai is
unaffected.
