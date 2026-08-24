# SP-E — runtime-free low-bit PTQ of Qwen3.6-27B vs PrismML Bonsai

## Correction — 2026-08-18 (current acceptance)

**Status:** This is the only current quantitative acceptance for the Bonsai
arms in this directory. The 2026-08-17 tables and every cross-arm comparison
built from them are **withdrawn** (see historical sections below).

### Why the original numbers are invalid

The 2026-08-17 SP-E run scored **candidates under asym3 KV** against a
**teacher / reference produced with f32 KV**. That KV-mode mismatch confounds
absolute KLD and every cross-arm ranking that used those values. It is not a
small bias that can be renormalized away: the reported 0.536 / 0.629 class
figures are not accepted quality evidence.

**Scope of this correction:** only the PrismML Bonsai TQ2 (ternary) and BQ1
(binary) arms were re-scored under a matched all-f32 fixture. **No claim is
made that the old SP-E PTQ arms (tq2-sweep, tq2-awqim, bq1-*, mq2-*, mq3-control)
were re-run in f32.** Every comparison that involves those old arms remains
**withdrawn pending a complete all-f32 rerun**.

### Corrected fixture (Bonsai only)

| Field | Value |
|---|---|
| Date | 2026-08-18 |
| Host | gfx1151 |
| Teacher | canonical native MQ4 Qwen3.6-27B |
| Teacher SHA-256 | `86a5f80fd29d545abb1093dead242725ced6d68b8607c6d566d897b1a82442dc` |
| Slice | wikitext2-1024s-2048ctx, 8 chunks × n_ctx 512 |
| Scored tokens | 2040 / arm |
| Top-k | 256 |
| KV | **f32 for teacher and both candidates** |
| Scoring | per-token (deliberate; see note below) |
| TQ2 artifact SHA-256 | `b2ea3f7f2160fe961d5ed256f474a4428b0f88e3a923e70fa80d3bf3bde5098e` |
| BQ1 artifact SHA-256 | `83dc56409efa0503e171d91fffcad25c8a9f000120f61cf9cbba589a09fbbb07` |

Checked-in per-sequence evidence for this fixture:

- `per-seq/bonsai-ternary__qwen36-27b__per-token.kldseq`
- `per-seq/bonsai-binary__qwen36-27b__per-token.kldseq`

### Corrected native f32-KV Bonsai scores

| Variant | bits | Mean KLD | 95% CI | p99 | PPL |
|---|---|---:|---|---:|---:|
| bonsai-ternary (TQ2 / PrismML) | 2.125 | **0.504695** | 0.4098–0.6140 | 5.874 | 16.2301 |
| bonsai-binary (BQ1 / PrismML) | 1.14 | **0.620867** | 0.5379–0.7167 | 7.237 | 17.5932 |

Native teacher oracle on this fixture: mean NLL **2.260262**, PPL **9.5856**.

Do **not** cite 0.536 / 0.629 (or any 2026-08-17 cross-arm delta derived from
them) as accepted values. Those figures belong only to the withdrawn historical
record below.

### Generation coherence (serve batteries)

Both Bonsai arms passed the 5-genre serve battery with coherent output on
**gfx1100**, **gfx1030**, and **gfx1201** (in addition to the earlier gfx1151 checks).

### Scoring mode / batchability note

TQ2G128 and BQ1G128 **are** in `is_batchable_la` now (formats are batchable).
This corrected fixture still used **per-token** scoring deliberately, for
alignment with the original SP-E protocol and with the port-fidelity arm — not
because prefill is unavailable.

---

## Historical record — 2026-08-17 (WITHDRAWN — non-acceptance)

> **WITHDRAWN / NON-ACCEPTANCE.** Everything from here through the end of
> **Findings (historical)** was produced under candidate-asym3 vs teacher-f32
> confounding. Tables, codebook rankings, width-sweep orderings, and narrative
> conclusions that depend on those KLDs are **historical interpretation only**.
> They must not be read as current acceptance evidence. The independently valid
> **Port fidelity** section is carved out and remains accepted.

**Original run metadata (historical):** Date 2026-08-17 · Host gfx1151 ·
Slice wikitext2-1024s-2048ctx, 8 chunks × n_ctx 512 (2040 scored tokens/arm) ·
**KV: asym3 on candidates** · Scoring: per-token · Teacher / reference:
`qwen3.6-27b.mq4` via `build_kld_ref_native` (top-256). KLD was distance from
the mq4 teacher, not from FP16 — but the KV mismatch voids absolute and
cross-arm use of the numbers.

### Results (historical, withdrawn)

| Variant | bits | Mean KLD ± 95% CI | PPL | Generates? |
|---|---|---|---:|---|
| bonsai-ternary (PrismML) | 2.125 | **0.5363** (0.4223–0.6715) — *withdrawn* | 16.69 | yes — coherent reasoning → "Paris" |
| bonsai-binary (PrismML) | 1.14 | **0.6292** (0.5476–0.7226) — *withdrawn* | 17.76 | yes (per SP-B) |
| spe-tq2-awqim (ours + AWQ imatrix) | 2.125 | 2.2418 (2.0376–2.3978) — *withdrawn* | 86.57 | **no** — emits `<think>` then EOS |
| spe-tq2-sweep (ours, uniform) | 2.125 | 5.1007 (4.9463–5.2548) — *withdrawn* | 1436.38 | **no** — multilingual token soup |
| spe-bq1-sweep (ours, uniform) | 1.14 | 7.9984 (7.7765–8.2272) — *withdrawn* | 30183.40 | no |
| spe-bq1-awqim (ours + AWQ imatrix) | 1.14 | 8.4162 (8.0920–8.7121) — *withdrawn* | 45482.57 | no |

Reference points recorded with the withdrawn run: the mq4 teacher scored
NLL ≈ 2.00 on that slice; uniform over the 248320-token vocab is NLL 11.9 /
KLD ≈ 12. Teacher oracle under the corrected f32 fixture is NLL 2.260262 /
PPL 9.5856 (see correction above).

### Codebook controls (historical, withdrawn)

Added after the first pass of the withdrawn run. Same source, same
asym3-confounded teacher comparison, same slice; only the **codebook and
rotation** varied at a fixed ~2 bpw. **All KLD/PPL cells and the ranking
narrative are withdrawn** until the SP-E arms are re-scored all-f32.

| target | bpw | codebook | rotated | KLD | PPL |
|---|---|---|---|---:|---:|
| spe-tq2-sweep | 2.125 | uniform 3-level | no | 5.1007 *withdrawn* | 1436.38 |
| spe-mq2-uniform | 2.25 | uniform 4-level | yes | 3.9225 *withdrawn* | 472.05 |
| spe-tq2-awqim | 2.125 | uniform 3-level + imatrix | no | 2.2418 *withdrawn* | 86.57 |
| spe-mq2-lloyd | 2.25 | Lloyd-Max non-uniform | yes | 0.6125 *withdrawn* | 17.04 |
| bonsai-ternary (PrismML) | 2.125 | uniform 3-level + *their transform* | no | 0.5363 *withdrawn* | 16.69 |

Historical note (not current acceptance): `spe-mq2-lloyd` generated coherently
under the original serve check (clean reasoning trace → "Paris"), unlike either
withdrawn ternary PTQ arm. That generation observation is separate from the
voided KLD ranking; it does **not** restore the 0.61-vs-0.54 "within noise of
Bonsai" claim, which depended on confounded absolute KLDs.

The original write-up argued that a plain PTQ with a non-uniform per-block
codebook landed within noise of PrismML's proprietary transform, that ~2 bpw
was not the problem and the fixed uniform level set was, and that rotation alone
was worth little versus the non-uniform codebook. **Those comparative claims are
withdrawn** with the KLDs they rest on. They remain here only so the withdrawn
interpretation is auditable.

### Width sweep (historical, withdrawn)

| target | bpw | codebook | KLD | PPL |
|---|---|---|---:|---:|
| mq4 (the teacher itself) | 4.25 | uniform 16-level | 0.0000 *withdrawn context* | 7.42 |
| spe-mq3-control | 3.25 | uniform 8-level | 0.2767 *withdrawn* | 13.02 |
| bonsai-ternary (PrismML) | 2.125 | uniform 3-level + transform | 0.5363 *withdrawn* | 16.69 |
| spe-mq2-lloyd | 2.25 | Lloyd non-uniform 4-level | 0.6125 *withdrawn* | 17.04 |
| spe-tq2-awqim | 2.125 | uniform 3-level + imatrix | 2.2418 *withdrawn* | 86.57 |
| spe-mq2-uniform | 2.25 | uniform 4-level | 3.9225 *withdrawn* | 472.05 |
| spe-tq2-sweep | 2.125 | uniform 3-level | 5.1007 *withdrawn* | 1436.38 |

Historical narrative (withdrawn): a uniform codebook looked fine at 8 levels
and fell apart at 4 and 3; MQ3 looked like the best plain-PTQ quality/bit trade
and MQ2-Lloyd the working floor; PrismML ternary sat between them. **None of
that ordering is current acceptance evidence.** Practical size notes (artifact
GB, embed precision policy) are engineering context, not a restored quality
ranking.

### Findings (historical, withdrawn)

The following findings were written against the asym3-confounded KLDs. Keep
them only as the withdrawn interpretation of that run.

**1. AWQ sidecars as imatrix (mechanism note; magnitude withdrawn).**
`compute_awq_scales` emits `s = C·in_sum2^(α/2)`, so `in_sum2 ∝ s^(2/α)` and the
per-tensor constant cancels inside the packers' per-block argmin. The withdrawn
run reported ternary moving 5.10 → 2.24 under that weighting. The mechanism may
still be real; the **reported magnitudes and "2.3×" claim are not acceptance
evidence** until remeasured all-f32.

**2. At 1 bit the same imatrix slightly hurt in the withdrawn run**
(7.998 → 8.416). Same caveat: directionally interesting, not accepted.

**3. Mid-range KLD ≠ working model (still good practice).** Both ternary PTQ
arms failed generation under the same command that produced a coherent Bonsai
trace. Pairing KLD with a generation smoke remains required practice; it does
not rehabilitate the withdrawn absolute KLDs.

**4. Wire-format fixed level set vs PrismML transform (comparative claim
withdrawn).** The codebook-control story that `mq2lloyd` reached KLD 0.612 vs
Bonsai 0.536 with no transform, and that Bonsai's remaining edge was "0.08 nats
inside n=8 CI", is **void** under the KV confound. Bonsai's *1-bit* usability
gap vs our 1-bit PTQ arms was the more striking qualitative observation in that
write-up; qualitative serve failure of the PTQ 1-bit arms is not itself a KLD
claim, but any numeric gap cited from the withdrawn table is non-acceptance.

---

## Port fidelity — hipfire vs llama.cpp on byte-identical weights

**(Accepted — independent of the withdrawn SP-E absolute/cross-arm KLDs.)**

The SP-E arms above (historical) were scored against a hipfire-native teacher,
which measures model-vs-model distance, NOT whether our port of Bonsai is
faithful. That needed its own check: score our byte-verbatim
`ternary-bonsai-27b.hfq` against **PrismML's own llama.cpp running the same
GGUF**, on llama's tokens, **f32 KV on both sides**.

```
build_kld_ref --bf16-gguf Ternary-Bonsai-27B-Q2_0.gguf --n-ctx 512 --top-k 256 \
    --llama-perplexity-bin /data/prism-ref/build/bin/llama-perplexity
eval_hipfire --model ternary-bonsai-27b.hfq --ref bonsai-llamacpp.kldref.bin \
    --kv-mode f32 --scoring-mode per-token --max-chunks 4
```

Evidence file: `per-seq/bonsai-hipfire-vs-llamacpp__qwen36-27b__per-token.kldseq`.

**Result: mean KLD = 0.000153** (per-chunk 0.000176 / 0.000139 / 0.000191 /
0.000107; p99 ≈ 1.5e-3). Corroborated on the realized-token metric: llama.cpp's
own cumulative PPL after 4 chunks is **14.1907**, hipfire scores **14.1456** on
the same tokens — 0.3% apart.

So the ternary port is exact to numerical noise (GPU vs CPU reduction order,
fp16 storage). This also validates the whole chain independently of the
hipfire-native teacher: convert, loader, TQ2G128 GEMV, ternary lm_head,
DeltaNet forward and scoring window all agree with the reference implementation.

This port-fidelity result was already f32-matched and is **not** withdrawn by
the 2026-08-18 correction.

### This contradicts the recorded cross-engine floor

`build_kld_ref_native`'s header and
`docs/plans/2026-06-02-hfqv2-implementation/experiments/self-sufficient-eval/F2-native-eval.md`
document a "~0.30-0.36 nat" hipfire-vs-llama.cpp floor (F1 measured 0.357 nats,
87% top-1, top-256 logit cosine 0.854) and attribute it to "the two different
Gated-DeltaNet ports". That premise is why the native reference tool exists.

It does not reproduce here: **0.000153 vs a claimed 0.35 floor**, on a hybrid
model whose layers are mostly linear-attn, with byte-identical weights on both
sides — a strictly tighter control than F1's (which compared hipfire-F32
against llama-bf16).

Most likely explanation: the floor was a real hipfire bug that has since been
fixed — SP-A (2026-07-15) found and fixed exactly this class of defect, a
double-applied RMSNorm `+1` bias on qwen3.5/3.6 norms (`4af90702`), which
postdates the F1/F2 measurement. Worth re-running F1 before continuing to treat
llama-sourced references as structurally unusable; if this holds generally, the
cross-harness confound that motivated F2 is gone, and llama.cpp tooling
(imatrix, perplexity) becomes directly usable again.

## Provenance / validity

The harness was validated before any of these numbers were believed:

- **Identity control:** scoring `qwen3.6-27b.mq4` against its OWN reference
  (per-token, kv f32) gives `slice-mean KLD = 0.000000` exactly.
- **Reference soundness:** the oracle's own NLL recovered offline from the
  .kldref = 1.815 with a 96.9% top-256 hit rate and 2.1% residual mass (figures
  from the 2026-08-17 harness check). Corrected-fixture teacher oracle:
  mean NLL 2.260262, PPL 9.5856.
- **Scoring mode:** prefill and per-token agreed bit-for-bit (0.608557 both) on
  the original harness check. The 2026-08-17 text claimed TQ2G128/BQ1G128 were
  not in `is_batchable_la` and that prefill took an exact per-token fallback.
  **That batchability claim is stale:** the formats are batchable now. The
  corrected 2026-08-18 Bonsai fixture still chose per-token deliberately for
  protocol alignment, not because batch prefill is unavailable.

The superseded 2026-07-16 canary (bonsai-ternary KLD 6.15, spe-tq2-r0 KLD
13.30) remains **void**: its Bonsai arm scored a .hfq built before that day's
norm-bias fix, and its PTQ arm predates the AWQ-fold, scale-sweep and
code-3 fixes. `.hfq` files now carry a `hipfire_provenance` stamp and
`spe_ablation.sh` prints it for every arm before scoring, so this class of
error is visible rather than silent.

The 2026-08-17 SP-E absolute/cross-arm table is a **second, separate
withdrawal**: valid harness identity controls do not salvage candidate-asym3 vs
teacher-f32 KLD.

## Reproduce

Corrected Bonsai acceptance (f32 KV, per-token, hashes above) is evidenced by:

- `per-seq/bonsai-ternary__qwen36-27b__per-token.kldseq`
- `per-seq/bonsai-binary__qwen36-27b__per-token.kldseq`

Full SP-E ablation (still produces the historical arm set; treat its KLD table
as non-acceptance until an all-f32 rerun lands):

```
benchmarks/quality-baselines/harness/spe_ablation.sh full 8
```

Arm models are built with:

```
hipfire-quantize --input ~/.hipfire/models/qwen3.6-27b.mq4 \
    --output /data/hipfire-models/spe-tq2-awqim.hfq \
    --format ternary --awq-imatrix 0.55 --allow-lowbit-ptq
```

`--allow-lowbit-ptq` is required as of the original run: requantizing an
ordinary checkpoint down to ternary/binary is gated by default
(`lowbit_ptq_gate`), matching how `mq2` and `mq2-lloyd` are handled. The GGUF
byte-verbatim passthrough path used for Bonsai is unaffected.
