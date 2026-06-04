<!-- Copyright (c) 2026 Kaden Schutt -->
# F3 — Matched-Harness hipfire-vs-GGUF: does hipfire's 4-bit beat GGUF's?

Branch: `foundation/native-bf16-fp32-eval`. Box: mi300 (gfx942 / MI300X VF),
ROCm 7.0, `/root/hipfire`. Date: 2026-06-04.

## Method (why this is honest, and what we did NOT do)

GGUF bytes are NOT drop-in for hipfire's forward (llama.cpp's qwen35 converter
applies a V-head grouped→tiled permutation to the Gated-DeltaNet projections;
importing them is a reinterpretation, not an honest score — see
`F3-gguf-in-hipfire.md`, fully reverted). Instead: **matched-harness** — each
engine scores its OWN quant against its OWN bf16/F32 oracle on the SAME token
span. Valid because F2 proved the two oracles are equivalent to ~0.0008 nats
for a 4-bit candidate (PPLs match to +0.09%). llama.cpp is used ONLY as an
external benchmark tool (`llama-perplexity`), never coupled into hipfire crates.

- Span: representative mid-corpus window, byte offset 3,000,000, length
  1,200,000 of the canonical wikitext slice (`wikitext2-1024s-2048ctx.txt`,
  slice md5 83b0205a...). Window md5 = **4e86d460e2c2fec261b35e8d401ff49d**
  (matches `F3-gguf-in-hipfire.md`'s documented span). Reconstructed exactly.
- Ref: `/workspace/qwen3.5-9b-f32-native-repr128.kldref.bin`
  (md5 e060a9e4...), n_ctx=512, n_chunk=128, top_k=256, vocab=248320,
  32,640 scored tokens, **oracle F32 PPL = 9.3198** (mean NLL 2.232145).
- hipfire candidates score via `eval_hipfire` (--kv-mode f32, reads tokens
  FROM the ref → identical token stream). GGUF candidates score via
  `llama-perplexity` (external) on the same window.

---

## STEP 1 — GATE: tokenizer parity (PASSED)

The repr128 ref was built with hipfire's OWN BPE (`--tokenize-mode hipfire`).
To validate the matched comparison, llama.cpp's GGUF tokenizer must produce
IDENTICAL token IDs on the same window.

- Ran `llama-perplexity` on the window with `--kl-divergence-base` -> dumped
  llama's `_logits_` token stream (540 chunks total; window > 128-chunk cap).
- Compared llama's first 65,536 token IDs against the ref's 65,536 hipfire
  token IDs.

**RESULT: 65,536 / 65,536 = 100.0000% EXACT match. No divergence.**
- llama-first-65536 md5 = `9e2b60280b97c37fe45ed768eb0b9088`
- ref (hipfire) token-stream md5 = `9e2b60280b97c37fe45ed768eb0b9088` (identical)

The matched comparison is VALID on this span. (The harness's documented 45.9%
corpus-wide BPE divergence lives in special whitespace/markup regions
elsewhere; this representative prose region agrees byte-for-byte — same as the
F2 first-60KB region.)

---

## STEP 2 — GATE: harness alignment (PASSED)

The hipfire-F32 oracle PPL (9.3198 over the first 128 chunks) must match
llama-bf16 PPL on the SAME 128 chunks within ~0.2%.

Computed llama-bf16 NLL over the first 128 chunks (32,640 scored positions,
identical scored window [n_ctx/2 .. n_ctx-2]) directly from the `_logits_`
dump (exact, no re-run rounding):

- **llama bf16 PPL (first 128 chunks) = 9.3065** (mean NLL 2.230709, 32640 tok)
- hipfire F32 oracle PPL (first 128 chunks) = 9.3198 (mean NLL 2.232145)
- **Delta = +0.0133 PPL = +0.14%** — within ~0.2% tolerance. ALIGNED.

(Full-window llama bf16 PPL over all 540 chunks = 9.3677, for reference; the
128-chunk subset is the matched span.) This confirms F2's cross-oracle
equivalence (~0.0008 nats) carries to this representative span.

---

## STEP 3 — hipfire side (vs native F32 oracle ref, --kv-mode f32, per-token)

eval_hipfire reads tokens FROM the ref (identical 65,536-token stream). 32,640
scored tokens. KLD is **top-256-of-ref** (the ref stores top-256 log-probs).

| candidate | bpw | KLD (top-256) | PPL | ΔPPL (vs oracle 9.3198) |
|---|---:|---:|---:|---:|
| hipfire AWQ-GPTQ v3 | ~4.25–4.35 | 0.076936 | 9.6442 | **+0.3244** |
| hipfire Q8          | ~8.3       | 0.010736 | 9.2948 | −0.0250 |

(Q8 PPL dips just below the oracle — Q8≈oracle; the tiny negative ΔPPL is NLL
noise, consistent with KLD 0.0107 ≈ a near-perfect match.)

## STEP 4 — GGUF side (llama-perplexity, external; SAME 128-chunk span)

llama-perplexity on the window truncated to 128 chunks (`--chunks 128`), vs a
128-chunk llama-bf16 base dump (`--kl-divergence-base`, full-vocab
`--kl-divergence`). llama-bf16 base PPL on this window = **9.3065** (matches the
Step-2 computed 9.3065 exactly — same second-half scored window). KLD is
**full-vocab** (llama's native KLD).

| candidate | bpw | KLD (full-vocab) | PPL(Q) | ΔPPL (vs base 9.3065) |
|---|---:|---:|---:|---:|
| GGUF Q4_K_S | 4.76 | 0.070983 | 8.9512  | −0.3552 |
| GGUF IQ3_S  | 3.88 | 1.578275 | 36.2915 | **+26.985** |
| GGUF Q8_0   | 8.50 | 0.006168 | 9.3227  | +0.0162 |

Notes:
- Q4_K_S PPL(Q) 8.9512 is BELOW its bf16 base (9.3065) → ΔPPL negative.
  Quantization noise acts as mild regularization on the realized-token NLL on
  this small/hard slice, even though full-distribution KLD (0.071) is real and
  positive. So raw ΔPPL FLATTERS Q4_K_S here; KLD is the faithful quality axis.
- IQ3_S blows up to PPL 36 on this representative-but-hard prose slice (cf. F2:
  flat-MQ4 hit PPL 104 on the 26-chunk outlier). 3.88-bpw IQ3_S is decisively
  worse than hipfire AWQ-GPTQ (~4.3 bpw) here.
- Q8_0 ΔPPL +0.016, KLD 0.0062 — clean Q8 anchor, sane.


## STEP 4b — REFINEMENT: full-vocab KLD for the hipfire side (no GGUF, no top-K)

The hipfire KLD in Step 3 is **top-256-of-ref** (the native ref only stores the
oracle's top-256 log-probs). GGUF's `--kl-divergence` KLD is **full-vocab**. To
remove that confound, a new internal-only tool was added (NO GGUF):

- `crates/hipfire-runtime/examples/eval_hipfire_fullvocab.rs` (NEW, Kaden Schutt
  header, registered in Cargo.toml). Runs BOTH the F32 oracle AND the quant
  candidate forward over the SAME ref token stream (true FP32 KV, DeltaNet reset
  per chunk, second-half scored window) and computes the EXACT full-vocab
  KL(P_oracle ‖ P_cand) = Σ_v P_o(v)·(logP_o(v) − logP_c(v)) over all 248,320
  vocab entries. No ref top-K, no GGUF — purely hipfire oracle vs hipfire quant.
  (Read-only eval; forward path untouched → no coherence-gate trigger.)
  Self-check: the tool's candidate PPL = 9.6347 matches eval_hipfire's
  standalone AWQ-GPTQ PPL 9.6442 to within 0.1% — candidate forward is correct.

**hipfire AWQ-GPTQ full-vocab KLD = 0.089757** (32,640 scored tokens).
Top-256 under-reported it by ~17% (0.0769 → 0.0898): the tail mass beyond the
oracle's top-256 is real and the approximation flatters the candidate. The
full-vocab number is the one directly comparable to GGUF's `--kl-divergence`.

## STEP 5 — THE TABLE + VERDICT (all on the same repr128 span, 32,640 scored tok)

bpw note: GGUF bpw is llama's effective all-tensor figure. hipfire AWQ-GPTQ
file-level bpw = 5,313,750,016 B × 8 / 9.20e9 params ≈ **4.62** (F3 doc cites
~4.25 weights-only / 4.35 all incl. some excluded tensors; either way ≤ Q4_K_S).

| model | bpw | PPL | ΔPPL | KLD |
|---|---:|---:|---:|---:|
| **hipfire AWQ-GPTQ** | ~4.6 (4.25–4.35 wt) | 9.6442 | +0.3244 (vs oracle 9.3198) | **0.089757 full-vocab** (0.0769 top-256) |
| **GGUF Q4_K_S**      | 4.76               | 8.9512 | −0.3552 (vs base 9.3065)   | **0.070983 full-vocab** |
| GGUF IQ3_S           | 3.88               | 36.2915| +26.985                    | 1.578275 full-vocab |
| hipfire Q8           | ~8.3               | 9.2948 | −0.0250                    | 0.029810 full-vocab (0.0107 top-256) |
| GGUF Q8_0            | 8.50               | 9.3227 | +0.0162                    | 0.006168 full-vocab |

ΔPPL is each engine's candidate − its own bf16/F32 oracle (oracles match to
+0.14%, Step 2). KLD now full-vocab on BOTH sides for the 4-bit row.

### VERDICT (bpw-adjusted, 4-bit)

**On the clean apples-to-apples axis (full-vocab KLD), GGUF Q4_K_S BEATS hipfire
AWQ-GPTQ at 4-bit: 0.0710 vs 0.0898 nats — GGUF is ~21% lower KLD, AND it does
so at *higher* bpw (4.76 vs ~4.6).** So hipfire's 4-bit quant does NOT beat
GGUF's here; GGUF wins on quality even after the bpw handicap. This holds on
both KLD (GGUF lower) — the faithful full-distribution quality metric.

- The top-256 KLD (0.0769) made AWQ-GPTQ look near-tied with Q4_K_S (0.0710),
  but full-vocab scoring shows the true gap (0.0898 vs 0.0710). The top-K
  approximation was flattering hipfire by ~17%; the refinement was decisive.
- ΔPPL FLATTERS Q4_K_S in the opposite direction (its PPL drops BELOW its own
  bf16 base on this slice → quant-noise regularization on realized-token NLL),
  so ΔPPL is not a reliable quality axis here; KLD is. But even setting ΔPPL
  aside, KLD agrees: GGUF Q4_K_S is the better 4-bit quant on this span.
- vs IQ3_S (3.88 bpw): hipfire AWQ-GPTQ DECISIVELY beats IQ3_S (KLD 0.090 vs
  1.578; PPL 9.64 vs 36.3). At the ~3.9-bpw tier hipfire wins big — but that's
  a lower-bpw GGUF tier, not the matched 4-bit comparison.
- Q8 sanity anchor: both engines near-lossless (hipfire 0.0107 top-256 /
  0.029810 full-vocab; GGUF Q8_0 0.006168 full-vocab). Sane.

### RESIDUAL CONFOUNDS (do not overstate)
1. **Cross-oracle slop ~0.0008 nats** (F2): the two bf16/F32 oracles are not
   byte-identical (different DeltaNet/RoPE/norm ports). At the 0.071–0.090
   scale this is ~1% — it does NOT close the 0.019-nat (21%) gap. Verdict robust.
2. **Tokenizer parity = 100%** on this span (Step 1) — zero confound there.
3. **KV precision**: hipfire scored with true FP32 KV (--kv-mode f32); llama
   perplexity uses its default (F16) KV. KV precision affects attention layers
   (4/32 here are full-attn); F2 measured q8-vs-f32 KV as ~0.0008 nats on
   AWQ-GPTQ — negligible vs the 0.019 gap.
4. **top-256 vs full-vocab**: ELIMINATED for the 4-bit row (both full-vocab now).
   hipfire Q8 KLD remains top-256 in the table (full-vocab pending/anchor only).
5. **Slice**: this is ONE representative 128-chunk mid-corpus span. IQ3_S/flat
   blow-ups show low-bit quants are slice-sensitive; a different span could
   shift absolute KLDs. The 4-bit AWQ-GPTQ-vs-Q4_K_S *ordering* (GGUF lower)
   is the finding; magnitude is span-specific.

### BOTTOM LINE
At 4-bit, bpw-adjusted, on this matched honest harness: **GGUF Q4_K_S (4.76 bpw,
full-vocab KLD 0.0710) beats hipfire AWQ-GPTQ (~4.6 bpw, full-vocab KLD 0.0898)**
— GGUF is the better 4-bit quant here by ~21% KLD despite using more bits.
hipfire's AWQ-GPTQ only wins against the lower-bit IQ3_S (3.88 bpw). Gates
passed: tokenizer parity (100%), harness alignment (+0.14%).

### Q8 anchor on the full-vocab axis (corroborates the verdict direction)
On full-vocab KLD, GGUF Q8_0 (0.006168) is also lower than hipfire Q8
(0.029810). The hipfire Q8 number is higher partly from cross-oracle slop
(~0.0008) but mostly real — hipfire's Q8 path also quantizes the DeltaNet
recurrent state and uses different per-tensor granularity than GGUF Q8_0. So at
BOTH the 4-bit and 8-bit tiers, the GGUF quant sits tighter against its own
oracle than the hipfire quant does against the F32 oracle. The 4-bit ordering
(GGUF Q4_K_S < hipfire AWQ-GPTQ in KLD) is consistent with this, not an outlier.

## DATA PROVENANCE
- Window: `dd bs=1 skip=3000000 count=1200000` of slice md5 83b0205a → window
  md5 4e86d460e2c2fec261b35e8d401ff49d. Truncated to 300,000 B for the
  128-chunk runs (`--chunks 128`).
- Native ref: /workspace/qwen3.5-9b-f32-native-repr128.kldref.bin (md5
  e060a9e4...), oracle PPL 9.3198.
- hipfire artifacts: AWQ-GPTQ /workspace/qwen3.5-9b.mq4-awq-pr266-gptq-v3 (md5
  a6a51adf..., the validated "0.1257" model), Q8 /workspace/qwen3.5-9b-q8.hfq;
  F32 oracle /workspace/qwen3.5-9b-f32-oracle.hfq.
- GGUFs: /workspace/explore2-gguf/qwen3.5-9b-{bf16,Q4_K_S,IQ3_S,Q8_0}.gguf.
- llama.cpp: /tmp/llama.cpp/build/bin/llama-perplexity, commit 94a220c (ballpark
  oracle, not the pinned 9dcf835 — KLD insensitive to the exact commit; F2 §0).
- Tools: eval_hipfire (existing), llama-perplexity --kl-divergence (external),
  eval_hipfire_fullvocab (NEW, this session). NO GGUF wired into hipfire crates.
- All runs HIP_VISIBLE_DEVICES=0, mi300 gfx942, --kv-mode f32 per-token for
  hipfire; llama default KV. Outputs in /tmp/f3-*.{kldseq,log}.
