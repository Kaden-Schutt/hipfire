# F2 — Native F32-Oracle KLD Reference (retire the llama.cpp dependency)

Branch: `foundation/native-bf16-fp32-eval` (continues F1; F1 = 110ea419).
Box: mi300 (gfx942 / CDNA3 / MI300X), ROCm 7.0, checkout `/root/hipfire`.
Date: 2026-06-04.

Goal: make the KLD *reference* come from hipfire's OWN F32 forward oracle
(F1 deliverable) instead of llama-perplexity, prove the oracle is sound, and
demonstrate the new native harness end-to-end. This file is the durable
deliverable; numbers appended the moment measured.

---

## STEP 0 — PROVENANCE OF THE OLD kldref (verified on the branch)

### How the prior qwen3.5-9b kldref was generated
- `crates/hipfire-runtime/examples/build_kld_ref.rs` spawns **llama-perplexity**
  (`--kl-divergence-base <fifo>`) on the BF16 GGUF, reads its full-vocab uint16
  log-prob stream, top-K=256 reduces, writes HFKLDR v1.
- Manifest entry `benchmarks/quality-baselines/harness/manifest.json ::
  references["qwen3.5-9b-bf16.kldref.bin"]`:
  - producer_cmd = `build_kld_ref --bf16-gguf .../Qwen3.5-9B-BF16.gguf
    --slice benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
    --top-k 256 --output .../qwen3.5-9b-bf16.kldref.bin`
  - llamacpp_commit_pinned = `9dcf83552887bb898b4a98a5761361e504e31fc3`
  - slice_md5 = `83b0205a304bf4e52172ecdb05f2e895` (wikitext-2 train, 1024 seqs x 2048 ctx)
  - top_k=256, n_ctx=2048, n_vocab=248320, n_chunk=1175, size 2.48 GB.
  - host_arch gfx1151, ~53 min wall, uploaded to HF dataset hipfire-models/qwen-kldref.
- The 2.48 GB bin is **gitignored** (`benchmarks/quality-baselines/refs/.gitignore`
  excludes `*.kldref.bin`); only the small imatrix.gguf sidecars are committed.
  So the bin is NOT on this box — it lives external on HF.

### Tokenization caveat (why this matters)
- `benchmarks/quality-baselines/slice/README.md` + harness Step 1.5 (2026-05-08):
  **45.9% structural divergence** between hipfire's HF-Qwen BPE and llama.cpp's
  GGUF-bundled BPE on this slice. The pipeline sidesteps it: `eval_hipfire` reads
  token IDs FROM the kldref (written by llama), and feeds those to the candidate
  forward — it never re-tokenizes the slice. So every prior hipfire quant KLD was
  scored on **llama's token stream against llama's bf16 distribution**.

### Conclusion (cross-harness confound, as F1 predicted)
- All prior hipfire quant KLDs (e.g. "MQ4-AWQ-GPTQ = 0.1257") were measured vs this
  LLAMA-generated bf16 reference => cross-harness. F1's direct cross-check
  (hipfire-F32 vs llama-bf16 on identical tokens) was 87% top-1 / 0.357 nats / 0.854
  top-256 logit-cosine — NOT a bug (hipfire predicts real next-token on par: 64.8%
  vs llama 63.0%), but a systematic cross-engine SHAPE difference rooted in the two
  different Gated-DeltaNet ports (28/32 layers are linear-attn). That ~0.30-0.36 nat
  floor was baked into every "KLD" number. F2 removes it by making the reference
  hipfire's own F32 forward.

### Local llama.cpp state (for the Step 1 ballpark check)
- llama-perplexity present at `/tmp/llama.cpp/build/bin/llama-perplexity`, commit
  `94a220c` (NOT the pinned 9dcf835 — newer master). Fine for an oracle ballpark
  PPL cross-check; F1 already validated the HFKLDR reconstruction math against it.

---
## STEP 1 — ORACLE SOUNDNESS (PPL on identical tokens)

Bounded slice (first 60 KB of the canonical wikitext slice) -> llama-perplexity
tokenized it into **26 chunks x 512 ctx = 13312 tokens**, scored window = second
half per chunk = 6630 scored positions.

- **llama.cpp bf16 PPL = 11.1652** (+/- 0.39871) over the 26 chunks
  (`llama-perplexity -m qwen3.5-9b-bf16.gguf -c 512 -b 512 -ngl 99
   --kl-divergence-base /tmp/f2_llama_bf16.bin`).
- **hipfire F32-oracle PPL = 11.1758**  (mean NLL = 2.413748) over the
  IDENTICAL 6630 scored tokens (llama's exact token IDs fed via
  `build_kld_ref_native --tokenize-mode tokens-bin`, true FP32 KV).

**Delta = +0.0106 PPL = +0.09%.** Same ballpark => the oracle is SOUND.
(The gap is far smaller than F1's 0.30-0.36 nat KL because PPL only scores the
actual-next-token NLL; the larger KL was a full-distribution SHAPE difference.
Both engines assign near-identical probability to the realized next token —
exactly the "equally-good predictor" finding from F1, now quantified as PPL.)

Tool: `crates/hipfire-runtime/examples/build_kld_ref_native.rs` (NEW, F2,
Kaden Schutt header). Registered in hipfire-runtime/Cargo.toml
(required-features arch-qwen35,deltanet). Built clean.

---
## STEP 2 — THE NATIVE kldref (core deliverable, llama-free)

Tool: `build_kld_ref_native` runs the F1 F32 oracle forward (true FP32 KV,
DeltaNet reset per chunk, score second-half window) and writes per-token
top-K=256 log-probs in the EXACT HFKLDR β v1 format eval_hipfire consumes.
Format is byte-compatible: same 32-byte header (magic HFKLDR\0\0, version 1,
n_ctx, n_vocab, n_chunk, top_k=256, flags, reserved), same token block, same
per-token block (u32 top_indices[256] + f32 top_log_probs[256] + f32
sum_p_residual + f32 pad). Verified eval_hipfire reads it with NO code change
(Step 3 ran against it directly).

- **Native kldref written:**
  `benchmarks/quality-baselines/refs/qwen3.5-9b-f32-native.kldref.bin`
  (13.68 MB; n_ctx=512, n_chunk=26, top_k=256, vocab=248320, 6630 scored).
  Built with hipfire's OWN BPE (`--tokenize-mode hipfire`) over the slice —
  the truly llama-free path.

- Note on tokenization on THIS slice: hipfire's BPE and llama's BPE produced
  the **byte-identical token stream** (13312/13312 = 100% match) on the first
  60 KB of the canonical wikitext slice. The harness's documented 45.9%
  corpus-wide BPE divergence is dominated by special whitespace/markup regions
  elsewhere; this clean prose region agrees exactly. Convenient: it means the
  native kldref and the llama kldref (Step 3) sit on the SAME tokens here, so
  the native-vs-llama delta isolates purely the reference DISTRIBUTION (the
  cross-engine confound), not tokenization.

---
## STEP 3 — NATIVE HARNESS END-TO-END (+ cross-engine confound delta)

Quantized the SAME plain 9B from the HF bf16 snapshot on mi300 with stock
hipfire-quantize (NO AWQ/GPTQ/imatrix — those are the next phase):
- `/workspace/qwen3.5-9b-q8.hfq`  (`--format q8`,  9.53 GB)
- `/workspace/qwen3.5-9b-mq4.hfq` (`--format mq4`, 5.31 GB, flat g256, uncalibrated)

eval_hipfire reads the native HFKLDR ref with ZERO code changes. Per-token
scoring, 6630 scored tokens, vocab 248320, n_ctx=512, top_k=256.

### KLD vs the NATIVE F32 oracle reference (the FIRST trustworthy in-harness numbers)

| candidate | KV mode | KLD (nats) | PPL |
|---|---|---|---|
| Q8       | f32 | **0.008576** | 11.1997 |
| Q8       | q8  | **0.012005** | 11.2247 |
| flat-MQ4 | f32 | **2.433096** | 104.5264 |
| flat-MQ4 | q8  | **2.428937** | 104.0993 |
| flat-MQ4 | f32 (prefill scoring) | 2.431655 | 104.3659 |

Oracle PPL itself = 11.1758. Q8 barely perturbs it (+0.024 PPL, KLD 0.0086) —
a clean, tiny, believable Q8 error. q8-KV adds ~0.0034 nats over f32-KV (sane).

### Cross-engine confound delta (native-ref vs llama-ref, IDENTICAL tokens & forward)

Built a llama bf16 HFKLDR ref on the SAME 6630 tokens (reduced from the
existing llama-perplexity `_logits_` dump; byte-identical token stream to the
native ref — see Step 2). Only the REFERENCE distribution differs.

| candidate | KLD vs NATIVE-ref | KLD vs LLAMA-ref | delta (llama minus native) |
|---|---|---|---|
| Q8       | 0.008576 | 0.015738 | **+0.007162** (llama-ref inflates Q8 error ~84%) |
| flat-MQ4 | 2.433096 | 2.434844 | +0.001748 (negligible vs the 2.43 quant error) |

**Interpretation:**
- For Q8 (small genuine quant error), the cross-engine confound is the DOMINANT
  term: the llama reference reports 0.0157 but ~0.0072 of that is just the
  hipfire-vs-llama DeltaNet/RoPE/norm port difference, NOT Q8 quantization. The
  native oracle strips it out, giving true Q8 KLD = 0.0086. This is the F2
  thesis, quantified: low-error quants were being penalized ~2x by the wrong
  reference.
- For flat-MQ4 the confound is swamped by a genuinely LARGE quant error
  (2.43 nats, PPL ~104, looping greedy output) — flat uncalibrated MQ4-g256 on
  this hybrid 9B is severely degraded on its own merits; both refs agree.
  Stable across KV mode (f32/q8) and scoring mode (per-token/prefill), so it is
  a real MQ4 weakness, not a harness artifact. (Confirms memory's "uncalibrated
  low-bit pervasively broken; AWQ is the lever" theme — that is the next phase.)

### Deliverable status
- Native, llama-FREE kldref produced + format-compatible with eval_hipfire
  (proven: ran directly, no code change). Oracle is sound (Step 1, +0.09% PPL).
- The cross-engine confound is now MEASURED (Q8: +0.0072 nats, ~84% of the
  llama-ref's reported Q8 KLD). First trustworthy in-harness quant KLDs:
  Q8 = 0.0086 (f32-KV) / 0.0120 (q8-KV).

### Bounded-scope note
This used a 26-chunk x 512-ctx bounded slice subset (6630 scored tokens), not
the full 1024-seq x 2048-ctx canonical slice (~1.07M scored tokens) the
production refs use, to stay within session time (oracle forward ~35 tok/s
per-token on the F32 35.8 GB model). The pipeline + math + format + delta are
fully demonstrated; the only remaining gap to a production native ref is wall-
clock: a full 2048-ctx x 1175-chunk native ref = ~1.07M scored tokens at
35 tok/s ~= 8.5 h on this box. Prefill-mode batching of the ORACLE forward
would cut this; `build_kld_ref_native` currently uses per-token forward_scratch
for maximal correctness, matching how F1 validated the oracle.

---

## NOTE — does the PPL match (11.1758 vs 11.1652) rule out an engine error?

No, not conclusively — PPL is a WEAK test, and this nuance matters:
- PPL scores only the NLL of the single REALIZED next token per position
  (one scalar, -log p(actual_next)). Two forwards can assign near-identical
  probability to the actual token while disagreeing on the rest of the
  distribution. F1 saw exactly this: tight next-token agreement (64.8% vs
  63.0% top-1; +0.09% PPL here) yet full-distribution KL 0.30-0.36 nats and
  top-256 centered-logit cosine only 0.854.
- So the PPL match rules out a GROSS engine error (a broken forward predicts
  real text far worse than llama — it does not), and establishes the oracle is
  a SOUND reference for PPL-class + quant-perturbation measurement. It does NOT
  rule out a systematic distribution-SHAPE difference (which F1 detected and
  attributed to the two engines' different Gated-DeltaNet ports).
- A conclusive "no bug" would need an exact match vs a SAME-engine ground
  truth, which is impossible cross-engine for a DeltaNet hybrid. The strongest
  available evidence is the INTERNAL self-consistency F1 measured: hipfire-F32
  vs hipfire-Q8-KV = KL 0.0002 / 100% top-1. Verdict: no gross bug, engine is
  self-consistent and an on-par next-token predictor — but "provably bug-free"
  is not what a PPL match buys.
