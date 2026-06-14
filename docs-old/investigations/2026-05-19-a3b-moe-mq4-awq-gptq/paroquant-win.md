# ParoQuant breaks the A3B-MoE structural floor (2026-05-20)

## Result

ParoQuant on Qwen3.6-35B-A3B (shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5
unpacked) lands **KLD = 0.0933** at c16 q8-KV prefill on gfx1151 —
a **10.3× reduction** vs the MQ4+AWQ structural floor (0.946) that I1
(corpus), I4 (α sweep), the calibration-bottleneck probe, the GPTQ math
gate, and I5 (precision floor) all converged on.

| Variant | KLD @ c16 | NLL | PPL | Δ vs mode-1 baseline |
|---|---:|---:|---:|---:|
| mq4-kmap1 (no-AWQ baseline) | 0.9566 | 2.6482 | 14.13 | — |
| mq4-awq-f1-a025 (best F1 AWQ) | 0.9460 | 2.6336 | 13.92 | -1.11% |
| mq4 (kmap mode-0, MQ6 promote) | 0.9452 | 2.6397 | 14.01 | -1.19% |
| hfq6 (pure 6-bit) | 0.9500 | 2.6486 | 14.13 | -0.69% |
| **shisa-ai A3B-PARO (gated rotations)** | **0.0933** | **1.8552** | **6.39** | **-90.2%** |

Source log line:
- `eval_hipfire: slice-mean KLD = 0.093279  mean NLL = 1.855190  PPL = 6.3929`

## What this confirms

The probe + GPTQ math gate predicted that the ~0.95 floor was a
**representational** limit — top-k routing-conditioned activation
variance has no DOF in per-row weight scaling (AWQ) or per-row Hessian
correction (GPTQ). ParoQuant's gated rotations DO have that DOF — they
operate at the rotation layer, before activation quantization happens —
and the 10× KLD reduction confirms the structural reading.

The investigation's representational-vs-precision-floor hypothesis (I5
§"What this means") is now closed in favor of "representational floor,
ParoQuant breaks it".

## Engine work this required (`feat/paroquant-native`)

Three bugs caught and fixed to make hipfire load + run shisa-ai's
ParoQuant-quantized A3B model:

1. **GemmaRMSNorm `+1` offset dropped on PARO load path.** `paro_load_norm_raw`
   (no `+1`) was used for every norm; the matching kernel expects pre-baked
   weights. Fixed at qwen35.rs:1794 (merged to single `paro_load_norm` that
   bakes `+1`). hipEngine's LESSONS-LEARNED.md documents the same fix as
   "KLD 12.71 → 0.19".
2. **MoE loader was a `panic!("not yet implemented")`.** Implemented
   `(LinearAttention, MoE)` and `(FullAttention, MoE)` arms at qwen35.rs:75
   via `paro_load_moe_ffn` (qwen35.rs:1274), `alias_paro_rotation`
   (qwen35.rs:1248), and `MoeParoSidecars` struct (qwen35.rs:395). Per-layer
   shared PARO sidecars aliased into each of 256 experts via
   `unsafe DeviceBuffer::alias()`; gate∥up fused at byte-row level at upload.
3. **`lm_head` hardcoded as tied to `embed_tokens`.** shisa-ai's checkpoint
   has `tie_word_embeddings: false` and ships a separate `lm_head.weight`
   tensor. Every logit was projected against the wrong unembedding matrix,
   producing token-118401 attractor ("出错" / "error") even though all
   layer-0 numerics were clean. Fixed at qwen35.rs:2081 with conditional
   load (prefers separate `lm_head.weight`, falls back to `embed_tokens`).

Plus two infrastructure pieces:
- 136-commit merge of upstream/master into feat/paroquant-native (RoPE
  halfsplit fix + 135 others). Conflicts resolved across qwen35.rs,
  dflash.rs, hfq.rs, llama.rs, tokenizer.rs.
- `eval_hipfire` extended with safetensors auto-route (mirrors daemon.rs
  pattern) so KLD bench works on safetensors directories.

## Open questions

1. **Our 0.0933 vs shisa-ai's reported 0.03468.** A 2.7× gap. Possible causes:
   - Different KLD aggregation (slice-mean vs token-mean)
   - Different BF16 reference (our `qwen3.6-35b-a3b-bf16.kldref.bin` vs
     shisa-ai's HF native BF16 forward)
   - Different eval corpus (wikitext slice vs `tx4/quality3` validation)
   - Possible residual hipfire bug producing additional quality loss
2. **`HIPFIRE_PARO_DEBUG=1` trace prints are still in qwen35.rs** at lines
   3728, 7562, 8083+. Gated by env var (off by default). Probably worth
   removing or refining as part of a cleanup commit.
3. **`HIPFIRE_GRAPH=0` is required** to dodge a stream/graph capture
   conflict on the PARO path. Worth a separate fix before this branch ships,
   but doesn't affect correctness.
4. **Per-seq dump preserved** at `.codeinsight+research/a3b-moe-awq-gptq/per-seq/a3b-paro-shisa__gfx1151__prefill__c16.kldseq`.

## Decision

The original investigation (`docs/investigations/2026-05-19-a3b-moe-mq4-awq-gptq/`)
closes with a clear answer: **ship ParoQuant for A3B production**, not MQ4+AWQ.

- Best MQ4+AWQ candidate: KLD 0.946 (mq4-awq-f1-a025, 21.3 GiB)
- ParoQuant candidate: KLD 0.093, 21.7 GiB unpacked / 19.07 GiB packed
- ~10× quality improvement at essentially identical size

The MQ4+AWQ+GPTQ investigation (Task 13) closes "falsified by structure".
The ParoQuant lever (formerly hypothesis #2 in the I5 surviving levers list)
is the production answer.

## Pinned artifacts

- Branch: `feat/paroquant-native` (HEAD = 09428d11 + uncommitted engine
  fixes: norm `+1` bake, MoE loader, lm_head conditional, eval_hipfire
  safetensors auto-route, PARO_DEBUG trace prints)
- Model: `/home/bjoern/.hipfire/models/shisa-Qwen3.6-35B-A3B-PARO-unpacked`
  (z-lab/Qwen3.6-35B-A3B-PARO is broken at calibration; do not use)
- Bench log: `.codeinsight+research/a3b-moe-awq-gptq/a3b-paro-shisa-c16-kld.log`
- Per-seq dump: `.codeinsight+research/a3b-moe-awq-gptq/per-seq/a3b-paro-shisa__gfx1151__prefill__c16.kldseq`
- Divergence investigation: `paroquant-divergence.md`
- Comparison report: `paroquant-comparison.md`
