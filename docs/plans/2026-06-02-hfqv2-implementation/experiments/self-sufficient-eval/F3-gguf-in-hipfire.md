<!-- Copyright (c) 2026 Kaden Schutt -->
# F3 — GGUF-in-hipfire: the apples-to-apples "does hipfire beat GGUF" measurement

Branch: `foundation/native-bf16-fp32-eval` (continues F2-eval = 5f9655d2).
Box: mi300 (gfx942 / CDNA3 / MI300X VF), ROCm 7.0, checkout `/root/hipfire`.
Date: 2026-06-04.

## The point of F3

Every prior hipfire-vs-GGUF comparison was CROSS-HARNESS: hipfire quants scored
through hipfire forward, GGUF quants scored through llama.cpp `llama-perplexity`
(see `eval_gguf.rs`). Different oracle, corpus, tokenizer, top-K, KV mode. The
gap between "hipfire MQ4 = 0.1257" and "GGUF Q4_K = X" was therefore unprovable.

F3 removes ALL of it: load the GGUF tensor set INTO hipfire (CPU dequant -> F32 ->
GPU), run it through the SAME `qwen35::forward_*` path, SAME top-K-of-ref KLD
scorer, SAME corpus/tokens, vs the SAME hipfire-native F32 oracle reference. The
ONLY variable left is the quantization of the weights. This is the headline
measurement the whole quant-quality program was built to make.

## Architecture reconciliation (GGUF qwen35 <-> hipfire Qwen35Weights)

llama.cpp fork ships `general.architecture = qwen35` (src/models/qwen35.cpp) — a
hybrid Gated-DeltaNet (linear attn) + full-attn model, block_count=33,
full_attention_interval=4. DeltaNet dims (verified from BOTH the GGUF metadata
and the oracle .hfq metadata_json — the code default `unwrap_or(16)` for
value_heads is STALE; real config = 32 value heads):

- ssm_d_state=128, ssm_n_group=16 (k heads), ssm_dt_rank=32 (v heads),
  ssm_d_inner=4096, ssm_d_conv=4.
- key_dim = 128*16 = 2048; value_dim = 128*32 = 4096; conv_dim = 2048*2+4096 = 8192.

Tensor map (GGUF ne=[k,m] storage == hipfire row-major [m,k], so direct for 2D linear):

| hipfire (.hfq) tensor              | [m,k]        | GGUF tensor          | notes                          |
|------------------------------------|--------------|----------------------|--------------------------------|
| layers.N.linear_attn.in_proj_qkv   | [8192,4096]  | blk.N.attn_qkv       | Q+K+V direct                   |
| layers.N.linear_attn.in_proj_z     | [4096,4096]  | blk.N.attn_gate      | gate Z direct                  |
| layers.N.linear_attn.in_proj_a     | [32,4096]    | blk.N.ssm_alpha      | direct                         |
| layers.N.linear_attn.in_proj_b     | [32,4096]    | blk.N.ssm_beta       | direct                         |
| layers.N.linear_attn.A_log         | [32]         | blk.N.ssm_a          | F32 direct                     |
| layers.N.linear_attn.dt_bias       | [32]         | blk.N.ssm_dt.bias    | F32 direct                     |
| layers.N.linear_attn.conv1d.weight | [8192*4]flat | blk.N.ssm_conv1d     | **TRANSPOSE** ne=[4,8192]->[8192,4] |
| layers.N.linear_attn.norm.weight   | [128]        | blk.N.ssm_norm       | F32 direct (NO +1 — gated norm)|
| layers.N.linear_attn.out_proj      | [4096,4096]  | blk.N.ssm_out        | direct                         |
| layers.N.self_attn.q_proj          | [8192,4096]  | blk.N.attn_q         | 2x wide (q+gate) direct        |
| layers.N.self_attn.k_proj          | [1024,4096]  | blk.N.attn_k         | direct                         |
| layers.N.self_attn.v_proj          | [1024,4096]  | blk.N.attn_v         | direct                         |
| layers.N.self_attn.o_proj          | [4096,4096]  | blk.N.attn_output    | direct                         |
| layers.N.self_attn.q_norm          | [256]        | blk.N.attn_q_norm    | norm (NO +1)                   |
| layers.N.self_attn.k_norm          | [256]        | blk.N.attn_k_norm    | norm (NO +1)                   |
| layers.N.{input,post_attn}_layernorm| [4096]      | blk.N.attn_norm / post_attention_norm | RMSNorm +1.0 bake |
| layers.N.mlp.{gate,up,down}_proj   |              | blk.N.ffn_{gate,up,down} | direct                     |
| embed_tokens / lm_head / norm      |              | token_embd / output / output_norm |                   |

Norm convention: GGUF bakes (1+w) at conversion time for the RMSNorm weights
(attn_norm, post_attention_norm, output_norm) — so loaded GGUF norm values are
ALREADY the runtime scale and MUST NOT get the +1.0 bake again. q_norm/k_norm/
ssm_norm are plain (no +1) in both. (Verified empirically by the Q8_0 sanity gate.)


NORM-CONVENTION CORRECTION (verified llama.cpp conversion/qwen.py:302-303): the
GGUF conversion adds +1 to EVERY *norm.weight EXCEPT linear_attn.norm.weight.
So ALL GGUF norm tensors load into hipfire as PLAIN F32 with NO +1 re-bake (they
are already runtime-scale). q_norm/k_norm/attn_norm/post_attn_norm/output_norm =
already +1-baked in GGUF; ssm_norm = never +1-baked (matches hipfire). The Q8_0
sanity gate is what confirms this end-to-end.

## WIRING (Step 2) — what was built

New code on branch foundation/native-bf16-fp32-eval (local, uncommitted):
- crates/hipfire-runtime/src/llama.rs: added `pub fn dequantize_q5_k` (a GGML
  `dequantize_row_q5_K` port — needed for the 8 Q5_K tensors in Q4_K_S) plus a
  Q5K dispatch arm in `load_tensor_f32`. (gguf.rs already recognized Q5K=13.)
- crates/hipfire-arch-qwen35/src/qwen35.rs: added `pub fn load_weights_gguf` —
  builds `Qwen35Weights` from a GGUF by CPU-dequantizing every tensor to F32 and
  uploading as DType::F32, so the forward path is IDENTICAL to the oracle (only
  weight VALUES differ). Handles the qkv/gate split, the conv1d transpose, and
  the no-+1 norm convention. Helpers: gguf_dequant_f32 / gguf_weight_f32 /
  gguf_load_weight / gguf_load_vec / gguf_load_conv.
- crates/hipfire-runtime/examples/eval_gguf_in_hipfire.rs (NEW) — loads the GGUF
  via load_weights_gguf (config from the oracle .hfq), runs the SAME per-token
  forward_scratch + top-K-of-ref KLD scorer eval_hipfire uses, vs the native
  HFKLDR ref. Registered in Cargo.toml. Built clean on mi300/gfx942.

GGUFs located: /workspace/explore2-gguf/qwen3.5-9b-{Q8_0,Q4_K_S,IQ3_S,bf16}.gguf
(all general.architecture=qwen35, block_count=33, dense 9B — confirmed plain).
  - Q8_0   9.79 GB md5 e5bf6963 (types: Q8_0 + F32)
  - Q4_K_S 5.49 GB md5 380e16bc (types: Q4_K x249 + Q5_K x8 + Q6_K x1 + F32)
  - IQ3_S  4.48 GB md5 fac5bb8c (types: IQ3_S x224 + Q4_K x33 + Q6_K x1 + F32)
NOTE: the hipfire gguf reader rejects unknown GGML types at open; IQ3_S
(type 21) has no CPU dequant, so IQ3_S (Step 4c) is the BONUS, attempted only
if time remains after the headline.

## EFFECTIVE BPW (all-tensors, computed from GGUF n_bytes / n_elements over 9.20B params)

| candidate              | effective bpw | size    |
|------------------------|--------------:|---------|
| GGUF Q8_0              | 8.50          | 9.79 GB |
| GGUF Q4_K_S            | **4.76**      | 5.49 GB |
| GGUF IQ3_S             | 3.88          | 4.48 GB |
| GGUF bf16 (oracle src) | 16.00         | 18.4 GB |
| hipfire MQ4-AWQ-GPTQ   | ~4.35 all / 4.25 weights | 5.00 GB |
| hipfire Q8             | ~8.3          | 9.53 GB |

The headline same-bar comparison is hipfire MQ4-AWQ-GPTQ (~4.25-4.35 bpw) vs
GGUF Q4_K_S (4.76 bpw): hipfire is at LOWER bpw, so equal-or-better KLD = a win.

## STEP 1 — REPRESENTATIVE NATIVE F32-ORACLE REFERENCE (the headline slice)

The existing F2 native ref was 26 chunks over the FIRST 60 KB of the canonical
wikitext slice — an outlier-hard prose region (oracle PPL 11.18, where flat-MQ4
blows up to PPL 104). For the F3 headline we built a NEW, representative ref.

- Source corpus: the SAME canonical slice
  `benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt`
  (md5 83b0205a304bf4e52172ecdb05f2e895, wikitext-2).
- Representative span: a contiguous 1.20 MB middle window starting at byte
  offset 3,000,000 (skips the first-60KB outlier region entirely; general-prose
  Wikipedia articles — chemistry/cadmium etc.). md5 of the extracted window
  4e86d460e2c2fec261b35e8d401ff49d. hipfire-BPE tokenized to 276,730 tokens.
- Chunked into n_ctx=512, capped at **128 contiguous chunks** = 65,536 tokens;
  scored window = second half per chunk = **32,640 scored positions**.
- Tool: `build_kld_ref_native` (F2), hipfire's OWN BPE (`--tokenize-mode hipfire`),
  the F1 F32 oracle (`/workspace/qwen3.5-9b-f32-oracle.hfq`, ~35.8 GB), true FP32
  KV, DeltaNet reset per chunk, top-K=256, HFKLDR v1 format.
- Output: `/workspace/qwen3.5-9b-f32-native-repr128.kldref.bin` (67.4 MB,
  md5 e060a9e43a1fcd4580af30651c95fbf7), built in 932 s (~35 tok/s).

**ORACLE PPL on this representative slice = 9.3198** (mean NLL 2.232145, 32640
scored tokens). This is the F32 oracle's own perplexity — the reference floor.
(Notably lower / easier than the 26-chunk outlier's 11.18, as intended: this is
a representative span, not the hard tail.)

---

## HALT (2026-06-04) — GGUF-into-hipfire is the WRONG approach; reverted

User directive mid-session: "we shouldn't be wiring ggufs into hipfire." Correct —
and the Q8_0 sanity gate had just proven WHY, empirically. All code wiring was
reverted (`git checkout` of qwen35.rs / llama.rs / gguf.rs / Cargo.toml; deleted
the eval_gguf_in_hipfire.rs example). This doc is kept as the record of the
finding so the dead end is documented and not re-walked.

### What was built then reverted
- `load_weights_gguf` (qwen35.rs): GGUF -> F32 -> GPU into Qwen35Weights.
- `dequantize_q5_k` + `dequantize_iq3_s` (+ IQ3S grid) in llama.rs; Q5K/IQ3S
  dispatch in gguf.rs from_u32/block_size/block_bytes.
- `eval_gguf_in_hipfire.rs` example (per-token KLD vs native ref).
All built clean on mi300/gfx942 and RAN — the wiring itself worked.

### The Q8_0 SANITY GATE FAILED — and that is the load-bearing result
GGUF Q8_0 loaded in-hipfire, scored vs the native repr-128 F32 oracle ref:
  **slice-mean KLD = 10.19, PPL = 224,158** (vs the expected ~0.009 Q8 floor).
A PPL of 224K = near-random output = the forward consumed mis-laid-out weights.

### ROOT CAUSE (why GGUF tensors are NOT drop-in for hipfire's forward)
llama.cpp's qwen35 GGUF converter applies a **V-head reorder permutation** to the
Gated-DeltaNet projections. From `conversion/qwen.py::_reorder_v_heads`:
  "reorders V heads from grouped to tiled order for ggml broadcast ... HF stores
   V heads grouped by K head [G0_v0..v{r-1}, G1_v0..]; ggml binary ops use tiled
   broadcast [K0,K1,...,K0,K1,...]. We reorder so ggml_repeat can replace the
   expensive interleaved repeat."
Applied to `in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj`. hipfire's loader
reads the HF safetensors layout directly and its DeltaNet kernels expect the
GROUPED (HF) order. So a GGUF tensor set is permuted relative to what hipfire's
forward consumes — they are NOT the same byte layout, and dequant alone cannot
fix it. (norms, conv1d-transpose, qkv/gate split were all handled correctly; the
V-head reorder is the breaker, and it is engine-specific by design.)

### Why this also makes the approach conceptually wrong, not just buggy
Even if the permutation were inverted tensor-by-tensor, you'd be reverse-
engineering llama.cpp's converter to re-derive HF weights, then re-quantizing the
*intent* of a GGUF quant through hipfire's own forward — which is no longer "the
GGUF quant scored honestly," it's "a hipfire re-interpretation of GGUF codes."
The clean, defensible apples-to-apples bar does NOT require importing GGUF bytes
into hipfire. The right design keeps each engine running its own weights and
controls the CONFOUND instead (shared tokens, shared corpus, shared top-K, and —
critically — the now-available hipfire-native F32 oracle as the single reference
that BOTH a hipfire candidate and a llama/GGUF candidate are scored against via
their own forwards). That is what `eval_gguf.rs` (llama-perplexity FIFO) + the
native HFKLDR ref already enable without any byte-level import.

### Salvage / what survives this halt (still valid, NOT reverted)
1. **Representative native F32-oracle ref** — the durable, reusable asset:
   `/workspace/qwen3.5-9b-f32-native-repr128.kldref.bin` (67.4 MB, md5
   e060a9e43a1fcd4580af30651c95fbf7), 128 contiguous 512-ctx chunks from a
   representative mid-corpus span (byte offset 3.0 MB of the canonical wikitext
   slice), 32,640 scored tokens, **oracle PPL = 9.3198**. This replaces the
   26-chunk outlier ref for any future headline and is ref-engine-agnostic.
2. The effective-bpw table (Q8_0 8.50 / Q4_K_S 4.76 / IQ3_S 3.88 / hipfire
   MQ4-AWQ-GPTQ ~4.25-4.35) — independent of the import approach.
3. The architecture reconciliation notes above remain accurate as documentation
   of the GGUF qwen35 layout (and the V-head-reorder gotcha is now on record).

### Correct next step (for whoever picks this up)
Do the apples-to-apples WITHOUT importing GGUF bytes: score the GGUF candidates
through llama-perplexity (existing `eval_gguf.rs`) against THIS representative
native HFKLDR ref, and score the hipfire candidates through `eval_hipfire`
against the SAME ref. Shared ref + shared tokens (read from the ref) = the
confound is controlled to the cross-engine forward-shape term F2 already measured
(~0.0008 nats for a 4-bit candidate). No byte-level weight import required.
