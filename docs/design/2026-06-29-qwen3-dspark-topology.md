# Qwen3-8B DSpark drafter — ingest topology & attention mode (Task 0)

**Target arch (nail this first):** the drafter targets **plain Qwen3-8B dense** =
hipfire **`arch_id=1`, the `hipfire-arch-llama` crate** (`config.json`:
`model_type: qwen3`, 36 target layers, dim 4096, 32h/8kv/hd128, QK-norm, rope
θ=1e6). This is **NOT** Qwen3.5/3.6 (`arch_id=5/6`, the `hipfire-arch-qwen35`
DeltaNet-hybrid crate) — that arch and its existing "DFlash" spec machinery are
unrelated and must not be touched. The verify path + per-layer dense compute ride
llama; only the *drafting algorithm shape* is borrowed from deepseek4 (below).

**Provenance (corrected — both weights AND source now verified):**
- Tensor table below: **verified from the real drafter weights** —
  `/home/bjoern/dspark-work/qwen3/ckpt/model.safetensors` (4.7GB BF16, single file,
  64 tensors), downloaded 2026-07-01 (the original note asserted this table without
  the weights on disk).
- Forward-logic claims (bidirectional block, `main_x` dual-stream KV-fusion):
  **byte-confirmed from the real `Qwen3DSparkModel` source** — DeepSpec (DeepSeek's
  spec-decode codebase, MIT, released 2026-06-27), cloned to
  `/home/bjoern/dspark-work/DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py`
  (repo `github.com/deepseek-ai/DeepSpec`). This source does **not** ship with the
  HF checkpoint — it lives only in the DeepSpec repo. Key confirmations:
  `fc = Linear(5*hidden, hidden, bias=False)` then `hidden_norm(fc(...))` before the
  layer loop (modeling.py:130-134,275); noise embedding and `target_hidden` are
  **separate dual-stream layer inputs**, not summed (:277-284); `is_causal = False`
  (:102) with `create_dspark_attention_mask` (:313); and the KV fusion is literally
  `k = cat([k_proj(target_hidden), k_proj(hidden)])` reusing the same `k_proj`/`v_proj`
  (:148-152). The deepseek4 analog (`~/dspark-work/ref/model.py`) matches.

**Headline finding (revises the plan):** the qwen3 DSpark drafter is **not** a
plain dense self-attention transformer with a modified layer-0 input. It is
**deepseek4's DSpark body algorithm with dense Qwen3 layers** — each layer's
attention is **bidirectional** and prepends a **projected target-hidden context**
to its KV (`KV = cat([k_proj(target_ctx), k_proj(block)])`, reusing the layer's own
`k_proj`/`v_proj` — consistent with the single per-layer q/k/v/o set in the weights).
hipfire's deepseek4 `dspark_forward`
(`crates/hipfire-arch-deepseek4/src/forward.rs:8911+`) already implements this exact
structure (MoE/MLA variant); the qwen3 body is the dense variant of the same thing.

## Tensor name → role table (`dspark_qwen3_8b_block7`, verified from `model.safetensors`)

Exact tensor names — note **no `model.` prefix**, and markov tensors are nested
under `markov_head.` (the original draft mis-prefixed both; the loader must use
these names).

| tensor (safetensors) | shape | dtype | role |
|---|---|---|---|
| `embed_tokens.weight` | [151936, 4096] | bf16 | token embedding (noise block) |
| `layers.{0..4}.self_attn.q_proj.weight` | [4096, 4096] | bf16 | Q (32h × 128) |
| `layers.{0..4}.self_attn.{k,v}_proj.weight` | [1024, 4096] | bf16 | K/V (8kv × 128, GQA) |
| `layers.{0..4}.self_attn.o_proj.weight` | [4096, 4096] | bf16 | attn out |
| `layers.{0..4}.self_attn.{q,k}_norm.weight` | [128] | bf16 | QK-norm (Qwen3) |
| `layers.{0..4}.{input_layernorm,post_attention_layernorm}.weight` | [4096] | bf16 | block norms |
| `layers.{0..4}.mlp.{gate,up}_proj.weight` | [12288, 4096] | bf16 | SwiGLU MLP (inter 12288) |
| `layers.{0..4}.mlp.down_proj.weight` | [4096, 12288] | bf16 | SwiGLU MLP down |
| `fc.weight` (`main_proj`) | **[4096, 20480]** | bf16 | **single concat** `[hidden, 5*hidden]` ingest ✓verified |
| `hidden_norm.weight` | [4096] | bf16 | RMSNorm after `fc`, before layers |
| `norm.weight` | [4096] | bf16 | final norm → `x_head` |
| `markov_head.markov_w1.weight` | [151936, 256] | bf16 | vanilla markov embed (`[vocab, rank]`) |
| `markov_head.markov_w2.weight` | [151936, 256] | bf16 | vanilla markov bias proj (`[vocab, rank]`) |
| `confidence_head.proj.weight` | [1, 4352] | bf16 | confidence Linear (dim+rank = 4096+256) |
| `confidence_head.proj.bias` | [1] | bf16 | **confidence HAS BIAS** ✓verified present (deepseek4 has none) |
| `lm_head.weight` | [151936, 4096] | bf16 | separate lm_head (untied, `tie_word_embeddings:false`) |

## The 7 questions

1. **`main_proj` shape:** single `[4096, 20480]` = `[hidden, 5*hidden]` **concat**
   (named `fc`). Generalizes deepseek4's `[hidden, 3*hidden]` to `n_targets=5`.
2. **Layer-0 / ingest rule:** NOT `embed(noise)+main_proj` sum. `main_x =
   hidden_norm(fc(main_hidden))` is computed once and fed as a **separate context
   side-channel** to every layer. Per layer: query = block hidden (layer 0 =
   `embed(noise_block)`); `KV = cat([k_proj(main_x_context), k_proj(block)])`,
   `V` likewise. `hidden_norm` applies after `fc`, before the layer loop. This is
   structurally identical to deepseek4 `dspark_forward` step A (`main_x =
   main_norm(main_proj(main_hidden))`, forward.rs:9014-9058) + its custom
   bidirectional stager (forward.rs:9100 `n_valid[b]=n_committed+block`).
3. **Block attention:** **bidirectional.** Reference `DSparkAttention.is_causal =
   False`; at inference `attention_mask=None` ⇒ every block slot attends to all
   context + all block slots. Matches deepseek4's bidirectional mode.
4. **KV lifetime (reworded — the original "NO" was imprecise):** the *draft-block*
   KV is **cropped per block** (`deepspec_draft_ops.py`:
   `past_key_values_draft.crop(start)`), so the speculative block's own KV is
   discarded each window. But the *context/main* KV **is persistent** — deepseek4
   keeps a `win`-sized `main_kv` ring (`model.py:763-768/783-784`) that carries the
   projected committed context across windows; only the per-window seed is written
   into it. So: committed context = a persistent `main_x`-derived KV ring; draft
   block KV = rebuilt per block.
5. **Markov head:** vanilla rank-256, identical to deepseek4 (`markov_w1`
   Emb[vocab,256], `markov_w2` Linear[256→vocab] used as bias). matches
   `forward.rs:9663-9715`.
6. **Confidence head:** `Linear[dim+rank=4352, 1]` over `[x_head ++ markov_emb]` —
   **with bias** ✓**verified**: `confidence_head.proj.weight [1,4352]` +
   `confidence_head.proj.bias [1]` both present in the weights (deepseek4's is
   bias-free). Loader + forward must include the `confidence_head.proj.bias` term —
   already carried as `dspark_core::DsparkWeights::confidence_bias` (None-safe
   optional, added in the bias-add at `run_heads` ~468-471).
7. **`x_head` / `lm_head`:** `x_head` = post-`norm` hidden; `lm_head` is a separate
   `[151936,4096]` tensor (`tie_word_embeddings:false`).

## Branch points for later tasks

| affects | decision |
|---|---|
| ingest (`fc`) | single concat `[4096, 20480]`; generalize core `n_targets*dim` width |
| body attention | **bidirectional block over `[main_x context ++ block]` KV** — NOT llama's plain causal self-attn forward |
| body layers | dense Qwen3 (rmsnorm → GQA self-attn w/ qk-norm → swiglu) — kernels exist in llama/rdna-compute |
| closest existing impl | **deepseek4 `dspark_forward`** (forward.rs:8911-9739), dense variant — NOT llama `forward_scratch_layers` |
| confidence | include bias term |
| heads | separate `lm_head`; vanilla markov == deepseek4 |

## Deployed sidecar naming convention

Deployed sidecar = `<target_stem>-dspark.<target_ext>`; discovery in `carriers.rs:599`.
For target `qwen3-8b.mq4` the discovered path is `qwen3-8b-dspark.mq4` (HFQ container,
`.mq4` ext mirrors the target). Source-of-truth build artifact may be kept as
`qwen3-8b-dspark.hfq`; the deployed copy at the canonical name is byte-identical.

## Plan impact (escalated to human)

The Stage-1 plan's premise — "the qwen3 body fits llama's existing forward
unchanged; only gemma4 forces generalization" — is **false**. The qwen3 body
needs the DSpark bidirectional-block-attention-over-projected-context machinery,
which deepseek4 has and llama does not. The body should be built by adapting
deepseek4's `dspark_forward` (swap MoE/MLA/HC → dense Qwen3 layers), or by a new
masked-batched-GQA path over `[context ++ block]` keys (the #483
`attention_q8_0_kv_batched_masked` can express the all-visible bias). This also
suggests the bidirectional-block-staging belongs in `dspark-core` (shared with
deepseek4), with only the per-layer compute as the arch seam. `hipfire-dense`
(gemma knobs) remains a separate Stage-2 concern.
