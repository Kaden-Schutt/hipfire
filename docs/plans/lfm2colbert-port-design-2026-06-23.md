# hipfire Port Design — LiquidAI/LFM2.5-ColBERT-350M (bidirectional late-interaction retriever)

> Source: `hipfire-colbert-embedding-design` workflow (recon → design → adversarial review), 2026-06-23.
> **Status: PARKED — Phase 2.** Review verdict: **feasible-with-corrections, ~2-3 weeks.**
> WARNING: the design body was written pre-review. Apply the "Adversarial review" corrections at the bottom before implementing — especially: use `rope_batched_f32` (NOT `rope_f32`); conv is NEW work (not reuse); `rope_theta=1e6` (not the lfm2moe 5e6); honor the intermediate-size auto-adjust (~4608, not 6656).

---

The recon checks out against the actual code. The lfm2moe attention path is confirmed causal-by-construction (single-token decode, `pos_buf`-limited context, no prefill forward), the dispatch enum has the non-causal `AttnFull*` variants, `derive_arch_id` has lfm2=11 / cohere2=12 with 13+ free, and the carrier/registry pattern is exactly as described. Here is the design.

---

# hipfire Port Design — LiquidAI/LFM2.5-ColBERT-350M (bidirectional late-interaction retriever)

## 0. Scope and shape of the problem

LFM2.5-ColBERT-350M is **not a decoder**. It is a bidirectional LFM2 backbone (10 conv + 6 attention layers, `hidden=1024`, `intermediate=6656`, 16 Q-heads / 8 KV-heads, RoPE θ=1e6) topped by a **bias-free Dense 1024→128 projection** that emits one 128-d vector *per token*. Retrieval is **late-interaction MaxSim**, computed host-side over those per-token matrices.

Three consequences drive the whole design:

1. **The attention must be bidirectional.** Every existing hipfire attention path (`attention_q8_0_kv`, lfm2moe's whole forward) is single-token causal decode. This is the one genuinely new GPU kernel.
2. **There is no token loop, no KV-cache, no sampler.** We run a *single batched prefill* over the full sequence (≤512 tokens), read out hidden states, project, download. No daemon decode machinery applies.
3. **It is encode-only.** The "inference" product surface is `encode(text) -> [seq, 128] f32`, plus a host-side index + MaxSim scorer. The daemon's load/tokenizer/dispatch plumbing is reusable; its generate loop is not.

This is much closer to the **dots.ocr vision tower** (one-shot encoder, no cache, non-causal) than to any text decoder we ship — so the design deliberately fuses *dots.ocr's encoder topology* with *lfm2moe's conv/attn/MoE weight stack*.

---

## 1. New arch: id, crate, carrier

**arch_id = 14.** (11=lfm2_moe, 12=cohere2moe, 13=gemma4 are taken; 14 is the next free slot in both the `derive_arch_id` switch and the HFQ-header namespace. Do **not** overload 11 — the LFM2 decoder carrier `Lfm2MoeCarrier` builds a decode-only `Lfm2MoeState` with KV + rolling conv state, which is wrong for a bidirectional encoder, and reusing the id would trip the registry-overlap detector at `lib.rs:812-823`.)

**Crate:** `crates/hipfire-arch-lfm2colbert/` (family name `lfm2colbert`). It is a *sibling* of `hipfire-arch-lfm2moe`, not a feature flag on it, because the state struct, forward signature, and attention kernel all differ. It depends on `hipfire-arch-lfm2moe` only to **reuse the conv mixer + weight structs** (see §2), nothing else.

```
crates/hipfire-arch-lfm2colbert/
  src/
    arch.rs        # zero-sized Architecture marker (mirror lfm2moe/src/arch.rs:1-51)
    config.rs      # Lfm2ColbertConfig: layer_types, hidden, interm, heads/kv,
                   #   rope_theta, eps, proj_dim=128, query_marker=64400, doc_marker=64401
    weights.rs     # Lfm2ColbertWeights: reuse ConvWeights/AttnWeights from lfm2moe,
                   #   + dense FFN (Q8 SwiGLU), + projection [128,1024] bias-free
    state.rs       # Lfm2ColbertEncoderState (batched, no KV/conv-ring) — see §2.4
    forward.rs     # forward_encode(...) -> [seq,128] — the new bidirectional path
  examples/
    encode_colbert.rs   # spike harness (Phase B1): text -> [seq,128] JSON
```

**Carrier** — clone `DotsOcrCarrier` (`hipfire-loader/src/carriers.rs:531-586`), the existing *non-generative* carrier template:

```rust
pub struct Lfm2ColbertCarrier;
impl Carrier for Lfm2ColbertCarrier {
    fn name(&self) -> &'static str { "lfm2colbert" }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool { arch_id == 14 }
    fn load(&self, src, ctx) -> Result<LoadedModel,String> { /* HFQ or Dir → cfg+weights+encoder state */ }
}
```

Register it in `REGISTRY` (`lib.rs:54-63`) and add `"lfm2" (retriever variant)` routing. **Routing subtlety:** safetensors `model_type` for both LFM2.5 retrievers is still `lfm2`, which `derive_arch_id` currently maps to 11. Disambiguate inside `derive_arch_id` (`safetensors_source.rs:255-283`) by inspecting the config for the late-interaction signature — presence of a `Dense`/projection module (`out_features==128, bias==false`) or `architectures: ["LFM2BidirectionalModel"]` / sentence-transformers `modules.json` → 14, else 11. The HFQ path is unambiguous: the quantizer (`hipfire-quantize` `auto_arch_id`) stamps `arch_id=14` in the header at ingest time, so HFQ files self-route.

---

## 2. Encoder forward — reuse map (what's borrowed vs new)

The forward is `forward_encode(state, cfg, weights, embeds[seq, 1024]) -> proj[seq, 128]`, structurally a **dots.ocr-style one-shot encoder loop** (`dots_ocr.rs:1238-1743`) but with LFM2's per-layer conv-or-attention mixer (`lfm2moe/forward.rs:155-237`).

### 2.1 REUSE from lfm2moe (`crates/hipfire-arch-lfm2moe/`) — verbatim or near-verbatim

| Piece | Source | How reused |
|---|---|---|
| **Conv mixer block** | `forward.rs:161-173` (`conv1d_gated_decode_f32`) | The depthwise short-conv is causal *by definition* (kernel=3, left-padded). In an encoder we run it as **batched causal conv over the full sequence** — `in_proj → [B|C|x] → gated conv1d → out_proj+resid`. The math is identical; we just feed all `seq` positions instead of one. **Either** add a `conv1d_gated_prefill_f32` kernel that processes `[seq, hidden]` with left-zero-padding (cheap — it's a sliding window), **or** loop the existing decode kernel `seq` times seeding the ring buffer. Causal conv is *correct in a bidirectional encoder* — LFM2 retrievers ship exactly this hybrid; only the *attention* is bidirectional. |
| **Weight structs** `ConvWeights`, `AttnWeights` | `lfm2moe.rs:199-222` | Reused directly: in_proj/conv_weight/out_proj; wq/wk/wv/wo + per-head q_norm/k_norm [head_dim]. |
| **Per-head QK-norm** | `forward.rs:185-188` `rmsnorm_batched` | Reused verbatim — already batch-mode over `n_heads × head_dim`, exactly what a multi-token encoder needs. |
| **Full-dim rotate_half RoPE** | `forward.rs:191-200` `rope_f32` | Reused. RoPE is symmetric/position-indexed, valid bidirectionally. Encoder feeds positions `0..seq_len` (a `pos_buf` vector instead of a scalar) — see §2.3 new-work note. |
| **Dense SwiGLU FFN** (Q8: w1 gate, w3 up, silu_mul, w2 down) | `forward.rs` dense path | Reused for all FFN sublayers (retriever is small; all-dense, `num_experts==0` → `num_dense_layers==all`, per `config.rs:243-248`). The MoE expert path is **not** used by 350M-ColBERT and is excluded from this crate. |

### 2.2 REUSE from dots.ocr (`crates/hipfire-arch-dots-ocr/`) — encoder topology + primitives

| Piece | Source | How reused |
|---|---|---|
| **Encoder loop topology** | `dots_ocr.rs:1238-1743` | The "patch_embed → N blocks → post_trunk_norm → (skip merger)" skeleton is the template for `forward_encode`. We inject **zero** decode/cache/sampler state — exactly the property we want. |
| **Non-causal attention dispatch interface** | `dots_ocr.rs:1550-1626`, `FullAttnParams{q,k,v,out,n_heads,n_kv_heads,head_dim,...}` via `attention_family.run_full_attention` | This is the dispatch shape for our new bidirectional GQA kernel. Selected via `KernelKey::AttnFullF32` (already exists, `dispatch/types.rs:337-338`). |
| **`qkv_split_interleaved_f32`** | `dots_ocr.rs:1514-1534` | LFM2 uses separate wq/wk/wv (not fused QKV), so we likely *don't* need the deinterleave — but keep it available if a fused-QKV HFQ packing is chosen at quant time. |
| **RMSNorm primitive** | `dots_ocr.rs:1395` `gpu.rmsnorm_f32(x,w,out,eps)` | Reused for pre/post-mixer and final norm. eps from config (LFM2 1e-5). |
| **Linear / bias-free GEMM** | `dots_ocr.rs:1108-1166` `linear_f16(...)` (bias-free variant) | This is exactly the primitive for the **128-d projection head** (bias-free 1024→128, F16 weight × F32 activations, WMMA on gfx11/12, scalar fallback). One call, `out_dim=128, in_dim=1024, n=seq`. |
| **Load-time weight fusion** | `dots_ocr.rs:478-489, 695-729` `load_f16_or_dequant_concat_rows` | Reused to fuse dense fc1+fc3 into one GEMM (one-shot encoder ⇒ no decode-time fc1/fc3 reuse concern, so the dots.ocr fusion assumption holds — *this* is the case lfm2moe's risk note warned about, and it's safe here). |
| **Norm/bias loaders, BF16→F16 widening** | `dots_ocr.rs:556-612, 863-909` | Reused — LFM2.5 ships **BF16**; the dots.ocr loader already widens BF16→F16 through the HFQ numerical path. |

### 2.3 GENUINELY NEW — the bidirectional attention kernel (the hard core)

The single load-bearing new kernel. Existing `attention_q8_0_kv` (`kernels/src/attention_q8_0_kv.hip:19-105`) computes `seq_len = pos_buf[0]+1` and softmaxes over `[0, seq_len)` — causality is *implicit* in the decoder limiting context to one growing position. There is **no explicit mask to flip**; we need an all-positions-attend-all kernel.

Two implementation options, pick by effort:

- **Option A (fast path to a spike):** route through the **already-bidirectional** `attention_dflash.hip` (`kernels/src/attention_dflash.hip:1-30` — "each query position attends to ALL key/value positions, non-causal, tiled online-softmax, grid `[n_heads, B, 1]`"). It already does exactly bidirectional GQA flash attention; wire LFM2's `fa_q/fa_k/fa_v` into its `FullAttnParams` via `AttnFullF32`. **This is the recommended Phase-B1 path** — no new HIP source, just a dispatch wiring + GQA-head-mapping check.
- **Option B (clean sibling, later):** new `kernels/src/attention_q8_0_kv_full.hip` — clone `attention_q8_0_kv.hip` but take `seq_len` as a real argument (not `pos_buf[0]+1`), keep the QK^T + online-softmax (`lines 44-88` are *already* non-causal), drop the decode `pos_buf` scalar. Q8 KV keeps the encoder cheap; only needed if Option A's F32 flash path is a memory/perf problem at seq=512 (it won't be — 512×1024×6 layers is tiny).

Other NEW work:

- **`pos_buf` as a vector, not a scalar.** Decode RoPE/attention consume `pos_buf[0]`. Encoder needs `pos_buf = [0,1,...,seq_len-1]` uploaded once. `rope_f32` already indexes per-row; verify it reads `pos_buf[row]` not `pos_buf[0]` (small kernel tweak if it's scalar-pinned — flagged in lfm2moe recon risk).
- **The 128-d projection head.** New weight tensor `projection.weight [128, 1024]` bias-free. Implemented purely via reused `linear_f16` (§2.2) — *zero new kernel*, just a load path + one GEMM after `post_trunk_norm`. Output **L2-normalized per token** (ColBERT MaxSim assumes unit vectors) — add a `gpu.l2norm_rows_f32(proj, 128, seq)` (trivial new kernel or fold into the projection epilogue).
- **Encoder state struct** `Lfm2ColbertEncoderState` (new, §2.4).

### 2.4 NEW state struct

`Lfm2ColbertEncoderState` replaces the decode-centric `Lfm2MoeState` (`lfm2moe.rs:1113-1156`, which assumes 1-token batch + KV ring + conv ring):

```
h:           [seq, 1024] f32   residual stream (all positions resident)
fa_q/k/v:    [seq, n*hd] f32   full-sequence projections (NOT [kv_dim] per-token)
attn_out:    [seq, n_heads*hd]
conv_scratch:[seq, 3*hidden]   batched conv in_proj output
pos_buf:     [seq] i32         = 0..seq_len  (vector, not scalar)
proj_out:    [seq, 128] f32    final per-token vectors
```

No `KvCache`, no `conv_states` ring buffers. Allocation is `seq`-major; seq ≤ 512 (the real cap — config's 128000 `max_position_embeddings` is vestigial, tokenizer caps at 511/512 per recon).

### 2.5 Tokenizer + query/doc markers (NEW wiring, reuses existing tokenizer)

- Tokenizer loads via the existing path (`hipfire-runtime/src/tokenizer.rs`, `.encode(text) -> Vec<u32>`) from HFQ `metadata_json` or `tokenizer.json`. The SentencePiece/`PreTrainedTokenizerFast` vocab (64402) loads unchanged.
- **NEW: marker injection.** ColBERT prepends a role marker after BOS:
  - Query: `BOS(<|startoftext|>) + [Q](64400) + tokens`, max 32, **query padding** (ColBERT pads queries with mask/pad to fixed length for stable MaxSim; document is not padded).
  - Document: `BOS + [D](64401) + tokens`, max 512, right-truncate.
  - `add_bos_token=true`, `padding_side=right`. These are config constants (`query_marker=64400`, `doc_marker=64401`, `bos`, `pad`), injected in the encode entrypoint, **not** baked into the tokenizer. Do **not** reuse dots.ocr's `IMGPAD_ID/IMG_START_ID` constants (recon risk — different vocab).

---

## 3. Encode API + host-side MaxSim + index layout

### 3.1 Daemon encode command (reuses load/dispatch, bypasses generate)

Add a daemon match arm (`hipfire-runtime/examples/daemon.rs`, alongside the `load` handler at `:1130-1200`), modeled on `hipfire-arch-cohere2moe/examples/encode.rs:1-61` (which already does HFQ→tokenizer→`encode()`→JSON):

```jsonc
// request
{"type":"embed", "role":"document"|"query", "text":"...", "model":"<path>"}
// response
{"type":"embedding", "n_tokens":N, "dim":128,
 "vectors":"<base64 f32 [N*128]>", "token_ids":[...]}  // base64 to avoid JSON float bloat
```

Handler: tokenize + inject marker (§2.5) → `forward_encode` (single batched prefill, no token loop) → `gpu.download_f32(proj_out)` once after prefill (batch download, **not** per-token streaming — recon risk #6) → base64 → return. `temperature`/`top_p`/sampler are absent by construction.

### 3.2 Host-side MaxSim (NEW, CPU, no GPU)

Late-interaction score between a query (Q tokens × 128) and a document (D tokens × 128), all unit-normalized:

```
MaxSim(q, d) = Σ_{i∈query tokens}  max_{j∈doc tokens}  (q_i · d_j)
```

i.e. for each query token, take its best-matching document token (cosine = dot, since unit), and sum over query tokens. Implemented as a host `f32` routine in a new `crates/hipfire-retrieval/` crate (CPU-only, no ROCm dep — this is also the **divorce-first seam**, §7). For a candidate set it's a batched `[Q,128]·[D,128]^T → [Q,D] → rowmax → sum`. Reference behavior = PyLate `rank.rerank()` (recon cites it); we replicate the formula, not the library.

### 3.3 Index layout (Phase B2)

Two-stage retrieval (standard ColBERT):

```
index/
  meta.json            # dim=128, model sha, normalize=l2, n_docs, marker ids
  doc_offsets.bin      # u32 [n_docs+1] prefix-sum of token counts (ragged → flat)
  doc_vectors.f32      # flat [total_doc_tokens, 128] f32 (or int8 + scale for compression)
  doc_token_map.bin    # u32 [total_doc_tokens] -> doc_id  (for MaxSim gather)
  centroids.f32        # OPTIONAL k-means centroids for candidate generation (PLAID-style)
```

- **Phase B2 minimal:** brute-force — score query against *all* docs' token matrices via MaxSim, top-k. Correct, O(n_docs); fine for ≤10⁴–10⁵ docs.
- **Phase B2+ (optional, not in scope):** PLAID-style centroid pre-filter (`centroids.f32`) to prune candidates before exact MaxSim. Listed as a layout hook, deferred.

---

## 4. Parity validation plan

**Oracle:** PyLate / sentence-transformers loading `LiquidAI/LFM2.5-ColBERT-350M` in BF16 on CPU (or a reference GPU), dumping per-token 128-d vectors for a fixed prompt set. This mirrors the project's "llama.cpp/vLLM as KLD oracle" discipline and the `dump_*_hidden_states.rs` pattern.

**What we compare (in increasing strictness):**

1. **Pre-projection hidden states** `[seq, 1024]` after `post_trunk_norm`, per layer if a mismatch appears — isolates backbone (conv/attn/RoPE/norm) bugs from projection bugs. Bisect by layer like the integration RDNA4 regression hunt.
2. **Post-projection, post-L2-norm vectors** `[seq, 128]` — the product output.
3. **End-to-end MaxSim scores** on a small query×doc grid — the user-visible number.

**Metrics + tolerances** (BF16 reference vs our F16/Q8 path, so expect ULP-scale + quant drift, *not* bit-exactness — per the project's "byte-parity is meaningless under stochastic state" rule):

- **Per-token cosine similarity** ours vs oracle: **mean ≥ 0.999, min ≥ 0.995** across all tokens of the fixed set. (Cosine, not L2, because the product *is* a cosine-space embedding and downstream MaxSim only sees angles.)
- **MaxSim score relative error:** `|s_ours − s_oracle| / |s_oracle| ≤ 1%`.
- **Retrieval rank agreement:** top-10 **Recall@10 ≥ 0.98** and **Kendall-τ ≥ 0.95** on a fixed query/corpus.

**Determinism harness:** run parity with **FP32 backbone state + `HIPFIRE_DETERMINISTIC=1`** for the first-pass backbone-isolation comparison (rule: byte/precision claims pin FP32 + deterministic, since Q8 stochastic rounding masks real seam bugs). Once FP32 passes, re-run at the shipping Q8-KV / F16-weight precision and confirm the cosine ≥0.999 band holds.

**Coherence-gate analog:** there's no token-attractor failure mode here (no autoregression), so the DFlash/coherence gates don't apply. The *encoder* gate is a new `scripts/parity-gate-colbert.sh`: encode the fixed prompt set, assert the cosine + MaxSim + rank thresholds against committed oracle dumps, hard-fail on NaN/zero-vector/dim-mismatch. Wire it into the pre-commit hotspot globs for `hipfire-arch-lfm2colbert/**` and `attention_q8_0_kv_full.hip`.

---

## 5. Phased plan

### Phase B1 — minimal encode-only spike (one passage → 128-d/token, parity-checked)
Goal: prove the bidirectional backbone + projection is numerically correct on **one** document.

1. Crate scaffold `hipfire-arch-lfm2colbert` (arch.rs/config.rs from lfm2moe templates).
2. Loader: HFQ + Dir → cfg + weights, reusing dots.ocr BF16→F16 widening loaders; load `projection.weight [128,1024]`. Quantizer `auto_arch_id` stamps arch_id=14.
3. `forward_encode`: dots.ocr loop skeleton + lfm2 conv mixer (batched-causal) + **Option A attention via `attention_dflash.hip`/`AttnFullF32`** + dense SwiGLU + post_trunk_norm + `linear_f16` projection + L2-norm.
4. `pos_buf` vector + RoPE per-row-position fix.
5. `examples/encode_colbert.rs`: text → marker-inject → forward → `[seq,128]` JSON.
6. Parity: dump PyLate oracle for one fixed doc; assert per-token cosine ≥0.999 (FP32-deterministic first, then Q8/F16). Layer-bisect on failure.

**Exit criterion:** one passage, cosine ≥0.999 vs PyLate. No daemon, no index.

### Phase B2 — full corpus index + MaxSim retrieval
1. Daemon `{"type":"embed"}` arm (batch download, base64).
2. `crates/hipfire-retrieval/` CPU crate: MaxSim scorer + brute-force ranker + index read/write (`doc_offsets/doc_vectors/doc_token_map`).
3. CLI: `index build <corpus> -> index/`, `search "<query>" --top-k`.
4. Query-vs-document marker + padding asymmetry ([Q]/[D]).
5. `scripts/parity-gate-colbert.sh`: full cosine + MaxSim-relerr + Recall@10/Kendall-τ on a fixed corpus.
6. (Optional, deferred) centroid pre-filter.

**Exit criterion:** corpus indexed, Recall@10 ≥0.98 + Kendall-τ ≥0.95 vs PyLate on fixed query set.

---

## 6. Effort + the 3 hardest risks

**Effort (single engineer, GPU on the fleet):**

- **Phase B1 ≈ 3–5 days.** Most reuse is mechanical (loaders, conv mixer, dense FFN, projection = one `linear_f16`). The risk-concentrated piece is bidirectional attention — Option A (wire `attention_dflash` via `AttnFullF32`) is ~1 day if GQA head-mapping is clean; the rest is the parity loop + layer-bisect.
- **Phase B2 ≈ 4–6 days.** Daemon arm + base64 plumbing is small; the CPU `hipfire-retrieval` crate + index format + ranker is the bulk; parity-gate + Recall/τ harness ~1 day.
- **Total ≈ 1.5–2.5 weeks** to a parity-passing brute-force retriever. Centroid/PLAID acceleration is out of scope.

**The 3 hardest risks:**

1. **Bidirectional attention correctness (GQA + RoPE positions).** Every shipped attention path is causal single-token; flipping to all-attend-all is *the* new surface. Even via `attention_dflash` (already non-causal), the GQA 16Q/8KV head mapping, the per-row `pos_buf` RoPE (vs scalar `pos_buf[0]`), and the QK-norm-before-RoPE ordering must all match the reference exactly — a 5%-class attention error silently degrades cosine to ~0.95 and tanks retrieval ranks with *no crash*. Mitigation: FP32-deterministic per-layer hidden-state bisect against PyLate before trusting any end-to-end number.
2. **Causal-conv-in-bidirectional-model semantic trap.** LFM2 conv layers are causal by construction. It is *correct* to keep them causal in this encoder (LFM2 retrievers ship exactly this hybrid), but it is intuitively wrong and easy to "fix" into a bug by symmetrizing the conv — which would diverge from the reference. The batched-prefill conv (`conv1d_gated_prefill_f32` or seq-looped decode kernel seeding the ring) must reproduce the left-padded causal window *per position*; an off-by-one in padding shifts every downstream token. Mitigation: parity-check the conv-only sublayer output in isolation first.
3. **Tokenizer/marker + L2-norm fidelity feeding MaxSim.** MaxSim is a *sum of per-token maxes* over unit vectors — it is unforgiving of (a) wrong/missing `[Q]`/`[D]` marker or BOS placement, (b) query-padding handling differences, and (c) un-normalized or mis-axis L2 norm. Any of these passes a "looks coherent" eyeball but silently shifts ranking. Mitigation: assert token-id parity with the HF tokenizer on the fixed set *before* comparing vectors, and gate on Recall@10/Kendall-τ, not just per-token cosine.

*(Plus a non-engineering risk surfaced in recon: **LFM Open License v1.0 caps commercial use at $10M revenue** — verify deployment eligibility before any production ship. Not a code risk; a go/no-go gate.)*

---

## 7. Divorce-first note (backend-pluggable encoder core)

Design the encoder so hipfire/HIP is *one* backend behind a trait, not the only path — so the same ColBERT encoder can later run portable/CPU (or another accelerator) with hipfire as a plugin:

- **Split the crate at the GPU boundary now.** Put **all** math behind a small `ColbertBackend` trait — `qkv_proj`, `qk_norm`, `rope`, `attention_full` (bidirectional), `swiglu`, `rmsnorm`, `projection_128`, `l2norm_rows`, `download`. `forward_encode` is written **once** against the trait; `HipColbertBackend` implements it via the reused `gpu.*` kernels (`rmsnorm_f32`, `rmsnorm_batched`, `rope_f32`, `linear_f16`, the new full-attention dispatch). This mirrors hipfire's existing `dlopen`-FFI seam philosophy (`hip-bridge`/`hsa-bridge`) — the engine never *links* the backend.
- **Keep MaxSim + index + tokenizer-marker logic in `hipfire-retrieval`, a pure-CPU crate with zero ROCm/HIP dependency** (no `rdna-compute`, no `hip-bridge`). It already needs to be host-side; making it backend-agnostic is free. A `CpuColbertBackend` (ndarray/`gemm`) can implement the same `ColbertBackend` trait for CI/no-GPU parity and for the eventual portable build — and doubles as the **parity oracle harness** in §4 (run our CPU backend vs PyLate to isolate kernel bugs from algorithm bugs).
- **No HIP types in the public encode API.** The daemon `embed` request/response and the `[seq,128]` contract are plain bytes (base64 f32) — already backend-neutral. Index files are plain f32/u32 blobs. Nothing in the index or wire format assumes a GPU produced the vectors.

Net: hipfire ships the *fast* `HipColbertBackend`; the encoder core, retrieval, index, and tokenizer marker logic are all GPU-divorced and survive a future port to CPU or any other backend without touching `forward_encode`.

---

### Key file/kernel anchors (for the implementer)
- Reuse-from: `lfm2moe/forward.rs:155-237` (conv+attn mixer), `lfm2moe.rs:199-222` (weight structs), `dots_ocr.rs:1238-1743` (encoder loop), `dots_ocr.rs:1108-1166` (`linear_f16` → projection head), `dots_ocr.rs:556-612,863-909` (BF16→F16 loaders), `carriers.rs:531-586` (`DotsOcrCarrier` template).
- New kernel: `kernels/src/attention_q8_0_kv_full.hip` (Option B) **or** reuse `kernels/src/attention_dflash.hip` via `KernelKey::AttnFullF32` (`dispatch/types.rs:337-338`) (Option A, recommended for B1).
- New crates: `crates/hipfire-arch-lfm2colbert/`, `crates/hipfire-retrieval/`.
- Register: `derive_arch_id` (`safetensors_source.rs:255-283`, lfm2→14 when projection-head present), `REGISTRY` (`lib.rs:54-63`), quantizer `auto_arch_id`.
- Gate: new `scripts/parity-gate-colbert.sh`.

---

## Adversarial review

**Verdict:** feasible-with-corrections

The design is fundamentally sound and lands on the right reference model (dots.ocr one-shot encoder, not the lfm2moe decoder) and the right new-kernel insight (bidirectional attention is genuine new wiring). The two load-bearing claims I verified hold: (1) the existing LFM2 attention path IS causal-only-by-construction — every GPU call in lfm2moe/forward.rs:155-237 is single-token decode (weight_GEMV, scalar pos_buf, attention_q8_0_kv with seq_len=pos_buf[0]+1 implicit causality), so bidirectional attention is real new work the design correctly owns; and (2) attention_dflash_f32 (kernels/src/attention_dflash.hip) IS genuinely non-causal GQA, takes B (queries) and L (keys) as real runtime args with rep=n_heads/n_kv_heads GQA mapping, no internal RoPE, no causal mask — so Option A (route LFM2 Q/K/V through AttnFullF32 -> attention_dflash_f32) composes cleanly for an encoder where B==L==seq_len. The dispatch enum AttnFullF32 -> DflashScalar -> attention_dflash_f32 chain is confirmed real (types.rs:85,394; attention.rs:5801). arch_id=14 is free (derive_arch_id in this tree stops at cohere2_moe=12; gemma4=13 is in memory but not in this branch's switch). The DotsOcrCarrier clone template, registry overlap detection (lib.rs:882-891), and tokenizer/encode-only plumbing (cohere2moe encode.rs) all check out. HOWEVER the design mislabels two pieces of genuine new work as 'reuse', and one numeric config detail must be overridden. With those corrections it is a ~2-3 week build, not a blocked one. The MaxSim/index/retrieval host side and parity plan are well-conceived. The LICENSE is a product go/no-go gate, not a code blocker.

### Corrections (apply before implementing)

1. RoPE: the design (§2.2) says 'reuse rope_f32 verbatim, already indexes per-row' — this is FALSE and self-contradicts §2.3 which flags it as scalar-pinned. rope.hip:20 reads `int pos = pos_buf[0]` (a single scalar) and applies one cos/sin to ALL heads of ONE row. The CORRECT reuse is the already-existing `rope_batched_f32` (norm.rs:631+, kernel rope_batched), which takes a positions VECTOR, launches grid.y=batch_size, one position per row — exactly the encoder need. So: no new kernel, but the design names the wrong function. Replace every 'rope_f32 + pos_buf vector' reference with 'rope_batched_f32 + positions tensor [0..seq_len)'. This eliminates the §2.3 'small kernel tweak' as already-solved.

2. Conv: the reuse table (§2.1) lists the conv mixer as 'REUSE verbatim or near-verbatim'. It is NOT reusable as-is. conv1d_gated_decode_f32 (kernels/src/conv1d_gated_decode.hip) is strictly single-token: it reads a per-channel rolling [K-1] ring buffer and advances it IN PLACE. There is NO batched/prefill conv kernel in the tree (only conv1d_decode/conv1d_gated_decode single-token + conv1d_silu_split variants for qwen35). The design's own Option-A-for-conv ('loop the decode kernel seq times seeding the ring') is the only zero-new-kernel path and IS workable, but it is sequential (seq serial launches) and must be moved from the 'reuse' column to 'new work / wiring'. A proper batched conv1d_gated_prefill_f32 is the clean answer and should be budgeted (~0.5-1 day + parity).

3. rope_theta: the design's config table says θ=1e6 (correct for ColBERT), but lfm2moe's config default and the 350M-MoE it was built for use rope_theta=5e6 (config.rs:12 comment, default_rope_theta()=1e6 is the serde default but the loaded A1B value is 5e6). The new Lfm2ColbertConfig MUST read rope_theta from the ColBERT config.json (1e6) and pass it explicitly to rope_batched_f32 — do not inherit any lfm2moe constant. A wrong θ silently degrades every attention layer's position encoding (risk-1 class: cosine ~0.95, no crash).

4. intermediate_size: the design lists intermediate=6656 as the FFN dim. lfm2moe applies a LLaMA-style auto-adjust (config.rs:96-99): the REAL dense SwiGLU dim is round_to(256, multiplier*2/3*6656) ≈ 4608, NOT 6656. The loaded w1/w2/w3 tensors use the adjusted dim. The ColBERT config must honor block_auto_adjust_ff_dim / block_ff_dim exactly as lfm2moe does, or the projection GEMM shapes mismatch the safetensors. Verify against the actual ColBERT config.json's block_* fields before allocating.

5. Q8 KV in the encoder path: routing through AttnFullF32/attention_dflash_f32 means K/V are consumed as F32 (the kernel signature is all-float). But lfm2moe writes KV as Q8_0 via kv_cache_write_q8_0. For the encoder you do NOT need a KV cache at all — feed fa_k/fa_v as F32 [seq, n_kv*hd] directly into attention_dflash_f32 and DELETE the kv_cache_write_q8_0 calls and the KvCache entirely from the encoder state. The design's §2.4 already says 'no KvCache' — good — but §2.3 Option B ('Q8 KV keeps the encoder cheap') reintroduces it; drop Option B's Q8-KV framing, F32 flash is correct and cheap at seq<=512.

6. derive_arch_id disambiguation: the design's plan to detect ColBERT-vs-decoder by projection-head presence inside derive_arch_id is right, but note this tree's derive_arch_id (safetensors_source.rs:266-270) has NO gemma4=13 entry — so when adding lfm2->14, also confirm whether gemma4 needs adding, and put the lfm2-retriever check BEFORE the `'lfm2'|'lfm2_moe' => 11` arm or it will be shadowed. The HFQ self-routing via auto_arch_id stamp is the robust path; lean on it and treat safetensors-dir routing as best-effort.

### Risks (ranked, highest first)

1. Bidirectional attention numerical correctness (GQA mapping + per-row RoPE + QK-norm-before-RoPE ordering). This is the real new surface. attention_dflash_f32 is non-causal and GQA-correct, but: (a) it was written for DFlash where B(queries) != L(keys); using B==L==seq means re-validating the qi/kv indexing at the diagonal; (b) RoPE must be rope_batched_f32 with positions [0..seq), NOT rope_f32 scalar — getting this wrong gives every-token-position-0 RoPE that looks coherent but tanks retrieval; (c) the reference applies q_norm/k_norm BEFORE RoPE (forward.rs:185-200 order) — must match. A 5%-class attention error degrades cosine to ~0.95 with no crash and silently wrecks ranks. Mitigation as designed (FP32-deterministic per-layer hidden-state bisect vs PyLate) is correct and mandatory.

2. Causal-conv-in-encoder correctness via the seq-looped decode kernel. There is no batched conv kernel, so the only zero-new-kernel path is sequential per-position launches seeding the in-place ring buffer — correct in principle (LFM2 conv IS causal even in the bidirectional retriever) but easy to get wrong: ring-buffer state must reset to zero per sequence, the left-zero-pad of the first K-1 positions must be exact, and an off-by-one shifts every downstream token. Plus seq serial launches at seq=512 over 10 conv layers is a latency wart (not correctness). Parity-check conv-only sublayer output in isolation first, exactly as the design says. Budget a real conv1d_gated_prefill_f32 to remove both the off-by-one risk and the serial-launch cost.

3. Wrong numeric config inheritance from lfm2moe (rope_theta 5e6 vs ColBERT 1e6; auto-adjusted intermediate_size 4608 vs nominal 6656). These are silent: no crash, just degraded vectors. Because the new crate depends on hipfire-arch-lfm2moe and 'reuses' its structs, an implementer is likely to inherit a default. Must read every numeric from the ColBERT config.json and assert against PyLate dumps.

4. Tokenizer/marker + L2-norm fidelity feeding MaxSim. MaxSim (sum of per-query-token max over unit doc vectors) is unforgiving of [Q]=64400/[D]=64401 marker placement, BOS(<|startoftext|>) ordering, query right-padding-to-32 asymmetry, and L2-norm axis/normalization. All pass an eyeball, all shift ranking. Mitigation: assert token-id parity with the HF tokenizer on a fixed set BEFORE comparing vectors, and gate on Recall@10/Kendall-tau, not only per-token cosine. The design has this right; severity stays high because it's the most common silent failure for ColBERT ports.

5. Parity tolerance soundness. mean cosine >=0.999 / min >=0.995 vs a BF16 PyLate oracle at our F16-weight/F32-activation path is reasonable and consistent with project discipline (FP32+HIPFIRE_DETERMINISTIC=1 for first-pass backbone isolation, then re-run at ship precision). One gap: the oracle should be PyLate/sentence-transformers in the SAME dtype the math runs in, and the design should commit the CpuColbertBackend (§7) as the algorithm-vs-kernel discriminator BEFORE trusting the GPU number — otherwise a backbone-math bug and a kernel bug are indistinguishable. Add an explicit CPU-backend-vs-PyLate parity step to Phase B1.

6. Effort realism. Phase B1 (3-5 days) is slightly optimistic given the corrections: RoPE is actually free (rope_batched_f32 exists), but conv needs either careful seq-looping or a new prefill kernel (+~1 day), and the FP32-deterministic per-layer bisect loop against PyLate is itself 1-2 days of fiddly work. Realistic B1 is 5-7 days; total 2-3 weeks. Not a blocker, just re-baseline.

7. LFM Open License v1.0 commercial cap at $10M revenue (Section 5b). This is a product go/no-go gate, not a code risk. It does not block building/validating the port. Must be cleared before any production/commercial ship; flag to the product owner now so engineering effort isn't sunk if the answer is no-go.

