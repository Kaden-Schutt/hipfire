# Stage 3e — expert-parallel MoE for A3B (scope)

**Status:** scoping (2026-05-29). Goal: shard the 256 routed experts of the
35B-A3B MoE across GPUs so A3B (a) FITS with headroom and (b) gets the decode
win (each rank reads ½ the expert weights — the same bandwidth lever that gave
3c its +11% decode win). Model: `~/.hipfire/models/qwen3-35b-a3b.mq4-awq` (22 GB).

## Why EP, and which comm primitive

A3B = 256 experts, top-8/token, `moe_intermediate=512` (small per-expert FFN),
always-on shared expert, router `[256, hidden]`. The 256 experts are ~the whole
35B (only 3B active/token). Attention + DeltaNet are the same hybrid as dense 27B.

**Comm fork (the key decision):**
- **All-to-all EP (textbook):** each rank owns 256/tp experts; dispatch each
  token's hidden to the rank(s) owning its routed experts, compute, gather back.
  Comm = all-to-all of activations. **REJECTED for gfx1201**: per
  [[project_rccl_on_gfx1201]] RCCL all-to-all *ties/loses* the host ring on this
  box; and it needs new dispatch/gather kernels.
- **All-reduce EP (CHOSEN):** each rank owns 256/tp experts; the router runs
  REPLICATED (every rank computes the same global top-8 + renorm); each rank
  computes only the routed experts IT OWNS into a `[N×dim]` partial; rank 0 also
  adds the shared expert; then **all-reduce the partials** → full MoE output →
  add to residual. Comm = 1 all-reduce `[N×dim]` per MoE layer — the SAME proven
  primitive that won 3c decode and that 3d/forward_scratch_tp already wires
  (`tp_allreduce_add` / `tp_allreduce_add_batched`). Reuses everything; leanest
  comm on gfx1201; delivers the memory + decode-bandwidth win.

This is "tensor-parallel over the expert dimension via all-reduce" — not
classic EP, but the right fit for a 2–4 card gfx1201 box.

## The kernel-light trick (no MoE-kernel rewrites for v1)

The indexed MoE gemv (`gemv_hfq4g256_moe_gate_up_k8_indexed` etc.) reads expert
pointers from a `[num_experts]` device pointer table via the top-8 GLOBAL ids,
and `moe_down_combine_k8*` weights each expert's output by `topk_weights`. On
rank r we make the kernels compute only owned experts WITHOUT touching them:

1. **Pointer table per rank `[256]`:** owned experts → their real resident
   pointer; **non-owned → a DUMMY resident pointer** (rank r's first owned
   expert). Non-owned slots compute garbage but never deref unmapped memory.
2. **Mask `topk_weights` on rank r:** zero the weight of any top-8 expert rank r
   does NOT own (a tiny element-wise multiply by an ownership mask, after the
   replicated softmax+renorm). The combine then adds 0 for the garbage slots.
3. **All-reduce** the per-rank `[N×dim]` partial → Σ over all 8 experts with
   their correct GLOBALLY-renormalized weights (renorm is replicated, identical
   on every rank). Exact.

Net: rank r's partial = Σ_{owned ∩ top8} w_e·expert_e(x); all-reduce → full MoE.
Shared expert + router are replicated; shared runs on rank 0 ONLY (so the
all-reduce counts it once). NO changes to the MoE GEMV/combine kernels for v1.

## Sharding map (per MoE layer)
| weight | shard | notes |
|--------|-------|-------|
| router `[256,hidden]` | **replicated** | every rank needs global top-8 |
| shared_expert (gate/up/down), shared_expert_gate | **replicated, run on rank 0 only** | always-on; counted once via all-reduce |
| experts[e] (gate_up `[2·512,hidden]`, down `[hidden,512]`) | **owned by `expert_to_rank[e]`** | load only owned; ½ the expert weight/bandwidth per rank at tp=2 |
| attn_norm/ffn_norm, attention (wq..wo / wqkv..) | replicated OR TP'd | v1: replicate attention (focus the win on experts); later compose with 3b/3c attn sharding |

`ShardConfig.expert_to_rank` (already exists) is the ownership map;
`ExpertAssign::Stride` (e%tp) balances load well for random routing.

## Refined implementation plan (from the 2026-05-29 code deep-dive)

Single-GPU MoE FFN is `moe_ffn_decode_impl` (qwen35.rs ~3430). Its GPU fast path
(k=8, mq4-awq, `use_gpu_topk`): rotate x → `fused_qkvza_hfq4g256` (router +
shared_gate + shared.gate + shared.up) → `softmax_f32` + `moe_topk_renorm_k8` →
shared-expert down via `gemv_hfq4g256_residual_sigmoid_scaled_gpu` (+= x_residual)
→ routed `gemv_hfq4g256_moe_gate_up_k8_indexed` → `fused_silu_mul_rotate_mq_batched_for`
→ `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` → `moe_down_combine_k8_batched`
(+= x_residual). The indexed gemvs read `ffn.expert_gate_up_ptrs[topk_id]` (global
id → device ptr); `experts[0].down` is the shared-AWQ-scale representative.

**EP reuse (minimal new code):**
- **Loader (`load_weights_tp` MoE arm):** load_weights already loads all 256
  experts full. Add a 3rd slice-loop (after the FullAttn + DeltaNet loops) that,
  for each `DeltaNetMoe`/`FullAttnMoe`, **frees non-owned experts + COMPACTS owned
  to the front** (so `experts[0]` = first OWNED — valid representative), then
  rebuilds `expert_gate_up_ptrs`/`expert_down_ptrs` `[2·n_exp]`: owned global id →
  its compacted ptr; **non-owned → a shared ZEROED buffer** (zeroed bytes read as
  HFQ4 → scale 0 → gate_up output 0 → silu·=0 → rot=0 → down output 0; contributes
  0 with NO masking kernel). down ptr for non-owned = experts[0].down (harmless,
  rot=0). Router/shared/attention stay FULL (replicated). Store the zero buffer on
  `Qwen35Weights` (one field, shared across layers — same expert shape) so it
  outlives the ptr tables; free in `free_gpu`.
- **`moe_ffn_decode_impl` + `skip_shared: bool`:** gate the shared-expert-down
  section (~3642-3667) on `!skip_shared`. Default false (existing callers
  unaffected). A thin `moe_ffn_decode_ep` wrapper passes `x_residual = s.o`
  (pre-zeroed partial) + `skip_shared = rank != 0`.
- **Forward (`forward_scratch_tp` MoE arms):** v1 replicates attention. Need the
  attention for the 2 MoE variants → COPY the attention sequence from the inline
  arms (DeltaNetMoe ~10455-10565: fused_qkvza→sigmoid_alpha→conv→qk_norm→
  gated_delta_net→gated_norm→wo into s.x; FullAttnMoe analogous) into helpers
  `run_dn_moe_attn`/`run_fa_moe_attn` (don't refactor the inline arms — production
  safety). Per MoE layer, each rank: run_*_moe_attn → s.x (replicated, identical) →
  zero s.o → `moe_ffn_decode_ep(x_norm=rmsnorm(s.x), x_residual=s.o, skip_shared=rank!=0)`
  → `tp_allreduce_add(s.o → s.x)`. rank-0 includes shared in s.o; all-reduce sums
  shared(rank0) + Σ owned-routed = full MoE. Final norm+lm_head on rank 0.
- **Harness:** `tp_attn_parity` already loads via `load_weights_tp` + runs
  `forward_scratch_tp`; point it at `qwen3-35b-a3b.mq4-awq`, confirm the MoE arms
  hit, validate TP=2↔1 decode parity (fp32-equiv) + argmax, then decode tok/s.
  q8 KV ok; DeltaNet state fp32 for the parity gate (3c). NOTE config: A3B MoE EP
  v1 uses the FULL config (attention replicated), NOT `local_attn_config`.

**Risk on the zero-expert trick:** assumes zeroed HFQ4 bytes dequant to exactly 0
(symmetric quant, f16 scale 0x0000 = +0.0 → yes). The parity test confirms it; if
it ever fails, fall back to a tiny `moe_mask_topk_weights_by_owner` kernel
(`w[k] *= owns_mask[topk_id[k]]`) + real (non-zero) dummy pointers.

## Build steps (decode-FIRST — simplest + where A3B's win is)
1. **ShardConfig expert helpers** (THIS step): `experts_per_rank`,
   `owns_expert(rank,e)`, `local_expert_ids(rank)`, `validate_moe` + CPU tests.
2. **load_weights_tp MoE arm** (`LayerWeights::DeltaNetMoe`/`FullAttnMoe`): load
   only `local_expert_ids(rank)` experts; build the `[256]` pointer table with
   dummy fill for non-owned; replicate router/shared/attention. Free non-owned.
3. **Decode MoE in `forward_scratch_tp`**: add the MoE layer arms. Attention
   replicated (run_*_layer_body Full into s.x), then MoE FFN: replicated
   router+top-k+softmax+renorm → **mask topk_weights to owned** → indexed
   gate_up/down gemv (dummy ptrs for non-owned) → combine into `s.o` partial;
   rank 0 also adds shared expert to `s.o`; → `tp_allreduce_add(s.o→s.x)`.
4. **Validate**: `tp_attn_parity` on A3B (new model arg) — TP=2 vs TP=1 decode
   logit parity at fp32-equivalent + argmax. (q8 KV fine; DeltaNet state fp32 for
   the gate per 3c.) Then decode tok/s TP=1 vs TP=2 (expect a win: ½ expert
   bandwidth + tiny `[1×dim]` all-reduce).
5. **Batched prefill EP** (later): extend `forward_prefill_chunk_tp` with the MoE
   arms (Path-1 indexed or Path-2 grouped, masked the same way) → `[N×dim]`
   partial → `tp_allreduce_add_batched`. Note the 3d lesson: prefill all-reduce
   scales with N, but EP runs only 1 all-reduce/layer and halves the (large)
   expert GEMMs, so it may break even where dense prefill TP lost.

## Risks / watch-items
- **topk-weight masking** must happen AFTER the global renorm (renorm is over the
  full top-8; masking before would re-weight wrong). A tiny mask-multiply kernel
  or fold into a new `moe_topk_renorm_k8_masked`.
- **Dummy-pointer slots** compute real garbage GEMVs (wasted flops, ≤ non-owned
  fraction) but are zeroed in combine — correctness OK, minor compute waste. A
  later kernel that skips null pointers removes the waste.
- **Load balance:** Stride assignment + random routing ≈ even; a token whose top-8
  all land on one rank is rare but makes the other rank idle that token. Fine for
  v1 (we all-reduce regardless).
- **Paged experts** (`paged_experts=true`): out of scope for v1 (EP assumes
  resident owned experts). The 22 GB awq model is resident on 32 GB cards.
- **Attention replicated in v1** → no attention bandwidth win, but the experts
  are the bulk. Compose with 3b/3c attn sharding in a later step.
