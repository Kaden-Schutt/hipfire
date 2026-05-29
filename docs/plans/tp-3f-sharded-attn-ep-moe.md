# Stage 3f — sharded-attention EP-MoE (the A3B decode win)

**Status:** scoping (2026-05-29). 3e validated EP-MoE decode correct (TP=2↔1
32/32, 7.5e-7) but decode tok/s LOSES −35% (57 vs 88) because attention is
REPLICATED (the dense bulk of decode bandwidth — A3B is 30 DeltaNetMoe + 10
FullAttnMoe of 40) and the zero-expert dummies still execute. 3f shards the
attention (the real lever, like 3c's +11% dense decode win) + skips non-owned
expert compute, to flip the loss into a win.

## The two levers (impact-ordered)
1. **Shard the MoE-layer attention (BIG lever).** A3B decode bandwidth is
   dominated by the dense per-layer attention weights (DeltaNet wqkv/wz/wo+conv ×30,
   FA wq/wk/wv/wo ×10), read every token, currently REPLICATED (full on both
   ranks). Sharding them (each rank ½ the heads → partial wo → all-reduce) halves
   that bandwidth — the same lever that won 3c (+11%) on the dense 27B. Reuses the
   3b (FA) + 3c (DeltaNet) slicing + sharded-attention machinery verbatim; only the
   layer-WEIGHTS variant differs (DeltaNetMoe/FullAttnMoe vs DeltaNet/FullAttn).
2. **Skip non-owned expert gemvs (small lever).** Experts are sparse (8/256
   active), so this is minor bandwidth, but the current zero-expert dummies waste
   4 gemvs/layer/rank. A `moe_mask_topk_weights_by_owner` kernel (or topk
   compaction to owned-only) lets the indexed gemv skip non-owned. Do AFTER (1).

## Build (gate behind HIPFIRE_EP_SHARD_ATTN so 3e's validated replicated path
## stays the default until 3f validates; then make it the default).

1. **`load_weights_tp` MoE-attention slicing.** The FA/DN slice loops currently
   `continue` past the *Moe variants. Extend them (or add 2 MoE loops) to slice
   the MoE-layer attention IDENTICALLY to 3b/3c: FullAttnMoe → wq rows / wk,wv rows
   / wo cols (reuse `wq_row_range`/`kv_head_range`/`wo_col_range` + `load_weight_tensor_sliced`);
   DeltaNetMoe → wqkv 3-sub-range / wz,w_alpha,w_beta by value head / conv_weight
   channel gather / a_log,dt_bias gather / wo cols (reuse `slice_quant_rows_multi`/
   `gather_f32_ranges` + the 3c DeltaNet slice block). Experts stay EP-sharded
   (`shard_moe_experts`, unchanged). The FFN (router/shared) stays full (replicated).
2. **Config:** use `local_attn_config` for MoE (shrinks n_heads/n_kv_heads/linear
   heads; `num_experts`/`moe_intermediate` untouched → EP unaffected; `hidden_dim`
   is unused under MoE so its shrink is moot). Harness: for MoE, switch from the
   3e full-config to `local_attn_config`. dn_states/kv/scratch sized local.
3. **Sharded MoE-attention helpers.** Add a phase to `run_dn_moe_attn`/`run_fa_moe_attn`
   (or new `_sharded` variants): attention on LOCAL heads → partial wo into `s.o`
   (NON-residual) → return. Mirrors `run_dn_layer_body`'s DnPhase::Attn /
   `run_fa_layer_body`'s FaPhase::TpAttn. (Keep the per-weight `weight_gemv_prerotated`
   dispatch — the 3e NaN fix — for the wqkv/wz/w_beta/w_alpha; A3B's tiny
   w_alpha/w_beta aren't MQ4.) The DeltaNet recurrent state shards per local value
   head (DeltaNetState from local config), exactly as 3c.
4. **`forward_scratch_tp` MoE arms:** sharded attn (per rank → s.o) → `tp_allreduce_add`
   (s.o→s.x) → EP MoE FFN (run_moe_ffn_ep → s.o) → `tp_allreduce_add` (s.o→s.x).
   2 all-reduces/layer (like 3c) — the attention bandwidth halving should outweigh
   the extra all-reduce, as it did for 3c decode.
5. **Skip non-owned experts** (lever 2): `moe_mask_topk_weights_by_owner` kernel
   (zero topk_weight for non-owned) + make the indexed gemv skip zero-weight slots
   (or compact topk to owned). Removes the dummy-gemv waste.
6. **Validate:** `tp_attn_parity` on A3B — TP=2↔1 decode parity fp32 (expect
   ~1e-6, sharded DeltaNet state like 3c) + argmax + decode tok/s (expect a WIN vs
   the −35% of 3e: ½ attention bandwidth + ½ DeltaNet state/rank).

## Risks
- DeltaNet state sharding under TP at q8 amplifies (3c finding) — validate at fp32.
- The sharded DeltaNet MoE-attention is the intricate part (wqkv 3-range + state
  shard); reuse the 3c code path closely. Expect a debug pass (the 3e NaN was a
  dtype-dispatch miss — the same class lurks here).
- 2 all-reduces/layer × 40 layers; if the attention-bandwidth win doesn't outweigh
  (A3B attention may be smaller per-layer than 27B), decode could still lose —
  measure. If so, EP's value remains memory/capacity + batched throughput.
- This is a ~3c-sized build (loader + sharded helpers + arms + a kernel + validate)
  and likely a debug cycle — best tackled with fresh context.
