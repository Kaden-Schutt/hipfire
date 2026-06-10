# DeepSeek-V4-Flash EP port (Ship 6 substrate-EP, #16)

Mirror of the MiniMax EP port (`forward_ep` + shard-aware load), with DeepSeek's
extra structure. Model: `deepseek-v4-flash.mq2lloyd` (81 GB, on hipx → transfer
to hiptrx; an MTP variant `deepseek-v4-flash-mtp.mq2lloyd` 1.9 GB exists for
later spec-decode). Needs EP N=4 on hiptrx (doesn't fit one 32 GB card).

## Arch structure (forward.rs)
- Decode lowered: `Deepseek4Bindings` (run_attend=`ds4_attn_block` MLA; run_moe=
  `ds4_moe_block`), `ds4_lower_program() = [Attend, Moe]`, `decode_step_body_lowered`.
- `ds4_moe_block` = `mhc_pre` → if !skip_ffn { `ffn_stub` (SHARED expert → ffn_out)
  + (`ffn_hash_routed` if layer<num_hash_layers else `ffn_routed`) (routed → ffn_out) }
  → `hc_ffn_mix` (folds `state.ffn_out` into `state.residual_streams`).
- Routed experts: packed blobs `expert_gate_up_blob` / `expert_w2_blob` +
  `expert_gate_up_ptrs` (base+e*stride ptr table) — SAME packed layout as MiniMax.
- `n_shared_experts=1`, `n_routed_experts=256`, top-`num_experts_per_tok` (6),
  `routed_scaling_factor`, biased-vs-unbiased router scores (`moe_route` +
  `deepseek4_moe_topk_bias_aware_f32`). `num_hash_layers` early layers are
  shared-expert-ONLY (no routed selection → no cross-rank reduce).

## EP wiring (per-arch, mirrors qwen35/minimax)
1. **`ffn_routed` gains `routed_out: Option<&GpuTensor>`** — redirect the routed
   combine target from `state.ffn_out` to the zeroed partial. Shared (`ffn_stub`)
   stays writing `ffn_out` (replicated per rank). `ffn_hash_routed` is shared-only
   → never redirects (its layers have no routed combine).
2. **`Deepseek4Bindings::run_moe_ep`** = `mhc_pre` + `ffn_stub` + `ffn_routed`
   (routed→partial) — i.e. ds4_moe_block WITHOUT `hc_ffn_mix`. For hash layers
   (shared-only) the partial stays zero (all-reduce of zero = identity, or skip).
3. **`ep_add_into_residual`** = `ffn_out += partial` (all-reduced routed) THEN
   `hc_ffn_mix(ffn_out → residual_streams)`. (DeepSeek's mix can't run until the
   full FFN output is assembled, so it moves here from inside the block.)
   ⚠ Refactor `ds4_moe_block` so the lowered (non-EP) `run_moe` path is unchanged
   (still does stub+routed+mix), while EP splits at the mix. Keep byte-identical
   for the non-EP default (gate the split).
4. **`DeepseekV4State` per rank** + a `[hidden]` routed partial per rank.
   Binding holds `&mut State` → build N bindings via `state_per_rank.iter_mut()`
   (disjoint). The executor (`run_layer_program_ep`) is generic — already works.
5. **`forward_ep`** N-rank decode driver: embed/pos replicated, Attend replicated
   (MLA full per rank), Moe all-reduce-EP'd, final norm+head on rank 0.
6. **Shard-aware load**: the expert blob pack (arch.rs ~231-255 + the per-layer
   `ffn.experts.{e}` loop) reads all experts but uploads only rank-owned into a
   compact blob; non-owned ptr → zeroed gate_up dummy (Lloyd centroids zero → 0).
   Shared expert + MLA + router + lm_head loaded full (replicated). Mirror
   `MiniMaxWeights::load(.., Some((shard, rank)))`.
7. **`ep_deepseek4.rs`** example (mirror `ep_minimax.rs`): init_tp(4), shard-load
   per rank, forward_ep greedy decode, print + FNV.

## Validation (coherence-gated, hiptrx N=4)
- Coherent generation (factual prompt) on gfx1201 ×4; FNV vs unfused path.
- Check the hash-routed early layers (shared-only) produce correct output under EP
  (partial=0, shared replicated).
- Per-card VRAM headroom (81 GB / 4 ≈ same ~24 GB/card class as MiniMax).
- Decode tok/s baseline; peer-direct all-reduce already default for prefill,
  RCCL for decode (shared executor).

## Risks vs MiniMax
- The `hc_ffn_mix` split (mix must run AFTER the all-reduce) is the main new seam.
- Hash-routed layers: ensure the EP path handles "no routed experts" cleanly.
- `&mut State` binding borrows (disjoint iter_mut — fine, but watch the closure).
- MLA state size (latent KV) replicated per rank — VRAM check.
- routed_scaling_factor / biased-vs-unbiased router preserved through the redirect.
