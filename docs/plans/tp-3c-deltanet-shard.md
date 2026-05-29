# Stage 3c — DeltaNet (LinearAttention) sharding

**Status: ✅ DONE + validated (2026-05-29, uncommitted) — and it WINS decode.**
Full FA + DeltaNet sharding (attention, recurrent state, FFN). Validated
`tp_attn_parity --slice` TP=2↔TP=1: 0.8B fp32 **3.846e-6 / 32-32**, 27B-3.6 fp32
**1.050e-6 / 32-32**. Replicated regression unchanged (2.604e-6).

**Decode (27B-3.6, 3 fresh processes): TP=2 = 39.0 tok/s vs TP=1 = 35.1 → +11%.**

**CRITICAL CORRECTION (the win was hiding behind a self-inflicted bug):** an
earlier measurement showed decode REGRESSING (−6/−15/−18% as sharding grew) and
I wrongly concluded "TP decode is collective-latency-bound, can't win." That was
false — the cost was a REDUNDANT per-layer `device_synchronize` in
`forward_scratch_tp` (a full GPU drain, 128/token at 3c-B) added during pos_buf
debugging under the false belief that kernels run on the default stream. They run
on `active_stream` (dispatch.rs:1150), same as the all-reduce, so the
gemv→all_reduce→add chain is already stream-ordered and RCCL handles cross-rank
sync. Replacing the device_synchronize/stream_synchronize with the lean
`tp_allreduce_add` (just all_reduce + add, no host sync) turned −18% into +11% —
a +36% swing. **The all-reduce was never the bottleneck; my sync was.** Lesson:
profile/measure before asserting a perf cause.

---

Original scope below. Goal: shard the 48/64 (27B) / 18/24 (0.8B) DeltaNet layers
so TP gives a real win. Acceptance (MET): `tp_attn_parity --slice` stays fp32
~1e-6 / 32-32 with DeltaNet sharded, and 27B decode TP=2 beats TP=1.

## Head geometry (verified)
- `linear_key_head_dim = linear_value_head_dim = 128` (both models).
- 0.8B: `linear_num_key_heads=16`, `linear_num_value_heads=16` (ratio 1).
- 27B-3.6: `linear_num_key_heads=16`, `linear_num_value_heads=48` (ratio **3**).
- `k_dim = n_key_heads·128`, `v_dim = n_value_heads·128`,
  `qkv_dim = 2·k_dim + v_dim` (wqkv output = `[q(k_dim) | k(k_dim) | v(v_dim)]`).
- **Shards cleanly by value head** at tp=2/4, key heads following the ratio:
  27B tp=2 → 8 key + 24 value/rank (ratio 3 ✓); tp=4 → 4+12 (✓); 0.8B tp=2 → 8+8.
  ShardConfig must validate `n_value_heads % tp == 0`, `n_key_heads % tp == 0`, and
  `n_value_heads/tp ÷ n_key_heads/tp == n_value_heads/n_key_heads` (ratio preserved).

## Per-weight sharding map (LinearAttention layer)
Read `run_dn_layer_body` + the DeltaNet load branch first (re-grep; lines drift).

| weight | shape | shard | notes |
|--------|-------|-------|-------|
| wqkv `in_proj_qkv` | `[2·k_dim + v_dim, dim]` | **column, 3-sub-range** | local key heads from the q block + the k block, local value heads from the v block → **NEW multi-range row slicer** (concat 3 `slice_quant_rows` sub-ranges). AWQ `[dim]` unchanged. |
| wz `in_proj_z` | `[v_dim, dim]` | column by value head | contiguous rows `[v_range·128]`. AWQ full. |
| w_beta `in_proj_b`, w_alpha `in_proj_a` | `[n_value_heads, dim]` | column by value head | 1 row/value head → contiguous rows `[v_range]`. AWQ full. |
| a_log, dt_bias | `[n_value_heads]` F32 | by value head | slice `[v_range]` (F32 gather, small). |
| conv_weight `conv1d.weight` | `[qkv_dim · kernel]` F32 flat | **3-sub-range by channel** | local q/k/v channels' kernel rows (same 3-range as wqkv, F32). |
| norm_weight `linear_attn.norm` | `[value_head_dim=128]` | **shared, full** | per-head-dim RMSNorm weight (like q_norm). |
| wo `out_proj` | `[dim, v_dim]` | **row** (`Cols` by value head) | input = local value heads → partial → all_reduce → add. v_dim/tp must be %256 (27B 6144/2=3072 ✓; 0.8B 2048/2=1024 ✓). AWQ `[v_dim]` slice `[c0,c1)`. |
| attn_norm, ffn_norm | `[dim]` | full | operate on s.x. |
| FFN w_gate/w_up/w_down | (dense FFN) | like FA (col/col/row) | reuse the 3b FFN slice + sharded tail. |

## Recurrent-state sharding (the real memory win)
`DeltaNetState` per delta layer: `s_matrices[l]` = `[n_value_heads·128·128]`,
`s_scales[l]` = `[n_value_heads]` (Q8: `[n_value_heads·128]`),
`conv_states[l]` = `[qkv_dim·(kernel−1)]`. ALL derived from config head counts →
**a local DeltaNet config makes `DeltaNetState::new_with_quant` allocate local
state automatically** (each rank holds + evolves ONLY its value heads' S + its
local conv channels). Per-head independent → no cross-rank dep until wo.

## Compute path
- **Local config:** extend `local_attn_config` (or a sibling) to ALSO shrink
  `linear_num_value_heads`, `linear_num_key_heads`, and now `hidden_dim`
  (the 3b "keep hidden_dim full" gotcha LIFTS because the DeltaNet FFN is sharded
  too in 3c — but re-derive `Qwen35Scratch` sizing carefully: dn_qkv, dn_z, dn_q/k/v,
  dn_attn_out, dn_normed, gate_ffn/up/ffn_hidden all become local; verify no
  full-FFN/full-head consumer remains).
- `run_dn_layer_body` uses `config.linear_num_*` throughout → local config makes
  conv1d_silu_split / fused_qk_l2_norm_scale / repeat_interleave_qk /
  gated_delta_net / gated_norm all run on local heads for free. The wqkv→qkv split
  (k_dim/v_dim) is local. **RoPE-free** (DeltaNet has no positional rotation), so
  no head-index subtlety like FA.
- **Phase split (mirror FaPhase):** add `DnAttn` (run through wo as a PARTIAL into
  `s.o`, NON-residual, return) and `DnFfnShard` (sharded FFN partial w_down → s.o).
  `forward_scratch_tp` orchestrates DeltaNet exactly like FA:
  `DnAttn → device_sync → all_reduce(s.o) → stream_sync → add(s.x) → DnFfnShard →
  device_sync → all_reduce(s.o) → stream_sync → add(s.x)`.
  Currently DeltaNet wo + FFN both do `weight_gemv_residual`/`swiglu_residual` into
  s.x — change the sharded path to partial-into-s.o (reuse the FA `run_fa_ffn_*`
  decomposition: silu_mul + non-residual `weight_gemv`).

## Loader
Extend `load_weights_tp` to also handle the `LayerWeights::DeltaNet` arm:
slice wqkv (3-range), wz/w_beta/w_alpha (value-head rows), a_log/dt_bias (F32
value-head gather), conv_weight (3-range F32 channel gather), wo (Cols by value
head), FFN (col/col/row) — free the full buffers. **New helpers:**
`slice_quant_rows_multi(data, m, &[(start,end)])` (concat sub-ranges) for wqkv;
a F32 `[n·stride]` channel gather for conv_weight; F32 element gather for
a_log/dt_bias. The harness builds `dn_states[]` from the local DeltaNet config.

## ShardConfig additions
`dn_value_head_range(rank, n_value_heads)`, `dn_key_head_range(rank, n_key_heads)`
(contiguous splits), and a `validate_deltanet(n_value_heads, n_key_heads)` ratio
check. Reuse the row/col-range helpers conceptually (value head → rows for
wz/etc., cols for wo).

## Validation
1. `tp_attn_parity --slice` on 0.8B fp32 → must stay ~1e-6 / 32-32 (now exercises
   sharded DeltaNet on every non-FA layer). RANKDIFF diag if ranks drift.
2. 27B-3.6 `--slice` fp32 → same bar.
3. Re-measure decode tok/s on 27B: with DeltaNet sharded, TP=2 should ≥ TP=1
   (was 32.9 vs 35.1). Memory/rank should also drop (state + weights sharded).

## Risks / watch-items
- **wqkv 3-range slice** correctness — caught by parity (blows up if wrong).
- **conv_weight channel gather** (F32, depthwise per qkv channel) — must match the
  qkv `[q|k|v]` ordering and the local channel selection exactly.
- **state local sizing** — `new_with_quant(local_dn_config)` must match the local
  compute head counts; mismatch → OOB (HIP 700) like the 3b scratch bug.
- **Scratch sizing** with the now-shrunk hidden_dim + local dn_* buffers — audit.
- Q8 KV/state amplification (from 3b memory) still applies at q8; validate at fp32.

## First implementation steps
1. ShardConfig dn head ranges + validate. 2. `slice_quant_rows_multi` + conv/F32
gathers (+ CPU tests). 3. local DeltaNet config. 4. load_weights_tp DeltaNet arm.
5. DnAttn/DnFfnShard phases + forward_scratch_tp DeltaNet orchestration.
6. Local `dn_states[]` + scratch in harness. 7. Parity 0.8B → 27B → decode timing.
