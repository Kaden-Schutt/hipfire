# TP Stage 3c (DeltaNet sharding) → 3d (batched-TP prefill) — goal/handoff

**WHY (the decision).** FA-only TP is validated correct (27B-3.6 parity 1.351e-6,
32/32) but gives NO speedup on the Qwen3.5/3.6 hybrid — decode TP=2 measured **−6%**
vs TP=1 on 27B. Root cause: DeltaNet is **48/64 layers (75%)** and runs **REPLICATED**
(full compute on every rank, no memory/bandwidth win) while only the 16 FA layers
shard. The real win is in the DeltaNet layers. Do **3c (DeltaNet sharding) FIRST**,
then **3d (batched-TP prefill)** — by which point both FA and DeltaNet shard, so
prefill should finally show a real win.

## Environment / state
- Worktree `/home/kaden/hipfire/.claude/worktrees/tp-scoping-ds4`, branch
  `worktree-tp-scoping-ds4` (local-only). Base = origin/master + PR #352 merge
  (62fe152e) — do NOT rebase to drop it.
- Box: 4× R9700 gfx1201 (hiptrx). GPU lock: `source scripts/gpu-lock.sh` then
  `gpu_acquire "<label>" && { …; gpu_release }`. Goes stale if a run is interrupted
  before `gpu_release` — `rm -f /tmp/hipfire-gpu.lock` to clear.
- Commits: 624fc664 (Stage 3 forward_scratch_tp), ebf52a3e (3b FA slicing),
  664063c4 (AWQ slicing + 27B validation + decode timing). Working tree clean.
- **HARD CONSTRAINTS:** no push / no PR / no review-visible action without explicit
  go-ahead. Local commits ONLY when the user says "commit" (user prefers
  `git commit --no-verify`, skipping the coherence gate). Do not rebase away #352.

## What's already built (reuse — all in crates/hipfire-arch-qwen35/src/qwen35.rs)
- `slice_quant_rows` / `slice_quant_cols` / `quant_group_size` — format-agnostic byte
  slicers (`row_bytes=data.len()/m`, `group_bytes=row_bytes*gs/k`); 4 CPU tests
  (`tp_slice_tests`).
- `load_weight_tensor_sliced` — pread path; slices the weight AND its AWQ sidecar
  (`[k]` F16: Rows keep full, Cols slice `[c0,c1)`).
- `load_weights_tp(hfq, config, gpu, shard, rank)` — load-then-slice; currently slices
  FA wq/wk/wv (rows) + wo (cols) + w_gate/w_up (rows) + w_down (cols). **3c extends this
  to the DeltaNet layers.**
- `local_attn_config(config, shard)` — shrinks n_heads/n_kv_heads ONLY (NOT hidden_dim:
  scratch is shared with the replicated DeltaNet FFN; shrinking it OOMs). 3c needs a
  local config that ALSO shrinks `linear_num_value_heads`/`linear_num_key_heads` once
  DeltaNet is sharded, and can then shrink hidden_dim (DeltaNet FFN also sharded in 3c).
- `forward_scratch_tp(gpus, shard, weights, configs[], token, pos, kv_caches[],
  dn_states[], scratches[], fa_masks: Option)` — per-rank single-token TP forward.
  FA: TpAttn(mask None ⇒ sliced) → all_reduce(s.o) → add → TpFfnShard → all_reduce →
  add. DeltaNet: `run_dn_layer_body` REPLICATED (← 3c changes to sharded). fa_masks=None
  ⇒ sliced mode.
- `run_dn_layer_body(gpu, weights, config, layer_idx, delta_layer_idx, dn_state, s,
  hidden_rb)` — the DeltaNet layer body. **THIS is what 3c shards.**
- `run_fa_ffn_body_sharded` / `run_fa_ffn_gate_up` — FA sharded FFN tail (factor a
  DeltaNet analog, or generalize).
- Harness `examples/tp_attn_parity.rs` — `--slice` (HIPFIRE_PARITY_SLICE=1) loads via
  load_weights_tp + local config + no masks; force-feeds the reference token path;
  asserts TP=2↔TP=1 logit parity (fp32 default) + 32/32 argmax; prints decode tok/s.
  Diag knobs: `HIPFIRE_PARITY_{NTOK,REFREF,REFSTREAM,RANKDIFF,KV,STATE,Q8}`.
  Acceptance bar: fp32 KV+state must stay ~1e-6 / 32-32.

## Stage 3c plan — shard value/key heads + the recurrent state (Megatron on LA)
Read `run_dn_layer_body` first; re-grep line numbers (they drift).
- **wqkv** (in_proj_qkv, `[k_dim*2 + v_dim, dim]`): column-shard by value/key head.
  CHECK the q/k/v output layout + `linear_num_key_heads` vs `linear_num_value_heads`
  (27B: value=48; grep key). Respect the repeat-interleave GQA ratio at the shard
  boundary, like FA q/kv heads.
- **wz, w_beta, w_alpha** (per value head): column-shard by value head.
- **conv_states / s_matrices / s_scales** (`dn_state`, per value head): **SHARD by value
  head** — each rank holds + evolves ONLY its local value heads' recurrent state. The
  NEW piece and the real memory win (state not replicated). Per-head independent → no
  cross-rank dep until wo.
- conv1d_silu_split, fused_qk_l2_norm_scale, repeat_interleave_qk, gated_delta_net
  (q8/fp32/q4), gated_norm: run on LOCAL value heads (local DeltaNet config counts).
- **wo** (out_proj, `[dim, v_dim]`): row-shard (input = local value heads) → PARTIAL into
  s.o → all_reduce → add s.x. Exactly the FA wo pattern.
- **DeltaNet FFN** (w_gate/w_up/w_down inline in run_dn_layer_body): shard like FA FFN
  + 2nd all-reduce. Factor a sharded DeltaNet FFN tail or generalize the helper.
- Add a `run_dn_layer_body` phase split mirroring `FaPhase`: `DnAttn` (partial wo → s.o,
  return) + `DnFfnShard` (partial w_down → s.o). `forward_scratch_tp` orchestrates
  DeltaNet like FA: DnAttn → all_reduce → add → DnFfnShard → all_reduce → add.
- `DeltaNetState` ctor + `dn_states[]` in harness allocate LOCAL value-head state per
  rank (new_with_quant sizes from config → pass the local DeltaNet config).
- **GOTCHA (from 3b):** scratch (`gate_ffn`/`up`/`ffn_hidden` + dn_* buffers) must fit
  whatever runs there. With DeltaNet now sharded you CAN shrink hidden_dim + the dn_*
  sizes locally — but re-derive Qwen35Scratch sizing from the local config carefully and
  verify no full-FFN consumer remains. (In 3b, hidden_dim had to stay full because
  DeltaNet FFN was replicated-full; that constraint lifts once DeltaNet FFN is sharded.)

## Validation (same bar as 3b)
`tp_attn_parity --slice` on `qwen3.5-0.8b.mq4` (re-check linear_num_value_heads;
FullAttn at 3,7,11,15,19,23) must stay fp32 ~1e-6 / 32-32 with DeltaNet sharded. Then
`~/.hipfire/models/qwen3-27b-3.6.mq4-awq.remote-mi300x` (AWQ, 64 layers). RANKDIFF diag
if ranks diverge. Then re-measure decode tok/s — with DeltaNet sharded, TP=2 should no
longer regress (and ideally beat TP=1 on the big model).

## Stage 3d (AFTER 3c) — batched-TP prefill
`forward_prefill_chunk` (qwen35.rs ~6582): batched FA arm (~7291–7897, ~600 lines of
fused-GEMM dtype branches) + batched DeltaNet arm (~6816–7290), NO FaPhase seam. Extract
batched FA + DeltaNet bodies with TP phases (partial wo/down → all_reduce[N×dim] → add),
orchestrate per-layer across ranks like forward_scratch_tp. Sliced weights already make
the existing batched GEMMs produce local outputs (smaller m/k) — the only new work is the
all-reduce insertions + per-layer cross-rank orchestration. Measure prefill throughput
TP=1 vs TP=2 on 27B (warm cache/DPM, fresh process, byte-identical prompt + md5, per
docs/methodology/perf-benchmarking.md).

## Docs / memory
docs/plans/tp-3b-fullattn-shard-scope.md (3b, done), docs/plans/
tp-3-forward-scratch-tp-impl.md (Stage 3). Write `docs/plans/tp-3c-deltanet-shard.md`
as you scope. Memory `project_tp_a3b_stage3.md` has the running state + ceiling rationale.

## FIRST STEP
Re-read `run_dn_layer_body` in full; grep `linear_num_key_heads`/`linear_num_value_heads`
+ the wqkv/wz/w_beta/w_alpha output layout + DeltaNetState field shapes
(conv_states/s_matrices/s_scales); confirm the GQA repeat-interleave ratio shards cleanly
at tp=2/tp=4; then scope 3c into `tp-3c-deltanet-shard.md` BEFORE coding. Do NOT start
coding until the layout + state sharding is mapped.
