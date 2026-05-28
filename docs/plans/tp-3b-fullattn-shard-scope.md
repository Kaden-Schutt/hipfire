# Stage 3b — FullAttn per-rank weight slicing (scope)

**Status: ✅ IMPLEMENTED + validated (2026-05-28, uncommitted).** Both
milestones green on `qwen3.5-0.8b.mq4`, TP=2↔TP=1 along an identical forced
path:

| mode | KV | DeltaNet state | worst rel logit Δ | argmax |
|------|------|------|------|------|
| replicated (Stage 3, regression) | fp32 | fp32 | 2.604e-6 | 32/32 ✓ |
| **sliced attn-only (Milestone A)** | fp32 | fp32 | **2.604e-6** | 32/32 ✓ |
| **sliced attn+FFN (Milestone B)** | fp32 | fp32 | **2.872e-6** | 32/32 ✓ |
| sliced attn+FFN | **q8** | fp32 | **4.128e-6** | 32/32 ✓ |

The sliced path matches single-GPU to the reassociation floor (each extra
all-reduce adds ~1e-7). Notably sliced q8 KV (4e-6) is far cleaner than the
*replicated* q8 KV path (4e-3): in sliced mode each rank's local KV head
quantizes identically to the reference's corresponding head.

What landed (in `crates/hipfire-arch-qwen35/src/qwen35.rs` + the
`tp_attn_parity --slice` harness):
- `slice_quant_rows` / `slice_quant_cols` (+ `quant_group_size`) — format-
  agnostic byte slicers, 4 CPU unit tests green.
- `load_weight_tensor_sliced` (pread path, asserts no AWQ) + `load_weights_tp`
  (load-then-slice: wq/wk/wv rows, wo cols, w_gate/w_up rows, w_down cols).
- `local_attn_config` (shrinks head counts; **keeps `hidden_dim` full** — see
  gotcha below) + `FaPhase::TpAttn{mask: Option}` + `TpFfnShard` +
  `run_fa_ffn_gate_up`/`run_fa_ffn_body_sharded` + `forward_scratch_tp`
  (per-rank configs, `fa_masks: Option`, 2nd all-reduce for sharded FFN).

**GOTCHA burned in:** `local_attn_config` must NOT shrink `hidden_dim`.
DeltaNet layers run **replicated with their full FFN**, sharing the
`gate_ffn`/`up`/`ffn_hidden` scratch; sizing it to `local_ffn` OOMs on the
first DeltaNet FFN (HIP 700 illegal access, surfaced async at the next
`hipModuleLoad`). The FullAttn FFN's locality comes from the sliced *weights*
(`w_*.m`), not the config — the sharded FFN computes into the full scratch's
local prefix (silu over the full buffer is harmless: weight_gemv reads only
the local `k`).

**Remaining (not done):** the perf datapoint (§6) — needs a batched-TP prefill
path (forward_scratch_tp is single-token) for a meaningful compute-bound
measurement; single-stream decode on this tiny DeltaNet-heavy hybrid is
expected to regress (2 all-reduces/FA layer, only 6/24 layers sharded). The
from-scratch sliced loader (no full-load peak) and DeltaNet/MoE sharding are
later stages.

---

Builds on Stage 3 (`forward_scratch_tp`, committed `624fc664`) which proved the
TP forward math correct (fp32 TP=2↔TP=1 = 2.6e-6) but ran **replicated** compute
(each rank does the full attention then masks). 3b makes each rank load and
compute only its **slice** — the actual compute + memory win.

## 0. Goal + honest expectation

Turn "100% work + full weights per rank" into "~1/N work + ~1/N FA weights
per rank" for the **FullAttention layers**. Concretely on `qwen3.5-0.8b`
(n_heads=8, n_kv_heads=2, head_dim=256, dim=1024) at TP=2: each rank holds
heads {0–3} or {4–7}, half of wq/wk/wv/wo and half of the FFN.

**Caveat up front — this alone is a minority win on the hybrid.** Qwen3.5 is
DeltaNet-heavy: 0.8B has FullAttn at only layers 3,7,11,15,19,23 (6 of 24);
the other 18 are DeltaNet and stay **replicated** in 3b. So 3b-FullAttn
shards ~25% of layers. It is the right *next* step because (a) it de-risks
and reuses the slicing machinery, and (b) FullAttn is where the validated
TP scaffold already lives. But total memory/throughput won't move much until
DeltaNet sharding (16 value heads + recurrent state — harder, later) and,
for A3B, expert-parallel MoE (where the 35B bulk actually lives). Don't
expect a big end-to-end number from 3b-FullAttn in isolation.

## 1. Megatron sharding map (gated attention + FFN)

Row-major quant weights `[m_out × k_in]`. Column-parallel = shard `m`
(output rows, contiguous). Row-parallel = shard `k` (input cols, per-row
group gather) + all-reduce the partial output.

| weight  | shape `[m × k]`            | shard       | per-rank result |
|---------|----------------------------|-------------|-----------------|
| wq      | `[2·n_heads·hd × dim]` (gated) | **column** (by head-block, `2·hd` each — `ShardConfig::wq_row_range`) | local Q+gate heads |
| wk, wv  | `[n_kv·hd × dim]`          | **column** (by KV head; replicate if tp>n_kv via `tp_kv_replicate`) | local KV heads |
| wo      | `[dim × n_heads·hd]`       | **row** (`wo_col_range`) | partial → all-reduce |
| w_gate  | `[ffn × dim]`              | **column** (local ffn rows) | local ffn |
| w_up    | `[ffn × dim]`              | **column** | local ffn |
| w_down  | `[dim × ffn]`              | **row** (local ffn cols) | partial → all-reduce |

→ **two all-reduces per FullAttn layer**: one after `wo` (already in
`forward_scratch_tp`), one new after `w_down`.

## 2. Slicing mechanism (format-agnostic)

Hook: `load_weight_tensor` pulls `(info.quant_type, buf)` from the hfq, then
calls `load_weight_tensor_raw(gpu, qt, &buf, m, k)`. Slice `buf` (CPU bytes)
before upload — no kernel changes, the existing gemv just gets smaller m/k.

The tensor is a dense `[m × k]` quant matrix with uniform per-row encoding, so:
- `row_bytes = buf.len() / m`            (exact; no per-format byte table)
- `group_bytes = row_bytes · GS / k`     (GS = group size, 256 for *G256)

**Column-parallel slice** (shard output rows `[m0..m1)`):
```
dst = buf[m0·row_bytes .. m1·row_bytes]   // contiguous
WeightTensor{ m: m1-m0, k, .. }
```
AWQ sidecar is input-indexed `[k]` → **unchanged** (keep full). Trivial.

**Row-parallel slice** (shard input cols `[c0..c1)`, group-aligned):
```
g0 = c0/GS; g1 = c1/GS                      // require c0%GS==0, c1%GS==0
for r in 0..m:                              // m separate copies (restride)
    dst[r·new_row .. ] = buf[r·row_bytes + g0·group_bytes .. r·row_bytes + g1·group_bytes]
WeightTensor{ m, k: c1-c0, .. }
```
AWQ sidecar `[k]` → slice to `[c0..c1)` (contiguous). MQ4G256/MQ3G256 carry
the sidecar (`DType::supports_awq_sidecar`); slice it in lockstep or the
dequant is wrong. PARO sidecars: same rule, scope per-field when needed.

## 3. Loader changes

Add `load_weights_tp(hfq, config, gpu, shard, rank)` (or thread `Option<(&ShardConfig, usize)>`
into `load_weights`). For **FullAttn layers only**, route the 6 weights
through the slicing loaders above using `ShardConfig` ranges; load DeltaNet
layers and norms/embeddings/lm_head **full** (unchanged). The `m`/`k` stored
in each sliced `WeightTensor` become the LOCAL dims — that is what makes the
existing `weight_gemv` compute the right (smaller) matmul with zero kernel
changes.

Two small helpers in `llama.rs` beside `load_weight_tensor_raw`:
`slice_quant_rows(buf, m, k, m0, m1)` and
`slice_quant_cols(buf, m, k, c0, c1, gs)` returning `Vec<u8>`, plus sidecar
slicing.

## 4. Compute-path changes (`run_fa_layer_body`)

The function keys attention dims off `config.{n_heads, n_kv_heads, hidden_dim}`.
Cleanest: pass a **per-rank local config** (clone with `n_heads→q_heads_per_rank`,
`n_kv_heads→kv_heads_per_rank`, `hidden_dim→ffn/tp`; head_dim, rope_theta,
norm_eps, partial_rotary_factor stay global). Then:
- `deinterleave_f32`, `rmsnorm_batched(q/k)`, `rope_partial_interleaved`,
  `attention_*` all use local counts automatically.
- `weight_gemv(wo)` uses `layer.wo.{m,k}` (already local) → partial in `s.o`.
- KV cache is per-rank with local kv heads (already per-rank in the harness).

**FFN:** add a `TpFfn`-sharded path in `run_fa_ffn_body`: local-ffn gate/up
(`layer.w_gate/up.m` already local), `w_down` row-parallel → partial s.x
contribution → **second all-reduce** → add. `forward_scratch_tp`'s FA layer
becomes: TpAttn → all-reduce(s.o) → add → TpFfn(partial) → all-reduce → add.

Audit: confirm nothing reads `config.n_heads` for a *global* meaning inside
the FA path (mask shape, flash_partials sizing). flash_partials is sized in
`Qwen35Scratch` (§5) so it follows local heads.

## 5. Scratch sizing

`Qwen35Scratch` buffers `fa_q_full, fa_q, fa_gate, fa_k, fa_v, fa_attn_out,
gate_ffn, up, ffn_hidden, flash_partials` must size to LOCAL heads / local
ffn. Add a ctor that takes the local config (or local head/ffn counts).
`s.o`/`s.x` stay `[dim]` (residual is full-width on every rank). The mask
buffer from Stage 3 is no longer needed (slicing replaces masking).

## 6. Validation

1. **Parity (must stay green):** extend `tp_attn_parity` with
   `HIPFIRE_PARITY_SLICE=1` → load via `load_weights_tp` and run the sliced
   path. At fp32 KV+state it MUST reproduce **2.6e-6 / 32-32** — slicing is
   the *same math* as Stage 3's mask-replicate, just without computing the
   zeroed heads. Any larger delta = a slicing/sidecar/restride bug.
2. **Numerical cross-check:** assert a sliced+all-reduced layer == the Stage-3
   masked-replicate layer bitwise-close (isolates slicing from orchestration).
3. **Perf (the point):** measure TP=2 vs TP=1 **prefill** throughput on 27B
   (compute-bound → where sharding should show), per
   `docs/methodology/perf-benchmarking.md` (warm cache/DPM, fresh process,
   byte-identical prompt + md5). Expect decode single-stream to stay flat or
   regress (bandwidth-bound + 2 all-reduces/layer); that's fine — prefill /
   capacity is the win.

## 7. Constraints to check before coding

- `tp_size | n_heads` (8/2 ✓). `n_kv_heads`: if `tp > n_kv_heads` use
  `tp_kv_replicate` (0.8B tp=2 → 1 KV head/rank, no replicate; tp=4 → replicate).
- **Group alignment:** every row-parallel local-k must be `% GS(256) == 0`.
  Check: wo k=attn_dim; w_down k=ffn. 0.8B tp=2: wo local_k=1024 ✓;
  ffn=intermediate_size → confirm `ffn/tp % 256 == 0` (27B/A3B too). If not,
  pad or fall back to replicate for that weight.
- Sidecars (AWQ for MQ4/MQ3; PARO) sliced consistently (§2).
- gated wq slice is in `2·head_dim` blocks (`wq_row_range` already does this).

## 8. Risks

- **Restride bug** in row-parallel slicing (off-by-group, sidecar mismatch) —
  caught by the §6.1 parity (will blow up to ~1e-1 like the flash bug did).
- **Group-misalignment** on some tp/dim combo — guard with an assert + clear
  error; pad or replicate that weight.
- **Per-rank-config leakage** — a global `n_heads` read inside the FA path
  would corrupt the slice; audit (§4) + the parity gate guard it.
- **Second all-reduce latency** — more collective per layer; acceptable, but
  it's why single-stream decode may not speed up (see §0 expectation).
- **DeltaNet-heavy reality** (§0) — 3b-FullAttn shards a minority of layers;
  set expectations and plan DeltaNet sharding + A3B expert-parallel next.

## 9. Sequencing

1. `slice_quant_rows` / `slice_quant_cols` (+ sidecar) helpers + CPU unit
   tests (slice→reassemble == original; group-aligned).
2. `load_weights_tp` for FullAttn layers (DeltaNet full).
3. Per-rank local config + local-sized `Qwen35Scratch` ctor.
4. Sharded `run_fa_ffn_body` + second all-reduce in `forward_scratch_tp`.
5. `tp_attn_parity --slice` → confirm 2.6e-6 at fp32.
6. 27B prefill TP=2 vs TP=1 perf datapoint.

Each step is independently testable; step 5 is the correctness gate, step 6
is the first real acceleration number.
