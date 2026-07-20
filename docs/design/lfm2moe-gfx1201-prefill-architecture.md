# LFM2.5 gfx1201 batched-prefill architecture — FROZEN CONTRACT

**Status:** ✅ FROZEN by conductor (Main) 2026-07-18. Reviewer `PrimitiveLeech` FINAL VERDICT = APPROVE/freezable after 3 adversarial cycles (BLOCK → BLOCK → APPROVE) resolving 7 blockers + 3 residuals (grouped-MoE 16-alignment, flash-partials footprint, capacity cap) + the per-cohort manifest correction (dense vocab 65536 / θ 1e6 / 350M hidden 1024; only 8B is 2048/128000/5e6). Oversight `WhisperingPike` PASS; baseline Amdahl (`local://lfm-baseline-amdahl.md`) sets lane priority (dense-GEMM > attention > conv). Frozen contracts: `forward_prefill_batch`/`forward_prefill_chunk` API (§2), additive lazy `Lfm2MoeState` scratch (§4), chunking/state-transition contract (§3), per-cohort admissions (§8). Phase 0 head-elision is committed (`62dedc41a`) + hardened parity oracle (`0198fe6c0`). Implementation MAY now proceed per the §10 vertical-slice order (350m.q8 first); §13 items are implementation/promotion gates, not design blockers.

**Checkout witnessed:** `lfm-redline` at `62dedc41aa5d335f0518e70cb9e748da332b6138`.

**Inputs:** the approved `local://lfm-gfx12-prefill-plan.md`, corrected `local://lfm-discovery.md`, and the read-only scout source maps named in the provenance section.

## 1. Scope and invariants

This design is only for prompt prefill for LFM2.5 (`arch_id == 11`) on **gfx1201**. It does not change decode, DFlash, DDTree, MTP, serving semantics, PM4, non-LFM models, or any non-gfx1201 GPU. It does not invoke the coherence gate.

The new path MUST:

- advance the same absolute KV slots, convolution-layer rings, and exact `state.n_tokens` as sequential `decode_step`; floating values follow the explicit Section-12 tolerances because WMMA uses F16 activations;
- return host `Vec<f32>` logits for the final prompt token only;
- leave the existing eager `decode_step` path, its fields, allocation sizes, and public signature intact;
- add exactly one lazy optional prefill scratch owner to `Lfm2MoeState`; eager-only/non-gfx1201/flag-off state construction allocates **zero** batched-scratch bytes;
- run only when `gpu.arch_caps.is_gfx1201()` and `HIPFIRE_LFM2_PREFILL_BATCH=1`; selection occurs before allocating scratch or mutating model state, and every other case uses the existing eager loop;
- never fall back to eager after a batched chunk has started, because that would duplicate state updates. An error is returned and the caller must reset before retrying.

Phase 0 head elision is already committed and is the scalar reference. It is not batched prefill.

## 2. Frozen Rust API

Add these functions to `crates/hipfire-arch-lfm2moe/src/forward.rs`:

```rust
/// Batched prompt prefill for LFM2.5 on gfx1201.
///
/// Ingests non-empty `token_ids` beginning at absolute `start_pos`, advances
/// KV, convolution state, and `state.n_tokens` as sequential decode steps
/// would, and returns only the final token's host logits.
pub fn forward_prefill_batch(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_ids: &[u32],
    start_pos: u32,
) -> Result<Vec<f32>, String>;

/// Executes one non-empty contiguous chunk. `token_ids.len()` must not exceed
/// `state.prefill_batch.as_ref().unwrap().max_batch`. When `emit_head` is true,
/// this is the final chunk: copy its final residual row to the existing
/// one-token head scratch and write final-token logits into `state.logits`.
/// It never downloads logits.
fn forward_prefill_chunk(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_ids: &[u32],
    start_pos: u32,
    emit_head: bool,
) -> Result<(), String>;
```

`crates/hipfire-arch-lfm2moe/src/lib.rs` MUST add `forward_prefill_batch` to the existing `forward` re-export list. `forward_prefill_chunk` remains private. No `decode_step` signature changes.

### 2.1 Entry validation and errors

`forward_prefill_batch` performs all validation before the first GPU mutation:

1. reject an empty `token_ids` slice;
2. reject a non-gfx1201 device;
3. reject unless `HIPFIRE_LFM2_PREFILL_BATCH` is exactly `"1"`; the public re-export cannot bypass the opt-in gate;
4. require `start_pos as usize == state.n_tokens`;
5. reject any token ID greater than `i32::MAX` **or** `>= cfg.vocab_size` before `i32::try_from`; the unchecked address arithmetic in `embedding_q8_batched` never sees an invalid row;
6. use checked addition for `start_pos + token_ids.len()` and reject overflow;
7. reject an exclusive end position greater than `i32::MAX` before any i32 position or kernel dimension conversion;
8. reject an end position beyond `state.max_seq` / KV physical capacity;
9. checked-multiply every derived i32 kernarg dimension (`N*n_heads`, `N*n_kv_heads`, `N*k_top`, and stage numel/grid products) and reject a value greater than `i32::MAX`;
10. require every layer weight dtype to match its admitted cohort before the first chunk (Section 8).

Every `HipResult` is converted to `String` with stage, layer index, chunk base, and chunk length. There is no rollback after a GPU failure and no eager fallback after mutation.

After the final chunk succeeds, `forward_prefill_batch` performs exactly one `gpu.download_f32(&state.logits)` and returns it. The vector length is exactly `cfg.vocab_size`: 65,536 for both dense cohorts and 128,000 for 8B-A1B.

## 3. Chunking and state-transition contract

Define:

```rust
const LFM2_PREFILL_MAX_BATCH: usize = 256;
const LFM2_PREFILL_MAX_BATCH_LIMIT: usize = 512;
const LFM2_FLASH_PARTIALS_BATCH: usize = 16;
```

`Lfm2MoeState::new_with_max_seq` does **not** allocate batched scratch; it stores `prefill_batch: None`. After every pre-mutation check in Section 2.1 succeeds, `forward_prefill_batch` calls `state.ensure_prefill_batch(gpu, cfg)`. That method rechecks `is_gfx1201()` and the exact opt-in flag, then lazily allocates `Lfm2PrefillBatchScratch` only when the option is `None`.

`Lfm2PrefillBatchScratch::new` computes its capacity once:

```text
raw = HIPFIRE_LFM2_PREFILL_MAX_BATCH when present and valid UTF-8
parsed = raw parsed as usize
if parsed > 512: return Err before allocation
candidate = parsed when parsing succeeds and parsed >= 2, otherwise 256
abi_batch_cap = floor(i32::MAX / max(n_heads, n_kv_heads, k_top, 1))
max_batch = min(candidate, max(max_seq, 1), abi_batch_cap)
flash_partials_batch = min(16, max_batch), always >= 1
```

Changing the environment after lazy scratch allocation has no effect; changing it before the first gated call does. Actual chunks may contain one token. Configured capacities above 512 are rejected rather than silently clamped, so every admitted capacity is within the frozen `{128,256,512}` sweep ceiling; values 2 through 512 remain valid diagnostic chunk sizes.

For original input `token_ids`, original absolute base `start_pos`, and chunk offset `o`:

```text
p = (start_pos as usize) + o
N = min(state.prefill_batch.as_ref().unwrap().max_batch, token_ids.len() - o)
chunk = token_ids[o .. o+N]
positions[i] = (p + i) as i32, 0 <= i < N
max_seq = p + N
max_ctx_len = p + N
emit_head = (o + N == token_ids.len())
```

Per chunk, in this order:

1. upload the chunk's token IDs and absolute positions;
2. batched embedding lookup into `h_batch[0..N]`;
3. execute every layer in model order, choosing its convolution or attention mixer and dense or MoE FFN;
4. attention writes every K/V row for the chunk before causal flash attention; row `i` can attend only positions `<= positions[i]`;
5. every convolution layer seeds its K=3 scan from that layer's existing two-value-per-channel `conv_states` tail and commits the new tail in place;
6. after all layer-body KV/convolution/residual work succeeds, set `state.n_tokens = (p + N) as usize`; this records the irreversible ingestion before optional head work;
7. only when `emit_head`, copy `h_batch[(N-1) * hidden .. N * hidden]` to the existing one-token `state.h`, run final norm and tied head into existing `state.final_norm_buf` and `state.logits`;
8. continue with the next chunk, or download `state.logits` once after the final chunk. A final-head failure returns an error with KV, convolution tails, and `n_tokens` already advanced, so the caller must reset before retrying.

KV tensors are already sequence-indexed and are reused unchanged. `conv_states` are already one ring per convolution layer and are reused unchanged. There is no separate KV cursor: absolute batched writes plus `state.n_tokens` are the cursor contract. Chunk boundaries are not model boundaries.

The eager single-token buffers remain allocated at their current sizes. They are not resized, repurposed, or read by batched layer bodies. The only handoff is the final-row copy into the existing one-token head scratch; this deliberately avoids an unnecessary `max_batch × vocab` allocation.

## 4. Exact additive `Lfm2MoeState` scratch

Declare `pub(crate) struct Lfm2PrefillBatchScratch` in `lfm2moe.rs`. `Lfm2MoeState` gains exactly one field:

```rust
pub(crate) prefill_batch: Option<Lfm2PrefillBatchScratch>,
```

`new_with_max_seq` initializes it to `None`. Construction and lazy access have these exact interfaces:

```rust
impl Lfm2PrefillBatchScratch {
    fn new(
        gpu: &mut Gpu,
        cfg: &Lfm2MoeConfig,
        max_seq: usize,
    ) -> Result<Self, String>;
}

impl Lfm2MoeState {
    pub(crate) fn ensure_prefill_batch(
        &mut self,
        gpu: &mut Gpu,
        cfg: &Lfm2MoeConfig,
    ) -> Result<&mut Lfm2PrefillBatchScratch, String>;
}
```

It errors unless `gpu.arch_caps.is_gfx1201()` and the opt-in flag is exactly `"1"`, allocates `Lfm2PrefillBatchScratch::new(gpu, cfg, self.max_seq)` once, and returns the stored mutable reference. `reset` retains the allocation and overwrites live prefixes on the next use. `free_gpu` frees every `Some` tensor in the scratch before freeing eager state. Therefore `Architecture::new_state` on non-gfx1201 or flag-off paths retains the pre-change VRAM footprint.

All fields enumerated below, including `max_batch`, belong to `Lfm2PrefillBatchScratch`, not directly to `Lfm2MoeState`.

Symbols used below:

- `B = scratch.max_batch` (allocation capacity), `N <= B` (current chunk);
- `hidden = cfg.hidden_size`;
- `n_heads = cfg.num_attention_heads`, `n_kv_heads = cfg.num_key_value_heads`, `head_dim = cfg.head_dim`;
- `q_dim = cfg.q_dim()`, `kv_dim = cfg.kv_dim()`;
- `dense_inter = cfg.intermediate_size`;
- `moe_inter = cfg.moe_intermediate_size`, `n_exp = cfg.num_experts`, `k_top = cfg.num_experts_per_tok`;
- `vocab = cfg.vocab_size`;
- `flash_tiles = ceil(state.max_seq / 128)` and `flash_stride = 2 + cfg.head_dim` floats;
- `M = scratch.flash_partials_batch = min(16, B)`, the query-row multiplier used only to size the worst-case partials capacity at `state.max_seq`;
Grouped allocation and live launch sizing use this exact LFM-local helper; there is no dependency on Qwen's private helper or private block constant:

```rust
const LFM2_MOE_GROUPED_BLOCK_M: usize = 16;

fn lfm2_moe_grouped_m_total_bound(
    rows: usize,
    n_exp: usize,
    k_top: usize,
) -> Result<usize, String> {
    let total_slots = rows
        .checked_mul(k_top)
        .ok_or_else(|| "lfm2moe: grouped slot count overflow".to_string())?;
    let live_experts = total_slots.min(n_exp);
    let padding = live_experts
        .checked_mul(LFM2_MOE_GROUPED_BLOCK_M - 1)
        .ok_or_else(|| "lfm2moe: grouped padding overflow".to_string())?;
    let unaligned = total_slots
        .checked_add(padding)
        .ok_or_else(|| "lfm2moe: grouped m_total overflow".to_string())?;
    let rounded = unaligned
        .checked_add(LFM2_MOE_GROUPED_BLOCK_M - 1)
        .ok_or_else(|| "lfm2moe: grouped alignment overflow".to_string())?;
    Ok((rounded / LFM2_MOE_GROUPED_BLOCK_M) * LFM2_MOE_GROUPED_BLOCK_M)
}
```

`m_total_capacity = lfm2_moe_grouped_m_total_bound(B, n_exp, k_top)?`; for a live chunk, `m_total_live = lfm2_moe_grouped_m_total_bound(N, n_exp, k_top)?`. Both are multiples of 16. Tensors allocate for capacity and Path 2 receives the live value. This is the authoritative LFM transcription of Qwen's proven `align_up(total_slots + min(total_slots,n_exp)*15, 16)` bound; the unaligned DeepSeek `total_slots+n_exp*16` formula is forbidden.

All shared/dense tensor fields are `GpuTensor`. `token_ids_batch`, `positions_batch`, `topk_indices_batch`, and the grouped indexing tensors use four-byte storage allocated as `DType::F32` but carry i32 bit patterns, matching the existing batched kernel ABI.

### 4.1 Existing fields — unchanged

The current state remains verbatim:

| Current field | Current shape / type |
|---|---|
| `kv` | `KvCache`; one Q8 slot per attention layer, logically `[state.max_seq, cfg.num_key_value_heads, cfg.head_dim]` for K and V |
| `conv_states` | one F32 `[cfg.hidden_size * (cfg.conv_kernel_size-1)]` tensor per convolution layer |
| `pos_buf` | device i32 scalar |
| `graph_warmed_up` | `bool` |
| `max_seq`, `n_tokens` | `usize` |
| `h`, `tmp` | `[cfg.hidden_size]` each |
| `fa_q` | `[cfg.q_dim()]` |
| `fa_k`, `fa_v` | `[cfg.kv_dim()]` each |
| `fa_attn_out` | `[cfg.q_dim()]` |
| `conv_bcx` | `[3 * cfg.hidden_size]` |
| `conv_y` | `[cfg.hidden_size]` |
| `ffn_tmp`, `ffn_x_rot` | `[cfg.hidden_size]` each |
| `dense_gate`, `dense_up`, `dense_act` | `[cfg.intermediate_size]` each |
| `router_logits` | `[cfg.num_experts]` |
| `topk_indices`, `topk_weights` | `[cfg.num_experts_per_tok]` each; indices are i32 bits |
| `gate_batch`, `up_batch`, `rot_batch` | `[cfg.num_experts_per_tok * cfg.moe_intermediate_size]` each |
| `down_expanded` | `[cfg.num_experts_per_tok * cfg.hidden_size]` |
| `final_norm_buf` | `[cfg.hidden_size]` |
| `logits` | `[cfg.vocab_size]` |

### 4.2 Shared and dense fields inside the lazy scratch — exact names

Allocate these fields only in `Lfm2PrefillBatchScratch::new`:

| New field | Element type | Allocation shape | Current-chunk view / purpose |
|---|---|---:|---|
| `max_batch` | `usize` | scalar | immutable capacity `B` |
| `flash_partials_batch` | `usize` | scalar | immutable sub-batch capacity `M=min(16,B)` |
| `token_ids_batch` | i32 bits | `[B]` | `[N]` embedding IDs |
| `positions_batch` | i32 bits | `[B]` | `[N]`, value `p+i` |
| `h_batch` | F32 | `[B * cfg.hidden_size]` | `[N, cfg.hidden_size]` residual stream |
| `tmp_batch` | F32 | `[B * cfg.hidden_size]` | `[N, cfg.hidden_size]` operator-norm output |
| `operator_x_rot_batch` | F32 | `[B * cfg.hidden_size]` | `[N, cfg.hidden_size]` MQ4/FWHT-rotated operator-norm output; unused by Q8 cohorts |
| `fa_q_batch` | F32 | `[B * cfg.q_dim()]` | `[N, cfg.num_attention_heads, cfg.head_dim]` Q |
| `fa_k_batch` | F32 | `[B * cfg.kv_dim()]` | `[N, cfg.num_key_value_heads, cfg.head_dim]` K |
| `fa_v_batch` | F32 | `[B * cfg.kv_dim()]` | `[N, cfg.num_key_value_heads, cfg.head_dim]` V |
| `fa_attn_out_batch` | F32 | `[B * cfg.q_dim()]` | `[N, cfg.num_attention_heads, cfg.head_dim]` attention output |
| `fa_attn_out_rot_batch` | F32 | `[B * cfg.q_dim()]` | `[N, cfg.q_dim()]` MQ4/FWHT-rotated attention-out projection input; unused by Q8 cohorts |
| `fa_partials_batch` | F32 | `[M * cfg.num_attention_heads * flash_tiles * (2 + cfg.head_dim)]` | capacity for M rows at worst-case `state.max_seq`; not a per-launch row limit |
| `conv_bcx_batch` | F32 | `[B * 3 * cfg.hidden_size]` | `[N, 3, cfg.hidden_size]`, row order `B_gate | C_gate | x` |
| `conv_y_batch` | F32 | `[B * cfg.hidden_size]` | `[N, cfg.hidden_size]` scan output |
| `conv_y_rot_batch` | F32 | `[B * cfg.hidden_size]` | `[N, cfg.hidden_size]` MQ4/FWHT-rotated convolution-out projection input; unused by Q8 cohorts |
| `ffn_tmp_batch` | F32 | `[B * cfg.hidden_size]` | `[N, cfg.hidden_size]` FFN-norm output |
| `ffn_x_rot_batch` | F32 | `[B * cfg.hidden_size]` | `[N, cfg.hidden_size]` FWHT-rotated projection input; unused by Q8 cohorts |
| `dense_gate_batch` | F32 | `[B * cfg.intermediate_size]` | `[N, cfg.intermediate_size]` gate projection |
| `dense_up_batch` | F32 | `[B * cfg.intermediate_size]` | `[N, cfg.intermediate_size]` up projection |
| `dense_act_batch` | F32 | `[B * cfg.intermediate_size]` | `[N, cfg.intermediate_size]` `SiLU(gate) * up` |
| `dense_act_rot_batch` | F32 | `[B * cfg.intermediate_size]` | `[N, cfg.intermediate_size]` MQ4/FWHT-rotated dense-down projection input; unused by Q8 cohorts |

The batched Q8 vertical slice uses every row above except the five `*_rot_batch` projection inputs and the MoE-only fields below. `fa_partials_batch` is a capacity buffer sized for M rows at `state.max_seq`. At call time, `launch_asym_flash_batched` computes `per_pos_bytes` from `max_ctx_len=p+N`, then `sub_batch = min(N, max(1, partials_capacity / per_pos_bytes))`. Shorter context reduces per-row partial bytes, so `sub_batch` may safely exceed M; M bounds buffer allocation, not launch rows. At `state.max_seq=8192`, M=16, hd64, capacity is 4.125 MiB for 350M (16 heads) and 8.25 MiB for 1.2B/8B (32 heads); at baseline `state.max_seq=4096`, 2.0625 and 4.125 MiB. `state.max_seq` remains runtime state, not a cohort constant.

### 4.3 New MoE twin fields

| New field | Element type | Allocation shape | Twinned current field / role |
|---|---|---:|---|
| `router_logits_batch` | F32 | `[B * cfg.num_experts]` | `router_logits`, one router row per token |
| `topk_indices_batch` | i32 bits | `[B * cfg.num_experts_per_tok]` | `topk_indices` |
| `topk_weights_batch` | F32 | `[B * cfg.num_experts_per_tok]` | `topk_weights` |
| `moe_gate_batch` | F32 | `[B * cfg.num_experts_per_tok * cfg.moe_intermediate_size]` | `gate_batch` after grouped gate/up unscatter |
| `moe_up_batch` | F32 | `[B * cfg.num_experts_per_tok * cfg.moe_intermediate_size]` | `up_batch` |
| `moe_rot_batch` | F32 | `[B * cfg.num_experts_per_tok * cfg.moe_intermediate_size]` | `rot_batch`, SwiGLU then expert-down FWHT input |
| `moe_down_expanded_batch` | F32 | `[B * cfg.num_experts_per_tok * cfg.hidden_size]` | exact `down_expanded` twin; reserved for parity/Path-1 oracle and not selected by production Path 2 |

### 4.4 New grouped-MoE Path-2 fields

`run_moe_prefill` Path 2 additionally requires the following persistent scratch. These fields prevent per-chunk allocation and are not aliases of eager buffers:

| New field | Element type | Allocation shape | Role |
|---|---|---:|---|
| `moe_expert_counts` | i32 bits | `[cfg.num_experts]` | routed slot count by expert |
| `moe_expert_offsets` | i32 bits | `[cfg.num_experts + 1]` | exclusive padded offsets |
| `moe_sorted_slot_index` | i32 bits | `[m_total_capacity]` | grouped row to flat `token*cfg.num_experts_per_tok+slot`, `-1` for padding |
| `moe_expert_tile_ids` | i32 bits | `[m_total_capacity / LFM2_MOE_GROUPED_BLOCK_M]` | expert id per fully aligned WMMA tile |
| `moe_inverse_perm` | i32 bits | `[B * cfg.num_experts_per_tok]` | flat route slot to grouped row |
| `moe_grouped_gate_up` | F32 | `[m_total_capacity * 2 * cfg.moe_intermediate_size]` | grouped expert gate/up output |
| `moe_grouped_down` | F32 | `[m_total_capacity * cfg.hidden_size]` | grouped expert down output |

Expert pointer tables already exist in loaded `MoeFfn`; scratch does not duplicate them. Grouped down reads `moe_rot_batch[token*cfg.num_experts_per_tok+slot, :]`; `moe_sorted_slot_index` maps each padded row back to that flat row with `x_row_div=1`. No regrouped-down input or second scatter is permitted. Every live call passes `m_total_live` and asserts `m_total_live <= m_total_capacity`.

### 4.5 Deliberately absent fields

There is no `final_norm_buf_batch` and no `logits_batch`. The head consumes one row, so existing `[cfg.hidden_size]` and `[cfg.vocab_size]` fields suffice. `[B * cfg.vocab_size]` would waste 64 MiB for dense vocab 65,536 and 125 MiB for A1B vocab 128,000 at B=256.

`reset` continues to zero `conv_states` and reset `n_tokens`; batched activation scratch needs no clearing because every consumed prefix is overwritten. KV clearing semantics remain unchanged.

## 5. Frozen 350M-Q8 vertical slice

This is the first and only Phase-1 cohort: `cfg.hidden_size=1024`, `cfg.num_attention_heads=16`, `cfg.num_key_value_heads=8`, `cfg.head_dim=64`, `cfg.q_dim()=1024`, `cfg.kv_dim()=512`, `cfg.intermediate_size=4608`, `cfg.vocab_size=65536`, `cfg.rope_theta=1_000_000.0`, 16 layers (10 conv/6 attention), Q8 projections/embed/head, and no FWHT. For every call below, `N = token_ids.len()` for the current chunk.

All Q8 GEMM entries automatically select their `_gfx12` kernel on `is_rdna4()`; the outer gate remains stricter (`is_gfx1201()`). Inputs are row-major F32. Q8 K dimensions are multiples of 32.

### 5.1 Ordered stage binding

After the lazy allocator returns, bind the stored scratch and construct exact live prefix views / upload byte slices:

```rust
let scratch = state
    .prefill_batch
    .as_ref()
    .ok_or_else(|| "lfm2moe: prefill scratch missing".to_string())?;
let dense_gate_n = scratch
    .dense_gate_batch
    .sub_offset(0, N * cfg.intermediate_size);
let dense_up_n = scratch
    .dense_up_batch
    .sub_offset(0, N * cfg.intermediate_size);
let dense_act_n = scratch
    .dense_act_batch
    .sub_offset(0, N * cfg.intermediate_size);

// All IDs/ranges were validated before lazy allocation.
let token_ids_i32: Vec<i32> = token_ids
    .iter()
    .map(|&id| i32::try_from(id).unwrap())
    .collect();
let positions_i32: Vec<i32> = (0..N).map(|i| i32::try_from(p + i).unwrap()).collect();
let token_bytes = unsafe {
    std::slice::from_raw_parts(
        token_ids_i32.as_ptr().cast::<u8>(),
        token_ids_i32.len() * std::mem::size_of::<i32>(),
    )
};
let position_bytes = unsafe {
    std::slice::from_raw_parts(
        positions_i32.as_ptr().cast::<u8>(),
        positions_i32.len() * std::mem::size_of::<i32>(),
    )
};
```

`GpuTensor::sub_offset` creates non-owning views and those views MUST NOT be freed. Explicit-N GEMMs may receive capacity tensors because `batch_size=N` bounds access; `silu_mul_f32` MUST receive the prefix views because it launches over `gate.numel()`. Loaded `ConvWeights::conv_state_idx` and `AttnWeights::kv_idx` are authoritative, contiguous type-ordinal slot mappings; admission requires uniqueness, in-range values, and complete coverage of the corresponding state/cache arrays.

| Stage | Source-verified Rust call and dimensions | Batch parameter | gfx1201 kernel / behavior |
|---|---|---|---|
| Input upload | `gpu.hip.memcpy_htod(&scratch.token_ids_batch.buf, token_bytes)` and `gpu.hip.memcpy_htod(&scratch.positions_batch.buf, position_bytes)` | exactly `N * size_of::<i32>()` bytes each | `memcpy_htod` accepts `&[u8]`; no compute kernel |
| Batched embed | `gpu.embedding_lookup_q8_batched(&weights.embed, &scratch.h_batch, &scratch.token_ids_batch, N, cfg.hidden_size)` | `n=N` | `embedding_q8_batched`; 350M dim=1024 |
| Operator RMSNorm | `gpu.rmsnorm_batched(&scratch.h_batch, &layer.operator_norm, &scratch.tmp_batch, N, cfg.hidden_size, cfg.rms_norm_eps)` | `batch=N` | `rmsnorm_f32`; 350M n=1024 |
| Conv in projection | `gpu.gemm_q8_0_wmma(&c.in_proj.buf, &scratch.tmp_batch, &scratch.conv_bcx_batch, 3*cfg.hidden_size, cfg.hidden_size, N)` | `batch_size=N` | `gemm_q8_0_wmma_gfx12`; 350M M=3072, K=1024 |
| Conv scan | `kernels::conv1d_gated_scan_n(gpu, &scratch.conv_bcx_batch, &state.conv_states[c.conv_state_idx], &c.conv_weight, &scratch.conv_y_batch, N, cfg.hidden_size)` | `n_tokens=N` | new `conv1d_gated_scan_n_f32`; admission fixes `cfg.conv_kernel_size=3` |
| Conv out + residual | `gpu.gemm_q8_0_residual_wmma(&c.out_proj.buf, &scratch.conv_y_batch, &scratch.h_batch, cfg.hidden_size, cfg.hidden_size, N)` | `batch_size=N` | `gemm_q8_0_residual_wmma_gfx12`; 350M M=K=1024 |
| Attention fused QKV | `gpu.gemm_qkv_q8_0_wmma(&a.wq.buf, &a.wk.buf, &a.wv.buf, &scratch.tmp_batch, &scratch.fa_q_batch, &scratch.fa_k_batch, &scratch.fa_v_batch, cfg.q_dim(), cfg.kv_dim(), cfg.kv_dim(), cfg.hidden_size, N)` | `batch_size=N` | `gemm_qkv_q8_0_wmma_gfx12`; 350M M=1024/512/512, K=1024 |
| Q RMSNorm | `gpu.rmsnorm_batched(&scratch.fa_q_batch, &a.q_norm, &scratch.fa_q_batch, N*cfg.num_attention_heads, cfg.head_dim, cfg.rms_norm_eps)` | `batch=N*cfg.num_attention_heads` | in-place per-head `rmsnorm_f32` |
| K RMSNorm | `gpu.rmsnorm_batched(&scratch.fa_k_batch, &a.k_norm, &scratch.fa_k_batch, N*cfg.num_key_value_heads, cfg.head_dim, cfg.rms_norm_eps)` | `batch=N*cfg.num_key_value_heads` | in-place per-head `rmsnorm_f32` |
| Full RoPE | `gpu.rope_batched_f32(&scratch.fa_q_batch, &scratch.fa_k_batch, &scratch.positions_batch, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rope_theta, N)` | `batch_size=N` | full rotate-half RoPE; 350M theta=1e6 |
| K cache write | `gpu.kv_cache_write_q8_0_batched(&state.kv.k_gpu[a.kv_idx], &scratch.fa_k_batch, &scratch.positions_batch, cfg.num_key_value_heads, cfg.head_dim, N)` | `batch_size=N` | Q8 K writes at absolute positions |
| V cache write | `gpu.kv_cache_write_q8_0_batched(&state.kv.v_gpu[a.kv_idx], &scratch.fa_v_batch, &scratch.positions_batch, cfg.num_key_value_heads, cfg.head_dim, N)` | `batch_size=N` | Q8 V writes at absolute positions |
| Causal flash | `gpu.attention_flash_q8_0_batched_masked(&scratch.fa_q_batch, &state.kv.k_gpu[a.kv_idx], &state.kv.v_gpu[a.kv_idx], &scratch.fa_attn_out_batch, &scratch.positions_batch, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, p+N, p+N, N, &scratch.fa_partials_batch, None, 0, 0)` | logical `batch_size=N`; wrapper derives safe call-time `sub_batch` from capacity and `p+N` | `attention_flash_q8_0_tile_batched` + reduce; 350M hd64 gives dpt=2 |
| Attention out + residual | `gpu.gemm_q8_0_residual_wmma(&a.wo.buf, &scratch.fa_attn_out_batch, &scratch.h_batch, cfg.hidden_size, cfg.q_dim(), N)` | `batch_size=N` | `gemm_q8_0_residual_wmma_gfx12`; 350M M=K=1024 |
| FFN RMSNorm | `gpu.rmsnorm_batched(&scratch.h_batch, &layer.ffn_norm, &scratch.ffn_tmp_batch, N, cfg.hidden_size, cfg.rms_norm_eps)` | `batch=N` | `rmsnorm_f32` |
| Fused gate/up | `gpu.gemm_gate_up_q8_0_wmma(&d.w1.buf, &d.w3.buf, &scratch.ffn_tmp_batch, &scratch.dense_gate_batch, &scratch.dense_up_batch, cfg.intermediate_size, cfg.intermediate_size, cfg.hidden_size, N)` | `batch_size=N` | `gemm_gate_up_q8_0_wmma_gfx12`; 350M M=4608/4608, K=1024 |
| SwiGLU | `gpu.silu_mul_f32(&dense_gate_n, &dense_up_n, &dense_act_n)` | view `numel=N*cfg.intermediate_size` | `silu_mul_f32` |
| Dense down + residual | `gpu.gemm_q8_0_residual_wmma(&d.w2.buf, &dense_act_n, &scratch.h_batch, cfg.hidden_size, cfg.intermediate_size, N)` | `batch_size=N` | `gemm_q8_0_residual_wmma_gfx12`; 350M M=1024, K=4608 |
| Final-row copy | `gpu.memcpy_dtod_at_auto(&state.h.buf, 0, &scratch.h_batch.buf, (N-1)*cfg.hidden_size*std::mem::size_of::<f32>(), cfg.hidden_size*std::mem::size_of::<f32>())` | one `cfg.hidden_size` row | exact `DeviceBuffer` API; only when `emit_head=true` |
| Final RMSNorm | `gpu.rmsnorm_f32(&state.h, &weights.embedding_norm, &state.final_norm_buf, cfg.rms_norm_eps)` | batch 1 | existing `rmsnorm_f32` |
| Tied head | `weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)` | batch 1; `weights.lm_head.m=cfg.vocab_size`, `.k=cfg.hidden_size` | admitted Q8 resolves through `GemvFamily::run_auto` to `gemv_q8_0` |
| Host result | `gpu.download_f32(&state.logits)` in `forward_prefill_batch` | one `[cfg.vocab_size]` D2H | 350M: 65,536 F32 = 262,144 bytes |

`p` in the table is the chunk's absolute base. Q, K, V, KV writes, and flash use the same `positions_batch[0..N]`.

## 6. New gfx1201 convolution scan — authoring contract

### 6.1 File, symbol, and wrapper

Add exactly one new HIP source during implementation:

```text
kernels/src/conv1d_gated_scan_n.gfx1201.hip
```

It exports `conv1d_gated_scan_n_f32`. The crate-local wrapper in `crates/hipfire-arch-lfm2moe/src/kernels.rs` is:

```rust
pub fn conv1d_gated_scan_n(
    gpu: &mut Gpu,
    bcx: &GpuTensor,
    state: &GpuTensor,
    weight: &GpuTensor,
    out_y: &GpuTensor,
    n_tokens: usize,
    channels: usize,
) -> HipResult<()>;
```

The HIP kernargs are pointers in the same order followed by `n_tokens` and `channels` as i32. K is deliberately fixed to 3 in this gfx1201/LFM kernel; it is not a runtime design choice. Registration uses the existing crate-local `include_str!` + `ensure_kernel_public` + `launch_external_kernel` path. No rdna-compute-wide kernel registration is introduced.

### 6.2 Exact I/O

```text
bcx:    const float [N, 3*C], row-major
state:  float       [C, 2], in/out, oldest then newest prior bx
weight: const float [C, 3], oldest-to-current tap order
out_y:  float       [N, C], row-major
N:      one sequence's consecutive time steps, not independent sequences
C:      cfg.hidden_size (1024 for the 350M vertical slice)
```

For token row `i` and channel `c`:

```text
b_gate = bcx[i, c]
c_gate = bcx[i, C + c]
x_in   = bcx[i, 2*C + c]
bx     = b_gate * x_in                     // no SiLU
oldest = state[c, 0]
newest = state[c, 1]
acc     = 0.0f32
acc    += oldest * weight[c, 0]
acc    += newest * weight[c, 1]
acc    += bx     * weight[c, 2]             // preserve all three += operations
out_y[i, c] = c_gate * acc                  // no SiLU, no bias
state[c, 0] = newest
state[c, 1] = bx
```

Each work-item owns one channel and loops `i=0..N` serially. It loads the two pre-chunk state values once, performs the same multiply/add order as `conv1d_gated_decode_f32`, and stores the final two-value tail once. No work-item may own two channels' state, and no cross-channel synchronization is required.

The split is exactly **`B_gate | C_gate | x`**. Both gates are plain multiplication. There is no activation in the scan. The pre-convolution gate multiplies `B_gate * x`; the post-convolution gate multiplies `C_gate * convolution`. The convolution is depthwise and causal. Tap 0 multiplies the oldest prior gated input; tap 2 multiplies the current gated input.

The correctness oracle initializes identical state and inputs, runs this kernel once over N rows, runs existing `conv1d_gated_decode_f32(batch=1, K=3)` N times over the same rows, and requires bit-identical `out_y` and final state. Test N values MUST straddle chunk/state boundaries: 1, 2, 3, 127, 128, 255, 256, and a two-chunk 257 case.

## 7. Dtype/rotation binding

### 7.1 Q8

Q8 projection inputs are never FWHT-rotated. Calling an MQ rotate on a Q8 cohort is a hard admission error. Q8 projection families are:

- `gemm_q8_0_wmma`;
- `gemm_qkv_q8_0_wmma`;
- `gemm_gate_up_q8_0_wmma`;
- `gemm_q8_0_residual_wmma`.

All receive `batch_size=N` and auto-route to `_gfx12` on RDNA4.

### 7.2 Dense MQ4G256

Dense `.mq4` projections are `DType::MQ4G256` (quant type 13), 136 bytes/group, 4.25 bpw. Their bytes share the HFQ4G256 kernel layout, but their activation domain is MagnumQuant/FWHT. Therefore every batched projection input MUST be rotated before calling an HFQ4-layout GEMM:

- norm-fed conv in/QKV: `fused_rmsnorm_rotate_mq_batched_for` writes the shared rotated N-row input to `operator_x_rot_batch`;
- norm-fed dense gate/up: `fused_rmsnorm_rotate_mq_batched_for` writes the rotated N-row input to `ffn_x_rot_batch`;
- conv out: `rotate_x_mq_batched_for` writes `conv_y_batch` to `conv_y_rot_batch`;
- attention out: `rotate_x_mq_batched_for` writes `fa_attn_out_batch` to `fa_attn_out_rot_batch`;
## 8. Frozen per-cohort admissions

Every admission is an exact parsed-config, topology, shape, and dtype guard evaluated before lazy allocation. `Lfm2MoeConfig` values below are authoritative runtime values; all kernel/scratch dimensions consume `cfg.*` rather than cohort literals.

### 8.1 Source-confirmed cohort dimensions

| Cohort | `cfg.hidden_size` | `cfg.vocab_size` | heads q/kv/hd | `cfg.q_dim()` / `cfg.kv_dim()` | `cfg.rope_theta` | `cfg.intermediate_size` | layers / mixer |
|---|---:|---:|---:|---:|---:|---:|---|
| 350M dense | 1024 | 65,536 | 16 / 8 / 64 | 1024 / 512 | 1,000,000 | 4608 | 16; `C C A C C A C C A C A C A C A C` |
| 1.2B Instruct + Thinking | 2048 | 65,536 | 32 / 8 / 64 | 2048 / 512 | 1,000,000 | 8192 | 16; same 10-conv/6-attention sequence |
| 8B-A1B | 2048 | 128,000 | 32 / 8 / 64 | 2048 / 512 | 5,000,000 | 7168 | 24; attention at 2,6,10,14,18,21 |

The dense HF configs store raw FF bases 6656 (350M) and 12288 (1.2B). `config.rs:226-242` applies LFM2's `2/3` auto-adjust and 256 alignment, yielding runtime `cfg.intermediate_size` 4608 and 8192. `head_dim` is absent in those JSONs and `config.rs:200-202` derives `hidden_size/num_attention_heads = 64`. The architecture does not freeze a cohort `max_seq`: all sizes use `state.max_seq`; authoritative baseline runs used 4096.

### 8.2 Admission order and dtype

| Order | Cohort | Exact projection path | Initial chunk |
|---:|---|---|---:|
| 1 | `lfm2.5-350m.q8` | every projection Q8_0; no FWHT; Section 5 | 256 |
| 2 | `lfm2.5-1.2b*.q8` | every projection Q8_0; no FWHT; same cfg-driven calls | 256 |
| 3 | `lfm2.5-350m.mq4` | projection MQ4G256 qt13; existing batched FWHT then HFQ4-layout gfx12 WMMA | 256 |
| 4 | `lfm2.5-1.2b*.mq4` | projection MQ4G256 qt13; existing batched FWHT then HFQ4-layout gfx12 WMMA | 256 |
| 5 | pinned `lfm2.5-8b-a1b.mq4`, md5 `34f35422` | non-expert Q8_0; experts MQ4G256 Path 2 | 256 |

All admissions require `arch_id=11`, `cfg.rms_norm_eps=1e-5`, Q8 embedding/tied head, Q8 KV, `cfg.conv_kernel_size=3`, `layer_types.len()==cfg.num_hidden_layers`, and loaded mixer/FFN weights matching each declared layer.

Dense admission requires `cfg.num_hidden_layers=16`, `cfg.num_experts=0`, every FFN dense, and exactly the mixer sequence in Section 8.1. 350M is the row with hidden1024/heads16/vocab65536/theta1e6/inter4608; 1.2B is hidden2048/heads32/vocab65536/theta1e6/inter8192. Q8 requires every conv in/out, attention q/k/v/out, and dense w1/w2/w3 `gpu_dtype=Q8_0`. Dense MQ4 requires every such tensor `gpu_dtype=MQ4G256`, qt13, group_bytes136.

Instruct and Thinking 1.2B artifacts are shape-identical and use the same guard.

The 8B-A1B selector is exact:

```text
cfg.hidden_size=2048; cfg.vocab_size=128000
cfg.num_attention_heads=32; cfg.num_key_value_heads=8; cfg.head_dim=64
cfg.q_dim()=2048; cfg.kv_dim()=512; cfg.rope_theta=5_000_000
cfg.num_hidden_layers=24; attention layers={2,6,10,14,18,21}; all others convolution
layers 0..=1 Dense with cfg.intermediate_size=7168
layers 2..=23 MoE with cfg.num_experts=32, cfg.num_experts_per_tok=4, cfg.moe_intermediate_size=1792
all non-expert WeightTensor projections, including router, are Q8_0 with loaded m/k shapes
each expert gate_up is `[2*cfg.moe_intermediate_size, cfg.hidden_size]` MQ4G256 qt13, group_bytes136
each expert down is `[cfg.hidden_size, cfg.moe_intermediate_size]` MQ4G256 qt13, group_bytes136
expert pointer tables each hold cfg.num_experts pointers; expert_bias is F32 `[cfg.num_experts]`
```

Norms and the loaded convolution filter are F32 at execution for every cohort. Any missing weight, mixed projection dtype within a cohort, unsupported group byte count, noncontiguous/out-of-range `conv_state_idx` or `kv_idx`, or layer-count/topology mismatch is a pre-mutation error. The md5 identifies the empirically inspected artifact but is not available through `Lfm2MoeConfig`/`Lfm2MoeWeights` and is therefore not a runtime predicate; the complete shape/dtype guard is the admission predicate.

For the pinned 8B-A1B artifact, Phase0Verify observed loaded layer-2 `self_attn.q_proj = Q8_0` and `expert[0].gate_up = MQ4G256`. Therefore:

- attention q/k/v/out, convolution in/out, dense L0-1 w1/w3/w2, and router use Q8 batched WMMA with **no FWHT**;
- after FFN RMSNorm, router consumes unrotated `ffn_tmp_batch` through Q8 GEMM and `gpu.sigmoid_f32` produces unbiased scores;
- call existing `gpu.deepseek4_moe_topk_bias_aware_batched_f32(scores, &m.expert_bias, topk_indices, topk_weights, cfg.num_experts as i32, cfg.num_experts_per_tok as i32, cfg.routed_scaling_factor, N as i32)`: selection uses `scores + expert_bias`; returned weights use selected unbiased scores normalized over top-k then multiplied by `routed_scaling_factor`;
- separately rotate the same normalized FFN input with existing `rotate_x_mq_batched_for` for MQ4 expert gate/up;
- before calling `run_moe_prefill`, require `gpu.flags.moe_grouped_gemm && gpu.arch_caps.has_wmma()`; otherwise return a pre-mutation admission error. Also require `!gpu.flags.moe_grouped_m2`, `!gpu.flags.moe_grouped_i8.unwrap_or(false)`, and `!gpu.flags.moe_grouped_i8_k4_gfx12` for this frozen base candidate. This makes `MoePrefillResolution::use_path2` mandatory and forbids silent Path-1 `_k8`, m2, or i8 fallback;
- construct `MoePrefillParams` with `batch_size=N`, `m_total_max=m_total_live`, precomputed `topk_indices_batch` / `topk_weights_batch`, rotated input, loaded `m.expert_gate_up_ptrs` / `m.expert_down_ptrs`, and the live views of persistent grouped scratch, then call `run_moe_prefill` Path 2;
- Path 2 is scatter (`moe_scatter_fused_k8`) → `gemm_hfq4g256_moe_grouped_wmma_gfx12` gate/up → `moe_gate_up_unscatter_k8` → SwiGLU plus existing batched FWHT → grouped down through the same gfx12 WMMA family → `moe_down_combine_grouped_k8` residual combine;
  Grouped gate/up uses `ffn_x_rot_batch` with `moe_sorted_slot_index` and `x_row_div=cfg.num_experts_per_tok`; a non-padding `token*k_top+slot` gathers activation row `token`, while `-1` contributes zeros.
  Grouped down uses `moe_rot_batch` with the same gather map and `x_row_div=1`; no second regroup step is permitted.
- do not call `run_moe_decode`, any `_k8` indexed GEMV production path, grouped `m2`, or grouped i8/MMQ.

The admission asserts the observed dtypes and exact fused-expert shapes at load time, so projection-MQ4 `.mq4p` or a differently quantized artifact fails explicitly and requires a separate future admission. The pinned md5 `34f35422` is provenance for the resolved dtype claim, not an unavailable API input.

Empirical tile/chunk winners are not frozen here without measurements. The design freezes the candidate sets and promotion rule: chunk `{128,256,512}`, dense gate/up BT `{4,8,12}` where supported, base grouped 16x16 MoE tile. A candidate is admitted only after end-to-end ABBA improvement over noise in two adjacent prompt buckets with correctness intact. Until then, initial value 256 and existing default BT selection are the candidate, not a performance claim.

## 9. Control-plane integration

During Phases 1-3:

```text
use_batched = gpu.arch_caps.is_gfx1201()
              && HIPFIRE_LFM2_PREFILL_BATCH == "1"
```

`generate_lfm2moe` and the arch-11 `bench_prefill` arm make this selection before any prompt token is processed. The eager Phase0 path is the exact else branch. The environment flag is opt-in until Phase-4 certification; only the conductor may freeze a default-on-with-`=0` opt-out cutover.

The batched function is not graph-captured in the vertical slice. `HIPFIRE_FORWARD_LOWERED` and `HIPFIRE_LFM2_GRAPH` remain eager diagnostic controls and do not alter the batched contract.

## 10. Vertical-slice implementation order

No later cohort work starts until 350M-Q8 passes every correctness gate and demonstrates the batched path is active.

1. Add the frozen API/export and additive scratch with the path still opt-in and unreachable until complete.
2. Add chunk input upload, embedding, absolute positions, entry validation, and state commit.
3. Add `conv1d_gated_scan_n_f32` and pass its bit-parity oracle before connecting it to the model.
4. Add the Q8 attention branch exactly as Section 5, including batched KV writes and hd64 causal flash.
5. Add Q8 dense FFN and residual bindings.
6. Add final-row-only head and one logits D2H.
7. Prove 350M-Q8 final-logit, per-layer hidden, KV, convolution-state, and `n_tokens` parity; prove pp > tg and rocprof evidence of gfx12 WMMA before accepting the vertical slice.
8. Admit 1.2B-Q8 by shape only, then dense MQ4 cohorts with existing batched FWHT, then 8B-A1B Path 2 last.

## 11. Phase-2 subsystem lanes and exclusive file ownership

Phase 1 creates the helper modules below so Phase 2 has disjoint ownership. Public entry and chunk signatures remain in `forward.rs` as frozen; helpers are crate-private.

| Lane | Exclusive production files during lane work | Frozen responsibility | Forbidden overlap |
|---|---|---|---|
| Conv scan | `kernels/src/conv1d_gated_scan_n.gfx1201.hip` | scan mapping, vector loads, tail staging; symbol and math contract unchanged | no Rust orchestration, attention, GEMM, or state-layout edits |
| Attention | `crates/hipfire-arch-lfm2moe/src/prefill_attention.rs` | QKV, Q/K norm, RoPE, KV writes, hd64 causal flash, out residual | no `forward.rs`, state struct, conv source, dense/MoE helper edits |
| Dense projection/FFN | `crates/hipfire-arch-lfm2moe/src/prefill_dense.rs` | Q8 and MQ4/FWHT dense call sites, gate/up BT choice, residual down | no attention, MoE, state sizing, or chunk loop edits |
| MoE Path 2 | `crates/hipfire-arch-lfm2moe/src/prefill_moe.rs` | LFM sigmoid+bias routing adapter and existing grouped-WMMA Path-2 invocation | no decode `_k8`, no dense/attention/state ownership, no m2/i8 |
| State/scratch/chunking | `crates/hipfire-arch-lfm2moe/src/lfm2moe.rs`, `crates/hipfire-arch-lfm2moe/src/forward.rs` | allocation, env capacity, chunk loop, cross-chunk state, final head | no stage helper or HIP kernel edits |

`crates/hipfire-arch-lfm2moe/src/kernels.rs`, `src/lib.rs`, and daemon selection are frozen after Phase 1 and are conductor-owned integration files, not lane files. Lane branches do not edit them. Shared parity and benchmark harnesses are conductor-run and are not lane-owned. Integration is by reviewed cherry-pick, never rsync.

## 12. Correctness and promotion gates

The isolated convolution oracle supplies identical F32 `bcx`, initial state, and weights to both paths. Only this isolated oracle is bit-exact: `out_y.to_bits()` and final `state.to_bits()` MUST equal N sequential `conv1d_gated_decode_f32(batch=1,K=3)` calls. Its scan uses `acc=0.0f32` and all three ordered `+=` operations.

End-to-end batched-versus-eager comparisons are **not** bit-exact: eager Q8 GEMV consumes F32 activations while the WMMA GEMMs source-verified above convert X to F16. All compared values must be finite, and the frozen pass thresholds are:

| Quantity | Q8 cohorts | dense MQ4 / A1B cohorts |
|---|---:|---:|
| every per-layer hidden row cosine | `>= 0.999` | `>= 0.999` |
| every per-layer hidden row max-abs | `<= 0.10` | `<= 0.15` |
| dequantized written K and V, per layer/token cosine | `>= 0.999` | `>= 0.999` |
| dequantized written K and V max-abs | `<= 0.10` | `<= 0.15` |
| final convolution-tail cosine per layer | `>= 0.999` | `>= 0.999` |
| final convolution-tail max-abs | `<= 0.10` | `<= 0.15` |
| final-token logits max-abs | `<= 0.05` | `<= 0.10` |
| `KL(softmax(eager) || softmax(batched))`, mean across oracle prompts | `<= 1e-4` | `<= 5e-4` |
| same KL, maximum prompt | `<= 1e-3` | `<= 5e-3` |

The exact gfx1201 `lfm2.5-350m.mq4` cohort (artifact md5
`cb5284b8ad5c6f9e4ca859c0aff0bcd0`, Q8 KV) has a narrower,
reviewer-approved exception to the dense-MQ4 column:

| Quantity | 350M MQ4 gfx1201 ceiling |
|---|---:|
| dequantized written K and V max-abs | `<= 0.20` |
| final-token logits max-abs | `<= 0.40` |
| mean KL across the oracle matrix | `<= 1e-3` |

All cosine limits, hidden/conv max-abs limits, maximum-prompt KL `<=5e-3`,
and exact state/cursor/position/chunk checks remain unchanged. This exception
is based on actual-shape N=2 plus BT4/BT8/BT12 HFQ4 gfx12 WMMA channel tests,
the full enabled-WMMA oracle matrix, and an `HIPFIRE_FP16=0` causal control.
It MUST NOT be applied to 1.2B or A1B cohorts without independent evidence.

`kld_logits.rs` currently reports rather than enforces; the batched parity harness MUST compute its existing KL formula and fail on the numeric mean/max limits above. `state.n_tokens`, the set of absolute KV write positions, token ordering, layer-to-cache/ring slot mapping, and chunk coverage remain exact discrete comparisons. For A1B, top-k expert indices must also match eager exactly; top-k weights use the same cohort numeric tolerances.

The oracle matrix MUST include 1, 2, 3, 127, 128, 255, 256, and 257 tokens, and fixed prompts through the LFM serve harness. A failure of any single bound rejects the cohort.

Performance promotion requires fresh-process ABBA, a gain above the gfx1201 noise band in at least two adjacent prompt buckets, no shared-cohort regression, and rocprof evidence that the intended WMMA symbols execute. No microkernel-only win is promoted.

## 13. Review-blocker resolutions and remaining OPEN RISKS

### Closed by this source-verified revision

1. **Call signatures:** every Section-5 call now uses source types: upload byte slices, `WeightTensor.buf`, `GpuTensor::sub_offset`, and `DeviceBuffer` D2D arguments.
2. **Eager VRAM isolation:** state construction stores `prefill_batch=None`; scratch is allocated lazily only after the same gfx1201 + exact opt-in gate.
3. **Conv operation identity:** the scan begins at `0.0f32` and performs the source kernel's three ordered `+=` operations.
4. **Grouped bound:** LFM owns the aligned capacity/live helper `align_up(total_slots + min(total_slots,n_exp)*15,16)`; both results are multiples of 16, so scatter initializes every tile the grouped GEMM can launch.
5. **Path-2 enforcement:** disabled grouped-GEMM/WMMA or enabled m2/i8 research flags are pre-admission errors, never `_k8` fallbacks.
6. **Numerical policy:** only isolated conv is bit-exact; explicit hidden/KV/conv/logit/KL thresholds are frozen in Section 12.
7. **Embedding bounds:** every token ID and i32-derived launch value is validated before upload or lazy allocation.
8. **Flash partials footprint:** fixed `M=min(16,B)` uses source capacity-based sub-batching; at state.max_seq8192 it is 4.125 MiB for 350M and 8.25 MiB for 32-head cohorts.
9. **Chunk capacity ceiling:** configured capacity above 512 is a pre-allocation error; the environment cannot escape the frozen sweep ceiling.
10. **Per-cohort dimensions:** 350M/1.2B/8B config values and parser-adjusted FFN widths are pinned separately; all scratch and kernel arguments derive from `cfg.*`.

### Remaining OPEN RISKS for adversarial review

1. **Attention bound naming.** This contract intentionally passes both `max_seq` and `max_ctx_len` as the chunk-visible causal bound `p+N`; review must confirm no path expects physical KV capacity in `max_ctx_len`.
2. **Conv compiler contraction.** The source operations now match exactly, but compiler FMA/contraction settings must match the decode kernel. The isolated bit oracle decides.
3. **In-place Q/K RMSNorm aliasing.** The source explicitly permits x/out aliasing; N>1 still requires oracle coverage before promotion.
4. **MQ4 rotate/shape coupling.** MQ4G256 shares HFQ4 bytes but not activation semantics. Every projection must prove the correct rotate exactly once; double rotate and missing rotate are both silent-corruption risks.
5. **A1B routing adapter.** The existing bias-aware batched top-k method freezes semantics, but route indices/weights still require eager parity.
6. **Grouped scratch runtime validation.** Checked capacity/live arithmetic plus `m_total_live <= m_total_capacity` are mandatory before scatter/GEMM.
7. **A1B artifact identity.** The Q8 non-expert/MQ4 expert result is authoritative for md5 `34f35422`; a future `.mq4p` must fail the shape/dtype admission.
8. **Failure atomicity.** GPU errors can leave KV or convolution tails partially advanced. The API provides no rollback; callers must reset and must not retry the same batch in place.
9. **State/head handoff.** Batched bodies do not touch eager scratch, but the final row is copied to existing `state.h`; verify the post-prefill consumer contract.
10. **VRAM by cohort.** Lazy scratch B=256 must be measured before B=512 is admitted. Chunk 512 is only a candidate.
11. **Lowered/graph separation.** Compare the explicit batched path against normal lowered eager execution and its hand-loop escape hatch.
12. **Phase-2 file split.** If implementation keeps every stage in `forward.rs`, lanes are not independent and must not fan out.

## 14. Provenance

No GPU or timing command was used and no production source file was edited. The revision's call and allocation contracts were verified directly against the checkout named at the top:

- `crates/hip-bridge/src/ffi.rs:804-821` — H2D requires `&[u8]`;
- `crates/hipfire-runtime/src/llama.rs:499-518,731+` — `WeightTensor.buf` and `weight_gemv`;
- `crates/rdna-compute/src/dispatch.rs:93-157,1120-1142,400-405` — `GpuTensor::sub_offset`, D2D `DeviceBuffer` API, `Gpu.flags`;
- `crates/rdna-compute/src/{embedding.rs:146+,norm.rs:91-180,369+,658+,gemm.rs:18748+,18962+,19050+,19123+,attention.rs:1502+,1837+}` — every Section-5 Gpu signature and batch argument;
- `crates/hipfire-arch-lfm2moe/src/{arch.rs:47-49,lfm2moe.rs:201-250,320-410,1145-1300,config.rs:200-269}` — all-device state construction, loaded fields, current eager state, head-dim derivation, RoPE parsing, and dense FFN auto-adjust;
- `kernels/src/{conv1d_gated_decode.hip:35-70,embedding_q8_batched.hip:10-30,moe_scatter_fused_k8.hip:51-59,gemm_hfq4g256_moe_grouped_wmma.gfx12.hip:38-49}` — exact convolution operations, unchecked embedding row address, and grouped tile-id alignment requirement;
- `crates/hipfire-arch-qwen35/src/qwen35.rs:4803-4845,5860-5879,8018-8029`, `crates/rdna-compute/src/{attention.rs:3245-3266,gemm.rs:11040-11064}` — partials-capacity sub-batching and authoritative aligned grouped bound;
- `crates/hipfire-dispatch/src/families/moe.rs:429-645`, `crates/rdna-compute/src/{feature_flags.rs:142-155,455-476,moe.rs:811-857}` — Path-2 feature gate/fallback and bias-aware batched top-k.
- `/mnt/nas/kaden/cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/7728373d9f752dc3669ee3bf70786aef397874bb/config.json`;
- `/mnt/nas/kaden/cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/868df74dd56ff8a0c2ac5dbf281690c2dbebe4c9/config.json`;
- `/mnt/nas/kaden/cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Thinking/snapshots/95053d21d8e0b7ca99421a2127ae39c64f685ff3/config.json`;
- `/mnt/nas/kaden/cache/huggingface/hub/models--LiquidAI--LFM2.5-8B-A1B/snapshots/5673e0de372b64331504de73bbbc33b0dde71903/config.json`.

The pre-existing scouts remain supporting provenance for cross-file discovery, and Phase0Verify's runtime dtype observation was relayed by `NervousAnteater`/Main. The revised proposal incorporates the corrected MQ4/FWHT and pinned A1B dtype facts.
