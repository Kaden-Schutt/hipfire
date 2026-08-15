# DeepSeek-V4 Flash reference index-space conventions

Source of truth: `.codeinsight+research/ds4-parent-ref/inference/model.py`
`Attention.forward` (~498-540), `get_window_topk_idxs` (260-271),
`get_compress_topk_idxs` (274-283), `Indexer.forward` (408-439).

Measured by TorchOracle2 against `model.py` imported verbatim.

## Prefill (`start_pos == 0`)

| Item | Reference |
|------|-----------|
| KV fed to `sparse_attn` | `cat([token_kv[B,S,D], kv_compress[B,S//ratio,D]], dim=1)` |
| KV length | `S + S//ratio` |
| Window indices | **Absolute token positions** in `[0, S)` via SWA formula |
| Compress indices | `slot + offset` with **`offset = kv.size(1) = S`** (length of token_kv *before* cat) |
| Unified ranges | window `[0, S)`; compress `[S, S + n_comp)` — **disjoint** |
| Ring write | Primes `kv_cache` for later decode only; **not** read by prefill `sparse_attn` |
| Ring layout after prefill | last `W` tokens rotated: `cutoff=S%W`; `cache[cutoff:W], cache[:cutoff] = kv[:,-W:].split(...)` |

## Decode (`start_pos > 0`)

| Item | Reference |
|------|-----------|
| KV fed to `sparse_attn` | `self.kv_cache[:bsz]` single buffer |
| Buffer layout | `[0:window)` = SWA ring; `[window:]` = compressed (`compressor.kv_cache` alias) |
| Window indices | **Ring slots** (rotated order when `start_pos >= window-1`) |
| Compress indices | `slot + offset` with **`offset = window`** |

## Top-k budget

```
topk_idxs = cat([window_idxs, compress_topk_idxs], dim=-1)
```

- `index_topk` applies **only** to the compressed half.
- SWA window is **EXEMPT** from the 512 budget.
- At `seq=1024`, ratio-4: `n_comp=256 < 512` → top-k is a **no-op** (selects all visible compressed slots).
- Filtering needs `seq > index_topk * ratio` (= 2048 for ratio 4; 65536 for ratio 128).

## Parent port (`parent/attention.rs`)

- Split buffers: `swa_staged` + `main_kv_cache` / `topk_staged` (not unified).
- Passes `offset=0` into indexer; indices are direct `main_kv_cache` rows.
- Joint softmax: `n_total = n_valid_swa + n_active_topk` (`deepseek4_attn_swa_topk_batched`).
- `n_active_topk[r] = min(index_topk, n_vis_comp)` — compressed-only budget (matches reference scope).
- `max_n_compressed = max(max_rows, 512)` — at 1024-token prefill holds 256 ratio-4 slots.
- When `n_comp <= 512`, indexer top-k fast path writes **identity** `0..n_comp-1`; causal mask makes `-1`s trailing so packing into `n_active` is safe.

## Cardinality at seq=1024 ratio-4 (both sides)

| pos | win | comp | joint |
|-----|-----|------|-------|
| 400 | 128 | 100 | 228 |
| 500 | 128 | 125 | 253 |
| 600 | 128 | 150 | 278 |
| 800 | 128 | 200 | 328 |
| 1000| 128 | 250 | 378 |
| 1023| 128 | 256 | 384 |

Max joint = 384 < 512. Even a wrong joint-scoped budget of 512 would **not** truncate at 1024/ratio4.

## Implication for the [448,512) top-1 step

Selection / budget-scope cannot explain the latching drop at ~512 on the 1024-token parent capture. Next suspects: compressed-KV **content** (compressor, RoPE phase `compress_rope_theta=160000`), SWA staging vs absolute gather equivalence, or joint softmax value path.
