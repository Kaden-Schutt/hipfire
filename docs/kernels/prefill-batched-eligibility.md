# Batched prefill eligibility & KVarN/TQ readiness (gfx1103)

Handoff note for the KVarN / TurboQuant(TQV) runtime-KV work. Summarizes the
prefill investigation so the next session doesn't re-derive it.

## Key finding: prefill is NOT a kernel problem

The batched prefill path (`qwen35::forward_prefill_batch`) is **fast on gfx1103
— ~1900 tok/s warm** for qwen3.5-0.8b (32× the per-token fallback). It uses real
batched kernels (`gemm_hfq4g256_residual_mmq_full_set`, `attention_q8_0_kv_batched`,
batched/chunked SSM, lm_head once on the last token). No new prefill kernel is
needed.

The slow prefill we chased was a **cascade**, not a kernel gap:

1. MQ4/Q8 models keep 1-D **norm tensors at BF16**.
2. `hfq_has_bf16_weights` (any `quant_type==16`) → the old load rule **forced
   `kv_mode=fp32`**.
3. F32 KV → `prefill_batch_pbs_eligible` fails the **`kv_f32` guard** (F32 KV has
   only a `BatchEq(1)` kernel → MissingImpl batched) → per-token fallback.
4. The per-token fallback also recomputed lm_head every token (37%).

(Earlier "168 tok/s with q8 KV" was a **cold-JIT artifact** — the batched kernels
had never been compiled; warm is ~1900.)

## Fixes already landed (chaingun)

- `1effd1d7` perf(qwen35): skip lm_head for non-final prefill tokens. The lowered
  forward (`forward_scratch_layers_lowered`) ignored the no-logits request; added a
  `needs_logits` param. 59 → 105 tok/s on the per-token path; decode unchanged.
- `70f3ae9b` fix(load): un-force fp32 KV for quantized models. Replaced
  `hfq_has_bf16_weights` with **`hfq_is_bf16_dominant`** (majority of 2-D weight
  tensors BF16) for the KV-force decision:
  - BF16-dominant model → force fp32 KV (unchanged; correct).
  - Quantized model → **honor explicit `kv_mode`**; empty → fp32 default (for now).
  - DeltaNet **state** still forced FP32 for any-BF16 models (`is_bf16_artifact`
    retained for `dn_quant`) — orthogonal to KV; preserves tiny-model recurrent
    correctness (cumulative error → collapse).
- `25befcb1` diag: `HIPFIRE_DEBUG_PREFILL_ELIGIBLE=1` prints the eligibility
  decision (`final/base/kv_f32/dn_quant/kv flags`). Use it to verify KVarN → eligible.

Net: `--kv q8` (or any quantized KV) now → eligible → ~1900 tok/s prefill, FP32
DeltaNet state intact, output coherent.

## Two distinct precision axes (do not conflate)

| | DeltaNet state | Attention KV cache |
|---|---|---|
| type | `DeltaNetState` (`qwen35.rs`) | `KvCache` (`llama.rs`) |
| knob | `StateQuant::{FP32,Q8,Q4}` (`state_quant`) | `kv_mode` |
| layers | linear-attn (gated DeltaNet SSM) | full-attention |
| error | **cumulative** → tiny models need FP32 | per-position → tolerates quant |

Quantizing the KV does **not** touch the required FP32 DeltaNet state.

## KVarN / TQ readiness

- **`KvCache` lives in `llama.rs`** (shared, arch-agnostic; ~45 constructors).
- **Batch-prefill-eligible KV modes** (set `quantized=true`): q8, int8/int8c,
  hfq4/hfq8, asym2/3/4, fwht2/3/4. NOT eligible: fp32, q4.
- **KVarN today** = CPU codec only (`hipfire-quantize/src/kvarn.rs`:
  `variance_normalize` Sinkhorn + `quantize_tile` + `pack_kvarn_tile`) + sim toggle
  `HIPFIRE_KVARN_SIM`. **No runtime KV mode.**
- **TurboQuant/TQV today** = calibration tables (`calib/tqv/*.json`) +
  `scripts/tqv_fit_tables.py`. No runtime KV mode.

### To wire KVarN (and TQ) as a runtime KV mode
1. `quant_kvarn` flag + `new_gpu_kvarn` constructor on `KvCache` (set
   `quantized=true` → automatically batch-prefill eligible via the proven path).
2. GPU kernels for the KVarN record (4-bit nibbles + per-row `scale_abs`/`zp_abs`
   f16 + per-col `s_col` f16; dequant = `(q·scale+zp)·s_col`): a
   `kv_cache_write_kvarn` and attention-read for **batched (prefill) + decode**,
   **no-LDS** (gfx1103 LDS-hang hazard).
   - Sinkhorn `variance_normalize` operates on a **tile**, so KVarN KV must be
     quantized **block-wise** (can't normalize a single per-token row) — like the
     asym modes' per-block quant.
3. **Reuse path:** KVarN (4-bit K + dual scale) is closest to the **asym4 family**
   (`kv_cache_write_asym_k_*`, `attention_flash_asym4_*`) — likely cheapest to
   implement KVarN as an asym4 variant (swap givens rotation for the Sinkhorn
   `s_col`/`s_row` scales) rather than greenfield kernels.
4. `kvarn.rs` CPU codec is the **golden oracle** (golden-hash test pattern, like
   the generic kernel library).
5. Default `kv_mode` flips to KVarN once it's a runtime mode (user decision).

## Repro
```
# default (fp32 KV, per-token): ~104 tok/s
printf '{"type":"load","model":"~/.hipfire/models/qwen3.5-0.8b-mq4+.hfq"}\n{"type":"bench_prefill","tokens":512}\n' | \
  LD_LIBRARY_PATH=/opt/rocm/lib hipfire-daemon
# quantized KV (eligible, batched): ~1900 tok/s warm
#   ...,"params":{"kv_mode":"q8"}  + HIPFIRE_DEBUG_PREFILL_ELIGIBLE=1
```
