# Q8 KV long-context prefill: no-LDS-cap batched-masked flash attention

Branch `fix/q8-batched-masked-no-lds-cap` (off master).

## Problem

`attention_q8_0_kv_batched_masked` stages `scores[max_ctx_len]` in LDS
(`shared_mem = (max_ctx_len + block + head_dim) * 4`). At 120k ctx that's
~481 KB — far past the ~64 KB hardware LDS limit. The guard
`LDS_CTX_LIMIT = 15000` falls back to a **per-position loop** (one
`attention_flash_q8_0` launch per query row) above 15k. Correct, but it
serializes prefill into `n` separate kernels per chunk per layer.

The 27B-awqg serve config runs **Q8 KV at 122880 ctx** (asym3 quality is
too compromised on the deep stack — confirmed by user), so every long
prompt hits this fallback. asym3/asym4/fwht already avoid the cliff via
the tiled `flash_partials` reduction; only Q8 lacked a batched no-cap
variant.

## Fix

New `attention_flash_q8_0_tile_batched.hip` — the asym3-batched tile
skeleton (online-softmax tiling, LDS = `tile_size` only, batched via
`blockIdx.z`, tree-mask support) with K/V dequant swapped to Q8_0 and
the Givens rotation dropped. Reuses the existing K/V-format-agnostic
`attention_flash_asym_reduce_batched`. Dispatch wrapper
`attention_flash_q8_0_batched_masked` reuses the shared
`launch_asym_flash_batched` (cos/sin args passed as dummy `q`, ignored).
Wired into 3 call sites: `qwen35.rs` (dense FA + MoE FA) and `llama.rs`
generic prefill.

## Correctness gate — NIAH (needle-in-a-haystack), 9B Q8 on gfx906

| fixture | tokens | path exercised | result |
|---|---|---|---|
| niah_16k | 10881 | below cliff (old batched-masked) | PASS |
| niah_32k | 21551 | **>15k → new batched kernel** | **PASS** |

Both recover `mauve-velociraptor-7741`. The 32k run (21551 tok) is the
definitive gate: correct full-context attention through the new kernel.

## Perf — microbench (q8_batched_attn_microbench, gfx906)

Single FA-layer scale, NEW batched vs OLD per-position loop (the
fallback this replaces), median of 5, Qwen3.5-9B FA shape (nh=40,
nkv=8, hd=256):

| n | ctx | NEW (ms) | OLD (ms) | speedup |
|---|---|---|---|---|
| 128 | 20000 | 117.3 | 186.3 | 1.59× |
| 512 | 20000 | 464.5 | 741.6 | 1.60× |
| 512 | 32000 | 750.0 | 1148.2 | 1.53× |
| 256 | 60000 | 707.3 | 1054.9 | 1.49× |

Consistent **1.5–1.6× over the fallback**, stable across shapes.

## Caveat — absolute speed is poor

NIAH 32k full prefill: **2,222,498 ms = 37 min (10 tok/s)** for 21551
tokens. The kernel is correct and beats the fallback, but it is a
scalar wave32-per-tile design (`__launch_bounds__(32, 16)`, sequential
per-token dot products). This is acceptable for the fix's scope —
remove the cliff, beat the fallback, stay correct — but the kernel is
the next optimization target (see acceleration plan).

No accelerated (MFMA/WMMA/dot4) long-context FA exists in the codebase
for ANY KV mode: asym3/asym4/fwht tile kernels are all the same scalar
wave32 shape. The only WMMA FA is the dflash-WMMA family
(`attention_dflash_wmma_*`, f16 KV, RDNA3-only) used for spec-decode
verify and the dots-ocr v3_causal text-prefill kernel — neither
dequantizes Q8/asym3 nor handles head_dim=256.
