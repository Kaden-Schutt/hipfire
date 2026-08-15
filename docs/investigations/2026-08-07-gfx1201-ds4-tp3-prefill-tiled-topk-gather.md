# gfx1201 DeepSeek V4 TP3 prefill: tiled top-K KV gather

Date: 2026-08-07  
Branch: `ds4-gfx1201-opt`  
Promotion commit: `0cfb8a4b8`  
Kernel/micro commit: `1f008ce2c`

## Result

The canonical DeepSeek V4 Flash 0731 MQ2R TP3 product fixture improved from
360.6392 to **422.3075 prefill tok/s** (+17.10%). Decode remained flat at
53.2448 tok/s. The 512-token decoded answer is byte-identical to the accepted
golden output.

The route is an automatic code default only when the loaded GPU architecture is
exactly `gfx1201`. `HIPFIRE_DEEPSEEK4_GFX1201_TOPK_GATHER_TILED=0` is retained
only as a rollback/A-B override. No Qwen, gfx1100, gfx1151, weight, format,
sampling, expert-count, KV, or speculation behavior changes.

## Mechanism

The portable batched gather launches one 512-thread block for every
`(batch_row, selected_k)` pair. Its cache reads are contiguous, but each wave's
output stores are separated by the 512-float output stride.

`deepseek4_topk_kv_gather_batched_tiled_gfx1201` stages a padded 32x32
`[selected_k, head_dimension]` tile in LDS and writes its transpose. Both cache
reads and dimension-major output writes are contiguous, and the padded LDS row
removes transpose bank conflicts.

## Micro screen

Product shape: B=1024, K=512, D=512, N=512, with visibility increasing from
256 to 512 and `-1` sentinels matching the second canonical prefill chunk.

| Route | Time |
|---|---:|
| Portable gather | 7,645.969 us |
| gfx1201 tiled gather | 1,857.910 us |
| Speedup | **4.115x** |

Full raw-bit comparison at B=8 reported zero mismatches.

## Product fixture

- Model: `/home/kaden/models/deepseek-v4-flash-0731.mq2r`
- Prompt: `benchmarks/prompts/ds4-gfx942-ar-2048.txt`
- Prompt MD5: `25e22faef15a20ae53501f1956e62b79`
- Effective prompt/context: 2052 tokens
- Generation: 512 tokens, greedy, thinking off, speculation off
- Experts/token: checkpoint default 6
- Devices: three gfx1201 R9700 GPUs, TP3
- KV request: q8; DS4 currently uses its F32 contiguous cache implementation

| Metric | Previous accepted | Candidate | Delta |
|---|---:|---:|---:|
| Prefill | 360.6392 tok/s | **422.3075 tok/s** | **+17.10%** |
| Decode | 53.1964 tok/s | 53.2448 tok/s | flat |
| Prefill wall | 5,689.897 ms | 4,859.019 ms | -830.878 ms |

Correctness:

- assistant SHA-256:
  `b6255240b6ccd34d621f152cc9898c71c340fa9393680ddb3d3efde2172b9c41`
- generated tokens: 512
- answer words: 395
- empty responses: 0
- attractor failures: 0

Binary:

- daemon SHA-256:
  `819a674096001d73bc1e8c01c8ccf311baae6902b0b6eefc3e04a045d53e10cb`

## Evidence

- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-topk-gather-tiled-micro/`
- `hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-topk-gather-tiled-product/`

## Skipped

No repeated product samples, long-context, TP4, weights, format, quality,
sampling, top-k, expert-count, KV, speculation, retained PM4, gfx1100/gfx1151
runtime, or Qwen runtime were changed or re-certified at this checkpoint. The
next gate re-profiles this promoted TP3 route before selecting another lever.
