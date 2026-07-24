# XDNA2 native-Q8 full-array projection diagnostic

Fixed gate shape:

- batch M=256
- reduction K=2048
- outputs N=2048
- native Hipfire Q8_0 `W[N,K]`
- BF16 activations
- F32 output
- eight columns and 32 compute cores

Each core accumulates a 64x64 C tile. A 64x64 BF16 activation tile and a
native packed Q8 `W[64,64]` tile stream for every K=64 step. The packed weight
tile is gathered directly from the row-major model buffer, decoded into an
explicit allocator-visible local BF16 buffer, and consumed by 8x8x8 MMUL.
K=512 A/B panels are double-buffered in memory-tile SRAM so external transfers
remain bank-local and the smaller K=64 objects are formed inside the array.

The diagnostic defaults to AIE2P's BFP16-emulated BF16 throughput mode;
`BFP16_EMULATION=0` selects native BF16 for comparison.
`Q8_FULL_ARRAY_COMPUTE_ONLY` and `Q8_FULL_ARRAY_DMA_ONLY` are profiling-only
build flags and are never production artifacts.

The entire 256x2048 projection runs under one retained command. This artifact
is diagnostic and must not be admitted as production `q8_w8a16_f32`.
