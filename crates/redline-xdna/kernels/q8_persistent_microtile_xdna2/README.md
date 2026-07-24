# Persistent Q8_0 microtile proof for XDNA2

This diagnostic artifact proves the retained-panel execution order needed by
the production Hipfire overlay:

1. Consume one native Hipfire Q8_0 weight panel (`W[16,64]`).
2. Decode it once into a placer-accounted 2 KiB AIE-local `Buffer` in BF16
   MMUL layout.
3. Keep that panel resident while one command processes 2, 8, or 16 logical
   activation chunks.
4. Accumulate and emit F32 through the native BF16 MMUL path.

Activations and outputs use explicit AIE tile-major staging layouts. The
artifact is tagged `q8_w8a16_microtile_diagnostic`, so it cannot be admitted by
the production `q8_w8a16_f32` loader.

Build the three variants inside an activated mlir-aie/Peano environment:

```sh
make CHUNKS=2
make CHUNKS=8
make CHUNKS=16
```
