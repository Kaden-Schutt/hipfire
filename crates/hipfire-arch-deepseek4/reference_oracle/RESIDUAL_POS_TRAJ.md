# Reference per-position residual trajectory (43 layers, seq=1024, GPU)

Date: 2026-08-02. Script: `residual_pos_traj.py` (stream-one-Block).
Device: MI300X, torch 2.13.0+rocm7.2. model.py imported verbatim.

Artifact: `artifacts/residual_pos_traj.json.compact.json` (+ full dense JSON on
mi300x `/tmp/residual_pos_traj.json` and `artifacts/` if synced).

## Method

- `Transformer(n_layers=0)` shell for embed + global dtype setup
- For each layer_id 0..42: construct `Block(layer_id)`, load that layer only
  (incl. 256 experts packed FP4), `block.to(cuda)`, forward, free
- Capture dense per-row HC residual L2 after every layer (length 1024)
- Peak VRAM ~4.7 GiB (one layer + activations)

## Headline numbers

| Metric | Value |
|--------|-------|
| h0 global L2 | 256.33 |
| L42 global L2 | 124858.6 (×487 from embed) |
| mean late/early excl pos0 over 43L | **1.026** |
| L0 late/early | 1.046 |
| L42 late/early all rows | 0.486 (dominated by pos0) |
| L42 late/early excl pos0 | **1.311** |
| L38 pos0 / median | **269×** (117077 vs 435) |

## Shape verdict

**Reference residual magnitude is NOT position-degrading.**

- Excluding the known pos0 massive-activation outlier, late/early ≈ 1 across
  the stack (mean 1.026, min ~0.80, max ~1.31).
- 64-wide buckets at L2/L20 are flat across positions.
- L42 buckets excl first: mild mid-sequence hump (820 at [512,576)) then
  settle ~750–800 — **not** a collapse after 512.
- L37→L38 global L2 19k→118k is **entirely pos0** (Main already retired this).

Therefore the parent top-1 collapse after ~448–512 tokens is **not** explained
by reference residual growth with position. If parent per-pos residual tracks
this flat shape, the defect is in **attention content** (compressor KV / joint
softmax value path / SWA staging equivalence), not residual accumulation.

## Selected layer probes (row L2)

```
L  ratio  early128  late128  L/E_all  p0       p512    p1023
0  0      14.45     15.10    1.046    13.74    16.21   11.21
2  4      15.08     14.42    0.956    24.68    15.02    9.42
10 4      42.43     25.41    0.599    1821.8   32.59   11.31
20 4      167.2     123.7    0.740    3851.9   130.3   104.2
30 4      319.6     321.1    1.005    7323.6   445.9   336.1
38 4      1325      457.1    0.345    117077   585.9   469.7
42 4      1519      738.4    0.486    122876   972.0  1023.7
```

(L10+ L/E_all depressed by pos0; see excl-pos0 table in analysis log.)
