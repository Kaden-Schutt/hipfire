# R13 — memtile routing works, but one column ≠ a win

R12 hit the shim-channel wall (a column shim directly feeds ~1 core). R13 proves the
**memtile distribute/broadcast/join mechanism** that gets past it: `r13_column.mlir` routes
one column's feed shim→memtile(0,1)→4 cores, using `aie.objectfifo.link`:
- **distribute** — A super-block split into 4 core-slices (`link [@a_sh] -> [@a_0..3] ([] [0,6144,12288,18432])`)
- **broadcast** — one W block sent identically to all 4 cores (`link [@w_sh] -> [@w_bc] ([] [0])`)
- **join** — 4 cores' C gathered back to the shim (`link [@c_0..3] -> [@c_sh] ([0,2304,4608,6912] [])`)

Built and ran correct (c0 = 256). But **0.39 TOPS** — *below* R12's 4-cores-across-4-columns
(0.75): packing 4 cores behind **one** shim/memtile shares a single DDR path, so the W-reuse
intensity gain (→64 mac/byte) is wasted on a bandwidth-starved column. The win needs all 4
column shims **and** cross-column A reuse → the full 4×4 (`../r14/`).

Repro: build `r11.o` (LM=6 LN=12 KT=16), `aiecc r13_column.mlir`, dispatch via npu_gemm_bench
(A=12582912 W=6291456 C=18874368, macs=512·4·6·12·16·512, expect 256).
