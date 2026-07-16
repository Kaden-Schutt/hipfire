# TODO

## Redline retained-replay optimization

Ordered after the first product PM4 replay win. Every arm keeps automatic GPU
clocks, exact-output/coherence gates, and the sampled eight-turn serve harness.

1. **Fence/coherence specialization (complete).** Preserve the HIP-to-PM4 entry
   acquire, repeat-interleave/RoPE acquires, and terminal compute idle;
   fused-SiLU/MQ-rotation acquires were redundant and are removed by the
   certified `required-only` default. PM4 compute waits are now derived from
   allocation-wide read/write effects across the full outstanding frontier;
   unknown kernels or pointers fail closed instead of relying on kernel names.
2. **Stateful PM4 encoding (complete).** Queue-global invariant register writes
   are retained by default, reducing the FWHT3 tape by 30.4% with a measured
   +0.61% at 8K and neutral `tg128`. Full program/resource/workgroup retention
   remains opt-in because it reduced the tape further but slightly regressed
   `tg128`.
3. **GFX12 temporal/cache policy (complete).** Default RT raw-buffer addressing
   covers the zero-scratch hot HFQ4 family. The indexed MoE gate/up path uses one
   expert-wide SRD, scalar row/group offsets, and separate gate/up load-consume
   stages; this clears the earlier private-memory spill.

### Next ordinary-AR work

1. **MoE gate/up buffer RT (complete).** Zero-scratch, bit-identical, and
   isolated to gfx1201. The corrected real-grid microbench and sampled
   eight-turn serve battery both retain matching outputs.
2. **Context-bucketed retained tapes (closed: neutral/regressive).** A retained
   AQL/PM4 geometry patch reduced every FWHT3 FA grid from the 32K physical
   capacity to the exact live 128-token tile bucket. It was bit-identical but
   measured −0.005% at 8K and −0.72% at 2K, where 240/256 tile rows were
   removed. Empty-tile early exits are effectively free on this workload; the
   mutable replay machinery was not retained.
3. **Suballocation-aware dependency boundaries (closed: no candidates).** The
   live 833-launch tape has 832 covered boundaries and 130 resource-independent
   edges. An exact pointer-start census found zero remaining waits caused only
   by different subviews of one allocation: every blocked edge includes a true
   read/write or write/write dependency at the same device pointer. Keep the
   allocation-wide fail-closed policy; no additional wait is safely removable
   through subrange metadata.
4. **256-token FWHT attention tiles.** Test against the current long-context
   attention path under the selected context buckets.
5. **Compatible K/V writer fusion.** Apply only after the tile shape is chosen;
   retain only exact long-context wins.
6. **Multi-token retained replay (deferred).** Do not touch this during items
   1-5. It is expected to matter more for future MTP/speculative draft-verify
   overlap, potentially including independent lm-head work.

Closed for this workload: wider queue counts, CU partitioning/priority, and
explicit shared-LDS GQA reuse.

## FWHT Residual QJL Transform

- Implement a Johnson-Lindenstrauss / QJL transformation on the residual in the FWHT path. The current FWHT path applies a signed-FWHT rotation to Q/K for attention and leaves the residual stream without a separate QJL transform.

## PARO group_size=64 support (SmolLM2-360M)

The SmolLM2-360M PARO model uses group_size=64 (hidden_size=960 not divisible by 128).
Need to generalize the hardcoded group_size=128 assumptions:

1. ✅ **Repacker** (`paro.rs` + `hfq.rs`): parameterized — `bytes_per_group = 8 + gs/2`, loop `gs/2`
   - Verified: SmolLM2 PARO loads all 32 layers through `load_weights_paroquant_llama` → `ParoBackend` → `load_layer`
2. ❌ **DType**: add `ParoQ4G64` variant (or rename `ParoQ4G128` → `ParoQ4` + runtime group_size)
3. ❌ **GEMM guard** (`gemm.rs:129`): relax `k % 128 == 0` → `k % gs == 0`
4. ❌ **GPU kernels** (6 `.hip` files): new kernels with GROUP_SIZE=64 byte layout (40 bytes/group)
   - Existing G128 kernels will silently produce wrong results for K not divisible by 128
5. ❌ **Profile** (`profile.rs`): parameterize hardcoded 128/72 constants
