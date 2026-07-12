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
3. **GFX12 temporal cache policy.** Fresh-process A/B for streamed weight loads
   versus reusable KV/scratch loads; verify the intended ISA hint changes and
   do not rely on the unavailable GL2/GCEA counters.
4. **Context-bucketed retained tapes.** Keep replay plans for bounded context
   ranges so flash attention does not launch the physical-capacity grid when
   most tiles would immediately return.
5. **Attention traffic reductions.** Test 256-token FWHT tiles, then compatible
   K/V writer fusion; retain only exact, long-context serve wins.

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
