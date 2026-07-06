# TODO

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
