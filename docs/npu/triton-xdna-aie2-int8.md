# Triton-XDNA on aie2 (Phoenix/npu1): bf16 works, int8 blocked by an mlir-air bug

**Box:** Ryzen 7 7840HS, gfx1103 + XDNA1/Phoenix NPU (aie2, npu1). **Date:** 2026-06-24.

## Goal & outcome

Re-test the real Phoenix int8-matmul ceiling against the IRON-reference floor
(~2.5 TOPS, see `project-oq-npu-spike-nogo`) and the ~16 TOPS measured on the
sister 7840HS. IRON is aie2p-centric and under-tunes Phoenix; **Triton-XDNA**
(`~/build/Triton-XDNA`, lowers Triton → MLIR-AIR → MLIR-AIE, has a dedicated
`transform_aie2.mlir`, claims handwritten-parity) is the better aie2 path.

**Result:**
- **bf16 matmul COMPILES + RUNS CORRECT on the aie2 NPU via Triton-XDNA** (the
  hardware + flow work; my "NPU NO-GO" was a tooling floor, not silicon).
- **int8 matmul is BLOCKED by an `mlir-air` compiler bug** —
  `aircc` SIGABRTs in `AIRSplitL2MemrefForBufferConstraintPass` for *every* int8
  tiling/size. Two distinct assertions; **no available mlir-air version both
  drives this triton-xdna AND compiles int8** (see version analysis below). So the
  int8 ceiling could not be measured through this stack — a compiler result, not perf.

## Why nix1: things to try there

nix1 originally built the NPU kernels (`NPU-RESULTS.md`) and may have a different
toolchain (e.g. the proprietary **chess/xchesscc** compiler, a patched/newer
`mlir-air`, or a different `aircc`). The int8 crash is in an **air-level** pass
(before peano/chess), so chess alone may not help — but a newer/patched mlir-air
with the `AIRSplitL2Memref` `eraseOp` bug fixed would. Repro steps below.

## Build (this box; reproduce on nix1)

Triton-XDNA Option-2 (pip-from-source, prebuilt dep wheels). Gotchas hit here:
1. `git submodule update --init --recursive third_party/triton third_party/triton_shared`
   (remove any stray `.patches_applied` markers first — they block the clone).
2. System deps: `sudo apt install python3.12-dev uuid-dev`.
3. **LLVM pre-stage** (Triton downloads a 1.8 GB LLVM at cmake time; throttled host →
   use aria2c -x16, then `export LLVM_SYSPATH=~/.triton/llvm/llvm-<hash>-ubuntu-x64`
   to skip the in-build download). URL pattern:
   `https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-<8hash>-ubuntu-x64.tar.gz`
   (hash from `third_party/triton/cmake/llvm-hash.txt`).
4. **NVIDIA toolchain patch** — `third_party/triton/python/build_helpers.py`:
   early-`return` from `download_and_copy_dependencies()` (skips the CUDA redist
   download we don't need). **Keep the nvidia *backend* enabled** in
   `setup.py:373` (`["nvidia","amd"]`) — Triton's CMake hard-references
   `TritonNVIDIAGPUConversionPassIncGen`, so dropping it breaks configure.
5. `pip install . --no-build-isolation --find-links <mlir-aie> <llvm-aie> <mlir-air>`
   (CPU torch into the sandbox: `pip install torch --index-url …/whl/cpu`).

Verify: `python -c "import triton; from triton.backends import backends; print(list(backends))"`
→ must include `amd_triton_npu`.

## Run / repro

Env every run:
```
source /opt/xilinx/xrt/setup.sh
export LD_LIBRARY_PATH=~/.cache/hipfire-npu-deps/lib:/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH  # boost-1.83 for xclbinutil
export PYTHONPATH=/opt/xilinx/xrt/python:$PYTHONPATH
export AIR_TRANSFORM_TILING_SCRIPT=transform_aie2.mlir
```
Generate an aie2 transform + bench (vendored script `tools/npu/triton_xdna/bench_matmul.py`,
copy into a Triton-XDNA `examples/<dir>/`):
```
python examples/matmul_transform.py --l1-m 64 --l1-n 64 --l2-k 64 \
    --pack-sizes 4 8 8 --accum-type i32 --contract-input-type i16 -o transform_aie2.mlir   # aie2 int8 mac=(4,8,8)
DT=bf16 BM=256 BN=256 python bench_matmul.py 1024 1024 1024    # WORKS → prints RESULT … TOPS=…
DT=i8   BM=256 BN=256 python bench_matmul.py 1024 1024 512     # CRASHES aircc (see below)
```

## The int8 blocker (exact)

`aircc … died with SIGABRT` inside `AIRSplitL2MemrefForBufferConstraintPass::runOnOperation`:
- pinned mlir-air `dfa6d08` (May 8): `isDefaultDataAccessPattern` →
  `std::optional::_M_get(): Assertion 'this->_M_is_engaged()'`.
- mlir-air `0e738e9` (Jun 5, has fix `0cce221e`): first bug gone, now
  `PatternMatch.cpp eraseOp: Assertion 'op->use_empty()'` — **unfixed in every version**.

Triggered for all int8 tilings (verified l1 32–128, BM/BN 128–512, C_L2 64–512 KB).
bf16 only works via the *shipped* tiny `transform_aie2.mlir`; no working int8-aie2 transform exists.

### Version analysis (no version works)
| | date |
|---|---|
| crashing pass logic (`tileChannelOpByFactor`/`isDefaultDataAccessPattern`) | **Mar 2024** |
| `--stack-size` CLI this triton-xdna requires (min compatible aircc) | Apr 21 2026 |
| first-bug fix `0cce221e` | May 20 2026 |
| second-bug (`eraseOp`) fix | **never** |

CLI-compatible window = Apr 21→present; Apr21–May20 has bug #1, May20–present has bug #2.
The pass logic is 2 years old → no version is old enough to predate the bug yet new
enough to drive this triton-xdna. Checked newer **and** older wheels exhaustively.

## Recommendations
1. **Hand-write the aie2 int8 kernel in raw IRON/mlir-aie** (bypass mlir-air's split
   pass) — the only NPU route not gated on this bug; IRON int8 already works (2.5 TOPS),
   gap to ~16 is occupancy tuning (`tools/npu/oq_gemm_design.py` is the IRON starting point).
2. **Patch `AIRSplitL2Memref` `eraseOp` + rebuild mlir-air from source** — unblocks
   Triton-XDNA int8 (LLVM-scale build + a real C++ fix).
3. **File upstream** (clean repro: any int8 matmul, `--device npu1`).
4. Ship OQ/SpinQuant-W4A8 on the **GPU** now; NPU as a parallel hand-kernel track.
