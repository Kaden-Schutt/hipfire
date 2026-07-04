# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
#
# Triton-XDNA aie2 matmul benchmark (bf16 / int8) — used to probe the Phoenix
# (npu1/aie2) matmul ceiling vs the IRON-reference floor.
#
# Copy into a Triton-XDNA `examples/<dir>/` (it imports the sibling `benchmark`
# helper via `..`) and run with the env from docs/npu/triton-xdna-aie2-int8.md:
#   AIR_TRANSFORM_TILING_SCRIPT=transform_aie2.mlir  (generate via matmul_transform.py)
#   DT=bf16 BM=256 BN=256 python bench_matmul.py 1024 1024 1024   # WORKS
#   DT=i8   BM=256 BN=256 python bench_matmul.py 1024 1024 512     # crashes aircc (mlir-air bug)
#
# Status (2026-06-24, this box): bf16 compiles+runs correct; int8 SIGABRTs in
# mlir-air AIRSplitL2MemrefForBufferConstraintPass for every tiling. See the doc.

import sys, os, time, torch, triton
import triton.language as tl

sys.path.append(os.path.abspath(".."))
import benchmark  # Triton-XDNA examples/benchmark.py (NPU backend selector)


@triton.jit
def mm(
    A,
    B,
    C,
    M,
    N,
    K,
    sam,
    sak,
    sbk,
    sbn,
    scm,
    scn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    on = pn * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ok = tl.arange(0, BLOCK_SIZE_K)
    a = tl.load(A + om[:, None] * sam + ok[None, :] * sak)
    b = tl.load(B + ok[:, None] * sbk + on[None, :] * sbn)
    tl.store(C + om[:, None] * scm + on[None, :] * scn, tl.dot(a, b))


def main():
    benchmark.select_npu_backend()
    M, N, K = (int(x) for x in sys.argv[1:4])
    BM, BN = int(os.environ["BM"]), int(os.environ["BN"])
    dt = os.environ.get("DT", "i8")
    if dt == "i8":
        a = torch.randint(-8, 8, (M, K), dtype=torch.int8)
        b = torch.randint(-8, 8, (K, N), dtype=torch.int8)
        c = torch.zeros((M, N), dtype=torch.int32)
    else:
        a = torch.randn(M, K, dtype=torch.bfloat16)
        b = torch.randn(K, N, dtype=torch.bfloat16)
        c = torch.zeros(M, N, dtype=torch.float32)
    grid = lambda mt: (triton.cdiv(M, mt["BLOCK_SIZE_M"]), triton.cdiv(N, mt["BLOCK_SIZE_N"]))
    args = (a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    mm[grid](*args, BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=K)  # compile + warmup
    ref = (a.to(torch.int32) @ b.to(torch.int32)) if dt == "i8" else (a.float() @ b.float())
    ok = torch.equal(c, ref) if dt == "i8" else torch.allclose(c, ref, atol=1, rtol=1e-2)
    print("correct:", ok)
    it = 20
    t0 = time.perf_counter()
    for _ in range(it):
        mm[grid](*args, BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=K)
    t = (time.perf_counter() - t0) / it
    print(f"RESULT {dt} M={M} N={N} K={K} BM={BM} BN={BN} time={t * 1e3:.3f}ms TOPS={2 * M * N * K / t / 1e12:.2f}")


if __name__ == "__main__":
    main()
