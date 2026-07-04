# Two-pass Oq4-on-NPU: (1) dequant int4+gs256 -> bf16, (2) bf16 gemm. Non-fused
# (materializes bf16 weights, no feed win) but a real, verified end-to-end number.
import numpy as np, ml_dtypes
import aie.utils as aie_utils
from iron.common import AIEContext
from iron.operators.dequant.op import Dequant
from iron.operators.dequant.reference import generate_golden_reference as dq_golden
from iron.operators.gemm.op import GEMM
from iron.operators.gemm.reference import generate_golden_reference as gm_golden
from iron.common.test_utils import run_test

ctx = AIEContext()
ncols = aie_utils.get_current_device().cols
gs = 256

# pass 1: Oq4 dequant (int4 nibbles + per-256 scale -> bf16)
nch = 2; tc = ncols * nch; KN = 16384 * tc; tile_size = KN // tc
dqg = dq_golden(input_length=KN, tile_size=tile_size, group_size=gs)
dq = Dequant(size=KN, num_aie_columns=ncols, num_channels=nch,
             tile_size=tile_size, group_size=gs, context=ctx)
dq_err, dq_lat, _ = run_test(dq, {"input": dqg["input"].flatten()},
                             {"output": dqg["output"].flatten()},
                             rel_tol=0.01, abs_tol=1e-6)

# pass 2: bf16 gemm at the Oq4 weight shape (K*N = dequant span)
M, K, N = 2048, 2048, 2048  # known-good verified 8-col config (b_col_maj, m128/k32/n32)
gg = gm_golden(M, K, N, dtype="bf16", b_col_maj=True, c_col_maj=False)
gm = GEMM(M=M, K=K, N=N, tile_m=128, tile_k=32, tile_n=32,
          num_aie_columns=ncols, b_col_maj=True, c_col_maj=False,
          prio_accuracy=True, emulate_bf16_mmul_with_bfp16=False, context=ctx)
gm_err, gm_lat, _ = run_test(gm, {"A": gg["input"].flatten(),
                                  "B": gg["input_b"][0].flatten()},
                             {"C": gg["output"][0].flatten()},
                             rel_tol=0.005, abs_tol=0.005)
tops = 2 * M * K * N / (gm_lat / 1e6) / 1e12
print(f"OQ4-TWOPASS: dequant(gs256 n={KN}) {dq_lat:.1f}us err={bool(dq_err)} | "
      f"gemm({M}x{K}x{N} bf16) {gm_lat:.1f}us {tops:.2f}TOPS err={bool(gm_err)} | "
      f"total {dq_lat + gm_lat:.1f}us")
