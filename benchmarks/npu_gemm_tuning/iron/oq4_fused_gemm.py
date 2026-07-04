# Fused Oq4 int4-dequant -> gemm on the NPU (weights stay int4 to L1, unpacked in-core).
import numpy as np, ml_dtypes
import aie.utils as aie_utils
from iron.common import AIEContext
from iron.common.fusion import FusedMLIROperator
from iron.operators.dequant.op import Dequant
from iron.operators.gemm.op import GEMM
from iron.common.test_utils import run_test

ctx = AIEContext()
ncols = aie_utils.get_current_device().cols
M, K, N, gs = 512, 512, 512, 256
KN = K * N
nch = 2; total_cores = ncols * nch; tile_size = KN // total_cores
dq = Dequant(size=KN, num_aie_columns=ncols, num_channels=nch,
             tile_size=tile_size, group_size=gs, context=ctx)
gm = GEMM(M=M, K=K, N=N, tile_m=64, tile_k=64, tile_n=64, num_aie_columns=ncols, context=ctx)
runlist = [(dq, "Wp", "Wdeq"), (gm, "A", "Wdeq", "C")]
fused = FusedMLIROperator("oq4_gemm", runlist, input_args=["A", "Wp"],
                          output_args=["C"], buffer_sizes={"Wdeq": KN})
A = np.random.randn(M, K).astype(ml_dtypes.bfloat16)
Wp = np.zeros(KN // 2 + (KN // gs) * 2, dtype=np.uint8)  # int4 nibbles + scales (dummy for timing)
C = np.zeros((M, N), dtype=ml_dtypes.bfloat16)
errors, lat, bw = run_test(fused, {"A": A.flatten(), "Wp": Wp}, {"C": C.flatten()})
print(f"OQ4-FUSED-GEMM {M}x{K}x{N} gs={gs}: lat={lat:.1f}us errors={bool(errors)}")
