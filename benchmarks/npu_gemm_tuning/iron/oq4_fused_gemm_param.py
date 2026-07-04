import os, numpy as np, ml_dtypes
import aie.utils as aie_utils
from iron.common import AIEContext
from iron.common.fusion import FusedMLIROperator
from iron.operators.dequant.op import Dequant
from iron.operators.gemm.op import GEMM
from iron.common.test_utils import run_test
M=int(os.environ["M"]); K=int(os.environ["K"]); N=int(os.environ["N"])
gs=int(os.environ.get("GS",256)); t=int(os.environ.get("T",64)); nch=int(os.environ.get("NCH",2))
ctx = AIEContext(); ncols = int(os.environ.get("COLS", aie_utils.get_current_device().cols))
KN=K*N; total_cores=ncols*nch; tile_size=KN//total_cores
print(f"CFG M={M} K={K} N={N} gs={gs} tile={t} nch={nch} ncols={ncols}", flush=True)
dq = Dequant(size=KN, num_aie_columns=ncols, num_channels=nch, tile_size=tile_size, group_size=gs, context=ctx)
gm = GEMM(M=M, K=K, N=N, tile_m=t, tile_k=t, tile_n=t, num_aie_columns=ncols, context=ctx)
runlist=[(dq,"Wp","Wdeq"),(gm,"A","Wdeq","C")]
fused=FusedMLIROperator("oq4_gemm",runlist,input_args=["A","Wp"],output_args=["C"],buffer_sizes={"Wdeq":KN*2})
A=np.random.randn(M,K).astype(ml_dtypes.bfloat16)
Wp=np.zeros(KN//2+(KN//gs)*2,dtype=np.uint8); C=np.zeros((M,N),dtype=ml_dtypes.bfloat16)
errors,lat,bw=run_test(fused,{"A":A.flatten(),"Wp":Wp},{"C":C.flatten()})
print(f"RESULT OQ4-FUSED {M}x{K}x{N} gs={gs} tile={t}: lat={lat:.1f}us errors={bool(errors)}")
