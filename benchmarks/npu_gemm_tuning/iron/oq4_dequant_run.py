# Run IRON's int4 dequant at group_size=256 (Oq4G256's group) on the NPU.
import aie.utils as aie_utils
from iron.common import AIEContext
from iron.operators.dequant.op import Dequant
from iron.operators.dequant.reference import generate_golden_reference
from iron.common.test_utils import run_test

ctx = AIEContext()
ncols = aie_utils.get_current_device().cols
nch = 2
total_cores = ncols * nch
gs = 256                       # Oq4 group size
tile_size = 16384
size = tile_size * total_cores  # one-shot dequant span
golden = generate_golden_reference(input_length=size, tile_size=tile_size, group_size=gs)
op = Dequant(size=size, num_aie_columns=ncols, num_channels=nch,
             tile_size=tile_size, group_size=gs, context=ctx)
errors, lat, bw = run_test(op, {"input": golden["input"].flatten()},
                           {"output": golden["output"].flatten()},
                           rel_tol=0.01, abs_tol=1e-6)
print(f"OQ4-DEQUANT gs=256 size={size} cols={ncols}x{nch}: "
      f"lat={lat:.1f}us bw={bw:.4f}GB/s errors={bool(errors)}")
