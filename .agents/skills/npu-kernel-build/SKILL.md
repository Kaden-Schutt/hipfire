---
name: npu-kernel-build
description: Compile MLIR-AIE kernels for AMD XDNA NPUs via IRON + aiecc. Use when adding or iterating on an NPU compute kernel that feeds libhipfire_xdna1.so.
---

# NPU kernel build — MLIR-AIE / IRON

## AMD NPU naming map

AMD's NPU naming is a tangle of marketing names, microarchitecture names, and toolchain
identifiers that don't align. Reference table:

| Marketing name | Silicon examples | NPU gen | Tile ISA | Toolchain target | TOPS |
|----------------|-----------------|---------|----------|------------------|------|
| XDNA (Ryzen AI) | Phoenix (7040), Hawk Point (8040) | NPU1 | AIE2 / AIE-ML | `aie2`, `aie2txn`, `aie2dpu` | 16 |
| XDNA2 (Ryzen AI 300) | Strix Point, Strix Halo | NPU2 | AIE2P / AIE-ML v2 | `aie2p` | 50 |
| XDNA3 (future) | Kraken | NPU3 | AIE4 | `aie4` | TBD |

Name equivalences:
- **AIE** = AI Engine (Xilinx/Versal origin)
- **AIE2** = second-gen AI Engine = AMD's **AIE-ML** (NPU1 / XDNA)
- **AIE2P** = AIE2+ = AMD's **AIE-ML v2** (NPU2 / XDNA2); adds hardware tanh, wider accumulators
- **XDNA / XDNA2** = AMD's brand for the NPU IP block, not the tile ISA
- **NPU1 / NPU2** = MLIR-AIE device identifiers (`AIEDevice.npu1`, `AIEDevice.npu2`)

**Critical ISA difference**: AIE2 needs LUT-based tanh (`getTanhBf16` from `<lut_based_ops.h>`).
AIE2P has hardware tanh (`aie::tanh<bfloat16>(accum.to_vector<float>())`). Using the wrong one
fails to compile or produces wrong output. See the tanh section below for exact call patterns.
Check `AGENTS.local.md` for which NPU generation is on this machine.

## AIE tile microarchitecture

Each AIE compute tile contains **two separate processors** that run in parallel:

**1. Scalar VLIW processor** (the "chess core")
- 32-bit RISC-like, handles control flow, addressing, scalar arithmetic
- Runs loop counters, pointer increments, branch logic
- `AIE_PREPARE_FOR_PIPELINING` tells the Peano scheduler to overlap scalar and vector
  ops across iterations (software pipelining) — scalar computes addresses for N+1
  while the vector unit executes N

**2. Vector SIMD processor** (the "vector unit")
- Processes 16 BF16 values per cycle (AIE2: 256-bit; AIE2P: 512-bit wide but same API)
- Target of `aie::mul`, `aie::add`, `aie::tanh`, etc.
- `aie::begin_restrict_vector<16>` feeds the vector unit load/store path

**Plus a DMA engine** running concurrently with both processors
- Moves data between tile-local SRAM and the NoC / neighboring tiles
- ObjectFIFOs in IRON map to DMA channels — tile N+1 is fetched while N executes

**Key gotcha — `auto` vs explicit vector type**: `aie::mul(vec, vec)` returns
`accum<__accfloat, 16>`, not `vector<bfloat16, 16>`. Using `auto` for intermediates
keeps them as accum; the scalar core can't bypass an accumulator back to the vector
input path, stalling the pipeline and breaking chained ops (no `mul(vec, accum)` overload).
Always declare intermediates as `aie::vector<bfloat16, 16>` explicitly.

**Tile local memory**: 32 KB SRAM per compute tile, shared between both processors and the DMA.
At BF16 (2 bytes): 32 KB = 16 K elements. The streaming tile pattern (objectfifo) exists
precisely because full tensors don't fit — only the active tile slice lives in SRAM at once.

## Toolchain locations

| Tool | Path |
|------|------|
| Python venv | `~/.venv` |
| aiecc | `~/.venv/bin/aiecc` |
| Peano (clang++) | `~/.venv/lib/python3.12/site-packages/llvm-aie/bin/clang++` |
| AIE headers | `~/.venv/lib/python3.12/site-packages/mlir_aie/include/` |
| Pre-built AIE2 kernels | `~/.venv/lib/python3.12/site-packages/mlir_aie/include/aie_kernels/aie2/` |
| Pre-built AIE2P kernels | `~/.venv/lib/python3.12/site-packages/mlir_aie/include/aie_kernels/aie2p/` |
| AIE2 runtime lib (LUT tanh) | `~/.venv/lib/python3.12/site-packages/mlir_aie/aie_runtime_lib/AIE2/` |
| AIE2P runtime lib | `~/.venv/lib/python3.12/site-packages/mlir_aie/aie_runtime_lib/AIE2P/` |
| xclbinutil | `/opt/xilinx/xrt/bin/xclbinutil` — **not on PATH by default; prepend manually** |

**Critical**: `xclbinutil` is at `/opt/xilinx/xrt/bin/` but not on PATH. aiecc silently skips
xclbin assembly if it can't find it — no error, just no `.xclbin` output. Always prepend:
```python
os.environ["PATH"] = "/opt/xilinx/xrt/bin" + os.pathsep + os.environ.get("PATH", "")
```

## IRON API quick reference

```python
from ml_dtypes import bfloat16      # NOT np.bfloat16 — numpy doesn't have it
import numpy as np
from aie.iron import ExternalFunction
from aie.iron.algorithms.transform import transform_parallel_binary, transform_binary
from aie.iron.device import NPU1, NPU2
from aie.utils import set_current_device
from aie.utils.compile import compile_mlir_module, compile_external_kernel

# Prefer detect_npu() over hardcoding; see "Autodetection" section below.
set_current_device(NPU1())   # must be called before any transform_* call

tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]

kernel = ExternalFunction(
    name="my_kernel",
    source_file="/path/to/kernel.cc",
    arg_types=[tile_ty, tile_ty, tile_ty, np.int32],  # (in0, in1, out, n)
    include_dirs=[str(AIE_INCLUDE), str(AIE_RUNTIME_LIB)],  # AIE2: runtime lib has lut_based_ops.h
)

# Single-core (for debugging / small sizes)
mlir_module = transform_binary(kernel, in0_buf, in1_buf, out_buf, tile_size=tile_size)

# All columns in parallel (preferred)
mlir_module = transform_parallel_binary(kernel, in0_buf, in1_buf, out_buf, tile_size=tile_size)
```

`tile_size` must be a multiple of 16. `hidden_size` must be divisible by `tile_size * num_cols`.
Get `num_cols` from `get_current_device().cols` after `set_current_device`.

## Autodetection

Detect the installed NPU generation at build time via pyxrt rather than hardcoding:

```python
_NAME_TO_NPU = {
    "npu1": "npu1", "Phoenix": "npu1",
    "npu4": "npu2", "npu5": "npu2", "npu6": "npu2", "Strix": "npu2", "Krackan": "npu2",
}

def detect_npu() -> str:
    import ctypes, sys
    ctypes.CDLL("/opt/xilinx/xrt/lib/libxrt_coreutil.so.2", mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0, "/opt/xilinx/xrt/python")
    import pyxrt
    name = pyxrt.device(0).get_info(pyxrt.xrt_info_device.name)
    for substr, key in _NAME_TO_NPU.items():
        if substr in name:
            return key
    raise RuntimeError(f"Unknown device: {name!r}")
```

`pyxrt.xrt_info_device.name` returns strings like `"RyzenAI-npu1"` or `"RyzenAI-npu4"`.
See `tools/npu/build_qwen35_swiglu.py` for the full implementation. The Cargo
`build.rs` passes `--npu auto` by default; override with `HIPFIRE_NPU_TARGETS=npu1,npu2`.

## Compilation

```python
import tempfile, shutil
from pathlib import Path

# target_arch: "aie2" for NPU1, "aie2p" for NPU2
with tempfile.TemporaryDirectory(prefix="hipfire_npu_build_") as tmp:
    tmp = Path(tmp)
    compile_external_kernel(kernel, tmp, target_arch="aie2")
    compile_mlir_module(
        mlir_module=mlir_module,
        insts_path=tmp / "insts.bin",
        xclbin_path=tmp / "final.xclbin",
        work_dir=tmp,
    )
    shutil.copy2(tmp / "final.xclbin", out_dir / xclbin_name)
    shutil.copy2(tmp / "insts.bin",    out_dir / instr_name)
```

Produces:
- `final.xclbin` — path1 for `xdna1_bf16_*_create`; DPU kernel id `0x901`, kernel name `MLIR_AIE`
- `insts.bin` — path2 for `xdna1_bf16_*_create`; raw NPU instruction binary

## Kernel C++ conventions (AIE2 + AIE2P portable)

The toolchain is identical across NPU generations — same Peano compiler, same
mlir_aie package. Only the `target_arch` flag (`"aie2"` / `"aie2p"`) and the
tanh implementation differ. Gate on `__AIE_ARCH__` (20=AIE2, 21=AIE2P).

```cpp
#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
// AIE2: include BOTH lut_based_ops files — .h has getTanhBf16 (inline) + extern
// table decls; .cpp has the actual table data.  The .cpp does NOT include .h.
// AIE2P: no LUT needed; aie::tanh is hardware but takes float input (see below).
#if __AIE_ARCH__ == 20
#  include "lut_based_ops.h"
#  include "lut_based_ops.cpp"
#endif
#include <stdint.h>
using namespace aie;

extern "C" {
void my_kernel(bfloat16 *restrict in0, bfloat16 *restrict in1,
               bfloat16 *restrict out, const int32_t n) {
  auto it0  = aie::begin_restrict_vector<16>(in0);
  auto it1  = aie::begin_restrict_vector<16>(in1);
  auto it_o = aie::begin_restrict_vector<16>(out);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < n; i += 16) {
    aie::vector<bfloat16, 16> a = *it0++;
    aie::vector<bfloat16, 16> b = *it1++;
    // Use explicit vector<bfloat16,16>, not auto, for all intermediates
    aie::vector<bfloat16, 16> result = aie::mul(a, b);
    *it_o++ = result;
  }
}
}
```

### `aie::mul` return type and the `auto` trap

`aie::mul(vec, vec)` returns `accum<__accfloat, N>`, **not** `vector<bfloat16, N>`.

- **AIE2**: The chess scalar core cannot bypass an accumulator back into the
  vector input path. Using `auto` for any intermediate that feeds another `mul`
  or `add` stalls the pipeline because there's no vec×accum overload. Always
  use explicit `aie::vector<bfloat16, 16>` for those intermediates — the
  implicit `accum → vector<bfloat16>` conversion fires on assignment.

- **AIE2P**: `auto` is generally fine for non-tanh paths, but any variable that
  will be passed as an argument to another `aie::mul` must be explicitly typed
  as `aie::vector<bfloat16, 16>`. There is **no** `mul(vector<bfloat16>, accum)`
  overload on either arch.

### tanh: arch-specific call patterns

**AIE2 (NPU1)** — LUT-based:
```cpp
// x must be vector<bfloat16,16>
aie::vector<bfloat16, 16> t = getTanhBf16(x);
```

**AIE2P (NPU2)** — hardware, but input must be `vector<float>`:
```cpp
// half_x is the accum from aie::mul(x, 0.5f)
auto half_x_acc = aie::mul(x, half);
auto t = aie::tanh<bfloat16>(half_x_acc.to_vector<float>());
```
- `aie::tanh` only has a `vector<float, N>` overload — passing `vector<bfloat16>`
  directly gives "no matching function".
- `.to_vector<float>()` is a method on `accum`, **not** on `vector<bfloat16>`.
  If `half_x` has already been stored as `vector<bfloat16>`, there is no
  `.to_vector` on it — keep the accum alive with `auto` until the tanh call.
- The `<bfloat16>` template parameter on `aie::tanh` selects the **return** type.

## Tile grid and column_width

xrt-smi "6×5=30" = 6 columns × 5 rows = 30 total tiles (all types). Breakdown:
- 6 physical columns: 1 shim-only column + 5 data columns
- 5 rows: 1 shim row + 4 core rows
- `NPU1().cols = 4` = compute columns that accept kernel PDIs
- Firmware `total_col = 5` = max column-width a hw_context may request

Driver validation: `num_col = xclbin.column_width * core_rows / core_rows = xclbin.column_width`.
If `column_width > 5` → `DRM_AMDXDNA_CREATE_HWCTX` returns EINVAL.

`transform_parallel_binary` with `set_current_device(NPU1())` correctly generates `column_width=4`.
**A stale xclbin built when the device was set to NPU2 will have `column_width=8` and silently fail
at runtime.** Always rebuild xclbins after switching `--npu` target or changing the build script.

## Host runtime

`aie.xrt.XCLBin` is broken under XRT 2.25 (hardcoded assertion `"RyzenAI-Phoenix"`,
but XRT 2.25 returns `"RyzenAI-npu1"`). Use `XRTHostRuntime` instead:

```python
import ctypes, os, sys
# Must be loaded BEFORE pyxrt to supply a weak vtable symbol
ctypes.CDLL("/opt/xilinx/xrt/lib/libxrt_coreutil.so.2", mode=ctypes.RTLD_GLOBAL)
sys.path.insert(0, "/opt/xilinx/xrt/python")

from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel
import numpy as np
from ml_dtypes import bfloat16

# Allocate host-accessible device buffers (group_id=0, host_only flags)
t_in0 = XRTTensor(input_array, dtype=bfloat16, device="cpu")
t_out = XRTTensor((hidden_size,), dtype=bfloat16, device="cpu")

npu_kernel = NPUKernel(xclbin_path=xclbin, insts_path=instr, kernel_name="MLIR_AIE")
rt = XRTHostRuntime()
handle = rt.load(npu_kernel)
result = rt.run(handle, [t_in0, t_out])

t_out.to("cpu")
output = t_out.numpy()
```

The internal kernel call is `kernel(3, insts_bo, insts_nbytes, *data_bos)`.
Instruction BO uses `flags=pyxrt.bo.cacheable, group_id=kernel.group_id(1)`;
data BOs use `flags=pyxrt.bo.host_only, group_id=0`.

## libhipfire_xdna1.so FFI contract

Artifacts feed into `xdna1_bf16_swiglu_create(xclbin_path, instr_path, hidden_size, result_out)`.
See `crates/hipfire-arch-qwen35/src/xdna1_ffi.rs` for the full Rust FFI wrapper.

Env vars to activate:
```bash
export HIPFIRE_QWEN35_FFN_BF16=xdna1
export HIPFIRE_QWEN35_XDNA1_XCLBIN=/path/to/qwen35-swiglu-8960.xclbin
export HIPFIRE_QWEN35_XDNA1_INSTR=/path/to/qwen35-swiglu-8960-instr.bin
export HIPFIRE_XDNA1_LIB=target/npu/libhipfire_xdna1.so   # default
export HIPFIRE_QWEN35_FFN_BF16_TRACE=1                     # per-layer timing
```

## Build script template

See `tools/npu/build_qwen35_swiglu.py` and `tools/npu/silu_mul_bf16.cc`.

## Pre-built kernel sources

`~/.venv/lib/python3.12/site-packages/mlir_aie/include/aie_kernels/`

Both `aie2/` and `aie2p/` subdirectories exist. Available kernels:

| File | Computes |
|------|----------|
| `swiglu.cc` | `silu(x·w2) · (x·w1)` — fused weight-mul (use our `silu_mul_bf16.cc` when projections are pre-done) |
| `silu.cc` | `silu(x)` elementwise |
| `rms_norm.cc` | RMSNorm (reduction + scale) |
| `rope.cc` | RoPE rotary embeddings |
| `mm.cc` | Matrix multiply |
| `gelu.cc` | GeLU activation |
| `softmax.cc` | Softmax |
