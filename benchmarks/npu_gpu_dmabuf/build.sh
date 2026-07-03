#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

ROCM_HOME=${ROCM_HOME:-/opt/rocm}
XRT_HOME=${XRT_HOME:-/opt/xilinx/xrt}
BUILD_DIR=${BUILD_DIR:-"${REPO_ROOT}/target/npu_gpu_dmabuf"}
SRC="${SCRIPT_DIR}/npu_gpu_dmabuf_bench.hip"
OUT="${BUILD_DIR}/npu_gpu_dmabuf_bench"
SYNC_SRC="${SCRIPT_DIR}/syncobj_probe.cpp"
SYNC_OUT="${BUILD_DIR}/syncobj_probe"

mkdir -p "${BUILD_DIR}"

export PKG_CONFIG_PATH="${XRT_HOME}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

DRM_FLAGS=$(pkg-config --cflags --libs libdrm_amdgpu libdrm)
VXDNA_FLAGS=$(pkg-config --cflags --libs vxdna)

"${ROCM_HOME}/bin/hipcc" \
  -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter \
  -I"${ROCM_HOME}/include" \
  "${SRC}" \
  ${DRM_FLAGS} \
  ${VXDNA_FLAGS} \
  -I"${XRT_HOME}/include" \
  -L"${XRT_HOME}/lib" \
  -Wl,-rpath,"${XRT_HOME}/lib" \
  -Wl,--no-as-needed \
  -lxrt_coreutil -lxrt_core -lxrt++ -lvxdna \
  -Wl,--as-needed \
  -ldl -luuid -pthread \
  -o "${OUT}"

"${ROCM_HOME}/bin/hipcc" \
  -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter \
  -I"${ROCM_HOME}/include" \
  -I"${XRT_HOME}/include" \
  -I"/home/sadara/xdna-driver/src" \
  -I"/home/sadara/xdna-driver/src/include/uapi" \
  -I"/home/sadara/xdna-driver/xrt/src/runtime_src" \
  -I"/home/sadara/xdna-driver/xrt/src/runtime_src/core/include" \
  "${SYNC_SRC}" \
  ${DRM_FLAGS} \
  ${VXDNA_FLAGS} \
  -L"${XRT_HOME}/lib" \
  -Wl,-rpath,"${XRT_HOME}/lib" \
  -Wl,--no-as-needed \
  -lxrt_coreutil -lxrt_core -lxrt++ -lvxdna -l:libxrt_driver_xdna.so.2 \
  -Wl,--as-needed \
  -ldl -luuid -pthread \
  -o "${SYNC_OUT}"

printf '%s\n' "${OUT}"
printf '%s\n' "${SYNC_OUT}"
