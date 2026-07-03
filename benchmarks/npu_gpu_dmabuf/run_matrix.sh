#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
XRT_PAYLOAD_DIR=${XRT_PAYLOAD_DIR:-/home/sadara/xdna-driver/build/vtd_extract/strx}
XCLBIN=${XCLBIN:-"${XRT_PAYLOAD_DIR}/validate_df_bandwidth.xclbin"}
ELF=${ELF:-"${XRT_PAYLOAD_DIR}/df_bw.elf"}
ROCM_HOME=${ROCM_HOME:-/opt/rocm}
HIPCC=${HIPCC:-"${ROCM_HOME}/bin/hipcc"}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR=${OUT_DIR:-"${REPO_ROOT}/benchmarks/results/npu-gpu-dmabuf-${STAMP}"}
RAW_DIR="${OUT_DIR}/raw"

mkdir -p "${RAW_DIR}"

build_status=0
"${SCRIPT_DIR}/build.sh" > "${RAW_DIR}/build.stdout" 2> "${RAW_DIR}/build.stderr" || build_status=$?
BIN=$(tail -n 1 "${RAW_DIR}/build.stdout" 2>/dev/null)

uname -a > "${RAW_DIR}/uname.stdout" 2> "${RAW_DIR}/uname.stderr"
"${HIPCC}" --version > "${RAW_DIR}/hipcc-version.stdout" 2> "${RAW_DIR}/hipcc-version.stderr"
if command -v amdgpu-arch >/dev/null 2>&1; then
  amdgpu-arch > "${RAW_DIR}/amdgpu-arch.stdout" 2> "${RAW_DIR}/amdgpu-arch.stderr"
else
  /opt/rocm/lib/llvm/bin/amdgpu-arch > "${RAW_DIR}/amdgpu-arch.stdout" 2> "${RAW_DIR}/amdgpu-arch.stderr"
fi

examine_status=0
xrt-smi examine --batch > "${RAW_DIR}/xrt-smi-examine.stdout" 2> "${RAW_DIR}/xrt-smi-examine.stderr" || examine_status=$?

validate_status=0
xrt-smi validate -r df-bw -p "${XRT_PAYLOAD_DIR}" > "${RAW_DIR}/xrt-smi-validate-df-bw.stdout" 2> "${RAW_DIR}/xrt-smi-validate-df-bw.stderr" || validate_status=$?

single_status=0
pingpong_status=0

if [[ ${build_status} -eq 0 && -x "${BIN}" ]]; then
  for size in 4K 64K 1M 16M 64M; do
    "${BIN}" \
      --mode single \
      --size "${size}" \
      --iters 1 \
      --xclbin "${XCLBIN}" \
      --elf "${ELF}" \
      --json "${OUT_DIR}/single-${size}.json" \
      > "${RAW_DIR}/single-${size}.stdout" \
      2> "${RAW_DIR}/single-${size}.stderr" || single_status=$?
  done

  for size in 1M 16M 64M; do
    "${BIN}" \
      --mode pingpong \
      --size "${size}" \
      --iters 100 \
      --xclbin "${XCLBIN}" \
      --elf "${ELF}" \
      --json "${OUT_DIR}/pingpong-${size}.json" \
      > "${RAW_DIR}/pingpong-${size}.stdout" \
      2> "${RAW_DIR}/pingpong-${size}.stderr" || pingpong_status=$?
  done
fi

kernel_release=$(uname -r)
hipcc_line=$(sed -n '1p' "${RAW_DIR}/hipcc-version.stdout")
gpu_arch=$(tr '\n' ' ' < "${RAW_DIR}/amdgpu-arch.stdout" | sed 's/[[:space:]]*$//')
xrt_version=$(awk -F: '/  Version[[:space:]]*:/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' "${RAW_DIR}/xrt-smi-examine.stdout")
amdxdna_version=$(awk -F: '/amdxdna Version/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' "${RAW_DIR}/xrt-smi-examine.stdout")

cat > "${OUT_DIR}/README.md" <<EOF
# NPU to GPU dma-buf results ${STAMP}

- Host date UTC: ${STAMP}
- Kernel release: ${kernel_release}
- HIP compiler: ${hipcc_line}
- GPU arch: ${gpu_arch}
- XRT version: ${xrt_version}
- amdxdna version: ${amdxdna_version}
- XRT payload dir: ${XRT_PAYLOAD_DIR}
- XCLBIN: ${XCLBIN}
- ELF: ${ELF}
- Build status: ${build_status}
- xrt-smi examine status: ${examine_status}
- xrt-smi validate df-bw status: ${validate_status}
- single matrix status: ${single_status}
- pingpong matrix status: ${pingpong_status}

Raw logs are in \`raw/\`. JSON benchmark reports are next to this README.

Commands:

\`\`\`bash
benchmarks/npu_gpu_dmabuf/build.sh
xrt-smi examine --batch
xrt-smi validate -r df-bw -p ${XRT_PAYLOAD_DIR}
target/npu_gpu_dmabuf/npu_gpu_dmabuf_bench --mode single --size <4K|64K|1M|16M|64M> --iters 1 --xclbin ${XCLBIN} --elf ${ELF} --json <out>.json
target/npu_gpu_dmabuf/npu_gpu_dmabuf_bench --mode pingpong --size <1M|16M|64M> --iters 100 --xclbin ${XCLBIN} --elf ${ELF} --json <out>.json
\`\`\`

The requested df-bw preflight command is recorded even if this installed
\`xrt-smi\` does not support \`-r df-bw -p <dir>\`.
EOF

if [[ ${build_status} -ne 0 ]]; then
  exit "${build_status}"
fi

if [[ ${single_status} -ne 0 ]]; then
  exit "${single_status}"
fi

if [[ ${pingpong_status} -ne 0 ]]; then
  exit "${pingpong_status}"
fi

exit 0
