#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
UPROF_HOME=${UPROF_HOME:-/opt/AMDuProf_5.3-518}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR=${OUT_DIR:-"${REPO_ROOT}/benchmarks/results/npu-gpu-dmabuf-profile-${STAMP}"}
RAW_DIR="${OUT_DIR}/raw"
SIZE=${SIZE:-64M}
ITERS=${ITERS:-100}

mkdir -p "${RAW_DIR}"

build_status=0
"${SCRIPT_DIR}/build.sh" > "${RAW_DIR}/build.stdout" 2> "${RAW_DIR}/build.stderr" || build_status=$?
BIN=$(tail -n 1 "${RAW_DIR}/build.stdout" 2>/dev/null)

COMMON=(--mode pingpong --size "${SIZE}" --iters "${ITERS}" --skip-cpu-spot)

"${UPROF_HOME}/bin/AMDuProfCLI" info --list gpu-events \
  > "${RAW_DIR}/uprof-gpu-events.stdout" \
  2> "${RAW_DIR}/uprof-gpu-events.stderr"

memory_probe_status=0
"${UPROF_HOME}/bin/AMDuProfPcm" -m memory -C -A system -d 1 -o "${RAW_DIR}/uprof-memory-probe.csv" \
  > "${RAW_DIR}/uprof-memory-probe.stdout" \
  2> "${RAW_DIR}/uprof-memory-probe.stderr" || memory_probe_status=$?

perf_events=(
  amd_iommu/mem_trans_total/
  amd_iommu/mem_pass_untrans/
  amd_iommu/mem_pass_pretrans/
  amd_iommu/mem_iommu_tlb_pte_hit/
  amd_iommu/mem_iommu_tlb_pte_mis/
  amdgpu:amdgpu_bo_move
  amdgpu:amdgpu_bo_create
  amdgpu:amdgpu_vm_bo_map
  amdgpu:amdgpu_vm_bo_update
  amdxdna:xdna_job
  gpu_scheduler:drm_sched_job_run
  dma_buf:dma_buf_export
  dma_buf:dma_buf_fd
)
perf_event_csv=$(IFS=,; printf '%s' "${perf_events[*]}")

npu_perf_status=0
full_perf_status=0
npu_uprof_status=0
full_uprof_status=0

if [[ ${build_status} -eq 0 && -x "${BIN}" ]]; then
  perf stat -a -e "${perf_event_csv}" -- \
    "${BIN}" "${COMMON[@]}" --skip-hip-compare --json "${OUT_DIR}/npu-only.json" \
    > "${RAW_DIR}/npu-only.perf.stdout" \
    2> "${RAW_DIR}/npu-only.perf.stderr" || npu_perf_status=$?

  perf stat -a -e "${perf_event_csv}" -- \
    "${BIN}" "${COMMON[@]}" --json "${OUT_DIR}/full.json" \
    > "${RAW_DIR}/full.perf.stdout" \
    2> "${RAW_DIR}/full.perf.stderr" || full_perf_status=$?

  "${UPROF_HOME}/bin/AMDuProfPcm" -m dc -C -A system -o "${OUT_DIR}/uprof-dc-npu-only.csv" -- \
    "${BIN}" "${COMMON[@]}" --skip-hip-compare --json "${OUT_DIR}/uprof-npu-only.json" \
    > "${RAW_DIR}/uprof-dc-npu-only.stdout" \
    2> "${RAW_DIR}/uprof-dc-npu-only.stderr" || npu_uprof_status=$?

  "${UPROF_HOME}/bin/AMDuProfPcm" -m dc -C -A system -o "${OUT_DIR}/uprof-dc-full.csv" -- \
    "${BIN}" "${COMMON[@]}" --json "${OUT_DIR}/uprof-full.json" \
    > "${RAW_DIR}/uprof-dc-full.stdout" \
    2> "${RAW_DIR}/uprof-dc-full.stderr" || full_uprof_status=$?
fi

cat > "${OUT_DIR}/README.md" <<EOF
# NPU to GPU dma-buf profile ${STAMP}

- Size: ${SIZE}
- Iterations: ${ITERS}
- Build status: ${build_status}
- uProf GPU-events probe: see \`raw/uprof-gpu-events.stdout\`
- uProf memory metric probe status: ${memory_probe_status}
- perf NPU-only status: ${npu_perf_status}
- perf full status: ${full_perf_status}
- uProf DC NPU-only status: ${npu_uprof_status}
- uProf DC full status: ${full_uprof_status}

Commands:

\`\`\`bash
perf stat -a -e ${perf_event_csv} -- ${BIN} ${COMMON[*]} --skip-hip-compare --json npu-only.json
perf stat -a -e ${perf_event_csv} -- ${BIN} ${COMMON[*]} --json full.json
${UPROF_HOME}/bin/AMDuProfPcm -m dc -C -A system -o uprof-dc-npu-only.csv -- ${BIN} ${COMMON[*]} --skip-hip-compare --json uprof-npu-only.json
${UPROF_HOME}/bin/AMDuProfPcm -m dc -C -A system -o uprof-dc-full.csv -- ${BIN} ${COMMON[*]} --json uprof-full.json
\`\`\`

Interpretation notes:

- \`--skip-hip-compare --skip-cpu-spot\` isolates NPU writes into the imported amdgpu GTT BO.
- Full mode keeps CPU spot-check disabled and adds only the HIP GPU compare reads.
- This uProf install reports \`GPU-events are not supported on this system\`.
- \`AMDuProfPcm -m memory\` is not available for this family/model on the current kernel, so direct UMC/DDR bandwidth is not available through uProf here.
- The useful proxy is the delta in \`amd_iommu/mem_trans_total/\` between NPU-only and full mode.
EOF

if [[ ${build_status} -ne 0 ]]; then
  exit "${build_status}"
fi
if [[ ${npu_perf_status} -ne 0 ]]; then
  exit "${npu_perf_status}"
fi
if [[ ${full_perf_status} -ne 0 ]]; then
  exit "${full_perf_status}"
fi
if [[ ${npu_uprof_status} -ne 0 ]]; then
  exit "${npu_uprof_status}"
fi
if [[ ${full_uprof_status} -ne 0 ]]; then
  exit "${full_uprof_status}"
fi

exit 0
