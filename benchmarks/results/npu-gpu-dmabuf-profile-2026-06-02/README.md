# NPU to GPU dma-buf profile 2026-06-02

- Command shape: `--mode pingpong --size 64M --iters 100 --skip-cpu-spot`
- NPU-only variant: adds `--skip-hip-compare`
- Full variant: includes HIP import plus GPU compare reads
- uProf version: `5.3.518.0`
- uProf GPU-events probe: `GPU-events are not supported on this system.`
- uProf direct memory metric probe: `AMDuProfPcm -m memory` is unsupported on this family/model with this kernel.

## Perf Counter Summary

| Counter | NPU-only | Full NPU+HIP | Delta |
|---|---:|---:|---:|
| `amd_iommu/mem_trans_total/` | 35,830 | 1,740,675 | 1,704,845 |
| `amd_iommu/mem_iommu_tlb_pte_hit/` | 28,346 | 15,071 | -13,275 |
| `amd_iommu/mem_iommu_tlb_pte_mis/` | 7,449 | 2,936 | -4,513 |
| `amdgpu:amdgpu_bo_move` | 52 | 122 | 70 |
| `amdgpu:amdgpu_bo_create` | 32 | 67 | 35 |
| `amdxdna:xdna_job` | 408 | 408 | 0 |
| `gpu_scheduler:drm_sched_job_run` | 114 | 114 | 0 |
| `dma_buf:dma_buf_export` | 2 | 2 | 0 |

The full run adds about `1,704,845` IOMMU translations over the NPU-only
run. At 4 KiB pages that is about `6.98 GB` of additional translated
device access. The full command performs 102 HIP compares of 64 MiB
buffers including warmups/final check, about `6.84 GB` of GPU reads.

That match is the current best evidence that the full path is:

```text
DDR source -> NPU -> amdgpu GTT/DDR output -> GPU reads GTT/DDR
```

not a direct NPU write into GPU-resident memory. The benchmark still
proves zero CPU copy between NPU output and GPU input.

## uProf CPU DC Summary

uProf CPU data-cache metrics are nearly identical between NPU-only and
full mode and CPU spot-check was disabled, which is consistent with no
large CPU read/copy participating in the measured output path.

| Metric | NPU-only | Full |
|---|---:|---:|
| `DC Fills From Local Memory or I/O (pti)` | 0.24 | 0.24 |
| `Demand DC Fills From Local Memory or I/O (pti)` | 0.05 | 0.05 |
| `Remote DRAM Reads %` | 0.00 | 0.00 |

Raw logs and CSVs are in `raw/` and the adjacent `uprof-dc-*.csv` files.
