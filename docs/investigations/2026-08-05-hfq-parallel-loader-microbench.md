# HFQ bounded parallel-loader microbenchmark

Date: 2026-08-05  
Branch: `ds4-beta-staging`  
Benchmark introduction commit: `d92ef9144a384fc8b783993abb5ca526ad0785cb`

## Purpose

Measure the storage and packing portion of HFQ model loading before changing the
production loader. The benchmark keeps GPU allocation/upload consumption in
canonical HFQ order while allowing a bounded number of independent readers to
fill final packed host buffers. It compares:

- `scratch-pack`, one lane: current DS4 loader-shaped expert packing, where each
  source tensor is read into reusable scratch and copied into the final packed
  allocation;
- `direct-pack`, one lane: source tensors are read directly into their final
  packed offsets;
- `direct-pack`, two or four lanes: multiple final packed jobs are filled in
  parallel, then consumed in canonical order.

The tool is
[`hfq_load_pipeline_bench.rs`](../../crates/hipfire-runtime/examples/hfq_load_pipeline_bench.rs).
It is a standalone screening example. No production load path, inference path,
model file, allocation order, or PM4 route is changed.

## Fixture

- Host: `hipx`
- Storage: local Kingston OM8TAP42048K1-A00 NVMe, ext4 on LVM
- Model:
  `/home/kaden/ds4-gfx1151-evidence/2026-08-03-ds4-dspark-pm4-canary/model-e8/deepseek-v4-flash-0731.mq2r`
- HFQ architecture ID: 9
- Selection: DS4 routed-expert layers `[0, 4)`
- Plan: 8 canonical packed jobs, 3,072 source segments, 7.248 GB
- Plan hash: `668d35b77d91a2d2`
- Cache policy: `POSIX_FADV_DONTNEED` before each case and after every source
  read
- Full-data checksum: `7a3363a16250f12c`
- Every candidate matched the baseline checksum

DS4 packing is represented exactly: one concatenated `w2` job per layer and one
concatenated `w1`+`w3` gate-up job per layer. The pipeline submits only one new
job when the next canonical job is consumed, bounding live host output buffers
to roughly the reader-lane count.

## Read-only storage screen

Two repetitions per case:

| Path | Run 1 | Run 2 | Median wall | Median GB/s | Change from baseline |
|---|---:|---:|---:|---:|---:|
| scratch-pack, 1 lane | 9.811 s | 10.378 s | 10.094 s | 0.718 | baseline |
| direct-pack, 1 lane | 8.849 s | 9.094 s | 8.972 s | 0.808 | 1.12x, -11.1% wall |
| direct-pack, 2 lanes | 6.020 s | 6.001 s | 6.011 s | 1.206 | 1.68x, -40.5% wall |
| direct-pack, 4 lanes | 3.697 s | 3.883 s | 3.790 s | 1.914 | 2.66x, -62.5% wall |

The scratch path spent 1.34-1.35 seconds copying the 3,072 source tensors into
their final packed buffers. Direct packing removes that cost. The rest of the
gain comes from storage concurrency across independent canonical jobs.

Command:

```bash
target/release/examples/hfq_load_pipeline_bench \
  --model /home/kaden/ds4-gfx1151-evidence/2026-08-03-ds4-dspark-pm4-canary/model-e8/deepseek-v4-flash-0731.mq2r \
  --ds4-expert-layers 0:4 --lanes 1,2,4 --repeat 2 --max-bytes 8GiB \
  --json-out /home/kaden/ds4-gfx1151-evidence/2026-08-05-hfq-loader-microbench/ds4-layers0-4-read-r2.json
```

## Canonical gfx1151 upload screen

The same 7.248 GB plan was repeated with the existing synchronous
`Gpu::upload_raw` consumer. The repository GPU lock was held, and runtime device
discovery printed `GPU dev 1: gfx1151 (103.1 GB VRAM, HIP 7.14)`.

| Path | Pipeline wall | Effective GB/s | Summed upload | Peak live host output | Out-of-order completions |
|---|---:|---:|---:|---:|---:|
| scratch-pack, 1 lane | 10.032 s | 0.722 | 283 ms | 1.208 GB | 0 |
| direct-pack, 1 lane | 9.183 s | 0.789 | 259 ms | 1.208 GB | 0 |
| direct-pack, 2 lanes | 6.182 s | 1.172 | 244 ms | 1.812 GB | 2 |
| direct-pack, 4 lanes | 4.121 s | 1.759 | 211 ms | 3.624 GB | 4 |

The four-lane result is 2.43x faster than the current-shaped baseline and cuts
wall time by 58.9%. Canonical HIP upload consumes only 2.8% of baseline wall and
5.1% of the four-lane wall, so synchronous upload does not erase the storage
win. Every GPU-upload case consumed jobs in canonical order and matched the
full-data checksum.

Command:

```bash
source scripts/gpu-lock.sh
gpu_acquire hfq-loader-upload-screen
target/release/examples/hfq_load_pipeline_bench \
  --model /home/kaden/ds4-gfx1151-evidence/2026-08-03-ds4-dspark-pm4-canary/model-e8/deepseek-v4-flash-0731.mq2r \
  --ds4-expert-layers 0:4 --lanes 1,2,4 --max-bytes 8GiB \
  --gpu-upload --device 1 \
  --json-out /home/kaden/ds4-gfx1151-evidence/2026-08-05-hfq-loader-microbench/ds4-layers0-4-gfx1151-upload.json
gpu_release
```

## Interpretation

Four reader lanes are justified for the production prototype. The measured
gain survives the real upload consumer while retaining deterministic allocation
order. The implementation should remain bounded and ordered:

1. derive the canonical destination plan before starting readers;
2. let four independent file handles pread directly into final host buffers;
3. retain at most four completed/in-flight output buffers;
4. upload and install tensors strictly in canonical plan order;
5. join all readers and copies before kernel prewarm or retained-PM4 capture.

PM4 cannot accelerate file ingestion and is not part of this loader lever. It
remains valid after the load-completion barrier because pointer-table and
allocation order stay canonical. Static-weight VMM is not required to realize
this measured win; it remains a possible later mechanism for reserving stable
virtual destinations.

These results screen a DS4-shaped 7.248 GB slice, not complete model startup.
The next gate is a production-loader prototype followed by full 82 GB cold-load
timing, resident-byte/pointer-order equality, generation coherence, and retained
PM4 route identity.
