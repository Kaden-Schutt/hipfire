# RDNA Loader DMA + HFQ Layout Plan

Date: 2026-05-19
Hardware used for first probe: gfx1151 / Strix Halo, HIP 7.13

## Current path

The Qwen3.5 loader already avoids the worst UMA failure mode by dropping the
long-lived mmap and reading tensor payloads with `pread + fadvise_dontneed`.
The remaining hot path is:

1. `pread` tensor bytes into host staging memory.
2. `hipMalloc` a device buffer for that tensor.
3. `hipMemcpy` host staging to device.

That means RDNA shader instructions are not the first lever. File-to-device
load speed is mostly controlled by host I/O, HIP allocation behavior, SDMA/H2D
copy behavior, and the granularity of tensor ranges.

## First transport probe

Implemented an experimental `PinnedH2DTransport`:

- reads with positional `pread` directly into `hipHostMalloc` page-locked host
  memory
- uploads with `hipMemcpyAsync`
- keeps the existing `Transport` trait shape, so this can be A/B tested against
  `PreadH2DTransport`

Probe command:

```bash
cargo run -p hipfire-runtime --features deltanet --example load_transport_probe -- \
  ~/.hipfire/models/qwen3.5-0.8b.mq4 --transport pread

cargo run -p hipfire-runtime --features deltanet --example load_transport_probe -- \
  ~/.hipfire/models/qwen3.5-0.8b.mq4 --transport pinned
```

Observed debug-build results on the local 0.8B MQ4 HFQ:

| Scope | pread heap staging | pinned staging |
|---|---:|---:|
| first 4 tensors, 270,187,296 bytes | 0.50 GiB/s | 2.63 GiB/s |
| full file, 320 tensors, 534,877,312 bytes | 0.85 GiB/s | 0.85 GiB/s |

Interpretation: pinned staging can improve large contiguous H2D transfers, but
the full-file path is limited by per-tensor granularity and small-copy overhead.
That points the format work toward bigger load ranges and fewer allocation/copy
submissions before chasing lower-level copy kernels.

## Slab probe update

Extended `load_transport_probe` with:

- `--mode tensor`: current one-copy-per-tensor model
- `--mode slab`: coalesces contiguous tensor payloads into GPU slabs, then
  creates non-owning `GpuTensor` aliases at tensor offsets inside the slab
- `--bank-size-mib N`: caps slab size to model layer/bank-sized chunks
- `--drop-cache`: asks Linux to evict the model file from page cache via
  `posix_fadvise(..., POSIX_FADV_DONTNEED)` before measuring
- `--profile-io`: splits slab load time into read, `hipMalloc`, H2D copy, and
  alias construction
- `--read-mode direct`: opens the model with `O_DIRECT` for the profiling path
- `--prealloc-slabs`: allocates all GPU slabs before the read/copy loop, then
  copies into those preallocated destinations and aliases tensors in place

Release-build 9B MQ4 results:

| Probe | Banks | Cache state | Throughput |
|---|---:|---|---:|
| tensor + pread | 427 tensors | warm | 4.59 GiB/s |
| tensor + pinned | 427 tensors | warm | 4.66 GiB/s |
| slab + pread, 256 MiB | 17 banks | warm | 5.34 GiB/s |
| slab + pread, 512 MiB | 10 banks | warm | 5.14 GiB/s |
| slab + pinned, 256 MiB | 17 banks | warm | 4.63 GiB/s |
| slab + pinned, 512 MiB | 10 banks | warm | 4.73 GiB/s |
| slab + pread, 256 MiB | 17 banks | `--drop-cache` | 2.54 GiB/s |
| slab + pinned, 256 MiB | 17 banks | `--drop-cache` | 2.61 GiB/s |

Follow-up finding: the initial cold-ish 2.5-2.6 GiB/s result was partly an
artifact of using `POSIX_FADV_RANDOM` on the file descriptor even though the
slab load is sequential. The transport now uses `POSIX_FADV_SEQUENTIAL`.

Split-profile results on 9B MQ4, 256 MiB banks:

| Probe | Total | Read | Alloc | Copy | Read BW | Copy BW | End-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|
| cached pread profile | 1.300 s | 0.985 s | 0.184 s | 0.131 s | 5.01 GiB/s | 37.53 GiB/s | 3.79 GiB/s |
| cached pinned profile | 1.225 s | 0.916 s | 0.185 s | 0.123 s | 5.38 GiB/s | 40.13 GiB/s | 4.03 GiB/s |
| direct profile | 0.927 s | 0.597 s | 0.198 s | 0.132 s | 8.27 GiB/s | 37.37 GiB/s | 5.32 GiB/s |
| direct profile, `--drop-cache` | 1.196 s | 0.719 s | 0.269 s | 0.207 s | 6.86 GiB/s | 23.79 GiB/s | 4.13 GiB/s |
| direct transport | 1.055-1.117 s | combined | combined | combined | n/a | n/a | 4.42-4.68 GiB/s |
| direct transport, `--drop-cache` | 1.210-1.493 s | combined | combined | combined | n/a | n/a | 3.30-4.08 GiB/s |

Preallocated slab results on 9B MQ4:

| Probe | Banks | Total | Prealloc | Load phase | Read | Copy | Load BW | Total BW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prealloc + cached pread, `--drop-cache` | 17 | 1.257 s | 0.180 s | 1.076 s | 0.945 s | 0.131 s | 4.58 GiB/s | 3.93 GiB/s |
| prealloc + cached pinned, `--drop-cache` | 17 | 1.140 s | 0.182 s | 0.958 s | 0.846 s | 0.112 s | 5.15 GiB/s | 4.33 GiB/s |
| prealloc + direct, `--drop-cache` | 17 | 0.988 s | 0.210 s | 0.779 s | 0.641 s | 0.138 s | 6.34 GiB/s | 4.99 GiB/s |
| prealloc + direct, `--drop-cache` repeat | 17 | 0.883 s | 0.190 s | 0.693 s | 0.570 s | 0.124 s | 7.12 GiB/s | 5.59 GiB/s |
| prealloc + direct, 512 MiB banks, `--drop-cache` | 10 | 0.911 s | 0.191 s | 0.720 s | 0.605 s | 0.114 s | 6.85 GiB/s | 5.42 GiB/s |

122B A10B MQ6 results on the real daemon loader and transport probes:

| Probe | Tensors/Banks | Payload | Elapsed | Throughput |
|---|---:|---:|---:|---:|
| real daemon loader, `max_seq=512`, `kv_mode=asym3` | 25311 tensors | 62.05 GiB | 25.54 s | 2.43 GiB/s |
| tensor transport probe, pread, `--drop-cache` | 25311 tensors | 62.05 GiB | 17.92 s | 3.46 GiB/s |
| direct slab transport, 512 MiB banks, `--drop-cache` | 124 banks | 62.05 GiB | 8.68 s | 7.15 GiB/s |
| direct preallocated slabs, 512 MiB banks, `--drop-cache` | 124 banks | 62.05 GiB | 9.33 s | 6.65 GiB/s total / 8.94 GiB/s load phase |
| real daemon GPU slab loader, `HIPFIRE_GPU_SLAB_LOAD=1`, 512 MiB banks | 124 banks | 62.05 GiB | 8.54 s | 7.27 GiB/s |

Real daemon GPU slab loader detail on 122B A10B MQ6:

```text
GPU slab preload: banks=124 tensors=25082 payload=62.05 GiB prealloc=2.35s load=6.05s read=4.97s copy=1.06s total_bw=7.39 GiB/s load_bw=10.26 GiB/s
weights loaded: elapsed=8.54s throughput=7.27 GiB/s payload=62.05 GiB streamed=62.05 GiB
```

The real slab loader defaults to `auto`: enabled when HIP reports the target GPU
as integrated/UMA, disabled on discrete GPUs. The env var can force behavior:

```bash
HIPFIRE_GPU_SLAB_LOAD=1 HIPFIRE_GPU_SLAB_MIB=512 ./target/release/examples/daemon  # force on
HIPFIRE_GPU_SLAB_LOAD=0 ./target/release/examples/daemon                           # force off
HIPFIRE_GPU_SLAB_LOAD=auto HIPFIRE_GPU_SLAB_MIB=512 ./target/release/examples/daemon
```

Interpretation:

- Slabs are enough to reach the quoted drive-class bandwidth from warm cache.
- Cold-ish cached reads are not H2D-limited: copy is ~37-40 GiB/s, while the
  read phase is ~5 GiB/s and `hipMalloc` costs ~180-200 ms over the 9B load.
- Direct I/O materially improves the cold path in the profiling loop and is now
  available as `DirectH2DTransport` via `HIPFIRE_LOAD_TRANSPORT=direct`, but
  the normal transport path still varies under true cache-drop conditions.
- Preallocating the GPU slab plan before reading removes `hipMalloc` from the
  serialized read/copy loop. That pushes the load phase to 6.3-7.1 GiB/s on
  direct reads; total load including preallocation lands around 5.0-5.6 GiB/s.
- On the 122B MQ6 file, persistent model-owned GPU slabs and non-owning tensor
  aliases close the gap between the synthetic slab probe and the real daemon
  loader. The remaining overhead is now mostly slab preallocation plus file read
  time, not per-tensor allocation and upload churn.
- Plain heap staging is at least as good as pinned staging for large slab copies
  on this UMA gfx1151 path. Pinned staging remains useful as a transport option,
  but it is not the main lever once copy granularity is fixed.

## HFQ v1 layout constraint

HFQ v1 has one 4096-aligned data section. Tensor payload offsets are implicit:
the reader reconstructs each offset by cumulatively summing `data_size` in index
order. Individual tensors are not independently aligned and there is no segment
table.

This makes simple per-tensor `pread` correct, but it is not ideal for:

- O_DIRECT or io_uring fixed-buffer reads
- batched DMA submissions
- one-allocation-per-layer or one-allocation-per-segment load plans
- direct storage / dma-buf paths that need explicit aligned ranges

## Proposed HFQ v2 additions

Keep HFQ v1 readable. Add a v2 index form or optional metadata extension with:

1. Explicit per-tensor `data_offset`, `data_size`, and `padded_size`.
2. Segment table with 4K-aligned ranges:
   - `embeddings`
   - `lm_head`
   - `layer.N`
   - `layer.N.experts` for MoE
   - `draft` / sidecar payloads where present
3. Optional 2 MiB segment alignment for large layers to improve huge-page and
   direct-I/O behavior.
4. Load-plan metadata:
   - segment name
   - target device / pipeline band
   - tensor offsets relative to the segment base
   - whether the segment can be copied as one opaque raw slab
5. Logical vs physical size split, so readers can ignore padding while DMA paths
   submit aligned ranges.

The immediate benefit is a `SegmentedH2DTransport` plus a `ModelGpuStorage`
owner: build the full slab plan from the tensor index, allocate all required GPU
slabs up front, submit one read/copy per segment, then create non-owning tensor
aliases inside each slab. This removes `hipMalloc` from the serialized load loop
without changing kernel weight formats.

## Next probes

1. A/B `HSA_ENABLE_SDMA=1` vs `HSA_ENABLE_SDMA=0` on the same transport probe to
   see whether SDMA or compute-copy dominates on gfx1151.
2. Prototype HFQ v2 bank metadata: explicit bank ranges, tensor offsets within
   banks, and logical vs padded sizes.
3. Turn the direct transport into an io_uring fixed-buffer experiment. Direct
   `pread` already helps; the next question is whether lower submission
   overhead and registered buffers close the remaining gap between direct
   profiling and the normal transport path.
4. Test a mapped-host zero-copy kernel that reads weights from
   `hipHostMallocMapped` memory. Expect this to be useful for demand-paged cold
   experts, not for always-hot dense weights.
