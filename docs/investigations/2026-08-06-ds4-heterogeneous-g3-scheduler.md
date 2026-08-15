<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 heterogeneous G3 — generic cross-device scheduler

## Verdict

G3 is complete. The generic scheduler runs exact-target `gfx1100` and
`gfx1151` code objects on separate persistent AQL queues, transfers ownership
with ROCr SDMA, overlaps the shared and routed branches at all 43 layer forks,
and joins them in the original order without host synchronization inside a
decode or prefill graph. Every measured arm is raw-bit exact against its
single-`gfx1100` HIP oracle.

The measured overlap supports DeepSeek lowering: at B=1024 the overlapped
graph is 1.851x faster than the same kernels and transfers serialized, and the
same 1024-row graph sustains about 653 rows/s at 2K, 20K, and 86K synthetic
prompt depths.

This is scheduler evidence, not a DeepSeek prefill or generation claim. The
fixture uses matrix-shaped unsigned integer kernels with DS4-sized hidden
vectors; it does not use the production MFP4/MQ2 kernels or model weights.

## Identity and implementation

- Branch: `ds4-beta-staging`
- Final source commit: `2b54a824e866c603747c4e0b82315a3c58f107c2`
- ROCm/HIP: 7.14
- Dense device: `gfx1100`, PCI `0000:66:00.0`
- Expert device: `gfx1151`, PCI `0000:bf:00.0`
- Layers: 43
- Hidden width: 4096
- Queues: two persistent exact-target AQL queues
- Transfer: `hsa_amd_memory_async_copy_on_engine`, engine `0x2` forward and
  engine `0x1` return
- Persistent prefill resources: 15 device allocations, 172 kernargs, 258
  completion signals, two queues, and two parity slots

The ROCr wrapper now exposes optional async-copy profiling through
`hsa_amd_profiling_async_copy_enable` and
`hsa_amd_profiling_get_async_copy_time`. Loading is optional and fail-closed,
so an older ROCr runtime retains ordinary replay behavior and reports profiling
as unavailable rather than changing synchronization.

Sealed binary and code-object SHA256 identities:

| Artifact | SHA256 |
|---|---|
| `target/release/examples/hetero_gfx11_cooperative` | `6f4bdd6419ea63752181518b446c3e00c34f398c6e57362737e2495df2fe22e2` |
| `gfx1100/hetero_g1_gfx1100.hsaco` | `fd2dc1b407f02f7d5efc1ce29a54cb9d666fd22adac3d5514f5b4e6b9758d434` |
| `gfx1151/hetero_g1_gfx1151.hsaco` | `9246dfdc03d77634dc740a4c032d3fa1115b6c7fea1e8ace551fb46b85fc6e1f` |

## Decode result

The B=1 fixture runs a producer on `gfx1100`, copies the 4096-element
activation plus route metadata to `gfx1151`, runs shared and expert-shaped
work concurrently, returns a routed-only partial, and performs the ordered
join on `gfx1100`.

| Metric | Median / result |
|---|---:|
| Samples per device-scheduled arm | 3 |
| Host-synchronized HIP | 140.946 ms |
| Device-scheduled serial | 102.792 ms |
| Device-scheduled overlap | 67.216 ms |
| Overlap vs serial | 1.529x |
| Overlap vs host synchronization | 2.097x |
| Forks with measured overlap | 43 / 43 |
| Raw-bit oracle | exact in every sample |

The representative overlapped timeline attributes about 66.1 ms to the
shared branch, 33.7 ms to the expert branch, 0.315 ms to forward copies, and
0.796 ms to return copies. Host enqueue is 34–38 us. The only host wait is the
terminal completion wait after the complete 43-layer graph.

## Prefill matrix

The batched fixture uses `[B, 4096+16]` activation packets and `[B, 4096]`
result packets. Each layer performs matrix-shaped producer, shared, routed,
and ordered-join work. It is not a row-at-a-time decode loop.

| Batch | Samples | Serial median | Overlap median | Speedup | Rows/s | PCIe bytes / graph | Transactions | Overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 3 | 42.218 ms | 36.202 ms | 1.166x | 441.969 | 22,588,416 | 86 | 43 / 43 |
| 128 | 3 | 436.698 ms | 247.672 ms | 1.763x | 516.813 | 180,707,328 | 86 | 43 / 43 |
| 512 | 3 | 1,583.515 ms | 847.876 ms | 1.868x | 603.862 | 722,829,312 | 86 | 43 / 43 |
| 1024 | 3 | 2,902.987 ms | 1,568.220 ms | 1.851x | 652.970 | 1,445,658,624 | 86 | 43 / 43 |

All samples were raw-bit exact. Median host enqueue cost for the complete
43-layer graph was 35.717, 28.353, 45.045, and 49.783 us respectively.

## Prompt-depth screen

The depth driver repeats the exact B=1024 batched graph. It performs one
terminal completion wait per 1024-row chunk so that correctness can be checked
and the parity slot reused; there is no host wait inside a 43-layer graph and
no host-mediated cross-agent ownership transition. Therefore these rows prove
depth stability of the scheduler but do not claim that an entire 86K prompt is
submitted as one device-resident graph.

| Requested / processed rows | Chunks | Elapsed | Rows/s | Host enqueue | PCIe bytes | Transactions | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2,048 | 2 | 3,133.665 ms | 653.548 | 103.955 us | 2,891,317,248 | 172 | raw-bit exact |
| 20,480 | 20 | 31,329.469 ms | 653.698 | 978.324 us | 28,913,172,480 | 1,720 | raw-bit exact |
| 86,016 | 84 | 131,602.729 ms | 653.603 | 4,086.203 us | 121,435,324,416 | 7,224 | raw-bit exact |

The first 2K depth run compared two chained chunks with a one-chunk oracle and
failed exactness. It is preserved as `prefill-depth-2048-screen.log`, excluded
from evidence, and fixed by commit `2b54a824e` by chaining the single-device
oracle over the identical number of chunks.

## Software validation

- `cargo test -p redline-rocr --lib`: 37 passed.
- `cargo check -p rdna-compute --example hetero_gfx11_cooperative`: passed.
- `git diff --check`: passed.

## Evidence

Durable root on hipx:

```text
/home/kaden/ds4-gfx1151-evidence/2026-08-06-ds4-heterogeneous-g3/
```

Evidence SHA256 identities:

| File | SHA256 |
|---|---|
| `decode-b1-final.log` | `87daa803d885091d223a6f995bc36cf697fe1ce8e193b6a119a17203fecbd6ad` |
| `prefill-b16-final.log` | `c36c2bbc65611364c55923951e9fb81453bb365f8923e8459987367cc1241a76` |
| `prefill-b128-final.log` | `8606351ae550669d82cc448bc6528616082a7bb800420143a73ad1e3e651b1f9` |
| `prefill-b512-final.log` | `9412d19bbcf50024ba6b2bf22a9f29a97ce759e54d68ee9e7bbdec399c5404af` |
| `prefill-b1024-final.log` | `a8995bd7339077bf17525c3c77fd594cb757f217c70dcf9d08d635179c179a79` |
| `prefill-depth-2048-v2.log` | `6d335c56f552282af80701ec8c3119134dcd867592fd1d219a185dc6d0047eb4` |
| `prefill-depth-20480.log` | `ac409773b0fecd132ee25c1c13e6e337f641b5af5c40be26267e691858310626` |
| `prefill-depth-86016.log` | `e840aacb10977ca5a8f723086667d40d3af3f84ccc6d08fe948f0f75c6c8a56a` |
| excluded failed screen | `46d4262c126b3d4dbe9d992199dff9831d78a3ac8e0cc7ec00d0b274d42cdb13` |

No DS4 lowering, product tok/s, PM4, speculation, or model-quality claim is
made at this gate. G4 is the first production-model lowering gate.
