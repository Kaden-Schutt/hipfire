# Persistent-B XDNA status

The XDNA integration is fail-closed. Hipfire's existing HIP projection path is
unchanged and `accelerator.xdna` defaults to `off`.

## Implemented

- Raw amdxdna UAPI runtime with RAII ownership for devices, contexts, buffer
  objects, sync objects, programs, retained command rings, and submission
  tickets.
- ROCr portable dma-buf export of HIP-owned allocations and amdxdna PRIME
  import without XRT.
- Versioned, checksummed PDI/instruction manifests with device, firmware,
  layout, shape, ordered binding access/size/alignment, and instruction-count
  validation. Submission fails before an ioctl if the runtime bindings do not
  exactly match the artifact ABI.
- Manifest version 2 adds a typed arithmetic contract. Production loading
  requires `q8_w8a16_f32`; the substrate probes are explicitly labeled
  `bf16_bf16_f32_diagnostic`, `q8_decode_bf16_diagnostic`, or
  `q8_w8a16_microtile_diagnostic`, and the stop-gate artifact is
  `q8_w8a16_full_array_diagnostic`. A diagnostic artifact cannot be loaded by
  the Q8 projection path even if its dimensions match.
- Typed `off`, `shadow`, `auto`, and `force` configuration plus a lazy
  `XdnaController` on `rdna_compute::Gpu`.
- Model-epoch buffer registration, identity-based reuse, narrow Qwen3.6 A3B
  admission, manifest-backed shape and masked-tail revalidation at submission,
  poisoning, route counters, and diagnostic timing.
- Daemon load/unload advances the model epoch before HIP owners can be freed,
  dropping healthy imports while their allocations are still live. A poisoned
  controller remains quarantined across model lifecycle changes.
- Real amdxdna node discovery with sysfs driver validation. Compatibility
  symlinks are rejected, and multiple NPUs require an explicit real path.
- Quarantine-on-fault: an unfinished command is never implicitly waited or
  destroyed from `Drop`. Hipfire permanently poisons the route, retains the
  failed runtime until process exit, and never attempts an automatic unbind,
  module reload, PCI reset, warmboot, or reboot.
- Bounded 1–2000 ms submission waits and diagnostic recovery guidance,
  including the owner PID and quarantined buffer accounting.
- The daemon's existing `diag` response now includes the full XDNA safety
  state, artifact/device identity and arithmetic contract, admission reason,
  route counters, poison reason, quarantine accounting, and timing breakdown.

## July 23 hipx incident and recovery

The NPU was restored without rebooting. The recovered path now certifies the
BF16 substrate interop and retained-command lifecycle, but not the production
packed-Q8 overlay:

- HIP allocation -> ROCr dma-buf -> amdxdna PRIME import succeeded.
- A real 512x2048x2048 BF16 AIE dispatch completed. Reading the output directly
  through the imported dma-buf mapping returned the same corrupt tiny values as
  the known-good amdxdna SHARE-buffer test, so this run did not isolate a
  HIP-cache visibility failure.
- `AMDXDNA_SYNC_BO` with `FROM_DEVICE` faulted in
  `amdxdna_hwctx_sync_debug_bo`. The installed driver invokes a debug-BO path
  for that direction. Redline no longer exposes a `FROM_DEVICE` method at all;
  diagnostic readback uses the dma-buf CPU-access protocol.
- The faulting process remained as a zombie leader with two live KFD wait
  threads. `/usr/local/sbin/gpukill -t 10 <pid>` released it cleanly on
  `SIGINT`; no hard kill was required.
- After all device holders were gone, an NPU-only amdxdna unbind/rebind reloaded
  `amdnpu/17f0_11/npu_7.sbin`. The known-safe SHARE-buffer BF16
  512x2048x2048 verifier then passed with zero bad samples at approximately
  898 us median.
- The real node became `/dev/accel/accel1`. A compatibility
  `/dev/accel/accel0 -> accel1` symlink must not be used: ROCr pairs the device
  basename with `/sys/class/accel/<basename>` and otherwise reports
  `HSA_STATUS_ERROR_OUT_OF_RESOURCES`. Redline now discovers the one real
  amdxdna character node, or validates `HIPFIRE_XDNA_DEVICE` when explicitly
  set.
- `hipx-warmboot` and reboot are not recovery options for this integration.
  Hipfire must stop at poison/fallback and leave recovery to the operator.
- A guarded post-recovery probe selected HIP device 1 (`gfx1151`) and the real
  `/dev/accel/accel1` node. The dma-buf mapping and a subsequent HIP read both
  returned the exact expected output.
- Thirty retained submissions completed at 1085.592 us p50 and 1151.135 us
  p99. A 1,000-submission soak completed at 1015.181 us p50 and 1132.139 us
  p99, with no remaining process, `/dev/kfd`, or accel-node holders and no new
  kernel errors.
- The existing complete-panel measurement was approximately 500 us p50, above
  the required 340 us p50 stop threshold.
- No production packed-Q8 W8A16 PDI/instruction artifact is present yet.

Accordingly, automatic and forced XDNA execution remain compile-time locked.
The retained BF16 measurements are for M=512, K=N=2048 and time the ticket
lifecycle; they are not the M=256 packed-Q8 acceptance measurement. The Qwen
projection sites and layer-major driver are intentionally not wired.

## M=256 XDNA2 compute-ceiling probe

The existing Apache-licensed mlir-aie XDNA2 campaign was used to answer the
shape question without changing its certified M=512 result. Its best i8 R6
geometry was rebuilt for M=256, K=N=2048 in an isolated temporary artifact
root.

The stock design initially rejected M=256 before compilation: its C-output
DMA tiler always described two 256-row transfer groups. The runtime loop
already bounds the final group. A temporary shape adapter changed the static C
transfer-group repeat from two to the number of available row groups, which is
one for M=256. The certified M=512 source and artifacts were not modified.

The adapted i8 x i8 -> i32 artifact passed two-seed full verification with zero
mismatches. Five fresh-process sampled runs produced 150 timed submissions:

| Metric | Result | Ceiling gate |
|---|---:|---:|
| Aggregate p50 | 312.886 us | <=340 us: pass |
| Aggregate p99 | 474.449 us | <=400 us: fail |
| Range | 296.655–490.048 us | — |
| Samples <=340 us | 114/150 | — |
| Samples <=400 us | 138/150 | — |

Each fresh-process median was below 340 us (305.963–316.889 us). The p99 miss
came from the first several submissions while the NPU ramped; steady samples
were approximately 297–320 us. The separate full-verification run measured
300.193 us p50 and 477.414 us maximum while checking both fixed seeds.

This is useful evidence that the 256-token geometry has enough XDNA2 compute
ceiling after fixing the transfer shape. It does **not** pass the production
stop gate: the probe uses CPU-owned SHARE buffers and i8 activations/weights.
It excludes HIP dma-buf transition cost, BF16-to-staging conversion, direct
Q8_0 scale handling, F32 rescaling, and GPU visibility. The tail requirement
also still fails. No NPU reset was needed, and the real accel node was
holder-free after the run.

The installed AIE2P API does not expose a BF16 x i8 MMUL specialization. It
does expose BF16 x BF16, BF16 x `bfp16ebs8`, and int16 x int8 paths. The first
production kernel should therefore retain BF16 activations and decode each
native Q8_0 B microtile into BF16 in AIE/MemTile SRAM. That decoded microtile
is then reused across every prompt chunk before eviction while F32
accumulators remain live. This keeps the model's imported Q8_0 allocation as
the only weight backing, avoids a model-sized repack, and amortizes the decode
over the exact layer-major reuse dimension. An int16-scaled activation x int8
variant remains a separately gated fallback because it changes the activation
quantization and scale ABI.

`Scottcjn/open-xdna` was reviewed as a design reference only. It targets
XDNA1/AIE2, uses XRT/IRON, and is AGPL-3.0, so no source or artifact was copied
into Hipfire. Its pruning examples reinforce a later experimental rule:
selection must be fused with compaction and timed together with readback,
gather, and the downstream operation. They are not evidence that Qwen prefill
is faster.

## MQ4P target profile

The production Qwen target is `qwen3.6-35b-a3b.mq4p`. On locked gfx1151
profiling, its mixed grouped MoE kernel accounted for 38.5% of attributed
kernel time at 512 tokens and 35.2% at 2048. The complete Q8 projection bundle
accounted for 37.1% and 34.3%, respectively.

The MoE kernel's average call time stayed effectively constant while the call
count scaled from 160 to 640. That confirms the current 256-token chunk-major
schedule is the reusable-work boundary. A later MQ4P MoE overlay should
aggregate routes across all prompt chunks under one layer/expert program.

MQ4P is not a uniform W4 model: its routed-expert tensor population is 50.0%
MQ3G256Lloyd, 30.1% MQ4G256, and 19.9% MQ6G256. The existing uniform Q8
projection overlay remains the first lower-risk kernel. Any MoE artifact must
support and independently certify all three packed formats.

See `qwen-a3b-moe-crossover-2026-07-23.md` for the measurements and rejected
small-batch route trial.

## Direct-Q8 and persistent-microtile proof

The first native-Q8 XDNA2 sources now live under
`crates/redline-xdna/kernels/`. They were built in an isolated temporary root
on hipx; the dirty certified autoresearch campaign was not modified.

The decoder consumes Hipfire's exact row-major Q8_0 bytes: one little-endian
FP16 scale followed by 32 signed bytes for every K=32 block. It converts the
FP16 scale with an AIE-side bit-exact IEEE decoder, emits BF16, and never
creates a model-sized repack. A K=2048 GPU-owned input/output round trip passed
all 2,048 values bit-for-bit through both mapped dma-buf and HIP visibility.
Ten retained submissions measured 269.845 us p50 and 392.254 us p99.

The retained W8A16 diagnostic then decoded one native `W[16,64]` Q8_0 panel
once into a placer-accounted 2 KiB AIE-local BF16 buffer. One command reused
that panel across 2, 8, or 16 logical 8-row chunks and emitted F32 through the
native AIE2P BF16 MMUL path:

| Chunks | Rows | p50 | p99 | p50 / chunk |
|---:|---:|---:|---:|---:|
| 2 | 16 | 173.685 us | 326.080 us | 86.843 us |
| 8 | 64 | 179.817 us | 332.072 us | 22.477 us |
| 16 | 128 | 187.902 us | 366.276 us | 11.744 us |

Every run used one imported B buffer and one B decode per submission. All
outputs were exact (`cosine=1.0`, `NRMSE=0`) through mapped dma-buf and HIP
readback. The near-flat command time and 7.4x reduction in amortized cost from
2 to 16 chunks validate the persistent-panel lever.

An early 16-chunk variant timed out because default FIFO depth duplicated the
whole A/C objects. A depth-1 follow-up exposed a second issue: the implicit
C++ decoded array could overlap the activation placement for small shapes.
The final artifact uses an explicit IRON `Buffer`, so the allocator accounts
for the decoded panel. The guarded timeout left no process or device holders;
no gpukill, unbind/rebind, warmboot, or reboot was used.

This is still a microtile proof, not the complete K=2048, batch-256 projection
panel and not the 340/400 us production gate. The next shape is a streamed
kernel: keep a 16x64 decoded B tile resident, stream 8x64 activation tiles from
every 256-token chunk, keep F32 accumulators live across all 32 K tiles, and
distribute output panels across shims. Whole chunks must not be allocated in
compute-tile SRAM.

See `q8-persistent-microtile-2026-07-23.md` for the exact contract and
validation record.

## M=256 native-Q8 full-array stop gate

The complete 32-core diagnostic now exists under
`crates/redline-xdna/kernels/q8_full_array_xdna2/`. It consumes the GPU-owned
native Q8_0 allocation, emits F32 into a separate GPU-owned output allocation,
and executes the full M=256, K=N=2048 projection under one retained command.
Both mapped dma-buf and subsequent HIP readback were exact.

Thirty submissions measured 2,285.008 us p50 and 4,556.432 us p99 with native
BF16 MMUL. AIE2P's BFP16-emulated 8x8x8 mode improved this to 1,674.234 us p50
and 3,936.571 us p99 on the exact synthetic fixture. A retained-panel
DMA-only build reached 258.594 us p50, proving that K=512 bank-local panels
remove the steady transport ceiling; packed-Q8 decode/transpose and BF16 MMUL
remain too expensive. The cold first-submit tail also remains above 2 ms.

The complete-panel 340/400 us gate therefore fails and the delivery sequence
stops before scheduler integration. The full-array arithmetic identity is
diagnostic-only, production loading rejects it, and `off` remains the only
executable production state. See `q8-full-array-2026-07-23.md` for the staged
measurements and shape verdict.

## Required before scheduler integration

1. Replace the rejected replicated Q8-decode/BF16-MMUL dataflow with a
   technically distinct kernel that passes the 340 us p50 / 400 us p99
   complete-panel gate, including the cold tail.
2. Revalidate real-tensor numerics if BFP16 or scaled integer activations are
   used; the synthetic full-array fixture is not sufficient.
3. Run masked-tail, shadow-parity, KLD, and
   end-to-end acceptance tests on a known-good, holder-free NPU.

## Manual hipx recovery

Hipfire never performs these host operations. An operator may use this
no-reboot sequence after an XDNA poison:

1. Stop the owning process with `sudo /usr/local/sbin/gpukill -t 10 <pid>`.
2. Verify the PID is gone, no matching DRM clients remain, and `/dev/accel/*`
   plus `/dev/kfd` have no holders.
3. Write `0000:c0:00.1` to `/sys/bus/pci/drivers/amdxdna/unbind`, then to
   `/sys/bus/pci/drivers/amdxdna/bind`.
4. Validate with an ordinary SHARE-buffer workload before retrying interop.

Do not use `AMDXDNA_SYNC_BO FROM_DEVICE`, an `accel0` compatibility symlink,
PCI FLR, warmboot, or reboot as part of this runbook.
