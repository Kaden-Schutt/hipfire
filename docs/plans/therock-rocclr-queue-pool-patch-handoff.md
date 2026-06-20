# TheRock ROCclr Queue Pool Patch Handoff

Status and plan for moving the gfx1103 LDS HIP-719 investigation to a larger
machine that can build TheRock with a diagnostic ROCclr patch.

## Current Hipfire State

- Reference branch: `chaingun`.
- Latest relevant hipfire commit at handoff:
  `6c4eebaf Record TheRock queue cleanup source mapping`.
- Main living investigation note:
  `docs/plans/gfx1103-lds-hip719-investigation.md`.
- This handoff assumes the full hipfire repo is copied with that living note.

## Current Failure Model

The strongest current reduced repro is the promoted direct-AB no-output HIP
probe, not the production GEMM kernel:

- Probe wrapper: `scripts/lds_direct_ab_multi_exec_matrix.sh`
- Kernel source: `scripts/lds_direct_ab_phase_probe.hip`
- Parent/child runner: `scripts/lds_direct_ab_multi_exec_parent.cpp`
- Summarizers:
  - `scripts/lds_direct_ab_artifact_summary.sh`
  - `scripts/lds_direct_ab_summary_compare.sh`

Focused failing shape:

```text
ACTIVE_X=30
ACTIVE_Y=1
ACTIVE_X_START=3
ACTIVE_Y_START=0
BLOCK_X=34
BLOCK_Y=1
LAYOUT_X=34
LAYOUT_Y=1
READS=2
ITERS=448
GRID=512x86
```

Key codegen/resource signature:

```text
selected normalized ISA hash: bb1a56b38225028e
group_segment_fixed_size: 280
private_segment_fixed_size: 0
sgpr_count: 5
vgpr_count: 8
wavefront_size: 32
s_barrier count: 8
DS op count: 12
s_waitcnt count: 12
branch count: 9
visible ds_load_b32 offset:12
```

Canonical failure signature:

```text
HIP 719 / unspecified launch failure
GFXHUB fault address: 0x000074669d000000
GCVM protection status: 0x841051
Decoded GCVM: MORE_FAULTS, PERMISSION_FAULTS, RW, cid=8, rw=1, vmid=8
regGDS_PROTECTION_FAULT: 0x3f000007
regGDS_VM_PROTECTION_FAULT: 0x0fc00113
dmesg family: MES REMOVE_QUEUE timeout, KFD queue eviction failure, MODE2 reset
```

Current behavioral model:

- Not a simple LDS instruction, barrier, LDS allocation, global memory, Rust
  runtime, or hipfire JIT issue.
- Tied to direct A/B LDS codegen/shape, high-lane shifted active placement, and
  process-local launch/queue lifetime state.
- Same-process work fails in a low band around global launch `43-46` for early
  `20+480` splits.
- `hipDeviceSynchronize`, stream recreate, `hipDeviceReset`, and HIP primary
  context reset/release do not clear that low same-process band.
- Child process exit clears the low same-process band when each child stays
  below its child-local threshold.
- Oversized child processes still fail. Example: `20,480` fails in child1
  around local sync `376`; process exit is containment, not a root-cause fix.

## TheRock Source Already Inspected

Local source snapshot used for inspection:

```text
/tmp/therock superproject:
  a8d56de8b2879b76ff2c4d5251b1c2750a8498a4

/tmp/therock/rocm-systems submodule:
  a0952b2b339b4603050acee1672b0aa0d8abb702
```

Important source findings:

- `projects/clr/hipamd/src/hip_context.cpp`
  - `hipDevicePrimaryCtxRelease()` validates `dev` and returns success.
  - `hipDevicePrimaryCtxReset()` returns success immediately.
  - These APIs are effectively not a queue/runtime cleanup boundary.
- `projects/clr/hipamd/src/hip_device_runtime.cpp`
  - `hipDeviceReset()` calls `hip::getCurrentDevice()->Reset()`.
- `projects/clr/hipamd/src/hip_device.cpp`
  - `Device::Reset()` releases HIP memory pools, destroys HIP streams, purges
    HIP memory objects, and recreates the HIP device wrapper.
  - The inspected code does not show it destroying the underlying ROCclr device
    or its normal active hardware queue pool.
- `projects/clr/rocclr/device/rocm/rocdevice.cpp`
  - `AcquireActiveQueue()` calls
    `acquireQueue(... managed=true, dedicated_queue=false, ...)`.
  - Normal active queues are inserted into `queuePool_`.
  - `ReleaseActiveQueue()` only calls `releaseQueue()` when the persistent queue
    count exceeds `settings().max_hw_queues_`.
  - `releaseQueue()` destroys CU-mask/cooperative queues, but normal managed
    queues remain pooled.
  - `Device::~Device()` destroys every queue in `queuePool_` with
    `Hsa::queue_destroy()`.
- ROCR/KFD queue-destroy chain:
  - `runtime/hsa-runtime/core/runtime/hsa.cpp`: `hsa_queue_destroy()`
    calls `Queue::Destroy()`.
  - `runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp`:
    `AqlQueue::Destroy()` deletes the queue, and `AqlQueue::~AqlQueue()` calls
    `Inactivate()`.
  - `AqlQueue::Inactivate()` calls `agent_->driver().DestroyQueue(queue_id_)`.
  - `runtime/hsa-runtime/core/driver/kfd/amd_kfd_driver.cpp`:
    `KfdDriver::DestroyQueue()` calls `hsaKmtDestroyQueue()`.
  - `libhsakmt/src/queues.c`: `hsaKmtDestroyQueue()` delegates to
    `hsaKmtDestroyQueueCtx()`.

Interpretation:

- The source explains why HIP primary-context APIs, stream recreation, and
  `hipDeviceReset()` behave unlike child-process exit in the repro.
- Ordinary active queues can remain pooled inside the same ROCclr device/runtime
  lifetime.
- A TheRock runtime patch can directly test whether normal active queue pooling
  is the retained state that causes the low same-process band.

## Diagnostic Patch Goal

Patch ROCclr behind an environment variable so default ROCm behavior is
unchanged:

```text
HIPFIRE_ROCCLR_DESTROY_ACTIVE_QUEUE_ON_RELEASE=1
```

When the env var is set:

- Avoid reusing normal active queues from `queuePool_`.
- On release of a normal managed active queue, decrement refcount as usual.
- If refcount reaches zero, erase the queue from `queuePool_` and call
  `Hsa::queue_destroy(queue)`.

This should make stream destruction/recreation and possibly `hipDeviceReset()`
reach a real KFD queue-destroy boundary for ordinary active queues.

Expected diagnostic outcomes:

- If same-process or stream-recreate splits start passing, ROCclr active queue
  lifetime is strongly implicated.
- If they still fail but child-process splits pass, the retained state is below
  ROCclr queue pooling: ROCR process runtime, KFD process state, GPUVM, MES, or
  firmware.
- If thresholds move but failures remain, queue lifetime is a contributor but
  not the whole bug.

## Suggested Patch Points

Patch file:

```text
/tmp/therock/rocm-systems/projects/clr/rocclr/device/rocm/rocdevice.cpp
```

Candidate helper:

```cpp
static bool DestroyActiveQueueOnRelease() {
  const char* v = std::getenv("HIPFIRE_ROCCLR_DESTROY_ACTIVE_QUEUE_ON_RELEASE");
  return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}
```

Likely changes:

- Include `<cstdlib>` and `<cstring>` if not already available.
- In `acquireQueue()`, bypass normal pool reuse when the env var is enabled for
  non-cooperative, non-CU-mask, managed active queues.
- In `releaseQueue()`, after decrementing refcount, if the env var is enabled
  and the queue is normal/non-CU-mask/non-cooperative and refcount is zero:
  remove it from `queuePool_` under the lock, then destroy it after releasing
  the lock.

Important care point:

- Do not call `Hsa::queue_destroy()` while still using an iterator/reference
  into `queuePool_`.
- Do not destroy a queue with `refCount > 0`.
- Keep default behavior byte-for-byte close unless the env var is set.

## Build Plan On Stronger Machine

The original small system had only about `30G` free and was missing `ninja`.
Use a larger system with enough disk. A full source path can require much more
than the minimal runtime artifacts imply, especially if `amd-llvm` has to be
built.

Recommended baseline:

- Ubuntu 24.04-like environment.
- At least `100G` free; `200G+` is safer for an LLVM-including build.
- `ninja-build`, `cmake`, `ccache`, `g++`, `pkg-config`, Python venv tools, and
  TheRock dependencies installed.

Setup:

```bash
sudo apt update
sudo apt install -y \
  gfortran git ninja-build cmake g++ pkg-config xxd automake libtool \
  python3-venv python3-dev libegl1-mesa-dev texinfo bison flex \
  ccache curl make

git clone https://github.com/ROCm/TheRock.git /tmp/therock
cd /tmp/therock
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 ./build_tools/fetch_sources.py
```

If TheRock requires its pinned `patchelf`, follow its README:

```bash
sudo env INSTALL_PREFIX=/usr/local ./dockerfiles/install_pinned_patchelf.sh
```

Configure minimal runtime build:

```bash
cd /tmp/therock
source .venv/bin/activate
eval "$(./build_tools/setup_ccache.py)"

cmake -B build-queuepatch -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx110X-all \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_CORE_RUNTIME=ON \
  -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
  -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```

Build:

```bash
cmake --build build-queuepatch --target core-runtime+dist core-hip+dist
cmake --build build-queuepatch --target therock-dist-rocm
```

If target names differ after configure:

```bash
ninja -C build-queuepatch -t targets | rg 'core-(runtime|hip)|dist-rocm'
```

Use the patched runtime without installing over `/opt/rocm`:

```bash
export ROCM_PATH=/tmp/therock/build-queuepatch/dist/rocm
export HIP_PATH="$ROCM_PATH"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
```

Verify:

```bash
which hipcc
hipcc --version
```

## Focused Test Matrix

Run each row once with the env var unset and once with:

```bash
export HIPFIRE_ROCCLR_DESTROY_ACTIVE_QUEUE_ON_RELEASE=1
```

Use a fresh artifact root for every row.

### Build-Only Preflight

```bash
cd /path/to/hipfire
BUILD_ONLY=1 \
ACTIVE_X=30 ACTIVE_Y=1 ACTIVE_X_START=3 ACTIVE_Y_START=0 \
BLOCK_X=34 BLOCK_Y=1 LAYOUT_X=34 LAYOUT_Y=1 \
READS=2 ITERS=448 GRID_X=512 GRID_Y=86 \
scripts/lds_direct_ab_multi_exec_matrix.sh \
  /tmp/hipfire-therock-queuepatch-buildonly
```

### Same-Process Early Split

Expected current behavior without patch: fail around global `43-46`.

```bash
BUILD_ONLY=0 PHASE_MODE=same PHASE_COUNTS=20,480 \
ACTIVE_X=30 ACTIVE_Y=1 ACTIVE_X_START=3 ACTIVE_Y_START=0 \
BLOCK_X=34 BLOCK_Y=1 LAYOUT_X=34 LAYOUT_Y=1 \
READS=2 ITERS=448 GRID_X=512 GRID_Y=86 \
scripts/lds_direct_ab_multi_exec_matrix.sh \
  /tmp/hipfire-therock-queuepatch-same20
```

### Stream-Recreate Split

If the env-var patch works, this is the row most likely to improve.

```bash
BUILD_ONLY=0 PHASE_MODE=stream_recreate PHASE_COUNTS=20,480 \
ACTIVE_X=30 ACTIVE_Y=1 ACTIVE_X_START=3 ACTIVE_Y_START=0 \
BLOCK_X=34 BLOCK_Y=1 LAYOUT_X=34 LAYOUT_Y=1 \
READS=2 ITERS=448 GRID_X=512 GRID_Y=86 \
scripts/lds_direct_ab_multi_exec_matrix.sh \
  /tmp/hipfire-therock-queuepatch-stream20
```

### Device-Reset Split

Current behavior: `hipDeviceReset()` returns OK but behaves like same-process
work for the low band.

```bash
BUILD_ONLY=0 PHASE_MODE=device_reset PHASE_COUNTS=20,480 \
ACTIVE_X=30 ACTIVE_Y=1 ACTIVE_X_START=3 ACTIVE_Y_START=0 \
BLOCK_X=34 BLOCK_Y=1 LAYOUT_X=34 LAYOUT_Y=1 \
READS=2 ITERS=448 GRID_X=512 GRID_Y=86 \
scripts/lds_direct_ab_multi_exec_matrix.sh \
  /tmp/hipfire-therock-queuepatch-reset20
```

### Child-Process Control

Current behavior: `40,40` passes, while oversized child `20,480` fails in the
second child around local sync `376`.

```bash
BUILD_ONLY=0 CHILD_COUNTS=40,40 \
ACTIVE_X=30 ACTIVE_Y=1 ACTIVE_X_START=3 ACTIVE_Y_START=0 \
BLOCK_X=34 BLOCK_Y=1 LAYOUT_X=34 LAYOUT_Y=1 \
READS=2 ITERS=448 GRID_X=512 GRID_Y=86 \
scripts/lds_direct_ab_multi_exec_matrix.sh \
  /tmp/hipfire-therock-queuepatch-child40

BUILD_ONLY=0 CHILD_COUNTS=20,480 \
ACTIVE_X=30 ACTIVE_Y=1 ACTIVE_X_START=3 ACTIVE_Y_START=0 \
BLOCK_X=34 BLOCK_Y=1 LAYOUT_X=34 LAYOUT_Y=1 \
READS=2 ITERS=448 GRID_X=512 GRID_Y=86 \
scripts/lds_direct_ab_multi_exec_matrix.sh \
  /tmp/hipfire-therock-queuepatch-child20-480
```

## Summarize Results

After running:

```bash
scripts/lds_direct_ab_artifact_summary.sh /tmp/hipfire-therock-queuepatch-same20
scripts/lds_direct_ab_artifact_summary.sh /tmp/hipfire-therock-queuepatch-stream20
scripts/lds_direct_ab_artifact_summary.sh /tmp/hipfire-therock-queuepatch-reset20
scripts/lds_direct_ab_artifact_summary.sh /tmp/hipfire-therock-queuepatch-child40
scripts/lds_direct_ab_artifact_summary.sh /tmp/hipfire-therock-queuepatch-child20-480
```

Capture for the living note:

- TheRock superproject SHA and `rocm-systems` SHA.
- The ROCclr patch diff.
- Patched runtime path used in `ROCM_PATH`.
- Whether the selected normalized ISA hash remains `bb1a56b38225028e`.
- Pass/fail row for each run.
- Failure local/global sync index.
- dmesg family and devcoredump GCVM/GDS signature.

## Decision Criteria

- **Patch fixes same-process / stream-recreate low band:** queue pooling is a
  root contributor. Next step is a cleaner upstreamable ROCclr debug/flush
  patch or a HIP-facing queue teardown option.
- **Patch does not change low band:** retained state is lower than ROCclr
  active queue pooling. Focus shifts to ROCR process lifetime, KFD process
  queues, GPUVM/MES state, or firmware.
- **Patch shifts thresholds only:** queue lifetime contributes, but the reduced
  LDS/codegen shape remains independently hazardous.

