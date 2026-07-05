//! W5 — reusable NPU kernel dispatch: the `run_smoke` flow behind a Rust API the
//! runtime can call. An [`NpuKernel`] is one compiled mlir-aie kernel (one shape);
//! it opens the device, allocates the heap, creates a hwctx, loads the tile
//! program (PDI) and the instruction stream once, and then dispatches repeatedly.
//!
//! xclbins are built offline by the mlir-aie toolchain (Python is not in the
//! inference hot path); the runtime loads the cached bytes and dispatches through
//! this HIP-direct-adjacent amdxdna path.
//!
//! Linux-only (amdxdna DRM ioctls).
#![cfg(target_os = "linux")]

use crate::submit::{self, QosInfo, AMDXDNA_BO_CMD, AMDXDNA_BO_SHMEM};
use crate::xclbin::Axlf;
use crate::{DeviceBuffer, XdnaDevice, XdnaError};

/// A prepared ERT command BO for a fixed set of argument buffers. Building it costs
/// a CREATE_BO + GET_BO_INFO + mmap (~tens of µs); caching it across dispatches with
/// the same buffers removes that from the per-dispatch path (measured ~100µs → far
/// less), which matters for the runtime offload seam.
struct CachedCmd {
    arg_handles: Vec<u32>,
    exec_handles: Vec<u32>, // arg_handles + instr_bo
    cmd_bo: DeviceBuffer,
}

/// A single compiled NPU kernel with its hwctx and loaded program. Bind argument
/// buffers with [`Self::alloc_arg`], fill inputs, then [`Self::dispatch`].
pub struct NpuKernel {
    dev: XdnaDevice,
    // Backing heap for the PDI + instruction DEV BOs; must outlive the hwctx.
    _heap: DeviceBuffer,
    hwctx: u32,
    syncobj: u32,
    instr_bo: u32,
    instr_addr: u64,
    instr_size: usize,
    // Reused across dispatches; one entry per distinct argument set (e.g. the two
    // C-buffers of a pipelined loop), so alternating arg sets don't thrash the cache.
    cmd_cache: std::cell::RefCell<Vec<CachedCmd>>,
}

impl NpuKernel {
    /// Heap size backing the PDI + instruction streams. The AIE2 dev-mem window is
    /// 64 MiB; PDIs/instructions are a few KiB, so one 64 MiB heap is ample.
    const HEAP_BYTES: usize = 64 * 1024 * 1024;

    /// Load a compiled kernel: `xclbin` bytes (for the PDI) and its `insts`
    /// instruction stream. Sets up the hwctx and loads the program on hardware.
    pub fn load(xclbin: &[u8], insts: &[u8]) -> Result<Self, XdnaError> {
        let dev = XdnaDevice::open_default()?;
        let mut heap = dev.alloc_dev_heap(Self::HEAP_BYTES)?;

        let axlf = Axlf::parse(xclbin)?;
        let part = axlf.aie_partition().ok_or(XdnaError::NoAiePartition)?;
        let num_tiles = part.column_width as u32 * 4; // aie2p: 4 core rows/column

        let (hwctx, syncobj) = dev.create_hwctx(num_tiles, 0, 0x800, &QosInfo::default())?;
        let (pdi_bo, _) = dev.alloc_dev_bo(&mut heap, part.pdi)?;
        if let Err(e) = dev.config_hwctx_cu(hwctx, pdi_bo) {
            let _ = dev.destroy_hwctx(hwctx);
            return Err(e);
        }
        let (instr_bo, instr_addr) = match dev.alloc_dev_bo(&mut heap, insts) {
            Ok(v) => v,
            Err(e) => {
                let _ = dev.destroy_hwctx(hwctx);
                return Err(e);
            }
        };

        Ok(Self {
            dev,
            _heap: heap,
            hwctx,
            syncobj,
            instr_bo,
            instr_addr,
            instr_size: insts.len(),
            cmd_cache: std::cell::RefCell::new(Vec::new()),
        })
    }

    /// Allocate a SHMEM argument buffer (host-visible, NPU-accessible via PASID).
    /// The caller fills inputs and reads outputs directly through its slices.
    pub fn alloc_arg(&self, size: usize) -> Result<DeviceBuffer, XdnaError> {
        self.dev.alloc_buffer(size, AMDXDNA_BO_SHMEM)
    }

    /// Import an external dma-buf (e.g. an amdgpu GTT BO) as an argument buffer, zero-copy:
    /// the kernel then reads/writes the *same physical pages* as the exporting engine (the
    /// GPU) — no host round-trip. See [`crate::XdnaDevice::import_dmabuf`].
    pub fn import_dmabuf(
        &self,
        fd: i32,
        size: usize,
        map: bool,
    ) -> Result<DeviceBuffer, XdnaError> {
        self.dev.import_dmabuf(fd, size, map)
    }

    /// Run the kernel over `args` in kernel-signature order (e.g. A, W, C). Flushes
    /// the argument buffers to the device, submits the command, and blocks until it
    /// completes; on return the output buffers are readable directly (SHMEM is
    /// coherent once the timeline signals).
    pub fn dispatch(&self, args: &[&DeviceBuffer]) -> Result<(), XdnaError> {
        let seq = self.submit(args)?;
        self.wait(seq)
    }

    /// Non-blocking submit: flush inputs and enqueue the command, returning the
    /// timeline sequence to [`Self::wait`] on. Lets the caller overlap host work (e.g.
    /// reading a previous dispatch's output) with this dispatch's execution — commands
    /// on the hwctx run in submit order, so a later [`Self::wait`] still sees this one
    /// complete. Pair each `submit` with exactly one `wait`, and double-buffer any
    /// output the next submit would overwrite before you read it.
    pub fn submit(&self, args: &[&DeviceBuffer]) -> Result<u64, XdnaError> {
        self.submit_synced(args, None)
    }

    /// Like [`Self::submit`], but only flush the args whose `sync[i]` is true. `None`
    /// flushes all (same as `submit`). Use this to skip the `sync_bo` on inputs the host
    /// did not change since their last dispatch, and on pure outputs (the kernel writes
    /// them, so they never need a host→device flush; SHMEM is coherent for read-back once
    /// the timeline signals). Each unnecessary flush is a full-buffer cache op.
    pub fn submit_synced(
        &self,
        args: &[&DeviceBuffer],
        sync: Option<&[bool]>,
    ) -> Result<u64, XdnaError> {
        for (i, a) in args.iter().enumerate() {
            if sync.is_none_or(|s| s[i]) {
                self.dev
                    .sync_bo(a.handle(), submit::SYNC_DIRECT_TO_DEVICE, a.len())?;
            }
        }

        // Reuse the command BO per argument set — the packet's device addresses are
        // fixed per buffer, so only the first submit of a given set pays CREATE_BO +
        // mmap. One cache entry per set so alternating (pipelined) sets don't thrash.
        let arg_handles: Vec<u32> = args.iter().map(|b| b.handle()).collect();
        let mut cache = self.cmd_cache.borrow_mut();
        if !cache.iter().any(|c| c.arg_handles == arg_handles) {
            let addrs: Vec<u64> = args.iter().map(|b| b.host_addr()).collect();
            let packet = submit::dpu_cmd_packet(self.instr_addr, self.instr_size, &addrs);
            let mut cmd_bo = self.dev.alloc_buffer(4096, AMDXDNA_BO_CMD)?;
            cmd_bo.as_mut_slice()[..packet.len()].copy_from_slice(&packet);
            let mut exec_handles = arg_handles.clone();
            exec_handles.push(self.instr_bo); // instruction BO is an EXEC arg (residency)
            cache.push(CachedCmd {
                arg_handles: arg_handles.clone(),
                exec_handles,
                cmd_bo,
            });
        }
        let cmd = cache.iter().find(|c| c.arg_handles == arg_handles).unwrap();
        self.dev
            .exec_cmd(self.hwctx, cmd.cmd_bo.handle(), &cmd.exec_handles)
    }

    /// Block until the submitted command at timeline point `seq` completes; its output
    /// buffers are then readable directly (SHMEM is coherent once the timeline signals).
    pub fn wait(&self, seq: u64) -> Result<(), XdnaError> {
        self.dev.syncobj_wait(self.syncobj, seq)
    }

    /// Submit `args` WITHOUT blocking, returning an in-flight handle. Where
    /// [`Self::dispatch`] fuses submit + wait, this lets the caller drive another engine
    /// (the GPU) while the NPU runs, then [`Self::poll`] / [`Self::wait`] for completion —
    /// the basis for a GPU‖NPU microbatch pipeline (e.g. GPU verifies step N while the NPU
    /// drafts N+1). The argument buffers must outlive the handle (the caller owns them),
    /// and the handle must be waited (or polled to completion) before it is dropped.
    pub fn submit(&self, args: &[&DeviceBuffer]) -> Result<NpuInFlight, XdnaError> {
        self.submit_tagged(args, 0)
    }

    /// [`Self::submit`] with a caller-defined `tag` carried on the handle. The scheduler
    /// stamps the microbatch / layer / expert id so it can correlate NPU completions with
    /// the per-token grouping it is pipelining across the GPU without a side table — the
    /// explicit shared state between dispatcher and scheduler.
    ///
    /// Each submit builds its OWN command BO, owned by the returned handle so it stays
    /// resident until the dispatch completes. (The single-slot cache behind `dispatch`
    /// cannot back multiple in-flight dispatches: a second submit with different buffers
    /// would free the first's command BO mid-flight.)
    pub fn submit_tagged(
        &self,
        args: &[&DeviceBuffer],
        tag: u64,
    ) -> Result<NpuInFlight, XdnaError> {
        for a in args {
            self.dev
                .sync_bo(a.handle(), submit::SYNC_DIRECT_TO_DEVICE, a.len())?;
        }
        let addrs: Vec<u64> = args.iter().map(|b| b.host_addr()).collect();
        let packet = submit::dpu_cmd_packet(self.instr_addr, self.instr_size, &addrs);
        let mut cmd_bo = self.dev.alloc_buffer(4096, AMDXDNA_BO_CMD)?;
        cmd_bo.as_mut_slice()[..packet.len()].copy_from_slice(&packet);
        let mut exec_handles: Vec<u32> = args.iter().map(|b| b.handle()).collect();
        exec_handles.push(self.instr_bo); // instruction BO is an EXEC arg (residency)
        let seq = self
            .dev
            .exec_cmd(self.hwctx, cmd_bo.handle(), &exec_handles)?;
        Ok(NpuInFlight {
            seq,
            tag,
            _cmd_bo: cmd_bo,
        })
    }

    /// Non-blocking completion check for an in-flight dispatch (`true` = done). Lets the
    /// scheduler reap finished NPU work between GPU steps without parking the GPU thread.
    pub fn poll(&self, f: &NpuInFlight) -> Result<bool, XdnaError> {
        self.dev.syncobj_poll(self.syncobj, f.seq)
    }

    /// Block until an in-flight dispatch completes; on return its output buffers are
    /// readable (SHMEM is coherent once the timeline signals). Consumes the handle,
    /// freeing its command BO. Dispatches on one kernel complete in submission order.
    pub fn wait(&self, f: NpuInFlight) -> Result<(), XdnaError> {
        self.dev.syncobj_wait(self.syncobj, f.seq)
    }
}

/// An in-flight NPU dispatch from [`NpuKernel::submit`]. Owns the command BO backing the
/// submission, keeping it resident until [`NpuKernel::wait`] (or drop). Carries the
/// timeline `seq` (submission order on the kernel's hwctx) and a caller `tag` (the
/// scheduler's microbatch / layer / expert id) so the dispatcher and scheduler share
/// in-flight state explicitly: poll by handle, correlate by tag, order by seq.
pub struct NpuInFlight {
    seq: u64,
    tag: u64,
    _cmd_bo: DeviceBuffer,
}

impl NpuInFlight {
    /// Timeline point for this dispatch — monotonic per kernel; the scheduler's ordering hint.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Caller-defined correlation id (microbatch / layer / expert).
    pub fn tag(&self) -> u64 {
        self.tag
    }

    /// Re-tag in place, e.g. when the scheduler regroups tokens across a layer boundary.
    pub fn set_tag(&mut self, tag: u64) {
        self.tag = tag;
    }
}

impl Drop for NpuKernel {
    fn drop(&mut self) {
        let _ = self.dev.destroy_hwctx(self.hwctx);
    }
}
