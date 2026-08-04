// SPDX-License-Identifier: MIT
// Copyright (c) 2026 mad-lab-kbando
// hipfire — see LICENSE and NOTICE in the project root.

//! Weight pager — runtime residency management for MoE/dense weights (MAD-93 v0.1).
//!
//! Hipfire today loads all weights to VRAM at startup. For models that exceed
//! VRAM (Qwen3.5-REAP-97B-A10B is the v0.1 target), we need to keep most experts
//! on host (read from the HFQ file) and page them in to VRAM on demand based on
//! routing decisions made by [`crate::cpu_router::CpuRouter`].
//!
//! ## Architecture (foundational)
//!
//! - **CPU is the scheduler authority.** [`CpuRouter`](crate::cpu_router::CpuRouter)
//!   replicates the per-layer router GEMV on CPU so we know top-k expert indices
//!   without a GPU→CPU sync inside the forward path. The pager consumes those
//!   indices and decides what to fetch / evict.
//! - **Transport is abstracted.** Today it's pread + `hipMemcpyAsync`
//!   ([`PreadH2DTransport`]). In a future commit we drop in
//!   `IoUringP2PTransport` for true NVMe→VRAM DMA without changing anything
//!   above this layer.
//! - **Stable per-weight identity.** [`WeightId`] is a small, hashable enum so
//!   the residency map and the (file_offset, byte_len) lookup table can be
//!   keyed identically. This is what an io_uring submission queue would consume
//!   directly.
//! - **Pager owns its VRAM.** All paged weight allocations route through the
//!   pager (not ad-hoc `gpu.alloc_tensor`), so we can later export the slabs as
//!   `dma_buf` for P2P DMA without reorganizing call sites.
//!
//! ## v0.1 scope
//!
//! - [`Transport`] trait + [`PreadH2DTransport`] impl
//! - [`WeightPager`] with residency map and `ensure_resident` (synchronous, no
//!   real eviction yet — assumes VRAM is large enough)
//! - [`WeightId`] schema covering MoE experts, dense attention, norms, embeds
//!
//! Real eviction, async transfer overlap, and predictive prefetch land in
//! follow-up commits — the trait shapes here are the seams those commits plug
//! into without changing the forward path.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::path::Path;

use hip_bridge::HipResult;
use rdna_compute::{Gpu, GpuTensor};

use crate::hfq::HfqFile;

// ---------------------------------------------------------------------------
// Identity: WeightId
// ---------------------------------------------------------------------------

/// Stable identity for a weight that the pager can move between host and VRAM.
///
/// The variants enumerate every kind of weight that participates in paging.
/// For v0.1 only [`WeightId::Expert`] actually pages — the others are listed
/// so the residency map can track them as "always resident" without a special
/// case at the call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightId {
    /// Routed expert weight (one of the 256 experts in Qwen3.5-MoE-A3B).
    /// `role` distinguishes the fused gate_up matrix from the down matrix.
    Expert {
        layer: u16,
        expert: u16,
        role: ExpertRole,
    },
    /// Always-on shared expert (one per layer).
    SharedExpert { layer: u16, role: SharedRole },
    /// Per-layer router weight (small, always-resident in v0.1, but tracked
    /// here so future commits can page it for very large MoE configs).
    Router { layer: u16 },
    /// Dense attention weight (q/k/v/o).
    DenseAttn { layer: u16, role: AttnRole },
    /// RMSNorm gain vector. Tiny but per-layer.
    Norm { layer: u16, kind: NormKind },
    /// Token embedding table (always resident in v0.1).
    Embed,
    /// LM head (often shares storage with Embed; pager tracks separately).
    LmHead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertRole {
    /// Fused gate || up: shape `[2 * moe_intermediate, hidden]`.
    GateUp,
    /// Down projection: shape `[hidden, moe_intermediate]`.
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedRole {
    Gate,
    Up,
    Down,
    /// Scalar sigmoid gate on the shared-expert add: `[1, hidden]` row vector.
    SigmoidGate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttnRole {
    Q,
    K,
    V,
    O,
    /// Fused QKV when the model stores them as one tensor.
    Qkv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormKind {
    /// Pre-attention norm.
    Attn,
    /// Pre-MoE / pre-FFN norm.
    Ffn,
    /// Final norm before LM head.
    Final,
}

// ---------------------------------------------------------------------------
// Transport: how bytes get from HFQ file to VRAM
// ---------------------------------------------------------------------------

/// Opaque handle for an in-flight or completed transfer. Submit returns one,
/// `wait` consumes a slice of them. v0.1 uses a simple counter; future
/// async-overlap impls will back this with a `hipEvent` or io_uring CQE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferHandle(u64);

/// One range to read into a destination buffer, for
/// [`Transport::fetch_batch_into`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FetchReq {
    /// Byte offset in the HFQ file.
    pub hfq_offset: usize,
    /// Byte length.
    pub len: usize,
    /// Byte offset within the destination tensor.
    pub dst_byte_offset: usize,
}

/// Bounds check for [`Transport::fetch_into`], factored out so the arithmetic
/// is testable without a GPU.
///
/// This is the check that stops a cache-slot write from spilling into the
/// NEIGHBOURING expert's bytes. That failure mode would not crash — it would
/// silently corrupt a different expert's weights and show up only as degraded
/// output, which is precisely the class of bug expert paging must not
/// introduce. Hence: verify before any I/O, and fail closed.
pub fn check_fetch_into_bounds(
    dst_bytes: usize,
    dst_byte_offset: usize,
    len: usize,
) -> Result<(), String> {
    let end = dst_byte_offset
        .checked_add(len)
        .ok_or_else(|| format!("fetch_into: offset {dst_byte_offset} + len {len} overflows"))?;
    if end > dst_bytes {
        return Err(format!(
            "fetch_into: write of {len} B at offset {dst_byte_offset} \
             exceeds destination of {dst_bytes} B"
        ));
    }
    Ok(())
}

/// Abstraction over how the pager moves bytes from host storage to VRAM.
///
/// **This is the migration seam for the NVMe→VRAM DMA future.** Today's impl
/// ([`PreadH2DTransport`]) does `pread` into a host staging buffer then
/// `hipMemcpyAsync` to VRAM. A future `IoUringP2PTransport` reads directly
/// into VRAM via `dma_buf` + io_uring with no host hop. The pager never sees
/// the difference.
///
/// `fetch` is responsible for both the allocation (so io_uring can use a
/// `dma_buf`-exportable slab when needed) and the transfer. `wait` exists for
/// future async overlap; today's pread path completes synchronously inside
/// `fetch` and `wait` is a no-op.
pub trait Transport: Send {
    /// Allocate a `GpuTensor` (with `DType::Raw`, shape `[len]`) and populate
    /// it with `len` bytes from `hfq_offset` in the HFQ file. Returns the
    /// fresh tensor and a handle that can be waited on.
    ///
    /// In v0.1 the transfer is synchronous (handle is informational); in
    /// follow-ups the transport may submit async and have callers `wait`.
    fn fetch(
        &mut self,
        hfq_offset: usize,
        len: usize,
        gpu: &mut Gpu,
    ) -> HipResult<(GpuTensor, TransferHandle)>;

    /// Read `len` bytes from `hfq_offset` directly into an EXISTING device
    /// buffer at `dst_byte_offset`.
    ///
    /// Unlike [`Transport::fetch`] this allocates nothing. That is what lets a
    /// pager guarantee it never calls an allocator after load: a cache miss
    /// reuses a slot it already owns, so there is no path from a miss to an
    /// allocation failure — and, on a swapless box, no path to an OOM kill.
    fn fetch_into(
        &mut self,
        hfq_offset: usize,
        len: usize,
        dst: &GpuTensor,
        dst_byte_offset: usize,
        gpu: &mut Gpu,
    ) -> HipResult<TransferHandle>;

    /// Read several ranges into ONE destination buffer.
    ///
    /// The expert pager fills up to `k_top * segments` slots per layer per
    /// token, and issuing those one at a time leaves the storage device with a
    /// single outstanding request — measured at 88-92% of the expert-read
    /// cost. A batch lets an implementation overlap them; the default is the
    /// serial loop so existing transports keep working unchanged.
    fn fetch_batch_into(
        &mut self,
        reqs: &[FetchReq],
        dst: &GpuTensor,
        gpu: &mut Gpu,
    ) -> HipResult<TransferHandle> {
        let mut last = TransferHandle(0);
        for r in reqs {
            last = self.fetch_into(r.hfq_offset, r.len, dst, r.dst_byte_offset, gpu)?;
        }
        Ok(last)
    }

    /// Block until every handle in `handles` has completed. v0.1 no-op
    /// because `fetch` is synchronous; defined for forward compatibility.
    fn wait(&mut self, handles: &[TransferHandle]) -> HipResult<()>;

    /// Hint: does this transport need pager-allocated VRAM slabs to be
    /// exported as `dma_buf` for P2P DMA? Pager checks this at allocation
    /// time. Default false (host-staged path doesn't care).
    fn requires_dma_buf_alloc(&self) -> bool {
        false
    }

    /// Hint: required alignment for `hfq_offset` in bytes. `O_DIRECT` paths
    /// need 4 KB; pread doesn't care. Pager validates on submit.
    fn alignment(&self) -> usize {
        1
    }
}

/// v0.1 transport: pread the requested byte range from the HFQ file into a
/// reusable host buffer, then upload to VRAM via [`Gpu::upload_raw`]
/// (which internally does `hipMalloc` + `hipMemcpy(H2D)`).
///
/// Synchronous in this commit. A follow-up commit replaces the staging with a
/// pool of pinned (`hipHostMalloc`'d) buffers and uses `hipMemcpyAsync` on a
/// dedicated stream so the next-layer prefetch can overlap with current-layer
/// compute.
pub struct PreadH2DTransport {
    /// Owned file handle for the HFQ file. We open our own (rather than
    /// borrowing `HfqFile`'s) so a future `IoUringP2PTransport` can register
    /// its fd with io_uring + `dma_buf` independently. Path is held alongside
    /// for diagnostics.
    file: File,
    path: std::path::PathBuf,
    /// Reusable host staging buffer. Grows monotonically to the largest
    /// tensor size we've seen.
    staging: Vec<u8>,
    /// Monotonic handle ID. v0.1 fetches complete synchronously, so this is
    /// purely informational; future async impls will key real completion
    /// state on this id.
    next_handle: u64,
    /// Split of `fetch_into` cost: disk vs host-to-device copy. Decides
    /// whether a dedicated copy stream (which needs pinned host memory and
    /// new FFI) is the right optimisation, or whether the pread dominates and
    /// concurrent reads are.
    ns_pread: u64,
    ns_copy: u64,
}

impl PreadH2DTransport {
    /// Open the HFQ file at `path` for paged reads.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        // Hint sequential-ish access for the page-cache layer. Tensors don't
        // overlap so reads are effectively sequential within a tensor and
        // random across tensors; the kernel's readahead handles the within
        // case correctly with this advice.
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            unsafe {
                libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM);
            }
        }
        Ok(Self {
            file,
            path: path.to_path_buf(),
            staging: Vec::new(),
            next_handle: 0,
            ns_pread: 0,
            ns_copy: 0,
        })
    }

    /// Path the transport was opened with. Useful for diagnostics + the
    /// future io_uring impl which needs to register the same path with
    /// io_uring SQE buffers.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read every range into the staging buffer and hand it back, WITHOUT
    /// touching the device.
    ///
    /// Split out from `fetch_batch_into` because a caller may need to
    /// transform the bytes between disk and slot. Tensor-parallel expert
    /// slicing (PR #527) is the motivating case: each rank keeps only
    /// `inter/tp` of every expert, so the bytes read are the FULL expert and
    /// the bytes written are a row-gathered subset — read length and write
    /// length differ, which the combined read-and-copy path cannot express.
    ///
    /// Returns the staging prefix holding the requests back to back in order.
    pub fn read_batch(&mut self, reqs: &[FetchReq]) -> Result<&[u8], String> {
        use rayon::prelude::*;

        if reqs.is_empty() {
            return Ok(&self.staging[..0]);
        }
        let total: usize = reqs.iter().map(|r| r.len).sum();
        if self.staging.len() < total {
            self.staging.resize(total, 0);
        }
        let mut rest: &mut [u8] = &mut self.staging[..total];
        let mut parts: Vec<(&mut [u8], usize)> = Vec::with_capacity(reqs.len());
        for r in reqs {
            let (head, tail) = rest.split_at_mut(r.len);
            parts.push((head, r.hfq_offset));
            rest = tail;
        }
        let t = std::time::Instant::now();
        let file = &self.file;
        let errs: Vec<String> = parts
            .into_par_iter()
            .filter_map(|(buf, offset)| {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::FileExt;
                    match file.read_exact_at(buf, offset as u64) {
                        Ok(()) => None,
                        Err(e) => Some(format!("pread at {offset}: {e}")),
                    }
                }
                #[cfg(not(unix))]
                {
                    let _ = (buf, offset, file);
                    Some("concurrent pread requires unix".to_string())
                }
            })
            .collect();
        self.ns_pread += t.elapsed().as_nanos() as u64;
        match errs.first() {
            Some(e) => Err(e.clone()),
            None => Ok(&self.staging[..total]),
        }
    }

    /// `(pread_ns, h2d_copy_ns)` accumulated across all `fetch_into` calls.
    pub fn io_split(&self) -> (u64, u64) {
        (self.ns_pread, self.ns_copy)
    }

    /// Grow the host staging buffer to `len` bytes up front.
    ///
    /// [`Transport::fetch_into`] grows `staging` on demand, which would be an
    /// allocation on the forward path — exactly what a bounded pager promises
    /// not to do. Calling this once at load with the largest expert size makes
    /// every later `fetch_into` allocation-free.
    pub fn reserve_staging(&mut self, len: usize) {
        if self.staging.len() < len {
            self.staging.resize(len, 0);
        }
    }

    fn next_handle(&mut self) -> TransferHandle {
        let h = TransferHandle(self.next_handle);
        self.next_handle += 1;
        h
    }

    /// Read `len` bytes at `offset` into `self.staging[..len]`. Linux uses
    /// `pread` (positional read, no seek state); other platforms fall back
    /// to `seek + read_exact` (correct but loses thread-safety on the file
    /// — a non-issue today since the pager is single-threaded).
    fn pread_into_staging(&mut self, offset: usize, len: usize) -> std::io::Result<()> {
        if self.staging.len() < len {
            self.staging.resize(len, 0);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file
                .read_exact_at(&mut self.staging[..len], offset as u64)?;
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            self.file.seek(SeekFrom::Start(offset as u64))?;
            self.file.read_exact(&mut self.staging[..len])?;
        }
        Ok(())
    }
}

impl Transport for PreadH2DTransport {
    fn fetch(
        &mut self,
        hfq_offset: usize,
        len: usize,
        gpu: &mut Gpu,
    ) -> HipResult<(GpuTensor, TransferHandle)> {
        // 1. Host: pread the bytes into our staging buffer.
        self.pread_into_staging(hfq_offset, len).map_err(|e| {
            hip_bridge::HipError::new(
                0,
                &format!("pread {} bytes at offset {}: {}", len, hfq_offset, e),
            )
        })?;
        // 2. GPU: alloc + memcpy_htod via the existing rdna-compute helper.
        //    `dtype: Raw` because the pager doesn't care about element layout
        //    — that interpretation belongs to `WeightTensor` at the call site.
        let tensor = gpu.upload_raw(&self.staging[..len], &[len])?;
        Ok((tensor, self.next_handle()))
    }

    fn fetch_into(
        &mut self,
        hfq_offset: usize,
        len: usize,
        dst: &GpuTensor,
        dst_byte_offset: usize,
        gpu: &mut Gpu,
    ) -> HipResult<TransferHandle> {
        // Bounds-check against the destination BEFORE any I/O. Writing past a
        // slot would corrupt the neighbouring expert's weights — a silent
        // quality regression rather than a crash — so fail closed here.
        check_fetch_into_bounds(dst.byte_size(), dst_byte_offset, len)
            .map_err(|m| hip_bridge::HipError::new(0, &m))?;
        let t_pread = std::time::Instant::now();
        // 1. Host: pread into the staging buffer (same path as `fetch`).
        self.pread_into_staging(hfq_offset, len).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("pread {len} bytes at offset {hfq_offset}: {e}"))
        })?;
        self.ns_pread += t_pread.elapsed().as_nanos() as u64;
        // 2. GPU: copy into the caller's existing buffer. No allocation.
        gpu.bind_thread()?;
        let t_copy = std::time::Instant::now();
        gpu.hip
            .memcpy_htod_offset(&dst.buf, dst_byte_offset, &self.staging[..len])?;
        self.ns_copy += t_copy.elapsed().as_nanos() as u64;
        Ok(self.next_handle())
    }

    /// Concurrent preads, then the host-to-device copies.
    ///
    /// io_uring was implemented for this and REMOVED: on the full 43-layer ds4
    /// model it measured ~1.9x SLOWER on pread and ~17% slower end-to-end,
    /// consistently across an interleaved A/B/A/B (ring 8122/8512 ms vs pread
    /// 4119/4656 ms; 6.5-6.6 vs 7.7-7.8 tok/s), so it was not drift.
    ///
    /// Hypothesis, unproven: expert fills are a handful of MULTI-MEGABYTE
    /// buffered reads (18 per layer at 2.4-4.7 MB each) — the shape io_uring
    /// is worst at. The kernel serves them from its worker pool with no more
    /// parallelism than rayon already extracts, while each SQE adds submission
    /// overhead. Where io_uring should win is registered buffers + O_DIRECT,
    /// which is a different experiment, not a knob on this one.
    ///
    /// `pread` is positional and takes `&File`, so N threads can read the same
    /// descriptor at once with no seek state to race on. The staging buffer is
    /// carved into one disjoint slice per request, so the threads never
    /// overlap. Rayon's pool is process-wide and already resident, so this
    /// spawns nothing per call.
    ///
    /// The copies stay serial and on the calling thread: they are only ~8% of
    /// the cost, and HIP calls from rayon workers would need per-thread device
    /// binding for no measurable gain.
    fn fetch_batch_into(
        &mut self,
        reqs: &[FetchReq],
        dst: &GpuTensor,
        gpu: &mut Gpu,
    ) -> HipResult<TransferHandle> {
        use rayon::prelude::*;

        if reqs.is_empty() {
            return Ok(self.next_handle());
        }
        if reqs.len() == 1 {
            // One request cannot overlap with itself; skip the pool.
            let r = reqs[0];
            return self.fetch_into(r.hfq_offset, r.len, dst, r.dst_byte_offset, gpu);
        }
        // Bounds-check EVERY request before any I/O — a partially applied
        // batch would leave some slots holding the wrong expert.
        let dst_bytes = dst.byte_size();
        for r in reqs {
            check_fetch_into_bounds(dst_bytes, r.dst_byte_offset, r.len)
                .map_err(|m| hip_bridge::HipError::new(0, &m))?;
        }
        let total: usize = reqs.iter().map(|r| r.len).sum();
        if self.staging.len() < total {
            self.staging.resize(total, 0);
        }

        // Carve disjoint staging slices, one per request, in request order.
        let mut rest: &mut [u8] = &mut self.staging[..total];
        let mut parts: Vec<(&mut [u8], usize)> = Vec::with_capacity(reqs.len());
        for r in reqs {
            let (head, tail) = rest.split_at_mut(r.len);
            parts.push((head, r.hfq_offset));
            rest = tail;
        }

        let t_pread = std::time::Instant::now();
        let file = &self.file;
        let errs: Vec<String> = parts
            .into_par_iter()
            .filter_map(|(buf, offset)| {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::FileExt;
                    match file.read_exact_at(buf, offset as u64) {
                        Ok(()) => None,
                        Err(e) => Some(format!("pread at {offset}: {e}")),
                    }
                }
                #[cfg(not(unix))]
                {
                    let _ = (buf, offset, file);
                    Some("concurrent pread requires unix".to_string())
                }
            })
            .collect();
        self.ns_pread += t_pread.elapsed().as_nanos() as u64;
        if let Some(e) = errs.first() {
            return Err(hip_bridge::HipError::new(0, e));
        }

        // Copies, in request order, from each request's own staging slice.
        gpu.bind_thread()?;
        let t_copy = std::time::Instant::now();
        let mut at = 0usize;
        for r in reqs {
            gpu.hip.memcpy_htod_offset(
                &dst.buf,
                r.dst_byte_offset,
                &self.staging[at..at + r.len],
            )?;
            at += r.len;
        }
        self.ns_copy += t_copy.elapsed().as_nanos() as u64;
        Ok(self.next_handle())
    }

    fn wait(&mut self, _handles: &[TransferHandle]) -> HipResult<()> {
        // Transfers complete synchronously inside `fetch` in v0.1.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Construction-time config. Keep this small and explicit — runtime flags
/// belong on `Qwen35Config`, not here.
#[derive(Debug, Clone)]
pub struct PagerConfig {
    /// Soft cap on VRAM bytes the pager is allowed to hold for paged weights.
    /// Eviction kicks in when adding a new resident weight would exceed this.
    /// `u64::MAX` means "unlimited" (effectively disables eviction — useful
    /// for testing the routing path without VRAM pressure).
    pub vram_budget_bytes: u64,
    /// If true, the pager prints structured residency events to stderr.
    /// Disabled by default; useful when debugging eviction policy.
    pub trace: bool,
}

impl Default for PagerConfig {
    fn default() -> Self {
        Self {
            vram_budget_bytes: u64::MAX,
            trace: false,
        }
    }
}

// ---------------------------------------------------------------------------
// WeightPager
// ---------------------------------------------------------------------------

/// Tracks which weights are currently resident in VRAM and provides the
/// `ensure_resident` / `evict_lru_until` primitives the forward path uses.
///
/// **This is the GPU-side of the pager.** The CPU-side scheduling
/// (compute router → decide top-k → call ensure_resident) happens in the
/// caller; see [`crate::cpu_router::CpuRouter`].
pub struct WeightPager {
    /// What's currently in VRAM. Maps weight identity to a `Resident` record
    /// holding the buffer + bookkeeping for LRU.
    resident: HashMap<WeightId, Resident>,
    /// Recency queue for LRU eviction. Most-recently-used at the back.
    /// We use VecDeque because mutations are O(n) but n is tiny (top-k
    /// experts × layers, max ~thousands), and VecDeque iteration is
    /// cache-friendly.
    lru: VecDeque<WeightId>,
    /// Bytes currently held by `resident`.
    vram_used_bytes: u64,
    /// Per-weight (file_offset, byte_len) for cold-load via Transport.
    /// Populated at registration time when the model loader walks the HFQ
    /// tensor index. Stable across the run.
    catalog: HashMap<WeightId, ByteRange>,
    /// Transport implementation (v0.1: pread + H2D, future: io_uring + P2P).
    transport: Box<dyn Transport>,
    /// Construction-time config.
    config: PagerConfig,
}

#[derive(Debug, Clone, Copy)]
pub struct ByteRange {
    pub offset: usize,
    pub len: usize,
}

/// Uniform-shape MoE expert metadata for v0.1.
///
/// Qwen3.5-MoE-A3B has 256 experts that all share the same gate_up and down
/// shape, so we store one set of dimensions per layer instead of per-expert.
/// When we add a heterogeneous-shape MoE arch (Mixtral derivatives, etc.),
/// this generalizes to `Vec<ExpertShape>` indexed by expert index.
#[derive(Debug, Clone, Copy)]
pub struct ExpertShape {
    /// Output rows of the fused gate_up matrix = `2 * moe_intermediate_size`.
    pub gate_up_m: usize,
    /// Input cols of gate_up = `hidden_size`.
    pub gate_up_k: usize,
    /// Output rows of down = `hidden_size`.
    pub down_m: usize,
    /// Input cols of down = `moe_intermediate_size`.
    pub down_k: usize,
}

struct Resident {
    /// The actual VRAM tensor (pager owns its lifecycle). `dtype: Raw` —
    /// callers reinterpret the bytes via their own `WeightTensor` wrapper at
    /// access time. Storing as `GpuTensor` (rather than the lower-level
    /// `DeviceBuffer`) keeps the pager idiomatic with the rest of the
    /// rdna-compute API and lets us free via `gpu.free_tensor`.
    tensor: GpuTensor,
    /// Cached byte length so eviction can update `vram_used_bytes` cheaply.
    bytes: u64,
}

impl WeightPager {
    pub fn new(transport: Box<dyn Transport>, config: PagerConfig) -> Self {
        Self {
            resident: HashMap::new(),
            lru: VecDeque::new(),
            vram_used_bytes: 0,
            catalog: HashMap::new(),
            transport,
            config,
        }
    }

    /// Convenience: open `hfq_path` with the v0.1 pread+H2D transport.
    /// Equivalent to constructing a [`PreadH2DTransport`] manually and passing
    /// it to [`Self::new`].
    pub fn with_pread_transport(hfq_path: &Path, config: PagerConfig) -> std::io::Result<Self> {
        let transport = PreadH2DTransport::open(hfq_path)?;
        Ok(Self::new(Box::new(transport), config))
    }

    /// Register that `id` lives at `range` in the HFQ file. Called by the
    /// loader when it walks the tensor index. Must be called before any
    /// `ensure_resident(id)` for that id.
    pub fn register(&mut self, id: WeightId, range: ByteRange) {
        self.catalog.insert(id, range);
    }

    /// Number of registered weights. Useful for diagnostics.
    pub fn registered_count(&self) -> usize {
        self.catalog.len()
    }

    /// Returns true if `id` is currently in VRAM.
    pub fn is_resident(&self, id: WeightId) -> bool {
        self.resident.contains_key(&id)
    }

    /// Ensure `id` is in VRAM. Synchronous.
    ///
    /// Behavior:
    /// - Already resident → mark recently-used, return Ok.
    /// - Not registered (loader bug) → `NotRegistered` error, no GPU work.
    /// - Cold (registered but not resident) → if adding `id` would exceed
    ///   `config.vram_budget_bytes`, evict LRU residents until enough room.
    ///   Then fetch via transport, populate, track residency.
    pub fn ensure_resident(&mut self, id: WeightId, gpu: &mut Gpu) -> Result<(), WeightPagerError> {
        if self.resident.contains_key(&id) {
            self.touch_lru(id);
            return Ok(());
        }
        let range = *self
            .catalog
            .get(&id)
            .ok_or(WeightPagerError::NotRegistered(id))?;
        let need = range.len as u64;
        // Hard cap: a single weight that exceeds `vram_budget_bytes` can't
        // fit no matter how much we evict. Reject up front rather than
        // dutifully draining the residency map and then fetching anyway —
        // that path used to silently violate the budget because
        // `evict_lru_until` interpreted `need > budget` as "free everything"
        // and stopped, allowing the subsequent fetch to push usage past the
        // cap.
        self.would_fit(need)?;
        // Evict if cold-loading `id` would exceed budget. Skip when budget is
        // u64::MAX (the unlimited / testing default — saves the LRU walk).
        if self.config.vram_budget_bytes != u64::MAX
            && self.vram_used_bytes.saturating_add(need) > self.config.vram_budget_bytes
        {
            self.evict_lru_until(need, gpu)?;
        }
        let (tensor, _handle) = self.transport.fetch(range.offset, range.len, gpu)?;
        self.vram_used_bytes = self.vram_used_bytes.saturating_add(need);
        self.resident.insert(
            id,
            Resident {
                tensor,
                bytes: need,
            },
        );
        self.lru.push_back(id);
        if self.config.trace {
            eprintln!(
                "[weight_pager] cold-load {id:?} ({} bytes) — {} resident, {} bytes used",
                range.len,
                self.resident.len(),
                self.vram_used_bytes
            );
        }
        Ok(())
    }

    /// Patch the device-side `expert_*_ptrs` indirection table so the indexed
    /// MoE GEMV kernels read the currently-resident buffer pointers for the
    /// active experts in `top_indices` for `layer`.
    ///
    /// The ptr tables are laid out as `[num_experts × u64]` (8-byte device
    /// pointers per expert slot). For each `idx` in `top_indices`, we write
    /// the GPU pointer of that expert's resident gate_up buffer into
    /// `gate_up_ptrs.buf[idx * 8 .. idx * 8 + 8]`, same for down_ptrs.
    ///
    /// Caller must have already called `ensure_resident` for both
    /// `WeightId::Expert{layer, expert: idx, role: GateUp}` and
    /// `WeightId::Expert{layer, expert: idx, role: Down}` for every idx in
    /// `top_indices` — this method asserts that and panics on miss (loader bug).
    pub fn patch_expert_ptr_table(
        &self,
        layer: u16,
        top_indices: &[u16],
        gate_up_ptrs: &GpuTensor,
        down_ptrs: &GpuTensor,
        gpu: &mut Gpu,
    ) -> HipResult<()> {
        for &idx in top_indices {
            let gate_up_id = WeightId::Expert {
                layer,
                expert: idx,
                role: ExpertRole::GateUp,
            };
            let down_id = WeightId::Expert {
                layer,
                expert: idx,
                role: ExpertRole::Down,
            };
            let gate_up_tensor = self
                .resident
                .get(&gate_up_id)
                .unwrap_or_else(|| panic!("patch_expert_ptr_table: {gate_up_id:?} not resident"));
            let down_tensor = self
                .resident
                .get(&down_id)
                .unwrap_or_else(|| panic!("patch_expert_ptr_table: {down_id:?} not resident"));
            // u64 pointer values, written into the device table at expert idx's slot.
            let gate_up_ptr = gate_up_tensor.tensor.buf.as_ptr() as u64;
            let down_ptr = down_tensor.tensor.buf.as_ptr() as u64;
            let offset = (idx as usize) * 8;
            gpu.hip
                .memcpy_htod_offset(&gate_up_ptrs.buf, offset, &gate_up_ptr.to_le_bytes())?;
            gpu.hip
                .memcpy_htod_offset(&down_ptrs.buf, offset, &down_ptr.to_le_bytes())?;
        }
        Ok(())
    }

    /// Evict residents from the LRU front (least-recently-used) until at
    /// least `need_bytes` would fit under the budget. Returns
    /// [`WeightPagerError::BudgetExhausted`] if nothing more can be evicted
    /// but space is still insufficient, OR if `need_bytes > budget` and
    /// the budget is finite (no amount of eviction can fit the requested
    /// weight in that case, and the prior implementation silently drained
    /// the residency map and let the caller violate the cap anyway).
    ///
    /// Frees evicted tensors via `gpu.free_tensor` — the underlying VRAM
    /// returns to the rdna-compute allocator pool, available for the next
    /// `transport.fetch`.
    pub fn evict_lru_until(
        &mut self,
        need_bytes: u64,
        gpu: &mut Gpu,
    ) -> Result<(), WeightPagerError> {
        // Reject up front when the requested weight is alone bigger than
        // the budget. Without this guard, `target_used` saturates to 0 and
        // the loop drains the residency map without erroring.
        self.would_fit(need_bytes)?;
        let budget = self.config.vram_budget_bytes;
        // How much we need to free so that vram_used + need <= budget.
        let target_used = budget.saturating_sub(need_bytes);
        while self.vram_used_bytes > target_used {
            let id = self
                .lru
                .pop_front()
                .ok_or(WeightPagerError::BudgetExhausted {
                    need_bytes,
                    in_use: self.vram_used_bytes,
                    budget,
                })?;
            if let Some(r) = self.resident.remove(&id) {
                self.vram_used_bytes = self.vram_used_bytes.saturating_sub(r.bytes);
                let _ = gpu.free_tensor(r.tensor);
                if self.config.trace {
                    eprintln!(
                        "[weight_pager] evict {id:?} ({} bytes) — {} resident, {} used",
                        r.bytes,
                        self.resident.len(),
                        self.vram_used_bytes
                    );
                }
            } else {
                // LRU and residency map are out of sync — caller pushed onto
                // LRU without inserting into `resident`, or removed without
                // updating LRU. In release this is a silent drop; debug
                // builds catch the invariant violation.
                debug_assert!(
                    false,
                    "weight_pager: LRU contained {id:?} but residency map did not"
                );
            }
        }
        Ok(())
    }

    /// Free all resident tensors back to the GPU pool. Called on model
    /// teardown so VRAM goes back to the system. After this, the pager is
    /// effectively reset (catalog stays, residency map is empty).
    pub fn free_all(&mut self, gpu: &mut Gpu) {
        for (_id, r) in self.resident.drain() {
            let _ = gpu.free_tensor(r.tensor);
        }
        self.lru.clear();
        self.vram_used_bytes = 0;
    }

    /// Get the resident tensor for `id`. Returns `None` if not resident
    /// (caller should `ensure_resident` first). Does not affect LRU.
    pub fn get(&self, id: WeightId) -> Option<&GpuTensor> {
        self.resident.get(&id).map(|r| &r.tensor)
    }

    /// Insert an already-resident weight. Used by the loader for
    /// always-resident weights (token embeds, norms, the router itself in
    /// v0.1) — they live in VRAM from startup but the pager tracks them so
    /// they're visible to `get()` and accounted in `vram_used_bytes`.
    pub fn insert_resident(&mut self, id: WeightId, tensor: GpuTensor, bytes: u64) {
        if let Some(prev) = self.resident.remove(&id) {
            self.vram_used_bytes = self.vram_used_bytes.saturating_sub(prev.bytes);
            self.lru.retain(|x| *x != id);
        }
        self.resident.insert(id, Resident { tensor, bytes });
        self.lru.push_back(id);
        self.vram_used_bytes = self.vram_used_bytes.saturating_add(bytes);
    }

    /// Mark `id` as recently used. No-op if not resident.
    pub fn touch_lru(&mut self, id: WeightId) {
        if let Some(pos) = self.lru.iter().position(|x| *x == id) {
            self.lru.remove(pos);
            self.lru.push_back(id);
        }
    }

    /// Bytes currently held resident. Cheap (cached, not a sum).
    pub fn vram_used_bytes(&self) -> u64 {
        self.vram_used_bytes
    }

    /// Number of currently-resident weights.
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// Pre-flight budget check. Returns Err if a fetch of `need` bytes
    /// could not fit even with full eviction. Used by `ensure_resident`
    /// and `evict_lru_until`; exposed so unit tests can exercise the
    /// invariant without constructing a real Gpu.
    pub fn would_fit(&self, need: u64) -> Result<(), WeightPagerError> {
        let budget = self.config.vram_budget_bytes;
        if budget != u64::MAX && need > budget {
            return Err(WeightPagerError::BudgetExhausted {
                need_bytes: need,
                in_use: self.vram_used_bytes,
                budget,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum WeightPagerError {
    /// Weight wasn't registered with the pager. Loader bug.
    NotRegistered(WeightId),
    /// Hipfire HIP error (transfer / alloc failed).
    Hip(hip_bridge::HipError),
    /// Eviction couldn't free enough room — budget too small for the
    /// requested weight. User needs to raise `vram_budget_bytes` or
    /// reduce the working set somehow.
    BudgetExhausted {
        need_bytes: u64,
        in_use: u64,
        budget: u64,
    },
    /// Stub for paths still being filled in.
    Unimplemented(&'static str),
}

impl std::fmt::Display for WeightPagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotRegistered(id) => write!(f, "weight not registered: {id:?}"),
            Self::Hip(e) => write!(f, "hip error: {e}"),
            Self::BudgetExhausted {
                need_bytes,
                in_use,
                budget,
            } => write!(
                f,
                "weight pager: cannot evict to fit {need_bytes} bytes \
                 (in_use={in_use}, budget={budget}); raise vram_budget_bytes \
                 or reduce paged working set"
            ),
            Self::Unimplemented(why) => write!(f, "weight pager: unimplemented ({why})"),
        }
    }
}

impl std::error::Error for WeightPagerError {}

impl From<hip_bridge::HipError> for WeightPagerError {
    fn from(e: hip_bridge::HipError) -> Self {
        Self::Hip(e)
    }
}

// ---------------------------------------------------------------------------
// Convenience: open an HfqFile by path. The loader uses the existing
// HfqFile::open directly; this re-export keeps the module's surface minimal.
// ---------------------------------------------------------------------------

/// Forwarding helper so callers don't need a separate `use crate::hfq::HfqFile`.
pub fn open_hfq(path: &Path) -> std::io::Result<HfqFile> {
    HfqFile::open(path)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::check_fetch_into_bounds;

    #[test]
    fn fetch_into_bounds_accepts_an_exact_fit() {
        // A slot write that ends exactly at the buffer end is legal.
        assert!(check_fetch_into_bounds(2_359_296 * 4, 2_359_296 * 3, 2_359_296).is_ok());
    }

    #[test]
    fn fetch_into_bounds_accepts_slot_zero() {
        assert!(check_fetch_into_bounds(2_359_296, 0, 2_359_296).is_ok());
    }

    #[test]
    fn fetch_into_bounds_rejects_a_write_past_the_end() {
        // One byte too far: would corrupt whatever follows the pool.
        let err = check_fetch_into_bounds(2_359_296 * 4, 2_359_296 * 3, 2_359_296 + 1)
            .expect_err("must reject");
        assert!(err.contains("exceeds destination"), "got {err}");
    }

    #[test]
    fn fetch_into_bounds_rejects_an_off_by_one_slot_index() {
        // Slot 4 in a 4-slot pool: the classic off-by-one, which would write
        // into the next blob rather than erroring.
        let stride = 2_359_296usize;
        let err = check_fetch_into_bounds(stride * 4, stride * 4, stride).expect_err("must reject");
        assert!(err.contains("exceeds destination"), "got {err}");
    }

    #[test]
    fn fetch_into_bounds_rejects_overflowing_arithmetic() {
        let err = check_fetch_into_bounds(1024, usize::MAX, 4096).expect_err("must reject");
        assert!(err.contains("overflows"), "got {err}");
    }

    use super::*;
    use std::io::Write;

    #[test]
    fn weight_id_is_hashable() {
        let mut map = HashMap::new();
        let a = WeightId::Expert {
            layer: 0,
            expert: 0,
            role: ExpertRole::GateUp,
        };
        let b = WeightId::Expert {
            layer: 0,
            expert: 0,
            role: ExpertRole::Down,
        };
        map.insert(a, 1u32);
        map.insert(b, 2u32);
        assert_eq!(map.get(&a), Some(&1));
        assert_eq!(map.get(&b), Some(&2));
    }

    /// Write some bytes to a temp file and verify `PreadH2DTransport::open`
    /// can read arbitrary ranges via the staging buffer. The actual upload
    /// to GPU is exercised in integration tests; here we directly call the
    /// pread helper to keep the unit test device-free.
    #[test]
    fn pread_transport_reads_range() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-pager-test-{}.bin", std::process::id()));
        let payload: Vec<u8> = (0..1024u32).flat_map(|i| (i as u8).to_le_bytes()).collect();
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&payload)
            .unwrap();

        let mut t = PreadH2DTransport::open(&path).unwrap();
        // Read [256..768) — should match payload[256..768].
        t.pread_into_staging(256, 512).unwrap();
        assert_eq!(&t.staging[..512], &payload[256..768]);
        // Read a smaller range; staging must cover it.
        t.pread_into_staging(0, 16).unwrap();
        assert_eq!(&t.staging[..16], &payload[..16]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn pager_starts_empty() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-pager-empty-{}.bin", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(b"x")
            .unwrap();
        let pager = WeightPager::with_pread_transport(&path, PagerConfig::default()).unwrap();
        assert_eq!(pager.registered_count(), 0);
        assert_eq!(pager.vram_used_bytes(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn register_then_get_returns_none_until_resident() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-pager-reg-{}.bin", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(b"x")
            .unwrap();
        let mut pager = WeightPager::with_pread_transport(&path, PagerConfig::default()).unwrap();
        let id = WeightId::Expert {
            layer: 0,
            expert: 0,
            role: ExpertRole::GateUp,
        };
        pager.register(id, ByteRange { offset: 0, len: 1 });
        assert_eq!(pager.registered_count(), 1);
        // Catalog hit, not yet resident → get returns None.
        assert!(!pager.is_resident(id));
        assert!(pager.get(id).is_none());
        // ensure_resident requires a real Gpu — exercised in integration tests.
        let _ = std::fs::remove_file(&path);
    }

    /// Regression: a need bigger than the entire budget must NOT cause
    /// `evict_lru_until` to silently drain the residency map and return Ok.
    /// Prior to this guard, `target_used = budget.saturating_sub(need)` was 0
    /// and the loop exited cleanly even though the subsequent fetch in
    /// `ensure_resident` would push usage past the cap.
    #[test]
    fn would_fit_rejects_need_bigger_than_budget() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-pager-budget-{}.bin", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(b"x")
            .unwrap();
        let pager = WeightPager::with_pread_transport(
            &path,
            PagerConfig {
                vram_budget_bytes: 100,
                trace: false,
            },
        )
        .unwrap();
        // need <= budget → ok
        assert!(pager.would_fit(50).is_ok());
        assert!(pager.would_fit(100).is_ok());
        // need > budget → BudgetExhausted, even on an empty pager
        match pager.would_fit(1000) {
            Err(WeightPagerError::BudgetExhausted {
                need_bytes,
                in_use,
                budget,
            }) => {
                assert_eq!(need_bytes, 1000);
                assert_eq!(in_use, 0);
                assert_eq!(budget, 100);
            }
            other => panic!("expected BudgetExhausted, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Sanity: with the unlimited budget (default), would_fit accepts
    /// arbitrary sizes — the cap is a no-op in that mode.
    #[test]
    fn would_fit_accepts_anything_when_budget_unlimited() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-pager-unlim-{}.bin", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(b"x")
            .unwrap();
        let pager = WeightPager::with_pread_transport(&path, PagerConfig::default()).unwrap();
        // Default is u64::MAX; even u64::MAX - 1 fits.
        assert!(pager.would_fit(u64::MAX - 1).is_ok());
        let _ = std::fs::remove_file(&path);
    }
}
