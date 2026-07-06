//! `NpuGemmMp` — the M-parallel W-broadcast W4A8 GEMM primitive (productionized
//! `r6_gen_mp.py` array + `r6_gemm_ts.cc` tensor-stream kernel). This is the best
//! runtime-callable NPU GEMM path: one xclbin handles any M, weights are packed and loaded
//! ONCE and broadcast to all cores, and A/C move ROW-MAJOR (the kernel's tensor streams
//! tile in-core — no CPU marshaling). Each dispatch computes `COLS` distinct M-blocks over
//! full N; [`Self::run`] tiles M over blocking dispatches (reliable — no pipelined-readback
//! coherence hazard). Measured ~1.45 TOPS e2e on halo, flat across batch (weight-bandwidth-
//! bound). See benchmarks/npu_gemm_tuning/r6/README.md for the topology + ceiling analysis.
//!
//! Shape contract: `K == k()` (single K-chunk), `N == n()`, `M % rows_per_dispatch() == 0`.
//! Linux-only (amdxdna).
#![cfg(target_os = "linux")]

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const MR: usize = 4; // mmul M
const MK: usize = 16; // mmul K
const MN: usize = 16; // mmul N
const NT: usize = 4; // N-blocks per slab (r6_gemm_ts.cc accumulator count)

/// A loaded M-parallel W-broadcast R6 kernel specialized for (COLS, MT, KCHUNK, NB), with
/// reusable arg buffers and the broadcast weights resident.
pub struct NpuGemmMp {
    kernel: NpuKernel,
    cols: usize,
    mt: usize,
    kchunk: usize,
    nb: usize,
    a_buf: DeviceBuffer, // COLS M-blocks (row-major), one per core, per dispatch
    w_buf: DeviceBuffer, // NB broadcast weight slabs (tile-major int4), loaded once
    c_buf: DeviceBuffer, // COLS*NB output blocks (row-major)
    w_loaded: bool,
}

impl NpuGemmMp {
    /// M rows computed per dispatch (`COLS` M-blocks of `MT·MR` rows).
    pub fn rows_per_dispatch(&self) -> usize {
        self.cols * self.mt * MR
    }
    /// N this kernel computes (full N in one dispatch): `NB·NT·MN`.
    pub fn n(&self) -> usize {
        self.nb * NT * MN
    }
    /// K contracted per dispatch (single chunk): `KCHUNK·MK`.
    pub fn k(&self) -> usize {
        self.kchunk * MK
    }

    fn aw(&self) -> usize {
        self.mt * self.kchunk * MR * MK
    }
    fn ww(&self) -> usize {
        NT * self.kchunk * MK * MN / 2
    }
    fn cw(&self) -> usize {
        self.mt * NT * MR * MN
    }

    /// Load an M-parallel xclbin built with `r6_gen_mp.py` (COLS cores, ROUNDS=1) for
    /// (mt, kchunk, nb) and allocate its arg buffers. Call [`Self::load_weights`] before
    /// [`Self::run`].
    pub fn load(
        xclbin: &[u8],
        insts: &[u8],
        cols: usize,
        mt: usize,
        kchunk: usize,
        nb: usize,
    ) -> Result<Self, XdnaError> {
        let kernel = NpuKernel::load(xclbin, insts)?;
        let aw = mt * kchunk * MR * MK;
        let ww = NT * kchunk * MK * MN / 2;
        let cw = mt * NT * MR * MN;
        let a_buf = kernel.alloc_arg(cols * aw)?;
        let w_buf = kernel.alloc_arg(nb * ww)?;
        let c_buf = kernel.alloc_arg(cols * nb * cw * 4)?;
        Ok(Self {
            kernel,
            cols,
            mt,
            kchunk,
            nb,
            a_buf,
            w_buf,
            c_buf,
            w_loaded: false,
        })
    }

    /// Load from a standard `r6_cache.sh` cache dir, parsing (COLS, MT, KCHUNK, NB) from its
    /// name (`..._{MT}x{NT}x{KCHUNK}_c{COLS}_nb{NB}`) so the config can't silently mismatch
    /// the xclbin. Rejects whole-GEMM `_r{ROUNDS}` builds (different layout) and any NT≠4.
    pub fn load_cached(dir: &str) -> Result<Self, XdnaError> {
        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).map_err(XdnaError::Open)?;
        let insts = std::fs::read(format!("{dir}/insts.bin")).map_err(XdnaError::Open)?;
        let base = std::path::Path::new(dir)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let toks: Vec<&str> = base.split('_').collect();
        let bad = || XdnaError::BadCacheName(base.to_string());
        // A `_r{N}` token means a whole-GEMM (ROUNDS) build — not this per-dispatch primitive.
        if toks.iter().any(|t| {
            t.strip_prefix('r')
                .is_some_and(|r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit()))
        }) {
            return Err(bad());
        }
        let pfx = |p: &str| {
            toks.iter()
                .find_map(|t| t.strip_prefix(p).and_then(|r| r.parse().ok()))
        };
        let nb: usize = pfx("nb").ok_or_else(bad)?;
        let cols: usize = pfx("c").ok_or_else(bad)?;
        let dims = toks
            .iter()
            .find(|t| t.split('x').count() == 3)
            .ok_or_else(bad)?;
        let d: Vec<usize> = dims.split('x').filter_map(|s| s.parse().ok()).collect();
        if d.len() != 3 || d[1] != NT {
            return Err(bad());
        }
        Self::load(&xclbin, &insts, cols, d[0], d[2], nb)
    }

    /// Pack a full `K×N` int4 weight matrix (`-8..=7`, one value per byte, row-major) into
    /// the broadcast slab layout — the slow bit-packing that must NOT happen per inference
    /// (weights are static). Returns `NB·ww` bytes for [`Self::load_weights`].
    pub fn prepack_weights(&self, k: usize, n: usize, w_int4: &[i8]) -> Vec<u8> {
        assert_eq!(k, self.k(), "K");
        assert_eq!(n, self.n(), "N");
        let (kc, nb, ww) = (self.kchunk, self.nb, self.ww());
        let mut out = vec![0u8; nb * ww];
        for j in 0..nb {
            for nt in 0..NT {
                for ki in 0..kc {
                    for kk in 0..MK {
                        for nn in 0..MN {
                            let kg = ki * MK + kk;
                            let ng = j * NT * MN + nt * MN + nn;
                            let idx = (nt * kc + ki) * (MK * MN) + kk * MN + nn;
                            let u = (w_int4[kg * n + ng] & 0xf) as u8;
                            out[j * ww + idx / 2] |= if idx % 2 == 0 { u } else { u << 4 };
                        }
                    }
                }
            }
        }
        out
    }

    /// Load packed weights (from [`Self::prepack_weights`]) into the resident broadcast
    /// buffer once; every [`Self::run`] dispatch reuses them (fanned to all cores).
    pub fn load_weights(&mut self, packed_w: &[u8]) {
        self.w_buf.as_mut_slice().copy_from_slice(packed_w);
        self.w_loaded = true;
    }

    /// Full GEMM `C[M,N] = A[M,K] · W[K,N]` (W4A8), tiling M over blocking dispatches. `a`
    /// row-major `M×K` int8, `c` row-major `M×N` int32. Requires `load_weights` first,
    /// `K == k()`, `N == n()`, and `M % rows_per_dispatch() == 0`.
    pub fn run(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[i8],
        c: &mut [i32],
    ) -> Result<(), XdnaError> {
        assert!(self.w_loaded, "call load_weights() before run()");
        assert_eq!(k, self.k(), "K");
        assert_eq!(n, self.n(), "N");
        let rows_per = self.rows_per_dispatch();
        assert!(m % rows_per == 0, "M must be a multiple of {rows_per}");
        let (cols, mt, aw) = (self.cols, self.mt, self.aw());
        for d in 0..(m / rows_per) {
            let row0 = d * rows_per;
            // COLS row-major M-blocks -> a_buf (the kernel's A tensor stream tiles in-core).
            {
                let s = self.a_buf.as_mut_slice();
                for ci in 0..cols {
                    for lr in 0..mt * MR {
                        let src = (row0 + ci * mt * MR + lr) * k;
                        for kk in 0..k {
                            s[ci * aw + lr * k + kk] = a[src + kk] as u8;
                        }
                    }
                }
            }
            self.kernel
                .dispatch(&[&self.a_buf, &self.w_buf, &self.c_buf])?;
            self.read_c_tile(row0, n, c); // de-block c_buf -> rows [row0,+) of row-major c
        }
        Ok(())
    }

    // De-block the current c_buf (COLS*NB blocks, each (MT·MR)×(NT·MN) row-major) into rows
    // [row0, row0+rows_per_dispatch()) of a row-major `c` `M×N`. This host copy is exactly
    // what zero-copy avoids — a GPU consumer reads the block layout from the shared buffer
    // directly (see `run_into_shared` + `c_block_offset`).
    fn read_c_tile(&self, row0: usize, n: usize, c: &mut [i32]) {
        let (cols, mt, nb, cw) = (self.cols, self.mt, self.nb, self.cw());
        let out: &[i32] = unsafe {
            std::slice::from_raw_parts(self.c_buf.as_slice().as_ptr() as *const i32, cols * nb * cw)
        };
        for ci in 0..cols {
            for j in 0..nb {
                for lr in 0..mt * MR {
                    let base = (ci * nb + j) * cw + lr * (NT * MN);
                    let dst = (row0 + ci * mt * MR + lr) * n + j * NT * MN;
                    c[dst..dst + NT * MN].copy_from_slice(&out[base..base + NT * MN]);
                }
            }
        }
    }

    /// Byte size the output buffer must be for one dispatch's C: `COLS·NB·(MT·NT·MR·MN)·4`.
    pub fn c_buf_bytes(&self) -> usize {
        self.cols * self.nb * self.cw() * 4
    }

    /// Replace the SHMEM output buffer with an imported GPU dma-buf (zero-copy). After this,
    /// [`Self::run_into_shared`] writes C straight into the GPU-shared pages — no host copy.
    /// `size` must be [`Self::c_buf_bytes`] (one dispatch's C). The dma-buf is typically an
    /// amdgpu GTT BO exported via `PRIME_HANDLE_TO_FD`; the driver `dma_buf_get`s the fd.
    pub fn attach_output_dmabuf(&mut self, fd: i32, size: usize) -> Result<(), XdnaError> {
        assert_eq!(size, self.c_buf_bytes(), "output dma-buf size");
        self.c_buf = self.kernel.import_dmabuf(fd, size, true)?;
        Ok(())
    }

    /// Byte offset (into the output buffer) of the C block for (`core`, `slab`): a
    /// (MT·MR)×(NT·MN) row-major int32 tile covering global rows
    /// [`core*MT*MR`, +), cols [`slab*NT*MN`, +) of this dispatch's M-tile. Lets a GPU
    /// consumer index the block-layout C in the shared buffer directly.
    pub fn c_block_offset_i32(&self, core: usize, slab: usize) -> usize {
        (core * self.nb + slab) * self.cw()
    }

    /// Run ONE M-block (`a` = `rows_per_dispatch()×K` row-major int8) with C written directly
    /// into the attached output dma-buf — **no host readback**. The result lands in the
    /// GPU-shared pages in the NPU block layout (see [`Self::c_block_offset_i32`]); the GPU
    /// reads it with zero host involvement. Requires [`Self::attach_output_dmabuf`] +
    /// [`Self::load_weights`]. For full M, drive this per M-tile and consume between calls
    /// (the single output buffer is reused each dispatch).
    pub fn run_into_shared(&mut self, k: usize, n: usize, a: &[i8]) -> Result<(), XdnaError> {
        assert!(
            self.w_loaded,
            "call load_weights() before run_into_shared()"
        );
        assert_eq!(k, self.k(), "K");
        assert_eq!(n, self.n(), "N");
        let rows_per = self.rows_per_dispatch();
        assert_eq!(a.len(), rows_per * k, "A must be exactly one M-block");
        let (cols, mt, aw) = (self.cols, self.mt, self.aw());
        {
            let s = self.a_buf.as_mut_slice();
            for ci in 0..cols {
                for lr in 0..mt * MR {
                    let src = (ci * mt * MR + lr) * k;
                    for kk in 0..k {
                        s[ci * aw + lr * k + kk] = a[src + kk] as u8;
                    }
                }
            }
        }
        self.kernel
            .dispatch(&[&self.a_buf, &self.w_buf, &self.c_buf])?;
        Ok(()) // C is now in the shared dma-buf; no host copy
    }

    /// De-block the shared/output buffer's current C into a row-major `rows_per × N` host
    /// buffer — for validation or a host (non-GPU) consumer of [`Self::run_into_shared`].
    pub fn read_shared_rowmajor(&self, n: usize, c: &mut [i32]) {
        self.read_c_tile(0, n, c);
    }
}
