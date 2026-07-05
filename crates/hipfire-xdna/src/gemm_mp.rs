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
        let (cols, mt, nb) = (self.cols, self.mt, self.nb);
        let (aw, cw) = (self.aw(), self.cw());
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
            // C: core ci slab j block at (ci*nb+j)*cw, row-major -> output rows
            // [row0+ci*mt*MR, +), cols [j*NT*MN, +).
            let out: &[i32] = unsafe {
                std::slice::from_raw_parts(
                    self.c_buf.as_slice().as_ptr() as *const i32,
                    cols * nb * cw,
                )
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
        Ok(())
    }
}
