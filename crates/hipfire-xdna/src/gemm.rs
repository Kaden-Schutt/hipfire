//! Wire-in step 2 — `NpuGemm`: a W4A8 GEMM primitive over the R6-TS kernel
//! (`r6_gemm_ts.cc`). Turns a standard row-major GEMM into R6 dispatches. A and C move
//! ROW-MAJOR: the kernel's in-core tensor buffer streams do the tile reshuffle via the
//! address generators (no CPU marshaling, no strided DMA), so `a_buf`/`c_buf` are plain
//! row-major blocks. Only W is pre-packed into the kernel's tile-major int4 layout — a
//! static, once-per-load cost (see [`Self::prepack_weights`]). Validated by
//! `r6_ts_verify` (0 mismatches at MT=8 and the MT=24 peak).
//!
//! `groups` = how many `NT·MN`-wide N-slabs one dispatch computes: 1 for the single
//! core (`r6_cache.sh` COLS=1), or `COLS·NB` for the array (COLS cores × NB streamed
//! blocks) — the latter is where the 20.7-TOPS throughput lives. Each dispatch does
//! one M-block (`MT·MR` rows, shared A) × one K-chunk (`KCHUNK·MK`) × `groups` N-slabs;
//! [`Self::run`] tiles the full GEMM over that (M/N independent, K accumulated).
//! Linux-only (amdxdna).
#![cfg(target_os = "linux")]

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const MR: usize = 4; // mmul M
const MK: usize = 16; // mmul K
const MN: usize = 16; // mmul N

/// A loaded R6 kernel specialized for (MT, NT, KCHUNK, groups) with reusable buffers.
pub struct NpuGemm {
    kernel: NpuKernel,
    mt: usize,
    nt: usize,
    kchunk: usize,
    groups: usize,
    a_buf: DeviceBuffer, // MT*KCHUNK tiles of MR*MK int8 (one M-block, shared)
    w_buf: DeviceBuffer, // groups * NT*KCHUNK tiles of MK*MN int4 (2/byte)
    c_buf: [DeviceBuffer; 2], // double-buffered: one dispatch writes while the host
                         // reads the other back (pipelined run_packed).
}

impl NpuGemm {
    /// M rows computed per dispatch.
    pub fn block_m(&self) -> usize {
        self.mt * MR
    }
    /// N cols computed per dispatch (`groups` N-slabs).
    pub fn block_n(&self) -> usize {
        self.groups * self.nt * MN
    }
    /// K contracted per dispatch (one chunk).
    pub fn block_k(&self) -> usize {
        self.kchunk * MK
    }

    /// Load an R6 xclbin built for (mt, nt, kchunk) with `groups` = COLS·NB N-slabs
    /// (1 for the single-core cache) and allocate its arg buffers.
    pub fn load(
        xclbin: &[u8],
        insts: &[u8],
        mt: usize,
        nt: usize,
        kchunk: usize,
        groups: usize,
    ) -> Result<Self, XdnaError> {
        let kernel = NpuKernel::load(xclbin, insts)?;
        let a_buf = kernel.alloc_arg(mt * kchunk * MR * MK)?;
        let w_buf = kernel.alloc_arg(groups * nt * kchunk * MK * MN / 2)?;
        let csz = groups * mt * nt * MR * MN * 4;
        let c_buf = [kernel.alloc_arg(csz)?, kernel.alloc_arg(csz)?];
        Ok(Self {
            kernel,
            mt,
            nt,
            kchunk,
            groups,
            a_buf,
            w_buf,
            c_buf,
        })
    }

    /// Full GEMM `C[M,N] = A[M,K] · W[K,N]` (W4A8) by tiling over R6 dispatches: M and N
    /// split into blocks, K accumulated. `a` row-major `M×K` int8, `w_int4` row-major
    /// `K×N` int4 values (`-8..=7`, one per byte), `c` row-major `M×N` int32.
    /// M/N/K must be multiples of `block_m()`/`block_n()`/`block_k()`.
    pub fn run(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[i8],
        w_int4: &[i8],
        c: &mut [i32],
    ) -> Result<(), XdnaError> {
        let (bm, bn, bk) = (self.block_m(), self.block_n(), self.block_k());
        assert!(
            m % bm == 0 && n % bn == 0 && k % bk == 0,
            "M/N/K must tile evenly (block {bm}x{bn}x{bk})"
        );
        let mut a_sub = vec![0i8; bm * bk];
        let mut w_sub = vec![0i8; bk * bn];
        let mut c_blk = vec![0i32; bm * bn];
        let mut c_acc = vec![0i32; bm * bn];
        for mo in (0..m).step_by(bm) {
            for no in (0..n).step_by(bn) {
                c_acc.iter_mut().for_each(|x| *x = 0);
                for ko in (0..k).step_by(bk) {
                    for i in 0..bm {
                        a_sub[i * bk..(i + 1) * bk]
                            .copy_from_slice(&a[(mo + i) * k + ko..(mo + i) * k + ko + bk]);
                    }
                    for i in 0..bk {
                        w_sub[i * bn..(i + 1) * bn]
                            .copy_from_slice(&w_int4[(ko + i) * n + no..(ko + i) * n + no + bn]);
                    }
                    self.run_slab(&a_sub, &w_sub, &mut c_blk)?;
                    for (acc, &v) in c_acc.iter_mut().zip(c_blk.iter()) {
                        *acc += v;
                    }
                }
                for i in 0..bm {
                    c[(mo + i) * n + no..(mo + i) * n + no + bn]
                        .copy_from_slice(&c_acc[i * bn..(i + 1) * bn]);
                }
            }
        }
        Ok(())
    }

    /// One dispatch: `a` row-major `(MT·MR) × (KCHUNK·MK)` int8; `w_int4` row-major
    /// `(KCHUNK·MK) × (groups·NT·MN)` int4 values; `c` gets row-major
    /// `(MT·MR) × (groups·NT·MN)` int32. A/C are copied row-major (the kernel's tensor
    /// streams tile them in-core); W is pre-packed tile-major. Dispatches, copies C out.
    pub fn run_slab(&mut self, a: &[i8], w_int4: &[i8], c: &mut [i32]) -> Result<(), XdnaError> {
        let (mt, nt, kc, g) = (self.mt, self.nt, self.kchunk, self.groups);
        let k = kc * MK;
        let n = g * nt * MN; // full N of this slab
        assert_eq!(a.len(), mt * MR * k, "A shape");
        assert_eq!(w_int4.len(), k * n, "W shape");
        assert_eq!(c.len(), mt * MR * n, "C shape");

        self.load_a(a); // row-major A block -> a_buf (kernel tensor-streams the tiling)
                        // W -> per-group tile-major + int4 pack. Group gi owns the N-slab
                        // [gi*NT*MN, (gi+1)*NT*MN); its w_buf region starts at gi*(NT*KCHUNK tiles).
        {
            let s = self.w_buf.as_mut_slice();
            s.fill(0);
            let tiles_per_group = nt * kc; // 128-B tiles
            for gi in 0..g {
                let wbase = gi * tiles_per_group * (MK * MN); // int4 elements
                let ncol0 = gi * nt * MN;
                for nti in 0..nt {
                    for ki in 0..kc {
                        for kk in 0..MK {
                            for nn in 0..MN {
                                let v = (w_int4[(ki * MK + kk) * n + ncol0 + nti * MN + nn] & 0xf)
                                    as u8;
                                let idx = wbase + (nti * kc + ki) * (MK * MN) + kk * MN + nn;
                                s[idx / 2] |= if idx % 2 == 0 { v } else { v << 4 };
                            }
                        }
                    }
                }
            }
        }

        self.kernel
            .dispatch(&[&self.a_buf, &self.w_buf, &self.c_buf[0]])?;
        self.unpack_c_block(0, c, 0, n); // c is exactly the bm×bn (=n-wide) block
        Ok(())
    }

    // Size of the W SHMEM buffer (one dispatch's marshaled weights), in bytes.
    fn wbuf_len(&self) -> usize {
        self.groups * self.nt * self.kchunk * MK * MN / 2
    }

    /// Marshal one `(KCHUNK·MK) × (groups·NT·MN)` int4 W slab into `out` (a
    /// `wbuf_len()`-sized tile-major, int4-packed buffer). Pure CPU; no dispatch.
    fn pack_w_slab(&self, w_int4: &[i8], out: &mut [u8]) {
        let (nt, kc, g) = (self.nt, self.kchunk, self.groups);
        let n = g * nt * MN;
        out.fill(0);
        for gi in 0..g {
            let wbase = gi * (nt * kc) * (MK * MN);
            let ncol0 = gi * nt * MN;
            for nti in 0..nt {
                for ki in 0..kc {
                    for kk in 0..MK {
                        for nn in 0..MN {
                            let v =
                                (w_int4[(ki * MK + kk) * n + ncol0 + nti * MN + nn] & 0xf) as u8;
                            let idx = wbase + (nti * kc + ki) * (MK * MN) + kk * MN + nn;
                            out[idx / 2] |= if idx % 2 == 0 { v } else { v << 4 };
                        }
                    }
                }
            }
        }
    }

    // Copy c_buf[slot] into `dst` at `dst[base + row*row_stride + col]`. The R6-TS kernel
    // writes each group's block ROW-MAJOR (its C tensor stream de-tiles in-core), so this
    // is a straight per-group block copy — no reshuffle. `base`/`row_stride` place the
    // (MT·MR)×(groups·NT·MN) result anywhere in a larger output (e.g. the (mo,no) block of
    // the full GEMM), so no intermediate copy is needed on the single-K-chunk path.
    fn unpack_c_block(&self, slot: usize, dst: &mut [i32], base: usize, row_stride: usize) {
        let (mt, nt, g) = (self.mt, self.nt, self.groups);
        let bw = nt * MN; // group width (cols per group)
        let bh = mt * MR; // rows
        let out: &[i32] = unsafe {
            std::slice::from_raw_parts(
                self.c_buf[slot].as_slice().as_ptr() as *const i32,
                g * bh * bw,
            )
        };
        for gi in 0..g {
            let cbase = gi * bh * bw;
            let gcol0 = gi * bw; // this group's column offset within the block
            for row in 0..bh {
                let src = cbase + row * bw;
                let d = base + row * row_stride + gcol0;
                dst[d..d + bw].copy_from_slice(&out[src..src + bw]);
            }
        }
    }

    /// Pre-marshal a full `K×N` weight matrix ONCE into the tile-major, int4-packed
    /// form the kernel consumes — the slow bit-packing that must NOT happen per
    /// inference (weights are static). The result is indexed by (K-chunk, N-slab):
    /// block `(ko, no)` at `(ko*n_slabs + no) * wbuf_len()`. Pass to [`Self::run_packed`].
    pub fn prepack_weights(&self, k: usize, n: usize, w_int4: &[i8]) -> Vec<u8> {
        let (bn, bk) = (self.block_n(), self.block_k());
        assert!(k % bk == 0 && n % bn == 0, "K/N must tile evenly");
        let (nks, nns, wl) = (k / bk, n / bn, self.wbuf_len());
        let mut packed = vec![0u8; nks * nns * wl];
        let mut w_sub = vec![0i8; bk * bn];
        for ko_i in 0..nks {
            for no_i in 0..nns {
                for i in 0..bk {
                    let src = (ko_i * bk + i) * n + no_i * bn;
                    w_sub[i * bn..(i + 1) * bn].copy_from_slice(&w_int4[src..src + bn]);
                }
                let off = (ko_i * nns + no_i) * wl;
                self.pack_w_slab(&w_sub, &mut packed[off..off + wl]);
            }
        }
        packed
    }

    /// Full GEMM using pre-marshaled weights (from [`Self::prepack_weights`]): the
    /// per-dispatch weight cost is a `memcpy`, not a re-pack — the whole point of the
    /// hot path. Only `a` (activations) is marshaled per inference.
    pub fn run_packed(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[i8],
        packed_w: &[u8],
        c: &mut [i32],
    ) -> Result<(), XdnaError> {
        let (bm, bn, bk) = (self.block_m(), self.block_n(), self.block_k());
        assert!(
            m % bm == 0 && n % bn == 0 && k % bk == 0,
            "M/N/K must tile evenly"
        );
        let (nms, nns, nks) = (m / bm, n / bn, k / bk);
        let wl = self.wbuf_len();
        let ndisp = nms * nns * nks;
        if ndisp == 0 {
            return Ok(());
        }

        let mut a_sub = vec![0i8; bm * bk];
        let mut c_blk = vec![0i32; bm * bn]; // K-accumulation scratch (nks > 1 only)
        let mut c_acc = vec![0i32; bm * bn];

        // Flat dispatch order: mo outer, no, ko inner. coord(i) -> (mo_i, no_i, ko_i).
        let coord = |i: usize| {
            let per_m = nns * nks;
            let (mo_i, r) = (i / per_m, i % per_m);
            (mo_i, r / nks, r % nks)
        };

        // Software-pipelined: submit dispatch i (into c_buf[i%2]) BEFORE reading dispatch
        // i-1 back, so the host read-back overlaps dispatch i's execution on the NPU.
        // a_buf/w_buf are single-buffered but only refilled after wait(i-1), so the
        // in-flight dispatch never sees torn inputs; c_buf is double-buffered so dispatch
        // i's output can't clobber i-1's before we read it. W copy is skipped when the
        // slab is unchanged from the previous dispatch (same (ko,no)).
        let mut prev: Option<(u64, usize)> = None; // (timeline seq, dispatch index)
        let mut last_w_off = usize::MAX;
        for i in 0..ndisp {
            if let Some((seq, _)) = prev {
                self.kernel.wait(seq)?; // i-1 done: a_buf/w_buf free, its C readable
            }
            let (mo_i, no_i, ko_i) = coord(i);
            for r in 0..bm {
                let src = (mo_i * bm + r) * k + ko_i * bk;
                a_sub[r * bk..(r + 1) * bk].copy_from_slice(&a[src..src + bk]);
            }
            self.load_a(&a_sub);
            let off = (ko_i * nns + no_i) * wl;
            if off != last_w_off {
                self.w_buf
                    .as_mut_slice()
                    .copy_from_slice(&packed_w[off..off + wl]);
                last_w_off = off;
            }
            let seq = self
                .kernel
                .submit(&[&self.a_buf, &self.w_buf, &self.c_buf[i % 2]])?;
            if let Some((_, pi)) = prev {
                let (pm, pn, pk) = coord(pi);
                self.readback(pi % 2, pm, pn, pk, nks, &mut c_acc, &mut c_blk, n, c);
            }
            prev = Some((seq, i));
        }
        if let Some((seq, pi)) = prev {
            self.kernel.wait(seq)?;
            let (pm, pn, pk) = coord(pi);
            self.readback(pi % 2, pm, pn, pk, nks, &mut c_acc, &mut c_blk, n, c);
        }
        Ok(())
    }

    /// Read one completed dispatch's C back. Single K-chunk (`nks == 1`): unpack straight
    /// into the output block, no host accumulation. Otherwise accumulate this K-chunk into
    /// `c_acc`, flushing the finished `(mo,no)` block to `c` on the last chunk.
    #[allow(clippy::too_many_arguments)]
    fn readback(
        &self,
        slot: usize,
        mo_i: usize,
        no_i: usize,
        ko_i: usize,
        nks: usize,
        c_acc: &mut [i32],
        c_blk: &mut [i32],
        n: usize,
        c: &mut [i32],
    ) {
        let (bm, bn) = (self.block_m(), self.block_n());
        if nks == 1 {
            self.unpack_c_block(slot, c, mo_i * bm * n + no_i * bn, n);
            return;
        }
        if ko_i == 0 {
            c_acc.fill(0);
        }
        self.unpack_c_block(slot, c_blk, 0, bn);
        for (acc, &v) in c_acc.iter_mut().zip(c_blk.iter()) {
            *acc += v;
        }
        if ko_i == nks - 1 {
            for r in 0..bm {
                let dst = (mo_i * bm + r) * n + no_i * bn;
                c[dst..dst + bn].copy_from_slice(&c_acc[r * bn..(r + 1) * bn]);
            }
        }
    }

    // Copy `a` `(MT·MR) × (KCHUNK·MK)` int8 row-major into a_buf. The R6-TS kernel's A
    // tensor stream tiles it in-core, so no CPU reshuffle — just an int8->u8 block copy.
    fn load_a(&mut self, a: &[i8]) {
        let s = self.a_buf.as_mut_slice();
        debug_assert_eq!(s.len(), a.len());
        for (d, &v) in s.iter_mut().zip(a.iter()) {
            *d = v as u8;
        }
    }
}
