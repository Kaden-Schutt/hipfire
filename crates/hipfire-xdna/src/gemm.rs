//! W5/wire-in step 2 — `NpuGemm`: a W4A8 GEMM primitive over the R6 kernel. Turns a
//! standard row-major GEMM into R6 dispatches by marshaling A/W into the kernel's
//! tile-major SHMEM layout (all tiles row-major; validated 0/256 by `r6_verify`).
//!
//! A block is one R6 dispatch computing `(MT·4) × (NT·16)` output over `KCHUNK·16`
//! contraction. [`NpuGemm::run_block`] handles that; larger GEMMs tile M/N/K over it
//! (M/N = independent blocks, K = accumulate). Linux-only (amdxdna).
#![cfg(target_os = "linux")]

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const MR: usize = 4; // mmul M
const MK: usize = 16; // mmul K
const MN: usize = 16; // mmul N

/// A loaded R6 kernel specialized for (MT, NT, KCHUNK) with reusable SHMEM buffers.
pub struct NpuGemm {
    kernel: NpuKernel,
    mt: usize,
    nt: usize,
    kchunk: usize,
    a_buf: DeviceBuffer, // MT*KCHUNK tiles of MR*MK int8
    w_buf: DeviceBuffer, // NT*KCHUNK tiles of MK*MN int4 (2/byte)
    c_buf: DeviceBuffer, // MT*NT tiles of MR*MN int32
}

impl NpuGemm {
    /// Block dims this kernel computes per dispatch.
    pub fn block_m(&self) -> usize {
        self.mt * MR
    }
    pub fn block_n(&self) -> usize {
        self.nt * MN
    }
    pub fn block_k(&self) -> usize {
        self.kchunk * MK
    }

    /// Load an R6 xclbin built for (mt, nt, kchunk) and allocate its arg buffers.
    pub fn load(
        xclbin: &[u8],
        insts: &[u8],
        mt: usize,
        nt: usize,
        kchunk: usize,
    ) -> Result<Self, XdnaError> {
        let kernel = NpuKernel::load(xclbin, insts)?;
        let a_buf = kernel.alloc_arg(mt * kchunk * MR * MK)?;
        let w_buf = kernel.alloc_arg(nt * kchunk * MK * MN / 2)?;
        let c_buf = kernel.alloc_arg(mt * nt * MR * MN * 4)?;
        Ok(Self {
            kernel,
            mt,
            nt,
            kchunk,
            a_buf,
            w_buf,
            c_buf,
        })
    }

    /// Full GEMM `C[M,N] = A[M,K] · W[K,N]` (W4A8) by tiling over R6 blocks: M and N
    /// split into independent blocks, K accumulated. `a` row-major `M×K` int8, `w_int4`
    /// row-major `K×N` int4 values (`-8..=7`, one per byte), `c` row-major `M×N` int32.
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
            "M/N/K must tile evenly"
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
                    self.run_block(&a_sub, &w_sub, &mut c_blk)?;
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

    /// One R6 block: `a` is row-major `(MT·MR) × (KCHUNK·MK)` int8; `w_int4` is
    /// row-major `(KCHUNK·MK) × (NT·MN)` with each element an int4 value in `-8..=7`
    /// (unpacked, one per byte); `c` receives row-major `(MT·MR) × (NT·MN)` int32.
    pub fn run_block(&mut self, a: &[i8], w_int4: &[i8], c: &mut [i32]) -> Result<(), XdnaError> {
        let (mt, nt, kc) = (self.mt, self.nt, self.kchunk);
        let k = kc * MK;
        let n = nt * MN;
        assert_eq!(a.len(), mt * MR * k, "A shape");
        assert_eq!(w_int4.len(), k * n, "W shape");
        assert_eq!(c.len(), mt * MR * n, "C shape");

        // A -> tile-major SHMEM: a_buf[(mt_i*KCHUNK+ki)*(MR*MK) + m*MK + kk].
        {
            let s = self.a_buf.as_mut_slice();
            for mti in 0..mt {
                for ki in 0..kc {
                    for m in 0..MR {
                        for kk in 0..MK {
                            s[(mti * kc + ki) * (MR * MK) + m * MK + kk] =
                                a[(mti * MR + m) * k + ki * MK + kk] as u8;
                        }
                    }
                }
            }
        }
        // W -> tile-major + int4 pack: w_buf int4 at (nt_i*KCHUNK+ki)*(MK*MN) + kk*MN + nn.
        {
            let s = self.w_buf.as_mut_slice();
            s.fill(0);
            for nti in 0..nt {
                for ki in 0..kc {
                    for kk in 0..MK {
                        for nn in 0..MN {
                            let v = (w_int4[(ki * MK + kk) * n + nti * MN + nn] & 0xf) as u8;
                            let idx = (nti * kc + ki) * (MK * MN) + kk * MN + nn;
                            s[idx / 2] |= if idx % 2 == 0 { v } else { v << 4 };
                        }
                    }
                }
            }
        }

        self.kernel
            .dispatch(&[&self.a_buf, &self.w_buf, &self.c_buf])?;

        // C tile-major -> row-major: c_buf[(mt_i*NT+nt_i)*(MR*MN) + m*MN + nn].
        let out: &[i32] = unsafe {
            std::slice::from_raw_parts(
                self.c_buf.as_slice().as_ptr() as *const i32,
                mt * nt * MR * MN,
            )
        };
        for mti in 0..mt {
            for nti in 0..nt {
                for m in 0..MR {
                    for nn in 0..MN {
                        c[(mti * MR + m) * n + nti * MN + nn] =
                            out[(mti * nt + nti) * (MR * MN) + m * MN + nn];
                    }
                }
            }
        }
        Ok(())
    }
}
