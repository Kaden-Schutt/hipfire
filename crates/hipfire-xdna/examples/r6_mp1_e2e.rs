//! Whole-GEMM-in-ONE-dispatch M-parallel W-broadcast (r6_gen_mp.py ROUNDS>1). One dispatch
//! streams COLS*ROUNDS M-blocks continuously (core c owns the contiguous M-chunk [c*ROUNDS,
//! c*ROUNDS+ROUNDS)); the array never stalls on host A-reload/C-read mid-GEMM, there is one
//! exec + one C read-back (no per-dispatch latency, no pipelined-readback coherence issue).
//! Layouts: A block g=c*ROUNDS+r at g*AW; W broadcast (NB slabs) once; C block (g,j) at
//! (g*NB+j)*CW. Requires K==KCHUNK*MK, N==NB*NT*MN, M==COLS*ROUNDS*MT*MR.
//!
//! Build: R6_KERNEL_SRC=<r6>/r6_gemm_ts.cc R6_GEN=r6_gen_mp.py R6_OUT_TAG=r6mp R6_ROUNDS=R <r6>/r6_cache.sh MT 4 KCHUNK COLS NB
//! Run:   cargo run --release -p hipfire-xdna --example r6_mp1_e2e -- <dir> COLS MT KCHUNK NB ROUNDS [iters]

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuKernel;
        use std::time::Instant;
        let a: Vec<String> = std::env::args().collect();
        let dir = &a[1];
        let p = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
        let (cols, mt, kc, nb, rounds) = (p(2, 8), p(3, 8), p(4, 32), p(5, 64), p(6, 3));
        let iters = p(7, 20) as u32;
        const NT: usize = 4;
        const MR: usize = 4;
        const MK: usize = 16;
        const MN: usize = 16;
        let nblk = cols * rounds; // total M-blocks
        let m = nblk * mt * MR;
        let k = kc * MK;
        let n = nb * NT * MN;
        let aw = mt * kc * MR * MK;
        let ww = NT * kc * MK * MN / 2;
        let cw = mt * NT * MR * MN;

        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
        let insts = std::fs::read(format!("{dir}/insts.bin")).expect("insts");
        let kern = NpuKernel::load(&xclbin, &insts).expect("load");
        let mut abuf = kern.alloc_arg(nblk * aw).expect("A");
        let mut wbuf = kern.alloc_arg(rounds * nb * ww).expect("W"); // W replicated ROUNDS x
        let cbuf = kern.alloc_arg(nblk * nb * cw * 4).expect("C");

        let rnd = |i: usize| -> i8 {
            let s = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(0x9e37_79b9);
            (((s >> 13) & 0xf) as i32 - 8) as i8
        };
        let av: Vec<i8> = (0..m * k).map(rnd).collect();
        let wv: Vec<i8> = (0..k * n).map(|i| rnd(7_777_777 + i)).collect();
        let mut cv = vec![0i32; m * n];

        // W packed into the broadcast slab layout, then replicated ROUNDS times in DRAM
        // (the objectfifo can't replay a stream, so each round reads its own copy).
        {
            let s = wbuf.as_mut_slice();
            s.fill(0);
            for j in 0..nb {
                for nt in 0..NT {
                    for ki in 0..kc {
                        for kk in 0..MK {
                            for nn in 0..MN {
                                let kg = ki * MK + kk;
                                let ng = j * NT * MN + nt * MN + nn;
                                let idx = (nt * kc + ki) * (MK * MN) + kk * MN + nn;
                                let u = (wv[kg * n + ng] & 0xf) as u8;
                                s[j * ww + idx / 2] |= if idx % 2 == 0 { u } else { u << 4 };
                            }
                        }
                    }
                }
            }
            let slab = nb * ww;
            for r in 1..rounds {
                s.copy_within(0..slab, r * slab);
            }
        }

        // A: block g at g*aw, row-major (mt*MR)x(kc*MK). Block g covers M-rows [g*mt*MR,+).
        let fill_a = |abuf: &mut hipfire_xdna::DeviceBuffer| {
            let s = abuf.as_mut_slice();
            for g in 0..nblk {
                for lr in 0..mt * MR {
                    let src = (g * mt * MR + lr) * k;
                    for kk in 0..k {
                        s[g * aw + lr * k + kk] = av[src + kk] as u8;
                    }
                }
            }
        };
        // C: block (g,j) at (g*nb+j)*cw, row-major (mt*MR)x(NT*MN) -> global rows [g*mt*MR,+),
        // N-cols [j*NT*MN,+).
        let read_c = |cbuf: &hipfire_xdna::DeviceBuffer, cv: &mut [i32]| {
            let out: &[i32] = unsafe {
                std::slice::from_raw_parts(cbuf.as_slice().as_ptr() as *const i32, nblk * nb * cw)
            };
            for g in 0..nblk {
                for j in 0..nb {
                    for lr in 0..mt * MR {
                        let base = (g * nb + j) * cw + lr * (NT * MN);
                        let dst = (g * mt * MR + lr) * n + j * NT * MN;
                        cv[dst..dst + NT * MN].copy_from_slice(&out[base..base + NT * MN]);
                    }
                }
            }
        };

        fill_a(&mut abuf);
        kern.dispatch(&[&abuf, &wbuf, &cbuf]).expect("dispatch");
        read_c(&cbuf, &mut cv);
        // Per-M-block correctness (one representative row per block) to localize failures.
        let mut bad_blocks: Vec<usize> = Vec::new();
        for g in 0..nblk {
            let mm = g * mt * MR; // first row of block g
            let mut brow = 0usize;
            for nn in 0..n {
                let acc: i32 = (0..k)
                    .map(|kk| av[mm * k + kk] as i32 * wv[kk * n + nn] as i32)
                    .sum();
                if cv[mm * n + nn] != acc {
                    brow += 1;
                }
            }
            if brow != 0 {
                bad_blocks.push(g);
            }
        }
        if !bad_blocks.is_empty() {
            eprintln!(
                "CORRECTNESS FAIL: {}/{nblk} M-blocks wrong: {:?} (block g -> core {}, round {})",
                bad_blocks.len(),
                bad_blocks,
                bad_blocks[0] / rounds,
                bad_blocks[0] % rounds
            );
            std::process::exit(4);
        }

        let t = Instant::now();
        for _ in 0..iters {
            fill_a(&mut abuf);
            kern.dispatch(&[&abuf, &wbuf, &cbuf]).expect("dispatch");
            read_c(&cbuf, &mut cv);
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let tops = 2.0 * m as f64 * k as f64 * n as f64 / (ms * 1e-3) / 1e12;
        println!("M={m} K={k} N={n}  COLS={cols} MT={mt} KCHUNK={kc} NB={nb} ROUNDS={rounds}  1 dispatch/run");
        println!(
            "whole-GEMM e2e: {ms:.2} ms/run  =>  {tops:.3} TOPS  (all {nblk} M-blocks correct)"
        );
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
