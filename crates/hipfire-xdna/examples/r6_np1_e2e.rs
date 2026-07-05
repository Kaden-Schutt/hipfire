//! Whole-GEMM-in-ONE-dispatch N-parallel (r6_gen.py ROUNDS>1): all COLS cores share the
//! same ROUNDS M-blocks (A broadcast), each owning a contiguous N-chunk of NB slabs
//! (independent W, re-streamed per M-block). One dispatch streams ROUNDS M-blocks x full N.
//! This probes whether the high-MT N-parallel tile (up to 20.7 TOPS compute) can keep that
//! rate end-to-end when the per-dispatch host overhead is removed. Layouts:
//!   A: ROUNDS blocks, block r at r*AW, row-major (MT*MR)x(KCHUNK*MK) -> rows [r*MT*MR,+).
//!   W: core c at c*ROUNDS*NB*WW; round r at +r*NB*WW, slab s at +s*WW (same W each round,
//!      tile-major, covering global N-slab t=c*NB+s).
//!   C: core c round r slab s at c*ROUNDS*NB*CW + r*NB*CW + s*CW -> rows [r*MT*MR,+),
//!      cols [(c*NB+s)*NT*MN,+).
//! M = ROUNDS*MT*MR, K = KCHUNK*MK, N = COLS*NB*NT*MN.
//!
//! Build: R6_KERNEL_SRC=<r6>/r6_gemm_ts.cc R6_OUT_TAG=r6ts R6_ROUNDS=R <r6>/r6_cache.sh MT 4 KCHUNK COLS NB
//! Run:   cargo run --release -p hipfire-xdna --example r6_np1_e2e -- <dir> COLS MT KCHUNK NB ROUNDS [iters]

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuKernel;
        use std::time::Instant;
        let a: Vec<String> = std::env::args().collect();
        let dir = &a[1];
        let p = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
        let (cols, mt, kc, nb, rounds) = (p(2, 8), p(3, 24), p(4, 8), p(5, 8), p(6, 8));
        let iters = p(7, 20) as u32;
        const NT: usize = 4;
        const MR: usize = 4;
        const MK: usize = 16;
        const MN: usize = 16;
        let m = rounds * mt * MR;
        let k = kc * MK;
        let n = cols * nb * NT * MN;
        let aw = mt * kc * MR * MK;
        let ww = NT * kc * MK * MN / 2;
        let cw = mt * NT * MR * MN;

        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
        let insts = std::fs::read(format!("{dir}/insts.bin")).expect("insts");
        let kern = NpuKernel::load(&xclbin, &insts).expect("load");
        let mut abuf = kern.alloc_arg(rounds * aw).expect("A");
        let mut wbuf = kern.alloc_arg(cols * rounds * nb * ww).expect("W");
        let cbuf = kern.alloc_arg(cols * rounds * nb * cw * 4).expect("C");

        let rnd = |i: usize| -> i8 {
            let s = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(0x9e37_79b9);
            (((s >> 13) & 0xf) as i32 - 8) as i8
        };
        let av: Vec<i8> = (0..m * k).map(rnd).collect();
        let wv: Vec<i8> = (0..k * n).map(|i| rnd(7_777_777 + i)).collect();
        let mut cv = vec![0i32; m * n];

        // W: core c owns global N-slabs [c*nb, (c+1)*nb). Pack each slab tile-major, then it
        // is the same for every round -> replicate ROUNDS times in the core's region.
        {
            let s = wbuf.as_mut_slice();
            s.fill(0);
            for c in 0..cols {
                for sl in 0..nb {
                    let t = c * nb + sl; // global N-slab
                    let wbase = (c * rounds * nb + sl) * ww; // round 0, slab sl
                    for nt in 0..NT {
                        for ki in 0..kc {
                            for kk in 0..MK {
                                for nn in 0..MN {
                                    let kg = ki * MK + kk;
                                    let ng = t * NT * MN + nt * MN + nn;
                                    let idx = (nt * kc + ki) * (MK * MN) + kk * MN + nn;
                                    let u = (wv[kg * n + ng] & 0xf) as u8;
                                    s[wbase + idx / 2] |= if idx % 2 == 0 { u } else { u << 4 };
                                }
                            }
                        }
                    }
                }
                // replicate this core's nb slabs across the remaining rounds
                let per = nb * ww;
                for r in 1..rounds {
                    let base = c * rounds * nb * ww;
                    s.copy_within(base..base + per, base + r * per);
                }
            }
        }

        // A: block r at r*aw, row-major (mt*MR)x(kc*MK) -> global rows [r*mt*MR,+).
        let fill_a = |abuf: &mut hipfire_xdna::DeviceBuffer| {
            let s = abuf.as_mut_slice();
            for r in 0..rounds {
                for lr in 0..mt * MR {
                    let src = (r * mt * MR + lr) * k;
                    for kk in 0..k {
                        s[r * aw + lr * k + kk] = av[src + kk] as u8;
                    }
                }
            }
        };
        let read_c = |cbuf: &hipfire_xdna::DeviceBuffer, cv: &mut [i32]| {
            let out: &[i32] = unsafe {
                std::slice::from_raw_parts(
                    cbuf.as_slice().as_ptr() as *const i32,
                    cols * rounds * nb * cw,
                )
            };
            for c in 0..cols {
                for r in 0..rounds {
                    for sl in 0..nb {
                        let t = c * nb + sl;
                        for lr in 0..mt * MR {
                            let base = ((c * rounds + r) * nb + sl) * cw + lr * (NT * MN);
                            let dst = (r * mt * MR + lr) * n + t * NT * MN;
                            cv[dst..dst + NT * MN].copy_from_slice(&out[base..base + NT * MN]);
                        }
                    }
                }
            }
        };

        fill_a(&mut abuf);
        kern.dispatch(&[&abuf, &wbuf, &cbuf]).expect("dispatch");
        read_c(&cbuf, &mut cv);
        let mut bad = 0usize;
        for &mm in &[0usize, m / 2, m - 1] {
            for nn in 0..n {
                let acc: i32 = (0..k)
                    .map(|kk| av[mm * k + kk] as i32 * wv[kk * n + nn] as i32)
                    .sum();
                if cv[mm * n + nn] != acc {
                    bad += 1;
                }
            }
        }
        if bad != 0 {
            eprintln!("CORRECTNESS FAIL: {bad} across rows 0/{}/{}", m / 2, m - 1);
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
        println!("N-parallel whole-GEMM e2e: {ms:.2} ms/run  =>  {tops:.3} TOPS  (rows 0/mid/last correct)");
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
