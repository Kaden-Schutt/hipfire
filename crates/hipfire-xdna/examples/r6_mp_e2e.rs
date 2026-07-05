//! End-to-end throughput of the M-parallel W-broadcast R6 array (r6_gen_mp.py): the full
//! M×K×N W4A8 GEMM as ceil(M / (COLS·MT·MR)) dispatches, each computing COLS distinct
//! M-blocks over full N against ONE broadcast W (packed + loaded ONCE). Compares to the
//! N-parallel npu_gemm_e2e: M-parallel trades 24 small dispatches for ~3 big ones (less
//! dispatch latency + W read once) against the memtile broadcast-sync feed cost.
//! Requires K == KCHUNK·MK (single K-chunk) and N == NB·NT·MN.
//!
//! Build: R6_KERNEL_SRC=<r6>/r6_gemm_ts.cc R6_GEN=r6_gen_mp.py R6_OUT_TAG=r6mp <r6>/r6_cache.sh MT 4 KCHUNK COLS NB
//! Run:   cargo run --release -p hipfire-xdna --example r6_mp_e2e -- <dir> COLS MT KCHUNK NB [M iters]

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuKernel;
        use std::time::Instant;
        let a: Vec<String> = std::env::args().collect();
        let dir = &a[1];
        let p = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
        let (cols, mt, kc, nb) = (p(2, 8), p(3, 8), p(4, 32), p(5, 64));
        const NT: usize = 4;
        const MR: usize = 4;
        const MK: usize = 16;
        const MN: usize = 16;
        let rows_per = cols * mt * MR; // M rows per dispatch
        let m = p(6, 768);
        let iters = p(7, 20) as u32;
        let k = kc * MK;
        let n = nb * NT * MN;
        assert!(
            m % rows_per == 0,
            "M must be a multiple of COLS*MT*MR={rows_per}"
        );
        let ndisp = m / rows_per;
        let aw = mt * kc * MR * MK;
        let ww = NT * kc * MK * MN / 2;
        let cw = mt * NT * MR * MN;

        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
        let insts = std::fs::read(format!("{dir}/insts.bin")).expect("insts");
        let kern = NpuKernel::load(&xclbin, &insts).expect("load");
        let mut abuf = kern.alloc_arg(cols * aw).expect("A");
        let mut wbuf = kern.alloc_arg(nb * ww).expect("W");
        let cbuf = kern.alloc_arg(cols * nb * cw * 4).expect("C");

        let rnd = |i: usize| -> i8 {
            let s = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(0x9e37_79b9);
            (((s >> 13) & 0xf) as i32 - 8) as i8
        };
        let av: Vec<i8> = (0..m * k).map(rnd).collect();
        let wv: Vec<i8> = (0..k * n).map(|i| rnd(7_777_777 + i)).collect();
        let mut cv = vec![0i32; m * n];

        // W packed ONCE into the broadcast slab layout (static — the whole point).
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
        }

        // One full GEMM: for each M-tile, load COLS M-blocks of A, dispatch, read C back.
        let run = |kern: &NpuKernel, abuf: &mut hipfire_xdna::DeviceBuffer, cv: &mut [i32]| {
            for d in 0..ndisp {
                let row0 = d * rows_per;
                {
                    let s = abuf.as_mut_slice();
                    for c in 0..cols {
                        for r in 0..mt * MR {
                            let src = (row0 + c * mt * MR + r) * k;
                            for kk in 0..k {
                                s[c * aw + r * k + kk] = av[src + kk] as u8;
                            }
                        }
                    }
                }
                kern.dispatch(&[abuf, &wbuf, &cbuf]).expect("dispatch");
                let out: &[i32] = unsafe {
                    std::slice::from_raw_parts(
                        cbuf.as_slice().as_ptr() as *const i32,
                        cols * nb * cw,
                    )
                };
                for c in 0..cols {
                    for j in 0..nb {
                        for r in 0..mt * MR {
                            let mg = row0 + c * mt * MR + r;
                            let base = (c * nb + j) * cw + r * (NT * MN);
                            let dst = mg * n + j * NT * MN;
                            cv[dst..dst + NT * MN].copy_from_slice(&out[base..base + NT * MN]);
                        }
                    }
                }
            }
        };

        run(&kern, &mut abuf, &mut cv); // warm + correctness
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
            run(&kern, &mut abuf, &mut cv);
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let tops = 2.0 * m as f64 * k as f64 * n as f64 / (ms * 1e-3) / 1e12;
        println!(
            "M={m} K={k} N={n}  COLS={cols} MT={mt} KCHUNK={kc} NB={nb}  {ndisp} dispatches/run"
        );
        println!("M-parallel e2e: {ms:.2} ms/run  =>  {tops:.3} TOPS  (rows 0/mid/last correct)");
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
