//! Validate the productionized NpuGemmMp primitive: prepack W once, load it, run a full
//! row-major W4A8 GEMM tiled over M-parallel dispatches, compare to a CPU reference. Point
//! it at an M-parallel xclbin (r6_gen_mp.py, ROUNDS=1) built for (COLS, MT, KCHUNK, NB).
//!
//! Build: R6_KERNEL_SRC=<r6>/r6_gemm_ts.cc R6_GEN=r6_gen_mp.py R6_OUT_TAG=r6mp <r6>/r6_cache.sh MT 4 KCHUNK COLS NB
//! Run:   cargo run -p hipfire-xdna --example npu_gemm_mp_verify -- <dir>  (config from dir name)

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuGemmMp;
        let a: Vec<String> = std::env::args().collect();
        let dir = &a[1];
        // Self-describing: config (COLS/MT/KCHUNK/NB) is parsed from the cache dir name.
        let mut g = NpuGemmMp::load_cached(dir).unwrap();
        let (k, n, rows_per) = (g.k(), g.n(), g.rows_per_dispatch());

        let rnd = |i: usize| -> i8 {
            let s = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(0x9e37_79b9);
            (((s >> 13) & 0xf) as i32 - 8) as i8
        };
        let wv: Vec<i8> = (0..k * n).map(|i| rnd(7_777_777 + i)).collect();
        g.load_weights(&g.prepack_weights(k, n, &wv));

        // Two sizes: one dispatch, and a 3-tile M (exercises the M-loop).
        for &m in &[rows_per, rows_per * 3] {
            let av: Vec<i8> = (0..m * k).map(rnd).collect();
            let mut cv = vec![0i32; m * n];
            g.run(m, k, n, &av, &mut cv).unwrap();
            // CPU reference on rows 0, m/2, m-1.
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
            println!(
                "M={m} K={k} N={n} ({} dispatches): {bad} mismatches",
                m / rows_per
            );
            if bad != 0 {
                eprintln!("NpuGemmMp WRONG");
                std::process::exit(4);
            }
        }
        println!("NpuGemmMp W4A8 GEMM CORRECT — M-parallel W-broadcast, row-major, weights broadcast once");
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
