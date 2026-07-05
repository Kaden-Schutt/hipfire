//! End-to-end NpuGemm throughput: times the FULL run_packed path (per-inference A copy +
//! per-dispatch W memcpy + dispatch + C copy) on a realistic prefill shape, and checks the
//! result vs a CPU W4A8 reference. This is the number that matters for offload viability —
//! the R6-TS kernel computes at ~20.7 TOPS, but the deliverable rate is gated on the
//! surrounding host cost. Before tensor streams this shape ran at ~0.02 TOPS (CPU tile
//! marshaling dominated); with row-major A/C it should be far higher.
//!
//! Build the array xclbin with the TS kernel:
//!   R6_KERNEL_SRC=<r6>/r6_gemm_ts.cc R6_OUT_TAG=r6ts <r6>/r6_cache.sh MT 4 KCHUNK COLS NB
//! Run: cargo run --release -p hipfire-xdna --example npu_gemm_e2e -- <dir> MT KCHUNK GROUPS [M K N iters]

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuGemm;
        use std::time::Instant;

        let a: Vec<String> = std::env::args().collect();
        let dir = &a[1];
        let p = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
        let (mt, kc, g) = (p(2, 24), p(3, 8), p(4, 32));
        let (m, k, n) = (p(5, 768), p(6, 512), p(7, 4096));
        let iters = p(8, 20) as u32;

        let x = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
        let i = std::fs::read(format!("{dir}/insts.bin")).expect("insts");
        let mut gemm = NpuGemm::load(&x, &i, mt, 4, kc, g).expect("load");
        let (bm, bn, bk) = (gemm.block_m(), gemm.block_n(), gemm.block_k());
        assert!(
            m % bm == 0 && n % bn == 0 && k % bk == 0,
            "shape must tile (block {bm}x{bn}x{bk})"
        );
        let dispatches = (m / bm) * (n / bn) * (k / bk);

        let rnd = |i: usize| -> i8 {
            let s = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(0x9e37_79b9);
            (((s >> 13) & 0xf) as i32 - 8) as i8
        };
        let av: Vec<i8> = (0..m * k).map(rnd).collect();
        let wv: Vec<i8> = (0..k * n).map(|i| rnd(7_777_777 + i)).collect();
        let mut cv = vec![0i32; m * n];

        // Pre-pack weights once (static cost), then time the per-inference path.
        let tpp = Instant::now();
        let packed = gemm.prepack_weights(k, n, &wv);
        let pp_ms = tpp.elapsed().as_secs_f64() * 1e3;

        gemm.run_packed(m, k, n, &av, &packed, &mut cv)
            .expect("run"); // warm + correctness

        // Correctness vs CPU on rows 0, m/2, m-1 (covers early/mid/late M-blocks so a
        // pipelining hazard across blocks would show).
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
            eprintln!(
                "CORRECTNESS FAIL: {bad} mismatches across rows 0/{}/{}",
                m / 2,
                m - 1
            );
            std::process::exit(4);
        }

        let t = Instant::now();
        for _ in 0..iters {
            gemm.run_packed(m, k, n, &av, &packed, &mut cv)
                .expect("run");
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let tops = 2.0 * m as f64 * k as f64 * n as f64 / (ms * 1e-3) / 1e12;
        println!(
            "M={m} K={k} N={n}  block {bm}x{bn}x{bk}  {dispatches} dispatches/run  (prepack {pp_ms:.1} ms once)"
        );
        println!("run_packed: {ms:.2} ms/run  =>  {tops:.3} TOPS end-to-end  (row 0 correct)");
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
