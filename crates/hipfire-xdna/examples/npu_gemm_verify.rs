//! Validate NpuGemm vs a CPU W4A8 reference for any (MT,NT,KCHUNK,groups) config:
//! one dispatch block, and a tiled 2×(M,N) / 2×K shape (exercises K-accumulation and
//! multi-block tiling).
//!
//! Run: cargo run -p hipfire-xdna --example npu_gemm_verify -- <dir> [MT NT KCHUNK GROUPS]
fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuGemm;
        let a: Vec<String> = std::env::args().collect();
        let dir = &a[1];
        let p = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
        let (mt, nt, kc, g) = (p(2, 16), p(3, 4), p(4, 16), p(5, 1));
        let x = std::fs::read(format!("{dir}/final.xclbin")).unwrap();
        let i = std::fs::read(format!("{dir}/insts.bin")).unwrap();
        let mut gemm = NpuGemm::load(&x, &i, mt, nt, kc, g).unwrap();
        let (bm, bn, bk) = (gemm.block_m(), gemm.block_n(), gemm.block_k());
        println!("block M={bm} N={bn} K={bk} (MT={mt} NT={nt} KCHUNK={kc} groups={g})");

        let rnd = |i: usize| -> i32 {
            let s = (i as u32).wrapping_mul(2654435761).wrapping_add(0x9e3779b9);
            ((s >> 13) & 0xf) as i32 - 8
        };
        let cpu = |m: usize, k: usize, n: usize, aa: &[i8], w: &[i8], c: &[i32]| -> usize {
            let mut mism = 0;
            for mm in 0..m {
                for nn in 0..n {
                    let acc: i32 = (0..k)
                        .map(|kk| aa[mm * k + kk] as i32 * w[kk * n + nn] as i32)
                        .sum();
                    if c[mm * n + nn] != acc {
                        mism += 1;
                    }
                }
            }
            mism
        };
        let run = |gemm: &mut NpuGemm, m: usize, k: usize, n: usize, salt: usize| -> usize {
            let av: Vec<i8> = (0..m * k).map(|i| rnd(salt + i) as i8).collect();
            let wv: Vec<i8> = (0..k * n)
                .map(|i| rnd(salt + 7_777_777 + i) as i8)
                .collect();
            let mut cv = vec![0i32; m * n];
            gemm.run(m, k, n, &av, &wv, &mut cv).unwrap();
            cpu(m, k, n, &av, &wv, &cv)
        };

        let m1 = run(&mut gemm, bm, bk, bn, 1);
        println!("one-block  M={bm} K={bk} N={bn}    : {m1} mismatches");
        let (m2, k2, n2) = (bm * 2, bk * 2, bn * 2);
        let mm2 = run(&mut gemm, m2, k2, n2, 2);
        println!("tiled 2x   M={m2} K={k2} N={n2} : {mm2} mismatches");
        if m1 != 0 || mm2 != 0 {
            eprintln!("NpuGemm WRONG");
            std::process::exit(4);
        }
        println!("NpuGemm W4A8 GEMM CORRECT");
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("Linux-only");
}
