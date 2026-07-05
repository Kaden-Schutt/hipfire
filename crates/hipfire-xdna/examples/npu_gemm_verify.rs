//! Validate NpuGemm vs a CPU W4A8 reference: one R6 block (M=64 K=256 N=64) and a
//! tiled shape (M=128 K=512 N=128 = 2x2x2 blocks, exercises K-accumulation).
//! Run: cargo run -p hipfire-xdna --example npu_gemm_verify -- <dir-MT16-NT4-KC16>
fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuGemm;
        let dir = std::env::args().nth(1).unwrap();
        let x = std::fs::read(format!("{dir}/final.xclbin")).unwrap();
        let i = std::fs::read(format!("{dir}/insts.bin")).unwrap();
        let mut g = NpuGemm::load(&x, &i, 16, 4, 16).unwrap();
        let rnd = |i: usize| -> i32 {
            let s = (i as u32).wrapping_mul(2654435761).wrapping_add(0x9e3779b9);
            ((s >> 13) & 0xf) as i32 - 8
        };
        let cpu = |m: usize, k: usize, n: usize, a: &[i8], w: &[i8], c: &[i32]| -> usize {
            let mut mism = 0;
            for mm in 0..m {
                for nn in 0..n {
                    let acc: i32 = (0..k)
                        .map(|kk| a[mm * k + kk] as i32 * w[kk * n + nn] as i32)
                        .sum();
                    if c[mm * n + nn] != acc {
                        mism += 1;
                    }
                }
            }
            mism
        };
        // block
        let (m, k, n) = (64, 256, 64);
        let a: Vec<i8> = (0..m * k).map(|i| rnd(i) as i8).collect();
        let w: Vec<i8> = (0..k * n).map(|i| rnd(1_000_000 + i) as i8).collect();
        let mut c = vec![0i32; m * n];
        g.run(m, k, n, &a, &w, &mut c).unwrap();
        println!(
            "block  M=64  K=256 N=64  : {}/{} mismatches",
            cpu(m, k, n, &a, &w, &c),
            m * n
        );
        // tiled 2x2x2
        let (m, k, n) = (128, 512, 128);
        let a: Vec<i8> = (0..m * k).map(|i| rnd(i) as i8).collect();
        let w: Vec<i8> = (0..k * n).map(|i| rnd(2_000_000 + i) as i8).collect();
        let mut c = vec![0i32; m * n];
        g.run(m, k, n, &a, &w, &mut c).unwrap();
        let mism = cpu(m, k, n, &a, &w, &c);
        println!("tiled  M=128 K=512 N=128 : {}/{} mismatches", mism, m * n);
        if mism != 0 {
            eprintln!("NpuGemm tiled WRONG");
            std::process::exit(4);
        }
        println!("NpuGemm W4A8 GEMM CORRECT (single-block + tiled K-accumulation)");
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("Linux-only");
}
