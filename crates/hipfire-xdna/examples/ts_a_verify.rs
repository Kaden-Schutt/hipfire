//! Validate the in-core tensor-buffer-stream A reshuffle (benchmarks/npu_gemm_tuning/ts):
//! the kernel reads ROW-MAJOR A via an aie::tensor_descriptor and must emit exactly the
//! `NpuGemm::pack_a` tile-major layout — WITHOUT any CPU marshaling or strided DMA. If
//! this passes, R6's A-read can drop the marshaler and consume row-major activations.
//!
//! Build the xclbin first: benchmarks/npu_gemm_tuning/ts/ts_build.sh [MT] [KCHUNK]
//! Run: cargo run -p hipfire-xdna --example ts_a_verify -- <workdir> [MT] [KCHUNK]

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuKernel;

        let a: Vec<String> = std::env::args().collect();
        let dir = a.get(1).cloned().unwrap_or_else(|| {
            eprintln!("usage: ts_a_verify <workdir> [MT] [KCHUNK]");
            std::process::exit(2);
        });
        let mt: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
        let kc: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
        const MR: usize = 4;
        const MK: usize = 16;
        let rows = mt * MR;
        let kb = kc * MK;
        let n = rows * kb; // bytes in = out

        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
        let insts = std::fs::read(format!("{dir}/insts.bin")).expect("insts");
        let k = NpuKernel::load(&xclbin, &insts).expect("load");

        let mut ain = k.alloc_arg(n).expect("A");
        let out = k.alloc_arg(n).expect("O");

        // Deterministic pseudo-random bytes, keyed by index.
        fn rnd(i: usize) -> i8 {
            let s = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(0x9e37_79b9);
            (((s >> 13) & 0xff) as u8) as i8
        }
        let src: Vec<i8> = (0..n).map(rnd).collect();
        ain.as_mut_slice()
            .copy_from_slice(&src.iter().map(|&v| v as u8).collect::<Vec<u8>>());

        k.dispatch(&[&ain, &out]).expect("dispatch");
        let got: &[u8] = out.as_slice();

        // pack_a reference: out[(mti*kc+ki)*64 + m*16 + kk] = a[(mti*4+m)*kb + ki*16 + kk]
        let mut mism = 0usize;
        for mti in 0..mt {
            for ki in 0..kc {
                for m in 0..MR {
                    for kk in 0..MK {
                        let dst = (mti * kc + ki) * (MR * MK) + m * MK + kk;
                        let want = src[(mti * MR + m) * kb + ki * MK + kk] as u8;
                        if got[dst] != want {
                            mism += 1;
                        }
                    }
                }
            }
        }
        println!("{mism}/{n} mismatches (MT={mt} KCHUNK={kc})");
        if mism != 0 {
            eprintln!("tensor-buffer-stream reshuffle WRONG");
            std::process::exit(4);
        }
        println!("tensor-buffer-stream A reshuffle CORRECT (row-major in == pack_a out)");
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
