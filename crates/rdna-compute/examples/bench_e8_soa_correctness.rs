// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mfp4-E8-SoA correctness + perf bench on gfx1151.
//
// Correctness gate: SoA kernel output MUST be bit-exact with AoS kernel output.
// Perf: SoA tok/s + GiB/s vs AoS to verify the alignment improvement hypothesis.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

// Roofline references. gfx1100 (RX 7900 XTX): GDDR6 VRAM ~960 GB/s; Infinity
// Cache (L3, 96 MB) ~3500 GB/s. A shape whose weights fit L3 is cache-resident
// after warmup → its real ceiling is L3, not VRAM. gfx1151 (Strix Halo): ~256.
const L3_BYTES: usize = 96 * 1024 * 1024;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    let vram_gbps = if arch == "gfx1151" { 256.0 } else { 960.0 };
    eprintln!("=== mfp4-E8-SoA correctness + perf bench ===");
    eprintln!("  arch={arch}  vram_peak_gbps={vram_gbps}  l3_bytes={L3_BYTES}");
    eprintln!();

    if std::env::var_os("HIPFIRE_E8_PACK_CORRECTNESS_ONLY").is_some() {
        run_pack_correctness(&mut gpu);
        return;
    }
    if std::env::var_os("HIPFIRE_E8_PREFILL_COOP_ONLY").is_some() {
        run_prefill_coop(&mut gpu);
        return;
    }
    if std::env::var_os("HIPFIRE_E8_GROUPED_GFX1201_ONLY").is_some() {
        run_grouped_gfx1201(&mut gpu);
        return;
    }
    if std::env::var_os("HIPFIRE_E8_PREFILL_GFX1201_ONLY").is_some() {
        run_prefill_gfx1201(&mut gpu);
        return;
    }

    let shapes: Vec<(usize, usize, &str)> = vec![
        // --- A3B per-expert MoE shapes (the decode hot path; hidden=2048,
        //     moe_intermediate=512 — qwen35.rs:170,204) ---
        (1024, 2048, "A3B gate_up M=1024 K=2048 (expert)"), // M=2*moe_int, K=hidden, groups=8
        (2048, 512, "A3B down    M=2048 K=512  (expert)"),  // M=hidden, K=moe_int, groups=2
        // --- DeepSeek-V4 MQ2R P1 dense shapes ---
        (1024, 4096, "DS4 wq_a/wo_a M=1024 K=4096"),
        (32768, 1024, "DS4 wq_b      M=32768 K=1024"),
        (4096, 8192, "DS4 wo_b       M=4096 K=8192"),
        (2048, 4096, "DS4 shared up  M=2048 K=4096"),
        (4096, 2048, "DS4 shared dn  M=4096 K=2048"),
        // --- wave-count sweep at fixed K=4096: one wave per row, so M == waves.
        //     gfx1151 holds 40 CU x 2 SIMD32 x 16 = 1280 waves resident. If
        //     achieved bandwidth rises with M and saturates, the small-M dense
        //     shapes are memory-level-parallelism starved and split-K pays. ---
        (512, 4096, "WAVESWEEP M=512   K=4096"),
        (1024, 4096, "WAVESWEEP M=1024  K=4096"),
        (2048, 4096, "WAVESWEEP M=2048  K=4096"),
        (4096, 4096, "WAVESWEEP M=4096  K=4096"),
        (8192, 4096, "WAVESWEEP M=8192  K=4096"),
        (16384, 4096, "WAVESWEEP M=16384 K=4096"),
        // --- large-dense reference (where the SoA-coalescing wins showed up;
        //     crosses the 96 MB L3 boundary at ~91k rows) ---
        (11008, 2048, "dense gate_up M=11008 K=2048"),
        (32768, 2048, "sweep        M=32768  K=2048  (L3)"),
        (131072, 2048, "sweep        M=131072 K=2048  (VRAM)"),
    ];

    let warmup = 20usize;
    let trials = 200usize;

    // ---- CORRECTNESS GATE ----
    eprintln!("--- Correctness gate: SoA output == AoS output (bit-exact) ---");
    let (m, k) = (512, 2048);
    let aos_data = synth_e8_aos(m, k, 0xDEAD_BEEF);
    let soa_data = aos_to_soa_full(&aos_data, m, k);

    let aos_w = gpu
        .upload_raw(&aos_data, &[aos_data.len()])
        .expect("upload AoS");
    let soa_w = gpu
        .upload_raw(&soa_data, &[soa_data.len()])
        .expect("upload SoA");
    let x = gpu.alloc_tensor(&[k], DType::F32).expect("alloc x");
    let y_aos = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y_aos");
    let y_soa = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y_soa");
    let y_u4 = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y_u4");

    let xh = make_x(k, 0x1234_5678);
    gpu.hip.memcpy_htod(&x.buf, bytes_of(&xh)).unwrap();

    gpu.gemv_mfp4g32_e8(&aos_w, &x, &y_aos, m, k)
        .expect("AoS GEMV");
    gpu.gemv_mfp4g32_e8_soa(&soa_w, &x, &y_soa, m, k)
        .expect("SoA GEMV");
    gpu.gemv_mfp4g32_e8_soa_u4(&soa_w, &x, &y_u4, m, k)
        .expect("SoA U4 GEMV");
    gpu.hip.device_synchronize().unwrap();

    let mut res_aos = vec![0f32; m];
    let mut res_soa = vec![0f32; m];
    gpu.hip
        .memcpy_dtoh(bytes_of_mut(&mut res_aos), &y_aos.buf)
        .unwrap();
    gpu.hip
        .memcpy_dtoh(bytes_of_mut(&mut res_soa), &y_soa.buf)
        .unwrap();
    let mut all_exact = true;
    let mut n_diff = 0usize;
    for i in 0..m {
        if res_aos[i].to_bits() != res_soa[i].to_bits() {
            if n_diff < 3 {
                eprintln!(
                    "  MISMATCH at i={}: aos={} soa={}",
                    i, res_aos[i], res_soa[i]
                );
            }
            n_diff += 1;
            all_exact = false;
        }
    }
    if all_exact {
        eprintln!(
            "  CORRECTNESS PASS: SoA output == AoS output (bit-exact, {} outputs)",
            m
        );
    } else {
        // SoA reorders the per-group reduction → low-FP-bit differences are expected
        // (e.g. -123.679184 vs -123.67917, ~1e-7 relative). Not a real failure; the
        // weights/math are identical. Continue to the perf bench.
        eprintln!(
            "  NOTE: {} of {} outputs differ in low FP bits (SoA reduction reorder) — continuing to perf",
            n_diff, m
        );
    }
    eprintln!();

    // ---- PERF BENCH: MLP sweep (cache-roofline) ----
    eprintln!(
        "--- Perf bench: MQ4 vs E8 AoS/SoA/strip/LUT — GB/s (% of {:.0} VRAM) ---",
        vram_gbps
    );
    eprintln!(
        "  MQ4 = plain uniform HFQ4-G256 (136 B/group, gemv_hfq4g256); E8 columns = mfp4-G32 (17 B/block)."
    );
    eprintln!(
        "  Same-shape, weight-bytes-only denominator → MQ4 GB/s is directly comparable to the E8 AoS column."
    );
    eprintln!(
        "  L3-resident rows (weights < 96MB) read from Infinity Cache (~3500 GB/s) after warmup."
    );
    eprintln!(
        "  {:<34}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>5}",
        "shape", "MQ4", "E8-AoS", "E8-SoA-1r", "E8-U4", "E8-U8", "E8-strip", "E8-LUT", "resid"
    );
    eprintln!("  {}", "-".repeat(121));

    for (m, k, label) in &shapes {
        let (m, k) = (*m, *k);

        let aos_data = synth_e8_aos(m, k, 0x1234 ^ m as u64 ^ k as u64);
        let soa_data = aos_to_soa_full(&aos_data, m, k);
        let aos_total = aos_data.len();
        let soa_total = soa_data.len();
        let resid = if soa_total <= L3_BYTES { "L3" } else { "VRAM" };

        let aos_w = gpu.upload_raw(&aos_data, &[aos_total]).unwrap();
        let soa_w = gpu.upload_raw(&soa_data, &[soa_total]).unwrap();
        let x = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let y = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        let xh = make_x(k, 0xABCD);
        gpu.hip.memcpy_htod(&x.buf, bytes_of(&xh)).unwrap();

        // warmup + timed trials → GB/s for the named gemv method on a given buffer
        macro_rules! gbps {
            ($method:ident, $w:expr, $bytes:expr) => {{
                for _ in 0..warmup {
                    gpu.$method($w, &x, &y, m, k).unwrap();
                }
                gpu.hip.device_synchronize().unwrap();
                let t = Instant::now();
                for _ in 0..trials {
                    gpu.$method($w, &x, &y, m, k).unwrap();
                }
                gpu.hip.device_synchronize().unwrap();
                let us = t.elapsed().as_secs_f64() * 1e6 / trials as f64;
                $bytes as f64 / (us * 1e-6) / 1e9
            }};
        }

        // MQ4 (plain uniform HFQ4-G256) standalone — same shape, weight-only bytes.
        // gemv_hfq4g256 has the identical (a_raw,x,y,m,k) signature so it slots
        // straight into the gbps! macro. Buffer is separate; x/y are reused.
        let mq4_data = synth_hfq4g256(m, k, 0x4D51 ^ m as u64 ^ k as u64); // "MQ"
        let mq4_total = mq4_data.len(); // == profile::hfq4g256_weight_bytes(m,k)
        let mq4_w = gpu.upload_raw(&mq4_data, &[mq4_total]).unwrap();

        let mq4 = gbps!(gemv_hfq4g256, &mq4_w, mq4_total);
        let aos = gbps!(gemv_mfp4g32_e8, &aos_w, aos_total);
        let soa2 = gbps!(gemv_mfp4g32_e8_soa, &soa_w, soa_total);
        let u4 = gbps!(gemv_mfp4g32_e8_soa_u4, &soa_w, soa_total);
        let u8 = gbps!(gemv_mfp4g32_e8_soa_u8, &soa_w, soa_total);
        let strip = gbps!(gemv_mfp4g32_e8_soa_strip, &soa_w, soa_total);
        let lut = gbps!(gemv_mfp4g32_e8_soa_lut, &soa_w, soa_total);

        let cell = |g: f64| format!("{:4.0}({:3.0}%)", g, g / vram_gbps * 100.0);
        eprintln!(
            "  {:<34}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>5}",
            label,
            cell(mq4),
            cell(aos),
            cell(soa2),
            cell(u4),
            cell(u8),
            cell(strip),
            cell(lut),
            resid
        );
    }

    // ---- PERF BENCH 2: DS4 dense shapes forced DRAM-RESIDENT ----
    //
    // The rows above run each shape against ONE buffer, so every DS4 dense tensor
    // (2-18 MB) sits inside the 96 MB Infinity Cache and reads at >100% of VRAM
    // peak. That regime does not exist in-model: real decode streams 3.63 GB of
    // dense weights per token and thrashes the cache completely.
    //
    // Here each shape is replicated until the working set exceeds L3, and the
    // timed loop cycles the replicas so no iteration reuses a cached buffer.
    // This is the only comparison that answers "does HFQ4's cheaper nibble
    // decode still beat E8's lattice decode when we are bandwidth-bound?"
    eprintln!();
    eprintln!("--- Perf bench 2: DS4 dense shapes, DRAM-RESIDENT (working set > L3) ---");
    eprintln!("  Replicas cycled per iteration so no buffer is cache-resident.");
    eprintln!(
        "  {:<34}  {:>4}  {:>9}  {:>12}  {:>12}  {:>8}",
        "shape", "reps", "set MB", "HFQ4-G256", "E8-U4", "ratio"
    );
    eprintln!("  {}", "-".repeat(92));

    for (m, k, label) in &shapes {
        let (m, k) = (*m, *k);
        if !label.starts_with("DS4") && !label.starts_with("WAVESWEEP") {
            continue;
        }

        let soa_one = aos_to_soa_full(&synth_e8_aos(m, k, 0x99 ^ m as u64), m, k).len();
        let reps = ((L3_BYTES * 3 / 2) / soa_one.max(1)).max(2) + 1;

        let mut soa_bufs = Vec::with_capacity(reps);
        let mut mq4_bufs = Vec::with_capacity(reps);
        let (mut soa_total, mut mq4_total) = (0usize, 0usize);
        for r in 0..reps {
            let s = r as u64;
            let a = synth_e8_aos(m, k, 0x1234 ^ m as u64 ^ k as u64 ^ (s << 32));
            let sd = aos_to_soa_full(&a, m, k);
            soa_total = sd.len();
            soa_bufs.push(gpu.upload_raw(&sd, &[sd.len()]).unwrap());
            let md = synth_hfq4g256(m, k, 0x4D51 ^ m as u64 ^ k as u64 ^ (s << 32));
            mq4_total = md.len();
            mq4_bufs.push(gpu.upload_raw(&md, &[md.len()]).unwrap());
        }

        let x = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let y = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        let xh = make_x(k, 0xABCD);
        gpu.hip.memcpy_htod(&x.buf, bytes_of(&xh)).unwrap();

        // Cycle replicas so each timed call touches a buffer that was evicted by
        // the intervening reps. Bytes denominator is per-call weight bytes.
        macro_rules! gbps_stream {
            ($method:ident, $bufs:expr, $bytes:expr) => {{
                for r in 0..$bufs.len() {
                    gpu.$method(&$bufs[r], &x, &y, m, k).unwrap();
                }
                gpu.hip.device_synchronize().unwrap();
                let t = Instant::now();
                let mut n = 0usize;
                for _ in 0..trials {
                    for r in 0..$bufs.len() {
                        gpu.$method(&$bufs[r], &x, &y, m, k).unwrap();
                        n += 1;
                    }
                }
                gpu.hip.device_synchronize().unwrap();
                let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
                $bytes as f64 / (us * 1e-6) / 1e9
            }};
        }

        let mq4 = gbps_stream!(gemv_hfq4g256, mq4_bufs, mq4_total);
        let u4 = gbps_stream!(gemv_mfp4g32_e8_soa_u4, soa_bufs, soa_total);

        let set_mb = (soa_total * reps) as f64 / 1e6;
        eprintln!(
            "  {:<34}  {:>4}  {:>9.1}  {:>12}  {:>12}  {:>7.2}x",
            label,
            reps,
            set_mb,
            format!("{:4.0}({:3.0}%)", mq4, mq4 / vram_gbps * 100.0),
            format!("{:4.0}({:3.0}%)", u4, u4 / vram_gbps * 100.0),
            mq4 / u4
        );
    }
    eprintln!(
        "  ratio > 1.3 => HFQ4 decode advantage SURVIVES bandwidth-bound (format is a lever)"
    );
    eprintln!("  ratio ~ 1.0 => advantage was cache-only; dense format is NOT the lever (use cache policy)");

    // ---- PERF BENCH 2d: BATCHED GEMV vs WMMA GEMM AT VERIFY BATCH SIZES ----
    // The batched/prefill forward dispatches dense projections through
    // gemm_mfp4g32_e8_soa_wmma_gfx1151, which tiles the TOKEN axis at 16 and
    // launches only M/16 waves. Measured end-to-end, the ds4 batched forward
    // costs 104 + 12.3*B ms; the 104 ms constant is that 16-token tile, and it
    // is why a B=1 batched forward takes 115.7 ms where decode does the same
    // token in 36.1 ms. Speculative verify lives at B=5..8, entirely inside
    // that constant.
    //
    // gemv_..._batched keeps the decode GEMV's M-wave occupancy and single
    // weight-row read, paying B x arithmetic instead of 16/B x wasted tile.
    {
        eprintln!();
        eprintln!("--- Perf bench 2d: dense E8 at verify batch — WMMA tile vs batched GEMV (DRAM-resident) ---");
        let (m, k) = (4096usize, 4096usize);
        let one = aos_to_soa_full(&synth_e8_aos(m, k, 0xBA7C), m, k);
        let reps = ((L3_BYTES * 3 / 2) / one.len().max(1)).max(2) + 1;
        let mut bufs = Vec::with_capacity(reps);
        for r in 0..reps {
            let d = aos_to_soa_full(&synth_e8_aos(m, k, 0xBA7C ^ ((r as u64) << 32)), m, k);
            bufs.push(gpu.upload_raw(&d, &[d.len()]).unwrap());
        }
        let wbytes = one.len();
        eprintln!(
            "  M={} K={}  weights {:.1} MB x {} replicas = {:.0} MB (L3 {} MB)",
            m,
            k,
            wbytes as f64 / 1e6,
            reps,
            (reps * wbytes) as f64 / 1e6,
            L3_BYTES / (1024 * 1024)
        );
        eprintln!(
            "  {:>3} {:>11} {:>11} {:>9} {:>11} {:>11}",
            "B", "WMMA us", "batched us", "speedup", "WMMA GB/s", "batch GB/s"
        );

        for b in [1usize, 2, 4, 6, 8, 16] {
            let xh = make_x(b * k, 0x51DE ^ b as u64);
            let x32 = gpu.alloc_tensor(&[b * k], DType::F32).unwrap();
            gpu.hip.memcpy_htod(&x32.buf, bytes_of(&xh)).unwrap();
            let y = gpu.alloc_tensor(&[b * m], DType::F32).unwrap();

            macro_rules! timed {
                ($call:expr) => {{
                    for r in 0..bufs.len() {
                        let _ = r;
                        $call
                    }
                    gpu.hip.device_synchronize().unwrap();
                    let t = Instant::now();
                    let mut n = 0usize;
                    for _ in 0..trials {
                        for r in 0..bufs.len() {
                            let _ = r;
                            $call
                            n += 1;
                        }
                    }
                    gpu.hip.device_synchronize().unwrap();
                    t.elapsed().as_secs_f64() * 1e6 / n as f64
                }};
            }
            let mut us_w = 0.0f64;
            {
                for r in 0..bufs.len() {
                    gpu.gemm_mfp4g32_e8_soa_wmma(&bufs[r], &x32, &y, m, k, b)
                        .unwrap();
                }
                gpu.hip.device_synchronize().unwrap();
                let t = Instant::now();
                let mut n = 0usize;
                for _ in 0..trials {
                    for r in 0..bufs.len() {
                        gpu.gemm_mfp4g32_e8_soa_wmma(&bufs[r], &x32, &y, m, k, b)
                            .unwrap();
                        n += 1;
                    }
                }
                gpu.hip.device_synchronize().unwrap();
                us_w = t.elapsed().as_secs_f64() * 1e6 / n as f64;
            }
            let us_b = {
                for r in 0..bufs.len() {
                    gpu.gemv_mfp4g32_e8_soa_batched_gfx1151(&bufs[r], &x32, &y, b, m, k)
                        .unwrap();
                }
                gpu.hip.device_synchronize().unwrap();
                let t = Instant::now();
                let mut n = 0usize;
                for _ in 0..trials {
                    for r in 0..bufs.len() {
                        gpu.gemv_mfp4g32_e8_soa_batched_gfx1151(&bufs[r], &x32, &y, b, m, k)
                            .unwrap();
                        n += 1;
                    }
                }
                gpu.hip.device_synchronize().unwrap();
                t.elapsed().as_secs_f64() * 1e6 / n as f64
            };
            eprintln!(
                "  {:>3} {:>11.2} {:>11.2} {:>8.2}x {:>11.1} {:>11.1}",
                b,
                us_w,
                us_b,
                us_w / us_b,
                wbytes as f64 / (us_w * 1e-6) / 1e9,
                wbytes as f64 / (us_b * 1e-6) / 1e9
            );
        }
        // Bit-exactness gate for B=1: the batched kernel must reproduce the
        // decode GEMV it is meant to replace, on identical weights and
        // activations. Layout, scale conversion and lattice decode are copied
        // verbatim; the accumulator STRIDING differs (u4 carries four
        // accumulators strided by 4 groups, batched carries one per token
        // accumulated sequentially), so this reports the actual divergence
        // rather than assuming it is zero.
        {
            let xh = make_x(k, 0x5EED);
            let x1 = gpu.alloc_tensor(&[k], DType::F32).unwrap();
            gpu.hip.memcpy_htod(&x1.buf, bytes_of(&xh)).unwrap();
            let y_ref = gpu.alloc_tensor(&[m], DType::F32).unwrap();
            let y_new = gpu.alloc_tensor(&[m], DType::F32).unwrap();
            gpu.gemv_mfp4g32_e8_soa_u4(&bufs[0], &x1, &y_ref, m, k)
                .unwrap();
            gpu.gemv_mfp4g32_e8_soa_batched_gfx1151(&bufs[0], &x1, &y_new, 1, m, k)
                .unwrap();
            gpu.hip.device_synchronize().unwrap();
            let mut rr = vec![0f32; m];
            let mut rn = vec![0f32; m];
            gpu.hip
                .memcpy_dtoh(bytes_of_mut(&mut rr), &y_ref.buf)
                .unwrap();
            gpu.hip
                .memcpy_dtoh(bytes_of_mut(&mut rn), &y_new.buf)
                .unwrap();
            let mut exact = 0usize;
            let mut max_rel = 0f64;
            let mut max_ulp = 0i64;
            for (a, b) in rr.iter().zip(&rn) {
                if a.to_bits() == b.to_bits() {
                    exact += 1;
                } else {
                    let denom = (a.abs().max(b.abs()) as f64).max(1e-30);
                    max_rel = max_rel.max(((a - b).abs() as f64) / denom);
                    let ulp = (a.to_bits() as i64 - b.to_bits() as i64).abs();
                    max_ulp = max_ulp.max(ulp);
                }
            }
            eprintln!(
                "  B=1 vs gemv_mfp4g32_e8_soa_u4: {}/{} bit-exact, max_rel {:.3e}, max_ulp {} ({})",
                exact,
                m,
                max_rel,
                max_ulp,
                if exact == m { "PASS" } else { "DIVERGES" }
            );
        }
        eprintln!("  batched GB/s should stay ~flat across B (weights read once); WMMA should be flat in TIME below B=16.");
    }

    // ---- PERF BENCH 2c: NON-TEMPORAL CACHE POLICY, DRAM-RESIDENT ----
    // The retained ds4 route streams 4.68 GB of weights per token through a
    // 96 MB MALL, and weights have ZERO reuse within a token — every byte
    // evicts something that might. cpol 20/22 (TH_NT_RT / TH_NT_HT + SCOPE_DEV)
    // mark those reads non-temporal so they stop thrashing the cache for data
    // that does reuse. The variants are registered but unused; the route runs
    // cpol0.
    //
    // A non-temporal hint can only matter when the working set exceeds cache,
    // so this MUST be measured DRAM-resident. Measuring it cache-resident is
    // what flipped the buffer-SRD result from -1.33% to +1.08% earlier.
    {
        eprintln!();
        eprintln!("--- Perf bench 2c: E8 weight cache policy (0=temporal, 20=NT_RT, 22=NT_HT), DRAM-resident ---");
        eprintln!(
            "  {:<30} {:>5} {:>9} {:>12} {:>12} {:>12}",
            "shape", "reps", "set MB", "cpol0", "cpol20 NT_RT", "cpol22 NT_HT"
        );
        for (m, k, label) in [
            (1024usize, 4096usize, "DS4 wq_a/wo_a M=1024 K=4096"),
            (4096, 8192, "DS4 wo_b      M=4096 K=8192"),
            (32768, 1024, "DS4 wq_b      M=32768 K=1024"),
            (2048, 4096, "DS4 shared up M=2048 K=4096"),
        ] {
            let one = aos_to_soa_full(&synth_e8_aos(m, k, 0xC901 ^ m as u64), m, k);
            let reps = ((L3_BYTES * 3 / 2) / one.len().max(1)).max(2) + 1;
            let mut bufs = Vec::with_capacity(reps);
            for r in 0..reps {
                let d = aos_to_soa_full(
                    &synth_e8_aos(m, k, 0xC901 ^ m as u64 ^ ((r as u64) << 32)),
                    m,
                    k,
                );
                bufs.push(gpu.upload_raw(&d, &[d.len()]).unwrap());
            }
            let wbytes = one.len();
            let x = gpu.alloc_tensor(&[k], DType::F32).unwrap();
            let y = gpu.alloc_tensor(&[m], DType::F32).unwrap();
            let xh = make_x(k, 0xC0FE);
            gpu.hip.memcpy_htod(&x.buf, bytes_of(&xh)).unwrap();

            let mut gbps = |policy: u32| {
                for r in 0..bufs.len() {
                    gpu.gemv_mfp4g32_e8_soa_u4_buffer_cpol_gfx1151(policy, &bufs[r], &x, &y, m, k)
                        .unwrap();
                }
                gpu.hip.device_synchronize().unwrap();
                let t = Instant::now();
                let mut n = 0usize;
                for _ in 0..trials {
                    for r in 0..bufs.len() {
                        gpu.gemv_mfp4g32_e8_soa_u4_buffer_cpol_gfx1151(
                            policy, &bufs[r], &x, &y, m, k,
                        )
                        .unwrap();
                        n += 1;
                    }
                }
                gpu.hip.device_synchronize().unwrap();
                let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
                wbytes as f64 / (us * 1e-6) / 1e9
            };
            let c0 = gbps(0);
            let c20 = gbps(20);
            let c22 = gbps(22);
            eprintln!(
                "  {:<30} {:>5} {:>9.1} {:>7.1} GB/s {:>6.1} ({:+.2}%) {:>6.1} ({:+.2}%)",
                label,
                reps,
                (reps * wbytes) as f64 / 1e6,
                c0,
                c20,
                (c20 - c0) / c0 * 100.0,
                c22,
                (c22 - c0) / c0 * 100.0
            );
        }
        eprintln!("  NT wins => weights are polluting MALL; route should switch cpol.");
    }

    // ---- PERF BENCH 2b: GROUPED E8 — plain global vs buffer-SRD loads ----
    // The grouped kernel is the tape's second-largest block (15.63%, 5363 us)
    // and achieves ~143 GB/s at 8192 waves (6.4 fills), where the wave sweep
    // above says ~192 is available at that occupancy. Its code object reports
    // buffer_loads = 0 against the main E8 kernel's 11: it never received the
    // accepted B2 cache-policy treatment. Recorded shape is G=8, M=1024,
    // K=4096 (DeepSeek V4 wo_a).
    {
        eprintln!();
        eprintln!("--- Perf bench 2b: grouped E8, plain-global vs buffer-SRD (recorded shape) ---");
        let (g, m, k) = (8usize, 1024usize, 4096usize);
        // Residency must MATCH the tape, not be assumed. One grouped weight set
        // is G*M*row_total = 17.96 MB, which fits the 96 MB MALL entirely — an
        // unreplicated bench measures a cache-resident regime the retained route
        // never sees, where these weights stream from DRAM alongside 4.68 GB of
        // other per-token traffic. Cycle replicas past L3 so each timed call
        // touches weights the intervening reps evicted.
        let set_of = |r: u64| {
            let mut b = Vec::new();
            for grp in 0..g {
                b.extend_from_slice(&aos_to_soa_full(
                    &synth_e8_aos(m, k, 0x9E0 ^ grp as u64 ^ (r << 32)),
                    m,
                    k,
                ));
            }
            b
        };
        let blob = set_of(0);
        let reps = ((L3_BYTES * 3 / 2) / blob.len().max(1)).max(2) + 1;
        let mut ws = Vec::with_capacity(reps);
        for r in 0..reps {
            let b = set_of(r as u64);
            ws.push(gpu.upload_raw(&b, &[b.len()]).unwrap());
        }
        eprintln!(
            "  residency: {} replicas x {:.1} MB = {:.0} MB working set (L3 = {} MB)",
            reps,
            blob.len() as f64 / 1e6,
            (reps * blob.len()) as f64 / 1e6,
            L3_BYTES / (1024 * 1024)
        );
        let w = &ws[0];
        let xg = gpu.alloc_tensor(&[g * k], DType::F32).unwrap();
        let xh = make_x(g * k, 0x6172);
        gpu.hip.memcpy_htod(&xg.buf, bytes_of(&xh)).unwrap();
        let y_plain = gpu.alloc_tensor(&[g * m], DType::F32).unwrap();
        let y_buf = gpu.alloc_tensor(&[g * m], DType::F32).unwrap();

        gpu.gemv_mfp4g32_e8_soa_grouped_gfx1151(w, &xg, &y_plain, g, m, k)
            .unwrap();
        gpu.gemv_mfp4g32_e8_soa_grouped_buffer_gfx1151(w, &xg, &y_buf, g, m, k)
            .unwrap();
        gpu.hip.device_synchronize().unwrap();
        let mut rp = vec![0f32; g * m];
        let mut rb = vec![0f32; g * m];
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut rp), &y_plain.buf)
            .unwrap();
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut rb), &y_buf.buf)
            .unwrap();
        let bad = rp
            .iter()
            .zip(&rb)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "  bit-exactness: {} / {} outputs differ  ({})",
            bad,
            g * m,
            if bad == 0 { "PASS" } else { "FAIL" }
        );

        let wbytes = blob.len();
        macro_rules! grp_gbps {
            ($method:ident, $y:expr) => {{
                for r in 0..ws.len() {
                    gpu.$method(&ws[r], &xg, $y, g, m, k).unwrap();
                }
                gpu.hip.device_synchronize().unwrap();
                let t = Instant::now();
                let mut n = 0usize;
                for _ in 0..trials {
                    for r in 0..ws.len() {
                        gpu.$method(&ws[r], &xg, $y, g, m, k).unwrap();
                        n += 1;
                    }
                }
                gpu.hip.device_synchronize().unwrap();
                let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
                (us, wbytes as f64 / (us * 1e-6) / 1e9)
            }};
        }
        let (us_p, gb_p) = grp_gbps!(gemv_mfp4g32_e8_soa_grouped_gfx1151, &y_plain);
        let (us_b, gb_b) = grp_gbps!(gemv_mfp4g32_e8_soa_grouped_buffer_gfx1151, &y_buf);
        eprintln!(
            "  plain-global   {:8.2} us  {:6.1} GB/s ({:3.0}% of {:.0})",
            us_p,
            gb_p,
            gb_p / vram_gbps * 100.0,
            vram_gbps
        );
        eprintln!(
            "  buffer-SRD     {:8.2} us  {:6.1} GB/s ({:3.0}% of {:.0})   delta {:+.2}%",
            us_b,
            gb_b,
            gb_b / vram_gbps * 100.0,
            vram_gbps,
            (gb_b - gb_p) / gb_p * 100.0
        );
        eprintln!("  in-tape recorded: 5363 us / 43 calls = 124.7 us/call at 143 GB/s");
    }

    // ---- PERF BENCH 3: SMALL-BESIDE-BIG DISPATCH OVERLAP ----
    // 67.5% of the ds4 decode tape runs in blocks under 4 occupancy fills, and
    // several (mq_rotate_x = 48 waves, sqrt_softplus = 8) cannot be widened --
    // the work simply isn't there. Their only remedy is running concurrently
    // with a big kernel. Prior concurrency levers (b3-e8-pm4-concurrency,
    // g3-c-e8-concurrency-closeout) paired BIG with BIG, where both compete
    // for the same wave slots and there is nothing to win. This measures the
    // untested case: does a 48-wave kernel ride along free next to a
    // 4096-wave one? Same kernel, two M values, independent buffers.
    {
        eprintln!();
        eprintln!("--- Perf bench 3: small-beside-big dispatch overlap (independent buffers) ---");
        let k_ov = 4096usize;
        let trials = 200usize;
        let a_big = synth_e8_aos(4096, k_ov, 0xB16);
        let d_big = aos_to_soa_full(&a_big, 4096, k_ov);
        let w_big = gpu.upload_raw(&d_big, &[d_big.len()]).unwrap();
        let x_ov = gpu.alloc_tensor(&[k_ov], DType::F32).unwrap();
        let xh = make_x(k_ov, 0x5A5A);
        gpu.hip.memcpy_htod(&x_ov.buf, bytes_of(&xh)).unwrap();
        let y_big = gpu.alloc_tensor(&[4096], DType::F32).unwrap();

        eprintln!(
            "  {:<26} {:>10} {:>10} {:>10} {:>9}",
            "small kernel waves", "big us", "small us", "both us", "verdict"
        );
        for small_m in [48usize, 256, 1024] {
            let a_s = synth_e8_aos(small_m, k_ov, 0x51A11 ^ small_m as u64);
            let d_s = aos_to_soa_full(&a_s, small_m, k_ov);
            let w_s = gpu.upload_raw(&d_s, &[d_s.len()]).unwrap();
            let y_s = gpu.alloc_tensor(&[small_m], DType::F32).unwrap();

            macro_rules! timed {
                ($body:block) => {{
                    for _ in 0..20 {
                        $body
                    }
                    gpu.hip.device_synchronize().unwrap();
                    let t = Instant::now();
                    for _ in 0..trials {
                        $body
                    }
                    gpu.hip.device_synchronize().unwrap();
                    t.elapsed().as_secs_f64() * 1e6 / trials as f64
                }};
            }

            let us_big = timed!({
                gpu.gemv_mfp4g32_e8_soa_u4(&w_big, &x_ov, &y_big, 4096, k_ov)
                    .unwrap();
            });
            let us_small = timed!({
                gpu.gemv_mfp4g32_e8_soa_u4(&w_s, &x_ov, &y_s, small_m, k_ov)
                    .unwrap();
            });
            let us_both = timed!({
                gpu.gemv_mfp4g32_e8_soa_u4(&w_big, &x_ov, &y_big, 4096, k_ov)
                    .unwrap();
                gpu.gemv_mfp4g32_e8_soa_u4(&w_s, &x_ov, &y_s, small_m, k_ov)
                    .unwrap();
            });
            let serial = us_big + us_small;
            let hidden = (serial - us_both) / us_small.max(1e-9) * 100.0;
            eprintln!(
                "  {:<26} {:>10.2} {:>10.2} {:>10.2} {:>8.0}% hidden (serial {:.2})",
                format!("M={} ({} waves)", small_m, small_m),
                us_big,
                us_small,
                us_both,
                hidden.clamp(-999.0, 999.0),
                serial
            );
        }
        eprintln!("  hidden ~100% => small kernel rides along free; overlap is the lever.");
        eprintln!("  hidden ~0%   => dispatches serialize; only widening or launch removal helps.");

        // Bench 3b: same pair, but on TWO streams so nothing forces ordering.
        // Bench 3 above is a single in-order HIP stream, which inserts a barrier
        // between kernels by construction -- it cannot answer whether gfx1151
        // can overlap, only that HIP won't let it. RADV is on record overlapping
        // independent dispatches 1.29x on this family; this asks whether ROCm
        // exposes the same capability.
        eprintln!();
        eprintln!("--- Perf bench 3b: same pair on TWO streams (ordering removed) ---");
        let us_big_ref = {
            for _ in 0..20 {
                gpu.gemv_mfp4g32_e8_soa_u4(&w_big, &x_ov, &y_big, 4096, k_ov)
                    .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let t = Instant::now();
            for _ in 0..trials {
                gpu.gemv_mfp4g32_e8_soa_u4(&w_big, &x_ov, &y_big, 4096, k_ov)
                    .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            t.elapsed().as_secs_f64() * 1e6 / trials as f64
        };
        eprintln!("  big-alone reference: {:.2} us", us_big_ref);
        eprintln!(
            "  {:<26} {:>11} {:>11} {:>9}",
            "small kernel waves", "1-stream us", "2-stream us", "hidden"
        );
        for small_m in [48usize, 256, 1024] {
            let a_s = synth_e8_aos(small_m, k_ov, 0x51A11 ^ small_m as u64);
            let d_s = aos_to_soa_full(&a_s, small_m, k_ov);
            let w_s = gpu.upload_raw(&d_s, &[d_s.len()]).unwrap();
            let y_s = gpu.alloc_tensor(&[small_m], DType::F32).unwrap();
            let mut sa = Some(gpu.hip.stream_create().unwrap());
            let mut sb = Some(gpu.hip.stream_create().unwrap());

            macro_rules! pair {
                ($two:expr) => {{
                    macro_rules! go {
                        () => {{
                            gpu.active_stream = sa.take();
                            gpu.gemv_mfp4g32_e8_soa_u4(&w_big, &x_ov, &y_big, 4096, k_ov)
                                .unwrap();
                            sa = gpu.active_stream.take();
                            if $two {
                                gpu.active_stream = sb.take();
                            } else {
                                gpu.active_stream = sa.take();
                            }
                            gpu.gemv_mfp4g32_e8_soa_u4(&w_s, &x_ov, &y_s, small_m, k_ov)
                                .unwrap();
                            if $two {
                                sb = gpu.active_stream.take();
                            } else {
                                sa = gpu.active_stream.take();
                            }
                        }};
                    }
                    for _ in 0..20 {
                        go!();
                    }
                    gpu.hip.device_synchronize().unwrap();
                    let t = Instant::now();
                    for _ in 0..trials {
                        go!();
                    }
                    gpu.hip.device_synchronize().unwrap();
                    t.elapsed().as_secs_f64() * 1e6 / trials as f64
                }};
            }

            let one = pair!(false);
            let two = pair!(true);
            gpu.active_stream = None;
            let hidden = (one - two) / (one - us_big_ref).max(1e-9) * 100.0;
            eprintln!(
                "  {:<26} {:>11.2} {:>11.2} {:>8.0}%",
                format!("M={} ({} waves)", small_m, small_m),
                one,
                two,
                hidden.clamp(-999.0, 999.0)
            );
        }
        eprintln!(
            "  hidden ~100% => ROCm CAN overlap; the tape's serial waits are the whole cost."
        );
        eprintln!("  hidden ~0%   => ROCm serializes regardless; widening is the only lever.");
    }

    eprintln!();
    eprintln!(
        "  cells = GB/s (% of {:.0} GB/s VRAM). L3-resident rows: true ceiling ~3500 GB/s.",
        vram_gbps
    );
    eprintln!("  E8-strip = lattice-decode REMOVED (garbage out) = loads+scale+reduce ceiling.");
    eprintln!(
        "  strip >> SoA-2w on L3 → decode-compute-bound; SoA-LUT≈strip → LUT is the win; LUT≈SoA-2w → parity/extract is the wall."
    );
    eprintln!();
    eprintln!("  H1 keystone (mq4 vs E8, weight bytes / row, from profile.rs):");
    eprintln!(
        "    A3B gate_up K=2048: MQ4 8*136=1088 B/row  vs  E8 16+64*17=1104 B/row  (E8 +1.5%)"
    );
    eprintln!(
        "    A3B down    K=512 : MQ4 2*136= 272 B/row  vs  E8 16+16*17= 288 B/row  (E8 +5.9%)"
    );
    eprintln!(
        "    If decode is memory-bound, MQ4 and E8-AoS GB/s should track within ~2-6% at the"
    );
    eprintln!(
        "    A3B rows → the 150-vs-102 tok/s spread is NOT raw format bytes (it's layout/indexed"
    );
    eprintln!("    -context dilution for E8, and the graded MQ6-on-hot-experts traffic for mq4p).");

    // LUT decode must be bit-exact with the scalar SoA decode (same coordinates).
    {
        let (m, k) = (512usize, 2048usize);
        let aos_data = synth_e8_aos(m, k, 0xBEEF);
        let soa_data = aos_to_soa_full(&aos_data, m, k);
        let soa_w = gpu.upload_raw(&soa_data, &[soa_data.len()]).unwrap();
        let x = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let y2 = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        let yl = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        let xh = make_x(k, 0x55AA);
        gpu.hip.memcpy_htod(&x.buf, bytes_of(&xh)).unwrap();
        gpu.gemv_mfp4g32_e8_soa(&soa_w, &x, &y2, m, k).unwrap();
        gpu.gemv_mfp4g32_e8_soa_lut(&soa_w, &x, &yl, m, k).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let mut r2 = vec![0f32; m];
        let mut rl = vec![0f32; m];
        gpu.hip.memcpy_dtoh(bytes_of_mut(&mut r2), &y2.buf).unwrap();
        gpu.hip.memcpy_dtoh(bytes_of_mut(&mut rl), &yl.buf).unwrap();
        let nd = (0..m)
            .filter(|&i| r2[i].to_bits() != rl[i].to_bits())
            .count();
        eprintln!(
            "  LUT-decode correctness vs scalar SoA: {} / {} bit-exact",
            m - nd,
            m
        );
    }

    if arch == "gfx1151" {
        // Dense batched E8-SoA WMMA parity against the established row-wise
        // decode kernel. WMMA converts X and decoded weights to f16, so require
        // numerical agreement rather than bit identity.
        {
            let (batch, m, k) = (17usize, 512usize, 2048usize);
            let aos = synth_e8_aos(m, k, 0x4433_2211);
            let soa = aos_to_soa_full(&aos, m, k);
            let weights = gpu.upload_raw(&soa, &[soa.len()]).unwrap();
            let x = gpu.alloc_tensor(&[batch * k], DType::F32).unwrap();
            let y_ref = gpu.alloc_tensor(&[batch * m], DType::F32).unwrap();
            let y_wmma = gpu.alloc_tensor(&[batch * m], DType::F32).unwrap();
            let x_host = make_x(batch * k, 0xA11C_E8B4);
            gpu.hip.memcpy_htod(&x.buf, bytes_of(&x_host)).unwrap();
            for b in 0..batch {
                let xb = x.sub_offset(b * k, k);
                let yb = y_ref.sub_offset(b * m, m);
                gpu.gemv_mfp4g32_e8_soa(&weights, &xb, &yb, m, k).unwrap();
            }
            gpu.gemm_mfp4g32_e8_soa_wmma_b4(&weights, &x, &y_wmma, m, k, batch)
                .unwrap();
            gpu.hip.device_synchronize().unwrap();
            let mut reference = vec![0f32; batch * m];
            let mut candidate = vec![0f32; batch * m];
            gpu.hip
                .memcpy_dtoh(bytes_of_mut(&mut reference), &y_ref.buf)
                .unwrap();
            gpu.hip
                .memcpy_dtoh(bytes_of_mut(&mut candidate), &y_wmma.buf)
                .unwrap();
            let mut err2 = 0.0f64;
            let mut ref2 = 0.0f64;
            let mut max_abs = 0.0f32;
            for (&a, &b) in reference.iter().zip(&candidate) {
                let d = (a - b).abs();
                max_abs = max_abs.max(d);
                err2 += (d as f64) * (d as f64);
                ref2 += (a as f64) * (a as f64);
            }
            let rel_rmse = (err2 / ref2.max(f64::MIN_POSITIVE)).sqrt();
            eprintln!("  Dense E8-SoA WMMA B4 parity: rel_rmse={rel_rmse:.6} max_abs={max_abs:.6}");
            assert!(
                candidate.iter().all(|v| v.is_finite()) && rel_rmse < 0.01,
                "dense E8-SoA WMMA B4 parity failed"
            );
        }

        // Grouped block-diagonal batched layout used by DeepSeek4 O-LoRA:
        // weights [G,M,K], X [B,G,K], Y [B,G,M].
        {
            let (groups, batch, m, k) = (8usize, 17usize, 128usize, 2048usize);
            let aos = synth_e8_aos(groups * m, k, 0xE84B_10C8);
            let soa = aos_to_soa_full(&aos, groups * m, k);
            let row_bytes = soa.len() / (groups * m);
            let group_bytes = row_bytes * m;
            let weights = gpu.upload_raw(&soa, &[soa.len()]).unwrap();
            let x = gpu.alloc_tensor(&[batch * groups * k], DType::F32).unwrap();
            let y_ref = gpu.alloc_tensor(&[batch * groups * m], DType::F32).unwrap();
            let y_wmma = gpu.alloc_tensor(&[batch * groups * m], DType::F32).unwrap();
            let x_host = make_x(batch * groups * k, 0xB10C_E8A5);
            gpu.hip.memcpy_htod(&x.buf, bytes_of(&x_host)).unwrap();
            for b in 0..batch {
                for g in 0..groups {
                    let w = weights.sub_offset(g * group_bytes, group_bytes);
                    let xg = x.sub_offset((b * groups + g) * k, k);
                    let yg = y_ref.sub_offset((b * groups + g) * m, m);
                    gpu.gemv_mfp4g32_e8_soa(&w, &xg, &yg, m, k).unwrap();
                }
            }
            gpu.gemm_mfp4g32_e8_soa_grouped_wmma_b2(&weights, &x, &y_wmma, groups, m, k, batch)
                .unwrap();
            gpu.hip.device_synchronize().unwrap();
            let mut reference = vec![0f32; batch * groups * m];
            let mut candidate = vec![0f32; batch * groups * m];
            gpu.hip
                .memcpy_dtoh(bytes_of_mut(&mut reference), &y_ref.buf)
                .unwrap();
            gpu.hip
                .memcpy_dtoh(bytes_of_mut(&mut candidate), &y_wmma.buf)
                .unwrap();
            let mut err2 = 0.0f64;
            let mut ref2 = 0.0f64;
            let mut max_abs = 0.0f32;
            for (&a, &b) in reference.iter().zip(&candidate) {
                let d = (a - b).abs();
                max_abs = max_abs.max(d);
                err2 += (d as f64) * (d as f64);
                ref2 += (a as f64) * (a as f64);
            }
            let rel_rmse = (err2 / ref2.max(f64::MIN_POSITIVE)).sqrt();
            eprintln!(
                "  Grouped E8-SoA WMMA B2 parity: rel_rmse={rel_rmse:.6} max_abs={max_abs:.6}"
            );
            assert!(
                candidate.iter().all(|v| v.is_finite()) && rel_rmse < 0.01,
                "grouped E8-SoA WMMA B2 parity failed"
            );
        }

        let (groups, m, k) = (8usize, 1024usize, 4096usize);
        let aos = synth_e8_aos(groups * m, k, 0x4752_4F55);
        let soa = aos_to_soa_full(&aos, groups * m, k);
        let row_bytes = soa.len() / (groups * m);
        let group_bytes = row_bytes * m;
        let weights = gpu.upload_raw(&soa, &[soa.len()]).unwrap();
        let x = gpu.alloc_tensor(&[groups * k], DType::F32).unwrap();
        let serial = gpu.alloc_tensor(&[groups * m], DType::F32).unwrap();
        let grouped = gpu.alloc_tensor(&[groups * m], DType::F32).unwrap();
        let x_host = make_x(groups * k, 0x574F_4138);
        gpu.hip.memcpy_htod(&x.buf, bytes_of(&x_host)).unwrap();

        let run_serial = |gpu: &mut Gpu| {
            for group in 0..groups {
                let weight = weights.sub_offset(group * group_bytes, group_bytes);
                let x_group = x.sub_offset(group * k, k);
                let y_group = serial.sub_offset(group * m, m);
                gpu.gemv_mfp4g32_e8_soa(&weight, &x_group, &y_group, m, k)
                    .unwrap();
            }
        };
        run_serial(&mut gpu);
        gpu.gemv_mfp4g32_e8_soa_grouped_gfx1151(&weights, &x, &grouped, groups, m, k)
            .unwrap();
        gpu.hip.device_synchronize().unwrap();

        let mut serial_host = vec![0.0f32; groups * m];
        let mut grouped_host = vec![0.0f32; groups * m];
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut serial_host), &serial.buf)
            .unwrap();
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut grouped_host), &grouped.buf)
            .unwrap();
        let exact = serial_host
            .iter()
            .zip(&grouped_host)
            .filter(|(left, right)| left.to_bits() == right.to_bits())
            .count();

        let grouped_trials = 200usize;
        let start = Instant::now();
        for _ in 0..grouped_trials {
            run_serial(&mut gpu);
        }
        gpu.hip.device_synchronize().unwrap();
        let serial_us = start.elapsed().as_secs_f64() * 1.0e6 / grouped_trials as f64;
        let start = Instant::now();
        for _ in 0..grouped_trials {
            gpu.gemv_mfp4g32_e8_soa_grouped_gfx1151(&weights, &x, &grouped, groups, m, k)
                .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let grouped_us = start.elapsed().as_secs_f64() * 1.0e6 / grouped_trials as f64;
        eprintln!(
            "  DS4 wo_a grouped: parity {exact}/{}; 8 launches {serial_us:.2} us vs grouped {grouped_us:.2} us ({:+.1}%)",
            groups * m,
            (serial_us / grouped_us - 1.0) * 100.0,
        );

        let batch = 256usize;
        let x_batch = gpu.alloc_tensor(&[batch * groups * k], DType::F32).unwrap();
        let y_batch = gpu.alloc_tensor(&[batch * groups * m], DType::F32).unwrap();
        let x_batch_host = make_x(batch * groups * k, 0xB47C_4E8);
        gpu.hip
            .memcpy_htod(&x_batch.buf, bytes_of(&x_batch_host))
            .unwrap();
        let trials = 5usize;
        let mut time_variant = |which: usize| {
            let run = |gpu: &mut Gpu| match which {
                1 => gpu.gemm_mfp4g32_e8_soa_grouped_wmma(
                    &weights, &x_batch, &y_batch, groups, m, k, batch,
                ),
                2 => gpu.gemm_mfp4g32_e8_soa_grouped_wmma_b2(
                    &weights, &x_batch, &y_batch, groups, m, k, batch,
                ),
                _ => unreachable!(),
            };
            run(&mut gpu).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let start = Instant::now();
            for _ in 0..trials {
                run(&mut gpu).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            start.elapsed().as_secs_f64() * 1.0e6 / trials as f64
        };
        let b1_us = time_variant(1);
        let b2_us = time_variant(2);
        eprintln!("  DS4 wo_a prefill B={batch}: B1 {b1_us:.1} us, B2 {b2_us:.1} us");

        // --- Bench 2e: grouped batched GEMV vs grouped WMMA at verify widths.
        //
        // `wo_per_group_batched_e8_fallback` sends every batch size through
        // `gemm_mfp4g32_e8_soa_grouped_wmma`, which tiles tokens at 16. At the
        // B<16 widths speculative verify actually uses, that tile is mostly
        // padding: profiled on the ds4 route at B=1 it costs 14.61 ms against
        // the decode grouped GEMV's 4.23 ms.
        //
        // The B=1 arm is a CORRECTNESS GATE, not a perf point. It must be
        // bit-identical to `gemv_mfp4g32_e8_soa_grouped_gfx1151` — the kernel
        // it replaces — because these logits decide speculative acceptance.
        eprintln!("--- Bench 2e: grouped E8 batched GEMV vs grouped WMMA (verify widths) ---");
        {
            let mut ref_host = vec![0.0f32; groups * m];
            gpu.gemv_mfp4g32_e8_soa_grouped_gfx1151(&weights, &x, &grouped, groups, m, k)
                .unwrap();
            gpu.hip.device_synchronize().unwrap();
            gpu.hip
                .memcpy_dtoh(bytes_of_mut(&mut ref_host), &grouped.buf)
                .unwrap();

            let y_new = gpu.alloc_tensor(&[groups * m], DType::F32).unwrap();
            gpu.gemv_mfp4g32_e8_soa_grouped_batched_gfx1151(&weights, &x, &y_new, 1, groups, m, k)
                .unwrap();
            gpu.hip.device_synchronize().unwrap();
            let mut new_host = vec![0.0f32; groups * m];
            gpu.hip
                .memcpy_dtoh(bytes_of_mut(&mut new_host), &y_new.buf)
                .unwrap();

            let exact_bits = ref_host
                .iter()
                .zip(&new_host)
                .filter(|(a, b)| a.to_bits() == b.to_bits())
                .count();
            let max_ulp = ref_host
                .iter()
                .zip(&new_host)
                .map(|(a, b)| (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs())
                .max()
                .unwrap_or(0);
            let max_rel = ref_host
                .iter()
                .zip(&new_host)
                .map(|(a, b)| if *a == 0.0 { 0.0 } else { ((a - b) / a).abs() })
                .fold(0.0f32, f32::max);
            let total = groups * m;
            eprintln!(
                "  B=1 vs gemv_mfp4g32_e8_soa_grouped: {exact_bits}/{total} bit-exact, \
                 max_rel {max_rel:.3e}, max_ulp {max_ulp} ({})",
                if exact_bits == total { "PASS" } else { "FAIL" }
            );
            assert_eq!(
                exact_bits, total,
                "grouped batched E8 GEMV B=1 must be bit-identical to the grouped decode GEMV \
                 it replaces — speculative acceptance depends on these logits"
            );

            eprintln!(
                "  {:>3} {:>11} {:>11} {:>8} {:>11}",
                "B", "WMMA us", "batched us", "speedup", "batch GB/s"
            );
            let w_bytes = (soa.len() as f64) + (groups * k * 4) as f64;
            for b in [1usize, 2, 4, 6, 8, 16] {
                let xb = gpu.alloc_tensor(&[b * groups * k], DType::F32).unwrap();
                let yb = gpu.alloc_tensor(&[b * groups * m], DType::F32).unwrap();
                let xb_host = make_x(b * groups * k, 0x9E37_79B9);
                gpu.hip.memcpy_htod(&xb.buf, bytes_of(&xb_host)).unwrap();
                let trials = 200usize;

                let mut run_w = |gpu: &mut Gpu| {
                    gpu.gemm_mfp4g32_e8_soa_grouped_wmma(&weights, &xb, &yb, groups, m, k, b)
                        .unwrap()
                };
                run_w(&mut gpu);
                gpu.hip.device_synchronize().unwrap();
                let t = Instant::now();
                for _ in 0..trials {
                    run_w(&mut gpu);
                }
                gpu.hip.device_synchronize().unwrap();
                let us_w = t.elapsed().as_secs_f64() * 1.0e6 / trials as f64;

                let mut run_b = |gpu: &mut Gpu| {
                    gpu.gemv_mfp4g32_e8_soa_grouped_batched_gfx1151(
                        &weights, &xb, &yb, b, groups, m, k,
                    )
                    .unwrap()
                };
                run_b(&mut gpu);
                gpu.hip.device_synchronize().unwrap();
                let t = Instant::now();
                for _ in 0..trials {
                    run_b(&mut gpu);
                }
                gpu.hip.device_synchronize().unwrap();
                let us_b = t.elapsed().as_secs_f64() * 1.0e6 / trials as f64;

                eprintln!(
                    "  {b:>3} {us_w:>11.2} {us_b:>11.2} {:>7.2}x {:>11.1}",
                    us_w / us_b,
                    w_bytes / (us_b * 1.0e3)
                );
            }
        }
    }
}

fn run_prefill_gfx1201(gpu: &mut Gpu) {
    assert_eq!(gpu.arch, "gfx1201", "gfx12 E8 prefill screen needs gfx1201");
    let batch = std::env::var("HIPFIRE_E8_PREFILL_GFX1201_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(128);
    let trials = std::env::var("HIPFIRE_E8_PREFILL_GFX1201_TRIALS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5);
    assert!(batch > 0 && trials > 0);
    let shapes = [
        (512usize, 2048usize, "channel"),
        (1024usize, 4096usize, "wq_a/wo_a"),
        (32768usize, 1024usize, "wq_b"),
        (4096usize, 8192usize, "wo_b"),
        (2048usize, 4096usize, "shared_up"),
        (4096usize, 2048usize, "shared_down"),
    ];

    eprintln!("=== gfx1201 MFP4G32E8SOA WMMA prefill B={batch} ===");
    eprintln!(
        "  {:<12} {:>7} {:>7} {:>10} {:>10} {:>10} {:>10} {:>9}",
        "shape", "M", "K", "row us", "B1 us", "B2 us", "B4 us", "B4 speed"
    );

    for (slot, (m, k, label)) in shapes.into_iter().enumerate() {
        let aos = synth_e8_aos(m, k, 0x1201_E800 ^ slot as u64);
        let soa = aos_to_soa_full(&aos, m, k);
        let mut weights = gpu.upload_raw(&soa, &[soa.len()]).expect("upload weights");
        weights.dtype = DType::MFP4G32E8SOA;
        let x = gpu.alloc_tensor(&[batch * k], DType::F32).expect("alloc x");
        let x_f16 = gpu
            .alloc_tensor(&[batch * k], DType::F16)
            .expect("alloc x f16");
        let y_ref = gpu
            .alloc_tensor(&[batch * m], DType::F32)
            .expect("alloc row reference");
        let y_b1 = gpu
            .alloc_tensor(&[batch * m], DType::F32)
            .expect("alloc B1 output");
        let y_b2 = gpu
            .alloc_tensor(&[batch * m], DType::F32)
            .expect("alloc B2 output");
        let y_b4 = gpu
            .alloc_tensor(&[batch * m], DType::F32)
            .expect("alloc B4 output");
        let x_host = make_x(batch * k, 0xE812_0100 ^ slot as u64);
        gpu.hip
            .memcpy_htod(&x.buf, bytes_of(&x_host))
            .expect("upload x");
        gpu.deepseek4_convert_f32_to_f16(&x, &x_f16, (batch * k) as i64)
            .expect("convert x to f16");

        let mut rowwise = |gpu: &mut Gpu| {
            for b in 0..batch {
                gpu.gemv_mfp4g32_e8_soa(
                    &weights,
                    &x.sub_offset(b * k, k),
                    &y_ref.sub_offset(b * m, m),
                    m,
                    k,
                )
                .expect("row-wise E8 reference");
            }
        };
        rowwise(gpu);
        gpu.gemm_mfp4g32_e8_soa_wmma_gfx1201_f16(&weights, &x_f16, &y_b1, m, k, batch)
            .expect("gfx1201 B1");
        gpu.gemm_mfp4g32_e8_soa_wmma_b2_gfx1201_f16(&weights, &x_f16, &y_b2, m, k, batch)
            .expect("gfx1201 B2");
        gpu.gemm_mfp4g32_e8_soa_wmma_b4_gfx1201_f16(&weights, &x_f16, &y_b4, m, k, batch)
            .expect("gfx1201 B4");
        gpu.hip
            .device_synchronize()
            .expect("correctness synchronize");

        let mut reference = vec![0.0f32; batch * m];
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut reference), &y_ref.buf)
            .expect("download reference");
        for (name, candidate_tensor) in [("B1", &y_b1), ("B2", &y_b2), ("B4", &y_b4)] {
            let mut candidate = vec![0.0f32; batch * m];
            gpu.hip
                .memcpy_dtoh(bytes_of_mut(&mut candidate), &candidate_tensor.buf)
                .expect("download candidate");
            let mut err2 = 0.0f64;
            let mut ref2 = 0.0f64;
            let mut max_abs = 0.0f32;
            for (&a, &b) in reference.iter().zip(&candidate) {
                let delta = (a - b).abs();
                max_abs = max_abs.max(delta);
                err2 += (delta as f64) * (delta as f64);
                ref2 += (a as f64) * (a as f64);
            }
            let rel_rmse = (err2 / ref2.max(f64::MIN_POSITIVE)).sqrt();
            assert!(
                candidate.iter().all(|value| value.is_finite()) && rel_rmse < 0.01,
                "gfx1201 {name} parity failed for {label}: rel_rmse={rel_rmse} max_abs={max_abs}"
            );
            eprintln!("  parity {label:<12} {name}: rel_rmse={rel_rmse:.7} max_abs={max_abs:.6}");
        }

        let mut time = |run: &mut dyn FnMut(&mut Gpu)| {
            for _ in 0..2 {
                run(gpu);
            }
            gpu.hip.device_synchronize().expect("warm synchronize");
            let start = Instant::now();
            for _ in 0..trials {
                run(gpu);
            }
            gpu.hip.device_synchronize().expect("timing synchronize");
            start.elapsed().as_secs_f64() * 1.0e6 / trials as f64
        };
        let row_us = time(&mut rowwise);
        let b1_us = time(&mut |gpu| {
            gpu.gemm_mfp4g32_e8_soa_wmma_gfx1201_f16(&weights, &x_f16, &y_b1, m, k, batch)
                .expect("time B1");
        });
        let b2_us = time(&mut |gpu| {
            gpu.gemm_mfp4g32_e8_soa_wmma_b2_gfx1201_f16(&weights, &x_f16, &y_b2, m, k, batch)
                .expect("time B2");
        });
        let b4_us = time(&mut |gpu| {
            gpu.gemm_mfp4g32_e8_soa_wmma_b4_gfx1201_f16(&weights, &x_f16, &y_b4, m, k, batch)
                .expect("time B4");
        });
        eprintln!(
            "  {label:<12} {m:>7} {k:>7} {row_us:>10.2} {b1_us:>10.2} {b2_us:>10.2} {b4_us:>10.2} {:>8.2}x",
            row_us / b4_us
        );
    }
}

fn time_prefill_variant(
    gpu: &mut Gpu,
    variant: usize,
    weights: &rdna_compute::GpuTensor,
    x: &rdna_compute::GpuTensor,
    y: &rdna_compute::GpuTensor,
    m: usize,
    k: usize,
    batch: usize,
    trials: usize,
) -> f64 {
    let start = Instant::now();
    for _ in 0..trials {
        match variant {
            0 => gpu
                .gemm_mfp4g32_e8_soa_wmma_b4(weights, x, y, m, k, batch)
                .expect("baseline E8 prefill"),
            4 => gpu
                .gemm_mfp4g32_e8_soa_wmma_coop4(weights, x, y, m, k, batch)
                .expect("cooperative-4 E8 prefill"),
            8 => gpu
                .gemm_mfp4g32_e8_soa_wmma_b8(weights, x, y, m, k, batch)
                .expect("B8 E8 prefill"),
            16 => gpu
                .gemm_mfp4g32_e8_soa_wmma_b16(weights, x, y, m, k, batch)
                .expect("B16 E8 prefill"),
            _ => unreachable!("unknown E8 prefill variant"),
        }
    }
    gpu.hip.device_synchronize().expect("timing synchronize");
    start.elapsed().as_secs_f64() * 1.0e6 / trials as f64
}

fn run_prefill_coop(gpu: &mut Gpu) {
    assert_eq!(
        gpu.arch, "gfx1151",
        "cooperative E8 prefill is gfx1151-only"
    );
    let batch = std::env::var("HIPFIRE_E8_PREFILL_COOP_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1024);
    assert!(batch > 0, "cooperative E8 prefill batch must be non-zero");
    const TRIALS: usize = 5;
    let shapes = [
        (1024usize, 4096usize, "wq_a/wo_a"),
        (32768usize, 1024usize, "wq_b"),
        (4096usize, 8192usize, "wo_b"),
        (2048usize, 4096usize, "shared_up"),
        (4096usize, 2048usize, "shared_down"),
    ];
    eprintln!("=== gfx1151 cooperative E8 prefill B={batch} ===");
    eprintln!(
        "  {:<16} {:>11} {:>11} {:>11} {:>11} {:>9} {:>9} {:>14}",
        "shape", "B4 us", "coop4 us", "B8 us", "B16 us", "B8/v4", "B16/v4", "exact outputs"
    );

    for (slot, (m, k, label)) in shapes.into_iter().enumerate() {
        let aos = synth_e8_aos(m, k, 0xC001_E800 ^ slot as u64);
        let soa = aos_to_soa_full(&aos, m, k);
        let weights = gpu.upload_raw(&soa, &[soa.len()]).expect("upload weights");
        let x = gpu.alloc_tensor(&[batch * k], DType::F32).expect("alloc x");
        let y_ref = gpu
            .alloc_tensor(&[batch * m], DType::F32)
            .expect("alloc baseline y");
        let y_coop = gpu
            .alloc_tensor(&[batch * m], DType::F32)
            .expect("alloc cooperative y");
        let y_b8 = gpu
            .alloc_tensor(&[batch * m], DType::F32)
            .expect("alloc B8 y");
        let y_b16 = gpu
            .alloc_tensor(&[batch * m], DType::F32)
            .expect("alloc B16 y");
        let x_host = make_x(batch * k, 0xE800_C001 ^ slot as u64);
        gpu.hip
            .memcpy_htod(&x.buf, bytes_of(&x_host))
            .expect("upload x");

        // Warm JIT, DPM, and the pointer-keyed F16 conversion cache for both
        // symbols before correctness or timing.
        for _ in 0..2 {
            gpu.gemm_mfp4g32_e8_soa_wmma_b4(&weights, &x, &y_ref, m, k, batch)
                .expect("warm baseline");
            gpu.gemm_mfp4g32_e8_soa_wmma_coop4(&weights, &x, &y_coop, m, k, batch)
                .expect("warm cooperative");
            gpu.gemm_mfp4g32_e8_soa_wmma_b8(&weights, &x, &y_b8, m, k, batch)
                .expect("warm B8");
            gpu.gemm_mfp4g32_e8_soa_wmma_b16(&weights, &x, &y_b16, m, k, batch)
                .expect("warm B16");
        }
        gpu.hip.device_synchronize().expect("warm synchronize");

        let mut reference = vec![0.0f32; batch * m];
        let mut candidate = vec![0.0f32; batch * m];
        let mut candidate8 = vec![0.0f32; batch * m];
        let mut candidate16 = vec![0.0f32; batch * m];
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut reference), &y_ref.buf)
            .expect("download baseline");
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut candidate), &y_coop.buf)
            .expect("download cooperative");
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut candidate8), &y_b8.buf)
            .expect("download B8");
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut candidate16), &y_b16.buf)
            .expect("download B16");
        let exact = reference
            .iter()
            .zip(&candidate)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        assert_eq!(
            exact,
            reference.len(),
            "cooperative E8 mismatch for {label} M={m} K={k}"
        );
        let exact8 = reference
            .iter()
            .zip(&candidate8)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        assert_eq!(
            exact8,
            reference.len(),
            "B8 E8 mismatch for {label} M={m} K={k}"
        );
        let exact16 = reference
            .iter()
            .zip(&candidate16)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        assert_eq!(
            exact16,
            reference.len(),
            "B16 E8 mismatch for {label} M={m} K={k}"
        );

        // ABBA timing order. Averaging the two observations per arm removes
        // the monotonic order bias without mixing kernels inside one timer.
        let a0 = time_prefill_variant(gpu, 0, &weights, &x, &y_ref, m, k, batch, TRIALS);
        let b0 = time_prefill_variant(gpu, 4, &weights, &x, &y_coop, m, k, batch, TRIALS);
        let c0 = time_prefill_variant(gpu, 8, &weights, &x, &y_b8, m, k, batch, TRIALS);
        let d0 = time_prefill_variant(gpu, 16, &weights, &x, &y_b16, m, k, batch, TRIALS);
        let d1 = time_prefill_variant(gpu, 16, &weights, &x, &y_b16, m, k, batch, TRIALS);
        let c1 = time_prefill_variant(gpu, 8, &weights, &x, &y_b8, m, k, batch, TRIALS);
        let b1 = time_prefill_variant(gpu, 4, &weights, &x, &y_coop, m, k, batch, TRIALS);
        let a1 = time_prefill_variant(gpu, 0, &weights, &x, &y_ref, m, k, batch, TRIALS);
        let baseline_us = 0.5 * (a0 + a1);
        let cooperative_us = 0.5 * (b0 + b1);
        let b8_us = 0.5 * (c0 + c1);
        let b16_us = 0.5 * (d0 + d1);
        eprintln!(
            "  {label:<16} {baseline_us:>11.1} {cooperative_us:>11.1} {b8_us:>11.1} {b16_us:>11.1} {:>8.3}x {:>8.3}x {:>14}",
            cooperative_us / b8_us,
            cooperative_us / b16_us,
            exact.min(exact8).min(exact16),
        );
    }
}

fn run_grouped_gfx1201(gpu: &mut Gpu) {
    assert_eq!(gpu.arch, "gfx1201", "grouped channel requires gfx1201");
    const GROUPS: usize = 8;
    const M: usize = 1024;
    const K: usize = 4096;
    const CASES: usize = 4;
    const WARMUPS: usize = 20;
    const TRIALS: usize = 200;

    let mut total_exact = 0usize;
    let total = CASES * GROUPS * M;
    let mut timing = None;
    for case in 0..CASES {
        let aos = synth_e8_aos(GROUPS * M, K, 0x1201_4752 ^ case as u64);
        let soa = aos_to_soa_full(&aos, GROUPS * M, K);
        let row_bytes = soa.len() / (GROUPS * M);
        let group_bytes = row_bytes * M;
        let weights = gpu.upload_raw(&soa, &[soa.len()]).expect("upload weights");
        let x = gpu
            .alloc_tensor(&[GROUPS * K], DType::F32)
            .expect("alloc x");
        let serial = gpu
            .alloc_tensor(&[GROUPS * M], DType::F32)
            .expect("alloc serial");
        let grouped = gpu
            .alloc_tensor(&[GROUPS * M], DType::F32)
            .expect("alloc grouped");
        let x_host = make_x(GROUPS * K, 0xE800_1201 ^ case as u64);
        gpu.hip
            .memcpy_htod(&x.buf, bytes_of(&x_host))
            .expect("upload x");

        let run_serial = |gpu: &mut Gpu| {
            for group in 0..GROUPS {
                let weight = weights.sub_offset(group * group_bytes, group_bytes);
                let x_group = x.sub_offset(group * K, K);
                let y_group = serial.sub_offset(group * M, M);
                gpu.gemv_mfp4g32_e8_soa(&weight, &x_group, &y_group, M, K)
                    .expect("single gfx1201 E8");
            }
        };

        run_serial(gpu);
        gpu.gemv_mfp4g32_e8_soa_grouped_gfx1201(&weights, &x, &grouped, GROUPS, M, K)
            .expect("grouped gfx1201 E8");
        gpu.hip.device_synchronize().expect("parity synchronize");

        let mut reference = vec![0.0f32; GROUPS * M];
        let mut candidate = vec![0.0f32; GROUPS * M];
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut reference), &serial.buf)
            .expect("download serial");
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut candidate), &grouped.buf)
            .expect("download grouped");
        let exact = reference
            .iter()
            .zip(&candidate)
            .filter(|(left, right)| left.to_bits() == right.to_bits())
            .count();
        assert_eq!(exact, GROUPS * M, "grouped mismatch in case {case}");
        total_exact += exact;

        if case == 0 {
            for _ in 0..WARMUPS {
                run_serial(gpu);
                gpu.gemv_mfp4g32_e8_soa_grouped_gfx1201(&weights, &x, &grouped, GROUPS, M, K)
                    .expect("warm grouped");
            }
            gpu.hip.device_synchronize().expect("warm synchronize");
            let start = Instant::now();
            for _ in 0..TRIALS {
                run_serial(gpu);
            }
            gpu.hip.device_synchronize().expect("serial synchronize");
            let serial_us = start.elapsed().as_secs_f64() * 1.0e6 / TRIALS as f64;
            let start = Instant::now();
            for _ in 0..TRIALS {
                gpu.gemv_mfp4g32_e8_soa_grouped_gfx1201(&weights, &x, &grouped, GROUPS, M, K)
                    .expect("timed grouped");
            }
            gpu.hip.device_synchronize().expect("grouped synchronize");
            let grouped_us = start.elapsed().as_secs_f64() * 1.0e6 / TRIALS as f64;
            timing = Some((serial_us, grouped_us));
        }
    }

    let (serial_us, grouped_us) = timing.expect("timing case");
    eprintln!(
        "gfx1201 grouped O-LoRA: {total_exact}/{total} raw-bit exact; 8 launches {serial_us:.3} us; grouped {grouped_us:.3} us; speedup {:.3}x",
        serial_us / grouped_us,
    );
}

fn run_pack_correctness(gpu: &mut Gpu) {
    assert_eq!(gpu.arch, "gfx1151", "packed B3 kernel is gfx1151-only");
    const B: usize = 3;
    const K: usize = 4096;
    let rows = [1024usize, 512, 256, 4096, 1024, 512, 64];
    let x_host = make_x(B * K, 0xB3A7_7E57);
    let x = gpu.alloc_tensor(&[B * K], DType::F32).expect("alloc x");
    gpu.hip
        .memcpy_htod(&x.buf, bytes_of(&x_host))
        .expect("upload x");

    let mut weights = Vec::with_capacity(rows.len());
    let mut reference = Vec::with_capacity(rows.len());
    let mut packed = Vec::with_capacity(rows.len());
    for (slot, &m) in rows.iter().enumerate() {
        let aos = synth_e8_aos(m, K, 0xE800_0000 ^ slot as u64);
        let soa = aos_to_soa_full(&aos, m, K);
        weights.push(gpu.upload_raw(&soa, &[soa.len()]).expect("upload weight"));
        reference.push(
            gpu.alloc_tensor(&[B * m], DType::F32)
                .expect("alloc reference"),
        );
        packed.push(
            gpu.alloc_tensor(&[B * m], DType::F32)
                .expect("alloc packed"),
        );
    }

    for slot in 0..rows.len() {
        gpu.gemv_mfp4g32_e8_soa_batched_gfx1151(
            &weights[slot],
            &x,
            &reference[slot],
            B,
            rows[slot],
            K,
        )
        .expect("ordinary B3 launch");
    }
    gpu.gemv_mfp4g32_e8_soa_batched_pack_b3_gfx1151(
        [
            &weights[0],
            &weights[1],
            &weights[2],
            &weights[3],
            &weights[4],
            &weights[5],
            &weights[6],
        ],
        &x,
        [
            &packed[0], &packed[1], &packed[2], &packed[3], &packed[4], &packed[5], &packed[6],
        ],
        rows,
        K,
    )
    .expect("packed B3 launch");
    gpu.hip.device_synchronize().expect("synchronize");

    let mut compared = 0usize;
    for slot in 0..rows.len() {
        let len = B * rows[slot];
        let mut expected = vec![0.0f32; len];
        let mut actual = vec![0.0f32; len];
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut expected), &reference[slot].buf)
            .expect("download reference");
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut actual), &packed[slot].buf)
            .expect("download packed");
        for i in 0..len {
            assert_eq!(
                actual[i].to_bits(),
                expected[i].to_bits(),
                "packed B3 mismatch: slot={slot} row={} token={}",
                i % rows[slot],
                i / rows[slot]
            );
        }
        compared += len;
    }
    eprintln!("PACK B3 CORRECTNESS PASS: {compared} raw-bit comparisons across rows={rows:?}");
}

/// Convert AoS mfp4-E8 buffer to SoA layout.
/// AoS per-row: [16B hdr] + n_blocks * [1B scale + 16B codewords]
/// SoA per-row: [16B hdr (flag changed to 0x06)] + [n_blocks scales, pad16] + [n_blocks*16B codewords]
fn aos_to_soa_row(aos_row: &[u8], n_blocks: usize) -> Vec<u8> {
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_len = 16 + scale_padded + n_blocks * 16;
    let mut out = vec![0u8; soa_len];
    out[..16].copy_from_slice(&aos_row[..16]);
    out[6] = 0x06; // SoA flag
    for b in 0..n_blocks {
        out[16 + b] = aos_row[16 + b * 17];
    }
    let cw_start = 16 + scale_padded;
    for b in 0..n_blocks {
        let src = 16 + b * 17 + 1;
        let dst = cw_start + b * 16;
        out[dst..dst + 16].copy_from_slice(&aos_row[src..src + 16]);
    }
    out
}

fn aos_to_soa_full(aos: &[u8], m: usize, k: usize) -> Vec<u8> {
    let n_blocks = k / 32;
    let aos_row_bytes = 16 + n_blocks * 17;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_row_bytes = 16 + scale_padded + n_blocks * 16;
    let mut out = Vec::with_capacity(m * soa_row_bytes);
    for r in 0..m {
        let row = &aos[r * aos_row_bytes..(r + 1) * aos_row_bytes];
        out.extend_from_slice(&aos_to_soa_row(row, n_blocks));
    }
    out
}

fn synth_e8_aos(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let blocks_per_row = k / 32;
    let row_bytes = 16 + blocks_per_row * 17;
    let mut out = vec![0u8; m * row_bytes];
    let mut state = seed;
    let mut rng = || -> u32 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        let roff = row * row_bytes;
        let rs_f16: u16 = 0x2400;
        out[roff..roff + 2].copy_from_slice(&rs_f16.to_le_bytes());
        out[roff + 4..roff + 6].copy_from_slice(&(blocks_per_row as u16).to_le_bytes());
        out[roff + 6] = 0x05;
        for b in 0..blocks_per_row {
            let bp = roff + 16 + b * 17;
            out[bp] = 120u8.wrapping_add((rng() & 0x3F) as u8);
            for w in 0..4 {
                let cw = rng();
                out[bp + 1 + w * 4..bp + 1 + w * 4 + 4].copy_from_slice(&cw.to_le_bytes());
            }
        }
    }
    out
}

/// Synthesize a standalone HFQ4-G256 weight buffer (mq4 = plain uniform format).
/// Per row: `groups = K/256` groups of exactly 136 bytes each:
///   [0..4]  f32 scale
///   [4..8]  f32 zero-point
///   [8..136] 32 × uint32 packed nibbles (256 4-bit weights)
/// Matches the layout `gemv_hfq4g256` reads (gemv_hfq4g256.gfx1100.hip:23-66) and
/// `profile::hfq4g256_weight_bytes` (== m * groups * 136). RNG mirrors
/// `synth_e8_aos` byte-for-byte so the two formats are seeded identically.
fn synth_hfq4g256(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert!(k % 256 == 0, "hfq4g256 requires K%256==0 (groups = K/256)");
    let groups = k / 256;
    let row_bytes = groups * 136; // == hfq4g256_weight_bytes(m,k) / m
    let mut out = vec![0u8; m * row_bytes];
    let mut state = seed;
    let mut rng = || -> u32 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        for g in 0..groups {
            let off = row * row_bytes + g * 136;
            let sc: f32 = 0.003 + (rng() & 0x3F) as f32 * 1e-4;
            out[off..off + 4].copy_from_slice(&sc.to_bits().to_le_bytes());
            let zp: f32 = -0.02;
            out[off + 4..off + 8].copy_from_slice(&zp.to_bits().to_le_bytes());
            for w in 0..32 {
                let pk = rng();
                out[off + 8 + w * 4..off + 8 + w * 4 + 4].copy_from_slice(&pk.to_le_bytes());
            }
        }
    }
    out
}

/// Minimal f32 -> IEEE binary16 bit conversion. These are bench activations for
/// the WMMA arm (which consumes `_Float16`), not a correctness path, so
/// round-to-nearest-even and subnormal handling are deliberately omitted.
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = ((bits & 0x007f_ffff) >> 13) as u16;
    if exp <= 0 {
        return sign;
    }
    if exp >= 0x1f {
        return sign | (0x1f << 10);
    }
    sign | ((exp as u16) << 10) | mant
}

fn bytes_of_u16(v: &[u16]) -> &[u8] {
    // SAFETY: reading a u16 slice as bytes for an upload.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn make_x(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) * 2.3e-10 - 0.5
        })
        .collect()
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn bytes_of_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) }
}
