// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1201 TP4 graph host-overhead micro-screen.

use std::time::Instant;

use hipfire_hardware::Gpus;

fn percentile(mut values: Vec<f64>, quantile: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * quantile).round() as usize]
}

fn main() {
    const RANKS: usize = 4;
    const WARMUPS: usize = 32;
    const REPEATS: usize = 512;

    let device_opts = hipfire_runtime::config::get().device_resolve_opts();
    let gpus = Gpus::init_tp(&device_opts, RANKS, 43).expect("init exact gfx1201 TP4");
    assert!(
        gpus.devices
            .iter()
            .all(|device| device.arch_caps.is_gfx1201()),
        "this micro requires four exact gfx1201 devices"
    );
    let mut signals = Vec::with_capacity(RANKS);
    for rank in 0..RANKS {
        let gpu = &gpus.devices[rank];
        gpu.bind_thread().expect("bind allocation rank");
        signals.push(gpu.hip.malloc_signal(8).expect("allocate signal"));
    }

    for _ in 0..WARMUPS {
        for rank in 0..RANKS {
            let gpu = &gpus.devices[rank];
            gpu.bind_thread().expect("bind warm reset rank");
            gpu.hip.memset(&signals[rank], 0, 8).expect("warm reset");
        }
        for rank in 0..RANKS {
            let gpu = &gpus.devices[rank];
            gpu.bind_thread().expect("bind warm sync rank");
            gpu.hip.device_synchronize().expect("warm sync");
        }
    }

    let mut reset_us = Vec::with_capacity(REPEATS);
    let mut sync4_us = Vec::with_capacity(REPEATS);
    let mut sync1_us = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let start = Instant::now();
        for rank in 0..RANKS {
            let gpu = &gpus.devices[rank];
            gpu.bind_thread().expect("bind timed reset rank");
            gpu.hip.memset(&signals[rank], 0, 8).expect("timed reset");
        }
        reset_us.push(start.elapsed().as_secs_f64() * 1.0e6);

        let start = Instant::now();
        for rank in 0..RANKS {
            let gpu = &gpus.devices[rank];
            gpu.bind_thread().expect("bind timed sync rank");
            gpu.hip.device_synchronize().expect("timed sync");
        }
        sync4_us.push(start.elapsed().as_secs_f64() * 1.0e6);

        let start = Instant::now();
        gpus.devices[0]
            .bind_thread()
            .expect("bind timed rank-zero sync");
        gpus.devices[0]
            .hip
            .device_synchronize()
            .expect("timed rank-zero sync");
        sync1_us.push(start.elapsed().as_secs_f64() * 1.0e6);
    }

    println!(
        "MICRO ranks={RANKS} repeats={REPEATS} reset4_us_median={:.3} reset4_us_p95={:.3} \
         empty_sync4_us_median={:.3} empty_sync4_us_p95={:.3} \
         empty_sync1_us_median={:.3} empty_sync1_us_p95={:.3}",
        percentile(reset_us.clone(), 0.5),
        percentile(reset_us, 0.95),
        percentile(sync4_us.clone(), 0.5),
        percentile(sync4_us, 0.95),
        percentile(sync1_us.clone(), 0.5),
        percentile(sync1_us, 0.95),
    );

    for rank in 0..RANKS {
        let gpu = &gpus.devices[rank];
        gpu.bind_thread().expect("bind free rank");
        gpu.hip.free(signals.remove(0)).expect("free signal");
    }
}
