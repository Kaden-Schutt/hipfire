// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1201 DeepSeek-V4 dense tensor-parallel compute screen.
//
// This is a channel bench, not a product route. It compares the full replicated
// qt35 work used by DS4 EP with the natural four-rank Megatron shard for each
// projection family. Every arm cycles a >192 MiB weight set so Infinity Cache
// residency cannot manufacture a win. Communication is intentionally excluded:
// combine these compute numbers with the measured collective separately before
// deciding whether loader/forward-path TP work is justified.

use rdna_compute::{DType, Gpu, GpuTensor};

const DRAM_SET_BYTES: usize = 192 * 1024 * 1024;
const TRIALS: usize = 7;

#[derive(Clone, Copy)]
struct Arm {
    m: usize,
    k: usize,
    jobs: usize,
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    full: Arm,
    shard: Arm,
    layers: usize,
    needs_new_collective: bool,
}

const CASES: &[Case] = &[
    Case {
        name: "q_heads: wq_b column shard",
        full: Arm {
            m: 32768,
            k: 1024,
            jobs: 1,
        },
        shard: Arm {
            m: 8192,
            k: 1024,
            jobs: 1,
        },
        layers: 43,
        needs_new_collective: false,
    },
    Case {
        name: "o_lora: 8 groups -> 2 groups/rank",
        full: Arm {
            m: 1024,
            k: 2048,
            jobs: 8,
        },
        shard: Arm {
            m: 1024,
            k: 2048,
            jobs: 2,
        },
        layers: 43,
        needs_new_collective: false,
    },
    Case {
        name: "attn wo_b row shard",
        full: Arm {
            m: 4096,
            k: 8192,
            jobs: 1,
        },
        shard: Arm {
            m: 4096,
            k: 2048,
            jobs: 1,
        },
        layers: 43,
        needs_new_collective: true,
    },
    Case {
        name: "shared w1+w3 column shard",
        full: Arm {
            m: 2048,
            k: 4096,
            jobs: 2,
        },
        shard: Arm {
            m: 512,
            k: 4096,
            jobs: 2,
        },
        layers: 43,
        needs_new_collective: false,
    },
    Case {
        name: "shared w2 row shard",
        full: Arm {
            m: 4096,
            k: 2048,
            jobs: 1,
        },
        shard: Arm {
            m: 4096,
            k: 512,
            jobs: 1,
        },
        layers: 43,
        needs_new_collective: false,
    },
];

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn row_bytes(k: usize) -> usize {
    assert_eq!(k % 256, 0, "qt35 TP screen requires K divisible by 256");
    let blocks = k / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    16 + scale_padded + blocks * 16
}

fn build_e8_soa(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let blocks = k / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    let stride = row_bytes(k);
    let mut packed = vec![0u8; m * stride];
    let mut state = seed;
    for row in 0..m {
        let off = row * stride;
        let row_scale = [0x3400u16, 0x3800, 0x3c00, 0x4000][row & 3];
        packed[off..off + 2].copy_from_slice(&row_scale.to_le_bytes());
        packed[off + 4..off + 6].copy_from_slice(&(blocks as u16).to_le_bytes());
        packed[off + 6] = 0x06;
        for block in 0..blocks {
            packed[off + 16 + block] = [0x01, 0x07, 0x38, 0x7f][block & 3];
            let codeword_off = off + 16 + scale_padded + block * 16;
            for slot in 0..4 {
                let codeword = lcg(&mut state);
                packed[codeword_off + slot * 4..codeword_off + slot * 4 + 4]
                    .copy_from_slice(&codeword.to_le_bytes());
            }
        }
    }
    packed
}

fn make_x(k: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..k)
        .map(|i| match i & 15 {
            0 => 0.0,
            1 => -0.0,
            2 => 0.125,
            3 => -0.25,
            _ => (lcg(&mut state) as f32 / u32::MAX as f32 - 0.5) * 0.25,
        })
        .collect()
}

fn upload_weight(gpu: &Gpu, packed: &[u8]) -> GpuTensor {
    let mut weight = gpu
        .upload_raw(packed, &[packed.len()])
        .expect("upload qt35 weight");
    weight.dtype = DType::MFP4G32E8SOA;
    weight
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

fn time_arm(gpu: &mut Gpu, arm: Arm, ordinal: usize) -> (f64, usize, usize) {
    let bytes_per_weight = arm.m * row_bytes(arm.k);
    let mut replicas = DRAM_SET_BYTES.div_ceil(bytes_per_weight).max(arm.jobs * 2);
    replicas = replicas.div_ceil(arm.jobs) * arm.jobs;
    let weights: Vec<GpuTensor> = (0..replicas)
        .map(|replica| {
            let packed = build_e8_soa(
                arm.m,
                arm.k,
                0x1201_0000_0000_0000u64 ^ ((ordinal as u64) << 32) ^ replica as u64,
            );
            upload_weight(gpu, &packed)
        })
        .collect();
    let x = gpu
        .upload_f32(&make_x(arm.k, 0x1201_cafe ^ ordinal as u64), &[arm.k])
        .expect("upload x");
    let y = gpu.alloc_tensor(&[arm.m], DType::F32).expect("alloc y");
    let workloads = replicas / arm.jobs;

    for weight in &weights {
        gpu.gemv_mfp4g32_e8_soa(weight, &x, &y, arm.m, arm.k)
            .expect("warm qt35 GEMV");
    }
    gpu.hip.device_synchronize().expect("warm sync");

    let start = gpu.hip.event_create().expect("create start event");
    let stop = gpu.hip.event_create().expect("create stop event");
    let mut samples = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        gpu.hip.event_record(&start, None).expect("record start");
        for weight in &weights {
            gpu.gemv_mfp4g32_e8_soa(weight, &x, &y, arm.m, arm.k)
                .expect("timed qt35 GEMV");
        }
        gpu.hip.event_record(&stop, None).expect("record stop");
        gpu.hip.event_synchronize(&stop).expect("wait stop");
        let total_us = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed") as f64 * 1000.0;
        samples.push(total_us / workloads as f64);
    }
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    (median(&mut samples), replicas, replicas * bytes_per_weight)
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1201", "this screen requires exact gfx1201");
    println!(
        "DS4 dense TP compute screen: arch={} dram_set_mib={} trials={TRIALS}",
        gpu.arch,
        DRAM_SET_BYTES / (1024 * 1024)
    );
    println!(
        "{:42} {:>10} {:>10} {:>9} {:>12} {:>8}",
        "case", "full_us", "shard_us", "speedup", "save_ms/tok", "new_AR"
    );

    let mut total_saved_ms = 0.0;
    let mut new_collectives = 0usize;
    for (ordinal, case) in CASES.iter().enumerate() {
        let (full_us, full_reps, full_bytes) = time_arm(&mut gpu, case.full, ordinal * 2);
        let (shard_us, shard_reps, shard_bytes) = time_arm(&mut gpu, case.shard, ordinal * 2 + 1);
        let saved_ms = (full_us - shard_us) * case.layers as f64 / 1000.0;
        total_saved_ms += saved_ms;
        if case.needs_new_collective {
            new_collectives += case.layers;
        }
        println!(
            "{:42} {:10.3} {:10.3} {:9.3} {:12.4} {:8}",
            case.name,
            full_us,
            shard_us,
            full_us / shard_us,
            saved_ms,
            if case.needs_new_collective {
                "yes"
            } else {
                "no"
            }
        );
        println!(
            "  full M={} K={} jobs={} reps={} set_mib={:.1}; shard M={} K={} jobs={} reps={} set_mib={:.1}",
            case.full.m,
            case.full.k,
            case.full.jobs,
            full_reps,
            full_bytes as f64 / 1048576.0,
            case.shard.m,
            case.shard.k,
            case.shard.jobs,
            shard_reps,
            shard_bytes as f64 / 1048576.0,
        );
    }

    println!(
        "COMPUTE_ONLY total_saved_ms_per_token={total_saved_ms:.4} new_collectives_per_token={new_collectives}"
    );
    println!(
        "SIZE WITH MEASURED COMM: net_saved_ms = compute_saved_ms - new_collectives * collective_us / 1000"
    );
}
