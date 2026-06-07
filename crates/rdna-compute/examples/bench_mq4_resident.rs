//! Cache-resident MQ4 / MQ4-Lloyd GEMV microbench.
//!
//! This is intentionally not a full-model benchmark. It repeatedly launches the
//! same small synthetic matrices so the weight payload can stay resident after
//! warmup. The "logical GiB/s" column is therefore a kernel-throughput metric,
//! not a DDR bandwidth claim.

use rdna_compute::{DType, Gpu, GpuTensor, LLOYD_MQ4_GROUP_BYTES};

const HFQ4_GROUP_BYTES: usize = 136;
const GROUP: usize = 256;

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    m: usize,
    k: usize,
    iters: usize,
}

fn f32_to_f16_le(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    let h = if exp == 0xff {
        (sign << 15) | (0x1f << 10) | if mant != 0 { 0x200 } else { 0 }
    } else {
        let half_exp = exp - 127 + 15;
        if half_exp <= 0 {
            sign << 15
        } else if half_exp >= 31 {
            (sign << 15) | (0x1f << 10)
        } else {
            let round_bits = mant & 0x1fff;
            let mut half_mant = (mant >> 13) as u16;
            if round_bits > 0x1000 || (round_bits == 0x1000 && (half_mant & 1) != 0) {
                half_mant += 1;
            }
            let mut half_exp = half_exp as u16;
            if half_mant == 0x400 {
                half_mant = 0;
                half_exp += 1;
            }
            (sign << 15) | (half_exp << 10) | half_mant
        }
    };
    h.to_le_bytes()
}

fn build_hfq4(m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % GROUP, 0);
    let groups = k / GROUP;
    let mut out = Vec::with_capacity(m * groups * HFQ4_GROUP_BYTES);
    for row in 0..m {
        for g in 0..groups {
            let scale = 0.015625f32 + ((row ^ g) & 3) as f32 * 0.0005;
            let zero = -0.125f32 + ((row.wrapping_mul(3) + g) & 7) as f32 * 0.001;
            out.extend_from_slice(&scale.to_le_bytes());
            out.extend_from_slice(&zero.to_le_bytes());
            for i in 0usize..128 {
                let lo = ((row.wrapping_mul(13) ^ g.wrapping_mul(17) ^ i) & 0xf) as u8;
                let hi =
                    ((row.wrapping_mul(29) ^ g.wrapping_mul(7) ^ i.wrapping_mul(3)) & 0xf) as u8;
                out.push(lo | (hi << 4));
            }
        }
    }
    out
}

fn build_mq4_lloyd(m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % GROUP, 0);
    let groups = k / GROUP;
    let mut out = Vec::with_capacity(m * groups * LLOYD_MQ4_GROUP_BYTES);
    for row in 0..m {
        for g in 0..groups {
            let base = ((row.wrapping_mul(5) + g.wrapping_mul(11)) % 17) as f32 * 0.002 - 0.02;
            for c in 0..16 {
                let v = base + (c as f32 - 7.5) * 0.0175;
                out.extend_from_slice(&f32_to_f16_le(v));
            }
            for i in 0usize..128 {
                let lo =
                    ((row.wrapping_mul(19) ^ g.wrapping_mul(23) ^ i.wrapping_mul(5)) & 0xf) as u8;
                let hi =
                    ((row.wrapping_mul(31) ^ g.wrapping_mul(3) ^ i.wrapping_mul(9)) & 0xf) as u8;
                out.push(lo | (hi << 4));
            }
        }
    }
    out
}

fn build_x(k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| ((i as i32 % 23) as f32 - 11.0) * 0.0078125)
        .collect()
}

fn time_us(gpu: &mut Gpu, iters: usize, mut f: impl FnMut(&mut Gpu)) -> f32 {
    for _ in 0..16 {
        f(gpu);
    }
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..iters {
        f(gpu);
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    gpu.hip.event_destroy(start).unwrap();
    gpu.hip.event_destroy(stop).unwrap();
    ms * 1000.0 / iters as f32
}

fn logical_gib_s(bytes_per_call: usize, us: f32) -> f64 {
    bytes_per_call as f64 / (us as f64 * 1e-6) / (1024.0 * 1024.0 * 1024.0)
}

fn run_one(
    gpu: &mut Gpu,
    shape: Shape,
    hfq4: &GpuTensor,
    lloyd: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
) -> (f32, f64, f32, f64) {
    let hfq4_us = time_us(gpu, shape.iters, |gpu| {
        gpu.gemv_mq4g256_prerotated(hfq4, x, y, shape.m, shape.k)
            .unwrap();
    });
    let lloyd_us = time_us(gpu, shape.iters, |gpu| {
        gpu.gemv_mq4g256_lloyd(lloyd, x, y, shape.m, shape.k)
            .unwrap();
    });

    let groups = shape.k / GROUP;
    let hfq4_bytes = shape.m * groups * HFQ4_GROUP_BYTES + shape.k * 4 + shape.m * 4;
    let lloyd_bytes = shape.m * groups * LLOYD_MQ4_GROUP_BYTES + shape.k * 4 + shape.m * 4;
    (
        hfq4_us,
        logical_gib_s(hfq4_bytes, hfq4_us),
        lloyd_us,
        logical_gib_s(lloyd_bytes, lloyd_us),
    )
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    let shapes = [
        Shape {
            name: "expert-ish",
            m: 512,
            k: 2048,
            iters: 1000,
        },
        Shape {
            name: "l2-small",
            m: 2048,
            k: 2048,
            iters: 500,
        },
        Shape {
            name: "l2-large",
            m: 4096,
            k: 4096,
            iters: 250,
        },
        Shape {
            name: "over-l2-ish",
            m: 12288,
            k: 4096,
            iters: 100,
        },
    ];

    eprintln!("=== bench_mq4_resident ===");
    eprintln!("GPU: {}", gpu.arch);
    eprintln!(
        "Each row is one 32-thread GEMV workgroup; repeated launches reuse the same buffers."
    );
    eprintln!(
        "{:<12} {:>9} {:>8} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "shape", "M", "K", "hfq4_us", "hfq4_GiB", "lloyd_us", "lloyd_GiB", "ratio"
    );

    for shape in shapes {
        let hfq4_host = build_hfq4(shape.m, shape.k);
        let lloyd_host = build_mq4_lloyd(shape.m, shape.k);
        let x_host = build_x(shape.k);
        let hfq4 = gpu.upload_raw(&hfq4_host, &[hfq4_host.len()]).unwrap();
        let lloyd = gpu.upload_raw(&lloyd_host, &[lloyd_host.len()]).unwrap();
        let x = gpu.upload_f32(&x_host, &[shape.k]).unwrap();
        let y = gpu.zeros(&[shape.m], DType::F32).unwrap();

        let (hfq4_us, hfq4_gib, lloyd_us, lloyd_gib) =
            run_one(&mut gpu, shape, &hfq4, &lloyd, &x, &y);
        eprintln!(
            "{:<12} {:>9} {:>8} {:>9.2} {:>9.1} {:>9.2} {:>9.1} {:>8.2}",
            shape.name,
            shape.m,
            shape.k,
            hfq4_us,
            hfq4_gib,
            lloyd_us,
            lloyd_gib,
            lloyd_us / hfq4_us,
        );

        gpu.free_tensor(hfq4).unwrap();
        gpu.free_tensor(lloyd).unwrap();
        gpu.free_tensor(x).unwrap();
        gpu.free_tensor(y).unwrap();
    }
}
