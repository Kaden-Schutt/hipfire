//! Dual-GPU smoke: Gpu::init_with_device + bind_thread + per-device
//! pointer ownership via hipPointerGetAttributes.
//!
//! Run: HIP_VISIBLE_DEVICES=0,1 cargo run -p rdna-compute --example dual_gpu_smoke

use rdna_compute::{DType, Gpu};

fn main() {
    let count = {
        let probe = hip_bridge::HipRuntime::load().expect("load HIP runtime");
        probe.device_count().expect("device_count")
    };
    println!("Visible devices: {count}");
    assert!(
        count >= 2,
        "dual_gpu_smoke requires ≥2 visible devices (got {count}). Set HIP_VISIBLE_DEVICES=0,1"
    );

    println!("\n── init Gpu on dev 0 + dev 1 ──────────────────────────────");
    let mut gpu0 = Gpu::init_with_device(0).expect("init dev 0");
    assert_eq!(gpu0.device_id, 0, "gpu0.device_id should be 0");
    let mut gpu1 = Gpu::init_with_device(1).expect("init dev 1");
    assert_eq!(gpu1.device_id, 1, "gpu1.device_id should be 1");
    println!("  gpu0: device_id={} arch={}", gpu0.device_id, gpu0.arch);
    println!("  gpu1: device_id={} arch={}", gpu1.device_id, gpu1.arch);
    assert_eq!(
        gpu0.arch, gpu1.arch,
        "this smoke test assumes homogeneous arch (got {} vs {})",
        gpu0.arch, gpu1.arch
    );

    println!("\n── bind_thread alternation ────────────────────────────────");
    gpu0.bind_thread().expect("bind 0");
    assert_eq!(gpu0.hip.current_device().unwrap(), 0, "after bind on gpu0");
    println!("  bind gpu0 → current_device=0");

    gpu1.bind_thread().expect("bind 1");
    assert_eq!(gpu1.hip.current_device().unwrap(), 1, "after bind on gpu1");
    println!("  bind gpu1 → current_device=1");

    gpu0.bind_thread().expect("bind 0 again");
    assert_eq!(gpu0.hip.current_device().unwrap(), 0, "after re-bind on gpu0");
    println!("  re-bind gpu0 → current_device=0 (cached path also exercised)");

    println!("\n── tensor allocation + pointer_get_attributes ─────────────");

    let vocab = 128usize;
    let dim = 64usize;
    let table0_data: Vec<f32> = (0..vocab * dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    gpu0.bind_thread().expect("bind 0 for upload");
    let table0 = gpu0
        .upload_f32(&table0_data, &[vocab, dim])
        .expect("upload table on dev 0");
    let attr_table0 = gpu0
        .hip
        .pointer_get_attributes(&table0.buf)
        .expect("ptr-attr table0");
    assert_eq!(attr_table0.device, 0, "table0 should live on dev 0");
    println!("  table0 (vocab={vocab}, dim={dim}) on dev 0 — attr.device={}", attr_table0.device);

    let out0 = gpu0.zeros(&[dim], DType::F32).expect("zeros out on dev 0");
    let attr_out0 = gpu0.hip.pointer_get_attributes(&out0.buf).expect("ptr-attr out0");
    assert_eq!(attr_out0.device, 0, "out0 should live on dev 0");

    // Different pattern so misroutes are visible in the assert_ne! below.
    let table1_data: Vec<f32> = (0..vocab * dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
        .collect();
    gpu1.bind_thread().expect("bind 1 for upload");
    let table1 = gpu1
        .upload_f32(&table1_data, &[vocab, dim])
        .expect("upload table on dev 1");
    let attr_table1 = gpu1
        .hip
        .pointer_get_attributes(&table1.buf)
        .expect("ptr-attr table1");
    assert_eq!(attr_table1.device, 1, "table1 should live on dev 1");
    println!("  table1 (vocab={vocab}, dim={dim}) on dev 1 — attr.device={}", attr_table1.device);

    let out1 = gpu1.zeros(&[dim], DType::F32).expect("zeros out on dev 1");
    let attr_out1 = gpu1.hip.pointer_get_attributes(&out1.buf).expect("ptr-attr out1");
    assert_eq!(attr_out1.device, 1, "out1 should live on dev 1");

    // ── embedding_lookup on each device ──────────────────────────────
    println!("\n── embedding_lookup on each device ────────────────────────");
    let token_id = 42u32;

    gpu0.bind_thread().expect("bind 0 for lookup");
    gpu0.embedding_lookup(&table0, &out0, token_id, dim)
        .expect("lookup dev 0");
    let out0_host = gpu0.download_f32(&out0).expect("download out0");
    let row0_expected = &table0_data[(token_id as usize) * dim..(token_id as usize + 1) * dim];
    assert_eq!(&out0_host, row0_expected, "dev 0 lookup row mismatch");
    println!("  dev 0: looked up token {token_id} — first 4 elements: {:?}", &out0_host[..4]);

    gpu1.bind_thread().expect("bind 1 for lookup");
    gpu1.embedding_lookup(&table1, &out1, token_id, dim)
        .expect("lookup dev 1");
    let out1_host = gpu1.download_f32(&out1).expect("download out1");
    let row1_expected = &table1_data[(token_id as usize) * dim..(token_id as usize + 1) * dim];
    assert_eq!(&out1_host, row1_expected, "dev 1 lookup row mismatch");
    println!("  dev 1: looked up token {token_id} — first 4 elements: {:?}", &out1_host[..4]);

    // The two rows must differ — if they accidentally come back identical
    // it would indicate dev_1 picked up dev_0's table (the multi-GPU
    // corruption signature this whole audit is set up to catch).
    assert_ne!(
        out0_host, out1_host,
        "dev 0 and dev 1 lookups returned identical rows — possible cross-device pointer mixup"
    );

    // ── Alternating allocations stress check ─────────────────────────
    // Tight loop: bind 0, alloc, bind 1, alloc, bind 0, alloc … verify
    // each allocation lands on the device we claimed it would. If
    // bind_thread silently fails to switch, attr.device would lag the
    // intended target.
    println!("\n── alternating malloc stress (32 iterations) ──────────────");
    for i in 0..32 {
        let target = if i % 2 == 0 { &mut gpu0 } else { &mut gpu1 };
        target.bind_thread().expect("bind in stress loop");
        let t = target
            .zeros(&[1024], DType::F32)
            .expect("alloc in stress loop");
        let attr = target.hip.pointer_get_attributes(&t.buf).expect("ptr-attr in loop");
        assert_eq!(
            attr.device, target.device_id,
            "stress iter {i}: alloc landed on dev {} but expected {}",
            attr.device, target.device_id
        );
        target.free_tensor(t).expect("free in stress loop");
    }
    println!("  all 32 alloc/free pairs landed on the correct device.");

    // ── Cleanup ──────────────────────────────────────────────────────
    gpu0.bind_thread().expect("bind 0 for cleanup");
    gpu0.free_tensor(table0).expect("free table0");
    gpu0.free_tensor(out0).expect("free out0");
    gpu1.bind_thread().expect("bind 1 for cleanup");
    gpu1.free_tensor(table1).expect("free table1");
    gpu1.free_tensor(out1).expect("free out1");

    println!("\ndual_gpu_smoke: PASS");
}
