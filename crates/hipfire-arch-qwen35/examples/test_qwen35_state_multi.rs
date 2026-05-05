//! Stage 4 smoke: build the full multi-GPU state stack on a real model:
//! `Qwen35ScratchSet` (per-device), `KvCache::new_gpu_asym3_capped_multi`
//! (per-layer K/V on band-owning device + per-device givens replicas),
//! `DeltaNetState::new_with_quant_multi` (per-LA-layer state on owning
//! device). Verifies each per-layer / per-device buffer's
//! `pointer_get_attributes` matches the layer-band assignment, then runs
//! a HipRuntime bind-gap regression battery.
//!
//! Run: HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
//!         -p hipfire-arch-qwen35 \
//!         --example test_qwen35_state_multi -- \
//!         ~/.hipfire/models/qwen3.5-0.8b.mq4

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, LayerType, Qwen35ScratchSet, StateQuant};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::multi_gpu::Gpus;
use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: ... <model.mq4>");
    let hfq = HfqFile::open(Path::new(&path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config_from_hfq");
    eprintln!(
        "config: {} layers (head_dim={}, n_kv_heads={}, vocab={})",
        config.n_layers, config.head_dim, config.n_kv_heads, config.vocab_size,
    );

    let mut gpus = Gpus::init_uniform(2, config.n_layers).expect("init_uniform");
    let n_dev = gpus.devices.len();
    let out_dev = gpus.output_device;
    println!(
        "Gpus: {n_dev} devices, output_device={out_dev}, layer_to_device={:?}",
        gpus.layer_to_device,
    );

    println!("\n── Qwen35ScratchSet::new_with_kv_max_multi ───────────────");
    let scratch_set = Qwen35ScratchSet::new_with_kv_max_multi(&mut gpus, &config, 64, 4096)
        .expect("ScratchSet");
    assert_eq!(scratch_set.per_device.len(), n_dev);
    for (dev_idx, scratch) in scratch_set.per_device.iter().enumerate() {
        let attr = gpus.devices[dev_idx]
            .hip
            .pointer_get_attributes(&scratch.x.buf)
            .expect("attr scratch.x");
        println!("  per_device[{dev_idx}].x: attr.device={} (expect {dev_idx})", attr.device);
        assert_eq!(attr.device, dev_idx as i32);
    }

    println!("\n── KvCache::new_gpu_asym3_capped_multi ───────────────────");
    let kv = KvCache::new_gpu_asym3_capped_multi(
        &mut gpus,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        4096,
        4096,
    )
    .expect("KvCache asym3 multi");
    assert_eq!(kv.k_gpu.len(), config.n_layers);
    let probe_layers = [0usize, config.n_layers / 2, config.n_layers - 1];
    for &i in &probe_layers {
        let dev_idx = gpus.device_for_layer(i);
        let attr_k = gpus.devices[dev_idx]
            .hip
            .pointer_get_attributes(&kv.k_gpu[i].buf)
            .expect("attr kv k");
        let attr_v = gpus.devices[dev_idx]
            .hip
            .pointer_get_attributes(&kv.v_gpu[i].buf)
            .expect("attr kv v");
        println!(
            "  layer {i}: dev_idx={dev_idx}, k.device={}, v.device={}",
            attr_k.device, attr_v.device
        );
        assert_eq!(attr_k.device, dev_idx as i32, "kv.k_gpu[{i}] off-device");
        assert_eq!(attr_v.device, dev_idx as i32, "kv.v_gpu[{i}] off-device");
    }

    println!("\n── givens_*_per_dev replicas ─────────────────────────────");
    assert_eq!(gpus.givens_cos_per_dev.len(), n_dev);
    assert_eq!(gpus.givens_sin_per_dev.len(), n_dev);
    for dev_idx in 0..n_dev {
        let attr_cos = gpus.devices[dev_idx]
            .hip
            .pointer_get_attributes(&gpus.givens_cos_per_dev[dev_idx].buf)
            .expect("attr givens cos");
        let attr_sin = gpus.devices[dev_idx]
            .hip
            .pointer_get_attributes(&gpus.givens_sin_per_dev[dev_idx].buf)
            .expect("attr givens sin");
        println!(
            "  givens_cos_per_dev[{dev_idx}].device={}, sin.device={}",
            attr_cos.device, attr_sin.device,
        );
        assert_eq!(attr_cos.device, dev_idx as i32);
        assert_eq!(attr_sin.device, dev_idx as i32);
    }

    println!("\n── DeltaNetState::new_with_quant_multi ───────────────────");
    let (dn, la_to_device) =
        DeltaNetState::new_with_quant_multi(&mut gpus, &config, StateQuant::Q8).expect("DN multi");
    let n_la = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::LinearAttention)
        .count();
    assert_eq!(dn.s_matrices.len(), n_la);
    assert_eq!(la_to_device.len(), n_la);
    println!("  n_la={n_la}, la_to_device={:?}", la_to_device);
    let probe_la = [0usize, n_la / 2, n_la - 1];
    for &i in &probe_la {
        let dev_idx = la_to_device[i] as usize;
        let attr = gpus.devices[dev_idx]
            .hip
            .pointer_get_attributes(&dn.s_matrices[i].buf)
            .expect("attr dn s_matrix");
        println!("  la {i}: dev_idx={dev_idx}, s_matrix.device={}", attr.device);
        assert_eq!(attr.device, dev_idx as i32);
    }

    println!("\n── enable_peer_all (post-allocation per ROCm gotcha) ─────");
    let peer_ok = gpus.enable_peer_all().expect("enable_peer_all");
    assert!(peer_ok);
    println!("  peer_access_enabled = true");

    println!("\n── free everything on owning devices ─────────────────────");
    dn.free_gpu_multi(&mut gpus, &la_to_device);
    kv.free_gpu_multi(&mut gpus);
    scratch_set.free_gpu_multi(&mut gpus);
    println!("  all state freed");

    println!("\n══ HipRuntime bind-gap regression checks ══════════════════");
    // Stage 2 finding (carried into Stage 4): gpu.hip.* (HipRuntime methods)
    // bypass the bind_thread audit. In multi-GPU contexts a raw hipMalloc/
    // memset/memcpy_htod lands on whatever device the host thread last
    // bound, not necessarily gpu.device_id. The fix in
    // DeltaNetState::new_with_quant_multi was an explicit g.bind_thread()
    // before the raw HIP ops. These checks deliberately misbind the
    // "wrong" device before each multi ctor call so any future regression
    // (a contributor removing the explicit bind) trips the device assert.

    println!("\n  [1/3] DeltaNetState::new_with_quant_multi Q8 — first LA on dev 0");
    gpus.devices[1].bind_thread().expect("misbind dev 1");
    let (dn2, la_to_device2) =
        DeltaNetState::new_with_quant_multi(&mut gpus, &config, StateQuant::Q8)
            .expect("DN regression");
    let attr = gpus.devices[0]
        .hip
        .pointer_get_attributes(&dn2.s_matrices[0].buf)
        .expect("attr dn2 s_matrix[0]");
    assert_eq!(
        attr.device, 0,
        "REGRESSION: DeltaNetState s_matrix[0] expected on dev 0 \
         but landed on dev {} after pre-misbinding dev 1. \
         Likely a removed bind_thread in new_with_quant_multi.",
        attr.device,
    );
    println!("    s_matrices[0].device={} OK", attr.device);
    dn2.free_gpu_multi(&mut gpus, &la_to_device2);

    println!("  [2/3] KvCache::new_gpu_asym3_capped_multi — layer 0 on dev 0");
    gpus.devices[1].bind_thread().expect("misbind dev 1");
    let kv2 = KvCache::new_gpu_asym3_capped_multi(
        &mut gpus,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        4096,
        4096,
    )
    .expect("KvCache regression");
    let attr_k = gpus.devices[0]
        .hip
        .pointer_get_attributes(&kv2.k_gpu[0].buf)
        .expect("attr kv2 k_gpu[0]");
    assert_eq!(
        attr_k.device, 0,
        "REGRESSION: KvCache k_gpu[0] expected on dev 0 but landed on dev {} \
         after pre-misbinding dev 1. Likely a removed bind in new_gpu_asym3_capped_multi.",
        attr_k.device,
    );
    println!("    k_gpu[0].device={} OK", attr_k.device);
    let attr_givens = gpus.devices[0]
        .hip
        .pointer_get_attributes(&gpus.givens_cos_per_dev[0].buf)
        .expect("attr givens dev 0");
    assert_eq!(
        attr_givens.device, 0,
        "REGRESSION: givens_cos_per_dev[0] expected on dev 0 after misbind+ctor",
    );
    println!("    givens_cos_per_dev[0].device={} OK", attr_givens.device);
    kv2.free_gpu_multi(&mut gpus);

    println!("  [3/3] Qwen35ScratchSet::new_with_kv_max_multi — per_device[0] on dev 0");
    gpus.devices[1].bind_thread().expect("misbind dev 1");
    let scratch_set2 = Qwen35ScratchSet::new_with_kv_max_multi(&mut gpus, &config, 64, 4096)
        .expect("ScratchSet regression");
    let attr = gpus.devices[0]
        .hip
        .pointer_get_attributes(&scratch_set2.per_device[0].x.buf)
        .expect("attr scratch_set2 per_device[0].x");
    assert_eq!(
        attr.device, 0,
        "REGRESSION: ScratchSet per_device[0].x expected on dev 0 but landed on dev {} \
         after pre-misbinding dev 1.",
        attr.device,
    );
    println!("    per_device[0].x.device={} OK", attr.device);
    let attr_pos = gpus.devices[0]
        .hip
        .pointer_get_attributes(&scratch_set2.per_device[0].pos_buf)
        .expect("attr scratch_set2 pos_buf");
    assert_eq!(
        attr_pos.device, 0,
        "REGRESSION: ScratchSet per_device[0].pos_buf (raw gpu.hip.malloc) expected on dev 0 \
         but landed on dev {} after pre-misbinding dev 1.",
        attr_pos.device,
    );
    println!(
        "    per_device[0].pos_buf.device={} OK (raw gpu.hip.malloc honored bind)",
        attr_pos.device
    );
    scratch_set2.free_gpu_multi(&mut gpus);

    println!("\ntest_qwen35_state_multi: PASS");
}
