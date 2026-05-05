//! Stage 4 smoke: load qwen3.5 weights distributed across 2 GPUs via
//! `qwen35::load_weights_multi`. Verifies Variant 2 placement
//! (token_embd → dev 0, output_norm + lm_head → dev_last, per-layer
//! weights → gpus.device_for_layer(i)) using `pointer_get_attributes`.
//!
//! Run: HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
//!         -p hipfire-arch-qwen35 \
//!         --example test_qwen35_load_multi -- \
//!         ~/.hipfire/models/qwen3.5-0.8b.mq4

use hipfire_arch_qwen35::qwen35;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::multi_gpu::Gpus;
use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: ... <model.mq4>");
    let hfq = HfqFile::open(Path::new(&path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config_from_hfq");
    eprintln!(
        "config: {} layers, vocab={}, dim={}, hidden={}",
        config.n_layers, config.vocab_size, config.dim, config.hidden_dim,
    );

    let mut gpus = Gpus::init_uniform(2, config.n_layers).expect("init_uniform");
    let n = gpus.devices.len();
    let out_dev = gpus.output_device;
    println!(
        "Gpus: {n} devices, output_device={out_dev}, layer_to_device={:?}",
        gpus.layer_to_device,
    );

    println!("\n── load_weights_multi ────────────────────────────────────");
    let weights = qwen35::load_weights_multi(&hfq, &config, &mut gpus).expect("load_weights_multi");

    println!("\n── verify per-tensor device placement ───────────────────");
    let attr0 = gpus.devices[0]
        .hip
        .pointer_get_attributes(&weights.token_embd.buf)
        .expect("attr token_embd");
    println!("  token_embd: attr.device={} (expect 0)", attr0.device);
    assert_eq!(attr0.device, 0, "token_embd must live on dev 0");

    let attr_norm = gpus.devices[out_dev]
        .hip
        .pointer_get_attributes(&weights.output_norm.buf)
        .expect("attr output_norm");
    println!("  output_norm: attr.device={} (expect {out_dev})", attr_norm.device);
    assert_eq!(attr_norm.device, out_dev as i32, "output_norm must live on dev_last");

    let attr_out = gpus.devices[out_dev]
        .hip
        .pointer_get_attributes(&weights.output.buf.buf)
        .expect("attr output");
    println!("  output (lm_head): attr.device={} (expect {out_dev})", attr_out.device);
    assert_eq!(attr_out.device, out_dev as i32, "output must live on dev_last");

    let probe_layers = [0usize, config.n_layers / 2, config.n_layers - 1];
    for &i in &probe_layers {
        let dev_idx = gpus.device_for_layer(i);
        let attn_norm_buf = match &weights.layers[i] {
            qwen35::LayerWeights::DeltaNet(l) => &l.attn_norm,
            qwen35::LayerWeights::FullAttn(l) => &l.attn_norm,
            qwen35::LayerWeights::DeltaNetMoe(l) => &l.attn_norm,
            qwen35::LayerWeights::FullAttnMoe(l) => &l.attn_norm,
        };
        let attr = gpus.devices[dev_idx]
            .hip
            .pointer_get_attributes(&attn_norm_buf.buf)
            .expect("attr layer attn_norm");
        println!(
            "  layer {i}: dev_idx={dev_idx}, attn_norm.attr.device={} (expect {dev_idx})",
            attr.device
        );
        assert_eq!(
            attr.device, dev_idx as i32,
            "layer {i} attn_norm must live on dev {dev_idx}"
        );
    }

    println!("\n── enable_peer_all (post-load per ROCm gotcha) ──────────");
    let peer_ok = gpus.enable_peer_all().expect("enable_peer_all");
    assert!(peer_ok, "peer access bidirectional on 2× 7900 XTX");
    println!("  peer_access_enabled = true");

    println!("\n── free_gpu_multi ───────────────────────────────────────");
    weights.free_gpu_multi(&mut gpus);
    println!("  all weights freed on owning devices");

    println!("\ntest_qwen35_load_multi: PASS");
}
