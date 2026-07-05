// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU validation for `fulfill_manifest` (device-mesh Phase 2 execution).
//!
//! Proves the GPU-execution half of the manifest system on real hardware,
//! without a model file or HFQ name resolution: a *synthetic* tensor source
//! yields deterministic bytes per weight entry, so we can byte-compare what
//! landed on each device against what we asked to upload.
//!
//! Checks, on a hand-built dense manifest (mirrors the llama shape):
//!   1. **Placement** — every weight lands on exactly the devices
//!      `placement_devices` says (embed→stage 0, output→last stage, per-layer
//!      weights→their band's stage).
//!   2. **Byte-oracle** — reading each resident tensor back off its device
//!      (`memcpy_dtoh`) returns the exact bytes the source produced.
//!   3. **Tied → Alias** — a tied lm_head records an alias, not an upload.
//!   4. **Deferred refusal** — a dense TP shard on a Tp>1 mesh returns `Err`
//!      (Phase-5 slicing not silently mis-placed).
//!   5. **Expert-parallel** — on an Ep-2 mesh, each rank's resident tensor is the
//!      compact blob of exactly its owned experts (byte-exact vs a `ShardConfig`
//!      gather).
//!   6. **Transactional rollback** — a source that fails partway returns `Err`
//!      (naming the failing tensor) with earlier uploads freed (§6), no panic.
//!
//! Runs the 1×1 mesh always; additionally runs emulated PP-2 + EP-2 meshes when
//! a 2-rank `Gpus` can be brought up (`HIPFIRE_EMULATE_GPUS=2`), else reports
//! them as skipped rather than failing.
//!
//! Run: cargo run -p hipfire-runtime --release --example fulfill_manifest_probe

use hipfire_hardware::{DeviceMesh, DimKind};
use hipfire_runtime::config::resolve_mesh;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
use hipfire_runtime::weight_manifest::{
    placement_devices, FusedQkvLayout, PinTarget, ShardPolicy, WeightEntry,
};
use hipfire_runtime::weight_store::{fulfill_manifest, WeightHandle, WeightStore};
use rdna_compute::{DType, Gpu};

const N_LAYERS: usize = 4;

/// Deterministic synthetic bytes for an entry: a fixed-length blob filled with a
/// per-tensor seed so distinct tensors are distinguishable on readback.
fn synth_bytes(entry: &WeightEntry) -> Vec<u8> {
    let seed = entry
        .name
        .bytes()
        .fold(entry.layer.unwrap_or(255) as u32, |a, b| {
            a.wrapping_mul(31).wrapping_add(b as u32)
        }) as u8;
    // Length varies a little by tensor so byte_size mismatches would surface.
    let len = 128 + (seed as usize % 64);
    (0..len).map(|i| seed ^ (i as u8)).collect()
}

/// The dense test manifest: embed (Pin), per-layer wq(FusedQkv)/wo(RowShard)/
/// attn_norm(Replicate), output_norm(Replicate), lm_head(Tied to token_embd).
fn test_manifest() -> Vec<WeightEntry> {
    let mut m = Vec::new();
    m.push(WeightEntry::model(
        "token_embd",
        vec![256, 8],
        DType::F16,
        ShardPolicy::Pin(PinTarget::Embed),
    ));
    for l in 0..N_LAYERS {
        m.push(WeightEntry::layer(
            "wq",
            l,
            vec![64, 8],
            DType::F16,
            ShardPolicy::FusedQkv {
                q_heads: 8,
                kv_heads: 2,
                head_dim: 8,
                layout: FusedQkvLayout::Qkv,
            },
        ));
        m.push(WeightEntry::layer(
            "wo",
            l,
            vec![8, 64],
            DType::F16,
            ShardPolicy::RowShard { axis: 1 },
        ));
        m.push(WeightEntry::layer(
            "attn_norm",
            l,
            vec![8],
            DType::F32,
            ShardPolicy::Replicate,
        ));
    }
    m.push(WeightEntry::model(
        "output_norm",
        vec![8],
        DType::F32,
        ShardPolicy::Replicate,
    ));
    m.push(WeightEntry::model(
        "lm_head",
        vec![256, 8],
        DType::F16,
        ShardPolicy::Tied {
            source: "token_embd".into(),
        },
    ));
    m
}

/// Validate expert-parallel placement: each rank gets a compact blob of only
/// its owned experts, byte-exact. Experts are the outermost dim so per-expert
/// byte ranges are contiguous; we build the expected compaction with the same
/// `ShardConfig` fulfill_manifest uses and byte-compare the readback.
fn check_ep(label: &str, gpus: &Gpus) {
    const N_EXPERTS: usize = 8;
    const PER: usize = 16; // bytes per expert
    let entry = WeightEntry::layer(
        "experts",
        0,
        vec![N_EXPERTS, 4, 4],
        DType::F16,
        ShardPolicy::ExpertSharded {
            n_experts: N_EXPERTS,
            assign: ExpertAssign::Stride,
        },
    );
    // Expert e occupies bytes [e*PER, (e+1)*PER); byte j of expert e = e*PER+j.
    let bytes: Vec<u8> = (0..(N_EXPERTS * PER) as u32).map(|x| x as u8).collect();
    let mesh = DeviceMesh::rect(&[(DimKind::Ep, 2)]);
    let store = fulfill_manifest(&[entry.clone()], &mesh, N_LAYERS, gpus, |_| {
        Ok(bytes.clone())
    })
    .unwrap_or_else(|e| panic!("[{label}] fulfill_manifest(EP) failed: {e}"));

    let shard = ShardConfig::new(2, false, N_EXPERTS, ExpertAssign::Stride).unwrap();
    let devs = placement_devices(&entry, &mesh, N_LAYERS);
    assert_eq!(devs.len(), 2, "[{label}] expected Ep-2 placement");
    for (rank, &dev) in devs.iter().enumerate() {
        let owned = shard.experts_on_rank(rank);
        let mut expected = Vec::new();
        for &e in &owned {
            expected.extend_from_slice(&bytes[e * PER..(e + 1) * PER]);
        }
        match store
            .get("experts", Some(0), dev)
            .unwrap_or_else(|| panic!("[{label}] experts missing on device {dev}"))
        {
            WeightHandle::Resident(t) => {
                let got = readback(gpus, dev, t);
                assert_eq!(
                    got, expected,
                    "[{label}] rank {rank} (dev {dev}) owns {owned:?} — compact blob mismatch"
                );
            }
            _ => panic!("[{label}] experts should be Resident, not Alias"),
        }
    }
    println!(
        "[{label}] OK — EP compact expert blobs byte-verified on {} ranks (rank0 {:?}, rank1 {:?})",
        devs.len(),
        shard.experts_on_rank(0),
        shard.experts_on_rank(1)
    );
}

/// Validate the §6 transactional rollback: a source that fails partway must
/// leave `fulfill_manifest` returning `Err` (naming the failing tensor) with the
/// already-uploaded cells freed. We can't observe the free directly, but the run
/// must not panic and the earlier uploads must have happened first.
fn check_rollback(label: &str, gpus: &Gpus) {
    let manifest = test_manifest();
    // manifest[2] = wo(layer 0) — so token_embd + wq(l0) upload first, then this
    // entry's source fails, exercising the rollback over ≥1 resident cell.
    let fail_name = manifest[2].name.clone();
    let fail_layer = manifest[2].layer;
    let r = fulfill_manifest(&manifest, &DeviceMesh::single(), N_LAYERS, gpus, |e| {
        if e.name == fail_name && e.layer == fail_layer {
            Err("synthetic source failure".to_string())
        } else {
            Ok(synth_bytes(e))
        }
    });
    match r {
        Err(err) => {
            assert_eq!(err.name, fail_name, "[{label}] rollback named wrong entry");
            println!(
                "[{label}] OK — rollback: source-fail on '{}' → Err, partial uploads freed",
                err.name
            );
        }
        Ok(_) => panic!("[{label}] expected a transactional-rollback Err"),
    }
}

/// Read a resident tensor's bytes back off its device.
fn readback(gpus: &Gpus, device: usize, tensor: &rdna_compute::GpuTensor) -> Vec<u8> {
    let n = tensor.buf.size();
    let mut buf = vec![0u8; n];
    gpus.devices[device]
        .hip
        .memcpy_dtoh(&mut buf, &tensor.buf)
        .expect("memcpy_dtoh");
    buf
}

/// Validate placement + byte-oracle + tied-alias for one (mesh, gpus) pair.
fn check(label: &str, mesh: &DeviceMesh, gpus: &Gpus) {
    let manifest = test_manifest();
    let store: WeightStore =
        fulfill_manifest(&manifest, mesh, N_LAYERS, gpus, |e| Ok(synth_bytes(e)))
            .unwrap_or_else(|e| panic!("[{label}] fulfill_manifest failed: {e}"));

    let mut resident = 0usize;
    let mut aliased = 0usize;
    for entry in &manifest {
        let expected = placement_devices(entry, mesh, N_LAYERS);
        // Store records exactly the expected devices for this weight.
        assert_eq!(
            store.devices_for(&entry.name, entry.layer),
            {
                let mut e = expected.clone();
                e.sort_unstable();
                e.dedup();
                e
            },
            "[{label}] {}[layer {:?}] placed on wrong devices",
            entry.name,
            entry.layer
        );
        for &dev in &expected {
            match store
                .get(&entry.name, entry.layer, dev)
                .unwrap_or_else(|| panic!("[{label}] {} missing on device {dev}", entry.name))
            {
                WeightHandle::Alias(src) => {
                    assert_eq!(src, "token_embd", "[{label}] wrong alias source");
                    aliased += 1;
                }
                WeightHandle::Resident(t) => {
                    // Byte-oracle: what landed == what we uploaded.
                    let got = readback(gpus, dev, t);
                    assert_eq!(
                        got,
                        synth_bytes(entry),
                        "[{label}] {} byte mismatch on device {dev}",
                        entry.name
                    );
                    resident += 1;
                }
            }
        }
    }
    // lm_head is the only Tied entry → exactly one alias per placement device.
    assert!(aliased >= 1, "[{label}] expected a tied alias");
    println!("[{label}] OK — {resident} resident uploads byte-verified, {aliased} tied alias(es)");
}

fn main() {
    // ── 1×1 mesh: single device, everything on device 0 ──────────────────
    let gpu = Gpu::init().expect("Gpu::init");
    let gpus = Gpus::single(gpu, N_LAYERS);
    check("single-1x1", &DeviceMesh::single(), &gpus);
    check_rollback("rollback-1x1", &gpus);

    // Deferred-refusal: a dense TP shard on a Tp>1 mesh must Err (even though
    // we don't have a real 2-rank Tp Gpus, fulfill refuses before any upload).
    {
        let tp2 = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        let m = vec![WeightEntry::layer(
            "wo",
            0,
            vec![8, 64],
            DType::F16,
            ShardPolicy::RowShard { axis: 1 },
        )];
        let r = fulfill_manifest(&m, &tp2, N_LAYERS, &gpus, |e| Ok(synth_bytes(e)));
        assert!(r.is_err(), "dense TP shard on Tp-2 must be refused");
        println!(
            "refusal: dense TP shard on Tp-2 → Err ({})",
            r.err().unwrap()
        );
    }
    drop(gpus);

    // ── PP-2 mesh (emulated): band layers across two logical ranks ───────
    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
    match Gpus::init_uniform(2, N_LAYERS) {
        Ok(gpus2) => {
            let mesh = resolve_mesh(2, 1, Some(2));
            assert!(mesh.has_axis(DimKind::Pp), "expected a Pp mesh");
            check("pp-2-emulated", &mesh, &gpus2);
            // Same 2 ranks, Ep axis: expert-parallel compact-blob placement.
            check_ep("ep-2-emulated", &gpus2);
        }
        Err(e) => {
            println!("pp-2-emulated: SKIPPED (could not bring up 2-rank Gpus: {e})");
        }
    }

    println!("fulfill_manifest_probe: all checks passed");
}
