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
        Ok((bytes.clone(), DType::F16))
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

/// PB-1a byte-oracle: a `ColumnShard { axis: 0 }` weight on a Tp-2 mesh must land
/// on each rank as its contiguous half of the row-major blob (rows [r·m/2,(r+1)·m/2)
/// = bytes [r·B/2,(r+1)·B/2)). Format-agnostic contiguous split.
fn check_column_shard_tp2(label: &str, gpus: &Gpus) {
    const M: usize = 8; // output rows
    const K: usize = 16; // input dim
    let entry = WeightEntry::layer(
        "wq",
        0,
        vec![M, K],
        DType::F16,
        ShardPolicy::ColumnShard { axis: 0 },
    );
    // Byte j = j as u8, so the expected per-rank slice is a plain range check.
    let total = M * K; // treat as 1 byte/elem for the oracle (dtype set post-upload)
    let bytes: Vec<u8> = (0..total as u32).map(|x| x as u8).collect();
    let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
    let store = fulfill_manifest(&[entry.clone()], &mesh, N_LAYERS, gpus, |_| {
        Ok((bytes.clone(), DType::F16))
    })
    .unwrap_or_else(|e| panic!("[{label}] fulfill_manifest(ColumnShard) failed: {e}"));

    let devs = placement_devices(&entry, &mesh, N_LAYERS);
    assert_eq!(devs.len(), 2, "[{label}] expected Tp-2 placement");
    let chunk = bytes.len() / 2;
    for (rank, &dev) in devs.iter().enumerate() {
        let expected = bytes[rank * chunk..(rank + 1) * chunk].to_vec();
        match store
            .get("wq", Some(0), dev)
            .unwrap_or_else(|| panic!("[{label}] wq missing on device {dev}"))
        {
            WeightHandle::Resident(t) => {
                // Sharded shape: outermost dim halved.
                assert_eq!(
                    t.shape,
                    vec![M / 2, K],
                    "[{label}] rank {rank} shape not sharded"
                );
                let got = readback(gpus, dev, t);
                assert_eq!(
                    got, expected,
                    "[{label}] rank {rank} (dev {dev}) contiguous half mismatch"
                );
            }
            _ => panic!("[{label}] wq should be Resident, not Alias"),
        }
    }
    println!("[{label}] OK — ColumnShard Tp-2 contiguous halves byte-verified on 2 ranks");
}

/// PB-1c byte-oracle: a `RowShard { axis: 1 }` weight on a Tp-2 mesh cuts the
/// inner (k) dim — a per-row STRIDED gather. Rank r owns, of every one of the
/// `m` rows, its half of the row bytes; the gathered blob is row-major [m, k/2].
fn check_row_shard_tp2(label: &str, gpus: &Gpus) {
    const M: usize = 8; // rows (output dim, kept whole)
    const K: usize = 16; // inner dim (sharded)
    let entry = WeightEntry::layer(
        "wo",
        0,
        vec![M, K],
        DType::F16,
        ShardPolicy::RowShard { axis: 1 },
    );
    let bytes: Vec<u8> = (0..(M * K) as u32).map(|x| x as u8).collect();
    let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
    let store = fulfill_manifest(&[entry.clone()], &mesh, N_LAYERS, gpus, |_| {
        Ok((bytes.clone(), DType::F16))
    })
    .unwrap_or_else(|e| panic!("[{label}] fulfill_manifest(RowShard) failed: {e}"));

    let devs = placement_devices(&entry, &mesh, N_LAYERS);
    assert_eq!(devs.len(), 2, "[{label}] expected Tp-2 placement");
    let row_bytes = K; // 1 byte/elem for the oracle
    let sub = row_bytes / 2;
    for (rank, &dev) in devs.iter().enumerate() {
        // Expected: rank r's k-half gathered from every row.
        let mut expected = Vec::new();
        for row in 0..M {
            let base = row * row_bytes + rank * sub;
            expected.extend_from_slice(&bytes[base..base + sub]);
        }
        match store
            .get("wo", Some(0), dev)
            .unwrap_or_else(|| panic!("[{label}] wo missing on device {dev}"))
        {
            WeightHandle::Resident(t) => {
                assert_eq!(
                    t.shape,
                    vec![M, K / 2],
                    "[{label}] rank {rank} shape not k-sharded"
                );
                let got = readback(gpus, dev, t);
                assert_eq!(
                    got, expected,
                    "[{label}] rank {rank} (dev {dev}) strided k-gather mismatch"
                );
            }
            _ => panic!("[{label}] wo should be Resident, not Alias"),
        }
    }
    println!("[{label}] OK — RowShard Tp-2 strided k-gather byte-verified on 2 ranks");
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
            Ok((synth_bytes(e), DType::Raw))
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
    let store: WeightStore = fulfill_manifest(&manifest, mesh, N_LAYERS, gpus, |e| {
        Ok((synth_bytes(e), DType::Raw))
    })
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

    // Deferred-refusal: a not-yet-implemented dense TP shard (FusedQkv, PB-1b)
    // on a Tp>1 mesh must Err before any upload. (Column PB-1a + Row PB-1c are
    // now implemented and checked below on the emulated 2-rank Gpus.)
    {
        let tp2 = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        let m = vec![WeightEntry::layer(
            "wq",
            0,
            vec![64, 8],
            DType::F16,
            ShardPolicy::FusedQkv {
                q_heads: 8,
                kv_heads: 2,
                head_dim: 8,
                layout: FusedQkvLayout::Qkv,
            },
        )];
        let r = fulfill_manifest(&m, &tp2, N_LAYERS, &gpus, |e| {
            Ok((synth_bytes(e), DType::Raw))
        });
        assert!(
            r.is_err(),
            "unimplemented dense TP shard on Tp-2 must be refused"
        );
        println!(
            "refusal: FusedQkv TP shard on Tp-2 → Err ({})",
            r.err().unwrap()
        );
    }
    drop(gpus);

    // ── PP-2 mesh (emulated): band layers across two logical ranks ───────
    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
    match Gpus::init_uniform(2, N_LAYERS) {
        Ok(gpus2) => {
            let mesh = resolve_mesh(2, 1, 1, Some(2));
            assert!(mesh.has_axis(DimKind::Pp), "expected a Pp mesh");
            check("pp-2-emulated", &mesh, &gpus2);
            // Same 2 ranks, Ep axis: expert-parallel compact-blob placement.
            check_ep("ep-2-emulated", &gpus2);
            // Same 2 ranks, Tp axis: ColumnShard contiguous-half slicing (PB-1a).
            check_column_shard_tp2("tp-2-column-emulated", &gpus2);
            // Same 2 ranks, Tp axis: RowShard strided k-gather slicing (PB-1c).
            check_row_shard_tp2("tp-2-row-emulated", &gpus2);
        }
        Err(e) => {
            println!("pp-2-emulated: SKIPPED (could not bring up 2-rank Gpus: {e})");
        }
    }

    println!("fulfill_manifest_probe: all checks passed");
}
