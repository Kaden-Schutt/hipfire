//! Proves `Gpus::from_mesh` reproduces the exact layer layout of the per-axis
//! primitive it delegates to. Run under emulation (no real multi-GPU needed):
//!
//!   HIPFIRE_EMULATE_GPUS=4 cargo run --release --example gpus_from_mesh_parity
//!
//! Exit 0 = all three axes byte-identical; panics on any mismatch.

use hipfire_hardware::{DeviceMesh, DimKind, Gpus};

fn assert_same(label: &str, a: &Gpus, b: &Gpus) {
    assert_eq!(
        a.layer_to_device, b.layer_to_device,
        "{label}: layer_to_device differs"
    );
    assert_eq!(a.band_starts, b.band_starts, "{label}: band_starts differs");
    assert_eq!(
        a.devices.len(),
        b.devices.len(),
        "{label}: device count differs"
    );
    println!(
        "  {label}: OK ({} layers, {} devices)",
        a.layer_to_device.len(),
        a.devices.len()
    );
}

fn main() {
    // Emulate enough logical devices for the largest degree below.
    if std::env::var("HIPFIRE_EMULATE_GPUS").is_err() {
        std::env::set_var("HIPFIRE_EMULATE_GPUS", "4");
    }
    let n_layers = 24usize;

    println!("from_mesh vs init_uniform/init_tp parity (n_layers={n_layers}):");

    // Tp axis → init_uniform
    let via_mesh = Gpus::from_mesh(&DeviceMesh::rect(&[(DimKind::Tp, 2)]), n_layers).unwrap();
    let direct = Gpus::init_uniform(2, n_layers).unwrap();
    assert_same("Tp-2 → init_uniform(2)", &via_mesh, &direct);

    // Pp axis → init_uniform
    let via_mesh = Gpus::from_mesh(&DeviceMesh::rect(&[(DimKind::Pp, 2)]), n_layers).unwrap();
    let direct = Gpus::init_uniform(2, n_layers).unwrap();
    assert_same("Pp-2 → init_uniform(2)", &via_mesh, &direct);

    // Ep axis → init_tp
    let via_mesh = Gpus::from_mesh(&DeviceMesh::rect(&[(DimKind::Ep, 4)]), n_layers).unwrap();
    let direct = Gpus::init_tp(4, n_layers).unwrap();
    assert_same("Ep-4 → init_tp(4)", &via_mesh, &direct);

    // Single-device mesh must be rejected (single-GPU path owns that case).
    assert!(
        Gpus::from_mesh(&DeviceMesh::single(), n_layers).is_err(),
        "single-device mesh must return Err"
    );
    println!("  single() → Err: OK");

    println!("ALL PARITY CHECKS PASSED");
}
