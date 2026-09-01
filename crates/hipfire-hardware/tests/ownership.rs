// SPDX-License-Identifier: MIT
// Copyright (c) 2026 alpineq
// hipfire — see LICENSE and NOTICE in the project root.

use hipfire_hardware::{CollectiveHint, DeviceMesh, DimKind, Gpus};
use std::path::Path;

#[test]
fn hardware_leaf_exposes_owner_and_named_topology() {
    let _owner = std::any::TypeId::of::<Gpus>();
    let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2), (DimKind::Tp, 2)]).unwrap();
    assert_eq!(mesh.n_devices(), 4);
    assert_eq!(mesh.group_along(DimKind::Tp, &[1, 0]), vec![2, 3]);
    assert_eq!(
        mesh.band_xfer_after(0, 2),
        Some(CollectiveHint::BandXfer { src: 0, dst: 1 })
    );
}

#[test]
fn runtime_has_no_legacy_owner_or_compatibility_reexport() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(!crate_root
        .join("../hipfire-runtime/src/multi_gpu.rs")
        .exists());
    let runtime_lib = std::fs::read_to_string(crate_root.join("../hipfire-runtime/src/lib.rs"))
        .expect("runtime lib source");
    assert!(!runtime_lib.contains("pub mod multi_gpu"));
    assert!(!runtime_lib.contains("pub use hipfire_hardware::*"));
}
