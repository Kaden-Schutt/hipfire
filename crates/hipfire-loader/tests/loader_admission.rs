// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase 1 Oracle blocker regression tests (loader admission lane).
//!
//! RED against the current tree; they define the production contracts the
//! loader remediation must implement:
//!
//! - Oracle blocker 3: normalized admissions (dense EP→Single, DeepSeek/
//!   MiniMax TP→EP) must select the effective mesh inside `load_admitted`
//!   instead of rejecting the caller's original requested mesh with the
//!   topology-mismatch error.
//! - Oracle blocker 3 (validation gap): that effective-mesh selection must
//!   be gated on the caller's mesh matching the admission's REQUESTED
//!   degrees — an unrelated mesh must be refused with the `[CAP-001]`
//!   topology diagnostic before any GPU work, not silently discarded in
//!   favor of the effective mesh.
//! - Oracle blocker 2: the daemon's production compatibility wrapper
//!   (`load_model_ep_with_kv_mode`, the tp>1 load path) must refuse
//!   Planned / Unsupported parallel routes with the `[CAP-001]` admission
//!   diagnostic before any tokenizer/config/GPU construction.

use hipfire_loader::parallel_capability::RawParallelRequest;
use hipfire_loader::{admit_path, load_admitted, load_model_ep_with_kv_mode, ModelLoadOptions};
use hipfire_runtime::hfq::write_hfqm_package_mem;
use hipfire_runtime::multi_gpu::{DeviceMesh, DimKind};

/// Unique scratch directory per test (integration tests run in parallel).
fn fixture_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-loader-admission-{}-{}",
        std::process::id(),
        name
    ));
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    dir
}

/// Minimal but real HFQ container: valid HFQM header with `arch_id`, empty
/// metadata (`{}`) and no tensors. Enough for source opening, carrier probe,
/// and variant classification — the admission contract never reads tensor
/// payloads, and the absent tokenizer is a deterministic pre-GPU fault for
/// any code that reaches deeper than admission.
fn write_minimal_hfq(dir: &std::path::Path, arch_id: u32) -> std::path::PathBuf {
    let path = dir.join("model.hfq");
    write_hfqm_package_mem(&path, arch_id, "{}", &[]).expect("write minimal HFQ fixture");
    path
}

/// Oracle blocker 3 — dense EP→Single normalization.
///
/// Qwen2 (arch_id 7) is dense: the `(Qwen2, Ep)` policy cell is
/// `NormalizeToSingle`, so an EP=2 request admits with effective degrees
/// (1,1,1) on the Single axis. `load_admitted` must select that effective
/// mesh and route on it when the caller passes the ORIGINAL requested mesh
/// (ep=2). Today it returns the topology-mismatch error instead.
///
/// The deterministic no-GPU next error — `[CAP-001] Single load requires a
/// GPU (got None)` — is the contract proof: the Single axis refuses on a
/// missing GPU before any allocation, so this test only passes when the
/// effective mesh was selected before routing. A fix that compares on the
/// effective mesh but still routes on the original mesh would hit the
/// `EP not supported for arch_id=7` path instead and still fail.
#[test]
fn dense_ep_request_selects_effective_single_mesh_before_allocation() {
    let dir = fixture_dir("dense_ep_to_single");
    let path = write_minimal_hfq(&dir, 7);

    // Original requested mesh: EP=2 — never the effective (1,1,1) mesh.
    let mesh = DeviceMesh::rect(&[(DimKind::Pp, 1), (DimKind::Tp, 1), (DimKind::Ep, 2)]);
    let admitted = admit_path(&path, RawParallelRequest::new(1, 1, 2))
        .expect("dense EP=2 request admits via normalization");
    assert_eq!(
        admitted.admission().effective(),
        RawParallelRequest::new(1, 1, 1),
        "dense EP request must normalize to the Single axis"
    );
    assert!(
        admitted.admission().was_normalized(),
        "dense EP normalization must be recorded on the admission"
    );

    let err = match load_admitted(admitted, &mesh, ModelLoadOptions::new(64), None) {
        Ok(_) => panic!("load_admitted must not succeed with gpu=None"),
        Err(err) => err,
    };
    assert_eq!(
        err,
        "[CAP-001] Single load requires a GPU (got None)",
        "normalized admission must route on the effective mesh; the \
         topology-mismatch error is the unfixed behavior"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Oracle blocker 3 (validation gap) — normalized admissions still require
/// the caller's mesh to match the REQUESTED degrees before the effective
/// mesh is selected for routing.
///
/// The dense EP→Single admission above was resolved against requested
/// (1,1,2) and normalized to effective (1,1,1). `load_admitted` must
/// validate the caller's mesh against the admission's REQUESTED degrees
/// and only then select the effective mesh — an unrelated TP=8 mesh is
/// neither, so it must be refused with the canonical CAP topology
/// diagnostic before any GPU work.
///
/// Today `load_admitted` discards the caller's mesh whenever the admission
/// was normalized (it fabricates the effective mesh via `select_load_mesh`
/// and compares against the effective degrees), so the unrelated TP=8 mesh
/// is silently accepted and the load reaches the Single axis, failing with
/// `[CAP-001] Single load requires a GPU (got None)` instead of the
/// topology refusal. The exact diagnostic below names the admission's
/// REQUESTED degrees (1,1,2), not the effective (1,1,1) — that is what
/// distinguishes requested-mesh validation from effective routing.
#[test]
fn normalized_admission_refuses_unrelated_mesh_before_routing() {
    let dir = fixture_dir("dense_ep_unrelated_mesh");
    let path = write_minimal_hfq(&dir, 7);

    // Same normalized admission as the original-mesh test: requested
    // (1,1,2), effective (1,1,1).
    let admitted = admit_path(&path, RawParallelRequest::new(1, 1, 2))
        .expect("dense EP=2 request admits via normalization");
    assert_eq!(
        admitted.admission().effective(),
        RawParallelRequest::new(1, 1, 1),
        "dense EP request must normalize to the Single axis"
    );
    assert!(
        admitted.admission().was_normalized(),
        "dense EP normalization must be recorded on the admission"
    );

    // Unrelated mesh: TP=8 — matches neither the requested (1,1,2) nor
    // the effective (1,1,1) degrees.
    let unrelated = DeviceMesh::rect(&[(DimKind::Pp, 1), (DimKind::Tp, 8), (DimKind::Ep, 1)]);
    let err = match load_admitted(admitted, &unrelated, ModelLoadOptions::new(64), None) {
        Ok(_) => panic!("load_admitted must refuse a mesh unrelated to the admission"),
        Err(err) => err,
    };
    assert_eq!(
        err,
        "[CAP-001] load_admitted: mesh request (RawParallelRequest { pp: 1, tp: 8, ep: 1 }) \
         does not match admission requested (RawParallelRequest { pp: 1, tp: 1, ep: 2 })",
        "the caller mesh must be validated against the admission's REQUESTED \
         degrees before the effective mesh is selected; the Single gpu=None \
         error is the current silent-accept behavior"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Oracle blocker 2 — daemon compatibility wrapper CAP admission.
///
/// The daemon's tp>1 load path calls `load_model_ep_with_kv_mode`. Qwen3.5
/// MoE (arch_id 6) EP is a Planned cell under CAP-001 (owner AXIS-002), so
/// the wrapper must refuse it with the `[CAP-001]` Planned diagnostic before
/// touching the source's tokenizer/config or constructing any GPU state.
///
/// The minimal fixture carries no tokenizer metadata: if the wrapper bypasses
/// admission (current behavior) the error is the EP constructor's tokenizer
/// failure, which carries no `[CAP-001]` tag — the test fails for exactly the
/// Oracle finding. Asserting the Planned owner AXIS-002 pins the refusal to
/// the Qwen3.5 MoE EP cell rather than any generic error path.
#[test]
fn daemon_ep_wrapper_refuses_planned_route_with_cap001_before_gpu() {
    let dir = fixture_dir("planned_qwen35moe_ep");
    let path = write_minimal_hfq(&dir, 6);
    let path_str = path.to_str().expect("fixture path is UTF-8");

    let err = match load_model_ep_with_kv_mode(path_str, 64, 4, None, None) {
        Ok(_) => panic!("planned Qwen3.5 MoE EP must be refused before loading"),
        Err(err) => err,
    };
    assert!(
        err.contains("[CAP-001]"),
        "refusal must carry the CAP-001 tag; got: {err}"
    );
    assert!(
        err.contains("Planned"),
        "refusal must be the Planned-cell diagnostic; got: {err}"
    );
    assert!(
        err.contains("AXIS-002"),
        "refusal must name the Qwen3.5 MoE EP owner AXIS-002; got: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
