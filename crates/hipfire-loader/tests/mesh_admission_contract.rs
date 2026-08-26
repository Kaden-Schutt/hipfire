// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Device-mesh admission + reset-context contract tests.
//!
//! Admission contract (pure — no GPU): the daemon's load preflight runs
//! `admit_path` BEFORE any prior-model unload, so these tests pin the cells
//! that preflight admits/refuses:
//!
//! - Admitted dense TP/PP cells (MeshCarrier reachable end-to-end):
//!   LlamaQkNorm (arch 0) TP + PP, PlainQwen3 (arch 1) TP + PP,
//!   LlamaNoQkNorm PP.
//! - Explicit `params.ep`: DeepSeek4/MiniMax EP (direct or via legacy tp→ep
//!   remap) admits with effective Ep; Qwen3.5-MoE EP refuses Planned
//!   (AXIS-002); dense EP normalizes to Single — never silently dropped.
//! - Refusal-before-unload: every refusal is a canonical `[CAP-001]` /
//!   `[COMP-001]` diagnostic from the pure admission path (no tokenizer /
//!   config / GPU construction happens before it).
//!
//! Reset-context contract (GPU): `LoadedModel::reset_context` is the single
//! lifecycle authority — host session/cache clear, arch-state reset,
//! speculator reset with EXACT per-step error retention, graph/replay
//! invalidation, sync attestation.

use hipfire_loader::parallel_capability::{ParallelAxis, RawParallelRequest};
use hipfire_loader::{
    admit_path, load_model_ep_with_kv_mode, load_model_tp, validate_mtp_k,
    validate_state_quant, EagerLoadPreflight, LoadedModel,
};
use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::spec::{PrefillOutcome, SpecGrammar, SpecStep, Speculator};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::path::PathBuf;

/// Unique scratch directory per test (integration tests run in parallel).
fn fixture_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-loader-mesh-{}-{}",
        std::process::id(),
        name
    ));
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    dir
}

/// Minimal HFQ container with enough real config metadata for carrier
/// classification:
/// - `arch_id` 1 (PlainQwen3) or 0 (LLaMA) with a `config` block the llama
///   carrier's `config_from_hfq` parses;
/// - `qk_norm: true` adds a `q_norm.weight` index entry so arch 0 classifies
///   as LlamaQkNorm (absent → LlamaNoQkNorm).
/// The admission contract never reads tensor payloads, so entries carry zero
/// data; anything that reaches deeper fails deterministically pre-GPU.
fn write_mesh_hfq(dir: &std::path::Path, arch_id: u32, qk_norm: bool) -> PathBuf {
    let path = dir.join("model.hfq");
    let config = serde_json::json!({
        "model_type": if arch_id == 1 { "qwen3" } else { "llama" },
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 64,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 2048,
        "rope_theta": 1_000_000.0,
    });
    let metadata = serde_json::json!({
        "config": config,
        "tokenizer": "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},\"added_tokens\":[]}",
    })
    .to_string();
    let mut tensors: Vec<HfqMemTensor> = Vec::new();
    if qk_norm {
        tensors.push(HfqMemTensor {
            name: "model.layers.0.self_attn.q_norm.weight".into(),
            quant_type: 1,
            shape: vec![64],
            group_size: 0,
            data: Vec::new(),
        });
    }
    write_hfqm_package_mem(&path, arch_id, &metadata, &tensors).expect("write mesh HFQ fixture");
    path
}

/// Qwen35-MoE (arch 6) minimal fixture — variant classification reads only
/// arch_id, so the shared llama config shape is irrelevant to the policy row.
fn write_qwen35_hfq(dir: &std::path::Path, arch_id: u32) -> PathBuf {
    let path = dir.join("qwen35.hfq");
    write_hfqm_package_mem(
        &path,
        arch_id,
        &serde_json::json!({
            "tokenizer": "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},\"added_tokens\":[]}",
        })
        .to_string(),
        &[],
    )
    .expect("write qwen35 HFQ fixture");
    path
}

fn assert_effective(req: RawParallelRequest, axis: ParallelAxis) {
    let admitted = admit_path(fixture_dir("tmp").join("unused"), req).expect("admission ok");
    assert_eq!(admitted.admission().effective().axis(), axis);
}

// ─── Admitted dense TP/PP cells (the MeshCarrier daemon route) ────────────

#[test]
fn plain_qwen3_tp_and_pp_are_admitted() {
    let dir = fixture_dir("plain_qwen3");
    let path = write_mesh_hfq(&dir, 1, false);
    let tp = admit_path(&path, RawParallelRequest::new(1, 2, 1)).expect("PlainQwen3 TP admitted");
    assert_eq!(tp.admission().effective().axis(), ParallelAxis::Tp);
    let pp = admit_path(&path, RawParallelRequest::new(2, 1, 1)).expect("PlainQwen3 PP admitted");
    assert_eq!(pp.admission().effective().axis(), ParallelAxis::Pp);
}

#[test]
fn llama_qk_norm_tp_and_pp_are_admitted() {
    let dir = fixture_dir("llama_qk");
    let path = write_mesh_hfq(&dir, 0, true);
    let tp = admit_path(&path, RawParallelRequest::new(1, 2, 1)).expect("LlamaQkNorm TP admitted");
    assert_eq!(tp.admission().effective().axis(), ParallelAxis::Tp);
    let pp = admit_path(&path, RawParallelRequest::new(2, 1, 1)).expect("LlamaQkNorm PP admitted");
    assert_eq!(pp.admission().effective().axis(), ParallelAxis::Pp);
}

#[test]
fn llama_no_qk_norm_pp_admitted_tp_refused() {
    let dir = fixture_dir("llama_noqk");
    let path = write_mesh_hfq(&dir, 0, false);
    let pp = admit_path(&path, RawParallelRequest::new(2, 1, 1)).expect("LlamaNoQkNorm PP admitted");
    assert_eq!(pp.admission().effective().axis(), ParallelAxis::Pp);
    let err = admit_path(&path, RawParallelRequest::new(1, 2, 1)).expect_err("TP refused");
    assert!(err.contains("TP not supported"), "{err}");
    assert!(err.starts_with("[CAP-001]"), "{err}");
}

// ─── Explicit params.ep / axis explicitness ───────────────────────────────

#[test]
fn ds4_and_minimax_ep_are_admitted_via_ep_or_tp() {
    let dir = fixture_dir("ds4_ep");
    let ds4 = write_qwen35_hfq(&dir, 9);
    // Explicit ep=2.
    let explicit = admit_path(&ds4, RawParallelRequest::new(1, 1, 2)).expect("ds4 ep=2 admitted");
    assert_eq!(explicit.admission().effective().axis(), ParallelAxis::Ep);
    assert_eq!(explicit.admission().effective().ep, 2);
    // Legacy tp=2 remap → Ep, degree preserved.
    let remap = admit_path(&ds4, RawParallelRequest::new(1, 2, 1)).expect("ds4 tp=2 admitted");
    assert_eq!(remap.admission().effective().axis(), ParallelAxis::Ep);
    assert_eq!(remap.admission().effective().ep, 2);
    assert!(remap.admission().was_normalized());

    let minimax = write_qwen35_hfq(&dir, 10);
    let mm = admit_path(&minimax, RawParallelRequest::new(1, 1, 2)).expect("minimax ep=2 admitted");
    assert_eq!(mm.admission().effective().axis(), ParallelAxis::Ep);
}

#[test]
fn qwen35_moe_ep_refuses_planned_before_any_construction() {
    let dir = fixture_dir("qwen35moe_ep");
    let path = write_qwen35_hfq(&dir, 6);
    let err = admit_path(&path, RawParallelRequest::new(1, 1, 2)).expect_err("arch-6 EP refused");
    assert!(err.contains("AXIS-002"), "{err}");
    assert!(err.starts_with("[CAP-001]"), "{err}");
    // The production EP wrapper must refuse the same cell through the same
    // admission path — never reach an EP constructor.
    let err = match load_model_ep_with_kv_mode(path.to_str().unwrap(), 4096, 2, None, None) {
        Err(e) => e,
        Ok(_) => panic!("EP wrapper must refuse Planned arch-6 EP"),
    };
    assert!(err.contains("AXIS-002"), "{err}");
}

#[test]
fn qwen35_dense_tp_and_qwen2_pp_refuse_planned() {
    let dir = fixture_dir("planned_cells");
    let q35 = write_qwen35_hfq(&dir, 5);
    let err = admit_path(&q35, RawParallelRequest::new(1, 2, 1)).expect_err("qwen35 TP refused");
    assert!(err.contains("AXIS-002"), "{err}");
    // The TP wrapper (daemon's mesh route entry) must refuse the same cell.
    let err = match load_model_tp(q35.to_str().unwrap(), 4096, &mesh_for(1, 2, 1), Default::default())
    {
        Err(e) => e,
        Ok(_) => panic!("load_model_tp must refuse Planned arch-5 TP"),
    };
    assert!(err.contains("AXIS-002"), "{err}");

    let q2 = write_qwen35_hfq(&dir, 7);
    let err = admit_path(&q2, RawParallelRequest::new(2, 1, 1)).expect_err("qwen2 PP refused");
    assert!(err.contains("AXIS-001"), "{err}");
}

#[test]
fn dense_ep_normalizes_to_single_explicitly() {
    let dir = fixture_dir("dense_ep");
    let path = write_mesh_hfq(&dir, 1, false);
    let admitted = admit_path(&path, RawParallelRequest::new(1, 1, 2)).expect("dense EP normalizes");
    // The policy's NormalizeToSingle cell is an explicit, attested resolution
    // — never a silently dropped ep degree.
    assert!(admitted.admission().was_normalized());
    assert_eq!(admitted.admission().effective(), RawParallelRequest::new(1, 1, 1));
    assert_eq!(admitted.admission().effective().axis(), ParallelAxis::Single);
}

#[test]
fn composition_refused_before_anything() {
    let dir = fixture_dir("composition");
    let path = write_mesh_hfq(&dir, 1, false);
    let err = admit_path(&path, RawParallelRequest::new(1, 2, 2)).expect_err("tp×ep refused");
    assert!(err.contains("COMP-001"), "{err}");
    let err = admit_path(&path, RawParallelRequest::new(2, 2, 1)).expect_err("pp×tp refused");
    assert!(err.contains("CAP-001"), "{err}");
}

fn mesh_for(pp: usize, tp: usize, ep: usize) -> hipfire_runtime::multi_gpu::DeviceMesh {
    use hipfire_runtime::multi_gpu::{DeviceMesh, DimKind};
    DeviceMesh::rect(&[
        (DimKind::Pp, pp),
        (DimKind::Tp, tp),
        (DimKind::Ep, ep),
    ])
}

// ─── reset_context (single lifecycle authority) ───────────────────────────

/// Minimal failing/ok speculator: only `reset` is exercised by
/// `reset_context`; every other required method is an unreachable stub.
struct StubSpeculator {
    fail_reset: bool,
}

impl Speculator for StubSpeculator {
    fn prefill(
        &mut self,
        _gpu: &mut Gpu,
        _target: &mut dyn hipfire_runtime::spec::SpecTarget,
        _prompt_tokens: &[u32],
        _prefill_tokens: &[u32],
        _prefill_start: usize,
        _cache_hit: bool,
        _resume_from: Option<usize>,
        _abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        unreachable!("stub speculator prefill")
    }
    fn step(
        &mut self,
        _gpu: &mut Gpu,
        _target: &mut dyn hipfire_runtime::spec::SpecTarget,
        _position: usize,
        _seed: u32,
        _emitted: &[u32],
        _grammar: Option<&mut dyn SpecGrammar>,
        _temp: f32,
        _max_emit: usize,
    ) -> Result<SpecStep, String> {
        unreachable!("stub speculator step")
    }
    fn reset(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        if self.fail_reset {
            Err("boom".into())
        } else {
            Ok(())
        }
    }
    fn block_size(&self) -> usize {
        1
    }
    fn ctx_capacity(&self) -> usize {
        4096
    }
    fn free(self: Box<Self>, _gpu: &mut Gpu) {}
}

fn skeleton_model() -> LoadedModel {
    // A tokenizer is required for a real skeleton; the reset contract only
    // touches host session/cache + speculator, so a minimal one suffices.
    let tokenizer = Tokenizer::from_hf_json(
        r#"{"model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#,
    )
    .expect("test tokenizer");
    let mut m = LoadedModel::skeleton(5, tokenizer, 4096, 4096, "fixture".into(), None);
    m.seq_pos = 123;
    m.conversation_tokens = vec![1, 2, 3];
    m.asst_turn_cache.insert(
        0,
        hipfire_runtime::prompt_frame::CachedAssistantTurn {
            reasoning: None,
            tools: Vec::new(),
            content: None,
        },
    );
    m
}

#[test]
fn reset_context_clears_session_and_attests() {
    let mut gpu = Gpu::init().expect("GPU required for reset attestation");
    let mut m = skeleton_model();
    let reset = m.reset_context(&mut gpu, None, None);
    assert!(reset.rolled_back, "skeleton reset attests: {:?}", reset.context);
    assert!(reset.context.is_none());
    assert_eq!(m.seq_pos, 0);
    assert!(m.conversation_tokens.is_empty());
    assert!(!m.asst_turn_cache.contains_key(&0));
}

#[test]
fn reset_context_spec_failure_retains_exact_error() {
    let mut gpu = Gpu::init().expect("GPU required for reset attestation");
    let mut m = skeleton_model();
    m.speculator = Some(Box::new(StubSpeculator { fail_reset: true }));
    let reset = m.reset_context(&mut gpu, None, None);
    assert!(!reset.rolled_back);
    let ctx = reset.context.expect("exact error retained");
    assert!(ctx.contains("spec.reset"), "{ctx}");
    assert!(ctx.contains("boom"), "{ctx}");
    // Host session still cleared even though attestation failed.
    assert_eq!(m.seq_pos, 0);
}

// ─── Eager-route preflight (loader-owned authority) ──────────────────────

#[test]
fn preflight_rejects_invalid_qwen_state_quant() {
    let err = validate_state_quant(Some("bogus")).expect_err("invalid state_quant refused");
    assert!(err.contains("unsupported DeltaNet state_quant"), "{err}");
    assert!(validate_state_quant(Some("q8")).is_ok());
    assert!(validate_state_quant(Some("f32")).is_ok());
    assert!(validate_state_quant(Some("q4")).is_ok());
    assert!(validate_state_quant(None).is_ok());
}

#[test]
fn preflight_mtp_k_boundary() {
    // normalize_mtp_k's 1..=8 window, exposed pre-teardown.
    assert!(validate_mtp_k(Some(1)).is_ok());
    assert!(validate_mtp_k(Some(8)).is_ok());
    let err = validate_mtp_k(Some(0)).expect_err("mtp_k=0 refused");
    assert!(err.contains("MTP K must be in 1..=8"), "{err}");
    let err = validate_mtp_k(Some(9)).expect_err("mtp_k=9 refused");
    assert!(err.contains("MTP K must be in 1..=8"), "{err}");
}

#[test]
fn preflight_kv_backend_vmm_allowlist_and_pp() {
    use hipfire_loader::{parse_kv_backend, validate_kv_backend_for_load};
    let vmm = parse_kv_backend("vmm").expect("vmm parses");
    // Unsupported carrier (eager Single route, pre-unload refusal).
    let err = validate_kv_backend_for_load(vmm, "llama", 1).expect_err("llama+vmm refused");
    assert!(err.contains("supports qwen3.5, deepseek4, and Muse Glimmer only"), "{err}");
    // Allowed carrier but pp>1 (legacy qwen35 PP, pre-unload refusal).
    let err = validate_kv_backend_for_load(vmm, "qwen35", 2).expect_err("vmm+pp refused");
    assert!(err.contains("single-device and does not support pipeline parallelism"), "{err}");
    // Allowed carrier, single-device.
    assert!(validate_kv_backend_for_load(vmm, "qwen35", 1).is_ok());
    // Unparseable selector refused with the load path's error mapping.
    assert!(parse_kv_backend("bogus").is_err());
    let _ = KvBackend::default();
}

#[test]
fn preflight_dflash_target_quant_and_source() {
    let dir = fixture_dir("dflash_target");
    // No lm_head/embed entry → unsupported target source.
    let bare = write_mesh_hfq(&dir, 1, false);
    let src = ModelSource::from_path(bare.to_str().unwrap()).expect("open bare hfq");
    let err = hipfire_loader::preflight_dflash_target(&src, "gfx1201")
        .expect_err("missing lm_head refused");
    assert!(err.contains("no lm_head/embed_tokens tensor found"), "{err}");

    // MQ4G256 (qt 13) is always supported on every arch.
    let path = dir.join("supported.hfq");
    let config = serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 64,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 2048,
        "rope_theta": 1_000_000.0,
    });
    let metadata = serde_json::json!({ "config": config }).to_string();
    let tensors = vec![HfqMemTensor {
        name: "lm_head.weight".into(),
        quant_type: 13,
        shape: vec![64, 64],
        group_size: 256,
        data: Vec::new(),
    }];
    write_hfqm_package_mem(&path, 1, &metadata, &tensors).expect("write supported hfq");
    let src = ModelSource::from_path(path.to_str().unwrap()).expect("open supported hfq");
    assert!(hipfire_loader::preflight_dflash_target(&src, "gfx1201").is_ok());
    // Legacy qt17 is WMMA-gated (gfx11/gfx12 only); a non-WMMA arch refuses.
    let path = dir.join("legacy17.hfq");
    let tensors = vec![HfqMemTensor {
        name: "lm_head.weight".into(),
        quant_type: 17,
        shape: vec![64, 64],
        group_size: 256,
        data: Vec::new(),
    }];
    write_hfqm_package_mem(&path, 1, &metadata, &tensors).expect("write qt17 hfq");
    let src = ModelSource::from_path(path.to_str().unwrap()).expect("open qt17 hfq");
    assert!(hipfire_loader::preflight_dflash_target(&src, "gfx1201").is_ok());
    let err = hipfire_loader::preflight_dflash_target(&src, "gfx1030")
        .expect_err("qt17 on non-WMMA arch refused");
    assert!(err.contains("quant_type=17"), "{err}");
}

#[test]
fn preflight_eager_load_orchestrates_shared_authority() {
    let dir = fixture_dir("preflight_eager");
    let hfq = write_mesh_hfq(&dir, 1, false);
    let src = ModelSource::from_path(hfq.to_str().unwrap()).expect("open hfq");
    let base = EagerLoadPreflight {
        kv_mode: None,
        kv_backend: None,
        state_quant: None,
        mtp_k: None,
        mtp_k_admitted_route: false,
        draft_requested: false,
        gpu_arch: "gfx1201",
        deepseek4_experts_per_token: None,
        deepseek4_compute_placement: None,
        deepseek4_dspark: None,
        path: hfq.to_str().unwrap(),
        max_seq: 4096,
        cask: &hipfire_runtime::loader_api::CaskConfig::default(),
        gemma4_drafter: None,
        kv_adaptive: None,
        draft_path: None,
        tp: 1,
    };
    let mut gpu = Gpu::init().expect("GPU required for preflight harness");

    assert!(hipfire_loader::preflight_eager_load(&src, "llama", 1, &base, &mut gpu).is_ok());
    // state_quant is qwen35-only (covered by preflight_state_quant_is_qwen35_only);
    // on llama the knob is ignored, so no assertion here.
    // DFlash target refusal surfaces through the orchestrator.
    let draft = EagerLoadPreflight {
        draft_requested: true,
        ..base.clone()
    };
    let err = hipfire_loader::preflight_eager_load(&src, "llama", 1, &draft, &mut gpu)
        .expect_err("dflash target refused via orchestrator");
    assert!(err.contains("no lm_head/embed_tokens tensor found"), "{err}");
    // VMM allowlist refusal surfaces through the orchestrator.
    let vmm = EagerLoadPreflight {
        kv_backend: Some("vmm"),
        ..base.clone()
    };
    let err = hipfire_loader::preflight_eager_load(&src, "llama", 1, &vmm, &mut gpu)
        .expect_err("vmm allowlist refused via orchestrator");
    assert!(err.contains("supports qwen3.5, deepseek4, and Muse Glimmer only"), "{err}");
}

#[test]
fn preflight_state_quant_is_qwen35_only() {
    let dir = fixture_dir("state_quant_gate");
    let q3 = write_mesh_hfq(&dir, 1, false);
    let src = ModelSource::from_path(q3.to_str().unwrap()).expect("open hfq");
    let base = EagerLoadPreflight {
        kv_mode: None,
        kv_backend: None,
        state_quant: Some("bogus"),
        mtp_k: None,
        mtp_k_admitted_route: false,
        draft_requested: false,
        gpu_arch: "gfx1201",
        deepseek4_experts_per_token: None,
        deepseek4_compute_placement: None,
        deepseek4_dspark: None,
        path: q3.to_str().unwrap(),
        max_seq: 4096,
        cask: &hipfire_runtime::loader_api::CaskConfig::default(),
        gemma4_drafter: None,
        kv_adaptive: None,
        draft_path: None,
        tp: 1,
    };
    let mut gpu = Gpu::init().expect("GPU required for preflight harness");

    // Non-qwen35 carriers ignore the state_quant knob — must NOT refuse.
    assert!(
        hipfire_loader::preflight_eager_load(&src, "llama", 1, &base, &mut gpu).is_ok(),
        "state_quant is qwen35-only; other carriers ignore it"
    );
    // qwen35 carriers validate it (comes before the config parse, so the
    // minimal fixture's missing config does not mask the refusal).
    let q35 = write_qwen35_hfq(&dir, 5);
    let src = ModelSource::from_path(q35.to_str().unwrap()).expect("open qwen35 hfq");
    let err = hipfire_loader::preflight_eager_load(&src, "qwen35", 1, &base, &mut gpu)
        .expect_err("qwen35 state_quant refused");
    assert!(err.contains("unsupported DeltaNet state_quant"), "{err}");
}

#[test]
fn preflight_mtp_k_route_gating() {
    let dir = fixture_dir("mtp_k_gate");
    let hfq = write_mesh_hfq(&dir, 1, false);
    let src = ModelSource::from_path(hfq.to_str().unwrap()).expect("open hfq");
    let base = EagerLoadPreflight {
        kv_mode: None,
        kv_backend: None,
        state_quant: None,
        mtp_k: None,
        mtp_k_admitted_route: false,
        draft_requested: false,
        gpu_arch: "gfx1201",
        deepseek4_experts_per_token: None,
        deepseek4_compute_placement: None,
        deepseek4_dspark: None,
        path: hfq.to_str().unwrap(),
        max_seq: 4096,
        cask: &hipfire_runtime::loader_api::CaskConfig::default(),
        gemma4_drafter: None,
        kv_adaptive: None,
        draft_path: None,
        tp: 1,
    };
    let mut gpu = Gpu::init().expect("GPU required for preflight harness");

    // Wrapper routes (legacy qwen35 PP / load_model_with_gemma4_drafter):
    // mtp_k 9/10 is documented qwen35 MTP K (spec_build clamps 1..=10) and
    // normalize_mtp_k never runs — the preflight must accept it.
    for k in [Some(9), Some(10)] {
        let opts = EagerLoadPreflight { mtp_k: k, ..base.clone() };
        assert!(
            hipfire_loader::preflight_eager_load(&src, "qwen35", 2, &opts, &mut gpu).is_ok(),
            "wrapper route accepts mtp_k={k:?}"
        );
    }
    // Admitted routes run normalize_mtp_k (1..=8): 9/10 refuse, 8 accepts.
    let admitted = EagerLoadPreflight {
        mtp_k: Some(9),
        mtp_k_admitted_route: true,
        ..base.clone()
    };
    let err = hipfire_loader::preflight_eager_load(&src, "llama", 1, &admitted, &mut gpu)
        .expect_err("admitted route refuses mtp_k=9");
    assert!(err.contains("MTP K must be in 1..=8"), "{err}");
    let ok = EagerLoadPreflight {
        mtp_k: Some(8),
        mtp_k_admitted_route: true,
        ..base.clone()
    };
    assert!(hipfire_loader::preflight_eager_load(&src, "llama", 1, &ok, &mut gpu).is_ok());
}

#[test]
fn preflight_ds4_heterogeneous_uses_kv_mode_not_backend() {
    let dir = fixture_dir("ds4_hetero");
    let ds4 = write_qwen35_hfq(&dir, 9);
    let src = ModelSource::from_path(ds4.to_str().unwrap()).expect("open ds4 hfq");
    // Any non-Single placement exercises the heterogeneous branch.
    let placement = hipfire_config::Deepseek4ComputePlacement::DenseExpertSplit {
        dense: hipfire_config::DeviceSelector::ExactArch("gfx1201".into()),
        experts: hipfire_config::DeviceSelector::ExactArch("gfx1201".into()),
    };
    let mut gpu = Gpu::init().expect("GPU required for preflight harness");
    let base = EagerLoadPreflight {
        kv_mode: None,
        kv_backend: None,
        state_quant: None,
        mtp_k: None,
        mtp_k_admitted_route: false,
        draft_requested: false,
        gpu_arch: "gfx1201",
        deepseek4_experts_per_token: None,
        deepseek4_compute_placement: Some(placement.clone()),
        deepseek4_dspark: None,
        path: ds4.to_str().unwrap(),
        max_seq: 4096,
        cask: &hipfire_runtime::loader_api::CaskConfig::default(),
        gemma4_drafter: None,
        kv_adaptive: None,
        draft_path: None,
        tp: 1,
    };

    // kv_mode=f16 resolves to the F16 compressor → refused pre-teardown.
    let f16 = EagerLoadPreflight {
        kv_mode: Some("f16"),
        ..base.clone()
    };
    let err = hipfire_loader::preflight_eager_load(&src, "deepseek4", 1, &f16, &mut gpu)
        .expect_err("kv_mode=f16 heterogeneous refused");
    assert!(
        err.contains("kv_cache=f16 currently requires gfx1201 MQ2R TP3/TP4"),
        "{err}"
    );
    // kv_backend=vmm must NOT be fed to the compressor resolver — it is a
    // valid backend on the heterogeneous route. The refusal (if any) comes
    // from the pure device-selector resolution (this box's visible devices
    // may not match the fixture's gfx1201 selectors), never a
    // "kv_cache=vmm not recognised" compressor misrejection.
    let vmm = EagerLoadPreflight {
        kv_backend: Some("vmm"),
        ..base
    };
    let err = match hipfire_loader::preflight_eager_load(&src, "deepseek4", 1, &vmm, &mut gpu) {
        Err(e) => e,
        Ok(()) => return, // box matches the selectors: backend accepted as-is
    };
    assert!(
        !err.contains("kv_cache=vmm") && !err.contains("not recognised"),
        "backend leaked into compressor resolver: {err}"
    );
}


#[test]
fn preflight_tokenizer_metadata_required() {
    let dir = fixture_dir("tokenizer_preflight");
    // Bare HFQ without a tokenizer field: metadata parse fails preflight.
    let path = dir.join("bare.hfq");
    let config = serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 64,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 2048,
        "rope_theta": 1_000_000.0,
    });
    let metadata = serde_json::json!({ "config": config }).to_string();
    write_hfqm_package_mem(&path, 1, &metadata, &[]).expect("write bare hfq");
    let src = ModelSource::from_path(path.to_str().unwrap()).expect("open hfq");
    let base = EagerLoadPreflight {
        kv_mode: None,
        kv_backend: None,
        state_quant: None,
        mtp_k: None,
        mtp_k_admitted_route: false,
        draft_requested: false,
        gpu_arch: "gfx1201",
        deepseek4_experts_per_token: None,
        deepseek4_compute_placement: None,
        deepseek4_dspark: None,
        path: path.to_str().unwrap(),
        max_seq: 4096,
        cask: &hipfire_runtime::loader_api::CaskConfig::default(),
        gemma4_drafter: None,
        kv_adaptive: None,
        draft_path: None,
        tp: 1,
    };
    let mut gpu = Gpu::init().expect("GPU required for preflight harness");
    let err = hipfire_loader::preflight_eager_load(&src, "llama", 1, &base, &mut gpu)
        .expect_err("missing tokenizer metadata refused preflight");
    assert!(err.contains("tokenizer not found"), "{err}");
}

#[test]
fn preflight_pp_layer_bands_parse_count_sum() {
    use hipfire_loader::parse_pp_layer_bands;
    // Unset/empty -> None (uniform split).
    assert!(parse_pp_layer_bands(None, 2, 32).unwrap().is_none());
    assert!(parse_pp_layer_bands(Some(""), 2, 32).unwrap().is_none());
    // Valid bands.
    let bands = parse_pp_layer_bands(Some("16,16"), 2, 32).expect("valid bands");
    assert_eq!(bands.as_deref(), Some(&[16usize, 16][..]));
    // Parse error.
    let err = parse_pp_layer_bands(Some("16,x"), 2, 32).expect_err("parse refused");
    assert!(err.contains("HIPFIRE_PP_LAYERS parse"), "{err}");
    // Count mismatch.
    let err = parse_pp_layer_bands(Some("16,16,16"), 2, 32).expect_err("count refused");
    assert!(err.contains("entries, expected pp=2"), "{err}");
    // Sum mismatch.
    let err = parse_pp_layer_bands(Some("16,8"), 2, 32).expect_err("sum refused");
    assert!(err.contains("sum=24 != n_layers=32"), "{err}");
}

#[test]
fn preflight_gemma4_draft_refused() {
    let dir = fixture_dir("gemma4_draft");
    let hfq = write_mesh_hfq(&dir, 1, false);
    let src = ModelSource::from_path(hfq.to_str().unwrap()).expect("open hfq");
    let base = EagerLoadPreflight {
        kv_mode: None,
        kv_backend: None,
        state_quant: None,
        mtp_k: None,
        mtp_k_admitted_route: false,
        draft_requested: true,
        gpu_arch: "gfx1201",
        deepseek4_experts_per_token: None,
        deepseek4_compute_placement: None,
        deepseek4_dspark: None,
        path: hfq.to_str().unwrap(),
        max_seq: 4096,
        cask: &hipfire_runtime::loader_api::CaskConfig::default(),
        gemma4_drafter: None,
        kv_adaptive: None,
        draft_path: Some("draft.hfq"),
        tp: 1,
    };
    let mut gpu = Gpu::init().expect("GPU required for preflight harness");
    let err = hipfire_loader::preflight_eager_load(&src, "gemma4", 1, &base, &mut gpu)
        .expect_err("gemma4 params.draft refused preflight");
    assert!(
        err.contains("DFlash draft path not supported"),
        "{err}"
    );
}

#[test]
fn preflight_muse_glimmer_gates() {
    let dir = fixture_dir("glimmer_gates");
    let hfq = write_mesh_hfq(&dir, 1, false);
    let src = ModelSource::from_path(hfq.to_str().unwrap()).expect("open hfq");
    let base = EagerLoadPreflight {
        kv_mode: None,
        kv_backend: None,
        state_quant: None,
        mtp_k: None,
        mtp_k_admitted_route: false,
        draft_requested: false,
        gpu_arch: "gfx1201",
        deepseek4_experts_per_token: None,
        deepseek4_compute_placement: None,
        deepseek4_dspark: None,
        path: hfq.to_str().unwrap(),
        max_seq: 4096,
        cask: &hipfire_runtime::loader_api::CaskConfig::default(),
        gemma4_drafter: None,
        kv_adaptive: None,
        draft_path: None,
        tp: 1,
    };
    let mut gpu = Gpu::init().expect("GPU required for preflight harness");
    // VMM + CASK sidecar refused.
    let cask = hipfire_runtime::loader_api::CaskConfig {
        sidecar: Some("sidecar".into()),
        cask_m_folding: false,
        handoff_tokens: 0,
        budget: 0,
        beta: 0,
        core_frac: 0.0,
        fold_m: 0,
    };
    let vmm = EagerLoadPreflight {
        kv_backend: Some("vmm"),
        cask: &cask,
        ..base.clone()
    };
    let err = hipfire_loader::preflight_eager_load(&src, "muse_glimmer", 1, &vmm, &mut gpu)
        .expect_err("glimmer vmm+cask refused");
    assert!(err.contains("does not support CASK/TriAttention eviction"), "{err}");
    // pp>1 refused.
    let err = hipfire_loader::preflight_eager_load(&src, "muse_glimmer", 2, &base, &mut gpu)
        .expect_err("glimmer pp>1 refused");
    assert!(err.contains("pipeline-parallel (pp>1) unsupported"), "{err}");
    // gemma4 EAGLE drafter refused.
    let eagle = EagerLoadPreflight {
        gemma4_drafter: Some("eagle.hfq"),
        ..base
    };
    let err = hipfire_loader::preflight_eager_load(&src, "muse_glimmer", 1, &eagle, &mut gpu)
        .expect_err("glimmer gemma4 drafter refused");
    assert!(err.contains("Gemma4 EAGLE") || err.contains("params.drafter"), "{err}");
}


#[test]
fn preflight_qwen2_gates() {
    let dir = fixture_dir("qwen2_gates");
    let q2 = write_qwen35_hfq(&dir, 7);
    let src = ModelSource::from_path(q2.to_str().unwrap()).expect("open qwen2 hfq");
    let base = EagerLoadPreflight {
        kv_mode: None,
        kv_backend: None,
        state_quant: None,
        mtp_k: None,
        mtp_k_admitted_route: false,
        draft_requested: false,
        gpu_arch: "gfx1201",
        deepseek4_experts_per_token: None,
        deepseek4_compute_placement: None,
        deepseek4_dspark: None,
        path: q2.to_str().unwrap(),
        max_seq: 4096,
        cask: &hipfire_runtime::loader_api::CaskConfig::default(),
        gemma4_drafter: None,
        kv_adaptive: None,
        draft_path: None,
        tp: 1,
    };
    let mut gpu = Gpu::init().expect("GPU required for preflight harness");
    // params.draft refused.
    let draft = EagerLoadPreflight {
        draft_requested: true,
        draft_path: Some("draft.hfq"),
        ..base.clone()
    };
    let err = hipfire_loader::preflight_eager_load(&src, "qwen2", 1, &draft, &mut gpu)
        .expect_err("qwen2 draft refused preflight");
    assert!(err.contains("DFlash not supported on arch_id=7"), "{err}");
    // CASK sidecar refused.
    let cask = hipfire_runtime::loader_api::CaskConfig {
        sidecar: Some("sidecar".into()),
        cask_m_folding: false,
        handoff_tokens: 0,
        budget: 0,
        beta: 0,
        core_frac: 0.0,
        fold_m: 0,
    };
    let casky = EagerLoadPreflight {
        cask: &cask,
        ..base.clone()
    };
    let err = hipfire_loader::preflight_eager_load(&src, "qwen2", 1, &casky, &mut gpu)
        .expect_err("qwen2 cask refused preflight");
    assert!(err.contains("CASK eviction not supported on arch_id=7"), "{err}");
    // Safetensors dir without F16 .weight tensors refused.
    let dir = dir.join("qwen2_dir");
    std::fs::create_dir_all(&dir).expect("qwen2 dir");
    let config = serde_json::json!({
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string(&config).expect("serialize qwen2 config"),
    )
    .expect("write qwen2 config.json");
    let mut st = vec![2u8, 0, 0, 0, 0, 0, 0, 0];
    st.extend_from_slice(b"{}");
    std::fs::write(dir.join("model.safetensors"), st).expect("write qwen2 safetensors");
    std::fs::write(
        dir.join("tokenizer.json"),
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},\"added_tokens\":[]}",
    )
    .expect("write qwen2 tokenizer.json");
    let src = ModelSource::from_path(dir.to_str().unwrap()).expect("open qwen2 dir");
    let err = hipfire_loader::preflight_eager_load(&src, "qwen2", 1, &base, &mut gpu)
        .expect_err("qwen2 dir without F16 weights refused preflight");
    assert!(err.contains("has no F16 `.weight` tensors"), "{err}");
}


#[test]
fn preflight_carrier_config_parse_refused_for_all_eager_arches() {
    // Table-driven: every eager Single carrier whose config parse runs only
    // post-teardown today must refuse a malformed config pre-teardown.
    let cases: &[(u32, &str)] = &[
        (7, "qwen2"),
        (8, "dots_ocr"),
        (10, "minimax"),
        (11, "lfm2moe"),
        (12, "cohere2moe"),
        (14, "muse_glimmer"),
    ];
    let tok = "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},\"added_tokens\":[]}";
    for (arch_id, carrier) in cases {
        let dir = fixture_dir(&format!("config_{carrier}"));
        let path = dir.join("model.hfq");
        let metadata = serde_json::json!({
            // Wrong-typed field: typed serde config parsers must refuse.
            "config": { "model_type": "bogus", "hidden_size": "bogus" },
            "tokenizer": tok,
        })
        .to_string();
        write_hfqm_package_mem(&path, *arch_id, &metadata, &[]).expect("write malformed hfq");
        let src = ModelSource::from_path(path.to_str().unwrap()).expect("open hfq");
        let base = EagerLoadPreflight {
            kv_mode: None,
            kv_backend: None,
            state_quant: None,
            mtp_k: None,
            mtp_k_admitted_route: false,
            draft_requested: false,
            gpu_arch: "gfx1201",
            deepseek4_experts_per_token: None,
            deepseek4_compute_placement: None,
            deepseek4_dspark: None,
            path: path.to_str().unwrap(),
            max_seq: 4096,
            cask: &hipfire_runtime::loader_api::CaskConfig::default(),
            gemma4_drafter: None,
            kv_adaptive: None,
            draft_path: None,

        tp: 1,        };
        let mut gpu = Gpu::init().expect("GPU required for preflight harness");
        let err = hipfire_loader::preflight_eager_load(&src, carrier, 1, &base, &mut gpu)
            .expect_err(&format!("{carrier} malformed config refused preflight"));
        // The carrier's own config parser produced the refusal (each arch's
        // parser prefixes differently — e.g. glimmer: missing hidden_size);
        // any non-empty refusal BEFORE teardown is the contract.
        assert!(!err.is_empty(), "{carrier}: empty error");
        assert!(
            !err.contains("tokenizer not found"),
            "{carrier}: tokenizer preflight masked the config refusal: {err}"
        );
    }

    // Dir sources: the full config parsers the carrier loads run post-teardown
    // (dots_ocr/cohere2moe/lfm2moe from_source/from_safetensors, qwen35 Dir
    // Single config_from_safetensors) must refuse malformed configs preflight.
    let dir_cases: &[(&str, &str)] = &[
        ("qwen35", "Qwen3.5ForCausalLM"),
        ("dots_ocr", "Qwen2VlForConditionalGeneration"),
        ("cohere2moe", "Cohere2ForCausalLM"),
        ("lfm2moe", "Lfm2MoeForCausalLM"),
    ];
    for (carrier, arch_name) in dir_cases {
        let dir = fixture_dir(&format!("dir_config_{carrier}"));
        std::fs::create_dir_all(&dir).expect("dir fixture");
        let config = serde_json::json!({
            "architectures": [arch_name],
            "model_type": "bogus",
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string(&config).expect("serialize config"),
        )
        .expect("write config.json");
        let mut st = vec![2u8, 0, 0, 0, 0, 0, 0, 0];
        st.extend_from_slice(b"{}");
        std::fs::write(dir.join("model.safetensors"), st).expect("write safetensors");
        std::fs::write(
            dir.join("tokenizer.json"),
            "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},\"added_tokens\":[]}",
        )
        .expect("write tokenizer.json");
        let src = ModelSource::from_path(dir.to_str().unwrap()).expect("open dir");
        let base = EagerLoadPreflight {
            kv_mode: None,
            kv_backend: None,
            state_quant: None,
            mtp_k: None,
            mtp_k_admitted_route: false,
            draft_requested: false,
            gpu_arch: "gfx1201",
            deepseek4_experts_per_token: None,
            deepseek4_compute_placement: None,
            deepseek4_dspark: None,
            path: dir.to_str().unwrap(),
            max_seq: 4096,
            cask: &hipfire_runtime::loader_api::CaskConfig::default(),
            gemma4_drafter: None,
            kv_adaptive: None,
            draft_path: None,

        tp: 1,        };
        let mut gpu = Gpu::init().expect("GPU required for preflight harness");
        let err = hipfire_loader::preflight_eager_load(&src, carrier, 1, &base, &mut gpu)
            .expect_err(&format!("{carrier} dir malformed config refused preflight"));
        assert!(!err.is_empty(), "{carrier}: empty error");
        assert!(
            !err.contains("tokenizer not found"),
            "{carrier}: tokenizer preflight masked the config refusal: {err}"
        );
    }
}


#[test]
fn preflight_llama_mesh_topology_degree_bound() {
    let dir = fixture_dir("llama_topology");
    // The mesh fixture's config declares 8 layers.
    let hfq = write_mesh_hfq(&dir, 1, false);
    let src = ModelSource::from_path(hfq.to_str().unwrap()).expect("open hfq");
    let base = EagerLoadPreflight {
        kv_mode: None,
        kv_backend: None,
        state_quant: None,
        mtp_k: None,
        mtp_k_admitted_route: false,
        draft_requested: false,
        gpu_arch: "gfx1201",
        deepseek4_experts_per_token: None,
        deepseek4_compute_placement: None,
        deepseek4_dspark: None,
        path: hfq.to_str().unwrap(),
        max_seq: 4096,
        cask: &hipfire_runtime::loader_api::CaskConfig::default(),
        gemma4_drafter: None,
        kv_adaptive: None,
        draft_path: None,
        tp: 1,
    };
    let mut gpu = Gpu::init().expect("GPU required for preflight harness");
    // Admitted dense PP (eager) with pp > n_layers refuses pre-teardown with
    // the exact init_uniform authority message.
    let err = hipfire_loader::preflight_eager_load(&src, "llama", 9, &base, &mut gpu)
        .expect_err("pp > n_layers refused preflight");
    assert!(
        err.contains("n_layers (8) < n_devices (9)"),
        "{err}"
    );
    // pp == n_layers and below pass.
    assert!(
        hipfire_loader::preflight_eager_load(&src, "llama", 8, &base, &mut gpu).is_ok()
    );
    assert!(
        hipfire_loader::preflight_eager_load(&src, "llama", 2, &base, &mut gpu).is_ok()
    );
    // Dense TP defensively: tp > n_layers refuses; tp == n_layers passes.
    let tp9 = EagerLoadPreflight { tp: 9, ..base.clone() };
    let err = hipfire_loader::preflight_eager_load(&src, "llama", 1, &tp9, &mut gpu)
        .expect_err("tp > n_layers refused preflight");
    assert!(err.contains("n_layers (8) < n_devices (9)"), "{err}");
    let tp8 = EagerLoadPreflight { tp: 8, ..base };
    assert!(hipfire_loader::preflight_eager_load(&src, "llama", 1, &tp8, &mut gpu).is_ok());
}
