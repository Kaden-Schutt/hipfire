// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G4 lifecycle evidence.
//!
//! CPU tests exercise the loader-owned reset/eviction and terminal contracts on
//! every run. The GPU tests are intentionally ignored: they need the exact
//! gfx1151 fixture set and the opt-in fault hooks described in each test's
//! ignore reason.

#![allow(clippy::all)]

use std::path::{Path, PathBuf};
use std::sync::{atomic::AtomicBool, Arc, Barrier, Mutex, OnceLock};

use hipfire_engine::terminal::{
    activate_terminal_control, clear_terminal_control, set_active_attempt_id,
};
use hipfire_generate::ar::GenerationRoute;
use hipfire_runtime::llama::{EmbeddingFormat, WeightTensor};
use hipfire_runtime::loader_api::{CaskConfig, LoadFaultStage, SpecLoadCfg};
use hipfire_runtime::model_load::WeightSource;
use rdna_compute::{Gpu, GpuTensor};

// ── production reset lifecycle matrix ──────────────────────────────────────

fn reset_test_owner() -> hipfire_loader::Eviction {
    let mut owner = hipfire_runtime::triattn::EvictionCtx::for_test(32, 8);
    owner.set_activation_gate(Arc::new(AtomicBool::new(false)));
    hipfire_loader::Eviction::Plain(owner)
}

fn seeded_reset_state() -> (usize, Vec<u32>, Vec<u32>, i32, Vec<u32>) {
    (37, vec![1, 2, 3, 4], vec![41, 42], 13, vec![99, 100])
}

#[test]
fn production_reset_lifecycle_dispatches_all_topologies_and_preserves_owner() {
    for route in [
        hipfire_loader::ResetRoute::Single,
        hipfire_loader::ResetRoute::PipelineParallel,
        hipfire_loader::ResetRoute::TensorParallel,
        hipfire_loader::ResetRoute::ExpertParallel,
    ] {
        let (
            mut seq_pos,
            mut conversation_tokens,
            mut request_tokens,
            mut compact_offset,
            mut speculative_pending,
        ) = seeded_reset_state();
        let mut owner = reset_test_owner();
        let owner_ptr = &owner as *const _;
        let mut phases = Vec::new();
        let mut run = |seen_route: hipfire_loader::ResetRoute,
                       phase: hipfire_loader::ResetPhase|
         -> Result<(), String> {
            assert_eq!(seen_route, route);
            phases.push(phase);
            Ok(())
        };
        let result = hipfire_loader::reset_lifecycle(
            route,
            hipfire_loader::ResetRequestState {
                seq_pos: &mut seq_pos,
                conversation_tokens: &mut conversation_tokens,
                asst_turn_cache: None,
                request_tokens: Some(&mut request_tokens),
                compact_offset: Some(&mut compact_offset),
                speculative_pending: Some(&mut speculative_pending),
            },
            Some(&owner),
            &mut hipfire_loader::ResetOperations { run: &mut run },
        );
        assert!(result.is_ok(), "{route:?} reset: {result:?}");
        assert_eq!(seq_pos, 0);
        assert!(conversation_tokens.is_empty());
        assert!(request_tokens.is_empty());
        assert_eq!(compact_offset, 0);
        assert!(speculative_pending.is_empty());
        assert_eq!(
            phases,
            vec![
                hipfire_loader::ResetPhase::Checkpoints,
                hipfire_loader::ResetPhase::Architecture,
                hipfire_loader::ResetPhase::AdaptiveKv,
                hipfire_loader::ResetPhase::Batch,
                hipfire_loader::ResetPhase::Speculator,
                hipfire_loader::ResetPhase::EvictionRequest,
                hipfire_loader::ResetPhase::GraphsAndSynchronize,
            ]
        );
        assert_eq!(
            &owner as *const _, owner_ptr,
            "{route:?} replaced eviction owner"
        );
        assert_eq!(
            match &owner {
                hipfire_loader::Eviction::Plain(ctx) => ctx.request_reset_count(),
                hipfire_loader::Eviction::Cask(ctx) => ctx.base.request_reset_count(),
            },
            1
        );
        assert_eq!(owner.budget(), 32);
        assert_eq!(owner.beta(), 8);
    }
}

#[test]
fn production_reset_lifecycle_attempts_all_phases_after_recurrent_failure() {
    for route in [
        hipfire_loader::ResetRoute::Single,
        hipfire_loader::ResetRoute::PipelineParallel,
        hipfire_loader::ResetRoute::TensorParallel,
        hipfire_loader::ResetRoute::ExpertParallel,
    ] {
        let (
            mut seq_pos,
            mut conversation_tokens,
            mut request_tokens,
            mut compact_offset,
            mut speculative_pending,
        ) = seeded_reset_state();
        let mut owner = reset_test_owner();
        let mut phases = Vec::new();
        let mut run = |seen_route: hipfire_loader::ResetRoute,
                       phase: hipfire_loader::ResetPhase|
         -> Result<(), String> {
            assert_eq!(seen_route, route);
            phases.push(phase);
            if phase == hipfire_loader::ResetPhase::Architecture {
                Err("injected recurrent reset".to_string())
            } else {
                Ok(())
            }
        };
        let result = hipfire_loader::reset_lifecycle(
            route,
            hipfire_loader::ResetRequestState {
                seq_pos: &mut seq_pos,
                conversation_tokens: &mut conversation_tokens,
                asst_turn_cache: None,
                request_tokens: Some(&mut request_tokens),
                compact_offset: Some(&mut compact_offset),
                speculative_pending: Some(&mut speculative_pending),
            },
            Some(&owner),
            &mut hipfire_loader::ResetOperations { run: &mut run },
        );
        let error = result.expect_err("injected recurrent failure must be reported");
        assert!(error.contains("architecture: injected recurrent reset"));
        assert_eq!(phases.len(), 7, "{route:?} reset stopped after failure");
        assert_eq!(seq_pos, 0);
        assert!(conversation_tokens.is_empty());
        assert!(request_tokens.is_empty());
        assert_eq!(compact_offset, 0);
        assert!(speculative_pending.is_empty());
        assert_eq!(
            match &owner {
                hipfire_loader::Eviction::Plain(ctx) => ctx.request_reset_count(),
                hipfire_loader::Eviction::Cask(ctx) => ctx.base.request_reset_count(),
            },
            1
        );
    }
}

// ── concrete route adapter race/cardinality matrix ─────────────────────────

fn terminal_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

fn parse_json_lines(bytes: &[u8]) -> Vec<serde_json::Value> {
    String::from_utf8(bytes.to_vec())
        .expect("writer output utf8")
        .lines()
        .map(|line| serde_json::from_str(line).expect("writer JSONL"))
        .collect()
}

#[test]
fn every_generation_route_has_a_concrete_lifecycle_adapter() {
    for route in GenerationRoute::ALL {
        let adapter = hipfire_generate::ar::generation_route_adapter(*route)
            .unwrap_or_else(|| panic!("missing lifecycle adapter for {}", route.name()));
        assert_eq!(adapter.route, *route);
    }
}

#[test]
fn route_adapters_start_first_and_race_one_semantic_terminal() {
    let _lock = terminal_test_lock();
    let contenders = [
        hipfire_generate::ar::RouteTerminal::Done,
        hipfire_generate::ar::RouteTerminal::Error,
        hipfire_generate::ar::RouteTerminal::Cancel,
    ];
    for (route_idx, route) in GenerationRoute::ALL.iter().copied().enumerate() {
        let adapter = hipfire_generate::ar::generation_route_adapter(route)
            .unwrap_or_else(|| panic!("missing lifecycle adapter for {}", route.name()));
        let id = format!("g4-{}-{route_idx}", route.name());
        let attempt = 700 + route_idx as u64;
        clear_terminal_control();
        activate_terminal_control(&id, attempt);
        set_active_attempt_id(attempt);
        let mut output = Vec::new();
        adapter.emit_start(&mut output, &id);
        adapter.emit_start(&mut output, &id);
        let start_lines = parse_json_lines(&output);
        assert_eq!(
            start_lines.len(),
            1,
            "{} adapter duplicated gen_start",
            route.name()
        );
        assert_eq!(
            start_lines.first().and_then(|line| line.get("type")),
            Some(&serde_json::Value::String("gen_start".to_string())),
            "{} adapter did not emit gen_start first",
            route.name()
        );

        let barrier = Arc::new(Barrier::new(contenders.len() + 1));
        let mut joins = Vec::new();
        for contender in contenders {
            let barrier = Arc::clone(&barrier);
            let adapter = adapter;
            let id = id.clone();
            joins.push(std::thread::spawn(move || {
                set_active_attempt_id(attempt);
                barrier.wait();
                let mut local = Vec::new();
                adapter.emit_terminal(&mut local, &id, attempt, contender);
                local
            }));
        }
        barrier.wait();
        for join in joins {
            output.extend(join.join().expect("route terminal writer thread"));
        }

        let lines = parse_json_lines(&output);
        assert_eq!(
            lines.first().and_then(|line| line.get("type")),
            Some(&serde_json::Value::String("gen_start".to_string())),
            "{} terminal race removed gen_start",
            route.name()
        );
        let tail = &lines[start_lines.len()..];
        let done = tail
            .iter()
            .filter(|line| line.get("type").and_then(|v| v.as_str()) == Some("done"))
            .count();
        let error = tail
            .iter()
            .filter(|line| line.get("type").and_then(|v| v.as_str()) == Some("error"))
            .count();
        let aborted = tail
            .iter()
            .filter(|line| line.get("type").and_then(|v| v.as_str()) == Some("aborted"))
            .count();
        assert!(
            aborted <= 1,
            "{} emitted multiple cancel terminals",
            route.name()
        );
        let semantic_owners = done + error + aborted - aborted.min(done);
        assert_eq!(
            semantic_owners,
            1,
            "{} terminal race emitted {semantic_owners} semantic owners",
            route.name()
        );

        let before_late = output.clone();
        for contender in contenders {
            adapter.emit_terminal(&mut output, &id, attempt, contender);
        }
        assert_eq!(
            output,
            before_late,
            "{} terminal claim leaked a late writer",
            route.name()
        );
        clear_terminal_control();
        set_active_attempt_id(0);
    }
}

// ── ignored live-GPU ownership tests ───────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WeightFaultStage {
    Embed,
    FinalNorm,
    Output,
    Layer(usize),
}

struct FaultingSource<S> {
    inner: S,
    stage: WeightFaultStage,
}

impl<S> FaultingSource<S> {
    fn new(inner: S, stage: WeightFaultStage) -> Self {
        Self { inner, stage }
    }

    fn fail(&self, stage: WeightFaultStage) -> hip_bridge::HipResult<()> {
        if self.stage == stage {
            Err(hip_bridge::HipError::new(
                0x4734,
                &format!("G4 fault at {stage:?}"),
            ))
        } else {
            Ok(())
        }
    }
}

impl<S: WeightSource> WeightSource for FaultingSource<S> {
    type Layer = S::Layer;

    fn n_layers(&self) -> usize {
        self.inner.n_layers()
    }

    fn prepare(&mut self, n_devices: usize) -> hip_bridge::HipResult<()> {
        self.inner.prepare(n_devices)
    }

    fn read_embed(&mut self, gpu: &mut Gpu) -> hip_bridge::HipResult<(GpuTensor, EmbeddingFormat)> {
        self.fail(WeightFaultStage::Embed)?;
        self.inner.read_embed(gpu)
    }

    fn read_final_norm(&mut self, gpu: &mut Gpu) -> hip_bridge::HipResult<GpuTensor> {
        self.fail(WeightFaultStage::FinalNorm)?;
        self.inner.read_final_norm(gpu)
    }

    fn read_output(
        &mut self,
        gpu: &mut Gpu,
        embd: &GpuTensor,
        embd_fmt: EmbeddingFormat,
        can_alias: bool,
    ) -> hip_bridge::HipResult<(WeightTensor, bool)> {
        self.fail(WeightFaultStage::Output)?;
        self.inner.read_output(gpu, embd, embd_fmt, can_alias)
    }

    fn read_layer(
        &mut self,
        gpu: &mut Gpu,
        layer_idx: usize,
    ) -> hip_bridge::HipResult<Self::Layer> {
        self.fail(WeightFaultStage::Layer(layer_idx))?;
        self.inner.read_layer(gpu, layer_idx)
    }

    fn free_layer(&mut self, gpu: &mut Gpu, layer: Self::Layer) {
        self.inner.free_layer(gpu, layer)
    }
}

fn required_path(var: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(var).unwrap_or_else(|_| panic!("set {var} for ignored G4 GPU test")),
    )
}

fn assert_gpu_baseline(gpu: &mut Gpu, baseline: usize) {
    gpu.ensure_vmm_cleaned()
        .expect("VMM cleanup after lifecycle attempt");
    gpu.drain_pool();
    assert_eq!(
        gpu.vmm_allocation_count(),
        baseline,
        "VMM owner leaked across lifecycle attempt"
    );
}

#[test]
#[ignore = "requires exact gfx1151 HIP device plus warm HFQ fixture in HIPFIRE_G4_HFQ_FIXTURE; ignored on CPU"]
fn gpu_hfq_staged_failure_sweep_then_success_repeated_unload() {
    let path = required_path("HIPFIRE_G4_HFQ_FIXTURE");
    let mut gpu = Gpu::init().expect("HIP device");
    assert_eq!(
        gpu.arch, "gfx1151",
        "G4 fixture is certified only on gfx1151"
    );
    let baseline = gpu.vmm_allocation_count();

    // Warm baseline and two complete unload cycles prove publication ownership,
    // alias handling, and repeated unload do not accumulate a stale owner.
    for _ in 0..2 {
        let mut hfq = hipfire_runtime::hfq::HfqFile::open(&path).expect("HFQ fixture");
        let cfg = hipfire_arch_qwen35::qwen35::config_from_hfq(&hfq).expect("Qwen config");
        let mut source = hipfire_arch_qwen35::qwen35::HfqSource::new(&mut hfq, &cfg);
        let weights = hipfire_arch_qwen35::qwen35::load_weights(
            &mut source,
            std::slice::from_mut(&mut gpu),
            &hipfire_runtime::model_load::Layout::single(cfg.n_layers),
        )
        .expect("warm HFQ load");
        weights.free_gpu(&mut gpu);
        assert_gpu_baseline(&mut gpu, baseline);
    }

    for stage in [
        WeightFaultStage::Embed,
        WeightFaultStage::FinalNorm,
        WeightFaultStage::Output,
        WeightFaultStage::Layer(0),
        WeightFaultStage::Layer(2),
    ] {
        let mut hfq = hipfire_runtime::hfq::HfqFile::open(&path).expect("HFQ fixture");
        let cfg = hipfire_arch_qwen35::qwen35::config_from_hfq(&hfq).expect("Qwen config");
        let source = hipfire_arch_qwen35::qwen35::HfqSource::new(&mut hfq, &cfg);
        let mut source = FaultingSource::new(source, stage);
        let result = hipfire_arch_qwen35::qwen35::load_weights(
            &mut source,
            std::slice::from_mut(&mut gpu),
            &hipfire_runtime::model_load::Layout::single(cfg.n_layers),
        );
        assert!(result.is_err(), "fault stage {stage:?} unexpectedly loaded");
        assert_gpu_baseline(&mut gpu, baseline);

        // A failed load must not poison the next successful load on the same
        // warm GPU, and a second unload must remain clean.
        let mut hfq = hipfire_runtime::hfq::HfqFile::open(&path).expect("HFQ fixture");
        let cfg = hipfire_arch_qwen35::qwen35::config_from_hfq(&hfq).expect("Qwen config");
        let mut source = hipfire_arch_qwen35::qwen35::HfqSource::new(&mut hfq, &cfg);
        let weights = hipfire_arch_qwen35::qwen35::load_weights(
            &mut source,
            std::slice::from_mut(&mut gpu),
            &hipfire_runtime::model_load::Layout::single(cfg.n_layers),
        )
        .expect("post-failure HFQ load");
        weights.free_gpu(&mut gpu);
        assert_gpu_baseline(&mut gpu, baseline);
    }
}

#[test]
#[ignore = "requires exact gfx1151 HIP device plus warm ParoQuant safetensors fixture in HIPFIRE_G4_PARO_DIR; ignored on CPU"]
fn gpu_paro_staged_failure_sweep_then_success_repeated_unload() {
    let path = required_path("HIPFIRE_G4_PARO_DIR");
    let mut gpu = Gpu::init().expect("HIP device");
    assert_eq!(
        gpu.arch, "gfx1151",
        "G4 fixture is certified only on gfx1151"
    );
    let baseline = gpu.vmm_allocation_count();

    for stage in [
        WeightFaultStage::Embed,
        WeightFaultStage::FinalNorm,
        WeightFaultStage::Output,
        WeightFaultStage::Layer(0),
        WeightFaultStage::Layer(2),
    ] {
        let source_file = hipfire_runtime::safetensors_source::SafetensorsSource::open(&path)
            .expect("ParoQuant fixture");
        let cfg = hipfire_arch_qwen35::qwen35::config_from_safetensors(&source_file)
            .expect("Paro config");
        let source =
            hipfire_arch_qwen35::qwen35::ParoSource::new(&source_file, &cfg).expect("Paro source");
        let mut source = FaultingSource::new(source, stage);
        let result = hipfire_arch_qwen35::qwen35::load_weights(
            &mut source,
            std::slice::from_mut(&mut gpu),
            &hipfire_runtime::model_load::Layout::single(cfg.n_layers),
        );
        assert!(result.is_err(), "fault stage {stage:?} unexpectedly loaded");
        assert_gpu_baseline(&mut gpu, baseline);

        let source_file = hipfire_runtime::safetensors_source::SafetensorsSource::open(&path)
            .expect("ParoQuant fixture");
        let cfg = hipfire_arch_qwen35::qwen35::config_from_safetensors(&source_file)
            .expect("Paro config");
        let mut source =
            hipfire_arch_qwen35::qwen35::ParoSource::new(&source_file, &cfg).expect("Paro source");
        let weights = hipfire_arch_qwen35::qwen35::load_weights(
            &mut source,
            std::slice::from_mut(&mut gpu),
            &hipfire_runtime::model_load::Layout::single(cfg.n_layers),
        )
        .expect("post-failure Paro load");
        weights.free_gpu(&mut gpu);
        assert_gpu_baseline(&mut gpu, baseline);
    }
}

fn try_load_model(
    gpu: &mut Gpu,
    target: &Path,
    draft: Option<&Path>,
    spec: SpecLoadCfg,
) -> Result<hipfire_loader::LoadedModel, String> {
    let cask = CaskConfig::default();
    hipfire_loader::load_model(
        target.to_str().expect("target utf8"),
        1024,
        draft.map(|path| path.to_str().expect("draft utf8")),
        None,
        None,
        None,
        &cask,
        1,
        spec,
        gpu,
    )
}

fn load_and_unload_model(gpu: &mut Gpu, target: &Path, draft: Option<&Path>, spec: SpecLoadCfg) {
    let model = try_load_model(gpu, target, draft, spec).expect("model load");
    hipfire_loader::unload_model(model, gpu).expect("model unload");
}

fn expect_load_failure(
    gpu: &mut Gpu,
    target: &Path,
    draft: Option<&Path>,
    spec: SpecLoadCfg,
    expected_stage: LoadFaultStage,
) {
    match try_load_model(gpu, target, draft, spec) {
        Err(error) => assert_eq!(
            error,
            format!(
                "{}: target-owner-published; draft-owner-published; injected failure",
                expected_stage.label()
            ),
            "fault did not stop at the requested stage with both owners published"
        ),
        Ok(model) => {
            let _ = hipfire_loader::unload_model(model, gpu);
            panic!("fault fixture unexpectedly loaded");
        }
    }
}

#[test]
#[ignore = "requires exact gfx1151 HIP device plus valid DFlash/DSpark target+draft fixtures in HIPFIRE_G4_*_TARGET and HIPFIRE_G4_*_DRAFT; ignored on CPU"]
fn gpu_dflash_dspark_target_verify_and_head_failures_recover_without_double_free() {
    let dflash_target = required_path("HIPFIRE_G4_DFLASH_TARGET");
    let dflash_draft = required_path("HIPFIRE_G4_DFLASH_DRAFT");
    let dspark_target = required_path("HIPFIRE_G4_DSPARK_TARGET");
    let dspark_draft = required_path("HIPFIRE_G4_DSPARK_DRAFT");

    let mut gpu = Gpu::init().expect("HIP device");
    assert_eq!(
        gpu.arch, "gfx1151",
        "G4 fixtures are certified only on gfx1151"
    );
    let baseline = gpu.vmm_allocation_count();

    load_and_unload_model(
        &mut gpu,
        &dflash_target,
        Some(&dflash_draft),
        SpecLoadCfg {
            dspark: Some(false),
            ..Default::default()
        },
    );
    assert_gpu_baseline(&mut gpu, baseline);
    expect_load_failure(
        &mut gpu,
        &dflash_target,
        Some(&dflash_draft),
        SpecLoadCfg {
            dspark: Some(false),
            lifecycle_fault: Some(LoadFaultStage::DflashTargetVerifyScratch),
            ..Default::default()
        },
        LoadFaultStage::DflashTargetVerifyScratch,
    );
    assert_gpu_baseline(&mut gpu, baseline);
    load_and_unload_model(
        &mut gpu,
        &dflash_target,
        Some(&dflash_draft),
        SpecLoadCfg {
            dspark: Some(false),
            ..Default::default()
        },
    );
    assert_gpu_baseline(&mut gpu, baseline);
    expect_load_failure(
        &mut gpu,
        &dspark_target,
        Some(&dspark_draft),
        SpecLoadCfg {
            dspark: Some(true),
            lifecycle_fault: Some(LoadFaultStage::DsparkHead),
            ..Default::default()
        },
        LoadFaultStage::DsparkHead,
    );
    assert_gpu_baseline(&mut gpu, baseline);
    load_and_unload_model(
        &mut gpu,
        &dspark_target,
        Some(&dspark_draft),
        SpecLoadCfg {
            dspark: Some(true),
            ..Default::default()
        },
    );
    assert_gpu_baseline(&mut gpu, baseline);
}
