#![allow(unused_imports)]
use super::*;
use std::collections::BTreeMap;
use std::path::PathBuf;
// Import tooling now lives in the offline hipfire-diffusion-coexist crate.
use hipfire_diffusion_coexist::{
    import_diffusers_to_hfq, ldm_unet_native_tensor_name, ldm_vae_native_tensor_name,
    parse_pytorch_state_dict, pytorch_tensor_is_contiguous, reorder_pytorch_storage_to_contiguous,
    DiffusersImportOptions,
};
use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
use std::fs;
use super::*;

#[test]
fn linear_scheduler_euler_step_moves_toward_next_sigma() {
    let schedule = DiffusionSchedule::linear(2).unwrap();
    let mut latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 2,
        data: vec![1.0, -1.0],
    };

    schedule.euler_step(&mut latents, &[0.25, -0.5], 0).unwrap();

    assert_eq!(schedule.timesteps, vec![1.0, 0.0]);
    assert_eq!(schedule.sigmas, vec![1.0, 0.0, 0.0]);
    assert_eq!(latents.data, vec![0.75, -0.5]);
}

#[test]
fn scheduler_config_uses_diffusers_beta_sigmas_and_train_timesteps() {
    let config = SchedulerConfig {
        class_name: "EulerDiscreteScheduler".into(),
        beta_start: Some(0.0001),
        beta_end: Some(0.02),
        beta_schedule: Some("linear".into()),
        num_train_timesteps: Some(10),
        prediction_type: Some("epsilon".into()),
        ..SchedulerConfig::default()
    };

    let schedule = DiffusionSchedule::from_config(&config, 3).unwrap();

    assert_eq!(schedule.timesteps, vec![9.0, 5.0, 0.0]);
    assert_eq!(schedule.sigmas.len(), 4);
    assert!(schedule.sigmas[0] > schedule.sigmas[1]);
    assert!(schedule.sigmas[1] > schedule.sigmas[2]);
    assert_eq!(schedule.sigmas[3], 0.0);
}

#[test]
fn dpm_solver_config_uses_diffusers_linspace_timesteps() {
    let config = SchedulerConfig {
        class_name: "DPMSolverMultistepScheduler".into(),
        beta_start: Some(0.00085),
        beta_end: Some(0.012),
        beta_schedule: Some("scaled_linear".into()),
        num_train_timesteps: Some(1000),
        prediction_type: Some("epsilon".into()),
        algorithm_type: Some("dpmsolver++".into()),
        solver_order: Some(2),
        solver_type: Some("midpoint".into()),
        lower_order_final: Some(true),
        thresholding: Some(false),
        timestep_spacing: Some("linspace".into()),
        steps_offset: Some(1),
        use_karras_sigmas: Some(false),
        set_alpha_to_one: None,
        ..SchedulerConfig::default()
    };

    let schedule = DiffusionSchedule::from_config(&config, 3).unwrap();

    assert_eq!(schedule.train_timesteps, vec![999, 666, 333]);
    assert_eq!(schedule.timesteps, vec![999.0, 666.0, 333.0]);
    assert_eq!(
        schedule.solver,
        SchedulerSolver::DpmSolverMultistep {
            algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
            solver_order: 2,
            solver_type: DpmSolverType::Midpoint,
            lower_order_final: true,
            thresholding: false,
            dynamic_thresholding_ratio: 0.995,
            sample_max_value: 1.0,
        }
    );
    assert_eq!(schedule.input_scaling, SchedulerInputScaling::None);
    assert_eq!(schedule.initial_noise_sigma(), 1.0);
}

#[test]
fn dpm_solver_config_preserves_dynamic_thresholding_settings() {
    let config = SchedulerConfig {
        class_name: "DPMSolverMultistepScheduler".into(),
        beta_start: Some(0.00085),
        beta_end: Some(0.012),
        beta_schedule: Some("scaled_linear".into()),
        num_train_timesteps: Some(1000),
        prediction_type: Some("epsilon".into()),
        algorithm_type: Some("dpmsolver++".into()),
        solver_order: Some(2),
        solver_type: Some("midpoint".into()),
        thresholding: Some(true),
        dynamic_thresholding_ratio: Some(0.9),
        sample_max_value: Some(2.0),
        ..SchedulerConfig::default()
    };

    let schedule = DiffusionSchedule::from_config(&config, 2).unwrap();

    assert_eq!(
        schedule.solver,
        SchedulerSolver::DpmSolverMultistep {
            algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
            solver_order: 2,
            solver_type: DpmSolverType::Midpoint,
            lower_order_final: true,
            thresholding: true,
            dynamic_thresholding_ratio: 0.9,
            sample_max_value: 2.0,
        }
    );
}

#[test]
fn flow_match_euler_scheduler_uses_shifted_sigmas_and_terminal_rescale() {
    let config = SchedulerConfig {
        class_name: "FlowMatchEulerDiscreteScheduler".into(),
        num_train_timesteps: Some(1000),
        shift: Some(1.0),
        shift_terminal: Some(0.02),
        invert_sigmas: Some(false),
        ..SchedulerConfig::default()
    };

    let schedule = DiffusionSchedule::from_config(&config, 3).unwrap();

    assert_eq!(schedule.solver, SchedulerSolver::FlowMatchEuler);
    assert_eq!(schedule.input_scaling, SchedulerInputScaling::None);
    assert_eq!(schedule.sigmas, vec![1.0, 0.51, 0.02, 0.0]);
    assert_eq!(schedule.timesteps, vec![1000.0, 510.0, 20.0]);
    assert_eq!(schedule.initial_noise_sigma(), 1.0);
}

#[test]
fn flow_match_euler_step_uses_model_output_as_velocity() {
    let config = SchedulerConfig {
        class_name: "FlowMatchEulerDiscreteScheduler".into(),
        num_train_timesteps: Some(1000),
        shift: Some(1.0),
        ..SchedulerConfig::default()
    };
    let schedule = DiffusionSchedule::from_config(&config, 2).unwrap();
    let mut latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 2,
        data: vec![1.0, -1.0],
    };
    let mut state = SchedulerStepState::default();

    schedule
        .step(&mut latents, &[0.25, -0.5], 0, &mut state)
        .unwrap();

    assert_eq!(schedule.sigmas, vec![1.0, 0.0, 0.0]);
    assert_eq!(latents.data, vec![0.75, -0.5]);
}

fn flow_match_base_schedule_for_refine() -> DiffusionSchedule {
    let config = SchedulerConfig {
        class_name: "FlowMatchEulerDiscreteScheduler".into(),
        num_train_timesteps: Some(1000),
        shift: Some(1.0),
        ..SchedulerConfig::default()
    };
    DiffusionSchedule::from_config(&config, 12).unwrap()
}

#[test]
fn refine_direct_sigma_single_step_is_ramp_to_zero_with_scaled_timestep() {
    let base = flow_match_base_schedule_for_refine();
    let refine = base.refine_direct_sigma(0.12, 1, false).unwrap();

    // One step: sigmas [first_sigma, 0]; timestep = sigma * 1000 (base scale).
    assert_eq!(refine.solver, SchedulerSolver::FlowMatchEuler);
    assert_eq!(refine.sigmas, vec![0.12, 0.0]);
    assert_eq!(refine.timesteps.len(), 1);
    assert!((refine.timesteps[0] - 120.0).abs() < 1e-3);
}

#[test]
fn refine_direct_sigma_single_step_ignores_shifted_flag() {
    let base = flow_match_base_schedule_for_refine();
    let linear = base.refine_direct_sigma(0.16, 1, false).unwrap();
    let shifted = base.refine_direct_sigma(0.16, 1, true).unwrap();
    // For a single refine step the shifted and linear schedules are identical.
    assert_eq!(linear.sigmas, shifted.sigmas);
    assert_eq!(linear.sigmas, vec![0.16, 0.0]);
}

#[test]
fn refine_direct_sigma_multistep_ramps_from_first_sigma_to_zero() {
    let base = flow_match_base_schedule_for_refine();
    let refine = base.refine_direct_sigma(0.2, 4, false).unwrap();

    assert_eq!(refine.sigmas.len(), 5);
    assert_eq!(refine.timesteps.len(), 4);
    assert!((refine.sigmas[0] - 0.2).abs() < 1e-6);
    assert_eq!(*refine.sigmas.last().unwrap(), 0.0);
    // Monotonically decreasing toward zero.
    for pair in refine.sigmas.windows(2) {
        assert!(pair[0] >= pair[1]);
    }

    // Shifted variant keeps the same endpoints but bends the interior.
    let shifted = base.refine_direct_sigma(0.2, 4, true).unwrap();
    assert!((shifted.sigmas[0] - 0.2).abs() < 1e-6);
    assert_eq!(*shifted.sigmas.last().unwrap(), 0.0);
    assert_ne!(shifted.sigmas, refine.sigmas);
}

#[test]
fn refine_direct_sigma_rejects_bad_params_and_non_flow_match() {
    let base = flow_match_base_schedule_for_refine();
    assert!(base.refine_direct_sigma(0.0, 1, false).is_err());
    assert!(base.refine_direct_sigma(1.0, 1, false).is_err());
    assert!(base.refine_direct_sigma(0.12, 0, false).is_err());

    // A non-flow-match (linear/Euler) schedule is rejected.
    let euler = DiffusionSchedule::linear(4).unwrap();
    assert!(euler.refine_direct_sigma(0.12, 1, false).is_err());
}

#[test]
fn flow_match_refine_noise_then_euler_step_recovers_clean_latent() {
    // The refine round trip: inject flow-match noise into a clean x0, then take
    // one Euler step with the exact flow-match velocity (noise - x0). The result
    // must recover x0 -- the property the additive noising would violate.
    let base = flow_match_base_schedule_for_refine();
    let refine = base.refine_direct_sigma(0.12, 1, false).unwrap();

    let x0 = vec![0.5_f32, -0.25, 1.0, 0.0];
    let noise = vec![0.9_f32, 0.1, -0.4, 0.2];
    let mut latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 4,
        data: x0.clone(),
    };

    refine
        .add_flow_match_refine_noise(&mut latents, &noise)
        .unwrap();
    // x = (1 - 0.12) * x0 + 0.12 * noise
    for ((got, x0v), n) in latents.data.iter().zip(&x0).zip(&noise) {
        let expected = 0.88 * x0v + 0.12 * n;
        assert!((got - expected).abs() < 1e-6, "noised: {got} vs {expected}");
    }

    // Flow-match velocity target for x_t = (1-s) x0 + s noise is (noise - x0).
    let velocity: Vec<f32> = noise.iter().zip(&x0).map(|(n, x)| n - x).collect();
    let mut state = SchedulerStepState::default();
    refine.step(&mut latents, &velocity, 0, &mut state).unwrap();

    for (got, x0v) in latents.data.iter().zip(&x0) {
        assert!((got - x0v).abs() < 1e-6, "recovered: {got} vs {x0v}");
    }
}

#[test]
fn karras_scheduler_uses_power_law_sigmas_and_nearest_train_timesteps() {
    let mut config = SchedulerConfig {
        class_name: "DPMSolverMultistepScheduler".into(),
        beta_start: Some(0.00085),
        beta_end: Some(0.012),
        beta_schedule: Some("scaled_linear".into()),
        num_train_timesteps: Some(1000),
        prediction_type: Some("epsilon".into()),
        algorithm_type: Some("dpmsolver++".into()),
        solver_order: Some(2),
        solver_type: Some("midpoint".into()),
        lower_order_final: Some(true),
        thresholding: Some(false),
        timestep_spacing: Some("linspace".into()),
        steps_offset: Some(1),
        use_karras_sigmas: Some(false),
        set_alpha_to_one: None,
        ..SchedulerConfig::default()
    };
    let normal = DiffusionSchedule::from_config(&config, 4).unwrap();
    config.use_karras_sigmas = Some(true);

    let karras = DiffusionSchedule::from_config(&config, 4).unwrap();

    assert_eq!(karras.sigmas.len(), 5);
    assert!((karras.sigmas[0] - normal.sigmas[0]).abs() / normal.sigmas[0].max(1.0) < 1e-4);
    assert_eq!(karras.sigmas[4], 0.0);
    assert!(karras.sigmas[0] > karras.sigmas[1]);
    assert!(karras.sigmas[1] > karras.sigmas[2]);
    assert!(karras.sigmas[2] > karras.sigmas[3]);
    assert_ne!(karras.sigmas, normal.sigmas);
    assert_eq!(karras.train_timesteps.len(), 4);
    assert!(karras
        .train_timesteps
        .windows(2)
        .all(|pair| pair[0] >= pair[1]));
}

#[test]
fn scheduler_request_aliases_select_actual_sampler_config() {
    let config = tiny_sd_scheduler_config_for_tests();

    let dpm = config.resolve_request_scheduler("DPM++ 2M").unwrap();
    let dpm_karras = config.resolve_request_scheduler("DPM++ 2M Karras").unwrap();
    let dpm3 = config.resolve_request_scheduler("DPM++ 3M").unwrap();
    let dpm3_karras = config.resolve_request_scheduler("DPM++ 3M Karras").unwrap();
    let euler = config.resolve_request_scheduler("Euler").unwrap();
    let euler_karras = config.resolve_request_scheduler("Euler Karras").unwrap();
    let euler_a = config.resolve_request_scheduler("Euler a").unwrap();
    let ddim = config.resolve_request_scheduler("DDIM").unwrap();

    assert_eq!(dpm.class_name, "DPMSolverMultistepScheduler");
    assert_eq!(dpm_karras.class_name, "DPMSolverMultistepScheduler");
    assert_eq!(dpm_karras.use_karras_sigmas, Some(true));
    assert_eq!(dpm3.class_name, "DPMSolverMultistepScheduler");
    assert_eq!(dpm3.algorithm_type.as_deref(), Some("dpmsolver++"));
    assert_eq!(dpm3.solver_order, Some(3));
    assert_eq!(dpm3_karras.solver_order, Some(3));
    assert_eq!(dpm3_karras.use_karras_sigmas, Some(true));
    assert_eq!(euler.class_name, "EulerDiscreteScheduler");
    assert_eq!(euler.algorithm_type, None);
    assert_eq!(euler_karras.class_name, "EulerDiscreteScheduler");
    assert_eq!(euler_karras.use_karras_sigmas, Some(true));
    assert_eq!(euler_a.class_name, "EulerAncestralDiscreteScheduler");
    assert_eq!(ddim.class_name, "DDIMScheduler");
    assert!(config.resolve_request_scheduler("not a sampler").is_err());
}

#[test]
fn scheduler_request_alias_changes_run_plan_schedule() {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-diffusion-scheduler-alias-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let hfq_path = dir.join("tiny-scheduler-alias.hfq");
    let metadata = tiny_runtime_metadata();
    write_hfqm_package_mem(
        &hfq_path,
        HFQ_ARCH_DIFFUSION,
        &serde_json::to_string(&metadata).unwrap(),
        &tiny_complete_runtime_tensors(),
    )
    .unwrap();
    let mut pipeline = DiffusionPipeline::open_hfq(&hfq_path).unwrap();
    pipeline.config.scheduler = tiny_sd_scheduler_config_for_tests();
    let mut request = DiffusionBatchRequest {
        conditioning: None,

        prompts: vec![DiffusionPrompt {
            prompt: "a".into(),
            negative_prompt: String::new(),
            seed: 1,
            subseed: None,
        }],
        width: 2,
        height: 2,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        seed_resize_from_width: None,
        seed_resize_from_height: None,
        crop_x: 0,
        crop_y: 0,
        steps: 2,
        cfg_scale: 1.0,
        distilled_guidance_scale: None,
        scheduler: "DPM++ 2M".into(),
        subseed_strength: 0.0,
        send_images: true,
        save_images: false,
    };

    let dpm_plan = pipeline.prepare_run_plan(&request).unwrap();
    request.scheduler = "Euler".into();
    let euler_plan = pipeline.prepare_run_plan(&request).unwrap();
    request.scheduler = "DDIM".into();
    let ddim_plan = pipeline.prepare_run_plan(&request).unwrap();

    assert!(matches!(
        dpm_plan.schedule.solver,
        SchedulerSolver::DpmSolverMultistep { .. }
    ));
    assert_eq!(euler_plan.schedule.solver, SchedulerSolver::Euler);
    assert_eq!(
        euler_plan.schedule.input_scaling,
        SchedulerInputScaling::Sigma
    );
    assert_eq!(
        ddim_plan.schedule.solver,
        SchedulerSolver::Ddim {
            set_alpha_to_one: true
        }
    );
    assert_eq!(
        ddim_plan.schedule.input_scaling,
        SchedulerInputScaling::None
    );
    assert_ne!(dpm_plan.latents.data, euler_plan.latents.data);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn ddim_scheduler_step_matches_deterministic_epsilon_update() {
    let schedule = DiffusionSchedule {
        timesteps: vec![2.0, 1.0],
        sigmas: vec![0.8, 0.6, 0.0],
        prediction_type: SchedulerPredictionType::Epsilon,
        input_scaling: SchedulerInputScaling::None,
        solver: SchedulerSolver::Ddim {
            set_alpha_to_one: true,
        },
        train_timesteps: vec![2, 1],
        alpha_t: vec![1.0, 0.8, 0.6],
        sigma_t: vec![0.0, 0.6, 0.8],
        lambda_t: Vec::new(),
    };
    let mut latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 1,
        data: vec![1.4],
    };
    let mut state = SchedulerStepState::default();

    schedule.step(&mut latents, &[0.5], 0, &mut state).unwrap();

    let pred_original = (1.4 - 0.8 * 0.5) / 0.6;
    let expected = 0.8 * pred_original + 0.6 * 0.5;
    assert!((latents.data[0] - expected).abs() < 1e-6);
}

#[test]
fn dpm_solver_multistep_updates_with_model_output_history() {
    let lambda = |alpha: f32, sigma: f32| alpha.ln() - sigma.ln();
    let schedule = DiffusionSchedule {
        timesteps: vec![2.0, 1.0],
        sigmas: vec![0.3, 0.2, 0.0],
        prediction_type: SchedulerPredictionType::Epsilon,
        input_scaling: SchedulerInputScaling::None,
        solver: SchedulerSolver::DpmSolverMultistep {
            algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
            solver_order: 2,
            solver_type: DpmSolverType::Midpoint,
            lower_order_final: false,
            thresholding: false,
            dynamic_thresholding_ratio: 0.995,
            sample_max_value: 1.0,
        },
        train_timesteps: vec![2, 1],
        alpha_t: vec![0.9, 0.8, 0.7],
        sigma_t: vec![0.1, 0.2, 0.3],
        lambda_t: vec![lambda(0.9, 0.1), lambda(0.8, 0.2), lambda(0.7, 0.3)],
    };
    let mut latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 1,
        data: vec![1.0],
    };
    let mut state = SchedulerStepState::default();

    schedule.step(&mut latents, &[0.5], 0, &mut state).unwrap();
    let first = latents.data[0];
    schedule.step(&mut latents, &[0.25], 1, &mut state).unwrap();

    assert_eq!(state.lower_order_nums, 2);
    assert_eq!(state.model_outputs.len(), 2);
    assert!(first.is_finite());
    assert!(latents.data[0].is_finite());
    assert_ne!(latents.data[0], first);
}

#[test]
fn dpm_solver_dynamic_thresholding_clips_predicted_original_sample() {
    let schedule = DiffusionSchedule {
        timesteps: vec![0.0],
        sigmas: vec![0.0, 0.0],
        prediction_type: SchedulerPredictionType::Sample,
        input_scaling: SchedulerInputScaling::None,
        solver: SchedulerSolver::DpmSolverMultistep {
            algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
            solver_order: 2,
            solver_type: DpmSolverType::Midpoint,
            lower_order_final: true,
            thresholding: true,
            dynamic_thresholding_ratio: 1.0,
            sample_max_value: 4.0,
        },
        train_timesteps: vec![0],
        alpha_t: vec![1.0],
        sigma_t: vec![0.0],
        lambda_t: vec![0.0],
    };
    let sample = CpuTensor {
        shape: vec![2, 1, 1, 4],
        data: vec![0.0; 8],
    };
    let model_output = [-0.5, 0.5, 2.0, -4.0, 0.2, -3.0, 6.0, -9.0];

    let output = schedule
        .dpm_convert_model_output(&model_output, 0, &sample)
        .unwrap();

    assert_eq!(
        output,
        vec![-0.125, 0.125, 0.5, -1.0, 0.05, -0.75, 1.0, -1.0]
    );
}

#[test]
fn dpm_solver_third_order_update_matches_diffusers_formula() {
    let lambda = |alpha: f32, sigma: f32| alpha.ln() - sigma.ln();
    let schedule = DiffusionSchedule {
        timesteps: vec![3.0, 2.0, 1.0],
        sigmas: vec![0.4, 0.3, 0.2, 0.0],
        prediction_type: SchedulerPredictionType::Sample,
        input_scaling: SchedulerInputScaling::None,
        solver: SchedulerSolver::DpmSolverMultistep {
            algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
            solver_order: 3,
            solver_type: DpmSolverType::Midpoint,
            lower_order_final: false,
            thresholding: false,
            dynamic_thresholding_ratio: 0.995,
            sample_max_value: 1.0,
        },
        train_timesteps: vec![3, 2, 1],
        alpha_t: vec![0.95, 0.85, 0.75, 0.65],
        sigma_t: vec![0.10, 0.20, 0.30, 0.40],
        lambda_t: vec![
            lambda(0.95, 0.10),
            lambda(0.85, 0.20),
            lambda(0.75, 0.30),
            lambda(0.65, 0.40),
        ],
    };
    let sample = CpuTensor {
        shape: vec![1, 1, 1, 1],
        data: vec![1.25],
    };
    let state = SchedulerStepState {
        model_outputs: vec![vec![0.20], vec![0.40], vec![0.70]],
        lower_order_nums: 2,
    };

    let next = schedule
        .dpm_third_order_update(3, 2, 1, 0, &sample, &state)
        .unwrap();

    let lambda_t = schedule.scheduler_lambda(0).unwrap();
    let lambda_s0 = schedule.scheduler_lambda(1).unwrap();
    let lambda_s1 = schedule.scheduler_lambda(2).unwrap();
    let lambda_s2 = schedule.scheduler_lambda(3).unwrap();
    let h = lambda_t - lambda_s0;
    let h0 = lambda_s0 - lambda_s1;
    let h1 = lambda_s1 - lambda_s2;
    let r0 = h0 / h;
    let r1 = h1 / h;
    let m0 = 0.70;
    let m1 = 0.40;
    let m2 = 0.20;
    let d1_0 = (m0 - m1) / r0;
    let d1_1 = (m1 - m2) / r1;
    let d1 = d1_0 + (r0 / (r0 + r1)) * (d1_0 - d1_1);
    let d2 = (d1_0 - d1_1) / (r0 + r1);
    let exp_neg_h = (-h).exp();
    let expected = (schedule.scheduler_sigma(0).unwrap() / schedule.scheduler_sigma(1).unwrap())
        * sample.data[0]
        - (schedule.scheduler_alpha(0).unwrap() * (exp_neg_h - 1.0)) * m0
        + (schedule.scheduler_alpha(0).unwrap() * ((exp_neg_h - 1.0) / h + 1.0)) * d1
        - (schedule.scheduler_alpha(0).unwrap() * ((exp_neg_h - 1.0 + h) / (h * h) - 0.5)) * d2;

    assert!((next.data[0] - expected).abs() < 1e-6);
}

#[test]
fn dpm_solver_order_three_step_uses_third_order_history() {
    let lambda = |alpha: f32, sigma: f32| alpha.ln() - sigma.ln();
    let schedule = DiffusionSchedule {
        timesteps: vec![3.0, 2.0, 1.0],
        sigmas: vec![0.4, 0.3, 0.2, 0.0],
        prediction_type: SchedulerPredictionType::Sample,
        input_scaling: SchedulerInputScaling::None,
        solver: SchedulerSolver::DpmSolverMultistep {
            algorithm_type: DpmSolverAlgorithm::DpmSolverPlusPlus,
            solver_order: 3,
            solver_type: DpmSolverType::Midpoint,
            lower_order_final: false,
            thresholding: false,
            dynamic_thresholding_ratio: 0.995,
            sample_max_value: 1.0,
        },
        train_timesteps: vec![3, 2, 1],
        alpha_t: vec![0.95, 0.85, 0.75, 0.65],
        sigma_t: vec![0.10, 0.20, 0.30, 0.40],
        lambda_t: vec![
            lambda(0.95, 0.10),
            lambda(0.85, 0.20),
            lambda(0.75, 0.30),
            lambda(0.65, 0.40),
        ],
    };
    let mut latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 1,
        data: vec![1.0],
    };
    let mut state = SchedulerStepState::default();

    schedule.step(&mut latents, &[0.20], 0, &mut state).unwrap();
    schedule.step(&mut latents, &[0.40], 1, &mut state).unwrap();
    let second = latents.data[0];
    schedule.step(&mut latents, &[0.70], 2, &mut state).unwrap();

    assert_eq!(state.lower_order_nums, 3);
    assert_eq!(state.model_outputs.len(), 3);
    assert!(latents.data[0].is_finite());
    assert_ne!(latents.data[0], second);
}

#[test]
fn scheduler_config_falls_back_to_linear_when_beta_metadata_is_missing() {
    let schedule = DiffusionSchedule::from_config(&SchedulerConfig::default(), 2).unwrap();

    assert_eq!(schedule.timesteps, vec![1.0, 0.0]);
    assert_eq!(schedule.sigmas, vec![1.0, 0.0, 0.0]);
    assert_eq!(schedule.prediction_type, SchedulerPredictionType::Epsilon);
    assert_eq!(schedule.input_scaling, SchedulerInputScaling::None);
}

#[test]
fn denoise_progress_callback_can_interrupt_generation() {
    let latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 1,
        data: vec![0.0],
    };
    let schedule = DiffusionSchedule::from_config(&SchedulerConfig::default(), 2).unwrap();
    let positive = CpuTensor {
        shape: vec![1, 1, 1],
        data: vec![0.0],
    };
    let negative = positive.clone();
    let mut events = Vec::new();

    let error = denoise_latents_with_cfg_progress(
        latents,
        &schedule,
        1.0,
        &positive,
        &negative,
        |sample, _timesteps, _encoder_states, _attention_mask, _sdxl_conditioning| {
            Ok(CpuTensor {
                shape: sample.shape.clone(),
                data: vec![0.0; sample.data.len()],
            })
        },
        None,
        None,
        None,
        None,
        None,
        None,
        Some(&mut |progress| {
            events.push(progress);
            Err(DiffusionError::Interrupted("test interrupt".to_string()))
        }),
    )
    .unwrap_err();

    assert!(matches!(error, DiffusionError::Interrupted(_)));
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].completed_steps, 1);
    assert_eq!(events[0].total_steps, 2);
}

#[test]
fn scheduler_scales_model_input_for_euler_class() {
    let config = SchedulerConfig {
        class_name: "EulerDiscreteScheduler".into(),
        beta_start: Some(0.0001),
        beta_end: Some(0.02),
        beta_schedule: Some("linear".into()),
        num_train_timesteps: Some(1000),
        prediction_type: Some("epsilon".into()),
        ..SchedulerConfig::default()
    };
    let schedule = DiffusionSchedule::from_config(&config, 1).unwrap();
    let sample = CpuTensor {
        shape: vec![1, 1, 1, 1],
        data: vec![2.0],
    };

    let scaled = schedule.scale_model_input(&sample, 0).unwrap();

    assert_eq!(schedule.input_scaling, SchedulerInputScaling::Sigma);
    assert!(scaled.data[0] < sample.data[0]);
}

#[test]
fn scheduler_scales_initial_latents_for_euler_class() {
    let config = SchedulerConfig {
        class_name: "EulerDiscreteScheduler".into(),
        beta_start: Some(0.0001),
        beta_end: Some(0.02),
        beta_schedule: Some("linear".into()),
        num_train_timesteps: Some(1000),
        prediction_type: Some("epsilon".into()),
        ..SchedulerConfig::default()
    };
    let schedule = DiffusionSchedule::from_config(&config, 2).unwrap();
    let mut latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 2,
        data: vec![1.0, -2.0],
    };
    let sigma = schedule.initial_noise_sigma();

    schedule.scale_initial_latents(&mut latents);

    assert!(sigma > 1.0);
    assert_eq!(latents.data, vec![sigma, -2.0 * sigma]);
}

#[test]
fn scheduler_step_supports_sample_prediction_type() {
    let mut schedule = DiffusionSchedule::linear(1).unwrap();
    schedule.prediction_type = SchedulerPredictionType::Sample;
    let mut latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 1,
        data: vec![2.0],
    };

    schedule.euler_step(&mut latents, &[1.5], 0).unwrap();

    assert_eq!(latents.data, vec![1.5]);
}

#[test]
fn scheduler_step_supports_v_prediction_type() {
    let mut schedule = DiffusionSchedule::linear(1).unwrap();
    schedule.prediction_type = SchedulerPredictionType::VPrediction;
    let mut latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 1,
        data: vec![2.0],
    };

    schedule.euler_step(&mut latents, &[0.5], 0).unwrap();

    let expected = 2.0 - scheduler_derivative(2.0, 0.5, 1.0, SchedulerPredictionType::VPrediction);
    assert!((latents.data[0] - expected).abs() < 1e-6);
}

#[test]
fn denoise_loop_applies_classifier_free_guidance() {
    let latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 2,
        data: vec![1.0, -1.0],
    };
    let schedule = DiffusionSchedule::linear(1).unwrap();
    let positive = CpuTensor {
        shape: vec![1, 1, 1],
        data: vec![1.0],
    };
    let negative = CpuTensor {
        shape: vec![1, 1, 1],
        data: vec![0.0],
    };
    let out = denoise_latents_with_cfg(
        latents,
        &schedule,
        2.0,
        &positive,
        &negative,
        // Batch-aware mock: batched CFG fuses the uncond/cond passes into one
        // call with encoder rows [positive; negative], so emit a prediction
        // per batch row (negative row value 0.0 -> [0.25,-0.25], else
        // positive -> [0.75,0.25]). Same per-row predictions as before.
        |_sample, _timesteps, encoder| {
            let rows = encoder.shape[0];
            let mut data = Vec::with_capacity(rows * 2);
            for r in 0..rows {
                if encoder.data[r] == 0.0 {
                    data.extend_from_slice(&[0.25, -0.25]);
                } else {
                    data.extend_from_slice(&[0.75, 0.25]);
                }
            }
            Ok(CpuTensor {
                shape: vec![rows, 1, 1, 2],
                data,
            })
        },
    )
    .unwrap();

    assert_eq!(out.data, vec![-0.25, -1.75]);
}

#[test]
fn denoise_loop_skips_negative_prediction_when_cfg_is_identity() {
    let latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 2,
        data: vec![1.0, -1.0],
    };
    let schedule = DiffusionSchedule::linear(1).unwrap();
    let positive = CpuTensor {
        shape: vec![1, 1, 1],
        data: vec![1.0],
    };
    let negative = CpuTensor {
        shape: vec![1, 1, 1],
        data: vec![0.0],
    };
    let mut calls = 0usize;
    let out = denoise_latents_with_cfg(
        latents,
        &schedule,
        1.0,
        &positive,
        &negative,
        |_sample, _timesteps, encoder| {
            calls += 1;
            assert_eq!(encoder.data, positive.data);
            Ok(CpuTensor {
                shape: vec![1, 1, 1, 2],
                data: vec![0.75, 0.25],
            })
        },
    )
    .unwrap();

    assert_eq!(calls, 1);
    assert_eq!(out.batch, 1);
    assert_eq!(out.channels, 1);
    assert_eq!(out.height, 1);
    assert_eq!(out.width, 2);
}

#[test]
fn denoise_loop_uses_scheduler_model_input_scaling() {
    let latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 1,
        data: vec![2.0],
    };
    let mut schedule = DiffusionSchedule::linear(1).unwrap();
    schedule.input_scaling = SchedulerInputScaling::Sigma;
    let positive = CpuTensor {
        shape: vec![1, 1, 1],
        data: vec![1.0],
    };
    let negative = CpuTensor {
        shape: vec![1, 1, 1],
        data: vec![0.0],
    };
    let mut seen_sample = None;
    let _ = denoise_latents_with_cfg(
        latents,
        &schedule,
        1.0,
        &positive,
        &negative,
        |sample, _timesteps, _encoder| {
            seen_sample.get_or_insert(sample.data[0]);
            Ok(CpuTensor {
                shape: vec![1, 1, 1, 1],
                data: vec![0.0],
            })
        },
    )
    .unwrap();

    assert!((seen_sample.unwrap() - std::f32::consts::SQRT_2).abs() < 1e-6);
}

#[test]
fn denoise_loop_rejects_bad_conditioning_and_noise_shapes() {
    let latents = LatentBatch {
        batch: 1,
        channels: 1,
        height: 1,
        width: 1,
        data: vec![0.0],
    };
    let schedule = DiffusionSchedule::linear(1).unwrap();
    let positive = CpuTensor {
        shape: vec![1, 1, 1],
        data: vec![1.0],
    };
    let negative_bad_batch = CpuTensor {
        shape: vec![2, 1, 1],
        data: vec![0.0, 0.0],
    };
    assert!(denoise_latents_with_cfg(
        latents.clone(),
        &schedule,
        1.0,
        &positive,
        &negative_bad_batch,
        |_sample, _timesteps, _encoder| unreachable!(),
    )
    .is_err());

    let negative = CpuTensor {
        shape: vec![1, 1, 1],
        data: vec![0.0],
    };
    assert!(denoise_latents_with_cfg(
        latents,
        &schedule,
        1.0,
        &positive,
        &negative,
        |_sample, _timesteps, _encoder| Ok(CpuTensor {
            shape: vec![1, 1, 1, 2],
            data: vec![0.0, 0.0],
        }),
    )
    .is_err());
}
