// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Diffusion sampling schedules: beta/sigma construction (linear, scaled-linear,
//! cosine, Karras), timestep spacing, and the Euler / DDIM / flow-match /
//! DPM-Solver++ step implementations. Ported to match diffusers semantics.

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionSchedule {
    pub timesteps: Vec<f32>,
    pub sigmas: Vec<f32>,
    pub prediction_type: SchedulerPredictionType,
    pub input_scaling: SchedulerInputScaling,
    pub solver: SchedulerSolver,
    pub(crate) train_timesteps: Vec<usize>,
    pub(crate) alpha_t: Vec<f32>,
    pub(crate) sigma_t: Vec<f32>,
    pub(crate) lambda_t: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SchedulerSolver {
    Euler,
    FlowMatchEuler,
    Ddim {
        set_alpha_to_one: bool,
    },
    DpmSolverMultistep {
        algorithm_type: DpmSolverAlgorithm,
        solver_order: usize,
        solver_type: DpmSolverType,
        lower_order_final: bool,
        thresholding: bool,
        dynamic_thresholding_ratio: f32,
        sample_max_value: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpmSolverAlgorithm {
    DpmSolverPlusPlus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpmSolverType {
    Midpoint,
    Heun,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SchedulerStepState {
    pub(crate) model_outputs: Vec<Vec<f32>>,
    pub(crate) lower_order_nums: usize,
}

impl SchedulerSolver {
    pub(crate) fn from_config(config: &SchedulerConfig) -> DiffusionResult<Self> {
        if config.class_name == "FlowMatchEulerDiscreteScheduler" {
            return Ok(Self::FlowMatchEuler);
        }
        if config.class_name == "DDIMScheduler" {
            return Ok(Self::Ddim {
                set_alpha_to_one: config.set_alpha_to_one.unwrap_or(true),
            });
        }
        if config.class_name != "DPMSolverMultistepScheduler" {
            return Ok(Self::Euler);
        }
        let algorithm_type = match config.algorithm_type.as_deref().unwrap_or("dpmsolver++") {
            "dpmsolver++" => DpmSolverAlgorithm::DpmSolverPlusPlus,
            other => {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "unsupported DPM-Solver algorithm_type {other:?}"
                )));
            }
        };
        let solver_type = match config.solver_type.as_deref().unwrap_or("midpoint") {
            "midpoint" => DpmSolverType::Midpoint,
            "heun" => DpmSolverType::Heun,
            other => {
                return Err(DiffusionError::InvalidMetadata(format!(
                    "unsupported DPM-Solver solver_type {other:?}"
                )));
            }
        };
        let solver_order = config.solver_order.unwrap_or(2);
        if !(1..=3).contains(&solver_order) {
            return Err(DiffusionError::InvalidMetadata(format!(
                "unsupported DPM-Solver solver_order {solver_order}; only 1, 2, and 3 are implemented"
            )));
        }
        Ok(Self::DpmSolverMultistep {
            algorithm_type,
            solver_order,
            solver_type,
            lower_order_final: config.lower_order_final.unwrap_or(true),
            thresholding: config.thresholding.unwrap_or(false),
            dynamic_thresholding_ratio: normalize_dynamic_thresholding_ratio(
                config.dynamic_thresholding_ratio,
            ),
            sample_max_value: normalize_dynamic_thresholding_sample_max(config.sample_max_value),
        })
    }
}

fn normalize_dynamic_thresholding_ratio(value: Option<f32>) -> f32 {
    match value {
        Some(value) if value.is_finite() => value.clamp(0.0, 1.0),
        _ => 0.995,
    }
}

fn normalize_dynamic_thresholding_sample_max(value: Option<f32>) -> f32 {
    match value {
        Some(value) if value.is_finite() => value.max(1.0),
        _ => 1.0,
    }
}

fn dynamic_threshold_sample(
    data: &mut [f32],
    shape: &[usize],
    ratio: f32,
    sample_max_value: f32,
) -> DiffusionResult<()> {
    let batch = shape.first().copied().ok_or_else(|| {
        DiffusionError::InvalidMetadata(
            "DPM-Solver dynamic thresholding requires a batch dimension".to_string(),
        )
    })?;
    if batch == 0 || data.is_empty() {
        return Ok(());
    }
    if !data.len().is_multiple_of(batch) {
        return Err(DiffusionError::InvalidMetadata(format!(
            "DPM-Solver dynamic thresholding data length {} is not divisible by batch {batch}",
            data.len()
        )));
    }
    let values_per_batch = data.len() / batch;
    if values_per_batch == 0 {
        return Ok(());
    }

    let ratio = normalize_dynamic_thresholding_ratio(Some(ratio));
    let sample_max_value = normalize_dynamic_thresholding_sample_max(Some(sample_max_value));
    let mut sorted_abs = Vec::with_capacity(values_per_batch);
    for chunk in data.chunks_mut(values_per_batch) {
        sorted_abs.clear();
        sorted_abs.extend(chunk.iter().map(|value| value.abs()));
        sorted_abs.sort_by(|left, right| left.total_cmp(right));

        let threshold = if sorted_abs.len() == 1 {
            sorted_abs[0]
        } else {
            let rank = ratio * (sorted_abs.len() - 1) as f32;
            let lower = rank.floor() as usize;
            let upper = rank.ceil() as usize;
            let frac = rank - lower as f32;
            sorted_abs[lower] + (sorted_abs[upper] - sorted_abs[lower]) * frac
        };
        let threshold = threshold.clamp(1.0, sample_max_value);
        for value in chunk {
            *value = value.clamp(-threshold, threshold) / threshold;
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerPredictionType {
    Epsilon,
    Sample,
    VPrediction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerInputScaling {
    None,
    Sigma,
}

impl SchedulerInputScaling {
    pub(crate) fn from_scheduler_class(class_name: &str) -> Self {
        match class_name {
            "EulerDiscreteScheduler" | "EulerAncestralDiscreteScheduler" => Self::Sigma,
            _ => Self::None,
        }
    }
}

impl SchedulerPredictionType {
    pub(crate) fn from_config(value: Option<&str>) -> DiffusionResult<Self> {
        match value.unwrap_or("epsilon") {
            "epsilon" => Ok(Self::Epsilon),
            "sample" => Ok(Self::Sample),
            "v_prediction" => Ok(Self::VPrediction),
            other => Err(DiffusionError::InvalidMetadata(format!(
                "unsupported scheduler prediction_type {other:?}"
            ))),
        }
    }
}

impl DiffusionSchedule {
    pub fn linear(steps: u32) -> DiffusionResult<Self> {
        if steps == 0 {
            return Err(DiffusionError::InvalidRequest(
                "scheduler steps must be greater than zero".to_string(),
            ));
        }
        let steps = steps as usize;
        let mut timesteps = Vec::with_capacity(steps);
        let mut sigmas = Vec::with_capacity(steps + 1);
        for idx in 0..steps {
            let frac = if steps == 1 {
                1.0
            } else {
                1.0 - idx as f32 / (steps - 1) as f32
            };
            timesteps.push(frac);
            sigmas.push(frac);
        }
        sigmas.push(0.0);
        Ok(Self {
            timesteps,
            sigmas,
            prediction_type: SchedulerPredictionType::Epsilon,
            input_scaling: SchedulerInputScaling::None,
            solver: SchedulerSolver::Euler,
            train_timesteps: Vec::new(),
            alpha_t: Vec::new(),
            sigma_t: Vec::new(),
            lambda_t: Vec::new(),
        })
    }

    pub fn from_config(config: &SchedulerConfig, steps: u32) -> DiffusionResult<Self> {
        if steps == 0 {
            return Err(DiffusionError::InvalidRequest(
                "scheduler steps must be greater than zero".to_string(),
            ));
        }
        if config.class_name == "FlowMatchEulerDiscreteScheduler" {
            return Self::flow_match_euler(config, steps);
        }
        let (Some(beta_start), Some(beta_end), Some(num_train_timesteps)) = (
            config.beta_start,
            config.beta_end,
            config.num_train_timesteps,
        ) else {
            let mut schedule = Self::linear(steps)?;
            schedule.prediction_type =
                SchedulerPredictionType::from_config(config.prediction_type.as_deref())?;
            schedule.input_scaling =
                SchedulerInputScaling::from_scheduler_class(&config.class_name);
            return Ok(schedule);
        };
        if beta_start <= 0.0 || beta_end <= 0.0 || num_train_timesteps == 0 {
            let mut schedule = Self::linear(steps)?;
            schedule.prediction_type =
                SchedulerPredictionType::from_config(config.prediction_type.as_deref())?;
            schedule.input_scaling =
                SchedulerInputScaling::from_scheduler_class(&config.class_name);
            return Ok(schedule);
        }
        let prediction_type =
            SchedulerPredictionType::from_config(config.prediction_type.as_deref())?;
        let input_scaling = SchedulerInputScaling::from_scheduler_class(&config.class_name);
        let solver = SchedulerSolver::from_config(config)?;
        let betas = scheduler_betas(
            beta_start,
            beta_end,
            num_train_timesteps,
            config.beta_schedule.as_deref().unwrap_or("linear"),
        )?;
        let alpha_cumprod = betas
            .iter()
            .scan(1.0f32, |acc, beta| {
                *acc *= 1.0 - beta;
                Some(*acc)
            })
            .collect::<Vec<_>>();
        let mut train_indices =
            inference_train_timesteps(config, num_train_timesteps, steps as usize)?;
        let mut timesteps = Vec::with_capacity(train_indices.len());
        let mut sigmas = Vec::with_capacity(train_indices.len() + 1);
        for idx in &train_indices {
            let alpha = alpha_cumprod[*idx].clamp(f32::MIN_POSITIVE, 1.0);
            timesteps.push(*idx as f32);
            sigmas.push(((1.0 - alpha) / alpha).max(0.0).sqrt());
        }
        sigmas.push(0.0);
        if config.use_karras_sigmas.unwrap_or(false) && sigmas.len() > 1 {
            let training_sigmas = alpha_cumprod
                .iter()
                .map(|alpha| {
                    let alpha = alpha.clamp(f32::MIN_POSITIVE, 1.0);
                    ((1.0 - alpha) / alpha).max(0.0).sqrt()
                })
                .collect::<Vec<_>>();
            sigmas = karras_sigmas(&sigmas[..sigmas.len() - 1]);
            train_indices = sigmas[..sigmas.len() - 1]
                .iter()
                .map(|sigma| nearest_training_timestep_for_sigma(&training_sigmas, *sigma))
                .collect();
            timesteps = train_indices.iter().map(|idx| *idx as f32).collect();
        }
        let mut alpha_t = Vec::with_capacity(alpha_cumprod.len());
        let mut sigma_t = Vec::with_capacity(alpha_cumprod.len());
        let mut lambda_t = Vec::with_capacity(alpha_cumprod.len());
        for alpha_cumprod in &alpha_cumprod {
            let alpha = alpha_cumprod.clamp(f32::MIN_POSITIVE, 1.0).sqrt();
            let sigma = (1.0 - alpha_cumprod).max(f32::MIN_POSITIVE).sqrt();
            alpha_t.push(alpha);
            sigma_t.push(sigma);
            lambda_t.push(alpha.ln() - sigma.ln());
        }
        Ok(Self {
            timesteps,
            sigmas,
            prediction_type,
            input_scaling,
            solver,
            train_timesteps: train_indices,
            alpha_t,
            sigma_t,
            lambda_t,
        })
    }

    pub(crate) fn flow_match_euler(config: &SchedulerConfig, steps: u32) -> DiffusionResult<Self> {
        let steps = steps as usize;
        let train_timesteps = config.num_train_timesteps.unwrap_or(1000).max(1);
        let shift = config.shift.unwrap_or(1.0).max(f32::MIN_POSITIVE);
        let mut sigmas = Vec::with_capacity(steps + 1);
        for idx in 0..steps {
            let frac = if steps == 1 {
                1.0
            } else {
                1.0 - idx as f32 / (steps - 1) as f32
            };
            let sigma = if (shift - 1.0).abs() <= f32::EPSILON {
                frac
            } else {
                (shift * frac) / (1.0 + (shift - 1.0) * frac)
            };
            sigmas.push(sigma.clamp(0.0, 1.0));
        }
        if config.invert_sigmas.unwrap_or(false) {
            for sigma in &mut sigmas {
                *sigma = 1.0 - *sigma;
            }
            sigmas.reverse();
        }
        if let Some(terminal) = config.shift_terminal {
            rescale_sigmas_to_terminal(&mut sigmas, terminal.clamp(0.0, 1.0));
        }
        let timesteps = sigmas
            .iter()
            .map(|sigma| sigma * train_timesteps as f32)
            .collect::<Vec<_>>();
        sigmas.push(0.0);
        Ok(Self {
            timesteps,
            sigmas,
            prediction_type: SchedulerPredictionType::Sample,
            input_scaling: SchedulerInputScaling::None,
            solver: SchedulerSolver::FlowMatchEuler,
            train_timesteps: Vec::new(),
            alpha_t: Vec::new(),
            sigma_t: Vec::new(),
            lambda_t: Vec::new(),
        })
    }

    pub fn scale_model_input(&self, sample: &CpuTensor, step: usize) -> DiffusionResult<CpuTensor> {
        match self.input_scaling {
            SchedulerInputScaling::None => Ok(sample.clone()),
            SchedulerInputScaling::Sigma => {
                let sigma = *self.sigmas.get(step).ok_or_else(|| {
                    DiffusionError::InvalidRequest(format!("missing sigma for step {step}"))
                })?;
                let scale = (sigma * sigma + 1.0).sqrt().recip();
                Ok(CpuTensor {
                    shape: sample.shape.clone(),
                    data: sample.data.iter().map(|value| value * scale).collect(),
                })
            }
        }
    }

    pub fn initial_noise_sigma(&self) -> f32 {
        match self.input_scaling {
            SchedulerInputScaling::None => 1.0,
            SchedulerInputScaling::Sigma => {
                self.sigmas.iter().copied().fold(0.0, f32::max).max(1.0)
            }
        }
    }

    pub fn scale_initial_latents(&self, latents: &mut LatentBatch) {
        let sigma = self.initial_noise_sigma();
        if (sigma - 1.0).abs() <= f32::EPSILON {
            return;
        }
        for value in &mut latents.data {
            *value *= sigma;
        }
    }

    pub fn add_noise_to_latents(
        &self,
        latents: &mut LatentBatch,
        noise: &[f32],
        step: usize,
    ) -> DiffusionResult<()> {
        if noise.len() != latents.data.len() {
            return Err(DiffusionError::InvalidRequest(format!(
                "noise length {} != latent length {}",
                noise.len(),
                latents.data.len()
            )));
        }
        if let Some(timestep) = self.train_timesteps.get(step).copied() {
            let alpha = self.scheduler_alpha(timestep)?;
            let sigma = self.scheduler_sigma(timestep)?;
            for (latent, noise) in latents.data.iter_mut().zip(noise) {
                *latent = *latent * alpha + *noise * sigma;
            }
            return Ok(());
        }
        let sigma = *self.sigmas.get(step).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("missing sigma for step {step}"))
        })?;
        for (latent, noise) in latents.data.iter_mut().zip(noise) {
            *latent += *noise * sigma;
        }
        Ok(())
    }

    pub fn slice_from_step(&self, start_step: usize) -> DiffusionResult<Self> {
        if start_step > self.timesteps.len() {
            return Err(DiffusionError::InvalidRequest(format!(
                "scheduler start step {start_step} exceeds {} steps",
                self.timesteps.len()
            )));
        }
        Ok(Self {
            timesteps: self.timesteps[start_step..].to_vec(),
            sigmas: self.sigmas[start_step..].to_vec(),
            prediction_type: self.prediction_type,
            input_scaling: self.input_scaling,
            solver: self.solver,
            train_timesteps: if self.train_timesteps.is_empty() {
                Vec::new()
            } else {
                self.train_timesteps[start_step..].to_vec()
            },
            alpha_t: self.alpha_t.clone(),
            sigma_t: self.sigma_t.clone(),
            lambda_t: self.lambda_t.clone(),
        })
    }

    pub fn euler_step(
        &self,
        latents: &mut LatentBatch,
        noise_pred: &[f32],
        step: usize,
    ) -> DiffusionResult<()> {
        if noise_pred.len() != latents.data.len() {
            return Err(DiffusionError::InvalidRequest(format!(
                "noise prediction length {} != latent length {}",
                noise_pred.len(),
                latents.data.len()
            )));
        }
        let sigma = *self.sigmas.get(step).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("missing sigma for step {step}"))
        })?;
        let next_sigma = *self.sigmas.get(step + 1).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("missing next sigma for step {step}"))
        })?;
        let dt = next_sigma - sigma;
        for (latent, model_output) in latents.data.iter_mut().zip(noise_pred) {
            let derivative =
                scheduler_derivative(*latent, *model_output, sigma, self.prediction_type);
            *latent += derivative * dt;
        }
        Ok(())
    }

    pub fn step(
        &self,
        latents: &mut LatentBatch,
        noise_pred: &[f32],
        step: usize,
        state: &mut SchedulerStepState,
    ) -> DiffusionResult<()> {
        match self.solver {
            SchedulerSolver::Euler => self.euler_step(latents, noise_pred, step),
            SchedulerSolver::FlowMatchEuler => {
                self.flow_match_euler_step(latents, noise_pred, step)
            }
            SchedulerSolver::Ddim { .. } => self.ddim_step(latents, noise_pred, step),
            SchedulerSolver::DpmSolverMultistep { .. } => {
                self.dpm_solver_multistep_step(latents, noise_pred, step, state)
            }
        }
    }

    pub(crate) fn flow_match_euler_step(
        &self,
        latents: &mut LatentBatch,
        model_output: &[f32],
        step: usize,
    ) -> DiffusionResult<()> {
        if model_output.len() != latents.data.len() {
            return Err(DiffusionError::InvalidRequest(format!(
                "noise prediction length {} != latent length {}",
                model_output.len(),
                latents.data.len()
            )));
        }
        let sigma = *self.sigmas.get(step).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("missing sigma for step {step}"))
        })?;
        let next_sigma = *self.sigmas.get(step + 1).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("missing next sigma for step {step}"))
        })?;
        let dt = next_sigma - sigma;
        for (latent, output) in latents.data.iter_mut().zip(model_output) {
            *latent += output * dt;
        }
        Ok(())
    }

    pub(crate) fn ddim_step(
        &self,
        latents: &mut LatentBatch,
        model_output: &[f32],
        step: usize,
    ) -> DiffusionResult<()> {
        if model_output.len() != latents.data.len() {
            return Err(DiffusionError::InvalidRequest(format!(
                "noise prediction length {} != latent length {}",
                model_output.len(),
                latents.data.len()
            )));
        }
        let timestep = *self.train_timesteps.get(step).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("missing DDIM timestep for step {step}"))
        })?;
        let alpha = self.scheduler_alpha(timestep)?;
        let sigma = self.scheduler_sigma(timestep)?;
        let SchedulerSolver::Ddim { set_alpha_to_one } = self.solver else {
            return self.euler_step(latents, model_output, step);
        };
        let (prev_alpha, prev_sigma) =
            if let Some(prev_timestep) = self.train_timesteps.get(step + 1) {
                (
                    self.scheduler_alpha(*prev_timestep)?,
                    self.scheduler_sigma(*prev_timestep)?,
                )
            } else if set_alpha_to_one {
                (1.0, 0.0)
            } else {
                (self.scheduler_alpha(0)?, self.scheduler_sigma(0)?)
            };
        for (sample, output) in latents.data.iter_mut().zip(model_output) {
            let (pred_original, pred_epsilon) = match self.prediction_type {
                SchedulerPredictionType::Epsilon => ((*sample - sigma * output) / alpha, *output),
                SchedulerPredictionType::Sample => {
                    let epsilon = if sigma.abs() <= f32::MIN_POSITIVE {
                        0.0
                    } else {
                        (*sample - alpha * output) / sigma
                    };
                    (*output, epsilon)
                }
                SchedulerPredictionType::VPrediction => {
                    let pred_original = alpha * *sample - sigma * output;
                    let pred_epsilon = alpha * output + sigma * *sample;
                    (pred_original, pred_epsilon)
                }
            };
            *sample = prev_alpha * pred_original + prev_sigma * pred_epsilon;
        }
        Ok(())
    }

    pub(crate) fn dpm_solver_multistep_step(
        &self,
        latents: &mut LatentBatch,
        model_output: &[f32],
        step: usize,
        state: &mut SchedulerStepState,
    ) -> DiffusionResult<()> {
        let SchedulerSolver::DpmSolverMultistep {
            solver_order,
            lower_order_final: use_lower_order_final,
            ..
        } = self.solver
        else {
            return self.euler_step(latents, model_output, step);
        };
        if model_output.len() != latents.data.len() {
            return Err(DiffusionError::InvalidRequest(format!(
                "noise prediction length {} != latent length {}",
                model_output.len(),
                latents.data.len()
            )));
        }
        let timestep = *self.train_timesteps.get(step).ok_or_else(|| {
            DiffusionError::InvalidRequest(format!("missing DPM timestep for step {step}"))
        })?;
        let prev_timestep = if step + 1 == self.train_timesteps.len() {
            0
        } else {
            self.train_timesteps[step + 1]
        };
        let sample = latents.as_nchw_tensor();
        let converted = self.dpm_convert_model_output(model_output, timestep, &sample)?;
        state.model_outputs.push(converted);
        if state.model_outputs.len() > solver_order {
            state.model_outputs.remove(0);
        }

        let lower_order_final = step + 1 == self.train_timesteps.len()
            && use_lower_order_final
            && self.train_timesteps.len() < 15;
        let lower_order_second = step + 2 == self.train_timesteps.len()
            && use_lower_order_final
            && self.train_timesteps.len() < 15;

        let prev_sample =
            if solver_order == 1 || state.lower_order_nums < 1 || lower_order_final {
                self.dpm_first_order_update(
                    state.model_outputs.last().unwrap(),
                    timestep,
                    prev_timestep,
                    &sample,
                )?
            } else if solver_order == 2 || state.lower_order_nums < 2 || lower_order_second {
                let previous_timestep = *self
                    .train_timesteps
                    .get(step.wrapping_sub(1))
                    .ok_or_else(|| {
                        DiffusionError::InvalidRequest("missing previous DPM timestep".to_string())
                    })?;
                self.dpm_second_order_update(
                    previous_timestep,
                    timestep,
                    prev_timestep,
                    &sample,
                    state,
                )?
            } else {
                let previous_timestep = *self
                    .train_timesteps
                    .get(step.wrapping_sub(1))
                    .ok_or_else(|| {
                        DiffusionError::InvalidRequest("missing previous DPM timestep".to_string())
                    })?;
                let previous_previous_timestep = *self
                    .train_timesteps
                    .get(step.wrapping_sub(2))
                    .ok_or_else(|| {
                        DiffusionError::InvalidRequest(
                            "missing second previous DPM timestep".to_string(),
                        )
                    })?;
                self.dpm_third_order_update(
                    previous_previous_timestep,
                    previous_timestep,
                    timestep,
                    prev_timestep,
                    &sample,
                    state,
                )?
            };

        latents.data = prev_sample.data;
        if state.lower_order_nums < solver_order {
            state.lower_order_nums += 1;
        }
        Ok(())
    }

    pub(crate) fn dpm_convert_model_output(
        &self,
        model_output: &[f32],
        timestep: usize,
        sample: &CpuTensor,
    ) -> DiffusionResult<Vec<f32>> {
        let SchedulerSolver::DpmSolverMultistep {
            algorithm_type,
            thresholding,
            dynamic_thresholding_ratio,
            sample_max_value,
            ..
        } = self.solver
        else {
            return Ok(model_output.to_vec());
        };
        let alpha = self.scheduler_alpha(timestep)?;
        let sigma = self.scheduler_sigma(timestep)?;
        let mut output = match algorithm_type {
            DpmSolverAlgorithm::DpmSolverPlusPlus => match self.prediction_type {
                SchedulerPredictionType::Epsilon => sample
                    .data
                    .iter()
                    .zip(model_output)
                    .map(|(sample, noise)| (sample - sigma * noise) / alpha)
                    .collect(),
                SchedulerPredictionType::Sample => model_output.to_vec(),
                SchedulerPredictionType::VPrediction => sample
                    .data
                    .iter()
                    .zip(model_output)
                    .map(|(sample, value)| alpha * sample - sigma * value)
                    .collect(),
            },
        };
        if thresholding {
            dynamic_threshold_sample(
                &mut output,
                &sample.shape,
                dynamic_thresholding_ratio,
                sample_max_value,
            )?;
        }
        Ok(output)
    }

    pub(crate) fn dpm_first_order_update(
        &self,
        model_output: &[f32],
        timestep: usize,
        prev_timestep: usize,
        sample: &CpuTensor,
    ) -> DiffusionResult<CpuTensor> {
        let lambda_t = self.scheduler_lambda(prev_timestep)?;
        let lambda_s = self.scheduler_lambda(timestep)?;
        let alpha_t = self.scheduler_alpha(prev_timestep)?;
        let sigma_t = self.scheduler_sigma(prev_timestep)?;
        let sigma_s = self.scheduler_sigma(timestep)?;
        let h = lambda_t - lambda_s;
        let data = sample
            .data
            .iter()
            .zip(model_output)
            .map(|(sample, model_output)| {
                (sigma_t / sigma_s) * sample - (alpha_t * ((-h).exp() - 1.0)) * model_output
            })
            .collect();
        Ok(CpuTensor {
            shape: sample.shape.clone(),
            data,
        })
    }

    pub(crate) fn dpm_second_order_update(
        &self,
        previous_timestep: usize,
        timestep: usize,
        prev_timestep: usize,
        sample: &CpuTensor,
        state: &SchedulerStepState,
    ) -> DiffusionResult<CpuTensor> {
        let SchedulerSolver::DpmSolverMultistep { solver_type, .. } = self.solver else {
            unreachable!("DPM second-order update called for non-DPM scheduler");
        };
        let m0 = state.model_outputs.last().ok_or_else(|| {
            DiffusionError::InvalidRequest("missing current DPM model output".to_string())
        })?;
        let m1 = state
            .model_outputs
            .get(state.model_outputs.len().saturating_sub(2))
            .ok_or_else(|| {
                DiffusionError::InvalidRequest("missing previous DPM model output".to_string())
            })?;
        let lambda_t = self.scheduler_lambda(prev_timestep)?;
        let lambda_s0 = self.scheduler_lambda(timestep)?;
        let lambda_s1 = self.scheduler_lambda(previous_timestep)?;
        let alpha_t = self.scheduler_alpha(prev_timestep)?;
        let sigma_t = self.scheduler_sigma(prev_timestep)?;
        let sigma_s0 = self.scheduler_sigma(timestep)?;
        let h = lambda_t - lambda_s0;
        let h0 = lambda_s0 - lambda_s1;
        if h.abs() <= f32::MIN_POSITIVE || h0.abs() <= f32::MIN_POSITIVE {
            return self.dpm_first_order_update(m0, timestep, prev_timestep, sample);
        }
        let r0 = h0 / h;
        let data = sample
            .data
            .iter()
            .zip(m0.iter().zip(m1))
            .map(|(sample, (m0, m1))| {
                let d1 = (m0 - m1) / r0;
                match solver_type {
                    DpmSolverType::Midpoint => {
                        (sigma_t / sigma_s0) * sample
                            - (alpha_t * ((-h).exp() - 1.0)) * m0
                            - 0.5 * (alpha_t * ((-h).exp() - 1.0)) * d1
                    }
                    DpmSolverType::Heun => {
                        (sigma_t / sigma_s0) * sample - (alpha_t * ((-h).exp() - 1.0)) * m0
                            + (alpha_t * (((-h).exp() - 1.0) / h + 1.0)) * d1
                    }
                }
            })
            .collect();
        Ok(CpuTensor {
            shape: sample.shape.clone(),
            data,
        })
    }

    pub(crate) fn dpm_third_order_update(
        &self,
        previous_previous_timestep: usize,
        previous_timestep: usize,
        timestep: usize,
        prev_timestep: usize,
        sample: &CpuTensor,
        state: &SchedulerStepState,
    ) -> DiffusionResult<CpuTensor> {
        let m0 = state.model_outputs.last().ok_or_else(|| {
            DiffusionError::InvalidRequest("missing current DPM model output".to_string())
        })?;
        let m1 = state
            .model_outputs
            .get(state.model_outputs.len().saturating_sub(2))
            .ok_or_else(|| {
                DiffusionError::InvalidRequest("missing previous DPM model output".to_string())
            })?;
        let m2 = state
            .model_outputs
            .get(state.model_outputs.len().saturating_sub(3))
            .ok_or_else(|| {
                DiffusionError::InvalidRequest(
                    "missing second previous DPM model output".to_string(),
                )
            })?;
        let lambda_t = self.scheduler_lambda(prev_timestep)?;
        let lambda_s0 = self.scheduler_lambda(timestep)?;
        let lambda_s1 = self.scheduler_lambda(previous_timestep)?;
        let lambda_s2 = self.scheduler_lambda(previous_previous_timestep)?;
        let alpha_t = self.scheduler_alpha(prev_timestep)?;
        let sigma_t = self.scheduler_sigma(prev_timestep)?;
        let sigma_s0 = self.scheduler_sigma(timestep)?;
        let h = lambda_t - lambda_s0;
        let h0 = lambda_s0 - lambda_s1;
        let h1 = lambda_s1 - lambda_s2;
        if h.abs() <= f32::MIN_POSITIVE
            || h0.abs() <= f32::MIN_POSITIVE
            || h1.abs() <= f32::MIN_POSITIVE
            || (h0 + h1).abs() <= f32::MIN_POSITIVE
        {
            return self.dpm_second_order_update(
                previous_timestep,
                timestep,
                prev_timestep,
                sample,
                state,
            );
        }
        let r0 = h0 / h;
        let r1 = h1 / h;
        if r0.abs() <= f32::MIN_POSITIVE
            || r1.abs() <= f32::MIN_POSITIVE
            || (r0 + r1).abs() <= f32::MIN_POSITIVE
        {
            return self.dpm_second_order_update(
                previous_timestep,
                timestep,
                prev_timestep,
                sample,
                state,
            );
        }
        let exp_neg_h = (-h).exp();
        let data = sample
            .data
            .iter()
            .zip(m0.iter().zip(m1.iter().zip(m2)))
            .map(|(sample, (m0, (m1, m2)))| {
                let d0 = *m0;
                let d1_0 = (m0 - m1) / r0;
                let d1_1 = (m1 - m2) / r1;
                let d1 = d1_0 + (r0 / (r0 + r1)) * (d1_0 - d1_1);
                let d2 = (d1_0 - d1_1) / (r0 + r1);
                (sigma_t / sigma_s0) * sample - (alpha_t * (exp_neg_h - 1.0)) * d0
                    + (alpha_t * ((exp_neg_h - 1.0) / h + 1.0)) * d1
                    - (alpha_t * ((exp_neg_h - 1.0 + h) / (h * h) - 0.5)) * d2
            })
            .collect();
        Ok(CpuTensor {
            shape: sample.shape.clone(),
            data,
        })
    }

    pub(crate) fn scheduler_alpha(&self, timestep: usize) -> DiffusionResult<f32> {
        self.alpha_t.get(timestep).copied().ok_or_else(|| {
            DiffusionError::InvalidRequest(format!(
                "missing scheduler alpha for timestep {timestep}"
            ))
        })
    }

    pub(crate) fn scheduler_sigma(&self, timestep: usize) -> DiffusionResult<f32> {
        self.sigma_t.get(timestep).copied().ok_or_else(|| {
            DiffusionError::InvalidRequest(format!(
                "missing scheduler sigma for timestep {timestep}"
            ))
        })
    }

    pub(crate) fn scheduler_lambda(&self, timestep: usize) -> DiffusionResult<f32> {
        self.lambda_t.get(timestep).copied().ok_or_else(|| {
            DiffusionError::InvalidRequest(format!(
                "missing scheduler lambda for timestep {timestep}"
            ))
        })
    }
}

pub(crate) fn scheduler_derivative(
    sample: f32,
    model_output: f32,
    sigma: f32,
    prediction_type: SchedulerPredictionType,
) -> f32 {
    if sigma.abs() <= f32::MIN_POSITIVE {
        return model_output;
    }
    match prediction_type {
        SchedulerPredictionType::Epsilon => model_output,
        SchedulerPredictionType::Sample => (sample - model_output) / sigma,
        SchedulerPredictionType::VPrediction => {
            let sigma_sq = sigma * sigma;
            let denom = sigma_sq + 1.0;
            let pred_original_sample = model_output * (-sigma / denom.sqrt()) + sample / denom;
            (sample - pred_original_sample) / sigma
        }
    }
}

fn scheduler_betas(
    beta_start: f32,
    beta_end: f32,
    num_train_timesteps: usize,
    schedule: &str,
) -> DiffusionResult<Vec<f32>> {
    if num_train_timesteps == 1 {
        return Ok(vec![beta_end.clamp(0.0, 0.999)]);
    }
    match schedule {
        "linear" => Ok((0..num_train_timesteps)
            .map(|idx| {
                let frac = idx as f32 / (num_train_timesteps - 1) as f32;
                beta_start + (beta_end - beta_start) * frac
            })
            .collect()),
        "scaled_linear" => {
            let start = beta_start.sqrt();
            let end = beta_end.sqrt();
            Ok((0..num_train_timesteps)
                .map(|idx| {
                    let frac = idx as f32 / (num_train_timesteps - 1) as f32;
                    let value = start + (end - start) * frac;
                    value * value
                })
                .collect())
        }
        "squaredcos_cap_v2" => Ok(betas_for_alpha_bar(num_train_timesteps)),
        other => Err(DiffusionError::InvalidMetadata(format!(
            "unsupported scheduler beta_schedule {other:?}"
        ))),
    }
}

fn betas_for_alpha_bar(num_train_timesteps: usize) -> Vec<f32> {
    pub(crate) fn alpha_bar(time: f32) -> f32 {
        let value = (time + 0.008) / 1.008 * std::f32::consts::FRAC_PI_2;
        value.cos().powi(2)
    }
    (0..num_train_timesteps)
        .map(|idx| {
            let t1 = idx as f32 / num_train_timesteps as f32;
            let t2 = (idx + 1) as f32 / num_train_timesteps as f32;
            (1.0 - alpha_bar(t2) / alpha_bar(t1)).min(0.999)
        })
        .collect()
}

fn karras_sigmas(base_sigmas: &[f32]) -> Vec<f32> {
    if base_sigmas.is_empty() {
        return vec![0.0];
    }
    let rho = 7.0f32;
    let sigma_max = base_sigmas
        .first()
        .copied()
        .unwrap_or(0.0)
        .max(f32::MIN_POSITIVE);
    let sigma_min = base_sigmas
        .last()
        .copied()
        .unwrap_or(sigma_max)
        .max(f32::MIN_POSITIVE);
    let min_inv_rho = sigma_min.powf(1.0 / rho);
    let max_inv_rho = sigma_max.powf(1.0 / rho);
    let denom = base_sigmas.len().saturating_sub(1).max(1) as f32;
    let mut sigmas = (0..base_sigmas.len())
        .map(|idx| {
            let ramp = idx as f32 / denom;
            (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)).powf(rho)
        })
        .collect::<Vec<_>>();
    sigmas.push(0.0);
    sigmas
}

fn rescale_sigmas_to_terminal(sigmas: &mut [f32], terminal: f32) {
    let Some(first) = sigmas.first().copied() else {
        return;
    };
    let Some(last) = sigmas.last().copied() else {
        return;
    };
    let denom = first - last;
    if denom.abs() <= f32::EPSILON {
        return;
    }
    for sigma in sigmas {
        let normalized = (*sigma - last) / denom;
        *sigma = terminal + normalized * (first - terminal);
    }
}

fn nearest_training_timestep_for_sigma(training_sigmas: &[f32], sigma: f32) -> usize {
    training_sigmas
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            let left_delta = (*left - sigma).abs();
            let right_delta = (*right - sigma).abs();
            left_delta
                .partial_cmp(&right_delta)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn inference_train_timesteps(
    config: &SchedulerConfig,
    num_train_timesteps: usize,
    steps: usize,
) -> DiffusionResult<Vec<usize>> {
    if steps == 1 {
        return Ok(vec![num_train_timesteps - 1]);
    }
    if config.class_name == "DPMSolverMultistepScheduler" {
        return dpm_solver_train_timesteps(config, num_train_timesteps, steps);
    }
    Ok((0..steps)
        .map(|idx| {
            let frac = idx as f32 / (steps - 1) as f32;
            ((num_train_timesteps - 1) as f32 * (1.0 - frac)).round() as usize
        })
        .collect())
}

fn dpm_solver_train_timesteps(
    config: &SchedulerConfig,
    num_train_timesteps: usize,
    steps: usize,
) -> DiffusionResult<Vec<usize>> {
    let last_timestep = num_train_timesteps;
    let spacing = config.timestep_spacing.as_deref().unwrap_or("linspace");
    let offset = config.steps_offset.unwrap_or(0);
    let mut timesteps = match spacing {
        "linspace" => (0..=steps)
            .map(|idx| {
                let frac = idx as f32 / steps as f32;
                ((last_timestep - 1) as f32 * frac).round() as i32
            })
            .rev()
            .take(steps)
            .collect::<Vec<_>>(),
        "leading" => {
            let step_ratio = last_timestep / (steps + 1);
            (0..=steps)
                .map(|idx| (idx * step_ratio) as i32 + offset)
                .rev()
                .take(steps)
                .collect()
        }
        "trailing" => {
            let step_ratio = num_train_timesteps as f32 / steps as f32;
            (0..steps)
                .map(|idx| (last_timestep as f32 - idx as f32 * step_ratio).round() as i32 - 1)
                .collect()
        }
        other => {
            return Err(DiffusionError::InvalidMetadata(format!(
                "unsupported scheduler timestep_spacing {other:?}"
            )));
        }
    };
    timesteps.dedup();
    let mut out = Vec::with_capacity(timesteps.len());
    for timestep in timesteps {
        if timestep < 0 || timestep as usize >= num_train_timesteps {
            return Err(DiffusionError::InvalidMetadata(format!(
                "DPM-Solver timestep {timestep} is outside 0..{num_train_timesteps}"
            )));
        }
        out.push(timestep as usize);
    }
    Ok(out)
}
