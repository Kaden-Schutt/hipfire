// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `DiffusionPipeline::preflight` — extracted from the pipeline
//! god-impl in lib.rs into its own `impl` block (3.8 Part 2). Uses `super::*`
//! so the pipeline's helpers + types resolve unchanged; the struct's fields are
//! pub(crate) so this block can read them.

use super::*;

impl DiffusionPipeline {
    pub fn preflight_hip_runtime(
        &self,
        request: &DiffusionBatchRequest,
        options: DiffusionHipRuntimeOptions,
    ) -> DiffusionResult<DiffusionHipPreflight> {
        validate_batch_request(&self.metadata, request)?;
        let memory_plan = self.hip_memory_plan(request)?;
        let mut gpu = hipfire_rdna::Gpu::init_with_device(options.device_id)
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        gpu.bind_thread()
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;

        for bytes in [
            memory_plan.latent_bytes,
            memory_plan.denoise_input_bytes,
            memory_plan.conditioning_bytes,
            memory_plan.vae_decode_bytes,
            memory_plan.rgb_bytes,
            memory_plan.scheduler_scratch_bytes,
        ] {
            if bytes == 0 {
                continue;
            }
            let buffer = gpu
                .hip
                .malloc(bytes)
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            gpu.hip
                .memset(&buffer, 0, bytes)
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
            gpu.hip
                .free(buffer)
                .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        }

        let probe = b"hipfire-diffusion-rocm-preflight";
        let probe_buffer = gpu
            .hip
            .malloc(probe.len())
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        gpu.hip
            .memcpy_htod(&probe_buffer, probe)
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        let mut roundtrip = vec![0u8; probe.len()];
        gpu.hip
            .memcpy_dtoh(&mut roundtrip, &probe_buffer)
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        gpu.hip
            .free(probe_buffer)
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        gpu.hip
            .device_synchronize()
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        if roundtrip != probe {
            return Err(DiffusionError::BackendUnavailable(
                "HIP preflight probe roundtrip mismatch".to_string(),
            ));
        }
        let kernel_probe_tensor = CpuTensor {
            shape: vec![1, 3, 2, 2],
            data: vec![
                -1.0, 0.0, 1.0, 0.25, -0.5, 0.5, -0.25, 0.75, 1.0, -1.0, 0.1, -0.1,
            ],
        };
        let cpu_reference = rgb_tensor_to_u8(&kernel_probe_tensor)?;
        let gpu_output = rgb_tensor_to_u8_hip_on_gpu(&mut gpu, &kernel_probe_tensor)?;
        let kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_rgb_tensor_to_u8".to_string(),
            input_elements: kernel_probe_tensor.data.len(),
            output_bytes: gpu_output.data.len(),
            matched_cpu_reference: gpu_output == cpu_reference,
        };
        if !kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion RGB kernel output differed from CPU reference".to_string(),
            ));
        }
        let vae_image_probe = RgbImageBatch {
            batch: 2,
            width: 2,
            height: 2,
            data: vec![
                0, 128, 255, 255, 0, 128, 32, 64, 96, 192, 224, 16, 10, 20, 30, 40, 50, 60, 70, 80,
                90, 100, 110, 120,
            ],
        };
        let rgb_to_vae_tensor_cpu_reference = rgb_batch_to_vae_tensor(&vae_image_probe)?;
        let rgb_to_vae_tensor_gpu_output =
            rgb_batch_to_vae_tensor_hip_on_gpu(&mut gpu, &vae_image_probe)?;
        let rgb_to_vae_tensor_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_rgb_u8_to_vae_nchw_f32".to_string(),
            input_elements: vae_image_probe.data.len(),
            output_bytes: rgb_to_vae_tensor_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: rgb_to_vae_tensor_gpu_output.shape
                == rgb_to_vae_tensor_cpu_reference.shape
                && f32_slices_close(
                    &rgb_to_vae_tensor_gpu_output.data,
                    &rgb_to_vae_tensor_cpu_reference.data,
                    1e-6,
                ),
        };
        if !rgb_to_vae_tensor_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion RGB-to-VAE kernel output differed from CPU reference".to_string(),
            ));
        }
        let inpaint_mask_probe = RgbImageBatch {
            batch: 2,
            width: 4,
            height: 4,
            data: (0..96)
                .map(|idx| ((idx * 37 + 11) % 256) as u8)
                .collect::<Vec<_>>(),
        };
        let inpaint_image_probe = RgbImageBatch {
            batch: 2,
            width: 4,
            height: 4,
            data: (0..96)
                .map(|idx| ((idx * 19 + 5) % 256) as u8)
                .collect::<Vec<_>>(),
        };
        let inpaint_latent_probe = LatentBatch {
            batch: 2,
            channels: 2,
            height: 2,
            width: 2,
            data: (0..16)
                .map(|idx| idx as f32 / 7.0 - 1.0)
                .collect::<Vec<_>>(),
        };
        let latent_mask_weights_cpu_reference =
            latent_mask_weights_from_rgb_batch(&inpaint_mask_probe, &inpaint_latent_probe)?;
        let latent_mask_weights_gpu_output = latent_mask_weights_from_rgb_batch_hip_on_gpu(
            &mut gpu,
            &inpaint_mask_probe,
            &inpaint_latent_probe,
        )?;
        let latent_mask_weights_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_latent_mask_weights_from_rgb_f32".to_string(),
            input_elements: inpaint_mask_probe.data.len(),
            output_bytes: latent_mask_weights_gpu_output.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: f32_slices_close(
                &latent_mask_weights_gpu_output,
                &latent_mask_weights_cpu_reference,
                1e-6,
            ),
        };
        if !latent_mask_weights_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion latent-mask kernel output differed from CPU reference".to_string(),
            ));
        }
        let masked_rgb_inpaint_cpu_reference =
            masked_rgb_batch_for_inpaint(&inpaint_image_probe, &inpaint_mask_probe)?;
        let masked_rgb_inpaint_gpu_output = masked_rgb_batch_for_inpaint_hip_on_gpu(
            &mut gpu,
            &inpaint_image_probe,
            &inpaint_mask_probe,
        )?;
        let masked_rgb_inpaint_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_masked_rgb_for_inpaint_u8".to_string(),
            input_elements: inpaint_image_probe.data.len() + inpaint_mask_probe.data.len(),
            output_bytes: masked_rgb_inpaint_gpu_output.data.len(),
            matched_cpu_reference: masked_rgb_inpaint_gpu_output
                == masked_rgb_inpaint_cpu_reference,
        };
        if !masked_rgb_inpaint_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion masked-RGB kernel output differed from CPU reference".to_string(),
            ));
        }
        let generated_latents_probe = LatentBatch {
            batch: 2,
            channels: 2,
            height: 2,
            width: 2,
            data: (0..16)
                .map(|idx| (idx as f32 % 9.0 - 4.0) / 3.0)
                .collect::<Vec<_>>(),
        };
        let mut blend_latents_cpu_reference = generated_latents_probe.clone();
        blend_latents_with_mask(
            &mut blend_latents_cpu_reference,
            &inpaint_latent_probe,
            &latent_mask_weights_cpu_reference,
        )?;
        let blend_latents_gpu_output = blend_latents_with_mask_hip_on_gpu(
            &mut gpu,
            &generated_latents_probe,
            &inpaint_latent_probe,
            &latent_mask_weights_cpu_reference,
        )?;
        let blend_latents_with_mask_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_blend_latents_with_mask_f32".to_string(),
            input_elements: generated_latents_probe.data.len()
                + inpaint_latent_probe.data.len()
                + latent_mask_weights_cpu_reference.len(),
            output_bytes: blend_latents_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: blend_latents_gpu_output.batch
                == blend_latents_cpu_reference.batch
                && blend_latents_gpu_output.channels == blend_latents_cpu_reference.channels
                && blend_latents_gpu_output.height == blend_latents_cpu_reference.height
                && blend_latents_gpu_output.width == blend_latents_cpu_reference.width
                && f32_slices_close(
                    &blend_latents_gpu_output.data,
                    &blend_latents_cpu_reference.data,
                    1e-6,
                ),
        };
        if !blend_latents_with_mask_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion latent-blend kernel output differed from CPU reference".to_string(),
            ));
        }
        let model_input_sample = vec![-1.0, -0.25, 0.0, 0.5, 1.0, 2.0, -2.0, 0.125];
        let model_input_scale = 0.5;
        let model_input_cpu_reference = model_input_sample
            .iter()
            .map(|sample| sample * model_input_scale)
            .collect::<Vec<_>>();
        let model_input_gpu_output =
            scale_model_input_hip_on_gpu(&mut gpu, &model_input_sample, model_input_scale)?;
        let model_input_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_scale_model_input_f32".to_string(),
            input_elements: model_input_sample.len(),
            output_bytes: model_input_gpu_output.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: f32_slices_close(
                &model_input_gpu_output,
                &model_input_cpu_reference,
                1e-6,
            ),
        };
        if !model_input_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion model-input scaling kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let guidance_negative = vec![-1.0, -0.25, 0.0, 0.5, 1.0, 2.0, -2.0, 0.125];
        let guidance_positive = vec![0.5, -0.5, 0.25, -0.25, 1.5, -1.0, 0.75, -0.125];
        let guidance_scale = 7.5;
        let guidance_cpu_reference = guidance_negative
            .iter()
            .zip(&guidance_positive)
            .map(|(negative, positive)| negative + guidance_scale * (positive - negative))
            .collect::<Vec<_>>();
        let guidance_gpu_output = cfg_guidance_hip_on_gpu(
            &mut gpu,
            &guidance_negative,
            &guidance_positive,
            guidance_scale,
        )?;
        let guidance_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_cfg_guidance_f32".to_string(),
            input_elements: guidance_negative.len(),
            output_bytes: guidance_gpu_output.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: f32_slices_close(
                &guidance_gpu_output,
                &guidance_cpu_reference,
                1e-6,
            ),
        };
        if !guidance_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion CFG guidance kernel output differed from CPU reference".to_string(),
            ));
        }
        let scheduler_sample = vec![-1.0, -0.25, 0.0, 0.5, 1.0, 2.0, -2.0, 0.125];
        let scheduler_model_output = vec![0.5, -0.5, 0.25, -0.25, 1.5, -1.0, 0.75, -0.125];
        let scheduler_sigma = 1.0;
        let scheduler_next_sigma = 0.5;
        let scheduler_cpu_reference = scheduler_sample
            .iter()
            .zip(&scheduler_model_output)
            .map(|(sample, model_output)| {
                sample
                    + scheduler_derivative(
                        *sample,
                        *model_output,
                        scheduler_sigma,
                        SchedulerPredictionType::Epsilon,
                    ) * (scheduler_next_sigma - scheduler_sigma)
            })
            .collect::<Vec<_>>();
        let scheduler_gpu_output = euler_step_hip_on_gpu(
            &mut gpu,
            &scheduler_sample,
            &scheduler_model_output,
            scheduler_sigma,
            scheduler_next_sigma,
            SchedulerPredictionType::Epsilon,
        )?;
        let scheduler_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_euler_step_f32".to_string(),
            input_elements: scheduler_sample.len(),
            output_bytes: scheduler_gpu_output.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: scheduler_gpu_output == scheduler_cpu_reference,
        };
        if !scheduler_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion scheduler kernel output differed from CPU reference".to_string(),
            ));
        }
        let center_input = CpuTensor {
            shape: vec![1, 2, 2, 2],
            data: vec![-1.0, -0.25, 0.0, 0.5, 1.0, 2.0, -2.0, 0.125],
        };
        let center_unet_input_cpu_reference = maybe_center_unet_input(&center_input, true);
        let center_unet_input_gpu_output =
            maybe_center_unet_input_hip_on_gpu(&mut gpu, &center_input, true)?;
        let center_unet_input_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_center_unet_input_f32".to_string(),
            input_elements: center_input.data.len(),
            output_bytes: center_unet_input_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: center_unet_input_gpu_output.shape
                == center_unet_input_cpu_reference.shape
                && center_unet_input_gpu_output.data == center_unet_input_cpu_reference.data,
        };
        if !center_unet_input_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion centered UNet input kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let timestep_values = vec![999.0, 500.5, 0.25];
        let timestep_embedding_dim = 7;
        let timestep_embedding_cpu_reference =
            timestep_embedding(&timestep_values, timestep_embedding_dim, true, 1.0)?;
        let timestep_embedding_gpu_output = timestep_embedding_hip_on_gpu(
            &mut gpu,
            &timestep_values,
            timestep_embedding_dim,
            true,
            1.0,
        )?;
        let timestep_embedding_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_timestep_embedding_f32".to_string(),
            input_elements: timestep_values.len(),
            output_bytes: timestep_embedding_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: timestep_embedding_gpu_output.shape
                == timestep_embedding_cpu_reference.shape
                && f32_slices_close(
                    &timestep_embedding_gpu_output.data,
                    &timestep_embedding_cpu_reference.data,
                    1e-5,
                ),
        };
        if !timestep_embedding_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion timestep embedding kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let tensor_add_left = CpuTensor {
            shape: vec![1, 2, 2, 3],
            data: (0..12)
                .map(|idx| idx as f32 / 8.0 - 0.75)
                .collect::<Vec<_>>(),
        };
        let tensor_add_right = CpuTensor {
            shape: vec![1, 2, 2, 3],
            data: (0..12)
                .map(|idx| (idx as f32 % 5.0 - 2.0) / 3.0)
                .collect::<Vec<_>>(),
        };
        let tensor_add_cpu_reference = tensor_add(&tensor_add_left, &tensor_add_right)?;
        let tensor_add_gpu_output =
            tensor_add_hip_on_gpu(&mut gpu, &tensor_add_left, &tensor_add_right)?;
        let tensor_add_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_tensor_add_f32".to_string(),
            input_elements: tensor_add_left.data.len() + tensor_add_right.data.len(),
            output_bytes: tensor_add_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: tensor_add_gpu_output.shape == tensor_add_cpu_reference.shape
                && f32_slices_close(
                    &tensor_add_gpu_output.data,
                    &tensor_add_cpu_reference.data,
                    1e-6,
                ),
        };
        if !tensor_add_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion tensor-add kernel output differed from CPU reference".to_string(),
            ));
        }
        let channel_bias_input = CpuTensor {
            shape: vec![2, 3, 2, 2],
            data: (0..24)
                .map(|idx| idx as f32 / 10.0 - 1.0)
                .collect::<Vec<_>>(),
        };
        let channel_bias = CpuTensor {
            shape: vec![2, 3],
            data: vec![0.25, -0.5, 0.75, -1.0, 0.5, -0.25],
        };
        let mut add_channel_bias_cpu_reference = channel_bias_input.clone();
        add_channel_bias_nchw(&mut add_channel_bias_cpu_reference, &channel_bias)?;
        let add_channel_bias_gpu_output =
            add_channel_bias_nchw_hip_on_gpu(&mut gpu, &channel_bias_input, &channel_bias)?;
        let add_channel_bias_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_add_channel_bias_nchw_f32".to_string(),
            input_elements: channel_bias_input.data.len() + channel_bias.data.len(),
            output_bytes: add_channel_bias_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: add_channel_bias_gpu_output.shape
                == add_channel_bias_cpu_reference.shape
                && f32_slices_close(
                    &add_channel_bias_gpu_output.data,
                    &add_channel_bias_cpu_reference.data,
                    1e-6,
                ),
        };
        if !add_channel_bias_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion channel-bias kernel output differed from CPU reference".to_string(),
            ));
        }
        let layout_input = CpuTensor {
            shape: vec![2, 3, 2, 4],
            data: (0..48)
                .map(|idx| idx as f32 / 11.0 - 1.75)
                .collect::<Vec<_>>(),
        };
        let nchw_to_bsc_cpu_reference = nchw_to_bsc(&layout_input)?;
        let nchw_to_bsc_gpu_output = nchw_to_bsc_hip_on_gpu(&mut gpu, &layout_input)?;
        let nchw_to_bsc_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_nchw_to_bsc_f32".to_string(),
            input_elements: layout_input.data.len(),
            output_bytes: nchw_to_bsc_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: nchw_to_bsc_gpu_output.shape == nchw_to_bsc_cpu_reference.shape
                && nchw_to_bsc_gpu_output.data == nchw_to_bsc_cpu_reference.data,
        };
        if !nchw_to_bsc_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion NCHW-to-BSC kernel output differed from CPU reference".to_string(),
            ));
        }
        let bsc_to_nchw_cpu_reference = bsc_to_nchw(&nchw_to_bsc_cpu_reference, 2, 3, 2, 4)?;
        let bsc_to_nchw_gpu_output =
            bsc_to_nchw_hip_on_gpu(&mut gpu, &nchw_to_bsc_cpu_reference, 2, 3, 2, 4)?;
        let bsc_to_nchw_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_bsc_to_nchw_f32".to_string(),
            input_elements: nchw_to_bsc_cpu_reference.data.len(),
            output_bytes: bsc_to_nchw_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: bsc_to_nchw_gpu_output.shape == bsc_to_nchw_cpu_reference.shape
                && bsc_to_nchw_gpu_output.data == bsc_to_nchw_cpu_reference.data,
        };
        if !bsc_to_nchw_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion BSC-to-NCHW kernel output differed from CPU reference".to_string(),
            ));
        }
        let concat_channels_left = CpuTensor {
            shape: vec![1, 2, 2, 3],
            data: (0..12)
                .map(|idx| idx as f32 / 7.0 - 0.75)
                .collect::<Vec<_>>(),
        };
        let concat_channels_right = CpuTensor {
            shape: vec![1, 3, 2, 3],
            data: (0..18)
                .map(|idx| (idx as f32 % 11.0 - 5.0) / 6.0)
                .collect::<Vec<_>>(),
        };
        let concat_channels_cpu_reference =
            concat_channels_nchw(&concat_channels_left, &concat_channels_right)?;
        let concat_channels_gpu_output = concat_channels_nchw_hip_on_gpu(
            &mut gpu,
            &concat_channels_left,
            &concat_channels_right,
        )?;
        let concat_channels_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_concat_channels_nchw_f32".to_string(),
            input_elements: concat_channels_left.data.len() + concat_channels_right.data.len(),
            output_bytes: concat_channels_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: concat_channels_gpu_output.shape
                == concat_channels_cpu_reference.shape
                && concat_channels_gpu_output.data == concat_channels_cpu_reference.data,
        };
        if !concat_channels_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion channel-concat kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let concat_2d_left = CpuTensor {
            shape: vec![3, 2],
            data: vec![0.25, -0.5, 0.75, 1.0, -1.25, 0.5],
        };
        let concat_2d_right = CpuTensor {
            shape: vec![3, 3],
            data: vec![-1.0, 0.0, 1.0, 0.5, -0.25, 0.75, 1.25, -0.75, 0.25],
        };
        let concat_last_dim_2d_cpu_reference =
            concat_last_dim_2d(&concat_2d_left, &concat_2d_right)?;
        let concat_last_dim_2d_gpu_output =
            concat_last_dim_2d_hip_on_gpu(&mut gpu, &concat_2d_left, &concat_2d_right)?;
        let concat_last_dim_2d_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_concat_last_dim_2d_f32".to_string(),
            input_elements: concat_2d_left.data.len() + concat_2d_right.data.len(),
            output_bytes: concat_last_dim_2d_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: concat_last_dim_2d_gpu_output.shape
                == concat_last_dim_2d_cpu_reference.shape
                && concat_last_dim_2d_gpu_output.data == concat_last_dim_2d_cpu_reference.data,
        };
        if !concat_last_dim_2d_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion 2D last-dim concat kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let concat_3d_left = CpuTensor {
            shape: vec![2, 2, 2],
            data: (0..8).map(|idx| idx as f32 / 4.0 - 0.5).collect::<Vec<_>>(),
        };
        let concat_3d_right = CpuTensor {
            shape: vec![2, 2, 3],
            data: (0..12)
                .map(|idx| (idx as f32 % 7.0 - 3.0) / 5.0)
                .collect::<Vec<_>>(),
        };
        let concat_last_dim_3d_cpu_reference =
            concat_last_dim_3d(&concat_3d_left, &concat_3d_right)?;
        let concat_last_dim_3d_gpu_output =
            concat_last_dim_3d_hip_on_gpu(&mut gpu, &concat_3d_left, &concat_3d_right)?;
        let concat_last_dim_3d_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_concat_last_dim_3d_f32".to_string(),
            input_elements: concat_3d_left.data.len() + concat_3d_right.data.len(),
            output_bytes: concat_last_dim_3d_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: concat_last_dim_3d_gpu_output.shape
                == concat_last_dim_3d_cpu_reference.shape
                && concat_last_dim_3d_gpu_output.data == concat_last_dim_3d_cpu_reference.data,
        };
        if !concat_last_dim_3d_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion 3D last-dim concat kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let conv2d_input = CpuTensor {
            shape: vec![1, 2, 3, 4],
            data: (0..24)
                .map(|idx| idx as f32 / 8.0 - 1.5)
                .collect::<Vec<_>>(),
        };
        let conv2d_weight = CpuTensor {
            shape: vec![3, 2, 3, 2],
            data: (0..36)
                .map(|idx| (idx as f32 % 7.0 - 3.0) / 5.0)
                .collect::<Vec<_>>(),
        };
        let conv2d_bias = CpuTensor {
            shape: vec![3],
            data: vec![0.25, -0.5, 0.75],
        };
        let conv2d_cpu_reference =
            conv2d_nchw_with_stride(&conv2d_input, &conv2d_weight, Some(&conv2d_bias), 1, 2)?;
        let conv2d_gpu_output = conv2d_nchw_hip_on_gpu(
            &mut gpu,
            &mut RocmWeightCache::default(),
            &conv2d_input,
            &conv2d_weight,
            Some(&conv2d_bias),
            1,
            2,
        )?;
        let conv2d_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_conv2d_nchw_f32".to_string(),
            input_elements: conv2d_input.data.len() + conv2d_weight.data.len(),
            output_bytes: conv2d_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: conv2d_gpu_output.shape == conv2d_cpu_reference.shape
                && f32_slices_close(&conv2d_gpu_output.data, &conv2d_cpu_reference.data, 1e-5),
        };
        if !conv2d_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion Conv2D kernel output differed from CPU reference".to_string(),
            ));
        }
        let group_norm_input = CpuTensor {
            shape: vec![1, 4, 2, 3],
            data: (0..24)
                .map(|idx| idx as f32 / 7.0 - 1.5)
                .collect::<Vec<_>>(),
        };
        let group_norm_weight = CpuTensor {
            shape: vec![4],
            data: vec![1.0, 0.5, -1.0, 1.5],
        };
        let group_norm_bias = CpuTensor {
            shape: vec![4],
            data: vec![0.0, 0.25, -0.5, 0.75],
        };
        let group_norm_cpu_reference = group_norm_nchw(
            &group_norm_input,
            &group_norm_weight,
            &group_norm_bias,
            2,
            1e-5,
        )?;
        let group_norm_gpu_output = group_norm_nchw_hip_on_gpu(
            &mut gpu,
            &group_norm_input,
            &group_norm_weight,
            &group_norm_bias,
            2,
            1e-5,
        )?;
        let group_norm_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_group_norm_nchw_f32".to_string(),
            input_elements: group_norm_input.data.len()
                + group_norm_weight.data.len()
                + group_norm_bias.data.len(),
            output_bytes: group_norm_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: group_norm_gpu_output.shape == group_norm_cpu_reference.shape
                && f32_slices_close(
                    &group_norm_gpu_output.data,
                    &group_norm_cpu_reference.data,
                    1e-5,
                ),
        };
        if !group_norm_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion GroupNorm kernel output differed from CPU reference".to_string(),
            ));
        }
        let silu_input = CpuTensor {
            shape: vec![1, 2, 2, 2],
            data: vec![-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 4.0],
        };
        let silu_cpu_reference = tensor_map(&silu_input, silu);
        let silu_gpu_output = silu_hip_on_gpu(&mut gpu, &silu_input)?;
        let silu_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_silu_f32".to_string(),
            input_elements: silu_input.data.len(),
            output_bytes: silu_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: silu_gpu_output.shape == silu_cpu_reference.shape
                && f32_slices_close(&silu_gpu_output.data, &silu_cpu_reference.data, 1e-6),
        };
        if !silu_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion SiLU kernel output differed from CPU reference".to_string(),
            ));
        }
        let quick_gelu_input = CpuTensor {
            shape: vec![2, 4],
            data: vec![-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 4.0],
        };
        let quick_gelu_cpu_reference = tensor_map(&quick_gelu_input, quick_gelu);
        let quick_gelu_gpu_output = quick_gelu_hip_on_gpu(&mut gpu, &quick_gelu_input)?;
        let quick_gelu_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_quick_gelu_f32".to_string(),
            input_elements: quick_gelu_input.data.len(),
            output_bytes: quick_gelu_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: quick_gelu_gpu_output.shape == quick_gelu_cpu_reference.shape
                && f32_slices_close(
                    &quick_gelu_gpu_output.data,
                    &quick_gelu_cpu_reference.data,
                    1e-6,
                ),
        };
        if !quick_gelu_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion QuickGELU kernel output differed from CPU reference".to_string(),
            ));
        }
        let clip_token_embedding_probe = CpuTensor {
            shape: vec![4, 3],
            data: (0..12)
                .map(|idx| idx as f32 / 10.0 - 0.4)
                .collect::<Vec<_>>(),
        };
        let clip_position_embedding_probe = CpuTensor {
            shape: vec![3, 3],
            data: (0..9)
                .map(|idx| (idx as f32 % 5.0 - 2.0) / 7.0)
                .collect::<Vec<_>>(),
        };
        let clip_token_probe = vec![0, 3, 1];
        let clip_token_position_embedding_cpu_reference = clip_token_position_embeddings(
            &clip_token_embedding_probe,
            &clip_position_embedding_probe,
            &clip_token_probe,
        )?;
        let clip_token_position_embedding_gpu_output = clip_token_position_embeddings_hip_on_gpu(
            &mut gpu,
            &clip_token_embedding_probe,
            &clip_position_embedding_probe,
            &clip_token_probe,
        )?;
        let clip_token_position_embedding_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_clip_token_position_embedding_f32".to_string(),
            input_elements: clip_token_embedding_probe.data.len()
                + clip_position_embedding_probe.data.len()
                + clip_token_probe.len(),
            output_bytes: clip_token_position_embedding_gpu_output.data.len()
                * std::mem::size_of::<f32>(),
            matched_cpu_reference: clip_token_position_embedding_gpu_output.shape
                == clip_token_position_embedding_cpu_reference.shape
                && f32_slices_close(
                    &clip_token_position_embedding_gpu_output.data,
                    &clip_token_position_embedding_cpu_reference.data,
                    1e-6,
                ),
        };
        if !clip_token_position_embedding_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion CLIP token-position embedding kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let upsample_input = CpuTensor {
            shape: vec![1, 2, 2, 3],
            data: (0..12)
                .map(|idx| idx as f32 / 5.0 - 1.0)
                .collect::<Vec<_>>(),
        };
        let upsample_cpu_reference = upsample_nearest2d_nchw(&upsample_input, 2)?;
        let upsample_gpu_output = upsample_nearest2d_nchw_hip_on_gpu(&mut gpu, &upsample_input, 2)?;
        let upsample_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_upsample_nearest2d_nchw_f32".to_string(),
            input_elements: upsample_input.data.len(),
            output_bytes: upsample_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: upsample_gpu_output.shape == upsample_cpu_reference.shape
                && upsample_gpu_output.data == upsample_cpu_reference.data,
        };
        if !upsample_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion nearest-upsample kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let linear_input = CpuTensor {
            shape: vec![3, 4],
            data: (0..12)
                .map(|idx| idx as f32 / 6.0 - 1.0)
                .collect::<Vec<_>>(),
        };
        let linear_weight = CpuTensor {
            shape: vec![5, 4],
            data: (0..20)
                .map(|idx| (idx as f32 % 11.0 - 5.0) / 7.0)
                .collect::<Vec<_>>(),
        };
        let linear_bias = CpuTensor {
            shape: vec![5],
            data: vec![0.0, 0.25, -0.5, 0.75, -1.0],
        };
        let linear_cpu_reference = linear(&linear_input, &linear_weight, &linear_bias)?;
        let linear_gpu_output = linear_optional_bias_hip_on_gpu(
            &mut gpu,
            &mut RocmWeightCache::default(),
            &linear_input,
            &linear_weight,
            Some(&linear_bias),
        )?;
        let linear_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_linear_f32".to_string(),
            input_elements: linear_input.data.len()
                + linear_weight.data.len()
                + linear_bias.data.len(),
            output_bytes: linear_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: linear_gpu_output.shape == linear_cpu_reference.shape
                && f32_slices_close(&linear_gpu_output.data, &linear_cpu_reference.data, 1e-5),
        };
        if !linear_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion linear kernel output differed from CPU reference".to_string(),
            ));
        }
        let layer_norm_input = CpuTensor {
            shape: vec![3, 4],
            data: (0..12)
                .map(|idx| idx as f32 / 5.0 - 1.25)
                .collect::<Vec<_>>(),
        };
        let layer_norm_weight = CpuTensor {
            shape: vec![4],
            data: vec![1.0, 0.5, -1.0, 1.5],
        };
        let layer_norm_bias = CpuTensor {
            shape: vec![4],
            data: vec![0.0, 0.25, -0.5, 0.75],
        };
        let layer_norm_cpu_reference = layer_norm(
            &layer_norm_input,
            &layer_norm_weight,
            &layer_norm_bias,
            1e-5,
        )?;
        let layer_norm_gpu_output = layer_norm_hip_on_gpu(
            &mut gpu,
            &layer_norm_input,
            &layer_norm_weight,
            &layer_norm_bias,
            1e-5,
        )?;
        let layer_norm_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_layer_norm_f32".to_string(),
            input_elements: layer_norm_input.data.len()
                + layer_norm_weight.data.len()
                + layer_norm_bias.data.len(),
            output_bytes: layer_norm_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: layer_norm_gpu_output.shape == layer_norm_cpu_reference.shape
                && f32_slices_close(
                    &layer_norm_gpu_output.data,
                    &layer_norm_cpu_reference.data,
                    1e-5,
                ),
        };
        if !layer_norm_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion LayerNorm kernel output differed from CPU reference".to_string(),
            ));
        }
        let softmax_input = CpuTensor {
            shape: vec![3, 4],
            data: vec![
                1.0, 2.0, 3.0, 4.0, -2.0, -0.5, 0.25, 1.5, 10.0, 9.0, 8.0, 7.0,
            ],
        };
        let mut softmax_cpu_reference = softmax_input.clone();
        for row in softmax_cpu_reference.data.chunks_mut(4) {
            softmax_in_place(row);
        }
        let softmax_gpu_output = softmax_rows_hip_on_gpu(&mut gpu, &softmax_input)?;
        let softmax_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_softmax_rows_f32".to_string(),
            input_elements: softmax_input.data.len(),
            output_bytes: softmax_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: softmax_gpu_output.shape == softmax_cpu_reference.shape
                && f32_slices_close(&softmax_gpu_output.data, &softmax_cpu_reference.data, 1e-6),
        };
        if !softmax_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion softmax kernel output differed from CPU reference".to_string(),
            ));
        }
        let sdpa_q = CpuTensor {
            shape: vec![1, 2, 4],
            data: vec![0.2, -0.4, 0.6, 1.0, -1.2, 0.3, 0.5, -0.7],
        };
        let sdpa_k = CpuTensor {
            shape: vec![1, 3, 4],
            data: vec![
                -0.5, 0.25, 0.75, -1.0, 1.25, -0.75, 0.5, 0.0, 0.1, 0.9, -0.3, 0.4,
            ],
        };
        let sdpa_v = CpuTensor {
            shape: vec![1, 3, 4],
            data: vec![
                0.5, -1.0, 0.25, 0.75, -0.4, 0.6, -0.8, 1.2, 1.0, 0.2, -0.5, -0.1,
            ],
        };
        let sdpa_cpu_reference = scaled_dot_product_attention(&sdpa_q, &sdpa_k, &sdpa_v, 2)?;
        let sdpa_gpu_output =
            scaled_dot_product_attention_hip_on_gpu(&mut gpu, &sdpa_q, &sdpa_k, &sdpa_v, 2)?;
        let sdpa_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_sdpa_3d_f32".to_string(),
            input_elements: sdpa_q.data.len() + sdpa_k.data.len() + sdpa_v.data.len(),
            output_bytes: sdpa_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: sdpa_gpu_output.shape == sdpa_cpu_reference.shape
                && f32_slices_close(&sdpa_gpu_output.data, &sdpa_cpu_reference.data, 1e-5),
        };
        if !sdpa_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion SDPA kernel output differed from CPU reference".to_string(),
            ));
        }
        let clip_attn_q = CpuTensor {
            shape: vec![4, 4],
            data: vec![
                0.2, -0.4, 0.6, 1.0, -1.2, 0.3, 0.5, -0.7, 1.0, 0.0, -0.5, 0.25, -0.25, 0.75, -1.0,
                0.5,
            ],
        };
        let clip_attn_k = CpuTensor {
            shape: vec![4, 4],
            data: vec![
                -0.5, 0.25, 0.75, -1.0, 1.25, -0.75, 0.5, 0.0, 0.1, 0.9, -0.3, 0.4, 0.7, -0.2, 0.3,
                -0.8,
            ],
        };
        let clip_attn_v = CpuTensor {
            shape: vec![4, 4],
            data: vec![
                0.5, -1.0, 0.25, 0.75, -0.4, 0.6, -0.8, 1.2, 1.0, 0.2, -0.5, -0.1, -0.9, 0.3, 0.8,
                -0.2,
            ],
        };
        let clip_causal_attention_cpu_reference =
            clip_causal_self_attention(&clip_attn_q, &clip_attn_k, &clip_attn_v, 2)?;
        let clip_causal_attention_gpu_output = clip_causal_self_attention_hip_on_gpu(
            &mut gpu,
            &clip_attn_q,
            &clip_attn_k,
            &clip_attn_v,
            2,
        )?;
        let clip_causal_attention_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_clip_causal_attention_f32".to_string(),
            input_elements: clip_attn_q.data.len()
                + clip_attn_k.data.len()
                + clip_attn_v.data.len(),
            output_bytes: clip_causal_attention_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: clip_causal_attention_gpu_output.shape
                == clip_causal_attention_cpu_reference.shape
                && f32_slices_close(
                    &clip_causal_attention_gpu_output.data,
                    &clip_causal_attention_cpu_reference.data,
                    1e-5,
                ),
        };
        if !clip_causal_attention_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion CLIP causal attention kernel output differed from CPU reference"
                    .to_string(),
            ));
        }
        let geglu_projected = CpuTensor {
            shape: vec![1, 3, 6],
            data: vec![
                0.5, -0.25, 1.0, -1.5, 0.2, 0.75, -0.4, 0.9, -1.1, 0.6, -0.8, 1.25, 1.5, -1.0, 0.3,
                0.0, 1.1, -0.6,
            ],
        };
        let geglu_gate_cpu_reference = geglu_gate_3d(&geglu_projected)?;
        let geglu_gate_gpu_output = geglu_gate_3d_hip_on_gpu(&mut gpu, &geglu_projected)?;
        let geglu_gate_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_geglu_gate_3d_f32".to_string(),
            input_elements: geglu_projected.data.len(),
            output_bytes: geglu_gate_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: geglu_gate_gpu_output.shape == geglu_gate_cpu_reference.shape
                && f32_slices_close(
                    &geglu_gate_gpu_output.data,
                    &geglu_gate_cpu_reference.data,
                    1e-5,
                ),
        };
        if !geglu_gate_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion GeGLU gate kernel output differed from CPU reference".to_string(),
            ));
        }
        let vae_moments = CpuTensor {
            shape: vec![2, 4, 2, 2],
            data: (0..32)
                .map(|idx| idx as f32 / 9.0 - 1.5)
                .collect::<Vec<_>>(),
        };
        let vae_moments_to_latents_cpu_reference =
            vae_moments_to_latents(&vae_moments, &VaeLatentNorm::scalar(0.18215))?;
        let vae_moments_to_latents_gpu_output =
            vae_moments_to_latents_hip_on_gpu(&mut gpu, &vae_moments, 0.18215)?;
        let vae_moments_to_latents_kernel_probe = DiffusionHipKernelProbe {
            name: "diffusion_vae_moments_to_latents_f32".to_string(),
            input_elements: vae_moments.data.len(),
            output_bytes: vae_moments_to_latents_gpu_output.data.len() * std::mem::size_of::<f32>(),
            matched_cpu_reference: vae_moments_to_latents_gpu_output.batch
                == vae_moments_to_latents_cpu_reference.batch
                && vae_moments_to_latents_gpu_output.channels
                    == vae_moments_to_latents_cpu_reference.channels
                && vae_moments_to_latents_gpu_output.height
                    == vae_moments_to_latents_cpu_reference.height
                && vae_moments_to_latents_gpu_output.width
                    == vae_moments_to_latents_cpu_reference.width
                && f32_slices_close(
                    &vae_moments_to_latents_gpu_output.data,
                    &vae_moments_to_latents_cpu_reference.data,
                    1e-6,
                ),
        };
        if !vae_moments_to_latents_kernel_probe.matched_cpu_reference {
            return Err(DiffusionError::BackendUnavailable(
                "HIP diffusion VAE moments-to-latents kernel output differed from CPU reference"
                    .to_string(),
            ));
        }

        Ok(DiffusionHipPreflight {
            device_id: gpu.device_id,
            arch: gpu.arch.clone(),
            integrated: gpu.integrated,
            memory_plan,
            probe_bytes: probe.len(),
            kernel_probe,
            rgb_to_vae_tensor_kernel_probe,
            latent_mask_weights_kernel_probe,
            masked_rgb_inpaint_kernel_probe,
            blend_latents_with_mask_kernel_probe,
            model_input_kernel_probe,
            guidance_kernel_probe,
            scheduler_kernel_probe,
            center_unet_input_kernel_probe,
            timestep_embedding_kernel_probe,
            tensor_add_kernel_probe,
            add_channel_bias_kernel_probe,
            nchw_to_bsc_kernel_probe,
            bsc_to_nchw_kernel_probe,
            concat_channels_kernel_probe,
            concat_last_dim_2d_kernel_probe,
            concat_last_dim_3d_kernel_probe,
            conv2d_kernel_probe,
            group_norm_kernel_probe,
            silu_kernel_probe,
            quick_gelu_kernel_probe,
            clip_token_position_embedding_kernel_probe,
            upsample_kernel_probe,
            linear_kernel_probe,
            layer_norm_kernel_probe,
            softmax_kernel_probe,
            sdpa_kernel_probe,
            clip_causal_attention_kernel_probe,
            geglu_gate_kernel_probe,
            vae_moments_to_latents_kernel_probe,
        })
    }
}
