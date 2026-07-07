// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! ROCm/HIP GPU boundary ops for diffusion. GPU code always compiles (the CPU
//! reference path is a runtime choice, not a cargo feature). Each
//! `*_hip_on_gpu` function dispatches one kernel from [`crate::hip_kernels`]
//! and round-trips its activation through the host; the `*_resident` functions
//! keep activations device-resident across ops (Phase 1b). The rest are shared
//! launch/transfer helpers.

use super::*;

/// Lightweight, env-gated phase profiler for the resident DiT hot path.
///
/// Enable with `HIPFIRE_PROFILE=1`. When on, the resident linear and attention
/// paths `device_synchronize()` around each phase and accumulate wall time into
/// per-phase counters, which the block-stack loop prints and resets per step.
/// The syncs serialize the otherwise-async launches, so absolute numbers run a
/// bit slower than a normal step — read the *relative* breakdown (weight-prep
/// vs GEMM vs attention), which is what informs the w4a8/w8a8 kernel work.
pub(crate) mod profile {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub static PREP_NS: AtomicU64 = AtomicU64::new(0);
    pub static GEMM_NS: AtomicU64 = AtomicU64::new(0);
    pub static ATTN_NS: AtomicU64 = AtomicU64::new(0);
    pub static PREP_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GEMM_FLOPS: AtomicU64 = AtomicU64::new(0);
    pub static CACHE_MISS: AtomicU64 = AtomicU64::new(0);
    pub static CACHE_HIT: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        std::env::var("HIPFIRE_PROFILE").map(|v| !v.is_empty()).unwrap_or(false)
    }

    pub fn add(counter: &AtomicU64, v: u64) {
        counter.fetch_add(v, Ordering::Relaxed);
    }

    pub fn take(counter: &AtomicU64) -> u64 {
        counter.swap(0, Ordering::Relaxed)
    }
}

pub(crate) fn ensure_and_launch_diffusion_kernel(
    gpu: &mut hipfire_rdna::Gpu,
    module_name: &str,
    source: &str,
    func_name: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    kernargs: &mut hip_bridge::KernargBlob,
) -> DiffusionResult<()> {
    gpu.ensure_kernel_public(module_name, source, func_name)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.launch_kernel_blob(func_name, grid, block, shared_mem, kernargs.as_mut_slice())
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))
}

pub(crate) fn rgb_tensor_to_u8_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    tensor: &CpuTensor,
) -> DiffusionResult<RgbImageBatch> {
    let [batch, channels, height, width] = shape4(tensor)?;
    if channels != 3 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "expected RGB tensor with 3 channels, got {channels}"
        )));
    }
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input = gpu
        .upload_f32(&tensor.data, &tensor.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_bytes = batch
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(width))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidMetadata("RGB output size overflows".to_string()))?;
    let output = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_rgb_tensor_to_u8";
    let function_name = "diffusion_rgb_tensor_to_u8";
    let kernel_source = DIFFUSION_RGB_TENSOR_TO_U8_HIP_SRC;
    let total_pixels = batch
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(width))
        .ok_or_else(|| DiffusionError::InvalidMetadata("RGB pixel count overflows".to_string()))?;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input.buf.as_ptr());
    kernargs.push_ptr(output.as_ptr());
    kernargs.push_i32(total_pixels as i32);
    kernargs.push_i32(height as i32);
    kernargs.push_i32(width as i32);
    kernargs.pad_to(16);
    let grid = [((total_pixels as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mut data = vec![0u8; output_bytes];
    gpu.hip
        .memcpy_dtoh(&mut data, &output)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .free(output)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(RgbImageBatch {
        batch,
        width,
        height,
        data,
    })
}

pub(crate) fn rgb_batch_to_vae_tensor_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    batch: &RgbImageBatch,
) -> DiffusionResult<CpuTensor> {
    let bytes_per_image = batch
        .width
        .checked_mul(batch.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidRequest("image dimensions overflow".to_string()))?;
    let expected = bytes_per_image
        .checked_mul(batch.batch)
        .ok_or_else(|| DiffusionError::InvalidRequest("image batch size overflows".to_string()))?;
    if batch.data.len() != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "RGB image batch has {} bytes, expected {expected}",
            batch.data.len()
        )));
    }
    let output_shape = [batch.batch, 3, batch.height, batch.width];
    let output_elements = checked_shape_elements("RGB-to-VAE tensor output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("RGB-to-VAE tensor output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .hip
        .malloc(batch.data.len())
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .memcpy_htod(&input_gpu, &batch.data)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_rgb_u8_to_vae_nchw_f32";
    let function_name = "diffusion_rgb_u8_to_vae_nchw_f32";
    let kernel_source = DIFFUSION_VAE_BOUNDARY_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "RGB-to-VAE output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("RGB-to-VAE height", batch.height)?);
    kernargs.push_i32(i32_kernel_dim("RGB-to-VAE width", batch.width)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .free(input_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

pub(crate) fn vae_moments_to_latents_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    moments: &CpuTensor,
    scaling_factor: f32,
) -> DiffusionResult<LatentBatch> {
    let [batch, channels, height, width] = shape4(moments)?;
    if channels % 2 != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "VAE encoder moments channel count {channels} is not even"
        )));
    }
    let latent_channels = channels / 2;
    let output_shape = [batch, latent_channels, height, width];
    let output_elements = checked_shape_elements("VAE moments-to-latents output", &output_shape)?;
    if output_elements == 0 {
        return Ok(LatentBatch {
            batch,
            channels: latent_channels,
            height,
            width,
            data: Vec::new(),
        });
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata(
                "VAE moments-to-latents output size overflows".to_string(),
            )
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let moments_gpu = gpu
        .upload_f32(&moments.data, &moments.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_vae_moments_to_latents_f32";
    let function_name = "diffusion_vae_moments_to_latents_f32";
    let kernel_source = DIFFUSION_VAE_BOUNDARY_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(moments_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "VAE moments-to-latents output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("VAE moments channels", channels)?);
    kernargs.push_i32(i32_kernel_dim("VAE latent channels", latent_channels)?);
    kernargs.push_i32(i32_kernel_dim("VAE latent height", height)?);
    kernargs.push_i32(i32_kernel_dim("VAE latent width", width)?);
    kernargs.push_f32(scaling_factor.max(f32::MIN_POSITIVE));
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(LatentBatch {
        batch,
        channels: latent_channels,
        height,
        width,
        data,
    })
}

pub(crate) fn latent_mask_weights_from_rgb_batch_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    mask: &RgbImageBatch,
    latents: &LatentBatch,
) -> DiffusionResult<Vec<f32>> {
    if mask.batch != latents.batch {
        return Err(DiffusionError::InvalidRequest(format!(
            "mask batch {} != latent batch {}",
            mask.batch, latents.batch
        )));
    }
    let bytes_per_image = mask
        .width
        .checked_mul(mask.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidRequest("mask dimensions overflow".to_string()))?;
    let expected = bytes_per_image.checked_mul(mask.batch).ok_or_else(|| {
        DiffusionError::InvalidRequest("mask batch dimensions overflow".to_string())
    })?;
    if mask.data.len() != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "mask has {} bytes, expected {expected}",
            mask.data.len()
        )));
    }
    let output_elements = latents
        .batch
        .checked_mul(latents.height)
        .and_then(|pixels| pixels.checked_mul(latents.width))
        .ok_or_else(|| DiffusionError::InvalidRequest("latent mask size overflows".to_string()))?;
    if output_elements == 0 {
        return Ok(Vec::new());
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("latent mask output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mask_gpu = gpu
        .hip
        .malloc(mask.data.len())
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .memcpy_htod(&mask_gpu, &mask.data)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_latent_mask_weights_from_rgb_f32";
    let function_name = "diffusion_latent_mask_weights_from_rgb_f32";
    let kernel_source = DIFFUSION_INPAINT_MASK_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(mask_gpu.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "latent mask output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("latent mask source height", mask.height)?);
    kernargs.push_i32(i32_kernel_dim("latent mask source width", mask.width)?);
    kernargs.push_i32(i32_kernel_dim("latent mask output height", latents.height)?);
    kernargs.push_i32(i32_kernel_dim("latent mask output width", latents.width)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .free(mask_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(output)
}

pub(crate) fn masked_rgb_batch_for_inpaint_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    image: &RgbImageBatch,
    mask: &RgbImageBatch,
) -> DiffusionResult<RgbImageBatch> {
    if image.batch != mask.batch || image.width != mask.width || image.height != mask.height {
        return Err(DiffusionError::InvalidRequest(format!(
            "inpaint image shape [{}x{}x{}] != mask shape [{}x{}x{}]",
            image.batch, image.width, image.height, mask.batch, mask.width, mask.height
        )));
    }
    let expected = image
        .batch
        .checked_mul(image.width)
        .and_then(|pixels| pixels.checked_mul(image.height))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| DiffusionError::InvalidRequest("image dimensions overflow".to_string()))?;
    if image.data.len() != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "image has {} bytes, expected {expected}",
            image.data.len()
        )));
    }
    if mask.data.len() != expected {
        return Err(DiffusionError::InvalidRequest(format!(
            "mask has {} bytes, expected {expected}",
            mask.data.len()
        )));
    }
    if expected == 0 {
        return Ok(RgbImageBatch {
            batch: image.batch,
            width: image.width,
            height: image.height,
            data: Vec::new(),
        });
    }
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let image_gpu = gpu
        .hip
        .malloc(image.data.len())
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .memcpy_htod(&image_gpu, &image.data)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mask_gpu = gpu
        .hip
        .malloc(mask.data.len())
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .memcpy_htod(&mask_gpu, &mask.data)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(expected)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_masked_rgb_for_inpaint_u8";
    let function_name = "diffusion_masked_rgb_for_inpaint_u8";
    let kernel_source = DIFFUSION_INPAINT_MASK_HIP_SRC;
    let total_pixels = expected / 3;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(image_gpu.as_ptr());
    kernargs.push_ptr(mask_gpu.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("masked RGB pixels", total_pixels)?);
    kernargs.pad_to(16);
    let grid = [((total_pixels as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mut data = vec![0u8; expected];
    gpu.hip
        .memcpy_dtoh(&mut data, &output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .free(mask_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .free(image_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(RgbImageBatch {
        batch: image.batch,
        width: image.width,
        height: image.height,
        data,
    })
}

pub(crate) fn blend_latents_with_mask_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    generated: &LatentBatch,
    init: &LatentBatch,
    mask_weights: &[f32],
) -> DiffusionResult<LatentBatch> {
    if generated.batch != init.batch
        || generated.channels != init.channels
        || generated.height != init.height
        || generated.width != init.width
    {
        return Err(DiffusionError::InvalidRequest(format!(
            "generated latent shape [{}x{}x{}x{}] != init latent shape [{}x{}x{}x{}]",
            generated.batch,
            generated.channels,
            generated.height,
            generated.width,
            init.batch,
            init.channels,
            init.height,
            init.width
        )));
    }
    let expected_mask = generated.batch * generated.height * generated.width;
    if mask_weights.len() != expected_mask {
        return Err(DiffusionError::InvalidRequest(format!(
            "latent mask has {} weights, expected {expected_mask}",
            mask_weights.len()
        )));
    }
    let output_elements = generated
        .batch
        .checked_mul(generated.channels)
        .and_then(|elements| elements.checked_mul(generated.height))
        .and_then(|elements| elements.checked_mul(generated.width))
        .ok_or_else(|| {
            DiffusionError::InvalidRequest("latent output size overflows".to_string())
        })?;
    if generated.data.len() != output_elements || init.data.len() != output_elements {
        return Err(DiffusionError::InvalidRequest(format!(
            "latent data length mismatch for shape [{}x{}x{}x{}]",
            generated.batch, generated.channels, generated.height, generated.width
        )));
    }
    if output_elements == 0 {
        return Ok(generated.clone());
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("latent blend output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let generated_gpu = gpu
        .upload_f32(&generated.data, &[output_elements])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let init_gpu = gpu
        .upload_f32(&init.data, &[output_elements])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mask_gpu = gpu
        .upload_f32(mask_weights, &[mask_weights.len()])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_blend_latents_with_mask_f32";
    let function_name = "diffusion_blend_latents_with_mask_f32";
    let kernel_source = DIFFUSION_INPAINT_MASK_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(generated_gpu.buf.as_ptr());
    kernargs.push_ptr(init_gpu.buf.as_ptr());
    kernargs.push_ptr(mask_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "latent blend output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("latent blend channels", generated.channels)?);
    kernargs.push_i32(i32_kernel_dim("latent blend height", generated.height)?);
    kernargs.push_i32(i32_kernel_dim("latent blend width", generated.width)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(LatentBatch {
        batch: generated.batch,
        channels: generated.channels,
        height: generated.height,
        width: generated.width,
        data,
    })
}

pub(crate) fn launch_diffusion_vector_kernel(
    gpu: &mut hipfire_rdna::Gpu,
    function_name: &str,
    source: &str,
    output_gpu: &hip_bridge::DeviceBuffer,
    input_a: &hipfire_rdna::GpuTensor,
    input_b: Option<&hipfire_rdna::GpuTensor>,
    n: i32,
    scalar: f32,
    synchronize: bool,
) -> DiffusionResult<()> {
    let module_name = function_name;
    let kernel_source = source;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_a.buf.as_ptr());
    if let Some(input_b) = input_b {
        kernargs.push_ptr(input_b.buf.as_ptr());
    }
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(n);
    kernargs.push_f32(scalar);
    kernargs.pad_to(16);
    let grid = [((n as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    maybe_synchronize(gpu, synchronize)
}

/// Synchronize the device only when the caller round-trips its output to the
/// host immediately. Resident ops pass `false` and rely on the single
/// end-of-chain sync in [`download_resident`], collapsing ~200 per-op syncs per
/// denoise step into one.
fn maybe_synchronize(gpu: &mut hipfire_rdna::Gpu, synchronize: bool) -> DiffusionResult<()> {
    if synchronize {
        gpu.hip
            .device_synchronize()
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    }
    Ok(())
}

pub(crate) fn download_f32_buffer(
    gpu: &mut hipfire_rdna::Gpu,
    buffer: &hip_bridge::DeviceBuffer,
    elements: usize,
) -> DiffusionResult<Vec<f32>> {
    let output_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| DiffusionError::InvalidMetadata("f32 output size overflows".to_string()))?;
    let mut raw = vec![0u8; output_bytes];
    gpu.hip
        .memcpy_dtoh(&mut raw, buffer)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mut output = Vec::with_capacity(elements);
    for chunk in raw.chunks_exact(std::mem::size_of::<f32>()) {
        output.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(output)
}

pub(crate) fn scale_model_input_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    sample: &[f32],
    scale: f32,
) -> DiffusionResult<Vec<f32>> {
    if sample.is_empty() {
        return Ok(Vec::new());
    }
    let n = i32::try_from(sample.len()).map_err(|_| {
        DiffusionError::InvalidRequest(format!("model input length {} exceeds i32", sample.len()))
    })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let sample_gpu = gpu
        .upload_f32(sample, &[sample.len()])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_bytes = sample
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("model input output size overflows".to_string())
        })?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_vector_kernel(
        gpu,
        "diffusion_scale_model_input_f32",
        DIFFUSION_DENOISE_VECTOR_HIP_SRC,
        &output_gpu,
        &sample_gpu,
        None,
        n,
        scale,
        true,
    )?;
    let output = download_f32_buffer(gpu, &output_gpu, sample.len())?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(output)
}

pub(crate) fn cfg_guidance_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    negative_pred: &[f32],
    positive_pred: &[f32],
    cfg_scale: f32,
) -> DiffusionResult<Vec<f32>> {
    if negative_pred.len() != positive_pred.len() {
        return Err(DiffusionError::InvalidRequest(format!(
            "negative prediction length {} != positive prediction length {}",
            negative_pred.len(),
            positive_pred.len()
        )));
    }
    if negative_pred.is_empty() {
        return Ok(Vec::new());
    }
    let n = i32::try_from(negative_pred.len()).map_err(|_| {
        DiffusionError::InvalidRequest(format!(
            "CFG prediction length {} exceeds i32",
            negative_pred.len()
        ))
    })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let negative_gpu = gpu
        .upload_f32(negative_pred, &[negative_pred.len()])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let positive_gpu = gpu
        .upload_f32(positive_pred, &[positive_pred.len()])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_bytes = negative_pred
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| DiffusionError::InvalidMetadata("CFG output size overflows".to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_vector_kernel(
        gpu,
        "diffusion_cfg_guidance_f32",
        DIFFUSION_DENOISE_VECTOR_HIP_SRC,
        &output_gpu,
        &negative_gpu,
        Some(&positive_gpu),
        n,
        cfg_scale,
        true,
    )?;
    let output = download_f32_buffer(gpu, &output_gpu, negative_pred.len())?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(output)
}

pub(crate) fn tensor_add_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    a: &CpuTensor,
    b: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    if a.shape != b.shape {
        return Err(DiffusionError::InvalidMetadata(format!(
            "tensor_add shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        )));
    }
    if a.data.is_empty() {
        return Ok(CpuTensor::zeros(&a.shape));
    }
    let n = i32::try_from(a.data.len()).map_err(|_| {
        DiffusionError::InvalidRequest(format!("tensor_add length {} exceeds i32", a.data.len()))
    })?;
    let output_bytes = a
        .data
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("tensor_add output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let a_gpu = gpu
        .upload_f32(&a.data, &a.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let b_gpu = gpu
        .upload_f32(&b.data, &b.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_vector_kernel(
        gpu,
        "diffusion_tensor_add_f32",
        DIFFUSION_DENOISE_VECTOR_HIP_SRC,
        &output_gpu,
        &a_gpu,
        Some(&b_gpu),
        n,
        0.0,
        true,
    )?;
    let data = download_f32_buffer(gpu, &output_gpu, a.data.len())?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: a.shape.clone(),
        data,
    })
}

pub(crate) fn maybe_center_unet_input_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    sample: &CpuTensor,
    center_input_sample: bool,
) -> DiffusionResult<CpuTensor> {
    if !center_input_sample {
        return Ok(sample.clone());
    }
    if sample.data.is_empty() {
        return Ok(CpuTensor::zeros(&sample.shape));
    }
    let n = i32::try_from(sample.data.len()).map_err(|_| {
        DiffusionError::InvalidRequest(format!(
            "UNet input length {} exceeds i32",
            sample.data.len()
        ))
    })?;
    let output_bytes = sample
        .data
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("UNet centered input size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let sample_gpu = gpu
        .upload_f32(&sample.data, &sample.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_vector_kernel(
        gpu,
        "diffusion_center_unet_input_f32",
        DIFFUSION_DENOISE_VECTOR_HIP_SRC,
        &output_gpu,
        &sample_gpu,
        None,
        n,
        0.0,
        true,
    )?;
    let data = download_f32_buffer(gpu, &output_gpu, sample.data.len())?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: sample.shape.clone(),
        data,
    })
}

pub(crate) fn i32_kernel_dim(label: &str, value: usize) -> DiffusionResult<i32> {
    i32::try_from(value)
        .map_err(|_| DiffusionError::InvalidRequest(format!("{label} value {value} exceeds i32")))
}

pub(crate) fn launch_diffusion_layout_kernel(
    gpu: &mut hipfire_rdna::Gpu,
    function_name: &str,
    input_gpu: &hipfire_rdna::GpuTensor,
    bias_gpu: Option<&hipfire_rdna::GpuTensor>,
    output_gpu: &hip_bridge::DeviceBuffer,
    output_elements: usize,
    channels: usize,
    height: usize,
    width: usize,
    synchronize: bool,
) -> DiffusionResult<()> {
    let module_name = function_name;
    let kernel_source = DIFFUSION_LAYOUT_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    if let Some(bias_gpu) = bias_gpu {
        kernargs.push_ptr(bias_gpu.buf.as_ptr());
    }
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("layout output elements", output_elements)?);
    kernargs.push_i32(i32_kernel_dim("layout channels", channels)?);
    kernargs.push_i32(i32_kernel_dim("layout height", height)?);
    kernargs.push_i32(i32_kernel_dim("layout width", width)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    maybe_synchronize(gpu, synchronize)
}

pub(crate) fn add_channel_bias_nchw_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    input: &CpuTensor,
    bias: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let [batch, channels, height, width] = shape4(input)?;
    if bias.shape.as_slice() != [batch, channels] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "channel bias shape {:?} != [{batch}, {channels}]",
            bias.shape
        )));
    }
    let output_elements = checked_shape_elements("channel-bias output", &input.shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&input.shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("channel-bias output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let bias_gpu = gpu
        .upload_f32(&bias.data, &bias.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_layout_kernel(
        gpu,
        "diffusion_add_channel_bias_nchw_f32",
        &input_gpu,
        Some(&bias_gpu),
        &output_gpu,
        output_elements,
        channels,
        height,
        width,
        true,
    )?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: input.shape.clone(),
        data,
    })
}

pub(crate) fn nchw_to_bsc_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    input: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let [batch, channels, height, width] = shape4(input)?;
    let seq = height
        .checked_mul(width)
        .ok_or_else(|| DiffusionError::InvalidMetadata("BSC sequence overflows".to_string()))?;
    let output_shape = [batch, seq, channels];
    let output_elements = checked_shape_elements("NCHW-to-BSC output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("NCHW-to-BSC output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_layout_kernel(
        gpu,
        "diffusion_nchw_to_bsc_f32",
        &input_gpu,
        None,
        &output_gpu,
        output_elements,
        channels,
        height,
        width,
        true,
    )?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

pub(crate) fn bsc_to_nchw_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    input: &CpuTensor,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> DiffusionResult<CpuTensor> {
    let [input_batch, seq, input_channels] = shape3(input)?;
    if input_batch != batch || input_channels != channels || seq != height * width {
        return Err(DiffusionError::InvalidMetadata(format!(
            "BSC tensor shape {:?} cannot reshape to [{batch}, {channels}, {height}, {width}]",
            input.shape
        )));
    }
    let output_shape = [batch, channels, height, width];
    let output_elements = checked_shape_elements("BSC-to-NCHW output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("BSC-to-NCHW output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_layout_kernel(
        gpu,
        "diffusion_bsc_to_nchw_f32",
        &input_gpu,
        None,
        &output_gpu,
        output_elements,
        channels,
        height,
        width,
        true,
    )?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_diffusion_concat_kernel(
    gpu: &mut hipfire_rdna::Gpu,
    function_name: &str,
    a_gpu: &hipfire_rdna::GpuTensor,
    b_gpu: &hipfire_rdna::GpuTensor,
    output_gpu: &hip_bridge::DeviceBuffer,
    kernargs_tail: impl FnOnce(&mut hip_bridge::KernargBlob) -> DiffusionResult<()>,
    output_elements: usize,
    synchronize: bool,
) -> DiffusionResult<()> {
    let module_name = function_name;
    let kernel_source = DIFFUSION_CONCAT_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(a_gpu.buf.as_ptr());
    kernargs.push_ptr(b_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("concat output elements", output_elements)?);
    kernargs_tail(&mut kernargs)?;
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    maybe_synchronize(gpu, synchronize)
}

pub(crate) fn concat_channels_nchw_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    a: &CpuTensor,
    b: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let [batch, a_channels, height, width] = shape4(a)?;
    let [b_batch, b_channels, b_height, b_width] = shape4(b)?;
    if batch != b_batch || height != b_height || width != b_width {
        return Err(DiffusionError::InvalidMetadata(format!(
            "cannot concatenate NCHW tensors with shapes {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let out_channels = a_channels.checked_add(b_channels).ok_or_else(|| {
        DiffusionError::InvalidMetadata("concat channel count overflows".to_string())
    })?;
    let output_shape = [batch, out_channels, height, width];
    let output_elements = checked_shape_elements("NCHW channel concat output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("NCHW channel concat output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let a_gpu = gpu
        .upload_f32(&a.data, &a.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let b_gpu = gpu
        .upload_f32(&b.data, &b.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_concat_kernel(
        gpu,
        "diffusion_concat_channels_nchw_f32",
        &a_gpu,
        &b_gpu,
        &output_gpu,
        |kernargs| {
            kernargs.push_i32(i32_kernel_dim("concat left channels", a_channels)?);
            kernargs.push_i32(i32_kernel_dim("concat right channels", b_channels)?);
            kernargs.push_i32(i32_kernel_dim("concat height", height)?);
            kernargs.push_i32(i32_kernel_dim("concat width", width)?);
            Ok(())
        },
        output_elements,
        true,
    )?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

pub(crate) fn concat_last_dim_2d_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    a: &CpuTensor,
    b: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let [rows, left_width] = shape2(a)?;
    let [b_rows, right_width] = shape2(b)?;
    if rows != b_rows {
        return Err(DiffusionError::InvalidMetadata(format!(
            "cannot concatenate 2D tensors with shapes {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let output_width = left_width.checked_add(right_width).ok_or_else(|| {
        DiffusionError::InvalidMetadata("2D concat output width overflows".to_string())
    })?;
    concat_last_dim_hip_on_gpu(gpu, a, b, &[rows, output_width], left_width, right_width)
}

pub(crate) fn concat_last_dim_3d_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    a: &CpuTensor,
    b: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let [batch, seq, left_width] = shape3(a)?;
    let [b_batch, b_seq, right_width] = shape3(b)?;
    if batch != b_batch || seq != b_seq {
        return Err(DiffusionError::InvalidMetadata(format!(
            "cannot concatenate 3D tensors with shapes {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let output_width = left_width.checked_add(right_width).ok_or_else(|| {
        DiffusionError::InvalidMetadata("3D concat output width overflows".to_string())
    })?;
    concat_last_dim_hip_on_gpu(
        gpu,
        a,
        b,
        &[batch, seq, output_width],
        left_width,
        right_width,
    )
}

pub(crate) fn concat_last_dim_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    a: &CpuTensor,
    b: &CpuTensor,
    output_shape: &[usize],
    left_width: usize,
    right_width: usize,
) -> DiffusionResult<CpuTensor> {
    let output_elements = checked_shape_elements("last-dim concat output", output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("last-dim concat output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let a_gpu = gpu
        .upload_f32(&a.data, &a.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let b_gpu = gpu
        .upload_f32(&b.data, &b.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    launch_diffusion_concat_kernel(
        gpu,
        "diffusion_concat_last_dim_f32",
        &a_gpu,
        &b_gpu,
        &output_gpu,
        |kernargs| {
            kernargs.push_i32(i32_kernel_dim("concat left width", left_width)?);
            kernargs.push_i32(i32_kernel_dim("concat right width", right_width)?);
            Ok(())
        },
        output_elements,
        true,
    )?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

pub(crate) fn conv2d_nchw_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    cache: &mut RocmWeightCache,
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
    padding: usize,
    stride: usize,
) -> DiffusionResult<CpuTensor> {
    if stride == 0 {
        return Err(DiffusionError::InvalidRequest(
            "conv2d stride must be positive".to_string(),
        ));
    }
    let [batch, in_channels, in_h, in_w] = shape4(input)?;
    let [out_channels, weight_in_channels, kernel_h, kernel_w] = shape4(weight)?;
    if in_channels != weight_in_channels {
        return Err(DiffusionError::InvalidMetadata(format!(
            "conv2d input channels {in_channels} != weight input channels {weight_in_channels}"
        )));
    }
    if let Some(bias) = bias {
        if bias.shape.as_slice() != [out_channels] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "conv2d bias shape {:?} != [{out_channels}]",
                bias.shape
            )));
        }
    }
    let padded_h = in_h + 2 * padding;
    let padded_w = in_w + 2 * padding;
    if kernel_h > padded_h || kernel_w > padded_w {
        return Err(DiffusionError::InvalidMetadata(format!(
            "conv2d kernel [{kernel_h}, {kernel_w}] is larger than padded input [{padded_h}, {padded_w}]"
        )));
    }
    let out_h = (padded_h - kernel_h) / stride + 1;
    let out_w = (padded_w - kernel_w) / stride + 1;
    let output_elements =
        checked_shape_elements("conv2d output", &[batch, out_channels, out_h, out_w])?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&[batch, out_channels, out_h, out_w]));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("conv2d output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    // Weights/bias are resident: uploaded once and reused across every step and
    // CFG pass. resident_ptr returns a Copy raw pointer so no cache borrow is held
    // across the per-call input upload / output alloc below.
    let weight_ptr = cache.resident_ptr(gpu, weight)?;
    let bias_ptr = match bias {
        Some(bias) => cache.resident_ptr(gpu, bias)?,
        None => std::ptr::null_mut(),
    };
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_conv2d_nchw_f32";
    let function_name = "diffusion_conv2d_nchw_f32";
    let kernel_source = DIFFUSION_CONV2D_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(weight_ptr);
    kernargs.push_ptr(bias_ptr);
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("conv2d output elements", output_elements)?);
    kernargs.push_i32(i32_kernel_dim("conv2d batch", batch)?);
    kernargs.push_i32(i32_kernel_dim("conv2d input channels", in_channels)?);
    kernargs.push_i32(i32_kernel_dim("conv2d input height", in_h)?);
    kernargs.push_i32(i32_kernel_dim("conv2d input width", in_w)?);
    kernargs.push_i32(i32_kernel_dim("conv2d output channels", out_channels)?);
    kernargs.push_i32(i32_kernel_dim("conv2d output height", out_h)?);
    kernargs.push_i32(i32_kernel_dim("conv2d output width", out_w)?);
    kernargs.push_i32(i32_kernel_dim("conv2d kernel height", kernel_h)?);
    kernargs.push_i32(i32_kernel_dim("conv2d kernel width", kernel_w)?);
    kernargs.push_i32(i32_kernel_dim("conv2d padding", padding)?);
    kernargs.push_i32(i32_kernel_dim("conv2d stride", stride)?);
    kernargs.push_i32(if bias.is_some() { 1 } else { 0 });
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .free(input_gpu.buf)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: vec![batch, out_channels, out_h, out_w],
        data,
    })
}

pub(crate) fn group_norm_nchw_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    groups: usize,
    eps: f32,
) -> DiffusionResult<CpuTensor> {
    let [_batch, channels, height, width] = shape4(input)?;
    if groups == 0 || channels % groups != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "group_norm channels {channels} not divisible by groups {groups}"
        )));
    }
    if weight.shape.as_slice() != [channels] || bias.shape.as_slice() != [channels] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "group_norm weight/bias shapes {:?}/{:?} != [{channels}]",
            weight.shape, bias.shape
        )));
    }
    let output_elements = checked_shape_elements("group_norm output", &input.shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&input.shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("group_norm output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let weight_gpu = gpu
        .upload_f32(&weight.data, &weight.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let bias_gpu = gpu
        .upload_f32(&bias.data, &bias.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_group_norm_nchw_f32";
    let function_name = "diffusion_group_norm_nchw_f32";
    let kernel_source = DIFFUSION_GROUP_NORM_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(weight_gpu.buf.as_ptr());
    kernargs.push_ptr(bias_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "group_norm output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("group_norm channels", channels)?);
    kernargs.push_i32(i32_kernel_dim("group_norm height", height)?);
    kernargs.push_i32(i32_kernel_dim("group_norm width", width)?);
    kernargs.push_i32(i32_kernel_dim("group_norm groups", groups)?);
    kernargs.push_f32(eps);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: input.shape.clone(),
        data,
    })
}

pub(crate) fn silu_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    input: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let elements = checked_shape_elements("SiLU input", &input.shape)?;
    if elements == 0 {
        return Ok(CpuTensor::zeros(&input.shape));
    }
    let n = i32_kernel_dim("SiLU elements", elements)?;
    let output_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| DiffusionError::InvalidMetadata("SiLU output size overflows".to_string()))?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_silu_f32";
    let function_name = "diffusion_silu_f32";
    let kernel_source = DIFFUSION_SILU_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(n);
    kernargs.pad_to(16);
    let grid = [((elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: input.shape.clone(),
        data,
    })
}

/// Elementwise LeakyReLU: `x >= 0 ? x : alpha * x`. `alpha` is the negative
/// slope (0.2 for RealESRGAN / RRDBNet). Used by the super-resolution model in
/// the MrFlow staged-sampling pipeline.
#[allow(dead_code)]
pub(crate) fn leaky_relu_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    input: &CpuTensor,
    alpha: f32,
) -> DiffusionResult<CpuTensor> {
    let elements = checked_shape_elements("LeakyReLU input", &input.shape)?;
    if elements == 0 {
        return Ok(CpuTensor::zeros(&input.shape));
    }
    let n = i32_kernel_dim("LeakyReLU elements", elements)?;
    let output_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("LeakyReLU output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(n);
    kernargs.push_f32(alpha);
    kernargs.pad_to(16);
    let grid = [((elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_leaky_relu_f32",
        DIFFUSION_LEAKY_RELU_HIP_SRC,
        "diffusion_leaky_relu_f32",
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: input.shape.clone(),
        data,
    })
}

pub(crate) fn quick_gelu_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    input: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let elements = checked_shape_elements("QuickGELU input", &input.shape)?;
    if elements == 0 {
        return Ok(CpuTensor::zeros(&input.shape));
    }
    let n = i32_kernel_dim("QuickGELU elements", elements)?;
    let output_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("QuickGELU output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_quick_gelu_f32";
    let function_name = "diffusion_quick_gelu_f32";
    let kernel_source = DIFFUSION_QUICK_GELU_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(n);
    kernargs.pad_to(16);
    let grid = [((elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: input.shape.clone(),
        data,
    })
}

pub(crate) fn clip_token_position_embeddings_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    token_embedding: &CpuTensor,
    position_embedding: &CpuTensor,
    tokens: &[u32],
) -> DiffusionResult<CpuTensor> {
    let (vocab, hidden) = token_embedding.rows_cols()?;
    let (max_positions, position_hidden) = position_embedding.rows_cols()?;
    if position_hidden != hidden {
        return Err(DiffusionError::InvalidMetadata(format!(
            "CLIP position embedding hidden size {position_hidden} != token hidden size {hidden}"
        )));
    }
    if tokens.len() > max_positions {
        return Err(DiffusionError::InvalidRequest(format!(
            "CLIP token length {} exceeds position embedding length {max_positions}",
            tokens.len()
        )));
    }
    for &token in tokens {
        let token = token as usize;
        if token >= vocab {
            return Err(DiffusionError::InvalidRequest(format!(
                "CLIP token id {token} exceeds vocab {vocab}"
            )));
        }
    }
    let output_shape = [tokens.len(), hidden];
    let output_elements =
        checked_shape_elements("CLIP token-position embedding output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata(
                "CLIP token-position embedding output size overflows".to_string(),
            )
        })?;
    let token_bytes = tokens
        .iter()
        .flat_map(|token| token.to_ne_bytes())
        .collect::<Vec<_>>();
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let token_embedding_gpu = gpu
        .upload_f32(&token_embedding.data, &token_embedding.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let position_embedding_gpu = gpu
        .upload_f32(&position_embedding.data, &position_embedding.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let tokens_gpu = gpu
        .upload_raw(&token_bytes, &[tokens.len()])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_clip_token_position_embedding_f32";
    let function_name = "diffusion_clip_token_position_embedding_f32";
    let kernel_source = DIFFUSION_CLIP_EMBEDDINGS_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(token_embedding_gpu.buf.as_ptr());
    kernargs.push_ptr(position_embedding_gpu.buf.as_ptr());
    kernargs.push_ptr(tokens_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "CLIP token-position embedding output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim(
        "CLIP token-position embedding hidden size",
        hidden,
    )?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

pub(crate) fn upsample_nearest2d_nchw_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    input: &CpuTensor,
    scale: usize,
) -> DiffusionResult<CpuTensor> {
    if scale == 0 {
        return Err(DiffusionError::InvalidRequest(
            "upsample scale must be positive".to_string(),
        ));
    }
    let [batch, channels, in_h, in_w] = shape4(input)?;
    let out_h = in_h.checked_mul(scale).ok_or_else(|| {
        DiffusionError::InvalidRequest("upsample output height overflows".to_string())
    })?;
    let out_w = in_w.checked_mul(scale).ok_or_else(|| {
        DiffusionError::InvalidRequest("upsample output width overflows".to_string())
    })?;
    let output_shape = [batch, channels, out_h, out_w];
    let output_elements = checked_shape_elements("upsample output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("upsample output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_upsample_nearest2d_nchw_f32";
    let function_name = "diffusion_upsample_nearest2d_nchw_f32";
    let kernel_source = DIFFUSION_UPSAMPLE_NEAREST2D_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("upsample output elements", output_elements)?);
    kernargs.push_i32(i32_kernel_dim("upsample channels", channels)?);
    kernargs.push_i32(i32_kernel_dim("upsample input height", in_h)?);
    kernargs.push_i32(i32_kernel_dim("upsample input width", in_w)?);
    kernargs.push_i32(i32_kernel_dim("upsample output height", out_h)?);
    kernargs.push_i32(i32_kernel_dim("upsample output width", out_w)?);
    kernargs.push_i32(i32_kernel_dim("upsample scale", scale)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

/// Space-to-depth by `scale` (inverse of pixel-shuffle), NCHW:
/// `[N, C, H, W] -> [N, C*scale*scale, H/scale, W/scale]`. Matches
/// PyTorch/basicsr `pixel_unshuffle`. Used by the RealESRGAN x2 (RRDBNet) input
/// stage in the MrFlow super-resolution model.
#[allow(dead_code)]
pub(crate) fn pixel_unshuffle_nchw_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    input: &CpuTensor,
    scale: usize,
) -> DiffusionResult<CpuTensor> {
    if scale == 0 {
        return Err(DiffusionError::InvalidRequest(
            "pixel_unshuffle scale must be positive".to_string(),
        ));
    }
    let [batch, channels, in_h, in_w] = shape4(input)?;
    if in_h % scale != 0 || in_w % scale != 0 {
        return Err(DiffusionError::InvalidRequest(format!(
            "pixel_unshuffle input [{in_h}, {in_w}] not divisible by scale {scale}"
        )));
    }
    let out_h = in_h / scale;
    let out_w = in_w / scale;
    let out_channels = channels.checked_mul(scale * scale).ok_or_else(|| {
        DiffusionError::InvalidRequest("pixel_unshuffle output channels overflows".to_string())
    })?;
    let output_shape = [batch, out_channels, out_h, out_w];
    let output_elements = checked_shape_elements("pixel_unshuffle output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("pixel_unshuffle output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("pixel_unshuffle output elements", output_elements)?);
    kernargs.push_i32(i32_kernel_dim("pixel_unshuffle output channels", out_channels)?);
    kernargs.push_i32(i32_kernel_dim("pixel_unshuffle output height", out_h)?);
    kernargs.push_i32(i32_kernel_dim("pixel_unshuffle output width", out_w)?);
    kernargs.push_i32(i32_kernel_dim("pixel_unshuffle scale", scale)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_pixel_unshuffle_nchw_f32",
        DIFFUSION_PIXEL_UNSHUFFLE_HIP_SRC,
        "diffusion_pixel_unshuffle_nchw_f32",
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

pub(crate) fn linear_optional_bias_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    cache: &mut RocmWeightCache,
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
) -> DiffusionResult<CpuTensor> {
    let (rows, in_features) = input.rows_cols()?;
    let (out_features, weight_in) = weight.rows_cols()?;
    if in_features != weight_in {
        return Err(DiffusionError::InvalidMetadata(format!(
            "linear input width {in_features} != weight input width {weight_in}"
        )));
    }
    if let Some(bias) = bias {
        if bias.shape.as_slice() != [out_features] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "linear bias shape {:?} != [{out_features}]",
                bias.shape
            )));
        }
    }
    let output_shape = [rows, out_features];
    let output_elements = checked_shape_elements("linear output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("linear output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;

    // BF16-native WMMA path (the memory-efficient, high-throughput default). The
    // weight stays resident as BF16 (half the F32 footprint, matching the model's
    // native bf16 storage) and the naive f32 GEMM is replaced by
    // gemm_bf16_x_bf16_wmma: bf16 weight x f32 activation (staged to bf16
    // kernel-side) -> f32 output. Map A=weight[out,in] (M=out,K=in),
    // X=act[rows,in] (B=rows) -> Y[rows,out]. Requires wave32 WMMA and a
    // 16-aligned in_features (the WMMA K tile); otherwise the naive f32 path runs.
    if gpu.arch_caps.has_wmma_w32() && in_features % 16 == 0 {
        let weight_ptr = cache.resident_bf16_ptr(gpu, weight)?;
        let weight_bytes = out_features
            .checked_mul(in_features)
            .and_then(|v| v.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| {
                DiffusionError::InvalidMetadata("linear weight size overflows".to_string())
            })?;
        let weight_view = hipfire_rdna::GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(weight_ptr, weight_bytes) },
            shape: vec![out_features, in_features],
            dtype: hipfire_rdna::DType::BF16,
        };
        let output_gpu = gpu
            .alloc_tensor(&[rows, out_features], hipfire_rdna::DType::F32)
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        gpu.gemm_bf16_x_bf16_wmma(
            &weight_view,
            &input_gpu,
            &output_gpu,
            out_features,
            in_features,
            rows,
        )
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        if let Some(bias) = bias {
            let bias_ptr = cache.resident_ptr(gpu, bias)?;
            let mut bias_args = hip_bridge::KernargBlob::new();
            bias_args.push_ptr(output_gpu.buf.as_ptr());
            bias_args.push_ptr(bias_ptr);
            bias_args.push_i32(i32_kernel_dim("linear bias elements", output_elements)?);
            bias_args.push_i32(i32_kernel_dim("linear out features", out_features)?);
            bias_args.pad_to(16);
            let bias_grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
            ensure_and_launch_diffusion_kernel(
                gpu,
                "diffusion_row_bias_f32",
                DIFFUSION_ROW_BIAS_HIP_SRC,
                "diffusion_row_bias_f32",
                bias_grid,
                [256, 1, 1],
                0,
                &mut bias_args,
            )?;
        }
        gpu.hip
            .device_synchronize()
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        let data = download_f32_buffer(gpu, &output_gpu.buf, output_elements)?;
        gpu.free_tensor(output_gpu)
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        gpu.free_tensor(input_gpu)
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        return Ok(CpuTensor {
            shape: output_shape.to_vec(),
            data,
        });
    }

    // Naive f32 fallback (non-16-aligned in_features or no wave32 WMMA).
    // Resident (cached) weight/bias; only the activation is uploaded per call.
    let weight_ptr = cache.resident_ptr(gpu, weight)?;
    let bias_ptr = match bias {
        Some(bias) => cache.resident_ptr(gpu, bias)?,
        None => std::ptr::null_mut(),
    };
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_linear_f32";
    let function_name = "diffusion_linear_f32";
    let kernel_source = DIFFUSION_LINEAR_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(weight_ptr);
    kernargs.push_ptr(bias_ptr);
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("linear output elements", output_elements)?);
    kernargs.push_i32(i32_kernel_dim("linear input features", in_features)?);
    kernargs.push_i32(i32_kernel_dim("linear output features", out_features)?);
    kernargs.push_i32(if bias.is_some() { 1 } else { 0 });
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .free(input_gpu.buf)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

/// Linear over a source-reference `ResidentWeight`, using the persistent
/// name-keyed BF16 weight cache (uploaded once, reused every step) and the bf16
/// WMMA GEMM. Falls back to decoding + the CpuTensor linear when the bf16 WMMA
/// path is unavailable (no wave32 WMMA, or non-16-aligned in_features).
pub(crate) fn linear_resident_weight_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    cache: &mut RocmWeightCache,
    input: &CpuTensor,
    weight: &ResidentWeight,
    bias: Option<&CpuTensor>,
) -> DiffusionResult<CpuTensor> {
    let (out_features, in_features) = match weight.shape() {
        [out, inf] => (*out, *inf),
        other => {
            return Err(DiffusionError::InvalidMetadata(format!(
                "resident linear weight must be 2-D [out, in], got {other:?}"
            )))
        }
    };
    let (rows, input_in) = input.rows_cols()?;
    if input_in != in_features {
        return Err(DiffusionError::InvalidMetadata(format!(
            "linear input width {input_in} != weight input width {in_features}"
        )));
    }
    // Non-eligible dims / archs: decode once and use the CpuTensor linear.
    if !(gpu.arch_caps.has_wmma_w32() && in_features % 16 == 0) {
        return linear_optional_bias_hip_on_gpu(gpu, cache, input, &weight.decode()?, bias);
    }
    let output_shape = [rows, out_features];
    let output_elements = checked_shape_elements("linear output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let weight_ptr = cache.resident_bf16_named(gpu, weight)?;
    let weight_bytes = out_features
        .checked_mul(in_features)
        .and_then(|v| v.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| DiffusionError::InvalidMetadata("linear weight size overflows".to_string()))?;
    let weight_view = hipfire_rdna::GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(weight_ptr, weight_bytes) },
        shape: vec![out_features, in_features],
        dtype: hipfire_rdna::DType::BF16,
    };
    let output_gpu = gpu
        .alloc_tensor(&[rows, out_features], hipfire_rdna::DType::F32)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.gemm_bf16_x_bf16_wmma(
        &weight_view,
        &input_gpu,
        &output_gpu,
        out_features,
        in_features,
        rows,
    )
    .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    if let Some(bias) = bias {
        let bias_ptr = cache.resident_ptr(gpu, bias)?;
        let mut bias_args = hip_bridge::KernargBlob::new();
        bias_args.push_ptr(output_gpu.buf.as_ptr());
        bias_args.push_ptr(bias_ptr);
        bias_args.push_i32(i32_kernel_dim("linear bias elements", output_elements)?);
        bias_args.push_i32(i32_kernel_dim("linear out features", out_features)?);
        bias_args.pad_to(16);
        let bias_grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
        ensure_and_launch_diffusion_kernel(
            gpu,
            "diffusion_row_bias_f32",
            DIFFUSION_ROW_BIAS_HIP_SRC,
            "diffusion_row_bias_f32",
            bias_grid,
            [256, 1, 1],
            0,
            &mut bias_args,
        )?;
    }
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu.buf, output_elements)?;
    gpu.free_tensor(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.free_tensor(input_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

/// Fully resident linear: a resident `GpuTensor` activation × a streaming bf16
/// `ResidentWeight` -> a resident `GpuTensor` output, with no host round-trip
/// for the activation. The weight is uploaded once (name-keyed bf16 cache); the
/// activation stays on-device. Leading dims of `input` are preserved; only the
/// last dim (`in_features`) is replaced by `out_features`. Requires wave32 WMMA
/// and 16-aligned `in_features` (the DiT/encoder hidden dims satisfy this).
#[allow(dead_code)]
pub(crate) fn linear_resident_weight_resident(
    gpu: &mut hipfire_rdna::Gpu,
    cache: &mut RocmWeightCache,
    input: &hipfire_rdna::GpuTensor,
    weight: &ResidentWeight,
    bias: Option<&CpuTensor>,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let (out_features, in_features) = match weight.shape() {
        [out, inf] => (*out, *inf),
        other => {
            return Err(DiffusionError::InvalidMetadata(format!(
                "resident linear weight must be 2-D [out, in], got {other:?}"
            )))
        }
    };
    let in_dim = input.shape.last().copied().ok_or_else(|| {
        DiffusionError::InvalidMetadata("resident linear input has no dims".to_string())
    })?;
    if in_dim != in_features {
        return Err(DiffusionError::InvalidMetadata(format!(
            "resident linear input width {in_dim} != weight input width {in_features}"
        )));
    }
    if !(gpu.arch_caps.has_wmma_w32() && in_features % 16 == 0) {
        return Err(DiffusionError::BackendUnavailable(format!(
            "resident linear needs wave32 WMMA and 16-aligned in_features (got {in_features})"
        )));
    }
    let total = checked_shape_elements("resident linear input", &input.shape)?;
    let rows = total / in_features;
    let mut output_shape = input.shape.clone();
    *output_shape.last_mut().expect("input has a last dim") = out_features;
    let output_elements = rows
        .checked_mul(out_features)
        .ok_or_else(|| DiffusionError::InvalidMetadata("resident linear size overflows".to_string()))?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &output_shape)?;
    if output_elements == 0 {
        return Ok(output);
    }
    let prof = profile::enabled();
    // W8A8 (oq8) path: quantize the weight once at load to int8 + per-group f32
    // scales, dynamic-int8 quantize the activation, and run the register-tiled
    // oq8 GEMM — ~2x the tiled bf16 kernel on the DiT FFN shapes and ~half the
    // resident weight footprint. gfx1151 keeps its own m128 bf16 path; requires
    // 256-aligned in_features. Opt in with HIPFIRE_DIFFUSION_OQ8=1.
    let use_oq8 = gpu.arch != "gfx1151"
        && in_features % 256 == 0
        && std::env::var("HIPFIRE_DIFFUSION_OQ8").ok().as_deref() == Some("1");
    if use_oq8 {
        const GROUP: usize = 256;
        let prep_start = if prof { Some(std::time::Instant::now()) } else { None };
        let (w_i8_ptr, w_scales_ptr, ng) = cache.resident_oq8(gpu, weight)?;
        let w_i8_bytes = out_features
            .checked_mul(in_features)
            .ok_or_else(|| DiffusionError::InvalidMetadata("oq8 weight size overflows".into()))?;
        if let Some(start) = prep_start {
            let _ = gpu.hip.device_synchronize();
            profile::add(&profile::PREP_NS, start.elapsed().as_nanos() as u64);
            profile::add(&profile::PREP_BYTES, w_i8_bytes as u64);
        }
        let w_i8_view = hipfire_rdna::GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(w_i8_ptr, w_i8_bytes) },
            shape: vec![w_i8_bytes],
            dtype: hipfire_rdna::DType::Raw,
        };
        let w_scales_view = hipfire_rdna::GpuTensor {
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(w_scales_ptr, out_features * ng * 4)
            },
            shape: vec![out_features * ng],
            dtype: hipfire_rdna::DType::F32,
        };
        let xq = gpu
            .alloc_tensor(&[total], hipfire_rdna::DType::Raw)
            .map_err(|e| DiffusionError::BackendUnavailable(e.to_string()))?;
        let xs = gpu
            .alloc_tensor(&[rows * ng], hipfire_rdna::DType::F32)
            .map_err(|e| DiffusionError::BackendUnavailable(e.to_string()))?;
        gpu.quantize_act_oq8(input, &xq, &xs, rows, in_features, GROUP)
            .map_err(|e| DiffusionError::BackendUnavailable(e.to_string()))?;
        let gemm_start = if prof { Some(std::time::Instant::now()) } else { None };
        gpu.gemm_oq8_tiled_wmma(
            &w_i8_view, &w_scales_view, &xq, &xs, &output, out_features, in_features, rows, GROUP,
            2, 4,
        )
        .map_err(|e| DiffusionError::BackendUnavailable(e.to_string()))?;
        if let Some(start) = gemm_start {
            let _ = gpu.hip.device_synchronize();
            profile::add(&profile::GEMM_NS, start.elapsed().as_nanos() as u64);
            let flops = (out_features as u64)
                .saturating_mul(in_features as u64)
                .saturating_mul(rows as u64)
                .saturating_mul(2);
            profile::add(&profile::GEMM_FLOPS, flops);
        }
        free_resident(gpu, xq)?;
        free_resident(gpu, xs)?;
    } else {
    let prep_start = if prof {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let weight_ptr = cache.resident_bf16_named(gpu, weight)?;
    let weight_bytes = out_features
        .checked_mul(in_features)
        .and_then(|v| v.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| DiffusionError::InvalidMetadata("linear weight size overflows".to_string()))?;
    if let Some(start) = prep_start {
        // Sync so a cache-miss upload is fully attributed to weight-prep.
        let _ = gpu.hip.device_synchronize();
        profile::add(&profile::PREP_NS, start.elapsed().as_nanos() as u64);
        profile::add(&profile::PREP_BYTES, weight_bytes as u64);
    }
    let weight_view = hipfire_rdna::GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(weight_ptr, weight_bytes) },
        shape: vec![out_features, in_features],
        dtype: hipfire_rdna::DType::BF16,
    };
    let gemm_start = if prof {
        Some(std::time::Instant::now())
    } else {
        None
    };
    // Register-tiled 4x4 WMMA is ~2.4x the naive one-tile-per-wave kernel on the
    // dense DiT shapes (gfx1103). gfx1151 keeps its own m128 LDS path inside
    // gemm_bf16_x_bf16_wmma, so only route the tiled kernel off gfx1151. Opt out
    // with HIPFIRE_DIFFUSION_TILED_GEMM=0.
    let use_tiled = gpu.arch != "gfx1151"
        && std::env::var("HIPFIRE_DIFFUSION_TILED_GEMM").ok().as_deref() != Some("0");
    if use_tiled {
        gpu.gemm_bf16_tiled_wmma(&weight_view, input, &output, out_features, in_features, rows, 4, 4)
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    } else {
        gpu.gemm_bf16_x_bf16_wmma(&weight_view, input, &output, out_features, in_features, rows)
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    }
    if let Some(start) = gemm_start {
        let _ = gpu.hip.device_synchronize();
        profile::add(&profile::GEMM_NS, start.elapsed().as_nanos() as u64);
        // 2 FLOPs per MAC.
        let flops = (out_features as u64)
            .saturating_mul(in_features as u64)
            .saturating_mul(rows as u64)
            .saturating_mul(2);
        profile::add(&profile::GEMM_FLOPS, flops);
    }
    }
    if let Some(bias) = bias {
        let bias_ptr = cache.resident_ptr(gpu, bias)?;
        let mut bias_args = hip_bridge::KernargBlob::new();
        bias_args.push_ptr(output.buf.as_ptr());
        bias_args.push_ptr(bias_ptr);
        bias_args.push_i32(i32_kernel_dim("linear bias elements", output_elements)?);
        bias_args.push_i32(i32_kernel_dim("linear out features", out_features)?);
        bias_args.pad_to(16);
        let bias_grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
        ensure_and_launch_diffusion_kernel(
            gpu,
            "diffusion_row_bias_f32",
            DIFFUSION_ROW_BIAS_HIP_SRC,
            "diffusion_row_bias_f32",
            bias_grid,
            [256, 1, 1],
            0,
            &mut bias_args,
        )?;
    }
    Ok(output)
}

pub(crate) fn layer_norm_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    eps: f32,
) -> DiffusionResult<CpuTensor> {
    let (rows, cols) = input.rows_cols()?;
    if weight.shape.as_slice() != [cols] || bias.shape.as_slice() != [cols] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "layer_norm weight/bias shapes {:?}/{:?} do not match width {cols}",
            weight.shape, bias.shape
        )));
    }
    let output_shape = [rows, cols];
    let output_elements = checked_shape_elements("layer_norm output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("layer_norm output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let weight_gpu = gpu
        .upload_f32(&weight.data, &weight.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let bias_gpu = gpu
        .upload_f32(&bias.data, &bias.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_layer_norm_f32";
    let function_name = "diffusion_layer_norm_f32";
    let kernel_source = DIFFUSION_LAYER_NORM_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(weight_gpu.buf.as_ptr());
    kernargs.push_ptr(bias_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "layer_norm output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("layer_norm width", cols)?);
    kernargs.push_f32(eps);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

pub(crate) fn softmax_rows_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    input: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let (rows, cols) = input.rows_cols()?;
    let output_elements = checked_shape_elements("softmax output", &input.shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&input.shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("softmax output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&input.data, &input.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_softmax_rows_f32";
    let function_name = "diffusion_softmax_rows_f32";
    let kernel_source = DIFFUSION_SOFTMAX_ROWS_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("softmax rows", rows)?);
    kernargs.push_i32(i32_kernel_dim("softmax cols", cols)?);
    kernargs.pad_to(16);
    let grid = [((rows as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: input.shape.clone(),
        data,
    })
}

pub(crate) fn scaled_dot_product_attention_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    heads: usize,
) -> DiffusionResult<CpuTensor> {
    let [batch, q_seq, hidden] = shape3(q)?;
    let [k_batch, k_seq, k_hidden] = shape3(k)?;
    let [v_batch, v_seq, v_hidden] = shape3(v)?;
    if batch != k_batch || batch != v_batch || k_seq != v_seq || k_hidden != v_hidden {
        return Err(DiffusionError::InvalidMetadata(format!(
            "attention q/k/v shapes {:?}/{:?}/{:?} are incompatible",
            q.shape, k.shape, v.shape
        )));
    }
    if heads == 0 || hidden != k_hidden || hidden % heads != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "attention hidden size {hidden} is incompatible with key size {k_hidden} and heads {heads}"
        )));
    }
    let head_dim = hidden / heads;
    let output_shape = [batch, q_seq, hidden];
    let output_elements = checked_shape_elements("SDPA output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| DiffusionError::InvalidMetadata("SDPA output size overflows".to_string()))?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let q_gpu = gpu
        .upload_f32(&q.data, &q.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let k_gpu = gpu
        .upload_f32(&k.data, &k.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let v_gpu = gpu
        .upload_f32(&v.data, &v.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_sdpa_3d_f32";
    let function_name = "diffusion_sdpa_3d_f32";
    let kernel_source = DIFFUSION_SDPA_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(q_gpu.buf.as_ptr());
    kernargs.push_ptr(k_gpu.buf.as_ptr());
    kernargs.push_ptr(v_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim("SDPA output elements", output_elements)?);
    kernargs.push_i32(i32_kernel_dim("SDPA query sequence", q_seq)?);
    kernargs.push_i32(i32_kernel_dim("SDPA key sequence", k_seq)?);
    kernargs.push_i32(i32_kernel_dim("SDPA hidden size", hidden)?);
    kernargs.push_i32(i32_kernel_dim("SDPA heads", heads)?);
    kernargs.push_i32(i32_kernel_dim("SDPA head dim", head_dim)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

pub(crate) fn clip_causal_self_attention_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    n_heads: usize,
) -> DiffusionResult<CpuTensor> {
    let (seq, hidden) = q.rows_cols()?;
    if k.shape.as_slice() != [seq, hidden] || v.shape.as_slice() != [seq, hidden] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "CLIP causal attention q/k/v shapes {:?}/{:?}/{:?} are incompatible",
            q.shape, k.shape, v.shape
        )));
    }
    if n_heads == 0 || hidden % n_heads != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "CLIP hidden size {hidden} is not divisible by {n_heads} heads"
        )));
    }
    let head_dim = hidden / n_heads;
    let output_shape = [seq, hidden];
    let output_elements = checked_shape_elements("CLIP causal attention output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata(
                "CLIP causal attention output size overflows".to_string(),
            )
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let q_gpu = gpu
        .upload_f32(&q.data, &q.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let k_gpu = gpu
        .upload_f32(&k.data, &k.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let v_gpu = gpu
        .upload_f32(&v.data, &v.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_clip_causal_attention_f32";
    let function_name = "diffusion_clip_causal_attention_f32";
    let kernel_source = DIFFUSION_CLIP_CAUSAL_ATTENTION_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(q_gpu.buf.as_ptr());
    kernargs.push_ptr(k_gpu.buf.as_ptr());
    kernargs.push_ptr(v_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "CLIP causal attention output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("CLIP causal attention sequence", seq)?);
    kernargs.push_i32(i32_kernel_dim("CLIP causal attention hidden size", hidden)?);
    kernargs.push_i32(i32_kernel_dim("CLIP causal attention heads", n_heads)?);
    kernargs.push_i32(i32_kernel_dim("CLIP causal attention head dim", head_dim)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

pub(crate) fn geglu_gate_3d_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    projected: &CpuTensor,
) -> DiffusionResult<CpuTensor> {
    let [batch, seq, width] = shape3(projected)?;
    if width % 2 != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "GEGLU projection width {width} is not even"
        )));
    }
    let inner = width / 2;
    let output_shape = [batch, seq, inner];
    let output_elements = checked_shape_elements("GeGLU gate output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("GeGLU gate output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let input_gpu = gpu
        .upload_f32(&projected.data, &projected.shape)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_geglu_gate_3d_f32";
    let function_name = "diffusion_geglu_gate_3d_f32";
    let kernel_source = DIFFUSION_GEGLU_GATE_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "GeGLU gate output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("GeGLU gate inner width", inner)?);
    kernargs.push_i32(i32_kernel_dim("GeGLU gate projected width", width)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

pub(crate) fn timestep_embedding_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    timesteps: &[f32],
    dim: usize,
    flip_sin_to_cos: bool,
    freq_shift: f32,
) -> DiffusionResult<CpuTensor> {
    if dim == 0 {
        return Err(DiffusionError::InvalidRequest(
            "timestep embedding dimension must be positive".to_string(),
        ));
    }
    let output_shape = [timesteps.len(), dim];
    let output_elements = checked_shape_elements("timestep embedding output", &output_shape)?;
    if output_elements == 0 {
        return Ok(CpuTensor::zeros(&output_shape));
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("timestep embedding output size overflows".to_string())
        })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let timesteps_gpu = gpu
        .upload_f32(timesteps, &[timesteps.len()])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_timestep_embedding_f32";
    let function_name = "diffusion_timestep_embedding_f32";
    let kernel_source = DIFFUSION_TIMESTEP_EMBEDDING_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(timesteps_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "timestep embedding output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("timestep embedding dimension", dim)?);
    kernargs.push_i32(i32_kernel_dim(
        "timestep embedding half dimension",
        dim / 2,
    )?);
    kernargs.push_i32(if flip_sin_to_cos { 1 } else { 0 });
    kernargs.push_f32(freq_shift);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &output_gpu, output_elements)?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(CpuTensor {
        shape: output_shape.to_vec(),
        data,
    })
}

pub(crate) fn scheduler_prediction_type_id(prediction_type: SchedulerPredictionType) -> i32 {
    match prediction_type {
        SchedulerPredictionType::Epsilon => 0,
        SchedulerPredictionType::Sample => 1,
        SchedulerPredictionType::VPrediction => 2,
    }
}

pub(crate) fn euler_step_hip_on_gpu(
    gpu: &mut hipfire_rdna::Gpu,
    sample: &[f32],
    model_output: &[f32],
    sigma: f32,
    next_sigma: f32,
    prediction_type: SchedulerPredictionType,
) -> DiffusionResult<Vec<f32>> {
    if sample.len() != model_output.len() {
        return Err(DiffusionError::InvalidRequest(format!(
            "sample length {} != model output length {}",
            sample.len(),
            model_output.len()
        )));
    }
    if sample.is_empty() {
        return Ok(Vec::new());
    }
    let n = i32::try_from(sample.len()).map_err(|_| {
        DiffusionError::InvalidRequest(format!(
            "scheduler input length {} exceeds i32",
            sample.len()
        ))
    })?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let sample_gpu = gpu
        .upload_f32(sample, &[sample.len()])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let model_output_gpu = gpu
        .upload_f32(model_output, &[model_output.len()])
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output_bytes = sample
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("scheduler output size overflows".to_string())
        })?;
    let output_gpu = gpu
        .hip
        .malloc(output_bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let module_name = "diffusion_euler_step_f32";
    let function_name = "diffusion_euler_step_f32";
    let kernel_source = DIFFUSION_EULER_STEP_HIP_SRC;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(sample_gpu.buf.as_ptr());
    kernargs.push_ptr(model_output_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.as_ptr());
    kernargs.push_i32(n);
    kernargs.push_f32(sigma);
    kernargs.push_f32(next_sigma);
    kernargs.push_i32(scheduler_prediction_type_id(prediction_type));
    kernargs.pad_to(16);
    let grid = [((sample.len() as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        module_name,
        kernel_source,
        function_name,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mut raw = vec![0u8; output_bytes];
    gpu.hip
        .memcpy_dtoh(&mut raw, &output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    gpu.hip
        .free(output_gpu)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let mut output = Vec::with_capacity(sample.len());
    for chunk in raw.chunks_exact(std::mem::size_of::<f32>()) {
        output.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Phase 1b — device-resident activation ops.
//
// The `*_hip_on_gpu` functions above round-trip every activation through the
// host (`upload input → launch → sync → download → free`). For a UNet/VAE
// forward that is ~200 round-trips per denoise step and dominates wall-clock.
//
// The `*_resident` functions below take device-resident inputs (`&GpuTensor`)
// and return a device-resident output (`GpuTensor`), so intermediate
// activations never touch the host between ops. The caller uploads the latents
// once at the top of the forward pass and downloads the result once at the
// bottom. Weights/bias still come from the resident `RocmWeightCache`.
//
// Output buffers are allocated through `Gpu::alloc_tensor`, which is backed by
// the recycling `GpuPool` (`crates/hipfire-rdna/src/pool.rs`) — so there is no
// per-op `hipMalloc`/`hipFree` churn and, because `GpuTensor` has no `Drop`,
// the caller is responsible for `free_resident`-ing every intermediate it no
// longer needs (a missed free leaks device memory over a run). A `*_resident`
// op never frees its inputs; the orchestrating forward chain owns lifetimes.
//
// These ops keep the per-op `device_synchronize` for now (correctness-first);
// dropping it for a single end-of-step sync is a later Phase 1b step.
// ---------------------------------------------------------------------------

/// Allocate a pooled, device-resident F32 output tensor of `shape`.
pub(crate) fn alloc_resident_f32(
    gpu: &mut hipfire_rdna::Gpu,
    shape: &[usize],
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    gpu.alloc_tensor(shape, hipfire_rdna::DType::F32)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))
}

/// Return a resident intermediate to the pool. Call this on every `GpuTensor`
/// the forward chain stops needing — `GpuTensor` has no `Drop`.
pub(crate) fn free_resident(
    gpu: &mut hipfire_rdna::Gpu,
    tensor: hipfire_rdna::GpuTensor,
) -> DiffusionResult<()> {
    gpu.free_tensor(tensor)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))
}

/// Device-to-device copy of a resident tensor into a fresh pooled buffer. Used
/// for UNet skip snapshots, where the host path clones an activation that is
/// then mutated further down the chain.
pub(crate) fn clone_resident(
    gpu: &mut hipfire_rdna::Gpu,
    src: &hipfire_rdna::GpuTensor,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let elements = checked_shape_elements("resident clone", &src.shape)?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DiffusionError::InvalidMetadata("resident clone size overflows".to_string())
        })?;
    let dst = alloc_resident_f32(gpu, &src.shape)?;
    gpu.copy_d2d(src, &dst, bytes)
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    Ok(dst)
}

/// Download a resident tensor to a host `CpuTensor` (used once at the end of a
/// resident forward chain). Does not free `tensor`. This is the single
/// synchronization point for the resident path: the per-op `device_synchronize`
/// calls are skipped (the ops run in submission order on one stream), so we sync
/// once here before reading device memory back to the host.
pub(crate) fn download_resident(
    gpu: &mut hipfire_rdna::Gpu,
    tensor: &hipfire_rdna::GpuTensor,
) -> DiffusionResult<CpuTensor> {
    let elements = checked_shape_elements("resident download", &tensor.shape)?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let data = download_f32_buffer(gpu, &tensor.buf, elements)?;
    Ok(CpuTensor {
        shape: tensor.shape.clone(),
        data,
    })
}

fn resident_dims4(shape: &[usize], label: &str) -> DiffusionResult<[usize; 4]> {
    match shape {
        [a, b, c, d] => Ok([*a, *b, *c, *d]),
        _ => Err(DiffusionError::InvalidMetadata(format!(
            "{label} expected a 4D tensor, got shape {shape:?}"
        ))),
    }
}

fn resident_dims3(shape: &[usize], label: &str) -> DiffusionResult<[usize; 3]> {
    match shape {
        [a, b, c] => Ok([*a, *b, *c]),
        _ => Err(DiffusionError::InvalidMetadata(format!(
            "{label} expected a 3D tensor, got shape {shape:?}"
        ))),
    }
}

/// Device-resident NCHW conv2d. Weights/bias are resident via `cache`; only the
/// (already-resident) activation flows through. Mirrors `conv2d_nchw_hip_on_gpu`
/// minus the input upload and output download.
pub(crate) fn conv2d_nchw_resident(
    gpu: &mut hipfire_rdna::Gpu,
    cache: &mut RocmWeightCache,
    input: &hipfire_rdna::GpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
    padding: usize,
    stride: usize,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    if stride == 0 {
        return Err(DiffusionError::InvalidRequest(
            "conv2d stride must be positive".to_string(),
        ));
    }
    let [batch, in_channels, in_h, in_w] = resident_dims4(&input.shape, "conv2d input")?;
    let [out_channels, weight_in_channels, kernel_h, kernel_w] = shape4(weight)?;
    if in_channels != weight_in_channels {
        return Err(DiffusionError::InvalidMetadata(format!(
            "conv2d input channels {in_channels} != weight input channels {weight_in_channels}"
        )));
    }
    if let Some(bias) = bias {
        if bias.shape.as_slice() != [out_channels] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "conv2d bias shape {:?} != [{out_channels}]",
                bias.shape
            )));
        }
    }
    let padded_h = in_h + 2 * padding;
    let padded_w = in_w + 2 * padding;
    if kernel_h > padded_h || kernel_w > padded_w {
        return Err(DiffusionError::InvalidMetadata(format!(
            "conv2d kernel [{kernel_h}, {kernel_w}] is larger than padded input [{padded_h}, {padded_w}]"
        )));
    }
    let out_h = (padded_h - kernel_h) / stride + 1;
    let out_w = (padded_w - kernel_w) / stride + 1;
    let output_shape = [batch, out_channels, out_h, out_w];
    let output_elements = checked_shape_elements("conv2d output", &output_shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let weight_ptr = cache.resident_ptr(gpu, weight)?;
    let bias_ptr = match bias {
        Some(bias) => cache.resident_ptr(gpu, bias)?,
        None => std::ptr::null_mut(),
    };
    let output = alloc_resident_f32(gpu, &output_shape)?;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input.buf.as_ptr());
    kernargs.push_ptr(weight_ptr);
    kernargs.push_ptr(bias_ptr);
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(i32_kernel_dim("conv2d output elements", output_elements)?);
    kernargs.push_i32(i32_kernel_dim("conv2d batch", batch)?);
    kernargs.push_i32(i32_kernel_dim("conv2d input channels", in_channels)?);
    kernargs.push_i32(i32_kernel_dim("conv2d input height", in_h)?);
    kernargs.push_i32(i32_kernel_dim("conv2d input width", in_w)?);
    kernargs.push_i32(i32_kernel_dim("conv2d output channels", out_channels)?);
    kernargs.push_i32(i32_kernel_dim("conv2d output height", out_h)?);
    kernargs.push_i32(i32_kernel_dim("conv2d output width", out_w)?);
    kernargs.push_i32(i32_kernel_dim("conv2d kernel height", kernel_h)?);
    kernargs.push_i32(i32_kernel_dim("conv2d kernel width", kernel_w)?);
    kernargs.push_i32(i32_kernel_dim("conv2d padding", padding)?);
    kernargs.push_i32(i32_kernel_dim("conv2d stride", stride)?);
    kernargs.push_i32(if bias.is_some() { 1 } else { 0 });
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_conv2d_nchw_f32",
        DIFFUSION_CONV2D_HIP_SRC,
        "diffusion_conv2d_nchw_f32",
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}

/// Phase 3 device-resident NCHW conv via im2col + WMMA GEMM. Lowers the
/// activation to a column matrix, runs `Gpu::gemm_f16_wmma` (F16 weights, F32
/// activations cast lane-side, F32 accumulate) once per batch — whose `[OC,
/// OH*OW]` output lands directly as the NCHW slice — then adds bias. Falls back
/// to the direct F32 conv on architectures without wave32 WMMA (e.g. RDNA2).
///
/// Precision: the GEMM inputs are F16 (the accumulator is F32), so results match
/// the F32 reference only to ~F16 tolerance, not 1e-5. This is the standard
/// SD/DiT inference tradeoff and is gated accordingly.
pub(crate) fn conv2d_nchw_wmma_resident(
    gpu: &mut hipfire_rdna::Gpu,
    cache: &mut RocmWeightCache,
    input: &hipfire_rdna::GpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
    padding: usize,
    stride: usize,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    if stride == 0 {
        return Err(DiffusionError::InvalidRequest(
            "conv2d stride must be positive".to_string(),
        ));
    }
    // No matrix cores (RDNA2 etc.) → keep the portable direct-conv path.
    if !gpu.arch_caps.has_wmma_w32() {
        return conv2d_nchw_resident(gpu, cache, input, weight, bias, padding, stride);
    }
    let [batch, in_channels, in_h, in_w] = resident_dims4(&input.shape, "conv2d(wmma) input")?;
    let [out_channels, weight_in_channels, kernel_h, kernel_w] = shape4(weight)?;
    if in_channels != weight_in_channels {
        return Err(DiffusionError::InvalidMetadata(format!(
            "conv2d input channels {in_channels} != weight input channels {weight_in_channels}"
        )));
    }
    if let Some(bias) = bias {
        if bias.shape.as_slice() != [out_channels] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "conv2d bias shape {:?} != [{out_channels}]",
                bias.shape
            )));
        }
    }
    let padded_h = in_h + 2 * padding;
    let padded_w = in_w + 2 * padding;
    if kernel_h > padded_h || kernel_w > padded_w {
        return Err(DiffusionError::InvalidMetadata(format!(
            "conv2d kernel [{kernel_h}, {kernel_w}] is larger than padded input [{padded_h}, {padded_w}]"
        )));
    }
    let out_h = (padded_h - kernel_h) / stride + 1;
    let out_w = (padded_w - kernel_w) / stride + 1;
    let output_shape = [batch, out_channels, out_h, out_w];
    let output_elements = checked_shape_elements("conv2d output", &output_shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &output_shape)?;
    if output_elements == 0 {
        return Ok(output);
    }
    let k_dim = in_channels
        .checked_mul(kernel_h)
        .and_then(|v| v.checked_mul(kernel_w))
        .ok_or_else(|| DiffusionError::InvalidMetadata("conv2d K dim overflows".to_string()))?;
    let spatial = out_h
        .checked_mul(out_w)
        .ok_or_else(|| DiffusionError::InvalidMetadata("conv2d spatial overflows".to_string()))?;

    // Resident F16 weight [OC, K] and resident F32 bias.
    let weight_f16_ptr = cache.resident_f16_ptr(gpu, weight)?;
    let bias_ptr = match bias {
        Some(bias) => Some(cache.resident_ptr(gpu, bias)?),
        None => None,
    };

    // Implicit-GEMM conv: one fused WMMA kernel that gathers the im2col columns
    // from the input on the fly (no materialized column matrix). grid.z = batch;
    // each (16×16) output tile is W_f16[OC,K] @ X_bᵀ for one batch slice, landing
    // directly as that batch's NCHW output.
    let grid = [
        i32_kernel_dim("conv grid M", out_channels.div_ceil(16))? as u32,
        i32_kernel_dim("conv grid N", spatial.div_ceil(16))? as u32,
        i32_kernel_dim("conv grid batch", batch)? as u32,
    ];
    let mut conv_args = hip_bridge::KernargBlob::new();
    conv_args.push_ptr(weight_f16_ptr);
    conv_args.push_ptr(input.buf.as_ptr());
    conv_args.push_ptr(output.buf.as_ptr());
    conv_args.push_i32(i32_kernel_dim("conv OC", out_channels)?);
    conv_args.push_i32(i32_kernel_dim("conv K", k_dim)?);
    conv_args.push_i32(i32_kernel_dim("conv OHW", spatial)?);
    conv_args.push_i32(i32_kernel_dim("conv IC", in_channels)?);
    conv_args.push_i32(i32_kernel_dim("conv IH", in_h)?);
    conv_args.push_i32(i32_kernel_dim("conv IW", in_w)?);
    conv_args.push_i32(i32_kernel_dim("conv OH", out_h)?);
    conv_args.push_i32(i32_kernel_dim("conv OW", out_w)?);
    conv_args.push_i32(i32_kernel_dim("conv KH", kernel_h)?);
    conv_args.push_i32(i32_kernel_dim("conv KW", kernel_w)?);
    conv_args.push_i32(i32_kernel_dim("conv pad", padding)?);
    conv_args.push_i32(i32_kernel_dim("conv stride", stride)?);
    conv_args.pad_to(16);
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_conv2d_implicit_wmma_f16",
        DIFFUSION_CONV2D_IMPLICIT_WMMA_HIP_SRC,
        "diffusion_conv2d_implicit_wmma_f16",
        grid,
        [32, 1, 1],
        0,
        &mut conv_args,
    )?;

    // Per-output-channel bias.
    if let Some(bias_ptr) = bias_ptr {
        let mut bias_args = hip_bridge::KernargBlob::new();
        bias_args.push_ptr(output.buf.as_ptr());
        bias_args.push_ptr(bias_ptr);
        bias_args.push_i32(i32_kernel_dim("conv bias elements", output_elements)?);
        bias_args.push_i32(i32_kernel_dim("conv bias spatial", spatial)?);
        bias_args.push_i32(i32_kernel_dim("conv bias out channels", out_channels)?);
        bias_args.pad_to(16);
        let bias_grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
        ensure_and_launch_diffusion_kernel(
            gpu,
            "diffusion_conv_bias_nchw_f32",
            DIFFUSION_CONV_BIAS_NCHW_HIP_SRC,
            "diffusion_conv_bias_nchw_f32",
            bias_grid,
            [256, 1, 1],
            0,
            &mut bias_args,
        )?;
    }
    Ok(output)
}

/// Device-resident NCHW group-norm. Weight/bias are uploaded once via `cache`.
pub(crate) fn group_norm_nchw_resident(
    gpu: &mut hipfire_rdna::Gpu,
    cache: &mut RocmWeightCache,
    input: &hipfire_rdna::GpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    groups: usize,
    eps: f32,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let [batch, channels, height, width] = resident_dims4(&input.shape, "group_norm input")?;
    if groups == 0 || channels % groups != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "group_norm channels {channels} not divisible by groups {groups}"
        )));
    }
    if weight.shape.as_slice() != [channels] || bias.shape.as_slice() != [channels] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "group_norm weight/bias shapes {:?}/{:?} != [{channels}]",
            weight.shape, bias.shape
        )));
    }
    let output_elements = checked_shape_elements("group_norm output", &input.shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let weight_ptr = cache.resident_ptr(gpu, weight)?;
    let bias_ptr = cache.resident_ptr(gpu, bias)?;
    let output = alloc_resident_f32(gpu, &input.shape)?;
    if output_elements == 0 {
        return Ok(output);
    }
    // Two-pass: a wave-per-group reduction computes mean/inv_std (O(N)), then an
    // elementwise pass applies the affine transform.
    let bg = batch.checked_mul(groups).ok_or_else(|| {
        DiffusionError::InvalidMetadata("group_norm batch*groups overflows".to_string())
    })?;
    let mean = alloc_resident_f32(gpu, &[bg])?;
    let inv_std = alloc_resident_f32(gpu, &[bg])?;
    let mut stats_args = hip_bridge::KernargBlob::new();
    stats_args.push_ptr(input.buf.as_ptr());
    stats_args.push_ptr(mean.buf.as_ptr());
    stats_args.push_ptr(inv_std.buf.as_ptr());
    stats_args.push_i32(i32_kernel_dim("group_norm channels", channels)?);
    stats_args.push_i32(i32_kernel_dim("group_norm height", height)?);
    stats_args.push_i32(i32_kernel_dim("group_norm width", width)?);
    stats_args.push_i32(i32_kernel_dim("group_norm groups", groups)?);
    stats_args.push_f32(eps);
    stats_args.pad_to(16);
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_group_norm_stats_f32",
        DIFFUSION_GROUP_NORM_STATS_HIP_SRC,
        "diffusion_group_norm_stats_f32",
        [bg as u32, 1, 1],
        [32, 1, 1],
        0,
        &mut stats_args,
    )?;
    let mut apply_args = hip_bridge::KernargBlob::new();
    apply_args.push_ptr(input.buf.as_ptr());
    apply_args.push_ptr(mean.buf.as_ptr());
    apply_args.push_ptr(inv_std.buf.as_ptr());
    apply_args.push_ptr(weight_ptr);
    apply_args.push_ptr(bias_ptr);
    apply_args.push_ptr(output.buf.as_ptr());
    apply_args.push_i32(i32_kernel_dim(
        "group_norm output elements",
        output_elements,
    )?);
    apply_args.push_i32(i32_kernel_dim("group_norm channels", channels)?);
    apply_args.push_i32(i32_kernel_dim("group_norm height", height)?);
    apply_args.push_i32(i32_kernel_dim("group_norm width", width)?);
    apply_args.push_i32(i32_kernel_dim("group_norm groups", groups)?);
    apply_args.pad_to(16);
    let apply_grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_group_norm_apply_f32",
        DIFFUSION_GROUP_NORM_APPLY_HIP_SRC,
        "diffusion_group_norm_apply_f32",
        apply_grid,
        [256, 1, 1],
        0,
        &mut apply_args,
    )?;
    free_resident(gpu, mean)?;
    free_resident(gpu, inv_std)?;
    Ok(output)
}

/// Device-resident SiLU.
pub(crate) fn silu_resident(
    gpu: &mut hipfire_rdna::Gpu,
    input: &hipfire_rdna::GpuTensor,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let elements = checked_shape_elements("SiLU input", &input.shape)?;
    let n = i32_kernel_dim("SiLU elements", elements)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &input.shape)?;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input.buf.as_ptr());
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(n);
    kernargs.pad_to(16);
    let grid = [((elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_silu_f32",
        DIFFUSION_SILU_HIP_SRC,
        "diffusion_silu_f32",
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}

/// Device-resident LeakyReLU: `x >= 0 ? x : alpha * x`. `alpha` is the negative
/// slope (0.2 for RealESRGAN / RRDBNet). The resident sibling of
/// [`leaky_relu_hip_on_gpu`], for the super-resolution forward chain.
#[allow(dead_code)]
pub(crate) fn leaky_relu_resident(
    gpu: &mut hipfire_rdna::Gpu,
    input: &hipfire_rdna::GpuTensor,
    alpha: f32,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let elements = checked_shape_elements("LeakyReLU input", &input.shape)?;
    let n = i32_kernel_dim("LeakyReLU elements", elements)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &input.shape)?;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input.buf.as_ptr());
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(n);
    kernargs.push_f32(alpha);
    kernargs.pad_to(16);
    let grid = [((elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_leaky_relu_f32",
        DIFFUSION_LEAKY_RELU_HIP_SRC,
        "diffusion_leaky_relu_f32",
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}

/// Device-resident elementwise add (`a + b`). Shapes must match.
pub(crate) fn tensor_add_resident(
    gpu: &mut hipfire_rdna::Gpu,
    a: &hipfire_rdna::GpuTensor,
    b: &hipfire_rdna::GpuTensor,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    if a.shape != b.shape {
        return Err(DiffusionError::InvalidMetadata(format!(
            "tensor_add shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        )));
    }
    let elements = checked_shape_elements("tensor_add output", &a.shape)?;
    let n = i32_kernel_dim("tensor_add elements", elements)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &a.shape)?;
    launch_diffusion_vector_kernel(
        gpu,
        "diffusion_tensor_add_f32",
        DIFFUSION_DENOISE_VECTOR_HIP_SRC,
        &output.buf,
        a,
        Some(b),
        n,
        0.0,
        false,
    )?;
    Ok(output)
}

/// Device-resident scaled add (`a + scale * b`). Shapes must match. RRDBNet
/// scales each residual-dense and RRDB residual by 0.2 before adding; this is
/// that fused op for the MrFlow super-resolution forward chain.
#[allow(dead_code)]
pub(crate) fn scaled_add_resident(
    gpu: &mut hipfire_rdna::Gpu,
    a: &hipfire_rdna::GpuTensor,
    b: &hipfire_rdna::GpuTensor,
    scale: f32,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    if a.shape != b.shape {
        return Err(DiffusionError::InvalidMetadata(format!(
            "scaled_add shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        )));
    }
    let elements = checked_shape_elements("scaled_add output", &a.shape)?;
    let n = i32_kernel_dim("scaled_add elements", elements)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &a.shape)?;
    launch_diffusion_vector_kernel(
        gpu,
        "diffusion_scaled_add_f32",
        DIFFUSION_DENOISE_VECTOR_HIP_SRC,
        &output.buf,
        a,
        Some(b),
        n,
        scale,
        false,
    )?;
    Ok(output)
}

/// Device-resident channel-bias add (`input[n,c,h,w] += bias[n,c]`), returning a
/// new resident tensor. Used by the UNet resnet time-embedding path (Phase 1b
/// step 4); kept here with the rest of the resident op set.
#[allow(dead_code)]
pub(crate) fn add_channel_bias_nchw_resident(
    gpu: &mut hipfire_rdna::Gpu,
    input: &hipfire_rdna::GpuTensor,
    bias: &hipfire_rdna::GpuTensor,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let [batch, channels, height, width] = resident_dims4(&input.shape, "channel-bias input")?;
    if bias.shape.as_slice() != [batch, channels] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "channel bias shape {:?} != [{batch}, {channels}]",
            bias.shape
        )));
    }
    let output_elements = checked_shape_elements("channel-bias output", &input.shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &input.shape)?;
    launch_diffusion_layout_kernel(
        gpu,
        "diffusion_add_channel_bias_nchw_f32",
        input,
        Some(bias),
        &output.buf,
        output_elements,
        channels,
        height,
        width,
        false,
    )?;
    Ok(output)
}

/// Device-resident nearest-neighbour 2× (or `scale`×) upsample in NCHW.
pub(crate) fn upsample_nearest2d_nchw_resident(
    gpu: &mut hipfire_rdna::Gpu,
    input: &hipfire_rdna::GpuTensor,
    scale: usize,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    if scale == 0 {
        return Err(DiffusionError::InvalidRequest(
            "upsample scale must be positive".to_string(),
        ));
    }
    let [batch, channels, in_h, in_w] = resident_dims4(&input.shape, "upsample input")?;
    let out_h = in_h.checked_mul(scale).ok_or_else(|| {
        DiffusionError::InvalidRequest("upsample output height overflows".to_string())
    })?;
    let out_w = in_w.checked_mul(scale).ok_or_else(|| {
        DiffusionError::InvalidRequest("upsample output width overflows".to_string())
    })?;
    let output_shape = [batch, channels, out_h, out_w];
    let output_elements = checked_shape_elements("upsample output", &output_shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &output_shape)?;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input.buf.as_ptr());
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(i32_kernel_dim("upsample output elements", output_elements)?);
    kernargs.push_i32(i32_kernel_dim("upsample channels", channels)?);
    kernargs.push_i32(i32_kernel_dim("upsample input height", in_h)?);
    kernargs.push_i32(i32_kernel_dim("upsample input width", in_w)?);
    kernargs.push_i32(i32_kernel_dim("upsample output height", out_h)?);
    kernargs.push_i32(i32_kernel_dim("upsample output width", out_w)?);
    kernargs.push_i32(i32_kernel_dim("upsample scale", scale)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_upsample_nearest2d_nchw_f32",
        DIFFUSION_UPSAMPLE_NEAREST2D_HIP_SRC,
        "diffusion_upsample_nearest2d_nchw_f32",
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}

/// Device-resident space-to-depth by `scale` (inverse of pixel-shuffle), NCHW:
/// `[N, C, H, W] -> [N, C*scale*scale, H/scale, W/scale]`. The resident sibling
/// of [`pixel_unshuffle_nchw_hip_on_gpu`], for the RRDBNet input stage.
#[allow(dead_code)]
pub(crate) fn pixel_unshuffle_nchw_resident(
    gpu: &mut hipfire_rdna::Gpu,
    input: &hipfire_rdna::GpuTensor,
    scale: usize,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    if scale == 0 {
        return Err(DiffusionError::InvalidRequest(
            "pixel_unshuffle scale must be positive".to_string(),
        ));
    }
    let [batch, channels, in_h, in_w] = resident_dims4(&input.shape, "pixel_unshuffle input")?;
    if in_h % scale != 0 || in_w % scale != 0 {
        return Err(DiffusionError::InvalidRequest(format!(
            "pixel_unshuffle input [{in_h}, {in_w}] not divisible by scale {scale}"
        )));
    }
    let out_h = in_h / scale;
    let out_w = in_w / scale;
    let out_channels = channels.checked_mul(scale * scale).ok_or_else(|| {
        DiffusionError::InvalidRequest("pixel_unshuffle output channels overflows".to_string())
    })?;
    let output_shape = [batch, out_channels, out_h, out_w];
    let output_elements = checked_shape_elements("pixel_unshuffle output", &output_shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &output_shape)?;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input.buf.as_ptr());
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(i32_kernel_dim("pixel_unshuffle output elements", output_elements)?);
    kernargs.push_i32(i32_kernel_dim("pixel_unshuffle output channels", out_channels)?);
    kernargs.push_i32(i32_kernel_dim("pixel_unshuffle output height", out_h)?);
    kernargs.push_i32(i32_kernel_dim("pixel_unshuffle output width", out_w)?);
    kernargs.push_i32(i32_kernel_dim("pixel_unshuffle scale", scale)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_pixel_unshuffle_nchw_f32",
        DIFFUSION_PIXEL_UNSHUFFLE_HIP_SRC,
        "diffusion_pixel_unshuffle_nchw_f32",
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}

/// Device-resident NCHW→BSC layout change ([b,c,h,w] → [b,h*w,c]).
pub(crate) fn nchw_to_bsc_resident(
    gpu: &mut hipfire_rdna::Gpu,
    input: &hipfire_rdna::GpuTensor,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let [batch, channels, height, width] = resident_dims4(&input.shape, "NCHW-to-BSC input")?;
    let seq = height
        .checked_mul(width)
        .ok_or_else(|| DiffusionError::InvalidMetadata("BSC sequence overflows".to_string()))?;
    let output_shape = [batch, seq, channels];
    let output_elements = checked_shape_elements("NCHW-to-BSC output", &output_shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &output_shape)?;
    launch_diffusion_layout_kernel(
        gpu,
        "diffusion_nchw_to_bsc_f32",
        input,
        None,
        &output.buf,
        output_elements,
        channels,
        height,
        width,
        false,
    )?;
    Ok(output)
}

/// Device-resident BSC→NCHW layout change ([b,h*w,c] → [b,c,h,w]).
pub(crate) fn bsc_to_nchw_resident(
    gpu: &mut hipfire_rdna::Gpu,
    input: &hipfire_rdna::GpuTensor,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let [input_batch, seq, input_channels] = resident_dims3(&input.shape, "BSC-to-NCHW input")?;
    if input_batch != batch || input_channels != channels || seq != height * width {
        return Err(DiffusionError::InvalidMetadata(format!(
            "BSC tensor shape {:?} cannot reshape to [{batch}, {channels}, {height}, {width}]",
            input.shape
        )));
    }
    let output_shape = [batch, channels, height, width];
    let output_elements = checked_shape_elements("BSC-to-NCHW output", &output_shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &output_shape)?;
    launch_diffusion_layout_kernel(
        gpu,
        "diffusion_bsc_to_nchw_f32",
        input,
        None,
        &output.buf,
        output_elements,
        channels,
        height,
        width,
        false,
    )?;
    Ok(output)
}

/// Device-resident linear (`y = x·Wᵀ + b`). Accepts a 2D `[rows, in]` or 3D
/// `[b, seq, in]` resident input; the output preserves the leading dims with the
/// last dim replaced by `out_features`. Weight/bias are resident via `cache`.
pub(crate) fn linear_optional_bias_resident(
    gpu: &mut hipfire_rdna::Gpu,
    cache: &mut RocmWeightCache,
    input: &hipfire_rdna::GpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let (out_features, in_features) = weight.rows_cols()?;
    let last = input.shape.last().copied().ok_or_else(|| {
        DiffusionError::InvalidMetadata("linear input must have at least one dim".to_string())
    })?;
    if last != in_features {
        return Err(DiffusionError::InvalidMetadata(format!(
            "linear input width {last} != weight input width {in_features}"
        )));
    }
    if let Some(bias) = bias {
        if bias.shape.as_slice() != [out_features] {
            return Err(DiffusionError::InvalidMetadata(format!(
                "linear bias shape {:?} != [{out_features}]",
                bias.shape
            )));
        }
    }
    let total = checked_shape_elements("linear input", &input.shape)?;
    if in_features == 0 || total % in_features != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "linear input element count {total} is not a multiple of input width {in_features}"
        )));
    }
    let mut output_shape = input.shape.clone();
    *output_shape
        .last_mut()
        .expect("input shape checked non-empty above") = out_features;
    let output_elements = checked_shape_elements("linear output", &output_shape)?;
    let rows = total / in_features;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &output_shape)?;
    if output_elements == 0 {
        return Ok(output);
    }

    // Progressive precision schedule: when an oq4 rung is active and the input
    // dim is 256-aligned (the oq4 GEMM constraint), run the linear at reduced
    // precision (oq4 weight + FWHT-rotated activation). Other linears (e.g.
    // in=320/640) and the F16 setting fall through to the f16 WMMA path. The
    // per-layer index is advanced for every resident linear so the per-layer
    // policy (every Nth layer, skip first/last) indexes consistently.
    let layer_idx = cache.linear_index;
    cache.linear_index = cache.linear_index.wrapping_add(1);
    let precision = cache.resolve_linear_precision(layer_idx);
    if precision != LinearPrecision::F16 && in_features % 256 == 0 && gpu.arch_caps.has_wmma_w32() {
        let w_ptr = cache.resident_oq4_ptr(gpu, weight, out_features, in_features)?;
        let packed_len = oq4_arch_combined_len(out_features, in_features);
        let w_view = hipfire_rdna::GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(w_ptr, packed_len) },
            shape: vec![packed_len],
            dtype: hipfire_rdna::DType::Raw,
        };
        let x_rot = gpu
            .alloc_tensor(&[total], hipfire_rdna::DType::F32)
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        gpu.rotate_x_mq_batched(input, &x_rot, in_features, rows)
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        let launched = match precision {
            LinearPrecision::W4A16 => gpu.gemm_oq4_grouped_f16_wmma(
                &w_view,
                &x_rot,
                &output,
                out_features,
                in_features,
                rows,
                256,
            ),
            LinearPrecision::W4A8 => gpu.gemm_oq4_residual_mmq(
                &w_view,
                &x_rot,
                &output,
                out_features,
                in_features,
                rows,
                false,
            ),
            LinearPrecision::W4A4 => gpu.gemm_oq4_grouped_act_batched(
                &w_view,
                &x_rot,
                &output,
                out_features,
                in_features,
                rows,
            ),
            LinearPrecision::F16 => unreachable!("F16 handled by the fall-through path"),
        };
        launched.map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        free_resident(gpu, x_rot)?;
        if let Some(bias) = bias {
            let bias_ptr = cache.resident_ptr(gpu, bias)?;
            let mut bias_args = hip_bridge::KernargBlob::new();
            bias_args.push_ptr(output.buf.as_ptr());
            bias_args.push_ptr(bias_ptr);
            bias_args.push_i32(i32_kernel_dim("linear bias elements", output_elements)?);
            bias_args.push_i32(i32_kernel_dim("linear out features", out_features)?);
            bias_args.pad_to(16);
            let bias_grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
            ensure_and_launch_diffusion_kernel(
                gpu,
                "diffusion_row_bias_f32",
                DIFFUSION_ROW_BIAS_HIP_SRC,
                "diffusion_row_bias_f32",
                bias_grid,
                [256, 1, 1],
                0,
                &mut bias_args,
            )?;
        }
        return Ok(output);
    }

    if gpu.arch_caps.has_wmma_w32() && in_features % 16 == 0 {
        // BF16-native WMMA path (the memory-efficient default). The weight stays
        // resident as BF16 — half the F32 footprint and lossless from the model's
        // bf16 source, vs the bf16 -> f32 -> f16 round-trip. gemm_bf16_x_bf16_wmma
        // computes Y[B,M] = A_bf16[M,K] @ X[B,K]^T with the F32 input staged to
        // bf16 kernel-side. Map A=weight[out,in] (M=out,K=in), X=act[rows,in]
        // (B=rows) -> Y[rows,out], the linear's natural layout, no transpose.
        let weight_ptr = cache.resident_bf16_ptr(gpu, weight)?;
        let weight_bytes = out_features
            .checked_mul(in_features)
            .and_then(|v| v.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| {
                DiffusionError::InvalidMetadata("linear weight size overflows".to_string())
            })?;
        let weight_view = hipfire_rdna::GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(weight_ptr, weight_bytes) },
            shape: vec![out_features, in_features],
            dtype: hipfire_rdna::DType::BF16,
        };
        gpu.gemm_bf16_x_bf16_wmma(
            &weight_view,
            input,
            &output,
            out_features,
            in_features,
            rows,
        )
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        if let Some(bias) = bias {
            let bias_ptr = cache.resident_ptr(gpu, bias)?;
            let mut bias_args = hip_bridge::KernargBlob::new();
            bias_args.push_ptr(output.buf.as_ptr());
            bias_args.push_ptr(bias_ptr);
            bias_args.push_i32(i32_kernel_dim("linear bias elements", output_elements)?);
            bias_args.push_i32(i32_kernel_dim("linear out features", out_features)?);
            bias_args.pad_to(16);
            let bias_grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
            ensure_and_launch_diffusion_kernel(
                gpu,
                "diffusion_row_bias_f32",
                DIFFUSION_ROW_BIAS_HIP_SRC,
                "diffusion_row_bias_f32",
                bias_grid,
                [256, 1, 1],
                0,
                &mut bias_args,
            )?;
        }
        return Ok(output);
    }

    if gpu.arch_caps.has_wmma_w32() {
        // Phase 3 WMMA path (fallback for non-16-aligned in_features).
        // gemm_f16_wmma computes Y[M,N] = W_f16[M,K] @ X_f32[N,K]^T.
        // Mapping M=rows, K=in, N=out with the *activation* as the F16 W operand
        // and the F32 weight as the X operand (cast lane-side) yields
        // Y[rows, out] directly — the linear's natural layout, no transpose.
        let act_f16 = gpu
            .alloc_tensor(&[total], hipfire_rdna::DType::F16)
            .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        let mut convert_args = hip_bridge::KernargBlob::new();
        convert_args.push_ptr(input.buf.as_ptr());
        convert_args.push_ptr(act_f16.buf.as_ptr());
        convert_args.push_i32(i32_kernel_dim("linear activation elements", total)?);
        convert_args.pad_to(16);
        let convert_grid = [((total as u32).saturating_add(255)) / 256, 1, 1];
        ensure_and_launch_diffusion_kernel(
            gpu,
            "diffusion_f32_to_f16",
            DIFFUSION_F32_TO_F16_HIP_SRC,
            "diffusion_f32_to_f16",
            convert_grid,
            [256, 1, 1],
            0,
            &mut convert_args,
        )?;
        // Resident F32 weight, wrapped as a non-owning X[out, in] operand.
        let weight_ptr = cache.resident_ptr(gpu, weight)?;
        let weight_bytes = out_features
            .checked_mul(in_features)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                DiffusionError::InvalidMetadata("linear weight size overflows".to_string())
            })?;
        let weight_view = hipfire_rdna::GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(weight_ptr, weight_bytes) },
            shape: vec![out_features, in_features],
            dtype: hipfire_rdna::DType::F32,
        };
        gpu.gemm_f16_wmma(
            &act_f16,
            &weight_view,
            &output,
            rows,
            in_features,
            out_features,
        )
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
        free_resident(gpu, act_f16)?;
        if let Some(bias) = bias {
            let bias_ptr = cache.resident_ptr(gpu, bias)?;
            let mut bias_args = hip_bridge::KernargBlob::new();
            bias_args.push_ptr(output.buf.as_ptr());
            bias_args.push_ptr(bias_ptr);
            bias_args.push_i32(i32_kernel_dim("linear bias elements", output_elements)?);
            bias_args.push_i32(i32_kernel_dim("linear out features", out_features)?);
            bias_args.pad_to(16);
            let bias_grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
            ensure_and_launch_diffusion_kernel(
                gpu,
                "diffusion_row_bias_f32",
                DIFFUSION_ROW_BIAS_HIP_SRC,
                "diffusion_row_bias_f32",
                bias_grid,
                [256, 1, 1],
                0,
                &mut bias_args,
            )?;
        }
        return Ok(output);
    }

    // Fallback for architectures without wave32 WMMA: the naive direct linear.
    let weight_ptr = cache.resident_ptr(gpu, weight)?;
    let bias_ptr = match bias {
        Some(bias) => cache.resident_ptr(gpu, bias)?,
        None => std::ptr::null_mut(),
    };
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input.buf.as_ptr());
    kernargs.push_ptr(weight_ptr);
    kernargs.push_ptr(bias_ptr);
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(i32_kernel_dim("linear output elements", output_elements)?);
    kernargs.push_i32(i32_kernel_dim("linear input features", in_features)?);
    kernargs.push_i32(i32_kernel_dim("linear output features", out_features)?);
    kernargs.push_i32(if bias.is_some() { 1 } else { 0 });
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_linear_f32",
        DIFFUSION_LINEAR_HIP_SRC,
        "diffusion_linear_f32",
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}

/// Device-resident scaled-dot-product attention over 3D `[b, seq, hidden]` q/k/v.
pub(crate) fn scaled_dot_product_attention_resident(
    gpu: &mut hipfire_rdna::Gpu,
    q: &hipfire_rdna::GpuTensor,
    k: &hipfire_rdna::GpuTensor,
    v: &hipfire_rdna::GpuTensor,
    heads: usize,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let [batch, q_seq, hidden] = resident_dims3(&q.shape, "SDPA query")?;
    let [k_batch, k_seq, k_hidden] = resident_dims3(&k.shape, "SDPA key")?;
    let [v_batch, v_seq, v_hidden] = resident_dims3(&v.shape, "SDPA value")?;
    if batch != k_batch || batch != v_batch || k_seq != v_seq || k_hidden != v_hidden {
        return Err(DiffusionError::InvalidMetadata(format!(
            "attention q/k/v shapes {:?}/{:?}/{:?} are incompatible",
            q.shape, k.shape, v.shape
        )));
    }
    if heads == 0 || hidden != k_hidden || hidden % heads != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "attention hidden size {hidden} is incompatible with key size {k_hidden} and heads {heads}"
        )));
    }
    let head_dim = hidden / heads;
    let output_shape = [batch, q_seq, hidden];
    let output_elements = checked_shape_elements("SDPA output", &output_shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &output_shape)?;
    if output_elements == 0 {
        return Ok(output);
    }

    // Flash path: one wave per (batch, head, query), online softmax, no LDS, no
    // seq×seq materialization. Capped at 16 channels/lane = head_dim 512; above
    // that, fall back to the naive kernel.
    const FLASH_MAX_HEAD_DIM: usize = 512;
    if head_dim <= FLASH_MAX_HEAD_DIM {
        let waves = batch
            .checked_mul(heads)
            .and_then(|v| v.checked_mul(q_seq))
            .ok_or_else(|| {
                DiffusionError::InvalidMetadata("SDPA wave count overflows".to_string())
            })?;
        const WAVES_PER_BLOCK: usize = 8; // 256 threads / 32
        let mut kernargs = hip_bridge::KernargBlob::new();
        kernargs.push_ptr(q.buf.as_ptr());
        kernargs.push_ptr(k.buf.as_ptr());
        kernargs.push_ptr(v.buf.as_ptr());
        kernargs.push_ptr(output.buf.as_ptr());
        kernargs.push_i32(i32_kernel_dim("flash batch", batch)?);
        kernargs.push_i32(i32_kernel_dim("flash query sequence", q_seq)?);
        kernargs.push_i32(i32_kernel_dim("flash key sequence", k_seq)?);
        kernargs.push_i32(i32_kernel_dim("flash hidden size", hidden)?);
        kernargs.push_i32(i32_kernel_dim("flash heads", heads)?);
        kernargs.push_i32(i32_kernel_dim("flash head dim", head_dim)?);
        kernargs.pad_to(16);
        let blocks = waves.div_ceil(WAVES_PER_BLOCK);
        let grid = [i32_kernel_dim("flash grid", blocks)? as u32, 1, 1];
        ensure_and_launch_diffusion_kernel(
            gpu,
            "diffusion_flash_attention_f32",
            DIFFUSION_FLASH_ATTENTION_HIP_SRC,
            "diffusion_flash_attention_f32",
            grid,
            [(WAVES_PER_BLOCK * 32) as u32, 1, 1],
            0,
            &mut kernargs,
        )?;
        return Ok(output);
    }

    // Fallback: naive SDPA for very large head_dim (> 512).
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(q.buf.as_ptr());
    kernargs.push_ptr(k.buf.as_ptr());
    kernargs.push_ptr(v.buf.as_ptr());
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(i32_kernel_dim("SDPA output elements", output_elements)?);
    kernargs.push_i32(i32_kernel_dim("SDPA query sequence", q_seq)?);
    kernargs.push_i32(i32_kernel_dim("SDPA key sequence", k_seq)?);
    kernargs.push_i32(i32_kernel_dim("SDPA hidden size", hidden)?);
    kernargs.push_i32(i32_kernel_dim("SDPA heads", heads)?);
    kernargs.push_i32(i32_kernel_dim("SDPA head dim", head_dim)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_sdpa_3d_f32",
        DIFFUSION_SDPA_HIP_SRC,
        "diffusion_sdpa_3d_f32",
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}

/// Device-resident layer-norm over the last dim. Accepts a 2D `[rows, width]` or
/// 3D `[b, seq, width]` resident input; the output keeps the same shape.
/// Weight/bias are resident via `cache`.
pub(crate) fn layer_norm_resident(
    gpu: &mut hipfire_rdna::Gpu,
    cache: &mut RocmWeightCache,
    input: &hipfire_rdna::GpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    eps: f32,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let width = input.shape.last().copied().ok_or_else(|| {
        DiffusionError::InvalidMetadata("layer_norm input must have at least one dim".to_string())
    })?;
    if weight.shape.as_slice() != [width] || bias.shape.as_slice() != [width] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "layer_norm weight/bias shapes {:?}/{:?} do not match width {width}",
            weight.shape, bias.shape
        )));
    }
    let output_elements = checked_shape_elements("layer_norm output", &input.shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &input.shape)?;
    if output_elements == 0 || width == 0 {
        return Ok(output);
    }
    let weight_ptr = cache.resident_ptr(gpu, weight)?;
    let bias_ptr = cache.resident_ptr(gpu, bias)?;
    // Wave-per-row two-pass normalization (O(rows*cols), no LDS).
    let rows = output_elements / width;
    const WAVES_PER_BLOCK: usize = 8;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input.buf.as_ptr());
    kernargs.push_ptr(weight_ptr);
    kernargs.push_ptr(bias_ptr);
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(i32_kernel_dim("layer_norm rows", rows)?);
    kernargs.push_i32(i32_kernel_dim("layer_norm width", width)?);
    kernargs.push_f32(eps);
    kernargs.pad_to(16);
    let blocks = rows.div_ceil(WAVES_PER_BLOCK);
    let grid = [i32_kernel_dim("layer_norm grid", blocks)? as u32, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_layer_norm_rows_f32",
        DIFFUSION_LAYER_NORM_ROWS_HIP_SRC,
        "diffusion_layer_norm_rows_f32",
        grid,
        [(WAVES_PER_BLOCK * 32) as u32, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}

/// Device-resident weighted RMSNorm over the last dim: consumes and returns a
/// resident `GpuTensor`, keeping the activation on-device (the DiT norm was a
/// pure-CPU op that round-tripped every call). `out = x/sqrt(mean(x^2)+eps) * w`.
#[allow(dead_code)]
pub(crate) fn rms_norm_resident(
    gpu: &mut hipfire_rdna::Gpu,
    cache: &mut RocmWeightCache,
    input: &hipfire_rdna::GpuTensor,
    weight: &CpuTensor,
    eps: f32,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let width = input.shape.last().copied().ok_or_else(|| {
        DiffusionError::InvalidMetadata("rms_norm input must have at least one dim".to_string())
    })?;
    if weight.shape.as_slice() != [width] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "rms_norm weight shape {:?} does not match width {width}",
            weight.shape
        )));
    }
    let output_elements = checked_shape_elements("rms_norm output", &input.shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &input.shape)?;
    if output_elements == 0 || width == 0 {
        return Ok(output);
    }
    let weight_ptr = cache.resident_ptr(gpu, weight)?;
    let rows = output_elements / width;
    const WAVES_PER_BLOCK: usize = 8;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input.buf.as_ptr());
    kernargs.push_ptr(weight_ptr);
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(i32_kernel_dim("rms_norm rows", rows)?);
    kernargs.push_i32(i32_kernel_dim("rms_norm width", width)?);
    kernargs.push_f32(eps);
    kernargs.pad_to(16);
    let blocks = rows.div_ceil(WAVES_PER_BLOCK);
    let grid = [i32_kernel_dim("rms_norm grid", blocks)? as u32, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_rms_norm_rows_f32",
        DIFFUSION_RMS_NORM_ROWS_HIP_SRC,
        "diffusion_rms_norm_rows_f32",
        grid,
        [(WAVES_PER_BLOCK * 32) as u32, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}

/// Device-resident GeGLU gate over a 3D `[b, seq, width]` projection; output is
/// `[b, seq, width/2]` (`x * gelu(gate)`).
pub(crate) fn geglu_gate_3d_resident(
    gpu: &mut hipfire_rdna::Gpu,
    projected: &hipfire_rdna::GpuTensor,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let [batch, seq, width] = resident_dims3(&projected.shape, "GeGLU projection")?;
    if width % 2 != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "GEGLU projection width {width} is not even"
        )));
    }
    let inner = width / 2;
    let output_shape = [batch, seq, inner];
    let output_elements = checked_shape_elements("GeGLU gate output", &output_shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &output_shape)?;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(projected.buf.as_ptr());
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "GeGLU gate output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("GeGLU gate inner width", inner)?);
    kernargs.push_i32(i32_kernel_dim("GeGLU gate projected width", width)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_geglu_gate_3d_f32",
        DIFFUSION_GEGLU_GATE_HIP_SRC,
        "diffusion_geglu_gate_3d_f32",
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}

/// Device-resident SwiGLU gate from two resident projections: `up * silu(gate)`.
/// Both inputs share shape; the activation stays on-device.
#[allow(dead_code)]
pub(crate) fn swiglu_gate_3d_resident(
    gpu: &mut hipfire_rdna::Gpu,
    up: &hipfire_rdna::GpuTensor,
    gate: &hipfire_rdna::GpuTensor,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    two_input_gate_resident(
        gpu,
        up,
        gate,
        "diffusion_swiglu_gate_f32",
        "SwiGLU",
    )
}

/// Device-resident sigmoid gate from two resident tensors: `value * sigmoid(gate)`
/// (the Krea2 single-stream attention output gate).
#[allow(dead_code)]
pub(crate) fn sigmoid_gate_3d_resident(
    gpu: &mut hipfire_rdna::Gpu,
    value: &hipfire_rdna::GpuTensor,
    gate: &hipfire_rdna::GpuTensor,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    two_input_gate_resident(
        gpu,
        value,
        gate,
        "diffusion_sigmoid_gate_f32",
        "sigmoid gate",
    )
}

#[allow(dead_code)]
fn two_input_gate_resident(
    gpu: &mut hipfire_rdna::Gpu,
    a: &hipfire_rdna::GpuTensor,
    b: &hipfire_rdna::GpuTensor,
    kernel: &str,
    label: &str,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    if a.shape != b.shape {
        return Err(DiffusionError::InvalidMetadata(format!(
            "{label} input shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        )));
    }
    let elements = checked_shape_elements(label, &a.shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &a.shape)?;
    if elements == 0 {
        return Ok(output);
    }
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(a.buf.as_ptr());
    kernargs.push_ptr(b.buf.as_ptr());
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(i32_kernel_dim(label, elements)?);
    kernargs.pad_to(16);
    let grid = [((elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        kernel,
        DIFFUSION_TWO_INPUT_GATE_HIP_SRC,
        kernel,
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}

/// Device-resident per-head RMSNorm for QK-norm: normalizes each head over
/// `head_dim`. Consumes `input` (freeing it) and returns a fresh resident
/// tensor of the same `[batch, seq, heads*head_dim]` shape. `weight` is
/// `[head_dim]`; `None` returns `input` unchanged. Reuses `rms_norm_resident`
/// on a `[batch*seq*heads, head_dim]` view (same buffer, per-head rows).
#[allow(dead_code)]
pub(crate) fn qk_norm_heads_resident(
    gpu: &mut hipfire_rdna::Gpu,
    cache: &mut RocmWeightCache,
    input: hipfire_rdna::GpuTensor,
    weight: Option<&CpuTensor>,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let Some(weight) = weight else {
        return Ok(input);
    };
    let [batch, seq, width] = resident_dims3(&input.shape, "QK-norm input")?;
    if heads == 0 || head_dim == 0 || width != heads * head_dim {
        return Err(DiffusionError::InvalidMetadata(format!(
            "QK-norm width {width} incompatible with heads {heads} head_dim {head_dim}"
        )));
    }
    let rows = batch * seq * heads;
    let byte_len = rows
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| DiffusionError::InvalidMetadata("QK-norm size overflows".to_string()))?;
    let view = hipfire_rdna::GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(input.buf.as_ptr(), byte_len) },
        shape: vec![rows, head_dim],
        dtype: hipfire_rdna::DType::F32,
    };
    let mut normed = rms_norm_resident(gpu, cache, &view, weight, eps)?;
    free_resident(gpu, input)?;
    normed.shape = vec![batch, seq, width];
    Ok(normed)
}

/// Device-resident GQA expand: `[batch, seq, kv_heads*head_dim] ->
/// [batch, seq, heads*head_dim]`, query head `h` reading KV head
/// `h/(heads/kv_heads)`. Identity clone when `heads == kv_heads`.
#[allow(dead_code)]
pub(crate) fn repeat_kv_heads_resident(
    gpu: &mut hipfire_rdna::Gpu,
    input: &hipfire_rdna::GpuTensor,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let [batch, seq, width] = resident_dims3(&input.shape, "GQA expand input")?;
    if kv_heads == 0 || head_dim == 0 || width != kv_heads * head_dim {
        return Err(DiffusionError::InvalidMetadata(format!(
            "GQA expand input width {width} incompatible with kv_heads {kv_heads} head_dim {head_dim}"
        )));
    }
    if heads == 0 || heads % kv_heads != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "GQA expand target heads {heads} not a multiple of kv_heads {kv_heads}"
        )));
    }
    if heads == kv_heads {
        return clone_resident(gpu, input);
    }
    let out_width = heads * head_dim;
    let output_shape = [batch, seq, out_width];
    let total_out = checked_shape_elements("GQA expand output", &output_shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &output_shape)?;
    if total_out == 0 {
        return Ok(output);
    }
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input.buf.as_ptr());
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(i32_kernel_dim("GQA expand total", total_out)?);
    kernargs.push_i32(i32_kernel_dim("GQA expand seq", seq)?);
    kernargs.push_i32(i32_kernel_dim("GQA expand heads", heads)?);
    kernargs.push_i32(i32_kernel_dim("GQA expand kv_heads", kv_heads)?);
    kernargs.push_i32(i32_kernel_dim("GQA expand head_dim", head_dim)?);
    kernargs.pad_to(16);
    let grid = [((total_out as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_repeat_kv_heads_f32",
        DIFFUSION_REPEAT_KV_HIP_SRC,
        "diffusion_repeat_kv_heads_f32",
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}

/// Device-resident adaLN modulate: `out = input*(1+scale) + shift`, with
/// `shift`/`scale` resident `[batch, width]` broadcast over seq.
#[allow(dead_code)]
pub(crate) fn modulate_3d_resident(
    gpu: &mut hipfire_rdna::Gpu,
    input: &hipfire_rdna::GpuTensor,
    shift: &hipfire_rdna::GpuTensor,
    scale: &hipfire_rdna::GpuTensor,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let [batch, seq, width] = resident_dims3(&input.shape, "modulate input")?;
    if shift.shape.as_slice() != [batch, width] || scale.shape.as_slice() != [batch, width] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "modulate shift/scale shapes {:?}/{:?} != [{batch}, {width}]",
            shift.shape, scale.shape
        )));
    }
    let total = checked_shape_elements("modulate output", &input.shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &input.shape)?;
    if total == 0 {
        return Ok(output);
    }
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input.buf.as_ptr());
    kernargs.push_ptr(shift.buf.as_ptr());
    kernargs.push_ptr(scale.buf.as_ptr());
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(i32_kernel_dim("modulate total", total)?);
    kernargs.push_i32(i32_kernel_dim("modulate seq", seq)?);
    kernargs.push_i32(i32_kernel_dim("modulate width", width)?);
    kernargs.pad_to(16);
    let grid = [((total as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_modulate_3d_f32",
        DIFFUSION_ADALN_HIP_SRC,
        "diffusion_modulate_3d_f32",
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}

/// Device-resident gated residual: `out = residual + update*gate`, `gate`
/// resident `[batch, width]` broadcast over seq.
#[allow(dead_code)]
pub(crate) fn gated_residual_3d_resident(
    gpu: &mut hipfire_rdna::Gpu,
    residual: &hipfire_rdna::GpuTensor,
    update: &hipfire_rdna::GpuTensor,
    gate: &hipfire_rdna::GpuTensor,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let [batch, seq, width] = resident_dims3(&residual.shape, "gated residual")?;
    if update.shape != residual.shape || gate.shape.as_slice() != [batch, width] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "gated residual shape mismatch residual/update/gate {:?}/{:?}/{:?}",
            residual.shape, update.shape, gate.shape
        )));
    }
    let total = checked_shape_elements("gated residual output", &residual.shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &residual.shape)?;
    if total == 0 {
        return Ok(output);
    }
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(residual.buf.as_ptr());
    kernargs.push_ptr(update.buf.as_ptr());
    kernargs.push_ptr(gate.buf.as_ptr());
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(i32_kernel_dim("gated residual total", total)?);
    kernargs.push_i32(i32_kernel_dim("gated residual seq", seq)?);
    kernargs.push_i32(i32_kernel_dim("gated residual width", width)?);
    kernargs.pad_to(16);
    let grid = [((total as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_gated_residual_3d_f32",
        DIFFUSION_ADALN_HIP_SRC,
        "diffusion_gated_residual_3d_f32",
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}

/// Device-resident Qwen interleaved RoPE over `[batch, seq, heads*head_dim]`.
/// `cos`/`sin` are `[seq, head_dim/2]` frequency tables (resident-cached by
/// pointer, stable across steps). Keeps the activation on-device.
#[allow(dead_code)]
pub(crate) fn rope_qwen_resident(
    gpu: &mut hipfire_rdna::Gpu,
    cache: &mut RocmWeightCache,
    input: &hipfire_rdna::GpuTensor,
    cos: &CpuTensor,
    sin: &CpuTensor,
    heads: usize,
    head_dim: usize,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let [batch, seq, width] = resident_dims3(&input.shape, "RoPE input")?;
    if heads == 0 || head_dim == 0 || head_dim % 2 != 0 || width != heads * head_dim {
        return Err(DiffusionError::InvalidMetadata(format!(
            "RoPE input width {width} incompatible with heads {heads} head_dim {head_dim}"
        )));
    }
    let freq_width = head_dim / 2;
    if cos.shape.as_slice() != [seq, freq_width] || sin.shape.as_slice() != [seq, freq_width] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "RoPE cos/sin shapes {:?}/{:?} != [{seq}, {freq_width}]",
            cos.shape, sin.shape
        )));
    }
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &input.shape)?;
    let total_pairs = batch * seq * heads * freq_width;
    if total_pairs == 0 {
        return Ok(output);
    }
    let cos_ptr = cache.resident_ptr(gpu, cos)?;
    let sin_ptr = cache.resident_ptr(gpu, sin)?;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input.buf.as_ptr());
    kernargs.push_ptr(cos_ptr);
    kernargs.push_ptr(sin_ptr);
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(i32_kernel_dim("RoPE total pairs", total_pairs)?);
    kernargs.push_i32(i32_kernel_dim("RoPE seq", seq)?);
    kernargs.push_i32(i32_kernel_dim("RoPE heads", heads)?);
    kernargs.push_i32(i32_kernel_dim("RoPE head_dim", head_dim)?);
    kernargs.pad_to(16);
    let grid = [((total_pairs as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_rope_qwen_f32",
        DIFFUSION_ROPE_QWEN_HIP_SRC,
        "diffusion_rope_qwen_f32",
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}

/// Device-resident NCHW channel concatenation ([n,ca,h,w] ++ [n,cb,h,w] ->
/// [n,ca+cb,h,w]).
pub(crate) fn concat_channels_nchw_resident(
    gpu: &mut hipfire_rdna::Gpu,
    a: &hipfire_rdna::GpuTensor,
    b: &hipfire_rdna::GpuTensor,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let [batch, a_channels, height, width] = resident_dims4(&a.shape, "channel concat left")?;
    let [b_batch, b_channels, b_height, b_width] =
        resident_dims4(&b.shape, "channel concat right")?;
    if batch != b_batch || height != b_height || width != b_width {
        return Err(DiffusionError::InvalidMetadata(format!(
            "cannot concatenate NCHW tensors with shapes {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let out_channels = a_channels.checked_add(b_channels).ok_or_else(|| {
        DiffusionError::InvalidMetadata("concat channel count overflows".to_string())
    })?;
    let output_shape = [batch, out_channels, height, width];
    let output_elements = checked_shape_elements("NCHW channel concat output", &output_shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &output_shape)?;
    launch_diffusion_concat_kernel(
        gpu,
        "diffusion_concat_channels_nchw_f32",
        a,
        b,
        &output.buf,
        |kernargs| {
            kernargs.push_i32(i32_kernel_dim("concat left channels", a_channels)?);
            kernargs.push_i32(i32_kernel_dim("concat right channels", b_channels)?);
            kernargs.push_i32(i32_kernel_dim("concat height", height)?);
            kernargs.push_i32(i32_kernel_dim("concat width", width)?);
            Ok(())
        },
        output_elements,
        false,
    )?;
    Ok(output)
}

/// Device-resident QuickGELU (`x * sigmoid(1.702 * x)`).
pub(crate) fn quick_gelu_resident(
    gpu: &mut hipfire_rdna::Gpu,
    input: &hipfire_rdna::GpuTensor,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let elements = checked_shape_elements("QuickGELU input", &input.shape)?;
    let n = i32_kernel_dim("QuickGELU elements", elements)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &input.shape)?;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(input.buf.as_ptr());
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(n);
    kernargs.pad_to(16);
    let grid = [((elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_quick_gelu_f32",
        DIFFUSION_QUICK_GELU_HIP_SRC,
        "diffusion_quick_gelu_f32",
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}

/// Device-resident CLIP causal self-attention over 2D `[seq, hidden]` q/k/v.
pub(crate) fn clip_causal_self_attention_resident(
    gpu: &mut hipfire_rdna::Gpu,
    q: &hipfire_rdna::GpuTensor,
    k: &hipfire_rdna::GpuTensor,
    v: &hipfire_rdna::GpuTensor,
    n_heads: usize,
) -> DiffusionResult<hipfire_rdna::GpuTensor> {
    let (seq, hidden) = match q.shape.as_slice() {
        [s, h] => (*s, *h),
        other => {
            return Err(DiffusionError::InvalidMetadata(format!(
                "CLIP causal attention expected a 2D tensor, got shape {other:?}"
            )))
        }
    };
    if k.shape.as_slice() != [seq, hidden] || v.shape.as_slice() != [seq, hidden] {
        return Err(DiffusionError::InvalidMetadata(format!(
            "CLIP causal attention q/k/v shapes {:?}/{:?}/{:?} are incompatible",
            q.shape, k.shape, v.shape
        )));
    }
    if n_heads == 0 || hidden % n_heads != 0 {
        return Err(DiffusionError::InvalidMetadata(format!(
            "CLIP hidden size {hidden} is not divisible by {n_heads} heads"
        )));
    }
    let head_dim = hidden / n_heads;
    let output_shape = [seq, hidden];
    let output_elements = checked_shape_elements("CLIP causal attention output", &output_shape)?;
    gpu.bind_thread()
        .map_err(|error| DiffusionError::BackendUnavailable(error.to_string()))?;
    let output = alloc_resident_f32(gpu, &output_shape)?;
    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(q.buf.as_ptr());
    kernargs.push_ptr(k.buf.as_ptr());
    kernargs.push_ptr(v.buf.as_ptr());
    kernargs.push_ptr(output.buf.as_ptr());
    kernargs.push_i32(i32_kernel_dim(
        "CLIP causal attention output elements",
        output_elements,
    )?);
    kernargs.push_i32(i32_kernel_dim("CLIP causal attention sequence", seq)?);
    kernargs.push_i32(i32_kernel_dim("CLIP causal attention hidden size", hidden)?);
    kernargs.push_i32(i32_kernel_dim("CLIP causal attention heads", n_heads)?);
    kernargs.push_i32(i32_kernel_dim("CLIP causal attention head dim", head_dim)?);
    kernargs.pad_to(16);
    let grid = [((output_elements as u32).saturating_add(255)) / 256, 1, 1];
    ensure_and_launch_diffusion_kernel(
        gpu,
        "diffusion_clip_causal_attention_f32",
        DIFFUSION_CLIP_CAUSAL_ATTENTION_HIP_SRC,
        "diffusion_clip_causal_attention_f32",
        grid,
        [256, 1, 1],
        0,
        &mut kernargs,
    )?;
    Ok(output)
}
