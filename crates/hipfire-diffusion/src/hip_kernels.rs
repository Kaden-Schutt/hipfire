// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! HIP kernel source strings for the diffusion GPU boundary ops. Compiled only
//! under the `rocm` feature; each constant is the device source for one
//! `*_hip_on_gpu` dispatch in the crate root.

pub(crate) const DIFFUSION_RGB_TENSOR_TO_U8_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_rgb_tensor_to_u8(
    const float* input,
    unsigned char* output,
    int total_pixels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) {
        return;
    }
    int pixels_per_batch = height * width;
    int b = idx / pixels_per_batch;
    int rem = idx - b * pixels_per_batch;
    int y = rem / width;
    int x = rem - y * width;
    for (int c = 0; c < 3; ++c) {
        int input_idx = ((b * 3 + c) * height + y) * width + x;
        float value = input[input_idx] * 0.5f + 0.5f;
        value = fminf(fmaxf(value, 0.0f), 1.0f);
        output[idx * 3 + c] = (unsigned char)floorf(value * 255.0f + 0.5f);
    }
}
"#;

pub(crate) const DIFFUSION_VAE_BOUNDARY_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_rgb_u8_to_vae_nchw_f32(
    const unsigned char* input,
    float* output,
    int total_outputs,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int x = idx % width;
    int t = idx / width;
    int y = t % height;
    t /= height;
    int c = t % 3;
    int b = t / 3;
    int rgb_idx = (b * height * width + y * width + x) * 3 + c;
    output[idx] = ((float)input[rgb_idx]) / 127.5f - 1.0f;
}

extern "C" __global__ void diffusion_vae_moments_to_latents_f32(
    const float* moments,
    float* output,
    int total_outputs,
    int moments_channels,
    int latent_channels,
    int height,
    int width,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int x = idx % width;
    int t = idx / width;
    int y = t % height;
    t /= height;
    int c = t % latent_channels;
    int b = t / latent_channels;
    int moments_idx = ((b * moments_channels + c) * height + y) * width + x;
    output[idx] = moments[moments_idx] * scale;
}
"#;

pub(crate) const DIFFUSION_INPAINT_MASK_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <math.h>

extern "C" __global__ void diffusion_latent_mask_weights_from_rgb_f32(
    const unsigned char* mask,
    float* output,
    int total_outputs,
    int mask_height,
    int mask_width,
    int latent_height,
    int latent_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int latent_pixels = latent_height * latent_width;
    int b = idx / latent_pixels;
    int rem = idx - b * latent_pixels;
    int y = rem / latent_width;
    int x = rem - y * latent_width;
    int source_y = (y * mask_height) / latent_height;
    int source_x = (x * mask_width) / latent_width;
    int max_y = mask_height > 0 ? mask_height - 1 : 0;
    int max_x = mask_width > 0 ? mask_width - 1 : 0;
    source_y = source_y < max_y ? source_y : max_y;
    source_x = source_x < max_x ? source_x : max_x;
    int mask_idx = (b * mask_height * mask_width + source_y * mask_width + source_x) * 3;
    float luma = ((float)mask[mask_idx] + (float)mask[mask_idx + 1] + (float)mask[mask_idx + 2])
        / (3.0f * 255.0f);
    output[idx] = fminf(fmaxf(luma, 0.0f), 1.0f);
}

extern "C" __global__ void diffusion_masked_rgb_for_inpaint_u8(
    const unsigned char* image,
    const unsigned char* mask,
    unsigned char* output,
    int total_pixels
) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= total_pixels) {
        return;
    }
    int idx = pixel * 3;
    float weight = ((float)mask[idx] + (float)mask[idx + 1] + (float)mask[idx + 2])
        / (3.0f * 255.0f);
    float keep = 1.0f - fminf(fmaxf(weight, 0.0f), 1.0f);
    output[idx] = (unsigned char)fminf(fmaxf(floorf((float)image[idx] * keep + 0.5f), 0.0f), 255.0f);
    output[idx + 1] = (unsigned char)fminf(fmaxf(floorf((float)image[idx + 1] * keep + 0.5f), 0.0f), 255.0f);
    output[idx + 2] = (unsigned char)fminf(fmaxf(floorf((float)image[idx + 2] * keep + 0.5f), 0.0f), 255.0f);
}

extern "C" __global__ void diffusion_blend_latents_with_mask_f32(
    const float* generated,
    const float* init,
    const float* mask,
    float* output,
    int total_outputs,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int x = idx % width;
    int t = idx / width;
    int y = t % height;
    t /= height;
    int c = t % channels;
    int b = t / channels;
    int mask_idx = (b * height + y) * width + x;
    float weight = mask[mask_idx];
    output[idx] = init[idx] * (1.0f - weight) + generated[idx] * weight;
}
"#;

pub(crate) const DIFFUSION_EULER_STEP_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <float.h>

extern "C" __global__ void diffusion_euler_step_f32(
    const float* sample,
    const float* model_output,
    float* output,
    int n,
    float sigma,
    float next_sigma,
    int prediction_type
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float s = sample[idx];
    float m = model_output[idx];
    float derivative = m;
    if (fabsf(sigma) > FLT_MIN) {
        if (prediction_type == 1) {
            derivative = (s - m) / sigma;
        } else if (prediction_type == 2) {
            float sigma_sq = sigma * sigma;
            float denom = sigma_sq + 1.0f;
            float pred_original_sample = m * (-sigma / sqrtf(denom)) + s / denom;
            derivative = (s - pred_original_sample) / sigma;
        }
    }
    output[idx] = s + derivative * (next_sigma - sigma);
}
"#;

pub(crate) const DIFFUSION_DENOISE_VECTOR_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_scale_model_input_f32(
    const float* sample,
    float* output,
    int n,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    output[idx] = sample[idx] * scale;
}

extern "C" __global__ void diffusion_cfg_guidance_f32(
    const float* negative_pred,
    const float* positive_pred,
    float* output,
    int n,
    float cfg_scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float negative = negative_pred[idx];
    float positive = positive_pred[idx];
    output[idx] = negative + cfg_scale * (positive - negative);
}

extern "C" __global__ void diffusion_tensor_add_f32(
    const float* a,
    const float* b,
    float* output,
    int n,
    float unused
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    output[idx] = a[idx] + b[idx];
}

extern "C" __global__ void diffusion_center_unet_input_f32(
    const float* sample,
    float* output,
    int n,
    float unused
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    output[idx] = sample[idx] * 2.0f - 1.0f;
}
"#;

pub(crate) const DIFFUSION_TIMESTEP_EMBEDDING_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_timestep_embedding_f32(
    const float* timesteps,
    float* output,
    int total_outputs,
    int dim,
    int half,
    int flip_sin_to_cos,
    float freq_shift
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int col = idx % dim;
    int row = idx / dim;
    if (half <= 0 || col >= half * 2) {
        output[idx] = 0.0f;
        return;
    }
    int frequency_idx = col < half ? col : col - half;
    float denom = fmaxf((float)half - freq_shift, 1.0f);
    float frequency = expf(-logf(10000.0f) * (float)frequency_idx / denom);
    float value = timesteps[row] * frequency;
    if (col < half) {
        output[idx] = flip_sin_to_cos ? cosf(value) : sinf(value);
    } else {
        output[idx] = flip_sin_to_cos ? sinf(value) : cosf(value);
    }
}
"#;

pub(crate) const DIFFUSION_CONV2D_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_conv2d_nchw_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int total_outputs,
    int batch,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int padding,
    int stride,
    int has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int ox = idx % out_w;
    int t = idx / out_w;
    int oy = t % out_h;
    t /= out_h;
    int oc = t % out_channels;
    int b = t / out_channels;

    float acc = has_bias ? bias[oc] : 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ky = 0; ky < kernel_h; ++ky) {
            int iy_with_pad = oy * stride + ky;
            if (iy_with_pad < padding || iy_with_pad >= in_h + padding) {
                continue;
            }
            int iy = iy_with_pad - padding;
            for (int kx = 0; kx < kernel_w; ++kx) {
                int ix_with_pad = ox * stride + kx;
                if (ix_with_pad < padding || ix_with_pad >= in_w + padding) {
                    continue;
                }
                int ix = ix_with_pad - padding;
                int input_idx = ((b * in_channels + ic) * in_h + iy) * in_w + ix;
                int weight_idx = ((oc * in_channels + ic) * kernel_h + ky) * kernel_w + kx;
                acc += input[input_idx] * weight[weight_idx];
            }
        }
    }
    output[idx] = acc;
}
"#;

pub(crate) const DIFFUSION_GROUP_NORM_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_group_norm_nchw_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int total_elements,
    int channels,
    int height,
    int width,
    int groups,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    int t = idx / width;
    t /= height;
    int c = t % channels;
    int b = t / channels;
    int channels_per_group = channels / groups;
    int group = c / channels_per_group;
    int c_start = group * channels_per_group;
    int c_end = c_start + channels_per_group;
    int elems_per_group = channels_per_group * height * width;

    float sum = 0.0f;
    for (int gc = c_start; gc < c_end; ++gc) {
        for (int gy = 0; gy < height; ++gy) {
            for (int gx = 0; gx < width; ++gx) {
                int sample_idx = ((b * channels + gc) * height + gy) * width + gx;
                sum += input[sample_idx];
            }
        }
    }
    float mean = sum / (float)elems_per_group;

    float var_sum = 0.0f;
    for (int gc = c_start; gc < c_end; ++gc) {
        for (int gy = 0; gy < height; ++gy) {
            for (int gx = 0; gx < width; ++gx) {
                int sample_idx = ((b * channels + gc) * height + gy) * width + gx;
                float centered = input[sample_idx] - mean;
                var_sum += centered * centered;
            }
        }
    }
    float inv_std = rsqrtf(var_sum / (float)elems_per_group + eps);
    output[idx] = (input[idx] - mean) * inv_std * weight[c] + bias[c];
}
"#;

pub(crate) const DIFFUSION_SILU_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_silu_f32(
    const float* input,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float value = input[idx];
    output[idx] = value / (1.0f + expf(-value));
}
"#;

pub(crate) const DIFFUSION_QUICK_GELU_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_quick_gelu_f32(
    const float* input,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float value = input[idx];
    output[idx] = value / (1.0f + expf(-1.702f * value));
}
"#;

pub(crate) const DIFFUSION_CLIP_EMBEDDINGS_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_clip_token_position_embedding_f32(
    const float* token_embedding,
    const float* position_embedding,
    const unsigned int* tokens,
    float* output,
    int total_outputs,
    int hidden
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int col = idx % hidden;
    int pos = idx / hidden;
    unsigned int token = tokens[pos];
    output[idx] = token_embedding[token * hidden + col] + position_embedding[pos * hidden + col];
}
"#;

pub(crate) const DIFFUSION_UPSAMPLE_NEAREST2D_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_upsample_nearest2d_nchw_f32(
    const float* input,
    float* output,
    int total_outputs,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int ox = idx % out_w;
    int t = idx / out_w;
    int oy = t % out_h;
    t /= out_h;
    int c = t % channels;
    int b = t / channels;
    int iy = oy / scale;
    int ix = ox / scale;
    int input_idx = ((b * channels + c) * in_h + iy) * in_w + ix;
    output[idx] = input[input_idx];
}
"#;

pub(crate) const DIFFUSION_LAYOUT_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_add_channel_bias_nchw_f32(
    const float* input,
    const float* bias,
    float* output,
    int total_elements,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    int t = idx / width;
    t /= height;
    int c = t % channels;
    int b = t / channels;
    output[idx] = input[idx] + bias[b * channels + c];
}

extern "C" __global__ void diffusion_nchw_to_bsc_f32(
    const float* input,
    float* output,
    int total_elements,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    int x = idx % width;
    int t = idx / width;
    int y = t % height;
    t /= height;
    int c = t % channels;
    int b = t / channels;
    int seq = height * width;
    int s = y * width + x;
    output[(b * seq + s) * channels + c] = input[idx];
}

extern "C" __global__ void diffusion_bsc_to_nchw_f32(
    const float* input,
    float* output,
    int total_elements,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    int x = idx % width;
    int t = idx / width;
    int y = t % height;
    t /= height;
    int c = t % channels;
    int b = t / channels;
    int seq = height * width;
    int s = y * width + x;
    output[idx] = input[(b * seq + s) * channels + c];
}
"#;

pub(crate) const DIFFUSION_CONCAT_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_concat_channels_nchw_f32(
    const float* a,
    const float* b,
    float* output,
    int total_outputs,
    int a_channels,
    int b_channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int out_channels = a_channels + b_channels;
    int x = idx % width;
    int t = idx / width;
    int y = t % height;
    t /= height;
    int c = t % out_channels;
    int batch = t / out_channels;
    if (c < a_channels) {
        output[idx] = a[((batch * a_channels + c) * height + y) * width + x];
    } else {
        int bc = c - a_channels;
        output[idx] = b[((batch * b_channels + bc) * height + y) * width + x];
    }
}

extern "C" __global__ void diffusion_concat_last_dim_f32(
    const float* a,
    const float* b,
    float* output,
    int total_outputs,
    int left_width,
    int right_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int out_width = left_width + right_width;
    int col = idx % out_width;
    int row = idx / out_width;
    if (col < left_width) {
        output[idx] = a[row * left_width + col];
    } else {
        output[idx] = b[row * right_width + (col - left_width)];
    }
}
"#;

pub(crate) const DIFFUSION_LINEAR_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_linear_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int total_outputs,
    int in_features,
    int out_features,
    int has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int out_col = idx % out_features;
    int row = idx / out_features;
    int input_row = row * in_features;
    int weight_row = out_col * in_features;
    float acc = has_bias ? bias[out_col] : 0.0f;
    for (int k = 0; k < in_features; ++k) {
        acc += input[input_row + k] * weight[weight_row + k];
    }
    output[idx] = acc;
}
"#;

pub(crate) const DIFFUSION_LAYER_NORM_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_layer_norm_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int total_outputs,
    int cols,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int col = idx % cols;
    int row = idx / cols;
    int base = row * cols;

    float sum = 0.0f;
    for (int k = 0; k < cols; ++k) {
        sum += input[base + k];
    }
    float mean = sum / (float)cols;

    float var_sum = 0.0f;
    for (int k = 0; k < cols; ++k) {
        float centered = input[base + k] - mean;
        var_sum += centered * centered;
    }
    float inv_std = rsqrtf(var_sum / (float)cols + eps);
    output[idx] = (input[idx] - mean) * inv_std * weight[col] + bias[col];
}
"#;

pub(crate) const DIFFUSION_SOFTMAX_ROWS_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_softmax_rows_f32(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int base = row * cols;
    float max_value = input[base];
    for (int col = 1; col < cols; ++col) {
        max_value = fmaxf(max_value, input[base + col]);
    }

    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
        float value = expf(input[base + col] - max_value);
        output[base + col] = value;
        sum += value;
    }
    if (sum > 0.0f) {
        for (int col = 0; col < cols; ++col) {
            output[base + col] /= sum;
        }
    }
}
"#;

pub(crate) const DIFFUSION_SDPA_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_sdpa_3d_f32(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int total_outputs,
    int q_seq,
    int k_seq,
    int hidden,
    int heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int d = idx % hidden;
    int t = idx / hidden;
    int qi = t % q_seq;
    int b = t / q_seq;
    int head = d / head_dim;
    int head_off = head * head_dim;
    int local_d = d - head_off;
    float scale = rsqrtf((float)head_dim);

    float max_score = -INFINITY;
    for (int ki = 0; ki < k_seq; ++ki) {
        float dot = 0.0f;
        for (int hd = 0; hd < head_dim; ++hd) {
            int q_idx = ((b * q_seq + qi) * hidden) + head_off + hd;
            int k_idx = ((b * k_seq + ki) * hidden) + head_off + hd;
            dot += q[q_idx] * k[k_idx];
        }
        float score = dot * scale;
        max_score = fmaxf(max_score, score);
    }

    float sum = 0.0f;
    float acc = 0.0f;
    for (int ki = 0; ki < k_seq; ++ki) {
        float dot = 0.0f;
        for (int hd = 0; hd < head_dim; ++hd) {
            int q_idx = ((b * q_seq + qi) * hidden) + head_off + hd;
            int k_idx = ((b * k_seq + ki) * hidden) + head_off + hd;
            dot += q[q_idx] * k[k_idx];
        }
        float weight = expf(dot * scale - max_score);
        int v_idx = ((b * k_seq + ki) * hidden) + head_off + local_d;
        acc += weight * v[v_idx];
        sum += weight;
    }
    output[idx] = sum > 0.0f ? acc / sum : 0.0f;
}
"#;

pub(crate) const DIFFUSION_CLIP_CAUSAL_ATTENTION_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_clip_causal_attention_f32(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int total_outputs,
    int seq,
    int hidden,
    int heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int d = idx % hidden;
    int qi = idx / hidden;
    int head = d / head_dim;
    int head_off = head * head_dim;
    int local_d = d - head_off;
    float scale = rsqrtf((float)head_dim);

    float max_score = -INFINITY;
    for (int ki = 0; ki <= qi; ++ki) {
        float dot = 0.0f;
        for (int hd = 0; hd < head_dim; ++hd) {
            dot += q[qi * hidden + head_off + hd] * k[ki * hidden + head_off + hd];
        }
        max_score = fmaxf(max_score, dot * scale);
    }

    float sum = 0.0f;
    float acc = 0.0f;
    for (int ki = 0; ki <= qi; ++ki) {
        float dot = 0.0f;
        for (int hd = 0; hd < head_dim; ++hd) {
            dot += q[qi * hidden + head_off + hd] * k[ki * hidden + head_off + hd];
        }
        float weight = expf(dot * scale - max_score);
        acc += weight * v[ki * hidden + head_off + local_d];
        sum += weight;
    }
    output[idx] = sum > 0.0f ? acc / sum : 0.0f;
}
"#;

pub(crate) const DIFFUSION_GEGLU_GATE_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_geglu_gate_3d_f32(
    const float* input,
    float* output,
    int total_outputs,
    int inner,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int col = idx % inner;
    int row = idx / inner;
    int src = row * width;
    float value = input[src + col];
    float gate_value = input[src + inner + col];
    float gelu_arg = 1.1283791670955126f * (gate_value + 0.044715f * gate_value * gate_value * gate_value);
    float gate = 0.5f * gate_value * (1.0f + tanhf(gelu_arg));
    output[idx] = value * gate;
}
"#;
