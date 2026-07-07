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

extern "C" __global__ void diffusion_scaled_add_f32(
    const float* a,
    const float* b,
    float* output,
    int n,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    output[idx] = a[idx] + scale * b[idx];
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

// Phase 3 — im2col + WMMA-GEMM convolution.
//
// The naive direct conv above is one thread per output element with an
// in_channels*kh*kw inner loop; it is the dominant cost of the VAE decode
// (high-resolution convs). These three kernels feed the matrix-core GEMM
// (`Gpu::gemm_f16_wmma`, the no-LDS register-tiled WMMA kernel) instead:
//   1. im2col lowers the activation into a [B*OH*OW, IC*KH*KW] column matrix,
//   2. the conv weight is reshaped [OC, IC*KH*KW] and cast to F16 (once, cached),
//   3. per batch, gemm computes Y[OC, OH*OW] = W_f16 @ X^T — which is exactly the
//      NCHW output slice for that batch, so no transpose is needed,
//   4. a per-output-channel bias is added.

// Implicit-GEMM convolution: a WMMA GEMM that gathers the im2col columns from
// the NCHW input on the fly instead of materializing a [B*OH*OW, K] column
// matrix. This is the fused replacement for a separate im2col kernel + generic
// gemm_f16_wmma — it removes the column-matrix allocation and its write+read
// memory traffic.
//
// Per batch slice (blockIdx.z = b): Y_b[OC, OH*OW] = W_f16[OC, K] @ X_bᵀ where
// X_b[n, c] is the im2col value for output position n=(oy,ox) and tap
// c=(ic,ky,kx): input[b, ic, oy*stride-pad+ky, ox*stride-pad+kx] (0 if OOB).
// Y_b[OC, OH*OW] lands directly as the NCHW output slice for batch b.
//
// WMMA layout matches gemm_f16_wmma (validated with the AMD matrix calculator):
// A lane t holds row t cols 0..15; D[reg j][lane t] -> (2j+(t>>4), t&15).
pub(crate) const DIFFUSION_CONV2D_IMPLICIT_WMMA_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void diffusion_conv2d_implicit_wmma_f16(
    const _Float16* __restrict__ W,     // [OC, K] f16, K = IC*KH*KW
    const float*    __restrict__ input, // [B, IC, IH, IW] f32
    float*          __restrict__ Y,     // [B, OC, OH*OW] f32
    int OC, int K, int OHW,
    int IC, int IH, int IW, int OH, int OW,
    int KH, int KW, int pad, int stride
) {
    const int row_start = blockIdx.x * 16;   // output channels
    const int col_start = blockIdx.y * 16;   // output positions n
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    if (row_start >= OC || col_start >= OHW) return;

    const _Float16* Wb = W;                              // weights shared across batch
    const float* in_b = input + (long long)b * IC * IH * IW;
    float* Yb = Y + (long long)b * OC * OHW;

    const int my_a_row = row_start + (tid & 15);         // output channel
    const int my_b_row = col_start + (tid & 15);         // output position n
    const bool a_in = (my_a_row < OC);
    const bool b_in = (my_b_row < OHW);
    const int oy = b_in ? (my_b_row / OW) : 0;
    const int ox = b_in ? (my_b_row % OW) : 0;

    float8_t acc = {0,0,0,0,0,0,0,0};

    for (int k0 = 0; k0 < K; k0 += 16) {
        half16_t a;
        if (a_in) {
            const _Float16* src = Wb + (long long)my_a_row * K + k0;
            #pragma unroll
            for (int j = 0; j < 16; j++) a[j] = (k0 + j < K) ? src[j] : (_Float16)0.0f;
        } else {
            #pragma unroll
            for (int j = 0; j < 16; j++) a[j] = (_Float16)0.0f;
        }

        half16_t bb;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            _Float16 val = (_Float16)0.0f;
            int c = k0 + j;
            if (b_in && c < K) {
                int kx = c % KW;
                int t = c / KW;
                int ky = t % KH;
                int ic = t / KH;
                int iy = oy * stride - pad + ky;
                int ix = ox * stride - pad + kx;
                if (iy >= 0 && iy < IH && ix >= 0 && ix < IW) {
                    val = (_Float16)in_b[((long long)ic * IH + iy) * IW + ix];
                }
            }
            bb[j] = val;
        }

        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bb, acc);
    }

    const int out_col = col_start + (tid & 15);
    if (out_col < OHW) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < OC) {
                Yb[(long long)out_row * OHW + out_col] = acc[j];
            }
        }
    }
}
"#;

pub(crate) const DIFFUSION_F32_TO_F16_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void diffusion_f32_to_f16(
    const float* input,
    _Float16* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (_Float16)input[idx];
    }
}
"#;

pub(crate) const DIFFUSION_CONV_BIAS_NCHW_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

// Add a per-output-channel bias to an NCHW output [B, OC, OH*OW].
extern "C" __global__ void diffusion_conv_bias_nchw_f32(
    float* output,
    const float* bias,
    int total,
    int spatial,
    int out_channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int oc = (idx / spatial) % out_channels;
    output[idx] += bias[oc];
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

// Phase 3 — two-pass group-norm.
//
// The single-kernel `diffusion_group_norm_nchw_f32` above recomputes the full
// per-group mean and variance reduction *inside every output thread*, i.e.
// O(N * group_size) ~ O((H*W)^2) per call. At VAE-decode resolutions that
// dominates wall-clock. These two kernels split it into an O(N) reduction
// (one wave per (batch, group), pure register + wave-shuffle, no LDS — so it
// is wedge-safe on gfx1103) followed by an O(N) elementwise apply.

// Phase 3 — per-row bias for the WMMA linear. The WMMA GEMM writes
// Y[rows, out_features] row-major with no bias; this adds bias[out] to every row.
pub(crate) const DIFFUSION_ROW_BIAS_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_row_bias_f32(
    float* output,
    const float* bias,
    int total,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    output[idx] += bias[idx % out_features];
}
"#;

pub(crate) const DIFFUSION_GROUP_NORM_STATS_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

// One wave (32 lanes) per (batch, group). A group's channels are contiguous in
// NCHW within a batch, so its elements are one contiguous span. Two passes over
// that span (mean, then sum of squared deviations) match the CPU reference
// formula and avoid catastrophic cancellation.
extern "C" __global__ void diffusion_group_norm_stats_f32(
    const float* input,
    float* mean_out,
    float* inv_std_out,
    int channels,
    int height,
    int width,
    int groups,
    float eps
) {
    int bg = blockIdx.x;                 // 0 .. batch*groups - 1
    int group = bg % groups;
    int b = bg / groups;
    int cpg = channels / groups;
    long hw = (long)height * (long)width;
    long group_size = (long)cpg * hw;
    long base = ((long)b * channels + (long)group * cpg) * hw;
    int lane = threadIdx.x;              // 0 .. 31

    float s = 0.0f;
    for (long i = lane; i < group_size; i += 32) {
        s += input[base + i];
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down(s, off);
    }
    float total = __shfl(s, 0);
    float mean = total / (float)group_size;

    float vs = 0.0f;
    for (long i = lane; i < group_size; i += 32) {
        float d = input[base + i] - mean;
        vs += d * d;
    }
    for (int off = 16; off > 0; off >>= 1) {
        vs += __shfl_down(vs, off);
    }
    if (lane == 0) {
        mean_out[bg] = mean;
        inv_std_out[bg] = rsqrtf(vs / (float)group_size + eps);
    }
}
"#;

pub(crate) const DIFFUSION_GROUP_NORM_APPLY_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_group_norm_apply_f32(
    const float* input,
    const float* mean_in,
    const float* inv_std_in,
    const float* weight,
    const float* bias,
    float* output,
    int total_elements,
    int channels,
    int height,
    int width,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    long hw = (long)height * (long)width;
    int c = (int)((idx / hw) % channels);
    int b = (int)(idx / ((long)channels * hw));
    int cpg = channels / groups;
    int group = c / cpg;
    int bg = b * groups + group;
    float mean = mean_in[bg];
    float inv_std = inv_std_in[bg];
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

pub(crate) const DIFFUSION_LEAKY_RELU_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_leaky_relu_f32(
    const float* input,
    float* output,
    int n,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float value = input[idx];
    output[idx] = value >= 0.0f ? value : alpha * value;
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

pub(crate) const DIFFUSION_PIXEL_UNSHUFFLE_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

// Space-to-depth by `scale` (inverse of pixel-shuffle), NCHW. Matches
// PyTorch/basicsr pixel_unshuffle: for an r=scale block, output channel
// c_out = c * r*r + dy * r + dx gathers input[n, c, oh*r + dy, ow*r + dx].
// Used by the RealESRGAN x2 (RRDBNet) input stage.
extern "C" __global__ void diffusion_pixel_unshuffle_nchw_f32(
    const float* input,
    float* output,
    int total_outputs,
    int out_channels,
    int out_h,
    int out_w,
    int scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }
    int ow = idx % out_w;
    int t = idx / out_w;
    int oh = t % out_h;
    t /= out_h;
    int c_out = t % out_channels;
    int b = t / out_channels;

    int rr = scale * scale;
    int in_channels = out_channels / rr;
    int c = c_out / rr;
    int rem = c_out - c * rr;
    int dy = rem / scale;
    int dx = rem - dy * scale;

    int in_h = out_h * scale;
    int in_w = out_w * scale;
    int ih = oh * scale + dy;
    int iw = ow * scale + dx;
    int input_idx = ((b * in_channels + c) * in_h + ih) * in_w + iw;
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

// Phase 3 — wave-per-row layer-norm. The naive kernel below recomputes the full
// per-row mean+variance reduction inside every output thread (O(rows*cols^2)).
// This assigns one wave (32 lanes) per row: lanes stride over cols, reduce via
// __shfl for mean then variance, and write the row's outputs — O(rows*cols), no
// LDS (wedge-safe on gfx1103). Used by the resident path; the naive kernel stays
// for the preflight probe.
pub(crate) const DIFFUSION_LAYER_NORM_ROWS_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void diffusion_layer_norm_rows_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int rows,
    int cols,
    float eps
) {
    int lane = threadIdx.x & 31;
    int wave = threadIdx.x >> 5;
    int row = blockIdx.x * (blockDim.x >> 5) + wave;
    if (row >= rows) {
        return;
    }
    long base = (long)row * cols;

    float s = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        s += input[base + c];
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down(s, off);
    }
    float mean = __shfl(s, 0) / (float)cols;

    float vs = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        float d = input[base + c] - mean;
        vs += d * d;
    }
    for (int off = 16; off > 0; off >>= 1) {
        vs += __shfl_down(vs, off);
    }
    float inv_std = rsqrtf(__shfl(vs, 0) / (float)cols + eps);

    for (int c = lane; c < cols; c += 32) {
        output[base + c] = (input[base + c] - mean) * inv_std * weight[c] + bias[c];
    }
}
"#;

pub(crate) const DIFFUSION_RMS_NORM_ROWS_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

// Weighted RMSNorm, wave-per-row, single reduction pass, no LDS (gfx1103-safe).
// out[c] = x[c] / sqrt(mean(x^2) + eps) * weight[c]. Matches the CPU
// rms_norm_3d reference exactly.
extern "C" __global__ void diffusion_rms_norm_rows_f32(
    const float* input,
    const float* weight,
    float* output,
    int rows,
    int cols,
    float eps
) {
    int lane = threadIdx.x & 31;
    int wave = threadIdx.x >> 5;
    int row = blockIdx.x * (blockDim.x >> 5) + wave;
    if (row >= rows) {
        return;
    }
    long base = (long)row * cols;

    float ss = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        float v = input[base + c];
        ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_down(ss, off);
    }
    float inv_rms = rsqrtf(__shfl(ss, 0) / (float)cols + eps);

    for (int c = lane; c < cols; c += 32) {
        output[base + c] = input[base + c] * inv_rms * weight[c];
    }
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

// Phase 3 — flash-style attention (online softmax, no seq×seq materialization).
//
// The naive `diffusion_sdpa_3d_f32` below is one thread per output element and
// recomputes the full QKᵀ score row once PER output channel AND twice (a max
// pass then a sum pass) — ~2·head_dim× redundant. This kernel assigns one wave
// (32 lanes) to each (batch, head, query): the lanes split head_dim, compute
// each score once via a wave-shuffle dot reduction, and stream keys with an
// online-softmax running (max, sum, accumulator). No LDS (pure registers +
// `__shfl` → wedge-safe on gfx1103), F32 throughout (matches the reference
// closely). head_dim is capped at 16 channels/lane = 512; the host falls back
// to the naive kernel above that.
pub(crate) const DIFFUSION_FLASH_ATTENTION_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

#define FLASH_ACC_MAX 16

extern "C" __global__ void diffusion_flash_attention_f32(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch,
    int q_seq,
    int k_seq,
    int hidden,
    int heads,
    int head_dim
) {
    int lane = threadIdx.x & 31;
    int wave = threadIdx.x >> 5;
    int waves_per_block = blockDim.x >> 5;
    int gid = blockIdx.x * waves_per_block + wave;
    int total = batch * heads * q_seq;
    if (gid >= total) {
        return;
    }
    int qi = gid % q_seq;
    int t = gid / q_seq;
    int head = t % heads;
    int b = t / heads;
    int head_off = head * head_dim;
    float scale = rsqrtf((float)head_dim);
    long qbase = ((long)(b * q_seq + qi) * hidden) + head_off;

    float qreg[FLASH_ACC_MAX];
    float acc[FLASH_ACC_MAX];
    int np = 0;
    for (int c = lane; c < head_dim; c += 32) {
        qreg[np] = q[qbase + c];
        acc[np] = 0.0f;
        np++;
    }

    float m = -INFINITY;
    float l = 0.0f;
    for (int ki = 0; ki < k_seq; ++ki) {
        long kbase = ((long)(b * k_seq + ki) * hidden) + head_off;
        float part = 0.0f;
        int ii = 0;
        for (int c = lane; c < head_dim; c += 32) {
            part += qreg[ii] * k[kbase + c];
            ii++;
        }
        for (int off = 16; off > 0; off >>= 1) {
            part += __shfl_down(part, off);
        }
        float s = __shfl(part, 0) * scale;
        float new_m = fmaxf(m, s);
        float corr = expf(m - new_m);
        float p = expf(s - new_m);
        l = l * corr + p;
        long vbase = ((long)(b * k_seq + ki) * hidden) + head_off;
        ii = 0;
        for (int c = lane; c < head_dim; c += 32) {
            acc[ii] = acc[ii] * corr + p * v[vbase + c];
            ii++;
        }
        m = new_m;
    }
    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    int ii = 0;
    for (int c = lane; c < head_dim; c += 32) {
        output[qbase + c] = acc[ii] * inv;
        ii++;
    }
}
"#;

// Register-tiled flash attention: each wave owns a tile of FLASH_Q_TILE queries
// of one (batch, head) and streams K/V once for the whole tile (each K/V element
// loaded once and reused across all FLASH_Q_TILE queries), instead of one wave
// per query re-reading all of K/V. Cuts the dominant redundant K/V traffic by
// FLASH_Q_TILE and runs FLASH_Q_TILE independent online-softmax chains for ILP.
// Zero LDS. Same online-softmax math and output layout as the 1-query kernel.
//
// FLASH_NP (channels/lane) is compile-time so the per-query register arrays are
// statically indexed and stay in VGPRs (a runtime channel count spills them to
// scratch and erases the win). FLASH_NP=4 ⇒ head_dim=128; the dispatch only
// selects this kernel when head_dim == FLASH_NP*32.
pub(crate) const DIFFUSION_FLASH_ATTENTION_QTILE_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

#define FLASH_NP 4          // channels per lane (head_dim = FLASH_NP * 32 = 128)
#define FLASH_Q_TILE 8

extern "C" __global__ void diffusion_flash_attention_qtile_f32(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch,
    int q_seq,
    int k_seq,
    int hidden,
    int heads,
    int head_dim
) {
    int lane = threadIdx.x & 31;
    int wave = threadIdx.x >> 5;
    int waves_per_block = blockDim.x >> 5;
    int gid = blockIdx.x * waves_per_block + wave;
    int q_tiles = (q_seq + FLASH_Q_TILE - 1) / FLASH_Q_TILE;
    int total = batch * heads * q_tiles;
    if (gid >= total) {
        return;
    }
    int qtile = gid % q_tiles;
    int t = gid / q_tiles;
    int head = t % heads;
    int b = t / heads;
    int head_off = head * head_dim;
    int q0 = qtile * FLASH_Q_TILE;
    float scale = rsqrtf((float)head_dim);

    float qreg[FLASH_Q_TILE][FLASH_NP];
    float acc[FLASH_Q_TILE][FLASH_NP];
    float m[FLASH_Q_TILE];
    float l[FLASH_Q_TILE];
    #pragma unroll
    for (int qt = 0; qt < FLASH_Q_TILE; ++qt) {
        int qi = q0 + qt;
        int qq = (qi < q_seq) ? qi : (q_seq - 1);
        long qbase = ((long)(b * q_seq + qq) * hidden) + head_off + lane;
        #pragma unroll
        for (int j = 0; j < FLASH_NP; ++j) {
            qreg[qt][j] = q[qbase + j * 32];
            acc[qt][j] = 0.0f;
        }
        m[qt] = -INFINITY;
        l[qt] = 0.0f;
    }

    for (int ki = 0; ki < k_seq; ++ki) {
        long kbase = ((long)(b * k_seq + ki) * hidden) + head_off + lane;
        float kreg[FLASH_NP];
        float vreg[FLASH_NP];
        #pragma unroll
        for (int j = 0; j < FLASH_NP; ++j) {
            kreg[j] = k[kbase + j * 32];
            vreg[j] = v[kbase + j * 32];
        }
        #pragma unroll
        for (int qt = 0; qt < FLASH_Q_TILE; ++qt) {
            float part = 0.0f;
            #pragma unroll
            for (int j = 0; j < FLASH_NP; ++j) part += qreg[qt][j] * kreg[j];
            for (int off = 16; off > 0; off >>= 1) part += __shfl_down(part, off);
            float s = __shfl(part, 0) * scale;
            float new_m = fmaxf(m[qt], s);
            float corr = expf(m[qt] - new_m);
            float p = expf(s - new_m);
            l[qt] = l[qt] * corr + p;
            #pragma unroll
            for (int j = 0; j < FLASH_NP; ++j) acc[qt][j] = acc[qt][j] * corr + p * vreg[j];
            m[qt] = new_m;
        }
    }
    #pragma unroll
    for (int qt = 0; qt < FLASH_Q_TILE; ++qt) {
        int qi = q0 + qt;
        if (qi >= q_seq) continue;
        long qbase = ((long)(b * q_seq + qi) * hidden) + head_off + lane;
        float inv = (l[qt] > 0.0f) ? (1.0f / l[qt]) : 0.0f;
        #pragma unroll
        for (int j = 0; j < FLASH_NP; ++j) output[qbase + j * 32] = acc[qt][j] * inv;
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

pub(crate) const DIFFUSION_TWO_INPUT_GATE_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

// SwiGLU gate from two separate projections: out = up * silu(gate).
extern "C" __global__ void diffusion_swiglu_gate_f32(
    const float* up,
    const float* gate,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float g = gate[idx];
    output[idx] = up[idx] * (g / (1.0f + expf(-g)));
}

// Sigmoid gate: out = value * sigmoid(gate).
extern "C" __global__ void diffusion_sigmoid_gate_f32(
    const float* value,
    const float* gate,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    output[idx] = value[idx] * (1.0f / (1.0f + expf(-gate[idx])));
}
"#;

pub(crate) const DIFFUSION_ROPE_QWEN_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

// Qwen interleaved RoPE: one thread per (b, token, head, pair). Rotates the
// (real=2p, imag=2p+1) pair in each head by cos/sin[token, pair]. Matches the
// CPU apply_qwen_rotary_embedding. cos/sin are [seq, head_dim/2].
extern "C" __global__ void diffusion_rope_qwen_f32(
    const float* input,
    const float* cos_tab,
    const float* sin_tab,
    float* output,
    int total_pairs,
    int seq,
    int heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) {
        return;
    }
    int freq_width = head_dim >> 1;
    int width = heads * head_dim;

    int pair = idx % freq_width;
    int t = idx / freq_width;
    int head = t % heads;
    int t2 = t / heads;
    int token = t2 % seq;
    int b = t2 / seq;

    long token_base = ((long)(b * seq + token)) * width + (long)head * head_dim;
    long real_idx = token_base + (long)pair * 2;
    long imag_idx = real_idx + 1;
    long freq_idx = (long)token * freq_width + pair;

    float real = input[real_idx];
    float imag = input[imag_idx];
    float c = cos_tab[freq_idx];
    float s = sin_tab[freq_idx];
    output[real_idx] = real * c - imag * s;
    output[imag_idx] = real * s + imag * c;
}
"#;

pub(crate) const DIFFUSION_ADALN_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

// adaLN modulate: out[b,s,c] = in[b,s,c]*(1+scale[b,c]) + shift[b,c]. scale/shift
// are [batch, width], broadcast over seq.
extern "C" __global__ void diffusion_modulate_3d_f32(
    const float* input,
    const float* shift,
    const float* scale,
    float* output,
    int total,
    int seq,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int c = idx % width;
    int token = idx / width;
    int b = token / seq;
    int mod_base = b * width + c;
    output[idx] = input[idx] * (1.0f + scale[mod_base]) + shift[mod_base];
}

// Gated residual: out[b,s,c] = residual[b,s,c] + update[b,s,c]*gate[b,c].
extern "C" __global__ void diffusion_gated_residual_3d_f32(
    const float* residual,
    const float* update,
    const float* gate,
    float* output,
    int total,
    int seq,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int c = idx % width;
    int token = idx / width;
    int b = token / seq;
    output[idx] = residual[idx] + update[idx] * gate[b * width + c];
}
"#;

pub(crate) const DIFFUSION_REPEAT_KV_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

// GQA expand: [b, s, kv_heads*head_dim] -> [b, s, heads*head_dim]. Query head h
// reads KV head h/(heads/kv_heads) (PyTorch repeat_kv ordering). One thread per
// output element.
extern "C" __global__ void diffusion_repeat_kv_heads_f32(
    const float* input,
    float* output,
    int total_out,
    int seq,
    int heads,
    int kv_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) {
        return;
    }
    int in_width = kv_heads * head_dim;
    int dim = idx % head_dim;
    int t = idx / head_dim;
    int head = t % heads;
    int t2 = t / heads;
    int token = t2 % seq;
    int b = t2 / seq;

    int group = heads / kv_heads;
    int kv_head = head / group;
    long src = ((long)(b * seq + token)) * in_width + (long)kv_head * head_dim + dim;
    output[idx] = input[src];
}
"#;
