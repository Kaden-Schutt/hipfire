#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// PARO4-G128 v1, native Paro/AWQ decode kernel.
//
// Tensor byte layout preserves the z-lab/paroquant runtime buffers:
//   qweight:        int32 [K, M/8]
//   qzeros:         int32 [K/128, M/8]
//   scales:         f16   [K/128, M]
//   pairs:          int16 [8, K]
//   theta:          f16   [8, K/2]
//   channel_scales: f16   [K]
//
// Runtime math mirrors RotateQuantizedLinear:
//   x_rot = rotation.rotate(x, pairs, theta, channel_scales)
//   y = awq_gemm(x_rot, qweight, qzeros, scales)
//
// This is a correctness anchor for the native Paro producer-consumer contract.
// One block computes the eight output rows packed into one AWQ int32 column,
// so the expensive Paro activation rotation is shared across those rows.

static __device__ __forceinline__ float load_f16_as_f32(const _Float16* p) {
    return (float)(*p);
}
static __device__ __forceinline__ int awq_slot_for_output(int out_lane) {
    // AutoAWQ packs logical lanes through [0,2,4,6,1,3,5,7].
    // Inverse lookup: logical lane -> physical 4-bit slot in the int32 word.
    constexpr int inv[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    return inv[out_lane & 7];
}

static __device__ __forceinline__ void paro4g128_offsets(
    int M,
    int K,
    long long* pairs_off,
    long long* theta_off,
    long long* channel_scales_off
) {
    constexpr int GROUP_SIZE = 128;
    constexpr int KROT = 8;
    constexpr int PACK = 8;
    const int groups = K / GROUP_SIZE;
    const int m_pack = M / PACK;
    const long long qweight_off = 0;
    const long long qzeros_off = qweight_off + (long long)K * m_pack * (long long)sizeof(int);
    const long long scales_off = qzeros_off + (long long)groups * m_pack * (long long)sizeof(int);
    *pairs_off = scales_off + (long long)groups * M * (long long)sizeof(_Float16);
    *theta_off = *pairs_off + (long long)KROT * K * (long long)sizeof(short);
    *channel_scales_off = *theta_off + (long long)KROT * (K / 2) * 2LL * (long long)sizeof(float);
}

static __device__ __forceinline__ void paro4g128_apply_pair(
    float* __restrict__ xg,
    float s,
    float c,
    int i,
    int j
) {
    const float xi = xg[i];
    const float xj = xg[j];
    xg[i] = fmaf(c, xi, s * xj);
    xg[j] = fmaf(c, xj, -s * xi);
}

template <bool RESIDUAL, bool SWIGLU, bool PRECOMPUTED_TRIG, bool QWEIGHT_TILED>
static __device__ void gemv_paro4g128_impl(
    const char* __restrict__ A,
    const float* __restrict__ x,
    const float* __restrict__ up,
    float* __restrict__ y,
    int M, int K
) {
    const int m_col = blockIdx.x;
    const int tid = threadIdx.x;

    constexpr int GROUP_SIZE = 128;
    constexpr int KROT = 8;
    constexpr int PACK = 8;
    if (m_col * PACK >= M) return;

    const int groups = K / GROUP_SIZE;
    const int m_pack = M / PACK;

    const long long qweight_off = 0;
    const long long qzeros_off = qweight_off + (long long)K * m_pack * (long long)sizeof(int);
    const long long scales_off = qzeros_off + (long long)groups * m_pack * (long long)sizeof(int);
    const long long pairs_off = scales_off + (long long)groups * M * (long long)sizeof(_Float16);
    const long long theta_elems = (long long)KROT * (K / 2) * (PRECOMPUTED_TRIG ? 2LL : 1LL);
    const long long theta_elem_bytes = PRECOMPUTED_TRIG ? (long long)sizeof(float) : (long long)sizeof(_Float16);
    const long long theta_off = pairs_off + (long long)KROT * K * (long long)sizeof(short);
    const long long channel_scales_off = theta_off + theta_elems * theta_elem_bytes;

    const int* __restrict__ qweight = (const int*)(A + qweight_off);
    const int* __restrict__ qzeros = (const int*)(A + qzeros_off);
    const _Float16* __restrict__ scales = (const _Float16*)(A + scales_off);
    const short* __restrict__ pairs = (const short*)(A + pairs_off);
    const _Float16* __restrict__ theta = (const _Float16*)(A + theta_off);
    const _Float16* __restrict__ channel_scales = (const _Float16*)(A + channel_scales_off);

    __shared__ float xg[GROUP_SIZE];
    float acc[PACK] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups; ++g) {
        const int base = g * GROUP_SIZE;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int lane = tid * 4 + i;
            float xv = x[base + lane];
            if constexpr (SWIGLU) {
                xv = (xv / (1.0f + expf(-xv))) * up[base + lane];
            }
            xg[lane] = xv * load_f16_as_f32(channel_scales + base + lane);
        }
        __syncthreads();

        #pragma unroll
        for (int r = 0; r < KROT; ++r) {
            const int pair_base = r * K + base;
            const int theta_base = r * (K / 2) + g * (GROUP_SIZE / 2);

            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int pair_idx = tid + p * 32;
                const int i = (int)pairs[pair_base + 2 * pair_idx];
                const int j = (int)pairs[pair_base + 2 * pair_idx + 1];
                float s, c;
                if constexpr (PRECOMPUTED_TRIG) {
                    const float* __restrict__ trig = (const float*)(A + theta_off);
                    const int trig_idx = (theta_base + pair_idx) * 2;
                    s = trig[trig_idx];
                    c = trig[trig_idx + 1];
                } else {
                    const float th = load_f16_as_f32(theta + theta_base + pair_idx);
                    __sincosf(th, &s, &c);
                }
                const float xi = xg[i];
                const float xj = xg[j];
                xg[i] = fmaf(c, xi, s * xj);
                xg[j] = fmaf(c, xj, -s * xi);
            }
            __syncthreads();
        }

        const int zp_word = qzeros[g * m_pack + m_col];
        float scale[PACK];
        float bias[PACK];
        #pragma unroll
        for (int out = 0; out < PACK; ++out) {
            const int row = m_col * PACK + out;
            const int slot = awq_slot_for_output(out);
            const int zp = (zp_word >> (4 * slot)) & 0xF;
            scale[out] = load_f16_as_f32(scales + (long long)g * M + row);
            bias[out] = -scale[out] * (float)zp;
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int lane = tid * 4 + i;
            const int k_idx = base + lane;
            const int qw_word = QWEIGHT_TILED
                ? qweight[(long long)m_col * K + k_idx]
                : qweight[(long long)k_idx * m_pack + m_col];
            const float xv = xg[lane];
            #pragma unroll
            for (int out = 0; out < PACK; ++out) {
                const int slot = awq_slot_for_output(out);
                const int q = (qw_word >> (4 * slot)) & 0xF;
                acc[out] += (scale[out] * (float)q + bias[out]) * xv;
            }
        }
        __syncthreads();
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int out = 0; out < PACK; ++out) {
            acc[out] += __shfl_down(acc[out], offset);
        }
    }
    if (tid == 0) {
        #pragma unroll
        for (int out = 0; out < PACK; ++out) {
            const int row = m_col * PACK + out;
            if constexpr (RESIDUAL) {
                y[row] += acc[out];
            } else {
                y[row] = acc[out];
            }
        }
    }
}

__launch_bounds__(32, 8)
extern "C" __global__ void gemv_paro4g128(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_paro4g128_impl<false, false, false, false>(A, x, nullptr, y, M, K);
}

__launch_bounds__(32, 8)
extern "C" __global__ void gemv_paro4g128_residual(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_paro4g128_impl<true, false, false, false>(A, x, nullptr, y, M, K);
}

template <bool SWIGLU, bool PRECOMPUTED_TRIG>
static __device__ void paro4g128_rotate_impl(
    const char* __restrict__ A,
    const float* __restrict__ x,
    const float* __restrict__ up,
    float* __restrict__ x_rot,
    int M, int K
) {
    const int g = blockIdx.x;
    const int tid = threadIdx.x;

    constexpr int GROUP_SIZE = 128;
    constexpr int KROT = 8;
    constexpr int PACK = 8;

    const int groups = K / GROUP_SIZE;
    if (g >= groups) return;
    const int m_pack = M / PACK;

    const long long qweight_off = 0;
    const long long qzeros_off = qweight_off + (long long)K * m_pack * (long long)sizeof(int);
    const long long scales_off = qzeros_off + (long long)groups * m_pack * (long long)sizeof(int);
    const long long pairs_off = scales_off + (long long)groups * M * (long long)sizeof(_Float16);
    const long long theta_elems = (long long)KROT * (K / 2) * (PRECOMPUTED_TRIG ? 2LL : 1LL);
    const long long theta_elem_bytes = PRECOMPUTED_TRIG ? (long long)sizeof(float) : (long long)sizeof(_Float16);
    const long long theta_off = pairs_off + (long long)KROT * K * (long long)sizeof(short);
    const long long channel_scales_off = theta_off + theta_elems * theta_elem_bytes;

    const short* __restrict__ pairs = (const short*)(A + pairs_off);
    const _Float16* __restrict__ theta = (const _Float16*)(A + theta_off);
    const _Float16* __restrict__ channel_scales = (const _Float16*)(A + channel_scales_off);

    __shared__ float xg[GROUP_SIZE];
    const int base = g * GROUP_SIZE;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int lane = tid * 4 + i;
        float xv = x[base + lane];
        if constexpr (SWIGLU) {
            xv = (xv / (1.0f + expf(-xv))) * up[base + lane];
        }
        xg[lane] = xv * load_f16_as_f32(channel_scales + base + lane);
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < KROT; ++r) {
        const int pair_base = r * K + base;
        const int theta_base = r * (K / 2) + g * (GROUP_SIZE / 2);

        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int pair_idx = tid + p * 32;
            const int i = (int)pairs[pair_base + 2 * pair_idx];
            const int j = (int)pairs[pair_base + 2 * pair_idx + 1];
            float s, c;
            if constexpr (PRECOMPUTED_TRIG) {
                const float* __restrict__ trig = (const float*)(A + theta_off);
                const int trig_idx = (theta_base + pair_idx) * 2;
                s = trig[trig_idx];
                c = trig[trig_idx + 1];
            } else {
                const float th = load_f16_as_f32(theta + theta_base + pair_idx);
                __sincosf(th, &s, &c);
            }
            const float xi = xg[i];
            const float xj = xg[j];
            xg[i] = fmaf(c, xi, s * xj);
            xg[j] = fmaf(c, xj, -s * xi);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int lane = tid * 4 + i;
        x_rot[base + lane] = xg[lane];
    }
}

template <bool RESIDUAL, bool QWEIGHT_TILED>
static __device__ void gemv_paro4g128_prerotated_impl(
    const char* __restrict__ A,
    const float* __restrict__ x_rot,
    float* __restrict__ y,
    int M, int K
) {
    const int m_col = blockIdx.x;
    const int tid = threadIdx.x;

    constexpr int GROUP_SIZE = 128;
    constexpr int KROT = 8;
    constexpr int PACK = 8;
    if (m_col * PACK >= M) return;

    const int groups = K / GROUP_SIZE;
    const int m_pack = M / PACK;

    const long long qweight_off = 0;
    const long long qzeros_off = qweight_off + (long long)K * m_pack * (long long)sizeof(int);
    const long long scales_off = qzeros_off + (long long)groups * m_pack * (long long)sizeof(int);

    const int* __restrict__ qweight = (const int*)(A + qweight_off);
    const int* __restrict__ qzeros = (const int*)(A + qzeros_off);
    const _Float16* __restrict__ scales = (const _Float16*)(A + scales_off);

    float acc[PACK] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups; ++g) {
        const int base = g * GROUP_SIZE;
        const int zp_word = qzeros[g * m_pack + m_col];
        float scale[PACK];
        float bias[PACK];
        #pragma unroll
        for (int out = 0; out < PACK; ++out) {
            const int row = m_col * PACK + out;
            const int slot = awq_slot_for_output(out);
            const int zp = (zp_word >> (4 * slot)) & 0xF;
            scale[out] = load_f16_as_f32(scales + (long long)g * M + row);
            bias[out] = -scale[out] * (float)zp;
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int lane = tid * 4 + i;
            const int k_idx = base + lane;
            const int qw_word = QWEIGHT_TILED
                ? qweight[(long long)m_col * K + k_idx]
                : qweight[(long long)k_idx * m_pack + m_col];
            const float xv = x_rot[k_idx];
            #pragma unroll
            for (int out = 0; out < PACK; ++out) {
                const int slot = awq_slot_for_output(out);
                const int q = (qw_word >> (4 * slot)) & 0xF;
                acc[out] += (scale[out] * (float)q + bias[out]) * xv;
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int out = 0; out < PACK; ++out) {
            acc[out] += __shfl_down(acc[out], offset);
        }
    }
    if (tid == 0) {
        #pragma unroll
        for (int out = 0; out < PACK; ++out) {
            const int row = m_col * PACK + out;
            if constexpr (RESIDUAL) {
                y[row] += acc[out];
            } else {
                y[row] = acc[out];
            }
        }
    }
}

template <bool RESIDUAL, bool QWEIGHT_TILED>
static __device__ void gemv_paro4g128_prerotated_pack4_impl(
    const char* __restrict__ A,
    const float* __restrict__ x_rot,
    float* __restrict__ y,
    int M, int K
) {
    const int pack4_col = blockIdx.x;
    const int tid = threadIdx.x;

    constexpr int GROUP_SIZE = 128;
    constexpr int KROT = 8;
    constexpr int PACK = 8;
    constexpr int OUT = 4;
    if (pack4_col * OUT >= M) return;

    const int m_col = pack4_col >> 1;
    const int out_base = (pack4_col & 1) * OUT;
    const int groups = K / GROUP_SIZE;
    const int m_pack = M / PACK;

    const long long qweight_off = 0;
    const long long qzeros_off = qweight_off + (long long)K * m_pack * (long long)sizeof(int);
    const long long scales_off = qzeros_off + (long long)groups * m_pack * (long long)sizeof(int);

    const int* __restrict__ qweight = (const int*)(A + qweight_off);
    const int* __restrict__ qzeros = (const int*)(A + qzeros_off);
    const _Float16* __restrict__ scales = (const _Float16*)(A + scales_off);

    float acc[OUT] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups; ++g) {
        const int base = g * GROUP_SIZE;
        const int zp_word = qzeros[g * m_pack + m_col];
        float scale[OUT];
        float bias[OUT];
        #pragma unroll
        for (int out = 0; out < OUT; ++out) {
            const int row = m_col * PACK + out_base + out;
            const int slot = awq_slot_for_output(out_base + out);
            const int zp = (zp_word >> (4 * slot)) & 0xF;
            scale[out] = load_f16_as_f32(scales + (long long)g * M + row);
            bias[out] = -scale[out] * (float)zp;
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int lane = tid * 4 + i;
            const int k_idx = base + lane;
            const int qw_word = QWEIGHT_TILED
                ? qweight[(long long)m_col * K + k_idx]
                : qweight[(long long)k_idx * m_pack + m_col];
            const float xv = x_rot[k_idx];
            #pragma unroll
            for (int out = 0; out < OUT; ++out) {
                const int slot = awq_slot_for_output(out_base + out);
                const int q = (qw_word >> (4 * slot)) & 0xF;
                acc[out] += (scale[out] * (float)q + bias[out]) * xv;
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int out = 0; out < OUT; ++out) {
            acc[out] += __shfl_down(acc[out], offset);
        }
    }
    if (tid == 0) {
        #pragma unroll
        for (int out = 0; out < OUT; ++out) {
            const int row = m_col * PACK + out_base + out;
            if constexpr (RESIDUAL) {
                y[row] += acc[out];
            } else {
                y[row] = acc[out];
            }
        }
    }
}

__launch_bounds__(32, 8)
extern "C" __global__ void paro4g128_rotate(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ x_rot,
    int M, int K
) {
    paro4g128_rotate_impl<false, false>(A, x, nullptr, x_rot, M, K);
}

__launch_bounds__(32, 8)
extern "C" __global__ void paro4g128_swiglu_rotate(
    const char* __restrict__ A,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ x_rot,
    int M, int K
) {
    paro4g128_rotate_impl<true, false>(A, gate, up, x_rot, M, K);
}

__launch_bounds__(32, 8)
extern "C" __global__ void paro4g128t_rotate(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ x_rot,
    int M, int K
) {
    paro4g128_rotate_impl<false, true>(A, x, nullptr, x_rot, M, K);
}

__launch_bounds__(32, 8)
extern "C" __global__ void paro4g128t_swiglu_rotate(
    const char* __restrict__ A,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ x_rot,
    int M, int K
) {
    paro4g128_rotate_impl<true, true>(A, gate, up, x_rot, M, K);
}

__launch_bounds__(32, 8)
extern "C" __global__ void paro4g128t_dual_rotate(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const float* __restrict__ x,
    float* __restrict__ x_rot_gate,
    float* __restrict__ x_rot_up,
    int M, int K
) {
    const bool up_lane = blockIdx.y != 0;
    const char* A = up_lane ? A_up : A_gate;
    float* x_rot = up_lane ? x_rot_up : x_rot_gate;
    paro4g128_rotate_impl<false, true>(A, x, nullptr, x_rot, M, K);
}

__launch_bounds__(32, 8)
extern "C" __global__ void paro4g128t_quad_rotate(
    const char* __restrict__ A0,
    const char* __restrict__ A1,
    const char* __restrict__ A2,
    const char* __restrict__ A3,
    const float* __restrict__ x,
    float* __restrict__ x_rot0,
    float* __restrict__ x_rot1,
    float* __restrict__ x_rot2,
    float* __restrict__ x_rot3,
    int M0, int M1, int M2, int M3, int K
) {
    const int lane = blockIdx.y;
    const char* A = A0;
    float* x_rot = x_rot0;
    int M = M0;
    if (lane == 1) {
        A = A1;
        x_rot = x_rot1;
        M = M1;
    } else if (lane == 2) {
        A = A2;
        x_rot = x_rot2;
        M = M2;
    } else if (lane == 3) {
        A = A3;
        x_rot = x_rot3;
        M = M3;
    }
    if (M <= 0) return;
    paro4g128_rotate_impl<false, true>(A, x, nullptr, x_rot, M, K);
}

__launch_bounds__(32, 8)
extern "C" __global__ void gemv_paro4g128_prerotated(
    const char* __restrict__ A,
    const float* __restrict__ x_rot,
    float* __restrict__ y,
    int M, int K
) {
    gemv_paro4g128_prerotated_impl<false, false>(A, x_rot, y, M, K);
}

__launch_bounds__(32, 8)
extern "C" __global__ void gemv_paro4g128_prerotated_residual(
    const char* __restrict__ A,
    const float* __restrict__ x_rot,
    float* __restrict__ y,
    int M, int K
) {
    gemv_paro4g128_prerotated_impl<true, false>(A, x_rot, y, M, K);
}

__launch_bounds__(32, 8)
extern "C" __global__ void gemv_paro4g128t_prerotated(
    const char* __restrict__ A,
    const float* __restrict__ x_rot,
    float* __restrict__ y,
    int M, int K
) {
    gemv_paro4g128_prerotated_impl<false, true>(A, x_rot, y, M, K);
}

__launch_bounds__(32, 8)
extern "C" __global__ void gemv_paro4g128t_prerotated_residual(
    const char* __restrict__ A,
    const float* __restrict__ x_rot,
    float* __restrict__ y,
    int M, int K
) {
    gemv_paro4g128_prerotated_impl<true, true>(A, x_rot, y, M, K);
}

__launch_bounds__(32, 12)
extern "C" __global__ void fused_gate_up_paro4g128t_pack4(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const float* __restrict__ x_rot_gate,
    const float* __restrict__ x_rot_up,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int M, int K
) {
    const bool up_lane = blockIdx.y != 0;
    const char* A = up_lane ? A_up : A_gate;
    const float* x_rot = up_lane ? x_rot_up : x_rot_gate;
    float* y = up_lane ? y_up : y_gate;
    gemv_paro4g128_prerotated_pack4_impl<false, true>(A, x_rot, y, M, K);
}

__launch_bounds__(32, 12)
extern "C" __global__ void fused_qkvza_paro4g128t_pack4(
    const char* __restrict__ A0,
    const char* __restrict__ A1,
    const char* __restrict__ A2,
    const char* __restrict__ A3,
    const float* __restrict__ x_rot0,
    const float* __restrict__ x_rot1,
    const float* __restrict__ x_rot2,
    const float* __restrict__ x_rot3,
    float* __restrict__ y0,
    float* __restrict__ y1,
    float* __restrict__ y2,
    float* __restrict__ y3,
    int M0, int M1, int M2, int M3, int K
) {
    const int lane = blockIdx.y;
    const char* A = A0;
    const float* x_rot = x_rot0;
    float* y = y0;
    int M = M0;
    if (lane == 1) {
        A = A1;
        x_rot = x_rot1;
        y = y1;
        M = M1;
    } else if (lane == 2) {
        A = A2;
        x_rot = x_rot2;
        y = y2;
        M = M2;
    } else if (lane == 3) {
        A = A3;
        x_rot = x_rot3;
        y = y3;
        M = M3;
    }
    if (M <= 0) return;
    gemv_paro4g128_prerotated_pack4_impl<false, true>(A, x_rot, y, M, K);
}
