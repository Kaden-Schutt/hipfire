#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>

enum class Mode : int {
    Full = 0,
    NoStore = 1,
    AOnly = 2,
    BOnly = 3,
    NoGlobal = 4,
    AOnlyNoStore = 5,
    BOnlyNoStore = 6,
    NoGlobalNoStore = 7,
};

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err__ = (call);                                             \
        if (err__ != hipSuccess) {                                             \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,                \
                         hipGetErrorString(err__), static_cast<int>(err__));   \
            return 2;                                                          \
        }                                                                      \
    } while (0)

template <int TILE>
__global__ void gemm_lds_probe(const float* A, const float* B, float* C,
                               int M, int N, int K, int k_limit, int mode_raw) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const Mode mode = static_cast<Mode>(mode_raw);
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        const bool load_a = mode == Mode::Full || mode == Mode::NoStore ||
                            mode == Mode::AOnly || mode == Mode::AOnlyNoStore;
        const bool load_b = mode == Mode::Full || mode == Mode::NoStore ||
                            mode == Mode::BOnly || mode == Mode::BOnlyNoStore;
        As[ty][tx] = (load_a && m < M && kt + tx < K)
                         ? A[(size_t)m * K + kt + tx]
                         : static_cast<float>(ty + tx + kt);
        Bs[ty][tx] = (load_b && kt + ty < K && n < N)
                         ? B[(size_t)(kt + ty) * N + n]
                         : static_cast<float>(ty - tx + kt);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            sum += As[ty][kk] * Bs[kk][tx];
        }
        __syncthreads();
    }

    const bool store_c = mode == Mode::Full || mode == Mode::AOnly ||
                         mode == Mode::BOnly || mode == Mode::NoGlobal;
    if (store_c && m < M && n < N) {
        C[(size_t)m * N + n] = sum;
    }
}

template <int TILE>
__global__ void gemm_lds_synth_probe(int M, int N, int K, int k_limit) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        Bs[ty][tx] = static_cast<float>((n & 7) - ty + kt);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            sum += As[ty][kk] * Bs[kk][tx];
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_synth_masked_probe(int M, int N, int K, int k_limit) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;
    const bool active = tx < TILE && ty < TILE;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        if (active) {
            As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
            Bs[ty][tx] = static_cast<float>((n & 7) - ty + kt);
        }
        __syncthreads();

        if (active) {
#pragma unroll
            for (int kk = 0; kk < TILE; ++kk) {
                sum += As[ty][kk] * Bs[kk][tx];
            }
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_nolds_synth_probe(int M, int N, int K, int k_limit) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float a = static_cast<float>((m & 7) + kk + kt);
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += a * b;
        }
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_barrier_synth_probe(int M, int N, int K, int k_limit) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float a = static_cast<float>((m & 7) + kk + kt);
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += a * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_barrier3_synth_probe(int M, int N, int K, int k_limit) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        __syncthreads();
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float a = static_cast<float>((m & 7) + kk + kt);
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += a * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_one_synth_probe(int M, int N, int K, int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += As[ty][kk] * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_two_store_one_read_probe(int M, int N, int K,
                                                  int k_limit) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        Bs[ty][tx] = static_cast<float>((n & 7) - ty + kt);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += As[ty][kk] * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_padded_one_synth_probe(int M, int N, int K,
                                                int k_limit) {
    __shared__ volatile float As[2 * TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += As[ty][kk] * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_two_store_once_one_read_probe(int M, int N, int K,
                                                       int k_limit) {
    __shared__ float As[TILE][TILE];
    __shared__ volatile float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    Bs[ty][tx] = static_cast<float>((n & 7) - ty);
    __syncthreads();

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += As[ty][kk] * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_forced_same_second_store_probe(int M, int N, int K,
                                                        int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        As[ty][tx] = As[ty][tx] + 1.0f;
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += As[ty][kk] * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_forced_same_load_only_probe(int M, int N, int K,
                                                     int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        const float tmp = As[ty][tx];
        asm volatile("" ::"v"(tmp));
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += As[ty][kk] * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_same_phase_no_wide_read_probe(int M, int N, int K,
                                                       int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        const float tmp = As[ty][tx] + 1.0f;
        As[ty][tx] = tmp;
        __syncthreads();

        sum += tmp * static_cast<float>((n & 7) - ty + kt);
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE, int READS>
__global__ void gemm_lds_forced_same_readwidth_probe(int M, int N, int K,
                                                     int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        As[ty][tx] = As[ty][tx] + 1.0f;
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < READS; ++kk) {
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += As[ty][kk] * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_second_store_only_read6_probe(int M, int N, int K,
                                                       int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        As[ty][tx] = static_cast<float>((m & 7) + tx + kt + 1);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += As[ty][kk] * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_load_independent_store_read6_probe(int M, int N, int K,
                                                            int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        const float tmp = As[ty][tx];
        asm volatile("" ::"v"(tmp));
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt + 1);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += As[ty][kk] * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_load_next_store_same_read6_probe(int M, int N, int K,
                                                          int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;
    const int next = (tx + 1) % TILE;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        const float tmp = As[ty][next];
        asm volatile("" ::"v"(tmp));
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt + 1);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += As[ty][kk] * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_load_same_store_next_read6_probe(int M, int N, int K,
                                                          int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;
    const int next = (tx + 1) % TILE;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        const float tmp = As[ty][tx];
        asm volatile("" ::"v"(tmp));
        As[ty][next] = static_cast<float>((m & 7) + tx + kt + 1);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += As[ty][kk] * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_store_then_load_read6_probe(int M, int N, int K,
                                                     int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        As[ty][tx] = static_cast<float>((m & 7) + tx + kt + 1);
        __syncthreads();

        const float tmp = As[ty][tx];
        asm volatile("" ::"v"(tmp));

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += As[ty][kk] * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_store_then_load_dynamiccols_read6_probe(int M, int N,
                                                                 int K,
                                                                 int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    int c0 = 0;
    int c1 = 1;
    int c2 = 2;
    int c3 = 3;
    int c4 = 4;
    int c5 = 5;
    asm volatile("" : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3),
                 "+v"(c4), "+v"(c5));

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        As[ty][tx] = static_cast<float>((m & 7) + tx + kt + 1);
        __syncthreads();

        const float tmp = As[ty][tx];
        asm volatile("" ::"v"(tmp));

        const float b = static_cast<float>((n & 7) - ty + kt);
        sum += As[ty][c0] * b;
        sum += As[ty][c1] * b;
        sum += As[ty][c2] * b;
        sum += As[ty][c3] * b;
        sum += As[ty][c4] * b;
        sum += As[ty][c5] * b;
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_store_then_load_dynamiccols_load5_use5_probe(
    int M, int N, int K, int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    int c0 = 0;
    int c1 = 1;
    int c2 = 2;
    int c3 = 3;
    int c4 = 4;
    asm volatile("" : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3),
                 "+v"(c4));

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        As[ty][tx] = static_cast<float>((m & 7) + tx + kt + 1);
        __syncthreads();

        const float tmp = As[ty][tx];
        asm volatile("" ::"v"(tmp));

        const float b = static_cast<float>((n & 7) - ty + kt);
        sum += As[ty][c0] * b;
        sum += As[ty][c1] * b;
        sum += As[ty][c2] * b;
        sum += As[ty][c3] * b;
        sum += As[ty][c4] * b;
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_store_then_load_dynamiccols_load5_serial_use5_probe(
    int M, int N, int K, int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    int c0 = 0;
    int c1 = 1;
    int c2 = 2;
    int c3 = 3;
    int c4 = 4;
    asm volatile("" : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3),
                 "+v"(c4));

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        As[ty][tx] = static_cast<float>((m & 7) + tx + kt + 1);
        __syncthreads();

        const float tmp = As[ty][tx];
        asm volatile("" ::"v"(tmp));

        const float b = static_cast<float>((n & 7) - ty + kt);
        float r0 = As[ty][c0];
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        sum += r0 * b;
        float r1 = As[ty][c1];
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        sum += r1 * b;
        float r2 = As[ty][c2];
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        sum += r2 * b;
        float r3 = As[ty][c3];
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        sum += r3 * b;
        float r4 = As[ty][c4];
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        sum += r4 * b;
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_store_then_load_dynamiccols_load5_split4_use5_probe(
    int M, int N, int K, int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    int c0 = 0;
    int c1 = 1;
    int c2 = 2;
    int c3 = 3;
    int c4 = 4;
    asm volatile("" : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3),
                 "+v"(c4));

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        As[ty][tx] = static_cast<float>((m & 7) + tx + kt + 1);
        __syncthreads();

        const float tmp = As[ty][tx];
        asm volatile("" ::"v"(tmp));

        const float b = static_cast<float>((n & 7) - ty + kt);
        float r0 = As[ty][c0];
        float r1 = As[ty][c1];
        float r2 = As[ty][c2];
        float r3 = As[ty][c3];
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        float r4 = As[ty][c4];
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        sum += r0 * b;
        sum += r1 * b;
        sum += r2 * b;
        sum += r3 * b;
        sum += r4 * b;
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_store_then_load_dynamiccols_load4_use4_probe(
    int M, int N, int K, int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    int c0 = 0;
    int c1 = 1;
    int c2 = 2;
    int c3 = 3;
    asm volatile("" : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3));

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        As[ty][tx] = static_cast<float>((m & 7) + tx + kt + 1);
        __syncthreads();

        const float tmp = As[ty][tx];
        asm volatile("" ::"v"(tmp));

        const float b = static_cast<float>((n & 7) - ty + kt);
        sum += As[ty][c0] * b;
        sum += As[ty][c1] * b;
        sum += As[ty][c2] * b;
        sum += As[ty][c3] * b;
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE, int READS>
__global__ void gemm_lds_store_then_load_readwidth_probe(int M, int N, int K,
                                                         int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        As[ty][tx] = static_cast<float>((m & 7) + tx + kt + 1);
        __syncthreads();

        const float tmp = As[ty][tx];
        asm volatile("" ::"v"(tmp));

#pragma unroll
        for (int kk = 0; kk < READS; ++kk) {
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += As[ty][kk] * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_store_then_load_nextrow_read6_probe(int M, int N, int K,
                                                             int k_limit) {
    __shared__ float As[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int other = (ty + 1) % TILE;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        As[ty][tx] = static_cast<float>((m & 7) + tx + kt + 1);
        __syncthreads();

        const float tmp = As[other][tx];
        asm volatile("" ::"v"(tmp));

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += As[ty][kk] * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_store_then_load_separate_tile_read6_probe(int M, int N,
                                                                   int K,
                                                                   int k_limit) {
    __shared__ float As[TILE][TILE];
    __shared__ float Xs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    Xs[ty][tx] = static_cast<float>((m & 3) + tx);
    __syncthreads();

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        As[ty][tx] = static_cast<float>((m & 7) + tx + kt + 1);
        __syncthreads();

        const float tmp = Xs[ty][tx];
        asm volatile("" ::"v"(tmp));

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += As[ty][kk] * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

template <int TILE>
__global__ void gemm_lds_store_then_load_separate_readtile_probe(int M, int N,
                                                                 int K,
                                                                 int k_limit) {
    __shared__ float As[TILE][TILE];
    __shared__ float Xs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * TILE + ty;
    const int n = blockIdx.x * TILE + tx;
    const int K_run = k_limit > 0 && k_limit < K ? k_limit : K;

    Xs[ty][tx] = static_cast<float>((m & 3) + tx);
    __syncthreads();

    float sum = 0.0f;
    for (int kt = 0; kt < K_run; kt += TILE) {
        As[ty][tx] = static_cast<float>((m & 7) + tx + kt);
        __syncthreads();

        As[ty][tx] = static_cast<float>((m & 7) + tx + kt + 1);
        __syncthreads();

        const float tmp = As[ty][tx];
        asm volatile("" ::"v"(tmp));

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            const float b = static_cast<float>((n & 7) - ty + kt);
            sum += Xs[ty][kk] * b;
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

static int parse_mode(const char* mode) {
    if (std::strcmp(mode, "full") == 0) return static_cast<int>(Mode::Full);
    if (std::strcmp(mode, "nostore") == 0) return static_cast<int>(Mode::NoStore);
    if (std::strcmp(mode, "aonly") == 0) return static_cast<int>(Mode::AOnly);
    if (std::strcmp(mode, "bonly") == 0) return static_cast<int>(Mode::BOnly);
    if (std::strcmp(mode, "noglobal") == 0) return static_cast<int>(Mode::NoGlobal);
    if (std::strcmp(mode, "aonly_nostore") == 0) return static_cast<int>(Mode::AOnlyNoStore);
    if (std::strcmp(mode, "bonly_nostore") == 0) return static_cast<int>(Mode::BOnlyNoStore);
    if (std::strcmp(mode, "noglobal_nostore") == 0) return static_cast<int>(Mode::NoGlobalNoStore);
    return -1;
}

static int run_variant(const char* variant, const char* mode_name, int launches,
                       int m, int n, int k, int k_limit) {
    const int mode = parse_mode(mode_name);
    if (mode < 0) {
        std::fprintf(stderr, "unknown mode: %s\n", mode_name);
        return 2;
    }

    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    HIP_CHECK(hipMalloc(&A, (size_t)m * k * sizeof(float)));
    HIP_CHECK(hipMalloc(&B, (size_t)k * n * sizeof(float)));
    HIP_CHECK(hipMalloc(&C, (size_t)m * n * sizeof(float)));
    HIP_CHECK(hipMemset(A, 0, (size_t)m * k * sizeof(float)));
    HIP_CHECK(hipMemset(B, 0, (size_t)k * n * sizeof(float)));
    HIP_CHECK(hipMemset(C, 0, (size_t)m * n * sizeof(float)));

    for (int i = 0; i < launches; ++i) {
        if (std::strcmp(variant, "tile5") == 0) {
            dim3 block(5, 5, 1);
            dim3 grid((n + 4) / 5, (m + 4) / 5, 1);
            hipLaunchKernelGGL((gemm_lds_probe<5>), grid, block, 0, 0, A, B, C,
                               m, n, k, k_limit, mode);
        } else if (std::strcmp(variant, "tile6") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_probe<6>), grid, block, 0, 0, A, B, C,
                               m, n, k, k_limit, mode);
        } else if (std::strcmp(variant, "tile6_synth") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_synth_probe<6>), grid, block, 0, 0,
                               m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_synth_masked") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_synth_masked_probe<6>), grid, block,
                               0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_nolds_synth") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_nolds_synth_probe<6>), grid, block, 0, 0,
                               m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_barrier_synth") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_barrier_synth_probe<6>), grid, block, 0,
                               0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_barrier3_synth") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_barrier3_synth_probe<6>), grid, block, 0,
                               0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_one_synth") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_one_synth_probe<6>), grid, block, 0,
                               0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_two_store_one_read") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_two_store_one_read_probe<6>), grid,
                               block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_padded_one_synth") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_padded_one_synth_probe<6>), grid,
                               block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_two_store_once_one_read") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_two_store_once_one_read_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_forced_same_second_store") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_forced_same_second_store_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_forced_same_load_only") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_forced_same_load_only_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_same_phase_no_wide_read") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_same_phase_no_wide_read_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_forced_same_read1") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_forced_same_readwidth_probe<6, 1>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_forced_same_read2") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_forced_same_readwidth_probe<6, 2>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_forced_same_read4") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_forced_same_readwidth_probe<6, 4>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_forced_same_read5") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_forced_same_readwidth_probe<6, 5>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_second_store_only_read6") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_second_store_only_read6_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_load_independent_store_read6") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_load_independent_store_read6_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_load_next_store_same_read6") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_load_next_store_same_read6_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_load_same_store_next_read6") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_load_same_store_next_read6_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_store_then_load_read6") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_store_then_load_read6_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_store_then_load_dynamiccols_read6") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_store_then_load_dynamiccols_read6_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_store_then_load_dynamiccols_load5_use5") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_store_then_load_dynamiccols_load5_use5_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_store_then_load_dynamiccols_load5_serial_use5") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_store_then_load_dynamiccols_load5_serial_use5_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_store_then_load_dynamiccols_load5_split4_use5") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_store_then_load_dynamiccols_load5_split4_use5_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_store_then_load_dynamiccols_load4_use4") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_store_then_load_dynamiccols_load4_use4_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_store_then_load_read0") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_store_then_load_readwidth_probe<6, 0>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_store_then_load_read1") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_store_then_load_readwidth_probe<6, 1>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_store_then_load_read2") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_store_then_load_readwidth_probe<6, 2>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_store_then_load_read4") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_store_then_load_readwidth_probe<6, 4>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_store_then_load_read5") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_store_then_load_readwidth_probe<6, 5>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_store_then_load_nextrow_read6") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_store_then_load_nextrow_read6_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_store_then_load_separate_tile_read6") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_store_then_load_separate_tile_read6_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile6_lds_store_then_load_separate_readtile") == 0) {
            dim3 block(6, 6, 1);
            dim3 grid((n + 5) / 6, (m + 5) / 6, 1);
            hipLaunchKernelGGL((gemm_lds_store_then_load_separate_readtile_probe<6>),
                               grid, block, 0, 0, m, n, k, k_limit);
        } else if (std::strcmp(variant, "tile16") == 0) {
            dim3 block(16, 16, 1);
            dim3 grid((n + 15) / 16, (m + 15) / 16, 1);
            hipLaunchKernelGGL((gemm_lds_probe<16>), grid, block, 0, 0, A, B, C,
                               m, n, k, k_limit, mode);
        } else {
            std::fprintf(stderr, "unknown variant: %s\n", variant);
            hipFree(A);
            hipFree(B);
            hipFree(C);
            return 2;
        }

        hipError_t launch_err = hipGetLastError();
        if (launch_err != hipSuccess) {
            std::fprintf(stderr, "launch %d failed: %s (%d)\n", i,
                         hipGetErrorString(launch_err),
                         static_cast<int>(launch_err));
            hipFree(A);
            hipFree(B);
            hipFree(C);
            return 3;
        }

        hipError_t sync_err = hipDeviceSynchronize();
        if (sync_err != hipSuccess) {
            std::fprintf(stderr, "sync %d failed: %s (%d)\n", i,
                         hipGetErrorString(sync_err),
                         static_cast<int>(sync_err));
            hipFree(A);
            hipFree(B);
            hipFree(C);
            return 4;
        }
    }

    float first = 1.0f;
    HIP_CHECK(hipMemcpy(&first, C, sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(A));
    HIP_CHECK(hipFree(B));
    HIP_CHECK(hipFree(C));
    std::printf("gemm variant=%s mode=%s launches=%d shape=%dx%dx%d k_limit=%d first=%f OK\n",
                variant, mode_name, launches, m, n, k, k_limit, first);
    return 0;
}

int main(int argc, char** argv) {
    const char* variant = argc > 1 ? argv[1] : "tile6";
    const char* mode = argc > 2 ? argv[2] : "full";
    int launches = argc > 3 ? std::atoi(argv[3]) : 100;
    int m = argc > 4 ? std::atoi(argv[4]) : 512;
    int n = argc > 5 ? std::atoi(argv[5]) : 3072;
    int k = argc > 6 ? std::atoi(argv[6]) : 3072;
    int k_limit = argc > 7 ? std::atoi(argv[7]) : 0;
    return run_variant(variant, mode, launches, m, n, k, k_limit);
}
