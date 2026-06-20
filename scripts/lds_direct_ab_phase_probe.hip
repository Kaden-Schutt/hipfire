#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef ACTIVE_X
#define ACTIVE_X 6
#endif

#ifndef ACTIVE_Y
#define ACTIVE_Y 6
#endif

#ifndef ACTIVE_X_START
#define ACTIVE_X_START 0
#endif

#ifndef ACTIVE_Y_START
#define ACTIVE_Y_START 0
#endif

#ifndef BLOCK_X
#define BLOCK_X ACTIVE_X
#endif

#ifndef BLOCK_Y
#define BLOCK_Y ACTIVE_Y
#endif

#ifndef LAYOUT_X
#define LAYOUT_X ACTIVE_X
#endif

#ifndef LAYOUT_Y
#define LAYOUT_Y ACTIVE_Y
#endif

#ifndef READS
#define READS 6
#endif

#ifndef ITERS
#define ITERS 320
#endif

#ifndef PRE_SYNC_EACH_LAUNCH
#define PRE_SYNC_EACH_LAUNCH 0
#endif

#ifndef FORCE_WRAP_CNDMASK
#define FORCE_WRAP_CNDMASK 0
#endif

#if ACTIVE_X_START == 0 && ACTIVE_Y_START == 0 && \
    ACTIVE_X == BLOCK_X && ACTIVE_Y == BLOCK_Y
#define ALL_LANES_ACTIVE 1
#else
#define ALL_LANES_ACTIVE 0
#endif

__global__ void lds_direct_ab_phase_probe() {
    static_assert(ACTIVE_X <= BLOCK_X, "ACTIVE_X must fit inside BLOCK_X");
    static_assert(ACTIVE_Y <= BLOCK_Y, "ACTIVE_Y must fit inside BLOCK_Y");
    static_assert(ACTIVE_X_START >= 0, "ACTIVE_X_START must be non-negative");
    static_assert(ACTIVE_Y_START >= 0, "ACTIVE_Y_START must be non-negative");
    static_assert(ACTIVE_X_START + ACTIVE_X <= BLOCK_X,
                  "ACTIVE_X range must fit inside BLOCK_X");
    static_assert(ACTIVE_Y_START + ACTIVE_Y <= BLOCK_Y,
                  "ACTIVE_Y range must fit inside BLOCK_Y");
    static_assert(ACTIVE_X_START + ACTIVE_X <= LAYOUT_X,
                  "ACTIVE_X range must fit inside LAYOUT_X");
    static_assert(ACTIVE_Y_START + ACTIVE_Y <= LAYOUT_Y,
                  "ACTIVE_Y range must fit inside LAYOUT_Y");
    static_assert(ACTIVE_X <= LAYOUT_X, "ACTIVE_X must fit inside LAYOUT_X");
    static_assert(ACTIVE_Y <= LAYOUT_Y, "ACTIVE_Y must fit inside LAYOUT_Y");

    __shared__ float A[LAYOUT_Y][LAYOUT_X];
    __shared__ float B[LAYOUT_Y][LAYOUT_X];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
#if ALL_LANES_ACTIVE
    const bool active = true;
#else
    const bool active =
        tx >= ACTIVE_X_START && tx < ACTIVE_X_START + ACTIVE_X &&
        ty >= ACTIVE_Y_START && ty < ACTIVE_Y_START + ACTIVE_Y;
#endif
    float sum = 0.0f;

    for (int it = 0; it < ITERS; ++it) {
        if (active) {
            A[ty][tx] = static_cast<float>(
                (blockIdx.x + 1) * (ty + 1) + tx + it);
            B[ty][tx] = static_cast<float>(
                (blockIdx.y + 1) * (tx + 1) - ty + it);
        }
        __syncthreads();

        if (active) {
#pragma unroll
            for (int kk = 0; kk < READS; ++kk) {
#if FORCE_WRAP_CNDMASK
                static_assert(READS <= 2,
                              "FORCE_WRAP_CNDMASK is only valid for READS<=2");
                const int ax_raw = tx + kk;
                const int ax =
                    ax_raw == ACTIVE_X_START + ACTIVE_X ? ACTIVE_X_START
                                                        : ax_raw;
                const int ay = ty;
                const int bx = tx;
                const int by_raw = ty + kk;
                const int by =
                    by_raw == ACTIVE_Y_START + ACTIVE_Y ? ACTIVE_Y_START
                                                        : by_raw;
#else
                const int ax =
                    ACTIVE_X_START + ((tx - ACTIVE_X_START + kk) % ACTIVE_X);
                const int ay = ty;
                const int bx = tx;
                const int by =
                    ACTIVE_Y_START + ((ty - ACTIVE_Y_START + kk) % ACTIVE_Y);
#endif
                sum += A[ay][ax] * B[by][bx] * static_cast<float>(kk + 1);
            }
        }
        __syncthreads();
    }

    asm volatile("" ::"v"(sum));
}

static int run_phase(const char* label, int launches, int grid_x, int grid_y,
                     hipStream_t stream, bool stream_sync, int global_offset) {
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    for (int i = 0; i < launches; ++i) {
#if PRE_SYNC_EACH_LAUNCH
        hipError_t pre_sync_err =
            stream_sync ? hipStreamSynchronize(stream) : hipDeviceSynchronize();
        if (pre_sync_err != hipSuccess) {
            std::fprintf(stderr,
                         "%s pre-sync %d global %d failed: %s (%d)\n", label,
                         i, global_offset + i,
                         hipGetErrorString(pre_sync_err),
                         static_cast<int>(pre_sync_err));
            return 4;
        }
#endif

        hipLaunchKernelGGL(lds_direct_ab_phase_probe, grid, block, 0, stream);

        hipError_t launch_err = hipGetLastError();
        if (launch_err != hipSuccess) {
            std::fprintf(stderr,
                         "%s launch %d global %d failed: %s (%d)\n", label, i,
                         global_offset + i, hipGetErrorString(launch_err),
                         static_cast<int>(launch_err));
            return 3;
        }

        hipError_t sync_err =
            stream_sync ? hipStreamSynchronize(stream) : hipDeviceSynchronize();
        if (sync_err != hipSuccess) {
            std::fprintf(stderr, "%s sync %d global %d failed: %s (%d)\n",
                         label, i, global_offset + i,
                         hipGetErrorString(sync_err),
                         static_cast<int>(sync_err));
            return 4;
        }
    }

    std::printf("%s active=%dx%d block=%dx%d reads=%d iters=%d launches=%d "
                "grid=%dx%d lds_bytes=%d layout=%dx%d pre_sync=%d OK\n",
                label, ACTIVE_X, ACTIVE_Y, BLOCK_X, BLOCK_Y, READS, ITERS,
                launches, grid_x, grid_y, 8 * LAYOUT_X * LAYOUT_Y, LAYOUT_X,
                LAYOUT_Y, PRE_SYNC_EACH_LAUNCH);
    return 0;
}

static int check_hip(const char* label, hipError_t err) {
    if (err == hipSuccess) {
        std::printf("%s OK\n", label);
        return 0;
    }
    std::fprintf(stderr, "%s failed: %s (%d)\n", label, hipGetErrorString(err),
                 static_cast<int>(err));
    return 5;
}

static int check_hsa(const char* label, hsa_status_t status) {
    if (status == HSA_STATUS_SUCCESS) {
        std::printf("%s OK\n", label);
        return 0;
    }
    const char* message = nullptr;
    hsa_status_string(status, &message);
    std::fprintf(stderr, "%s failed: %s (%d)\n", label,
                 message == nullptr ? "<no hsa status string>" : message,
                 static_cast<int>(status));
    return 6;
}

static bool mode_is(const char* mode, const char* expected) {
    return std::strcmp(mode, expected) == 0;
}

static int run_two_phase(int phase1, int phase2, int grid_x, int grid_y,
                         const char* mode) {
    std::printf("phase-probe mode=%s phase1=%d phase2=%d active=%dx%d "
                "block=%dx%d layout=%dx%d reads=%d iters=%d grid=%dx%d "
                "lds_bytes=%d\n",
                mode, phase1, phase2, ACTIVE_X, ACTIVE_Y, BLOCK_X, BLOCK_Y,
                LAYOUT_X, LAYOUT_Y, READS, ITERS, grid_x, grid_y,
                8 * LAYOUT_X * LAYOUT_Y);

    if (mode_is(mode, "same")) {
        int rc = run_phase("phase1", phase1, grid_x, grid_y, nullptr, false, 0);
        if (rc != 0) return rc;
        rc = check_hip("boundary hipDeviceSynchronize", hipDeviceSynchronize());
        if (rc != 0) return rc;
        rc = run_phase("phase2", phase2, grid_x, grid_y, nullptr, false,
                       phase1);
        if (rc != 0) return rc;
        std::printf("phase-probe mode=%s total=%d OK\n", mode,
                    phase1 + phase2);
        return 0;
    }

    if (mode_is(mode, "stream_same")) {
        hipStream_t stream = nullptr;
        int rc = check_hip("hipStreamCreate phase stream",
                           hipStreamCreate(&stream));
        if (rc != 0) return rc;
        rc = run_phase("phase1", phase1, grid_x, grid_y, stream, true, 0);
        if (rc == 0) {
            rc = run_phase("phase2", phase2, grid_x, grid_y, stream, true,
                           phase1);
        }
        int destroy_rc = check_hip("hipStreamDestroy phase stream",
                                   hipStreamDestroy(stream));
        return rc != 0 ? rc : destroy_rc;
    }

    if (mode_is(mode, "stream_recreate")) {
        hipStream_t stream = nullptr;
        int rc = check_hip("hipStreamCreate phase1",
                           hipStreamCreate(&stream));
        if (rc != 0) return rc;
        rc = run_phase("phase1", phase1, grid_x, grid_y, stream, true, 0);
        int destroy_rc =
            check_hip("hipStreamDestroy phase1", hipStreamDestroy(stream));
        if (rc != 0) return rc;
        if (destroy_rc != 0) return destroy_rc;

        stream = nullptr;
        rc = check_hip("hipStreamCreate phase2", hipStreamCreate(&stream));
        if (rc != 0) return rc;
        rc = run_phase("phase2", phase2, grid_x, grid_y, stream, true,
                       phase1);
        destroy_rc =
            check_hip("hipStreamDestroy phase2", hipStreamDestroy(stream));
        return rc != 0 ? rc : destroy_rc;
    }

    if (mode_is(mode, "device_reset")) {
        int rc = run_phase("phase1", phase1, grid_x, grid_y, nullptr, false, 0);
        if (rc != 0) return rc;
        rc = check_hip("boundary hipDeviceReset", hipDeviceReset());
        if (rc != 0) return rc;
        rc = run_phase("phase2", phase2, grid_x, grid_y, nullptr, false,
                       phase1);
        if (rc != 0) return rc;
        std::printf("phase-probe mode=%s total=%d OK\n", mode,
                    phase1 + phase2);
        return 0;
    }

    if (mode_is(mode, "primary_ctx_reset")) {
        int rc = run_phase("phase1", phase1, grid_x, grid_y, nullptr, false, 0);
        if (rc != 0) return rc;
        rc = check_hip("boundary hipDevicePrimaryCtxReset(0)",
                       hipDevicePrimaryCtxReset(0));
        if (rc != 0) return rc;
        rc = run_phase("phase2", phase2, grid_x, grid_y, nullptr, false,
                       phase1);
        if (rc != 0) return rc;
        std::printf("phase-probe mode=%s total=%d OK\n", mode,
                    phase1 + phase2);
        return 0;
    }

    if (mode_is(mode, "primary_ctx_release")) {
        int rc = run_phase("phase1", phase1, grid_x, grid_y, nullptr, false, 0);
        if (rc != 0) return rc;
        rc = check_hip("boundary hipDevicePrimaryCtxRelease(0)",
                       hipDevicePrimaryCtxRelease(0));
        if (rc != 0) return rc;
        rc = run_phase("phase2", phase2, grid_x, grid_y, nullptr, false,
                       phase1);
        if (rc != 0) return rc;
        std::printf("phase-probe mode=%s total=%d OK\n", mode,
                    phase1 + phase2);
        return 0;
    }

    if (mode_is(mode, "hsa_shutdown")) {
        int rc = run_phase("phase1", phase1, grid_x, grid_y, nullptr, false, 0);
        if (rc != 0) return rc;
        rc = check_hsa("boundary hsa_shut_down", hsa_shut_down());
        if (rc != 0) return rc;
        rc = run_phase("phase2", phase2, grid_x, grid_y, nullptr, false,
                       phase1);
        if (rc != 0) return rc;
        std::printf("phase-probe mode=%s total=%d OK\n", mode,
                    phase1 + phase2);
        return 0;
    }

    if (mode_is(mode, "hsa_shutdown_init")) {
        int rc = run_phase("phase1", phase1, grid_x, grid_y, nullptr, false, 0);
        if (rc != 0) return rc;
        rc = check_hsa("boundary hsa_shut_down", hsa_shut_down());
        if (rc != 0) return rc;
        rc = check_hsa("boundary hsa_init", hsa_init());
        if (rc != 0) return rc;
        rc = run_phase("phase2", phase2, grid_x, grid_y, nullptr, false,
                       phase1);
        if (rc != 0) return rc;
        std::printf("phase-probe mode=%s total=%d OK\n", mode,
                    phase1 + phase2);
        return 0;
    }

    if (mode_is(mode, "hsa_shutdown_hip_reset")) {
        int rc = run_phase("phase1", phase1, grid_x, grid_y, nullptr, false, 0);
        if (rc != 0) return rc;
        rc = check_hsa("boundary hsa_shut_down", hsa_shut_down());
        if (rc != 0) return rc;
        rc = check_hip("boundary hipDeviceReset after hsa_shut_down",
                       hipDeviceReset());
        if (rc != 0) return rc;
        rc = run_phase("phase2", phase2, grid_x, grid_y, nullptr, false,
                       phase1);
        if (rc != 0) return rc;
        std::printf("phase-probe mode=%s total=%d OK\n", mode,
                    phase1 + phase2);
        return 0;
    }

    std::fprintf(stderr,
                 "unknown mode '%s' (use same, stream_same, "
                 "stream_recreate, device_reset, primary_ctx_reset, "
                 "primary_ctx_release, hsa_shutdown, hsa_shutdown_init, "
                 "hsa_shutdown_hip_reset)\n",
                 mode);
    return 2;
}

int main(int argc, char** argv) {
    int phase1 = argc > 1 ? std::atoi(argv[1]) : 98;
    int phase2 = argc > 2 ? std::atoi(argv[2]) : 1;
    int grid_x = argc > 3 ? std::atoi(argv[3]) : 511;
    int grid_y = argc > 4 ? std::atoi(argv[4]) : 86;
    const char* mode = argc > 5 ? argv[5] : "same";
    return run_two_phase(phase1, phase2, grid_x, grid_y, mode);
}
