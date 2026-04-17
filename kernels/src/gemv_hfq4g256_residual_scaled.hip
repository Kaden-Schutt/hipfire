#include <hip/hip_runtime.h>

// HFQ4-G256 GEMV with fused SCALED residual add: y[row] += scale * (A[row] · x).
//
// Two kernel variants — identical math, differ only in how the scale is
// delivered:
//   _scaled_cpu     — scale is a kernarg f32, passed from the host. Used
//                     for the MoE routed-expert accumulator (weight comes
//                     from CPU top-K selection).
//   _scaled_gpu     — scale is read from c_buf[0] of a 1-element device
//                     buffer. Used for the MoE shared-expert accumulator
//                     (scale = sigmoid(shared_expert_gate · x), computed
//                     entirely on-device to avoid a D2H sync).
//
// Identical math to gemv_hfq4g256_residual except the final write multiplies
// acc by `scale` before the += — bit-exact with running the baseline kernel
// followed by an independent scaled_add.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_residual_scaled_cpu(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K,
    float scale
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
        float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
        float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
        float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));

        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

        int base = g * 256 + tid * 8;

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

        DOG(pk0, sc0, zp0, base, acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
    }

    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] += scale * acc;   // scaled residual add
}

// GPU-scalar variant — identical except `scale` is loaded from a device
// buffer. Avoids the D2H sync that a CPU-scalar version would force when
// the scale is already computed on-device (e.g. sigmoid(gate)).
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_residual_scaled_gpu(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ c_buf,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
        float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
        float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
        float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));

        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

        int base = g * 256 + tid * 8;

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

        DOG(pk0, sc0, zp0, base, acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
    }

    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] += c_buf[0] * acc;   // scaled residual add, gpu-scalar
}

// Variant: same math as _scaled_gpu but applies sigmoid to c_buf[0] before
// scaling. Lets the caller skip a separate `sigmoid_f32` launch on the 1-elem
// shared-expert gate scalar — one less launch per MoE layer (~40 launches
// saved per forward on A3B). Bit-equivalent to sigmoid_f32 followed by
// _scaled_gpu up to single-precision float precision.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_residual_sigmoid_scaled_gpu(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ c_buf,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
        float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
        float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
        float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));

        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

        int base = g * 256 + tid * 8;

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

        DOG(pk0, sc0, zp0, base, acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
    }

    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        float gate = 1.0f / (1.0f + __expf(-c_buf[0]));
        y[row] += gate * acc;
    }
}
