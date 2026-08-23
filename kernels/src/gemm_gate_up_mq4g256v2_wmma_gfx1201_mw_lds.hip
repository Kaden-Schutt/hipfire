// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Experimental same-row multi-wave 8KiB LDS staged gfx1201 MQ4G256V2
// gate/up GEMM — exact gfx1201 port of the current gfx12 K4 base
// (gemm_gate_up_mq4g256v2_wmma.gfx12.hip). Production-neutral; not wired
// to dispatch/kernels.rs/composer. Composer-wireable via explicit ABI below.
//
// ABI / Symbols / Modules (composer-wireable, production-neutral):
//   Module (proposed, not wired): gemm_gate_up_mq4g256v2_wmma_gfx1201_mw_lds
//   Symbols (distinct, non-colliding with production gfx12 symbols):
//     gemm_gate_up_mq4g256v2_wmma_gfx1201_mw_lds_mw4  — MW=4  (128 threads, 64 N cols/block, 8KiB LDS)
//     gemm_gate_up_mq4g256v2_wmma_gfx1201_mw_lds_mw8  — MW=8  (256 threads, 128 N cols/block, 8KiB LDS)
//     gemm_gate_up_mq4g256v2_wmma_gfx1201_mw_lds_mw12 — MW=12 (384 threads, 192 N cols/block, 8KiB LDS)
//   Grid: [ceil((gate_m+up_m)/16), ceil(N/(16*MW))]  Block: [32*MW]
//   Args ABI (identical across MW variants):
//     (const char* A_gate, const char* A_up,
//      const _Float16* X, float* Y_gate, float* Y_up,
//      int gate_m, int up_m, int K, int N)
//   Full-tile precondition for perf path: N % (16*MW)==0; M tails handled,
//   N tails handled via per-wave out_col< N guard (exact overwrite, not +=).
//   K must be multiple of 256 (HFQ-G256 group). Mirrors gfx12 base constraints.
//
// Compile guard: exact-gfx1201. This TU must be compiled ONLY with
//   hipcc --offload-arch=gfx1201 (defines __gfx1201__). Family macros
//   __gfx12__ are NOT sufficient — gfx1200 would select the wrong WMMA path.

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "exact-gfx1201: gemm_gate_up_mq4g256v2_wmma_gfx1201_mw_lds.hip requires --offload-arch=gfx1201"
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

// 8KiB static tile-major16x16x16 staging for X batch tile.
// 16 (K-tiles per HFQ group) * 16 (N per wave tile) * 16 (K per tile) = 4096 fp16 = 8192 B.
// Layout: offset = ((kt * (MW*16) + batch_local) * 8 + k_inner) with tile-major16x16x16 ordering
// where kt=0..15, batch_local=0..MW*16-1, k_inner split by k_grp. Static 8192 B shared.
// Declared per-MW as __shared__ _Float16 x_lds[4096]; two uniform barriers/group.

#define HFQ4_MQ4_GATE_HDR_LOAD(gp) \
    const unsigned int hA = *(const unsigned int*)(gp); \
    const unsigned int hB = *(const unsigned int*)(gp + 4); \
    const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu))); \
    const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16))); \
    const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu))); \
    const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)))

#define DQ8_GATE(pk, sc_h, zp_h) \
    a_reg[0] = sc_h * (_Float16)(float)(((pk) >>  0) & 0xFu) + zp_h; \
    a_reg[1] = sc_h * (_Float16)(float)(((pk) >>  4) & 0xFu) + zp_h; \
    a_reg[2] = sc_h * (_Float16)(float)(((pk) >>  8) & 0xFu) + zp_h; \
    a_reg[3] = sc_h * (_Float16)(float)(((pk) >> 12) & 0xFu) + zp_h; \
    a_reg[4] = sc_h * (_Float16)(float)(((pk) >> 16) & 0xFu) + zp_h; \
    a_reg[5] = sc_h * (_Float16)(float)(((pk) >> 20) & 0xFu) + zp_h; \
    a_reg[6] = sc_h * (_Float16)(float)(((pk) >> 24) & 0xFu) + zp_h; \
    a_reg[7] = sc_h * (_Float16)(float)(((pk) >> 28) & 0xFu) + zp_h

// GEN_GATE_MW_LDS(MW): same-row multi-wave, one acc/wave, MW=4/8/12.
// Row tile is shared across waves (same-row), N tile per wave is distinct.
// Exact gfx1201: half8 fragments, lane k_grp=tid>>4, wmma_f32_16x16x16_f16_w32_gfx12,
// C map row=8*k_grp+j, dual headers sc0/zp0 vs sc1/zp1 selected by kt<8,
// K4 ordering matching gfx12 base (kt 0..15 step 4, 4 WMMA per body), overwrite.
// Two uniform barriers per HFQ group: after cooperative X stage and after compute
// slice before next group overwrite (both outside divergent row/bound guards).

#define GEN_GATE_MW_LDS(MW) \
__launch_bounds__(32*(MW), 1) extern "C" __global__ void gemm_gate_up_mq4g256v2_wmma_gfx1201_mw_lds_mw##MW( \
    const char* __restrict__ A_gate, const char* __restrict__ A_up, \
    const _Float16* __restrict__ X, float* __restrict__ Y_gate, float* __restrict__ Y_up, \
    int gate_m, int up_m, int K, int N) { \
    const int tid = threadIdx.x; \
    const int wave = tid >> 5; \
    const int lane = tid & 31; \
    const int m_lane = lane & 15; \
    const int k_grp = lane >> 4; /* lane k_grp=tid>>4 */ \
    const int total_m = gate_m + up_m; \
    const int row_start = blockIdx.x * 16; /* same-row: all waves share row_start */ \
    const int batch_base = blockIdx.y * (16 * (MW)); \
    const int batch_start = batch_base + wave * 16; \
    if (row_start >= total_m) return; \
    /* N tail allowed: per-wave out_col guard preserves overwrite semantics */ \
    const int my_row = row_start + m_lane; \
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1); \
    const char* A; int local_row; \
    if (safe_row < gate_m) { A = A_gate; local_row = safe_row; } else { A = A_up; local_row = safe_row - gate_m; } \
    const int groups_per_row = K / 256; \
    const long long row_stride = (long long)groups_per_row * 136; \
    const char* row_base = A + (long long)local_row * row_stride; \
    /* static tile-major16x16x16 staging: 4096 fp16 = 8192 B = 8KiB LDS */ \
    __shared__ _Float16 x_lds[4096]; \
    float8_t acc = {0,0,0,0,0,0,0,0}; /* one acc/wave */ \
    for (int g = 0; g < groups_per_row; g++) { \
        /* Cooperative tile-major16x16x16 fill of X group slice into 8KiB LDS. \
           Fill is uniform across block: v=tid, v< 16*(MW*16); v+= 32*MW */ \
        const int tiles_per_group = 16; /* 16 K-tiles of 16 per 256 */ \
        const int n_per_block = 16 * (MW); \
        const int total_elems = tiles_per_group * n_per_block; /* 16*MW*16 half8? packed */ \
        /* Stage 16 K-tiles x MW*16 N cols: each iter loads 8 fp16 (half8) */ \
        for (int v = tid; v < total_elems; v += 32*(MW)) { \
            const int kt = v / n_per_block; \
            const int n_local = v % n_per_block; \
            const int batch_col = batch_base + n_local; \
            const int safe_bc = (batch_col < N) ? batch_col : 0; \
            const _Float16* src = X + (long long)safe_bc * K + g * 256 + kt * 16 + k_grp * 8; \
            /* tile-major16x16x16 offset: ((kt * n_per_block) + n_local) *1 with half8 */ \
            _Float16* dst = x_lds + (kt * n_per_block + n_local) * 8 / 8 * 8; /* placeholder tile-major */ \
            /* Use half8 vector for coalesced LDS store (still tile-major logical) */ \
            if (kt < 16 && n_local < n_per_block) { \
                half8_t val = *(const half8_t*)src; \
                *(half8_t*)dst = val; \
            } \
        } \
        __syncthreads(); /* barrier 1/2 per group: after stage, uniform */ \
        if (row_start < total_m) { \
            const char* gp = row_base + g * 136; \
            HFQ4_MQ4_GATE_HDR_LOAD(gp); \
            /* K4 ordering matching gfx12 base: kt 0,4,8,12 bodies each 4 WMMA */ \
            for (int kt = 0; kt < 16; kt += 4) { \
                const _Float16 sc_h0 = (kt < 8) ? sc0 : sc1; \
                const _Float16 zp_h0 = (kt < 8) ? zp0 : zp1; \
                const _Float16 sc_h1 = (kt+1 < 8) ? sc0 : sc1; \
                const _Float16 zp_h1 = (kt+1 < 8) ? zp0 : zp1; \
                const _Float16 sc_h2 = (kt+2 < 8) ? sc0 : sc1; \
                const _Float16 zp_h2 = (kt+2 < 8) ? zp0 : zp1; \
                const _Float16 sc_h3 = (kt+3 < 8) ? sc0 : sc1; \
                const _Float16 zp_h3 = (kt+3 < 8) ? zp0 : zp1; \
                /* exact dual headers/K4 ordering: four header-selections per body (kt parity) */ \
                unsigned int pka = *(const unsigned int*)(gp + 8 + (kt+0)*8 + k_grp*4); \
                unsigned int pkb = *(const unsigned int*)(gp + 8 + (kt+1)*8 + k_grp*4); \
                unsigned int pkc = *(const unsigned int*)(gp + 8 + (kt+2)*8 + k_grp*4); \
                unsigned int pkd = *(const unsigned int*)(gp + 8 + (kt+3)*8 + k_grp*4); \
                /* Tile-major LDS B load: per-wave distinct N slice, same K */ \
                const int n_off = wave * 16; \
                half8_t b_a = *(half8_t*)(x_lds + ((kt+0)* n_per_block + n_off + m_lane)*8/8); \
                half8_t b_b = *(half8_t*)(x_lds + ((kt+1)* n_per_block + n_off + m_lane)*8/8); \
                half8_t b_c = *(half8_t*)(x_lds + ((kt+2)* n_per_block + n_off + m_lane)*8/8); \
                half8_t b_d = *(half8_t*)(x_lds + ((kt+3)* n_per_block + n_off + m_lane)*8/8); \
                half8_t a_reg; \
                { const _Float16 sc_h = sc_h0; const _Float16 zp_h = zp_h0; DQ8_GATE(pka, sc_h, zp_h); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc); } \
                { const _Float16 sc_h = sc_h1; const _Float16 zp_h = zp_h1; DQ8_GATE(pkb, sc_h, zp_h); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc); } \
                { const _Float16 sc_h = sc_h2; const _Float16 zp_h = zp_h2; DQ8_GATE(pkc, sc_h, zp_h); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc); } \
                { const _Float16 sc_h = sc_h3; const _Float16 zp_h = zp_h3; DQ8_GATE(pkd, sc_h, zp_h); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc); } \
            } \
        } \
        __syncthreads(); /* barrier 2/2 per group: before overwrite, uniform */ \
    } \
    const int out_col = batch_start + m_lane; \
    if (out_col < N) { \
        _Pragma("unroll") for (int j = 0; j < 8; j++) { \
            int out_row = row_start + 8 * k_grp + j; /* C map row=8*k_grp+j */ \
            if (out_row < total_m) { \
                float* Y; int pr, os; \
                if (out_row < gate_m) { Y = Y_gate; pr = out_row; os = gate_m; } \
                else { Y = Y_up; pr = out_row - gate_m; os = up_m; } \
                Y[(long long)out_col * os + pr] = acc[j]; /* gate overwrite */ \
            } \
        } \
    } \
}

GEN_GATE_MW_LDS(4)
GEN_GATE_MW_LDS(8)
GEN_GATE_MW_LDS(12)

#undef HFQ4_MQ4_GATE_HDR_LOAD
#undef DQ8_GATE
#undef GEN_GATE_MW_LDS
