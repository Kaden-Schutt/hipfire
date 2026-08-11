// Host-side bit-parity simulator for the MQ3-Lloyd MoE-down row tile.
//
// Reproduces, in scalar C++, the exact lane/LDS/shuffle structure of
//   kernels/src/gemv_mq3g256_lloyd_moe_down_indexed.hip      (incumbent, R=1)
//   kernels/src/gemv_mq3g256_lloyd_moe_down_indexed_r4.hip   (R=2 and R=4)
// and asserts the per-(row, krank) contribution `scale * acc` is BITWISE equal
// across all three. This is a CPU check of the index arithmetic (group->slot
// assignment, LDS slot mapping, x offsets, row clamping, reduction tree) — the
// places a silent indexing bug would hide. It is NOT a GPU measurement and says
// nothing about performance.
//
// build: g++ -O2 -std=c++17 parity_sim.cpp -o parity_sim && ./parity_sim

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>

static const int WAVE = 32;

// ---- fp16 (IEEE binary16) -> float, matching __half2float ----
static float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) { out = sign; }
        else {
            int e = -1;
            uint32_t m = man;
            do { e++; m <<= 1; } while ((m & 0x400) == 0);
            out = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FF) << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

struct Model {
    int M, K, ktop, groups;
    std::vector<uint8_t> W;   // [n_exp][M][groups][112]
    std::vector<float> x;     // [ktop][K]
    std::vector<float> wgt;   // [ktop]
    std::vector<int> idx;     // [ktop]
    int n_exp;
    const uint8_t* rowptr(int e, int row) const {
        return W.data() + ((size_t)e * M + row) * groups * 112;
    }
};

static uint32_t pk3(const uint8_t* gp, int boff) {
    const uint8_t* d = gp + 16 + boff;
    return (uint32_t)d[0] | ((uint32_t)d[1] << 8) | ((uint32_t)d[2] << 16);
}

// Identical product tree to DOG3_LDS / mq3_down_dog3.
static float dog3(const float* cb, uint32_t pk, const float* X) {
    return cb[ pk        & 7u] * X[0]
         + cb[(pk >>  3) & 7u] * X[1]
         + cb[(pk >>  6) & 7u] * X[2]
         + cb[(pk >>  9) & 7u] * X[3]
         + cb[(pk >> 12) & 7u] * X[4]
         + cb[(pk >> 15) & 7u] * X[5]
         + cb[(pk >> 18) & 7u] * X[6]
         + cb[(pk >> 21) & 7u] * X[7];
}

// __shfl_down with width=32: out-of-range source returns the lane's own value.
static float wave_reduce(float v[WAVE]) {
    float a[WAVE];
    std::memcpy(a, v, sizeof(a));
    for (int off = 16; off > 0; off >>= 1) {
        float t[WAVE];
        for (int i = 0; i < WAVE; i++) t[i] = a[i] + ((i + off < WAVE) ? a[i + off] : a[i]);
        std::memcpy(a, t, sizeof(a));
    }
    return a[0];
}

// ---------------- incumbent (R=1, quads + tail, cb_lds[32] reused) ----------
static float incumbent(const Model& m, int row, int krank) {
    const uint8_t* A = m.rowptr(m.idx[krank], 0);
    const uint8_t* rp = A + (size_t)row * m.groups * 112;
    const float* x = m.x.data() + (size_t)krank * m.K;
    float acc[WAVE][4];
    for (int l = 0; l < WAVE; l++) for (int s = 0; s < 4; s++) acc[l][s] = 0.0f;
    float cb_lds[32];
    const int quads = m.groups >> 2, tail = m.groups & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        for (int tid = 0; tid < WAVE; tid++) {          // cooperative load
            const uint8_t* gp = rp + (size_t)(g + (tid >> 3)) * 112;
            cb_lds[tid] = h2f(*(const uint16_t*)(gp + 2 * (tid & 7)));
        }
        for (int tid = 0; tid < WAVE; tid++) {
            for (int j = 0; j < 4; j++) {
                uint32_t pk = pk3(rp + (size_t)(g + j) * 112, tid * 3);
                acc[tid][j] += dog3(cb_lds + j * 8, pk, x + (g + j) * 256 + tid * 8);
            }
        }
    }
    for (int t = 0; t < tail; t++) {
        const int g = (quads << 2) + t;
        for (int tid = 0; tid < 8; tid++)               // only 8 lanes load
            cb_lds[tid] = h2f(*(const uint16_t*)(rp + (size_t)g * 112 + 2 * tid));
        for (int tid = 0; tid < WAVE; tid++) {
            uint32_t pk = pk3(rp + (size_t)g * 112, tid * 3);
            acc[tid][t] += dog3(cb_lds, pk, x + g * 256 + tid * 8);
        }
    }
    float red[WAVE];
    for (int l = 0; l < WAVE; l++)
        red[l] = (acc[l][0] + acc[l][1]) + (acc[l][2] + acc[l][3]);
    return m.wgt[krank] * wave_reduce(red);
}

// ---------------- row tile (R=2 / R=4), disjoint LDS slots ------------------
static void rowtile(const Model& m, int row_base, int krank, int R, float* out) {
    const uint8_t* A = m.rowptr(m.idx[krank], 0);
    const float* x = m.x.data() + (size_t)krank * m.K;
    std::vector<const uint8_t*> rp(R);
    std::vector<char> live(R);
    for (int r = 0; r < R; r++) {
        live[r] = (row_base + r) < m.M;
        int rr = live[r] ? (row_base + r) : row_base;
        rp[r] = A + (size_t)rr * m.groups * 112;
    }
    std::vector<std::vector<std::array<float, 4>>> dummy;  // unused
    std::vector<float> acc((size_t)WAVE * R * 4, 0.0f);
    auto AC = [&](int l, int r, int s) -> float& { return acc[((size_t)l * R + r) * 4 + s]; };
    std::vector<float> cb_lds((size_t)R * 32);

    for (int g = 0; g < m.groups; g += 4) {
        const int n = (m.groups - g) < 4 ? (m.groups - g) : 4;
        for (int tid = 0; tid < WAVE; tid++) {          // cooperative fill
            const int j_ = tid >> 3, e_ = tid & 7;
            if (j_ < n)
                for (int r = 0; r < R; r++)
                    cb_lds[r * 32 + j_ * 8 + e_] =
                        h2f(*(const uint16_t*)(rp[r] + (size_t)(g + j_) * 112 + 2 * e_));
        }
        for (int j = 0; j < 4; j++) {
            if (j >= n) continue;
            for (int tid = 0; tid < WAVE; tid++) {
                const float* X = x + (g + j) * 256 + tid * 8;
                for (int r = 0; r < R; r++) {
                    uint32_t pk = pk3(rp[r] + (size_t)(g + j) * 112, tid * 3);
                    AC(tid, r, j) += dog3(&cb_lds[r * 32 + j * 8], pk, X);
                }
            }
        }
    }
    for (int r = 0; r < R; r++) {
        float red[WAVE];
        for (int l = 0; l < WAVE; l++)
            red[l] = (AC(l, r, 0) + AC(l, r, 1)) + (AC(l, r, 2) + AC(l, r, 3));
        out[r] = live[r] ? m.wgt[krank] * wave_reduce(red) : NAN;
    }
}

static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static int run_shape(int M, int K, int ktop, unsigned seed) {
    Model m;
    m.M = M; m.K = K; m.ktop = ktop; m.groups = K / 256; m.n_exp = 8;
    srand(seed);
    m.W.resize((size_t)m.n_exp * M * m.groups * 112);
    for (auto& b : m.W) b = (uint8_t)(rand() & 0xFF);
    // Keep codebook headers finite: overwrite each group's 16-byte header with
    // small well-formed fp16 values.
    for (int e = 0; e < m.n_exp; e++)
        for (int r = 0; r < M; r++)
            for (int g = 0; g < m.groups; g++) {
                uint8_t* gp = m.W.data() + (((size_t)e * M + r) * m.groups + g) * 112;
                for (int t = 0; t < 8; t++) {
                    uint16_t h = (uint16_t)(0x3000 + (rand() & 0x0FFF));
                    if (rand() & 1) h |= 0x8000;
                    std::memcpy(gp + 2 * t, &h, 2);
                }
            }
    m.x.resize((size_t)ktop * K);
    for (auto& v : m.x) v = (float)((rand() % 2001) - 1000) / 512.0f;
    m.wgt.resize(ktop);
    for (auto& v : m.wgt) v = (float)(rand() % 1000 + 1) / 1000.0f;
    m.idx.resize(ktop);
    for (auto& v : m.idx) v = rand() % m.n_exp;

    int bad = 0, checked = 0;
    for (int R : {2, 4}) {
        for (int krank = 0; krank < ktop; krank++) {
            for (int rb = 0; rb < M; rb += R) {
                std::vector<float> got(R);
                rowtile(m, rb, krank, R, got.data());
                for (int r = 0; r < R; r++) {
                    if (rb + r >= M) continue;
                    float want = incumbent(m, rb + r, krank);
                    checked++;
                    if (bits(got[r]) != bits(want)) {
                        if (bad < 5)
                            printf("  MISMATCH R=%d row=%d krank=%d: got %.9g (0x%08x) "
                                   "want %.9g (0x%08x)\n",
                                   R, rb + r, krank, got[r], bits(got[r]), want, bits(want));
                        bad++;
                    }
                }
            }
        }
    }
    printf("  M=%-5d K=%-5d ktop=%d groups=%d : %d comparisons, %d bitwise mismatches -> %s\n",
           M, K, ktop, m.groups, checked, bad, bad ? "FAIL" : "BIT-EXACT");
    return bad;
}

int main() {
    int bad = 0;
    printf("MQ3-Lloyd MoE-down row-tile bit-parity (host simulation, no GPU)\n");
    bad += run_shape(2048, 512, 8, 1);    // a3b decode shape (groups=2, tail-only)
    bad += run_shape(64, 1024, 4, 2);     // groups=4, exactly one quad
    bad += run_shape(48, 1536, 3, 3);     // groups=6, quad + 2-group tail
    bad += run_shape(40, 2048, 2, 4);     // groups=8, two quads
    bad += run_shape(35, 512, 2, 5);      // M not divisible by 4 -> partial tile
    bad += run_shape(33, 768, 2, 6);      // groups=3 (odd tail) + partial tile
    printf("%s\n", bad ? "OVERALL: FAIL" : "OVERALL: all shapes BIT-EXACT vs incumbent");
    return bad != 0;
}
