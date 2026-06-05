#!/usr/bin/env bash
# HipFire Tiered Validation Harness
# Run after each approach attempt. Records which tiers pass.
# Usage: ./harness.sh [approach_name]

set -euo pipefail

APPROACH="${1:-baseline}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="experiments"
RESULT_FILE="${RESULTS_DIR}/${APPROACH}_${TIMESTAMP}.md"

mkdir -p "$RESULTS_DIR"

PASS=0
FAIL=0
MAX_TIER=-1
HIP_ARCH=""

log() { echo "$1" | tee -a "$RESULT_FILE"; }
pass() { log "  ✅ PASS: $1"; PASS=$((PASS+1)); MAX_TIER=$2; }
fail() { log "  ❌ FAIL: $1"; FAIL=$((FAIL+1)); }

log "# Harness Results: ${APPROACH}"
log "**Timestamp:** ${TIMESTAMP}"
log "**Machine:** $(hostname)"
log ""

# ═══════════════════════════════════════════════════════════════
# TIER 0: Kernel driver loaded
# ═══════════════════════════════════════════════════════════════
log "## Tier 0: Kernel Driver"

if lsmod 2>/dev/null | grep amdgpu >/dev/null 2>&1; then
    pass "amdgpu module loaded" 0
else
    fail "amdgpu module NOT loaded"
    log ""
    log "**Max Tier Reached: -1 (driver not loaded)**"
    exit 1
fi

RENDER_NODES=$(ls /dev/dri/renderD* 2>/dev/null || true)
if [ -n "$RENDER_NODES" ]; then
    pass "render nodes exist: ${RENDER_NODES}" 0
else
    fail "no render nodes in /dev/dri/"
fi

if [ -e /dev/kfd ]; then
    pass "KFD node exists: /dev/kfd" 0
else
    fail "no KFD node at /dev/kfd"
fi

DMESG_ERRORS=$(dmesg 2>/dev/null | grep -i amdgpu | grep -ci "error\|fault\|fail" || true)
if [ "$DMESG_ERRORS" -lt 3 ]; then
    pass "dmesg amdgpu errors: ${DMESG_ERRORS} (acceptable)" 0
else
    fail "dmesg amdgpu errors: ${DMESG_ERRORS} (excessive)"
fi

log ""

# ═══════════════════════════════════════════════════════════════
# TIER 1: Userspace runtime sees the card
# ═══════════════════════════════════════════════════════════════
log "## Tier 1: Userspace Detection"

# Try rocm-smi first
if command -v rocm-smi &>/dev/null; then
    if rocm-smi --showid 2>/dev/null | grep -i "gpu\|device" >/dev/null 2>&1; then
        pass "rocm-smi detects GPU" 1
    else
        fail "rocm-smi exists but doesn't detect GPU"
    fi
else
    log "  ⚠️  rocm-smi not installed (skipping, not fatal)"
fi

# Try rocminfo
if command -v rocminfo &>/dev/null; then
    ROCMINFO_OUT=$(rocminfo 2>/dev/null || true)
    GFX_ID=$(echo "$ROCMINFO_OUT" | sed -nE 's/.*Name:[[:space:]]*(gfx[0-9a-z]+).*/\1/p; s/.*amdgcn-amd-amdhsa--(gfx[0-9a-z]+).*/\1/p' | head -1)
    if [ -n "$GFX_ID" ]; then
        HIP_ARCH="$GFX_ID"
        pass "rocminfo detects GPU: ${GFX_ID}" 1
    else
        fail "rocminfo exists but doesn't detect an AMD GPU ISA"
    fi
else
    log "  ⚠️  rocminfo not installed (skipping)"
fi

log ""

# ═══════════════════════════════════════════════════════════════
# TIER 2: Compute runtime initializes
# ═══════════════════════════════════════════════════════════════
log "## Tier 2: Compute Runtime Init"

# HIP runtime check
if command -v hipcc &>/dev/null; then
    HIP_TEST=$(cat <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
int main() {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err == hipSuccess && count > 0) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        printf("HIP OK: %s (%s)\n", prop.name, prop.gcnArchName);
        return 0;
    }
    printf("HIP FAIL: err=%d count=%d\n", err, count);
    return 1;
}
EOF
)
    TMPDIR=$(mktemp -d)
    echo "$HIP_TEST" > "$TMPDIR/test.cpp"
    if hipcc "$TMPDIR/test.cpp" -o "$TMPDIR/test" 2>/dev/null; then
        HIP_OUT=$("$TMPDIR/test" 2>&1 || true)
        if echo "$HIP_OUT" | grep -q "HIP OK"; then
            HIP_ARCH_FROM_HIP=$(echo "$HIP_OUT" | sed -nE 's/.*\((gfx[0-9a-z]+)\).*/\1/p' | head -1)
            if [ -n "$HIP_ARCH_FROM_HIP" ]; then
                HIP_ARCH="$HIP_ARCH_FROM_HIP"
            fi
            pass "HIP runtime initialized: ${HIP_OUT}" 2
        else
            fail "HIP compiled but runtime failed: ${HIP_OUT}"
        fi
    else
        fail "hipcc compilation failed"
    fi
    rm -rf "$TMPDIR"
else
    log "  ⚠️  hipcc not found"
fi

log ""

# ═══════════════════════════════════════════════════════════════
# TIER 3: GPU memory allocation and data transfer
# ═══════════════════════════════════════════════════════════════
log "## Tier 3: Memory Operations"

if command -v hipcc &>/dev/null; then
    MEMTEST=$(cat <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>
int main() {
    float *d_buf, h_buf[1024], h_out[1024];
    for (int i = 0; i < 1024; i++) h_buf[i] = (float)i;
    memset(h_out, 0, sizeof(h_out));
    hipError_t e1 = hipMalloc(&d_buf, 1024 * sizeof(float));
    if (e1 != hipSuccess) {
        printf("MEMTEST: alloc=%d (%s)\n", e1, hipGetErrorString(e1));
        return 1;
    }
    hipError_t e2 = hipMemcpy(d_buf, h_buf, 1024 * sizeof(float), hipMemcpyHostToDevice);
    if (e2 != hipSuccess) {
        printf("MEMTEST: h2d=%d (%s)\n", e2, hipGetErrorString(e2));
        hipFree(d_buf);
        return 1;
    }
    hipError_t e3 = hipMemcpy(h_out, d_buf, 1024 * sizeof(float), hipMemcpyDeviceToHost);
    if (e3 != hipSuccess) {
        printf("MEMTEST: d2h=%d (%s)\n", e3, hipGetErrorString(e3));
        hipFree(d_buf);
        return 1;
    }
    hipFree(d_buf);
    int correct = (memcmp(h_buf, h_out, 1024 * sizeof(float)) == 0);
    printf("MEMTEST: alloc=%d h2d=%d d2h=%d verify=%s\n", e1, e2, e3, correct ? "PASS" : "FAIL");
    return correct ? 0 : 1;
}
EOF
)
    TMPDIR=$(mktemp -d)
    echo "$MEMTEST" > "$TMPDIR/memtest.cpp"
    if hipcc "$TMPDIR/memtest.cpp" -o "$TMPDIR/memtest" 2>/dev/null; then
        MEM_OUT=$("$TMPDIR/memtest" 2>&1 || true)
        if echo "$MEM_OUT" | grep -q "verify=PASS"; then
            pass "GPU memory round-trip verified: ${MEM_OUT}" 3
        else
            fail "Memory test failed: ${MEM_OUT}"
        fi
    else
        fail "Memory test compilation failed"
    fi
    rm -rf "$TMPDIR"
else
    log "  ⚠️  hipcc not found — skipping HIP memory test"
fi

log ""

# ═══════════════════════════════════════════════════════════════
# TIER 4: Simple compute kernel executes correctly
# ═══════════════════════════════════════════════════════════════
log "## Tier 4: Compute Kernel Execution"

if command -v hipcc &>/dev/null; then
    KERNEL_TEST=$(cat <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CHECK_HIP(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        printf("KERNEL: %s failed: %d (%s), result=FAIL\n", #call, err, hipGetErrorString(err)); \
        return 1; \
    } \
} while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 65536;
    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));
    if (!h_a || !h_b || !h_c) {
        printf("KERNEL: host allocation failed, result=FAIL\n");
        return 1;
    }
    for (int i = 0; i < N; i++) { h_a[i] = (float)i; h_b[i] = (float)(N - i); }

    CHECK_HIP(hipMalloc(&d_a, N * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_b, N * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_c, N * sizeof(float)));
    CHECK_HIP(hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b, h_b, N * sizeof(float), hipMemcpyHostToDevice));

    vector_add<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(h_c, d_c, N * sizeof(float), hipMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_c[i] - (float)N) > 0.001f) errors++;
    }
    printf("KERNEL: vector_add %d elements, errors=%d, result=%s\n", N, errors, errors == 0 ? "PASS" : "FAIL");

    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return errors == 0 ? 0 : 1;
}
EOF
)
    TMPDIR=$(mktemp -d)
    echo "$KERNEL_TEST" > "$TMPDIR/kernel.cpp"
    ARCH_FLAGS=()
    if [ -n "$HIP_ARCH" ]; then
        ARCH_FLAGS=(--offload-arch="$HIP_ARCH")
    fi
    if hipcc "$TMPDIR/kernel.cpp" -o "$TMPDIR/kernel" "${ARCH_FLAGS[@]}" 2>/dev/null; then
        K_OUT=$("$TMPDIR/kernel" 2>&1 || true)
        if echo "$K_OUT" | grep -q "result=PASS"; then
            if [ -n "$HIP_ARCH" ]; then
                pass "Compute kernel correct (${HIP_ARCH}): ${K_OUT}" 4
            else
                pass "Compute kernel correct: ${K_OUT}" 4
            fi
        else
            fail "Compute kernel wrong results: ${K_OUT}"
        fi
    elif [ -n "$HIP_ARCH" ] && hipcc "$TMPDIR/kernel.cpp" -o "$TMPDIR/kernel" 2>/dev/null; then
        K_OUT=$("$TMPDIR/kernel" 2>&1 || true)
        if echo "$K_OUT" | grep -q "result=PASS"; then
            pass "Compute kernel correct (default arch): ${K_OUT}" 4
        else
            fail "Compute kernel wrong results: ${K_OUT}"
        fi
    else
        fail "Kernel compilation failed${HIP_ARCH:+ for ${HIP_ARCH}}"
    fi
    rm -rf "$TMPDIR"
else
    log "  ⚠️  hipcc not found — skipping kernel test"
fi

log ""

# ═══════════════════════════════════════════════════════════════
# TIER 5: Matrix multiply kernel
# ═══════════════════════════════════════════════════════════════
log "## Tier 5: Matrix Multiply"

if command -v hipcc &>/dev/null && [ "$MAX_TIER" -ge 4 ]; then
    MATMUL_TEST=$(cat <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CHECK_HIP(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        printf("MATMUL: %s failed: %d (%s), result=FAIL\n", #call, err, hipGetErrorString(err)); \
        return 1; \
    } \
} while (0)

__global__ void naive_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 512, N = 512, K = 512;
    size_t sA = M*K*sizeof(float), sB = K*N*sizeof(float), sC = M*N*sizeof(float);
    float *hA = (float*)malloc(sA), *hB = (float*)malloc(sB);
    float *hC = (float*)malloc(sC), *hRef = (float*)malloc(sC);
    float *dA, *dB, *dC;
    if (!hA || !hB || !hC || !hRef) {
        printf("MATMUL: host allocation failed, result=FAIL\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < M*K; i++) hA[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K*N; i++) hB[i] = (float)(rand() % 100) / 100.0f;

    // CPU reference
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += hA[i*K+k] * hB[k*N+j];
            hRef[i*N+j] = s;
        }

    CHECK_HIP(hipMalloc(&dA, sA));
    CHECK_HIP(hipMalloc(&dB, sB));
    CHECK_HIP(hipMalloc(&dC, sC));
    CHECK_HIP(hipMemcpy(dA, hA, sA, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, hB, sB, hipMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N+15)/16, (M+15)/16);

    // Warmup
    naive_matmul<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    // Timed run
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < 10; i++)
        naive_matmul<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    float ms = 0;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));

    CHECK_HIP(hipMemcpy(hC, dC, sC, hipMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < M*N; i++) {
        if (fabsf(hC[i] - hRef[i]) > 0.1f) errors++;
    }

    float gflops = (2.0f * M * N * K * 10) / (ms / 1000.0f) / 1e9;
    printf("MATMUL: %dx%dx%d, errors=%d, time=%.2fms/10iter, %.2f GFLOPS, result=%s\n",
           M, N, K, errors, ms, gflops, errors == 0 ? "PASS" : "FAIL");

    hipFree(dA); hipFree(dB); hipFree(dC);
    free(hA); free(hB); free(hC); free(hRef);
    return errors == 0 ? 0 : 1;
}
EOF
)
    TMPDIR=$(mktemp -d)
    echo "$MATMUL_TEST" > "$TMPDIR/matmul.cpp"
    ARCH_FLAGS=()
    if [ -n "$HIP_ARCH" ]; then
        ARCH_FLAGS=(--offload-arch="$HIP_ARCH")
    fi
    if hipcc "$TMPDIR/matmul.cpp" -o "$TMPDIR/matmul" "${ARCH_FLAGS[@]}" 2>/dev/null || \
       { [ -n "$HIP_ARCH" ] && hipcc "$TMPDIR/matmul.cpp" -o "$TMPDIR/matmul" 2>/dev/null; }; then
        MM_OUT=$("$TMPDIR/matmul" 2>&1 || true)
        if echo "$MM_OUT" | grep -q "result=PASS"; then
            pass "Matmul correct${HIP_ARCH:+ (${HIP_ARCH})}: ${MM_OUT}" 5
        else
            fail "Matmul failed: ${MM_OUT}"
        fi
    else
        fail "Matmul compilation failed"
    fi
    rm -rf "$TMPDIR"
else
    log "  ⚠️  Skipping (requires Tier 4 pass + hipcc)"
fi

log ""

# ═══════════════════════════════════════════════════════════════
# TIER 6: Performance benchmarks
# ═══════════════════════════════════════════════════════════════
log "## Tier 6: Performance"

if [ "$MAX_TIER" -ge 5 ]; then
    log "  Tier 5 passed — performance benchmarks would run here."
    log "  Detected HIP arch: ${HIP_ARCH:-unknown}"
    log "  TODO: bandwidth test (memcpy), roofline analysis, optimized GEMV"
    pass "Performance tier reached (detailed benchmarks TBD)" 6
else
    log "  ⚠️  Skipping (requires Tier 5 pass)"
fi

log ""

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
log "---"
log "## Summary"
log "**Approach:** ${APPROACH}"
log "**Max Tier Reached:** ${MAX_TIER}"
log "**Passed:** ${PASS}"
log "**Failed:** ${FAIL}"
log "**Result File:** ${RESULT_FILE}"

echo ""
echo "═══════════════════════════════════════════"
echo "  MAX TIER: ${MAX_TIER} | PASS: ${PASS} | FAIL: ${FAIL}"
echo "═══════════════════════════════════════════"
