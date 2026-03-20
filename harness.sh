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
    if echo "$ROCMINFO_OUT" | grep -i "gfx10" >/dev/null 2>&1; then
        GFX_ID=$(echo "$ROCMINFO_OUT" | grep -oP "gfx\d+" | head -1)
        pass "rocminfo detects GPU: ${GFX_ID}" 1
    else
        fail "rocminfo exists but doesn't detect gfx10 GPU"
    fi
else
    log "  ⚠️  rocminfo not installed (skipping)"
fi

# Vulkan detection (should always work via RADV)
if command -v vulkaninfo &>/dev/null; then
    if vulkaninfo --summary 2>/dev/null | grep -i "radeon\|navi\|5700" >/dev/null 2>&1; then
        pass "Vulkan detects GPU via RADV" 1
    else
        fail "vulkaninfo doesn't detect AMD GPU"
    fi
elif command -v vkcube &>/dev/null; then
    log "  ⚠️  vulkaninfo not found, but vkcube exists"
else
    log "  ⚠️  no Vulkan tools installed"
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

# Vulkan compute check
if command -v vulkaninfo &>/dev/null; then
    COMPUTE_QUEUES=$(vulkaninfo 2>/dev/null | grep -c "COMPUTE" || true)
    if [ "$COMPUTE_QUEUES" -gt 0 ]; then
        pass "Vulkan compute queues available: ${COMPUTE_QUEUES}" 2
    else
        fail "No Vulkan compute queues detected"
    fi
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
    hipError_t e1 = hipMalloc(&d_buf, 1024 * sizeof(float));
    hipError_t e2 = hipMemcpy(d_buf, h_buf, 1024 * sizeof(float), hipMemcpyHostToDevice);
    hipError_t e3 = hipMemcpy(h_out, d_buf, 1024 * sizeof(float), hipMemcpyDeviceToHost);
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
    for (int i = 0; i < N; i++) { h_a[i] = (float)i; h_b[i] = (float)(N - i); }

    hipMalloc(&d_a, N * sizeof(float));
    hipMalloc(&d_b, N * sizeof(float));
    hipMalloc(&d_c, N * sizeof(float));
    hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, N * sizeof(float), hipMemcpyHostToDevice);

    vector_add<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
    hipDeviceSynchronize();

    hipMemcpy(h_c, d_c, N * sizeof(float), hipMemcpyDeviceToHost);

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
    if hipcc "$TMPDIR/kernel.cpp" -o "$TMPDIR/kernel" --offload-arch=gfx1010 2>/dev/null; then
        K_OUT=$("$TMPDIR/kernel" 2>&1 || true)
        if echo "$K_OUT" | grep -q "result=PASS"; then
            pass "Compute kernel correct: ${K_OUT}" 4
        else
            fail "Compute kernel wrong results: ${K_OUT}"
        fi
    elif hipcc "$TMPDIR/kernel.cpp" -o "$TMPDIR/kernel" 2>/dev/null; then
        K_OUT=$("$TMPDIR/kernel" 2>&1 || true)
        if echo "$K_OUT" | grep -q "result=PASS"; then
            pass "Compute kernel correct (default arch): ${K_OUT}" 4
        else
            fail "Compute kernel wrong results: ${K_OUT}"
        fi
    else
        fail "Kernel compilation failed for gfx1010"
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

    hipMalloc(&dA, sA); hipMalloc(&dB, sB); hipMalloc(&dC, sC);
    hipMemcpy(dA, hA, sA, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, sB, hipMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N+15)/16, (M+15)/16);

    // Warmup
    naive_matmul<<<grid, block>>>(dA, dB, dC, M, N, K);
    hipDeviceSynchronize();

    // Timed run
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
    hipEventRecord(start);
    for (int i = 0; i < 10; i++)
        naive_matmul<<<grid, block>>>(dA, dB, dC, M, N, K);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);

    hipMemcpy(hC, dC, sC, hipMemcpyDeviceToHost);

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
    if hipcc "$TMPDIR/matmul.cpp" -o "$TMPDIR/matmul" --offload-arch=gfx1010 2>/dev/null || \
       hipcc "$TMPDIR/matmul.cpp" -o "$TMPDIR/matmul" 2>/dev/null; then
        MM_OUT=$("$TMPDIR/matmul" 2>&1 || true)
        if echo "$MM_OUT" | grep -q "result=PASS"; then
            pass "Matmul correct: ${MM_OUT}" 5
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
    log "  RX 5700 XT theoretical: ~9.75 TFLOPS FP32, ~448 GB/s bandwidth"
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
