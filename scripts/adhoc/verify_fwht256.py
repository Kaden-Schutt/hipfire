#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""
Verify FWHT-256 shuffle kernel correctness.

Simulates the 32-thread x 8-element register+shuffle decomposition
and compares against the scalar triple-loop reference implementation.
Also validates round-trip (forward then inverse = identity up to scale).
"""
import numpy as np

def gen_fwht_signs(seed, n):
    """Match the Rust gen_fwht_signs exactly (LCG, bit 16)."""
    state = seed & 0x7fffffff
    signs = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0x7fffffff
        signs.append(1.0 if ((state >> 16) & 1) == 1 else -1.0)
    return np.array(signs, dtype=np.float64)

def fwht_scalar_forward_256(x, signs1, signs2):
    """Reference scalar FWHT-256: signs1, butterfly, scale, signs2."""
    x = x.copy() * signs1
    n = 256
    stride = 1
    while stride < n:
        for i in range(0, n, stride * 2):
            for j in range(stride):
                a, b = x[i+j], x[i+j+stride]
                x[i+j]        = a + b
                x[i+j+stride] = a - b
        stride <<= 1
    x *= 1.0 / np.sqrt(256.0)
    x *= signs2
    return x

def fwht_scalar_inverse_256(x, signs1, signs2):
    """Reference scalar inverse FWHT-256: signs2, butterfly, scale*signs1."""
    x = x.copy() * signs2
    n = 256
    stride = 1
    while stride < n:
        for i in range(0, n, stride * 2):
            for j in range(stride):
                a, b = x[i+j], x[i+j+stride]
                x[i+j]        = a + b
                x[i+j+stride] = a - b
        stride <<= 1
    x *= (1.0 / np.sqrt(256.0)) * signs1
    return x

def fwht_shfl_forward_256(x, signs1, signs2):
    """
    Simulate the 32-thread shuffle kernel.
    Thread tid owns elements [tid*8 .. tid*8+7].
    """
    N_THREADS = 32
    EPT = 8  # elements per thread
    x = x.copy()

    # Distribute into per-thread registers
    regs = []
    for tid in range(N_THREADS):
        d0 = tid * EPT
        v = x[d0:d0+EPT].copy()
        # Apply signs1
        v *= signs1[d0:d0+EPT]
        regs.append(v)

    # Pass 1: stride 1 (pairs 0-1, 2-3, 4-5, 6-7)
    for tid in range(N_THREADS):
        v = regs[tid]
        for base in range(0, EPT, 2):
            a, b = v[base], v[base+1]
            v[base]   = a + b
            v[base+1] = a - b

    # Pass 2: stride 2 (pairs 0-2, 1-3, 4-6, 5-7)
    for tid in range(N_THREADS):
        v = regs[tid]
        for base in range(0, EPT, 4):
            for j in range(2):
                a, b = v[base+j], v[base+j+2]
                v[base+j]   = a + b
                v[base+j+2] = a - b

    # Pass 3: stride 4 (pairs 0-4, 1-5, 2-6, 3-7)
    for tid in range(N_THREADS):
        v = regs[tid]
        for j in range(4):
            a, b = v[j], v[j+4]
            v[j]   = a + b
            v[j+4] = a - b

    # Passes 4-8: warp shuffle (__shfl_xor)
    # Thread strides: 1, 2, 4, 8, 16
    for ts in [1, 2, 4, 8, 16]:
        new_regs = [r.copy() for r in regs]
        for tid in range(N_THREADS):
            partner = tid ^ ts
            for j in range(EPT):
                if tid & ts:
                    new_regs[tid][j] = regs[partner][j] - regs[tid][j]
                else:
                    new_regs[tid][j] = regs[tid][j] + regs[partner][j]
        regs = new_regs

    # Scale and apply signs2
    s = 1.0 / np.sqrt(256.0)
    for tid in range(N_THREADS):
        d0 = tid * EPT
        regs[tid] *= s * signs2[d0:d0+EPT]

    # Reassemble
    result = np.zeros(256, dtype=np.float64)
    for tid in range(N_THREADS):
        d0 = tid * EPT
        result[d0:d0+EPT] = regs[tid]
    return result

def fwht_shfl_inverse_256(x, signs1, signs2):
    """
    Simulate the 32-thread inverse shuffle kernel.
    """
    N_THREADS = 32
    EPT = 8
    x = x.copy()

    regs = []
    for tid in range(N_THREADS):
        d0 = tid * EPT
        v = x[d0:d0+EPT].copy()
        v *= signs2[d0:d0+EPT]
        regs.append(v)

    # Passes 4-8 (shuffle, same as forward)
    for ts in [1, 2, 4, 8, 16]:
        new_regs = [r.copy() for r in regs]
        for tid in range(N_THREADS):
            partner = tid ^ ts
            for j in range(EPT):
                if tid & ts:
                    new_regs[tid][j] = regs[partner][j] - regs[tid][j]
                else:
                    new_regs[tid][j] = regs[tid][j] + regs[partner][j]
        regs = new_regs

    # Pass 3 reverse: stride 4
    for tid in range(N_THREADS):
        v = regs[tid]
        for j in range(4):
            a, b = v[j], v[j+4]
            v[j]   = a + b
            v[j+4] = a - b

    # Pass 2 reverse: stride 2
    for tid in range(N_THREADS):
        v = regs[tid]
        for base in range(0, EPT, 4):
            for j in range(2):
                a, b = v[base+j], v[base+j+2]
                v[base+j]   = a + b
                v[base+j+2] = a - b

    # Pass 1 reverse: stride 1
    for tid in range(N_THREADS):
        v = regs[tid]
        for base in range(0, EPT, 2):
            a, b = v[base], v[base+1]
            v[base]   = a + b
            v[base+1] = a - b

    # Scale and apply signs1
    s = 1.0 / np.sqrt(256.0)
    for tid in range(N_THREADS):
        d0 = tid * EPT
        regs[tid] *= s * signs1[d0:d0+EPT]

    result = np.zeros(256, dtype=np.float64)
    for tid in range(N_THREADS):
        d0 = tid * EPT
        result[d0:d0+EPT] = regs[tid]
    return result


def main():
    np.random.seed(7)
    signs1 = gen_fwht_signs(42, 256)
    signs2 = gen_fwht_signs(1042, 256)

    print("=== FWHT-256 Verification ===\n")

    # Test 1: forward shuffle == forward scalar
    print("Test 1: Forward shuffle matches scalar reference")
    for trial in range(10):
        x = np.random.randn(256)
        ref = fwht_scalar_forward_256(x, signs1, signs2)
        shfl = fwht_shfl_forward_256(x, signs1, signs2)
        err = np.max(np.abs(ref - shfl))
        assert err < 1e-10, f"Trial {trial}: max error {err}"
    print(f"  PASS (10 trials, max element error < 1e-10)\n")

    # Test 2: inverse shuffle == inverse scalar
    print("Test 2: Inverse shuffle matches scalar reference")
    for trial in range(10):
        x = np.random.randn(256)
        ref = fwht_scalar_inverse_256(x, signs1, signs2)
        shfl = fwht_shfl_inverse_256(x, signs1, signs2)
        err = np.max(np.abs(ref - shfl))
        assert err < 1e-10, f"Trial {trial}: max error {err}"
    print(f"  PASS (10 trials, max element error < 1e-10)\n")

    # Test 3: round-trip forward then inverse = identity
    print("Test 3: Round-trip (forward then inverse) = identity")
    for trial in range(10):
        x = np.random.randn(256)
        fwd = fwht_shfl_forward_256(x, signs1, signs2)
        roundtrip = fwht_shfl_inverse_256(fwd, signs1, signs2)
        err = np.max(np.abs(x - roundtrip))
        assert err < 1e-10, f"Trial {trial}: max error {err}"
    print(f"  PASS (10 trials, max round-trip error < 1e-10)\n")

    # Test 4: Hadamard property - verify WHT of unit vector
    print("Test 4: WHT of e_0 (all elements should be +/- 1/sqrt(256))")
    e0 = np.zeros(256); e0[0] = 1.0
    # Use identity signs to check raw WHT
    ones = np.ones(256)
    fwd = fwht_shfl_forward_256(e0, ones, ones)
    expected_mag = 1.0 / np.sqrt(256.0)
    assert np.allclose(np.abs(fwd), expected_mag, atol=1e-14)
    print(f"  PASS (all magnitudes = {expected_mag:.6f})\n")

    # Test 5: Parseval's theorem (energy preservation)
    print("Test 5: Parseval's theorem (||x||^2 == ||WHT(x)||^2)")
    for trial in range(10):
        x = np.random.randn(256)
        fwd = fwht_shfl_forward_256(x, signs1, signs2)
        e_in = np.sum(x**2)
        e_out = np.sum(fwd**2)
        rel_err = abs(e_in - e_out) / e_in
        assert rel_err < 1e-12, f"Trial {trial}: relative energy error {rel_err}"
    print(f"  PASS (10 trials, relative energy error < 1e-12)\n")

    # Test 6: Verify butterfly pass indexing by checking known small cases
    print("Test 6: Verify pass structure against textbook 8-point WHT")
    # An 8-point WHT of [1,0,0,0,0,0,0,0] should give [1,1,1,1,1,1,1,1]/sqrt(8)
    # Simulate just the local passes (no shuffle) for one thread
    v = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    # Pass 1: stride 1
    v[0], v[1] = v[0]+v[1], v[0]-v[1]
    v[2], v[3] = v[2]+v[3], v[2]-v[3]
    v[4], v[5] = v[4]+v[5], v[4]-v[5]
    v[6], v[7] = v[6]+v[7], v[6]-v[7]
    # Pass 2: stride 2
    v[0], v[2] = v[0]+v[2], v[0]-v[2]
    v[1], v[3] = v[1]+v[3], v[1]-v[3]
    v[4], v[6] = v[4]+v[6], v[4]-v[6]
    v[5], v[7] = v[5]+v[7], v[5]-v[7]
    # Pass 3: stride 4
    v[0], v[4] = v[0]+v[4], v[0]-v[4]
    v[1], v[5] = v[1]+v[5], v[1]-v[5]
    v[2], v[6] = v[2]+v[6], v[2]-v[6]
    v[3], v[7] = v[3]+v[7], v[3]-v[7]
    assert np.allclose(v, np.ones(8)), f"8-point local WHT failed: {v}"
    print(f"  PASS (8-point WHT of e0 = all-ones)\n")

    print("=== All 6 tests passed. Butterfly indexing verified. ===")


if __name__ == "__main__":
    main()
