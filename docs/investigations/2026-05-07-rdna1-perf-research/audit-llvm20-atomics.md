# Audit: LLVM 20 atomic fadd intrinsic scan

**Date:** 2026-05-07
**Trigger:** LLVM 20.1.0 removed `llvm.amdgcn.{flat,global}.atomic.fadd` intrinsics; the equivalent `atomicrmw fadd` form is now canonical (per LLVM 20.1.0 release notes). Any HIP kernel using the removed builtins fails to link with LLVM 20+.

## Method

Grep across `crates/` and `kernels/` for both the C-level builtin and the LLVM intrinsic forms:

```bash
grep -rn "__builtin_amdgcn_global_atomic_fadd\|__builtin_amdgcn_flat_atomic_fadd" \
       crates/ kernels/
grep -rn "llvm.amdgcn.flat.atomic.fadd\|llvm.amdgcn.global.atomic.fadd" \
       crates/ kernels/
grep -rn "atomicAdd.*float\|atomicAdd.*double\|amdgcn_ds_atomic_fadd" \
       crates/ kernels/
```

## Result

**CLEAN.** Zero hits across all three patterns.

- No occurrences of `__builtin_amdgcn_global_atomic_fadd` or `__builtin_amdgcn_flat_atomic_fadd` in HIP sources.
- No occurrences of the lowered LLVM IR intrinsic forms.
- No `atomicAdd` calls on `float` / `double` operands. No uses of `__builtin_amdgcn_ds_atomic_fadd` (LDS variant).

## Cross-validation

Audit performed on both local hiptrx worktree and hipx hetero-from-origin worktree. Identical (empty) result.

## gfx1010 silicon context

RDNA1 has no native FP atomic add hardware. Any FP atomic add lowers to a CAS loop on this arch regardless. So even if we did have such intrinsics, the LLVM 20 change would not be a perf concern on gfx1010 specifically.

## Recommendation if Kaden bumps to LLVM 20+

**No action required.** The codebase is already on the canonical `atomicrmw fadd` form (where it uses any FP atomics at all, which appears to be nowhere). The LLVM 20 bump is safe from this specific concern.

If a future kernel needs FP atomic add, prefer:

```cpp
// Canonical (works LLVM 17+, including LLVM 20+)
__atomic_fetch_add(ptr, val, __ATOMIC_RELAXED);
// or in HIP/CUDA-compat code:
atomicAdd(ptr, val);  // compiler lowers to atomicrmw fadd
```

Avoid:

```cpp
// Removed in LLVM 20.1.0
__builtin_amdgcn_global_atomic_fadd_f32(ptr, val);
__builtin_amdgcn_flat_atomic_fadd_f32(ptr, val);
```

## Sources

- [LLVM 20.1.0 release notes](https://releases.llvm.org/20.1.0/docs/ReleaseNotes.html)
- [LLVM AMDGPU backend](https://llvm.org/docs/AMDGPUUsage.html)
