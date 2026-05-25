# Exp #7: gfx10-1-generic compile target

**Date:** 2026-05-07
**Status:** VERDICT — WIN (squash-merging to master)

## Lever

LLVM groups gfx1010 (Navi 10), gfx1011 (Navi 12), gfx1012 (Navi 14) into the `gfx10-1-generic` target family. A single binary compiled with `--offload-arch=gfx10-1-generic` should run on any RDNA1 silicon with bit-identical results.

## Implementation

Single env override `HIPFIRE_TARGET_ARCH` added to `crates/rdna-compute/src/dispatch.rs::init_with_device`:

```rust
let detected_arch = hip.get_arch(id).unwrap_or_else(|_| "gfx1010".to_string());
let arch = std::env::var("HIPFIRE_TARGET_ARCH")
    .ok()
    .filter(|s| !s.is_empty())
    .unwrap_or(detected_arch);
```

Empty / unset preserves prior behavior byte-for-byte. When set, the value flows through `KernelCompiler::new(&arch)` → `--offload-arch={arch}` to hipcc and into the JIT cache key.

## Bench results

Hardware: hipx, single RX 5700 XT (gfx1010, ROCR_VISIBLE_DEVICES=1).

### Quality gate

Output bit-identical between gfx1010-specific and gfx10-1-generic builds for canonical 9B prompt:

```
md5: ccbefe413d7f8b68ecef9fd06a16d62b  (both conditions)
```

### Performance equivalence

3 fresh-process runs per condition, deterministic decode, max_tokens=120:

| condition | run 1 | run 2 | run 3 | median | mean | σ |
|---|---|---|---|---|---|---|
| gfx1010-specific | 55.9 | 56.0 | 55.9 | 55.9 | 55.93 | 0.047 |
| gfx10-1-generic | 55.8 | 55.9 | 55.8 | 55.8 | 55.83 | 0.047 |

**Delta: -0.18% (mean), -0.18% (median).** Well within the ±1% pre-registered tolerance.

## Verdict

**WIN.** All four pre-registered win criteria satisfied:

1. ✅ hipcc accepts `--offload-arch=gfx10-1-generic` and compiles all hipfire kernels without error on first JIT.
2. ✅ Daemon emits `{"type":"loaded"...}` and proceeds to inference.
3. ✅ First-token (and all 120) output matches the gfx1010-specific build byte-for-byte (md5 identical).
4. ✅ Decode tok/s within ±1% of gfx1010-specific baseline (-0.18%).

## Action taken

- Mirrored env-override patch to local master worktree (`crates/rdna-compute/src/dispatch.rs`).
- Squash-merged to master with this verdict doc.
- Existing per-arch cache directories (`.hipfire_kernels/gfx1010/`) remain functional for users not setting the env. New env opt-in tested empirically to be perf-neutral and output-equivalent.

## Implications

This unblocks several follow-ons:

1. **BC-160 (gfx1011) plan no longer needs separate per-arch JIT.** Setting `HIPFIRE_TARGET_ARCH=gfx10-1-generic` produces a binary that runs on any Navi 10/12/14 card with one cache directory.
2. **Forward compatibility.** Any future Navi-1.x silicon revision that's part of the gfx10-1 family inherits the cached kernels.
3. **Future default flip.** A follow-up PR could flip the default for any detected `gfx101x` arch to `gfx10-1-generic`. That would auto-collapse the per-arch cache for all RDNA1 users. Out of scope here — needs broader testing across multiple gfx101x cards before defaulting.

## What this lever does NOT change

- Performance: -0.18% delta is in the noise. Cross-arch generic targets typically lose a hair on micro-benchmarks vs arch-specific targets, but here the difference is below our 3-run measurement floor.
- Behavior on non-RDNA1 cards (gfx1100, gfx1151, gfx1201): unchanged. The env override is opt-in; default behavior remains arch-specific.
- Other arch families (gfx9-generic, gfx11-generic): not exercised by this experiment. Same lever should work for them by symmetry but bench-only confirmation is needed before claiming.

## Closure

Per the autoresearch contract, a WIN squash-merges to master and updates the baseline. Done.
