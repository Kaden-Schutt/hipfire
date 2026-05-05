---
name: gfx-kernel-metadata
description: Extract VGPR/SGPR/LDS/spill counts and AMDGPU notes from a compiled HIP kernel `.hsaco` so you can compute theoretical occupancy and identify register-pressure / LDS-pressure constraints on AMD GPUs.
---

# Reading AMD GPU kernel metadata from `.hsaco`

When tuning HIP kernels you often need to know what the compiler actually
allocated: VGPRs, SGPRs, LDS bytes, and whether anything spilled. Those
numbers gate occupancy and reveal whether the kernel is register-bound,
LDS-bound, or neither.

## What `.hsaco` actually is

A `.hsaco` is **not** a raw ELF — it's a `__CLANG_OFFLOAD_BUNDLE__`
container that wraps an `amdgcn-amd-amdhsa--gfxNNN` ELF (and possibly a
host stub). You must unbundle it before any objdump / readelf tool will
recognize it. First two bytes will look like `__CLANG_OF`, confirming
the wrapper.

## Step-by-step

```bash
# 1. List bundled targets (confirm the gfx target inside)
/opt/rocm/llvm/bin/clang-offload-bundler \
    --list --type=o \
    --input=path/to/kernel.hsaco
# → hipv4-amdgcn-amd-amdhsa--gfx906
#   host-x86_64-unknown-linux-gnu-

# 2. Unbundle to a real ELF
/opt/rocm/llvm/bin/clang-offload-bundler \
    --type=o --unbundle \
    --input=path/to/kernel.hsaco \
    --output=/tmp/kernel.elf \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx906   # match the arch

# 3. Read the AMDGPU note section — this is the metadata
/opt/rocm/llvm/bin/llvm-readelf --notes /tmp/kernel.elf
```

The `--notes` output contains a YAML block under `amdhsa.kernels:`
with the relevant fields:

```yaml
.vgpr_count:                29       # VGPRs allocated per wave
.sgpr_count:                14       # SGPRs (scalar regs) per wave
.vgpr_spill_count:          0        # >0 means spill pressure
.sgpr_spill_count:          0
.group_segment_fixed_size:  0        # static LDS bytes per workgroup
.private_segment_fixed_size:0        # private (scratch) bytes per lane
.max_flat_workgroup_size:   1024
.wavefront_size:            64       # wave64 vs wave32
.uses_dynamic_stack:        false
```

For a fast multi-kernel comparison:

```bash
for k in kernel1 kernel2 kernel3; do
  echo "=== $k ==="
  /opt/rocm/llvm/bin/llvm-readelf --notes "$k.elf" 2>&1 | \
    grep -E "\.(vgpr_count|sgpr_count|vgpr_spill|sgpr_spill|group_segment_fixed_size|private_segment_fixed_size|wavefront_size):"
done
```

## Disassembly (when you need to see actual instructions)

```bash
/opt/rocm/llvm/bin/llvm-objdump --disassemble --mcpu=gfx906 /tmp/kernel.elf
```

Note: the `--mcpu=` flag must match the target the ELF was built for.
Without it, you'll get incorrect or no decode. Useful for: spotting
`global_load_*` patterns (memory parallelism), `v_dot4_i32_i8` (dp4a)
usage, `ds_*` (LDS) instructions, `s_waitcnt` placement.

## Interpreting the numbers — gfx906 (Vega 20, MI50/MI60) wave64

VGPR file = 256 per SIMD × 4 SIMDs/CU = 1024 VGPRs/CU. Allocation
granule is 4 VGPRs per wave. Theoretical wave-occupancy table:

| VGPRs/wave | Max waves/SIMD | Max waves/CU |
|---:|---:|---:|
| ≤ 24 | 10 | 40 |
| 25–28 | 9 | 36 |
| 29–32 | 8 | 32 |
| 33–36 | 7 | 28 |
| 37–40 | 6 | 24 |
| 41–48 | 5 | 20 |
| 49–64 | 4 | 16 |
| 65–84 | 3 | 12 |
| 85–128 | 2 | 8 |
| 129–256 | 1 | 4 |

LDS budget: 64 KB per CU on gfx906 (configurable; the kernel claims
its share via `.group_segment_fixed_size`). LDS occupancy =
`floor(65536 / .group_segment_fixed_size)` workgroups per CU; multiply
by waves-per-WG to get waves/CU from the LDS side. Actual occupancy =
`min(VGPR-occupancy, LDS-occupancy, SGPR-occupancy, max-waves-per-CU)`.

SGPRs are usually not the binder — gfx906 has 800+ SGPRs/CU.

## Architecture cheat-sheet

| Arch | Wave | VGPRs/CU | LDS/CU |
|---|---:|---:|---:|
| gfx906 (Vega/MI50) | 64 | 1024 | 64 KB |
| gfx908 (MI100) | 64 | 1024 | 64 KB |
| gfx90a (MI200) | 64 | 1024 | 64 KB |
| gfx942 (MI300) | 64 | 1024 | 64 KB |
| gfx1010-gfx1100 (RDNA1-3) | 32 | 1024 | 64–128 KB |
| gfx1200 (RDNA4) | 32 | 1536 | 128 KB |

Wave64 architectures (CDNA: gfx906/908/90a/942) need ~2× the VGPRs of
wave32 RDNA for the same per-lane register pressure, so a 32-VGPR
wave32 RDNA kernel is roughly equivalent to 64-VGPR wave64 — not 32.
That's why CDNA kernels frequently sit at 40–80 VGPR.

## What "high occupancy" does and doesn't tell you

- **Spills (>0) = bug or budget breach.** Always investigate. Common
  culprits: `#pragma unroll` on too-large loops, deep accumulator
  trees, large constant arrays. Measure spilled-vs-rolled both ways
  before committing.
- **High theoretical occupancy + low VALUBusy = memory-bound.** More
  occupancy won't help; you need more in-flight HBM transactions per
  wave (multi-quad interleave, half-wave splits, prefetch).
- **Low occupancy (≤2 waves/SIMD) + high VALUBusy = look for register
  reuse / loop fusion** to drop VGPR pressure.
- **LDS = 0 + GEMV-shaped kernel** is normal for memory-streaming
  kernels (decode GEMV). LDS only helps if you reuse data.

## Reproducing — copy/paste recipe

```bash
# Path defaults that work on this machine
ROCM=/opt/rocm/llvm/bin
KERNEL_DIR=/home/kread/git/hipfire/kernels/compiled/gfx906
TMP=/tmp/hsaco-extract
mkdir -p "$TMP" && cd "$TMP"

for K in gemv_hfq4g256_residual_wave64 fused_gate_up_hfq4g256_wave64; do
  $ROCM/clang-offload-bundler --type=o --unbundle \
      --input="$KERNEL_DIR/$K.hsaco" \
      --output="$K.elf" \
      --targets=hipv4-amdgcn-amd-amdhsa--gfx906
  echo "=== $K ==="
  $ROCM/llvm-readelf --notes "$K.elf" | \
      grep -E "\.(vgpr_count|sgpr_count|.*spill|group_segment|wavefront_size):"
done
```

## Gotchas

- The bundler `--targets=` string must match exactly. For gfx1100 use
  `hipv4-amdgcn-amd-amdhsa--gfx1100`. For multi-arch fat binaries,
  list with `--list` first.
- `llvm-objdump` will fail with "not a valid object file" if you
  forget to unbundle. The error is unhelpful — always unbundle first.
- `vgpr_count` is post-allocation (granule-rounded). The actual
  in-use count may be lower; check disassembly's highest `v<N>`
  reference if you need the un-rounded value.
- `code-object-v4` and `v5` differ in metadata layout; `--notes` works
  for both but raw `.amdhsa_*` directive parsing does not.
