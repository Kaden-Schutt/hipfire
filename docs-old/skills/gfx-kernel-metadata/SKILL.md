---
name: gfx-kernel-metadata
description: Extract VGPR/SGPR/LDS/spill counts and AMDGPU notes from a compiled HIP kernel `.hsaco` for any AMD GPU arch (CDNA wave64 — gfx906/908/90a/942 — and RDNA wave32 — gfx1010 through gfx12). Lets you compute theoretical occupancy and identify register-pressure / LDS-pressure constraints.
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
recognize it. First 16 bytes will look like `__CLANG_OFFLOAD_`,
confirming the wrapper.

## Step-by-step

```bash
# Pick your arch — auto-detect from the cache directory if hipfire's
# JIT cache is around, or set explicitly. This works for any RDNA or
# CDNA arch (gfx906/908/90a/942/1010/1030/1100/1101/1102/1150/1151/1200/1201).
ARCH="${ARCH:-$(basename "$(ls -1d .hipfire_kernels/gfx* 2>/dev/null | head -1)")}"
ARCH="${ARCH:-gfx1100}"  # fallback if no cache directory exists

# 1. List bundled targets (confirm the gfx target inside)
/opt/rocm/llvm/bin/clang-offload-bundler \
    --list --type=o \
    --input=path/to/kernel.hsaco
# → hipv4-amdgcn-amd-amdhsa--$ARCH
#   host-x86_64-unknown-linux-gnu-

# 2. Unbundle to a real ELF
/opt/rocm/llvm/bin/clang-offload-bundler \
    --type=o --unbundle \
    --input=path/to/kernel.hsaco \
    --output=/tmp/kernel.elf \
    --targets=hipv4-amdgcn-amd-amdhsa--$ARCH

# 3. Read the AMDGPU note section — this is the metadata
/opt/rocm/llvm/bin/llvm-readelf --notes /tmp/kernel.elf
```

The `--notes` output contains a YAML block under `amdhsa.kernels:`
with the relevant fields:

```yaml
.vgpr_count:                82       # VGPRs allocated per wave
.sgpr_count:                18       # SGPRs (scalar regs) per wave
.vgpr_spill_count:          0        # >0 means spill pressure
.sgpr_spill_count:          0
.group_segment_fixed_size:  256      # static LDS bytes per workgroup
.private_segment_fixed_size:0        # private (scratch) bytes per lane
.max_flat_workgroup_size:   32
.wavefront_size:            32       # 64 = CDNA, 32 = RDNA
.uses_dynamic_stack:        false
```

**Spill canary:** `.vgpr_spill_count` and `.sgpr_spill_count` may be
elided from the YAML when zero (depends on toolchain version). The
reliable proxy is **`.private_segment_fixed_size`** — if it's `0`, the
kernel uses no scratch memory and therefore has no register spills.
Any non-zero value means VGPRs are spilling to private memory and you
should investigate.

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
/opt/rocm/llvm/bin/llvm-objdump --disassemble --mcpu=$ARCH /tmp/kernel.elf
```

Note: the `--mcpu=` flag must match the target the ELF was built for.
Without it, you'll get incorrect or no decode. Useful for: spotting
`global_load_*` patterns (memory parallelism), `v_dot4_i32_i8` (dp4a),
`v_wmma_*` (RDNA3+ matrix), `v_mfma_*` (CDNA matrix), `ds_*` (LDS)
instructions, `s_waitcnt` and `s_barrier` placement.

## Architecture cheat-sheet

| Arch | Wave | VGPRs/SIMD | Max waves/SIMD | LDS/CU | Matrix path |
|---|---:|---:|---:|---:|---|
| gfx906 (Vega/MI50)         | 64 | 256  | 10 | 64 KB  | dp4a (v_dot4) |
| gfx908 (MI100)             | 64 | 256  | 10 | 64 KB  | MFMA |
| gfx90a (MI210/250)         | 64 | 512  | 8  | 64 KB  | MFMA |
| gfx942 (MI300X)            | 64 | 512  | 8  | 64 KB  | MFMA |
| gfx1010 (Navi 10, RDNA1)   | 32 | 1024 | 20 | 64 KB  | dp4a only |
| gfx1030 (Navi 21, RDNA2)   | 32 | 1024 | 16 | 64 KB  | dp4a only |
| gfx1100 (Navi 31, RDNA3)   | 32 | 1024 | 16 | 64 KB  | WMMA |
| gfx1101 / gfx1102          | 32 | 1024 | 16 | 64 KB  | WMMA |
| gfx1150 / gfx1151 (Strix)  | 32 | 1024 | 16 | 64 KB  | WMMA |
| gfx1200 / gfx1201 (RDNA4)  | 32 | 1536 | 16 | 128 KB | WMMA32 |

Wave64 (CDNA) needs ~2× the VGPRs of wave32 RDNA for the same per-
lane pressure: a 32-VGPR wave32 RDNA kernel ≈ 64-VGPR wave64. CDNA
kernels frequently sit at 40–80 VGPR; RDNA kernels in the 30–90 range.

## Interpreting the numbers — occupancy tables per arch family

### Wave64 — gfx906/908 (256 VGPRs/SIMD)

| VGPRs/wave | Max waves/SIMD | Max waves/CU (×4 SIMDs) |
|---:|---:|---:|
| ≤ 24  | 10 | 40 |
| 25–28 | 9  | 36 |
| 29–32 | 8  | 32 |
| 33–36 | 7  | 28 |
| 37–40 | 6  | 24 |
| 41–48 | 5  | 20 |
| 49–64 | 4  | 16 |
| 65–84 | 3  | 12 |
| 85–128| 2  | 8  |
| 129–256| 1 | 4  |

### Wave64 — gfx90a/942 (512 VGPRs/SIMD)

Theoretical waves/SIMD = `floor(512 / vgpr_count)` capped at 8.
Common allocation granule = 8 VGPRs.

| VGPRs/wave | Max waves/SIMD |
|---:|---:|
| ≤ 64  | 8 |
| 65–80 | 6 |
| 81–96 | 5 |
| 97–128| 4 |
| 129–170| 3 |
| 171–256| 2 |

### Wave32 — gfx1010+ RDNA (1024 VGPRs/SIMD; gfx1200+ has 1536)

Theoretical waves/SIMD = `floor(VGPRs_per_SIMD / vgpr_count)` capped
at 16 (RDNA1: 20). Common allocation granule = 24 VGPRs (rounded up
in groups of 8 for some toolchain versions).

For gfx1100 / gfx1151 (RDNA3, 1024 VGPRs/SIMD):

| VGPRs/wave | Max waves/SIMD | VGPRs/wave | Max waves/SIMD |
|---:|---:|---:|---:|
| ≤ 64    | 16 | 65–84   | 12 |
| 85–96   | 10 | 97–112  | 9  |
| 113–128 | 8  | 129–168 | 6  |
| 169–204 | 5  | 205–256 | 4  |

For gfx1200 / gfx1201 (RDNA4, 1536 VGPRs/SIMD), divide 1536 by
vgpr_count instead of 1024 — same table shifts by 1.5×.

LDS budget: 64 KB per CU on gfx9xx and gfx10xx-1101; 128 KB on
gfx1200+. The kernel claims its share via `.group_segment_fixed_size`.
LDS occupancy = `floor(LDS_per_CU / group_segment_fixed_size)`
workgroups per CU; multiply by waves-per-WG to get waves/CU from the
LDS side. Actual occupancy = `min(VGPR-occupancy, LDS-occupancy,
SGPR-occupancy, max-waves-per-CU)`.

SGPRs are rarely the binder — every modern AMD arch has 800+ SGPRs/CU.

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
- **WMMA / MFMA kernels run hot on VGPRs** because the matrix
  accumulator (`float8_t` for WMMA-f32-16x16x16, `float16_t` etc.)
  occupies many VGPRs per wave. 80–120 VGPRs is common for fused
  GEMM kernels even without spills.

## Reproducing — copy/paste recipe (parameterized)

```bash
# Defaults — override via environment.
ROCM="${ROCM:-/opt/rocm/llvm/bin}"
ARCH="${ARCH:-$(basename "$(ls -1d .hipfire_kernels/gfx* 2>/dev/null | head -1)")}"
ARCH="${ARCH:-gfx1100}"
KERNEL_DIR="${KERNEL_DIR:-.hipfire_kernels/$ARCH}"
TMP="${TMP:-/tmp/hsaco-extract-$ARCH}"
mkdir -p "$TMP"

# Inspect ALL kernels in the cache, or pass an explicit list as $@:
KERNELS=("$@")
if [ ${#KERNELS[@]} -eq 0 ]; then
  mapfile -t KERNELS < <(ls -1 "$KERNEL_DIR"/*.hsaco 2>/dev/null | xargs -n1 basename | sed 's/\.hsaco$//')
fi

printf "%-50s %4s %4s %5s %5s %6s\n" "kernel" "VGPR" "SGPR" "spill" "LDS" "wave"
for K in "${KERNELS[@]}"; do
  ELF="$TMP/$K.elf"
  $ROCM/clang-offload-bundler --type=o --unbundle \
      --input="$KERNEL_DIR/$K.hsaco" \
      --output="$ELF" \
      --targets=hipv4-amdgcn-amd-amdhsa--$ARCH 2>/dev/null
  notes=$($ROCM/llvm-readelf --notes "$ELF" 2>/dev/null)
  vgpr=$(echo "$notes" | grep -m1 vgpr_count | awk '{print $NF}')
  sgpr=$(echo "$notes" | grep -m1 sgpr_count | awk '{print $NF}')
  spill=$(echo "$notes" | grep -m1 private_segment_fixed_size | awk '{print $NF}')
  lds=$(echo "$notes" | grep -m1 group_segment_fixed_size | awk '{print $NF}')
  wave=$(echo "$notes" | grep -m1 wavefront_size | awk '{print $NF}')
  printf "%-50s %4s %4s %5s %5s %6s\n" "$K" "$vgpr" "$sgpr" "$spill" "$lds" "$wave"
done
```

(Hipfire's modern JIT-cache directory is `.hipfire_kernels/<arch>/`;
older builds may use `kernels/compiled/<arch>/`. Adjust `KERNEL_DIR`
accordingly.)

## Gotchas

- The bundler `--targets=` string must match exactly. For gfx1100 use
  `hipv4-amdgcn-amd-amdhsa--gfx1100`. List with `--list` first if
  uncertain.
- `llvm-objdump` will fail with "not a valid object file" if you
  forget to unbundle. The error is unhelpful — always unbundle first.
- `vgpr_count` is post-allocation (granule-rounded). The actual
  in-use count may be lower; check disassembly's highest `v<N>`
  reference if you need the un-rounded value.
- `code-object-v4` and `v5` differ in metadata layout; `--notes` works
  for both but raw `.amdhsa_*` directive parsing does not.
- The spill fields (`.vgpr_spill_count`, `.sgpr_spill_count`) are
  sometimes omitted from the YAML when zero. **`.private_segment_fixed_size: 0`
  is the reliable "no spills" indicator** — non-zero always means
  spills, regardless of which fields the toolchain emits.
- `.max_flat_workgroup_size` reports `__launch_bounds__(N, ...)` 's
  N value (not the M parameter). To find M (waves/SIMD limit), look
  at `.amdhsa_*` directives via raw disassembly.
