---
name: gfx-kernel-metadata
description: Extract VGPR/SGPR/LDS/spill counts and AMDGPU notes from compiled HIP .hsaco/.co objects for RDNA and CDNA targets. Use when auditing register pressure, theoretical occupancy, spill canaries, or ISA shape after a kernel compile — before claiming a tuning win.
---

# gfx-kernel-metadata

Read what the compiler actually allocated for a hipfire kernel object:
VGPRs, SGPRs, LDS, private/scratch, spills, wavefront size. Use this for
static resource checks; use Kernel Atlas when you also need profiled
phase rows and ISA Fit View.

Sole skill root: `.agents/skills/` (this path). Do not invent alternate
skill trees or aliases outside it.

## Toolchain (current)

Prefer tools already used by `scripts/kernel_atlas.py`:

| role | binary (PATH or ROCm) |
|---|---|
| list / unbundle | `clang-offload-bundler` → `/opt/rocm/llvm/bin/clang-offload-bundler` |
| AMDGPU notes | `llvm-readobj --notes` (Atlas default) or `llvm-readelf --notes` |
| disassembly | `llvm-objdump -d --no-show-raw-insn` (Atlas) or `--disassemble --mcpu=$ARCH` |

A `.hsaco` is usually a `__CLANG_OFFLOAD_BUNDLE__` container, not a bare
ELF. Unbundle before readobj/objdump or tools report "not a valid object".

## Cache locations

| layout | path |
|---|---|
| JIT / local cache (common) | `.hipfire_kernels/<arch>/*.hsaco` or flat `.hipfire_kernels/*.hsaco` |
| Packaged install blobs | `~/.hipfire/bin/kernels/` (and per-arch subdirs when present) |
| Older tree layout | `kernels/compiled/<arch>/` |

Detect arch from the cache when possible:

```bash
ARCH="${ARCH:-$(basename "$(ls -1d .hipfire_kernels/gfx* 2>/dev/null | head -1)")}"
ARCH="${ARCH:-gfx1100}"
KERNEL_DIR="${KERNEL_DIR:-.hipfire_kernels/$ARCH}"
# flat cache fallback:
[ -d "$KERNEL_DIR" ] || KERNEL_DIR=.hipfire_kernels
```

## One-shot extract (manual)

```bash
ROCM="${ROCM:-/opt/rocm/llvm/bin}"
ARCH="${ARCH:-gfx1201}"
HSACO="${HSACO:-.hipfire_kernels/$ARCH/some_kernel.hsaco}"
# or flat: HSACO=.hipfire_kernels/some_kernel.hsaco

# 1) Confirm bundle target
"$ROCM/clang-offload-bundler" --list --type=o --input="$HSACO"
# → hipv4-amdgcn-amd-amdhsa--$ARCH

# 2) Unbundle AMDGPU ELF into a per-run temp dir (never shared /tmp names)
TMPDIR_RUN="$(mktemp -d "${TMPDIR:-/tmp}/hsaco-meta.XXXXXX")"
trap 'rm -rf "$TMPDIR_RUN"' EXIT
ELF="$TMPDIR_RUN/kernel.elf"
"$ROCM/clang-offload-bundler" --type=o --unbundle \
  --input="$HSACO" \
  --output="$ELF" \
  --targets="hipv4-amdgcn-amd-amdhsa--$ARCH"

# 3) Resource notes (YAML under amdhsa.kernels)
"$ROCM/llvm-readobj" --notes "$ELF"
# equivalent: "$ROCM/llvm-readelf" --notes "$ELF"
```

Fields that matter:

```text
.vgpr_count                 # VGPRs per wave (allocation granule-rounded)
.sgpr_count
.vgpr_spill_count           # may be omitted when 0
.sgpr_spill_count           # may be omitted when 0
.group_segment_fixed_size   # static LDS bytes / workgroup
.private_segment_fixed_size # scratch/private bytes / lane — not spill proof
.max_flat_workgroup_size
.wavefront_size             # 32 RDNA, 64 CDNA
```

**Scratch/private canary (not spill proof):** `.vgpr_spill_count` /
`.sgpr_spill_count` are the only fields that directly attest
compiler-reported SGPR/VGPR spills (they may be omitted when 0).
`.private_segment_fixed_size` only establishes private/scratch allocation;
non-zero can mean spills, explicit private storage, or stack use. Treat a
non-zero private segment as a scratch/private-traffic canary that needs
attribution before promotion — never as proof of register spilling.

## Batch table (copy/paste)

```bash
ROCM="${ROCM:-/opt/rocm/llvm/bin}"
ARCH="${ARCH:-$(basename "$(ls -1d .hipfire_kernels/gfx* 2>/dev/null | head -1)")}"
ARCH="${ARCH:-gfx1100}"
KERNEL_DIR="${KERNEL_DIR:-.hipfire_kernels/$ARCH}"
[ -d "$KERNEL_DIR" ] || KERNEL_DIR=.hipfire_kernels
TMPDIR_RUN="$(mktemp -d "${TMPDIR:-/tmp}/hsaco-extract.XXXXXX")"
trap 'rm -rf "$TMPDIR_RUN"' EXIT

printf "%-48s %4s %4s %6s %6s %5s\n" "kernel" "VGPR" "SGPR" "priv" "LDS" "wave"
for HSACO in "$KERNEL_DIR"/*.hsaco; do
  [ -f "$HSACO" ] || continue
  K=$(basename "$HSACO" .hsaco)
  ELF="$TMPDIR_RUN/$K.elf"
  "$ROCM/clang-offload-bundler" --type=o --unbundle \
    --input="$HSACO" --output="$ELF" \
    --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" 2>/dev/null || continue
  notes=$("$ROCM/llvm-readobj" --notes "$ELF" 2>/dev/null)
  vgpr=$(printf '%s\n' "$notes" | awk '/vgpr_count:/ {print $NF; exit}')
  sgpr=$(printf '%s\n' "$notes" | awk '/sgpr_count:/ {print $NF; exit}')
  priv=$(printf '%s\n' "$notes" | awk '/private_segment_fixed_size:/ {print $NF; exit}')
  lds=$(printf '%s\n' "$notes" | awk '/group_segment_fixed_size:/ {print $NF; exit}')
  wave=$(printf '%s\n' "$notes" | awk '/wavefront_size:/ {print $NF; exit}')
  printf "%-48s %4s %4s %6s %6s %5s\n" "$K" "$vgpr" "$sgpr" "$priv" "$lds" "$wave"
done
```

Filter with a glob or pass explicit basenames; do not dump an entire
multi-GB install tree into chat.

## Atlas-backed extract (preferred when measuring)

When the goal is metadata attached to a bench row or Fit View, use the
canonical owner instead of hand-parsing:

```bash
# Manifest only (no full Atlas row required)
python3 scripts/kernel_atlas.py collect-ar \
  --model ~/.hipfire/models/<model>.mq4 \
  --workload <name> --model-size <size> --quant mq4 \
  --prefill 32 --gen 5 \
  --isa-dir .hipfire_kernels/$ARCH \
  --isa-filter 'gemm_hfq4g256|gemv_hfq4g256' \
  --isa-limit 8 \
  --isa-output .codeinsight+research/kernel-atlas/runs/isa-$ARCH.json \
  --output .codeinsight+research/kernel-atlas/runs/atlas-$ARCH.jsonl
```

Atlas `inspect_isa_object` unbundles with `clang-offload-bundler`, reads
notes via `llvm-readobj --notes`, and summarizes opcodes via
`llvm-objdump -d --no-show-raw-insn`. Schema owner:
`scripts/kernel_atlas.py` + `docs/methodology/kernel-atlas.md`.

Render Fit View: `.agents/skills/hipfire-kernel-atlas/`.

## Disassembly

```bash
# Match --mcpu to the object arch or decode is wrong/empty.
# Use the same per-run ELF from the extract steps above ($ELF), never a shared /tmp path.
"$ROCM/llvm-objdump" --disassemble --mcpu="$ARCH" "$ELF"
# Atlas-style (no raw bytes):
"$ROCM/llvm-objdump" -d --no-show-raw-insn "$ELF"
```

Look for: `global_load_*` / `buffer_*` (memory), `v_dot4_*` (dp4a),
`v_wmma_*` (RDNA3+ matrix), `v_mfma_*` (CDNA matrix), `ds_*` (LDS),
`s_waitcnt` / `s_barrier` placement.

## Architecture budgets (interpretation aid)

Do **not** keep a mutable VGPR/LDS/wave resource matrix in this skill.
Wave size, matrix ISA, and family resource ceilings change with hardware
and compiler — own them at the current arch sources:

- Capability / WMMA-MFMA matrix: `.agents/skills/hipfire-arch-port/wmma-matrix.md`
- New-arch port + validation: `.agents/skills/hipfire-arch-port/`
- Live hardware/SKU evidence: `rocm-smi` / device props and the object notes
  extracted above (claim-scoped only)

This skill keeps **object-derived** metadata (counts from `--notes`) and
claim-scoped reading rules. For theoretical occupancy, compare those
object fields against the current arch owner and measured hardware — not
a copy of family ceilings here.

Wave64 CDNA needs roughly 2× VGPRs of wave32 RDNA for the same per-lane
live set. WMMA/MFMA accumulators routinely push fused GEMMs into a high
VGPR band without spills — high VGPR alone is not a bug.

## Reading the numbers

- **Non-zero private segment / reported spill counts** — stop; attribute
  private traffic (spill vs explicit private/stack) and reduce live ranges,
  unroll, or launch_bounds before claiming a perf win. Spill-count fields
  prove compiler spills; private size alone does not.
- **High theoretical occupancy + low VALUBusy** — memory/launch bound;
  more occupancy will not help.
- **≤2 waves/SIMD + high VALUBusy** — chase register reuse / fusion.
- **LDS = 0 on decode GEMV** — normal for pure streaming kernels.
- Metadata is **not** a roofline and **not** end-to-end tok/s proof.

## Guards

- Unbundle before objdump/readobj; match `--targets=` / `--mcpu=` to the
  object arch (`--list` first if unsure).
- `vgpr_count` is post-allocation (granule-rounded); highest `vN` in
  disassembly is the un-rounded live hint.
- code-object v4 vs v5: `--notes` works for both; do not parse raw
  `.amdhsa_*` directives as if layouts were identical.
- Do not treat one HSACO dump as a shipped benchmark. Perf claims need
  the methodology owners below.
- Do not delete `.hipfire_kernels/` wholesale while diagnosing — rebuild
  cost is high; remove only the one stale object you invalidated.

## Canonical owners (link, do not fork)

| concern | owner |
|---|---|
| ISA collect + Fit View + suggest/task | `.agents/skills/hipfire-kernel-atlas/` + `scripts/kernel_atlas.py` |
| Atlas methodology / row schema | `docs/methodology/kernel-atlas.md` |
| Tuning levers + cross-arch gates | `.agents/skills/hipfire-kernel-tuning/` |
| New arch port / WMMA matrix | `.agents/skills/hipfire-arch-port/` |
| Crate/dispatch map | `docs/ARCHITECTURE.md` |
| Bench noise / fresh-process protocol | `docs/methodology/perf-benchmarking.md` |
| Direct-KMD HSACO decode research | `crates/redline/src/hsaco.rs` (not the default HIP path) |
