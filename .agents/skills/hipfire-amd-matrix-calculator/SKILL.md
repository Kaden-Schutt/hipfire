---
name: hipfire-amd-matrix-calculator
description: Use Hipfire's vendored AMD Matrix Instruction Calculator in ./third_party/amd_matrix_instruction_calculator to inspect WMMA, SWMMAC, MFMA, and SMFMAC instruction details, lane/register mappings, matrix layouts, operand modifiers, and throughput/register-use facts for AMD RDNA/CDNA kernel work. Use when designing or debugging WMMA/MFMA kernels, especially BF16/F16 16x16x16 mappings, C/D accumulator layouts, OPSEL/NEG modifiers, or arch-specific matrix-instruction availability.
---

# Hipfire AMD Matrix Calculator

Use this skill when a Hipfire kernel task needs AMD matrix-instruction facts
that should come from the vendored calculator rather than memory or ad hoc
layout guesses.

## Tool Path

Run from the Hipfire repo root:

```bash
python3 third_party/amd_matrix_instruction_calculator/matrix_calculator.py <args>
```

If Python reports a missing dependency, install the tool requirements into the
current environment:

```bash
python3 -m pip install -r third_party/amd_matrix_instruction_calculator/requirements.txt
```

Do not shell out to web docs for this unless the vendored calculator lacks the
instruction or architecture being investigated.

## Standard Workflow

1. Identify the target architecture using Hipfire names where possible:
   `gfx1100`, `gfx1151`, `gfx1200`, `gfx1201`, or `gfx942`.
2. List supported matrix instructions for that arch:

   ```bash
   python3 third_party/amd_matrix_instruction_calculator/matrix_calculator.py \
     --architecture gfx1151 --list-instructions
   ```

3. Inspect the exact instruction before writing or changing a kernel:

   ```bash
   python3 third_party/amd_matrix_instruction_calculator/matrix_calculator.py \
     --architecture gfx1151 \
     --instruction v_wmma_f32_16x16x16_bf16 \
     --detail-instruction
   ```

4. Query C/D layout before mapping accumulator registers to output rows:

   ```bash
   python3 third_party/amd_matrix_instruction_calculator/matrix_calculator.py \
     --architecture gfx1151 \
     --instruction v_wmma_f32_16x16x16_bf16 \
     --D-matrix --register-layout
   ```

5. For a single output element, ask for the full input calculation too:

   ```bash
   python3 third_party/amd_matrix_instruction_calculator/matrix_calculator.py \
     --architecture gfx1151 \
     --instruction v_wmma_f32_16x16x16_bf16 \
     --D-matrix --get-register -I 0 -J 0 --output-calculation
   ```

6. Use `--matrix-entry -r <reg> -l <lane>` when auditing an existing kernel's
   register unpacking:

   ```bash
   python3 third_party/amd_matrix_instruction_calculator/matrix_calculator.py \
     --architecture gfx1151 \
     --instruction v_wmma_f32_16x16x16_bf16 \
     --D-matrix --matrix-entry --register 0 --lane 0
   ```

## Hipfire Kernel Rules

- Treat calculator output as the source of truth for register/lane layout, but
  still channel-test any new WMMA/MFMA kernel against a scalar or known-good
  reference. Calculator layout correctness does not prove the kernel's indexing
  code is correct.
- Prefer `--markdown` or `--csv` when saving layout evidence into docs,
  comments, or review artifacts.
- Re-run `--detail-instruction` separately for gfx11 and gfx12. Do not assume
  RDNA3 and RDNA4 WMMA operand layouts are identical.
- For CDNA targets, inspect MFMA instructions with the same flow; for RDNA
  targets, inspect WMMA/SWMMAC instructions.

## Common Hipfire Queries

BF16 WMMA on gfx1151:

```bash
python3 third_party/amd_matrix_instruction_calculator/matrix_calculator.py \
  -a gfx1151 -i v_wmma_f32_16x16x16_bf16 -d
```

F16 WMMA on gfx1100:

```bash
python3 third_party/amd_matrix_instruction_calculator/matrix_calculator.py \
  -a gfx1100 -i v_wmma_f32_16x16x16_f16 -d
```

RDNA4 instruction availability:

```bash
python3 third_party/amd_matrix_instruction_calculator/matrix_calculator.py \
  -a gfx1201 -L
```

MI300/CDNA3 MFMA availability:

```bash
python3 third_party/amd_matrix_instruction_calculator/matrix_calculator.py \
  -a gfx942 -L
```
