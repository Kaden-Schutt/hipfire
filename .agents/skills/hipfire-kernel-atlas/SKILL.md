---
name: hipfire-kernel-atlas
description: Use Kernel Atlas to collect phase-aware hipfire measurements and render ISA Fit View visualizations for AMD GPU kernels, quant formats, and architectures. Use when a user asks how MQ/HFQ/HFP/Q8 quants occupy hardware, asks for an ASCII ISA visualization, wants to compare gfx1010/gfx1030/gfx11/gfx12 kernel fit, or wants an agent-readable "left on table" summary from Atlas rows. Atlas output is measurement evidence only — never runtime or promotion proof.
---

# hipfire-kernel-atlas

Thin agent wrapper around `scripts/kernel_atlas.py` and
`.agents/skills/hipfire-kernel-atlas/render-fit.sh`.

**Canonical methodology (phases, corpus layout, ISA/dispatch manifests, task/eval
loop):** [`docs/methodology/kernel-atlas.md`](../../../docs/methodology/kernel-atlas.md).

**Promotion / correctness routes:** [`docs/VALIDATION.md`](../../../docs/VALIDATION.md)
and, for timed claims, [`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md).
Atlas does not replace those owners.

## What Atlas is

| Atlas produces | Atlas does **not** prove |
|---|---|
| Phase-tagged JSONL rows (`prefill`, `decode_ar`, `decode_dflash`) | Shippable kernel or dispatch wins |
| Optional ISA manifests (VGPR/SGPR/LDS/spills, opcode mix) | Full hardware occupancy or roofline |
| Optional dispatch/source provenance for profiled names | A unique runtime branch |
| ASCII ISA Fit View + heuristic “likely limit / left on table” | Product defaults or admissions |
| `suggest` experiment queues; `task` / `eval` local ledgers | Correctness, serve semantics, or Redline route proof |

Treat every row as **measured** evidence tied to binary/prompt/git identity in
the row. Dirty worktrees: cite `provenance.diff_md5` and do not compare as a
shipped baseline.

## Producer / renderer surface

CLI (verify with `python3 scripts/kernel_atlas.py --help`):

| Subcommand | Role |
|---|---|
| `collect-ar` | AR prefill + decode_ar rows from bench output |
| `collect-dflash` | Spec-decode rows (acceptance/tau when printed) |
| `parse` | Metrics-only parse of saved bench/DFlash text → bare JSON metrics (not an identity-bearing Atlas row) |
| `render-fit` | ASCII ISA/quant fit view |
| `suggest` | Ranked experiment ideas (not predicted wins) |
| `task` / `task-pytorch` | Bounded edit + eval contract bundles |
| `eval` | Rerun task bench/correctness commands; local ledger |
| `graph-ab` | `HIPFIRE_GRAPH=0/1` A/B from a row |

Renderer wrapper (repo-root aware):

```bash
.agents/skills/hipfire-kernel-atlas/render-fit.sh \
  --row .codeinsight+research/kernel-atlas/runs/atlas.jsonl \
  --row-index 0 \
  --isa .codeinsight+research/kernel-atlas/runs/isa.json
# --dispatch optional; also optional if the row already references manifests
```

Raw corpus stays private/ignored:

```bash
mkdir -p .codeinsight+research/kernel-atlas/runs
mkdir -p .codeinsight+research/kernel-atlas/tasks
```

## Workflow

1. **Collect or locate rows** under `.codeinsight+research/kernel-atlas/runs/`.
   Prefer existing JSONL before re-collecting.
2. **Attach ISA** with `--isa-file` or `--isa-dir` + `--isa-filter`; prefer
   `--isa-output <path>.json` so many rows share one manifest. Needs ROCm LLVM
   tools when inspecting HSACO (`clang-offload-bundler`, `llvm-readobj`,
   `llvm-objdump`).
3. **Attach dispatch provenance** with `--dispatch-provenance` and
   `--dispatch-output` when profiled kernel names exist. Evidence to inspect,
   not proof of one branch. Ranking is arch-aware when arch-specific sources
   exist (e.g. `*.gfx1201.hip`).
4. **Render** via `render-fit.sh`. With `artifacts.profile_kernels`, the view
   joins profiled names to ISA symbols and scopes the summary; unmatched hot
   names are printed on purpose.
5. **`suggest`** → experiment queue only. Auto-loads history from
   `.codeinsight+research/kernel-atlas/tasks/` unless extra `--history` paths.
6. **`task`** → `task.json` + `TASK.md`. Pass `--allowed-file` for every editable
   path. Pass `--correctness-command` only as a **claim-scoped** command that
   still exists and matches [`docs/VALIDATION.md`](../../../docs/VALIDATION.md)
   for that change (path-specific oracle, `test_kernels`, serve harness, etc.).
   Do **not** treat retired batteries as acceptance; use VALIDATION.
7. **`eval`** → refresh baseline first when the row carried profiling env
   (`--refresh-baseline`); compare candidates with `--baseline`. Status
   `needs_baseline` or `unstable` → no speedup claim. Ledger is local lineage,
   not a public benchmark.

## Example commands

Paths and model files must exist on the machine; swap tags/files from
[`registry/models.json`](../../../registry/models.json).

Collect AR smoke with ISA + dispatch (illustrative):

```bash
python3 scripts/kernel_atlas.py collect-ar \
  --model ~/.hipfire/models/qwen3.5-0.8b.mq4 \
  --workload qwen3.5-0.8b \
  --model-size 0.8b \
  --quant mq4 \
  --prefill 32 \
  --gen 5 \
  --kv-mode asym3 \
  --profile-prefill \
  --profile-decode \
  --isa-dir .hipfire_kernels \
  --isa-filter 'gemm_hfq4g256|gemv_hfq4g256' \
  --isa-output .codeinsight+research/kernel-atlas/runs/isa.json \
  --dispatch-provenance \
  --dispatch-output .codeinsight+research/kernel-atlas/runs/dispatch.json \
  --output .codeinsight+research/kernel-atlas/runs/atlas.jsonl
```

DFlash collection (prompts under `benchmarks/prompts/` when present):

```bash
python3 scripts/kernel_atlas.py collect-dflash \
  --target ~/.hipfire/models/qwen3.5-27b.mq4 \
  --draft ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq \
  --prompt-file benchmarks/prompts/merge_sort_thinking_off.txt \
  --workload qwen3.5-27b-dflash-merge-sort \
  --max-tokens 256 \
  --ctx 2048 \
  --kv-mode q8 \
  --output .codeinsight+research/kernel-atlas/runs/atlas-dflash.jsonl
```

Suggest / task / eval:

```bash
python3 scripts/kernel_atlas.py suggest \
  --row .codeinsight+research/kernel-atlas/runs/atlas.jsonl \
  --row-index 1 \
  --isa .codeinsight+research/kernel-atlas/runs/isa.json \
  --dispatch .codeinsight+research/kernel-atlas/runs/dispatch.json \
  --format markdown

python3 scripts/kernel_atlas.py task \
  --row .codeinsight+research/kernel-atlas/runs/atlas.jsonl \
  --row-index 1 \
  --isa .codeinsight+research/kernel-atlas/runs/isa.json \
  --dispatch .codeinsight+research/kernel-atlas/runs/dispatch.json \
  --allowed-file kernels/src/gemv_hfq4g256_multirow.hip \
  --output-dir .codeinsight+research/kernel-atlas/tasks/example-gemv

python3 scripts/kernel_atlas.py eval \
  --task .codeinsight+research/kernel-atlas/tasks/example-gemv/task.json \
  --runs 5 --warmup-runs 1 \
  --refresh-baseline \
  --output-dir .codeinsight+research/kernel-atlas/tasks/example-gemv/eval-baseline

python3 scripts/kernel_atlas.py eval \
  --task .codeinsight+research/kernel-atlas/tasks/example-gemv/task.json \
  --baseline .codeinsight+research/kernel-atlas/tasks/example-gemv/eval-baseline/baseline.json \
  --runs 5 --warmup-runs 1 \
  --output-dir .codeinsight+research/kernel-atlas/tasks/example-gemv/eval-001
```

PyTorch-shape task shell (no automatic PyTorch kernel extract yet):

```bash
python3 scripts/kernel_atlas.py task-pytorch \
  --name example-rmsnorm-shape \
  --op rmsnorm \
  --input-shape 1,2048,4096 \
  --dtype float16 \
  --eval-command 'python3 bench_rmsnorm.py' \
  --allowed-file kernels/src/rmsnorm_candidate.hip \
  --output-dir .codeinsight+research/kernel-atlas/tasks/example-rmsnorm-shape
```

## Interpretation rules

- **ISA fit ≠ occupancy.** Counters, residency, clocks, cache, and launch
  overlap are out of band unless separately measured.
- Matrix units present with zero observed matrix ops → ask whether the phase
  should use WMMA/MFMA or is a memory/launch-dominated decode GEMV.
- High VGPR/SGPR/spills → register pressure before bandwidth narratives.
- DFlash rows: tok/s is not correctness. Use VALIDATION’s claim-scoped route
  (serve semantics, path oracle, etc.) — not Atlas alone, and not retired
  batteries as current acceptance.
- `eval` `unstable` → measurement failure until run shape / thermal / DPM settles.
- Multi-host (e.g. hiptrx vs another box): compare only when prompt, binary md5,
  git diff md5, model, and variant env match; pin arch with
  `HIPFIRE_TARGET_ARCH` / `ROCR_VISIBLE_DEVICES` when multiple GPUs are visible.

## After Atlas, before any ship claim

1. Rerun the **VALIDATION** route for the actual claim class (kernel channel,
   path oracle, serve harness, speed-gate / probe, Redline ladder — as applicable).
2. Follow perf protocol for any tok/s delta (fresh process, warmup, identity).
3. Do not write Atlas medians into product docs as floors or admissions.
4. Kernel-edit workflow after a hot kernel is identified: skill
   `hipfire-kernel-tuning`. New ISA targets: `hipfire-arch-port`.

## Good agent output

Include:

- rendered fit section (or path to full render)
- row path, row index, ISA/dispatch manifest paths
- arch, quant, phase, shape bucket
- runtime metric used for the readout
- one short reading of `likely limit` / `left on table`
- explicit label: **measured evidence, not promotion proof**

Avoid:

- calling the heuristic a roofline model
- claiming a perf win from smoke or single dirty-tree rows
- mixing prompts/binaries without saying so
- citing Atlas success as merge/admission authority
