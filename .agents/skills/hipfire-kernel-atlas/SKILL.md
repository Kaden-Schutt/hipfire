---
name: hipfire-kernel-atlas
description: Use Kernel Atlas to collect phase-aware hipfire measurements and render ISA Fit View visualizations for AMD GPU kernels, quant formats, and architectures. Use when a user asks how MQ/HFQ/HFP/Q8 quants occupy hardware, asks for an ASCII ISA visualization, wants to compare gfx1010/gfx1030/gfx11/gfx12 kernel fit, or wants an agent-readable "left on table" summary from Atlas rows.
---

# hipfire-kernel-atlas

Use this skill when the task is to explain or visualize how a hipfire quant
format and kernel use an AMD GPU ISA target. The primary tool is
`scripts/kernel_atlas.py`; this skill is a thin agent wrapper around that CLI.

## Core Workflow

1. **Collect or locate Atlas rows**
   - Prefer existing JSONL under `.codeinsight+research/kernel-atlas/runs/`.
   - For AR prefill/decode, collect with `collect-ar`.
   - Use `--profile-prefill` / `--profile-decode` for AR rows when the user wants the ISA view scoped to runtime-hot kernels and tagged by op role.
   - For speculative decode, collect with `collect-dflash`.
   - Keep raw run data in `.codeinsight+research/`; it is ignored and may be private.

2. **Attach ISA metadata**
   - Use `--isa-file` for one known HSACO/code object.
   - Use `--isa-dir .hipfire_kernels/<arch>` plus `--isa-filter` for a bounded set.
   - Prefer `--isa-output <path>.json` so multiple rows reference one manifest.

3. **Attach dispatch/source provenance**
   - Use `--dispatch-provenance` when rows have profiled kernel names.
   - Prefer `--dispatch-output <path>.json` so multiple rows reference one manifest.
   - Treat dispatch references as evidence to inspect, not proof of a unique runtime branch.
   - Prefer rows with a known `arch`; source ranking is target-arch-aware when arch-specific kernel files exist.

4. **Render the ISA Fit View**
   - Use `.agents/skills/hipfire-kernel-atlas/render-fit.sh`.
   - If a row has `artifacts.profile_kernels`, the view joins profiled kernel names to ISA object kernel names/symbols and summarizes only matched objects.
   - If a row has dispatch provenance, the view prints hot-kernel op/source/dispatch attribution.
   - Report the visual plus a short readout of `likely limit` and `left on table`.

5. **Ask Atlas for candidate experiments**
   - Use `python3 scripts/kernel_atlas.py suggest --row ... --isa ... --dispatch ...`.
   - Prefer `--format markdown` for humans and JSON for automation.
   - Let `suggest` auto-load default history from `.codeinsight+research/kernel-atlas/tasks/`; use `--history` only for extra history paths.
   - Treat suggestions as an experiment queue, not as predicted wins.
   - Each suggestion should name the lever type, hot kernel, files, risk, rationale, and eval contract.

6. **Create an optimization task**
   - Use `python3 scripts/kernel_atlas.py task` to turn a row into `task.json` and `TASK.md`.
   - Include `--allowed-file` for every path an agent may edit.
   - Include correctness commands for DFlash or risky runtime changes.
   - Generated tasks strip known profiling/instrumentation env from eval and preserve the original row env as `baseline.row_env`.

7. **Evaluate a candidate**
   - Use `python3 scripts/kernel_atlas.py eval --task ... --runs 5 --warmup-runs 1 --output-dir ...`.
   - Use `--refresh-baseline` first to write `baseline.json`; use `--baseline <baseline.json>` for candidate comparisons.
   - Report `result.json` status, selected metric median, speedup, stability, and any failed command output tail.
   - Treat the local `ledger.jsonl` as experiment lineage, not a public benchmark.
   - If status is `needs_baseline`, do not claim a speedup; refresh or provide a clean baseline first.

## Commands

Render an existing row:

```bash
.agents/skills/hipfire-kernel-atlas/render-fit.sh \
  --row .codeinsight+research/kernel-atlas/runs/atlas.jsonl \
  --row-index 0 \
  --isa .codeinsight+research/kernel-atlas/runs/isa.json
```

Collect a small AR smoke with ISA:

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
  --isa-dir .hipfire_kernels/gfx1030 \
  --isa-filter 'gemm_hfq4g256|gemv_hfq4g256' \
  --isa-output .codeinsight+research/kernel-atlas/runs/isa-gfx1030.json \
  --dispatch-provenance \
  --dispatch-output .codeinsight+research/kernel-atlas/runs/dispatch-gfx1030.json \
  --output .codeinsight+research/kernel-atlas/runs/atlas-gfx1030.jsonl
```

Suggest candidate experiments from a profiled row:

```bash
python3 scripts/kernel_atlas.py suggest \
  --row .codeinsight+research/kernel-atlas/runs/atlas-gfx1201.jsonl \
  --row-index 1 \
  --isa .codeinsight+research/kernel-atlas/runs/isa-gfx1201.json \
  --dispatch .codeinsight+research/kernel-atlas/runs/dispatch-gfx1201.json \
  --format markdown
```

Create a bounded task from a profiled row:

```bash
python3 scripts/kernel_atlas.py task \
  --row .codeinsight+research/kernel-atlas/runs/atlas-gfx1201.jsonl \
  --row-index 1 \
  --isa .codeinsight+research/kernel-atlas/runs/isa-gfx1201.json \
  --dispatch .codeinsight+research/kernel-atlas/runs/dispatch-gfx1201.json \
  --allowed-file kernels/src/gemv_hfq4g256_multirow.hip \
  --output-dir .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4
```

Create a PyTorch-shape task for non-Qwen work:

```bash
python3 scripts/kernel_atlas.py task-pytorch \
  --name llama-rmsnorm-shape \
  --op rmsnorm \
  --input-shape 1,2048,4096 \
  --dtype float16 \
  --eval-command 'python3 bench_rmsnorm.py' \
  --allowed-file kernels/src/rmsnorm_candidate.hip \
  --output-dir .codeinsight+research/kernel-atlas/tasks/llama-rmsnorm-shape
```

Refresh a stable baseline and then evaluate a candidate:

```bash
python3 scripts/kernel_atlas.py eval \
  --task .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4/task.json \
  --runs 5 \
  --warmup-runs 1 \
  --refresh-baseline \
  --output-dir .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4/eval-baseline

python3 scripts/kernel_atlas.py eval \
  --task .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4/task.json \
  --baseline .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4/eval-baseline/baseline.json \
  --runs 5 \
  --warmup-runs 1 \
  --output-dir .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4/eval-001
```

## Interpretation Rules

- Treat the view as **ISA fit**, not full hardware occupancy. True occupancy
  also needs counters, wave residency, clocks, cache behavior, and launch
  overlap.
- If matrix units are available but observed matrix ops are zero, ask whether
  the workload phase should route through WMMA/MFMA or whether it is a decode
  GEMV path where memory/launch dominates.
- If VGPR/SGPR/spills are high, prioritize register pressure and spill removal
  before claiming a bandwidth win.
- If the row is DFlash, do not treat tok/s alone as correctness evidence. Run
  the DFlash coherence gate before claiming a spec-decode improvement.
- If `eval` reports `unstable`, do not claim a win or regression; tighten the
  run shape or rerun after DPM/thermal state settles.
- For PyTorch-shape tasks, treat the eval command as the source of truth until
  Atlas has a real PyTorch profiler/extractor producer.
- If the worktree is dirty, cite the row's `provenance.diff_md5` and avoid
  comparing it as a shipped baseline.

## Good Agent Output

Include:

- the rendered ASCII fit view, or the most relevant section of it
- the row path and ISA manifest path
- arch, quant, phase, and shape bucket
- runtime metric used for the readout
- one concise interpretation of `likely limit` and `left on table`

Avoid:

- calling the heuristic a roofline model
- claiming a perf win from smoke runs
- mixing rows from different prompts or dirty binaries without saying so
