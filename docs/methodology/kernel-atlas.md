# Kernel Atlas MVP

Kernel Atlas starts as a phase-aware measurement corpus for hipfire kernels.
The first harness does not generate kernels or rewrite dispatch policy. It
turns existing benchmark output into JSONL rows that can later feed ranking,
autotuning, or an advisor model.

## Phases

Each row has exactly one phase:

- `prefill`: AR prompt processing from `bench_qwen35_mq4 --prefill N`.
- `decode_ar`: target-only AR generation from the same bench run.
- `decode_dflash`: speculative decode from `dflash_spec_demo`.

This keeps prefill visible for users who never enable DFlash, while preserving
DFlash-specific metrics like acceptance and tau.

## Private corpus location

Keep raw Atlas runs out of git:

```bash
mkdir -p .codeinsight+research/kernel-atlas/runs
```

That directory is already ignored. Commit only harness changes, summarized
findings, or carefully redacted public samples.

## AR collection

Example for a 27B MQ4 AR prefill/decode capture:

```bash
python3 scripts/kernel_atlas.py collect-ar \
  --model ~/.hipfire/models/qwen3.5-27b.mq4 \
  --workload qwen3.5-27b \
  --model-size 27b \
  --quant mq4 \
  --prefill 32 \
  --prefill 128 \
  --gen 50 \
  --kv-mode asym3 \
  --graph \
  --output .codeinsight+research/kernel-atlas/runs/$(date -u +%Y%m%dT%H%M%SZ)-ar.jsonl
```

The harness records one `prefill` row and one `decode_ar` row per bench run.
Use `--env KEY=VALUE` for variant knobs such as `HIPFIRE_GEMV_ROWS=4`.
Use `--profile-prefill` and `--profile-decode` when you want Atlas to
capture the bench's per-kernel profile tables into `artifacts.profile_kernels`.
Those kernel rows also get a first-pass op attribution such as
`linear.residual_gemv`, `attention.qkvza_projection`, or `norm.normalization`.

Atlas rows also record benchmark binary md5, git dirty state, and a diff md5.
That provenance is required before comparing dirty-worktree measurements
against committed baselines.

## DFlash collection

Example for the canonical merge-sort DFlash shape:

```bash
python3 scripts/kernel_atlas.py collect-dflash \
  --target ~/.hipfire/models/qwen3.5-27b.mq4 \
  --draft ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq \
  --prompt-file benchmarks/prompts/merge_sort_thinking_off.txt \
  --workload qwen3.5-27b-dflash-merge-sort \
  --max-tokens 256 \
  --ctx 2048 \
  --kv-mode asym3 \
  --output .codeinsight+research/kernel-atlas/runs/$(date -u +%Y%m%dT%H%M%SZ)-dflash.jsonl
```

DFlash rows always record prompt md5 plus `decode_tok_s`, tau, TTFT, emitted
tokens, cycles, and accepted tokens when the demo prints them. Target and draft
model md5s are opt-in via `--hash-models` because hashing a 15 GB target on
every capture makes the harness too slow for routine sweeps.

## ISA manifests

Atlas can attach real ISA information from compiled HSACO/code-object files.
This is opt-in because inspecting every kernel in `.hipfire_kernels/` is more
expensive and produces larger rows.

Inline a small manifest into each row:

```bash
python3 scripts/kernel_atlas.py collect-ar \
  --model ~/.hipfire/models/qwen3.5-0.8b.mq4 \
  --workload qwen3.5-0.8b \
  --model-size 0.8b \
  --prefill 32 \
  --gen 5 \
  --isa-dir .hipfire_kernels \
  --isa-filter 'gemm_hfq4g256_residual' \
  --isa-limit 1 \
  --output .codeinsight+research/kernel-atlas/runs/atlas-with-isa.jsonl
```

Write the ISA manifest once and reference it from each row:

```bash
python3 scripts/kernel_atlas.py collect-ar \
  --model ~/.hipfire/models/qwen3.5-27b.mq4 \
  --workload qwen3.5-27b \
  --model-size 27b \
  --prefill 32 \
  --prefill 128 \
  --gen 50 \
  --isa-dir .hipfire_kernels \
  --isa-filter 'gemv_hfq4g256|fused_qkv' \
  --isa-output .codeinsight+research/kernel-atlas/runs/isa-gfx1201.json \
  --output .codeinsight+research/kernel-atlas/runs/atlas-gfx1201.jsonl
```

The ISA manifest records:

- source HSACO/code-object path and md5
- offload bundle target when present
- `amdhsa.target`
- per-kernel VGPR, SGPR, LDS/group segment, private segment, spills,
  workgroup size, and wavefront size from `llvm-readobj --notes`
- instruction count, opcode counts, category counts, and kernel symbols from
  `llvm-objdump -d --no-show-raw-insn`

Bundled hipcc outputs are unbundled with `clang-offload-bundler` before
inspection, so the recorded metadata is from the actual AMDGPU code object
loaded by HIP.

## Dispatch provenance

Atlas can also attach a dispatch/source manifest for the profiled kernels:

```bash
python3 scripts/kernel_atlas.py collect-ar \
  --model ~/.hipfire/models/qwen3.5-27b.mq4 \
  --workload qwen3.5-27b \
  --model-size 27b \
  --prefill 32 \
  --gen 50 \
  --profile-prefill \
  --profile-decode \
  --dispatch-provenance \
  --dispatch-output .codeinsight+research/kernel-atlas/runs/dispatch-gfx1201.json \
  --output .codeinsight+research/kernel-atlas/runs/atlas-gfx1201.jsonl
```

The dispatch manifest is deliberately evidence-oriented. For each profiled
kernel name it records:

- source candidates under `kernels/src/`, with md5s
- dispatch/source references under `crates/`, `cli/`, and `kernels/src/`
- inferred env controls such as `HIPFIRE_GEMV_ROWS`
- the same op attribution stored on profile rows

This does not prove a unique runtime branch by itself. It gives agents and
developers enough evidence to ask the right next question: whether the hot
runtime kernel really came from the source/dispatch path being tuned.

When `collect-ar` or `collect-dflash` knows the target arch, dispatch/source
ranking is arch-aware. For example, a `gfx1201` row prefers
`gemv_hfq4g256.gfx1201.hip` over the generic `gemv_hfq4g256.hip` when the
arch-specific source exists. If no target-arch source exists, Atlas falls back
to the exact generic kernel source instead of substring matches or stale docs.

## ISA fit view

Render a terminal view that combines an Atlas row with an ISA manifest:

```bash
.agents/skills/hipfire-kernel-atlas/render-fit.sh \
  --row .codeinsight+research/kernel-atlas/runs/atlas-gfx1201.jsonl \
  --row-index 0 \
  --isa .codeinsight+research/kernel-atlas/runs/isa-gfx1201.json \
  --dispatch .codeinsight+research/kernel-atlas/runs/dispatch-gfx1201.json
```

If the row already references `artifacts.isa.manifest_path`, `--isa` is
optional. If the row already references `artifacts.dispatch.manifest_path`,
`--dispatch` is optional. The view is intentionally heuristic: it visualizes
arch capability, observed ISA mix, resource shape, quant intent, hot-kernel
op/source attribution, and a first-pass "left on table" interpretation. It is
not a replacement for hardware counters or full occupancy modeling.

When a row contains `artifacts.profile_kernels`, `render-fit` first joins those
profiled kernel names to ISA object kernel names/symbols and scopes the ISA
summary to the matched objects. It also prints unmatched hot names so a user
can tell when the manifest filter missed the runtime kernel. Without profile
names, the view falls back to all inspected ISA objects.

## Task and eval loop

`suggest` turns the same row + ISA + dispatch evidence into a ranked queue of
candidate experiments:

```bash
python3 scripts/kernel_atlas.py suggest \
  --row .codeinsight+research/kernel-atlas/runs/atlas-gfx1201.jsonl \
  --row-index 1 \
  --isa .codeinsight+research/kernel-atlas/runs/isa-gfx1201.json \
  --dispatch .codeinsight+research/kernel-atlas/runs/dispatch-gfx1201.json \
  --format markdown
```

Suggestions are not perf claims. Each item records a lever type, hot kernel,
risk, expected impact, allowed files, rationale, candidate steps, and the eval
contract inherited from the row. Use them to queue empirical candidates, then
turn the chosen item into a scoped edit and validate with `eval`.

`suggest` automatically scans `.codeinsight+research/kernel-atlas/tasks/` for
prior Atlas `result.json`, `ledger.jsonl`, and `task*.json` files. Matching
stable losers are annotated as history-backed rejections and demoted before the
queue is ranked, so repeated runs move on to untested levers. Use `--history
PATH` only when adding another history file or directory outside the default
task tree.

`task` turns a row into a bounded optimization contract for an agent or human:

```bash
python3 scripts/kernel_atlas.py task \
  --row .codeinsight+research/kernel-atlas/runs/atlas-gfx1201.jsonl \
  --row-index 1 \
  --isa .codeinsight+research/kernel-atlas/runs/isa-gfx1201.json \
  --dispatch .codeinsight+research/kernel-atlas/runs/dispatch-gfx1201.json \
  --allowed-file kernels/src/gemv_hfq4g256_multirow.hip \
  --correctness-command './scripts/coherence-gate-dflash.sh' \
  --output-dir .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4
```

The task directory contains `task.json` for tooling and `TASK.md` for an
agent-readable brief. It includes the hot kernel, op attribution, source and
dispatch refs, matched ISA objects, baseline metrics, allowed write files, and
the exact benchmark/correctness commands.

Profiled rows intentionally carry profiling env such as `HIPFIRE_PROFILE=1`
and `HIPFIRE_PROFILE_DECODE=1`. Generated tuning tasks strip known
instrumentation env from the eval environment and preserve it separately as
`baseline.row_env`. When anything is stripped, `eval.requires_fresh_baseline`
is true. Run `eval --refresh-baseline` before comparing a candidate; otherwise
Atlas reports `needs_baseline` instead of computing a misleading speedup
against a profiled row metric.

After a candidate edit, `eval` reruns the task contract and appends a local
ledger entry:

```bash
python3 scripts/kernel_atlas.py eval \
  --task .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4/task.json \
  --runs 5 \
  --warmup-runs 1 \
  --refresh-baseline \
  --output-dir .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4/eval-baseline
```

`eval` writes `result.json` plus `ledger.jsonl`, recording pass/fail status,
the selected metric, speedup against the task baseline when available, git diff
md5, command output tails, all benchmark runs, and median/min/max/stdev/MAD
summaries. `--refresh-baseline` also writes `baseline.json` and compares the
run against that refreshed median baseline.

For candidate evaluations, compare against a stable baseline:

```bash
python3 scripts/kernel_atlas.py eval \
  --task .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4/task.json \
  --baseline .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4/eval-baseline/baseline.json \
  --runs 5 \
  --warmup-runs 1 \
  --output-dir .codeinsight+research/kernel-atlas/tasks/gfx1201-gemv-r4/eval-001
```

If `(max - min) / median` for the selected metric exceeds
`--max-rel-spread` (default `0.20`), the result status becomes `unstable`.
Treat unstable results as measurement failures until the run shape, DPM state,
prompt, or benchmark command is tightened.

## PyTorch shape tasks

Atlas can also emit a PyTorch-shape task without a hipfire row. This is the
first step toward PyTorch-to-HIP and lm-eval-like producer workflows for
non-Qwen tensor shapes:

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

This does not yet extract kernels from PyTorch automatically. It gives the
same task/eval/ledger contract a PyTorch producer shape so later model-profile
or lm-eval-style harnesses can feed Atlas without pretending every workload is
Qwen decode.

## hiptrx usage

On `hiptrx`, keep raw task/eval artifacts under
`.codeinsight+research/kernel-atlas/` in the active worktree. For gfx1201, use
`HIPFIRE_TARGET_ARCH=gfx1201` or the appropriate `ROCR_VISIBLE_DEVICES` value
in the task/eval environment if multiple GPUs are visible. Do not compare a
hiptrx row to hipx rows unless the prompt, binary md5, git diff md5, model,
and variant env match.

Agent-facing instructions live in `.agents/skills/hipfire-kernel-atlas/`. Agents
should use that skill when a user asks for a visual readout of how a quant or
kernel occupies the target ISA.

## Validation rule

Atlas rows are evidence, not proof of a shippable win. Before using a row to
justify a kernel or dispatch change, rerun the relevant speed gate and, for
DFlash, the coherence gate with byte-identical prompts.
