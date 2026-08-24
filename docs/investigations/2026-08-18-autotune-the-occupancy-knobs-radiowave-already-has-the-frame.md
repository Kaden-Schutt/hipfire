# The occupancy knobs are hand-written; radiowave already autotunes everything else

**Date:** 2026-08-18 · **Status:** opportunity assessment, not a measurement record.
Measurements backing it live in
[`../perf-checkpoints/2026-08-18-vgpr-capping-is-a-pessimization-occupancy-is-not-throughput.md`](../perf-checkpoints/2026-08-18-vgpr-capping-is-a-pessimization-occupancy-is-not-throughput.md).

## The observation

`crates/radiowave` is already an autotuning campaign framework, and it already runs:

- `main.rs` exposes a `campaign` subcommand.
- `campaign.rs` — append-only JSONL round accounting. Infrastructure failures do not
  consume a round; an identical candidate identity can never consume two completed
  rounds; a configuration SHA is paired with the exact code-object SHA as identity.
- `recipes.rs` — architecture-neutral recipes promoted per-architecture by
  `RecipeEvidence`, explicitly so "autoresearch [can] try the same candidate on a new
  target without silently inheriting another GPU's winner."
- `oracle.rs` — correctness certification.
- `SchedulerProfile` — `Default`, `MaxIlp`, `IterativeIlp`, `MemoryClause`,
  `PipelineIlp`.

Candidate dimensions swept today: **`scheduler_profile`** and **`unroll`**.

Not swept, and hand-written instead:

| knob | how it is set | scale |
|---|---|---|
| `__launch_bounds__` 2nd argument | hand-written per kernel | **540 files** (168 `(32,2)`, 155 `(32,16)`, 57 `(32,20)`, 56 `(32,8)`, …) |
| `__attribute__((amdgpu_num_vgpr(N)))` | hand-written | 1 file, defaulted off |

## Which knob is worth adding

Measured on the v1 residual GEMV (gfx1201, m=k=5120):

| knob | nature | measured range |
|---|---|---|
| `__launch_bounds__(32, N)` | **advisory** — the compiler emitted 93 VGPR / occupancy 16 for every N tried and merely reordered | 6.3% spread; shipped value within **1.5%** of best; apparent optimum inside run noise |
| `amdgpu_num_vgpr(N)` | **hard cap** — forces spilling or a scheduler re-plan | up to **1.28×** |

So **add `num_vgpr`, not `launch_bounds`.** The advisory knob has a low ceiling and a
sweep would mostly re-derive the existing defaults while consuming GPU rounds. The
hard cap has a 28% range; on the one kernel measured it swung the *wrong* way, which
is exactly the argument for searching it rather than assuming a direction.

The plumbing is small: `CompileRequest` already has `.define()` and
`.scheduler_profile()`, and `kernels/src/gemv_mq4cg256_residual.hip` deliberately
takes its cap from `HIPFIRE_MQ4C_VGPR_CAP` (default `0` = no attribute) rather than
hardcoding it, so it is already a sweepable dimension.

## Two conditions, or this ships regressions

1. **Noise is the same size as the signal.** Thermal drift on these kernels is ~6%
   and grows with run count (samples spanned 15.5–25.5 µs over 7 runs), while the
   launch_bounds signal is ~6%. An autotuner without interleaved sample-by-sample
   comparison, ≥32 warmups, fresh processes, and minimum-not-mean statistics will
   chase drift and promote settings *worse* than the hand-picked constants. The
   campaign framework provides reproducible candidate **identity**; it does not
   provide reproducible **measurement**. That discipline has to come from the bench —
   see `crates/rdna-compute/examples/bench_vgpr_cap_sweep.rs`.
2. **Correctness must gate promotion, via `oracle.rs`.** This campaign shipped a
   quant format that ran at 163 tok/s and returned pure noise; and among the cap
   variants, `num_vgpr(96)` *spilled* and still produced plausible timings. An
   autotuner optimising throughput alone will promote a fast wrong kernel. Candidates
   need a parity oracle (`mq4v2_parity`, `mq4v2_gemm_parity`), not compile-and-run.

## Scope worth doing

Restrict to the ~13 translation units on the dense decode/prefill path rather than all
540 files. Per-arch is where hand-tuning genuinely fails to scale: gfx1030 already
behaves oppositely to gfx1201 on the v2 header (14–18% regression there, because
Infinity Cache makes that kernel VALU-bound rather than bandwidth-bound), and
13 kernels × 5 architectures is 65 hand-decisions nobody will make well.
`RecipeEvidence`'s per-architecture promotion is built for exactly this.

## Why the intuition being corrected here is worth writing down

The generalisable finding is that **occupancy is a means to memory-level parallelism,
not a goal**. Bytes in flight = waves × outstanding-loads-per-wave, and a
bandwidth-bound kernel only cares about the product. The clean evidence is not
"occupancy 12 was fine" but that at **fixed** occupancy 16, throughput still varied
1.04×–1.28× with register count alone. Any autotuner scoring candidates by occupancy,
VGPR count, or spill count instead of by measured throughput will optimise the wrong
objective — the uncapped 97-VGPR / occupancy-12 build was the fastest configuration
tried and looked worst on every static metric.
