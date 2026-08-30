# Radiowave sweep — MQ4V2 MMQ residual (gfx1151)

**TU:** `kernels/src/gemm_mq4g256v2_residual_mmq.hip` (production) + `kernels/src/gemm_mq4g256v2_residual_mmq_kvariants.hip` (candidate k2/k4/ksplit)
**Arch:** gfx1151 (HIP_VISIBLE_DEVICES=1 on hipx)
**Lever:** scheduler_profile × unroll — cheapest lever per 2026-08-18 tuning-debt amendment; v1's 54 residual variants vs 3 for qt44 justified this sweep before any container change.
**Coverage:** 5 scheduler profiles (Default, MaxIlp, IterativeIlp, MemoryClause, PipelineIlp) × 4 unroll settings = 20 codes per TU. PipelineIlp includes relaxed-occupancy + exact-solver + pipeliner (see `crates/radiowave/src/lib.rs:SchedulerProfile::llvm_args`).

## Invocation

```bash
# hipx, gfx1151 is device 1 (HIP_VISIBLE_DEVICES=1)
HIP_VISIBLE_DEVICES=1 bash scripts/radiowave_mq4v2_mmq_sweep.sh
# outputs under /tmp/radiowave_mq4v2_mmq/ : *.hsaco + *.radiowave.json + *.inspect.json
```

Or per-combination manually:

```bash
cargo run -p radiowave -- compile \
  --source kernels/src/gemm_mq4g256v2_residual_mmq.hip \
  --output /tmp/mq4v2_mmq_default_u1.hsaco \
  --arch gfx1151 --wave32 --scheduler-profile default --define HIPFIRE_SWEEP_UNROLL=1

cargo run -p radiowave -- compile \
  --source kernels/src/gemm_mq4g256v2_residual_mmq.hip \
  --output /tmp/mq4v2_mmq_pipeline-ilp_u8.hsaco \
  --arch gfx1151 --wave32 --scheduler-profile pipeline-ilp --define HIPFIRE_SWEEP_UNROLL=8
# repeat for max-ilp, iterative-ilp, memory-clause
# same for _kvariants.hip (covers the three candidate symbols in one TU)
```

## Campaign wiring (optional, not required this wave)

```bash
cargo run -p radiowave -- recipes builtin --output /tmp/catalog.json
cargo run -p radiowave -- recipes select --arch gfx1151 --kernel gemm_mq4g256v2_residual_mmq --catalog /tmp/catalog.json --candidates
# after timing, ingest winning JSONL rows:
cargo run -p radiowave -- recipes ingest --catalog /tmp/catalog.json --ledger /tmp/mq4v2_kvariants_run1.jsonl --output /tmp/catalog_promoted.json
```

## Timing + parity (HipxMeasure owns GPU)

```bash
# 3 fresh-process runs per arm, byte-identical fixtures, device events, median is lead statistic
HIP_VISIBLE_DEVICES=1 cargo run --release -p rdna-compute --example bench_mq4v2_mmq_kvariants_gfx11 2>&1 | tee /tmp/mq4v2_kvariants_run1.jsonl
# repeat run1..run3, keep whole JSON outputs
```

Record with each table: model md5 `e45d15bfe0c9a87132697101d17cbed6`, binary md5, commit hash.

## Kill bars

- Port axis (k2/k4/ksplit): < +10% kernel-level at contract shapes (M=17408/5120, K=5120/17408, N=512/2048) => DEAD.
- Radiowave axis: < +5% vs incumbent Default/MaxIlp after sweep => DEAD.

No production dispatch changes ship in this wave; sweep is measurement-only.
