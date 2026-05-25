# Results — quant-quality eval output tables

Each eval cohort lands as a dated subdirectory:

```
results/
├── 2026-MM-DD/
│   ├── per-seq/                     # raw per-sequence-KLD files (HFKSEQ)
│   │   ├── qwen3.5-9b.mq4-uniform__gfx1100.kldseq
│   │   ├── qwen3.5-9b.mq3-uniform__gfx1100.kldseq
│   │   └── ... (one per variant × arch)
│   ├── result-table.md              # markdown table (kld_reduce.py output)
│   ├── result-data.json             # same data, JSON for plot scripts
│   └── 2026-MM-DD-quant-pareto.md   # human-written write-up + Pareto plot ref
```

Cohorts are not interchangeable — changing the slice, n_ctx, or eval-mode
flags starts a new dated cohort. Don't try to merge old and new numbers;
treat as a fresh baseline.

The first cohort lands after Step 4-9 of the plan. Currently empty.
