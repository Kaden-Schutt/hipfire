# RoPE frequency table diff (model.py vs parent)

Date: 2026-08-02. Script: `rope_table_diff.py`, `apply_rope_diff.py`.

## Config binding (confirmed against model.py Attention.__init__ 482-487)

| Path | original_seq_len | base (theta) | YaRN |
|------|------------------|--------------|------|
| A ratio-0 SWA | 0 | 10000 | OFF |
| B ratio>0 main Q/KV + compressor | 65536 | 160000 | ON factor=16 βf=32 βs=1 |
| Indexer Q | **same as B** (shares `attn.freqs_cis`) | 160000 | ON |

Parent wiring matches: `attention.rs:1008-1025`, `indexer.rs:681-696` (YaRN on),
`compressor.rs:822-832`. Stale comment in `indexer.rs:20` saying "plain" is wrong;
code is correct.

## Table element-wise f64 relative error

| Config | max_rel | argmax dim | max phase err @ pos1000 |
|--------|---------|------------|-------------------------|
| A plain 10k | 6.97e-8 | — | **1.05e-5 rad** |
| B YaRN 160k | 6.36e-8 | 31 | **2.30e-5 rad** |
| C plain 160k (counterfactual) | 6.36e-8 | — | 2.30e-5 rad |

YaRN correction range (low, high) = **(15, 25)** exact match.

Phase errors are ~1e-5 rad — **orders of magnitude below** the ~1 rad
threshold that would destroy a dimension. **RoPE tables are not the defect.**

## YaRN vs plain magnitude (same base 160k)

If parent had wrongly used plain-C for ratio>0, max_rel would be **15×** on
high dims (i=25..31) with phase@1000 up to **0.33 rad**. Parent does **not**
do this — both code paths use YaRN for ratio>0.

## Apply path

`apply_rope_interleaved` vs `model.py apply_rotary_emb` on full positions
0..1023 YaRN: max_abs **2.4e-7** (f32 cos/sin noise), no growth with position.

Artifacts: `artifacts/rope_table_diff.json`, `artifacts/apply_rope_diff.json`.
