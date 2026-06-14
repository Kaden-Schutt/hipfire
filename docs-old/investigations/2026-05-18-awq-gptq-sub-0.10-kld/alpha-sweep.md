# Alpha Sensitivity Sweep — Paper Formula + AWQ-aware GPTQ

5-point sweep across α ∈ {no-AWQ, 0.3, 0.5, 0.7, 1.0}, all with AWQ-aware GPTQ on top, c512 q8 prefill on gfx1201.

## Raw data

| α | KLD | KLD 95% CI lo | KLD 95% CI hi | p99 KLD | PPL | NLL mean |
|---:|---:|---:|---:|---:|---:|---:|
| no AWQ | 0.2686 | 0.2581 | 0.2795 | 17.87 | 8.905 | 2.1866 |
| 0.30 | 0.1514 | 0.1440 | 0.1592 | 14.26 | 9.340 | 2.2343 |
| **0.50** | **0.1257** | **0.1196** | **0.1322** | **13.71** | 9.310 | 2.2313 |
| 0.70 | 0.1663 | 0.1584 | 0.1748 | 15.33 | 8.965 | 2.1933 |
| 1.00 | 0.2032 | 0.1945 | 0.2120 | 16.00 | 8.885 | 2.1844 |

## Interpretation

**KLD curve**: convex U-shape with global minimum at α=0.5. Going either direction (less or more aggressive AWQ) strictly increases KLD.

**PPL curve**: monotonically decreasing toward higher α. Best PPL is at α=1.0 (8.885), not α=0.5. The KLD-PPL inversion that PR #273 documented for F2 is also present for F1 — it's a property of paper-formula AWQ stacking with GPTQ in general, not just F2's whitelist scope.

**Cross-axis**: no single α minimizes both KLD and PPL simultaneously. α=0.5 wins KLD; α=0.7-1.0 wins PPL. This is the signature of *per-tensor* optimum heterogeneity — different layers want different α.

## What this tells us about the next move

1. **Global α tuning is exhausted.** Further sweeps in the [0.4, 0.6] range would yield marginal KLD changes within the CI of v3 (0.0006 range).
2. **Per-tensor reasoning required.** The KLD-PPL inversion is direct evidence that some tensors want lower α (more AWQ → KLD-friendly) while others want higher α (less AWQ → PPL-friendly).
3. **Iterative refinement is the cleanest formulation of "per-tensor"**, because each round's per-tensor reconstruction error (computed under the current-round's scales) implicitly tunes per-tensor sensitivity through the KM fixed-point iteration.

## How this curve was produced

3-GPU parallel sweep (GPU 0/1/2 fired simultaneously, GPU 3 added for α=1.0 separately):

```bash
# Each cell quantizes its own AWQ base at the target α, runs AWQ-aware GPTQ on GPU N,
# then benches c512 q8 with ROCR_VISIBLE_DEVICES=N.
# Total wall: 30 min (4 cells in parallel × 30 min each = 30 min)

# Cell template (substitute $ALPHA, $GPU, $LABEL):
./target/release/hipfire-quantize \
  --input <BF16> --output ${BASE}-${LABEL}-base \
  --format mq4 --awq --awq-alpha $ALPHA --imatrix $IMATRIX

$PYTHON scripts/mq4_masked_calib.py quantize \
  --base ${BASE}-${LABEL}-base \
  --source-dir $BF16_DIR --mask $MASK --stats-npz $HESSIAN \
  --output ${BASE}-${LABEL}-gptq \
  --method gptq --gpu $GPU \
  --awq-aware-hessian ${BASE}-${LABEL}-base \
  --tensor-filter "lm_head,in_proj_a,..." --skip-unsupported

ROCR_VISIBLE_DEVICES=$GPU ./target/release/examples/eval_hipfire \
  --model ${BASE}-${LABEL}-gptq --ref $REF \
  --output per-seq-c512/${LABEL}__gfx1201__prefill.kldseq \
  --kv-mode q8 --scoring-mode prefill --max-chunks 512
```

The "no AWQ" cell skips the AWQ base quantize step and uses the existing flat-mq4 as the GPTQ base.

## Reproducibility

All 5 cells share:
- Same calibration imatrix (unsloth-published, 184-tensor coverage on 9B)
- Same Hessian (`stats-merged.npz`, c64 chunks at ctx=256)
- Same GPTQ mask (67 linear tensors per REPORT-9b.md)
- Same engine binary (origin/master post-PR-#273 + `awq-aware-hessian` branch)
- Same bench harness (eval_hipfire, BF16 reference dump)
