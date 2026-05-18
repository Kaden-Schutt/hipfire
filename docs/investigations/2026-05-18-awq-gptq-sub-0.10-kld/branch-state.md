# Branch State — Where the Code Lives

## Branch tree

```
origin/master  (a99b4643, post-PR-#273 F2 AWQ Stage A)
    │
    ├─ awq-aware-hessian  (85f6d055 latest, contains GPU port + AWQ-aware Hessian
    │   │                   + source pre-scale fix + opt-in AutoAWQ formula)
    │   │
    │   ├─ 442f48ca: feat(quant): add --gpu N torch path to mq4 GPTQ solve functions
    │   ├─ c5825848: feat(quantize): AWQ-aware Hessian transform for GPTQ stacking
    │   ├─ f0bcfabd: fix(quantize): AWQ-aware GPTQ also pre-scales source weights
    │   └─ 85f6d055: feat(quantize): AutoAWQ formula with weight-magnitude term
    │
    └─ iterative-awq-gptq  (13f8fb3d latest, includes all of awq-aware-hessian)
        ├─ f286bade: feat(quantize): iterative AWQ+GPTQ refinement rounds (Codex impl)
        └─ 13f8fb3d: fix(iterate): propagate skip_unsupported + tensor_filter
```

## Key commits

| SHA | Branch | What it contains |
|---|---|---|
| `442f48ca` | `awq-aware-hessian` | Codex: torch GPU GPTQ solve (`--gpu N` flag), 1.3e-6 parity vs CPU numpy |
| `c5825848` | `awq-aware-hessian` | Codex: AWQ-aware Hessian transform (`--awq-aware-hessian PATH`). Half the math. |
| `f0bcfabd` | `awq-aware-hessian` | Manual fix: source-weight pre-scale to complete the AWQ-aware GPTQ math. **This unblocked v3.** |
| `85f6d055` | `awq-aware-hessian` | AutoAWQ formula `--awq-formula autoawq` (opt-in, default=paper). Tested + falsified at α=0.5. |
| `f286bade` | `iterative-awq-gptq` | Codex: iterative AWQ+GPTQ refinement rounds with KM damping. 977 LOC + 4 parity tests (3/4 pass; reproducibility-seed test missing but non-blocker). |
| `13f8fb3d` | `iterative-awq-gptq` | Manual fix: iterate subcommand wasn't propagating `skip_unsupported` / `tensor_filter` to its inner GPTQ pass; crashed on the first conv1d. |

## Files modified per branch

### `awq-aware-hessian`
- `crates/hipfire-quantize/src/main.rs` — AutoAWQ formula + `--awq-formula` flag (opt-in)
- `scripts/mq4_masked_calib.py` — `--gpu N`, `--awq-aware-hessian PATH`, source pre-scale, AutoAWQ helper
- `tests/test_gptq_gpu_parity.py` — CPU↔torch.cpu parity gate (1.3e-6 rel L2)

### `iterative-awq-gptq` (additionally)
- `scripts/mq4_masked_calib.py` — `iterate` subcommand, `--candidate-mq4` flag on `collect-stats`, `quantize_iterate_round`, `run_iterative_awq_gptq`, AWQ-sidecar HFQ writer for damped scales
- `tests/test_iterative_awq_gptq.py` — 3 parity tests (identity round, damping=0, synthetic FPI shrinkage)

## Code locations referenced in the investigation

### Rust runtime (read-only references)
- `crates/hipfire-runtime/src/llama.rs:687` — `fused_rmsnorm_rotate_mq_batched_for`: the AWQ-aware dispatch wrapper that proves prefill IS AWQ-wired (sibling-weight invariant: all input-side AWQ scales for q/k/v share imatrix → byte-identical → picking any is mathematically correct)
- `crates/hipfire-arch-qwen35/src/qwen35.rs:855` — `load_awq_scale_for`: the loader pattern hipfire uses; `iterative-awq-gptq`'s `--candidate-mq4` uses a Python equivalent
- `crates/hipfire-runtime/examples/eval_hipfire.rs` — bench harness; takes `--kv-mode {q8,asym2,asym3,asym4,fwht4}` (fwht4 added in `awq-kmap-investigation` branch's commit `2ad2ce43` for an earlier KV-mode sweep)

### Quantizer (where AWQ math lives)
- `crates/hipfire-quantize/src/main.rs:1740` — `kmap_resolve_mode` (K-map tier mapping, not used by v3 since v3 has no `--kmap-dense`)
- `crates/hipfire-quantize/src/main.rs:2525` — `compute_awq_scales` (paper formula, geo-mean normalized)
- `crates/hipfire-quantize/src/main.rs:2525+` — `compute_awq_scales_autoawq` (AutoAWQ formula, behind `--awq-formula autoawq` flag, opt-in only)
- `crates/hipfire-quantize/src/main.rs:2563` — `awq_pre_scale_weights` (the W·diag(s) application at quantize time)
- `crates/hipfire-quantize/src/main.rs:3296` — CLI flag echo line: `"AWQ pre-scaling: ENABLED (alpha=…, formula=…, geo-mean normalized to 1)"`

### Python GPTQ + iterative pipeline
- `scripts/mq4_masked_calib.py:610` — `solve_mq4_gptq_group` (CPU GPTQ core math)
- `scripts/mq4_masked_calib.py:627` — `quantize_mq4_gptq` (CPU dispatch entrypoint)
- `scripts/mq4_masked_calib.py:1304` — `quantize_candidate` (the call site that gets `--awq-aware-hessian`)
- `scripts/mq4_masked_calib.py:1837` — `quantize_iterate_round` (single round driver)
- `scripts/mq4_masked_calib.py:1900+` — `run_iterative_awq_gptq` (KM iteration outer loop)
- `scripts/mq4_masked_calib.py:2030+` — `iterate_awq_gptq` (CLI dispatch)

## Eventual upstream path

When this work is ready to upstream:
1. `awq-aware-hessian` commits should land as a single PR titled "AWQ-aware GPTQ stacking for hipfire-quantize" — pure addition, no behavior change to existing one-shot pipelines, gated behind `--awq-aware-hessian PATH` opt-in
2. `iterative-awq-gptq` commits as a follow-on PR "Iterative AWQ+GPTQ refinement rounds" — needs more polish (the missing reproducibility-seed test, error handling on degenerate Hessians)
3. The `--awq-formula autoawq` flag should NOT be upstreamed at α=0.5 default — but can be left as opt-in with α restricted to >0.85 (where the weight-magnitude term contribution is small enough not to break MQ4 G=256)
