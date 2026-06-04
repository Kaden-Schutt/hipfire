# Smoke Scripts

These scripts validate eval-harness wiring and artifact production. They are
not benchmark evidence and should not be used for quality or performance
claims.

## Scripts

- `eval-harness-nogpu-smoke.sh` runs the CI-safe mock executor path and checks
  manifest, summary, provenance, comparison, admission, prompt ledger, and
  evidence ingestion artifacts.
- `eval-harness-gpu-smoke.sh` runs examples-backed smoke rows on a local small
  model. It validates real model execution, artifacts, and optional paired
  DFlash wiring.
- `eval-harness-model-eval-smoke.sh` runs a candidate-vs-baseline integration
  smoke with generated quality/evidence fixtures plus examples-backed speed
  rows.

`scripts/no-gpu-ci.sh` invokes the no-GPU smoke and syntax-checks the GPU smoke
scripts.
