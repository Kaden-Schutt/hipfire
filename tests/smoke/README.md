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
- `diffusion-sdapi-smoke.sh` starts `hipfire serve` and validates the
  stable-diffusion-webui-compatible HTTP API against a local runnable diffusion
  `.hfq`: `txt2img`, `img2img`, masked `img2img`, PNG dimensions, `png-info`,
  options, samplers, LoRA metadata, and progress. Set
  `HIPFIRE_DIFFUSION_SMOKE_MODEL` when the default
  `/tmp/hipfire-tiny-sd-diffusion.hfq` is not present. Set
  `HIPFIRE_DIFFUSION_SMOKE_BATCH_SIZE` and `HIPFIRE_DIFFUSION_SMOKE_N_ITER` to
  validate WebUI batch and iteration semantics against batch-capable artifacts.
  Set `HIPFIRE_DIFFUSION_SMOKE_ROCM_DEVICE_ID` to validate the hybrid ROCm
  runtime path; the smoke will use a release cargo run with `--features rocm`
  unless `HIPFIRE_DIFFUSION_SDAPI_SMOKE_CARGO_FEATURES` or
  `HIPFIRE_DIFFUSION_SDAPI_SMOKE_CMD` overrides it.
- `diffusion-tiny-sd-hfq-admission.sh` imports a Tiny-SD HFQ from the local
  Hugging Face cache when needed, then runs the native CLI diffusion admission
  smoke. Set `HIPFIRE_DIFFUSION_ADMISSION_ROCM_DEVICE_ID` to validate the
  hybrid ROCm runtime path; the smoke will use `--features rocm` unless
  `HIPFIRE_DIFFUSION_ADMISSION_CARGO_FEATURES` overrides it. ROCm liveness
  checks can narrow the run with `HIPFIRE_DIFFUSION_ADMISSION_BATCH_SIZE=1` and
  `HIPFIRE_DIFFUSION_ADMISSION_TXT2IMG_ONLY=1`; CPU admission should keep the
  default full txt2img/img2img/masked-img2img coverage.

`tests/no-gpu-ci.sh` invokes the no-GPU smoke and syntax-checks the optional
model/server smoke scripts.
