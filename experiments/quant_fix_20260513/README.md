# Qwen3.5-9B MQ4 quant-fix ledger

Goal: drive MQ4 toward llama.cpp Q4-class quality using tensor-selective weight-only calibration.

Hard 20-chunk gate:
- KLD <= 0.09
- PPL <= 9.3404
- throughput >= 328 tok/s

Acceptance rule for iterations: KLD must strictly decrease and PPL must stay <= 9.3404, otherwise reject/revert.

Runtime/kernel changes are out of scope.
