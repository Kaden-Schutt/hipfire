# Gemma 4 E-Series gfx1201 Validation

## Environment

- Host GPU: AMD Radeon AI PRO R9700, `gfx1201`, 32 GiB VRAM
- ROCm/HIP: `7.14.60850`
- Build: `cargo build --release --features deltanet --example daemon -p hipfire-runtime`
- Build result: success in 54.40 seconds
- KV cache: Q8
- Sampling: greedy, temperature 0
- Runtime graph and EAGLE: disabled

## Model Artifacts

| Model | SHA256 |
|---|---|
| Gemma 4 E2B Q8 | `7dc98107c0e802036c30b1ee906f74a4214cbba75f81c7983e5bf184ee357a2d` |
| Gemma 4 E4B Q8 | `0224a0e3240030c0b5d14dcbf4f3ace1c96bea3ae166e8372068c843f589e80a` |

## GSM8K Smoke

The first 30 fixed GSM8K CoT prompts were run on each model. Both models completed every request without a runtime error.

| Metric | E2B gfx1201 | E4B gfx1201 |
|---|---:|---:|
| Completed | 30/30 | 30/30 |
| Runtime errors | 0 | 0 |
| Flexible numeric accuracy | 80.0% | 90.0% |
| Median prefill | 513.685 tok/s | 346.525 tok/s |
| Median decode | 96.575 tok/s | 64.435 tok/s |
| Median TTFT | 1,490.46 ms | 2,208.07 ms |
| Median wall time | 3.738 s | 4.005 s |

## Cross-Architecture Comparison

The same 30 prompts and model artifacts were compared against the prior W7900/gfx1100 run.

| Model | Metric | W7900/gfx1100 | R9700/gfx1201 | Delta |
|---|---|---:|---:|---:|
| E2B | Prefill | 414.685 tok/s | 513.685 tok/s | +23.87% |
| E2B | Decode | 102.490 tok/s | 96.575 tok/s | -5.77% |
| E2B | TTFT | 1,844.80 ms | 1,490.46 ms | -19.21% |
| E2B | Wall time | 3.968 s | 3.738 s | -5.80% |
| E4B | Prefill | 281.685 tok/s | 346.525 tok/s | +23.02% |
| E4B | Decode | 68.225 tok/s | 64.435 tok/s | -5.55% |
| E4B | TTFT | 2,713.77 ms | 2,208.07 ms | -18.63% |
| E4B | Wall time | 4.415 s | 4.005 s | -9.28% |

All 30 normalized predictions and correctness decisions matched between gfx1100 and gfx1201 for both models. Generated text was byte-identical for 23/30 E2B prompts and 24/30 E4B prompts. The remaining responses diverged during greedy generation but retained the same normalized final answer and correctness result. This supports functional parity for the tested set, not full bitwise parity across GPU architectures.

## Batched Prefill Promotion

The original gfx1201 validation used the sequential prefill routes. The E-series batched embedding, PLE branch, and fused PLE activation paths were initially restricted to gfx1100 even though their underlying kernels already had compatible gfx1201 implementations. In particular, the Q8 batched dispatcher selects the RDNA4-specific `gemm_q8_0_wmma_gfx12` source, while the embedding and PLE activation kernels are architecture-generic.

Each route was first enabled independently on gfx1201. Measurements used one release binary, prefill batch 64, ten fixed GSM8K 8-shot prompts, three repeats, and a 16-token output cap. Every non-target route was explicitly disabled.

| Model | Route | Prefill median | Delta vs off | TTFT median | Decode median |
|---|---|---:|---:|---:|---:|
| E2B Q8 | all off | 1,367.745 tok/s | - | 572.091 ms | 96.97 tok/s |
| E2B Q8 | batched embedding | 1,379.430 tok/s | +0.85% | 567.458 ms | 96.97 tok/s |
| E2B Q8 | batched PLE branch | 2,154.735 tok/s | +57.54% | 367.399 ms | 95.81 tok/s |
| E2B Q8 | fused PLE activation | 1,564.115 tok/s | +14.36% | 501.499 ms | 96.97 tok/s |
| E4B Q8 | all off | 989.360 tok/s | - | 789.586 ms | 65.04 tok/s |
| E4B Q8 | batched embedding | 989.500 tok/s | +0.01% | 783.998 ms | 64.52 tok/s |
| E4B Q8 | batched PLE branch | 1,445.315 tok/s | +46.09% | 540.555 ms | 64.26 tok/s |
| E4B Q8 | fused PLE activation | 1,099.025 tok/s | +11.09% | 709.006 ms | 64.26 tok/s |

The three original candidate routes compose successfully in the short-output timing gate:

| Model | Policy | Prefill median | Speedup | TTFT median | TTFT reduction |
|---|---|---:|---:|---:|---:|
| E2B Q8 | all off | 1,365.125 tok/s | 1.00x | 574.190 ms | - |
| E2B Q8 | auto | 2,890.540 tok/s | 2.12x | 275.578 ms | 52.01% |
| E4B Q8 | all off | 989.360 tok/s | 1.00x | 790.524 ms | - |
| E4B Q8 | auto | 1,802.975 tok/s | 1.82x | 437.593 ms | 44.64% |

Direct sequential-versus-batched validation passed at B=1/2/4/64 for both models under the combined policy. Every last-token argmax and next-token KV-check argmax matched. The minimum observed logits cosine was 0.9999465 for E2B and 0.9999427 for E4B.

A separate 30-question GSM8K A/B used a 512-token output cap. Both policies completed all requests without a runtime error. E2B accuracy moved from 43.3% to 46.7%; E4B moved from 26.7% to 33.3%. These truncated runs are a non-regression gate rather than an accuracy claim. Median prefill improved from 1,340.835 to 2,812.645 tok/s on E2B and from 968.555 to 1,767.860 tok/s on E4B; decode medians were unchanged within 0.1%. In the shorter 30-output gate, E4B was byte-identical in 30/30 cases and E2B in 27/30; the E2B differences were stable greedy wording divergences from the non-bit-identical WMMA route.

The original auto-policy rerun fixed `HIPFIRE_Q8_BATCHED_LEGACY=0`, verified `gfx1201` through the daemon diagnostic protocol, and recorded the daemon, model, and GSM8K dataset hashes. Decode medians moved from 96.97 to 95.24 tok/s on E2B and from 65.04 to 64.26 tok/s on E4B; the candidate routes are prefill optimizations rather than decode changes. The later full-logit evidence and current safe policy are documented in `GEMMA4_ESERIES_LOGIT_STABILITY.md`.

The gfx1201 production policy uses prefill batch 64 and enables batched embedding and fused PLE activation by default. Batched PLE branch projections remain an explicit experiment: the faster F16-activation WMMA route changed low-margin greedy choices in the later LongBench hard30 gate. gfx1200 and other unvalidated architectures retain sequential prefill and disabled feature defaults.

The final safe-policy hard30 rerun explicitly disabled PLE branch batching. E2B prefill improved from 494.255 to 519.645 tok/s (+5.14%) and E4B improved from 428.385 to 451.550 tok/s (+5.41%); TTFT medians fell by 4.89% and 5.13%. All 60 paired predictions were byte-identical, with unchanged accuracy and zero correctness regressions. Full-logit and cross-architecture evidence is in `GEMMA4_ESERIES_LOGIT_STABILITY.md`.

Reproduce the isolated route matrix with `scripts/bench-gemma4-gfx12-prefill-routes-ab.sh` and the longer-output gate with `scripts/bench-gemma4-gfx12-prefill-quality-ab.sh`.
