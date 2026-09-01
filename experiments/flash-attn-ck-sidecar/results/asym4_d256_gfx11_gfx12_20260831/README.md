# Asym4 D256 CK validation

This record validates the optional Asym4-Givens/FWHT K plus Q8 V loader and
CK attention route. It does not claim an end-to-end speedup because the native
Asym4 prefill baseline on this revision fails before a valid A/B can complete.

## Raw ABI correctness

The same sidecar source was built separately for `gfx1100` and `gfx1201` and
run with `smoke_raw_abi`.

| GPU target | Cell | Max absolute error | Mean absolute error |
| --- | --- | ---: | ---: |
| `gfx1100` | Asym4-Givens D256 GQA causal | `5.862117e-05` | `1.000180e-05` |
| `gfx1100` | Asym4-FWHT D256 GQA causal | `6.847084e-05` | `1.012444e-05` |
| `gfx1201` | Asym4-Givens D256 GQA causal | `5.862117e-05` | `1.000134e-05` |
| `gfx1201` | Asym4-FWHT D256 GQA causal | `6.847084e-05` | `1.012253e-05` |

Both targets reported a 65,536-byte workspace for the smoke shape. Existing
dense, Q8, and Asym3 cells also passed in the same runs.

## Production-path validation

Configuration: Qwen3.6-27B MQ4, Asym4 KV, caller-owned 512 MiB transient
workspace, CK sidecar enabled. Speculative decoding, DSpark, MTP, and n-gram
drafting were disabled.

| Target | Prompt | Runs | Prefill median | Long decode | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| W7900 `gfx1100` | 8192 | 5 | `778.8 tok/s` | `34.3 tok/s` over 4096 tokens | next token `248046` |
| R9700 `gfx1201` | 8192 | 5 | `868.8 tok/s` | `32.3 tok/s` over 4096 tokens | next token `248046` |

The decode phase remains on the native path; these measurements establish that
the CK prefill handoff leaves a valid cache for long autoregressive decode.

## LongBench hard30

Both exact-architecture sidecars ran the same LongBench-v2 hard30 sample with
20K--30K-token inputs, `max_seq=65536`, and `max_tokens=16384`. All cases
terminated naturally before the output limit.

| Target | Completed | Errors | Scored | Accuracy | Prefill median | Decode median | Maximum output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W7900 `gfx1100` | 30/30 | 0 | 30/30 | 13/30 (`43.33%`) | `730.6 tok/s` | `29.9 tok/s` | 4061 tokens |
| R9700 `gfx1201` | 30/30 | 0 | 30/30 | 13/30 (`43.33%`) | `774.45 tok/s` | `28.7 tok/s` | 2971 tokens |

The per-case correctness decisions agree on all 30 cases. Full generated text
is byte-identical on 14/30 cases; the remaining generations differ across
architectures without changing any of the 30 task-level correctness outcomes.

## Native baseline blocker

With the CK route absent or forced off, the same production binary fails at
PP8192 on both gfx1100 and gfx1201 with `hipError 700` reported by the next H2D
copy. The gfx1201 runtime identifies the faulting kernel as
`attention_flash_asym4_wmma_tile_batched_gfx12`. A
binary built without the `flash-attn-ck` feature reproduces the PP2048 failure.
Launching that binary from `/tmp`, where the worktree's precompiled kernel
directory cannot be discovered, also reproduces the failure; stale kernel
blobs are therefore not the cause.
The CK-enabled PP2048 run completes and logs
`selected_asym4_givens_d256`, so the failure is outside the optional loader and
prevents a valid native-versus-CK performance comparison on this revision.
