# GFX1201 HFQ4 weight buffer-load policy

**Date:** 2026-07-11

**Host:** `hiptrx`, Radeon AI PRO R9700 (`gfx1201`)

**Model:** `qwen3.6-35b-a3b.mq4r`

**Clock policy:** automatic; no clock or performance-level override

## Result

Three zero-scratch HFQ4 decode kernels now use device-scope raw-buffer loads
for weights on gfx1201:

- `fused_qkvza_hfq4g256`;
- `fused_qkv_hfq4g256`;
- `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded`.

The winning policy is ordinary temporal (`TH_RT`, `SCOPE_DEV`, aux `16`). It
improved retained-PM4 Qwen A3B decode by about 1% at both tested contexts:

| Regime | Global-load control | Buffer RT | Change |
| --- | ---: | ---: | ---: |
| Q8, context 128, 128 positions | 183.483 tok/s | 185.399 tok/s | **+1.044%** |
| FWHT3, context 8192, process mean | 173.154 tok/s | 174.886 tok/s | **+0.999%** |

The 8K process medians were `173.237, 173.070` for global loads and
`174.824, 174.947` for buffer RT. Each process used three warmups followed by
ten measured rows of 100 positions. Policy arms used separate JIT caches and
fresh daemon processes; the winning pair was repeated in reverse order.

## Temporal-policy screen

The explicit GFX12 temporal hints did not beat the buffer-addressing control.
All rows below are fresh-process retained-PM4 medians at context 8192:

| Weight policy | Aux | Median | Versus buffer RT |
| --- | ---: | ---: | ---: |
| global-load control | — | 173.237 tok/s | -0.908% |
| buffer `TH_RT`, device scope | 16 | **174.824 tok/s** | — |
| buffer `TH_HT`, device scope | 18 | 174.115 tok/s | -0.406% |
| buffer `TH_NT_RT`, device scope | 20 | 174.206 tok/s | -0.354% |
| buffer `TH_NT_HT`, device scope | 22 | 174.182 tok/s | -0.367% |

The result is therefore a buffer-addressing win, not evidence for a
non-default temporal hint. The default remains `TH_RT`.

## Scope and register-pressure guard

This is deliberately not blanket HFQ4 conversion. The tuned MoE gate/up
kernel is already at 96 VGPR. Two row-local SRDs made LLVM allocate 12 bytes of
private scratch; an expert-wide single SRD made it allocate 16 bytes. Redline
correctly rejected both PM4 tapes because retained dispatch does not support
scratch-bearing kernels. Gate/up remains on its existing global-load path.

The accepted kernels remain zero-scratch. Their gfx1201 register metadata
changed as follows:

| Kernel | Global VGPR | Buffer VGPR |
| --- | ---: | ---: |
| fused QKVZA | 81 | 80 |
| fused QKV | 72 | 80 |
| MoE down expanded | 92 | 92 |

This metadata check is part of the eligibility decision; adding another
weight kernel requires proving it remains zero-scratch before model timing.

## Correctness and isolation

- Final no-environment default: 15 consecutive positions exact against both
  direct HIP and exact captured-kernarg execution (833 launches / 27 kernels).
- Tested aux-16 arm: same 15-position exact gate.
- The final no-environment gfx1201 `.text` is byte-identical to the measured
  explicit aux-16 arm for all three converted kernels.
- The three gfx1100 `.text` sections are byte-identical to pre-change HEAD.
- ISA inspection confirms weight `buffer_load` instructions carry
  `scope:SCOPE_DEV`; explicit variants emit `TH_LOAD_HT`, `TH_LOAD_NT_RT`, and
  `TH_LOAD_NT_HT`. Activation/x loads remain ordinary global loads.

## Operator surface

GFX1201 defaults to `rt`. The policy can be selected before `Gpu::init()` with:

```text
HIPFIRE_GFX12_WEIGHT_LOAD_POLICY=global|rt|ht|nt-rt|nt-ht
```

`global` is the rollback and measurement control. The selection is folded into
the existing hipcc cache hash, so policies cannot reuse one another's code
objects. Other GPU architectures ignore this policy.

## Artifacts

Raw reports are retained on `hiptrx` under:

```text
/home/kaden/.redline-work/hipfire-cache-policy/.redline-work/cache-policy/
```

Notable files are `product-8k-{global,rt_dev,ht_dev,nt_rt_dev,nt_ht_dev}.json`,
`product-8k-{rt_dev,global}-reverse.json`,
`product-tg128-{global,rt_dev}.json`, and `final-default-shadow15.json`.
