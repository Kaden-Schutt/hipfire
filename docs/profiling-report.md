# hipfire Profiling Report

Generated: 20260322_021138
GPU: AMD RX 5700 XT (gfx1010, RDNA1, 8GB GDDR6)

## Qwen3-0.6b — 20 generation tokens

**System during profiling:** GPU 0°C, 0W, SCLK 2075MHz, MCLK 875MHz, GPU util 68%, VRAM 1291/8176MB, CPU 52°C, RAM 5192/64207MB

**Per-token average:** 10026µs (99.7 tok/s)

**llama.cpp ROCm:** 193.2 tok/s generation, 1711.6 tok/s prefill
**hipfire vs llama.cpp:** 0.52x generation

### Non-layer overhead

| Component | Time (µs) | % of total |
|-----------|-----------|-----------|
| Embedding lookup | 43.1 | 0.4% |
| Output RMSNorm | 17.2 | 0.2% |
| Output projection | 372.3 | 3.7% |
| Sampling | 951.9 | 9.5% |
| **All layers** | **8636** | **86.1%** |
| **Total** | **10026** | **100%** |

### Per-operation breakdown (summed across all layers)

| Operation | Total µs | % of layer time | % of token |
|-----------|----------|-----------------|-----------|
| attention | 905 | 10.5% | 9.0% |
| down proj | 637 | 7.4% | 6.4% |
| gate proj | 634 | 7.3% | 6.3% |
| up proj | 632 | 7.3% | 6.3% |
| rope | 609 | 7.0% | 6.1% |
| o proj | 567 | 6.6% | 5.7% |
| q proj | 561 | 6.5% | 5.6% |
| qk norm | 531 | 6.1% | 5.3% |
| attn norm | 483 | 5.6% | 4.8% |
| ffn norm | 476 | 5.5% | 4.7% |
| v proj | 475 | 5.5% | 4.7% |
| kv cache | 475 | 5.5% | 4.7% |
| k proj | 473 | 5.5% | 4.7% |
| attn residual | 388 | 4.5% | 3.9% |
| ffn residual | 384 | 4.4% | 3.8% |
| silu mul | 382 | 4.4% | 3.8% |

### Layer 0 detail

| Op | µs |
|----|---:|
| attn norm | 18.7 |
| q proj | 21.1 |
| k proj | 17.3 |
| v proj | 17.8 |
| qk norm | 19.8 |
| rope | 23.1 |
| kv cache | 18.6 |
| attention | 33.9 |
| o proj | 21.1 |
| attn residual | 15.2 |
| ffn norm | 17.6 |
| gate proj | 23.1 |
| up proj | 23.1 |
| silu mul | 14.7 |
| down proj | 23.0 |
| ffn residual | 14.8 |
| total | 324.2 |

### Layer 27 detail

| Op | µs |
|----|---:|
| attn norm | 17.3 |
| q proj | 20.1 |
| k proj | 16.8 |
| v proj | 17.0 |
| qk norm | 19.3 |
| rope | 21.8 |
| kv cache | 17.0 |
| attention | 32.4 |
| o proj | 20.2 |
| attn residual | 13.9 |
| ffn norm | 16.9 |
| gate proj | 22.9 |
| up proj | 22.8 |
| silu mul | 13.9 |
| down proj | 22.8 |
| ffn residual | 13.7 |
| total | 309.5 |

## Qwen3-0.6b — 128 generation tokens

**System during profiling:** GPU 0°C, 0W, SCLK 2075MHz, MCLK 875MHz, GPU util 69%, VRAM 1291/8176MB, CPU 53°C, RAM 5175/64207MB

**Per-token average:** 10419µs (96.0 tok/s)

**llama.cpp ROCm:** 193.6 tok/s generation, 1708.4 tok/s prefill
**hipfire vs llama.cpp:** 0.50x generation

### Non-layer overhead

| Component | Time (µs) | % of total |
|-----------|-----------|-----------|
| Embedding lookup | 42.7 | 0.4% |
| Output RMSNorm | 17.1 | 0.2% |
| Output projection | 372.6 | 3.6% |
| Sampling | 948.7 | 9.1% |
| **All layers** | **9033** | **86.7%** |
| **Total** | **10419** | **100%** |

### Per-operation breakdown (summed across all layers)

| Operation | Total µs | % of layer time | % of token |
|-----------|----------|-----------------|-----------|
| attention | 1348 | 14.9% | 12.9% |
| down proj | 635 | 7.0% | 6.1% |
| gate proj | 630 | 7.0% | 6.0% |
| up proj | 630 | 7.0% | 6.0% |
| rope | 604 | 6.7% | 5.8% |
| o proj | 563 | 6.2% | 5.4% |
| q proj | 559 | 6.2% | 5.4% |
| qk norm | 529 | 5.9% | 5.1% |
| attn norm | 482 | 5.3% | 4.6% |
| v proj | 474 | 5.2% | 4.5% |
| kv cache | 472 | 5.2% | 4.5% |
| ffn norm | 471 | 5.2% | 4.5% |
| k proj | 470 | 5.2% | 4.5% |
| ffn residual | 382 | 4.2% | 3.7% |
| silu mul | 380 | 4.2% | 3.6% |
| attn residual | 379 | 4.2% | 3.6% |

### Layer 0 detail

| Op | µs |
|----|---:|
| attn norm | 18.5 |
| q proj | 21.1 |
| k proj | 17.3 |
| v proj | 17.5 |
| qk norm | 19.3 |
| rope | 22.6 |
| kv cache | 17.8 |
| attention | 49.1 |
| o proj | 20.6 |
| attn residual | 14.1 |
| ffn norm | 17.2 |
| gate proj | 22.9 |
| up proj | 22.9 |
| silu mul | 14.2 |
| down proj | 23.2 |
| ffn residual | 13.7 |
| total | 333.1 |

### Layer 27 detail

| Op | µs |
|----|---:|
| attn norm | 17.5 |
| q proj | 19.8 |
| k proj | 16.7 |
| v proj | 16.9 |
| qk norm | 18.7 |
| rope | 21.5 |
| kv cache | 16.8 |
| attention | 48.1 |
| o proj | 20.0 |
| attn residual | 13.4 |
| ffn norm | 16.7 |
| gate proj | 22.5 |
| up proj | 22.5 |
| silu mul | 13.5 |
| down proj | 22.6 |
| ffn residual | 13.5 |
| total | 321.6 |

## Qwen3-8b — 20 generation tokens

**System during profiling:** GPU 0°C, 0W, SCLK 2075MHz, MCLK 875MHz, GPU util 90%, VRAM 5422/8176MB, CPU 54°C, RAM 5188/64207MB

**Per-token average:** 23959µs (41.7 tok/s)

**llama.cpp ROCm:** 44.3 tok/s generation, 196.1 tok/s prefill
**hipfire vs llama.cpp:** 0.94x generation

### Non-layer overhead

| Component | Time (µs) | % of total |
|-----------|-----------|-----------|
| Embedding lookup | 46.5 | 0.2% |
| Output RMSNorm | 25.4 | 0.1% |
| Output projection | 1165.1 | 4.9% |
| Sampling | 307.3 | 1.3% |
| **All layers** | **22408** | **93.5%** |
| **Total** | **23959** | **100%** |

### Per-operation breakdown (summed across all layers)

| Operation | Total µs | % of layer time | % of token |
|-----------|----------|-----------------|-----------|
| down proj | 3626 | 16.2% | 15.1% |
| gate proj | 3555 | 15.9% | 14.8% |
| up proj | 3553 | 15.9% | 14.8% |
| o proj | 1539 | 6.9% | 6.4% |
| q proj | 1528 | 6.8% | 6.4% |
| attention | 1380 | 6.2% | 5.8% |
| rope | 975 | 4.4% | 4.1% |
| attn norm | 898 | 4.0% | 3.7% |
| ffn norm | 892 | 4.0% | 3.7% |
| k proj | 786 | 3.5% | 3.3% |
| v proj | 786 | 3.5% | 3.3% |
| qk norm | 702 | 3.1% | 2.9% |
| kv cache | 624 | 2.8% | 2.6% |
| silu mul | 516 | 2.3% | 2.2% |
| ffn residual | 508 | 2.3% | 2.1% |
| attn residual | 506 | 2.3% | 2.1% |

### Layer 0 detail

| Op | µs |
|----|---:|
| attn norm | 26.4 |
| q proj | 43.4 |
| k proj | 22.3 |
| v proj | 22.4 |
| qk norm | 20.2 |
| rope | 28.4 |
| kv cache | 18.5 |
| attention | 39.2 |
| o proj | 42.6 |
| attn residual | 15.0 |
| ffn norm | 25.0 |
| gate proj | 98.4 |
| up proj | 98.9 |
| silu mul | 15.1 |
| down proj | 101.5 |
| ffn residual | 14.1 |
| total | 632.5 |

### Layer 35 detail

| Op | µs |
|----|---:|
| attn norm | 25.0 |
| q proj | 42.7 |
| k proj | 21.9 |
| v proj | 22.1 |
| qk norm | 19.6 |
| rope | 27.2 |
| kv cache | 17.3 |
| attention | 38.6 |
| o proj | 42.5 |
| attn residual | 14.0 |
| ffn norm | 24.5 |
| gate proj | 99.0 |
| up proj | 98.4 |
| silu mul | 14.4 |
| down proj | 101.5 |
| ffn residual | 14.6 |
| total | 624.1 |

## Qwen3-8b — 128 generation tokens

**System during profiling:** GPU 0°C, 0W, SCLK 2075MHz, MCLK 875MHz, GPU util 91%, VRAM 5422/8176MB, CPU 54°C, RAM 5182/64207MB

**Per-token average:** 24246µs (41.2 tok/s)

**llama.cpp ROCm:** 44.3 tok/s generation, 196.1 tok/s prefill
**hipfire vs llama.cpp:** 0.93x generation

### Non-layer overhead

| Component | Time (µs) | % of total |
|-----------|-----------|-----------|
| Embedding lookup | 46.3 | 0.2% |
| Output RMSNorm | 24.9 | 0.1% |
| Output projection | 1157.3 | 4.8% |
| Sampling | 305.2 | 1.3% |
| **All layers** | **22705** | **93.6%** |
| **Total** | **24246** | **100%** |

### Per-operation breakdown (summed across all layers)

| Operation | Total µs | % of layer time | % of token |
|-----------|----------|-----------------|-----------|
| down proj | 3601 | 15.9% | 14.9% |
| gate proj | 3535 | 15.6% | 14.6% |
| up proj | 3531 | 15.6% | 14.6% |
| attention | 1932 | 8.5% | 8.0% |
| o proj | 1516 | 6.7% | 6.3% |
| q proj | 1509 | 6.6% | 6.2% |
| rope | 958 | 4.2% | 4.0% |
| attn norm | 889 | 3.9% | 3.7% |
| ffn norm | 874 | 3.8% | 3.6% |
| k proj | 768 | 3.4% | 3.2% |
| v proj | 763 | 3.4% | 3.1% |
| qk norm | 687 | 3.0% | 2.8% |
| kv cache | 611 | 2.7% | 2.5% |
| ffn residual | 499 | 2.2% | 2.1% |
| attn residual | 499 | 2.2% | 2.1% |
| silu mul | 498 | 2.2% | 2.1% |

### Layer 0 detail

| Op | µs |
|----|---:|
| attn norm | 26.5 |
| q proj | 43.6 |
| k proj | 22.3 |
| v proj | 22.2 |
| qk norm | 19.8 |
| rope | 27.8 |
| kv cache | 18.2 |
| attention | 54.8 |
| o proj | 42.0 |
| attn residual | 14.6 |
| ffn norm | 24.8 |
| gate proj | 97.8 |
| up proj | 98.3 |
| silu mul | 14.6 |
| down proj | 100.6 |
| ffn residual | 13.8 |
| total | 642.7 |

### Layer 35 detail

| Op | µs |
|----|---:|
| attn norm | 24.4 |
| q proj | 41.7 |
| k proj | 21.2 |
| v proj | 21.1 |
| qk norm | 18.9 |
| rope | 26.6 |
| kv cache | 16.9 |
| attention | 53.7 |
| o proj | 42.4 |
| attn residual | 13.8 |
| ffn norm | 24.0 |
| gate proj | 97.9 |
| up proj | 97.4 |
| silu mul | 13.8 |
| down proj | 99.9 |
| ffn residual | 13.9 |
| total | 628.5 |

---
*Generated by bench/compile_results.py*