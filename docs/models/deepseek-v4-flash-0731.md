---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash
tags:
  - hipfire
  - deepseek
  - moe
  - quantized
---

# DeepSeek-V4-Flash-0731 — hipfire MQ2-Lloyd

DeepSeek V4 Flash (0731), quantised to MQ2-Lloyd for [hipfire](https://github.com/warpfront/hipfire).
43 layers, 256 routed experts + 1 shared, top-6 per token, `arch_id=9`.

| file | size | what |
|---|---:|---|
| `deepseek-v4-flash-0731.mq2lloyd` | 86.2 GB | trunk |
| `deepseek-v4-flash-0731-dspark.mq2lloyd` | 6.0 GB | DSpark 3-stage draft sidecar |
| `deepseek-v4-flash-0731-adapter-r128.bin` | 44.6 MB | expert-prediction adapter (experimental, **off by default**) |

The 0731 checkpoint ships the DSpark draft chain in place of the classic nextn
MTP head, so it pairs with a `-dspark` sidecar, **not** a `-mtp` one.
`temperature=1.0` mirrors the base checkpoint's serving guidance.

## Running it when the experts do not fit

The trunk needs ~86 GB. Routed experts are 72.2 GiB of that, and can be paged
from disk instead of held resident:

```bash
export HIPFIRE_DEEPSEEK4_EXPERT_CACHE_GB=auto   # or an explicit GiB budget
```

`auto` takes whatever fits after non-expert weights, KV and headroom, clamped
against MemAvailable. If everything fits it disables paging rather than paging
pointlessly.

### Cache budget is the main performance dial

Measured on Radeon 8060S / gfx1151 (Strix Halo), ds4 MQ2 decode:

| budget | slots/blob | hit rate | tok/s | ms/token | expert I/O share |
|-------:|-----------:|---------:|------:|---------:|-----------------:|
|  4 GiB |  14 | 40.5% | 7.02 | 142 | — |
|  8 GiB |  28 | 59.4% | 7.06 | 141.7 | 41.9% |
| 16 GiB |  56 | 74.4% | — | — | — |
| 32 GiB | 112 | 86.5% | 9.09 | 110.0 | 26.5% |
| 64 GiB | 224 | 91.0% | — | — | — |
| resident | 256 | 91.3% | — | — | — |

**8 GiB → 32 GiB is +28.8% throughput.** Hit rate saturates at 91.3% (the rest
is compulsory cold-miss), so past ~32 GiB the curve flattens hard — 64 GiB buys
only another 4.5pp. Pick 32 GiB if you have it.

Where the time goes at 8 GiB, from ablation (routed MoE off vs on):

| component | ms/token | share |
|---|---:|---:|
| expert paging I/O | 59.4 | **42%** |
| routed expert GEMV compute | 14.6 | 10% |
| attention, shared experts, norms, lm_head | 67.7 | 48% |

The routed-MoE path costs 74 ms/token and **80% of that is I/O, not compute** —
this is an I/O-bound workload wearing a compute-bound costume.

## Expert-prediction adapter (experimental — OFF by default)

`deepseek-v4-flash-0731-adapter-r128.bin` predicts **layer L+1's expert
selection from layer L's hidden state**, so a paged runtime can start fetching
a layer early rather than stalling.

> **It currently makes decode slower. Published for reproduction, not as a win.**
> Every configuration measured came out below 1.0x. Off unless you explicitly
> set `HIPFIRE_DEEPSEEK4_EXPERT_ADAPTER`.

### How it works

```
z_{L+1}  ~=  B · (A · h_L)  +  gate_bias_{L+1}       A: [128,4096]  B: [256,128]
```

Rank, take top-M as prefetch candidates. The **frozen native router still makes
the real selection** — the adapter only decides what to stage, so a wrong
prediction costs a wasted fetch, never a wrong token. Output is byte-identical
with it on or off (verified, greedy, 9/9 arms).

Fitted by closed-form ridge on 113,433 wikitext-prefill positions, 80/20 split,
SVD-truncated to rank 128. 22.3M params, ~22M MACs/token (<1% of per-token
compute).

### Accuracy

ExpertRecall@M — of the 6 experts actually chosen, how many appear in the top-M:

| top_M | recall | covers of 6 | wasted | fetched late | total fetches vs baseline |
|------:|-------:|------------:|-------:|-------------:|--------------------------:|
| 4  | 59.6% | 3.58 | 0.42 | 2.42 | 1.07x |
| 6  | 75.9% | 4.55 | 1.45 | 1.45 | 1.24x |
| 8  | 83.2% | 4.99 | 3.01 | 1.01 | 1.50x |
| 12 | 89.2% | 5.35 | 6.65 | 0.65 | 2.11x |

Training-free alternative (running the real `gate_{L+1}` on `h_L`): **27.6%**.
ds4's mHC replaces the residual stream with 4 Sinkhorn-mixed streams, so there
is no slowly-evolving residual to read through and a trained map is needed.

Rank matters more than in published work — SpecPrefetch reports ~84% at r=32 on
64-expert DeepSeek-VL2, but ds4 is 256-choose-6 and reaches only 54.9% at r=32:

| rank | params/layer | total | recall@6 |
|-----:|-------------:|------:|---------:|
| 32   | 139,264   |  5.6M | 54.9% |
| 64   | 278,528   | 11.1M | 66.9% |
| 128  | 557,056   | 22.3M | 75.6% |
| full | 1,048,576 | 41.9M | 78.7% |

### Why it is not a win

Best result at each cache size, 2-3 reps, interleaved:

| cache | best top_M | speedup |
|------:|-----------:|--------:|
|  4 GiB | 3 | 0.89x |
|  8 GiB | 4 | 0.93x |
| 32 GiB | 6 | 0.86x |

No cache size makes it positive, and tightening to 4 GiB made it *worse* — the
"constrained caches are where prefetch pays" hypothesis was tested and refuted.

The cause, measured directly on the same workload:

```
off:  70,736 accesses,  25,218 misses,   83.1 GiB read
on : 147,200 accesses,  52,900 misses,  174.4 GiB read   (2.10x)
```

Speculation more than doubles bytes read. Hit rate stays flat at ~64% while
bytes double — churn, not caching: staged experts evict entries the real
dispatch then needs. No amount of overlap pays for 2.1x the I/O.

Prefetch overhead is a fixed per-layer cost while its benefit scales with how
much I/O there is to hide, so it **competes with the cache budget rather than
composing with it**. Raising the budget is strictly the better lever.

### Usage

```bash
export HIPFIRE_DEEPSEEK4_EXPERT_CACHE_GB=8
export HIPFIRE_DEEPSEEK4_EXPERT_ADAPTER=/path/to/deepseek-v4-flash-0731-adapter-r128.bin
export HIPFIRE_DEEPSEEK4_PREFETCH_TOPM=4     # default k+2=8; 4 measured least-bad
```

Load errors are fatal, not silent — running without the adapter you asked for
would quietly benchmark the wrong configuration.

### Where it might still pay (untested)

- **Discrete GPUs** (R9700 gfx1201, 7900 XTX gfx1100). On unified memory the H2D
  half is a plain memcpy with nothing to hide; on a discrete card it is a real
  PCIe DMA with an engine to hide it behind, so the trade may invert.
- **Removing the per-dispatch host round-trip.** The pager is host-driven, so
  every dispatch copies routing back to the host — including the ~64% that are
  pure hits needing no host action.

### Format

```
magic "HFADPT\0\0"  8 B
u32 version=1, u32 n_entries, u32 d_model=4096, u32 n_exp=256, u32 rank=128
per entry: u32 src_layer (predicts src_layer+1), f16 A[rank,d_model], f16 B[n_exp,rank]
```

Trained and evaluated on wikitext only — cross-domain generalisation unmeasured.
