# HoloKV Report

Date: 2026-06-22

## Executive summary

HoloKV should not be implemented in hipfire as a training-free KV compression
path. The local exploration found that HoloKV-style CDMA/Hadamard
superposition looks plausible on synthetic peaked-attention tests, but is
dominated on real qwen3.5 attention tensors. On diffuse attention, demodulation
cross-talk accumulates across many attended tokens: raw `k=2` HoloKV measured
only `out_cos=0.71`, and even a CASK importance split that kept 50% of tokens
exact reached only `out_cos=0.878 @ 192 B/tok`, worse than KVarN-2bit's
`out_cos=0.965 @ 256 B/tok`.

The practical replacement is already the direction hipfire took:
deferred-hierarchical KV compression, where recent tokens stay exact/hot and
old tokens are compacted by importance-weighted CASK-style average merge plus
KVarN cold-tile quantization. The design shipped default-off behind
`HIPFIRE_KV_HIERARCHICAL=1`; HoloKV remains useful only as prior-art evidence
for why superposition is not the right training-free merge primitive here.

## Sources checked

- `~/.claude/history.jsonl` had two user prompts from the
  `/home/sadara/hipfire-akimbo` project asking to explore HoloKV and then test
  `k=2` first.
- `~/.claude/projects/-home-sadara-hipfire-akimbo/memory/project_kv_compression_explore.md`
  records the full project memory for the KV compression branch.
- `~/.claude/projects/-home-sadara-hipfire-akimbo/766d161d-fea0-44ec-b122-cd7a8ddc096f.jsonl`
  contains the main exploration session and the later follow-up write-up.
- `~/.codex` had no older HoloKV project history beyond this current request and
  a prior GitHub repository listing that included `xynexus/HoloKV`.
- Repo docs/code checked: `Quantization/HoloKV/`, `Quantization/kv_explore/`,
  `crates/hipfire-kvquant/src/kv_compact.rs`,
  `docs/plans/2026-06-22-hierarchical-kv-followups.md`, and `NEXT-STEPS.md`.

## What HoloKV proposes

The vendored HoloKV reference describes a KV-cache compression scheme inspired
by CDMA: fold `k` temporal tokens into one physical slot by multiplying each
token with a static orthogonal `+1/-1` phase key, summing, and demodulating later
with the corresponding key. The intended memory reduction is `O(N/k)`, with
variance normalization by `sqrt(k)` and strict even-boundary phase assignment for
RoPE compatibility.

The upstream reference also includes a LoRA/distillation path. That matters:
the attractive README result is not pure training-free compression. It uses a
Qwen-0.5B simulator with injected HoloKV attention and LoRA-style query/value
and output adapters trained by knowledge distillation on a narrow synthetic
retrieval task. The script itself notes that it is a mathematical simulator and
does not provide physical VRAM savings without a fused hardware kernel.

For hipfire, the relevant question was narrower: does the training-free
superposition primitive beat existing KV quantization/merge machinery on real
runtime tensors? The answer was no.

## Local HoloKV test artifacts

`Quantization/HoloKV/` contains the upstream proof-of-concept plus local numpy
test scripts that were not committed to that nested HoloKV git checkout:

- `holokv_math_simulator.py`: PyTorch simulator with Qwen1.5-0.5B, `k=4`,
  CDMA phase keys, variance normalization, and trained LoRA/KD adapters.
- `holokv_k2_recovery_test.py`: training-free `k=2` Hadamard demodulation test
  measuring score recovery and attention KL under Gaussian and outlier-channel
  K distributions, with optional FWHT rotation.
- `holokv_k2_peaked_test.py`: synthetic peaked-attention test measuring
  attention-output cosine and top-token recovery.
- `holokv_kvarn_compose_test.py`: HoloKV superposition composed with KVarN-4bit
  quantization on the superposed slots.
- `holokv_kfold_sweep.py`: fold sweep for `k in {2,4,8}` comparing HoloKV-only
  and HoloKV+KVarN on synthetic peaked attention.

These scripts explain the false start. HoloKV can look good when attention is
highly peaked around a few relevant tokens, because recovering a small set of
token identities is the thing being measured. Real model attention in the
captured qwen3.5 tensors was more diffuse, and the aggregate cross-talk from
many weakly attended tokens damaged the final attention output.

## Real-tensor exploration result

The decisive evidence is in `Quantization/kv_explore/FINDINGS.md`. The harness
used real qwen3.5-0.8b FullAttn layer-3 post-RoPE Q/K/V:

- 256 tokens
- 8 query heads
- 2 KV heads
- head dim 256
- GQA ratio 4
- metric: causal-attention output cosine against f16 KV reference
- sanity: no-compression `k=1` path reached `out_cos=1.0000`

The Pareto result:

| KV B/tok | Compression | Winner | Output cosine |
| ---: | ---: | --- | ---: |
| 512 | 4x | KVarN-4b (+rot) / rank-128 | 0.999 |
| 256 | 8x | low-rank SVD r=64 | 0.991 |
| 192 | 10.7x | CASK m=2 + KVarN-4b | 0.990 |
| 160 | 12.8x | CASK m=4 + KVarN-4b | 0.989 |
| 112 | 18.3x | CASK m=4, core 0.25 | 0.968 |
| 88 | 23.3x | CASK m=8, core 0.25 | 0.966 |

HoloKV was dominated everywhere in this regime. The most important local numbers
recorded were:

- raw HoloKV `k=2`: `out_cos=0.71`
- CASK importance-split plus HoloKV, 50% exact: `out_cos=0.878 @ 192 B/tok`
- KVarN-2bit reference: `out_cos=0.965 @ 256 B/tok`
- CASK average merge plus KVarN-4bit: about `12x @ out_cos=0.99` and
  `23x @ out_cos=0.966`

The core lesson is simple: for cold, unimportant, diffuse tokens, the attention
reader mostly needs their collective contribution, not per-token recoverability.
HoloKV spends its capacity trying to preserve token identities through
superposition. CASK-style weighted averaging directly preserves the aggregate
contribution, so it wins.

## Current hipfire implementation state

The HoloKV rejection fed into the deferred-hierarchical KV path:

- `crates/hipfire-kvquant/src/kv_compact.rs` implements the CPU cold-tier
  producer: keep high-importance core tokens exact, merge the rest `fold_m:1` by
  importance-weighted average, optionally rotate, then KVarN-quantize cold tiles.
- The file comment explicitly records the exploration result:
  "CASK average-merge >> HoloKV superposition on real attention; ~12-23x KV @
  cos 0.97-0.99."
- `docs/plans/2026-06-22-hierarchical-kv-followups.md` records the shipped
  hierarchical KV feature and states that the full design history includes
  "HoloKV dead" as a negative result.

The merged design keeps HoloKV out of the runtime path. It uses:

- hot tier: recent tokens kept exact/raw f32 in a per-layer ring
- cold tier: older tokens compacted into KVarN cold segments
- read path: two-tier attention via cold-slot attention plus online softmax merge
- compaction timing: migration on hot overflow and `idle_compact` between turns
- default: off unless `HIPFIRE_KV_HIERARCHICAL=1`

Quality measurements for the shipped hierarchical path, from the project memory
and follow-up doc, show that the remaining loss is from token merge, not cold-tile
quantization:

| Config | PPL | Note |
| --- | ---: | --- |
| all-KVarN baseline | 30.81 | reference |
| hierarchical fold_m=1 | 26.13 | no merge; beats baseline due to recent f32 hot tier |
| hierarchical fold_m=4 uniform | 40.84 | merge is the whole cost |
| hierarchical fold_m=4 vnorm | 34.84 | better importance signal |
| hierarchical fold_m=4 vnorm + position-local | 34.00 | shipped default behavior |
| hierarchical fold_m=4, 2-bit cold | 34.56 | cold quantization still not the bottleneck |

## Recommendation

Do not build a HoloKV runtime path for hipfire unless the scope changes to a
trained/adapted HoloKV variant with a real hardware kernel and a clear quality
target. For the training-free local-inference path hipfire is pursuing, HoloKV is
a closed negative result.

Recommended next steps:

1. Keep the HoloKV artifacts as prior art and negative-result evidence.
2. Continue hierarchical KV work: segment defragmentation, scale-overhead
   reduction, 1-bit cold probe, and multi-chunk quality/perf A/B.
3. Treat low-rank SVD KV as the parallel long-context alternative, not HoloKV.
4. If HoloKV is revisited, test only a trained/LoRA-denoised variant and compare
   against CASK+KVarN on real model PPL/coherence, not synthetic peaked retrieval.

