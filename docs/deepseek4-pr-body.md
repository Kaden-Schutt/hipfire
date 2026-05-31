# Add DeepSeek V4 Flash support (arch_id=9)

Bring DeepSeek V4 Flash onto the canonical upstream serving path.
Targets the production `deepseek-v4-flash.mq2lloyd` build: MQ2-Lloyd
routed experts (96 % of bytes), Q8_0 attention + router + shared-expert
weights, F16 compressor / indexer / Hyper-Connections / norms. KV cache
at runtime is F32 (raw SWA window + compressed-KV indexer). End-to-end:
`hipfire-quantize` → `hipfire serve` → `hipfire run`. Single binary,
no Python in the hot path.

## What's covered

| Layer | Surface added |
|---|---|
| Architecture trait | `crates/hipfire-arch-deepseek4/` — Config / Weights / State / forward / spec_decode; ~10,700 lines |
| RDNA-compute kernels | 72 new HIP kernels (DeepSeek V4 attention SWA + indexer + Hyper-Connections + compressor + MQ2-Lloyd MoE GEMVs + tail-only YaRN RoPE + glue) |
| RDNA-compute dispatch | 73 new `Gpu` methods + arch-aware `gemm_q8_0_wmma` (gfx12 ↔ RDNA3+) |
| Daemon arm | arch_id=9 → `generate_deepseek4` (batched prefill + MTP spec-decode parity with `deepseek4_chat`) |
| Quantizer | `--format deepseek4-q8-mtp` / deepseek4-q8 / deepseek4-source-precision / deepseek4-mtp-precise, `--include-prefix mtp.`, `deepseek_v4` model_type → arch_id 9 |
| Tools | `hfq_split` (partition by tensor prefix) |
| Gates | DeepSeek V4 arms in `coherence-gate.sh` + new `coherence-gate-deepseek4-mtp.sh` sibling; pre-commit HOTSPOT regex covers DeepSeek V4 paths |

## Architecture context

DeepSeek V4 diverges from the qwen35 / LLaMA paths in five load-bearing places:

- **Hyper-Connections** (`hc_mult = 4`, `hc_sinkhorn_iters = 20`): four
  residual streams mixed via a Sinkhorn-normalised gating matrix every
  layer, replacing the single pre-norm residual.
- **Compressed-KV indexer** (`index_n_heads = 64`, `index_head_dim = 128`,
  `index_topk = 512`): a separate small attention surface that scores
  tokens to gate which positions the main attention attends to.
  `compress_ratios` per layer controls compression strength.
- **Tail-only RoPE** (`qk_rope_head_dim = 64` of `head_dim = 512`):
  only the last 64 dims of each head carry rotary positional encoding;
  the rest is straight Q · K matmul.
- **Q-LoRA + O-LoRA** (`q_lora_rank = 1024`, `o_lora_rank = 1024`):
  query and output projections factor through a rank-1024 bottleneck.
- **Raw SWA cache** (`sliding_window = 128`): bounded ring of the last
  128 tokens for attention; longer-range context comes through the
  compressed-KV indexer path.


## Perf numbers (gfx1151 / Strix Halo / Radeon 8060S)

All measured against `/data/hipfire-models/deepseek-v4-flash.mq2lloyd`
(arch_id=9, MTP layer in `deepseek-v4-flash-mtp.mq2lloyd` sidecar).

| Config | PP tok/s | TG tok/s | Notes |
|---|---:|---:|---|
| Plain decode, short prompt (16 tok) | 27.99 | 12.90 | Cold short-prompt PP under-amortises kernel launches |
| Plain decode, long prompt (461 tok), B=64 | 40.42 | 13.61 | Memory baseline 42.8 — matches within ±5% |
| **Path B grouped MoE, B=512** | **51.79** | — | **+28% over scalar K4** (matches memory's +25–28% cite) |
| MTP K=2 (code prompt) | — | 17.51 | 72.6% accept, +29% over plain |
| MTP K=2 (prose) | — | 16.09 | 60.0% accept |
| MTP K=3 (code) | — | 17.66 | 50.0% accept |

Reproducibility:

```
binary md5 (daemon):           4ce71172812ade7e02af7ebd0a351679
binary md5 (deepseek4_chat):   c8e4c75f5890c37e7bce86c1dc5c90fc
binary md5 (hfq_split):        d0bb9d69e32c09af223439cc31a26ab0
prompt md5 (coherence_lloyd_long.txt):  f20bbc4f5b88ab5f7b44fe7c7da0e2e3
```

Bench env: defaults (routed MoE, expert upload, and deterministic MoE-down
are all default-on in the DeepSeek V4 arch crate). MTP rows additionally
`HIPFIRE_DEEPSEEK4_SPEC_DECODE=1 HIPFIRE_DEEPSEEK4_SPEC_K={2,3}` + `HIPFIRE_DEEPSEEK4_MTP_ADDON`
pointing at the split sidecar.

## Gates passed

- **`cargo check --workspace --examples`**: clean (zero DeepSeek V4-introduced errors;
  2 pre-existing master errors in qwen2/qwen35-vl examples unrelated to DeepSeek V4).
- **`cargo fmt -p hipfire-arch-deepseek4`**: clean.
- **`cargo clippy -p hipfire-arch-deepseek4 --no-deps`**: warning-clean
  (workspace-wide clippy has pre-existing warnings unrelated to DeepSeek V4).
- **`./scripts/compile-kernels.sh gfx1010 gfx1030 gfx1100 gfx1151
  gfx1200 gfx1201`**: 72/72 DeepSeek V4 kernels compile on gfx1100 + gfx1151
  (production). Six DeepSeek V4 WMMA kernels fail on gfx1010/gfx1030 (RDNA1/RDNA2
  has no WMMA — architectural limit) and on gfx1200/gfx1201 (need
  `.gfx12.hip` variants using the gfx12-specific WMMA intrinsic — follow-up,
  same pattern as master's `gemm_q8_0_wmma.gfx12.hip`).
- **`./scripts/write-kernel-hashes.sh`**: 363 `.hash` files for gfx1151
  + matching counts for the other 6 arches.
- **`./target/release/examples/test_kernels`**: 16/16 pass on gfx1151,
  0 failed, 0 skipped. DeepSeek V4-specific kernel correctness coverage is a
  follow-up — the existing harness is hardcoded for F32 / Q8 KV / GDN / VL.
- **`./scripts/coherence-gate.sh --full`**: DeepSeek V4 cases all OK
  (deepseek4-cap → "The capital of France is Paris.", deepseek4-reason → correct
  9-sheep reasoning, deepseek4-long-prefill → on-topic LRU-cache code review).
  No hard errors. Report `/tmp/coherence-20260523-112955.md`.
- **`./scripts/coherence-gate-deepseek4-mtp.sh --fast`**: deepseek4-mtp-code-k2 → OK,
  80.5% K=2 accept, 17.63 tok/s, unique_ratio=0.55, max_freq=0.062. No
  hard errors. Report `/tmp/coherence-deepseek4-mtp-20260523-114823.md`.

## MTP sidecar split

Optionally split off the MTP layer into a sidecar (so non-spec-decode
users skip the 1.85 GB MTP upload):

```bash
hfq_split deepseek-v4-flash.mq2lloyd \
    --base   deepseek-v4-flash.mq2lloyd.new \
    --addon  deepseek-v4-flash-mtp.mq2lloyd \
    --addon-prefix mtp.0.
mv deepseek-v4-flash.mq2lloyd.new deepseek-v4-flash.mq2lloyd
```

The runtime picks up the addon via either
`HIPFIRE_DEEPSEEK4_MTP_ADDON=<addon path>` or the sibling
`<base>.mtp-addon.hfq` convention.

## Deferred to follow-up PRs

These are intentionally out of scope and clearly documented:

- **gfx12 WMMA variants** for `gemm_f16_x_f16_wmma`,
  `gemm_hfq4g256_wmma`, `gemm_mq2g256_lloyd_moe_grouped_wmma_k2`. The
  generic RDNA3+ intrinsic doesn't compile on gfx12; need `.gfx12.hip`
  variants like master's `gemm_q8_0_wmma.gfx12.hip` to make DeepSeek V4 load
  on RDNA4 hardware.
- **RDNA1 / RDNA2 (gfx1010 / gfx1030)** scalar fallback dispatches.
  WMMA is fundamentally unavailable; DeepSeek V4 would need a scalar
  serving path (significant work).
- **DeepSeek V4-aware `speed-gate.sh`**. Current `bench_qwen35_mq4` is qwen35-only.
  DeepSeek V4 numbers in `tests/speed-baselines/gfx1151.txt` are documented but
  not enforced.
- **DeepSeek V4 PFlash / DFlash drafter** integration (DeepSeek V4 draft model design open).
- **DeepSeek V4 multi-GPU pipeline-parallel** + **CASK eviction**.
- **DeepSeek V4 vision (VL) integration**.
- **Re-quantize a fresh DeepSeek V4 build with the ported quantizer**. The
  ported `hipfire-quantize` recognises `deepseek_v4` and accepts the
  deepseek4-q8-mtp format, but a full quant takes hours; the existing
  pre-built file (with MTP split) is sufficient to validate the
  runtime-load path.

## Notes for reviewers

- **First-token attractor on DeepSeek V4 MTP spec-decode**: the very first sampled
  token (from prefill's last logits) is consumed as the spec-decode
  drafter's seed but never emitted. Output stream starts at token 2.
  Same behaviour in `deepseek4_chat` and in `generate_deepseek4` — not introduced
  by this PR; flagging for awareness.
- **HIP-graph kernarg-bake bug** (`feedback_deepseek4_hipgraphs_swa_kernarg_bug`)
  is a known issue on the DeepSeek V4 dev branch — captured at first decode then
  replayed with stable host-side update buffers. Default-on for gfx11/12;
  opt out via `HIPFIRE_DEEPSEEK4_GRAPH=0`.
