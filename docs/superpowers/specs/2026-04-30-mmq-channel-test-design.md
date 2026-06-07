# MMQ Channel-Test: Bit-Comparison Analysis of GEMM Call Sites

**Date:** 2026-04-30
**Ref:** Kaden-Schutt/hipfire#87 (auto-MMQ regression on tool-call output, gfx1151)
**Deliverable:** `crates/engine/examples/channel_test_mmq.rs`

## Problem

Auto-MMQ (i8 WMMA + Q8_1 activation quantization) at batch sizes >= 128 on
gfx1151 (Strix Halo) corrupts tool-call output. ChatML special tokens
(`<|im_start|>`) leak into visible output instead of being structured inside
`<tool_call>` blocks. The feature was reverted at `d1506d0` (original PR #84
delivered +63-96% prefill speedup: 545 -> 890 tok/s at pp128).

The root cause is believed to be Q8_1 quantization precision loss in narrow
probability cells around special-token IDs, but it's unknown which of the four
GEMM call sites is responsible, which output channels are affected, and whether
the problem is concentrated in specific layers.

## Goal

Build a diagnostic binary that answers three questions in order:

1. **Which GEMM call site(s)** produce unacceptable error under MMQ vs f16 WMMA?
2. **Which output channels (rows)** within the guilty site(s) are worst?
3. **Which transformer layers** concentrate the error?

## Approach: Zero dispatch.rs changes

All necessary APIs are already public:

- **Weight tensors:** `Qwen35Weights.layers[i]` exposes per-layer weight
  `GpuTensor` handles for each site (DeltaNet: `wqkv`/`wz`/`w_alpha`/`w_beta`,
  `w_gate`/`w_up`, `wo`; FullAttn: `wq`/`wk`/`wv`, `w_gate`/`w_up`, `wo`).
- **GEMM methods:** Both MMQ (`gemm_hfq4g256_mmq_set_prequant`,
  `gemm_hfq4g256_residual_mmq`) and f16 WMMA (`gemm_*_wmma`) are public on `Gpu`.
- **Tensor I/O:** `upload_f32`, `download_f32`, `upload_raw` are public.
- **capture_mode:** Public bool field on `Gpu`; set to `true` to skip the
  rocBLAS fast path and force the WMMA/MMQ dispatch branches.

The binary loads the model, runs a real prefill pass (f16 WMMA, `HIPFIRE_MMQ=0`)
to obtain realistic per-layer activations, then replays each layer's activations
through both the f16 WMMA and MMQ paths and diffs the outputs on CPU.

## Interface

```
cargo run --release --example channel_test_mmq -- \
  --model models/qwen3.6-27b.mq4 \
  --prompt benchmarks/prompts/tool_call_read_file.txt \
  --system benchmarks/prompts/tool_call_system.txt \
  --stage {site-scan|channel-map|layer-sweep} \
  [--layer N]          # channel-map: which layer (default: worst from site-scan)
  [--site NAME]        # channel-map/layer-sweep: filter to one site
  [--threshold 0.01]   # abs error threshold for flagging bad elements
  [--json FILE]        # optional: write structured report to FILE
```

## Stage 1: site-scan

**Question:** Which of the four GEMM call sites produces the most error?

**Method:**

For each transformer layer L:
1. Obtain the activation tensor `x_L` (the input to that layer's GEMM sites).
   This is done by running a single forward prefill pass with f16 WMMA and
   intercepting `x` at each layer. The interception uses the existing
   `forward_prefill_batch` path with a small wrapper that saves intermediate
   `x` tensors to host memory via `download_f32` after RMSNorm (the input to
   the GEMM sites).
2. Re-upload `x_L` to GPU.
3. For each site (qkvza/qkv, gate_up, residual):
   a. Allocate two output tensors `y_wmma` and `y_mmq` (zeroed).
   b. Call the f16 WMMA method with the layer's weight tensor + `x_L` -> `y_wmma`.
   c. Call the MMQ method with the same weight tensor + `x_L` -> `y_mmq`.
   d. `download_f32` both, compute error stats on CPU.

**Activation capture detail:**

Rather than modifying the forward pass, we build `x_L` from scratch per layer:
- Load model, allocate scratch.
- Run `forward_prefill_batch` once with `HIPFIRE_MMQ=0` to completion (producing
  correct logits as a sanity check).
- Then, for the comparison pass, run a **manual layer-by-layer loop** that
  mirrors the forward pass structure:
  1. Embed tokens -> `x`
  2. For each layer: RMSNorm -> `x_norm`. Save `x_norm` via `download_f32`.
     Run the rest of the layer normally (f16 path). Update `x` with residual.
  3. After collecting all `x_norm` vectors, replay each through both GEMM paths.

This avoids any dispatch.rs changes. The manual loop reuses the same public
methods the forward pass calls (`rmsnorm`, `gemm_*`, `rope`, `attention`, etc.).

**Fallback (if manual loop proves too fragile):**

If replicating the forward loop is too complex or produces divergent logits
(see sanity check in Risk section), fall back to: use the tool-call prompt's
token embeddings as `x` for all layers (just the embedding output, batch_size =
prompt_length). This is a valid first-layer activation. It loses realism for
deeper layers but still surfaces per-site error patterns from the weight matrix
+ Q8_1 interaction, which is sufficient for site identification.

**F16 reference path:** On gfx1151 (RDNA3.5), the f16 reference is
`gemm_*_wmma` (the `has_wmma_f16` path, not `has_wmma_f16_gfx12`). The binary
detects arch from `gpu.arch` and selects the correct WMMA variant.

**Output:**

```
Layer | Site     | MaxErr   | MeanErr  | >Thresh  | Shape
------|----------|----------|----------|----------|--------
  0   | qkvza    | 0.0023   | 0.0001   |    0     | 4608x3584
  0   | gate_up  | 0.0089   | 0.0004   |   12     | 18944x3584
  0   | residual | 0.1340   | 0.0091   |  847     | 3584x3584
  ...
```

Top-3 worst (site, layer) pairs highlighted at the end. Exit code 1 if any
site exceeds threshold.

## Stage 2: channel-map

**Question:** Which output rows (channels) in the guilty site are worst?

**Input:** A specific site name (from stage 1) and optionally a layer number.
Defaults to the worst (site, layer) pair from stage 1.

**Method:**

Same activation capture as stage 1, but instead of aggregate stats, compute
per-output-row error:

For each row `r` in `[0, M)`:
- `row_max_err = max(|y_wmma[r, b] - y_mmq[r, b]|)` over batch dim `b`
- `row_mean_err = mean(|y_wmma[r, b] - y_mmq[r, b]|)` over batch dim `b`

Also report weight-matrix statistics for the worst rows:
- HFQ4 scale range (min/max scale across groups in that row)
- Dynamic range of the dequantized row
- Whether the row corresponds to a special-token-related output dimension
  (for the residual/Wo site, cross-reference with the output embedding to
  identify which rows feed special-token logits)

**Output:**

```
Site: residual | Layer: 14
Row  | MaxErr   | MeanErr  | ScaleRange     | DynRange
-----|----------|----------|----------------|----------
2731 | 0.1340   | 0.0312   | 0.019 - 0.024  | 1.23
2733 | 0.0980   | 0.0198   | 0.017 - 0.022  | 1.18
 ... (top 20 worst rows)

Special-token cross-reference (Wo -> output embedding -> token ID):
  Row 2731 contributes most to token 151644 (<|im_start|>) via output[151644]
```

## Stage 3: layer-sweep

**Question:** Is the error concentrated in specific layers?

**Input:** A specific site name. Runs across all layers.

**Method:**

Same as stage 1, but for a single site across all layers. Produces a
`[num_layers]` error vector.

**Output:**

```
Site: residual
Layer | MaxErr   | MeanErr  | >Thresh
------|----------|----------|--------
  0   | 0.0012   | 0.0001   |    0
  1   | 0.0014   | 0.0001   |    0
 ...
 26   | 0.1340   | 0.0091   |  847   << WORST
 27   | 0.0890   | 0.0045   |  312
```

## Implementation plan

### Single file: `crates/engine/examples/channel_test_mmq.rs`

Estimated ~350-450 lines. Structure:

```
main()
  parse_args()
  load_model()          // reuse qwen35::load_weights
  tokenize_prompt()     // reuse tokenizer
  capture_activations() // manual layer-by-layer forward, save x_norm per layer
  match stage:
    site_scan()         // loop layers x sites, call both paths, diff
    channel_map()       // single (site, layer), per-row diff
    layer_sweep()       // single site, all layers
  print_report()
  optionally write_json()
```

### Key functions

**`capture_activations(gpu, weights, tokens) -> Vec<Vec<f32>>`**

Manual layer loop:
1. Embed tokens -> `x` (on GPU)
2. For each layer:
   - `gpu.rmsnorm(x, norm_weight) -> x_norm`
   - `download_f32(x_norm)` -> push to `Vec`
   - Run rest of layer normally (f16 WMMA GEMM, RoPE, attention, FFN, residual add)
3. Return collected `x_norm` vectors (one per layer)

This mirrors `qwen35::forward_prefill_batch` but with download hooks. The
function needs access to the same scratch tensors the forward pass uses
(KV cache, attention scratch, etc.). We allocate these via the existing
`Qwen35Scratch` or equivalent.

**`compare_site(gpu, weight, x_data, m, k, batch_size, site_name) -> SiteStats`**

1. `upload_f32(x_data) -> x_gpu`
2. Allocate `y_wmma = upload_f32(zeros)` and `y_mmq = upload_f32(zeros)`
3. Set `gpu.capture_mode = true` (skip rocBLAS)
4. Call the appropriate WMMA method: `gpu.gemm_*_wmma(weight, x_gpu, y_wmma, ...)`
5. Call the MMQ method: `gpu.gemm_hfq4g256_mmq_set_prequant(weight, xq, y_mmq, ...)`
   (after `ensure_q8_1_mmq_x` to quantize activations)
6. `download_f32` both, compute stats
7. Set `gpu.capture_mode = false`

**`per_row_stats(y_wmma, y_mmq, m, batch_size) -> Vec<RowStats>`**

CPU-side diff, trivial.

### Layer type handling

Qwen 3.5/3.6 models have two layer types:
- **DeltaNet layers:** Use `gemm_qkvza_hfq4g256` (4-way fused: QKV+Z+alpha+beta)
- **FullAttn layers:** Use `gemm_qkv_hfq4g256` (3-way: Q+K+V)

Both use `gemm_gate_up_hfq4g256` and `gemm_hfq4g256_residual` for FFN.

The binary handles both via a `match` on `LayerWeights::DeltaNet` vs
`LayerWeights::FullAttn`, calling the corresponding GEMM variant.

For MoE layers (`DeltaNetMoe`, `FullAttnMoe`), the FFN uses routed experts
instead of dense gate_up/down. The channel test skips the MoE FFN sites
(the MMQ path is not used for expert dispatch) and only tests the attention
GEMM sites in MoE layers.

### What this does NOT change

- No changes to `dispatch.rs`
- No changes to kernels
- No changes to the forward pass
- No changes to the coherence gate
- No new crate dependencies (uses existing `clap` or manual arg parsing
  matching other examples in the crate)

### Output artifacts

- stderr: human-readable tables (always)
- optional `--json FILE`: structured JSON with all stats for scripting

```json
{
  "model": "qwen3.6-27b.mq4",
  "prompt_md5": "a1b2c3...",
  "stage": "site-scan",
  "arch": "gfx1151",
  "results": [
    {
      "layer": 0,
      "site": "qkvza",
      "shape": [4608, 3584],
      "max_err": 0.0023,
      "mean_err": 0.0001,
      "bad_count": 0,
      "threshold": 0.01
    }
  ]
}
```

## Success criteria

1. Stage 1 identifies which site(s) produce error above threshold on
   the tool-call prompt with real qwen3.6-27b weights on gfx1151.
2. Stage 2 identifies specific output rows responsible for the error.
3. Stage 3 shows whether the problem is layer-concentrated or distributed.
4. Results are reproducible (same model + prompt md5 = same numbers).
5. Binary runs without any changes to existing crate code.

## Risk: activation capture complexity

The manual layer-by-layer loop is the most complex part. It needs to replicate
the forward pass's layer logic (RMSNorm, GEMM, RoPE, attention, FFN, residual
add) correctly enough that the captured activations are representative.

**Mitigation:** Before running comparisons, verify that the manual loop produces
the same final logits as `forward_prefill_batch`. If they diverge, the captured
activations are wrong and the comparison is meaningless. The binary includes this
sanity check as a mandatory first step.

**Fallback:** If the manual loop proves too fragile to maintain, fall back to the
"simpler alternative" noted in stage 1: use embedding output as a uniform
first-layer activation for all layers. This loses realism (later layers get
wrong activations) but still surfaces per-site error patterns from the weight
matrix + Q8_1 interaction. The simpler approach is sufficient for stage 1
(site identification); stages 2-3 can use it too if the error pattern is
weight-dominated rather than activation-dominated.
