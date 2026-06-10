# Multi-GPU serving

hipfire has two multi-GPU modes, selected per model load and **mutually
exclusive** (the daemon refuses a load that sets both):

| Mode | Load param | What's sharded | For | Arches |
|---|---|---|---|---|
| **Expert-parallel (EP)** | `tp` (`hipfire serve --tp N`) | Routed MoE experts (rank `r` owns experts `e % tp == r`); attention / KV / norms replicated per rank | Big-MoE models whose *weights* don't fit one card (DeepSeek V4 Flash ~82 GB, MiniMax-M2) | DeepSeek V4 (arch 9), MiniMax-M2 (arch 10) |
| **Pipeline-parallel (PP)** | `pp` | Contiguous layer bands; the residual stream flows dev0 → dev1 → … | Dense / hybrid models that need more *context or headroom* than one card | Qwen 3.5/3.6 dense + MoE (arch 5/6) |

---

# Expert-parallel (EP) — `--tp N`

**Status:** v1 landed (task #26 — design doc
`docs/plans/daemon-ep-wiring.md`, substrate validation log
`docs/plans/ship6-substrate-ep.md`). Greedy AR decode, validated on
4× gfx1201 (32 GiB each) for DeepSeek V4 Flash and MiniMax-M2
(MiniMax: coherent at 51.7 tok/s decode, ~24 GB/card shard load) —
models whose MQ2-Lloyd tiers (~82–86 GB) cannot fit any single 32 GB
card.

## How it works

`Gpus::init_tp(tp, n_layers)` brings up `tp` ranks; each rank loads the
full non-expert weight set (attention, norms, router, shared expert)
plus **only its owned routed experts** (`ExpertAssign::Stride`: rank
`r` owns experts `e % tp == r`; non-owned slots are zeroed dummies that
contribute 0). Every rank runs the full forward; the routed-expert
combine is the only cross-GPU step — one peer-direct copy+add
all-reduce of a `[hidden_size]` partial per MoE layer. Logits land on
rank 0 and sampling happens on the host, so the EP seam is
forward-only.

## Serving

```bash
# DeepSeek V4 Flash across 4 GPUs
HIP_VISIBLE_DEVICES=0,1,2,3 hipfire serve --tp 4
hipfire run deepseek-v4-flash "What is the capital of France?"
```

Driving the daemon JSON directly:

```json
{"type":"load","model":".../deepseek-v4-flash.mq2lloyd","params":{"max_seq":8192,"tp":4}}
```

`hipfire serve --tp N` sets `HIPFIRE_TP=N`; the CLI forwards it as
`params.tp` on every model load (only when > 1, so single-GPU loads
stay byte-identical). Loads are refused with a structured error when:

- `tp > 1` **and** `pp > 1` (mutually exclusive),
- `tp > 1` with a DFlash drafter configured (no spec-decode under EP v1),
- the model's arch_id is not 9 (DeepSeek V4) or 10 (MiniMax-M2),
- `init_tp` finds fewer devices than `tp` (check
  `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`).

## Generate semantics (v1)

The EP generate path (`generate_ep` in the daemon) accepts the normal
generate fields — `prompt`, `system_prompt`, `messages` (multi-turn
history), `tools`, `max_tokens`, `stop`, think mode — with v1 limits:

- **Greedy decode only.** Sampling params (temperature / top_p /
  penalties) are not applied on the EP path yet; logits are argmax'd
  host-side from rank 0.
- **MTP / DFlash / CASK / PFlash / VL: not available** under EP.
- Prompt rendering matches single-GPU: DeepSeek V4 goes through its
  DSML prompt builder (`<｜User｜>…<｜Assistant｜>`), MiniMax-M2 through
  its Jinja chat template with `tools` and the full message history
  threaded (see [JINJA.md](JINJA.md)).

**Tool calls are wired on the EP path.** For DeepSeek V4 the daemon
builds per-request tool schemas from the OpenAI `tools` array and runs
the DSML stream parser plus a **grammar-constrained token mask** during
decode — tokens that would malform a tool call are masked out of the
logits, and emitted calls stream as structured `tool_calls` events
with `finish_reason: "tool_calls"`. For MiniMax-M2 the template
renders tool definitions into the prompt; output streams as text.

> **Fixed in v0.2.1:** earlier ds4 EP builds emitted degraded tool-call
> syntax on RDNA4 (gfx12) because every F16-weight GEMV fell back to a
> broken kernel that corrupted the DSA compressor (fixed in 245ae3fe).
> EP tool-calling on hiptrx-class tp4 now produces token streams
> identical to the validated single-GPU path; on the canonical weather
> tool prompt the daemon emits `tool_calls` with
> `finish_reason: "tool_calls"` in both think and non-think modes.

Caching: MiniMax-M2 EP carries the single-GPU LCP prefix cache — every
rank's KV cursor rewinds to the longest common prefix of the rendered
conversation and only the divergent suffix re-prefills
(interleaved-thinking partial reuse). The MiniMax `<think>\n` opener is
re-emitted display-only so the assistant turn stays well-formed.
DeepSeek V4 EP keeps an assistant-turn cache for verbatim history
replay.

## Environment / debug knobs (EP)

| Variable | Effect |
|---|---|
| `HIP_VISIBLE_DEVICES=0,1,2,3` | Standard ROCm device filter — expose exactly the ranks you want |
| `HIPFIRE_TP=N` | What `hipfire serve --tp N` sets; any CLI-initiated load forwards it as `params.tp` |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB=N` | Pre-flight VRAM-asymmetry tolerance (default 2.0). Raise it when a busy card makes the uniform-VRAM check flag a false asymmetry |
| `HIPFIRE_DEEPSEEK4_DUMP_PROMPT=1` | Dump the rendered EP prompt (token count + decoded text) to stderr |
| `HIPFIRE_DEEPSEEK4_CACHE_TRACE=1` | Trace ds4 assistant-turn cache stores |
| `HIPFIRE_QWEN_CACHE_TRACE=1` | Trace the MiniMax EP LCP prefix-cache decisions |

## Validation harnesses (EP)

GPU-side parity and bring-up harnesses, runnable without the daemon:

```bash
# DeepSeek V4 Flash EP greedy decode (4× gfx1201 recipe)
HIP_VISIBLE_DEVICES=0,1,2,3 cargo run --release \
    -p hipfire-arch-deepseek4 --example ep_deepseek4 -- \
    --model ~/.hipfire/models/deepseek-v4-flash.mq2lloyd --tp 4 --max 48 \
    --prompt "The capital of France is"

# MiniMax-M2 EP greedy decode
HIP_VISIBLE_DEVICES=0,1,2,3 cargo run --release --features deltanet \
    -p hipfire-arch-minimax --example ep_minimax -- \
    --model ~/.hipfire/models/minimax-m2.mq2lloyd --tp 4 --max 32 \
    --prompt "The capital of France is"

# EP ≡ single-GPU decode parity (Qwen3.x-A3B substrate harness):
# run at tp=1 then tp=2 and diff the printed FNV hash.
HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
    -p hipfire-runtime --example ep_decode_parity -- \
    ~/.hipfire/models/qwen3.6-35b-a3b.mq4 2 24 "The capital of France is"
```

The EP examples print an FNV-1a fingerprint over the emitted token
ids; a `--tp N` run is expected to match the `--tp 1` / single-GPU
fingerprint (the EP combine is accumulate-style, so EP ≡ non-EP).

---

# Pipeline-parallel (PP) — `pp ≥ 2`

**Status:** v1 feature-complete — tracking
issue [#58](https://github.com/Kaden-Schutt/hipfire/issues/58). Stages
0–9 of the v2 plan are merged; refusal contracts (DFlash / VL / CASK +
pp>1) are wired and validated. This section is the source of truth for
memory budget, deployment recipes, throughput, and known limitations.

## Why PP

hipfire on a single 24 GB card hits VRAM walls on:

- 27B at `--max-ctx ≥ 16K` with `kv_mode=asym3` (`AGENTS.md:356`)
- 35B-A3B at `--max-ctx ≥ 4K` with FP32 KV
- hypothetical 80B-A3B at any context

Pipeline-Parallel (PP) shards layers across N devices. Each device owns a contiguous "band"
of consecutive layers. The residual stream `s.x` flows through the bands sequentially: dev_0
runs layers `0..k1`, copies `s.x` to dev_1, dev_1 runs layers `k1..k2`, and so on. Final
`output_norm + lm_head` run on the last device (dev_last) — its `s.logits` is read by the
sampler in place.

**What PP gives you on 2× 24 GB:**
- Run 27B / 35B-A3B that don't fit on one card with extended context
- Unlock max_ctx on 27B beyond single-GPU OOM limits
- ~50-70% of single-GPU throughput on already-fitting models (sequential PP=2 is slower per token)

**What PP does NOT give you:**
- Faster multi-user serving — that's TP (tensor parallel), separate roadmap
- Speedup on models that already fit on one card

## Memory budget (per-card, PP=2)

Numbers below are **measured** on 2× Radeon RX 7900 XTX (gfx1100, 25.8 GiB VRAM
each) via `crates/hipfire-runtime/examples/pp2_vram_probe.rs` —
`hipMemGetInfo` deltas captured at each allocation stage
(`load_weights_multi`, `Qwen35ScratchSet`,
`KvCache::new_gpu_asym3_capped_multi`, `DeltaNetState`). Per-card columns
report the worst-of-two (the device that holds more — typically dev_last,
which carries `output_norm + lm_head`). `total` is the sum across both cards.

| Model | quant | n_layers | dim | KV mode | ctx | weights | KV/card | scratch+DN/card | total | per-card max | fits 24 GiB? |
|-------|-------|----------|-----|---------|-----|---------|---------|-----------------|-------|--------------|--------------|
| qwen3.5:0.8b | mq4 | 24 | 1024 | asym3 | 4096 | 1.3 GB | 50 MB | 8 MB | 1.3 GB | 0.7 GB | yes |
| qwen3.5:4b | mq4 | 32 | 2560 | asym3 | 4096 | 4.0 GB | 134 MB | 15 MB | 4.0 GB | 2.0 GB | yes |
| qwen3.5:9b | mq4 | 32 | 4096 | asym3 | 4096 | 5.6 GB | 134 MB | 19 MB | 5.6 GB | 2.8 GB | yes |
| qwen3.5:9b | mq4 | 32 | 4096 | asym3 | 16K | 6.2 GB | 436 MB | 46 MB | 6.2 GB | 3.1 GB | yes |
| qwen3.5:9b | mq3 | 32 | 4096 | asym3 | 4096 | 4.4 GB | 134 MB | 19 MB | 4.4 GB | 2.2 GB | yes |
| qwen3.5:27b *(via 3.6 proxy)* | mq4 | 64 | 5120 | asym3 | 4096 | 15.5 GB | 268 MB | 42 MB | 15.5 GB | 7.8 GB | yes |
| qwen3.5:27b *(via 3.6 proxy)* | mq4 | 64 | 5120 | asym3 | 16K | 16.8 GB | 872 MB | 80 MB | 16.8 GB | 8.4 GB | yes |
| qwen3.5:27b | mq3 | 64 | 5120 | asym3 | 4096 | 12.6 GB | 268 MB | 40 MB | 12.6 GB | 6.3 GB | yes |
| qwen3.5:27b | mq3 | 64 | 5120 | asym3 | 16K | 13.8 GB | 872 MB | 78 MB | 13.8 GB | 6.9 GB | yes |
| qwen3.6:35b-a3b | mq4 | 40 | 2048 | asym3 | 4096 | 23.5 GB | 103 MB | 21 MB | 23.5 GB | 11.8 GB | yes |
| (hypothetical) 80B-a3b | mq4 | 80 | 8192 | asym3 | 4096 | ~42 GB | ~250 MB | ~2 GB | ~44 GB | ~22 GB | estimate, tight |

Notes:
- `qwen3.5:27b mq4` rows use `qwen3.6:27b mq4` as a measurement proxy (same
  `n_layers=64`, `dim=5120`, `head_dim=256`, `n_kv_heads=4` — VRAM-equivalent).
  When `qwen3.5:27b mq4` lands as a downloadable artifact the rows can be
  re-measured directly.
- `qwen3.6:35b-a3b mq4` is the current-generation A3B; `qwen3.5:35b-a3b mq4`
  ships local-only with the same MoE shape — measurement carries over.
- The 80B row stays an estimate — no public artifact exists.
- "scratch+DN/card" combines `Qwen35ScratchSet::per_device[i]` (residual
  stream, attention/FFN scratch, flash partials, logits) and the
  `DeltaNetState` slice owned by that device's LA-layer band.

**Asymmetry under Variant 2 (lm_head on dev_last):**
- dev_0 carries `token_embd`
- dev_last carries `output_norm + lm_head`
- Layer count shifts by ±1 for `n_layers % 2 != 0` (uniform split formula
  `base + (i < rem ? 1 : 0)`)

Probed on this hardware: per-card max < 24 GiB at every measured shape. The
A3B model at 11.8 GB/card has the largest headroom consumer; everything
else stays under 8.5 GB/card with 4 K context, under 9 GB/card at 16 K.

To reproduce on your hardware:

```sh
HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
    -p hipfire-runtime --example pp2_vram_probe -- \
    ~/.hipfire/models/qwen3.5-9b.mq4 4096
```

## Deployment recipes

The daemon takes a `pp` field in the load message (default `1` =
single-GPU, identical behavior to pre-PP code paths):

```sh
# Filter to the two 7900 XTX (drop the iGPU)
HIP_VISIBLE_DEVICES=0,1 hipfire run qwen3.5:27b --max-ctx 16384 "Hi"

# Bypass the inherited `gemm_..._wmma_ksplit` non-determinism (k-split
# atomicAdd reduction varies by warp scheduling — see commit f54ca71
# and kernels/src/gemm_hfq4g256_residual_wmma_ksplit.hip:22). Required
# for byte-equivalent output across processes / pp configurations.
HIPFIRE_DETERMINISTIC=1 HIP_VISIBLE_DEVICES=0,1 hipfire run qwen3.5:9b "Hi"

# Override uniform-VRAM tolerance (default 2 GiB; arches must match)
HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB=4 hipfire run qwen3.5:27b "Hi"
```

Direct daemon JSON (driving without the CLI):

```json
{"type":"load","model":".../qwen3.5-27b.mq4","params":{"max_seq":16384,"pp":2}}
```

### Environment variables

| Variable | Effect |
|----------|--------|
| `HIP_VISIBLE_DEVICES=0,1` | HIP runtime device filter (standard ROCm) |
| `HIPFIRE_DETERMINISTIC=1` | Force k2 WMMA reduction (no atomicAdd) — bit-identical across processes/pp configs at ~33% perf cost on small-batch decode |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB=N` | Pre-flight VRAM-asymmetry tolerance for `Gpus::init_uniform` (default 2.0) |
| `HIPFIRE_PREFILL_BATCHED=0` | Disable batched WMMA prefill (per-token fallback). Diagnostic for ksplit non-det isolation |
| `HIPFIRE_PREFILL_MAX_BATCH=N` | Override per-chunk prefill batch (default `PREFILL_MAX_BATCH`); chunks > N split with peer-copy at boundary |
| `HIPFIRE_WO_WMMA_VARIANT={k2,ksplit,k4,…}` | Manual override of the wo-residual GEMM variant — see `dispatch.rs` auto-dispatch |

### Refusal matrix at load (`pp > 1`)

| Feature | Behavior | Why |
|---------|----------|-----|
| arch_id ∈ {5, 6} (Qwen3.5 dense + MoE/A3B) | Accepted | Validated end-to-end |
| arch_id = others (LLaMA / Qwen3) | Refused | Single-GPU only in v1 |
| VL models (vision_config + vision tensors) | Refused | v1.1 |
| DFlash draft (`draft` field set) | Refused | v1.1 — see `feedback_cask_mfold_dflash_broken.md` for the v1 ship-blocker |
| CASK / TriAttention sidecar | Refused | Eviction context is single-device — v1.1 |

## Throughput baseline (gfx1100 × 2)

Measured on 2× Radeon RX 7900 XTX, ROCm 6.4.3, with
`HIPFIRE_DETERMINISTIC=1` (bit-equivalent pp=1 ↔ pp=2 output).

| Model | Prompt | pp=1 prefill | pp=2 prefill | pp=1 decode | pp=2 decode | pp=2/pp=1 decode |
|-------|--------|--------------|--------------|-------------|-------------|------------------|
| 0.8B mq4 | 22 tok | 838 tok/s | 588 tok/s | 332 tok/s | 227 tok/s | 68% |
| 0.8B mq4 | 322 tok (chunked) | 6493 tok/s | 5490 tok/s | 315 tok/s | 212 tok/s | 67% |
| 35B-A3B mq4 (MoE) | 15 tok | 331 tok/s | 258 tok/s | 142 tok/s | 97 tok/s | 68% |

The pp=2 decode penalty is inherent to v1: per-token
`forward_scratch_multi` pays one HIP launch per kernel per layer with
no graph capture (vs pp=1 which captures + replays the AR-step graph
after warmup). Pipelined decode + per-band graph capture lift this in
v1.1.

## Limitations (v1)

Refused at load time:
- `pp > 1` + DFlash speculative decode
- `pp > 1` + CASK/TriAttention sidecar (eviction is single-device)
- `pp > 1` + VL models (vision encoder is single-device)
- `pp > 1` + arch_id ∉ {5, 6} (LLaMA / Qwen3 dense are pp=1 only)

Architectural limits in v1:
- Homogeneous arch only (`init_uniform` hard-fails on arch mismatch)
- Uniform layer split — `init_layers(per_device)` is the manual escape hatch; `init_vram_weighted` stubbed
- Per-token decode (no async stream pipeline / per-band graph capture) — v1.1
- Pipelined prefill (chunk N+1 on dev_0 while chunk N processes on dev_1) — v1.1

## Validation (Stage 9)

```sh
# Multi-GPU gate. Skips silently when fewer than 2 GPU visible.
./scripts/pp-gate.sh

# Just the parity smoke (no daemon end-to-end), faster
./scripts/pp-gate.sh --skip-end-to-end

# Underlying byte-equivalence example
HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
    -p hipfire-runtime --example pp_parity_chatml -- \
    ~/.hipfire/models/qwen3.5-0.8b.mq4
```

The `pp-gate.sh` battery checks:
1. Per-token `forward_scratch_multi` ≡ `forward_scratch` bit-exact (pp_parity_chatml)
2. Daemon `pp=1` ≡ `pp=2` byte-identical with `HIPFIRE_DETERMINISTIC=1` (greedy ChatML)
3. DFlash + pp=2 refusal at load
4. CASK + pp=2 refusal at load

A pre-commit hook calls `pp-gate.sh` automatically when staged files
match the `multi_gpu|pp_|peer_access|pipeline|stages` hotspot regex.

## Verifying peer access on your hardware

```sh
rocm-smi --showtopo            # weights, hops, link types
rocm-smi --showtoponuma        # NUMA placement
rocm-smi --showtopoaccess      # peer accessibility matrix
```

A `True` in every cell of `--showtopoaccess` for the cards you plan to use means peer-access
should work. If not, hipfire falls back to host-staging via pinned buffers (slower but correct).

## Open questions (will be filled in as Stages land)

- DFlash + PP integration scope — pending maintainer guidance on issue #58
- Whether mixed-arch should be soft-warn or hard-fail — currently hard-fail
- API surface for `HIPFIRE_DEVICES` — current direction is logical IDs post-VISIBLE
