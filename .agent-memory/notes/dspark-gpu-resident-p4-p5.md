---
title: DSpark P4 GPU-resident markov chain + P5 GPU-resident hidden reuse (dGPU portability, byte-identical)
date: 2026-07-02
tags: [dspark,spec-decode,dgpu,portability,qwen3,deepseek4]
---

Landed the two DSpark optimizations the perf-audit session deferred as
"UMA-neutral / invasive". They were skipped because on gfx1151 UMA the
D2H/H2D copies rocprof flagged are ~free (coherent memory, warm L3). Directive:
"a dedicated vram device needs to work properly too" → do them as **dGPU
portability wins**. Branch `feature/dspark-qwen3`: P4 `d2d640cc`, P5 `b6f1db96`
(on top of `6aea6968`).

**P4 — GPU-resident markov head chain** (`run_heads`, `dspark_core.rs`).
The ≤8-slot sequential markov loop did 2 blocking host round-trips/slot: an F16
`markov_w1` embed (D2H row + host f16→f32 + H2D) and `argmax_f32`
(malloc+memset+D2H+free). Now chains fully on-GPU with the SAME primitives
qwen35 MTP uses: `argmax_token_chain_f32` writes each argmax into a device i32
token buffer (4-byte F32-typed, like `mtp_token_chain`), the next slot's embed
reads that id via a batched device-token-indexed lookup; block ids come back in
ONE D2H. New kernel `embedding_lookup_f16_batched` (+ `embedding_f16_batched.hip`)
for the F16 markov table; Q8_0/HFQ4-G256 reuse existing batched lookups;
other dtypes fall back to the byte-identical host path (`dspark_embed_one`).

**P5 — GPU-resident accepted-prefix hidden reuse** (`mtp_step`). Verify's
captured extract-layer hidden was D2H'd into a host Vec then a slice H2D'd back
(~215 KB D2H + 129 KB H2D + per-extract-layer stalls/window). New
`SpecTarget::verify_block_capture_gpu` (default `Err`; only llama + deepseek4
implement it — the 4 non-DSpark arches are untouched) captures straight into a
caller-owned GPU buffer; drafter slices the accepted prefix GPU→GPU. deepseek4
already captured into `dspark_caps` (swap D2H→dtod); llama's `HiddenCaptureSink`
grew an optional `hidden_gpu` dest writing position-major slots inline. Returns
a `captured` bool so llama's n<4 per-token fallback still re-bootstraps (matches
old empty-Vec branch).

**Both byte-identical BY CONSTRUCTION** — F16→F32 widening is exact,
`argmax_token_chain_f32` reduction == `argmax_f32` (strict `>`, low-index tie),
P5 is the same floats/layout copied GPU→GPU instead of via host. Cannot measure
the dGPU win here (box is UMA-only); validated correctness instead — the P1
precedent (`b1adf411`). Evidence: qwen3+deepseek4 dspark parity PASS
(deepseek4 `markov_w1 embed` bit-exact, max_abs 0.0), qwen3 dspark coherence
gate `--full` fluent incl. long-context multi-window, and **greedy output
byte-identical to P4-HEAD across the whole gate matrix** (git-stash A/B diff),
workspace `--locked` build + 414 lib tests green.

Reusable primitives added: `Gpu::embedding_lookup_f16_batched`,
`SpecTarget::verify_block_capture_gpu`, `llama_spec::verify_block_argmax_capture_gpu`,
`HiddenCaptureSink.hidden_gpu`. Related: `measure-spec-decode-on-the-daemon`,
`mtp-lmhead-not-the-lever`.
