<!-- SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Kaden Schutt; hipfire — see LICENSE and NOTICE in the project root. -->

# Audit: ArchQwen

slice=ArchQwen

JSON contract:
{
  "slice": "ArchQwen",
  "broken": [
    {"title": "Arch-6 EP MoE loads; generate routes dense TP", "path_line": "crates/hipfire-generate/src/qwen.rs:240", "verified": true, "summary": "5|6 → ep_serve_qwen35_dense_tp; dense_tp refuses num_experts!=0 (config.rs:214) and MoE layers (forward.rs:4168); forward_ep exists but unwired. #683."},
    {"title": "VL weight fail still sets vision_config", "path_line": "crates/hipfire-loader/src/carriers.rs:533-542", "verified": true, "summary": "load_weights map_err eprintln .ok(); has_vision_encoder only checks vision_config (lib.rs:1177) → image gate open with None weights."},
    {"title": "Multi-GPU DeltaNet EF residual empty", "path_line": "crates/hipfire-arch-qwen35/src/qwen35/weights.rs:1746-1749", "verified": true, "summary": "new_with_quant_multi leaves s_ef_residual empty → stochastic DN; single-GPU Q8 EF default-on."},
    {"title": "DecodeBatchState hardcodes Q8 KV/DN", "path_line": "crates/hipfire-arch-qwen35/src/qwen35/batch.rs:538-545", "verified": true, "summary": "new always Q8 filtered KV + StateQuant::Q8; reset_lane uses q8_lane_view only."},
    {"title": "dtype_from_quant_type subset of load quants", "path_line": "crates/hipfire-arch-qwen35/src/qwen35/weights.rs:186-221", "verified": true, "summary": "Graded EP map misses load-admitted qts; fails closed as unsupported quant_type."},
    {"title": "Dense TP has no mrope/VL path", "path_line": "crates/hipfire-arch-qwen35/src/qwen35/forward.rs:4132-4143", "verified": true, "summary": "forward_scratch_dense_tp lacks MropeCtx; VL single-device uses forward_scratch_mrope."},
    {"title": "forward_ep embed format allow-list narrow", "path_line": "crates/hipfire-arch-qwen35/src/qwen35/ep_batch.rs:2180-2196", "verified": false, "summary": "Only HFQ4G256/128, Q8_0, F32; suspicious vs wider single-GPU embed."}
  ],
  "missing": [
    {"title": "ep_serve_qwen35_moe / admission refuse for arch 6 EP", "path_line": "crates/hipfire-generate/src/qwen.rs:240", "verified": true, "summary": "Arch implements forward_ep + EP batch state; generate never drives them. #683."},
    {"title": "Atomic VL config+weights publish", "path_line": "crates/hipfire-loader/src/lib.rs:2102-2103", "verified": true, "summary": "finish_qwen35_load assigns pair without cross-check."},
    {"title": "Multi-GPU EF + batch non-Q8 DN/KV", "path_line": "crates/hipfire-arch-qwen35/src/qwen35/weights.rs:1746", "verified": true, "summary": "No multi EF; batch path not parameterized on StateQuant/KvMode."},
    {"title": "Serve/SlotEngine continuous-batch and EP bridge", "path_line": "crates/hipfire-arch-qwen35/src/serve_engine.rs:42-43", "verified": true, "summary": "Single-weight single-Gpu alternate path; CB deferred; no EP."},
    {"title": "Q4 DN tree-verify preflight", "path_line": "crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:4531-4536", "verified": true, "summary": "Runtime refuse only; no load-time gate."},
    {"title": "qwen2/llama shared transformer load extraction", "path_line": "crates/hipfire-arch-qwen2/src/qwen2.rs:22-28", "verified": true, "summary": "Duplicated load helpers; baseline capability deltas vs qwen35."},
    {"title": "Vision GPU quant + non-gfx1100 validation", "path_line": "crates/hipfire-arch-qwen35-vl/src/qwen35_vl.rs:351-370", "verified": true, "summary": "Host dequant HFQ4→F16; warn-only other archs."},
    {"title": "EP beyond frozen 4×gfx1201", "path_line": "crates/hipfire-arch-qwen35/src/qwen35/ep_batch.rs:185-220", "verified": true, "summary": "rank_count!=4, REAP, paged refused."}
  ],
  "changes": [
    {"title": "Refuse arch-6 EP at admit or wire moe serve", "path_line": "crates/hipfire-generate/src/qwen.rs:240", "cost": "hours (refuse) / days–week (wire)", "summary": "Split 5 vs 6; drive forward_ep or fail before 19GB alloc."},
    {"title": "VL load fail-closed", "path_line": "crates/hipfire-loader/src/carriers.rs:533-542", "cost": "hours", "summary": "Pair config with weights; fix has_vision_encoder."},
    {"title": "Quant matrix test load↔dtype_from_quant_type", "path_line": "crates/hipfire-arch-qwen35/src/qwen35/weights.rs:186", "cost": "hours–1 day", "summary": "Table every qt; document intentional EP unsupported set."},
    {"title": "Multi-GPU DN EF or explicit degrade", "path_line": "crates/hipfire-arch-qwen35/src/qwen35/weights.rs:1746", "cost": "days (wire) / hours (degrade)", "summary": "Per-device residual + reset, or loud EF off on multi."},
    {"title": "Batch quant param or refuse non-Q8", "path_line": "crates/hipfire-arch-qwen35/src/qwen35/batch.rs:538", "cost": "days / hours", "summary": "Align DecodeBatchState with carrier modes."},
    {"title": "Dense TP VL refuse or mrope", "path_line": "crates/hipfire-arch-qwen35/src/qwen35/forward.rs:4132", "cost": "hours / days", "summary": "Admit-time refuse cheaper than TP mrope."},
    {"title": "Document SlotEngine vs EP vs CB", "path_line": "crates/hipfire-arch-qwen35/src/serve_engine.rs:1-15", "cost": "hours", "summary": "Maintainer map of three multi-request shapes."},
    {"title": "qwen2/llama extraction + bias docs", "path_line": "crates/hipfire-arch-qwen2/src/qwen2.rs:22", "cost": "days+ / hours", "summary": "Shared transformer load; Dir bias policy clarity."}
  ],
  "report": "(full markdown in summary field; parent persists — scouts have no Write/local://)"
}
