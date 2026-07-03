# hipfire Source File Index (2026-07-03)

Companion to `architectural-review-2026-07-03.md`. One row per source file (large example/kernel-variant trees are clustered). Descriptions derived from doc comments and public items, not filenames.

# Architectural Source Index — GPU Core Crates

Scope: crates/rdna-compute, crates/hip-bridge, crates/hsa-bridge. Read-only survey.

## crates/rdna-compute

Kernel compilation, caching, and per-architecture dispatch layer for RDNA/GCN/CDNA GPUs.

### src/ top-level

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 27 | Crate root; re-exports dispatch, compiler, pool, arch, profiling modules. |
| src/arch_caps.rs | 688 | Per-architecture capability defaults (WMMA, wave size, LDS, dtype support). |
| src/compiler.rs | 714 | Compiles HIP kernel sources to .hsaco code objects via hipcc. |
| src/kernels.rs | 4751 | Built-in HIP kernel source strings for all inference operations. |
| src/feature_flags.rs | 481 | Typed immutable env-var resolution for rdna-compute tuning knobs. |
| src/generic_warn.rs | 138 | Runtime coverage map warning when generic fallback kernels are hit. |
| src/pool.rs | 100 | GPU memory pool eliminating hipMalloc/hipFree overhead in the hot loop. |
| src/profile.rs | 373 | Per-kernel bandwidth profiling accounting. |
| src/profiler.rs | 604 | Kernel efficiency profiler for RDNA GPUs. |
| src/profile_rocprof.rs | 343 | rocprofv3 stats CSV parser and internal-vs-rocprof coverage cross-checker. |

### src/dispatch/ (per-op dispatch, file-level)

| file | LOC | description |
| --- | --- | --- |
| dispatch/mod.rs | 5153 | High-level GPU dispatch interface; owns state, macros, method routing. |
| dispatch/activation.rs | 1169 | Elementwise/activation kernels (silu/gelu/swiglu, softmax, layernorm). |
| dispatch/norm.rs | 307 | RMSNorm dispatch (f32, batched, train fwd/bwd, slot-buf). |
| dispatch/embedding.rs | 309 | Token/position embedding lookup (F32/Q8/Q4K/HFQ4, batched). |
| dispatch/sampling.rs | 491 | Sampling/selection kernels (top-p, greedy-accept, top-k, argmax, max-reduce). |
| dispatch/rocblas.rs | 126 | rocBLAS GEMM fallback wrappers plus arch-eligibility helpers. |
| dispatch/rope.rs | 1316 | RoPE plus MQ/FWHT rotation dispatch and rotate scratch-prep bridge. |
| dispatch/gated.rs | 937 | Gated-norm and gated-delta-net (DeltaNet/GLA) dispatch. |
| dispatch/conv1d.rs | 629 | Depthwise causal short-conv dispatch (decode, gated-decode, SiLU). |
| dispatch/mamba2.rs | 625 | Mamba-2 SSD selective state-space mixer dispatch (nemotron_h). |
| dispatch/kv.rs | 1732 | KV cache write/read dispatch (all quant formats) plus KVarN sliding-window. |
| dispatch/fused.rs | 4308 | Fused kernels (gate-up, QKV+Z+A, rmsnorm+rope+rotate fusions). |
| dispatch/attention.rs | 5563 | Attention dispatch: flash-decode, prefill, GQA, SWA, TriAttention, ViT. |
| dispatch/gemm_base.rs | 1053 | Base-dtype GEMM dispatch (f16/bf16/f32, WMMA, train). |
| dispatch/gemm_gate.rs | 2603 | Fused gate/up MLP-projection GEMMs across all dtypes. |
| dispatch/gemm_qkv.rs | 5890 | Fused QKV and QKV+Z+A projection GEMMs across all dtypes. |
| dispatch/gemm_hfq.rs | 2916 | HFQ-quantized GEMM dispatch (hfq2/3/4/6/8 g128/g256). |
| dispatch/gemm_misc.rs | 2133 | Remaining GEMM dtypes (q8_0, mq/lloyd, fp4/oq/iu, paro, s4s4). |
| dispatch/gemv.rs | 6121 | Decode-shape GEMV dispatch across all dtypes and MoE-scalar-indexed. |
| dispatch/moe.rs | 770 | MoE gating/routing dispatch (router top-k, expert gather). |
| dispatch/quant.rs | 388 | Reference-layer activation quant kernels (W4A4 oq4, W4A8 oq8, per-token). |
| dispatch/misc.rs | 1191 | Misc ops: residual-quant, Paro, Givens, deinterleave, cross-entropy, cast, transpose. |
| dispatch/deepseek4.rs | 3536 | DeepSeek V4 Flash cluster: hyper-connections, NSA indexer/compressor, hash routing. |
| dispatch/zaya_cca.rs | 561 | Dispatch wrappers for ZAYA1 CCA plus EDA/MoD custom kernels. |

### src/dispatch/overlays/ (per-arch whole-method GPU overlays)

| file | LOC | description |
| --- | --- | --- |
| overlays/mod.rs | 15 | Declares per-arch overlay submodules of whole-method GPU dispatch. |
| overlays/gfx11.rs | 232 | Generic RDNA3 (gfx1100) kernel-dispatch overlays. |
| overlays/gfx1151.rs | 2014 | RDNA3.5 Strix Halo APU kernel-dispatch overlays. |
| overlays/gfx12.rs | 2235 | RDNA4 (gfx12xx) kernel-dispatch overlays. |
| overlays/gfx906.rs | 633 | GCN5 Vega20 (gfx906 MI50) kernel-dispatch overlays. |
| overlays/gfx942.rs | 354 | CDNA3 MI300 (gfx942) kernel-dispatch overlays. |

### examples/ (137 files — GPU benches, parity checks, correctness tests; grouped)

| file(s) | LOC | description |
| --- | --- | --- |
| bench_*.rs (21 files) | 99–412 | Throughput/bandwidth microbenchmarks for gemv/gemm/attention/moe/quant kernels. |
| parity_*.rs (35 files) | 89–343 | CPU-vs-GPU numerical parity checks for gemm/gemv/flash/kvarn/quant paths. |
| test_*.rs (72 files) | 41–581 | Correctness harnesses for gemm/gemv/moe/conv1d/mamba2/attention kernel variants. |
| validate_*.rs (2 files) | 280–289 | End-to-end recipe validation for Opus W4A4 quantization pipelines. |
| examples/common/q8_test_utils.rs | 70 | Shared Q8 test-data generation/dequant helpers for example harnesses. |
| examples/gen_kernel_hashes.rs | 154 | Generates reference kernel-source hashes for cache-invalidation checks. |
| examples/hip_graph_poc.rs | 333 | HIP graph capture/replay proof-of-concept for kernel launch batching. |
| examples/zaya_kernels_smoke.rs | 288 | Smoke test exercising ZAYA1 CCA custom kernel dispatch. |
| examples/dual_gpu_smoke.rs | 174 | Multi-GPU launch/peer smoke exercise. |

Notable example clusters: test_moe_grouped_wmma_*.rs (7 files, 253–581 LOC) cover grouped-MoE WMMA across dtypes; test_gemm_*/test_gemv_* dominate the quant-format matrix; parity_kvarn_*.rs (6 files) cover the KVarN sliding-window read/write/flash paths.

## crates/hip-bridge

Thin dlopen FFI wrapper around libamdhip64.so plus rocBLAS/RCCL for HIP runtime, memory, and kernel launch.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 109 | Crate root wiring HIP runtime, error, kernarg, rccl, rocblas modules. |
| src/error.rs | 108 | Error types for HIP runtime operations. |
| src/ffi.rs | 1531 | FFI bindings to libamdhip64.so via dlopen (device, memory, stream, launch). |
| src/kernarg.rs | 182 | Kernarg blob builder for HipRuntime::launch_kernel_blob. |
| src/rccl.rs | 379 | Minimal FFI wrapper around librccl.so for tensor-parallel collectives. |
| src/rocblas.rs | 298 | Minimal FFI wrapper around librocblas.so for MFMA-accelerated GEMMs. |
| examples/smoke.rs | 45 | Basic HIP runtime load/device-init smoke test. |
| examples/smokeQA.rs | 75 | Extended HIP runtime QA smoke exercise. |
| examples/kernel_launch.rs | 133 | Demonstrates compiling and launching a HIP kernel through the bridge. |
| examples/peer_smoke.rs | 144 | Multi-GPU peer-access enable/copy smoke test. |
| examples/rccl_smoke.rs | 152 | RCCL collectives (allreduce) smoke test. |

## crates/hsa-bridge

Thin dlopen Rust wrapper around libhsa-runtime64.so for low-level HSA queue/agent/dispatch access.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 726 | hsa-bridge public API wrapping libhsa-runtime64.so agents and queues. |
| src/error.rs | 45 | HSA error handling. |
| src/ffi.rs | 453 | FFI bindings to libhsa-runtime64.so via dlopen (agents, queues, dispatch packets). |
| examples/hsa_vs_hip_launch.rs | 434 | Compares HSA-direct vs HIP kernel launch latency. |
| examples/hip_graph_gemv_poc.rs | 743 | HIP graph GEMV capture/replay proof-of-concept via HSA path. |
| examples/hip_graph_extra_poc.rs | 587 | Additional HIP graph capture proof-of-concept variants. |

## crates/hipfire-runtime

The inference hot path: model loading, LLaMA/hybrid execution, KV cache, quant decode, and speculative decoding on RDNA GPUs.

### src (file level)

| file | LOC | description |
| --- | --- | --- |
| arch.rs | 790 | Per-architecture bring-up contract trait each model implements |
| bf16_loader.rs | 209 | BF16 HuggingFace safetensors model loader scaffold |
| calibration.rs | 867 | Model-agnostic activation-capture calibration collector |
| cask.rs | 753 | Core-aware selective KV compression (CASK) |
| config.rs | 117 | Typed immutable env-var resolution for the runtime |
| cpu_router.rs | 200 | CPU-side MoE router replica |
| ddtree.rs | 1164 | Tree-structured speculative verification built from DFlash draft |
| dflash.rs | 1609 | DFlash draft forward pass in native Rust+HIP |
| dispatch.rs | 95 | Generic kernel-dispatch family accessors and re-exports |
| env_docs.rs | 4259 | Generated catalog documenting all runtime environment variables |
| eos_filter.rs | 10 | Compatibility re-export for generation output filtering |
| ep.rs | 160 | Expert-parallel executor for the super-op substrate |
| gguf.rs | 11 | Compatibility re-export for the GGUF artifact parser |
| hfq_modules.rs | 313 | Module metadata for HFQM v2 containers |
| hfq.rs | 2526 | HFQ quantized-model (.hfq) file loader |
| host_profile.rs | 1455 | Measured host capability profiling for eval reports |
| kld_eval.rs | 336 | Single forward seam plus self-score / KLD evaluation |
| kv_adaptive.rs | 366 | Runtime VRAM-fit downshift of K/V cache precision |
| kv_hier.rs | 547 | Deferred-hierarchical KV cache (flag-gated) |
| kv.rs | 2596 | Generic GPU KV cache for autoregressive generation |
| lib.rs | 58 | Crate root wiring module exports |
| llama.rs | 3153 | LLaMA model implementation using RDNA GPU compute |
| logging.rs | 27 | Stderr logging initialization helper |
| loop_guard.rs | 52 | Compatibility wrapper for generation loop-guard policy |
| model_source.rs | 28 | Runtime adapter for concrete model-source opening |
| mtp_mirror.rs | 200 | Cross-device weight mirroring for hetero MTP |
| multi_gpu.rs | 926 | Multi-GPU pipeline-parallel orchestration and layer bands |
| quant.rs | 384 | Generic dequant codecs and half/bf16 conversions |
| safetensors_source.rs | 312 | Load HuggingFace safetensors models directly |
| sampler.rs | 860 | Logit-space sampling: top-p, temperature, repeat penalty |
| sequence_state.rs | 202 | Unified per-sequence decode state container |
| speed_bench.rs | 233 | Shared speed-benchmark utilities |
| tokenizer.rs | 11 | Compatibility re-export for the tokenizer implementation |
| tool_call.rs | 660 | Per-arch tool-call output parsers |
| tp_shard.rs | 575 | Tensor-parallel shard configuration |
| transformer.rs | 195 | Shared batched-prefill composition seam for dense/hybrid arches |
| triattn.rs | 1275 | TriAttention KV-cache compression via trigonometric scoring |
| weight_pager.rs | 1554 | Runtime residency management for MoE/dense weights |
| weights.rs | 1549 | GPU weight/embedding types plus weight-GEMV and rotation |

### src/bin (file level)

| file | LOC | description |
| --- | --- | --- |
| bin/hfq.rs | (see file) | CLI to inspect and edit HFQ containers and sidecars |
| bin/hipfire_host_profile.rs | (see file) | CLI emitting measured host capability profile |

### examples (148 files, ~42292 LOC, clustered by theme)

| cluster | count | description |
| --- | --- | --- |
| test_* | 31 | Ad-hoc runtime/model behavior test drivers |
| bench_* | 16 | Throughput and latency microbenchmark harnesses |
| profile_* | 11 | Profiling and instrumentation drivers |
| triattn_* | 9 | TriAttention parity, accuracy, and decode demos (e.g. triattn_gpu_parity.rs, triattn_accuracy_sweep.rs) |
| dump_* | 6 | Tensor/state dump utilities |
| infer_* | 5 | End-to-end inference entrypoint demos |
| dflash_* / mtp_* / pp_* | 12 | Speculative-decode, multi-token-prediction, and pipeline-parallel demos (dflash_spec_demo.rs, mtp_mirror_smoke.rs, pp_parity_chatml.rs) |
| debug_* / diag_* / probe_* / check_* | ~10 | Debugging, diagnostics, and coherence probes (coherence_probe.rs) |
| parity singles | ~6 | Numeric parity harnesses (f16_gemv_parity.rs, oq4_weight_gemv_parity.rs, parity_weight_gemm_w8a8.rs, ep_decode_parity.rs) |
| a3b_* | 3 | A3B model smoke/load/multiturn checks (a3b_smoke_forward.rs) |
| other singles | ~39 | Assorted one-off tools (perplexity.rs, imatrix_collect.rs, oq4_repack.rs, quant/render/jinja/tokenize/encode/decode) |

Notable examples: perplexity.rs, imatrix_collect.rs, coherence_probe.rs, triattn_gpu_parity.rs, pp_parity_chatml.rs.

# hipfire-arch-* Source File Index

## hipfire-arch-deepseek4
DeepSeek V4 Flash architecture (arch_id 9): MoE decoder with DSML tool-calling, MTP spec-decode, and grammar-guided output.

| file | LOC | description |
| --- | --- | --- |
| examples/deepseek4_chat.rs | 475 | Interactive chat driver applying the DeepSeek chat template |
| examples/dump_hfq_dtypes.rs | 29 | Utility dumping tensor dtypes from a DeepSeek HFQ |
| examples/ep_deepseek4.rs | 357 | Expert-parallel greedy decode across N GPUs (Ship 6) |
| src/arch.rs | 1279 | Architecture trait impl with host-only and sharded weight loading |
| src/deepseek4.rs | 1327 | Config, Weights, and State types plus HFQ parsing |
| src/dsml.rs | 1003 | DSML tool-calling markup rendering and prompt assembly |
| src/forward.rs | 9202 | Full forward pass: decode step, hipGraph capture, EP forward |
| src/grammar.rs | 1202 | Grammar-guided constrained decoding for DSML tool calls |
| src/kld.rs | 71 | KLD-scoring seam over loose resident weights |
| src/lib.rs | 67 | Crate root re-exporting the DeepSeek V4 modules |
| src/sampling.rs | 213 | Greedy argmax sampler with Xorshift RNG |
| src/spec_decode.rs | 471 | MTP-head speculative decoding step with grammar support |

## hipfire-arch-dots-ocr
dots.ocr layout-analysis VLM (Qwen2-VL family): vision tower plus 2-D spatial RoPE for OCR.

| file | LOC | description |
| --- | --- | --- |
| examples/dump_proj_weight.rs | 106 | Dumps a vision-tower attention projection weight tensor |
| examples/infer_dots_ocr.rs | 468 | Vision-tower inference validation driver |
| examples/ocr_e2e.rs | 384 | End-to-end OCR validation (Strategy A step 2) |
| src/arch.rs | 169 | Architecture trait impl dispatching text and vision forward |
| src/dots_ocr.rs | 1743 | Model config, weights, and vision-config defaults |
| src/image.rs | 739 | Image preprocessing: smart-resize, CLIP normalise, patch extraction |
| src/lib.rs | 76 | Crate root for the dots.ocr VLM |
| src/rope.rs | 375 | 2-D spatial RoPE table construction for vision patches |

## hipfire-arch-gemma3
Gemma3 text decoder (arch_id 12): bring-up triple with per-token forward and HFQ weight loader.

| file | LOC | description |
| --- | --- | --- |
| examples/infer_gemma3.rs | 175 | Standalone forward-pass bring-up driver |
| src/arch.rs | 227 | Architecture trait triple plus serving backend |
| src/calibration.rs | 361 | Text-only calibration artifact collection with capture hooks |
| src/config.rs | 297 | Config type with embed/attn/query scaling helpers |
| src/forward.rs | 566 | Per-decode GPU scratch, F32 KV cache, per-token forward |
| src/lib.rs | 39 | Crate root for the Gemma3 text architecture |
| src/weights.rs | 644 | GPU-resident layer weights and HFQ loader |

## hipfire-arch-gemma3-vl
Gemma3 multimodal (arch_id 13): SigLIP vision tower plus multi-modal projector into the text decoder.

| file | LOC | description |
| --- | --- | --- |
| examples/infer_gemma3_vl.rs | 238 | Multimodal bring-up inference driver |
| src/arch.rs | 488 | Multimodal serving backend with image encoding and embed serving |
| src/config.rs | 192 | SigLIP and combined vision+text config types |
| src/forward.rs | 303 | SigLIP ViT forward over patch tensor |
| src/image.rs | 137 | Image decode into SigLIP patch tensor |
| src/lib.rs | 30 | Crate root for the Gemma3 multimodal model |
| src/loader.rs | 64 | Full multimodal weight loader (text + vision + projector) |
| src/projector.rs | 170 | Multi-modal projector mapping vision hidden to text space |
| src/vision.rs | 229 | GPU-resident SigLIP weights and loader |

## hipfire-arch-lfm2moe
LFM2.5-8B-A1B hybrid architecture (arch_id 11): interleaved conv/attention mixers with MoE and DFlash draft bridge.

| file | LOC | description |
| --- | --- | --- |
| examples/dump_lfm2moe_hidden_states.rs | 136 | Dumps per-layer hidden states for offline analysis |
| examples/graph_parity_lfm2moe.rs | 128 | hipGraph parity check against eager execution |
| examples/infer_lfm2moe.rs | 148 | Minimal greedy end-to-end coherence check |
| examples/kld_logits.rs | 281 | Per-position KL divergence between two models |
| examples/prefill_parity_lfm2moe.rs | 252 | Batched-prefill parity check |
| examples/seam_generate.rs | 118 | Greedy generation via the SimpleAr seam |
| examples/lfm2_dflash_*.rs (7 files) | ~4670 | DFlash sidecar training/eval cluster: teacher dump, fit fc/down/norm, acceptance/block-teacher eval, seed smoke |
| src/arch.rs | 143 | SimpleAr + ServingBackend serving seam |
| src/calibration.rs | 232 | Calibration artifact collection with capture backend |
| src/config.rs | 260 | Config with mixer-kind enum, parsed from HFQ metadata |
| src/dflash.rs | 873 | DFlash target/draft bridge for speculative decode |
| src/forward.rs | 2070 | Forward pass free functions plus hidden-state capture |
| src/kld.rs | 71 | KLD-scoring seam over loose resident weights |
| src/lfm2moe.rs | 1078 | Weights, mixer enum, conv/attn/FFN types, decode state |
| src/lib.rs | 39 | Crate root documenting the hybrid model |

## hipfire-arch-llama
LLaMA / Mistral / plain-Qwen3 dense text architecture.

| file | LOC | description |
| --- | --- | --- |
| src/arch.rs | 504 | Architecture trait impl, scratch-layer forward, serving backend |
| src/lib.rs | 60 | Crate root for the LLaMA-family architecture |

## hipfire-arch-minimax
MiniMax-M2 Mixtral-style MoE architecture (arch_id 10) with expert-parallel decode.

| file | LOC | description |
| --- | --- | --- |
| examples/dump_minimax_hidden_states.rs | 146 | Dumps per-layer hidden states for offline analysis |
| examples/ep_minimax.rs | 232 | Expert-parallel greedy decode across N GPUs (Ship 6) |
| examples/infer_minimax.rs | 122 | Minimal greedy end-to-end coherence check |
| src/arch.rs | 65 | Architecture trait impl for MiniMax-M2 |
| src/calibration.rs | 231 | Calibration artifact collection with capture backend |
| src/forward.rs | 1773 | Forward pass: decode step, capture, hipGraph, batch prefill |
| src/kld.rs | 70 | KLD-scoring seam over loose resident weights |
| src/lib.rs | 37 | Crate root for the MiniMax-M2 architecture |
| src/minimax.rs | 1199 | Config/weights/state with q/kv dimension helpers |

## hipfire-arch-nemotron
nemotron_h flat-pattern architecture (arch_id 14): Mamba-2 SSD mixer, GQA NoPE attention, ReLU² MLP/MoE blocks.

| file | LOC | description |
| --- | --- | --- |
| examples/test_*_gpu.rs + bench/bisect/probe (24 files) | ~2540 | GPU-vs-CPU correctness, prefill equivalence, HFQ-vs-f32, and throughput benches for each block |
| examples/bench_ssd_chunk_wmma.rs | 138 | SSD bf16-WMMA chunked prefill throughput bench |
| examples/bisect_nano4b.rs | 107 | HF-reference numeric bisect for Nano-4B |
| examples/gemv_probe.rs | 114 | FU4 single quantized-gemv isolation probe |
| examples/test_ssd_chunk_wmma_gpu.rs | 140 | Correctness for SSD chunked bf16-WMMA prefill kernel |
| src/arch.rs | 88 | SimpleAr + ServingBackend serving seam |
| src/attn.rs | 314 | NoPE GQA attention block: decode, KV cache, GPU forward |
| src/block.rs | 308 | Mamba-2 mixer block decode CPU reference oracle |
| src/block_gpu.rs | 551 | Mamba-2 mixer block GPU decode with optional Q8 state |
| src/calibration.rs | 147 | Full Hessian + imatrix calibration sidecar generation |
| src/kld.rs | 108 | KLD-eval seam: chunk loop, top-k reference, scoring |
| src/lib.rs | 770 | Crate root with block-pattern enum and parser |
| src/loader.rs | 305 | BF16 safetensors and HFQ linear weight loader |
| src/mlp.rs | 167 | ReLU² dense MLP block CPU oracle and GPU forward |
| src/model.rs | 772 | Full model decode forward composing the three block types |
| src/moe.rs | 333 | Routed ReLU² MoE block with GPU forward |
| src/ssd.rs | 415 | Mamba-2 SSD recurrence CPU reference (decode/sequence/chunked) |
| src/weight.rs | 164 | Linear weight enum abstracting F32 vs quantized gemv/gemm |

## hipfire-arch-qwen2
Plain Qwen2 dense text decoder.

| file | LOC | description |
| --- | --- | --- |
| examples/infer_qwen2.rs | 340 | Standalone forward-pass driver |
| examples/inspect_hfq.rs | 76 | Opens a Qwen2 HFQ and parses its config |
| src/arch.rs | 206 | Architecture trait impl plus serving backend |
| src/lib.rs | 78 | Crate root for the Qwen2 dense decoder |
| src/qwen2.rs | 2115 | Config/Weights/State types with HFQ and metadata parsing |

## hipfire-arch-qwen35
Qwen3.5 architecture (dense + MoE): hybrid DeltaNet linear attention, native MTP, speculative/PFlash decode, XDNA1 NPU offload.

| file | LOC | description |
| --- | --- | --- |
| build.rs | 608 | Build script (kernel/codegen setup) |
| examples/mtp_head_smoke.rs | 281 | Minimal end-to-end native MTP head smoke test |
| examples/test_qwen35_load_multi.rs | 107 | Stage-4 multi-GPU weight-load smoke |
| examples/test_qwen35_state_multi.rs | 237 | Stage-4 multi-GPU state-stack smoke |
| src/arch.rs | 88 | Architecture trait implementation for Qwen3.5 |
| src/ffn_bf16.rs | 440 | CPU BF16 oracle for dense SwiGLU/down FFN epilogue |
| src/lib.rs | 55 | Crate root for the Qwen3.5 architecture |
| src/mtp_compose.rs | 1377 | DFlash + MTP linear-chain composition state |
| src/mtp_head.rs | 2683 | Native MTP (NextN) head config and dense/MoE weights |
| src/mtp_probe.rs | 485 | Training-free Qualcomm-style MTP probe |
| src/mtp_spec.rs | 3990 | MTP-only speculative decode: trunk plus native head loop |
| src/paro_la_gates_codec.rs | 299 | MQ4G128 codec for linear-attention gate projections |
| src/pflash.rs | 2179 | Speculative prefill compression for long-context inputs |
| src/qwen35.rs | 32648 | Core model: hybrid DeltaNet + attention, MoE router histograms |
| src/speculative.rs | 12697 | Speculative decoding infrastructure and seed-oracle stats |
| src/xdna1_ffi.rs | 1092 | dlopen FFI bindings to libhipfire_xdna1.so |
| tests/pp_parity.rs | 243 | Env-gated multi-GPU pipeline-parity test |

## hipfire-arch-qwen35-vl
Qwen3.5-VL vision-language architecture: SigLIP-2 ViT vision encoder plus spatial merger.

| file | LOC | description |
| --- | --- | --- |
| src/arch.rs | 92 | Architecture trait impl for the vision tower |
| src/image.rs | 350 | Image load/preprocess: smart-resize and patch extraction |
| src/lib.rs | 39 | Crate root for the Qwen3.5-VL model |
| src/qwen35_vl.rs | 976 | Vision encoder: SigLIP-2 ViT, config, weights, spatial merger |
| tests/channel_order.rs | 126 | Regression test for vision-encoder CHW channel ordering |
| tests/image_from_bytes.rs | 248 | Tests for byte-decode path and decompression-bomb guard |

## hipfire-arch-toy
Reference template for new arch crates (minimum-viable Architecture impl).

| file | LOC | description |
| --- | --- | --- |
| src/arch.rs | 136 | Minimum-viable Architecture trait reference impl |
| src/lib.rs | 20 | Crate root for the toy reference architecture |
| src/toy_model.rs | 84 | Stub config and weights model types |

## hipfire-arch-zaya
Zyphra ZAYA1 uniform-layer architecture (arch_id 16): CCA attention, MoE experts, residual scaling, with CPU/GPU forwards.

| file | LOC | description |
| --- | --- | --- |
| examples/bench.rs | 84 | Prefill latency and greedy decode throughput micro-bench |
| examples/cpu_golden.rs | 165 | Validates CPU reference forward against golden dump |
| examples/eval.rs | 82 | Coherence eval across diverse greedy-completion prompts |
| examples/generate.rs | 70 | End-to-end greedy generation via SimpleAr seam |
| examples/gpu_golden.rs | 135 | Validates GPU prefill forward against golden dump |
| examples/kld.rs | 175 | KLD A/B teacher-forcing bf16 reference vs quantized |
| examples/tok_time.rs | 41 | Per-token timing micro-benchmark |
| src/arch.rs | 134 | SimpleAr + ServingBackend serving seam |
| src/calibration.rs | 155 | LDLQ full-Hessian + imatrix calibration sidecar generation |
| src/cpu.rs | 565 | CPU reference forward (f32) porting upstream math |
| src/gpu.rs | 1793 | GPU forward loading per-quant linears, with weight enum |
| src/lib.rs | 532 | Crate root with layer-kind, attn-window, config types |
| src/weights.rs | 257 | Host f32 weight loader for native converted checkpoint |

# Architectural Source Index — diffusion & quant crates

## hipfire-diffusion
Native HFQ-backed diffusion serving: UNet/transformer/VAE denoisers, schedulers, GPU ops, and weight quantization.

| file | LOC | description |
|------|-----|-------------|
| src/lib.rs | 10349 | Pipeline orchestration, HFQ metadata, batch generation, CLIP encoder, memory planning |
| src/gpu_ops.rs | 3481 | ROCm/HIP GPU boundary ops with CPU reference fallback for diffusion primitives |
| src/hip_kernels.rs | 1236 | HIP device source strings for diffusion GPU ops, rocm-feature only |
| src/layers.rs | 1069 | Shared NN blocks: conv, groupnorm, resnet, time embeddings, attention |
| src/transformer.rs | 3523 | Native transformer denoiser (Qwen-Image/Flux/Krea): IO projection, modulation, attention |
| src/unet.rs | 1288 | UNet2DConditionModel assembly: Transformer2DModel and down/up/mid block paths |
| src/vae.rs | 1364 | VAE encoder/decoder blocks, moments sampling, latent normalization, image conversion |
| src/scheduler.rs | 1225 | Sampling schedules: beta/sigma construction and Euler/DDIM/flow-match/DPM solvers |
| src/tokenizer.rs | 177 | CLIP byte-level BPE tokenizer for text conditioning, HFQ loading |
| src/quant_encode.rs | 513 | Tensor payload encoders plus post-process quantizer for diffusion .hfq artifacts |
| src/quant_decode.rs | 254 | Dequantize HFQ tensor formats (f16, bf16, Q4/Q8, Q4_K, HFQ4/6) to f32 |
| src/quant_calib.rs | 256 | Activation calibration for diffusion weight quantization, sidecar writing |
| src/tests.rs | 13491 | Extensive unit/integration tests for pipeline, denoise backends, and metadata |

## hipfire-quantize
Offline model quantizer CLI plus codecs (MQ/OQ/GPTQ/QTIP/LDLQ), Hessian IO, and HF/GGUF import binaries.

| file | LOC | description |
|------|-----|-------------|
| src/main.rs | 13875 | Quantizer CLI entrypoint driving import, calibration, and codec dispatch |
| src/codecs.rs | 2243 | Pure quantization codecs mapping f32 weights to HFQ/MQ wire formats |
| src/gptq.rs | 1659 | GPTQ column-sequential quantization for MQ4G256 with clip-search and AWQ |
| src/fixture.rs | 1578 | Emits tiny random-init HF-format models for fast kernel/plumbing gating |
| src/qtip.rs | 1103 | QTIP trellis-coded quantization encoder: codebook build and beam encode |
| src/ldlq.rs | 883 | QTIP-LDLQ output-aware Hessian trellis quantization and OQ pack routines |
| src/hessian_io.rs | 806 | Reader for per-tensor Hessians from unified .calib.hfq (HFQM) packages |
| src/gguf_input.rs | 522 | Minimal GGUF reader and dequant for import into hipfire formats |
| src/hfhs_diag.rs | 242 | Reader for retired HFHS-v1 standalone Hessian sidecar diagonals |
| src/roughquant.rs | 75 | RoughQuant PCA rotation into the activation-Hessian eigenbasis |
| src/lib.rs | 52 | Library surface exposing clip-search toggle for the quantizer |
| src/bin/dflash_convert.rs | 687 | Convert HuggingFace DFlash draft safetensors plus config into HFQ |
| src/bin/mtp_extract.rs | 1298 | Extract Qwen3.5/3.6 dense MTP head from safetensors into .hfq |
| src/bin/draft_to_mq4.rs | 244 | Convert a draft model to MQ4-quantized HFQ artifact |
| src/bin/mq4_merge_mtp.rs | 169 | Bundle trunk MQ4 HFQ with MTP sidecar into single file |
| examples/quant_opus_mqplus.rs | 1110 | Head-to-head MQ4 vs MQ+ vs Opus Quant benchmark on gfx1103 |
| examples/quant_w4a4_improve.rs | 392 | W4A4 quality-pushing study stacking iu4-preserving quant techniques |
| examples/quant_explore.rs | 297 | First-principles quantization-scheme exploration for gfx1103 |
| examples/quant_wxax_explore.rs | 197 | Study of activation precision for the fused-iu4 path |
| examples/deepseek4_dequant_check.rs | 157 | Verify FP8 E4M3 plus UE8M0-scale dequantizer against DeepSeek V4 |

## hipfire-kvquant
Offline KV-cache quantization: variance-normalized 4-bit KVarN codec and deferred cold-tier compaction.

| file | LOC | description |
|------|-----|-------------|
| src/kvarn.rs | 483 | KVarN variance-normalized 4-bit KV-cache quantization tile encode/decode |
| src/kv_compact.rs | 501 | Deferred-hierarchical cold-tier KV compression producer and two-tier attend |
| src/lib.rs | 8 | Leaf-lib re-exports of KVarN codec and KV-compaction |

## hipfire-quant-format
Canonical on-disk HFQ quant_type byte-contract enum shared across crates.

| file | LOC | description |
|------|-----|-------------|
| src/lib.rs | 193 | QuantType enum defining the on-disk quant_type byte codes |

## hipfire-kld
Pure, GPU-independent KLD scoring core: reference archives, codecs, and reduction math.

| file | LOC | description |
|------|-----|-------------|
| src/archive.rs | 340 | HFKREF persistable KLD reference archive encode/decode and file IO |
| src/meta.rs | 297 | Self-describing reference metadata with compatibility mismatch check |
| src/eval.rs | 260 | Arch-independent chunk loop, top-k reduction, and KLD-vs-reference scoring |
| src/math.rs | 237 | Pure fp64 reduction and KLD scoring math, top-k log-softmax |
| src/config.rs | 191 | KldConfig env contract for reference build and candidate scoring |
| src/codec.rs | 151 | Per-blob storage codecs and bit-packing for .kldref payloads |
| src/hfkseq.rs | 123 | HFKSEQ per-sequence KLD result file encode/decode |
| src/refblock.rs | 97 | Top-K reference-distribution block bytes serialization |
| src/lib.rs | 46 | Crate surface for the pure KLD scoring core |

## hipfire-lora-hfq
Binary .lora.hfq container: serialize, read, and merge LoRA rank-1 residual adapters.

| file | LOC | description |
|------|-----|-------------|
| src/lib.rs | 349 | Write/read/merge .lora.hfq adapter containers and bundled-LoRA detection |

# Serving-layer source index

## hipfire-server
HTTP API surface (OpenAI-compatible chat/responses, SD API, admin console) fronting the daemon engine.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 692 | Crate root wiring modules and building the axum router/app. |
| src/admin_ui.rs | 62 | Serves embedded Leptos admin UI dist behind feature gate. |
| src/auth.rs | 207 | Admin-console auth via local bearer secret or argon2id browser session. |
| src/scheduler.rs | 94 | Bridges server state to accelerator inventory and prefill scheduler policy. |
| src/state.rs | 194 | Shared server state: engine handle, config, diffusion runtime defaults. |
| src/telemetry.rs | 9 | Re-exports host GPU telemetry reader from hipfire-sysinfo. |
| src/model/mod.rs | 1 | Module declaration for model discovery. |
| src/model/discovery.rs | 29 | Resolves model identifiers to local registry file paths. |
| src/routes/mod.rs | 10 | Declares route submodules. |
| src/routes/admin.rs | 1480 | Admin endpoints: diagnostics, logs, stats, config schema/resolution. |
| src/routes/batches.rs | 809 | OpenAI batches API: create, poll, cancel batch jobs. |
| src/routes/chat.rs | 3008 | Chat-completions endpoint with streaming SSE and daemon dispatch. |
| src/routes/chat_ui.rs | 79 | Serves embedded Leptos chat UI dist behind feature gate. |
| src/routes/files.rs | 211 | OpenAI files API: multipart upload, retrieve, delete. |
| src/routes/health.rs | 294 | Health/readiness endpoints aggregating scheduler and worker health. |
| src/routes/models.rs | 116 | Lists available models from the local registry. |
| src/routes/responses.rs | 809 | OpenAI Responses API endpoint with SSE streaming. |
| src/routes/sdapi.rs | 8249 | Automatic1111-compatible Stable Diffusion API: txt2img, img2img, extras, png-info. |
| src/routes/training.rs | 160 | Endpoints listing training runs and streaming run events. |

## hipfire-serving-core
Shared model-serving orchestration: load/unload, per-arch generate paths, sessions, and IPC event emission.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 34 | Crate root exporting serving orchestration modules. |
| src/dummy.rs | 258 | GPU-free token-counter backend exercising daemon protocol in tests. |
| src/events.rs | 177 | JSONL stream-event emitters for the daemon IPC protocol. |
| src/evidence.rs | 141 | Per-generation evidence writers for timings and MoE router histograms. |
| src/generate.rs | 3678 | Default Qwen3.5/llama autoregressive text decode path. |
| src/generate_arch.rs | 2988 | Per-arch text generate paths for non-qwen35 model families. |
| src/generate_vl.rs | 1290 | Vision-language generate: image plus prompt through encoder to decode. |
| src/lfm2_prefill.rs | 699 | LFM2 batch-prefill support for the generate path. |
| src/load.rs | 3401 | Loads/unloads HFQ or safetensors artifacts into a runtime model. |
| src/memory.rs | 254 | VRAM/host memory accounting for a loaded model's scratch/KV/state. |
| src/model.rs | 476 | Daemon's in-memory model representation and satellite structures. |
| src/output_filter.rs | 228 | Prompt normalization, stop filtering, and sampling-guard attractor blockers. |
| src/qwen35_decode.rs | 1350 | Qwen3.5 per-token/batched decode kernels and capability validators. |
| src/qwen35_prefill.rs | 1933 | Qwen3.5 prefill turning prompt batches into resident KV/DeltaNet state. |
| src/request.rs | 40 | Request-shape types parsed from the JSONL protocol. |
| src/session.rs | 2435 | Per-request session state and model-worker lifecycle management. |
| src/tiny_harness.rs | 636 | Tokenizer-free tiny-model probe harness for KLD and calibration. |
| examples/tiny_quant_probe.rs | 124 | Example CLI driving tiny_harness for tiny_quant eval battery. |

## hipfire-daemon
The engine daemon process communicating with the server via JSONL over stdin/stdout.

| file | LOC | description |
| --- | --- | --- |
| src/main.rs | 6434 | Daemon entrypoint dispatching JSONL IPC commands to serving-core. |

## hipfire-daemon-protocol
Data contracts defining the daemon JSONL wire protocol.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 557 | Daemon JSONL protocol request/response contract types. |

## hipfire-daemon-adapter
Async host-side adapter spawning and driving the daemon subprocess.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 1387 | Async daemon subprocess adapter over the JSONL protocol. |

## hipfire-operator
Lightweight serde/filesystem API models for operator HTTP routes and training runs.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 8 | Crate root exposing shared operator API models. |
| src/training.rs | 596 | Training-run listing, detail loading, and event reading from filesystem. |

## hipfire-admin-types
Pure serde data types shared between the server and the WASM admin console.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 394 | Shared serde admin/telemetry types for native and wasm consumers. |

# Core Libraries Source Index

## hipfire-model
Shared model artifact identity, GGUF parsing, and tokenizer loading.

| file | LOC | description |
|------|-----|-------------|
| src/gguf.rs | 335 | Zero-copy memory-mapped GGUF header, metadata, and tensor-info parser. |
| src/lib.rs | 3332 | Model artifact identity, manifest/HFQ metadata contracts, and arch-family classification. |
| src/model_support_generated.rs | 244 | Generated per-arch capability, quant, and gate tables from model-support.toml. |
| src/tokenizer.rs | 2327 | BPE tokenizer loading from GGUF/HF JSON with encode and decode. |

## hipfire-generate
Typed generation request/event contracts and stream-level output guards.

| file | LOC | description |
|------|-----|-------------|
| src/loop_guard.rs | 183 | Per-request n-gram loop detector that force-stops pathological repetition. |
| src/sampler.rs | 155 | Pure sampler policy knobs and token-history guard helpers. |
| src/eos_filter.rs | 590 | Filters decoded byte stream for hold-back, tag-strip, EOT suppression. |
| src/lib.rs | 2522 | Generation request, sampling-policy, event, and batch-plan contracts. |

## hipfire-prompt
ChatML prompt framing and text normalization single source of truth.

| file | LOC | description |
|------|-----|-------------|
| src/lib.rs | 2378 | Assembles ChatML token sequences and normalizes prompt text plus chat templates. |

## hipfire-state
Shared sequence-state handles, descriptors, and reservation helpers.

| file | LOC | description |
|------|-----|-------------|
| src/lib.rs | 2592 | Sequence-state arenas, reservations, handles, and per-worker runtime views. |

## hipfire-scheduler
Priority scheduling and session batching policy shared by control planes.

| file | LOC | description |
|------|-----|-------------|
| src/lib.rs | 1850 | Priority classes, dispatch policy, prefill controls, and request-session drafts. |

## hipfire-config
Shared CLI/server configuration, layered resolution, and schema metadata.

| file | LOC | description |
|------|-----|-------------|
| src/resolve.rs | 403 | Layered config resolution tracking value provenance across sources. |
| src/lib.rs | 907 | Configuration model, per-model resolution, diagnostics, and filesystem paths. |
| src/schema.rs | 597 | Config field schema: scope, mutability, restart impact, and type metadata. |

## hipfire-detect
Coherence-gate detector bank over generation token/text streams.

| file | LOC | description |
|------|-----|-------------|
| src/think.rs | 274 | Detects empty or stalled `<think>` blocks on visible text. |
| src/attractor.rs | 368 | Path-A token-attractor detectors over first/last committed-token windows. |
| src/eos_immediate.rs | 121 | Hard-fails when generation emitted zero visible bytes before stopping. |
| src/ngram.rs | 285 | Committed-token n-gram density and loop-guard mirror signals. |
| src/self_check.rs | 591 | Self-check against detector rot via synthetic and replay phases. |
| src/special_leak.rs | 132 | Hard-fails when ChatML special-token literals leak into visible text. |
| src/timing.rs | 153 | Opt-in per-token step-time spike detector over rolling median. |
| src/toolcall.rs | 179 | Validates tool-call block shape: stacking, JSON, and schema checks. |
| src/whitespace_only.rs | 108 | Hard-fails when visible output is entirely whitespace. |
| src/report.rs | 250 | Markdown and JSON report renderers with prompt md5 pinning. |
| src/lib.rs | 271 | Detector trait, event, severity, and verdict core types. |
| src/parity.rs | 170 | Compares AR baseline versus DFlash committed-token streams for equality. |
| src/rollback.rs | 235 | Parses DFlash rollback-replay and verify-graph stat summaries into verdicts. |
| tests/replay.rs | 125 | Phase-B JSONL replay running detector banks against captured streams. |

## hipfire-sysinfo
Portable HIP-free host/GPU/NPU memory and telemetry for admin surfaces.

| file | LOC | description |
|------|-----|-------------|
| examples/mem.rs | 128 | Example one-shot memory snapshot rendering the way UI consumers do. |
| src/fdinfo.rs | 217 | Per-process GPU memory attribution from `/proc/<pid>/fdinfo` amdgpu clients. |
| src/gpu.rs | 201 | Per-GPU VRAM/GTT telemetry scanned from DRM sysfs. |
| src/gpu_metrics.rs | 180 | Parses versioned firmware `gpu_metrics` binary table for power/temp/bandwidth. |
| src/host.rs | 83 | Host memory pressure from `/proc/meminfo` total and available. |
| src/lib.rs | 97 | Portable telemetry snapshot aggregating host/GPU sources without HIP. |
| src/npu.rs | 42 | NPU telemetry wrapping XDNA device queries into wasm-safe shape. |

## hipfire-hash
Stable hashing primitives for model identity and evidence contracts.

| file | LOC | description |
|------|-----|-------------|
| src/lib.rs | 93 | Stable file/byte hashing and score helpers shared across workspace. |

## hipfire-primitives
Dependency-free pure-Rust numeric building blocks for the workspace.

| file | LOC | description |
|------|-----|-------------|
| src/conv.rs | 206 | IEEE half-float and bfloat16 bit-level conversion routines. |
| src/fwht.rs | 78 | Per-256 signed fast Walsh-Hadamard transform and sign generation. |
| src/lib.rs | 19 | Crate root re-exporting float-conversion and FWHT primitives. |

## hipfire-lock
Reusable `flock(2)` file-lock primitive for all hipfire mutexes.

| file | LOC | description |
|------|-----|-------------|
| src/lib.rs | 326 | FlockGuard exclusive locking with holder metadata and lock-state queries. |

## hipfire-rocm
Typed ROCm backend contracts and module evidence adapters.

| file | LOC | description |
|------|-----|-------------|
| src/lib.rs | 283 | Device identity, module-kind, and backend-path descriptors for ROCm evidence. |

## hipfire-build-info
Git-derived `--version` build identity for hipfire binaries.

| file | LOC | description |
|------|-----|-------------|
| build.rs | 40 | Emits git-describe version env var via vergen-gitcl at build. |
| src/lib.rs | 15 | Exposes build-time git-derived VERSION string with fallback. |

## hipfire-coherence
Shared coherence detector policy and report serialization helpers.

| file | LOC | description |
|------|-----|-------------|
| src/lib.rs | 638 | Detector profiles, coherence run config/output, and hard-fail aggregation. |

## hipfire-cpu
Deterministic CPU oracle backends and module evidence contracts.

| file | LOC | description |
|------|-----|-------------|
| src/lib.rs | 894 | CPU FFN backend selection, diff stats, and reference oracle contracts. |

## hipfire-vision-cache
Content-addressed on-disk LRU cache for projected vision embeddings.

| file | LOC | description |
|------|-----|-------------|
| src/lib.rs | 713 | Cache keys, cached-embedding entries, stats, and LRU eviction for vision rows. |

## hipfire-media
Video and multi-frame decode preprocessing for the vision path.

| file | LOC | description |
|------|-----|-------------|
| src/lib.rs | 19 | Crate root exposing video frame-decode preprocessing entry points. |
| src/video.rs | 321 | ffmpeg-based frame extraction, uniform sampling, and PNG-bytes output. |

## hipfire-mixer
Per-layer token-mixer taxonomy and sequence-state shape for the family seam.

| file | LOC | description |
|------|-----|-------------|
| src/lib.rs | 249 | MixerKind taxonomy and MixerProfile describing KV/recurrent/conv state shape. |

# Architectural Source Index — Tooling & Eval Crates

## hipfire-cli
The user-facing `hipfire` CLI: subcommand dispatch plus hidden doc/schema generators.

| file | LOC | description |
| --- | --- | --- |
| src/main.rs | 199 | Top-level clap `Cli`/`Command` enum and dispatch entry point. |
| src/model.rs | 10 | Thin helper to locate a model artifact on disk. |
| src/commands/mod.rs | 12 | Module aggregator re-exporting command submodules. |
| src/commands/admin.rs | 369 | `admin` subcommand: daemon lifecycle and administrative actions. |
| src/commands/chat.rs | 357 | Interactive `chat` subcommand streaming a conversation. |
| src/commands/detect.rs | 300 | `detect` host/hardware capabilities with rollback source handling. |
| src/commands/diffusion.rs | 1724 | Diffusion pipeline commands: calibrate, quantize, generate. |
| src/commands/forward.rs | 450 | Forwards eval/host-profile/collect/repack to resolved runner binaries. |
| src/commands/gen_config_schema.rs | 235 | Hidden: render shared config schema as JSON/other format. |
| src/commands/gen_docs.rs | 130 | Hidden: render clap definitions into CLI documentation. |
| src/commands/gen_env_docs.rs | 800 | Hidden: scan source tree to document environment variables. |
| src/commands/gen_model_support.rs | 627 | Hidden: render model-support/quant matrix from registry. |
| src/commands/list.rs | 209 | `list` local models as capability/artifact matrix. |
| src/commands/lock.rs | 300 | `lock {acquire,release,status}` flock(2) GPU resource lock. |
| src/commands/serve.rs | 38 | `serve` subcommand launching the inference server. |

## hipfire-tui
Terminal UI front-end for browsing models, status, config, chat, and training.

| file | LOC | description |
| --- | --- | --- |
| src/main.rs | 123 | TUI binary entry: terminal setup and event loop. |
| src/app.rs | 475 | Central `App` state, tab model, and control actions. |
| src/ui.rs | 1132 | Ratatui rendering: draws all tabs and widgets. |
| src/hipfire/mod.rs | 58 | Path discovery and admin authorization helpers. |
| src/hipfire/chat.rs | 117 | Chat message model and streaming event source. |
| src/hipfire/config.rs | 278 | Config load, host probing, editable row generation. |
| src/hipfire/registry.rs | 483 | Model/sidecar registry state and displayed rows. |
| src/hipfire/status.rs | 394 | Daemon status polling and background serve control. |
| src/hipfire/training.rs | 123 | Training run state and active-run tracking. |

## hipfire-train
fp32 training/fine-tuning path with forward+backward op twins, gradchecks, and quant-recovery experiments.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 33 | Crate root wiring the training modules together. |
| src/block.rs | 570 | Pre-norm LLaMA transformer block with LoRA on q/v. |
| src/checkpoint.rs | 239 | Save/resume labels and drafter weights. |
| src/config.rs | 106 | LLaMA config parsing from dir/hfq/JSON. |
| src/drafter.rs | 591 | PFlash importance-scorer drafter scaffold. |
| src/hfq_patch.rs | 161 | Minimal .hfq reader and in-place norm patcher. |
| src/kv_noise.rs | 129 | KV-compression sim-noise (KVarN-4bit + CASK merge). |
| src/labels.rs | 139 | PFlash drafter label IO and shuffling. |
| src/loader.rs | 239 | Load dense LLaMA safetensors/hfq into fp32 GPU tensors. |
| src/model.rs | 765 | Full dense fp32 LLaMA model for training. |
| src/optim.rs | 135 | AdamW optimizer with decoupled weight decay. |
| src/oqplus_quant.rs | 101 | OQ+ (W4A8) sim-quant for recovery probes. |
| src/qtip_quant.rs | 169 | QTIP quantize→dequant codebook builder. |
| src/ssm_block.rs | 291 | GLA-lite selective-SSM block (fwd/bwd/grad/activations). |
| src/ssm_drafter.rs | 461 | SSM-body PFlash importance-scorer drafter. |
| src/tensor.rs | 58 | fp32 GPU tensor with optional gradient buffer. |
| src/train_loop.rs | 258 | Shared drafter training loop, ListNet loss + AdamW. |
| src/ops/mod.rs | 20 | Aggregator for forward+backward op pairs. |
| src/ops/attention.rs | 183 | Single-head SDPA and GQA forward/backward. |
| src/ops/cross_entropy.rs | 24 | Fused cross-entropy with ignore_index masking. |
| src/ops/distill.rs | 19 | KL distillation soft-target loss. |
| src/ops/gated_scan.rs | 40 | Gated linear-recurrence scan forward/backward. |
| src/ops/linear.rs | 62 | Linear op with input/weight gradients. |
| src/ops/lora.rs | 83 | LoRA-adapted linear forward/backward. |
| src/ops/pflash_score.rs | 44 | PFlash cosine-importance head forward/backward. |
| src/ops/rmsnorm.rs | 39 | RMSNorm forward/backward. |
| src/ops/rope.rs | 32 | RoPE half-split forward/backward. |
| src/ops/sigmoid.rs | 26 | Elementwise sigmoid forward/backward. |
| src/ops/softmax.rs | 30 | Row-softmax forward/backward. |
| src/ops/swiglu.rs | 27 | SwiGLU elementwise forward/backward. |
| examples/ (gradcheck cluster, 20 files) | ~2160 | Finite-difference gradient checks per op and end-to-end. |
| examples/gradcheck_model.rs | 167 | End-to-end finite-difference gradcheck for the full model. |
| examples/gradcheck_block.rs | 173 | End-to-end gradcheck for one transformer block. |
| examples/gradcheck_ssm_block.rs | 214 | Composed fwd+bwd gradcheck for GLA-lite SSM block. |
| examples/pflash_drafter_train.rs | 448 | Train PFlash drafter to reproduce mid-layer block. |
| examples/pflash_scaling_probe.rs | 403 | PFlash scaling-trend probe over dense-Llama ladder. |
| examples/ (recovery/probe cluster, 17 files) | ~2400 | Supra-50M quant-damage, OQ+/QTIP/KV-noise recovery and export probes. |

## hipfire-steer
Refusal-direction steering / abliteration: capture, derive, apply, score, LoRA export.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 878 | Steering core: modes, spec, capture means, direction derivation. |
| src/driver.rs | 481 | Phase 2/3 orchestration and refusal scoring harness. |
| src/lora.rs | 359 | Rank-1 residual LoRA adapter read/write for directions. |
| examples/gpu_validate.rs | 192 | GPU validation harness for the on-GPU apply path. |

## hipfire-steer-harness
Daemon-backed adapter and CLI driving the steer pipeline through hipfire-daemon.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 333 | DaemonHarness: connect, load LoRA, set scale over daemon. |
| src/main.rs | 314 | CLI running capture→derive→apply against daemon subprocess. |

## hipfire-atlas
Kernel benchmark/profile result parsing, schema, scoring, and rendering.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 31 | Crate root aggregating atlas modules. |
| src/main.rs | 269 | Atlas CLI entry: run/parse and emit reports. |
| src/eval.rs | 68 | Execute task files and collect command results. |
| src/parse.rs | 282 | Parse bench/dflash/profile summary sections. |
| src/profile_report.rs | 108 | Owned rocprofv3 coverage report data types. |
| src/render.rs | 74 | Render the atlas fit-view output. |
| src/schema.rs | 158 | AtlasRow schema with metric setters. |
| src/suggest.rs | 96 | Generate per-row tuning suggestions and markdown. |
| src/task.rs | 92 | Build task bundles from rows (incl. PyTorch). |

## hipfire-coexistence
Offline import/export/conversion/interop tooling kept out of inference binaries.

| file | LOC | description |
| --- | --- | --- |
| src/main.rs | 209 | Coexistence CLI dispatch for format conversion/interop. |

## hipfire-npu
NPU accelerator inventory and XDNA module artifact modeling.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 462 | NPU module targets and XDNA module artifact management. |
| examples/npu_inventory.rs | 22 | Print NPU inventory from live config+hardware probe. |

## hipfire-xdna
XDNA NPU sensor/utilization telemetry.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 391 | XDNA errors, NPU sensors, utilization and resource info. |
| examples/npu_busy.rs | 73 | amdgpu_top-style per-column NPU utilization monitor. |

## hipfire-hneurons
H-Neuron identification core (CETT features) per arXiv 2512.01797.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 329 | CETT feature accumulation and H-neuron identification. |

## redline
Direct-KMD GPU compute engine for AMD RDNA via bare libdrm_amdgpu/KFD.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 73 | Crate root and RedlineError type. |
| src/device.rs | 288 | GPU device init, info queries, VRAM allocation. |
| src/dispatch.rs | 617 | High-level compute dispatch, PM4 build, kernarg layout. |
| src/drm.rs | 323 | dlopen FFI bindings to libdrm_amdgpu.so. |
| src/hsaco.rs | 283 | Parse .hsaco ELF code objects and kernel metadata. |
| src/kfd.rs | 638 | KFD user-mode AQL compute queue interface. |
| src/pm4.rs | 162 | PM4 command-buffer builder for GFX10 dispatch. |
| src/queue.rs | 200 | Submit PM4 buffers and wait on the compute queue. |
| examples/ (PoC/test cluster, 16 files) | ~3200 | Bare-libdrm bring-up PoCs and dispatch/submit tests. |
| examples/poc_dispatch_raw.rs | 596 | Hand-assembled s_endpgm shader dispatch bisection. |
| examples/bench_dispatch.rs | 344 | Benchmark Redline dispatch overhead. |
| examples/poc_gemm.rs | 269 | GEMM matrix-multiply via bare libdrm_amdgpu. |
| examples/test_chain_dispatch.rs | 244 | Chain multiple dispatches in one IB submission. |

## hipfire-eval
Model/runtime evaluation batteries: config, datasets, executors, evidence, profiling.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 8117 | Eval core: tiers, battery orchestration, scoring surface. |
| src/main.rs | 10 | Binary entry delegating to run_from_env. |
| src/config.rs | 574 | Argument parsing, usage, version report, default batteries. |
| src/datasets.rs | 998 | Built-in evaluation dataset definitions and loaders. |
| src/driver.rs | 943 | Battery execution driver coordinating executors. |
| src/evidence.rs | 911 | Emit structured evaluation evidence artifacts. |
| src/executor_daemon.rs | 1973 | Executor running evals against the daemon. |
| src/executor_examples.rs | 4310 | Executor invoking runtime example binaries. |
| src/executor_mock.rs | 463 | Mock executor for tests without GPU/model. |
| src/executor_tinyquant.rs | 720 | Tiny-quant executor for quantization batteries. |
| src/host_profile.rs | 423 | Host/hardware profiling for eval context. |
| src/performance.rs | 293 | Performance metric computation for results. |
| src/quality.rs | 253 | Quality metric computation for results. |
| src/result.rs | 244 | Eval result data types and aggregation. |
| src/rocprof.rs | 476 | rocprof profiling integration and parsing. |
| src/run.rs | 487 | run_from_env top-level eval orchestration. |

## hipfire-evidence
Structured runtime/router evidence artifact schemas and writers.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 3089 | Evidence specs, runtime one-shot evidence, router histogram layers. |

## hipfire-dispatch
Unified kernel dispatch abstraction: families, registry tables, pipeline lowering.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 22 | Crate root for unified kernel dispatch. |
| src/context.rs | 46 | DispatchCtx construction incl. test helper. |
| src/coverage_tests.rs | 1125 | Guardrail catching missing dispatch arms. |
| src/traits.rs | 7 | KernelFamily trait definition. |
| src/types.rs | 768 | Core dispatch enums: ops, variants, tile impls. |
| src/tests.rs | 1449 | Unit tests for the dispatch layer. |
| src/families/mod.rs | 49 | Kernel-family module aggregator. |
| src/families/attention.rs | 1420 | Attention family params and DFlash dispatch. |
| src/families/fused_qkv.rs | 717 | Fused-QKV family params and registry. |
| src/families/gemm.rs | 233 | GEMM family across quant formats. |
| src/families/gemv.rs | 605 | GEMV family with rotation/weight refs. |
| src/families/kv_tier.rs | 747 | KV-tier paired plan derivation. |
| src/families/moe.rs | 532 | MoE expert-GEMM family resolution. |
| src/families/rotation.rs | 201 | Rotation family params and run. |
| src/model_ext/mod.rs | 9 | Model-specific extension aggregator. |
| src/model_ext/deepseek4.rs | 276 | DeepSeek V4 kernel extensions (FWHT, joint-KV, QLoRA). |
| src/model_ext/qwen35.rs | 237 | Qwen3.5 DeltaNet step/batch/tree extensions. |
| src/pipeline/mod.rs | 1789 | Pipeline type, satisfiability, linear params. |
| src/pipeline/steps.rs | 1589 | Op-list interpreter and fused-pattern matching. |
| src/pipeline/superop.rs | 486 | Lowered super-op weight/scratch/act/rope flavors. |
| src/resource/mod.rs | 19 | ResourceManager construction incl. test helper. |
| src/tables/mod.rs | 129 | KernelRegistry register/resolve. |
| src/tables/attention_table.rs | 405 | Populate attention kernel registry entries. |
| src/tables/fused_qkv_table.rs | 210 | Populate fused-QKV registry entries. |
| src/tables/gemm_table.rs | 317 | Populate GEMM registry entries. |
| src/tables/gemv_table.rs | 157 | Populate GEMV registry entries. |
| src/tables/moe_table.rs | 61 | Populate MoE registry entries. |
| src/tables/rotation_table.rs | 97 | Populate rotation registry entries. |

## hipfire-dispatch-tests
Per-model-family dispatch conformance tests and golden token-stream checks.

| file | LOC | description |
| --- | --- | --- |
| src/lib.rs | 28 | Test-crate root and module wiring. |
| src/arch_caps.rs | 334 | Atom exclusivity, molecule membership, capability matrix tests. |
| src/dtype.rs | 288 | DType size, AWQ sidecar, quant-family dispatch tests. |
| src/deepseek4.rs | 59 | DeepSeek V4 Flash (arch_id=9) dispatch tests. |
| src/llama.rs | 138 | LLaMA/Mistral/Qwen3 (arch_id=0/1) dispatch tests. |
| src/qwen2.rs | 73 | Qwen2 (arch_id=7) minimal dispatch tests. |
| src/qwen35.rs | 276 | Qwen3.5 dense+MoE (arch_id=5/6) dispatch tests. |
| tests/golden.rs | 47 | Assert build reproduces legacy golden token streams. |

