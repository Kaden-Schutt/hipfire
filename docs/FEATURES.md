# hipfire — Feature Inventory

hipfire is a Rust + HIP/ROCm-direct inference (and increasingly training) engine
for AMD RDNA/CDNA GPUs (RDNA1→RDNA4, consumer/pro/APU + MI-series), shipped as a
single binary with no Python in the hot path. This page inventories engine,
training, serving, and platform features as verified against `crates/`,
`kernels/`, and `docs/` on the `chaingun` branch.

Status tags: **shipped** / **partial** (works but incomplete or guarded) /
**design** (planned, not yet implemented).

---

## Part 1 — Inference Engine

### Speculative decoding & decode acceleration

- **DFlash** — draft-model speculative decode (the headline feature). Auto-
  discovers a paired `.dflash.hfq` draft sidecar; MoE/A3B-aware verify path;
  genre-conditional speedups (≈4× on code prompts).
  `hipfire-arch-qwen35/src/dflash.rs`, `kernels/src/attention_dflash.hip`.
- **DDTree** — tree-attention speculative decode: binary-tree branching explores
  multiple token paths, linearized with a tree mask overlaid onto asym-flash
  attention. `hipfire-runtime/src/ddtree.rs`.
- **MTP (multi-token prediction)** — built-in DeepSeek-style MTP head emitting K
  draft tokens, verified against the main model. Qwen3.5 + DeepSeek V4.
  `hipfire-arch-qwen35/src/mtp_head.rs`, `mtp_spec.rs`.
- **Adaptive block-size (`adaptive-b`)** — block-size-aware verification scratch
  sized to `max(block_size, tree_budget)`; dynamic tree-depth capping.
  `hipfire-arch-qwen35/src/speculative.rs`.

### Attention & KV cache

- **Asym-aware Flash Attention** — `attention_flash` kernels with an asym-KV-aware
  variant; partial/online-softmax tiling. `kernels/src/attention_flash.hip`.
- **KV-cache quantization (asym4 / asym3 / asym2)** — 4/3/2-bit asymmetric KV with
  FWHT/Givens rotation, tree-aware, paired (write, attend) dispatch.
  `hipfire-kvquant`, `hipfire-dispatch/src/families/kv_tier.rs`.
- **Hierarchical KV (hot/cold tiering)** — recent tokens in a VRAM ring buffer;
  older tokens compacted/pruned by importance during idle decode.
  `hipfire-runtime/src/kv_hier.rs`.
- **CASK** — KV-cache eviction controller for long-context without OOM: generate a
  sidecar (`hipfire sidecar-gen`), enable with `cask-profile {balanced,…}` /
  `cask_beta`. `hipfire-runtime/src/cask.rs`.
- **TriAttn** — sparse attention with calibrated per-(layer, head, band) centers
  (phase / magnitude / mean-resultant-length), FWHT-rotated. `triattn.rs`.
- **PFlash** — long-context **prompt compression** (not speculative prefill): past
  a ~32K trigger, a tiny drafter mean-pools K per 128-token block and cosine-scores
  token importance to select/keep spans before prefill. Drafter consumption is
  **partial** (scaffolding). `hipfire-arch-qwen35/src/pflash.rs`.

### Batching & scheduling

- **Microbatching / continuous batching** — `PriorityPrefillScheduler` +
  `PriorityDecodeScheduler` with bucket selection, request coalescing, aging-based
  promotion, opportunistic pairing, backpressure, and health telemetry.
  `hipfire-scheduler/src/lib.rs`.
- **Batch / chunked prefill** — `forward_prefill_batch` / `…_chunk` per-arch, with
  multi-GPU chunking and per-layer scratch. `hipfire-arch-*/forward.rs`.
- **Prefix caching & reuse** — cross-request KV reuse via `hipfire-prompt` prefix
  fingerprinting + verbatim assistant-turn splice; surfaces as `cached_tokens` /
  `cache_write_tokens` in usage. `hipfire-prompt/src/lib.rs`.

### Sampling & decoding controls

- **Samplers** — temperature (0.0 = greedy argmax), top-p (nucleus), repeat /
  presence / frequency penalties with a configurable `repeat_window`.
  `hipfire-generate/src/sampler.rs`, `hipfire-runtime/src/sampler.rs`. *(top-k /
  min-p / typical appear in examples but are not wired into the core request API;
  user-controllable seed is not exposed.)*
- **Token blocking / attractor guard** — `blocked_tokens` forced to −∞; paired
  open/close token depth-tracking to prevent malformed structured output.
  `hipfire-generate/src/sampler.rs`. *(No float logit-bias or allow-list API.)*
- **Stop control** — up to 4 stop strings (≤64 chars each), `max_tokens` bounds,
  template-driven end tokens. `routes/chat.rs`.
- **Grammar-constrained decoding** — DeepSeek V4 tool-call grammar: a state-machine
  token mask (DSML `Matcher`) enforces valid tool/parameter structure.
  `hipfire-arch-deepseek4/src/grammar.rs`. *(No general JSON-schema / regex / GBNF
  grammar engine.)*
- **Thinking / reasoning mode** — `thinking_mode` (chat/thinking/max),
  `reasoning_effort` (none…xhigh, with token budgets), `assistant_prefix`
  (open/closed `<think>`), `max_think_tokens`. `routes/chat.rs`, `cli/commands/chat.rs`.
- **Prompt normalization** — optional `\n{3,}`→`\n\n` collapse before tokenize
  (`HIPFIRE_NORMALIZE_PROMPT`). `hipfire-config`.
- **Chat templates** — Jinja2 (minijinja) chat templates with HF `chat_template`
  sidecars, `enable_thinking` / tool context vars, ChatML structural tokens.
  `hipfire-prompt/src/lib.rs`. *(logprobs/top-logprobs computed internally for
  KLD/spec but not exposed in the response API.)*

### Output coherence detection

- **Detector bank** — n-gram density, loop-guard mirror, first-128 attractor,
  think-stall, special-token leak, EOS-immediate, tool-call-shape, whitespace-only
  detectors with pass/warn/fail verdicts. `hipfire-detect`, `hipfire-coherence`
  (the engine behind `hipfire detect` and the coherence gate).

### Quantization formats (weights)

- **MagnumQuant (MQ)** — FWHT-rotated: `mq4`, `mq3`, `mq2`, plus Lloyd-Max codebook
  `mq4l`/`mq3l`.
- **OpusQuant (OQ)** — symmetric signed int: `oq4`, `oq8`, grouped variants.
- **Modifiers** — `+` = activation-aware clip/SmoothQuant/AWQ; `++` = Hessian/LDLQ
  error feedback; mixed-precision decimals (`mq4.5+`, `oq4.25++`).
- **HFQ4-G256 / HFP4 / MFP4** — 4-bit group-256 and FP4 families; FWHT/Givens
  rotation plans. `hipfire-quantize/src/codecs.rs`.

### Parallelism & heterogeneous execution

- **Pipeline parallelism (pp≥2)** — layer distribution across GPUs over RCCL;
  admission/memory-budget checks. `serving-core/src/load.rs` (`load_model_pp`).
- **Expert parallelism (EP)** — MoE token→expert routing across devices
  (`forward_ep`, `forward_prefill_batch_ep`). `hipfire-dispatch/src/pipeline`.
- **Multi-GPU coordination** — device enumeration + RCCL collectives.
  `hipfire-runtime/src/multi_gpu.rs`, `hip-bridge/src/rccl.rs`. *(No distinct
  tensor-parallel single-matmul shard path.)*
- **NPU / XDNA1 offload** — **partial**, opt-in SwiGLU FFN / headnorm / rope on AMD
  NPU (Strix Halo). `hipfire-npu`, `hipfire-arch-qwen35/src/xdna1_ffi.rs`.
- **CPU offload / hybrid** — dense-FFN CPU fallback with per-module GPU/CPU/NPU
  backend selection. `hipfire-cpu`, `hipfire-rocm`.

### Multimodal & diffusion

- **Vision/VL** — Gemma3-VL (SigLIP encoder + projector), Qwen3.5-VL, Dots.OCR;
  image-embedding **vision-cache** sidecar keyed by xxh64; video frame extraction
  via `hipfire-media` (ffmpeg → PNG frames, CPU-only preprocessing).
  `hipfire-arch-gemma3-vl`, `hipfire-vision-cache`, `serving-core/src/generate_vl.rs`.
- **Diffusion image generation** — SD 1.x/2.x, SDXL (txt2img / img2img / inpaint),
  plus MMDiT transformer denoisers (Qwen-Image, Krea2). Schedulers: Euler,
  FlowMatchEuler, DDIM, DPM-Solver(++ multistep), Karras sigmas. Quantized weights
  supported (Q4/Q8/BF16/HFQ4-G256/…). `hipfire-diffusion`.

### Model architectures supported

`llama`/Mistral, `qwen2`, `qwen35` (+VL, DeltaNet hybrid attn, MTP, DFlash),
`gemma3` (+VL), `deepseek4` (V4 Flash, compressed-KV indexer, MTP), `minimax`
(M2, 256-expert MoE), `lfm2moe` (hybrid linear/attn MoE), `nemotron` (Mamba-2
hybrid — topology only, **design**), `mamba2` (SSM, no KV), `zaya` (ZAYA1 CCA +
EDA/MoD — **partial** bring-up), `dots-ocr`. Dense + MoE/A3B; hybrid/linear-
attention families included. Layer composition is described by `hipfire-mixer`
(`MixerKind`: attention / recurrent / MoE / hybrid).

---

## Part 2 — Training & Calibration

### Training (`hipfire-train`)

- **LoRA SFT** — **shipped** (Phase 0): LoRA adapters on frozen base weights, real
  fp32 forward + matching backward built on `gemm_f32_train` (rdna-compute).
  `hipfire-train/src/lib.rs`.
- **Optimizer & loop** — AdamW (decoupled weight decay, bias correction), LR
  scheduling, per-parameter moment buffers; finite-diff gradchecked.
  `hipfire-train/src/optim.rs`.
- **Drafter training** — trains small SSM/attention drafters for DFlash/PFlash via
  ListNet top-1 ranking loss on importance-weighted block scores; example
  `ssm_drafter_train.rs`. `hipfire-train/src/drafter.rs`.
- **QAT (fake-quant + STE)** — **design**, for the PFlash drafter.
- **Train-as-daemon-op** — **design**: run training through the resident daemon's
  HIP context (new `Collect`/`train_drafter` ops) to avoid sidecar reload + two-
  process GPU locking. `docs/plans/2026-06-19-train-as-daemon-op.md`.
- **Checkpoints / resume / datasets** — dependency-free binary formats (`PFLB`
  label cache, `PFDC` drafter checkpoint with AdamW moments + epoch), `--resume`;
  corpus tokenized into fixed SEQ=512 chunks, label cache keyed by geometry hash.
  `hipfire-train/src/checkpoint.rs`.
- **Arch coverage** — LLaMA training **shipped** (gradchecked); Qwen3.5 training in
  **design** (Scope A full train vs Scope B forward-only).
  `docs/plans/2026-06-18-qwen35-training-support.md`.

### Calibration / artifact collection

- **imatrix** — per-linear `Σx²` importance vectors via `calib_sumsq_reduce_f32`;
  the K-vector basis for AWQ, and the only artifact stored for MoE routed experts.
- **Hessian** — full `Σxxᵀ` via tiled `calib_hessian_outer_f32`, stored compactly
  (F32 diagonal + BF16 lower-triangle); streaming writer for large models; verified
  byte-identical to a Python reference. Feeds LDLQ/`oq++`.
- **CLI / API** — `hipfire collect-artifacts` (and a `hipfire_runtime::calibration::
  CalibCollector` lib path; daemon `Collect` op is **design**).
  `docs/calibration/collector-status.md`.
- **AWQ** — derived at quant time from the captured imatrix + weights (no separate
  stored artifact); `awq_*` ablation scripts + Astrea recipe stage. *(No
  SmoothQuant / clip-search implementation found.)*
- **CASK sidecar / TriAttn band-center calibration** — `BandCenter` phase /
  magnitude / MRL with FWHT rotation encoding; **partial** (sidecar generation not
  yet fully wired). Rotation-plan calibration is **scaffolding**.

### Training & calibration surfaces

- **HTTP** — `GET /admin/training/runs`, `/admin/training/runs/{id}`,
  `…/{id}/events` (read-only run status/metrics/events). `routes/training.rs`,
  backed by `hipfire-operator`.
- **TUI** — training tab (run list/detail/event streaming, read-only).
  `hipfire-tui/src/hipfire/training.rs`.

---

## Part 3 — Serving & Tooling

### HTTP API surface (`hipfire serve`, default `0.0.0.0:11435`)

- **OpenAI Chat Completions** — `POST /v1/chat/completions`, SSE streaming, usage
  with cached-token detail. `routes/chat.rs`.
- **OpenAI Responses API** — `POST /v1/responses` (newer shape) with streaming SSE
  events. `routes/responses.rs`.
- **Tool / function calling** — OpenAI-native + inline-XML parsers, streamed tool-
  call chunks, defensive repair of known attractor malformations. `runtime/src/tool_call.rs`.
- **Models / Batches / Files** — `GET /v1/models`, `/v1/batches/{id}`, `/v1/files`.
- **AUTOMATIC1111-compatible diffusion API** — full `/sdapi/v1/*`: `txt2img`,
  `img2img`, `progress`, `interrupt`, `skip`, plus `samplers`, `schedulers`,
  `sd-models`, `sd-vae`, `loras`, `embeddings`, `options`, `png-info`, `upscalers`,
  etc. Works with stable-diffusion-webui clients. `routes/sdapi.rs`.
- **Health / metrics** — `GET /health` (worker/batch/diffusion status),
  `/admin/stats`, telemetry module.
- **Not present:** legacy `/v1/completions`, text `/v1/embeddings`, OpenAI
  `/v1/images/generations` (image gen is A1111-only), and native Ollama `/api/*`.

### Web UIs, terminal, daemon

- **Admin WebUI** — Leptos/WASM console at `/admin/ui` (legacy `/admin`): config
  schema + resolved config, diagnostics, logs, stats, model registry, training
  runs; bearer/session gated. `hipfire-admin-ui`.
- **Chat WebUI** — Leptos/WASM chat at `/` and `/chat`, image attachments.
  `hipfire-chat-ui`.
- **TUI** — ratatui app: chat, model picker, config tabs (GPU/scheduler/training),
  status pane (health, logs, resource locks, kernel cache), registry browser,
  daemon spawn. `hipfire-tui`.
- **Resident daemon** — `hipfire-daemon` holds the engine, manages model
  lifecycle/swap, draft auto-discovery, generation streams, and in-daemon KLD eval;
  JSONL wire protocol (`hipfire-daemon-protocol`: `DaemonRequest`/`Response`,
  `KldEval*`, `Collect*`); `flock(2)` resource leases.

### CLI (`hipfire …`)

`serve`, `chat`, `run`, `list`, `pull` (Ollama-style model/draft fetch),
`quantize`, `config` (global/per-model), `eval`, `sidecar-gen`, `detect` (token
coherence), `diffusion` (import/inspect diffusion `.hfq`), `admin`,
`lock {acquire,release,status}` (GPU/NPU/CPU resource mutex), `host-profile`
(bandwidth/capability profiling), `collect-artifacts` (Hessian/imatrix), `repack`
(arch-optimal weight layout), plus config/doc/schema generators.

### Quantization, evaluation, evidence

- **`hipfire quantize`** — HF safetensors / GGUF input → `.hfq`; flags `--imatrix`,
  `--awq`, `--ldlq`, `--hessian`; emits MQ/OQ families.
- **`hipfire eval` batteries** — KLD-based correctness/quality + runtime (batching,
  prefix reuse, KV admission, pp admission) evidence; daemon-resident and example
  executors. `hipfire-eval`, `hipfire-kld`.
- **Evidence records** — structured `EvidenceRecord`/`EvidenceArtifact` (phase
  timing, launch counts, MoE router stats, memory, DFlash traces).
  `hipfire-evidence`.
- **Kernel Atlas** — typed JSONL benchmark corpus + analysis helpers for kernel
  perf tracking. `hipfire-atlas`.
- **Model management** — registry/tags + BYO file paths; HuggingFace fetch; content
  hashing for cache keys / model identity (`hipfire-hash`); canonical `.hfq` naming
  with feature sidecars (`.dflash.`, `.mtp.`, `.vl.`, `.triattn.`).

---

## Part 4 — Platform & Internals

- **HIP/ROCm-direct backend** — `dlopen` of `libamdhip64` (no ROCm userspace stack
  at runtime); safe Rust FFI. `hip-bridge`.
- **Alternate dispatch backends (research)** — `hsa-bridge` (direct
  `libhsa-runtime64` / KFD) and `redline` (bare-libdrm / direct-KMD PM4 dispatch)
  explore bypassing HIP for latency; **partial/research**.
- **hipGraph capture & replay** — CUDA-graph-style command batching for decode
  (`HIPFIRE_GRAPH`, `HIPFIRE_VERIFY_GRAPH`); used in DFlash draft forward.
  `hip-bridge/src/ffi.rs`.
- **Kernel compilation & cache** — offline `hipcc` → `.hsaco` ELF with mtime-
  validated cache (`KernelCompiler`); precompiled blobs loaded at runtime; per-model
  caches (`mmq_screen_cache`, `fp16_shadow_cache`) drained on unload. No runtime JIT.
  `rdna-compute/src/compiler.rs`.
- **OwnedTensor RAII scratch** — RAII transient GPU tensors with a deferred-free
  mailbox for graph-gated reclaim (`alloc_owned`/`zeros_owned`).
  `rdna-compute/src/dispatch/mod.rs`.
- **Sequence state management** — `SequenceStateHandle` / page descriptors for KV +
  recurrent-state allocation/reservation. `hipfire-state`.
- **Resource locking** — single `flock(2)` primitive (`FlockGuard` /
  `hipfire lock`) coordinating GPU/NPU/CPU leases across daemon and non-daemon
  callers. `hipfire-lock`.
- **Version identity** — Git-derived `vX.Y.Z-N-gSHA` embedded at build time.
  `hipfire-build-info`.

---

## Part 5 — Comparison vs. vLLM and llama.cpp

The hipfire column is verified against this repo (see citations above). The vLLM
and llama.cpp columns reflect general knowledge of those projects, not this
repo's source, and move fast — treat them as orientation, not a spec. Legend:
`yes` / `partial` / `no` / `n/a`.

### Hardware & runtime

| Capability | hipfire | vLLM | llama.cpp |
|---|---|---|---|
| Primary target | AMD RDNA/CDNA (incl. consumer RDNA) | NVIDIA CUDA (AMD ROCm/others secondary) | Cross-vendor (CUDA/ROCm/Metal/Vulkan/CPU/SYCL) |
| Backend approach | HIP/ROCm-direct via `dlopen libamdhip64`, no ROCm userspace at runtime | Python + CUDA/C++ | C/C++, per-backend |
| Distribution | Single Rust binary, no Python hot path | Python package + native libs | Native binaries / libs |
| Consumer-RDNA focus | yes (RDNA1→RDNA4) | no | partial (via HIP/Vulkan) |
| Sub-HIP / direct-KMD dispatch | partial (HSA/redline research) | no | no |
| NPU offload | partial (XDNA1, experimental) | no | no |
| GPU graph capture/replay | yes | yes | partial |

### Decode acceleration & speculation

| Capability | hipfire | vLLM | llama.cpp |
|---|---|---|---|
| Draft-model speculative decode | yes (DFlash) | yes | yes |
| Tree/branch speculation | yes (DDTree) | partial (EAGLE/Medusa-style) | no |
| Multi-token-prediction head | yes (MTP) | yes (MTP/EAGLE) | no |
| n-gram / lookahead spec | no | yes (n-gram) | partial (lookahead) |

### Batching, scheduling, KV

| Capability | hipfire | vLLM | llama.cpp |
|---|---|---|---|
| Continuous / in-flight batching | yes (priority prefill+decode scheduler) | yes (core strength) | partial (server slots) |
| Paged KV attention | no (hierarchical KV + CASK eviction instead) | yes (PagedAttention) | no |
| Chunked prefill | yes | yes | partial |
| Automatic prefix caching/reuse | yes (prompt-fingerprint) | yes | partial (prompt cache) |
| KV-cache quantization | yes (asym4/3/2, rotated) | yes (FP8/int) | yes (q8/q4 flags) |
| Long-context KV eviction | yes (CASK) + prompt compression (PFlash) | partial | no |

### Quantization & calibration

| Capability | hipfire | vLLM | llama.cpp |
|---|---|---|---|
| Built-in quantizer tool | yes (`hipfire quantize`) | no (external: llm-compressor/autoawq) | yes (`llama-quantize`) |
| Weight formats | MQ/OQ (FWHT-rotated), Lloyd, HFQ4-G256, HFP4/MFP4 | GPTQ/AWQ/FP8/Marlin/bitsandbytes | GGUF k-quants/i-quants |
| Activation-aware / error-feedback | yes (AWQ `+`, Hessian/LDLQ `++`) | yes (AWQ/GPTQ) | yes (imatrix) |
| imatrix / Hessian collection | yes (`collect-artifacts`) | external | yes (imatrix) |

### Parallelism

| Capability | hipfire | vLLM | llama.cpp |
|---|---|---|---|
| Tensor parallelism | no | yes | partial (row/tensor split) |
| Pipeline parallelism | yes (pp≥2) | yes | partial (layer split / RPC) |
| Expert parallelism (MoE) | yes | yes | partial |
| CPU offload / hybrid | yes | partial (CPU offload) | yes (core strength) |

### Decoding controls

| Capability | hipfire | vLLM | llama.cpp |
|---|---|---|---|
| temperature / top-p / penalties | yes | yes | yes |
| top-k / min-p / typical | no (core API) | yes | yes |
| user-seeded reproducibility | no | yes | yes |
| JSON-schema / regex / GBNF grammar | partial (tool-call grammar only) | yes (guided/outlines/xgrammar) | yes (GBNF) |
| Jinja chat templates | yes | yes | yes |
| Thinking/reasoning-effort controls | yes | partial | partial |
| logprobs in API | no | yes | yes |

### Modalities & serving

| Capability | hipfire | vLLM | llama.cpp |
|---|---|---|---|
| Vision / VLM | yes (Gemma3-VL, Qwen-VL, Dots.OCR) | yes (broad) | partial (llava etc.) |
| Diffusion image generation | yes (SD/SDXL/MMDiT, in-process) | no | no (separate stable-diffusion.cpp) |
| OpenAI Chat Completions API | yes | yes | yes |
| OpenAI Responses API | yes | partial | no |
| Legacy completions / embeddings API | no | yes | partial |
| AUTOMATIC1111 diffusion API | yes (`/sdapi/v1/*`) | no | n/a |
| Tool / function calling | yes | yes | partial |
| Admin + chat web UIs | yes (Leptos/WASM) | no (dashboard only) | partial (bundled web UI) |
| Terminal UI | yes (`hipfire-tui`) | no | partial (CLI REPL) |

### Training

| Capability | hipfire | vLLM | llama.cpp |
|---|---|---|---|
| On-device training / fine-tune | partial (LoRA SFT shipped; QAT/Qwen3.5 in design) | no (inference only) | partial (finetune/LoRA tooling) |
| Drafter / spec-model training | yes (SSM/attn drafter) | no | no |
| Autograd + optimizer (AdamW) | yes | n/a | partial |

### Where each tends to win

- **hipfire** — consumer/pro AMD RDNA (incl. APUs) without the ROCm userspace
  stack, a single static binary, rotated low-bit weight + KV quantization with a
  built-in quantizer/calibrator, in-process diffusion with an A1111-compatible API,
  reasoning-mode controls, and on-device LoRA/drafter training. Narrower
  model/hardware coverage; no tensor parallelism, paged attention, or general
  grammar engine.
- **vLLM** — highest multi-tenant throughput on datacenter NVIDIA via
  PagedAttention + continuous batching, broad model coverage, full TP/PP/EP scaling,
  and rich guided decoding. Python stack; inference only; consumer-AMD secondary.
- **llama.cpp** — the broadest hardware reach (every vendor + CPU), the GGUF
  ecosystem, GBNF grammars, and the strongest CPU/GPU hybrid offload for running
  large models on limited VRAM. Less throughput-oriented serving than vLLM.
