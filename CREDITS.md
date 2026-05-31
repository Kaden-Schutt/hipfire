# Credits

hipfire is an LLM inference engine for AMD RDNA GPUs. It builds on a long
chain of upstream research, drivers, and runtime code. This file
documents every significant source that informed the architecture, the
ROCm/HIP path, the kernels, and the ship-line behavior.

The shape of this file is lifted from
[ncdrone/rustane's CREDITS.md](https://github.com/ncdrone/rustane/blob/master/CREDITS.md).

## Foundational Sources

| Project | Author | What We Learned |
|---------|--------|-----------------|
| [autoresearch](https://github.com/karpathy/autoresearch) | Andrej Karpathy | Methodology pattern: `program.md` (strategy) > agent modifies one file > fixed eval > keep/discard > repeat. We adapt it for hardware/driver exploration; the "fixed eval" equivalent is the tiered ROCm validation harness in `harness.sh`. |
| [rustane](https://github.com/ncdrone/rustane) | ncdrone | Rust-native FFI to private/undocumented hardware APIs via `dlopen`. Their `ane-bridge` > `metal-decode` > `engine` decomposition is what we adapted into `hip-bridge` > `rdna-compute` > `engine`. The CREDITS.md shape here is also lifted from theirs. |
| [Mesa (radeonsi / radv)](https://gitlab.freedesktop.org/mesa/mesa) | Mesa contributors | Open AMD GPU driver source. The gfx10 register headers (`sid.h`, `gfx10_format_table.h`) and the compute-relevant register documentation that ROCm does not publish. Reference for what gfx1010 can actually do at the hardware level vs. what ROCm chooses to expose. |
| [amdgpu kernel driver](https://gitlab.freedesktop.org/agd5f/linux) | AMD + upstream contributors | KMD ioctl surface for `/dev/dri/renderD*`, PM4 command buffer format, doorbell semantics. Backstop for the redline (direct-KMD) crate and for diagnosing firmware/driver mismatches. |
| [ROCm / HSA runtime](https://github.com/ROCm/ROCm) | AMD | The runtime stack hipfire FFIs into. `libhsa-runtime64.so` and `libamdhip64.so` loaded via `libloading`; we deliberately stay off the ROCm Python and ROCm userspace stacks at runtime. |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | ggerganov + upstream contributors | GGUF format reference for the import path; the MQ4/MQ3/MQ2 family is a parallel Magnum Quants line, not a fork, but the GGUF reader cribs from llama.cpp's parser. Also our standing prefill/decode comparison baseline and the source of most of the "what does this RDNA card do under llama.cpp" intuition. |
| [candle](https://github.com/huggingface/candle) | Hugging Face | Rust ML reference for tensor layout, safetensors import, and quantization-format plumbing. We do not depend on candle at runtime; it is the closest existing Rust-native reference for "what a clean inference engine looks like" and informs the engine crate's API shape. |
| [Lucebox DFlash on ggml](https://www.lucebox.com/blog/dflash27b) | Lucebox | Standalone C++/ggml/CUDA DFlash for Qwen 3.5-27B on a single RTX 3090. Concrete published numbers to target, n_gen-aware bench methodology, and the shape of Path C (DDTree wire-up). |
| [ds4](https://github.com/antirez/ds4) | antirez | Standalone C99 reference inference for DeepSeek V4 Flash. Source of truth for `crates/hipfire-arch-deepseek4`: MTP head wiring, Hyper-Connections head-reduction algebra, raw-SWA + compressed-KV cache layout, and the tail-only YaRN RoPE convention. Our forward pass matches its numeric outputs at temp=0 within FMA-order noise. |

## Rust Crates and Runtimes

| Crate / Runtime | Use |
|-----------------|-----|
| [libloading](https://docs.rs/libloading) | `dlopen` of `libhsa-runtime64.so` / `libamdhip64.so` from `hip-bridge` and `hsa-bridge`. The whole "no ROCm install pain" story rests on this. |
| [memmap2](https://docs.rs/memmap2) | Zero-copy weight load for safetensors / GGUF / HFQ4 blobs in the engine and quantize crates. |
| [serde](https://docs.rs/serde) / [serde_json](https://docs.rs/serde_json) | Config files, registry JSON, OpenAI-compatible HTTP API, daemon IPC. |
| [rayon](https://docs.rs/rayon) | CPU-side parallelism for quantization passes and tokenizer batch encode. |
| [byteorder](https://docs.rs/byteorder) | GGUF / safetensors little-endian readers. |
| [thiserror](https://docs.rs/thiserror) | Error type derivation in the bridge crates. |
| [image](https://docs.rs/image) | PNG / JPEG decode for vision-model preprocessing. |
| [libc](https://docs.rs/libc) | ioctl / syscall plumbing for the redline direct-KMD path. |
| [Bun](https://bun.sh) | TypeScript runtime for the `hipfire` CLI in `cli/`. Picked over Node for fast startup and zero-build single-file scripts. |

## Papers

Only papers whose findings shipped as concrete behavior. Read-and-mined
papers without ship-line impact are deliberately omitted.

Author attributions are intentionally omitted: arxiv author lists are
the authoritative record, and listing a single name here invites
miscredit. Click through for the canonical author list.

| Paper | Relevance |
|-------|-----------|
| [DFlash (arXiv:2602.06036)](https://arxiv.org/abs/2602.06036) | Speculative-decode method that ships in `crates/engine/src/dflash.rs`. Target layer fusion, non-causal bidirectional attention within block, post-FFN residual hidden extraction. Our 9B / 27B DFlash perf headlines come from this. |
| [DDTree (arXiv:2604.12989)](https://arxiv.org/abs/2604.12989) | Block-diffusion draft tree, best-first heap, ancestor-only verify mask. Algorithm 1 ships in `crates/engine/src/speculative.rs`; informed the Path C PRD and the gfx1100 tree-mode FA tuning. |
| [CACTUS (arXiv:2604.04987)](https://arxiv.org/abs/2604.04987) | KL-bumped acceptance threshold replacing Leviathan `min(1, q/p)`. Shipped as the `temp>0` rejection-acceptance path so DFlash on creative content is no longer penalized for the draft being distilled on argmax. |
| [Fail-Fast drafting (arXiv:2512.20573)](https://arxiv.org/abs/2512.20573) | Per-block confidence-gated speculation length. Informs the A3B DFlash default-off gate and the dynamic draft-length policy that collapses to AR when the draft is uncertain. |
| [MoBiLE (arXiv:2510.12357)](https://arxiv.org/abs/2510.12357) | Per-token big/little MoE expert switching. Frames the A3B 24 GB consumer-card OOM mitigation; the eviction-aware sidecar work tracks this paper. |

Papers we deliberately do NOT credit: Orion (rustane-relevant, not load-bearing for hipfire); S2D2, Fast-dVLM, MineDraft (read during DFlash recon, none shipped). Performative credit is worse than no credit.

## Contributors

Listed by merged-PR count, then PR date. Core author Kaden Schutt is
omitted from this section by convention; this list is for everyone else
who has shipped code.

This section is regenerated by `scripts/refresh-credits.sh` (run after
new PRs merge). Hand-edits inside the auto block will be overwritten.

<!-- contributors:auto-start -->
### Björn Bösel ([@fivetide](https://github.com/fivetide)) - 5 PRs

- #104: feat(dispatch): per-weight MMQ screening to prevent Q8_1 outlier corruption (#87)
- #103: fix(cli): resolve raw filenames to registry tags for per-model config
- #83: fix(gfx1151): restore fp16 WMMA for fused projection prefill — 9B pp3……2 214→421 tok/s
- #81: gfx1151: require ROCm 7.2+ (RDNA 3.5 segfaults on <7.2)
- #66: tests/speed-baselines: add gfx1151 (Strix Halo) baseline

### Robin Van Cauter ([@RobinVanCauter](https://github.com/RobinVanCauter)) - 4 PRs

- #91: perf(gfx12-wmma): K4 K-tile unroll on the WMMA GEMM family
- #71: dispatch(gfx12 residual): wire WMMA kernel — 9B/27B prefill BW gap
- #62: dispatch(gfx12): wire WMMA kernels (issue #57)
- #56: Port gfx12 arch wmma + channel tests

### nickfinease ([@nickfinease](https://github.com/nickfinease)) - 3 PRs

- #35: fix(vision): resolve #23 color misidentification + address PR #22 review
- #48: triattn_validate: surface r̄ contamination + R_f plateau at runtime
- #28: fix(gfx1100): deterministic + coherent batched prefill on ROCm 7.2

### Daniil Markevich ([@KotDath](https://github.com/KotDath)) - 2 PRs

- #88: fix(gfx1151): avoid k2x32 for DFlash verify batches
- #59: fix strix halo gfx1151 autodetect

### Benedikt ([@Nereuxofficial](https://github.com/Nereuxofficial)) - 1 PR

- #75: fix: List mq6 models downloaded

### Dark ([@darkamgine](https://github.com/darkamgine)) - 1 PR

- #117: fix(ui): equivalent serve -d command on windows

### domvox ([@domvox](https://github.com/domvox)) - 1 PR

- #9: fix(cli): OpenAI-compatible serve responses, stream lock lifecycle, timeout, param passthrough, cross-platform sysfs reads, SSE headers

### Grégory D ([@flamme-demon](https://github.com/flamme-demon)) - 1 PR

- #72: DDTree wire-up + Path C PRD (validated on RDNA3 by Lucebox/buun)

### mad-lab-kbando ([@kmbandy](https://github.com/kmbandy)) - 1 PR

- #80: speed-baselines: add gfx1030 floor (RX 6900 XT, 16 GB)

### Linus Gubenis ([@linus-amg](https://github.com/linus-amg)) - 1 PR

- #67: gfx908 / MI100 (CDNA1) bring-up: wave64 dispatch + new fused_gate_up kernel

### Matt Yaple ([@myaple](https://github.com/myaple)) - 1 PR

- #93: feat(gcn5): add gfx906 bring-up path

### Kevin Read ([@unverbraucht](https://github.com/unverbraucht)) - 1 PR

- #118: perf(prefill): MMQ auto-dispatch at batch_size ≥ 256 — pp512 9B +27% (issue #60)

<!-- contributors:auto-end -->

### Co-authors (no merged PR of their own)

- **beanssec** ([@beanssec](https://github.com/beanssec)) - co-author on PR #35 (vision color misidentification fix) and PR #48 (`triattn_validate` r̄ contamination surfacing).
- **Dominik** (`git@domko.sbs`) - co-author on PR #9 (OpenAI-compatible serve, streaming lock lifecycle, SSE headers).

## License

hipfire is dual-licensed under MIT or Apache-2.0 at your option (see
[LICENSE](LICENSE), [LICENSE-MIT](LICENSE-MIT),
[LICENSE-APACHE](LICENSE-APACHE), and [NOTICE](NOTICE)). The
canonical repository transitioned from MIT-only to dual-licensing in
May 2026; see
[docs/governance/relicense-2026-05.md](docs/governance/relicense-2026-05.md)
for the full decision record including the course correction from
a unilateral Apache-2.0 relicense to dual licensing.

This CREDITS.md is the authoritative contributor inventory referenced
by NOTICE; Apache-2.0 § 4(c) requires preservation of attribution
notices in the Source form, which includes this file when
distribution is under Apache-2.0. Upstream sources listed above
retain their own licenses.
