# Changelog

## v0.1.9-alpha (2026-05-02)

Headline: **MQ3 is production-ready.** The sub-4-bit Magnum Quant from
v0.1.8-alpha is now a full first-class citizen alongside MQ4 — K4-unrolled
decode GEMV, WMMA prefill family, DFlash cross-quant matrix, gfx12 port.
27B MQ3 fits 128K context in 24 GB where MQ4 OOMs at ~115K. Plus six
contributor PRs land in the same cycle, two arch bring-ups, and a sweep
of cache-lifecycle and parser hardening in response to user reports.

### Highlights — MQ3 production push

- **K4-unrolled MQ3 decode GEMV + fused residual** (gfx1100). 9B MQ3
  decode 114 → 141 tok/s (+24%); 4B and 0.8B see proportional wins. Same
  pattern as the v0.1.8 K4 unroll on HFQ4: 4 weight reads + 4 X reads
  hoisted, 4 dequant + accumulate pairs in the body. Kernel matches MQ4
  decode within 2% on every size despite the 104 vs 136 B/group.
- **WMMA prefill family for HFQ3** — `gemm_qkvza_hfq3g256_wmma`,
  `gemm_qkv_hfq3g256_wmma`, `gemm_gate_up_hfq3g256_wmma`,
  `gemm_hfq3g256_residual_wmma`. Closes the 17× prefill gap that gated
  ship: 9B MQ3 pp32 962 tok/s, pp128 1527 tok/s. Arch-gated to gfx11
  wave32 WMMA (`gfx1100/1101/1102/1150/1151`); gfx12 K4 variant landed in
  this same cycle.
- **gfx12 (RDNA4) MQ3 WMMA port** — full 4-kernel family ported to
  `_w32_gfx12` builtin with K4 unroll + half8_t lane-split matching the
  v0.1.8 HFQ4 work. gfx1201 baseline + speed-baselines committed.
- **DFlash + MQ3 cross-quant matrix.** MQ3-target ↔ MQ3-draft, MQ3-target
  ↔ MQ4-draft, MQ4-target ↔ MQ3-draft all validated end-to-end on
  gfx1100. Refusal logic narrowed: MoE/A3B + MQ3 still refused (no MoE
  branched WMMA path); dense MQ3 ships. CLI auto-discovery prefers
  `dirname(target)` first then mq3↔mq4 cross-quant fallback dirs.
- **27B MQ3 context-fit data** — fits 128K context in 24 GB on gfx1100
  with `asym3` KV (10.7 GiB weights vs MQ4's 13.4 GiB). The 2.7 GiB
  saved on weights is the difference between fitting 128K and OOMing
  at 112K.

### Highlights — contributor PRs

- **PR #118 — Per-weight MMQ auto-dispatch** (@fivetide). HFQ4 prefill
  routes to the MMQ i8-WMMA path automatically when `batch_size ≥ 256`
  and arch supports it (`gfx1100/1101/1102/1103/1150/1151/1152`). 9B
  pp512 +27% (1672 → 2122 tok/s). Default-on; opt out with
  `HIPFIRE_MMQ=off`. Tri-state config (`off`/`on`/`auto`) exposed in
  hipfire-tui.
- **PR #117 — Windows `serve -d` parity** (@fivetide). `hipfire serve`
  daemon mode wired through PowerShell on Windows; matches the Linux
  `--detach` UX. Includes `compile-kernels.ps1` (PowerShell port of
  `compile-kernels.sh`), `install.ps1` parity with daemon precompile,
  and `hipcc.exe`-first preference for paths with spaces.
- **PR #103 — Raw-filename → registry-tag** (@fivetide). `hipfire run
  qwen3.5-9b.mq4 "..."` now resolves the per-model overrides
  (`max_think_tokens`, `temp`, etc.) the same way as the registry-tag
  form. Prior behavior silently fell back to global defaults.
- **PR #91 — gfx12 WMMA K4 K-tile unroll** (@RobinVanCauter). 7
  `.gfx12.hip` kernels rewritten with `kt += 4` inner loop;
  `tests/speed-baselines/gfx1201.txt` refreshed; closes #65. Includes
  `bench-cold.sh` for cold-process N-run distribution capture.
- **PR #93 — gfx906 / Vega 20 / MI50 bring-up** (@myaple). Family-141
  rev-detect splits gfx906 from gfx900 in `redline::device`; wave64
  dispatch list extended to gfx906; `compile-kernels.sh` skips
  WMMA/dot8 on gfx906; HSA_OVERRIDE + rocminfo fallbacks added. 196/196
  kernel compiles + 16/16 channel-tests on local MI50.
- **PR #109 — MQ2 refuse + MQ3 advisory + sweep harness** (mine, Codex
  follow-ups). MQ2 refused-by-default (severe quality cliff confirmed);
  MQ3 emits an advisory on sub-9B; sweep harness reproducibility per
  CLAUDE.md (committed prompts as files with md5 manifest, scratch off
  /tmp).

### Engine

- **Cache-invalidation lifecycle** (Codex stop-time follow-ups). Three
  cross-cutting issues fixed:
  - `Gpu::invalidate_weight_caches()` clears `mmq_screen_cache` and
    drains `fp16_shadow_cache` on `unload_model`. Previous behavior left
    pointer-keyed cache hits on freed buffers — silent corruption on
    next model load if HIP reused the address.
  - `Gpu::invalidate_graph_state()` calls `graph_destroy +
    verify_graph_destroy_all + replay_graph_destroy_all` on unload.
    Captured hipGraphs over freed weight tensors would replay against
    garbage on the next `forward_scratch_warmed_up` call.
  - `graph_destroy()` resets `ar_forward_warmed_up = false`. Without
    this, the next `forward_scratch` would skip the warmup path and try
    to replay a destroyed graph.
- **Defensive `parseToolCalls`** (#111 stopgap). Three known
  malformations now repaired before the OpenAI shape returns: spec form,
  flat form, and XML-tag corruption. Token-attractor root cause
  (calibration retrain) deferred to a follow-up release.
- **Daemon UX hardening**. `Gpu::init()` failures convert from
  `expect()` panic to a friendly platform-specific checklist via
  `report_gpu_init_failure()` + `exit(1)`. Bun stack on the CLI side
  also caught: `Engine.recv()` cleanly `process.exit(code)`s when the
  daemon early-exits, instead of throwing through a stack trace.
- **gfx1152 / Strix Halo APU arch gating**. Added to all RDNA 3.5
  dispatch lists. Does not yet fix #50 (segfault on `--precompile`);
  awaiting reporter backtrace.

### Tooling

- **`scripts/speed-gate.sh` DPM warmup** (this release). `bench_run`
  now sets `HIPFIRE_DPM_WARMUP_SECS=3` so `pp32` measurements are
  reproducible regardless of GPU thermal state. Cold-DPM penalty was
  ~16% on the 32-token prefill probe; baseline 1240 was implicitly
  warm-captured and unreproducible across fresh-process runs without
  this fix.

### Known caveats

- **MQ3 collapses on sub-9B models**. 0.8B and 4B in MQ3 are advisory
  only — they parse and dispatch but quality drops below MQ4 by a wide
  margin on real prompts. Matches QuIP# / sub-4-bit literature.
- **MQ2 is refused by default**. The quantizer requires
  `--format mq2 --i-know-this-is-broken` to opt in. Lloyd-Max MQ2
  (qt=19) and Lloyd-Max MQ3 (qt=20) are the path forward; spike
  shipped this cycle, full PRs to follow.
- **MQ3 + MoE / A3B is unsupported**. The MQ3 batched path lacks an
  MoE-branched WMMA kernel; daemon refuses MQ3 weights inside
  DeltaNetMoe / FullAttnMoe layers at load time.
- **#111 token-attractor unresolved**. The parser stopgap masks
  symptoms; calibration retrain is the real fix and lands in a
  follow-up release.
- **#50 (gfx1152 segfault)** still open pending reporter data.
- **#119 (ROCm 7.2 / clang 22 regression on Strix Halo)** filed this
  cycle; no engine-side fix yet.

### Upgrade

```bash
hipfire update                      # if installed via curl-bash
# or
git pull && cargo install --path crates/engine
```

Windows: re-run `install.ps1` — daemon.exe + kernel blobs refresh
automatically.

## v0.1.8-alpha.2 (2026-04-27)

Released the same day as alpha.1 — a second cycle's worth of contributor PRs and
infrastructure hardening landed too quickly to bundle. Five PRs merged
(#71, #72, #73, #74, #75), three Codex-flagged config-path hardening passes on
the just-merged DDTree wire-up, plus a stale-path bugfix that was blocking 27B
DFlash measurement in the speed-gate.

### Highlights

- **gfx12 WMMA dispatch is now feature-complete** for HFQ4 prefill on RDNA4.
  PR #62 wired the qkv / qkvza / gate_up scaffolds; PR #71 (@RobinVanCauter) closes
  the last un-ported GEMM hot path: `gemm_hfq4g256_residual_wmma_gfx12`. The
  residual kernel was ~42% of 9B prefill GEMM time on the dot2 fallback. R9700
  numbers vs master: 9B prefill +29.7%/+31.3% (pp32/pp128), 27B prefill
  +27.8%/+42.4%, 4B prefill +26.0%/+25.5%. Decode unaffected. New
  `tests/speed-baselines/gfx1201.txt` floor committed.
- **DDTree wire-up + Path C PRD** (PR #72, @flamme-demon). DDTree was implemented
  in `speculative.rs` but never reachable from production. Now opt-in via
  `HIPFIRE_DDTREE_BUDGET=<n>`; default decode path bit-exact preserved. The PR
  also ships `select_main_path()` + 6 unit tests in `ddtree.rs` (the first brick
  of Path C) and a 382-line PRD documenting the main-path-first orchestrator
  pattern as a follow-up to Path A revert (`ecbc49d`) and Path B dead-end
  (`39aa358`). Local validation on 7900 XTX with Qwen3.6-27B + DFlash MQ4 draft:
  12/12 attractor-clean across the `path-c-smoke.sh --full` battery, on the
  exact target/draft pair where Paths A/B1 single-token-attractor failed. Path C
  avoids the linearization-slot RoPE phase skew by construction (heap-pop
  ordering invariant). See `docs/plans/ddtree-path-c-main-path-first-from-lucebox.prd`.
- **gfx908 / MI100 CDNA1 bring-up** (PR #67, @linus-amg). 2× MI100 hardware
  validation. Wave64 dispatch added to 4 fused projection sites + a new
  `fused_gate_up_hfq4g256_wave64` kernel (HFQ4 gate+up FFN GEMV fused; same
  shape as `fused_qkv_hfq4g256_wave64`, no MFMA). Cross-process verified
  across 5 fresh-process runs per metric: 9B decode +9.3% (64.4 → 70.3 tok/s),
  4B decode +4.9%, A3B MoE decode +11.0% (86.6 → 96.1). Also extracted
  `has_wave64_native(arch)` predicate replacing 7 inline `matches!()` —
  gfx90a (CDNA2) is now a one-line addition once it has hardware to validate.
  New `tests/speed-baselines/gfx908.txt`. Two negative results documented in
  the PR body so the next CDNA1 contributor doesn't re-burn the hours
  (`v_dot2_f32_f16` regresses prefill on gfx908 despite the instruction
  existing; wave64 batched GEMM loses to fp16-packed at small batch).
- **Opt-in HFQ4 MMQ prefill path** (PR #73, @KotDath). Q8_1 activation
  pre-quantize + i8 WMMA over 128×128 output/batch tiles, similar in shape to
  llama.cpp's AMD MMQ prompt-processing path. Gated behind `HIPFIRE_MMQ=1`,
  architecture-gated to gfx1100/1101/1102/1103/1150/1151. Targets the
  Strix Halo prefill gap vs llama.cpp (#60) where the author measured the
  largest wins; on gfx1100, +19.8% on 4B pp256 once the per-batch quantize
  amortizes (small batch is dominated by quantize overhead and is not a
  target workload). Default behavior unchanged — gate verified bit-exact on
  master baseline.

### API

- **`/v1/chat/completions` thinking-mode fix** (#74, fae2867). Per-model
  `max_think_tokens` was silently dropped on the OpenAI-compatible API path,
  so models with `thinking: "on"` could consume the entire `max_tokens`
  budget inside a single `<think>...</think>` block; the downstream strip
  then left `message.content` empty while `completion_tokens` reported the
  full burn. Reproducer in #74. Also fixed `prompt_tokens: 0` hardcode —
  `total_tokens` now correctly reports `prompt + completion`.

### CLI

- **`hipfire list` shows `.mq6` models** (PR #75, @Nereuxofficial, 8c352d7).
  `listLocal()` was missing `.mq6` from its discovery filter. One-line fix.

### Speculative decode hardening

- **HFQ6 WMMA graph-capture safety** (83358c6). All 6 HFQ6 WMMA wrappers
  (3 gfx11 + 3 gfx12) used raw `unsafe self.hip.launch_kernel` instead of
  `launch_maybe_blob` — same bug class as the hipGraph dangling-kernarg
  story (#19, `project_hipgraph_moe_investigation`). Pre-fix: latent on gfx11
  (HIPFIRE_GRAPH=1 + MQ6 + prefill is a niche combination; speed-gate uses
  `.mq4` so the bug was dormant). PR #62 routed gfx12 HFQ6 to the same
  broken wrappers; Codex stop-time review caught it before any user hit it.
  Migrated all 6 to `launch_maybe_blob` with proper `KernargBlob` builders.
- **DDTree daemon config hardening** (0931afb, ce36dc8, c8ba1c1). Three
  Codex-flagged crashable env-var paths in the just-merged DDTree wire-up:
  - `HIPFIRE_DDTREE_BUDGET` had no upper bound. `DdtreeScratch::attn_bias`
    is `max_n²`; budget=10000 silently allocated 400 MB, budget=100000
    OOMed. Capped at 256 (paper Algorithm 1 typically uses ≤22).
  - `HIPFIRE_DDTREE_TOPK` was clamped to vocab_size (152064 on Qwen3.6).
    The active kernel `run_dflash_draft_for_topk_gpu` asserts `k <= 8`
    (speculative.rs:3302); my first cap of 32 still let values 9-32 panic
    the kernel. Re-capped at `min(8, vocab_size)`. Default scales to
    tiny-vocab models too: `min(4, vocab_size.max(1))`.
  - `HIPFIRE_DDTREE_PATH_C` was re-read inside the per-spec-cycle decode
    loop (microseconds of waste on the hot path) AND silently accepted
    invalid values like `"phase3"`. Hoisted out, eagerly validated, warns
    once on bad input.
  All three replace silent OOM / silent fallback with clear stderr lines.

### Tooling

- **`scripts/speed-gate.sh` 27B DFlash draft path fix** (#61, a3b8f11).
  Reported by @m0n5t3r as MISSING_DRAFT despite the file being downloaded.
  Root cause: gate hardcoded `qwen35-27b-dflash.mq4` (legacy basename +
  extension); registry standardized on `qwen35-27b-dflash-mq4.hfq` when the
  `<base>-<quant>.hfq` convention landed. Gate now accepts both names.
  9B path was already correct so it's untouched. Fixes 27B DFlash anchor
  rows in any future `--update-baselines` capture across all arches.

### Issues filed for follow-up

- **#65** — gfx12 WMMA: tune 9B prefill (multi-row, K-tile, s_prefetch,
  launch_bounds). RDNA4 follow-up to PR #71. Each lever is a discrete
  experiment. Hardware: R9700 / 9070 XT.
- **#70** — gfx908 / MI100 CDNA1: port MFMA prefill kernels (4 kernels +
  channel-tests). Closes ~35× prefill gap vs gfx1100. Toolchain validated;
  per-PR-#56 channel-test discipline applies.
- **#41** — DDTree on gfx1100 RoPE phase-skew. Superseded by PR #72's Path C
  orchestrator (different mechanism, attractor-clean). Closing in 7 days
  unless reopened.

### Upgrade

```bash
hipfire update                      # if installed via curl-bash
# or
git pull && cargo install --path crates/engine
```

Windows: re-run `install.ps1` — the dynamic-release-query (#69) will pull the
fresh `daemon.exe` automatically; asset-id cache stamp prevents a stale binary
from being preserved.

### Known issues

- **#60 reporter (@h2252) hit a `--gen 0` panic** on `bench_qwen35_mq4` pre-alpha.2.
  Cannot reproduce on master; many of today's commits could have addressed it
  incidentally. If you hit this on alpha.2, please retry with
  `RUST_BACKTRACE=1` and post the trace on #60.
- **#68 (Windows + qwen3.6:27b VL trace)** — root-caused to the
  v0.1.0-alpha-pinned daemon.exe; alpha.1's fresh binary should resolve.
  Awaiting reporter confirmation.
- **#50 (gfx1152 / Strix Halo APU segfault)** — different SKU than the
  gfx1151 work that landed today. Awaiting bt + dmesg + cache-clean repro
  from reporter.

## v0.1.8-alpha.1 (2026-04-27)

Point release rolling up the post-v0.1.8-alpha work. Two contributor PRs land in
this cycle (gfx1201 RDNA4 WMMA port from @RobinVanCauter, gfx1151 Strix Halo
autodetect from @KotDath), plus a feature on the input side (GGUF →
HFQ4/MQ4 conversion) and a docs nuke + rewrite that swaps a 39-file legacy
tree for 10 canonical pages.

### Highlights

- **RDNA4 / 9070 XT unblock end-to-end** (#54). gfx1201 WMMA codegen crash
  resolved via dispatch fallback (`6e100c2`); first canonical gfx12 WMMA
  scaffold (`6924f2a`) with C-output mapping hypothesis derived from the
  CK trait swap; full validated 5-kernel + 6-channel-test contributor port
  (PR #56) hardware-tested on R9700 silicon. C-mapping
  `acc[j] = C[8*(tid>>4) + j][tid & 15]` validated, propagated across the
  family. Public dispatch still routes gfx12 through dot2 fallback pending
  perf measurement (#57); the WMMA methods on `Gpu` are exposed for
  channel-tests now and ready to flip when numbers land.
- **gfx1151 / Strix Halo autodetect fix** (PR #59, @KotDath). KFD
  `gfx_target_version 110501` was decoding to `gfx11051` instead of
  `gfx1151`; fixed via explicit known-version table in `cli/index.ts`
  + `scripts/install.sh`. Same refactor incidentally fixes a latent
  same-class bug for any arch with non-zero step bytes
  (`100302 → gfx1003` was equally wrong before this PR). Hardware-validated
  on Ryzen AI Max+ 395 / Radeon 8060S; speed-baseline contribution welcome
  at #61. Issue #50 (gfx1152, separate Strix Halo SKU) untouched —
  detection now correctly resolves the arch, but the engine-side segfault
  reported there still needs reproduction info.
- **GGUF → HFQ4 / MQ4 import**. New `hipfire quantize <file.gguf>` mode
  accepts any GGUF the engine can load (`Q4_K_M` / `Q8_0` / `Q4_0` / `Q6_K`
  / `F16` / `BF16` / `F32` source quantizations) and re-quantizes to
  hipfire's native HFQ4-G256 (default for dense Llama / Mistral / older
  Qwen) or MQ4-G256 (FWHT-rotated, opt-in for Qwen 3.5+ family). Tensor
  names are translated GGUF → safetensors style at write time so the
  engine's existing `load_weights_hfq` consumes the output unchanged;
  the GGUF tokenizer is preserved verbatim under `meta.gguf_meta` and
  `Tokenizer::from_hfq_metadata` reads it directly (no GGUF-on-disk
  fallback). End-to-end UX:
  ```bash
  hipfire quantize ./tinyllama.Q4_K_M.gguf --install --register tinyllama:1b-gguf
  hipfire run tinyllama:1b-gguf "..."
  ```
  Quality is lower than quantizing from full-precision safetensors (it's a
  double-quant roundtrip — raise to `--format hf6` or `--format mq6` if
  you have the disk space). Format defaults are dense-aware: HF4 for GGUF
  input, MQ4 for safetensors directories.
- **`dflash_mode` default flipped to `off`** (was `auto`). DFlash is now
  opt-in: `hipfire config set dflash_mode auto` re-enables the genre-
  conditional auto-routing (DFlash on for dense Qwen 3.5+ targets, off
  for A3B without a TriAttention sidecar). Bare `hipfire run <target>`
  without that config flip stays pure AR even when a paired draft is on
  disk; the daemon logs `[hipfire] DFlash disabled (dflash_mode=off)` so
  it's not a silent footgun. Background: per-genre measurements show
  DFlash a clear win on code, modest on instruct, and a net loss on
  long-form prose. Default-on overpromised; default-off + opt-in matches
  the actual win surface.
- **Docs rewrite + LICENSE**. The 39-file `docs/` tree (mix of canonical
  user docs and operational artifacts: agent prompts, daily standups,
  port plans, perf checkpoints) consolidated to 10 canonical pages —
  `GETTING_STARTED` / `CLI` / `MODELS` / `QUANTIZE` / `CONFIG` / `SERVE` /
  `BENCHMARKS` / `ARCHITECTURE` / `QUANTIZATION` /
  `methodology/perf-benchmarking`. README cut 371 → 89 lines; first-time
  visitors see the pitch + headline benchmark + install in 10 seconds
  rather than scrolling through the model catalog. New top-level `LICENSE`
  file (was missing despite the README and `Cargo.toml` declaring MIT).
- **New `hipfire-kernel-tuning` agent skill**. Sibling to the
  `hipfire-arch-port` skill from earlier this cycle. Codifies the
  empirical kernel-perf methodology from this repo's git log: 6-step
  workflow (measure → root-cause → pick lever → cross-arch verify →
  three gates → cross-process measure), levers catalog (multi-row,
  K-tile depth, wave64 port, `s_prefetch_data`, WMMA / MFMA, fused
  projections, ISA flags, rocBLAS fallback), cross-arch dispatch
  routing rules, and five worked case studies — wave64 CDNA3 port
  (+2× MI300X decode, `4105035`), nontemporal-load fake-win revert
  (-13% caught only by clean-baseline bisect, `34eb024`), k2x32 null
  result kept for posterity (`f670e16`), gfx11 WMMA C-mapping silent
  corruption (~6 weeks before catch, `b7ac66a`), and 27B DFlash perf
  recovery root-caused to a single newline character in a bench
  prompt (`9a2c667`).
- **Vision correctness** (#23 / PR #35). `load_and_preprocess` writes
  pixel bytes in R,B,G order so the upstream HuggingFace
  `patch_embed`-export channel transposition cancels at inference; full
  details below in "Fixed".
- **27B DFlash perf restored** (`9a2c667`, ~40% recovery). PR #32
  cleanup-dead-wmma-kernels removed `gemm_hfq4g256_residual_wmma{,2,_k4}.hip`
  thinking they were dead — they were on the K4 / WMMA dispatch path for
  27B verify-shape GEMMs. Per-cycle cost on 64-layer × B=16 verify forward
  was 57 → 100+ ms. Fix landed via revert + cherry-pick of the 8 master
  commits that did NOT introduce the regression. Empirical anchor: 27B-3.5
  LRU code DFlash @ max=120 = 199 tok/s τ=10.36 (was: 95 tok/s in
  pre-revert state).

### Added

- **`hipfire quantize <file.gguf>`** — see GGUF import in highlights.
  New `crates/hipfire-quantize/src/gguf_input.rs` (self-contained reader +
  dequant for Q4_0 / Q8_0 / Q4_K / Q6_K / F16 / BF16 / F32). New
  `Tokenizer::from_gguf_meta_json` engine-side path so converted files
  carry their own tokenizer metadata.
- **gfx12 (RDNA4) WMMA kernels**: 6 new `kernels/src/gemm_*_wmma.gfx12.hip`
  kernels with channel tests in `crates/engine/examples/test_wmma_*_gfx12.rs`.
  Compile-tested green on gfx1200 + gfx1201 via the family-tag override
  in `scripts/compile-kernels.sh`.
- **`scripts/_detect-gpu.sh`** — shared `rocminfo` + `amdgpu-arch` GPU
  detection helper. Three previously hardcoded "RX 5700 XT" bench banners
  now derive from `hipfire_gpu_banner`.
- **`hipfire-kernel-tuning` skill** + extension to the `hipfire-arch-port`
  skill (canonical kernel referenced, validated C-mapping documented).
- **`gfx_target_version` known-version table** (PR #59). Explicit
  Record<number, string> for 100100 / 100300 / 100302 / 110000 / 110001 /
  110501 / 120000 / 120001, with algorithmic fallback for unknown versions.
- **CONTRIBUTING.md rewrite**. 271 → 216 lines, tester path now genuinely
  uses installer-provided binaries (`hipfire diag` + `hipfire bench`), all
  four agent skills indexed.
- **Top-level `LICENSE` file** (MIT, copyright 2026 Kaden Schutt).

### Changed

- **`prompt_normalize` default ON** (was opt-in since v0.1.8-alpha).
  Engine collapses `\n{3,}` → `\n\n` at engine entry, lifting 27B-3.5 LRU
  DFlash by +24% (159 → 199 tok/s). Opt out via `HIPFIRE_NORMALIZE_PROMPT=0`
  or `prompt_normalize=false` config when raw `\n{3,}` whitespace is
  semantically load-bearing (rare). Zero correctness cost on Qwen3.5/3.6
  vocab — `\n\n\n` was a rare BPE token (rank 1102) getting in the way of
  the much hotter `\n\n` (rank 271).
- **`dflash_mode` default OFF** (was `auto`). See highlights.
- **GGUF input format default**: `--format hf4` (was implicitly `mq4`).
  MQ4's FWHT rotation is calibrated for Qwen 3.5+ training; on Llama-style
  dense models it adds runtime overhead with no quality benefit. Override
  with `--format mq4` for Qwen 3.5+ family GGUFs.
- **CLI test-kernels.sh + megabench-q35.sh + bench-matrix.sh** auto-detect
  arch + GPU name via `_detect-gpu.sh`; the previous hardcoded "RX 5700 XT
  / gfx1010" defaults that bled into bench reports are gone.
- **AGENTS.md** — DFlash default-off surfaced in §3.6 pull-flow recipe and
  added to the §6 pitfalls table; flag table corrected
  (`HIPFIRE_DFLASH_DRAFT` description now says "filename auto-match" not
  "auto-discover", with empty-string opt-out path documented).

### Fixed

- **gfx1201 WMMA codegen crash on first dispatch** (#54). Routes to dot2
  fallback until per-arch WMMA kernels land. The dispatch predicate
  `has_wmma_f16` now matches gfx11 only; the gfx12 WMMA path is exposed
  on `Gpu` for channel-tests and will flip via #57 when perf is measured
  on R9700.
- **gfx1151 (Strix Halo) autodetect** (PR #59). KFD `gfx_target_version
  110501` decoded to `gfx11051` instead of `gfx1151` due to a
  `padStart(2, '0')` + `replace(/^(gfx\d{4})0$/, '$1')` interaction in
  the version decoder. Same class of bug also affected `100302 →
  gfx1003` (should be `gfx1030`). Fixed via explicit known-version table
  + algorithmic fallback consolidation in `cli/index.ts` and
  `scripts/install.sh`.
- **27B DFlash perf** — see highlights.
- **#23 — VL model misidentifies green and blue objects.** Pure-color
  probing (red/green/blue PNGs, temp=0 greedy decoding) showed the
  vision encoder reading green pixels as blue and blue pixels as green
  while red came through correctly — a classic G↔B transposition. Root
  cause is most likely a channel permutation in the HuggingFace
  `patch_embed` weight export (input conv channels 1 and 2 appear
  transposed); the repair lives in preprocessing — `load_and_preprocess`
  now writes pixel bytes in R,B,G order so the two transpositions
  cancel. Regression test pins the contract at
  `crates/engine/tests/channel_order.rs`.
- **Vision weight upload shape encoding.** `qwen35_vl::load_f16_gpu`
  passed byte-length as the tensor shape on both the F16-direct and
  HFQ4-dequant paths, so downstream shape-aware dispatch saw a tensor
  shaped `[byte_count]` instead of `[element_count]`. Corrected to use
  the element count.
- **Quant-format visibility for vision weights.** The loader now logs
  the detected quant format (F16 / HFQ4-G256 / HFQ4-G128) for
  `model.visual.patch_embed.proj.weight` at load time so HFQ4 models
  can be distinguished from F16 models at a glance during debugging.
- **Dead kernel file cleanup.** Removed `gemm_f16_wmma_tiled.hip` and
  `vit_attention_flash.hip` — neither was referenced via `ensure_kernel`
  dispatch and both were stale copies superseded by the active
  `vit_attention` and `gemm` kernels.
- **CLI `quantize` `--format hf6` symmetry** — the safetensors path's
  `use_hfq6` flag now accepts the `hf6` short alias to match the GGUF
  path's `GgufFormat::from_flag`, eliminating a silent fall-through where
  `hipfire quantize <safetensors-dir> --format hf6` would silently
  downgrade to the q4k default.
- **Output extension on GGUF conversion** — was `.hfq4` / `.hfq6` (which
  the CLI's `resolveModelTag` / `list` / fuzzy lookup don't recognize),
  now `.hf4` / `.hf6` so converted files surface in `hipfire list` and
  resolve via tag aliases.
- **CONTRIBUTING tester path** — was claiming "no Rust required" but
  pointed at scripts that run `cargo build`. Tester path now uses
  installer-provided `hipfire diag` + `hipfire bench`.

### Documentation

- 10 canonical `docs/` pages replace the prior 39-file tree (archive at
  `~/hipfire-docs-archive-2026-04-27/`, history preserved by git).
- `.skills/hipfire-kernel-tuning/` — new agent skill (5 markdown files,
  893 lines total).
- `README.md` cut 371 → 89 lines.
- `CONTRIBUTING.md` rewritten end-to-end.
- `AGENTS.md` updated for default-off DFlash + flag table corrections.

### Issues filed for follow-up

- **#57** — gfx12 WMMA dispatch wiring + perf vs dot2 (R9700 / 9070 XT
  hardware-gated; PR #56 landed kernels but didn't flip dispatch).
- **#58** — multi-GPU support roadmap. Pipeline-parallel first cut design
  open for discussion.
- **#60** — prefill scaling regression vs llama.cpp at pp≥512 on 9B+.
  Diagnostic phase needs no kernel writing, anyone with a 7900 XTX can
  contribute the per-kernel `HIPFIRE_PROFILE=1` breakdown.
- **#61** — gfx1151 (Strix Halo) speed-baseline + perf bench. One-command
  bootstrap for any Strix Halo owner.

### Upgrade

```
hipfire update
```

No config migration. `~/.hipfire/config.json` from v0.1.8-alpha remains
compatible. If you were relying on default-on DFlash, re-enable
explicitly:

```
hipfire config set dflash_mode auto
```

---

## v0.1.7-alpha.2 (2026-04-18)

Hotfix release for three user-visible regressions in v0.1.7-alpha. No
behavior changes beyond the fixes listed — intended as a drop-in
replacement for anyone running v0.1.7-alpha.

### Fixes

- **`hipfire config` TUI crash** (`TypeError: undefined is not an object
  (evaluating 'meta[k].label')`). The v0.1.7-alpha release added 8 new
  config keys (`experimental_budget_alert`, `dflash_adaptive_b`,
  `cask_sidecar`, `cask`, `cask_budget`, `cask_beta`, `cask_core_frac`,
  `cask_fold_m`) to `CONFIG_DEFAULTS` without matching entries in the
  TUI's `meta` field descriptor table, so every interactive `hipfire
  config` invocation on a real TTY threw on first render. Non-interactive
  `hipfire config list|get|set` flows were unaffected. Added full meta
  entries + boolean option round-tripping in `cycleOption` / `commitEdit`.
- **A3B DFlash default-on perf regression** (2-5× slower than plain AR on
  code/prose). A3B drafts reject most tokens (τ≈1.0-1.5 outside math),
  and the spec cycle overhead dominates the AR win. New `dflash_mode`
  per-model config key: `on | off | auto`. `auto` keeps dense targets
  running DFlash as before and flips A3B off unless a `cask_sidecar` is
  configured (A3B long-context on 24 GB consumer cards needs eviction to
  fit). Daemon-side belt-and-suspenders: `dflash_mode=off` skips draft
  load outright even when a draft path is supplied.
- **`hipfire config set dflash_mode <value>` → "Unknown key"**. The
  dflash_mode key was not in the released alpha's validKeys list. Ships
  as part of the same commit as the default-off gate above.

### Upgrade path

```
curl -fsSL https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/install.sh | bash
# or: hipfire update
```

No config migration needed — `~/.hipfire/config.json` written by
v0.1.7-alpha remains compatible. If you want to explicitly disable
DFlash on A3B (defaults to auto-off now anyway), either edit config.json
or run:

```
hipfire config set dflash_mode off
hipfire config qwen3.5:35b-a3b set dflash_mode off   # per-model override
```

Full v0.1.7 stable release (rocBLAS MFMA on MI300X, hipGraph+MoE fix,
full Hermes agent validation) tracking on `dflash` branch.

## v0.1.7-alpha (2026-04-18)

Pre-release tag cutting the dflash branch against master. Gated to full
v0.1.7 on the outcome of the Hermes-agent + hipfire stack validation
currently running on MI300X.

### Highlights

- **FlashTriAttn long-context wins shipped.** DFlash speculative decode +
  TriAttention KV eviction composes cleanly. Measured on 7900 XTX, 9B MQ4,
  ~1500-token prompt, 200-token decode, `--cask-budget 512 --cask-beta 128`:
  baseline 150 tok/s τ=5.31 → **FlashTriAttn 214 tok/s τ=5.36 (+42% speedup,
  τ unchanged)**. With 1M-token wikitext sidecars, τ no longer drops — earlier
  builds lost ~27% τ because the sidecar was under-calibrated.
- **CASK core-aware m-folding** merges non-core KV instead of dropping.
  Composes with FlashTriAttn. Still has a ~3% τ drop from merge smoothing —
  the GPU merge kernel (task #82) eliminates the CPU hop; full tok/s win
  lands in 0.1.7 stable.
- **Qwen3.5-35B-A3B and Qwen3.6-35B-A3B MoE** end-to-end in DFlash. Batched
  MoE prefill, fused sigmoid+residual GEMV, indexed expert dispatch. On
  7900 XTX A3B decodes at ~115 tok/s (single turn) / 96 tok/s (multi-turn).
- **MI300X (gfx942) wave64 port.** 10 hot HFQ4 kernels re-written for
  block=[64,1,1] 2-rows-per-block pattern. A3B decode 48.6 → **96 tok/s**
  on MI300X (matches 7900 XTX baseline despite the 4× memory bandwidth gap
  between consumer and datacenter silicon).
- **DFlash tape-replay rollback** lets multi-turn state recover from an
  incorrect verify without a full target re-run.
- **Batched-prefill TriAttention tap** (4.5–5× faster sidecar cals) — what
  made it possible to calibrate 1M-token sidecars across 5 targets on one
  MI300X overnight.

### Bench snapshot (7900 XTX, MQ4, branch @ `a306013`)

DFlash τ + tok/s per prompt class (ctx=4K, no CASK):

| model | short | code | math |
|-------|-------|------|------|
| 4B    | 53 tok/s τ=1.27 | 92 tok/s τ=2.49 | 148 tok/s τ=6.0 |
| 9B    | 112 tok/s τ=1.52 | **461 tok/s τ=9.95** | 288 tok/s τ=5.77 |
| 27B   | 20 tok/s τ=2.21 | 41 tok/s τ=5.66 | 42 tok/s τ=6.14 |

Sidecar reconstruction r̄ (1M wikitext tokens, default validation prompt):

| model | mean r̄ | % heads > 0.95 R |
|-------|---------|-----------------|
| 4B    | 0.564   | 5.7% |
| 9B    | 0.629   | 5.8% |
| 27B   | 0.542   | — |
| 3.5-A3B | 0.552 | — |
| 3.6-A3B | 0.552 | — |

Paper Figure 3 target is r̄ ≈ 0.5; we're above it on every model.

### CLI + daemon config (0.1.7-alpha knobs)

Per-model config (via `hipfire config` or `~/.hipfire/per_model_config.json`):

```
dflash_adaptive_b   boolean   default true     # τ-window trip-wire block shrink
dflash_mode         enum      default auto     # on | off | auto (A3B-aware)
cask_sidecar        string    default ""       # path to a .triattn.bin
cask                boolean   default false    # enable m-folding (on top of sidecar)
cask_budget         int       default 512
cask_beta           int       default 128
cask_core_frac      float     default 0.5
cask_fold_m         int       default 2
```

The daemon protocol accepts all of these in the `load` message's `params` object.
`cask_sidecar` is accepted and logged today; the generate-loop integration
lands in 0.1.7 stable (current serve users run DFlash without eviction —
use `dflash_spec_demo` directly for the `--cask-sidecar` path).

### Post-alpha fixes (land in v0.1.7 stable)

- **`dflash_mode` gate** — A3B DFlash silently routed every temp=0 request
  through DFlash in the alpha; a 7900 XTX sweep showed it's 2-5× slower than
  plain AR on code/prose (A3B draft rejects most drafted tokens — τ≈1.0-1.5
  — and the cycle overhead dwarfs the AR win). New per-model config key
  `dflash_mode: on | off | auto`. `auto` keeps dense-on, flips A3B off
  unless a `cask_sidecar` is configured (long-ctx A3B on 24 GB consumer
  cards needs eviction for correctness, and that combo wins on τ too).
  Daemon-side belt-and-suspenders: `dflash_mode=off` skips draft load even
  when a draft path is supplied. Also fixes the draft-discovery regex so
  A3B targets pick up `qwen3{N}-35b-a3b-dflash-*.hfq` under `on`/`auto+sidecar`.

### Pending for v0.1.7 stable

- Wire `cask_sidecar` + adaptive-B through the daemon's generate loop so
  `hipfire serve` honors it automatically.
- Hermes agent + hipfire stack validation on MI300X (task #125) — gates the
  stable release.
- GPU-side CASK merge kernel (task #82) to flip FlashCASK net-positive.
- DDTree integration into the CLI/daemon (currently τ-positive but not yet
  tok/s-positive without hipGraph coverage).

## v0.1.6 "deltacut" (2026-04-14)

Focus: **Qwen3.5-35B-A3B (MoE) support** end-to-end — quantizer, loader,
forward path, daemon wiring, and a stack of fused MoE kernels that take the
first-working-dense-compute path from 28 tok/s to 115 tok/s of production
decode throughput on gfx1100. Plus serve/install/bench polish.

### Qwen3.5-35B-A3B — first MoE model

35B total params / 3B activated per token. 256 experts, top-8 routing, plus
one always-on shared expert. Hybrid attention (30 DeltaNet + 10 FullAttn)
like the dense 9B, with A3B-specific shape differences: head_dim=256, 16 Q
heads / 2 KV heads, `partial_rotary_factor=0.25`, `attn_output_gate=true`.

- **Quantizer** (`hipfire-quantize`): recognizes `qwen3_5_moe` (arch id 6),
  splits the 3D-stacked `mlp.experts.{gate_up,down}_proj` tensors per-expert
  into 256 MQ4G256 blobs apiece. Rayon-parallelized across experts (80% of
  cores by default; override with `--threads N` or `HIPFIRE_QUANT_THREADS`).
  67 GB safetensors → 18.7 GB MQ4 in ~30 s.
- **Engine**: new `DeltaNetMoe` / `FullAttnMoe` `LayerWeights` variants,
  separate `SharedExpertWeights { gate, up, down }` struct (the loader was
  previously stashing `gate_proj` into the routed-expert fused slot and
  silently skipping `up_proj`), and a `moe_ffn_decode` hot path that routes
  through four new kernels (below).
- **Daemon / CLI**: `arch_id=6` dispatches through the same `qwen35` path
  as dense 5, with the loaded response reporting `arch: "qwen3_5_moe"`.
  Registry entry `qwen3.5:35b-a3b` is marked local-only (`repo: ""`) until
  the HF upload lands; `hipfire pull` short-circuits with a clear message
  instead of 404'ing.

### MoE fused-kernel stack (four new kernels)

Built up across nine incremental optimizations (each commit verified byte-
identical or byte-equivalent against the previous stage through the A3B
smoke test). Final routed-expert compute is **3 kernel launches per layer**,
down from 24 in the dense-compute reference.

- **`moe_softmax_topk_renorm_k8`** — single-workgroup GPU softmax + top-8
  selection + (optional) renormalization. Writes `[k]` indices and `[k]`
  weights to device buffers, eliminating the per-layer D2H sync the
  CPU-side top-K path needed.
- **`gemv_hfq4g256_moe_gate_up_k8_indexed`** — eight top-K experts' fused
  `gate_up` HFQ4-G256 GEMV in one launch. Reads expert IDs from a
  device-side `topk_indices` buffer; weight bases come from a per-layer
  `expert_gate_up_ptrs` pointer table built once at load. Output is split
  `[k × mi]` gate + `[k × mi]` up so the existing batched
  `fused_silu_mul_rotate_mq` consumes it unchanged.
- **`gemv_hfq4g256_moe_down_residual_scaled_k8_indexed`** — same pattern
  for the down projection. Reads scales from `topk_weights`, atomicAdds
  the weighted contribution into `x_residual`.
- **`scaled_add_inplace`** (CPU-scalar + GPU-scalar variants) — fuses the
  old (`scale_f32` + `add_inplace_f32`) pair used by the per-expert
  accumulator. The GPU-scalar variant reads the scale from a 1-element
  device buffer, keeping the shared-expert sigmoid gate on-device.
- **`gemv_hfq4g256_residual_scaled`** (CPU + GPU scalar) — one-kernel
  replacement for the `weight_gemv_residual` + explicit scale pair on the
  MQ4 SwiGLU down tail.

### MoE decode speed progression (gfx1100, A3B MQ4, greedy chat)

Each stage is a separate commit and a separate incremental win:

| Stage | tok/s | vs P1 |
|-------|-------|-------|
| Phase 1 dense-compute reference | 28 | 1.00× |
| Phase 2a (GPU sigmoid + fused scaled-add) | 77 | 2.75× |
| Phase 2a-ii (fused MQ4 `gemv_residual_scaled`) | 88 | 3.15× |
| Phase 2a-iii (pre-rotate x\_norm once per layer) | 102 | 3.65× |
| Phase 2c step 1 (fused 8-expert gate\_up) | 111 | 3.98× |
| Phase 2c step 2 (batched silu\_mul\_rotate) | 125 | 4.48× |
| Phase 2c step 3 (fused 8-expert down + atomicAdd) | 140 | 5.01× |
| Phase 2b+2c (GPU top-K + indexed kernels) | 153 | 5.46× |
| + hipGraph (single-turn smoke test only — see Known Issues) | 162 | 5.80× |

Production daemon path: **~115 tok/s** at `HIPFIRE_KV_MODE=asym3` (default).
Prefill is still per-token-fallback for MoE (`forward_prefill_batch`
eligibility check requires a dense DeltaNet layer), so pp ≈ decode at
~143 tok/s on 641 tokens — batched MoE prefill is v0.1.7 material.

### Daemon / serve / install polish

- **Daemon flock mutex** (`~/.hipfire/daemon.pid`). A second daemon process
  exits with `FATAL: hipfire daemon already running (PID N)` before
  touching the GPU instead of silently double-consuming VRAM. Fd released
  automatically on kill, so stale PID content is harmless.
- **Install precompiles MQ4 + asym3 defaults** for the detected arch at
  install time, so the first `hipfire run` doesn't eat a multi-minute JIT
  stall. `hipfire update` syncs the CLI before the cargo rebuild so the
  registry change propagates in the same invocation.
- **Serve**: frees weights on idle eviction (was leaking across eviction
  cycles), respects the per-model `max_tokens` config (default was a
  hardcoded 512 even after you set one), bumps the detach readiness
  timeout from 30 s to 5 min for cold kernel JIT, and enforces the KV
  budget end-to-end so oversized requests return a clean error rather
  than writing past the cache.
- **`hipfire run`** surfaces KV-budget errors instead of exiting 0 with no
  output. Spawns cargo/git via absolute paths detected via `autodetect`
  so `HIPFIRE_UPDATE` behaves the same whether invoked via a shell shim
  or directly.
- **`hipfire bench`** gained pp128/pp512/pp1024 prefill-scaling numbers,
  explicit prefill + decode split, and TTFT. Fixed a GPU-sync bug that
  was reporting prefill tok/s 5–10× too optimistic.

### Experimental

- **Gated `think-budget` alert injection.** When the model has burned
  more than `experimental_budget_alert_tokens` inside an open `<think>`
  block, the daemon splices a configurable nudge string into the stream
  — tokens are emitted to stdout AND forward-fed through the KV cache so
  the next sample sees the model having "said" them. Hard-gated behind
  config; off by default. See `experimental_budget_alert_tokens` /
  `experimental_budget_alert_text`.

### Known issues

- **hipGraph + MoE multi-turn corruption** ([#19](https://github.com/Kaden-Schutt/hipfire/issues/19)).
  Single-shot short decodes with `HIPFIRE_GRAPH=1` on A3B look healthy
  (162 tok/s, byte-coherent at 30 tokens), but state diverges from the
  direct path after ~40 decoded tokens — the model starts skipping a
  number in a count, loops on a single token, etc. Root cause unclear
  after a full kernel audit (all individually graph-safe). `forward_scratch`
  gates `use_graph` on `config.num_experts == 0`; dense Qwen3.5 still
  takes the graph fast path. Cost: ~30% of the potential A3B decode
  ceiling. Tracking for v0.1.7.

## v0.1.5 "redline" (2026-04-13)

First full (non-alpha) release. Focus: **RotorQuant asymmetric KV cache** for
multi-turn recall, plus a full UX overhaul that makes hipfire feel like
Ollama — background daemon, idle eviction, interactive TUI config, per-model
overrides, and `hipfire run` auto-connecting to a running serve.

### Asymmetric KV cache (asym{4,3,2}) — replaces givens

K is rotated-quantized at 2/3/4-bit with Lloyd-Max centroids; V stays Q8_0
in normal space. Value-side reuses the existing Q8_0 flash reduce path so
only K needs the rotation machinery. Always flash, always batched prefill.

- **asym3 is the new default** on every RDNA3/RDNA4 card (5.5× compression
  vs fp32, verbatim rare-token recall on Qwen 3.5 9B multi-turn).
- **asym4** — 5.1× compression for headroom-to-spare workflows.
- **asym2** — 6.0× compression for 8 GB cards (still recall-safe for
  common tokens).
- **Legacy aliases:** `turbo`/`turbo3` → asym3, `turbo4` → asym4,
  `turbo2` → asym2.

The givens2/givens4 rotation family has been fully removed from kernels,
dispatch, and the daemon. `KvCache::new_gpu_givens{4,2}` /
`new_gpu_givens4_deferred` are gone. 11 kernel files deleted.

### Multi-turn recall — fixed

Multi-turn prompts like "My name is Kaden. … What is my name?" were
returning "Kendall" / "Kade" on 9B MQ4 + givens4 KV. Root-caused to
**two bugs** landing together:

1. **K kernel head_dim=256 half-coverage.** All rotated-K kernels had
   `tid×4 × 32threads = 128` only — second half of Qwen 3.5's 256-dim head
   was silently uninitialized. Fixed via explicit 2-pass loop
   (`half=0,1`). Invisible to md5, perf benchmarks, or single-turn tests.
2. **KV precision for rare tokens.** 4-bit K collapses the outlier
   components that carry rare-token identity ("aden" subtoken). asym3's
   3-bit quantization is precise enough — asymmetric because V reuses Q8_0.

Verified: MQ4 + asym3 KV recalls "Kaden" correctly on 0.8B/4B/9B/27B.

### Flash attention — configurable per codepath

- `flash_mode` config key, tri-state `auto|always|never`.
- Only affects the Q8 path (asym modes are flash-only — no non-flash
  kernel exists). TUI surfaces `(ignored — asym is flash-only)` when a
  user has asym KV selected.
- `HIPFIRE_ATTN_FLASH` env var accepts any of `auto|always|never|0|1|2|off|on|force`.
- Dispatch: `use_flash = capture_mode || mode==2 || (mode==1 && ctx≥2048) || ctx>15000`.

### Daemon UX — Ollama-style

- **`hipfire serve -d`** / `--detach` — forks via setsid+nohup, writes PID
  to `~/.hipfire/serve.pid`, logs to `~/.hipfire/serve.log`. Polls
  `/health` up to 30s to confirm up.
- **`hipfire stop`** — SIGTERM + 5s grace + SIGKILL fallback.
- **`hipfire ps`** — lists daemons, quantize jobs, HF uploads with ETIME
  + RSS + serve-port status.
- **`hipfire run` HTTP fallback** — if a serve is running on `cfg.port`,
  run streams through its `/v1/chat/completions` instead of spawning its
  own cold-start daemon. Skips the 2-5s load cost per invocation.
- **Idle eviction** — `idle_timeout` config (default 300s). Serve unloads
  the model when no request has arrived within the window; next request
  reloads. 0 = never unload.

### Interactive config TUI

`hipfire config` launches a keyboard-driven settings editor. No more
hunt-and-peck `config set X Y`.

- ↑↓ nav, ←→/space cycle enum values, -/+ tweak numbers, Enter edits
  free-text, `r` resets/removes-override, `s` saves, `q` save+quit,
  Ctrl+C aborts.
- Long enum lists collapse to `←→ cycle (N/M)` to avoid line-wrap.
- Values color-coded by source: green if user-set, dim if default.
- Scripting still works: `hipfire config set <key> <value>`,
  `hipfire config get <key>`, `hipfire config reset [key]`.

### Per-model config overlays

- `hipfire config <model:tag>` launches the same TUI scoped to that model.
  Rows show `(inherited)` vs `(overridden)` with cyan highlighting; `r`
  removes the override instead of resetting.
- Stored as sparse JSON at `~/.hipfire/per_model_config.json` — only
  overridden keys are persisted.
- Resolution order: `--flag > per-model > global > registry default > engine fallback`.
- Overridable keys: kv_cache, flash_mode, temperature, top_p,
  repeat_penalty, max_tokens, max_seq, thinking, max_think_tokens.
  Global-only: port, idle_timeout, default_model.
- Global TUI has a "[per-model configs]" nav row at the bottom; Enter
  opens a model picker sub-TUI that lists all registered tags with
  override count + drill-down.

### New config keys

- **`max_seq`** (default 32768) — KV cache capacity allocated at model
  load. Wired through to daemon via `params.max_seq` — fixes the pre-
  existing panic when `max_tokens > 4096` with the old hardcoded default.
- **`flash_mode`** (default auto) — see above.
- **`thinking`** (default on) — `on` = model uses `<think>...</think>`
  (stripped from display); `off` = prepends a no-think directive to the
  system prompt. Advisory (instruction-tuned models comply).
- **`max_think_tokens`** (default 0 = unlimited) — reasoning budget per
  turn. Stored + passed to daemon today; hard enforcement (forced
  `</think>` emission) is a follow-up.
- **`idle_timeout`** (default 300s) — serve auto-eviction window.

### Quantize CLI — one-shot download→quantize→upload

`hipfire quantize <hf-id|local-dir>` now supports:
- `--both` (shorthand for `--format mq4 --format mq6`)
- `--stem <name>` overrides the output basename
- `--output-dir <dir>` for multi-format outputs
- `--upload <owner/repo>` — pushes to HuggingFace after quantize
- `--create-repo` — invokes `hf repos create --exist-ok` first
- `--install` — copies to `~/.hipfire/models/` so `hipfire run` finds it
- `--register <tag>` — writes a user alias to `~/.hipfire/models.json`
  so the custom tag resolves alongside the built-in registry

Example: `hipfire quantize Jackrong/Qwopus3.5-4B-v3 --both --upload schuttdev/hipfire-qwopus-4b --create-repo --install --register qwopus:4b`

### HuggingFace uploads this cycle

- `schuttdev/hipfire-qwen3.5-{0.8b,4b,9b,27b}` — MQ6 added alongside MQ4
- `schuttdev/hipfire-qwopus-{4b,9b,27b}` — MQ4 + MQ6 (Jackrong Qwopus 3.5 v3)
- `schuttdev/hipfire-carnice-{9b,27b}` — MQ4 + MQ6 (kai-os Carnice)

### Misc

- **First-run banner** on bare `hipfire` when `~/.hipfire/config.json`
  and `~/.hipfire/models/` are both absent — walks new users through
  `diag → pull → run → config`.
- **User aliases** — `findModel` consults `~/.hipfire/models.json` before
  the built-in REGISTRY, so custom fine-tunes addressed by their
  registered tag always resolve.
- **Sampler greedy fast-path** for `temperature ≤ 1e-6` — avoids the
  `1/0 → NaN` path that surfaced at temp=0.
- **`speed-gate.sh`** switched from the retired `HIPFIRE_KV_MODE=givens4`
  to `asym3`.

## v0.1.5-alpha "ichigo" (2026-04-11)

The ichigo release focuses on one thing: **MagnumQuant**, a new 4-bit weight
format that delivers Q8-grade output quality at Q4 memory bandwidth, protected
by a mandatory byte-exact quality gate. The supporting work — cross-architecture
fused projection kernels, a silent-corruption fix in the 4-accumulator GEMV
inner loop, and arch-aware quality baselines — lands in the same cycle because
MQ4 wouldn't be trustworthy without them.

### MagnumQuant (MQ4) — new quantization format

FWHT-rotated 4-bit weights in 256-element groups. Matches Q8 output quality
at Q4 bandwidth on every model we've measured.

- **Qwen3.5 MQ4 family on Hugging Face** — `schuttdev/hipfire-qwen3.5-{0.8b,4b,9b,27b}` with model cards
- **`.mq4` file extension** — recognized by CLI, daemon, and weight loader
- **CLI tags** — `hipfire pull qwen3.5:{size}-mq4` pulls the quality-gated MQ4 variant
- **HF4 remains the default** (still the fastest path) — MQ4 is explicit opt-in for quality-sensitive workloads
- **`magnum` research crate** — butterfly rotation + adaptive-mode quantizer, used for the encoder

### Mandatory byte-exact quality gate

Every change to kernels, quant formats, dispatch, fusion, rotation, rmsnorm,
or the forward pass must pass `scripts/quality-gate.sh --fast` before being
committed. Enforced automatically via `.githooks/pre-commit`.

- **Deterministic greedy decoding** (temp=0, no sampling, no repeat penalty)
- **9-test matrix** — 3 models (0.8B / 4B / 9B MQ4) × 3 prompts (compiler, math, federalist)
- **Per-GPU baselines** — `tests/quality-baselines/{gfx1010,gfx1100}/` with auto-detection via `amdgpu-arch` / `offload-arch`, honors `HSA_OVERRIDE_GFX_VERSION`
- **Byte-exact token-ID comparison** — stricter than prose coherence or md5 checks

### Silent MQ4 corruption fix — 4-accumulator interleave

A tail-group accumulator bug in the gfx1100 4x-unroll HFQ4 GEMV was dumping
all tail groups into `acc0` instead of distributing them across `acc[g%4]`.
Output was visually coherent and benchmarks passed, but token IDs diverged
from reference on any hidden_dim where `hidden_dim % (4*64) != 0`. The bug
hid for weeks because 9B/27B happened to have no tail.

- **Fixed in `5302926`** (gfx1100 4x-unroll variant)
- Same 4-accumulator interleave pattern ported to `gemv_hfq4g256` (default),
  `gemv_hfq4g256_wide`, `fused_gate_up_hfq4g256`, and `gemv_q8_0_wide`
- **The quality gate above was designed around catching this class of bug.**
  Every quality difference is now a signal until proven otherwise with
  byte-exact evidence.

### Cross-architecture fused projection kernels

The three fused GEMV projections that originated as gfx1100-tuned single-arch
kernels now compile and run on any RDNA arch from one source family, consolidated
via the 4-accumulator interleave pattern.

- **4-way LA projection** — `wqkv + wz + w_beta + w_alpha` in one launch
- **3-way FA projection** — `wq + wk + wv` in one launch
- **FFN gate+up** — `gate + up` MQ4/HF4 GEMV in one launch
- Active on gfx1010 / gfx1013 / gfx1030 / gfx1100 via dtype gate (no per-arch fork)
- Consolidation landed in `9d05c9f` (net −187 lines)

### Qwen3.5 forward-pass fusions (gfx1100)

Every layer boundary in the DeltaNet hybrid got at least one kernel fusion
this cycle.

- **conv1d + SiLU + Q/K/V split** → single kernel
- **l2_norm(Q) + l2_norm(K) + scale(Q)** → single kernel
- **sigmoid(dn_beta) + alpha_gate(dn_alpha)** → single kernel
- **sigmoid(fa_gate) + mul(fa_attn_out, fa_gate)** → single kernel
- **rmsnorm + FWHT rotation** → single kernel (Phase 3.6)
- **residual add + wo / w_down GEMV** → single kernel (Phase 3.7)
- **SwiGLU + MQ4 w_down rotation** → single kernel (Phase 3.8)
- **Per-head Q/K memcpy loop** → fused deinterleave kernel (+52%–76%)

### Multi-row HFQ4 GEMV on non-RDNA3

`R=2` multi-row HFQ4 GEMV is the new default on gfx1010 / gfx1013 / gfx1030
(RDNA1/RDNA2). Single-row was already at the bandwidth ceiling on gfx1100,
so it keeps `R=1`.

- **+2.75% measured on BC-250** (gfx1013)
- Configurable via `HIPFIRE_GEMV_ROWS` env var
- Kept opt-in on gfx1100 since the multi-row sweep showed monotonic regression

### Performance (RX 7900 XTX, gfx1100, forward-only MQ4)

| Model          | tok/s   |
|----------------|---------|
| Qwen3.5-0.8B   | **447** |
| Qwen3.5-4B     | **187** |
| Qwen3.5-9B     | **135** |
| Qwen3.5-27B    | **46**  |

End-to-end steady-state with the default CPU sampler is ~82% of forward-only;
the gap is a fixed sampling pipeline cost, not throughput-bound.

### Performance (Radeon Pro V620, gfx1030)

Baseline from an external tester on V620 (32 GB, ROCm 7.2.0) measured at
`dcd928e` — i.e. **before** the cross-arch fused-projection consolidation.
Post-consolidation V620 numbers pending hardware access; expect an uplift
on top of these.

| Model            | tok/s    | vs master |
|------------------|----------|-----------|
| Qwen3.5-9B HF4   | **61.8** | +118%     |
| Qwen3.5-9B MQ4   | **62.4** | —         |
| Qwen3.5-27B HF4  | **21.0** | —         |
| Qwen3.5-27B MQ4  | **20.9** | —         |

**27B MQ4 matches 27B HF4 throughput within 0.5%** — the 0.7 GB FWHT metadata
overhead is bandwidth-free on the RDNA2 L2 cache.

### Experimental: GPU-assisted top-K sampling

Off by default. Enable with `HIPFIRE_GPU_TOPK=1`. Net-neutral on gfx1100
(top-K extraction cost ≈ saved CPU sampling time) but lays the hardware
groundwork for a fully on-device sampler. Debug harness via
`HIPFIRE_SAMPLE_COMPARE=1` cross-checks CPU vs GPU paths byte-exact.

### Experimental: hipGraph / kernarg blob

Kernarg blob path in `hip-bridge` makes kernel launches hipGraph-capture-safe
for gfx1100. Real-kernel POC on gfx1013 produced a **negative result** (capture
hangs on RDNA1), documented in `6da45fd`. hipGraph integration is parked until
the gfx1013 regression is understood.

### Experimental: Redline / HSA bridge

Thin Rust FFI to `libhsa-runtime64.so` via the new `hsa-bridge` crate, part
of the Phase 1/2 redline audit for a direct-KMD dispatch path that bypasses
the full ROCm userspace stack.

### Experimental: speculative decoding (infrastructure)

Dual model slot + autoregressive verify-and-accept loop + DFlash hidden-state
extraction land in-tree but are not wired to the main inference path yet.
Expect activation in a later release.

### CLI / Serve

- `hipfire pull qwen3.5:{size}-mq4` — MQ4 family tags wired into the registry
- `.mq4` extension recognized across CLI, daemon, and model loader
- **`listLocal()` bug fix** — stale dangling symlinks no longer abort the local-model scan and drop every file after the bad entry
- Fuzzy model search requires explicit tag for `.mq4` (won't silently substitute for HF4)

### Diagnostics & profiling

- **Per-kernel bandwidth profiler** for the gfx1100 forward pass — each kernel's effective GB/s vs theoretical ceiling
- **Per-arch bench + profile + top-5 logit dump** examples
- Kernel efficiency profiler with hardware caps + occupancy analysis

### Known limitations

- **Non-RDNA3 byte-exact re-verification pending.** The cross-arch consolidation
  (`9d05c9f`) passes the gfx1100 byte-exact quality gate (9/9 on 2026-04-11),
  but post-consolidation byte-exact verification on gfx1010 / gfx1013 / gfx1030
  is deferred pending hardware access. The V620 baseline above is functionally
  validated at `dcd928e` (prose coherence + factual accuracy + bandwidth).
  Tracked in #64.
- **llama.cpp Q4_K_M comparison on non-RDNA3** — deferred; tracked in #65.
- **MQ6 family** — not included in 0.1.5; tracked in #67.
- **HF4/HF6 daemon HTTP response trailing-bytes bug** reported on an external
  V620 setup; investigated on k9lin (7900 XTX / Bun 1.3.5 / current tree) and
  **not reproducible**. If you hit it, please file with `bun --version` and
  `curl -v -o body.bin` output.

## v0.1.4-alpha (2026-04-08)

### Sampling
- **Frequency-scaled repeat penalty** — replaces the flat penalty with a
  count-based score weighted by recency decay. Tokens seen once far back get
  barely penalized (~1.01x); tokens repeated 3x recently get hit hard (~p³).
  Fixes long-generation word salad on all architectures. Default penalty
  dropped 1.3 → 1.15 (effective range now 1.0–1.5x).

### Kernels
- **`ds_swizzle_b32` FWHT butterfly passes** — replaces `__shfl_xor`
  (`ds_bpermute`) in all FWHT butterfly passes. 40 instructions upgraded,
  -3 VGPRs in turbo attention kernels (31→28 on gfx1010). Verified on
  gfx1010 / gfx1030 / gfx1100 / gfx1200 / gfx1201.

### gfx1100 DeltaNet correctness
- RDNA3-specific DeltaNet code path fix (details in commit `2abf27a`).

## v0.1.3-alpha (2026-04-05)

### DeltaNet Quality Fix
- **Stochastic rounding** in Q8/Q4 state requantization — fixes coherence degradation after ~500 tokens
- Gate activation verified correct (matches flash-linear-attention reference)
- Coherent output at 5000+ tokens on 4B/9B models

### 3x Speed Improvement
- **Deinterleave kernel** replaces per-head memcpy loop in full-attention layers
- 576 individual HIP memcpy calls → 9 single kernel dispatches per token
- 9B Q4: 15 → 43 tok/s

### Multi-Turn Conversation
- Cumulative KV cache + DeltaNet state across turns
- System prompt support via ChatML (`<|im_start|>system`)
- KV capacity guard with auto-reset + DeltaNet state zeroing
- Correct ChatML boundary handling (newline token run through forward)

### Interactive REPL
- `hipfire run` — ollama-style interactive chat
- `--system`, `--turbo`, `--asym`, `--hf4`, `--boundary`, `--temp`, `--max-seq` flags
- `/reset`, `/stats`, `/quit`, `/help` commands
- Thinking blocks shown dimmed, speed stats per response

### Asymmetric KV Cache (TurboQuant+)
- Q8 keys + turbo4 values — 5.1x compression vs FP32
- Attention kernel rewritten for warp-cooperative structure
- Boundary layer protection (LA-V7): first/last N KV layers at Q8
- Polynomial centroid dequant: pure ALU, zero constant memory traffic
- 9B fits at 8K+ context on 8GB VRAM (was OOM at >2K)

### Redline Engine (experimental)
- Direct-KMD GPU compute via bare libdrm_amdgpu — no HIP/ROCm needed
- 30.5µs FastDispatch, 0.5ms startup, 2.8MB RSS
- RELEASE_MEM + WAIT_REG_MEM compute barriers on gfx1010
- Dispatch API: load module, kernel, command buffer, chain dispatch
- Benchmarks: redline vs HIP numbers in benchmarks/redline_vs_hip.md

### Universal GPU Support
- JIT kernel compilation via hipcc for any detected GPU arch
- Removed pre-compiled kernel blobs (9MB, stale cache source)
- Dynamic arch detection from gfx_target_version (no whitelist)
- Targets: RDNA1-4, APUs (Strix Halo), datacenter (BC-250)

### Windows Fix
- .exe extension for daemon/infer/run binary lookup

### HF4-V Experiment
- Hipfire-native 4-bit V format (no FWHT, 32 VGPRs)
- Benchmarked: FWHT rotation confirmed as memory access optimization on RDNA1
- Turbo4+poly remains optimal compressed V path

## v0.1.2-alpha (2026-03-29)

- Initial Qwen3.5 DeltaNet support
- TurboQuant KV cache (turbo2/3/4)
- HFQ4/HFQ6 weight formats
- CLI: pull, run, serve, update, diag
