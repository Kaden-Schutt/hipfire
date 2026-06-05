# Contributing to hipfire

hipfire is alpha. Real-world testing on cards we don't have, kernel work
on archs we don't ship for, bug reports with full reproduction, and new
model architecture support — all welcome.

## Two ways to help, no Rust required

Both paths below use only the installer-provided binaries (the
`hipfire` wrapper, daemon, and quantizer dropped into `~/.hipfire/bin/`
by `scripts/install.sh`). No `cargo`, no ROCm SDK, no source build.

### 1. Run the bench matrix on your GPU

If you have an RDNA card the maintainer doesn't, the highest-leverage
thing is running the standard tester workflow and posting numbers.

```bash
hipfire diag                            # ROCm + arch detection sanity check
hipfire pull qwen3.5:0.8b               # ~0.5 GB; fits any RDNA card
hipfire pull qwen3.5:9b                 # ~5.3 GB; needs 6 GB+ VRAM
hipfire bench qwen3.5:0.8b --runs 5     # decode + prefill tok/s over 5 runs
hipfire bench qwen3.5:9b   --runs 5
```

For 16 GB+ cards, also pull and bench `qwen3.5:27b`. For mixed-arch
or non-Linux setups, see `hipfire diag --help` for environment-
specific guidance.

Open an issue titled `Benchmarks: <your GPU>` and paste the `diag`
output + each `bench` block. Results land in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md). The `hipfire-tester` skill
in `.skills/hipfire-tester/` walks an AI agent through this end-to-
end if you want help.

### 2. Diagnose and report a bug

```
hipfire diag                            # capture everything first
```

Open an issue with: GPU + ROCm version, exact command, full error
output (not just the last line), and the diag output. The
`hipfire-autoheal` skill (in `.skills/hipfire-autoheal/`) is a
fix-catalog walkthrough that an agent can apply on your behalf for
common runtime issues; if it doesn't resolve cleanly, that's exactly
the case we want filed.

---

## Developer workflow

### Setup

```bash
git clone https://github.com/Kaden-Schutt/hipfire
cd hipfire
cargo build --release --features deltanet --example daemon -p hipfire-runtime
cargo build --release --features deltanet --example test_kernels -p hipfire-runtime
cargo build --release -p hipfire-quantize
./scripts/install-hooks.sh
```

Requires Rust 1.75+ and ROCm 6+ (the dev workflow needs `hipcc` for
kernel JIT). Pre-compiled kernel blobs ship for gfx1010 / gfx1030 /
gfx1100 / gfx1200; other arches JIT-compile on first load.

`scripts/install-hooks.sh` is idempotent; it sets
`core.hooksPath=.githooks` and makes the local pre-commit hook
executable.

### No-GPU CI subset

The default CI path intentionally avoids AMD GPU access and model
downloads:

```bash
./scripts/no-gpu-ci.sh
```

It runs `cargo check --workspace --examples`, no-GPU Rust unit tests,
CPU Python tests, the env/docs drift check, and Bun tests/typecheck
when Bun is installed. GPU coherence and speed gates remain required
for kernel, dispatch, quant, forward-pass, and spec-decode changes.

### GPU kernel correctness check

```bash
./target/release/examples/test_kernels      # ~30s, no model needed
```

Validates every dispatched kernel against a CPU reference on the
detected arch. This is the load-bearing correctness gate for any
arch port; if it fails on your hardware we want to hear about it
(see issue template / autoheal skill).

### The three gates

Any change to kernels, dispatch, fusion, rotation, rmsnorm, sampling,
the spec-decode path, or the forward pass MUST pass the relevant gates
before commit. The pre-commit hook runs them automatically when staged
files match the hotspot regex.

```bash
./scripts/coherence-gate.sh             # AR coherence (panic / zero-tokens / timeout = hard fail)
./scripts/coherence-gate-dflash.sh      # spec-decode token-attractor detection
./scripts/speed-gate.sh --fast          # 4B prefill+decode regression vs tests/speed-baselines/<arch>.txt
```

**Don't bypass with `--no-verify`.** A regression the gate catches is
information. Authorized exceptions need explicit written sign-off from
the maintainer for that specific change. Read
[docs/methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md)
before claiming any perf win — within-session A/B noise is ±10–15% on
gfx1100, and the bench harness has a stale-binary trap that's bitten
us before.

### New kernel files

```bash
# Kernel source: kernels/src/<name>.hip
# Per-arch overrides: kernels/src/<name>.gfx12.hip       (family tag)
#                     kernels/src/<name>.gfx1100.hip     (chip tag)

# Register in: crates/rdna-compute/src/kernels.rs
# Wire dispatch in: crates/rdna-compute/src/dispatch.rs

# After editing any .hip file, regenerate hashes for the pre-compiled
# blob loader (otherwise the runtime falls back to JIT):
./scripts/write-kernel-hashes.sh

# Compile-check across the supported arch matrix:
./scripts/compile-kernels.sh gfx1010 gfx1030 gfx1100 gfx1200 gfx1201
```

Architecture deep-dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Quantization design (MQ4 / HF4 / asym KV math):
[docs/QUANTIZATION.md](docs/QUANTIZATION.md). Tuning an existing
kernel for perf (multi-row, K-tile depth, wave64 port, prefetch,
ISA flags) — see [`.skills/hipfire-kernel-tuning/`](.skills/hipfire-kernel-tuning/),
which catalogs the empirical methodology + every lever this repo's
git log has actually used.

### Porting to a new GPU arch

The `.skills/hipfire-arch-port/` directory is the canonical entry
point — playbook, WMMA matrix, validation procedure, contributor
onboarding. Don't write code before reading it; six-week silent-
corruption bugs from getting WMMA C-mappings wrong are how every
prior arch port has gone sideways.

Recent reference: PR #56 (RobinVanCauter, gfx1201 / 9070 XT) walked
the skill end-to-end and shipped a full validated 5-kernel port +
6 channel tests in one round. That's the bar.

### New model architectures

Start with [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)'s "Two model
paths" section — `crates/hipfire-runtime/src/llama.rs` is the template
for dense Llama-style models, `crates/hipfire-arch-qwen35/src/qwen35.rs`
is the Qwen 3.5 hybrid path with DeltaNet linear attention. Add the architecture string to
`from_gguf` / `from_hfq` and patch the tensor-shape divergences.

For a new GGUF dequant type (Q5_K, IQ-quants, etc.), port from
llama.cpp's `ggml-quants.c` into
`crates/hipfire-quantize/src/gguf_input.rs`. ~150 lines per format.

### Branch naming

| Type | Pattern |
|---|---|
| Feature | `feature/<short-name>` |
| Bug fix | `fix/<short-name>` |
| Arch port | `port/<arch>-<kernel>` |
| Benchmark contribution | `bench/<gpu-name>` |

### PR template

Concise description, before/after numbers if perf-sensitive, mention
which gates passed. One logical change per PR. Run `cargo fmt` and
`cargo clippy` before submit; CI enforces both.

For perf claims: include the binary md5
(`md5sum target/release/examples/bench_qwen35_mq4`) and the prompt
md5 if the bench is prompt-dependent. Without these, the result is
unreproducible — see
[docs/methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md).

### Code style

- `cargo fmt` — required, CI-enforced.
- `cargo clippy` — no new warnings.
- **No Python in the inference hot path.** Python is fine for
  tooling, benchmarks, and offline analysis; never in the engine.
- Comment HIP kernel parameters: VGPR budget, wave occupancy, LDS
  usage, K-tile depth — anything a reader needs to understand the
  perf shape without inspecting `--save-temps` output.

---

## Crate topology

The 0.1.20 modularization split `crates/engine/` into a runtime crate
and per-arch crates. The post-modular workspace:

```
crates/
  hip-bridge/              HIP/ROCm FFI
  rdna-compute/            kernel dispatch + per-RDNA-arch routing
                           (gfx1100/01/02/1150/1151/1152/1200/1201)
  hipfire-runtime/         LM runtime: KV cache, sampler, loop_guard,
                           prompt_frame, eos_filter, spec decode,
                           eviction, paging
  hipfire-arch-qwen35/     Qwen3.5 family (DeltaNet hybrid, MoE)
  hipfire-arch-qwen35-vl/  Qwen3.5-VL (vision)
  hipfire-arch-llama/      Llama-family (currently a facade — see
                           PR 14 for physical split)
  hipfire-arch-toy/        minimal stub arch (reference for porters)
  hipfire-quantize/        safetensors → .mq4 / .hfq quantizer CLI
```

### Where does X go?

- **"I want to add a new model architecture"** → new
  `crates/hipfire-arch-<name>/` crate, implement `Architecture` trait.
  Copy `crates/hipfire-arch-toy/` as a template.
- **"I want to fix a kernel bug or add a kernel"** → `kernels/src/*.hip`
  for the kernel + `crates/rdna-compute/src/dispatch.rs` for the
  dispatch wiring. Stays in rdna-compute regardless of which arch
  uses it.
- **"I want to tune sampler / repeat_penalty / blocked tokens
  behavior"** → `crates/hipfire-runtime/src/sampler.rs`.
- **"I want to add an end-of-turn marker for an arch"** → arch crate's
  `eos_filter_overrides()` returning
  `EosFilterOverrides { stop_at: ..., holdback_prefixes: ... }`.
- **"I want to add a CLI feature / daemon API endpoint"** →
  `crates/hipfire-runtime/examples/daemon.rs`. (Or, if it's CLI-side,
  the cli crate / TUI.)
- **"I want to optimize for a specific RDNA generation"** →
  `crates/rdna-compute/src/dispatch.rs`. NEVER inside an arch crate
  (that fragments per-arch knowledge across the workspace).
- **"I want to add a new quant format"** → `kernels/src/` for the
  kernel + `crates/rdna-compute` for dispatch routing +
  `crates/hipfire-quantize` for the quantizer CLI. Arches consume via
  the runtime API automatically.

### Per-arch overrides via the `Architecture` trait

Every arch crate `impl Architecture for Foo` and may override four
behavior structs. Defaults assume Qwen3.5 family conventions (ChatML
prompt frame, `<think>` strip, default sampler/loop-guard config).

| Override | When to use | Example |
|---|---|---|
| `LoopGuardOverrides` | Base/instruct model legitimately repeats short phrases (structured output, code boilerplate) | `LoopGuardOverrides { ngram_threshold: Some(8), .. }` |
| `SamplerOverrides` | Add arch-specific blocked tokens (e.g. `<tool_call>` openers) or per-arch `repeat_penalty` | `SamplerOverrides { blocked_tokens: vec![99999], .. }` |
| `PromptFrameOverrides` | Non-ChatML completion model (no `<|im_start|>` framing) | `PromptFrameOverrides { raw: Some(true) }` |
| `EosFilterOverrides` | Arch-specific end-of-turn markers (Gemma's `<end_of_turn>`, etc.) | `EosFilterOverrides { stop_at: vec![b"<end_of_turn>".to_vec()], .. }` |

Field-level docs live on
`crates/hipfire-runtime/src/arch.rs`; worked examples are in
`crates/hipfire-arch-toy/src/arch.rs` (one of each, default-bodied).

---

## Skills (agent-driven workflows)

| Skill | When to use |
|---|---|
| [`hipfire-tester`](.skills/hipfire-tester/) | First-time bringup + bench submission on a new GPU. |
| [`hipfire-diag`](.skills/hipfire-diag/) | "Hipfire isn't working — what's wrong?" Captures GPU/HIP/kernel state. |
| [`hipfire-autoheal`](.skills/hipfire-autoheal/) | Runtime issue triage: daemon hangs, JIT failures, port conflicts, OOM. |
| [`hipfire-arch-port`](.skills/hipfire-arch-port/) | Porting hipfire to a new GPU arch (gfx12, gfx1152, gfx94x, …). |
| [`hipfire-kernel-tuning`](.skills/hipfire-kernel-tuning/) | Optimize an existing kernel — pick a lever (multi-row, K-tile depth, prefetch, wave64 port, WMMA/MFMA, fused projections, ISA flags) and validate the win across the supported arch matrix. |

Each skill has a `SKILL.md` (or `skill.json` + sibling `.md` files)
that any agent framework can load. Designed for Claude Code / Cursor /
Codex but framework-agnostic.

---

## Where the active asks are

These three are real open questions where contributor input would
land cleanly:

- **Issue #57** — gfx12 (RDNA4) WMMA dispatch wiring + perf
  measurement vs the dot2 fallback. Needs R9700 / 9070 XT hardware.
  PR #56 landed the kernels; #57 measures and flips the dispatch.
- **Issue #58** — multi-GPU support roadmap. Pipeline-parallel first
  cut design open for discussion. Mostly weighing in vs writing code,
  unless you have a multi-GPU rig and want to prototype the
  device-aware tensor allocator.
- **Issue #50** — gfx1152 (Strix Halo APU) crash — awaiting a
  reproducer + dmesg from the original reporter. If you have one,
  comment there.

For everything else, `hipfire list -r` + a benchmark issue on any
arch we don't have local numbers for is welcome.

---

## Licensing & attribution

hipfire is dual-licensed under either:

- **MIT License** (see [LICENSE-MIT](LICENSE-MIT))
- **Apache License 2.0** (see [LICENSE-APACHE](LICENSE-APACHE))

at the recipient's option. See [LICENSE](LICENSE) for the dual-
license pointer and [NOTICE](NOTICE) for contributor attribution
details and per-file SPDX semantics. The decision record (including
the 2026-05-19 course correction from a unilateral Apache-2.0
relicense to dual licensing) is at
[docs/governance/relicense-2026-05.md](docs/governance/relicense-2026-05.md).

### For new contributors

- **New contributions default to Apache-2.0.** By submitting a
  contribution and signing off via `git commit -s` (Developer
  Certificate of Origin — <https://developercertificate.org/>), you
  certify that you have the right to license your contribution under
  Apache-2.0 and that you intend to do so.
- All commits MUST be signed off via `git commit -s`. PRs without a
  DCO sign-off line on every commit will be asked to amend.
- **Contributors may explicitly elect MIT-only for their
  contribution.** State this in the PR description (a short note like
  "license: MIT only" is enough) and the merger will tag the relevant
  files accordingly. The SPDX header will read
  `SPDX-License-Identifier: MIT`. The project still ships dual-
  licensed overall; your specific files are MIT-only.
- Add an SPDX header to every new source file you create. Templates
  live in [docs/governance/relicense-2026-05.md](docs/governance/relicense-2026-05.md).
  For sole-author files the default (Apache-2.0) template is:
  ```
  // SPDX-License-Identifier: Apache-2.0
  // Copyright (c) 2026 <Your Name>
  // hipfire — see LICENSE and NOTICE in the project root.
  ```
- For substantial modifications (>30% of lines rewritten in an
  existing file), add your own copyright line BELOW existing ones.
  Do NOT remove existing copyright lines.

### For existing (pre-2026-05-19) contributors

Your prior contributions remain licensed exactly as you originally
submitted them — under the MIT license that hipfire used at the time.
Nothing in the dual-licensing transition revokes or modifies that
grant.

If you would like your prior contributions to be available under
Apache-2.0 as well (so that hipfire's NOTICE-backed attribution
machinery applies to them and downstream forks pulling under
Apache-2.0 get your code too), you can opt in by commenting on the
relicense tracking issue (link to be added here once the issue is
opened). After opt-in, the maintainer will re-run
`scripts/governance/apply_spdx_headers.py --rewrite-spdx` to refresh
the SPDX tags on the files where you are a substantive author.

Opt-in is **entirely voluntary**. Files where you are the sole
substantive author currently carry `SPDX-License-Identifier: MIT`;
that stays unless you elect otherwise. Files of mixed authorship
where you are one of multiple substantive authors carry
`SPDX-License-Identifier: MIT OR Apache-2.0` until everyone
involved on that file has either opted in or declined.

### For downstream users / forks

Because hipfire is dual-licensed, you pick which license applies to
your use:

- If you redistribute under **MIT**, the LICENSE-MIT text applies:
  preserve the copyright notice and permission text.
- If you redistribute under **Apache-2.0**, the LICENSE-APACHE text
  applies, including § 4 obligations:
  - (a) Include a copy of the Apache-2.0 license to recipients.
  - (b) Mark modified files prominently as modified.
  - (c) Preserve per-file copyright, patent, trademark, and
        attribution notices in the Source form. Do not strip SPDX
        headers, copyright lines, or the CREDITS.md inventory.
  - (d) Include a readable copy of NOTICE in derivative-work
        distributions.

For files tagged `SPDX-License-Identifier: MIT OR Apache-2.0` you
may pick either license for your use of that file. Files tagged
`SPDX-License-Identifier: MIT` are MIT-only — you do not get an
Apache-2.0 patent grant from them — and files tagged
`SPDX-License-Identifier: Apache-2.0` are Apache-only.

Stripping attribution when redistributing hipfire code is a license
violation, treated as copyright infringement per `Jacobsen v.
Katzer`, 535 F.3d 1373 (Fed. Cir. 2008). The point of the
dual-license transition is **accreditation protection, not IP
control**: forks remain welcome under either license at the
recipient's option, but attribution MUST travel with the code.

See [AGENTS.md](AGENTS.md) for the project-level notice addressed
to AI agents helping users derive from this codebase, and
[PRIOR-ART.md](PRIOR-ART.md) for the inventory of original
architectural innovations originating in hipfire (with dates and
canonical commit hashes) that derivative works should attribute even
when no code is copied verbatim.
