# Contributing to hipfire

hipfire is alpha. Real-world testing on cards we don't have, kernel work
on archs we don't ship for, bug reports with full reproduction, and new
model architecture support are all welcome.

## Two ways to help (no Rust required)

Both paths use installer binaries only (`scripts/install.sh` →
`~/.hipfire/bin/`). No `cargo`, no ROCm SDK, no source build.

### 1. Run the bench matrix on your GPU

```bash
hipfire diag
hipfire pull qwen3.5:0.8b
hipfire pull qwen3.5:9b
hipfire bench qwen3.5:0.8b --runs 5
hipfire bench qwen3.5:9b   --runs 5
```

For 16 GB+ cards, also bench a larger tag from `hipfire list -r`.
Open an issue titled `Benchmarks: <your GPU>` with `diag` output and each
`bench` block. Numbers are measured contributions for
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) — not live floors.

Agent walkthrough: [`.agents/skills/hipfire-tester/`](.agents/skills/hipfire-tester/).

### 2. Diagnose and report a bug

```bash
hipfire diag
```

Open an issue with GPU + ROCm version, exact command, full error output,
and the diag dump. Runtime triage catalog:
[`.agents/skills/hipfire-autoheal/`](.agents/skills/hipfire-autoheal/).
Diagnostics skill:
[`.agents/skills/hipfire-diag/`](.agents/skills/hipfire-diag/).

---

## Developer setup

```bash
git clone https://github.com/Kaden-Schutt/hipfire
cd hipfire
cargo build --release --features deltanet --example daemon -p hipfire-runtime
cargo build --release --features deltanet --example test_kernels -p hipfire-runtime
cargo build --release -p hipfire-quantize
./scripts/install-hooks.sh
```

Requires Rust **1.85+** and a HIP/ROCm install with `hipcc` for kernel JIT.
Baseline consumer stacks commonly use ROCm 6+; **RDNA4 (`gfx1200`/`gfx1201`)
needs ROCm 6.4+**, and **Strix Halo / `gfx115x` needs ROCm 7.2+** (see
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)). Pre-compiled kernel blobs
ship for common consumer arches; others JIT on first load.

`scripts/install-hooks.sh` is idempotent (`core.hooksPath=.githooks`).

### No-GPU checks

```bash
./scripts/no-gpu-ci.sh
```

Runs workspace `cargo check`, selected no-GPU Rust lib tests, CPU pytest,
focused docs-reliability unit tests
(`python3 -m unittest tests.test_docs_reliability`), the canonical
documentation reliability checker
(`scripts/check-docs-reliability.py`; env-table coverage included), and Bun
test/typecheck when Bun is present. The former standalone env-table checker is
subsumed. This matches the automatic no-GPU CI body. It does
**not** cover kernel numeric correctness, serve semantics, Redline route
proof, or perf admission.

Documentation reliability local commands (no GPU / no model downloads) and
pre-commit docs-only short-circuit behavior are owned by
[docs/VALIDATION.md](docs/VALIDATION.md) — do not duplicate the claim→route
matrix or path/gate selectors here. Staged pre-commit form and commit-tree CI
form both require an explicit `--base-ref`; CI supplies the integrated commit
and base externally and records commit SHA, tree SHA, source refs, checker
results, and the semantic matrix in the job artifact.

Do not hand-edit generated `registry/v1.json` — it is produced by
`scripts/registry_gen.py` from curated `cli/registry.json`. Prefer
`scripts/fmt-changed.sh` over bare `cargo fmt` on large trees.

---

## Making changes

### Where does X go?

Crate layout overview: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Architecture ids: [docs/architecture-ids.md](docs/architecture-ids.md).

| Intent | Place |
|---|---|
| New model architecture | New `crates/hipfire-arch-<name>/` implementing `Architecture`; start from `crates/hipfire-arch-toy/` |
| Kernel bug / new `.hip` | `kernels/src/` + dispatch wiring in `crates/rdna-compute` / `hipfire-dispatch` |
| Sampler / loop-guard / prompt frame | `crates/hipfire-runtime` (+ arch `Architecture` overrides) |
| CLI / daemon API | Bun CLI under `cli/`; daemon in `crates/hipfire-runtime/examples/daemon.rs` |
| Arch-specific GPU tuning | Dispatch tables — not inside an arch crate |
| New quant format | Kernels + dispatch + `crates/hipfire-quantize`; see [docs/QUANTIZATION.md](docs/QUANTIZATION.md) |

### New kernel files

```text
kernels/src/<name>.hip
kernels/src/<name>.gfx12.hip      # family tag
kernels/src/<name>.gfx1100.hip    # chip tag
```

Register and dispatch in the compute/dispatch crates, then:

```bash
./scripts/write-kernel-hashes.sh
./scripts/compile-kernels.sh gfx1010 gfx1030 gfx1100 gfx1200 gfx1201
```

Tuning levers and historical methodology:
[`.agents/skills/hipfire-kernel-tuning/`](.agents/skills/hipfire-kernel-tuning/).

### Porting to a new GPU arch

Canonical entry:
[`.agents/skills/hipfire-arch-port/`](.agents/skills/hipfire-arch-port/).
Validation procedure (channel / speed — not retired coherence batteries):
[docs/methodology/arch-port-validation.md](docs/methodology/arch-port-validation.md).

### Branch naming

| Type | Pattern |
|---|---|
| Feature | `feature/<short-name>` |
| Bug fix | `fix/<short-name>` |
| Arch port | `port/<arch>-<kernel>` |
| Benchmark contribution | `bench/<gpu-name>` |

### PR expectations

- One logical change per PR; concise description.
- `cargo fmt` / `cargo clippy` clean (CI enforces).
- State which [validation route(s)](docs/VALIDATION.md) you ran and on
  what model/arch.
- Perf claims: binary md5 + prompt md5 when prompt-dependent; follow
  [docs/methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md).
- **Don't bypass hooks with `--no-verify`.** Exceptions need explicit
  maintainer sign-off for that change.

Code style: no Python in the inference hot path; comment HIP kernel
parameters (VGPR/LDS/occupancy/K-tile) that a reader needs without
`--save-temps`.

---

## Validation

**Sole human route selector:** [docs/VALIDATION.md](docs/VALIDATION.md).

Rules of thumb:

1. Pick the **narrowest** route for the changed surface.
2. Automatic no-GPU CI green never substitutes for a required manual
   GPU/model route.
3. There is **no universal replacement gate**.
4. Fixed `scripts/coherence-gate-*.sh` batteries are **retired** as
   acceptance evidence (historical reproduction only).
5. Harness success without the proof the claim needs is insufficient —
   especially Redline timed-arm route proof ([docs/REDLINE.md](docs/REDLINE.md)).
6. Admissions are only rows in [docs/admissions.yml](docs/admissions.yml)
   (empty / fail closed under schema v1).

Docs reliability checker commands, pre-commit docs-only short-circuit (always
check, then skip runtime/GPU gates for documentation/governance/tooling-only
staged sets; mixed commits keep applicable runtime gates), and CI identity
rules: same file ([docs/VALIDATION.md](docs/VALIDATION.md) — Documentation
reliability checker / Automatic entrypoints). VALIDATION remains the sole
route selector; this page does not restate the claim matrix.

Navigation and ownership map: [docs/INDEX.md](docs/INDEX.md).

---

## Skills (executable agent workflows)

**Sole executable skill root:** [`.agents/skills/`](.agents/skills/).

| Skill | When |
|---|---|
| [`hipfire-tester`](.agents/skills/hipfire-tester/) | Bring-up + bench submission on a new GPU |
| [`hipfire-diag`](.agents/skills/hipfire-diag/) | GPU/HIP/kernel readiness diagnostics |
| [`hipfire-autoheal`](.agents/skills/hipfire-autoheal/) | Runtime triage (daemon, JIT, port, OOM) |
| [`hipfire-arch-port`](.agents/skills/hipfire-arch-port/) | New GPU arch port |
| [`hipfire-kernel-tuning`](.agents/skills/hipfire-kernel-tuning/) | Kernel perf levers + cross-arch validation |
| [`hipfire-kernel-atlas`](.agents/skills/hipfire-kernel-atlas/) | Kernel Atlas / ISA fit measurements |
| [`astrea`](.agents/skills/astrea/) | Quant calibration / quality experiments |
| [`rebase-onto-modular`](.agents/skills/rebase-onto-modular/) | Porting pre-modular branches onto post-split master |

Load skills only from that root.

---

## Licensing and attribution

hipfire is dual-licensed under either:

- **MIT License** ([LICENSE-MIT](LICENSE-MIT))
- **Apache License 2.0** ([LICENSE-APACHE](LICENSE-APACHE))

at the recipient's option. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
Decision record (including the 2026-05-19 course correction from a
unilateral Apache-2.0 relicense to dual licensing):
[docs/governance/relicense-2026-05.md](docs/governance/relicense-2026-05.md).

### New contributors

- **New contributions default to Apache-2.0.** By submitting a
  contribution and signing off via `git commit -s` (Developer
  Certificate of Origin — <https://developercertificate.org/>), you
  certify that you have the right to license your contribution under
  Apache-2.0 and that you intend to do so.
- All commits MUST be signed off via `git commit -s`. PRs without a
  DCO sign-off line on every commit will be asked to amend.
- **Contributors may explicitly elect MIT-only** for their contribution.
  State this in the PR description (e.g. `license: MIT only`); the merger
  tags files accordingly (`SPDX-License-Identifier: MIT`). The project
  still ships dual-licensed overall.
- Add an SPDX header to every new source file. Templates live in
  [docs/governance/relicense-2026-05.md](docs/governance/relicense-2026-05.md).
  Sole-author default:

  ```text
  // SPDX-License-Identifier: Apache-2.0
  // Copyright (c) 2026 <Your Name>
  // hipfire — see LICENSE and NOTICE in the project root.
  ```

- For substantial modifications (>30% of lines rewritten), add your
  copyright line **below** existing ones. Do **not** remove existing
  copyright lines.

### Existing (pre-2026-05-19) contributors

Prior contributions remain licensed exactly as originally submitted
(MIT at the time). Nothing in the dual-licensing transition revokes that
grant.

Optional Apache-2.0 opt-in for prior work is voluntary; after opt-in the
maintainer re-runs `scripts/governance/apply_spdx_headers.py` as needed.
Files of mixed authorship stay `MIT OR Apache-2.0` until every
substantive author has opted in or declined.

### Downstream users / forks

- **MIT redistribution:** preserve copyright notice and permission text.
- **Apache-2.0 redistribution:** LICENSE-APACHE § 4 — include the license,
  mark modified files, preserve Source-form notices (SPDX, copyright,
  CREDITS), and include a readable NOTICE copy.
- Per-file SPDX tags control which grant applies to that file.
- Stripping attribution when redistributing is a license violation
  (`Jacobsen v. Katzer`, 535 F.3d 1373 (Fed. Cir. 2008)). Dual license is
  **accreditation protection, not IP control**: forks remain welcome;
  attribution MUST travel with the code.

Project-level agent notice and innovation inventory:
[AGENTS.md](AGENTS.md), [PRIOR-ART.md](PRIOR-ART.md).
