# Contributor onboarding for GPU arch ports

Hipfire expands ISA coverage when someone **with the target GPU** lands a validated port. There is no full emulator path for WMMA/MFMA channel proof.

If you have hardware hipfire does not yet exercise well, this is the path from “I have a GPU” to a reviewable PR.

## What you need

- Target GPU on Linux with ROCm new enough for your chip (`rocminfo` lists e.g. `gfx1201` or `gfx942`).
- Time for channel tests and, when applicable, required speed-gate / model-level routes on real hardware (yours or a baseline holder’s).
- GitHub **fork** + comfort with `git`, `cargo`, `bash`.
- Optional but useful: an agent that loads **this** skill under `.agents/skills/hipfire-arch-port/` (sole executable skill root).

## What you do not need

- Commit bit on the upstream repo (fork + PR).
- Continuous maintainer pairing — the playbook and source patterns are meant to stand alone.
- An LLM — helpful, not required.

## Non-goals / policy

- **No universal gate.** Pick routes from [`docs/VALIDATION.md`](../../../docs/VALIDATION.md).
- **Retired** `scripts/coherence-gate*.sh` batteries (`coherence-gate.sh` may be absent) **must not be used as current acceptance evidence**. Do not block on them.
- **Implementation ≠ certification.** A merged kernel path is still not a product admission; [`docs/admissions.yml`](../../../docs/admissions.yml) stays empty until earned rows exist.
- Model IDs and crates: [`docs/architecture-ids.md`](../../../docs/architecture-ids.md) — separate from GPU ISA work.
- Hard contribution rules (hooks, DCO, SPDX, branch names): [`CONTRIBUTING.md`](../../../CONTRIBUTING.md).

---

## Workflow

### 1. Fork, clone, and install hooks

1. Fork `https://github.com/warpfront/hipfire` on GitHub (you have no upstream commit bit).
2. Clone **your fork** as `origin`, add upstream for fetch/rebase:

```bash
git clone https://github.com/<you>/hipfire.git
cd hipfire
git remote add upstream https://github.com/warpfront/hipfire.git
git fetch upstream
git checkout -b port/<arch>-<kernel>   # e.g. port/gfx1201-qkv-wmma or port/gfx942-residual-mfma
./scripts/install-hooks.sh             # required; sets core.hooksPath=.githooks
rocminfo | rg -n 'gfx[0-9]+' | head
cargo build --release --features deltanet -p hipfire-runtime --example daemon
# exercise the failing path the issue describes, or a small pull+run smoke
```

If you cannot reproduce a reported crash, comment on the issue with GPU / ROCm / exact command / log — that still helps.

Proactive port: if `rocminfo` shows a chip that only hits a slow fallback or fails codegen, start from [`playbook.md`](playbook.md).

### 2. Read the skill owners

1. [`playbook.md`](playbook.md)
2. [`wmma-matrix.md`](wmma-matrix.md) — then verify builtins on **your** ROCm tree (WMMA for RDNA, MFMA for CDNA)
3. [`docs/methodology/arch-port-validation.md`](../../../docs/methodology/arch-port-validation.md)
4. [`docs/VALIDATION.md`](../../../docs/VALIDATION.md) claim → route map
5. [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) — DCO, SPDX, hooks, PR expectations

Do not assume a single-file `#ifdef` WMMA rename is enough (gfx11 vs gfx12 operand packing). CDNA is MFMA + wave64, not a WMMA rename.

### 3. Start with one kernel you can channel-test

Prefer a fused GEMM with an existing **same-family** sibling over the largest multi-output kernel. Get lane mapping right first; scale later.

Inventory current sources (pick the family you are porting):

```bash
# RDNA / WMMA
ls kernels/src/*wmma* kernels/src/*gfx12* 2>/dev/null | head
rg -n 'has_wmma_w32_gfx12|is_rdna4' crates/rdna-compute/src/gemm.rs | head

# CDNA / MFMA
ls kernels/src/*mfma* kernels/src/*gfx942* 2>/dev/null | head
rg -n 'is_cdna3|is_wave64_native' crates/rdna-compute/src/gemm.rs | head
```

Reference siblings: RDNA `kernels/src/gemm_qkv_hfq4g256_wmma.gfx12.hip`; CDNA `kernels/src/gemm_hfq4g256_residual_mfma.gfx942.hip`.

### 4. Author tagged `.hip` + wire `rdna-compute`

- File forms (one tag segment): `kernels/src/<base_name>.gfxNNNN.hip` (chip) or `kernels/src/<base_name>.gfxNN.hip` (family) — see playbook naming. Examples: `….gfx12.hip`, `….gfx1201.hip`, `….gfx942.hip`.
- **New `.hip` files:** add SPDX header (default Apache-2.0 sole-author template in CONTRIBUTING / governance relicense doc).
- Register: `crates/rdna-compute/src/kernels.rs` (`include_str!`).
- Launch method: `crates/rdna-compute/src/gemm.rs` / related + `ArchCaps` branch (RDNA: `has_wmma*`; CDNA: `is_cdna3` / `is_wave64_native`). More specific arm above broader; remove dead clauses same diff.
- Public entries: keep `bind_thread` discipline (`scripts/verify-bind-thread.sh`).
- **Compile first**, every chip you touch: `./scripts/compile-kernels.sh …` (all selected arches must succeed).
- **Only after** successful compiles, when committing precompiled blobs: `./scripts/write-kernel-hashes.sh` so new `.hsaco` blobs get matching trust sidecars.

Do **not** bury new ISA matrices inside a random `hipfire-arch-*` crate unless the kernel is intentionally crate-local.

### 5. Add channel coverage on the new symbol

Extend `crates/hipfire-runtime/examples/test_kernels.rs` (or a focused example) so the new path is forced on your GPU. A port without numeric coverage on its kernel is not ready.

### 6. Run claim-scoped validation (on hardware)

Use [`docs/VALIDATION.md`](../../../docs/VALIDATION.md) and [`docs/methodology/arch-port-validation.md`](../../../docs/methodology/arch-port-validation.md). Validation is **not optional** for merge of a new ISA route.

**Required for a new/changed numeric GPU-port path:**

```bash
cargo build --release --features deltanet -p hipfire-runtime --example test_kernels
./target/release/examples/test_kernels          # Tier C — on TARGET hardware

# bind invariant (if hooks not installed)
./scripts/verify-bind-thread.sh
```

Then, from VALIDATION / arch-port-validation:

1. **Model-level manual route** for the arch under test after any new/changed numeric `.hip` path (channel alone is incomplete).
2. **Tier P** whenever the change can alter forward/state — run the path-specific parity/oracle for that arch/surface. If **no oracle exists**, record parity as **blocked** (fail closed); do not substitute serve smoke.
3. **Tier S** on **every baseline arch** whose shared edited path/predicate the diff can touch, when a committed `tests/speed-baselines/<arch>.txt` exists:

```bash
# on each applicable baseline box — not optional when the path is shared
# rm the bench binary first if comparing before/after
./scripts/speed-gate.sh --fast
```

**Blocked hardware handoff:** if you cannot run a required route (no target GPU for Tier C, no baseline box for Tier S, no oracle for Tier P), **do not treat it as skipped**. In the PR, name the route **blocked**, what hardware/oracle is missing, and hand it to a holder who can execute it. Do not merge on “should be identical.”

If the change can break forward numbers/state and an oracle **does** exist, run it. User-facing smoke (`scripts/serve_harness.py --model …`) is semantics only **after** numeric routes.

**Retired** coherence-gate scripts **must not be used as current acceptance evidence** or merge criteria.

Perf claims: follow [`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md) (fresh process, warmup, identity hashes). Measured numbers are not admissions.

### 7. Commit (DCO), push branch, open PR

Every commit must use DCO sign-off (`git commit -s`). PRs without sign-off on every commit will be asked to amend. Do not bypass hooks with `--no-verify` unless the maintainer authorized that exact change in writing.

```bash
git add -A
git status   # no debug prints / scratch files
git commit -s -m "port: <arch> <kernel> …"
git push -u origin HEAD
```

Then open a PR from `port/<arch>-<kernel>` on your fork against upstream `master` (or the branch the issue names).

Suggested structure:

```markdown
## What this is
GPU ISA port: gfx<XYZ> (<product name>). Family: RDNA WMMA | CDNA MFMA.

## What it adds
- Kernels: `kernels/src/…` (list; SPDX on new files)
- `crates/rdna-compute/src/kernels.rs` registration
- `ArchCaps` / `gemm.rs` (or related) branches (point at functions)
- Channel coverage: test_kernels / example names

## Hardware tested
- GPU, distro, kernel, ROCm, hipcc/LLVM where relevant
- Tier C channel: PASS/FAIL per new kernel symbol (target GPU)
- Bind check: pass/fail
- Model-level manual route: command + model identity + result (or blocked + reason)
- Tier P oracle: name + result (or blocked — no oracle / no hardware)
- Tier S speed-gate: every baseline arch touched + result (or blocked — no baseline box / no baseline file)
- Any serve routes actually run (name script + model identity) — semantics only

## Scope / not in this PR
- Explicit non-goals (other quant paths, default-on flip, other chips)

## Limitations
- Env gates still off? chip-strict vs family?
- Capability only — not requesting admissions.yml rows unless policy says so
- Blocked routes handed to: <who/hardware>
```

Expect review on lane mapping and routing breadth. Style: `cargo fmt` / clippy clean; one logical change per PR. Branch naming: `port/<arch>-<kernel>` ([CONTRIBUTING.md](../../../CONTRIBUTING.md)).

---

## Working with an agent

```
Read .agents/skills/hipfire-arch-port/ first. Port <gfxNNNN> <WMMA|MFMA> for <kernel>.
I have a <GPU> for channel tests. Retired coherence-gate scripts must not be used as current acceptance evidence.
Select validation from docs/VALIDATION.md (Tier C + model route + Tier P when numbers can change + Tier S on shared baseline arches).
Record blocked routes; do not skip them silently.
Follow CONTRIBUTING.md: hooks, git commit -s, SPDX on new kernels.
```

Useful prompts:

- Walk LDS / operand changes gfx11 → gfx12 (or MFMA layout on gfx942) for a named kernel; no code yet.
- Cite ROCm header evidence for the C-mapping / accumulator hypothesis.
- Channel-test failed with `expected … got …` at index N — derive lane mapping.

Guardrails:

- No `--no-verify` to skip hooks without maintainer written OK for that change.
- No treating serve harness green as numeric proof.
- No inventing a universal replacement gate.
- No treating Tier S / model-level / Tier P as optional when VALIDATION says they apply.
- Check `git status` before commit so debug prints and scratch files stay out.
- Sole executable skill root is `.agents/skills/` — do not add parallel skill trees.

---

## Communication

- Issues: https://github.com/warpfront/hipfire/issues
- Ping maintainer when: direction check before a large port, PR ready, channel-test wall after self-debug, or blocked Tier S/C handoff needs a baseline holder.
- Prefer issue-thread progress notes over pings for routine status.

## Provenance (historical, not procedure)

Past sessions documented: issue #54 codegen crash class on 9070 XT; stale-binary speed-gate false regression; early gfx12 WMMA pattern file; family-tag resolution in `compile-kernels.sh`. Current default-on status of any path is **whatever source + env gates say today** — re-read `crates/rdna-compute/src/kernels.rs` / `gemm.rs` / feature flags rather than this paragraph.

External RDNA4 / CDNA (and other) hardware remains the source of truth for channel data on newly routed symbols.

## Bar

Gates and harnesses exist because each class caught real bugs. PRs that skip channel proof on the target ISA, hand-wave C-mapping/MFMA layout, treat applicable Tier S or model-level routes as optional, or claim product certification from a single smoke run do not merge cleanly.

Welcome aboard.
