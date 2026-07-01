# TODO: bug-reporting / crash-diagnostic capture

Status: TODO (idea, not yet approved for build).
Date: 2026-07-01

## Motivation

Debugging hipfire failures currently means scraping ad-hoc logs after the fact,
and the most painful failures are transient and context-dependent — by the time
you look, the GPU state, the failing op, and the surrounding log are gone. Recent
examples from one afternoon of H-Neurons work:

- **GPU wedges** — `HipError 719 "unspecified launch failure"`, `700 "illegal
  memory access"`, sticky-719 in `sample_top_p` / `decode_loop` / `prefill`.
  Transient, hardware/driver-adjacent (nix2 LDS hazard → MES hang → reset). No
  captured context: which kernel, dispatch dims, kernarg, seq pos, model, arch.
- **Correctness bugs that only bite at scale** — KV state accumulating across
  daemon `generate` requests until `pos >= max_seq` at ~request 128. The early
  symptom (silently polluted context) was invisible; the late symptom (a hard
  error) had no breadcrumb back to the cause.
- **Tooling gaps surfaced as opaque load errors** — `failed to parse
  Gemma3Config` (source config omits `vocab_size`), `tensor not found:
  model.embed_tokens.weight` (VL `language_model.` prefix). Actionable, but only
  after manual spelunking.

A first-class capture path would turn each of these into a self-contained report
with enough context to triage without a live repro.

## Goal

On a qualifying failure, write a **self-contained report bundle** to a known dir
(`$HIPFIRE_BUG_DIR`, default `~/.hipfire/bugs/<utc-stamp>-<short-hash>/`) and
print its path. Plus a manual `hipfire bug` subcommand for user-filed reports and
for attaching to an already-captured bundle.

Non-goals: no network upload by default (local-only; opt-in export is a separate,
explicitly-authorized step — this is diagnostic data that may contain prompts).
No Python in the capture path (it must work when the runtime is wedged).

## What a report captures

Bundle = a dir with:

- `report.json` — structured envelope:
  - build info (`hipfire-build-info`: commit, dirty, profile, features)
  - trigger (`panic` | `hip_error` | `gate_fail` | `manual`) + one-line summary
  - GPU/accel: arch (gfx…), device, HIP/ROCm version, VRAM free/total
  - the failing op: subsystem (daemon/quantize/eval), request `type`, model
    artifact (path + arch_id + quant format), seq pos / batch / layer if known
  - the HipError code + message, or the panic message + location
  - `hipfire lock status` snapshot (holder, epoch)
- `backtrace.txt` — captured with `RUST_BACKTRACE` semantics (a panic hook +
  an explicit capture on the HipError path).
- `gpu.txt` — `rocm-smi` snapshot + the tail of `dmesg | grep amdgpu` (ring
  reset / page fault / MES timeout lines) when readable. This is what actually
  distinguishes a driver wedge from a logic bug.
- `log.txt` — the last N KB of the component's log ring (see below).
- `env.txt` — `HIPFIRE_*` env (values redacted to presence/shape), CPU/mem.
- `repro.md` — best-effort: the command line (argv), and for daemon ops the
  request JSON with prompt/response bodies **redacted by default** (length +
  hash, not content).

## Shapes

1. **Auto-capture hook (library).** A small `hipfire-bugreport` crate (or fold
   into `hipfire-sysinfo`, which already gathers host/accel info) exposing
   `capture(trigger, ctx) -> PathBuf`. Wire it at:
   - a process-wide **panic hook** (daemon, quantize, eval binaries),
   - the **HipError boundary** — where `HipError` is converted to a user-facing
     error / emitted as a daemon `error` event, call `capture` first when the
     code is in the wedge set (719/700/other launch failures),
   - **gate failures** — the coherence / tiny-quant / speed gates already write
     reports; have them drop a bug bundle on hard-fail so CI artifacts are
     uniform.
2. **`hipfire bug` CLI** (in `hipfire-cli`):
   - `hipfire bug report --title "<...>" [--attach <path>...]` — manual bundle
     (grabs the same GPU/build/env snapshot + user note).
   - `hipfire bug list` / `hipfire bug show <id>` — browse local bundles.
   - `hipfire bug export <id> --to <path>` — tar a bundle for sharing; this is
     the **only** step that surfaces raw prompt/response content, and it must
     prompt / require an explicit `--include-payloads` flag (outward-facing;
     confirm before it leaves the box).

## Log ring prerequisite

`log.txt` needs a bounded in-memory ring the capture path can dump. `logging.rs`
already initializes the subscriber; add a ring-buffer layer (last ~256 KB) that
`capture()` snapshots. Without it, `log.txt` falls back to "not available".

## Redaction / safety

- Default to **content-free**: prompts, responses, tool args → length + xxh64
  hash, never raw bytes, in any auto-captured bundle.
- Raw payloads only via explicit `--include-payloads` on `export`, with a
  confirm — bug bundles are diagnostic, not a data exfiltration path.
- Env values redacted to presence/shape (`HIP_VISIBLE_DEVICES=set`, not the
  value) except a known-safe allowlist (arch, profile flags).

## Open questions

- Crate placement: new `hipfire-bugreport` vs extend `hipfire-sysinfo`. Leaning
  sysinfo for the snapshot primitives + a thin bugreport layer for the bundle
  format, so the daemon doesn't grow a new heavy dep.
- Wedge-set definition: which HipError codes auto-trigger. Start with {719, 700,
  and any "launch failure" / "illegal memory access" / "hipMalloc" message},
  matching the `looks_like_wedge` heuristic already in the hneurons probe —
  factor that into the shared crate so probe and capture agree.
- Cost/gating: auto-capture must be near-zero on the happy path (only fires on
  panic/HipError), and must not itself allocate GPU memory (the GPU may be
  wedged) — snapshot via host-side `rocm-smi`/`dmesg`, not HIP calls.
- Retention: cap `~/.hipfire/bugs/` size; prune oldest.

## First increment

`hipfire-sysinfo` snapshot primitive (arch/HIP/VRAM/dmesg-amdgpu/rocm-smi) +
`report.json` + a panic hook in the daemon that writes a bundle. That alone
converts every daemon panic into a triage-ready artifact; the HipError-boundary
hook and the `hipfire bug` CLI follow.
