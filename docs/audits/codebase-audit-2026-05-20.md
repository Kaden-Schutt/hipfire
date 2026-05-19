# Codebase Audit - 2026-05-20

Scope: live checkout audit of code, scripts, and documentation in `hipfire-uma-loader`.

Validation snapshot at audit time:
- `bun test cli/parse_tool_calls.test.ts`: passed, 18 tests.
- `bun install --frozen-lockfile` followed by `bunx tsc -p cli/tsconfig.json --noEmit --pretty false`: passed.
- `cargo check --workspace --all-targets`: passed with warnings.

## Critical Fixes Needed

### 1. Public unauthenticated HTTP bind

- Location: `cli/index.ts` (`DEFAULT_HOST`, `Bun.serve`), `docs/SERVE.md`
- Severity: High
- Description: Code defaults to `0.0.0.0:11435` and exposes inference endpoints without authentication, while docs describe `localhost:11435`.
- Impact: On a LAN or workstation with exposed ports, other clients can submit inference work, consume GPU/VRAM, enumerate local models, and cause denial of service.
- Recommended fix:

```ts
const DEFAULT_HOST = "127.0.0.1";
const API_KEY = process.env.HIPFIRE_API_KEY ?? "";

function authFailure(req: Request): Response | null {
  if (!API_KEY) return null;
  if (req.headers.get("authorization") !== `Bearer ${API_KEY}`) {
    return Response.json({ error: "unauthorized" }, { status: 401 });
  }
  return null;
}

// First line inside the serve fetch handler:
const denied = authFailure(req);
if (denied) return denied;
```

Documentation text to add:

> The default bind is `127.0.0.1:11435`. Use `hipfire serve 0.0.0.0 11435` only on trusted networks, preferably with `HIPFIRE_API_KEY` set.

### 2. DFlash unload leaks GPU allocations

- Location: `crates/hipfire-runtime/examples/daemon.rs`, `crates/hipfire-arch-qwen35/src/speculative.rs`
- Severity: High
- Description: `unload_model` frees DFlash draft weights and draft scratch, but comments state that ring, snapshot, tape, and verify scratch buffers leak until daemon exit. Some backing types already expose `free_gpu`, but `DeltaNetSnapshot`, `HiddenStateRingBuffer`, and aggregate `DflashState` do not.
- Impact: Long-running `hipfire serve` sessions that swap DFlash models can retain VRAM and later fail loads.
- Recommended fix:

```rust
impl DeltaNetSnapshot {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for b in self.s_matrix_bufs { let _ = gpu.hip.free(b); }
        for b in self.s_scale_bufs { let _ = gpu.hip.free(b); }
        for b in self.conv_state_bufs { let _ = gpu.hip.free(b); }
    }
}

impl HiddenStateRingBuffer {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self.layer_bufs { let _ = gpu.free_tensor(t); }
        for t in self.staging_bufs { let _ = gpu.free_tensor(t); }
    }
}

impl DflashState {
    fn free_gpu(self, gpu: &mut rdna_compute::Gpu) {
        self.draft_weights.free_gpu(gpu);
        self.draft_scratch.free_gpu(gpu);
        self.hidden_rb.free_gpu(gpu);
        self.verify_scratch.free_gpu(gpu);
        self.target_snap.free_gpu(gpu);
        self.gdn_tape.free_gpu(gpu);
        if let Some(dd) = self.ddtree { dd.free_gpu(gpu); }
    }
}
```

Then replace the partial DFlash free in `unload_model` with `df.free_gpu(gpu)`.

### 3. Unbounded request queue and retained model cache

- Location: `cli/index.ts` (`queue`, `acquireLock`), `crates/hipfire-runtime/examples/daemon.rs` (`retained_models`)
- Severity: High if bound publicly, Medium otherwise
- Description: HTTP requests wait in an unbounded in-memory queue. `unload_policy=memory_pressure` can retain multiple full models until a later pressure failure.
- Impact: A small number of concurrent clients can create unbounded latency and memory/VRAM pressure.
- Recommended fix:

```ts
const MAX_PENDING_REQUESTS = 8;

async function acquireLock(): Promise<boolean> {
  if (!busy) { busy = true; return true; }
  if (queue.length >= MAX_PENDING_REQUESTS) return false;
  await new Promise<void>(resolve => queue.push({ resolve }));
  busy = true;
  return true;
}

if (!(await acquireLock())) {
  return Response.json({ error: "server busy" }, { status: 503 });
}
```

For model retention, add an explicit `max_retained_models` setting and LRU-evict retained entries by calling `unload_model`.

## Technical Debt and Refactoring

### 4. Tool-call parser is duplicated instead of imported by tests

- Location: `cli/index.ts`, `cli/parse_tool_calls.test.ts`
- Severity: Medium
- Description: Tests duplicate production parser logic because importing `index.ts` triggers CLI top-level side effects.
- Impact: Parser tests can pass while production behavior drifts.
- Recommended fix:

```ts
// cli/tool_calls.ts
export function parseToolCalls(text: string) { /* existing implementation */ }
export function parseOneToolCall(raw: string) { /* existing implementation */ }

// cli/index.ts
import { parseToolCalls } from "./tool_calls.ts";
```

Move command dispatch behind `if (import.meta.main)` so helpers can be imported safely.

### 5. Malformed historical `tool_calls` can throw during prompt reconstruction

- Location: `cli/index.ts`, legacy inline ChatML construction
- Severity: Medium
- Description: The structured message path catches malformed JSON arguments; the legacy prompt path uses unguarded `JSON.parse(fn.arguments || "{}")`.
- Impact: One malformed historical assistant tool call can produce HTTP 500.
- Recommended fix:

```ts
const parseArgs = (v: any) => {
  if (typeof v !== "string") return v ?? {};
  try { return JSON.parse(v || "{}"); }
  catch { return { _raw: v }; }
};

text += `\n<tool_call>\n${JSON.stringify({
  name: fn.name ?? "unknown",
  arguments: parseArgs(fn.arguments),
})}\n</tool_call>`;
```

### 6. Large monolithic modules are expensive to audit

- Location: `cli/index.ts`, `crates/rdna-compute/src/dispatch.rs`, `crates/hipfire-runtime/examples/daemon.rs`
- Severity: Medium
- Description: `cli/index.ts` is over 5k lines, `dispatch.rs` is near 20k lines, and `daemon.rs` is over 4k lines.
- Impact: Review cost stays high, ownership boundaries are unclear, and repeated "keep in sync" patterns accumulate.
- Recommended fix: split `cli/index.ts` into `config.ts`, `serve.ts`, `pull.ts`, `tool_calls.ts`, and `quantize.ts`; split RDNA dispatch by operation family such as GEMV, GEMM, attention, graph capture, cache management, and memory utilities.

### 7. Model pull path lacks integrity verification

- Location: `cli/index.ts`, `cli/registry.json`, `AGENTS.md`
- Severity: Medium
- Description: AGENTS includes manual md5s, but the registry has no checksum fields and `pull()` accepts downloaded bytes without digest verification.
- Impact: Corrupt or tampered artifacts are accepted silently.
- Recommended fix: add `sha256` fields to registry entries and verify before final rename.

```ts
const hasher = new Bun.CryptoHasher("sha256");
for await (const chunk of res.body as AsyncIterable<Uint8Array>) {
  hasher.update(chunk);
  writer.write(chunk);
}
const digest = hasher.digest("hex");
if (entry.sha256 && digest !== entry.sha256) {
  unlinkSync(tmpDest);
  throw new Error(`checksum mismatch for ${entry.file}`);
}
```

## Documentation Updates

### 8. AGENTS.md is release-stale

- Location: `AGENTS.md`, `Cargo.toml`, `README.md`
- Severity: Medium
- Description: AGENTS says v0.1.9-alpha, while the workspace and README identify v0.1.20.
- Impact: Agents may follow an older testing surface after modularization.
- Recommended text: "This guide targets hipfire v0.1.20. MQ3/DFlash guidance below is retained from v0.1.9, but crate paths follow the v0.1.20 modular layout."

### 9. Documented API endpoints do not match server behavior

- Location: `docs/SERVE.md`, `cli/index.ts`
- Severity: Medium
- Description: Docs claim `POST /v1/completions`; code only accepts `POST /v1/chat/completions`. Docs claim `/health` returns 503 during load; code returns 200 JSON while the HTTP server is responsive.
- Impact: OpenAI-compatible clients using completions fail, and readiness automation can make wrong assumptions.
- Recommended text:

> Implemented endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`. `/v1/completions` is not implemented. `/health` returns 200 when the HTTP server is responsive; inspect `model` for load state.

### 10. Config default docs contradict code

- Location: `docs/GETTING_STARTED.md`, `docs/CONFIG.md`, `cli/index.ts`
- Severity: Low
- Description: `GETTING_STARTED.md` says `dflash_mode` defaults to `auto`; code defaults to `off`. `CONFIG.md` says `default_model` default is empty; code uses `qwen3.5:9b`.
- Impact: New users misconfigure DFlash and prewarm behavior.
- Recommended text: "Common overrides: `temperature` default `0.30`, `kv_cache` arch-dependent, `dflash_mode` default `off`." Change the server table to `default_model | qwen3.5:9b`.

### 11. CLI docs advertise unsupported commands

- Location: `docs/CLI.md`, `docs/MODELS.md`, `cli/index.ts`
- Severity: Low
- Description: Docs mention `hipfire config view` and `hipfire config set-model ...`; parser supports `list|get|set|reset|cask-profile` and per-model shape `hipfire config <tag> set ...`.
- Impact: Copy-paste onboarding failures.
- Recommended text: Replace `hipfire config view` with `hipfire config list`; replace `hipfire config set-model qwen3.6:35b-a3b max_think_tokens 1024` with `hipfire config qwen3.6:35b-a3b set max_think_tokens 1024`.

## Validation Run: 2026-05-20

### Passed

- `cargo check --workspace --all-targets`
- `cargo test --workspace --all-targets --no-run`
- `cargo test --workspace --lib --bins --tests -- --nocapture`
  - 334 tests passed.
  - 1 test ignored: `crates/hipfire-arch-qwen35/tests/pp_parity.rs`.
- `bun install --frozen-lockfile` in `cli/`
- `bunx tsc -p cli/tsconfig.json --noEmit --pretty false`
- `bun test cli/parse_tool_calls.test.ts cli/chat_pure.test.ts`
  - 121 tests passed.
- Shell syntax: 80 shell scripts passed `bash -n`.
- ShellCheck: 80 shell scripts checked; 141 findings reported.
  - 84 warnings, 45 info findings, 12 style findings.
  - Most common warning classes: unused variables (`SC2034`, 27), unchecked `cd` (`SC2164`, 26), declare-and-assign masking exit status (`SC2155`, 13), and trap strings expanding too early (`SC2064`, 11).
- Python syntax: 41 Python files passed `python3 -m py_compile`.

### Blockers and Warnings Found

- `cargo fmt --all -- --check` fails. Rustfmt reports formatting diffs across the workspace; the captured check output was 88,819 lines. The first diffs are in `crates/hip-bridge/examples/kernel_launch.rs`, `crates/hip-bridge/examples/peer_smoke.rs`, `crates/hip-bridge/examples/smoke.rs`, and `crates/hip-bridge/src/ffi.rs`.
- `cargo clippy --workspace --all-targets -- -W warnings` fails on `crates/hsa-bridge/src/lib.rs:700` with `clippy::not_unsafe_ptr_arg_deref`. The public function dereferences a raw pointer and should either be marked `unsafe` or changed to avoid exposing a raw-pointer dereference contract through a safe API.
- Re-running Clippy with `-A clippy::not_unsafe_ptr_arg_deref` completes and reports a large warning backlog. Main themes:
  - Unused imports, variables, fields, constants, and dead helper paths across `hipfire-runtime`, `hipfire-quantize`, `hsa-bridge`, `rdna-compute`, `redline`, and arch crates.
  - Mechanical modernization opportunities such as `.div_ceil()`, `.is_multiple_of()`, `is_some_and`, `contains`, `Default` implementations for `new()`, and avoiding needless casts/borrows.
  - Complexity warnings in large runtime examples, especially `daemon.rs`, `dflash_spec_demo.rs`, and `profile_layers.rs`.
  - FFI/resource warnings such as `std::mem::forget` on non-`Drop` wrapper types in `crates/hip-bridge/src/ffi.rs`.
- Python lint and type checks were run from `/home/sadara/.venv/bin` because those tools were not on the non-interactive shell `PATH`.
  - `ruff 0.15.13`: `ruff check` fails with 57 issues; 47 are auto-fixable. Main codes: `F541` f-string without placeholders (30), `F401` unused imports (10), `E401` multiple imports on one line (6), `E702` multiple statements on one line (5), `F841` unused local variables (2), `E402` imports not at top of file (2), `E741` ambiguous variable name (1), `E731` assigned lambda (1).
  - `mypy 2.1.0`: `mypy` fails with 27 errors in 13 files. Main clusters are `scripts/mq4_masked_calib.py` typing/object-shape errors, missing `dflash.model` imports in DFlash diagnostic scripts, missing stubs for `datasets`, and unsafe `module_from_spec` handling in `tests/test_kernel_atlas.py`.
  - `pylint 4.0.5`: `pylint` exits 30 and reports 1,246 messages, rated 8.65/10. Category counts: 952 convention, 177 refactor, 109 warning, 8 error. Main codes: missing function docstrings (`C0116`, 473), long lines (`C0301`, 316), too many locals (`R0914`, 68), import outside toplevel (`C0415`, 64), invalid names (`C0103`, 50), f-string without interpolation (`W1309`, 30), duplicate code (`R0801`, 29), and too many arguments (`R0913`, 27).
- PowerShell syntax was not checked because neither `pwsh` nor `powershell` is installed in the environment.

## Second-Pass Warning Audit

This pass inspected warning sites in source context to distinguish cosmetic lint from deeper defects or code smells.

### A. KFD AQL queue teardown leaks mapped allocations

- Location: `crates/redline/src/kfd.rs`, `AqlQueue::new`, `AqlQueue::destroy`
- Severity: High
- Description: Clippy's unread-field warnings on `ring_handle`, `eop_handle`, and `signal_handle` point at a real resource-lifecycle problem. `AqlQueue::new` allocates and maps ring, EOP, signal, CWSR, write-pointer, and read-pointer userptr memory plus an mmap'd doorbell page. `AqlQueue::destroy` only destroys the queue and closes the KFD fd. It does not unmap/free KFD allocations, unmap userptr CPU memory, or unmap the doorbell page. CWSR, wptr, and rptr handles are not stored at all, so they cannot be released later.
- Impact: Repeated AQL smoke tests or long-running Redline use can leak host mappings, KFD allocation handles, and kernel mappings. This is exactly the kind of issue an "unused field" warning can hide.
- Recommended fix:

```rust
struct KfdMappedUserAlloc {
    handle: u64,
    cpu_ptr: *mut u8,
    size: usize,
}

pub struct AqlQueue {
    kfd_fd: i32,
    queue_id: u32,
    doorbell_page: *mut libc::c_void,
    doorbell_len: usize,
    ring: KfdMappedUserAlloc,
    eop: KfdMappedUserAlloc,
    signal: KfdMappedUserAlloc,
    cwsr: KfdMappedUserAlloc,
    wptr: KfdMappedUserAlloc,
    rptr: KfdMappedUserAlloc,
}

impl Drop for AqlQueue {
    fn drop(&mut self) {
        self.destroy_inner();
    }
}
```

Add KFD unmap/free ioctls and make `destroy_inner` idempotent. Store all allocation handles created in `new`, not only the subset used during dispatch.

### B. Safe public raw-pointer packet builders expose undefined behavior

- Location: `crates/hsa-bridge/src/lib.rs`, `build_dispatch_packet`, `publish_dispatch_packet`
- Severity: High
- Description: `build_dispatch_packet` and `publish_dispatch_packet` are public safe functions that dereference raw packet pointers. A caller can pass null, misaligned, dangling, or aliased pointers and trigger undefined behavior through a safe API.
- Impact: This is a soundness issue in the Rust API boundary, not cosmetic Clippy noise.
- Recommended fix:

```rust
pub unsafe fn build_dispatch_packet(
    slot: NonNull<HsaKernelDispatchPacket>,
    kernel: &HsaKernel,
    grid: [u32; 3],
    block: [u32; 3],
    kernarg_ptr: NonNull<u8>,
    completion_signal: HsaSignalHandle,
) {
    let p = unsafe { slot.as_ptr().as_mut().unwrap_unchecked() };
    // fill packet fields
}
```

Alternatively keep a safe function but accept `&mut HsaKernelDispatchPacket` and make callers perform the raw-pointer conversion at the unsafe boundary.

### C. HIP handle wrappers use manual ownership and no-op `mem::forget`

- Location: `crates/hip-bridge/src/lib.rs`, `DeviceBuffer`; `crates/hip-bridge/src/ffi.rs`, `Stream`, `Event`, `Graph`, `GraphExec`, `free`, `stream_destroy`, `event_destroy`, `graph_destroy`, `graph_exec_destroy`
- Severity: Medium
- Description: Clippy's `forget_non_drop` warnings are correct: these wrappers do not implement `Drop`, so `std::mem::forget` has no effect. The current design is manual-destroy-only. That is acceptable for some FFI layers, but the comments imply RAII/double-free semantics that do not exist.
- Impact: Any early return, panic, or dropped `DeviceBuffer`/stream/event/graph without an explicit destroy leaks GPU resources. The misleading comments make that easier to miss during review.
- Recommended fix: either make the types explicitly non-owning/raw and remove the `forget` comments, or introduce owning RAII handles tied to the runtime.

```rust
pub struct DeviceBuffer {
    ptr: *mut c_void,
    size: usize,
    hip: Arc<HipRuntimeInner>,
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { (self.hip.fn_free)(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
```

### D. Stale DFlash diagnostic scripts are not portable

- Location: `scripts/dflash_ref_spec_test.py`, `scripts/dflash_spec_debug.py`, `scripts/dflash_train_poc.py`, related `dflash_diag_*` scripts
- Severity: Medium
- Description: mypy's missing `dflash.model` errors point to real environment coupling. Some scripts insert repo-relative `.dflash-reference`, while others hardcode `/root/hipfire` and `/root/.cache/...` paths.
- Impact: Diagnostics may work only on one historical machine layout. New users or agents can get import failures or silently test the wrong reference checkout.
- Recommended fix:

```python
REPO_ROOT = Path(__file__).resolve().parents[1]
DFLASH_REF = Path(os.environ.get("HIPFIRE_DFLASH_REFERENCE", REPO_ROOT / ".dflash-reference"))
if not (DFLASH_REF / "dflash" / "model.py").exists():
    raise SystemExit(f"missing DFlash reference at {DFLASH_REF}")
sys.path.insert(0, str(DFLASH_REF))
```

Use an env override for cache/model paths and remove hardcoded `/root/hipfire` references.

### E. MQ4 calibration helper uses overly loose object-shaped records

- Location: `scripts/mq4_masked_calib.py`, `selected_iterate_targets`, `load_round_hessians`, `write_round_summary`, related helpers
- Severity: Medium
- Description: Most mypy errors in this file come from treating structured records as `dict[str, object]`, then indexing `object` values as if they were strings, arrays, or nested dicts.
- Impact: The script is currently relying on runtime shape assumptions. A malformed mask/stats record can fail late or produce confusing errors in long calibration runs.
- Recommended fix:

```python
from typing import TypedDict

class MaskTensor(TypedDict):
    hfq_name: str
    packable_flat_mq4: bool
    numel: int
    base_data_size: int

class RoundBench(TypedDict, total=False):
    status: str
    output: str
    metrics: dict[str, float | None]
```

Parse external JSON into these shapes at the boundary and validate required keys before computation.

### F. Shell command construction relies on fragile word splitting

- Location: `scripts/bench-matrix.sh`, `scripts/pflash-gate.sh`, `scripts/pflash-niah-bench.sh`, `scripts/bisect-test.sh`
- Severity: Medium
- Description: Several scripts build a string such as `extra="$extra --pflash $DRAFTER --keep-ratio ..."` and later expand it unquoted. ShellCheck flags this as `SC2086`; in this repo it is often intentional to split flags, but it becomes unsafe when any path or argument contains whitespace or shell glob characters.
- Impact: Bench harnesses can pass the wrong argv, expand globs, or fail only on particular paths.
- Recommended fix:

```bash
extra=()
if [ "$PRETOK" = "1" ]; then extra+=(--pretok); fi
if [ -n "$DRAFTER" ]; then
  extra+=(--pflash "$DRAFTER" --keep-ratio "$KEEP_RATIO" --block-size "$BLOCK_SIZE")
fi
out=$("$EXE" "$TARGET" "$FIXTURE" --maxgen "$MAXGEN" "$KV_MODE" "${extra[@]}" 2>&1)
```

### G. Trap cleanup expands variables at registration time

- Location: `tests/e2e_bump_reload.sh`, `tests/e2e_kv_budget.sh`, `tests/e2e_kv_reject.sh`, `tests/e2e_run_reject.sh`
- Severity: Medium
- Description: ShellCheck's `SC2064` findings are not purely style. Double-quoted trap bodies expand `$PID`, `$LOG`, and `$TMPCFG` when the trap is registered, not when it runs. These scripts currently set variables before registering the trap, so the present behavior is probably stable, but it is brittle and unsafe for future edits.
- Impact: Cleanup can target stale paths/PIDs after refactors, and unquoted temp paths would break if they ever contained whitespace.
- Recommended fix:

```bash
cleanup() {
  kill -TERM "${PID:-}" 2>/dev/null || true
  wait "${PID:-}" 2>/dev/null || true
  rm -rf -- "${TMPCFG:-}" "${LOG:-}"
}
trap cleanup EXIT
```

### H. Reversed redirection drops stderr from a reported artifact

- Location: `scripts/quant_cohort.sh`, per-tensor MSE step
- Severity: Medium
- Description: ShellCheck's `SC2069` is a real behavior bug: `cmd 2>&1 > file` sends stderr to the original stdout, not to `file`.
- Impact: The script says output is captured, but failure diagnostics on stderr may be missing from `${PV}.mse.txt`.
- Recommended fix:

```bash
./target/release/examples/quant_quality_mse "$ST_DIR" "$HFQ_PATH" >"${PV}.mse.txt" 2>&1
```

### I. Identical branch warnings mark stale conditions

- Location: `crates/hipfire-quantize/src/bin/dflash_convert.rs`, `crates/hipfire-quantize/src/main.rs`
- Severity: Low to Medium
- Description: Some `if_same_then_else` Clippy findings are harmless readability issues, such as norm tensors and `--keep-f32` both choosing F32. The HFQ3 path in `main.rs`, however, checks `k_dim % 128 == 0` and then executes the same quantization branch either way.
- Impact: The check no longer enforces or documents a real constraint, so future maintainers cannot tell whether padding, validation, or a fallback path was intended.
- Recommended fix: remove dead conditionals when they are intentionally equivalent, or make constraints explicit:

```rust
if !k_dim.is_multiple_of(128) {
    return Err(anyhow!("HFQ3G128 requires K divisible by 128, got {k_dim}"));
}
```

### J. Rustfmt failure is maintainability debt, not a hidden correctness bug

- Location: workspace-wide; `cargo fmt --all -- --check`
- Severity: Low
- Description: Rustfmt reports diffs in 248 files. The inspected diffs are formatting-only.
- Impact: Large unformatted areas increase review noise and make semantic changes harder to isolate.
- Recommended fix: run `cargo fmt --all` on a dedicated formatting-only commit or PR so behavioral changes remain reviewable.
