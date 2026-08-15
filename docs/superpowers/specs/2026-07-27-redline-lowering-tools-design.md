# Redline Lowering Tools Design

**Status:** Implemented interim developer tooling (design + thin wrapper landed)
**Lifecycle:** `active` (interim; not a stable product surface)
**Allowed claim states:** Interim developer-only tooling; no stable UX or compatibility promise; product/runtime/performance claims remain disclaimed
**Canonical domain:** Developer CLI discoverability for existing Radiowave kernel lowering and Redline retained-PM4 lowering surfaces
**Last checked:** 2026-07-28

This document uses **MUST**, **MUST NOT**, **SHOULD**, and **MAY** normatively. It describes the thin Python delegation layer under `python3 -m tools.redline lower` as **interim developer-only tooling**. The wrapper exists for contributor discoverability; it is **not** a stable product UX, public API, or compatibility contract. Product, runtime, and performance claims remain owned by Radiowave, the daemon/retained-PM4 path, `docs/REDLINE.md`, and `docs/VALIDATION.md`—this surface MUST NOT be cited for those claims.

Python is an interim discoverability and delegation shim only—not a stable layer. Compiler and lowering logic remain in the existing Rust and runtime surfaces (`radiowave`, daemon/retained-PM4 path via `scripts/redline_daemon_harness.py`). This design MUST NOT duplicate lowering algorithms or move Rust sources.

## Motivation

Contributors already lower kernels with the `radiowave` binary and exercise retained-PM4 lowering through `scripts/redline_daemon_harness.py --pm4`. Those engines are correct ownership boundaries; what is missing is a single discoverable entry under the existing `tools.redline` package.

Without a unified surface:

- kernel compile/inspect/oracle workflows and PM4 harness runs are found only by institutional knowledge;
- argv conventions and failure modes diverge across ad-hoc wrappers;
- new tooling risks re-implementing Radiowave or PM4 lowering in Python.

The fix is thin subprocess delegation: expose both engines under `python3 -m tools.redline lower` without changing their semantics.

## Goals

1. Add `tools/redline/lower.py` as the implementation module for lowering subcommands.
2. Expose two commands under the unified namespace:
   - `python3 -m tools.redline lower kernel [Radiowave args...]`
   - `python3 -m tools.redline lower pm4 [Redline harness args...]`
3. Extend the existing `tools.redline` dispatcher (`tools/redline/__main__.py`) later so `lower` is a first-class peer of `golden`, `bench`, and `serve-diff`.
4. Resolve the Radiowave binary with a fixed preference order (see Binary and script resolution).
5. Run PM4 lowering via the current Python interpreter on `scripts/redline_daemon_harness.py`, enforcing `--pm4` per the zero/one/many contract below, and otherwise preserving caller argv byte-for-byte and position-for-position when exactly one `--pm4` is already present.
6. Pass through child stdout, stderr, and exit status on successful process start.
7. Exit `2` with a clear error naming the attempted command when a required surface cannot be resolved or started.
8. Keep tests process-mocked: unit tests mock process execution; a smoke path exercises delegated help without requiring a GPU or a built daemon.

## Non-goals

- Duplicate Radiowave compile, inspect, oracle, recipe, campaign, or assessment logic in Python.
- Move, rewrite, or re-home Rust sources under `crates/radiowave` or runtime PM4 lowering.
- Replace `scripts/redline_daemon_harness.py`, `python3 -m tools.redline golden`, or `python3 -m tools.redline bench`.
- Change retained-route certification policy, Golden floors, admissions, or product defaults (`docs/REDLINE.md`, `docs/VALIDATION.md`).
- Add new Radiowave subcommands, new harness flags, or a second PM4 lowering implementation.
- Require formatters, linters, or full-suite GPU validation as part of this design’s acceptance.
- Bundle kernel compile and PM4 harness into one mixed command or shared flag parser.

## CLI surface

### Dispatcher

Today:

```text
python3 -m tools.redline {golden|bench|serve-diff} ...
```

After the dispatcher extension (implementation follow-on to `lower.py`):

```text
python3 -m tools.redline {golden|bench|serve-diff|lower} ...
```

`lower` without a mode MUST print a short usage to stderr and exit `2` (same family as unknown/`-h` handling in `__main__.py`, except bare `lower` is not success help).

Mode help:

```text
python3 -m tools.redline lower -h
python3 -m tools.redline lower --help
python3 -m tools.redline lower kernel -h
python3 -m tools.redline lower pm4 -h
```

`-h` / `--help` on `lower` itself SHOULD document the two modes and that args after the mode are forwarded. Mode-specific help SHOULD prefer delegating to the child (`radiowave -h` / harness `-h`) once resolution succeeds, so flag truth stays with the engine.

### `lower kernel`

```bash
python3 -m tools.redline lower kernel [Radiowave args...]
```

Forwards `[Radiowave args...]` verbatim to the resolved Radiowave command prefix. No Python-side flag parsing of Radiowave options.

Examples:

```bash
# compile (args are Radiowave's)
python3 -m tools.redline lower kernel compile \
  --source kernels/src/example.hip \
  --output /tmp/example.hsaco \
  --arch gfx1201

# inspect
python3 -m tools.redline lower kernel inspect \
  --input /tmp/example.hsaco \
  --arch gfx1201

# explicit binary override
python3 -m tools.redline lower kernel --radiowave /path/to/radiowave compile \
  --source kernels/src/example.hip \
  --output /tmp/example.hsaco \
  --arch gfx1151

# delegated help
python3 -m tools.redline lower kernel -h
```

`--radiowave PATH` is consumed only by the Python resolver and MUST appear immediately after `kernel`; it MUST NOT be forwarded to Radiowave. Treating later occurrences as wrapper flags is deliberately out of scope so the wrapper never scans or reinterprets child-engine arguments.

### `lower pm4`

```bash
python3 -m tools.redline lower pm4 [Redline harness args...]
```

Runs:

```text
<current python> scripts/redline_daemon_harness.py [normalized harness args containing exactly one --pm4]
```

working directory = repository root (same root `tools.redline` already assumes).

`--pm4` is mandatory for this mode. Count exact argv tokens equal to `--pm4` among the caller’s forwarded harness args (no scanning inside other flags’ values). Behavior:

- **Zero** `--pm4`: insert exactly one `--pm4` immediately after the script path (before all forwarded args).
- **Exactly one** `--pm4`: preserve the entire caller argument list byte-for-byte and position-for-position—do not move, strip, or re-insert `--pm4`; spawn with caller args unchanged after the script path.
- **More than one** `--pm4`: reject at the wrapper with exit `2`, a stderr diagnostic prefixed `tools.redline.lower:` that contains `multiple --pm4`, and **no child spawn**.

The wrapper MUST NOT strip or rewrite other harness flags (`--model`, `--daemon`, `--prefix`, `--kv-mode`, etc.) except for the zero-`--pm4` insertion above. Missing-harness attempted-command diagnostics MUST describe the same normalized argv that would have been spawned (including inserted `--pm4` when applicable).

Examples:

```bash
python3 -m tools.redline lower pm4 \
  --model ~/.hipfire/models/qwen3.6-35b-a3b.mq4r \
  --daemon target/release/examples/daemon \
  --skip-prefill \
  --decode-context 128 \
  --shadow-iterations 1

# caller may pass exactly one --pm4; location and all other argv preserved as given
python3 -m tools.redline lower pm4 --pm4 \
  --model "$MODEL" \
  --daemon target/release/examples/daemon \
  --prefix 32

python3 -m tools.redline lower pm4 -h
```

## Component boundaries

| Component | Role | Owns |
|---|---|---|
| `tools/redline/__main__.py` | Package dispatcher | Route `lower` → `tools.redline.lower`; usage for unknown package subcommands |
| `tools/redline/lower.py` | Thin delegation | Mode parse (`kernel` / `pm4`), Radiowave resolution, harness path, `--pm4` enforcement, `subprocess` exec, exit mapping |
| `radiowave` binary / `cargo run -p radiowave` | Kernel lowering engine | compile, inspect, oracle, recipes, assess, campaign, all compiler policy |
| `scripts/redline_daemon_harness.py` | Retained capture / PM4 harness | daemon lifecycle, capture, shadow, `--pm4` retained-PM4 IB lowering request |
| Runtime / daemon PM4 path | Actual PM4 lowering | Architecture-specific retained PM4 command bodies (unchanged) |

`lower.py` MUST:

- resolve commands;
- build argv;
- exec and stream I/O;
- return the child exit code when the child starts.

`lower.py` MUST NOT:

- parse Radiowave compile flags into Python structures for re-encoding;
- implement HSACO/PM4 transforms;
- import radiowave crate internals via FFI for this surface;
- decide certification or Golden pass/fail.

## Binary and script resolution

Repository root `REPO` is `Path(__file__).resolve().parents[2]` from `tools/redline/lower.py` (same convention as `tools/redline/golden.py`).

### Kernel / Radiowave

Build a command prefix using the first match in order:

1. **Explicit override:** `--radiowave PATH`
   - `PATH` MUST exist and be executable enough to spawn (file present).
   - Prefix: `[PATH]`.
2. **Release binary:** `REPO / "target/release/radiowave"` if it is a file.
3. **Debug binary:** `REPO / "target/debug/radiowave"` if it is a file.
4. **Cargo fallback:** `["cargo", "run", "-q", "-p", "radiowave", "--"]`
   - Used only when no binary file matched.
   - Requires `cargo` on `PATH` at spawn time; failure to spawn is handled as a missing-surface error if the OS cannot execute cargo, otherwise cargo’s own exit status passes through.

Resolution is about the **program prefix** only. Radiowave subcommand args follow the prefix unchanged.

### PM4 / harness

- Script path: `REPO / "scripts/redline_daemon_harness.py"`.
- Interpreter: `sys.executable` (the same Python running `tools.redline`).
- If the script path is missing → missing-surface exit `2`.
- No alternate harness search path in this design.

## Argv, stdio, and exit-code behavior

### Successful spawn

| Concern | Contract |
|---|---|
| stdout | Child inherited (pass-through); no Python filtering |
| stderr | Child inherited (pass-through); wrapper messages only before spawn or on wrapper-level failure |
| stdin | Inherited |
| cwd | `REPO` |
| env | Inherited `os.environ` (no forced Redline env in the wrapper; harness sets its own child-daemon env) |
| exit code | Child process return code, unchanged |

Use `subprocess.run(..., check=False)` or equivalent with inherited stdio. Do not capture output in the success path.

### Wrapper-level failures (child not started)

Exit **`2`**, message on stderr, including the **attempted command** (shell-join or `list` repr of the argv that would have run, or the resolution step that failed). For PM4 missing-harness errors, the attempted argv MUST match the normalized spawn argv (zero → inserted `--pm4` after script path; exactly one → caller args preserved). Multiple `--pm4` is a wrapper reject before spawn and MUST NOT invent a child argv as if spawn were attempted.

Cases:

- unknown `lower` mode;
- missing `kernel`/`pm4` mode token;
- more than one `--pm4` in `lower pm4` forwarded args;
- `--radiowave` provided without `PATH`, or path not found;
- no Radiowave binary and cargo prefix cannot be constructed (implementation still attempts cargo fallback when binaries are absent; spawn failure of cargo is exit `2` with attempted argv if `FileNotFoundError`, else cargo exit code);
- harness script missing;
- `OSError` / `FileNotFoundError` on spawn.

### Child started but failed

Return the child’s exit code (for example harness `SystemExit(1)` or radiowave non-zero). Do not translate into `2` unless the wrapper itself failed before/at spawn with a resolution error.

### Help conventions

- `python3 -m tools.redline lower` with no mode → stderr usage, exit `2`.
- `python3 -m tools.redline lower {kernel|pm4}` with engine help flags → delegated; exit status from child.
- Package-level `python3 -m tools.redline -h` remains owned by `__main__.py` and MUST list `lower` once the dispatcher is extended.

## Error handling

Stderr prefixes SHOULD be stable for grepping:

```text
tools.redline.lower: ...
```

Required information on missing surfaces:

1. What was missing (mode, binary, script, `--radiowave` path).
2. What was attempted (full argv prefix or path list tried).

Example shapes (informative, not exact string freeze):

```text
tools.redline.lower: radiowave not found (tried --radiowave, target/release/radiowave, target/debug/radiowave, cargo run -q -p radiowave --); attempted: cargo run -q -p radiowave -- ...
tools.redline.lower: missing scripts/redline_daemon_harness.py; attempted: /usr/bin/python3 /repo/scripts/redline_daemon_harness.py --pm4 ...
tools.redline.lower: multiple --pm4 in lower pm4 args; refusing to spawn
tools.redline.lower: unknown mode 'foo' (expected kernel or pm4)
```

Do not raise uncaught stack traces for expected user errors; expected paths end in `return 2` or `raise SystemExit(2)` from `main`.

## Test strategy

Scope: fast, offline, no GPU, no requirement for a built daemon or radiowave artifact in CI unit tests.

1. **Process execution mocked**
   - Patch `subprocess.run` (or the single internal helper `lower.py` uses).
   - Assert argv for:
     - `lower kernel compile ...` → release/debug/`cargo` prefix + forwarded args;
     - `lower kernel --radiowave /tmp/rw ...` → `["/tmp/rw", ...]` and `--radiowave` stripped;
     - `lower pm4 --model M` → `[sys.executable, str(harness), "--pm4", "--model", "M"]` (zero → insert after script path);
     - `lower pm4 --pm4 --model M` → `[sys.executable, str(harness), "--pm4", "--model", "M"]` (exactly one → full caller list preserved, including `--pm4` position);
     - `lower pm4 --model M --pm4` → `[sys.executable, str(harness), "--model", "M", "--pm4"]` (exactly one mid/tail position preserved);
     - `lower pm4 --pm4 --model M --pm4` → exit `2`, stderr contains `multiple --pm4`, `subprocess` not called.
   - Assert return code equals mocked child return code.

2. **Resolution unit tests**
   - Fake filesystem or temporary dirs covering preference order: explicit > release > debug > cargo.
   - Missing harness file → exit `2`, attempted command in stderr.

3. **Smoke delegated help**
   - Optional lightweight test that invokes module main with `kernel -h` or `pm4 -h` under mock, or integration smoke when a radiowave binary exists.
   - MUST NOT require GPU.
   - Goal: prove delegation wiring, not engine correctness (engine tests stay in `crates/radiowave` and harness/scripts coverage).

4. **Dispatcher**
   - When `__main__.py` gains `lower`, one test that unknown package subcommands still exit `2` and that `lower` dispatches into `tools.redline.lower.main`.

Non-goals for tests: bit-exact PM4 bodies, tok/s, Golden floors, real `hipcc` compiles.

## Rollout

1. **Land `tools/redline/lower.py`** implementing `kernel` and `pm4` per this contract, with `main(argv) -> int`.
2. **Extend `tools/redline/__main__.py`** to dispatch `command == "lower"` and update usage strings to include `lower`.
3. **Add unit tests** under `tools/redline/tests/` (or the package’s existing test layout) mocking subprocess and resolution.
4. **Docs touch (optional follow-on, not required to implement the module):** one-line pointers from `docs/VALIDATION.md` / `docs/REDLINE.md` tool tables to `python3 -m tools.redline lower …` as discoverability only—no policy change.
5. **Do not** remove direct `radiowave` or `scripts/redline_daemon_harness.py` invocation paths; wrappers are additive.

Failure of any rollout step MUST leave existing `golden` / `bench` / `serve-diff` behavior unchanged.

## Implementation sketch (non-normative shape)

```text
tools/redline/lower.py
  REPO = parents[2]
  def resolve_radiowave(argv) -> tuple[list[str], list[str]]
  def run_kernel(args) -> int
  def run_pm4(args) -> int
  def main(argv: list[str] | None) -> int
```

Dispatcher fragment:

```python
if command == "lower":
    from tools.redline.lower import main as lower_main
    return lower_main(rest)
```

## Acceptance checklist

- [ ] Only thin delegation; no duplicated lowering logic; no Rust moves.
- [ ] Commands: `lower kernel [Radiowave args...]`, `lower pm4 [Redline harness args...]`.
- [ ] Radiowave resolution: `--radiowave PATH`, then `target/release/radiowave`, then `target/debug/radiowave`, then `cargo run -q -p radiowave --`.
- [ ] PM4: `sys.executable` + `scripts/redline_daemon_harness.py`; zero `--pm4` inserts one after script path; exactly one preserves full caller argv positions; more than one exits `2` with `multiple --pm4` and no spawn; missing-harness diagnostics use the same normalized argv.
- [ ] Child stdout/stderr/exit pass through after successful spawn.
- [ ] Missing surfaces exit `2` with attempted command on stderr.
- [ ] Tests mock process execution and smoke delegated help.
- [ ] Python remains interim discoverability/delegation only (not a stable product layer); compiler/lowering stay in existing Radiowave and runtime/harness surfaces.
