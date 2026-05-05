#!/usr/bin/env bash
# Verify that every pub fn in `impl Gpu` of dispatch.rs either calls
# `self.bind_thread()?;` (or `let _ = self.bind_thread();`) as the first
# statement, or carries a `// bind_thread: skip — <reason>` whitelist marker.
#
# Also runs a multi-GPU heuristic: anywhere `let g = &mut gpus.devices[X]`
# appears, the first `g.<method>` call MUST NOT be `g.hip.<rawop>` (raw
# HipRuntime methods bypass the bind_thread audit; if the first thing the
# function does on `g` is a raw hipMalloc/memset/memcpy, the allocation
# may land on a wrongly-bound device — see Stage 5 finding).
#
# Run by .githooks/pre-commit when dispatch.rs or any *_multi.rs is staged.

set -eu

FILE="${1:-crates/rdna-compute/src/dispatch.rs}"

if [ ! -f "$FILE" ]; then
    echo "verify-bind-thread: $FILE not found" >&2
    exit 2
fi

python3 - "$FILE" <<'PYEOF'
import re, sys
from pathlib import Path

WHITELIST_SPECIAL = {"init", "init_with_device", "bind_thread", "bind_thread_or_warn"}

path = Path(sys.argv[1])
lines = path.read_text().splitlines()

impl_start = impl_end = None
for i, l in enumerate(lines):
    if l.startswith("impl Gpu {") and impl_start is None:
        impl_start = i
    elif impl_start is not None and l.startswith("impl Drop for Gpu"):
        impl_end = i
        break
if impl_start is None or impl_end is None:
    print("verify-bind-thread: impl Gpu boundaries not found", file=sys.stderr)
    sys.exit(2)

violations = []
total = 0
i = impl_start
while i < impl_end:
    m = re.match(r"^    pub fn (\w+)", lines[i])
    if not m:
        i += 1
        continue
    fn_name = m.group(1)
    fn_line = i + 1
    total += 1

    if fn_name in WHITELIST_SPECIAL:
        i += 1
        continue

    # Find body opener (line ending with ' {')
    sig_end = i
    while sig_end < impl_end:
        s = lines[sig_end].rstrip()
        if s.endswith("{") and not s.endswith("!{") and not s.endswith("::{"):
            break
        sig_end += 1
    if sig_end >= impl_end:
        violations.append((fn_line, fn_name, "body opener not found"))
        i += 1
        continue

    # Walk first ~8 non-blank/non-doc body lines for the marker. Anchored
    # patterns: marker must be the leading token on its line (catches
    # "self.bind_thread() already" prose comments, etc.).
    call_re = re.compile(r"^self\.bind_thread(?:_or_warn)?\(\)")
    skip_re = re.compile(r"^//\s*bind_thread:\s*skip\b")
    found = False
    j = sig_end + 1
    checked = 0
    while j < impl_end and checked < 8:
        s = lines[j].strip()
        if s == "" or s.startswith("///"):
            j += 1
            continue
        if call_re.match(s) or skip_re.match(s):
            found = True
            break
        if s.startswith("//"):
            j += 1
            checked += 1
            continue
        # First real statement — if not bind_thread, the fn is non-compliant
        break

    if not found:
        violations.append((fn_line, fn_name, "missing bind_thread"))

    i = sig_end + 1

if total == 0:
    print(
        "verify-bind-thread: regex matched zero `pub fn` in impl Gpu — "
        "indentation changed or impl boundary moved. Audit is broken.",
        file=sys.stderr,
    )
    sys.exit(2)

if violations:
    print(
        f"verify-bind-thread: {len(violations)} of {total} pub fn in impl Gpu "
        f"missing bind_thread:",
        file=sys.stderr,
    )
    for ln, name, reason in violations[:20]:
        print(f"  {path}:{ln}: {name} — {reason}", file=sys.stderr)
    if len(violations) > 20:
        print(f"  ... and {len(violations) - 20} more", file=sys.stderr)
    print(
        "\nFix: add `self.bind_thread()?;` (or `let _ = self.bind_thread();` for\n"
        "non-HipResult fn) as the first statement, OR add a\n"
        "`// bind_thread: skip — <reason>` comment for pure-state queries.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"verify-bind-thread: OK — {total} pub fn audited.")
PYEOF

# ─── Multi-GPU bind heuristic ─────────────────────────────────────────
# Flag any `let <name> = &mut gpus.devices[X]` site whose first call on
# `<name>` is a raw `<name>.hip.<rawop>` instead of either `<name>.bind_thread()`
# or any other `<name>.<method>` (Gpu-method calls auto-bind via the Stage 2b
# audit). Strict — to whitelist, add `// bind: pre-bound — <reason>` on the
# `let` line.

python3 - <<'PYEOF2'
import re, sys, subprocess
from pathlib import Path

RAW_OPS = "malloc|memset|memcpy_htod|memcpy_dtoh|memcpy_dtod|memset_async|memcpy_htod_async|memcpy_dtoh_async|memcpy_dtod_async"
LET_GPUS = re.compile(r"^\s*let\s+(\w+)\s*=\s*&mut\s+gpus\.devices\[")
ALLOW_MARKER = re.compile(r"//\s*bind:\s*pre-bound\b")

# Scope: scan all .rs files under crates/
out = subprocess.check_output(
    ["git", "ls-files", "crates/", "*.rs"], text=True
).splitlines()

violations = []
for fpath in out:
    if not fpath.endswith(".rs"):
        continue
    p = Path(fpath)
    if not p.is_file():
        continue
    lines = p.read_text().splitlines()
    for i, line in enumerate(lines):
        m = LET_GPUS.match(line)
        if not m:
            continue
        if ALLOW_MARKER.search(line):
            continue
        var = m.group(1)
        # Patterns matched on each candidate line. Order of checks matters —
        # `var.hip.<rawop>` and `var.bind_thread` are NOT also `var.<method>(`,
        # so each is searched independently. Whichever matches first on the
        # earliest non-blank line is the "first call on var".
        hip_raw_re = re.compile(
            rf"\b{re.escape(var)}\.hip\.(?:{RAW_OPS})\b"
        )
        bind_re = re.compile(rf"\b{re.escape(var)}\.bind_thread\b")
        any_method_re = re.compile(rf"\b{re.escape(var)}\.(\w+)\(")

        decision = None  # ("ok", line) | ("violation", line)
        for j in range(i + 1, min(i + 26, len(lines))):
            s = lines[j].strip()
            if s == "" or s.startswith("//"):
                continue
            line_text = lines[j]
            if hip_raw_re.search(line_text):
                decision = ("violation", line_text)
                break
            if bind_re.search(line_text):
                decision = ("ok", line_text)
                break
            if any_method_re.search(line_text):
                decision = ("ok", line_text)
                break
            # Otherwise the line references var but neither calls a method nor
            # routes through .hip — keep walking.
        if decision and decision[0] == "violation":
            violations.append((fpath, i + 1, var, decision[1].strip()))

if violations:
    print(
        f"verify-multi-gpu-bind: {len(violations)} suspect raw HIP call(s) "
        "in multi-GPU contexts:",
        file=sys.stderr,
    )
    for fpath, line_no, var, snippet in violations[:20]:
        print(f"  {fpath}:{line_no}: `{var}` → {snippet}", file=sys.stderr)
    if len(violations) > 20:
        print(f"  ... and {len(violations) - 20} more", file=sys.stderr)
    print(
        "\nIn each `let <name> = &mut gpus.devices[X]` block the first call on\n"
        "`<name>` MUST be either `<name>.bind_thread()` or any non-`hip` Gpu\n"
        "method (which auto-binds via the Stage 2b audit). Raw `<name>.hip.*`\n"
        "lands on whatever device the host thread last bound — likely wrong\n"
        "in multi-GPU paths. To accept the call site as already-bound, add\n"
        "`// bind: pre-bound — <reason>` on the `let` line.",
        file=sys.stderr,
    )
    sys.exit(1)

print("verify-multi-gpu-bind: OK")
PYEOF2
