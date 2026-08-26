#!/usr/bin/env bash
# Redline golden-fixture reproduction package.
#
# WHY THIS EXISTS
#
# `python3 -m tools.redline golden` pins the model, sampling profile, benchmark
# contract, PM4 policy and route identity — but NOT the compiled code objects
# or the host toolchain. Contributors reproducing the gfx1201 fixture have
# reported PM4 slower than HIP on the same source commit, the same model
# SHA-256 and the same ROCm version, while that commit passes elsewhere. When
# the route identity matches byte-for-byte (same dispatch count, kernel count
# and sequence hash) but throughput does not, the divergence lives BELOW the
# tape: code objects, toolchain, or machine state.
#
# This script captures what the golden run does not, diffs it against a box
# that cannot reproduce, and can force our exact code objects to be used.
#
# HOW KERNEL PINNING WORKS  (crates/rdna-compute/src/compiler.rs)
#
#   precompiled kernels/compiled/<arch>/<name>.hsaco exists?
#     .hash sidecar matches src_hash        -> use it
#     no valid hash AND no device compiler  -> use it, engine warns
#     no valid hash AND compiler present    -> RECOMPILE (pinning defeated)
#
# src_hash folds in `toolchain_id`, so a different toolchain invalidates our
# sidecars by design. `run --pin-kernels` therefore installs the blobs AND
# hides the device compiler from the daemon's PATH, taking the middle branch so
# our exact code objects execute. The engine prints its own "Output may be
# incorrect" warning in that state; under --pin-kernels that warning is
# EXPECTED and is precisely the confirmation that our blobs are in use.
#
# TOOLCHAIN NOTE
#
# The engine currently probes `hipcc` only. hipcc is being wound down upstream
# in favour of invoking amdclang++/clang++ directly, and on ROCm 7.14 hipcc is
# already just an ELF wrapper around amdclang++. This script therefore treats
# the device compiler as a SET (see DEVICE_COMPILERS) rather than assuming
# hipcc, so pinning keeps working after the engine moves off hipcc.
#
# USAGE
#
#   # On a box where the fixture passes, after a successful golden run:
#   scripts/redline-repro-package.sh capture --arch gfx1201
#
#   # On a box that cannot reproduce:
#   scripts/redline-repro-package.sh verify --package repro-gfx1201-*.tar.gz
#   scripts/redline-repro-package.sh run    --package repro-gfx1201-*.tar.gz
#   scripts/redline-repro-package.sh run    --package ... --pin-kernels
#   scripts/redline-repro-package.sh run    --package ... --force
#
# Exit codes: 0 ok, 1 usage/IO error, 2 blocking mismatch (overridden by --force).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCHEMA_VERSION=1
PY=python3

# Device compilers, most-specific first. hipcc is listed for today's engine;
# the amdclang/clang entries keep --pin-kernels correct once the engine invokes
# the LLVM driver directly.
DEVICE_COMPILERS=(hipcc amdclang++ amdclang clang++ clang)

die() { echo "error: $*" >&2; exit 1; }
command -v "$PY" >/dev/null || die "python3 required (this repo deliberately does not depend on jq)"

# `cmd | head -1 || echo fallback` concatenates BOTH outputs under pipefail:
# head closes the pipe, the pipeline reports failure, and the fallback still
# runs. Capture the whole output first, slice afterwards.
first_line() {
  local out
  out="$("$@" 2>/dev/null || true)"
  [ -n "$out" ] || { echo "unavailable"; return; }
  printf '%s\n' "${out%%$'\n'*}"
}

nth_line() {
  local n="$1"; shift
  local out
  out="$("$@" 2>/dev/null || true)"
  printf '%s\n' "$out" | sed -n "${n}p"
}

# The code-generator identity. amdclang++ reports it on line 1; hipcc reports
# the HIP runtime version on line 1 and the AMD clang version on line 2. The
# engine currently fingerprints hipcc line 1, which is the weaker of the two.
codegen_version() {
  if command -v amdclang++ >/dev/null 2>&1; then
    first_line amdclang++ --version
  elif command -v hipcc >/dev/null 2>&1; then
    nth_line 2 hipcc --version
  else
    echo unknown
  fi
}

detect_arch() {
  local device="${1:-0}"
  command -v rocminfo >/dev/null || die "rocminfo not found; pass --arch explicitly"
  rocminfo 2>/dev/null | grep -oE 'gfx[0-9]+' | sed -n "$((device + 1))p"
}

# ROCm installs are not always at /opt/rocm — this fleet uses
# /opt/rocm/core-<ver>. Derive the root from the resolved compiler first, then
# fall back to $ROCM_PATH and a glob, so side-by-side installs report the
# version actually in use rather than whichever one is symlinked.
rocm_version() {
  local cc root
  for cc in "${DEVICE_COMPILERS[@]}"; do
    if command -v "$cc" >/dev/null 2>&1; then
      root="$(cd "$(dirname "$(readlink -f "$(command -v "$cc")")")/.." && pwd)"
      [ -f "$root/.info/version" ] && { cat "$root/.info/version"; return; }
    fi
  done
  [ -n "${ROCM_PATH:-}" ] && [ -f "$ROCM_PATH/.info/version" ] && { cat "$ROCM_PATH/.info/version"; return; }
  local f
  for f in /opt/rocm/.info/version /opt/rocm*/.info/version /opt/rocm/core-*/.info/version; do
    [ -f "$f" ] && { cat "$f"; return; }
  done
  echo unknown
}

usage() { sed -n '2,53p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# ── capture ──────────────────────────────────────────────────────────────────

cmd_capture() {
  local arch="" device=0 out_dir="." attestation=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --arch) arch="$2"; shift 2 ;;
      --device) device="$2"; shift 2 ;;
      --out) out_dir="$2"; shift 2 ;;
      --attestation) attestation="$2"; shift 2 ;;
      -h|--help) usage 0 ;;
      *) die "unknown capture flag: $1" ;;
    esac
  done
  [ -n "$arch" ] || arch="$(detect_arch "$device")"
  [ -n "$arch" ] || die "could not detect arch; pass --arch"

  local stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local stage; stage="$(mktemp -d)"
  trap 'rm -rf "$stage"' RETURN
  mkdir -p "$stage/hsaco/$arch"

  local cache="${HIPFIRE_KERNEL_CACHE:-.hipfire_kernels}/$arch"
  [ -d "$cache" ] || die "no kernel cache at $cache — run the golden fixture first so code objects exist"
  cp "$cache"/*.hsaco "$stage/hsaco/$arch/" 2>/dev/null || die "no .hsaco found in $cache"
  # .hash sidecars let a same-toolchain box use these without --pin-kernels.
  cp "$cache"/*.hash "$stage/hsaco/$arch/" 2>/dev/null || true

  if [ -z "$attestation" ]; then
    attestation="$(ls -1t .redline-work/golden/*"$arch"*.golden.json 2>/dev/null | head -1 || true)"
  fi
  if [ -n "$attestation" ] && [ -f "$attestation" ]; then
    cp "$attestation" "$stage/attestation.json"
    local report="${attestation%.golden.json}.json"
    [ -f "$report" ] && cp "$report" "$stage/product-report.json" || true
  else
    echo "warning: no golden attestation for $arch — package carries no reference result" >&2
  fi

  cp registry/redline-golden-v1.json "$stage/fixture-registry.json"

  {
    echo "captured_at_utc=$stamp"
    echo "host=$(hostname)"
    echo "kernel=$(uname -r)"
    echo "rocm_version=$(rocm_version)"
    # Record every device compiler on PATH. The engine fingerprints only the
    # first line of `hipcc --version`, which on ROCm 7.14 is the HIP runtime
    # version rather than the code generator — so capture both explicitly.
    for cc in "${DEVICE_COMPILERS[@]}"; do
      if command -v "$cc" >/dev/null 2>&1; then
        echo "compiler_${cc//+/p}_path=$(command -v "$cc")"
        echo "compiler_${cc//+/p}_version=$("$cc" --version 2>/dev/null | head -1)"
      fi
    done
    echo "hipcc_version=$(first_line hipcc --version)"
    echo "codegen_version=$(codegen_version)"
  } > "$stage/env.txt"

  rocm-smi --showallinfo > "$stage/gpu.txt" 2>&1 || amd-smi static > "$stage/gpu.txt" 2>&1 || echo unavailable > "$stage/gpu.txt"

  "$PY" - "$stage" "$arch" "$stamp" "target/release/daemon" "$SCHEMA_VERSION" <<'PYCAP'
import hashlib, json, pathlib, subprocess, sys

stage, arch, stamp, daemon, schema = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
stage = pathlib.Path(stage)

def sha_file(p):
    p = pathlib.Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

blob_dir = stage / "hsaco" / arch
blobs = sorted(blob_dir.glob("*.hsaco"))
files = {b.name: sha_file(b) for b in blobs}
mh = hashlib.sha256()
for name in sorted(files):
    mh.update(name.encode()); mh.update(bytes.fromhex(files[name]))

reg = json.loads((stage / "fixture-registry.json").read_text())
fixture = next((f for f in reg["fixtures"] if f["architecture"] == arch), None)
att = stage / "attestation.json"
env = dict(l.split("=", 1) for l in (stage / "env.txt").read_text().splitlines() if "=" in l)
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

manifest = {
    "schema_version": schema,
    "captured_at_utc": stamp,
    "architecture": arch,
    "source_commit": commit or None,
    "daemon_sha256": sha_file(daemon),
    "env": env,
    "model": reg["model"],
    "pm4_policy": reg["pm4_policy"],
    "benchmark": (fixture or {}).get("benchmark"),
    "acceptance": (fixture or {}).get("acceptance"),
    "route": (fixture or {}).get("route"),
    "hsaco": {
        "count": len(files),
        "total_bytes": sum(b.stat().st_size for b in blobs),
        "manifest_sha256": mh.hexdigest(),
        "have_hash_sidecars": bool(list(blob_dir.glob("*.hash"))),
        "files": files,
    },
    "observed": json.loads(att.read_text()) if att.exists() else None,
}
(stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"  arch             {arch}")
print(f"  source commit    {manifest['source_commit']}")
print(f"  daemon sha256    {manifest['daemon_sha256']}")
print(f"  code objects     {len(files)} files, {manifest['hsaco']['total_bytes']:,} bytes")
print(f"  hsaco manifest   {mh.hexdigest()}")
print(f"  rocm             {env.get('rocm_version','unknown')}")
print(f"  codegen          {env.get('codegen_version','unknown')}")
PYCAP

  mkdir -p "$out_dir"
  local tarball="$out_dir/repro-$arch-$stamp.tar.gz"
  tar -czf "$tarball" -C "$stage" .
  echo "package: $tarball ($(du -h "$tarball" | cut -f1))"
}

# ── verify ───────────────────────────────────────────────────────────────────

unpack() {
  local pkg="$1"
  [ -f "$pkg" ] || die "package not found: $pkg"
  local dir; dir="$(mktemp -d)"
  tar -xzf "$pkg" -C "$dir"
  [ -f "$dir/manifest.json" ] || die "no manifest.json in package: $pkg"
  echo "$dir"
}

do_verify() {
  local dir="$1" device="$2"
  local local_arch; local_arch="$(detect_arch "$device" 2>/dev/null || echo unknown)"
  local local_rocm; local_rocm="$(rocm_version)"
  local local_hipcc; local_hipcc="$(first_line hipcc --version)"
  local local_codegen; local_codegen="$(codegen_version)"
  local local_commit; local_commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

  "$PY" - "$dir" "$local_arch" "$local_rocm" "$local_hipcc" "$local_codegen" \
        "$local_commit" "target/release/daemon" \
        "${HIPFIRE_KERNEL_CACHE:-.hipfire_kernels}" <<'PYVER'
import hashlib, json, os, pathlib, sys

dir_, arch, rocm, hipcc, codegen, commit, daemon, cache = sys.argv[1:9]
m = json.loads((pathlib.Path(dir_) / "manifest.json").read_text())

def sha_file(p):
    p = pathlib.Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []
blocking = 0

def check(label, expected, actual, severity, note="", ok_note=""):
    # note is shown only on mismatch; ok_note only when the check passes.
    global blocking
    if expected != actual and severity == "BLOCKING":
        blocking += 1
    rows.append((label, "ok" if expected == actual else severity, expected, actual, note, ok_note))

check("gpu arch", m["architecture"], arch, "BLOCKING", "a different arch cannot reproduce this fixture")

model = m["model"]
model_path = pathlib.Path.home() / ".hipfire" / "models" / model["file"]
check("model sha256", model["sha256"], sha_file(model_path), "BLOCKING", f"expected at {model_path}")

# An unset PM4 var means the committed default applies, which IS the policy.
for key, want in sorted(m.get("pm4_policy", {}).items()):
    got = os.environ.get(key)
    check(f"env {key}", str(want), str(want) if got is None else got, "BLOCKING",
          note="explicitly overridden in your shell — unset it or you measure a different route",
          ok_note="unset = committed default" if got is None else "matches policy")

check("source commit", m["source_commit"], commit, "ADVISORY",
      "differs -> route-compatible-reproduction, not exact-reference-binary")
check("daemon sha256", m["daemon_sha256"], sha_file(daemon), "ADVISORY", "rebuild differences are expected")
check("rocm version", m["env"].get("rocm_version"), rocm, "ADVISORY", "folded into the kernel cache hash")
check("hipcc version", m["env"].get("hipcc_version"), hipcc, "ADVISORY",
      "NOTE: this line is the HIP runtime version, not the code generator")
check("codegen version", m["env"].get("codegen_version"), codegen, "ADVISORY",
      "THIS is what actually determines the code objects; differs -> use --pin-kernels")

local_dir = pathlib.Path(cache) / arch
local_blobs = sorted(local_dir.glob("*.hsaco")) if local_dir.is_dir() else []
local_manifest = None
if local_blobs:
    mh = hashlib.sha256()
    for b in local_blobs:
        mh.update(b.name.encode()); mh.update(bytes.fromhex(sha_file(b)))
    local_manifest = mh.hexdigest()
check("hsaco manifest", m["hsaco"]["manifest_sha256"], local_manifest, "ADVISORY",
      f"{len(local_blobs)} local vs {m['hsaco']['count']} packaged code objects")

w = max(len(r[0]) for r in rows)
print()
for label, status, expected, actual, note, ok_note in rows:
    print(f"  {label.ljust(w)}  {status.ljust(9)}  ", end="")
    if status == "ok":
        print(ok_note)
    else:
        print(f"expected {expected!r}")
        print(f"  {' '.ljust(w)}             actual   {actual!r}")
        if note:
            print(f"  {' '.ljust(w)}             note     {note}")

obs = ((m.get("observed") or {}).get("validation")) or {}
print()
if obs:
    print(f"  reference result:  {obs.get('classification', 'unknown')} (valid={obs.get('valid')})")
    print(f"  reference perf:    HIP {obs['hip_median_tok_s']:.3f} -> PM4 "
          f"{obs['pm4_median_tok_s']:.3f} tok/s ({obs['speedup']:.4f}x)")
else:
    print("  reference result:  none recorded")
if m.get("acceptance"):
    a = m["acceptance"]
    print(f"  acceptance floors: pm4 >= {a.get('minimum_pm4_tok_s')} tok/s, speedup >= {a.get('minimum_speedup')}x")
if m.get("route"):
    r = m["route"]
    print(f"  required tape:     {r.get('dispatches')} dispatches / {r.get('unique_kernels')} kernels / {r.get('sequence_hash')}")
print()
if blocking:
    print(f"  {blocking} BLOCKING mismatch(es) — pass --force to proceed anyway.")
sys.exit(2 if blocking else 0)
PYVER
}

cmd_verify() {
  local pkg="" device=0 force=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --package) pkg="$2"; shift 2 ;;
      --device) device="$2"; shift 2 ;;
      --force) force=1; shift ;;
      -h|--help) usage 0 ;;
      *) die "unknown verify flag: $1" ;;
    esac
  done
  [ -n "$pkg" ] || die "verify requires --package"
  local dir; dir="$(unpack "$pkg")"
  trap 'rm -rf "$dir"' RETURN
  local rc=0
  do_verify "$dir" "$device" || rc=$?
  if [ "$rc" -eq 2 ] && [ "$force" -eq 1 ]; then
    echo "  --force: continuing despite blocking mismatches"
    return 0
  fi
  return "$rc"
}

# ── run ──────────────────────────────────────────────────────────────────────

cmd_run() {
  local pkg="" device=0 force=0 pin=0
  local -a passthrough=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --package) pkg="$2"; shift 2 ;;
      --device) device="$2"; shift 2 ;;
      --force) force=1; shift ;;
      --pin-kernels) pin=1; shift ;;
      -h|--help) usage 0 ;;
      *) passthrough+=("$1"); shift ;;
    esac
  done
  [ -n "$pkg" ] || die "run requires --package"
  local dir; dir="$(unpack "$pkg")"
  local SHIM_DIR=""
  trap 'rm -rf "$dir" "${SHIM_DIR:-/nonexistent}"' RETURN

  local arch
  arch="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['architecture'])" "$dir/manifest.json")"

  local rc=0
  do_verify "$dir" "$device" || rc=$?
  if [ "$rc" -eq 2 ]; then
    [ "$force" -eq 1 ] || { echo "  refusing to run with blocking mismatches; pass --force to override" >&2; return 2; }
    echo "  --force: running despite blocking mismatches"
  fi

  local -a env_prefix=()
  if [ "$pin" -eq 1 ]; then
    mkdir -p "kernels/compiled/$arch"
    cp "$dir/hsaco/$arch"/*.hsaco "kernels/compiled/$arch/"
    cp "$dir/hsaco/$arch"/*.hash "kernels/compiled/$arch/" 2>/dev/null || true
    echo "  pinned $(ls -1 "kernels/compiled/$arch"/*.hsaco 2>/dev/null | wc -l) packaged code objects -> kernels/compiled/$arch"

    # Evict locally-JIT'd blobs so the pinned ones are what actually load.
    rm -rf "${HIPFIRE_KERNEL_CACHE:-.hipfire_kernels}/$arch"

    # Force the engine to treat the device compiler as absent.
    #
    # This used to be done by shadowing compiler names on PATH, which worked
    # while the probe was a bare `Command::new("hipcc")`. Resolved-root
    # discovery now finds the compiler regardless of PATH, so shadowing
    # silently recompiled instead of pinning. Use the explicit switch.
    env_prefix=(env "HIPFIRE_NO_DEVICE_COMPILER=1")
    echo "  HIPFIRE_NO_DEVICE_COMPILER=1 — engine will treat the compiler as absent"
    echo "  the engine will warn 'Output may be incorrect' — that is EXPECTED here"
    echo "  and is the confirmation that OUR code objects are executing."
  fi

  echo
  echo "  running golden fixture for $arch ..."
  "${env_prefix[@]}" "$PY" -m tools.redline golden --arch "$arch" ${passthrough[@]+"${passthrough[@]}"}
}

[ $# -gt 0 ] || usage 1
case "$1" in
  capture) shift; cmd_capture "$@" ;;
  verify)  shift; cmd_verify "$@" ;;
  run)     shift; cmd_run "$@" ;;
  -h|--help) usage 0 ;;
  *) die "unknown subcommand: $1 (expected capture|verify|run)" ;;
esac
