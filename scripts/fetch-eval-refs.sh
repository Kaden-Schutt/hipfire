#!/usr/bin/env bash
# fetch-eval-refs.sh — pull BF16 reference dumps from HF Hub, verify sha256.
#
# Reads benchmarks/quality-baselines/harness/manifest.json. For each entry
# under .references:
#   - if the file already exists at benchmarks/quality-baselines/refs/<name>
#     and its sha256 matches the manifest, skip;
#   - else if .hf_repo is set, fetch via huggingface_hub.hf_hub_download
#     and verify sha256;
#   - else (file missing AND no hf_repo) skip with a warning — typically
#     means the reference is local-only pre-upload.
#
# Refs land at benchmarks/quality-baselines/refs/<name> (gitignored), the
# same path that eval_hipfire / eval_gguf's verify_ref_sha256 walks
# (parent-of-parent → harness/manifest.json).
#
# Dependencies: project venv at REPO_ROOT/.venv with `huggingface_hub`
# installed. Set up with:
#   python3 -m venv .venv && .venv/bin/pip install huggingface_hub
#
# Usage:
#   ./scripts/fetch-eval-refs.sh

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [ ! -x "$PYTHON" ]; then
    echo "fetch-eval-refs.sh: $PYTHON not found." >&2
    echo "Set up the venv:  cd $REPO_ROOT && python3 -m venv .venv && .venv/bin/pip install huggingface_hub" >&2
    exit 2
fi

MANIFEST="benchmarks/quality-baselines/harness/manifest.json"
DEST_DIR="benchmarks/quality-baselines/refs"

if [ ! -f "$MANIFEST" ]; then
    echo "fetch-eval-refs.sh: manifest not found at $MANIFEST" >&2
    exit 2
fi

mkdir -p "$DEST_DIR"

"$PYTHON" - "$MANIFEST" "$DEST_DIR" <<'PYEOF'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
dest_dir = Path(sys.argv[2])

with manifest_path.open() as f:
    manifest = json.load(f)

refs = manifest.get("references", {})
if not refs:
    print("manifest has no references; nothing to fetch.", file=sys.stderr)
    sys.exit(0)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_download(repo_id: str, filename: str, dest_dir: Path, repo_type: str) -> Path:
    from huggingface_hub import hf_hub_download

    out = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_dir),
        repo_type=repo_type,
    )
    return Path(out)


failed: list[str] = []
for name, entry in refs.items():
    expected = entry.get("sha256")
    hf_repo = entry.get("hf_repo")
    hf_repo_type = entry.get("hf_repo_type", "model")
    dest = dest_dir / name

    if not expected:
        print(f"[{name}] manifest entry missing sha256 — skipping", file=sys.stderr)
        failed.append(name)
        continue

    if dest.exists():
        print(f"[{name}] present locally; verifying sha256...", file=sys.stderr)
        actual = sha256_of(dest)
        if actual == expected:
            print(f"[{name}] OK ({actual})", file=sys.stderr)
        else:
            print(f"[{name}] SHA256 MISMATCH", file=sys.stderr)
            print(f"  expected: {expected}", file=sys.stderr)
            print(f"  actual:   {actual}", file=sys.stderr)
            print(
                "  refusing to delete; rename or remove manually then re-run.",
                file=sys.stderr,
            )
            failed.append(name)
        continue

    if not hf_repo:
        print(
            f"[{name}] absent locally and no hf_repo in manifest; SKIPPED "
            "(typical pre-upload state)",
            file=sys.stderr,
        )
        continue

    print(f"[{name}] fetching {hf_repo_type} {hf_repo}/{name} ...", file=sys.stderr)
    try:
        out = hf_download(hf_repo, name, dest_dir, hf_repo_type)
    except Exception as e:
        print(f"[{name}] download failed: {e}", file=sys.stderr)
        failed.append(name)
        continue

    actual = sha256_of(out)
    if actual != expected:
        print(f"[{name}] sha256 mismatch after download", file=sys.stderr)
        print(f"  expected: {expected}", file=sys.stderr)
        print(f"  actual:   {actual}", file=sys.stderr)
        failed.append(name)
        continue

    print(f"[{name}] downloaded + verified ({actual})", file=sys.stderr)

if failed:
    print(
        f"\nfetch-eval-refs.sh: {len(failed)} failure(s): {failed}",
        file=sys.stderr,
    )
    sys.exit(1)

print("\nfetch-eval-refs.sh: all references verified.", file=sys.stderr)
PYEOF
