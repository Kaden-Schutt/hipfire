#!/usr/bin/env bash
# mi300x_bootstrap.sh — one-shot bootstrap for a DigitalOcean MI300x droplet.
#
# Assumes:
#   - Droplet template: PyTorch 2.6.0 + ROCm 7.0.0 (Ubuntu 22.04)
#   - 1× MI300x (192 GB HBM3, gfx942)
#   - Persistent block storage mounted at /workspace (override via WORK env)
#   - Root or sudo-able user
#
# Output:
#   - $WORK/hipfire           — built repo (cargo target/release/* ready)
#   - $WORK/hf-cache          — HF model snapshots (pinned revisions)
#   - $WORK/imatrix/<repo>/   — unsloth imatrix files
#   - $WORK/results/<ts>/     — empty, populated by downstream scripts
#
# Idempotent: re-running this script skips phases whose outputs already exist.
# Phases are numbered so a partial failure can be resumed cleanly.

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────────
WORK="${WORK:-/workspace}"
HIPFIRE="${HIPFIRE_DIR:-${WORK}/hipfire}"
HF_HOME="${HF_HOME:-${WORK}/hf-cache}"
IMATRIX_DIR="${WORK}/imatrix"
RESULTS_DIR="${WORK}/results"
HIPFIRE_BRANCH="${HIPFIRE_BRANCH:-worktree-awq-raw-sumsq-converter}"
HIPFIRE_REMOTE="${HIPFIRE_REMOTE:-https://github.com/Kaden-Schutt/hipfire.git}"
TARGET_ARCH="${TARGET_ARCH:-gfx942}"

mkdir -p "$WORK" "$HF_HOME" "$IMATRIX_DIR" "$RESULTS_DIR"
export HF_HOME
# huggingface_hub reads HF_HOME at import time, so child processes (incl.
# screen sessions invoked later) must inherit it. Re-exporting after mkdir
# ensures the path exists when first cached.

# HuggingFace revisions pinned for reproducibility against hiptrx baselines.
# Lines: <hf_repo> <revision>
read -r -d '' MODEL_REVISIONS <<'EOF' || true
Qwen/Qwen3.5-0.8B           2fc06364715b967f1860aea9cf38778875588b17
Qwen/Qwen3.5-9B             c202236235762e1c871ad0ccb60c8ee5ba337b9a
Qwen/Qwen3.5-27B            b7ca741b86de18df552fd2cc952861e04621a4bd
Qwen/Qwen3.6-27B            6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
Qwen/Qwen3.6-35B-A3B        7da1103448ba36029c34ce1a9a741dfe93ee0c50
z-lab/Qwen3.5-27B-DFlash    b0400439c04be32c24e04d9dce3821b582c1a68a
EOF

# Per-imatrix repo. Filename is always imatrix_unsloth.gguf_file.
IMATRIX_REPOS=(
    unsloth/Qwen3.5-0.8B-GGUF
    unsloth/Qwen3.5-9B-GGUF
    unsloth/Qwen3.5-27B-GGUF
    unsloth/Qwen3.6-27B-GGUF
    unsloth/Qwen3.6-35B-A3B-GGUF
)

# ── Logging ─────────────────────────────────────────────────────────────────
phase() {
    echo
    echo "═══ [$(date +%H:%M:%S)] $* ═══"
}
ok()   { printf "    \033[32m✓\033[0m %s\n" "$*"; }
warn() { printf "    \033[33m!\033[0m %s\n" "$*"; }
die()  { printf "    \033[31m✗\033[0m %s\n" "$*" >&2; exit 1; }

# ── Phase 0: sanity ─────────────────────────────────────────────────────────
phase "0/9  Sanity — ROCm + GPU detection"
if ! command -v rocm-smi >/dev/null; then
    die "rocm-smi missing — wrong container? Expected PyTorch+ROCm template."
fi
rocm-smi --showproductname | sed 's/^/    /'
if rocm-smi --showproductname 2>/dev/null | grep -qi MI300X; then
    ok "MI300x detected"
else
    warn "No MI300x in rocm-smi output; continuing anyway (override TARGET_ARCH=$TARGET_ARCH)"
fi
ok "ROCm version: $(rocm-smi --showversion 2>/dev/null | grep -i rocm | head -1 | awk -F: '{print $2}' | xargs || echo unknown)"

# ── Phase 1: system deps ────────────────────────────────────────────────────
phase "1/9  System build deps"
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi
ok "rustc: $(rustc --version)"

if [ ! -f "$WORK/.apt-deps-installed" ]; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        pkg-config libssl-dev build-essential git curl jq ca-certificates \
        > /dev/null
    touch "$WORK/.apt-deps-installed"
fi
ok "apt deps installed"

# ── Phase 2: clone hipfire ──────────────────────────────────────────────────
phase "2/9  Clone hipfire @ $HIPFIRE_BRANCH"
if [ ! -d "$HIPFIRE/.git" ]; then
    git clone --branch "$HIPFIRE_BRANCH" --depth 1 "$HIPFIRE_REMOTE" "$HIPFIRE"
fi
cd "$HIPFIRE"
git fetch --depth 1 origin "$HIPFIRE_BRANCH"
git reset --hard "origin/$HIPFIRE_BRANCH"
ok "HEAD: $(git rev-parse --short HEAD)  $(git log -1 --pretty=%s | head -c 70)"

# ── Phase 3: compile HIP kernels for gfx942 ─────────────────────────────────
phase "3/9  Compile HIP kernels for $TARGET_ARCH"
# WMMA kernels use RDNA-only builtins. Patch compile-kernels.sh to skip them
# on gfx942 before invoking. The patch is in-place and idempotent.
if ! grep -q "WMMA SKIP for gfx94" scripts/compile-kernels.sh; then
    python3 - <<'PYPATCH'
from pathlib import Path
p = Path("scripts/compile-kernels.sh")
src = p.read_text()
marker = '        # gfx906-specific kernels (sdot4 dp4a, etc.) only build on gfx906.'
patch = '''        # WMMA SKIP for gfx942 (CDNA3 has MFMA, not RDNA WMMA builtins). The
        # *_wave64.hip family covers MI300x via MFMA.
        if [[ "$arch" == gfx94* ]]; then
            case "$name" in
                *wmma*)
                    echo "  - $name SKIP (RDNA-only WMMA on $arch)"
                    continue
                    ;;
            esac
        fi

'''
if marker not in src:
    raise SystemExit("compile-kernels.sh layout changed; patch marker not found")
src = src.replace(marker, patch + marker, 1)
p.write_text(src)
PYPATCH
    ok "compile-kernels.sh patched to skip WMMA on gfx94*"
fi

if [ -d "kernels/compiled/$TARGET_ARCH" ] && \
   [ "$(ls "kernels/compiled/$TARGET_ARCH/"*.hsaco 2>/dev/null | wc -l)" -gt 50 ]; then
    ok "kernels already compiled for $TARGET_ARCH ($(ls kernels/compiled/$TARGET_ARCH/*.hsaco | wc -l) files)"
else
    JOBS="${JOBS:-$(nproc)}" ./scripts/compile-kernels.sh "$TARGET_ARCH" 2>&1 \
        | tail -30
    n=$(ls "kernels/compiled/$TARGET_ARCH/"*.hsaco 2>/dev/null | wc -l)
    [ "$n" -gt 50 ] || die "kernel compile produced only $n .hsaco files; check log"
    ok "compiled $n kernels for $TARGET_ARCH"
fi

# ── Phase 4: cargo build ───────────────────────────────────────────────────
phase "4/9  cargo build --release"
RUSTC_WRAPPER="${RUSTC_WRAPPER:-}" \
HIPFIRE_TARGET_ARCH="$TARGET_ARCH" \
cargo build --release \
    -p hipfire-quantize \
    -p hipfire-runtime \
    --features deltanet \
    --example eval_hipfire \
    --example coherence_probe \
    --example daemon \
    2>&1 | tail -20
for b in target/release/hipfire-quantize \
         target/release/examples/eval_hipfire \
         target/release/examples/coherence_probe \
         target/release/examples/daemon; do
    [ -x "$b" ] || die "missing binary: $b"
done
ok "binaries built"

# ── Phase 5: python deps ───────────────────────────────────────────────────
phase "5/9  Python deps for quantize scripts"
pip install --quiet --upgrade \
    "gguf>=0.19" "huggingface_hub>=0.27" "accelerate>=1.0" \
    "safetensors>=0.4" "transformers>=4.40" sentencepiece einops
ok "pip deps installed"

# ── Phase 6: HF login ──────────────────────────────────────────────────────
phase "6/9  HuggingFace authentication"
if python3 -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1; then
    user=$(python3 -c "from huggingface_hub import whoami; print(whoami()['name'])")
    ok "already logged in as $user"
else
    if [ -n "${HF_TOKEN:-}" ]; then
        python3 -c "from huggingface_hub import login; login('$HF_TOKEN', add_to_git_credential=False)"
        ok "logged in via HF_TOKEN env var"
    else
        warn "No HF_TOKEN set and no cached login. Run:"
        warn "    python3 -c 'from huggingface_hub import login; login()'"
        warn "Then re-run this script."
        exit 2
    fi
fi

# ── Phase 7: download BF16 models (pinned) ─────────────────────────────────
phase "7/9  Download BF16 source models (pinned revisions)"
echo "$MODEL_REVISIONS" | while read -r repo rev; do
    [ -n "$repo" ] || continue
    echo "    → $repo @ ${rev:0:12}"
    python3 - "$repo" "$rev" <<'PYDL'
import sys
from huggingface_hub import snapshot_download
repo, rev = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo,
    revision=rev,
    allow_patterns=[
        "*.json", "*.safetensors", "*.txt", "*.model",
        "tokenizer*", "*.tiktoken",
    ],
    max_workers=8,
)
PYDL
done
ok "BF16 models cached under $HF_HOME"

# ── Phase 8: download unsloth imatrixes ────────────────────────────────────
phase "8/9  Download unsloth imatrix files"
for repo in "${IMATRIX_REPOS[@]}"; do
    base=$(basename "$repo")
    target_dir="$IMATRIX_DIR/$base"
    target="$target_dir/imatrix_unsloth.gguf_file"
    if [ -s "$target" ]; then
        sz=$(stat -c%s "$target")
        ok "$repo: $sz bytes (cached)"
        continue
    fi
    mkdir -p "$target_dir"
    url="https://huggingface.co/$repo/resolve/main/imatrix_unsloth.gguf_file"
    curl -sL "$url" -o "$target"
    sz=$(stat -c%s "$target")
    [ "$sz" -gt 100000 ] || die "imatrix for $repo too small ($sz bytes); auth issue?"
    # Validate magic bytes
    magic=$(head -c 4 "$target" | od -An -c | tr -d ' \n')
    [ "$magic" = "GGUF" ] || die "$target is not a GGUF file (magic=$magic)"
    ok "$repo: $sz bytes (downloaded)"
done

# ── Phase 9: verify calibration corpus ─────────────────────────────────────
phase "9/9  Calibration corpus check"
calib="$HIPFIRE/benchmarks/calib/calib-1m.txt"
[ -s "$calib" ] || die "missing $calib"
calib_sz=$(stat -c%s "$calib")
ok "calib-1m.txt: $calib_sz bytes"

echo
echo "═══ BOOTSTRAP COMPLETE ═══"
echo "  Work dir:     $WORK"
echo "  hipfire HEAD: $(git -C "$HIPFIRE" rev-parse --short HEAD)"
echo "  GPU arch:     $TARGET_ARCH"
echo
echo "  Next: bash $HIPFIRE/scripts/mi300x_smoke_gfx942.sh"
echo
