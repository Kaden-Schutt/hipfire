#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# rebase-onto-modular.sh — port a pre-modular branch to post-modular master.
#
# As of 0.1.20, hipfire's monolithic `engine` crate split into
# `hipfire-runtime` + per-arch crates. Branches authored against pre-modular
# master need their import paths + Cargo deps rewritten to compile.
#
# This script does the mechanical 80% — path rewrites + Cargo dep renames.
# The remaining 20% (semantic conflicts, arch-internal API changes) needs
# human judgment; the script reports which files still need attention.
#
# Usage:
#   ./scripts/rebase-onto-modular.sh                    # rebases current branch
#   ./scripts/rebase-onto-modular.sh <branch-or-ref>    # rebases that ref
#
# Outputs:
#   - The current branch is updated in place (commits are rewritten).
#   - A summary of files still needing manual review is printed at the end.
#
# Safety:
#   - Refuses to run on master / main.
#   - Refuses to run if working tree is dirty.
#   - Creates a backup tag (rebase-onto-modular-backup-$timestamp)
#     before doing anything destructive.

set -euo pipefail

REF="${1:-HEAD}"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
BACKUP="rebase-onto-modular-backup-$(date +%s)"

if [[ "$BRANCH" == "master" || "$BRANCH" == "main" ]]; then
    echo "refuse: don't rebase master/main onto itself. checkout your feature branch first." >&2
    exit 1
fi

if ! git diff-index --quiet HEAD --; then
    echo "refuse: working tree is dirty. commit or stash first." >&2
    exit 1
fi

echo "→ creating backup tag: $BACKUP"
git tag "$BACKUP" "$BRANCH"
echo "→ if anything goes wrong: git reset --hard $BACKUP"
echo

# ---- Path rewrites ---------------------------------------------------------
# Map of old paths → new paths. These run as `git mv` if the file moved AND
# the contributor's branch touches it. Run as plain text rewrites for
# `use engine::*` style imports.

declare -A PATH_MAP=(
    ["crates/engine/src/qwen35.rs"]="crates/hipfire-arch-qwen35/src/qwen35.rs"
    ["crates/engine/src/qwen35_vl.rs"]="crates/hipfire-arch-qwen35-vl/src/qwen35_vl.rs"
    ["crates/engine/src/image.rs"]="crates/hipfire-arch-qwen35-vl/src/image.rs"
    ["crates/engine/src/speculative.rs"]="crates/hipfire-arch-qwen35/src/speculative.rs"
    ["crates/engine/src/pflash.rs"]="crates/hipfire-arch-qwen35/src/pflash.rs"
    ["crates/engine/src/loop_guard.rs"]="crates/hipfire-runtime/src/loop_guard.rs"
    ["crates/engine/src/sampler.rs"]="crates/hipfire-runtime/src/sampler.rs"
    ["crates/engine/src/prompt_frame.rs"]="crates/hipfire-runtime/src/prompt_frame.rs"
    ["crates/engine/src/eos_filter.rs"]="crates/hipfire-runtime/src/eos_filter.rs"
)

# Default rule: any other crates/engine/src/X.rs → crates/hipfire-runtime/src/X.rs
# (catches llama.rs, hfq.rs, gguf.rs, tokenizer.rs, dflash.rs, ddtree.rs,
# triattn.rs, cask.rs, cpu_router.rs, weight_pager.rs, pflash.rs, etc.)

# ---- Step 1: rebase onto current master ------------------------------------

echo "→ fetching origin"
git fetch origin --quiet

if ! git merge-base --is-ancestor "$REF" origin/master 2>/dev/null; then
    echo "→ rebasing $BRANCH onto origin/master (will likely conflict — that's expected)"
    if ! git rebase origin/master; then
        echo
        echo "rebase produced conflicts. these are expected post-modular."
        echo "this script will help by rewriting paths once you resolve to master baseline."
        echo
        echo "next steps for you:"
        echo "  1. resolve each conflict by FAVORING master (theirs):  git checkout --theirs <file>"
        echo "  2. re-stage:  git add <file>"
        echo "  3. continue:  git rebase --continue"
        echo "  4. when rebase completes, re-run this script to apply path/import rewrites"
        echo
        echo "if you want to abort: git rebase --abort && git reset --hard $BACKUP"
        exit 2
    fi
fi

# ---- Step 2: explicit path renames -----------------------------------------

echo "→ applying path renames"
for OLD in "${!PATH_MAP[@]}"; do
    NEW="${PATH_MAP[$OLD]}"
    if [[ -e "$OLD" ]]; then
        # File exists at old path — should not happen post-rebase but check anyway
        mkdir -p "$(dirname "$NEW")"
        git mv "$OLD" "$NEW"
        echo "    moved: $OLD → $NEW"
    fi
done

# Default rule for any remaining engine/src/*.rs files the branch added
if [[ -d "crates/engine" ]]; then
    while IFS= read -r f; do
        REL="${f#crates/engine/}"
        NEW="crates/hipfire-runtime/$REL"
        mkdir -p "$(dirname "$NEW")"
        git mv "$f" "$NEW"
        echo "    moved (default): $f → $NEW"
    done < <(find crates/engine -type f 2>/dev/null)
    rmdir crates/engine 2>/dev/null || true
fi

# ---- Step 3: import-path rewrites ------------------------------------------
#
# Order matters: do specific arch crates first, then catch-all engine→runtime.

echo "→ rewriting import paths"

REWRITES=(
    # Arch-specific imports first
    's|use engine::qwen35::|use hipfire_arch_qwen35::qwen35::|g'
    's|use engine::qwen35_vl::|use hipfire_arch_qwen35_vl::qwen35_vl::|g'
    's|use engine::image::|use hipfire_arch_qwen35_vl::image::|g'
    's|use engine::speculative::|use hipfire_arch_qwen35::speculative::|g'
    's|use engine::pflash::|use hipfire_arch_qwen35::pflash::|g'
    # Default: everything else in engine:: → hipfire_runtime::
    's|use engine::|use hipfire_runtime::|g'
    's|engine::qwen35::|hipfire_arch_qwen35::qwen35::|g'
    's|engine::qwen35_vl::|hipfire_arch_qwen35_vl::qwen35_vl::|g'
    's|engine::image::|hipfire_arch_qwen35_vl::image::|g'
    's|engine::speculative::|hipfire_arch_qwen35::speculative::|g'
    's|engine::pflash::|hipfire_arch_qwen35::pflash::|g'
    's|engine::|hipfire_runtime::|g'
    # Cargo.toml dep renames
    's|engine = { path = "../engine"|hipfire-runtime = { path = "../hipfire-runtime"|g'
    's|engine = { path =|hipfire-runtime = { path =|g'
    's|^engine\\b|hipfire-runtime|g'
)

# Apply only to .rs and .toml under your branch's modified files.
# Use git diff against origin/master to limit scope.
MODIFIED=$(git diff --name-only origin/master..HEAD -- '*.rs' '*.toml' 2>/dev/null || true)

if [[ -z "$MODIFIED" ]]; then
    echo "    (no .rs / .toml files modified by your branch — nothing to rewrite)"
else
    while IFS= read -r f; do
        [[ -f "$f" ]] || continue
        for SED_EXPR in "${REWRITES[@]}"; do
            sed -i "$SED_EXPR" "$f" 2>/dev/null || true
        done
    done <<< "$MODIFIED"
    echo "    rewrote imports in $(echo "$MODIFIED" | wc -l) files"
fi

# ---- Step 4: stage + amend -------------------------------------------------

if ! git diff --quiet; then
    echo "→ committing path/import rewrites"
    git add -A
    git commit -m "chore(rebase): port to post-modular topology

    Auto-applied by scripts/rebase-onto-modular.sh.

    Path renames + use-statement rewrites against the 0.1.20 crate split:
      crates/engine/* → crates/hipfire-runtime/* (default)
                     → crates/hipfire-arch-qwen35/*    (qwen35, speculative, pflash)
                     → crates/hipfire-arch-qwen35-vl/* (qwen35_vl, image)
      use engine::*  → use hipfire_runtime::*  (or arch crate)
      Cargo deps:     engine = ...  → hipfire-runtime = ...

    See CHANGELOG.md and CONTRIBUTING.md for the full topology.
    "
else
    echo "    (no changes to commit — branch already aligned)"
fi

# ---- Step 5: build + tell user what's next ---------------------------------

echo
echo "→ trying cargo build (will surface any remaining semantic issues)"
if cargo build --release --features deltanet --workspace 2>&1 | tee /tmp/rebase-build.log | tail -20; then
    echo
    echo "✅ build clean. mechanical rebase complete."
    echo "   verify: cargo test --lib --features deltanet --workspace"
    echo "   if all green:  git push --force-with-lease origin $BRANCH"
    echo "   to roll back:  git reset --hard $BACKUP"
else
    echo
    echo "⚠️  build failed. mechanical rewrites done, but some manual fixes needed."
    echo "   common reasons:"
    echo "     - your branch added new paths in crates/engine/ that the script's PATH_MAP doesn't know about"
    echo "     - your branch's API depends on private fields/methods that moved with the split"
    echo "     - your branch touched daemon arch_id dispatch sites (arch_id branches restructured for trait dispatch)"
    echo "   look at /tmp/rebase-build.log for cargo's specific error messages."
    echo "   to roll back: git reset --hard $BACKUP"
    exit 3
fi
