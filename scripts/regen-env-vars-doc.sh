#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Regenerate the quick-reference table check in docs/env-vars.md.
#
# Source coverage: concrete env::var(), env::var_os(), and process.env.X reads
# in tracked Rust/TypeScript files. Reports any HIPFIRE_* source read without a
# documentation row. The generated inventory is intentionally broader
# (Python/shell token hits and retained harness knobs), so doc-only rows are
# expected and are not removal candidates.
#
# Note the (_os)? group in the regex: compiler.rs uses std::env::var_os(...)
# rather than std::env::var(...). A regex matching only env::var( would
# silently miss HIPFIRE_KERNEL_CACHE; this was caught post-merge by Codex
# stop-gate review and is the reason the recipe covers both forms.
#
# Exit codes:
#   0 - source and doc agree (or doc only has more entries than source)
#   1 - source has HIPFIRE_* vars not in the doc table
#   2 - doc table or source extraction failed

set -u
cd "$(dirname "$0")/.."

DOC=docs/env-vars.md
if [ ! -f "$DOC" ]; then
    echo "regen-env-vars-doc: $DOC not found" >&2
    exit 2
fi

src_list=$(mktemp /tmp/hipfire-env-vars-src.XXXXXX)
doc_list=$(mktemp /tmp/hipfire-env-vars-doc.XXXXXX)
trap 'rm -f "$src_list" "$doc_list"' EXIT

# Extract concrete HIPFIRE_* reads from source. Match the variable in the same
# expression so another process.env name on that line cannot leak into output.
git ls-files '*.rs' '*.ts' \
    | xargs grep -hoE 'env::var(_os)?\([[:space:]]*"HIPFIRE_[A-Z_0-9]+"\)|process\.env\.HIPFIRE_[A-Z_0-9]+' 2>/dev/null \
    | sed -E 's/.*"(HIPFIRE_[A-Z_0-9]+)".*/\1/; s/process\.env\.//' \
    | sort -u > "$src_list"

# Extract from doc table: rows of the form `| `VAR` | category | default | location |`
grep -oE '^\| `[A-Z][A-Z_0-9]*` \|' "$DOC" \
    | sed -E 's/^\| `//; s/` \|//' \
    | sort -u > "$doc_list"

src_count=$(wc -l < "$src_list")
doc_count=$(wc -l < "$doc_list")

echo "regen-env-vars-doc: source has $src_count unique env vars, doc has $doc_count"

missing_in_doc=$(comm -23 "$src_list" "$doc_list" || true)
missing_in_src=$(comm -13 "$src_list" "$doc_list" || true)

if [ -n "$missing_in_doc" ]; then
    echo
    echo "MISSING from $DOC (present in source, no doc row):"
    echo "$missing_in_doc" | sed 's/^/  - /'
fi

if [ -n "$missing_in_src" ]; then
    doc_only_count=$(printf '%s\n' "$missing_in_src" | wc -l)
    echo "regen-env-vars-doc: note: $doc_only_count documented rows are outside"
    echo "  the concrete Rust/TypeScript read set (expected for Python/shell/history)"
fi

if [ -n "$missing_in_doc" ]; then
    echo
    echo "Action: add the missing vars to the quick-reference table in $DOC"
    echo "and write a one-line entry under the relevant category guide section."
    exit 1
fi

echo "regen-env-vars-doc: source-to-table coverage ok"
exit 0
