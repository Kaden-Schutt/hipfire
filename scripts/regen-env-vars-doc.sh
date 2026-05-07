#!/usr/bin/env bash
# Regenerate the quick-reference table check in docs/env-vars.md.
#
# Source-of-truth: env::var(), env::var_os(), and process.env.X across tracked
# files. Surfaces the diff between source-extracted env vars and the table in
# docs/env-vars.md so a contributor can see (a) any HIPFIRE_* var that was
# added to source without a doc row, and (b) any doc row that no longer has
# a matching source reference.
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

# Extract from source: env::var(...), env::var_os(...), process.env.X
git ls-files | grep -E '\.(rs|ts)$' \
    | xargs grep -hE 'env::var(_os)?\("HIPFIRE_|process\.env\.HIPFIRE_' 2>/dev/null \
    | grep -oE '(env::var(_os)?\("[A-Z_0-9]+"\)|process\.env\.[A-Z_0-9]+)' \
    | sed -E 's/env::var(_os)?\("//; s/"\)$//; s/process\.env\.//' \
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
    echo
    echo "STALE in $DOC (doc row, no source reference - candidates for retire):"
    echo "$missing_in_src" | sed 's/^/  - /'
fi

if [ -n "$missing_in_doc" ]; then
    echo
    echo "Action: add the missing vars to the quick-reference table in $DOC"
    echo "and write a one-line entry under the relevant category guide section."
    exit 1
fi

if [ -z "$missing_in_doc" ] && [ -z "$missing_in_src" ]; then
    echo "regen-env-vars-doc: source and doc agree (no missing or stale entries)"
fi
exit 0
