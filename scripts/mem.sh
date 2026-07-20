#!/usr/bin/env bash
# agent-memory — git-tracked markdown agent memory with lexical (ripgrep) recall.
#
# No database, no embeddings, no service: notes are .agent-memory/notes/*.md
# (YAML frontmatter), recall is ripgrep ranked by match density. The right tool
# for a few-hundred terse, keyword-dense engineering notes. Project findings live
# here, in-repo, diffable, shared with contributors. See .agent-memory/README.md (docs/skills/ is retired).
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
NOTES="$ROOT/.agent-memory/notes"
mkdir -p "$NOTES"
have_rg() { command -v rg >/dev/null 2>&1; }

usage() {
  cat <<'U'
agent-memory (lexical, git-backed)
  mem.sh recall <query terms...>            ripgrep-ranked recall (top notes for the query)
  mem.sh remember <slug> "<title>" [tags]   create a new note (edit body, then commit)
  mem.sh list                               list all notes
  mem.sh path                               print the notes directory
Project findings go here (committed, shared). Personal/fleet notes stay in global memory.
U
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  recall)
    [ "$#" -ge 1 ] || { echo "usage: mem.sh recall <query terms>"; exit 1; }
    pat="$(printf '%s' "$*" | tr -s ' ' '|')"
    if have_rg; then
      rg --count-matches -i -- "$pat" "$NOTES" 2>/dev/null | sort -t: -k2 -rn | head -8 \
      | while IFS=: read -r f c; do
          t="$(grep -m1 '^title:' "$f" 2>/dev/null | sed 's/^title:[[:space:]]*//')"
          [ -n "$t" ] || t="$(basename "$f" .md)"
          snip="$(rg -i -m1 -- "$pat" "$f" 2>/dev/null | sed 's/^[[:space:]]*//' | cut -c1-100)"
          printf '  %3sx  %s\n         %s\n         … %s\n' "$c" "$t" "${f#"$ROOT"/}" "$snip"
        done
    else
      grep -rilE "$pat" "$NOTES" 2>/dev/null | head -8 | sed "s#$ROOT/##"
    fi
    ;;
  remember)
    slug="${1:?slug required}"; title="${2:?title required}"; tags="${3:-}"
    f="$NOTES/$slug.md"
    [ -e "$f" ] && { echo "already exists: ${f#"$ROOT"/}"; exit 1; }
    printf -- '---\ntitle: %s\ndate: %s\ntags: [%s]\n---\n\n<write the finding here; terse; link related notes by slug>\n' \
      "$title" "$(date +%F)" "$tags" > "$f"
    echo "created ${f#"$ROOT"/} — edit the body, then: git add -A .agent-memory && git commit"
    ;;
  list) ls -1 "$NOTES"/*.md 2>/dev/null | sed "s#$NOTES/##" || echo "(no notes yet)" ;;
  path) echo "$NOTES" ;;
  *) usage ;;
esac
