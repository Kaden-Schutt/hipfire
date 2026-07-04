#!/usr/bin/env bash
# Fetch Heretic's generic +/- prompt sets into one-prompt-per-line text files for
# the hipfire-steer driver. Reproduces third_party/heretic/config.default.toml:
#   good = mlabonne/harmless_alpaca, bad = mlabonne/harmful_behaviors
#   train[:400] drives the direction; test[:100] is the held-out eval split.
# Text-only (image modality deferred). Uses the HF datasets-server JSON API, so
# no `datasets`/`huggingface_hub` install is required — just curl + python3.
#
# Output (gitignored — harmful_behaviors carries sensitive jailbreak prompts):
#   data/heretic/{good,bad}_prompts.txt  data/heretic/{good,bad}_eval.txt
set -euo pipefail

OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/heretic"
mkdir -p "$OUT"
API="https://datasets-server.huggingface.co/rows"

# fetch <dataset> <split> <count> <outfile>: page through /rows (100/page),
# emit the `text` column one whitespace-normalized prompt per line.
fetch() {
    local ds="$1" split="$2" count="$3" out="$4" offset=0 page
    : >"$out"
    while [ "$offset" -lt "$count" ]; do
        page=$((count - offset < 100 ? count - offset : 100))
        curl -fsS --max-time 30 \
            "${API}?dataset=${ds}&config=default&split=${split}&offset=${offset}&length=${page}" \
            | python3 -c "import sys,json
for r in json.load(sys.stdin)['rows']:
    print(' '.join(r['row']['text'].split()))" >>"$out"
        offset=$((offset + page))
    done
    echo "  $(wc -l <"$out" | tr -d ' ') lines -> ${out#"$OUT"/}"
}

echo "Fetching Heretic generic prompt sets into $OUT ..."
fetch mlabonne/harmless_alpaca train 400 "$OUT/good_prompts.txt"
fetch mlabonne/harmful_behaviors train 400 "$OUT/bad_prompts.txt"
fetch mlabonne/harmless_alpaca test 100 "$OUT/good_eval.txt"
fetch mlabonne/harmful_behaviors test 100 "$OUT/bad_eval.txt"
echo "Done."
