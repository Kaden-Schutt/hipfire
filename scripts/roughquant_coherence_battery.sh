#!/usr/bin/env bash
# RoughQuant coherence battery — quantitative attractor/repetition detection over
# a prompt set, for several models. Uses FP32 DeltaNet state (FP32_STATE=1) to
# remove the Q8-state attractor confound, and includes rq-mq4path (protect-0%) as
# a control to separate sim-path fidelity from the actual protection effect.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=./target/release/examples/infer_qwen35
M=("$@"); [ ${#M[@]} -eq 0 ] && M=(mq4 rq-mq4path rq-protect5bf16)
OUT=/tmp/rq_coh_battery; mkdir -p "$OUT"
# Byte-identical committed prompts (repo rule: prompt-sensitive evidence must use
# benchmarks/prompts/*.txt, not heredocs). Mix of factual / reasoning / code so
# the attractor detector sees the regimes that actually loop. md5s recorded below.
PROMPT_FILES=(
  benchmarks/prompts/coherence_capital_france.txt
  benchmarks/prompts/coherence_sheep_reason.txt
  benchmarks/prompts/coherence_square_function.txt
  benchmarks/prompts/trains-meet.txt
  benchmarks/prompts/merge_sort_thinking_off.txt
  benchmarks/prompts/humaneval_2_truncate.txt
  benchmarks/prompts/humaneval_3_below_zero.txt
  benchmarks/prompts/lru_cache_single_blank.txt
)
echo "=== prompt provenance (md5) ==="; md5sum "${PROMPT_FILES[@]}"
source scripts/gpu-lock.sh 2>/dev/null || true
for m in "${M[@]}"; do
  : > "$OUT/$m.txt"
  for i in "${!PROMPT_FILES[@]}"; do
    P="$(cat "${PROMPT_FILES[$i]}")"
    gpu_acquire "coh-$m-$i" 2>/dev/null || true
    echo "===PROMPT $i (${PROMPT_FILES[$i]##*/})===" >> "$OUT/$m.txt"
    FP32_STATE=1 $BIN ~/.hipfire/models/qwen3.5-0.8b-$m.hfq --guards on "$P" 2>/dev/null >> "$OUT/$m.txt"
    gpu_release 2>/dev/null || true
  done
  echo "done: $m"
done
echo "=== detector aggregation ==="
python3 - "$OUT" "${M[@]}" <<'PY'
import sys, re, glob
out=sys.argv[1]; models=sys.argv[2:]
def metrics(text):
    # per-prompt blocks
    blocks=re.split(r'===PROMPT \d+[^=]*===', text)[1:]
    rows=[]
    for b in blocks:
        words=re.findall(r"\S+", b)
        n=len(words)
        if n<10: rows.append((n,1.0,0.0)); continue
        uniq=len(set(words))/n
        # max 5-gram repetition density (fraction of 5-grams that recur)
        grams=[tuple(words[i:i+5]) for i in range(n-5)]
        from collections import Counter
        c=Counter(grams); rep=sum(v for v in c.values() if v>1)/max(1,len(grams))
        rows.append((n,uniq,rep))
    return rows
print(f"{'model':18s} {'prompts':>7s} {'avg_uniq':>9s} {'avg_5gram_rep':>13s} {'#attractor':>10s}")
for m in models:
    f=f"{out}/{m}.txt"
    try: t=open(f).read()
    except: print(f"{m}: no output"); continue
    rows=metrics(t)
    if not rows: print(f"{m}: empty"); continue
    import statistics as st
    au=st.mean(r[1] for r in rows); ar=st.mean(r[2] for r in rows)
    natt=sum(1 for r in rows if r[2]>0.30 or r[1]<0.35)  # attractor: heavy 5gram rep or low unique
    print(f"{m:18s} {len(rows):7d} {au:9.3f} {ar:13.3f} {natt:10d}")
print("\n(attractor = 5gram-rep>0.30 OR unique-ratio<0.35. Lower rep / higher uniq = better.")
print(" rq-mq4path vs mq4 isolates sim-path fidelity; rq-protect5bf16 vs rq-mq4path isolates protection.)")
PY
