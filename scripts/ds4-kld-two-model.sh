#!/usr/bin/env bash
# Exact full-vocab KLD, MQ2-Lloyd vs FP4, with TEACHER FORCING.
#
# Pass 1: FP4 generates freely and its committed tokens become the reference
#         sequence; its per-position logits are captured.
# Pass 2: MQ2 is FORCED along that exact sequence, so position i in both dumps
#         is conditioned on identical context and KL(P_i || Q_i) is meaningful.
#
# Without pass 2's forcing the models diverge at token ~0 and the "KL" compares
# unrelated continuations (measured ~30 nats — that was a bug in the method,
# not a property of the quant).
set -u

OUT=/tmp/claude-1000/-tmp/ac2132a4-a2b1-4dc8-af3f-a222009177ec/scratchpad
REPO=/home/nick/claude/wt/ds4-paging-beta
EXE=$REPO/target/release/examples/daemon
FP4=/data/hipfire-models/deepseek-v4-flash-0731.fp4
MQ2=/data/hipfire-models/deepseek-v4-flash-0731.mq2lloyd
N=${N:-64}
PROMPT="Explain how a hash table works, step by step, and describe what happens on a collision."

cd "$REPO" || exit 1
rm -f "$OUT/kld2.done" "$OUT/kld2-fp4.bin" "$OUT/kld2-mq2.bin"

run() {  # $1=model $2=tag $3=dump $4=forced-or-empty
  printf '%s\n%s\n%s\n' \
    "{\"type\":\"load\",\"model\":\"$1\",\"params\":{\"max_seq\":4096,\"dspark_mode\":\"off\",\"mtp_mode\":\"off\"}}" \
    "{\"type\":\"generate\",\"id\":\"g\",\"attempt_id\":0,\"prompt\":\"$PROMPT\",\"temperature\":0.0,\"max_tokens\":$N,\"repeat_penalty\":1.0}" \
    '{"type":"unload"}' \
  | systemd-run --user --scope --quiet -p MemoryMax=60G -p MemorySwapMax=0 \
      env HIPFIRE_EMIT_TOKEN_IDS=1 \
          HIPFIRE_DEEPSEEK4_GRAPH=0 \
          HIPFIRE_DEEPSEEK4_EXPERT_CACHE_GB=24 \
          HIPFIRE_DEEPSEEK4_EXPERT_CACHE_RESERVE_GB=24 \
          HIPFIRE_DS4_LOGIT_DUMP="$3" \
          HIPFIRE_DS4_FORCE_TOKENS="$4" \
          "$EXE" > "$OUT/kld2-$2.jsonl" 2> "$OUT/kld2-$2.err"
}

t0=$(date +%s)
run "$FP4" fp4 "$OUT/kld2-fp4.bin" ""
REF=$(grep -a '"type":"committed"' "$OUT/kld2-fp4.jsonl" \
      | sed 's/.*"tok_id":[[:space:]]*\([0-9]*\).*/\1/' | paste -sd, -)
echo "reference sequence: $(echo "$REF" | tr ',' '\n' | wc -l) tokens" > "$OUT/kld2.seq"
run "$MQ2" mq2 "$OUT/kld2-mq2.bin" "$REF"
t1=$(date +%s)

python3 - "$OUT/kld2-fp4.bin" "$OUT/kld2-mq2.bin" "$OUT/kld2-fp4.jsonl" "$OUT/kld2-mq2.jsonl" > "$OUT/kld2.summary" 2>&1 <<'PY'
import struct, sys, math, json
def load(p):
    out=[]
    with open(p,'rb') as f:
        while True:
            h=f.read(8)
            if len(h)<8: break
            pos,v=struct.unpack('<II',h)
            d=f.read(4*v)
            if len(d)<4*v: break
            out.append(struct.unpack(f'<{v}f',d))
    return out
def toks(p):
    o=[]
    for line in open(p,errors="replace"):
        if '"type":"committed"' in line:
            try: o.append(json.loads(line)["tok_id"])
            except Exception: pass
    return o
P,Q=load(sys.argv[1]),load(sys.argv[2])
tp,tq=toks(sys.argv[3]),toks(sys.argv[4])
n=min(len(P),len(Q))
m=min(len(tp),len(tq))
agree=sum(1 for i in range(m) if tp[i]==tq[i])
print(f"positions: fp4={len(P)} mq2={len(Q)} -> comparing {n}")
print(f"forced-sequence check: {agree}/{m} committed tokens identical "
      f"({'OK - teacher forcing held' if agree==m else 'MISMATCH - forcing failed, results invalid'})")
if n==0: print("NO LOGITS"); raise SystemExit
def ls(v):
    mx=max(v); s=sum(math.exp(x-mx) for x in v); z=mx+math.log(s)
    return [x-z for x in v]
kl=[]
for i in range(n):
    a,b=ls(list(P[i])),ls(list(Q[i]))
    t=0.0
    for x,y in zip(a,b):
        px=math.exp(x)
        if px>1e-12: t+=px*(x-y)
    kl.append(t)
k=sorted(kl); mean=sum(kl)/len(kl)
def pc(q): return k[min(len(k)-1,int(q*len(k)))]
print(f"\nKL(P_fp4 || Q_mq2) — full 129280-entry vocab, nats, teacher-forced")
print(f"  positions : {len(kl)}")
print(f"  mean      : {mean:.5f}")
print(f"  median    : {pc(0.50):.5f}")
print(f"  p90       : {pc(0.90):.5f}")
print(f"  p99       : {pc(0.99):.5f}")
print(f"  max       : {k[-1]:.5f}")
print("  first 8   : " + ", ".join(f"{x:.4f}" for x in kl[:8]))
PY

{ echo "wall=$((t1-t0))s N=$N"; cat "$OUT/kld2.seq"; cat "$OUT/kld2.summary"; } > "$OUT/kld2.final" 2>&1
echo 0 > "$OUT/kld2.done"
