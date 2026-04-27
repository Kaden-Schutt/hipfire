#!/usr/bin/env bash
# fetch_calibration_corpus.sh — build a corpus by concatenating one or
# more HuggingFace datasets, flattened to a ChatML-wrapped plain-text
# stream. **Used by two distinct flows; do not confuse them:**
#
#   1. Sidecar calibration  — feed to `triattn_validate --corpus`,
#      produces `.triattn.bin`.
#
#   2. DFlash draft training — feed to `generate_target_responses.py`
#      (Phase C2 of task #93), then to `dflash_train_poc.py` after
#      target-regeneration. Produces `.hfq` draft weights.
#
# Same recipes (`agentic`, `agentic_xl`, `reasoning`, `blended`, `all`)
# serve both flows; the consumer differs. The legacy file name kept for
# git history continuity — do not rename.
#
# Recipes (bundles):
#   agentic     — lambda/hermes-agent-reasoning-traces   (tool-calling traces)
#   agentic_xl  — hermes + nemotron_agentic + hermes_filtered + tool_calls_mt + xlam
#                 (2026-04-19: ~90× bigger, for 8× cluster runs. ~32B+ tokens.
#                 Nemotron alone is 335k rows / ~32B tokens — streaming load.)
#   reasoning   — Opus-4.6-Reasoning + Qwen3.5-reasoning + claude-opus-4.6-10000x
#   chat        — fka/prompts.chat                       (persona-style prompts)
#   blended     — mix of agentic + reasoning + chat
#   all         — every known source, deduped
#
# Or: --dataset <name>[,<name>,...] to pick individual sources.
#
# Usage:
#   bash fetch_calibration_corpus.sh [OUT_PATH] [--recipe NAME] [--dataset n1,n2]
#                                    [--max-rows N]
#
# Default OUT_PATH: /root/calibration_corpus.txt
# Default recipe: agentic
#
# Needs: pip install huggingface_hub pyarrow
# Set HF_TOKEN if any source is gated.

set -euo pipefail

OUT="${1:-}"
RECIPE="agentic"
DATASETS=""
MAX_ROWS="${MAX_ROWS:-0}"

# Strip the positional if given
if [ -n "$OUT" ] && [[ "$OUT" != --* ]]; then
    shift
else
    OUT="/root/calibration_corpus.txt"
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --recipe) RECIPE="$2"; shift 2 ;;
        --dataset) DATASETS="$2"; shift 2 ;;
        --max-rows) MAX_ROWS="$2"; shift 2 ;;
        --help|-h) sed -n '1,25p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

# Resolve recipe → dataset list if user didn't give --dataset
if [ -z "$DATASETS" ]; then
    case "$RECIPE" in
        agentic)     DATASETS="hermes" ;;
        reasoning)   DATASETS="opus_reason,qwen_reason,claude_opus" ;;
        chat)        DATASETS="prompts_chat" ;;
        blended)     DATASETS="hermes,opus_reason,qwen_reason,claude_opus,prompts_chat" ;;
        all)         DATASETS="hermes,opus_reason,qwen_reason,claude_opus,prompts_chat" ;;
        # agentic_xl: 2026-04-19 recipe for 8× cluster. ~90× bigger than
        # plain agentic (Nemotron alone is 335k rows / ~32B tokens).
        agentic_xl)  DATASETS="hermes,nemotron_agentic,hermes_filtered,tool_calls_mt,xlam" ;;
        *) echo "unknown recipe: $RECIPE" >&2; exit 2 ;;
    esac
fi

log() { printf '[calib-corpus] %s\n' "$*"; }
log "out:       $OUT"
log "recipe:    $RECIPE"
log "datasets:  $DATASETS"
log "max_rows:  ${MAX_ROWS:-all}"

PY=python3
if [ -f /root/pytorch_env/bin/python3 ]; then PY=/root/pytorch_env/bin/python3; fi

"$PY" - "$OUT" "$DATASETS" "$MAX_ROWS" <<'PY'
import csv, json, os, sys
from pathlib import Path

out_path, datasets_str, max_rows_str = sys.argv[1], sys.argv[2], sys.argv[3]
max_rows = int(max_rows_str) if max_rows_str else 0
wanted = [d.strip() for d in datasets_str.split(',') if d.strip()]

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    import pyarrow.parquet as pq
except ImportError as e:
    print(f"[calib-corpus] ERROR: missing deps ({e}); pip install huggingface_hub pyarrow")
    sys.exit(3)

# ChatML emitter — shared across all source parsers.
def em(role, value):
    if role == 'system':    return f"<|im_start|>system\n{value}<|im_end|>"
    if role in ('human','user'): return f"<|im_start|>user\n{value}<|im_end|>"
    if role in ('gpt','assistant'): return f"<|im_start|>assistant\n{value}<|im_end|>"
    if role == 'tool':      return f"<|im_start|>tool\n{value}<|im_end|>"
    return ""

# ── Source adapters. Each yields a flattened ChatML block per row. ──
def src_hermes(cap):
    root = snapshot_download('lambda/hermes-agent-reasoning-traces', repo_type='dataset')
    count = 0
    for cfg in ('kimi', 'glm-5.1'):
        p = Path(root) / 'data' / cfg / 'train.parquet'
        if not p.exists(): continue
        t = pq.read_table(str(p))
        for row in t.to_pylist():
            lines = [em(c.get('from',''), c.get('value','')) for c in row.get('conversations', [])]
            lines = [l for l in lines if l]
            if lines:
                yield '\n'.join(lines) + '\n\n'
                count += 1
                if cap and count >= cap: return

def src_opus_reason(cap):
    p = hf_hub_download('nohurry/Opus-4.6-Reasoning-3000x-filtered',
                        'distilled_corpus_400k_with_cot-filtered.jsonl', repo_type='dataset')
    count = 0
    for line in open(p):
        m = json.loads(line)
        prob = m.get('problem','')
        think = m.get('thinking','')
        sol = m.get('solution','')
        asst = f"<think>\n{think}\n</think>\n{sol}" if think else sol
        block = '\n'.join([em('user', prob), em('assistant', asst)]) + '\n\n'
        yield block
        count += 1
        if cap and count >= cap: return

def src_qwen_reason(cap):
    p = hf_hub_download('Jackrong/Qwen3.5-reasoning-700x',
                        'distilled_stage2.jsonl', repo_type='dataset')
    count = 0
    for line in open(p):
        m = json.loads(line)
        convs = m.get('conversation', [])
        if isinstance(convs, list) and convs:
            lines = [em(c.get('from',''), c.get('value','')) for c in convs]
        else:
            # Fallback: input+output pair
            lines = [em('user', m.get('input','')), em('assistant', m.get('output',''))]
        lines = [l for l in lines if l]
        if lines:
            yield '\n'.join(lines) + '\n\n'
            count += 1
            if cap and count >= cap: return

def src_claude_opus(cap):
    p = hf_hub_download('Roman1111111/claude-opus-4.6-10000x',
                        'opus46_final.jsonl', repo_type='dataset')
    count = 0
    for line in open(p):
        m = json.loads(line)
        msgs = m.get('messages', [])
        lines = [em(msg.get('role',''), msg.get('content','')) for msg in msgs]
        lines = [l for l in lines if l]
        if lines:
            yield '\n'.join(lines) + '\n\n'
            count += 1
            if cap and count >= cap: return

def src_prompts_chat(cap):
    p = hf_hub_download('fka/prompts.chat', 'prompts.csv', repo_type='dataset')
    count = 0
    with open(p, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            act = row.get('act','').strip()
            prompt = row.get('prompt','').strip()
            if not prompt: continue
            # prompts.chat has no canonical assistant response; emit the act
            # as a system persona + prompt as user turn. Not ideal but the
            # tokens are representative of task-framing prompts we care about.
            sys_turn = f"You are acting as: {act}." if act else "You are a helpful assistant."
            lines = [em('system', sys_turn), em('user', prompt)]
            yield '\n'.join(lines) + '\n\n'
            count += 1
            if cap and count >= cap: return

def src_nemotron_agentic(cap):
    """nvidia/Nemotron-Agentic-v1 — 335k rows, ~32B tokens. Schema:
    messages=[{role, content}], tools=[{...}], reasoning=str.
    Splits: interactive_agent (19k), tool_calling (316k).
    Uses datasets.load_dataset (streaming) so we don't materialize 5GB.
    """
    from datasets import load_dataset
    count = 0
    for split in ("tool_calling", "interactive_agent"):
        try:
            ds = load_dataset("nvidia/Nemotron-Agentic-v1", split=split, streaming=True)
        except Exception as e:
            print(f"[calib-corpus]   nemotron {split} load failed: {e}")
            continue
        for row in ds:
            msgs = row.get("messages", []) or []
            tools = row.get("tools")
            lines = []
            # If the row has tools but the first message isn't system, synthesize
            # a system turn with <tools> JSON so the draft sees the structural
            # framing it'll encounter at inference.
            has_system = bool(msgs) and msgs[0].get("role") == "system"
            if tools and not has_system:
                try:
                    tjson = json.dumps(tools) if not isinstance(tools, str) else tools
                except Exception:
                    tjson = str(tools)
                lines.append(em('system',
                                "You are a function calling AI model. "
                                f"You are provided with function signatures within <tools></tools> XML tags.\n<tools>\n{tjson}\n</tools>"))
            for m in msgs:
                lines.append(em(m.get('role', ''), m.get('content', '')))
            lines = [l for l in lines if l]
            if lines:
                yield '\n'.join(lines) + '\n\n'
                count += 1
                if cap and count >= cap: return

def src_hermes_filtered(cap):
    """DJLougen/hermes-agent-traces-filtered — 3,679 quality-filtered rows.
    Same ShareGPT schema as lambda/hermes (conversations=[{from, value}]).
    """
    from datasets import load_dataset
    ds = load_dataset("DJLougen/hermes-agent-traces-filtered", split="train")
    count = 0
    for row in ds:
        convs = row.get("conversations", []) or []
        lines = [em(c.get('from', ''), c.get('value', '')) for c in convs]
        lines = [l for l in lines if l]
        if lines:
            yield '\n'.join(lines) + '\n\n'
            count += 1
            if cap and count >= cap: return

def src_tool_calls_mt(cap):
    """interstellarninja/tool-calls-multiturn — 1,890 rows, 2026 multi-turn.
    ShareGPT schema: conversations=[{from, value}].
    """
    from datasets import load_dataset
    ds = load_dataset("interstellarninja/tool-calls-multiturn", split="train")
    count = 0
    for row in ds:
        convs = row.get("conversations", []) or []
        lines = [em(c.get('from', ''), c.get('value', '')) for c in convs]
        lines = [l for l in lines if l]
        if lines:
            yield '\n'.join(lines) + '\n\n'
            count += 1
            if cap and count >= cap: return

def src_xlam(cap):
    """Salesforce/xlam-function-calling-60k — 60k rows, single-turn breadth.
    Schema: {query, tools, answers}. We emit as system(tools)+user(query)+
    assistant(answers json). Useful for exposing the draft to diverse API
    signatures.
    """
    from datasets import load_dataset
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    count = 0
    for row in ds:
        query = row.get('query', '')
        tools = row.get('tools', '')
        answers = row.get('answers', '')
        if not query or not answers:
            continue
        tjson = tools if isinstance(tools, str) else json.dumps(tools)
        ajson = answers if isinstance(answers, str) else json.dumps(answers)
        lines = [
            em('system',
               f"You are a function calling AI model. Available tools:\n<tools>\n{tjson}\n</tools>"),
            em('user', query),
            em('assistant', f"<tool_call>\n{ajson}\n</tool_call>"),
        ]
        yield '\n'.join(lines) + '\n\n'
        count += 1
        if cap and count >= cap: return

ADAPTERS = {
    'hermes':            src_hermes,
    'opus_reason':       src_opus_reason,
    'qwen_reason':       src_qwen_reason,
    'claude_opus':       src_claude_opus,
    'prompts_chat':      src_prompts_chat,
    # 2026-04-19 additions (agentic_xl recipe)
    'nemotron_agentic':  src_nemotron_agentic,
    'hermes_filtered':   src_hermes_filtered,
    'tool_calls_mt':     src_tool_calls_mt,
    'xlam':              src_xlam,
}

unknown = [d for d in wanted if d not in ADAPTERS]
if unknown:
    print(f"[calib-corpus] ERROR: unknown datasets: {unknown}")
    print(f"               known: {list(ADAPTERS.keys())}")
    sys.exit(2)

total_rows = 0
total_bytes = 0
per_cap = max_rows // len(wanted) if max_rows > 0 else 0

os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as fout:
    for name in wanted:
        print(f"[calib-corpus] fetching {name} (cap={per_cap or 'all'}) ...")
        try:
            for block in ADAPTERS[name](per_cap):
                fout.write(block)
                total_rows += 1
                total_bytes += len(block.encode('utf-8'))
        except Exception as e:
            print(f"[calib-corpus]   WARN {name} failed: {e} (continuing)")
print(f"[calib-corpus] wrote {total_rows} conversations, {total_bytes/1e6:.1f} MB → {out_path}")
PY

log "done: $(wc -c < "$OUT" | awk '{printf "%.1f MB\n", $1/1024/1024}') at $OUT"
