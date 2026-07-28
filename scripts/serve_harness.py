#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
"""serve_harness — the go-to tool for testing user-facing serve behavior.

Model-agnostic. Drives a hipfire serve and captures, per turn, everything needed
to tell coherent output from runaway/empty and to read real perf:

  finish_reason (stop vs LENGTH=runaway) · content/think word-split + preview ·
  cached_tokens (prefix cache) · prefill_ms / prefill_tok_s · decode_tok_s · tau ·
  ttft · attractor tiers · recall · empty + runaway flags.

TWO-STEP DISCIPLINE: `--show-config` resolves and prints the CONCRETE config —
every sampling value with its source ([registry]/[default]/[recipe]/[explicit]),
`thinking_budget med -> 2048 tok`, `max_tokens`, `kv`, `mtp`, model — WITHOUT
running, so you eyeball exactly what is and isn't set before anything fires. The
sampling DEFAULT is production-sampled (the model's registry recommended_settings),
never greedy or the 0.3 CLI fallback.

Modes:
  battery — single-turn genre battery (code/reason/factual/prose/instruct), fresh
            conversation each prompt (no cache); surfaces genre-specific runaway.
  chain   — the genre prompts chained into one growing conversation; exercises the
            prefix cache (cached_tokens) + cross-turn prefill/decode.
  session — an existing N-turn session file (recall + attractor), e.g. the 8-turn
            session_coding.json the coherence gate uses.
"""
import argparse, atexit, json, os, re, shutil, signal, subprocess, sys, time, urllib.request

# Mirror of the Rust configuration schema's reasoning budgets (resolved here so the pre-flight shows the
# concrete token cap, not just the preset name).
THINKING_BUDGET = {"low": 512, "med": 2048, "high": 8192, "xhigh": 24576, "max": 32768, "uncapped": 0}

# Qwen card recipes (thinking-mode general/coding, instruct non-thinking). pp varies
# by model (a3b general uses 1.5; 27b general uses 0) so registry mode is preferred;
# these are explicit overrides for the sweep. reasoning_effort=none drives non-thinking.
RECIPES = {
    "general": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
    "coding":  {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
    "nothink": {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                "reasoning_effort": "none"},
}
SAMPLE_KEYS = ["temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty", "reasoning_effort"]

GENRE_BATTERY = [
    ("code",     "Write a Python function `merge_sorted(a, b)` that merges two already-sorted lists "
                 "into one sorted list without using sorted(). Include a short docstring."),
    ("reason",   "A train goes 60 mph for 2.5 hours, then 40 mph for 1.5 hours. How far did it travel "
                 "in total? Show your steps and give the final number."),
    ("factual",  "What causes the seasons on Earth? Answer in exactly three sentences."),
    ("prose",    "Write a four-sentence story about a lighthouse keeper who finds something unexpected "
                 "washed up on the rocks."),
    ("instruct", "List exactly five tips for writing maintainable code, as a numbered list, one line each."),
]

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_sampling(spec, tag, registry_path):
    """Return (values_dict, source_dict). spec: 'registry'|'registry:<mode>'|'greedy'|'recipe:NAME'|json string."""
    src = {}
    if spec == "greedy":
        return {"temperature": 0.0}, {"temperature": "explicit(greedy)"}
    if spec.startswith("recipe:"):
        name = spec.split(":", 1)[1]
        if name not in RECIPES:
            sys.exit(f"unknown recipe {name!r}; choose {list(RECIPES)}")
        return dict(RECIPES[name]), {k: f"recipe({name})" for k in RECIPES[name]}
    if spec.startswith("json:"):
        v = json.loads(spec[5:])
        return v, {k: "explicit" for k in v}
    if spec == "registry" or spec.startswith("registry:"):
        # production behavior: the serve applies the model's recommended_settings.
        # We resolve them HERE so they are explicit + visible (and reproducible).
        # `registry` = the default profile (recommended_settings). `registry:<mode>`
        # (general|coding|instruct) selects a named per-mode sampling profile;
        # `general` falls back to recommended_settings when no profile map is set.
        profile = spec.split(":", 1)[1] if ":" in spec else None
        rec = {}
        try:
            reg = json.load(open(registry_path))["models"]
            entry = reg.get(tag, {})
            if profile is None:
                rec = entry.get("recommended_settings", {}) or {}
            else:
                profiles = entry.get("sampling_profiles") or {}
                if profile == "general":
                    rec = profiles.get("general") or entry.get("recommended_settings", {}) or {}
                else:
                    rec = profiles.get(profile, {}) or {}
        except Exception as e:
            print(f"  [warn] could not read registry {registry_path}: {e}", file=sys.stderr)
        label = f"registry({tag}:{profile})" if profile else f"registry({tag})"
        vals, source = {}, {}
        for k in ["temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty"]:
            if k in rec:
                vals[k] = rec[k]; source[k] = label
        # The instruct profile is the non-thinking mode: drive reasoning_effort=none
        # through the existing budget machinery (no daemon request-JSON change).
        if profile == "instruct":
            vals["reasoning_effort"] = "none"; source["reasoning_effort"] = label
        if not vals:
            what = f"sampling_profiles.{profile}" if profile else "recommended_settings"
            sys.exit(f"  registry has no {what} for tag {tag!r} — refusing to fall back to a "
                     f"naive default. Pass --tag <registry-tag> or --sampling recipe:coding explicitly.")
        return vals, source
    sys.exit(f"bad --sampling {spec!r}")


def infer_tag(model_path):
    """Best-effort registry tag from a model filename, e.g. qwen3.6-27b-awq.mq4 -> qwen3.6:27b."""
    b = os.path.basename(model_path)
    m = re.match(r"(qwen3\.\d+)-(\d+b(?:-a\d+b)?)", b)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return None


def build_config(args):
    tag = args.tag or infer_tag(args.model)
    samp, samp_src = resolve_sampling(args.sampling, tag, args.registry)
    think_cap = THINKING_BUDGET.get(args.thinking)
    if think_cap is None:
        sys.exit(f"thinking_budget {args.thinking!r} not a key of {list(THINKING_BUDGET)}")
    return {
        "model": args.model, "tag": tag, "kv": args.kv, "mtp": args.mtp,
        "dflash": getattr(args, "dflash", "off"),
        "thinking_budget": args.thinking, "thinking_cap_tokens": think_cap,
        "max_tokens": args.max_tokens, "sampling": samp, "sampling_source": samp_src,
        "mode": args.mode, "port": args.port, "seed": getattr(args, "seed", None),
        "prompts_file": getattr(args, "prompts_file", None),
    }


def load_prompt_battery(prompts_file):
    """Return [(genre, prompt), ...] from a --prompts-file JSON, else the built-in GENRE_BATTERY."""
    if not prompts_file:
        return list(GENRE_BATTERY)
    import json
    rows = json.load(open(prompts_file))
    return [(r.get("genre", "prose"), r["prompt"]) for r in rows]


def show_config(cfg):
    print("==================== serve_harness pre-flight (CONFIRM before run) ====================")
    print(f"  model         : {cfg['model']}")
    print(f"  registry tag  : {cfg['tag'] or '(none — sampling cannot be registry-resolved)'}")
    print(f"  kv_mode       : {cfg['kv']}   mtp_mode: {cfg['mtp']}   mode: {cfg['mode']}")
    print(f"  seed          : {cfg.get('seed')}   prompts_file: {cfg.get('prompts_file') or '(built-in battery)'}")
    print(f"  thinking_budget: {cfg['thinking_budget']} -> {cfg['thinking_cap_tokens']} tok (CONCRETE cap)")
    _cap = cfg['thinking_cap_tokens']
    _note = ('uncapped think budget' if _cap == 0
             else f'> think cap {_cap} — model can answer' if cfg['max_tokens'] > _cap
             else f'<= think cap {_cap} — INVALID (think-only); run will hard-fail')
    print(f"  max_tokens     : {cfg['max_tokens']}  ({_note})")
    print("  sampling (what IS set):")
    for k in SAMPLE_KEYS:
        if k in cfg["sampling"]:
            print(f"      {k:18}= {cfg['sampling'][k]:<8} [{cfg['sampling_source'].get(k,'?')}]")
    notset = [k for k in SAMPLE_KEYS if k not in cfg["sampling"]]
    print(f"  sampling (NOT set, serve/daemon default applies): {', '.join(notset) or '(none)'}")
    print("=======================================================================================")


# ---------- serve spawn (robust, self-contained) ----------
_serve_proc = None

def _kill_serve():
    """Kill ONLY this harness's own native serve tree (the Rust CLI + its child daemon),
    scoped by process group — NOT a broad `pkill -x daemon`, which would execute
    the parallel autoresearch daemons pinned to OTHER GPUs. spawn_serve starts the
    serve in its own session (start_new_session) so this group kill is exact."""
    global _serve_proc
    if _serve_proc is not None:
        try:
            os.killpg(os.getpgid(_serve_proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try: _serve_proc.kill()
            except Exception: pass
    _serve_proc = None


def _native_cli():
    """Resolve the Rust control-plane binary."""
    candidates = [
        os.environ.get("HIPFIRE_CLI_BIN"),
        os.path.join(REPO, "target", "release", "hipfire"),
        os.path.expanduser("~/.hipfire/bin/hipfire"),
        shutil.which("hipfire"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    sys.exit("serve_harness: native hipfire CLI not found; build `cargo build --release -p hipfire-cli` "
             "or set HIPFIRE_CLI_BIN")


def _write_native_config(cfg, home):
    """Write the isolated harness configuration in the native sparse TOML format."""
    text = f"""[serve]
host = "127.0.0.1"
port = {cfg["port"]}
default_model = {json.dumps(cfg["model"])}

[memory]
max_seq = {cfg.get("max_seq", 32768)}
kv_cache = {json.dumps(cfg["kv"])}

[speculation]
dflash = {json.dumps(cfg["dflash"])}
mtp = {json.dumps(cfg["mtp"])}
ngram = "off"

[generation]
max_tokens = 16384

[reasoning]
budget = {json.dumps(cfg["thinking_budget"])}
"""
    with open(os.path.join(home, ".hipfire", "config.toml"), "w") as handle:
        handle.write(text)


def _native_service_warm(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
            health = json.load(response)
        return bool(health.get("model")) and not health.get("loading_model")
    except Exception:
        return False


def spawn_serve(cfg, home, log):
    """Spawn native `hipfire serve` with the resolved serve config; retry on the flaky
    daemon-spawn; return when warm. The per-request sampling is sent by the driver, so
    one serve handles all recipes/modes."""
    global _serve_proc
    os.makedirs(os.path.join(home, ".hipfire"), exist_ok=True)
    models = os.path.expanduser(os.environ.get("HIPFIRE_MODELS_DIR", "~/.hipfire/models"))
    for ln in ("models", "templates"):
        dst = os.path.join(home, ".hipfire", ln)
        try:
            if os.path.lexists(dst): os.remove(dst)
            os.symlink(os.path.join(models, "..", ln) if ln == "templates" else models, dst)
        except OSError:
            pass
    _write_native_config(cfg, home)
    # Honor a caller-provided per-GPU daemon binary (a renamed copy → distinct
    # process comm → the CLI's reapOrphans `pkill -x <name>` stays scoped to THIS
    # instance). HIPFIRE_DAEMON_NAME/ID pass through from os.environ untouched.
    env = dict(os.environ, HOME=home, HIP_VISIBLE_DEVICES=os.environ.get("HIP_VISIBLE_DEVICES","0"),
               HIPFIRE_DAEMON_BIN=os.environ.get("HIPFIRE_DAEMON_BIN", os.path.join(REPO, "target/release/examples/daemon")),
               HIPFIRE_KV_MODE=cfg["kv"], HIPFIRE_CASK_OFF="1", HIPFIRE_MODEL=cfg["model"])
    if cfg["mtp"] == "on":
        env.update(HIPFIRE_QWEN_MTP="1", HIPFIRE_MTP_SAMPLED="1", HIPFIRE_MTP_PREFIX_CACHE="1")
    cli = _native_cli()
    atexit.register(_kill_serve)
    for attempt in range(1, 5):
        _kill_serve(); time.sleep(3)
        open(log, "w").close()
        _serve_proc = subprocess.Popen(
            [cli, "serve", "127.0.0.1", str(cfg["port"])],
            cwd=REPO, env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT,
            start_new_session=True)   # own process group so _kill_serve's group-kill is exact + scoped
        for _ in range(90):
            txt = open(log).read() if os.path.exists(log) else ""
            if _native_service_warm(cfg["port"]):
                return True
            if re.search(r"out of memory|error loading|panic", txt, re.I):
                break
            time.sleep(2)
        print(f"  [serve spawn attempt {attempt} failed]", file=sys.stderr)
    return False


# ---------- request + capture ----------
def uniq(toks): return len(set(toks)) / len(toks) if toks else 1.0
def maxfreq(toks):
    if not toks: return 0.0
    from collections import Counter
    return Counter(toks).most_common(1)[0][1] / len(toks)
def gram3(toks):
    if len(toks) < 6: return 0.0
    g = [tuple(toks[i:i+3]) for i in range(len(toks)-2)]
    from collections import Counter
    c = Counter(g); return sum(v for v in c.values() if v > 1) / len(g)

def send(cfg, messages):
    body = {"model": cfg["model"], "messages": messages, "max_tokens": cfg["max_tokens"],
            "stream": True, "stream_options": {"include_usage": True}}
    body.update(cfg["sampling"])
    if cfg.get("seed") is not None:
        body["seed"] = cfg["seed"]
    t0 = time.time(); ttft = None; think = []; ans = []; tools = []
    usage = {}; timings = {}; finish = None
    req = urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    for raw in urllib.request.urlopen(req, timeout=1800):
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"): continue
        p = line[5:].strip()
        if p == "[DONE]": break
        try: ck = json.loads(p)
        except Exception: continue
        if ck.get("usage"): usage = ck["usage"]
        if ck.get("timings"): timings = ck["timings"]
        ch = (ck.get("choices") or [{}])[0]
        if ch.get("finish_reason"): finish = ch["finish_reason"]
        d = ch.get("delta") or {}
        if isinstance(d.get("reasoning_content"), str):
            if ttft is None and d["reasoning_content"]: ttft = time.time() - t0
            think.append(d["reasoning_content"])
        if isinstance(d.get("content"), str):
            if ttft is None and d["content"]: ttft = time.time() - t0
            ans.append(d["content"])
        if d.get("tool_calls"): tools.append(json.dumps(d["tool_calls"]))
    wall = time.time() - t0
    dtoks = usage.get("completion_tokens", 0)
    decode_ts = timings.get("decode_tok_s")
    decode_est = False
    if decode_ts is None and dtoks > 1 and ttft is not None and (wall - ttft) > 1e-6:
        decode_ts = round((dtoks - 1) / (wall - ttft), 1)
        decode_est = True
    think_s = "".join(think); ans_s = "".join(ans); tool_s = "".join(tools)
    visible = (ans_s + " " + tool_s).strip()
    toks = re.findall(r"\S+", (think_s + " " + visible).strip())
    first, last, half = toks[:128], toks[-128:], toks[len(toks)//2:]
    bad = (bool(first) and (uniq(first) < 0.15 or maxfreq(first) > 0.50)) or \
          (bool(last) and (uniq(last) < 0.30 or maxfreq(last) > 0.50)) or (gram3(half) > 0.50)
    return {
        "ctx": usage.get("prompt_tokens", 0),
        "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "gen": usage.get("completion_tokens", 0), "finish": finish,
        "think_words": len(re.findall(r"\S+", think_s)), "ans_words": len(re.findall(r"\S+", visible)),
        "prefill_ms": timings.get("prefill_ms"), "prefill_tok_s": timings.get("prefill_tok_s"),
        "decode_tok_s": decode_ts, "decode_estimated": decode_est, "tau": timings.get("tau"),
        "ttft_s": round(ttft or 0, 3), "wall_s": round(wall, 3),
        "attractor": bad, "empty": (cfg.get("expect_visible", True) and len(visible) == 0),
        "runaway": (finish == "length"),
        "ans_preview": (visible or "<<no visible content>>")[:90],
        "assistant_content": ans_s if ans_s else (tool_s if tool_s else think_s),
    }


def turn_line(i, r, recall=""):
    flags = []
    if r["runaway"]: flags.append("RUNAWAY")
    if r["empty"]:   flags.append("EMPTY")
    if r["attractor"]: flags.append("ATTRACTOR")
    fl = (" !" + ",".join(flags)) if flags else ""
    dec = r["decode_tok_s"]
    dec_str = f"{dec}~" if (dec is not None and r.get("decode_estimated")) else f"{dec}"
    return (f"  t{i:<2} finish={str(r['finish']):<6} ctx={r['ctx']:<6} cached={r['cached']:<6} "
            f"gen={r['gen']:<5}(think {r['think_words']}/ans {r['ans_words']}w) "
            f"prefill={r['prefill_ms']}ms decode={dec_str} tau={r['tau']}"
            f"{recall}{fl} | {r['ans_preview']!r}")


def run(cfg, args):
    label = f"{os.path.basename(cfg['model'])}|{cfg['mtp']}|{cfg['mode']}"
    print(f"### RUN {label}  kv={cfg['kv']} sampling={cfg['sampling']} seed={cfg.get('seed')} ###", flush=True)
    rows = []
    battery = load_prompt_battery(cfg.get("prompts_file"))
    if cfg["mode"] == "battery":
        for genre, prompt in battery:
            r = send(cfg, [{"role": "user", "content": prompt}])
            rows.append(r); print(f"  [{genre}]" + turn_line(len(rows), r)[2:], flush=True)
    elif cfg["mode"] == "chain":
        messages = []
        for genre, prompt in battery:
            messages.append({"role": "user", "content": prompt})
            r = send(cfg, messages)
            messages.append({"role": "assistant", "content": r["assistant_content"]})
            rows.append(r); print(f"  [{genre}]" + turn_line(len(rows), r)[2:], flush=True)
    elif cfg["mode"] == "session":
        turns = json.load(open(args.session))
        messages = []
        for i, t in enumerate(turns):
            messages.append({"role": "user", "content": t["content"]})
            r = send(cfg, messages)
            messages.append({"role": "assistant", "content": r["assistant_content"]})
            recall = ""
            if t.get("expect"):
                hit = sum(1 for e in t["expect"] if e.lower() in r["assistant_content"].lower())
                recall = f" recall={hit}/{len(t['expect'])}"
            rows.append(r); print(turn_line(i+1, r, recall), flush=True)
    g = rows
    dec = [r["decode_tok_s"] for r in g if isinstance(r["decode_tok_s"], (int, float))]
    pf = [(r["prefill_ms"] or 0)/1000 for r in g]
    print(f"[{label} DONE] turns={len(g)} runaway={sum(r['runaway'] for r in g)} "
          f"empty={sum(r['empty'] for r in g)} attractor={sum(r['attractor'] for r in g)} "
          f"avg_decode={sum(dec)/len(dec):.1f}tok/s" if dec else f"[{label} DONE] turns={len(g)}", flush=True)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model file path to serve")
    ap.add_argument("--tag", default=None, help="registry tag for recommended_settings (else inferred)")
    ap.add_argument("--registry", default=os.path.join(REPO, "registry/v1.json"))
    ap.add_argument("--kv", default="fwht3")
    ap.add_argument("--mtp", default="off", choices=["off", "on", "auto"])
    ap.add_argument("--dflash", default="off", choices=["off", "on", "auto"],
                    help="DFlash draft decode; pin the draft with HIPFIRE_DFLASH_DRAFT=<path>")
    ap.add_argument("--thinking", default="med", help="thinking_budget preset key")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-seq", type=int, default=32768)
    ap.add_argument("--sampling", default="registry",
                    help="registry | registry:general|coding|instruct | greedy | recipe:general|coding|nothink | json:{...}")
    ap.add_argument("--mode", default="battery", choices=["battery", "chain", "session"])
    ap.add_argument("--session", default="/home/kaden/mv/session_coding.json")
    ap.add_argument("--port", type=int, default=11520)
    ap.add_argument("--home", default=os.path.expanduser("~/.cache/serve_harness_home"))
    ap.add_argument("--serve-log", default="/tmp/serve_harness.serve.log")
    ap.add_argument("--out", default=None, help="write per-turn json")
    ap.add_argument("--show-config", action="store_true", help="resolve+print config, do NOT run")
    ap.add_argument("--no-spawn", action="store_true", help="connect to an already-running serve")
    ap.add_argument("--seed", type=int, default=None,
                    help="per-request sampler seed (sent in the body -> daemon initial rng_state). The "
                         "certify's coherence arm invokes with a seed-SET (one seed per call) for the rate test.")
    ap.add_argument("--prompts-file", default=None,
                    help="JSON [{\"genre\":..,\"prompt\":..}] replacing the built-in genre battery "
                         "(e.g. battery + the coherence_prompts_<arch> guard set).")
    args = ap.parse_args()
    cfg = build_config(args); cfg["max_seq"] = args.max_seq
    show_config(cfg)
    if args.show_config:
        return
    if cfg['thinking_cap_tokens'] and args.max_tokens <= cfg['thinking_cap_tokens']:
        sys.exit(
            f"serve_harness: max_tokens ({args.max_tokens}) <= thinking budget "
            f"'{cfg['thinking_budget']}' ({cfg['thinking_cap_tokens']} tok) guarantees "
            f"think-only output with zero visible answer. Raise --max-tokens above "
            f"{cfg['thinking_cap_tokens']}, lower --thinking (low={THINKING_BUDGET['low']}), "
            f"or use --thinking uncapped."
        )
    if not args.no_spawn:
        if not spawn_serve(cfg, args.home, args.serve_log):
            sys.exit("serve_harness: serve failed to warm after retries")
        head = subprocess.run(f"grep -c 'MTP head loaded' {args.serve_log}", shell=True,
                              capture_output=True, text=True).stdout.strip()
        print(f"  [serve warm; MTP head loaded lines={head}]", flush=True)
    run(cfg, args)
    if not args.no_spawn:
        _kill_serve()


if __name__ == "__main__":
    main()
