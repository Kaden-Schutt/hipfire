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
the resolved effort and independent thinking cap, `max_tokens`, `kv`, `mtp`, model — WITHOUT
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
import argparse, atexit, errno, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

# Mirror of the Rust configuration schema's reasoning budgets (resolved here so the pre-flight shows the
# concrete token cap, not just the preset name).
# Mirror of the Rust reasoning-budget presets. `off` resolves to a cap of 1 — the
# engine's "no thinking" sentinel (the daemon reads `enable_thinking:
# max_think_tokens != 1`), which is why it is not 0: 0 means `uncapped`.
THINKING_BUDGET = {"off": 1, "low": 512, "med": 2048, "high": 8192, "xhigh": 24576,
                   "max": 32768, "uncapped": 0}

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
        for k in ["temperature", "top_p", "top_k", "min_p", "presence_penalty",
                  "repeat_penalty", "reasoning_effort", "thinking_budget"]:
            if k in rec:
                vals[k] = rec[k]; source[k] = label
        # The instruct profile is the non-thinking mode: drive reasoning_effort=none
        # through the existing budget machinery (no daemon request-JSON change).
        if profile == "instruct" and "reasoning_effort" not in vals:
            vals["reasoning_effort"] = "none"; source["reasoning_effort"] = label
        # Registry stays the PREFERENCE; this only covers entries that carry nothing
        # usable. Guard on the sampling keys, not on `vals` — the instruct profile has
        # already inserted reasoning_effort above, so `not vals` would never fire here.
        if not [k for k in vals if k not in ("reasoning_effort", "thinking_budget")]:
            # 38/54 registry models still lack recommended_settings, and a hard exit
            # strands callers that cannot pass --sampling (tools.redline bench's
            # coherence smoke hard-codes "registry" and forwards no --tag).
            fb = "coding" if profile == "coding" else "nothink" if profile == "instruct" else "general"
            what = f"sampling_profiles.{profile}" if profile else "recommended_settings"
            print(f"  [warn] registry has no {what} for tag {tag!r} — falling back to "
                  f"recipe({fb}). Pass --tag <registry-tag> or --sampling explicitly to pin it.",
                  file=sys.stderr)
            for k, v in RECIPES[fb].items():
                vals[k] = v
                source[k] = f"registry-fallback:recipe({fb})"
        return vals, source
    sys.exit(f"bad --sampling {spec!r}")


def infer_tag(model_path):
    """Best-effort registry tag from a model filename, e.g. qwen3.6-27b-awq.mq4 -> qwen3.6:27b.

    Sizes may be decimal (qwen3.5-0.8b.mq4 -> qwen3.5:0.8b); the registry carries
    those tags, so the size group must admit a fractional part.
    """
    b = os.path.basename(model_path)
    if b.startswith("deepseek-v4-flash-0731"):
        if b.endswith(".mq2r"):
            return "deepseek-v4-flash:mq2r"
        if b.endswith(".mq2lloyd"):
            return "deepseek-v4-flash"
    m = re.match(r"(qwen3\.\d+)-(\d+(?:\.\d+)?b(?:-a\d+b)?)", b)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return None


def build_config(args):
    tag = args.tag or infer_tag(args.model)
    samp, samp_src = resolve_sampling(args.sampling, tag, args.registry)
    # Effort is parent-model prompt semantics. Budget is an independent hipfire
    # cap policy and must never be inferred from the effort level.
    registry_budget = samp.pop("thinking_budget", None)
    samp_src.pop("thinking_budget", None)
    effort = getattr(args, "thinking_effort", None)
    if effort:
        samp = dict(samp)
        samp["reasoning_effort"] = effort
        samp_src = dict(samp_src)
        samp_src["reasoning_effort"] = "explicit(--thinking-effort)"
    selected_budget = args.thinking
    if selected_budget is None:
        if registry_budget is not None:
            selected_budget = registry_budget
        elif samp.get("reasoning_effort") in ("low", "high", "max"):
            selected_budget = "uncapped"
        else:
            selected_budget = "med"
    think_cap = THINKING_BUDGET.get(selected_budget)
    if think_cap is None:
        sys.exit(f"thinking_budget {selected_budget!r} not a key of {list(THINKING_BUDGET)}")
    draft = getattr(args, "draft", None)
    if draft:
        draft = os.path.abspath(os.path.expanduser(draft))
    else:
        # Preserve caller-pinned draft env; do not invent a path.
        env_draft = os.environ.get("HIPFIRE_DFLASH_DRAFT")
        draft = os.path.abspath(os.path.expanduser(env_draft)) if env_draft else None
    return {
        "model": args.model, "tag": tag, "kv": args.kv, "mtp": args.mtp,
        "kv_backend": getattr(args, "kv_backend", "contiguous") or "contiguous",
        "dflash": getattr(args, "dflash", "off") or "off",
        "draft": draft,
        "thinking_budget": selected_budget, "thinking_cap_tokens": think_cap,
        "max_tokens": args.max_tokens, "sampling": samp, "sampling_source": samp_src,
        "mode": args.mode, "port": args.port, "seed": getattr(args, "seed", None),
        "prompts_file": getattr(args, "prompts_file", None),
        "prompt_file": getattr(args, "prompt_file", None),
        "niah_file": getattr(args, "niah_file", None),
        "speculation_mode": getattr(args, "speculation", None),
        "deepseek4_experts_per_token": getattr(args, "deepseek4_experts_per_token", None),
        "deepseek4_compute_placement": getattr(args, "deepseek4_compute_placement", "single"),
        "devices": getattr(args, "devices", None),
        "tp": getattr(args, "tp", None),
        "replay_route_proof_log": bool(getattr(args, "replay_route_proof_log", False)),
    }




def load_prompt_battery(prompts_file, prompt_file=None, niah_file=None):
    """Return ``(genre, prompt, expected_substrings)`` prompt rows.

    ``--niah-file`` consumes the repository's committed long-context fixture
    format directly.  Keeping the JSON fixture as the source of truth avoids a
    second flattened prompt whose whitespace could silently drift.
    """
    if prompt_file:
        text = Path(prompt_file).read_bytes().decode("utf-8")
        return [("prose", text, [])]
    if niah_file:
        raw = Path(niah_file).read_text(encoding="utf-8")
        stripped = raw.lstrip()
        records = json.loads(raw) if stripped.startswith("[") else [json.loads(line) for line in raw.splitlines() if line.strip()]
        rows = []
        for index, record in enumerate(records):
            filler = record.get("filler_text")
            question = record.get("question")
            if not isinstance(filler, str) or not isinstance(question, str):
                raise ValueError(f"NIAH row {index} requires string filler_text and question")
            expected = record.get("expected_answer_substrings")
            if expected is None:
                expected = [record.get("expected_answer_substring")]
            if not isinstance(expected, list) or not expected or not all(isinstance(item, str) and item for item in expected):
                raise ValueError(f"NIAH row {index} requires nonempty expected answer substring(s)")
            rows.append((record.get("genre", "longctx-niah"), f"{filler}\n\n{question}", expected))
        return rows
    if not prompts_file:
        return [(genre, prompt, []) for genre, prompt in GENRE_BATTERY]
    rows = json.load(open(prompts_file))
    return [(r.get("genre", "prose"), r["prompt"], r.get("expect", [])) for r in rows]


def show_config(cfg):
    print("==================== serve_harness pre-flight (CONFIRM before run) ====================")
    print(f"  model         : {cfg['model']}")
    print(f"  registry tag  : {cfg['tag'] or '(none — sampling cannot be registry-resolved)'}")
    print(f"  kv_mode       : {cfg['kv']}   kv_backend: {cfg.get('kv_backend', 'contiguous')}   mtp_mode: {cfg['mtp']}   mode: {cfg['mode']}")
    print(f"  dflash        : {cfg.get('dflash', 'off')}   draft: {cfg.get('draft') or '(none / filename auto-match)'}")
    _spec = cfg.get("speculation_mode")
    print(f"  speculation   : {_spec or '(derived from --dflash/--mtp above)'}"
          f"{'   <-- OVERRIDES dflash/mtp' if _spec else ''}")
    print(
        "  ds4 experts/tok: "
        f"{cfg.get('deepseek4_experts_per_token') or '(checkpoint default)'}"
    )
    print(f"  ds4 placement : {cfg.get('deepseek4_compute_placement', 'single')}")
    print(f"  devices       : {cfg.get('devices') or '(runtime default)'}")
    print(f"  expert parallel: tp={cfg.get('tp') or 1}")
    prompt_source = cfg.get("prompt_file") or cfg.get("prompts_file") or cfg.get("niah_file") or "(built-in battery)"
    print(f"  seed          : {cfg.get('seed')}   prompt_source: {prompt_source}")
    _cap = cfg['thinking_cap_tokens']
    _thinking_off = cfg['thinking_budget'] == 'off'
    _resolved = ('thinking DISABLED (sentinel cap 1)' if _thinking_off
                 else 'uncapped' if _cap == 0
                 else f'{_cap} tok (CONCRETE cap)')
    print(f"  thinking_budget: {cfg['thinking_budget']} -> {_resolved}")
    print(f"  reasoning_effort: {cfg['sampling'].get('reasoning_effort', 'auto')}"
          "  (parent prompt semantics; independent of cap)")
    _note = ('no think block emitted' if _thinking_off
             else 'uncapped think budget' if _cap == 0
             else f'> think cap {_cap} — model can answer' if cfg['max_tokens'] > _cap
             else f'<= think cap {_cap} — INVALID (think-only); run will hard-fail')
    print(f"  max_tokens     : {cfg['max_tokens']}  ({_note})")
    print("  sampling (what IS set):")
    for k in SAMPLE_KEYS:
        if k in cfg["sampling"]:
            print(f"      {k:18}= {cfg['sampling'][k]:<8} [{cfg['sampling_source'].get(k,'?')}]")
    notset = [k for k in SAMPLE_KEYS if k not in cfg["sampling"]]
    print(f"  sampling (NOT set, serve/daemon default applies): {', '.join(notset) or '(none)'}")
    # Surface inherited DFlash/DDTree env knobs the harness must not clobber.
    for env_key in (
        "HIPFIRE_DFLASH_DRAFT",
        "HIPFIRE_DFLASH_TREE",
        "HIPFIRE_DFLASH_FAST_SAMPLE",
        "HIPFIRE_DDTREE_BUDGET",
        "HIPFIRE_DDTREE_TOPK",
    ):
        if env_key in os.environ:
            print(f"  env {env_key}={os.environ[env_key]!r} (pass-through)")
    print("=======================================================================================")




# ---------- serve spawn (robust, self-contained) ----------
# Popen(start_new_session=True) makes the CLI leader PID also the session PGID.
# Retain that known PGID so cleanup can killpg even after the leader exits
# (os.getpgid(leader) then returns ESRCH while descendants may still live).
_serve_proc = None
_serve_pgid = None


def _pid_file_path():
    """Optional cross-process PID path for the active CLI process-group leader."""
    path = os.environ.get("HIPFIRE_SERVE_HARNESS_PID_FILE")
    return path if path else None


def _clear_pid_file():
    path = _pid_file_path()
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _write_pid_file(pid):
    """Atomically publish the CLI process-group leader PID for a parent observer."""
    path = _pid_file_path()
    if not path:
        return
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".serve_harness_pid.", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(f"{int(pid)}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _kill_serve():
    """Kill ONLY this harness's own native serve tree (the Rust CLI + its child daemon),
    scoped by process group — NOT a broad `pkill -x daemon`, which would execute
    the parallel autoresearch daemons pinned to OTHER GPUs. spawn_serve starts the
    serve in its own session (start_new_session) so this group kill is exact.

    Always killpg the retained session PGID (equal to the CLI Popen PID). Do not
    gate on os.getpgid(leader): after the leader exits getpgid returns ESRCH even
    when the process group still has descendants. ESRCH from killpg is benign."""
    global _serve_proc, _serve_pgid
    pgid = _serve_pgid
    if pgid is None and _serve_proc is not None:
        pgid = _serve_proc.pid
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as err:
            # ESRCH: group already gone (benign race). Other errors: last-ditch
            # kill the Popen handle if it is still around.
            if getattr(err, "errno", None) != errno.ESRCH and _serve_proc is not None:
                try:
                    _serve_proc.kill()
                except Exception:
                    pass
    _serve_proc = None
    _serve_pgid = None
    _clear_pid_file()


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
    """Write the isolated harness configuration in the native sparse TOML format.

    product_bench isolates HIPFIRE_HOME so the daemon never inherits
    ~/.hipfire/config.toml. Opt-in diagnostics such as route_proof_log are
    requested here as temporary TOML rather than ad-hoc env exports.

    DFlash selection uses the canonical speculation selector:
      --dflash on  → mode=dflash, dflash=on, mtp=off, ngram=off
      --dflash auto → mode=auto, dflash=auto (mtp left as requested)
      --dflash off + --mtp off → mode=off (plain AR; no sidecar auto-discovery)
      --dflash off + other MTP setting → dflash=off (mode remains auto)

    `--speculation MODE` overrides all of the above with an explicit selector,
    mirroring the CLI's apply_speculation_selector. This is the only way to ask
    for DSpark: the DFlash/MTP matrix above can reach it solely by accident, by
    leaving `mode` at its schema default of `auto` so the sidecar is
    auto-discovered. DeepSeek V4 ships its speculative module inside the
    checkpoint (see the model card: "it comes with a speculative decoding module
    attached"), so `--speculation dspark` is the supported way to exercise it.
    """
    explicit = cfg.get("speculation_mode")
    dflash = cfg.get("dflash", "off") or "off"
    mtp = cfg["mtp"]
    if explicit:
        # Mirrors apply_speculation_selector(): each named selector pins every
        # sibling off so the arms are mutually exclusive and legible in the log.
        pins = {
            "off":    ('off',    'off',  'off', 'off'),
            "dflash": ('dflash', 'on',   'off', 'off'),
            "mtp":    ('mtp',    'off',  'on',  'off'),
            "ngram":  ('ngram',  'off',  'off', 'on'),
            "dspark": ('dspark', 'off',  'off', 'off'),
            "auto":   ('auto',   'auto', mtp,   'off'),
        }[explicit]
        speculation = (
            '[speculation]\n'
            f'mode = {json.dumps(pins[0])}\n'
            f'dflash = {json.dumps(pins[1])}\n'
            f'mtp = {json.dumps(pins[2])}\n'
            f'ngram = {json.dumps(pins[3])}\n'
        )
    elif dflash == "on":
        # Exclusive DFlash selector — mirrors apply_speculation_selector("dflash").
        speculation = (
            '[speculation]\n'
            'mode = "dflash"\n'
            'dflash = "on"\n'
            'mtp = "off"\n'
            'ngram = "off"\n'
        )
    elif dflash == "auto":
        speculation = (
            '[speculation]\n'
            'mode = "auto"\n'
            f'dflash = "auto"\n'
            f'mtp = {json.dumps(mtp)}\n'
            'ngram = "off"\n'
        )
    else:
        # `--dflash off --mtp off` is the ordinary-AR contract. Leaving mode
        # at its schema default (`auto`) would still auto-discover DSpark.
        mode = 'mode = "off"\n' if mtp == "off" else ""
        speculation = (
            '[speculation]\n'
            f'{mode}'
            f'dflash = "off"\n'
            f'mtp = {json.dumps(mtp)}\n'
            'ngram = "off"\n'
        )
    model = ""
    if cfg.get("deepseek4_experts_per_token") is not None:
        model = (
            "[model]\n"
            f"deepseek4_experts_per_token = {cfg['deepseek4_experts_per_token']}\n\n"
        )
    placement = cfg.get("deepseek4_compute_placement", "single")
    devices = cfg.get("devices")
    devices_line = f"devices = {json.dumps(devices)}\n" if devices else ""
    hardware = f"""[hardware]
{devices_line}deepseek4_compute_placement = {json.dumps(placement)}

"""
    text = f"""[serve]
host = "127.0.0.1"
port = {cfg["port"]}
default_model = {json.dumps(cfg["model"])}

{model}{hardware}[memory]
max_seq = {cfg.get("max_seq", 32768)}
kv_cache = {json.dumps(cfg["kv"])}

{speculation}
[generation]
max_tokens = {cfg.get("max_tokens", 16384)}

[reasoning]
budget = {json.dumps(cfg["thinking_budget"])}
"""
    effort = cfg.get("sampling", {}).get("reasoning_effort")
    if effort:
        text += f"effort = {json.dumps(effort)}\n"
    if cfg.get("replay_route_proof_log"):
        text += """
[diagnostic.replay]
route_proof_log = true
"""
    with open(os.path.join(home, ".hipfire", "config.toml"), "w") as handle:
        handle.write(text)


def _serve_log_offset(log):
    """Byte size of *log* (0 if missing) — capture immediately before a spawn attempt."""
    if not os.path.exists(log):
        return 0
    return os.path.getsize(log)


def _serve_log_text(log, offset=0):
    """Return log bytes from *offset* onward (current-attempt slice)."""
    if not os.path.exists(log):
        return ""
    with open(log, encoding="utf-8", errors="replace") as handle:
        handle.seek(max(0, int(offset or 0)))
        return handle.read()


def _startup_path_proof_failures(cfg, txt):
    """VMM + DFlash draft-load failures for one attempt's log slice (no sys.exit)."""
    failures = []

    if cfg.get("kv_backend") == "vmm":
        # KvCache constructors emit e.g. "KV cache: q8 vmm (...", "KV cache: fwht3 vmm (...".
        if not re.search(r"KV cache:.*\bvmm\b", txt):
            failures.append(
                "kv_backend=vmm requested but serve log has no 'KV cache: … vmm' marker"
            )

    dflash = cfg.get("dflash", "off") or "off"
    if dflash == "on":
        loaded = (
            "DFlash draft loaded:" in txt
            or "DFlash generic speculator loaded" in txt
        )
        skipped = "dflash_mode=off — skipping draft load" in txt
        failed = "DFlash draft load failed" in txt
        disabled = "DFlash disabled (dflash_mode=off)" in txt
        if skipped or disabled:
            failures.append(
                "dflash=on requested but serve log shows DFlash disabled/skipped"
            )
        elif failed:
            failures.append(
                "dflash=on requested but serve log shows 'DFlash draft load failed'"
            )
        elif not loaded:
            failures.append(
                "dflash=on requested but serve log lacks "
                "'DFlash draft loaded:' / 'DFlash generic speculator loaded' proof"
            )
    return failures


def _row_has_dflash_execution(row):
    """True only for an explicit request-level DFlash route identity."""
    return isinstance(row, dict) and row.get("dflash") is True


# Log-side request-level DFlash route identities. Generic tau/cycle metrics are
# intentionally excluded: both AR fallback and MTP can emit them.
_DFLASH_EXEC_LOG_RE = re.compile(
    r'("dflash"\s*:\s*true)|(?:^|\s)drafter=dflash(?:\s|$)',
    re.I | re.M,
)


def _log_has_dflash_execution(txt):
    return bool(txt and _DFLASH_EXEC_LOG_RE.search(txt))


def _dflash_request_proof_failures(cfg, rows, log_txt=""):
    """After requests: dflash=on requires an explicit DFlash route identity."""
    dflash = cfg.get("dflash", "off") or "off"
    if dflash != "on":
        return []
    rows = rows or []
    if any(_row_has_dflash_execution(r) for r in rows):
        return []
    if _log_has_dflash_execution(log_txt):
        return []
    return [
        "dflash=on requested but no request-level DFlash execution evidence "
        "(need timings.dflash=true or drafter=dflash); draft-load alone is not sufficient"
    ]


def _emit_path_proof_failure(failures, log, when):
    for msg in failures:
        print(f"  [serve path proof FAILED] {msg}", file=sys.stderr)
    sys.exit(
        f"serve_harness: production serve path proof failed {when} "
        f"({'; '.join(failures)}). See {log}"
    )


def _assert_serve_path_proofs(cfg, log, offset=0):
    """Fail closed when an explicit VMM/DFlash path was requested but the current
    attempt's warm log slice lacks startup engagement markers (PR #549)."""
    txt = _serve_log_text(log, offset)
    failures = _startup_path_proof_failures(cfg, txt)
    if failures:
        _emit_path_proof_failure(failures, log, "after warm")


def _assert_dflash_request_proofs(cfg, rows, log, offset=0):
    """Fail closed when dflash=on but no request exercised speculative decode."""
    txt = _serve_log_text(log, offset)
    failures = _dflash_request_proof_failures(cfg, rows, txt)
    if failures:
        _emit_path_proof_failure(failures, log, "after requests")


def _self_test_serve_path_proofs():
    """Deterministic coverage for current-attempt log slicing + DFlash request proof.

    Run: ``python3 scripts/serve_harness.py --self-test``
    or ``HIPFIRE_SERVE_HARNESS_SELFTEST=1``.
    """
    import tempfile

    def check(cond, msg):
        if not cond:
            raise AssertionError(msg)

    with tempfile.NamedTemporaryFile("w+b", delete=False) as tmp:
        path = tmp.name
        # Prior attempt / prior run markers (must NOT satisfy current proof).
        stale = (
            b"KV cache: q8 vmm (stale prior attempt)\n"
            b"DFlash draft loaded: /stale/draft.hfq\n"
            b'{"type":"done","dflash":true,"tau":9.5,"cycles":12}\n'
        )
        tmp.write(stale)
        tmp.flush()
        offset = tmp.tell()

        # --- stale prior markers alone must fail VMM + draft-load ---
        cfg_vmm = {"kv_backend": "vmm", "dflash": "off"}
        stale_txt = _serve_log_text(path, offset)  # empty suffix
        check(stale_txt == "", "suffix after offset must be empty before current write")
        fails = _startup_path_proof_failures(cfg_vmm, stale_txt)
        check(any("vmm" in f for f in fails), f"stale-only VMM must fail, got {fails!r}")

        # Full-file read would false-pass — document the bug we closed.
        full_false_pass = _startup_path_proof_failures(cfg_vmm, _serve_log_text(path, 0))
        check(not full_false_pass, "precondition: full log still contains stale VMM marker")

        # --- current-attempt VMM marker after offset passes ---
        with open(path, "ab") as ap:
            ap.write(b"KV cache: fwht3 vmm (current attempt)\n")
        cur = _serve_log_text(path, offset)
        fails = _startup_path_proof_failures(cfg_vmm, cur)
        check(not fails, f"current-attempt VMM must pass, got {fails!r}")

        # --- draft-loaded without request execution must fail ---
        cfg_df = {"kv_backend": "contiguous", "dflash": "on"}
        with open(path, "ab") as ap:
            ap.write(b"DFlash draft loaded: /current/draft.hfq\n")
        cur = _serve_log_text(path, offset)
        load_fails = _startup_path_proof_failures(cfg_df, cur)
        check(not load_fails, f"current draft-load startup must pass, got {load_fails!r}")
        req_fails = _dflash_request_proof_failures(cfg_df, rows=[], log_txt=cur)
        check(req_fails, f"draft-load without execution must fail, got {req_fails!r}")
        # Rows without tau/cycles/dflash also fail even if draft loaded.
        ar_rows = [{"tau": None, "cycles": None, "dflash": None, "gen": 8}]
        req_fails = _dflash_request_proof_failures(cfg_df, rows=ar_rows, log_txt=cur)
        check(req_fails, f"all-AR rows must fail DFlash request proof, got {req_fails!r}")
        # The daemon's explicit AR fallback summary includes tau=1.00. Generic
        # tau/cycle metrics must not certify DFlash execution.
        ar_log = cur + "\n[req req-1] drafter=ar tau=1.00 tok/s=88.0 (autoregressive)\n"
        req_fails = _dflash_request_proof_failures(cfg_df, rows=ar_rows, log_txt=ar_log)
        check(req_fails, f"AR fallback tau log must fail DFlash proof, got {req_fails!r}")

        # MTP also reports tau/cycles, but is not DFlash.
        mtp_rows = [{"tau": 3.0, "cycles": 4, "dflash": None, "mtp": True, "gen": 12}]
        req_fails = _dflash_request_proof_failures(cfg_df, rows=mtp_rows, log_txt=cur)
        check(req_fails, f"MTP timings must fail DFlash request proof, got {req_fails!r}")

        # --- request-level DFlash success via row tau ---
        ok_rows = [{"tau": 4.2, "cycles": 3, "dflash": True, "gen": 16}]
        req_fails = _dflash_request_proof_failures(cfg_df, rows=ok_rows, log_txt=cur)
        check(not req_fails, f"tau/cycles row must pass, got {req_fails!r}")

        # --- request-level success via log discriminator alone ---
        with open(path, "ab") as ap:
            ap.write(b'{"type":"done","dflash":true,"tau":3.25,"cycles":4}\n')
        cur = _serve_log_text(path, offset)
        req_fails = _dflash_request_proof_failures(cfg_df, rows=ar_rows, log_txt=cur)
        check(not req_fails, f"log dflash/tau evidence must pass, got {req_fails!r}")

        # Non-DFlash / non-VMM configs stay silent.
        plain = {"kv_backend": "contiguous", "dflash": "off"}
        check(not _startup_path_proof_failures(plain, ""), "plain startup must be no-op")
        check(not _dflash_request_proof_failures(plain, [], ""), "plain request proof must be no-op")

    os.unlink(path)
    print("serve_harness: path-proof self-test OK", flush=True)


def _self_test_prompt_sources():
    """Exercise NIAH lowering without touching a model or GPU."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        json.dump({
            "filler_text": "alpha needle omega",
            "question": "What was in the middle?",
            "expected_answer_substring": "needle",
        }, tmp)
        path = tmp.name
    try:
        rows = load_prompt_battery(None, niah_file=path)
        assert rows == [("longctx-niah", "alpha needle omega\n\nWhat was in the middle?", ["needle"])]
    finally:
        os.unlink(path)
    print("serve_harness: prompt-source self-test OK", flush=True)


def _self_test_device_config():
    """Prove multi-device visibility is explicit in the isolated TOML."""
    with tempfile.TemporaryDirectory() as home:
        Path(home, ".hipfire").mkdir()
        cfg = {
            "port": 11520,
            "model": "/models/deepseek4.mq2r",
            "kv": "q8",
            "mtp": "off",
            "dflash": "off",
            "thinking_budget": "off",
            "deepseek4_compute_placement":
                "dense-expert-split(dense=arch:gfx1100,experts=arch:gfx1151)",
            "devices": "0,1",
        }
        _write_native_config(cfg, home)
        config = Path(home, ".hipfire", "config.toml").read_text(encoding="utf-8")
        assert '[hardware]\ndevices = "0,1"\n' in config
    print("serve_harness: device-config self-test OK", flush=True)



def _native_service_warm(port, expected_model=None, proc=None):
    """True only when health is ready for *this* spawn.

    Rejects an unrelated warm service after the newly spawned leader has already
    exited: require ``proc`` still alive when provided, and when
    ``expected_model`` is set require health reports that same model path.
    """
    if proc is not None and proc.poll() is not None:
        return False
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
            health = json.load(response)
    except Exception:
        return False
    if not bool(health.get("model")) or health.get("loading_model"):
        return False
    if expected_model is not None:
        reported = health.get("model")
        if not isinstance(reported, str):
            return False
        # Compare resolved paths so relative vs absolute forms still match.
        try:
            if os.path.realpath(reported) != os.path.realpath(expected_model):
                return False
        except OSError:
            if reported != expected_model:
                return False
    if proc is not None and proc.poll() is not None:
        return False
    return True


def spawn_serve(cfg, home, log):
    """Spawn native `hipfire serve` with the resolved serve config; retry on the flaky
    daemon-spawn; return the successful attempt's pre-launch log byte offset, or None.

    The per-request sampling is sent by the driver, so one serve handles all recipes/modes.
    Proofs must inspect only ``_serve_log_text(log, offset)`` for that attempt."""
    global _serve_proc, _serve_pgid
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
    # Explicit --draft pins HIPFIRE_DFLASH_DRAFT. When absent, preserve any
    # caller-inherited value (do not pop/clear) so parent gates can pin the draft.
    if cfg.get("draft"):
        env["HIPFIRE_DFLASH_DRAFT"] = cfg["draft"]
    # Plain DFlash / DDTree knobs (TREE/BUDGET/TOPK/FAST_SAMPLE) pass through
    # from the parent environment unchanged — harness never rewrites them.
    # Harness-only parent IPC: must never reach hipfire serve / process config
    # (would otherwise lower as developer.serve_harness_pid_file).
    env.pop("HIPFIRE_SERVE_HARNESS_PID_FILE", None)
    cli = _native_cli()
    serve_cmd = [cli, "serve", "127.0.0.1", str(cfg["port"]),
                 "--kv-backend", cfg.get("kv_backend", "contiguous")]
    if cfg.get("tp"):
        serve_cmd.extend(["--tp", str(cfg["tp"])])
    atexit.register(_kill_serve)
    # Append-only log: prior attempts remain for debugging; proofs use per-attempt offsets.
    os.makedirs(os.path.dirname(os.path.abspath(log)) or ".", exist_ok=True)
    open(log, "a").close()
    for attempt in range(1, 5):
        _kill_serve(); time.sleep(3)
        # Drop any stale observer PID before the next CLI process-group leader exists.
        _clear_pid_file()
        # Capture offset immediately before launch — only this attempt's suffix is proof.
        log_offset = _serve_log_offset(log)
        _serve_proc = subprocess.Popen(
            serve_cmd,
            cwd=REPO, env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT,
            start_new_session=True)   # own process group so _kill_serve's group-kill is exact + scoped
        # Leader PID == PGID under start_new_session; retain it for dead-leader cleanup.
        _serve_pgid = _serve_proc.pid
        _write_pid_file(_serve_pgid)
        warm_timeout_secs = max(1, int(cfg.get("serve_warm_timeout_secs", 180)))
        for _ in range((warm_timeout_secs + 1) // 2):
            txt = _serve_log_text(log, log_offset)
            if _native_service_warm(cfg["port"], expected_model=cfg.get("model"), proc=_serve_proc):
                return log_offset
            # A CLI that has already exited cannot become warm. Waiting the
            # full 180-second startup window hid immediate config-validation
            # failures behind four long retries.
            if _serve_proc.poll() is not None:
                break
            if re.search(r"out of memory|error loading|panic", txt, re.I):
                break
            time.sleep(2)
        print(f"  [serve spawn attempt {attempt} failed]", file=sys.stderr)
    return None




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
    usage = {}; timings = {}; finish = None; completion_id = None
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
        if isinstance(ck.get("id"), str):
            completion_id = ck["id"]
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
        "request_id": completion_id,
        "ctx": usage.get("prompt_tokens", 0),
        "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "gen": usage.get("completion_tokens", 0), "finish": finish,
        "think_words": len(re.findall(r"\S+", think_s)), "ans_words": len(re.findall(r"\S+", visible)),
        "prefill_ms": timings.get("prefill_ms"), "prefill_tok_s": timings.get("prefill_tok_s"),
        "decode_tok_s": decode_ts, "decode_estimated": decode_est, "tau": timings.get("tau"),
        "cycles": timings.get("cycles"), "dflash": timings.get("dflash"), "mtp": timings.get("mtp"),
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
    prefill_tok_s = r.get("prefill_tok_s")
    prefill_str = f"{prefill_tok_s}" if prefill_tok_s is not None else "n/a"
    return (f"  t{i:<2} finish={str(r['finish']):<6} ctx={r['ctx']:<6} cached={r['cached']:<6} "
            f"gen={r['gen']:<5}(think {r['think_words']}/ans {r['ans_words']}w) "
            f"prefill={r['prefill_ms']}ms/{prefill_str}tok/s "
            f"decode={dec_str}tok/s tau={r['tau']}"
            f"{recall}{fl} | {r['ans_preview']!r}")


def run(cfg, args):
    label = f"{os.path.basename(cfg['model'])}|{cfg['mtp']}|{cfg['mode']}"
    print(f"### RUN {label}  kv={cfg['kv']} sampling={cfg['sampling']} seed={cfg.get('seed')} ###", flush=True)
    rows = []
    battery = load_prompt_battery(
        cfg.get("prompts_file"), cfg.get("prompt_file"), cfg.get("niah_file")
    )
    if cfg["mode"] == "battery":
        for genre, prompt, expected in battery:
            r = send(cfg, [{"role": "user", "content": prompt}])
            r["prompt_md5"] = hashlib.md5(prompt.encode("utf-8")).hexdigest()
            missing = [item for item in expected if item.lower() not in r["assistant_content"].lower()]
            r["expected_substrings"] = expected
            r["retrieval_missing"] = missing
            recall = f" recall={len(expected) - len(missing)}/{len(expected)}" if expected else ""
            rows.append(r); print(f"  [{genre}]" + turn_line(len(rows), r, recall)[2:], flush=True)
    elif cfg["mode"] == "chain":
        messages = []
        for genre, prompt, expected in battery:
            messages.append({"role": "user", "content": prompt})
            r = send(cfg, messages)
            r["prompt_md5"] = hashlib.md5(prompt.encode("utf-8")).hexdigest()
            messages.append({"role": "assistant", "content": r["assistant_content"]})
            missing = [item for item in expected if item.lower() not in r["assistant_content"].lower()]
            r["expected_substrings"] = expected
            r["retrieval_missing"] = missing
            recall = f" recall={len(expected) - len(missing)}/{len(expected)}" if expected else ""
            rows.append(r); print(f"  [{genre}]" + turn_line(len(rows), r, recall)[2:], flush=True)
    elif cfg["mode"] == "session":
        turns = json.load(open(args.session))
        messages = []
        for i, t in enumerate(turns):
            messages.append({"role": "user", "content": t["content"]})
            r = send(cfg, messages)
            messages.append({"role": "assistant", "content": r["assistant_content"]})
            recall = ""
            expected = t.get("expect", [])
            missing = [
                item
                for item in expected
                if item.lower() not in r["assistant_content"].lower()
            ]
            r["expected_substrings"] = expected
            r["retrieval_missing"] = missing
            if expected:
                recall = f" recall={len(expected) - len(missing)}/{len(expected)}"
            rows.append(r); print(turn_line(i+1, r, recall), flush=True)
    g = rows
    dec = [r["decode_tok_s"] for r in g if isinstance(r["decode_tok_s"], (int, float))]
    prefill = [
        r["prefill_tok_s"]
        for r in g
        if isinstance(r.get("prefill_tok_s"), (int, float))
    ]
    summary = (
        f"[{label} DONE] turns={len(g)} runaway={sum(r['runaway'] for r in g)} "
        f"empty={sum(r['empty'] for r in g)} attractor={sum(r['attractor'] for r in g)} "
        f"retrieval_miss={sum(bool(r.get('retrieval_missing')) for r in g)}"
    )
    if prefill:
        summary += f" avg_prefill={sum(prefill)/len(prefill):.1f}tok/s"
    if dec:
        summary += f" avg_decode={sum(dec)/len(dec):.1f}tok/s"
    print(summary, flush=True)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=0)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model file path to serve")
    ap.add_argument(
        "--deepseek4-compute-placement",
        default="single",
        help="typed DS4 placement, for example dense-expert-split(dense=arch:gfx1100,experts=arch:gfx1151)",
    )
    ap.add_argument(
        "--devices",
        default=None,
        help="physical GPU selectors written to [hardware].devices, for example 0,1",
    )
    ap.add_argument(
        "--tp",
        type=int,
        choices=range(1, 65),
        default=None,
        help="expert-parallel degree forwarded to native hipfire serve",
    )
    ap.add_argument("--tag", default=None, help="registry tag for recommended_settings (else inferred)")
    ap.add_argument("--registry", default=os.path.join(REPO, "registry/v1.json"))
    ap.add_argument("--kv", default="fwht3")
    ap.add_argument("--kv-backend", default="contiguous", choices=["contiguous", "vmm"],
                    help="hipfire serve --kv-backend override (default contiguous; use vmm for PR #549 path)")
    ap.add_argument("--mtp", default="off", choices=["off", "on", "auto"])
    ap.add_argument("--dflash", default="off", choices=["off", "auto", "on"],
                    help="DFlash mode written to temporary [speculation] TOML (default off). "
                         "'on' emits mode=dflash + dflash=on + mtp/ngram off.")
    ap.add_argument("--speculation", default=None,
                    choices=["off", "auto", "ngram", "dflash", "mtp", "dspark"],
                    help="explicit speculation selector; overrides --dflash/--mtp and mirrors the "
                         "CLI's apply_speculation_selector. Required to reach DSpark: the "
                         "--dflash/--mtp matrix can only get there by accident, via the schema "
                         "default mode=auto auto-discovering the sidecar. DeepSeek V4 ships its "
                         "speculative module in the checkpoint, so use --speculation dspark.")
    ap.add_argument(
        "--deepseek4-experts-per-token",
        type=int,
        choices=range(1, 7),
        default=None,
        help="DeepSeek V4 routed experts per token for this model load; omitted preserves the checkpoint default.",
    )
    ap.add_argument("--thinking-effort", default=None,
                    choices=["none", "low", "high", "max"],
                    help="parent-model reasoning_effort prompt semantics; independent of "
                         "--thinking. With no explicit/registry budget, low/high/max is uncapped.")
    ap.add_argument("--draft", default=None,
                    help="Optional DFlash draft path; sets HIPFIRE_DFLASH_DRAFT for the serve child. "
                         "When omitted, any caller-inherited HIPFIRE_DFLASH_DRAFT is preserved.")
    ap.add_argument("--thinking", default=None, choices=list(THINKING_BUDGET),
                    help="explicit reasoning cap policy. Default: registry thinking_budget; "
                         "otherwise uncapped for an explicit effort, med for legacy callers. "
                         "\"off\" disables thinking (cap sentinel 1).")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-seq", type=int, default=32768)
    ap.add_argument("--sampling", default="registry",
                    help="registry | registry:general|coding|instruct | greedy | recipe:general|coding|nothink | json:{...}")
    ap.add_argument("--mode", default="battery", choices=["battery", "chain", "session"])
    ap.add_argument("--session", default="/home/kaden/mv/session_coding.json")
    ap.add_argument("--port", type=int, default=11520)
    ap.add_argument("--home", default=os.path.expanduser("~/.cache/serve_harness_home"))
    ap.add_argument("--serve-log", default="/tmp/serve_harness.serve.log")
    ap.add_argument(
        "--serve-warm-timeout-secs",
        type=int,
        default=180,
        help="seconds to wait for a spawned serve to finish loading (default 180)",
    )
    ap.add_argument("--out", default=None, help="write per-turn json")
    ap.add_argument("--show-config", action="store_true", help="resolve+print config, do NOT run")
    ap.add_argument("--no-spawn", action="store_true", help="connect to an already-running serve")
    ap.add_argument("--seed", type=int, default=None,
                    help="per-request sampler seed (sent in the body -> daemon initial rng_state). The "
                         "certify's coherence arm invokes with a seed-SET (one seed per call) for the rate test.")
    prompt_source = ap.add_mutually_exclusive_group()
    prompt_source.add_argument("--prompts-file", default=None,
                               help="JSON [{\"genre\":..,\"prompt\":..}] replacing the built-in genre battery "
                                    "(e.g. battery + the coherence_prompts_<arch> guard set).")
    prompt_source.add_argument(
        "--prompt-file",
        default=None,
        help="UTF-8 prompt bytes lowered to one prose battery row without newline normalization.",
    )
    prompt_source.add_argument(
        "--niah-file",
        default=None,
        help="Committed NIAH JSON/JSONL fixture. Lowers filler_text + question exactly and "
             "fails when expected_answer_substring(s) are absent.",
    )
    ap.add_argument(
        "--replay-route-proof-log",
        action="store_true",
        help="Write diagnostic.replay.route_proof_log=true into the temporary "
             "$HIPFIRE_HOME/config.toml so the daemon emits one retained-replay "
             "proof marker per successful serve request (coherence/product gates).",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic serve path-proof self-tests (no GPU / no serve) and exit.",
    )
    args = ap.parse_args()
    if args.self_test or os.environ.get("HIPFIRE_SERVE_HARNESS_SELFTEST") == "1":
        _self_test_serve_path_proofs()
        _self_test_prompt_sources()
        _self_test_device_config()
        return
    if not args.model:
        ap.error("--model is required unless --self-test")
    cfg = build_config(args); cfg["max_seq"] = args.max_seq
    cfg["serve_warm_timeout_secs"] = args.serve_warm_timeout_secs
    show_config(cfg)
    if args.show_config:
        return
    # `off` resolves to the sentinel cap 1, which is not a real think budget — no
    # think block is emitted at all, so the think-only-output guard does not apply.
    if (cfg['thinking_budget'] != 'off'
            and cfg['thinking_cap_tokens']
            and args.max_tokens <= cfg['thinking_cap_tokens']):
        sys.exit(
            f"serve_harness: max_tokens ({args.max_tokens}) <= thinking budget "
            f"'{cfg['thinking_budget']}' ({cfg['thinking_cap_tokens']} tok) guarantees "
            f"think-only output with zero visible answer. Raise --max-tokens above "
            f"{cfg['thinking_cap_tokens']}, lower --thinking (low={THINKING_BUDGET['low']}), "
            f"or use --thinking uncapped."
        )
    log_offset = 0
    if not args.no_spawn:
        log_offset = spawn_serve(cfg, args.home, args.serve_log)
        if log_offset is None:
            sys.exit("serve_harness: serve failed to warm after retries")
        head = subprocess.run(f"grep -c 'MTP head loaded' {args.serve_log}", shell=True,
                              capture_output=True, text=True).stdout.strip()
        print(f"  [serve warm; MTP head loaded lines={head}]", flush=True)
        _assert_serve_path_proofs(cfg, args.serve_log, offset=log_offset)
    rows = run(cfg, args)
    if not args.no_spawn:
        _assert_dflash_request_proofs(cfg, rows, args.serve_log, offset=log_offset)
        _kill_serve()
    missing = [r.get("retrieval_missing") for r in rows if r.get("retrieval_missing")]
    if missing:
        sys.exit(f"serve_harness: retrieval gate failed; missing expected substrings: {missing}")


if __name__ == "__main__":
    main()
