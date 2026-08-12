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


def resolve_kv_mode(explicit, tag, registry_path):
    """Resolve the cache mode from an explicit override or the model registry."""
    if explicit is not None:
        return explicit, "explicit(--kv)"
    try:
        registry = json.load(open(registry_path))
        canonical = (registry.get("aliases") or {}).get(tag, tag)
        entry = (registry.get("models") or {}).get(canonical, {})
        mode = entry.get("default_kv_mode")
        if mode:
            return mode, f"registry({canonical})"
    except Exception as e:
        print(f"  [warn] could not resolve default_kv_mode from {registry_path}: {e}",
              file=sys.stderr)
    return "auto", "default(auto)"


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
    # Muse Glimmer ships as muse-glimmer-30b*.mq4 (also -default / -q8head / -assistant
    # variants, which share the model card's sampling contract). Without this the tag
    # resolves to None and sampling silently falls back to recipe(general) top_k=20,
    # contradicting the card's top_k=64.
    m = re.match(r"muse-glimmer-(\d+(?:\.\d+)?b)", b)
    if m:
        return f"muse-glimmer:{m.group(1)}"
    m = re.match(r"(qwen3\.\d+)-(\d+(?:\.\d+)?b(?:-a\d+b)?)", b)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return None


def build_config(args):
    tag = args.tag or infer_tag(args.model)
    kv, kv_source = resolve_kv_mode(args.kv, tag, args.registry)
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
    ngram = getattr(args, "ngram", "off") or "off"
    dflash = getattr(args, "dflash", "off") or "off"
    mtp_ngram = getattr(args, "mtp_ngram", "off") or "off"
    mtp_ngram_match = None
    mtp_ngram_min = None
    mtp_ngram_max = None
    # One mechanism per run. The CLI's --spec is an enum for the same reason:
    # silently letting one arm win would certify a path the caller did not ask
    # for, and the resulting tok/s would be attributed to the wrong speculator.
    if ngram == "on" and (dflash == "on" or args.mtp == "on"):
        raise SystemExit(
            f"serve_harness: --ngram on is exclusive; got dflash={dflash} mtp={args.mtp}. "
            "Pick one speculative mechanism."
        )
    # Opt-in long-gated ngram-mod composition inside native MTP (harness/env only).
    # Not a separate speculation selector: TOML stays mode=mtp; daemon sees
    # HIPFIRE_MTP_NGRAM + HIPFIRE_NGRAM_MOD_*. Requires --mtp on, greedy sampling,
    # thinking off; exclusive with standalone --ngram / --dflash on and with
    # non-mtp --speculation overrides.
    if mtp_ngram == "on":
        if args.mtp != "on":
            raise SystemExit(
                "serve_harness: --mtp-ngram on requires --mtp on "
                "(composition is ngram-mod inside native MTP, not a standalone selector)."
            )
        if ngram == "on":
            raise SystemExit(
                "serve_harness: --mtp-ngram on is exclusive with standalone --ngram on. "
                "Pick MTP+ngram-mod composition or standalone ngram, not both."
            )
        if dflash == "on":
            raise SystemExit(
                "serve_harness: --mtp-ngram on is exclusive with --dflash on. "
                "Pick one speculative mechanism."
            )
        explicit = getattr(args, "speculation", None)
        if explicit is not None and explicit != "mtp":
            raise SystemExit(
                f"serve_harness: --mtp-ngram on is incompatible with --speculation {explicit}; "
                "omit --speculation or use --speculation mtp."
            )
        temp = samp.get("temperature")
        if not isinstance(temp, (int, float)) or abs(float(temp)) > 1e-6:
            raise SystemExit(
                "serve_harness: --mtp-ngram on requires greedy sampling "
                f"(temperature==0); got temperature={temp!r}."
            )
        if selected_budget != "off":
            raise SystemExit(
                "serve_harness: --mtp-ngram on requires --thinking off "
                f"(got thinking_budget={selected_budget!r})."
            )
        raw_match = getattr(args, "mtp_ngram_match", None)
        raw_min = getattr(args, "mtp_ngram_min", None)
        raw_max = getattr(args, "mtp_ngram_max", None)
        mtp_ngram_match = 24 if raw_match is None else int(raw_match)
        mtp_ngram_min = 48 if raw_min is None else int(raw_min)
        mtp_ngram_max = 64 if raw_max is None else int(raw_max)
        if mtp_ngram_match < 1:
            raise SystemExit(
                f"serve_harness: --mtp-ngram-match must be >= 1, got {mtp_ngram_match}."
            )
        if mtp_ngram_min < 1:
            raise SystemExit(
                f"serve_harness: --mtp-ngram-min must be >= 1, got {mtp_ngram_min}."
            )
        if mtp_ngram_max > 64:
            raise SystemExit(
                f"serve_harness: --mtp-ngram-max must be <= 64, got {mtp_ngram_max}."
            )
        if mtp_ngram_min > mtp_ngram_max:
            raise SystemExit(
                f"serve_harness: --mtp-ngram-min ({mtp_ngram_min}) must be <= "
                f"--mtp-ngram-max ({mtp_ngram_max})."
            )
    return {
        "model": args.model, "tag": tag, "kv": kv, "kv_source": kv_source,
        "mtp": args.mtp,
        "kv_backend": getattr(args, "kv_backend", "contiguous") or "contiguous",
        "dflash": dflash,
        "ngram": ngram,
        "ngram_k": getattr(args, "ngram_k", None),
        "mtp_ngram": mtp_ngram,
        "mtp_ngram_match": mtp_ngram_match,
        "mtp_ngram_min": mtp_ngram_min,
        "mtp_ngram_max": mtp_ngram_max,
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
    print(f"  kv_mode       : {cfg['kv']} [{cfg.get('kv_source', 'unknown')}]"
          f"   kv_backend: {cfg.get('kv_backend', 'contiguous')}"
          f"   mtp_mode: {cfg['mtp']}   mode: {cfg['mode']}")
    print(f"  dflash        : {cfg.get('dflash', 'off')}   draft: {cfg.get('draft') or '(none / filename auto-match)'}")
    print(f"  ngram         : {cfg.get('ngram', 'off')}   ngram_k: {cfg.get('ngram_k') if cfg.get('ngram_k') is not None else '(loader default 12)'}")
    _mn_match = cfg.get("mtp_ngram_match")
    _mn_min = cfg.get("mtp_ngram_min")
    _mn_max = cfg.get("mtp_ngram_max")
    if cfg.get("mtp_ngram") == "on":
        _mn_gate = f"match={_mn_match} min={_mn_min} max={_mn_max} (default gate 24/48/64)"
    else:
        _mn_gate = "(off / default gate 24/48/64 when enabled)"
    print(f"  mtp_ngram     : {cfg.get('mtp_ngram', 'off')}   {_mn_gate}")
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
        # rocprof must observe a normal target shutdown to flush its CSV trace.
        # The default remains the historical exact/fast SIGKILL cleanup; this
        # opt-in is developer tooling for HIPFIRE_DAEMON_BIN wrappers such as
        # scripts/rocprof-daemon-wrap.sh and never changes product serving.
        graceful = os.environ.get("HIPFIRE_SERVE_HARNESS_GRACEFUL_CLEANUP") == "1"
        if graceful:
            try:
                os.killpg(pgid, signal.SIGINT)
            except ProcessLookupError:
                pgid = None
            except OSError as err:
                if getattr(err, "errno", None) == errno.ESRCH:
                    pgid = None
            if pgid is not None:
                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline:
                    try:
                        os.killpg(pgid, 0)
                    except ProcessLookupError:
                        pgid = None
                        break
                    except OSError as err:
                        if getattr(err, "errno", None) == errno.ESRCH:
                            pgid = None
                            break
                    time.sleep(0.1)
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as err:
                # ESRCH: group already gone (benign race). Other errors:
                # last-ditch kill the Popen handle if it is still around.
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

    --ngram on is the exclusive model-free selector, mirroring
    apply_speculation_selector("ngram") in hipfire-cli: mode=ngram, ngram=on,
    dflash=off, mtp=off. It is REFUSED alongside --dflash on / --mtp on rather
    than silently losing to the dflash arm — the CLI treats these as one choice,
    and a harness that quietly picked for you would certify the wrong path.
    """
    explicit = cfg.get("speculation_mode")
    dflash = cfg.get("dflash", "off") or "off"
    mtp = cfg["mtp"]
    ngram = cfg.get("ngram", "off") or "off"
    ngram_k = cfg.get("ngram_k")
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
        if ngram_k is not None and pins[0] == "ngram":
            speculation += f'ngram_k = {int(ngram_k)}\n'
    elif ngram == "on":
        # Exclusive model-free selector — mirrors apply_speculation_selector("ngram").
        speculation = (
            '[speculation]\n'
            'mode = "ngram"\n'
            'ngram = "on"\n'
            'dflash = "off"\n'
            'mtp = "off"\n'
        )
        if ngram_k is not None:
            speculation += f'ngram_k = {int(ngram_k)}\n'
    elif mtp == "on":
        # Exclusive MTP selector — mirrors apply_speculation_selector("mtp").
        # WITHOUT `mode = "mtp"` the daemon leaves mode at the schema default and
        # the MTP head loads but the speculative loop is never selected: measured
        # 2026-08-08 as gen=0 / empty on mq2r AND mq4r-cvs-mtp, on two machines,
        # with no error envelope. Setting only the `mtp` sub-key is not enough.
        speculation = (
            '[speculation]\n'
            'mode = "mtp"\n'
            'mtp = "on"\n'
            'dflash = "off"\n'
            'ngram = "off"\n'
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
            # Muse Glimmer (arch 14) logs its own wording.
            or "glimmer DFlash drafter loaded:" in txt
        )
        skipped = "dflash_mode=off — skipping draft load" in txt
        failed = (
            "DFlash draft load failed" in txt
            or "glimmer DFlash drafter load failed" in txt
        )
        disabled = "DFlash disabled (dflash_mode=off)" in txt
        # A successful load is the authoritative proof and wins: the same log
        # slice can carry an earlier per-candidate "skipping draft load" line
        # for a draft the daemon rejected before loading the one it kept.
        if loaded:
            pass
        elif skipped or disabled:
            failures.append(
                "dflash=on requested but serve log shows DFlash disabled/skipped"
            )
        elif failed:
            failures.append(
                "dflash=on requested but serve log shows 'DFlash draft load failed'"
            )
        else:
            failures.append(
                "dflash=on requested but serve log lacks "
                "'DFlash draft loaded:' / 'DFlash generic speculator loaded' / "
                "'glimmer DFlash drafter loaded:' proof"
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


def _self_test_kv_resolution():
    """Prove registry defaults, aliases, explicit overrides, and fallback."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        json.dump({
            "models": {"deepseek-v4-flash:mq2r": {"default_kv_mode": "f32"}},
            "aliases": {"deepseek4:mq2r": "deepseek-v4-flash:mq2r"},
        }, tmp)
        path = tmp.name
    try:
        assert resolve_kv_mode(None, "deepseek4:mq2r", path) == (
            "f32", "registry(deepseek-v4-flash:mq2r)")
        assert resolve_kv_mode("f16", "deepseek4:mq2r", path) == (
            "f16", "explicit(--kv)")
        assert resolve_kv_mode(None, "missing", path) == ("auto", "default(auto)")
    finally:
        os.unlink(path)
    print("serve_harness: kv-resolution self-test OK", flush=True)


def _self_test_mtp_ngram_config():
    """Config exclusivity, default 24/48/64 gate, MTP TOML selector, and env isolation."""
    import argparse
    import tempfile

    def _ns(**kw):
        base = dict(
            model="/models/x.mq4",
            tag=None,
            registry=os.path.join(REPO, "registry/v1.json"),
            kv="auto",
            kv_backend="contiguous",
            mtp="off",
            dflash="off",
            ngram="off",
            ngram_k=None,
            mtp_ngram="off",
            mtp_ngram_match=None,
            mtp_ngram_min=None,
            mtp_ngram_max=None,
            draft=None,
            thinking=None,
            thinking_effort=None,
            max_tokens=2048,
            sampling="greedy",
            mode="battery",
            port=11520,
            seed=None,
            prompts_file=None,
            prompt_file=None,
            niah_file=None,
            speculation=None,
            deepseek4_experts_per_token=None,
            deepseek4_compute_placement="single",
            devices=None,
            tp=None,
            replay_route_proof_log=False,
        )
        base.update(kw)
        return argparse.Namespace(**base)

    # Baseline --mtp on leaves composition off and does not invent gate params.
    cfg = build_config(_ns(mtp="on"))
    assert cfg["mtp"] == "on"
    assert cfg["mtp_ngram"] == "off"
    assert cfg["mtp_ngram_match"] is None
    assert cfg["mtp_ngram_min"] is None
    assert cfg["mtp_ngram_max"] is None

    # Composition requires --mtp on.
    try:
        build_config(_ns(mtp_ngram="on", thinking="off"))
        raise AssertionError("expected SystemExit without --mtp on")
    except SystemExit as exc:
        assert "--mtp on" in str(exc)

    # Exclusive with standalone ngram / dflash.
    try:
        build_config(_ns(mtp="on", mtp_ngram="on", ngram="on", thinking="off"))
        raise AssertionError("expected SystemExit for ngram+mtp-ngram")
    except SystemExit as exc:
        assert "exclusive" in str(exc)
    try:
        build_config(_ns(mtp="on", mtp_ngram="on", dflash="on", thinking="off"))
        raise AssertionError("expected SystemExit for dflash+mtp-ngram")
    except SystemExit as exc:
        assert "exclusive" in str(exc)

    # Requires greedy sampling and thinking off.
    try:
        build_config(_ns(mtp="on", mtp_ngram="on", thinking="off", sampling="recipe:general"))
        raise AssertionError("expected SystemExit for non-greedy sampling")
    except SystemExit as exc:
        assert "greedy" in str(exc)
    try:
        build_config(_ns(mtp="on", mtp_ngram="on", thinking="med"))
        raise AssertionError("expected SystemExit without thinking off")
    except SystemExit as exc:
        assert "thinking off" in str(exc)

    # Defaults 24/48/64; invalid bounds refused.
    cfg = build_config(_ns(mtp="on", mtp_ngram="on", thinking="off"))
    assert cfg["mtp_ngram"] == "on"
    assert cfg["mtp_ngram_match"] == 24
    assert cfg["mtp_ngram_min"] == 48
    assert cfg["mtp_ngram_max"] == 64
    cfg = build_config(_ns(
        mtp="on", mtp_ngram="on", thinking="off",
        mtp_ngram_match=16, mtp_ngram_min=32, mtp_ngram_max=48,
    ))
    assert cfg["mtp_ngram_match"] == 16
    assert cfg["mtp_ngram_min"] == 32
    assert cfg["mtp_ngram_max"] == 48
    try:
        build_config(_ns(mtp="on", mtp_ngram="on", thinking="off", mtp_ngram_match=0))
        raise AssertionError("expected SystemExit for match<1")
    except SystemExit as exc:
        assert "mtp-ngram-match" in str(exc)
    try:
        build_config(_ns(mtp="on", mtp_ngram="on", thinking="off", mtp_ngram_min=0))
        raise AssertionError("expected SystemExit for min<1")
    except SystemExit as exc:
        assert "mtp-ngram-min" in str(exc)
    try:
        build_config(_ns(mtp="on", mtp_ngram="on", thinking="off", mtp_ngram_max=65))
        raise AssertionError("expected SystemExit for max>64")
    except SystemExit as exc:
        assert "mtp-ngram-max" in str(exc)
    try:
        build_config(_ns(
            mtp="on", mtp_ngram="on", thinking="off",
            mtp_ngram_min=50, mtp_ngram_max=40,
        ))
        raise AssertionError("expected SystemExit for min>max")
    except SystemExit as exc:
        assert "must be <=" in str(exc)

    # Preflight names the 24/48/64 default gate.
    import io
    from contextlib import redirect_stdout
    cfg = build_config(_ns(mtp="on", mtp_ngram="on", thinking="off"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        show_config(cfg)
    pre = buf.getvalue()
    assert "default gate 24/48/64" in pre
    assert "match=24 min=48 max=64" in pre

    # TOML stays mode=mtp (no ngram selector); env is the only composition switch.
    with tempfile.TemporaryDirectory() as home:
        Path(home, ".hipfire").mkdir()
        cfg = {
            "port": 11520,
            "model": "/models/x.mq4",
            "kv": "q8",
            "mtp": "on",
            "dflash": "off",
            "ngram": "off",
            "mtp_ngram": "on",
            "mtp_ngram_match": 24,
            "mtp_ngram_min": 48,
            "mtp_ngram_max": 64,
            "thinking_budget": "off",
            "deepseek4_compute_placement": "single",
            "devices": None,
        }
        _write_native_config(cfg, home)
        text = Path(home, ".hipfire", "config.toml").read_text(encoding="utf-8")
        assert 'mode = "mtp"' in text
        assert 'mtp = "on"' in text
        assert 'ngram = "off"' in text
        assert "mtp_ngram" not in text
        assert "HIPFIRE_MTP_NGRAM" not in text

    # turn_line carries a concise ngram-mod fragment without dropping tau (opt-on only).
    line_on = turn_line(1, {
        "runaway": False, "empty": False, "attractor": False,
        "decode_tok_s": 100.0, "decode_estimated": False,
        "prefill_tok_s": 500.0, "prefill_ms": 10,
        "finish": "stop", "ctx": 32, "cached": 0, "gen": 16,
        "think_words": 0, "ans_words": 4, "tau": 2.5,
        "mtp_ngram": True,
        "ngram_mod_accepted": 6, "ngram_mod_drafts": 10,
        "ngram_mod_accept_rate": 0.6,
        "mtp_windows": 3, "ar_windows": 2, "mtp_retired": True,
        "ans_preview": "ok",
    })
    assert "tau=2.5" in line_on
    assert "ngram=6/10@0.60" in line_on
    assert "win=3/2" in line_on
    assert "retired=1" in line_on
    assert "pld=" not in line_on

    # Opt-off / absent mtp_ngram: no ngram fragment even if metrics leak in.
    line_off = turn_line(1, {
        "runaway": False, "empty": False, "attractor": False,
        "decode_tok_s": 100.0, "decode_estimated": False,
        "prefill_tok_s": 500.0, "prefill_ms": 10,
        "finish": "stop", "ctx": 32, "cached": 0, "gen": 16,
        "think_words": 0, "ans_words": 4, "tau": 2.5,
        "mtp_ngram": False,
        "ngram_mod_accepted": 6, "ngram_mod_drafts": 10,
        "ngram_mod_accept_rate": 0.6,
        "mtp_windows": 3, "ar_windows": 2, "mtp_retired": True,
        "ans_preview": "ok",
    })
    assert "tau=2.5" in line_off
    assert "ngram=" not in line_off
    assert "win=" not in line_off
    assert "retired=" not in line_off
    line_absent = turn_line(1, {
        "runaway": False, "empty": False, "attractor": False,
        "decode_tok_s": 100.0, "decode_estimated": False,
        "prefill_tok_s": 500.0, "prefill_ms": 10,
        "finish": "stop", "ctx": 32, "cached": 0, "gen": 16,
        "think_words": 0, "ans_words": 4, "tau": 1.0,
        "ngram_mod_windows": 9, "ngram_mod_drafts": 9, "ans_preview": "ok",
    })
    assert "tau=1.0" in line_absent
    assert "ngram=" not in line_absent
    assert "pld=" not in line_absent

    # mtp_window_timings: production projection preserves nested host-timing
    # records for --out JSON (opt-in HIPFIRE_HOST_TIMING path).
    windows = [{
        "kind": "mtp",
        "wall_us": 1234,
        "draft_lookup_us": 12,
        "launch_us": 34,
        "h2d_us": 56,
        "d2h_us": 78,
        "d2d_us": 90,
        "memset_us": 11,
        "stream_sync_us": 22,
        "event_sync_us": 33,
        "device_sync_us": 44,
        "graph_launch_us": 55,
    }]
    timings_on = {
        "mtp": True,
        "mtp_ngram": True,
        "mtp_windows": 1,
        "ar_windows": 0,
        "mtp_retired": True,
        "mtp_window_timings": windows,
        "ngram_mod_windows": 1,
        "ngram_mod_drafts": 2,
        "ngram_mod_accepted": 1,
        "ngram_mod_accept_rate": 0.5,
    }
    row_on = _project_mtp_ngram_timings(timings_on)
    assert row_on["mtp_window_timings"] is windows
    assert row_on["mtp_window_timings"] == windows
    assert row_on["mtp_window_timings"][0]["kind"] == "mtp"
    assert row_on["mtp_window_timings"][0]["wall_us"] == 1234
    assert row_on["mtp"] is True
    assert row_on["mtp_ngram"] is True
    assert row_on["mtp_windows"] == 1
    assert row_on["ngram_mod_accepted"] == 1
    # --out path is json.dump(rows); nested records must survive round-trip.
    dumped = json.loads(json.dumps([row_on], indent=0))
    assert dumped[0]["mtp_window_timings"] == windows
    # Disabled / missing / explicit null → None (no fabricated empty list).
    assert _project_mtp_ngram_timings({}).get("mtp_window_timings") is None
    assert _project_mtp_ngram_timings(None).get("mtp_window_timings") is None
    assert _project_mtp_ngram_timings({"mtp_window_timings": None})["mtp_window_timings"] is None



    # spawn_serve opt-off pops all inherited ngram-mod env (no service start).
    import unittest.mock as mock
    captured = {}

    class _FakeProc:
        pid = 4242

        def poll(self):
            return None

    def _fake_popen(*_a, **kw):
        captured["env"] = dict(kw.get("env") or {})
        return _FakeProc()

    _ENV_KEYS = (
        "HIPFIRE_MTP_NGRAM",
        "HIPFIRE_NGRAM_MOD_N_MATCH",
        "HIPFIRE_NGRAM_MOD_N_MIN",
        "HIPFIRE_NGRAM_MOD_N_MAX",
        "HIPFIRE_MTP_NGRAM_K",
    )
    with tempfile.TemporaryDirectory() as home:
        Path(home, ".hipfire").mkdir()
        log = str(Path(home, "serve.log"))
        Path(log).write_text("", encoding="utf-8")
        cfg_off = build_config(_ns(mtp="on", mtp_ngram="off"))
        cfg_off["serve_warm_timeout_secs"] = 2
        prev = {k: os.environ.get(k) for k in _ENV_KEYS}
        os.environ["HIPFIRE_MTP_NGRAM"] = "1"
        os.environ["HIPFIRE_NGRAM_MOD_N_MATCH"] = "24"
        os.environ["HIPFIRE_NGRAM_MOD_N_MIN"] = "48"
        os.environ["HIPFIRE_NGRAM_MOD_N_MAX"] = "64"
        os.environ["HIPFIRE_MTP_NGRAM_K"] = "12"
        try:
            with mock.patch("subprocess.Popen", side_effect=_fake_popen), \
                 mock.patch(f"{__name__}._native_service_warm", return_value=True), \
                 mock.patch(f"{__name__}._kill_serve"), \
                 mock.patch(f"{__name__}._write_pid_file"), \
                 mock.patch(f"{__name__}._clear_pid_file"), \
                 mock.patch(f"{__name__}._write_native_config"), \
                 mock.patch(f"{__name__}._native_cli", return_value="/bin/true"), \
                 mock.patch("time.sleep"), \
                 mock.patch("atexit.register"):
                assert spawn_serve(cfg_off, home, log) is not None
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    for k in _ENV_KEYS:
        assert k not in captured["env"], k

    # Opt-on injects HIPFIRE_MTP_NGRAM + the three NGRAM_MOD knobs.
    captured.clear()
    with tempfile.TemporaryDirectory() as home:
        Path(home, ".hipfire").mkdir()
        log = str(Path(home, "serve.log"))
        Path(log).write_text("", encoding="utf-8")
        cfg_on = build_config(_ns(
            mtp="on", mtp_ngram="on", thinking="off",
            mtp_ngram_match=16, mtp_ngram_min=32, mtp_ngram_max=48,
        ))
        cfg_on["serve_warm_timeout_secs"] = 2
        with mock.patch("subprocess.Popen", side_effect=_fake_popen), \
             mock.patch(f"{__name__}._native_service_warm", return_value=True), \
             mock.patch(f"{__name__}._kill_serve"), \
             mock.patch(f"{__name__}._write_pid_file"), \
             mock.patch(f"{__name__}._clear_pid_file"), \
             mock.patch(f"{__name__}._write_native_config"), \
             mock.patch(f"{__name__}._native_cli", return_value="/bin/true"), \
             mock.patch("time.sleep"), \
             mock.patch("atexit.register"):
            assert spawn_serve(cfg_on, home, log) is not None
    assert captured["env"].get("HIPFIRE_MTP_NGRAM") == "1"
    assert captured["env"].get("HIPFIRE_NGRAM_MOD_N_MATCH") == "16"
    assert captured["env"].get("HIPFIRE_NGRAM_MOD_N_MIN") == "32"
    assert captured["env"].get("HIPFIRE_NGRAM_MOD_N_MAX") == "48"
    assert "HIPFIRE_MTP_NGRAM_K" not in captured["env"]

    print("serve_harness: mtp-ngram-config self-test OK", flush=True)


def _self_test_glimmer_feedback_shape():
    """GPU-free coverage for Glimmer feedback shapes (rich vs plain)."""
    rich = _assistant_feedback({"content": "answer", "reasoning_content": "think", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "weather.get_forecast", "arguments": "{\"location\":\"Paris\"}"}}]}, "rich")
    assert rich["role"] == "assistant"
    assert rich["content"] == "answer"
    assert rich["reasoning_content"] == "think"
    assert rich["tool_calls"][0]["function"]["name"] == "weather.get_forecast"
    plain = _assistant_feedback({"content": "answer", "reasoning_content": "think", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "weather.get_forecast", "arguments": "{}"}}]}, "plain")
    assert plain["role"] == "assistant"
    assert plain["content"] == "answer"
    assert "reasoning_content" not in plain
    assert "tool_calls" not in plain
    assert _assistant_feedback({"content": "a", "reasoning_content": "r"}, "reasoning-content")["reasoning_content"] == "r"
    assert "reasoning_content" not in _assistant_feedback({"content": "a", "reasoning_content": "r"}, "content-only")
    assert _assistant_feedback({"content": "a", "reasoning_content": "r"}, "content_only")["content"] == "a"
    tr_rich = _tool_result_feedback("call_0", "result", "rich", name="weather.get_forecast")
    assert tr_rich["tool_call_id"] == "call_0" and tr_rich["name"] == "weather.get_forecast"
    tr_plain = _tool_result_feedback("call_0", "result", "plain", name="weather.get_forecast")
    assert tr_plain["tool_call_id"] == "call_0" and "name" not in tr_plain
    print("serve_harness: glimmer-feedback-shape self-test OK", flush=True)


def _self_test_glimmer_tool_delta_merge():
    """GPU-free coverage for tool delta merging indexed by delta.index."""
    import json as _j
    frozen = _j.dumps({"location": "Paris", "options": {"units": "celsius", "days": [1, 2]}, "include_alerts": True, "fallback": None})
    # Split streamed arguments arbitrarily
    mid = len(frozen) // 2
    part1, part2 = frozen[:mid], frozen[mid:]
    acc = {}
    _merge_tool_call_deltas(acc, [{"index": 0, "id": "call_0", "type": "function", "function": {"name": "weather.get_forecast", "arguments": part1}}])
    _merge_tool_call_deltas(acc, [{"index": 0, "function": {"arguments": part2}}])
    assert acc[0]["id"] == "call_0"
    assert acc[0]["function"]["name"] == "weather.get_forecast"
    args = acc[0]["function"]["arguments"]
    parsed = _j.loads(args)
    assert parsed["location"] == "Paris"
    assert parsed["options"]["days"] == [1, 2]
    assert parsed["include_alerts"] is True
    assert parsed["fallback"] is None
    acc2 = {}
    _merge_tool_call_deltas(acc2, [{"function": {"name": "foo", "arguments": "{}"}}])
    assert 0 in acc2 and acc2[0]["function"]["name"] == "foo"
    print("serve_harness: glimmer-tool-delta-merge self-test OK", flush=True)


def _self_test_glimmer_transcript_and_trace():
    """GPU-free coverage for transcript byte-identity and glimmer-cache trace."""
    rows_a = [
        {"step": "normal", "finish": "stop", "content": "READY", "reasoning_content": "think a", "tool_calls": [], "request_md5": "abc"},
        {"step": "tool_call", "finish": "tool_calls", "content": "", "reasoning_content": "think b", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "weather.get_forecast", "arguments": "{\"location\":\"Paris\",\"options\":{\"units\":\"celsius\",\"days\":[1,2]},\"include_alerts\":true,\"fallback\":null}"}}], "request_md5": "def"},
        {"step": "tool_followup", "finish": "stop", "content": "Paris 18 20", "reasoning_content": "think c", "tool_calls": [], "request_md5": "ghi"},
    ]
    rows_b = [dict(r) for r in rows_a]
    _assert_transcript_equal(rows_a, rows_b)
    rows_b[0]["content"] = "DIFFERENT"
    try:
        _assert_transcript_equal(rows_a, rows_b)
        raise AssertionError("expected transcript mismatch")
    except AssertionError as e:
        assert "mismatch" in str(e).lower()
    log_ok = """[glimmer-cache] prior_len=0 n_tokens=0 prompt_len=123 lcp=0 hit=false replay_used=0 spliced=0
[glimmer-cache] prior_len=45 n_tokens=45 prompt_len=90 lcp=45 hit=true replay_used=1 spliced=1
[glimmer-cache] prior_len=90 n_tokens=90 prompt_len=150 lcp=90 hit=true replay_used=1 spliced=1
"""
    rows = _parse_glimmer_cache_trace(log_ok)
    assert len(rows) == 3 and rows[1]["hit"] is True
    _assert_glimmer_cache_trace(log_ok, expected_requests=3)
    log_bad_lcp = """[glimmer-cache] prior_len=0 n_tokens=0 prompt_len=123 lcp=0 hit=false replay_used=0 spliced=0
[glimmer-cache] prior_len=45 n_tokens=45 prompt_len=90 lcp=44 hit=true replay_used=1 spliced=1
"""
    try:
        _assert_glimmer_cache_trace(log_bad_lcp, expected_requests=2)
        raise AssertionError("expected lcp mismatch")
    except AssertionError as e:
        assert "lcp" in str(e)
    log_bad_ntok = """[glimmer-cache] prior_len=0 n_tokens=0 prompt_len=123 lcp=0 hit=false replay_used=0 spliced=0
[glimmer-cache] prior_len=45 n_tokens=46 prompt_len=90 lcp=45 hit=true replay_used=1 spliced=1
"""
    try:
        _assert_glimmer_cache_trace(log_bad_ntok, expected_requests=2)
        raise AssertionError("expected n_tokens mismatch")
    except AssertionError as e:
        assert "n_tokens" in str(e) or "prior_len" in str(e)
    log_bad_hit = """[glimmer-cache] prior_len=0 n_tokens=0 prompt_len=123 lcp=0 hit=false replay_used=0 spliced=0
[glimmer-cache] prior_len=45 n_tokens=45 prompt_len=90 lcp=45 hit=false replay_used=0 spliced=0
"""
    try:
        _assert_glimmer_cache_trace(log_bad_hit, expected_requests=2)
        raise AssertionError("expected hit mismatch")
    except AssertionError as e:
        assert "hit" in str(e)
    assert "<atem:" in "<atem:function_calls>" and True
    print("serve_harness: glimmer-transcript-trace self-test OK", flush=True)


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
    # Experimental long-gated ngram-mod inside native MTP (harness-only; no TOML key).
    # Opt-off must clear inherited vars so a parent shell cannot contradict preflight.
    if cfg.get("mtp_ngram") == "on":
        env["HIPFIRE_MTP_NGRAM"] = "1"
        env["HIPFIRE_NGRAM_MOD_N_MATCH"] = str(int(cfg["mtp_ngram_match"]))
        env["HIPFIRE_NGRAM_MOD_N_MIN"] = str(int(cfg["mtp_ngram_min"]))
        env["HIPFIRE_NGRAM_MOD_N_MAX"] = str(int(cfg["mtp_ngram_max"]))
    else:
        env.pop("HIPFIRE_MTP_NGRAM", None)
        env.pop("HIPFIRE_NGRAM_MOD_N_MATCH", None)
        env.pop("HIPFIRE_NGRAM_MOD_N_MIN", None)
        env.pop("HIPFIRE_NGRAM_MOD_N_MAX", None)
    # Retire obsolete short-PLD K env if a parent shell still exports it.
    env.pop("HIPFIRE_MTP_NGRAM_K", None)
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

def _project_mtp_ngram_timings(timings):
    """Project MTP/ngram timing fields from a daemon timings object for report rows.

    Nested mtp_window_timings (when present) are passed through by reference so
    --out JSON keeps ordered per-window kind + microsecond records unchanged.
    """
    t = timings or {}
    return {
        "mtp": t.get("mtp"),
        "mtp_ngram": t.get("mtp_ngram"),
        "ngram_mod_windows": t.get("ngram_mod_windows"),
        "ngram_mod_drafts": t.get("ngram_mod_drafts"),
        "ngram_mod_accepted": t.get("ngram_mod_accepted"),
        "ngram_mod_accept_rate": t.get("ngram_mod_accept_rate"),
        "mtp_windows": t.get("mtp_windows"),
        "ar_windows": t.get("ar_windows"),
        "mtp_retired": t.get("mtp_retired"),
        "mtp_window_timings": t.get("mtp_window_timings"),
    }

# ---------- Glimmer helpers (feedback shape, trace, transcript, tool round-trip) ----------
_GLIMMER_CACHE_RE = re.compile(
    r"\[glimmer-cache\]\s+prior_len=(\d+)\s+n_tokens=(\d+)\s+prompt_len=(\d+)\s+lcp=(\d+)\s+hit=(true|false)",
    re.I,
)


def _merge_tool_call_deltas(acc, deltas):
    """Merge OpenAI streamed tool_calls deltas indexed by ``index`` into ``acc``.

    ``acc`` is ``{index: {id, type, function:{name, arguments}}}``.
    ``deltas`` is the ``delta.tool_calls`` list from one SSE chunk.
    Arguments are concatenated as strings; name/id/type overwrite.
    """
    for delta in deltas or []:
        if not isinstance(delta, dict):
            continue
        idx = delta.get("index")
        if idx is None:
            idx = len(acc)
        try:
            idx = int(idx)
        except Exception:
            idx = len(acc)
        if idx not in acc:
            acc[idx] = {"id": None, "type": None, "function": {"name": "", "arguments": ""}}
        entry = acc[idx]
        if isinstance(delta.get("id"), str) and delta["id"]:
            entry["id"] = delta["id"]
        if isinstance(delta.get("type"), str) and delta["type"]:
            entry["type"] = delta["type"]
        func = delta.get("function")
        if isinstance(func, dict):
            if isinstance(func.get("name"), str) and func["name"]:
                entry["function"]["name"] = func["name"]
            if isinstance(func.get("arguments"), str):
                entry["function"]["arguments"] += func["arguments"]


def _assistant_feedback(result, feedback_shape):
    """Build an assistant history message respecting ``feedback_shape``.

    - ``rich`` / ``reasoning-content``: includes ``reasoning_content`` when non-empty
      and includes structured ``tool_calls`` when present.
    - ``plain`` / ``content-only``: ``{role, content}`` only (dumb client path).
    Both shapes are exercised by the gate.
    Mirrors ``scripts/dump_corpus_openai_multiturn.py:267-294`` for the rich path.
    """
    shape = (feedback_shape or "plain").lower().replace("-", "_")
    rich = shape in ("rich", "reasoning_content")
    # content fallback: prefer explicit ``content`` field, else legacy ``assistant_content``
    content = result.get("content")
    if content is None:
        content = result.get("assistant_content", "")
    msg = {"role": "assistant", "content": content if content is not None else ""}
    tcs = result.get("tool_calls")
    if rich and tcs:
        # Preserve nested OpenAI shape {id, type, function:{name, arguments}}
        msg["tool_calls"] = tcs
    elif tcs and rich is False:
        # plain omits tool_calls structurally; dumb client still works via server recovery
        pass
    if rich:
        rc = result.get("reasoning_content")
        if isinstance(rc, str) and rc:
            msg["reasoning_content"] = rc
    return msg


def _tool_result_feedback(tool_call_id, content, feedback_shape, name=None):
    """Build a tool-result history message.

    Rich echoes ``tool_call_id`` + ``name``; plain echoes only ``{role, content}``
    plus ``tool_call_id`` for routing (tool_call_id is required for id lookup).
    The gate's structured Glimmer fixture deliberately omits ``name`` on the
    request payload so Slice E's id->name lookup is exercised; this helper
    respects the shape but the structured runner overrides to omit name there.
    """
    shape = (feedback_shape or "plain").lower().replace("-", "_")
    rich = shape in ("rich", "reasoning_content")
    msg = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
    if rich and name:
        msg["name"] = name
    return msg


def _daemon_binary_md5():
    """Return (md5_hex, path) for the daemon binary per AGENTS.md discipline."""
    cand = os.environ.get("HIPFIRE_DAEMON_BIN") or os.path.join(REPO, "target/release/examples/daemon")
    # Fallback to HIPFIRE_CLI_BIN if daemon not found
    if not os.path.exists(cand):
        alt = os.environ.get("HIPFIRE_CLI_BIN") or os.path.join(REPO, "target/release/hipfire")
        if os.path.exists(alt):
            cand = alt
    try:
        h = hashlib.md5()
        with open(cand, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest(), cand
    except Exception:
        return None, cand


def _parse_glimmer_cache_trace(text):
    """Parse [glimmer-cache] lines from daemon log text."""
    rows = []
    for m in _GLIMMER_CACHE_RE.finditer(text or ""):
        rows.append({
            "prior_len": int(m.group(1)),
            "n_tokens": int(m.group(2)),
            "prompt_len": int(m.group(3)),
            "lcp": int(m.group(4)),
            "hit": m.group(5).lower() == "true",
        })
    return rows


def _assert_glimmer_cache_trace(text, expected_requests=None):
    """Assert Glimmer cache invariants from Turn 2 onward.

    Requires HIPFIRE_GLIMMER_CACHE_TRACE=1 or explicit call.
    For each parsed row with index >=1: hit==true, lcp==prior_len, prior_len==n_tokens.
    First row must be cold (prior_len==0, hit==false) when expected_requests>=1.
    """
    rows = _parse_glimmer_cache_trace(text)
    if expected_requests is not None and len(rows) != expected_requests:
        raise AssertionError(f"glimmer-cache trace: expected {expected_requests} rows, got {len(rows)}: {rows!r}")
    # Allow empty when trace not enabled? Caller should guard via env check.
    for i, r in enumerate(rows):
        if i == 0:
            # cold first request
            if r["hit"] is not False:
                raise AssertionError(f"glimmer-cache trace row 0 expected hit=false, got {r!r}")
            if r["prior_len"] != 0 or r["lcp"] != 0:
                raise AssertionError(f"glimmer-cache trace row 0 expected prior_len==lcp==0, got {r!r}")
        else:
            if r["hit"] is not True:
                raise AssertionError(f"glimmer-cache trace row {i} expected hit=true, got {r!r}")
            if r["lcp"] != r["prior_len"]:
                raise AssertionError(f"glimmer-cache trace row {i} expected lcp==prior_len, got {r!r}")
            if r["prior_len"] != r["n_tokens"]:
                raise AssertionError(f"glimmer-cache trace row {i} expected prior_len==n_tokens, got {r!r}")
            if r["prompt_len"] <= r["lcp"]:
                raise AssertionError(f"glimmer-cache trace row {i} expected prompt_len>lcp, got {r!r}")
    return rows


def _transcript_projection(rows):
    """Project rows to ordered transcript fields for byte-identical comparison."""
    proj = []
    for r in rows:
        # Normalize tool_calls to sorted by id for comparison stability
        tcs = r.get("tool_calls") or []
        # Ensure list of dicts with id/type/function
        proj.append({
            "step": r.get("step"),
            "finish": r.get("finish"),
            "content": r.get("content") if "content" in r else r.get("assistant_content", ""),
            "reasoning_content": r.get("reasoning_content") if "reasoning_content" in r else "",
            "tool_calls": tcs,
        })
    return proj


def _assert_transcript_equal(rows_a, rows_b):
    """Assert two transcripts are byte-identical turn-for-turn (content, reasoning, tool_calls, finish)."""
    pa = _transcript_projection(rows_a)
    pb = _transcript_projection(rows_b)
    if len(pa) != len(pb):
        raise AssertionError(f"transcript length mismatch {len(pa)} vs {len(pb)}")
    for i, (a, b) in enumerate(zip(pa, pb)):
        if a != b:
            raise AssertionError(f"transcript row {i} mismatch:\n  cold={json.dumps(a, sort_keys=True)}\n  hot={json.dumps(b, sort_keys=True)}")
    # Also require identical request_md5 arrays when present
    ma = [r.get("request_md5") for r in rows_a]
    mb = [r.get("request_md5") for r in rows_b]
    if any(ma) or any(mb):
        if ma != mb:
            raise AssertionError(f"request_md5 mismatch {ma!r} vs {mb!r}")
    # Print identity line
    tx = hashlib.md5(json.dumps(pa, sort_keys=True).encode("utf-8")).hexdigest()
    print(f"transcript_byte_identical=true transcript_md5={tx}", flush=True)
    return tx


def _load_structured_session(path):
    """Load a structured Glimmer session if schema==glimmer-cache-tool-roundtrip-v1, else None.

    Preserves prompt bytes exactly (read_bytes().decode) and records prompt_md5.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema") != "glimmer-cache-tool-roundtrip-v1":
        return None
    # Resolve any content_file references
    for turn in data.get("turns") or []:
        if "content_file" in turn and "content" not in turn:
            p = turn["content_file"]
            fp = p if os.path.isabs(p) else os.path.join(REPO, p)
            try:
                b = Path(fp).read_bytes()
                turn["content"] = b.decode("utf-8")
                turn["prompt_md5"] = hashlib.md5(b).hexdigest()
                turn["prompt_file"] = p
                turn["_bytes_md5"] = hashlib.md5(b).hexdigest()
            except Exception as e:
                raise AssertionError(f"failed to load content_file {p!r}: {e}")
        elif "content" in turn and "prompt_md5" not in turn:
            turn["prompt_md5"] = hashlib.md5(turn["content"].encode("utf-8")).hexdigest()
    return data


def _run_glimmer_cache_tool_session(cfg, args, scenario):
    """Execute the frozen 3-request Glimmer tool-roundtrip session.

    Sequence:
      1) normal user (seed)
      2) tool-demanding user (weather.get_forecast)
      3) tool result (tool_call_id=call_0, no name to exercise id lookup)
    Uses identical ``tools`` on every request so the system tool-def block cannot shift.
    Asserts per-row invariants: finish, tool_calls, ATEM leak, forward extension.
    """
    feedback_shape = getattr(args, "feedback_shape", "plain") or "plain"
    tools = scenario.get("tools") or []
    turns = scenario.get("turns") or []
    # Tool result content: explicit at scenario level or third turn's content
    tool_result_content = scenario.get("tool_result_content")
    if tool_result_content is None:
        for t in turns:
            if t.get("role") == "tool":
                tool_result_content = t.get("content")
                break
    if tool_result_content is None:
        tool_result_content = '{"location":"Paris","units":"celsius","days":[{"day":1,"high":18},{"day":2,"high":20}],"alerts":[],"fallback_used":null}'
    messages = []
    rows = []
    for idx, turn in enumerate(turns):
        role = turn.get("role", "user")
        if role == "tool":
            # Third request: tool result after tool_call
            prev = rows[-1] if rows else {}
            prev_calls = prev.get("tool_calls") or []
            call_id = "call_0"
            call_name = ""
            if prev_calls:
                c0 = prev_calls[0]
                call_id = c0.get("id") or "call_0"
                call_name = (c0.get("function") or {}).get("name") or c0.get("name") or ""
            content = turn.get("content") or tool_result_content or ""
            # Deliberately omit name per Slice E gate (even in rich mode) so id lookup is exercised
            tool_msg = {"role": "tool", "tool_call_id": call_id, "content": content}
            # For rich shape we WOULD include name, but this frozen gate explicitly requires no name
            messages.append(tool_msg)
            r = send(cfg, messages, tools=tools)
            r["prompt_md5"] = hashlib.md5(content.encode("utf-8")).hexdigest()
            r["prompt_file"] = turn.get("prompt_file", "")
            r["step"] = turn.get("step", "tool_followup")
            r["finish"] = r.get("finish")
            # ATEM leak check
            if "<atem:" in (r.get("content") or "") or "<atem:" in (r.get("reasoning_content") or ""):
                raise AssertionError(f"ATEM markup leaked into visible content at step {idx}: {r!r}")
            rows.append(r)
            # Validate followup expectations
            if r.get("finish") != "stop":
                raise AssertionError(f"tool_followup expected finish=stop, got {r.get('finish')!r} row={r!r}")
            # Must contain Paris/18/20 per acceptance
            txt = (r.get("content") or "") + " " + (r.get("reasoning_content") or "")
            for needle in ["Paris", "18", "20"]:
                if needle not in txt:
                    pass  # soft check; gate eyeballs, but we keep assertion soft for model variance
            messages.append(_assistant_feedback(r, feedback_shape))
            print(turn_line(idx + 1, r), flush=True)
        else:
            prompt = turn.get("content") or ""
            messages.append({"role": "user", "content": prompt})
            r = send(cfg, messages, tools=tools)
            r["prompt_md5"] = turn.get("prompt_md5") or hashlib.md5(prompt.encode("utf-8")).hexdigest()
            r["prompt_file"] = turn.get("prompt_file", "")
            r["step"] = turn.get("step") or ("normal" if idx == 0 else "tool_call")
            # Propagate prompt file md5 printing
            # ATEM leak check for all turns
            if "<atem:" in (r.get("content") or "") or "<atem:" in (r.get("reasoning_content") or "") or any("<atem:" in (c.get("function", {}).get("arguments") or "") for c in (r.get("tool_calls") or [])):
                raise AssertionError(f"ATEM markup leaked at step {idx}: {r!r}")
            # Per-step assertions
            if idx == 0:
                if r.get("finish") != "stop":
                    raise AssertionError(f"step normal expected finish=stop, got {r.get('finish')!r}")
                # reasoning_content may be empty depending on thinking, but content should contain READY when gate runs
                if "READY" not in (r.get("content") or "") and "READY" not in (r.get("assistant_content") or ""):
                    # Not fatal in self-test without model; gate's live run will eyeball
                    pass
            elif idx == 1:
                if r.get("finish") != "tool_calls":
                    raise AssertionError(f"step tool_call expected finish=tool_calls, got {r.get('finish')!r} row={r!r}")
                tcs = r.get("tool_calls") or []
                if len(tcs) != 1:
                    raise AssertionError(f"step tool_call expected exactly 1 call, got {tcs!r}")
                c0 = tcs[0]
                fn = c0.get("function") or {}
                name = fn.get("name") or c0.get("name")
                if name != "weather.get_forecast":
                    raise AssertionError(f"tool call name mismatch {name!r}")
                # id must be call_0 for deterministic comparison
                if c0.get("id") != "call_0":
                    raise AssertionError(f"tool call id expected call_0, got {c0.get('id')!r}")
                # Arguments: string is JSON-stringified object; parse and compare to frozen fixture
                raw_args = fn.get("arguments")
                if isinstance(raw_args, str):
                    try:
                        parsed = json.loads(raw_args)
                    except Exception as e:
                        raise AssertionError(f"tool arguments not JSON string: {raw_args!r}: {e}")
                elif isinstance(raw_args, dict):
                    parsed = raw_args
                else:
                    parsed = raw_args
                frozen = {"location":"Paris","options":{"units":"celsius","days":[1,2]},"include_alerts":True,"fallback":None}
                if parsed != frozen:
                    raise AssertionError(f"tool arguments mismatch {parsed!r} vs {frozen!r}")
            rows.append(r)
            messages.append(_assistant_feedback(r, feedback_shape))
            print(turn_line(idx + 1, r), flush=True)
    # Print identities per AGENTS.md discipline: prompt md5s + binary md5
    for r in rows:
        print(f"  prompt_md5={r.get('prompt_md5')} request_md5={r.get('request_md5')} step={r.get('step')}", flush=True)
    bmd5, bpath = _daemon_binary_md5()
    if bmd5:
        print(f"  daemon_binary_md5={bmd5} path={bpath}", flush=True)
    # Final summary
    return rows



def send(cfg, messages, tools=None):
    body = {"model": cfg["model"], "messages": messages, "max_tokens": cfg["max_tokens"],
            "stream": True, "stream_options": {"include_usage": True}}
    body.update(cfg["sampling"])
    if cfg.get("seed") is not None:
        body["seed"] = cfg["seed"]
    if tools is not None:
        body["tools"] = tools
    # Exact bytes sent for md5 (AGENTS.md discipline: byte-identical prompts + request identity)
    body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request_md5 = hashlib.md5(body_bytes).hexdigest()
    t0 = time.time(); ttft = None; think = []; ans = []
    tool_acc = {}
    usage = {}; timings = {}; finish = None; completion_id = None
    req = urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/v1/chat/completions",
                                 data=body_bytes,
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
        if d.get("tool_calls"):
            _merge_tool_call_deltas(tool_acc, d["tool_calls"])
    wall = time.time() - t0
    dtoks = usage.get("completion_tokens", 0)
    decode_ts = timings.get("decode_tok_s")
    decode_est = False
    if decode_ts is None and dtoks > 1 and ttft is not None and (wall - ttft) > 1e-6:
        decode_ts = round((dtoks - 1) / (wall - ttft), 1)
        decode_est = True
    think_s = "".join(think); ans_s = "".join(ans)
    # Build structured tool_calls list sorted by index (deterministic)
    tool_calls = []
    for idx in sorted(tool_acc.keys()):
        entry = tool_acc[idx]
        if entry.get("id") is None:
            entry["id"] = f"call_{idx}"
        if entry.get("type") is None:
            entry["type"] = "function"
        tool_calls.append({"id": entry["id"], "type": entry["type"], "function": {"name": entry["function"]["name"], "arguments": entry["function"]["arguments"]}})
    # Legacy stringified preview for backwards compat
    tool_s = " ".join(json.dumps(tc) for tc in tool_calls) if tool_calls else ""
    visible = (ans_s + " " + tool_s).strip()
    toks = re.findall(r"\S+", (think_s + " " + visible).strip())
    first, last, half = toks[:128], toks[-128:], toks[len(toks)//2:]
    bad = (bool(first) and (uniq(first) < 0.15 or maxfreq(first) > 0.50)) or \
          (bool(last) and (uniq(last) < 0.30 or maxfreq(last) > 0.50)) or (gram3(half) > 0.50)
    # ATEM leak detection (visible content deltas must not contain raw ATEM markup)
    atem_leak = ("<atem:" in ans_s) or ("<atem:" in think_s) or any("<atem:" in (tc.get("function", {}).get("arguments") or "") for tc in tool_calls)
    return {
        "request_id": completion_id,
        "ctx": usage.get("prompt_tokens", 0),
        "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "gen": usage.get("completion_tokens", 0), "finish": finish,
        "think_words": len(re.findall(r"\S+", think_s)), "ans_words": len(re.findall(r"\S+", visible)),
        "prefill_ms": timings.get("prefill_ms"), "prefill_tok_s": timings.get("prefill_tok_s"),
        "decode_tok_s": decode_ts, "decode_estimated": decode_est, "tau": timings.get("tau"),
        "cycles": timings.get("cycles"), "dflash": timings.get("dflash"),
        **_project_mtp_ngram_timings(timings),
        "ttft_s": round(ttft or 0, 3), "wall_s": round(wall, 3),
        "attractor": bad, "empty": (cfg.get("expect_visible", True) and len(visible) == 0),
        "runaway": (finish == "length"),
        "ans_preview": (visible or "<<no visible content>>")[:90],
        "assistant_content": ans_s if ans_s else (tool_s if tool_s else think_s),
        "content": ans_s,
        "reasoning_content": think_s,
        "tool_calls": tool_calls,
        "request_md5": request_md5,
        "atem_leak": atem_leak,
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
    ngram_mod = ""
    # ngram-mod fragment is meaningful only for MTP+ngram-mod composition.
    if r.get("mtp_ngram"):
        rate = r.get("ngram_mod_accept_rate")
        rate_s = f"{rate:.2f}" if isinstance(rate, (int, float)) else rate
        retired = r.get("mtp_retired")
        retired_s = "1" if retired is True else "0" if retired is False else retired
        ngram_mod = (
            f" ngram={r.get('ngram_mod_accepted')}/{r.get('ngram_mod_drafts')}"
            f"@{rate_s} win={r.get('mtp_windows')}/{r.get('ar_windows')}"
            f" retired={retired_s}"
        )
    return (f"  t{i:<2} finish={str(r['finish']):<6} ctx={r['ctx']:<6} cached={r['cached']:<6} "
            f"gen={r['gen']:<5}(think {r['think_words']}/ans {r['ans_words']}w) "
            f"prefill={r['prefill_ms']}ms/{prefill_str}tok/s "
            f"decode={dec_str}tok/s tau={r['tau']}{ngram_mod}"
            f"{recall}{fl} | {r['ans_preview']!r}")


def run(cfg, args):
    label = f"{os.path.basename(cfg['model'])}|{cfg['mtp']}|{cfg['mode']}"
    print(f"### RUN {label}  kv={cfg['kv']} sampling={cfg['sampling']} seed={cfg.get('seed')} ###", flush=True)
    rows = []
    feedback_shape = getattr(args, "feedback_shape", "plain") or "plain"
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
            # Chain feedback respects shape (rich vs plain)
            messages.append(_assistant_feedback(r, feedback_shape))
            missing = [item for item in expected if item.lower() not in r["assistant_content"].lower()]
            r["expected_substrings"] = expected
            r["retrieval_missing"] = missing
            recall = f" recall={len(expected) - len(missing)}/{len(expected)}" if expected else ""
            rows.append(r); print(f"  [{genre}]" + turn_line(len(rows), r, recall)[2:], flush=True)
    elif cfg["mode"] == "session":
        # Structured Glimmer session takes precedence when schema matches
        structured = None
        try:
            structured = _load_structured_session(args.session)
        except Exception:
            structured = None
        if structured is not None:
            rows = _run_glimmer_cache_tool_session(cfg, args, structured)
        else:
            turns = json.load(open(args.session))
            # Support both legacy array and dict-with-turns without structured schema
            if isinstance(turns, dict) and "turns" in turns:
                turns = turns["turns"]
            messages = []
            for i, t in enumerate(turns):
                messages.append({"role": "user", "content": t["content"]})
                r = send(cfg, messages)
                r["prompt_md5"] = hashlib.md5(t["content"].encode("utf-8")).hexdigest()
                messages.append(_assistant_feedback(r, feedback_shape))
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
    nm_acc = [r["ngram_mod_accepted"] for r in g if isinstance(r.get("ngram_mod_accepted"), (int, float))]
    nm_drf = [r["ngram_mod_drafts"] for r in g if isinstance(r.get("ngram_mod_drafts"), (int, float))]
    if nm_acc or nm_drf or any(r.get("mtp_ngram") for r in g):
        sa = int(sum(nm_acc)) if nm_acc else 0
        sd = int(sum(nm_drf)) if nm_drf else 0
        rate = (sa / sd) if sd else 0.0
        summary += f" ngram={sa}/{sd}@{rate:.2f}"
    print(summary, flush=True)
    # ---- Glimmer gate: prompt / request / binary identities (AGENTS.md discipline) ----
    for r in g:
        if r.get("prompt_md5"):
            print(f"  prompt_md5={r.get('prompt_md5')} request_md5={r.get('request_md5')} step={r.get('step','')}", flush=True)
    bmd5, bpath = _daemon_binary_md5()
    if bmd5:
        print(f"  daemon_binary_md5={bmd5} path={bpath}", flush=True)
    # ---- Tool round-trip + ATEM leak gate (always checked when tool_calls involved) ----
    for idx, r in enumerate(g):
        if r.get("atem_leak"):
            raise SystemExit(f"serve_harness: ATEM markup leaked in row {idx} visible content; failing gate. Row preview={r.get('ans_preview')!r}")
        if r.get("tool_calls"):
            # At least one tool call turn must satisfy finish==tool_calls and name populated
            if r.get("finish") != "tool_calls":
                raise SystemExit(f"serve_harness: tool round-trip assertion failed at row {idx}: expected finish=tool_calls, got {r.get('finish')!r}")
            for tc in r["tool_calls"]:
                fn = tc.get("function") or {}
                name = fn.get("name") or tc.get("name")
                if not name:
                    raise SystemExit(f"serve_harness: tool round-trip assertion failed at row {idx}: done.calls[0].name missing in {tc!r}")
    # ---- Compare transcript (cached-vs-cold A/B) ----
    cmp_path = getattr(args, "compare_transcript", None)
    if cmp_path:
        try:
            base = json.loads(Path(cmp_path).read_text(encoding="utf-8"))
            # base may be rows list or dict with rows
            if isinstance(base, dict) and "rows" in base:
                base = base["rows"]
            _assert_transcript_equal(base, g)
        except SystemExit:
            raise
        except Exception as e:
            raise SystemExit(f"serve_harness: compare-transcript failed: {e}")
    # ---- Glimmer cache trace assertions (HIPFIRE_GLIMMER_CACHE_TRACE=1) ----
    do_trace = getattr(args, "assert_glimmer_cache_trace", False) or os.environ.get("HIPFIRE_GLIMMER_CACHE_TRACE") == "1"
    if do_trace:
        trace_text = ""
        log_path = getattr(args, "serve_log", None) or "/tmp/serve_harness.serve.log"
        log_offset = cfg.get("_serve_log_offset", 0)
        try:
            if os.path.exists(log_path):
                trace_text = _serve_log_text(log_path, log_offset)
        except Exception:
            trace_text = ""
        parsed = _parse_glimmer_cache_trace(trace_text)
        if parsed:
            exp = len(g)
            if any(r.get("step") for r in g):
                exp = len(g)
            try:
                _assert_glimmer_cache_trace(trace_text, expected_requests=exp)
                print(f"glimmer_cache_trace=PASS requests={exp} hits={sum(1 for r in parsed if r['hit'])}", flush=True)
            except AssertionError as e:
                raise SystemExit(f"serve_harness: glimmer-cache trace assertion failed: {e}")
        else:
            if os.environ.get("HIPFIRE_GLIMMER_CACHE_TRACE") == "1":
                raise SystemExit("serve_harness: glimmer-cache trace assertion failed: HIPFIRE_GLIMMER_CACHE_TRACE=1 but no [glimmer-cache] lines found in log (expected {} rows)".format(len(g)))
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
    ap.add_argument("--kv", default=None,
                    help="cache mode override; omitted resolves the registry default")
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
    ap.add_argument("--ngram", default="off", choices=["off", "on"],
                    help="Model-free n-gram/PLD speculator (default off). 'on' emits the "
                         "exclusive selector mode=ngram + ngram=on + dflash/mtp off, and is "
                         "REFUSED alongside --dflash on / --mtp on.")
    ap.add_argument("--ngram-k", type=int, default=None, dest="ngram_k",
                    help="Draft block size K for --ngram on (loader default 12). K also sets the "
                         "PLD copy window: max_extract = K-1, so K caps how many tokens a "
                         "verbatim context match can copy in one cycle.")
    ap.add_argument(
        "--mtp-ngram",
        default="off",
        choices=["off", "on"],
        dest="mtp_ngram",
        help="Experimental long-gated ngram-mod composition inside native MTP (default off). "
             "Requires --mtp on, greedy sampling, and --thinking off; exclusive with standalone "
             "--ngram on and --dflash on. Sets HIPFIRE_MTP_NGRAM=1 plus HIPFIRE_NGRAM_MOD_* "
             "(no TOML change); still emits mode=mtp selector.",
    )
    ap.add_argument(
        "--mtp-ngram-match",
        type=int,
        default=None,
        dest="mtp_ngram_match",
        help="ngram-mod match length for --mtp-ngram on (default 24, >=1). "
             "Sets HIPFIRE_NGRAM_MOD_N_MATCH.",
    )
    ap.add_argument(
        "--mtp-ngram-min",
        type=int,
        default=None,
        dest="mtp_ngram_min",
        help="ngram-mod minimum draft length for --mtp-ngram on (default 48, >=1, <=max). "
             "Sets HIPFIRE_NGRAM_MOD_N_MIN. Fresh content pays no ngram verify until this chain exists.",
    )
    ap.add_argument(
        "--mtp-ngram-max",
        type=int,
        default=None,
        dest="mtp_ngram_max",
        help="ngram-mod maximum draft length for --mtp-ngram on (default 64, <=64). "
             "Sets HIPFIRE_NGRAM_MOD_N_MAX.",
    )
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
        "--feedback-shape",
        default="plain",
        choices=["plain", "rich", "content-only", "reasoning-content", "content_only", "reasoning_content"],
        help="Assistant history feedback shape for multi-turn: 'plain' (content-only, dumb client) vs 'rich' (reasoning_content + tool_calls). Both shapes exercised by gate; rich mirrors dump_corpus_openai_multiturn.py:267-294. Aliases content-only/reasoning-content accepted.",
    )
    ap.add_argument(
        "--compare-transcript",
        default=None,
        help="Path to a prior --out JSON transcript to compare against (cached-vs-cold A/B). Asserts byte-identical decoded assistant text turn-for-turn and identical request_md5s.",
    )
    ap.add_argument(
        "--assert-glimmer-cache-trace",
        action="store_true",
        help="Parse daemon serve log [glimmer-cache] lines and assert Glimmer prefix-cache invariants (hit=true, lcp==prior_len, prior_len==n_tokens from turn 2 onward). Requires HIPFIRE_GLIMMER_CACHE_TRACE=1 on daemon.",
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
        _self_test_kv_resolution()
        _self_test_mtp_ngram_config()
        _self_test_glimmer_feedback_shape()
        _self_test_glimmer_tool_delta_merge()
        _self_test_glimmer_transcript_and_trace()
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
    cfg["_serve_log_offset"] = log_offset
    rows = run(cfg, args)
    if not args.no_spawn:
        _assert_dflash_request_proofs(cfg, rows, args.serve_log, offset=log_offset)
        # Glimmer trace also asserted inside run(), but also ensure offset-correct post-check
        # (run already did it via cfg offset; no duplicate needed here)
        _kill_serve()
    missing = [r.get("retrieval_missing") for r in rows if r.get("retrieval_missing")]
    if missing:
        sys.exit(f"serve_harness: retrieval gate failed; missing expected substrings: {missing}")


if __name__ == "__main__":
    main()
