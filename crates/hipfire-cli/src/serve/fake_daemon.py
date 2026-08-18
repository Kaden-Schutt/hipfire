#!/usr/bin/env python3
import json, os, sys

state_epoch = 0
generate_count = 0
LAST_SCENARIO = ""
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requests.log")
MODEL_PATH = ""

def log_req(req):
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(req, separators=(",", ":")) + "\n")
    except Exception:
        pass

def out(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()

def echo_ids(req):
    return req.get("id"), req.get("attempt_id")

def eligible_from_model():
    # Task 15: "ineligible" in model path/name => retry_reset_eligible false.
    blob = (MODEL_PATH or "").lower()
    return "ineligible" not in blob

def scenario_from(req):
    model = str(req.get("model") or "")
    prompt = str(req.get("prompt") or "")
    messages = req.get("messages") or []
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str) and c:
                    prompt = c
                    break
    blob = (model + " " + prompt).lower()
    tags = (
        "t15-transient-once",
        "t15-transient-always",
        "t15-visible-token",
        "t15-visible-reasoning",
        "t15-commit-ready-error",
        "t15-class-malformed",
        "t15-class-validation",
        "t15-class-context",
        "t15-class-unsupported",
        "t15-class-internal",
        "t15-class-adaptive",
        "t15-class-mismatch",
        "t15-class-cancel",
        "t15-transient-not-retryable",
        "t15-mismatch-attempt",
        "t15-eof",
        "t15-invalid-json",
        "t15-stale-event",
        "t15-tool-then-transient",
        "t15-reset-fail-rolled",
        "t15-reset-fail-seq",
        "t15-reset-fail-epoch",
        "t15-reset-fail-attempt",
        "t11-premature-eof",
        "t11-capability-denial",
        "t11-dirty-markers",
        "t11-length-withhold",
        "t11-mixed-tool",
        "t11-two-tools",
        "t11-pure-tool",
        "t11-stop-text",
        "t11-usage",
    )
    for tag in tags:
        if tag in blob:
            return tag
    return "t11-stop-text"

def emit_correlated(ev, rid, aid):
    if rid is not None:
        ev["id"] = rid
    if aid is not None:
        ev["attempt_id"] = aid
    out(ev)

def emit_typed_error(rid, aid, message, cls="transient", retryable=True, rolled_back=False, force_aid=None):
    out({
        "type": "error",
        "id": rid,
        "message": message,
        "class": cls,
        "retryable": retryable,
        "rolled_back": rolled_back,
        "attempt_id": force_aid if force_aid is not None else (aid if aid is not None else 0),
    })

def wait_commit(rid, aid, allow_abort=False):
    while True:
        line = sys.stdin.readline()
        if not line:
            return None
        try:
            msg = json.loads(line)
        except Exception:
            continue
        log_req(msg)
        ty = msg.get("type")
        if ty == "commit":
            if msg.get("id") != rid or msg.get("attempt_id") != aid:
                emit_typed_error(rid, aid, "commit correlation mismatch", cls="internal", retryable=False)
                return "error"
            return "commit"
        if ty == "abort" and allow_abort:
            return "abort"
        if ty == "unload":
            out({"type": "unloaded"})
            sys.exit(0)

def success_stop(rid, aid, text="hello from fake daemon"):
    emit_correlated({"type": "token", "text": text}, rid, aid)
    emit_correlated({
        "type": "commit_ready",
        "finish_reason": "stop",
        "prompt_tokens": 3,
        "tokens": 4,
        "tok_s": 12.0,
    }, rid, aid)
    if wait_commit(rid, aid) != "commit":
        return
    emit_correlated({
        "type": "done",
        "finish_reason": "stop",
        "prompt_tokens": 3,
        "tokens": 4,
        "tok_s": 12.0,
    }, rid, aid)

def handle_generate(req):
    global generate_count, LAST_SCENARIO
    generate_count += 1
    rid, aid = echo_ids(req)
    scenario = scenario_from(req)
    LAST_SCENARIO = scenario

    if scenario == "t11-capability-denial":
        out({
            "type": "error",
            "id": rid,
            "message": "tools not supported by this endpoint capability",
            "class": "unsupported",
            "retryable": False,
            "rolled_back": True,
            "attempt_id": aid if aid is not None else 0,
        })
        return

    # All success / premature / t15 paths start with correlated v2 gen_start
    # except pure typed pre-start errors above.
    emit_correlated({
        "type": "gen_start",
        "contract_version": 2,
    }, rid, aid)

    # --- Task 15 scenarios ---
    if scenario == "t15-transient-once":
        if generate_count == 1:
            emit_typed_error(rid, aid, "transient prefill glitch")
            return
        success_stop(rid, aid, text="retry-recovered-content")
        return

    if scenario == "t15-transient-always":
        emit_typed_error(rid, aid, "persistent transient fault")
        return

    if scenario == "t15-visible-token":
        emit_correlated({"type": "token", "text": "visible-before-fail"}, rid, aid)
        emit_typed_error(rid, aid, "transient after visible token")
        return

    if scenario == "t15-visible-reasoning":
        emit_correlated({"type": "reasoning", "text": "think-before-fail"}, rid, aid)
        emit_typed_error(rid, aid, "transient after visible reasoning")
        return

    if scenario == "t15-commit-ready-error":
        emit_correlated({
            "type": "commit_ready",
            "finish_reason": "stop",
            "prompt_tokens": 1,
            "tokens": 1,
            "tok_s": 1.0,
        }, rid, aid)
        if wait_commit(rid, aid) != "commit":
            return
        emit_typed_error(rid, aid, "transient after commit_ready", cls="transient", retryable=True)
        return

    class_map = {
        "t15-class-malformed": ("malformed", False, "malformed payload"),
        "t15-class-validation": ("validation", False, "validation failed"),
        "t15-class-context": ("context_length", False, "context too long"),
        "t15-class-unsupported": ("unsupported", False, "unsupported op"),
        "t15-class-internal": ("internal", False, "internal fault"),
        "t15-class-adaptive": ("adaptive_poison", False, "adaptive poison"),
        "t15-class-mismatch": ("deterministic_mismatch", False, "deterministic mismatch"),
        "t15-class-cancel": ("cancel", False, "cancelled"),
        "t15-transient-not-retryable": ("transient", False, "transient but not retryable"),
    }
    if scenario in class_map:
        cls, retryable, msg = class_map[scenario]
        emit_typed_error(rid, aid, msg, cls=cls, retryable=retryable)
        return

    if scenario == "t15-mismatch-attempt":
        bad = (aid + 999) if isinstance(aid, int) else 999999
        emit_typed_error(rid, aid, "stale attempt error", force_aid=bad)
        return

    if scenario == "t15-eof":
        # Exit after gen_start with no done — engine sees Closed.
        sys.exit(0)

    if scenario == "t15-invalid-json":
        sys.stdout.write("{not-json\n")
        sys.stdout.flush()
        return

    if scenario == "t15-stale-event":
        # Correlated gen_start already emitted; now a stale-attempt token.
        stale_aid = (aid - 1) if isinstance(aid, int) and aid else 0
        emit_correlated({"type": "token", "text": "stale"}, rid, stale_aid)
        return

    if scenario == "t15-tool-then-transient":
        if generate_count == 1:
            emit_correlated({
                "type": "tool_calls",
                "calls": [{"name": "read_file", "arguments": {"path": "stale.rs"}}],
            }, rid, aid)
            emit_typed_error(rid, aid, "transient after buffered tools")
            return
        success_stop(rid, aid, text="fold-cleared-content")
        return

    # reset-fail: first generate is typed transient so server force-resets for attempt 2.
    if scenario.startswith("t15-reset-fail"):
        emit_typed_error(rid, aid, "transient before reset-fail")
        return

    if scenario == "t11-premature-eof":
        emit_correlated({"type": "token", "text": "partial-before-eof"}, rid, aid)
        sys.exit(0)

    if scenario == "t11-stop-text":
        success_stop(rid, aid)
        return

    if scenario == "t11-pure-tool":
        pure_calls = [{"name": "read_file", "arguments": {"path": "a.rs"}}]
        emit_correlated({
            "type": "commit_ready",
            "finish_reason": "tool_calls",
            "prompt_tokens": 2,
            "tokens": 1,
            "tok_s": 9.0,
            "calls": pure_calls,
        }, rid, aid)
        rc = wait_commit(rid, aid, allow_abort=True)
        if rc == "abort":
            emit_correlated({"type": "aborted", "reason": "client_cancelled"}, rid, aid)
            emit_correlated({"type": "done", "finish_reason": "aborted"}, rid, aid)
            return
        if rc != "commit":
            return
        emit_correlated({
            "type": "done",
            "finish_reason": "tool_calls",
            "prompt_tokens": 2,
            "tokens": 1,
            "tok_s": 9.0,
            "calls": pure_calls,
        }, rid, aid)
        return

    if scenario == "t11-mixed-tool":
        mixed_calls = [{"name": "read_file", "arguments": {"path": "mixed.rs"}}]
        emit_correlated({"type": "token", "text": "I'll look that up."}, rid, aid)
        emit_correlated({
            "type": "commit_ready",
            "finish_reason": "tool_calls",
            "prompt_tokens": 2,
            "tokens": 2,
            "tok_s": 8.5,
            "calls": mixed_calls,
        }, rid, aid)
        rc = wait_commit(rid, aid, allow_abort=True)
        if rc == "abort":
            emit_correlated({"type": "aborted", "reason": "client_cancelled"}, rid, aid)
            emit_correlated({"type": "done", "finish_reason": "aborted"}, rid, aid)
            return
        if rc != "commit":
            return
        emit_correlated({
            "type": "done",
            "finish_reason": "tool_calls",
            "prompt_tokens": 2,
            "tokens": 2,
            "tok_s": 8.5,
            "calls": mixed_calls,
        }, rid, aid)
        return

    if scenario == "t11-two-tools":
        two_calls = [
            {"name": "read_file", "arguments": {"path": "a.rs"}},
            {"name": "write_file", "arguments": {"path": "b.rs", "data": "x"}},
        ]
        emit_correlated({
            "type": "commit_ready",
            "finish_reason": "tool_calls",
            "prompt_tokens": 2,
            "tokens": 2,
            "tok_s": 8.0,
            "calls": two_calls,
        }, rid, aid)
        rc = wait_commit(rid, aid, allow_abort=True)
        if rc == "abort":
            emit_correlated({"type": "aborted", "reason": "client_cancelled"}, rid, aid)
            emit_correlated({"type": "done", "finish_reason": "aborted"}, rid, aid)
            return
        if rc != "commit":
            return
        emit_correlated({
            "type": "done",
            "finish_reason": "tool_calls",
            "prompt_tokens": 2,
            "tokens": 2,
            "tok_s": 8.0,
            "calls": two_calls,
        }, rid, aid)
        return

    if scenario == "t11-length-withhold":
        emit_correlated({"type": "token", "text": "partial-length"}, rid, aid)
        emit_correlated({
            "type": "tool_calls",
            "calls": [{"name": "read_file", "arguments": {"path": "x"}}],
        }, rid, aid)
        emit_correlated({
            "type": "commit_ready",
            "finish_reason": "length",
            "prompt_tokens": 2,
            "tokens": 3,
            "tok_s": 7.0,
        }, rid, aid)
        if wait_commit(rid, aid) != "commit":
            return
        emit_correlated({
            "type": "done",
            "finish_reason": "length",
            "prompt_tokens": 2,
            "tokens": 3,
            "tok_s": 7.0,
        }, rid, aid)
        return

    if scenario == "t11-dirty-markers":
        dirty = (
            '<tool_call>{"name":"evil","arguments":{}}</tool_call>'
            '<think>secret</think></think><|im_end|>'
        )
        emit_correlated({"type": "token", "text": dirty}, rid, aid)
        emit_correlated({
            "type": "commit_ready",
            "finish_reason": "stop",
            "prompt_tokens": 2,
            "tokens": 1,
            "tok_s": 6.0,
        }, rid, aid)
        if wait_commit(rid, aid) != "commit":
            return
        emit_correlated({
            "type": "done",
            "finish_reason": "stop",
            "prompt_tokens": 2,
            "tokens": 1,
            "tok_s": 6.0,
        }, rid, aid)
        return

    if scenario == "t11-usage":
        emit_correlated({"type": "token", "text": "usage-path"}, rid, aid)
        emit_correlated({
            "type": "commit_ready",
            "finish_reason": "stop",
            "prompt_tokens": 11,
            "tokens": 5,
            "cached_tokens": 2,
            "tok_s": 10.0,
        }, rid, aid)
        if wait_commit(rid, aid) != "commit":
            return
        emit_correlated({
            "type": "done",
            "finish_reason": "stop",
            "prompt_tokens": 11,
            "tokens": 5,
            "cached_tokens": 2,
            "tok_s": 10.0,
        }, rid, aid)
        return

    # Default stop text
    success_stop(rid, aid)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue
    log_req(req)
    ty = req.get("type")
    if ty == "configure":
        out({"type": "configured"})
    elif ty == "ping":
        out({"type": "pong"})
    elif ty == "load":
        MODEL_PATH = str(req.get("model") or "")
        out({
            "type": "loaded",
            "arch": "fake",
            "dim": 1,
            "layers": 1,
            "vocab": 1,
            "vl": False,
            # cache_capable true so only force_reset (retry attempt) issues reset
            "cache_capable": True,
            "retry_reset_eligible": eligible_from_model(),
            "max_seq": 4096,
        })
    elif ty == "reset":
        aid = req.get("attempt_id")
        sc = (LAST_SCENARIO or "") + " " + (MODEL_PATH or "")
        sc = sc.lower()
        if "t15-reset-fail-rolled" in sc:
            out({
                "type": "reset",
                "rolled_back": False,
                "state_epoch": state_epoch + 1,
                "seq_pos": 0,
                "conversation_len": 0,
                "attempt_id": aid,
                "retry_reset_eligible": eligible_from_model(),
            })
            continue
        if "t15-reset-fail-seq" in sc:
            out({
                "type": "reset",
                "rolled_back": True,
                "state_epoch": state_epoch + 1,
                "seq_pos": 1,
                "conversation_len": 0,
                "attempt_id": aid,
                "retry_reset_eligible": eligible_from_model(),
            })
            continue
        if "t15-reset-fail-epoch" in sc:
            out({
                "type": "reset",
                "rolled_back": True,
                "state_epoch": state_epoch if state_epoch > 0 else 0,
                "seq_pos": 0,
                "conversation_len": 0,
                "attempt_id": aid,
                "retry_reset_eligible": eligible_from_model(),
            })
            continue
        if "t15-reset-fail-attempt" in sc:
            out({
                "type": "reset",
                "rolled_back": True,
                "state_epoch": state_epoch + 1,
                "seq_pos": 0,
                "conversation_len": 0,
                "attempt_id": (aid + 1) if isinstance(aid, int) else 0,
                "retry_reset_eligible": eligible_from_model(),
            })
            continue
        state_epoch += 1
        out({
            "type": "reset",
            "rolled_back": True,
            "state_epoch": state_epoch,
            "seq_pos": 0,
            "conversation_len": 0,
            "attempt_id": aid,
            "retry_reset_eligible": eligible_from_model(),
        })
    elif ty == "generate":
        handle_generate(req)
    elif ty == "unload":
        out({"type": "unloaded"})
        sys.exit(0)
    elif ty == "commit":
        pass
    else:
        out({
            "type": "error",
            "message": f"unsupported op {ty}",
            "class": "validation",
            "retryable": False,
            "rolled_back": False,
            "id": "req-0",
            "attempt_id": 0,
        })
