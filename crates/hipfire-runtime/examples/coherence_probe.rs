//! coherence_probe — user-facing model behavior debugger.
//!
//! Spawns the daemon as a child process, drives a prompt through it,
//! parses the JSONL output stream, runs the `hipfire-detect` detectors
//! live, and prints a structured report. Strictly observational:
//! detectors never block, mutate, or interfere with generation.
//!
//! The daemon emits `{"type":"committed",...}` events alongside the
//! existing `{"type":"token","text":"..."}` events when
//! `HIPFIRE_EMIT_TOKEN_IDS=1` is in its environment — the probe sets
//! that on the daemon child it spawns.
//!
//! Usage:
//!     coherence_probe --model PATH --prompt-file PATH \
//!         [--system PATH] [--max-tokens N] [--temperature F] \
//!         [--report-json OUT.json] [--agentic] [--stall-tokens N] \
//!         [--detect-timing] [--no-strip-think] \
//!         [--max-seq N]
//!     coherence_probe --self-check
//!
//! Exit codes:
//!     0  every detector OK or only soft warnings
//!     1  one or more hard fails (or self-check miss)
//!     2  build / env / I/O error
//!
//! Example end-to-end:
//!     coherence_probe \
//!         --model ~/.hipfire/models/qwen3.6-27b.mq4 \
//!         --prompt-file benchmarks/prompts/lru_cache_pep8_strict.txt \
//!         --max-tokens 200 --temperature 0.0
//!
//! Self-check (no GPU needed):
//!     coherence_probe --self-check

use hipfire_detect::{
    attractor::{AttractorFirst128, AttractorLast128},
    eos_immediate::EosImmediate,
    ngram::{LoopGuardMirror, NgramDensity},
    report::{prompt_md5, Report, ReportHeader},
    self_check,
    special_leak::SpecialLeak,
    think::{ThinkEmpty, ThinkStall},
    timing::StepTimeSpike,
    toolcall::ToolcallShape,
    whitespace_only::WhitespaceOnly,
    DetectorBank, Event, Severity, Verdict,
};

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Instant;

#[derive(Debug, Default)]
struct Args {
    model: Option<String>,
    prompt_file: Option<String>,
    system: Option<String>,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    max_seq: Option<usize>,
    report_json: Option<String>,
    agentic: bool,
    stall_tokens: Option<usize>,
    detect_timing: bool,
    no_strip_think: bool,
    self_check: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => args.model = it.next(),
            "--prompt-file" => args.prompt_file = it.next(),
            "--system" => args.system = it.next(),
            "--max-tokens" => {
                args.max_tokens = it.next().and_then(|v| v.parse().ok());
            }
            "--temperature" => {
                args.temperature = it.next().and_then(|v| v.parse().ok());
            }
            "--max-seq" => {
                args.max_seq = it.next().and_then(|v| v.parse().ok());
            }
            "--report-json" => args.report_json = it.next(),
            "--agentic" => args.agentic = true,
            "--stall-tokens" => {
                args.stall_tokens = it.next().and_then(|v| v.parse().ok());
            }
            "--detect-timing" => args.detect_timing = true,
            "--no-strip-think" => args.no_strip_think = true,
            "--self-check" => args.self_check = true,
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {}", other)),
        }
    }
    Ok(args)
}

fn print_help() {
    eprintln!(
        "coherence_probe - user-facing model behavior debugger\n\n\
        Usage:\n  \
          coherence_probe --model PATH --prompt-file PATH [flags...]\n  \
          coherence_probe --self-check\n\n\
        Flags:\n  \
          --model PATH          model file (.mq3/.mq4/.hfq, etc.)\n  \
          --prompt-file PATH    user prompt file\n  \
          --system PATH         optional system-prompt file\n  \
          --max-tokens N        max generated tokens (default 200)\n  \
          --temperature F       sampling temperature (default 0.0)\n  \
          --max-seq N           daemon max_seq override (default 4096)\n  \
          --report-json OUT     also write the report as JSON\n  \
          --agentic             auto-engage tool-call shape detector\n  \
          --stall-tokens N      enable think_stall detector with budget N\n  \
          --detect-timing       enable per-token step-time spike detector\n  \
          --no-strip-think      ask daemon to leave <think> bytes intact\n  \
          --self-check          run synthetic+replay self-check (no GPU needed)\n"
    );
}

fn build_bank(args: &Args) -> DetectorBank {
    let mut bank = DetectorBank::new();
    bank.add(Box::new(AttractorFirst128::new()));
    bank.add(Box::new(AttractorLast128::new()));
    bank.add(Box::new(NgramDensity::new()));
    bank.add(Box::new(LoopGuardMirror::new()));
    bank.add(Box::new(ThinkEmpty::new()));
    if let Some(b) = args.stall_tokens {
        bank.add(Box::new(ThinkStall::new(b)));
    }
    bank.add(Box::new(SpecialLeak::new()));
    if args.agentic {
        bank.add(Box::new(ToolcallShape::new()));
    } else {
        // Auto-engage if the prompt itself looks tool-call-shaped.
        // The probe binary handles that decision and adds to the bank
        // earlier in `run` if the user prompt or system contains tool
        // schema text. See `decide_agentic` below.
    }
    bank.add(Box::new(EosImmediate::new()));
    bank.add(Box::new(WhitespaceOnly::new()));
    if args.detect_timing {
        bank.add(Box::new(StepTimeSpike::new()));
    }
    bank
}

fn decide_agentic(prompt: &str, system: Option<&str>) -> bool {
    let combined = format!("{}\n{}", system.unwrap_or(""), prompt);
    let s = combined.to_ascii_lowercase();
    // Heuristic: prompt mentions tool_call schema or function-call json.
    s.contains("<tool_call>")
        || (s.contains("\"name\"") && s.contains("\"arguments\""))
        || (s.contains("function") && s.contains("\"arguments\""))
}

fn read_text(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))
}

fn find_daemon_binary() -> Result<PathBuf, String> {
    // Prefer release; fall back to debug. Mirror the gate scripts'
    // discovery behaviour.
    let candidates = [
        "target/release/examples/daemon",
        "target/debug/examples/daemon",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Ok(p);
        }
    }
    Err("daemon binary not found; run `cargo build --release --example daemon --features deltanet` first".into())
}

fn print_live(name: &str, verdict: &Verdict, t_ms: u64, pos: Option<usize>) {
    let label = verdict.label();
    let detail = match verdict {
        Verdict::Ok => return, // never print OK live
        Verdict::Skip { .. } => return,
        Verdict::Fired { detail, .. } => detail.clone(),
    };
    let pos_str = pos.map(|p| format!(" tok={}", p)).unwrap_or_default();
    eprintln!(
        "[t={:.3}s{}] {:<5} {:<22} {}",
        t_ms as f64 / 1000.0,
        pos_str,
        label,
        name,
        detail
    );
}

struct DaemonChild {
    child: Child,
    /// `Option` so we can drop the write end on shutdown without
    /// destructuring the whole struct. Daemon's main `for line in stdin
    /// .lock().lines()` loop terminates on stdin EOF — without dropping
    /// our write end, the daemon keeps polling for the next command and
    /// `child.wait()` blocks forever.
    stdin: Option<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
}

impl DaemonChild {
    fn close_stdin(&mut self) {
        self.stdin = None;
    }
}

fn spawn_daemon(daemon: &PathBuf) -> Result<DaemonChild, String> {
    let mut cmd = Command::new(daemon);
    cmd.env("HIPFIRE_EMIT_TOKEN_IDS", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let mut child = cmd.spawn().map_err(|e| format!("spawn daemon: {}", e))?;
    let stdin = child.stdin.take().ok_or("daemon stdin")?;
    let stdout = BufReader::new(child.stdout.take().ok_or("daemon stdout")?);
    Ok(DaemonChild {
        child,
        stdin: Some(stdin),
        stdout,
    })
}

fn send(d: &mut DaemonChild, msg: &serde_json::Value) -> Result<(), String> {
    let stdin = d
        .stdin
        .as_mut()
        .ok_or("daemon stdin already closed")?;
    let line = serde_json::to_string(msg).map_err(|e| format!("encode: {}", e))?;
    writeln!(stdin, "{}", line).map_err(|e| format!("write daemon: {}", e))?;
    stdin
        .flush()
        .map_err(|e| format!("flush daemon: {}", e))?;
    Ok(())
}

fn recv_until<F>(d: &mut DaemonChild, mut visitor: F) -> Result<serde_json::Value, String>
where
    F: FnMut(&serde_json::Value),
{
    let mut line = String::new();
    loop {
        line.clear();
        let n = d.stdout.read_line(&mut line).map_err(|e| format!("read: {}", e))?;
        if n == 0 {
            return Err("daemon closed stdout unexpectedly".into());
        }
        let v: serde_json::Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[probe] non-JSON line from daemon: {} ({})", line.trim(), e);
                continue;
            }
        };
        visitor(&v);
        let ty = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
        // Caller decides which message terminates the wait by inspecting the
        // visitor's local state. Here we simply return on common terminators;
        // for finer control, callers can pre-filter.
        match ty {
            "loaded" | "unloaded" | "done" | "error" => return Ok(v),
            _ => {}
        }
    }
}

#[derive(Debug, Default)]
#[allow(dead_code)]
struct DoneStats {
    total_tokens: usize,
    total_visible_bytes: usize,
    wall_ms: u64,
    ttft_ms: u64,
    /// Daemon-reported authoritative timings from its `done` event. The
    /// probe's own `wall_ms` / `ttft_ms` are wall-clock and confused by
    /// stripped think tokens (TTFT becomes "first visible character",
    /// which on a thinking model is prefill + think_phase + </think>).
    /// The daemon timestamps real prefill end and real decode separately,
    /// so trust those for perf reporting.
    daemon_prefill_ms: f64,
    daemon_prefill_tok_s: f64,
    daemon_decode_tok_s: f64,
    daemon_ttft_ms: f64,
    daemon_tok_s: f64,
}

fn drive_generate(
    d: &mut DaemonChild,
    bank: &mut DetectorBank,
    args: &Args,
    prompt: &str,
    system: Option<&str>,
) -> Result<DoneStats, String> {
    let req_id = "probe-1";
    let mut req = serde_json::json!({
        "type": "generate",
        "id": req_id,
        "prompt": prompt,
        "temperature": args.temperature.unwrap_or(0.0),
        "max_tokens": args.max_tokens.unwrap_or(200),
    });
    if let Some(sys) = system {
        req.as_object_mut()
            .unwrap()
            .insert("system".to_string(), serde_json::Value::String(sys.to_string()));
    }
    send(d, &req)?;

    // Stream events until we see {"type":"done"} or {"type":"error"}.
    let t_start = Instant::now();
    let mut visible_bytes: usize = 0;
    let mut ttft_ms: Option<u64> = None;
    let mut last_pos: Option<usize> = None;
    let done_stats: DoneStats;

    loop {
        let mut line = String::new();
        let n = d
            .stdout
            .read_line(&mut line)
            .map_err(|e| format!("read: {}", e))?;
        if n == 0 {
            return Err("daemon closed stdout during generation".into());
        }
        let v: serde_json::Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ty = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
        match ty {
            "committed" => {
                let tok_id = v.get("tok_id").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                let pos = v.get("pos").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                let t_ms = v.get("t_ms").and_then(|x| x.as_u64()).unwrap_or(0);
                last_pos = Some(pos);
                let ev = Event::Committed { tok_id, pos, t_ms };
                let trans = bank.observe(&ev);
                for (n, vd) in trans {
                    print_live(n, &vd, t_ms, Some(pos));
                }
            }
            "token" => {
                let text = v.get("text").and_then(|x| x.as_str()).unwrap_or("");
                let synthetic = v
                    .get("synthetic")
                    .and_then(|x| x.as_bool())
                    .unwrap_or(false);
                let t_ms = t_start.elapsed().as_millis() as u64;
                if !synthetic {
                    visible_bytes += text.len();
                    if ttft_ms.is_none() {
                        ttft_ms = Some(t_ms);
                    }
                }
                let ev = Event::Token {
                    text,
                    t_ms,
                    synthetic,
                };
                let trans = bank.observe(&ev);
                for (n, vd) in trans {
                    print_live(n, &vd, t_ms, last_pos);
                }
            }
            "done" => {
                let total_tokens = v
                    .get("tokens")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(0) as usize;
                let wall_ms = t_start.elapsed().as_millis() as u64;
                let ttft = ttft_ms.unwrap_or(wall_ms);
                let ev = Event::Done {
                    total_tokens,
                    total_visible_bytes: visible_bytes,
                    wall_ms,
                    ttft_ms: ttft,
                };
                let trans = bank.observe(&ev);
                for (n, vd) in trans {
                    print_live(n, &vd, wall_ms, last_pos);
                }
                // Daemon-authoritative perf metrics from the done event.
                // Default to 0 if absent (older daemons / non-Qwen35 paths).
                let daemon_prefill_ms = v.get("prefill_ms").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let daemon_prefill_tok_s = v.get("prefill_tok_s").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let daemon_decode_tok_s = v.get("decode_tok_s").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let daemon_ttft_ms = v.get("ttft_ms").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let daemon_tok_s = v.get("tok_s").and_then(|x| x.as_f64()).unwrap_or(0.0);
                done_stats = DoneStats {
                    total_tokens,
                    total_visible_bytes: visible_bytes,
                    wall_ms,
                    ttft_ms: ttft,
                    daemon_prefill_ms,
                    daemon_prefill_tok_s,
                    daemon_decode_tok_s,
                    daemon_ttft_ms,
                    daemon_tok_s,
                };
                break;
            }
            "error" => {
                let msg = v.get("message").and_then(|x| x.as_str()).unwrap_or("?");
                return Err(format!("daemon error: {}", msg));
            }
            _ => {} // ignore other event types
        }
    }
    Ok(done_stats)
}

fn arch_host() -> (String, String) {
    let arch = std::env::var("HIPFIRE_BASELINE_ARCH").unwrap_or_else(|_| {
        // Best-effort: try amdgpu-arch, then KFD topology, then "unknown".
        if let Ok(out) = std::process::Command::new("amdgpu-arch").output() {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout);
                let line = s.lines().next().unwrap_or("").trim();
                if !line.is_empty() {
                    return line.to_string();
                }
            }
        }
        "unknown".to_string()
    });
    let host = std::env::var("HOSTNAME").unwrap_or_else(|_| {
        std::process::Command::new("hostname")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    });
    (arch, host)
}

fn run_self_check() -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();
    let r = self_check::run_full();

    // Phase A — synthetic payloads.
    let mut a_pass = 0;
    for (name, ok, detail) in &r.phase_a {
        if *ok {
            a_pass += 1;
        } else {
            errors.push(format!("Phase A miss: {} ({})", name, detail));
        }
    }
    eprintln!(
        "self-check Phase A: {} / {} detectors fired correctly",
        a_pass,
        r.phase_a.len()
    );

    // Phase B — captured-JSONL replay against shipped fixtures.
    let mut b_pass = 0;
    for (label, ok, detail) in &r.phase_b {
        if *ok {
            b_pass += 1;
            eprintln!("self-check Phase B: {} {}", label, detail);
        } else {
            errors.push(format!("Phase B miss: {} ({})", label, detail));
        }
    }
    eprintln!(
        "self-check Phase B: {} / {} fixtures replayed correctly",
        b_pass,
        r.phase_b.len()
    );

    if errors.is_empty() {
        eprintln!(
            "self-check passed: Phase A {} / {} synthetic + Phase B {} / {} replay",
            a_pass,
            r.phase_a.len(),
            b_pass,
            r.phase_b.len()
        );
        Ok(())
    } else {
        Err(errors)
    }
}

fn run() -> Result<i32, String> {
    let args = parse_args().map_err(|e| {
        print_help();
        e
    })?;

    if args.self_check {
        return match run_self_check() {
            Ok(()) => Ok(0),
            Err(errs) => {
                for e in errs {
                    eprintln!("{}", e);
                }
                Ok(2)
            }
        };
    }

    let model = args.model.clone().ok_or("--model required")?;
    let prompt_path = args.prompt_file.clone().ok_or("--prompt-file required")?;
    let prompt = read_text(&prompt_path)?;
    let system = match args.system.as_deref() {
        Some(p) => Some(read_text(p)?),
        None => None,
    };
    let agentic = args.agentic || decide_agentic(&prompt, system.as_deref());
    let mut effective_args = Args { agentic, ..args };
    effective_args.agentic = agentic;

    let mut bank = build_bank(&effective_args);
    if agentic && !effective_args.agentic {
        // (defensive — already added above; keep for clarity)
    }

    let prompt_label = format!(
        "{}{}",
        prompt_path,
        effective_args
            .system
            .as_deref()
            .map(|s| format!(" + {}", s))
            .unwrap_or_default()
    );
    let combined_for_md5 = format!(
        "{}\n----\n{}",
        system.as_deref().unwrap_or(""),
        prompt
    );
    let md5 = prompt_md5(combined_for_md5.as_bytes());

    let daemon = find_daemon_binary()?;
    let mut child = spawn_daemon(&daemon)?;

    // Load
    let max_seq = effective_args.max_seq.unwrap_or(4096);
    let load = serde_json::json!({
        "type": "load",
        "model": model,
        "params": { "max_seq": max_seq },
    });
    send(&mut child, &load)?;
    let loaded = recv_until(&mut child, |_| {})?;
    let ty = loaded.get("type").and_then(|x| x.as_str()).unwrap_or("");
    if ty != "loaded" {
        let _ = send(&mut child, &serde_json::json!({ "type": "unload" }));
        child.close_stdin();
        let _ = child.child.wait();
        return Err(format!("expected loaded, got {}", ty));
    }

    // Drive generate
    let stats = match drive_generate(
        &mut child,
        &mut bank,
        &effective_args,
        &prompt,
        system.as_deref(),
    ) {
        Ok(s) => s,
        Err(e) => {
            let _ = send(&mut child, &serde_json::json!({ "type": "unload" }));
            child.close_stdin();
            let _ = child.child.wait();
            return Err(e);
        }
    };

    // Unload + wait. Closing stdin AFTER receiving "unloaded" lets the
    // daemon's `for line in stdin.lock().lines()` loop terminate cleanly
    // on EOF; otherwise `child.wait()` deadlocks because the daemon
    // keeps polling for the next command.
    let _ = send(&mut child, &serde_json::json!({ "type": "unload" }));
    let _ = recv_until(&mut child, |_| {});
    child.close_stdin();
    let _ = child.child.wait();

    let finals = bank.finalize();
    let (arch, host) = arch_host();
    let tok_s = if stats.wall_ms > 0 {
        stats.total_tokens as f64 * 1000.0 / stats.wall_ms as f64
    } else {
        0.0
    };
    // Generation-only rate: subtract TTFT (which includes any optional
    // HIPFIRE_DPM_WARMUP_SECS pin the daemon performs after load) from wall
    // time. This is the apples-to-apples number against in-process bench tools
    // like bench_qwen35_mq4's `gen_tok_s`. Falls back to wall-clock tok_s if
    // we somehow saw a zero gen window (single-token request, error path).
    let gen_tok_s = if stats.wall_ms > stats.ttft_ms && stats.total_tokens > 0 {
        let gen_ms = stats.wall_ms - stats.ttft_ms;
        stats.total_tokens as f64 * 1000.0 / gen_ms as f64
    } else {
        tok_s
    };
    let header = ReportHeader {
        prompt_md5: md5,
        prompt_label,
        model,
        arch,
        host,
        total_tokens: stats.total_tokens,
        tok_s,
        gen_tok_s,
        ttft_ms: stats.ttft_ms,
        daemon_prefill_ms: stats.daemon_prefill_ms,
        daemon_prefill_tok_s: stats.daemon_prefill_tok_s,
        daemon_decode_tok_s: stats.daemon_decode_tok_s,
        daemon_ttft_ms: stats.daemon_ttft_ms,
        daemon_tok_s: stats.daemon_tok_s,
    };
    let report = Report::new(header, finals);

    // Markdown to stdout.
    println!("{}", report.to_markdown());

    // Optional JSON.
    if let Some(p) = &effective_args.report_json {
        std::fs::write(p, report.to_json()).map_err(|e| format!("write json: {}", e))?;
        eprintln!("[probe] json report: {}", p);
    }

    let exit = if report.hard_fails > 0 {
        1
    } else if report.rows.iter().any(|r| {
        matches!(
            r.verdict,
            Verdict::Fired {
                severity: Severity::Warn,
                ..
            }
        )
    }) {
        // Soft warns alone don't fail the exit code per plan.
        0
    } else {
        0
    };
    Ok(exit)
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("[probe] {}", e);
            std::process::exit(2);
        }
    }
}
