//! hipfire engine daemon — JSON lines over stdin/stdout.
//! The Bun CLI spawns this process and communicates via IPC.
//! Usage: daemon (reads JSON from stdin, writes JSON to stdout)
//!
//! Exactly one daemon runs at a time per machine — enforced by an exclusive
//! flock(2) on ~/.hipfire/daemon.pid. A second daemon invocation exits with
//! `FATAL: hipfire daemon already running (PID N)` before touching the GPU,
//! preventing orphan doubles from silently double-consuming VRAM.
//!
//! Protocol:
//!   → {"type":"load","model":"path.hfq","params":{"max_seq":4096}}
//!   ← {"type":"loaded","arch":"qwen3_5","dim":4096,"layers":32,"vocab":248320,"vl":true}
//!   → {"type":"generate","id":"r1","prompt":"Hello","temperature":0.3,"max_tokens":512}
//!   → {"type":"generate","id":"r1","prompt":"Describe this","image":"/path/to/img.png","temperature":0.3,"max_tokens":512}
//!   ← {"type":"token","id":"r1","text":"The"}
//!   ← {"type":"done","id":"r1","tokens":42,"tok_s":44.5}
//!   → {"type":"unload"}
//!   ← {"type":"unloaded"}

use engine::hfq::HfqFile;
use engine::llama;
use engine::qwen35;
use engine::qwen35::DeltaNetState;
use engine::qwen35_vl;
use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Instant;

/// Acquire a machine-wide exclusive lock on ~/.hipfire/daemon.pid via flock(2).
/// Ensures only one hipfire daemon runs at a time; a second instance exits
/// immediately with a clear error naming the running PID. The kernel releases
/// the flock automatically on process death (including SIGKILL), so no manual
/// cleanup is required — stale PID file contents are fine, the fd is what
/// holds the lock.
///
/// Returns the File handle; caller MUST keep it alive for the process
/// lifetime (dropping it closes the fd and releases the lock).
fn acquire_daemon_lock() -> std::fs::File {
    use std::io::{Read, Seek, Write};
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::io::AsRawFd;

    let home = std::env::var("HOME").expect("HOME environment variable not set");
    let hipfire_dir = std::path::PathBuf::from(home).join(".hipfire");
    std::fs::create_dir_all(&hipfire_dir).expect("failed to create ~/.hipfire");
    let pid_path = hipfire_dir.join("daemon.pid");

    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .open(&pid_path)
        .expect("failed to open ~/.hipfire/daemon.pid");

    let rc = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc != 0 {
        let mut existing = String::new();
        let _ = f.read_to_string(&mut existing);
        let pid = existing.trim();
        let pid_display = if pid.is_empty() { "<unknown>" } else { pid };
        let kill_arg = if pid.is_empty() { "<pid>" } else { pid };
        eprintln!(
            "FATAL: hipfire daemon already running (PID {}). Run `kill {}` and retry.",
            pid_display, kill_arg
        );
        std::process::exit(1);
    }

    // Got the lock. Truncate any stale content and write our PID so tooling
    // and the user-facing error above can both show a useful number.
    f.set_len(0).ok();
    f.seek(std::io::SeekFrom::Start(0)).ok();
    writeln!(f, "{}", std::process::id()).ok();
    f.flush().ok();
    f
}

const IMAGE_SIZE: usize = 448;
const IMAGE_PAD_ID: u32 = 248056;
const VISION_START_ID: u32 = 248053;
const VISION_END_ID: u32 = 248054;

struct LoadedModel {
    arch_id: u32,
    // Qwen3.5 state
    q35_config: Option<qwen35::Qwen35Config>,
    q35_weights: Option<qwen35::Qwen35Weights>,
    q35_scratch: Option<qwen35::Qwen35Scratch>,
    kv_cache: Option<llama::KvCache>,
    dn_state: Option<DeltaNetState>,
    // Qwen3 state
    llama_config: Option<llama::LlamaConfig>,
    llama_weights: Option<llama::LlamaWeights>,
    llama_scratch: Option<llama::ForwardScratch>,
    llama_kv: Option<llama::KvCache>,
    // Vision state (VL models only)
    vision_config: Option<qwen35_vl::VisionConfig>,
    vision_weights: Option<qwen35_vl::VisionWeights>,
    // Shared
    tokenizer: Option<engine::tokenizer::Tokenizer>,
    // Multi-turn conversation state
    seq_pos: usize,              // current position in KV cache / DeltaNet state
    max_seq: usize,              // KV cache capacity
    conversation_tokens: Vec<u32>, // full token history for repeat penalty
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // --precompile: compile all kernels for this GPU, write hash files, exit.
    // Used by scripts/install.sh and `hipfire update` so first `hipfire run`
    // isn't a 2-minute hipcc wait.
    //
    // Covers the current default path (mq4 weights + asym3 KV) plus the legacy
    // compat paths (hfq4, hfq6, q8 weights × asym3, q8 KV) so models from any
    // era of the registry start instantly.
    if args.iter().any(|a| a == "--precompile") {
        // Pre-create the expected precompiled-dir next to this binary so the
        // compiler's writeback path fires. Without this, Gpu::init probes for
        // an existing dir and silently disables writeback if it's missing —
        // meaning fresh installs would compile but never cache cross-invocation.
        if let Some(exe_dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())) {
            // Arch is unknown until Gpu::init; use a broad mkdir for the common arches
            // we support so the probe picks one up. The real arch check after init
            // will log the active dir.
            for arch in ["gfx1010", "gfx1013", "gfx1030", "gfx1031", "gfx1100", "gfx1101", "gfx1102", "gfx1151", "gfx1200", "gfx1201"] {
                let _ = std::fs::create_dir_all(exe_dir.join("kernels").join("compiled").join(arch));
            }
        }
        let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
        eprintln!("Pre-compiling kernels for {}...", gpu.arch);
        let mut errors = 0usize;
        for kv in &["asym3", "q8"] {
            for wq in &["mq4", "mq6", "hfq4", "hfq6", "q8"] {
                if let Err(e) = gpu.precompile_qwen35(wq, kv, 256) {
                    eprintln!("  {wq}/{kv}: {e}");
                    errors += 1;
                }
            }
        }
        if errors > 0 {
            eprintln!("Kernel precompilation finished with {errors} failure(s) — the missing kernels will JIT on first use.");
        } else {
            eprintln!("Kernel precompilation done.");
        }
        return;
    }

    // Machine-wide mutex — prevents orphan daemons from silently coexisting
    // (observed 2026-04-13: two daemons at 100% CPU survived pkill -f rounds
    // because they'd been reparented to PID 1 after their bun parent died).
    // Kept in a binding so the fd lives for the full process lifetime.
    let _daemon_lock = acquire_daemon_lock();

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    let mut model: Option<LoadedModel> = None;

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() { continue; }

        let msg: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(stdout, r#"{{"type":"error","message":"invalid JSON: {}"}}"#, e);
                let _ = stdout.flush();
                continue;
            }
        };

        let msg_type = msg.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match msg_type {
            "load" => {
                // Unload previous if any
                if let Some(m) = model.take() {
                    unload_model(m, &mut gpu);
                }

                let path = msg.get("model").and_then(|v| v.as_str()).unwrap_or("");
                let max_seq = msg.get("params").and_then(|p| p.get("max_seq")).and_then(|v| v.as_u64()).unwrap_or(4096) as usize;
                match load_model(path, max_seq, &mut gpu) {
                    Ok(m) => {
                        let arch = if m.arch_id == 5 { "qwen3_5" } else { "qwen3" };
                        let vl = m.vision_config.is_some();
                        let (dim, layers, vocab) = if let Some(ref c) = m.q35_config {
                            (c.dim, c.n_layers, c.vocab_size)
                        } else if let Some(ref c) = m.llama_config {
                            (c.dim, c.n_layers, c.vocab_size)
                        } else { (0, 0, 0) };
                        let _ = writeln!(stdout, r#"{{"type":"loaded","arch":"{}","dim":{},"layers":{},"vocab":{},"vl":{}}}"#, arch, dim, layers, vocab, vl);
                        model = Some(m);
                    }
                    Err(e) => {
                        let (vram_free, vram_total) = gpu.hip.get_vram_info().unwrap_or((0, 0));
                        let free_mb = vram_free / (1024 * 1024);
                        let total_mb = vram_total / (1024 * 1024);
                        let _ = writeln!(stdout, r#"{{"type":"error","message":"load failed: {}. GPU: {} ({} MB free / {} MB total)"}}"#, e, gpu.arch, free_mb, total_mb);
                    }
                }
                let _ = stdout.flush();
            }

            "generate" => {
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        let _ = writeln!(stdout, r#"{{"type":"error","message":"no model loaded"}}"#);
                        let _ = stdout.flush();
                        continue;
                    }
                };

                let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("0");
                let prompt = msg.get("prompt").and_then(|v| v.as_str()).unwrap_or("Hello");
                let system = msg.get("system").and_then(|v| v.as_str());
                let image = msg.get("image").and_then(|v| v.as_str());
                let temp = msg.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.3) as f32;
                let max_tokens = msg.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(512) as usize;
                let top_p = msg.get("top_p").and_then(|v| v.as_f64()).unwrap_or(0.8) as f32;
                let repeat_penalty = msg.get("repeat_penalty").and_then(|v| v.as_f64()).unwrap_or(1.3) as f32;
                let repeat_window = msg.get("repeat_window").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
                // Experimental: inject a nudge string at a specific generated-
                // token count. The nudge tokens get forward-fed through the KV
                // cache so the model "sees" them as part of its own trajectory,
                // and are emitted to stdout so the client stream includes them.
                // Used to test whether telling a thinking model "time's up"
                // gets it to close </think> and commit to an answer.
                //
                // GATED: off by default. The feature has a real UX hazard — if
                // the alert fires after </think> has already closed, the nudge
                // leaks into the visible answer. Only honor the params when the
                // operator has explicitly opted in via config
                // (`experimental_budget_alert: true` → HIPFIRE_EXPERIMENTAL_
                // BUDGET_ALERT=1 set by the CLI). Research use only; not a
                // stable contract.
                let experimental_ok = std::env::var("HIPFIRE_EXPERIMENTAL_BUDGET_ALERT").ok().as_deref() == Some("1");
                let budget_alert_at_tok = if experimental_ok {
                    msg.get("budget_alert_at_tok").and_then(|v| v.as_u64()).unwrap_or(0) as usize
                } else { 0 };
                let budget_alert_text = if experimental_ok {
                    msg.get("budget_alert_text").and_then(|v| v.as_str()).unwrap_or("").to_string()
                } else { String::new() };

                if image.is_some() && m.vision_config.is_some() {
                    generate_vl(m, &mut gpu, &mut stdout, id, prompt, system, image.unwrap(), temp, top_p, max_tokens, repeat_penalty, repeat_window);
                } else {
                    generate(m, &mut gpu, &mut stdout, id, prompt, system, temp, top_p, max_tokens, repeat_penalty, repeat_window, budget_alert_at_tok, &budget_alert_text);
                }
            }

            "reset" => {
                // Reset conversation state without unloading the model
                if let Some(ref mut m) = model {
                    m.seq_pos = 0;
                    m.conversation_tokens.clear();
                    // Zero DeltaNet recurrent state (Qwen3.5)
                    if let Some(ref dn) = m.dn_state {
                        for s in &dn.s_matrices {
                            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                        }
                        for s in &dn.s_scales {
                            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                        }
                        for s in &dn.conv_states {
                            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                        }
                    }
                    let _ = writeln!(stdout, r#"{{"type":"reset","seq_pos":0}}"#);
                } else {
                    let _ = writeln!(stdout, r#"{{"type":"error","message":"no model loaded"}}"#);
                }
                let _ = stdout.flush();
            }

            "unload" => {
                if let Some(m) = model.take() {
                    unload_model(m, &mut gpu);
                }
                let _ = writeln!(stdout, r#"{{"type":"unloaded"}}"#);
                let _ = stdout.flush();
            }

            "ping" => {
                let _ = writeln!(stdout, r#"{{"type":"pong"}}"#);
                let _ = stdout.flush();
            }

            "diag" => {
                let (vram_free, vram_total) = gpu.hip.get_vram_info().unwrap_or((0, 0));
                let hip_ver = gpu.hip.runtime_version().unwrap_or((0, 0));
                let has_model = model.is_some();
                let model_arch = model.as_ref().map(|m| if m.arch_id == 5 { "qwen3_5" } else { "qwen3" }).unwrap_or("none");
                // Count pre-compiled kernels
                let kernel_dir = std::env::current_exe().ok()
                    .and_then(|e| e.parent().map(|p| p.join("kernels").join("compiled").join(&gpu.arch)))
                    .filter(|p| p.is_dir());
                let (hsaco_count, hash_count) = kernel_dir.map(|d| {
                    let hsaco = std::fs::read_dir(&d).map(|r| r.filter(|e| e.as_ref().ok().map(|e| e.path().extension().map(|x| x == "hsaco").unwrap_or(false)).unwrap_or(false)).count()).unwrap_or(0);
                    let hash = std::fs::read_dir(&d).map(|r| r.filter(|e| e.as_ref().ok().map(|e| e.path().extension().map(|x| x == "hash").unwrap_or(false)).unwrap_or(false)).count()).unwrap_or(0);
                    (hsaco, hash)
                }).unwrap_or((0, 0));
                let _ = writeln!(stdout,
                    r#"{{"type":"diag","arch":"{}","hip_version":"{}.{}","vram_free_mb":{},"vram_total_mb":{},"model_loaded":{},"model_arch":"{}","kernels":{},"kernel_hashes":{}}}"#,
                    gpu.arch, hip_ver.0, hip_ver.1,
                    vram_free / (1024 * 1024), vram_total / (1024 * 1024),
                    has_model, model_arch, hsaco_count, hash_count
                );
                let _ = stdout.flush();
            }

            "bench_prefill" => {
                // Synthetic prefill benchmark — measures forward_prefill_batch on N
                // deterministic tokens from a zeroed state. Used by `hipfire bench`
                // to produce canonical pp128/pp512/pp1024 numbers that don't depend
                // on the user's prompt tokenizing to a round number.
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        let _ = writeln!(stdout, r#"{{"type":"error","message":"no model loaded"}}"#);
                        let _ = stdout.flush();
                        continue;
                    }
                };
                let n = msg.get("tokens").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
                // Guard max_seq — reserve 32 slots of headroom so a subsequent
                // generate request against the loaded model still has room.
                if n + 32 > m.max_seq {
                    let _ = writeln!(stdout,
                        r#"{{"type":"error","message":"bench_prefill tokens={} exceeds loaded max_seq={}"}}"#,
                        n, m.max_seq);
                    let _ = stdout.flush();
                    continue;
                }
                // Deterministic synthetic token IDs. Skip 0 (often <pad>) and the
                // low specials by offsetting, and wrap in a 1000-wide window so the
                // embedding lookup cost stays realistic rather than hitting one
                // cache-hot row repeatedly.
                let synthetic: Vec<u32> = (0..n as u32).map(|i| 10 + (i % 1000)).collect();

                // Reset state BEFORE timing so we're measuring cold prefill, not
                // prefill-on-top-of-prior-state.
                m.seq_pos = 0;
                m.conversation_tokens.clear();
                if let Some(ref dn) = m.dn_state {
                    for s in &dn.s_matrices { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
                    for s in &dn.s_scales { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
                    for s in &dn.conv_states { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
                }

                // Flush any residual GPU work so it doesn't bleed into the
                // measured interval, then time forward_prefill_batch + a
                // trailing device_synchronize so we capture actual GPU
                // completion (kernel launches are async by default).
                let _ = gpu.hip.device_synchronize();
                let t0 = Instant::now();
                let run_ok = if m.arch_id == 5 {
                    let config = m.q35_config.as_ref().unwrap();
                    let weights = m.q35_weights.as_ref().unwrap();
                    let scratch = m.q35_scratch.as_ref().unwrap();
                    let kv = m.kv_cache.as_mut().unwrap();
                    let dn = m.dn_state.as_mut().unwrap();
                    qwen35::forward_prefill_batch(&mut gpu, weights, config, &synthetic, 0, kv, dn, scratch).is_ok()
                } else {
                    let config = m.llama_config.as_ref().unwrap();
                    let weights = m.llama_weights.as_ref().unwrap();
                    let scratch = m.llama_scratch.as_ref().unwrap();
                    let kv = m.llama_kv.as_mut().unwrap();
                    let mut ok = true;
                    for (i, &tok) in synthetic.iter().enumerate() {
                        if llama::forward_scratch(&mut gpu, weights, config, tok, i, kv, scratch, 0.0, 1.0, 42, 0, 1.0).is_err() {
                            ok = false;
                            break;
                        }
                    }
                    ok
                };
                let _ = gpu.hip.device_synchronize();
                let elapsed = t0.elapsed().as_secs_f64();

                // Reset state AFTER measurement — we've written N KV slots and a
                // DeltaNet state that the next real request must not inherit.
                m.seq_pos = 0;
                m.conversation_tokens.clear();
                if let Some(ref dn) = m.dn_state {
                    for s in &dn.s_matrices { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
                    for s in &dn.s_scales { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
                    for s in &dn.conv_states { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
                }

                if run_ok {
                    let tok_s = if elapsed > 0.0 { n as f64 / elapsed } else { 0.0 };
                    let _ = writeln!(stdout,
                        r#"{{"type":"prefill_result","tokens":{},"ms":{:.2},"tok_s":{:.1}}}"#,
                        n, elapsed * 1000.0, tok_s);
                } else {
                    let _ = writeln!(stdout, r#"{{"type":"error","message":"bench_prefill forward failed"}}"#);
                }
                let _ = stdout.flush();
            }

            "profile" => {
                // Precompile kernels for common configurations so we have something to profile.
                // If a model is loaded its kernels are already compiled; this fills in the rest.
                // Cover all KV modes × weight formats × head_dims to catch all kernel variants.
                #[cfg(feature = "deltanet")]
                for kv in &["q8"] {
                    for wq in &["hfq4", "hfq6", "q8"] {
                        for hd in &[128usize, 256] {
                            let _ = gpu.precompile_qwen35(wq, kv, *hd);
                        }
                    }
                }
                let (cap, kernels) = gpu.profile();
                let kernels_json: Vec<String> = kernels.iter().map(|k| k.to_json()).collect();
                let _ = writeln!(stdout,
                    r#"{{"type":"profile","gpu":{},"kernels":[{}]}}"#,
                    cap.to_json(), kernels_json.join(",")
                );
                let _ = stdout.flush();
            }

            _ => {
                let _ = writeln!(stdout, r#"{{"type":"error","message":"unknown type: {}"}}"#, msg_type);
                let _ = stdout.flush();
            }
        }
    }
}

fn load_model(path: &str, max_seq: usize, gpu: &mut rdna_compute::Gpu) -> Result<LoadedModel, String> {
    let kv_mode = std::env::var("HIPFIRE_KV_MODE").unwrap_or_default();
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .ok_or("tokenizer not found")?;

    if hfq.arch_id == 5 {
        // Qwen3.5 DeltaNet
        let config = qwen35::config_from_hfq(&hfq).ok_or("failed to read Qwen3.5 config")?;

        // Detect VL model: check if vision config AND vision tensors are present
        // Text-only models may have vision config in metadata but no actual vision weights
        let vision_config = qwen35_vl::vision_config_from_hfq(&hfq);
        let has_vision_tensors = hfq.tensor_data("model.visual.patch_embed.proj.weight").is_some();
        let (vision_config, vision_weights) = if let Some(vc) = vision_config {
            if has_vision_tensors {
                let vw = qwen35_vl::load_vision_weights(&hfq, &vc, gpu).map_err(|e| format!("{e}"))?;
                eprintln!("  VL model: vision encoder (hidden={}, layers={})", vc.hidden_size, vc.num_layers);
                (Some(vc), Some(vw))
            } else {
                (None, None) // text-only model, no vision tensors
            }
        } else {
            (None, None)
        };

        let weights = qwen35::load_weights(&hfq, &config, gpu).map_err(|e| format!("{e}"))?;
        // KV cache modes (RotorQuant-style asymmetric: K rotated + V Q8):
        //   asym3 (default) — K at 3-bit rotated, V at Q8_0. 5.5× vs fp32.
        //                     Best quality/compression tradeoff — RotorQuant "planar3".
        //   asym4 — K at 4-bit rotated, V at Q8_0. 5.1× (slightly safer).
        //   asym2 — K at 2-bit rotated, V at Q8_0. 6.0× (loses rare-token tail).
        //   q8    — K+V both Q8_0. 3.76× (reference quality).
        //
        // Legacy "turbo{2,3,4}" aliases map to asym{2,3,4} for backward compat.
        let kv = match kv_mode.as_str() {
            "q8" => {
                eprintln!("  KV cache: Q8");
                llama::KvCache::new_gpu_q8(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq).map_err(|e| format!("{e}"))?
            }
            "asym4" | "turbo4" => {
                llama::KvCache::new_gpu_asym4(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq).map_err(|e| format!("{e}"))?
            }
            "asym2" | "turbo2" => {
                llama::KvCache::new_gpu_asym2(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq).map_err(|e| format!("{e}"))?
            }
            "asym3" | "turbo3" | "turbo" | "auto" | "" => {
                llama::KvCache::new_gpu_asym3(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq).map_err(|e| format!("{e}"))?
            }
            other => {
                eprintln!("  KV cache: unrecognized '{other}', defaulting to asym3");
                llama::KvCache::new_gpu_asym3(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq).map_err(|e| format!("{e}"))?
            }
        };
        let dn = DeltaNetState::new(gpu, &config).map_err(|e| format!("{e}"))?;
        let scratch = qwen35::Qwen35Scratch::new(gpu, &config, 128).map_err(|e| format!("{e}"))?;
        Ok(LoadedModel {
            arch_id: 5,
            q35_config: Some(config), q35_weights: Some(weights), q35_scratch: Some(scratch),
            kv_cache: Some(kv), dn_state: Some(dn),
            llama_config: None, llama_weights: None, llama_scratch: None, llama_kv: None,
            vision_config, vision_weights,
            tokenizer: Some(tokenizer),
            seq_pos: 0, max_seq, conversation_tokens: Vec::new(),
        })
    } else {
        // Qwen3 / LLaMA
        let config = engine::hfq::config_from_hfq(&hfq).ok_or("failed to read LLaMA config")?;
        let weights = engine::hfq::load_weights_hfq(&hfq, &config, gpu).map_err(|e| format!("{e}"))?;
        eprintln!("  KV cache: Q8");
        let kv = llama::KvCache::new_gpu_q8(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq).map_err(|e| format!("{e}"))?;
        let scratch = llama::ForwardScratch::new(gpu, &config).map_err(|e| format!("{e}"))?;
        Ok(LoadedModel {
            arch_id: hfq.arch_id,
            q35_config: None, q35_weights: None, q35_scratch: None,
            kv_cache: None, dn_state: None,
            llama_config: Some(config), llama_weights: Some(weights), llama_scratch: Some(scratch), llama_kv: Some(kv),
            vision_config: None, vision_weights: None,
            tokenizer: Some(tokenizer),
            seq_pos: 0, max_seq, conversation_tokens: Vec::new(),
        })
    }
}

fn unload_model(m: LoadedModel, gpu: &mut rdna_compute::Gpu) {
    // Free KV cache + DeltaNet state + scratch first (small fraction of VRAM).
    if let Some(kv) = m.kv_cache { kv.free_gpu(gpu); }
    if let Some(dn) = m.dn_state { dn.free_gpu(gpu); }
    if let Some(s) = m.q35_scratch { s.free_gpu(gpu); }
    if let Some(kv) = m.llama_kv { kv.free_gpu(gpu); }
    if let Some(s) = m.llama_scratch { s.free_gpu(gpu); }
    // Weights are the bulk of VRAM (~80%). Free them too so idle eviction
    // actually returns VRAM to the system, not just the cache.
    if let Some(w) = m.q35_weights { w.free_gpu(gpu); }
    if let Some(w) = m.llama_weights { w.free_gpu(gpu); }
    if let Some(w) = m.vision_weights { w.free_gpu(gpu); }
    gpu.drain_pool();
}

fn generate(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu, stdout: &mut std::io::Stdout, id: &str, prompt: &str, system_prompt: Option<&str>, temp: f32, top_p: f32, max_tokens: usize, repeat_penalty: f32, repeat_window: usize, budget_alert_at_tok: usize, budget_alert_text: &str) {
    // Auto-reset on multi-turn rollover. The estimate here is intentionally
    // rough (ignores system prompt, which is only prepended on seq_pos==0);
    // undercounting only means we reset slightly later, and the EXACT
    // post-build guard below is the hard guarantee against KV overrun.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let prompt_est = tokenizer.encode(prompt).len() + 20;
    if m.seq_pos + prompt_est + max_tokens > m.max_seq {
        eprintln!("[daemon] context full ({}/{}) — resetting conversation", m.seq_pos, m.max_seq);
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        // Zero DeltaNet state on reset
        if let Some(ref dn) = m.dn_state {
            for s in &dn.s_matrices { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
            for s in &dn.s_scales { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
            for s in &dn.conv_states { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
        }
    }

    let im_start = tokenizer.encode("<|im_start|>");
    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let user_tok = tokenizer.encode("user");
    let asst_tok = tokenizer.encode("assistant");
    let q_tokens = tokenizer.encode(prompt);

    let mut new_tokens = Vec::new();

    // System prompt: prepend on first turn only (seq_pos == 0)
    if m.seq_pos == 0 {
        if let Some(sys) = system_prompt {
            let sys_tok = tokenizer.encode("system");
            let sys_content = tokenizer.encode(sys);
            new_tokens.extend_from_slice(&im_start);
            new_tokens.extend_from_slice(&sys_tok);
            new_tokens.extend_from_slice(&nl);
            new_tokens.extend_from_slice(&sys_content);
            new_tokens.extend_from_slice(&im_end);
            new_tokens.extend_from_slice(&nl);
        }
    }

    // User turn
    new_tokens.extend_from_slice(&im_start);
    new_tokens.extend_from_slice(&user_tok);
    new_tokens.extend_from_slice(&nl);
    new_tokens.extend_from_slice(&q_tokens);
    new_tokens.extend_from_slice(&im_end);
    new_tokens.extend_from_slice(&nl);
    new_tokens.extend_from_slice(&im_start);
    new_tokens.extend_from_slice(&asst_tok);
    new_tokens.extend_from_slice(&nl);

    // EXACT KV-budget guard. `new_tokens.len()` is the precise prefill cost
    // (including system prompt on seq_pos==0 and all ChatML framing tokens);
    // plus max_tokens for generation, plus nl.len() reserved for the ChatML
    // trailer we run through forward AFTER an im_end termination (see the
    // `for &t in &nl` loops below — they increment seq_pos and call
    // forward_scratch at the new position). Without reserving those slots,
    // a request that exactly fills max_seq and terminates naturally on
    // im_end silently writes past the KV buffer.
    let trailer = nl.len();
    if m.seq_pos + new_tokens.len() + max_tokens + trailer > m.max_seq {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > max_seq={} — reload model with a larger max_seq"}}"#,
            id, m.seq_pos, new_tokens.len(), max_tokens, trailer, m.max_seq
        );
        let _ = stdout.flush();
        return;
    }

    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };
    let prefill_tokens = new_tokens.len();
    let t0 = Instant::now();

    if m.arch_id == 5 {
        // Qwen3.5 path — multi-turn: prefill only the NEW turn tokens,
        // continuing from m.seq_pos (KV cache + DeltaNet state are cumulative)
        let config = m.q35_config.as_ref().unwrap();
        let weights = m.q35_weights.as_ref().unwrap();
        let scratch = m.q35_scratch.as_ref().unwrap();
        let kv = m.kv_cache.as_mut().unwrap();
        let dn = m.dn_state.as_mut().unwrap();

        // Prefill this turn's tokens via the batched prefill entry point.
        // On gfx11+ for MQ4/HFQ4/MQ6/HFQ6 weights this hits the WMMA GEMM
        // fast path; other archs fall back to dp2 / FP16-packed / scalar
        // variants. The one sequential hotspot inside is the gated_delta_net
        // Q8 state update (N sequential per-token calls per LA layer, byte-
        // exact with decode to keep the quality gate green).
        //
        // Note: forward_prefill_batch launches HIP kernels asynchronously.
        // The t_prefill mark below lives AFTER the first sample_top_p, whose
        // D2H readback of tok0 forces a device sync — that's the point at
        // which the first token is actually ready to stream. Placing the
        // mark earlier captures CPU-dispatch time, which under-reports
        // prefill by a large factor (prefill_tok_s ~5–10× too optimistic).
        qwen35::forward_prefill_batch(
            gpu, weights, config, &new_tokens, m.seq_pos, kv, dn, scratch,
        ).unwrap();
        m.seq_pos += new_tokens.len();
        m.conversation_tokens.extend_from_slice(&new_tokens);

        // ngram scope for the repeat penalty: ONLY generated tokens (never the
        // prompt). Prior design included the user's prompt as an anti-loop
        // anchor, but that penalizes the very tokens we're asked to recall
        // (names, numbers, facts) under MQ4/MQ6 quantizations that are more
        // RP-sensitive than llama.cpp's Q4_K. First sample: empty scope (no
        // generated tokens yet); subsequent samples: generated-so-far only.
        let ngram_scope_start = m.conversation_tokens.len();

        // Generate. GPU-side sampling eliminates per-token logits download +
        // CPU softmax + CPU repeat penalty. Closes the 2× gap between raw
        // bench throughput and daemon throughput.
        //
        // Kernel signature reads `repeat_tokens[0..repeat_window]`, so we
        // only need to upload the tokens that will actually be read — no
        // need to clear the buffer between calls. The upload is on the same
        // stream as the sample kernel launch, so the copy and compute pipeline
        // naturally.
        let vocab_size = config.vocab_size;
        let mut rng_state: u32 = 0x13579BDFu32;
        let repeat_buf_cap = scratch.repeat_buf.buf.size() / 4;

        // First sample: use conversation so far as scope.
        let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
        let scope_start0 = ngram_scope.len().saturating_sub(repeat_buf_cap);
        let scope0 = &ngram_scope[scope_start0..];
        let bytes0: Vec<u8> = scope0.iter().flat_map(|t| t.to_ne_bytes()).collect();
        if !bytes0.is_empty() {
            gpu.hip.memcpy_htod(&scratch.repeat_buf.buf, &bytes0).unwrap();
        }
        let (tok0, rng0) = gpu.sample_top_p(
            &scratch.logits, &scratch.sample_buf, &scratch.repeat_buf,
            vocab_size, temp, top_p, rng_state, scope0.len(), repeat_penalty,
        ).unwrap();
        // First token is ready (sample_top_p's D2H forces GPU sync). This is
        // the user-observable "time to first token" boundary — prefill above,
        // decode loop below.
        let t_prefill = Instant::now();
        let mut next_token = tok0;
        rng_state = rng0;

        let mut generated = 0;
        let mut streamed_tokens: Vec<u32> = Vec::new();
        let mut emitted_bytes = 0usize;
        let mut alert_fired = false;

        // `while` instead of `for 0..max_tokens` so budget-alert injection
        // (which increments `generated` beyond the iteration count) can't
        // push generated past max_tokens: each loop start rechecks the cap.
        while generated < max_tokens {
            generated += 1;
            m.conversation_tokens.push(next_token);
            streamed_tokens.push(next_token);
            // Incremental UTF-8: only emit complete sequences
            let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
            let new_bytes = &all_bytes[emitted_bytes..];
            let vl = match std::str::from_utf8(new_bytes) { Ok(_) => new_bytes.len(), Err(e) => e.valid_up_to() };
            if vl > 0 {
                let text = std::str::from_utf8(&new_bytes[..vl]).unwrap();
                let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
                let _ = stdout.flush();
                emitted_bytes += vl;
            }

            // Write this token's K/V to the cache FIRST so the next turn
            // always starts from a fully-written context. Breaking before
            // forward_scratch used to leave a hole at the im_end/eos
            // position — the next turn then attended over zero-init K/V
            // at that slot.
            let pos = m.seq_pos + generated - 1;
            qwen35::forward_scratch(gpu, weights, config, next_token, pos, kv, dn, scratch).unwrap();

            if next_token == config.eos_token { break; }
            if im_end_token == Some(next_token) { break; }

            // Budget-alert injection: once we hit the configured token count,
            // splice the nudge text into the stream. Tokens are emitted to
            // stdout (so the client sees them) AND forward-fed through the KV
            // cache (so the model's next sample is conditioned on having
            // "said" them itself). Injected tokens count against `max_tokens`
            // — we never exceed the caller's requested budget — so we clip
            // the nudge if not enough room remains, and break out of the
            // outer loop if the budget is fully spent after injection.
            if !alert_fired && budget_alert_at_tok > 0 && generated >= budget_alert_at_tok && !budget_alert_text.is_empty() {
                alert_fired = true;
                // Only inject while the model is inside an open <think> block.
                // The whole point of the feature is to nudge the model's
                // reasoning; firing past </think> just graffities the visible
                // answer with a system-alert string. Check the raw decoded
                // text rather than token IDs since <think> tokenizes as a
                // multi-token sequence in Qwen3.5's vocab.
                let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
                let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
                let think_open_idx = raw_str.rfind("<think>");
                let think_close_idx = raw_str.rfind("</think>");
                let in_think = match (think_open_idx, think_close_idx) {
                    (Some(o), Some(c)) => o > c,
                    (Some(_), None) => true,
                    _ => false,
                };
                if !in_think {
                    let _ = writeln!(stdout, r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not inside an open <think> block"}}"#, id);
                    let _ = stdout.flush();
                    // Fall through — resample next token as normal
                    let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
                    let scope_start = ngram_scope.len().saturating_sub(repeat_buf_cap);
                    let scope = &ngram_scope[scope_start..];
                    let bytes: Vec<u8> = scope.iter().flat_map(|t| t.to_ne_bytes()).collect();
                    if !bytes.is_empty() {
                        gpu.hip.memcpy_htod(&scratch.repeat_buf.buf, &bytes).unwrap();
                    }
                    let (tok, rng) = gpu.sample_top_p(
                        &scratch.logits, &scratch.sample_buf, &scratch.repeat_buf,
                        vocab_size, temp, top_p, rng_state, scope.len(), repeat_penalty,
                    ).unwrap();
                    next_token = tok;
                    rng_state = rng;
                    continue;
                }
                let nudge_tokens = tokenizer.encode(budget_alert_text);
                let budget_left = max_tokens.saturating_sub(generated);
                let nudge_len = nudge_tokens.len().min(budget_left);
                // KV headroom check — don't run past max_seq. If we don't have
                // room for the clipped nudge, skip entirely rather than emit a
                // partial nudge that poisons the trajectory. Trailer reservation
                // matches the post-loop ChatML `\n` write so we don't slide past
                // the KV budget guard established at the top of `generate`.
                let need_kv = m.seq_pos + generated + nudge_len + (max_tokens - generated - nudge_len) + nl.len();
                if nudge_len > 0 && need_kv <= m.max_seq {
                    for &tok in &nudge_tokens[..nudge_len] {
                        m.conversation_tokens.push(tok);
                        streamed_tokens.push(tok);
                        // Emit the injected token's text to stdout so the client
                        // sees it as part of the stream (will be inside <think>
                        // if that's the current state, and get stripped client-
                        // side just like any other think token).
                        let all_bytes2 = tokenizer.decode_bytes(&streamed_tokens);
                        let new_bytes2 = &all_bytes2[emitted_bytes..];
                        let vl2 = match std::str::from_utf8(new_bytes2) { Ok(_) => new_bytes2.len(), Err(e) => e.valid_up_to() };
                        if vl2 > 0 {
                            let t = std::str::from_utf8(&new_bytes2[..vl2]).unwrap();
                            let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&t).unwrap_or_default());
                            let _ = stdout.flush();
                            emitted_bytes += vl2;
                        }
                        let pos2 = m.seq_pos + generated;
                        qwen35::forward_scratch(gpu, weights, config, tok, pos2, kv, dn, scratch).unwrap();
                        generated += 1;
                    }
                } else if nudge_len < nudge_tokens.len() {
                    let _ = writeln!(stdout, r#"{{"type":"info","id":"{}","message":"budget_alert clipped or skipped: nudge_len={} budget_left={}"}}"#, id, nudge_len, budget_left);
                    let _ = stdout.flush();
                } else {
                    let _ = writeln!(stdout, r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not enough KV headroom"}}"#, id);
                    let _ = stdout.flush();
                }
                // Respect max_tokens: if injection used the remainder, bail
                // before sampling another model token.
                if generated >= max_tokens { break; }
            }

            // Upload fresh repeat window (scope = generated tokens so far).
            let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
            let scope_start = ngram_scope.len().saturating_sub(repeat_buf_cap);
            let scope = &ngram_scope[scope_start..];
            let bytes: Vec<u8> = scope.iter().flat_map(|t| t.to_ne_bytes()).collect();
            if !bytes.is_empty() {
                gpu.hip.memcpy_htod(&scratch.repeat_buf.buf, &bytes).unwrap();
            }
            // GPU sample: reads scratch.logits (already on GPU), writes token+rng
            // to scratch.sample_buf. Blocks only on the 8-byte D2H readback.
            let (tok, rng) = gpu.sample_top_p(
                &scratch.logits, &scratch.sample_buf, &scratch.repeat_buf,
                vocab_size, temp, top_p, rng_state, scope.len(), repeat_penalty,
            ).unwrap();
            next_token = tok;
            rng_state = rng;
        }
        m.seq_pos += generated;

        // ChatML requires \n after <|im_end|>. Run it through forward so KV cache
        // and DeltaNet state stay in sync with seq_pos.
        if im_end_token == Some(*m.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
            for &t in &nl {
                qwen35::forward_scratch(gpu, weights, config, t, m.seq_pos, kv, dn, scratch).unwrap();
                m.seq_pos += 1;
                m.conversation_tokens.push(t);
            }
        }

        let t_end = Instant::now();
        let total_s = t_end.duration_since(t0).as_secs_f64();
        let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
        let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
        let tok_s = if total_s > 0.0 { generated as f64 / total_s } else { 0.0 };
        let prefill_tok_s = if prefill_s > 0.0 { prefill_tokens as f64 / prefill_s } else { 0.0 };
        let decode_tok_s = if decode_s > 0.0 { generated as f64 / decode_s } else { 0.0 };
        let _ = writeln!(
            stdout,
            r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}}}"#,
            id, generated, tok_s, prefill_tokens,
            prefill_s * 1000.0, prefill_tok_s, decode_tok_s, prefill_s * 1000.0
        );
        let _ = stdout.flush();
    } else {
        // Qwen3 / LLaMA path — multi-turn aware
        let config = m.llama_config.as_ref().unwrap();
        let weights = m.llama_weights.as_ref().unwrap();
        let scratch = m.llama_scratch.as_ref().unwrap();
        let kv = m.llama_kv.as_mut().unwrap();

        let mut rng_state = 42u32;
        for (i, &tok) in new_tokens.iter().enumerate() {
            let pos = m.seq_pos + i;
            let (_, rng) = llama::forward_scratch(gpu, weights, config, tok, pos, kv, scratch, temp, top_p, rng_state, 0, 1.0).unwrap();
            rng_state = rng;
        }
        let this_turn_prompt_len_llama = new_tokens.len();
        m.seq_pos += new_tokens.len();
        m.conversation_tokens.extend_from_slice(&new_tokens);
        let ngram_scope_start_llama = m.conversation_tokens.len() - this_turn_prompt_len_llama;

        let mut out_bytes = [0u8; 8];
        gpu.hip.memcpy_dtoh(&mut out_bytes, &scratch.sample_buf.buf).unwrap();
        let mut next_token = u32::from_ne_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);
        rng_state = u32::from_ne_bytes([out_bytes[4], out_bytes[5], out_bytes[6], out_bytes[7]]);
        // Prefill ends here: prompt is processed AND first token is ready (D2H
        // sync is the user-observable "time to first token" boundary). Decode
        // below measures the pure forward+sample steady-state.
        let t_prefill = Instant::now();

        let mut generated = 0;
        let mut streamed_tokens: Vec<u32> = Vec::new();
        let mut emitted_bytes = 0usize;

        for _ in 0..max_tokens {
            generated += 1;
            m.conversation_tokens.push(next_token);
            streamed_tokens.push(next_token);
            let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
            let new_bytes = &all_bytes[emitted_bytes..];
            let vl = match std::str::from_utf8(new_bytes) { Ok(_) => new_bytes.len(), Err(e) => e.valid_up_to() };
            if vl > 0 {
                let text = std::str::from_utf8(&new_bytes[..vl]).unwrap();
                let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
                let _ = stdout.flush();
                emitted_bytes += vl;
            }

            // Scope repeat_buf to this turn's prompt + generated tokens
            // (same logic as the Qwen3.5 path: prompt anchor + current turn).
            let rw = repeat_window.min(64);
            let scope_start = ngram_scope_start_llama.max(m.conversation_tokens.len().saturating_sub(rw));
            let hist_slice = &m.conversation_tokens[scope_start..];
            let hist_bytes: Vec<u8> = hist_slice.iter().flat_map(|t| t.to_ne_bytes()).collect();
            gpu.hip.memcpy_htod(&scratch.repeat_buf.buf, &hist_bytes).unwrap();

            // Write K/V for this token FIRST so the next turn's context is
            // always fully populated. The sampled next_token from this call
            // is discarded when we break on im_end/eos — wasteful by one
            // launch but avoids a KV cache gap at the terminator.
            let pos = m.seq_pos + generated - 1;
            let (tok, rng) = llama::forward_scratch(gpu, weights, config, next_token, pos, kv, scratch, temp, top_p, rng_state, hist_slice.len(), repeat_penalty).unwrap();

            if next_token == config.eos_token { break; }
            if im_end_token == Some(next_token) { break; }

            next_token = tok;
            rng_state = rng;
        }
        m.seq_pos += generated;

        // ChatML \n boundary — run through forward to keep KV cache in sync
        if im_end_token == Some(*m.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
            for &t in &nl {
                let (_, rng2) = llama::forward_scratch(gpu, weights, config, t, m.seq_pos, kv, scratch, temp, top_p, rng_state, 0, 1.0).unwrap();
                rng_state = rng2;
                m.seq_pos += 1;
                m.conversation_tokens.push(t);
            }
        }

        let t_end = Instant::now();
        let total_s = t_end.duration_since(t0).as_secs_f64();
        let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
        let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
        let tok_s = if total_s > 0.0 { generated as f64 / total_s } else { 0.0 };
        let prefill_tok_s = if prefill_s > 0.0 { prefill_tokens as f64 / prefill_s } else { 0.0 };
        let decode_tok_s = if decode_s > 0.0 { generated as f64 / decode_s } else { 0.0 };
        let _ = writeln!(
            stdout,
            r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}}}"#,
            id, generated, tok_s, prefill_tokens,
            prefill_s * 1000.0, prefill_tok_s, decode_tok_s, prefill_s * 1000.0
        );
        let _ = stdout.flush();
    }
}

fn generate_vl(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu, stdout: &mut std::io::Stdout, id: &str, prompt: &str, system_prompt: Option<&str>, image_path: &str, temp: f32, top_p: f32, max_tokens: usize, repeat_penalty: f32, repeat_window: usize) {
    // Capacity guard — VL prompts include vision tokens + text + ChatML framing
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let vision_config = m.vision_config.as_ref().unwrap();
    let n_patches = (IMAGE_SIZE / vision_config.patch_size) * (IMAGE_SIZE / vision_config.patch_size);
    let n_visual_tokens = n_patches / (vision_config.spatial_merge_size * vision_config.spatial_merge_size);
    let prompt_est = tokenizer.encode(prompt).len() + n_visual_tokens + 20; // text + vision + ChatML overhead
    if m.seq_pos + prompt_est + max_tokens > m.max_seq {
        eprintln!("[daemon/vl] context full ({}/{}) — resetting conversation", m.seq_pos, m.max_seq);
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        // Zero DeltaNet state on reset
        if let Some(ref dn) = m.dn_state {
            for s in &dn.s_matrices { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
            for s in &dn.s_scales { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
            for s in &dn.conv_states { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
        }
    }
    let config = m.q35_config.as_ref().unwrap();
    let vision_config = m.vision_config.as_ref().unwrap();
    let vision_weights = m.vision_weights.as_ref().unwrap();
    let weights = m.q35_weights.as_ref().unwrap();
    let scratch = m.q35_scratch.as_ref().unwrap();
    let kv = m.kv_cache.as_mut().unwrap();
    let dn = m.dn_state.as_mut().unwrap();

    // Load and preprocess image
    let pixels = engine::image::load_and_preprocess(Path::new(image_path), IMAGE_SIZE);
    let grid_h = IMAGE_SIZE / vision_config.patch_size;
    let grid_w = IMAGE_SIZE / vision_config.patch_size;
    let n_patches = grid_h * grid_w;
    let n_visual_tokens = n_patches / (vision_config.spatial_merge_size * vision_config.spatial_merge_size);

    // Extract patches and run vision encoder
    let patches = engine::image::extract_patches(
        &pixels, 3, IMAGE_SIZE, IMAGE_SIZE,
        vision_config.patch_size, vision_config.temporal_patch_size,
    );
    let visual_tokens = qwen35_vl::vision_forward(gpu, vision_weights, vision_config, &patches, grid_h, grid_w)
        .expect("vision forward failed");

    // Build VL prompt
    let im_start = tokenizer.encode("<|im_start|>");
    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let user_tok = tokenizer.encode("user");
    let asst_tok = tokenizer.encode("assistant");
    let q_tokens = tokenizer.encode(prompt);

    let mut prompt_tokens: Vec<u32> = Vec::new();

    // System prompt on first turn
    if m.seq_pos == 0 {
        if let Some(sys) = system_prompt {
            let sys_tok = tokenizer.encode("system");
            let sys_content = tokenizer.encode(sys);
            prompt_tokens.extend_from_slice(&im_start);
            prompt_tokens.extend_from_slice(&sys_tok);
            prompt_tokens.extend_from_slice(&nl);
            prompt_tokens.extend_from_slice(&sys_content);
            prompt_tokens.extend_from_slice(&im_end);
            prompt_tokens.extend_from_slice(&nl);
        }
    }

    prompt_tokens.extend_from_slice(&im_start);
    prompt_tokens.extend_from_slice(&user_tok);
    prompt_tokens.extend_from_slice(&nl);
    prompt_tokens.push(VISION_START_ID);
    for _ in 0..n_visual_tokens {
        prompt_tokens.push(IMAGE_PAD_ID);
    }
    prompt_tokens.push(VISION_END_ID);
    prompt_tokens.extend_from_slice(&nl);
    prompt_tokens.extend_from_slice(&q_tokens);
    prompt_tokens.extend_from_slice(&im_end);
    prompt_tokens.extend_from_slice(&nl);
    prompt_tokens.extend_from_slice(&im_start);
    prompt_tokens.extend_from_slice(&asst_tok);
    prompt_tokens.extend_from_slice(&nl);

    // EXACT KV-budget guard — same contract as `generate` (reserves nl.len()
    // for the ChatML trailer written post-generation on im_end termination).
    let trailer = nl.len();
    if m.seq_pos + prompt_tokens.len() + max_tokens + trailer > m.max_seq {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > max_seq={} — reload model with a larger max_seq"}}"#,
            id, m.seq_pos, prompt_tokens.len(), max_tokens, trailer, m.max_seq
        );
        let _ = stdout.flush();
        return;
    }

    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };
    let prefill_tokens = prompt_tokens.len();
    let t0 = Instant::now();

    // Prefill with vision token embedding for IMAGE_PAD positions
    let mut visual_idx = 0usize;
    for (i, &token) in prompt_tokens.iter().enumerate() {
        let pos = m.seq_pos + i;
        if token == IMAGE_PAD_ID && visual_idx < n_visual_tokens {
            let emb = &visual_tokens[visual_idx * config.dim..(visual_idx + 1) * config.dim];
            qwen35::forward_scratch_embed(gpu, weights, config, emb, pos, kv, dn, scratch)
                .expect("forward_scratch_embed failed");
            visual_idx += 1;
        } else {
            qwen35::forward_scratch(gpu, weights, config, token, pos, kv, dn, scratch)
                .expect("forward_scratch failed");
        }
    }
    m.seq_pos += prompt_tokens.len();
    m.conversation_tokens.extend_from_slice(&prompt_tokens);

    // Generate
    let mut logits = gpu.download_f32(&scratch.logits).unwrap();
    let mut next_token = llama::sample_top_p(&logits, temp, top_p);
    let t_prefill = Instant::now();
    let mut generated = 0;

    for _ in 0..max_tokens {
        generated += 1;
        m.conversation_tokens.push(next_token);
        let text = tokenizer.decode(&[next_token]);
        let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
        let _ = stdout.flush();

        if next_token == config.eos_token { break; }
        if im_end_token == Some(next_token) { break; }

        let pos = m.seq_pos + generated - 1;
        qwen35::forward_scratch(gpu, weights, config, next_token, pos, kv, dn, scratch).unwrap();
        logits = gpu.download_f32(&scratch.logits).unwrap();
        llama::apply_ngram_block(&mut logits, &m.conversation_tokens);
        llama::apply_repeat_penalty(&mut logits, &m.conversation_tokens, repeat_window, repeat_penalty);
        next_token = llama::sample_top_p(&logits, temp, top_p);
    }
    m.seq_pos += generated;

    // ChatML \n boundary — run through forward to keep KV cache + DeltaNet in sync
    if im_end_token == Some(*m.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
        for &t in &nl {
            qwen35::forward_scratch(gpu, weights, config, t, m.seq_pos, kv, dn, scratch).unwrap();
            m.seq_pos += 1;
            m.conversation_tokens.push(t);
        }
    }

    let t_end = Instant::now();
    let total_s = t_end.duration_since(t0).as_secs_f64();
    let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
    let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
    let tok_s = if total_s > 0.0 { generated as f64 / total_s } else { 0.0 };
    let prefill_tok_s = if prefill_s > 0.0 { prefill_tokens as f64 / prefill_s } else { 0.0 };
    let decode_tok_s = if decode_s > 0.0 { generated as f64 / decode_s } else { 0.0 };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}}}"#,
        id, generated, tok_s, prefill_tokens,
        prefill_s * 1000.0, prefill_tok_s, decode_tok_s, prefill_s * 1000.0
    );
    let _ = stdout.flush();
}
