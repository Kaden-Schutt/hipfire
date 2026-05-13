
//! Dump Qwen3.5 hipfire post-layer hidden states for PyTorch oracle comparison.
//!
//! Outputs:
//!   <out-prefix>.hidden.f32 = layer-major f32 [n_layers_dumped, seq_len, dim]
//!   <out-prefix>.logits.f32 = final logits f32 [vocab]
//!   <out-prefix>.final_norm_last.f32 = final RMSNorm output for the last token [dim]
//!   <out-prefix>.meta.json  = token/layer/layout metadata

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_arch_qwen35::speculative::HiddenStateRingBuffer;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use rdna_compute::{DType, GpuTensor};
    use serde_json::json;
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use std::path::{Path, PathBuf};

    #[derive(Debug)]
    struct Args {
        model: PathBuf,
        prompt: String,
        tokens: Option<Vec<u32>>,
        out_prefix: PathBuf,
        layers: Option<Vec<usize>>,
        kv_mode: String,
    }

    fn parse_csv_u32(s: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        s.split(',').filter(|p| !p.trim().is_empty()).map(|p| Ok(p.trim().parse::<u32>()?)).collect()
    }

    fn parse_csv_usize(s: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        s.split(',').filter(|p| !p.trim().is_empty()).map(|p| Ok(p.trim().parse::<usize>()?)).collect()
    }

    fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
        let argv: Vec<String> = std::env::args().collect();
        let mut model = None;
        let mut prompt = "The quick brown fox jumps over the lazy dog.".to_string();
        let mut tokens = None;
        let mut out_prefix = None;
        let mut layers = None;
        let mut kv_mode = "f32".to_string();
        let mut i = 1usize;
        while i < argv.len() {
            match argv[i].as_str() {
                "--model" => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
                "--prompt" => { prompt = argv[i + 1].clone(); i += 2; }
                "--tokens" => { tokens = Some(parse_csv_u32(&argv[i + 1])?); i += 2; }
                "--out-prefix" => { out_prefix = Some(PathBuf::from(&argv[i + 1])); i += 2; }
                "--layers" => {
                    let v = &argv[i + 1];
                    if v != "all" { layers = Some(parse_csv_usize(v)?); }
                    i += 2;
                }
                "--kv-mode" => { kv_mode = argv[i + 1].clone(); i += 2; }
                "-h" | "--help" => {
                    eprintln!("Usage: dump_qwen35_hidden --model <model.hfq> --out-prefix <prefix> [--prompt TEXT | --tokens 1,2,3] [--layers all|1,8,15] [--kv-mode f32|q8|asym3|asym4|asym2]");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown arg: {other}").into()),
            }
        }
        Ok(Args {
            model: model.ok_or("--model required")?,
            prompt,
            tokens,
            out_prefix: out_prefix.ok_or("--out-prefix required")?,
            layers,
            kv_mode,
        })
    }

    fn write_f32s(path: &Path, xs: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        let mut w = BufWriter::new(File::create(path)?);
        let bytes = unsafe {
            std::slice::from_raw_parts(xs.as_ptr() as *const u8, xs.len() * std::mem::size_of::<f32>())
        };
        w.write_all(bytes)?;
        Ok(())
    }

    fn alloc_hidden_rb(
        gpu: &mut rdna_compute::Gpu,
        layers: Vec<usize>,
        seq_len: usize,
        dim: usize,
    ) -> Result<HiddenStateRingBuffer, Box<dyn std::error::Error>> {
        let mut layer_bufs = Vec::<GpuTensor>::with_capacity(layers.len());
        let mut staging_bufs = Vec::<GpuTensor>::with_capacity(layers.len());
        for _ in &layers {
            layer_bufs.push(gpu.alloc_tensor(&[seq_len * dim], DType::F32)?);
            staging_bufs.push(gpu.alloc_tensor(&[dim], DType::F32)?);
        }
        Ok(HiddenStateRingBuffer {
            layer_bufs,
            extract_layers: layers,
            max_positions: seq_len,
            hidden_dim: dim,
            head: 0,
            written: 0,
            staging_bufs,
            max_batch: 1,
        })
    }

    let args = parse_args()?;
    let mut hfq = HfqFile::open(&args.model)?;
    let config = qwen35::config_from_hfq(&hfq).ok_or("read qwen35 config")?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .ok_or("tokenizer metadata missing or invalid")?;
    let tokens = args.tokens.clone().unwrap_or_else(|| tokenizer.encode(&args.prompt));
    if tokens.is_empty() { return Err("prompt/token list produced zero tokens".into()); }
    let layers: Vec<usize> = args.layers.clone().unwrap_or_else(|| (0..config.n_layers).collect());
    for &layer in &layers {
        if layer >= config.n_layers {
            return Err(format!("layer {layer} out of range for {} layers", config.n_layers).into());
        }
    }

    eprintln!("dump_qwen35_hidden: model={} layers={} dim={} seq={} kv_mode={}",
        args.model.display(), config.n_layers, config.dim, tokens.len(), args.kv_mode);
    eprintln!("dump_qwen35_hidden: tokens={tokens:?}");

    let mut gpu = rdna_compute::Gpu::init()?;
    eprintln!("dump_qwen35_hidden: gpu={}", gpu.arch);
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu)?;
    let kv_len = tokens.len() + 8;
    let mut kv_cache = match args.kv_mode.as_str() {
        "f32" => KvCache::new_gpu(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_len)?,
        "q8" => KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_len)?,
        "asym4" => KvCache::new_gpu_asym4(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_len)?,
        "asym3" => KvCache::new_gpu_asym3(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_len)?,
        "asym2" => KvCache::new_gpu_asym2(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_len)?,
        other => return Err(format!("unknown --kv-mode {other}").into()),
    };
    let mut dn_state = DeltaNetState::new_with_quant(&mut gpu, &config, qwen35::StateQuant::FP32)?;
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, kv_len)?;
    let mut hidden_rb = alloc_hidden_rb(&mut gpu, layers.clone(), tokens.len(), config.dim)?;

    for (pos, &tok) in tokens.iter().enumerate() {
        qwen35::forward_scratch_with_hidden(
            &mut gpu, &weights, &config, tok, pos,
            &mut kv_cache, &mut dn_state, &scratch, &mut hidden_rb,
        )?;
    }
    gpu.hip.device_synchronize()?;

    let hidden_path = args.out_prefix.with_extension("hidden.f32");
    let logits_path = args.out_prefix.with_extension("logits.f32");
    let final_norm_last_path = args.out_prefix.with_extension("final_norm_last.f32");
    let meta_path = args.out_prefix.with_extension("meta.json");

    let mut hidden_writer = BufWriter::new(File::create(&hidden_path)?);
    for t in &hidden_rb.layer_bufs {
        let host = gpu.download_f32(t)?;
        let bytes = unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * std::mem::size_of::<f32>())
        };
        hidden_writer.write_all(bytes)?;
    }
    hidden_writer.flush()?;

    let logits = gpu.download_f32(&scratch.logits)?;
    write_f32s(&logits_path, &logits)?;
    let final_norm_last = gpu.download_f32(&scratch.tmp)?;
    write_f32s(&final_norm_last_path, &final_norm_last)?;

    let meta = json!({
        "schema": "hipfire.qwen35.hidden_dump.v0",
        "model_path": args.model,
        "prompt": args.prompt,
        "tokens": tokens,
        "layers": layers,
        "n_layers": config.n_layers,
        "dim": config.dim,
        "seq_len": hidden_rb.written,
        "vocab_size": config.vocab_size,
        "kv_mode": args.kv_mode,
        "dn_state_quant": "fp32",
        "hidden_path": hidden_path,
        "logits_path": logits_path,
        "final_norm_last_path": final_norm_last_path,
        "hidden_layout": "layer_major_f32[layers,seq_len,dim]",
        "logits_layout": "f32[vocab]",
        "final_norm_last_layout": "f32[dim] for final token after output RMSNorm"
    });
    std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta)?)?;
    eprintln!("dump_qwen35_hidden: wrote {}", meta_path.display());
    Ok(())
}
