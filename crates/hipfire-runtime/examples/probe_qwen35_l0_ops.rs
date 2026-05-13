//! Probe Qwen3.5 layer-0 MQ4 projection math for PyTorch/HFQ oracle checks.
//!
//! This is intentionally narrow: token 0, layer 0 linear-attention input norm,
//! MQ rotation, and the fused qkv/z/beta/alpha projection used by decode.

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_arch_qwen35::qwen35::{self, LayerWeights};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::EmbeddingFormat;
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
    }

    fn parse_csv_u32(s: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        s.split(',').filter(|p| !p.trim().is_empty()).map(|p| Ok(p.trim().parse::<u32>()?)).collect()
    }

    fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
        let argv: Vec<String> = std::env::args().collect();
        let mut model = None;
        let mut prompt = "The quick brown fox jumps over the lazy dog.".to_string();
        let mut tokens = None;
        let mut out_prefix = None;
        let mut i = 1usize;
        while i < argv.len() {
            match argv[i].as_str() {
                "--model" => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
                "--prompt" => { prompt = argv[i + 1].clone(); i += 2; }
                "--tokens" => { tokens = Some(parse_csv_u32(&argv[i + 1])?); i += 2; }
                "--out-prefix" => { out_prefix = Some(PathBuf::from(&argv[i + 1])); i += 2; }
                "-h" | "--help" => {
                    eprintln!("Usage: probe_qwen35_l0_ops --model MODEL.hfq --out-prefix PREFIX [--prompt TEXT | --tokens 1,2,3]");
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

    fn dump_tensor(
        gpu: &rdna_compute::Gpu,
        prefix: &Path,
        suffix: &str,
        tensor: &GpuTensor,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let path = prefix.with_extension(format!("{suffix}.f32"));
        let host = gpu.download_f32(tensor)?;
        write_f32s(&path, &host)?;
        Ok(path)
    }

    let args = parse_args()?;
    let mut hfq = HfqFile::open(&args.model)?;
    let config = qwen35::config_from_hfq(&hfq).ok_or("read qwen35 config")?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .ok_or("tokenizer metadata missing or invalid")?;
    let tokens = args.tokens.clone().unwrap_or_else(|| tokenizer.encode(&args.prompt));
    let token = *tokens.first().ok_or("prompt/token list produced zero tokens")?;

    let mut gpu = rdna_compute::Gpu::init()?;
    eprintln!("probe_qwen35_l0_ops: gpu={} token={token}", gpu.arch);
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu)?;

    let layer = match &weights.layers[0] {
        LayerWeights::DeltaNet(layer) => layer,
        _ => return Err("layer 0 is not a dense DeltaNet layer".into()),
    };
    if layer.wqkv.gpu_dtype != DType::MQ4G256
        || layer.wz.gpu_dtype != DType::MQ4G256
        || layer.w_beta.gpu_dtype != DType::MQ4G256
        || layer.w_alpha.gpu_dtype != DType::MQ4G256
    {
        return Err(format!(
            "expected MQ4 L0 qkv/z/beta/alpha, got {:?}/{:?}/{:?}/{:?}",
            layer.wqkv.gpu_dtype, layer.wz.gpu_dtype, layer.w_beta.gpu_dtype, layer.w_alpha.gpu_dtype,
        ).into());
    }

    let dim = config.dim;
    let x = gpu.alloc_tensor(&[dim], DType::F32)?;
    match &weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x, token, dim)?,
        _ => return Err("unsupported embedding format for probe".into()),
    }

    let x_norm = gpu.alloc_tensor(&[dim], DType::F32)?;
    let x_rot_split = gpu.alloc_tensor(&[dim], DType::F32)?;
    let x_rot_fused = gpu.alloc_tensor(&[dim], DType::F32)?;
    gpu.rmsnorm_f32(&x, &layer.attn_norm, &x_norm, config.norm_eps)?;
    gpu.rotate_x_mq(&x_norm, &x_rot_split, dim)?;
    gpu.fused_rmsnorm_rotate_mq(&x, &layer.attn_norm, &x_rot_fused, dim, config.norm_eps)?;

    let qkv_m = layer.wqkv.m;
    let z_m = layer.wz.m;
    let beta_m = layer.w_beta.m;
    let alpha_m = layer.w_alpha.m;
    let qkv_split = gpu.alloc_tensor(&[qkv_m], DType::F32)?;
    let z_split = gpu.alloc_tensor(&[z_m], DType::F32)?;
    let beta_split = gpu.alloc_tensor(&[beta_m], DType::F32)?;
    let alpha_split = gpu.alloc_tensor(&[alpha_m], DType::F32)?;
    let qkv_fused = gpu.alloc_tensor(&[qkv_m], DType::F32)?;
    let z_fused = gpu.alloc_tensor(&[z_m], DType::F32)?;
    let beta_fused = gpu.alloc_tensor(&[beta_m], DType::F32)?;
    let alpha_fused = gpu.alloc_tensor(&[alpha_m], DType::F32)?;

    gpu.gemv_hfq4g256(&layer.wqkv.buf, &x_rot_split, &qkv_split, qkv_m, layer.wqkv.k)?;
    gpu.gemv_hfq4g256(&layer.wz.buf, &x_rot_split, &z_split, z_m, layer.wz.k)?;
    gpu.gemv_hfq4g256(&layer.w_beta.buf, &x_rot_split, &beta_split, beta_m, layer.w_beta.k)?;
    gpu.gemv_hfq4g256(&layer.w_alpha.buf, &x_rot_split, &alpha_split, alpha_m, layer.w_alpha.k)?;
    gpu.fused_qkvza_hfq4g256(
        &layer.wqkv.buf, &layer.wz.buf, &layer.w_beta.buf, &layer.w_alpha.buf,
        &x_rot_fused,
        &qkv_fused, &z_fused, &beta_fused, &alpha_fused,
        qkv_m, z_m, beta_m, alpha_m, layer.wqkv.k,
    )?;
    gpu.hip.device_synchronize()?;

    let x_path = dump_tensor(&gpu, &args.out_prefix, "x", &x)?;
    let x_norm_path = dump_tensor(&gpu, &args.out_prefix, "x_norm", &x_norm)?;
    let x_rot_split_path = dump_tensor(&gpu, &args.out_prefix, "x_rot_split", &x_rot_split)?;
    let x_rot_fused_path = dump_tensor(&gpu, &args.out_prefix, "x_rot_fused", &x_rot_fused)?;
    let qkv_split_path = dump_tensor(&gpu, &args.out_prefix, "qkv_split", &qkv_split)?;
    let z_split_path = dump_tensor(&gpu, &args.out_prefix, "z_split", &z_split)?;
    let beta_split_path = dump_tensor(&gpu, &args.out_prefix, "beta_split", &beta_split)?;
    let alpha_split_path = dump_tensor(&gpu, &args.out_prefix, "alpha_split", &alpha_split)?;
    let qkv_fused_path = dump_tensor(&gpu, &args.out_prefix, "qkv_fused", &qkv_fused)?;
    let z_fused_path = dump_tensor(&gpu, &args.out_prefix, "z_fused", &z_fused)?;
    let beta_fused_path = dump_tensor(&gpu, &args.out_prefix, "beta_fused", &beta_fused)?;
    let alpha_fused_path = dump_tensor(&gpu, &args.out_prefix, "alpha_fused", &alpha_fused)?;

    let meta_path = args.out_prefix.with_extension("meta.json");
    let meta = json!({
        "schema": "hipfire.qwen35.l0_ops_probe.v0",
        "model_path": args.model,
        "prompt": args.prompt,
        "tokens": tokens,
        "token": token,
        "arch": gpu.arch,
        "dim": dim,
        "norm_eps": config.norm_eps,
        "tensor_names": {
            "wqkv": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "wz": "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            "w_beta": "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            "w_alpha": "model.language_model.layers.0.linear_attn.in_proj_a.weight"
        },
        "shapes": {
            "x": [dim],
            "x_norm": [dim],
            "x_rot": [dim],
            "qkv": [qkv_m],
            "z": [z_m],
            "beta": [beta_m],
            "alpha": [alpha_m]
        },
        "paths": {
            "x": x_path,
            "x_norm": x_norm_path,
            "x_rot_split": x_rot_split_path,
            "x_rot_fused": x_rot_fused_path,
            "qkv_split": qkv_split_path,
            "z_split": z_split_path,
            "beta_split": beta_split_path,
            "alpha_split": alpha_split_path,
            "qkv_fused": qkv_fused_path,
            "z_fused": z_fused_path,
            "beta_fused": beta_fused_path,
            "alpha_fused": alpha_fused_path
        }
    });
    std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta)?)?;
    eprintln!("probe_qwen35_l0_ops: wrote {}", meta_path.display());
    Ok(())
}
