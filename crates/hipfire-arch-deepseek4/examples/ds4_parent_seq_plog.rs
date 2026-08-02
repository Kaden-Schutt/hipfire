// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Sequential parent forward (token-by-token) for long-seq diagnostics.
use hipfire_arch_deepseek4::parent::head::{parent_logits_to_plog, PARENT_VOCAB};
use hipfire_arch_deepseek4::parent::inventory::ParentInventory;
use hipfire_arch_deepseek4::parent::model::{parent_model_forward, ParentModelScratch};
use hipfire_arch_deepseek4::parent::plog::PlogWriter;
use hipfire_arch_deepseek4::parent::weights::{ParentLoadPlan, ParentWeights};
use hipfire_arch_deepseek4::parent::Ds4ParentBackend;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::{DType, Gpu};
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut model = PathBuf::from("/mnt/scratch/models/DeepSeek-V4-Flash-0731");
    let mut tokens_path = PathBuf::from(
        "/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens_128.bin",
    );
    let mut plog = PathBuf::from(
        "/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/parent_128_seq.plog",
    );
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = PathBuf::from(args.next().ok_or("missing --model")?),
            "--token-ids" => {
                tokens_path = PathBuf::from(args.next().ok_or("missing --token-ids")?)
            }
            "--plog" => plog = PathBuf::from(args.next().ok_or("missing --plog")?),
            other => return Err(format!("unknown arg {other}")),
        }
    }
    let mut raw = Vec::new();
    File::open(&tokens_path)
        .map_err(|e| format!("open tokens: {e}"))?
        .read_to_end(&mut raw)
        .map_err(|e| format!("read tokens: {e}"))?;
    if raw.len() % 4 != 0 {
        return Err("token-ids not multiple of 4".into());
    }
    let token_ids: Vec<u32> = raw
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let n = token_ids.len();
    println!("sequential parent forward n={n}");

    let mut gpu = Gpu::init().map_err(|e| format!("Gpu::new: {e:?}"))?;
    let source = SafetensorsSource::open(&model).map_err(|e| format!("source: {e}"))?;
    let (backend, cfg) = Ds4ParentBackend::admit(&source, &mut gpu)?;
    let inv = ParentInventory::build(&source, &cfg)?;
    let plan = ParentLoadPlan {
        layers: 0..cfg.num_hidden_layers,
        load_experts: true,
    };
    let t0 = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    println!("load {:.1}s", t0.elapsed().as_secs_f64());

    let mut scratch = ParentModelScratch::new(&mut gpu, &cfg, 1)?;
    let logits_row = gpu
        .zeros(&[1, PARENT_VOCAB], DType::F32)
        .map_err(|e| format!("logits: {e:?}"))?;
    let mut all = vec![0.0f32; n * PARENT_VOCAB];
    let t1 = Instant::now();
    for (pos, &tid) in token_ids.iter().enumerate() {
        parent_model_forward(
            &mut gpu,
            backend,
            &weights,
            &cfg,
            &mut scratch,
            &[tid],
            pos,
            &logits_row,
        )?;
        let row = gpu
            .download_f32(&logits_row)
            .map_err(|e| format!("dl: {e:?}"))?;
        all[pos * PARENT_VOCAB..(pos + 1) * PARENT_VOCAB].copy_from_slice(&row);
        if pos % 8 == 0 || pos + 1 == n {
            println!("  pos {pos}/{n} elapsed {:.1}s", t1.elapsed().as_secs_f64());
        }
    }
    println!("forward {:.1}s", t1.elapsed().as_secs_f64());

    let logits_full = gpu
        .upload_f32(&all, &[n, PARENT_VOCAB])
        .map_err(|e| format!("upload: {e:?}"))?;
    let mut w = PlogWriter::create(&plog, n, PARENT_VOCAB)?;
    parent_logits_to_plog(&gpu, &logits_full, n, PARENT_VOCAB, &mut w)?;
    println!("wrote {}", plog.display());
    Ok(())
}
