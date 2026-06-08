// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// DFlash step 6b: extract the frozen target embed (== tied lm_head) from a
// Qwen3.6-27B .hfq into a single fp32 [vocab, dim] .bin. Qwen ties
// lm_head=embed_tokens, so this one table serves both the draft's input-embed
// (row gather) and output-lm_head (full GEMM). Gathers all `vocab` rows via the
// quant-format-matched embedding-lookup kernel (the runtime's own dequant).
//
//   dflash_extract_head <hfq> <out.bin>

use hipfire_arch_qwen35::qwen35::config_from_hfq;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{DType, Gpu};
use std::io::Write;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 3 {
        eprintln!("usage: dflash_extract_head <hfq> <out.bin>");
        std::process::exit(1);
    }
    let hfq_path = &argv[1];
    let out_path = &argv[2];

    let hfq = HfqFile::open(std::path::Path::new(hfq_path)).expect("open hfq");
    let cfg = config_from_hfq(&hfq).expect("config_from_hfq");
    let (vocab, dim) = (cfg.vocab_size, cfg.dim);
    eprintln!("hfq config: vocab={vocab} dim={dim}");

    // locate embed + any separate lm_head tensor
    let mut embed_name = None;
    let mut lmhead_name = None;
    for t in hfq.tensors() {
        let n = &t.name;
        let nl = n.to_lowercase();
        if nl.contains("embed_tokens") || nl == "model.embed_tokens.weight" {
            eprintln!("  embed candidate: {} qt={} shape={:?}", n, t.quant_type, t.shape);
            if embed_name.is_none() { embed_name = Some(n.clone()); }
        }
        if (nl.contains("lm_head") || nl.ends_with("output.weight") || nl == "lm_head.weight") && !nl.contains("norm") {
            eprintln!("  lm_head candidate: {} qt={} shape={:?}", n, t.quant_type, t.shape);
            lmhead_name = Some(n.clone());
        }
    }
    let embed_name = embed_name.unwrap_or_else(|| {
        // fallback: the [vocab, dim] shaped tensor
        hfq.tensors().iter().find(|t| t.shape == vec![vocab as u32, dim as u32]).map(|t| t.name.clone()).expect("no embed tensor")
    });
    eprintln!("using embed tensor: {embed_name}");
    eprintln!("separate lm_head: {lmhead_name:?} (None => tied to embed)");

    let (info, bytes) = hfq.tensor_data_vec(&embed_name).expect("embed data");
    let qt = info.quant_type;
    eprintln!("embed quant_type={qt} bytes={} ({} MB)", bytes.len(), bytes.len() / 1_000_000);

    let mut gpu = Gpu::init().expect("gpu init");

    // host-side dequant for F16(1)/F32(2); GPU gather for HFQ4G256(6)/Q8_0(3)
    let table_f32: Vec<f32> = match qt {
        6 | 3 => {
            let table = gpu.upload_raw(&bytes, &[bytes.len()]).expect("upload_raw embed");
            // token ids 0..vocab as i32 payload in an F32-width tensor
            let ids_t = gpu.alloc_tensor(&[vocab], DType::F32).unwrap();
            let id_bytes: Vec<u8> = (0..vocab as i32).flat_map(|i| i.to_le_bytes()).collect();
            gpu.hip.memcpy_htod(&ids_t.buf, &id_bytes).unwrap();
            let out = gpu.zeros(&[vocab, dim], DType::F32).unwrap();
            if qt == 6 {
                gpu.embedding_lookup_hfq4g256_batched(&table, &out, &ids_t, vocab, dim).expect("gather hfq4g256");
            } else {
                gpu.embedding_lookup_q8_batched(&table, &out, &ids_t, vocab, dim).expect("gather q8");
            }
            gpu.download_f32(&out).unwrap()
        }
        1 => bytes.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        other => { eprintln!("unsupported embed quant_type {other}"); std::process::exit(2); }
    };
    assert_eq!(table_f32.len(), vocab * dim, "embed table size mismatch");

    let mean = table_f32.iter().sum::<f32>() / table_f32.len() as f32;
    let absmax = table_f32.iter().fold(0f32, |a, &x| a.max(x.abs()));
    let finite = table_f32.iter().all(|x| x.is_finite());
    eprintln!("extracted embed/lm_head: [{vocab},{dim}] mean={mean:.5} absmax={absmax:.4} finite={finite}");

    // write [vocab, dim] f32 LE with a small header: magic + vocab + dim
    let mut f = std::io::BufWriter::new(std::fs::File::create(out_path).expect("create out"));
    f.write_all(b"DFHEAD\0\0").unwrap();
    f.write_all(&(vocab as u32).to_le_bytes()).unwrap();
    f.write_all(&(dim as u32).to_le_bytes()).unwrap();
    let raw: &[u8] = unsafe { std::slice::from_raw_parts(table_f32.as_ptr() as *const u8, table_f32.len() * 4) };
    f.write_all(raw).unwrap();
    f.flush().unwrap();
    eprintln!("wrote {out_path} ({} MB)", (16 + table_f32.len() * 4) / 1_000_000);
    println!("dflash_extract_head: OK (vocab={vocab} dim={dim} finite={finite})");
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1; let exp = (h >> 10) & 0x1f; let mant = h & 0x3ff;
    let bits = if exp == 0 {
        if mant == 0 { (sign as u32) << 31 } else {
            let mut e = -14i32; let mut m = mant as u32;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            m &= 0x3ff; ((sign as u32) << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13)
    } else {
        ((sign as u32) << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | ((mant as u32) << 13)
    };
    f32::from_bits(bits)
}
