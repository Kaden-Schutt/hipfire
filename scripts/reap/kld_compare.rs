// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
//
// Fast, dependency-free comparator for two DS4PPL01 logit dumps.
// Usage: kld_compare <reference.logits> <candidate.logits>

use std::convert::TryInto;
use std::env;
use std::fs::File;
use std::io::{self, BufReader, Read};

const MAGIC: &[u8; 8] = b"DS4PPL01";

struct Dump {
    reader: BufReader<File>,
    vocab: usize,
    rows: usize,
    bytes: Vec<u8>,
    logits: Vec<f32>,
}

impl Dump {
    fn open(path: &str) -> io::Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut magic = [0_u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{path}: bad DS4PPL01 magic"),
            ));
        }
        let vocab = read_u32(&mut reader)? as usize;
        let rows = read_u32(&mut reader)? as usize;
        Ok(Self {
            reader,
            vocab,
            rows,
            bytes: vec![0; vocab * 4],
            logits: vec![0.0; vocab],
        })
    }

    fn read_row(&mut self) -> io::Result<(u32, u32)> {
        let position = read_u32(&mut self.reader)?;
        let target = read_u32(&mut self.reader)?;
        self.reader.read_exact(&mut self.bytes)?;
        for (dst, src) in self.logits.iter_mut().zip(self.bytes.chunks_exact(4)) {
            *dst = f32::from_le_bytes(src.try_into().expect("four-byte chunk"));
        }
        Ok((position, target))
    }
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <reference.logits> <candidate.logits>", args[0]);
        std::process::exit(2);
    }

    let mut reference = Dump::open(&args[1])?;
    let mut candidate = Dump::open(&args[2])?;
    if reference.vocab != candidate.vocab || reference.rows != candidate.rows {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "dump mismatch: vocab {}/{} rows {}/{}",
                reference.vocab, candidate.vocab, reference.rows, candidate.rows
            ),
        ));
    }

    let mut kl_pq = 0.0_f64;
    let mut kl_qp = 0.0_f64;
    let mut top1 = 0_usize;

    for row in 0..reference.rows {
        let ref_identity = reference.read_row()?;
        let candidate_identity = candidate.read_row()?;
        if ref_identity != candidate_identity {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "row {row} identity mismatch: {:?}/{:?}",
                    ref_identity, candidate_identity
                ),
            ));
        }

        let (max_p, argmax_p) = max_and_argmax(&reference.logits);
        let (max_q, argmax_q) = max_and_argmax(&candidate.logits);
        top1 += usize::from(argmax_p == argmax_q);

        let mut sum_p = 0.0_f64;
        let mut sum_q = 0.0_f64;
        let mut weighted_p_delta = 0.0_f64;
        let mut weighted_q_delta = 0.0_f64;
        for (&p, &q) in reference.logits.iter().zip(&candidate.logits) {
            let p = p as f64;
            let q = q as f64;
            let exp_p = (p - max_p).exp();
            let exp_q = (q - max_q).exp();
            sum_p += exp_p;
            sum_q += exp_q;
            weighted_p_delta += exp_p * (p - q);
            weighted_q_delta += exp_q * (q - p);
        }
        let lse_p = max_p + sum_p.ln();
        let lse_q = max_q + sum_q.ln();
        kl_pq += (lse_q - lse_p) + weighted_p_delta / sum_p;
        kl_qp += (lse_p - lse_q) + weighted_q_delta / sum_q;

        if (row + 1) % 128 == 0 {
            eprintln!(
                "  {}/{} KL(f||p)={:.4}",
                row + 1,
                reference.rows,
                kl_pq / (row + 1) as f64
            );
        }
    }

    let rows = reference.rows as f64;
    println!("positions:               {}", reference.rows);
    println!("mean KL(full || pruned): {:.6} nats", kl_pq / rows);
    println!("mean KL(pruned || full): {:.6} nats", kl_qp / rows);
    println!(
        "top-1 argmax agreement:  {}/{} = {:.2}%",
        top1,
        reference.rows,
        100.0 * top1 as f64 / rows
    );
    Ok(())
}

fn max_and_argmax(values: &[f32]) -> (f64, usize) {
    let mut max = f32::NEG_INFINITY;
    let mut argmax = 0;
    for (index, &value) in values.iter().enumerate() {
        if value > max {
            max = value;
            argmax = index;
        }
    }
    (max as f64, argmax)
}
