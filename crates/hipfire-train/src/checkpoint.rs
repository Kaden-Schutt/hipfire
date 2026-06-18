// SPDX-License-Identifier: Apache-2.0
//! Checkpoint / resume for the PFlash drafter trainer.
//!
//! Two artifacts:
//!  - **Label cache** (`PFLB`): the target's per-chunk mid-layer + shallow block
//!    scores. Deterministic per (target, corpus, SEQ, BLOCK, mid) → keyed by a
//!    hash so a rerun skips the expensive 3B capture.
//!  - **Drafter checkpoint** (`PFDC`): drafter weights + AdamW moments + epoch,
//!    so a long run can be stopped and resumed (`--resume`).
//!
//! Simple little-endian binary; no external serialization deps.

use crate::drafter::Drafter;
use crate::optim::AdamW;
use rdna_compute::{Gpu, HipResult};
use std::io::{self, Read, Write};

fn wu32(w: &mut impl Write, x: u32) -> io::Result<()> { w.write_all(&x.to_le_bytes()) }
fn wu64(w: &mut impl Write, x: u64) -> io::Result<()> { w.write_all(&x.to_le_bytes()) }
fn wi32(w: &mut impl Write, x: i32) -> io::Result<()> { w.write_all(&x.to_le_bytes()) }
fn wvec(w: &mut impl Write, v: &[f32]) -> io::Result<()> {
    wu32(w, v.len() as u32)?;
    for &x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}
fn ru32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn ru64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn ri32(r: &mut impl Read) -> io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}
fn rvec(r: &mut impl Read) -> io::Result<Vec<f32>> {
    let n = ru32(r)? as usize;
    let mut buf = vec![0u8; n * 4];
    r.read_exact(&mut buf)?;
    Ok(buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}
fn rmagic(r: &mut impl Read, want: &[u8; 4]) -> io::Result<bool> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(&b == want)
}

// ── label cache ───────────────────────────────────────────────────────────
const LBL_MAGIC: &[u8; 4] = b"PFLB";

fn wvec_u32(w: &mut impl Write, v: &[u32]) -> io::Result<()> {
    wu32(w, v.len() as u32)?;
    for &x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}
fn rvec_u32(r: &mut impl Read) -> io::Result<Vec<u32>> {
    let n = ru32(r)? as usize;
    let mut buf = vec![0u8; n * 4];
    r.read_exact(&mut buf)?;
    Ok(buf.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Stores the token CHUNKS alongside the labels so a HIT reuses the exact corpus
/// the labels were computed from — decoupled from live repo files (the corpus is
/// globbed from docs/+crates/, which churn). Key is therefore geometry-only.
pub fn save_labels(
    path: &str,
    key: u64,
    chunks: &[Vec<u32>],
    label_mid: &[Vec<f32>],
    base_shallow: &[Vec<f32>],
) -> io::Result<()> {
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(LBL_MAGIC)?;
    wu32(&mut f, 2)?; // version 2: now carries chunks
    wu64(&mut f, key)?;
    wu32(&mut f, label_mid.len() as u32)?;
    for c in chunks {
        wvec_u32(&mut f, c)?;
    }
    for v in label_mid {
        wvec(&mut f, v)?;
    }
    for v in base_shallow {
        wvec(&mut f, v)?;
    }
    f.flush()
}

/// `Some((chunks, label_mid, base_shallow))` iff the file exists, is v2, and its
/// key matches (same target + geometry); otherwise `None` (recapture).
pub fn load_labels(path: &str, key: u64) -> Option<(Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let mut f = io::BufReader::new(std::fs::File::open(path).ok()?);
    if !rmagic(&mut f, LBL_MAGIC).ok()? {
        return None;
    }
    if ru32(&mut f).ok()? != 2 {
        return None; // old format → recapture
    }
    if ru64(&mut f).ok()? != key {
        return None;
    }
    let n = ru32(&mut f).ok()? as usize;
    let chunks: Vec<Vec<u32>> = (0..n).map(|_| rvec_u32(&mut f)).collect::<io::Result<_>>().ok()?;
    let label_mid: Vec<Vec<f32>> = (0..n).map(|_| rvec(&mut f)).collect::<io::Result<_>>().ok()?;
    let base_shallow: Vec<Vec<f32>> = (0..n).map(|_| rvec(&mut f)).collect::<io::Result<_>>().ok()?;
    Some((chunks, label_mid, base_shallow))
}

// ── drafter checkpoint ────────────────────────────────────────────────────
const CKPT_MAGIC: &[u8; 4] = b"PFDC";

pub fn save_drafter(
    gpu: &mut Gpu,
    path: &str,
    drafter: &Drafter,
    opt: &AdamW,
    epoch: u32,
) -> HipResult<()> {
    let params = drafter.params();
    let weights: Vec<Vec<f32>> =
        params.iter().map(|t| gpu.download_f32(t)).collect::<HipResult<_>>()?;
    let (m, v, t) = opt.save_state(gpu)?;
    let tmp = format!("{path}.tmp");
    let mut f = io::BufWriter::new(std::fs::File::create(&tmp).map_err(io_err)?);
    (|| -> io::Result<()> {
        f.write_all(CKPT_MAGIC)?;
        wu32(&mut f, 1)?;
        wu32(&mut f, epoch)?;
        wu32(&mut f, weights.len() as u32)?;
        for w in &weights {
            wvec(&mut f, w)?;
        }
        wi32(&mut f, t)?;
        for x in &m {
            wvec(&mut f, x)?;
        }
        for x in &v {
            wvec(&mut f, x)?;
        }
        f.flush()
    })()
    .map_err(io_err)?;
    std::fs::rename(&tmp, path).map_err(io_err)?; // atomic replace
    Ok(())
}

/// Load weights + AdamW state into an already-constructed drafter/optimizer
/// (same config). Returns the saved epoch, or `None` if the file is absent.
pub fn load_drafter(
    gpu: &mut Gpu,
    path: &str,
    drafter: &Drafter,
    opt: &mut AdamW,
) -> HipResult<Option<u32>> {
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let mut f = io::BufReader::new(file);
    let (epoch, weights, t, m, v) = read_ckpt(&mut f).map_err(io_err)?;

    // upload into the existing device buffers
    let params = drafter.params();
    let sizes = drafter.param_sizes();
    assert_eq!(weights.len(), params.len(), "checkpoint param count mismatch");
    for (i, w) in weights.iter().enumerate() {
        assert_eq!(w.len(), sizes[i], "checkpoint param[{i}] size mismatch");
        gpu.memcpy_htod_auto(&params[i].buf, bytemuck_f32(w))?;
    }
    opt.load_state(gpu, &m, &v, t)?;
    Ok(Some(epoch))
}

#[allow(clippy::type_complexity)]
fn read_ckpt<R: Read>(
    f: &mut R,
) -> io::Result<(u32, Vec<Vec<f32>>, i32, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    if !rmagic(f, CKPT_MAGIC)? {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad checkpoint magic"));
    }
    let _ver = ru32(f)?;
    let epoch = ru32(f)?;
    let np = ru32(f)? as usize;
    let weights: Vec<Vec<f32>> = (0..np).map(|_| rvec(f)).collect::<io::Result<_>>()?;
    let t = ri32(f)?;
    let m: Vec<Vec<f32>> = (0..np).map(|_| rvec(f)).collect::<io::Result<_>>()?;
    let v: Vec<Vec<f32>> = (0..np).map(|_| rvec(f)).collect::<io::Result<_>>()?;
    Ok((epoch, weights, t, m, v))
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn io_err(e: io::Error) -> rdna_compute::HipError {
    rdna_compute::HipError { code: u32::MAX, message: format!("checkpoint io: {e}") }
}
