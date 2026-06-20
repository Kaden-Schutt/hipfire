// SPDX-License-Identifier: Apache-2.0
// hipfire — video frame extraction via ffmpeg. See LICENSE / NOTICE.

//! Extract still frames from a video for the vision path.
//!
//! `decode_frames` shells out to `ffmpeg` to dump every coded frame as a
//! lossless PNG into a scratch dir, then uniformly samples up to `max_frames`
//! of them (preserving slice order) and returns the PNG bytes. The arch image
//! preprocessors (`image::load_from_memory`) consume those bytes unchanged.
//!
//! ## Quality preservation
//!
//! The sample dataset probes as `yuv420p`, `color_range=tv` (limited 16–235),
//! BT.601. VP9 is lossy and the grayscale was already clamped to limited range,
//! so the outermost ~16 levels are gone — unrecoverable. Extraction avoids
//! adding *more* loss:
//!
//! - **PNG** (lossless) — no generational re-compression (never JPEG).
//! - **limited→full range expansion** (`scale=in_range=…:out_range=full`) — the
//!   high-impact step. Without it luma stays 16–235 and the model's `/255`
//!   normalize squashes diagnostic contrast into ~6–92% of range. We probe the
//!   stream's `color_range` and only expand when the source is limited (`tv`),
//!   so an already-full (`pc`) clip is not double-expanded.
//! - **native resolution** — no ffmpeg downscale (`scale=iw:ih:…`); the model
//!   does the single 896² resize.
//! - **`-vsync 0`** — emit only coded frames so sampling picks distinct slices,
//!   not interpolated duplicates.
//! - no denoise/sharpen filters (they alter diagnostically meaningful texture).

use std::path::{Path, PathBuf};
use std::process::Command;

/// Video container extensions we route through `ffmpeg` instead of the still
/// `image` decoder.
const VIDEO_EXTS: &[&str] = &["webm", "mp4", "mkv", "mov", "avi", "m4v", "ogv", "wmv"];

/// True if `path`'s extension is a known video container (case-insensitive).
pub fn is_video(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let e = e.to_ascii_lowercase();
            VIDEO_EXTS.contains(&e.as_str())
        })
        .unwrap_or(false)
}

/// Uniformly pick `k` indices out of `n` items, inclusive of the endpoints,
/// preserving order and without duplicates. `n == 0` → empty; `k >= n` → all
/// `0..n`; `k == 1` → the middle item.
///
/// Endpoints-inclusive spacing: `round(i * (n-1) / (k-1))` for `i in 0..k`.
pub fn sample_indices(n: usize, k: usize) -> Vec<usize> {
    if n == 0 || k == 0 {
        return Vec::new();
    }
    if k >= n {
        return (0..n).collect();
    }
    if k == 1 {
        return vec![n / 2];
    }
    let mut out = Vec::with_capacity(k);
    let span = (n - 1) as f64;
    let steps = (k - 1) as f64;
    let mut last: Option<usize> = None;
    for i in 0..k {
        let idx = (i as f64 * span / steps).round() as usize;
        // Guard against rounding collisions (can't happen for k < n with this
        // formula, but keep the invariant explicit + duplicate-free).
        if last != Some(idx) {
            out.push(idx);
            last = Some(idx);
        }
    }
    out
}

/// Decode a video into up to `max_frames` PNG frames (uniformly sampled across
/// the clip, slice order preserved). Returns one `Vec<u8>` of PNG bytes per
/// frame. `max_frames == 0` is treated as "all frames".
///
/// Requires `ffmpeg` on `PATH` (and `ffprobe` for range detection; absence of
/// `ffprobe` falls back to limited-range expansion, the common VP9 case).
pub fn decode_frames(path: &Path, max_frames: usize) -> Result<Vec<Vec<u8>>, String> {
    if !path.exists() {
        return Err(format!(
            "hipfire-media: video not found: {}",
            path.display()
        ));
    }
    ensure_ffmpeg()?;

    let scratch = ScratchDir::new()?;
    let range = probe_color_range(path);
    let pattern = scratch.dir.join("f_%05d.png");

    // Extract every coded frame as lossless PNG at native resolution, expanding
    // limited→full luma range when the source is limited.
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i"])
        .arg(path)
        .args([
            "-vsync",
            "0",
            "-vf",
            &scale_filter(range),
            "-pix_fmt",
            "rgb24",
        ])
        .arg(&pattern)
        .output()
        .map_err(|e| format!("hipfire-media: spawn ffmpeg: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "hipfire-media: ffmpeg failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }

    // Collect frames in lexical (== capture) order — %05d zero-pads so lexical
    // sort is numeric.
    let mut files: Vec<PathBuf> = std::fs::read_dir(&scratch.dir)
        .map_err(|e| format!("hipfire-media: read scratch dir: {e}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("png"))
        .collect();
    files.sort();

    if files.is_empty() {
        return Err(format!(
            "hipfire-media: ffmpeg produced no frames from {}",
            path.display()
        ));
    }

    let keep = if max_frames == 0 {
        (0..files.len()).collect()
    } else {
        sample_indices(files.len(), max_frames)
    };

    let mut frames = Vec::with_capacity(keep.len());
    for i in keep {
        let bytes = std::fs::read(&files[i])
            .map_err(|e| format!("hipfire-media: read frame {}: {e}", files[i].display()))?;
        frames.push(bytes);
    }
    Ok(frames)
}

/// `color_range` reported by the first video stream, lowercased (e.g. `"tv"`,
/// `"pc"`), or `None` if `ffprobe` is unavailable/says unknown.
fn probe_color_range(path: &Path) -> Option<String> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=color_range",
            "-of",
            "default=nk=1:nw=1",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let v = String::from_utf8_lossy(&out.stdout)
        .trim()
        .to_ascii_lowercase();
    if v.is_empty() || v == "unknown" {
        None
    } else {
        Some(v)
    }
}

/// Build the `-vf` scale filter: keep native size (`iw:ih`), and expand the luma
/// range to full unless the source is already full (`pc`/`jpeg`). Unknown range
/// defaults to limited→full (the common VP9/`tv` case).
fn scale_filter(range: Option<String>) -> String {
    let in_range = match range.as_deref() {
        Some("pc") | Some("jpeg") | Some("full") => "full",
        _ => "tv",
    };
    format!("scale=iw:ih:in_range={in_range}:out_range=full")
}

fn ensure_ffmpeg() -> Result<(), String> {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-version"])
        .output()
        .map(|_| ())
        .map_err(|_| {
            "hipfire-media: `ffmpeg` not found on PATH — required to decode video frames"
                .to_string()
        })
}

/// A unique scratch directory removed on drop.
struct ScratchDir {
    dir: PathBuf,
}

impl ScratchDir {
    fn new() -> Result<Self, String> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir =
            std::env::temp_dir().join(format!("hipfire-media-{}-{}", std::process::id(), nanos));
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("hipfire-media: create scratch dir {}: {e}", dir.display()))?;
        Ok(Self { dir })
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn is_video_matches_known_containers() {
        assert!(is_video(Path::new("a/b/MRI BRAIN - Set 1.webm")));
        assert!(is_video(Path::new("clip.MP4")));
        assert!(is_video(Path::new("scan.mkv")));
        assert!(!is_video(Path::new("frame.png")));
        assert!(!is_video(Path::new("frame.jpg")));
        assert!(!is_video(Path::new("noext")));
    }

    #[test]
    fn sample_indices_basic() {
        assert_eq!(sample_indices(0, 8), Vec::<usize>::new());
        assert_eq!(sample_indices(5, 0), Vec::<usize>::new());
        // k >= n → all
        assert_eq!(sample_indices(3, 8), vec![0, 1, 2]);
        assert_eq!(sample_indices(8, 8), vec![0, 1, 2, 3, 4, 5, 6, 7]);
        // k == 1 → middle
        assert_eq!(sample_indices(24, 1), vec![12]);
        // endpoints inclusive, evenly spaced
        assert_eq!(sample_indices(24, 4), vec![0, 8, 15, 23]);
        let s = sample_indices(119, 8);
        assert_eq!(s.first(), Some(&0));
        assert_eq!(s.last(), Some(&118));
        assert_eq!(s.len(), 8);
        // strictly increasing, no dups
        assert!(s.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn scale_filter_expands_limited_only() {
        assert_eq!(
            scale_filter(Some("tv".into())),
            "scale=iw:ih:in_range=tv:out_range=full"
        );
        assert_eq!(scale_filter(None), "scale=iw:ih:in_range=tv:out_range=full");
        assert_eq!(
            scale_filter(Some("pc".into())),
            "scale=iw:ih:in_range=full:out_range=full"
        );
    }

    /// End-to-end decode, skipped when ffmpeg is absent. Synthesizes a tiny
    /// 3-frame webm, then asserts decode_frames returns 3 decodable PNGs.
    #[test]
    fn decode_frames_roundtrip() {
        if Command::new("ffmpeg").args(["-version"]).output().is_err() {
            eprintln!("skipping: ffmpeg not on PATH");
            return;
        }
        let scratch = ScratchDir::new().unwrap();
        let webm = scratch.dir.join("synth.webm");
        // 3 solid frames at 1 fps, tiny resolution, VP9 limited range.
        let ok = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=gray:s=64x48:r=1:d=3",
                "-c:v",
                "libvpx-vp9",
                "-frames:v",
                "3",
            ])
            .arg(&webm)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            eprintln!("skipping: ffmpeg cannot encode libvpx-vp9");
            return;
        }
        let frames = decode_frames(&webm, 8).unwrap();
        assert_eq!(frames.len(), 3, "expected all 3 frames (k>=n)");
        for f in &frames {
            // Each must be a decodable PNG of the right size.
            let img = image::load_from_memory(f).expect("frame is a valid image");
            assert_eq!((img.width(), img.height()), (64, 48));
        }
        // Sampling down to 2 keeps endpoints.
        assert_eq!(decode_frames(&webm, 2).unwrap().len(), 2);
    }
}
