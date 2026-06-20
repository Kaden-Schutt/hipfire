// SPDX-License-Identifier: Apache-2.0
// hipfire — media (video → frames) preprocessing. See LICENSE / NOTICE.

//! Video / multi-frame decode for the vision path.
//!
//! Turns a video file (an MRI slice-stack `.webm`, etc.) into a list of still
//! PNG frames the arch image-preprocessors already understand. Pure
//! preprocessing — shells out to `ffmpeg`, touches no GPU and no inference
//! hot-path code (Rule 1: ffmpeg-as-subprocess is tooling, not Python in the
//! engine).
//!
//! Quality preservation matters for medical grayscale (see `video.rs`): frames
//! are extracted as lossless PNG at native resolution, with the source's limited
//! (`tv`) luma range expanded to full so the model's `/255` normalize sees full
//! contrast, and `-vsync 0` so only real coded frames are sampled.

mod video;

pub use video::{decode_frames, is_video, sample_indices};
