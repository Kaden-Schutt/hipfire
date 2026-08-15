// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bounded, ordered HFQ reads for production model loading.
//!
//! Reader lanes fill caller-declared final host buffers concurrently. Results
//! are returned in plan order, so architecture loaders can preserve their
//! existing GPU allocation and pointer-table order exactly. Callers deliberately
//! submit small windows (at most four large jobs); this bounds live host memory
//! while still saturating the local NVMe on `hipx`/Strix Halo.
//!
//! Attached HFQ overlays are refused. Their tensor offsets belong to a second
//! file, while this reader owns one base path per lane. Architecture loaders must
//! retain their existing overlay-aware path as the fallback.

use crate::hfq::HfqFile;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Production default established by the DS4-shaped storage and canonical
/// upload screens. Four lanes reached 1.76-1.91 GB/s without changing GPU
/// allocation order; wider lane counts regressed on the same NVMe.
pub const HFQ_READER_LANES: usize = 4;

#[derive(Clone, Debug)]
struct HfqReadSegment {
    source_offset: usize,
    len: usize,
    destination_offset: usize,
}

/// One canonical host output buffer. A job may be a single tensor or a packed
/// concatenation of many source tensors (for example DS4 expert `w1 || w3`).
#[derive(Clone, Debug)]
pub struct HfqReadJob {
    label: String,
    output_len: usize,
    segments: Vec<HfqReadSegment>,
}

impl HfqReadJob {
    /// Build a direct read job for one tensor.
    pub fn tensor(hfq: &HfqFile, name: &str) -> io::Result<Self> {
        Self::packed(hfq, name, [name])
    }

    /// Build one final buffer by concatenating `tensor_names` in the supplied
    /// order. Source tensors are validated before any reader thread starts.
    pub fn packed<I, S>(
        hfq: &HfqFile,
        label: impl Into<String>,
        tensor_names: I,
    ) -> io::Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        if hfq.has_overlay() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "parallel HFQ reads do not cross attached overlay files",
            ));
        }
        let mut output_len = 0usize;
        let mut segments = Vec::new();
        for name in tensor_names {
            let name = name.as_ref();
            let info = hfq.find_tensor_info(name).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("HFQ tensor not found: {name}"),
                )
            })?;
            let next_len = output_len.checked_add(info.data_size).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "packed HFQ job size overflow")
            })?;
            segments.push(HfqReadSegment {
                source_offset: info.data_offset,
                len: info.data_size,
                destination_offset: output_len,
            });
            output_len = next_len;
        }
        if segments.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "parallel HFQ read job has no tensors",
            ));
        }
        Ok(Self {
            label: label.into(),
            output_len,
            segments,
        })
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn output_len(&self) -> usize {
        self.output_len
    }
}

#[derive(Debug)]
pub struct HfqReadResult {
    pub label: String,
    pub data: Vec<u8>,
}

/// Fill all jobs with up to four independent file handles and return them in
/// the exact order supplied. Callers then upload/install in their pre-existing
/// canonical order. The function never allocates GPU memory or changes model
/// state.
pub fn read_hfq_jobs_ordered(hfq: &HfqFile, jobs: &[HfqReadJob]) -> io::Result<Vec<HfqReadResult>> {
    if hfq.has_overlay() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "parallel HFQ reads do not cross attached overlay files",
        ));
    }
    read_jobs_from_path(hfq.path(), jobs, HFQ_READER_LANES)
}

fn read_jobs_from_path(
    path: &Path,
    jobs: &[HfqReadJob],
    lanes: usize,
) -> io::Result<Vec<HfqReadResult>> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    let lanes = lanes.clamp(1, jobs.len());
    let next_job = AtomicUsize::new(0);
    let slots = Arc::new(Mutex::new(
        (0..jobs.len())
            .map(|_| None)
            .collect::<Vec<Option<io::Result<HfqReadResult>>>>(),
    ));
    let path = PathBuf::from(path);

    std::thread::scope(|scope| {
        for _ in 0..lanes {
            let slots = Arc::clone(&slots);
            let path = path.clone();
            let next_job = &next_job;
            scope.spawn(move || {
                let file = match File::open(&path) {
                    Ok(file) => file,
                    Err(error) => {
                        loop {
                            let index = next_job.fetch_add(1, Ordering::Relaxed);
                            if index >= jobs.len() {
                                break;
                            }
                            slots.lock().expect("HFQ result slots poisoned")[index] =
                                Some(Err(io::Error::new(
                                    error.kind(),
                                    format!("open {}: {error}", path.display()),
                                )));
                        }
                        return;
                    }
                };
                advise_sequential(&file);
                loop {
                    let index = next_job.fetch_add(1, Ordering::Relaxed);
                    if index >= jobs.len() {
                        break;
                    }
                    let result = read_job(&file, &jobs[index]);
                    slots.lock().expect("HFQ result slots poisoned")[index] = Some(result);
                }
            });
        }
    });

    let mut slots = slots.lock().expect("HFQ result slots poisoned");
    let mut ordered = Vec::with_capacity(jobs.len());
    for (index, slot) in slots.iter_mut().enumerate() {
        let result = slot.take().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("parallel HFQ reader did not fill job {index}"),
            )
        })??;
        debug_assert_eq!(result.label, jobs[index].label);
        ordered.push(result);
    }
    Ok(ordered)
}

fn read_job(file: &File, job: &HfqReadJob) -> io::Result<HfqReadResult> {
    let mut data = vec![0u8; job.output_len];
    for segment in &job.segments {
        let end = segment
            .destination_offset
            .checked_add(segment.len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "HFQ segment overflow"))?;
        read_exact_at(
            file,
            &mut data[segment.destination_offset..end],
            segment.source_offset as u64,
        )?;
        advise_dontneed(file, segment.source_offset, segment.len);
    }
    Ok(HfqReadResult {
        label: job.label.clone(),
        data,
    })
}

#[cfg(unix)]
fn read_exact_at(file: &File, mut buffer: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    while !buffer.is_empty() {
        let read = file.read_at(buffer, offset)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short parallel HFQ pread",
            ));
        }
        offset += read as u64;
        buffer = &mut buffer[read..];
    }
    Ok(())
}

#[cfg(not(unix))]
fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = file.try_clone()?;
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buffer)
}

#[cfg(unix)]
fn advise_sequential(file: &File) {
    use std::os::fd::AsRawFd;
    let _ = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL) };
}

#[cfg(not(unix))]
fn advise_sequential(_file: &File) {}

#[cfg(unix)]
fn advise_dontneed(file: &File, offset: usize, len: usize) {
    use std::os::fd::AsRawFd;
    let _ = unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        )
    };
}

#[cfg(not(unix))]
fn advise_dontneed(_file: &File, _offset: usize, _len: usize) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn job(label: &str, segments: &[(usize, usize)]) -> HfqReadJob {
        let mut destination_offset = 0usize;
        let segments = segments
            .iter()
            .map(|&(source_offset, len)| {
                let segment = HfqReadSegment {
                    source_offset,
                    len,
                    destination_offset,
                };
                destination_offset += len;
                segment
            })
            .collect();
        HfqReadJob {
            label: label.to_string(),
            output_len: destination_offset,
            segments,
        }
    }

    #[test]
    fn parallel_reads_preserve_job_and_segment_order() {
        let mut source = tempfile::NamedTempFile::new().unwrap();
        let bytes = (0u8..=127).collect::<Vec<_>>();
        source.write_all(&bytes).unwrap();
        source.flush().unwrap();
        let jobs = vec![
            job("first", &[(16, 8), (0, 4)]),
            job("second", &[(80, 12)]),
            job("third", &[(40, 3), (60, 5), (120, 8)]),
            job("fourth", &[(8, 8)]),
        ];
        let results = read_jobs_from_path(source.path(), &jobs, 4).unwrap();
        assert_eq!(results[0].label, "first");
        assert_eq!(results[0].data, [&bytes[16..24], &bytes[0..4]].concat());
        assert_eq!(results[1].label, "second");
        assert_eq!(results[1].data, bytes[80..92]);
        assert_eq!(results[2].label, "third");
        assert_eq!(
            results[2].data,
            [&bytes[40..43], &bytes[60..65], &bytes[120..128]].concat()
        );
        assert_eq!(results[3].label, "fourth");
        assert_eq!(results[3].data, bytes[8..16]);
    }

    #[test]
    fn short_read_fails_closed() {
        let mut source = tempfile::NamedTempFile::new().unwrap();
        source.write_all(&[1, 2, 3, 4]).unwrap();
        source.flush().unwrap();
        let error = read_jobs_from_path(source.path(), &[job("short", &[(2, 8)])], 1).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }
}
