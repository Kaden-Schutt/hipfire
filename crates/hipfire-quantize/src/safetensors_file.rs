//! Validated, mmap-backed safetensors reader shared by quantizer binaries.

use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute byte offsets within `mmap`.
    pub data_offsets: [usize; 2],
}

pub struct SafetensorsFile {
    file: File,
    mmap: Mmap,
    pub tensors: HashMap<String, TensorMeta>,
}

impl SafetensorsFile {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let parsed = SafeTensors::deserialize(&mmap)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mmap_start = mmap.as_ptr() as usize;
        let mut tensors = HashMap::with_capacity(parsed.len());
        for (name, view) in parsed.iter() {
            let start = (view.data().as_ptr() as usize)
                .checked_sub(mmap_start)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("safetensors tensor {name} is outside its mmap"),
                    )
                })?;
            let end = start.checked_add(view.data().len()).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("safetensors tensor {name} offset overflow"),
                )
            })?;
            tensors.insert(
                name.to_string(),
                TensorMeta {
                    dtype: view.dtype().to_string(),
                    shape: view.shape().to_vec(),
                    data_offsets: [start, end],
                },
            );
        }

        Ok(Self {
            file,
            mmap,
            tensors,
        })
    }

    pub fn tensor_data(&self, name: &str) -> Option<(&TensorMeta, &[u8])> {
        let meta = self.tensors.get(name)?;
        Some((meta, &self.mmap[meta.data_offsets[0]..meta.data_offsets[1]]))
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names: Vec<_> = self.tensors.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Advise the kernel to drop page cache for a tensor's data region.
    #[cfg(unix)]
    pub fn drop_tensor_pages(&self, name: &str) {
        if let Some(meta) = self.tensors.get(name) {
            let start = meta.data_offsets[0];
            let len = meta.data_offsets[1] - start;
            use std::os::unix::io::AsRawFd;
            unsafe {
                libc::posix_fadvise(
                    self.file.as_raw_fd(),
                    start as libc::off_t,
                    len as libc::off_t,
                    libc::POSIX_FADV_DONTNEED,
                );
            }
        }
    }

    #[cfg(not(unix))]
    pub fn drop_tensor_pages(&self, _name: &str) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    #[test]
    fn truncated_header_is_an_error_not_a_panic() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&[0, 1, 2, 3]).unwrap();
        let err = SafetensorsFile::open(file.path()).err().unwrap();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
