use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::os::raw::{c_int, c_long};
use std::path::Path;
use std::ptr::NonNull;
use std::time::{Duration, Instant};

use super::metadata::MetadataEntry;
use super::tensors::{parse_mapping, TensorInfo};

const PROT_READ: c_int = 0x01;
const MAP_SHARED: c_int = 0x0001;
const POSIX_MADV_WILLNEED: c_int = 3;

extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
    fn posix_madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
    fn getpagesize() -> c_int;
}

#[derive(Debug, Clone)]
pub struct WarmReport {
    pub mode: &'static str,
    pub bytes: u64,
    pub mapped_bytes: u64,
    pub page_size: u64,
    pub touched_pages: u64,
    pub view_count: u32,
    pub checksum: u64,
    pub elapsed: Duration,
    pub madvise_error: Option<String>,
}

pub struct MappedGguf {
    _file: File,
    ptr: NonNull<u8>,
    len: usize,
    pub version: u32,
    pub metadata: Vec<MetadataEntry>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    pub tensor_data_pos: u64,
}

unsafe impl Send for MappedGguf {}
unsafe impl Sync for MappedGguf {}

impl MappedGguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref())
            .map_err(|err| format!("cannot open model {}: {err}", path.as_ref().display()))?;
        let len_u64 = file
            .metadata()
            .map_err(|err| format!("cannot stat model {}: {err}", path.as_ref().display()))?
            .len();
        if len_u64 < 32 {
            return Err("model file is too small to be GGUF".to_string());
        }
        let len = usize::try_from(len_u64)
            .map_err(|_| "model file is too large for this process".to_string())?;

        let raw = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ,
                MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if raw as isize == -1 {
            return Err(format!("cannot mmap model: {}", io::Error::last_os_error()));
        }
        let ptr = NonNull::new(raw.cast::<u8>()).ok_or("mmap returned a null pointer")?;

        let parse = unsafe { parse_mapping(ptr.as_ptr(), len) };
        match parse {
            Ok(parsed) => Ok(Self {
                _file: file,
                ptr,
                len,
                version: parsed.version,
                metadata: parsed.metadata,
                tensors: parsed.tensors,
                alignment: parsed.alignment,
                tensor_data_pos: parsed.tensor_data_pos,
            }),
            Err(err) => {
                unsafe {
                    munmap(ptr.as_ptr().cast::<c_void>(), len);
                }
                Err(err)
            }
        }
    }

    pub fn len(&self) -> u64 {
        self.len as u64
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn max_tensor_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .map(|tensor| tensor.bytes)
            .max()
            .unwrap_or(0)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Byte length of the mapping (the metadata accessors in `metadata.rs` use
    /// this to bound their reads).
    pub(crate) fn byte_len(&self) -> usize {
        self.len
    }

    pub(crate) fn bytes_at(&self, pos: usize, len: usize) -> Option<&[u8]> {
        let end = pos.checked_add(len)?;
        if end > self.len {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().add(pos), len) })
    }

    pub fn warm_tensor_pages(&self) -> WarmReport {
        let start = self.tensor_data_pos.min(self.len());
        let end = self.len();
        if start >= end {
            return WarmReport {
                mode: "cpu",
                bytes: 0,
                mapped_bytes: 0,
                page_size: page_size(),
                touched_pages: 0,
                view_count: 0,
                checksum: 0,
                elapsed: Duration::ZERO,
                madvise_error: None,
            };
        }

        let bytes = end - start;
        let base = unsafe { self.ptr.as_ptr().add(start as usize) };
        let madvise_rc =
            unsafe { posix_madvise(base.cast::<c_void>(), bytes as usize, POSIX_MADV_WILLNEED) };
        let madvise_error = if madvise_rc == 0 {
            None
        } else {
            Some(io::Error::from_raw_os_error(madvise_rc).to_string())
        };

        let page = page_size();
        let t0 = Instant::now();
        let mut checksum = 0u64;
        let mut touched = 0u64;
        let mut off = start;
        while off < end {
            let byte = unsafe { std::ptr::read_volatile(self.ptr.as_ptr().add(off as usize)) };
            checksum = checksum.wrapping_add(u64::from(byte));
            touched += 1;
            off = off.saturating_add(page);
            if off == u64::MAX {
                break;
            }
        }
        let last = unsafe { std::ptr::read_volatile(self.ptr.as_ptr().add((end - 1) as usize)) };
        checksum = checksum.wrapping_add(u64::from(last));

        WarmReport {
            mode: "cpu",
            bytes,
            mapped_bytes: bytes,
            page_size: page,
            touched_pages: touched,
            view_count: 0,
            checksum,
            elapsed: t0.elapsed(),
            madvise_error,
        }
    }
}

impl Drop for MappedGguf {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr.as_ptr().cast::<c_void>(), self.len);
        }
    }
}

fn page_size() -> u64 {
    let page = unsafe { getpagesize() as c_long };
    if page > 0 {
        page as u64
    } else {
        16 * 1024
    }
}
