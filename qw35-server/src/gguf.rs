use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::os::raw::{c_int, c_long};
use std::path::Path;
use std::ptr::NonNull;
use std::time::{Duration, Instant};

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION: u32 = 3;

const PROT_READ: c_int = 0x01;
const MAP_SHARED: c_int = 0x0001;
const POSIX_MADV_WILLNEED: c_int = 3;

const GGUF_VALUE_UINT8: u32 = 0;
const GGUF_VALUE_INT8: u32 = 1;
const GGUF_VALUE_UINT16: u32 = 2;
const GGUF_VALUE_INT16: u32 = 3;
const GGUF_VALUE_UINT32: u32 = 4;
const GGUF_VALUE_INT32: u32 = 5;
const GGUF_VALUE_FLOAT32: u32 = 6;
const GGUF_VALUE_BOOL: u32 = 7;
const GGUF_VALUE_STRING: u32 = 8;
const GGUF_VALUE_ARRAY: u32 = 9;
const GGUF_VALUE_UINT64: u32 = 10;
const GGUF_VALUE_INT64: u32 = 11;
const GGUF_VALUE_FLOAT64: u32 = 12;

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
pub enum MetadataValue {
    U32(u32),
    U64(u64),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array {
        item_type: u32,
        len: u64,
        data_pos: u64,
    },
    Skipped {
        type_id: u32,
        value_pos: u64,
    },
}

#[derive(Debug, Clone)]
pub struct MetadataEntry {
    pub key: String,
    pub value: MetadataValue,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub type_id: u32,
    pub rel_offset: u64,
    pub abs_offset: u64,
    pub elements: u64,
    pub bytes: u64,
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

    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        self.metadata.iter().find_map(|entry| {
            if entry.key == key {
                if let MetadataValue::String(value) = &entry.value {
                    return Some(value.as_str());
                }
            }
            None
        })
    }

    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        self.metadata.iter().find_map(|entry| {
            if entry.key == key {
                if let MetadataValue::U32(value) = entry.value {
                    return Some(value);
                }
            }
            None
        })
    }

    pub fn metadata_u64(&self, key: &str) -> Option<u64> {
        self.metadata.iter().find_map(|entry| {
            if entry.key == key {
                if let MetadataValue::U64(value) = entry.value {
                    return Some(value);
                }
            }
            None
        })
    }

    pub fn metadata_f32(&self, key: &str) -> Option<f32> {
        self.metadata.iter().find_map(|entry| {
            if entry.key == key {
                if let MetadataValue::F32(value) = entry.value {
                    return Some(value);
                }
            }
            None
        })
    }

    pub fn metadata_bool(&self, key: &str) -> Option<bool> {
        self.metadata.iter().find_map(|entry| {
            if entry.key == key {
                if let MetadataValue::Bool(value) = entry.value {
                    return Some(value);
                }
            }
            None
        })
    }

    pub fn metadata_array_len(&self, key: &str) -> Option<u64> {
        self.metadata.iter().find_map(|entry| {
            if entry.key == key {
                if let MetadataValue::Array { len, .. } = entry.value {
                    return Some(len);
                }
            }
            None
        })
    }

    pub fn metadata_array_i32(&self, key: &str) -> Option<Vec<i32>> {
        let (item_type, len, data_pos) = self.metadata_array_parts(key)?;
        if item_type != GGUF_VALUE_INT32 {
            return None;
        }
        let byte_len = usize::try_from(len.checked_mul(4)?).ok()?;
        let bytes = self.bytes_at(usize::try_from(data_pos).ok()?, byte_len)?;
        let mut values = Vec::with_capacity(usize::try_from(len).ok()?);
        for chunk in bytes.chunks_exact(4) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            values.push(i32::from_le_bytes(buf));
        }
        Some(values)
    }

    pub fn metadata_array_string_at(&self, key: &str, index: u64) -> Option<&str> {
        let (item_type, len, data_pos) = self.metadata_array_parts(key)?;
        if item_type != GGUF_VALUE_STRING || index >= len {
            return None;
        }

        let mut pos = usize::try_from(data_pos).ok()?;
        for idx in 0..=index {
            let len_bytes = self.bytes_at(pos, 8)?;
            let mut buf = [0u8; 8];
            buf.copy_from_slice(len_bytes);
            let s_len = usize::try_from(u64::from_le_bytes(buf)).ok()?;
            let start = pos.checked_add(8)?;
            let end = start.checked_add(s_len)?;
            if end > self.len {
                return None;
            }
            if idx == index {
                return std::str::from_utf8(self.bytes_at(start, s_len)?).ok();
            }
            pos = end;
        }
        None
    }

    pub fn metadata_array_string_iter(&self, key: &str) -> Option<MetadataStringIter<'_>> {
        let (item_type, len, data_pos) = self.metadata_array_parts(key)?;
        if item_type != GGUF_VALUE_STRING {
            return None;
        }
        Some(MetadataStringIter {
            gguf: self,
            pos: usize::try_from(data_pos).ok()?,
            remaining: len,
        })
    }

    fn metadata_array_parts(&self, key: &str) -> Option<(u32, u64, u64)> {
        self.metadata.iter().find_map(|entry| {
            if entry.key == key {
                if let MetadataValue::Array {
                    item_type,
                    len,
                    data_pos,
                } = entry.value
                {
                    return Some((item_type, len, data_pos));
                }
            }
            None
        })
    }

    fn bytes_at(&self, pos: usize, len: usize) -> Option<&[u8]> {
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

pub struct MetadataStringIter<'a> {
    gguf: &'a MappedGguf,
    pos: usize,
    remaining: u64,
}

impl<'a> Iterator for MetadataStringIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let len_bytes = self.gguf.bytes_at(self.pos, 8)?;
        let mut buf = [0u8; 8];
        buf.copy_from_slice(len_bytes);
        let s_len = usize::try_from(u64::from_le_bytes(buf)).ok()?;
        let start = self.pos.checked_add(8)?;
        let end = start.checked_add(s_len)?;
        let bytes = self.gguf.bytes_at(start, s_len)?;
        self.pos = end;
        self.remaining -= 1;
        std::str::from_utf8(bytes).ok()
    }
}

impl Drop for MappedGguf {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr.as_ptr().cast::<c_void>(), self.len);
        }
    }
}

struct ParsedGguf {
    version: u32,
    metadata: Vec<MetadataEntry>,
    tensors: Vec<TensorInfo>,
    alignment: u64,
    tensor_data_pos: u64,
}

unsafe fn parse_mapping(ptr: *const u8, len: usize) -> Result<ParsedGguf, String> {
    let bytes = std::slice::from_raw_parts(ptr, len);
    let mut c = Cursor::new(bytes);

    let magic = c.u32()?;
    if magic != GGUF_MAGIC {
        return Err("model is not a GGUF file".to_string());
    }
    let version = c.u32()?;
    if version != GGUF_VERSION {
        return Err(format!("only GGUF v3 is supported, found v{version}"));
    }
    let n_tensors = c.u64()?;
    let n_kv = c.u64()?;

    let mut alignment = 32u64;
    let mut metadata = Vec::with_capacity(usize::try_from(n_kv).unwrap_or(0));
    for _ in 0..n_kv {
        let key = c.string()?;
        let type_id = c.u32()?;
        let value_pos = c.pos() as u64;
        let value = read_metadata_value(&mut c, type_id, 0)?;
        if key == "general.alignment" {
            if let MetadataValue::U32(value) = value {
                if value != 0 {
                    alignment = u64::from(value);
                }
                metadata.push(MetadataEntry {
                    key,
                    value: MetadataValue::U32(value),
                });
                continue;
            }
        }
        metadata.push(MetadataEntry {
            key,
            value: match value {
                MetadataValue::Skipped { .. } => MetadataValue::Skipped { type_id, value_pos },
                other => other,
            },
        });
    }

    let mut tensors = Vec::with_capacity(usize::try_from(n_tensors).unwrap_or(0));
    for _ in 0..n_tensors {
        let name = c.string()?;
        let ndim = c.u32()?;
        if ndim == 0 || ndim > 8 {
            return Err(format!(
                "tensor {name} has unsupported dimension count {ndim}"
            ));
        }
        let mut dims = Vec::with_capacity(ndim as usize);
        let mut elements = 1u64;
        for _ in 0..ndim {
            let dim = c.u64()?;
            if dim != 0 {
                elements = elements
                    .checked_mul(dim)
                    .ok_or_else(|| format!("tensor {name} element count overflow"))?;
            }
            dims.push(dim);
        }
        let type_id = c.u32()?;
        let rel_offset = c.u64()?;
        let bytes = tensor_nbytes(type_id, elements)
            .ok_or_else(|| format!("tensor {name} has unsupported GGUF type {type_id}"))?;
        tensors.push(TensorInfo {
            name,
            dims,
            type_id,
            rel_offset,
            abs_offset: 0,
            elements,
            bytes,
        });
    }

    let tensor_data_pos = align_up(c.pos() as u64, alignment);
    let file_len = len as u64;
    for tensor in &mut tensors {
        tensor.abs_offset = tensor_data_pos
            .checked_add(tensor.rel_offset)
            .ok_or_else(|| format!("tensor {} offset overflow", tensor.name))?;
        if tensor.bytes != 0
            && (tensor.abs_offset > file_len || tensor.bytes > file_len - tensor.abs_offset)
        {
            return Err(format!("tensor {} points outside GGUF file", tensor.name));
        }
    }

    Ok(ParsedGguf {
        version,
        metadata,
        tensors,
        alignment,
        tensor_data_pos,
    })
}

fn read_metadata_value(
    c: &mut Cursor<'_>,
    type_id: u32,
    depth: u8,
) -> Result<MetadataValue, String> {
    if depth > 8 {
        return Err("metadata array nesting is too deep".to_string());
    }

    Ok(match type_id {
        GGUF_VALUE_UINT8 => MetadataValue::Skipped {
            type_id,
            value_pos: c.skip(1)? as u64,
        },
        GGUF_VALUE_INT8 => MetadataValue::Skipped {
            type_id,
            value_pos: c.skip(1)? as u64,
        },
        GGUF_VALUE_UINT16 => MetadataValue::Skipped {
            type_id,
            value_pos: c.skip(2)? as u64,
        },
        GGUF_VALUE_INT16 => MetadataValue::Skipped {
            type_id,
            value_pos: c.skip(2)? as u64,
        },
        GGUF_VALUE_UINT32 => MetadataValue::U32(c.u32()?),
        GGUF_VALUE_INT32 => MetadataValue::I32(c.i32()?),
        GGUF_VALUE_FLOAT32 => MetadataValue::F32(c.f32()?),
        GGUF_VALUE_BOOL => MetadataValue::Bool(c.u8()? != 0),
        GGUF_VALUE_STRING => MetadataValue::String(c.string()?),
        GGUF_VALUE_ARRAY => {
            let item_type = c.u32()?;
            let len = c.u64()?;
            let data_pos = c.pos() as u64;
            skip_array(c, item_type, len, depth + 1)?;
            MetadataValue::Array {
                item_type,
                len,
                data_pos,
            }
        }
        GGUF_VALUE_UINT64 => MetadataValue::U64(c.u64()?),
        GGUF_VALUE_INT64 => MetadataValue::Skipped {
            type_id,
            value_pos: c.skip(8)? as u64,
        },
        GGUF_VALUE_FLOAT64 => MetadataValue::Skipped {
            type_id,
            value_pos: c.skip(8)? as u64,
        },
        other => return Err(format!("unknown GGUF metadata type {other}")),
    })
}

fn skip_array(c: &mut Cursor<'_>, item_type: u32, len: u64, depth: u8) -> Result<(), String> {
    if let Some(size) = scalar_value_size(item_type) {
        let bytes = len
            .checked_mul(size)
            .ok_or_else(|| "metadata array is too large".to_string())?;
        c.skip(usize::try_from(bytes).map_err(|_| "metadata array is too large".to_string())?)?;
        return Ok(());
    }

    if item_type == GGUF_VALUE_STRING || item_type == GGUF_VALUE_ARRAY {
        for _ in 0..len {
            read_metadata_value(c, item_type, depth)?;
        }
        return Ok(());
    }

    Err(format!("unknown GGUF array item type {item_type}"))
}

fn scalar_value_size(type_id: u32) -> Option<u64> {
    match type_id {
        GGUF_VALUE_UINT8 | GGUF_VALUE_INT8 | GGUF_VALUE_BOOL => Some(1),
        GGUF_VALUE_UINT16 | GGUF_VALUE_INT16 => Some(2),
        GGUF_VALUE_UINT32 | GGUF_VALUE_INT32 | GGUF_VALUE_FLOAT32 => Some(4),
        GGUF_VALUE_UINT64 | GGUF_VALUE_INT64 | GGUF_VALUE_FLOAT64 => Some(8),
        _ => None,
    }
}

fn tensor_nbytes(type_id: u32, elements: u64) -> Option<u64> {
    let (block_elems, block_bytes) = match type_id {
        0 => (1, 4),
        1 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 40),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 110),
        20 => (32, 18),
        21 => (256, 110),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1),
        25 => (1, 2),
        26 => (1, 4),
        27 => (1, 8),
        28 => (1, 8),
        29 => (256, 56),
        30 => (1, 2),
        _ => return None,
    };
    let blocks = elements.checked_add(block_elems - 1)? / block_elems;
    blocks.checked_mul(block_bytes)
}

pub fn tensor_type_name(type_id: u32) -> &'static str {
    match type_id {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
        6 => "q5_0",
        7 => "q5_1",
        8 => "q8_0",
        9 => "q8_1",
        10 => "q2_k",
        11 => "q3_k",
        12 => "q4_k",
        13 => "q5_k",
        14 => "q6_k",
        15 => "q8_k",
        16 => "iq2_xxs",
        17 => "iq2_xs",
        18 => "iq3_xxs",
        19 => "iq1_s",
        20 => "iq4_nl",
        21 => "iq3_s",
        22 => "iq2_s",
        23 => "iq4_xs",
        24 => "i8",
        25 => "i16",
        26 => "i32",
        27 => "i64",
        28 => "f64",
        29 => "iq1_m",
        30 => "bf16",
        _ => "unknown",
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + alignment - rem
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

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| "GGUF cursor overflow".to_string())?;
        if end > self.bytes.len() {
            return Err("unexpected end of GGUF file".to_string());
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn skip(&mut self, len: usize) -> Result<usize, String> {
        let old = self.pos;
        self.take(len)?;
        Ok(old)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(buf))
    }

    fn i32(&mut self) -> Result<i32, String> {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(self.take(4)?);
        Ok(i32::from_le_bytes(buf))
    }

    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn u64(&mut self) -> Result<u64, String> {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(buf))
    }

    fn string(&mut self) -> Result<String, String> {
        let len = self.u64()?;
        let len = usize::try_from(len).map_err(|_| "GGUF string is too long".to_string())?;
        let raw = self.take(len)?;
        std::str::from_utf8(raw)
            .map(str::to_owned)
            .map_err(|err| format!("GGUF string is not UTF-8: {err}"))
    }
}
