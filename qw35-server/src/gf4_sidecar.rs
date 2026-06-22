use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::os::raw::c_int;
use std::path::Path;
use std::ptr::NonNull;

const MAGIC: &[u8; 8] = b"QW35GF4\0";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 24;
const PROT_READ: c_int = 0x01;
const MAP_SHARED: c_int = 0x0001;

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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gf4TensorRecord {
    pub name: String,
    pub source_type: u16,
    pub rows: u32,
    pub cols: u32,
    pub data_offset: u64,
    pub data_nbytes: u64,
    pub groups_per_row: u64,
    pub prepared_codes: bool,
}

pub struct MappedGf4Sidecar {
    _file: File,
    ptr: NonNull<u8>,
    len: usize,
    pub tensors: Vec<Gf4TensorRecord>,
}

unsafe impl Send for MappedGf4Sidecar {}
unsafe impl Sync for MappedGf4Sidecar {}

impl MappedGf4Sidecar {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|err| format!("cannot open GF4 sidecar {}: {err}", path.display()))?;
        let len_u64 = file
            .metadata()
            .map_err(|err| format!("cannot stat GF4 sidecar {}: {err}", path.display()))?
            .len();
        let len = usize::try_from(len_u64)
            .map_err(|_| "GF4 sidecar is too large for this process".to_string())?;
        if len < HEADER_LEN {
            return Err("GF4 sidecar is too small".to_string());
        }
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
            return Err(format!(
                "cannot mmap GF4 sidecar {}: {}",
                path.display(),
                io::Error::last_os_error()
            ));
        }
        let ptr = NonNull::new(raw.cast::<u8>()).ok_or("mmap returned a null pointer")?;
        let parse = unsafe { parse_mapping(ptr.as_ptr(), len) };
        match parse {
            Ok(tensors) => Ok(Self {
                _file: file,
                ptr,
                len,
                tensors,
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

    pub fn tensor(&self, name: &str) -> Option<&Gf4TensorRecord> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Validate the sidecar for Qw35 decode: every record must have sound GF4
    /// geometry and lie inside the mapping, and the full FFN tensor set must
    /// be present (the decode FFN path switches wholly to GF4 when a sidecar
    /// is loaded). Additional tensors (ssm_out, attention projections,
    /// output.weight) are optional and picked up automatically.
    pub fn validate_qw35(&self, block_count: u32) -> Result<(), String> {
        for tensor in &self.tensors {
            let name = &tensor.name;
            if !tensor.prepared_codes {
                return Err(format!(
                    "GF4 sidecar {name} is not stored with prepared GF4 codes"
                ));
            }
            if tensor.cols % 8 != 0 || tensor.groups_per_row != u64::from(tensor.cols / 8) {
                return Err(format!(
                    "GF4 sidecar {name} has invalid GF4 row geometry (cols={}, groups_per_row={})",
                    tensor.cols, tensor.groups_per_row
                ));
            }
            let expected_nbytes = u64::from(tensor.rows) * u64::from(tensor.cols / 8) * 4;
            if tensor.data_nbytes != expected_nbytes {
                return Err(format!(
                    "GF4 sidecar {name} byte size {}, expected {expected_nbytes}",
                    tensor.data_nbytes
                ));
            }
            let end = tensor
                .data_offset
                .checked_add(tensor.data_nbytes)
                .ok_or_else(|| format!("GF4 sidecar {name} byte range overflows"))?;
            if end > self.len() {
                return Err(format!(
                    "GF4 sidecar {name} byte range ends at {end}, beyond file length {}",
                    self.len()
                ));
            }
        }

        for layer in 0..block_count {
            for (kind, rows, cols) in [
                ("down", 4096u32, 12288u32),
                ("gate", 12288u32, 4096u32),
                ("up", 12288u32, 4096u32),
            ] {
                let name = format!("blk.{layer}.ffn_{kind}.weight");
                let tensor = self
                    .tensor(&name)
                    .ok_or_else(|| format!("GF4 sidecar missing {name}"))?;
                if tensor.rows != rows || tensor.cols != cols {
                    return Err(format!(
                        "GF4 sidecar {name} has rows={} cols={}, expected rows={rows} cols={cols}",
                        tensor.rows, tensor.cols
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Drop for MappedGf4Sidecar {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr.as_ptr().cast::<c_void>(), self.len);
        }
    }
}

unsafe fn parse_mapping(ptr: *const u8, len: usize) -> Result<Vec<Gf4TensorRecord>, String> {
    let data = std::slice::from_raw_parts(ptr, len);
    parse_bytes(data)
}

fn parse_bytes(data: &[u8]) -> Result<Vec<Gf4TensorRecord>, String> {
    if data.len() < HEADER_LEN {
        return Err("GF4 sidecar is too small".to_string());
    }
    if &data[..8] != MAGIC {
        return Err("GF4 sidecar magic mismatch".to_string());
    }
    let version = read_u32(data, 8)?;
    if version != VERSION {
        return Err(format!(
            "unsupported GF4 sidecar version {version}, expected {VERSION}"
        ));
    }
    let header_count = read_u32(data, 12)?;
    let table_offset = read_u64(data, 16)? as usize;
    if table_offset < HEADER_LEN || table_offset >= data.len() {
        return Err(format!("invalid GF4 sidecar table offset {table_offset}"));
    }

    let mut pos = table_offset;
    let table_count = read_u32_at(data, &mut pos)?;
    if table_count != header_count {
        return Err(format!(
            "GF4 sidecar table count {table_count} does not match header count {header_count}"
        ));
    }
    let mut tensors = Vec::with_capacity(table_count as usize);
    for _ in 0..table_count {
        let name_len = read_u16_at(data, &mut pos)? as usize;
        let name_end = pos
            .checked_add(name_len)
            .ok_or("GF4 sidecar tensor name length overflows")?;
        if name_end > data.len() {
            return Err("GF4 sidecar tensor name extends beyond file".to_string());
        }
        let name = std::str::from_utf8(&data[pos..name_end])
            .map_err(|err| format!("GF4 sidecar tensor name is not UTF-8: {err}"))?
            .to_string();
        pos = name_end;
        let source_type = read_u16_at(data, &mut pos)?;
        let rows = read_u32_at(data, &mut pos)?;
        let cols = read_u32_at(data, &mut pos)?;
        let data_offset = read_u64_at(data, &mut pos)?;
        let data_nbytes = read_u64_at(data, &mut pos)?;
        let groups_per_row = read_u64_at(data, &mut pos)?;
        let _reserved = read_u32_at(data, &mut pos)?;
        let prepared_codes = read_bool_at(data, &mut pos)?;
        tensors.push(Gf4TensorRecord {
            name,
            source_type,
            rows,
            cols,
            data_offset,
            data_nbytes,
            groups_per_row,
            prepared_codes,
        });
    }
    Ok(tensors)
}

fn read_u16(data: &[u8], pos: usize) -> Result<u16, String> {
    let bytes = data
        .get(pos..pos + 2)
        .ok_or_else(|| format!("GF4 sidecar truncated reading u16 at {pos}"))?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(data: &[u8], pos: usize) -> Result<u32, String> {
    let bytes = data
        .get(pos..pos + 4)
        .ok_or_else(|| format!("GF4 sidecar truncated reading u32 at {pos}"))?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u64(data: &[u8], pos: usize) -> Result<u64, String> {
    let bytes = data
        .get(pos..pos + 8)
        .ok_or_else(|| format!("GF4 sidecar truncated reading u64 at {pos}"))?;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn read_u16_at(data: &[u8], pos: &mut usize) -> Result<u16, String> {
    let value = read_u16(data, *pos)?;
    *pos += 2;
    Ok(value)
}

fn read_u32_at(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    let value = read_u32(data, *pos)?;
    *pos += 4;
    Ok(value)
}

fn read_u64_at(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    let value = read_u64(data, *pos)?;
    *pos += 8;
    Ok(value)
}

fn read_bool_at(data: &[u8], pos: &mut usize) -> Result<bool, String> {
    let value = *data
        .get(*pos)
        .ok_or_else(|| format!("GF4 sidecar truncated reading bool at {}", *pos))?;
    *pos += 1;
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(format!("GF4 sidecar invalid bool value {value}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn sample_sidecar() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&VERSION.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&64u64.to_le_bytes());
        data.resize(64, 0);
        data.extend_from_slice(&1u32.to_le_bytes());
        let name = b"blk.0.ffn_gate.weight";
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&12u16.to_le_bytes());
        data.extend_from_slice(&12288u32.to_le_bytes());
        data.extend_from_slice(&4096u32.to_le_bytes());
        data.extend_from_slice(&24u64.to_le_bytes());
        data.extend_from_slice(&128u64.to_le_bytes());
        data.extend_from_slice(&512u64.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.push(1);
        data
    }

    #[test]
    fn parses_embedded_tensor_table() {
        let records = parse_bytes(&sample_sidecar()).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].name, "blk.0.ffn_gate.weight");
        assert_eq!(records[0].source_type, 12);
        assert_eq!(records[0].rows, 12288);
        assert_eq!(records[0].cols, 4096);
        assert!(records[0].prepared_codes);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut data = sample_sidecar();
        data[0] = b'X';
        assert!(parse_bytes(&data).unwrap_err().contains("magic"));
    }

    #[test]
    fn mmaps_sidecar_file() {
        let path =
            std::env::temp_dir().join(format!("qw35-gf4-sidecar-test-{}.bin", std::process::id()));
        {
            let mut file = File::create(&path).unwrap();
            file.write_all(&sample_sidecar()).unwrap();
        }
        let mapped = MappedGf4Sidecar::open(&path).unwrap();
        assert_eq!(mapped.tensors.len(), 1);
        assert!(mapped.tensor("blk.0.ffn_gate.weight").is_some());
        let _ = std::fs::remove_file(path);
    }
}
