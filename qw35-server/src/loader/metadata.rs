use super::cursor::Cursor;
use super::mmap::MappedGguf;

pub(crate) const GGUF_VALUE_UINT8: u32 = 0;
pub(crate) const GGUF_VALUE_INT8: u32 = 1;
pub(crate) const GGUF_VALUE_UINT16: u32 = 2;
pub(crate) const GGUF_VALUE_INT16: u32 = 3;
pub(crate) const GGUF_VALUE_UINT32: u32 = 4;
pub(crate) const GGUF_VALUE_INT32: u32 = 5;
pub(crate) const GGUF_VALUE_FLOAT32: u32 = 6;
pub(crate) const GGUF_VALUE_BOOL: u32 = 7;
pub(crate) const GGUF_VALUE_STRING: u32 = 8;
pub(crate) const GGUF_VALUE_ARRAY: u32 = 9;
pub(crate) const GGUF_VALUE_UINT64: u32 = 10;
pub(crate) const GGUF_VALUE_INT64: u32 = 11;
pub(crate) const GGUF_VALUE_FLOAT64: u32 = 12;

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

impl MappedGguf {
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
            if end > self.byte_len() {
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

pub(crate) fn read_metadata_value(
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
