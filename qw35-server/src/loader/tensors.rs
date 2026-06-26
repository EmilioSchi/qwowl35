use super::cursor::{Cursor, GGUF_MAGIC, GGUF_VERSION};
use super::metadata::{read_metadata_value, MetadataEntry, MetadataValue};

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

pub(crate) struct ParsedGguf {
    pub(crate) version: u32,
    pub(crate) metadata: Vec<MetadataEntry>,
    pub(crate) tensors: Vec<TensorInfo>,
    pub(crate) alignment: u64,
    pub(crate) tensor_data_pos: u64,
}

pub(crate) unsafe fn parse_mapping(ptr: *const u8, len: usize) -> Result<ParsedGguf, String> {
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

fn tensor_nbytes(type_id: u32, elements: u64) -> Option<u64> {
    // GF2 (qw35 2-bit sidecar codec) stores two planes — 16 two-bit codes
    // packed in one uint32 plus one fp8(e5m2) scale byte per group of 16, i.e.
    // 5 bytes per 16 elements — which does not fit the single (block_elems,
    // block_bytes) tuple math below, so size it explicitly.
    if type_id == 101 {
        return elements
            .checked_add(15)
            .map(|padded| padded / 16)
            .and_then(|groups| groups.checked_mul(5));
    }
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
        // qw35 GF4 sidecar codec: 8 weights per group packed into one uint32
        // (eight 3-bit codes + one fp8 e5m2 scale byte) = 4 bytes / 8 elems.
        100 => (8, 4),
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
        100 => "gf4",
        101 => "gf2",
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
