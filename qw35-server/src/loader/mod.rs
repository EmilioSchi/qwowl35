//! GGUF model loader: mmap the file, parse the header/metadata/tensor table,
//! and expose tensor + codec descriptors to the Metal runtime.

mod cursor;
mod metadata;
mod mmap;
mod tensors;

pub use metadata::{MetadataEntry, MetadataStringIter, MetadataValue};
pub use mmap::{MappedGguf, WarmReport};
pub use tensors::{tensor_type_name, TensorInfo};
