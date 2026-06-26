pub(crate) const GGUF_MAGIC: u32 = 0x4655_4747;
pub(crate) const GGUF_VERSION: u32 = 3;

pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    pub(crate) fn pos(&self) -> usize {
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

    pub(crate) fn skip(&mut self, len: usize) -> Result<usize, String> {
        let old = self.pos;
        self.take(len)?;
        Ok(old)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u32(&mut self) -> Result<u32, String> {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(buf))
    }

    pub(crate) fn i32(&mut self) -> Result<i32, String> {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(self.take(4)?);
        Ok(i32::from_le_bytes(buf))
    }

    pub(crate) fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(self.u32()?))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, String> {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(buf))
    }

    pub(crate) fn string(&mut self) -> Result<String, String> {
        let len = self.u64()?;
        let len = usize::try_from(len).map_err(|_| "GGUF string is too long".to_string())?;
        let raw = self.take(len)?;
        std::str::from_utf8(raw)
            .map(str::to_owned)
            .map_err(|err| format!("GGUF string is not UTF-8: {err}"))
    }
}
