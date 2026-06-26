use crate::loader::MappedGguf;
use std::collections::{HashMap, HashSet};

const TOKEN_TYPE_NORMAL: i32 = 1;
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;

#[derive(Debug, Clone)]
pub struct QwenTokenizerSpec {
    pub model: String,
    pub pre: String,
    pub vocab_size: u64,
    pub merge_count: u64,
    pub eos_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub padding_token_id: Option<u32>,
    pub add_bos_token: bool,
    pub im_start_token_id: Option<u32>,
    pub im_end_token_id: Option<u32>,
    pub think_token_id: Option<u32>,
    pub end_think_token_id: Option<u32>,
    pub tool_call_token_id: Option<u32>,
    pub end_tool_call_token_id: Option<u32>,
    pub vision_start_token_id: Option<u32>,
    pub vision_end_token_id: Option<u32>,
    pub image_pad_token_id: Option<u32>,
    pub video_pad_token_id: Option<u32>,
}

impl QwenTokenizerSpec {
    pub fn load(gguf: &MappedGguf) -> Result<Self, String> {
        let tokenizer = QwenTokenizer::load(gguf)?;
        Ok(tokenizer.spec)
    }
}

#[derive(Debug, Clone)]
pub struct QwenTokenizer {
    pub spec: QwenTokenizerSpec,
    tokens: Vec<String>,
    token_types: Vec<i32>,
    token_to_id: HashMap<String, u32>,
    merge_ranks: HashMap<(String, String), u32>,
    special_token_ids: HashSet<u32>,
    special_tokens_desc: Vec<(String, u32)>,
    byte_to_unicode: [String; 256],
    unicode_to_byte: HashMap<String, u8>,
}

#[derive(Debug, Clone, Default)]
pub struct DecodeState {
    byte_buf: Vec<u8>,
}

impl QwenTokenizer {
    pub fn load(gguf: &MappedGguf) -> Result<Self, String> {
        let model = gguf
            .metadata_string("tokenizer.ggml.model")
            .ok_or("missing tokenizer.ggml.model")?
            .to_string();
        let pre = gguf
            .metadata_string("tokenizer.ggml.pre")
            .ok_or("missing tokenizer.ggml.pre")?
            .to_string();
        if model != "gpt2" || pre != "qwen35" {
            return Err(format!(
                "unsupported tokenizer model={model:?} pre={pre:?}; Qw35 expects gpt2/qwen35"
            ));
        }

        let tokens: Vec<String> = gguf
            .metadata_array_string_iter("tokenizer.ggml.tokens")
            .ok_or("missing tokenizer.ggml.tokens")?
            .map(str::to_string)
            .collect();
        let merges: Vec<String> = gguf
            .metadata_array_string_iter("tokenizer.ggml.merges")
            .ok_or("missing tokenizer.ggml.merges")?
            .map(str::to_string)
            .collect();
        let vocab_size = tokens.len() as u64;
        let merge_count = merges.len() as u64;

        let mut token_types = gguf
            .metadata_array_i32("tokenizer.ggml.token_type")
            .unwrap_or_default();
        if token_types.len() < tokens.len() {
            token_types.resize(tokens.len(), TOKEN_TYPE_NORMAL);
        }

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (idx, token) in tokens.iter().enumerate() {
            let id = u32::try_from(idx).map_err(|_| "token id overflow".to_string())?;
            token_to_id.insert(token.clone(), id);
        }

        let mut merge_ranks = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
            let Some(pos) = merge
                .as_bytes()
                .get(1..)
                .and_then(|bytes| bytes.iter().position(|byte| *byte == b' '))
                .map(|pos| pos + 1)
            else {
                continue;
            };
            let left = merge[..pos].to_string();
            let right = merge[pos + 1..].to_string();
            let rank = u32::try_from(rank).map_err(|_| "merge rank overflow".to_string())?;
            merge_ranks.insert((left, right), rank);
        }

        let mut special_token_ids = HashSet::new();
        let mut special_tokens = Vec::new();
        for (idx, token) in tokens.iter().enumerate() {
            let token_type = token_types.get(idx).copied().unwrap_or(TOKEN_TYPE_NORMAL);
            if is_special_token_text(token)
                || matches!(token_type, TOKEN_TYPE_CONTROL | TOKEN_TYPE_USER_DEFINED)
            {
                let id = u32::try_from(idx).map_err(|_| "token id overflow".to_string())?;
                special_token_ids.insert(id);
                special_tokens.push((token.clone(), id));
            }
        }
        special_tokens.sort_by(|(a, _), (b, _)| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));

        let (byte_to_unicode, unicode_to_byte) = byte_maps();
        let find = |token: &str| token_to_id.get(token).copied();
        let spec = QwenTokenizerSpec {
            model,
            pre,
            vocab_size,
            merge_count,
            eos_token_id: gguf.metadata_u32("tokenizer.ggml.eos_token_id"),
            bos_token_id: gguf.metadata_u32("tokenizer.ggml.bos_token_id"),
            padding_token_id: gguf.metadata_u32("tokenizer.ggml.padding_token_id"),
            add_bos_token: gguf
                .metadata_bool("tokenizer.ggml.add_bos_token")
                .unwrap_or(false),
            im_start_token_id: find("<|im_start|>"),
            im_end_token_id: find("<|im_end|>"),
            think_token_id: find("<think>"),
            end_think_token_id: find("</think>"),
            tool_call_token_id: find("<tool_call>"),
            end_tool_call_token_id: find("</tool_call>"),
            vision_start_token_id: find("<|vision_start|>"),
            vision_end_token_id: find("<|vision_end|>"),
            image_pad_token_id: find("<|image_pad|>"),
            video_pad_token_id: find("<|video_pad|>"),
        };

        Ok(Self {
            spec,
            tokens,
            token_types,
            token_to_id,
            merge_ranks,
            special_token_ids,
            special_tokens_desc: special_tokens,
            byte_to_unicode,
            unicode_to_byte,
        })
    }

    pub fn encode(&self, text: &str, parse_special: bool) -> Result<Vec<u32>, String> {
        let mut out = Vec::new();
        if self.spec.add_bos_token {
            if let Some(id) = self.spec.bos_token_id {
                out.push(id);
            }
        }

        let mut pos = 0usize;
        while pos < text.len() {
            if parse_special {
                if let Some((token, id)) = self.match_special(text, pos) {
                    out.push(id);
                    pos += token.len();
                    continue;
                }
            }

            let next_special = if parse_special {
                self.find_next_special(text, pos).unwrap_or(text.len())
            } else {
                text.len()
            };
            if next_special == pos {
                let ch = text[pos..].chars().next().ok_or("invalid UTF-8 boundary")?;
                self.encode_raw(&ch.to_string(), &mut out)?;
                pos += ch.len_utf8();
            } else {
                self.encode_raw(&text[pos..next_special], &mut out)?;
                pos = next_special;
            }
        }

        Ok(out)
    }

    pub fn decode(&self, ids: &[u32], include_special: bool) -> String {
        let mut out = String::new();
        let mut state = DecodeState::default();
        for &id in ids {
            out.push_str(&self.decode_one(id, include_special, &mut state));
        }
        out.push_str(&self.finish_decode(&mut state));
        out
    }

    pub fn decode_one(&self, id: u32, include_special: bool, state: &mut DecodeState) -> String {
        let Some(piece) = self.tokens.get(id as usize) else {
            return String::new();
        };
        let mut out = String::new();
        if self.special_token_ids.contains(&id) {
            out.push_str(&finish_decode_state(state));
            if include_special {
                out.push_str(piece);
            }
            return out;
        }

        for ch in piece.chars() {
            let mut buf = [0u8; 4];
            let key = ch.encode_utf8(&mut buf);
            if let Some(byte) = self.unicode_to_byte.get(key) {
                state.byte_buf.push(*byte);
                out.push_str(&flush_valid_utf8(state));
            } else {
                out.push_str(&finish_decode_state(state));
                out.push(ch);
            }
        }
        out
    }

    pub fn finish_decode(&self, state: &mut DecodeState) -> String {
        finish_decode_state(state)
    }

    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// True when the token's surface text ends with a newline — a natural
    /// sentence/paragraph boundary. Used by the thinking-token budget to close
    /// a reasoning block at the end of a sentence within its grace window
    /// rather than mid-word.
    pub fn token_ends_with_newline(&self, id: u32) -> bool {
        let mut state = DecodeState::default();
        self.decode_one(id, false, &mut state).ends_with('\n')
    }

    pub fn token_type(&self, id: u32) -> Option<i32> {
        self.token_types.get(id as usize).copied()
    }

    pub fn special_token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn stop_token_ids(&self) -> HashSet<u32> {
        let mut ids = HashSet::new();
        if let Some(id) = self.spec.eos_token_id {
            ids.insert(id);
        }
        if let Some(id) = self.spec.im_end_token_id {
            ids.insert(id);
        }
        for token in ["<|endoftext|>", "<|eot_id|>", "<|im_end|>"] {
            if let Some(id) = self.special_token_id(token) {
                ids.insert(id);
            }
        }
        ids
    }

    fn encode_raw(&self, text: &str, out: &mut Vec<u32>) -> Result<(), String> {
        for piece in qwen35_pretokenize(text) {
            let encoded = self.byte_encode(piece.as_bytes());
            for token in self.bpe(&encoded) {
                if let Some(id) = self.token_to_id.get(&token) {
                    out.push(*id);
                } else {
                    self.encode_fallback_piece(&token, out)?;
                }
            }
        }
        Ok(())
    }

    fn bpe(&self, encoded: &str) -> Vec<String> {
        let mut symbols: Vec<String> = encoded.chars().map(|ch| ch.to_string()).collect();
        if symbols.len() <= 1 {
            return symbols;
        }

        loop {
            let mut best: Option<(usize, u32)> = None;
            for idx in 0..symbols.len().saturating_sub(1) {
                let key = (symbols[idx].clone(), symbols[idx + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&key) {
                    match best {
                        Some((_, best_rank)) if best_rank <= rank => {}
                        _ => best = Some((idx, rank)),
                    }
                }
            }
            let Some((idx, _)) = best else {
                break;
            };
            let right = symbols.remove(idx + 1);
            symbols[idx].push_str(&right);
        }

        symbols
    }

    fn encode_fallback_piece(&self, piece: &str, out: &mut Vec<u32>) -> Result<(), String> {
        for ch in piece.chars() {
            let token = ch.to_string();
            if let Some(id) = self.token_to_id.get(&token) {
                out.push(*id);
            } else {
                return Err(format!("tokenizer has no token for BPE piece {token:?}"));
            }
        }
        Ok(())
    }

    fn byte_encode(&self, bytes: &[u8]) -> String {
        let mut out = String::new();
        for &byte in bytes {
            out.push_str(&self.byte_to_unicode[byte as usize]);
        }
        out
    }

    fn match_special<'a>(&'a self, text: &'a str, pos: usize) -> Option<(&'a str, u32)> {
        let tail = text.get(pos..)?;
        self.special_tokens_desc
            .iter()
            .find_map(|(token, id)| tail.starts_with(token).then_some((token.as_str(), *id)))
    }

    fn find_next_special(&self, text: &str, pos: usize) -> Option<usize> {
        self.special_tokens_desc
            .iter()
            .filter_map(|(token, _)| text[pos..].find(token).map(|idx| pos + idx))
            .min()
    }
}

fn flush_valid_utf8(state: &mut DecodeState) -> String {
    if state.byte_buf.is_empty() {
        return String::new();
    }

    match std::str::from_utf8(&state.byte_buf) {
        Ok(text) => {
            let out = text.to_string();
            state.byte_buf.clear();
            out
        }
        Err(err) => {
            let valid_up_to = err.valid_up_to();
            if valid_up_to > 0 {
                let out = String::from_utf8_lossy(&state.byte_buf[..valid_up_to]).into_owned();
                state.byte_buf.drain(..valid_up_to);
                return out;
            }
            if let Some(error_len) = err.error_len() {
                state.byte_buf.drain(..error_len);
                "\u{fffd}".to_string()
            } else {
                String::new()
            }
        }
    }
}

fn finish_decode_state(state: &mut DecodeState) -> String {
    if state.byte_buf.is_empty() {
        String::new()
    } else {
        let out = String::from_utf8_lossy(&state.byte_buf).into_owned();
        state.byte_buf.clear();
        out
    }
}

fn is_special_token_text(token: &str) -> bool {
    (token.starts_with("<|") && token.ends_with("|>"))
        || (token.starts_with('<') && token.ends_with('>') && token.len() > 2)
}

fn qwen35_pretokenize(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut pieces = Vec::new();
    let mut pos = 0usize;

    while pos < chars.len() {
        let start = chars[pos].0;
        let c = chars[pos].1;

        if c == '\'' {
            if let Some(end) = contraction_end(&chars, pos) {
                pieces.push(&text[start..byte_pos(text, &chars, end)]);
                pos = end;
                continue;
            }
        }

        if c != '\r' && c != '\n' && !is_number(c) && (is_letter_or_mark(c) || chars
                    .get(pos + 1)
                    .map(|(_, ch)| is_letter_or_mark(*ch))
                    .unwrap_or(false)) {
            pos += 1;
            while chars
                .get(pos)
                .map(|(_, ch)| is_letter_or_mark(*ch))
                .unwrap_or(false)
            {
                pos += 1;
            }
            pieces.push(&text[start..byte_pos(text, &chars, pos)]);
            continue;
        }

        if is_number(c) {
            pos += 1;
            pieces.push(&text[start..byte_pos(text, &chars, pos)]);
            continue;
        }

        let flags2 = if c == ' ' {
            chars.get(pos + 1).map(|(_, ch)| *ch)
        } else {
            Some(c)
        };
        if let Some(ch2) = flags2 {
            if !is_whitespace(ch2) && !is_letter_or_mark(ch2) && !is_number(ch2) {
                if c == ' ' {
                    pos += 1;
                }
                while chars
                    .get(pos)
                    .map(|(_, ch)| {
                        !is_whitespace(*ch) && !is_letter_or_mark(*ch) && !is_number(*ch)
                    })
                    .unwrap_or(false)
                {
                    pos += 1;
                }
                while chars
                    .get(pos)
                    .map(|(_, ch)| *ch == '\r' || *ch == '\n')
                    .unwrap_or(false)
                {
                    pos += 1;
                }
                pieces.push(&text[start..byte_pos(text, &chars, pos)]);
                continue;
            }
        }

        if is_whitespace(c) {
            let mut scan = pos;
            let mut last_end_r_or_n = None;
            while chars
                .get(scan)
                .map(|(_, ch)| is_whitespace(*ch))
                .unwrap_or(false)
            {
                let ch = chars[scan].1;
                scan += 1;
                if ch == '\r' || ch == '\n' {
                    last_end_r_or_n = Some(scan);
                }
            }
            if let Some(end) = last_end_r_or_n {
                pieces.push(&text[start..byte_pos(text, &chars, end)]);
                pos = end;
                continue;
            }
            if scan - pos > 1 && scan < chars.len() {
                let end = scan - 1;
                pieces.push(&text[start..byte_pos(text, &chars, end)]);
                pos = end;
                continue;
            }
            pieces.push(&text[start..byte_pos(text, &chars, scan)]);
            pos = scan;
            continue;
        }

        pos += 1;
        pieces.push(&text[start..byte_pos(text, &chars, pos)]);
    }

    pieces
}

fn contraction_end(chars: &[(usize, char)], pos: usize) -> Option<usize> {
    let c1 = chars.get(pos + 1)?.1.to_ascii_lowercase();
    if matches!(c1, 's' | 't' | 'm' | 'd') {
        return Some(pos + 2);
    }
    let c2 = chars.get(pos + 2)?.1.to_ascii_lowercase();
    if matches!((c1, c2), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
        return Some(pos + 3);
    }
    None
}

fn byte_pos(text: &str, chars: &[(usize, char)], char_pos: usize) -> usize {
    chars
        .get(char_pos)
        .map(|(idx, _)| *idx)
        .unwrap_or(text.len())
}

fn is_letter_or_mark(ch: char) -> bool {
    ch.is_alphabetic() || is_combining_mark(ch)
}

fn is_number(ch: char) -> bool {
    ch.is_numeric()
}

fn is_whitespace(ch: char) -> bool {
    ch.is_whitespace()
}

fn is_combining_mark(ch: char) -> bool {
    matches!(
        ch as u32,
        0x0300..=0x036F
            | 0x1AB0..=0x1AFF
            | 0x1DC0..=0x1DFF
            | 0x20D0..=0x20FF
            | 0xFE20..=0xFE2F
    )
}

fn byte_maps() -> ([String; 256], HashMap<String, u8>) {
    let mut byte_to_unicode: [String; 256] = std::array::from_fn(|_| String::new());
    let mut unicode_to_byte = HashMap::with_capacity(256);
    let mut used = [false; 256];

    for ch in 0x21u32..=0x7E {
        insert_byte_mapping(
            ch as u8,
            ch,
            &mut used,
            &mut byte_to_unicode,
            &mut unicode_to_byte,
        );
    }
    for ch in 0xA1u32..=0xAC {
        insert_byte_mapping(
            ch as u8,
            ch,
            &mut used,
            &mut byte_to_unicode,
            &mut unicode_to_byte,
        );
    }
    for ch in 0xAEu32..=0xFF {
        insert_byte_mapping(
            ch as u8,
            ch,
            &mut used,
            &mut byte_to_unicode,
            &mut unicode_to_byte,
        );
    }

    let mut next = 0u32;
    for byte in 0u16..=255 {
        let byte = byte as u8;
        if used[byte as usize] {
            continue;
        }
        let codepoint = 256 + next;
        next += 1;
        insert_byte_mapping(
            byte,
            codepoint,
            &mut used,
            &mut byte_to_unicode,
            &mut unicode_to_byte,
        );
    }

    (byte_to_unicode, unicode_to_byte)
}

fn insert_byte_mapping(
    byte: u8,
    codepoint: u32,
    used: &mut [bool; 256],
    byte_to_unicode: &mut [String; 256],
    unicode_to_byte: &mut HashMap<String, u8>,
) {
    let text = char::from_u32(codepoint)
        .expect("valid GPT-2 byte encoder codepoint")
        .to_string();
    used[byte as usize] = true;
    byte_to_unicode[byte as usize] = text.clone();
    unicode_to_byte.insert(text, byte);
}

#[cfg(test)]
mod tests {
    use super::{qwen35_pretokenize, DecodeState, QwenTokenizer, QwenTokenizerSpec};
    use crate::loader::MappedGguf;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn loads_qwen35_tokenizer_metadata() {
        let path = write_tiny_tokenizer_gguf();
        let gguf = MappedGguf::open(&path).unwrap();
        let spec = QwenTokenizerSpec::load(&gguf).unwrap();
        assert_eq!(spec.model, "gpt2");
        assert_eq!(spec.pre, "qwen35");
        assert_eq!(spec.vocab_size, 15);
        assert_eq!(spec.merge_count, 5);
        assert_eq!(spec.bos_token_id, Some(1));
        assert_eq!(spec.eos_token_id, Some(2));
        assert_eq!(spec.im_start_token_id, Some(9));
        assert_eq!(spec.im_end_token_id, Some(10));
        assert_eq!(spec.think_token_id, Some(11));
        fs::remove_file(path).ok();
    }

    #[test]
    fn qwen35_pretokenizer_matches_key_shapes() {
        assert_eq!(qwen35_pretokenize("hello world"), vec!["hello", " world"]);
        assert_eq!(qwen35_pretokenize("can't"), vec!["can", "'t"]);
        assert_eq!(qwen35_pretokenize("hi!\n"), vec!["hi", "!\n"]);
        assert_eq!(qwen35_pretokenize("  x"), vec![" ", " x"]);
    }

    #[test]
    fn encodes_decodes_byte_level_bpe_and_special_tokens() {
        let path = write_tiny_tokenizer_gguf();
        let gguf = MappedGguf::open(&path).unwrap();
        let tokenizer = QwenTokenizer::load(&gguf).unwrap();

        let ids = tokenizer.encode(" hello<|im_start|>hello", true).unwrap();
        assert_eq!(ids, vec![8, 9, 7]);
        assert_eq!(tokenizer.decode(&ids, false), " hellohello");
        assert_eq!(tokenizer.decode(&ids, true), " hello<|im_start|>hello");

        let mut state = DecodeState::default();
        let mut streamed = String::new();
        for &id in &ids {
            streamed.push_str(&tokenizer.decode_one(id, false, &mut state));
        }
        streamed.push_str(&tokenizer.finish_decode(&mut state));
        assert_eq!(streamed, tokenizer.decode(&ids, false));

        assert!(tokenizer.stop_token_ids().contains(&2));
        assert!(tokenizer.stop_token_ids().contains(&10));
        fs::remove_file(path).ok();
    }

    #[test]
    fn token_ends_with_newline_detects_sentence_boundary() {
        let path = write_tiny_tokenizer_gguf();
        let gguf = MappedGguf::open(&path).unwrap();
        let tokenizer = QwenTokenizer::load(&gguf).unwrap();

        // "!Ċ" decodes to "!\n" — a sentence boundary; "hello" is not.
        let newline_id = tokenizer.special_token_id("!Ċ").unwrap();
        let word_id = tokenizer.special_token_id("hello").unwrap();
        assert!(tokenizer.token_ends_with_newline(newline_id));
        assert!(!tokenizer.token_ends_with_newline(word_id));
        fs::remove_file(path).ok();
    }

    fn write_tiny_tokenizer_gguf() -> PathBuf {
        const GGUF_MAGIC: u32 = 0x4655_4747;

        let mut bytes = Vec::new();
        put_u32(&mut bytes, GGUF_MAGIC);
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 0);
        put_u64(&mut bytes, 10);

        put_kv_string(&mut bytes, "tokenizer.ggml.model", "gpt2");
        put_kv_string(&mut bytes, "tokenizer.ggml.pre", "qwen35");
        put_string_array(
            &mut bytes,
            "tokenizer.ggml.tokens",
            &[
                "!",
                "<bos>",
                "<eos>",
                "h",
                "e",
                "he",
                "l",
                "hello",
                "Ġhello",
                "<|im_start|>",
                "<|im_end|>",
                "<think>",
                "</think>",
                "Ġ",
                "!Ċ",
            ],
        );
        put_i32_array(
            &mut bytes,
            "tokenizer.ggml.token_type",
            &[1, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1],
        );
        put_string_array(
            &mut bytes,
            "tokenizer.ggml.merges",
            &["h e", "he l", "hel l", "hell o", "Ġ hello"],
        );
        put_kv_u32(&mut bytes, "tokenizer.ggml.padding_token_id", 0);
        put_kv_u32(&mut bytes, "tokenizer.ggml.bos_token_id", 1);
        put_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 2);
        put_kv_bool(&mut bytes, "tokenizer.ggml.add_bos_token", false);
        put_kv_u32(&mut bytes, "general.alignment", 32);

        let path = std::env::temp_dir().join(format!(
            "qw35-tokenizer-{}-{}.gguf",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(&path, bytes).unwrap();
        path
    }

    fn put_kv_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
        const VALUE_STRING: u32 = 8;
        put_string(bytes, key);
        put_u32(bytes, VALUE_STRING);
        put_string(bytes, value);
    }

    fn put_kv_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
        const VALUE_UINT32: u32 = 4;
        put_string(bytes, key);
        put_u32(bytes, VALUE_UINT32);
        put_u32(bytes, value);
    }

    fn put_kv_bool(bytes: &mut Vec<u8>, key: &str, value: bool) {
        const VALUE_BOOL: u32 = 7;
        put_string(bytes, key);
        put_u32(bytes, VALUE_BOOL);
        bytes.push(u8::from(value));
    }

    fn put_string_array(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
        const VALUE_STRING: u32 = 8;
        const VALUE_ARRAY: u32 = 9;
        put_string(bytes, key);
        put_u32(bytes, VALUE_ARRAY);
        put_u32(bytes, VALUE_STRING);
        put_u64(bytes, values.len() as u64);
        for value in values {
            put_string(bytes, value);
        }
    }

    fn put_i32_array(bytes: &mut Vec<u8>, key: &str, values: &[i32]) {
        const VALUE_INT32: u32 = 5;
        const VALUE_ARRAY: u32 = 9;
        put_string(bytes, key);
        put_u32(bytes, VALUE_ARRAY);
        put_u32(bytes, VALUE_INT32);
        put_u64(bytes, values.len() as u64);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn put_string(bytes: &mut Vec<u8>, value: &str) {
        put_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn put_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}
