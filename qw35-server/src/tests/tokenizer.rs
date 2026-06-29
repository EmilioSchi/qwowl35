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
