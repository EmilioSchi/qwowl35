    use super::{plan_qwen35, qwen_metal_type_supported};
    use crate::loader::MappedGguf;
    use std::collections::BTreeMap;
    use std::path::Path;

    #[test]
    fn supports_real_qwen35_gguf_quant_formats() {
        for name in ["f32", "q4_k", "q5_k", "q6_k", "q8_0", "gf4"] {
            assert!(qwen_metal_type_supported(name), "{name}");
        }

        assert!(!qwen_metal_type_supported("q3_k"));
        assert!(!qwen_metal_type_supported("iq1_s"));
        // gf2 has a decode matvec but no tiled prefill kernel yet.
        assert!(!qwen_metal_type_supported("gf2"));
    }

    #[test]
    fn real_qwen35_gguf_graph_is_ready_when_present() {
        let path = Path::new(".gguf/Qwen3.5-9B-Q4_K_M.gguf");
        if !path.exists() {
            return;
        }

        let gguf = MappedGguf::open(path).unwrap();
        let plan = plan_qwen35(&gguf);
        assert_eq!(gguf.tensors.len(), 427);
        assert_eq!(plan.hparams.block_count, 32);
        assert_eq!(plan.hparams.transformer_layers, 32);
        assert_eq!(plan.hparams.nextn_predict_layers, 0);
        assert_eq!(plan.delta_layers.len(), 24);
        assert_eq!(plan.attention_layers.len(), 8);
        assert!(
            plan.missing_tensors.is_empty(),
            "{:?}",
            plan.missing_tensors
        );
        assert!(
            plan.unsupported_tensor_types.is_empty(),
            "{:?}",
            plan.unsupported_tensor_types
        );
        assert!(plan.decoder_ready());
    }

    #[test]
    fn real_qwen35_gguf_tensor_type_counts_match_supported_formats() {
        let path = Path::new(".gguf/Qwen3.5-9B-Q4_K_M.gguf");
        if !path.exists() {
            return;
        }

        let gguf = MappedGguf::open(path).unwrap();
        let plan = plan_qwen35(&gguf);
        let counts: BTreeMap<_, _> = plan.tensor_type_counts.iter().cloned().collect();
        assert_eq!(counts.get("f32"), Some(&177));
        assert_eq!(counts.get("q4_k"), Some(&132));
        assert_eq!(counts.get("q5_k"), Some(&48));
        assert_eq!(counts.get("q6_k"), Some(&22));
        assert_eq!(counts.get("q8_0"), Some(&48));
        assert!(plan.unsupported_tensor_types.is_empty());
    }
