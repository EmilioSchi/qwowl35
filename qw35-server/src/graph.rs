use crate::loader::{tensor_type_name, MappedGguf};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct GraphPlan {
    pub hparams: QwenHparams,
    pub delta_layers: Vec<u32>,
    pub attention_layers: Vec<u32>,
    pub missing_tensors: Vec<String>,
    pub tensor_type_counts: Vec<(String, u64)>,
    pub unsupported_tensor_types: Vec<String>,
    pub execution_blockers: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct QwenHparams {
    pub block_count: u32,
    pub transformer_layers: u32,
    pub nextn_predict_layers: u32,
    pub embedding_length: u32,
    pub feed_forward_length: u32,
    pub attention_heads: u32,
    pub attention_kv_heads: u32,
    pub attention_key_length: u32,
    pub attention_value_length: u32,
    pub rope_dimension_count: u32,
    pub rope_dimension_sections: Vec<i32>,
    pub rope_freq_base: f32,
    pub rms_epsilon: f32,
    pub ssm_conv_kernel: u32,
    pub ssm_state_size: u32,
    pub ssm_group_count: u32,
    pub ssm_time_step_rank: u32,
    pub ssm_inner_size: u32,
    pub full_attention_interval: u32,
}

impl GraphPlan {
    pub fn decoder_ready(&self) -> bool {
        self.missing_tensors.is_empty()
            && self.unsupported_tensor_types.is_empty()
            && self.execution_blockers.is_empty()
    }

    pub fn blocker_summary(&self) -> String {
        let mut parts = Vec::new();
        if !self.missing_tensors.is_empty() {
            parts.push(format!(
                "{} expected tensors are missing",
                self.missing_tensors.len()
            ));
        }
        if !self.unsupported_tensor_types.is_empty() {
            parts.push(format!(
                "missing Metal kernels for tensor formats {}",
                self.unsupported_tensor_types.join(", ")
            ));
        }
        parts.extend(self.execution_blockers.iter().cloned());
        parts.join("; ")
    }
}

pub fn plan_qwen35(gguf: &MappedGguf) -> GraphPlan {
    let hparams = read_hparams(gguf);
    let mut missing_tensors = Vec::new();

    require_tensor(gguf, "token_embd.weight", &mut missing_tensors);
    require_tensor(gguf, "output_norm.weight", &mut missing_tensors);
    require_tensor(gguf, "output.weight", &mut missing_tensors);

    let mut delta_layers = Vec::new();
    let mut attention_layers = Vec::new();
    for layer in 0..hparams.transformer_layers {
        if (layer + 1) % hparams.full_attention_interval.max(1) == 0 {
            attention_layers.push(layer);
            require_attention_layer(gguf, layer, &mut missing_tensors);
        } else {
            delta_layers.push(layer);
            require_delta_layer(gguf, layer, &mut missing_tensors);
        }
        require_ffn(gguf, layer, &mut missing_tensors);
    }

    let mut counts = BTreeMap::<String, u64>::new();
    for tensor in &gguf.tensors {
        *counts
            .entry(tensor_type_name(tensor.type_id).to_string())
            .or_insert(0) += 1;
    }
    let tensor_type_counts: Vec<_> = counts.into_iter().collect();
    let unsupported_tensor_types = tensor_type_counts
        .iter()
        .filter_map(|(name, _)| {
            if qwen_metal_type_supported(name) {
                None
            } else {
                Some(name.clone())
            }
        })
        .collect();

    GraphPlan {
        hparams,
        delta_layers,
        attention_layers,
        missing_tensors,
        tensor_type_counts,
        unsupported_tensor_types,
        execution_blockers: Vec::new(),
    }
}

fn read_hparams(gguf: &MappedGguf) -> QwenHparams {
    let block_count = gguf.metadata_u32("qwen35.block_count").unwrap_or(32);
    let nextn_predict_layers = gguf
        .metadata_u32("qwen35.nextn_predict_layers")
        .unwrap_or(0);
    QwenHparams {
        block_count,
        transformer_layers: block_count.saturating_sub(nextn_predict_layers),
        nextn_predict_layers,
        embedding_length: gguf.metadata_u32("qwen35.embedding_length").unwrap_or(4096),
        feed_forward_length: gguf
            .metadata_u32("qwen35.feed_forward_length")
            .unwrap_or(12_288),
        attention_heads: gguf
            .metadata_u32("qwen35.attention.head_count")
            .unwrap_or(16),
        attention_kv_heads: gguf
            .metadata_u32("qwen35.attention.head_count_kv")
            .unwrap_or(4),
        attention_key_length: gguf
            .metadata_u32("qwen35.attention.key_length")
            .unwrap_or(256),
        attention_value_length: gguf
            .metadata_u32("qwen35.attention.value_length")
            .unwrap_or(256),
        rope_dimension_count: gguf
            .metadata_u32("qwen35.rope.dimension_count")
            .unwrap_or(64),
        rope_dimension_sections: gguf
            .metadata_array_i32("qwen35.rope.dimension_sections")
            .unwrap_or_else(|| vec![11, 11, 10, 0]),
        rope_freq_base: gguf
            .metadata_f32("qwen35.rope.freq_base")
            .unwrap_or(10_000_000.0),
        rms_epsilon: gguf
            .metadata_f32("qwen35.attention.layer_norm_rms_epsilon")
            .unwrap_or(1.0e-6),
        ssm_conv_kernel: gguf.metadata_u32("qwen35.ssm.conv_kernel").unwrap_or(4),
        ssm_state_size: gguf.metadata_u32("qwen35.ssm.state_size").unwrap_or(128),
        ssm_group_count: gguf.metadata_u32("qwen35.ssm.group_count").unwrap_or(16),
        ssm_time_step_rank: gguf.metadata_u32("qwen35.ssm.time_step_rank").unwrap_or(32),
        ssm_inner_size: gguf.metadata_u32("qwen35.ssm.inner_size").unwrap_or(4096),
        full_attention_interval: gguf
            .metadata_u32("qwen35.full_attention_interval")
            .unwrap_or(4),
    }
}

fn require_delta_layer(gguf: &MappedGguf, layer: u32, missing: &mut Vec<String>) {
    for suffix in [
        "attn_gate.weight",
        "attn_norm.weight",
        "attn_qkv.weight",
        "post_attention_norm.weight",
        "ssm_a",
        "ssm_alpha.weight",
        "ssm_beta.weight",
        "ssm_conv1d.weight",
        "ssm_dt.bias",
        "ssm_norm.weight",
        "ssm_out.weight",
    ] {
        require_tensor(gguf, &format!("blk.{layer}.{suffix}"), missing);
    }
}

fn require_attention_layer(gguf: &MappedGguf, layer: u32, missing: &mut Vec<String>) {
    for suffix in [
        "attn_k.weight",
        "attn_k_norm.weight",
        "attn_norm.weight",
        "attn_output.weight",
        "attn_q.weight",
        "attn_q_norm.weight",
        "attn_v.weight",
        "post_attention_norm.weight",
    ] {
        require_tensor(gguf, &format!("blk.{layer}.{suffix}"), missing);
    }
}

fn require_ffn(gguf: &MappedGguf, layer: u32, missing: &mut Vec<String>) {
    for suffix in ["ffn_down.weight", "ffn_gate.weight", "ffn_up.weight"] {
        require_tensor(gguf, &format!("blk.{layer}.{suffix}"), missing);
    }
}

fn require_tensor(gguf: &MappedGguf, name: &str, missing: &mut Vec<String>) {
    if gguf.tensor(name).is_none() {
        missing.push(name.to_string());
    }
}

fn qwen_metal_type_supported(name: &str) -> bool {
    // "gf4" is the unified .gguf FFN codec: both the single-token decode matvec
    // and the tiled multi-token prefill matmul have GF4 kernels. ("gf2" has no
    // tiled prefill kernel, so it is intentionally not prefill-capable here.)
    matches!(name, "f32" | "q4_k" | "q5_k" | "q6_k" | "q8_0" | "gf4")
}

#[cfg(test)]
mod tests {
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
}
