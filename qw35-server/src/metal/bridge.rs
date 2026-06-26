use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::Duration;

use crate::loader::{MappedGguf, WarmReport};
use crate::graph::QwenHparams;

pub const WARMUP_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qw35.metallib"));

/// What the output head computes for an eval call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogitsMode {
    /// No output head (non-final prefill chunk).
    None,
    /// Fused argmax only; the logits vector is never materialized.
    Argmax,
    /// Full logits vector for CPU sampling.
    Full,
}

impl LogitsMode {
    fn as_i32(self) -> i32 {
        match self {
            LogitsMode::None => 0,
            LogitsMode::Argmax => 1,
            LogitsMode::Full => 2,
        }
    }
}

/// Storage format of the attention KV cache. Two encodings are supported:
/// `q8_0` (the default — symmetric 8-bit, byte-identical to f16 output at
/// fp16-parity decode speed and half the memory) and `f16` (the lossless
/// reference, kept for comparing future KV-cache optimizations).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheType {
    /// 16-bit float cache rows (2 bytes per element).
    F16,
    /// q8_0 block-quantized rows (34-byte blocks of 32 elements, 8.5 bpw):
    /// symmetric signed int8 + one fp16 scale per block (llama.cpp/ggml layout).
    Q8_0,
}

impl KvCacheType {
    fn as_u32(self) -> u32 {
        match self {
            KvCacheType::F16 => 0,
            KvCacheType::Q8_0 => 1,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            KvCacheType::F16 => "f16",
            KvCacheType::Q8_0 => "q8_0",
        }
    }

    /// Parse a CLI/string value into a KV cache type.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "f16" => Some(KvCacheType::F16),
            "q8_0" => Some(KvCacheType::Q8_0),
            _ => None,
        }
    }

    /// Bytes per K (or V) cache row for one position across one layer.
    pub fn row_bytes(self, kv_dim: u64) -> u64 {
        match self {
            KvCacheType::F16 => kv_dim * 2,
            KvCacheType::Q8_0 => kv_dim / 32 * 34,
        }
    }
}

#[cfg(target_os = "macos")]
extern "C" {
    fn qw35_metal_has_device() -> i32;
    fn qw35_metal_warm_model_views(
        model_map: *const c_void,
        model_size: u64,
        map_offset: u64,
        map_size: u64,
        max_tensor_bytes: u64,
        metallib: *const u8,
        metallib_len: usize,
        stride: u64,
        report: *mut MetalWarmReport,
        err: *mut i8,
        err_len: usize,
    ) -> i32;
    fn qw35_metal_runtime_create(
        model_map: *const c_void,
        model_size: u64,
        tensors: *const MetalTensorDesc,
        tensor_count: usize,
        hparams: *const MetalHparams,
        ctx_size: u32,
        vocab_size: u32,
        prefill_chunk: u32,
        kv_cache_type: u32,
        metallib: *const u8,
        metallib_len: usize,
        err: *mut i8,
        err_len: usize,
    ) -> *mut c_void;
    fn qw35_metal_runtime_destroy(runtime: *mut c_void);
    fn qw35_metal_runtime_reset(runtime: *mut c_void, err: *mut i8, err_len: usize) -> i32;
    fn qw35_metal_runtime_state_checkpoint_save(
        runtime: *mut c_void,
        err: *mut i8,
        err_len: usize,
    ) -> i32;
    fn qw35_metal_runtime_state_checkpoint_restore(
        runtime: *mut c_void,
        err: *mut i8,
        err_len: usize,
    ) -> i32;
    fn qw35_metal_runtime_sync(runtime: *mut c_void, err: *mut i8, err_len: usize) -> i32;
    fn qw35_metal_runtime_set_attn_sink(runtime: *mut c_void, sink: i32);
    fn qw35_metal_runtime_eval_token(
        runtime: *mut c_void,
        token: u32,
        pos: u32,
        logits_mode: i32,
        err: *mut i8,
        err_len: usize,
    ) -> i32;
    fn qw35_metal_runtime_eval_tokens(
        runtime: *mut c_void,
        tokens: *const u32,
        len: usize,
        pos0: u32,
        logits_mode: i32,
        err: *mut i8,
        err_len: usize,
    ) -> i32;
    fn qw35_metal_runtime_argmax(
        runtime: *mut c_void,
        token: *mut u32,
        logit: *mut f32,
        err: *mut i8,
        err_len: usize,
    ) -> i32;
    fn qw35_metal_runtime_copy_logits(
        runtime: *mut c_void,
        dst: *mut f32,
        len: usize,
        err: *mut i8,
        err_len: usize,
    ) -> i32;
}

pub fn require_metal_device() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        if unsafe { qw35_metal_has_device() } == 0 {
            return Err("Metal is required, but no default Metal device is available".to_string());
        }
        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    {
        Err("qw35 inference is Metal-only and currently requires macOS".to_string())
    }
}

pub fn warm_model_views(gguf: &MappedGguf) -> Result<WarmReport, String> {
    #[cfg(target_os = "macos")]
    {
        let mut report = MetalWarmReport::default();
        let mut err = vec![0i8; 1024];
        let ok = unsafe {
            qw35_metal_warm_model_views(
                gguf.as_ptr().cast::<c_void>(),
                gguf.len(),
                gguf.tensor_data_pos,
                gguf.len().saturating_sub(gguf.tensor_data_pos),
                gguf.max_tensor_bytes(),
                WARMUP_METALLIB.as_ptr(),
                WARMUP_METALLIB.len(),
                1024 * 1024,
                &mut report,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if ok == 0 {
            return Err(c_string_from_i8(&err));
        }
        Ok(WarmReport {
            mode: "metal",
            bytes: report.bytes,
            mapped_bytes: report.mapped_bytes,
            page_size: 0,
            touched_pages: report.touches,
            view_count: report.view_count,
            checksum: report.checksum,
            elapsed: Duration::from_secs_f64(report.elapsed_ms / 1000.0),
            madvise_error: None,
        })
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = gguf;
        Err("Metal model-view warmup requires macOS".to_string())
    }
}

pub fn verify_native_qwen_kernels() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        // Defensive guard against an empty embedded metallib (build
        // misconfiguration). `is_empty()` is const-false for the normal build.
        #[allow(clippy::const_is_empty)]
        if WARMUP_METALLIB.is_empty() {
            return Err("Qw35 Metal library was not built".to_string());
        }
        for name in REQUIRED_NATIVE_KERNELS {
            if !contains_bytes(WARMUP_METALLIB, name.as_bytes()) {
                return Err(format!(
                    "Qw35 Metal library is missing native kernel {name}"
                ));
            }
        }
        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    {
        Err("Qw35 native Metal kernel verification requires macOS".to_string())
    }
}

pub struct MetalRuntime {
    #[cfg(target_os = "macos")]
    raw: NonNull<c_void>,
    vocab_size: usize,
}

unsafe impl Send for MetalRuntime {}

impl MetalRuntime {
    pub fn new(
        gguf: &MappedGguf,
        hparams: &QwenHparams,
        ctx_size: u32,
        vocab_size: u32,
        prefill_chunk: u32,
        kv_cache_type: KvCacheType,
        attn_window: i32,
        attn_sink: i32,
    ) -> Result<Self, String> {
        #[cfg(target_os = "macos")]
        {
            // The unified .gguf carries its GF4 FFN as type-id 100 tensors in
            // the normal tensor table, so a single descriptor list covers every
            // weight (the type-id selects GF4 vs Q4_K kernels per tensor).
            let tensors: Vec<MetalTensorDesc> = gguf
                .tensors
                .iter()
                .map(|tensor| {
                    let mut dims = [1u64; 4];
                    for (idx, dim) in tensor.dims.iter().take(4).enumerate() {
                        dims[idx] = *dim;
                    }
                    MetalTensorDesc {
                        name: tensor.name.as_ptr(),
                        name_len: tensor.name.len(),
                        dims,
                        n_dims: tensor.dims.len() as u32,
                        type_id: tensor.type_id,
                        abs_offset: tensor.abs_offset,
                        bytes: tensor.bytes,
                        elements: tensor.elements,
                    }
                })
                .collect();
            let mut hparams = MetalHparams::from(hparams);
            hparams.attn_window = attn_window;
            hparams.attn_sink = attn_sink;
            let mut err = vec![0i8; 4096];
            let raw = unsafe {
                qw35_metal_runtime_create(
                    gguf.as_ptr().cast::<c_void>(),
                    gguf.len(),
                    tensors.as_ptr(),
                    tensors.len(),
                    &hparams,
                    ctx_size,
                    vocab_size,
                    prefill_chunk,
                    kv_cache_type.as_u32(),
                    WARMUP_METALLIB.as_ptr(),
                    WARMUP_METALLIB.len(),
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            let raw = NonNull::new(raw).ok_or_else(|| c_string_from_i8(&err))?;
            Ok(Self {
                raw,
                vocab_size: vocab_size as usize,
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = (gguf, hparams, ctx_size, prefill_chunk, kv_cache_type);
            Ok(Self {
                vocab_size: vocab_size as usize,
            })
        }
    }

    /// Snapshot the SSM/conv recurrent state into the runtime's internal
    /// checkpoint slot. KV cache rows are positional and need no snapshot.
    pub fn state_checkpoint_save(&mut self) -> Result<(), String> {
        #[cfg(target_os = "macos")]
        {
            let mut err = vec![0i8; 1024];
            let ok = unsafe {
                qw35_metal_runtime_state_checkpoint_save(
                    self.raw.as_ptr(),
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            if ok == 0 {
                return Err(c_string_from_i8(&err));
            }
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            Err("Qw35 native state checkpoint requires macOS Metal".to_string())
        }
    }

    /// Restore the SSM/conv recurrent state from the internal checkpoint slot.
    pub fn state_checkpoint_restore(&mut self) -> Result<(), String> {
        #[cfg(target_os = "macos")]
        {
            let mut err = vec![0i8; 1024];
            let ok = unsafe {
                qw35_metal_runtime_state_checkpoint_restore(
                    self.raw.as_ptr(),
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            if ok == 0 {
                return Err(c_string_from_i8(&err));
            }
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            Err("Qw35 native state checkpoint requires macOS Metal".to_string())
        }
    }

    pub fn reset(&mut self) -> Result<(), String> {
        #[cfg(target_os = "macos")]
        {
            let mut err = vec![0i8; 1024];
            let ok =
                unsafe { qw35_metal_runtime_reset(self.raw.as_ptr(), err.as_mut_ptr(), err.len()) };
            if ok == 0 {
                return Err(c_string_from_i8(&err));
            }
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            Err("Qw35 native runtime reset requires macOS Metal".to_string())
        }
    }

    /// Set the decode-time sliding-window attention sink (first N positions
    /// always attended). Set per request to pin the system prompt + first user
    /// turn so the window never evicts the tool-call format or the task.
    pub fn set_attn_sink(&mut self, sink: i32) {
        #[cfg(target_os = "macos")]
        unsafe {
            qw35_metal_runtime_set_attn_sink(self.raw.as_ptr(), sink);
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sink;
        }
    }

    /// Wait for all in-flight GPU work to complete.
    pub fn sync(&mut self) -> Result<(), String> {
        #[cfg(target_os = "macos")]
        {
            let mut err = vec![0i8; 1024];
            let ok =
                unsafe { qw35_metal_runtime_sync(self.raw.as_ptr(), err.as_mut_ptr(), err.len()) };
            if ok == 0 {
                return Err(c_string_from_i8(&err));
            }
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            Err("Qw35 native runtime sync requires macOS Metal".to_string())
        }
    }

    pub fn eval_token(&mut self, token: u32, pos: u32, logits: LogitsMode) -> Result<(), String> {
        #[cfg(target_os = "macos")]
        {
            let mut err = vec![0i8; 4096];
            let ok = unsafe {
                qw35_metal_runtime_eval_token(
                    self.raw.as_ptr(),
                    token,
                    pos,
                    logits.as_i32(),
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            if ok == 0 {
                return Err(c_string_from_i8(&err));
            }
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = (token, pos, logits);
            Err("Qw35 native token evaluation requires macOS Metal".to_string())
        }
    }

    pub fn eval_prefill_chunk(
        &mut self,
        tokens: &[u32],
        pos0: u32,
        logits: LogitsMode,
    ) -> Result<(), String> {
        if tokens.is_empty() {
            return Ok(());
        }
        #[cfg(target_os = "macos")]
        {
            let mut err = vec![0i8; 4096];
            let ok = unsafe {
                qw35_metal_runtime_eval_tokens(
                    self.raw.as_ptr(),
                    tokens.as_ptr(),
                    tokens.len(),
                    pos0,
                    logits.as_i32(),
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            if ok == 0 {
                return Err(c_string_from_i8(&err));
            }
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = (tokens, pos0, logits);
            Err("Qw35 native chunk prefill requires macOS Metal".to_string())
        }
    }

    pub fn argmax(&mut self) -> Result<(u32, f32), String> {
        #[cfg(target_os = "macos")]
        {
            let mut token = 0u32;
            let mut logit = 0.0f32;
            let mut err = vec![0i8; 1024];
            let ok = unsafe {
                qw35_metal_runtime_argmax(
                    self.raw.as_ptr(),
                    &mut token,
                    &mut logit,
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            if ok == 0 {
                return Err(c_string_from_i8(&err));
            }
            Ok((token, logit))
        }

        #[cfg(not(target_os = "macos"))]
        {
            Err("Qw35 native argmax requires macOS Metal".to_string())
        }
    }

    pub fn copy_logits(&mut self) -> Result<Vec<f32>, String> {
        let mut logits = vec![0.0f32; self.vocab_size];
        #[cfg(target_os = "macos")]
        {
            let mut err = vec![0i8; 1024];
            let ok = unsafe {
                qw35_metal_runtime_copy_logits(
                    self.raw.as_ptr(),
                    logits.as_mut_ptr(),
                    logits.len(),
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            if ok == 0 {
                return Err(c_string_from_i8(&err));
            }
            Ok(logits)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Err("Qw35 native logits readback requires macOS Metal".to_string())
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for MetalRuntime {
    fn drop(&mut self) {
        unsafe {
            qw35_metal_runtime_destroy(self.raw.as_ptr());
        }
    }
}

const REQUIRED_NATIVE_KERNELS: &[&str] = &[
    "qw35_touch_u8_stride",
    "qw35_get_row_q4_k_f32",
    "qw35_get_rows_q4_k_f32",
    "qw35_rms_norm_weight_f32",
    "qw35_rms_norm_weight_batch_f32",
    "qw35_residual_rms_norm_weight_f32",
    "qw35_residual_rms_norm_weight_batch_f32",
    "qw35_swiglu_f32",
    "qw35_decode_matmul_q8_0_f32",
    "qw35_decode_matmul_q4_k_2row_f32",
    "qw35_decode_matmul_q4_k_2row_residual_f32",
    "qw35_decode_matmul_q5_k_2row_f32",
    "qw35_decode_matmul_q6_k_llama_f32",
    "qw35_decode_matmul_q6_k_llama_residual_f32",
    "qw35_mul_mm_q8_0_f32",
    "qw35_mul_mm_q4_k_f32",
    "qw35_mul_mm_q5_k_f32",
    "qw35_mul_mm_q6_k_f32",
    "qw35_mul_mm_gf4_f32",
    "qw35_attn_decode_preprocess_f32",
    "qw35_attn_prefill_preprocess_f32",
    "qw35_attention_gqa_flash_decode_f32",
    "qw35_attention_gqa_prefill_f32",
    "qw35_attn_decode_preprocess_q8_0_f32",
    "qw35_attn_prefill_preprocess_q8_0_f32",
    "qw35_attention_gqa_flash_decode_q8_0_f32",
    "qw35_attention_gqa_prefill_q8_0_f32",
    "qw35_ssm_conv_recurrent_gate_norm_step128_f32",
    "qw35_ssm_conv1d_step4_batch_f32",
    "qw35_ssm_l2_repeat_qk_batch_f32",
    "qw35_ssm_recurrent_step128_batch_rows_f32",
    "qw35_ssm_gate_norm_batch_f32",
    "qw35_output_q6_k_argmax_partials_16row_f32",
    "qw35_output_argmax_reduce_partials_f32",
    "qw35_ffn_gate_up_swiglu_gf4_f32",
    "qw35_decode_matmul_gf4_2row_f32",
    "qw35_decode_matmul_gf4_2row_residual_f32",
    "qw35_output_gf4_argmax_partials_16row_f32",
];

#[cfg(target_os = "macos")]
#[repr(C)]
struct MetalTensorDesc {
    name: *const u8,
    name_len: usize,
    dims: [u64; 4],
    n_dims: u32,
    type_id: u32,
    abs_offset: u64,
    bytes: u64,
    elements: u64,
}

#[cfg(target_os = "macos")]
#[repr(C)]
struct MetalHparams {
    transformer_layers: u32,
    embedding_length: u32,
    feed_forward_length: u32,
    attention_heads: u32,
    attention_kv_heads: u32,
    attention_key_length: u32,
    attention_value_length: u32,
    rope_dimension_count: u32,
    rope_sections: [i32; 4],
    rope_freq_base: f32,
    rms_epsilon: f32,
    ssm_conv_kernel: u32,
    ssm_state_size: u32,
    ssm_group_count: u32,
    ssm_time_step_rank: u32,
    ssm_inner_size: u32,
    full_attention_interval: u32,
    attn_window: i32,
    attn_sink: i32,
}

#[cfg(target_os = "macos")]
impl From<&QwenHparams> for MetalHparams {
    fn from(value: &QwenHparams) -> Self {
        let mut rope_sections = [0i32; 4];
        for (idx, section) in value.rope_dimension_sections.iter().take(4).enumerate() {
            rope_sections[idx] = *section;
        }
        Self {
            transformer_layers: value.transformer_layers,
            embedding_length: value.embedding_length,
            feed_forward_length: value.feed_forward_length,
            attention_heads: value.attention_heads,
            attention_kv_heads: value.attention_kv_heads,
            attention_key_length: value.attention_key_length,
            attention_value_length: value.attention_value_length,
            rope_dimension_count: value.rope_dimension_count,
            rope_sections,
            rope_freq_base: value.rope_freq_base,
            rms_epsilon: value.rms_epsilon,
            ssm_conv_kernel: value.ssm_conv_kernel,
            ssm_state_size: value.ssm_state_size,
            ssm_group_count: value.ssm_group_count,
            ssm_time_step_rank: value.ssm_time_step_rank,
            ssm_inner_size: value.ssm_inner_size,
            full_attention_interval: value.full_attention_interval,
            // Filled in by MetalRuntime::new from EngineConfig (not a model hparam).
            attn_window: 0,
            attn_sink: 0,
        }
    }
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Default)]
struct MetalWarmReport {
    bytes: u64,
    mapped_bytes: u64,
    touches: u64,
    checksum: u64,
    elapsed_ms: f64,
    view_count: u32,
}

#[cfg(target_os = "macos")]
fn c_string_from_i8(bytes: &[i8]) -> String {
    let nul = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    let raw: Vec<u8> = bytes[..nul].iter().map(|&byte| byte as u8).collect();
    String::from_utf8_lossy(&raw).into_owned()
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::{contains_bytes, REQUIRED_NATIVE_KERNELS, WARMUP_METALLIB};

    #[test]
    fn qw35_metallib_contains_native_kernel_names() {
        if !cfg!(target_os = "macos") {
            return;
        }

        for name in REQUIRED_NATIVE_KERNELS {
            assert!(
                contains_bytes(WARMUP_METALLIB, name.as_bytes()),
                "Qw35 Metal library is missing {name}"
            );
        }
    }
}
