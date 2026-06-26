use crate::loader::{MappedGguf, WarmReport};
use crate::graph::{plan_qwen35, GraphPlan};
use crate::metal;
use crate::tokenizer::{DecodeState, QwenTokenizer};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

mod prompt;
mod sampling;
mod stop_sequence;
mod text_filter;
mod think_budget;
use prompt::{
    render_qwen35_chat_prompt, render_qwen35_chat_prompt_with_boundaries,
};
use sampling::sample_from_logits;
use stop_sequence::{earliest_stop_match, StopSequenceWatcher};
use text_filter::GeneratedTextFilter;
use think_budget::ThinkBudget;

pub const DEFAULT_MODEL_ID: &str = "qwen3.5-9b";
pub const DEFAULT_PREFILL_CHUNK: u32 = 32;
pub const MAX_PREFILL_CHUNK: u32 = 256;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_path: PathBuf,
    pub model_id: String,
    pub ctx_size: u32,
    pub prefill_chunk: u32,
    pub kv_cache_type: metal::KvCacheType,
    pub session_cache: bool,
    /// Decode-time sliding-window attention (0 = full attention) and leading
    /// sink tokens, from --attn-window/--attn-sink. Passed to the Metal runtime.
    pub attn_window: i32,
    pub attn_sink: i32,
    pub warm_weights: bool,
    pub test_responder: bool,
    pub verbose: bool,
}

#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub model: String,
    pub messages: Vec<ChatTurn>,
    pub max_tokens: TokenLimit,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
    pub repeat_last_n: i32,
    pub enable_thinking: bool,
    pub preserve_thinking: bool,
    /// Cap on tokens the model may spend inside the `<think>` block. `None`
    /// leaves reasoning uncapped (the legacy behaviour and the `xhigh` level);
    /// `Some(n)` forces `</think>` once the count reaches `n`. Derived from the
    /// caller's `reasoning_effort` — see `thinking_budget_for` in the server.
    pub thinking_budget: Option<u32>,
    /// Wrap-up message forced (pre-tokenized) just before `</think>` when the
    /// thinking budget is exhausted, so the model conditions on a clean "answer
    /// now" handoff (mirrors llama.cpp's `--reasoning-budget-message`). `None` or
    /// empty forces a bare `</think>`. Only used on the first block's forced
    /// close; reopened blocks always close bare.
    pub reasoning_budget_message: Option<String>,
    pub ignore_eos: bool,
    /// Generation halts (excluding the match) when the visible text contains
    /// any of these sequences; maps to the OpenAI `stop` field.
    pub stop_sequences: Vec<String>,
    /// Pass `<think>`/`</think>` markers and thinking content through as
    /// literal text instead of suppressing them, so the caller can route
    /// reasoning separately from content.
    pub emit_reasoning: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenLimit {
    Fixed(u32),
    Context,
}

#[derive(Debug, Clone)]
pub struct Generation {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Prompt tokens served from the session prefix cache (not re-evaluated).
    pub cached_tokens: u32,
    pub finish_reason: FinishReason,
    pub timings: GenerationTimings,
}

/// Why decoding ended: a natural stop (stop token or stop sequence) or the
/// token limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FinishReason {
    #[default]
    Stop,
    Length,
}

#[derive(Debug, Clone, Default)]
pub struct GenerationTimings {
    pub total_duration: Duration,
    pub render_duration: Duration,
    pub tokenize_duration: Duration,
    pub runtime_lock_duration: Duration,
    pub reset_duration: Duration,
    pub prompt_eval_duration: Duration,
    pub eval_duration: Duration,
    pub decode_eval_duration: Duration,
    pub sample_duration: Duration,
    pub detokenize_duration: Duration,
    pub stream_callback_duration: Duration,
    pub first_token_duration: Option<Duration>,
    pub prompt_eval_count: u32,
    pub eval_count: u32,
    pub cached_prompt_tokens: u32,
    pub session_path: SessionPath,
    pub prefill_chunk: u32,
    pub prefill_path: PrefillPath,
}

/// How the session prefix cache was used for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SessionPath {
    /// State was cleared and the whole prompt evaluated from position 0.
    #[default]
    Reset,
    /// The prompt extends the live state; only the suffix was evaluated.
    Extend,
    /// The recurrent state was rewound to the saved prompt-boundary
    /// checkpoint; evaluation resumed from there.
    Checkpoint,
}

impl SessionPath {
    pub fn as_str(self) -> &'static str {
        match self {
            SessionPath::Reset => "reset",
            SessionPath::Extend => "extend",
            SessionPath::Checkpoint => "checkpoint",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillPath {
    Scalar,
    TiledMm,
}

impl PrefillPath {
    pub fn as_str(self) -> &'static str {
        match self {
            PrefillPath::Scalar => "scalar",
            PrefillPath::TiledMm => "tiled-mm",
        }
    }
}

impl Default for PrefillPath {
    fn default() -> Self {
        Self::Scalar
    }
}

#[derive(Debug, Clone)]
pub enum GenerateError {
    BadRequest(String),
    InferenceUnavailable(String),
}

pub struct Engine {
    metal_runtime: Option<Mutex<RuntimeSession>>,
    // The GPU reads the GF4 FFN weights directly from this mapping; it must
    // outlive metal_runtime (fields drop in declaration order).
    gguf: MappedGguf,
    model_path: PathBuf,
    model_id: String,
    ctx_size: u32,
    prefill_chunk: u32,
    kv_cache_type: metal::KvCacheType,
    session_cache: bool,
    attn_window: i32,
    attn_sink: i32,
    test_responder: bool,
    verbose: bool,
    created: u64,
    warm_report: Option<WarmReport>,
    graph_plan: GraphPlan,
    tokenizer: QwenTokenizer,
}

/// The Metal runtime plus the token bookkeeping that makes its GPU state
/// reusable across requests. `evaluated` mirrors exactly the tokens whose
/// evaluation has been submitted (KV rows 0..len written, SSM state advanced).
/// `checkpoint` holds the token prefix that the runtime's saved recurrent
/// state corresponds to (taken at the end of the latest prompt prefill); the
/// hybrid SSM state cannot rewind, so this is the only rollback point.
struct RuntimeSession {
    runtime: metal::MetalRuntime,
    evaluated: Vec<u32>,
    checkpoint: Option<Vec<u32>>,
}

/// How much of the live session state a new prompt can reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionReuse {
    /// Prompt strictly extends the evaluated tokens: prefill from this offset.
    Extend(usize),
    /// Prompt matches the checkpoint prefix: restore the recurrent state and
    /// prefill from this offset.
    Checkpoint(usize),
    Reset,
}

/// Decide how to reuse session state for `prompt`. At least one prompt token
/// must always be (re-)evaluated so the output head runs on the last token.
fn plan_session_reuse(
    evaluated: &[u32],
    checkpoint: Option<&[u32]>,
    prompt: &[u32],
) -> SessionReuse {
    let cap = prompt.len().saturating_sub(1);
    if !evaluated.is_empty() && evaluated.len() <= cap && prompt[..evaluated.len()] == *evaluated {
        return SessionReuse::Extend(evaluated.len());
    }
    if let Some(ckpt) = checkpoint {
        if !ckpt.is_empty() && ckpt.len() <= cap && prompt[..ckpt.len()] == *ckpt {
            return SessionReuse::Checkpoint(ckpt.len());
        }
    }
    SessionReuse::Reset
}

impl Engine {
    pub fn open(config: EngineConfig) -> Result<Self, String> {
        validate_prefill_chunk(config.prefill_chunk)?;
        if !config.test_responder {
            metal::require_metal_device()?;
            metal::verify_native_qwen_kernels()?;
        }

        let gguf = MappedGguf::open(&config.model_path)?;
        validate_model(&gguf)?;
        let tokenizer = QwenTokenizer::load(&gguf)?;
        let graph_plan = plan_qwen35(&gguf);
        if !config.test_responder && !graph_plan.decoder_ready() {
            return Err(format!(
                "native decoder prerequisites are not satisfied: {}",
                graph_plan.blocker_summary()
            ));
        }

        let warm_report = if config.warm_weights {
            let report = if config.test_responder {
                gguf.warm_tensor_pages()
            } else {
                metal::warm_model_views(&gguf)?
            };
            Some(report)
        } else {
            None
        };

        let vocab_size = tokenizer.spec.vocab_size.try_into().map_err(|_| {
            "tokenizer vocabulary is too large for the native Metal runtime".to_string()
        })?;

        // The GF4 FFN lives in the unified .gguf (GGUF type-id 100 tensors), read
        // through the normal tensor table — no separate sidecar, no discovery.
        let metal_runtime = if config.test_responder {
            None
        } else {
            Some(Mutex::new(RuntimeSession {
                runtime: metal::MetalRuntime::new(
                    &gguf,
                    &graph_plan.hparams,
                    config.ctx_size,
                    vocab_size,
                    config.prefill_chunk,
                    config.kv_cache_type,
                    config.attn_window,
                    config.attn_sink,
                )?,
                evaluated: Vec::new(),
                checkpoint: None,
            }))
        };

        Ok(Self {
            metal_runtime,
            gguf,
            model_path: config.model_path,
            model_id: config.model_id,
            ctx_size: config.ctx_size,
            prefill_chunk: config.prefill_chunk,
            kv_cache_type: config.kv_cache_type,
            session_cache: config.session_cache,
            attn_window: config.attn_window,
            attn_sink: config.attn_sink,
            test_responder: config.test_responder,
            verbose: config.verbose,
            created: unix_time(),
            warm_report,
            graph_plan,
            tokenizer,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn created(&self) -> u64 {
        self.created
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn ctx_size(&self) -> u32 {
        self.ctx_size
    }

    pub fn prefill_chunk(&self) -> u32 {
        self.prefill_chunk
    }

    pub fn kv_cache_type(&self) -> metal::KvCacheType {
        self.kv_cache_type
    }

    pub fn session_cache(&self) -> bool {
        self.session_cache
    }

    /// True when decode uses GF4 FFN weights — i.e. the unified .gguf with its
    /// FFN baked as type-id 100 (a plain base GGUF decodes on Q4_K instead).
    pub fn gf4_active(&self) -> bool {
        self.ffn_is_gf4()
    }

    /// True when the GGUF itself carries a GF4 (type-id 100) FFN — i.e. the
    /// unified Qwowl3.5-9B .gguf rather than a plain base GGUF.
    fn ffn_is_gf4(&self) -> bool {
        self.gguf
            .tensor("blk.0.ffn_gate.weight")
            .is_some_and(|t| t.type_id == 100)
    }

    /// Label for the startup summary `ffn=` line.
    pub fn ffn_label(&self) -> &'static str {
        if self.ffn_is_gf4() {
            "gf4-unified"
        } else {
            "gguf"
        }
    }

    pub fn test_responder(&self) -> bool {
        self.test_responder
    }

    pub fn decoder_ready(&self) -> bool {
        self.test_responder || (self.graph_plan.decoder_ready() && self.metal_runtime.is_some())
    }

    pub fn warm_report(&self) -> Option<&WarmReport> {
        self.warm_report.as_ref()
    }

    pub fn graph_plan(&self) -> &GraphPlan {
        &self.graph_plan
    }

    pub fn inference_unavailable_message(&self) -> String {
        format!(
            "the HTTP server, mmap GGUF loader, Qwen35 tokenizer, native Qw35 Metal kernels, and Metal mmap warmup are wired, but the native Qwen3.5-9B decoder is not ready yet: {}",
            self.graph_plan.blocker_summary()
        )
    }

    pub fn metadata_summary(&self) -> ModelSummary {
        ModelSummary {
            architecture: self
                .gguf
                .metadata_string("general.architecture")
                .unwrap_or("unknown")
                .to_string(),
            name: self
                .gguf
                .metadata_string("general.name")
                .unwrap_or("unknown")
                .to_string(),
            size_label: self
                .gguf
                .metadata_string("general.size_label")
                .unwrap_or("unknown")
                .to_string(),
            block_count: self.gguf.metadata_u32("qwen35.block_count").unwrap_or(0),
            context_length: self.gguf.metadata_u32("qwen35.context_length").unwrap_or(0),
            embedding_length: self
                .gguf
                .metadata_u32("qwen35.embedding_length")
                .unwrap_or(0),
            vocab_size: self
                .gguf
                .metadata_array_len("tokenizer.ggml.tokens")
                .unwrap_or(0),
            tensor_count: self.gguf.tensors.len() as u64,
            mapped_bytes: self.gguf.len(),
            tensor_data_pos: self.gguf.tensor_data_pos,
        }
    }

    pub fn generate(&self, request: &GenerateRequest) -> Result<Generation, GenerateError> {
        let total_start = Instant::now();
        self.validate_generate_request(request)?;

        if !self.test_responder {
            return self.generate_native(request, |_| Ok(()), |_, _| {});
        }

        let last_user = request
            .messages
            .iter()
            .rev()
            .find(|msg| msg.role == "user")
            .map(|msg| msg.content.as_str())
            .unwrap_or("");

        let mut text = if let Some(exact) = exact_requested_text(last_user) {
            exact.to_string()
        } else if last_user.to_ascii_lowercase().contains("2+2") {
            "4".to_string()
        } else {
            format!(
                "Qw35 HTTP test responder: model metadata is loaded through mmap, and the chat endpoint received {} message(s). Last user turn: {}",
                request.messages.len(),
                one_line(last_user)
            )
        };

        let render_start = Instant::now();
        let prompt = render_qwen35_chat_prompt(
            &request.messages,
            request.enable_thinking,
            request.preserve_thinking,
        );
        let render_duration = render_start.elapsed();

        let tokenize_start = Instant::now();
        let prompt_tokens = self
            .tokenizer
            .encode(&prompt, true)
            .map(|tokens| tokens.len() as u32)
            .unwrap_or_else(|_| rough_token_count(&prompt));
        let max_tokens = self.resolve_max_tokens(request.max_tokens, prompt_tokens as usize)?;
        let mut finish_reason = FinishReason::Stop;
        if let Some(idx) = earliest_stop_match(&text, &request.stop_sequences) {
            text.truncate(idx);
        }
        let limited = limit_rough_tokens(&text, max_tokens);
        if limited.len() < text.len() {
            finish_reason = FinishReason::Length;
        }
        text = limited;
        let completion_tokens = self
            .tokenizer
            .encode(&text, false)
            .map(|tokens| tokens.len() as u32)
            .unwrap_or_else(|_| rough_token_count(&text));
        let tokenize_duration = tokenize_start.elapsed();

        Ok(Generation {
            text,
            prompt_tokens,
            completion_tokens,
            cached_tokens: 0,
            finish_reason,
            timings: GenerationTimings {
                total_duration: total_start.elapsed(),
                render_duration,
                tokenize_duration,
                prompt_eval_count: prompt_tokens,
                eval_count: completion_tokens,
                prefill_chunk: self.prefill_chunk,
                prefill_path: PrefillPath::Scalar,
                ..GenerationTimings::default()
            },
        })
    }

    pub fn generate_stream<F>(
        &self,
        request: &GenerateRequest,
        on_text: F,
    ) -> Result<Generation, GenerateError>
    where
        F: FnMut(&str) -> Result<(), String>,
    {
        self.generate_stream_with_progress(request, on_text, |_, _| {})
    }

    /// Like [`generate_stream`], but also reports prompt-processing (prefill)
    /// progress through `on_prefill(processed_tokens, total_tokens)`. This is a
    /// pure server-side side-channel; the OpenAI streaming contract is unchanged
    /// (the handler ships progress in choice-less chunks that clients ignore).
    pub fn generate_stream_with_progress<F, P>(
        &self,
        request: &GenerateRequest,
        mut on_text: F,
        on_prefill: P,
    ) -> Result<Generation, GenerateError>
    where
        F: FnMut(&str) -> Result<(), String>,
        P: FnMut(usize, usize),
    {
        let total_start = Instant::now();
        self.validate_generate_request(request)?;

        if !self.test_responder {
            return self.generate_native(request, on_text, on_prefill);
        }

        let mut generation = self.generate(request)?;
        let mut callback_duration = Duration::ZERO;
        let mut first_token_duration = None;
        for chunk in stream_text_chunks(&generation.text) {
            let callback_start = Instant::now();
            on_text(chunk).map_err(GenerateError::InferenceUnavailable)?;
            callback_duration += callback_start.elapsed();
            if first_token_duration.is_none() && !chunk.is_empty() {
                first_token_duration = Some(total_start.elapsed());
            }
        }
        generation.timings.stream_callback_duration = callback_duration;
        generation.timings.first_token_duration = first_token_duration;
        generation.timings.total_duration = total_start.elapsed();
        Ok(generation)
    }

    pub fn count_prompt_tokens(&self, request: &GenerateRequest) -> Result<u32, GenerateError> {
        self.validate_generate_request(request)?;
        let prompt = render_qwen35_chat_prompt(
            &request.messages,
            request.enable_thinking,
            request.preserve_thinking,
        );
        match self.tokenizer.encode(&prompt, true) {
            Ok(tokens) => Ok(tokens.len() as u32),
            Err(_) if self.test_responder => Ok(rough_token_count(&prompt)),
            Err(err) => Err(GenerateError::BadRequest(format!(
                "tokenization failed: {err}"
            ))),
        }
    }

    fn validate_generate_request(&self, request: &GenerateRequest) -> Result<(), GenerateError> {
        if request.model != self.model_id {
            return Err(GenerateError::BadRequest(format!(
                "model {:?} is not available; use {:?}",
                request.model, self.model_id
            )));
        }
        if request.messages.is_empty() {
            return Err(GenerateError::BadRequest(
                "messages must contain at least one chat turn".to_string(),
            ));
        }
        if matches!(request.max_tokens, TokenLimit::Fixed(0)) {
            return Err(GenerateError::BadRequest(
                "max_tokens must be greater than zero or -1".to_string(),
            ));
        }
        if !request.temperature.is_finite() || request.temperature < 0.0 {
            return Err(GenerateError::BadRequest(
                "temperature must be a finite non-negative number".to_string(),
            ));
        }
        if !request.top_p.is_finite() || request.top_p < 0.0 {
            return Err(GenerateError::BadRequest(
                "top_p must be a finite non-negative number".to_string(),
            ));
        }
        if !request.min_p.is_finite() || request.min_p < 0.0 {
            return Err(GenerateError::BadRequest(
                "min_p must be a finite non-negative number".to_string(),
            ));
        }
        if !request.presence_penalty.is_finite() {
            return Err(GenerateError::BadRequest(
                "presence_penalty must be finite".to_string(),
            ));
        }
        if !request.frequency_penalty.is_finite() {
            return Err(GenerateError::BadRequest(
                "frequency_penalty must be finite".to_string(),
            ));
        }
        if !request.repetition_penalty.is_finite() || request.repetition_penalty <= 0.0 {
            return Err(GenerateError::BadRequest(
                "repetition_penalty must be finite and greater than zero".to_string(),
            ));
        }
        if request.repeat_last_n < -1 {
            return Err(GenerateError::BadRequest(
                "repeat_last_n must be -1 or greater".to_string(),
            ));
        }
        Ok(())
    }

    fn resolve_max_tokens(
        &self,
        limit: TokenLimit,
        prompt_tokens: usize,
    ) -> Result<u32, GenerateError> {
        let ctx_size = self.ctx_size as usize;
        if prompt_tokens >= ctx_size {
            return Err(GenerateError::BadRequest(format!(
                "prompt requires {prompt_tokens} context slots, but --ctx is {}",
                self.ctx_size
            )));
        }

        match limit {
            TokenLimit::Fixed(max_tokens) => {
                let max_needed = prompt_tokens.saturating_add(max_tokens as usize);
                if max_needed > ctx_size {
                    return Err(GenerateError::BadRequest(format!(
                        "prompt plus max_tokens requires {max_needed} context slots, but --ctx is {}",
                        self.ctx_size
                    )));
                }
                Ok(max_tokens)
            }
            TokenLimit::Context => Ok((ctx_size - prompt_tokens) as u32),
        }
    }

    /// Evaluate prompt positions [start, end). The output head runs only on
    /// the final prompt token, so `head_mode` takes effect only when the
    /// range covers it.
    fn prefill_range(
        &self,
        runtime: &mut metal::MetalRuntime,
        prompt_tokens: &[u32],
        start: usize,
        end: usize,
        head_mode: metal::LogitsMode,
        on_prefill: &mut dyn FnMut(usize, usize),
    ) -> Result<(), String> {
        if start >= end {
            return Ok(());
        }
        let total = prompt_tokens.len();
        let last_prompt_pos = total - 1;
        let segment = &prompt_tokens[start..end];
        if self.prefill_chunk <= 1 || segment.len() == 1 {
            for (off, &token) in segment.iter().enumerate() {
                let pos = start + off;
                let mode = if pos == last_prompt_pos {
                    head_mode
                } else {
                    metal::LogitsMode::None
                };
                runtime.eval_token(token, pos as u32, mode)?;
                on_prefill(pos + 1, total);
            }
        } else {
            let chunk_len = self.prefill_chunk as usize;
            for (chunk_idx, chunk) in segment.chunks(chunk_len).enumerate() {
                let pos0 = start + chunk_idx * chunk_len;
                let mode = if pos0 + chunk.len() == prompt_tokens.len() {
                    head_mode
                } else {
                    metal::LogitsMode::None
                };
                runtime.eval_prefill_chunk(chunk, pos0 as u32, mode)?;
                on_prefill(pos0 + chunk.len(), total);
            }
        }
        Ok(())
    }

    fn generate_native<F, P>(
        &self,
        request: &GenerateRequest,
        mut on_text: F,
        mut on_prefill: P,
    ) -> Result<Generation, GenerateError>
    where
        F: FnMut(&str) -> Result<(), String>,
        P: FnMut(usize, usize),
    {
        let total_start = Instant::now();
        let (prompt, stable_len, preamble_len) = render_qwen35_chat_prompt_with_boundaries(
            &request.messages,
            request.enable_thinking,
            request.preserve_thinking,
        );
        let render_duration = total_start.elapsed();

        let tokenize_start = Instant::now();
        let prompt_tokens = self
            .tokenizer
            .encode(&prompt, true)
            .map_err(|err| GenerateError::BadRequest(format!("tokenization failed: {err}")))?;
        // Token-space position of the stable prefix, used as the session
        // checkpoint boundary. The generation header begins with the special
        // <|im_start|> token, which fences BPE merges, so the prefix encodes
        // to a strict token-prefix of the full prompt; the comparison guards
        // against any exception by disabling the checkpoint (boundary 0).
        let history_len = if self.session_cache && stable_len > 0 {
            match self.tokenizer.encode(&prompt[..stable_len], true) {
                Ok(history)
                    if history.len() < prompt_tokens.len()
                        && prompt_tokens[..history.len()] == history[..] =>
                {
                    history.len()
                }
                _ => 0,
            }
        } else {
            0
        };
        // Sliding-window attention sink floor: pin the system block + first user
        // turn (the preamble) so a long session never evicts the tool-call format
        // or the task from the windowed attention layers. Only needed when the
        // window is active; tokenize the preamble prefix for its token length.
        let preamble_tokens = if self.attn_window > 0 && preamble_len > 0 {
            self.tokenizer
                .encode(&prompt[..preamble_len], true)
                .map(|t| t.len())
                .unwrap_or(0)
        } else {
            0
        };
        let tokenize_duration = tokenize_start.elapsed();
        if prompt_tokens.is_empty() {
            return Err(GenerateError::BadRequest(
                "rendered prompt produced no tokens".to_string(),
            ));
        }
        let max_tokens = self.resolve_max_tokens(request.max_tokens, prompt_tokens.len())?;

        let runtime = self.metal_runtime.as_ref().ok_or_else(|| {
            GenerateError::InferenceUnavailable(self.inference_unavailable_message())
        })?;
        let lock_start = Instant::now();
        let mut session_guard = runtime.lock().map_err(|_| {
            GenerateError::InferenceUnavailable("native Metal runtime lock is poisoned".to_string())
        })?;
        let session = &mut *session_guard;
        let runtime_lock_duration = lock_start.elapsed();

        // Pin system prompt + first user turn under the sliding window: the
        // effective sink is at least the preamble length, so the windowed
        // attention layers keep seeing the tool-call format and the task.
        if self.attn_window > 0 {
            let sink = self.attn_sink.max(preamble_tokens as i32);
            session.runtime.set_attn_sink(sink);
        }

        // Session prefix cache: reuse live GPU state when the new prompt
        // extends it, rewind to the prompt-boundary checkpoint when it
        // diverged inside the previous generation, else start from scratch.
        let reuse = if self.session_cache {
            plan_session_reuse(
                &session.evaluated,
                session.checkpoint.as_deref(),
                &prompt_tokens,
            )
        } else {
            SessionReuse::Reset
        };

        let reset_start = Instant::now();
        let (cached, session_path) = match reuse {
            SessionReuse::Extend(n) => (n, SessionPath::Extend),
            SessionReuse::Checkpoint(c) => {
                if session.runtime.state_checkpoint_restore().is_ok() {
                    (c, SessionPath::Checkpoint)
                } else {
                    session.checkpoint = None;
                    (0, SessionPath::Reset)
                }
            }
            SessionReuse::Reset => (0, SessionPath::Reset),
        };
        if cached == 0 {
            session
                .runtime
                .reset()
                .map_err(GenerateError::InferenceUnavailable)?;
            // The full prompt will overwrite KV rows from position 0, so the
            // checkpoint's cache rows are no longer trustworthy.
            session.checkpoint = None;
        }
        // Pessimistic: the live state is unknown until this request finishes
        // cleanly; the checkpoint stays valid because positions below it are
        // never re-evaluated on the Extend/Checkpoint paths.
        session.evaluated.clear();
        let reset_duration = reset_start.elapsed();

        // Greedy decode never materializes the logits vector; sampling needs it.
        let head_mode = if request.temperature <= 0.0 {
            metal::LogitsMode::Argmax
        } else {
            metal::LogitsMode::Full
        };

        let prompt_eval_start = Instant::now();
        let prefill_path = prefill_path_for(self.prefill_chunk, prompt_tokens.len() - cached);
        // Phase A: conversation history up to the stable boundary, then save
        // the recurrent state as the rewind point for the next request. The
        // generation header after the boundary is never reproduced by future
        // prompts, so checkpointing past it would be useless.
        let phase_a_end = history_len.max(cached);
        self.prefill_range(
            &mut session.runtime,
            &prompt_tokens,
            cached,
            phase_a_end,
            head_mode,
            &mut on_prefill,
        )
        .map_err(GenerateError::InferenceUnavailable)?;
        if history_len > 0 && cached < history_len {
            // Waits for phase A internally; failure only disables the rewind.
            if session.runtime.state_checkpoint_save().is_ok() {
                session.checkpoint = Some(prompt_tokens[..history_len].to_vec());
            }
        }
        // Phase B: the generation header (plus any history already covered by
        // a longer cached prefix).
        self.prefill_range(
            &mut session.runtime,
            &prompt_tokens,
            phase_a_end,
            prompt_tokens.len(),
            head_mode,
            &mut on_prefill,
        )
        .map_err(GenerateError::InferenceUnavailable)?;
        // Evals are committed asynchronously; wait here so prompt processing
        // is attributed to prompt_eval rather than the first decode sample.
        session
            .runtime
            .sync()
            .map_err(GenerateError::InferenceUnavailable)?;
        let prompt_eval_duration = prompt_eval_start.elapsed();
        session.evaluated = prompt_tokens.clone();

        let mut all_seen = prompt_tokens.clone();
        let stop_ids = self.tokenizer.stop_token_ids();
        let mut completion = Vec::<u32>::new();
        let mut rendered = String::new();
        let mut decode_state = DecodeState::default();
        let decode_start = Instant::now();
        let mut sample_duration = Duration::ZERO;
        let mut decode_eval_duration = Duration::ZERO;
        let mut detokenize_duration = Duration::ZERO;
        let mut stream_callback_duration = Duration::ZERO;
        let mut first_token_duration = None;
        let mut output_filter = GeneratedTextFilter::new(request.emit_reasoning);
        let mut stop_watcher = StopSequenceWatcher::new(&request.stop_sequences);
        let mut finish_reason = FinishReason::Length;
        let mut stopped_by_sequence = false;

        // Thinking-token budget (see `ThinkBudget`). Generation begins right
        // after the prompt's `<think>\n`, so we start inside the reasoning block.
        // Past the soft budget the tracker ramps a `</think>` logit bias, then at
        // a sentence boundary (or the hard ceiling) forces a wrap-up message +
        // `</think>` that the model conditions on for its answer. The wrap-up
        // message is pre-tokenized here (empty => bare close, today's behavior).
        let message_tokens: Vec<u32> = request
            .reasoning_budget_message
            .as_deref()
            .filter(|message| !message.is_empty())
            .and_then(|message| self.tokenizer.encode(message, false).ok())
            .unwrap_or_default();
        let mut think_budget = ThinkBudget::new(
            self.tokenizer.spec.end_think_token_id,
            self.tokenizer.spec.think_token_id,
            request.enable_thinking,
            request.thinking_budget,
            message_tokens,
        );

        // Fast-path per-window decode-tps trace (QW35_DECODE_TRACE): prints the
        // instantaneous tok/s every 128 decoded tokens with the current ctx, so a
        // session-length slowdown can be localized on the production path without
        // the serialised stage profiler perturbing thermals/timing.
        let decode_trace = std::env::var("QW35_DECODE_TRACE").is_ok();
        let mut trace_win_start = Instant::now();
        let mut trace_win_tokens = 0u32;

        for step in 0..max_tokens {
            let sample_start = Instant::now();
            // While the thinking-budget tracker is draining a forced close
            // sequence (wrap-up message + `</think>`), emit those tokens one per
            // step and skip sampling; they still flow through detokenize/eval, so
            // the model conditions on them for its answer.
            let next = if let Some(forced) = think_budget.forced_next() {
                forced
            } else {
                let sampled = if request.temperature <= 0.0 {
                    // Greedy: GPU argmax. The `</think>` ramp bias is a
                    // sampling-path feature (logits are not materialized on the
                    // argmax head), so greedy relies on the message/ceiling tiers.
                    session.runtime.argmax().map(|(token, _)| token)
                } else {
                    let bias = think_budget.bias();
                    session.runtime.copy_logits().map(|mut logits| {
                        if let Some((id, amount)) = bias {
                            if let Some(slot) = logits.get_mut(id as usize) {
                                *slot += amount;
                            }
                        }
                        sample_from_logits(&logits, request, &all_seen, prompt_tokens.len())
                    })
                };
                let sampled = match sampled {
                    Ok(token) => token,
                    Err(err) => {
                        // The GPU state can no longer be trusted as a prefix.
                        session.evaluated.clear();
                        return Err(GenerateError::InferenceUnavailable(err));
                    }
                };
                // Enforce the thinking-token budget on the freshly sampled token.
                think_budget.observe(sampled, |token| {
                    self.tokenizer.token_ends_with_newline(token)
                })
            };
            sample_duration += sample_start.elapsed();

            if !request.ignore_eos && stop_ids.contains(&next) {
                finish_reason = FinishReason::Stop;
                break;
            }
            if first_token_duration.is_none() {
                first_token_duration = Some(decode_start.elapsed());
            }

            completion.push(next);
            all_seen.push(next);
            if decode_trace {
                trace_win_tokens += 1;
                if trace_win_tokens >= 128 {
                    let dt = trace_win_start.elapsed().as_secs_f64();
                    eprintln!(
                        "[QW35_DECODE_TRACE] ctx={} win={}tok {:.2}tok/s",
                        prompt_tokens.len() + step as usize,
                        trace_win_tokens,
                        trace_win_tokens as f64 / dt
                    );
                    trace_win_start = Instant::now();
                    trace_win_tokens = 0;
                }
            }
            let detokenize_start = Instant::now();
            let delta = self.tokenizer.decode_one(next, false, &mut decode_state);
            detokenize_duration += detokenize_start.elapsed();
            let delta = output_filter.visible_token_delta(next, &delta, &self.tokenizer);
            if !delta.is_empty() {
                let (visible, stop_hit) = stop_watcher.feed(&delta);
                if !visible.is_empty() {
                    let callback_start = Instant::now();
                    on_text(&visible).map_err(GenerateError::InferenceUnavailable)?;
                    stream_callback_duration += callback_start.elapsed();
                    rendered.push_str(&visible);
                }
                if stop_hit {
                    finish_reason = FinishReason::Stop;
                    stopped_by_sequence = true;
                    break;
                }
            }

            if step + 1 < max_tokens {
                let pos = prompt_tokens.len() as u32 + step;
                let eval_start = Instant::now();
                if let Err(err) = session.runtime.eval_token(next, pos, head_mode) {
                    session.evaluated.clear();
                    return Err(GenerateError::InferenceUnavailable(err));
                }
                session.evaluated.push(next);
                decode_eval_duration += eval_start.elapsed();
            }
        }
        let detokenize_start = Instant::now();
        let delta = self.tokenizer.finish_decode(&mut decode_state);
        detokenize_duration += detokenize_start.elapsed();
        let delta = output_filter.visible_finish_delta(&delta);
        if !stopped_by_sequence {
            let (mut visible, stop_hit) = stop_watcher.feed(&delta);
            if stop_hit {
                finish_reason = FinishReason::Stop;
            } else {
                visible.push_str(&stop_watcher.finish());
            }
            if !visible.is_empty() {
                let callback_start = Instant::now();
                on_text(&visible).map_err(GenerateError::InferenceUnavailable)?;
                stream_callback_duration += callback_start.elapsed();
                rendered.push_str(&visible);
            }
        }
        let eval_duration = decode_start.elapsed();
        // prompt_eval_count is what was actually evaluated this request; the
        // cached prefix is reported separately.
        let prompt_eval_count = (prompt_tokens.len() - cached) as u32;
        let eval_count = completion.len() as u32;

        let generation = Generation {
            text: rendered,
            prompt_tokens: prompt_tokens.len() as u32,
            completion_tokens: eval_count,
            cached_tokens: cached as u32,
            finish_reason,
            timings: GenerationTimings {
                total_duration: total_start.elapsed(),
                render_duration,
                tokenize_duration,
                runtime_lock_duration,
                reset_duration,
                prompt_eval_duration,
                eval_duration,
                decode_eval_duration,
                sample_duration,
                detokenize_duration,
                stream_callback_duration,
                first_token_duration,
                prompt_eval_count,
                eval_count,
                cached_prompt_tokens: cached as u32,
                session_path,
                prefill_chunk: self.prefill_chunk,
                prefill_path,
            },
        };
        if self.verbose {
            // Logged after the final sync, outside any timed section, so the
            // report itself never shows up in the numbers it prints.
            let t = &generation.timings;
            eprintln!(
                "qw35: prompt={} tok (cached {}, session={}) | prefill {} tok in {:.3}s ({:.1} tok/s, path={}) | decode {} tok in {:.3}s ({:.1} tok/s) | ttft {:.0}ms | finish={}",
                generation.prompt_tokens,
                t.cached_prompt_tokens,
                t.session_path.as_str(),
                t.prompt_eval_count,
                t.prompt_eval_duration.as_secs_f64(),
                tokens_per_second(t.prompt_eval_count, t.prompt_eval_duration),
                t.prefill_path.as_str(),
                t.eval_count,
                t.eval_duration.as_secs_f64(),
                tokens_per_second(t.eval_count, t.eval_duration),
                (t.prompt_eval_duration + t.first_token_duration.unwrap_or(Duration::ZERO))
                    .as_secs_f64()
                    * 1000.0,
                match generation.finish_reason {
                    FinishReason::Stop => "stop",
                    FinishReason::Length => "length",
                },
            );
        }
        Ok(generation)
    }
}

fn tokens_per_second(count: u32, duration: Duration) -> f64 {
    if duration.is_zero() {
        return 0.0;
    }
    count as f64 / duration.as_secs_f64()
}

#[derive(Debug, Clone)]
pub struct ModelSummary {
    pub architecture: String,
    pub name: String,
    pub size_label: String,
    pub block_count: u32,
    pub context_length: u32,
    pub embedding_length: u32,
    pub vocab_size: u64,
    pub tensor_count: u64,
    pub mapped_bytes: u64,
    pub tensor_data_pos: u64,
}

pub fn validate_prefill_chunk(prefill_chunk: u32) -> Result<(), String> {
    if prefill_chunk == 0 {
        return Err("--prefill-chunk must be greater than zero".to_string());
    }
    if prefill_chunk > MAX_PREFILL_CHUNK {
        return Err(format!(
            "--prefill-chunk must be at most {MAX_PREFILL_CHUNK}"
        ));
    }
    Ok(())
}

fn prefill_path_for(prefill_chunk: u32, prompt_tokens: usize) -> PrefillPath {
    if prefill_chunk <= 1 || prompt_tokens <= 1 {
        PrefillPath::Scalar
    } else {
        PrefillPath::TiledMm
    }
}


fn validate_model(gguf: &MappedGguf) -> Result<(), String> {
    let architecture = gguf
        .metadata_string("general.architecture")
        .ok_or("missing general.architecture metadata")?;
    if architecture != "qwen35" {
        return Err(format!(
            "unsupported architecture {architecture:?}; qw35 expects the Qwen3.5 GGUF layout"
        ));
    }

    // Only the original Qwen3.5-9B is supported; fine-tunes are not. They reuse
    // the base layout but rewrite general.name to a commit hash and bump
    // general.size_label to "9.0B", so gate on both before the structural check
    // pins the exact Qwen3.5-9B layout.
    let name = gguf
        .metadata_string("general.name")
        .ok_or("missing general.name metadata")?;
    let size_label = gguf
        .metadata_string("general.size_label")
        .ok_or("missing general.size_label metadata")?;
    if name != "Qwen3.5-9B" || size_label != "9B" {
        return Err(format!(
            "unsupported model {name:?} (size_label {size_label:?}); qw35 only supports the original Qwen3.5-9B, not fine-tunes"
        ));
    }

    let block_count = gguf
        .metadata_u32("qwen35.block_count")
        .ok_or("missing qwen35.block_count metadata")?;
    let emb = gguf
        .metadata_u32("qwen35.embedding_length")
        .ok_or("missing qwen35.embedding_length metadata")?;
    let ffn = gguf
        .metadata_u32("qwen35.feed_forward_length")
        .ok_or("missing qwen35.feed_forward_length metadata")?;
    let ctx = gguf
        .metadata_u32("qwen35.context_length")
        .ok_or("missing qwen35.context_length metadata")?;
    let attention_heads = gguf
        .metadata_u32("qwen35.attention.head_count")
        .ok_or("missing qwen35.attention.head_count metadata")?;
    let attention_kv_heads = gguf
        .metadata_u32("qwen35.attention.head_count_kv")
        .ok_or("missing qwen35.attention.head_count_kv metadata")?;
    let attention_key_length = gguf
        .metadata_u32("qwen35.attention.key_length")
        .ok_or("missing qwen35.attention.key_length metadata")?;
    let attention_value_length = gguf
        .metadata_u32("qwen35.attention.value_length")
        .ok_or("missing qwen35.attention.value_length metadata")?;
    let rope_dim = gguf
        .metadata_u32("qwen35.rope.dimension_count")
        .ok_or("missing qwen35.rope.dimension_count metadata")?;
    let ssm_conv_kernel = gguf
        .metadata_u32("qwen35.ssm.conv_kernel")
        .ok_or("missing qwen35.ssm.conv_kernel metadata")?;
    let ssm_state_size = gguf
        .metadata_u32("qwen35.ssm.state_size")
        .ok_or("missing qwen35.ssm.state_size metadata")?;
    let ssm_group_count = gguf
        .metadata_u32("qwen35.ssm.group_count")
        .ok_or("missing qwen35.ssm.group_count metadata")?;
    let ssm_time_step_rank = gguf
        .metadata_u32("qwen35.ssm.time_step_rank")
        .ok_or("missing qwen35.ssm.time_step_rank metadata")?;
    let nextn = gguf
        .metadata_u32("qwen35.nextn_predict_layers")
        .unwrap_or(0);
    let vocab = gguf
        .metadata_array_len("tokenizer.ggml.tokens")
        .unwrap_or(248_320);
    let tensor_count = gguf.tensors.len() as u64;

    if block_count != 32
        || emb != 4096
        || ffn != 12_288
        || ctx != 262_144
        || attention_heads != 16
        || attention_kv_heads != 4
        || attention_key_length != 256
        || attention_value_length != 256
        || rope_dim != 64
        || ssm_conv_kernel != 4
        || ssm_state_size != 128
        || ssm_group_count != 16
        || ssm_time_step_rank != 32
        || vocab != 248_320
        || tensor_count != 427
        || nextn != 0
    {
        return Err(format!(
            "unexpected Qwen3.5-9B layout: block_count={block_count}, embedding_length={emb}, feed_forward_length={ffn}, context_length={ctx}, attention_heads={attention_heads}, attention_kv_heads={attention_kv_heads}, attention_key_length={attention_key_length}, attention_value_length={attention_value_length}, rope_dimension_count={rope_dim}, ssm_conv_kernel={ssm_conv_kernel}, ssm_state_size={ssm_state_size}, ssm_group_count={ssm_group_count}, ssm_time_step_rank={ssm_time_step_rank}, vocab={vocab}, tensors={tensor_count}, nextn_predict_layers={nextn}"
        ));
    }

    Ok(())
}

fn exact_requested_text(prompt: &str) -> Option<&str> {
    let lower = prompt.to_ascii_lowercase();
    let marker = "say exactly:";
    let idx = lower.find(marker)?;
    let exact = prompt[idx + marker.len()..].trim();
    if exact.is_empty() {
        None
    } else {
        Some(exact.trim_matches('"'))
    }
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn rough_token_count(text: &str) -> u32 {
    let chars = text.chars().count();
    let words = text.split_whitespace().count();
    u32::try_from(words.max(chars.div_ceil(4)).max(1)).unwrap_or(u32::MAX)
}

fn limit_rough_tokens(text: &str, max_tokens: u32) -> String {
    let max_chars = usize::try_from(max_tokens)
        .unwrap_or(usize::MAX / 4)
        .saturating_mul(4)
        .max(1);
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    text.chars().take(max_chars).collect()
}

fn stream_text_chunks(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return vec![""];
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    for (idx, _) in text.char_indices() {
        if idx > start && idx - start >= 24 {
            chunks.push(&text[start..idx]);
            start = idx;
        }
    }
    if start < text.len() {
        chunks.push(&text[start..]);
    }
    chunks
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::{
        plan_session_reuse, render_qwen35_chat_prompt, ChatTurn, Engine,
        EngineConfig, GenerateRequest, SessionReuse, TokenLimit, DEFAULT_MODEL_ID,
    };
    use std::path::PathBuf;

    #[test]
    fn session_reuse_extends_live_state() {
        // Prompt strictly extends the evaluated tokens.
        let evaluated = vec![1, 2, 3, 4];
        let prompt = vec![1, 2, 3, 4, 5, 6];
        assert_eq!(
            plan_session_reuse(&evaluated, None, &prompt),
            SessionReuse::Extend(4)
        );
    }

    #[test]
    fn session_reuse_always_leaves_one_token_to_evaluate() {
        // Identical prompt: the live state cannot be reused as-is because no
        // token would be left to produce logits, and the SSM state cannot
        // rewind. Falls back to the checkpoint only if it is short enough.
        let evaluated = vec![1, 2, 3, 4];
        let prompt = vec![1, 2, 3, 4];
        assert_eq!(
            plan_session_reuse(&evaluated, None, &prompt),
            SessionReuse::Reset
        );
        let checkpoint = vec![1, 2, 3];
        assert_eq!(
            plan_session_reuse(&evaluated, Some(&checkpoint), &prompt),
            SessionReuse::Checkpoint(3)
        );
        // A checkpoint covering the whole prompt is unusable too.
        let full = vec![1, 2, 3, 4];
        assert_eq!(
            plan_session_reuse(&evaluated, Some(&full), &prompt),
            SessionReuse::Reset
        );
    }

    #[test]
    fn session_reuse_rewinds_to_checkpoint_on_divergence() {
        // The previous reply re-tokenized differently: the prompt diverges
        // inside the generated region but still matches the prompt-boundary
        // checkpoint.
        let evaluated = vec![1, 2, 3, 9, 9];
        let checkpoint = vec![1, 2, 3];
        let prompt = vec![1, 2, 3, 7, 8, 9];
        assert_eq!(
            plan_session_reuse(&evaluated, Some(&checkpoint), &prompt),
            SessionReuse::Checkpoint(3)
        );
    }

    #[test]
    fn session_reuse_resets_for_unrelated_prompts() {
        let evaluated = vec![1, 2, 3, 4];
        let checkpoint = vec![1, 2, 3];
        let prompt = vec![5, 6, 7, 8];
        assert_eq!(
            plan_session_reuse(&evaluated, Some(&checkpoint), &prompt),
            SessionReuse::Reset
        );
        assert_eq!(plan_session_reuse(&[], None, &prompt), SessionReuse::Reset);
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "loads the real Qwen3.5 GGUF and runs native Metal prefill comparisons"]
    fn real_model_q4_q5_q6_k_tiled_prefill_matches_reference_first_token() {
        let _gpu = real_model_gpu_lock();
        // All chunks above 1 use the tiled simdgroup matmul; chunk 8 serves as
        // the reference against other chunk geometries.
        let reference = open_real_engine(8);
        let reference_logits = prefill_logits(&reference);
        let reference_token = argmax(&reference_logits);

        // The scalar path decodes the FFN with GF4 sidecar weights when one
        // is present, so cross-path logit parity only holds without it.
        if !reference.gf4_active() {
            let scalar = open_real_engine(1);
            assert_eq!(argmax(&prefill_logits(&scalar)), reference_token, "scalar");
        }

        for chunk in [9, 64] {
            let tiled = open_real_engine(chunk);
            let logits = prefill_logits(&tiled);
            assert_eq!(argmax(&logits), reference_token, "chunk {chunk}");

            if chunk == 64 {
                let max_abs = reference_logits
                    .iter()
                    .zip(&logits)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs < 2.0e-2,
                    "chunk {chunk} max logit absolute diff {max_abs}"
                );
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "loads the real Qwen3.5 GGUF and compares native Metal greedy decode paths"]
    fn real_model_tiled_prefill_decode_matches_batch_reference_first_tokens() {
        let _gpu = real_model_gpu_lock();
        let reference = open_real_engine(8);
        let reference_tokens = greedy_tokens_after_prefill(&reference, 4);

        let tiled = open_real_engine(64);
        let tiled_tokens = greedy_tokens_after_prefill(&tiled, 4);

        assert_eq!(tiled_tokens, reference_tokens);
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "loads the real Qwen3.5 GGUF and checks repeated prompt prefill after runtime reset"]
    fn real_model_repeated_prefill_reset_is_stable_across_requests() {
        let _gpu = real_model_gpu_lock();
        for prefill_chunk in [1, 64] {
            let engine = open_real_engine(prefill_chunk);
            let first = argmax(&prefill_logits(&engine));
            for run in 2..=3 {
                let next = argmax(&prefill_logits(&engine));
                assert_eq!(
                    next, first,
                    "prefill_chunk {prefill_chunk} changed first-token argmax on run {run}"
                );
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "loads the real Qwen3.5 GGUF and runs a three-turn greedy chat smoke test"]
    fn real_model_chat_ping_pong_greedy_smoke_three_turns() {
        let _gpu = real_model_gpu_lock();
        let engine = open_real_engine(1);
        let mut messages = Vec::new();

        for (idx, user) in [
            "Reply with one short greeting.",
            "Now name one color.",
            "Now name one shape.",
        ]
        .into_iter()
        .enumerate()
        {
            messages.push(ChatTurn {
                role: "user".to_string(),
                content: user.to_string(),
            });
            let generation = engine
                .generate(&GenerateRequest {
                    model: DEFAULT_MODEL_ID.to_string(),
                    messages: messages.clone(),
                    max_tokens: TokenLimit::Fixed(16),
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: 0,
                    min_p: 0.0,
                    presence_penalty: 0.0,
                    frequency_penalty: 0.0,
                    repetition_penalty: 1.0,
                    repeat_last_n: -1,
                    enable_thinking: false,
                    preserve_thinking: false,
                    thinking_budget: None,
                    reasoning_budget_message: None,
                    ignore_eos: false,
                    stop_sequences: Vec::new(),
                    emit_reasoning: false,
                })
                .unwrap_or_else(|err| panic!("turn {} failed: {err:?}", idx + 1));
            eprintln!("turn {} assistant: {:?}", idx + 1, generation.text);
            assert!(
                generation.completion_tokens > 0,
                "turn {} generated no completion tokens",
                idx + 1
            );
            messages.push(ChatTurn {
                role: "assistant".to_string(),
                content: generation.text,
            });
        }
    }

    /// The real-model tests each open one or more engines that stream the
    /// full 5.3 GiB model through the GPU. Running them concurrently on the
    /// 16 GB target produces memory-pressure artifacts, so they serialize on
    /// this lock.
    #[cfg(target_os = "macos")]
    static REAL_MODEL_GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[cfg(target_os = "macos")]
    fn real_model_gpu_lock() -> std::sync::MutexGuard<'static, ()> {
        REAL_MODEL_GPU_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "diagnostic: prints prefill logit divergence across chunk sizes"]
    fn real_model_prefill_chunk_divergence_report() {
        let _gpu = real_model_gpu_lock();
        let reference = open_engine_at(".gguf/debug-nogf4.gguf", 64);
        let reference_logits = prefill_logits(&reference);
        for chunk in [1u32, 2, 4, 8, 9, 16] {
            let engine = open_engine_at(".gguf/debug-nogf4.gguf", chunk);
            let logits = prefill_logits(&engine);
            let max_abs = reference_logits
                .iter()
                .zip(&logits)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "chunk {chunk:>2} vs 64: max_abs={max_abs:.5} argmax {} vs {}",
                argmax(&logits),
                argmax(&reference_logits)
            );
        }
    }

    /// Teacher-force `tokens` through one of the two evaluation paths and
    /// return the full logits vector at each probe position (a probe at p
    /// yields the logits that predict token p+1).
    #[cfg(target_os = "macos")]
    fn teacher_forced_probe_logits(
        engine: &Engine,
        tokens: &[u32],
        probes: &[usize],
        single_token_path: bool,
    ) -> Vec<Vec<f32>> {
        let mut session = engine
            .metal_runtime
            .as_ref()
            .expect("real engine should have native runtime")
            .lock()
            .expect("native runtime lock should not be poisoned");
        session.runtime.reset().expect("runtime should reset");

        let mut out = Vec::with_capacity(probes.len());
        let mut start = 0usize;
        for &probe in probes {
            assert!(probe >= start && probe < tokens.len());
            if single_token_path {
                // `pos` is the absolute sequence position passed to eval_token,
                // not just an index, so the range loop is intentional.
                #[allow(clippy::needless_range_loop)]
                for pos in start..=probe {
                    let mode = if pos == probe {
                        crate::metal::LogitsMode::Full
                    } else {
                        crate::metal::LogitsMode::None
                    };
                    session
                        .runtime
                        .eval_token(tokens[pos], pos as u32, mode)
                        .expect("scalar eval should succeed");
                }
            } else {
                let chunk_len = engine.prefill_chunk.max(2) as usize;
                let mut pos0 = start;
                for chunk in tokens[start..=probe].chunks(chunk_len) {
                    let mode = if pos0 + chunk.len() == probe + 1 {
                        crate::metal::LogitsMode::Full
                    } else {
                        crate::metal::LogitsMode::None
                    };
                    session
                        .runtime
                        .eval_prefill_chunk(chunk, pos0 as u32, mode)
                        .expect("chunked eval should succeed");
                    pos0 += chunk.len();
                }
            }
            session.runtime.sync().expect("sync should succeed");
            out.push(session.runtime.copy_logits().expect("logits should copy"));
            start = probe + 1;
        }
        out
    }

    /// Diagnostic for the repetition-loop investigation: teacher-force the
    /// captured looping transcript through the chunked prefill path and the
    /// single-token decode path on the base GGUF and report per-probe argmax
    /// flips and logit divergence. Both paths use the same base weights, so any
    /// flip is pure kernel-path divergence.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "diagnostic: prints decode-vs-prefill logit parity along a repetitive transcript"]
    fn real_model_decode_path_parity_report() {
        let _gpu = real_model_gpu_lock();

        let messages = vec![ChatTurn {
            role: "user".to_string(),
            content: "make a reverse count from 218 to 156, jumping by 2 one time and other by 4"
                .to_string(),
        }];
        let mut completion = String::from(
            "Here is the reverse count from **218** down to **156**, alternating between \
             subtracting **2** and subtracting **4** at each step:\n\n**218** (start)\n\
             -2 \u{2192} **216**\n-4 \u{2192} **212**\n-2 \u{2192} **210**\n-4 \u{2192} **206**\n\
             -2 \u{2192} **206** (Wait, let's re-calculate the sequence carefully)\n\n\
             Let's trace the logic:\n1.  Start: **218**\n2.  Subtract 2: **216**\n\
             3.  Subtract 4: **206**\n4.  Subtract 2: **204**\n5.  Subtract 4: **202**\n\
             6.  Subtract 4: **200**\n7.  Subtract 2: **198**\n8.  Subtract 4: **194**\n\
             9.  Subtract 2: **194** (Wait, 194 - 4 = 190? No, 202 - 4 = 190? No. ",
        );
        for _ in 0..28 {
            completion.push_str("202 - 4 = 198? No. ");
        }

        let engine = open_real_engine_with(32, 2048);
        let mut text = render_qwen35_chat_prompt(&messages, false, false);
        text.push_str(&completion);
        let tokens = engine
            .tokenizer
            .encode(&text, true)
            .expect("transcript should tokenize");
        let probes: Vec<usize> = (15..tokens.len()).step_by(16).collect();

        let batch = teacher_forced_probe_logits(&engine, &tokens, &probes, false);
        let single = teacher_forced_probe_logits(&engine, &tokens, &probes, true);

        let mut flips = 0usize;
        let mut worst_abs = 0.0f32;
        for (idx, probe) in probes.iter().enumerate() {
            let (a, b) = (&batch[idx], &single[idx]);
            let (ta, tb) = (argmax(a), argmax(b));
            let max_abs = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            worst_abs = worst_abs.max(max_abs);
            if ta != tb {
                flips += 1;
                eprintln!(
                    "probe pos {probe:>4}: ARGMAX FLIP batch={ta} ({:+.4}) single={tb} ({:+.4}) max_abs={max_abs:.4}",
                    a[ta as usize], b[tb as usize]
                );
            }
        }
        eprintln!(
            "probes={} argmax_flips={flips} worst_max_abs_logit_diff={worst_abs:.4}",
            probes.len()
        );
    }

    /// Shared (label, user prompt, fixed assistant continuation) corpus used by
    /// both activation calibration and the deterministic quality report, so the
    /// AWQ scales are fit on the same domains they are scored on.
    #[cfg(target_os = "macos")]
    fn sidecar_calibration_corpus() -> Vec<(&'static str, &'static str, &'static str)> {
        vec![
            (
                "math-code",
                "Write a python script solve_real_root.py that finds all real roots of ((1 - x) / 5)^4 - 16 = 0",
                "We solve ((1 - x)/5)^4 = 16, so (1 - x)/5 = ±2, giving 1 - x = ±10. \
                 Thus x = 1 - 10 = -9 or x = 1 + 10 = 11. The two real roots are -9 and 11.\n\n\
                 ```python\nimport numpy as np\n\ndef real_roots():\n    \
                 roots = [1 - 5 * 2, 1 + 5 * 2]\n    return sorted(roots)\n\n\
                 if __name__ == \"__main__\":\n    print(real_roots())\n```\n",
            ),
            (
                "reasoning",
                "Explain step by step why the sky appears blue during the day.",
                "Sunlight contains all colors. As it passes through the atmosphere, shorter \
                 wavelengths (blue) scatter far more than longer ones because Rayleigh scattering \
                 grows as one over wavelength to the fourth power. That scattered blue light reaches \
                 our eyes from every direction, so the daytime sky looks blue.",
            ),
            (
                "rust-code",
                "Write an iterative Rust function that returns the nth Fibonacci number.",
                "```rust\nfn fib(n: u64) -> u64 {\n    let (mut a, mut b) = (0u64, 1u64);\n    \
                 for _ in 0..n {\n        let next = a + b;\n        a = b;\n        b = next;\n    }\n    a\n}\n```\n\
                 It runs in O(n) time and O(1) space, returning 0 for n = 0.",
            ),
            (
                "multilingual",
                "Traduci in italiano: The cat sleeps on the warm roof while it rains.",
                "Il gatto dorme sul tetto caldo mentre piove. La frase mantiene il presente \
                 e descrive un'azione continua sotto la pioggia.",
            ),
            (
                "math-word",
                "A train travels 60 km in the first hour and 90 km in the next two hours. \
                 What is its average speed over the whole trip?",
                "Total distance is 60 + 90 = 150 km. Total time is 1 + 2 = 3 hours. \
                 Average speed = distance / time = 150 / 3 = 50 km/h. Note this is the \
                 time-weighted average, not the mean of 60 and 45.",
            ),
            (
                "py-debug",
                "Why does this raise IndexError: `xs = [1,2,3]; print(xs[len(xs)])` and how do I fix it?",
                "Python lists are zero-indexed, so valid indices are 0..len(xs)-1. `xs[len(xs)]` \
                 is `xs[3]`, one past the end, which raises IndexError. Use `xs[len(xs)-1]` or \
                 `xs[-1]` to get the last element, and guard with `if xs:` for empty lists.",
            ),
            (
                "prose",
                "Summarize the water cycle in three sentences.",
                "Water evaporates from oceans, lakes, and rivers, turning into vapor that rises \
                 into the atmosphere. As the vapor cools it condenses into clouds and eventually \
                 falls back to the surface as precipitation such as rain or snow. The water then \
                 collects in bodies of water or soaks into the ground, and the cycle repeats.",
            ),
            (
                "json",
                "Return a JSON object describing a book with title, author, year, and a list of two tags.",
                "```json\n{\n  \"title\": \"The Pragmatic Programmer\",\n  \
                 \"author\": \"Andrew Hunt and David Thomas\",\n  \"year\": 1999,\n  \
                 \"tags\": [\"software\", \"craftsmanship\"]\n}\n```\n",
            ),
            (
                "sql",
                "Write a SQL query to select the names of employees in the 'Sales' department earning more than 50000.",
                "```sql\nSELECT e.name\nFROM employees AS e\nJOIN departments AS d ON e.dept_id = d.id\n\
                 WHERE d.name = 'Sales' AND e.salary > 50000\nORDER BY e.name;\n```\n",
            ),
        ]
    }

    /// Drive the calibration corpus through the capture-routed decode path so
    /// the runtime accumulates per-channel FFN activation magnitudes. Set
    /// QW35_CAPTURE_ACT_OUT to the desired stats path (defaults next to the
    /// GGUF). Uses base weights (no GF4) so the stats reflect the reference.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "diagnostic: capture per-channel FFN activation stats for AWQ"]
    fn real_model_capture_activations() {
        let _gpu = real_model_gpu_lock();
        let out = std::env::var("QW35_CAPTURE_ACT_OUT")
            .unwrap_or_else(|_| ".gguf/act-stats.bin".to_string());
        std::env::set_var("QW35_CAPTURE_ACT_OUT", &out);

        let engine = open_real_engine_with(32, 4096);
        let mut total = 0usize;
        for (label, user, assistant) in sidecar_calibration_corpus() {
            let messages = vec![ChatTurn {
                role: "user".to_string(),
                content: user.to_string(),
            }];
            let mut text = render_qwen35_chat_prompt(&messages, false, false);
            text.push_str(assistant);
            let tokens = engine.tokenizer.encode(&text, true).expect("tokenize");
            let last = tokens.len() - 1;
            let _ = teacher_forced_probe_logits(&engine, &tokens, &[last], true);
            total += tokens.len();
            eprintln!("  captured [{label}] {} tokens", tokens.len());
        }
        drop(engine); // dealloc flushes the final (<16-token) window
        eprintln!("capture complete: ~{total} tokens -> {out}");
        assert!(
            std::path::Path::new(&out).exists(),
            "capture stats file should be written"
        );
    }

    /// KL(P || Q) in nats for two logit vectors, P = softmax(reference),
    /// Q = softmax(candidate). Numerically stable via max-subtraction.
    #[cfg(target_os = "macos")]
    fn kl_div_logits(reference: &[f32], candidate: &[f32]) -> f64 {
        let lse = |v: &[f32]| -> (f32, f32) {
            let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = v.iter().map(|x| (x - m).exp()).sum();
            (m, s.ln())
        };
        let (pm, plse) = lse(reference);
        let (qm, qlse) = lse(candidate);
        let mut kl = 0.0f64;
        for (a, b) in reference.iter().zip(candidate) {
            let log_p = (a - pm - plse) as f64;
            let log_q = (b - qm - qlse) as f64;
            let p = log_p.exp();
            if p > 0.0 {
                kl += p * (log_p - log_q);
            }
        }
        kl
    }

    /// Diagnostic: print qw35's token ids for the cross-engine repro prompt
    /// so they can be diffed against `llama-server /tokenize` output.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "diagnostic: dumps tokenization of the cross-engine repro prompt"]
    fn real_model_tokenization_dump() {
        let _gpu = real_model_gpu_lock();
        let engine = open_real_engine_with(32, 128);
        let prompt = "<|im_start|>user\nmake a reverse count from 218 to 156, jumping by 2 one time and other by 4<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let ids = engine.tokenizer.encode(prompt, true).expect("encode");
        eprintln!("qw35 tokens ({}): {:?}", ids.len(), ids);
    }

    #[cfg(target_os = "macos")]
    fn open_engine_at(path: &str, prefill_chunk: u32) -> Engine {
        open_engine_at_ctx(path, prefill_chunk, 128)
    }

    #[cfg(target_os = "macos")]
    fn open_engine_at_ctx(path: &str, prefill_chunk: u32, ctx_size: u32) -> Engine {
        Engine::open(EngineConfig {
            model_path: PathBuf::from(path),
            model_id: DEFAULT_MODEL_ID.to_string(),
            ctx_size,
            prefill_chunk,
            kv_cache_type: crate::metal::KvCacheType::Q8_0,
            session_cache: false,
            attn_window: 0,
            attn_sink: 0,
            warm_weights: false,
            test_responder: false,
            verbose: false,
        })
        .unwrap_or_else(|err| panic!("failed to open real native engine: {err}"))
    }

    /// Cross-engine deterministic quality of the unified .gguf: its GF4/AWQ DECODE
    /// path vs the base GGUF reference (its Q4_K PREFILL path), teacher-forced on
    /// the SAME tokens. The unified .gguf runs GF4 on both of a single engine's
    /// paths, so an intra-engine prefill-vs-decode report cannot measure
    /// quality-vs-base for it — this opens the two models SEQUENTIALLY
    /// (base first: capture reference logits, then drop it; candidate second) so
    /// peak memory is a single model. Set QW35_MODEL_UNDER_TEST to the unified
    /// .gguf (default .gguf/Qwowl3.5-9B.gguf); QW35_BASE_MODEL overrides
    /// the reference (default the Q4_K GGUF). Reports argmax flips, top-1
    /// agreement, mean/worst KL(base||candidate), worst |Δlogit|, and the first
    /// divergence.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "diagnostic: cross-engine quality of the unified .gguf vs the base GGUF"]
    fn real_model_unified_quality_report() {
        let _gpu = real_model_gpu_lock();
        let base_path = std::env::var("QW35_BASE_MODEL")
            .unwrap_or_else(|_| ".gguf/Qwen3.5-9B-Q4_K_M.gguf".to_string());
        let cand_path = std::env::var("QW35_MODEL_UNDER_TEST")
            .unwrap_or_else(|_| ".gguf/Qwowl3.5-9B.gguf".to_string());
        let corpus = sidecar_calibration_corpus();

        // Pass 1 — base reference (Q4_K prefill path). Capture tokens, probe
        // positions, and reference logits per transcript, then free the engine.
        let mut refs: Vec<(String, Vec<u32>, Vec<usize>, Vec<Vec<f32>>)> = Vec::new();
        {
            let base = open_engine_at_ctx(&base_path, 32, 4096);
            for (label, user, assistant) in &corpus {
                let messages = vec![ChatTurn {
                    role: "user".to_string(),
                    content: (*user).to_string(),
                }];
                let mut text = render_qwen35_chat_prompt(&messages, false, false);
                text.push_str(assistant);
                let tokens = base.tokenizer.encode(&text, true).expect("tokenize");
                let begin = (tokens.len() / 4).max(15);
                let probes: Vec<usize> = (begin..tokens.len()).step_by(16).collect();
                if probes.is_empty() {
                    continue;
                }
                let ref_logits = teacher_forced_probe_logits(&base, &tokens, &probes, false);
                refs.push(((*label).to_string(), tokens, probes, ref_logits));
            }
            drop(base);
        }

        // Pass 2 — candidate decode path (GF4/AWQ), scored on the same tokens.
        let cand = open_engine_at_ctx(&cand_path, 32, 4096);
        eprintln!(
            "unified_quality: base={base_path} candidate={cand_path} ffn={}",
            cand.ffn_label()
        );
        let mut total_probes = 0usize;
        let mut total_flips = 0usize;
        let mut sum_kl = 0.0f64;
        let mut worst_abs = 0.0f32;
        let mut worst_kl = 0.0f64;
        let mut first_divergence: Option<(usize, String)> = None;
        for (label, tokens, probes, ref_logits) in &refs {
            let cand_logits = teacher_forced_probe_logits(&cand, tokens, probes, true);
            let mut flips = 0usize;
            let mut kl_sum = 0.0f64;
            for (idx, &probe) in probes.iter().enumerate() {
                let (a, b) = (&ref_logits[idx], &cand_logits[idx]);
                let (ta, tb) = (argmax(a), argmax(b));
                let max_abs = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                let kl = kl_div_logits(a, b);
                worst_abs = worst_abs.max(max_abs);
                worst_kl = worst_kl.max(kl);
                kl_sum += kl;
                if ta != tb {
                    flips += 1;
                    if first_divergence.is_none() {
                        first_divergence = Some((probe, format!("{label}: base={ta} cand={tb}")));
                    }
                }
            }
            total_probes += probes.len();
            total_flips += flips;
            sum_kl += kl_sum;
            eprintln!(
                "  [{label:<12}] probes={:>3} flips={flips:>2} mean_kl={:.5}",
                probes.len(),
                kl_sum / probes.len() as f64,
            );
        }
        let agree = if total_probes > 0 {
            100.0 * (total_probes - total_flips) as f64 / total_probes as f64
        } else {
            0.0
        };
        eprintln!(
            "unified_quality TOTAL: probes={total_probes} flips={total_flips} \
             top1_agreement={agree:.2}% mean_kl={:.5} worst_kl={worst_kl:.5} worst_abs={worst_abs:.4}",
            sum_kl / total_probes.max(1) as f64,
        );
        match &first_divergence {
            Some((probe, what)) => eprintln!("  first divergence at probe pos {probe}: {what}"),
            None => eprintln!("  no argmax divergence across the corpus"),
        }
    }

    #[cfg(target_os = "macos")]
    fn open_real_engine(prefill_chunk: u32) -> Engine {
        open_real_engine_with(prefill_chunk, 128)
    }

    #[cfg(target_os = "macos")]
    fn open_real_engine_with(prefill_chunk: u32, ctx_size: u32) -> Engine {
        Engine::open(EngineConfig {
            model_path: PathBuf::from(".gguf/Qwen3.5-9B-Q4_K_M.gguf"),
            model_id: DEFAULT_MODEL_ID.to_string(),
            ctx_size,
            prefill_chunk,
            kv_cache_type: crate::metal::KvCacheType::Q8_0,
            session_cache: false,
            attn_window: 0,
            attn_sink: 0,
            warm_weights: false,
            test_responder: false,
            verbose: false,
        })
        .unwrap_or_else(|err| panic!("failed to open real native engine: {err}"))
    }

    #[cfg(target_os = "macos")]
    fn prefill_logits(engine: &Engine) -> Vec<f32> {
        let prompt_tokens = test_prompt_tokens(engine);
        assert!(prompt_tokens.len() > 8);

        let mut session = engine
            .metal_runtime
            .as_ref()
            .expect("real engine should have native runtime")
            .lock()
            .expect("native runtime lock should not be poisoned");
        session.runtime.reset().expect("runtime should reset");
        eval_prompt_tokens(
            engine,
            &mut session.runtime,
            &prompt_tokens,
            crate::metal::LogitsMode::Full,
        );
        session.runtime.copy_logits().expect("logits should copy")
    }

    #[cfg(target_os = "macos")]
    fn greedy_tokens_after_prefill(engine: &Engine, count: usize) -> Vec<u32> {
        let prompt_tokens = test_prompt_tokens(engine);
        assert!(prompt_tokens.len() > 8);

        let mut session = engine
            .metal_runtime
            .as_ref()
            .expect("real engine should have native runtime")
            .lock()
            .expect("native runtime lock should not be poisoned");
        session.runtime.reset().expect("runtime should reset");
        eval_prompt_tokens(
            engine,
            &mut session.runtime,
            &prompt_tokens,
            crate::metal::LogitsMode::Argmax,
        );

        let mut out = Vec::with_capacity(count);
        for step in 0..count {
            let next = session
                .runtime
                .argmax()
                .expect("argmax should produce a token")
                .0;
            out.push(next);
            if step + 1 < count {
                let pos = prompt_tokens.len() as u32 + step as u32;
                session
                    .runtime
                    .eval_token(next, pos, crate::metal::LogitsMode::Argmax)
                    .expect("decode token should evaluate");
            }
        }
        out
    }

    #[cfg(target_os = "macos")]
    fn test_prompt_tokens(engine: &Engine) -> Vec<u32> {
        let request = GenerateRequest {
            model: DEFAULT_MODEL_ID.to_string(),
            messages: vec![ChatTurn {
                role: "user".to_string(),
                content: "Say hi, then name one color.".to_string(),
            }],
            max_tokens: TokenLimit::Fixed(1),
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            repeat_last_n: -1,
            enable_thinking: false,
            preserve_thinking: false,
            thinking_budget: None,
            reasoning_budget_message: None,
            ignore_eos: true,
            stop_sequences: Vec::new(),
            emit_reasoning: false,
        };
        let prompt = render_qwen35_chat_prompt(
            &request.messages,
            request.enable_thinking,
            request.preserve_thinking,
        );
        
        engine
            .tokenizer
            .encode(&prompt, true)
            .expect("test prompt should tokenize")
    }

    #[cfg(target_os = "macos")]
    fn eval_prompt_tokens(
        engine: &Engine,
        runtime: &mut crate::metal::MetalRuntime,
        prompt_tokens: &[u32],
        head_mode: crate::metal::LogitsMode,
    ) {
        if engine.prefill_chunk <= 1 {
            let last = prompt_tokens.len() - 1;
            for (pos, &token) in prompt_tokens.iter().enumerate() {
                let mode = if pos == last {
                    head_mode
                } else {
                    crate::metal::LogitsMode::None
                };
                runtime
                    .eval_token(token, pos as u32, mode)
                    .expect("scalar prefill should evaluate");
            }
        } else {
            let chunk_len = engine.prefill_chunk as usize;
            for (chunk_idx, chunk_tokens) in prompt_tokens.chunks(chunk_len).enumerate() {
                let pos0 = (chunk_idx * chunk_len) as u32;
                let mode = if pos0 as usize + chunk_tokens.len() == prompt_tokens.len() {
                    head_mode
                } else {
                    crate::metal::LogitsMode::None
                };
                runtime
                    .eval_prefill_chunk(chunk_tokens, pos0, mode)
                    .expect("chunked prefill should evaluate");
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx as u32)
            .expect("logits should not be empty")
    }
}
