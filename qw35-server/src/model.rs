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
mod tool_penalty;
use prompt::{
    render_qwen35_chat_prompt, render_qwen35_chat_prompt_with_boundaries,
};
use sampling::sample_from_logits;
use stop_sequence::{earliest_stop_match, StopSequenceWatcher};
use text_filter::GeneratedTextFilter;
use think_budget::ThinkBudget;
use tool_penalty::ToolCallPenaltyGuard;

pub const DEFAULT_MODEL_ID: &str = "qwen3.5-9b";
pub const DEFAULT_PREFILL_CHUNK: u32 = 32;
pub const MAX_PREFILL_CHUNK: u32 = 256;
/// Default depth of the session checkpoint stack (recurrent-state snapshots
/// at prompt history boundaries). Each snapshot costs `state_size()` bytes of
/// host RAM (~tens of MB); see --checkpoints.
pub const DEFAULT_CHECKPOINT_CAP: usize = 8;
/// Default context size of the lazily-created scratch session (judge/editor
/// standalone contexts are short); see --scratch-ctx.
pub const DEFAULT_SCRATCH_CTX: u32 = 16384;
/// KV slab granularity of the Metal cache (mirrors QW35_KV_INITIAL_SLAB in
/// Qw35MetalRuntime.m); on-demand context growth rounds new ceilings up to it.
const CTX_GROW_SLAB: u32 = 8192;
/// Decode headroom guaranteed when growing a session's context for an
/// until-EOS request (TokenLimit::Context), so a grown session never resumes
/// with a sliver of output budget.
const CTX_GROW_HEADROOM: u32 = 8192;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_path: PathBuf,
    pub model_id: String,
    pub ctx_size: u32,
    pub prefill_chunk: u32,
    pub kv_cache_type: metal::KvCacheType,
    pub session_cache: bool,
    /// Max recurrent-state snapshots kept for prefix rewinds (--checkpoints).
    /// 1 reproduces the legacy single-checkpoint behavior; 0 disables rewinds
    /// (extend-or-reset only).
    pub checkpoint_cap: usize,
    /// Context size of the lazily-created scratch session (--scratch-ctx).
    /// 0 disables the scratch session (qw35_session=scratch requests fail).
    pub scratch_ctx: u32,
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
    /// Stream tool-call bodies incrementally as raw XML fragments plus
    /// `qw35_tool_call` side-channel chunks (qwowl35's live display). Off =
    /// the OpenAI-compatible buffered behavior. Only the chat-completions
    /// streaming path honors it.
    pub stream_tool_call_xml: bool,
    /// Parse model-emitted `<tool_call>` XML into structured tool_calls.
    /// True only when the request advertised tools (or a named tool_choice
    /// forced a call opening); otherwise the XML passes through verbatim as
    /// content so plain-text clients see it instead of losing it.
    pub parse_tool_calls: bool,
    /// `tool_choice: required`/named must-call instruction, rendered as a
    /// user turn past the stable prompt boundary so it applies to this
    /// generation only and never enters the session-cache checkpoint prefix.
    pub tool_choice_enforcement: Option<String>,
    /// Named-tool_choice hard enforcement: the `<tool_call>\n<function=X>\n`
    /// opening injected into the prompt right after the generation header
    /// (same volatile region), so the model can only complete the call's
    /// parameters. Prompt bytes never reach the emitted text, so generation
    /// seeds the text stream and the penalty guard with the prefix. The
    /// caller must disable thinking when setting this (the render falls back
    /// to the closed-think header regardless).
    pub forced_tool_prefix: Option<String>,
    /// Which GPU session to run on (`qw35_session` request field). Scratch
    /// keeps short standalone contexts off the main lineage so they never
    /// invalidate its KV rows or checkpoints.
    pub session: SessionKind,
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
    /// Which GPU session served the request (main lineage or scratch).
    pub session: SessionKind,
    /// The serving session's live context ceiling after any on-demand growth.
    /// Clients size their context-usage display against this.
    pub session_ctx: u32,
    /// Entries in the session's checkpoint stack after this request.
    pub checkpoint_depth: u32,
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

/// Which GPU session a request runs on. `Main` is the long-lived pipeline
/// lineage with the checkpoint stack. `Scratch` and `Plan` are smaller
/// auxiliary runtimes for contexts that must not clobber each other or the
/// main lineage (KV rows are positional; a divergent prompt resets from
/// position 0): qwowl35 runs judge/editor interludes on `Scratch` and keeps
/// the planner's persistent context on `Plan`, so the plan↔execute
/// alternation never re-prefills the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SessionKind {
    #[default]
    Main,
    Scratch,
    Plan,
}

impl SessionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SessionKind::Main => "main",
            SessionKind::Scratch => "scratch",
            SessionKind::Plan => "plan",
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
    /// Smaller auxiliary runtimes, created lazily on first use (weights are
    /// no-copy views over the shared mmap; the real cost per slot is one KV
    /// slab + activation buffers + pipeline warmup). `None` inside the mutex
    /// until then. `scratch` hosts short interludes (qwowl35's judge and
    /// editor); `plan` hosts the planner's persistent context so the
    /// plan↔execute alternation never re-prefills it.
    scratch_runtime: Mutex<Option<RuntimeSession>>,
    plan_runtime: Mutex<Option<RuntimeSession>>,
    // The GPU reads the GF4 FFN weights directly from this mapping; it must
    // outlive metal_runtime (fields drop in declaration order).
    gguf: MappedGguf,
    model_path: PathBuf,
    model_id: String,
    ctx_size: u32,
    prefill_chunk: u32,
    kv_cache_type: metal::KvCacheType,
    session_cache: bool,
    checkpoint_cap: usize,
    scratch_ctx: u32,
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
/// `checkpoints` holds recurrent-state snapshots taken at successive prompt
/// history boundaries; the hybrid SSM state cannot rewind, so these are the
/// only rollback points.
struct RuntimeSession {
    runtime: metal::MetalRuntime,
    evaluated: Vec<u32>,
    checkpoints: CheckpointStack,
    /// The session's live context ceiling. Starts at the configured size
    /// (--ctx for main, --scratch-ctx for aux) and grows on demand toward
    /// --ctx when a prompt outgrows it; the Metal KV cache extends lazily
    /// underneath, so growth is a scalar bump, not a reallocation.
    ctx_limit: u32,
}

/// One recurrent-state snapshot: the token prefix it corresponds to plus the
/// exported SSM/conv state bytes. KV rows for positions below `tokens.len()`
/// stay valid in the shared cache as long as nothing shallower is re-evaluated,
/// so restoring is an import + suffix prefill.
struct Checkpoint {
    tokens: Vec<u32>,
    state: Box<[u8]>,
    last_used: u64,
}

/// A small set of checkpoints along the session's evaluated lineage. All
/// entries are prefixes of one token history (deeper entries extend shallower
/// ones); `plan_session_reuse` picks the deepest entry matching a new prompt,
/// so short rewinds (dropping one tool exchange, starting the next pipeline
/// stage or todo task) never re-prefill the kept context.
struct CheckpointStack {
    entries: Vec<Checkpoint>,
    cap: usize,
    clock: u64,
}

impl CheckpointStack {
    fn new(cap: usize) -> Self {
        Self {
            entries: Vec::new(),
            cap,
            clock: 0,
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    /// Deepest entry whose tokens are a strict prefix of `prompt` (at least
    /// one prompt token must remain to evaluate).
    fn deepest_match(&self, prompt: &[u32]) -> Option<(usize, usize)> {
        let cap = prompt.len().saturating_sub(1);
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                let len = entry.tokens.len();
                len > 0 && len <= cap && prompt[..len] == entry.tokens[..]
            })
            .max_by_key(|(_, entry)| entry.tokens.len())
            .map(|(index, entry)| (index, entry.tokens.len()))
    }

    fn state_of(&self, index: usize) -> &[u8] {
        &self.entries[index].state
    }

    /// After restoring `index`, prefill overwrites the KV rows above its
    /// boundary: every deeper entry's rows are about to become garbage, so
    /// drop them. Shallower entries keep only rows below their own (smaller)
    /// boundary and stay valid. Also refreshes the restored entry's LRU stamp.
    fn mark_restored(&mut self, index: usize) {
        self.clock += 1;
        self.entries[index].last_used = self.clock;
        let len = self.entries[index].tokens.len();
        self.entries.retain(|entry| entry.tokens.len() <= len);
    }

    /// Record a snapshot at a new boundary. An entry with identical tokens is
    /// refreshed instead of duplicated. On overflow the least-recently-used
    /// entry is evicted, but never the deepest (the equivalent of the old
    /// single checkpoint — the most likely rewind target for the next
    /// request) and never the entry just saved.
    fn save(&mut self, tokens: Vec<u32>, state: Box<[u8]>) {
        if self.cap == 0 {
            return;
        }
        self.clock += 1;
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.tokens == tokens) {
            entry.state = state;
            entry.last_used = self.clock;
            return;
        }
        self.entries.push(Checkpoint {
            tokens,
            state,
            last_used: self.clock,
        });
        if self.entries.len() > self.cap {
            let saved = self.entries.len() - 1;
            let deepest = self
                .entries
                .iter()
                .enumerate()
                .max_by_key(|(_, entry)| entry.tokens.len())
                .map(|(index, _)| index)
                .unwrap_or(saved);
            let victim = self
                .entries
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != saved && *index != deepest)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(index, _)| index)
                // Everything but the just-saved entry is protected only when
                // cap forces a choice between it and the deepest; keep the
                // fresh save (legacy single-slot semantics) and let the old
                // deepest go.
                .or_else(|| {
                    self.entries
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| *index != saved)
                        .min_by_key(|(_, entry)| entry.last_used)
                        .map(|(index, _)| index)
                });
            if let Some(victim) = victim {
                self.entries.remove(victim);
            }
        }
    }
}

/// How much of the live session state a new prompt can reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionReuse {
    /// Prompt strictly extends the evaluated tokens: prefill from this offset.
    Extend(usize),
    /// Prompt matches a checkpointed prefix: restore that entry's recurrent
    /// state and prefill from its boundary.
    Checkpoint { index: usize, len: usize },
    Reset,
}

/// Decide how to reuse session state for `prompt`. At least one prompt token
/// must always be (re-)evaluated so the output head runs on the last token.
fn plan_session_reuse(
    evaluated: &[u32],
    checkpoints: &CheckpointStack,
    prompt: &[u32],
) -> SessionReuse {
    let cap = prompt.len().saturating_sub(1);
    if !evaluated.is_empty() && evaluated.len() <= cap && prompt[..evaluated.len()] == *evaluated {
        return SessionReuse::Extend(evaluated.len());
    }
    if let Some((index, len)) = checkpoints.deepest_match(prompt) {
        return SessionReuse::Checkpoint { index, len };
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
                checkpoints: CheckpointStack::new(config.checkpoint_cap),
                ctx_limit: config.ctx_size,
            }))
        };

        Ok(Self {
            metal_runtime,
            scratch_runtime: Mutex::new(None),
            plan_runtime: Mutex::new(None),
            gguf,
            model_path: config.model_path,
            model_id: config.model_id,
            ctx_size: config.ctx_size,
            prefill_chunk: config.prefill_chunk,
            kv_cache_type: config.kv_cache_type,
            session_cache: config.session_cache,
            checkpoint_cap: config.checkpoint_cap,
            scratch_ctx: config.scratch_ctx,
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

    pub fn checkpoint_cap(&self) -> usize {
        self.checkpoint_cap
    }

    /// Build an auxiliary RuntimeSession (scratch or plan): same model and
    /// knobs as the main session but a smaller context and a minimal
    /// checkpoint stack. Two snapshot slots (not the main cap): a multi-turn
    /// aux context (the editor agent, the planner's review pings) re-renders
    /// its history each turn, so without a boundary snapshot every
    /// continuation would reset and re-prefill the whole context; the
    /// snapshot size is ctx-independent, so this costs ~two state copies of
    /// host RAM per slot.
    fn open_aux_session(&self, kind: SessionKind) -> Result<RuntimeSession, String> {
        if self.scratch_ctx == 0 {
            return Err(format!(
                "the {} session is disabled (--scratch-ctx 0); run this request on the main session",
                kind.as_str()
            ));
        }
        let vocab_size = self
            .tokenizer
            .spec
            .vocab_size
            .try_into()
            .map_err(|_| "tokenizer vocabulary is too large for the native Metal runtime".to_string())?;
        if self.verbose {
            eprintln!(
                "qw35: creating {} session (ctx={}, kv={})",
                kind.as_str(),
                self.scratch_ctx,
                self.kv_cache_type.as_str()
            );
        }
        Ok(RuntimeSession {
            runtime: metal::MetalRuntime::new(
                &self.gguf,
                &self.graph_plan.hparams,
                self.scratch_ctx,
                vocab_size,
                self.prefill_chunk,
                self.kv_cache_type,
                self.attn_window,
                self.attn_sink,
            )?,
            evaluated: Vec::new(),
            checkpoints: CheckpointStack::new(2),
            ctx_limit: self.scratch_ctx,
        })
    }

    /// Raise `session`'s context ceiling so the prompt plus its decode budget
    /// fits, instead of rejecting the request outright. The ceiling grows in
    /// KV-slab steps and is capped at --ctx, so an aux session that outgrows
    /// its cheap initial size ends up with the same budget as the main
    /// session. Growth failures are logged and non-fatal: admission then runs
    /// against the old ceiling and reports the honest limit error.
    fn ensure_session_ctx(
        &self,
        session: &mut RuntimeSession,
        kind: SessionKind,
        limit: TokenLimit,
        prompt_tokens: usize,
    ) -> u32 {
        let desired_output = match limit {
            TokenLimit::Fixed(max_tokens) => max_tokens as u64,
            TokenLimit::Context => CTX_GROW_HEADROOM as u64,
        };
        let needed = prompt_tokens as u64 + desired_output;
        let cap = self.ctx_size as u64;
        if needed <= session.ctx_limit as u64 || session.ctx_limit as u64 >= cap {
            return session.ctx_limit;
        }
        let slab = CTX_GROW_SLAB as u64;
        let target = (needed.div_ceil(slab) * slab).min(cap) as u32;
        if target <= session.ctx_limit {
            return session.ctx_limit;
        }
        match session.runtime.set_ctx_size(target) {
            Ok(()) => {
                eprintln!(
                    "qw35: grew {} session ctx {} -> {} (prompt {} tok)",
                    kind.as_str(),
                    session.ctx_limit,
                    target,
                    prompt_tokens
                );
                session.ctx_limit = target;
            }
            Err(err) => {
                eprintln!("qw35: failed to grow {} session ctx: {err}", kind.as_str());
            }
        }
        session.ctx_limit
    }

    /// Measured byte size of one recurrent-state snapshot (0 without the
    /// native runtime). Startup logs it so the --checkpoints RAM budget
    /// (cap x this) is visible.
    pub fn state_snapshot_size(&self) -> usize {
        self.metal_runtime
            .as_ref()
            .and_then(|runtime| runtime.lock().ok())
            .map(|session| session.runtime.state_size())
            .unwrap_or(0)
    }

    /// True when decode uses baked FFN weights — i.e. a unified .gguf whose
    /// FFN carries the GF4 (100) or GF2 (101) codec on any layer (a plain
    /// base GGUF decodes on Q4_K instead).
    pub fn gf4_active(&self) -> bool {
        let (gf4, gf2) = self.ffn_codec_layers();
        gf4 + gf2 > 0
    }

    /// Count layers whose FFN is baked as (GF4, GF2). A layer is attributed
    /// by its `ffn_gate` tensor; codecs may be mixed across layers in one
    /// unified .gguf.
    fn ffn_codec_layers(&self) -> (u32, u32) {
        let blocks = self.gguf.metadata_u32("qwen35.block_count").unwrap_or(0);
        let mut gf4 = 0;
        let mut gf2 = 0;
        for layer in 0..blocks {
            let name = format!("blk.{layer}.ffn_gate.weight");
            match self.gguf.tensor(&name).map(|t| t.type_id) {
                Some(100) => gf4 += 1,
                Some(101) => gf2 += 1,
                _ => {}
            }
        }
        (gf4, gf2)
    }

    /// Label for the startup summary `ffn=` line and /health.
    pub fn ffn_label(&self) -> String {
        match self.ffn_codec_layers() {
            (0, 0) => "gguf".to_string(),
            (_, 0) => "gf4-unified".to_string(),
            (0, _) => "gf2-unified".to_string(),
            (gf4, gf2) => format!("gf4+gf2({gf4}+{gf2})"),
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
            return self.generate_native(request, |_| Ok(()), |_, _, _| {});
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
            request.tool_choice_enforcement.as_deref(),
            request.forced_tool_prefix.as_deref(),
        );
        let render_duration = render_start.elapsed();

        let tokenize_start = Instant::now();
        let prompt_tokens = self
            .tokenizer
            .encode(&prompt, true)
            .map(|tokens| tokens.len() as u32)
            .unwrap_or_else(|_| rough_token_count(&prompt));
        let max_tokens =
            self.resolve_max_tokens(request.max_tokens, prompt_tokens as usize, self.ctx_size)?;
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
                session_ctx: self.ctx_size,
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
        self.generate_stream_with_progress(request, on_text, |_, _, _| {})
    }

    /// Like [`generate_stream`], but also reports prompt-processing (prefill)
    /// progress through `on_prefill(processed_tokens, total_tokens,
    /// session_ctx)` — the third value is the serving session's live context
    /// ceiling after any on-demand growth, so clients can scale a usage
    /// display. This is a pure server-side side-channel; the OpenAI streaming
    /// contract is unchanged (the handler ships progress in choice-less
    /// chunks that clients ignore).
    pub fn generate_stream_with_progress<F, P>(
        &self,
        request: &GenerateRequest,
        mut on_text: F,
        on_prefill: P,
    ) -> Result<Generation, GenerateError>
    where
        F: FnMut(&str) -> Result<(), String>,
        P: FnMut(usize, usize, u32),
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
            request.tool_choice_enforcement.as_deref(),
            request.forced_tool_prefix.as_deref(),
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
        session_ctx: u32,
    ) -> Result<u32, GenerateError> {
        let ctx_size = session_ctx as usize;
        if prompt_tokens >= ctx_size {
            return Err(GenerateError::BadRequest(format!(
                "prompt requires {prompt_tokens} context slots, but the session context is {session_ctx}"
            )));
        }

        match limit {
            TokenLimit::Fixed(max_tokens) => {
                let max_needed = prompt_tokens.saturating_add(max_tokens as usize);
                if max_needed > ctx_size {
                    return Err(GenerateError::BadRequest(format!(
                        "prompt plus max_tokens requires {max_needed} context slots, but the session context is {session_ctx}"
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
        mut report_prefill: P,
    ) -> Result<Generation, GenerateError>
    where
        F: FnMut(&str) -> Result<(), String>,
        P: FnMut(usize, usize, u32),
    {
        let total_start = Instant::now();
        let (prompt, stable_len, preamble_len) = render_qwen35_chat_prompt_with_boundaries(
            &request.messages,
            request.enable_thinking,
            request.preserve_thinking,
            request.tool_choice_enforcement.as_deref(),
            request.forced_tool_prefix.as_deref(),
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
        let lock_start = Instant::now();
        let mut main_guard;
        let mut aux_guard;
        let session: &mut RuntimeSession = match request.session {
            SessionKind::Main => {
                let runtime = self.metal_runtime.as_ref().ok_or_else(|| {
                    GenerateError::InferenceUnavailable(self.inference_unavailable_message())
                })?;
                main_guard = runtime.lock().map_err(|_| {
                    GenerateError::InferenceUnavailable(
                        "native Metal runtime lock is poisoned".to_string(),
                    )
                })?;
                &mut *main_guard
            }
            SessionKind::Scratch | SessionKind::Plan => {
                let slot = match request.session {
                    SessionKind::Plan => &self.plan_runtime,
                    _ => &self.scratch_runtime,
                };
                aux_guard = slot.lock().map_err(|_| {
                    GenerateError::InferenceUnavailable(
                        "auxiliary Metal runtime lock is poisoned".to_string(),
                    )
                })?;
                if aux_guard.is_none() {
                    // Lazy build under the lock: only ever one aux client per
                    // slot (the qwowl35 orchestrator runs agents sequentially).
                    *aux_guard = Some(
                        self.open_aux_session(request.session)
                            .map_err(GenerateError::InferenceUnavailable)?,
                    );
                }
                aux_guard.as_mut().expect("aux session just created")
            }
        };
        let runtime_lock_duration = lock_start.elapsed();

        // Admission runs against the session's LIVE ceiling, grown on demand
        // toward --ctx first: a long conversation upgrades its session instead
        // of dying on the aux size it happened to start with.
        let session_ctx = self.ensure_session_ctx(
            session,
            request.session,
            request.max_tokens,
            prompt_tokens.len(),
        );
        let max_tokens =
            self.resolve_max_tokens(request.max_tokens, prompt_tokens.len(), session_ctx)?;
        // Prefill progress carries the (possibly just-grown) live ceiling.
        let mut on_prefill =
            |processed: usize, total: usize| report_prefill(processed, total, session_ctx);

        // Pin system prompt + first user turn under the sliding window: the
        // effective sink is at least the preamble length, so the windowed
        // attention layers keep seeing the tool-call format and the task.
        if self.attn_window > 0 {
            let sink = self.attn_sink.max(preamble_tokens as i32);
            session.runtime.set_attn_sink(sink);
        }

        // Session prefix cache: reuse live GPU state when the new prompt
        // extends it, rewind to the deepest matching boundary checkpoint when
        // it diverged further back, else start from scratch.
        let reuse = if self.session_cache {
            plan_session_reuse(&session.evaluated, &session.checkpoints, &prompt_tokens)
        } else {
            SessionReuse::Reset
        };

        let reset_start = Instant::now();
        let (cached, session_path) = match reuse {
            SessionReuse::Extend(n) => (n, SessionPath::Extend),
            SessionReuse::Checkpoint { index, len } => {
                let restored = {
                    let state = session.checkpoints.state_of(index);
                    session.runtime.state_import(state).is_ok()
                };
                if restored {
                    // Prefill will overwrite KV rows above this boundary:
                    // deeper entries become garbage and are dropped.
                    session.checkpoints.mark_restored(index);
                    (len, SessionPath::Checkpoint)
                } else {
                    session.checkpoints.clear();
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
            // The full prompt will overwrite KV rows from position 0, so
            // every checkpoint's cache rows are no longer trustworthy.
            session.checkpoints.clear();
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
            let mut state = vec![0u8; session.runtime.state_size()].into_boxed_slice();
            if !state.is_empty() && session.runtime.state_export(&mut state).is_ok() {
                session
                    .checkpoints
                    .save(prompt_tokens[..history_len].to_vec(), state);
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

        // Suspend sampling penalties inside <tool_call> blocks (see
        // `ToolCallPenaltyGuard`): the body is a verbatim payload and the
        // unwindowed presence penalty otherwise corrupts long commands.
        let mut tool_penalty_guard = ToolCallPenaltyGuard::new(
            self.tokenizer.spec.tool_call_token_id,
            self.tokenizer.spec.end_tool_call_token_id,
        );

        // A forced tool-call opening was prefilled as prompt bytes, which the
        // emitted text and the penalty guard never see: seed both, so the
        // downstream parser reconstructs a complete `<tool_call>` block (the
        // OpenAI tool_calls array and finish_reason depend on it) and
        // penalties stay suspended inside the forced body. The stop watcher
        // is deliberately not fed — the forced region cannot end generation.
        if let Some(prefix) = request.forced_tool_prefix.as_deref() {
            tool_penalty_guard.arm();
            let callback_start = Instant::now();
            on_text(prefix).map_err(GenerateError::InferenceUnavailable)?;
            stream_callback_duration += callback_start.elapsed();
            rendered.push_str(prefix);
        }

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
                        sample_from_logits(
                            &logits,
                            request,
                            &all_seen,
                            prompt_tokens.len(),
                            tool_penalty_guard.active(),
                        )
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
            tool_penalty_guard.observe(next);

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
                session: request.session,
                session_ctx,
                checkpoint_depth: session.checkpoints.entries.len() as u32,
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
#[path = "tests/model.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/model_checkpoints.rs"]
mod tests_checkpoints;
