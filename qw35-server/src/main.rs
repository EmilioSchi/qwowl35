use qw35_server::metal::KvCacheType;
use qw35_server::model::{
    validate_prefill_chunk, Engine, EngineConfig, TokenLimit, DEFAULT_CHECKPOINT_CAP,
    DEFAULT_MODEL_ID, DEFAULT_PREFILL_CHUNK, DEFAULT_SCRATCH_CTX,
};
use qw35_server::reranker::{RerankEngine, RerankerConfig, DEFAULT_RERANKER_CTX};
use qw35_server::server::{self, GenerationDefaults};
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

// Cwd-relative default model path and the downloader that fetches it. Kept in
// one place so the `Config` default and the missing-model helper can't drift.
const DEFAULT_MODEL_PATH: &str = ".gguf/Qwowl3.5-9B.gguf";
const DOWNLOAD_SCRIPT: &str = "download_model.sh";
// Auto-loaded reranker (POST /v1/rerank) when present on disk: the RAW q8_0
// rank conversion, deliberately not the GF4 cook (rerank is prefill-bound —
// GF4 gains no speed there and costs ranking fidelity on a 0.6B; see
// MODEL_CARD). --reranker-model overrides; --no-reranker disables.
const DEFAULT_RERANKER_PATH: &str = ".gguf/qwen3-reranker-0.6b-q8_0.gguf";

// 128K is the throughput sweet spot: decode holds full speed (~19.3 tok/s) and
// time-to-first-token stays under ~0.5 s, whereas the full 262144 window collapses
// prefill to ~2.5 tok/s and TTFT to ~3 s (decode also drops ~16%). Pass --ctx 262144
// to use the full trained window when prompts genuinely need it.
const DEFAULT_CTX_SIZE: u32 = 131_072;

fn main() -> ExitCode {
    match Config::parse(std::env::args().skip(1)) {
        Ok(Command::Help) => {
            print_help();
            ExitCode::SUCCESS
        }
        Ok(Command::Run(config)) => run(config),
        Err(err) => {
            eprintln!("qw35: {err}");
            eprintln!("Try `qw35 --help`.");
            ExitCode::from(2)
        }
    }
}

fn run(config: Config) -> ExitCode {
    if let Some(code) = ensure_model_available(&config.model_path) {
        return code;
    }

    let engine_config = EngineConfig {
        model_path: config.model_path.clone(),
        model_id: config.model_id.clone(),
        ctx_size: config.ctx_size,
        prefill_chunk: config.prefill_chunk,
        kv_cache_type: config.kv_cache_type,
        session_cache: config.session_cache,
        checkpoint_cap: config.checkpoint_cap,
        scratch_ctx: config.scratch_ctx,
        // Decode-time sliding-window attention, passed straight to the Metal
        // runtime over the FFI (no env vars). Off by default (full attention).
        attn_window: config.attn_window,
        attn_sink: config.attn_sink,
        warm_weights: config.warm_weights,
        test_responder: config.test_responder,
        verbose: config.verbose,
    };

    // Weight/KV GPU residency pinning (MTLResidencySet, macOS 15+) is always on:
    // the unified .gguf footprint (~5 GiB, one mmap, no duplicate FFN) fits
    // comfortably resident on 16 GiB, keeping weight-bandwidth-bound decode fast
    // over a long session. The Metal runtime pins unconditionally at engine init.

    let engine = match Engine::open(engine_config) {
        Ok(engine) => Arc::new(engine),
        Err(err) => {
            eprintln!("qw35: failed to open engine: {err}");
            return ExitCode::FAILURE;
        }
    };

    // Optional second model: the reranker, its own engine over its own mmap.
    // An explicit --reranker-model must load or startup fails; otherwise the
    // default reranker GGUF is picked up automatically when it exists on disk
    // (skip with --no-reranker). Without a reranker /v1/rerank answers 501.
    let reranker_path = match &config.reranker_model {
        Some(path) => Some(path.clone()),
        None if !config.no_reranker && Path::new(DEFAULT_RERANKER_PATH).exists() => {
            Some(PathBuf::from(DEFAULT_RERANKER_PATH))
        }
        None => None,
    };
    let reranker = match &reranker_path {
        Some(path) => match RerankEngine::open(RerankerConfig {
            model_path: path.clone(),
            ctx_size: config.reranker_ctx,
            prefill_chunk: config.prefill_chunk,
            kv_cache_type: config.kv_cache_type,
            verbose: config.verbose,
        }) {
            Ok(reranker) => Some(Arc::new(reranker)),
            Err(err) => {
                eprintln!("qw35: failed to open reranker: {err}");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };

    let summary = engine.metadata_summary();
    let gen = &config.generation_defaults;
    eprintln!("qw35: mapped {}", engine.model_path().display());
    eprintln!(
        "  size={:.2} GiB",
        summary.mapped_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );
    eprintln!("  tensors={}", summary.tensor_count);
    eprintln!("  arch={}", summary.architecture);
    eprintln!("  ctx={}", config.ctx_size);
    eprintln!("  decoder_ready={}", engine.decoder_ready());
    eprintln!("  prefill_chunk={}", engine.prefill_chunk());
    eprintln!(
        "  ffn={}",
        engine.ffn_label()
    );
    eprintln!("  kv_cache={}", engine.kv_cache_type().as_str());
    if config.attn_window > 0 {
        eprintln!(
            "  attn_window={} sink={} (decode-only; bounds attention to keep tok/s flat)",
            config.attn_window, config.attn_sink
        );
    }
    match &reranker {
        Some(reranker) => eprintln!(
            "  reranker={} (ctx={}, ffn={})",
            reranker.model_name(),
            reranker.ctx_size(),
            reranker.ffn_label()
        ),
        None => eprintln!("  reranker=off"),
    }
    eprintln!(
        "  session_cache={}",
        if engine.session_cache() { "on" } else { "off" }
    );
    if engine.session_cache() {
        let snapshot = engine.state_snapshot_size();
        eprintln!(
            "  checkpoints={} (snapshot {:.1} MiB each, {:.1} MiB max)",
            engine.checkpoint_cap(),
            snapshot as f64 / 1024.0 / 1024.0,
            (engine.checkpoint_cap() * snapshot) as f64 / 1024.0 / 1024.0
        );
    }
    eprintln!("  temperature={}", gen.temperature);
    eprintln!("  top_p={}", gen.top_p);
    eprintln!("  top_k={}", gen.top_k);
    eprintln!("  min_p={}", gen.min_p);
    eprintln!("  presence_penalty={}", gen.presence_penalty);
    eprintln!("  repetition_penalty={}", gen.repetition_penalty);
    eprintln!("  repeat_last_n={}", gen.repeat_last_n);
    eprintln!("  enable_thinking={}", gen.enable_thinking);

    let plan = engine.graph_plan();
    if config.verbose {
        eprintln!(
            "qw35: graph plan delta_layers={}, attention_layers={}, unsupported_tensor_types={}",
            plan.delta_layers.len(),
            plan.attention_layers.len(),
            if plan.unsupported_tensor_types.is_empty() {
                "none".to_string()
            } else {
                plan.unsupported_tensor_types.join(",")
            }
        );
    }
    if !plan.execution_blockers.is_empty() && !config.test_responder {
        eprintln!(
            "qw35: decoder blockers: {}",
            plan.execution_blockers.join("; ")
        );
    }
    if let Some(report) = engine.warm_report() {
        eprintln!(
            "qw35: warmed mapped tensor pages via {}: {:.2} GiB in {:.3}s ({} touches, {} views, mapped {:.2} GiB, checksum={})",
            report.mode,
            report.bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            report.elapsed.as_secs_f64(),
            report.touched_pages,
            report.view_count,
            report.mapped_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            report.checksum
        );
        if let Some(err) = &report.madvise_error {
            eprintln!("qw35: warning: POSIX_MADV_WILLNEED failed: {err}");
        }
    }

    if config.check_only {
        return ExitCode::SUCCESS;
    }

    if !config.test_responder && !engine.decoder_ready() {
        eprintln!(
            "qw35: native decoder is not ready yet; /health and /v1/models will serve, chat returns 501 inference_unavailable"
        );
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    match rt.block_on(server::serve(
        &config.host,
        config.port,
        engine,
        reranker,
        config.generation_defaults,
    )) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("qw35: {err}");
            ExitCode::FAILURE
        }
    }
}

enum Command {
    Help,
    Run(Config),
}

struct Config {
    model_path: PathBuf,
    model_id: String,
    reranker_model: Option<PathBuf>,
    no_reranker: bool,
    reranker_ctx: u32,
    host: String,
    port: u16,
    ctx_size: u32,
    prefill_chunk: u32,
    kv_cache_type: KvCacheType,
    session_cache: bool,
    checkpoint_cap: usize,
    scratch_ctx: u32,
    attn_window: i32,
    attn_sink: i32,
    generation_defaults: GenerationDefaults,
    warm_weights: bool,
    test_responder: bool,
    check_only: bool,
    verbose: bool,
}

/// Per-parameter generation-default overrides collected during arg parsing.
/// Kept as `Option`s so resolution is order-independent: the `--mode` preset is
/// the base and any explicitly-passed flag overlays it, regardless of the order
/// the flags appear on the command line.
#[derive(Default)]
struct DefaultOverrides {
    max_tokens: Option<TokenLimit>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    min_p: Option<f32>,
    presence_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    repeat_last_n: Option<i32>,
    reasoning_budget_message: Option<String>,
}

// When the default model is missing, offer to fetch it with the downloader
// (only on a TTY — "sotto richiesta dell'user"); otherwise print a clear hint.
// Returns Some(exit code) to stop startup, or None to continue loading. A
// missing *custom* `-m` path is left untouched so `Engine::open` emits its
// canonical error.
fn ensure_model_available(model_path: &Path) -> Option<ExitCode> {
    if model_path.exists() {
        return None;
    }
    let is_default = model_path == Path::new(DEFAULT_MODEL_PATH);
    let script = Path::new(DOWNLOAD_SCRIPT);
    if !is_default || !script.exists() {
        return None;
    }

    if io::stdin().is_terminal() {
        eprintln!("qw35: model not found at {}", model_path.display());
        eprint!("qw35: download it now with ./{DOWNLOAD_SCRIPT}? [y/N] ");
        let _ = io::stderr().flush();
        let mut answer = String::new();
        if io::stdin().read_line(&mut answer).is_ok()
            && matches!(answer.trim(), "y" | "Y" | "yes" | "Yes")
        {
            match std::process::Command::new("sh").arg(DOWNLOAD_SCRIPT).status() {
                Ok(status) if status.success() && model_path.exists() => return None,
                _ => eprintln!(
                    "qw35: download did not produce {}; aborting.",
                    model_path.display()
                ),
            }
        }
        Some(ExitCode::FAILURE)
    } else {
        eprintln!(
            "qw35: model not found at {}. Run ./{DOWNLOAD_SCRIPT} to fetch it.",
            model_path.display()
        );
        Some(ExitCode::FAILURE)
    }
}

impl Config {
    fn parse<I>(args: I) -> Result<Command, String>
    where
        I: IntoIterator<Item = String>,
    {
        let mut config = Config {
            model_path: PathBuf::from(DEFAULT_MODEL_PATH),
            model_id: DEFAULT_MODEL_ID.to_string(),
            reranker_model: None,
            no_reranker: false,
            reranker_ctx: DEFAULT_RERANKER_CTX,
            host: "127.0.0.1".to_string(),
            port: 8080,
            ctx_size: DEFAULT_CTX_SIZE,
            prefill_chunk: DEFAULT_PREFILL_CHUNK,
            kv_cache_type: KvCacheType::Q8_0,
            session_cache: true,
            checkpoint_cap: DEFAULT_CHECKPOINT_CAP,
            scratch_ctx: DEFAULT_SCRATCH_CTX,
            attn_window: 0,
            attn_sink: 0,
            generation_defaults: GenerationDefaults::default(),
            warm_weights: false,
            test_responder: false,
            check_only: false,
            verbose: false,
        };

        // Resolved after the loop so flag order doesn't matter (see
        // DefaultOverrides). `--mode` seeds the base profile; `ov` overlays it.
        let mut mode: Option<server::Mode> = None;
        let mut ov = DefaultOverrides::default();

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => return Ok(Command::Help),
                "-m" | "--model" => config.model_path = PathBuf::from(need_arg(&arg, &mut args)?),
                "--model-id" => config.model_id = need_arg(&arg, &mut args)?,
                "--reranker-model" => {
                    config.reranker_model = Some(PathBuf::from(need_arg(&arg, &mut args)?));
                }
                "--no-reranker" => config.no_reranker = true,
                "--reranker-ctx" => {
                    config.reranker_ctx = parse_u32(&arg, &need_arg(&arg, &mut args)?)?;
                }
                "--host" => config.host = need_arg(&arg, &mut args)?,
                "--port" => {
                    config.port = parse_u16(&arg, &need_arg(&arg, &mut args)?)?;
                }
                "-c" | "--ctx" | "--num-ctx" => {
                    config.ctx_size = parse_u32(&arg, &need_arg(&arg, &mut args)?)?;
                }
                "--prefill-chunk" => {
                    config.prefill_chunk = parse_u32(&arg, &need_arg(&arg, &mut args)?)?;
                }
                "--kv-cache-type" => {
                    let value = need_arg(&arg, &mut args)?;
                    config.kv_cache_type = KvCacheType::parse(&value).ok_or_else(|| {
                        format!("invalid --kv-cache-type {value:?}: expected f16 or q8_0")
                    })?;
                }
                "--no-session-cache" => config.session_cache = false,
                "--checkpoints" => {
                    config.checkpoint_cap = parse_u32(&arg, &need_arg(&arg, &mut args)?)? as usize;
                }
                "--scratch-ctx" => {
                    config.scratch_ctx = parse_u32(&arg, &need_arg(&arg, &mut args)?)?;
                }
                "--attn-window" => {
                    config.attn_window = parse_u32(&arg, &need_arg(&arg, &mut args)?)? as i32;
                }
                "--attn-sink" => {
                    config.attn_sink = parse_u32(&arg, &need_arg(&arg, &mut args)?)? as i32;
                }
                "--mode" => {
                    let name = need_arg(&arg, &mut args)?;
                    mode = Some(server::Mode::from_name(&name).ok_or_else(|| {
                        format!(
                            "invalid --mode {name:?}: expected thinking-general, thinking-coding, \
                             instruct-general, or instruct-reasoning"
                        )
                    })?);
                }
                "-n" | "--tokens" => {
                    ov.max_tokens = Some(TokenLimit::Fixed(parse_u32(
                        &arg,
                        &need_arg(&arg, &mut args)?,
                    )?));
                }
                "--num-predict" => {
                    let value = parse_i64(&arg, &need_arg(&arg, &mut args)?)?;
                    ov.max_tokens = Some(server::parse_token_limit(&arg, value)?);
                }
                "--temperature" => {
                    ov.temperature = Some(parse_f32(&arg, &need_arg(&arg, &mut args)?)?);
                }
                "--top-p" => {
                    ov.top_p = Some(parse_f32(&arg, &need_arg(&arg, &mut args)?)?);
                }
                "--top-k" => {
                    ov.top_k = Some(parse_u32(&arg, &need_arg(&arg, &mut args)?)?);
                }
                "--min-p" => {
                    ov.min_p = Some(parse_f32(&arg, &need_arg(&arg, &mut args)?)?);
                }
                "--presence-penalty" => {
                    ov.presence_penalty = Some(parse_f32(&arg, &need_arg(&arg, &mut args)?)?);
                }
                "--repeat-penalty" | "--repetition-penalty" => {
                    ov.repetition_penalty = Some(parse_f32(&arg, &need_arg(&arg, &mut args)?)?);
                }
                "--repeat-last-n" => {
                    ov.repeat_last_n = Some(parse_i32(&arg, &need_arg(&arg, &mut args)?)?);
                }
                "--reasoning-budget-message" => {
                    ov.reasoning_budget_message = Some(need_arg(&arg, &mut args)?);
                }
                "--warm-weights" => config.warm_weights = true,
                "--test-responder" => config.test_responder = true,
                "--check" => config.check_only = true,
                "-v" | "--verbose" => config.verbose = true,
                other => return Err(format!("unknown option {other}")),
            }
        }

        if config.ctx_size == 0 {
            return Err("--ctx must be greater than zero".to_string());
        }

        // Resolve defaults: start from the --mode preset (or the thinking-coding
        // preset when no mode is given), then overlay any explicit flags.
        let mut defaults = mode.unwrap_or(server::Mode::ThinkingCoding).defaults();
        if let Some(v) = ov.max_tokens {
            defaults.max_tokens = v;
        }
        if let Some(v) = ov.temperature {
            defaults.temperature = v;
        }
        if let Some(v) = ov.top_p {
            defaults.top_p = v;
        }
        if let Some(v) = ov.top_k {
            defaults.top_k = v;
        }
        if let Some(v) = ov.min_p {
            defaults.min_p = v;
        }
        if let Some(v) = ov.presence_penalty {
            defaults.presence_penalty = v;
        }
        if let Some(v) = ov.repetition_penalty {
            defaults.repetition_penalty = v;
        }
        if let Some(v) = ov.repeat_last_n {
            defaults.repeat_last_n = v;
        }
        if let Some(v) = ov.reasoning_budget_message {
            // An empty value disables the wrap-up message (bare `</think>`).
            defaults.reasoning_budget_message = if v.is_empty() { None } else { Some(v) };
        }
        config.generation_defaults = defaults;
        config.generation_defaults.validate()?;
        validate_prefill_chunk(config.prefill_chunk)?;

        Ok(Command::Run(config))
    }
}

fn need_arg<I>(flag: &str, args: &mut I) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("missing value for {flag}: expected {}", arg_hint(flag)))
}

/// Describes the value each valued flag expects, so a bare `--flag` with no
/// argument tells the user what to write instead of just "missing value".
fn arg_hint(flag: &str) -> &'static str {
    match flag {
        "-m" | "--model" => "a GGUF model file path",
        "--model-id" => "the served OpenAI model id (e.g. qwen35-9b)",
        "--reranker-model" => "a reranker GGUF file path (e.g. .gguf/Qwowl3-Reranker-0.6B.gguf)",
        "--reranker-ctx" => "the reranker context size in tokens (e.g. 2048)",
        "--host" => "a bind address (e.g. 127.0.0.1)",
        "--port" => "a port number (e.g. 8080)",
        "-c" | "--ctx" | "--num-ctx" => "a context size in tokens (e.g. 120000)",
        "--prefill-chunk" => "a chunk size in tokens (e.g. 32)",
        "--kv-cache-type" => "one of: f16, q8_0",
        "--mode" => {
            "one of: thinking-general, thinking-coding, instruct-general, instruct-reasoning"
        }
        "-n" | "--tokens" => "a max output token count (e.g. 8192)",
        "--num-predict" => "a max output token count, or -1 for unlimited",
        "--temperature" => "a temperature (e.g. 0.35; 0 for greedy decoding)",
        "--top-p" => "a nucleus probability (e.g. 0.95)",
        "--top-k" => "a top-k limit (e.g. 20; 0 to disable)",
        "--min-p" => "a minimum probability ratio (e.g. 0)",
        "--presence-penalty" => "a presence penalty (e.g. 0.3)",
        "--repeat-penalty" | "--repetition-penalty" => "a repetition penalty (e.g. 1.1)",
        "--repeat-last-n" => "a repeat window in tokens (e.g. 64)",
        "--reasoning-budget-message" => "a wrap-up message forced before </think> (empty to disable)",
        _ => "a value",
    }
}

fn parse_u16(flag: &str, value: &str) -> Result<u16, String> {
    value
        .parse::<u16>()
        .map_err(|err| format!("invalid {flag} value {value:?}: {err}"))
}

fn parse_u32(flag: &str, value: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|err| format!("invalid {flag} value {value:?}: {err}"))
}

fn parse_i32(flag: &str, value: &str) -> Result<i32, String> {
    value
        .parse::<i32>()
        .map_err(|err| format!("invalid {flag} value {value:?}: {err}"))
}

fn parse_i64(flag: &str, value: &str) -> Result<i64, String> {
    value
        .parse::<i64>()
        .map_err(|err| format!("invalid {flag} value {value:?}: {err}"))
}

fn parse_f32(flag: &str, value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|err| format!("invalid {flag} value {value:?}: {err}"))?;
    if !parsed.is_finite() {
        return Err(format!("{flag} must be finite"));
    }
    Ok(parsed)
}

fn print_help() {
    println!(
        "Usage: qw35 [options]\n\
\n\
Model and runtime:\n\
  -m, --model FILE\n\
      GGUF model path. Default: .gguf/Qwowl3.5-9B.gguf\n\
  --model-id ID\n\
      Served OpenAI model id. Default: qwen35-9b\n\
  --reranker-model FILE\n\
      Qwen3-Reranker GGUF (rank conversion with a yes/no classification head) served alongside the main model on POST /v1/rerank. When omitted, .gguf/qwen3-reranker-0.6b-q8_0.gguf is auto-loaded if it exists (fetch it with ./download_model.sh reranker); with no reranker at all /v1/rerank answers 501 inference_unavailable.\n\
  --no-reranker\n\
      Skip the reranker entirely, including the auto-load of the default reranker GGUF.\n\
  --reranker-ctx N\n\
      Context size allocated for the reranker session (one (query, document) prompt must fit; longer documents are truncated). Default: 2048\n\
  -c, --ctx N, --num-ctx N\n\
      Context size allocated by the inference backend. Default: 131072 (128K — the throughput sweet spot: full decode speed and sub-0.5s TTFT; ~2.1 GiB of q8_0 KV cache). Pass --ctx 262144 for the full trained window (collapses prefill to ~2.5 tok/s and TTFT to ~3s). With --kv-cache-type f16 the cache is ~1.9x larger — pass a smaller --ctx on memory-constrained machines.\n\
  --kv-cache-type TYPE\n\
      Attention KV cache storage: q8_0 (8.5 bits/element blocks, default) or f16. q8_0 is byte-identical to f16 output in testing at fp16-parity decode speed and ~1.9x less memory; f16 is the lossless reference, kept for KV-cache comparisons.\n\
  --no-session-cache\n\
      Disable the session prefix cache. By default the server keeps the KV cache and SSM state alive between requests and re-evaluates only the new suffix when a prompt extends the previous conversation.\n\
  --checkpoints N\n\
      Max recurrent-state snapshots kept at prompt history boundaries for prefix rewinds (the session checkpoint stack). Each costs ~state-size bytes of host RAM (tens of MB, printed at startup). 1 = legacy single-checkpoint behavior; 0 = extend-or-reset only. Default: 8\n\
  --scratch-ctx N\n\
      Initial context size of the lazily-created scratch/plan GPU sessions (qw35_session=\"scratch\"/\"plan\" requests: standalone contexts like qwowl35's editor and planner that must not clobber the main session's KV rows and checkpoints). A session whose prompt outgrows it is grown on demand in 8192-token steps up to --ctx (the KV cache extends lazily, so unused ceiling costs nothing). 0 disables the aux sessions. Default: 16384\n\
  --attn-window N\n\
      Decode-time sliding-window attention for the 8 full-attention layers: attend only to the last N positions (plus --attn-sink leading tokens), bounding the per-token attention cost so decode tok/s stays FLAT across a long session instead of degrading with context (e.g. ~13 tok/s at any ctx vs ~10 at 8K / falling further). Default 0 = off (full attention). TRADE-OFF: the model loses exact recall of content older than the window in those layers (the DeltaNet layers carry only compressed long-range), so enable it only when sustained speed matters more than long-range recall. Suggested: 2048-4096.\n\
  --attn-sink N\n\
      Leading sink tokens always attended to under --attn-window (StreamingLLM-style). Default 0; 4-8 recommended when --attn-window is set.\n\
  -n, --tokens N\n\
      Default max output tokens when the client omits a limit. Default: -1 (generate until EOS or remaining context is exhausted)\n\
  --num-predict N\n\
      Default max output tokens when the client omits a limit. Use -1 to generate until EOS or remaining context is exhausted. Default: -1\n\
  --prefill-chunk N\n\
      Prompt tokens to evaluate per native Metal prefill chunk. Use 1 for the scalar compatibility path; any larger value uses the Q4_K/Q5_K/Q6_K tiled prefill path, fastest at multiples of 32. Default: 32\n\
  --warm-weights\n\
      Touch mapped tensor pages using the mmap warmup policy. This can pull the full 5+ GiB model into memory/file cache; default is off.\n\
  --test-responder\n\
      Serve a deterministic responder for HTTP and CLI harness tests without running model inference.\n\
  --check\n\
      Map and validate the model, print the startup summary, then exit without serving.\n\
  -v, --verbose\n\
      Print additional startup detail (graph plan summary) and a per-request\n\
      line with prefill/decode token counts, durations, speeds, session-cache\n\
      reuse, and time to first token. Logged after each request completes, so\n\
      it does not affect inference speed.\n\
\n\
HTTP API:\n\
  --host HOST\n\
      Bind address. Default: 127.0.0.1\n\
  --port N\n\
      Bind port. Default: 8080\n\
\n\
Mode preset:\n\
  --mode NAME\n\
      Seed the sampling defaults and the think/no-think default from an official Qwen3.5 profile. Per-request params (and explicit sampling flags below) override individual fields. Default: thinking-coding (thinking on, temperature 0.6, presence 0.0, repeat 1.1). Values:\n\
        thinking-general    thinking on,  temperature 1.0, top_p 0.95, top_k 20, presence 1.5, repeat 1.1\n\
        thinking-coding     thinking on,  temperature 0.6, top_p 0.95, top_k 20, presence 0.0, repeat 1.1\n\
        instruct-general    thinking off, temperature 0.7, top_p 0.80, top_k 20, presence 1.5, repeat 1.0\n\
        instruct-reasoning  thinking off, temperature 1.0, top_p 0.95, top_k 20, presence 1.5, repeat 1.0\n\
\n\
Sampling defaults:\n\
  --temperature F\n\
      Default request temperature when the client omits one. Use 0 for greedy decoding. Default: 0.6 (thinking-coding preset). Overrides --mode.\n\
  --top-k N\n\
      Default top-k sampling limit. Use 0 to disable top-k filtering. Default: 20\n\
  --top-p F\n\
      Default nucleus sampling probability. Default: 0.95\n\
  --min-p F\n\
      Default minimum probability ratio. Default: 0\n\
  --presence-penalty F\n\
      Default presence penalty when the client omits one. OpenAI semantics: a flat logit subtraction for any token already present in the generated output (never the prompt). Default: 0.3 (agentic-coding profile)\n\
  --repeat-penalty F, --repetition-penalty F\n\
      Default repetition penalty when the client omits one. Use 1 to disable. Windowed by --repeat-last-n so recurring code syntax outside the window is never penalized. Applies only to sampled decode (temperature > 0); greedy argmax never sees penalties. Default: 1.1\n\
  --repeat-last-n N\n\
      Repetition penalty window. Use -1 for all seen tokens, 0 to disable, or a positive token count. Default: 64\n\
  --reasoning-budget-message MSG\n\
      Wrap-up message forced just before </think> when the thinking budget is exhausted, so the model conditions on an 'answer now' handoff. Empty string disables it (bare </think>). Per-request 'reasoning_budget_message' overrides it. Default: a short first-person handoff.\n\
\n\
Supported endpoints:\n\
  GET  /health\n\
  GET  /v1/models\n\
  GET  /v1/models/{{model}}\n\
  POST /v1/chat/completions\n\
  POST /v1/rerank            (needs the reranker model, auto-loaded when present; body: query, documents[], optional instruction/top_n)\n\
  POST /v1/responses\n\
  POST /v1/responses/input_tokens\n\
\n\
Request fields accepted by /v1/chat/completions:\n\
  temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty,\n\
  repeat_last_n, preserve_thinking, ignore_eos, max_tokens,\n\
  max_completion_tokens, num_predict, stop (string or array),\n\
  tools, tool_choice, response_format (text or json_object).\n\
  enable_thinking controls the Qwen3.5 generation prompt (default false)\n\
  and routes thinking into message.reasoning_content.\n\
  Tool calls stream as OpenAI-style tool_calls deltas (stable index, id and\n\
  function.name in the first delta, incremental arguments fragments) and\n\
  finish with finish_reason \"tool_calls\"; truncation reports \"length\".\n\
  n > 1 and response_format json_schema are rejected with 400;\n\
  seed, frequency_penalty, logit_bias, logprobs, and user are ignored.\n\
\n\
Responses compatibility:\n\
  /v1/responses accepts text, message, reasoning, and tool replay input items,\n\
  injects function tools into the Qwen 3.5 prompt, parses model tool calls\n\
  into function_call output items (arguments as a JSON string), streams\n\
  function_call_arguments.delta/done events with sequence_number on every\n\
  event, and reports incomplete_details max_output_tokens on truncation.\n\
  previous_response_id and conversation are rejected because qw35 does not\n\
  persist OpenAI server-side response state; replay the full input instead.\n\
\n\
Example:\n\
  cargo run --release -- --port 8080\n"
    );
}
