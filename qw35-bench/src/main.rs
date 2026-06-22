use qw35_server::gguf::MappedGguf;
use qw35_server::graph::plan_qwen35;
use qw35_server::model::{
    validate_prefill_chunk, Engine, EngineConfig, GenerateRequest, GenerationTimings, TokenLimit,
    DEFAULT_MODEL_ID, DEFAULT_PREFILL_CHUNK,
};
use qw35_server::tokenizer::{DecodeState, QwenTokenizer};
use serde::Serialize;
use serde_json::Value;
use std::fs::File;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

const DEFAULT_URL: &str = "http://127.0.0.1:8080";
const DEFAULT_PROMPT: &str = "Write a short technical paragraph about how to separate network latency from model inference latency in a local benchmark.";

fn main() -> ExitCode {
    match parse_command(std::env::args().skip(1)) {
        Ok(Command::Help) => {
            print_help();
            ExitCode::SUCCESS
        }
        Ok(Command::Http(config)) => exit(run_http(config)),
        Ok(Command::Direct(config)) => exit(run_direct(config)),
        Ok(Command::Host(config)) => exit(run_host(config)),
        Err(err) => {
            eprintln!("qw35_bench: {err}");
            eprintln!("Try `qw35_bench --help`.");
            ExitCode::from(2)
        }
    }
}

fn exit(result: Result<(), String>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("qw35_bench: {err}");
            ExitCode::FAILURE
        }
    }
}

enum Command {
    Help,
    Http(HttpConfig),
    Direct(DirectConfig),
    Host(HostConfig),
}

#[derive(Clone)]
struct PromptConfig {
    model: String,
    system: Option<String>,
    prompt: String,
    tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    min_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    repetition_penalty: f32,
    enable_thinking: bool,
    preserve_thinking: bool,
    ignore_eos: bool,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL_ID.to_string(),
            system: Some("Answer directly and avoid filler.".to_string()),
            prompt: DEFAULT_PROMPT.to_string(),
            tokens: 128,
            temperature: 0.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            enable_thinking: false,
            preserve_thinking: false,
            ignore_eos: false,
        }
    }
}

struct HttpConfig {
    url: HttpUrl,
    endpoint: HttpEndpoint,
    runs: u32,
    warmup_runs: u32,
    timeout: Duration,
    prompt: PromptConfig,
    output: OutputConfig,
}

struct DirectConfig {
    model_path: PathBuf,
    ctx_size: u32,
    runs: u32,
    warmup_runs: u32,
    warm_weights: bool,
    test_responder: bool,
    reopen_each_run: bool,
    prefill_chunk: u32,
    kv_cache_type: qw35_server::metal::KvCacheType,
    prompt: PromptConfig,
    output: OutputConfig,
}

struct HostConfig {
    model_path: PathBuf,
    ctx_size: u32,
    runs: u32,
    prompt: PromptConfig,
    output: OutputConfig,
}

#[derive(Clone, Copy)]
enum HttpEndpoint {
    Health,
    Models,
    Chat,
    Stream,
    All,
}

impl HttpEndpoint {
    fn all(self) -> Vec<HttpEndpoint> {
        match self {
            HttpEndpoint::All => vec![
                HttpEndpoint::Health,
                HttpEndpoint::Models,
                HttpEndpoint::Chat,
                HttpEndpoint::Stream,
            ],
            other => vec![other],
        }
    }

    fn name(self) -> &'static str {
        match self {
            HttpEndpoint::Health => "health",
            HttpEndpoint::Models => "models",
            HttpEndpoint::Chat => "chat",
            HttpEndpoint::Stream => "stream",
            HttpEndpoint::All => "all",
        }
    }

    fn path(self) -> &'static str {
        match self {
            HttpEndpoint::Health => "/health",
            HttpEndpoint::Models => "/v1/models",
            HttpEndpoint::Chat | HttpEndpoint::Stream => "/v1/chat/completions",
            HttpEndpoint::All => unreachable!("expanded before request"),
        }
    }
}

#[derive(Clone)]
struct HttpUrl {
    host: String,
    port: u16,
    base_path: String,
}

impl HttpUrl {
    fn parse(raw: &str) -> Result<Self, String> {
        let rest = raw
            .strip_prefix("http://")
            .ok_or("only http:// benchmark URLs are supported")?;
        let (host_port, path) = rest.split_once('/').unwrap_or((rest, ""));
        if host_port.is_empty() {
            return Err("benchmark URL is missing a host".to_string());
        }
        let (host, port) = if let Some((host, port)) = host_port.rsplit_once(':') {
            let port = port
                .parse::<u16>()
                .map_err(|err| format!("invalid URL port {port:?}: {err}"))?;
            (host.to_string(), port)
        } else {
            (host_port.to_string(), 80)
        };
        if host.is_empty() {
            return Err("benchmark URL is missing a host".to_string());
        }
        let base_path = if path.is_empty() {
            String::new()
        } else {
            format!("/{}", path.trim_end_matches('/'))
        };
        Ok(Self {
            host,
            port,
            base_path,
        })
    }

    fn request_path(&self, endpoint_path: &str) -> String {
        if self.base_path.is_empty() {
            endpoint_path.to_string()
        } else {
            format!("{}{}", self.base_path, endpoint_path)
        }
    }

    fn host_header(&self) -> String {
        if self.port == 80 {
            self.host.clone()
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }
}

struct OutputConfig {
    format: OutputFormat,
    path: Option<PathBuf>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Csv,
            path: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Csv,
    Jsonl,
}

#[derive(Default, Serialize)]
struct BenchRow {
    mode: String,
    endpoint: String,
    run: u32,
    ok: bool,
    status: Option<u16>,
    error: Option<String>,
    load_ms: Option<f64>,
    connect_ms: Option<f64>,
    write_ms: Option<f64>,
    first_byte_ms: Option<f64>,
    headers_ms: Option<f64>,
    first_body_byte_ms: Option<f64>,
    first_event_ms: Option<f64>,
    first_content_ms: Option<f64>,
    done_ms: Option<f64>,
    body_ms: Option<f64>,
    total_ms: f64,
    client_minus_engine_ms: Option<f64>,
    request_bytes: u64,
    response_header_bytes: u64,
    response_body_bytes: u64,
    sse_events: u64,
    sse_content_events: u64,
    sse_content_bytes: u64,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    engine_total_ms: Option<f64>,
    engine_render_ms: Option<f64>,
    engine_tokenize_ms: Option<f64>,
    engine_runtime_lock_ms: Option<f64>,
    engine_reset_ms: Option<f64>,
    engine_prompt_eval_count: Option<u64>,
    engine_prompt_eval_ms: Option<f64>,
    engine_prompt_eval_tps: Option<f64>,
    engine_eval_count: Option<u64>,
    engine_eval_ms: Option<f64>,
    engine_eval_tps: Option<f64>,
    engine_decode_eval_ms: Option<f64>,
    engine_sample_ms: Option<f64>,
    engine_detokenize_ms: Option<f64>,
    engine_stream_callback_ms: Option<f64>,
    engine_first_token_ms: Option<f64>,
    engine_prefill_chunk: Option<u64>,
    engine_prefill_path: Option<String>,
    host_old_decode_ms: Option<f64>,
    host_incremental_decode_ms: Option<f64>,
    host_decode_speedup: Option<f64>,
    estimated_reset_before_bytes: Option<u64>,
    estimated_reset_after_bytes: Option<u64>,
    estimated_reset_saved_bytes: Option<u64>,
    estimated_prompt_output_projections_before: Option<u64>,
    estimated_prompt_output_projections_after: Option<u64>,
    estimated_prompt_output_projections_saved: Option<u64>,
    estimated_prompt_output_weight_bytes_saved: Option<u64>,
}

struct BenchSink {
    format: OutputFormat,
    writer: Box<dyn Write>,
    wrote_header: bool,
}

impl BenchSink {
    fn open(config: &OutputConfig) -> Result<Self, String> {
        let writer: Box<dyn Write> = match &config.path {
            Some(path) => Box::new(
                File::create(path)
                    .map_err(|err| format!("failed to create {}: {err}", path.display()))?,
            ),
            None => Box::new(std::io::stdout()),
        };
        Ok(Self {
            format: config.format,
            writer,
            wrote_header: false,
        })
    }

    fn write_row(&mut self, row: &BenchRow) -> Result<(), String> {
        match self.format {
            OutputFormat::Csv => {
                if !self.wrote_header {
                    writeln!(self.writer, "{}", CSV_HEADER.join(","))
                        .map_err(|err| format!("failed to write CSV header: {err}"))?;
                    self.wrote_header = true;
                }
                write_csv_row(&mut self.writer, row)
            }
            OutputFormat::Jsonl => {
                serde_json::to_writer(&mut self.writer, row)
                    .map_err(|err| format!("failed to write JSONL row: {err}"))?;
                writeln!(self.writer).map_err(|err| format!("failed to write JSONL row: {err}"))
            }
        }
    }
}

const CSV_HEADER: &[&str] = &[
    "mode",
    "endpoint",
    "run",
    "ok",
    "status",
    "error",
    "load_ms",
    "connect_ms",
    "write_ms",
    "first_byte_ms",
    "headers_ms",
    "first_body_byte_ms",
    "first_event_ms",
    "first_content_ms",
    "done_ms",
    "body_ms",
    "total_ms",
    "client_minus_engine_ms",
    "request_bytes",
    "response_header_bytes",
    "response_body_bytes",
    "sse_events",
    "sse_content_events",
    "sse_content_bytes",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "engine_total_ms",
    "engine_render_ms",
    "engine_tokenize_ms",
    "engine_runtime_lock_ms",
    "engine_reset_ms",
    "engine_prompt_eval_count",
    "engine_prompt_eval_ms",
    "engine_prompt_eval_tps",
    "engine_eval_count",
    "engine_eval_ms",
    "engine_eval_tps",
    "engine_decode_eval_ms",
    "engine_sample_ms",
    "engine_detokenize_ms",
    "engine_stream_callback_ms",
    "engine_first_token_ms",
    "engine_prefill_chunk",
    "engine_prefill_path",
    "host_old_decode_ms",
    "host_incremental_decode_ms",
    "host_decode_speedup",
    "estimated_reset_before_bytes",
    "estimated_reset_after_bytes",
    "estimated_reset_saved_bytes",
    "estimated_prompt_output_projections_before",
    "estimated_prompt_output_projections_after",
    "estimated_prompt_output_projections_saved",
    "estimated_prompt_output_weight_bytes_saved",
];

fn write_csv_row(writer: &mut Box<dyn Write>, row: &BenchRow) -> Result<(), String> {
    let values = [
        row.mode.clone(),
        row.endpoint.clone(),
        row.run.to_string(),
        row.ok.to_string(),
        fmt_opt_u16(row.status),
        row.error.clone().unwrap_or_default(),
        fmt_opt_f64(row.load_ms),
        fmt_opt_f64(row.connect_ms),
        fmt_opt_f64(row.write_ms),
        fmt_opt_f64(row.first_byte_ms),
        fmt_opt_f64(row.headers_ms),
        fmt_opt_f64(row.first_body_byte_ms),
        fmt_opt_f64(row.first_event_ms),
        fmt_opt_f64(row.first_content_ms),
        fmt_opt_f64(row.done_ms),
        fmt_opt_f64(row.body_ms),
        fmt_f64(row.total_ms),
        fmt_opt_f64(row.client_minus_engine_ms),
        row.request_bytes.to_string(),
        row.response_header_bytes.to_string(),
        row.response_body_bytes.to_string(),
        row.sse_events.to_string(),
        row.sse_content_events.to_string(),
        row.sse_content_bytes.to_string(),
        fmt_opt_u64(row.prompt_tokens),
        fmt_opt_u64(row.completion_tokens),
        fmt_opt_u64(row.total_tokens),
        fmt_opt_f64(row.engine_total_ms),
        fmt_opt_f64(row.engine_render_ms),
        fmt_opt_f64(row.engine_tokenize_ms),
        fmt_opt_f64(row.engine_runtime_lock_ms),
        fmt_opt_f64(row.engine_reset_ms),
        fmt_opt_u64(row.engine_prompt_eval_count),
        fmt_opt_f64(row.engine_prompt_eval_ms),
        fmt_opt_f64(row.engine_prompt_eval_tps),
        fmt_opt_u64(row.engine_eval_count),
        fmt_opt_f64(row.engine_eval_ms),
        fmt_opt_f64(row.engine_eval_tps),
        fmt_opt_f64(row.engine_decode_eval_ms),
        fmt_opt_f64(row.engine_sample_ms),
        fmt_opt_f64(row.engine_detokenize_ms),
        fmt_opt_f64(row.engine_stream_callback_ms),
        fmt_opt_f64(row.engine_first_token_ms),
        fmt_opt_u64(row.engine_prefill_chunk),
        row.engine_prefill_path.clone().unwrap_or_default(),
        fmt_opt_f64(row.host_old_decode_ms),
        fmt_opt_f64(row.host_incremental_decode_ms),
        fmt_opt_f64(row.host_decode_speedup),
        fmt_opt_u64(row.estimated_reset_before_bytes),
        fmt_opt_u64(row.estimated_reset_after_bytes),
        fmt_opt_u64(row.estimated_reset_saved_bytes),
        fmt_opt_u64(row.estimated_prompt_output_projections_before),
        fmt_opt_u64(row.estimated_prompt_output_projections_after),
        fmt_opt_u64(row.estimated_prompt_output_projections_saved),
        fmt_opt_u64(row.estimated_prompt_output_weight_bytes_saved),
    ];

    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            writer
                .write_all(b",")
                .map_err(|err| format!("failed to write CSV row: {err}"))?;
        }
        write_csv_field(writer, value)?;
    }
    writeln!(writer).map_err(|err| format!("failed to write CSV row: {err}"))
}

fn write_csv_field(writer: &mut Box<dyn Write>, value: &str) -> Result<(), String> {
    if value
        .bytes()
        .any(|b| matches!(b, b',' | b'"' | b'\n' | b'\r'))
    {
        writer
            .write_all(b"\"")
            .map_err(|err| format!("failed to write CSV field: {err}"))?;
        for byte in value.bytes() {
            if byte == b'"' {
                writer
                    .write_all(b"\"\"")
                    .map_err(|err| format!("failed to write CSV field: {err}"))?;
            } else {
                writer
                    .write_all(&[byte])
                    .map_err(|err| format!("failed to write CSV field: {err}"))?;
            }
        }
        writer
            .write_all(b"\"")
            .map_err(|err| format!("failed to write CSV field: {err}"))?;
        Ok(())
    } else {
        writer
            .write_all(value.as_bytes())
            .map_err(|err| format!("failed to write CSV field: {err}"))
    }
}

fn fmt_f64(value: f64) -> String {
    format!("{value:.3}")
}

fn fmt_opt_f64(value: Option<f64>) -> String {
    value.map(fmt_f64).unwrap_or_default()
}

fn fmt_opt_u16(value: Option<u16>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

fn fmt_opt_u64(value: Option<u64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

fn run_http(config: HttpConfig) -> Result<(), String> {
    let mut sink = BenchSink::open(&config.output)?;
    let endpoints = config.endpoint.all();
    let total_runs = config.warmup_runs.saturating_add(config.runs);

    for idx in 0..total_runs {
        let warmup = idx < config.warmup_runs;
        let run = idx.saturating_sub(config.warmup_runs) + 1;
        for endpoint in &endpoints {
            let row = run_http_once(&config, *endpoint, run)?;
            if !warmup {
                sink.write_row(&row)?;
            }
        }
    }
    Ok(())
}

fn run_http_once(
    config: &HttpConfig,
    endpoint: HttpEndpoint,
    run: u32,
) -> Result<BenchRow, String> {
    let stream = matches!(endpoint, HttpEndpoint::Stream);
    let (method, body) = match endpoint {
        HttpEndpoint::Health | HttpEndpoint::Models => ("GET", None),
        HttpEndpoint::Chat => ("POST", Some(chat_body(&config.prompt, false)?)),
        HttpEndpoint::Stream => ("POST", Some(chat_body(&config.prompt, true)?)),
        HttpEndpoint::All => unreachable!("expanded before request"),
    };
    let measurement = execute_http(
        &config.url,
        method,
        &config.url.request_path(endpoint.path()),
        body.as_deref(),
        stream,
        config.timeout,
    );

    let mut row = BenchRow {
        mode: "http".to_string(),
        endpoint: endpoint.name().to_string(),
        run,
        ..BenchRow::default()
    };

    match measurement {
        Ok(measurement) => {
            row.ok = measurement
                .status
                .map(|s| (200..300).contains(&s))
                .unwrap_or(false);
            row.status = measurement.status;
            row.connect_ms = Some(measurement.connect_ms);
            row.write_ms = Some(measurement.write_ms);
            row.first_byte_ms = measurement.first_byte_ms;
            row.headers_ms = measurement.headers_ms;
            row.first_body_byte_ms = measurement.first_body_byte_ms;
            row.body_ms = measurement
                .headers_ms
                .map(|headers_ms| (measurement.total_ms - headers_ms).max(0.0));
            row.total_ms = measurement.total_ms;
            row.request_bytes = measurement.request_bytes as u64;
            row.response_header_bytes = measurement.header_bytes as u64;
            row.response_body_bytes = measurement.body.len() as u64;

            if stream {
                if let Some(sse) = measurement.sse {
                    row.sse_events = sse.events;
                    row.sse_content_events = sse.content_events;
                    row.sse_content_bytes = sse.content_bytes;
                    row.first_event_ms = sse.first_event_ms;
                    row.first_content_ms = sse.first_content_ms;
                    row.done_ms = sse.done_ms;
                    row.error = sse.error;
                    fill_usage(&mut row, sse.usage.as_ref());
                    fill_engine_timings(&mut row, sse.timings.as_ref());
                }
            } else if matches!(endpoint, HttpEndpoint::Chat) {
                match serde_json::from_slice::<Value>(&measurement.body) {
                    Ok(value) => {
                        fill_usage(&mut row, value.get("usage"));
                        fill_engine_timings(&mut row, value.get("qw35_timings"));
                        if let Some(message) = value
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(Value::as_str)
                        {
                            row.error = Some(message.to_string());
                        }
                    }
                    Err(err) => row.error = Some(format!("invalid JSON response: {err}")),
                }
            } else if !row.ok {
                row.error = Some(String::from_utf8_lossy(&measurement.body).into_owned());
            }
            row.client_minus_engine_ms = row
                .engine_total_ms
                .map(|engine_ms| (row.total_ms - engine_ms).max(0.0));
        }
        Err(err) => {
            row.ok = false;
            row.error = Some(err);
        }
    }

    Ok(row)
}

fn execute_http(
    url: &HttpUrl,
    method: &str,
    path: &str,
    body: Option<&str>,
    parse_sse: bool,
    timeout: Duration,
) -> Result<HttpMeasurement, String> {
    let body_bytes = body.unwrap_or("").as_bytes();
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nUser-Agent: qw35-bench\r\nAccept: {}\r\n",
        url.host_header(),
        if parse_sse { "text/event-stream" } else { "application/json" }
    );
    if body.is_some() {
        request.push_str("Content-Type: application/json\r\n");
        request.push_str(&format!("Content-Length: {}\r\n", body_bytes.len()));
    }
    request.push_str("\r\n");

    let mut request_bytes = request.into_bytes();
    request_bytes.extend_from_slice(body_bytes);

    let start = Instant::now();
    let connect_start = Instant::now();
    let mut addrs = (url.host.as_str(), url.port)
        .to_socket_addrs()
        .map_err(|err| format!("failed to resolve {}:{}: {err}", url.host, url.port))?;
    let addr = addrs
        .next()
        .ok_or_else(|| format!("no socket addresses resolved for {}:{}", url.host, url.port))?;
    let mut stream = TcpStream::connect_timeout(&addr, timeout)
        .map_err(|err| format!("failed to connect to {addr}: {err}"))?;
    let connect_ms = ms(connect_start.elapsed());
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|err| format!("failed to set read timeout: {err}"))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|err| format!("failed to set write timeout: {err}"))?;

    let write_start = Instant::now();
    stream
        .write_all(&request_bytes)
        .map_err(|err| format!("failed to write HTTP request: {err}"))?;
    let write_ms = ms(write_start.elapsed());

    let mut raw = Vec::<u8>::new();
    let mut buf = [0u8; 8192];
    let mut first_byte_ms = None;
    let mut first_body_byte_ms = None;
    let mut headers_ms = None;
    let mut header_end = None;
    let mut processed_body_abs = 0usize;
    let mut sse = if parse_sse {
        Some(SseMetrics::default())
    } else {
        None
    };

    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                if first_byte_ms.is_none() {
                    first_byte_ms = Some(ms(start.elapsed()));
                }
                raw.extend_from_slice(&buf[..n]);

                if header_end.is_none() {
                    if let Some(idx) = find_bytes(&raw, b"\r\n\r\n") {
                        let end = idx + 4;
                        header_end = Some(end);
                        headers_ms = Some(ms(start.elapsed()));
                        processed_body_abs = end;
                    }
                }

                if let Some(end) = header_end {
                    if processed_body_abs < raw.len() {
                        if first_body_byte_ms.is_none() {
                            first_body_byte_ms = Some(ms(start.elapsed()));
                        }
                        if let Some(sse) = &mut sse {
                            sse.feed(&raw[processed_body_abs..], ms(start.elapsed()));
                        }
                        processed_body_abs = raw.len();
                    } else {
                        processed_body_abs = end;
                    }
                }
            }
            Err(err)
                if matches!(
                    err.kind(),
                    ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::ConnectionReset
                ) =>
            {
                break
            }
            Err(err) => return Err(format!("failed to read HTTP response: {err}")),
        }
    }

    let total_ms = ms(start.elapsed());
    let end = header_end.ok_or("HTTP response ended before headers were complete")?;
    let headers = &raw[..end];
    let body = raw[end..].to_vec();
    let status = parse_status(headers);

    if let Some(sse) = &mut sse {
        sse.finish(total_ms);
    }

    Ok(HttpMeasurement {
        status,
        body,
        header_bytes: end,
        request_bytes: request_bytes.len(),
        connect_ms,
        write_ms,
        first_byte_ms,
        headers_ms,
        first_body_byte_ms,
        total_ms,
        sse,
    })
}

struct HttpMeasurement {
    status: Option<u16>,
    body: Vec<u8>,
    header_bytes: usize,
    request_bytes: usize,
    connect_ms: f64,
    write_ms: f64,
    first_byte_ms: Option<f64>,
    headers_ms: Option<f64>,
    first_body_byte_ms: Option<f64>,
    total_ms: f64,
    sse: Option<SseMetrics>,
}

#[derive(Default)]
struct SseMetrics {
    buffer: String,
    events: u64,
    content_events: u64,
    content_bytes: u64,
    first_event_ms: Option<f64>,
    first_content_ms: Option<f64>,
    done_ms: Option<f64>,
    usage: Option<Value>,
    timings: Option<Value>,
    error: Option<String>,
}

impl SseMetrics {
    fn feed(&mut self, bytes: &[u8], elapsed_ms: f64) {
        self.buffer.push_str(&String::from_utf8_lossy(bytes));
        while let Some((idx, len)) = find_sse_event_end(&self.buffer) {
            let event = self.buffer[..idx].to_string();
            self.buffer.drain(..idx + len);
            self.consume_event(&event, elapsed_ms);
        }
    }

    fn finish(&mut self, elapsed_ms: f64) {
        if !self.buffer.trim().is_empty() {
            let event = std::mem::take(&mut self.buffer);
            self.consume_event(&event, elapsed_ms);
        }
    }

    fn consume_event(&mut self, event: &str, elapsed_ms: f64) {
        let mut data = Vec::<&str>::new();
        for line in event.lines() {
            let line = line.trim_end_matches('\r');
            if let Some(rest) = line.strip_prefix("data:") {
                data.push(rest.trim_start());
            }
        }
        if data.is_empty() {
            return;
        }

        let data = data.join("\n");
        self.events += 1;
        if self.first_event_ms.is_none() {
            self.first_event_ms = Some(elapsed_ms);
        }
        if data == "[DONE]" {
            self.done_ms = Some(elapsed_ms);
            return;
        }

        match serde_json::from_str::<Value>(&data) {
            Ok(value) => {
                if let Some(message) = value
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(Value::as_str)
                {
                    self.error = Some(message.to_string());
                }
                if let Some(choices) = value.get("choices").and_then(Value::as_array) {
                    for choice in choices {
                        if let Some(content) = choice
                            .get("delta")
                            .and_then(|d| d.get("content"))
                            .and_then(Value::as_str)
                        {
                            if !content.is_empty() {
                                self.content_events += 1;
                                self.content_bytes += content.len() as u64;
                                if self.first_content_ms.is_none() {
                                    self.first_content_ms = Some(elapsed_ms);
                                }
                            }
                        }
                    }
                }
                if let Some(usage) = value.get("usage") {
                    self.usage = Some(usage.clone());
                }
                if let Some(timings) = value.get("qw35_timings") {
                    self.timings = Some(timings.clone());
                }
            }
            Err(err) => {
                self.error = Some(format!("invalid SSE JSON: {err}"));
            }
        }
    }
}

fn find_sse_event_end(buffer: &str) -> Option<(usize, usize)> {
    match (buffer.find("\n\n"), buffer.find("\r\n\r\n")) {
        (Some(a), Some(b)) if a <= b => Some((a, 2)),
        (Some(_), Some(b)) => Some((b, 4)),
        (Some(a), None) => Some((a, 2)),
        (None, Some(b)) => Some((b, 4)),
        (None, None) => None,
    }
}

fn parse_status(headers: &[u8]) -> Option<u16> {
    let text = String::from_utf8_lossy(headers);
    let first = text.lines().next()?;
    let mut parts = first.split_whitespace();
    let _http = parts.next()?;
    parts.next()?.parse().ok()
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn chat_body(prompt: &PromptConfig, stream: bool) -> Result<String, String> {
    let mut messages = Vec::<Value>::new();
    if let Some(system) = &prompt.system {
        if !system.is_empty() {
            messages.push(serde_json::json!({"role": "system", "content": system}));
        }
    }
    messages.push(serde_json::json!({"role": "user", "content": prompt.prompt}));

    serde_json::to_string(&serde_json::json!({
        "model": prompt.model,
        "messages": messages,
        "max_tokens": prompt.tokens,
        "temperature": prompt.temperature,
        "top_p": prompt.top_p,
        "top_k": prompt.top_k,
        "min_p": prompt.min_p,
        "presence_penalty": prompt.presence_penalty,
        "frequency_penalty": prompt.frequency_penalty,
        "repetition_penalty": prompt.repetition_penalty,
        "enable_thinking": prompt.enable_thinking,
        "preserve_thinking": prompt.preserve_thinking,
        "ignore_eos": prompt.ignore_eos,
        "stream": stream,
        "stream_options": {"include_usage": true},
    }))
    .map_err(|err| format!("failed to build chat JSON: {err}"))
}

fn fill_usage(row: &mut BenchRow, usage: Option<&Value>) {
    let Some(usage) = usage else {
        return;
    };
    row.prompt_tokens = usage.get("prompt_tokens").and_then(Value::as_u64);
    row.completion_tokens = usage.get("completion_tokens").and_then(Value::as_u64);
    row.total_tokens = usage.get("total_tokens").and_then(Value::as_u64);
}

fn fill_engine_timings(row: &mut BenchRow, timings: Option<&Value>) {
    let Some(timings) = timings else {
        return;
    };
    row.engine_total_ms = timings.get("total_ms").and_then(Value::as_f64);
    row.engine_render_ms = timings.get("render_ms").and_then(Value::as_f64);
    row.engine_tokenize_ms = timings.get("tokenize_ms").and_then(Value::as_f64);
    row.engine_runtime_lock_ms = timings.get("runtime_lock_ms").and_then(Value::as_f64);
    row.engine_reset_ms = timings.get("reset_ms").and_then(Value::as_f64);
    row.engine_prompt_eval_count = timings.get("prompt_eval_count").and_then(Value::as_u64);
    row.engine_prompt_eval_ms = timings.get("prompt_eval_ms").and_then(Value::as_f64);
    row.engine_prompt_eval_tps = timings.get("prompt_eval_tps").and_then(Value::as_f64);
    row.engine_eval_count = timings.get("eval_count").and_then(Value::as_u64);
    row.engine_eval_ms = timings.get("eval_ms").and_then(Value::as_f64);
    row.engine_eval_tps = timings.get("eval_tps").and_then(Value::as_f64);
    row.engine_decode_eval_ms = timings.get("decode_eval_ms").and_then(Value::as_f64);
    row.engine_sample_ms = timings.get("sample_ms").and_then(Value::as_f64);
    row.engine_detokenize_ms = timings.get("detokenize_ms").and_then(Value::as_f64);
    row.engine_stream_callback_ms = timings.get("stream_callback_ms").and_then(Value::as_f64);
    row.engine_first_token_ms = timings.get("first_token_ms").and_then(Value::as_f64);
    row.engine_prefill_chunk = timings.get("prefill_chunk").and_then(Value::as_u64);
    row.engine_prefill_path = timings
        .get("prefill_path")
        .and_then(Value::as_str)
        .map(str::to_string);
}

fn run_direct(config: DirectConfig) -> Result<(), String> {
    let mut sink = BenchSink::open(&config.output)?;
    let total_runs = config.warmup_runs.saturating_add(config.runs);
    let engine = if config.reopen_each_run {
        None
    } else {
        let (engine, load_ms) = open_engine(&config)?;
        eprintln!("qw35_bench: direct engine loaded in {load_ms:.3} ms");
        Some((engine, load_ms))
    };

    for idx in 0..total_runs {
        let warmup = idx < config.warmup_runs;
        let run = idx.saturating_sub(config.warmup_runs) + 1;
        let (owned_engine, load_ms);
        let engine_ref = if config.reopen_each_run {
            let opened = open_engine(&config)?;
            load_ms = opened.1;
            owned_engine = opened.0;
            &owned_engine
        } else {
            let (engine, loaded_ms) = engine.as_ref().ok_or("direct engine was not initialized")?;
            load_ms = *loaded_ms;
            engine
        };

        let request = generate_request(&config.prompt);
        let wall_start = Instant::now();
        let result = engine_ref.generate(&request);
        let wall_ms = ms(wall_start.elapsed());

        let mut row = BenchRow {
            mode: "direct".to_string(),
            endpoint: "engine.generate".to_string(),
            run,
            load_ms: Some(load_ms),
            total_ms: wall_ms,
            ..BenchRow::default()
        };

        match result {
            Ok(generation) => {
                row.ok = true;
                row.prompt_tokens = Some(generation.prompt_tokens as u64);
                row.completion_tokens = Some(generation.completion_tokens as u64);
                row.total_tokens = Some(
                    generation
                        .prompt_tokens
                        .saturating_add(generation.completion_tokens) as u64,
                );
                fill_timings_from_generation(&mut row, &generation.timings);
                row.client_minus_engine_ms = row
                    .engine_total_ms
                    .map(|engine_ms| (row.total_ms - engine_ms).max(0.0));
            }
            Err(err) => {
                row.ok = false;
                row.error = Some(format!("{err:?}"));
            }
        }

        if !warmup {
            sink.write_row(&row)?;
        }
    }
    Ok(())
}

fn open_engine(config: &DirectConfig) -> Result<(Engine, f64), String> {
    let start = Instant::now();
    let engine = Engine::open(EngineConfig {
        model_path: config.model_path.clone(),
        model_id: config.prompt.model.clone(),
        ctx_size: config.ctx_size,
        warm_weights: config.warm_weights,
        test_responder: config.test_responder,
        prefill_chunk: config.prefill_chunk,
        kv_cache_type: config.kv_cache_type,
        gf4: true,
        // Benchmarks measure full prefill cost; cross-run prefix reuse would
        // skew repeated identical prompts.
        session_cache: false,
        // The bench reports its own timing; the engine's per-request log
        // would be redundant noise.
        verbose: false,
    })?;
    Ok((engine, ms(start.elapsed())))
}

fn generate_request(prompt: &PromptConfig) -> GenerateRequest {
    let mut messages = Vec::new();
    if let Some(system) = &prompt.system {
        if !system.is_empty() {
            messages.push(qw35_server::model::ChatTurn {
                role: "system".to_string(),
                content: system.clone(),
            });
        }
    }
    messages.push(qw35_server::model::ChatTurn {
        role: "user".to_string(),
        content: prompt.prompt.clone(),
    });
    GenerateRequest {
        model: prompt.model.clone(),
        messages,
        max_tokens: TokenLimit::Fixed(prompt.tokens),
        temperature: prompt.temperature,
        top_p: prompt.top_p,
        top_k: prompt.top_k,
        min_p: prompt.min_p,
        presence_penalty: prompt.presence_penalty,
        frequency_penalty: prompt.frequency_penalty,
        repetition_penalty: prompt.repetition_penalty,
        repeat_last_n: -1,
        enable_thinking: prompt.enable_thinking,
        preserve_thinking: prompt.preserve_thinking,
        thinking_budget: None,
        reasoning_budget_message: None,
        ignore_eos: prompt.ignore_eos,
        stop_sequences: Vec::new(),
        emit_reasoning: false,
    }
}

fn fill_timings_from_generation(row: &mut BenchRow, timings: &GenerationTimings) {
    row.engine_total_ms = Some(ms(timings.total_duration));
    row.engine_render_ms = Some(ms(timings.render_duration));
    row.engine_tokenize_ms = Some(ms(timings.tokenize_duration));
    row.engine_runtime_lock_ms = Some(ms(timings.runtime_lock_duration));
    row.engine_reset_ms = Some(ms(timings.reset_duration));
    row.engine_prompt_eval_count = Some(timings.prompt_eval_count as u64);
    row.engine_prompt_eval_ms = Some(ms(timings.prompt_eval_duration));
    row.engine_prompt_eval_tps = Some(rate(
        timings.prompt_eval_count,
        timings.prompt_eval_duration,
    ));
    row.engine_eval_count = Some(timings.eval_count as u64);
    row.engine_eval_ms = Some(ms(timings.eval_duration));
    row.engine_eval_tps = Some(rate(timings.eval_count, timings.eval_duration));
    row.engine_decode_eval_ms = Some(ms(timings.decode_eval_duration));
    row.engine_sample_ms = Some(ms(timings.sample_duration));
    row.engine_detokenize_ms = Some(ms(timings.detokenize_duration));
    row.engine_stream_callback_ms = Some(ms(timings.stream_callback_duration));
    row.engine_first_token_ms = timings.first_token_duration.map(ms);
    row.engine_prefill_chunk = Some(timings.prefill_chunk as u64);
    row.engine_prefill_path = Some(timings.prefill_path.as_str().to_string());
}

fn run_host(config: HostConfig) -> Result<(), String> {
    let mut sink = BenchSink::open(&config.output)?;
    let load_start = Instant::now();
    let gguf = MappedGguf::open(&config.model_path)?;
    let tokenizer = QwenTokenizer::load(&gguf)?;
    let plan = plan_qwen35(&gguf);
    let load_ms = ms(load_start.elapsed());

    let prompt = render_host_prompt(&config.prompt);
    let prompt_tokens = tokenizer
        .encode(&prompt, true)
        .map_err(|err| format!("tokenization failed: {err}"))?;
    if prompt_tokens.is_empty() {
        return Err("host benchmark prompt produced no tokens".to_string());
    }

    let decode_tokens = repeated_tokens(&prompt_tokens, config.prompt.tokens as usize);
    let estimates = HostEstimates::from_model(&gguf, &plan, config.ctx_size, prompt_tokens.len());

    for run in 1..=config.runs {
        let mut old_bytes = 0usize;
        let old_start = Instant::now();
        for end in 1..=decode_tokens.len() {
            let decoded = tokenizer.decode(&decode_tokens[..end], false);
            old_bytes = old_bytes.wrapping_add(std::hint::black_box(decoded.len()));
        }
        let old_ms = ms(old_start.elapsed());

        let mut state = DecodeState::default();
        let mut incremental = String::new();
        let incremental_start = Instant::now();
        for &id in &decode_tokens {
            incremental.push_str(&tokenizer.decode_one(id, false, &mut state));
        }
        incremental.push_str(&tokenizer.finish_decode(&mut state));
        let incremental_ms = ms(incremental_start.elapsed());
        std::hint::black_box(old_bytes);

        let row = BenchRow {
            mode: "host".to_string(),
            endpoint: "tokenizer_and_static_costs".to_string(),
            run,
            ok: true,
            load_ms: Some(load_ms),
            total_ms: old_ms + incremental_ms,
            prompt_tokens: Some(prompt_tokens.len() as u64),
            completion_tokens: Some(decode_tokens.len() as u64),
            response_body_bytes: incremental.len() as u64,
            host_old_decode_ms: Some(old_ms),
            host_incremental_decode_ms: Some(incremental_ms),
            host_decode_speedup: if incremental_ms > 0.0 {
                Some(old_ms / incremental_ms)
            } else {
                None
            },
            estimated_reset_before_bytes: Some(estimates.reset_before_bytes),
            estimated_reset_after_bytes: Some(estimates.reset_after_bytes),
            estimated_reset_saved_bytes: Some(estimates.reset_saved_bytes),
            estimated_prompt_output_projections_before: Some(estimates.prompt_output_before),
            estimated_prompt_output_projections_after: Some(estimates.prompt_output_after),
            estimated_prompt_output_projections_saved: Some(estimates.prompt_output_saved),
            estimated_prompt_output_weight_bytes_saved: Some(
                estimates.prompt_output_weight_bytes_saved,
            ),
            ..BenchRow::default()
        };
        sink.write_row(&row)?;
    }
    Ok(())
}

struct HostEstimates {
    reset_before_bytes: u64,
    reset_after_bytes: u64,
    reset_saved_bytes: u64,
    prompt_output_before: u64,
    prompt_output_after: u64,
    prompt_output_saved: u64,
    prompt_output_weight_bytes_saved: u64,
}

impl HostEstimates {
    fn from_model(
        gguf: &MappedGguf,
        plan: &qw35_server::graph::GraphPlan,
        ctx_size: u32,
        prompt_tokens: usize,
    ) -> Self {
        let h = &plan.hparams;
        let attention_layers = plan.attention_layers.len() as u64;
        let delta_layers = plan.delta_layers.len() as u64;
        let ctx_size = u64::from(ctx_size);
        let kv_heads = u64::from(h.attention_kv_heads);
        let key_len = u64::from(h.attention_key_length);
        let value_len = u64::from(h.attention_value_length);
        let conv_channels = u64::from(h.ssm_inner_size)
            + 2 * u64::from(h.ssm_group_count) * u64::from(h.ssm_state_size);
        let conv_elems =
            delta_layers * conv_channels * u64::from(h.ssm_conv_kernel.saturating_sub(1));
        let ssm_elems = delta_layers
            * u64::from(h.ssm_time_step_rank)
            * u64::from(h.ssm_state_size)
            * u64::from(h.ssm_state_size);
        let k_elems = attention_layers * ctx_size * kv_heads * key_len;
        let v_elems = attention_layers * ctx_size * kv_heads * value_len;
        let reset_before_bytes = (k_elems + v_elems + conv_elems + ssm_elems) * 4;
        let reset_after_bytes = (conv_elems + ssm_elems) * 4;
        let prompt_output_before = prompt_tokens as u64;
        let prompt_output_after = u64::from(prompt_tokens > 0);
        let prompt_output_saved = prompt_output_before.saturating_sub(prompt_output_after);
        let output_weight_bytes = gguf.tensor("output.weight").map(|t| t.bytes).unwrap_or(0);

        Self {
            reset_before_bytes,
            reset_after_bytes,
            reset_saved_bytes: reset_before_bytes.saturating_sub(reset_after_bytes),
            prompt_output_before,
            prompt_output_after,
            prompt_output_saved,
            prompt_output_weight_bytes_saved: prompt_output_saved
                .saturating_mul(output_weight_bytes),
        }
    }
}

fn render_host_prompt(prompt: &PromptConfig) -> String {
    let mut out = String::new();
    if let Some(system) = &prompt.system {
        if !system.is_empty() {
            out.push_str("<|im_start|>system\n");
            out.push_str(system.trim());
            out.push_str("<|im_end|>\n");
        }
    }
    out.push_str("<|im_start|>user\n");
    out.push_str(prompt.prompt.trim());
    out.push_str("<|im_end|>\n<|im_start|>assistant\n");
    out
}

fn repeated_tokens(seed: &[u32], len: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let remaining = len - out.len();
        out.extend(seed.iter().copied().take(remaining));
    }
    out
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn rate(count: u32, duration: Duration) -> f64 {
    let seconds = duration.as_secs_f64();
    if count == 0 || seconds <= 0.0 {
        0.0
    } else {
        f64::from(count) / seconds
    }
}

fn parse_command<I>(args: I) -> Result<Command, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let Some(command) = args.next() else {
        return Ok(Command::Help);
    };
    match command.as_str() {
        "-h" | "--help" | "help" => Ok(Command::Help),
        "http" => parse_http(args).map(Command::Http),
        "direct" => parse_direct(args).map(Command::Direct),
        "host" => parse_host(args).map(Command::Host),
        other => Err(format!(
            "unknown command {other:?}; expected http, direct, or host"
        )),
    }
}

fn parse_http<I>(args: I) -> Result<HttpConfig, String>
where
    I: IntoIterator<Item = String>,
{
    let mut config = HttpConfig {
        url: HttpUrl::parse(DEFAULT_URL)?,
        endpoint: HttpEndpoint::All,
        runs: 3,
        warmup_runs: 1,
        timeout: Duration::from_secs(900),
        prompt: PromptConfig::default(),
        output: OutputConfig::default(),
    };

    let mut args = args.into_iter().peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Err("use `qw35_bench --help` for usage".to_string()),
            "--url" => config.url = HttpUrl::parse(&need_arg(&arg, &mut args)?)?,
            "--endpoint" => config.endpoint = parse_endpoint(&need_arg(&arg, &mut args)?)?,
            "--runs" => config.runs = parse_u32(&arg, &need_arg(&arg, &mut args)?)?,
            "--warmup-runs" => {
                config.warmup_runs = parse_u32_allow_zero(&arg, &need_arg(&arg, &mut args)?)?
            }
            "--timeout-ms" => {
                config.timeout =
                    Duration::from_millis(parse_u64(&arg, &need_arg(&arg, &mut args)?)?)
            }
            _ => {
                parse_prompt_or_output_arg(&arg, &mut args, &mut config.prompt, &mut config.output)?
            }
        }
    }
    Ok(config)
}

fn parse_direct<I>(args: I) -> Result<DirectConfig, String>
where
    I: IntoIterator<Item = String>,
{
    let mut config = DirectConfig {
        model_path: PathBuf::from(".gguf/Qwen3.5-9B-Q4_K_M.gguf"),
        ctx_size: 4096,
        runs: 3,
        warmup_runs: 1,
        warm_weights: false,
        test_responder: false,
        reopen_each_run: false,
        prefill_chunk: DEFAULT_PREFILL_CHUNK,
        kv_cache_type: qw35_server::metal::KvCacheType::Q8_0,
        prompt: PromptConfig::default(),
        output: OutputConfig::default(),
    };

    let mut args = args.into_iter().peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Err("use `qw35_bench --help` for usage".to_string()),
            "-m" | "--model" => config.model_path = PathBuf::from(need_arg(&arg, &mut args)?),
            "-c" | "--ctx" => config.ctx_size = parse_u32(&arg, &need_arg(&arg, &mut args)?)?,
            "--runs" => config.runs = parse_u32(&arg, &need_arg(&arg, &mut args)?)?,
            "--warmup-runs" => {
                config.warmup_runs = parse_u32_allow_zero(&arg, &need_arg(&arg, &mut args)?)?
            }
            "--warm-weights" => config.warm_weights = true,
            "--test-responder" => config.test_responder = true,
            "--reopen-each-run" => config.reopen_each_run = true,
            "--prefill-chunk" => {
                config.prefill_chunk = parse_u32(&arg, &need_arg(&arg, &mut args)?)?
            }
            "--kv-cache-type" => {
                let value = need_arg(&arg, &mut args)?;
                config.kv_cache_type = qw35_server::metal::KvCacheType::parse(&value)
                    .ok_or_else(|| format!("invalid --kv-cache-type {value:?}: expected f16 or q8_0"))?;
            }
            _ => {
                parse_prompt_or_output_arg(&arg, &mut args, &mut config.prompt, &mut config.output)?
            }
        }
    }
    validate_prefill_chunk(config.prefill_chunk)?;
    Ok(config)
}

fn parse_host<I>(args: I) -> Result<HostConfig, String>
where
    I: IntoIterator<Item = String>,
{
    let mut config = HostConfig {
        model_path: PathBuf::from(".gguf/Qwen3.5-9B-Q4_K_M.gguf"),
        ctx_size: 4096,
        runs: 3,
        prompt: PromptConfig {
            tokens: 1812,
            ..PromptConfig::default()
        },
        output: OutputConfig::default(),
    };

    let mut args = args.into_iter().peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Err("use `qw35_bench --help` for usage".to_string()),
            "-m" | "--model" => config.model_path = PathBuf::from(need_arg(&arg, &mut args)?),
            "-c" | "--ctx" => config.ctx_size = parse_u32(&arg, &need_arg(&arg, &mut args)?)?,
            "--runs" => config.runs = parse_u32(&arg, &need_arg(&arg, &mut args)?)?,
            _ => {
                parse_prompt_or_output_arg(&arg, &mut args, &mut config.prompt, &mut config.output)?
            }
        }
    }
    Ok(config)
}

fn parse_prompt_or_output_arg<I>(
    arg: &str,
    args: &mut std::iter::Peekable<I>,
    prompt: &mut PromptConfig,
    output: &mut OutputConfig,
) -> Result<(), String>
where
    I: Iterator<Item = String>,
{
    match arg {
        "--model-id" => prompt.model = need_arg(arg, args)?,
        "--system" => prompt.system = Some(need_arg(arg, args)?),
        "--no-system" => prompt.system = None,
        "--prompt" => prompt.prompt = need_arg(arg, args)?,
        "--prompt-file" => {
            let path = PathBuf::from(need_arg(arg, args)?);
            prompt.prompt = std::fs::read_to_string(&path)
                .map_err(|err| format!("failed to read prompt file {}: {err}", path.display()))?;
        }
        "-n" | "--tokens" | "--max-tokens" => {
            prompt.tokens = parse_u32(arg, &need_arg(arg, args)?)?
        }
        "--temperature" => prompt.temperature = parse_f32(arg, &need_arg(arg, args)?)?,
        "--top-p" => prompt.top_p = parse_f32(arg, &need_arg(arg, args)?)?,
        "--top-k" => prompt.top_k = parse_u32(arg, &need_arg(arg, args)?)?,
        "--min-p" => prompt.min_p = parse_f32(arg, &need_arg(arg, args)?)?,
        "--presence-penalty" => prompt.presence_penalty = parse_f32(arg, &need_arg(arg, args)?)?,
        "--frequency-penalty" => prompt.frequency_penalty = parse_f32(arg, &need_arg(arg, args)?)?,
        "--repetition-penalty" => {
            prompt.repetition_penalty = parse_f32(arg, &need_arg(arg, args)?)?
        }
        "--thinking" => prompt.enable_thinking = true,
        "--no-thinking" => prompt.enable_thinking = false,
        "--preserve-thinking" => prompt.preserve_thinking = true,
        "--ignore-eos" => prompt.ignore_eos = true,
        "--format" => output.format = parse_output_format(&need_arg(arg, args)?)?,
        "--csv" => {
            output.format = OutputFormat::Csv;
            output.path = Some(PathBuf::from(need_arg(arg, args)?));
        }
        "--jsonl" => {
            output.format = OutputFormat::Jsonl;
            output.path = Some(PathBuf::from(need_arg(arg, args)?));
        }
        "--output" | "-o" => output.path = Some(PathBuf::from(need_arg(arg, args)?)),
        other => return Err(format!("unknown option {other}")),
    }
    Ok(())
}

fn need_arg<I>(flag: &str, args: &mut std::iter::Peekable<I>) -> Result<String, String>
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
        "--url" => "a server URL (e.g. http://127.0.0.1:8080)",
        "--endpoint" => "one of: health, models, chat, stream, all",
        "--runs" => "a run count (e.g. 5)",
        "--warmup-runs" => "a warmup run count (e.g. 1)",
        "--timeout-ms" => "a timeout in milliseconds (e.g. 30000)",
        "-m" | "--model" => "a GGUF model file path",
        "-c" | "--ctx" => "a context size in tokens (e.g. 120000)",
        "--prefill-chunk" => "a chunk size in tokens (e.g. 32)",
        "--kv-cache-type" => "one of: f16, q8_0",
        "--model-id" => "the served OpenAI model id (e.g. qwen35-9b)",
        "--system" => "a system prompt string",
        "--prompt" => "a prompt string",
        "--prompt-file" => "a path to a prompt file",
        "-n" | "--tokens" | "--max-tokens" => "a max output token count (e.g. 8192)",
        "--temperature" => "a temperature (e.g. 0.35; 0 for greedy decoding)",
        "--top-p" => "a nucleus probability (e.g. 0.95)",
        "--top-k" => "a top-k limit (e.g. 20; 0 to disable)",
        "--min-p" => "a minimum probability ratio (e.g. 0)",
        "--presence-penalty" => "a presence penalty (e.g. 0.3)",
        "--frequency-penalty" => "a frequency penalty (e.g. 0)",
        "--repetition-penalty" => "a repetition penalty (e.g. 1.1)",
        "--format" => "one of: csv, jsonl",
        "--csv" | "--jsonl" | "--output" | "-o" => "an output file path",
        _ => "a value",
    }
}

fn parse_endpoint(value: &str) -> Result<HttpEndpoint, String> {
    match value {
        "health" => Ok(HttpEndpoint::Health),
        "models" => Ok(HttpEndpoint::Models),
        "chat" => Ok(HttpEndpoint::Chat),
        "stream" => Ok(HttpEndpoint::Stream),
        "all" => Ok(HttpEndpoint::All),
        _ => Err(format!(
            "invalid --endpoint {value:?}; expected health, models, chat, stream, or all"
        )),
    }
}

fn parse_output_format(value: &str) -> Result<OutputFormat, String> {
    match value {
        "csv" => Ok(OutputFormat::Csv),
        "jsonl" => Ok(OutputFormat::Jsonl),
        _ => Err(format!("invalid --format {value:?}; expected csv or jsonl")),
    }
}

fn parse_u32(flag: &str, value: &str) -> Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|err| format!("invalid {flag} value {value:?}: {err}"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be greater than zero"));
    }
    Ok(parsed)
}

fn parse_u32_allow_zero(flag: &str, value: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|err| format!("invalid {flag} value {value:?}: {err}"))
}

fn parse_u64(flag: &str, value: &str) -> Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|err| format!("invalid {flag} value {value:?}: {err}"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be greater than zero"));
    }
    Ok(parsed)
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
        "Usage:\n\
  qw35_bench http [options]\n\
  qw35_bench direct [options]\n\
  qw35_bench host [options]\n\
\n\
HTTP mode measures the socket and OpenAI-compatible HTTP path of a running qw35 server.\n\
Direct mode runs Engine::generate in-process with the same prompt so HTTP overhead can be separated from inference.\n\
Host mode measures tokenizer decode overhead and reports static reset/projection cost estimates without Metal.\n\
\n\
Common prompt options:\n\
  --model-id ID              Default: qwen35-9b\n\
  --prompt TEXT              User prompt. A benchmark prompt is used by default.\n\
  --prompt-file FILE         Read the user prompt from FILE.\n\
  --system TEXT | --no-system\n\
  -n, --tokens N             Max decode tokens. Default: 128\n\
  --temperature F            Default: 0 for greedy timing\n\
  --top-p F --top-k N --min-p F\n\
  --presence-penalty F --repetition-penalty F\n\
  --thinking | --no-thinking Accepted for compatibility; prompt rendering does not inject thinking tags.\n\
  --ignore-eos               Keep decoding until --tokens for fixed-length throughput rows.\n\
  --format csv|jsonl         Default: csv\n\
  --csv FILE | --jsonl FILE | -o FILE\n\
\n\
HTTP options:\n\
  --url URL                  Default: http://127.0.0.1:8080\n\
  --endpoint NAME            health, models, chat, stream, or all. Default: all\n\
  --runs N                   Timed runs per endpoint. Default: 3\n\
  --warmup-runs N            Untimed warmup runs per endpoint. Default: 1\n\
  --timeout-ms N             Socket timeout. Default: 900000\n\
\n\
Direct options:\n\
  -m, --model FILE           Default: .gguf/Qwen3.5-9B-Q4_K_M.gguf\n\
  -c, --ctx N                Default: 4096\n\
  --runs N                   Timed generation runs. Default: 3\n\
  --warmup-runs N            Untimed generation runs after load. Default: 1\n\
  --warm-weights             Touch mapped tensor pages before benchmarking.\n\
  --prefill-chunk N          Native Metal prompt chunk size. Use 1 for scalar compatibility; any larger value uses the Q4_K/Q5_K/Q6_K tiled prefill path, fastest at multiples of 32. Default: 32\n\
  --kv-cache-type TYPE       KV cache storage for attention layers: q8_0 or f16. Default: q8_0\n\
  --reopen-each-run          Include a fresh Engine::open in every row.\n\
  --test-responder           Benchmark metadata/tokenizer/test path without native inference.\n\
\n\
Host options:\n\
  -m, --model FILE           Default: .gguf/Qwen3.5-9B-Q4_K_M.gguf\n\
  -c, --ctx N                Context used for reset-byte estimates. Default: 4096\n\
  --runs N                   Timed host runs. Default: 3\n\
  -n, --tokens N             Token count for decode-prefix benchmark. Default: 1812\n\
\n\
Examples:\n\
  cargo run --release --bin qw35_bench -- http --endpoint all --runs 5 --tokens 256 --ignore-eos\n\
  cargo run --release --bin qw35_bench -- direct --runs 3 --tokens 256 --ignore-eos\n\
  cargo run --release --bin qw35_bench -- host --runs 3 --tokens 1812\n"
    );
}
