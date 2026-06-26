use qw35_server::model::{
    Engine, EngineConfig, TokenLimit, DEFAULT_MODEL_ID, DEFAULT_PREFILL_CHUNK,
};
use qw35_server::server::{self, GenerationDefaults};
use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A test-responder server on an ephemeral port; shuts down and removes the
/// tiny GGUF on drop.
struct TestServer {
    addr: std::net::SocketAddr,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    handle: Option<thread::JoinHandle<()>>,
    model_path: PathBuf,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            tx.send(()).ok();
        }
        if let Some(handle) = self.handle.take() {
            handle.join().ok();
        }
        fs::remove_file(&self.model_path).ok();
    }
}

fn start_test_server() -> TestServer {
    let model_path = write_tiny_qwen35_gguf();
    let engine = Arc::new(
        Engine::open(EngineConfig {
            model_path: model_path.clone(),
            model_id: DEFAULT_MODEL_ID.to_string(),
            ctx_size: 4096,
            prefill_chunk: DEFAULT_PREFILL_CHUNK,
            kv_cache_type: qw35_server::metal::KvCacheType::Q8_0,
            session_cache: true,
            gf4: true,
            attn_window: 0,
            attn_sink: 0,
            warm_weights: true,
            test_responder: true,
            verbose: false,
        })
        .unwrap(),
    );

    let std_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    std_listener.set_nonblocking(true).unwrap();
    let addr = std_listener.local_addr().unwrap();

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let handle = thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let tokio_listener = tokio::net::TcpListener::from_std(std_listener).unwrap();
            server::serve_listener(
                tokio_listener,
                engine,
                GenerationDefaults {
                    max_tokens: TokenLimit::Fixed(128),
                    ..GenerationDefaults::default()
                },
                Some(shutdown_rx),
            )
            .await
            .unwrap();
        });
    });
    thread::sleep(Duration::from_millis(100));

    TestServer {
        addr,
        shutdown_tx: Some(shutdown_tx),
        handle: Some(handle),
        model_path,
    }
}

fn post_json(addr: std::net::SocketAddr, path: &str, body: &serde_json::Value) -> String {
    let body = body.to_string();
    http_request(
        addr,
        &format!(
            "POST {path} HTTP/1.1\r\nHost: local\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        ),
    )
}

/// Extracts the HTTP body, de-chunking when the response uses chunked
/// transfer encoding (SSE responses do).
fn http_body(response: &str) -> String {
    let Some(split) = response.find("\r\n\r\n") else {
        return response.to_string();
    };
    let (headers, rest) = response.split_at(split + 4);
    if !headers
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        return rest.to_string();
    }
    let mut out = String::new();
    let mut remaining = rest;
    loop {
        let Some(line_end) = remaining.find("\r\n") else {
            break;
        };
        let Ok(size) = usize::from_str_radix(remaining[..line_end].trim(), 16) else {
            break;
        };
        if size == 0 {
            break;
        }
        let start = line_end + 2;
        if remaining.len() < start + size {
            out.push_str(&remaining[start..]);
            break;
        }
        out.push_str(&remaining[start..start + size]);
        remaining = remaining[start + size..].trim_start_matches("\r\n");
    }
    out
}

fn response_body_json(response: &str) -> serde_json::Value {
    serde_json::from_str(&http_body(response))
        .unwrap_or_else(|err| panic!("body is not JSON ({err}): {response}"))
}

/// Parses every `data: {...}` SSE line into JSON, skipping `[DONE]`.
fn sse_data_chunks(response: &str) -> Vec<serde_json::Value> {
    http_body(response)
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| data.trim() != "[DONE]")
        .map(|data| {
            serde_json::from_str(data)
                .unwrap_or_else(|err| panic!("SSE data is not JSON ({err}): {data}"))
        })
        .collect()
}

#[test]
fn human_chat_completion_works_over_http() {
    let server = start_test_server();
    let addr = server.addr;

    let models = http_request(
        addr,
        "GET /v1/models HTTP/1.1\r\nHost: local\r\nConnection: close\r\n\r\n",
    );
    assert!(models.contains("HTTP/1.1 200 OK"), "{models}");
    assert!(models.contains(DEFAULT_MODEL_ID));
    assert!(models.contains("Qwen3.5-9B"));

    let model = http_request(
        addr,
        "GET /v1/models/qwen3.5-9b HTTP/1.1\r\nHost: local\r\nConnection: close\r\n\r\n",
    );
    assert!(model.contains("HTTP/1.1 200 OK"), "{model}");
    assert!(model.contains("\"object\":\"model\""), "{model}");

    let body = r#"{"model":"qwen3.5-9b","messages":[{"role":"system","content":"Answer tersely."},{"role":"user","content":"Say exactly: local HTTP chat works"}],"max_tokens":32,"stream":false}"#;
    let response = http_request(
        addr,
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        ),
    );
    assert!(response.contains("HTTP/1.1 200 OK"), "{response}");
    assert!(response.contains("local HTTP chat works"), "{response}");
    assert!(
        response.contains("\"finish_reason\":\"stop\""),
        "{response}"
    );
    assert!(response.contains("\"qw35_timings\""), "{response}");
    assert!(response.contains("\"prompt_eval_tps\""), "{response}");

    for expected in ["ping one", "ping two", "ping three"] {
        let ping_pong_messages =
            format!(r#"[{{"role":"user","content":"Say exactly: {expected}"}}]"#);
        let ping_body = format!(
            r#"{{"model":"qwen3.5-9b","messages":{ping_pong_messages},"max_tokens":16,"temperature":0,"stream":false}}"#
        );
        let ping_response = http_request(
            addr,
            &format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                ping_body.len(),
                ping_body
            ),
        );
        assert!(ping_response.contains("HTTP/1.1 200 OK"), "{ping_response}");
        assert!(ping_response.contains(expected), "{ping_response}");
    }

    let stream_body = r#"{"model":"qwen3.5-9b","messages":[{"role":"user","content":"Write one short sentence proving the streaming endpoint is alive."}],"max_tokens":64,"stream":true,"stream_options":{"include_usage":true}}"#;
    let stream_response = http_request(
        addr,
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            stream_body.len(),
            stream_body
        ),
    );
    assert!(
        stream_response
            .to_ascii_lowercase()
            .contains("content-type: text/event-stream"),
        "{stream_response}"
    );
    assert!(stream_response.contains("data: {"));
    assert!(stream_response.contains("Qw35 HTTP test responder"));
    assert!(stream_response.contains("\"usage\""));
    assert!(stream_response.contains("\"qw35_timings\""));
    assert!(stream_response.contains("data: [DONE]"));

    let responses_body = r#"{"model":"qwen3.5-9b","instructions":"Answer tersely.","input":"Say exactly: responses API works","max_output_tokens":32,"temperature":0,"store":false}"#;
    let responses_response = http_request(
        addr,
        &format!(
            "POST /v1/responses HTTP/1.1\r\nHost: local\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            responses_body.len(),
            responses_body
        ),
    );
    assert!(
        responses_response.contains("HTTP/1.1 200 OK"),
        "{responses_response}"
    );
    assert!(
        responses_response.contains("\"object\":\"response\""),
        "{responses_response}"
    );
    assert!(
        responses_response.contains("\"type\":\"output_text\""),
        "{responses_response}"
    );
    assert!(
        responses_response.contains("responses API works"),
        "{responses_response}"
    );
    assert!(
        responses_response.contains("\"input_tokens\""),
        "{responses_response}"
    );

    let input_tokens_body =
        r#"{"model":"qwen3.5-9b","input":"Count this prompt through the Responses token API."}"#;
    let input_tokens = http_request(
        addr,
        &format!(
            "POST /v1/responses/input_tokens HTTP/1.1\r\nHost: local\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            input_tokens_body.len(),
            input_tokens_body
        ),
    );
    assert!(input_tokens.contains("HTTP/1.1 200 OK"), "{input_tokens}");
    assert!(
        input_tokens.contains("\"object\":\"response.input_tokens\""),
        "{input_tokens}"
    );
    assert!(input_tokens.contains("\"input_tokens\":"), "{input_tokens}");

    let responses_stream_body = r#"{"model":"qwen3.5-9b","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Write one short sentence proving responses streaming is alive."}]}],"max_output_tokens":64,"stream":true}"#;
    let responses_stream = http_request(
        addr,
        &format!(
            "POST /v1/responses HTTP/1.1\r\nHost: local\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            responses_stream_body.len(),
            responses_stream_body
        ),
    );
    assert!(
        responses_stream
            .to_ascii_lowercase()
            .contains("content-type: text/event-stream"),
        "{responses_stream}"
    );
    assert!(
        responses_stream.contains("event: response.created"),
        "{responses_stream}"
    );
    assert!(
        responses_stream.contains("event: response.output_text.delta"),
        "{responses_stream}"
    );
    assert!(
        responses_stream.contains("event: response.completed"),
        "{responses_stream}"
    );

    let bad_responses_body = r#"{"model":"qwen3.5-9b","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"file://nope"}]}]}"#;
    let bad_responses = http_request(
        addr,
        &format!(
            "POST /v1/responses HTTP/1.1\r\nHost: local\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            bad_responses_body.len(),
            bad_responses_body
        ),
    );
    assert!(
        bad_responses.contains("HTTP/1.1 400 Bad Request"),
        "{bad_responses}"
    );
    assert!(bad_responses.contains("text-only"), "{bad_responses}");

    let stateful_responses_body =
        r#"{"model":"qwen3.5-9b","previous_response_id":"resp_old","input":"hello"}"#;
    let stateful_responses = http_request(
        addr,
        &format!(
            "POST /v1/responses HTTP/1.1\r\nHost: local\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            stateful_responses_body.len(),
            stateful_responses_body
        ),
    );
    assert!(
        stateful_responses.contains("HTTP/1.1 400 Bad Request"),
        "{stateful_responses}"
    );
    assert!(
        stateful_responses.contains("previous_response_id is not supported"),
        "{stateful_responses}"
    );
}

const CALL_TEXT: &str = "<tool_call>\n<get_weather city=\"Paris\"/>\n</tool_call>";

fn weather_tools() -> serde_json::Value {
    serde_json::json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }])
}

#[test]
fn chat_completions_tool_calling_works_over_http() {
    let server = start_test_server();
    let addr = server.addr;

    // Non-streaming: the echoed <tool_call> block must come back as an
    // OpenAI-shape tool_calls entry, with arguments as a JSON *string*.
    let body = serde_json::json!({
        "model": "qwen3.5-9b",
        "messages": [
            {"role": "system", "content": "Use tools when asked."},
            {"role": "user", "content": format!("Say exactly: {CALL_TEXT}")}
        ],
        "tools": weather_tools(),
        "tool_choice": "auto",
        "max_tokens": 64,
        "stream": false
    });
    let response = post_json(addr, "/v1/chat/completions", &body);
    assert!(response.contains("HTTP/1.1 200 OK"), "{response}");
    let json = response_body_json(&response);
    let choice = &json["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls", "{json}");
    let message = &choice["message"];
    assert_eq!(message["content"], serde_json::Value::Null, "{json}");
    let call = &message["tool_calls"][0];
    assert_eq!(call["type"], "function", "{json}");
    assert!(call["id"].as_str().unwrap().starts_with("call_"), "{json}");
    assert_eq!(call["function"]["name"], "get_weather", "{json}");
    let arguments = call["function"]["arguments"]
        .as_str()
        .expect("arguments must be a JSON string, not an object");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(arguments).unwrap(),
        serde_json::json!({"city": "Paris"})
    );

    // Streaming: first tool delta carries index+id+name with empty arguments,
    // later deltas carry argument fragments only with the same index and no
    // new ids; no content delta leaks tag text.
    let stream_body = serde_json::json!({
        "model": "qwen3.5-9b",
        "messages": [
            {"role": "user", "content": format!("Say exactly: {CALL_TEXT}")}
        ],
        "tools": weather_tools(),
        "max_tokens": 64,
        "stream": true,
        "stream_options": {"include_usage": true}
    });
    let stream_response = post_json(addr, "/v1/chat/completions", &stream_body);
    let chunks = sse_data_chunks(&stream_response);
    let mut tool_deltas = Vec::new();
    let mut finish_reasons = Vec::new();
    let mut saw_usage = false;
    for chunk in &chunks {
        if chunk["usage"].is_object() {
            saw_usage = true;
        }
        let choice = &chunk["choices"][0];
        if let Some(reason) = choice["finish_reason"].as_str() {
            finish_reasons.push(reason.to_string());
        }
        if let Some(content) = choice["delta"]["content"].as_str() {
            assert!(
                !content.contains("<tool_call"),
                "tag text leaked into content: {content:?}"
            );
        }
        if choice["delta"]["tool_calls"][0].is_object() {
            tool_deltas.push(choice["delta"]["tool_calls"][0].clone());
        }
    }
    assert!(!tool_deltas.is_empty(), "{stream_response}");
    let first = &tool_deltas[0];
    assert_eq!(first["index"], 0, "{stream_response}");
    assert!(first["id"].as_str().unwrap().starts_with("call_"));
    assert_eq!(first["type"], "function");
    assert_eq!(first["function"]["name"], "get_weather");
    assert_eq!(first["function"]["arguments"], "");
    let mut streamed_arguments = String::new();
    for delta in &tool_deltas[1..] {
        assert_eq!(delta["index"], 0, "{stream_response}");
        assert!(delta.get("id").is_none(), "later deltas must not mint ids");
        streamed_arguments.push_str(delta["function"]["arguments"].as_str().unwrap());
    }
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&streamed_arguments).unwrap(),
        serde_json::json!({"city": "Paris"})
    );
    assert_eq!(finish_reasons, vec!["tool_calls".to_string()]);
    assert!(saw_usage, "{stream_response}");

    // Agent replay: assistant turn with content:null + tool_calls, then a
    // tool-role result. This is what opencode/pi send on the next turn.
    let replay_body = serde_json::json!({
        "model": "qwen3.5-9b",
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 22C"},
            {"role": "user", "content": "Say exactly: thanks for the weather"}
        ],
        "tools": weather_tools(),
        "max_tokens": 32,
        "stream": false
    });
    let replay = post_json(addr, "/v1/chat/completions", &replay_body);
    assert!(replay.contains("HTTP/1.1 200 OK"), "{replay}");
    assert!(replay.contains("thanks for the weather"), "{replay}");

    // n > 1 is an explicit 400, not a silent single choice.
    let n_body = serde_json::json!({
        "model": "qwen3.5-9b",
        "messages": [{"role": "user", "content": "hi"}],
        "n": 2
    });
    let n_response = post_json(addr, "/v1/chat/completions", &n_body);
    assert!(
        n_response.contains("HTTP/1.1 400 Bad Request"),
        "{n_response}"
    );
    assert!(n_response.contains("n must be 1"), "{n_response}");

    // Stop sequences truncate before the match and finish with "stop".
    let stop_body = serde_json::json!({
        "model": "qwen3.5-9b",
        "messages": [{"role": "user", "content": "Say exactly: hello STOP world"}],
        "stop": ["STOP"],
        "max_tokens": 32,
        "stream": false
    });
    let stop_response = post_json(addr, "/v1/chat/completions", &stop_body);
    let stop_json = response_body_json(&stop_response);
    let content = stop_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap();
    assert!(content.contains("hello"), "{stop_json}");
    assert!(!content.contains("world"), "{stop_json}");
    assert_eq!(stop_json["choices"][0]["finish_reason"], "stop");

    // Token-limit truncation reports finish_reason "length".
    let length_body = serde_json::json!({
        "model": "qwen3.5-9b",
        "messages": [{"role": "user", "content": "Say exactly: aaaa bbbb cccc dddd eeee ffff"}],
        "max_tokens": 2,
        "stream": false
    });
    let length_response = post_json(addr, "/v1/chat/completions", &length_body);
    let length_json = response_body_json(&length_response);
    assert_eq!(
        length_json["choices"][0]["finish_reason"], "length",
        "{length_json}"
    );
}

#[test]
fn responses_tool_calling_works_over_http() {
    let server = start_test_server();
    let addr = server.addr;

    // A codex-shaped stateless payload: instructions, flat tools, message +
    // function_call/function_call_output replay items.
    let body = serde_json::json!({
        "model": "qwen3.5-9b",
        "instructions": "You are a coding agent.",
        "input": [
            {"type": "function_call", "call_id": "call_0", "name": "list_files",
             "arguments": "{\"path\": \"/var\"}"},
            {"type": "function_call_output", "call_id": "call_0", "output": "cache log tmp"},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text",
                 "text": "Say exactly: <tool_call>\n<list_files path=\"/tmp\"/>\n</tool_call>"}
            ]}
        ],
        "tools": [{
            "type": "function",
            "name": "list_files",
            "description": "List files in a directory",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
        }],
        "tool_choice": "auto",
        "parallel_tool_calls": true,
        "store": false,
        "max_output_tokens": 64,
        "stream": false
    });
    let response = post_json(addr, "/v1/responses", &body);
    assert!(response.contains("HTTP/1.1 200 OK"), "{response}");
    let json = response_body_json(&response);
    assert_eq!(json["status"], "completed", "{json}");
    let output = json["output"].as_array().unwrap();
    let call = output
        .iter()
        .find(|item| item["type"] == "function_call")
        .unwrap_or_else(|| panic!("no function_call item: {json}"));
    assert_eq!(call["status"], "completed", "{json}");
    assert_eq!(call["name"], "list_files", "{json}");
    assert!(call["call_id"].as_str().unwrap().starts_with("call_"));
    assert!(call["id"].as_str().unwrap().starts_with("fc_"));
    let arguments = call["arguments"]
        .as_str()
        .expect("arguments must be a JSON string, not an object");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(arguments).unwrap(),
        serde_json::json!({"path": "/tmp"})
    );

    // Streaming: function_call item lifecycle events with sequence numbers.
    let mut stream_body = body.clone();
    stream_body["stream"] = serde_json::json!(true);
    let stream_response = post_json(addr, "/v1/responses", &stream_body);
    let stream_text = http_body(&stream_response);
    for expected in [
        "event: response.created",
        "event: response.in_progress",
        "event: response.output_item.added",
        "event: response.function_call_arguments.delta",
        "event: response.function_call_arguments.done",
        "event: response.output_item.done",
        "event: response.completed",
    ] {
        assert!(
            stream_text.contains(expected),
            "missing {expected}: {stream_text}"
        );
    }
    let chunks = sse_data_chunks(&stream_response);
    let mut last_seq = None;
    for chunk in &chunks {
        let seq = chunk["sequence_number"]
            .as_u64()
            .unwrap_or_else(|| panic!("event without sequence_number: {chunk}"));
        if let Some(last) = last_seq {
            assert_eq!(seq, last + 1, "sequence numbers must be monotonic");
        }
        last_seq = Some(seq);
    }
    let added_call = chunks
        .iter()
        .find(|chunk| {
            chunk["type"] == "response.output_item.added"
                && chunk["item"]["type"] == "function_call"
        })
        .unwrap_or_else(|| panic!("no function_call output_item.added: {stream_text}"));
    assert_eq!(added_call["item"]["name"], "list_files");
    assert_eq!(added_call["item"]["arguments"], "");
    let args_done = chunks
        .iter()
        .find(|chunk| chunk["type"] == "response.function_call_arguments.done")
        .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(args_done["arguments"].as_str().unwrap())
            .unwrap(),
        serde_json::json!({"path": "/tmp"})
    );
    let completed = chunks
        .iter()
        .find(|chunk| chunk["type"] == "response.completed")
        .unwrap();
    assert!(completed["response"]["output"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["type"] == "function_call"));

    // Token-limit truncation: incomplete status with max_output_tokens reason.
    let truncated_body = serde_json::json!({
        "model": "qwen3.5-9b",
        "input": "Say exactly: aaaa bbbb cccc dddd eeee ffff",
        "max_output_tokens": 2,
        "stream": false
    });
    let truncated = post_json(addr, "/v1/responses", &truncated_body);
    let truncated_json = response_body_json(&truncated);
    assert_eq!(truncated_json["status"], "incomplete", "{truncated_json}");
    assert_eq!(
        truncated_json["incomplete_details"]["reason"], "max_output_tokens",
        "{truncated_json}"
    );
}

fn http_request(addr: std::net::SocketAddr, request: &str) -> String {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => response.extend_from_slice(&buf[..n]),
            Err(err)
                if matches!(
                    err.kind(),
                    ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::ConnectionReset
                ) =>
            {
                break
            }
            Err(err) => panic!("HTTP read failed: {err}"),
        }
    }
    String::from_utf8_lossy(&response).into_owned()
}

fn write_tiny_qwen35_gguf() -> PathBuf {
    const GGUF_MAGIC: u32 = 0x4655_4747;
    let mut bytes = Vec::new();
    put_u32(&mut bytes, GGUF_MAGIC);
    put_u32(&mut bytes, 3);
    put_u64(&mut bytes, 427);
    put_u64(&mut bytes, 21);

    put_kv_string(&mut bytes, "general.architecture", "qwen35");
    put_kv_string(&mut bytes, "general.name", "Qwen3.5-9B");
    put_kv_string(&mut bytes, "general.size_label", "9B");
    put_kv_u32(&mut bytes, "general.alignment", 32);
    put_kv_u32(&mut bytes, "qwen35.block_count", 32);
    put_kv_u32(&mut bytes, "qwen35.context_length", 262_144);
    put_kv_u32(&mut bytes, "qwen35.embedding_length", 4096);
    put_kv_u32(&mut bytes, "qwen35.feed_forward_length", 12_288);
    put_kv_u32(&mut bytes, "qwen35.attention.head_count", 16);
    put_kv_u32(&mut bytes, "qwen35.attention.head_count_kv", 4);
    put_kv_u32(&mut bytes, "qwen35.attention.key_length", 256);
    put_kv_u32(&mut bytes, "qwen35.attention.value_length", 256);
    put_kv_u32(&mut bytes, "qwen35.rope.dimension_count", 64);
    put_kv_u32(&mut bytes, "qwen35.ssm.conv_kernel", 4);
    put_kv_u32(&mut bytes, "qwen35.ssm.state_size", 128);
    put_kv_u32(&mut bytes, "qwen35.ssm.group_count", 16);
    put_kv_u32(&mut bytes, "qwen35.ssm.time_step_rank", 32);

    put_kv_string(&mut bytes, "tokenizer.ggml.model", "gpt2");
    put_kv_string(&mut bytes, "tokenizer.ggml.pre", "qwen35");
    put_tokenizer_tokens(&mut bytes);
    put_string_array(&mut bytes, "tokenizer.ggml.merges", &[]);

    put_string(&mut bytes, "token_embd.weight");
    put_u32(&mut bytes, 2);
    put_u64(&mut bytes, 4096);
    put_u64(&mut bytes, 1);
    put_u32(&mut bytes, 12);
    put_u64(&mut bytes, 0);
    for idx in 0..426 {
        put_string(&mut bytes, &format!("dummy.{idx}"));
        put_u32(&mut bytes, 1);
        put_u64(&mut bytes, 1);
        put_u32(&mut bytes, 0);
        put_u64(&mut bytes, 0);
    }

    while bytes.len() % 32 != 0 {
        bytes.push(0);
    }
    bytes.resize(bytes.len() + 2304, 0);

    let path = std::env::temp_dir().join(format!(
        "qw35-human-http-{}-{}.gguf",
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

fn put_tokenizer_tokens(bytes: &mut Vec<u8>) {
    const VALUE_STRING: u32 = 8;
    const VALUE_ARRAY: u32 = 9;
    put_string(bytes, "tokenizer.ggml.tokens");
    put_u32(bytes, VALUE_ARRAY);
    put_u32(bytes, VALUE_STRING);
    put_u64(bytes, 248_320);

    for idx in 0..248_320u32 {
        match idx {
            0 => put_string(bytes, "!"),
            1 => put_string(bytes, "<|im_start|>"),
            2 => put_string(bytes, "<|im_end|>"),
            3 => put_string(bytes, "<think>"),
            4 => put_string(bytes, "</think>"),
            5 => put_string(bytes, "<tool_call>"),
            6 => put_string(bytes, "</tool_call>"),
            _ => put_string(bytes, &format!("tok{idx}")),
        }
    }
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
