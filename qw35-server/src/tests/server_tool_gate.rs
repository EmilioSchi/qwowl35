    use super::*;

    // Tool-call parsing must be on exactly when the request advertised tools
    // (the `# Tools` block was injected) or a named tool_choice forced a call
    // opening; otherwise `<tool_call>` XML passes through as content.

    const TOOLS: &str = r#"[{"type": "function", "function": {"name": "get_weather"}}]"#;

    fn chat_parse_tool_calls(json: serde_json::Value) -> bool {
        let req: ChatRequest = serde_json::from_value(json).expect("valid ChatRequest");
        into_generate_request(&req, "m", &GenerationDefaults::default())
            .expect("request builds")
            .parse_tool_calls
    }

    fn responses_parse_tool_calls(json: serde_json::Value) -> bool {
        let req: ResponsesRequest = serde_json::from_value(json).expect("valid ResponsesRequest");
        into_responses_generate_request(&req, "m", &GenerationDefaults::default())
            .expect("request builds")
            .parse_tool_calls
    }

    fn tools() -> serde_json::Value {
        serde_json::from_str(TOOLS).unwrap()
    }

    #[test]
    fn chat_without_tools_disables_parsing() {
        assert!(!chat_parse_tool_calls(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
        })));
    }

    #[test]
    fn chat_with_tools_enables_parsing() {
        assert!(chat_parse_tool_calls(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools(),
        })));
        assert!(chat_parse_tool_calls(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools(),
            "tool_choice": "required",
        })));
    }

    #[test]
    fn chat_tool_choice_none_disables_parsing() {
        assert!(!chat_parse_tool_calls(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools(),
            "tool_choice": "none",
        })));
    }

    #[test]
    fn chat_empty_tools_array_disables_parsing() {
        assert!(!chat_parse_tool_calls(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
        })));
    }

    #[test]
    fn chat_named_choice_without_tools_keeps_parsing_for_forced_prefix() {
        // A named tool_choice injects a forced `<tool_call>` opening into the
        // text stream even with no tools advertised, so the parser must stay
        // on to reassemble the server-injected XML.
        assert!(chat_parse_tool_calls(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        })));
    }

    #[test]
    fn responses_without_tools_disables_parsing() {
        assert!(!responses_parse_tool_calls(serde_json::json!({
            "input": "hi",
        })));
    }

    #[test]
    fn responses_with_tools_enables_parsing() {
        // The responses endpoint also accepts the flat tool shape.
        assert!(responses_parse_tool_calls(serde_json::json!({
            "input": "hi",
            "tools": [{"type": "function", "name": "get_weather"}],
        })));
        assert!(responses_parse_tool_calls(serde_json::json!({
            "input": "hi",
            "tools": tools(),
        })));
    }

    #[test]
    fn responses_tool_choice_none_disables_parsing() {
        assert!(!responses_parse_tool_calls(serde_json::json!({
            "input": "hi",
            "tools": tools(),
            "tool_choice": "none",
        })));
    }
