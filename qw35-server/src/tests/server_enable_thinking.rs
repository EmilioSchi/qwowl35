    use super::*;

    fn chat_req(json: serde_json::Value) -> ChatRequest {
        serde_json::from_value(json).expect("valid ChatRequest")
    }

    fn defaults(enable_thinking: bool) -> GenerationDefaults {
        GenerationDefaults {
            enable_thinking,
            ..GenerationDefaults::default()
        }
    }

    fn resolve(json: serde_json::Value, default_thinking: bool) -> bool {
        into_generate_request(&chat_req(json), "m", &defaults(default_thinking))
            .expect("request builds")
            .enable_thinking
    }

    #[test]
    fn falls_back_to_server_default_when_no_signal() {
        let msgs = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(resolve(msgs.clone(), true));
        assert!(!resolve(msgs, false));
    }

    #[test]
    fn explicit_signals_override_server_default() {
        // top-level enable_thinking=false beats a thinking-on default
        assert!(!resolve(
            serde_json::json!({"messages": [{"role": "user", "content": "hi"}], "enable_thinking": false}),
            true
        ));
        // chat_template_kwargs.enable_thinking=true beats a thinking-off default
        assert!(resolve(
            serde_json::json!({"messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": {"enable_thinking": true}}),
            false
        ));
        // an explicit reasoning_effort implies thinking-on
        assert!(resolve(
            serde_json::json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low"}),
            false
        ));
    }

    #[test]
    fn stream_tool_call_xml_flag_plumbs_through_and_defaults_off() {
        let build = |json: serde_json::Value| {
            into_generate_request(&chat_req(json), "m", &defaults(false)).expect("request builds")
        };
        let msgs = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(!build(msgs).stream_tool_call_xml);
        let flagged = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stream_tool_call_xml": true,
        });
        assert!(build(flagged).stream_tool_call_xml);
    }
