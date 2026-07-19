    use super::*;

    fn run(feeds: &[&str], start_in_thinking: bool) -> Vec<AssistantEvent> {
        let mut parser = AssistantStreamParser::new(start_in_thinking);
        let mut events = Vec::new();
        for feed in feeds {
            events.extend(parser.feed(feed));
        }
        events.extend(parser.finish());
        events
    }

    fn run_split_everywhere(text: &str, start_in_thinking: bool) -> Vec<ParsedAssistantOutput> {
        let boundaries: Vec<usize> = text.char_indices().map(|(idx, _)| idx).skip(1).collect();
        let mut outputs = Vec::new();
        for &split in &boundaries {
            let events = run(&[&text[..split], &text[split..]], start_in_thinking);
            outputs.push(assemble_events(&events));
        }
        // Also one char at a time.
        let chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        let char_feeds: Vec<&str> = chars.iter().map(String::as_str).collect();
        outputs.push(assemble_events(&run(&char_feeds, start_in_thinking)));
        outputs
    }

    const CALL_BLOCK: &str = "<tool_call>\n<get_weather city=\"Paris\"/>\n</tool_call>";

    #[test]
    fn plain_text_passes_through() {
        let out = assemble_events(&run(&["Hello", " world"], false));
        assert_eq!(out.content, "Hello world");
        assert!(out.tool_calls.is_empty());
        assert!(out.reasoning.is_empty());
    }

    #[test]
    fn lone_angle_brackets_are_content() {
        let out = assemble_events(&run(&["a < b and a <t", "ag> too"], false));
        assert_eq!(out.content, "a < b and a <tag> too");
    }

    #[test]
    fn single_tool_call_parses_at_every_split_point() {
        for out in run_split_everywhere(CALL_BLOCK, false) {
            assert_eq!(out.content, "", "content leaked: {:?}", out.content);
            assert_eq!(out.tool_calls.len(), 1);
            assert_eq!(out.tool_calls[0].name, "get_weather");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&out.tool_calls[0].arguments).unwrap(),
                serde_json::json!({"city": "Paris"})
            );
            assert!(out.tool_calls[0].id.starts_with("call_"));
        }
    }

    #[test]
    fn forced_prefix_then_model_completion_assembles_one_call() {
        // The generate loop seeds the text stream with the forced opening
        // (`<tool_call>\n<function=X>\n`) before the model's sampled
        // parameter body arrives: the parser must reconstruct exactly one
        // well-formed call named X. The forced turn never starts in thinking
        // (the closed-think header is rendered), hence start_in_thinking =
        // false.
        let completion =
            "<parameter=verdict>\nyes\n</parameter>\n<parameter=meaning>\nholds the bug\n</parameter>\n</function>\n</tool_call>";
        let out = assemble_events(&run(
            &["<tool_call>\n<function=useful>\n", completion],
            false,
        ));
        assert_eq!(out.content, "", "content leaked: {:?}", out.content);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "useful");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&out.tool_calls[0].arguments).unwrap(),
            serde_json::json!({"verdict": "yes", "meaning": "holds the bug"})
        );
    }

    #[test]
    fn text_then_tool_call_then_text() {
        let text = format!("Let me check.\n{CALL_BLOCK}\nDone.");
        for out in run_split_everywhere(&text, false) {
            assert_eq!(out.content, "Let me check.\nDone.");
            assert_eq!(out.tool_calls.len(), 1);
        }
    }

    #[test]
    fn two_tool_calls_get_sequential_indexes() {
        let text = format!("{CALL_BLOCK}\n<tool_call>\n<list_files path=\"/tmp\"/>\n</tool_call>");
        let events = run(&[&text], false);
        let begins: Vec<(usize, String)> = events
            .iter()
            .filter_map(|event| match event {
                AssistantEvent::ToolCallBegin { index, name, .. } => Some((*index, name.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            begins,
            vec![
                (0, "get_weather".to_string()),
                (1, "list_files".to_string())
            ]
        );
        let out = assemble_events(&events);
        assert_eq!(out.tool_calls.len(), 2);
        assert_eq!(out.tool_calls[1].arguments, "{\"path\":\"/tmp\"}");
        assert_eq!(out.content, "");
    }

    fn bash_block(cmd: &str) -> String {
        format!(
            "<tool_call>\n<function=bash>\n<parameter=command>\n{cmd}\n</parameter>\n</function>\n</tool_call>"
        )
    }

    fn assert_bash_command_parses(cmd: &str) {
        let block = bash_block(cmd);
        for out in run_split_everywhere(&block, false) {
            assert_eq!(out.content, "", "content leaked for {cmd:?}");
            assert_eq!(out.reasoning, "", "reasoning leaked for {cmd:?}");
            assert_eq!(out.tool_calls.len(), 1, "no single call for {cmd:?}");
            assert_eq!(out.tool_calls[0].name, "bash");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&out.tool_calls[0].arguments).unwrap(),
                serde_json::json!({ "command": cmd }),
                "wrong command for {cmd:?}",
            );
        }
    }

    #[test]
    fn bash_command_containing_close_parameter_literal_parses() {
        // Dogfooding qw35 on its own repo: commands routinely cat/grep files that
        // contain the literal tool-call XML vocabulary. A verbatim `</parameter>`
        // in the command must not be mistaken for the parameter's close tag.
        assert_bash_command_parses("echo '</parameter> appears in this file'");
        assert_bash_command_parses("grep -n '</parameter>' qwowl35/tools/bash/guidance.py");
    }

    #[test]
    fn bash_command_containing_close_function_literal_parses() {
        assert_bash_command_parses("echo 'the </function> tag closes a call'");
    }


    #[test]
    fn bash_command_containing_close_param_inside_thinking_does_not_leak() {
        // The original bug: a think-origin call whose command holds `</parameter>`
        // demoted to raw text that surfaced in the thinking panel.
        let block = format!("Let me inspect it.\n{}\n", bash_block("echo '</parameter>'"));
        let out = assemble_events(&run(&[&block], true));
        assert_eq!(out.tool_calls.len(), 1, "call should parse, not leak");
        assert!(
            !out.reasoning.contains("<tool_call>"),
            "raw tool_call XML leaked into thinking: {:?}",
            out.reasoning
        );
        assert_eq!(out.reasoning.trim(), "Let me inspect it.");
    }

    #[test]
    fn streams_argument_fragments_with_name_in_first_event() {
        let chars: Vec<String> = CALL_BLOCK.chars().map(|c| c.to_string()).collect();
        let feeds: Vec<&str> = chars.iter().map(String::as_str).collect();
        let events = run(&feeds, false);
        let mut saw_begin = false;
        let mut fragments = String::new();
        for event in &events {
            match event {
                AssistantEvent::ToolCallBegin { name, .. } => {
                    assert!(!saw_begin);
                    assert_eq!(name, "get_weather");
                    saw_begin = true;
                }
                AssistantEvent::ToolCallArgs { fragment, .. } => {
                    assert!(saw_begin, "args before begin");
                    fragments.push_str(fragment);
                }
                AssistantEvent::ToolCallEnd { arguments, .. } => {
                    assert_eq!(&fragments, arguments);
                }
                AssistantEvent::Content(text) => panic!("unexpected content {text:?}"),
                AssistantEvent::Reasoning(_) => panic!("unexpected reasoning"),
                AssistantEvent::ToolCallName { .. } | AssistantEvent::ToolCallDemoted { .. } => {
                    panic!("raw-streaming event in plain mode")
                }
            }
        }
        assert!(saw_begin);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&fragments).unwrap(),
            serde_json::json!({"city": "Paris"})
        );
    }

    #[test]
    fn braces_and_escapes_inside_string_arguments() {
        let block = "<tool_call>\n<write text=\"a {b} &quot;c&quot; ] }\"/>\n</tool_call>";
        for out in run_split_everywhere(block, false) {
            assert_eq!(out.tool_calls.len(), 1);
            let args: serde_json::Value =
                serde_json::from_str(&out.tool_calls[0].arguments).unwrap();
            assert_eq!(args["text"], "a {b} \"c\" ] }");
            assert_eq!(out.content, "");
        }
    }

    #[test]
    fn xml_escaped_end_tag_inside_argument_survives_streaming() {
        let block = "<tool_call>\n<write text=\"x &lt;/tool_call&gt; y\"/>\n</tool_call>";
        let chars: Vec<String> = block.chars().map(|c| c.to_string()).collect();
        let feeds: Vec<&str> = chars.iter().map(String::as_str).collect();
        let out = assemble_events(&run(&feeds, false));
        assert_eq!(out.tool_calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&out.tool_calls[0].arguments).unwrap();
        assert_eq!(args["text"], "x </tool_call> y");
    }

    #[test]
    fn code_like_arguments_with_xml_chars_and_nested_json_parse() {
        let block = concat!(
            "<tool_call>\n",
            "<write code=\"if (a &lt; b &amp;&amp; c &gt; d) { return x &amp; y; }\" ",
            "meta=\"{&quot;items&quot;:[1,{&quot;v&quot;:&quot;&lt;&gt;&amp;&quot;}]}\"/>\n",
            "</tool_call>"
        );
        for out in run_split_everywhere(block, false) {
            assert_eq!(out.tool_calls.len(), 1);
            assert_eq!(out.tool_calls[0].name, "write");
            let args: serde_json::Value =
                serde_json::from_str(&out.tool_calls[0].arguments).unwrap();
            assert_eq!(args["code"], "if (a < b && c > d) { return x & y; }");
            assert_eq!(args["meta"]["items"][1]["v"], "<>&");
            assert_eq!(out.content, "");
        }
    }

    #[test]
    fn malformed_bash_attribute_recovers_shell_command() {
        let block = r#"<tool_call>
<bash command="python3 roots.py 2>&1 || python3 /dev/stdin << 'PYEOF'\nprint("hi")\nPYEOF"
</tool_call>"#;
        let out = assemble_events(&run(&[block], false));
        assert_eq!(out.content, "");
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "bash");
        let args: serde_json::Value = serde_json::from_str(&out.tool_calls[0].arguments).unwrap();
        assert_eq!(
            args["command"],
            "python3 roots.py 2>&1 || python3 /dev/stdin << 'PYEOF'\nprint(\"hi\")\nPYEOF"
        );
    }

    #[test]
    fn malformed_bash_attribute_inside_thinking_is_recovered() {
        let text = "Need a command.\n<tool_call>\n<bash command=\"echo hi 2>&1\"\n</tool_call>";
        let out = assemble_events(&run(&[text], true));
        assert_eq!(out.reasoning, "Need a command.\n");
        assert_eq!(out.content, "");
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "bash");
        let args: serde_json::Value = serde_json::from_str(&out.tool_calls[0].arguments).unwrap();
        assert_eq!(args["command"], "echo hi 2>&1");
    }

    #[test]
    fn json_tool_call_is_demoted_to_content() {
        let block = "<tool_call>\n{\"name\":\"get_weather\",\"city\":\"Rome\"}\n</tool_call>";
        let out = assemble_events(&run(&[block], false));
        assert!(out.tool_calls.is_empty());
        assert_eq!(out.content, block);
    }

    #[test]
    fn legacy_nested_json_arguments_are_demoted_to_content() {
        let block =
            "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Rome\"}}\n</tool_call>";
        let out = assemble_events(&run(&[block], false));
        assert!(out.tool_calls.is_empty());
        assert_eq!(out.content, block);
    }

    #[test]
    fn quoted_attribute_values_parse() {
        let block = "<tool_call>\n<run cmd='ls'></run>\n</tool_call>";
        for out in run_split_everywhere(block, false) {
            assert_eq!(out.tool_calls.len(), 1);
            assert_eq!(out.tool_calls[0].name, "run");
            let args: serde_json::Value =
                serde_json::from_str(&out.tool_calls[0].arguments).unwrap();
            assert_eq!(args["cmd"], "ls");
        }
    }

    #[test]
    fn name_only_call_gets_empty_object_arguments() {
        let block = "<tool_call>\n<ping/>\n</tool_call>";
        for out in run_split_everywhere(block, false) {
            assert_eq!(out.tool_calls.len(), 1);
            assert_eq!(out.tool_calls[0].name, "ping");
            assert_eq!(out.tool_calls[0].arguments, "{}");
        }
    }

    #[test]
    fn malformed_block_degrades_to_content() {
        let block = "<tool_call>\nnot xml at all\n</tool_call>";
        let out = assemble_events(&run(&[block], false));
        assert!(out.tool_calls.is_empty());
        assert_eq!(out.content, block);
    }

    #[test]
    fn unterminated_call_at_eof_still_ends_the_call() {
        let partial = "<tool_call>\n<get_weather city=\"Paris\"/>";
        let out = assemble_events(&run(&[partial], false));
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&out.tool_calls[0].arguments).unwrap();
        assert_eq!(args["city"], "Paris");
    }

    #[test]
    fn unterminated_header_at_eof_degrades_to_content() {
        let partial = "<tool_call>\n<get";
        let out = assemble_events(&run(&[partial], false));
        assert!(out.tool_calls.is_empty());
        assert_eq!(out.content, partial);
    }

    #[test]
    fn partial_tag_at_eof_flushes_as_content() {
        let out = assemble_events(&run(&["hello <tool_c"], false));
        assert_eq!(out.content, "hello <tool_c");
    }

    #[test]
    fn thinking_block_routes_to_reasoning() {
        let text = "I should check the weather.\n</think>\n\nSure thing.";
        for out in run_split_everywhere(text, true) {
            assert_eq!(out.reasoning, "I should check the weather.\n");
            assert_eq!(out.content, "Sure thing.");
        }
    }

    #[test]
    fn inline_think_block_routes_to_reasoning() {
        let text = "<think>\nplan it\n</think>\n\nAnswer.";
        for out in run_split_everywhere(text, false) {
            assert_eq!(out.reasoning, "plan it\n");
            assert_eq!(out.content, "Answer.");
        }
    }

    #[test]
    fn thinking_then_tool_call() {
        let text = format!("Need the weather.\n</think>\n\n{CALL_BLOCK}");
        for out in run_split_everywhere(&text, true) {
            assert_eq!(out.reasoning, "Need the weather.\n");
            assert_eq!(out.content, "");
            assert_eq!(out.tool_calls.len(), 1);
            assert_eq!(out.tool_calls[0].name, "get_weather");
        }
    }

    #[test]
    fn parseable_tool_call_inside_thinking_is_recovered() {
        let text = format!("Need the weather.\n{CALL_BLOCK}");
        for out in run_split_everywhere(&text, true) {
            assert_eq!(out.reasoning, "Need the weather.\n");
            assert_eq!(out.content, "");
            assert_eq!(out.tool_calls.len(), 1);
            assert_eq!(out.tool_calls[0].name, "get_weather");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&out.tool_calls[0].arguments).unwrap(),
                serde_json::json!({"city": "Paris"})
            );
        }
    }

    #[test]
    fn tool_call_tags_inside_thinking_stay_reasoning() {
        let text = "maybe <tool_call> here?\n</think>\nNo.";
        let out = assemble_events(&run(&[text], true));
        assert_eq!(out.reasoning, "maybe <tool_call> here?\n");
        assert_eq!(out.content, "No.");
        assert!(out.tool_calls.is_empty());
    }

    #[test]
    fn leading_newlines_are_trimmed_per_segment() {
        let out = assemble_events(&run(&["\n\nHello"], false));
        assert_eq!(out.content, "Hello");
        let text = format!("\n{CALL_BLOCK}\n\nAfter.");
        let out = assemble_events(&run(&[&text], false));
        assert_eq!(out.content, "After.");
    }

    #[test]
    fn degraded_block_does_not_consume_a_call_index() {
        let bad = "<tool_call>\nnope\n</tool_call>";
        let text = format!("{bad}\n{CALL_BLOCK}");
        let events = run(&[&text], false);
        let out = assemble_events(&events);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "get_weather");
        assert!(events
            .iter()
            .any(|event| matches!(event, AssistantEvent::ToolCallBegin { index: 0, .. })));
    }

    #[test]
    fn legacy_function_parameter_form_still_parses() {
        let block = "<tool_call>\n<function name=\"run\">\n<parameter name=\"cmd\">\nls\n</parameter>\n</function>\n</tool_call>";
        let out = assemble_events(&run(&[block], false));
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "run");
        let args: serde_json::Value = serde_json::from_str(&out.tool_calls[0].arguments).unwrap();
        assert_eq!(args["cmd"], "ls");
    }

    #[test]
    fn equals_function_parameter_form_parses() {
        let block = "<tool_call>\n<function=edit_lines_if_file_exists>\n<parameter=file>m.py</parameter>\n<parameter=anchor>2:aa</parameter>\n<parameter=content>\n    return 2\n</parameter>\n</function>\n</tool_call>";
        let out = assemble_events(&run(&[block], false));
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "edit_lines_if_file_exists");
        let args: serde_json::Value = serde_json::from_str(&out.tool_calls[0].arguments).unwrap();
        assert_eq!(args["file"], "m.py");
        assert_eq!(args["anchor"], "2:aa");
        assert_eq!(args["content"], "    return 2");
    }

    #[test]
    fn rootless_equals_function_form_parses_like_vllm() {
        let block = "<function=read_file>\n<parameter=file>solve.py</parameter>\n<parameter=context>4</parameter>\n</function>";
        for out in run_split_everywhere(block, false) {
            assert_eq!(out.tool_calls.len(), 1);
            assert_eq!(out.tool_calls[0].name, "read_file");
            let args: serde_json::Value =
                serde_json::from_str(&out.tool_calls[0].arguments).unwrap();
            assert_eq!(args["file"], "solve.py");
            assert_eq!(args["context"], "4");
        }
    }

    #[test]
    fn rootless_name_attribute_function_form_parses_like_vllm() {
        let block = "<function name=\"read_file\">\n<parameter name=\"file\">solve.py</parameter>\n</function>";
        let out = assemble_events(&run(&[block], false));
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "read_file");
        let args: serde_json::Value = serde_json::from_str(&out.tool_calls[0].arguments).unwrap();
        assert_eq!(args["file"], "solve.py");
    }

    #[test]
    fn parses_chat_and_responses_tool_defs() {
        let chat = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }]);
        let defs = parse_tool_defs(&chat).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "get_weather");
        assert!(defs[0].parameters.is_some());

        let responses = serde_json::json!([
            {"type": "function", "name": "list_files", "parameters": {"type": "object"}},
            {"type": "web_search"}
        ]);
        let defs = parse_tool_defs(&responses).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "list_files");

        assert!(parse_tool_defs(&serde_json::json!("nope")).is_err());
        assert!(parse_tool_defs(&serde_json::json!([{"type": "function"}])).is_err());
    }

    #[test]
    fn parses_tool_choice_shapes() {
        assert_eq!(parse_tool_choice(None), ToolChoice::Auto);
        assert_eq!(
            parse_tool_choice(Some(&serde_json::json!("auto"))),
            ToolChoice::Auto
        );
        assert_eq!(
            parse_tool_choice(Some(&serde_json::json!("none"))),
            ToolChoice::None
        );
        assert_eq!(
            parse_tool_choice(Some(&serde_json::json!("required"))),
            ToolChoice::Required
        );
        assert_eq!(
            parse_tool_choice(Some(
                &serde_json::json!({"type": "function", "function": {"name": "f"}})
            )),
            ToolChoice::Named("f".to_string())
        );
        assert_eq!(
            parse_tool_choice(Some(&serde_json::json!({"type": "function", "name": "g"}))),
            ToolChoice::Named("g".to_string())
        );
    }

    #[test]
    fn renders_tools_block_and_tool_call_block() {
        // The raw object is exactly what an OpenAI-style client sends; the block
        // dumps it with `tool | tojson`, matching the embedded chat_template.
        let raw = serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature units"
                        }
                    },
                    "required": ["city"]
                }
            }
        });
        let defs = vec![ToolDef {
            name: "get_weather".to_string(),
            description: Some("Get the weather".to_string()),
            parameters: Some(raw["function"]["parameters"].clone()),
            raw,
        }];
        // Regression guard: the rendered block must be byte-identical to the
        // model's embedded `tokenizer.chat_template` tools section for
        // ToolChoice::Auto. `verify_tool_template.py` derives this expected text
        // from the GGUF, so a wording/separator drift here fails the build.
        let block = render_tools_system_block(&defs).unwrap();
        let expected = "# Tools\n\nYou have access to the following functions:\n\n<tools>\n\
{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"City name\"}, \"units\": {\"type\": \"string\", \"enum\": [\"celsius\", \"fahrenheit\"], \"description\": \"Temperature units\"}}, \"required\": [\"city\"]}}}\n\
</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n\
<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n\
</function>\n</tool_call>\n\n\
<IMPORTANT>\nReminder:\n\
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
- Required parameters MUST be specified\n\
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n\
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n\
</IMPORTANT>";
        assert_eq!(block, expected);
        assert!(render_tools_system_block(&[]).is_none());

        // The must-call instruction is no longer part of the system block: it
        // renders past the stable prompt boundary so `tool_choice` never
        // perturbs the session-cache checkpoint prefix.
        assert_eq!(enforcement_suffix(&ToolChoice::Auto), None);
        assert_eq!(enforcement_suffix(&ToolChoice::None), None);
        assert_eq!(
            enforcement_suffix(&ToolChoice::Required).as_deref(),
            Some("You must call at least one function before answering.")
        );
        assert_eq!(
            enforcement_suffix(&ToolChoice::Named("verdict".to_string())).as_deref(),
            Some("You must call the function \"verdict\".")
        );

        // The named choice additionally hard-forces the call by injecting its
        // opening into the prompt; `required` stays instruction-only (the
        // model must still pick which function to call).
        assert_eq!(forced_call_prefix(&ToolChoice::Auto), None);
        assert_eq!(forced_call_prefix(&ToolChoice::None), None);
        assert_eq!(forced_call_prefix(&ToolChoice::Required), None);
        assert_eq!(
            forced_call_prefix(&ToolChoice::Named("useful".to_string())).as_deref(),
            Some("<tool_call>\n<function=useful>\n")
        );

        let call = render_tool_call_block("get_weather", "{\"city\": \"Paris\"}");
        assert_eq!(
            call,
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
        );
        let empty = render_tool_call_block("ping", "");
        assert_eq!(
            empty,
            "<tool_call>\n<function=ping>\n</function>\n</tool_call>"
        );
        let non_object = render_tool_call_block("run", "\"ls\"");
        assert_eq!(
            non_object,
            "<tool_call>\n<function=run>\n<parameter=arguments>\nls\n</parameter>\n</function>\n</tool_call>"
        );
        let colliding = render_tool_call_block("meta", "{\"name\":\"inner\"}");
        assert_eq!(
            colliding,
            "<tool_call>\n<function=meta>\n<parameter=name>\ninner\n</parameter>\n</function>\n</tool_call>"
        );
        let edit = render_tool_call_block(
            "edit_lines_if_file_exists",
            "{\"file\":\"m.py\",\"anchor\":\"2:aa\",\"content\":\"    return 2\"}",
        );
        assert!(edit.contains("<function=edit_lines_if_file_exists>"));
        assert!(edit.contains("<parameter=file>\nm.py\n</parameter>"));
        assert!(edit.contains("<parameter=anchor>\n2:aa\n</parameter>"));
        assert!(edit.contains("<parameter=content>\n    return 2\n</parameter>"));
    }

    #[test]
    fn tool_call_values_render_verbatim_not_escaped() {
        // The embedded chat_template renders parameter values verbatim (no entity
        // escaping). Escaping `<`/`>`/`&` here was off-distribution and made the
        // model loop converting `>`->`&gt;`. Pin verbatim rendering so it can't
        // regress.
        let call = render_tool_call_block(
            "edit_lines_if_file_exists",
            "{\"content\":\"print(f\\\"{a} > {b} < {c} && {d}\\\")\"}",
        );
        assert!(
            call.contains("<parameter=content>\nprint(f\"{a} > {b} < {c} && {d}\")\n</parameter>"),
            "value must render verbatim, got:\n{call}"
        );
        assert!(!call.contains("&gt;"), "no &gt; escaping: {call}");
        assert!(!call.contains("&lt;"), "no &lt; escaping: {call}");
        assert!(!call.contains("&amp;"), "no &amp; escaping: {call}");
    }

    // ── Raw-streaming mode (stream_tool_call_xml) ──────────────────────────

    fn run_streaming(feeds: &[&str], start_in_thinking: bool) -> Vec<AssistantEvent> {
        let mut parser = AssistantStreamParser::with_options(start_in_thinking, true, true);
        let mut events = Vec::new();
        for feed in feeds {
            events.extend(parser.feed(feed));
        }
        events.extend(parser.finish());
        events
    }

    /// Ignore ids (unique per builder) when comparing streamed vs whole parses.
    fn assert_same_output(streamed: &ParsedAssistantOutput, whole: &ParsedAssistantOutput) {
        assert_eq!(streamed.content, whole.content);
        assert_eq!(streamed.reasoning, whole.reasoning);
        assert_eq!(streamed.tool_calls.len(), whole.tool_calls.len());
        for (s, w) in streamed.tool_calls.iter().zip(&whole.tool_calls) {
            assert_eq!(s.name, w.name);
            assert_eq!(s.arguments, w.arguments);
        }
    }

    #[test]
    fn stream_raw_assembles_identically_at_every_split_point() {
        let fixtures = [
            bash_block("echo '</parameter> appears in this file'"),
            bash_block("cat <<'EOF' > f.py\nprint('x')\nprint('y')\nEOF"),
            CALL_BLOCK.to_string(),
            format!("Let me check.\n{CALL_BLOCK}\nDone."),
            "<tool_call>\nnope\n</tool_call>".to_string(),
        ];
        for text in &fixtures {
            let whole = parse_assistant_text(text, false, true);
            for split in text.char_indices().map(|(idx, _)| idx).skip(1) {
                let events = run_streaming(&[&text[..split], &text[split..]], false);
                assert_same_output(&assemble_events(&events), &whole);
            }
            let chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();
            let char_feeds: Vec<&str> = chars.iter().map(String::as_str).collect();
            let events = run_streaming(&char_feeds, false);
            assert_same_output(&assemble_events(&events), &whole);
        }
    }

    #[test]
    fn stream_raw_begin_fires_before_body_and_name_arrives_early() {
        let block = bash_block("echo hi");
        // Feed only up to the end of the `<function=bash>` header line.
        let header_end = block.find(">\n<parameter").unwrap() + 1;
        let mut parser = AssistantStreamParser::with_options(false, true, true);
        let events = parser.feed(&block[..header_end]);
        let begin_pos = events
            .iter()
            .position(|e| matches!(e, AssistantEvent::ToolCallBegin { index: 0, name, .. } if name.is_empty()))
            .expect("early Begin with empty name");
        let name_pos = events
            .iter()
            .position(|e| matches!(e, AssistantEvent::ToolCallName { index: 0, name } if name == "bash"))
            .expect("early Name from the streamed header");
        assert!(begin_pos < name_pos);
        // No content leaked; the body streamed as raw Args fragments.
        assert!(events.iter().all(|e| !matches!(e, AssistantEvent::Content(_))));
    }

    #[test]
    fn stream_raw_args_fragments_reconstruct_the_body_without_end_tag() {
        let cmd = "cat <<'EOF' > demo.txt\nline one\nline two </tool_\nEOF";
        let block = bash_block(cmd);
        // Char-at-a-time worst case: holdback must keep every partial
        // `</tool_call>` prefix out of the emitted fragments.
        let chars: Vec<String> = block.chars().map(|c| c.to_string()).collect();
        let char_feeds: Vec<&str> = chars.iter().map(String::as_str).collect();
        let events = run_streaming(&char_feeds, false);
        let streamed: String = events
            .iter()
            .filter_map(|e| match e {
                AssistantEvent::ToolCallArgs { fragment, .. } => Some(fragment.as_str()),
                _ => None,
            })
            .collect();
        let body_start = "<tool_call>".len();
        let body_end = block.len() - "</tool_call>".len();
        assert_eq!(streamed, &block[body_start..body_end]);
        assert!(!streamed.contains("</tool_call>"));
        // The end still carries the authoritative parsed arguments.
        let out = assemble_events(&events);
        assert_eq!(out.tool_calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&out.tool_calls[0].arguments).unwrap();
        assert_eq!(args["command"], cmd);
    }

    #[test]
    fn stream_raw_demote_rolls_back_the_call_index() {
        let bad = "<tool_call>\nnope\n</tool_call>";
        let text = format!("{bad}\n{CALL_BLOCK}");
        let events = run_streaming(&[&text], false);
        // The bad block committed index 0 then demoted it; the valid call must
        // re-use index 0.
        assert!(events
            .iter()
            .any(|e| matches!(e, AssistantEvent::ToolCallDemoted { index: 0 })));
        let begins: Vec<usize> = events
            .iter()
            .filter_map(|e| match e {
                AssistantEvent::ToolCallBegin { index, .. } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(begins, vec![0, 0]);
        let out = assemble_events(&events);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "get_weather");
        assert!(out.content.contains("<tool_call>\nnope\n</tool_call>"));
    }

    #[test]
    fn stream_raw_demote_inside_thinking_routes_to_reasoning() {
        let text = "<tool_call>\n<function=bash>\nbroken\n</tool_call>\nstill thinking";
        let events = run_streaming(&[text], true);
        assert!(events
            .iter()
            .any(|e| matches!(e, AssistantEvent::ToolCallDemoted { .. })));
        let out = assemble_events(&events);
        assert!(out.tool_calls.is_empty());
        assert!(out.content.is_empty(), "content leaked: {:?}", out.content);
        assert!(out.reasoning.contains("still thinking"));
    }

    #[test]
    fn stream_raw_bash_attribute_header_names_bash_early() {
        let mut parser = AssistantStreamParser::with_options(false, true, true);
        let events = parser.feed("<tool_call>\n<bash command=\"ls -");
        assert!(events
            .iter()
            .any(|e| matches!(e, AssistantEvent::ToolCallName { name, .. } if name == "bash")));
    }

    #[test]
    fn plain_mode_emits_no_raw_streaming_events() {
        let block = bash_block("cat <<'EOF' > f.py\nprint('x')\nEOF");
        let events = run(&[&block], false);
        assert!(events.iter().all(|e| !matches!(
            e,
            AssistantEvent::ToolCallName { .. } | AssistantEvent::ToolCallDemoted { .. }
        )));
        // Exactly the legacy shape: one Begin, one Args (full JSON), one End.
        let args: Vec<&String> = events
            .iter()
            .filter_map(|e| match e {
                AssistantEvent::ToolCallArgs { fragment, .. } => Some(fragment),
                _ => None,
            })
            .collect();
        assert_eq!(args.len(), 1);
        serde_json::from_str::<serde_json::Value>(args[0]).expect("legacy Args is full JSON");
    }

    // ── Tool-call parsing disabled (request advertised no tools) ───────────

    fn run_no_tools(feeds: &[&str], start_in_thinking: bool) -> Vec<AssistantEvent> {
        let mut parser = AssistantStreamParser::with_options(start_in_thinking, false, false);
        let mut events = Vec::new();
        for feed in feeds {
            events.extend(parser.feed(feed));
        }
        events.extend(parser.finish());
        events
    }

    #[test]
    fn disabled_mode_passes_tool_call_xml_as_content_at_every_split_point() {
        let boundaries: Vec<usize> = CALL_BLOCK
            .char_indices()
            .map(|(idx, _)| idx)
            .skip(1)
            .collect();
        for &split in &boundaries {
            let out = assemble_events(&run_no_tools(
                &[&CALL_BLOCK[..split], &CALL_BLOCK[split..]],
                false,
            ));
            assert_eq!(out.content, CALL_BLOCK, "split at {split}");
            assert!(out.tool_calls.is_empty());
            assert!(out.reasoning.is_empty());
        }
    }

    #[test]
    fn disabled_mode_keeps_think_extraction() {
        let text = format!("<think>\nplan it\n</think>\n\n{CALL_BLOCK}");
        let out = assemble_events(&run_no_tools(&[&text], false));
        assert_eq!(out.reasoning, "plan it\n");
        assert_eq!(out.content, CALL_BLOCK);
        assert!(out.tool_calls.is_empty());
    }

    #[test]
    fn disabled_mode_passes_rootless_function_xml_as_content() {
        let text = "<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>";
        let out = assemble_events(&run_no_tools(&[text], false));
        assert_eq!(out.content, text);
        assert!(out.tool_calls.is_empty());
    }

    #[test]
    fn disabled_mode_flushes_partial_tool_call_at_eof() {
        let text = "text <tool_call>\n<function=x>";
        let out = assemble_events(&run_no_tools(&[text], false));
        assert_eq!(out.content, text);
        assert!(out.tool_calls.is_empty());
    }
