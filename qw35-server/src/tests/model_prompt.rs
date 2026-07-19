    use super::ChatTurn;

    #[test]
    fn renders_qwen35_text_chat_prompt() {
        let prompt = super::render_qwen35_chat_prompt(
            &[
                ChatTurn {
                    role: "system".to_string(),
                    content: "Be concise.".to_string(),
                },
                ChatTurn {
                    role: "developer".to_string(),
                    content: "Prefer plain text.".to_string(),
                },
                ChatTurn {
                    role: "user".to_string(),
                    content: "Say hi.".to_string(),
                },
            ],
            false,
            false,
            None,
            None,
        );

        assert_eq!(
            prompt,
            "<|im_start|>system\nBe concise.\nPrefer plain text.<|im_end|>\n<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn preamble_boundary_covers_system_and_first_user_turn() {
        // System block + first user turn is the preamble; later turns are not.
        let messages = [
            ChatTurn {
                role: "system".to_string(),
                content: "Tool format: <parameter=...>".to_string(),
            },
            ChatTurn {
                role: "user".to_string(),
                content: "first task".to_string(),
            },
            ChatTurn {
                role: "assistant".to_string(),
                content: "ok".to_string(),
            },
            ChatTurn {
                role: "user".to_string(),
                content: "second turn".to_string(),
            },
        ];
        let (prompt, stable_len, preamble_len) =
            super::render_qwen35_chat_prompt_with_boundaries(&messages, false, false, None, None);
        // Preamble ends right after the first user turn, before the assistant turn.
        let expected = "<|im_start|>system\nTool format: <parameter=...><|im_end|>\n<|im_start|>user\nfirst task<|im_end|>\n";
        assert_eq!(&prompt[..preamble_len], expected);
        // It contains the system instructions and the first task, not later turns.
        assert!(prompt[..preamble_len].contains("Tool format"));
        assert!(prompt[..preamble_len].contains("first task"));
        assert!(!prompt[..preamble_len].contains("second turn"));
        // Preamble is within the stable prefix and non-empty.
        assert!(preamble_len > 0 && preamble_len <= stable_len);
    }

    #[test]
    fn enforcement_renders_past_stable_boundary() {
        // The tool_choice must-call instruction lives in the volatile region
        // with the generation header: the stable prefix is byte-identical
        // with and without it, so forcing a call never perturbs the
        // session-cache checkpoint prefix.
        let messages = [
            ChatTurn {
                role: "system".to_string(),
                content: "sys".to_string(),
            },
            ChatTurn {
                role: "user".to_string(),
                content: "task".to_string(),
            },
        ];
        let (plain, plain_stable, _) =
            super::render_qwen35_chat_prompt_with_boundaries(&messages, false, false, None, None);
        let (forced, forced_stable, _) = super::render_qwen35_chat_prompt_with_boundaries(
            &messages,
            false,
            false,
            Some("You must call the function \"verdict\"."),
            None,
        );
        assert_eq!(plain_stable, forced_stable);
        assert_eq!(plain[..plain_stable], forced[..forced_stable]);
        assert_eq!(
            &forced[forced_stable..],
            "<|im_start|>user\nYou must call the function \"verdict\".<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn forced_tool_prefix_renders_past_stable_boundary_with_closed_think() {
        // The named-tool_choice hard enforcement: the call opening renders
        // after the generation header, in the volatile region (the stable
        // prefix is byte-identical with and without it), and always on the
        // closed-think header — an open <think> would swallow the call.
        let messages = [
            ChatTurn {
                role: "system".to_string(),
                content: "sys".to_string(),
            },
            ChatTurn {
                role: "user".to_string(),
                content: "task".to_string(),
            },
        ];
        let (plain, plain_stable, _) =
            super::render_qwen35_chat_prompt_with_boundaries(&messages, false, false, None, None);
        let (forced, forced_stable, _) = super::render_qwen35_chat_prompt_with_boundaries(
            &messages,
            false,
            false,
            Some("You must call the function \"useful\"."),
            Some("<tool_call>\n<function=useful>\n"),
        );
        assert_eq!(plain_stable, forced_stable);
        assert_eq!(plain[..plain_stable], forced[..forced_stable]);
        assert_eq!(
            &forced[forced_stable..],
            "<|im_start|>user\nYou must call the function \"useful\".<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=useful>\n"
        );

        // Thinking requested + forced prefix -> the closed header still wins.
        let (forced_thinking, _, _) = super::render_qwen35_chat_prompt_with_boundaries(
            &messages,
            true,
            false,
            None,
            Some("<tool_call>\n<function=useful>\n"),
        );
        assert!(forced_thinking.ends_with(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=useful>\n"
        ));
    }

    #[test]
    fn preamble_boundary_is_zero_without_user_turn() {
        // No user/tool turn -> no preamble floor (engine falls back to config sink).
        let messages = [ChatTurn {
            role: "system".to_string(),
            content: "only system".to_string(),
        }];
        let (_prompt, _stable, preamble_len) =
            super::render_qwen35_chat_prompt_with_boundaries(&messages, false, false, None, None);
        assert_eq!(preamble_len, 0);
    }

    #[test]
    fn strips_prior_thinking_unless_preserved() {
        let messages = [ChatTurn {
            role: "assistant".to_string(),
            content: "<think>\nprivate\n</think>\n\nvisible".to_string(),
        }];

        let stripped = super::render_qwen35_chat_prompt(&messages, true, false, None, None);
        assert!(stripped.contains("\nvisible<|im_end|>"));
        assert!(!stripped.contains("private"));

        let preserved = super::render_qwen35_chat_prompt(&messages, true, true, None, None);
        assert!(preserved.contains("private"));
        assert!(preserved.contains("visible"));

        let unclosed = [ChatTurn {
            role: "assistant".to_string(),
            content: "<think>\nstill private".to_string(),
        }];
        let stripped_unclosed = super::render_qwen35_chat_prompt(&unclosed, true, false, None, None);
        assert!(!stripped_unclosed.contains("still private"));
    }

    #[test]
    fn renders_generation_thinking_control_markers() {
        let messages = [ChatTurn {
            role: "user".to_string(),
            content: "Say hi.".to_string(),
        }];

        let thinking = super::render_qwen35_chat_prompt(&messages, true, false, None, None);
        let non_thinking = super::render_qwen35_chat_prompt(&messages, false, false, None, None);

        assert!(thinking.ends_with("<|im_start|>assistant\n<think>\n"));
        assert!(non_thinking.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }
