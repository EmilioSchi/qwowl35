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
            super::render_qwen35_chat_prompt_with_boundaries(&messages, false, false);
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
    fn preamble_boundary_is_zero_without_user_turn() {
        // No user/tool turn -> no preamble floor (engine falls back to config sink).
        let messages = [ChatTurn {
            role: "system".to_string(),
            content: "only system".to_string(),
        }];
        let (_prompt, _stable, preamble_len) =
            super::render_qwen35_chat_prompt_with_boundaries(&messages, false, false);
        assert_eq!(preamble_len, 0);
    }

    #[test]
    fn strips_prior_thinking_unless_preserved() {
        let messages = [ChatTurn {
            role: "assistant".to_string(),
            content: "<think>\nprivate\n</think>\n\nvisible".to_string(),
        }];

        let stripped = super::render_qwen35_chat_prompt(&messages, true, false);
        assert!(stripped.contains("\nvisible<|im_end|>"));
        assert!(!stripped.contains("private"));

        let preserved = super::render_qwen35_chat_prompt(&messages, true, true);
        assert!(preserved.contains("private"));
        assert!(preserved.contains("visible"));

        let unclosed = [ChatTurn {
            role: "assistant".to_string(),
            content: "<think>\nstill private".to_string(),
        }];
        let stripped_unclosed = super::render_qwen35_chat_prompt(&unclosed, true, false);
        assert!(!stripped_unclosed.contains("still private"));
    }

    #[test]
    fn renders_generation_thinking_control_markers() {
        let messages = [ChatTurn {
            role: "user".to_string(),
            content: "Say hi.".to_string(),
        }];

        let thinking = super::render_qwen35_chat_prompt(&messages, true, false);
        let non_thinking = super::render_qwen35_chat_prompt(&messages, false, false);

        assert!(thinking.ends_with("<|im_start|>assistant\n<think>\n"));
        assert!(non_thinking.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }
