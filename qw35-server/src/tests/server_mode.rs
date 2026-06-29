    use super::*;

    #[test]
    fn presets_match_official_qwen35_values() {
        // (temp, top_p, top_k, min_p, presence, rep, enable_thinking)
        let g = Mode::ThinkingGeneral.defaults();
        assert_eq!(
            (
                g.temperature,
                g.top_p,
                g.top_k,
                g.min_p,
                g.presence_penalty,
                g.repetition_penalty,
                g.enable_thinking
            ),
            // rep=1.1: thinking presets keep the windowed loop-breaker on.
            (1.0, 0.95, 20, 0.0, 1.5, 1.1, true)
        );
        let c = Mode::ThinkingCoding.defaults();
        assert_eq!(
            (
                c.temperature,
                c.top_p,
                c.presence_penalty,
                c.repetition_penalty,
                c.enable_thinking
            ),
            (0.6, 0.95, 0.0, 1.1, true)
        );
        let ig = Mode::InstructGeneral.defaults();
        assert_eq!(
            (
                ig.temperature,
                ig.top_p,
                ig.presence_penalty,
                ig.enable_thinking
            ),
            (0.7, 0.80, 1.5, false)
        );
        let ir = Mode::InstructReasoning.defaults();
        assert_eq!(
            (
                ir.temperature,
                ir.top_p,
                ir.presence_penalty,
                ir.enable_thinking
            ),
            (1.0, 0.95, 1.5, false)
        );
    }

    #[test]
    fn from_name_roundtrips_and_rejects() {
        assert_eq!(
            Mode::from_name("thinking-general"),
            Some(Mode::ThinkingGeneral)
        );
        assert_eq!(
            Mode::from_name("thinking-coding"),
            Some(Mode::ThinkingCoding)
        );
        assert_eq!(
            Mode::from_name("instruct-general"),
            Some(Mode::InstructGeneral)
        );
        assert_eq!(
            Mode::from_name("instruct-reasoning"),
            Some(Mode::InstructReasoning)
        );
        assert_eq!(Mode::from_name("bogus"), None);
    }
