    use super::*;

    #[test]
    fn effort_scales_against_fixed_max_tokens() {
        let max = TokenLimit::Fixed(8192);
        assert_eq!(thinking_budget_for(Some("low"), true, max), Some(327));
        assert_eq!(thinking_budget_for(Some("medium"), true, max), Some(819));
        assert_eq!(thinking_budget_for(Some("high"), true, max), Some(1310));
        // xhigh falls back to the 0.16 backstop (not uncapped) so a looping
        // reasoner is always force-closed eventually.
        assert_eq!(thinking_budget_for(Some("xhigh"), true, max), Some(1310));
    }

    #[test]
    fn context_limit_scales_against_client_default() {
        let max = TokenLimit::Context;
        assert_eq!(thinking_budget_for(Some("medium"), true, max), Some(819));
    }

    #[test]
    fn uncapped_only_when_thinking_off() {
        let max = TokenLimit::Fixed(8192);
        // Thinking off → never capped, regardless of effort.
        assert_eq!(thinking_budget_for(Some("high"), false, max), None);
        assert_eq!(thinking_budget_for(None, false, max), None);
        // Thinking on with no/unknown effort → 0.16 backstop so the decoder
        // always forces `</think>` rather than looping to context end.
        assert_eq!(thinking_budget_for(None, true, max), Some(1310));
        assert_eq!(thinking_budget_for(Some("bogus"), true, max), Some(1310));
    }

    #[test]
    fn effort_is_case_insensitive_and_floored() {
        assert_eq!(
            thinking_budget_for(Some("HIGH"), true, TokenLimit::Fixed(8192)),
            Some(1310)
        );
        // Tiny budgets are floored to a usable minimum.
        assert_eq!(
            thinking_budget_for(Some("low"), true, TokenLimit::Fixed(8)),
            Some(16)
        );
    }
