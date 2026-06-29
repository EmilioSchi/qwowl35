    use super::{ThinkBudget, REOPEN_THINK_BUDGET, THINK_RAMP_MAX_BIAS};

    const THINK: u32 = 1; // `<think>`
    const END: u32 = 2; // `</think>`
    const WORD: u32 = 10; // an ordinary reasoning token
    const NL: u32 = 20; // a token that ends with a newline (sentence boundary)
    const MSG1: u32 = 100; // wrap-up message token
    const MSG2: u32 = 101;

    fn tracker(budget: u32, message: Vec<u32>) -> ThinkBudget {
        ThinkBudget::new(Some(END), Some(THINK), true, Some(budget), message)
    }

    /// Mirrors the decode loop: drain a forced token if any, else sample (here:
    /// the caller-provided `sampled`) and run it through `observe`.
    fn step(tb: &mut ThinkBudget, sampled: u32, ends_with_newline: impl FnOnce(u32) -> bool) -> u32 {
        if let Some(forced) = tb.forced_next() {
            forced
        } else {
            tb.observe(sampled, ends_with_newline)
        }
    }

    #[test]
    fn passthrough_when_thinking_off() {
        // Thinking off => disabled => every token, including the markers, passes.
        let mut tb = ThinkBudget::new(Some(END), Some(THINK), false, Some(3), vec![]);
        for tok in [WORD, THINK, WORD, END, WORD, WORD, WORD, WORD] {
            assert_eq!(step(&mut tb, tok, |_| false), tok);
        }
    }

    #[test]
    fn hard_ceiling_substitutes_close() {
        // budget 3 => ceiling 3 + max(3/10, 1) = 4; empty message => bare close.
        let mut tb = tracker(3, vec![]);
        assert_eq!(step(&mut tb, WORD, |_| false), WORD); // tokens = 1
        assert_eq!(step(&mut tb, WORD, |_| false), WORD); // tokens = 2
        assert_eq!(step(&mut tb, WORD, |_| false), WORD); // tokens = 3 (ramping)
        assert_eq!(step(&mut tb, WORD, |_| false), END); // tokens = 4 -> forced
    }

    #[test]
    fn graceful_close_on_sentence_boundary() {
        // Past the soft budget, the newline is let through and the close lands on
        // the next step, preserving the boundary (empty message => bare close).
        let mut tb = tracker(3, vec![]);
        assert_eq!(step(&mut tb, WORD, |_| false), WORD);
        assert_eq!(step(&mut tb, WORD, |_| false), WORD);
        assert_eq!(step(&mut tb, NL, |t| t == NL), NL); // newline let through
        assert_eq!(step(&mut tb, WORD, |_| false), END); // forced close next step
    }

    #[test]
    fn message_injection_drains_then_resumes() {
        // At the ceiling the wrap-up message + `</think>` is forced one token per
        // step, then normal sampling resumes.
        let mut tb = tracker(3, vec![MSG1, MSG2]);
        assert_eq!(step(&mut tb, WORD, |_| false), WORD); // 1
        assert_eq!(step(&mut tb, WORD, |_| false), WORD); // 2
        assert_eq!(step(&mut tb, WORD, |_| false), WORD); // 3 (ramping)
        assert_eq!(step(&mut tb, WORD, |_| false), MSG1); // 4 -> begin force
        assert_eq!(step(&mut tb, WORD, |_| false), MSG2); // drain message
        assert_eq!(step(&mut tb, WORD, |_| false), END); //  drain </think>
        assert_eq!(step(&mut tb, WORD, |_| false), WORD); // forcing done -> resume
    }

    #[test]
    fn graceful_close_injects_message() {
        // The graceful (sentence-boundary) path also forces the wrap-up message.
        let mut tb = tracker(3, vec![MSG1]);
        assert_eq!(step(&mut tb, WORD, |_| false), WORD);
        assert_eq!(step(&mut tb, WORD, |_| false), WORD);
        assert_eq!(step(&mut tb, NL, |t| t == NL), NL); // boundary let through
        assert_eq!(step(&mut tb, WORD, |_| false), MSG1); // message begins
        assert_eq!(step(&mut tb, WORD, |_| false), END); //  then </think>
        assert_eq!(step(&mut tb, WORD, |_| false), WORD); // resume
    }

    #[test]
    fn reopen_uses_small_budget_and_bare_close() {
        // First block has a large budget + a message and closes naturally. The
        // reopened block must be bounded by the small REOPEN_THINK_BUDGET (not the
        // large full budget) and close BARE (no message), so reopen-spam is cheap.
        let mut tb = tracker(1000, vec![MSG1, MSG2]);
        assert_eq!(step(&mut tb, WORD, |_| false), WORD);
        assert_eq!(step(&mut tb, END, |_| false), END); // natural close of block 1

        assert_eq!(step(&mut tb, THINK, |_| false), THINK); // reopen -> small budget
        let ceiling = REOPEN_THINK_BUDGET + (REOPEN_THINK_BUDGET / 10).max(1);
        for _ in 0..(ceiling - 1) {
            assert_eq!(step(&mut tb, WORD, |_| false), WORD); // bounded by ~48, not 1000
        }
        // Force-closes at the small ceiling with a bare `</think>` (== END, not MSG1).
        assert_eq!(step(&mut tb, WORD, |_| false), END);
    }

    #[test]
    fn ramp_bias_rises_within_window() {
        // budget 20 => ceiling 22, ramp span 2. No bias while counting; a bias
        // toward `</think>` rises from 0 across the window; gone after the close.
        let mut tb = tracker(20, vec![]);
        for _ in 0..19 {
            assert_eq!(step(&mut tb, WORD, |_| false), WORD);
            assert_eq!(tb.bias(), None); // still counting
        }
        assert_eq!(step(&mut tb, WORD, |_| false), WORD); // tokens = 20 -> ramping
        assert_eq!(tb.bias(), Some((END, 0.0))); // progress 0 at the soft budget
        assert_eq!(step(&mut tb, WORD, |_| false), WORD); // tokens = 21
        let (id, amount) = tb.bias().expect("ramping");
        assert_eq!(id, END);
        assert!((amount - THINK_RAMP_MAX_BIAS * 0.5).abs() < 1e-6); // progress 0.5
        assert_eq!(step(&mut tb, WORD, |_| false), END); // tokens = 22 -> forced
        assert_eq!(tb.bias(), None); // forcing/done
    }
