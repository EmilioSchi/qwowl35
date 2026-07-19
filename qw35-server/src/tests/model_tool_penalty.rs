    use super::ToolCallPenaltyGuard;

    const OPEN: u32 = 10;
    const CLOSE: u32 = 11;

    #[test]
    fn activates_on_open_and_releases_on_close() {
        let mut guard = ToolCallPenaltyGuard::new(Some(OPEN), Some(CLOSE));
        assert!(!guard.active());
        guard.observe(42);
        assert!(!guard.active());
        guard.observe(OPEN);
        assert!(guard.active());
        guard.observe(42); // body tokens keep it armed
        assert!(guard.active());
        guard.observe(CLOSE);
        assert!(!guard.active());
    }

    #[test]
    fn stray_close_without_open_is_a_no_op() {
        let mut guard = ToolCallPenaltyGuard::new(Some(OPEN), Some(CLOSE));
        guard.observe(CLOSE);
        assert!(!guard.active());
        guard.observe(42);
        assert!(!guard.active());
    }

    #[test]
    fn rearms_for_a_second_tool_call() {
        let mut guard = ToolCallPenaltyGuard::new(Some(OPEN), Some(CLOSE));
        guard.observe(OPEN);
        guard.observe(CLOSE);
        assert!(!guard.active());
        guard.observe(OPEN);
        assert!(guard.active());
    }

    #[test]
    fn arm_activates_without_observing_the_open_token() {
        // A forced tool-call prefix is prefilled prompt bytes: the decode
        // loop never observes the open token, so the guard is armed
        // explicitly; the model's own close still releases it.
        let mut guard = ToolCallPenaltyGuard::new(Some(OPEN), Some(CLOSE));
        guard.arm();
        assert!(guard.active());
        guard.observe(42);
        assert!(guard.active());
        guard.observe(CLOSE);
        assert!(!guard.active());

        // A disabled guard (missing ids) refuses to arm.
        let mut disabled = ToolCallPenaltyGuard::new(Some(OPEN), None);
        disabled.arm();
        assert!(!disabled.active());
    }

    #[test]
    fn missing_either_token_id_disables_the_guard() {
        // Without a close id an open could latch active() forever; the guard
        // must refuse to arm at all.
        let mut open_only = ToolCallPenaltyGuard::new(Some(OPEN), None);
        open_only.observe(OPEN);
        assert!(!open_only.active());

        let mut close_only = ToolCallPenaltyGuard::new(None, Some(CLOSE));
        close_only.observe(OPEN);
        close_only.observe(CLOSE);
        assert!(!close_only.active());
    }
