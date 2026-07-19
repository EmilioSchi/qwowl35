    use super::{plan_session_reuse, CheckpointStack, SessionReuse};

    fn stack_with_cap(cap: usize, prefixes: &[&[u32]]) -> CheckpointStack {
        let mut stack = CheckpointStack::new(cap);
        for prefix in prefixes {
            stack.save(prefix.to_vec(), Box::new([]));
        }
        stack
    }

    fn depths(stack: &CheckpointStack) -> Vec<usize> {
        let mut lens: Vec<usize> = stack.entries.iter().map(|e| e.tokens.len()).collect();
        lens.sort_unstable();
        lens
    }

    #[test]
    fn planner_picks_deepest_matching_checkpoint() {
        // Boundaries at 2, 4 and 6 along one lineage; the prompt shares the
        // first 5 tokens, so the depth-4 entry is the best rewind point.
        let stack = stack_with_cap(8, &[&[1, 2], &[1, 2, 3, 4], &[1, 2, 3, 4, 5, 6]]);
        let prompt = vec![1, 2, 3, 4, 5, 9, 9];
        assert_eq!(
            plan_session_reuse(&[], &stack, &prompt),
            SessionReuse::Checkpoint { index: 1, len: 4 }
        );
    }

    #[test]
    fn planner_prefers_live_extend_over_checkpoint() {
        let stack = stack_with_cap(8, &[&[1, 2], &[1, 2, 3, 4]]);
        let evaluated = vec![1, 2, 3, 4, 5];
        let prompt = vec![1, 2, 3, 4, 5, 6, 7];
        assert_eq!(
            plan_session_reuse(&evaluated, &stack, &prompt),
            SessionReuse::Extend(5)
        );
    }

    #[test]
    fn deepest_entry_must_leave_one_token_to_evaluate() {
        // The depth-6 entry covers the whole prompt: unusable (the output
        // head needs at least one token); fall back to depth 4.
        let stack = stack_with_cap(8, &[&[1, 2, 3, 4], &[1, 2, 3, 4, 5, 6]]);
        let prompt = vec![1, 2, 3, 4, 5, 6];
        assert_eq!(
            plan_session_reuse(&[], &stack, &prompt),
            SessionReuse::Checkpoint { index: 0, len: 4 }
        );
    }

    #[test]
    fn restore_drops_deeper_entries_and_keeps_shallower() {
        // Restoring at depth 4 will re-prefill positions >= 4: the depth-6
        // entry's KV rows are about to be overwritten, the depth-2 entry's
        // rows stay untouched.
        let mut stack = stack_with_cap(8, &[&[1, 2], &[1, 2, 3, 4], &[1, 2, 3, 4, 5, 6]]);
        let prompt = vec![1, 2, 3, 4, 9, 9];
        let SessionReuse::Checkpoint { index, len } = plan_session_reuse(&[], &stack, &prompt)
        else {
            panic!("expected a checkpoint plan");
        };
        assert_eq!(len, 4);
        stack.mark_restored(index);
        assert_eq!(depths(&stack), vec![2, 4]);
    }

    #[test]
    fn save_refreshes_identical_boundary_instead_of_duplicating() {
        let mut stack = stack_with_cap(8, &[&[1, 2, 3]]);
        stack.save(vec![1, 2, 3], Box::new([]));
        stack.save(vec![1, 2, 3], Box::new([]));
        assert_eq!(stack.entries.len(), 1);
    }

    #[test]
    fn eviction_is_lru_and_never_takes_the_deepest_or_just_saved() {
        // Fill to cap with boundaries 1..=3, then touch depth 1 (restore) so
        // depth 2 becomes the LRU. Saving depth 4 must evict depth 2.
        let mut stack = stack_with_cap(3, &[&[1], &[1, 2], &[1, 2, 3]]);
        let prompt = vec![1, 9];
        let SessionReuse::Checkpoint { index, .. } = plan_session_reuse(&[], &stack, &prompt)
        else {
            panic!("expected a checkpoint plan");
        };
        // mark_restored would drop deeper entries; refresh the LRU stamp
        // directly through save's dedup path instead (same clock bump).
        stack.save(stack.entries[index].tokens.clone(), Box::new([]));
        stack.save(vec![1, 2, 3, 4], Box::new([]));
        assert_eq!(depths(&stack), vec![1, 3, 4]);
    }

    #[test]
    fn cap_one_keeps_only_the_latest_save() {
        // Legacy single-checkpoint behavior: each save replaces the slot.
        let mut stack = stack_with_cap(1, &[&[1, 2]]);
        stack.save(vec![1, 2, 3, 4], Box::new([]));
        assert_eq!(depths(&stack), vec![4]);
        // Even when the new save is shallower than the old entry, the fresh
        // save wins (the old deepest is the fallback victim).
        stack.save(vec![1, 2, 3], Box::new([]));
        assert_eq!(depths(&stack), vec![3]);
    }

    #[test]
    fn cap_zero_disables_snapshots() {
        let mut stack = stack_with_cap(0, &[]);
        stack.save(vec![1, 2, 3], Box::new([]));
        assert_eq!(stack.entries.len(), 0);
        assert_eq!(
            plan_session_reuse(&[], &stack, &[1, 2, 3, 4]),
            SessionReuse::Reset
        );
    }

    #[test]
    fn clear_forgets_everything() {
        let mut stack = stack_with_cap(8, &[&[1, 2], &[1, 2, 3, 4]]);
        stack.clear();
        assert_eq!(stack.entries.len(), 0);
        assert_eq!(
            plan_session_reuse(&[], &stack, &[1, 2, 3]),
            SessionReuse::Reset
        );
    }
