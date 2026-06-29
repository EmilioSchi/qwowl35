//! Decode-loop enforcement of the thinking-token budget. See [`ThinkBudget`].

/// Small budget granted to a `<think>` block the model REOPENS after a close —
/// enough for a brief course-correction, not a second full reasoning pass. The
/// first block gets the caller's full effort budget; reopens get this.
const REOPEN_THINK_BUDGET: u32 = 48;

/// Maximum additive logit bias applied to `</think>` at the hard ceiling while
/// ramping; scales linearly from 0 at the soft budget. Nudges the model to close
/// reasoning on its own (in the sampling regime) before it is force-closed.
const THINK_RAMP_MAX_BIAS: f32 = 8.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BudgetState {
    /// No enforcement: thinking off, or no `</think>` token / no budget.
    Disabled,
    /// Inside a think block, still under the soft budget.
    Counting,
    /// Past the soft budget: biasing `</think>` and waiting for a sentence
    /// boundary (or the hard ceiling) to force the close.
    Ramping,
    /// Draining a forced close sequence (wrap-up message + `</think>`).
    Forcing,
    /// Thinking closed; watching for a reopened `<think>`.
    Done,
}

/// Decode-loop enforcement of the thinking-token budget, modeled on llama.cpp's
/// `common_reasoning_budget`. Once the soft budget is reached it (1) ramps a
/// logit bias toward `</think>` so the model tends to stop on its own, then
/// (2) at the next sentence boundary — or the hard ceiling (soft + ~10% grace) —
/// forces a pre-tokenized wrap-up message followed by `</think>`, which the model
/// conditions on for its answer. The first `<think>` block gets the full effort
/// budget; a block the model REOPENS afterwards re-arms with a small budget
/// (`REOPEN_THINK_BUDGET`) and force-closes with a bare `</think>` (no message),
/// so reopen-spam stays cheap while a brief legitimate second thought is allowed.
/// A disabled tracker (thinking off / no `</think>` token / no budget) is a
/// pass-through no-op.
#[derive(Debug, Clone)]
pub(super) struct ThinkBudget {
    end_think_id: Option<u32>,
    think_id: Option<u32>,
    /// Full effort budget for the first block.
    full_budget: u32,
    /// Active block's soft target, hard ceiling, and tokens spent in it.
    budget: u32,
    ceiling: u32,
    tokens: u32,
    state: BudgetState,
    /// Whether the active block is the first one (uses the wrap-up message on a
    /// forced close); reopened blocks close bare.
    first_block: bool,
    /// Pre-tokenized `message ++ [</think>]` for the first block's forced close.
    forced_message_close: Vec<u32>,
    /// The forced close sequence currently being drained, and our position in it.
    forced: Vec<u32>,
    force_pos: usize,
    /// A sentence boundary was let through last step; begin forcing on the next
    /// step (so the model's own closing newline is preserved before the wrap-up).
    pending_force: bool,
}

impl ThinkBudget {
    pub(super) fn new(
        end_think_id: Option<u32>,
        think_id: Option<u32>,
        enable_thinking: bool,
        thinking_budget: Option<u32>,
        message_tokens: Vec<u32>,
    ) -> Self {
        let budget = thinking_budget.filter(|_| enable_thinking && end_think_id.is_some());
        let (state, full_budget, soft, ceiling) = match budget {
            Some(b) => (BudgetState::Counting, b, b, Self::ceiling_of(b)),
            None => (BudgetState::Disabled, 0, 0, 0),
        };
        let forced_message_close = match end_think_id {
            Some(end) => {
                let mut v = message_tokens;
                v.push(end);
                v
            }
            None => Vec::new(),
        };
        Self {
            end_think_id,
            think_id,
            full_budget,
            budget: soft,
            ceiling,
            tokens: 0,
            state,
            first_block: true,
            forced_message_close,
            forced: Vec::new(),
            force_pos: 0,
            pending_force: false,
        }
    }

    fn ceiling_of(budget: u32) -> u32 {
        budget.saturating_add((budget / 10).max(1))
    }

    /// While draining a forced close sequence, returns the next forced token (the
    /// caller skips sampling for this step). `None` when not forcing.
    pub(super) fn forced_next(&mut self) -> Option<u32> {
        if self.pending_force {
            // A sentence boundary was let through last step; begin the forced
            // close now, returning its first token (keeps that closing newline).
            self.pending_force = false;
            return Some(self.begin_force());
        }
        if self.state != BudgetState::Forcing {
            return None;
        }
        let tok = self.forced.get(self.force_pos).copied()?;
        self.force_pos += 1;
        if self.force_pos >= self.forced.len() {
            self.state = BudgetState::Done;
            self.forced.clear();
            self.force_pos = 0;
        }
        Some(tok)
    }

    /// The `(</think> id, additive logit bias)` to apply before selection while
    /// ramping; `None` otherwise. The bias scales linearly from 0 at the soft
    /// budget to `THINK_RAMP_MAX_BIAS` at the hard ceiling.
    pub(super) fn bias(&self) -> Option<(u32, f32)> {
        if self.state != BudgetState::Ramping {
            return None;
        }
        let end = self.end_think_id?;
        let span = self.ceiling.saturating_sub(self.budget).max(1) as f32;
        let progress = (self.tokens.saturating_sub(self.budget) as f32 / span).clamp(0.0, 1.0);
        Some((end, THINK_RAMP_MAX_BIAS * progress))
    }

    /// Post-selection bookkeeping for the freshly sampled token. Returns the token
    /// to actually emit — `sampled`, or the first token of a forced close sequence
    /// when the budget fires. `ends_with_newline` is consulted only at the soft
    /// boundary, so the caller can defer the tokenizer lookup.
    pub(super) fn observe(&mut self, sampled: u32, ends_with_newline: impl FnOnce(u32) -> bool) -> u32 {
        match self.state {
            BudgetState::Disabled | BudgetState::Forcing => sampled,
            BudgetState::Counting | BudgetState::Ramping => {
                if Some(sampled) == self.end_think_id {
                    // Natural (or ramp-induced) close: no forced message.
                    self.state = BudgetState::Done;
                    return sampled;
                }
                self.tokens += 1;
                if self.tokens >= self.ceiling {
                    // Hard ceiling: force the close now, regardless of boundary.
                    return self.begin_force();
                }
                if self.tokens >= self.budget {
                    if self.state == BudgetState::Counting {
                        self.state = BudgetState::Ramping;
                    }
                    if ends_with_newline(sampled) {
                        // Graceful: let this newline through, then begin the
                        // forced wrap-up on the next step (preserves the boundary
                        // before the hard ceiling).
                        self.pending_force = true;
                    }
                }
                sampled
            }
            BudgetState::Done => {
                if Some(sampled) == self.think_id {
                    // Reopened reasoning: re-arm with the small reopen budget and
                    // mark the block so its forced close is bare (no message).
                    self.first_block = false;
                    self.budget = REOPEN_THINK_BUDGET.min(self.full_budget.max(1));
                    self.ceiling = Self::ceiling_of(self.budget);
                    self.tokens = 0;
                    self.state = BudgetState::Counting;
                }
                sampled
            }
        }
    }

    /// Transition to FORCING and return the first forced token. The first block
    /// forces the wrap-up message + `</think>`; reopened blocks force a bare
    /// `</think>` (no message spam).
    fn begin_force(&mut self) -> u32 {
        self.forced = if self.first_block {
            self.forced_message_close.clone()
        } else {
            self.end_think_id.into_iter().collect()
        };
        self.force_pos = 0;
        self.state = BudgetState::Forcing;
        self.forced_next()
            .expect("forced close sequence must be non-empty")
    }
}

#[cfg(test)]
#[path = "../tests/model_think_budget.rs"]
mod think_budget_tracker_tests;
