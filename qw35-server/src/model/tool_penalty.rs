//! Decode-loop suspension of sampling penalties inside tool-call blocks. See
//! [`ToolCallPenaltyGuard`].

/// Tracks whether decoding is inside a `<tool_call>`…`</tool_call>` block so the
/// sampler can suspend the presence/frequency/repetition penalties there. A
/// tool-call body is a verbatim payload (often a whole file in a heredoc):
/// repetition in it is legitimate, and the unwindowed presence penalty
/// otherwise accumulates against every token the payload needs until the
/// closing XML derails (observed as "corrupted tool calls" on long bash
/// commands in the instruct presets, which pair presence 1.5 with temp 1.0).
///
/// `<tool_call>` / `</tool_call>` are single special tokens, so detection is
/// one integer compare per token. Limitation: a rootless `<function=…>` call
/// (accepted by the parser as a fallback) is plain BPE text and does not arm
/// the guard; the system prompt mandates the `<tool_call>` form.
#[derive(Debug, Clone)]
pub(super) struct ToolCallPenaltyGuard {
    open: Option<u32>,
    close: Option<u32>,
    active: bool,
}

impl ToolCallPenaltyGuard {
    pub(super) fn new(open: Option<u32>, close: Option<u32>) -> Self {
        // Both ids are required: without the close id the guard could latch
        // active forever and silently disable penalties for the whole answer.
        let enabled = open.is_some() && close.is_some();
        Self {
            open: open.filter(|_| enabled),
            close: close.filter(|_| enabled),
            active: false,
        }
    }

    /// Whether the NEXT token is sampled inside a tool-call body (i.e. the
    /// opening tag has been emitted and the closing one has not).
    pub(super) fn active(&self) -> bool {
        self.active
    }

    /// Arm the guard as if the opening tag had been emitted — used when the
    /// `<tool_call>` opening is a forced prompt prefix (prefilled, never
    /// sampled), so the decode loop never observes its token id. The model's
    /// own `</tool_call>` disarms it as usual.
    pub(super) fn arm(&mut self) {
        if self.open.is_some() {
            self.active = true;
        }
    }

    /// Observe an emitted token (sampled or forced) and update the state.
    pub(super) fn observe(&mut self, token: u32) {
        if Some(token) == self.open {
            self.active = true;
        } else if Some(token) == self.close {
            self.active = false;
        }
    }
}

#[cfg(test)]
#[path = "../tests/model_tool_penalty.rs"]
mod tool_penalty_guard_tests;
