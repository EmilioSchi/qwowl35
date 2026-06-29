//! Stop-sequence watching over the visible text stream.

/// Watches the visible text stream for stop sequences across delta
/// boundaries. Text that could still grow into a match is held back until it
/// is resolved either way; on a match the sequence itself is discarded
/// (OpenAI semantics: the stop string is not included in the output).
#[derive(Debug)]
pub(super) struct StopSequenceWatcher {
    sequences: Vec<String>,
    pending: String,
}

impl StopSequenceWatcher {
    pub(super) fn new(sequences: &[String]) -> Self {
        Self {
            sequences: sequences
                .iter()
                .filter(|sequence| !sequence.is_empty())
                .cloned()
                .collect(),
            pending: String::new(),
        }
    }

    /// Returns the text safe to emit and whether a stop sequence completed.
    pub(super) fn feed(&mut self, delta: &str) -> (String, bool) {
        if self.sequences.is_empty() {
            return (delta.to_string(), false);
        }
        self.pending.push_str(delta);
        if let Some(idx) = earliest_stop_match(&self.pending, &self.sequences) {
            let visible = self.pending[..idx].to_string();
            self.pending.clear();
            return (visible, true);
        }

        // Hold back the longest tail that is still a prefix of some sequence.
        let max_hold = self
            .sequences
            .iter()
            .map(|sequence| sequence.len())
            .max()
            .unwrap_or(1)
            - 1;
        let mut hold = 0;
        for len in (1..=max_hold.min(self.pending.len())).rev() {
            let split = self.pending.len() - len;
            if !self.pending.is_char_boundary(split) {
                continue;
            }
            let tail = &self.pending[split..];
            if self
                .sequences
                .iter()
                .any(|sequence| sequence.starts_with(tail))
            {
                hold = len;
                break;
            }
        }
        let emit_to = self.pending.len() - hold;
        let visible = self.pending[..emit_to].to_string();
        self.pending.drain(..emit_to);
        (visible, false)
    }

    /// Flushes held-back text once the stream ends without a match.
    pub(super) fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

/// Byte index of the earliest occurrence of any stop sequence in `text`.
pub(super) fn earliest_stop_match(text: &str, sequences: &[String]) -> Option<usize> {
    sequences
        .iter()
        .filter(|sequence| !sequence.is_empty())
        .filter_map(|sequence| text.find(sequence.as_str()))
        .min()
}

#[cfg(test)]
#[path = "../tests/model_stop_sequence.rs"]
mod tests;
