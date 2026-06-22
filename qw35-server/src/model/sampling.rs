//! Token sampling from logits: presence/frequency/repetition penalties,
//! top-k/top-p/min-p filtering, and the temperature-weighted draw.

use super::GenerateRequest;
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) fn sample_from_logits(
    logits: &[f32],
    request: &GenerateRequest,
    seen: &[u32],
    prompt_len: usize,
) -> u32 {
    // OpenAI semantics: presence and frequency penalties consider only the
    // generated output (logit -= count * frequency + (count > 0) * presence).
    // Penalizing prompt tokens too would demote every identifier mentioned in
    // a long agentic prompt. The repetition penalty intentionally differs: it
    // is the llama.cpp-style knob and spans the full repeat_last_n window.
    let mut completion_counts: HashMap<u32, f32> = HashMap::new();
    for &token in &seen[prompt_len.min(seen.len())..] {
        *completion_counts.entry(token).or_insert(0.0) += 1.0;
    }
    let repetition_ids = repeat_penalty_token_set(seen, request.repeat_last_n);
    let mut candidates = Vec::with_capacity(logits.len());
    for (id, &logit) in logits.iter().enumerate() {
        if !logit.is_finite() {
            continue;
        }
        let id = id as u32;
        let mut adjusted = logit;
        if let Some(&count) = completion_counts.get(&id) {
            adjusted -= count * request.frequency_penalty + request.presence_penalty;
        }
        if repetition_ids.contains(&id) && request.repetition_penalty != 1.0 {
            if adjusted < 0.0 {
                adjusted *= request.repetition_penalty;
            } else {
                adjusted /= request.repetition_penalty;
            }
        }
        candidates.push((id, adjusted));
    }

    if candidates.is_empty() {
        return 0;
    }
    // Select-then-sort: with top_k set (default 20), a full descending sort
    // of the 248320-entry vocabulary costs more than the rest of sampling
    // combined. The comparator is a total order, so the top-k set matches
    // what a full sort would have kept.
    let descending = |a: &(u32, f32), b: &(u32, f32)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    };
    if request.top_k > 0 && (request.top_k as usize) < candidates.len() {
        let k = request.top_k as usize;
        candidates.select_nth_unstable_by(k - 1, descending);
        candidates.truncate(k);
    }
    candidates.sort_unstable_by(descending);

    let temperature = request.temperature.max(1.0e-6);
    let max_logit = candidates[0].1;
    let mut weighted = Vec::with_capacity(candidates.len());
    let mut total = 0.0f64;
    for (id, logit) in candidates {
        let weight = ((logit - max_logit) / temperature).exp() as f64;
        if weight.is_finite() && weight > 0.0 {
            total += weight;
            weighted.push((id, weight));
        }
    }
    if weighted.is_empty() || total <= 0.0 {
        return 0;
    }

    let top_p = request.top_p.clamp(0.0, 1.0);
    let min_p = request.min_p.clamp(0.0, 1.0);
    let max_prob = weighted[0].1 / total;
    let mut filtered = Vec::with_capacity(weighted.len());
    let mut cumulative = 0.0f64;
    for (id, weight) in weighted {
        let prob = weight / total;
        if min_p > 0.0 && prob < max_prob * f64::from(min_p) {
            continue;
        }
        filtered.push((id, weight));
        cumulative += prob;
        if top_p > 0.0 && cumulative >= f64::from(top_p) {
            break;
        }
    }
    if filtered.is_empty() {
        return 0;
    }

    let filtered_total: f64 = filtered.iter().map(|(_, weight)| *weight).sum();
    let mut draw = sample_unit_interval() * filtered_total;
    for (id, weight) in filtered {
        if draw <= weight {
            return id;
        }
        draw -= weight;
    }
    0
}

fn repeat_penalty_token_set(seen: &[u32], repeat_last_n: i32) -> HashSet<u32> {
    let window = match repeat_last_n {
        -1 => seen,
        0 => &seen[0..0],
        n if n > 0 => {
            let start = seen.len().saturating_sub(n as usize);
            &seen[start..]
        }
        _ => &seen[0..0],
    };
    window.iter().copied().collect()
}

fn sample_unit_interval() -> f64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let mut state = nanos ^ ((std::process::id() as u64) << 32);
    state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let value = state >> 11;
    (value as f64) * (1.0 / ((1u64 << 53) as f64))
}

#[cfg(test)]
mod tests {
    use super::sample_from_logits;
    use crate::model::{GenerateRequest, TokenLimit, DEFAULT_MODEL_ID};

    fn sampler_request(presence: f32, frequency: f32, repetition: f32) -> GenerateRequest {
        GenerateRequest {
            model: DEFAULT_MODEL_ID.to_string(),
            messages: Vec::new(),
            max_tokens: TokenLimit::Fixed(1),
            temperature: 1.0,
            top_p: 1.0,
            // top_k 1 makes sampling deterministic: the argmax of the
            // penalty-adjusted logits is the only surviving candidate.
            top_k: 1,
            min_p: 0.0,
            presence_penalty: presence,
            frequency_penalty: frequency,
            repetition_penalty: repetition,
            repeat_last_n: -1,
            enable_thinking: false,
            preserve_thinking: false,
            thinking_budget: None,
            reasoning_budget_message: None,
            ignore_eos: false,
            stop_sequences: Vec::new(),
            emit_reasoning: false,
        }
    }

    #[test]
    fn presence_penalty_ignores_prompt_only_tokens() {
        // Token 0 appears only in the prompt; OpenAI semantics leave it alone.
        let logits = [2.0, 1.0, 0.5];
        let request = sampler_request(10.0, 0.0, 1.0);
        assert_eq!(sample_from_logits(&logits, &request, &[0], 1), 0);
    }

    #[test]
    fn presence_penalty_applies_once_to_completion_tokens() {
        let logits = [0.0, 2.0, 1.0];
        let request = sampler_request(1.5, 0.0, 1.0);
        // Token 1 generated once: 2.0 - 1.5 = 0.5 < 1.0 -> token 2 wins.
        assert_eq!(sample_from_logits(&logits, &request, &[0, 1], 1), 2);
        // Repeating it does not deepen the presence penalty.
        let single = sample_from_logits(&logits, &request, &[0, 1], 1);
        let triple = sample_from_logits(&logits, &request, &[0, 1, 1, 1], 1);
        assert_eq!(single, triple);
    }

    #[test]
    fn frequency_penalty_scales_with_occurrence_count() {
        let logits = [0.0, 2.0, 1.0];
        let request = sampler_request(0.0, 0.4, 1.0);
        // One occurrence: 2.0 - 0.4 = 1.6 still beats 1.0.
        assert_eq!(sample_from_logits(&logits, &request, &[0, 1], 1), 1);
        // Three occurrences: 2.0 - 1.2 = 0.8 loses to 1.0.
        assert_eq!(sample_from_logits(&logits, &request, &[0, 1, 1, 1], 1), 2);
    }

    #[test]
    fn repetition_penalty_still_covers_prompt_tokens() {
        let logits = [2.0, 1.5, 0.1];
        let request = sampler_request(0.0, 0.0, 2.0);
        // Token 0 is prompt-only but the llama.cpp-style repetition penalty
        // spans the full window: 2.0 / 2.0 = 1.0 < 1.5 -> token 1 wins.
        assert_eq!(sample_from_logits(&logits, &request, &[0], 1), 1);
    }
}
