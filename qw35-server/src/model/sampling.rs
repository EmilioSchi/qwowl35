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
    // Penalties touch only a handful of token ids (the seen/repeat windows), so
    // apply them to those ids directly instead of probing a hash set/map for
    // every one of the ~248K vocab entries — that per-entry hashing dominated
    // sampling (measured ~7 ms/token; see bench_sample_from_logits_full_vocab).
    // We patch a working copy, then do a single hashing-free pass to build the
    // candidate set.
    let mut adjusted: Vec<f32> = logits.to_vec();

    // OpenAI semantics: presence and frequency penalties consider only the
    // generated output (logit -= count * frequency + (count > 0) * presence).
    // Penalizing prompt tokens too would demote every identifier mentioned in
    // a long agentic prompt.
    if request.presence_penalty != 0.0 || request.frequency_penalty != 0.0 {
        let mut completion_counts: HashMap<u32, f32> = HashMap::new();
        for &token in &seen[prompt_len.min(seen.len())..] {
            *completion_counts.entry(token).or_insert(0.0) += 1.0;
        }
        for (id, count) in completion_counts {
            if let Some(slot) = adjusted.get_mut(id as usize) {
                if slot.is_finite() {
                    *slot -= count * request.frequency_penalty + request.presence_penalty;
                }
            }
        }
    }

    // The repetition penalty is the llama.cpp-style knob and spans the full
    // repeat_last_n window (prompt tokens included).
    if request.repetition_penalty != 1.0 {
        for id in repeat_penalty_token_set(seen, request.repeat_last_n) {
            if let Some(slot) = adjusted.get_mut(id as usize) {
                if slot.is_finite() {
                    if *slot < 0.0 {
                        *slot *= request.repetition_penalty;
                    } else {
                        *slot /= request.repetition_penalty;
                    }
                }
            }
        }
    }

    let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(adjusted.len());
    for (id, &logit) in adjusted.iter().enumerate() {
        if logit.is_finite() {
            candidates.push((id as u32, logit));
        }
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
#[path = "../tests/model_sampling.rs"]
mod tests;
