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
    #[ignore = "diagnostic: CPU cost of sample_from_logits over the full 248K vocab"]
    fn bench_sample_from_logits_full_vocab() {
        let vocab = 248_320usize;
        let mut logits = vec![0.0f32; vocab];
        for (i, l) in logits.iter_mut().enumerate() {
            *l = ((i as f32) * 0.000_123).sin() * 5.0;
        }
        let mut request = sampler_request(0.0, 0.0, 1.1);
        request.temperature = 0.6;
        request.top_p = 0.95;
        request.top_k = 20;
        request.repeat_last_n = 64;
        // Realistic mid-session history (~2000 generated tokens).
        let seen: Vec<u32> = (0..2000u32).map(|i| (i.wrapping_mul(37)) % vocab as u32).collect();
        let prompt_len = 1900usize;
        let iters = 300u32;
        let start = std::time::Instant::now();
        let mut acc = 0u64;
        for _ in 0..iters {
            acc += u64::from(sample_from_logits(&logits, &request, &seen, prompt_len));
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
        eprintln!(
            "sample_from_logits: {ms:.3} ms/token over {vocab} vocab ({iters} iters) [sink={acc}]"
        );
    }

    #[test]
    fn repetition_penalty_still_covers_prompt_tokens() {
        let logits = [2.0, 1.5, 0.1];
        let request = sampler_request(0.0, 0.0, 2.0);
        // Token 0 is prompt-only but the llama.cpp-style repetition penalty
        // spans the full window: 2.0 / 2.0 = 1.0 < 1.5 -> token 1 wins.
        assert_eq!(sample_from_logits(&logits, &request, &[0], 1), 1);
    }
