    use super::{
        plan_session_reuse, render_qwen35_chat_prompt, ChatTurn, Engine,
        EngineConfig, GenerateRequest, SessionReuse, TokenLimit, DEFAULT_MODEL_ID,
    };
    use std::path::PathBuf;

    #[test]
    fn session_reuse_extends_live_state() {
        // Prompt strictly extends the evaluated tokens.
        let evaluated = vec![1, 2, 3, 4];
        let prompt = vec![1, 2, 3, 4, 5, 6];
        assert_eq!(
            plan_session_reuse(&evaluated, None, &prompt),
            SessionReuse::Extend(4)
        );
    }

    #[test]
    fn session_reuse_always_leaves_one_token_to_evaluate() {
        // Identical prompt: the live state cannot be reused as-is because no
        // token would be left to produce logits, and the SSM state cannot
        // rewind. Falls back to the checkpoint only if it is short enough.
        let evaluated = vec![1, 2, 3, 4];
        let prompt = vec![1, 2, 3, 4];
        assert_eq!(
            plan_session_reuse(&evaluated, None, &prompt),
            SessionReuse::Reset
        );
        let checkpoint = vec![1, 2, 3];
        assert_eq!(
            plan_session_reuse(&evaluated, Some(&checkpoint), &prompt),
            SessionReuse::Checkpoint(3)
        );
        // A checkpoint covering the whole prompt is unusable too.
        let full = vec![1, 2, 3, 4];
        assert_eq!(
            plan_session_reuse(&evaluated, Some(&full), &prompt),
            SessionReuse::Reset
        );
    }

    #[test]
    fn session_reuse_rewinds_to_checkpoint_on_divergence() {
        // The previous reply re-tokenized differently: the prompt diverges
        // inside the generated region but still matches the prompt-boundary
        // checkpoint.
        let evaluated = vec![1, 2, 3, 9, 9];
        let checkpoint = vec![1, 2, 3];
        let prompt = vec![1, 2, 3, 7, 8, 9];
        assert_eq!(
            plan_session_reuse(&evaluated, Some(&checkpoint), &prompt),
            SessionReuse::Checkpoint(3)
        );
    }

    #[test]
    fn session_reuse_resets_for_unrelated_prompts() {
        let evaluated = vec![1, 2, 3, 4];
        let checkpoint = vec![1, 2, 3];
        let prompt = vec![5, 6, 7, 8];
        assert_eq!(
            plan_session_reuse(&evaluated, Some(&checkpoint), &prompt),
            SessionReuse::Reset
        );
        assert_eq!(plan_session_reuse(&[], None, &prompt), SessionReuse::Reset);
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "loads the real Qwen3.5 GGUF and runs native Metal prefill comparisons"]
    fn real_model_q4_q5_q6_k_tiled_prefill_matches_reference_first_token() {
        let _gpu = real_model_gpu_lock();
        // All chunks above 1 use the tiled simdgroup matmul; chunk 8 serves as
        // the reference against other chunk geometries.
        let reference = open_real_engine(8);
        let reference_logits = prefill_logits(&reference);
        let reference_token = argmax(&reference_logits);

        // The scalar path decodes the FFN with GF4 sidecar weights when one
        // is present, so cross-path logit parity only holds without it.
        if !reference.gf4_active() {
            let scalar = open_real_engine(1);
            assert_eq!(argmax(&prefill_logits(&scalar)), reference_token, "scalar");
        }

        for chunk in [9, 64] {
            let tiled = open_real_engine(chunk);
            let logits = prefill_logits(&tiled);
            assert_eq!(argmax(&logits), reference_token, "chunk {chunk}");

            if chunk == 64 {
                let max_abs = reference_logits
                    .iter()
                    .zip(&logits)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs < 2.0e-2,
                    "chunk {chunk} max logit absolute diff {max_abs}"
                );
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "loads the real Qwen3.5 GGUF and compares native Metal greedy decode paths"]
    fn real_model_tiled_prefill_decode_matches_batch_reference_first_tokens() {
        let _gpu = real_model_gpu_lock();
        let reference = open_real_engine(8);
        let reference_tokens = greedy_tokens_after_prefill(&reference, 4);

        let tiled = open_real_engine(64);
        let tiled_tokens = greedy_tokens_after_prefill(&tiled, 4);

        assert_eq!(tiled_tokens, reference_tokens);
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "loads the real Qwen3.5 GGUF and checks repeated prompt prefill after runtime reset"]
    fn real_model_repeated_prefill_reset_is_stable_across_requests() {
        let _gpu = real_model_gpu_lock();
        for prefill_chunk in [1, 64] {
            let engine = open_real_engine(prefill_chunk);
            let first = argmax(&prefill_logits(&engine));
            for run in 2..=3 {
                let next = argmax(&prefill_logits(&engine));
                assert_eq!(
                    next, first,
                    "prefill_chunk {prefill_chunk} changed first-token argmax on run {run}"
                );
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "loads the real Qwen3.5 GGUF and runs a three-turn greedy chat smoke test"]
    fn real_model_chat_ping_pong_greedy_smoke_three_turns() {
        let _gpu = real_model_gpu_lock();
        let engine = open_real_engine(1);
        let mut messages = Vec::new();

        for (idx, user) in [
            "Reply with one short greeting.",
            "Now name one color.",
            "Now name one shape.",
        ]
        .into_iter()
        .enumerate()
        {
            messages.push(ChatTurn {
                role: "user".to_string(),
                content: user.to_string(),
            });
            let generation = engine
                .generate(&GenerateRequest {
                    model: DEFAULT_MODEL_ID.to_string(),
                    messages: messages.clone(),
                    max_tokens: TokenLimit::Fixed(16),
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: 0,
                    min_p: 0.0,
                    presence_penalty: 0.0,
                    frequency_penalty: 0.0,
                    repetition_penalty: 1.0,
                    repeat_last_n: -1,
                    enable_thinking: false,
                    preserve_thinking: false,
                    thinking_budget: None,
                    reasoning_budget_message: None,
                    ignore_eos: false,
                    stop_sequences: Vec::new(),
                    emit_reasoning: false,
                })
                .unwrap_or_else(|err| panic!("turn {} failed: {err:?}", idx + 1));
            eprintln!("turn {} assistant: {:?}", idx + 1, generation.text);
            assert!(
                generation.completion_tokens > 0,
                "turn {} generated no completion tokens",
                idx + 1
            );
            messages.push(ChatTurn {
                role: "assistant".to_string(),
                content: generation.text,
            });
        }
    }

    /// The real-model tests each open one or more engines that stream the
    /// full 5.3 GiB model through the GPU. Running them concurrently on the
    /// 16 GB target produces memory-pressure artifacts, so they serialize on
    /// this lock.
    #[cfg(target_os = "macos")]
    static REAL_MODEL_GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[cfg(target_os = "macos")]
    fn real_model_gpu_lock() -> std::sync::MutexGuard<'static, ()> {
        REAL_MODEL_GPU_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "diagnostic: prints prefill logit divergence across chunk sizes"]
    fn real_model_prefill_chunk_divergence_report() {
        let _gpu = real_model_gpu_lock();
        let reference = open_engine_at(".gguf/debug-nogf4.gguf", 64);
        let reference_logits = prefill_logits(&reference);
        for chunk in [1u32, 2, 4, 8, 9, 16] {
            let engine = open_engine_at(".gguf/debug-nogf4.gguf", chunk);
            let logits = prefill_logits(&engine);
            let max_abs = reference_logits
                .iter()
                .zip(&logits)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "chunk {chunk:>2} vs 64: max_abs={max_abs:.5} argmax {} vs {}",
                argmax(&logits),
                argmax(&reference_logits)
            );
        }
    }

    /// Teacher-force `tokens` through one of the two evaluation paths and
    /// return the full logits vector at each probe position (a probe at p
    /// yields the logits that predict token p+1).
    #[cfg(target_os = "macos")]
    fn teacher_forced_probe_logits(
        engine: &Engine,
        tokens: &[u32],
        probes: &[usize],
        single_token_path: bool,
    ) -> Vec<Vec<f32>> {
        let mut session = engine
            .metal_runtime
            .as_ref()
            .expect("real engine should have native runtime")
            .lock()
            .expect("native runtime lock should not be poisoned");
        session.runtime.reset().expect("runtime should reset");

        let mut out = Vec::with_capacity(probes.len());
        let mut start = 0usize;
        for &probe in probes {
            assert!(probe >= start && probe < tokens.len());
            if single_token_path {
                // `pos` is the absolute sequence position passed to eval_token,
                // not just an index, so the range loop is intentional.
                #[allow(clippy::needless_range_loop)]
                for pos in start..=probe {
                    let mode = if pos == probe {
                        crate::metal::LogitsMode::Full
                    } else {
                        crate::metal::LogitsMode::None
                    };
                    session
                        .runtime
                        .eval_token(tokens[pos], pos as u32, mode)
                        .expect("scalar eval should succeed");
                }
            } else {
                let chunk_len = engine.prefill_chunk.max(2) as usize;
                let mut pos0 = start;
                for chunk in tokens[start..=probe].chunks(chunk_len) {
                    let mode = if pos0 + chunk.len() == probe + 1 {
                        crate::metal::LogitsMode::Full
                    } else {
                        crate::metal::LogitsMode::None
                    };
                    session
                        .runtime
                        .eval_prefill_chunk(chunk, pos0 as u32, mode)
                        .expect("chunked eval should succeed");
                    pos0 += chunk.len();
                }
            }
            session.runtime.sync().expect("sync should succeed");
            out.push(session.runtime.copy_logits().expect("logits should copy"));
            start = probe + 1;
        }
        out
    }

    /// Diagnostic for the repetition-loop investigation: teacher-force the
    /// captured looping transcript through the chunked prefill path and the
    /// single-token decode path on the base GGUF and report per-probe argmax
    /// flips and logit divergence. Both paths use the same base weights, so any
    /// flip is pure kernel-path divergence.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "diagnostic: prints decode-vs-prefill logit parity along a repetitive transcript"]
    fn real_model_decode_path_parity_report() {
        let _gpu = real_model_gpu_lock();

        let messages = vec![ChatTurn {
            role: "user".to_string(),
            content: "make a reverse count from 218 to 156, jumping by 2 one time and other by 4"
                .to_string(),
        }];
        let mut completion = String::from(
            "Here is the reverse count from **218** down to **156**, alternating between \
             subtracting **2** and subtracting **4** at each step:\n\n**218** (start)\n\
             -2 \u{2192} **216**\n-4 \u{2192} **212**\n-2 \u{2192} **210**\n-4 \u{2192} **206**\n\
             -2 \u{2192} **206** (Wait, let's re-calculate the sequence carefully)\n\n\
             Let's trace the logic:\n1.  Start: **218**\n2.  Subtract 2: **216**\n\
             3.  Subtract 4: **206**\n4.  Subtract 2: **204**\n5.  Subtract 4: **202**\n\
             6.  Subtract 4: **200**\n7.  Subtract 2: **198**\n8.  Subtract 4: **194**\n\
             9.  Subtract 2: **194** (Wait, 194 - 4 = 190? No, 202 - 4 = 190? No. ",
        );
        for _ in 0..28 {
            completion.push_str("202 - 4 = 198? No. ");
        }

        let engine = open_real_engine_with(32, 2048);
        let mut text = render_qwen35_chat_prompt(&messages, false, false);
        text.push_str(&completion);
        let tokens = engine
            .tokenizer
            .encode(&text, true)
            .expect("transcript should tokenize");
        let probes: Vec<usize> = (15..tokens.len()).step_by(16).collect();

        let batch = teacher_forced_probe_logits(&engine, &tokens, &probes, false);
        let single = teacher_forced_probe_logits(&engine, &tokens, &probes, true);

        let mut flips = 0usize;
        let mut worst_abs = 0.0f32;
        for (idx, probe) in probes.iter().enumerate() {
            let (a, b) = (&batch[idx], &single[idx]);
            let (ta, tb) = (argmax(a), argmax(b));
            let max_abs = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            worst_abs = worst_abs.max(max_abs);
            if ta != tb {
                flips += 1;
                eprintln!(
                    "probe pos {probe:>4}: ARGMAX FLIP batch={ta} ({:+.4}) single={tb} ({:+.4}) max_abs={max_abs:.4}",
                    a[ta as usize], b[tb as usize]
                );
            }
        }
        eprintln!(
            "probes={} argmax_flips={flips} worst_max_abs_logit_diff={worst_abs:.4}",
            probes.len()
        );
    }

    /// Shared (label, user prompt, fixed assistant continuation) corpus used by
    /// both activation calibration and the deterministic quality report, so the
    /// AWQ scales are fit on the same domains they are scored on.
    #[cfg(target_os = "macos")]
    fn sidecar_calibration_corpus() -> Vec<(&'static str, &'static str, &'static str)> {
        vec![
            (
                "math-code",
                "Write a python script solve_real_root.py that finds all real roots of ((1 - x) / 5)^4 - 16 = 0",
                "We solve ((1 - x)/5)^4 = 16, so (1 - x)/5 = ±2, giving 1 - x = ±10. \
                 Thus x = 1 - 10 = -9 or x = 1 + 10 = 11. The two real roots are -9 and 11.\n\n\
                 ```python\nimport numpy as np\n\ndef real_roots():\n    \
                 roots = [1 - 5 * 2, 1 + 5 * 2]\n    return sorted(roots)\n\n\
                 if __name__ == \"__main__\":\n    print(real_roots())\n```\n",
            ),
            (
                "reasoning",
                "Explain step by step why the sky appears blue during the day.",
                "Sunlight contains all colors. As it passes through the atmosphere, shorter \
                 wavelengths (blue) scatter far more than longer ones because Rayleigh scattering \
                 grows as one over wavelength to the fourth power. That scattered blue light reaches \
                 our eyes from every direction, so the daytime sky looks blue.",
            ),
            (
                "rust-code",
                "Write an iterative Rust function that returns the nth Fibonacci number.",
                "```rust\nfn fib(n: u64) -> u64 {\n    let (mut a, mut b) = (0u64, 1u64);\n    \
                 for _ in 0..n {\n        let next = a + b;\n        a = b;\n        b = next;\n    }\n    a\n}\n```\n\
                 It runs in O(n) time and O(1) space, returning 0 for n = 0.",
            ),
            (
                "multilingual",
                "Traduci in italiano: The cat sleeps on the warm roof while it rains.",
                "Il gatto dorme sul tetto caldo mentre piove. La frase mantiene il presente \
                 e descrive un'azione continua sotto la pioggia.",
            ),
            (
                "math-word",
                "A train travels 60 km in the first hour and 90 km in the next two hours. \
                 What is its average speed over the whole trip?",
                "Total distance is 60 + 90 = 150 km. Total time is 1 + 2 = 3 hours. \
                 Average speed = distance / time = 150 / 3 = 50 km/h. Note this is the \
                 time-weighted average, not the mean of 60 and 45.",
            ),
            (
                "py-debug",
                "Why does this raise IndexError: `xs = [1,2,3]; print(xs[len(xs)])` and how do I fix it?",
                "Python lists are zero-indexed, so valid indices are 0..len(xs)-1. `xs[len(xs)]` \
                 is `xs[3]`, one past the end, which raises IndexError. Use `xs[len(xs)-1]` or \
                 `xs[-1]` to get the last element, and guard with `if xs:` for empty lists.",
            ),
            (
                "prose",
                "Summarize the water cycle in three sentences.",
                "Water evaporates from oceans, lakes, and rivers, turning into vapor that rises \
                 into the atmosphere. As the vapor cools it condenses into clouds and eventually \
                 falls back to the surface as precipitation such as rain or snow. The water then \
                 collects in bodies of water or soaks into the ground, and the cycle repeats.",
            ),
            (
                "json",
                "Return a JSON object describing a book with title, author, year, and a list of two tags.",
                "```json\n{\n  \"title\": \"The Pragmatic Programmer\",\n  \
                 \"author\": \"Andrew Hunt and David Thomas\",\n  \"year\": 1999,\n  \
                 \"tags\": [\"software\", \"craftsmanship\"]\n}\n```\n",
            ),
            (
                "sql",
                "Write a SQL query to select the names of employees in the 'Sales' department earning more than 50000.",
                "```sql\nSELECT e.name\nFROM employees AS e\nJOIN departments AS d ON e.dept_id = d.id\n\
                 WHERE d.name = 'Sales' AND e.salary > 50000\nORDER BY e.name;\n```\n",
            ),
        ]
    }

    /// Drive the calibration corpus through the capture-routed decode path so
    /// the runtime accumulates per-channel FFN activation magnitudes. Set
    /// QW35_CAPTURE_ACT_OUT to the desired stats path (defaults next to the
    /// GGUF). Uses base weights (no GF4) so the stats reflect the reference.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "diagnostic: capture per-channel FFN activation stats for AWQ"]
    fn real_model_capture_activations() {
        let _gpu = real_model_gpu_lock();
        let out = std::env::var("QW35_CAPTURE_ACT_OUT")
            .unwrap_or_else(|_| ".gguf/act-stats.bin".to_string());
        std::env::set_var("QW35_CAPTURE_ACT_OUT", &out);

        let engine = open_real_engine_with(32, 4096);
        let mut total = 0usize;
        for (label, user, assistant) in sidecar_calibration_corpus() {
            let messages = vec![ChatTurn {
                role: "user".to_string(),
                content: user.to_string(),
            }];
            let mut text = render_qwen35_chat_prompt(&messages, false, false);
            text.push_str(assistant);
            let tokens = engine.tokenizer.encode(&text, true).expect("tokenize");
            let last = tokens.len() - 1;
            let _ = teacher_forced_probe_logits(&engine, &tokens, &[last], true);
            total += tokens.len();
            eprintln!("  captured [{label}] {} tokens", tokens.len());
        }
        drop(engine); // dealloc flushes the final (<16-token) window
        eprintln!("capture complete: ~{total} tokens -> {out}");
        assert!(
            std::path::Path::new(&out).exists(),
            "capture stats file should be written"
        );
    }

    /// KL(P || Q) in nats for two logit vectors, P = softmax(reference),
    /// Q = softmax(candidate). Numerically stable via max-subtraction.
    #[cfg(target_os = "macos")]
    fn kl_div_logits(reference: &[f32], candidate: &[f32]) -> f64 {
        let lse = |v: &[f32]| -> (f32, f32) {
            let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = v.iter().map(|x| (x - m).exp()).sum();
            (m, s.ln())
        };
        let (pm, plse) = lse(reference);
        let (qm, qlse) = lse(candidate);
        let mut kl = 0.0f64;
        for (a, b) in reference.iter().zip(candidate) {
            let log_p = (a - pm - plse) as f64;
            let log_q = (b - qm - qlse) as f64;
            let p = log_p.exp();
            if p > 0.0 {
                kl += p * (log_p - log_q);
            }
        }
        kl
    }

    /// Diagnostic: print qw35's token ids for the cross-engine repro prompt
    /// so they can be diffed against `llama-server /tokenize` output.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "diagnostic: dumps tokenization of the cross-engine repro prompt"]
    fn real_model_tokenization_dump() {
        let _gpu = real_model_gpu_lock();
        let engine = open_real_engine_with(32, 128);
        let prompt = "<|im_start|>user\nmake a reverse count from 218 to 156, jumping by 2 one time and other by 4<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let ids = engine.tokenizer.encode(prompt, true).expect("encode");
        eprintln!("qw35 tokens ({}): {:?}", ids.len(), ids);
    }

    #[cfg(target_os = "macos")]
    fn open_engine_at(path: &str, prefill_chunk: u32) -> Engine {
        open_engine_at_ctx(path, prefill_chunk, 128)
    }

    #[cfg(target_os = "macos")]
    fn open_engine_at_ctx(path: &str, prefill_chunk: u32, ctx_size: u32) -> Engine {
        Engine::open(EngineConfig {
            model_path: PathBuf::from(path),
            model_id: DEFAULT_MODEL_ID.to_string(),
            ctx_size,
            prefill_chunk,
            kv_cache_type: crate::metal::KvCacheType::Q8_0,
            session_cache: false,
            attn_window: 0,
            attn_sink: 0,
            warm_weights: false,
            test_responder: false,
            verbose: false,
        })
        .unwrap_or_else(|err| panic!("failed to open real native engine: {err}"))
    }

    /// Cross-engine deterministic quality of the unified .gguf: its GF4/AWQ DECODE
    /// path vs the base GGUF reference (its Q4_K PREFILL path), teacher-forced on
    /// the SAME tokens. The unified .gguf runs GF4 on both of a single engine's
    /// paths, so an intra-engine prefill-vs-decode report cannot measure
    /// quality-vs-base for it — this opens the two models SEQUENTIALLY
    /// (base first: capture reference logits, then drop it; candidate second) so
    /// peak memory is a single model. Set QW35_MODEL_UNDER_TEST to the unified
    /// .gguf (default .gguf/Qwowl3.5-9B.gguf); QW35_BASE_MODEL overrides
    /// the reference (default the Q4_K GGUF). Reports argmax flips, top-1
    /// agreement, mean/worst KL(base||candidate), worst |Δlogit|, and the first
    /// divergence.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "diagnostic: cross-engine quality of the unified .gguf vs the base GGUF"]
    fn real_model_unified_quality_report() {
        let _gpu = real_model_gpu_lock();
        let base_path = std::env::var("QW35_BASE_MODEL")
            .unwrap_or_else(|_| ".gguf/Qwen3.5-9B-Q4_K_M.gguf".to_string());
        let cand_path = std::env::var("QW35_MODEL_UNDER_TEST")
            .unwrap_or_else(|_| ".gguf/Qwowl3.5-9B.gguf".to_string());
        let corpus = sidecar_calibration_corpus();

        // Pass 1 — base reference (Q4_K prefill path). Capture tokens, probe
        // positions, and reference logits per transcript, then free the engine.
        let mut refs: Vec<(String, Vec<u32>, Vec<usize>, Vec<Vec<f32>>)> = Vec::new();
        {
            let base = open_engine_at_ctx(&base_path, 32, 4096);
            for (label, user, assistant) in &corpus {
                let messages = vec![ChatTurn {
                    role: "user".to_string(),
                    content: (*user).to_string(),
                }];
                let mut text = render_qwen35_chat_prompt(&messages, false, false);
                text.push_str(assistant);
                let tokens = base.tokenizer.encode(&text, true).expect("tokenize");
                let begin = (tokens.len() / 4).max(15);
                let probes: Vec<usize> = (begin..tokens.len()).step_by(16).collect();
                if probes.is_empty() {
                    continue;
                }
                let ref_logits = teacher_forced_probe_logits(&base, &tokens, &probes, false);
                refs.push(((*label).to_string(), tokens, probes, ref_logits));
            }
            drop(base);
        }

        // Pass 2 — candidate decode path (GF4/AWQ), scored on the same tokens.
        let cand = open_engine_at_ctx(&cand_path, 32, 4096);
        eprintln!(
            "unified_quality: base={base_path} candidate={cand_path} ffn={}",
            cand.ffn_label()
        );
        let mut total_probes = 0usize;
        let mut total_flips = 0usize;
        let mut sum_kl = 0.0f64;
        let mut worst_abs = 0.0f32;
        let mut worst_kl = 0.0f64;
        let mut first_divergence: Option<(usize, String)> = None;
        for (label, tokens, probes, ref_logits) in &refs {
            let cand_logits = teacher_forced_probe_logits(&cand, tokens, probes, true);
            let mut flips = 0usize;
            let mut kl_sum = 0.0f64;
            for (idx, &probe) in probes.iter().enumerate() {
                let (a, b) = (&ref_logits[idx], &cand_logits[idx]);
                let (ta, tb) = (argmax(a), argmax(b));
                let max_abs = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                let kl = kl_div_logits(a, b);
                worst_abs = worst_abs.max(max_abs);
                worst_kl = worst_kl.max(kl);
                kl_sum += kl;
                if ta != tb {
                    flips += 1;
                    if first_divergence.is_none() {
                        first_divergence = Some((probe, format!("{label}: base={ta} cand={tb}")));
                    }
                }
            }
            total_probes += probes.len();
            total_flips += flips;
            sum_kl += kl_sum;
            eprintln!(
                "  [{label:<12}] probes={:>3} flips={flips:>2} mean_kl={:.5}",
                probes.len(),
                kl_sum / probes.len() as f64,
            );
        }
        let agree = if total_probes > 0 {
            100.0 * (total_probes - total_flips) as f64 / total_probes as f64
        } else {
            0.0
        };
        eprintln!(
            "unified_quality TOTAL: probes={total_probes} flips={total_flips} \
             top1_agreement={agree:.2}% mean_kl={:.5} worst_kl={worst_kl:.5} worst_abs={worst_abs:.4}",
            sum_kl / total_probes.max(1) as f64,
        );
        match &first_divergence {
            Some((probe, what)) => eprintln!("  first divergence at probe pos {probe}: {what}"),
            None => eprintln!("  no argmax divergence across the corpus"),
        }
    }

    #[cfg(target_os = "macos")]
    fn open_real_engine(prefill_chunk: u32) -> Engine {
        open_real_engine_with(prefill_chunk, 128)
    }

    #[cfg(target_os = "macos")]
    fn open_real_engine_with(prefill_chunk: u32, ctx_size: u32) -> Engine {
        Engine::open(EngineConfig {
            model_path: PathBuf::from(".gguf/Qwen3.5-9B-Q4_K_M.gguf"),
            model_id: DEFAULT_MODEL_ID.to_string(),
            ctx_size,
            prefill_chunk,
            kv_cache_type: crate::metal::KvCacheType::Q8_0,
            session_cache: false,
            attn_window: 0,
            attn_sink: 0,
            warm_weights: false,
            test_responder: false,
            verbose: false,
        })
        .unwrap_or_else(|err| panic!("failed to open real native engine: {err}"))
    }

    #[cfg(target_os = "macos")]
    fn prefill_logits(engine: &Engine) -> Vec<f32> {
        let prompt_tokens = test_prompt_tokens(engine);
        assert!(prompt_tokens.len() > 8);

        let mut session = engine
            .metal_runtime
            .as_ref()
            .expect("real engine should have native runtime")
            .lock()
            .expect("native runtime lock should not be poisoned");
        session.runtime.reset().expect("runtime should reset");
        eval_prompt_tokens(
            engine,
            &mut session.runtime,
            &prompt_tokens,
            crate::metal::LogitsMode::Full,
        );
        session.runtime.copy_logits().expect("logits should copy")
    }

    #[cfg(target_os = "macos")]
    fn greedy_tokens_after_prefill(engine: &Engine, count: usize) -> Vec<u32> {
        let prompt_tokens = test_prompt_tokens(engine);
        assert!(prompt_tokens.len() > 8);

        let mut session = engine
            .metal_runtime
            .as_ref()
            .expect("real engine should have native runtime")
            .lock()
            .expect("native runtime lock should not be poisoned");
        session.runtime.reset().expect("runtime should reset");
        eval_prompt_tokens(
            engine,
            &mut session.runtime,
            &prompt_tokens,
            crate::metal::LogitsMode::Argmax,
        );

        let mut out = Vec::with_capacity(count);
        for step in 0..count {
            let next = session
                .runtime
                .argmax()
                .expect("argmax should produce a token")
                .0;
            out.push(next);
            if step + 1 < count {
                let pos = prompt_tokens.len() as u32 + step as u32;
                session
                    .runtime
                    .eval_token(next, pos, crate::metal::LogitsMode::Argmax)
                    .expect("decode token should evaluate");
            }
        }
        out
    }

    #[cfg(target_os = "macos")]
    fn test_prompt_tokens(engine: &Engine) -> Vec<u32> {
        let request = GenerateRequest {
            model: DEFAULT_MODEL_ID.to_string(),
            messages: vec![ChatTurn {
                role: "user".to_string(),
                content: "Say hi, then name one color.".to_string(),
            }],
            max_tokens: TokenLimit::Fixed(1),
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            repeat_last_n: -1,
            enable_thinking: false,
            preserve_thinking: false,
            thinking_budget: None,
            reasoning_budget_message: None,
            ignore_eos: true,
            stop_sequences: Vec::new(),
            emit_reasoning: false,
        };
        let prompt = render_qwen35_chat_prompt(
            &request.messages,
            request.enable_thinking,
            request.preserve_thinking,
        );
        
        engine
            .tokenizer
            .encode(&prompt, true)
            .expect("test prompt should tokenize")
    }

    #[cfg(target_os = "macos")]
    fn eval_prompt_tokens(
        engine: &Engine,
        runtime: &mut crate::metal::MetalRuntime,
        prompt_tokens: &[u32],
        head_mode: crate::metal::LogitsMode,
    ) {
        if engine.prefill_chunk <= 1 {
            let last = prompt_tokens.len() - 1;
            for (pos, &token) in prompt_tokens.iter().enumerate() {
                let mode = if pos == last {
                    head_mode
                } else {
                    crate::metal::LogitsMode::None
                };
                runtime
                    .eval_token(token, pos as u32, mode)
                    .expect("scalar prefill should evaluate");
            }
        } else {
            let chunk_len = engine.prefill_chunk as usize;
            for (chunk_idx, chunk_tokens) in prompt_tokens.chunks(chunk_len).enumerate() {
                let pos0 = (chunk_idx * chunk_len) as u32;
                let mode = if pos0 as usize + chunk_tokens.len() == prompt_tokens.len() {
                    head_mode
                } else {
                    crate::metal::LogitsMode::None
                };
                runtime
                    .eval_prefill_chunk(chunk_tokens, pos0, mode)
                    .expect("chunked prefill should evaluate");
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx as u32)
            .expect("logits should not be empty")
    }
