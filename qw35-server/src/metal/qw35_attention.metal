// Attention preprocess and GQA kernels (f16 KV cache).
constant int QW35_ATTN_N_HEAD [[function_constant(FC_ATTN_HEADS)]];

constant int QW35_ATTN_N_KV_HEAD [[function_constant(FC_ATTN_KV_HEADS)]];

constant int QW35_ATTN_HEAD_DIM_CONST [[function_constant(FC_ATTN_HEAD_DIM)]];

constant int QW35_ATTN_ROPE_DIM_CONST [[function_constant(FC_ATTN_ROPE_DIM)]];

kernel void qw35_attn_decode_preprocess_f32(
    device const float * q_full [[buffer(0)]],
    device const float * k_src [[buffer(1)]],
    device const float * v_src [[buffer(2)]],
    device const float * q_norm_w [[buffer(3)]],
    device const float * k_norm_w [[buffer(4)]],
    device float * q_dst [[buffer(5)]],
    device float * gate_dst [[buffer(6)]],
    device ushort * k_cache [[buffer(7)]],
    device ushort * v_cache [[buffer(8)]],
    constant int64_t &n_head [[buffer(9)]],
    constant int64_t &n_kv_head [[buffer(10)]],
    constant int64_t &head_dim [[buffer(11)]],
    constant int64_t &pos [[buffer(12)]],
    constant float &eps [[buffer(13)]],
    device const float * rope_freq [[buffer(14)]],
    constant int64_t &rope_dim [[buffer(15)]],
    uint idx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const int q_dim = QW35_ATTN_N_HEAD * QW35_ATTN_HEAD_DIM_CONST;
    const int kv_dim = QW35_ATTN_N_KV_HEAD * QW35_ATTN_HEAD_DIM_CONST;
    const int hd = QW35_ATTN_HEAD_DIM_CONST;
    const int rope = QW35_ATTN_ROPE_DIM_CONST;
    const int half_rope = rope / 2;

    threadgroup float rope_cos[64];
    threadgroup float rope_sin[64];
    if (int(tid) < half_rope) {
        const float theta = float(pos) * rope_freq[tid];
        rope_cos[tid] = cos(theta);
        rope_sin[tid] = sin(theta);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (idx < uint(q_dim)) {
        const int h = int(idx) / hd;
        const int d = int(idx) - h * hd;
        const int q_base = h * hd * 2;
        float ss = 0.0f;
        for (int j = 0; j < hd; j++) {
            const float v = q_full[q_base + j];
            ss += v * v;
        }
        const float inv = rsqrt(ss / float(hd) + eps);
        if (d < rope) {
            const int pair = d < half_rope ? d : d - half_rope;
            const float x0 = q_full[q_base + pair] * inv * q_norm_w[pair];
            const float x1 = q_full[q_base + pair + half_rope] * inv * q_norm_w[pair + half_rope];
            q_dst[idx] = d < half_rope ? (x0 * rope_cos[pair] - x1 * rope_sin[pair])
                                       : (x0 * rope_sin[pair] + x1 * rope_cos[pair]);
        } else {
            q_dst[idx] = q_full[q_base + d] * inv * q_norm_w[d];
        }
        gate_dst[idx] = q_full[q_base + hd + d];
    }

    if (idx < uint(kv_dim)) {
        const int h = int(idx) / hd;
        const int d = int(idx) - h * hd;
        const int k_base = h * hd;
        float ss = 0.0f;
        for (int j = 0; j < hd; j++) {
            const float v = k_src[k_base + j];
            ss += v * v;
        }
        const float inv = rsqrt(ss / float(hd) + eps);
        const int cache_off = int(pos) * kv_dim + int(idx);
        if (d < rope) {
            const int pair = d < half_rope ? d : d - half_rope;
            const float x0 = k_src[k_base + pair] * inv * k_norm_w[pair];
            const float x1 = k_src[k_base + pair + half_rope] * inv * k_norm_w[pair + half_rope];
            const float kv = d < half_rope ? (x0 * rope_cos[pair] - x1 * rope_sin[pair])
                                           : (x0 * rope_sin[pair] + x1 * rope_cos[pair]);
            k_cache[cache_off] = qw35_f32_to_bf16(kv);
        } else {
            k_cache[cache_off] = qw35_f32_to_bf16(k_src[k_base + d] * inv * k_norm_w[d]);
        }
        v_cache[cache_off] = qw35_f32_to_bf16(v_src[idx]);
    }
}

kernel void qw35_attn_prefill_preprocess_f32(
    device const float * q_full [[buffer(0)]],
    device const float * k_src [[buffer(1)]],
    device const float * v_src [[buffer(2)]],
    device const float * q_norm_w [[buffer(3)]],
    device const float * k_norm_w [[buffer(4)]],
    device float * q_dst [[buffer(5)]],
    device float * gate_dst [[buffer(6)]],
    device ushort * k_cache [[buffer(7)]],
    device ushort * v_cache [[buffer(8)]],
    constant int64_t &n_head [[buffer(9)]],
    constant int64_t &n_kv_head [[buffer(10)]],
    constant int64_t &head_dim [[buffer(11)]],
    constant int64_t &pos0 [[buffer(12)]],
    constant float &eps [[buffer(13)]],
    device const float * rope_freq [[buffer(14)]],
    constant int64_t &rope_dim [[buffer(15)]],
    constant int64_t &n_tokens [[buffer(20)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const int q_dim = QW35_ATTN_N_HEAD * QW35_ATTN_HEAD_DIM_CONST;
    const int kv_dim = QW35_ATTN_N_KV_HEAD * QW35_ATTN_HEAD_DIM_CONST;
    const int work_dim = max(q_dim, kv_dim);
    if (work_dim <= 0) return;

    const int token = int(gid) / work_dim;
    const int idx = int(gid) - token * work_dim;
    if (token >= int(n_tokens)) return;

    const int hd = QW35_ATTN_HEAD_DIM_CONST;
    const int rope = QW35_ATTN_ROPE_DIM_CONST;
    const int half_rope = rope / 2;
    const int pos = int(pos0) + token;

    threadgroup float rope_cos[64];
    threadgroup float rope_sin[64];
    if (int(tid) < half_rope) {
        const float theta = float(pos) * rope_freq[tid];
        rope_cos[tid] = cos(theta);
        rope_sin[tid] = sin(theta);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (idx < q_dim) {
        const int h = idx / hd;
        const int d = idx - h * hd;
        const int q_base = token * q_dim * 2 + h * hd * 2;
        float ss = 0.0f;
        for (int j = 0; j < hd; j++) {
            const float v = q_full[q_base + j];
            ss += v * v;
        }
        const float inv = rsqrt(ss / float(hd) + eps);
        const int out = token * q_dim + idx;
        if (d < rope) {
            const int pair = d < half_rope ? d : d - half_rope;
            const float x0 = q_full[q_base + pair] * inv * q_norm_w[pair];
            const float x1 = q_full[q_base + pair + half_rope] * inv * q_norm_w[pair + half_rope];
            q_dst[out] = d < half_rope ? (x0 * rope_cos[pair] - x1 * rope_sin[pair])
                                       : (x0 * rope_sin[pair] + x1 * rope_cos[pair]);
        } else {
            q_dst[out] = q_full[q_base + d] * inv * q_norm_w[d];
        }
        gate_dst[out] = q_full[q_base + hd + d];
    }

    if (idx < kv_dim) {
        const int h = idx / hd;
        const int d = idx - h * hd;
        const int k_base = token * kv_dim + h * hd;
        float ss = 0.0f;
        for (int j = 0; j < hd; j++) {
            const float v = k_src[k_base + j];
            ss += v * v;
        }
        const float inv = rsqrt(ss / float(hd) + eps);
        const int cache_off = pos * kv_dim + idx;
        if (d < rope) {
            const int pair = d < half_rope ? d : d - half_rope;
            const float x0 = k_src[k_base + pair] * inv * k_norm_w[pair];
            const float x1 = k_src[k_base + pair + half_rope] * inv * k_norm_w[pair + half_rope];
            const float kv = d < half_rope ? (x0 * rope_cos[pair] - x1 * rope_sin[pair])
                                           : (x0 * rope_sin[pair] + x1 * rope_cos[pair]);
            k_cache[cache_off] = qw35_f32_to_bf16(kv);
        } else {
            k_cache[cache_off] = qw35_f32_to_bf16(k_src[k_base + d] * inv * k_norm_w[d]);
        }
        v_cache[cache_off] = qw35_f32_to_bf16(v_src[token * kv_dim + idx]);
    }
}

kernel void qw35_attention_gqa_flash_decode_f32(
    device const float * q [[buffer(0)]],
    device const ushort * k_cache [[buffer(1)]],
    device const ushort * v_cache [[buffer(2)]],
    device float * dst [[buffer(3)]],
    device const float * gate [[buffer(4)]],
    constant int64_t &n_head [[buffer(5)]],
    constant int64_t &n_kv_head [[buffer(6)]],
    constant int64_t &head_dim [[buffer(7)]],
    constant int64_t &seq_len [[buffer(8)]],
    constant float &sm_scale [[buffer(9)]],
    constant int &attn_window [[buffer(10)]],
    constant int &attn_sink [[buffer(11)]],
    uint h_u [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const int h = int(h_u);
    const int hd = QW35_ATTN_HEAD_DIM_CONST;
    if (h >= QW35_ATTN_N_HEAD || hd <= 0 || hd > 256 || seq_len <= 0) return;

    const int kv_group = QW35_ATTN_N_HEAD / QW35_ATTN_N_KV_HEAD;
    const int kv_h = h / kv_group;
    const int kv_dim = QW35_ATTN_N_KV_HEAD * hd;
    const int q_off = h * hd;
    const int d0 = int(lane) * 8;

    float qv[8];
    for (int i = 0; i < 8; i++) qv[i] = q[q_off + d0 + i];

    float m = -INFINITY;
    float l = 0.0f;
    float acc[8] = {0.0f};

    // Sliding-window + attention sink: attend to the first `sink` positions and
    // the last `attn_window` positions only, bounding the loop to sink+window
    // instead of O(seq_len). attn_window <= 0 (or covering the whole context)
    // means full attention.
    const int sl = int(seq_len);
    int sink = max(0, min(attn_sink, sl));
    int recent_start, total;
    if (attn_window <= 0 || sink + attn_window >= sl) {
        sink = 0; recent_start = 0; total = sl;
    } else {
        recent_start = sl - attn_window; total = sink + attn_window;
    }
    for (int idx = 0; idx < total; ++idx) {
        const int t = (idx < sink) ? idx : (recent_start + (idx - sink));
        const int k_off = t * kv_dim + kv_h * hd + d0;

        float dot = 0.0f;
        for (int i = 0; i < 8; i++) {
            dot += qv[i] * qw35_bf16_to_f32(k_cache[k_off + i]);
        }
        dot = simd_sum(dot);

        const float score = dot * sm_scale;
        const float next_m = max(m, score);
        const float old_scale = isinf(m) ? 0.0f : exp(m - next_m);
        const float score_scale = exp(score - next_m);

        for (int i = 0; i < 8; i++) {
            acc[i] = acc[i] * old_scale + score_scale * qw35_bf16_to_f32(v_cache[k_off + i]);
        }
        l = l * old_scale + score_scale;
        m = next_m;
    }

    const int out = h * hd + d0;
    for (int i = 0; i < 8; i++) {
        dst[out + i] = (acc[i] / l) * qw35_sigmoid_f32(gate[out + i]);
    }
}

// ---------------------------------------------------------------------------
// q8_0 block-quantized KV cache kernels.
//
// These mirror the f16 kernels above: K (post-RoPE) and V rows are staged in
// threadgroup memory and quantized one 32-element block per thread on the write
// path; the read path dequantizes inline (a trivial int8->float cast) and
// applies the per-block fp16 scale after the dot product. The four kernels are
// generated from one macro so the math stays identical to the f16 path.
// Requires kv_dim % 32 == 0. q8_0 reuses the weight block struct
// (qw35_block_q8_0) defined in qw35_types.metal.

#define QW35_QK8_0 32

/// Quantize 32 threadgroup floats into one q8_0 block (symmetric int8 + fp16 scale).
static inline void qw35_q8_0_quantize_block(threadgroup const float * x,
                                            device qw35_block_q8_0 * blk) {
    float amax = 0.0f;
    for (int i = 0; i < QW35_QK8_0; i++) amax = max(amax, fabs(x[i]));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    blk->d = half(d);
    for (int i = 0; i < QW35_QK8_0; i++) {
        blk->qs[i] = (char)clamp(int(round(x[i] * id)), -127, 127);
    }
}

/// Signed int8 quant of element e (0..31) of a q8_0 block. Multiply by float(blk->d).
static inline float qw35_q8_0_dequant_qi(device const qw35_block_q8_0 * blk, int e) {
    return float(blk->qs[e]);
}

// Generate the 4 KV kernels for one block format (currently instantiated for q8_0).
//   TAG: name suffix token (e.g. q8_0)
//   BT : block struct type        QFN: quantize-block fn
//   DFN: centered-dequant fn      QK : block size (32)
#define QW35_GEN_KV_KERNELS(TAG, BT, QFN, DFN, QK) \
kernel void qw35_attn_decode_preprocess_##TAG##_f32( \
    device const float * q_full [[buffer(0)]], \
    device const float * k_src [[buffer(1)]], \
    device const float * v_src [[buffer(2)]], \
    device const float * q_norm_w [[buffer(3)]], \
    device const float * k_norm_w [[buffer(4)]], \
    device float * q_dst [[buffer(5)]], \
    device float * gate_dst [[buffer(6)]], \
    device BT * k_cache [[buffer(7)]], \
    device BT * v_cache [[buffer(8)]], \
    constant int64_t &n_head [[buffer(9)]], \
    constant int64_t &n_kv_head [[buffer(10)]], \
    constant int64_t &head_dim [[buffer(11)]], \
    constant int64_t &pos [[buffer(12)]], \
    constant float &eps [[buffer(13)]], \
    device const float * rope_freq [[buffer(14)]], \
    constant int64_t &rope_dim [[buffer(15)]], \
    uint idx [[thread_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]] \
) { \
    const int q_dim = QW35_ATTN_N_HEAD * QW35_ATTN_HEAD_DIM_CONST; \
    const int kv_dim = QW35_ATTN_N_KV_HEAD * QW35_ATTN_HEAD_DIM_CONST; \
    const int hd = QW35_ATTN_HEAD_DIM_CONST; \
    const int rope = QW35_ATTN_ROPE_DIM_CONST; \
    const int half_rope = rope / 2; \
    threadgroup float rope_cos[64]; \
    threadgroup float rope_sin[64]; \
    threadgroup float tg_k[256]; \
    threadgroup float tg_v[256]; \
    if (int(tid) < half_rope) { \
        const float theta = float(pos) * rope_freq[tid]; \
        rope_cos[tid] = cos(theta); \
        rope_sin[tid] = sin(theta); \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (idx < uint(q_dim)) { \
        const int h = int(idx) / hd; \
        const int d = int(idx) - h * hd; \
        const int q_base = h * hd * 2; \
        float ss = 0.0f; \
        for (int j = 0; j < hd; j++) { ss += q_full[q_base + j] * q_full[q_base + j]; } \
        const float inv = rsqrt(ss / float(hd) + eps); \
        if (d < rope) { \
            const int pair = d < half_rope ? d : d - half_rope; \
            const float x0 = q_full[q_base + pair] * inv * q_norm_w[pair]; \
            const float x1 = q_full[q_base + pair + half_rope] * inv * q_norm_w[pair + half_rope]; \
            q_dst[idx] = d < half_rope ? (x0 * rope_cos[pair] - x1 * rope_sin[pair]) \
                                       : (x0 * rope_sin[pair] + x1 * rope_cos[pair]); \
        } else { \
            q_dst[idx] = q_full[q_base + d] * inv * q_norm_w[d]; \
        } \
        gate_dst[idx] = q_full[q_base + hd + d]; \
    } \
    if (idx < uint(kv_dim)) { \
        const int h = int(idx) / hd; \
        const int d = int(idx) - h * hd; \
        const int k_base = h * hd; \
        float ss = 0.0f; \
        for (int j = 0; j < hd; j++) { ss += k_src[k_base + j] * k_src[k_base + j]; } \
        const float inv = rsqrt(ss / float(hd) + eps); \
        if (d < rope) { \
            const int pair = d < half_rope ? d : d - half_rope; \
            const float x0 = k_src[k_base + pair] * inv * k_norm_w[pair]; \
            const float x1 = k_src[k_base + pair + half_rope] * inv * k_norm_w[pair + half_rope]; \
            tg_k[tid] = d < half_rope ? (x0 * rope_cos[pair] - x1 * rope_sin[pair]) \
                                      : (x0 * rope_sin[pair] + x1 * rope_cos[pair]); \
        } else { \
            tg_k[tid] = k_src[k_base + d] * inv * k_norm_w[d]; \
        } \
        tg_v[tid] = v_src[idx]; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (idx < uint(kv_dim) && (tid % QK) == 0) { \
        const int row = int(pos) * (kv_dim / QK); \
        const int blk = int(idx) / QK; \
        QFN(&tg_k[tid], &k_cache[row + blk]); \
        QFN(&tg_v[tid], &v_cache[row + blk]); \
    } \
} \
kernel void qw35_attn_prefill_preprocess_##TAG##_f32( \
    device const float * q_full [[buffer(0)]], \
    device const float * k_src [[buffer(1)]], \
    device const float * v_src [[buffer(2)]], \
    device const float * q_norm_w [[buffer(3)]], \
    device const float * k_norm_w [[buffer(4)]], \
    device float * q_dst [[buffer(5)]], \
    device float * gate_dst [[buffer(6)]], \
    device BT * k_cache [[buffer(7)]], \
    device BT * v_cache [[buffer(8)]], \
    constant int64_t &n_head [[buffer(9)]], \
    constant int64_t &n_kv_head [[buffer(10)]], \
    constant int64_t &head_dim [[buffer(11)]], \
    constant int64_t &pos0 [[buffer(12)]], \
    constant float &eps [[buffer(13)]], \
    device const float * rope_freq [[buffer(14)]], \
    constant int64_t &rope_dim [[buffer(15)]], \
    constant int64_t &n_tokens [[buffer(20)]], \
    uint gid [[thread_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]] \
) { \
    const int q_dim = QW35_ATTN_N_HEAD * QW35_ATTN_HEAD_DIM_CONST; \
    const int kv_dim = QW35_ATTN_N_KV_HEAD * QW35_ATTN_HEAD_DIM_CONST; \
    const int work_dim = max(q_dim, kv_dim); \
    if (work_dim <= 0) return; \
    const int token = int(gid) / work_dim; \
    const int idx = int(gid) - token * work_dim; \
    const bool active = token < int(n_tokens); \
    const int hd = QW35_ATTN_HEAD_DIM_CONST; \
    const int rope = QW35_ATTN_ROPE_DIM_CONST; \
    const int half_rope = rope / 2; \
    const int pos = int(pos0) + token; \
    threadgroup float rope_cos[64]; \
    threadgroup float rope_sin[64]; \
    threadgroup float tg_k[256]; \
    threadgroup float tg_v[256]; \
    if (active && int(tid) < half_rope) { \
        const float theta = float(pos) * rope_freq[tid]; \
        rope_cos[tid] = cos(theta); \
        rope_sin[tid] = sin(theta); \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (active && idx < q_dim) { \
        const int h = idx / hd; \
        const int d = idx - h * hd; \
        const int q_base = token * q_dim * 2 + h * hd * 2; \
        float ss = 0.0f; \
        for (int j = 0; j < hd; j++) { ss += q_full[q_base + j] * q_full[q_base + j]; } \
        const float inv = rsqrt(ss / float(hd) + eps); \
        const int out = token * q_dim + idx; \
        if (d < rope) { \
            const int pair = d < half_rope ? d : d - half_rope; \
            const float x0 = q_full[q_base + pair] * inv * q_norm_w[pair]; \
            const float x1 = q_full[q_base + pair + half_rope] * inv * q_norm_w[pair + half_rope]; \
            q_dst[out] = d < half_rope ? (x0 * rope_cos[pair] - x1 * rope_sin[pair]) \
                                       : (x0 * rope_sin[pair] + x1 * rope_cos[pair]); \
        } else { \
            q_dst[out] = q_full[q_base + d] * inv * q_norm_w[d]; \
        } \
        gate_dst[out] = q_full[q_base + hd + d]; \
    } \
    if (active && idx < kv_dim) { \
        const int h = idx / hd; \
        const int d = idx - h * hd; \
        const int k_base = token * kv_dim + h * hd; \
        float ss = 0.0f; \
        for (int j = 0; j < hd; j++) { ss += k_src[k_base + j] * k_src[k_base + j]; } \
        const float inv = rsqrt(ss / float(hd) + eps); \
        if (d < rope) { \
            const int pair = d < half_rope ? d : d - half_rope; \
            const float x0 = k_src[k_base + pair] * inv * k_norm_w[pair]; \
            const float x1 = k_src[k_base + pair + half_rope] * inv * k_norm_w[pair + half_rope]; \
            tg_k[tid] = d < half_rope ? (x0 * rope_cos[pair] - x1 * rope_sin[pair]) \
                                      : (x0 * rope_sin[pair] + x1 * rope_cos[pair]); \
        } else { \
            tg_k[tid] = k_src[k_base + d] * inv * k_norm_w[d]; \
        } \
        tg_v[tid] = v_src[token * kv_dim + idx]; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (active && idx < kv_dim && (tid % QK) == 0) { \
        const int row = pos * (kv_dim / QK); \
        const int blk = idx / QK; \
        QFN(&tg_k[tid], &k_cache[row + blk]); \
        QFN(&tg_v[tid], &v_cache[row + blk]); \
    } \
} \
kernel void qw35_attention_gqa_flash_decode_##TAG##_f32( \
    device const float * q [[buffer(0)]], \
    device const BT * k_cache [[buffer(1)]], \
    device const BT * v_cache [[buffer(2)]], \
    device float * dst [[buffer(3)]], \
    device const float * gate [[buffer(4)]], \
    constant int64_t &n_head [[buffer(5)]], \
    constant int64_t &n_kv_head [[buffer(6)]], \
    constant int64_t &head_dim [[buffer(7)]], \
    constant int64_t &seq_len [[buffer(8)]], \
    constant float &sm_scale [[buffer(9)]], \
    constant int &attn_window [[buffer(10)]], \
    constant int &attn_sink [[buffer(11)]], \
    uint h_u [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]] \
) { \
    const int h = int(h_u); \
    const int hd = QW35_ATTN_HEAD_DIM_CONST; \
    if (h >= QW35_ATTN_N_HEAD || hd <= 0 || hd > 256 || seq_len <= 0) return; \
    const int kv_group = QW35_ATTN_N_HEAD / QW35_ATTN_N_KV_HEAD; \
    const int kv_h = h / kv_group; \
    const int kv_dim = QW35_ATTN_N_KV_HEAD * hd; \
    const int blocks_per_row = kv_dim / QK; \
    const int q_off = h * hd; \
    const int d0 = int(lane) * 8; \
    const int eb = kv_h * hd + d0; \
    const int blk_i = eb / QK; \
    const int e0 = eb % QK; \
    float qv[8]; \
    for (int i = 0; i < 8; i++) qv[i] = q[q_off + d0 + i]; \
    float m = -INFINITY; \
    float l = 0.0f; \
    float acc[8] = {0.0f}; \
    const int sl = int(seq_len); \
    int sink = max(0, min(attn_sink, sl)); \
    int recent_start, total; \
    if (attn_window <= 0 || sink + attn_window >= sl) { sink = 0; recent_start = 0; total = sl; } \
    else { recent_start = sl - attn_window; total = sink + attn_window; } \
    for (int idx = 0; idx < total; ++idx) { \
        const int t = (idx < sink) ? idx : (recent_start + (idx - sink)); \
        device const BT * kb = &k_cache[t * blocks_per_row + blk_i]; \
        const float kd = float(kb->d); \
        float dot = 0.0f; \
        for (int i = 0; i < 8; i++) { dot += qv[i] * DFN(kb, e0 + i); } \
        dot = simd_sum(dot * kd); \
        const float score = dot * sm_scale; \
        const float next_m = max(m, score); \
        const float old_scale = isinf(m) ? 0.0f : exp(m - next_m); \
        const float score_scale = exp(score - next_m); \
        device const BT * vb = &v_cache[t * blocks_per_row + blk_i]; \
        const float vd = float(vb->d); \
        for (int i = 0; i < 8; i++) { \
            acc[i] = acc[i] * old_scale + score_scale * vd * DFN(vb, e0 + i); \
        } \
        l = l * old_scale + score_scale; \
        m = next_m; \
    } \
    const int out = h * hd + d0; \
    for (int i = 0; i < 8; i++) { \
        dst[out + i] = (acc[i] / l) * qw35_sigmoid_f32(gate[out + i]); \
    } \
}

QW35_GEN_KV_KERNELS(q8_0, qw35_block_q8_0, qw35_q8_0_quantize_block, qw35_q8_0_dequant_qi, QW35_QK8_0)

// ---------------------------------------------------------------------------
// Prefill attention. One 32-lane simdgroup per (token, head), 8 head-dim
// elems/lane, barrier-free simd_sum per KV position + online softmax (same shape
// as the decode kernel). Causal mask = the loop bound (seq_len = pos0+token+1).
// Dispatched (n_tokens, n_head, 1) x (32, 1, 1). Prefill throughput stays ~flat
// with context (matmul-bound) instead of collapsing under per-position barriers.
kernel void qw35_attention_gqa_prefill_f32(
    device const float * q [[buffer(0)]],
    device const ushort * k_cache [[buffer(1)]],
    device const ushort * v_cache [[buffer(2)]],
    device float * dst [[buffer(3)]],
    device const float * gate [[buffer(4)]],
    constant int64_t &n_head [[buffer(5)]],
    constant int64_t &n_kv_head [[buffer(6)]],
    constant int64_t &head_dim [[buffer(7)]],
    constant int64_t &pos0 [[buffer(8)]],
    constant float &sm_scale [[buffer(9)]],
    constant int64_t &n_tokens [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const int token = int(tg.x);
    const int h = int(tg.y);
    const int hd = QW35_ATTN_HEAD_DIM_CONST;
    if (token >= int(n_tokens) || h >= QW35_ATTN_N_HEAD || hd <= 0 || hd > 256) return;

    const int out_dim = QW35_ATTN_N_HEAD * hd;
    const int kv_group = QW35_ATTN_N_HEAD / QW35_ATTN_N_KV_HEAD;
    const int kv_h = h / kv_group;
    const int kv_dim = QW35_ATTN_N_KV_HEAD * hd;
    const int q_off = token * out_dim + h * hd;
    const int d0 = int(lane) * 8;
    const int seq_len = int(pos0) + token + 1;

    float qv[8];
    for (int i = 0; i < 8; i++) qv[i] = q[q_off + d0 + i];

    float m = -INFINITY;
    float l = 0.0f;
    float acc[8] = {0.0f};

    for (int t = 0; t < seq_len; ++t) {
        const int k_off = t * kv_dim + kv_h * hd + d0;
        float dot = 0.0f;
        for (int i = 0; i < 8; i++) dot += qv[i] * qw35_bf16_to_f32(k_cache[k_off + i]);
        dot = simd_sum(dot);
        const float score = dot * sm_scale;
        const float next_m = max(m, score);
        const float old_scale = isinf(m) ? 0.0f : exp(m - next_m);
        const float score_scale = exp(score - next_m);
        for (int i = 0; i < 8; i++)
            acc[i] = acc[i] * old_scale + score_scale * qw35_bf16_to_f32(v_cache[k_off + i]);
        l = l * old_scale + score_scale;
        m = next_m;
    }

    const int out = token * out_dim + h * hd + d0;
    for (int i = 0; i < 8; i++)
        dst[out + i] = (acc[i] / l) * qw35_sigmoid_f32(gate[out + i]);
}

kernel void qw35_attention_gqa_prefill_q8_0_f32(
    device const float * q [[buffer(0)]],
    device const qw35_block_q8_0 * k_cache [[buffer(1)]],
    device const qw35_block_q8_0 * v_cache [[buffer(2)]],
    device float * dst [[buffer(3)]],
    device const float * gate [[buffer(4)]],
    constant int64_t &n_head [[buffer(5)]],
    constant int64_t &n_kv_head [[buffer(6)]],
    constant int64_t &head_dim [[buffer(7)]],
    constant int64_t &pos0 [[buffer(8)]],
    constant float &sm_scale [[buffer(9)]],
    constant int64_t &n_tokens [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const int token = int(tg.x);
    const int h = int(tg.y);
    const int hd = QW35_ATTN_HEAD_DIM_CONST;
    if (token >= int(n_tokens) || h >= QW35_ATTN_N_HEAD || hd <= 0 || hd > 256) return;

    const int out_dim = QW35_ATTN_N_HEAD * hd;
    const int kv_group = QW35_ATTN_N_HEAD / QW35_ATTN_N_KV_HEAD;
    const int kv_h = h / kv_group;
    const int kv_dim = QW35_ATTN_N_KV_HEAD * hd;
    const int blocks_per_row = kv_dim / QW35_QK8_0;
    const int q_off = token * out_dim + h * hd;
    const int d0 = int(lane) * 8;
    const int eb = kv_h * hd + d0;
    const int blk_i = eb / QW35_QK8_0;
    const int e0 = eb % QW35_QK8_0;
    const int seq_len = int(pos0) + token + 1;

    float qv[8];
    for (int i = 0; i < 8; i++) qv[i] = q[q_off + d0 + i];

    float m = -INFINITY;
    float l = 0.0f;
    float acc[8] = {0.0f};

    for (int t = 0; t < seq_len; ++t) {
        device const qw35_block_q8_0 * kb = &k_cache[t * blocks_per_row + blk_i];
        const float kd = float(kb->d);
        float dot = 0.0f;
        for (int i = 0; i < 8; i++) dot += qv[i] * qw35_q8_0_dequant_qi(kb, e0 + i);
        dot = simd_sum(dot * kd);
        const float score = dot * sm_scale;
        const float next_m = max(m, score);
        const float old_scale = isinf(m) ? 0.0f : exp(m - next_m);
        const float score_scale = exp(score - next_m);
        device const qw35_block_q8_0 * vb = &v_cache[t * blocks_per_row + blk_i];
        const float vd = float(vb->d);
        for (int i = 0; i < 8; i++)
            acc[i] = acc[i] * old_scale + score_scale * vd * qw35_q8_0_dequant_qi(vb, e0 + i);
        l = l * old_scale + score_scale;
        m = next_m;
    }

    const int out = token * out_dim + h * hd + d0;
    for (int i = 0; i < 8; i++)
        dst[out + i] = (acc[i] / l) * qw35_sigmoid_f32(gate[out + i]);
}
