// Gated DeltaNet (SSM) kernels: fused single-token decode + prefill batch path.
kernel void qw35_ssm_conv1d_step4_batch_f32(
    device const float * qkv_in [[buffer(0)]],
    device const float * conv_w [[buffer(1)]],
    device float * conv_state [[buffer(2)]],
    device float * conv_out [[buffer(3)]],
    constant int &conv_channels [[buffer(4)]],
    constant int &n_tokens [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= uint(conv_channels)) return;
    const int c = int(tid);
    const int state_base = c * 3;
    const int weight_base = c * 4;
    float s0 = conv_state[state_base + 0];
    float s1 = conv_state[state_base + 1];
    float s2 = conv_state[state_base + 2];
    const float w0 = conv_w[weight_base + 0];
    const float w1 = conv_w[weight_base + 1];
    const float w2 = conv_w[weight_base + 2];
    const float w3 = conv_w[weight_base + 3];

    for (int t = 0; t < n_tokens; t++) {
        const int off = t * conv_channels + c;
        const float x = qkv_in[off];
        const float acc = s0 * w0 + s1 * w1 + s2 * w2 + x * w3;
        conv_out[off] = qw35_silu_f32(acc);
        s0 = s1;
        s1 = s2;
        s2 = x;
    }

    conv_state[state_base + 0] = s0;
    conv_state[state_base + 1] = s1;
    conv_state[state_base + 2] = s2;
}

kernel void qw35_ssm_l2_repeat_qk_batch_f32(
    device const float * q_src [[buffer(0)]],
    device const float * k_src [[buffer(1)]],
    device float * q_dst [[buffer(2)]],
    device float * k_dst [[buffer(3)]],
    constant int &num_k_heads [[buffer(4)]],
    constant int &num_v_heads [[buffer(5)]],
    constant int &head_dim [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    constant int &src_stride [[buffer(8)]],
    constant int &n_tokens [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const int token = int(tg.x);
    const int dst_head = int(tg.y);
    if (token >= n_tokens || dst_head >= num_v_heads || head_dim <= 0) return;
    threadgroup float partial[8];

    const int src_head = dst_head % num_k_heads;
    const int src_base = token * src_stride + src_head * head_dim;

    float q_ss = 0.0f;
    float k_ss = 0.0f;
    for (int i = int(tid); i < head_dim; i += 256) {
        const float q = q_src[src_base + i];
        const float k = k_src[src_base + i];
        q_ss += q * q;
        k_ss += k * k;
    }
    q_ss = qw35_threadgroup_sum_256(q_ss, partial, tid, lane, simd_group);
    k_ss = qw35_threadgroup_sum_256(k_ss, partial, tid, lane, simd_group);

    const int total = num_v_heads * head_dim;
    const int out_base = token * total + dst_head * head_dim;
    const float q_scale = 1.0f / max(sqrt(q_ss), eps);
    const float k_scale = 1.0f / max(sqrt(k_ss), eps);
    for (int d = int(tid); d < head_dim; d += 256) {
        q_dst[out_base + d] = q_src[src_base + d] * q_scale;
        k_dst[out_base + d] = k_src[src_base + d] * k_scale;
    }
}

kernel void qw35_ssm_conv_recurrent_gate_norm_step128_f32(
    device const float * qkv [[buffer(0)]],
    device const float * conv_w [[buffer(1)]],
    device float * conv_state [[buffer(2)]],
    device const float * beta_raw [[buffer(3)]],
    device const float * alpha_raw [[buffer(4)]],
    device const float * dt [[buffer(5)]],
    device const float * a [[buffer(6)]],
    device float * state [[buffer(7)]],
    device const float * z [[buffer(8)]],
    device const float * norm_w [[buffer(9)]],
    device float * gated [[buffer(10)]],
    constant int &conv_channels [[buffer(11)]],
    constant int &num_v_heads [[buffer(12)]],
    constant int &head_v_dim [[buffer(13)]],
    constant float &scale [[buffer(14)]],
    constant float &eps [[buffer(15)]],
    uint qk_group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    constexpr int GROUPS = 16;
    constexpr int STATE_SIZE = 128;
    constexpr int Q_OFF = 0;
    constexpr int K_OFF = GROUPS * STATE_SIZE;
    constexpr int V_OFF = 2 * GROUPS * STATE_SIZE;

    if (tg_size != 256 || conv_channels != 8192 || head_v_dim != STATE_SIZE || qk_group >= GROUPS) {
        return;
    }

    threadgroup float q_cache[STATE_SIZE];
    threadgroup float k_cache[STATE_SIZE];
    threadgroup float y_cache[256];
    threadgroup float scratch[256];
    threadgroup float inv_qk[2];
    threadgroup float inv_y[2];
    threadgroup float kq_cache;

    const int d = int(tid & 127u);
    if (tid < STATE_SIZE) {
        const int g = int(qk_group);
        const int q_c = Q_OFF + g * STATE_SIZE + d;
        const int k_c = K_OFF + g * STATE_SIZE + d;
        const int q_state_base = q_c * 3;
        const int k_state_base = k_c * 3;
        const int q_weight_base = q_c * 4;
        const int k_weight_base = k_c * 4;

        const float q_s0 = conv_state[q_state_base + 0];
        const float q_s1 = conv_state[q_state_base + 1];
        const float q_s2 = conv_state[q_state_base + 2];
        const float k_s0 = conv_state[k_state_base + 0];
        const float k_s1 = conv_state[k_state_base + 1];
        const float k_s2 = conv_state[k_state_base + 2];
        const float q_x = qkv[q_c];
        const float k_x = qkv[k_c];
        const float q = qw35_silu_f32(q_s0 * conv_w[q_weight_base + 0]
                                    + q_s1 * conv_w[q_weight_base + 1]
                                    + q_s2 * conv_w[q_weight_base + 2]
                                    + q_x  * conv_w[q_weight_base + 3]);
        const float k = qw35_silu_f32(k_s0 * conv_w[k_weight_base + 0]
                                    + k_s1 * conv_w[k_weight_base + 1]
                                    + k_s2 * conv_w[k_weight_base + 2]
                                    + k_x  * conv_w[k_weight_base + 3]);

        conv_state[q_state_base + 0] = q_s1;
        conv_state[q_state_base + 1] = q_s2;
        conv_state[q_state_base + 2] = q_x;
        conv_state[k_state_base + 0] = k_s1;
        conv_state[k_state_base + 1] = k_s2;
        conv_state[k_state_base + 2] = k_x;

        q_cache[d] = q;
        k_cache[d] = k;
        scratch[d] = q * q;
        scratch[STATE_SIZE + d] = k * k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float q_ss = 0.0f;
        float k_ss = 0.0f;
        for (int i = 0; i < STATE_SIZE; ++i) {
            q_ss += scratch[i];
            k_ss += scratch[STATE_SIZE + i];
        }
        inv_qk[0] = 1.0f / max(sqrt(q_ss), eps);
        inv_qk[1] = 1.0f / max(sqrt(k_ss), eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < STATE_SIZE) {
        q_cache[d] *= inv_qk[0];
        k_cache[d] *= inv_qk[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float kq = 0.0f;
        for (int ki = 0; ki < STATE_SIZE; ++ki) {
            kq += k_cache[ki] * q_cache[ki];
        }
        kq_cache = kq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int pair = int(tid >> 7);
    const int h = int(qk_group) + pair * GROUPS;
    const bool active_head = h < num_v_heads;

    float y = 0.0f;
    if (active_head) {
        const int v = h * STATE_SIZE + d;
        const int c = V_OFF + v;
        const int v_state_base = c * 3;
        const int v_weight_base = c * 4;
        const float v_s0 = conv_state[v_state_base + 0];
        const float v_s1 = conv_state[v_state_base + 1];
        const float v_s2 = conv_state[v_state_base + 2];
        const float v_x = qkv[c];
        const float v_conv = qw35_silu_f32(v_s0 * conv_w[v_weight_base + 0]
                                         + v_s1 * conv_w[v_weight_base + 1]
                                         + v_s2 * conv_w[v_weight_base + 2]
                                         + v_x  * conv_w[v_weight_base + 3]);
        conv_state[v_state_base + 0] = v_s1;
        conv_state[v_state_base + 1] = v_s2;
        conv_state[v_state_base + 2] = v_x;

        const int row = h * STATE_SIZE * head_v_dim + d * STATE_SIZE;
        const float beta = qw35_sigmoid_f32(beta_raw[h]);
        const float alpha = qw35_softplus_f32(alpha_raw[h] + dt[h]) * a[h];
        const float gate = exp(alpha);

        float k_dot = 0.0f;
        float q_dot = 0.0f;
        for (int ki = 0; ki < STATE_SIZE; ++ki) {
            const float old_s = state[row + ki];
            const float k = k_cache[ki];
            const float q = q_cache[ki];
            k_dot += old_s * k;
            q_dot += old_s * q;
        }

        const float delta = (v_conv - gate * k_dot) * beta;
        for (int ki = 0; ki < STATE_SIZE; ++ki) {
            const float old_s = state[row + ki];
            state[row + ki] = old_s * gate + k_cache[ki] * delta;
        }
        y = (gate * q_dot + delta * kq_cache) * scale;
    }

    y_cache[tid] = y;
    scratch[tid] = active_head ? y * y : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0 || tid == STATE_SIZE) {
        const int base = tid == 0 ? 0 : STATE_SIZE;
        float ss = 0.0f;
        for (int i = 0; i < STATE_SIZE; ++i) {
            ss += scratch[base + i];
        }
        inv_y[base / STATE_SIZE] = rsqrt(ss / float(STATE_SIZE) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active_head) {
        const int idx = h * STATE_SIZE + d;
        gated[idx] = y_cache[tid] * inv_y[pair] * norm_w[d] * qw35_silu_f32(z[idx]);
    }
}

// Parallel-reduction variant of the fused decode step above: the L2 norms,
// the k·q dot, and the per-head y-RMS run as threadgroup/simdgroup tree
// reductions instead of single-thread 128-element loops. The sums
// reassociate, so values can differ from the serial kernel in the last ulp —
// the serial kernel stays the QW35_SSM_SERIAL fallback.
kernel void qw35_ssm_conv_recurrent_gate_norm_step128_par_f32(
    device const float * qkv [[buffer(0)]],
    device const float * conv_w [[buffer(1)]],
    device float * conv_state [[buffer(2)]],
    device const float * beta_raw [[buffer(3)]],
    device const float * alpha_raw [[buffer(4)]],
    device const float * dt [[buffer(5)]],
    device const float * a [[buffer(6)]],
    device float * state [[buffer(7)]],
    device const float * z [[buffer(8)]],
    device const float * norm_w [[buffer(9)]],
    device float * gated [[buffer(10)]],
    constant int &conv_channels [[buffer(11)]],
    constant int &num_v_heads [[buffer(12)]],
    constant int &head_v_dim [[buffer(13)]],
    constant float &scale [[buffer(14)]],
    constant float &eps [[buffer(15)]],
    uint qk_group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr int GROUPS = 16;
    constexpr int STATE_SIZE = 128;
    constexpr int Q_OFF = 0;
    constexpr int K_OFF = GROUPS * STATE_SIZE;
    constexpr int V_OFF = 2 * GROUPS * STATE_SIZE;

    if (tg_size != 256 || conv_channels != 8192 || head_v_dim != STATE_SIZE || qk_group >= GROUPS) {
        return;
    }

    // float4-backed so the state sweep can read q/k as float4 without relying
    // on the default 4-byte alignment of a threadgroup float array.
    threadgroup float4 q_cache4[STATE_SIZE / 4];
    threadgroup float4 k_cache4[STATE_SIZE / 4];
    threadgroup float *q_cache = (threadgroup float *)q_cache4;
    threadgroup float *k_cache = (threadgroup float *)k_cache4;
    threadgroup float y_cache[2 * STATE_SIZE];
    threadgroup float partial8[8];

    const int d = int(tid & 127u);
    // Raw conv+SiLU q/k for this lane's element; threads 128..255 carry zeros
    // so the whole-threadgroup reductions below stay well-defined.
    float q = 0.0f;
    float k = 0.0f;
    if (tid < STATE_SIZE) {
        const int g = int(qk_group);
        const int q_c = Q_OFF + g * STATE_SIZE + d;
        const int k_c = K_OFF + g * STATE_SIZE + d;
        const int q_state_base = q_c * 3;
        const int k_state_base = k_c * 3;
        const int q_weight_base = q_c * 4;
        const int k_weight_base = k_c * 4;

        const float q_s0 = conv_state[q_state_base + 0];
        const float q_s1 = conv_state[q_state_base + 1];
        const float q_s2 = conv_state[q_state_base + 2];
        const float k_s0 = conv_state[k_state_base + 0];
        const float k_s1 = conv_state[k_state_base + 1];
        const float k_s2 = conv_state[k_state_base + 2];
        const float q_x = qkv[q_c];
        const float k_x = qkv[k_c];
        q = qw35_silu_f32(q_s0 * conv_w[q_weight_base + 0]
                        + q_s1 * conv_w[q_weight_base + 1]
                        + q_s2 * conv_w[q_weight_base + 2]
                        + q_x  * conv_w[q_weight_base + 3]);
        k = qw35_silu_f32(k_s0 * conv_w[k_weight_base + 0]
                        + k_s1 * conv_w[k_weight_base + 1]
                        + k_s2 * conv_w[k_weight_base + 2]
                        + k_x  * conv_w[k_weight_base + 3]);

        conv_state[q_state_base + 0] = q_s1;
        conv_state[q_state_base + 1] = q_s2;
        conv_state[q_state_base + 2] = q_x;
        conv_state[k_state_base + 0] = k_s1;
        conv_state[k_state_base + 1] = k_s2;
        conv_state[k_state_base + 2] = k_x;
    }

    // L2 norms, normalized k·q dot, and the normalized caches — parallel
    // tree reductions across the threadgroup instead of single-thread loops.
    // The explicit barriers between the reductions keep one call's final
    // partial8 read ordered before the next call's first partial8 write
    // (they share the scratch array).
    const float q_ss = qw35_threadgroup_sum_256(q * q, partial8, tid, lane, simd_group);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float k_ss = qw35_threadgroup_sum_256(k * k, partial8, tid, lane, simd_group);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float qn = q * (1.0f / max(sqrt(q_ss), eps));
    const float kn = k * (1.0f / max(sqrt(k_ss), eps));
    const float kq_dot = qw35_threadgroup_sum_256(qn * kn, partial8, tid, lane, simd_group);
    if (tid < STATE_SIZE) {
        q_cache[d] = qn;
        k_cache[d] = kn;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Delta-rule state update: one simdgroup per state row ────────────────
    //
    // The row-per-thread layout this replaced put consecutive threads 512 B
    // apart (`row = h*128*128 + d*128`), so a simdgroup touched 32 distinct
    // cache lines per instruction and a threadgroup's 128 KB working set
    // evicted from L1 before the later `ki` iterations consumed it — measured
    // at ~26 GB/s against ~91 GB/s on every other decode stage.
    //
    // Here each simdgroup owns a whole row and its 32 lanes cover the row's
    // 128 state floats as consecutive float4, so one load is one coalesced
    // 512 B line-aligned burst. This is the same decomposition the prefill
    // kernel `qw35_ssm_recurrent_step128_batch_rows_f32` already uses.
    // Loading `sv` once and reusing it for the write-back also drops the
    // second full pass the row-per-thread version needed: 3 state passes -> 2.
    //
    // Row assignment r = simd_group*ROWS_PER_SG + i covers both heads of this
    // qk group (r = 0..255). pair = r >> 7 is constant across a simdgroup
    // (simdgroups 0..3 -> pair 0, 4..7 -> pair 1), which keeps `h` and the
    // per-head gate scalars loop-invariant and preserves the partial8[4*pair +
    // 0..3] fold the per-head RMS below already expects.
    constexpr int ROWS_PER_SG = (2 * STATE_SIZE) / 8;  // 256 rows / 8 simdgroups

    const int sg_pair = int(simd_group) >> 2;
    const int sg_head = int(qk_group) + sg_pair * GROUPS;
    const bool sg_active = sg_head < num_v_heads;

    float y_sq = 0.0f;
    if (sg_active) {
        const float beta = qw35_sigmoid_f32(beta_raw[sg_head]);
        const float alpha = qw35_softplus_f32(alpha_raw[sg_head] + dt[sg_head]) * a[sg_head];
        const float gate = exp(alpha);
        const float4 kc = k_cache4[lane];
        const float4 qc = q_cache4[lane];

        for (int i = 0; i < ROWS_PER_SG; ++i) {
            const int r = int(simd_group) * ROWS_PER_SG + i;
            const int d_r = r & 127;

            const int c = V_OFF + sg_head * STATE_SIZE + d_r;
            const int v_state_base = c * 3;
            const int v_weight_base = c * 4;
            const float v_s0 = conv_state[v_state_base + 0];
            const float v_s1 = conv_state[v_state_base + 1];
            const float v_s2 = conv_state[v_state_base + 2];
            const float v_x = qkv[c];
            const float v_conv = qw35_silu_f32(v_s0 * conv_w[v_weight_base + 0]
                                             + v_s1 * conv_w[v_weight_base + 1]
                                             + v_s2 * conv_w[v_weight_base + 2]
                                             + v_x  * conv_w[v_weight_base + 3]);
            // Every lane recomputes v_conv from the same broadcast loads; only
            // one lane may retire the conv window shift.
            if (lane == 0) {
                conv_state[v_state_base + 0] = v_s1;
                conv_state[v_state_base + 1] = v_s2;
                conv_state[v_state_base + 2] = v_x;
            }

            device float4 *s4 = (device float4 *)(
                state + sg_head * STATE_SIZE * head_v_dim + d_r * STATE_SIZE);
            const float4 sv = s4[lane];
            const float k_dot = simd_sum(dot(sv, kc));
            const float q_dot = simd_sum(dot(sv, qc));

            const float delta = (v_conv - gate * k_dot) * beta;
            s4[lane] = sv * gate + kc * delta;

            const float y = (gate * q_dot + delta * kq_dot) * scale;
            if (lane == 0) y_cache[r] = y;
            y_sq += y * y;
        }
    } else {
        for (int i = 0; i < ROWS_PER_SG; ++i) {
            if (lane == 0) y_cache[int(simd_group) * ROWS_PER_SG + i] = 0.0f;
        }
    }

    // Per-head RMS of y: simdgroups 0..3 hold head `pair 0`, 4..7 hold head
    // `pair 1`; each simdgroup already carries the sum of y^2 over its own
    // rows, so lane 0 just parks it and each thread folds its half's four
    // partials.
    if (lane == 0) partial8[simd_group] = y_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int pair = int(tid >> 7);
    const int h = int(qk_group) + pair * GROUPS;
    if (h < num_v_heads) {
        const float y_ss = partial8[4 * pair + 0] + partial8[4 * pair + 1]
                         + partial8[4 * pair + 2] + partial8[4 * pair + 3];
        const float inv_y = rsqrt(y_ss / float(STATE_SIZE) + eps);
        const int idx = h * STATE_SIZE + d;
        gated[idx] = y_cache[tid] * inv_y * norm_w[d] * qw35_silu_f32(z[idx]);
    }
}

kernel void qw35_ssm_recurrent_step128_batch_rows_f32(
    device const float * q_rep [[buffer(0)]],
    device const float * k_rep [[buffer(1)]],
    device const float * v [[buffer(2)]],
    device const float * beta_raw [[buffer(3)]],
    device const float * alpha_raw [[buffer(4)]],
    device const float * dt [[buffer(5)]],
    device const float * a [[buffer(6)]],
    device float * state [[buffer(7)]],
    device float * core [[buffer(8)]],
    constant int &num_v_heads [[buffer(9)]],
    constant int &head_v_dim [[buffer(10)]],
    constant float &scale [[buffer(11)]],
    constant int &n_tokens [[buffer(12)]],
    constant int &v_stride [[buffer(13)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 ti [[thread_position_in_threadgroup]]
) {
    constexpr int STATE_SIZE = 128;
    constexpr int ROWS_PER_GROUP = 4;
    constexpr int STATE_PER_THREAD = 4;

    const int h = int(tg.y);
    const int row = int(tg.x) * ROWS_PER_GROUP + int(ti.y);
    const int tx = int(ti.x);
    if (h >= num_v_heads || head_v_dim != STATE_SIZE) return;

    const bool active = row < head_v_dim;
    const int value_dim = num_v_heads * head_v_dim;
    const int state_base = h * STATE_SIZE * head_v_dim + row * STATE_SIZE;

    float ls[STATE_PER_THREAD];
    for (int j = 0; j < STATE_PER_THREAD; ++j) {
        const int is = tx * STATE_PER_THREAD + j;
        ls[j] = active ? state[state_base + is] : 0.0f;
    }

    const float dt_h = dt[h];
    const float a_h = a[h];

    for (int token = 0; token < n_tokens; ++token) {
        const int qk_base = token * value_dim + (h & 15) * STATE_SIZE;
        const int v_base = token * v_stride + h * head_v_dim;
        const int core_base = token * value_dim + h * head_v_dim;
        const float beta = qw35_sigmoid_f32(beta_raw[token * num_v_heads + h]);
        const float alpha = qw35_softplus_f32(alpha_raw[token * num_v_heads + h] + dt_h) * a_h;
        const float gate = exp(alpha);

        float k_vals[STATE_PER_THREAD];
        float q_vals[STATE_PER_THREAD];
        for (int j = 0; j < STATE_PER_THREAD; ++j) {
            const int is = tx * STATE_PER_THREAD + j;
            k_vals[j] = k_rep[qk_base + is];
            q_vals[j] = q_rep[qk_base + is];
        }

        float s_k = 0.0f;
        for (int j = 0; j < STATE_PER_THREAD; ++j) {
            ls[j] *= gate;
            s_k += ls[j] * k_vals[j];
        }
        s_k = simd_sum(s_k);

        const float delta = active ? (v[v_base + row] - s_k) * beta : 0.0f;
        float y = 0.0f;
        for (int j = 0; j < STATE_PER_THREAD; ++j) {
            ls[j] += k_vals[j] * delta;
            y += ls[j] * q_vals[j];
        }
        y = simd_sum(y);

        if (active && tx == 0) {
            core[core_base + row] = y * scale;
        }
    }

    if (active) {
        for (int j = 0; j < STATE_PER_THREAD; ++j) {
            const int is = tx * STATE_PER_THREAD + j;
            state[state_base + is] = ls[j];
        }
    }
}

kernel void qw35_ssm_gate_norm_batch_f32(
    device const float * core [[buffer(0)]],
    device const float * z [[buffer(1)]],
    device const float * norm_w [[buffer(2)]],
    device float * gated [[buffer(3)]],
    constant int &num_v_heads [[buffer(4)]],
    constant int &head_v_dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    constant int &n_tokens [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const int token = int(tg.x);
    const int head = int(tg.y);
    if (token >= n_tokens || head >= num_v_heads) return;

    threadgroup float sums[8];
    const int value_dim = num_v_heads * head_v_dim;
    const int base = token * value_dim + head * head_v_dim;
    float ss = 0.0f;
    for (int i = int(tid); i < head_v_dim; i += 256) {
        const float x = core[base + i];
        ss += x * x;
    }
    const float total = qw35_threadgroup_sum_256(ss, sums, tid, lane, simd_group);
    const float inv = rsqrt(total / float(head_v_dim) + eps);
    for (int i = int(tid); i < head_v_dim; i += 256) {
        const int idx = base + i;
        gated[idx] = core[idx] * inv * norm_w[i] * qw35_silu_f32(z[idx]);
    }
}
