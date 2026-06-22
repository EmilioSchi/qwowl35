// Fused output-head argmax kernels (greedy decode never materializes logits).
kernel void qw35_output_q6_k_argmax_partials_16row_f32(
    device const qw35_block_q6_k * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device uint * partial_token [[buffer(2)]],
    device float * partial_logit [[buffer(3)]],
    constant int64_t &n_blocks [[buffer(4)]],
    constant int64_t &k [[buffer(5)]],
    constant int64_t &rows_per_token [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    (void)k;
    constexpr int NR0 = 4;
    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    constexpr int simdgroups_per_tg = 4;
    threadgroup float local_vals[16];
    threadgroup uint local_ids[16];

    const int nb = int(n_blocks);
    const int rows = int(rows_per_token);
    const int first_row = (int(tg) * simdgroups_per_tg + int(simd_group)) * NR0;
    const short tid = lane / 2;
    const short ix = lane % 2;
    const short ip = tid / 8;
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for (int b = ix; b < nb; b += 2) {
        float yl[16];
        device const float *yy = y + b * 256 + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4 * l + 0] = yy[l + 0];
            yl[4 * l + 1] = yy[l + 32];
            yl[4 * l + 2] = yy[l + 64];
            yl[4 * l + 3] = yy[l + 96];
        }

        for (short row = 0; row < NR0; ++row) {
            const int dst_row = first_row + int(row);
            if (dst_row >= rows) continue;

            device const qw35_block_q6_k &blk = x[dst_row * nb + b];
            device const uchar *q1 = blk.ql + q_offset_l;
            device const uchar *q2 = q1 + 32;
            device const uchar *qh = blk.qh + q_offset_h;
            device const char *sc = blk.scales + is;
            const float d = float(blk.d);

            float4 sums = {0.0f, 0.0f, 0.0f, 0.0f};
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4 * l + 0] * float(int((q1[l] & 0x0F) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4 * l + 1] * float(int((q2[l] & 0x0F) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4 * l + 2] * float(int((q1[l] >> 4)   | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4 * l + 3] * float(int((q2[l] >> 4)   | ((qh[l] & kmask4) >> 2)) - 32);
            }
            const float partial = d * (sums[0] * float(sc[0])
                                     + sums[1] * float(sc[2])
                                     + sums[2] * float(sc[4])
                                     + sums[3] * float(sc[6]));
            if (row == 0) {
                sum0 += partial;
            } else if (row == 1) {
                sum1 += partial;
            } else if (row == 2) {
                sum2 += partial;
            } else {
                sum3 += partial;
            }
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);
    if (lane == 0) {
        const uint base = uint(first_row);
        const uint slot = uint(simd_group) * 4u;
        if (first_row < rows) {
            local_vals[slot] = sum0;
            local_ids[slot] = base;
        } else {
            local_vals[slot] = -INFINITY;
            local_ids[slot] = 0xffffffffu;
        }
        if (first_row + 1 < rows) {
            local_vals[slot + 1u] = sum1;
            local_ids[slot + 1u] = base + 1u;
        } else {
            local_vals[slot + 1u] = -INFINITY;
            local_ids[slot + 1u] = 0xffffffffu;
        }
        if (first_row + 2 < rows) {
            local_vals[slot + 2u] = sum2;
            local_ids[slot + 2u] = base + 2u;
        } else {
            local_vals[slot + 2u] = -INFINITY;
            local_ids[slot + 2u] = 0xffffffffu;
        }
        if (first_row + 3 < rows) {
            local_vals[slot + 3u] = sum3;
            local_ids[slot + 3u] = base + 3u;
        } else {
            local_vals[slot + 3u] = -INFINITY;
            local_ids[slot + 3u] = 0xffffffffu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && lane == 0) {
        float best = local_vals[0];
        uint best_id = local_ids[0];
        for (uint i = 1; i < 16; ++i) {
            const float value = local_vals[i];
            const uint id = local_ids[i];
            if (value > best || (value == best && id < best_id)) {
                best = value;
                best_id = id;
            }
        }
        partial_token[tg] = best_id;
        partial_logit[tg] = best;
    }
}

kernel void qw35_output_argmax_reduce_partials_f32(
    device const uint * partial_token [[buffer(0)]],
    device const float * partial_logit [[buffer(1)]],
    device uint * out_token [[buffer(2)]],
    device float * out_logit [[buffer(3)]],
    constant int64_t &n_groups [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float best_vals[256];
    threadgroup uint best_ids[256];
    const int groups = int(n_groups);
    float best = -INFINITY;
    uint best_id = 0xffffffffu;
    for (int i = int(tid); i < groups; i += 256) {
        const float value = partial_logit[i];
        const uint id = partial_token[i];
        if (value > best || (value == best && id < best_id)) {
            best = value;
            best_id = id;
        }
    }
    best_vals[tid] = best;
    best_ids[tid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float rhs = best_vals[tid + stride];
            const uint rhs_id = best_ids[tid + stride];
            if (rhs > best_vals[tid] || (rhs == best_vals[tid] && rhs_id < best_ids[tid])) {
                best_vals[tid] = rhs;
                best_ids[tid] = rhs_id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out_token[0] = best_ids[0];
        out_logit[0] = best_vals[0];
    }
}
