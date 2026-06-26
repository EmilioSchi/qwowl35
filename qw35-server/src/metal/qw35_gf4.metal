// GF4 FFN kernels (calm-style 4-bit groups of 8: one fp8 e5m2 scale plus
// eight 3-bit values per uint32, prepared/xor-packed by the sidecar cooker).
//
// GF4 weights come from the cooked sidecar next to the GGUF; they replace the
// Q4_K/Q6_K FFN weights on the single-token decode path because the packed
// dot needs no K-quant block scale/min unpacking in the hot loop.

static inline float qw35_gf4_dot8_vec_f32(float4 x0, float4 x1, uint32_t word) {
    const int packed = int((word & 0xfff00000u) | ((word >> 4) & 0x0000fff0u));
    const float scale = float(as_type<half>(ushort(word << 8))) * -0.25f * 0.0001220703125f;

    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) {
        int shifted = packed << (9 - i * 3);
        if (i != 0) shifted &= 0xe000e000;
        const short2 q = as_type<short2>(shifted);
        acc += float(q.x) * x0[i] + float(q.y) * x1[i];
    }
    return acc * scale;
}

static inline float qw35_gf4_dot8_f32(device const float * x, int64_t c, uint32_t word) {
    const float4 x0 = *(device const float4 *)(x + c);
    const float4 x1 = *(device const float4 *)(x + c + 4);
    return qw35_gf4_dot8_vec_f32(x0, x1, word);
}

// --- Tiled (multi-token prefill) support -----------------------------------
// The unified .qw35 stores the FFN as GF4, so the prompt/prefill tiled matmul
// must dequantize GF4 just like the K-quant tiled path. We expose GF4 as a
// 256-element "super-block" (32 packed words) so it drops into the generic
// qw35_kernel_mul_mm template with nl=16 — identical pointer math to Q4_K,
// only the per-sub-block dequant differs. Row stride is (cols/8)*4 bytes =
// sizeof(qw35_block_gf4) per 256 columns, with no per-block header.
struct qw35_block_gf4 {
    uint32_t words[32];
};

// Dequantize sub-block `il` (0..15) = 16 contiguous weights = two GF4 words,
// reusing the exact bit/sign extraction of qw35_gf4_dot8_vec_f32 above.
template <typename type4x4>
void qw35_dequantize_gf4(device const qw35_block_gf4 *xb, short il, thread type4x4 &reg) {
    device const uint32_t *ws = xb->words + 2 * il;
    for (short hw = 0; hw < 2; ++hw) {
        const uint32_t word = ws[hw];
        const int packed = int((word & 0xfff00000u) | ((word >> 4) & 0x0000fff0u));
        const float scale = float(as_type<half>(ushort(word << 8))) * -0.25f * 0.0001220703125f;
        for (short i = 0; i < 4; ++i) {
            int shifted = packed << (9 - i * 3);
            if (i != 0) shifted &= 0xe000e000;
            const short2 q = as_type<short2>(shifted);
            const short e0 = hw * 8 + i;      // weights 0..3 of this word
            const short e1 = hw * 8 + i + 4;  // weights 4..7 of this word
            reg[e0 / 4][e0 % 4] = half(float(q.x) * scale);
            reg[e1 / 4][e1 % 4] = half(float(q.y) * scale);
        }
    }
}

// Fused single-token FFN gate+up matvec with SwiGLU applied in-kernel:
// swiglu_dst[row] = silu(gate_row . y) * (up_row . y).
kernel void qw35_ffn_gate_up_swiglu_gf4_f32(
    device const uint32_t * gate_w [[buffer(0)]],
    device const uint32_t * up_w [[buffer(1)]],
    device const float * y [[buffer(2)]],
    device float * swiglu_dst [[buffer(3)]],
    constant int64_t &n_groups [[buffer(4)]],
    constant int64_t &k [[buffer(5)]],
    constant int64_t &rows_per_token [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 ti [[thread_position_in_threadgroup]]
) {
    const int64_t row = int64_t(tg.x) * QW35_DECODE_ROWS_PER_GROUP + int64_t(ti.y);
    if (row >= rows_per_token) return;

    const int lane = int(ti.x);
    const int groups = int(n_groups);
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    device const uint32_t * gate_row = gate_w + row * groups;
    device const uint32_t * up_row = up_w + row * groups;

    for (int g = lane; g < groups; g += 32) {
        const uint32_t gate_word = gate_row[g];
        const uint32_t up_word = up_row[g];
        const int64_t c = int64_t(g) * 8;
        gate_sum += qw35_gf4_dot8_f32(y, c, gate_word);
        up_sum += qw35_gf4_dot8_f32(y, c, up_word);
    }

    gate_sum = simd_sum(gate_sum);
    up_sum = simd_sum(up_sum);
    if (lane == 0) {
        swiglu_dst[row] = qw35_silu_f32(gate_sum) * up_sum;
    }
}

// Single-token GF4 matvec (dst[row] = sum).
kernel void qw35_decode_matmul_gf4_2row_f32(
    device const uint32_t * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant int64_t &n_groups [[buffer(3)]],
    constant int64_t &k [[buffer(4)]],
    constant int64_t &rows_per_token [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    (void)k;
    constexpr int rows_per_simdgroup = 2;
    constexpr int simdgroups_per_tg = 2;
    const int first_row = (int(tg.x) * simdgroups_per_tg + int(simd_group)) * rows_per_simdgroup;
    const int groups = int(n_groups);
    const int rows = int(rows_per_token);

    float sumf[rows_per_simdgroup] = {0.0f, 0.0f};
    for (int g = int(lane); g < groups; g += 32) {
        const int col = g * 8;
        const float4 y0 = *(device const float4 *)(y + col);
        const float4 y1 = *(device const float4 *)(y + col + 4);
        for (short row = 0; row < rows_per_simdgroup; ++row) {
            const int out_row = first_row + row;
            if (out_row >= rows) continue;
            const uint32_t word = x[out_row * groups + g];
            sumf[row] += qw35_gf4_dot8_vec_f32(y0, y1, word);
        }
    }

    for (short row = 0; row < rows_per_simdgroup; ++row) {
        const int out_row = first_row + row;
        if (out_row >= rows) continue;
        const float sum = simd_sum(sumf[row]);
        if (lane == 0) dst[out_row] = sum;
    }
}

// Fused GF4 output-head matvec + argmax over 16 vocab rows per threadgroup;
// pairs with qw35_output_argmax_reduce_partials_f32.
kernel void qw35_output_gf4_argmax_partials_16row_f32(
    device const uint32_t * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device uint * partial_token [[buffer(2)]],
    device float * partial_logit [[buffer(3)]],
    constant int64_t &n_groups [[buffer(4)]],
    constant int64_t &k [[buffer(5)]],
    constant int64_t &rows_per_token [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    (void)k;
    constexpr int NR0 = 4;
    constexpr int simdgroups_per_tg = 4;
    threadgroup float local_vals[16];
    threadgroup uint local_ids[16];

    const int groups = int(n_groups);
    const int rows = int(rows_per_token);
    const int first_row = (int(tg) * simdgroups_per_tg + int(simd_group)) * NR0;

    float sums[NR0] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int g = int(lane); g < groups; g += 32) {
        const int col = g * 8;
        const float4 y0 = *(device const float4 *)(y + col);
        const float4 y1 = *(device const float4 *)(y + col + 4);
        for (short row = 0; row < NR0; ++row) {
            const int r = first_row + int(row);
            if (r >= rows) continue;
            sums[row] += qw35_gf4_dot8_vec_f32(y0, y1, x[r * groups + g]);
        }
    }

    for (short row = 0; row < NR0; ++row) {
        sums[row] = simd_sum(sums[row]);
    }
    if (lane == 0) {
        const uint base = uint(first_row);
        const uint slot = uint(simd_group) * uint(NR0);
        for (uint row = 0; row < uint(NR0); ++row) {
            if (first_row + int(row) < rows) {
                local_vals[slot + row] = sums[row];
                local_ids[slot + row] = base + row;
            } else {
                local_vals[slot + row] = -INFINITY;
                local_ids[slot + row] = 0xffffffffu;
            }
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

// Single-token GF4 matvec with the residual add folded in (dst[row] += sum).
kernel void qw35_decode_matmul_gf4_2row_residual_f32(
    device const uint32_t * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant int64_t &n_groups [[buffer(3)]],
    constant int64_t &k [[buffer(4)]],
    constant int64_t &rows_per_token [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    (void)k;
    constexpr int rows_per_simdgroup = 2;
    constexpr int simdgroups_per_tg = 2;
    const int first_row = (int(tg.x) * simdgroups_per_tg + int(simd_group)) * rows_per_simdgroup;
    const int groups = int(n_groups);
    const int rows = int(rows_per_token);

    float sumf[rows_per_simdgroup] = {0.0f, 0.0f};
    for (int g = int(lane); g < groups; g += 32) {
        const int col = g * 8;
        const float4 y0 = *(device const float4 *)(y + col);
        const float4 y1 = *(device const float4 *)(y + col + 4);
        for (short row = 0; row < rows_per_simdgroup; ++row) {
            const int out_row = first_row + row;
            if (out_row >= rows) continue;
            const uint32_t word = x[out_row * groups + g];
            sumf[row] += qw35_gf4_dot8_vec_f32(y0, y1, word);
        }
    }

    for (short row = 0; row < rows_per_simdgroup; ++row) {
        const int out_row = first_row + row;
        if (out_row >= rows) continue;
        const float sum = simd_sum(sumf[row]);
        if (lane == 0) dst[out_row] += sum;
    }
}
