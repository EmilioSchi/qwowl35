// GF2 FFN kernels: 2-bit codec (groups of 16). One uint32 holds 16 two-bit
// codes (code i in bits 2*i..2*i+1); one fp8(e5m2) scale byte per group.
// Dequant level = (q*2/3 - 1)*scale for q in 0..3, i.e. scale*{-1,-1/3,+1/3,+1}.
// 2.5 bpw vs GF4's 4.
//
// Storage is an interleaved 256-element "super-block" (like GF4's, but with
// the scales carried alongside the codes): 16 code words followed by their 16
// scale bytes = 80 bytes per 256 weights. One contiguous self-describing
// stream per tensor — row stride is (cols/256)*80 bytes — so the same block
// walks work for the single-token decode matvec, the generic tiled prefill
// template (qw35_mul_mm_gf2_f32, nl=16), and a single coalesced buffer bind
// (the previous split codes/scales planes cost a second far load per group).
struct qw35_block_gf2 {
    uint32_t codes[16];
    uchar scales[16];
};

static inline float qw35_gf2_scale(uchar scale_byte) {
    return float(as_type<half>(ushort(ushort(scale_byte) << 8)));
}

static inline float qw35_gf2_dot16(device const float * y,
                                   int64_t col,
                                   uint32_t word,
                                   uchar scale_byte) {
    float acc = 0.0f;
    for (int j = 0; j < 4; ++j) {
        const float4 yv = *(device const float4 *)(y + col + j * 4);
        for (int t = 0; t < 4; ++t) {
            const int i = j * 4 + t;
            const uint q = (word >> uint(2 * i)) & 0x3u;
            acc += (float(q) * (2.0f / 3.0f) - 1.0f) * yv[t];
        }
    }
    return acc * qw35_gf2_scale(scale_byte);
}

// Dequantize sub-block `il` (0..15) = 16 contiguous weights = one GF2 word +
// its scale byte, for the generic tiled mul_mm template (mirrors
// qw35_dequantize_gf4).
template <typename type4x4>
void qw35_dequantize_gf2(device const qw35_block_gf2 *xb, short il, thread type4x4 &reg) {
    const uint32_t word = xb->codes[il];
    const float scale = qw35_gf2_scale(xb->scales[il]);
    for (short i = 0; i < 16; ++i) {
        const uint q = (word >> uint(2 * i)) & 0x3u;
        reg[i / 4][i % 4] = half((float(q) * (2.0f / 3.0f) - 1.0f) * scale);
    }
}

// Single-token GF2 matvec (dst[row] = sum). 2 rows per simdgroup, 2 simdgroups
// per threadgroup — mirrors the GF4 matvec tiling and buffer order.
kernel void qw35_decode_matmul_gf2_2row_f32(
    device const qw35_block_gf2 * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant int64_t & n_groups [[buffer(3)]],
    constant int64_t & k [[buffer(4)]],
    constant int64_t & rows_per_token [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    (void)k;
    constexpr int rows_per_simdgroup = 2;
    constexpr int simdgroups_per_tg = 2;
    const int first_row = (int(tg.x) * simdgroups_per_tg + int(simd_group)) * rows_per_simdgroup;
    const int groups = int(n_groups);
    const int blocks_per_row = groups / 16;
    const int rows = int(rows_per_token);

    float sumf[rows_per_simdgroup] = {0.0f, 0.0f};
    for (int g = int(lane); g < groups; g += 32) {
        const int col = g * 16;
        for (short row = 0; row < rows_per_simdgroup; ++row) {
            const int out_row = first_row + row;
            if (out_row >= rows) continue;
            device const qw35_block_gf2 * blk = x + out_row * blocks_per_row + (g >> 4);
            sumf[row] += qw35_gf2_dot16(y, col, blk->codes[g & 15], blk->scales[g & 15]);
        }
    }

    for (short row = 0; row < rows_per_simdgroup; ++row) {
        const int out_row = first_row + row;
        if (out_row >= rows) continue;
        const float sum = simd_sum(sumf[row]);
        if (lane == 0) dst[out_row] = sum;
    }
}

// Single-token GF2 matvec with the residual add folded in (dst[row] += sum).
kernel void qw35_decode_matmul_gf2_2row_residual_f32(
    device const qw35_block_gf2 * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant int64_t & n_groups [[buffer(3)]],
    constant int64_t & k [[buffer(4)]],
    constant int64_t & rows_per_token [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    (void)k;
    constexpr int rows_per_simdgroup = 2;
    constexpr int simdgroups_per_tg = 2;
    const int first_row = (int(tg.x) * simdgroups_per_tg + int(simd_group)) * rows_per_simdgroup;
    const int groups = int(n_groups);
    const int blocks_per_row = groups / 16;
    const int rows = int(rows_per_token);

    float sumf[rows_per_simdgroup] = {0.0f, 0.0f};
    for (int g = int(lane); g < groups; g += 32) {
        const int col = g * 16;
        for (short row = 0; row < rows_per_simdgroup; ++row) {
            const int out_row = first_row + row;
            if (out_row >= rows) continue;
            device const qw35_block_gf2 * blk = x + out_row * blocks_per_row + (g >> 4);
            sumf[row] += qw35_gf2_dot16(y, col, blk->codes[g & 15], blk->scales[g & 15]);
        }
    }

    for (short row = 0; row < rows_per_simdgroup; ++row) {
        const int out_row = first_row + row;
        if (out_row >= rows) continue;
        const float sum = simd_sum(sumf[row]);
        if (lane == 0) dst[out_row] += sum;
    }
}
