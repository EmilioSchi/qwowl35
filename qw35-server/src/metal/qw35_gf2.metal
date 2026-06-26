// GF2 FFN kernels: 2-bit sub-4-bit codec (groups of 16). One uint32 holds 16
// two-bit codes (code i in bits 2*i..2*i+1); a separate fp8(e5m2) scale plane
// gives one byte per group. Dequant level = (q*2/3 - 1)*scale for q in 0..3,
// i.e. scale*{-1,-1/3,+1/3,+1}. 2.5 bpw vs GF4's 4. Cooked by the sidecar tool
// with --codec gf2; applied to FFN tensors on the single-token decode path.
//
// Buffer layout for a tensor (one mmap region): all code words
// [rows * groups] uint32, immediately followed by all scale bytes
// [rows * groups] uchar. The matvec binds codes at buffer(0) and scales at
// buffer(1) (same MTLBuffer, scales offset = rows*groups*4).

static inline float qw35_gf2_dot16(device const float * y,
                                   int64_t col,
                                   uint32_t word,
                                   uchar scale_byte) {
    const float scale = float(as_type<half>(ushort(ushort(scale_byte) << 8)));
    float acc = 0.0f;
    for (int j = 0; j < 4; ++j) {
        const float4 yv = *(device const float4 *)(y + col + j * 4);
        for (int t = 0; t < 4; ++t) {
            const int i = j * 4 + t;
            const uint q = (word >> uint(2 * i)) & 0x3u;
            acc += (float(q) * (2.0f / 3.0f) - 1.0f) * yv[t];
        }
    }
    return acc * scale;
}

// Single-token GF2 matvec (dst[row] = sum). 2 rows per simdgroup, 2 simdgroups
// per threadgroup — mirrors the GF4 matvec tiling.
kernel void qw35_decode_matmul_gf2_2row_f32(
    device const uint32_t * codes [[buffer(0)]],
    device const uchar * scales [[buffer(1)]],
    device const float * y [[buffer(2)]],
    device float * dst [[buffer(3)]],
    constant int64_t & n_groups [[buffer(4)]],
    constant int64_t & k [[buffer(5)]],
    constant int64_t & rows_per_token [[buffer(6)]],
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
        const int col = g * 16;
        for (short row = 0; row < rows_per_simdgroup; ++row) {
            const int out_row = first_row + row;
            if (out_row >= rows) continue;
            const uint32_t word = codes[out_row * groups + g];
            const uchar sb = scales[out_row * groups + g];
            sumf[row] += qw35_gf2_dot16(y, col, word, sb);
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
    device const uint32_t * codes [[buffer(0)]],
    device const uchar * scales [[buffer(1)]],
    device const float * y [[buffer(2)]],
    device float * dst [[buffer(3)]],
    constant int64_t & n_groups [[buffer(4)]],
    constant int64_t & k [[buffer(5)]],
    constant int64_t & rows_per_token [[buffer(6)]],
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
        const int col = g * 16;
        for (short row = 0; row < rows_per_simdgroup; ++row) {
            const int out_row = first_row + row;
            if (out_row >= rows) continue;
            const uint32_t word = codes[out_row * groups + g];
            const uchar sb = scales[out_row * groups + g];
            sumf[row] += qw35_gf2_dot16(y, col, word, sb);
        }
    }

    for (short row = 0; row < rows_per_simdgroup; ++row) {
        const int out_row = first_row + row;
        if (out_row >= rows) continue;
        const float sum = simd_sum(sumf[row]);
        if (lane == 0) dst[out_row] += sum;
    }
}
