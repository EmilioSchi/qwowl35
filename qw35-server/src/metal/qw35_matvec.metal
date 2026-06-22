// Quantized matvec/matmul kernels for the live Q4_K_M tensor set.
static inline float qw35_dequant_q4_k_at(device const qw35_block_q4_k &blk, int local_idx) {
    const float d = float(blk.d);
    const float dmin = float(blk.dmin);
    const int group = local_idx / 32;
    const int lane = local_idx & 31;
    const int q_index = (group / 2) * 32 + lane;
    uchar sc, m;
    qw35_get_scale_min_k4(group, blk.scales, sc, m);
    const int q = (group & 1) == 0 ? int(blk.qs[q_index] & 0x0f) : int(blk.qs[q_index] >> 4);
    return d * float(sc) * float(q) - dmin * float(m);
}

static inline uchar2 qw35_get_scale_min_k4_pair(int j, int k, device const uchar *q) {
    return j < 4 ? uchar2{uchar(q[j + k] & 63), uchar(q[j + 4 + k] & 63)}
                 : uchar2{uchar((q[j + 4 + k] & 0x0f) | ((q[j - 4 + k] & 0xc0) >> 2)),
                          uchar((q[j + 4 + k] >> 4) | ((q[j + k] & 0xc0) >> 2))};
}

template <typename type4x4>
void qw35_dequantize_q4_k(device const qw35_block_q4_k *xb, short il, thread type4x4 &reg) {
    device const uchar *q = xb->qs;

    const short is = (il / 4) * 2;
    q += (il / 4) * 32 + 16 * (il & 1);
    il &= 3;

    const uchar2 sc = qw35_get_scale_min_k4_pair(is, il / 2, xb->scales);
    const float d = float(xb->d) * (il < 2 ? 1.0f : 1.0f / 16.0f);
    const float dmin = float(xb->dmin);
    const float dl = d * float(sc[0]);
    const float ml = dmin * float(sc[1]);
    const ushort mask = il < 2 ? 0x0f : 0xf0;

    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * float(q[i] & mask) - ml;
    }
}

template <typename type4x4>
void qw35_dequantize_q5_k(device const qw35_block_q5_k *xb, short il, thread type4x4 &reg) {
    device const uchar *q = xb->qs;
    device const uchar *qh = xb->qh;

    const short is = (il / 4) * 2;
    q += 32 * (il / 4) + 16 * (il & 1);
    qh += 16 * (il & 1);
    const uchar high_mask = uchar(1u << uint(il / 2));
    il &= 3;

    const uchar2 sc = qw35_get_scale_min_k4_pair(is, il / 2, xb->scales);
    const float d = float(xb->d) * (il < 2 ? 1.0f : 1.0f / 16.0f);
    const float dmin = float(xb->dmin);
    const float dl = d * float(sc[0]);
    const float ml = dmin * float(sc[1]);
    const ushort low_mask = il < 2 ? 0x0f : 0xf0;
    const float high_value = il < 2 ? 16.0f : 256.0f;

    for (int i = 0; i < 16; ++i) {
        const float qv = float(q[i] & low_mask) + ((qh[i] & high_mask) ? high_value : 0.0f);
        reg[i / 4][i % 4] = dl * qv - ml;
    }
}

template <typename type4x4>
void qw35_dequantize_q6_k(device const qw35_block_q6_k *xb, short il, thread type4x4 &reg) {
    const float d_all = float(xb->d);
    device const ushort *ql = (device const ushort *)xb->ql;
    device const ushort *qh = (device const ushort *)xb->qh;
    device const char *scales = xb->scales;

    ql += 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
    qh += 16 * (il / 8) + 8 * (il & 1);
    const float sc = float(scales[(il % 2) + 2 * (il / 2)]);
    il = (il / 2) & 3;

    const uint kmask1 = il > 1 ? (il > 2 ? 0xc0c0c0c0u : 0x30303030u)
                               : (il > 0 ? 0x0c0c0c0cu : 0x03030303u);
    const uint kmask2 = il > 1 ? 0xf0f0f0f0u : 0x0f0f0f0fu;
    const float ml = d_all * sc * 32.0f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.0f;
    const float dl2 = dl0 / (256.0f * 256.0f);
    const float dl3 = dl0 / (256.0f * 256.0f * 256.0f);
    const uchar shr_h = il > 2 ? 2 : 0;
    const uchar shl_h = il > 1 ? 0 : (il > 0 ? 2 : 4);
    const uchar shr_l = il > 1 ? 4 : 0;

    for (int i = 0; i < 4; ++i) {
        const uint low = (uint(ql[2 * i]) | (uint(ql[2 * i + 1]) << 16)) & kmask2;
        const uint high = (uint(qh[2 * i]) | (uint(qh[2 * i + 1]) << 16)) & kmask1;
        const uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 * float(q & 0x000000ffu) - ml;
        reg[i][1] = dl1 * float(q & 0x0000ff00u) - ml;
        reg[i][2] = dl2 * float(q & 0x00ff0000u) - ml;
        reg[i][3] = dl3 * float(q & 0xff000000u) - ml;
    }
}

kernel void qw35_get_row_q4_k_f32(
    device const qw35_block_q4_k * weight [[buffer(0)]],
    device float * dst [[buffer(1)]],
    constant uint &row [[buffer(2)]],
    constant int64_t &k [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (int64_t(gid) >= k) return;
    const int64_t n_blocks = (k + 255) / 256;
    const int64_t block_idx = int64_t(row) * n_blocks + int64_t(gid) / 256;
    const int local_idx = int(int64_t(gid) & 255);
    dst[gid] = qw35_dequant_q4_k_at(weight[block_idx], local_idx);
}

kernel void qw35_get_rows_q4_k_f32(
    device const qw35_block_q4_k * weight [[buffer(0)]],
    device const uint * rows [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant int64_t &k [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const int64_t total = k;
    if (total <= 0) return;
    const int64_t token_idx = int64_t(gid) / total;
    const int64_t col = int64_t(gid) - token_idx * total;
    if (col >= total) return;
    const uint row = rows[token_idx];
    const int64_t n_blocks = (k + 255) / 256;
    const int64_t block_idx = int64_t(row) * n_blocks + col / 256;
    const int local_idx = int(col & 255);
    dst[gid] = qw35_dequant_q4_k_at(weight[block_idx], local_idx);
}

kernel void qw35_decode_matmul_q4_k_2row_f32(
    device const qw35_block_q4_k * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant int64_t &n_blocks [[buffer(3)]],
    constant int64_t &k [[buffer(4)]],
    constant int64_t &rows_per_token [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    (void)k;
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;
    constexpr int rows_per_simdgroup = 2;
    constexpr int simdgroups_per_tg = 2;

    const int ix = int(lane) / 8;
    const int it = int(lane) % 8;
    const int iq = it / 4;
    const int ir = it % 4;
    const int first_row = (int(tg.x) * simdgroups_per_tg + int(simd_group)) * rows_per_simdgroup;
    const int nb = int(n_blocks);
    const int rows = int(rows_per_token);

    float yl[16];
    float yh[16];
    float sumf[rows_per_simdgroup] = {0.0f, 0.0f};
    ushort sc16[4];
    thread const uchar *sc8 = (thread const uchar *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        const int y_base = ib * 256 + 64 * iq + 8 * ir;
        float4 sumy = {0.0f, 0.0f, 0.0f, 0.0f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y[y_base + i + 0];
            sumy[0] += yl[i + 0];
            yl[i + 8] = y[y_base + i + 32];
            sumy[1] += yl[i + 8];
            yh[i + 0] = y[y_base + i + 128];
            sumy[2] += yh[i + 0];
            yh[i + 8] = y[y_base + i + 160];
            sumy[3] += yh[i + 8];
        }

        for (short row = 0; row < rows_per_simdgroup; ++row) {
            const int out_row = first_row + row;
            if (out_row >= rows) continue;
            device const qw35_block_q4_k &blk = x[out_row * nb + ib];
            device const ushort *sc = (device const ushort *)blk.scales + iq;
            device const ushort *q1 = (device const ushort *)blk.qs + 16 * iq + 4 * ir;
            device const half *dh = &blk.d;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const ushort *q2 = q1 + 32;
            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};

            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * float(q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * float(q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * float(q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * float(q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * float(q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * float(q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * float(q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * float(q2[i] & 0xF000);
            }

            sumf[row] += float(dh[0]) * ((acc1[0] + (1.0f / 256.0f) * acc1[1]) * float(sc8[0]) +
                                         (acc1[2] + (1.0f / 256.0f) * acc1[3]) * float(sc8[1]) * (1.0f / 16.0f) +
                                         (acc2[0] + (1.0f / 256.0f) * acc2[1]) * float(sc8[4]) +
                                         (acc2[2] + (1.0f / 256.0f) * acc2[3]) * float(sc8[5]) * (1.0f / 16.0f)) -
                         float(dh[1]) * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                                         sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
        }
    }

    for (short row = 0; row < rows_per_simdgroup; ++row) {
        const int out_row = first_row + row;
        if (out_row >= rows) continue;
        const float sum = simd_sum(sumf[row]);
        if (lane == 0) dst[out_row] = sum;
    }
}

kernel void qw35_decode_matmul_q4_k_2row_residual_f32(
    device const qw35_block_q4_k * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant int64_t &n_blocks [[buffer(3)]],
    constant int64_t &k [[buffer(4)]],
    constant int64_t &rows_per_token [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    (void)k;
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;
    constexpr int rows_per_simdgroup = 2;
    constexpr int simdgroups_per_tg = 2;

    const int ix = int(lane) / 8;
    const int it = int(lane) % 8;
    const int iq = it / 4;
    const int ir = it % 4;
    const int first_row = (int(tg.x) * simdgroups_per_tg + int(simd_group)) * rows_per_simdgroup;
    const int nb = int(n_blocks);
    const int rows = int(rows_per_token);

    float yl[16];
    float yh[16];
    float sumf[rows_per_simdgroup] = {0.0f, 0.0f};
    ushort sc16[4];
    thread const uchar *sc8 = (thread const uchar *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        const int y_base = ib * 256 + 64 * iq + 8 * ir;
        float4 sumy = {0.0f, 0.0f, 0.0f, 0.0f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y[y_base + i + 0];
            sumy[0] += yl[i + 0];
            yl[i + 8] = y[y_base + i + 32];
            sumy[1] += yl[i + 8];
            yh[i + 0] = y[y_base + i + 128];
            sumy[2] += yh[i + 0];
            yh[i + 8] = y[y_base + i + 160];
            sumy[3] += yh[i + 8];
        }

        for (short row = 0; row < rows_per_simdgroup; ++row) {
            const int out_row = first_row + row;
            if (out_row >= rows) continue;
            device const qw35_block_q4_k &blk = x[out_row * nb + ib];
            device const ushort *sc = (device const ushort *)blk.scales + iq;
            device const ushort *q1 = (device const ushort *)blk.qs + 16 * iq + 4 * ir;
            device const half *dh = &blk.d;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const ushort *q2 = q1 + 32;
            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};

            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * float(q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * float(q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * float(q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * float(q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * float(q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * float(q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * float(q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * float(q2[i] & 0xF000);
            }

            sumf[row] += float(dh[0]) * ((acc1[0] + (1.0f / 256.0f) * acc1[1]) * float(sc8[0]) +
                                         (acc1[2] + (1.0f / 256.0f) * acc1[3]) * float(sc8[1]) * (1.0f / 16.0f) +
                                         (acc2[0] + (1.0f / 256.0f) * acc2[1]) * float(sc8[4]) +
                                         (acc2[2] + (1.0f / 256.0f) * acc2[3]) * float(sc8[5]) * (1.0f / 16.0f)) -
                         float(dh[1]) * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                                         sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
        }
    }

    for (short row = 0; row < rows_per_simdgroup; ++row) {
        const int out_row = first_row + row;
        if (out_row >= rows) continue;
        const float sum = simd_sum(sumf[row]);
        if (lane == 0) dst[out_row] += sum;
    }
}

kernel void qw35_decode_matmul_q5_k_2row_f32(
    device const qw35_block_q5_k * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant int64_t &n_blocks [[buffer(3)]],
    constant int64_t &k [[buffer(4)]],
    constant int64_t &rows_per_token [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    (void)k;
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;
    constexpr int rows_per_simdgroup = 2;
    constexpr int simdgroups_per_tg = 2;

    const short tid = lane / 4;
    const short ix = lane % 4;
    const short iq = tid / 4;
    const short ir = tid % 4;
    const short l0 = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;
    const int first_row = (int(tg.x) * simdgroups_per_tg + int(simd_group)) * rows_per_simdgroup;
    const int nb = int(n_blocks);
    const int rows = int(rows_per_token);

    const uchar hm1 = uchar(1u << (2 * iq));
    const uchar hm2 = hm1 << 1;
    const uchar hm3 = hm1 << 4;
    const uchar hm4 = hm2 << 4;

    float sumf[rows_per_simdgroup] = {0.0f, 0.0f};
    for (int b = ix; b < nb; b += 4) {
        device const float *y1 = y + b * 256 + y_offset;
        device const float *y2 = y1 + 128;
        float yl[16];
        float yh[16];
        float4 sumy = {0.0f, 0.0f, 0.0f, 0.0f};
        for (short l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l + 0];
            yl[l + 8] = y1[l + 32];
            yh[l + 0] = y2[l + 0];
            yh[l + 8] = y2[l + 32];
            sumy[0] += yl[l + 0];
            sumy[1] += yl[l + 8];
            sumy[2] += yh[l + 0];
            sumy[3] += yh[l + 8];
        }

        for (short row = 0; row < rows_per_simdgroup; ++row) {
            const int out_row = first_row + int(row);
            if (out_row >= rows) continue;

            device const qw35_block_q5_k &blk = x[out_row * nb + b];
            device const uchar *q1 = blk.qs + q_offset;
            device const uchar *q2 = q1 + 64;
            device const uchar *qh = blk.qh + l0;
            device const half *dh = &blk.d;
            device const ushort *a = (device const ushort *)blk.scales + iq;

            ushort sc16[4];
            thread const uchar *sc8 = (thread const uchar *)sc16;
            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
            for (short l = 0; l < 8; ++l) {
                const uchar h = qh[l];
                acc1[0] += yl[l + 0] * float(q1[l] & 0x0F);
                acc1[1] += yl[l + 8] * float(q1[l] & 0xF0);
                acc1[2] += yh[l + 0] * float(q2[l] & 0x0F);
                acc1[3] += yh[l + 8] * float(q2[l] & 0xF0);
                acc2[0] += (h & hm1) ? yl[l + 0] : 0.0f;
                acc2[1] += (h & hm2) ? yl[l + 8] : 0.0f;
                acc2[2] += (h & hm3) ? yh[l + 0] : 0.0f;
                acc2[3] += (h & hm4) ? yh[l + 8] : 0.0f;
            }

            sumf[row] += float(dh[0]) * (float(sc8[0]) * (acc1[0] + 16.0f * acc2[0])
                                       + float(sc8[1]) * (acc1[1] / 16.0f + 16.0f * acc2[1])
                                       + float(sc8[4]) * (acc1[2] + 16.0f * acc2[2])
                                       + float(sc8[5]) * (acc1[3] / 16.0f + 16.0f * acc2[3]))
                       - float(dh[1]) * (sumy[0] * float(sc8[2])
                                       + sumy[1] * float(sc8[3])
                                       + sumy[2] * float(sc8[6])
                                       + sumy[3] * float(sc8[7]));
        }
    }

    for (short row = 0; row < rows_per_simdgroup; ++row) {
        const int out_row = first_row + int(row);
        if (out_row >= rows) continue;
        const float sum = simd_sum(sumf[row]);
        if (lane == 0) dst[out_row] = sum;
    }
}

kernel void qw35_decode_matmul_q6_k_llama_f32(
    device const qw35_block_q6_k * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant int64_t &n_blocks [[buffer(3)]],
    constant int64_t &k [[buffer(4)]],
    constant int64_t &rows_per_token [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    (void)k;
    constexpr int NR0 = 2;
    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const int nb = int(n_blocks);
    const int rows = int(rows_per_token);
    const int first_row = (int(tg.x) * 2 + int(simd_group)) * NR0;
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
            } else {
                sum1 += partial;
            }
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0) {
        if (first_row < rows) {
            dst[first_row] = sum0;
        }
        if (first_row + 1 < rows) {
            dst[first_row + 1] = sum1;
        }
    }
}

kernel void qw35_decode_matmul_q6_k_llama_residual_f32(
    device const qw35_block_q6_k * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant int64_t &n_blocks [[buffer(3)]],
    constant int64_t &k [[buffer(4)]],
    constant int64_t &rows_per_token [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    (void)k;
    constexpr int NR0 = 2;
    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const int nb = int(n_blocks);
    const int rows = int(rows_per_token);
    const int first_row = (int(tg.x) * 2 + int(simd_group)) * NR0;
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
            } else {
                sum1 += partial;
            }
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0) {
        if (first_row < rows) {
            dst[first_row] += sum0;
        }
        if (first_row + 1 < rows) {
            dst[first_row + 1] += sum1;
        }
    }
}

kernel void qw35_decode_matmul_q8_0_f32(
    device const qw35_block_q8_0 * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant int64_t &n_blocks [[buffer(3)]],
    constant int64_t &k [[buffer(4)]],
    constant int64_t &rows_per_token [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 ti [[thread_position_in_threadgroup]]
) {
    const int64_t row = int64_t(tg.x) * QW35_DECODE_ROWS_PER_GROUP + int64_t(ti.y);
    if (row >= rows_per_token) return;

    const int lane = int(ti.x);
    float sum = 0.0f;
    device const qw35_block_q8_0 *x_row = x + row * n_blocks;
    for (int64_t b = 0; b < n_blocks; ++b) {
        const int64_t col = b * 32 + lane;
        if (col < k) {
            sum += y[col] * float(x_row[b].d) * float(x_row[b].qs[lane]);
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) dst[row] = sum;
}
