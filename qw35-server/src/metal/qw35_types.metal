// Shared Qw35 Metal types and helpers (Qwen3.5-9B Q4_K_M tensor set only).
struct qw35_block_q4_k {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

struct qw35_block_q5_k {
    half d;
    half dmin;
    uchar scales[12];
    uchar qh[32];
    uchar qs[128];
};

struct qw35_block_q6_k {
    uchar ql[128];
    uchar qh[64];
    char scales[16];
    half d;
};

struct qw35_block_q8_0 {
    half d;
    char qs[32];
};

template <typename type4x4>
void qw35_dequantize_q8_0(device const qw35_block_q8_0 *xb, short il, thread type4x4 &reg) {
    device const char *qs = xb->qs;
    const float d = float(xb->d);
    for (int i = 0; i < 16; i++) {
        reg[i/4][i%4] = half(float(qs[i + 16*il]) * d);
    }
}

#define QW35_DECODE_ROWS_PER_GROUP 8

static inline float qw35_silu_f32(float x) {
    return x / (1.0f + exp(-x));
}

static inline float qw35_sigmoid_f32(float x) {
    return 1.0f / (1.0f + exp(-x));
}

static inline float qw35_threadgroup_sum_256(
    float value,
    threadgroup float *partials,
    uint tid,
    uint lane,
    uint simd_group
) {
    value = simd_sum(value);
    if (lane == 0) partials[simd_group] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = tid < 8 ? partials[tid] : 0.0f;
    total = simd_sum(total);
    if (tid == 0) partials[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partials[0];
}

static inline float qw35_softplus_f32(float x) {
    return x > 20.0f ? x : log(1.0f + exp(x));
}

static inline void qw35_get_scale_min_k4(
    int j,
    device const uchar (&q)[12],
    thread uchar &sc,
    thread uchar &m
) {
    if (j < 4) {
        sc = q[j] & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4)   | ((q[j] >> 6) << 4);
    }
}
