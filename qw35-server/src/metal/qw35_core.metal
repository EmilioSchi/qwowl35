// Core normalization and elementwise kernels.
kernel void qw35_rms_norm_weight_f32(
    device const float * x [[buffer(0)]],
    device const float * w [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant float &eps [[buffer(3)]],
    constant int64_t &n_elements [[buffer(4)]],
    uint ti [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[8];
    const int n = int(n_elements);

    float sum = 0.0f;
    for (int i = int(ti); i < n; i += 256) {
        sum += x[i] * x[i];
    }
    const float total = qw35_threadgroup_sum_256(sum, partial, ti, lane, simd_group);
    const float inv_norm = rsqrt(total / float(n) + eps);
    for (int i = int(ti); i < n; i += 256) {
        dst[i] = x[i] * inv_norm * w[i];
    }
}

kernel void qw35_rms_norm_weight_batch_f32(
    device const float * x [[buffer(0)]],
    device const float * w [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant float &eps [[buffer(3)]],
    constant int64_t &n_elements [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint ti [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[8];
    const int n = int(n_elements);
    device const float *x_row = x + int(row) * n;
    device float *dst_row = dst + int(row) * n;

    float sum = 0.0f;
    for (int i = int(ti); i < n; i += 256) {
        sum += x_row[i] * x_row[i];
    }
    const float total = qw35_threadgroup_sum_256(sum, partial, ti, lane, simd_group);
    const float inv_norm = rsqrt(total / float(n) + eps);
    for (int i = int(ti); i < n; i += 256) {
        dst_row[i] = x_row[i] * inv_norm * w[i];
    }
}

kernel void qw35_residual_rms_norm_weight_batch_f32(
    device float * x [[buffer(0)]],
    device const float * residual [[buffer(1)]],
    device const float * w [[buffer(2)]],
    device float * dst [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    constant int64_t &n_elements [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint ti [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[8];
    const int n = int(n_elements);
    device float *x_row = x + int(row) * n;
    device const float *res_row = residual + int(row) * n;
    device float *dst_row = dst + int(row) * n;

    float sum = 0.0f;
    for (int i = int(ti); i < n; i += 256) {
        const float value = x_row[i] + res_row[i];
        x_row[i] = value;
        sum += value * value;
    }
    const float total = qw35_threadgroup_sum_256(sum, partial, ti, lane, simd_group);
    const float inv_norm = rsqrt(total / float(n) + eps);
    for (int i = int(ti); i < n; i += 256) {
        dst_row[i] = x_row[i] * inv_norm * w[i];
    }
}

kernel void qw35_residual_rms_norm_weight_f32(
    device float * x [[buffer(0)]],
    device const float * residual [[buffer(1)]],
    device const float * w [[buffer(2)]],
    device float * dst [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    constant int64_t &n_elements [[buffer(5)]],
    uint ti [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[8];
    const int n = int(n_elements);

    float sum = 0.0f;
    for (int i = int(ti); i < n; i += 256) {
        const float value = x[i] + residual[i];
        x[i] = value;
        sum += value * value;
    }
    const float total = qw35_threadgroup_sum_256(sum, partial, ti, lane, simd_group);
    const float inv_norm = rsqrt(total / float(n) + eps);
    for (int i = int(ti); i < n; i += 256) {
        dst[i] = x[i] * inv_norm * w[i];
    }
}

kernel void qw35_swiglu_f32(
    device const float * gate [[buffer(0)]],
    device const float * up [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant int64_t &n_elements [[buffer(3)]],
    uint ti [[thread_position_in_grid]]
) {
    if (ti >= n_elements) return;
    dst[ti] = qw35_silu_f32(gate[ti]) * up[ti];
}
