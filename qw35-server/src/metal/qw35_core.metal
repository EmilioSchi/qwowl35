// Core normalization and elementwise kernels.
//
// The row sweeps below stride by the threadgroup size (`i += 256`), so a
// thread's successive accesses are 1 KB apart and the compiler cannot merge
// them the way it merges a contiguous per-thread walk. Going float4 is
// therefore a real 4x cut in load/store instructions here, not a no-op — these
// kernels are latency-bound (one threadgroup for a whole 4096-wide row, so 7 of
// 8 GPU cores idle) rather than bandwidth-bound, which is why instruction count
// is what matters.
//
// Alignment: every call site binds these buffers at offset 0 (scratch) or at a
// GGUF tensor offset, and GGUF's general.alignment is 32, so the float4 casts
// are safe. `n % 4` elements are handled by a scalar tail for generality.
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

    const int n4 = n >> 2;
    device const float4 *x4 = (device const float4 *)x;
    device const float4 *w4 = (device const float4 *)w;
    device float4 *dst4 = (device float4 *)dst;

    float sum = 0.0f;
    for (int i = int(ti); i < n4; i += 256) {
        const float4 v = x4[i];
        sum += dot(v, v);
    }
    for (int i = (n4 << 2) + int(ti); i < n; i += 256) {
        sum += x[i] * x[i];
    }
    const float total = qw35_threadgroup_sum_256(sum, partial, ti, lane, simd_group);
    const float inv_norm = rsqrt(total / float(n) + eps);
    for (int i = int(ti); i < n4; i += 256) {
        dst4[i] = x4[i] * inv_norm * w4[i];
    }
    for (int i = (n4 << 2) + int(ti); i < n; i += 256) {
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

    const int n4 = n >> 2;
    device float4 *x4 = (device float4 *)x;
    device const float4 *res4 = (device const float4 *)residual;
    device const float4 *w4 = (device const float4 *)w;
    device float4 *dst4 = (device float4 *)dst;

    float sum = 0.0f;
    for (int i = int(ti); i < n4; i += 256) {
        const float4 value = x4[i] + res4[i];
        x4[i] = value;
        sum += dot(value, value);
    }
    for (int i = (n4 << 2) + int(ti); i < n; i += 256) {
        const float value = x[i] + residual[i];
        x[i] = value;
        sum += value * value;
    }
    const float total = qw35_threadgroup_sum_256(sum, partial, ti, lane, simd_group);
    const float inv_norm = rsqrt(total / float(n) + eps);
    for (int i = int(ti); i < n4; i += 256) {
        dst4[i] = x4[i] * inv_norm * w4[i];
    }
    for (int i = (n4 << 2) + int(ti); i < n; i += 256) {
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
