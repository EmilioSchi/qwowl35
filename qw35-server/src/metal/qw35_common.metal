#include <metal_stdlib>
using namespace metal;

#define FC_MUL_MM 700
#define FC_DELTA_TOKEN_COUNT 730
#define FC_SSM_TOKEN_COUNT 740
#define FC_SSM_KEEP_SNAPSHOTS 741
#define FC_ATTN_HEADS 750
#define FC_ATTN_KV_HEADS 751
#define FC_ATTN_HEAD_DIM 752
#define FC_ATTN_ROPE_DIM 753
#define FC_COMPRESSED_VOCAB_SIZE 760
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Attention KV cache slots are 16-bit floats, matching llama.cpp's default
// f16 cache type for this model. The helper names are kept stable because many
// kernels already call them.
static inline float qw35_bf16_to_f32(ushort x) {
    return float(as_type<half>(x));
}

static inline ushort qw35_f32_to_bf16(float x) {
    return as_type<ushort>(half(x));
}

// Coarse page-residency warmup kernel. The Rust engine currently performs the
// mmap page warmup on CPU; this static shader is compiled at build time so the
// Metal warmup path has a stable function name when the graph backend is wired.
kernel void qw35_touch_u8_stride(
    device const uchar *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant ulong &stride [[buffer(2)]],
    constant ulong &bytes [[buffer(3)]],
    constant ulong &dst_offset [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const ulong off = (ulong)gid * stride;
    if (off < bytes) {
        dst[dst_offset + gid] = src[off];
    }
}
