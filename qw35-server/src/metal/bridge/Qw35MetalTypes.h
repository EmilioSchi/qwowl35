// Qw35MetalTypes.h – Shared C structs for the Qw35 Metal runtime bridge.
// These are the FFI types exchanged between Rust and Objective-C.

#ifndef QW35_METAL_TYPES_H
#define QW35_METAL_TYPES_H

#include <stdint.h>

typedef struct {
    uint64_t bytes;
    uint64_t mapped_bytes;
    uint64_t touches;
    uint64_t checksum;
    double   elapsed_ms;
    uint32_t view_count;
} qw35_metal_warm_report;

typedef struct {
    const uint8_t *name;
    uintptr_t      name_len;
    uint64_t       dims[4];
    uint32_t       n_dims;
    uint32_t       type_id;
    uint64_t       abs_offset;
    uint64_t       bytes;
    uint64_t       elements;
} qw35_metal_tensor_desc;

typedef struct {
    uint32_t transformer_layers;
    uint32_t embedding_length;
    uint32_t feed_forward_length;
    uint32_t attention_heads;
    uint32_t attention_kv_heads;
    uint32_t attention_key_length;
    uint32_t attention_value_length;
    uint32_t rope_dimension_count;
    int32_t  rope_sections[4];
    float    rope_freq_base;
    float    rms_epsilon;
    uint32_t ssm_conv_kernel;
    uint32_t ssm_state_size;
    uint32_t ssm_group_count;
    uint32_t ssm_time_step_rank;
    uint32_t ssm_inner_size;
    uint32_t full_attention_interval;
    // Decode-time sliding-window attention config (from --attn-window/--attn-sink).
    // window <= 0 means full attention; sink = leading tokens always attended.
    int32_t  attn_window;
    int32_t  attn_sink;
} qw35_metal_hparams;

/// Output-head mode for an eval call.
typedef enum {
    QW35_LOGITS_NONE   = 0, // no output head (non-final prefill chunk)
    QW35_LOGITS_ARGMAX = 1, // fused argmax only; logits never materialized
    QW35_LOGITS_FULL   = 2, // full logits vector for CPU sampling
} qw35_logits_mode;

typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} qw35_metal_args_mul_mm;

#endif // QW35_METAL_TYPES_H
