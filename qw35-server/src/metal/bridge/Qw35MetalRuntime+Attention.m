// Qw35MetalRuntime+Attention.m – full-attention layers (decode + prefill) and the attention pipeline lookup.
//
// Split out of the monolithic Qw35MetalRuntime.m so this stage of the
// Qwen3.5 forward pass is recognizable by filename. Shares the runtime's
// instance variables, helpers, and private methods via the category header.

#import "Qw35MetalRuntime+Internal.h"

@implementation Qw35MetalRuntime (Attention)

- (id<MTLComputePipelineState>)attnPipeline:(NSString *)name error:(NSError **)error {
    return [_pipelineCache attnPipelineNamed:name
                                       heads:(int)_h.attention_heads
                                     kvHeads:(int)_h.attention_kv_heads
                                     headDim:(int)_h.attention_key_length
                                     ropeDim:(int)_h.rope_dimension_count
                                     hasGate:_h.attn_gate != 0
                                       error:error];
}

- (BOOL)encodeAttentionDecodeLayer:(id<MTLComputeCommandEncoder>)enc
                             layer:(Qw35LayerTensors *)layer
                              slot:(uint32_t)slot
                               pos:(uint32_t)pos
                             error:(NSError **)error {
    // Segmented KV: the current token lives entirely in slab s; bind that one
    // slab for the write at the fixed per-layer stride (_kvSlab, not capacity).
    const uint32_t s = pos / _kvSlab;
    const NSUInteger kv_off = (NSUInteger)((uint64_t)slot * _kvSlab * _kvRowBytes);
    int64_t cache_row = (int64_t)pos - (int64_t)s * (int64_t)_kvSlab;
    int kv_slab_i = (int)_kvSlab;
    int slot_i = (int)slot;
    const float scale = 1.0f / sqrtf((float)_h.attention_key_length);

    if (![self encodeDecodeMatvecTensor:enc weight:layer.attnQ input:_norm inputOffset:0 dst:_qkv residual:NO error:error]) return NO;
    if (![self encodeDecodeMatvecTensor:enc weight:layer.attnK input:_norm inputOffset:0 dst:_ffn_gate residual:NO error:error]) return NO;
    if (![self encodeDecodeMatvecTensor:enc weight:layer.attnV input:_norm inputOffset:0 dst:_ffn_up residual:NO error:error]) return NO;

    Qw35Tensor *q_norm = layer.attnQNorm;
    Qw35Tensor *k_norm = layer.attnKNorm;
    id<MTLComputePipelineState> prep = _psAttnPrepDecode
        ?: [self attnPipeline:_kvQ8 ? @"qw35_attn_decode_preprocess_q8_0_f32"
                                    : @"qw35_attn_decode_preprocess_f32"
                        error:error];
    if (!prep) return NO;

    int64_t n_head = _h.attention_heads;
    int64_t n_kv_head = _h.attention_kv_heads;
    int64_t head_dim = _h.attention_key_length;
    int64_t pos64 = pos;
    float eps = _h.rms_epsilon;
    int64_t rope_dim = _h.rope_dimension_count;
    [enc setComputePipelineState:prep];
    [enc setBuffer:_qkv offset:0 atIndex:0];
    [enc setBuffer:_ffn_gate offset:0 atIndex:1];
    [enc setBuffer:_ffn_up offset:0 atIndex:2];
    [enc setBuffer:q_norm.buffer offset:q_norm.offset atIndex:3];
    [enc setBuffer:k_norm.buffer offset:k_norm.offset atIndex:4];
    [enc setBuffer:_q_rep offset:0 atIndex:5];
    [enc setBuffer:_z_gate offset:0 atIndex:6];
    [enc setBuffer:_kSlabs[s] offset:kv_off atIndex:7];
    [enc setBuffer:_vSlabs[s] offset:kv_off atIndex:8];
    [enc setBytes:&n_head length:sizeof(n_head) atIndex:9];
    [enc setBytes:&n_kv_head length:sizeof(n_kv_head) atIndex:10];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:11];
    [enc setBytes:&pos64 length:sizeof(pos64) atIndex:12];
    [enc setBytes:&eps length:sizeof(eps) atIndex:13];
    [enc setBuffer:_rope_freq offset:0 atIndex:14];
    [enc setBytes:&rope_dim length:sizeof(rope_dim) atIndex:15];
    [enc setBytes:&cache_row length:sizeof(cache_row) atIndex:16];
    qw35_dispatch_1d(enc, (NSUInteger)(_h.attention_heads * _h.attention_key_length), 256);

    // Flash decode. Short contexts: one simdgroup per head walking the whole
    // attended range (barrier-free, sigmoid output gate applied in-kernel).
    // Long contexts: the split-K variant (8 simdgroups per head over
    // contiguous chunks + threadgroup merge) — the serial walk is what makes
    // decode tok/s decay with context length. The threshold is on the
    // attended length (window-bounded), below which the split's merge
    // overhead outweighs the parallelism.
    int64_t seq_len = (int64_t)pos + 1;
    int64_t attended = seq_len;
    if (_attnWindow > 0 && (int64_t)_attnSink + (int64_t)_attnWindow < seq_len) {
        attended = (int64_t)_attnSink + (int64_t)_attnWindow;
    }
    const BOOL useSplit = attended >= _splitKMin && _psAttnFlashDecodeSplit != nil;
    id<MTLComputePipelineState> attn = useSplit ? _psAttnFlashDecodeSplit
        : (_psAttnFlashDecode
            ?: [self attnPipeline:_kvQ8 ? @"qw35_attention_gqa_flash_decode_q8_0_f32"
                                        : @"qw35_attention_gqa_flash_decode_f32"
                            error:error]);
    if (!attn) return NO;
    [enc setComputePipelineState:attn];
    [enc setBuffer:_q_rep offset:0 atIndex:0];
    [enc setBuffer:_kSlabPtrs offset:0 atIndex:1];
    [enc setBuffer:_vSlabPtrs offset:0 atIndex:2];
    [enc setBuffer:_core offset:0 atIndex:3];
    [enc setBuffer:_z_gate offset:0 atIndex:4];
    [enc setBytes:&n_head length:sizeof(n_head) atIndex:5];
    [enc setBytes:&n_kv_head length:sizeof(n_kv_head) atIndex:6];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
    [enc setBytes:&seq_len length:sizeof(seq_len) atIndex:8];
    [enc setBytes:&scale length:sizeof(scale) atIndex:9];
    int attn_window = _attnWindow;
    int attn_sink = _attnSink;
    [enc setBytes:&attn_window length:sizeof(attn_window) atIndex:10];
    [enc setBytes:&attn_sink length:sizeof(attn_sink) atIndex:11];
    [enc setBytes:&kv_slab_i length:sizeof(kv_slab_i) atIndex:16];
    [enc setBytes:&slot_i length:sizeof(slot_i) atIndex:17];
    [self useKvSlabsForRead:enc];  // bindless slabs: ensure residency + write->read hazard
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)_h.attention_heads, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(useSplit ? 256 : 32, 1, 1)];

    if (![self encodeDecodeMatvecTensor:enc weight:layer.attnOutput input:_core inputOffset:0 dst:_act_b residual:NO error:error]) return NO;
    return YES;
}

- (BOOL)encodeAttentionPrefillLayer:(id<MTLComputeCommandEncoder>)enc
                               slot:(uint32_t)slot
                               pos0:(uint32_t)pos0
                             tokens:(uint32_t)tokensCount
                              layer:(Qw35LayerTensors *)layer
                              error:(NSError **)error {
    // Segmented KV: evalTokens guarantees this batch does not straddle a slab,
    // so the whole [pos0, pos0+tokens) range writes into slab s.
    const uint32_t s = pos0 / _kvSlab;
    const NSUInteger kv_off = (NSUInteger)((uint64_t)slot * _kvSlab * _kvRowBytes);
    int64_t cache_pos0 = (int64_t)pos0 - (int64_t)s * (int64_t)_kvSlab;
    int kv_slab_i = (int)_kvSlab;
    int slot_i = (int)slot;
    const float scale = 1.0f / sqrtf((float)_h.attention_key_length);

    if (![self encodeMatvecBatch:enc weight:layer.attnQ input:_norm inputOffset:0 dst:_qkv dstOffset:0 tokens:tokensCount error:error]) return NO;
    if (![self encodeMatvecBatch:enc weight:layer.attnK input:_norm inputOffset:0 dst:_ffn_gate dstOffset:0 tokens:tokensCount error:error]) return NO;
    if (![self encodeMatvecBatch:enc weight:layer.attnV input:_norm inputOffset:0 dst:_ffn_up dstOffset:0 tokens:tokensCount error:error]) return NO;

    Qw35Tensor *q_norm = layer.attnQNorm;
    Qw35Tensor *k_norm = layer.attnKNorm;
    if (!q_norm || !k_norm) return NO;
    id<MTLComputePipelineState> prep = _psAttnPrepPrefill
        ?: [self attnPipeline:_kvQ8 ? @"qw35_attn_prefill_preprocess_q8_0_f32"
                                    : @"qw35_attn_prefill_preprocess_f32"
                        error:error];
    if (!prep) return NO;

    int64_t n_head = _h.attention_heads;
    int64_t n_kv_head = _h.attention_kv_heads;
    int64_t head_dim = _h.attention_key_length;
    int64_t pos064 = pos0;
    int64_t n_tokens = tokensCount;
    float eps = _h.rms_epsilon;
    int64_t rope_dim = _h.rope_dimension_count;
    const uint64_t work_dim = _h.attention_heads * _h.attention_key_length;
    const uint64_t kv_work = _h.attention_kv_heads * _h.attention_key_length;
    const uint64_t max_work = work_dim > kv_work ? work_dim : kv_work;
    [enc setComputePipelineState:prep];
    [enc setBuffer:_qkv offset:0 atIndex:0];
    [enc setBuffer:_ffn_gate offset:0 atIndex:1];
    [enc setBuffer:_ffn_up offset:0 atIndex:2];
    [enc setBuffer:q_norm.buffer offset:q_norm.offset atIndex:3];
    [enc setBuffer:k_norm.buffer offset:k_norm.offset atIndex:4];
    [enc setBuffer:_q_rep offset:0 atIndex:5];
    [enc setBuffer:_z_gate offset:0 atIndex:6];
    [enc setBuffer:_kSlabs[s] offset:kv_off atIndex:7];
    [enc setBuffer:_vSlabs[s] offset:kv_off atIndex:8];
    [enc setBytes:&n_head length:sizeof(n_head) atIndex:9];
    [enc setBytes:&n_kv_head length:sizeof(n_kv_head) atIndex:10];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:11];
    [enc setBytes:&pos064 length:sizeof(pos064) atIndex:12];
    [enc setBytes:&eps length:sizeof(eps) atIndex:13];
    [enc setBuffer:_rope_freq offset:0 atIndex:14];
    [enc setBytes:&rope_dim length:sizeof(rope_dim) atIndex:15];
    [enc setBytes:&cache_pos0 length:sizeof(cache_pos0) atIndex:16];
    [enc setBytes:&n_tokens length:sizeof(n_tokens) atIndex:20];
    qw35_dispatch_1d(enc, (NSUInteger)((uint64_t)tokensCount * max_work), 256);

    id<MTLComputePipelineState> attn = _psAttnPrefill
        ?: [self attnPipeline:_kvQ8 ? @"qw35_attention_gqa_prefill_q8_0_f32"
                                    : @"qw35_attention_gqa_prefill_f32"
                        error:error];
    if (!attn) return NO;
    [enc setComputePipelineState:attn];
    [enc setBuffer:_q_rep offset:0 atIndex:0];
    [enc setBuffer:_kSlabPtrs offset:0 atIndex:1];
    [enc setBuffer:_vSlabPtrs offset:0 atIndex:2];
    [enc setBuffer:_core offset:0 atIndex:3];
    [enc setBuffer:_z_gate offset:0 atIndex:4];
    [enc setBytes:&n_head length:sizeof(n_head) atIndex:5];
    [enc setBytes:&n_kv_head length:sizeof(n_kv_head) atIndex:6];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
    [enc setBytes:&pos064 length:sizeof(pos064) atIndex:8];
    [enc setBytes:&scale length:sizeof(scale) atIndex:9];
    [enc setBytes:&n_tokens length:sizeof(n_tokens) atIndex:10];
    [enc setBytes:&kv_slab_i length:sizeof(kv_slab_i) atIndex:16];
    [enc setBytes:&slot_i length:sizeof(slot_i) atIndex:17];
    [self useKvSlabsForRead:enc];  // bindless slabs: ensure residency + write->read hazard
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tokensCount,
                                          (NSUInteger)_h.attention_heads,
                                          1)
         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    if (![self encodeMatvecBatch:enc weight:layer.attnOutput input:_core inputOffset:0 dst:_act_b dstOffset:0 tokens:tokensCount error:error]) return NO;
    return YES;
}

@end
