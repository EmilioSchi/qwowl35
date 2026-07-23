// Qw35MetalRuntime+SSM.m – SSM / delta (linear-attention) layers (decode + prefill).
//
// Split out of the monolithic Qw35MetalRuntime.m so this stage of the
// Qwen3.5 forward pass is recognizable by filename. Shares the runtime's
// instance variables, helpers, and private methods via the category header.

#import "Qw35MetalRuntime+Internal.h"

@implementation Qw35MetalRuntime (SSM)

- (BOOL)encodeDeltaDecodeLayer:(id<MTLComputeCommandEncoder>)enc
                         layer:(Qw35LayerTensors *)layer
                          slot:(uint32_t)slot
                         error:(NSError **)error {
    const int conv_channels = (int)(_h.ssm_inner_size + 2u * _h.ssm_group_count * _h.ssm_state_size);
    const uint64_t conv_slot_elems = (uint64_t)conv_channels * (_h.ssm_conv_kernel - 1);
    const uint64_t state_slot_elems = (uint64_t)_h.ssm_time_step_rank * _h.ssm_state_size * _h.ssm_state_size;
    const NSUInteger conv_off = (NSUInteger)(slot * conv_slot_elems * sizeof(float));
    const NSUInteger state_off = (NSUInteger)(slot * state_slot_elems * sizeof(float));
    const float scale = 1.0f / sqrtf((float)_h.ssm_state_size);

    if (![self encodeDecodeMatvecTensor:enc weight:layer.attnQkv input:_norm inputOffset:0 dst:_qkv residual:NO error:error]) return NO;
    if (![self encodeDecodeMatvecTensor:enc weight:layer.attnGate input:_norm inputOffset:0 dst:_z_gate residual:NO error:error]) return NO;
    if (![self encodeDecodeMatvecTensor:enc weight:layer.ssmBeta input:_norm inputOffset:0 dst:_beta residual:NO error:error]) return NO;
    if (![self encodeDecodeMatvecTensor:enc weight:layer.ssmAlpha input:_norm inputOffset:0 dst:_alpha residual:NO error:error]) return NO;

    // Single fused kernel: conv1d step + SiLU, L2 norm of q/k, gated
    // delta-rule state update, per-head group RMS norm and SiLU(z) gating.
    // Reads the z gate from _z_gate and overwrites it with the gated output.
    id<MTLComputePipelineState> pipe = _psSsmFused
        ?: [_pipelineCache pipelineNamed:@"qw35_ssm_conv_recurrent_gate_norm_step128_f32" error:error];
    if (!pipe) return NO;
    int num_v_heads = (int)_h.ssm_time_step_rank;
    int head_dim = (int)_h.ssm_state_size;
    float eps = _h.rms_epsilon;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:_qkv offset:0 atIndex:0];
    [enc setBuffer:layer.ssmConv.buffer offset:layer.ssmConv.offset atIndex:1];
    [enc setBuffer:_conv_state offset:conv_off atIndex:2];
    [enc setBuffer:_beta offset:0 atIndex:3];
    [enc setBuffer:_alpha offset:0 atIndex:4];
    [enc setBuffer:layer.ssmDt.buffer offset:layer.ssmDt.offset atIndex:5];
    [enc setBuffer:layer.ssmA.buffer offset:layer.ssmA.offset atIndex:6];
    [enc setBuffer:_ssm_state offset:state_off atIndex:7];
    [enc setBuffer:_z_gate offset:0 atIndex:8];
    [enc setBuffer:layer.ssmNorm.buffer offset:layer.ssmNorm.offset atIndex:9];
    [enc setBuffer:_z_gate offset:0 atIndex:10];
    [enc setBytes:&conv_channels length:sizeof(conv_channels) atIndex:11];
    [enc setBytes:&num_v_heads length:sizeof(num_v_heads) atIndex:12];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:13];
    [enc setBytes:&scale length:sizeof(scale) atIndex:14];
    [enc setBytes:&eps length:sizeof(eps) atIndex:15];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)_h.ssm_group_count, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    if (![self encodeDecodeMatvecTensor:enc weight:layer.ssmOut input:_z_gate inputOffset:0 dst:_act_b residual:NO error:error]) return NO;
    return YES;
}

- (BOOL)encodeDeltaPrefillLayer:(id<MTLComputeCommandEncoder>)enc
                           slot:(uint32_t)slot
                         tokens:(uint32_t)tokensCount
                          layer:(Qw35LayerTensors *)layer
                          error:(NSError **)error {
    const int conv_channels = (int)(_h.ssm_inner_size + 2u * _h.ssm_group_count * _h.ssm_state_size);
    const uint64_t conv_slot_elems = (uint64_t)conv_channels * (_h.ssm_conv_kernel - 1);
    const uint64_t state_slot_elems = (uint64_t)_h.ssm_time_step_rank * _h.ssm_state_size * _h.ssm_state_size;
    const NSUInteger conv_off = (NSUInteger)(slot * conv_slot_elems * sizeof(float));
    const NSUInteger state_off = (NSUInteger)(slot * state_slot_elems * sizeof(float));
    const float scale = 1.0f / sqrtf((float)_h.ssm_state_size);

    if (![self encodeMatvecBatch:enc weight:layer.attnQkv input:_norm inputOffset:0 dst:_qkv dstOffset:0 tokens:tokensCount error:error]) return NO;
    if (![self encodeMatvecBatch:enc weight:layer.attnGate input:_norm inputOffset:0 dst:_z_gate dstOffset:0 tokens:tokensCount error:error]) return NO;
    if (![self encodeMatvecBatch:enc weight:layer.ssmBeta input:_norm inputOffset:0 dst:_beta dstOffset:0 tokens:tokensCount error:error]) return NO;
    if (![self encodeMatvecBatch:enc weight:layer.ssmAlpha input:_norm inputOffset:0 dst:_alpha dstOffset:0 tokens:tokensCount error:error]) return NO;

    Qw35Tensor *conv_w = layer.ssmConv;
    if (!conv_w) return NO;
    id<MTLComputePipelineState> conv_pipe = _psSsmConvBatch
        ?: [_pipelineCache pipelineNamed:@"qw35_ssm_conv1d_step4_batch_f32" error:error];
    if (!conv_pipe) return NO;
    int n_tokens = (int)tokensCount;
    [enc setComputePipelineState:conv_pipe];
    [enc setBuffer:_qkv offset:0 atIndex:0];
    [enc setBuffer:conv_w.buffer offset:conv_w.offset atIndex:1];
    [enc setBuffer:_conv_state offset:conv_off atIndex:2];
    [enc setBuffer:_qkv offset:0 atIndex:3];
    [enc setBytes:&conv_channels length:sizeof(conv_channels) atIndex:4];
    [enc setBytes:&n_tokens length:sizeof(n_tokens) atIndex:5];
    qw35_dispatch_1d(enc, (NSUInteger)conv_channels, 256);

    id<MTLComputePipelineState> prep_pipe = _psSsmL2Batch
        ?: [_pipelineCache pipelineNamed:@"qw35_ssm_l2_repeat_qk_batch_f32" error:error];
    if (!prep_pipe) return NO;
    int num_k_heads = (int)_h.ssm_group_count;
    int num_v_heads = (int)_h.ssm_time_step_rank;
    int head_dim = (int)_h.ssm_state_size;
    int src_stride = conv_channels;
    float eps = _h.rms_epsilon;
    [enc setComputePipelineState:prep_pipe];
    [enc setBuffer:_qkv offset:0 atIndex:0];
    [enc setBuffer:_qkv offset:(NSUInteger)(_h.ssm_group_count * _h.ssm_state_size * sizeof(float)) atIndex:1];
    [enc setBuffer:_q_rep offset:0 atIndex:2];
    [enc setBuffer:_k_rep offset:0 atIndex:3];
    [enc setBytes:&num_k_heads length:sizeof(num_k_heads) atIndex:4];
    [enc setBytes:&num_v_heads length:sizeof(num_v_heads) atIndex:5];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
    [enc setBytes:&eps length:sizeof(eps) atIndex:7];
    [enc setBytes:&src_stride length:sizeof(src_stride) atIndex:8];
    [enc setBytes:&n_tokens length:sizeof(n_tokens) atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tokensCount,
                                          (NSUInteger)_h.ssm_time_step_rank,
                                          1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    Qw35Tensor *dt = layer.ssmDt;
    Qw35Tensor *a = layer.ssmA;
    if (!dt || !a) return NO;
    id<MTLComputePipelineState> rec_pipe = _psSsmRecBatch
        ?: [_pipelineCache pipelineNamed:@"qw35_ssm_recurrent_step128_batch_rows_f32" error:error];
    if (!rec_pipe) return NO;
    int v_stride = conv_channels;
    [enc setComputePipelineState:rec_pipe];
    [enc setBuffer:_q_rep offset:0 atIndex:0];
    [enc setBuffer:_k_rep offset:0 atIndex:1];
    [enc setBuffer:_qkv offset:(NSUInteger)(2u * _h.ssm_group_count * _h.ssm_state_size * sizeof(float)) atIndex:2];
    [enc setBuffer:_beta offset:0 atIndex:3];
    [enc setBuffer:_alpha offset:0 atIndex:4];
    [enc setBuffer:dt.buffer offset:dt.offset atIndex:5];
    [enc setBuffer:a.buffer offset:a.offset atIndex:6];
    [enc setBuffer:_ssm_state offset:state_off atIndex:7];
    [enc setBuffer:_core offset:0 atIndex:8];
    [enc setBytes:&num_v_heads length:sizeof(num_v_heads) atIndex:9];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:10];
    [enc setBytes:&scale length:sizeof(scale) atIndex:11];
    [enc setBytes:&n_tokens length:sizeof(n_tokens) atIndex:12];
    [enc setBytes:&v_stride length:sizeof(v_stride) atIndex:13];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)_h.ssm_state_size + 3u) / 4u,
                                          (NSUInteger)_h.ssm_time_step_rank,
                                          1)
         threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];

    Qw35Tensor *ssm_norm = layer.ssmNorm;
    if (!ssm_norm) return NO;
    id<MTLComputePipelineState> gate_norm_pipe = _psSsmGateNormBatch
        ?: [_pipelineCache pipelineNamed:@"qw35_ssm_gate_norm_batch_f32" error:error];
    if (!gate_norm_pipe) return NO;
    [enc setComputePipelineState:gate_norm_pipe];
    [enc setBuffer:_core offset:0 atIndex:0];
    [enc setBuffer:_z_gate offset:0 atIndex:1];
    [enc setBuffer:ssm_norm.buffer offset:ssm_norm.offset atIndex:2];
    [enc setBuffer:_z_gate offset:0 atIndex:3];
    [enc setBytes:&num_v_heads length:sizeof(num_v_heads) atIndex:4];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:5];
    [enc setBytes:&eps length:sizeof(eps) atIndex:6];
    [enc setBytes:&n_tokens length:sizeof(n_tokens) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tokensCount, (NSUInteger)_h.ssm_time_step_rank, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    if (![self encodeMatvecBatch:enc weight:layer.ssmOut input:_z_gate inputOffset:0 dst:_act_b dstOffset:0 tokens:tokensCount error:error]) return NO;
    return YES;
}

@end
