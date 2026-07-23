// Qw35MetalRuntime+FFN.m – FFN/MLP plus the shared per-operation encoders (embedding, RMS norm, matvec, tiled matmul, SwiGLU).
//
// Split out of the monolithic Qw35MetalRuntime.m so this stage of the
// Qwen3.5 forward pass is recognizable by filename. Shares the runtime's
// instance variables, helpers, and private methods via the category header.

#import "Qw35MetalRuntime+Internal.h"

@implementation Qw35MetalRuntime (FFN)

/// GF4 decode FFN: fused gate+up+SwiGLU into _ffn_gate, then down projection
/// accumulating into _act_a.
- (BOOL)encodeGf4FusedFfn:(id<MTLComputeCommandEncoder>)enc
                    layer:(Qw35LayerTensors *)layer
                    error:(NSError **)error {
    const int64_t emb = (int64_t)_h.embedding_length;
    const int64_t ffn = (int64_t)_h.feed_forward_length;

    id<MTLComputePipelineState> gateUp = _psGf4FusedFfn
        ?: [_pipelineCache pipelineNamed:@"qw35_ffn_gate_up_swiglu_gf4_f32" error:error];
    if (!gateUp) return NO;
    int64_t n_groups = emb / 8;
    [enc setComputePipelineState:gateUp];
    [enc setBuffer:layer.ffnGate.buffer offset:layer.ffnGate.offset atIndex:0];
    [enc setBuffer:layer.ffnUp.buffer offset:layer.ffnUp.offset atIndex:1];
    [enc setBuffer:_norm offset:0 atIndex:2];
    [enc setBuffer:_ffn_gate offset:0 atIndex:3];
    [enc setBytes:&n_groups length:sizeof(n_groups) atIndex:4];
    [enc setBytes:&emb length:sizeof(emb) atIndex:5];
    [enc setBytes:&ffn length:sizeof(ffn) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)qw35_div_up_u64((uint64_t)ffn, 8), 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];

    return [self encodeDecodeMatvecTensor:enc weight:layer.ffnDown input:_ffn_gate inputOffset:0 dst:_act_a residual:YES error:error];
}

// ---------------------------------------------------------------------------
#pragma mark - Per-operation encoders
// ---------------------------------------------------------------------------

/// Kernel names for the token-embedding row gather, selected by the
/// embedding tensor's quantization (q4_k on the 9B, q8_0 on the reranker).
static NSString *qw35_get_row_kernel(uint32_t type_id, BOOL batch, NSError **error) {
    switch (type_id) {
        case 8:  return batch ? @"qw35_get_rows_q8_0_f32" : @"qw35_get_row_q8_0_f32";
        case 12: return batch ? @"qw35_get_rows_q4_k_f32" : @"qw35_get_row_q4_k_f32";
        default:
            if (error) *error = qw35_error(@"unsupported token_embd.weight type %u", type_id);
            return nil;
    }
}

- (BOOL)encodeEmbedding:(id<MTLComputeCommandEncoder>)enc
                  token:(uint32_t)token
                  error:(NSError **)error {
    id<MTLComputePipelineState> pipe = _psGetRow;
    if (!pipe) {
        NSString *kernel = qw35_get_row_kernel(_tokenEmbd.type_id, NO, error);
        if (!kernel) return NO;
        pipe = [_pipelineCache pipelineNamed:kernel error:error];
        if (!pipe) return NO;
    }
    int64_t k = (int64_t)_h.embedding_length;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:_tokenEmbd.buffer offset:_tokenEmbd.offset atIndex:0];
    [enc setBuffer:_act_a offset:0 atIndex:1];
    [enc setBytes:&token length:sizeof(token) atIndex:2];
    [enc setBytes:&k length:sizeof(k) atIndex:3];
    qw35_dispatch_1d(enc, (NSUInteger)_h.embedding_length, 256);
    return YES;
}

- (BOOL)encodeEmbeddingBatch:(id<MTLComputeCommandEncoder>)enc
                       count:(uint32_t)count
                       error:(NSError **)error {
    Qw35Tensor *embd = [self tensorNamed:@"token_embd.weight" error:error];
    if (!embd) return NO;
    id<MTLComputePipelineState> pipe = embd.type_id == _tokenEmbd.type_id ? _psGetRows : nil;
    if (!pipe) {
        NSString *kernel = qw35_get_row_kernel(embd.type_id, YES, error);
        if (!kernel) return NO;
        pipe = [_pipelineCache pipelineNamed:kernel error:error];
        if (!pipe) return NO;
    }
    int64_t k = (int64_t)_h.embedding_length;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:embd.buffer offset:embd.offset atIndex:0];
    [enc setBuffer:_prefill_tokens offset:0 atIndex:1];
    [enc setBuffer:_act_a offset:0 atIndex:2];
    [enc setBytes:&k length:sizeof(k) atIndex:3];
    qw35_dispatch_1d(enc, (NSUInteger)count * _h.embedding_length, 256);
    return YES;
}

/// RMS norm of one act_a row into _norm.
- (BOOL)encodeRms:(id<MTLComputeCommandEncoder>)enc
              src:(id<MTLBuffer>)src
     weightTensor:(Qw35Tensor *)w
            error:(NSError **)error {
    id<MTLComputePipelineState> pipe = _psRms
        ?: [_pipelineCache pipelineNamed:@"qw35_rms_norm_weight_f32" error:error];
    if (!pipe) return NO;
    float eps = _h.rms_epsilon;
    int64_t n64 = (int64_t)_h.embedding_length;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:w.buffer offset:w.offset atIndex:1];
    [enc setBuffer:_norm offset:0 atIndex:2];
    [enc setBytes:&eps length:sizeof(eps) atIndex:3];
    [enc setBytes:&n64 length:sizeof(n64) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return YES;
}

/// Fused act_a += act_b followed by RMS norm into _norm (single row).
- (BOOL)encodeResidualRms:(id<MTLComputeCommandEncoder>)enc
             weightTensor:(Qw35Tensor *)w
                    error:(NSError **)error {
    id<MTLComputePipelineState> pipe = _psResidualRms
        ?: [_pipelineCache pipelineNamed:@"qw35_residual_rms_norm_weight_f32" error:error];
    if (!pipe) return NO;
    float eps = _h.rms_epsilon;
    int64_t n64 = (int64_t)_h.embedding_length;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:_act_a offset:0 atIndex:0];
    [enc setBuffer:_act_b offset:0 atIndex:1];
    [enc setBuffer:w.buffer offset:w.offset atIndex:2];
    [enc setBuffer:_norm offset:0 atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];
    [enc setBytes:&n64 length:sizeof(n64) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return YES;
}

- (BOOL)encodeRmsBatch:(id<MTLComputeCommandEncoder>)enc
          weightTensor:(Qw35Tensor *)w
                tokens:(uint32_t)tokensCount
                 error:(NSError **)error {
    id<MTLComputePipelineState> pipe = _psRmsBatch
        ?: [_pipelineCache pipelineNamed:@"qw35_rms_norm_weight_batch_f32" error:error];
    if (!pipe) return NO;
    float eps = _h.rms_epsilon;
    int64_t n64 = (int64_t)_h.embedding_length;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:_act_a offset:0 atIndex:0];
    [enc setBuffer:w.buffer offset:w.offset atIndex:1];
    [enc setBuffer:_norm offset:0 atIndex:2];
    [enc setBytes:&eps length:sizeof(eps) atIndex:3];
    [enc setBytes:&n64 length:sizeof(n64) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tokensCount, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return YES;
}

/// Fused act_a += act_b followed by RMS norm into _norm (all rows).
- (BOOL)encodeResidualRmsBatch:(id<MTLComputeCommandEncoder>)enc
                  weightTensor:(Qw35Tensor *)w
                        tokens:(uint32_t)tokensCount
                         error:(NSError **)error {
    id<MTLComputePipelineState> pipe = _psResidualRmsBatch
        ?: [_pipelineCache pipelineNamed:@"qw35_residual_rms_norm_weight_batch_f32" error:error];
    if (!pipe) return NO;
    float eps = _h.rms_epsilon;
    int64_t n64 = (int64_t)_h.embedding_length;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:_act_a offset:0 atIndex:0];
    [enc setBuffer:_act_b offset:0 atIndex:1];
    [enc setBuffer:w.buffer offset:w.offset atIndex:2];
    [enc setBuffer:_norm offset:0 atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];
    [enc setBytes:&n64 length:sizeof(n64) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tokensCount, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return YES;
}

/// Single-token quantized matvec. With residual == YES the kernel adds into
/// dst instead of overwriting it (Q4_K, Q6_K, GF4, and GF2 weights only).
/// Unified-.gguf baked FFN tensors carry type_id 100 (GF4) or 101 (GF2);
/// both store one contiguous interleaved stream, so every codec shares the
/// generic bind below (weight, input, dst, n_blocks, k, rows at 0..5).
- (BOOL)encodeDecodeMatvecTensor:(id<MTLComputeCommandEncoder>)enc
                          weight:(Qw35Tensor *)w
                           input:(id<MTLBuffer>)input
                     inputOffset:(NSUInteger)inputOffset
                             dst:(id<MTLBuffer>)dst
                        residual:(BOOL)residual
                           error:(NSError **)error {
    const int64_t k = (int64_t)w.dims[0];
    const int64_t rows = (int64_t)w.dims[1];

    NSString *kernel = nil;
    uint64_t block_elems = 256;
    BOOL q8_geometry = NO;
    switch (w.type_id) {
        case 8:
            kernel = residual ? @"qw35_decode_matmul_q8_0_residual_f32"
                              : @"qw35_decode_matmul_q8_0_f32";
            block_elems = 32;
            q8_geometry = YES;
            break;
        case 12:
            kernel = residual ? @"qw35_decode_matmul_q4_k_2row_residual_f32"
                              : @"qw35_decode_matmul_q4_k_2row_f32";
            break;
        case 13:
            kernel = @"qw35_decode_matmul_q5_k_2row_f32";
            break;
        case 14:
            kernel = residual ? @"qw35_decode_matmul_q6_k_llama_residual_f32"
                              : @"qw35_decode_matmul_q6_k_llama_f32";
            break;
        case 100:
            kernel = residual ? @"qw35_decode_matmul_gf4_2row_residual_f32"
                              : @"qw35_decode_matmul_gf4_2row_f32";
            block_elems = 8;
            break;
        case 101:
            // GF2 kernels take the group count (16 elems per code word) at
            // the n_blocks slot; the interleaved super-block walk is internal.
            kernel = residual ? @"qw35_decode_matmul_gf2_2row_residual_f32"
                              : @"qw35_decode_matmul_gf2_2row_f32";
            block_elems = 16;
            break;
        default:
            if (error) *error = qw35_error(@"unsupported tensor type %u for %@", w.type_id, w.name);
            return NO;
    }
    if (residual && w.type_id != 8 && w.type_id != 12 && w.type_id != 14 && w.type_id != 100 && w.type_id != 101) {
        if (error) *error = qw35_error(@"no residual matvec kernel for tensor type %u (%@)", w.type_id, w.name);
        return NO;
    }

    const int ci = qw35_matvec_codec_index(w.type_id);
    id<MTLComputePipelineState> pipe = ci >= 0 ? _psMatvec[ci][residual ? 1 : 0] : nil;
    if (!pipe) pipe = [_pipelineCache pipelineNamed:kernel error:error];
    if (!pipe) return NO;
    int64_t n_blocks = (int64_t)qw35_div_up_u64((uint64_t)k, block_elems);
    [enc setComputePipelineState:pipe];
    [enc setBuffer:w.buffer offset:w.offset atIndex:0];
    [enc setBuffer:input offset:inputOffset atIndex:1];
    [enc setBuffer:dst offset:0 atIndex:2];
    [enc setBytes:&n_blocks length:sizeof(n_blocks) atIndex:3];
    [enc setBytes:&k length:sizeof(k) atIndex:4];
    [enc setBytes:&rows length:sizeof(rows) atIndex:5];
    if (q8_geometry) {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)qw35_div_up_u64((uint64_t)rows, 8), 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    } else {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)qw35_div_up_u64((uint64_t)rows, 4), 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    }
    return YES;
}

/// Multi-token matvec for prefill: tiled simdgroup matmul for any chunk of
/// two or more tokens; single tokens fall back to the decode matvec path.
- (BOOL)encodeMatvecBatch:(id<MTLComputeCommandEncoder>)enc
                   weight:(Qw35Tensor *)w
                    input:(id<MTLBuffer>)input
              inputOffset:(NSUInteger)inputOffset
                      dst:(id<MTLBuffer>)dst
                dstOffset:(NSUInteger)dstOffset
                   tokens:(uint32_t)tokensCount
                    error:(NSError **)error {
    if (w.n_dims < 2) {
        if (error) *error = qw35_error(@"tensor %@ is not a matrix", w.name);
        return NO;
    }
    int64_t k = (int64_t)w.dims[0];
    int64_t rows = (int64_t)w.dims[1];

    if (tokensCount == 1) {
        return [self encodeDecodeMatvecTensor:enc weight:w input:input inputOffset:inputOffset dst:dst residual:NO error:error];
    }

    NSString *tiledKernel = nil;
    switch (w.type_id) {
        case 8:  tiledKernel = @"qw35_mul_mm_q8_0_f32"; break;
        case 12: tiledKernel = @"qw35_mul_mm_q4_k_f32"; break;
        case 13: tiledKernel = @"qw35_mul_mm_q5_k_f32"; break;
        case 14: tiledKernel = @"qw35_mul_mm_q6_k_f32"; break;
        // Unified .gguf baked FFN codecs: both GF4 (100) and GF2 (101) store
        // 256-element interleaved super-blocks, keeping the k%256 guard and
        // row_bytes=w.bytes/rows valid, so the tiled path below is reused
        // unchanged for either.
        case 100: tiledKernel = @"qw35_mul_mm_gf4_f32"; break;
        case 101: tiledKernel = @"qw35_mul_mm_gf2_f32"; break;
        default:
            if (error) *error = qw35_error(@"unsupported tensor type %u for %@", w.type_id, w.name);
            return NO;
    }
    return [self encodeTiledKMatmul:enc
                             tensor:w
                         kernelName:tiledKernel
                              input:input
                        inputOffset:inputOffset
                                dst:dst
                          dstOffset:dstOffset
                             tokens:tokensCount
                               rows:rows
                                  k:k
                              error:error];
}

- (BOOL)encodeTiledKMatmul:(id<MTLComputeCommandEncoder>)enc
                    tensor:(Qw35Tensor *)w
                kernelName:(NSString *)kernelName
                     input:(id<MTLBuffer>)input
               inputOffset:(NSUInteger)inputOffset
                       dst:(id<MTLBuffer>)dst
                 dstOffset:(NSUInteger)dstOffset
                    tokens:(uint32_t)tokensCount
                      rows:(int64_t)rows
                         k:(int64_t)k
                     error:(NSError **)error {
    if (rows <= 0 || k <= 0 || tokensCount == 0) return YES;
    if (((uint64_t)k % 256u) != 0) {
        if (error) *error = qw35_error(@"K-quant tiled matmul requires k to be a multiple of 256 for %@", w.name);
        return NO;
    }
    const uint64_t row_bytes = rows > 0 ? w.bytes / (uint64_t)rows : 0;
    if (row_bytes == 0) {
        if (error) *error = qw35_error(@"invalid K-quant row bytes for %@", w.name);
        return NO;
    }

    const BOOL bc_inp = ((uint64_t)k % 32u) != 0;
    const BOOL bc_out = ((uint64_t)rows % 64u) != 0 || (tokensCount % 32u) != 0;
    id<MTLComputePipelineState> pipe =
        [_pipelineCache mulMmPipelineNamed:kernelName bcInp:bc_inp bcOut:bc_out error:error];
    if (!pipe) return NO;

    qw35_metal_args_mul_mm args = {
        .ne00 = (int32_t)k,
        .ne02 = 1,
        .nb01 = row_bytes,
        .nb02 = row_bytes * (uint64_t)rows,
        .nb03 = row_bytes * (uint64_t)rows,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = (uint64_t)k * sizeof(float),
        .nb12 = (uint64_t)k * (uint64_t)tokensCount * sizeof(float),
        .nb13 = (uint64_t)k * (uint64_t)tokensCount * sizeof(float),
        .ne0 = (int32_t)rows,
        .ne1 = (int32_t)tokensCount,
        .r2 = 1,
        .r3 = 1,
    };

    [enc setComputePipelineState:pipe];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:w.buffer offset:w.offset atIndex:1];
    [enc setBuffer:input offset:inputOffset atIndex:2];
    [enc setBuffer:dst offset:dstOffset atIndex:3];
    // sa: 64x32 half (4096 B) + sb: 32x32 float (4096 B); the bounded-output
    // staging path reuses the same 8192 B as a 64x32 float tile.
    [enc setThreadgroupMemoryLength:8192u atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)tokensCount + 31u) / 32u,
                                          ((NSUInteger)rows + 63u) / 64u,
                                          1)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return YES;
}

/// SwiGLU over _ffn_gate/_ffn_up into _ffn_gate.
- (BOOL)encodeSwiGLU:(id<MTLComputeCommandEncoder>)enc
                   n:(uint64_t)n
               error:(NSError **)error {
    id<MTLComputePipelineState> pipe = _psSwiglu
        ?: [_pipelineCache pipelineNamed:@"qw35_swiglu_f32" error:error];
    if (!pipe) return NO;
    int64_t n64 = (int64_t)n;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:_ffn_gate offset:0 atIndex:0];
    [enc setBuffer:_ffn_up offset:0 atIndex:1];
    [enc setBuffer:_ffn_gate offset:0 atIndex:2];
    [enc setBytes:&n64 length:sizeof(n64) atIndex:3];
    qw35_dispatch_1d(enc, (NSUInteger)n, 256);
    return YES;
}

@end
