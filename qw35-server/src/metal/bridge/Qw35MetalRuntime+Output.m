// Qw35MetalRuntime+Output.m – output head (logits / fused argmax) and CPU readback.
//
// Split out of the monolithic Qw35MetalRuntime.m so this stage of the
// Qwen3.5 forward pass is recognizable by filename. Shares the runtime's
// instance variables, helpers, and private methods via the category header.

#import "Qw35MetalRuntime+Internal.h"

@implementation Qw35MetalRuntime (Output)

- (BOOL)encodeOutputHead:(id<MTLComputeCommandEncoder>)enc
           normRowOffset:(NSUInteger)normRowOffset
              logitsMode:(qw35_logits_mode)logitsMode
                   error:(NSError **)error {
    if (logitsMode == QW35_LOGITS_FULL) {
        return [self encodeDecodeMatvecTensor:enc weight:_outputWeight input:_norm inputOffset:normRowOffset dst:_logits residual:NO error:error];
    }

    // A classification head (cls.output.weight, n_cls_out rows) only ever
    // serves full-logits scoring; the fused vocab argmax below assumes the
    // LM head layout and is meaningless for it.
    if (_h.n_cls_out > 0) {
        if (error) *error = qw35_error(@"classification-head model supports only full logits readback");
        return NO;
    }

    // Greedy: fused matvec + per-threadgroup argmax over 16 vocab rows, then
    // a single reduction over the partials. Prefers GF4 output weights.
    const int64_t rows = (int64_t)_outputWeight.dims[1];
    const int64_t k = (int64_t)_h.embedding_length;
    const int64_t groups = (int64_t)qw35_div_up_u64((uint64_t)rows, 16);

    NSString *partialsKernel = _outputWeightIsGf4
        ? @"qw35_output_gf4_argmax_partials_16row_f32"
        : @"qw35_output_q6_k_argmax_partials_16row_f32";
    const int64_t n_blocks =
        _outputWeightIsGf4 ? k / 8 : (int64_t)qw35_div_up_u64((uint64_t)k, 256);

    id<MTLComputePipelineState> partials = _psArgmaxPartials
        ?: [_pipelineCache pipelineNamed:partialsKernel error:error];
    if (!partials) return NO;
    [enc setComputePipelineState:partials];
    [enc setBuffer:_outputWeight.buffer offset:_outputWeight.offset atIndex:0];
    [enc setBuffer:_norm offset:normRowOffset atIndex:1];
    [enc setBuffer:_argmax_partial_token offset:0 atIndex:2];
    [enc setBuffer:_argmax_partial_logit offset:0 atIndex:3];
    [enc setBytes:&n_blocks length:sizeof(n_blocks) atIndex:4];
    [enc setBytes:&k length:sizeof(k) atIndex:5];
    [enc setBytes:&rows length:sizeof(rows) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];

    id<MTLComputePipelineState> reduce = _psArgmaxReduce
        ?: [_pipelineCache pipelineNamed:@"qw35_output_argmax_reduce_partials_f32" error:error];
    if (!reduce) return NO;
    [enc setComputePipelineState:reduce];
    [enc setBuffer:_argmax_partial_token offset:0 atIndex:0];
    [enc setBuffer:_argmax_partial_logit offset:0 atIndex:1];
    [enc setBuffer:_argmax_token offset:0 atIndex:2];
    [enc setBuffer:_argmax_logit offset:0 atIndex:3];
    [enc setBytes:&groups length:sizeof(groups) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return YES;
}

// ---------------------------------------------------------------------------
#pragma mark - Readback
// ---------------------------------------------------------------------------

- (BOOL)readArgmaxToken:(uint32_t *)token logit:(float *)logit error:(NSError **)error {
    if (!token || !logit) {
        if (error) *error = qw35_error(@"invalid argmax output pointers");
        return NO;
    }
    if (![self waitForLastCommand:error]) return NO;
    *token = *((uint32_t *)[_argmax_token contents]);
    *logit = *((float *)[_argmax_logit contents]);
    return YES;
}

- (BOOL)copyLogits:(float *)dst len:(uintptr_t)len error:(NSError **)error {
    const uintptr_t n = _h.n_cls_out > 0 ? _h.n_cls_out : _vocabSize;
    if (!dst || len < n) {
        if (error) *error = qw35_error(@"invalid logits readback buffer");
        return NO;
    }
    if (![self waitForLastCommand:error]) return NO;
    memcpy(dst, [_logits contents], (size_t)n * sizeof(float));
    return YES;
}

@end
