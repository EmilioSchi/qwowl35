// Qw35MetalRuntime+Internal.h – shared private declarations for the
// Qw35MetalRuntime category files (one per layer/stage). NOT part of the
// public API; do not import outside the metal bridge.
//
// The neural-network forward pass is split across category files by stage so
// each layer is recognizable by filename: +Attention.m, +SSM.m, +FFN.m,
// +Output.m, with the core lifecycle/orchestration in Qw35MetalRuntime.m.
// They all need to share the same instance variables and private methods, so
// those live here:
//
//   * Qw35LayerTensors      – per-layer weight holder
//   * the class extension    – every instance variable, declared once
//   * the small C helpers    – static inline so each file gets its own copy
//   * the (Internal) methods – private encoders, visible to every category
//
// Under the modern (non-fragile) Objective-C ABI each category translation
// unit resolves the ivar offsets at runtime against the symbols emitted by the
// single @implementation Qw35MetalRuntime in the core file, so direct _ivar
// access works from every category.

#import "Qw35MetalRuntime.h"
#import "Qw35TensorStore.h"
#import "Qw35PipelineCache.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/// Per-layer weights resolved once at init so the per-token encode loop does
/// no string formatting or dictionary lookups. FFN matvec weights are GF4
/// (type_id 100) in the unified .gguf.
@interface Qw35LayerTensors : NSObject
@property (nonatomic, strong) Qw35Tensor *attnNorm;
@property (nonatomic, strong) Qw35Tensor *postAttentionNorm;
// Pre-FFN norm for decode. The unified .gguf bakes the AWQ fold into
// postAttentionNorm itself, so decode and prefill share it.
@property (nonatomic, strong) Qw35Tensor *postAttentionNormDecode;
// Delta (linear-attention) layers
@property (nonatomic, strong) Qw35Tensor *attnQkv;
@property (nonatomic, strong) Qw35Tensor *attnGate;
@property (nonatomic, strong) Qw35Tensor *ssmBeta;
@property (nonatomic, strong) Qw35Tensor *ssmAlpha;
@property (nonatomic, strong) Qw35Tensor *ssmConv;
@property (nonatomic, strong) Qw35Tensor *ssmDt;
@property (nonatomic, strong) Qw35Tensor *ssmA;
@property (nonatomic, strong) Qw35Tensor *ssmNorm;
@property (nonatomic, strong) Qw35Tensor *ssmOut;
// Full-attention layers
@property (nonatomic, strong) Qw35Tensor *attnQ;
@property (nonatomic, strong) Qw35Tensor *attnK;
@property (nonatomic, strong) Qw35Tensor *attnV;
@property (nonatomic, strong) Qw35Tensor *attnQNorm;
@property (nonatomic, strong) Qw35Tensor *attnKNorm;
@property (nonatomic, strong) Qw35Tensor *attnOutput;
// FFN (GF4 type-id 100 tensors in the unified .gguf)
@property (nonatomic, strong) Qw35Tensor *ffnGate;
@property (nonatomic, strong) Qw35Tensor *ffnUp;
@property (nonatomic, strong) Qw35Tensor *ffnDown;
@end

static inline NSError *qw35_error(NSString *fmt, ...) NS_FORMAT_FUNCTION(1, 2);

static inline NSError *qw35_error(NSString *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    NSString *msg = [[NSString alloc] initWithFormat:fmt arguments:args];
    va_end(args);
    return [NSError errorWithDomain:@"Qw35MetalRuntime"
                               code:-1
                           userInfo:@{NSLocalizedDescriptionKey: msg}];
}

static inline uint64_t qw35_div_up_u64(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

// Accumulate per-channel |x| into a double accumulator (calibration capture).
static inline void qw35_accum_absmean(double *acc, const float *x, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        acc[i] += fabs((double)x[i]);
    }
}

static inline void qw35_dispatch_1d(id<MTLComputeCommandEncoder> enc, NSUInteger n, NSUInteger threads) {
    if (n == 0) return;
    [enc dispatchThreadgroups:MTLSizeMake((n + threads - 1) / threads, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

@interface Qw35MetalRuntime () {
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;
    id<MTLLibrary> _library;
    Qw35TensorStore *_tensorStore;
    Qw35PipelineCache *_pipelineCache;
    qw35_metal_hparams _h;
    uint32_t _ctxSize;      // hard ceiling: max position the cache may ever hold
    uint32_t _kvCapacity;   // positions currently allocated per layer (grows on demand)
    uint32_t _kvSlab;       // growth granularity: capacity grows in slab-sized steps
    uint32_t _vocabSize;
    uint32_t _maxPrefillChunk;
    uint32_t _deltaLayerCount;
    uint32_t _attentionLayerCount;
    BOOL _kvQ8;            // q8_0 KV cache (34-byte blocks of 32) instead of f16
    uint64_t _kvRowBytes;  // bytes per (layer, position) K or V cache row

    // Activation / state buffers. Activations are f32; the KV cache is f16.
    id<MTLBuffer> _act_a;
    id<MTLBuffer> _act_b;
    id<MTLBuffer> _norm;
    id<MTLBuffer> _qkv;
    id<MTLBuffer> _z_gate;
    id<MTLBuffer> _beta;
    id<MTLBuffer> _alpha;
    id<MTLBuffer> _q_rep;
    id<MTLBuffer> _k_rep;
    id<MTLBuffer> _core;
    id<MTLBuffer> _ffn_gate;
    id<MTLBuffer> _ffn_up;
    id<MTLBuffer> _logits;
    id<MTLBuffer> _k_cache;
    id<MTLBuffer> _v_cache;
    id<MTLBuffer> _conv_state;
    id<MTLBuffer> _ssm_state;
    // Session checkpoint copies of the recurrent state (CPU-side, small).
    NSMutableData *_conv_state_ckpt;
    NSMutableData *_ssm_state_ckpt;
    BOOL _has_state_ckpt;
    id<MTLBuffer> _argmax_token;
    id<MTLBuffer> _argmax_logit;
    id<MTLBuffer> _argmax_partial_token;
    id<MTLBuffer> _argmax_partial_logit;
    id<MTLBuffer> _prefill_tokens;
    id<MTLBuffer> _rope_freq;

    // Per-layer decode weights (GF4-preferred), resolved once at init.
    NSArray<Qw35LayerTensors *> *_layers;
    Qw35Tensor *_tokenEmbd;
    Qw35Tensor *_outputNorm;
    Qw35Tensor *_outputWeight; // GF4-preferred
    BOOL _outputWeightIsGf4;

    // Last committed command buffer of the most recent eval.
    id<MTLCommandBuffer> _lastCB;

    // Calibration activation capture (QW35_CAPTURE_ACT_OUT). When enabled,
    // decode runs a serialised per-layer path that reads back the FFN inputs
    // and accumulates per-input-channel mean-abs magnitudes: the gate/up input
    // (`_norm`, dim embedding_length) and the down input (`_ffn_gate` after
    // SwiGLU, dim feed_forward_length). Used offline to drive AWQ scaling.
    BOOL _capEnabled;
    char *_capPath;          // output file (strdup of the env var)
    double *_capGateUp;      // [transformer_layers * embedding_length]
    double *_capDown;        // [transformer_layers * feed_forward_length]
    uint64_t _capTokens;

    // Per-stage GPU-time profiler (QW35_STAGE_PROFILE). Decode runs a serialised
    // per-stage path that commits ONE command buffer per stage class and sums
    // (GPUEndTime - GPUStartTime), attributing true on-GPU decode time to
    // attention vs delta vs FFN vs head as context grows. Buckets reset every 16
    // tokens so the printed us/tok reflects the CURRENT ctx (the trend is the
    // point). Diagnostic only — serialised, so absolute tok/s is NOT comparable.
    BOOL _stageProfile;
    double _prof_attn, _prof_delta, _prof_ffn, _prof_head, _prof_norm;
    uint64_t _prof_tokens;
    uint32_t _prof_pos;

    // Decode-time sliding-window attention for the full-attention layers, from
    // the --attn-window/--attn-sink CLI args (via qw35_metal_hparams). window <= 0
    // means full attention; otherwise each attention layer attends to the first
    // `sink` positions plus the last `window` positions, bounding the O(seq_len)
    // decode loop.
    int _attnWindow;
    int _attnSink;

    // MTLResidencySet (macOS 15+) pinning the mmap-backed weight + KV + scratch
    // buffers GPU-resident, so the driver does not page them out under
    // unified-memory pressure during a long session (decode is weight-bandwidth
    // bound; an evicted weight page re-faults from disk every token). Always set
    // up (nil only on older OSes). Held for the runtime's lifetime and updated
    // in place when the KV cache grows (see -ensureKvCapacityForPositions:).
    id<MTLResidencySet> _residencySet;
}
@end

// Private methods shared across the Qw35MetalRuntime category files.
@interface Qw35MetalRuntime (Internal)
- (instancetype)initWithModelMap:(const void *)modelMap
                       modelSize:(uint64_t)modelSize
                          tensors:(const qw35_metal_tensor_desc *)tensorDescs
                      tensorCount:(uintptr_t)tensorCount
                          hparams:(const qw35_metal_hparams *)hparams
                          ctxSize:(uint32_t)ctxSize
                        vocabSize:(uint32_t)vocabSize
                     prefillChunk:(uint32_t)prefillChunk
                      kvCacheType:(uint32_t)kvCacheType
                         metallib:(const uint8_t *)metallib
                      metallibLen:(uintptr_t)metallibLen
                            error:(NSError **)error;
- (void)setupResidency;
- (Qw35Tensor *)decodeWeightNamed:(NSString *)name error:(NSError **)error;
- (BOOL)resolveLayerTensors:(NSError **)error;
- (BOOL)allocateBuffers:(NSError **)error;
- (BOOL)ensureKvCapacityForPositions:(uint64_t)positions error:(NSError **)error;
- (id<MTLBuffer>)newFloatBuffer:(uint64_t)count label:(NSString *)label;
- (BOOL)prewarmPipelines:(NSError **)error;
- (id<MTLComputePipelineState>)attnPipeline:(NSString *)name error:(NSError **)error;
- (BOOL)waitForLastCommand:(NSError **)error;
- (BOOL)sync:(NSError **)error;
- (void)setAttnSink:(int)sink;
- (BOOL)reset:(NSError **)error;
- (BOOL)stateCheckpointSave:(NSError **)error;
- (BOOL)stateCheckpointRestore:(NSError **)error;
- (BOOL)initializeRopeFrequencies:(NSError **)error;
- (Qw35Tensor *)tensorNamed:(NSString *)name error:(NSError **)error;
- (BOOL)evalToken:(uint32_t)token
              pos:(uint32_t)pos
       logitsMode:(qw35_logits_mode)logitsMode
            error:(NSError **)error;
- (BOOL)captureRun:(BOOL (^)(id<MTLComputeCommandEncoder> enc, NSError **error))block
             error:(NSError **)error;
- (void)writeCaptureFile;
- (BOOL)captureEvalToken:(uint32_t)token
                     pos:(uint32_t)pos
              logitsMode:(qw35_logits_mode)logitsMode
                   error:(NSError **)error;
- (BOOL)profStage:(BOOL (^)(id<MTLComputeCommandEncoder> enc, NSError **error))block
            accum:(double *)accum
            error:(NSError **)error;
- (void)printStageProfile;
- (BOOL)evalTokenStageProfiled:(uint32_t)token
                           pos:(uint32_t)pos
                    logitsMode:(qw35_logits_mode)logitsMode
                         error:(NSError **)error;
- (BOOL)encodeDecodeLayers:(id<MTLComputeCommandEncoder>)enc
                     token:(uint32_t)token
                       pos:(uint32_t)pos
                layerBegin:(uint32_t)begin
                  layerEnd:(uint32_t)end
                 deltaSlot:(uint32_t)deltaSlot
                  attnSlot:(uint32_t)attnSlot
                logitsMode:(qw35_logits_mode)logitsMode
                     error:(NSError **)error;
- (BOOL)encodeDeltaDecodeLayer:(id<MTLComputeCommandEncoder>)enc
                         layer:(Qw35LayerTensors *)layer
                          slot:(uint32_t)slot
                         error:(NSError **)error;
- (BOOL)encodeGf4FusedFfn:(id<MTLComputeCommandEncoder>)enc
                    layer:(Qw35LayerTensors *)layer
                    error:(NSError **)error;
- (BOOL)encodeAttentionDecodeLayer:(id<MTLComputeCommandEncoder>)enc
                             layer:(Qw35LayerTensors *)layer
                              slot:(uint32_t)slot
                               pos:(uint32_t)pos
                             error:(NSError **)error;
- (BOOL)encodeOutputHead:(id<MTLComputeCommandEncoder>)enc
           normRowOffset:(NSUInteger)normRowOffset
              logitsMode:(qw35_logits_mode)logitsMode
                   error:(NSError **)error;
- (BOOL)evalTokens:(const uint32_t *)tokens
             count:(uintptr_t)count
              pos0:(uint32_t)pos0
        logitsMode:(qw35_logits_mode)logitsMode
             error:(NSError **)error;
- (BOOL)encodeDeltaPrefillLayer:(id<MTLComputeCommandEncoder>)enc
                           slot:(uint32_t)slot
                         tokens:(uint32_t)tokensCount
                         prefix:(NSString *)prefix
                          error:(NSError **)error;
- (BOOL)encodeAttentionPrefillLayer:(id<MTLComputeCommandEncoder>)enc
                               slot:(uint32_t)slot
                               pos0:(uint32_t)pos0
                             tokens:(uint32_t)tokensCount
                             prefix:(NSString *)prefix
                              error:(NSError **)error;
- (BOOL)encodeEmbedding:(id<MTLComputeCommandEncoder>)enc
                  token:(uint32_t)token
                  error:(NSError **)error;
- (BOOL)encodeEmbeddingBatch:(id<MTLComputeCommandEncoder>)enc
                       count:(uint32_t)count
                       error:(NSError **)error;
- (BOOL)encodeRms:(id<MTLComputeCommandEncoder>)enc
              src:(id<MTLBuffer>)src
     weightTensor:(Qw35Tensor *)w
            error:(NSError **)error;
- (BOOL)encodeResidualRms:(id<MTLComputeCommandEncoder>)enc
             weightTensor:(Qw35Tensor *)w
                    error:(NSError **)error;
- (BOOL)encodeRmsBatch:(id<MTLComputeCommandEncoder>)enc
                weight:(NSString *)weightName
                tokens:(uint32_t)tokensCount
                 error:(NSError **)error;
- (BOOL)encodeResidualRmsBatch:(id<MTLComputeCommandEncoder>)enc
                        weight:(NSString *)weightName
                        tokens:(uint32_t)tokensCount
                         error:(NSError **)error;
- (BOOL)encodeDecodeMatvecTensor:(id<MTLComputeCommandEncoder>)enc
                          weight:(Qw35Tensor *)w
                           input:(id<MTLBuffer>)input
                     inputOffset:(NSUInteger)inputOffset
                             dst:(id<MTLBuffer>)dst
                        residual:(BOOL)residual
                           error:(NSError **)error;
- (BOOL)encodeMatvecBatch:(id<MTLComputeCommandEncoder>)enc
                   weight:(NSString *)weightName
                    input:(id<MTLBuffer>)input
              inputOffset:(NSUInteger)inputOffset
                      dst:(id<MTLBuffer>)dst
                dstOffset:(NSUInteger)dstOffset
                   tokens:(uint32_t)tokensCount
                    error:(NSError **)error;
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
                     error:(NSError **)error;
- (BOOL)encodeSwiGLU:(id<MTLComputeCommandEncoder>)enc
                   n:(uint64_t)n
               error:(NSError **)error;
- (BOOL)readArgmaxToken:(uint32_t *)token logit:(float *)logit error:(NSError **)error;
- (BOOL)copyLogits:(float *)dst len:(uintptr_t)len error:(NSError **)error;
@end
