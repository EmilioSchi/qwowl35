// Qw35MetalRuntime.m – GPU inference orchestrator for the Qwen3.5-9B model.
//
// Decode path (one token): two command buffers per token — layers 0..mid in
// the first, layers mid..n plus the output head in the second. The first is
// committed while the CPU is still encoding the second, and the host never
// waits inside eval; readers (-readArgmaxToken:..., -copyLogits:...) wait.
//
// Per layer the residual add is folded into the down/output projection
// (residual matvec kernels) or into the following RMS norm (fused
// residual+norm kernel), and the next layer's input norm is encoded at the
// tail of the previous layer, so the residual stream `act_a` is only touched
// by fused kernels. The KV cache is f16. Greedy decode uses a fused
// Q6_K-matvec+argmax output head and never materializes the logits vector.

#import "Qw35MetalRuntime.h"
#import "Qw35TensorStore.h"
#import "Qw35PipelineCache.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Positions allocated for the KV cache at model load, and the granularity by
// which it grows (in slab-sized steps, capped at --ctx) as the live context
// advances. A short chat keeps only a slab resident — a few hundred MiB instead
// of the full --ctx worth of cache (Metal makes the whole buffer resident on
// first use, so a smaller buffer is a real physical-memory saving).
#define QW35_KV_INITIAL_SLAB 8192u

static NSError *qw35_error(NSString *fmt, ...) NS_FORMAT_FUNCTION(1, 2);

static NSError *qw35_error(NSString *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    NSString *msg = [[NSString alloc] initWithFormat:fmt arguments:args];
    va_end(args);
    return [NSError errorWithDomain:@"Qw35MetalRuntime"
                               code:-1
                           userInfo:@{NSLocalizedDescriptionKey: msg}];
}

static uint64_t qw35_div_up_u64(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

static void qw35_dispatch_1d(id<MTLComputeCommandEncoder> enc, NSUInteger n, NSUInteger threads) {
    if (n == 0) return;
    [enc dispatchThreadgroups:MTLSizeMake((n + threads - 1) / threads, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

/// Per-layer weights resolved once at init so the per-token encode loop does
/// no string formatting or dictionary lookups. Matvec weights prefer the GF4
/// sidecar tensor (type_id 100) when one exists.
@interface Qw35LayerTensors : NSObject
@property (nonatomic, strong) Qw35Tensor *attnNorm;
@property (nonatomic, strong) Qw35Tensor *postAttentionNorm;
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
// FFN (GF4 tensors when a sidecar is loaded)
@property (nonatomic, strong) Qw35Tensor *ffnGate;
@property (nonatomic, strong) Qw35Tensor *ffnUp;
@property (nonatomic, strong) Qw35Tensor *ffnDown;
@end

@implementation Qw35LayerTensors
@end

@interface Qw35MetalRuntime () {
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;
    id<MTLLibrary> _library;
    Qw35TensorStore *_tensorStore;
    Qw35TensorStore *_gf4Store; // nil without a cooked GF4 sidecar
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
}
@end

@implementation Qw35MetalRuntime

- (instancetype)initWithModelMap:(const void *)modelMap
                       modelSize:(uint64_t)modelSize
                          tensors:(const qw35_metal_tensor_desc *)tensorDescs
                      tensorCount:(uintptr_t)tensorCount
                          gf4Map:(const void *)gf4Map
                         gf4Size:(uint64_t)gf4Size
                      gf4Tensors:(const qw35_metal_tensor_desc *)gf4Tensors
                  gf4TensorCount:(uintptr_t)gf4TensorCount
                          hparams:(const qw35_metal_hparams *)hparams
                          ctxSize:(uint32_t)ctxSize
                        vocabSize:(uint32_t)vocabSize
                     prefillChunk:(uint32_t)prefillChunk
                      kvCacheType:(uint32_t)kvCacheType
                         metallib:(const uint8_t *)metallib
                      metallibLen:(uintptr_t)metallibLen
                            error:(NSError **)error {
    self = [super init];
    if (!self) return nil;
    if (!modelMap || !modelSize || !tensorDescs || !tensorCount || !hparams || !metallib || !metallibLen) {
        if (error) *error = qw35_error(@"invalid runtime creation inputs");
        return nil;
    }

    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        if (error) *error = qw35_error(@"no default Metal device is available");
        return nil;
    }
    _queue = [_device newCommandQueue];
    if (!_queue) {
        if (error) *error = qw35_error(@"failed to create Metal command queue");
        return nil;
    }

    dispatch_data_t libData = dispatch_data_create(metallib,
                                                   (size_t)metallibLen,
                                                   nil,
                                                   DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError *nsError = nil;
    _library = [_device newLibraryWithData:libData error:&nsError];
    if (!_library) {
        if (error) *error = nsError ?: qw35_error(@"failed to load Qw35 metallib");
        return nil;
    }

    _h = *hparams;
    _ctxSize = ctxSize;
    // Start with a slab and grow on demand; never reserve more than the ceiling.
    const uint32_t slab = QW35_KV_INITIAL_SLAB;
    _kvSlab = slab;
    _kvCapacity = ctxSize < slab ? ctxSize : slab;
    _vocabSize = vocabSize;
    _maxPrefillChunk = prefillChunk ? prefillChunk : 1;
    _kvQ8 = kvCacheType == 1;
    const uint64_t kv_dim_init = (uint64_t)_h.attention_kv_heads * _h.attention_key_length;
    if (_kvQ8 && (kv_dim_init % 32) != 0) {
        if (error) *error = qw35_error(@"q8_0 KV cache requires kv_dim to be a multiple of 32");
        return nil;
    }
    _kvRowBytes = _kvQ8 ? (kv_dim_init / 32) * 34 : kv_dim_init * sizeof(uint16_t);

    _tensorStore = [[Qw35TensorStore alloc] initWithModelMap:modelMap
                                                   modelSize:modelSize
                                                      tensors:tensorDescs
                                                  tensorCount:tensorCount
                                                       device:_device
                                                        error:error];
    if (!_tensorStore) return nil;
    if (![_tensorStore validateRequiredForEmbeddingLength:_h.embedding_length
                                                vocabSize:_vocabSize
                                                    error:error]) {
        return nil;
    }
    Qw35Tensor *outputWeight = [_tensorStore tensorNamed:@"output.weight"];
    if (outputWeight.type_id != 14) {
        if (error) *error = qw35_error(@"output.weight must be q6_k for the fused argmax head");
        return nil;
    }

    if (gf4Map && gf4Size && gf4Tensors && gf4TensorCount) {
        _gf4Store = [[Qw35TensorStore alloc] initWithModelMap:gf4Map
                                                    modelSize:gf4Size
                                                       tensors:gf4Tensors
                                                   tensorCount:gf4TensorCount
                                                        device:_device
                                                         error:error];
        if (!_gf4Store) return nil;
    }

    _pipelineCache = [[Qw35PipelineCache alloc] initWithLibrary:_library device:_device];

    if (![self resolveLayerTensors:error]) return nil;
    if (![self allocateBuffers:error]) return nil;
    if (![self prewarmPipelines:error]) return nil;
    if (![self reset:error]) return nil;
    return self;
}

/// A decode matvec weight: the GF4 sidecar tensor when present, else GGUF.
- (Qw35Tensor *)decodeWeightNamed:(NSString *)name error:(NSError **)error {
    Qw35Tensor *gf4 = [_gf4Store tensorNamed:name];
    if (gf4) return gf4;
    return [self tensorNamed:name error:error];
}

- (BOOL)resolveLayerTensors:(NSError **)error {
    _tokenEmbd = [self tensorNamed:@"token_embd.weight" error:error];
    _outputNorm = [self tensorNamed:@"output_norm.weight" error:error];
    if (!_tokenEmbd || !_outputNorm) return NO;
    _outputWeight = [self decodeWeightNamed:@"output.weight" error:error];
    if (!_outputWeight) return NO;
    _outputWeightIsGf4 = _outputWeight.type_id == 100;

    const uint32_t interval = _h.full_attention_interval ? _h.full_attention_interval : 4;
    NSMutableArray<Qw35LayerTensors *> *layers =
        [NSMutableArray arrayWithCapacity:_h.transformer_layers];
    for (uint32_t il = 0; il < _h.transformer_layers; il++) {
        NSString *prefix = [NSString stringWithFormat:@"blk.%u.", il];
        Qw35Tensor *(^plain)(NSString *) = ^Qw35Tensor *(NSString *suffix) {
            return [self tensorNamed:[prefix stringByAppendingString:suffix] error:error];
        };
        Qw35Tensor *(^weight)(NSString *) = ^Qw35Tensor *(NSString *suffix) {
            return [self decodeWeightNamed:[prefix stringByAppendingString:suffix] error:error];
        };

        Qw35LayerTensors *layer = [Qw35LayerTensors new];
        layer.attnNorm = plain(@"attn_norm.weight");
        layer.postAttentionNorm = plain(@"post_attention_norm.weight");
        layer.ffnGate = weight(@"ffn_gate.weight");
        layer.ffnUp = weight(@"ffn_up.weight");
        layer.ffnDown = weight(@"ffn_down.weight");
        if (!layer.attnNorm || !layer.postAttentionNorm || !layer.ffnGate || !layer.ffnUp ||
            !layer.ffnDown) {
            return NO;
        }

        if (((il + 1) % interval) == 0) {
            layer.attnQ = weight(@"attn_q.weight");
            layer.attnK = weight(@"attn_k.weight");
            layer.attnV = weight(@"attn_v.weight");
            layer.attnQNorm = plain(@"attn_q_norm.weight");
            layer.attnKNorm = plain(@"attn_k_norm.weight");
            layer.attnOutput = weight(@"attn_output.weight");
            if (!layer.attnQ || !layer.attnK || !layer.attnV || !layer.attnQNorm ||
                !layer.attnKNorm || !layer.attnOutput) {
                return NO;
            }
        } else {
            layer.attnQkv = weight(@"attn_qkv.weight");
            layer.attnGate = weight(@"attn_gate.weight");
            layer.ssmBeta = weight(@"ssm_beta.weight");
            layer.ssmAlpha = weight(@"ssm_alpha.weight");
            layer.ssmConv = plain(@"ssm_conv1d.weight");
            layer.ssmDt = plain(@"ssm_dt.bias");
            layer.ssmA = plain(@"ssm_a");
            layer.ssmNorm = plain(@"ssm_norm.weight");
            layer.ssmOut = weight(@"ssm_out.weight");
            if (!layer.attnQkv || !layer.attnGate || !layer.ssmBeta || !layer.ssmAlpha ||
                !layer.ssmConv || !layer.ssmDt || !layer.ssmA || !layer.ssmNorm ||
                !layer.ssmOut) {
                return NO;
            }
        }
        [layers addObject:layer];
    }
    _layers = layers;
    return YES;
}

- (BOOL)allocateBuffers:(NSError **)error {
    const uint64_t emb = _h.embedding_length;
    const uint64_t ffn = _h.feed_forward_length;
    const uint64_t ssm_conv_channels = _h.ssm_inner_size + 2ull * _h.ssm_group_count * _h.ssm_state_size;
    const uint64_t ssm_value = _h.ssm_time_step_rank * _h.ssm_state_size;
    const uint64_t attn_q = _h.attention_heads * _h.attention_key_length * 2ull;
    const uint64_t attn_out = _h.attention_heads * _h.attention_value_length;
    const uint64_t attn_q_row = _h.attention_heads * _h.attention_key_length;
    const uint64_t kv_dim = _h.attention_kv_heads * _h.attention_key_length;
    const uint64_t chunk = _maxPrefillChunk ? _maxPrefillChunk : 1;
    const uint64_t qkv_max = attn_q > ssm_conv_channels ? attn_q : ssm_conv_channels;
    const uint64_t stream_max = attn_out > ssm_value ? attn_out : ssm_value;
    const uint64_t q_rep_max = attn_q_row > ssm_value ? attn_q_row : ssm_value;
    const uint64_t ffn_max = ffn > kv_dim ? ffn : kv_dim;

    _deltaLayerCount = 0;
    _attentionLayerCount = 0;
    const uint32_t interval = _h.full_attention_interval ? _h.full_attention_interval : 4;
    for (uint32_t il = 0; il < _h.transformer_layers; il++) {
        if (((il + 1) % interval) == 0) _attentionLayerCount++;
        else _deltaLayerCount++;
    }

    _act_a = [self newFloatBuffer:emb * chunk label:@"act_a"];
    _act_b = [self newFloatBuffer:emb * chunk label:@"act_b"];
    _norm = [self newFloatBuffer:emb * chunk label:@"norm"];
    _qkv = [self newFloatBuffer:qkv_max * chunk label:@"qkv"];
    _z_gate = [self newFloatBuffer:stream_max * chunk label:@"z_gate"];
    _beta = [self newFloatBuffer:(uint64_t)_h.ssm_time_step_rank * chunk label:@"beta"];
    _alpha = [self newFloatBuffer:(uint64_t)_h.ssm_time_step_rank * chunk label:@"alpha"];
    _q_rep = [self newFloatBuffer:q_rep_max * chunk label:@"q_rep"];
    _k_rep = [self newFloatBuffer:ssm_value * chunk label:@"k_rep"];
    _core = [self newFloatBuffer:stream_max * chunk label:@"core"];
    _ffn_gate = [self newFloatBuffer:ffn_max * chunk label:@"ffn_gate"];
    _ffn_up = [self newFloatBuffer:ffn_max * chunk label:@"ffn_up"];
    _logits = [self newFloatBuffer:_vocabSize label:@"logits"];
    // KV cache: f16 rows (2 bytes/element) or q8_0 rows (34-byte blocks of 32).
    // Sized to the current capacity (a slab), not the full --ctx ceiling;
    // -ensureKvCapacityForPositions: grows it as the live context advances.
    const uint64_t kv_bytes = (uint64_t)_attentionLayerCount * _kvCapacity * _kvRowBytes;
    _k_cache = [_device newBufferWithLength:(NSUInteger)kv_bytes
                                    options:MTLResourceStorageModeShared];
    _k_cache.label = @"k_cache";
    _v_cache = [_device newBufferWithLength:(NSUInteger)kv_bytes
                                    options:MTLResourceStorageModeShared];
    _v_cache.label = @"v_cache";
    _conv_state = [self newFloatBuffer:(uint64_t)_deltaLayerCount * ssm_conv_channels * (_h.ssm_conv_kernel - 1) label:@"conv_state"];
    _ssm_state = [self newFloatBuffer:(uint64_t)_deltaLayerCount * _h.ssm_time_step_rank * _h.ssm_state_size * _h.ssm_state_size label:@"ssm_state"];
    _argmax_token = [_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    _argmax_logit = [_device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    const uint64_t argmax_groups = qw35_div_up_u64(_vocabSize, 16);
    _argmax_partial_token = [_device newBufferWithLength:(NSUInteger)(argmax_groups * sizeof(uint32_t))
                                                 options:MTLResourceStorageModeShared];
    _argmax_partial_token.label = @"argmax_partial_token";
    _argmax_partial_logit = [_device newBufferWithLength:(NSUInteger)(argmax_groups * sizeof(float))
                                                 options:MTLResourceStorageModeShared];
    _argmax_partial_logit.label = @"argmax_partial_logit";
    _prefill_tokens = [_device newBufferWithLength:(NSUInteger)(chunk * sizeof(uint32_t))
                                           options:MTLResourceStorageModeShared];
    _rope_freq = [_device newBufferWithLength:(NSUInteger)((_h.rope_dimension_count / 2u) * sizeof(float))
                                      options:MTLResourceStorageModeShared];
    _rope_freq.label = @"rope_freq";

    if (!_act_a || !_act_b || !_norm || !_qkv || !_z_gate || !_beta || !_alpha || !_q_rep ||
        !_k_rep || !_core || !_ffn_gate || !_ffn_up || !_logits || !_k_cache || !_v_cache ||
        !_conv_state || !_ssm_state || !_argmax_token || !_argmax_logit ||
        !_argmax_partial_token || !_argmax_partial_logit || !_prefill_tokens || !_rope_freq) {
        if (error) *error = qw35_error(@"failed to allocate Qw35 Metal decode buffers");
        return NO;
    }
    if (![self initializeRopeFrequencies:error]) return NO;
    return YES;
}

// Ensure the KV cache can address `positions` positions per layer, growing it
// if needed. The per-layer stride is _kvCapacity, so growth re-lays-out every
// layer's slice into a wider buffer; callers have already verified
// `positions <= _ctxSize`. No-op once the cache has reached the ceiling or the
// slab still has room.
//
// Growth is by whole slabs (not doubling): doubling over-reserves up to ~2x and
// maximizes the transient old+new peak, which on unified memory can evict the
// model's weight pages — and decode here is weight-bandwidth bound, so that
// shows up as a *sustained* tok/s drop, not just a hitch. Slab-stepping keeps
// the spike bounded while staying amortized.
//
// Growth is rare (once per slab crossed) so the brief drain below is cheap; the
// expensive part the old code got wrong was doubling, not the wait.
- (BOOL)ensureKvCapacityForPositions:(uint64_t)positions error:(NSError **)error {
    if (positions <= (uint64_t)_kvCapacity) return YES;

    // Round up to the next slab boundary, clamped to the --ctx ceiling.
    const uint64_t slab = _kvSlab ? (uint64_t)_kvSlab : 1;
    uint64_t newCap = ((positions + slab - 1) / slab) * slab;
    if (newCap > (uint64_t)_ctxSize) newCap = _ctxSize;
    if (newCap <= (uint64_t)_kvCapacity) return YES; // already at the ceiling

    const uint64_t rowBytes = _kvRowBytes;
    const uint64_t oldStride = (uint64_t)_kvCapacity * rowBytes;
    const uint64_t newStride = newCap * rowBytes;
    const uint64_t newBytes = (uint64_t)_attentionLayerCount * newStride;

    id<MTLBuffer> newK = [_device newBufferWithLength:(NSUInteger)newBytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> newV = [_device newBufferWithLength:(NSUInteger)newBytes
                                              options:MTLResourceStorageModeShared];
    if (!newK || !newV) {
        if (error) *error = qw35_error(@"failed to grow Qw35 KV cache to %llu positions", newCap);
        return NO;
    }
    newK.label = @"k_cache";
    newV.label = @"v_cache";

    // The old buffers may still be read/written by an in-flight command buffer
    // (built with commandBufferWithUnretainedReferences). Drain it before
    // copying so the snapshot is consistent and the old buffers are safe to
    // release via ARC when the ivars are reassigned below.
    if (![self waitForLastCommand:error]) return NO;

    // Re-lay-out each layer's existing rows at the new (wider) per-layer stride.
    // Copying the full old slab is safe: rows beyond the live position are never
    // read (the attention loop is bounded by seq_len).
    const uint8_t *oldK = (const uint8_t *)[_k_cache contents];
    const uint8_t *oldV = (const uint8_t *)[_v_cache contents];
    uint8_t *dstK = (uint8_t *)[newK contents];
    uint8_t *dstV = (uint8_t *)[newV contents];
    for (uint64_t slot = 0; slot < _attentionLayerCount; slot++) {
        memcpy(dstK + slot * newStride, oldK + slot * oldStride, (size_t)oldStride);
        memcpy(dstV + slot * newStride, oldV + slot * oldStride, (size_t)oldStride);
    }

    _k_cache = newK;
    _v_cache = newV;
    _kvCapacity = (uint32_t)newCap;
    return YES;
}

- (id<MTLBuffer>)newFloatBuffer:(uint64_t)count label:(NSString *)label {
    if (count == 0 || count > (UINT64_MAX / sizeof(float))) return nil;
    id<MTLBuffer> buffer = [_device newBufferWithLength:(NSUInteger)(count * sizeof(float))
                                                options:MTLResourceStorageModeShared];
    buffer.label = label;
    return buffer;
}

- (BOOL)prewarmPipelines:(NSError **)error {
    static NSString *const plain[] = {
        @"qw35_rms_norm_weight_f32",
        @"qw35_rms_norm_weight_batch_f32",
        @"qw35_residual_rms_norm_weight_f32",
        @"qw35_residual_rms_norm_weight_batch_f32",
        @"qw35_swiglu_f32",
        @"qw35_get_row_q4_k_f32",
        @"qw35_get_rows_q4_k_f32",
        @"qw35_decode_matmul_q8_0_f32",
        @"qw35_decode_matmul_q4_k_2row_f32",
        @"qw35_decode_matmul_q4_k_2row_residual_f32",
        @"qw35_decode_matmul_q5_k_2row_f32",
        @"qw35_decode_matmul_q6_k_llama_f32",
        @"qw35_decode_matmul_q6_k_llama_residual_f32",
        @"qw35_ssm_conv_recurrent_gate_norm_step128_f32",
        @"qw35_ssm_conv1d_step4_batch_f32",
        @"qw35_ssm_l2_repeat_qk_batch_f32",
        @"qw35_ssm_recurrent_step128_batch_rows_f32",
        @"qw35_ssm_gate_norm_batch_f32",
        @"qw35_output_q6_k_argmax_partials_16row_f32",
        @"qw35_output_argmax_reduce_partials_f32",
    };
    for (size_t i = 0; i < sizeof(plain) / sizeof(plain[0]); i++) {
        if (![_pipelineCache pipelineNamed:plain[i] error:error]) return NO;
    }
    static NSString *const attn_f16[] = {
        @"qw35_attn_decode_preprocess_f32",
        @"qw35_attn_prefill_preprocess_f32",
        @"qw35_attention_gqa_flash_decode_f32",
        @"qw35_attention_gqa_prefill_f32",
    };
    static NSString *const attn_q8[] = {
        @"qw35_attn_decode_preprocess_q8_0_f32",
        @"qw35_attn_prefill_preprocess_q8_0_f32",
        @"qw35_attention_gqa_flash_decode_q8_0_f32",
        @"qw35_attention_gqa_prefill_q8_0_f32",
    };
    NSString *const *attn = _kvQ8 ? attn_q8 : attn_f16;
    for (size_t i = 0; i < 4; i++) {
        if (![self attnPipeline:attn[i] error:error]) return NO;
    }
    if (_gf4Store) {
        if (![_pipelineCache pipelineNamed:@"qw35_ffn_gate_up_swiglu_gf4_f32" error:error]) return NO;
        if (![_pipelineCache pipelineNamed:@"qw35_decode_matmul_gf4_2row_residual_f32" error:error]) return NO;
        if (![_pipelineCache pipelineNamed:@"qw35_decode_matmul_gf4_2row_f32" error:error]) return NO;
        if ([_gf4Store tensorNamed:@"output.weight"]) {
            if (![_pipelineCache pipelineNamed:@"qw35_output_gf4_argmax_partials_16row_f32" error:error]) return NO;
        }
    }
    return YES;
}

- (id<MTLComputePipelineState>)attnPipeline:(NSString *)name error:(NSError **)error {
    return [_pipelineCache attnPipelineNamed:name
                                       heads:(int)_h.attention_heads
                                     kvHeads:(int)_h.attention_kv_heads
                                     headDim:(int)_h.attention_key_length
                                     ropeDim:(int)_h.rope_dimension_count
                                       error:error];
}

- (BOOL)waitForLastCommand:(NSError **)error {
    id<MTLCommandBuffer> cb = _lastCB;
    _lastCB = nil;
    if (!cb) return YES;
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        if (error) *error = cb.error ?: qw35_error(@"Metal command buffer failed");
        return NO;
    }
    return YES;
}

- (BOOL)sync:(NSError **)error {
    return [self waitForLastCommand:error];
}

- (BOOL)reset:(NSError **)error {
    if (![self waitForLastCommand:error]) return NO;
    NSArray<id<MTLBuffer>> *zeroed = @[_conv_state, _ssm_state];
    for (id<MTLBuffer> buffer in zeroed) {
        if (!buffer) continue;
        memset([buffer contents], 0, [buffer length]);
    }
    return YES;
}

- (BOOL)stateCheckpointSave:(NSError **)error {
    if (![self waitForLastCommand:error]) return NO;
    if (!_conv_state || !_ssm_state) {
        if (error) *error = qw35_error(@"recurrent state buffers are not allocated");
        return NO;
    }
    if (!_conv_state_ckpt) _conv_state_ckpt = [NSMutableData dataWithLength:[_conv_state length]];
    if (!_ssm_state_ckpt) _ssm_state_ckpt = [NSMutableData dataWithLength:[_ssm_state length]];
    if (!_conv_state_ckpt || !_ssm_state_ckpt) {
        if (error) *error = qw35_error(@"failed to allocate state checkpoint storage");
        return NO;
    }
    memcpy([_conv_state_ckpt mutableBytes], [_conv_state contents], [_conv_state length]);
    memcpy([_ssm_state_ckpt mutableBytes], [_ssm_state contents], [_ssm_state length]);
    _has_state_ckpt = YES;
    return YES;
}

- (BOOL)stateCheckpointRestore:(NSError **)error {
    if (![self waitForLastCommand:error]) return NO;
    if (!_has_state_ckpt) {
        if (error) *error = qw35_error(@"no recurrent state checkpoint has been saved");
        return NO;
    }
    memcpy([_conv_state contents], [_conv_state_ckpt bytes], [_conv_state length]);
    memcpy([_ssm_state contents], [_ssm_state_ckpt bytes], [_ssm_state length]);
    return YES;
}

- (BOOL)initializeRopeFrequencies:(NSError **)error {
    const uint32_t half_dim = _h.rope_dimension_count / 2u;
    if (half_dim == 0 || !_rope_freq) {
        if (error) *error = qw35_error(@"invalid Qw35 RoPE dimensions");
        return NO;
    }

    // MRoPE (ggml rope_multi semantics): the frequency ladder is one global
    // geometric progression over ALL rotary dims — freq[i] = base^(-2i/n_dims)
    // — and rope_sections only select WHICH position component (t/h/w/e)
    // feeds theta for each dim. Text-only inference has every component equal
    // to the token position, so the sections reduce to plain NeoX RoPE over
    // the full ladder. Restarting the ladder per section (the previous code)
    // re-emitted the fastest frequencies in every section and dropped most of
    // the slow, long-range ones: positions became indistinguishable at range,
    // which surfaced as repetition loops, counting drift, and indentation
    // slips. The model was trained against the global ladder.
    float *freq = (float *)[_rope_freq contents];
    for (uint32_t pair = 0; pair < half_dim; pair++) {
        freq[pair] = powf(_h.rope_freq_base,
                          -((float)(pair * 2u)) / (float)_h.rope_dimension_count);
    }
    return YES;
}

- (Qw35Tensor *)tensorNamed:(NSString *)name error:(NSError **)error {
    Qw35Tensor *tensor = [_tensorStore tensorNamed:name];
    if (!tensor && error) *error = qw35_error(@"missing tensor %@", name);
    return tensor;
}

// ---------------------------------------------------------------------------
#pragma mark - Single-token decode
// ---------------------------------------------------------------------------

- (BOOL)evalToken:(uint32_t)token
              pos:(uint32_t)pos
       logitsMode:(qw35_logits_mode)logitsMode
            error:(NSError **)error {
    if (pos >= _ctxSize) {
        if (error) *error = qw35_error(@"requested position exceeds allocated context");
        return NO;
    }
    if (![self ensureKvCapacityForPositions:(uint64_t)pos + 1 error:error]) return NO;
    if (token >= _vocabSize) {
        if (error) *error = qw35_error(@"token id is outside the Qw35 vocabulary");
        return NO;
    }

    const uint32_t mid = _h.transformer_layers / 2;
    const uint32_t interval = _h.full_attention_interval ? _h.full_attention_interval : 4;

    // Delta/attention slot counts consumed by the first command buffer.
    uint32_t cb0_delta = 0;
    uint32_t cb0_attn = 0;
    for (uint32_t il = 0; il < mid; il++) {
        if (((il + 1) % interval) == 0) cb0_attn++;
        else cb0_delta++;
    }

    id<MTLCommandBuffer> cb0 = [_queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc0 = [cb0 computeCommandEncoder];
    BOOL ok = [self encodeDecodeLayers:enc0
                                 token:token
                                   pos:pos
                            layerBegin:0
                              layerEnd:mid
                             deltaSlot:0
                              attnSlot:0
                            logitsMode:logitsMode
                                 error:error];
    [enc0 endEncoding];
    if (!ok) return NO;
    [cb0 commit];

    id<MTLCommandBuffer> cb1 = [_queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc1 = [cb1 computeCommandEncoder];
    ok = [self encodeDecodeLayers:enc1
                            token:token
                              pos:pos
                       layerBegin:mid
                         layerEnd:_h.transformer_layers
                        deltaSlot:cb0_delta
                         attnSlot:cb0_attn
                       logitsMode:logitsMode
                            error:error];
    if (ok && logitsMode != QW35_LOGITS_NONE) {
        ok = [self encodeOutputHead:enc1 normRowOffset:0 logitsMode:logitsMode error:error];
    }
    [enc1 endEncoding];
    if (!ok) return NO;
    [cb1 commit];
    _lastCB = cb1;
    return YES;
}

/// Encode layers [begin, end). On entry for begin == 0 this also encodes the
/// embedding lookup and the first attn_norm; for begin > 0 it assumes the
/// previous range left the layer-input norm in `_norm` (norm chaining).
/// Each layer leaves the NEXT layer's input norm in `_norm`; the final layer
/// leaves the output_norm result when logitsMode requires it.
- (BOOL)encodeDecodeLayers:(id<MTLComputeCommandEncoder>)enc
                     token:(uint32_t)token
                       pos:(uint32_t)pos
                layerBegin:(uint32_t)begin
                  layerEnd:(uint32_t)end
                 deltaSlot:(uint32_t)deltaSlot
                  attnSlot:(uint32_t)attnSlot
                logitsMode:(qw35_logits_mode)logitsMode
                     error:(NSError **)error {
    const uint32_t interval = _h.full_attention_interval ? _h.full_attention_interval : 4;

    if (begin == 0) {
        if (![self encodeEmbedding:enc token:token error:error]) return NO;
        if (![self encodeRms:enc src:_act_a weightTensor:_layers[0].attnNorm error:error]) return NO;
    }

    for (uint32_t il = begin; il < end; il++) {
        Qw35LayerTensors *layer = _layers[il];

        if (((il + 1) % interval) == 0) {
            if (![self encodeAttentionDecodeLayer:enc layer:layer slot:attnSlot pos:pos error:error]) return NO;
            attnSlot++;
        } else {
            if (![self encodeDeltaDecodeLayer:enc layer:layer slot:deltaSlot error:error]) return NO;
            deltaSlot++;
        }

        // act_a += act_b, then norm with post_attention_norm.
        if (![self encodeResidualRms:enc weightTensor:layer.postAttentionNorm error:error]) return NO;

        if (layer.ffnGate.type_id == 100 && layer.ffnUp.type_id == 100) {
            // GF4 FFN: fused gate+up+SwiGLU, then down with the residual add
            // folded in — two dispatches instead of four.
            if (![self encodeGf4FusedFfn:enc layer:layer error:error]) return NO;
        } else {
            // FFN: split gate/up matvecs, SwiGLU, down projection with the
            // residual add folded in.
            if (![self encodeDecodeMatvecTensor:enc weight:layer.ffnGate input:_norm inputOffset:0 dst:_ffn_gate residual:NO error:error]) return NO;
            if (![self encodeDecodeMatvecTensor:enc weight:layer.ffnUp input:_norm inputOffset:0 dst:_ffn_up residual:NO error:error]) return NO;
            if (![self encodeSwiGLU:enc n:_h.feed_forward_length error:error]) return NO;
            if (![self encodeDecodeMatvecTensor:enc weight:layer.ffnDown input:_ffn_gate inputOffset:0 dst:_act_a residual:YES error:error]) return NO;
        }

        // Chain the next layer's input norm (or the output norm).
        if (il + 1 < _h.transformer_layers) {
            if (![self encodeRms:enc src:_act_a weightTensor:_layers[il + 1].attnNorm error:error]) return NO;
        } else if (logitsMode != QW35_LOGITS_NONE) {
            if (![self encodeRms:enc src:_act_a weightTensor:_outputNorm error:error]) return NO;
        }
    }
    return YES;
}

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
    id<MTLComputePipelineState> pipe = [_pipelineCache pipelineNamed:@"qw35_ssm_conv_recurrent_gate_norm_step128_f32" error:error];
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

/// GF4 decode FFN: fused gate+up+SwiGLU into _ffn_gate, then down projection
/// accumulating into _act_a.
- (BOOL)encodeGf4FusedFfn:(id<MTLComputeCommandEncoder>)enc
                    layer:(Qw35LayerTensors *)layer
                    error:(NSError **)error {
    const int64_t emb = (int64_t)_h.embedding_length;
    const int64_t ffn = (int64_t)_h.feed_forward_length;

    id<MTLComputePipelineState> gateUp = [_pipelineCache pipelineNamed:@"qw35_ffn_gate_up_swiglu_gf4_f32" error:error];
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

- (BOOL)encodeAttentionDecodeLayer:(id<MTLComputeCommandEncoder>)enc
                             layer:(Qw35LayerTensors *)layer
                              slot:(uint32_t)slot
                               pos:(uint32_t)pos
                             error:(NSError **)error {
    const NSUInteger kv_off = (NSUInteger)((uint64_t)slot * _kvCapacity * _kvRowBytes);
    const float scale = 1.0f / sqrtf((float)_h.attention_key_length);

    if (![self encodeDecodeMatvecTensor:enc weight:layer.attnQ input:_norm inputOffset:0 dst:_qkv residual:NO error:error]) return NO;
    if (![self encodeDecodeMatvecTensor:enc weight:layer.attnK input:_norm inputOffset:0 dst:_ffn_gate residual:NO error:error]) return NO;
    if (![self encodeDecodeMatvecTensor:enc weight:layer.attnV input:_norm inputOffset:0 dst:_ffn_up residual:NO error:error]) return NO;

    Qw35Tensor *q_norm = layer.attnQNorm;
    Qw35Tensor *k_norm = layer.attnKNorm;
    id<MTLComputePipelineState> prep =
        [self attnPipeline:_kvQ8 ? @"qw35_attn_decode_preprocess_q8_0_f32"
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
    [enc setBuffer:_k_cache offset:kv_off atIndex:7];
    [enc setBuffer:_v_cache offset:kv_off atIndex:8];
    [enc setBytes:&n_head length:sizeof(n_head) atIndex:9];
    [enc setBytes:&n_kv_head length:sizeof(n_kv_head) atIndex:10];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:11];
    [enc setBytes:&pos64 length:sizeof(pos64) atIndex:12];
    [enc setBytes:&eps length:sizeof(eps) atIndex:13];
    [enc setBuffer:_rope_freq offset:0 atIndex:14];
    [enc setBytes:&rope_dim length:sizeof(rope_dim) atIndex:15];
    qw35_dispatch_1d(enc, (NSUInteger)(_h.attention_heads * _h.attention_key_length), 256);

    // Barrier-free flash decode: one simdgroup per head, sigmoid output gate
    // applied in-kernel.
    id<MTLComputePipelineState> attn =
        [self attnPipeline:_kvQ8 ? @"qw35_attention_gqa_flash_decode_q8_0_f32"
                                 : @"qw35_attention_gqa_flash_decode_f32"
                     error:error];
    if (!attn) return NO;
    int64_t seq_len = (int64_t)pos + 1;
    [enc setComputePipelineState:attn];
    [enc setBuffer:_q_rep offset:0 atIndex:0];
    [enc setBuffer:_k_cache offset:kv_off atIndex:1];
    [enc setBuffer:_v_cache offset:kv_off atIndex:2];
    [enc setBuffer:_core offset:0 atIndex:3];
    [enc setBuffer:_z_gate offset:0 atIndex:4];
    [enc setBytes:&n_head length:sizeof(n_head) atIndex:5];
    [enc setBytes:&n_kv_head length:sizeof(n_kv_head) atIndex:6];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
    [enc setBytes:&seq_len length:sizeof(seq_len) atIndex:8];
    [enc setBytes:&scale length:sizeof(scale) atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)_h.attention_heads, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    if (![self encodeDecodeMatvecTensor:enc weight:layer.attnOutput input:_core inputOffset:0 dst:_act_b residual:NO error:error]) return NO;
    return YES;
}

- (BOOL)encodeOutputHead:(id<MTLComputeCommandEncoder>)enc
           normRowOffset:(NSUInteger)normRowOffset
              logitsMode:(qw35_logits_mode)logitsMode
                   error:(NSError **)error {
    if (logitsMode == QW35_LOGITS_FULL) {
        return [self encodeDecodeMatvecTensor:enc weight:_outputWeight input:_norm inputOffset:normRowOffset dst:_logits residual:NO error:error];
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

    id<MTLComputePipelineState> partials = [_pipelineCache pipelineNamed:partialsKernel error:error];
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

    id<MTLComputePipelineState> reduce = [_pipelineCache pipelineNamed:@"qw35_output_argmax_reduce_partials_f32" error:error];
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
#pragma mark - Chunked prefill
// ---------------------------------------------------------------------------

- (BOOL)evalTokens:(const uint32_t *)tokens
             count:(uintptr_t)count
              pos0:(uint32_t)pos0
        logitsMode:(qw35_logits_mode)logitsMode
             error:(NSError **)error {
    if (!tokens || count == 0) return YES;
    if (count == 1) return [self evalToken:tokens[0] pos:pos0 logitsMode:logitsMode error:error];
    if (count > _maxPrefillChunk) {
        if (error) *error = qw35_error(@"requested prefill chunk exceeds configured --prefill-chunk");
        return NO;
    }
    if ((uint64_t)pos0 + (uint64_t)count > _ctxSize) {
        if (error) *error = qw35_error(@"requested prefill chunk exceeds allocated context");
        return NO;
    }
    if (![self ensureKvCapacityForPositions:(uint64_t)pos0 + (uint64_t)count error:error]) return NO;
    for (uintptr_t i = 0; i < count; i++) {
        if (tokens[i] >= _vocabSize) {
            if (error) *error = qw35_error(@"token id is outside the Qw35 vocabulary");
            return NO;
        }
    }

    // The previous eval may still read _prefill_tokens; wait before rewriting.
    if (![self waitForLastCommand:error]) return NO;
    memcpy([_prefill_tokens contents], tokens, (size_t)count * sizeof(uint32_t));

    const uint32_t n = (uint32_t)count;
    const uint32_t interval = _h.full_attention_interval ? _h.full_attention_interval : 4;

    id<MTLCommandBuffer> cb = [_queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    BOOL ok = NO;

    do {
        if (![self encodeEmbeddingBatch:enc count:n error:error]) break;
        if (![self encodeRmsBatch:enc weight:@"blk.0.attn_norm.weight" tokens:n error:error]) break;

        uint32_t delta_slot = 0;
        uint32_t attn_slot = 0;
        BOOL failed = NO;
        for (uint32_t il = 0; il < _h.transformer_layers && !failed; il++) {
            NSString *prefix = [NSString stringWithFormat:@"blk.%u.", il];

            if (((il + 1) % interval) == 0) {
                if (![self encodeAttentionPrefillLayer:enc slot:attn_slot pos0:pos0 tokens:n prefix:prefix error:error]) { failed = YES; break; }
                attn_slot++;
            } else {
                if (![self encodeDeltaPrefillLayer:enc slot:delta_slot tokens:n prefix:prefix error:error]) { failed = YES; break; }
                delta_slot++;
            }

            if (![self encodeResidualRmsBatch:enc weight:[prefix stringByAppendingString:@"post_attention_norm.weight"] tokens:n error:error]) { failed = YES; break; }

            if (![self encodeMatvecBatch:enc weight:[prefix stringByAppendingString:@"ffn_gate.weight"] input:_norm inputOffset:0 dst:_ffn_gate dstOffset:0 tokens:n error:error]) { failed = YES; break; }
            if (![self encodeMatvecBatch:enc weight:[prefix stringByAppendingString:@"ffn_up.weight"] input:_norm inputOffset:0 dst:_ffn_up dstOffset:0 tokens:n error:error]) { failed = YES; break; }
            if (![self encodeSwiGLU:enc n:(uint64_t)n * _h.feed_forward_length error:error]) { failed = YES; break; }
            if (![self encodeMatvecBatch:enc weight:[prefix stringByAppendingString:@"ffn_down.weight"] input:_ffn_gate inputOffset:0 dst:_act_b dstOffset:0 tokens:n error:error]) { failed = YES; break; }

            if (il + 1 < _h.transformer_layers) {
                NSString *next = [NSString stringWithFormat:@"blk.%u.attn_norm.weight", il + 1];
                if (![self encodeResidualRmsBatch:enc weight:next tokens:n error:error]) { failed = YES; break; }
            } else if (logitsMode != QW35_LOGITS_NONE) {
                if (![self encodeResidualRmsBatch:enc weight:@"output_norm.weight" tokens:n error:error]) { failed = YES; break; }
            }
        }
        if (failed) break;

        if (logitsMode != QW35_LOGITS_NONE) {
            const NSUInteger last_off = (NSUInteger)((uint64_t)(n - 1) * _h.embedding_length * sizeof(float));
            if (![self encodeOutputHead:enc normRowOffset:last_off logitsMode:logitsMode error:error]) break;
        }
        ok = YES;
    } while (0);

    [enc endEncoding];
    if (!ok) return NO;
    [cb commit];
    _lastCB = cb;
    return YES;
}

- (BOOL)encodeDeltaPrefillLayer:(id<MTLComputeCommandEncoder>)enc
                           slot:(uint32_t)slot
                         tokens:(uint32_t)tokensCount
                         prefix:(NSString *)prefix
                          error:(NSError **)error {
    const int conv_channels = (int)(_h.ssm_inner_size + 2u * _h.ssm_group_count * _h.ssm_state_size);
    const uint64_t conv_slot_elems = (uint64_t)conv_channels * (_h.ssm_conv_kernel - 1);
    const uint64_t state_slot_elems = (uint64_t)_h.ssm_time_step_rank * _h.ssm_state_size * _h.ssm_state_size;
    const NSUInteger conv_off = (NSUInteger)(slot * conv_slot_elems * sizeof(float));
    const NSUInteger state_off = (NSUInteger)(slot * state_slot_elems * sizeof(float));
    const float scale = 1.0f / sqrtf((float)_h.ssm_state_size);

    if (![self encodeMatvecBatch:enc weight:[prefix stringByAppendingString:@"attn_qkv.weight"] input:_norm inputOffset:0 dst:_qkv dstOffset:0 tokens:tokensCount error:error]) return NO;
    if (![self encodeMatvecBatch:enc weight:[prefix stringByAppendingString:@"attn_gate.weight"] input:_norm inputOffset:0 dst:_z_gate dstOffset:0 tokens:tokensCount error:error]) return NO;
    if (![self encodeMatvecBatch:enc weight:[prefix stringByAppendingString:@"ssm_beta.weight"] input:_norm inputOffset:0 dst:_beta dstOffset:0 tokens:tokensCount error:error]) return NO;
    if (![self encodeMatvecBatch:enc weight:[prefix stringByAppendingString:@"ssm_alpha.weight"] input:_norm inputOffset:0 dst:_alpha dstOffset:0 tokens:tokensCount error:error]) return NO;

    Qw35Tensor *conv_w = [self tensorNamed:[prefix stringByAppendingString:@"ssm_conv1d.weight"] error:error];
    if (!conv_w) return NO;
    id<MTLComputePipelineState> conv_pipe = [_pipelineCache pipelineNamed:@"qw35_ssm_conv1d_step4_batch_f32" error:error];
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

    id<MTLComputePipelineState> prep_pipe = [_pipelineCache pipelineNamed:@"qw35_ssm_l2_repeat_qk_batch_f32" error:error];
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

    Qw35Tensor *dt = [self tensorNamed:[prefix stringByAppendingString:@"ssm_dt.bias"] error:error];
    Qw35Tensor *a = [self tensorNamed:[prefix stringByAppendingString:@"ssm_a"] error:error];
    if (!dt || !a) return NO;
    id<MTLComputePipelineState> rec_pipe = [_pipelineCache pipelineNamed:@"qw35_ssm_recurrent_step128_batch_rows_f32" error:error];
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

    Qw35Tensor *ssm_norm = [self tensorNamed:[prefix stringByAppendingString:@"ssm_norm.weight"] error:error];
    if (!ssm_norm) return NO;
    id<MTLComputePipelineState> gate_norm_pipe = [_pipelineCache pipelineNamed:@"qw35_ssm_gate_norm_batch_f32" error:error];
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

    if (![self encodeMatvecBatch:enc weight:[prefix stringByAppendingString:@"ssm_out.weight"] input:_z_gate inputOffset:0 dst:_act_b dstOffset:0 tokens:tokensCount error:error]) return NO;
    return YES;
}

- (BOOL)encodeAttentionPrefillLayer:(id<MTLComputeCommandEncoder>)enc
                               slot:(uint32_t)slot
                               pos0:(uint32_t)pos0
                             tokens:(uint32_t)tokensCount
                             prefix:(NSString *)prefix
                              error:(NSError **)error {
    const NSUInteger kv_off = (NSUInteger)((uint64_t)slot * _kvCapacity * _kvRowBytes);
    const float scale = 1.0f / sqrtf((float)_h.attention_key_length);

    if (![self encodeMatvecBatch:enc weight:[prefix stringByAppendingString:@"attn_q.weight"] input:_norm inputOffset:0 dst:_qkv dstOffset:0 tokens:tokensCount error:error]) return NO;
    if (![self encodeMatvecBatch:enc weight:[prefix stringByAppendingString:@"attn_k.weight"] input:_norm inputOffset:0 dst:_ffn_gate dstOffset:0 tokens:tokensCount error:error]) return NO;
    if (![self encodeMatvecBatch:enc weight:[prefix stringByAppendingString:@"attn_v.weight"] input:_norm inputOffset:0 dst:_ffn_up dstOffset:0 tokens:tokensCount error:error]) return NO;

    Qw35Tensor *q_norm = [self tensorNamed:[prefix stringByAppendingString:@"attn_q_norm.weight"] error:error];
    Qw35Tensor *k_norm = [self tensorNamed:[prefix stringByAppendingString:@"attn_k_norm.weight"] error:error];
    if (!q_norm || !k_norm) return NO;
    id<MTLComputePipelineState> prep =
        [self attnPipeline:_kvQ8 ? @"qw35_attn_prefill_preprocess_q8_0_f32"
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
    [enc setBuffer:_k_cache offset:kv_off atIndex:7];
    [enc setBuffer:_v_cache offset:kv_off atIndex:8];
    [enc setBytes:&n_head length:sizeof(n_head) atIndex:9];
    [enc setBytes:&n_kv_head length:sizeof(n_kv_head) atIndex:10];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:11];
    [enc setBytes:&pos064 length:sizeof(pos064) atIndex:12];
    [enc setBytes:&eps length:sizeof(eps) atIndex:13];
    [enc setBuffer:_rope_freq offset:0 atIndex:14];
    [enc setBytes:&rope_dim length:sizeof(rope_dim) atIndex:15];
    [enc setBytes:&n_tokens length:sizeof(n_tokens) atIndex:20];
    qw35_dispatch_1d(enc, (NSUInteger)((uint64_t)tokensCount * max_work), 256);

    id<MTLComputePipelineState> attn =
        [self attnPipeline:_kvQ8 ? @"qw35_attention_gqa_prefill_q8_0_f32"
                                 : @"qw35_attention_gqa_prefill_f32"
                     error:error];
    if (!attn) return NO;
    [enc setComputePipelineState:attn];
    [enc setBuffer:_q_rep offset:0 atIndex:0];
    [enc setBuffer:_k_cache offset:kv_off atIndex:1];
    [enc setBuffer:_v_cache offset:kv_off atIndex:2];
    [enc setBuffer:_core offset:0 atIndex:3];
    [enc setBuffer:_z_gate offset:0 atIndex:4];
    [enc setBytes:&n_head length:sizeof(n_head) atIndex:5];
    [enc setBytes:&n_kv_head length:sizeof(n_kv_head) atIndex:6];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
    [enc setBytes:&pos064 length:sizeof(pos064) atIndex:8];
    [enc setBytes:&scale length:sizeof(scale) atIndex:9];
    [enc setBytes:&n_tokens length:sizeof(n_tokens) atIndex:10];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tokensCount,
                                          (NSUInteger)_h.attention_heads,
                                          1)
         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    if (![self encodeMatvecBatch:enc weight:[prefix stringByAppendingString:@"attn_output.weight"] input:_core inputOffset:0 dst:_act_b dstOffset:0 tokens:tokensCount error:error]) return NO;
    return YES;
}

// ---------------------------------------------------------------------------
#pragma mark - Per-operation encoders
// ---------------------------------------------------------------------------

- (BOOL)encodeEmbedding:(id<MTLComputeCommandEncoder>)enc
                  token:(uint32_t)token
                  error:(NSError **)error {
    id<MTLComputePipelineState> pipe = [_pipelineCache pipelineNamed:@"qw35_get_row_q4_k_f32" error:error];
    if (!pipe) return NO;
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
    id<MTLComputePipelineState> pipe = [_pipelineCache pipelineNamed:@"qw35_get_rows_q4_k_f32" error:error];
    if (!pipe) return NO;
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
    id<MTLComputePipelineState> pipe = [_pipelineCache pipelineNamed:@"qw35_rms_norm_weight_f32" error:error];
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
    id<MTLComputePipelineState> pipe = [_pipelineCache pipelineNamed:@"qw35_residual_rms_norm_weight_f32" error:error];
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
                weight:(NSString *)weightName
                tokens:(uint32_t)tokensCount
                 error:(NSError **)error {
    Qw35Tensor *w = [self tensorNamed:weightName error:error];
    if (!w) return NO;
    id<MTLComputePipelineState> pipe = [_pipelineCache pipelineNamed:@"qw35_rms_norm_weight_batch_f32" error:error];
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
                        weight:(NSString *)weightName
                        tokens:(uint32_t)tokensCount
                         error:(NSError **)error {
    Qw35Tensor *w = [self tensorNamed:weightName error:error];
    if (!w) return NO;
    id<MTLComputePipelineState> pipe = [_pipelineCache pipelineNamed:@"qw35_residual_rms_norm_weight_batch_f32" error:error];
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
/// dst instead of overwriting it (Q4_K, Q6_K, and GF4 weights only).
/// GF4 sidecar tensors carry type_id 100.
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
            kernel = @"qw35_decode_matmul_q8_0_f32";
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
        default:
            if (error) *error = qw35_error(@"unsupported tensor type %u for %@", w.type_id, w.name);
            return NO;
    }
    if (residual && w.type_id != 12 && w.type_id != 14 && w.type_id != 100) {
        if (error) *error = qw35_error(@"no residual matvec kernel for tensor type %u (%@)", w.type_id, w.name);
        return NO;
    }

    id<MTLComputePipelineState> pipe = [_pipelineCache pipelineNamed:kernel error:error];
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
                   weight:(NSString *)weightName
                    input:(id<MTLBuffer>)input
              inputOffset:(NSUInteger)inputOffset
                      dst:(id<MTLBuffer>)dst
                dstOffset:(NSUInteger)dstOffset
                   tokens:(uint32_t)tokensCount
                    error:(NSError **)error {
    Qw35Tensor *w = [self tensorNamed:weightName error:error];
    if (!w) return NO;
    if (w.n_dims < 2) {
        if (error) *error = qw35_error(@"tensor %@ is not a matrix", weightName);
        return NO;
    }
    int64_t k = (int64_t)w.dims[0];
    int64_t rows = (int64_t)w.dims[1];

    if (tokensCount == 1) {
        Qw35Tensor *preferred = [self decodeWeightNamed:weightName error:error];
        if (!preferred) return NO;
        return [self encodeDecodeMatvecTensor:enc weight:preferred input:input inputOffset:inputOffset dst:dst residual:NO error:error];
    }

    NSString *tiledKernel = nil;
    switch (w.type_id) {
        case 8:  tiledKernel = @"qw35_mul_mm_q8_0_f32"; break;
        case 12: tiledKernel = @"qw35_mul_mm_q4_k_f32"; break;
        case 13: tiledKernel = @"qw35_mul_mm_q5_k_f32"; break;
        case 14: tiledKernel = @"qw35_mul_mm_q6_k_f32"; break;
        default:
            if (error) *error = qw35_error(@"unsupported tensor type %u for %@", w.type_id, weightName);
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
    id<MTLComputePipelineState> pipe = [_pipelineCache pipelineNamed:@"qw35_swiglu_f32" error:error];
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
    if (!dst || len < _vocabSize) {
        if (error) *error = qw35_error(@"invalid logits readback buffer");
        return NO;
    }
    if (![self waitForLastCommand:error]) return NO;
    memcpy(dst, [_logits contents], (size_t)_vocabSize * sizeof(float));
    return YES;
}

@end
