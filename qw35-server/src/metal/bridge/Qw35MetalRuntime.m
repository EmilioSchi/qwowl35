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

#import "Qw35MetalRuntime+Internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// KV cache slab size: the per-layer stride and the granularity by which the
// cache grows (one slab appended per crossing; see -ensureKvCapacityForPositions:).
// The cache is a list of fixed-size slab buffers rather than one buffer that
// reallocates on growth, so a crossing never copies, never doubles the transient
// footprint, and never re-pins the whole cache — it just wires one fresh slab.
// A short chat keeps only the slabs it touches resident (one slab ~= 142 MiB of
// q8_0 KV across the 8 attention layers), not the full --ctx worth.
//
// 8192 is chosen by measurement: decode tok/s is independent of slab size across
// 1024..32768 (weight-bandwidth bound; the per-layer stride does not move it), so
// the size is set by other concerns — 8192 holds a typical large first prompt in
// one slab (few prefill-time crossings) and yields a small slab count (<=32 even
// at the largest supported --ctx 262144), keeping the read-path slab array tiny.
#define QW35_KV_INITIAL_SLAB 8192u
// QW35_MAX_SLABS is defined in Qw35MetalRuntime+Internal.h (it sizes the ivar
// slab arrays) and must match QW35_MAX_SLABS in qw35_attention.metal.

@implementation Qw35LayerTensors
@end

@implementation Qw35MetalRuntime

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

    const char *capPath = getenv("QW35_CAPTURE_ACT_OUT");
    if (capPath && *capPath) {
        _capEnabled = YES;
        _capPath = strdup(capPath);
    }

    const char *stageProf = getenv("QW35_STAGE_PROFILE");
    if (stageProf && *stageProf) {
        _stageProfile = YES;
        fprintf(stderr, "qw35: per-stage GPU profiler ON (QW35_STAGE_PROFILE); decode is serialised and SLOW\n");
    }

    _attnWindow = hparams->attn_window;
    _attnSink = hparams->attn_sink;

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
    const uint32_t slab = QW35_KV_INITIAL_SLAB;
    // The segmented KV cache addresses at most QW35_MAX_SLABS slabs. Reject a
    // larger --ctx outright rather than clamping silently: the Rust layer admits
    // prompts against its own ctx, so a cache that quietly held fewer positions
    // would let a request reach a slab index past the array (out of bounds).
    // QW35_MAX_SLABS * slab = 64 * 8192 = 524288, double the model's 262144
    // trained window, so this only ever rejects an unsupported oversize ctx.
    const uint64_t maxCtx = (uint64_t)QW35_MAX_SLABS * slab;
    if ((uint64_t)ctxSize > maxCtx) {
        if (error) *error = qw35_error(@"--ctx %u exceeds the maximum %llu tokens the KV cache can address (%u slabs x %u)",
                                       ctxSize, maxCtx, (unsigned)QW35_MAX_SLABS, slab);
        return nil;
    }
    _ctxSize = ctxSize;
    // Start with a single slab and grow on demand; never reserve past the ceiling.
    _kvSlab = slab;
    _kvCapacity = _ctxSize < slab ? _ctxSize : slab;
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

    _pipelineCache = [[Qw35PipelineCache alloc] initWithLibrary:_library device:_device];

    if (![self resolveLayerTensors:error]) return nil;
    if (![self allocateBuffers:error]) return nil;
    if (![self prewarmPipelines:error]) return nil;
    if (![self reset:error]) return nil;
    [self setupResidency];
    return self;
}

/// Pin the mmap-backed weight buffers, the KV cache, and the large scratch
/// buffers into an MTLResidencySet (macOS 15+) and request residency ONCE, but
/// deliberately do NOT associate the set with the command queue.
///
/// Why not associate it: a queue-associated residency set keeps its allocations
/// continuously resident for every command buffer, which holds the unified-
/// memory subsystem in a high-power state and heats the GPU into thermal
/// throttle ~1 min sooner under sustained decode — measured 14->7 tok/s within
/// 2 minutes on an M2/16 GiB. A one-shot requestResidency biases the pages into
/// fast memory without that continuous-residency power cost; sustained decode
/// then holds ~14 tok/s. (A/B confirmed: dropping the queue association — or the
/// request, or the weight pinning — each eliminates the droop; the full
/// combination is the only one that throttles.) macOS 15+ only.
- (void)setupResidency {
    // Runtime feature-detect instead of @available (which needs a runtime symbol
    // not linked under this build's -nodefaultlibs). MTLResidencySet is macOS 15+.
    if (![_device respondsToSelector:@selector(newResidencySetWithDescriptor:error:)]) {
        fprintf(stderr, "qw35: MTLResidencySet unavailable on this OS; residency skipped\n");
        return;
    }
    MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
    NSError *err = nil;
    id<MTLResidencySet> rs = [_device newResidencySetWithDescriptor:desc error:&err];
    if (!rs) {
        fprintf(stderr, "qw35: residency set creation failed: %s\n",
                err.localizedDescription.UTF8String);
        return;
    }
    NSMutableArray<id<MTLBuffer>> *bufs = [NSMutableArray array];
    [bufs addObjectsFromArray:[_tensorStore allBuffers]];
    id<MTLBuffer> scratch[] = {
        _conv_state, _ssm_state, _act_a, _act_b, _norm,
        _qkv, _z_gate, _q_rep, _k_rep, _core, _ffn_gate, _ffn_up, _logits, _rope_freq,
        _beta, _alpha, _argmax_partial_token, _argmax_partial_logit,
    };
    for (size_t i = 0; i < sizeof(scratch) / sizeof(scratch[0]); i++) {
        if (scratch[i]) [bufs addObject:scratch[i]];
    }
    // Segmented KV: pin every live slab buffer (more are added on growth, see
    // -ensureKvCapacityForPositions:). The ptr arg buffers are small but pinned
    // too so the bindless read can fetch slab addresses.
    for (uint32_t i = 0; i < _kvSlabCount; i++) {
        if (_kSlabs[i]) [bufs addObject:_kSlabs[i]];
        if (_vSlabs[i]) [bufs addObject:_vSlabs[i]];
    }
    if (_kSlabPtrs) [bufs addObject:_kSlabPtrs];
    if (_vSlabPtrs) [bufs addObject:_vSlabPtrs];
    uint64_t total = 0;
    for (id<MTLBuffer> b in bufs) {
        [rs addAllocation:b];
        total += b.allocatedSize;
    }
    [rs commit];
    [rs requestResidency];  // one-shot; NOT [_queue addResidencySet:rs] (see above)
    _residencySet = rs;
    // No stderr announcement: residency is hardcoded, not a configuration
    // parameter. The startup summary's `residency=on` line covers user-facing
    // state; the rare unavailable/failed cases above still warn.
}

/// A decode matvec weight from the unified .gguf tensor table. The tensor's
/// type-id (100 = GF4) selects the GF4 vs Q4_K kernel downstream.
- (Qw35Tensor *)decodeWeightNamed:(NSString *)name error:(NSError **)error {
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
        // The unified .gguf bakes the AWQ-folded norm directly into
        // post_attention_norm, so prefill and decode share it (both run the
        // GF4+AWQ FFN) — no decode-only override.
        layer.postAttentionNormDecode = layer.postAttentionNorm;
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
    // Segmented KV cache: allocate only the first slab. Each slab buffer holds
    // all attention layers for one slab-window of positions, with a fixed
    // per-layer stride of _kvSlab. -ensureKvCapacityForPositions: appends more
    // slabs (no restride/copy) as the live context advances.
    const uint64_t slab_bytes = (uint64_t)_attentionLayerCount * (uint64_t)_kvSlab * _kvRowBytes;
    _kSlabs[0] = [_device newBufferWithLength:(NSUInteger)slab_bytes
                                      options:MTLResourceStorageModeShared];
    _kSlabs[0].label = @"k_cache";
    _vSlabs[0] = [_device newBufferWithLength:(NSUInteger)slab_bytes
                                      options:MTLResourceStorageModeShared];
    _vSlabs[0].label = @"v_cache";
    _kvSlabCount = 1;
    _kSlabPtrs = [_device newBufferWithLength:(NSUInteger)(QW35_MAX_SLABS * sizeof(uint64_t))
                                      options:MTLResourceStorageModeShared];
    _kSlabPtrs.label = @"k_slab_ptrs";
    _vSlabPtrs = [_device newBufferWithLength:(NSUInteger)(QW35_MAX_SLABS * sizeof(uint64_t))
                                      options:MTLResourceStorageModeShared];
    _vSlabPtrs.label = @"v_slab_ptrs";
    if (_kSlabs[0] && _vSlabs[0] && _kSlabPtrs && _vSlabPtrs) [self updateSlabPtrs];
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
        !_k_rep || !_core || !_ffn_gate || !_ffn_up || !_logits || !_kSlabs[0] || !_vSlabs[0] ||
        !_kSlabPtrs || !_vSlabPtrs ||
        !_conv_state || !_ssm_state || !_argmax_token || !_argmax_logit ||
        !_argmax_partial_token || !_argmax_partial_logit || !_prefill_tokens || !_rope_freq) {
        if (error) *error = qw35_error(@"failed to allocate Qw35 Metal decode buffers");
        return NO;
    }
    if (![self initializeRopeFrequencies:error]) return NO;
    return YES;
}

// Write each live slab's gpuAddress into the bindless pointer arg buffers, so
// the read kernels (which take the slabs as an argument-buffer struct) can fetch
// slab base pointers by index. Call after every slab append.
- (void)updateSlabPtrs {
    uint64_t *kp = (uint64_t *)[_kSlabPtrs contents];
    uint64_t *vp = (uint64_t *)[_vSlabPtrs contents];
    for (uint32_t i = 0; i < _kvSlabCount; i++) {
        kp[i] = _kSlabs[i].gpuAddress;
        vp[i] = _vSlabs[i].gpuAddress;
    }
}

// The attention read kernels reach the slab buffers via raw gpuAddress through a
// bindless argument-buffer struct, which Metal does NOT hazard-track or keep
// resident automatically (unlike a setBuffer-bound resource). Declare every live
// slab as used-for-read on the encoder so the driver (1) keeps the slabs
// resident for the bindless fetch and (2) tracks the preceding current-token
// write (a setBuffer-bound access to the same slab) -> read dependency. Without
// this, slabs appended after setup (index >= 2) are read before they are
// reliably resident/ordered, perturbing attention from the 2nd crossing on.
- (void)useKvSlabsForRead:(id<MTLComputeCommandEncoder>)enc {
    for (uint32_t i = 0; i < _kvSlabCount; i++) {
        if (_kSlabs[i]) [enc useResource:_kSlabs[i] usage:MTLResourceUsageRead];
        if (_vSlabs[i]) [enc useResource:_vSlabs[i] usage:MTLResourceUsageRead];
    }
}

// Ensure the KV cache can address `positions` positions, growing it if needed.
// The cache is a list of fixed-stride slab buffers, so growth is append-only:
// each crossing allocates one fresh slab and pins it — no memcpy, no re-layout,
// no transient old+new doubling, and the residency set only ADDS the new slab
// (never re-pins the whole cache). That keeps a crossing to a fraction of a
// millisecond and, crucially, avoids the transient footprint spike that would
// otherwise evict the weight-bandwidth-bound model's pages on unified memory.
- (BOOL)ensureKvCapacityForPositions:(uint64_t)positions error:(NSError **)error {
    if (positions <= (uint64_t)_kvCapacity) return YES;

    const uint64_t slab = _kvSlab ? (uint64_t)_kvSlab : 1;
    uint64_t target = positions;
    if (target > (uint64_t)_ctxSize) target = _ctxSize;
    if (target <= (uint64_t)_kvCapacity) return YES;

    // Drain in-flight work before appending: a still-in-flight command buffer
    // may be reading the residency set / pointer table for the previous token.
    // Draining once here (growth is rare — once per slab crossed) establishes the
    // new slab and its gpuAddress entry before any command buffer references it,
    // complementing the per-encoder -useKvSlabsForRead: hazard tracking.
    if (![self waitForLastCommand:error]) return NO;

    const uint64_t slabBytes = (uint64_t)_attentionLayerCount * slab * _kvRowBytes;
    while ((uint64_t)_kvCapacity < target &&
           (uint64_t)_kvCapacity < (uint64_t)_ctxSize &&
           _kvSlabCount < QW35_MAX_SLABS) {
        id<MTLBuffer> nk = [_device newBufferWithLength:(NSUInteger)slabBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> nv = [_device newBufferWithLength:(NSUInteger)slabBytes options:MTLResourceStorageModeShared];
        if (!nk || !nv) {
            if (error) *error = qw35_error(@"failed to grow Qw35 KV cache (slab %u)", _kvSlabCount);
            return NO;
        }
        nk.label = @"k_cache";
        nv.label = @"v_cache";
        _kSlabs[_kvSlabCount] = nk;
        _vSlabs[_kvSlabCount] = nv;
        _kvSlabCount++;
        uint64_t cap = (uint64_t)_kvSlabCount * slab;
        if (cap > (uint64_t)_ctxSize) cap = _ctxSize;
        _kvCapacity = (uint32_t)cap;
        if (_residencySet) {
            [_residencySet addAllocation:nk];
            [_residencySet addAllocation:nv];
            [_residencySet commit];
            [_residencySet requestResidency];
        }
    }
    [self updateSlabPtrs];
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
    // Baked FFN codecs (GF4 type-100, GF2 type-101) are per-layer in a mixed
    // unified .gguf, so scan every layer's gate/up/down instead of sampling
    // blk.0 only; prewarm each codec's kernels once if any layer uses it.
    BOOL gf4Ffn = NO;
    BOOL gf2Ffn = NO;
    for (Qw35LayerTensors *layer in _layers) {
        Qw35Tensor *const parts[] = { layer.ffnGate, layer.ffnUp, layer.ffnDown };
        for (size_t i = 0; i < 3; i++) {
            if (parts[i] == nil) continue;
            if (parts[i].type_id == 100) gf4Ffn = YES;
            if (parts[i].type_id == 101) gf2Ffn = YES;
        }
    }
    if (gf4Ffn) {
        if (![_pipelineCache pipelineNamed:@"qw35_ffn_gate_up_swiglu_gf4_f32" error:error]) return NO;
        if (![_pipelineCache pipelineNamed:@"qw35_decode_matmul_gf4_2row_residual_f32" error:error]) return NO;
        if (![_pipelineCache pipelineNamed:@"qw35_decode_matmul_gf4_2row_f32" error:error]) return NO;
        // Tiled GF4 prefill matmul (full and bounded-output tail-chunk variants).
        if (![_pipelineCache mulMmPipelineNamed:@"qw35_mul_mm_gf4_f32" bcInp:NO bcOut:NO error:error]) return NO;
        if (![_pipelineCache mulMmPipelineNamed:@"qw35_mul_mm_gf4_f32" bcInp:NO bcOut:YES error:error]) return NO;
    }
    if (gf2Ffn) {
        if (![_pipelineCache pipelineNamed:@"qw35_decode_matmul_gf2_2row_residual_f32" error:error]) return NO;
        if (![_pipelineCache pipelineNamed:@"qw35_decode_matmul_gf2_2row_f32" error:error]) return NO;
        if (![_pipelineCache mulMmPipelineNamed:@"qw35_mul_mm_gf2_f32" bcInp:NO bcOut:NO error:error]) return NO;
        if (![_pipelineCache mulMmPipelineNamed:@"qw35_mul_mm_gf2_f32" bcInp:NO bcOut:YES error:error]) return NO;
    }
    Qw35Tensor *baseHead = [_tensorStore tensorNamed:@"output.weight"];
    if (baseHead != nil && baseHead.type_id == 100) {
        if (![_pipelineCache pipelineNamed:@"qw35_output_gf4_argmax_partials_16row_f32" error:error]) return NO;
    }
    return YES;
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

- (void)setAttnSink:(int)sink {
    _attnSink = sink < 0 ? 0 : sink;
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
    if (_capEnabled) {
        return [self captureEvalToken:token pos:pos logitsMode:logitsMode error:error];
    }
    if (_stageProfile) {
        return [self evalTokenStageProfiled:token pos:pos logitsMode:logitsMode error:error];
    }
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

- (void)dealloc {
    if (_capEnabled) {
        [self writeCaptureFile];
    }
    free(_capPath);
    free(_capGateUp);
    free(_capDown);
}

// ---------------------------------------------------------------------------
#pragma mark - Calibration activation capture (QW35_CAPTURE_ACT_OUT)
// ---------------------------------------------------------------------------

/// Encode one stage into a fresh command buffer and wait, so the just-written
/// scratch buffer can be read back on the CPU before the next stage clobbers it.
- (BOOL)captureRun:(BOOL (^)(id<MTLComputeCommandEncoder> enc, NSError **error))block
             error:(NSError **)error {
    id<MTLCommandBuffer> cb = [_queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    BOOL ok = block(enc, error);
    [enc endEncoding];
    if (!ok) return NO;
    [cb commit];
    [cb waitUntilCompleted];
    _lastCB = cb;
    return YES;
}

/// Write the cumulative per-channel mean-abs activation stats to _capPath.
/// Format: "QW35ACT\0", version u32, layers u32, gateup_dim u32, down_dim u32,
/// tokens u64, then [layers*gateup_dim] f32 gate/up means, then
/// [layers*down_dim] f32 down means.
- (void)writeCaptureFile {
    if (!_capPath || !_capGateUp || !_capDown || _capTokens == 0) return;
    FILE *f = fopen(_capPath, "wb");
    if (!f) return;

    const uint32_t layers = _h.transformer_layers;
    const uint32_t gu_dim = (uint32_t)_h.embedding_length;
    const uint32_t dn_dim = (uint32_t)_h.feed_forward_length;
    const char magic[8] = {'Q', 'W', '3', '5', 'A', 'C', 'T', '\0'};
    const uint32_t version = 1;
    fwrite(magic, 1, 8, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&layers, sizeof(layers), 1, f);
    fwrite(&gu_dim, sizeof(gu_dim), 1, f);
    fwrite(&dn_dim, sizeof(dn_dim), 1, f);
    fwrite(&_capTokens, sizeof(_capTokens), 1, f);

    const double inv = 1.0 / (double)_capTokens;
    const size_t gu_n = (size_t)layers * gu_dim;
    const size_t dn_n = (size_t)layers * dn_dim;
    float *buf = (float *)malloc((gu_n > dn_n ? gu_n : dn_n) * sizeof(float));
    if (buf) {
        for (size_t i = 0; i < gu_n; i++) buf[i] = (float)(_capGateUp[i] * inv);
        fwrite(buf, sizeof(float), gu_n, f);
        for (size_t i = 0; i < dn_n; i++) buf[i] = (float)(_capDown[i] * inv);
        fwrite(buf, sizeof(float), dn_n, f);
        free(buf);
    }
    fclose(f);
}

/// Calibration twin of -evalToken:. Runs a serialised per-layer decode and
/// accumulates the FFN gate/up input (`_norm`) and down input (`_ffn_gate`
/// after SwiGLU) per channel. Uses the split FFN path so the post-SwiGLU
/// activation is materialised in `_ffn_gate`. Slow (one CB per stage, with
/// readback) — calibration only. Enabled by QW35_CAPTURE_ACT_OUT.
- (BOOL)captureEvalToken:(uint32_t)token
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

    const uint32_t layers = _h.transformer_layers;
    const uint64_t emb = _h.embedding_length;
    const uint64_t ffn = _h.feed_forward_length;
    if (!_capGateUp) {
        _capGateUp = (double *)calloc((size_t)layers * emb, sizeof(double));
        _capDown = (double *)calloc((size_t)layers * ffn, sizeof(double));
        if (!_capGateUp || !_capDown) {
            if (error) *error = qw35_error(@"capture accumulator allocation failed");
            return NO;
        }
    }
    const uint32_t interval = _h.full_attention_interval ? _h.full_attention_interval : 4;

    uint32_t deltaSlot = 0, attnSlot = 0;
    for (uint32_t il = 0; il < layers; il++) {
        Qw35LayerTensors *layer = _layers[il];
        const BOOL isAttn = (((il + 1) % interval) == 0);
        const uint32_t aSlot = attnSlot;
        const uint32_t dSlot = deltaSlot;
        const BOOL first = (il == 0);
        const uint32_t ilc = il;

        // Mixer (+ embedding/first-norm on layer 0) then residual + post-norm:
        // leaves the FFN gate/up input in `_norm`.
        if (![self captureRun:^BOOL(id<MTLComputeCommandEncoder> enc, NSError **e) {
                if (first) {
                    if (![self encodeEmbedding:enc token:token error:e]) return NO;
                    if (![self encodeRms:enc src:_act_a weightTensor:_layers[0].attnNorm error:e]) return NO;
                }
                if (isAttn) {
                    if (![self encodeAttentionDecodeLayer:enc layer:layer slot:aSlot pos:pos error:e]) return NO;
                } else {
                    if (![self encodeDeltaDecodeLayer:enc layer:layer slot:dSlot error:e]) return NO;
                }
                return [self encodeResidualRms:enc weightTensor:layer.postAttentionNorm error:e];
            } error:error]) return NO;
        qw35_accum_absmean(_capGateUp + (size_t)il * emb, (const float *)[_norm contents], emb);
        if (isAttn) attnSlot++; else deltaSlot++;

        // Split FFN gate/up + SwiGLU: leaves the down input in `_ffn_gate`.
        if (![self captureRun:^BOOL(id<MTLComputeCommandEncoder> enc, NSError **e) {
                if (![self encodeDecodeMatvecTensor:enc weight:layer.ffnGate input:_norm inputOffset:0 dst:_ffn_gate residual:NO error:e]) return NO;
                if (![self encodeDecodeMatvecTensor:enc weight:layer.ffnUp input:_norm inputOffset:0 dst:_ffn_up residual:NO error:e]) return NO;
                return [self encodeSwiGLU:enc n:ffn error:e];
            } error:error]) return NO;
        qw35_accum_absmean(_capDown + (size_t)il * ffn, (const float *)[_ffn_gate contents], ffn);

        // Down projection (residual) + the next layer's input norm.
        if (![self captureRun:^BOOL(id<MTLComputeCommandEncoder> enc, NSError **e) {
                if (![self encodeDecodeMatvecTensor:enc weight:layer.ffnDown input:_ffn_gate inputOffset:0 dst:_act_a residual:YES error:e]) return NO;
                if (ilc + 1 < layers) {
                    return [self encodeRms:enc src:_act_a weightTensor:_layers[ilc + 1].attnNorm error:e];
                } else if (logitsMode != QW35_LOGITS_NONE) {
                    return [self encodeRms:enc src:_act_a weightTensor:_outputNorm error:e];
                }
                return YES;
            } error:error]) return NO;
    }

    if (logitsMode != QW35_LOGITS_NONE) {
        if (![self captureRun:^BOOL(id<MTLComputeCommandEncoder> enc, NSError **e) {
                return [self encodeOutputHead:enc normRowOffset:0 logitsMode:logitsMode error:e];
            } error:error]) return NO;
    }

    _capTokens++;
    if ((_capTokens % 16) == 0) [self writeCaptureFile];
    return YES;
}

// ---------------------------------------------------------------------------
#pragma mark - Per-stage GPU profiler (QW35_STAGE_PROFILE)
// ---------------------------------------------------------------------------

/// Encode one stage class into its own command buffer, wait, and add the
/// measured on-GPU time (GPUEndTime - GPUStartTime, seconds) to *accum. The
/// commit/wait CPU latency is NOT attributed (we read GPU timestamps), so the
/// per-stage buckets stay accurate even though the run is serialised.
- (BOOL)profStage:(BOOL (^)(id<MTLComputeCommandEncoder> enc, NSError **error))block
            accum:(double *)accum
            error:(NSError **)error {
    id<MTLCommandBuffer> cb = [_queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    BOOL ok = block(enc, error);
    [enc endEncoding];
    if (!ok) return NO;
    [cb commit];
    [cb waitUntilCompleted];
    double dt = cb.GPUEndTime - cb.GPUStartTime;
    if (dt > 0) *accum += dt;
    _lastCB = cb;
    return YES;
}

/// Print the current 16-token window's per-stage GPU breakdown at the current
/// context, then reset the accumulators so the next window shows the trend as
/// ctx grows. With --attn-window set, `1-attn-layer us/tok` should plateau once
/// ctx exceeds the window; if it keeps climbing with ctx, the window bound is
/// not reaching the attention kernel.
- (void)printStageProfile {
    double total = _prof_attn + _prof_delta + _prof_ffn + _prof_head + _prof_norm;
    if (total <= 0 || _prof_tokens == 0) { _prof_tokens = 0; return; }
    const double t = (double)_prof_tokens;
    const double us = 1.0e6;
    const uint32_t na = _attentionLayerCount ? _attentionLayerCount : 1;
    const uint32_t nd = _deltaLayerCount ? _deltaLayerCount : 1;
    const uint32_t nl = _h.transformer_layers ? _h.transformer_layers : 1;
    fprintf(stderr,
        "[QW35_STAGE_PROFILE] ctx=%u tokens=%llu gpu=%.2fms/tok  "
        "attn=%.1f%% delta=%.1f%% ffn=%.1f%% head=%.1f%% norm=%.1f%%\n",
        _prof_pos, (unsigned long long)_prof_tokens, total / t * 1000.0,
        _prof_attn / total * 100, _prof_delta / total * 100,
        _prof_ffn / total * 100, _prof_head / total * 100, _prof_norm / total * 100);
    fprintf(stderr,
        "[QW35_STAGE_PROFILE]   us/tok: 1-attn-layer=%.1f (x%u=%.1f)  "
        "1-delta-layer=%.1f (x%u=%.1f)  1-ffn=%.1f (x%u=%.1f)  head=%.1f\n",
        _prof_attn / t / na * us, na, _prof_attn / t * us,
        _prof_delta / t / nd * us, nd, _prof_delta / t * us,
        _prof_ffn / t / nl * us, nl, _prof_ffn / t * us,
        _prof_head / t * us);
    _prof_attn = _prof_delta = _prof_ffn = _prof_head = _prof_norm = 0;
    _prof_tokens = 0;
}

/// Profiling twin of -evalToken:. Mirrors -encodeDecodeLayers: exactly but
/// commits one command buffer per stage class so each stage's true on-GPU time
/// is attributed (see -profStage:accum:). Output is identical to the fast path
/// (same encode helpers, same order), so generation continues normally; only
/// wall-clock is slower. Enabled by QW35_STAGE_PROFILE.
- (BOOL)evalTokenStageProfiled:(uint32_t)token
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

    const uint32_t layers = _h.transformer_layers;
    const uint32_t interval = _h.full_attention_interval ? _h.full_attention_interval : 4;
    uint32_t deltaSlot = 0, attnSlot = 0;

    for (uint32_t il = 0; il < layers; il++) {
        Qw35LayerTensors *layer = _layers[il];
        const BOOL isAttn = (((il + 1) % interval) == 0);
        const BOOL first = (il == 0);
        const uint32_t aSlot = attnSlot, dSlot = deltaSlot;
        const uint32_t ilc = il;

        // Stage 1 — mixer (+ embedding/first input-norm on layer 0).
        if (![self profStage:^BOOL(id<MTLComputeCommandEncoder> enc, NSError **e) {
                if (first) {
                    if (![self encodeEmbedding:enc token:token error:e]) return NO;
                    if (![self encodeRms:enc src:_act_a weightTensor:_layers[0].attnNorm error:e]) return NO;
                }
                if (isAttn) return [self encodeAttentionDecodeLayer:enc layer:layer slot:aSlot pos:pos error:e];
                return [self encodeDeltaDecodeLayer:enc layer:layer slot:dSlot error:e];
            } accum:(isAttn ? &_prof_attn : &_prof_delta) error:error]) return NO;
        if (isAttn) attnSlot++; else deltaSlot++;

        // Stage 2 — residual add + post-attention norm.
        if (![self profStage:^BOOL(id<MTLComputeCommandEncoder> enc, NSError **e) {
                return [self encodeResidualRms:enc weightTensor:layer.postAttentionNormDecode error:e];
            } accum:&_prof_norm error:error]) return NO;

        // Stage 3 — FFN (fused GF4 when type-id 100, else split).
        if (![self profStage:^BOOL(id<MTLComputeCommandEncoder> enc, NSError **e) {
                if (layer.ffnGate.type_id == 100 && layer.ffnUp.type_id == 100) {
                    return [self encodeGf4FusedFfn:enc layer:layer error:e];
                }
                if (![self encodeDecodeMatvecTensor:enc weight:layer.ffnGate input:_norm inputOffset:0 dst:_ffn_gate residual:NO error:e]) return NO;
                if (![self encodeDecodeMatvecTensor:enc weight:layer.ffnUp input:_norm inputOffset:0 dst:_ffn_up residual:NO error:e]) return NO;
                if (![self encodeSwiGLU:enc n:_h.feed_forward_length error:e]) return NO;
                return [self encodeDecodeMatvecTensor:enc weight:layer.ffnDown input:_ffn_gate inputOffset:0 dst:_act_a residual:YES error:e];
            } accum:&_prof_ffn error:error]) return NO;

        // Stage 4 — chain the next layer's input norm (or the output norm).
        if (![self profStage:^BOOL(id<MTLComputeCommandEncoder> enc, NSError **e) {
                if (ilc + 1 < layers) return [self encodeRms:enc src:_act_a weightTensor:_layers[ilc + 1].attnNorm error:e];
                if (logitsMode != QW35_LOGITS_NONE) return [self encodeRms:enc src:_act_a weightTensor:_outputNorm error:e];
                return YES;
            } accum:&_prof_norm error:error]) return NO;
    }

    if (logitsMode != QW35_LOGITS_NONE) {
        if (![self profStage:^BOOL(id<MTLComputeCommandEncoder> enc, NSError **e) {
                return [self encodeOutputHead:enc normRowOffset:0 logitsMode:logitsMode error:e];
            } accum:&_prof_head error:error]) return NO;
    }

    _prof_pos = pos;
    _prof_tokens++;
    if ((_prof_tokens % 16) == 0) [self printStageProfile];
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
        if (![self encodeResidualRms:enc weightTensor:layer.postAttentionNormDecode error:error]) return NO;

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

    // Segmented KV cache: a prefill batch's writes must land in a single slab
    // (the per-token write binds one slab buffer). Split a batch that straddles
    // a slab boundary into slab-aligned sub-batches; only the final sub-batch
    // emits logits. Reads span slabs via the bindless slab array, so they are
    // unaffected. With the default slab (8192) this never triggers.
    const uint32_t slabSize = _kvSlab ? _kvSlab : 1;
    if (pos0 / slabSize != (uint32_t)((uint64_t)pos0 + count - 1) / slabSize) {
        uintptr_t off = 0;
        uint32_t p = pos0;
        while (off < count) {
            uint32_t boundary = (p / slabSize + 1) * slabSize;
            uintptr_t chunk = (uintptr_t)(boundary - p);
            if (chunk > count - off) chunk = count - off;
            qw35_logits_mode subMode = (off + chunk >= count) ? logitsMode : QW35_LOGITS_NONE;
            if (![self evalTokens:tokens + off count:chunk pos0:p logitsMode:subMode error:error]) return NO;
            off += chunk;
            p += (uint32_t)chunk;
        }
        return YES;
    }

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

@end
