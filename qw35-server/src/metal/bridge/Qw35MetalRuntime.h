// Qw35MetalRuntime.h – GPU inference runtime for the Qwen3.5-9B model.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "Qw35MetalTypes.h"

@class Qw35TensorStore;
@class Qw35PipelineCache;

/// Orchestrates the full Metal inference pipeline for a single Qwen3.5 model.
///
/// Lifecycle: created with a mmap-backed model and tensor descriptors,
/// used for one inference session, then destroyed.  Call -reset to start
/// a new session (clears conv_state and ssm_state).
///
/// The runtime owns all scratch buffers, the pipeline cache, and the
/// tensor store.  It does NOT perform CPU-side sampling — it only runs
/// the GPU graph and returns argmax + logits.
@interface Qw35MetalRuntime : NSObject

/// Initialize with the model mmap, tensor list, hyperparameters, and
/// compiled metallib bytes.  Allocates all scratch buffers and precomputes
/// RoPE frequency tables.
///
/// The unified .gguf carries its GF4 FFN as type-id 100 tensors in the tensor
/// list; the type_id selects GF4 vs Q4_K kernels per weight.
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

/// Reset convolution state and SSM state to zero for a new generation.
/// Waits for any in-flight GPU work first.
- (BOOL)reset:(NSError **)error;

/// Raise the context ceiling to `newCtx` positions. The segmented KV cache
/// grows lazily toward the ceiling as positions are actually reached, so this
/// only moves the scalar bound — no reallocation, no replay; live KV rows,
/// recurrent state, and snapshots all stay valid. Lowering is refused (KV rows
/// past a lowered ceiling would still be addressed by live sessions); a value
/// at or under the current ceiling is a no-op success.
- (BOOL)setCtxSize:(uint32_t)newCtx error:(NSError **)error;

/// Total byte size of the conv/SSM recurrent state (the export/import
/// payload). 0 if the state buffers are not allocated.
- (uint64_t)stateSize;

/// Copy the conv/SSM recurrent state into a caller-owned buffer of exactly
/// -stateSize bytes. Waits for in-flight GPU work first. Used by the session
/// prefix cache to mark rewind points (the hybrid SSM state cannot be rolled
/// back otherwise); the caller may hold any number of snapshots.
- (BOOL)stateExport:(void *)buf length:(uint64_t)len error:(NSError **)error;

/// Restore the conv/SSM recurrent state from a caller-owned buffer previously
/// filled by -stateExport:length:error:. KV cache rows are untouched: they
/// are positional and remain valid for every position evaluated before the
/// snapshot was taken.
- (BOOL)stateImport:(const void *)buf length:(uint64_t)len error:(NSError **)error;

/// Evaluate a single token through the entire model graph.
///
/// The work is committed asynchronously across two command buffers; the call
/// returns after encoding. Results become observable through
/// -readArgmaxToken:... or -copyLogits:..., which wait for completion.
- (BOOL)evalToken:(uint32_t)token
              pos:(uint32_t)pos
       logitsMode:(qw35_logits_mode)logitsMode
            error:(NSError **)error;

/// Evaluate a batch of tokens (prefill) through the model graph.
- (BOOL)evalTokens:(const uint32_t *)tokens
             count:(uintptr_t)count
              pos0:(uint32_t)pos0
        logitsMode:(qw35_logits_mode)logitsMode
             error:(NSError **)error;

/// Wait for all in-flight GPU work to complete.
- (BOOL)sync:(NSError **)error;

/// Set the decode-time sliding-window attention sink (first N positions always
/// attended). Set per request to keep the system prompt + first user turn pinned.
- (void)setAttnSink:(int)sink;

/// Wait for in-flight work, then read back the argmax token and its logit.
- (BOOL)readArgmaxToken:(uint32_t *)token logit:(float *)logit error:(NSError **)error;

/// Wait for in-flight work, then copy the full logits vector to CPU memory.
- (BOOL)copyLogits:(float *)dst len:(uintptr_t)len error:(NSError **)error;

@end
