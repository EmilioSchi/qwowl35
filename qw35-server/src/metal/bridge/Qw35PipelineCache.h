// Qw35PipelineCache.h – Lazy-compiled Metal compute pipeline cache.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

/// Thread-safe cache of MTLComputePipelineState objects.
/// Pipelines are lazily compiled on first access and reused thereafter.
@interface Qw35PipelineCache : NSObject

- (instancetype)initWithLibrary:(id<MTLLibrary>)library device:(id<MTLDevice>)device;

/// Get (or compile) a compute pipeline by kernel function name.
- (id<MTLComputePipelineState>)pipelineNamed:(NSString *)name error:(NSError **)error;

/// Get (or compile) a tiled matmul pipeline with function constants.
/// bcInp controls boundary-checking on the input dimension (index 700).
/// bcOut controls boundary-checking on the output dimensions (index 701).
- (id<MTLComputePipelineState>)mulMmPipelineNamed:(NSString *)name
                                            bcInp:(BOOL)bcInp
                                           bcOut:(BOOL)bcOut
                                            error:(NSError **)error;

/// Get (or compile) an attention pipeline specialized on the model shape
/// via function constants (indices 750-754). hasGate selects the fused
/// sigmoid output gate (Qwen3.5 hybrid attention, Q stride head_dim*2)
/// vs plain ungated attention (dense Qwen3).
- (id<MTLComputePipelineState>)attnPipelineNamed:(NSString *)name
                                           heads:(int)heads
                                         kvHeads:(int)kvHeads
                                         headDim:(int)headDim
                                         ropeDim:(int)ropeDim
                                         hasGate:(BOOL)hasGate
                                           error:(NSError **)error;

@end
