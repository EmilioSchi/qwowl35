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
/// via function constants (indices 750-753).
- (id<MTLComputePipelineState>)attnPipelineNamed:(NSString *)name
                                           heads:(int)heads
                                         kvHeads:(int)kvHeads
                                         headDim:(int)headDim
                                         ropeDim:(int)ropeDim
                                           error:(NSError **)error;

@end
