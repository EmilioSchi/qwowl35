// Qw35TensorStore.h – Zero-copy MTLBuffer wrapper for mmap-backed GGUF tensors.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "Qw35MetalTypes.h"

/// A single tensor backed by a page-aligned MTLBuffer view over the mmaped model.
@interface Qw35Tensor : NSObject
@property (nonatomic, copy)   NSString   *name;
@property (nonatomic, strong) id<MTLBuffer> buffer;
@property (nonatomic, assign) NSUInteger  offset;       // byte offset within buffer
@property (nonatomic, assign) uint32_t    type_id;
@property (nonatomic, assign) uint32_t    n_dims;
@property (nonatomic, assign) uint64_t    bytes;
@property (nonatomic, assign) uint64_t    elements;
/// Pointer to the internal 4-element dimension array (dims[0] is the row length).
- (uint64_t *)dims;
@end

/// Manages zero-copy MTLBuffer views of the mmap-backed GGUF tensor region.
/// Each tensor is wrapped in a page-aligned MTLBuffer created via
/// -[MTLDevice newBufferWithBytesNoCopy:length:options:deallocator:].
@interface Qw35TensorStore : NSObject

/// Initialize with the model mmap + array of tensor descriptors.
/// Creates a page-aligned no-copy MTLBuffer for each tensor.
- (instancetype)initWithModelMap:(const void *)modelMap
                       modelSize:(uint64_t)modelSize
                          tensors:(const qw35_metal_tensor_desc *)tensorDescs
                      tensorCount:(uintptr_t)tensorCount
                           device:(id<MTLDevice>)device
                            error:(NSError **)error;

/// Look up a tensor by name.  Returns nil if not found.
- (Qw35Tensor *)tensorNamed:(NSString *)name;

/// All distinct backing MTLBuffers (one per tensor view), for adding to a
/// residency set so the mmap-backed weight pages are not paged out under
/// unified-memory pressure during a long decode session.
- (NSArray<id<MTLBuffer>> *)allBuffers;

/// Validate that the essential tensors (embedding, output norm, output head)
/// exist and have the expected types and dimensions.
- (BOOL)validateRequiredForEmbeddingLength:(uint32_t)embeddingLength
                                 vocabSize:(uint32_t)vocabSize
                                     error:(NSError **)error;

@end
