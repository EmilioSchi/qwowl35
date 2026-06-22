// Qw35TensorStore.m – Implementation of zero-copy MTLBuffer tensor store.

#import "Qw35TensorStore.h"
#import <unistd.h>

@implementation Qw35Tensor {
    uint64_t _dims[4];
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _buffer  = nil;
        _offset  = 0;
        _type_id = 0;
        _n_dims  = 0;
        memset(_dims, 0, sizeof(_dims));
        _bytes    = 0;
        _elements = 0;
    }
    return self;
}

- (uint64_t *)dims {
    return _dims;
}

@end

// ---------------------------------------------------------------------------
#pragma mark - Qw35TensorStore
// ---------------------------------------------------------------------------

@interface Qw35TensorStore ()
@property (nonatomic, strong) NSMutableDictionary<NSString *, Qw35Tensor *> *tensors;
@end

@implementation Qw35TensorStore

static inline uint64_t round_up_u64(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}

- (instancetype)initWithModelMap:(const void *)modelMap
                       modelSize:(uint64_t)modelSize
                          tensors:(const qw35_metal_tensor_desc *)tensorDescs
                      tensorCount:(uintptr_t)tensorCount
                           device:(id<MTLDevice>)device
                            error:(NSError **)error {
    self = [super init];
    if (!self) return nil;

    if (!modelMap || modelSize == 0 || !tensorDescs || tensorCount == 0 || !device) {
        if (error) *error = [NSError errorWithDomain:@"Qw35TensorStore"
                                                code:-1
                                            userInfo:@{NSLocalizedDescriptionKey: @"invalid tensor store inputs"}];
        return nil;
    }

    const uint64_t page = (uint64_t)getpagesize();
    const uintptr_t model_addr = (uintptr_t)modelMap;
    if ((model_addr & (uintptr_t)(page - 1)) != 0) {
        if (error) *error = [NSError errorWithDomain:@"Qw35TensorStore"
                                                code:-2
                                            userInfo:@{NSLocalizedDescriptionKey: @"model mmap base is not page aligned"}];
        return nil;
    }

    uint64_t maxBuffer = (uint64_t)[device maxBufferLength];
    maxBuffer &= ~(page - 1);
    if (maxBuffer == 0) {
        if (error) *error = [NSError errorWithDomain:@"Qw35TensorStore"
                                                code:-3
                                            userInfo:@{NSLocalizedDescriptionKey: @"Metal maxBufferLength is too small"}];
        return nil;
    }

    _tensors = [NSMutableDictionary dictionaryWithCapacity:(NSUInteger)tensorCount];

    for (uintptr_t i = 0; i < tensorCount; i++) {
        const qw35_metal_tensor_desc *desc = &tensorDescs[i];
        NSString *name = [[NSString alloc] initWithBytes:desc->name
                                                   length:(NSUInteger)desc->name_len
                                                 encoding:NSUTF8StringEncoding];
        if (!name) {
            if (error) *error = [NSError errorWithDomain:@"Qw35TensorStore"
                                                    code:-4
                                                userInfo:@{NSLocalizedDescriptionKey: @"tensor name is not valid UTF-8"}];
            return nil;
        }

        if (desc->abs_offset > modelSize || desc->bytes > modelSize - desc->abs_offset) {
            if (error) *error = [NSError errorWithDomain:@"Qw35TensorStore"
                                                    code:-5
                                                userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"tensor %@ points outside the mapped model", name]}];
            return nil;
        }

        const uint64_t alignedOffset = desc->abs_offset & ~(page - 1);
        const uint64_t leading       = desc->abs_offset - alignedOffset;
        uint64_t bufferBytes         = round_up_u64(leading + desc->bytes, page);
        if (bufferBytes > modelSize - alignedOffset) {
            bufferBytes = modelSize - alignedOffset;
        }
        if (bufferBytes > maxBuffer) {
            if (error) *error = [NSError errorWithDomain:@"Qw35TensorStore"
                                                    code:-6
                                                userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"tensor %@ exceeds Metal maxBufferLength", name]}];
            return nil;
        }

        id<MTLBuffer> tensorBuffer =
            [device newBufferWithBytesNoCopy:(void *)(model_addr + alignedOffset)
                                      length:(NSUInteger)bufferBytes
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
        if (!tensorBuffer) {
            if (error) *error = [NSError errorWithDomain:@"Qw35TensorStore"
                                                    code:-7
                                                userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Metal could not wrap mmap-backed tensor %@", name]}];
            return nil;
        }
        tensorBuffer.label = name;

        Qw35Tensor *tensor = [[Qw35Tensor alloc] init];
        tensor.name     = name;
        tensor.buffer   = tensorBuffer;
        tensor.offset   = (NSUInteger)leading;
        tensor.type_id  = desc->type_id;
        tensor.n_dims   = desc->n_dims;
        tensor.bytes    = desc->bytes;
        tensor.elements = desc->elements;
        for (int d = 0; d < 4; d++) tensor.dims[d] = desc->dims[d];

        _tensors[name] = tensor;
    }

    return self;
}

- (Qw35Tensor *)tensorNamed:(NSString *)name {
    return _tensors[name];
}

- (BOOL)validateRequiredForEmbeddingLength:(uint32_t)embeddingLength
                                 vocabSize:(uint32_t)vocabSize
                                     error:(NSError **)error {
    NSArray<NSString *> *required = @[@"token_embd.weight", @"output_norm.weight", @"output.weight"];
    for (NSString *reqName in required) {
        if (!_tensors[reqName]) {
            if (error) *error = [NSError errorWithDomain:@"Qw35TensorStore"
                                                    code:-10
                                                userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"missing required tensor %@", reqName]}];
            return NO;
        }
    }

    Qw35Tensor *embd = _tensors[@"token_embd.weight"];
    if (embd.type_id != 12) { // must be q4_k
        if (error) *error = [NSError errorWithDomain:@"Qw35TensorStore"
                                                code:-11
                                            userInfo:@{NSLocalizedDescriptionKey: @"token_embd.weight must be q4_k"}];
        return NO;
    }
    if (embd.dims[0] != embeddingLength || embd.dims[1] != vocabSize) {
        if (error) *error = [NSError errorWithDomain:@"Qw35TensorStore"
                                                code:-12
                                            userInfo:@{NSLocalizedDescriptionKey: @"token_embd.weight has unexpected dimensions"}];
        return NO;
    }
    return YES;
}

@end
