// Qw35PipelineCache.m – Implementation of lazy Metal pipeline cache.

#import "Qw35PipelineCache.h"

@interface Qw35PipelineCache ()
@property (nonatomic, weak)   id<MTLDevice>  device;
@property (nonatomic, weak)   id<MTLLibrary> library;
@property (nonatomic, strong) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *cache;
@property (nonatomic, strong) dispatch_queue_t syncQueue;
@end

@implementation Qw35PipelineCache

- (instancetype)initWithLibrary:(id<MTLLibrary>)library device:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        _device  = device;
        _library = library;
        _cache   = [NSMutableDictionary dictionary];
        _syncQueue = dispatch_queue_create("com.qw35.pipeline-cache", DISPATCH_QUEUE_SERIAL);
    }
    return self;
}

- (id<MTLComputePipelineState>)pipelineNamed:(NSString *)name error:(NSError **)error {
    __block id<MTLComputePipelineState> result = nil;
    dispatch_sync(_syncQueue, ^{
        result = self.cache[name];
        if (result) return;

        id<MTLFunction> function = [self.library newFunctionWithName:name];
        if (!function) {
            if (error) *error = [NSError errorWithDomain:@"Qw35PipelineCache"
                                                    code:-1
                                                userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Metal function %@ not found", name]}];
            return;
        }

        NSError *nsError = nil;
        result = [self.device newComputePipelineStateWithFunction:function error:&nsError];
        if (!result) {
            if (error) *error = nsError ?: [NSError errorWithDomain:@"Qw35PipelineCache"
                                                               code:-2
                                                           userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"failed to compile pipeline %@", name]}];
            return;
        }
        self.cache[name] = result;
    });
    return result;
}

- (id<MTLComputePipelineState>)attnPipelineNamed:(NSString *)name
                                           heads:(int)heads
                                         kvHeads:(int)kvHeads
                                         headDim:(int)headDim
                                         ropeDim:(int)ropeDim
                                         hasGate:(BOOL)hasGate
                                           error:(NSError **)error {
    NSString *key = [NSString stringWithFormat:@"%@#attn=%d/%d/%d/%d/%d", name, heads, kvHeads, headDim, ropeDim, hasGate ? 1 : 0];

    __block id<MTLComputePipelineState> result = nil;
    dispatch_sync(_syncQueue, ^{
        result = self.cache[key];
        if (result) return;

        MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
        int headsVal = heads;
        int kvHeadsVal = kvHeads;
        int headDimVal = headDim;
        int ropeDimVal = ropeDim;
        BOOL hasGateVal = hasGate;
        [constants setConstantValue:&headsVal type:MTLDataTypeInt atIndex:750];
        [constants setConstantValue:&kvHeadsVal type:MTLDataTypeInt atIndex:751];
        [constants setConstantValue:&headDimVal type:MTLDataTypeInt atIndex:752];
        [constants setConstantValue:&ropeDimVal type:MTLDataTypeInt atIndex:753];
        [constants setConstantValue:&hasGateVal type:MTLDataTypeBool atIndex:754];

        NSError *nsError = nil;
        id<MTLFunction> function = [self.library newFunctionWithName:name
                                                     constantValues:constants
                                                              error:&nsError];
        if (!function) {
            if (error) *error = nsError ?: [NSError errorWithDomain:@"Qw35PipelineCache"
                                                               code:-5
                                                           userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Metal function %@ with attention constants not found", name]}];
            return;
        }

        result = [self.device newComputePipelineStateWithFunction:function error:&nsError];
        if (!result) {
            if (error) *error = nsError ?: [NSError errorWithDomain:@"Qw35PipelineCache"
                                                               code:-6
                                                           userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"failed to compile attention pipeline %@", name]}];
            return;
        }
        self.cache[key] = result;
    });
    return result;
}

- (id<MTLComputePipelineState>)mulMmPipelineNamed:(NSString *)name
                                            bcInp:(BOOL)bcInp
                                           bcOut:(BOOL)bcOut
                                            error:(NSError **)error {
    NSString *key = [NSString stringWithFormat:@"%@#bc_inp=%d#bc_out=%d", name, bcInp ? 1 : 0, bcOut ? 1 : 0];

    __block id<MTLComputePipelineState> result = nil;
    dispatch_sync(_syncQueue, ^{
        result = self.cache[key];
        if (result) return;

        MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
        BOOL bcInpVal = bcInp;
        BOOL bcOutVal = bcOut;
        [constants setConstantValue:&bcInpVal type:MTLDataTypeBool atIndex:700];
        [constants setConstantValue:&bcOutVal type:MTLDataTypeBool atIndex:701];

        NSError *nsError = nil;
        id<MTLFunction> function = [self.library newFunctionWithName:name
                                                     constantValues:constants
                                                              error:&nsError];
        if (!function) {
            if (error) *error = nsError ?: [NSError errorWithDomain:@"Qw35PipelineCache"
                                                               code:-3
                                                           userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Metal function %@ with constants not found", name]}];
            return;
        }

        result = [self.device newComputePipelineStateWithFunction:function error:&nsError];
        if (!result) {
            if (error) *error = nsError ?: [NSError errorWithDomain:@"Qw35PipelineCache"
                                                               code:-4
                                                           userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"failed to compile tiled pipeline %@", name]}];
            return;
        }
        self.cache[key] = result;
    });
    return result;
}

@end
