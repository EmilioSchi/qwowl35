// Qw35FFI.m – C FFI implementations bridging Rust to Qw35MetalRuntime.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#import "Qw35FFI.h"
#import "Qw35MetalRuntime.h"

static void qw35_set_error(char *err, uintptr_t err_len, const char *msg) {
    if (!err || err_len == 0) return;
    snprintf(err, (size_t)err_len, "%s", msg ? msg : "unknown Metal error");
}

static void qw35_set_ns_error(char *err, uintptr_t err_len, NSError *error) {
    qw35_set_error(err, err_len, error ? [[error localizedDescription] UTF8String] : NULL);
}

static double qw35_now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static uint64_t qw35_round_up_u64(uint64_t value, uint64_t alignment) {
    const uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}

int qw35_metal_has_device(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
}

int qw35_metal_warm_model_views(
        const void *model_map,
        uint64_t model_size,
        uint64_t map_offset,
        uint64_t map_size,
        uint64_t max_tensor_bytes,
        const uint8_t *metallib,
        uintptr_t metallib_len,
        uint64_t stride,
        qw35_metal_warm_report *report,
        char *err,
        uintptr_t err_len) {
    @autoreleasepool {
        if (report) memset(report, 0, sizeof(*report));
        if (!model_map || model_size == 0 || !metallib || metallib_len == 0) {
            qw35_set_error(err, err_len, "invalid Metal warmup inputs");
            return 0;
        }
        if (map_offset > model_size || map_size > model_size - map_offset) {
            qw35_set_error(err, err_len, "Metal warmup range is outside the mapped model");
            return 0;
        }
        if (stride == 0) stride = 1024ull * 1024ull;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            qw35_set_error(err, err_len, "no default Metal device is available");
            return 0;
        }

        const uint64_t page = (uint64_t)getpagesize();
        const uintptr_t model_addr = (uintptr_t)model_map;
        if ((model_addr & (uintptr_t)(page - 1)) != 0) {
            qw35_set_error(err, err_len, "model mmap base is not page aligned");
            return 0;
        }

        const uint64_t page_model_offset = map_offset & ~(page - 1);
        const uint64_t leading = map_offset - page_model_offset;
        const uint64_t mapped_model_size = qw35_round_up_u64(leading + map_size, page);
        uint64_t max_buffer = (uint64_t)[device maxBufferLength];
        max_buffer &= ~(page - 1);
        const uint64_t overlap = qw35_round_up_u64(max_tensor_bytes, page) + page;
        if (max_buffer == 0 || max_buffer <= overlap) {
            qw35_set_error(err, err_len, "Metal maxBufferLength is too small for model views");
            return 0;
        }

        dispatch_data_t lib_data = dispatch_data_create(
            metallib,
            (size_t)metallib_len,
            nil,
            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithData:lib_data error:&error];
        if (!library) {
            qw35_set_ns_error(err, err_len, error);
            return 0;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"qw35_touch_u8_stride"];
        if (!function) {
            qw35_set_error(err, err_len, "qw35_touch_u8_stride not found in Metal library");
            return 0;
        }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            qw35_set_ns_error(err, err_len, error);
            return 0;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            qw35_set_error(err, err_len, "failed to create Metal command queue");
            return 0;
        }

        NSMutableArray<id<MTLBuffer>> *views = [NSMutableArray array];
        const uint64_t step = max_buffer - overlap;
        uint64_t off = 0;
        uint64_t total_touches = 0;
        uint64_t mapped_bytes = 0;
        while (off < mapped_model_size) {
            uint64_t view_bytes = mapped_model_size - off;
            if (view_bytes > max_buffer) view_bytes = max_buffer;

            id<MTLBuffer> buffer =
                [device newBufferWithBytesNoCopy:(void *)(model_addr + page_model_offset + off)
                                          length:(NSUInteger)view_bytes
                                         options:MTLResourceStorageModeShared
                                     deallocator:nil];
            if (!buffer) {
                qw35_set_error(err, err_len, "Metal could not wrap mmaped model view");
                return 0;
            }
            buffer.label = [NSString stringWithFormat:@"qw35_model_view_%lu", (unsigned long)[views count]];
            [views addObject:buffer];
            total_touches += (view_bytes + stride - 1) / stride;
            mapped_bytes += view_bytes;

            if (off + view_bytes >= mapped_model_size) break;
            off += step;
        }

        if (total_touches == 0 || total_touches > (uint64_t)NSUIntegerMax) {
            qw35_set_error(err, err_len, "invalid Metal warmup touch count");
            return 0;
        }
        id<MTLBuffer> out = [device newBufferWithLength:(NSUInteger)total_touches
                                                options:MTLResourceStorageModeShared];
        if (!out) {
            qw35_set_error(err, err_len, "failed to allocate Metal warmup output buffer");
            return 0;
        }

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        uint64_t dst_offset = 0;
        for (id<MTLBuffer> view in views) {
            const uint64_t bytes = (uint64_t)[view length];
            const uint64_t n = (bytes + stride - 1) / stride;
            [enc setBuffer:view offset:0 atIndex:0];
            [enc setBuffer:out offset:0 atIndex:1];
            [enc setBytes:&stride length:sizeof(stride) atIndex:2];
            [enc setBytes:&bytes length:sizeof(bytes) atIndex:3];
            [enc setBytes:&dst_offset length:sizeof(dst_offset) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((n + 255) / 256), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            dst_offset += n;
        }
        [enc endEncoding];

        const double t0 = qw35_now_ms();
        [cb commit];
        [cb waitUntilCompleted];
        const double t1 = qw35_now_ms();

        if (cb.status == MTLCommandBufferStatusError) {
            qw35_set_ns_error(err, err_len, cb.error);
            return 0;
        }

        uint64_t checksum = 0;
        const uint8_t *out_bytes = (const uint8_t *)[out contents];
        for (uint64_t i = 0; i < total_touches; i++) checksum += out_bytes[i];

        if (report) {
            report->bytes = map_size;
            report->mapped_bytes = mapped_bytes;
            report->touches = total_touches;
            report->checksum = checksum;
            report->elapsed_ms = t1 - t0;
            report->view_count = (uint32_t)[views count];
        }
        return 1;
    }
}

void *qw35_metal_runtime_create(
        const void *model_map,
        uint64_t model_size,
        const qw35_metal_tensor_desc *tensors,
        uintptr_t tensor_count,
        const void *gf4_map,
        uint64_t gf4_size,
        const qw35_metal_tensor_desc *gf4_tensors,
        uintptr_t gf4_tensor_count,
        const qw35_metal_hparams *hparams,
        uint32_t ctx_size,
        uint32_t vocab_size,
        uint32_t prefill_chunk,
        uint32_t kv_cache_type,
        const uint8_t *metallib,
        uintptr_t metallib_len,
        char *err,
        uintptr_t err_len) {
    @autoreleasepool {
        NSError *error = nil;
        Qw35MetalRuntime *runtime =
            [[Qw35MetalRuntime alloc] initWithModelMap:model_map
                                             modelSize:model_size
                                               tensors:tensors
                                           tensorCount:tensor_count
                                                gf4Map:gf4_map
                                               gf4Size:gf4_size
                                            gf4Tensors:gf4_tensors
                                        gf4TensorCount:gf4_tensor_count
                                               hparams:hparams
                                               ctxSize:ctx_size
                                             vocabSize:vocab_size
                                          prefillChunk:prefill_chunk
                                           kvCacheType:kv_cache_type
                                              metallib:metallib
                                           metallibLen:metallib_len
                                                 error:&error];
        if (!runtime) {
            qw35_set_ns_error(err, err_len, error);
            return NULL;
        }
        return (void *)CFBridgingRetain(runtime);
    }
}

void qw35_metal_runtime_destroy(void *runtime) {
    if (!runtime) return;
    @autoreleasepool {
        CFRelease(runtime);
    }
}

int qw35_metal_runtime_reset(void *runtime, char *err, uintptr_t err_len) {
    @autoreleasepool {
        if (!runtime) {
            qw35_set_error(err, err_len, "invalid Qw35 Metal runtime");
            return 0;
        }
        NSError *error = nil;
        Qw35MetalRuntime *rt = (__bridge Qw35MetalRuntime *)runtime;
        if (![rt reset:&error]) {
            qw35_set_ns_error(err, err_len, error);
            return 0;
        }
        return 1;
    }
}

int qw35_metal_runtime_state_checkpoint_save(void *runtime, char *err, uintptr_t err_len) {
    @autoreleasepool {
        if (!runtime) {
            qw35_set_error(err, err_len, "invalid Qw35 Metal runtime");
            return 0;
        }
        NSError *error = nil;
        Qw35MetalRuntime *rt = (__bridge Qw35MetalRuntime *)runtime;
        if (![rt stateCheckpointSave:&error]) {
            qw35_set_ns_error(err, err_len, error);
            return 0;
        }
        return 1;
    }
}

int qw35_metal_runtime_state_checkpoint_restore(void *runtime, char *err, uintptr_t err_len) {
    @autoreleasepool {
        if (!runtime) {
            qw35_set_error(err, err_len, "invalid Qw35 Metal runtime");
            return 0;
        }
        NSError *error = nil;
        Qw35MetalRuntime *rt = (__bridge Qw35MetalRuntime *)runtime;
        if (![rt stateCheckpointRestore:&error]) {
            qw35_set_ns_error(err, err_len, error);
            return 0;
        }
        return 1;
    }
}

int qw35_metal_runtime_sync(void *runtime, char *err, uintptr_t err_len) {
    @autoreleasepool {
        if (!runtime) {
            qw35_set_error(err, err_len, "invalid Qw35 Metal runtime");
            return 0;
        }
        NSError *error = nil;
        Qw35MetalRuntime *rt = (__bridge Qw35MetalRuntime *)runtime;
        if (![rt sync:&error]) {
            qw35_set_ns_error(err, err_len, error);
            return 0;
        }
        return 1;
    }
}

int qw35_metal_runtime_eval_token(
        void *runtime,
        uint32_t token,
        uint32_t pos,
        int logits_mode,
        char *err,
        uintptr_t err_len) {
    @autoreleasepool {
        if (!runtime) {
            qw35_set_error(err, err_len, "invalid Qw35 Metal runtime");
            return 0;
        }
        NSError *error = nil;
        Qw35MetalRuntime *rt = (__bridge Qw35MetalRuntime *)runtime;
        if (![rt evalToken:token pos:pos logitsMode:(qw35_logits_mode)logits_mode error:&error]) {
            qw35_set_ns_error(err, err_len, error);
            return 0;
        }
        return 1;
    }
}

int qw35_metal_runtime_eval_tokens(
        void *runtime,
        const uint32_t *tokens,
        uintptr_t len,
        uint32_t pos0,
        int logits_mode,
        char *err,
        uintptr_t err_len) {
    @autoreleasepool {
        if (!runtime) {
            qw35_set_error(err, err_len, "invalid Qw35 Metal runtime");
            return 0;
        }
        NSError *error = nil;
        Qw35MetalRuntime *rt = (__bridge Qw35MetalRuntime *)runtime;
        if (![rt evalTokens:tokens count:len pos0:pos0 logitsMode:(qw35_logits_mode)logits_mode error:&error]) {
            qw35_set_ns_error(err, err_len, error);
            return 0;
        }
        return 1;
    }
}

int qw35_metal_runtime_argmax(
        void *runtime,
        uint32_t *token,
        float *logit,
        char *err,
        uintptr_t err_len) {
    @autoreleasepool {
        if (!runtime) {
            qw35_set_error(err, err_len, "invalid Qw35 Metal runtime");
            return 0;
        }
        NSError *error = nil;
        Qw35MetalRuntime *rt = (__bridge Qw35MetalRuntime *)runtime;
        if (![rt readArgmaxToken:token logit:logit error:&error]) {
            qw35_set_ns_error(err, err_len, error);
            return 0;
        }
        return 1;
    }
}

int qw35_metal_runtime_copy_logits(
        void *runtime,
        float *dst,
        uintptr_t len,
        char *err,
        uintptr_t err_len) {
    @autoreleasepool {
        if (!runtime) {
            qw35_set_error(err, err_len, "invalid Qw35 Metal runtime");
            return 0;
        }
        NSError *error = nil;
        Qw35MetalRuntime *rt = (__bridge Qw35MetalRuntime *)runtime;
        if (![rt copyLogits:dst len:len error:&error]) {
            qw35_set_ns_error(err, err_len, error);
            return 0;
        }
        return 1;
    }
}
