// Qw35GpuKeepalive.m – see Qw35GpuKeepalive.h for what this is for.

#import "Qw35GpuKeepalive.h"

#include <pthread.h>
#include <pthread/qos.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// One threadgroup of 256 threads. More threadgroups raise apparent utilization
// without adding pressure on the memory bus, but they also add heat, and the
// power state is already held by one. Keep it at the minimum that works.
#define QW35_KEEPALIVE_THREADS 256u
#define QW35_KEEPALIVE_THREADGROUPS 1u

// Iterations of the two-FMA chain per dispatch. Sets how long one keep-alive
// command buffer occupies the GPU, which is the granularity at which the loop
// below can notice that decode has stopped. A few milliseconds: long enough
// that per-dispatch encode/commit overhead is noise, short enough that the
// keep-alive stops promptly at the end of a response.
#define QW35_KEEPALIVE_ITERS 1200000u

// How long a single -noteDecodeStep holds the keep-alive on. Must comfortably
// exceed one decode step (tens of ms) so a generation never flickers the hold
// off between tokens, and stay short enough that an idle server goes quiet
// almost immediately after the last token.
#define QW35_KEEPALIVE_HOLD_MS 400ull

// Poll interval while the hold is expired. Only costs a wakeup; no GPU work is
// submitted in this state.
#define QW35_KEEPALIVE_IDLE_POLL_US 2000u

// Monotonic ns of the last -noteDecodeStep, or 0 when suspended. Read by the
// keep-alive thread, written by whichever thread is driving decode.
static _Atomic(uint64_t) g_keepalive_active_ns;

static id<MTLCommandQueue> g_keepalive_queue;
static id<MTLComputePipelineState> g_keepalive_pipeline;
static id<MTLBuffer> g_keepalive_buffer;
static pthread_t g_keepalive_thread;

static uint64_t qw35_keepalive_now_ns(void) {
    return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
}

static void *qw35_keepalive_main(void *arg) {
    (void)arg;
    pthread_setname_np("qw35-gpu-keepalive");
    // The thread spends almost all of its time blocked in -waitUntilCompleted;
    // it only needs enough priority to re-submit promptly, and must not compete
    // with the thread encoding decode command buffers.
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);

    const uint32_t iters = QW35_KEEPALIVE_ITERS;
    for (;;) {
        const uint64_t active = atomic_load_explicit(&g_keepalive_active_ns,
                                                     memory_order_relaxed);
        const uint64_t now = qw35_keepalive_now_ns();
        if (active == 0 || now - active > QW35_KEEPALIVE_HOLD_MS * 1000000ull) {
            usleep(QW35_KEEPALIVE_IDLE_POLL_US);
            continue;
        }
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g_keepalive_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:g_keepalive_pipeline];
            [enc setBuffer:g_keepalive_buffer offset:0 atIndex:0];
            [enc setBytes:&iters length:sizeof(iters) atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake(QW35_KEEPALIVE_THREADGROUPS, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(QW35_KEEPALIVE_THREADS, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
    return NULL;
}

@implementation Qw35GpuKeepalive

+ (void)startWithDevice:(id<MTLDevice>)device library:(id<MTLLibrary>)library {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        if (!device || !library) return;

        id<MTLFunction> function = [library newFunctionWithName:@"qw35_gpu_keepalive"];
        if (!function) {
            fprintf(stderr, "qw35: GPU keep-alive kernel missing; keep-alive off\n");
            return;
        }
        NSError *nsError = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&nsError];
        if (!pipeline) {
            fprintf(stderr, "qw35: GPU keep-alive pipeline failed (%s); keep-alive off\n",
                    nsError.localizedDescription.UTF8String);
            return;
        }
        // A dedicated queue: the keep-alive must overlap decode's command
        // buffers, not queue behind them.
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> buffer =
            [device newBufferWithLength:(NSUInteger)QW35_KEEPALIVE_THREADS * sizeof(float)
                                options:MTLResourceStorageModeShared];
        if (!queue || !buffer) {
            fprintf(stderr, "qw35: GPU keep-alive allocation failed; keep-alive off\n");
            return;
        }
        buffer.label = @"gpu_keepalive";
        memset([buffer contents], 0, [buffer length]);

        g_keepalive_pipeline = pipeline;
        g_keepalive_queue = queue;
        g_keepalive_buffer = buffer;

        if (pthread_create(&g_keepalive_thread, NULL, qw35_keepalive_main, NULL) != 0) {
            fprintf(stderr, "qw35: GPU keep-alive thread failed to start; keep-alive off\n");
            g_keepalive_pipeline = nil;
            g_keepalive_queue = nil;
            g_keepalive_buffer = nil;
            return;
        }
        pthread_detach(g_keepalive_thread);
    });
}

+ (void)noteDecodeStep {
    if (!g_keepalive_queue) return;
    atomic_store_explicit(&g_keepalive_active_ns, qw35_keepalive_now_ns(),
                          memory_order_relaxed);
}

+ (void)suspend {
    if (!g_keepalive_queue) return;
    atomic_store_explicit(&g_keepalive_active_ns, 0, memory_order_relaxed);
}

@end
