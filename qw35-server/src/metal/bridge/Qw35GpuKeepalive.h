// Qw35GpuKeepalive.h – process-wide GPU keep-alive for single-token decode.
//
// Decode has a poor GPU duty cycle: -readArgmaxToken:/-copyLogits: drain the
// queue every token, then the host samples, runs the text/tool filters and
// re-encodes ~324 dispatches before the GPU restarts. Those gaps let the SoC
// power manager drop the GPU/fabric performance state, which lowers the
// achievable memory bandwidth — the thing decode is actually bound on. A
// single ALU-only threadgroup running on a separate command queue keeps the
// GPU out of that low state without consuming memory bandwidth.
//
// The keep-alive is process-wide, not per-runtime: a process may hold several
// Qw35MetalRuntime instances (main, scratch and plan sessions all share one
// tensor store), and they share one GPU. One parasite threadgroup total is the
// intent; N of them would be N times the heat for no extra effect.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface Qw35GpuKeepalive : NSObject

/// Start the keep-alive thread the first time it is called; a no-op afterwards
/// (the device and library of the first caller win — every runtime in the
/// process targets the same default device). Never fails hard: on any error the
/// keep-alive stays off and decode runs exactly as it did before.
+ (void)startWithDevice:(id<MTLDevice>)device library:(id<MTLLibrary>)library;

/// Mark a single-token decode step. Holds the keep-alive on for
/// QW35_KEEPALIVE_HOLD_MS past this call, so it spans the intra-token host gap
/// and stops on its own shortly after a generation ends.
+ (void)noteDecodeStep;

/// Drop the hold immediately. Called on entry to a prefill batch: prefill
/// already saturates the GPU, so there the keep-alive is pure parasite.
+ (void)suspend;

@end
