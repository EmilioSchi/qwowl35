// Qw35FFI.h – C FFI surface exposed to the Rust side.
//
// Every function reports failure by returning 0/NULL and writing a
// NUL-terminated message into the caller-provided `err` buffer.

#ifndef QW35_FFI_H
#define QW35_FFI_H

#include <stddef.h>
#include <stdint.h>

#include "Qw35MetalTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Returns 1 if a default Metal device is available.
int qw35_metal_has_device(void);

/// Touch the mmap-backed model through GPU buffer views to fault pages in.
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
        uintptr_t err_len);

/// Create a Qw35MetalRuntime.  Returns an opaque retained handle, or NULL.
/// The unified .gguf carries its GF4 FFN as type-id 100 tensors in `tensors`.
void *qw35_metal_runtime_create(
        const void *model_map,
        uint64_t model_size,
        const qw35_metal_tensor_desc *tensors,
        uintptr_t tensor_count,
        const qw35_metal_hparams *hparams,
        uint32_t ctx_size,
        uint32_t vocab_size,
        uint32_t prefill_chunk,
        uint32_t kv_cache_type, // 0 = f16, 1 = q8_0
        const uint8_t *metallib,
        uintptr_t metallib_len,
        char *err,
        uintptr_t err_len);

void qw35_metal_runtime_destroy(void *runtime);

/// Wait for in-flight GPU work, then zero the conv/SSM recurrent state.
int qw35_metal_runtime_reset(void *runtime, char *err, uintptr_t err_len);

/// Copy the conv/SSM recurrent state into an internal checkpoint slot.
int qw35_metal_runtime_state_checkpoint_save(void *runtime, char *err, uintptr_t err_len);

/// Restore the conv/SSM recurrent state from the internal checkpoint slot.
int qw35_metal_runtime_state_checkpoint_restore(void *runtime, char *err, uintptr_t err_len);

/// Wait for all in-flight GPU work to complete.
int qw35_metal_runtime_sync(void *runtime, char *err, uintptr_t err_len);

/// Set the decode-time sliding-window attention sink (first N positions always
/// attended). The engine sets this per request to max(--attn-sink, preamble),
/// where preamble = system block + first user turn, so the window never evicts
/// the tool-call format or the task. No-op when --attn-window is 0.
void qw35_metal_runtime_set_attn_sink(void *runtime, int32_t sink);

/// Evaluate one token. Commits asynchronously: GPU errors surface at the
/// next argmax/copy_logits/reset call. logits_mode is a qw35_logits_mode.
int qw35_metal_runtime_eval_token(
        void *runtime,
        uint32_t token,
        uint32_t pos,
        int logits_mode,
        char *err,
        uintptr_t err_len);

int qw35_metal_runtime_eval_tokens(
        void *runtime,
        const uint32_t *tokens,
        uintptr_t len,
        uint32_t pos0,
        int logits_mode,
        char *err,
        uintptr_t err_len);

int qw35_metal_runtime_argmax(
        void *runtime,
        uint32_t *token,
        float *logit,
        char *err,
        uintptr_t err_len);

int qw35_metal_runtime_copy_logits(
        void *runtime,
        float *dst,
        uintptr_t len,
        char *err,
        uintptr_t err_len);

#ifdef __cplusplus
}
#endif

#endif // QW35_FFI_H
