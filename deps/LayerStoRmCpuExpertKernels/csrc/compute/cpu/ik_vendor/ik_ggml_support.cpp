// Minimal ggml runtime support for the vendored ik kernels (C-6 CPU path).
//
// The verbatim ik GEMM/quant TUs reference a few ggml.c globals/functions that we
// do NOT vendor the whole of ggml.c for. We provide EXACTLY those here:
//   * ggml_table_f32_f16[1<<16] — the fp16->fp32 lookup table (used by the
//     non-F16C GGML_FP16_TO_FP32 fallback path). Initialized once on first use,
//     identical to ggml.c's ggml_init() loop (GGML_COMPUTE_FP16_TO_FP32 per u16).
//   * ggml_abort(file,line,fmt,...) — fatal abort (GGML_ASSERT/GGML_ABORT).
//   * ggml_bf16_to_fp32(bf16)      — the trivial left-shift bf16 decode.
// These are the documented "stubs" replacing the rest of ggml.c (TECH_DEBT).
//
// CPU-only TU — no CUDA (INV-GPU-1).

#include "iqk_common.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

// ── fp16->fp32 lookup table (defined in ggml.c; declared in ggml-impl.h) ─────
float ggml_table_f32_f16[1 << 16];

namespace {
// Populate ggml_table_f32_f16 exactly as ggml.c ggml_init() does. Runs before
// main via a static initializer so the table is valid for all GEMM/dequant use.
struct TableInit {
    TableInit() {
        for (int i = 0; i < (1 << 16); ++i) {
            union { uint16_t u16; ggml_fp16_t fp16; } u;
            u.u16 = static_cast<uint16_t>(i);
            ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
        }
    }
};
const TableInit g_table_init;
}  // namespace

extern "C" void ggml_abort(const char* file, int line, const char* fmt, ...) {
    std::fflush(stdout);
    std::fprintf(stderr, "%s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
    std::abort();
}

// ggml_bf16_to_fp32: verbatim semantics from ggml.c (GGML_BF16_TO_FP32 = <<16).
float ggml_bf16_to_fp32(ggml_bf16_t x) {
    return GGML_BF16_TO_FP32(x);
}
