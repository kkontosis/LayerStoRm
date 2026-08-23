// Mini decode FFN chain with llama.cpp/ggml's CUDA kernels — the exact same
// GEMV chain (same real GLM-5.2 blk.10 packed weights, same input) as
// ours_chain.py, evaluated through ggml's CUDA backend (mmvq kernels + ggml's
// own CUDA-graph capture, which engages automatically on repeated identical
// graphs; disable with GGML_CUDA_DISABLE_GRAPHS=1).
//
//   o_proj(Q8_0 6144x8192) -> +res
//   gate/up(Q4_K 2048x6144) -> silu*mul -> down(Q5_K 6144x2048)
//   sh_gate/sh_up(Q8_0)     -> silu*mul -> sh_down(Q8_0)
//   out = r + d + sd
//
// Build: see build_ggml_chain.sh. Run: CUDA_VISIBLE_DEVICES=<d> ./ggml_chain
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

static std::vector<uint8_t> slurp(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", p.c_str()); exit(1); }
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    const std::string dir = (argc > 1 ? argv[1] : "weights");
    const int ITERS = 300, WARMUP = 30;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "cuda backend init failed\n"); return 1; }

    // ── persistent tensors (weights + inputs) ────────────────────────────
    struct WSpec { const char* name; ggml_type t; int64_t k, n; };
    const WSpec specs[] = {
        {"o_proj",  GGML_TYPE_Q8_0, 8192, 6144},
        {"gate",    GGML_TYPE_Q4_K, 6144, 2048},
        {"up",      GGML_TYPE_Q4_K, 6144, 2048},
        {"down",    GGML_TYPE_Q5_K, 2048, 6144},
        {"sh_gate", GGML_TYPE_Q8_0, 6144, 2048},
        {"sh_up",   GGML_TYPE_Q8_0, 6144, 2048},
        {"sh_down", GGML_TYPE_Q8_0, 2048, 6144},
    };
    ggml_init_params wip = { ggml_tensor_overhead() * 16, nullptr, true };
    ggml_context* wctx = ggml_init(wip);
    ggml_tensor* wt[7];
    for (int i = 0; i < 7; i++) {
        wt[i] = ggml_new_tensor_2d(wctx, specs[i].t, specs[i].k, specs[i].n);
        ggml_set_name(wt[i], specs[i].name);
    }
    ggml_tensor* x   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 8192);
    ggml_tensor* res = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 6144);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (!wbuf) { fprintf(stderr, "weight alloc failed\n"); return 1; }
    for (int i = 0; i < 7; i++) {
        auto v = slurp(dir + "/" + specs[i].name + ".bin");
        if (v.size() != ggml_nbytes(wt[i])) {
            fprintf(stderr, "%s: %zu != %zu\n", specs[i].name, v.size(), ggml_nbytes(wt[i]));
            return 1;
        }
        ggml_backend_tensor_set(wt[i], v.data(), 0, v.size());
    }
    { auto v = slurp(dir + "/x_f32.bin");   ggml_backend_tensor_set(x,   v.data(), 0, v.size()); }
    { auto v = slurp(dir + "/res_f32.bin"); ggml_backend_tensor_set(res, v.data(), 0, v.size()); }

    // ── compute graph ────────────────────────────────────────────────────
    ggml_init_params gip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context* gctx = ggml_init(gip);
    ggml_tensor* o  = ggml_mul_mat(gctx, wt[0], x);                       // [6144]
    ggml_tensor* r  = ggml_add(gctx, o, res);
    ggml_tensor* g  = ggml_mul_mat(gctx, wt[1], r);                       // [2048]
    ggml_tensor* u  = ggml_mul_mat(gctx, wt[2], r);
    ggml_tensor* a  = ggml_mul(gctx, ggml_silu(gctx, g), u);
    ggml_tensor* d  = ggml_mul_mat(gctx, wt[3], a);                       // [6144]
    ggml_tensor* sg = ggml_mul_mat(gctx, wt[4], r);
    ggml_tensor* su = ggml_mul_mat(gctx, wt[5], r);
    ggml_tensor* sa = ggml_mul(gctx, ggml_silu(gctx, sg), su);
    ggml_tensor* sd = ggml_mul_mat(gctx, wt[6], sa);                      // [6144]
    ggml_tensor* out = ggml_add(gctx, ggml_add(gctx, r, d), sd);
    ggml_cgraph* gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "galloc failed\n"); return 1; }

    // warmup (also lets ggml's CUDA-graph capture engage)
    for (int i = 0; i < WARMUP; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);

    // output for the numerical comparison
    std::vector<float> ov(6144);
    ggml_backend_tensor_get(out, ov.data(), 0, 6144 * sizeof(float));
    { FILE* f = fopen("out_ggml.f32.bin", "wb"); fwrite(ov.data(), 4, 6144, f); fclose(f); }

    // timed: N evals bulk / wall (each compute is stream-synchronized by ggml)
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ITERS; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;
    printf("ggml chain: %.1f us/iter (%d iters, cuda-graphs=%s)\n", us, ITERS,
           getenv("GGML_CUDA_DISABLE_GRAPHS") ? "OFF" : "auto");

    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(backend);
    return 0;
}
