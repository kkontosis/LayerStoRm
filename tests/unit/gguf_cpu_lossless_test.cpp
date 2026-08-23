// GGUF Stage 0: bit-compatible CPU Q4_K x Q8_1 expert GEMM (GPU-mmvq parity).
//
// The production TARGET is GGUF Q4_K; the GPU routed-expert GEMM quantizes
// activations to Q8_1 (block 32, d=amax/127 full-float for the quant, applied as
// FP16) and does the Q4_K int dot. The default CPU ik path uses Q8_2_X4 (d
// re-rounded to BF16) → different from the GPU → would drift the target's MoE
// hidden and could flip the TQ golden with CPU offload. cpu_gguf_grouped_gemm_
// q4k_lossless reproduces the GPU Q8_1 quant + mmvq int dot. Validated here vs
// independent references — CPU-only, no GPU.

#include "compute/cpu/gguf_lossless.h"
#include "compute/cpu/numa_thread_pool.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace lc = layerstorm::compute::cpu;

namespace {
float bf2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; std::memcpy(&f,&u,4); return f; }
uint16_t f2bf(float f){ uint32_t u; std::memcpy(&u,&f,4); u+=0x7fff+((u>>16)&1); return (uint16_t)(u>>16); }
// independent f32->f16->f32 RNE (bit manipulation) — cross-checks the kernel's _Float16.
float f16rt(float f){
    uint32_t x; std::memcpy(&x,&f,4);
    uint32_t sign=(x>>16)&0x8000; int32_t exp=((x>>23)&0xff)-127+15; uint32_t man=x&0x7fffff;
    if(exp<=0){ if(exp<-10) return sign?-0.f:0.f; man|=0x800000; uint32_t sh=14-exp; uint32_t h=man>>sh;
        uint32_t rem=man&((1u<<sh)-1), half=1u<<(sh-1); if(rem>half||(rem==half&&(h&1))) h++;
        uint16_t hf=sign|h; return (float)*reinterpret_cast<_Float16*>(&hf); }
    if(exp>=31){ uint16_t hf=sign|0x7c00; return (float)*reinterpret_cast<_Float16*>(&hf); }
    uint32_t h=(exp<<10)|(man>>13); uint32_t rem=man&0x1fff; if(rem>0x1000||(rem==0x1000&&(h&1))) h++;
    uint16_t hf=sign|h; return (float)*reinterpret_cast<_Float16*>(&hf);
}
void get_scale_min_k4(int j,const uint8_t*q,uint8_t&d,uint8_t&m){
    if(j<4){d=q[j]&63;m=q[j+4]&63;} else {d=(q[j+4]&0xF)|((q[j-4]>>6)<<4);m=(q[j+4]>>4)|((q[j]>>6)<<4);}
}
float h2f(uint16_t h){ return (float)*reinterpret_cast<_Float16*>(&h); }
void dequant_q4k_logical(const uint8_t* blk, float* y){  // ggml logical order
    uint16_t dh,dmh; std::memcpy(&dh,blk,2); std::memcpy(&dmh,blk+2,2);
    float d=h2f(dh), dmin=h2f(dmh);
    const uint8_t* sc=blk+4; const uint8_t* q=blk+16; int is=0;
    for(int j=0;j<4;j++){ uint8_t s1,m1,s2,m2; get_scale_min_k4(is,sc,s1,m1); get_scale_min_k4(is+1,sc,s2,m2);
        float d1=d*s1,mn1=dmin*m1,d2=d*s2,mn2=dmin*m2;
        for(int l=0;l<32;l++) y[j*64+l]=d1*(q[j*32+l]&0xF)-mn1;
        for(int l=0;l<32;l++) y[j*64+32+l]=d2*(q[j*32+l]>>4)-mn2; is+=2; }
}
void dequant_q5k_logical(const uint8_t* blk, float* y){  // ggml Q5_K (176B)
    uint16_t dh,dmh; std::memcpy(&dh,blk,2); std::memcpy(&dmh,blk+2,2);
    float d=h2f(dh), dmin=h2f(dmh);
    const uint8_t* sc=blk+4; const uint8_t* qh=blk+16; const uint8_t* qs=blk+48;
    int is=0; uint8_t u1=1,u2=2;
    for(int j=0;j<4;j++){ uint8_t s1,m1,s2,m2; get_scale_min_k4(is,sc,s1,m1); get_scale_min_k4(is+1,sc,s2,m2);
        float d1=d*s1,mn1=dmin*m1,d2=d*s2,mn2=dmin*m2;
        for(int l=0;l<32;l++) y[j*64+l]   =d1*((qs[j*32+l]&0xF)+((qh[l]&u1)?16:0))-mn1;
        for(int l=0;l<32;l++) y[j*64+32+l]=d2*((qs[j*32+l]>>4) +((qh[l]&u2)?16:0))-mn2;
        is+=2; u1<<=2; u2<<=2; }
}
void dequant_q6k_logical(const uint8_t* blk, float* y){  // ggml Q6_K (210B)
    const uint8_t* ql=blk; const uint8_t* qh=blk+128; const int8_t* sc=(const int8_t*)(blk+192);
    uint16_t dh; std::memcpy(&dh,blk+208,2); float d=h2f(dh);
    for(int n=0;n<2;n++){ const uint8_t* Ql=ql+n*64; const uint8_t* Qh=qh+n*32; const int8_t* S=sc+n*8; float* Y=y+n*128;
        for(int l=0;l<32;l++){ int is=l/16;
            int q1=((Ql[l]&0xF)|((Qh[l]&3)<<4))-32, q2=((Ql[l+32]&0xF)|(((Qh[l]>>2)&3)<<4))-32;
            int q3=((Ql[l]>>4)|(((Qh[l]>>4)&3)<<4))-32, q4=((Ql[l+32]>>4)|(((Qh[l]>>6)&3)<<4))-32;
            Y[l]=d*S[is+0]*q1; Y[l+32]=d*S[is+2]*q2; Y[l+64]=d*S[is+4]*q3; Y[l+96]=d*S[is+6]*q4; } }
}

// Build random packed weights + run kq_lossless, compare to an independent logical
// FP64 dequant dot. Returns max_rel. blkbytes/deq select the K-quant family.
double run_kq(lc::KQuantLossless t, int blkbytes, void(*deq)(const uint8_t*,float*), unsigned seed){
    std::mt19937 g(seed);
    const int K=512,N=8,ne=3; int tok[3]={2,1,3}; std::vector<int32_t> off(ne+1,0);
    for(int e=0;e<ne;e++) off[e+1]=off[e]+tok[e]; int total=off[ne];
    int nsb=K/256; size_t wrow=(size_t)nsb*blkbytes;
    std::vector<std::vector<uint8_t>> W(ne); std::vector<const void*> B(ne);
    std::uniform_int_distribution<int> byte(0,255); std::normal_distribution<float> sd(0.03f,0.01f);
    for(int e=0;e<ne;e++){ W[e].resize((size_t)N*wrow);
        for(int n=0;n<N;n++) for(int sb=0;sb<nsb;sb++){ uint8_t* blk=W[e].data()+(size_t)n*wrow+(size_t)sb*blkbytes;
            for(int i=0;i<blkbytes;i++) blk[i]=(uint8_t)byte(g);
            if(t==lc::KQuantLossless::Q6_K){ _Float16 d=(_Float16)(std::fabs(sd(g))+0.005f); std::memcpy(blk+208,&d,2); }
            else { _Float16 d=(_Float16)(std::fabs(sd(g))+0.005f), dm=(_Float16)(std::fabs(sd(g))*0.5f); std::memcpy(blk,&d,2); std::memcpy(blk+2,&dm,2);} }
        B[e]=W[e].data(); }
    std::vector<uint16_t> A((size_t)total*K); std::normal_distribution<float> nd(0,1.2f); for(auto&x:A)x=f2bf(nd(g));
    std::vector<uint16_t> D((size_t)total*N,0);
    lc::cpu_gguf_grouped_gemm_kq_lossless(t,D.data(),A.data(),B.data(),off.data(),ne,N,K,nullptr);
    double maxrel=0;
    for(int e=0;e<ne;e++) for(int m=off[e];m<off[e+1];m++){
        int nb=K/32; std::vector<float> ad(nb); std::vector<int8_t> aq(K);
        lc::q8_1_quantize_row(A.data()+(size_t)m*K,K,ad.data(),aq.data());
        std::vector<float> alog(K); for(int k=0;k<K;k++) alog[k]=ad[k/32]*(float)aq[k];
        for(int n=0;n<N;n++){ const uint8_t* wr=W[e].data()+(size_t)n*wrow;
            std::vector<float> wl(K); for(int sb=0;sb<nsb;sb++) deq(wr+(size_t)sb*blkbytes, wl.data()+sb*256);
            double acc=0; for(int k=0;k<K;k++) acc+=(double)wl[k]*(double)alog[k];
            float ref=(float)acc, got=bf2f(D[(size_t)m*N+n]);
            maxrel=std::max(maxrel,(double)(std::fabs(got-ref)/std::max(1.f,std::fabs(ref)))); } }
    return maxrel;
}
}  // namespace

// (T1) Q8_1 quantizer bit-matches an independent RNE Q8_1 implementation.
TEST(GgufCpuLosslessQ4K, Q8_1QuantizerMatchesRneReference) {
    std::mt19937 g(7); const int K=2048;
    std::vector<uint16_t> a(K); std::normal_distribution<float> nd(0,2.f);
    for(auto&x:a) x=f2bf(nd(g));
    const int nb=K/32; std::vector<float> kd(nb); std::vector<int8_t> kq(K);
    lc::q8_1_quantize_row(a.data(),K,kd.data(),kq.data());
    for(int b=0;b<nb;b++){
        float amax=0; for(int i=0;i<32;i++) amax=std::max(amax,std::fabs(bf2f(a[b*32+i])));
        const float d=amax/127.f; ASSERT_FLOAT_EQ(kd[b], f16rt(d)) << "block "<<b;
        for(int i=0;i<32;i++){ int v= d>0? (int)std::lrint(bf2f(a[b*32+i])/d):0; v=std::max(-127,std::min(127,v));
            ASSERT_EQ(kq[b*32+i], (int8_t)v) << "block "<<b<<" i "<<i; }
    }
}

// (T2) Q4_K x Q8_1 GEMM matches an INDEPENDENT logical-order FP64 dequant dot.
TEST(GgufCpuLosslessQ4K, GroupedGemmMatchesLogicalDequantReference) {
    std::mt19937 g(11);
    const int K=512, N=8, num_experts=3; int tok[3]={2,1,3};
    std::vector<int32_t> off(num_experts+1,0); for(int e=0;e<num_experts;e++) off[e+1]=off[e]+tok[e];
    const int total=off[num_experts]; const int nsb=K/256; const size_t wrow=(size_t)nsb*144;
    std::vector<std::vector<uint8_t>> W(num_experts); std::vector<const void*> Bptrs(num_experts);
    std::uniform_int_distribution<int> byte(0,255); std::normal_distribution<float> sd(0.05f,0.02f);
    for(int e=0;e<num_experts;e++){ W[e].resize((size_t)N*wrow);
        for(int n=0;n<N;n++) for(int sb=0;sb<nsb;sb++){
            uint8_t* blk=W[e].data()+(size_t)n*wrow+(size_t)sb*144;
            float d=std::fabs(sd(g))+0.001f, dmin=std::fabs(sd(g))*0.5f;
            _Float16 dh=(_Float16)d, dmh=(_Float16)dmin; std::memcpy(blk,&dh,2); std::memcpy(blk+2,&dmh,2);
            for(int i=4;i<144;i++) blk[i]=(uint8_t)byte(g); }
        Bptrs[e]=W[e].data(); }
    std::vector<uint16_t> A((size_t)total*K); std::normal_distribution<float> nd(0,1.5f);
    for(auto&x:A) x=f2bf(nd(g));
    std::vector<uint16_t> D((size_t)total*N,0);
    lc::cpu_gguf_grouped_gemm_q4k_lossless(D.data(),A.data(),Bptrs.data(),off.data(),num_experts,N,K,nullptr);

    double maxrel=0; int bad=0;
    for(int e=0;e<num_experts;e++) for(int m=off[e];m<off[e+1];m++){
        int nb=K/32; std::vector<float> ad(nb); std::vector<int8_t> aq(K);
        lc::q8_1_quantize_row(A.data()+(size_t)m*K,K,ad.data(),aq.data());
        std::vector<float> alog(K); for(int k=0;k<K;k++) alog[k]=ad[k/32]*(float)aq[k];
        for(int n=0;n<N;n++){ const uint8_t* wr=W[e].data()+(size_t)n*wrow;
            std::vector<float> wlog(K); for(int sb=0;sb<nsb;sb++) dequant_q4k_logical(wr+(size_t)sb*144, wlog.data()+sb*256);
            double acc=0; for(int k=0;k<K;k++) acc+=(double)wlog[k]*(double)alog[k];
            float ref=(float)acc, got=bf2f(D[(size_t)m*N+n]);
            maxrel=std::max(maxrel,(double)(std::fabs(got-ref)/std::max(1.f,std::fabs(ref)))); }
    }
    EXPECT_LT(maxrel, 0.02) << "Q4_KxQ8_1 int dot must equal the logical dequant dot to BF16 round-off";
}

// Whole-expert losslessness: down projection is Q5_K/Q6_K in Q4_K_XL. Both must
// match their independent ggml logical-dequant dot to BF16 round-off.
TEST(GgufCpuLosslessQ5K, GroupedGemmMatchesLogicalDequantReference) {
    EXPECT_LT(run_kq(lc::KQuantLossless::Q5_K, 176, dequant_q5k_logical, 5), 0.02);
}
TEST(GgufCpuLosslessQ6K, GroupedGemmMatchesLogicalDequantReference) {
    EXPECT_LT(run_kq(lc::KQuantLossless::Q6_K, 210, dequant_q6k_logical, 6), 0.02);
}

// ── VNNI vs scalar-reference bit-identity + benchmark ────────────────────────
// The production kernel (cpu_gguf_grouped_gemm_kq_lossless) computes the per-group
// integer dot with AVX512-VNNI; the scalar reference uses a plain int loop under
// the IDENTICAL GPU 32-lane warp-tree FP32 reduction. The integer dot is
// order-free, so the two MUST be bitwise-identical (0/N differ). Returns
// (num_differing, total) and, if `bench`, prints ms for both at these dims.
namespace {
struct BitMatch { int diff; int total; double vnni_ms; double scalar_ms; };
BitMatch run_vnni_vs_scalar(lc::KQuantLossless t, int blkbytes,
                            int N, int K, const int* tok, int ne, unsigned seed,
                            bool bench) {
    std::mt19937 g(seed);
    std::vector<int32_t> off(ne + 1, 0);
    for (int e = 0; e < ne; e++) off[e + 1] = off[e] + tok[e];
    const int total = off[ne];
    const int nsb = K / 256; const size_t wrow = (size_t)nsb * blkbytes;
    std::vector<std::vector<uint8_t>> W(ne); std::vector<const void*> B(ne);
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> sd(0.03f, 0.01f);
    for (int e = 0; e < ne; e++) {
        W[e].resize((size_t)N * wrow);
        for (int n = 0; n < N; n++) for (int sb = 0; sb < nsb; sb++) {
            uint8_t* blk = W[e].data() + (size_t)n * wrow + (size_t)sb * blkbytes;
            for (int i = 0; i < blkbytes; i++) blk[i] = (uint8_t)byte(g);
            if (t == lc::KQuantLossless::Q6_K) {
                _Float16 d = (_Float16)(std::fabs(sd(g)) + 0.005f); std::memcpy(blk + 208, &d, 2);
            } else {
                _Float16 d = (_Float16)(std::fabs(sd(g)) + 0.005f), dm = (_Float16)(std::fabs(sd(g)) * 0.5f);
                std::memcpy(blk, &d, 2); std::memcpy(blk + 2, &dm, 2);
            }
        }
        B[e] = W[e].data();
    }
    std::vector<uint16_t> A((size_t)total * K);
    std::normal_distribution<float> nd(0, 1.2f); for (auto& x : A) x = f2bf(nd(g));
    std::vector<uint16_t> Dv((size_t)total * N, 0), Ds((size_t)total * N, 0);

    auto t0 = std::chrono::steady_clock::now();
    lc::cpu_gguf_grouped_gemm_kq_lossless(t, Dv.data(), A.data(), B.data(), off.data(), ne, N, K, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    lc::cpu_gguf_grouped_gemm_kq_lossless_scalar_ref(t, Ds.data(), A.data(), B.data(), off.data(), ne, N, K, nullptr);
    auto t2 = std::chrono::steady_clock::now();

    int diff = 0; for (size_t i = 0; i < Dv.size(); i++) if (Dv[i] != Ds[i]) diff++;
    BitMatch bm{diff, (int)Dv.size(),
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - t1).count()};
    if (bench)
        std::printf("[bench] type=%d N=%d K=%d tokens=%d  VNNI %.3f ms  scalar %.3f ms  (%.1fx)\n",
                    (int)t, N, K, total, bm.vnni_ms, bm.scalar_ms,
                    bm.scalar_ms / std::max(1e-9, bm.vnni_ms));
    return bm;
}
}  // namespace

TEST(GgufCpuLosslessVnni, Q4KBitIdenticalToScalar) {
    int tok[3] = {2, 1, 3};
    auto bm = run_vnni_vs_scalar(lc::KQuantLossless::Q4_K, 144, 96, 512, tok, 3, 21, false);
    EXPECT_EQ(bm.diff, 0) << bm.diff << "/" << bm.total << " outputs differ (VNNI vs scalar)";
}
TEST(GgufCpuLosslessVnni, Q5KBitIdenticalToScalar) {
    int tok[3] = {2, 1, 3};
    auto bm = run_vnni_vs_scalar(lc::KQuantLossless::Q5_K, 176, 96, 512, tok, 3, 22, false);
    EXPECT_EQ(bm.diff, 0) << bm.diff << "/" << bm.total << " outputs differ (VNNI vs scalar)";
}
TEST(GgufCpuLosslessVnni, Q6KBitIdenticalToScalar) {
    int tok[3] = {2, 1, 3};
    auto bm = run_vnni_vs_scalar(lc::KQuantLossless::Q6_K, 210, 96, 512, tok, 3, 23, false);
    EXPECT_EQ(bm.diff, 0) << bm.diff << "/" << bm.total << " outputs differ (VNNI vs scalar)";
}

// Realistic keeper52 expert dims (single thread => per-expert ms): gate/up
// [N=2048,K=6144] Q4_K, down [N=6144,K=2048] Q6_K, one decode token (M_e=1).
// Reports VNNI vs scalar wall so the speedup and per-expert cost are on record.
TEST(GgufCpuLosslessVnni, BenchRealisticDims) {
    int tok1[1] = {1};
    auto gate = run_vnni_vs_scalar(lc::KQuantLossless::Q4_K, 144, 2048, 6144, tok1, 1, 31, true);
    auto up   = run_vnni_vs_scalar(lc::KQuantLossless::Q4_K, 144, 2048, 6144, tok1, 1, 32, true);
    auto down = run_vnni_vs_scalar(lc::KQuantLossless::Q6_K, 210, 6144, 2048, tok1, 1, 33, true);
    std::printf("[bench] single-thread per-expert (gate+up+down) VNNI = %.3f ms  scalar = %.3f ms\n",
                gate.vnni_ms + up.vnni_ms + down.vnni_ms,
                gate.scalar_ms + up.scalar_ms + down.scalar_ms);
    EXPECT_EQ(gate.diff + up.diff + down.diff, 0);
}

// THREADED per-expert wall on a real NUMA pool (node 3, all physical cores) — the
// number that decides the graph-hoist window feasibility (host-FFN/layer <=
// ~2.3ms at B=1, ~2 CPU experts/layer at 25% offload). Runs one decode token
// (M_e=1) through gate+up+down with the production VNNI kernel.
namespace {
double time_proj_pooled(lc::KQuantLossless t, int blkbytes, int N, int K,
                        lc::NumaThreadPool& pool, unsigned seed) {
    std::mt19937 g(seed);
    const int nsb = K / 256; const size_t wrow = (size_t)nsb * blkbytes;
    std::vector<uint8_t> W((size_t)N * wrow);
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> sd(0.03f, 0.01f);
    for (int n = 0; n < N; n++) for (int sb = 0; sb < nsb; sb++) {
        uint8_t* blk = W.data() + (size_t)n * wrow + (size_t)sb * blkbytes;
        for (int i = 0; i < blkbytes; i++) blk[i] = (uint8_t)byte(g);
        if (t == lc::KQuantLossless::Q6_K) { _Float16 d = (_Float16)(std::fabs(sd(g)) + 0.005f); std::memcpy(blk + 208, &d, 2); }
        else { _Float16 d = (_Float16)(std::fabs(sd(g)) + 0.005f), dm = (_Float16)(std::fabs(sd(g)) * 0.5f); std::memcpy(blk, &d, 2); std::memcpy(blk + 2, &dm, 2); }
    }
    const void* B[1] = {W.data()}; int32_t off[2] = {0, 1};
    std::vector<uint16_t> A((size_t)K); std::normal_distribution<float> nd(0, 1.2f);
    for (auto& x : A) x = f2bf(nd(g));
    std::vector<uint16_t> D((size_t)N, 0);
    // warm + timed median of a few
    lc::cpu_gguf_grouped_gemm_kq_lossless(t, D.data(), A.data(), B, off, 1, N, K, &pool);
    std::vector<double> ms;
    for (int r = 0; r < 7; r++) {
        auto t0 = std::chrono::steady_clock::now();
        lc::cpu_gguf_grouped_gemm_kq_lossless(t, D.data(), A.data(), B, off, 1, N, K, &pool);
        auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}
}  // namespace

TEST(GgufCpuLosslessVnni, BenchThreadedNode3) {
    lc::NumaThreadPool pool(/*numa_node=*/3, /*max_threads=*/0);
    std::printf("[bench] NumaThreadPool node=3 threads=%d\n", pool.num_threads());
    const double gate = time_proj_pooled(lc::KQuantLossless::Q4_K, 144, 2048, 6144, pool, 41);
    const double up   = time_proj_pooled(lc::KQuantLossless::Q4_K, 144, 2048, 6144, pool, 42);
    const double down = time_proj_pooled(lc::KQuantLossless::Q6_K, 210, 6144, 2048, pool, 43);
    std::printf("[bench] THREADED per-expert  gate=%.3f up=%.3f down=%.3f  total=%.3f ms/expert\n",
                gate, up, down, gate + up + down);
    std::printf("[bench] => ~2 CPU experts/layer at 25%% offload = %.3f ms/layer (window ~2.3 ms)\n",
                2.0 * (gate + up + down));
    SUCCEED();
}
