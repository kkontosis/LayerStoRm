// Standalone validation of the bit-compatible CPU Q4_K x Q8_1 kernel (GGUF Stage 0).
// Links ONLY the CPU expert kernels dep. No CUDA.
// Reference dequant logic derived from ggml/llama.cpp (MIT License,
// Copyright (c) 2023-2026 The ggml authors — see THIRD_PARTY_NOTICES.md).
//  (T1) q8_1_quantize_row vs an independent RNE Q8_1 quantizer (0/N differ on qs + d).
//  (T2) the Q4_K x Q8_1 grouped GEMM vs an INDEPENDENT logical-order FP64 dequant dot
//       (different code path: ggml logical dequant, not group-integer) — within tol.
#include "compute/cpu/gguf_lossless.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
namespace lc = layerstorm::compute::cpu;

static float bf2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }
static uint16_t f2bf(float f){ uint32_t u; memcpy(&u,&f,4); u+=0x7fff+((u>>16)&1); return (uint16_t)(u>>16); }
// independent f32->f16->f32 RNE (bit manipulation), for cross-checking the kernel's _Float16.
static float f16rt(float f){
    uint32_t x; memcpy(&x,&f,4);
    uint32_t sign=(x>>16)&0x8000; int32_t exp=((x>>23)&0xff)-127+15; uint32_t man=x&0x7fffff;
    if(exp<=0){ if(exp<-10) return sign?-0.f:0.f;
        man|=0x800000; uint32_t shift=14-exp; uint32_t h=man>>shift;
        uint32_t rem=man&((1u<<shift)-1), half=1u<<(shift-1);
        if(rem>half||(rem==half&&(h&1))) h++;
        uint16_t hf=sign|h; return (float)*reinterpret_cast<_Float16*>(&hf); }
    if(exp>=31){ uint16_t hf=sign|0x7c00; return (float)*reinterpret_cast<_Float16*>(&hf); }
    uint32_t h=(exp<<10)|(man>>13); uint32_t rem=man&0x1fff;
    if(rem>0x1000||(rem==0x1000&&(h&1))) h++;
    uint16_t hf=sign|h; return (float)*reinterpret_cast<_Float16*>(&hf);
}
static void get_scale_min_k4(int j,const uint8_t*q,uint8_t&d,uint8_t&m){
    if(j<4){d=q[j]&63;m=q[j+4]&63;} else {d=(q[j+4]&0xF)|((q[j-4]>>6)<<4);m=(q[j+4]>>4)|((q[j]>>6)<<4);}
}
// ggml logical dequant of one Q4_K super-block (256 vals) into y[256].
static void dequant_q4k_logical(const uint8_t* blk, float* y){
    uint16_t dh,dmh; memcpy(&dh,blk,2); memcpy(&dmh,blk+2,2);
    float d=(float)*reinterpret_cast<_Float16*>(&dh), dmin=(float)*reinterpret_cast<_Float16*>(&dmh);
    const uint8_t* sc=blk+4; const uint8_t* q=blk+16; int is=0;
    for(int j=0;j<4;j++){
        uint8_t s1,m1,s2,m2; get_scale_min_k4(is,sc,s1,m1); get_scale_min_k4(is+1,sc,s2,m2);
        float d1=d*s1,mn1=dmin*m1,d2=d*s2,mn2=dmin*m2;
        for(int l=0;l<32;l++) y[j*64+l]     = d1*(q[j*32+l]&0xF) - mn1;
        for(int l=0;l<32;l++) y[j*64+32+l]  = d2*(q[j*32+l]>>4)  - mn2;
        is+=2;
    }
}

int main(){
    std::mt19937 g(7); int fails=0;

    // ── T1: Q8_1 quantizer vs independent RNE ──
    {
        const int K=2048; std::vector<uint16_t> a(K); std::normal_distribution<float> nd(0,2.f);
        for(auto&x:a) x=f2bf(nd(g));
        int nb=K/32; std::vector<float> kd(nb); std::vector<int8_t> kq(K);
        lc::q8_1_quantize_row(a.data(),K,kd.data(),kq.data());
        int dqd=0,dqq=0;
        for(int b=0;b<nb;b++){
            float amax=0; for(int i=0;i<32;i++) amax=std::max(amax,std::fabs(bf2f(a[b*32+i])));
            float d=amax/127.f; float dx=f16rt(d);
            if(kd[b]!=dx) dqd++;
            for(int i=0;i<32;i++){ int v= d>0? (int)std::lrint(bf2f(a[b*32+i])/d):0; v=std::max(-127,std::min(127,v));
                if(kq[b*32+i]!=(int8_t)v) dqq++; }
        }
        printf("[T1] Q8_1 quant vs indep RNE: d differ=%d/%d, qs differ=%d/%d %s\n",
               dqd,nb,dqq,K,(dqd==0&&dqq==0)?"OK":"FAIL");
        if(dqd||dqq) fails++;
    }

    // ── T2: Q4_K x Q8_1 GEMM vs independent logical-order FP64 dequant dot ──
    {
        const int K=512, N=8, num_experts=3; int tok[3]={2,1,3};
        std::vector<int32_t> off(num_experts+1,0); for(int e=0;e<num_experts;e++) off[e+1]=off[e]+tok[e];
        int total=off[num_experts];
        const int nsb=K/256; const size_t wrow=(size_t)nsb*144;
        // random Q4_K weight rows per expert
        std::vector<std::vector<uint8_t>> W(num_experts);
        std::vector<const void*> Bptrs(num_experts);
        std::uniform_int_distribution<int> byte(0,255);
        std::normal_distribution<float> sd(0.05f,0.02f);
        for(int e=0;e<num_experts;e++){
            W[e].resize((size_t)N*wrow);
            for(int n=0;n<N;n++) for(int sb=0;sb<nsb;sb++){
                uint8_t* blk=W[e].data()+ (size_t)n*wrow + (size_t)sb*144;
                float d=std::fabs(sd(g))+0.001f, dmin=std::fabs(sd(g))*0.5f;
                _Float16 dh=(_Float16)d, dmh=(_Float16)dmin; memcpy(blk,&dh,2); memcpy(blk+2,&dmh,2);
                for(int i=4;i<144;i++) blk[i]=(uint8_t)byte(g);
            }
            Bptrs[e]=W[e].data();
        }
        std::vector<uint16_t> A((size_t)total*K); std::normal_distribution<float> nd(0,1.5f);
        for(auto&x:A) x=f2bf(nd(g));
        std::vector<uint16_t> D((size_t)total*N,0);
        lc::cpu_gguf_grouped_gemm_q4k_lossless(D.data(),A.data(),Bptrs.data(),off.data(),num_experts,N,K,nullptr);

        // reference: quantize activation to Q8_1 (kernel's own quantizer is fine as the
        // Q8_1 values; the INDEPENDENCE is in the DOT: logical dequant vs group-integer),
        // then logical dequant of weights, FP64 dot.
        double maxrel=0; int bad=0;
        for(int e=0;e<num_experts;e++) for(int m=off[e];m<off[e+1];m++){
            int nb=K/32; std::vector<float> ad(nb); std::vector<int8_t> aq(K);
            lc::q8_1_quantize_row(A.data()+(size_t)m*K,K,ad.data(),aq.data());
            std::vector<float> alog(K); for(int k=0;k<K;k++) alog[k]=ad[k/32]*(float)aq[k];
            for(int n=0;n<N;n++){
                const uint8_t* wr=W[e].data()+(size_t)n*wrow;
                std::vector<float> wlog(K); for(int sb=0;sb<nsb;sb++) dequant_q4k_logical(wr+(size_t)sb*144, wlog.data()+sb*256);
                double acc=0; for(int k=0;k<K;k++) acc+=(double)wlog[k]*(double)alog[k];
                float ref=(float)acc, got=bf2f(D[(size_t)m*N+n]);
                double rel=std::fabs(got-ref)/std::max(1.f,std::fabs(ref));
                maxrel=std::max(maxrel,rel); if(rel>0.02) bad++;
            }
        }
        printf("[T2] Q4_KxQ8_1 GEMM vs indep logical FP64 dot: max_rel=%.4g bad(>2%%)=%d/%d %s\n",
               maxrel,bad,total*N,bad==0?"OK":"FAIL");
        if(bad) fails++;
    }
    printf("\n%s (%d failing checks)\n",fails==0?"ALL GREEN":"FAILURES",fails);
    return fails?1:0;
}
