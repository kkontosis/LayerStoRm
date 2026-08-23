// Validate Q5_K + Q6_K lossless kernels vs independent ggml logical-dequant dots.
// Reference dequant logic derived from ggml/llama.cpp (MIT License,
// Copyright (c) 2023-2026 The ggml authors — see THIRD_PARTY_NOTICES.md).
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
static float h2f(uint16_t h){ return (float)*reinterpret_cast<_Float16*>(&h); }
static void gsm(int j,const uint8_t*q,uint8_t&d,uint8_t&m){ if(j<4){d=q[j]&63;m=q[j+4]&63;} else {d=(q[j+4]&0xF)|((q[j-4]>>6)<<4);m=(q[j+4]>>4)|((q[j]>>6)<<4);} }

// ggml logical dequant of one Q5_K super-block (176B) -> y[256].
static void deq_q5k(const uint8_t* blk, float* y){
    uint16_t dh,dmh; memcpy(&dh,blk,2); memcpy(&dmh,blk+2,2);
    float d=h2f(dh), dmin=h2f(dmh);
    const uint8_t* sc=blk+4; const uint8_t* qh=blk+16; const uint8_t* qs=blk+48;
    int is=0; uint8_t u1=1,u2=2;
    for(int j=0;j<4;j++){ uint8_t s1,m1,s2,m2; gsm(is,sc,s1,m1); gsm(is+1,sc,s2,m2);
        float d1=d*s1,mn1=dmin*m1,d2=d*s2,mn2=dmin*m2;
        for(int l=0;l<32;l++) y[j*64+l]    = d1*((qs[j*32+l]&0xF)+((qh[l]&u1)?16:0)) - mn1;
        for(int l=0;l<32;l++) y[j*64+32+l] = d2*((qs[j*32+l]>>4) +((qh[l]&u2)?16:0)) - mn2;
        is+=2; u1<<=2; u2<<=2; }
}
// ggml logical dequant of one Q6_K super-block (210B) -> y[256].
static void deq_q6k(const uint8_t* blk, float* y){
    const uint8_t* ql=blk; const uint8_t* qh=blk+128; const int8_t* sc=(const int8_t*)(blk+192);
    uint16_t dh; memcpy(&dh,blk+208,2); float d=h2f(dh);
    for(int n=0;n<2;n++){ const uint8_t* Ql=ql+n*64; const uint8_t* Qh=qh+n*32; const int8_t* S=sc+n*8; float* Y=y+n*128;
        for(int l=0;l<32;l++){ int is=l/16;
            int q1=((Ql[l]&0xF)|((Qh[l]&3)<<4))-32;
            int q2=((Ql[l+32]&0xF)|(((Qh[l]>>2)&3)<<4))-32;
            int q3=((Ql[l]>>4)|(((Qh[l]>>4)&3)<<4))-32;
            int q4=((Ql[l+32]>>4)|(((Qh[l]>>6)&3)<<4))-32;
            Y[l]=d*S[is+0]*q1; Y[l+32]=d*S[is+2]*q2; Y[l+64]=d*S[is+4]*q3; Y[l+96]=d*S[is+6]*q4; } }
}

static int run(lc::KQuantLossless t, const char* name, int blkbytes,
               void(*deq)(const uint8_t*,float*)){
    std::mt19937 g(t==lc::KQuantLossless::Q5_K?5:6);
    const int K=512,N=8,ne=3; int tok[3]={2,1,3}; std::vector<int32_t> off(ne+1,0);
    for(int e=0;e<ne;e++) off[e+1]=off[e]+tok[e]; int total=off[ne];
    int nsb=K/256; size_t wrow=(size_t)nsb*blkbytes;
    std::vector<std::vector<uint8_t>> W(ne); std::vector<const void*> B(ne);
    std::uniform_int_distribution<int> byte(0,255); std::normal_distribution<float> sd(0.03f,0.01f);
    for(int e=0;e<ne;e++){ W[e].resize((size_t)N*wrow);
        for(int n=0;n<N;n++) for(int sb=0;sb<nsb;sb++){ uint8_t* blk=W[e].data()+(size_t)n*wrow+(size_t)sb*blkbytes;
            for(int i=0;i<blkbytes;i++) blk[i]=(uint8_t)byte(g);
            // plausible fp16 super-scale(s)
            if(t==lc::KQuantLossless::Q6_K){ _Float16 d=(_Float16)(std::fabs(sd(g))+0.005f); memcpy(blk+208,&d,2); }
            else { _Float16 d=(_Float16)(std::fabs(sd(g))+0.005f), dm=(_Float16)(std::fabs(sd(g))*0.5f); memcpy(blk,&d,2); memcpy(blk+2,&dm,2); } }
        B[e]=W[e].data(); }
    std::vector<uint16_t> A((size_t)total*K); std::normal_distribution<float> nd(0,1.2f); for(auto&x:A)x=f2bf(nd(g));
    std::vector<uint16_t> D((size_t)total*N,0);
    lc::cpu_gguf_grouped_gemm_kq_lossless(t,D.data(),A.data(),B.data(),off.data(),ne,N,K,nullptr);
    double maxrel=0; int bad=0;
    for(int e=0;e<ne;e++) for(int m=off[e];m<off[e+1];m++){
        int nb=K/32; std::vector<float> ad(nb); std::vector<int8_t> aq(K);
        lc::q8_1_quantize_row(A.data()+(size_t)m*K,K,ad.data(),aq.data());
        std::vector<float> alog(K); for(int k=0;k<K;k++) alog[k]=ad[k/32]*(float)aq[k];
        for(int n=0;n<N;n++){ const uint8_t* wr=W[e].data()+(size_t)n*wrow;
            std::vector<float> wl(K); for(int sb=0;sb<nsb;sb++) deq(wr+(size_t)sb*blkbytes, wl.data()+sb*256);
            double acc=0; for(int k=0;k<K;k++) acc+=(double)wl[k]*(double)alog[k];
            float ref=(float)acc, got=bf2f(D[(size_t)m*N+n]);
            maxrel=std::max(maxrel,(double)(std::fabs(got-ref)/std::max(1.f,std::fabs(ref)))); if(maxrel>0.02) bad++; } }
    printf("[%s] vs indep logical FP64 dequant dot: max_rel=%.4g %s\n", name, maxrel, bad?"FAIL":"OK");
    return bad?1:0;
}
int main(){
    int f=0;
    f+=run(lc::KQuantLossless::Q5_K,"Q5_K",176,deq_q5k);
    f+=run(lc::KQuantLossless::Q6_K,"Q6_K",210,deq_q6k);
    printf("\n%s\n", f==0?"ALL GREEN":"FAILURES");
    return f?1:0;
}
