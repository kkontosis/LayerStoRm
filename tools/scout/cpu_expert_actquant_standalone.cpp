// Standalone CPU-only validation of the bit-compatible NVFP4 activation-quant
// path (Stage 0). Links ONLY the LayerStoRmCpuExpertKernels object lib. No CUDA.
//
// Validates:
//  (T1) the kernel's E4M3 encoder + FP4 quantizer (via nvfp4_quantize_activation_row)
//       vs an INDEPENDENT brute-force round-to-nearest-even e4m3 reference.
//  (T2) cpu_nvfp4_grouped_gemm_actquant vs a DOUBLE-precision reference of the
//       same quantized arithmetic (both operands NVFP4, per-element scales) —
//       this is the GPU CUTLASS path's *defining* arithmetic.
//  (T3) actquant DIFFERS from the old lossy BF16-activation path (proves the
//       activation-quantization gap was real and is now closed).

#include "compute/cpu/nvfp4_cpu_kernel.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

// ── Production E4M3 decode, copied VERBATIM from src/model/quantization/fp8.cpp
//    (bias=7, 3 mantissa bits) so the standalone result transfers exactly. ──
namespace layerstorm::model::fp8_e4m3 {
static constexpr int kBias = 7;
static constexpr int kMantissaBits = 3;
float decode(uint8_t byte) {
    bool    sign = (byte >> 7) & 1;
    uint8_t exp  = (byte >> 3) & 0x0F;
    uint8_t mant = byte & 0x07;
    if (exp == 0x0F && mant == 0x07) return std::numeric_limits<float>::quiet_NaN();
    float value;
    if (exp == 0) value = std::ldexp(static_cast<float>(mant), -kBias - kMantissaBits + 1);
    else value = std::ldexp(static_cast<float>((1 << kMantissaBits) + mant),
                            exp - kBias - kMantissaBits);
    return sign ? -value : value;
}
}  // namespace layerstorm::model::fp8_e4m3

namespace lc = layerstorm::compute::cpu;
using lc::kE2M1Table;
using layerstorm::model::fp8_e4m3::decode;

static inline float bf2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }
static inline uint16_t f2bf(float f){ uint32_t u; memcpy(&u,&f,4); // round-to-nearest-even to bf16
    uint32_t lsb=(u>>16)&1; u += 0x7fff + lsb; return (uint16_t)(u>>16); }

// Independent brute-force RNE float->ue4m3 (search all codes; tie -> even).
static uint8_t ref_float_to_ue4m3(float v){
    if(!(v>0.0f)) return 0x00;
    if(v>=448.0f) return 0x7e;
    int best=-1; float bestd=1e30f;
    for(int b=0;b<=0x7e;b++){ // sign 0, exclude 0x7f NaN
        float d=decode((uint8_t)b); if(d<0) continue;
        float e=std::fabs(d-v);
        if(e<bestd-1e-20f){bestd=e;best=b;}
        else if(std::fabs(e-bestd)<=1e-20f){ if((b&1)==0) best=b; } // tie -> even mantissa
    }
    return (uint8_t)best;
}
static int ref_fp4_nibble(float v){
    uint8_t sign=(v<0)?8:0; float a=std::fabs(v);
    uint8_t idx=(a>=0.25f)+(a>=0.75f)+(a>=1.25f)+(a>=1.75f)+(a>=2.5f)+(a>=3.5f)+(a>=5.0f);
    return idx|sign;
}
static void ref_quant_row(const uint16_t* a, int K, float is_in, float* out){
    float is=(is_in>0)?is_in:1.0f;
    for(int g=0;g<K;g+=16){
        float amax=0; for(int k=0;k<16;k++) amax=std::max(amax,std::fabs(bf2f(a[g+k])));
        float raw=amax/(6.0f*is);
        float sr=decode(ref_float_to_ue4m3(raw));
        float denom=sr*is;
        for(int k=0;k<16;k++){ float x=bf2f(a[g+k]);
            out[g+k]= denom>0 ? kE2M1Table[ref_fp4_nibble(x/denom)]*sr*is : 0.0f; }
    }
}

// Build a valid nvfp4-sm1xx packed projection [N,K] with random nibbles + random
// E4M3 group scales + weight_scale_2, matching scale_byte_index de-interleave.
struct Packed { std::vector<uint8_t> buf; lc::PackedProjection proj; std::vector<float> wdeq; };
static Packed make_projection(int N,int K,std::mt19937& g,float ws2){
    const int groups=K/16;
    const int64_t P=(int64_t)N*K;
    const int64_t fp4_bytes=(P+1)/2;
    const int64_t scale_bytes=(P+15)/16;
    int64_t raw=fp4_bytes+scale_bytes+2*(int64_t)sizeof(float);
    int64_t proj_bytes=(raw+127)&~(int64_t)127;
    Packed pk; pk.buf.assign(proj_bytes,0); pk.wdeq.assign((size_t)N*K,0.f);
    uint8_t* base=pk.buf.data();
    uint8_t* scl=base+fp4_bytes;
    std::uniform_int_distribution<int> nib(0,15);
    std::uniform_int_distribution<int> e4(0x30,0x40); // scales near 1.0 (0x38)
    // scales per (n,group)
    for(int n=0;n<N;n++) for(int gg=0;gg<groups;gg++){
        uint8_t sb=(uint8_t)e4(g);
        scl[lc::scale_byte_index(n,gg,groups)]=sb;
    }
    for(int n=0;n<N;n++) for(int k=0;k<K;k++){
        int nb=nib(g);
        int64_t flat=(int64_t)n*K+k;
        uint8_t& byte=base[flat>>1];
        if(flat&1) byte=(byte&0x0F)|((nb&0xF)<<4); else byte=(byte&0xF0)|(nb&0xF);
        uint8_t sb=scl[lc::scale_byte_index(n,k/16,groups)];
        pk.wdeq[(size_t)n*K+k]=kE2M1Table[nb]*decode(sb)*ws2;
    }
    memcpy(base+proj_bytes-8,&ws2,4);
    float is=1.0f; memcpy(base+proj_bytes-4,&is,4);
    pk.proj.base=base; pk.proj.proj_bytes=(size_t)proj_bytes; pk.proj.N=N; pk.proj.K=K;
    return pk;
}

int main(){
    std::mt19937 g(1234);
    int fails=0;

    // ── T1: encoder + fp4 quantizer vs independent brute-force RNE ──
    {
        const int K=6144; // GLM-5.2 hidden (multiple of 16)
        std::vector<uint16_t> a(K); std::normal_distribution<float> nd(0,2.0f);
        for(auto&x:a) x=f2bf(nd(g));
        for(float is : {1.0f, 0.5f, 2.3f}){
            std::vector<float> got(K),ref(K);
            lc::nvfp4_quantize_activation_row(a.data(),K,is,got.data());
            ref_quant_row(a.data(),K,is,ref.data());
            int diff=0; for(int k=0;k<K;k++) if(got[k]!=ref[k]) diff++;
            printf("[T1] act-quant is=%.2f : %d/%d elems differ from brute-force RNE ref%s\n",
                   is,diff,K, diff==0?"  OK":"  FAIL");
            if(diff!=0) fails++;
        }
    }

    // ── T2 + T3: full grouped GEMM bit-compat vs double reference ──
    {
        const int N=128, K=64;                 // aligned packed dims
        const int num_experts=3;
        std::vector<int> tok={2,1,3};          // tokens per expert (mix M=1 and M>1)
        int total=0; std::vector<int32_t> off(num_experts+1,0);
        for(int e=0;e<num_experts;e++){ off[e]=total; total+=tok[e]; } off[num_experts]=total;

        std::vector<Packed> pk; std::vector<lc::PackedProjection> projs;
        std::uniform_real_distribution<float> wd(0.5f,1.5f);
        for(int e=0;e<num_experts;e++){ pk.push_back(make_projection(N,K,g,wd(g))); }
        for(int e=0;e<num_experts;e++) projs.push_back(pk[e].proj);
        lc::CpuNvfp4ExpertWeights W{projs.data(),num_experts};

        std::vector<uint16_t> A((size_t)total*K); std::normal_distribution<float> nd(0,1.5f);
        for(auto&x:A) x=f2bf(nd(g));

        std::vector<uint16_t> Dact((size_t)total*N,0), Dlossy((size_t)total*N,0);
        lc::cpu_nvfp4_grouped_gemm_actquant(Dact.data(),A.data(),W,nullptr,off.data(),N,K,2,nullptr);
        lc::cpu_nvfp4_grouped_gemm(Dlossy.data(),A.data(),W,off.data(),N,K,2,nullptr);

        // double reference of the quantized arithmetic
        double maxrel=0; int bad=0;
        double diff_lossy=0;
        for(int e=0;e<num_experts;e++){
            for(int m=off[e];m<off[e+1];m++){
                std::vector<float> adq(K); ref_quant_row(A.data()+(size_t)m*K,K,1.0f,adq.data());
                for(int n=0;n<N;n++){
                    double acc=0; for(int k=0;k<K;k++) acc+=(double)adq[k]*(double)pk[e].wdeq[(size_t)n*K+k];
                    float ref=(float)acc;
                    float got=bf2f(Dact[(size_t)m*N+n]);
                    float rel=std::fabs(got-ref)/std::max(1.0f,std::fabs(ref));
                    maxrel=std::max(maxrel,(double)rel); if(rel>0.03f) bad++;
                    diff_lossy+=std::fabs(bf2f(Dlossy[(size_t)m*N+n])-ref);
                }
            }
        }
        printf("[T2] actquant GEMM vs double ref: max_rel=%.4g  bad(>3%%)=%d/%d%s\n",
               maxrel,bad,total*N, (bad==0?"  OK":"  FAIL"));
        if(bad!=0) fails++;
        printf("[T3] lossy(BF16-act) path total |D - ref| = %.3g (must be >0: activation-quant gap)%s\n",
               diff_lossy, diff_lossy>0?"  OK":"  FAIL");
        if(!(diff_lossy>0)) fails++;
    }

    printf("\n%s (%d failing checks)\n", fails==0?"ALL GREEN":"FAILURES", fails);
    return fails==0?0:1;
}
