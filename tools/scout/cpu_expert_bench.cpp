// Standalone CPU-expert-offload feasibility microbench.
// Two decision-relevant numbers the existing gtest benches don't give cleanly:
//   (A) Full expert FFN COMPUTE ROOFLINE at GLM-5.2 dims (H=6144, I=2048), BF16
//       gate+up+down GEMV/GEMM, N threads pinned to one NUMA node. This is the
//       hardware compute floor independent of quant format (real nvfp4/gguf
//       kernels are >= this and already benched on this box).
//   (B) DDR-node READ-BANDWIDTH CONTENTION: sustained read GB/s from a DDR node
//       with N threads. The FETCH_XRAY wall is per-DDR-node ~64 GiB/s; a CPU
//       expert reading that same node competes with the H2D DMA. This measures
//       how much read BW a CPU-expert workload consumes -> the contention term.
//
// No engine, no packed-weight formats, no GPU. Safe to run alongside anything.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

using clk = std::chrono::steady_clock;
static double now_s() {
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

static void pin(int cpu) {
  cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
  pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

// ---- GLM-5.2 routed expert dims ----
static const int H = 6144;   // hidden
static const int I = 2048;   // moe_intermediate

// BF16 helpers
static inline float bf2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }
static inline uint16_t f2bf(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }

// One projection GEMV: out[n] = sum_k a[k]*W[n*K+k], W row-major [N,K] BF16, a BF16.
// Rows partitioned across threads. FP32 accumulate. Uses AVX512-BF16 dot via cvt.
static void gemv_bf16(const uint16_t* W, const uint16_t* a, float* out,
                      int N, int K, int t, int nt) {
  int r0 = (int64_t)N*t/nt, r1=(int64_t)N*(t+1)/nt;
  for (int n=r0;n<r1;n++){
    const uint16_t* w=W+(int64_t)n*K;
    __m512 acc=_mm512_setzero_ps();
    int k=0;
    for(;k+16<=K;k+=16){
      __m256i wi=_mm256_loadu_si256((const __m256i*)(w+k));
      __m256i ai=_mm256_loadu_si256((const __m256i*)(a+k));
      // bf16 -> f32: shift left 16
      __m512 wf=_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(wi),16));
      __m512 af=_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(ai),16));
      acc=_mm512_fmadd_ps(wf,af,acc);
    }
    float s=_mm512_reduce_add_ps(acc);
    for(;k<K;k++) s+=bf2f(w[k])*bf2f(a[k]);
    out[n]=s;
  }
}

// GEMM for T tokens: out[t,n] loop (naive, T small). Rows(N) partitioned.
static void gemm_bf16(const uint16_t* W, const uint16_t* A, float* out,
                      int T, int N, int K, int tid, int nt){
  for(int tt=0;tt<T;tt++) gemv_bf16(W, A+(int64_t)tt*K, out+(int64_t)tt*N, N, K, tid, nt);
}

struct Barrier {
  std::atomic<int> cnt{0}; std::atomic<int> gen{0}; int n;
  void wait(){ int g=gen.load(); if(cnt.fetch_add(1)+1==n){cnt.store(0);gen.fetch_add(1);} else{ while(gen.load()==g) _mm_pause(); } }
};

int main(int argc, char** argv){
  int node = argc>1?atoi(argv[1]):3;         // DDR node (0-3 have cores)
  int nthreads = argc>2?atoi(argv[2]):14;    // physical cores/node
  // node N cores: base = node*14 .. +13 (from lscpu: node0:0-13, node1:14-27,...)
  int core0 = node*14;

  // ---- allocate weights (one expert: gate[I,H], up[I,H], down[H,I]) BF16 ----
  auto alloc=[&](size_t n){ void*p=nullptr; if(posix_memalign(&p,64,n)) {perror("alloc");exit(1);} return p; };
  size_t geB=(size_t)I*H*2, upB=(size_t)I*H*2, dnB=(size_t)H*I*2;
  uint16_t* Wg=(uint16_t*)alloc(geB);
  uint16_t* Wu=(uint16_t*)alloc(upB);
  uint16_t* Wd=(uint16_t*)alloc(dnB);
  // touch on the target node's first core so pages land there
  pin(core0);
  for(size_t i=0;i<(size_t)I*H;i++){ Wg[i]=f2bf(0.001f*((i%7)-3)); Wu[i]=f2bf(0.001f*((i%5)-2)); }
  for(size_t i=0;i<(size_t)H*I;i++) Wd[i]=f2bf(0.001f*((i%11)-5));

  size_t per_expert_bytes = geB+upB+dnB;
  printf("== GLM-5.2 expert dims H=%d I=%d, BF16 per-expert=%.1f MiB ==\n",
         H, I, per_expert_bytes/1048576.0);

  // ================= (A) FFN COMPUTE ROOFLINE =================
  for (int T : {1, 16}) {
    uint16_t* A=(uint16_t*)alloc((size_t)T*H*2);
    for(size_t i=0;i<(size_t)T*H;i++) A[i]=f2bf(0.01f*((i%13)-6));
    float* g=(float*)alloc((size_t)T*I*4);
    float* u=(float*)alloc((size_t)T*I*4);
    uint16_t* act=(uint16_t*)alloc((size_t)T*I*2);
    float* d=(float*)alloc((size_t)T*H*4);

    Barrier bar; bar.n=nthreads;
    std::atomic<double> tmin{1e9};
    const int iters=30;
    auto work=[&](int tid){
      pin(core0+tid);
      double best=1e9;
      for(int it=0; it<iters+3; it++){
        bar.wait();
        double t0=now_s();
        gemm_bf16(Wg,A,g,T,I,H,tid,nthreads);   // gate
        gemm_bf16(Wu,A,u,T,I,H,tid,nthreads);   // up
        bar.wait();
        // swiglu on partitioned rows of act[T,I]
        int r0=(int64_t)(T*I)*tid/nthreads, r1=(int64_t)(T*I)*(tid+1)/nthreads;
        for(int idx=r0; idx<r1; idx++){ float gg=g[idx]; float si=gg/(1.f+expf(-gg)); act[idx]=f2bf(si*u[idx]); }
        bar.wait();
        gemm_bf16(Wd,act,d,T,H,I,tid,nthreads); // down
        bar.wait();
        double t1=now_s();
        if(tid==0 && it>=3) best=std::min(best,t1-t0);
      }
      if(tid==0) tmin.store(best);
    };
    std::vector<std::thread> th;
    for(int t=1;t<nthreads;t++) th.emplace_back(work,t);
    work(0);
    for(auto&x:th)x.join();
    double us=tmin.load()*1e6;
    // FLOPs: 3 GEMMs * 2*T*N*K
    double flops = 2.0*T*I*H + 2.0*T*I*H + 2.0*T*H*I;
    printf("(A) FFN roofline T=%2d node=%d nthr=%d : %.1f us/expert  (%.1f el/s)  %.1f GFLOP/s\n",
           T, node, nthreads, us, T>0?(1e6/us*T):0.0, flops/(us*1e-6)/1e9);
    free(A);free(g);free(u);free(act);free(d);
  }

  // ================= (B) DDR-NODE READ-BW CONTENTION =================
  // Stream-read a large buffer on `node` with nthreads; report aggregate GB/s.
  // This is the read pressure a CPU-expert workload puts on the node the H2D
  // DMA also reads. Buffer sized > LLC to force DDR (not HBM-cache).
  {
    size_t bufB = (size_t)2*1024*1024*1024ULL; // 4 GiB
    uint8_t* buf=(uint8_t*)alloc(bufB);
    pin(core0); memset(buf,1,bufB); // first-touch on node
    for(int nt : {1,2,4,8,14}) {
      if(nt>nthreads) break;
      Barrier bar; bar.n=nt;
      double agg=0; std::mutex mtx;
      std::atomic<uint64_t> checksum{0};
      auto rd=[&](int tid){
        pin(core0+tid);
        size_t chunk=bufB/nt; uint8_t* p=buf+chunk*tid;
        uint64_t acc=0;
        bar.wait(); double t0=now_s();
        for(int rep=0;rep<2;rep++)
          for(size_t i=0;i<chunk;i+=64){ __m512i v=_mm512_loadu_si512((const void*)(p+i)); acc+=(uint64_t)_mm512_reduce_add_epi64(v);}
        double t1=now_s();
        checksum.fetch_add(acc);
        double gb=(double)chunk*2/1e9; std::lock_guard<std::mutex> lk(mtx); agg+=gb/(t1-t0);
      };
      std::vector<std::thread> th;
      for(int t=1;t<nt;t++) th.emplace_back(rd,t);
      rd(0);
      for(auto&x:th)x.join();
      printf("(B) DDR-node%d read BW: %2d threads -> %.1f GB/s aggregate\n", node, nt, agg);
    }
    free(buf);
  }
  return 0;
}
