// Standalone validator for CPU-as-an-assignable-solver-target (Stage 1).
// Links ONLY loader_solver.cpp (CUDA-free, json-free). Mirrors the append that
// loader_constants.cpp::append_cpu_expert_device performs, and drives gl::solve
// to prove: (A) CPU absorbs the fetch-bound tail when it lowers makespan;
// (B) inert when CPU compute is expensive (assignment identical to GPU-only);
// (C) the CPU cannot cheat the DDR bank-contention floor.
#include "core/gpu_loader/loader_solver.h"
#include <cstdio>
#include <vector>
namespace gl = layerstorm::gpu_loader;

static gl::LoaderConstants make_constants(int M, int B, double egress, double rate) {
  gl::LoaderConstants k; k.source="t"; k.expert_bytes=24772992.0;
  k.num_devices=M; k.num_banks=B; k.ncf={0.0,1.0,1.2,1.5};
  for(int d=0;d<M;d++){ gl::DeviceConstants dc; dc.position=d; dc.numa_node=d;
    dc.xfer_lat_us=5.0; dc.compute={50.0,0.0,1}; k.devices.push_back(dc); }
  for(int b=0;b<B;b++){ gl::BankConstants bc; bc.node=b; bc.egress_us=egress; bc.contention=1.0; k.banks.push_back(bc); }
  k.matrix.assign(B,std::vector<gl::TransferCell>(M));
  for(int b=0;b<B;b++) for(int d=0;d<M;d++) k.matrix[b][d]=gl::TransferCell{rate,(b==d?1:2),5.0};
  return k;
}
// mirror of append_cpu_expert_device (validated separately by the gtest)
static int append_cpu(gl::LoaderConstants& k,int node,gl::ComputeCurve c){
  int j=k.num_devices; gl::DeviceConstants dc; dc.position=j; dc.numa_node=node;
  dc.name="cpu"; dc.xfer_lat_us=0.0; dc.compute=c; k.devices.push_back(dc);
  for(int b=0;b<k.num_banks;b++) k.matrix[b].push_back(gl::TransferCell{0.0,1,0.0});
  k.num_devices=j+1; return j;
}
static gl::SolveRequest make_request(int M,int N,const std::vector<int>& banks){
  gl::SolveRequest req; req.num_devices=M; req.num_experts=N;
  req.bank_of=banks; req.cached.assign((size_t)N*M,0); return req;
}
static std::vector<int> counts(const gl::SolveResult& r,int M){ std::vector<int> c(M,0); for(int i=0;i<r.n;i++) c[r.assignment[i]]++; return c; }

int main(){
  int fails=0;

  // ── (A) CPU absorbs the fetch-bound tail ──
  {
    auto k=make_constants(2,1,/*egress=*/50.0,/*rate=*/500.0);
    auto req=make_request(2,6,std::vector<int>(6,0));
    auto r0=gl::solve(k,req); double T0=r0.predicted_us;
    auto c2=append_cpu(k,3,{0.0,120.0,1});   // 120us/expert CPU, P=1
    auto req2=make_request(3,6,std::vector<int>(6,0));
    auto r1=gl::solve(k,req2); auto cc=counts(r1,3);
    bool cpu_used = cc[c2]>0, cheaper = r1.predicted_us < T0 - 1e-6;
    printf("[A] GPU-only T=%.0f ; with CPU T=%.0f  counts=[%d,%d,cpu=%d]  %s\n",
           T0,r1.predicted_us,cc[0],cc[1],cc[c2], (cpu_used&&cheaper)?"OK":"FAIL");
    if(!(cpu_used&&cheaper)) fails++;
  }

  // ── (B) inert when CPU compute is expensive: identical to GPU-only ──
  {
    auto k=make_constants(2,1,50.0,500.0);
    auto req=make_request(2,6,std::vector<int>(6,0));
    auto r0=gl::solve(k,req); auto c0=counts(r0,2);
    auto c2=append_cpu(k,3,{0.0,100000.0,1});  // absurdly slow CPU
    auto req2=make_request(3,6,std::vector<int>(6,0));
    auto r1=gl::solve(k,req2); auto cc=counts(r1,3);
    bool inert = cc[c2]==0 && std::abs(r1.predicted_us-r0.predicted_us)<1e-6;
    printf("[B] expensive CPU: cpu_count=%d  T same? %.0f vs %.0f  %s\n",
           cc[c2], r0.predicted_us, r1.predicted_us, inert?"OK":"FAIL");
    if(!inert) fails++;
  }

  // ── (C) CPU cannot cheat the DDR bank-contention floor ──
  {
    // bank0 cheap (HBM-like egress 50), bank1 = bottleneck DDR (egress 600).
    std::vector<int> banks={0,0,0,0,1,1,1,1};   // 4 from each bank
    auto k0=make_constants(2,2,/*egress=*/50.0,/*rate=*/500.0); k0.banks[1].egress_us=600.0;
    auto r0=gl::solve(k0,make_request(2,8,banks));         // GPU-only baseline
    auto k=make_constants(2,2,/*egress=*/50.0,/*rate=*/500.0); k.banks[1].egress_us=600.0;
    auto c2=append_cpu(k,3,{0.0,80.0,1});       // fast CPU
    auto r=gl::solve(k,make_request(3,8,banks)); auto cc=counts(r,3);
    double bank1_floor = 4*600.0;               // strict-serial (contention 1)
    // Contention invariant: CPU still reads bank1 -> cannot drop below its egress
    // floor; and adding the CPU option can only lower (never raise) makespan.
    bool floor_respected = r.predicted_us >= bank1_floor - 1e-6;
    bool never_hurts     = r.predicted_us <= r0.predicted_us + 1e-6;
    printf("[C] T=%.0f (gpu-only %.0f)  bank1_floor=%.0f  cpu_total=%d  %s\n",
           r.predicted_us, r0.predicted_us, bank1_floor, cc[c2],
           (floor_respected&&never_hurts)?"OK":"FAIL");
    if(!(floor_respected&&never_hurts)) fails++;
  }

  printf("\n%s (%d failing checks)\n", fails==0?"ALL GREEN":"FAILURES", fails);
  return fails?1:0;
}
