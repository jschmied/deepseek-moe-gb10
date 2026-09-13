// Does an SM read host-pinned (cudaHostAlloc) memory fast enough for a host-pinned
// expert arena on GB10?  Decision test for "O_DIRECT straight into cudaHostAlloc".
// Compares cudaMalloc / cudaHostAlloc(mapped) / cudaMallocManaged under the same kernel.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

__global__ void stream_read(const float4* __restrict__ p, size_t n4, float* out){
  float acc = 0.f;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for(size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x; i < n4; i += stride){
    float4 v = p[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if(acc == 1.2345e-30f) out[0] = acc;   // never true; defeats DCE
}

// MoE-shaped: each block walks one "expert" of EXPERT_B bytes, blocks scattered over the arena
__global__ void expert_read(const float4* __restrict__ p, size_t n4, size_t e4,
                            const unsigned* ids, float* out){
  size_t base = (size_t)ids[blockIdx.x] * e4;
  float acc = 0.f;
  for(size_t i = threadIdx.x; i < e4; i += blockDim.x){
    size_t j = base + i; if(j >= n4) j -= n4;
    float4 v = p[j];
    acc += v.x + v.y + v.z + v.w;
  }
  if(acc == 1.2345e-30f) out[0] = acc;
}

double bench(const char* tag, float4* dptr, size_t bytes, int reps){
  size_t n4 = bytes / sizeof(float4);
  float* out; CK(cudaMalloc(&out, 4));
  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  int blocks = 48 * 32, threads = 256;          // 48 SMs
  stream_read<<<blocks,threads>>>(dptr,n4,out); CK(cudaDeviceSynchronize());   // warm
  CK(cudaEventRecord(a));
  for(int r=0;r<reps;r++) stream_read<<<blocks,threads>>>(dptr,n4,out);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms,a,b));
  double gbs = (double)bytes*reps/(ms/1e3)/1e9;
  printf("  %-26s stream read  %7.1f GB/s   (%.1f ms/pass over %.1f GiB)\n",
         tag, gbs, ms/reps, bytes/1073741824.0);
  cudaFree(out); return gbs;
}

double bench_expert(const char* tag, float4* dptr, size_t bytes, size_t expert_bytes, int nexp){
  size_t n4 = bytes/sizeof(float4), e4 = expert_bytes/sizeof(float4);
  unsigned* ids; CK(cudaMallocManaged(&ids, nexp*sizeof(unsigned)));
  size_t nslots = bytes/expert_bytes;
  for(int i=0;i<nexp;i++) ids[i] = (unsigned)((rand()% nslots));
  float* out; CK(cudaMalloc(&out,4));
  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  expert_read<<<nexp,256>>>(dptr,n4,e4,ids,out); CK(cudaDeviceSynchronize());
  int reps=20;
  CK(cudaEventRecord(a));
  for(int r=0;r<reps;r++) expert_read<<<nexp,256>>>(dptr,n4,e4,ids,out);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms,a,b));
  double gbs = (double)expert_bytes*nexp*reps/(ms/1e3)/1e9;
  printf("  %-26s expert gather %6.1f GB/s   (%d x %.2f MB, %.2f ms)\n",
         tag, gbs, nexp, expert_bytes/1048576.0, ms/reps);
  cudaFree(out); cudaFree(ids); return gbs;
}

int main(int argc, char** argv){
  size_t GiB = 1073741824ull;
  size_t bytes = (argc>1? atoll(argv[1]) : 8) * GiB;
  int reps = 10;
  size_t expert_bytes = 14454784;              // 14.45 MB, the DS4.1 expert size
  int nexp = 640;                              // ~ a decode step's arena reads

  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr,0));
  printf("device %s  SMs %d  canMapHostMemory %d  unifiedAddressing %d  pageableAccess %d\n",
         pr.name, pr.multiProcessorCount, pr.canMapHostMemory, pr.unifiedAddressing,
         pr.pageableMemoryAccess);
  printf("arena %.0f GiB  expert %.2f MB\n\n", bytes/1073741824.0, expert_bytes/1048576.0);

  { float4* d; CK(cudaMalloc(&d,bytes)); CK(cudaMemset(d,1,bytes));
    bench("cudaMalloc (device)", d, bytes, reps);
    bench_expert("cudaMalloc (device)", d, bytes, expert_bytes, nexp);
    CK(cudaFree(d)); }

  { void* h; CK(cudaHostAlloc(&h,bytes,cudaHostAllocMapped|cudaHostAllocPortable));
    void* dp; CK(cudaHostGetDevicePointer(&dp,h,0));
    printf("  host ptr %p -> device ptr %p  (%s)\n", h, dp, h==dp?"same, UVA":"remapped");
    memset(h,1,bytes);
    bench("cudaHostAlloc (pinned host)", (float4*)dp, bytes, reps);
    bench_expert("cudaHostAlloc (pinned host)", (float4*)dp, bytes, expert_bytes, nexp);
    CK(cudaFreeHost(h)); }

  { void* m; CK(cudaMallocManaged(&m,bytes)); CK(cudaMemset(m,1,bytes)); CK(cudaDeviceSynchronize());
    bench("cudaMallocManaged", (float4*)m, bytes, reps);
    bench_expert("cudaMallocManaged", (float4*)m, bytes, expert_bytes, nexp);
    CK(cudaFree(m)); }

  printf("\n== ALL DONE ==\n");
  return 0;
}
