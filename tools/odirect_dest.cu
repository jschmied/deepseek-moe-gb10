// Can NVMe O_DIRECT land straight in memory the SMs read?  Three destinations,
// real DS4.1 shards, the engine's own 14.45 MB expert-read shape.
#define _GNU_SOURCE
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <cuda_runtime.h>

static const size_t EXPERT = 14454784;           // 14.45 MB
static size_t CHUNK = 4u<<20;                    // engine reads in 4 MB pieces
static const char*  PATHS[] = {
  "/opt/llm/models/dsv41-shards/model-00003-of-00048.safetensors",
  "/opt/llm/models/dsv41-shards/model-00004-of-00048.safetensors",
  "/opt/llm/models/dsv41-shards/model-00005-of-00048.safetensors",
  "/opt/llm/models/dsv41-shards/model-00006-of-00048.safetensors"};
static const int NPATH = 4;

struct Arg { char* buf; int nreads; int tid; long long bytes; int err; };
static double now(){ struct timeval t; gettimeofday(&t,0); return t.tv_sec + t.tv_usec/1e6; }

static void* worker(void* v){
  Arg* a = (Arg*)v;
  int fd = open(PATHS[a->tid % NPATH], O_RDONLY | O_DIRECT);
  if(fd < 0){ a->err = errno; return 0; }
  off_t sz = lseek(fd, 0, SEEK_END);
  for(int r=0; r<a->nreads; r++){
    off_t off = ((off_t)(rand() % (int)((sz - EXPERT) / 4096))) * 4096;
    size_t done = 0;
    while(done < EXPERT){
      size_t want = EXPERT - done; if(want > CHUNK) want = CHUNK;
      ssize_t got = pread(fd, a->buf + done, want, off + done);
      if(got <= 0){ a->err = errno ? errno : -1; close(fd); return 0; }
      done += got;
    }
    a->bytes += done;
  }
  close(fd); return 0;
}

static void run(const char* tag, char* base, size_t stride, int nthread, int nreads){
  pthread_t th[64]; Arg ar[64];
  for(int i=0;i<nthread;i++){ ar[i] = {base + i*stride, nreads, i, 0, 0}; }
  double t0 = now();
  for(int i=0;i<nthread;i++) pthread_create(&th[i],0,worker,&ar[i]);
  long long tot = 0; int err = 0;
  for(int i=0;i<nthread;i++){ pthread_join(th[i],0); tot += ar[i].bytes; if(ar[i].err) err = ar[i].err; }
  double dt = now() - t0;
  if(err) printf("  %-28s T=%-2d  FAILED errno=%d (%s)\n", tag, nthread, err, strerror(err));
  else    printf("  %-28s T=%-2d  %5.2f GB/s   (%lld MB in %.2f s)\n",
                 tag, nthread, tot/dt/1e9, tot>>20, dt);
}

int main(){
  int NT = 32, NR = 12;
  if(getenv("CHUNK_MB")) CHUNK = (size_t)atoi(getenv("CHUNK_MB"))<<20;
  size_t stride = EXPERT;
  size_t need = stride * NT;

  printf("O_DIRECT preadv of %.2f MB experts in %.0f MB chunks into three destinations\n\n", EXPERT/1048576.0, CHUNK/1048576.0);

  { void* p; if(posix_memalign(&p, 4096, need)) { perror("memalign"); return 1; }
    memset(p,0,need);
    for(int t : {1,2,4,8,16,32}) run("posix_memalign (pageable)", (char*)p, stride, t, NR);
    free(p); printf("\n"); }

  { void* h; cudaError_t e = cudaHostAlloc(&h, need, cudaHostAllocMapped);
    if(e){ printf("  cudaHostAlloc failed: %s\n", cudaGetErrorString(e)); }
    else { void* dp; cudaHostGetDevicePointer(&dp,h,0);
           printf("  (host %p == device %p: %s)\n", h, dp, h==dp?"yes":"no");
           memset(h,0,need);
           for(int t : {1,2,4,8,16,32}) run("cudaHostAlloc (SM-readable)", (char*)h, stride, t, NR);
           cudaFreeHost(h); }
    printf("\n"); }

  { void* d; cudaError_t e = cudaMalloc(&d, need);
    if(e) printf("  cudaMalloc failed: %s\n", cudaGetErrorString(e));
    else { run("cudaMalloc (device)", (char*)d, stride, 1, 2); cudaFree(d); }
    printf("\n"); }

  { void* m; cudaError_t e = cudaMallocManaged(&m, need);
    if(e) printf("  cudaMallocManaged failed: %s\n", cudaGetErrorString(e));
    else { cudaMemset(m,0,need); cudaDeviceSynchronize();
           for(int t : {1,4}) run("cudaMallocManaged", (char*)m, stride, t, NR);
           cudaFree(m); } }

  printf("\n== ALL DONE ==\n");
  return 0;
}
