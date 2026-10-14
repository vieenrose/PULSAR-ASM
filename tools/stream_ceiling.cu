// What is the maximum sustainable DRAM read rate for this GPU?
// GM20B in the Jetson Nano = ONE SM. Grid-stride sum, uint4 loads.
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>
__global__ void rd(const uint4* __restrict__ a, size_t n4, int* sink) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  unsigned s = 0;
  for (; i < n4; i += stride) { uint4 v = a[i]; s += v.x + v.y + v.z + v.w; }
  if (s == 0xdeadbeefu) *sink = 1;   // keep loads alive, never true
}
int main() {
  const size_t MB = 1024*1024;
  size_t bytes = 512*MB;
  uint4* d; int* sink;
  cudaMalloc(&d, bytes); cudaMalloc(&sink, 4); cudaMemset(d, 1, bytes);
  size_t n4 = bytes/16;
  printf("buffer %.0f MB (DRAM-resident; L2 is 128KB)\n", bytes/(double)MB);
  int blk = 256;
  int grids[] = {1,2,4,5,10,20,40,80,160,320,640,1280};
  for (int g : grids) {
    for (int i=0;i<3;i++) rd<<<g,blk>>>(d,n4,sink);   // warm
    cudaDeviceSynchronize();
    timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    for (int i=0;i<5;i++) rd<<<g,blk>>>(d,n4,sink);
    cudaDeviceSynchronize(); clock_gettime(CLOCK_MONOTONIC,&t1);
    double ms = ((double)(t1.tv_sec-t0.tv_sec)*1e3 + (double)(t1.tv_nsec-t0.tv_nsec)*1e-6)/5.0;
    printf("  grid=%5d blocks  %8.2f ms  %7.2f GB/s   -> 1.06GB/token = %6.1f ms  = %5.2f tok/s\n",
           g, ms, bytes/ms/1e6, 1.06e9/(bytes/ms/1e6)/1e6, 1000.0/(1.06e9/(bytes/ms/1e6)/1e6));
  }
  cudaFree(d); cudaFree(sink); return 0;
}
