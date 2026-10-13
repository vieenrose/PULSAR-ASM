#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
static const char *ptx =
#include "dbgptx_str.h"
;
#define CHECK(x) do{ CUresult e=(x); if(e){printf("ERR %s -> %d\n",#x,e); return 1;} }while(0)
static float f32at(const unsigned char *p){ float v; memcpy(&v,p,4); return v; }
int main(){
  int rows=8, nb32=48, iters=(nb32+7)/8;
  size_t szw=(size_t)rows*nb32*18+64, szs=(size_t)rows*nb32*16, szx=65536, szd=(size_t)rows*2048*4;
  unsigned char *qw=(unsigned char*)malloc(szw), *qx=(unsigned char*)malloc(szx);
  float *qs=(float*)malloc(szs), *qd=(float*)calloc(rows*2048,4), *qy=(float*)calloc(rows,4);
  for(size_t i=0;i<szw;i++) qw[i]=(unsigned char)(i*37+11);
  for(size_t i=0;i<szx;i++) qx[i]=(unsigned char)(i*53-90);
  for(size_t i=0;i<szs/4;i++){ float v=0.001f*(i%29)-0.013f; memcpy(&qs[i],&v,4); }
  CHECK(cuInit(0)); CUdevice dev; CUcontext ctx;
  CHECK(cuDeviceGet(&dev,0)); CHECK(cuCtxCreate(&ctx,0,dev));
  CUmodule mod; CUfunction fn;
  CHECK(cuModuleLoadData(&mod,ptx));
  CHECK(cuModuleGetFunction(&fn,mod,"q4_gemv_dbg"));
  CUdeviceptr dw,dx,ds,dy,dd;
  CHECK(cuMemAlloc(&dw,szw)); CHECK(cuMemAlloc(&dx,szx)); CHECK(cuMemAlloc(&ds,szs));
  CHECK(cuMemAlloc(&dy,64)); CHECK(cuMemAlloc(&dd,szd));
  CHECK(cuMemcpyHtoD(dw,qw,szw)); CHECK(cuMemcpyHtoD(dx,qx,szx)); CHECK(cuMemcpyHtoD(ds,qs,szs));
  CHECK(cuMemsetD8(dy,0,64)); CHECK(cuMemsetD8(dd,0,szd));
  int r0=0,r1=rows; unsigned nb=nb32, rf=0;
  void *args[]={&dw,&ds,&dx,&dy,&r0,&r1,&nb,&rf,&dd};
  CHECK(cuLaunchKernel(fn,1,1,1,32,8,1,0,0,args,0));
  CHECK(cuCtxSynchronize());
  CHECK(cuMemcpyDtoH(qd,dd,szd)); CHECK(cuMemcpyDtoH(qy,dy,64));
  int bad=0;
  for(int r=0;r<rows;r++) for(int it=0;it<iters;it++) for(int t=0;t<32;t++){
    int s=t>>2, q=t&3, b=8*it+s;
    int p=0;
    for(int k=0;k<4;k++){
      int j=q+4*k;
      unsigned cb=qw[(size_t)r*nb32*18+2+18*b+j];
      int x8=292*it+32*s+j+4;
      int lo=(int)(cb&15)-8, hi=(int)((cb>>4)&15)-8;
      signed char a=(signed char)qx[x8], c=(signed char)qx[x8+16];
      p += lo*(int)a + hi*(int)c;
    }
    float want6=(float)p;
    float db=f32at(qx+292*it);
    float want7 = (b<nb32) ? (float)((double)qs[(size_t)r*nb32+b]*(double)db) : 0.0f;
    float got6=qd[(size_t)r*2048+it*32+t], got7=qd[(size_t)r*2048+1024+it*32+t];
    if(got6!=want6 || got7!=want7){
      printf("MISMATCH r%d it%d t%d: f6 got %f want %f | f7 got %f want %f\n",r,it,t,got6,want6,got7,want7);
      if(++bad>10) goto done;
    }
  }
done:
  printf("y0=%f\n",qy[0]);
  { int ybad=0;
    for(int r=0;r<rows;r++){
      float acc=0.f;
      const unsigned char* wp0=qw+(size_t)r*nb32*18+2;
      const float* sp=qs+(size_t)r*nb32;
      for(int b=0;b<nb32;b++){
        const unsigned char* wp=wp0+(size_t)b*18;
        const signed char* xb=(const signed char*)qx+(size_t)(b>>3)*292+4+(b&7)*32;
        int dot=0;
        for(int k=0;k<16;k++){ int lo=(wp[k]&15)-8, hi=(wp[k]>>4)-8; dot+=lo*(int)xb[k]+hi*(int)xb[16+k]; }
        float db; memcpy(&db,qx+(size_t)(b>>3)*292,4);
        acc=(sp[b]*db)*(float)dot+acc;
      }
      if(qy[r]!=acc){ printf("Y-DIFF r%d gpu=%f cpu=%f\n",r,qy[r],acc); ybad++; }
    }
    printf("ycheck: %s\n", ybad?"FAIL":"ALL 8 ROWS BIT-EXACT");
  }
  printf("bad=%d %s\n",bad, bad?"":"ALL PARTIALS EXACT");
  return 0;
}
