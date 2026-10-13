// bq1_ggml_wrap.cpp — hook the BQ1 SWAR kernel into llama.cpp ggml-cuda.
// Place next to ggml-cuda.cu and call from ggml_cuda_mul_mat_vec().
// Requires: w_bits/w_scales pre-repacked at load (see bq1_repack_*),
// x_q/x_scale produced per token by bq1_quant_row_i8 (mirror of
// fwd8.c:q8_K_quant, byte-identical nearest_int).
//
//   if (src0->type == GGML_TYPE_Q1_0 /* or custom GGML_TYPE_BQ1_0 */) {
//     ggml_cuda_bq1_mul_mat_vec(ctx, src0, src1, dst); return;
//   }
#include "bq1_wrap.h"

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern int bq1_gemv_sm53_launch(const uint32_t*, const unsigned short*,
  const signed char*, float, float*, int, int, void*);

struct Bq1DevBufs { uint32_t* bits; unsigned short* scales; signed char* xq; float* y; };

void ggml_cuda_bq1_mul_mat_vec(
  const uint32_t* w_bits_d, const unsigned short* w_scales_d,
  const float* x_f32_d, const int K, const int N,
  float* y_f32_d, void* stream) {
  // 1. quantize activations on GPU (1 block, float4 LDG.E.128).
  //    For bringup, quant on host then H2D is fine (K*1B, e.g. 4KB):
  //    bq1_quant_row_i8(x_host, xq_host, K); cudaMemcpyAsync(xq_d,...)
  // 2. GEMV (decode path, batch=1):
  //    extern int8_t*  g_bq1_xq; extern float g_bq1_xs;
  //    bq1_gemv_sm53_launch(w_bits_d, w_scales_d, g_bq1_xq, g_bq1_xs,
  //                         y_f32_d, K, N, stream);
}
#endif
