/* bq1_fwd8_shim.h — Nano port of fwd8.c to the BQ1 binary path, zero edits.
 * Compile fwd8.c with: gcc ... -include bq1_fwd8_shim.h ... (and do NOT
 * compile/link pq2_gemv.c, which needs ARMv8.2+dotprod absent on A57).
 * The shim renames fwd8.c's pq2_gemv() call (and its prototype) to the
 * compat entry in bq1_fwd8_glue.c, which lazily repacks PQ2_0 -> BQ1 split
 * (sign projection), quantizes from the float activations directly, and
 * dispatches to CUDA (sm_53) or A57 NEON. q8_K_quant still runs (harmless,
 * keeps the XQ path identical); compat ignores its output.
 */
#ifndef BQ1_FWD8_SHIM_H
#define BQ1_FWD8_SHIM_H
void bq1_pq2_compat_gemv(int rows, int cols, const unsigned char *w,
                         const float *x, float *y, const unsigned char *xq);
#define pq2_gemv bq1_pq2_compat_gemv
#endif

/* CUDA driver warmup: force libcuda init at process start, before the 2GB
 * GGUF mmap + repack traffic. Works around a Tegra libcuda/tcache
 * interaction that segfaults first cuMemAlloc late in startup. */
#ifdef BQ1_USE_CUDA
#include <cuda_runtime.h>
__attribute__((constructor)) static void bq1_cuda_early_init(void) {
  cudaFree(0);
}
#endif

