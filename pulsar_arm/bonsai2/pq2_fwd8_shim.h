/* pq2_fwd8_shim.h — run fwd8.c on the EXACT PQ2_0 device path, zero edits.
 * Compile fwd8.c with: -include pq2_fwd8_shim.h and link pq2_fwd8_glue.c
 * (instead of pq2_gemv.c, which needs ARMv8.2+dotprod / vdotq_s32 and does
 * not exist on A57). The shim renames fwd8's pq2_gemv() call to the exact
 * compat entry, which uses fwd8's own Q8_K activations (no requant) and the
 * sm_53 SWAR kernel (bit-exact vs the fork-faithful reference).
 */
#ifndef PQ2_FWD8_SHIM_H
#define PQ2_FWD8_SHIM_H
#include <stdint.h>
void pq2_pq2_compat_gemv(int rows, int cols, const unsigned char *w,
                         const float *x, float *y, const unsigned char *xq);
#define pq2_gemv pq2_pq2_compat_gemv
#endif
