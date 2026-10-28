/* lfm3_conv_test.c -- parity gate for the AArch64 NEON gated short conv.
 *
 * Build and run ON THE NANO (Cortex-A57):
 *   as -o lfm3_conv.o lfm3_conv.S
 *   gcc -O2 -o lfm3_conv_test lfm3_conv_test.c lfm3_conv.o -lm
 *   ./lm3_conv_test
 *
 * Gate: the NEON path must be BIT-IDENTICAL to the scalar reference, not
 * merely close. Both accumulate taps k = 0..6 ascending, so any difference is
 * a real bug (a wrong tap offset, a bad stride, a tail mishandled), not fp
 * reassociation. Sizes cover LFM3's real H=2048 plus odd sizes that exercise
 * the scalar tail, and a size that is not a multiple of 16 so the 64-byte
 * shift loop's tail is exercised.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define W 7

void lfm3_conv_step(float *out, const float *state, const float *wT,
                    const float *Cgate, long H);
void lfm3_conv_shift(float *state, long H);
void lfm3_conv_step_r(float *out, const float *const *srow,
                      const float *wT, const float *Cgate, long H);

/* Scalar reference: same operation order as the kernel. */
static void ref_conv(float *out, const float *state, const float *wT,
                     const float *Cgate, long H)
{
    for (long c = 0; c < H; c++) {
        float acc = 0.0f;
        for (int k = 0; k < W; k++)
            acc += state[(long)k * H + c] * wT[(long)k * H + c];
        out[c] = Cgate[c] * acc;
    }
}

static unsigned st = 2463534242u;
static float rnd(void)
{
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    return (float)((int)(st >> 8) % 20001 - 10000) / 1000.0f;
}

static int run(long H)
{
    size_t n = (size_t)W * H;
    float *state = malloc(n * 4), *wT = malloc(n * 4);
    float *xg = malloc(H * 4), *Cg = malloc(H * 4);
    float *a = malloc(H * 4), *b = malloc(H * 4);
    float *state2 = malloc(n * 4);
    if (!state || !wT || !xg || !Cg || !a || !b || !state2) { printf("OOM\n"); return 2; }

    for (size_t i = 0; i < n; i++) { state[i] = rnd(); wT[i] = rnd(); }
    for (long i = 0; i < H; i++) { xg[i] = rnd(); Cg[i] = rnd(); }

    /* shift, then write the new B*xg into slot 6 -- the engine's contract */
    memcpy(state2, state, n * 4);
    lfm3_conv_shift(state2, H);
    memcpy(state2 + (size_t)(W - 1) * H, xg, H * 4);

    /* scalar reference does the same shift in C */
    float *state_ref = malloc(n * 4);
    memmove(state_ref, state + H, (size_t)(W - 1) * H * 4);
    memcpy(state_ref + (size_t)(W - 1) * H, xg, H * 4);

    ref_conv(a, state_ref, wT, Cg, H);
    lfm3_conv_step(b, state2, wT, Cg, H);

    long bad = 0; double maxabs = 0;
    for (long i = 0; i < H; i++) {
        if (a[i] != b[i]) bad++;
        double d = fabs((double)a[i] - b[i]);
        if (d > maxabs) maxabs = d;
    }
    int strict = (H >= 4);
    /* H < 4 has no full vector block and falls to the documented UNFUSED
     * scalar path (binutils 2.30 on the Nano has no `fmaf' mnemonic), so it
     * is last-ulp from the fused reference by construction. Assert exactness
     * only where the fused vector body runs; below that, assert a tolerance
     * so a real bug still fails. Recompute `bad' from scratch here -- the
     * first pass above counts exact equality and must not leak into this. */
    if (!strict) {
        bad = 0;
        for (long i = 0; i < H; i++)
            if (fabs((double)a[i] - b[i]) > 1e-3) { bad = 1; break; }
    }
    /* verify the shift itself moved the right rows */
    long shiftbad = 0;
    for (int k = 0; k < W - 1; k++)
        for (long c = 0; c < H; c++)
            if (state2[(long)k * H + c] != state[(long)(k + 1) * H + c]) shiftbad++;

    printf("H=%-6ld conv %-14s maxabs %-9.3g shift %s\n", H,
           bad ? "MISMATCH" : (strict ? "BIT-EXACT" : "BIT-EXACT(tol)"),
           maxabs, shiftbad ? "MISMATCH" : "ok");
    free(state); free(wT); free(xg); free(Cg); free(a); free(b);
    free(state2); free(state_ref);
    return (bad || shiftbad) ? 1 : 0;
}

/* Decode-step throughput at LFM3's real shape. One conv step is T=1 over
 * H channels: it reads the 7-tap state, the 7-tap weight and the gate, and
 * writes H floats -- 8*H bytes of traffic, so bytes/step is what to divide by
 * (state+weight+gate+out = 7+7+1+1 vectors of H). The weight is L2-resident
 * in a hot loop, which is what a decode loop does too; the cold-DRAM number
 * for a full prefill comes from the streaming ceiling, not this. */
static void bench(long H, long iters)
{
    size_t n = (size_t)W * H;
    float *state = malloc(n * 4), *wT = malloc(n * 4);
    float *Cg = malloc(H * 4), *out = malloc(H * 4);
    for (size_t i = 0; i < n; i++) { state[i] = rnd(); wT[i] = rnd(); }
    for (long i = 0; i < H; i++) Cg[i] = rnd();

    struct timespec a, b;
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        clock_gettime(CLOCK_MONOTONIC, &a);
        for (long it = 0; it < iters; it++) {
            lfm3_conv_step(out, state, wT, Cg, H);
            lfm3_conv_shift(state, H);
        }
        clock_gettime(CLOCK_MONOTONIC, &b);
        double s = (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
        if (s < best) best = s;
    }
    double per = best / iters;
    double bytes = 8.0 * H;                 /* state+wT+gate+out, one row each */
    printf("bench H=%ld: %.2f us/step  %.2f GB/s  (best of 3 x %ld)\n",
           H, per * 1e6, bytes / per / 1e9, iters);
    free(state); free(wT); free(Cg); free(out);
}

/* The ring variant must be BIT-IDENTICAL to lfm3_conv_step, not merely fast:
 * same taps, same ascending order, same fused fmla. It is also checked against
 * the scalar reference so a shared bug in both asm paths cannot pass. */
static int run_ring(long H)
{
    size_t n = (size_t)W * H;
    float *state = malloc(n * 4), *wT = malloc(n * 4);
    float *Cg = malloc(H * 4), *a = malloc(H * 4), *b = malloc(H * 4);
    for (size_t i = 0; i < n; i++) { state[i] = rnd(); wT[i] = rnd(); }
    for (long i = 0; i < H; i++) Cg[i] = rnd();

    /* a ring at write index 3: tap k reads slot (3+k)%7, which is the case
     * the linear kernel cannot express -- so this exercises the table, not
     * just an alias for the same memory order. */
    const float *srow[W];
    for (int k = 0; k < W; k++) srow[k] = state + (long)((3 + k) % W) * H;

    lfm3_conv_step_r(b, srow, wT, Cg, H);
    /* reference path: linearise the same ring, then the verified kernel */
    float *lin = malloc(n * 4);
    for (int k = 0; k < W; k++) memcpy(lin + (long)k * H, state + (long)((3 + k) % W) * H, H * 4);
    lfm3_conv_step(a, lin, wT, Cg, H);

    long bad = 0; double maxabs = 0;
    for (long i = 0; i < H; i++) { if (a[i] != b[i]) bad++;
        double d = fabs((double)a[i] - b[i]); if (d > maxabs) maxabs = d; }

    /* The scalar reference must read the RING order (tap k -> slot (3+k)%7),
     * not slot k -- comparing against state[k] checks a different function. */
    float *ref = malloc(H * 4);
    ref_conv(ref, lin, wT, Cg, H);
    long bad2 = 0;
    for (long i = 0; i < H; i++)
        if (fabs((double)ref[i] - b[i]) > (H >= 4 ? 0.0 : 1e-3)) bad2++;
    free(state); free(wT); free(Cg); free(a); free(b); free(ref);
    printf("H=%-6ld ring %-12s vs step maxabs %-9.3g   vs scalar %s\n", H,
           (bad || bad2) ? "MISMATCH" : "BIT-EXACT", maxabs,
           bad2 ? "MISMATCH" : "ok");
    return (bad || bad2) ? 1 : 0;
}

/* Ring decode step: conv only, no shift -- the number to compare against the
 * 61.9 us/step of conv+shift. */
static void bench_ring(long H, long iters)
{
    size_t n = (size_t)W * H;
    float *state = malloc(n * 4), *wT = malloc(n * 4);
    float *Cg = malloc(H * 4), *out = malloc(H * 4);
    for (size_t i = 0; i < n; i++) { state[i] = rnd(); wT[i] = rnd(); }
    for (long i = 0; i < H; i++) Cg[i] = rnd();
    const float *srow[W];
    struct timespec a, b; double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        clock_gettime(CLOCK_MONOTONIC, &a);
        for (long it = 0; it < iters; it++) {
            int base = (int)(it % W);
            for (int k = 0; k < W; k++) srow[k] = state + (long)((base + k) % W) * H;
            lfm3_conv_step_r(out, srow, wT, Cg, H);
        }
        clock_gettime(CLOCK_MONOTONIC, &b);
        double s = (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
        if (s < best) best = s;
    }
    printf("bench_ring H=%ld: %.2f us/step  (best of 3 x %ld)\n",
           H, best / iters * 1e6, iters);
    free(state); free(wT); free(Cg); free(out);
}

int main(int argc, char **argv)
{
    if (argc > 1 && !strcmp(argv[1], "bench")) {
        bench(2048, 20000); bench_ring(2048, 20000); return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "ring")) {
        long sizes[] = { 2048, 768, 64, 4, 5, 7, 2047, 2050, 130 };
        int f = 0;
        for (size_t i = 0; i < sizeof sizes / sizeof *sizes; i++) f |= run_ring(sizes[i]);
        printf("%s\n", f ? "LFM3_RING_FAIL" : "LFM3_RING_OK");
        return f;
    }
    long sizes[] = { 2048, 768, 64, 4, 5, 7, 1, 2047, 2050, 130 };
    int fail = 0, n = (int)(sizeof sizes / sizeof *sizes);
    for (int i = 0; i < n; i++) {
        fprintf(stderr, "[size %ld]\n", sizes[i]);
        fflush(stderr);
        fail |= run(sizes[i]);
    }
    printf("%s\n", fail ? "LFM3_CONV_FAIL" : "LFM3_CONV_OK");
    return fail;
}
