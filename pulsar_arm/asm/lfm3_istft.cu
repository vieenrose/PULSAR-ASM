/* lfm3_istft.cu -- S5 SPIKE: the LiquidCodec decoder's iSTFT + overlap-add,
   on the Jetson Nano (Maxwell sm_53). Standalone tool, gated against torch.

   WHY A SPIKE FIRST. doc/LFM3_PORT_STAGES.md orders this before the codec
   transformer, because the inverse transform + overlap-add is the one surface
   in the whole port with no precedent in this repository: if the waveform
   cannot be reproduced, the audio-output half of the port is dead and we want
   to know that in hours, not after building the decoder stack.

   WHAT IT COMPUTES (the reference is modeling_liquid_codec.py, quoted exactly):
     LiquidCodecISTFT.forward:
       frames   = torch.fft.irfft(spec, 1280, dim=1, norm="backward") * window
       audio    = F.fold(frames,   output_size=(1,(F-1)*320+1280), kernel=(1,1280), stride=(1,320))
       envelope = F.fold(window^2, ...same...)
     LiquidCodecModel.decode (streaming_cache == None):
       trim = (1280-320)/2 = 480 ; tail = 960
       out  = (audio[:, trim:-trim] / envelope[trim:-trim]).float()      -> F*320 samples
   spec = torch.polar(log_magnitude.exp(), angle) with 641 = n_fft/2+1 bins,
   so with X[k] = a_k + i b_k and theta = 2*pi*k*n/1280:
     x[n] = (1/N) [ a_0 + (-1)^n a_{N/2} + 2 * sum_{k=1}^{N/2-1} (a_k cos - b_k sin) ]
   which is what irfft with norm="backward" means (forward unnormalised, inverse
   1/N, Hermitian completion). ACCUMULATION ORDER: k ascending, one f32
   accumulator per output sample, the a_0/(-1)^n a_{N/2} terms added last.
   torch uses cuFFT/pocketfft, so parity here is an SNR gate (>= 60 dB), never a
   bit claim -- a neural codec's waveform has no bit-exactness to promise, and
   the number is recorded, not fitted.

   BUILT with the house line: -fmad=false (the documented order is fmul-then-fadd;
   contraction would change it) and NO --use_fast_math (expf stays IEEE; the fast
   arm below is the explicit __sincosf).

   TWO ARMS, because the accuracy question must be measured, not assumed:
     LFM_ISTFT_TABLE=0 (default) twiddles from __sincosf (fast-math, ~2^-21 abs)
     LFM_ISTFT_TABLE=1           twiddles from a host-built double->float table
   The gate prints both SNRs; the cheaper one that clears 60 dB is what ships.

   usage: lfm3_istft <window.f32> <in.f32> <out.f32> <frames> [table.f32]
   in.f32  : 2*641*F float32 -- F rows of 641 log-magnitudes, then F rows of 641 angles
   out.f32 : F*320 float32
*/
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <time.h>

#define NFFT 1280
#define NHOP 320
#define NBIN (NFFT / 2 + 1)            /* 641 */

__global__ void k_irfft_window(const float* logmag, const float* angle, const float* win,
                               float* frames, int F, int use_table, const float* table) {
    int f = blockIdx.x;                                    /* one block per frame */
    __shared__ float sa[NBIN], sb[NBIN];
    for (int k = threadIdx.x; k < NBIN; k += blockDim.x) {
        float m = expf(logmag[(size_t)f * NBIN + k]);
        float c, s;
        __sincosf(angle[(size_t)f * NBIN + k], &s, &c);
        sa[k] = m * c;
        sb[k] = m * s;
    }
    __syncthreads();
    float a0 = sa[0], ah = sa[NFFT / 2];
    for (int n = blockIdx.x * 0 + threadIdx.x; n < NFFT; n += blockDim.x) {
        float acc = 0.0f;
        for (int k = 1; k < NFFT / 2; k++) {
            float c, s;
            if (use_table) {
                c = table[(size_t)n * (NFFT / 2 - 1) + (k - 1)];
                s = table[(size_t)(NFFT + n) * (NFFT / 2 - 1) + (k - 1)];
            } else {
                /* 32-bit: n,k < 1280 so n*k < 1.64e6 fits; a 64-bit % here is a software
                   divide per tap and made the kernel 100x slower than its FLOP count. */
                __sincosf(2.0f * 3.14159265358979f * (float)(((unsigned)n * (unsigned)k) % NFFT) / NFFT, &s, &c);
            }
            acc += sa[k] * c - sb[k] * s;
        }
        float x = (a0 + ((n & 1) ? -ah : ah) + 2.0f * acc) / NFFT;
        frames[(size_t)f * NFFT + n] = x * win[n];
    }
}

/* fold with a fixed contribution order: output t sums frames f ascending,
   f in [ceil((t-1279)/320), floor(t/320)] -- 4 terms, no atomics, so the
   result is reproducible run to run (atomics would make the sum order vary). */
__global__ void k_fold(const float* frames, const float* win, float* out, int F, int divide) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int L = (F - 1) * NHOP + NFFT;
    if (t >= L) return;
    float a = 0.0f, e = 0.0f;
    int f0 = (t - (NFFT - 1) + NHOP - 1) / NHOP; if (f0 < 0) f0 = 0;
    int f1 = t / NHOP; if (f1 > F - 1) f1 = F - 1;
    for (int f = f0; f <= f1; f++) {
        int j = t - f * NHOP;
        a += frames[(size_t)f * NFFT + j];
        e += win[j] * win[j];
    }
    /* divide=1 -> store the ENVELOPE sum; divide=0 -> store the audio sum.
       The first version wrote `a / e` here, so the envelope buffer held the
       normalised waveform and main's division returned the envelope itself
       (out == env, and NaN where the envelope was 0). Caught by the SNR gate:
       the symptom was out == the reference's envelope column. */
    out[t] = divide ? e : a;
}

/* POSIX read/write, not stdio: CUDA's cstdio declares device-side fopen/fread/
   fwrite overloads that win name lookup in host code and then fail with "too few
   arguments in function call" (CUDA 10.2). Documented here because the message
   points at the wrong line. */
static float* readf(const char* p, size_t n) {
    int fd = ::open(p, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open %s\n", p); exit(2); }
    float* b = (float*)malloc(n * 4);
    size_t got = 0;
    while (got < n) { ssize_t r = ::read(fd, b + got, (n - got) * 4); if (r <= 0) break; got += r / 4; }
    ::close(fd);
    if (got != n) { fprintf(stderr, "short read %s (%zu/%zu)\n", p, got, n); exit(2); }
    return b;
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <window.f32> <in.f32> <out.f32> <frames> [table.f32]\n", argv[0]); return 2; }
    int F = atoi(argv[4]);
    int use_table = (argc > 5) ? 1 : 0;
    const char* env = getenv("LFM_ISTFT_TABLE");
    if (env) use_table = atoi(env) && (argc > 5);
    size_t nin = (size_t)2 * NBIN * F, L = (size_t)(F - 1) * NHOP + NFFT;
    float *win = readf(argv[1], NFFT), *in = readf(argv[2], nin);
    float *tab = use_table ? readf(argv[5], (size_t)2 * NFFT * (NFFT / 2 - 1)) : NULL;

    float *d_win, *d_in, *d_tab, *d_frames, *d_audio, *d_env, *d_out;
    size_t ntab = use_table ? (size_t)2 * NFFT * (NFFT / 2 - 1) * 4 : 0;
    if (cudaMalloc(&d_win, NFFT * 4) || cudaMalloc(&d_in, nin * 4) || cudaMalloc(&d_tab, ntab ? ntab : 4) ||
        cudaMalloc(&d_frames, (size_t)F * NFFT * 4) || cudaMalloc(&d_audio, L * 4) ||
        cudaMalloc(&d_env, L * 4) || cudaMalloc(&d_out, L * 4)) {
        fprintf(stderr, "cudaMalloc failed (F=%d, frames=%.1f MB)\n", F, (double)F * NFFT * 4 / 1e6);
        return 3;
    }
    cudaMemcpy(d_win, win, NFFT * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, in, nin * 4, cudaMemcpyHostToDevice);
    if (use_table) cudaMemcpy(d_tab, tab, ntab, cudaMemcpyHostToDevice);

    int thr = 256;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_irfft_window<<<F, thr>>>(d_in, d_in + (size_t)NBIN * F, d_win, d_frames, F, use_table, d_tab);
    k_fold<<<(int)((L + thr - 1) / thr), thr>>>(d_frames, d_win, d_audio, F, 0);
    k_fold<<<(int)((L + thr - 1) / thr), thr>>>(d_frames, d_win, d_env, F, 1);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    float* host = (float*)malloc(L * 4);
    cudaMemcpy(host, d_audio, L * 4, cudaMemcpyDeviceToHost);
    float* env_h = (float*)malloc(L * 4);
    cudaMemcpy(env_h, d_env, L * 4, cudaMemcpyDeviceToHost);

    if (argc > 6) {                                     /* debug: dump irfft*window frames */
        float* fr = (float*)malloc((size_t)F * NFFT * 4);
        cudaMemcpy(fr, d_frames, (size_t)F * NFFT * 4, cudaMemcpyDeviceToHost);
        int fd2 = ::open(argv[6], O_WRONLY | O_CREAT | O_TRUNC, 0644);
        size_t put2 = 0, want2 = (size_t)F * NFFT;
        while (put2 < want2) { ssize_t w = ::write(fd2, fr + put2, (want2 - put2) * 4); if (w <= 0) break; put2 += w / 4; }
        ::close(fd2);
    }
    if (argc > 7) {                                     /* debug: dump folded audio + envelope */
        float* au = (float*)malloc(L * 4); float* ev = (float*)malloc(L * 4);
        cudaMemcpy(au, d_audio, L * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(ev, d_env, L * 4, cudaMemcpyDeviceToHost);
        int fd3 = ::open(argv[7], O_WRONLY | O_CREAT | O_TRUNC, 0644);
        size_t p3 = 0; while (p3 < L) { ssize_t w = ::write(fd3, au + p3, (L - p3) * 4); if (w <= 0) break; p3 += w / 4; }
        ::close(fd3);
        int fd4 = ::open(argv[8], O_WRONLY | O_CREAT | O_TRUNC, 0644);
        size_t p4 = 0; while (p4 < L) { ssize_t w = ::write(fd4, ev + p4, (L - p4) * 4); if (w <= 0) break; p4 += w / 4; }
        ::close(fd4);
    }
    int trim = (NFFT - NHOP) / 2;                       /* 480; out = F*320 samples */
    float* o = (float*)malloc((size_t)F * NHOP * 4);
    for (long t = 0; t < (long)F * NHOP; t++)
        o[t] = host[t + trim] / env_h[t + trim];
    int fd = ::open(argv[3], O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { fprintf(stderr, "open %s\n", argv[3]); return 2; }
    size_t want = (size_t)F * NHOP, put = 0;
    while (put < want) { ssize_t w = ::write(fd, o + put, (want - put) * 4); if (w <= 0) break; put += w / 4; }
    ::close(fd);
    if (put != want) { fprintf(stderr, "short write %s\n", argv[3]); return 2; }
    double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("ISTFT F=%d out=%ld table=%d gpu_ms=%.3f ms_per_frame=%.3f\n",
           F, (long)F * NHOP, use_table, ms, ms / F);
    return 0;
}
