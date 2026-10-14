/* POPC throughput probe for sm_53 (handoff lever 3). Measures ops/clk/SM for
 * POPC, LOP3 (xor-and), IADD and IMAD-ish chains with 8 independent streams
 * per thread so the number is throughput, not latency. Build on the Nano:
 *   nvcc -gencode arch=compute_53,code=sm_53 -O3 popc_probe.cu -o popc_probe
 * Clock from /sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq. */
#include <cstdio>
#include <cstdint>
#define CHAINS 8
template <int OP>
__global__ void probe(const unsigned *in, unsigned *out, int iters) {
    unsigned a[CHAINS], acc[CHAINS];
    #pragma unroll
    for (int c = 0; c < CHAINS; c++) { a[c] = in[(threadIdx.x + c * 97) & 1023]; acc[c] = c; }
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int c = 0; c < CHAINS; c++) {
            if (OP == 0) { acc[c] += __popc(a[c]); a[c] += 0x9e3779b9u; }                 /* POPC + IADD */
            if (OP == 1) { acc[c] += a[c]; a[c] += 0x9e3779b9u; }                          /* IADD + IADD */
            if (OP == 2) { acc[c] += (a[c] ^ acc[c]) & 0x55555555u; a[c] += 0x9e3779b9u; } /* LOP3 + IADD + IADD */
            if (OP == 3) { acc[c] += __popc(a[c] ^ acc[c]); a[c] += 0x9e3779b9u; }        /* LOP + POPC + IADD (T-MAC-like) */
        }
    }
    unsigned s = 0;
    #pragma unroll
    for (int c = 0; c < CHAINS; c++) s += acc[c];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
template <int OP>
static float run(const unsigned *din, unsigned *dout, int blocks, int threads, int iters) {
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    probe<OP><<<blocks, threads>>>(din, dout, iters); cudaDeviceSynchronize();
    cudaEventRecord(e0);
    probe<OP><<<blocks, threads>>>(din, dout, iters);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); return ms;
}
int main(int argc, char **argv) {
    double clk = argc > 1 ? atof(argv[1]) : 921.6e6;
    unsigned *din, *dout; cudaMalloc(&din, 4096); cudaMalloc(&dout, 1 << 20);
    unsigned h[1024]; for (int i = 0; i < 1024; i++) h[i] = i * 2654435761u; cudaMemcpy(din, h, 4096, cudaMemcpyHostToDevice);
    int blocks = 8, threads = 256, iters = 20000;   /* 8 blocks x 8 warps = 64 warps on the 1 SM */
    double n = (double)blocks * threads * iters * CHAINS;
    const char *names[4] = { "popc+iadd", "iadd+iadd", "lop3+iadd+iadd", "lop+popc+iadd" };
    float t[4] = { run<0>(din, dout, blocks, threads, iters), run<1>(din, dout, blocks, threads, iters),
                   run<2>(din, dout, blocks, threads, iters), run<3>(din, dout, blocks, threads, iters) };
    for (int k = 0; k < 4; k++) {
        double cyc = t[k] * 1e-3 * clk;
        printf("%-16s %8.3f ms  %.1f chain-steps/clk/SM\n", names[k], t[k], n / cyc);
    }
    printf("POPC alone: ~%.1f /clk/SM (popc+iadd minus iadd+iadd time), ratio to IADD rate %.2f\n",
           n / ((t[0] - t[1]) * 1e-3 * clk), (t[0] - t[1]) / (t[1] / 2));
    return 0;
}
