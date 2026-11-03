# Ideas backlog — RE-USE RTF

Ordered by expected payoff. Verify each with the single-block harness
(`tests/gpu_block_bench.c`, ~2 s/block) before a full measure.

## Tower (97% of model time)
1. **Register-blocked GEMM + double buffering for `in_proj`/`x_proj`.**
   in_proj is 5.4 GFlop at 163 GFLOPS; a plain 32x32 tile gave no gain, so it
   needs >=4 accumulators/thread (8x8 per thread), `__ldg`, and cp.async-free
   double buffering (CUDA 10.2, sm_53 safe). Target 400+ GFLOPS.
2. **Fuse a whole scan direction**: in_proj -> causal depthwise conv1d ->
   x_proj -> dt_proj -> scan -> gated out_proj in one kernel per
   (s, channel-tile), keeping the L=256 chain in registers/shared.
   Current: 6 launches + 6 full round-trips of SL*DINNER floats (~21 MB each at
   8 kHz). Massive traffic + launch-count win; launch count matters on the Nano.
3. **`k_scan` uses 32-thread blocks** (grid (S, DINNER/32), 1 warp/block ->
   648 warps for the whole tensor: latency-bound). Make blocks 256 threads
   (8 channels each), unroll over L, and consider a chunked/decoupled-lookback
   scan to parallelise over L (L=256 is a serial chain today).
4. **Concat + output_proj GEMV + LayerNorm fusion.** Three passes over
   SL*2*HID floats; each row is independent (2*HID = 128 per row) -> one block
   per row, keep the row in shared memory, do the LN in the same block.
5. **Residual scatter transposes** (`k_resid_scatter_freq/_out`, 0.14 s
   windows each): they exist because TFMamba needs both [C,T,F] and [T,F,C]
   layouts. Produce/consume one layout consistently, or use a shared-memory
   tiled transpose (32x32, padded) instead of the per-element div/mod version.
6. **Two streams for the time/freq branches** (independent within a block):
   helps the dev host; on the Nano (serialised) fusion matters more.
7. **CUDA graphs / launch reduction for the 30-block loop.** On the Nano,
   launches are serialised by CUDA_LAUNCH_BLOCKING; per-launch savings scale
   directly into Nano RTF.

## Memory / weights
8. **Store projection weights pre-transposed in the blob** so B-tiles load
   coalesced (blob layout is ours; regenerate with tools/mkblob.py). Removes
   the strided W reads in every projection.
9. **fp16 weight storage** (38.6 MB -> 19 MB) for bandwidth-bound kernels.
   Needs its own gate (SNR >= 30 dB + perceptual delta <= 0.02) and must not
   touch the fp32 gate path. Maxwell has no fp16 throughput advantage, so this
   only pays if a kernel is bandwidth-bound — measure first.

## Front/end
10. **GPU STFT** (DFT-GEMM, same shape as the GPU iSTFT): CPU 0.081 s at 8 kHz
    (RTF 0.016), 0.36 s at 16 kHz. Small on the dev host, 0.44 s on the Nano.
11. **Fuse build_input** (mag/pha -> xin) into the STFT epilogue.

## Structural (bigger swings, more risk)
12. **Persistent kernel** for the whole tower (one block per SM, work queue of
    (block, direction, channel-tile) items): removes ~3600 launches.
13. **Compensated fp32 (two-float) accumulation** to replace double in
    InstanceNorm / iDFT hot loops: on the Nano fp64 is 1/64 rate, so the iSTFT
    and InstanceNorm double accumulators are expensive there. Verify gates.
14. Workspace: the arena resets per stage; check peak usage to enable a smaller
    Nano arena (16 kHz needs 1.1 GiB today, which limits what else fits).

## Measured: multi-stream / multi-process scaling (dev host, 8 kHz, 2 reps)
Per-stream RTF_total: 1 stream 0.439 | 2 streams 0.932 | 4 streams 1.860 | 8 streams ~2.7 (2/8 OOM at 640 MB arena each).
Aggregate RTF (wall/total audio): 0.439 / 0.962 / 0.953 / 0.76 -> the GPU saturates at ~1 stream;
extra processes only add CUDA context switches. RTF<1 is reachable at 2 concurrent streams, not 4+.
15. **Batch>1 (8 clips in one process, batch dim in the GEMMs/scan).** The ONLY form of "8 parallel"
    that buys throughput: 8x work per launch, weights + launch overhead amortised, GEMM M dim 8x taller.
    Highest-value Nano item (launches are serialised by CUDA_LAUNCH_BLOCKING there). Needs batch>1
    threaded through the model graph (cfg says batch=1 today).
16. **GPU STFT is written and validated in isolation** (`reuse_gpu_stft` in reuse_gpu.cu +
    k_stft_prep/k_stft_dft/k_stft_head, opt-in REUSE_GPU_STFT, driver wired): 0.080 -> 0.010 s
    (RTF 0.0157 -> 0.0020, ~3.5% of rtf_total; ~0.43 s on the Nano). Bit-identical to the CPU
    reference (max|d|=0 on the real fixture) when run standalone. BLOCKED: in-app, 999/164381 mag
    bins and 504 phase bins differ (max 1.04 / 3.14 = pi flips) and end-to-end SNR vs the CPU-STFT
    run is 26.4 dB (deterministic, rep-independent, malloc-probe ruled out allocation). Next debug:
    compare k_stft_dft's re/im in-app (dump dre/dim), check the tile grid (64,11) coverage vs the
    isolated harness, and print wsize (app: nfft=320 hop=40 wsize=320 F=161 T=1021 nf=40800).
