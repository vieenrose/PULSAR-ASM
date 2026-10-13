# gen_v32.py: PTX UNCHANGED from v29/v31 (engine-only host-path change).
# v32 change over v31: persistent PAGE-LOCKED staging buffers for the per-call copies.
#  cuMemHostAlloc(&h_in, XQ_CAP=64KB) + cuMemHostAlloc(&h_out, Y_CAP=512KB) at init.
#  H2D: memcpy(h_in, xq) then cuMemcpyHtoDAsync(d_xq, h_in).
#  D2H: cuMemcpyDtoHAsync(h_out, dev_y) then cuCtxSynchronize, then memcpy(y, h_out).
#  Rationale: pageable copies cost ~63us (D2H) / 28us (H2D) latency via driver staging;
#  pre-pinned buffers DMA directly. Fallback: pageable async (v31) or sync path.
