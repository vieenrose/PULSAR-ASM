#!/usr/bin/env python3
# gen_v31.py: PTX generator UNCHANGED from v29 (v31 is an engine-only host-path change).
# Engine change patch (v29 -> v31):
#  (1) resolve cuMemcpyHtoDAsync_v2 / cuMemcpyDtoHAsync_v2
#  (2) set the CUDA context current once in cuda_init (ctx_current flag); cuda_gemv
#      only sets it if !ctx_current and NEVER unsets it (was SetCurrent(g_ctx) + SetCurrent(0) per call)
#  (3) H2D via cuMemcpyHtoDAsync(d_xq,...,NULL) [falls back to sync pMemcpyHtoD]
#  (4) D2H via cuMemcpyDtoHAsync(...,NULL) enqueued right after pLaunch, then one pCtxSync
#      (the legacy default stream orders copy->kernel->copy, so the host never blocks mid-call)
