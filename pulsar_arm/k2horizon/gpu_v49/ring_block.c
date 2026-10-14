/* Exp42 (run #69): pinned-ring streaming init.
   The serial path is `malloc(467MB) -> repack -> cuMemcpyHtoD`: the repack pays
   ~115k first-touch faults on the fresh buffer and the copy runs afterwards
   (981 ms measured for repack+H2D, h2dp3). The ring streams instead: the repack
   writes one 32MB PINNED slot (driver-pinned, pre-touched once), cuMemcpyHtoDAsync
   pushes it on g_stream while the CPU fills the next slot; NSLOT=2 keeps one slot
   free for writes, and the producer waits on pStreamSync when both are in flight.
   WHY PINNED: cuMemcpyHtoD from pageable memory stages through a driver bounce
   buffer (~1.2 GB/s); from pinned it runs 4.0-4.5 GB/s (h2dp3, idle board).
   WHY A RING AND NOT ONE BIG BUFFER: the run #69 probe cuMemHostAlloc'd 467MB in
   a single buffer and the OOM killer took the process down (dmesg: "Out of memory:
   Kill process (k2_gexp48) total-vm:9521780kB") - on the 4 GB Tegra a 467MB pinned
   buffer + the 467MB device alloc + the 714MB blob mmap do not fit. 2 x 32MB is
   the working set. Any allocation failure falls back to the serial path.
   NOTE: free() on a cuMemHostAlloc pointer is a double free (SIGSEGV seen at
   cuda_init+3396), so slots go back ONLY through cuMemFreeHost. */
#define Q4_SLOT ((size_t)32 << 20)
#define Q4_NSLOT 2

/* Shared repack body: 18B-stride rows -> 16B blocks into DST, in the exact device
   order (per-layer k matrices, then the head). The SAME code runs in the serial
   path, so ring and serial produce byte-identical device images by construction. */
#define Q4_REPACK_INTO(DST) \
    do { \
        int i_, k_; \
        for (i_ = 0; i_ < (int)q4_layers; i_) \
            for (k_ = 1; k_ <= 8; k_) { \
                uint32_t nb = tbl_kcols[k_] >> 5, r, b; \
                const uint8_t *src; \
                uint8_t *d = (DST) + (size_t)i_ * q4_layer_dev + q4_off[k_]; \
                if (!tbl_kcols[k_]) continue; \
                src = (const uint8_t *)(uintptr_t)(lb + (uint64_t)i_ * Q4_STRIDE + tbl_koff[k_]); \
                for (r = 0; r < tbl_krows[k_]; r++) \
                    for (b = 0; b < nb; b++) \
                        memcpy(d + ((size_t)r * nb + b) * 16, \
                               src + (size_t)r * (size_t)nb * 18 + 2 + (size_t)b * 18, 16); \
            } \
        { const uint8_t *src = (const uint8_t *)(uintptr_t)oh; \
          uint32_t nb = Q4_HID >> 5, r, b; \
          uint8_t *d = (DST) + (size_t)q4_head_off; \
          for (r = 0; r < g_vocab; r++) \
              for (b = 0; b < nb; b++) \
                  memcpy(d + ((size_t)r * nb + b) * 16, \
                         src + (size_t)r * (size_t)nb * 18 + 2 + (size_t)b * 18, 16); } \
    } while (0)

static void q4_ring_fini(uint8_t **slot) {
    int i;
    for (i = 0; i < Q4_NSLOT; i++)
        if (slot[i]) {
            int rc = pMemFreeHost ? pMemFreeHost(slot[i]) : -1;
            if (rc) printf("INIT unpin rc=%d(%s) - slot leaked\n", rc, k2err(rc));
            slot[i] = 0;
        }
}

/* K2VERIFY=1: build a serial reference repack on the host, read the DEVICE image
   back, compare byte-for-byte. The 216-step gate only samples model output, so a
   streaming-layout bug could hide inside it; this proves the device bytes are the
   proven serial layout. Diagnostic path only (needs 2 x 467MB host allocations). */
static void q4_verify(uint64_t lb, uint64_t oh, size_t total, uint64_t drp) {
    uint8_t *ref = (uint8_t *)malloc(total);
    uint8_t *back = (uint8_t *)malloc(total);
    size_t i, bad = 0;
    int rc;
    if (!ref || !back || !pMemcpyDtoH) {
        printf("VERIFY skipped (alloc/dtoH)\n"); free(ref); free(back); return;
    }
    Q4_REPACK_INTO(ref);
    rc = pMemcpyDtoH(back, drp, total);
    if (rc) { printf("VERIFY dtoH rc=%d(%s)\n", rc, k2err(rc)); free(ref); free(back); return; }
    for (i = 0; i < total; i++)
        if (ref[i] != back[i]) {
            if (bad < 5) printf("VERIFY diff at %llu: ref %02x dev %02x\n",
                                (unsigned long long)i, ref[i], back[i]);
            bad++;
        }
    printf("VERIFY %s (%llu bytes, %llu diffs)\n", bad ? "FAIL" : "OK",
           (unsigned long long)total, (unsigned long long)bad);
    free(ref); free(back);
}

static int q4_ring_flush(uint64_t drp, uint64_t *out, uint8_t **slot, int *si, int *npend) {
    int rc = pMemcpyHtoDAsync(drp + *out, slot[*si], Q4_SLOT, g_stream);
    if (rc) { printf("INIT ring h2da rc=%d(%s)\n", rc, k2err(rc)); return 0; }
    *out += Q4_SLOT;
    *si = (*si + 1) % Q4_NSLOT;
    if (*npend >= Q4_NSLOT - 1) {          /* keep one slot free for writes */
        if (pStreamSync(g_stream)) return 0;
        *npend = 0;
    } else (*npend)++;
    return 1;
}

static int q4_repack_ring(uint64_t lb, uint64_t oh, size_t total, uint64_t drp) {
    uint8_t *slot[Q4_NSLOT];
    int si = 0, npend = 0, rc, i2, k2;
    size_t fill = 0, out = 0;
    uint8_t *wp;
    int wrem;                       /* BYTES left in the current slot (16B blocks) */
    double t0, t1, t2;
    if (!pMemHostAlloc || !pMemFreeHost || !pMemcpyHtoDAsync || !pStreamSync || !g_stream)
        return 0;
    memset(slot, 0, sizeof(slot));
    t0 = now_ms();
    for (i2 = 0; i2 < Q4_NSLOT; i2++) {
        void *p = 0;
        rc = pMemHostAlloc(&p, Q4_SLOT, 0);
        if (rc) {
            printf("INIT pin slot %d rc=%d(%s) -> serial fallback\n", i2, rc, k2err(rc));
            q4_ring_fini(slot);
            return 0;
        }
        slot[i2] = (uint8_t *)p;
        memset(p, 0, Q4_SLOT);      /* pre-touch: faults cost ~7 ms/MB on pinned pages */
    }
    t1 = now_ms();
    printf("INIT ring pin+pretouch %d x %llu MB %.0f ms\n",
           Q4_NSLOT, (unsigned long long)(Q4_SLOT >> 20), t1 - t0);
    wp = slot[0];
    wrem = (int)Q4_SLOT;
    for (i2 = 0; i2 < (int)q4_layers; i2++)
        for (k2 = 1; k2 <= 8; k2++) {
            uint32_t nb = tbl_kcols[k2] >> 5, r, b;
            const uint8_t *src;
            if (!tbl_kcols[k2]) continue;
            src = (const uint8_t *)(uintptr_t)(lb + (uint64_t)i2 * Q4_STRIDE + tbl_koff[k2]);
            for (r = 0; r < tbl_krows[k2]; r++)
                for (b = 0; b < nb; b++) {
                    memcpy(wp, src + (size_t)r * (size_t)nb * 18 + 2 + (size_t)b * 18, 16);
                    wp += 16; fill += 16;
                    if ((wrem -= 16) == 0) {
                        if (!q4_ring_flush(drp, &out, slot, &si, &npend)) { q4_ring_fini(slot); return 0; }
                        wp = slot[si]; fill = 0; wrem = (int)Q4_SLOT;
                    }
                }
        }
    { const uint8_t *src = (const uint8_t *)(uintptr_t)oh;
      uint32_t nb = Q4_HID >> 5, r, b;
      for (r = 0; r < g_vocab; r++)
          for (b = 0; b < nb; b++) {
              memcpy(wp, src + (size_t)r * (size_t)nb * 18 + 2 + (size_t)b * 18, 16);
              wp += 16; fill += 16;
              if ((wrem -= 16) == 0) {
                  if (!q4_ring_flush(drp, &out, slot, &si, &npend)) { q4_ring_fini(slot); return 0; }
                  wp = slot[si]; fill = 0; wrem = (int)Q4_SLOT;
              }
          } }
    t2 = now_ms();
    printf("INIT ring repack %.0f ms (%llu MB streamed)\n", t2 - t1, (unsigned long long)(out >> 20));
    if (fill) {                       /* 467MB % 32MB tail */
        if (pMemcpyHtoDAsync(drp + out, slot[si], fill, g_stream)) { q4_ring_fini(slot); return 0; }
        out += fill;
    }
    if (out != total) {
        printf("INIT ring streamed %llu != total %llu -> serial fallback\n",
               (unsigned long long)out, (unsigned long long)total);
        q4_ring_fini(slot); return 0;
    }
    if (pStreamSync(g_stream)) { q4_ring_fini(slot); return 0; }
    printf("INIT ring h2d+sync %.0f ms (ring total %.0f ms)\n", now_ms() - t2, now_ms() - t0);
    q4_ring_fini(slot);
    return 1;
}

