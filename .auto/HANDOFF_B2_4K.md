# Bonsai-2 4B at 4k context on the Jetson Nano: limits and plan

State at 9ce1479 (2026-10-16, branch `bonsai-jetson-nano`). Engine `pulsar_arm/bonsai2/fwd35_sm53.cu`
(Ternary-Bonsai-2-4B-PTQ1_0, bit-exact vs the PrismML fork). Harness `.auto/measure_b2_4k.sh`:
3968-token zh-TW prompt + 64 greedy tokens, exact f16 K/V at ctx 4096, fork golden
`golden_b2_4b_4k.txt`, score = sqrt(prefill_tps x decode_tps). Gates `.auto/checks_b2.sh 2B 4B`.
History: `.auto/ideas.md` rows 59-61. TurboQuant and MTP/DFlash are out of scope for Bonsai-2 by
the user's decision; do not re-propose them.

| | prefill tps | decode tps | score | peak RSS |
|---|---|---|---|---|
| now (9ce1479) | 11.37 | 9.25 | 10.26 | 200 MB |
| realistic exact target | 21-26 | 12-12.5 | ~16-18 | <= 200 MB |
| bandwidth / roofline ideal | ~40-50 (non-exact fp16 GEMM at peak) | 16.6 (19.0 at the 25.6 GB/s spec) | ~26 | |

## Hardware ceilings (measured, `pulsar_arm/bonsai2/bench_sm53*.cu`, jetson_clocks, 921.6 MHz)

The Nano GPU is ONE SM (128 lanes): per-SM and whole-GPU rates are the same number.

| op | per clock | note |
|---|---|---|
| stream read | 22.3-22.4 GB/s | spec 25.6 |
| LOP3 / IADD / FFMA | 118 / 118 / 122 | full rate |
| HFMA2 | 104.5 (8 chains), ~113 (12 chains) | half2 ops, shares the FP32 pipe with FFMA |
| POPC / I2F / F2F | 31.3 / 31.4 / 31.4 | one shared quarter-rate pipe: 32/clk combined |
| SHFL | 29.6 | |
| DFMA | 3.9 | |
| LDS.128 (distinct addresses) | 7.3 | ~117 B/clk |

Mixed POPC + LOP3 streams run at the sum of their issue costs (dual issue buys ~5%), so the practical
issue ceiling is ~125 thread-instructions per clock.

## Bytes per decoded token at 4k

- weights (two bit planes + f16 scale per 128 = 2.125 bpw): layer GEMVs 948 MB, tied head 169 MB;
- K/V: 8 attention layers x 4 KV heads x 256 x (K + V) x 2 B x 4000 positions = 131 MB;
- GDN state: 24 layers x 32 heads x 128 x 128 x 4 B = 50 MB, read + written = 100 MB.

Total ~1.34 GB -> 60 ms at 22.3 GB/s -> 16.6 tok/s is the hard decode ceiling at 4k.

## Decode at position ~4000: measured vs floor (ms/token)

Measured with `nvprof --print-gpu-trace` (E_GRAPH=0; wall 110.9 ms/token under the profiler, 108.1
with graphs), split by `.auto/nvprof_split.py tr.csv 64 3968`.

| component | now | floor | realistic exact | lever |
|---|---|---|---|---|
| layer GEMVs: K=2560 x64 37.7, K=9216 x32 17.0, K=4096 x32 6.8 | 61.5 | 42.5 (bytes); 8-POPC body issue floor ~41 | ~48 | P1 |
| head GEMV (K=2560, 248320 rows; already 19.6 GB/s) | 8.6 | 7.6 | ~8 | - |
| attention: k_fvg 14.0, k_fs 7.1, k_fw 2.1, k_qk 0.4 | 23.6 | ~9.5 (K/V bytes 5.9 + V-chain f16 rounding on the quarter pipe) | ~17 | P5-P7 |
| GDN: k_gdn 6.4, k_ab 1.4, k_conv 0.75, k_gnorm 0.6, k_l2 0.5 | 9.6 | ~5 (state bytes 4.5) | ~6.5 | P8 |
| quant + norm: k_xq 2.4, k_norm_xq 1.1 | 3.5 | ~1 | ~2 | P8 |
| argmax, launch gaps (graphs on) | ~1.2 | ~0.2 | ~1 | - |
| **total** | **108** | **~60** | **~80-84 (12-12.5 tok/s)** | |

k_ssq (exact rms sum, 23.9 ms/token) runs concurrently with the residual GEMV on the second stream
and is not in the sum.

The V chain in k_fvg is the fork's f16 accumulator, `vk = f16(fma(v, w, vk))`, i.e. two quarter-pipe
conversions per (position, dim, head): 8 layers x 16 heads x 256 dims x 4000 positions = 131 M updates
per token. With the V element conversion shared by 2 heads, that is ~11 ms of quarter-pipe time on its own, and
k_fvg sits at ~80% of it.

## Prefill (positions 0..3967, chunk 1): measured vs floor (ms/token, averages)

| component | now | realistic exact | lever |
|---|---|---|---|
| layer GEMVs (no head) | 61.8 | 25-35 | P2 |
| attention (average position ~2000): k_fvg 7.1, k_fs 3.7, k_fw 1.2, k_qk 0.4 | 12.3 | ~7-8 | P4-P7 |
| GDN: k_gdn 6.5, k_ab 3.4, conv/gnorm/l2 1.9 | 11.7 | ~2-3 | P3, P8 |
| quant + norm | 3.5 | ~2 | P8 |
| **total** (87.9 wall with graphs) | **89.3** | **~38-47 (21-26 tok/s)** | |

Prefill is compute-bound once a chunk shares each weight load: 111.5 M words (32 weights) per token.
- An HFMA2 GEMM costs ~26 issue per word-token. It is exact because 16 ternary x int8 products are
  integers of at most 2032, and fp16 holds integers up to 2048 exactly. That gives ~25 ms.
- A batched POPC GEMM (8-POPC body, activations broadcast from shared memory) costs ~36 issue per
  word-token, ~35 ms.
- Bandwidth at chunk 8 is ~5 ms/token and does not bind.

## Why the decode GEMV is not at the bandwidth floor (ideas row 61)

The 8-POPC body (ideas row 61) cut GEMV instructions 20% and time 9%. `nvprof --metrics` (E_GRAPH=0)
on the three GEMV shapes:

| shape | occupancy | issue-slot use | mem-dependency | exec-dependency | sync |
|---|---|---|---|---|---|
| K=2560 `<20,5,1,3,3,1>` | 31% | 57% | 36% | 28% | 22% |
| K=9216 `<72,9,1,2,1,0>` | 42% | 52% | 23% | 19% | 42% |
| K=4096 `<32,1,8,4,2,1>` | 37% | 50% | 51% | 28% | 0% |

The issue floor (~42 instructions per word) now coincides with the byte floor. What is left is latency:
- the shared tile and the serial warp-0 chain cost 22-42% sync stalls;
- 96 registers per thread, which holds occupancy to 31-42%, leave too few bytes in flight.

## Ranked plan

Every step keeps the fork's integer sums and float chain order, passes `checks_b2.sh 2B 4B` and
the 4k golden, keeps RSS at or below ~200 MB, and is decided by a same-job ABBA.

1. **P1, decode GEMV latency (61.5 -> ~48 ms; decode ~+14%, chunk-1 prefill ~+18%).** DONE as k_li (ideas row 63): bit-exact, needed the 32-row tiled layout (row-major lane = row was 2x slower); GEMVs 19-20 GB/s, 4B decode +7%, 4k score 11.98 -> 12.57.
   - Design: lane = row. Repack each plane so that word k of the 32 rows of a warp is contiguous,
     giving coalesced LDG.128.
   - Keep the activation planes in shared memory, read as broadcast LDS.128 (2 per word).
   - Each lane runs its own row's group-order chain `sumf = fma(d, sumi, sumf)`. That removes the
     tile, the barrier and the serial warp-0 chain, and frees the `nq[8][4]` registers for more
     warps and deeper prefetch.
   - Prototype at K=2560 standalone first; kill the idea if it cannot beat `<20,5,1,3,3,1>` there.
2. **P2, batched prefill GEMM (61.8 -> 25-35 ms; prefill ~+50-60%).** DONE as k_hg (ideas row 62): GEMVs 1.5-1.67x at chunk 8, 4k prefill 11.37 -> 15.54 tps. The fp16 form's floor is ~0.8 issue per weight-token, not 0.5; what follows is the original plan.
   - Weights are decoded to half2 once per chunk (PRMT / shift unpack).
   - Activations are int8 values as fp16, broadcast from shared memory.
   - Per word-token: 16 HFMA2, then lo + hi through 2 F2F + FADD, then the fork chain.
   - Alternative: a batched POPC kernel with lane = row and shared-memory activations.
   - The existing `k_gemvN` loses at chunk > 1 (row 60: 5.2 / 6.2 / 7.0 vs 11.9 tps) because it
     re-reads the activation planes from global per token (~20 B per word-token), keeps RB = 2,
     and keeps the serial tile chain. Fixing those three is the point.
3. **P3, chunked GDN prefill (k_gdn 6.5 -> ~1.5 ms).** DONE as k_gdnc + k_abt (ideas row 64): 4k prefill 16.16 -> 17.76, score 13.17.
   - Each state row stays in registers across the chunk's tokens. Rows are independent and the
     per-row arithmetic is unchanged.
   - Also batch k_ab: it costs 3.4 ms in prefill vs 1.4 in decode.
4. **P4, batched prefill attention (12.3 -> ~8 ms).** PARTLY DONE (ideas rows 72-73): k_fsb shares K across the chunk with a transposed reduce (k_fs 2.4x), k_fvg straight-line groups + 4 heads per V conversion for chunks; V still re-read per token.
   - Share the K/V loads and the K/V f16 -> f32 conversions across the chunk.
   - The per-token V-chain rounding cannot be shared, so the gain is bounded at about 1/3 of the
     conversion work.
5. **P5, k_fs score reduction (7.1 -> ~4 ms at 4k).** MEASURED OUT (loop row 9: 0.82 -> 0.88 ms). Alone it loses; inside the chunk-batched k_fsb it is the main win (row 72).
   - Use a shuffle transpose-reduce (31 SHFL per 32 scores instead of 5 per score) that yields
     the same xor-butterfly tree bits.
   - The smem-tree variant was measured out in row 60; this one is different and untested.
6. **P6, k_fvg pipe balance (14.0 -> ~11-12 ms, uncertain).** MEASURED OUT (ideas row 66): the mixed ALU/F2F split is 0.90x at 1 of 4 chains, 0.46x at 3-4; k_fvg is issue bound.
   - Round part of the V chains on the FP32 pipe (Veltkamp split plus a magic-add for the
     subnormal range, certified by an exhaustive 2^32 check, no fallback) and the rest with F2F.
   - Row 60 measured the all-ALU unchecked rounding at only -6%, so only the balanced split is
     worth trying.
7. **P7, certified fp32 expf in k_fw (2.1 -> ~0.5 ms at 4k).** DONE as fw_expf (ideas row 67): exhaustively certified for x <= 0, score 13.52 -> 13.58. The FP64 glibc-exact expf binds it
   (row 60). It needs a proof or an exhaustive check over the input range that occurs.
8. **P8, small-kernel fusion (~2-3 ms).** Candidates: k_xq / k_norm_xq into the producing kernels,
   and the GDN side kernels (k_conv, k_gnorm, k_l2).

Expected landing: P1 + P5-P8 give decode ~12-12.5 tok/s; P1 + P2-P4 give prefill ~21-26 tok/s;
score ~16-18.

## Measured out: do not re-walk (ideas rows 59-60)

- GEMV shape sweeps (E_V16/20/32/48/72/96).
- k_fag grouped attention.
- Fused k_gnorm -> k_xq.
- Veltkamp f16 rounding in k_fv with exact fallback (slower); unchecked (not exact).
- Conversion-free V via the h2f trick (needs a sticky nonfinite flag to be exact).
- Smem-tree k_fs reduction.
- Conversion-free K in k_fs.
- Parallel prefix-max in k_fw.
- On-GPU embed / pipelined decode (0.2-0.5%).
- k_ab with more loads in flight.
- Software-pipelined k_gdn loads (-0.6%).
- Batched N > 1 decode, MTP/DFlash, TurboQuant (out of scope).
- CPU co-execution: Tegra's pinned memory is CPU-uncached and RSS would grow.

## Recipes

- **Stall reasons.** Graphs hide kernels from `--metrics`:
  `sudo -n env E_GRAPH=0 nvprof --kernels "::k_gemv:5" --metrics achieved_occupancy,issue_slot_utilization,stall_memory_dependency,stall_exec_dependency,stall_sync ./fwd35 <gguf> 3 1 9707`.
- **Per-token breakdown.** A full 4k trace is ~390 MB of CSV and ~7 min on the board:
  `sudo -n env E_GRAPH=0 nvprof --print-gpu-trace --csv --normalized-time-unit us --log-file tr.csv ./fwd35 <gguf> 64 $(cat prompt_b2_4k_ids.txt)`,
  then `python3 .auto/nvprof_split.py tr.csv 64 3968`.
- **Short ABBA.** 47-token decode: `fwd35 <gguf> 48 1 9707`. 400-token prefill: the first 400 ids of
  `p4k.txt` with gen 2. One sample is worth ~±0.5% on this board under jetson_clocks.
- **SASS.** `cuobjdump -sass fwd35 | grep -c POPC` is 2592 at 9ce1479. Compare builds by the hash
  of the SASS with the encodings stripped.
