# Bonsai ternary port (C + NEON, CPU-only)

Forward passes for the PrismML ternary checkpoints, validated end to end
against the fork's own server and operators. Lives here (not on the Pi):
it needs gigabytes of RAM, 20 threads and ARMv8.2+dotprod, i.e. the DGX
Spark. The asm engine in `pulsar_arm/asm` stays Pi-shaped; this port is
plain C with one NEON kernel family, no dependencies beyond the fork's
`libggml` (linked only for quantization helpers and validation oracles).

| file | what |
|---|---|
| `fwd.c` | Ternary-Bonsai-2-27B forward: embed → 64 layers → head → argmax/sample |
| `fwd8.c` | Ternary-Bonsai-8B forward: 36-layer dense Qwen3-GQA → argmax/sample |
| `tq_gemv_neon.c` | NEON PTQ1_0 GEMV (vectorized digit decode + SDOT), OpenMP over rows |
| `pq2_gemv.c` | NEON PQ2_0 × Q8_K GEMV (2-bit decode + SDOT) |
| `tq_gemv_int.c` | scalar integer-path reference (bit-exact, slow — kept as oracle) |
| `tq_xgemv.c` | scalar float-path reference (signs → FWHT → dot) |
| `fwht.S` | FWHT-1024 asm, 1/√1024 prescale, bit-exact vs the fork |
| `gdn_test.c`, `rope_test.c` | GDN + MRoPE transcriptions validated vs fork ops |
| `tq_probe.c`, `signs_test.c`, `fw_test.c`, `digit_math.c` | decoder/sign/FWHT/digit proofs |
| `gguf_map.py` | GGUF tensor inventory (offsets, shapes, types) |
| `sign_extract.py` | `prism.hadamard` sign tables → float32 bins |
| `tokdetok.cpp` | fork-API tokenizer/detokenizer (prompt ids, transcript text) |
| `dump_stage_rms.cpp` | public-API oracle: per-tensor magnitudes/values from the fork graph |

## Build (aarch64, ARMv8.2+dotprod)

```sh
LB=<llama.cpp>/build/bin   # fork libs: libggml-base, libggml-cpu
# sign tables once (float32, from the checkpoint itself):
python3 sign_extract.py Ternary-Bonsai-2-27B-PTQ1_0.gguf  # -> /tmp/sign{5120,6144,17408}.bin
# 27B engine:
gcc -O2 -fopenmp -march=armv8.2-a+dotprod -DTQ_XGEMV_LIB -Dmain=tq_neon_main \
  -c tq_gemv_neon.c -o fwd_neon_mt.o
gcc -O2 -fopenmp -march=armv8.2-a+dotprod -c fwd.c -o fwd.o
gcc -O2 -fopenmp -o fwd fwd.o fwd_neon_mt.o fwht.S \
  -L$LB -lggml-base -lggml-cpu -Wl,-rpath,$LB -lm
# 8B engine:
gcc -O2 -march=armv8.2-a+dotprod -Dmain=pq2_main -c pq2_gemv.c -o fwd8_pq2.o
gcc -O2 -march=armv8.2-a+dotprod -c fwd8.c -o fwd8.o
gcc -O2 -o fwd8 fwd8.o fwd8_pq2.o \
  -L$LB -lggml-base -lggml-cpu -Wl,-rpath,$LB -lm
```

## Run

```sh
# prompt ids come from the checkpoint's own template+tokenizer (tokdetok):
./tokdetok tokstr <model.gguf> <prompt words...>   # -> ids (add_bos=false)
# greedy (deterministic, the reference behaviour):
OMP_NUM_THREADS=20 taskset -c 0-19 ./fwd <27B.gguf> <ids...> --gen 200
# sampled (fork chain order: top-k -> top-p -> min-p on raw logits, temp last):
./fwd <27B.gguf> <ids...> --gen 200 --sample 0.5 0.85 20 <seed> 0.05
./fwd8 <8B.gguf> <ids...> --gen 150 --sample 0.6 0.95 20 <seed> 0.05
./tokdetok tok <model.gguf> <ids...>               # ids -> text
```

## Validation ledger

- **PTQ1_0 decoder** bit-exact vs fork dequantize (real rows).
- **FWHT-1024 / signs+FWHT** bit-exact (incl. prescale; involution clean).
- **Full projection** (signs → FWHT → dot, real 89M-weight matrix): maxrel 0.000
  over all 17408 rows.
- **Integer GEMV**: scalar path bit-exact vs fork `vec_dot`
  (fp32 accumulation order mirrored; a subtracted-correction theory failed on
  all 64 rows and was deleted); **NEON path bit-exact, 26.0 ms** per
  projection (0.29 ns/w) — 5.3× over scalar-int, 5.9× over float.
- **PQ2_0 GEMV** bit-exact vs fork (scale-first 34B blocks; values -1..+2),
  7.3 ms per 50M-weight projection (0.15 ns/w).
- **GDN**: single step 2.1e-6, final state exact 0.0; 3-token drift rel 1.4e-6
  (SIMD dot order only).
- **MRoPE** (27B): maxabs 4.47e-07. **YaRN** (8B): magnitude 1+0.1·ln4 verified
  via live-model cache/norm ratio; rope bugs that greedy survives were caught
  by distribution comparison, not argmax.
- **End to end**: 17-token greedy stream exact vs server temp-0 (27B);
  5-token greedy exact (8B); full **top-20 distribution match** at a sampled
  position (19/20 same ids, same order).
- **Per-head L2 over 128** on linear q/k (not joint-4096); `[hd,nk,rep]` →
  `[hd,rep,nk]` permute before `ssm_out` (`gdn_v_grouped=1`); fused wq split
  is per-head interleaved, not halved; conv kernel is `ker[c*4+t]` over
  `[cached×3, current]` — each of these was a real bug the oracle diff
  caught (see session notes in git history).

## Geometry (27B, from fork source)

64 layers (full attention where `(il+1)%4==0`, else linear GDN); hid 5120;
inter 17408; 24×256 heads, 4 kv; vocab 248320; untied 248320-row head;
norm eps 1e-6; linear state 150 MB + conv ring 5.9 MB; KV cache per full
layer. 8B: 36 dense layers, 4096/12288, 32+8 heads, vocab 151669, YaRN
(factor 4, base 1e6), per-head QK norms, `ffn_norm` naming.

## Caveats

- Q8 activation quantization errs up to ~4%/row on real data: validate GEMVs
  against identical Q8 inputs (bit-exact), never across quant boundaries.
- The engines are single-purpose (fixed dims via table lookup, CTX caps for
  caches); generality was traded for auditability, same as `core.S`.
- Bonsai 1.7B/4B were evaluated and set aside over output quality; the 8B and
  27B above are the demo line.
