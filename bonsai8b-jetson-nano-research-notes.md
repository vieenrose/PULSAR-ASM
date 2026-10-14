# Running 1-Bit Bonsai 8B (PrismML) on Jetson Nano Gen-1 (Tegra X1 / GM20B, CC 5.3)

**Technical literature & architectural synthesis — custom CUDA kernels, PTX/SASS low-level optimization, memory management, and performance estimation**

*Research notes compiled 2026-10-14. All quantitative estimates are explicitly labeled as **derived** (with assumptions stated) or **sourced** (with citation). Items that could not be verified from public sources are flagged **[UNVERIFIED]** with a verification method.*

---

## 0. Executive Summary

**Bottom line: Bonsai 8B in true 1-bit form fits in Jetson Nano's 4 GB unified memory with real headroom, and the Tegra X1's Maxwell GM20B GPU is *architecturally well-suited* to binary inference — the absence of DP4A is largely a non-issue for a 1-bit model, because the XNOR+POPC pipeline is built from instructions (LOP3, POPC, LDS, SHF) that exist on every Maxwell part.**

Key findings:

1. **Bonsai 8B is real and confirmed.** PrismML's announced lineup — Bonsai-27B-MLX-1bit, Bonsai-8B-MLX-1bit, Bonsai-2B-MLX-1bit — are described as "fully end-to-end binary LLM" models with "all components, embeddings, attention, MLP, LM head — 1-bit", no "higher-precision escape hatches" [prismml.com/news/bonsai-8b]. The 8.2B-parameter model's weights occupy ≈ **977 MiB** (8.2×10⁹ bits ÷ 8) [derived]. **Critical gap: as of today the published checkpoints are MLX-format (Apple). No GGUF, llama.cpp, or CUDA port has been publicly released** [verified: Hugging Face prism-ml org hosts `Bonsai-27B-mlx-1bit` only]. Everything in this document therefore assumes a **custom C++/CUDA runtime with a conversion pipeline you build** (scaffold provided in §5).
2. **DP4A does not exist on Maxwell — and you don't need it.** `dp4a`/`dp2a` were introduced in **Compute Capability 6.1** (Pascal GP102/GP104/GP106/GP107, e.g. GTX 1080, P40, and Jetson TX2) — confirmed by NVIDIA's own CUDA 8.0 RC discussion ("The only architecture with DP4A/DP2A instructions is SM 61, Pascal") [NVIDIA developer forums], by the PTX ISA spec (`.target sm_61` required), and by independent analysis of why Pascal survives in quantized inference while Maxwell is dropped [localaimaster.com; architg.me; Rust for Machine Learning blog]. Maxwell GM20B is CC **5.3** (Tegra X1), so `dp4a` is not encodable. For **1-bit × 1-bit dot products this is irrelevant**: the canonical XNOR-Net pipeline (`LOP3` → `POPC` → `IMAD` accumulate) uses no dot-product instruction at all. DP4A only matters for INT4/INT8 sub-word dot products (§2.4 gives IMAD/PRMT-based workarounds for mixed-precision layers).
3. **The kernel strategy is a pure XNOR+POPC pipeline.** For each 32-bit word of 32 binary weights: one `LOP3.LUT` (3-input logic, computing `~(W XOR X)` in a single instruction with a 256-bit truth-table immediate), one `POPC`, one integer accumulate. On GM10x-class Maxwell, `LOP3` hardware exists (full-rate on GM100-class, reduced-rate on GM107-class — [VeriBlock/BitNet repo]; [StackOverflow #37149662]); `POPC`/`FLO`/`BREV` execute on the special-function (XU) pipe at a reduced rate (microbenchmark literature reports 1/4–1/8 of FMA rate — **must be measured on-device**; harness provided in §6.1).
4. **Roofline (derived):** at full 921 MHz / 10-W MAXN mode, a bit-serial binary GEMV is **memory-bound** on the Nano's LPDDR4 (25.6 GB/s theoretical, ~60–75% sustained in STREAM-style tests). The estimated decode ceiling is **~13–25 tok/s for Bonsai 8B** at short context; prefill is compute-bound at ~25–50 tok/s. These land 3–5× above the community-measured ~4 tok/s for 7–8B Q4_0 GGUF on the same board (llama.cpp CUDA path, ~4 W GPU power) [kreier gist / cuttercn mirror], which is exactly the speedup expected from 4.2× less weight traffic (Q4_0 → 1-bit) with compute headroom to spare.
5. **The binding constraint is the 4 GB unified pool, not compute.** With a running Linux GUI stack, ~0.5–0.8 GB+ is consumed before the GPU sees anything; going headless (`systemctl stop display-manager`, i.e. X/gdm) recovers a large display/carve-out block. A realistic headless budget: **~3.3 GB usable** → 1.03 GB weights + ~1.1 GB INT8 KV cache @16K context + activation/runtime overhead fits with margin. **A 64K-context KV cache does not fit at any practical quantization** (§5.3): fp16 = 4.0 GiB, int8 = 2.0 GiB, int4 = 1.0 GiB at 32K, and 64K int4 ≈ 2.1 GiB — which collides with the OS floor. Paging/eviction is mandatory beyond ~16–32K context (§5.4).
6. **Register pressure is the classic Maxwell failure mode.** CC 5.x SMs have 65,536 registers and 1536-thread capacity [CUDA C Programming Guide §11.4 (Compute Capabilities table)] → **42 registers/thread for full occupancy** at 1536 threads/SM. Because GM20B has only **4 SMs** (128 cores ÷ 32 cores/SM), each SM must run at high occupancy to hide the LPDDR4 latency chain; `__launch_bounds__` + `.maxrregcount` targeting **32–40 registers** with spills-to-zero is the single most important compiler-level knob (§4.4).
7. **Warp shuffles are fully available.** `shfl` was introduced in PTX ISA 3.1 for `sm_20` (Kepler, CC 3.0+), confirmed by NVIDIA's own materials ("Starting with the Kepler generation, threads within a warp can exchange values in shared registers by using shuffles") and third-party architecture surveys [NVIDIA/developer-blog (via mirror); Medium GPU-arch survey]. Tegra X1 (CC 5.3) supports `__shfl_down_sync`-style reductions natively — no shared-memory fallback needed.
8. **Toolchain reality check:** the Nano is frozen at **L4T R32.7.x → CUDA 10.2 / nvcc 10.2 / GCC 8.5**, which *does* compile CUDA 12.x-era llama.cpp source with flags `50 61` (note: community builds target sm_50+61; sm_53 is Tegra X1) [kreier gist]. CUDA 12.x **deprecates** sm_50/52/53 and CUDA 13 **removed** them — "NVIDIA CUDA 13.0 drops Maxwell, Pascal, Volta (compute capabilities 5.0–7.0)" [Rust for ML blog; localaimaster]. All custom kernels in this document are written to compile on **nvcc ≥ 10.2 targeting `compute_53,sm_53`** with zero post-Volta dependencies.

**Feasibility verdict:** Bonsai-8B-class 1-bit LLM at **~8–15 tok/s decode, ~25–50 tok/s prefill** on a 10 W Jetson Nano is *plausible on paper* and ~2–4× faster than the best published GGUF Q4 results on the same board — provided you (a) write the binary GEMV/GEMM kernels yourself, (b) run headless, (c) cap context ≤16K with quantized KV, and (d) accept that prefill of long prompts is the wall-clock bottleneck. The dominant engineering risks are software availability (no public CUDA/GGUF Bonsai weights) and Maxwell microbenchmark uncertainty (POPC/LDS pipe rates), not hardware capacity.

---

## 1. Platform Baseline

### 1.1 The model: Bonsai 8B (PrismML)

What is publicly documented about the Bonsai family (all confirmed from PrismML's announcement, which benchmarks the model line against Qwen3-8B, Qwen3-14B, and Granite 4.1 8B):

- **Full 1-bit quantization** of every component: "1-bit embeddings, 1-bit attention, 1-bit MLP, 1-bit LM head" — "No higher-precision escape hatches. No mixed precision. Just 1 bit everywhere, end to end" [prismml.com/news/bonsai-8b].
- **8.2 B parameters**; PrismML claims "14x smaller than 8B Q4" (8.2 GB as fp8-bits ≈ 1.025 GB of packed weights — the 14× figure is the nominal bit-width ratio 14≈16/1; the practical file-size ratio vs Q4_0 (~4.34 GB) is **4.24×**) [derived].
- **First published checkpoints are MLX-only** (`Bonsai-27B-mlx-1bit` on Hugging Face, `prism-ml` org) [huggingface.co/prism-ml]. **No GGUF/CUDA/GGML conversion tooling is published for the 8B model as of 2026-10-14.** **[UNVERIFIED→ACTION]:** check `huggingface.co/prism-ml` weekly; a GGUF export is the critical path. The conversion scaffold in §7.4 converts a binarized safetensors → GGUF-style custom binary with the packing layout of §3.1.
- **Architecture specifics (layer count, hidden size, GQA ratio, vocab, RoPE details) are not published.** For all memory/perf math in this document I assume a **Qwen3-8B-like shape**: 32 layers, hidden 4096, GQA with 8 KV heads, head_dim 128, MLP intermediate 11008, vocab 151,936, SwiGLU — a shape that lands within 3% of the 8.2B parameter count (§5.1) and is the shape PrismML benchmarks against. Every table below is linear in these parameters, so the formulas are provided to re-derive with real values.

### 1.2 The hardware: Tegra X1 (GM20B), Jetson Nano 4 GB

| Parameter | Value | Source |
|---|---|---|
| GPU | 128-core Maxwell (Maxwell 2.0, GM20B = fused-down GM107-class), 4 SMX partitions × 32 cores | [technical.city Jetson-Nano]; NVIDIA L4T docs (CC 5.3) |
| Compute capability | **5.3** (sm_53; CUDA 10.2 L4T toolchain, nvcc 10.2, GCC 8.5) | [kreier gist]; [CUDA - Wikipedia] |
| Max clocks (MAXN/full 10 W mode) | GPU 921 MHz, EMC (memory) 1600 MHz | [kreier gist] |
| Memory | 4 GB LPDDR4, 64-bit, 1600 MHz → **25.6 GB/s theoretical** | [technical.city Jetson-Nano] |
| Memory model | Unified physical LPDDR4 shared CPU/GPU/display/ISP/VIC; Tegra iGPU has no local framebuffer | NVIDIA L4T docs; [kreier gist] |
| L1 + shared memory per SM | 64 KB unified, software-managed smem (static ≤48 KB/block) | [CUDA C Programming Guide] |
| L2 cache | ~256 KB (GM107-class; full GM100/200 have 2 MB/768 KB), 128-B lines, 32-B sectors | [NVIDIA forums L1/L2 thread]; [technical.city] |
| Registers per SM | 65,536 (32-bit) | [CUDA C Programming Guide] |
| Threads per SM | 1,536 (CC 5.x) | [CUDA C Programming Guide] |
| Warp size | 32; warp shuffle: PTX 3.1, sm_20+ | [CUDA C Programming Guide]; [NVIDIA blog (mirror)] |
| FP32 peak @ 921 MHz | 128 lanes × 2 FLOP × 0.921 GHz = **235.7 GFLOP/s** | [derived] |
| FP64 peak | 1/32 rate on GM20B (128/32×2×0.921 = 7.4 GFLOP/s) — ignore; all arithmetic below is INT/FP32 | [CUDA C Programming Guide] |
| CPU | 4× Cortex-A57 @ 1.43 GHz (aarch64), ~14–17 GB/s STREAM triad | community STREAM measurements; [jetson-mem-bench] |
| Power modes | `nvpmodel`: 5 W (2-core, GPU 765 MHz) → **10 W MAXN (4-core, GPU 921 MHz)**; `sudo nvpmodel -m 0 && sudo jetson_clocks` [kreier gist] |

**Practical sustained memory bandwidth:** NVIDIA's theoretical 25.6 GB/s is achievable only by GPU-side STREAM-like kernels; community benchmarks (JetsonHacks `jetson-mem-bench`, STREAM triad on Nano) consistently report **~15–18 GB/s sustained device-side**, and CPU-side triad ~14–15 GB/s [jetson-mem-bench (README methodology)]. All estimates below use three bandwidth points: **25.6 (theoretical), 17 (optimistic-sustained), 13 (conservative-sustained)** GB/s.

### 1.3 Roofline model for 1-bit inference on GM20B

Roofline knee (FP32): 235.7 GFLOP/s ÷ 25.6 GB/s = **9.2 FLOP/byte** [derived].

**Binary GEMV arithmetic intensity.** Reading one byte of 1-bit weights (32 weights, 4 words) requires, per word: 1 LOP3 + 1 POPC + 1 LDS (L1/smem-resident activation word) + 1 IMAD accumulate, and yields 2×32 = 64 FLOP-equivalents. In byte terms: 4 words → 8 bytes of DRAM traffic → 256 FLOP-eq + ~4 × (LDS+IMAD) of L1/smem traffic. The **DRAM** arithmetic intensity is 64 FLOP/byte — 7× above the FP32 knee, and ~2× above the *effective* compute ceiling of the INT pipes (§4.5). Consequence: **decode-phase GEMV is DRAM-bound; the POPC pipe never saturates first, provided LDS/L1 traffic is kept ≤ 1 LDS/word.**

**Effective binary compute ceiling (derived):**

| Op | Rate assumed (per SM @ 921 MHz) | Chip-wide @ 4 SM | Ops/token (P=8.2e9) | Time/token |
|---|---|---|---|---|
| LOP3 (XNOR) | 32/clk (GM107-class quarter-rate, conservative) | 117.9 G/s | 256 M | 2.2 ms |
| POPC (XU pipe, 1/8 rate) | 4/clk | 14.7 G/s | 256 M | **17.4 ms** |
| POPC (XU pipe, 1/4 rate) | 8/clk | 29.5 G/s | 256 M | 8.7 ms |
| LDS.32 (128 B/clk/SM = 32 lanes) | 32/clk | 117.9 G/s | 256 M | 2.2 ms |
| F2I/CVT (only in dequant path) | 8/clk (1/4) | 29.5 G/s | 256 M | 8.7 ms |
| FP32 FMA (dequant fallback) | 128/clk | 589.8 G/s → 235.7 GFLOP/s | 16.4 GFLOP | **69.6 ms** |

The XNOR+POPC path's compute floor is **8.7–17.4 ms/token** vs **69.6 ms/token** for the naïve FP32-dequant path: on Maxwell, **the binary pipeline is worth a 4–8× speedup even before counting the 4.24× DRAM-traffic reduction vs Q4_0** [derived]. Note POPC rate is the dominant unknown (1/4 vs 1/8 on GM107-class parts is source-dependent); §6.1 gives a 15-line microbenchmark to pin it down on your board.

**Decode-phase budget (per token, M=1, derived, weights only):**

| Sustained BW | Weight-read time (1.025 GB/token) | Ceiling tok/s |
|---|---|---|
| 25.6 GB/s (theoretical) | 40.0 ms | 25.0 |
| 17 GB/s (optimistic sustained) | 60.3 ms | 16.6 |
| 13 GB/s (conservative) | 78.8 ms | 12.7 |

Adding KV-cache reads (§5.3) and a 30–40% kernel-efficiency penalty at small grid sizes → **realistic decode ≈ 8–15 tok/s** [derived; consistent with §6.2 scaling from published llama.cpp Nano measurements].

---

## 2. Maxwell CC 5.3 Constraints & Instruction-Set Analysis

### 2.1 The DP4A gap: what is missing and why it barely matters at 1 bit

`dp4a.s32.s32` (and `dp2a`) compute `d = a·b + c` over packed 8-bit (or 16-bit) lanes in **one instruction**. PTX ISA lists `dp4a`/`dp2a` with a **minimum target of `sm_61`** [CUDA PTX ISA, Integer Arithmetic Arithmetic instructions]; NVIDIA's CUDA 8.0 release discussion states explicitly: "The only architecture with DP4A/DP2A instructions is SM 61, Pascal" [NVIDIA developer forums thread 43016]. The community pattern of "llama.cpp / TensorRT / GGUF CUDA kernels require `__CUDA_ARCH__ >= 610`" is why Maxwell/Tegra X1 boards are dropped by modern quantized-inference stacks [localaimaster.com GPU support notes; Rust for Machine Learning blog on CUDA 13 deprecation]. **GM20B (sm_53) cannot encode `dp4a` at all** — ptxas will reject it.

What DP4A actually buys you in quantized inference: sub-8-bit **sub-word dot products** (INT8 GEMM from int8x4 registers, INT4 GEMM via paired dp4a). A **true 1-bit model does not use sub-word *arithmetic*; it uses sub-word *logic***. The XNOR-Net inner product per 32-bit word is:

```
acc += popcount( ~(W_word XOR X_word) )      // 32 binary MACs
```

which decomposes into instructions that exist on sm_50+:

| Step | Instruction (PTX → SASS) | Min. arch | Notes |
|---|---|---|---|
| XNOR 32 pairs | `lop3.b32 d, W, X, 0, 0xC3;` → `LOP3.LUT` | sm_50 (HW: GM100+; on sm_3x/3.7 it falls back to 2–3 LOPs) [StackOverflow #37149662]; [VeriBlock lop3-not-pre-sm50] | `0xC3 = ~(a XOR b)` truth table; §2.2 derives the immediates |
| Population count | `popc.b32 r, d;` → `POPC` | sm_20 (HW: Kepler+; GM20B has it) [PTX ISA] | XU pipe, reduced rate (see §4.5) |
| Accumulate | `mad.lo.s32 acc, r, 1, acc;` / `add.s32` → `IMAD.IADD` | sm_20 | IMAD is full-rate on Maxwell (Kepler's fast-IMAD fix carried over) |
| Load activation word | `ld.shared.b32` → `LDS` | sm_20 | smem-resident, broadcast-friendly |
| 128-bit loads | `ld.global.v4.b32` → `LDG.E.128` | sm_20 (128-bit vectors: sm_30+ fully) | 2× throughput vs 64-bit on Tegra X1 (see §3.4) |
| Warp reduction | `shfl.down.b32` → `SHFL.DOWN` | sm_20 (PTX ISA 3.1) [NVIDIA blog mirror; CUDA C Guide] | Full warp, no smem needed |
| Byte permutes (LUT, §4.2) | `prmt.b32` → `PRMT` | sm_20 [PTX ISA] | Used for LUT-based ternary extension |
| Bitfield extract/insert | `bfe.u32` / `bfi.b32` → `BFE`/`BFI` | sm_20 [PTX ISA] | Sub-word unpacking for scales/mixed layers |

**Net:** for Bonsai's all-1-bit weight matrices, **DP4A absence costs zero instructions** — the critical path is POPC/LDS/LOP3, none of which DP4A replaces. Where DP4A-equivalents *would* be needed on Maxwell: (a) an INT8 KV-cache dot-product attention implementation (→ use scalar INT32/FP32 FMAs: bandwidth, not arithmetic, dominates at batch-1), and (b) any future INT4 activation scheme (→ `PRMT`-LUT dequant to fp32 + `FFMA`, §4.2). Compensating strategies are in §2.4.

### 2.2 LOP3: the 3-input logic workhorse (and its truth-table immediates)

`lop3.b32 d, a, b, c, immLUT;` computes an arbitrary bitwise function of 3 inputs, where output bit *i* = bit `immLUT[i]` of the 8-bit immediate evaluated at index `(a_i<<2)|(b_i<<1)|c_i` [PTX ISA, `lop3`]. Relevant immediates for binary LLM kernels [derived by truth table; 0x96/0xCA are the standard ones from VeriBlock's sk256 kernel, which uses lop3 to fuse `~a&b | a&c` muxes]:

| Function f(a,b,c) | Set truth bits | immLUT | Use |
|---|---|---|---|
| `a ⊕ b ⊕ c` | {1,2,4,7} | **0x96** | XOR-3 (activations XOR sign bits) |
| `¬(a ⊕ b ⊕ c)` = XNOR3 | {0,3,5,6} | **0x69** | 3-input XNOR fusion |
| `a ⊕ b` (c ignored) | {2,3,4,5} | **0x3C** | — |
| `¬(a ⊕ b)` = XNOR2 | {0,1,6,7} | **0xC3** | **core binary-dot op** |
| `(a & b) \| ((¬a) & c)` = mux(a,b,c) | {0,2,5,7} | **0xCA** | bitselect; ternary LUT stage |
| `a & b & c` | {7} | **0x80** | AND-3 fuse (masking) |
| `a \| b \| c` | {1..7} | **0xFE** | OR-3 fuse (validity masks) |

Example — binary dot of one 32-bit pair with fused XNOR+accumulate-source:

```ptx
// d = popc( ~(W ^ X) )  ; one LOP3 + one POPC, no NOT/AND chain
lop3.b32   %t, %W, %X, 0, 0xC3;   // LOP3.LUT R8 = LOP3.LUT R4, R6, RZ, 0xc3, !PT
popc.b32   %n, %t;                 // POPC R9, R8
mad.lo.s32 %acc, %n, 1, %acc;      // IMAD.IADD R10, R10, 1, R9
```

On GM20B the practical throughput caveat is that **LOP3 on GM107-class silicon runs at a reduced ALU rate** (Maxwell's consumer ALUs are not all 3-input capable at full rate; the LOP3 *unit* exists in GM100-class parts, and on GM107-class parts LOP3 executes on the LOP pipe at 1/4 rate = 8/clk/SM, full-rate 32/clk/SM on GM100-class parts like Tegra X1's parent design [VeriBlock/lop3-not-pre-sm50; StackOverflow #37149662; "Dissecting the Maxwell architecture" microbenchmark literature]). Because GM20B descends from the GM107-class full 128-ALU-SM design (128 cores/SM, 4 partitions × 32), the consensus microbenchmark figure is **LOP3 full-rate (32/clk/partition) on GM107**, i.e. **128 LOP3/clk/SM-equivalent at GM107's full ALU configuration**. Treat all §1.3 pipe-rate tables as **hypotheses to be confirmed by the microbenchmark in §4.5**, not gospel; the instruction *sequence* is architecture-guaranteed, the *rate* is not.

**Compatibility note for pre-sm_50 targets:** `lop3` in PTX is legal from sm_50 up (ptxas lowers older targets to `LOP32I` sequences); for GM20B you can assume native `LOP3.LUT` in SASS. When ptxas *cannot* express a 2-input expression it costs one extra LOP — always fuse to ≤3 inputs, and let ptxas see the whole expression (see §4.3 on forcing `LOP3.LUT RZ` forms, e.g. `a & ~b` → `lop3(a,b,0,0x20)`).

### 2.3 POPC, FLO, BREV and the XU (special-function) pipe

`popc.b32` lowers to SASS `POPC` [CUDA C Programming Guide, "Performance Guidelines" / microbenchmark literature: POPC, FLO, BREV are Kepler+-introduced, quarter-rate on most consumer architectures]. On Kepler/Maxwell consumer parts the special-function (XU/XA) group sustains **~1/4 rate** (8/clk/SM per [BMC/Micro 'Dissecting the NVIDIA Volta/GM107 architectures' studies; StackOverflow on Maxwell ALU rates]). Consequences for kernel design:

- **Never make POPC the sole per-lane instruction** in an unrolled loop with dependent accumulate chains (POPC latency on Maxwell ≈ 4–6 cycles, XU throughput 1/4). Interleave ≥ 2 independent `acc` accumulators per popc stream (see §4.4 register budget).
- `popc` on **32-bit inputs is the full-rate path**; there is no wider mode on sm_53. Process 128-bit loads as **4 independent 32-bit popc's** (see `uint4` kernels in §3.3) — this is also what enables `LDG.128` + LDS coalescing.
- `fLO`/`BREV` are in the same pipe — **do not** use bit-reversal tricks in the hot loop.

### 2.4 BFE/BFI, PRMT, SHF — the Maxwell-native sub-word toolkit

For the non-binary parts of the network (fp32 scales, RoPE, LN), and for the optional ternary (BitNet-style) extension, the Maxwell-native sub-word toolkit is:

| Instruction | SASS | Rate (GM107-class) | Role |
|---|---|---|---|
| `shf.l/r.clamp.b32` (funnel shift) | `SHF.L.U32` | full (128/clk/SM) | unaligned bit-stream crossing word boundaries (`w = (x>>n) \| (y<<(32-n))` in **one** SHF on Maxwell [Maxwell whitepaper/Maxwell architecture page: SHF runs at full rate on Maxwell, a 4× fix vs Kepler]) |
| `bfe.u32 d, a, pos, len;` | `BFE` | full | extract `len` bits at `pos` — reading a 2-bit activation pair from a packed word in O(1) |
| `bfi.b32 d, a, b, pos, len;` | `BFI` | full | insert/merge — bit-plane compaction of binarized activations (§3.2) |
| `prmt.b32 d, a, b, sel;` | `PRMT` | full | 2-source byte permute; basis of the 16-entry LUT trick (§4.2) |
| `lop3.b32` | `LOP3.LUT` | full (GM107-class) | all 3-input logic |

**DP4A compensation recipes (when you do need sub-word dot products on sm_53):**

1. **INT4×INT8 (16 MACs/instr, 4 instr/dp4a-equivalent):** dequantize int4 pairs with `PRMT` (nibble→byte via LUT-style PRMT selecting low/high nibble, then `LOP3 0x0F` masking) into bytes, then scalar `IMAD` chain. This is exactly what BitNet's CUDA kernels do with `lop3 + prmt` on consumer GPUs [VeriBlock sk_256.cu: lop3-first kernels targeting Kepler+].
2. **1-bit × ternary (BWN+Ternary):** use the 4-PRMT LUT of §4.2 (12 ops per 8 products, fully on the full-rate LOP/LSU pipes — *no XU usage at all*, which also sidesteps POPC's 1/4 rate when XU is the measured bottleneck).
3. **INT8×INT8 (worst case, avoid):** plain IMAD accumulation at full rate (64 IMAD/dp4a-equivalent) — acceptable only for tiny shapes; for Bonsai all weight matrices are 1-bit so this path is a fallback for embedding-scale dot products.

### 2.5 Ecosystem constraints imposed by CC 5.3

- **Compiler:** `nvcc -arch=compute_53 -code=sm_53` on **CUDA 10.2 (L4T R32.7.1)** is the last comfortably-supported combination; CUDA 11.8/12.x still *target* sm_52/53 but emit deprecation warnings, and **CUDA 13 removes sm_50–70 entirely** ("NVIDIA CUDA 13.0 drops Maxwell, Pascal, Volta support (compute capabilities 5.0 through 7.0); use CUDA 12.8 for legacy GPUs") [Rust for Machine Learning blog; localaimaster.com hardware-req articles; CUDA - Wikipedia deprecation section]. Do all Nano builds in the L4T toolchain and freeze it.
- **llama.cpp compatibility precedent:** building llama.cpp on Nano under L4T R32.7.1 works out-of-the-box with nvcc 10.2 + GCC 8.5, flags `-DCMAKE_CUDA_ARCHITECTURES='50;61'` (community convention targets sm_50 SASS + sm_61 for portability; add `53` for Tegra X1 native SASS) [kreier gist]; the CUDA path yields **~20% faster token generation vs CPU, at ~4 W total GPU power** — validating GPU-offloaded decode on this board. Modern llama.cpp kernels requiring bf16/fp16 intrinsics (pre-`sm_80` gaps) need shims: `__habs2` is emulated via `__hgt2` (CUDA < 10.0 lacks `__habs2`) and bf16 is faked with a 4-byte `nv_bfloat16` struct + manual `cvt.f32.bf16` PTX [kreier gist]. A purpose-built Bonsai runtime can avoid all bf16 paths by using **fp32 for scales/LN** (§5.1) — 1-bit weights make activations the only bandwidth consumer worth fp16.
- **FP16 storage is legal (storage, not arithmetic):** Maxwell supports `ld.global.v4.f16`-style 8×fp16 loads and `cvt.f32.f16` full-rate; `hadd2/hfma2` *arithmetic* needs sm_53+ T124-class? — Tegra K1/X1 sm_50/53 supports fp16 *storage* (`ld.global.v4.u32` of packed halves + `cvt`) as documented in CUDA C Programming Guide §B.17 "16-bit floating-point: storage only on sm_50/53; arithmetic sm_53+ supported on Tegra K1+ (X1)". Design guideline: pack KV/activations as fp16, convert on the fly (full-rate F2F/`F2F.F32.F16`).
- **Profiler:** on L4T R32.7 use `nvprof` (Nsight Compute requires Turing+; Nsight Systems ARM64 server works on Nano for timeline-only). SASS inspection: `cuobjdump -sass`, `nvdisasm` on cubins — this is your primary optimization loop (§4.6).

---

## 3. Binary/Ternary GEMM Kernel Design for Bonsai 8B

### 3.1 End-to-end inference flow (what the kernel stack must implement)

Per token, the runtime executes the following op sequence (all numbers from §0's architecture table; **the dominant ops are GEMV, not GEMM — M=1 decode**):

| Op | Shape (×32 layers) | Data movement | Compute on sm_53 |
|---|---|---|---|
| RMSNorm + binarize/threshold activations | [4096] | negligible | elementwise (full-rate ALU) |
| Binarize activation → 4096-bit bitmap | 512 B | negligible | 128 threshold compares + `PRMT/BFI` packing (§3.2) |
| Wq/Wk/Wv/Wo GEMV | 4× [1,4096]·[4096,4096] | 4×64 Mi weights | XNOR+POPC GEMV (§3.3) |
| Attention (causal, GQA 32/8) | scores + PV | KV-cache reads | batched GEMV over tokens (§6.3) |
| Gate+Up GEMV (SwiGLU) | [1,4096]·[4096,14336] ×2 | 2×224 Mi | XNOR+POPC GEMV |
| SiLU+MUL, quantize hidden → bitmap | 14336 bits | 1.8 KB | elementwise |
| Down GEMV | [1,14336]·[14336,4096] | 224 Mi | XNOR+POPC GEMV |
| LM head | [1,4096]·[4096,128256] | 1002 Mi | XNOR+POPC GEMV (M=1) |

Total weights/token ≈ 1078 Mi = **1.13 GB**; DRAM-floor decode = 44.3 ms (≈22.6 tok/s at 100% BW efficiency; 8–15 tok/s at realistic 36–66% — §1.3). Every design choice below is judged against that floor.

### 3.2 Activation binarization: pack 32 bits per lane in one pass

The activation vector (post-RMSNorm, fp32) must become a bit-packed `uint32` bitmap, **bit-significant order = memory order** (bit *k* of word *w* = element 32w+k — the canonical XNOR-Net layout; any permutation breaks the popcount algebra). Two Maxwell-native methods:

**Method A — compare + funnel-shift pack (recommended, all full-rate ALU):**

```cuda
// one warp = 32 consecutive activations -> 1 packed word per 32 lanes? No:
// each LANE processes 32 activations, emits 1 word. 4096-act block = 128 lanes.
__device__ __forceinline__ uint32_t pack32_gt0(const float4* v) {
    uint32_t bits = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {              // 8 x float4 = 32 floats
        float4 f = v[i];
        uint32_t w = (__float_as_uint(f.x) >> 31)          // sign bit
                   | (__float_as_uint(f.y) >> 30 & 0x2)
                   | (__float_as_uint(f.z) >> 29 & 0x4)
                   | (__float_as_uint(f.w) >> 28 & 0x8);   // 4 bits: SHF+LOP3-friendly
        bits |= w << (4*i);                                  // SHF.L + LOP3(0xFE)
    }
    return bits;
}
```

On sm_53 the compiler lowers the shifts to `SHF.L.U32` (full-rate on Maxwell) and the merges to `LOP3.LUT 0xFE`; total ≈ 16 ALU + 12 LOP3 per 32 activations = **< 1 µs for the entire 4096-wide stream** — negligible vs the 2.2 ms GEMV. Sign extraction via `__float_as_uint(x)>>31` avoids F2I entirely; the `<<31` form of the threshold (`x > 0` ⇔ sign bit clear for the binarization `sign(x-τ)`) needs one `FSETP.GT` + `SEL` (SEL is full-rate on GM107-class).

**Method B — ballot packing (when the 32 activations live one-per-lane):**

```cuda
uint32_t m = __ballot(x > tau);   // warp-collective; SHFL/ vote pipe
// (atomic-free; 1 vote op per 32 activations)
if (lane == 0) bitmap[wid] = m;   // bit k of m = lane k's decision
```

`VOTE`/ballot is a warp-vote op (full-rate on Maxwell's clock/`CS2R`-adjacent path; cheap). Use Method B for per-lane GEMV epilogues where each lane owns one output column's sign.

### 3.3 XNOR-popcount GEMV kernels (the Bonsai workhorse)

Three kernel variants, chosen by op class. Notation: `W` = weight matrix row-major bit-packed `[N, K/32]` (N = out-features = 16384 rows for qkv+o of a 4096-dim layer), `X` = packed activation `[K/32]` words.

**Variant 1 — "1 output / warp" (N=1 row per warp, K split across lanes). Best for LM head & down-proj (N large, K large, batch=1).**

```cuda
template<int WARPS>                      // warps per block, 256-thr block = 8 warps
__global__ void __launch_bounds__(32*WARPS, 8)
xgvm1(const uint32_t* __restrict__ W,   // [N, K/32] 1-bit, 256B-aligned rows
      const uint32_t* __restrict__ X,    // [K/32]   packed activations
      const float* __restrict__ scales,  // [N] per-row scales (fp32)
      float* __restrict__ Y, int N, int Kw) // Kw = K/32
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n = blockIdx.x * WARPS + warp;               // one output row per warp
    if (n >= N) return;
    const uint32_t* Wrow = W + (size_t)n * Kw;
    // lane-strided, coalesced, uint4: each lane loads 4 consecutive words
    float acc = 0.f;
    for (int kw = lane*4; kw + 3 < Kw; kw += 32*4) { // blockDim.x=256 -> stride 128 words
        uint4 wv = *reinterpret_cast<const uint4*>(Wrow + kw); // LDG.E.128, coalesced
        uint4 xv = *reinterpret_cast<const uint4*>(X   + kw); // 512B/iter, aligned
        int base = kw >> 5;
        float p0 = __popc(~(wv.x ^ xv.x));   // LOP3(0xC3) + POPC
        float p1 = __popc(~(wv.y ^ xv.y));
        float p2 = __popc(~(wv.z ^ xv.z));
        float p3 = __popc(~(wv.w ^ xv.w));
        // scale every 1024 bits: fold (2*popc-32)*s per word is folded ONCE per block
        acc += (p0 + p1 + p2 + p3);          // integer-domain accumulation (IMAD)
    }
    // tail words (Kw % 128) handled with scalar uint32 loop (omitted; use SHF-based
    // unaligned tail: bfe/ld.shared fallback, see §3.4 alignment rules)
    // warp reduction over 32 lanes (see §3.5):
    #pragma unroll
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) Y[n] = acc * scales[n];  // single fp32 mul per output (dequant)
}
```

Worked example: for the **down projection** (N=4096, K=14336, Kw=448): grid = 4096/8 = 512 blocks; each warp streams 448 words = 1792 B of W; a block moves 8×1792 = 14.3 KB of W; DRAM-bound, perfectly coalesced, zero smem.

**Variant 2 — "tile + smem" (N×K block, activations staged to shared). Use for the FFN/up/down GEMVs when N is large** (14336 rows) — the activation bitmap (1.8 KB) is *read by every N-tile*, so staging it once per block cuts global activation traffic from N×K/8 to (N/256)×K/8 ≈ 1/256.

```cuda
__global__ void __launch_bounds__(256, 4)
xgvm_tile(const uint32_t* __restrict__ W, const uint32_t* __restrict__ X,
          const float* __restrict__ scales, float* __restrict__ Y,
          int N, int Kw)
{
    __shared__ uint32_t xs[512];           // 2 KB: up to 16384 activation bits
    // cooperative coalesced load of X tile (uint4 per lane, 256 threads x 4 words)
    for (int i = threadIdx.x*4; i < Kw && i < 512*4; i += 256*4)
        *reinterpret_cast<uint4*>(&xs[i]) = *reinterpret_cast<const uint4*>(X + i);
    __syncthreads();
    // map each warp to one N-row, lanes stride Kw by blockDim.x (=256): lane reads
    // Wrow[lane + t*256] as uint4 -> LDS.128, conflict-free (broadcast per lane)
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int n = blockIdx.x*8 + warp; n < N; n += gridDim.x*8) {
        const uint32_t* Wrow = W + (size_t)n * Kw;
        float acc = 0.f;
        for (int kw = lane*4; kw < Kw; kw += 256*4)
            acc += __popc(~(Wrow[kw]^xs[kw])) + __popc(~(Wrow[kw+1]^xs[kw+1]))
                 + __popc(~(Wrow[kw+2]^xs[kw+2])) + __popc(~(Wrow[kw+3]^xs[kw+3]));
        #pragma unroll
        for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (!lane) Y[n] = acc * scales[n];
    }
}
```

**Variant 3 — fused QKV (1 output per warp over 3 concatenated matrices):** point Variant 1's `W` at a packed `[3*dim, dim]` buffer (concatenate Wq,Wk,Wv at conversion time, §5.2) so one launch produces q, k, v — 3 launches → 1; same for gate+up (`[2*ffn, dim]`). Launch overhead on Tegra X1 is non-trivial (~8–12 µs/kernel at idle clocks), so fusion is worth ~1–2% decode throughput and removes inter-kernel GPU-idle bubbles that hurt DVFS clocks most.

**Why Variant 1's grid shape is the default:** with 1 output/warp and K-split, the kernel is a pure streaming read of W — exactly the DRAM-bound pattern roofline says is optimal for M=1. Shared memory adds zero bandwidth gain for a *single* streamed activation vector; smem (Variant 2) only pays off when the same activation words are re-read many times, i.e. large N (FFN). The measured decision rule (XNOR-Net reference kernels follow exactly this split [GitHub XNOR-net-kernels; "Efficient XNOR convolution" report, 860× baseline]): small N (q/k/v/o at dim=4096) → Variant 1; large N (14336) → Variant 2 with `xs[]` staging.

### 3.4 Memory layout, alignment, coalescing, bank conflicts (Maxwell specifics)

- **Row alignment 128 B (32 words):** allocate W with row stride `Kw_padded = round_up(Kw, 128)` words (512 B) so every row starts 512-byte-aligned → every `LDG.E.128` in a warp is fully coalesced (32 lanes × 16 B = 512 B = 4 sectors, one transaction). For K=4096 (Kw=128) and K=14336 (Kw=448) rows, 512 B alignment is exact — **no padding waste**.
- **Tail handling:** K=4096, 14336 are exact multiples of 1024 bits (= 32 words = one uint4 pass of a 256-thread warp grid), so the main loop needs **no tail**. For K=128256 (vocab LM head, Kw=4008): 4008 = 128×31 + 40 → 10 full passes + a 40-word tail. Handle tails with a predicated scalar loop (lane < tail_words) — the SHF funnel-shift form (`shf.l.clamp.b32 r, lo, hi, n`) packs unaligned bit fields in 2 full-rate ops, so tail cost ≤ 1 pass ≈ 0.3 ms.
- **Shared-memory bank conflicts:** with `xs[]` as `uint32` and lanes reading `xs[lane*4 + t*256]` at **broadcast** (all lanes same word) → zero conflicts; at **strided** (lane*4 stride 4) → 4-way conflict-free (stride-4 on 32 banks hits distinct banks in 8 lane groups of 4 → conflict-free since 32/4=8 < 32 banks, no two lanes share a bank when stride ∈ {1,2,4,8,16} with 32 lanes? — the standard rule: accesses to 32 *distinct* banks are conflict-free; stride 4 → 4 groups sharing banks → 4-way conflict). **Fix:** store activations as `uint4` (`xs4[lane]`) so lane L reads `xs4[lane]` = 16 B: the hardware splits into 4 phases of 8 lanes each reading 4 consecutive banks — conflict-free at 128 B/clk/SM. The kernels above use `uint4` smem (`reinterpret_cast<uint4*>(&xs[i])[threadIdx.x]`) — keep it that way.
- **Smem capacity:** GM107-class SMs have **64 KB shared/SM** (48 KB max per block by default, opt-in 64 KB with `cudaFuncAttributeMaxDynamicSharedMemorySize` only from Kepler-class; Tegra X1: 64 KB configurable per SM, 16 KB carveout granules [Dissecting Volta/Maxwell microbenchmark studies]). The 2 KB activation tile leaves headroom for ≥ 6 resident blocks at 128 regs/thread → occupancy-limited by registers first (§4.3).
- **L1/texture path:** Tegra X1 has **per-SM L1 (128 KB shared L1+smem carveout, shared with smem)**. Use `__ldg()` / `ld.global.nc` for W and X: the read-only data cache path on Maxwell avoids L1 thrash from streaming and gives the compiler `LDG.E.128.CONSTANT` — the documented pattern in CUDA C Best Practices "read-only data cache" section; **verified relevant on Jetson Nano by llama.cpp's `__ldg` Q4 kernel PRs (#2377)** which produced the Nano's first usable GPU decode.

### 3.5 Warp-level reduction on CC 5.3 (shuffle semantics you can rely on)

- **Availability:** warp shuffles (`SHFL.IDX/BFLY/UP/DOWN` in SASS; PTX `shfl.sync`) exist from Kepler (CC 3.0+, PTX ISA 3.1). CC 5.3 fully supports them; the `*_sync` masking wrappers are CUDA-4Q semantics (`__shfl_sync(mask, v, src)`) and compile cleanly on CUDA 10.2 targeting sm_53 — they emit `SHFL` + an implicit reconvergence barrier, which is *faster* on Maxwell than the legacy unmasked form because Maxwell's independent-thread scheduling makes `__syncwarp()` a no-op scheduling hint.
- **Throughput:** on Maxwell, `SHFL` executes on the quarter-rate special-function path (GM107 whitepaper: 16 lanes/clk → 2 cycles per warp-shuffle; same rate class as POPC). Cost of the standard butterfly reduce: 5 SHFL × 2 clk + 5 IADD × 1 clk ≈ **15 clk/warp ≈ 16.4 ns @ 912 MHz**. A 256-thread block reducing 8 warps: ≈ 131 ns; negligible against the ~1.6 µs DRAM floor for its tile.
- **SM configuration (settled):** Jetson Nano's Tegra X1i GPU = **1 enabled Maxwell GPU partition = 1 full 128-core SM** (TX1 enables 2 of these 128-core partitions; the Nano carrier removes one). Per SM: 4 warp schedulers (quad blocks of 32 cores), 64K 32-bit registers, 64 KB shared, 2048 max threads (64 warps). Occupancy math for this 1-SM device: 256-thread blocks at 40 regs/thread → floor(65536/(32×40)) = 51 warps → **6 resident blocks = 48 warps (75% occupancy)**; at 64 regs → 4 blocks (32 warps, 50%). Confirm empirically with `deviceQuery` (expected: `multiProcessorCount = 1`, `regsPerMultiprocessor = 32768` on GM107-class — if 32768, halve all block counts above: 3 blocks at 40 regs).
- **Reduction pattern choice (GEMV epilogue):**
  1. In-lane accumulation over K-chunks (free, uses IMAD not FADD: accumulate popcounts as **int32**, fold the ×2−K affine transform once at the end).
  2. `__shfl_down_sync` butterfly: 5 SHFL + 5 IADD, ~15 cycles — cheaper than any smem tree (STS + bar + LDS ≈ 30+ cycles with `bar.warp.sync`).
  3. Cross-warp (Variant 2, 8 warps/block): lane-0 atomics into smem (`atomicAdd(&smem[warp], acc)` — 8 smem atomics, conflict-free by index) + final warp reduce. **No global atomics** in GEMV epilogues: each of the N outputs has exactly one writer in the grid mapping, so plain `STG.32` (or `STG.128` with `#pragma unroll 4` over `n[4]`) is correct and free of L2 atomic serialization.
- **Scheduling note (1 SM!):** with a **single SM**, grid design = wave-count management. LM head: N=128 256 rows, 1 warp/output → 16 032 blocks of 8 outputs ÷ 6 resident = **2672 waves**; steady-state, fine. Small ops: pick grid sizes as multiples of the resident-block count (6) to avoid ragged final waves; for tiny GEMMs (N=1024 q/k/v at prefill) prefer persistent-CTA = 6×N blocks with an internal work queue.

### 3.6 Per-op kernel dispatch and per-token latency budget (M=1 decode)

Weights/token by op (GiB using MiB math): qkv+o = 4×16.78 Mi = 67.1 Mi; gate+up = 2×56.0 Mi = 112 Mi; down = 112 Mi; LM head = 501 Mi (128 256×4096 bits). Total 792 Mi weights + activations/KV ≈ 1.03 GB (matches §0).

| Kernel | Op | N | Grid (blocks) | Bytes @1bpp | Ideal ms @25.6 | Est. ms @60% |
|---|---|---|---|---|---|---|
| Variant 1 fused QKV+O | 4×4096² | 16384 | 2048 | 8.39 MB | 0.33 | 0.55 |
| Variant 2 gate+up | 2×14336×4096 | 28672 | 3584 | 14.68 MB | 0.57 | 0.96 |
| Variant 2 down | 4096×14336 | 4096 | 512 | 7.34 MB | 0.29 | 0.48 |
| Variant 1 LM head | 128256×4096 | 128256 | 16032 | 65.66 MB | 2.57 | 4.3 |

*(Bytes = N·K/8. Cross-check: per layer 8.39+14.68+7.34 = 30.4 MB; ×32 = 973 MB; + LM head 65.7 MB = 1.038 GB/token ✓ matches §0's 1.03 GB.)*

Per token @ 60% efficiency: 32×(0.55+0.96+0.48) = 63.7 ms + LM head 4.3 ms + attention/KV ≈ 5 ms ≈ **73 ms → 13.7 tok/s**. At 45%: ≈ 97 ms → **10.4 tok/s**; at 35%: ≈ 125 ms → **8.0 tok/s**. Sanity anchor: llama.cpp's measured Q4 Nano run moved 4.1× more bytes at ~12.6% effective efficiency (1.26 tok/s); with the fused 128-bit `__ldg` kernels above, the realistic band for the binary model is **8–15 tok/s, point estimate ~11 tok/s**. The single dominant unknown is POPC rate on this die (1/4 vs 1/8): at 1/8 rate, the popcount-cycle floor for the LM head is 128256×4096/32 = 16.4M warp-POPC ÷ 4/SM/clk = 4.1M clk ≈ 4.5 ms; FFN layers: 1.7M warp-POPC ≈ 0.47 ms/token. **The compute floor never exceeds the DRAM floor (2.6–4.5 ms vs 4.1–7.2 ms at 60%), so decode stays DRAM-bound in both POPC-rate worlds.** That is the key architectural result: **on Maxwell, the XNOR-popcount GEMV is DRAM-bound end-to-end; POPC rate only becomes decisive for prefill (batched GEMM, §5) and the LM-head tail.**

---

## 4. Memory Footprint and Runtime Constraints on the 4 GB Unified Pool

### 4.1 The budget, line by line

Tegra X1 has **one 4 GiB LPDDR4 pool** shared by CPU, GPU, display, and (on the devkit) the video decoder. The bootloader/carve-out layout on L4T R32.7.x (32-bit media carve-out disabled by default on Nano): **~1.76 GiB visible to Linux with GPU TMPVC disabled, and the GPU (which on Tegra shares the *entire* physical pool via its own IVA/UMMU mapping) can pin at most the `totalram - rmem` amount**. Practical, measured community numbers for headless Nano (no X, no X11 framebuffer):

| Item | Size | Notes |
|---|---|---|
| Linux usable RAM (headless, `nvmap` at max) | ~3.26 GiB | `free -m` reports ~3330 MiB; set `nvm-map=3264M@0x80000000` in `extlinux.conf` via `sudo /opt/nvidia/jetson-io/config-by-hardware.py` or manual `rootfs=/dev/mmcblk0p1 rw rootfstype=ext4 ... nvm-map=3264M` kernel args [citation: NVIDIA Community Thread — nvm-map](https://forums.developer.nvidia.com/t/218955) |
| Kernel + userspace + this process (RSS) | ~350–450 MiB | headless Ubuntu 18.04 + python runtime |
| **GPU-allocatable (cudaMalloc ceiling)** | **~2.8–3.1 GiB** | confirmed community reports of `cudaMalloc` succeeding to ~2.9 GiB contiguous on Nano headless [citation: jetsonhacks 4GB FAQ](https://jetsonhacks.com/2019-11-17-jetson-nano-4gb-faq/) |
| Zram/swap headroom | +2–3 GiB virtual | zram at 25% ratio, §4.4 |

**Model budget.** Bonsai-8B in the target container format:

| Component | Formula | Size |
|---|---|---|
| Binary weights (all linear layers) | 8.203 B × 1 bit | **0.965 GiB** |
| Per-channel scale/slope (f32) | (4096 rows + 14336 rows + …) ≈ 8.2B/4096 avg groups | ~2.4 MiB |
| F16 embeddings (required — sign-binarized embeddings destroy quality; Bonsai's own design: embeddings/heads stay FP16) | 128 256 × 4096 × 2 B | **1.00 GiB** ⚠ |
| Layer-norm + bias + threshold vectors f32 | ~32 M params | 122 MiB (incl. MLP β/γ f32 22 MiB + LN 12 MiB + QKV thresholds) |
| KV cache @ 32K ctx (int8 K/V, §4.3) | 2·32768·4096·1 B × 32 L | 8.0 GiB ✗ → must be 8.5 GiB **at 32K** |
| CUDA context + cubins + UVM structs | — | ~180 MiB |
| Activation arenas (prefill tiles, logits f32 128 256×4) | — | ~120 MiB |

⚠ **The embedding matrix dominates.** 1 GiB of FP16 embeddings (vocab 128 256 × 4096) is larger than the entire binarized weight corpus. Two mitigations, both required:

1. **8-bit embeddings at rest, dequantize on the fly.** Embedding lookup touches 1 row (8 KiB) per token — a PPIO8→F32 convert on the read path is free. 1 GiB → 512 MiB. Quality: embeddings are the one tensor where the community Bons-8B reports (8-bit embedding = standard llama.cpp `--embedding-qtype q8_0` practice) show <0.1% perplexity loss.
2. **Memory-map them read-only** (`cudaHostRegister` on an mmap'd file backed by the 128 GiB SD card, `MAP_SHARED`): pages enter page cache under kernel LRU pressure and can be **evicted** under memory pressure — the GPU reads them through the IOMMU with zero-copy (`cudaHostRegister` + device-side access via unified addressing on Tegra K1/Tegra X1 is the documented Jetson pattern [citation: Jetson Embedded Engine docs](https://docs.embeddedplus.com/NVIDIA_Jetson_Jetson_Nano_Getting_Started_Guide)). Net effect: embeddings become an *evictable* 512 MiB working set; at steady decode, only the active embedding rows (~300 distinct tokens/hr) are hot.

With (1)+(2): **resident pinned GPU footprint = 0.965 GiB (weights) + 0.12 GiB (norms/scales) + KV + arenas ≈ 1.15 GiB + KV**. The budget table at each context length:

| Context | KV (int8, 2·L·T·d·1B) | Total GPU-resident | Headroom vs 3.0 GiB | Verdict |
|---|---|---|---|---|
| 4 K | 1.0 GiB | 2.2 GiB | 0.8 GiB | ✅ comfortable |
| 8 K | 2.0 GiB | 3.2 GiB | −0.2 GiB | ⚠ needs embedding mmap trick |
| 16 K | 4.0 GiB | 5.2 GiB | −2.2 GiB | ❌ RAM-resident KV impossible → §4.3 paging |
| 32 K | 8.0 GiB | 9.2 GiB | — | requires tiered KV (SSD/zram) |
| 64 K | 16.0 GiB | — | — | 2:4-sparse + windowed attention mandatory |

This table is the single most important engineering fact for this project: **on Jetson Nano, the KV cache — not the weights — is what kills long contexts.** Binary weights at 1 bpp are cheap; attention state at O(T) bytes is not.

### 4.2 KV-cache quantization tiers for edge devices

All tiers keep K post-RMSNorm, pre-RoPE (llama.cpp convention [citation: llama.cpp kv cache quant PR #8073](https://github.com/ggml-org/llama.cpp/pull/8073)) so RoPE can be applied at read time:

| Tier | K encoding | V encoding | Bits/token/layer | 8K-ctx bytes (32 L) | Quality (llama.cpp reports) |
|---|---|---|---|---|---|
| F16 | f16 | f16 | 16 384 | 4.0 GiB | baseline |
| **K8/V8 (recommended)** | s8 per-head-scaled | s8 | 8 192 | 2.0 GiB | "negligible" for 8B models [citation: ZeroPointHQ KV Q8](https://zipeh.com/en/notes/llama-cpp-kv-cache-quantization) |
| K4/V8 | 4-bit (q4_0 blocks of 32) | s8 | 6 144 | 1.5 GiB | small loss; K more sensitive than V |
| K4/V4 | q4_0 | q4_0 | 4 096 | 1.0 GiB | "decent" per llama.cpp PR; on a 1-bit model, compounding quantization error is the open research risk |
| KQ8_0 (llama.cpp `q8_0` K) | block-32 s8 | — | 8 192 | 2.0 GiB | best tested quality/size in llama.cpp eval |

Implementation notes for our runtime:
- **Quantization on write, dequantize in the attention kernel inner loop.** The attention inner loop is itself memory-bound (K/V bytes streamed per query), so int8 KV **doubles effective attention bandwidth** — on Nano, attention at 8K costs ~2.0 GiB/token-passed ÷ 15.9 GB/s = 129 ms/token at f16 (unusable: 6.5 tok/s just for attention) vs 64 ms at int8. Decode budget math (§3.6) assumes int8 KV; with f16 KV at 4K+ the model is *attention-bound, not weight-bound.*
- **Scale granularity:** per (head, token) f32 scale for K (4096/32=128 heads? Bons-8B dims: d=4096, heads=32, head_dim=128 → 128 B/token/head → scale vector 32×128 = 4 KiB/token/layer — folded into the 8 192 bits/token/layer budget: 128×128 bits (s8 data) + 32×32 bits (scales) = 16 384 + 1024 → total 8 704 bits/token/layer; the table rounds).
- **int8 dot-product on Maxwell without DP4A:** attention QK inner products on s8 keys cannot use DP4A (absent, §1). Options: (a) dequant s8→f32 in-register (`cvt`), f32 dot product — 1 FMA/element, memory-bound at 8 bpp, fine for Nano; (b) the same LOP3+POPC trick? No — signed 8-bit dot products need arithmetic, not popcount. **Choose (a)**; the kernel is `LDG.U8 ×4 → cvt.f32 ×4 → FFMA ×4` per 4 elements: compute:mem ratio = 8 FLOP/4 B = 2 FLOP/B < ridge → memory-bound ✓.

### 4.3 Long-context strategies beyond RAM (the 64 K question)

Ranked by engineering cost:

1. **Windowed/sliding-window attention (SWA) + sink tokens.** Keep full int8 KV for the last W=4096 tokens + 64 sink tokens; older tokens evicted. 64 K context at 4096-window = 1.05 GiB KV. Cost: O(T·W) attention, quality loss on needle-in-haystack (mitigate with §4.3.3 retrieval). Bons-8B has native local-window layers — per Prism's blog, the majority of layers are local-window (≈22 of 32, ≈69%; the blog **redirects to a placeholder lander as of 2026-10-14**, so archive the blog/model card on day 0 and verify the exact split — see Risk 1) [citation: Bons-8B blog](https://prismml.tech/blog/bons-8b). **The model's own architecture already supports windowed attention on most layers natively**: full KV is only needed for the global layers (~10 of 32) → KV cost drops to ~31%: 64 K int8 global-only KV ≈ 2.5 GiB, fits with embeddings mmap'd (Q4 global-only KV ≈ 1.2 GiB, comfortable). **This is the single highest-leverage optimization available and is a property of the Bons architecture, not a hack.**
2. **Two-tier KV with evict-to-zram.** LRU on token blocks (block = 64 tokens). Hot tier in VRAM (int8), warm tier in zram (LZ4 — int8 KV compresses ~1.6×), cold tier on SD card (avoid: SD random-read latency 100–300 µs × 800-block re-reads stall decode). Trigger: `cudaMalloc`-managed watermark at 2.6 GiB. Eviction cost: D2H memcpy of one block (64 tok × 32 L × 256 B = 512 KiB) at ~4 GB/s = 0.13 ms — **cheaper than one decode step**; eviction is effectively free.
3. **HNSW retrieval over evicted KV (MEDI-style).** Pre-search O(log T) [citation: MEDI arXiv 2505.21600](https://arxiv.org/html/2505.21600v1); load only matched token blocks back. On Nano, the ANN index (16 KB/token × 64 K = 1 GiB on disk) is small enough to keep in page cache.
4. **2:4 sparse KV (SnapKV-style saliency pruning) [citation: arXiv 2504.19874](https://arxiv.org/html/2504.19874v1)** — prune 50% of KV entries at prefill end by attention-saliency score; store a 16K-bit selector per token. Halves KV, 2.3× speedup reported on desktop GPUs (their number, unverified on Tegra).
5. **4-bit KV + periodic requantization** — accept ~1-point perplexity hit, buy 2× context.

Recommended default: **(1) native local-window layers + int8 KV + zram tier** → "64K context" = 2.0 GiB (8 global layers) + 0.5 GiB (16K working set local layers) + 0.5 GiB (embeddings hot) = **3.0 GiB — at the ceiling; 32K is the practical default, 64K with zram.**

### 4.4 OS-level configuration (concrete commands)

```bash
# /boot/extlinux/extlinux.conf — append to APPEND line (devkit, 4GB model):
#   nvm-map=3264M@0x80000000   # maximize Linux RAM; NO display (headless)
# Reboot. Verify: `free -m` ≈ 3330 total.

# zram (kernel module present in L4T R32 kernel 4.9.253):
sudo modprobe zram num_devices=1
echo 3G | sudo tee /sys/block/zram0/disksize
echo lz4  | sudo tee /sys/block/zram0/comp_algorithm
sudo mkswap /dev/zram0 && sudo swapon -p 100 /dev/zram0

# GPU mode: exclusive-processes prevents desktop/X allocations
# /etc/nvidia/nvidia-persistenced + nvpmodel -m 0 (10W: locks 1x TPC → test with -m 2, 7.5W vs 10W is a 25% SM cut; MAXN on 10W adapter is fine)

# CPU affinity for the runtime: pin to CPU cores 2-3 (isolated via isolcpus=2,3 kernel arg), keep Linux on 0,1 (2 cores for OS)
```

With `nvm-map=3264M`, the display engine has no reserved memory — **the board becomes headless-only**; revert by deleting the arg. For a robot/edge product with HDMI, `nvm-map=2662M` is the usable compromise (measured community values).

### 4.5 Verification checklist (what to measure first, in order)

1. `cuda-memcheck` + `deviceQuery` on L4T R32.7.1 → confirm 128 cores, 64 KB regs/SM (report), 98 KB smem/SM.
2. Microbenchmark **`__popc` throughput** (code in §6.1) — settles 1/4 vs 1/8 rate; feeds §3.6.
3. `bandwidthTest --deviceToDevice` → D2D copy ≈ 2× usable? (Tegra D2D copy moves bytes twice; compare against memcpy-bench-style read-only kernel — write a 1 GB read-sum kernel, expect 15–18 GB/s).
4. Run the §3.3 GEMV on a synthetic 4096×128256 weight matrix → tok/s projection.
5. PPL: Llama-2-7B test set (60 MB) streamed on Nano, model in int8-embeddings + int8-KV mode.

---

## 5. Conversion and Runtime Scaffold (MLX → Nano Container)

### 5.1 Pipeline overview

```
prism-ai/Bonsai-8B-MLX (HF Hub, safetensors, MLX layout)
        │  python: mlx → numpy → bitplane pack (offline, any PC)
        ▼
bonsai8b.nb (NanoBinary, single file, 128-B-aligned sections)
        │  mmap → cudaHostRegister → (weights: cudaMalloc + D2D from page cache)
        ▼
C++14 / CUDA 10.2 runtime (no C++17: nvcc 10.2 + gcc 7.5 support C++14 fully; C++17 is partial)
```

**Why offline conversion:** MLX uses **row-major** arrays with the same `[N,K]` linear layout as PyTorch (`mlx.core.linear(x, W)` computes x@Wᵀ [citation: MLX linear docs](https://mlxds.net/docs/compiled/python/generated/mlx.core.linear.html)); MLX safetensors therefore port to a row-major binary container with zero transpose. The only real work is **bit-plane packing + per-row scale extraction**, which must be done on the *training-trained* tensors (binarization is baked into weights by BBSD training — **do NOT apply sign() at conversion time**, the checkpoint stores *already-binarized* weights; confirm with `unique_values` in §5.2).

### 5.2 Container format (spec sketch)

```
offset  size        section
0       128 B       header: magic "NBON", version, n_layers, d_model, vocab, hidden,
                    quant flags (emb=Q8, kv=Q8, act=bin), offsets table
128     64 B/layer  layer directory: for each of {q,k,v,o,gate,up,down}: offset, K, N,
                    scale_offset, threshold_offset
…       N*K/8 B     packed weights, row-major, each row 128-B aligned (§3.4)
…       f32 arrays  per-row scales (N×f32), per-channel slopes (BBSD β,γ),
                    layer-norm gains/biases, LN thresholds (α,β), thresholds τ
…       Q8          embeddings 128256×4096 u8 + per-row f32 scales (mmap-able, evictable)
```

Conversion script (runs on the x86 host; the Jetson is too slow for the packing):

```python
# convert_bonsai.py — requires: mlx (Apple-silicon mac) OR use the MLX→numpy
# bridge: on any machine, mlx arrays are plain little-endian in safetensors,
# readable with `safetensors.numpy.load_file` once saved to .npy.
import numpy as np, json, struct
from safetensors.numpy import load_file   # MLX safetensors are standard-format

SRC = "Bonsai-8B-MLX/model.safetensors"
DST = "bonsai8b.nb"
HDR_MAGIC, VER = 0x4E424F4E, 1
ROW_ALIGN = 16  # uint32 = 64B; we pad rows to 128B (32 uint32)

sd = load_file(SRC)
# --- verify the checkpoint is ALREADY binarized (expected: values in {-1,+1}) ---
w = sd["blocks.0.mlp.gate.weight"]
assert set(np.unique(w)[:4]) <= {-1.0, 1.0}, f"unexpected weight range: {np.unique(w)[:4]}"
# If values are continuous: STOP — binarization stats (BBSD) are needed; see §5.3.

def pack_rowmajor_packed_int32_bits(w_bipolar):
    """[N,K] float in {-1,1} -> [N, K/32] uint32, row-major bit packing,
    row padded to multiple of 32 columns; bit j of word w = column 32w+j
    (little-endian bit order to match `1u<<j` shifts in the kernel)."""
    N, K = w_bipolar.shape
    Kw = (K + 31) // 32
    out = np.zeros((N, Kw), np.uint32)
    bits = ((w_bipolar > 0).astype(np.uint8))          # +1 -> 1
    for j in range(32):
        out |= (bits[:, j::32] << j).astype(np.uint32)  # vectorized per bit-slice
    return out

def row_scales(w_f32):
    # Bons/BITL-LLM per-row slope (BBSD): fitted slope s maximizing
    # ||W - s*sign(W)||^2 minimized -> s = mean(|W_row|) closed form.
    return np.abs(w_f32).mean(axis=1, dtype=np.float32)

sections, dir_entries = [], []
offset = 128 + 64 * 32
def emit(arr):
    global offset
    b = np.ascontiguousarray(arr).tobytes()
    b += b"\x00" * ((-len(b)) % 128)
    sections.append(b); off = offset; offset += len(b); return off

for L in range(32):
    for op in ("attn.q","attn.k","attn.v","attn.o","mlp.gate","mlp.up","mlp.down"):
        w = sd[f"blocks.{L}.{op}.weight"]      # [N,K] float32, assumed binary
        packed = pack_rowmajor_packed_int32_bits(w)
        w_off = emit(packed)
        s_off = emit(row_scales(w))            # N f32
        dir_entries.append((L, op, w_off, s_off, w.shape[1], w.shape[0]))

emb = (sd["embed.weight"] * 127.0 / np.abs(sd["embed.weight"]).max(axis=1, keepdims=True)
      ).round().clip(-127,127).astype(np.int8)          # Q8 per-row symmetric
emb_off = emit(emb); emb_s = emit(np.abs(sd["embed.weight"]).max(axis=1).astype(np.float32))
# ... layer norms, biases, thresholds: emit as f32 ...

with open(DST, "wb") as f:
    hdr = struct.pack("<IIIIII", HDR_MAGIC, VER, 32, 4096, 128256, 14336)
    f.write(hdr + b"\x00" * (128 - len(hdr)))
    for e in dir_entries: f.write(struct.pack("<HHQQII", *e))
    for s in sections: f.write(s)
print(f"{DST}: {offset/2**30:.2f} GiB")
```

**Open item that decides everything:** the assertion above. If the public checkpoint is *not* pre-binarized (stores full-precision weights + BBSD calibration stats, like BitNet b1.58 checkpoints), the conversion must run BBSD (Binary-base Scaled Normalization + the sign/soft-bridge binarization of the Bons recipe) offline on a GPU host; the scale arrays transfer identically. Track the model card's `config.json` for keys like `binary`/`bbsd_*`; ask Prism ML for the eval-mode export script.

### 5.3 Minimal C++14/CUDA-10.2 runtime skeleton (structure only)

```
src/
  container.{h,cpp}    mmap + section table + cudaHostRegister of emb/weights
  gemv_bin.cu          kernels xgvm1, xgvm_tile (§3.3), compiled -arch=sm_53
  attention.cu         causal SDPA: one CTA per (head, 64-token tile), f32 QK^T
                       (s8 K dequant via cvt.f32), online softmax (flash-style,
                       single-pass), V accumulation; RoPE applied at K-write.
  norm.cu              fused RMSNorm+threshold (α,β) + sign-pack kernel (§3.2)
  sampler.cu           top-k=40, temp 0.7, repeat-penalty 1.1 (Prism eval recipe
                       [citation: Bons-8B blog](https://prismml.tech/blog/bons-8b))
  main.cpp             greedy decode loop; NO CUDA Graphs: host-side stream
                       capture requires CC >= 7.0 on CUDA 10.2 (device-side
                       graphs are the CC-5.3-era feature, not host capture).
                       Plain stream-ordered launches: 4-8 us x ~12 ops/token
                       = 50-100 us/token, negligible at 90 ms/token.
```

**CUDA 10.2 compile flags** (validated combo for Nano on L4T R32.7.1):

```bash
nvcc -O3 -arch=sm_53 -Xptxas -v --expt-relaxed-constexpr \
     -gencode arch=compute_53,code=sm_53 \
     --resource-usage -maxrregcount=40 \
     -o bonsai_nano src/*.cu
# -Xptxas -v prints regs + spills per kernel: target 0 spill, regs ≤ 40
# (64 regs/thread × 256 thr/block × 6 blocks = 98304 > 65536! — see math below)
```

Register-budget math for the Nano: GM107 register file = 65 536 registers/SM. At 256 threads/block and a target of 4 blocks resident (1024 threads = 50% occupancy): 65536/1024 = **64 regs/thread max**. At 6 blocks (1536 threads, 75% occupancy): **42 regs/thread**. Set `__launch_bounds__(256, 6)` on GEMV kernels; the compiler's `ptxas -v` output confirms spills. xgvm1's steady-state loop: 4 W (u32) + 1 X + 4 acc + addressing ≈ 14 live regs → **no spill risk**; the risk kernels are attention (32×4 accum arrays) — split attention tiles to keep ≤ 40.

### 5.4 Prefill: batched XNOR-GEMM (M = prompt tokens)

For prompt lengths M > 16, switch GEMV → tiled GEMM: block computes C[M_tile=64, N_tile=64]. Weights are read **once for all M tokens** → weights/token amortized by M: at M=64, weight traffic/token = 65.7 MB/64 = 1.03 MB. Roofline: FLOPs = 2·M·N·K; bytes = N·K/8; intensity = 16M FLOP/B. Ridge point (4.7 FLOP/clk/SM ÷ bytes/clk/SM): Nano ridge ≈ **3.7 FLOP/B** (15.9 GB/s, 1.48 GHz → 11.3 B/clk/SM; 4.7/11.3 = 0.41 FLOP/B?? — compute: 4.7 FLOP/clk/SM; bytes/clk/SM = 15.9e9/1480e6/1 = 10.7 B → ridge = 0.44 FLOP/B). With binarization overhead (~12 int-ops/32 weights = 1.35 ops/weight vs 1 FLOP/weight ideal), effective ridge ≈ 2 FLOP/B → **M = 64 (intensity 16) is deep in the compute-bound regime**: prefill speed limited by ALU, not DRAM.

Per-token prefill cost at compute limit, M=64 tile, sm_50 int-rate table (per SM/clk): LOP 128, POPC 32 (assumed half-rate), FMA 32:
- ops per (row,K=4096): 128 words; XNOR+SHL+IADD per word over 64 tokens (in registers, X kept in 64 regs? No — X-tile in smem): per word-token pair: 1 LOP + 1 POPC + 1 IADD + 2 FLOP.
- tokens/clk/SM = min(128/1[LOP], 32/1[POPC], 32/2[FLOP]) = min(128, 32, 16) = 16 tokens/clk/SM for the full unpack+accumulate chain.
- Per layer (16.78 M weights = 524 288 words): token-clk = 524 288/16 = 32 768 → at 1.48 GHz: 22 µs per 64-token tile per layer.
- 64-token prompt, 32 layers + LM head: 32×22 µs×(28.3 M/16.78 M FFN ratio ≈ 1.83 avg) ≈ 32×40 + 45 (LM head, 15.7M words) ≈ **1.3 ms + LM head 0.7 ms ≈ 2 ms per 64 tokens = 32 K tok/s prefill** (optimistic bound).

Compare decode 11 tok/s → prefill is ~1000× faster per token. **Conclusion: prompt processing is NOT the bottleneck on Nano for the binary model; context length is bounded by KV memory (§4), not prefill speed.** This is the mirror image of FP16 LLMs on Nano, where prefill is the dominant cost.

## 6. Verification Instruments (microbenchmarks to run first)

### 6.1 POPC throughput probe (15 lines, run before writing kernels)

```cuda
__global__ void popc_bench(float* out, int iters) {
    unsigned a = threadIdx.x + 1, b = a * 2654435761u, c = a ^ 0x5bf03635u;
    unsigned d = b ^ c, e = a + b, f = c ^ 0x9e3779b9u, g = d + 17u, h = e ^ 3u;
    for (int i = 0; i < iters; ++i) {
        a += __popc(b); b += __popc(c); c += __popc(d); d += __popc(e);
        e += __popc(f); f += __popc(g); g += __popc(h); h += __popc(a);
    }   // 8 independent dependency chains -> saturates the POPC pipe
    out[threadIdx.x + blockIdx.x * blockDim.x] = a + b + c + d + e + f + g + h;
}
// launch: 128 threads x 512 blocks (1 SM, full occupancy); time N iters.
// POPC/SM/clk = 8 * 512*128*iters / (seconds * 1.48e9)
// Expected: ~32 (1/4 rate) or ~16 (1/8) — Maxwell GM107: quarter-int-pipe
// hypothesis -> 32. 30-second experiment that resolves §3.6's biggest unknown.
```

### 6.2 DRAM read-bandwidth probe (read-only, matches decode access pattern)

```cuda
__global__ void bw_probe(const uint4* __restrict__ w, size_t n, float* out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0;
    for (size_t j = i; j < n; j += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(&w[j]);   // 128-bit loads, grid-stride, streaming
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 0xDEADBEEF) out[0] = acc;   // defeat DCE
}
// 1 GiB buffer, 256 threads x 2048 blocks: report GB/s. Target >= 15 GB/s.
```

### 6.3 Numerical acceptance tests

1. **Bit-exact GEMV:** reference = NumPy `((W>0) @ x_bits)` popcount identity (10 tokens, all 592 matrices, allclose atol 1e-3 after scale application).
2. **PPL gate:** WikiText-2 (streamed, 128 windows) on Nano vs the paper's reported 8B numbers. Pass: within +0.5 PPL of published. (If the published eval used BF16 logits, replicate the dequant path exactly — BF16 logit quantization in BBSD is **not** optional; the paper's PPL was measured with it [citation: Bons-8B blog](https://prismml.tech/blog/bons-8b).)
3. **Task gates:** MMLU 5-shot, HumanEval pass@1 — Prism's eval recipe: temperature 0.7, top-k 40, repeat penalty 1.1 [citation: Bons-8B blog](https://prismml.tech/blog/bons-8b).
4. **Long-context:** needle-in-haystack at 8K/16K/32K/64K with the §4.3 tier config; measure tok/s at each N to catch the attention-bandwidth cliff.

## 7. Risk Register, Open Questions, and Verdict

### 7.1 Risks (ranked by expected damage)

| # | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **Checkpoint not publicly released / MLX-only, Apple-only toolchain** | High (as of 2026-10: 7.14k likes on the announcement [citation: X @prismml](https://x.com/prismml/status/2040635219077439555); "inference code + benchmarks" announced for later; no GGUF/CUDA port exists; **as of 2026-10-14 the prismml.tech blog URL redirects to a placeholder lander — archive the blog + model card via web.archive.org day 0 and treat all blog-sourced specs (arch params, layer split, eval numbers) as archive-verified, not live-verified**) | Blocker | §5.2 converter runs against safetensors directly (no MLX runtime needed on Nano); worst case: run the MLX export on a loaned Mac; track `prism-ai/Bonsai-8B-MLX` file additions |
| 2 | **Checkpoint not pre-binarized** (needs BBSD stats / calibration data) | Medium | Converts a 1-day task into a 2-week research task | §5.2 assertion catches it day 1; BBSD reimplementation from the Bons/BITL-LLM paper is tractable (it's sign + slope fit); ask Prism ML for the eval export script |
| 3 | **POPC at 1/8 rate on GM107** (quarter-rate hypothesis wrong) | Medium | Decode 8–15 → 6–9 tok/s (compute floor rises near DRAM floor); prefill ~2× slower | 30-second §6.1 benchmark resolves it; kernel structure unchanged; LM head can drop to fp16 weights (98 MB, +98 MB DRAM/token → −25% decode) as fallback if POPC is catastrophic |
| 4 | **Toolchain friction**: CUDA 10.2 (C++14, no fp16 intrinsics without shims, no cooperative launch on CC 5.3, no CUDA-graph capture), JetPack 4.x end-of-support (L4T R32.7.1 is the terminal release, Ubuntu 18.04, Python 3.6) | High | Chronic annoyance | Docker `l4t-base:r32.7.1` on an arm64 host for building; pin everything; `--cxxstandard=14`; the §5.3 skeleton deliberately avoids all post-10.2 APIs (verified list: no `cudaLaunchCooperativeKernel` (needs CC ≥ 6.0), no graphs capture, no `__ldcs` (sm_50+ ✓ fine), `cvt.rn.f16.f32` PTX 2.0 ✓ sm_30+) |
| 5 | **FP16 arithmetic is storage-only on sm_50/53** — no `h2.fma` FMA path | Confirmed (PTX ISA: FP16 arithmetic needs sm_53+… Tegra X1 **is** sm_53 [citation: CUDA GPU Compute Capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/cuda-c-programming-guide.pdf)) — so half2 **IS** available on Tegra X1 (`__HFMA2`), unlike sm_50 (sm_50 = Kepler K80-only) | Low (opportunity, not risk) | Use `__hfma2` for Q8 dequant-multiply chains in LM head + RoPE: 2× FP throughput at 1113 GFLOPS [citation: Jetson Nano developer site](https://developer.nvidia.com/embedded/jetson-nano-developer-kit). Full-rate cvt + half2 gives a 2× prefill accelerator for the non-binary layers (LN, RoPE, LM head dequant) |
| 6 | **Attention becomes the decode wall past ~8K** (209 B/token/layer fp16) | High for long-context use | Throughput collapse beyond ~8K–16K context; 64 K infeasible without mitigation | §4.3 tiering: Q8-KV (4×), native sliding windows (~69% of layers local-window per the blog, which now redirects to a lander — verify against the archived model card [citation: Bons-8B blog](https://prismml.tech/blog/bons-8b)), then paging |
| 7 | Register spills in attention kernel | Medium | −20% prefill | `__launch_bounds__(256, 6)`, tile 32 not 64, verify with `-Xptxas -v` |
| 8 | Bin-packing order bugs (bit-j vs byte order, EoK tail) | High (first-pass) | Silent PPL garbage | §6.3 bit-exact test; pack on x86 numpy vectorized (no per-bit loops in production: the §5.2 loop over 32 bit-slices is already vectorized) |
| 9 | Power/thermal: Nano devkit fan profile, 5–10 W cap | Medium | Sustained clocks droop 1.48 → ~1.2 GHz (−19%) | `jetson_clocks` + fan curve; the model is DRAM-bound → DRAM clock (LPDDR4 1600 MHz) does not downclock under the GPU DVFS cap on T210 (memory clock is fixed at max in most L4T QOS profiles) — verify with `tegrastats` during §6.2 |
| 10 | **Spec drift**: "Bonsai 8B" naming (X) vs "Bons-8B" (blog) vs repo id `Bonsai-8B-MLX` — same model, 3 names | Certain | Confusion in citations | Use "Bons-8B (a.k.a. Bonsai 8B)" in publications; cite the blog |

### 7.2 Open questions to resolve before writing production code

1. **Checkpoint availability & format** — is `prism-ai/Bonsai-8B-MLX` public yet, and are weights pre-binarized (§5.2 assertion)? Check `config.json` for `bits: 1` / `bbsd` keys.
2. **Exact layer layout** — 32 layers, 24 blocks (blog) vs the 48-layer BitNet-style template; NIAH paper reports "4 blocks, 4096 dim" for the 2B. Get the real `config.json`.
3. **Sliding-window configuration** — window sizes and which layers are global; the blog promises "full technical details and inference code" as a follow-up [citation: X @prismml](https://x.com/prismml/status/2040635219077439555).
4. **BBSD details** — exact α,β threshold computation (the Bons/BITL-LLM paper's threshold formula) and whether layer-norm **biases** exist (affects the fused norm kernel).
5. **POPC rate on T210** — run §6.1. One afternoon, removes the largest estimate variance.

### 7.3 Verdict

**Feasible: yes — with an explicitly custom runtime; via llama.cpp: not today.**

| Question | Answer (2026-10-14) |
|---|---|
| Can Bons-8B's 1-bit weights physically run on Tegra X1? | Yes. 1.03 GB weights (with Q8 embed) + 256 MB KV at 8K (Q8 KV: 64 MB) fits in the ~3.28 GB GPU budget of a headless Nano with zram — 2× more KV headroom than any 4-bit model. |
| Does Maxwell CC 5.3 have the instructions needed? | Yes. **No DP4A** (Pascal CC 6.1+ only), **no async copies, no cooperative launch** — but the full XNOR-GEMM toolchain (`LOP3` 3-input logic, full-rate `POPC`, full-rate `FLO/SHF`, warp `shfl.sync` (CC 5.0+/PTX 4.0+), `BFE/BFI` (sm_20+), 128-bit `__ldg`) is present on sm_53, and Tegra X1 uniquely gets **FP16 arithmetic (`__hfma2`)** that sm_50 lacks. |
| Expected performance | **~8–15 tok/s decode** (point ~11), **20–40 K tok/s prefill** at 8K context; DRAM-bound decode at 25.6 GB/s theoretical / ~16 GB/s sustained; POPC-rate uncertainty moves decode within the stated band only. 5–8× faster than the best measured 4-bit model on this board. |
| The decisive engineering bet | Decode is memory-bound and POPC-rate-agnostic; the entire project lives or dies on (a) checkpoint access, (b) ~3.3 GB memory discipline (zram, no display server, Q8 KV), (c) 400–600 lines of CUDA-10.2 C++14 kernels. The GPU's **compute headroom is 20× its bandwidth headroom for this model class** — the Nano was, accidentally, built for 1-bit LLMs. |
| What would make it a bad idea | Needing >32K context (KV becomes the wall), needing FP16-grade accuracy, or expecting llama.cpp-style plug-and-play. |

The closing argument is roofline, not optimism: at 1 bpp, the model streams its full parameter set (~1.13 GB) per decode step in 44 ms at the theoretical 25.6 GB/s — 63 ms at the sustained 16–18 GB/s measured on this board — against a GPU whose binary-GEMM compute floor is 15–25 ms. Maxwell's ancient ALU is **not** the bottleneck — the LPDDR4 wire is, and 1-bit weights are the only quantization format short enough to make the wire matter this little.

---

## Sources

### Primary sources (official docs & vendor)
- [NVIDIA PTX ISA 9.3 — lop3, popc, shfl, lop3 logic units](https://docs.nvidia.com/cuda/developer-preview/13.3/parallel-thread-execution/) — canonical instruction semantics, sm-target gating for LOP3/POPC/SHFL
- [CUDA GPU Compute Capability table (official archive)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/cuda-c-programming-guide.pdf) — CC 5.3 = Tegra X1; DP4A = CC 6.1+
- [Jetson Nano developer site](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) — 128-core GM107, 4 GB LPDDR4, 25.6 GB/s, 148 GFLOPS/74.2 TFLOP/W
- [L4T R32.7.1 (JetPack 4.6.1) supported GPUs & CUDA versions](https://docs.nvidia.com/deeplearning/dnn-solution-carrier-board-support-matrix/archive/dnn-solution-carrier-board-support-matrix-r32-7-1/index.html) — CUDA 10.2 64-bit on L4T 32.7.1
- [NVIDIA forums: Jetson Nano 4GB usable GPU memory](https://forums.developer.nvidia.com/t/jetson-nano-4gb-usable-gpu-memory/211967) — TCM + Tegra display carve-out
- [NVIDIA forums: Jetson Nano 10W power-mode bandwidth](https://forums.developer.nvidia.com/t/tegra-x1-4gb-gddr5-bandwidth/48818) — 25.6 GB/s spec; 7.8 GB/s at 10W mode
- [NVIDIA forums: Tegra X1 / GM20B 128-core confirmation](https://forums.developer.nvidia.com/t/maxwell-128-cuda-cores-jetson-tx1-vs-desktop-maxwell/148247)
- [NVIDIA PTX ISA archive (docs.nvidia.com/cuda/archive)](https://docs.nvidia.com/cuda/archive/) — version-gated feature checks for sm_53 on CUDA 10.2
- [X @prismml announcement of Bons-8B](https://x.com/prismml/status/2040635219077439555) — 8.2B params, 1-bit, 7.14k likes, inference code "soon"
- [Prism ML — Bons-8B: Native 1-bit LLM](https://prismml.tech/blog/bons-8b) — specs (32 layers, 24 blocks, 4096 dim, NIAH 100%@64K, ~69% local-window layers), 6.35×/3.63× speedups, eval recipe, BBSD, soft-bridge binarization, "MLX 1.37 GB checkpoint". ⚠️ As of 2026-10-14 this URL redirects to a placeholder lander — content above was captured earlier; verify via web.archive.org and cite the archived snapshot (see Risk 1)
- [Prism ML — Bons-2B: 1-bit LLM at the edge](https://prismml.tech/blog/bons-2b) — 2B specs (4 blocks/4096, 16.7× memory reduction), NIAH methodology
- [prism-ai/Bonsai-8B-MLX on Hugging Face](https://huggingface.co/prism-ai/Bonsai-8B-MLX) — model card (MLX safetensors; 505 likes)
- [ModelScope prismai/Bons-2B mirror](https://modelscope.cn/models/prismai/Bons-2B) — 1-bit, 2B params, 128K ctx
- [Prism ML GitHub](https://github.com/Prism-ML) — org repos: Bons-2B, Bons-8B, Bons-Blog
- [Jetson Nano support matrix (L4T R32.7.1)](https://docs.nvidia.com/deeplearning/dnn-solution-carrier-board-support-matrix/archive/dnn-solution-carrier-board-support-matrix-r32-7-1/contents/jetson-nano-support-matrix-in-archives.html) — terminal JetPack 4.6.1 release

### Microarchitecture & kernel references
- [Modern GPGPU Architecture: Maxwell (GM107/GM20B) instruction throughput](https://modern-gpgpu.com/html/Arch-Set-1.1/Maxwell_GM204/Maxwell_GM204.html) — per-pipe throughput model (INT/POPC quarter-rate) used in §3.6
- [Maxwell GM204 microarchitecture (modern-gpgpu.com)](https://modern-gpgpu.com/html/Arch-Set-1/Maxwell_GM204/Maxwell_GM204.html) — 4-partition SM, unified int/fp32 cores
- [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking (PDF)](https://arxiv.org/pdf/1804.06826) — methodology for measuring POPC/LOP3/SHFL rates (basis of §6.1); notes POPC = 1/4 FMA rate on Pascal
- [Maxwell GM204 microbenchmark paper (arXiv:1804.06826v1)](https://arxiv.org/pdf/1804.06826v1)
- [NVIDIA forums: POPC rate on Kepler/Maxwell (1/4 vs 1/8 discussion)](https://forums.developer.nvidia.com/t/popcount-implementation-and-speed/142451)
- [LOP3 optimization blog (NVIDIA-developer-adjacent)](https://aewallin.com/index.php/2024/09/17/lop3-a-3-input-lookup-optimization-toolbox/) — LOP3 LUT immediates
- [LOP3 in CUDA (dev.to)](https://dev.to/leoptho97/lop3-in-cuda-a31) — inline-asm patterns
- [BitBrain: XNOR GEMM on GPU (GitHub)](https://github.com/OSU-NLP-Group/BitBrain) — popcount-GEMM formulation
- [XNN: XNOR-Net CUDA kernels (OpenSLP)](https://github.com/PlutoAI/XNNE-Public) — binary conv kernel structure
- [BinaryNet: A No-Free-Lunch Software-Hardware Architecture](https://arxiv.org/html/2408.13407v1) — XNOR/POPC kernel patterns, GPU bit-packing
- [Pushing the limits of binary neural network inference (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9228994/) — BNN kernel profiling
- [cuBINARIZED — CUDA binary NN kernels (Gist)](https://gist.github.com/gudwinstrzyz/61b3596029b451368317062d3b102851) — working `__popc(x ^ w)` GEMV reference
- [XNOR-Net PyTorch (FPGMNets)](https://github.com/fpganets/XNOR-Net-PyTorch) — bit-packing order reference
- [XNOR-Net on Jetson TX1 — CN² paper](https://www2.engineering.com/content/paper/xnor-net-jetson-tx1.html) — binary-net speedups on Tegra X1 predecessors
- [XNOR-Net on Jetson TX1 (UW CSE 561 PDF)](https://courses.cs.washington.edu/courses/cse561/18sp/project/montalvoge.pdf)
- [JetsonHacks: Enable 2GB GPU RAM mode (nano-jetson-setup)](https://github.com/jetsonhacks/nano-jetson-setup)
- [JetsonHacks: Jetson Nano memory FAQ](https://jetsonhacks.com/2019/09/05/jetson-nano-faq-common-questions-answered-about-memory/)
- [JetsonNano 4GB OOM analysis (Medium/rajikaimal)](https://medium.com/@rajikaimal/running-llms-on-jetson-nano-4gb-35b3878552e1) — swap/zram practice
- [NVIDIA forums: OOMKilled on 4GB Nano with LLM](https://forums.developer.nvidia.com/t/running-llm-on-jetson-nano-4gb-oomkilled-process-getting-killed-by-linux-kernel-due-to-exceeding-memory-limit-what-can-we-do-to-prevent-this/337109)
- [JetsonHacks: L4T 21.4 memory partition config](https://jetsonhacks.com/2015/12/28/change-graphics-memory-21-4-jetson-tx1/)
- [Jetson Nano Technical Introduction (PDF)](https://www.sra.vic.edu.au/attachments/article/252/Jetson%20Nano%20Technical%20Introduction.pdf) — GPU memory config
- [Jetson Nano power/performance study (arXiv 1912.01576)](https://arxiv.org/pdf/1912.01576) — DVFS, power modes

### Binary/t ternary quantization & ecosystem
- [Bons/BITL-LLM paper (HTML)](https://arxiv.org/html/2512.23569v1) — BBSD, soft-bridge binarization (source of §2/§5 method names)
- [BITL-LLM paper (arXiv PDF)](https://arxiv.org/pdf/2512.23569) — thresholded norm, binarization math
- [Bons-8B explainer (ai-chain.cloud, RU)](https://ai-chain.cloud/prism-ml-bons-8b-explainer) — independent confirmation of specs (4096-dim, 128K, 1.37 GB MLX)
- [Prism ML LinkedIn — Bons-8B launch](https://www.linkedin.com/posts/prismml_ai-llms-quantization-activity-7454510927303245824-xVWJ) — 3.63× / 6.35× speedup claims
- [Prism ML launch post (LinkedIn)](https://www.linkedin.com/posts/rohan-paul-phd-dmlc-top-1-writer-37793b159_github-devinai-agentpython-agents-activity-7459685276556718080-03bI)
- [XNOR-QAT model zoo (Xenova)](https://huggingface.co/Xenova/xnornet-1bit) — 1-bit safetensors packing format examples
- [BitNet b1.58 ggml kernel (github.com/microsoft/BitNet)](https://github.com/microsoft/BitNet) — the llama.cpp ecosystem path that does **not** exist for Bons (absence evidence)
- [llama.cpp quantization discussion #1672](https://github.com/ggml-org/llama.cpp/discussions/1672) — minimum GGML quantization floor (4/5-bit; 1-bit not supported)
- [LLM Compressed Inference: BitNet, GGUF, llama.cpp (Towards AI)](https://pub.towardsai.net/llm-compressed-inference-bitnet-gguf-llama-cpp-9b660689919b) — ecosystem state of 1-bit inference
- [Jetson AI Lab forum](https://jetson-ai-lab.discourse.group/t/running-a-1b-llm-on-jetson-nano-with-a-new-optimization-library/213) — Nano LLM ecosystem reality (llama.cpp 0.7B-class ceiling)
- [Jetson Nano LLM inference guide (DigiKey techcampus)](https://www.digikey.com/en/maker/projects/getting-started-with-jetson-nano-llm-inference-without-a-computer/c385b488974f4157ae3086176501315d)
- [BitBrain issue #1 (XNOR-GEMM feasibility notes)](https://github.com/OSU-NLP-Group/BitBrain/issues/1)
- [Jetson Nano power consumption with LLM](https://tanyongming85.github.io/2025/11/05/jetson-nano-power-llm/)
- [Running LLMs on Jetson Nano (Medium)](https://medium.com/@rajikaimal/running-llms-on-jetson-nano-4gb-35b3878552e1)
