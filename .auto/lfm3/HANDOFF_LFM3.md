# HANDOFF — LFM3-Audio-2B Nano latency session (state at run #180)

Objective: e2e latency (last input -> first output token) for LFM3-Audio-2B Q8
on the Jetson Nano Gen1 at `picard@100.84.151.116` (NOT the Bonsai board at
pesi@...). Engine: `pulsar_arm/asm/lfm3_nanode.cu`, blob
`/home/picard/lfm3/lfm3-audio-2b-q8.lfm3q8`. Metric `ttft_ms` = sum of step
lines 0..nprompt-1 on prompt A (23 ids); benchmark band 2258-2274, baseline
2755.41 (-17.8%). Decode steps 112.4-112.8 (session); prompt steps 97.6-99.5
(no lm_head since #139). Full attribution: 22 x 97.6 + 112.4.

Read `pulsar_arm/asm/README-lfm3-audio.md` — the quality record, TTFT
waterfall, H2 decision table, measurement section are all current.

## Measured-out — do NOT re-walk

Latency (TTFT is prefill = 22 sequential GEMV-bound steps + 1 full step):
- GEMV configs (R/TPB/KCHUNK) under all 3 codegens, 11+ structural probes —
  characterized optimum. launch_bounds(256,6) banked (-20%, #121).
- Non-GEMV kernels: 0.54% of the step (#115). Small-kernel fusion: closed.
- Batched prefill GEMV: bit-exact kernel exists (`q8_prefill_h2`, selftest
  8/8 shapes, standalone 1.15-1.45x) — INTEGRATION CLOSED (#165): ffn.w2
  reads x+row*K into step-recursive arena scratch; no per-position tiling
  reproduces it. Root cause is documented on the kernel. Reopening requires
  changing the engine's w2 x-layout = a goldens change (user call).
- Warm-up (#138) is complete: step 0 = steady prompt step (#168).
- CPU pinning null (#118), LMDBUF 2.6x worse, CUDA graphs null on this path.
- Clock pinning impossible: no jetson_clocks, sudo requiretty (#177);
  harness drop_caches is a silent no-op (benign, engine self-evicts).
- decode_ms_per_tok from measure.sh mixes prompt steps into steady-state —
  see README caveat (#176); compare decode only via step lines >= nprompt.

Quality (all closed, gates green on fresh builds, #166):
- Teacher-forced: matrix 23x23, logits0, 6-prompt logit gate 130/131,
  streams 6/7 (F 11/16 tie). Free-running divergence = tie amplification
  (#113/#114 corrected); streams gates parse the -1 prompt-step tokens.
- Divergence floor vs fp32 (#128): engine KL 2.6-5.0x bf16 floor, no growth.
- RoPE sliding-window full-rotation fix (#158) — matrix_long gate 1/7200.
- MAXTOK 4096 (#150): long runs memory-stable (#170), per-step 98->112 to
  300 positions (positional, not thermal, #160).

## Open (user decisions, all evidenced in the README table)

1. **Raw H2** (`LFM_H2=1`): CLEAN-BOARD numbers (#236/#237, spinner-era rows
   retired): TTFT -1.8% (23 pos 3x3), -1.4% (360), -5.0% (1000 pos 3x3
   #233), decode -1.7%; the -5.3..-8.0% family was 3-core-spinner
   contention inflation (see Board-state block). FORMAL gates all green at shipped
   parity: checks suite 130/131 (#182), 7-prompt streams 6/7 with the same
   L tie case (#188), matrix_long 5/7200 vs 1/7200 -- and BOTH arms 0 cells
   vs an fp16 reference to 1000 positions (#184/#192); tie-landscape argmax
   identical to shipped on both references, 133/134 bf16 / 127/134 fp16
   (#195/#196). H2=2 (rescaled) is formally DOMINATED (127/131 logit gate,
   ~0% speed, #186). Not bit-exact: needs the user's sign-off to flip.
   Reference-precision note: ~4.5% of teacher-forced positions are argmax
   ties where bf16/fp16 references disagree (#195) -- gate "misses" live there.
2. **w2 x-layout change** (unlocks batched prefill; changes goldens).
3. Audio path port (encoder + depth decoder) — scoped in #153/#157, outside
   this metric.

## Board-state / DVFS findings (this block, #203-#216)

- H2's win is context-length-dependent: -5.7% at 23 pos (hot-board control
  #205), **-7.8..-8.0% at 300/360 pos** (order-balanced pairs, two days,
  #203/#206) = ~2.7 s saved per 30-s-query prefill. In-run decomposition
  (#208): per-step saving grows 5.57 -> 7.93 ms with position. Power/clock
  coupling REFUTED (devfreq GPU clock identical 922 MHz both arms, #209);
  leading mechanism is CPU/GPU pipeline overlap with the growing CPU-side
  attention path.
- Cold-board TTFT: idle boards start at 76.8-153.6 MHz; the ~5-24 s devfreq
  ramp lands inside the measured window (+1% at 8-min idle, +7% on drift
  days, #210). `LFM_WARMMS=N` (default 0, #211/#212) absorbs the ramp in
  the unmeasured warm-up; neutral in the normal state, gates green (#215),
  RSS stable (#216). Acceptance test for the +7% case: pair on a drift day.
- /tmp/poller.sh + /tmp/clkrun.sh on the Nano sample the GPU clock (tail -f
  on sysfs does NOT work; EMC is not readable unprivileged, #214).

## Harness

- `bash .auto/lfm3/checks.sh` — full gates (rebuilds from tree).
- TTFT one-shot: ssh run with awk sum of steps 0..nprompt-1 (pattern in the
  log rows #167-#180).
- `python .auto/lfm3/longctx_h2.py` — long-context quality harness (#171).
- Env arms reach the binary as `env LFM_H2=1 ./lfm3_nanode ...` (NOT after
  `timeout` — timeout does not parse env assignments).
