# Autoresearch: combined conversation RTF — LFM3 S2T + PrimeTTS v2.1 (zh/en, Jetson Nano)

## Objective
Minimise the wall time per second of audio handled by a two-stage bilingual conversation turn on
the Jetson Nano (`picard@100.84.151.116`):

    user speech (zh or en) --LFM3 ASR--> transcript ; reply ids --PrimeTTS v2.1--> reply audio

## Metrics
- **Primary**: `combined_rtf` = (asr_wall_s + tts_wall_s) / (audio_in_s + audio_out_s), averaged
  over the zh and en turns (lower is better).
- Secondaries: `s2t_rtf`, `tts_rtf`, `reply_rtf` ((asr+tts)/audio_in: "answer within the length of
  the user's utterance"), `asr_wall_s`, `tts_wall_s`.

## How to Run
`./.auto/measure.sh` (= `.auto/combined/measure.sh`): two turns (zh 9.0 s clip, en 9.0 s clip) ->
LFM3 ASR cold CLI -> transcript; reply ids T81 (3.616 s) -> `pt_v21_nano`. Emits METRIC lines.
`./.auto/checks.sh`: fixture md5s, ASR transcripts == frozen goldens, PrimeTTS SNDR >= 60 dB.
The previous objective's harness is preserved at `.auto/lfm3/measure_s2s.sh`.

## Files in Scope
- `pulsar_arm/asm/lfm3_asr.c` — the ASR orchestrator (per-turn process model today).
- `pulsar_arm/asm/lfm3_nanode.cu` — the 2B backbone; has `--serve` (per-step, S2S) and `--clip`
  (N clips in one process, load amortized).
- `pulsar_arm/asm/lfm3_enc_gpu.cu`, `lfm3_mel.c` — encoder tower + log-mel front end.
- `/home/louis/Codes/prime-tts` (branch `custom-runtime`) — PrimeTTS runtime; synced to
  `/home/picard/primetts` (that copy is NOT a git repo). Its own harness is `prime-tts/.auto/`.
- `runtime/v21/src/{pt_v21_host.c,pt_v21_cuda.cu,main_v21*.c}` — TTS interpreter, kernels, drivers.
- `.auto/combined/` — this objective's harness, fixtures and goldens.

## Measured (2026-11-11, baseline run #256)
combined_rtf **2.94** (s2t_rtf 4.03, tts_rtf 0.24). Turn: 9.0 s in + 3.616 s out.
- ASR 36.2 s/turn, decomposed by stage: mel+prompt+encoder 1.97 s, **STAGE 5 (backbone) 34.1 s**.
- Of that 34.1 s: ~21.2 s is the eMMC read of the 1.95 GB Q8 backbone blob (`cat` cold = 21.2 s,
  92 MB/s; warm = 1.04 s), ~8 s parse/CRC/upload, and only **~4.6 s of real compute**
  (prefill 133 ids ~ 34.8 ms/id + decode).
- The page cache cannot save it: the ASR arena takes 2.15 GB of the board's 3.96 GB, so the blob is
  evicted between turns and every turn re-reads it (three consecutive runs: 36.26 / 36.12 / 36.08 s).
- The read is DEVICE-bound, not code-bound (measured with the page cache dropped between arms):
  cold read of the blob is 21.96 / 21.94 / 22.04 s at 1 / 2 / 4 reader threads = 88.6 MB/s every
  time. Parallelising or re-pipelining the loader cannot help; only not reading it can. (Cached it
  reads at 2-3.7 GB/s, but the 2.15 GB arena evicts it between turns.)
- TTS 0.88 s for 3.616 s of audio **including process start**; the kernel split is conv/convT/matmul
  = 124/5/50 launches, and the repo's own README reports 0.50 s warm single-request on this board.

## Plan (ranked)
1. **Resident ASR service** (the whole game): load the backbone once, per-turn cost becomes
   ~4.6 s (prefill+decode) + ~0.2 s encoder GPU instead of 36.2 s. Implementation sketch, additive
   and reusing proven code: `lfm3_nanode --serve-clip` (the `--clip` per-clip body driven from stdin,
   one spec per turn, printing the turn's ids) + `lfm3_asr --serve` (mel+encoder per turn, keep one
   nanode alive, feed spec, parse ids). Gate: per-turn ids == the frozen cold-path goldens.
   Expected combined_rtf ~0.5-0.6.
2. **Encoder resident too** (~1.9 s -> ~0.2 s per turn): the encoder is a separate process today.
3. **ASR prefill**: 133 prompt ids at ~34.8 ms/id is ~4.6 s of the per-turn cost. The system/user
   template prefix (~15 ids) is identical every turn -> prefix-cache the KV and prefill only the
   audio rows; then any batched-prefill win.
4. **TTS**: already 0.24 RTF; keep it behind the parity gate. Server mode (`pt_v21_serve`) removes
   per-turn process start (~0.4 s) once the ASR is no longer the bottleneck.
5. Overlap: ASR is GPU-heavy, TTS is CPU-core-heavy — in a multi-turn loop they can overlap, so
   aggregate conversation throughput can exceed the single-turn latency chain.

## Constraints / gotchas
- Nano memory: 3.96 GB total, the Q8 ASR arena is 2.15 GB, PrimeTTS arenas are small. Never let the
  two stages both allocate device memory without checking `free -m` (the board hard-hangs).
- `prime-tts` on the board is a plain copy, not a git repo: always sync from `/home/louis/Codes/`.
- The TTS runtime consumes **phoneme ids**, not text; the text->ids front-end is host Python
  (`server/tts_gateway.py`, g2pw + g2p_en) and is NOT in the measured path today.
- `--clip` in nanode is NOT bit-identical to the cold single-clip path as invoked (DONE steps=6 vs
  a 9-id cold transcript). Verify equality before building on it; that is why plan step 1 says
  "gate on the frozen goldens".
- Root nvprof on this board leaks GPU mappings (a later 1.7 GB cudaMalloc fails until reboot).
