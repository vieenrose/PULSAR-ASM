# K2 meeting fine-tune — native engine validation

Same fine-tune as `NOTE.md`, now running in the **native asm engine** on the
Q4_0 blob (no LiteRT, no Python at inference time).

## Artifacts

- `k2_convert.py finetune/safetensors/k2-qat4 k2h_ft_q4.blob k2rope1536.npz --q4`
  → 0.769 GB, 255 tensors, header `vocab 69312, wtype 2, npos 1536`
- fp16 reference blob 2.188 GB (same weights, `gemv_ref` path)

## Engine fidelity

`k2_core` (pure asm) vs `fwd_k2q_v2` (C) on the FT blob, 20-id prompt +
8 gen, **full step lines (id_in/top/logit) 28/28 identical**.

## Quantization fidelity

Our Q4_0 (block-32, max/-8) is *not* LiteRT's int4 scheme. On an arbitrary
20-id prompt: 9/28 argmax agreement vs the fp16 reference; native vs the
LiteRT int4 greedy stream 37/400 tokens. Both are valid executions of the
same fine-tune under different quantization.

## Behavioral gate (the one that matters)

Native Q4 greedy on the full 680-id meeting window
(`eval/ft_native_q4_greedy.txt`) — 5/5 verifiable NOTEs, genuine
timestamps, `NEXT`, **no opening wobble** (the LiteRT int4 run had one):

```
NOTE [1:02:15] (NUMBER) 資訊系統預算總額 1200 萬元
NOTE [1:03:02] (PROPOSAL) 建議改用線上報名，減少現場排隊約 80 萬左右
NOTE [1:04:10] (DECISION) 預算案照案通過
NOTE [1:05:33] (ACTION) 報告負責人下週五前交書面報告
NOTE [1:06:20] (OPEN-ISSUE) 場地費 5 萬元尚未付，需儘速處理
NEXT
```

Base model, same window (base tokenizer, 868 ids), greedy 150
(`eval/base_q4_window_greedy.txt`): English meta-deliberation, quotes the
system-prompt rules, **0 NOTEs**.
