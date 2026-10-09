Decision: the FT demo stays zh-TW-only — these weights were trained
specifically on a Traditional Chinese meeting corpus, and the English
probe below degrades off-corpus (3/5 with converted millions; 5/5 only
with 萬 figures kept, content code-mixed). This file keeps the negative
result and both fixtures for the record; no English FT clip is shipped.

# FT meeting agent: English window (greedy, native Q4 engine)

Same system prompt + few-shot examples as the zh run; the 6-turn window
translated to English (`prompt_ft_enB.txt`, 698 ids vs 680 zh). Validated on
`k2h_ft_q4.blob` (Spark t8b8, phone t4b8 bit-identical, 1098 steps):

NOTE [1:02:15] (NUMBER) / [1:03:02] (PROPOSAL) / [1:04:10] (DECISION) /
[1:05:33] (ACTION) / [1:06:20] (OPEN-ISSUE) — 5/5, genuine timestamps,
clean NEXT stop. Content mixes English with Chinese fragments (zh-tuned
model); the OPEN-ISSUE trails into Chinese.

Variant A (same window, millions converted: "12 million", "800 thousand")
drops to 3/5 (DECISION/ACTION/OPEN-ISSUE only) — the model keys on the
source's 萬 figures, so the clip keeps them. Both fixtures and transcripts
in eval/ (`prompt_ft_en*.txt`, `ft_ids_en*.txt`, `ft_en*_asm.txt`).
