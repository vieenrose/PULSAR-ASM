# Gemma 4 E2B (QAT Q4_0) on the Jetson Nano: ceiling-prompt hunt

16 probes, 8 in English and 8 written in Traditional Chinese, of rising difficulty. Greedy,
`--gen 500`, the committed engine (`g4eng`) on the Nano, chat template
`<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n`. Ids in `probes/probes_ids.json`.
Verdicts are mine, read from the detokenized output.

| probe | verdict |
|---|---|
| en: List the four seasons and one thing that changes in each. | correct, complete |
| en: Explain why the sky is blue in three sentences. | correct, complete |
| en: A train leaves at 3:15 pm, trip 2 h 50 min. When does it arrive? | **wrong**: adds the carried hour twice, 7:05 pm |
| **en: A shirt costs $25 after a 20% discount. Original price? Show your work.** | **correct ($31.25), with a check, complete (292 tokens)** |
| en: Write a haiku about autumn leaves. | form wrong (8 syllables in line 2) |
| en: Compare TCP and UDP in a short table. | correct, complete |
| en: Python function, n-th Fibonacci with a loop. | correct code, cut at 500 tokens |
| en: Translate into French: 'The weather is nice today, let's go for a walk.' | contains wrong forms ("allons nous nous promener") |
| zh: 請列出四季，以及每一季會改變的一件事。 | fine but philosophical and cut at 500 tokens |
| zh: 請用三句話解釋為什麼天空是藍色的。 | correct (one dubious claim: sunlight "has more blue") |
| **zh: 一列火車下午3點15分出發，車程2小時50分鐘，幾點抵達？** | **correct (下午6點5分), complete (118 tokens)** |
| zh: 一件襯衫打八折後是500元，原價是多少？請寫出計算過程。 | **wrong** first (2500), self-corrects, cut at 500 |
| zh: 請寫一首關於秋天的五言絕句。 | form kept (4 x 5 characters); quality subjective |
| zh: 請用表格比較TCP和UDP的差異。 | mostly right, some factual slips, cut at 500 |
| zh: 請寫一個Python函式，用迴圈計算第n個費氏數。 | correct code, cut at 500 |
| zh: 請把這句話翻譯成英文：「今天天氣很好，我們去散步吧。」 | correct, complete |

Chosen for the demo: the two arithmetic word problems answered correctly and completely,
each with an objectively checkable answer: the discount problem in English, the train
problem in Traditional Chinese. (The same train problem in English is answered wrongly; the
same discount problem in Chinese starts wrong. Both are reported, not hidden.)
Both reruns are byte-identical (`probes/*_steps.txt`). Engine-reported rates: en 291 decode
steps 6.17 tok/s, prefill 33 tokens 6.61 tok/s; zh 117 steps 6.56 tok/s, prefill 35 tokens 6.47 tok/s.
