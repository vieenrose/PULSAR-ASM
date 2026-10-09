# K2-Horizon-0.9B base: zh-TW ceiling hunt (greedy, Q4 native engine)

The base model is English-leaning: zh prompts usually get English
deliberation that loops or errs. Twelve greedy probes (base tokenizer +
IFM chat template, `<ifm|think>\n` close, `k2h_09_q4.blob`, Spark):

| probe | prompt | verdict |
|---|---|---|
| seasons | 請列出四季… | EN deliberation loops on "the weather is pleasant" |
| gravity | 兩句話解釋地心引力 | echo-loop of 請解釋這句話 |
| health | 三個要點 | EN deliberation loops on asking for clarification |
| quantum | 一句話解釋量子力學 | engages in zh but garbles (重要分子 ×4), answer cut off |
| greet | 自我介紹 | answers as "ChatGPT/OpenAI", then loops |
| capital | 台灣首都在哪 | wrong facts (two capitals, Taoyuan in the south) |
| seasons_tw | 四季＋請用繁體中文回答 | echo-loop of the instruction |
| seasons_ans_zh | EN prompt, answer in zh | circular logic (Spring changes to summer) |
| translate | EN→zh, no explanation | deliberates 200 tok, never answers |
| complete_qm | 接續完成…量子力學是… | trivial echo of the prompt fragment |
| oneword | 下一個季節是？一詞 | correct (夏), clean EOS — but one word |
| **seasons_one** | **four season names in zh, no explanation** | **focused deliberation, correct 春夏秋冬, clean EOS at 88 gen** |

Winner (`zh_ceiling_ids.txt`, 26 ids → `zh_ceiling_q4_greedy.txt`, 114
steps, reproducible t8b8 == t1b1): the model lists 春夏秋冬 and closes its
turn. The deliberation's pinyin glosses are off, but the answer tokens are
right. Honest ceiling: single words / short phrases; anything longer
rambles or loops. Cut the GIF transcript before `<|ifm|im_end|>`, as with
NEXT.
