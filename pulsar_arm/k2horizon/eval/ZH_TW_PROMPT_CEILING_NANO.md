# K2-Horizon-0.9B base: zh-TW-written prompt ceiling on the Jetson Nano

The earlier zh ceiling (`ZH_CEILING.md`) used an English prompt asking for Chinese output.
This hunt uses prompts **written in Traditional Chinese**. Greedy, `--gen 300`, Q4 blob, the
committed Nano engine (`k2_gexp`, HEAD of `k2-horizon-jetson-nano`), IFM chat template
encoded with the base tokenizer (the encoder reproduces the known English prompt ids exactly).

| probe | prompt | verdict |
|---|---|---|
| seasons_names | 請用繁體中文列出四季的名稱，只要四個名稱，不要解釋。 | wrong (four people's names) |
| seasons_which | 一年四季是哪四個季節？只列出名稱。 | correct: 春、夏、秋、冬 |
| after_spring | 春天之後是什麼季節？ | correct: 夏季 |
| next_season | 下一個季節是？一詞 | never closes the think block (300 cap) |
| days_week | 一個星期有幾天？ | correct: 一週 = 7 天 |
| one_plus_one | 一加一等於多少？ | correct: 2 |
| translate_seasons | 請把 spring, summer, autumn, winter 翻譯成繁體中文。 | correct: 春、夏、秋、冬 |
| colors | 請列出三種顏色。 | never closes the think block |
| numbers | 請用中文寫出一到五。 | wrong (echoes 一到五) |
| sun | 請用一句話說明太陽是什麼。 | garbled sentence |
| animals | 請列出三種動物。 | answers in English |
| hello | 請用繁體中文說「你好」。 | answers in simplified Chinese |
| seasons_desc | 一年四季是哪四個季節？每個季節各用兩個字描述。 | ignores the description part |
| weekdays | 請列出一星期的七天。 | never closes the think block |
| directions | 請列出四個方位。 | never closes the think block |
| months | 一年有幾個月？ | never closes the think block |
| primary_colors | 請說出光的三原色。 | **correct sentence: 三原色（三主要颜色）是紅、綠、藍** (one simplified 颜) |
| water | 水的化學式是什麼？ | never closes the think block |
| seasons_order | 請依序列出一年四季。 | correct: 春、夏、秋、冬 |

Winner: **請說出光的三原色。** It is the only probe that needs domain knowledge (light, so
red/green/blue rather than the painter's primaries) and answers with a full sentence. 268 steps,
closes its turn cleanly, and reruns byte-identical. Slips kept as generated: the English
deliberation misquotes the prompt as 請出光的三原色, and the answer contains the simplified
颜. Decode rate on the Nano: 87.8 ms/token (11.4 tok/s) over 249 generated steps.
