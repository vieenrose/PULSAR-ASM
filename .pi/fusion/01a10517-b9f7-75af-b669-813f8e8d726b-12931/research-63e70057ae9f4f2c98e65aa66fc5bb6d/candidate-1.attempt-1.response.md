# FunctionGemma‑270M exact prompt/token spec (from the three declared docs)

Bottom line: the developer-role trigger sentence, the six control tokens, `<escape>`-delimited strings, and the turn delimiters are all explicitly documented and fully determine the format. **No sampling parameters other than `max_new_tokens=128` are documented anywhere** — the docs never mention temperature, top‑p or top‑k, and they never suggest fine‑tuning is required for the documented single‑turn cases. Your hand‑rolled whitespace/serialization is the highest‑risk remaining difference; the docs' authoritative byte string is the *decoded prompt* in the full‑sequence page, and it contains two ` }` spaces that the hand‑written example on the formatting page omits.

---

## 1. Exact developer-role content string

```
You are a model that can do function calling with the following functions
```

* Exact string, **no trailing period, no trailing whitespace/newline**, immediately followed by `<start_function_declaration>`.
* Formatting & best practices, "Turn 1: Tool Definition (Developer)": *"It is important to include the system prompt `You are a model that can do function calling with the following functions` to enable the model to call tools. This phrase acts as a prompt-based trigger to switch between tooling capability and general conversation."*
* Both HF pages repeat it twice as an in‑code comment: `# ESSENTIAL SYSTEM PROMPT: # This line activates the model's function calling logic.`, plus: *"To ensure FunctionGemma correctly interprets the available tools and generates a structured call instead of plain text, the **developer** message is essential."*
* Your repo already uses exactly this string (`pulsar_arm/tools/fc_truth.py:30`, `fc_ids.py:22`, `fc_render2.py:22`, `fc_write.py:22`) — so that hypothesis is confirmed correct, not a bug.
* Doc inconsistency worth knowing: the *second* HF example ("raw Python function", same page) shows a `message` list **without** the developer turn yet reports the same London output. Treat the Note as authoritative — the trigger sentence is required.

## 2. Exact serialized declaration (escaping rules)

Authoritative byte string — this is the *decoded* `apply_chat_template` output printed in **Full function calling sequence with FunctionGemma** (the only place the docs show a real processor render):

```
declaration:get_current_weather{description:<escape>Gets the current weather in a given location.<escape>,parameters:{properties:{location:{description:<escape>The city and state, e.g. "San Francisco, CA" or "Tokyo, JP"<escape>,type:<escape>STRING<escape>},unit:{description:<escape>The unit to return the temperature in.<escape>,enum:[<escape>celsius<escape>,<escape>fahrenheit<escape>],type:<escape>STRING<escape>} },required:[<escape>location<escape>],type:<escape>OBJECT<escape>} }
```

Rules derivable from that string plus the escaping section:

| Element | Rule |
|---|---|
| Body prefix | `declaration:` then function name (never escaped) |
| Schema keys | `name`, `description`, `parameters`, `properties`, `type`, `enum`, `required` — **not** escaped |
| All **string values** | wrapped `key:<escape>value<escape>` — *"All string literals in your function declarations, calls, and responses must be enclosed, like: `key:<escape>string value<escape>`"* |
| JSON type names | **upper‑cased and escaped**: `type:<escape>STRING<escape>`, `type:<escape>OBJECT<escape>` (JSON `"string"`/`"object"` → `STRING`/`OBJECT`) |
| Arrays | `enum:[<escape>celsius<escape>,<escape>fahrenheit<escape>]`, `required:[<escape>location<escape>]` — brackets bare, items escaped |
| Quotes inside a description | **kept verbatim and unescaped**: `e.g. "San Francisco, CA" or "Tokyo, JP"` (no `\"`, no stripping) |
| Numbers/booleans in responses | **not** escaped: `temperature:15` |
| Commas/colons | no space after them: `,parameters:{`, `,type:<escape>STRING<escape>` |
| Whitespace | two ` }` (space **before** a closing brace) appear after the `unit` object and after the `parameters` object. See warning below. |

**Whitespace warning (most likely root cause of your mismatch):** the *hand‑written* Turn‑1 example on the Formatting page shows the same schema **without** those spaces:

```
…type:<escape>STRING<escape>}},required:[<escape>location<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration>
```

while the actual decoded render has `…STRING<escape>} },required:…type:<escape>OBJECT<escape>} }`. The two differ only in that whitespace, and the rule that decides it is not documented (it alternates oddly with nesting depth — spaces appear before the closing brace at odd depths but not even depths). Therefore: **do not hand-serialize the schema.** Generate the prompt once with `processor.apply_chat_template(msgs, tools=[...], add_generation_prompt=True, tokenize=False)` and bake that exact byte string (or its token ID list) into the card, as your `tools/fc_write.py` already does. Also note that the docs never render a multi‑tool (`tools=[a, b]`) developer turn — do not guess whether declarations are concatenated or separated by a comma/newline; derive it from one HF render.

## 3. Turn delimiters and special token names

**Control tokens (six, per the docs' table):**
`<start_function_declaration>` / `<end_function_declaration>`, `<start_function_call>` / `<end_function_call>`, `<start_function_response>` / `<end_function_response>` — plus the separate string delimiter `<escape>`. Role framing is inherited from the Gemma prompt structure: `<start_of_turn>role` … `<end_of_turn>`, roles `user`, `model`, and `developer`.

**Complete verbatim prompt (Tokyo example, from the full-sequence page's decoded print):**

```
<bos><start_of_turn>developer
You are a model that can do function calling with the following functions<start_function_declaration>declaration:get_current_weather{…}<end_function_declaration><end_of_turn>
<start_of_turn>user
Hey, what's the weather in Tokyo right now?<end_of_turn>
<start_of_turn>model
```

**Model turn continuation (call → tool result → answer), same source:**

```
<start_function_call>call:get_current_weather{location:<escape>Tokyo, Japan<escape>}<end_function_call><start_function_response>response:get_current_weather{temperature:15,weather:<escape>sunny<escape>}<end_function_response>The current weather in Tokyo is sunny with a temperature of 15 degrees Celsius.<end_of_turn>
```

Structural facts a from-scratch engine must honour:

* Exactly one `<bos>`, **no newline after it** (`<bos><start_of_turn>`), and no second `<bos>` (`fc_ids.py` checks this).
* Turn shape: `<start_of_turn>` + role + `\n` + content + `<end_of_turn>` + `\n`. So there *is* a newline after each `<end_of_turn>` and after the role name, but **none** between the developer sentence and `<start_function_declaration>`.
* `add_generation_prompt=True` appends exactly `<start_of_turn>model\n` — no `<end_of_turn>` pre-filled.
* The call block is **not** closed by `<end_of_turn>`; `<start_function_response>` follows immediately inside the same model turn, and the final `<end_of_turn>` terminates the whole turn.
* Call payload grammar: `call:<name>{<key>:<value>,…}`; response payload: `response:<name>{…}`.
* Stop handling: *"` <start_function_response>` is an additional stop sequence for the inference engine."* (plus obviously `<end_of_turn>`/`<eos>`).
* **Detokenizer note (inference from the docs):** the doc prints the call with `processor.decode(..., skip_special_tokens=True)` and the control tokens still appear in the output, so in this tokenizer the six function tokens and `<escape>` are added tokens that are *not* flagged special. Emit them as literal text; only `<bos>`, `<eos>`, `<pad>`, `<start_of_turn>`, `<end_of_turn>` are stripped.
* Tool results are appended as `{"role": "tool", "content": {"name": …, "response": …}}` (list of these for parallel calls) so the template emits the `response:` block; the docs' Note says *"For optimal results, append the tool execution result to your message history using the specific format below."*
* The docs do **not** publish numeric token IDs. Your `core.S` `ct_tab` already encodes: `<bos>`=2, `<eos>`=1, `<start_of_turn>`=105, `<end_of_turn>`=106, `<escape>`=52, `<end_function_declaration>`=47, `<start_function_declaration>`=46. **Gap:** `<start_function_call>`, `<end_function_call>`, `<start_function_response>`, `<end_function_response>` are absent from that 7‑entry table — add them before feeding any post‑tool turn through the exact‑match encoder.

## 4. Generation parameters

**What the docs actually prescribe** (identical in all three code samples, no other knobs appear anywhere):

```python
out = model.generate(**inputs.to(model.device),
                     pad_token_id=processor.eos_token_id,
                     max_new_tokens=128)
generated = out[0][len(inputs["input_ids"][0]):]
output = processor.decode(generated, skip_special_tokens=True)
```

* `max_new_tokens = 128` in every example.
* `pad_token_id = processor.eos_token_id`.
* No `stop_strings`, no `eos_token_id` override, no temperature/top‑p/top‑k, no `do_sample`.
* **temperature / top_p / top_k: not specified by any of the three docs.** HF `generate()`'s default is non‑sampling (greedy) decoding, so the documented outputs are greedy outputs. To reproduce them, decode greedily. Your engine logs `temp: 1.0000000 topp: 0.9500000` (`core.S` / task logs) — that is *not* what the reference run did; switch to argmax for reproduction, and don't expect a doc‑sanctioned temperature to rescue a wrong prompt.
* Multi‑token budget for the full sequence: 128 new tokens per `generate()` call in both the first turn and the post‑tool turn.

## 5. Zero-shot vs fine-tuning

* **Base `google/functiongemma-270m-it` works zero-shot, unmodified, for the documented single‑turn cases.** The HF page loads it with plain `AutoProcessor`/`AutoModelForCausalLM` (no fine-tuning step anywhere in setup) and reports `Output: <start_function_call>call:get_current_temperature{location:<escape>London<escape>}<end_function_call>`; the full-sequence page does the same for `get_current_weather` **and** for the post‑tool final answer. Nothing in any of the three docs says fine-tuning is required to emit a call.
* Fine‑tuning is recommended only for **semantic‑nuance / indirect‑language** gaps: *"For production environments where users frequently use highly indirect language (e.g., 'is it nice out?' versus 'get weather'), we recommend fine-tuning the model on a dataset that explicitly maps these semantic nuances to the correct tool definitions."* Mitigation order given: enrich the tool description first (e.g. add *"This function can be used to determine if the weather is hot or cold in a given location"* — that change alone flips the model to `<start_function_call>call:get_current_temperature{location:<escape>Paris<escape>,unit:<escape>celsius<escape>}<end_function_call>`), then prompt engineering, then fine‑tuning. (The *Fine‑tune FunctionGemma* page is linked but is not one of the declared sources, so I have not used it.)
* **Trained scope:** explicitly trained on **single‑turn** and **parallel** calling. *"The model has **not** been explicitly trained on Multi‑Turn or Multi‑Step workflows… We expect the model to be able to generalize a bit to these scenarios, especially if fine‑tuned on specific use cases, but it has not been trained to perform these tasks out of the box."*
* Also stated: the function‑calling‑only specialisation means this checkpoint should not be expected to do good general chat.

## 6. What this implies for your engine

1. **Trigger sentence** — already correct in your tools; keep it verbatim and make sure no space/newline separates it from `<start_function_declaration>`.
2. **Serialize with the processor, not by hand.** The docs themselves are internally inconsistent on whitespace (`} }` vs `}}`); only the decoded full‑sequence render is real. Keep using `tools/fc_write.py` to emit `/tmp/tool_official.txt` and feed that to file mode.
3. **File mode must carry `<bos>` itself** — your `core.S` comment already says "no bos/template; prompt carries its own controls", and the docs' render starts with `<bos>` and nothing after it. Also confirm the encoder table has all 10 relevant special tokens (missing 4, see §3).
4. **Decode greedily** to reproduce the documented string; treat `temp 1.0 / top‑p 0.95` as a deliberate experiment, not the reference configuration.
5. **From the repo's task logs** (`.pi/tasks/…/b72fdd834.output` vs `bf94d63f2.output`): the HF‑written 565‑char prompt produced `tpl: 98` — matching AutoProcessor's `NIDS: 98` — and still rambled ("I am ready. I will be able to a list any time…") with `lg nan inf: 0 0`. Per these docs, once the prompt is byte‑identical and decoding is greedy there is no further documented knob; that points the remaining suspicion at engine internals (embedding scale/softcapping, RoPE, attention mask, pre‑vs‑post‑norm, logit processor) rather than at prompt format or sampling. (The separate `get_time` minimal‑prompt run tokenized to 72 — unrelated to this example.)
6. **Stop tokens:** treat `<start_function_response>`, `<end_of_turn>`, `<eos>` as terminators; the docs call out the first explicitly.

## Limitations / not covered by the declared sources

* No numeric special‑token IDs published (only the repo's verified table).
* No multi‑tool developer turn, no `parameters:{}` empty case, and no nested/complex‑object declaration are rendered in these three docs (the HF page discusses manual vs auto schema only conceptually, and its automatic‑vs‑manual example is plain `json.dumps` output, *not* the template's escaped form).
* No temperature/top‑p/top‑k guidance exists in these docs; my greedy recommendation is inferred from HF `generate()` defaults plus the absence of any sampling argument.
* The `<start_function_response>` "additional stop sequence" is stated as a NOTE without a code sample; I have not fetched the Fine‑tuning page (not a declared source).