# FunctionGemma‑270M‑it: exact prompt/token spec for a from‑scratch engine

All statements below are taken from the three declared Google AI docs (fetched this session) plus the PULSAR‑ASM source. Where a detail is **not** documented, I say so explicitly rather than guessing.

## 0. Verdict up front

`google/functiongemma-270m-it` **works zero‑shot** for single‑turn function calling. Both HF pages print the expected output from the unmodified base checkpoint, and the full‑sequence page drives the *entire* developer→model→tool→model loop with the base model. No fine‑tuning is required for the documented case. Your card already matches the documented structure almost exactly (see §6), so a "different developer wording / escaping / needs fine‑tuning" explanation is **not** supported by the docs — the remaining suspects are whitespace‑level serialization fidelity, stop‑condition handling, and numerics.

---

## 1. Exact developer‑role content string

```
You are a model that can do function calling with the following functions
```

Byte‑exact properties: no trailing period, no trailing newline, no colon, no tool list appended. It is the entire `content` of the `developer` message, verbatim in all three docs:

* **Formatting and best practices → "Turn 1: Tool Definition (Developer)"**: "It is important to include the system prompt `You are a model that can do function calling with the following functions` to enable the model to call tools. This phrase acts as a prompt‑based trigger to switch between tooling capability and general conversation."
* **Function calling with HF → "Passing tools"**: the same string is the `{"role": "developer", "content": ...}` value, annotated in‑code as `# ESSENTIAL SYSTEM PROMPT: / # This line activates the model's function calling logic.`
* **Full function calling sequence → "Model's Turn"**: identical string, identical annotation.

The HF page adds an explicit warning: "To ensure FunctionGemma correctly interprets the available tools and generates a structured call instead of plain text, the **developer** message is essential." So: **plain‑text rambling instead of a call is the documented failure mode of omitting/mangling the developer turn** — that is the single highest‑probability fix for your symptom.

Note the doc's own example (`get_current_temperature`, user asks "What's the temperature in London?") uses a developer turn; a second HF example shows the developer message omitted, but that snippet still passes `tools=` and still prints the correct call — treat the developer message as required anyway, since two of three pages call it ESSENTIAL and the formatting page says it enables tool use.

## 2. Exact serialized text between `<start_function_declaration>` and `<end_function_declaration>`

The literal block from **Formatting and best practices → Turn 1** (markdown backslash escapes removed; `[`/`]` are literal, the doc's `\[` is markdown link escaping):

```
<start_function_declaration>declaration:get_current_weather{description:<escape>Gets the current weather in a given location.<escape>,parameters:{properties:{location:{description:<escape>The city and state, e.g. "San Francisco, CA" or "Tokyo, JP"<escape>,type:<escape>STRING<escape>},unit:{description:<escape>The unit to return the temperature in.<escape>,enum:[<escape>celsius<escape>,<escape>fahrenheit<escape>],type:<escape>STRING<escape>}},required:[<escape>location<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration>
```

Serialization rules, all evidenced by that block plus the second, independently printed rendering in **Full function calling sequence** (`.../{.../...}`):

1. Prefix `declaration:` then the bare function name (`declaration:get_current_weather`), then a `{…}` body.
2. **Every string value is wrapped in `<escape>` … `<escape>`** — "All string literals in your function declarations, calls, and responses must be enclosed, like: `key:<escape>string value<escape>`" (Formatting page, "Delimiter for String Values"). Rationale given: it makes `{`, `}`, `,` and quotes inside a string literal text rather than structure.
3. This includes **type names, uppercased**: input JSON `"string"` → output `type:<escape>STRING<escape>`; `"object"` → `type:<escape>OBJECT<escape>`.
4. It also includes **`required` entries** (`required:[<escape>location<escape>]`) and **`enum` values** (`enum:[<escape>celsius<escape>,<escape>fahrenheit<escape>]`).
5. **No JSON string escaping inside `<escape>`**: double quotes stay literal (`e.g. "San Francisco, CA"`), i.e. no `\"`.
6. **Numeric/boolean values are not escaped** — see the response block `response:get_current_weather{temperature:15,weather:<escape>sunny<escape>}` (Full sequence page).
7. **Key order is alphabetical at every level**: `description` before `parameters`; `properties`, `required`, `type`; `description`, `enum`, `type` per property.
8. **Whitespace inside the block is the single highest‑risk detail.** The two docs disagree: the Formatting page's snippet is space‑free (`…<escape>},unit:{…`), while the Full sequence page's `apply_chat_template` printout (built from a raw Python function) shows `, ` separators and even `type:<escape>STRING<escape>} }` (stray space before `}`). The docs therefore render the block *illustratively*; the **authoritative bytes are whatever `processor.apply_chat_template(msgs, tools=[schema], add_generation_prompt=True)` emits**. Do not hand‑copy the doc's spacing — copy from the processor (see §6, step 1).
9. **Multi‑tool turns are not documented.** No doc shows two functions in one developer turn, so whether they are concatenated into a single `<start_function_declaration>…<end_function_declaration>` or emitted as repeated token pairs is unresolved by these sources. Derive it by running `apply_chat_template` with a two‑tool list.
10. **Prefer manual JSON schemas over auto‑generated ones** for anything nested: "the automatic converter may describe it simply as a generic 'object' without detailing its internal properties" (HF page, "Important Caveat").

## 3. Turn delimiters, special token names, and the exact emitted forms

FunctionGemma "builds on the Gemma prompt structure, using `<start_of_turn>role` and `<end_of_turn>` to describe conversational turns. The `role` is typically `user` or `model` (and sometimes `developer` …)" (Formatting page, "Base Prompt Structure"). Roles are `developer`, `user`, `model`; there is **no `system` role** in any example.

**Control tokens (Formatting page, "Control Tokens" — six special tokens, three pairs):**

| Pair | Purpose |
|---|---|
| `<start_function_declaration>` / `<end_function_declaration>` | defines a tool |
| `<start_function_call>` / `<end_function_call>` | model's request to use a tool |
| `<start_function_response>` / `<end_function_response>` | returns a tool's result |

Plus the single token **`<escape>`** as the string delimiter. The page states verbatim: "**`<start_function_response>` is an additional stop sequence for the inference engine.**"

**The full serialized prompt for the documented London example** (reconstructed by substituting the HF page's `get_current_temperature` schema into the structure the Full sequence page prints byte‑for‑byte):

```
<bos><start_of_turn>developer
You are a model that can do function calling with the following functions<start_function_declaration>declaration:get_current_temperature{description:<escape>Gets the current temperature for a given location.<escape>,parameters:{properties:{location:{description:<escape>The city name, e.g. San Francisco<escape>,type:<escape>STRING<escape>}},required:[<escape>location<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>
<start_of_turn>user
What's the temperature in London?<end_of_turn>
<start_of_turn>model
```

Delimiters: `<start_of_turn>role` + `\n`, content, `<end_of_turn>` + `\n`; `<bos>` first; a final trailing `\n` after `<start_of_turn>model` is the generation prompt (`add_generation_prompt=True`). This matches the doc's own printout (`<bos><start_of_turn>developer\n…<end_of_turn>\n<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`) and matches the card your repo already ships (`pulsar_arm/asm/core.S` file mode reads a raw prefill; `tools/fc_write.py`/`fc_render2.py` already contain this exact schema + message pair).

**Expected model output (the string you're trying to reproduce), HF page verbatim:**

```
<start_function_call>call:get_current_temperature{location:<escape>London<escape>}<end_function_call>
```

Shape rules for the call block: `call:` + name (underscores, no escaping), `{`, `key:<escape>value<escape>` comma‑separated, `}`. The doc's parsing regex (Full sequence page) confirms the grammar: `<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>` with args matched as `(\w+):(?:<escape>(.*?)<escape>|([^,}]*))` — i.e. **un‑escaped args are also legal** (numbers like `temperature:15`), and the same `cast()` logic expects bare `true`/`false`.

**Tool‑result turn (Turn 4 / `role: "tool"`):**

```
<start_function_response>response:get_current_weather{temperature:15,weather:<escape>sunny<escape>}<end_function_response>
```

Built from `{"role": "tool", "content": {"name": …, "response": …}}`, or a list of such dicts for parallel calls — the HF page notes "For optimal results, append the tool execution result to your message history using the specific format below. This ensures the chat template correctly generates the required token structure".

**Multi‑turn nuance (Full sequence page, final printout).** The printed history shows the call, the response and the final sentence sharing **one** `model` turn with **no `<end_of_turn>`** between them:

```
<start_of_turn>model
<start_function_call>call:get_current_weather{location:<escape>Tokyo, Japan<escape>}<end_function_call><start_function_response>response:get_current_weather{temperature:15,weather:<escape>sunny<escape>}<end_function_response>The current weather in Tokyo is sunny with a temperature of 15 degrees Celsius.<end_of_turn>
```

So the generation prompt for the *final* answer is that prefix ending in `…<end_function_response>`, and the turn is closed only by the final `<end_of_turn>`. Treat this printout as observed behavior, and confirm by printing `apply_chat_template` for the 5‑message history rather than assuming a separate turn.

## 4. Generation parameters

The docs specify **only** these two, in both `model.generate` calls (HF page and Full sequence page):

```python
out = model.generate(**inputs.to(model.device), pad_token_id=processor.eos_token_id, max_new_tokens=128)
```

* `max_new_tokens = 128` — the only documented budget; your engine's `gencap` default is already 128 (`core.S:1328`, usage string at `core.S:1267`).
* `pad_token_id = processor.eos_token_id`.
* **No `temperature`, `top_p`, `top_k`, `do_sample`, or `repetition_penalty` appear anywhere in the three docs.** The documented runs therefore use the HF default, i.e. **greedy / argmax decoding** (`do_sample=False`). Your engine's defaults are `temp=1.0`, `top_p=0.95` (mode 1) with mode 0 = greedy (`core.S:1316`, `sampler` at `core.S:4053`) — to reproduce the documented outputs byte‑for‑byte, run **mode 0 (greedy)**, and do not read `top_k` from the docs: it is not documented, so pick your own (irrelevant under greedy).

**Stop conditions you must implement** (three sources: the explicit NOTE on `<start_function_response>`, the emitted `<end_of_turn>`, and `pad/eos`):

1. `<eos>`
2. `<end_of_turn>` (plain‑text replies)
3. **`<start_function_response>`** — "an additional stop sequence for the inference engine". After a function call, stopping at `<start_function_response>` (or at `<end_function_call>`) prevents the model from hallucinating the tool's own result and talking to itself.

Your `file_run` loop currently terminates only on `w23 == #1` (`core.S`, `file_gloop`: `cmp w23, #1 / b.eq file_end // eos`), so a correct call would be followed by junk up to `gencap` — worth fixing regardless, since "rambling text" can also be post‑call spillover rather than a missing call.

## 5. Does the base checkpoint work zero‑shot?

**Yes, for single‑turn and parallel function calling — no fine‑tuning required.**

* Trained: "The model has been explicitly trained on **Single Turn** and **Parallel** function calling."
* Not trained: "The model has **not** been explicitly trained on **Multi‑Turn** or **Multi‑Step** workflows … we expect the model to be able to generalize a bit to these scenarios, especially if fine‑tuned on specific use cases, but it has not been trained to perform these tasks out of the box."
* Fine‑tuning is *recommended* (not required) for two situations only: (a) indirect/abstract user language — "we recommend fine‑tuning the model on a dataset that explicitly maps these semantic nuances to the correct tool definitions"; (b) multi‑turn/multi‑step generalization.
* Semantic mitigation that costs nothing: enrich the tool description. The doc's worked example shows that with the description "Get the current temperature at a location." the model may *fail* to fire on "is it cold in Paris?", while adding "This function can be used to determine if the weather is hot or cold in a given location." yields `<start_function_call>call:get_current_temperature{location:<escape>Paris<escape>,unit:<escape>celsius<escape>}<end_function_call>`. **A peaked 0.81 first token on prose is precisely the documented "Scenario A: Limited Description" symptom** — so also try enriching `get_current_temperature`'s description with hot/cold/weather keywords before concluding anything is broken.

## 6. Concrete deltas for PULSAR‑ASM (from repo facts, not speculation)

1. **Re‑derive the ground truth, don't hand‑copy the doc.** `pulsar_arm/tools/fc_render2.py` already calls `AutoProcessor.from_pretrained("google/functiongemma-270m-it")` + `apply_chat_template(…, tools=[schema], add_generation_prompt=True)` on exactly the doc's schema and messages; `fc_write.py` writes it to a file. Run `fc_render2.py` and copy its `repr()` output verbatim into your card — the doc snippets in §2 differ in whitespace between the two pages.
2. **Your control‑token table is incomplete.** `ct_tab` (`core.S:5005–5027`) maps only 7 tokens: `<bos>`=2, `<eos>`=1, `<end_of_turn>`=106, `<start_of_turn>`=105, `<escape>`=52, `<end_function_declaration>`=47, `<start_function_declaration>`=46. There is **no entry for `<start_function_call>`, `<end_function_call>`, `<start_function_response>`, `<end_function_response>`** anywhere in `core.S` (grep found none). Add all four (plus `<end_of_turn>` handling) to the table so the prefill tokenizer matches special tokens atomically and so the detokenizer prints them as single tokens. Get their numeric ids from `tokenizer.json` → `added_tokens` (as `fc_ctrl.py`/`fc_ids.py` already do for the rest) — **do not infer them**; they are unused vocab slots in a 262k vocab and guessing will silently produce garbage.
3. **Add the stop set** to the decode loop: `<eos>`, `<end_of_turn>`, and `<start_function_response>` (doc‑mandated), plus optionally `<end_function_call>`.
4. **Decode greedy** (mode 0) when comparing against documented outputs.
5. **If prefill ids still match HF exactly and the first token is prose**, the residual suspects are numerics, not formatting: Gemma‑3 270M's alternating sliding‑window attention (`layer_types` is read in `tools/fc_cfg.py`), attention mask construction, RoPE scaling/theta, and any Gemma‑3 softcapping/normalization detail. The docs cannot speak to these; a good discriminator is to compare first‑token logits for a *no‑tools* chat prompt and a *tools* prompt — a correct engine should show the probability mass moving onto `<start_function_call>` when the developer turn + declaration are present.

## 7. Limits of this answer

* The docs never print the serialized prompt for the `get_current_temperature`/`London` example, only the model output; §3's prompt is a faithful reconstruction of the structure, and step 1 above replaces it with processor output.
* Multi‑tool declarations, the exact `role: "tool"`/assistant rendering when re‑templating mid‑conversation, and all four function token ids are **not** in the declared sources; all must be read off `apply_chat_template` / `tokenizer.json`.
* No numeric sampling hyperparameters (temperature/top_p/top_k) are documented anywhere in these three pages; the documented behavior is HF‑default greedy at `max_new_tokens=128`.