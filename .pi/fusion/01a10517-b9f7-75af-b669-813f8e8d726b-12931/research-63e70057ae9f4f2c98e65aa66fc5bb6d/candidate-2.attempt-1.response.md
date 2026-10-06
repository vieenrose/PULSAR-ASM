# FunctionGemma-270m: exact prompt/token format spec (from the three declared docs)

Sources used (all three declared URLs fetched, HTTP 200):
- **[A]** *FunctionGemma formatting and best practices* — `https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices` (last updated 2025-12-18)
- **[B]** *Function calling with Hugging Face Transformers* — `https://ai.google.dev/gemma/docs/functiongemma/function-calling-with-hf` (last updated 2026-04-22)
- **[C]** *Full function calling sequence with FunctionGemma* — `https://ai.google.dev/gemma/docs/functiongemma/full-function-calling-sequence-with-functiongemma` (last updated 2026-04-22)

---

## 0. Bottom line for your bug

The official format is already what your repo renders. `pulsar_arm/tools/fc_render2.py` uses the literal official developer string (`{"role": "developer", "content": "You are a model that can do function calling with the following functions"}`), and you report prefill is bit-exact vs `AutoProcessor.apply_chat_template` (98/98) with 0 NaN/inf. **[B] and [C] both demonstrate the *unmodified* `google/functiongemma-270m-it` checkpoint emitting `<start_function_call>call:get_current_temperature{location:<escape>London<escape>}<end_function_call>` zero-shot, with the exact prompt format below.** So none of the four suspect causes in your brief (wording, escaping, generation settings, "needs fine-tuning") explains a failure — see §6 for what to check instead.

---

## 1. Exact developer-role content string

```
You are a model that can do function calling with the following functions
```

- Byte-exact: no trailing period, no colon, no newline, no trailing whitespace. The role name is `developer` (not `system`).
- [A], "Turn 1: Tool Definition (Developer)": *"It is important to include the system prompt `You are a model that can do function calling with the following functions` to enable the model to call tools. This phrase acts as a prompt-based trigger to switch between tooling capability and general conversation."*
- [B] and [C] both label it in-code: `# ESSENTIAL SYSTEM PROMPT: / # This line activates the model's function calling logic.`, and: *"To ensure FunctionGemma correctly interprets the available tools and generates a structured call instead of plain text, the **developer** message is essential. This specific system prompt instructs the model that it has permission and capability to perform function calling."*
- [C] also shows that with `tools` passed to `apply_chat_template`, this text is the *entire* developer content; the schema is **not** appended to it. It is emitted as the developer turn body, immediately followed (no separator) by `<start_function_declaration>`.

⚠️ Repo cross-check: `pulsar_arm/tools/fc_render.py` uses a different invented string (`"You are a model that can do function calling..."` variant: `"You can call functions. Available: " + str([schema])`). That is not the official trigger; `fc_render2.py` is the correct one.

## 2. Exact serialized text between `<start_function_declaration>` and `<end_function_declaration>`

Verbatim from [C]'s final `processor.decode(out[0], skip_special_tokens=False)` full-history dump (single line, wrapped here for readability):

```
declaration:get_current_weather{description:<escape>Gets the current weather in a given location.<escape>,parameters:{properties:{location:{description:<escape>The city and state, e.g. "San Francisco, CA" or "Tokyo, JP"<escape>,type:<escape>STRING<escape>},unit:{description:<escape>The unit to return the temperature in.<escape>,enum:[<escape>celsius<escape>,<escape>fahrenheit<escape>],type:<escape>STRING<escape>} },required:[<escape>location<escape>],type:<escape>OBJECT<escape>} }
```

From the Python source that produced it [C]:
```python
def get_current_weather(location: str, unit: str = "celsius"):
    """Gets the current weather in a given location.

    Args:
        location: The city and state, e.g. "San Francisco, CA" or "Tokyo, JP"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])

    Returns:
        temperature: ...
        weather: ...
    """
```

### Escaping / serialization rules (all traceable to [A] + [B] + [C])

1. **Structure**: `declaration:<function_name>{description:<escape>…<escape>,parameters:{…}}`. Keys and values separated by `:`, entries by `,`, braces/brackets **not** quoted. The outer `{"type": "function", "function": {...}}` wrapper is dropped; the `name` value becomes the token after `declaration:`.
2. **`<escape>` wraps EVERY string literal** — [A]: *"A single token, `<escape>`, is used as a delimiter for **all string values** within the structured data blocks… All string literals in your function declarations, calls, and responses must be enclosed, like: `key:<escape>string value<escape>`."*
   - That includes the **type names**: `type:<escape>STRING<escape>`, `type:<escape>OBJECT<escape>` — note **UPPERCASE**, whereas your Python schema uses lowercase `"string"`/`"object"`. **This is the single most common transcription bug.**
   - That includes **`required` array items**: `required:[<escape>location<escape>]`.
   - That includes **`enum` items**: `enum:[<escape>celsius<escape>,<escape>fahrenheit<escape>]`.
   - Non-string values stay bare: [C] response example `response:get_current_weather{temperature:15,weather:<escape>sunny<escape>}` — `15` has no `<escape>`.
   - Escaping exists so `{`, `}`, `,`, quotes inside a string stay literal ([A]); the literal `"` in `e.g. "San Francisco, CA"` is **kept verbatim inside** the `<escape>` pair — no `\"` escaping, no removal.
3. **Key ordering is not your source-JSON order** — it is template-normalized (alphabetical / `sort_keys`-like), observed consistently in every example:
   - function level: `description`, then `parameters`
   - `parameters`: `properties`, `required`, `type`
   - property level: `description`, `enum`, `type`
   (Input JSON had `type` first and `type`/`properties`/`required` in a different order.)
   *This ordering is inferred from the doc examples, not stated as a rule — treat the shipped `chat_template.jinja` in your local snapshot (`~/.cache/huggingface/hub/models--google--functiongemma-270m-it/snapshots/*/chat_template.jinja`) as ground truth and assert against it.*
4. **Whitespace**: [A]'s example renders `…type:<escape>STRING<escape>}}<end_function_declaration>` (no space before the closing braces) while [C]'s decoded dump has `…<escape>STRING<escape>} }` and `…<escape>OBJECT<escape>} }`. The docs are inconsistent here; do not hand-transcribe. Emit no stray spaces and verify by diffing your rendered string against `apply_chat_template(..., tokenize=False)`.

## 3. Turn delimiters and special token names

Base structure [A]: *"FunctionGemma builds on the Gemma prompt structure, using `<start_of_turn>role` and `<end_of_turn>` to describe conversational turns. The role is typically `user` or `model` (and sometimes `developer`…)."*

Exact delimiters:
- `<bos>` — document start (present in [C]'s decode; emitted by the template, not by you when pre-filling raw text).
- `<start_of_turn>` + role name + **`\n`** (one LF, no space) + content + `<end_of_turn>` + **`\n`** (one LF after the closing token — visible in [C]: `<end_of_turn>\n<start_of_turn>user`).
- Generation prompt = `<start_of_turn>model\n` appended last (`add_generation_prompt=True` in [B]/[C]).

Full literal prompt for the [B] target example (schema `get_current_temperature` / `{"location": {"type":"string","description":"The city name, e.g. San Francisco"}}`, `required:["location"]`; user: `What's the temperature in London?`):

```
<bos><start_of_turn>developer
You are a model that can do function calling with the following functions<start_function_declaration>declaration:get_current_temperature{description:<escape>Gets the current temperature for a given location.<escape>,parameters:{properties:{location:{description:<escape>The city name, e.g. San Francisco<escape>,type:<escape>STRING<escape>}},required:[<escape>location<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>
<start_of_turn>user
What's the temperature in London?<end_of_turn>
<start_of_turn>model
```
(Keys/values/whitespace normalized per §2 rules 3–4; the `<escape>`/`STRING`/`OBJECT`/order details are doc-derived, the exact byte layout must be validated against your local `chat_template.jinja`.)

### Six control tokens [A]
| Pair | Purpose |
|---|---|
| `<start_function_declaration>` / `<end_function_declaration>` | Defines a tool |
| `<start_function_call>` / `<end_function_call>` | Model's request to use a tool |
| `<start_function_response>` / `<end_function_response>` | Tool result back to the model |

Plus the single delimiter token **`<escape>`**.

### Model output [B]/[C]
```
<start_function_call>call:get_current_temperature{location:<escape>London<escape>}<end_function_call>
```
Note: emitted **immediately after** `<start_of_turn>model\n`, with no leading space, and typically **no trailing `<end_of_turn>`** in this single-turn case. Parsing regex given in [C]:
```python
r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>"
r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))"
```
i.e. values are either `<escape>…<escape>`-delimited strings or bare non-`,`/`}`-delimited scalars.

### Tool-result turn — two doc statements, use the message API
- [A] "Turn 4 (Developer)": `<start_function_response>response:get_current_weather{temperature:15,weather:<escape>sunny<escape>}<end_function_response>`
- [C]'s decoded full history places it inside the model turn right after `<end_function_call>`, with no `<start_of_turn>` wrapper:
```
<start_of_turn>model
<start_function_call>call:get_current_weather{location:<escape>Tokyo, Japan<escape>}<end_function_call><start_function_response>response:get_current_weather{temperature:15,weather:<escape>sunny<escape>}<end_function_response>The current weather in Tokyo is sunny with a temperature of 15 degrees Celsius.<end_of_turn>
```
- **Authoritative path (what a from-scratch engine should implement):** don't hand-write these; feed roles and let the template emit them. [C] message history:
```python
{"role": "developer", "content": "You are a model that can do function calling with the following functions"}
{"role": "user", "content": "Hey, what's the weather in Tokyo right now?"}
{"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_current_weather", "arguments": {"location": "Tokyo, Japan"}}}]}
{"role": "tool", "content": {"name": "get_current_weather", "response": {"temperature": 15, "weather": "sunny"}}}
{"role": "assistant", "content": "The current weather in Tokyo is sunny with a temperature of 15 degrees Celsius."}
```
[C] note: *"For optimal results, append the tool execution result to your message history using the specific format below. This ensures the chat template correctly generates the required token structure (e.g. `response:get_current_weather{temperature:15,weather:<escape>sunny<escape>}`)."* For parallel calls, `content` is a **list** of `{"name":…, "response":…}` dicts. Note the role vocabulary is `developer` / `user` / `assistant` / `tool` at the API level even though the rendered token stream shows `model`.

## 4. Generation parameters

**The docs specify only one sampling parameter.** Every `generate()` call in [B] and [C] is:

```python
out = model.generate(**inputs.to(model.device), pad_token_id=processor.eos_token_id, max_new_tokens=128)
```

- `max_new_tokens = 128` (used in both [B] and [C], both turns).
- **No `temperature`, `top_p`, `top_k`, or `do_sample` is set anywhere in the three docs.** With HF defaults this is effectively **greedy / deterministic decoding** (`do_sample=False`), so `temperature/top_p/top_k` are unspecified-by-doc and irrelevant in the greedy case. Recommended: `do_sample=False` (temperature=0), `max_new_tokens=128`, and don't invent top_p/top_k. (Your repo's `runtime/gemma4_model.py:350` defaults `temp=0.0, top_k=64, top_p=0.95`; at `temp=0.0` your sampler is bit-identical to greedy per your own run-21 log entry, so this is not the issue — but note `top_k=64` is not a documented FunctionGemma setting.)
- **Stop sequences:** [A] explicitly: *"`<start_function_response>` is an additional stop sequence for the inference engine."* Combine with the normal `<end_of_turn>` / `<eos>` turn enders. So the engine must be able to (a) stop on `<end_of_turn>`/`<eos>`, and (b) recognize `<start_function_response>` as a stop even though it is *not* a turn terminator.
- Decoding in the examples uses `skip_special_tokens=True` on generated tokens yet still prints `<start_function_call>…` — the special tokens are content to your parser, not to be stripped.

## 5. Does the base checkpoint work zero-shot, or does it need fine-tuning?

**Zero-shot, out of the box.** [B] loads `google/functiongemma-270m-it` via `AutoProcessor`/`AutoModelForCausalLM`, passes `tools=[weather_function_schema]` plus the developer message, and prints exactly `<start_function_call>call:get_current_temperature{location:<escape>London<escape>}<end_function_call>` — the very output you are trying to reproduce, unmodified weights, no fine-tuning step. [C] does the same with a raw Python function and reproduces `call:get_current_weather{location:<escape>Tokyo, Japan<escape>}`, then completes the tool-result turn in plain text.

Documented **limitations** (fine-tuning is only a *mitigation*, never a prerequisite for the basic case):
- Trained scope [A]: **Single Turn** and **Parallel** function calling only.
- **Not** trained: **Multi-Turn** and **Multi-Step/chaining** workflows — *"We expect the model to generalize a bit to these scenarios, especially if fine-tuned on specific use cases, but it has not been trained to perform these tasks out of the box."*
- **Semantic nuance** [A]: abstract/indirect queries may miss the tool (e.g. *"Hey, is it cold in Paris right now?"*). Fixes, in order of stated effectiveness: (1) **enriched tool description** — adding *"This function can be used to determine if the weather is hot or cold in a given location"* makes it emit `call:get_current_temperature{location:<escape>Paris<escape>,unit:<escape>celsius<escape>}`; (2) more explicit user phrasing; (3) **fine-tuning** for production with heavily indirect language.
- Schema-shape caveat [B]: auto-conversion from Python functions loses nested object detail (`Config` → generic `object`); use **manual JSON schemas** for nested parameters. Docstrings should follow the Google Python Style Guide for best results.
- A dedicated *Fine-tune FunctionGemma* guide exists (linked from all three pages) but is outside the declared sources here — not consulted.

## 6. What this implies for your failing run (repo-derived)

Given prefill is bit-exact vs `apply_chat_template` and the docs guarantee zero-shot on this exact format, treat "wrong prompt" as ruled out and hunt engine-side, in this order:

1. **Verify against HF on x86, not on the Pi.** Your run-218 log says the HF forward SIGILLs on the Pi; `pulsar_arm/tools/fc_truth.py` is the right tool — run it off-Pi for the *same* input_ids and print `top-8` at the last position. If HF's argmax is `<start_function_call>` and yours is 0.81 on a non-call token, the divergence is in the forward pass, not the prompt.
2. **First-token boundary off-by-one.** The peaked-wrong result at exactly the first generated token is the classic signature of using the wrong position (last prefill token vs. the position after it), a KV-cache write/read misalignment at the prefill→decode transition, or dropping `<bos>`/the trailing `\n` of `<start_of_turn>model\n`.
3. **Numerics.** `dtype="auto"` on a BF16 checkpoint means HF computes in BF16; your BF16→F32 path must match. Compare layer-by-layer activations (you already have `tools/probe_layer_stages.py`, `probe_batch_stages.py`).
4. **Control-token coverage gap in your encoder.** `pulsar_arm/asm/core.S` (~lines 5000–5030) hardcodes only seven table entries: `<bos>`=2, `<eos>`=1, `<end_of_turn>`=106, `<start_of_turn>`=105, `<escape>`=52, `<end_function_declaration>`=47, `<start_function_declaration>`=46. The four **function-call/response** control tokens (`<start_function_call>`, `<end_function_call>`, `<start_function_response>`, `<end_function_response>`) are **absent** from `ct_tab`. That does not affect prefill of the first turn, but it (a) prevents correct multi-turn re-feeding of model output, and (b) means your engine has no path for the `<start_function_response>` stop sequence required by [A]. Add them, taking their IDs from the FunctionGemma tokenizer's `added_tokens`/`tokenizer.json` in your snapshot — do not infer them from Gemma-3 chat-template conventions.
5. **Card contents to double-check** against §2: uppercase `STRING`/`OBJECT` inside `<escape>`, `<escape>`-wrapped `required` and `enum` items, bare numeric `15` in responses, and the exact LF placement around `<start_of_turn>`/`<end_of_turn>`. Your `fc_render2.py`/`fc_truth.py` path already does this correctly via `apply_chat_template`; the failure appears in *file mode*, so diff the hand-written card byte-for-byte against `apply_chat_template(..., tokenize=False)` for the identical schema.

### Known doc ambiguities (do not treat any of these as authoritative)
- Brace/space placement before `}<end_function_declaration>` differs between [A] and [C].
- Turn ownership of `<start_function_response>` is described as "Developer" in [A]'s prose but appears inside the `model` turn in [C]'s decode.
- No temperature/top_p/top_k guidance exists in any of the three docs; `max_new_tokens=128` + HF default greedy is the entire documented configuration.

**Ground-truth rule of thumb:** the shipped `chat_template.jinja` + `tokenizer.json` in your local `functiongemma-270m-it` snapshot, and `apply_chat_template(tokenize=False)`, outrank every doc example above. Pin your engine's rendered string to it with a byte-level assertion, then debug the forward pass.