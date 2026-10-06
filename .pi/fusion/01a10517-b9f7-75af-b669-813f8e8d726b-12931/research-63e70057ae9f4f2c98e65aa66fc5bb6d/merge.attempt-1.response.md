# FunctionGemma-270M prompt/token specification

## Conclusion

For the reported failing run, the prompt is already the prime suspect ruled out: a **98/98 token-ID match with `AutoProcessor.apply_chat_template` means the developer wording, declaration escaping, declaration whitespace, BOS placement, and turn framing cannot explain the wrong first token**.

The installed model snapshot’s processor, tokenizer, and chat template are the byte-level authority. Do not normalize or reconstruct them from documentation examples.

Sources:

- **[F]** [Formatting and best practices](https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices)
- **[HF]** [Function calling with Hugging Face Transformers](https://ai.google.dev/gemma/docs/functiongemma/function-calling-with-hf)
- **[FS]** [Full function calling sequence](https://ai.google.dev/gemma/docs/functiongemma/full-function-calling-sequence-with-functiongemma)

## 1. Exact developer message

The API message must be:

```python
{
    "role": "developer",
    "content": "You are a model that can do function calling with the following functions",
}
```

The content string has:

- No trailing period or colon
- No trailing space or newline
- No schema appended to it
- No separator between the sentence and `<start_function_declaration>`

[F] identifies this phrase as the trigger that enables function calling. [HF] calls the developer message essential and passes the tools separately through `tools=[...]`. [FS] uses the same message in the complete sequence.

The rendered start is therefore:

```text
<bos><start_of_turn>developer
You are a model that can do function calling with the following functions<start_function_declaration>
```

Use one BOS only, with no newline after it.

## 2. Function declaration serialization

### Complete rendered weather declaration

[F] contains a hand-rendered example without spaces before selected closing braces. [FS] shows the processor-decoded rendering with two spaces before selected closing braces. The complete [FS] weather declaration is:

```text
<start_function_declaration>declaration:get_current_weather{description:<escape>Gets the current weather in a given location.<escape>,parameters:{properties:{location:{description:<escape>The city and state, e.g. "San Francisco, CA" or "Tokyo, JP"<escape>,type:<escape>STRING<escape>},unit:{description:<escape>The unit to return the temperature in.<escape>,enum:[<escape>celsius<escape>,<escape>fahrenheit<escape>],type:<escape>STRING<escape>} },required:[<escape>location<escape>],type:<escape>OBJECT<escape>} }<end_function_declaration>
```

The spaces in `...STRING<escape>} }...` and `...OBJECT<escape>} }` are significant in that displayed processor rendering. However, the two documentation pages disagree, so neither hand-rendered form should be hard-coded as a universal rule.

Generate the exact declaration for the installed snapshot instead:

```python
messages = [
    {
        "role": "developer",
        "content": "You are a model that can do function calling with the following functions",
    },
    {
        "role": "user",
        "content": "What's the temperature in London?",
    },
]

prompt_text = processor.apply_chat_template(
    messages,
    tools=[temperature_schema],
    add_generation_prompt=True,
    tokenize=False,
)

inputs = processor.apply_chat_template(
    messages,
    tools=[temperature_schema],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
)
```

Write `prompt_text` to the raw-prompt file exactly as returned. Do not strip whitespace, append a newline, or insert another BOS. Compare both its `repr()` and its resulting token IDs with the engine input.

### Serialization rules

According to [F], with examples confirmed by [HF] and [FS]:

1. The declaration begins with:

   ```text
   declaration:<bare_function_name>{
   ```

   The function name is not escape-delimited.

2. Structural keys remain bare:

   ```text
   description:
   parameters:
   properties:
   type:
   enum:
   required:
   ```

3. Every serialized string value is enclosed by `<escape>`:

   ```text
   description:<escape>text<escape>
   ```

4. JSON type names are uppercased and escaped:

   ```text
   type:<escape>STRING<escape>
   type:<escape>OBJECT<escape>
   ```

5. Array items are also escaped:

   ```text
   enum:[<escape>celsius<escape>,<escape>fahrenheit<escape>]
   required:[<escape>location<escape>]
   ```

6. Quotes inside an escaped string remain literal:

   ```text
   description:<escape>The city and state, e.g. "San Francisco, CA"<escape>
   ```

   Do not remove the quotes or convert them to `\"`.

7. Non-string response values remain bare:

   ```text
   temperature:15
   ```

   Booleans similarly remain `true` or `false`.

8. Commas and colons are structural and are not escape-delimited.

9. Field ordering should come from the installed template. The documentation examples happen to use a normalized-looking order, but that ordering should not be independently imposed on the engine.

10. The outer JSON Schema wrapper is not present. The function name follows `declaration:` directly, and the body is the function schema fields.

For nested parameters, [HF] recommends a manual JSON schema because automatic conversion from nested Python objects may collapse them to a generic object.

## 3. Turn delimiters and special tokens

### Normal turn framing

The rendered structure is:

```text
<bos><start_of_turn>ROLE
CONTENT<end_of_turn>
<start_of_turn>ROLE
CONTENT<end_of_turn>
<start_of_turn>model
```

Specifically:

- Exactly one `<bos>` appears at the beginning.
- `<start_of_turn>` is followed immediately by the role name.
- One LF follows the role name.
- Content is followed directly by `<end_of_turn>`.
- One LF follows `<end_of_turn>`.
- `add_generation_prompt=True` appends:

  ```text
  <start_of_turn>model\n
  ```

The API role `assistant` renders as `model`. API roles `developer`, `user`, `assistant`, and `tool` should therefore not be confused with the literal serialized role names.

### Function-control tokens

[F] defines these six control tokens:

| Pair | Purpose |
|---|---|
| `<start_function_declaration>` | Begin function declaration |
| `<end_function_declaration>` | End function declaration |
| `<start_function_call>` | Begin model function call |
| `<end_function_call>` | End model function call |
| `<start_function_response>` | Begin tool response |
| `<end_function_response>` | End tool response |

`<escape>` is a separate string-value delimiter, not a seventh function-control token.

The regular turn tokens are:

```text
<bos>
<eos>
<start_of_turn>
<end_of_turn>
```

`<pad>` is used for padding/padding configuration rather than as a turn delimiter.

### Reconstructed London prompt

The declared pages show the London schema and expected output but do not print a complete byte-exact rendered London prompt. The following is a useful reconstruction, not a replacement for processor output:

```text
<bos><start_of_turn>developer
You are a model that can do function calling with the following functions<start_function_declaration>declaration:get_current_temperature{description:<escape>Gets the current temperature for a given location.<escape>,parameters:{properties:{location:{description:<escape>The city name, e.g. San Francisco<escape>,type:<escape>STRING<escape>}},required:[<escape>location<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>
<start_of_turn>user
What's the temperature in London?<end_of_turn>
<start_of_turn>model
```

The documented output from the unmodified checkpoint is:

```text
<start_function_call>call:get_current_temperature{location:<escape>London<escape>}<end_function_call>
```

That output is attributed directly to the Hugging Face example in [HF].

### Tool-response continuation

[FS] shows this complete rendered continuation:

```text
<start_function_call>call:get_current_weather{location:<escape>Tokyo, Japan<escape>}<end_function_call><start_function_response>response:get_current_weather{temperature:15,weather:<escape>sunny<escape>}<end_function_response>The current weather in Tokyo is sunny with a temperature of 15 degrees Celsius.
```

There is no `<start_of_turn>` between the call and response in that decoded history. Do not manually duplicate this behavior independent of the template. Use message history, including:

```python
{
    "role": "tool",
    "content": {
        "name": "get_current_weather",
        "response": {
            "temperature": 15,
            "weather": "sunny",
        },
    },
}
```

For parallel calls, [FS] uses a list of such tool-result objects.

## 4. Generation parameters

Both [HF] and [FS] use:

```python
out = model.generate(
    **inputs.to(model.device),
    pad_token_id=processor.eos_token_id,
    max_new_tokens=128,
)
```

Therefore:

| Parameter | Documented value |
|---|---|
| `max_new_tokens` | `128` |
| `pad_token_id` | `processor.eos_token_id` |
| `temperature` | Not specified |
| `top_p` | Not specified |
| `top_k` | Not specified |
| `do_sample` | Not specified |

Because the examples omit sampling arguments, Hugging Face’s default is non-sampling/greedy decoding. That is an inference from the HF API default, not an explicit FunctionGemma recommendation. To reproduce the documented deterministic output, use true argmax/greedy decoding and disregard top-p/top-k.

[F] explicitly calls `<start_function_response>` an additional inference stop sequence. Implement at least:

- `<eos>`
- `<end_of_turn>`
- `<start_function_response>`

Stopping at `<end_function_call>` can be an engine protocol choice, but it is not documented as a required stop and must be paired with a consistent way to reconstruct the tool-response continuation.

## 5. Zero-shot behavior and fine-tuning

The base checkpoint does **not** require fine-tuning for the documented basic examples:

- [HF] loads `google/functiongemma-270m-it` unmodified and reports the London function call.
- [FS] uses the unmodified checkpoint for the weather declaration, call, tool response, and final natural-language answer.

[F] states that FunctionGemma was explicitly trained for:

- Single-turn function calling
- Parallel function calling

It also states that the model was not explicitly trained for:

- Multi-turn workflows
- Multi-step or chained workflows

For those cases it may generalize, with fine-tuning recommended for specific use cases.

Indirect queries are a separate semantic-quality issue. [F] shows that enriching a weak tool description can improve selection—for example, explaining that the function determines whether weather is hot or cold. That example does not establish that the explicit query “What's the temperature in London?” should fail, and a `0.81` probability on prose is not evidence of a description-only failure.

## 6. Priority checks for the current PULSAR-ASM failure

Because the prompt is already processor-identical at 98/98 tokens, investigate the engine in this order:

1. **Compare the same next-token position.**  
   On an x86 machine, run Hugging Face on the identical `input_ids` and inspect `logits[:, -1, :]`. Compare its top tokens and probabilities with the engine’s distribution after the complete prefill.

2. **Check prefill-to-decode alignment.**  
   Verify that:
   - The first sampled token follows the final `<start_of_turn>model\n`
   - KV-cache length equals the number of prompt tokens
   - Position IDs and cache reads use the correct next-token position
   - No token is dropped or duplicated at the decode boundary

3. **Compare numerics and architecture behavior.**  
   Check BF16 versus FP32 execution, normalization order, attention masks, sliding-window/local-attention behavior, RoPE parameters and scaling, embeddings, residuals, and any Gemma softcapping or final logit scaling. Absence of NaN/Inf does not prove numerical equivalence.

4. **Complete the control-token table.**  
   The repository reportedly lacks mappings for:

   ```text
   <start_function_call>
   <end_function_call>
   <start_function_response>
   <end_function_response>
   ```

   Obtain each ID from the installed tokenizer, for example:

   ```python
   token_id = processor.tokenizer.convert_tokens_to_ids(token)
   ```

   Validate that it maps to the intended added token. Do not infer these IDs from Gemma conventions or guess them from unused vocabulary slots. These mappings are not needed before the first function call but are required for correct model-output refeeding and response-stop handling.

5. **Fix stop handling.**  
   If the file decode loop currently terminates only on EOS, add `<end_of_turn>` and `<start_function_response>`. This may prevent junk after a correct call, but it cannot explain a wrong **first** token.

6. **Leave prompt bytes unchanged.**  
   With the existing 98/98 match, changing the developer message, escaping rules, declaration spaces, or tool description should not be the first response. The strongest remaining discriminator is an HF-versus-PULSAR comparison of first-token logits using the exact explicit London prefill.