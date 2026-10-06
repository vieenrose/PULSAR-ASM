import json

fc = "/home/luigi/.cache/huggingface/hub/models--google--functiongemma-270m-it/snapshots/39eccb091651513a5dfb56892d3714c1b5b8276c/config.json"
c = json.load(open(fc))
print("FC:", c.get("num_hidden_layers"), c.get("hidden_size"),
      c.get("intermediate_size"), c.get("num_attention_heads"),
      c.get("num_key_value_heads"), c.get("vocab_size"),
      c.get("rms_norm_eps"), c.get("rope_theta"),
      c.get("sliding_window"), c.get("layer_types", ["?"])[0] if isinstance(c.get("layer_types"), list) else c.get("layer_types"),
      c.get("query_pre_attn_scalar"), flush=True)
import glob
j = glob.glob("/home/luigi/.cache/huggingface/hub/models--google--gemma-3-270m-it-qat-q4_0-unquantized/snapshots/*/config.json")
if j:
    c2 = json.load(open(j[0]))
    print("270M:", c2.get("num_hidden_layers"), c2.get("hidden_size"),
          c2.get("intermediate_size"), c2.get("num_attention_heads"),
          c2.get("num_key_value_heads"), c2.get("vocab_size"),
          c2.get("rms_norm_eps"), c2.get("rope_theta"),
          c2.get("sliding_window"), flush=True)
else:
    print("no 270m config.json found", flush=True)
