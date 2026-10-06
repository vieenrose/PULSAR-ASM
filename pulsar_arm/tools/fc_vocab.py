import json
import glob

snap = "/home/luigi/.cache/huggingface/hub/models--google--functiongemma-270m-it/snapshots/39eccb091651513a5dfb56892d3714c1b5b8276c"
tj = json.load(open(snap + "/tokenizer.json", encoding="utf-8"))
v = tj["model"]["vocab"]
print("VOCAB:", len(v), flush=True)
print("ADDED:", [(t.get("content"), t.get("id")) for t in tj.get("added_tokens", [])][:12], flush=True)
print("CHAT-TEMPLATE:", repr(tj.get("chat_template", ""))[:400], flush=True)
cfg = json.load(open(snap + "/config.json"))
print("ARCH:", cfg.get("num_hidden_layers"), cfg.get("hidden_size"),
      cfg.get("intermediate_size"), cfg.get("num_attention_heads"),
      cfg.get("num_key_value_heads"), cfg.get("vocab_size"), flush=True)
print("NORM-EPS:", cfg.get("rms_norm_eps"), "ROPE:", str(cfg.get("rope_parameters", ""))[:120], flush=True)
# compare vocab to 270m
j = glob.glob("/home/luigi/.cache/huggingface/hub/models--google--gemma-3-270m-it-qat-q4_0-unquantized/snapshots/*/tokenizer.json")[0]
v270 = json.load(open(j, encoding="utf-8"))["model"]["vocab"]
same = sum(1 for s, i in v.items() if v270.get(s) == i)
print(f"vocab overlap vs 270m: {same}/{len(v)}", flush=True)
