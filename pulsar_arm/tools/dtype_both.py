import struct
import json
import glob

for pat in ["models--google--gemma-3-270m-it-qat-q4_0-unquantized",
            "models--google--functiongemma-270m-it"]:
    f = glob.glob("/home/luigi/.cache/huggingface/hub/" + pat +
                  "/snapshots/*/model.safetensors")[0]
    b = open(f, "rb")
    n = struct.unpack("<Q", b.read(8))[0]
    hdr = json.loads(b.read(n))
    e = hdr["model.embed_tokens.weight"]
    q = hdr.get("model.layers.0.self_attn.q_proj.weight",
                hdr.get("model.layers.0.self_attn.q.weight",
                        list(hdr.values())[3]))
    print(pat[-20:], e["dtype"], e["shape"], "|", q["dtype"], q["shape"], flush=True)
