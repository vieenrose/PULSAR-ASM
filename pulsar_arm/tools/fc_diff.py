import struct
import json
import glob

def load_header(path):
    b = open(path, "rb")
    n = struct.unpack("<Q", b.read(8))[0]
    return json.loads(b.read(n))

fc = ("/home/luigi/.cache/huggingface/hub/models--google--functiongemma-270m-it"
      "/snapshots/39eccb091651513a5dfb56892d3714c1b5b8276c/model.safetensors")
mine = glob.glob("/home/luigi/.cache/huggingface/hub/models--google--gemma-3-270m-it-qat-q4_0-unquantized/snapshots/*/model.safetensors")[0]
H1, H2 = load_header(fc), load_header(mine)
K1 = sorted(k for k in H1 if k != "__metadata__")
K2 = sorted(k for k in H2 if k != "__metadata__")
print("FC:", len(K1), "MINE:", len(K2), flush=True)
# compare by (short name, shape, dtype): strip HF prefix
def short(k):
    return k.replace("model.layers.", "L").replace("model.", "")
S1 = sorted((short(k), tuple(H1[k]["shape"]), H1[k]["dtype"]) for k in K1)
S2 = sorted((short(k), tuple(H2[k]["shape"]), H2[k]["dtype"]) for k in K2)
only1 = [x for x in S1 if (x[0], x[1]) not in {(y[0], y[1]) for y in S2}]
only2 = [x for x in S2 if (x[0], x[1]) not in {(y[0], y[1]) for y in S1}]
print("ONLY-FC (name,shape):", only1[:10], flush=True)
print("ONLY-MINE:", only2[:10], flush=True)
# order comparison (file order, not sorted): does FC file order == MINE order?
O1 = [k for k in H1 if k != "__metadata__"]
with open(mine, "rb") as bb:
    n = struct.unpack("<Q", bb.read(8))[0]
    H2o = json.loads(bb.read(n))
O2 = [k for k in H2o if k != "__metadata__"]
print("SAMPLE-FC-ORDER:", O1[:4], flush=True)
print("SAMPLE-MINE-ORDER:", O2[:4], flush=True)
print("SAME-ORDER:", [short(a) for a in O1] == [short(b) for b in O2], flush=True)
