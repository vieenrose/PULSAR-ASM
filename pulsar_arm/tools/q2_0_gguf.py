"""GGUF -> PULSAR blob converter for prism ternary (Q2_0) checkpoints.

Reads a prism-ml GGUF (Qwen3 + Q2_0 ternary) and writes a safetensors-shaped
blob that the asm engine already knows how to mmap and index, with two custom
dtypes:

  F32    plain fp32 tensor (the 113 norm tensors)
  Q2_0   ternary tensor, 128-weight groups, 34 bytes per group:
         fp16 scale + 32 bytes of 2-bit codes, w = (code - 1) * scale

Layout note: GGUF stores [dims0, dims1] with the quantisation groups running
along dims0, i.e. each of the dims1 rows is dims0 contiguous values. Our kernel
wants rows = dims1, cols = dims0, groups every 128 along cols - so the bytes are
copied verbatim and only the shape is swapped.

Usage:  q2_0_gguf.py <model.gguf> <out.bin> [vocab.bin] [bpe.bin]
        (vocab/bpe are delegated to the existing gguf_vocab/gguf_tok tools)
"""
import json
import struct
import sys

import numpy as np

GGUF_MAGIC = 0x46554747
T_F32, T_F16, T_Q2_0 = 0, 1, 42
GROUP = 128                 # weights per ternary group
GROUP_BYTES = 2 + GROUP // 4    # Q2_0: fp16 scale + 2-bit codes = 34
TRITS_PER_BYTE = 5          # base-3 packing: 3**5 = 243 <= 256
B3_CODE_BYTES = -(-GROUP // TRITS_PER_BYTE)     # 26 bytes for 128 trits
B3_GROUP_BYTES = 2 + B3_CODE_BYTES              # fp16 scale + base-3 codes = 28

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
            7: "<B", 10: "<Q", 11: "<q", 12: "<d"}


def q2_0_to_codes(raw, rows, ng):
    """(rows, ng, 34) Q2_0 blocks -> (scales (rows,ng) fp16 bytes, codes (rows,ng,128) 0..2)."""
    b = np.frombuffer(raw, dtype=np.uint8)[: rows * ng * GROUP_BYTES]
    b = b.reshape(rows, ng, GROUP_BYTES)
    scales = b[:, :, :2].copy()
    codes = np.empty((rows, ng, GROUP), dtype=np.uint8)
    for k in range(4):
        codes[:, :, k::4] = (b[:, :, 2:] >> (2 * k)) & 3
    return scales, codes


def codes_to_b3(codes):
    """(rows, ng, 128) codes 0..2 -> (rows, ng, 26) base-3 bytes (5 trits each)."""
    rows, ng, _ = codes.shape
    c = codes.astype(np.uint16)
    out = np.zeros((rows, ng, B3_CODE_BYTES), dtype=np.uint8)
    for k in range(B3_CODE_BYTES):
        chunk = c[:, :, k * TRITS_PER_BYTE:(k + 1) * TRITS_PER_BYTE]
        w = (3 ** np.arange(chunk.shape[2])).astype(np.uint16)
        out[:, :, k] = (chunk * w).sum(axis=2).astype(np.uint8)
    return out


def repack_tensor(raw, rows, ng):
    """Q2_0 tensor bytes -> lossless base-3 group bytes (scale + 26 code bytes)."""
    scales, codes = q2_0_to_codes(raw, rows, ng)
    out = np.concatenate([scales, codes_to_b3(codes)], axis=2)
    return out.tobytes()


class GGUF:
    def __init__(self, path, keep_meta=True):
        self.path = path
        self.f = open(path, "rb")
        magic, self.ver, self.n_tensors, self.n_kv = \
            struct.unpack("<IIQQ", self.f.read(24))
        assert magic == GGUF_MAGIC, hex(magic)
        self.meta = {}
        for _ in range(self.n_kv):
            k = self._str()
            t = struct.unpack("<I", self.f.read(4))[0]
            v = self._val(t)
            if keep_meta:
                self.meta[k] = v
        self.tensors = {}
        self.order = []
        for _ in range(self.n_tensors):
            name = self._str()
            nd = struct.unpack("<I", self.f.read(4))[0]
            dims = [struct.unpack("<Q", self.f.read(8))[0] for _ in range(nd)]
            tt = struct.unpack("<I", self.f.read(4))[0]
            off = struct.unpack("<Q", self.f.read(8))[0]
            self.tensors[name] = (dims, tt, off)
            self.order.append(name)
        align = self.meta.get("general.alignment", 32)
        self.data_start = (self.f.tell() + align - 1) // align * align

    def _str(self):
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8", "replace")

    def _val(self, t):
        if t == 8:
            return self._str()
        if t == 9:                       # array: skip payload, keep the length
            et = struct.unpack("<I", self.f.read(4))[0]
            n = struct.unpack("<Q", self.f.read(8))[0]
            for _ in range(n):
                self._val(et)
            return f"<array {n}>"
        return struct.unpack(_SCALARS[t], self.f.read(struct.calcsize(_SCALARS[t])))[0]

    def raw(self, name, offset=0, nbytes=None):
        dims, tt, off = self.tensors[name]
        self.f.seek(self.data_start + off + offset)
        return self.f.read(nbytes) if nbytes else self.f.read(self.nbytes(name))

    def hdr_dtype(self, name):
        return self.tensors[name][1]

    def nbytes(self, name):
        dims, tt, _ = self.tensors[name]
        n = 1
        for d in dims:
            n *= d
        if tt == T_F32:
            return n * 4
        if tt == T_F16:
            return n * 2
        if tt == T_Q2_0:
            assert dims[0] % GROUP == 0, (name, dims)
            return n // GROUP * GROUP_BYTES
        raise SystemExit(f"unsupported gguf type {tt} for {name}")


# GGUF -> HF (Qwen3) tensor names, so the engine sees the names it expects.
_NAME_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def _hf_name(name):
    if name in _NAME_MAP:
        return _NAME_MAP[name]
    if name.startswith("blk."):
        layer, role = name[4:].split(".", 1)
        role = {"attn_norm.weight": "input_layernorm.weight",
                "attn_q.weight": "self_attn.q_proj.weight",
                "attn_k.weight": "self_attn.k_proj.weight",
                "attn_v.weight": "self_attn.v_proj.weight",
                "attn_output.weight": "self_attn.o_proj.weight",
                "attn_q_norm.weight": "self_attn.q_norm.weight",
                "attn_k_norm.weight": "self_attn.k_norm.weight",
                "ffn_norm.weight": "post_attention_layernorm.weight",
                "ffn_gate.weight": "mlp.gate_proj.weight",
                "ffn_up.weight": "mlp.up_proj.weight",
                "ffn_down.weight": "mlp.down_proj.weight"}.get(role, role)
        return f"model.layers.{layer}.{role}"
    return name


def tensor_lengths(g):
    """Byte length per tensor from the header offsets (authoritative).

    GGUF pads some tensors out to the file alignment, so the next tensor's
    offset can exceed offset + computed size; the padding is never weights.
    """
    ents = sorted(((g.tensors[n][2], n) for n in g.order))
    lens = {}
    for k, (off, n) in enumerate(ents):
        nxt = ents[k + 1][0] if k + 1 < len(ents) else None
        mine = g.nbytes(n)
        lens[n] = mine if nxt is None else max(mine, nxt - off)
    return lens


def pulsar_meta(g):
    """Flat scalars the asm loader reads to configure the Qwen3 path.

    Deliberately numeric only (no strings to parse in asm): arch_id 1 = qwen3.
    rope_theta matters: 1e6 for 1.7B, 5e6 for 4B.
    """
    m = g.meta
    return {
        "pulsar.arch_id": 1,
        "pulsar.hidden": m["qwen3.embedding_length"],
        "pulsar.inter": m["qwen3.feed_forward_length"],
        "pulsar.layers": m["qwen3.block_count"],
        "pulsar.n_head": m["qwen3.attention.head_count"],
        "pulsar.n_kv": m["qwen3.attention.head_count_kv"],
        "pulsar.head_dim": m["qwen3.attention.key_length"],
        "pulsar.vocab": g.tensors["token_embd.weight"][0][1],
        "pulsar.eps": m["qwen3.attention.layer_norm_rms_epsilon"],
        "pulsar.rope_theta": m["qwen3.rope.freq_base"],
        # integer forms for the asm loader (it parses integers, not floats):
        # rope_theta is exactly 1e6 or 5e6, eps is 1e-6 => 1e-6 * 1e12 = 1e6
        "pulsar.rope_theta_int": int(round(m["qwen3.rope.freq_base"])),
        "pulsar.eps_e12": int(round(m["qwen3.attention.layer_norm_rms_epsilon"] * 1e12)),
        "pulsar.ternary_group": GROUP,
    }


def build_header(g, lens, pack="b3"):
    hdr, payload, offs = {}, [], 0
    for name in g.order:
        dims, tt, _ = g.tensors[name]
        nb = lens[name]
        if tt == T_Q2_0:
            rows, cols = dims[1], dims[0]
            dt = "Q2_0" if pack == "q2_0" else "B3_128"
            if pack != "q2_0":
                assert cols % GROUP == 0
                nb = rows * (cols // GROUP) * B3_GROUP_BYTES
        elif tt == T_F32:
            dt = "F32"
            rows, cols = (dims[1], dims[0]) if len(dims) == 2 else (dims[0], 1)
        else:
            raise SystemExit(f"{name}: unexpected dtype {tt}")
        hdr[_hf_name(name)] = {
            "dtype": dt, "shape": [rows, cols], "data_offsets": [offs, offs + nb]}
        payload.append((name, offs, nb))
        offs += nb
    hdr.update(pulsar_meta(g))
    hdr["pulsar.pack"] = 0 if pack == "q2_0" else 1
    return hdr, payload, offs


def convert(src, out, pack="b3"):
    """pack: "q2_0" keeps the GGUF blocks verbatim; "b3" repacks the codes
    losslessly as 5 trits per byte (28 B/128 groups instead of 34)."""
    g = GGUF(src)
    lens = tensor_lengths(g)
    hdr, payload, offs = build_header(g, lens, pack)
    blob = bytearray()
    for name, o, nb in payload:
        raw = g.raw(name)
        if pack != "q2_0" and g.hdr_dtype(name) == T_Q2_0:
            rows, cols = hdr[_hf_name(name)]["shape"]
            raw = repack_tensor(raw, rows, cols // GROUP)
        blob += raw
    with open(out, "wb") as fh:
        h = json.dumps(hdr).encode()
        fh.write(struct.pack("<Q", len(h)))
        fh.write(h)
        fh.write(blob)
    mb = (8 + len(h) + len(blob)) / 1e6
    print(f"{src} -> {out}: {len(hdr)} tensors, {mb:.1f} MB "
          f"(header {len(h)} B, data {len(blob)/1e6:.1f} MB)", flush=True)
    return g


def main():
    src, out = sys.argv[1], sys.argv[2]
    pack = sys.argv[3] if len(sys.argv) > 3 else "b3"
    g = convert(src, out, pack)
    m = g.meta
    print("meta:", {k: v for k, v in m.items() if not isinstance(v, str)
                    or len(v) < 40})


if __name__ == "__main__":
    main()
