"""Dump GGUF tensor map: name, type, shape, absolute data offset, bytes.
Usage: python3 gguf_map.py <file.gguf> [prefix-filter]"""
import struct, sys

QNAMES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
 10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",
 18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",24:"I8",
 25:"I16",26:"I32",27:"I64",28:"F64",29:"IQ1_M",30:"BF16",31:"TQ1_0",32:"TQ2_0",
 33:"MXFP4",143:"PTQ1_0"}
# bytes per element (or per block) for size calc; block sizes for quantized
BSZ = {143:(128,28)}  # type -> (elems_per_block, bytes_per_block)

def rd(f, n):
    b = f.read(n)
    if len(b) < n:
        raise SystemExit("truncated file")
    return b

def su(f):
    n = struct.unpack("<Q", rd(f, 8))[0]
    return rd(f, n).decode()

def skip_val(f, t):
    if t in (0, 4, 10):
        rd(f, {0:1, 4:4, 10:8}[t])
    elif t in (1, 3, 5, 11):
        rd(f, {1:1, 3:2, 5:4, 11:8}[t])
    elif t == 2:
        rd(f, 2)
    elif t in (6, 12):
        rd(f, {6:4, 12:8}[t])
    elif t == 7:
        rd(f, 1)
    elif t == 8:
        n = struct.unpack("<Q", rd(f, 8))[0]
        rd(f, n)
    elif t == 9:
        # array: elem type + len + elems
        et = struct.unpack("<I", rd(f, 4))[0]
        n = struct.unpack("<Q", rd(f, 8))[0]
        if et == 8:
            for _ in range(n):
                su(f)
        else:
            sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,10:8,11:8,12:8}.get(et)
            if sz is None:
                raise SystemExit("unknown array elem type %d" % et)
            rd(f, n * sz)
    else:
        raise SystemExit("unknown kv type %d" % t)

def main(path, filt=""):
    f = open(path, "rb")
    assert rd(f, 4) == b"GGUF"
    ver = struct.unpack("<I", rd(f, 4))[0]
    nt, nkv = struct.unpack("<QQ", rd(f, 16))
    for _ in range(nkv):
        su(f)
        skip_val(f, struct.unpack("<I", rd(f, 4))[0])
    infos = []
    for _ in range(nt):
        name = su(f)
        nd = struct.unpack("<I", rd(f, 4))[0]
        dims = struct.unpack("<%dQ" % nd, rd(f, 8 * nd))
        typ = struct.unpack("<I", rd(f, 4))[0]
        off = struct.unpack("<Q", rd(f, 8))[0]
        infos.append((name, typ, dims, off))
    data_start = (f.tell() + 31) & ~31
    total = 0
    for (name, typ, dims, off) in infos:
        if filt and not name.startswith(filt):
            continue
        ne = 1
        for d in dims:
            ne *= d
        if typ in BSZ:
            eb, bb = BSZ[typ]
            nbytes = (ne // eb) * bb
        elif typ == 0:
            nbytes = ne * 4
        elif typ == 1:
            nbytes = ne * 2
        elif typ == 30:
            nbytes = ne * 2
        else:
            nbytes = -1
        total += max(nbytes, 0)
        print("%-48s %-7s %-22s off=%-12d nbytes=%d" % (
            name, QNAMES.get(typ, "T%d" % typ),
            "x".join(map(str, dims)), data_start + off, nbytes))
    if not filt:
        print("tensors=%d data_start=%d total_data=%d" % (nt, data_start, total))

main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
