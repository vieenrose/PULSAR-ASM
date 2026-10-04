"""Bisect an engine crash by inserting an early return after each stage."""
import subprocess
import sys

# Probe by stage: build the engine with an early return inserted after each
# statement, so a hang or fault localizes to one stage instead of "somewhere".
# When probing the LAYER, the copy is included by an otherwise unmodified engine
# (the layer has no PLSE trailer of its own, so building it standalone fails).
ENGINE = "engine/gemma4_engine_flat.asm"
LAYER = "sub_assemblies/sub_gemma4_layer_flat.asm"
TARGET = sys.argv[1] if len(sys.argv) > 1 else ENGINE
ENGINE_EPI = ("    mov     rax, 777\n    add     rsp, 136\n    pop     r15\n"
              "    pop     r14\n    pop     r13\n    pop     r12\n    pop     rbp\n"
              "    pop     rbx\n    ret")
LAYER_EPI = ("    mov     rax, 777\n    add     rsp, 192\n    pop     r14\n"
             "    pop     r13\n    pop     r12\n    pop     rbp\n    ret")
if TARGET == LAYER:
    ANCHORS = ["call    rmsnorm_avx2",                  # 0 input norm
               "GEMMB   <[rbx + CTR_BUF_Q]>",           # 1 q proj
               "jb      .ly_qrow",                      # 2 q norm + rope
               "jb      .ly_kvrow",                     # 3 k/v norm + rope
               "jb      .ly_wrow",                      # 4 attention
               "GEMMB   <[rbx + CTR_BUF_T1]>, <[r15 + D_OW]>",   # 5 o proj
               "call    geglu_avx2",                    # 6 mlp activations
               "call    scale_avx2",                    # 7 whole layer
    ]
    EPI, SRC = LAYER_EPI, LAYER
else:
    ANCHORS = ["jb      .step_emb",                     # 0 embedding done
               "GEMMB   <[rbx + CTR_BUF_PLE_NEXT]>",    # 1 ple projection done
               "jb      .step_plerow",                  # 2 ple combine done
               "mov     qword [rbx + CTR_MARK], 5",     # 3 layers done
               "GEMMB   <[rbx + CTR_BUF_LOGITS]>",      # 4 final norm done
               "mov     rax, [rbx + CTR_OUTTOK]"]       # 5 logits done
    EPI, SRC = ENGINE_EPI, ENGINE
RUN = """
import sys; sys.path.insert(0,'.')
from runtime import gemma4_model as G
G.ENGINE_SRC = 'engine/_probe.asm'
g = G.Gemma4('/mnt/edge/pulsar/gemma4_e2b.bin', max_seq=64, verbose=False)
print('OK', g.forward(2))
"""

for i, a in enumerate(ANCHORS):
    lines = open(SRC).read().split("\n")
    for j, l in enumerate(lines):
        if a in l:
            break
    else:
        print(f"stage {i}: anchor not found: {a!r}")
        continue
    lines.insert(j + 1, EPI)
    if SRC == LAYER:
        open("sub_assemblies/_probe_layer.asm", "w").write("\n".join(lines))
        eng = open(ENGINE).read().replace("../sub_assemblies/sub_gemma4_layer_flat.asm",
                                          "../sub_assemblies/_probe_layer.asm")
        open("engine/_probe.asm", "w").write(eng)
    else:
        open("engine/_probe.asm", "w").write("\n".join(lines))
    try:
        r = subprocess.run([sys.executable, "-u", "-c", RUN], capture_output=True,
                           text=True, timeout=45)
        rc, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = "HANG", "", ""
    except Exception as e:
        rc, out, err = repr(e), "", "" 
    line = (out.strip().splitlines() or [""])[-1]
    elines = [l for l in err.splitlines() if l.strip() and not l.startswith(" ")]
    print(f"stage {i} after {a!r:46s} rc={rc} {line or (elines[-1] if elines else '')}", flush=True)
