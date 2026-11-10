#!/usr/bin/env python3
"""Teacher-forced HF top-5 / top-2 margin at the first K+1 greedy steps of an ASR clip.
Usage: hf_step_margin.py <wav> <hf_golden.json> <K>   (venv lfm3, fp32 CPU)
Run #209: zh fixture step 2 margin 0.082 (437 vs 355) = the engine's divergence point."""
import sys, json, numpy as np, soundfile as sf, torch
from transformers import AutoModelForMultimodalLM, AutoProcessor
torch.set_grad_enabled(False); torch.set_num_threads(6)
M="/home/louis/models/lfm3/LFM3-Audio-2B"; W=sys.argv[1]; G=json.load(open(sys.argv[2]))["decoding"]["generated_token_ids"]
proc=AutoProcessor.from_pretrained(M,trust_remote_code=True,local_files_only=True)
w,_=sf.read(W,dtype="float32")
conv=[{"role":"system","content":"Perform ASR in English."},{"role":"user","content":[{"type":"audio","audio":np.ascontiguousarray(w)}]}]
inp=proc.apply_chat_template(conv,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt",processor_kwargs={"sampling_rate":16000})
n=inp["input_ids"].shape[1]; K=int(sys.argv[3])
ids=torch.cat([inp["input_ids"],torch.tensor([G[:K]])],1)
m=AutoModelForMultimodalLM.from_pretrained(M,dtype=torch.float32,trust_remote_code=True).eval()
out=m(input_ids=ids,attention_mask=torch.ones_like(ids),input_features=inp["input_features"],feature_attention_mask=inp["feature_attention_mask"])
L=out.logits[0]
for s in range(K+1):
    l=L[n-1+s].float(); v,i=l.topk(5)
    print("step %d gold %s top5 %s logits %s margin %.4f"%(s, G[s], i.tolist(), [round(x,3) for x in v.tolist()], (v[0]-v[1]).item()))
    for t in (355,437): print("   logit[%d]=%.4f"%(t,l[t].item()))
