#!/usr/bin/env python3
# HF reference dump for the LFM3 audio encoder (S3).
# Fixed input: deterministic features (128 mel bins x T frames), so no audio file or
# mel front end is involved. Saves every stage the port must match:
#   ds_out      downsampler output            (1, T/8, 768)
#   layer_NN    output of encoder layer NN    (1, T/8, 768)   NN = 0..16
#   enc_out     final encoder output (after embedding_norm)   (1, T/8, 768)
#   proj_out    projector output              (1, T/8, 2048)
# Run with -I (no script-directory imports) from a directory that is not the model dir.
import sys, os, numpy as np, torch
MODEL = "/home/louis/models/lfm3/LFM3-Audio-2B"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/enc_ref"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 400   # mel frames -> T/8 tokens
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, MODEL)
from transformers import AutoModelForSeq2SeqLM
torch.set_grad_enabled(False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32).eval()
M = model.model if hasattr(model, "model") and hasattr(model.model, "audio_encoder") else model
enc = M.audio_encoder
rng = np.random.default_rng(1234)
feats = torch.from_numpy(rng.standard_normal((1, 128, T)).astype(np.float32) * 0.8 - 0.2)
feats.numpy().tofile(os.path.join(OUT, "input_128xT_f32.bin"))
lengths = torch.tensor([T])
np.save(os.path.join(OUT, "input.npy"), feats.numpy())

ds = enc.downsample(feats, lengths).transpose(1, 2)           # (1, C, T/8) -> (1, T/8, C)
np.save(os.path.join(OUT, "ds_out.npy"), ds.numpy())

hs = []
h = ds
out = enc.lfm(inputs_embeds=ds, use_cache=False, output_hidden_states=True,
              sliding_window=enc.config.sliding_window,
              attention_mask={"full_attention": None, "conv": None})
# hidden_states[0] is the input, [i] the output of layer i-1; last one is post-norm
for i, hsi in enumerate(out.hidden_states):
    np.save(os.path.join(OUT, "hidden_%02d.npy" % i), hsi.numpy())
np.save(os.path.join(OUT, "enc_out.npy"), out.last_hidden_state.numpy())
proj = M.audio_projector(out.last_hidden_state)
np.save(os.path.join(OUT, "proj_out.npy"), proj.numpy())
print("T=%d tokens=%d layers=%d hidden_states=%d" % (T, ds.shape[1], enc.config.lfm.num_hidden_layers if hasattr(enc.config,'lfm') else -1, len(out.hidden_states)))
print("ds", tuple(ds.shape), "enc_out", tuple(out.last_hidden_state.shape), "proj", tuple(proj.shape))
