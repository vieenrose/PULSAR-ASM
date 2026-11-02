#!/usr/bin/env python3
"""Localization probe: find the FIRST encoder stage where our derivation diverges from HF,
on the 30 s sub-clip (480,000 samples -> 2999 mel frames -> 375 encoder tokens).

Why this probe exists. Run #318/#319 established that the numpy fp32 spec (§22's
re-derivation), the C reference (Q8) and the CUDA encoder agree with EACH OTHER to cos
0.9997 at every position, and all three drift from HF identically (per-row cos 1.0000 at
rows 0-20, 0.9755 at 21-99, 0.9398 at 100-199, 0.9099 at 200-299, 0.8775 at 300-374). HF
fp32 == HF bf16 (cos 0.99996), so it is not HF precision; a global RoPE offset is a
no-op by relative-position invariance. So the bug is in the SHARED DERIVATION and is
position-dependent. This script says which stage introduces it.

What it does
  1. mel: HF's own feature extractor over the fixture's first 480,000 samples, through the
     same chat-template call the golden makes; asserts num_frames == 2999 (and the
     checkpoint's own token formula == 375).
  2. HF dump: the CANONICAL encoder path - the body of Lfm3AudioEncoder.forward, including
     its create_sliding_window_causal_mask call - with per-layer activations saved under
     the names dump_encoder.py/check_enc_c.py/cmp_enc.py already use (input.npy,
     ds_out.npy, hidden_00..hidden_17.npy, enc_out.npy, proj_out.npy) into a NEW dir
     (ref30/), so ref/ and ref2/ (the §22 gate) are untouched. Two assertions pin the
     provenance: the dump equals `audio_encoder(feats, lengths)` and proj_out equals the
     golden's entry point `model.model.get_audio_features(...)`.
  3. Path control: dump_encoder.py's own stage calls
     (`attention_mask={"full_attention": None, "conv": None}`) are run too and saved under
     ref30/dumppath/, and per-stage cos against the canonical dump is reported. A mask dict
     is used AS GIVEN by Lfm3Model, so this is not necessarily HF's encoder forward - the
     §22 gate was measured against dumps from this path.
  4. Our side: enc_numpy.encoder()/projector() on HF's own weights, same mel. Per-stage,
     per-row cosine vs HF in the buckets 0-2, 3-20, 21-99, 100-199, 200-299, 300-374, for
     the stage OUTPUT, the stage INPUT (what it inherits) and the stage INCREMENT (out-in,
     what it introduces), plus mean row L2 both sides.
  5. Mechanism test (--window-test, on by default): re-run the same numpy spec with a
     W-token window on the attention layers and report the same table, so a candidate for
     the root cause is confirmed or killed by numbers, not by reading.

Usage
  /home/louis/venvs/lfm3/bin/python -I probe30.py [--out ref30] [--hf-device cpu]
      [--skip-numpy] [--skip-hf] [--window-test N] [--no-window-test] [--json FILE]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LFM3 = os.path.dirname(HERE)
sys.path.insert(0, LFM3)                               # for hf_asr_golden constants
sys.path.insert(0, HERE)                               # for enc_numpy

import hf_asr_golden as G                               # noqa: E402  (constants only)

CLIP30_SAMPLES = 480_000                                # 30.000 s @ 16 kHz
MEL_FRAMES_30S = 2999
TOKENS_30S = 375
# The row buckets the open comparison is written in (§36 / hf_asr_tower_fp32.py BUCKETS_C).
BUCKETS = [(0, 3), (3, 21), (21, 100), (100, 200), (200, 300), (300, 375)]
DROP = 0.999                                            # "first stage below this at rows 200+"
LAYER_TYPES = ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
               "conv", "conv", "full_attention", "conv", "conv", "full_attention",
               "conv", "conv", "full_attention", "full_attention", "full_attention"]


# --------------------------------------------------------------------------- metrics

def row_cos(a, b):
    """Per-row cosine, float64, shape (rows,)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = (a * b).sum(-1)
    den = np.sqrt((a * a).sum(-1)) * np.sqrt((b * b).sum(-1))
    return num / np.maximum(den, 1e-300)


def bstats(cos, lo, hi):
    seg = cos[lo:hi]
    return [float(seg.mean()), float(seg.min())]


def l2(m):
    return float(np.sqrt((np.asarray(m, np.float64) ** 2).sum(-1)).mean())


def stage_names():
    """(name, cols, hf_file, input_file) for ds_out, layer_00..16, enc_out, proj_out.
    hidden_00 is the encoder input (= ds_out) and hidden_NN the output of layer NN-1, the
    convention dump_encoder.py writes and check_enc_c.py/cmp_enc.py read."""
    names = [("ds_out", 768, "ds_out.npy", None)]
    for i in range(17):
        names.append(("layer_%02d" % i, 768, "hidden_%02d.npy" % (i + 1),
                      "ds_out.npy" if i == 0 else "hidden_%02d.npy" % i))
    names.append(("enc_out", 768, "enc_out.npy", "hidden_17.npy"))
    names.append(("proj_out", 2048, "proj_out.npy", "enc_out.npy"))
    return names


def stage_type(name):
    if name == "ds_out":
        return "mel"
    if name == "enc_out":
        return "norm"
    if name == "proj_out":
        return "proj"
    return "attn" if LAYER_TYPES[int(name[-2:])] == "full_attention" else "conv"


# --------------------------------------------------------------------------- mel

def mel_30s():
    """HF's own feature extractor over the fixture's first 480,000 samples, through the
    same chat-template call the golden and the fp32-tower test make."""
    import soundfile as sf
    from transformers import AutoProcessor

    waveform, rate = sf.read(G.FIXTURE, dtype="float32")
    assert rate == G.SAMPLING_RATE and waveform.ndim == 1, (rate, waveform.shape)
    assert waveform.shape[0] >= CLIP30_SAMPLES, waveform.shape
    wav = np.ascontiguousarray(waveform[:CLIP30_SAMPLES], dtype=np.float32)
    assert wav.shape[0] == CLIP30_SAMPLES, wav.shape

    proc = AutoProcessor.from_pretrained(G.MODEL, trust_remote_code=True, local_files_only=True)
    fe = proc.feature_extractor
    # The checkpoint's own formulas, asserted, not assumed.
    nfr = int(fe.num_frames(CLIP30_SAMPLES))
    ntok = int(fe.num_encoder_tokens(nfr))
    assert nfr == MEL_FRAMES_30S, (nfr, MEL_FRAMES_30S)
    assert ntok == TOKENS_30S, (ntok, TOKENS_30S)

    conversation = [
        {"role": "system", "content": G.SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "audio", "audio": wav}]},
    ]
    t0 = time.time()
    inputs = proc.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"sampling_rate": G.SAMPLING_RATE},
    )
    feats = inputs["input_features"].contiguous()
    mask = inputs["feature_attention_mask"].contiguous()
    assert tuple(feats.shape) == (1, 128, MEL_FRAMES_30S), tuple(feats.shape)
    assert int(mask.sum()) == MEL_FRAMES_30S, int(mask.sum())
    meta = {
        "fixture_sha256": G.sha256_file(G.FIXTURE),
        "clip_samples": int(wav.shape[0]),
        "clip_seconds": CLIP30_SAMPLES / G.SAMPLING_RATE,
        "mel_shape": list(feats.shape),
        "mel_frames": int(feats.shape[-1]),
        "mel_dtype": str(feats.dtype),
        "feature_extractor": type(fe).__name__,
        "feature_attention_mask_ones": int(mask.sum()),
        "num_frames_formula": nfr,
        "num_encoder_tokens_formula": ntok,
        "mel_seconds": round(time.time() - t0, 2),
    }
    return feats, mask, meta


# --------------------------------------------------------------------------- HF dumps

def _save(out_dir, ds, hidden, enc_out, proj, np):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "ds_out.npy"), ds)
    for i, h in enumerate(hidden):                             # hidden[0] == ds
        np.save(os.path.join(out_dir, "hidden_%02d.npy" % i), h)
    np.save(os.path.join(out_dir, "enc_out.npy"), enc_out)
    np.save(os.path.join(out_dir, "proj_out.npy"), proj)


def hf_dump(feats, mask, out_dir, device):
    """Run HF's tower on the 30 s mel and save the stage files.

    ref30/            the CANONICAL path (Lfm3AudioEncoder.forward, window mask included)
    ref30/dumppath/   dump_encoder.py's stage calls, for the path control
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    t0 = time.time()
    model = AutoModelForSeq2SeqLM.from_pretrained(G.MODEL, trust_remote_code=True,
                                                 dtype=torch.float32).eval()
    load_seconds = time.time() - t0
    M = model.model if hasattr(model, "model") and hasattr(model.model, "audio_encoder") else model
    enc, proj_mod = M.audio_encoder, M.audio_projector
    enc.to(device)
    proj_mod.to(device)
    dtypes = {str(p.dtype) for p in enc.parameters()} | {str(p.dtype) for p in proj_mod.parameters()}
    assert dtypes == {"torch.float32"}, dtypes
    # The checkpoint's own module object, so create_sliding_window_causal_mask is literally
    # the function HF's encoder calls (it uses relative imports, so import it from sys.modules).
    MA = sys.modules[type(model).__module__]
    assert hasattr(MA, "create_sliding_window_causal_mask"), type(model).__module__
    feats = feats.to(device)
    lengths = mask.sum(-1).to(torch.long).to(device)
    npy = np

    os.makedirs(out_dir, exist_ok=True)
    feats.cpu().numpy().astype(np.float32).tofile(os.path.join(out_dir, "input_128xT_f32.bin"))
    np.save(os.path.join(out_dir, "input.npy"), feats.cpu().numpy())

    def npy_dump(tag):
        """Run one encoder variant, return (ds, hidden[0..17], enc_out, proj) as numpy."""
        ds = enc.downsample(feats, lengths).transpose(1, 2)
        if tag == "canonical":
            # the body of Lfm3AudioEncoder.forward, verbatim, plus output_hidden_states
            enc.lfm.config.sliding_window = enc.config.sliding_window
            amask = MA.create_sliding_window_causal_mask(
                config=enc.lfm.config, inputs_embeds=ds, attention_mask=None,
                past_key_values=None, position_ids=None)
        elif tag == "dumppath":
            amask = None                       # what dump_encoder.py passes for full_attention
        else:
            raise ValueError(tag)
        out = enc.lfm(inputs_embeds=ds, use_cache=False, output_hidden_states=True,
                      sliding_window=enc.config.sliding_window,
                      attention_mask={"full_attention": amask, "conv": None})
        proj = proj_mod(out.last_hidden_state)
        return (ds.cpu().numpy(), [h.cpu().numpy() for h in out.hidden_states],
                out.last_hidden_state.cpu().numpy(), proj.cpu().numpy(), amask)

    t0 = time.time()
    ds, hidden, enc_out, proj, amask = npy_dump("canonical")
    canonical_seconds = time.time() - t0
    ds_d, hidden_d, enc_out_d, proj_d, _ = npy_dump("dumppath")

    # Provenance assertions: the canonical dump IS HF's encoder forward, and its proj_out
    # IS the tensor the golden splices into the prompt.
    canon, canon_lengths = enc(feats, lengths)
    canon_diff = float((canon.cpu() - torch.from_numpy(enc_out)).abs().max())
    assert canon_diff == 0.0, canon_diff
    gaf = M.get_audio_features(feats, mask)
    gaf_diff = float((gaf.cpu() - torch.from_numpy(proj)).abs().max())
    assert gaf_diff == 0.0, gaf_diff

    _save(out_dir, ds, hidden, enc_out, proj, npy)
    _save(os.path.join(out_dir, "dumppath"), ds_d, hidden_d, enc_out_d, proj_d, npy)

    # The mask HF actually hands its attention layers: allowed keys per query row.
    keys_per_row = None
    if amask is not None:
        m = amask[0, 0]                                        # (q, k) additive mask
        keys_per_row = (m == 0).sum(-1).cpu().tolist()

    lm = enc.lfm
    cfg = lm.config
    rope_keys = sorted(cfg.rope_parameters) if isinstance(cfg.rope_parameters, dict) else None
    theta = float(cfg.rope_parameters["rope_theta"]) if isinstance(cfg.rope_parameters, dict) \
        and "rope_theta" in cfg.rope_parameters else None
    partial = cfg.rope_parameters.get("partial_rotary_factor", 1.0) if isinstance(cfg.rope_parameters, dict) else 1.0
    first_attn = LAYER_TYPES.index("full_attention")
    layer = lm.layers[first_attn]
    attn = getattr(layer, "self_attn", None)
    h_in = torch.from_numpy(hidden[first_attn])
    gate_present = hasattr(attn, "gate_proj")
    gate_stats = {}
    if gate_present:
        g = torch.sigmoid(attn.gate_proj(layer.operator_norm(h_in)))
        gate_stats = {"hf_attn_gate_max_off_1": float((g - 1).abs().max()),
                      "hf_attn_gate_min": float(g.min()),
                      "hf_attn_gate_mean": float(g.mean())}
    inv = getattr(lm.rotary_emb, "inv_freq", None)
    rope_stats = {}
    if inv is not None:
        rope_stats["hf_rope_inv_freq_len"] = int(inv.numel())
        rope_stats["hf_rope_inv_freq_nonzero"] = int((inv != 0).sum())
        rope_stats["hf_rope_inv_freq_max"] = float(inv.max())
        rope_stats["hf_rope_inv_freq_min"] = float(inv.min())
        pos = torch.arange(TOKENS_30S, device=device)[None]
        cos, sin = lm.rotary_emb(torch.from_numpy(ds).to(device), position_ids=pos)
        rope_stats["hf_rope_max_cos_minus_1"] = float((cos - 1).abs().max())
        rope_stats["hf_rope_max_abs_sin"] = float(sin.abs().max())
    diag = {
        "hf_device": str(device),
        "hf_weight_dtypes": sorted(dtypes),
        "hf_attn_implementation": str(cfg._attn_implementation),
        "hf_lfm_class": type(lm).__name__,
        "hf_layer_class": type(layer).__name__,
        "hf_attn_class": type(attn).__name__,
        "hf_conv_class": type(getattr(lm.layers[0], "conv", None)).__name__,
        "hf_mlp_class": type(layer.feed_forward).__name__,
        "hf_mlp_intermediate_size": int(layer.feed_forward.w1.weight.shape[0]),
        "hf_attn_gate_present": bool(gate_present),
        "hf_attn_gate_first_layer": first_attn,
        "hf_load_seconds": round(load_seconds, 1),
        "hf_canonical_seconds": round(canonical_seconds, 1),
        "hf_canonical_forward_max_abs_diff": canon_diff,
        "hf_get_audio_features_max_abs_diff": gaf_diff,
        "hf_canonical_output_lengths": canon_lengths.cpu().tolist(),
        "hf_encoder_sliding_window": int(enc.config.sliding_window),
        "hf_lfm_sliding_window": int(getattr(cfg, "sliding_window", 0) or 0),
        "hf_layer_types": list(cfg.layer_types),
        "hf_use_sandwich_norm": bool(getattr(cfg, "use_sandwich_norm", False)),
        "hf_conv_L_cache": int(cfg.conv_L_cache),
        "hf_conv_bias": bool(cfg.conv_bias),
        "hf_norm_eps": float(cfg.norm_eps),
        "hf_head_dim": int(getattr(cfg, "head_dim",
                                   cfg.hidden_size // cfg.num_attention_heads)),
        "hf_rope_parameters_keys": rope_keys,
        "hf_rope_theta": theta,
        "hf_partial_rotary_factor": float(partial),
        "hf_mask_keys_per_row": None,
        "hf_mask_keys_min": None,
        "hf_mask_keys_max": None,
    }
    diag.update(gate_stats)
    diag.update(rope_stats)
    if keys_per_row is not None:
        diag["hf_mask_keys_per_row"] = [keys_per_row[i] for i in (0, 1, 2, 5, 30, 31, 32, 33, 100, 200, 374)]
        diag["hf_mask_keys_min"] = int(min(keys_per_row))
        diag["hf_mask_keys_max"] = int(max(keys_per_row))
    del model
    return diag


# --------------------------------------------------------------------------- our side

def attn_block_windowed(E, h, i, cos, sin, window):
    """enc_numpy.attn_block with a W-token window on the causal mask (W None = full causal).
    Same weights, same order, same RMSNorm/RoPE calls as the spec; the mask line is the
    only difference, so the test isolates the mask."""
    p = "encoder.lfm.lfm.layers.%d.self_attn." % i
    T = h.shape[0]
    q = (h @ E.W(p + "q_proj.weight").T).reshape(T, 8, E.HEAD)
    k = (h @ E.W(p + "k_proj.weight").T).reshape(T, 4, E.HEAD)
    v = (h @ E.W(p + "v_proj.weight").T).reshape(T, 4, E.HEAD)
    q = E.rms(q, E.W(p + "q_layernorm.weight"))
    k = E.rms(k, E.W(p + "k_layernorm.weight"))
    c, s = cos[:, None, :], sin[:, None, :]
    q = q * c + E.rot_half(q) * s
    k = k * c + E.rot_half(k) * s
    out = np.zeros((T, 8, E.HEAD), np.float32)
    i_idx = np.arange(T)[:, None]
    j_idx = np.arange(T)[None, :]
    allowed = (j_idx <= i_idx)
    if window is not None:
        allowed = allowed & (j_idx > i_idx - window)
    for hq in range(8):
        kv = hq // 2
        sc = (q[:, hq, :] @ k[:, kv, :].T) * (E.HEAD ** -0.5)
        sc = np.where(allowed, sc, -np.inf)
        sc = sc - sc.max(axis=-1, keepdims=True)
        pr = np.exp(sc)
        pr = pr / pr.sum(axis=-1, keepdims=True)
        out[:, hq, :] = pr @ v[:, kv, :]
    return out.reshape(T, 8 * E.HEAD) @ E.W(p + "out_proj.weight").T


def spec_stages(E, feat, window):
    """The §22 spec's encoder loop, with the attention mask selectable. window=None is the
    spec as written; window=W routes the attention layers through attn_block_windowed."""
    x = E.downsample(feat)
    cos, sin = E.rope_tables(x.shape[0])
    stages = {"ds_out": x}
    for i, t in enumerate(LAYER_TYPES):
        p = "encoder.lfm.lfm.layers.%d." % i
        h = E.rms(x, E.W(p + "operator_norm.weight"))
        if t == "conv":
            op = E.conv_block(h, i)
        elif window is None:
            op = E.attn_block(h, i, cos, sin)
        else:
            w = window.get(i) if isinstance(window, dict) else window
            op = attn_block_windowed(E, h, i, cos, sin, w)
        x = x + op
        x = x + E.mlp(E.rms(x, E.W(p + "ffn_norm.weight")), i)
        stages["layer_%02d" % i] = x
    x = E.rms(x, E.W("encoder.lfm.lfm.embedding_norm.weight"))
    stages["enc_out"] = x
    return stages


def table_against_hf(stages, out_dir, proj):
    """Per-stage per-bucket cos of output / input / increment, plus mean row L2 both sides."""
    def hf(file):
        return np.load(os.path.join(out_dir, file))[0]

    def mine(name):
        return proj if name == "proj_out" else stages[name]

    tab = {}
    for name, cols, hf_file, in_file in stage_names():
        o_mine, o_hf = mine(name), hf(hf_file)
        e = {"type": stage_type(name),
             "out": [bstats(row_cos(o_mine, o_hf), lo, hi) for lo, hi in BUCKETS]
                  + [[float(row_cos(o_mine, o_hf).mean()), float(row_cos(o_mine, o_hf).min())]],
             "l2_mine": [l2(mine(name)[lo:hi]) for lo, hi in BUCKETS],
             "l2_hf": [l2(hf(hf_file)[lo:hi]) for lo, hi in BUCKETS]}
        if in_file is None:
            e["in"] = None
            e["delta"] = e["out"]
        else:
            # the stage's input, on both sides: our dict entry and HF's hidden file
            i_mine = ({"ds_out.npy": lambda: stages["ds_out"],
                       "enc_out.npy": lambda: stages["enc_out"],
                       "proj_out.npy": lambda: proj,
                       }.get(in_file, lambda: stages["layer_%02d" % (int(in_file[7:9]) - 1)])())
            i_hf = np.load(os.path.join(out_dir, in_file))[0]
            e["in"] = [bstats(row_cos(i_mine, i_hf), lo, hi) for lo, hi in BUCKETS]
            # The projector changes width ([T,768] -> [T,2048]), so "what this stage
            # introduces" (out - in) is not defined there; skip it rather than crash.
            if o_mine.shape == i_mine.shape == o_hf.shape == i_hf.shape:
                e["delta"] = [bstats(row_cos(o_mine - i_mine, o_hf - i_hf), lo, hi)
                              for lo, hi in BUCKETS]
            else:
                e["delta"] = None
        tab[name] = e
    return tab


def print_table(tab, key, title):
    hdr = "%-9s %-4s " % ("stage", "typ") + " ".join("%13s" % ("%d-%d" % (lo, hi - 1))
                                                     for lo, hi in BUCKETS) + "  %10s" % "all-375"
    print("\n== %s ==\n%s" % (title, hdr))
    for name, _, _, _ in stage_names():
        e = tab[name]
        vals = e[key]
        if vals is None:
            print("%-9s %-4s  (input is the mel)" % (name, e["type"]))
            continue
        print("%-9s %-4s " % (name, e["type"]) + " ".join("%13.6f" % v[0] for v in vals)
              + "  %10.6f" % vals[-1][0])


def first_drop(tab):
    for name, _, _, _ in stage_names():
        if tab[name]["out"][4][0] < DROP or tab[name]["out"][5][0] < DROP:
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "ref30"))
    ap.add_argument("--hf-device", default="cpu")
    ap.add_argument("--skip-numpy", action="store_true", help="dump HF only")
    ap.add_argument("--skip-hf", action="store_true",
                    help="reuse the HF dumps already in --out (written by an earlier run)")
    ap.add_argument("--window-layers", default=None,
                    help="comma-separated attention layer indices to window; the rest stay "
                         "full causal. With --window-test this tests a per-layer schedule, "
                         "which is how the section 39 finding (HF windows the sliding "
                         "layers at 32 and leaves the last layer full) is confirmed.")
    ap.add_argument("--window-test", type=int, default=32,
                    help="re-run the spec with a W-token attention window (0 = skip)")
    ap.add_argument("--no-window-test", dest="window_test", action="store_const", const=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    feats, mask, meta = mel_30s()
    print("[mel] %s: %d samples (%.3f s) -> %s, mask ones %d, %.1fs, fixture sha %s"
          % (meta["feature_extractor"], meta["clip_samples"], meta["clip_seconds"],
             meta["mel_shape"], meta["feature_attention_mask_ones"], meta["mel_seconds"],
             meta["fixture_sha256"][:12]))
    if args.skip_hf:
        diag = json.load(open(os.path.join(args.out, "hf_meta.json")))
        meta = json.load(open(os.path.join(args.out, "mel_meta.json")))
    else:
        import torch

        del torch
        diag = hf_dump(feats, mask, args.out, args.hf_device)
        json.dump(diag, open(os.path.join(args.out, "hf_meta.json"), "w"), indent=1)
        json.dump(meta, open(os.path.join(args.out, "mel_meta.json"), "w"), indent=1)
    print("[hf] %s fp32 %s  load %.1fs canonical %.1fs  dump==audio_encoder() max|d| %.1e  "
          "proj_out==get_audio_features() max|d| %.1e"
          % (diag["hf_device"], diag["hf_attn_implementation"], diag["hf_load_seconds"],
             diag["hf_canonical_seconds"], diag["hf_canonical_forward_max_abs_diff"],
             diag["hf_get_audio_features_max_abs_diff"]))
    print("[hf] layer class %s attn %s mlp %s (inter %d) conv %s  layer_types=%s conv_L=%d "
          "sandwich=%s norm_eps=%g head=%d"
          % (diag["hf_lfm_class"], diag["hf_attn_class"], diag["hf_mlp_class"],
             diag["hf_mlp_intermediate_size"], diag["hf_conv_class"],
             "".join("C" if t == "conv" else "A" for t in diag["hf_layer_types"]),
             diag["hf_conv_L_cache"], diag["hf_use_sandwich_norm"], diag["hf_norm_eps"],
             diag["hf_head_dim"]))
    print("[hf] sliding_window: encoder %d, encoder.lfm.config %d (set by the encoder forward)"
          % (diag["hf_encoder_sliding_window"], diag["hf_lfm_sliding_window"]))
    print("[hf] mask keys per query row (rows 0,1,2,5,30,31,32,33,100,200,374): %s  min %s max %s"
          % (diag["hf_mask_keys_per_row"], diag["hf_mask_keys_min"], diag["hf_mask_keys_max"]))
    print("[hf] rope keys=%s theta=%s partial=%s inv_freq len/nonzero/max/min=%s/%s/%s/%s "
          "max|cos-1|=%s max|sin|=%s"
          % (diag["hf_rope_parameters_keys"], diag["hf_rope_theta"],
             diag["hf_partial_rotary_factor"], diag.get("hf_rope_inv_freq_len"),
             diag.get("hf_rope_inv_freq_nonzero"), diag.get("hf_rope_inv_freq_max"),
             diag.get("hf_rope_inv_freq_min"),
             diag["hf_rope_max_cos_minus_1"], diag["hf_rope_max_abs_sin"]))
    print("[hf] attention output gate: present=%s (layer %d) max|g-1|=%s min=%s mean=%s"
          % (diag["hf_attn_gate_present"], diag["hf_attn_gate_first_layer"],
             diag.get("hf_attn_gate_max_off_1"), diag.get("hf_attn_gate_min"),
             diag.get("hf_attn_gate_mean")))

    # ---- path control: is dump_encoder.py's stage-call path HF's encoder forward?
    print("\n== PATH CONTROL: dump_encoder.py's stage calls vs the canonical encoder forward ==")
    hdr = "%-9s %-4s " % ("stage", "typ") + " ".join("%13s" % ("%d-%d" % (lo, hi - 1))
                                                     for lo, hi in BUCKETS) + "  %10s" % "all-375"
    print(hdr)
    path_ctrl = {}
    for name, cols, hf_file, in_file in stage_names():
        a = np.load(os.path.join(args.out, "dumppath", hf_file))[0]
        b = np.load(os.path.join(args.out, hf_file))[0]
        c = row_cos(a, b)
        path_ctrl[name] = [bstats(c, lo, hi) for lo, hi in BUCKETS] + \
                          [[float(c.mean()), float(c.min())]]
        print("%-9s %-4s " % (name, stage_type(name)) + " ".join("%13.6f" % v[0] for v in path_ctrl[name])
              + "  %10.6f" % path_ctrl[name][-1][0])

    if args.skip_numpy:
        if args.json:
            json.dump({"meta": meta, "hf": diag, "path_control": path_ctrl},
                      open(args.json, "w"), indent=1)
        return 0

    # ---- our side: the §22 numpy fp32 spec, on HF's own weights, same mel
    import enc_numpy as E
    E.REF = args.out
    feat = np.load(os.path.join(args.out, "input.npy"))[0].astype(np.float32)
    assert feat.shape == (128, MEL_FRAMES_30S), feat.shape
    t0 = time.time()
    st = spec_stages(E, feat, None)
    proj = E.projector(st["enc_out"])
    print("\n[numpy] spec (full causal) ran %.1fs" % (time.time() - t0))
    tab = table_against_hf(st, args.out, proj)
    print_table(tab, "out", "PER-ROW COSINE vs HF (canonical), stage OUTPUT - the §22 spec as written")
    print_table(tab, "in", "PER-ROW COSINE vs HF, stage INPUT (what the stage inherits)")
    print_table(tab, "delta", "PER-ROW COSINE vs HF, stage INCREMENT out-in (what the stage introduces)")
    print("\n== mean row L2, ours / HF ==")
    print("%-9s %-4s " % ("stage", "typ") + " ".join("%19s" % ("%d-%d" % (lo, hi - 1))
                                                     for lo, hi in BUCKETS))
    for name, _, _, _ in stage_names():
        e = tab[name]
        print("%-9s %-4s " % (name, e["type"])
              + " ".join("%9.4f/%-9.4f" % (m, h) for m, h in zip(e["l2_mine"], e["l2_hf"])))
    first = first_drop(tab)

    tab_w = None
    first_w = None
    if args.window_test:
        t0 = time.time()
        if getattr(args, "window_layers", None):
            wl = {int(x): args.window_test for x in args.window_layers.split(",")
                  if x.strip() != ""}
            print("[window schedule] %s at W=%d, the rest full causal"
                  % (sorted(wl), args.window_test))
            st_w = spec_stages(E, feat, wl)
        else:
            st_w = spec_stages(E, feat, args.window_test)
        proj_w = E.projector(st_w["enc_out"])
        print("\n[numpy] spec with a %d-token attention window ran %.1fs"
              % (args.window_test, time.time() - t0))
        tab_w = table_against_hf(st_w, args.out, proj_w)
        print_table(tab_w, "out", "PER-ROW COSINE vs HF (canonical), stage OUTPUT - spec with a %d-token window"
                    % args.window_test)
        print_table(tab_w, "delta", "PER-ROW COSINE vs HF, stage INCREMENT - spec with a %d-token window"
                    % args.window_test)
        first_w = first_drop(tab_w)

    print("\n== VERDICT ==")
    for label, t, f in (("spec as written (full causal)", tab, first),
                        ("spec with a %d-token window" % args.window_test, tab_w, first_w)):
        if t is None:
            continue
        if f is None:
            print("%-34s: no stage below cos %.3f at rows 200+ (min over stages: %s)"
                  % (label, DROP, ", ".join(
                      "%s %.6f" % (n, t[n]["out"][5][0]) for n, _, _, _ in stage_names()
                      if t[n]["out"][5][0] < 0.9999)))
        else:
            prev_idx = [n for n, _, _, _ in stage_names()].index(f) - 1
            prev = [n for n, _, _, _ in stage_names()][prev_idx] if prev_idx >= 0 else None
            print("%-34s: FIRST stage with cos(rows 200+) < %.3f = %s (type %s)"
                  % (label, DROP, f, t[f]["type"]))
            if prev:
                print("   previous stage %-9s cos 200-299 %.6f  300-374 %.6f"
                      % (prev, t[prev]["out"][4][0], t[prev]["out"][5][0]))
            print("   %-9s cos 200-299 %.6f  300-374 %.6f | inherited-in %.6f | increment %.6f"
                  % (f, t[f]["out"][4][0], t[f]["out"][5][0],
                     t[f]["in"][4][0] if t[f]["in"] else float("nan"),
                     t[f]["delta"][4][0]))

    if args.json:
        json.dump({"meta": meta, "hf": diag, "path_control": path_ctrl,
                   "spec_full_causal": tab, "spec_window": tab_w,
                   "window": args.window_test or None,
                   "buckets": BUCKETS, "drop_threshold": DROP,
                   "first_stage_below": first, "first_stage_below_window": first_w},
                  open(args.json, "w"), indent=1)
        print("[json] %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
