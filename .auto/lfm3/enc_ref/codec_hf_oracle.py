"""S5 oracle: LiquidCodec DECODER reference (HF, CPU, float32) on a fixed codebook stream.

CODEC_Q8=<blob>: also evaluate HF on the same dequantised Q8 decoder weights the engine
ships (HF-as-contract, as in the depth gate); outputs land in codec_gate_q8/.

Writes .auto/lfm3/enc_ref/speech/codec_gate[_q8]/:
  codes.i32                [12, T]           codes from encoding a 24 kHz fixture
  fsq_latents.f32          [T, 48]           FSQ dequant, before project_out
  embeddings.f32           [T, 512]          from_quantizer output
  upsample.f32             [6T, 512]         nearest x6, the LFM input
  layers.json              per-layer rms + first4 (8 hooks)
  lin.f32                  [1282, 6T]        decoder.lin output (log-mag | angle)
  audio_raw.f32            [L] audio_raw.f32 / envelope.f32, pre-trim iSTFT
  audio.f32                [6T*320]          final decoded waveform (trim + /envelope)
  fixture_24k.f32          the encoded fixture (so the gate is reproducible from disk)
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "/home/louis/models/lfm3/LFM3-Audio-2B"
FIXTURE16 = os.path.join(os.path.dirname(HERE), "speech", "gate_60s_16k_mono.wav")
SECONDS = int(os.environ.get("CODEC_SECONDS", "6"))
Q8 = os.environ.get("CODEC_Q8")
OUT = os.path.join(HERE, "speech", "codec_gate_q8" if Q8 else "codec_gate")


def _remote_codec_module():
    """The codec ships as remote code in the model repo; transformers caches it as a package
    whose directory starts with a digit, so import it via importlib with a clean name."""
    import glob
    import importlib
    import importlib.util
    cands = sorted(glob.glob("/home/louis/.cache/huggingface/modules/transformers_modules/"
                             "LFM3_hyphen_Audio_hyphen_2B/*/modeling_liquid_codec.py"),
                   key=os.path.getmtime)
    if not cands:
        raise SystemExit("no cached modeling_liquid_codec.py; run any HF load of the model first")
    base = os.path.dirname(cands[-1])
    pkg = "lfm3_remote_codec"
    if pkg not in sys.modules:
        spec = importlib.util.spec_from_file_location(pkg, os.path.join(base, "__init__.py"),
                                                      submodule_search_locations=[base])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[pkg] = mod
        spec.loader.exec_module(mod)
    return importlib.import_module(pkg + ".modeling_liquid_codec")


def load_codec():
    from transformers import AutoConfig, AutoModel
    m = _remote_codec_module()
    AutoConfig.register("liquid_codec", m.LiquidCodecConfig, exist_ok=True)
    AutoModel.register(m.LiquidCodecConfig, m.LiquidCodecModel, exist_ok=True)
    return AutoModel.from_pretrained(os.path.join(MODEL, "audio_tokenizer"),
                                     dtype=torch.float32, trust_remote_code=True).eval()


def load_q8_blob(path):
    import struct
    with open(path, "rb") as f:
        assert f.read(8) == b"LFM3Q8\x01\x00", "bad magic"
        n = struct.unpack("<I", f.read(4))[0]
        hdr = json.loads(f.read(n))
        base = 12 + n
        out = {}
        for e in hdr["table"]:
            if e["dtype"] != 1:
                continue
            f.seek(base + e["off"])
            codes = np.frombuffer(f.read(e["ncodes"]), dtype=np.int8).reshape(e["ne"])
            scales = np.frombuffer(f.read(e["nscales"] * 4), dtype=np.float32).reshape(e["ne"][0], -1)
            out[e["name"]] = (codes.astype(np.float32) * np.repeat(scales, 32, axis=1)).astype(np.float32)
    return out


def apply_q8(codec, q8):
    dec = codec.decoder
    with torch.no_grad():
        dec.lin.weight.copy_(torch.from_numpy(q8["decoder.lin.weight"]))
        codec.from_quantizer.weight.copy_(torch.from_numpy(q8["from_quantizer.weight"]))
        if "quantizer.project_out.weight" in q8:
            codec.quantizer.project_out.weight.copy_(torch.from_numpy(q8["quantizer.project_out.weight"]))
        for li, layer in enumerate(dec.lfm.layers):
            p = f"decoder.lfm.layers.{li}"
            attn = getattr(layer, "self_attn", None)
            conv = getattr(layer, "conv", None)
            mods = [("feed_forward.w1", layer.feed_forward.w1), ("feed_forward.w2", layer.feed_forward.w2),
                    ("feed_forward.w3", layer.feed_forward.w3)]
            for nm in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"):
                mods.append((f"self_attn.{nm}", getattr(attn, nm, None)))
            for nm in ("in_proj", "out_proj"):
                mods.append((f"conv.{nm}", getattr(conv, nm, None)))
            for suffix, mod in mods:
                name = f"{p}.{suffix}.weight"
                if mod is not None and name in q8:      # small widths (48) stay f32 in the blob
                    mod.weight.copy_(torch.from_numpy(q8[name]))
    print("codec decoder weights replaced by the Q8 dequant", flush=True)


def main():
    import wave
    os.makedirs(OUT, exist_ok=True)
    w = wave.open(FIXTURE16)
    x16 = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    x16 = x16[: 16000 * SECONDS]
    # fixture-side resample 16k -> 24k (linear in time); the engine's WAV reader never resamples
    idx = np.linspace(0, len(x16) - 1, int(len(x16) * 1.5))
    x24 = np.interp(idx, np.arange(len(x16)), x16).astype(np.float32)
    x24.tofile(os.path.join(OUT, "fixture_24k.f32"))

    codec = load_codec()
    cfg = codec.config
    if Q8:
        apply_q8(codec, load_q8_blob(Q8))

    with torch.no_grad():
        codes = codec.encode(torch.from_numpy(x24)[None]).audio_codes          # [1,12,T]
    print(f"codes {tuple(codes.shape)} range [{int(codes.min())},{int(codes.max())}]", flush=True)
    codes[0].numpy().astype(np.int32).tofile(os.path.join(OUT, "codes.i32"))

    # the decode chain, stage by stage (mirrors LiquidCodecModel.decode with no cache)
    dec = codec.decoder
    with torch.no_grad():
        lat = codec.quantizer.decode(codes)                                     # [1,256,T]
        emb = codec.from_quantizer(lat.mT.to(codec.from_quantizer.weight.dtype))  # [1,T,512]
        torch.set_printoptions(precision=8, sci_mode=False)
        lat.mT[0].numpy().astype(np.float32).tofile(os.path.join(OUT, "fsq_latents.f32"))
        emb[0].numpy().astype(np.float32).tofile(os.path.join(OUT, "embeddings.f32"))

        ups = torch.nn.functional.interpolate(emb.mT, emb.shape[1] * cfg.stft_frame_stack,
                                              mode="nearest-exact").mT.contiguous()
        ups[0].numpy().astype(np.float32).tofile(os.path.join(OUT, "upsample.f32"))

        probes = {}
        def hook(i, mod, inp, out):
            probes[i] = dict(rms=float(out.float().pow(2).mean().sqrt()),
                             first4=[float(v) for v in out.float()[0, 0, :4]])
        hs = [dec.lfm.layers[i].register_forward_hook(lambda m, a, o, i=i: hook(i, m, a, o))
              for i in range(len(dec.lfm.layers))]
        dec.lfm.register_forward_hook(lambda m, a, o: hook(99, m, a, o.last_hidden_state))
        out = codec.decoder(emb)
        for h in hs:
            h.remove()
        json.dump(probes, open(os.path.join(OUT, "layers.json"), "w"), indent=1)

        hidden = dec.lfm(inputs_embeds=ups, use_cache=False).last_hidden_state
        lin = dec.lin(hidden).mT.contiguous()                                    # [1,1282,6T]
        lin[0].numpy().astype(np.float32).tofile(os.path.join(OUT, "lin.f32"))
        audio, envelope = out[0], out[1]
        audio.numpy().astype(np.float32).tofile(os.path.join(OUT, "audio_raw.f32"))
        envelope.numpy().astype(np.float32).tofile(os.path.join(OUT, "envelope.f32"))
        trim = (cfg.stft_win_length - cfg.stft_hop_length) // 2
        final = (audio[:, trim:-trim] / envelope[trim:-trim]).float()
        final[0].numpy().astype(np.float32).tofile(os.path.join(OUT, "audio.f32"))
        json.dump(dict(T=int(codes.shape[-1]), frames6=int(ups.shape[1]), n_fft=cfg.stft_n_fft,
                       hop=cfg.stft_hop_length, win=cfg.stft_win_length,
                       frame_stack=cfg.stft_frame_stack, n_q=cfg.n_q, levels=list(cfg.fsq_levels),
                       trim=trim, audio_len=int(final.shape[-1]),
                       audio_rms=float(final.pow(2).mean().sqrt()),
                       audio_peak=float(final.abs().max()), q8=bool(Q8)),
                  open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print("codec oracle written:", json.load(open(os.path.join(OUT, "manifest.json"))), flush=True)


if __name__ == "__main__":
    main()
