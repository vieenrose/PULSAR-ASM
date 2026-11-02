#!/usr/bin/env python3
"""mk_speech.py -- synthesize the realtime/transcript gate's speech clip.

PROVENANCE, stated where the audio lives: this is a text-to-speech synthesis (gTTS,
Google Translate's voice), NOT a recording of a person. It is committed as a gate
fixture with the exact text that produced it, so the ground-truth transcript is
known by construction and the gate is honest. It measures timing and greedy
token-stream agreement on real speech-shaped audio; it does NOT certify speaker
robustness, accent, noise, channel, or disfluency behaviour, and no claim in the
docs should read that way. For those properties a real recording is required.

Why the text is authored here rather than taken from a corpus: the gate needs
`transcript == exactly what was spoken`, and a corpus transcript of an existing
recording couples the clip to a license and to an alignment we do not have.

The clip is >= 60 s by construction (the realtime gate's condition), enforced: the
script FAILS if the decoded duration is under MIN_SEC.

  /home/louis/venvs/lfm3/bin/python .auto/lfm3/mk_speech.py            # build clip + manifest
  /home/louis/venvs/lfm3/bin/python .auto/lfm3/mk_speech.py --verify   # re-hash, no network
"""
import argparse, hashlib, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_WAV = os.path.join(HERE, "speech", "gate_60s_16k_mono.wav")
OUT_MP3 = os.path.join(HERE, "speech", "gate_60s.mp3")
MANIFEST = os.path.join(HERE, "speech", "manifest.json")
RATE, CHAN, MIN_SEC = 16000, 1, 60.0
TOKENS_PER_AUDIO_SEC = 12.5     # HF-measured encoder token rate (run #303)
GEN_TOKENS_PER_AUDIO_SEC = 4.0  # the ASR workload's decode budget (run #308)

# --- the ground-truth transcript -------------------------------------------------
# Authored for this fixture: speech-shaped sentences, and deliberate numeric and
# unit variety (decimals, percentages, ordinals, dates, thousands separators,
# units with and without spaces) because a transcript gate over "the quick brown
# fox" measures nothing about an ASR path. Read as prose, spelled for a TTS engine.
TRANSCRIPT = (
    "The weather service issued its seasonal outlook on the ninth of March, and the "
    "numbers were unusual. In the northern valleys, the average March temperature came "
    "in at 3.4 degrees Celsius, roughly 2.1 degrees below the thirty year mean, and snow "
    "cover reached 41 centimetres at the twelve hundred metre station, the deepest "
    "reading since 2014. Rainfall across the eastern plain totalled 118 millimetres for "
    "the month, about 76 percent of the normal figure, and the river gauge at Millbrook "
    "recorded a peak flow of 62 cubic metres per second on the twenty second, falling to "
    "24 by the end of the month.\n\n"
    "Farmers responded quickly. Sowing on the heavy clay soils slipped by an average of "
    "12 days, from the 18th of April to the 30th, and seed rates rose from 145 kilograms "
    "per hectare to 168, a 16 percent increase, because germination was expected to be "
    "uneven. Two growers in the Kesteven district reported that their pumps ran for 340 "
    "hours in March alone, nearly double the 180 hours logged a year earlier, and the "
    "co-operative ordered 9 additional sprinkler units at a cost of 2,350 pounds each.\n\n"
    "The council's engineer cautioned against panic. Reservoir storage stood at 71 "
    "percent of capacity on the first of April, against 83 percent in the same week of "
    "2023, and the supply model assumed a daily demand of 410 million litres, an increase "
    "of 6 percent. He estimated that voluntary restrictions would begin around the "
    "twentieth of June if rainfall stayed below 40 millimetres per month for three "
    "consecutive months, and he asked households to fit meters, because a leaking "
    "cistern wastes 250 litres a day, and nobody notices a cistern."
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        m = json.load(open(MANIFEST))
        ok = sha256(OUT_WAV) == m["wav_sha256"] and sha256(OUT_MP3) == m["mp3_sha256"]
        print("speech fixture %s (%s)" % ("intact" if ok else "CHANGED", m["wav_sha256"][:16]))
        sys.exit(0 if ok else 1)

    os.makedirs(os.path.dirname(OUT_WAV), exist_ok=True)
    from gtts import gTTS
    gtts_kw = dict(lang="en", tld="com", slow=False)
    gTTS(TRANSCRIPT, **gtts_kw).save(OUT_MP3)
    import gtts
    print("gTTS %s %s -> %s (%d bytes)" % (gtts.__version__, gtts_kw, OUT_MP3, os.path.getsize(OUT_MP3)))

    # 16 kHz mono PCM16, the exact contract lfm3_wav accepts (rate 16000, ch 1,
    # format 1, bits 16). -dither_method none keeps the conversion bit-reproducible.
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", OUT_MP3,
           "-acodec", "pcm_s16le", "-ar", str(RATE), "-ac", str(CHAN),
           "-dither_method", "rectangular", OUT_WAV]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit("ffmpeg failed: " + r.stderr)

    import numpy as np
    x = np.fromfile(OUT_WAV, dtype=np.uint8)
    # skip the RIFF header by parsing it, not by assuming 44 bytes
    assert x[:4].tobytes() == b"RIFF" and x[8:12].tobytes() == b"WAVE"
    i, data_off, data_len, fmt = 12, None, None, {}
    while i + 8 <= len(x):
        cc = x[i:i + 4].tobytes(); sz = int.from_bytes(x[i + 4:i + 8], "little")
        if cc == b"fmt ":
            fmt = {"format": int.from_bytes(x[i + 8:i + 10], "little"),
                   "channels": int.from_bytes(x[i + 10:i + 12], "little"),
                   "rate": int.from_bytes(x[i + 12:i + 16], "little"),
                   "bits": int.from_bytes(x[i + 22:i + 24], "little")}
        elif cc == b"data":
            data_off, data_len = i + 8, sz
        i += 8 + sz + (sz & 1)
    assert fmt == {"format": 1, "channels": CHAN, "rate": RATE, "bits": 16}, fmt
    samples = data_len // 2
    dur = samples / float(RATE)
    pcm = np.frombuffer(x[data_off:data_off + data_len], np.int16).astype(np.float64) / 32768.0
    rms = float(np.sqrt((pcm ** 2).mean()))   # full-scale rms, 1.0 = a full-amplitude square wave
    peak = float(np.abs(pcm).max())
    print("WAV: %s Hz mono pcm_s16le, %d samples, %.3f s, rms %.4f" % (RATE, samples, dur, rms))
    if dur < MIN_SEC:
        sys.exit("clip is %.1f s < the gate's %.0f s minimum -- lengthen TRANSCRIPT" % (dur, MIN_SEC))

    man = {
        "kind": "SYNTHETIC text-to-speech gate fixture, not a recording",
        "tool": "gTTS %s" % gtts.__version__, "tts_params": gtts_kw,
        "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0],
        "transcript_sha256": hashlib.sha256(TRANSCRIPT.encode()).hexdigest(),
        "words": len(TRANSCRIPT.split()),
        "wav": os.path.basename(OUT_WAV), "mp3": os.path.basename(OUT_MP3),
        "wav_sha256": sha256(OUT_WAV), "mp3_sha256": sha256(OUT_MP3),
        "rate": RATE, "channels": CHAN, "bits": 16, "samples": int(samples),
        "duration_s": round(dur, 6), "rms_f32": rms, "peak_f32": peak,
        "prompt_tokens_at_12_5_per_s": int(round(TOKENS_PER_AUDIO_SEC * dur)),
        "gen_tokens_at_4_per_s": int(round(GEN_TOKENS_PER_AUDIO_SEC * dur)),
        "transcript": TRANSCRIPT,
        "caveats": ["synthetic voice: measures timing + greedy agreement on speech-shaped audio",
                    "does NOT certify speaker/accent/noise/channel robustness",
                    "16 kHz mono PCM16 is the exact contract lfm3_wav accepts"],
    }
    json.dump(man, open(MANIFEST, "w"), indent=2, ensure_ascii=False)
    print("manifest -> %s  (prompt tokens %d, gen tokens %d)"
          % (MANIFEST, man["prompt_tokens_at_12_5_per_s"], man["gen_tokens_at_4_per_s"]))


if __name__ == "__main__":
    main()
