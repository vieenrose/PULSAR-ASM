#!/usr/bin/env python3
"""mk_speech_zh.py -- Chinese speech fixture (>=60s) for the S2T gate."""
import argparse, hashlib, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_WAV = os.path.join(HERE, "speech", "gate_60s_zh_16k_mono.wav")
OUT_MP3 = os.path.join(HERE, "speech", "gate_60s_zh.mp3")
MANIFEST = os.path.join(HERE, "speech", "manifest_zh.json")
RATE, CHAN, MIN_SEC = 16000, 1, 60.0

TRANSCRIPT = (
    "气象部门发布了本季度的季节性展望，数据相当不寻常。北部山谷地区三月份的平均气温为"
    "3.4摄氏度，比三十年均值低了约2.1度，海拔一千二百米的气象站积雪深度达到41厘米，"
    "这是自2014年以来最深的记录。东部平原三月份总降雨量为118毫米，约为正常值的76%，"
    "米尔布鲁克的河流水位计在二十二日记录了每秒62立方米的峰值流量，到月底降至每秒24立方米。\n\n"
    "市政工程师提醒大家不要恐慌。4月1日的水库蓄水量为容量的71%，而2023年同期为83%，"
    "供水模型假设每日需求为4.1亿升，增加了6%。他估计，如果连续三个月每月降雨量低于40毫米，"
    "自愿限水措施将在6月20日左右开始，并要求家庭安装水表，因为一个漏水的水箱每天浪费250升水，"
    "而没有人会注意到一个漏水的水箱。\n\n"
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
        ok = sha256(OUT_WAV) == m["wav_sha256"]
        print("speech fixture %s (%s)" % ("intact" if ok else "CHANGED", m["wav_sha256"][:16]))
        sys.exit(0 if ok else 1)

    os.makedirs(os.path.dirname(OUT_WAV), exist_ok=True)
    from gtts import gTTS
    gtts_kw = dict(lang="zh", tld="com", slow=False)
    gTTS(TRANSCRIPT, **gtts_kw).save(OUT_MP3)
    import gtts
    print("gTTS %s %s -> %s (%d bytes)" % (gtts.__version__, gtts_kw, OUT_MP3, os.path.getsize(OUT_MP3)))

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", OUT_MP3,
           "-acodec", "pcm_s16le", "-ar", str(RATE), "-ac", str(CHAN),
           "-dither_method", "rectangular", OUT_WAV]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit("ffmpeg failed: " + r.stderr)

    import numpy as np
    x = np.fromfile(OUT_WAV, dtype=np.uint8)
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
    rms = float(np.sqrt((pcm ** 2).mean()))
    peak = float(np.abs(pcm).max())
    print("WAV: %s Hz mono pcm_s16le, %d samples, %.3f s, rms %.4f" % (RATE, samples, dur, rms))
    if dur < MIN_SEC:
        sys.exit("clip is %.1f s < the gate's %.0f s minimum -- lengthen TRANSCRIPT" % (dur, MIN_SEC))

    words = len(TRANSCRIPT)
    manifest = {
        "wav_sha256": sha256(OUT_WAV),
        "mp3_sha256": sha256(OUT_MP3),
        "rate": RATE, "channels": CHAN, "bits": 16,
        "samples": samples, "duration_s": dur,
        "kind": "gTTS zh-CN speech, resampled to 16 kHz mono PCM16",
        "tool": "gtts 2.5.4 + ffmpeg",
        "transcript": TRANSCRIPT,
        "transcript_sha256": hashlib.sha256(TRANSCRIPT.encode()).hexdigest(),
        "words": words,
    }
    json.dump(manifest, open(MANIFEST, "w"), indent=2, ensure_ascii=False)
    print("manifest -> %s" % MANIFEST)


if __name__ == "__main__":
    main()
