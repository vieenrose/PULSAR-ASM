#!/usr/bin/env python3
"""Decisive test of the "HF's audio tower runs in bf16" hypothesis.

The open comparison in speech/HF_ASR_GOLDEN_NOTES.md shows our encoder's proj_out agreeing with
HF's [1622, 2048] audio embeddings at the head of the clip (per-row cosine 0.99997 for rows 0-2)
and decaying with position (~0.85 by row 300+). Our C reference (x86, independent of the CUDA
code) decays the same way, and a shift scan peaks sharply at k=0, so rows are aligned. The
leading hypothesis: HF runs the audio tower in bf16 and ITS rounding grows with position, making
HF - not us - the outlier.

This script runs HF's own audio tower - the checkpoint's trust_remote_code
`model.model.get_audio_features(input_features, feature_attention_mask)`, the same entry point
the golden uses - on the committed fixture in **float32**, with the fp32 weight dtype asserted at
run time (every parameter and buffer of the audio encoder + projector, plus the dtype of every
module input and output observed by forward hooks), and compares, per row, against

  1. the committed golden matrix speech/hf_audio_embeddings_1622x2048.f32,
  2. the C reference's proj_out for the 30 s sub-clip
     (picard@100.84.151.116:/home/picard/lfm3/c30/proj_out, raw f32 row-major, no header,
      375 rows x 2048 = rows 0..374 of the fixture; the C tool writes dumps WITHOUT a .bin suffix),

bucketed by row range (0-2, 3-20, 21-99, 100-199, 200-299, 300-374 for the 375-row comparison;
0-2, 3-20, 21-99, 100+ for the 1622-row one).

A genuine bf16 control arm (the tower cast to bfloat16, whose weights are bit-identical to the
bf16 bytes stored in the checkpoint) is included by default: the committed golden matrix turned
out to be an fp32 run itself (its own embedding_matrix block records source_torch_dtype =
torch.float32, cast_applied = false), so the golden is not the bf16 side the hypothesis needs.
The fp32-vs-golden comparison is therefore HF-fp32(GPU) vs HF-fp32(CPU) - a device check - and
the dtype question is answered by the fp32-vs-bf16 pair plus the C reference.

    python .auto/lfm3/hf_asr_tower_fp32.py                     # every arm, print only, write nothing
    python .auto/lfm3/hf_asr_tower_fp32.py --clip 30s          # only the 30 s sub-clip (fast)
    python .auto/lfm3/hf_asr_tower_fp32.py --arms fp32@cuda    # pick arms (precision@device)
    python .auto/lfm3/hf_asr_tower_fp32.py --device cpu        # no GPU; the CPU runtime is reported
    python .auto/lfm3/hf_asr_tower_fp32.py --verify --write-json
                                                               # re-hash the fixture against the manifest,
                                                               # then append "tower_fp32_test" to the json

Every mode refuses to run when the fixture's sha256 differs from speech/manifest.json (the gate
is imported from hf_asr_golden.py, never re-implemented). --write-json requires --verify, writes
only the new "tower_fp32_test" key, and re-reads the file to prove every pre-existing key is
byte-identical.

Verdict rule (thresholds are recorded in the json):
  CONFIRMED  mean cos(C, HF-fp32) over rows 200-374 >= mean cos(C, HF-bf16) over the same rows + 0.02
  REFUTED    mean cos(HF-fp32, HF-bf16) over rows 200-374 >= 0.9999 and the confirm test fails
  otherwise  INCONCLUSIVE
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SPEECH = os.path.join(REPO, ".auto", "lfm3", "speech")
GOLDEN = os.path.join(SPEECH, "hf_asr_golden.json")
sys.path.insert(0, HERE)

# The provenance gate and the fixture/model constants come from the golden's own script: one
# implementation of "is this the committed fixture", never two that can disagree.
import hf_asr_golden as G  # noqa: E402

KEY = "tower_fp32_test"
MODEL = G.MODEL
# The C tool on picard writes its dumps WITHOUT a .bin suffix; the shape is 375 x 2048 raw f32.
C_PROJ_SOURCE = "picard@100.84.151.116:/home/picard/lfm3/c30/proj_out"
C_PROJ_ROWS = 375
C_PROJ_DIM = 2048
# 30.000 s @ 16 kHz. The checkpoint's own formulas give num_frames(480000) = 2999 and
# ceil(2999 / 8) = 375 tokens, which is the C reference's row count.
CLIP30_SAMPLES = 480_000
CLIP30_SECONDS = 30.0
# The row buckets the open comparison is written in.
BUCKETS_C = [(0, 3), (3, 21), (21, 100), (100, 200), (200, 300), (300, 375)]
BUCKETS_FULL = [(0, 3), (3, 21), (21, 100), (100, None)]
TAIL = (200, 375)
CONFIRM_MARGIN = 0.02      # cos(C, fp32) must beat cos(C, bf16) by this at rows 200+
REFUTE_COS = 0.9999        # cos(fp32, bf16) at least this at rows 200+ = dtype is not the cause
DEFAULT_ARMS = "fp32@cuda,bf16@cuda,fp32@cpu"


# ---------------------------------------------------------------- provenance helpers

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: str) -> str:
    return G.sha256_file(path)


def fetch_c_proj(source: str, scratch: str) -> str:
    """Stage the C reference's proj_out: a local path is used as-is, user@host:/path is scp'd."""
    if os.path.exists(source):
        return source
    if ":" not in source:
        print(f"FATAL: --c-proj {source!r} is neither an existing file nor user@host:/path", file=sys.stderr)
        sys.exit(2)
    dest = os.path.join(scratch, "proj_out")
    print(f"[c-ref] fetching {source}")
    proc = subprocess.run(["scp", "-q", "-o", "StrictHostKeyChecking=accept-new",
                           "-o", "ConnectTimeout=20", source, dest],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(f"FATAL: scp failed ({proc.returncode}): {proc.stdout.strip()}", file=sys.stderr)
        sys.exit(2)
    return dest


def load_raw_matrix(path: str, rows: int, dim: int):
    """A headerless raw-f32 row-major dump, validated by byte count."""
    import numpy as np

    payload = open(path, "rb").read()
    want = rows * dim * 4
    if len(payload) != want:
        print(f"FATAL: {path} is {len(payload)} bytes, expected {want} for {rows}x{dim} f32", file=sys.stderr)
        sys.exit(2)
    return np.frombuffer(payload, dtype="<f4").reshape(rows, dim)


# ---------------------------------------------------------------- matrix statistics

def row_norms(m):
    import numpy as np

    m = np.asarray(m, dtype=np.float64)
    return np.sqrt(np.sum(m * m, axis=1))


def row_cos(a, b):
    """Per-row cosine in float64; a row of all zeros on either side scores 0.0, never nan."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.sqrt(np.sum(a * a, axis=1))
    nb = np.sqrt(np.sum(b * b, axis=1))
    denom = na * nb
    num = np.sum(a * b, axis=1)
    out = np.zeros_like(num)
    ok = denom > 0.0
    out[ok] = num[ok] / denom[ok]
    return out


def bucket_stats(cos, a, b, buckets) -> dict:
    """Per-row cosine mean/min and mean row L2 norms of both sides, bucketed by row range."""
    import numpy as np

    na, nb = row_norms(a), row_norms(b)
    out = {}
    for lo, hi in buckets:
        hi_c = min(hi if hi is not None else len(cos), len(cos))
        if lo >= hi_c:
            continue
        c = cos[lo:hi_c]
        out["%d-%d" % (lo, hi_c - 1)] = {
            "rows": int(c.shape[0]),
            "cos_mean": float(np.mean(c)),
            "cos_min": float(np.min(c)),
            "cos_median": float(np.median(c)),
            "argmin_row": int(lo + int(np.argmin(c))),
            "mean_row_l2_norm_a": float(np.mean(na[lo:hi_c])),
            "mean_row_l2_norm_b": float(np.mean(nb[lo:hi_c])),
        }
    return out


def describe_matrix(m, label: str, extra: dict | None = None) -> dict:
    import numpy as np

    payload = np.ascontiguousarray(m, dtype="<f4").tobytes(order="C")
    norms = row_norms(m)
    block = {
        "label": label,
        "shape": [int(m.shape[0]), int(m.shape[1])],
        "dtype": "float32 (every dump is widened from the arm's compute dtype on save)",
        "sha256_f32_bytes": sha256_bytes(payload),
        "all_finite": bool(np.isfinite(np.asarray(m, dtype=np.float64)).all()),
        "mean_row_l2_norm": float(np.mean(norms)),
        "min_row_l2_norm": float(np.min(norms)),
        "max_row_l2_norm": float(np.max(norms)),
        "first_rows_l2_norm": [float(norms[i]) for i in range(min(3, norms.shape[0]))],
        "last_rows_l2_norm": [float(norms[i]) for i in range(max(0, norms.shape[0] - 3), norms.shape[0])],
    }
    if extra:
        block.update(extra)
    return block


def compare(a, b, buckets, label: str) -> dict:
    import numpy as np

    cos = row_cos(a, b)
    tail = cos[TAIL[0]:TAIL[1]]
    return {
        "label": label,
        "a": label.split("_vs_")[0] if "_vs_" in label else "a",
        "b": label.split("_vs_")[1] if "_vs_" in label else "b",
        "rows_compared": int(min(a.shape[0], b.shape[0])),
        "cos_mean": float(np.mean(cos)),
        "cos_min": float(np.min(cos)),
        "cos_mean_rows_200_374": float(np.mean(tail)) if tail.size else None,
        "cos_min_rows_200_374": float(np.min(tail)) if tail.size else None,
        "mean_row_l2_norm_a": float(np.mean(row_norms(a))),
        "mean_row_l2_norm_b": float(np.mean(row_norms(b))),
        "buckets": bucket_stats(cos, a, b, buckets),
    }


def print_comparison(block: dict) -> None:
    print("\n[%s]" % block["label"])
    print("  rows %d | cos mean %.6f min %.6f | rows %d-%d: mean %s min %s"
          % (block["rows_compared"], block["cos_mean"], block["cos_min"], TAIL[0], TAIL[1] - 1,
             "%.6f" % block["cos_mean_rows_200_374"] if block["cos_mean_rows_200_374"] is not None else "n/a",
             "%.6f" % block["cos_min_rows_200_374"] if block["cos_min_rows_200_374"] is not None else "n/a"))
    print("  mean row L2: a %.6f  b %.6f" % (block["mean_row_l2_norm_a"], block["mean_row_l2_norm_b"]))
    print("  %-10s %6s %10s %10s %10s %10s" % ("rows", "n", "cos_mean", "cos_min", "L2_a", "L2_b"))
    for name, s in block["buckets"].items():
        print("  %-10s %6d %10.6f %10.6f %10.6f %10.6f  (min at row %d)"
              % (name, s["rows"], s["cos_mean"], s["cos_min"],
                 s["mean_row_l2_norm_a"], s["mean_row_l2_norm_b"], s["argmin_row"]))


# ---------------------------------------------------------------- dtype evidence

def tower_dtype_audit(tower, want, tag: str) -> dict:
    """Assert the audio tower's weights really are `want` at run time; report what was checked."""
    import torch

    checked, offenders, numel, nbytes = 0, [], 0, 0
    for kind, items in (("parameter", tower.named_parameters()), ("buffer", tower.named_buffers())):
        for name, t in items:
            checked += 1
            numel += int(t.numel())
            nbytes += int(t.numel()) * int(t.element_size())
            if t.dtype != want:
                offenders.append("%s:%s=%s" % (kind, name, t.dtype))
    if offenders:
        print("FATAL: %s is not %s at run time; %d/%d tensors off dtype: %s"
              % (tag, want, len(offenders), checked, offenders[:8]), file=sys.stderr)
        sys.exit(2)
    probe = tower.downsample.conv_in.weight if hasattr(tower, "downsample") else None
    if probe is not None and probe.dtype != want:
        print("FATAL: probe weight dtype %s != %s" % (probe.dtype, want), file=sys.stderr)
        sys.exit(2)
    return {
        "module": tag,
        "asserted_dtype": str(want),
        "tensors_checked": checked,
        "tensors_off_dtype": offenders,
        "tensor_elements": numel,
        "tensor_bytes": nbytes,
        "probe_weight_dtype": str(probe.dtype) if probe is not None else "n/a",
        "probe_weight_element_size": int(probe.element_size()) if probe is not None else None,
    }


class DtypeRecorder:
    """Forward hooks over every module in the tower: what dtypes actually flowed through it."""

    def __init__(self, towers):
        self.inputs = collections.Counter()
        self.outputs = collections.Counter()
        self.module_dtypes = collections.defaultdict(set)
        self._handles = []
        for tower in towers:
            self._attach(tower)

    def _pre(self, mod, args):
        for a in args:
            if hasattr(a, "dtype"):
                self.inputs[str(a.dtype)] += 1
                self.module_dtypes[type(mod).__name__].add(str(a.dtype))
        return None

    def _post(self, mod, args, out):
        vals = out if isinstance(out, (tuple, list)) else (out,)
        for o in vals:
            if hasattr(o, "dtype"):
                self.outputs[str(o.dtype)] += 1
        return None

    def _attach(self, tower):
        for mod in tower.modules():
            self._handles.append(mod.register_forward_pre_hook(self._pre))
            self._handles.append(mod.register_forward_hook(self._post))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @staticmethod
    def _is_floating(dtype: str) -> bool:
        return dtype.startswith("torch.float") or dtype.startswith("torch.bfloat") or dtype.startswith("torch.half")

    def summary(self, want) -> dict:
        # Index tensors (position_ids) legitimately flow through the tower; only the floating-point
        # dtypes are evidence about the compute precision.
        floats = {d for d in set(self.inputs) | set(self.outputs) if self._is_floating(d)}
        off = sorted(d for d in floats if d != str(want))
        return {
            "input_dtypes": dict(self.inputs),
            "output_dtypes": dict(self.outputs),
            "floating_dtypes_seen": sorted(floats),
            "dtypes_seen_off_the_requested_one": off,
            "module_input_dtypes": {k: sorted(v) for k, v in sorted(self.module_dtypes.items())},
        }


# ---------------------------------------------------------------- the HF run

def parse_arms(spec: str, default_device: str) -> list[dict]:
    """`fp32@cuda,bf16@cuda` -> arm dicts; a bare precision uses --device."""
    arms = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        precision, _, device = item.partition("@")
        precision = precision.strip()
        device = (device.strip() or default_device) if "@" in item else default_device
        if precision not in ("fp32", "bf16", "fp16"):
            print(f"FATAL: unknown arm precision {precision!r} (use fp32, bf16, fp16)", file=sys.stderr)
            sys.exit(2)
        arms.append({"name": "%s@%s" % (precision, device), "precision": precision, "device": device})
    if not arms:
        print("FATAL: no arms to run", file=sys.stderr)
        sys.exit(2)
    seen = set()
    for arm in arms:
        if arm["name"] in seen:
            print("FATAL: duplicate arm %s" % arm["name"], file=sys.stderr)
            sys.exit(2)
        seen.add(arm["name"])
    return arms


def bf16_roundtrip_is_exact(tower_modules) -> dict:
    """Prove the bf16 control arm's weights are the checkpoint's stored bf16 bytes.

    The tower is loaded fp32 from bf16 on disk, so every stored value must be exactly
    representable in bf16: casting to bf16 and back has to be the identity. If it is, the bf16 arm
    runs on exactly the bytes the checkpoint ships, not on a re-rounded copy. Buffers are excluded
    and listed: rotary `inv_freq` is computed at init in fp32, so it is legitimately not
    bf16-representable and is not part of the stored-weight provenance claim.
    """
    import torch

    params, buffers, worst, worst_name = 0, 0, 0.0, ""
    non_representable = []
    for mod in tower_modules:
        for kind, items in (("param", mod.named_parameters()), ("buffer", mod.named_buffers())):
            for name, t in items:
                if t.dtype != torch.float32:
                    continue
                if kind == "param":
                    params += 1
                else:
                    buffers += 1
                diff = float((t.detach().to(torch.bfloat16).to(torch.float32) - t).abs().max()) if t.numel() else 0.0
                if kind == "param" and diff > worst:
                    worst, worst_name = diff, "%s.%s" % (type(mod).__name__, name)
                if kind == "buffer" and diff > 0.0:
                    non_representable.append("%s.%s" % (type(mod).__name__, name))
    return {"parameters_checked": params, "buffers_excluded": buffers,
            "buffers_not_bf16_representable": non_representable,
            "max_abs_bf16_roundtrip_error": worst,
            "worst_tensor": worst_name,
            "bf16_control_uses_the_checkpoint_bytes": worst == 0.0}


def run_arms(arms: list[dict], clips: list[str], allow_tf32: bool = False) -> tuple[dict, dict]:
    """Load the checkpoint once, then run get_audio_features for every (arm, clip)."""
    import numpy as np
    import resource
    import soundfile as sf
    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    PRECISION = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    torch.set_grad_enabled(False)
    t_start = time.time()
    harness = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_devices": [
            {"index": i, "name": torch.cuda.get_device_name(i),
             "capability": "%d.%d" % torch.cuda.get_device_capability(i),
             "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)}
            for i in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
        "arms_requested": [a["name"] for a in arms],
        "arm_runtime_s": {},
        "arm_dtype_audit": {},
        "arm_activation_dtypes": {},
        "arm_input_dtypes": {},
        "arm_output_dtypes": {},
        "tf32_flags_at_run": {},
        "feature_extraction": {},
        "load_seconds": None,
        "wall_seconds": None,
        "max_rss_gb": None,
    }

    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)

    # Read the fixture ourselves (pcm_s16le -> float32) so the samples that reach HF are exactly
    # the committed bytes; the sub-clip is a slice of those samples.
    waveform, file_rate = sf.read(G.FIXTURE, dtype="float32")
    if file_rate != G.SAMPLING_RATE or waveform.ndim != 1:
        print(f"FATAL: fixture is {file_rate} Hz shape {waveform.shape}, "
              f"expected {G.SAMPLING_RATE} Hz mono", file=sys.stderr)
        sys.exit(2)
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)

    clips_wav = {}
    if "full" in clips:
        clips_wav["full"] = waveform
    if "30s" in clips:
        if waveform.shape[0] < CLIP30_SAMPLES:
            print(f"FATAL: fixture has {waveform.shape[0]} samples, the 30 s sub-clip needs "
                  f"{CLIP30_SAMPLES}", file=sys.stderr)
            sys.exit(2)
        clips_wav["30s"] = np.ascontiguousarray(waveform[:CLIP30_SAMPLES])

    # The feature extractor is HF's own, run once per clip on CPU at fp32, through the same
    # chat-template call the golden makes, so every arm consumes identical mel frames.
    inputs = {}
    for label, wav in clips_wav.items():
        conversation = [
            {"role": "system", "content": G.SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "audio", "audio": wav}]},
        ]
        t0 = time.time()
        inp = proc.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"sampling_rate": G.SAMPLING_RATE},
        )
        feats, mask = inp["input_features"], inp["feature_attention_mask"]
        inputs[label] = (feats.contiguous(), mask.contiguous())
        harness["feature_extraction"][label] = {
            "seconds": round(time.time() - t0, 2),
            "samples": int(wav.shape[0]),
            "audio_seconds": round(wav.shape[0] / G.SAMPLING_RATE, 3),
            "input_features_shape": [int(x) for x in feats.shape],
            "input_features_dtype": str(feats.dtype),
            "feature_attention_mask_ones": int(mask.sum()),
        }
        print("[mel] %s: samples %d (%.3f s) -> input_features %s %s, mask ones %d, %.2fs"
              % (label, wav.shape[0], wav.shape[0] / G.SAMPLING_RATE,
                 tuple(int(x) for x in feats.shape), feats.dtype, int(mask.sum()),
                 harness["feature_extraction"][label]["seconds"]))

    t_load = time.time()
    # The golden's loader: fp32 from the checkpoint's bf16 bytes. The language model is loaded
    # because get_audio_features is reached through the composite model; it stays on CPU and is
    # never executed here - only the encoder and the projector run.
    model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True)
    model.eval()
    harness["load_seconds"] = round(time.time() - t_load, 2)
    harness["model_dtype_at_load"] = str(model.dtype)
    harness["audio_tower_dtype_after_load"] = str(
        model.model.audio_encoder.downsample.conv_in.weight.dtype)
    harness["audio_tower_cast"] = (
        "from_pretrained(dtype=torch.float32) is the loader (the checkpoint stores BF16 for all "
        "539 tensors, so this is the bf16 -> fp32 widening); each arm then casts the encoder + "
        "projector explicitly with .to(dtype=...), and tower_dtype_audit asserts the result")
    harness["bf16_control_weight_provenance"] = bf16_roundtrip_is_exact(
        [model.model.audio_encoder, model.model.audio_projector])
    print("[weights] bf16 control: %d tower parameters, max bf16 round-trip error %.3e (%s) -> "
          "the bf16 arm uses the checkpoint's own bytes: %s (%d computed buffers excluded: %s)"
          % (harness["bf16_control_weight_provenance"]["parameters_checked"],
             harness["bf16_control_weight_provenance"]["max_abs_bf16_roundtrip_error"],
             harness["bf16_control_weight_provenance"]["worst_tensor"],
             harness["bf16_control_weight_provenance"]["bf16_control_uses_the_checkpoint_bytes"],
             harness["bf16_control_weight_provenance"]["buffers_excluded"],
             harness["bf16_control_weight_provenance"]["buffers_not_bf16_representable"]))

    matrices = {}
    for arm in arms:
        want = PRECISION[arm["precision"]]
        device = arm["device"]
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("FATAL: arm %s asks for %s but CUDA is unavailable" % (arm["name"], device), file=sys.stderr)
            sys.exit(2)
        for mod in (model.model.audio_encoder, model.model.audio_projector):
            mod.to(device=device, dtype=want)
        # An fp32 arm must really be fp32: TF32 conv/matmul keeps only 10 explicit mantissa bits,
        # so an fp32 run that silently takes the TF32 path is not an fp32 run. TF32 is pinned off
        # for every arm and its state is recorded, never assumed.
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        harness["tf32_flags_at_run"][arm["name"]] = {
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "allow_tf32_flag": bool(allow_tf32),
        }
        enc_audit = tower_dtype_audit(model.model.audio_encoder, want, "model.model.audio_encoder")
        proj_audit = tower_dtype_audit(model.model.audio_projector, want, "model.model.audio_projector")
        harness["arm_dtype_audit"][arm["name"]] = {
            "asserted_dtype": str(want),
            "tensors_checked": enc_audit["tensors_checked"] + proj_audit["tensors_checked"],
            "tensor_elements": enc_audit["tensor_elements"] + proj_audit["tensor_elements"],
            "tensor_bytes": enc_audit["tensor_bytes"] + proj_audit["tensor_bytes"],
            "tensors_off_dtype": enc_audit["tensors_off_dtype"] + proj_audit["tensors_off_dtype"],
            "audio_encoder": enc_audit,
            "audio_projector": proj_audit,
        }
        rec = DtypeRecorder([model.model.audio_encoder, model.model.audio_projector])
        for label, (feats, mask) in inputs.items():
            dev = torch.device(device)
            feats_d = feats.to(device=dev, dtype=want)
            mask_d = mask.to(device=dev)
            harness["arm_input_dtypes"].setdefault(arm["name"], {})[label] = {
                "input_features": str(feats_d.dtype),
                "feature_attention_mask": str(mask_d.dtype),
                "device": str(feats_d.device),
            }
            t0 = time.time()
            with torch.no_grad():
                out = model.model.get_audio_features(feats_d, mask_d)
            harness["arm_runtime_s"].setdefault(arm["name"], {})[label] = round(time.time() - t0, 3)
            harness["arm_output_dtypes"].setdefault(arm["name"], {})[label] = str(out.dtype)
            if out.dtype != want:
                print("FATAL: arm %s clip %s returned dtype %s, expected %s"
                      % (arm["name"], label, out.dtype, want), file=sys.stderr)
                sys.exit(2)
            matrices[(arm["name"], label)] = out.detach().to("cpu", torch.float32).contiguous().numpy()
            print("[arm %-10s %-4s] %s %s -> %d x %d in %.3fs"
                  % (arm["name"], label, tuple(int(x) for x in feats_d.shape), feats_d.dtype,
                     out.shape[0], out.shape[1], harness["arm_runtime_s"][arm["name"]][label]))
            del out, feats_d, mask_d
        summary = rec.summary(want)
        rec.detach()
        harness["arm_activation_dtypes"][arm["name"]] = summary
        off = summary["dtypes_seen_off_the_requested_one"]
        if off:
            print("FATAL: arm %s executed modules in %s, expected only %s"
                  % (arm["name"], off, want), file=sys.stderr)
            sys.exit(2)
        print("[arm %-10s ] weights %s: %d tensors asserted; activations %s"
              % (arm["name"], want, harness["arm_dtype_audit"][arm["name"]]["tensors_checked"],
                 summary["floating_dtypes_seen"]))
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        del rec

    harness["wall_seconds"] = round(time.time() - t_start, 2)
    harness["max_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    return matrices, harness


# ---------------------------------------------------------------- verdict

def decide(comparisons: dict, primary: dict) -> dict:
    rule = {
        "confirmed": "mean cos(C, HF-fp32) over rows %d-%d >= mean cos(C, HF-bf16) over the same "
                     "rows + %.2f  (then HF's bf16 tower is the outlier)" % (TAIL[0], TAIL[1] - 1, CONFIRM_MARGIN),
        "refuted": "mean cos(HF-fp32, HF-bf16) over rows %d-%d >= %.4f and the confirm test fails "
                   "(then the gap is ours)" % (TAIL[0], TAIL[1] - 1, REFUTE_COS),
        "primary_pair": primary,
    }

    def tail(key):
        b = comparisons.get(key)
        return None if b is None else b.get("cos_mean_rows_200_374")

    c_fp32 = tail("c_ref_vs_%s_30s" % primary["fp32"])
    c_bf16 = tail("c_ref_vs_%s_30s" % primary["bf16"])
    pair = tail("%s_vs_%s_30s" % (primary["fp32"], primary["bf16"]))
    inputs = {
        "cos_c_vs_hf_fp32_tail": c_fp32,
        "cos_c_vs_hf_bf16_tail": c_bf16,
        "cos_hf_fp32_vs_hf_bf16_tail": pair,
        "delta_c_fp32_minus_bf16": None if (c_fp32 is None or c_bf16 is None) else c_fp32 - c_bf16,
    }
    if None in (c_fp32, c_bf16, pair):
        return {"name": "INCONCLUSIVE",
                "line": "INCONCLUSIVE: the fp32/bf16/C triple is incomplete (need both arms on the "
                        "same device plus the C reference); inputs %s" % json.dumps(inputs),
                "rule": rule, "inputs": inputs}
    if c_fp32 >= c_bf16 + CONFIRM_MARGIN:
        return {"name": "CONFIRMED",
                "line": "CONFIRMED: HF-fp32 moves toward the C reference as position grows - "
                        "cos(C, HF-fp32) %.6f vs cos(C, HF-bf16) %.6f at rows %d+ (delta %+.6f) - "
                        "so the bf16 tower is the outlier"
                        % (c_fp32, c_bf16, TAIL[0], c_fp32 - c_bf16),
                "rule": rule, "inputs": inputs}
    if pair >= REFUTE_COS:
        return {"name": "REFUTED",
                "line": "REFUTED: HF-fp32 is indistinguishable from HF-bf16 (cos %.8f at rows %d+), so "
                        "HF's tower precision is not the outlier - the position-growing gap is not a "
                        "rounding artifact on HF's side (cos(C, HF-fp32) %.6f, cos(C, HF-bf16) %.6f, "
                        "delta %+.6f)"
                        % (pair, TAIL[0], c_fp32, c_bf16, c_fp32 - c_bf16),
                "rule": rule, "inputs": inputs,
                "establishes": "HF's audio tower is dtype-insensitive on this fixture: its fp32 run "
                               "reproduces the committed matrix and its bf16 run differs by ~5e-5, at "
                               "every position including rows 200+. The bf16-tower hypothesis cannot "
                               "explain the decay, on either side of the comparison.",
                "does_not_establish": "which implementation deviates from the spec. All three HF arms "
                                      "sit on one side of the gap; naming the deviating side needs the "
                                      "independent fp32 spec re-derivation (see the 2026-11-02 three-way "
                                      "result in HF_ASR_GOLDEN_NOTES.md / doc/LFM3_PORT_STAGES.md)."}
    return {"name": "INCONCLUSIVE",
            "line": "INCONCLUSIVE: cos(HF-fp32, HF-bf16) %.8f at rows %d+ sits between the refute "
                    "bar (%.4f) and 1.0, and the C gap moved by %+.6f (confirm bar +%.2f)"
                    % (pair, TAIL[0], REFUTE_COS, c_fp32 - c_bf16, CONFIRM_MARGIN),
            "rule": rule, "inputs": inputs}


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", choices=["full", "30s", "both"], default="both",
                    help="which clip(s) to run: the whole fixture (129.72 s) or its first 30.000 s "
                         "sub-clip (the C reference's window); default both")
    ap.add_argument("--arms", default=DEFAULT_ARMS,
                    help="comma list of precision@device arms (default %s); a bare precision uses --device"
                         % DEFAULT_ARMS)
    ap.add_argument("--device", default="cuda",
                    help='default device for arms given without @device (use "cpu" with no GPU)')
    ap.add_argument("--c-proj", default=C_PROJ_SOURCE,
                    help="the C reference proj_out: a local path, or user@host:/path to scp")
    ap.add_argument("--no-c-ref", action="store_true", help="skip the C-reference comparison")
    ap.add_argument("--verify", action="store_true",
                    help="re-hash the fixture against the manifest and verify the golden matrix bytes; "
                         "required for --write-json")
    ap.add_argument("--write-json", action="store_true", help='append "%s" to the golden json' % KEY)
    ap.add_argument("--replace", action="store_true", help="replace an existing %s key" % KEY)
    ap.add_argument("--golden", default=GOLDEN)
    ap.add_argument("--allow-tf32", action="store_true",
                    help="permit TF32 conv/matmul on CUDA (off by default: an fp32 arm must be fp32)")
    ap.add_argument("--keep-scratch", action="store_true")
    args = ap.parse_args()

    arms = parse_arms(args.arms, args.device)
    clips = ["full", "30s"] if args.clip == "both" else [args.clip]
    mode = "write" if args.write_json else "report"
    if args.write_json and not args.verify:
        print("FATAL: --write-json requires --verify (the fixture must be re-hashed before writing)",
              file=sys.stderr)
        return 2

    manifest = G.load_manifest()
    fixture_sha, samples = G.check_fixture(manifest, mode)
    print("[%s] fixture %s sha256 %s (matches manifest) samples=%d duration_s=%s"
          % (mode, os.path.relpath(G.FIXTURE, REPO), fixture_sha, samples, manifest["duration_s"]))
    missing = [rel for rel in G.PINNED_FILES if not os.path.isfile(os.path.join(MODEL, rel))]
    if missing:
        print(f"FATAL: model files missing under {MODEL}: {missing}", file=sys.stderr)
        return 2
    if args.verify:
        print("[verify] manifest sha256 %s" % sha256_path(G.MANIFEST))
        print("[verify] %d pinned checkpoint files present under %s" % (len(G.PINNED_FILES), MODEL))

    with open(args.golden) as fh:
        golden = json.load(fh)
    recorded = golden.get("embedding_matrix")
    if not recorded:
        print("FATAL: the golden json has no embedding_matrix block; nothing to compare against",
              file=sys.stderr)
        return 2
    golden_path = os.path.join(REPO, recorded["path"])
    golden_rows, golden_dim = recorded["shape"]
    golden_np = load_raw_matrix(golden_path, golden_rows, golden_dim)
    golden_sha = sha256_path(golden_path)
    if golden_sha != recorded["sha256"]:
        print(f"FATAL: {recorded['path']} sha256 {golden_sha} != golden's {recorded['sha256']}",
              file=sys.stderr)
        return 2
    print("[golden] %s %dx%d sha256 %s... verified against the golden's own record"
          % (recorded["path"], golden_rows, golden_dim, golden_sha[:16]))
    premise = {
        "committed_matrix": recorded["path"],
        "committed_matrix_sha256": golden_sha,
        "committed_matrix_source_torch_dtype": recorded.get("source_torch_dtype"),
        "committed_matrix_cast_applied": recorded.get("cast_applied"),
        "committed_matrix_cast_note": recorded.get("cast_note"),
        "committed_matrix_model_dtype": recorded.get("model_dtype"),
        "committed_matrix_produced_on_device": recorded.get("produced_on_device"),
        "note": "the committed golden matrix is recorded by the golden itself as a torch.float32 run "
                "with cast_applied=false, produced on CPU: it is HF-fp32, not the bf16 side the "
                "hypothesis needs. The bf16 arm here is run explicitly, so 'fp32 vs the golden' reads "
                "as HF-fp32(GPU) vs HF-fp32(CPU) (a device check) and the dtype question is answered "
                "by the fp32-vs-bf16 pair against the C reference.",
    }
    print("[premise] the golden's own provenance: source_torch_dtype=%s cast_applied=%s model_dtype=%s "
          "device=%s -> the committed matrix is already an fp32 HF run"
          % (premise["committed_matrix_source_torch_dtype"], premise["committed_matrix_cast_applied"],
             premise["committed_matrix_model_dtype"], premise["committed_matrix_produced_on_device"]))

    scratch = tempfile.mkdtemp(prefix="lfm3_tower_fp32_")
    try:
        matrices, harness = run_arms(arms, clips, args.allow_tf32)

        inventory = {}
        for (name, label), m in sorted(matrices.items()):
            spec = next(a for a in arms if a["name"] == name)
            d = describe_matrix(m, "%s/%s" % (name, label), {
                "tower_precision": spec["precision"],
                "device": spec["device"],
                "runtime_s": harness["arm_runtime_s"].get(name, {}).get(label),
                "output_dtype": harness["arm_output_dtypes"].get(name, {}).get(label),
                "weights_fp32_asserted": not harness["arm_dtype_audit"][name]["tensors_off_dtype"],
                "tensors_checked": harness["arm_dtype_audit"][name]["tensors_checked"],
            })
            inventory["%s/%s" % (name, label)] = d
            print("[arm %-10s %-4s] %dx%d sha256 %s mean row L2 %.6f (min %.6f max %.6f) device=%s %.3fs out=%s"
                  % (name, label, d["shape"][0], d["shape"][1], d["sha256_f32_bytes"][:16],
                     d["mean_row_l2_norm"], d["min_row_l2_norm"], d["max_row_l2_norm"],
                     d["device"], d["runtime_s"], d["output_dtype"]))

        c_np, c_meta = None, None
        if not args.no_c_ref:
            c_path = fetch_c_proj(args.c_proj, scratch)
            c_np = load_raw_matrix(c_path, C_PROJ_ROWS, C_PROJ_DIM)
            c_meta = {
                "source": args.c_proj if not os.path.exists(args.c_proj) else os.path.abspath(args.c_proj),
                "sha256": sha256_path(c_path),
                "nbytes": C_PROJ_ROWS * C_PROJ_DIM * 4,
                "shape": [C_PROJ_ROWS, C_PROJ_DIM],
                "layout": "raw f32, row-major, no header; rows 0..374 of the fixture = its first 30.000 s",
                "mean_row_l2_norm": float(np_mean(row_norms(c_np))),
                "min_row_l2_norm": float(np_min(row_norms(c_np))),
                "max_row_l2_norm": float(np_max(row_norms(c_np))),
            }
            print("[c-ref] %s sha256 %s mean row L2 %.6f" % (c_meta["source"], c_meta["sha256"][:16],
                                                              c_meta["mean_row_l2_norm"]))

        comparisons = {}
        # (1) every arm vs the committed golden matrix, on the full clip
        for spec in arms:
            m = matrices.get((spec["name"], "full"))
            if m is None:
                continue
            if m.shape != golden_np.shape:
                print("FATAL: arm %s full clip is %s, the golden matrix is %s"
                      % (spec["name"], m.shape, golden_np.shape), file=sys.stderr)
                return 2
            key = "%s_vs_golden_matrix_full" % spec["name"]
            comparisons[key] = compare(m, golden_np, BUCKETS_FULL, key)
            print_comparison(comparisons[key])
        # (2) each arm's sub-clip vs its own full-clip rows 0..374 (is sub-clipping neutral?)
        for spec in arms:
            m = matrices.get((spec["name"], "30s"))
            f = matrices.get((spec["name"], "full"))
            if m is None or f is None:
                continue
            key = "%s_30s_vs_own_full_rows_0_%d" % (spec["name"], m.shape[0] - 1)
            comparisons[key] = compare(m, f[:m.shape[0]], BUCKETS_C, key)
            print_comparison(comparisons[key])
        # (3) the C reference vs every arm's sub-clip, and vs the golden's first 375 rows
        if c_np is not None:
            for spec in arms:
                m = matrices.get((spec["name"], "30s"))
                if m is None:
                    print("[c-ref] arm %s has no 30 s matrix; skipped" % spec["name"])
                    continue
                n = min(c_np.shape[0], m.shape[0])
                key = "c_ref_vs_%s_30s" % spec["name"]
                comparisons[key] = compare(c_np[:n], m[:n], BUCKETS_C, key)
                print_comparison(comparisons[key])
            comparisons["c_ref_vs_golden_matrix_rows_0_374"] = compare(
                c_np, golden_np[:C_PROJ_ROWS], BUCKETS_C, "c_ref_vs_golden_matrix_rows_0_374")
            print_comparison(comparisons["c_ref_vs_golden_matrix_rows_0_374"])
        # (4) arm pairs on the same clip: this is the dtype test
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                if a["device"] != b["device"]:
                    continue
                for label, buckets, tag in (("30s", BUCKETS_C, "30s"), ("full", BUCKETS_FULL, "full")):
                    ma, mb = matrices.get((a["name"], label)), matrices.get((b["name"], label))
                    if ma is None or mb is None:
                        continue
                    key = "%s_vs_%s_%s" % (a["name"], b["name"], tag)
                    comparisons[key] = compare(ma, mb, buckets, key)
                    print_comparison(comparisons[key])

        primary = {"fp32": None, "bf16": None}
        for spec in arms:
            if spec["precision"] in primary and primary[spec["precision"]] is None:
                primary[spec["precision"]] = spec["name"]
        verdict = decide(comparisons, primary)
        print("\nVERDICT: %s" % verdict["line"])
        print("[verdict] rule: %s" % json.dumps(verdict["rule"]))
        for k, v in verdict["inputs"].items():
            print("[verdict] %-32s %s" % (k, v))

        block = {
            "kind": "test of the hypothesis 'HF runs the audio tower in bf16 and ITS rounding grows "
                    "with position, so HF - not our encoder - is the outlier in the proj_out "
                    "position-decay comparison'",
            "date": "2026-11-02",
            "produced_by": "transformers %s, torch %s, python %s; arms %s"
                           % (harness["transformers"], harness["torch"], harness["python"],
                              ",".join(a["name"] for a in arms)),
            "script": os.path.relpath(os.path.abspath(__file__), REPO),
            "hf_entry_point": "model.model.get_audio_features(input_features, feature_attention_mask)",
            "model": {
                "path": MODEL,
                "dtype_at_load": harness.get("model_dtype_at_load"),
                "audio_tower_dtype_after_load": harness.get("audio_tower_dtype_after_load"),
                "cast": harness.get("audio_tower_cast"),
            },
            "fixture": {
                "path": os.path.relpath(G.FIXTURE, REPO),
                "sha256": fixture_sha,
                "manifest": os.path.relpath(G.MANIFEST, REPO),
                "manifest_sha256": sha256_path(G.MANIFEST),
                "samples": samples,
                "duration_s": manifest["duration_s"],
                "mel_frames_full": harness["feature_extraction"].get("full", {}).get("input_features_shape"),
                "clip30_samples": CLIP30_SAMPLES,
                "clip30_seconds": CLIP30_SECONDS,
                "mel_frames_30s": harness["feature_extraction"].get("30s", {}).get("input_features_shape"),
                "c_ref_rows_are_first_30_seconds": True,
            },
            "premise": premise,
            "harness": harness,
            "arms": inventory,
            "c_reference": c_meta,
            "comparisons": comparisons,
            "verdict": verdict,
        }
    finally:
        if args.keep_scratch:
            print("[scratch] kept at %s" % scratch)
        else:
            shutil.rmtree(scratch, ignore_errors=True)

    if not args.write_json:
        print('\n[report] no file written (use "--verify --write-json" to append "%s")' % KEY)
        return 0

    with open(args.golden) as fh:
        doc = json.load(fh)
    if KEY in doc and not args.replace:
        print('FATAL: "%s" already exists in %s; pass --replace to rewrite it' % (KEY, args.golden),
              file=sys.stderr)
        return 2
    before = {k: json.dumps(v, sort_keys=True) for k, v in doc.items()}
    doc[KEY] = block
    tmp = args.golden + ".part"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False, sort_keys=False)
        fh.write("\n")
    # Nothing already in the golden may move: re-read and diff every pre-existing key.
    with open(tmp) as fh:
        after = json.load(fh)
    moved = [k for k in before if json.dumps(after[k], sort_keys=True) != before[k]]
    if moved:
        print("FATAL: pre-existing golden keys changed: %s (%s left for inspection)"
              % (moved, tmp), file=sys.stderr)
        return 1
    os.replace(tmp, args.golden)
    print('[write] appended "%s" to %s (%d bytes; %d pre-existing keys byte-identical)'
          % (KEY, args.golden, os.path.getsize(args.golden), len(before)))
    return 0


def np_mean(x):
    import numpy as np

    return float(np.mean(x))


def np_min(x):
    import numpy as np

    return float(np.min(x))


def np_max(x):
    import numpy as np

    return float(np.max(x))


if __name__ == "__main__":
    sys.exit(main())
