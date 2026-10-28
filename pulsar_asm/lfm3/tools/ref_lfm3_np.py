#!/usr/bin/env python3
"""NumPy reference for LFM3 forward pass, reading from the flat blob."""
import json
import os
import numpy as np

class Ref:
    def __init__(self, weights):
        self.manifest = weights + ".manifest.json"
        self.man = json.load(open(self.manifest))
        self.m = self.man["meta"]
        self._f = open(weights, "rb")

    def _blob(self, name, dtype):
        e = self.man["layout"][name]
        n = int(np.prod(e["shape"]))
        if n == 0:
            return np.array([])
        self._f.seek(e["offset"])
        data = self._f.read(n * np.dtype(dtype).itemsize)
        a = np.frombuffer(data, dtype=dtype)
        return a.reshape(e["shape"]) if len(e["shape"]) > 1 else a

    def f32(self, name):
        return self._blob(name, np.float32)

    def w(self, name):
        u = self._blob(name, np.uint16).astype(np.uint32)
        f = (u << 16).view(np.float32)
        sh = self.man["layout"][name]["shape"]
        return f.reshape(sh) if len(sh) > 1 else f
