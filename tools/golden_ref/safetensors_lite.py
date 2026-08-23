"""Minimal pure-python sharded-safetensors reader (TD-GOLDEN reference tooling).

Why not the `safetensors` pip package: (a) it is not installed in the project
venv, and (b) the NVFP4 scale tensors (`*.weight_scale`, `*.weight_scale_2`,
`*.input_scale`) of DeepSeek-V3.2-NVFP4 are present in the shard headers but
ABSENT from model.safetensors.index.json, so the index cannot be trusted as
the tensor catalogue — every shard header must be scanned anyway.

Format: 8-byte little-endian header length, JSON header mapping tensor name →
{dtype, shape, data_offsets [begin, end)} relative to the byte after the
header, then raw data.
"""

import json
import struct
from pathlib import Path

import numpy as np
import torch

_DTYPES = {
    "F32":     torch.float32,
    "F16":     torch.float16,
    "BF16":    torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "U8":      torch.uint8,
    "I8":      torch.int8,
    "I32":     torch.int32,
    "I64":     torch.int64,
    "U16":     torch.uint16,
}


class ShardedSafetensors:
    """Catalogue of all tensors across model*.safetensors shards in a dir."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        # name -> (file_path, dtype_str, shape, abs_begin, abs_end)
        self._index = {}
        shards = sorted(self.model_dir.glob("model*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no model*.safetensors in {model_dir}")
        for f in shards:
            with open(f, "rb") as fh:
                (hlen,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(hlen))
            base = 8 + hlen
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                b, e = meta["data_offsets"]
                self._index[name] = (f, meta["dtype"], tuple(meta["shape"]),
                                     base + b, base + e)

    def __contains__(self, name):
        return name in self._index

    def names(self):
        return self._index.keys()

    def meta(self, name):
        """Returns (dtype_str, shape)."""
        _, dt, shape, _, _ = self._index[name]
        return dt, shape

    def tensor(self, name):
        """Read a tensor; returns torch tensor in its stored dtype."""
        f, dt, shape, b, e = self._index[name]
        with open(f, "rb") as fh:
            fh.seek(b)
            raw = fh.read(e - b)
        t = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())
        t = t.view(_DTYPES[dt])
        return t.reshape(shape)

    def tensor_f32(self, name):
        return self.tensor(name).float()
