#!/usr/bin/env python3
"""Fetch the layer-0 KDA (Kimi Delta Attention) tensors of GLM-5.3-Flash.

The full checkpoint is ~500 GB and is NOT downloaded.  Only the handful of
small per-layer KDA parameters listed in ``TARGETS`` are pulled, via HTTP
``Range`` requests against the safetensors shard that holds them (~7 MB
total).

Mechanism
---------
1. The LOCAL ``model.safetensors.index.json`` maps tensor name -> shard file.
2. For each needed shard, two tiny range reads recover the safetensors header:
   bytes ``0..7`` hold a u64-LE header length ``N``; bytes ``8..7+N`` hold the
   header JSON.  Each entry gives ``dtype``, ``shape`` and ``data_offsets``,
   which are relative to byte ``8 + N``.
3. Each target tensor's exact byte span is range-fetched and written out
   verbatim (no dtype conversion, no reshaping) so downstream consumers see
   the identical bits the checkpoint holds.

Outputs land in ``test-data/GLM-5.3-Flash/kda-layer0/`` alongside a
``manifest.json`` describing every blob.

Usage:  python3 tools/fetch_glm53_kda_layer0.py [--out DIR] [--index PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import requests

REPO = "zai-org/GLM-5.3-Flash"
REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
BASE_URL = "https://huggingface.co/{repo}/resolve/{rev}/{path}"

HF_PREFIX = "model.language_model.layers.0.self_attn."

# suffix -> (expected safetensors dtype, expected shape)
TARGETS: "dict[str, tuple[str, list[int]]]" = {
    "b_proj.weight": ("BF16", [64, 4096]),
    "f_a_proj.weight": ("BF16", [128, 4096]),
    "f_b_proj.weight": ("BF16", [8192, 128]),
    "g_a_proj.weight": ("BF16", [128, 4096]),
    "g_b_proj.weight": ("BF16", [8192, 128]),
    "q_conv1d.weight": ("BF16", [8192, 1, 4]),
    "k_conv1d.weight": ("BF16", [8192, 1, 4]),
    "v_conv1d.weight": ("BF16", [8192, 1, 4]),
    "A_log": ("F32", [64]),
    "dt_bias": ("F32", [8192]),
    "o_norm.weight": ("BF16", [128]),
}

DTYPE_ITEMSIZE = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class FetchError(RuntimeError):
    pass


def short_name(suffix):
    return suffix.replace(".", "_")


def range_get(session, url, start, end_inclusive):
    """GET [start, end_inclusive] and verify the server honoured the range.

    ``resolve`` URLs 302 to a CDN; redirects are followed and the returned
    payload length is checked against the requested span so a CDN that ignores
    ``Range`` (and would stream the whole multi-GB shard) fails loudly instead
    of silently corrupting the output.
    """
    want = end_inclusive - start + 1
    headers = {"Range": "bytes=%d-%d" % (start, end_inclusive)}
    resp = session.get(url, headers=headers, allow_redirects=True, timeout=300)
    if resp.status_code not in (200, 206):
        raise FetchError(
            "HTTP %d for %s range %d-%d" % (resp.status_code, url, start, end_inclusive)
        )
    if resp.status_code == 200:
        raise FetchError(
            "server ignored Range (returned 200, not 206) for %s range %d-%d; "
            "refusing to download the whole shard" % (url, start, end_inclusive)
        )
    data = resp.content
    if len(data) != want:
        raise FetchError(
            "range mismatch for %s: requested %d bytes (%d-%d), got %d"
            % (url, want, start, end_inclusive, len(data))
        )
    return data


def read_safetensors_header(session, url):
    """Return (header_dict, data_start_offset) for a remote safetensors file."""
    raw_len = range_get(session, url, 0, 7)
    header_len = int.from_bytes(raw_len, "little", signed=False)
    if header_len <= 0 or header_len > (256 << 20):
        raise FetchError("implausible safetensors header length %d for %s" % (header_len, url))
    raw_header = range_get(session, url, 8, 7 + header_len)
    header = json.loads(raw_header.decode("utf-8"))
    return header, 8 + header_len


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="fetch GLM-5.3-Flash layer-0 KDA tensors")
    ap.add_argument(
        "--index",
        default=os.path.join(here, "test-data", "GLM-5.3-Flash", "model.safetensors.index.json"),
        help="local safetensors index json",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(here, "test-data", "GLM-5.3-Flash", "kda-layer0"),
        help="output directory for the raw tensor blobs + manifest",
    )
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()

    with open(args.index, "r", encoding="utf-8") as fh:
        index = json.load(fh)
    weight_map = index["weight_map"]

    # ---- resolve shard per target ---------------------------------------------
    shard_of = {}
    for suffix in TARGETS:
        hf_name = HF_PREFIX + suffix
        if hf_name not in weight_map:
            raise FetchError("%s not present in index %s" % (hf_name, args.index))
        shard_of[suffix] = weight_map[hf_name]

    shards = sorted(set(shard_of.values()))
    print("targets: %d tensors across %d shard(s): %s" % (len(TARGETS), len(shards), ", ".join(shards)))

    os.makedirs(args.out, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "layerstorm-kda-fixture/1.0"})

    headers = {}
    for shard in shards:
        url = BASE_URL.format(repo=args.repo, rev=args.revision, path=shard)
        hdr, data_start = read_safetensors_header(session, url)
        headers[shard] = (hdr, data_start, url)
        print("  header %s: %d entries, data starts at byte %d" % (shard, len(hdr), data_start))

    entries = []
    total_bytes = 0
    for suffix in TARGETS:
        want_dtype, want_shape = TARGETS[suffix]
        hf_name = HF_PREFIX + suffix
        shard = shard_of[suffix]
        hdr, data_start, url = headers[shard]
        if hf_name not in hdr:
            raise FetchError("%s missing from header of %s" % (hf_name, shard))
        meta = hdr[hf_name]
        dtype = meta["dtype"]
        shape = list(meta["shape"])
        off0, off1 = meta["data_offsets"]

        if dtype != want_dtype:
            raise FetchError("%s: dtype %s != expected %s" % (hf_name, dtype, want_dtype))
        if shape != want_shape:
            raise FetchError("%s: shape %s != expected %s" % (hf_name, shape, want_shape))

        itemsize = DTYPE_ITEMSIZE[dtype]
        nelem = 1
        for d in shape:
            nelem *= d
        want_bytes = nelem * itemsize
        span = off1 - off0
        if span != want_bytes:
            raise FetchError(
                "%s: data_offsets span %d != %d elems * %d B = %d"
                % (hf_name, span, nelem, itemsize, want_bytes)
            )

        start = data_start + off0
        blob = range_get(session, url, start, data_start + off1 - 1)
        if len(blob) != want_bytes:
            raise FetchError("%s: fetched %d bytes, expected %d" % (hf_name, len(blob), want_bytes))

        name = short_name(suffix)
        path = os.path.join(args.out, name + ".bin")
        with open(path, "wb") as fh:
            fh.write(blob)
        digest = hashlib.sha256(blob).hexdigest()
        total_bytes += want_bytes
        print(
            "  %-16s %-5s %-16s %9d B  sha256=%s..."
            % (name, dtype, str(shape), want_bytes, digest[:16])
        )

        entries.append(
            {
                "short_name": name,
                "hf_name": hf_name,
                "file": name + ".bin",
                "dtype": dtype,
                "shape": shape,
                "byte_size": want_bytes,
                "sha256": digest,
                "shard": shard,
                "data_offsets": [off0, off1],
                "revision": args.revision,
            }
        )

    manifest = {
        "repo": args.repo,
        "revision": args.revision,
        "source": "huggingface safetensors HTTP Range reads",
        "hf_prefix": HF_PREFIX,
        "layer": 0,
        "note": (
            "Raw little-endian tensor bytes exactly as stored in the checkpoint shard "
            "(BF16 stays BF16). Shapes are the safetensors shapes. 64 linear heads x "
            "head_dim 128 = 8192; short_conv_kernel_size 4; gate_lower_bound -5.0."
        ),
        "total_bytes": total_bytes,
        "tensors": entries,
    }
    mpath = os.path.join(args.out, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")

    print("")
    print("wrote %d tensors, %d bytes total -> %s" % (len(entries), total_bytes, args.out))
    print("manifest: %s" % mpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
