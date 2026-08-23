"""Build the optional Cython hot path (bridge._fastbridge) in place.

Usage:
    .venv/bin/python python/bridge/build_fastbridge.py

Produces ``_fastbridge.cpython-*.so`` next to this file.  The bridge works
without it (pure-ctypes fallback, ``bridge.ring_bridge.fastbridge_active()``
reports which path is live); building it removes the per-command ctypes
overhead on the ~320 ring round-trips of every decode step.

Requires ``cython`` + ``setuptools`` in the venv:
    .venv/bin/pip install cython setuptools
"""

from __future__ import annotations

import pathlib
import sys

from Cython.Build import cythonize
from setuptools import Extension
from setuptools.dist import Distribution

_HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    ext = Extension(
        "_fastbridge",
        sources=[str(_HERE / "_fastbridge.pyx")],
        extra_compile_args=["-O3"],
    )
    dist = Distribution({
        "ext_modules": cythonize(
            [ext],
            compiler_directives={"language_level": "3"},
            build_dir=str(_HERE / ".cython_build"),
        ),
    })
    cmd = dist.get_command_obj("build_ext")
    cmd.inplace = True
    cmd.build_lib = str(_HERE)
    cmd.ensure_finalized()
    cmd.run()
    # build_ext --inplace drops the .so relative to the INVOKING cwd (the
    # ext has no package prefix), or next to the source dir root depending
    # on setuptools version — sweep both into the package dir.
    for stray_dir in (_HERE.parent, pathlib.Path.cwd()):
        if stray_dir == _HERE:
            continue
        for so in stray_dir.glob("_fastbridge*.so"):
            so.rename(_HERE / so.name)
    built = list(_HERE.glob("_fastbridge*.so"))
    if not built:
        print("ERROR: no _fastbridge .so produced", file=sys.stderr)
        return 1
    print(f"built: {built[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
