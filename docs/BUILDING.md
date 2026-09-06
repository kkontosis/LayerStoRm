# Building LayerStoRm

The full build reference. If you just want to serve a model, the short path
in the top-level [README](../README.md) — `./scripts/setup.sh` plus two cmake
commands — is this document on autopilot; come here when something needs
adjusting.

## Toolchain

**CMake 3.25+, CUDA 12.8+, GCC with C++20, NCCL 2.20+, Node.js 18+,
Python 3.10+**; `libnuma` and `liburing` unlock NUMA pinning and the NVMe
tier (`nlohmann_json`/`spdlog`/CUTLASS are fetched automatically if absent).

Node.js is **required to configure**, not optional: the config parser
(`src/config/config_parser.{h,cpp}`) is generated from `config/schema.json`
by `tools/gen_config.mjs`, and CMake looks for `node` at configure time. It
needs no npm packages — only the interpreter. `./scripts/setup.sh` installs a
session-local copy if your system has none.

> **Dependencies note:** release archives don't carry submodule contents —
> clone the sibling repos / CUTLASS listed in `.gitmodules` into their paths
> (or use `git clone --recurse-submodules` once published with resolvable
> URLs) before configuring.

The kernel collections live in sibling repositories, wired in as submodules
(`deps/LayerStoRmKernels`, `deps/LayerStoRmGemmKernels`,
`deps/LayerStoRmExpertKernels`, plus `3rd-party/cutlass`):

```sh
git clone --recursive https://github.com/kkontosis/LayerStoRm.git
cd LayerStoRm
```

## One-command environment setup

```sh
./scripts/setup.sh
```

The script is checked in (read it — it's short) rather than a
`curl | bash` from the internet: same one-command convenience, smaller
security surface. It never uses sudo; it prints the exact `apt` prerequisite
line for you to run yourself, installs **session-local** toolchains only
(uv + the Python venv, and a local Node.js if needed, under `.toolchain/`
in the repo), verifies CUDA/nvcc visibility, and finishes by printing the
cmake and serve commands. It is idempotent — re-running it repairs a partial
setup. It does not run the build itself.

Everything it does can be done by hand; the rest of this page is the manual
version.

## If CUDA is not your system default

CMake now autodetects `nvcc` when neither `CUDACXX` nor
`-DCMAKE_CUDA_COMPILER` is set: it probes `nvcc` on `PATH`, then the
`/usr/local/cuda*` installs — the unversioned `/usr/local/cuda` default
symlink wins when present, otherwise the highest version — reports what
it picked, and errors with instructions if nothing is found.
*(Configure through the autodetect verified 2026-09-06: with `CUDACXX`,
`LIBRARY_PATH` and `LD_LIBRARY_PATH` all unset and CUDA stripped from
`PATH`, a fresh configure autodetected `/usr/local/cuda/bin/nvcc`
(13.1) via the glob branch and passed the CUDA compiler link probe and
toolkit resolution without the library-path exports below — CMake links
the CUDA runtime by absolute path, so those exports are not needed for
a CMake build. A full compile through this path is still UNTESTED; the
manual `CUDACXX` route below remains the known-good path.)*

With several CUDA versions installed, point the build at the one you want
before configuring — CMake picks up `CUDACXX`; the `PATH` and library
exports keep nvcc-adjacent tooling and runtime library resolution on the
same version:

```sh
export CUDACXX=/usr/local/cuda-13.1/bin/nvcc
export PATH="/usr/local/cuda-13.1/bin:$PATH"
export LIBRARY_PATH="/usr/local/cuda-13.1/targets/x86_64-linux/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
```

Skip this if `nvcc` on your `PATH` is already the version you intend to
build with.

## Python environment first

Create the virtualenv **before** configuring CMake — the `layerstorm_engine`
pybind11 module is built against it, and `find_package(pybind11)` resolves
through the packages installed here. These examples use
[uv](https://docs.astral.sh/uv/); `python -m venv` + `pip` works identically.
(`./scripts/setup.sh` does exactly this, with a session-local uv.)

```sh
# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python=3.12
uv pip install -r requirements.txt
```

No `activate` step: `uv pip` picks up `./.venv` from the working directory,
and every command below names the interpreter explicitly, so the flow works
in a plain shell. `source .venv/bin/activate` if you prefer it anyway.

`requirements.txt` covers building and serving. Two more sets are available:
`requirements-dev.txt` (pytest + the end-to-end HTTP client, plus the
optional Cython hot path) and `requirements-tools.txt` (torch/LightGBM for
the offline calibration, placement-solver and expert-prediction tooling
under `tools/` — not needed to serve).

Guided decoding (`--enable-auto-tool-choice`, JSON schema output) pulls
`xgrammar`, and with it torch and triton — several GB. Drop that one line
from `requirements.txt` if you only need plain completions; nothing else on
the serving path imports torch.

## Configure and build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build build -j$(nproc)

# command-ring hot path (Cython, built in place next to the bridge)
.venv/bin/python python/bridge/build_fastbridge.py

./build/tests/unit/layerstorm_unit_tests     # optional sanity
```

The two `-D` hints point CMake at the venv you just made; drop them if
pybind11 is installed system-wide, or pass
`-DLAYERSTORM_BUILD_PYTHON=OFF` to build the C++ engine alone (no serving).

The `build_fastbridge.py` step compiles `bridge._fastbridge`, which removes
the per-command ctypes overhead on the ~320 ring round-trips every decode
step makes. Serving works without it — the bridge falls back to pure ctypes,
and `bridge.ring_bridge.fastbridge_active()` reports which path is live —
but the fallback is measurably slower, so build it unless you have a reason
not to. Re-run it after changing the ring protocol.

## Troubleshooting

**`undefined symbol: ncclCommResume` (or another `libnccl` symbol) at
boot.** Two NCCLs are in play: the engine module links your *system*
`libnccl.so.2`, while torch ships its own under `.venv/.../nvidia/nccl/lib`.
Only one can serve a process — whichever loads first — so a torch newer
than the system NCCL ends up bound to the older library and cannot find a
symbol it needs.

The serving path imports torch before the engine so this cannot happen. If
you hit it in your own script or a test, do the same, or preload torch's
copy:

```sh
LD_PRELOAD=$(.venv/bin/python -c "import nvidia.nccl, os; print(os.path.join(list(nvidia.nccl.__path__)[0], 'lib', 'libnccl.so.2'))") \
  .venv/bin/python your_script.py
```

The newer NCCL is backward compatible, so the engine runs fine against it.
Installing a torch that matches your system NCCL works too, but pins you to
it.
