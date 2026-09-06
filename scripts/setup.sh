#!/usr/bin/env bash
# LayerStoRm one-command environment setup.
#
#   ./scripts/setup.sh
#
# What it does — and deliberately does NOT do:
#   * NO sudo, ever. System packages are your call: the script checks for
#     them and prints the exact `apt` line for anything missing.
#   * Session-local toolchains only, under <repo>/.toolchain and ./.venv —
#     nothing system-wide, nothing in $HOME dotfiles, no PATH edits to your
#     shell profile. Delete .toolchain/ and .venv/ to undo everything.
#   * Idempotent: re-running repairs a partial setup and skips what exists.
#   * It does NOT run the build. It ends by printing the exact next
#     commands (cmake configure/build + a serve example).
#
# Node.js note: `tools/gen_config.mjs` needs a Node interpreter (>= 18) at
# CMake configure time — no npm packages. When the system has none we
# unpack the OFFICIAL nodejs.org binary tarball locally: one auditable
# download, zero extra tooling (nodeenv/mise would add a manager to manage
# a single static binary).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLCHAIN="$REPO_ROOT/.toolchain"
BIN_DIR="$TOOLCHAIN/bin"
NODE_VERSION="${LS_NODE_VERSION:-v22.15.0}"   # LTS; interpreter only, no npm packages needed
PYTHON_VERSION="${LS_PYTHON_VERSION:-3.12}"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }

mkdir -p "$BIN_DIR"
export PATH="$BIN_DIR:$PATH"

# ── 1. System prerequisites (report only — no sudo here) ─────────────────
say "1/4 System prerequisites"
missing=()
command -v gcc   >/dev/null || missing+=(build-essential)
command -v g++   >/dev/null || missing+=(build-essential)
command -v cmake >/dev/null || missing+=(cmake)
command -v git   >/dev/null || missing+=(git)
command -v python3 >/dev/null || missing+=(python3)
have_lib() { ldconfig -p 2>/dev/null | grep -q "$1"; }
have_lib libnuma.so   || missing+=(libnuma-dev)
have_lib liburing.so  || missing+=(liburing-dev)
have_lib libnccl.so   || missing+=("libnccl2 libnccl-dev")
if ((${#missing[@]})); then
    # de-duplicate while keeping order
    pkgs="$(printf '%s\n' "${missing[@]}" | awk '!seen[$0]++' | tr '\n' ' ')"
    note "Missing system packages. Run this yourself (the script never sudos):"
    note ""
    note "    sudo apt install $pkgs"
    note ""
    note "(NCCL and CUDA may need NVIDIA's apt repo or installer on your"
    note "distro: https://developer.nvidia.com/cuda-downloads)"
    note "Then re-run ./scripts/setup.sh — it picks up where it left off."
    exit 1
fi
if command -v cmake >/dev/null; then
    cmv="$(cmake --version | head -1 | awk '{print $3}')"
    note "cmake $cmv (need >= 3.25)"
fi
note "OK: compiler, cmake, git, python3, libnuma, liburing, NCCL found."

# Submodules: deps/ kernel repos + CUTLASS must be populated — a clone made
# without --recursive leaves these directories EMPTY and the CMake configure
# later fails with an opaque error. Checked by emptiness (not a filename)
# so manually-populated trees from release archives pass too.
missing_subs=()
for d in deps/LayerStoRmKernels deps/LayerStoRmGemmKernels \
         deps/LayerStoRmExpertKernels 3rd-party/cutlass; do
    [[ -n "$(ls -A "$REPO_ROOT/$d" 2>/dev/null)" ]] || missing_subs+=("$d")
done
if ((${#missing_subs[@]})); then
    note "Submodule contents MISSING: ${missing_subs[*]}"
    note "This looks like a clone without --recursive. Fix it with:"
    note ""
    note "    git submodule update --init --recursive"
    note ""
    note "(Release archives ship no submodule contents — clone the repos"
    note "listed in .gitmodules into those paths; see docs/BUILDING.md.)"
    note "Then re-run ./scripts/setup.sh."
    exit 1
fi
note "OK: submodules populated (deps/ kernels + CUTLASS)."

# ── 2. uv + Python venv (the project's .venv pattern) ────────────────────
say "2/4 Python environment (uv + ./.venv)"
if command -v uv >/dev/null; then
    UV="$(command -v uv)"
    note "using system uv: $UV"
elif [[ -x "$TOOLCHAIN/uv/uv" ]]; then
    UV="$TOOLCHAIN/uv/uv"
    note "using session-local uv: $UV"
else
    note "installing uv locally into .toolchain/uv (official installer, no PATH edits)"
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR="$TOOLCHAIN/uv" UV_NO_MODIFY_PATH=1 sh
    UV="$TOOLCHAIN/uv/uv"
fi
ln -sf "$UV" "$BIN_DIR/uv"
cd "$REPO_ROOT"
if [[ ! -x .venv/bin/python ]]; then
    "$UV" venv --python="$PYTHON_VERSION"
fi
"$UV" pip install -r requirements.txt
note "venv ready: $REPO_ROOT/.venv (python $("$REPO_ROOT/.venv/bin/python" -V 2>&1 | awk '{print $2}'))"
note "optional extras: requirements-dev.txt (tests), requirements-tools.txt (offline tooling)"

# ── 3. Node.js (interpreter only, for tools/gen_config.mjs) ──────────────
say "3/4 Node.js (config-parser codegen needs the interpreter, >= 18)"
node_ok() { command -v node >/dev/null && [[ "$(node -e 'console.log(process.versions.node.split(".")[0])')" -ge 18 ]]; }
if node_ok; then
    note "using system node: $(command -v node) ($(node --version))"
else
    case "$(uname -m)" in
        x86_64)  node_arch=x64 ;;
        aarch64) node_arch=arm64 ;;
        *) echo "unsupported arch for the official Node tarball: $(uname -m)" >&2; exit 1 ;;
    esac
    node_dir="$TOOLCHAIN/node-$NODE_VERSION-linux-$node_arch"
    if [[ ! -x "$node_dir/bin/node" ]]; then
        note "downloading the official Node tarball $NODE_VERSION into .toolchain/"
        tarball="node-$NODE_VERSION-linux-$node_arch.tar.xz"
        curl -fL "https://nodejs.org/dist/$NODE_VERSION/$tarball" -o "$TOOLCHAIN/$tarball"
        tar -xJf "$TOOLCHAIN/$tarball" -C "$TOOLCHAIN"
        rm -f "$TOOLCHAIN/$tarball"
    fi
    ln -sf "$node_dir/bin/node" "$BIN_DIR/node"
    note "session-local node: $BIN_DIR/node ($("$BIN_DIR/node" --version))"
    note "CMake needs it on PATH at configure time; the next-steps below include the export."
fi

# ── 4. CUDA visibility ───────────────────────────────────────────────────
say "4/4 CUDA"
if [[ -n "${CUDACXX:-}" ]]; then
    note "CUDACXX is set: $CUDACXX (CMake will use it)"
elif command -v nvcc >/dev/null; then
    note "nvcc on PATH: $(command -v nvcc) ($(nvcc --version | grep -o 'release [0-9.]*' | head -1))"
else
    candidates=(/usr/local/cuda*/bin/nvcc)
    if [[ -x "${candidates[0]:-/nonexistent}" ]]; then
        note "nvcc not on PATH, but found: ${candidates[*]}"
        note "CMake will autodetect one of these (or set CUDACXX to choose; see docs/BUILDING.md)."
    else
        note "WARNING: no nvcc found (PATH or /usr/local/cuda*). Install the"
        note "CUDA Toolkit >= 12.8 before building: https://developer.nvidia.com/cuda-downloads"
    fi
fi

# ── Done: print the next commands (this script never builds) ─────────────
say "Setup complete — next steps"
cat <<EOF

   # if setup installed a session-local node, put it on PATH first:
   export PATH="$BIN_DIR:\$PATH"

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \\
     -DPYTHON_EXECUTABLE="\$PWD/.venv/bin/python" \\
     -Dpybind11_DIR="\$(.venv/bin/python -m pybind11 --cmakedir)"
   cmake --build build -j\$(nproc)
   .venv/bin/python python/bridge/build_fastbridge.py   # optional Cython fast path

   # then serve (see README.md for getting weights + tokenizer):
   CUDA_DEVICE_ORDER=PCI_BUS_ID \\
   .venv/bin/python python/cli/serve.py --autoconfig \\
       --model <path-to-weights>.gguf \\
       --max-sequence-length 1048576 --max-concurrent 2 \\
       --model-name my-model --host 127.0.0.1 --port 8000

EOF
