#!/usr/bin/env bash
# Build patched libnccl.so.2 from NVIDIA/nccl v2.30.4-1 with
# NCCL_TOPO_XML_MAX_NODES bumped from 256 to 2048.
#
# Requires: nvcc on PATH (host CUDA toolkit 12.8+).
# Output:   ./libnccl.so.2 + ./libnccl.so.2.sha256

set -euo pipefail
cd "$(dirname "$0")"

WORK=${WORK:-/tmp/nccl-v2.30.4-patched}
NCCL_TAG=v2.30.4-1

if [ ! -d "${WORK}/.git" ] && [ ! -f "${WORK}/.git" ]; then
  rm -rf "${WORK}"
  git clone --depth 1 --branch "${NCCL_TAG}" https://github.com/NVIDIA/nccl.git "${WORK}"
fi

cd "${WORK}"
git checkout "${NCCL_TAG}" 2>/dev/null || true

# Idempotent patch: only apply if still at stock 256.
if grep -qE '^#define NCCL_TOPO_XML_MAX_NODES 256$' src/graph/topo.h; then
  sed -i 's/#define NCCL_TOPO_XML_MAX_NODES 256$/#define NCCL_TOPO_XML_MAX_NODES 2048/' src/graph/topo.h
fi
grep -E '^#define NCCL_TOPO_XML_MAX_NODES ' src/graph/topo.h

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
NVCC_GENCODE=${NVCC_GENCODE:-"-gencode=arch=compute_90,code=sm_90"}
PARALLEL=${PARALLEL:-$(nproc)}

CUDA_HOME="${CUDA_HOME}" NVCC_GENCODE="${NVCC_GENCODE}" make -j"${PARALLEL}" src.build

SRC="${WORK}/build/lib/libnccl.so.2"
OUT="$(dirname "$0")/libnccl.so.2"
cp -f "${SRC}" "${OUT}"
sha256sum "${OUT}" | tee "${OUT}.sha256"
ls -la "${OUT}"
echo "[build.sh] done; ${OUT} is ready to commit"
