# syntax=docker/dockerfile:1.10
# =============================================================================
# deepep-v2-efa-base v0.2.5-sm90a  (Wave 23 patch-layer)
# =============================================================================
#
# This Dockerfile is a thin patch layer on top of ghcr.io/antonai-work/
# deepep-v2-efa-base:v0.2.4-sm90a. It overwrites libnccl.so.2 with a
# rebuild of NVIDIA/nccl tag v2.30.4-1 that contains a single-line change:
#
#   src/graph/topo.h:
#     -#define NCCL_TOPO_XML_MAX_NODES 256
#     +#define NCCL_TOPO_XML_MAX_NODES 2048
#
# Why:
#   Wave 19 / 21 / 22 on p5.48xlarge (H100 x 8, EFA x 32) all hit
#   "graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)"
#   during DeepEP V2 tests/elastic/test_ep.py cross-node bring-up.
#   NCCL's ncclTopoFuseXml intra-node fuse cannot be shrunk below 256
#   on a 32-NIC box: ncclTopoFuseXmlRecursive dedup requires ALL
#   attributes to match, so per-rank autogen (which includes rank /
#   pci_busid in attrs) appends 8 x ~125 nodes = ~1000 > 256. Neither
#   NCCL_TOPO_FILE override nor FI_EFA_DEVICE_LIST mitigate. The only
#   remaining fix is to raise the cap.
#
# Prior releases:
#   v0.2.0 ... v0.2.4-sm90a: the full 676-line base rebuild is preserved
#   at those git tags (the `Dockerfile` on main at each tag). Reproduce
#   v0.2.4 via `git checkout v0.2.4-sm90a && docker build .`
#
# Patch artifacts live under nccl-patches/. Rebuild the .so via
# `./nccl-patches/build.sh` (host CUDA toolkit required).
# =============================================================================

FROM ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.4-sm90a

ENV DEEPEP_BASE_VERSION=v0.2.5-sm90a

# Legacy build-args: accepted for compatibility with the existing GHA
# workflow (build-and-push.yml passes DEEPEP_FORK/BRANCH/SHA/NVIDIA_NCCL_PIN
# from pins.env). They are NOT consumed in this layered image; v0.2.5
# inherits its DeepEP + NCCL pins from v0.2.4. Kept here so the workflow
# doesn't fail on "unconsumed build-arg" if any checker is enabled.
ARG DEEPEP_FORK=unused-in-v0.2.5
ARG DEEPEP_BRANCH=unused-in-v0.2.5
ARG DEEPEP_SHA=unused-in-v0.2.5
ARG NVIDIA_NCCL_PIN=unused-in-v0.2.5

# The patched libnccl.so.2 is produced outside the Docker build by
# nccl-patches/build.sh and committed to the repo so the GHA runner does
# not have to own ~45 minutes of CUDA device-code compile time.
COPY nccl-patches/libnccl.so.2 /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2

RUN set -eux; \
    NCCL_LIB=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib; \
    chmod 0755 "${NCCL_LIB}/libnccl.so.2"; \
    ln -sf libnccl.so.2 "${NCCL_LIB}/libnccl.so"; \
    ls -la "${NCCL_LIB}"/libnccl.so*; \
    echo "[v0.2.5] probing GIN ABI symbols in the patched NCCL wheel..."; \
    gin_syms="$(nm -D --defined-only "${NCCL_LIB}/libnccl.so.2" 2>/dev/null | grep -Eo 'ncclTeamWorld|ncclCommQueryProperties|ncclGinBarrierCreateRequirement' | sort -u || true)"; \
    for sym in ncclTeamWorld ncclCommQueryProperties ncclGinBarrierCreateRequirement; do \
      echo "${gin_syms}" | grep -q "${sym}" \
        || (echo "[v0.2.5] FATAL: patched NCCL wheel missing GIN ABI symbol ${sym}" && exit 1); \
    done; \
    echo "[v0.2.5] GIN ABI OK (all 3 required symbols present)"; \
    ldconfig; \
    ldconfig -p | grep 'libnccl.so' | head -3

LABEL org.opencontainers.image.version="v0.2.5-sm90a"
LABEL org.opencontainers.image.description="DeepEP V2 EFA base with NCCL_TOPO_XML_MAX_NODES bumped 256 -> 2048 (Wave 23 unblock)"
