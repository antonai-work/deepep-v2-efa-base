# syntax=docker/dockerfile:1.10
# =============================================================================
# deepep-v2-efa-base - DeepEP V2 (NCCL Gin backend) on AWS EFA
# =============================================================================
#
# Purpose:
#   A clean, standalone base image that consumes DeepEP V2's Python API
#   (deep_ep.ElasticBuffer), using the NCCL Gin backend over AWS EFA.
#   Child images can FROM this and add their inference / training engine
#   on top without re-solving CUDA + EFA + NCCL + aws-ofi-nccl + DeepEP.
#
# Target arch: SM 9.0a (H100, H200).
#
# Pinned upstream sources (every line is an exact, verifiable anchor):
#
#   Base OS      ubuntu 24.04 noble (from nvidia/cuda:12.9.0-devel-ubuntu24.04)
#   CUDA         12.9.0                      NVIDIA registry
#   EFA user     1.48.0                      efa-installer.amazonaws.com tarball
#                                            installed via --build-ngc path
#                                            (libfabric1-aws + libnccl-ofi-ngc-v3)
#   NCCL         2.30.4                      pip wheel `nvidia-nccl-cu12>=2.30.4`
#                                            (Wave 9: unified to cu12 to match
#                                            torch cu129 ABI. HISTORICAL:
#                                            cu129 is a CUDA 12.9 build, NOT
#                                            cu13. Mixing cu12 and cu13 NCCL
#                                            caused Wave 8's invalid-device-
#                                            ordinal crash.)
#                                            apt libnccl2 2.26.x is purged
#   aws-ofi-nccl 6e504db3403931cde43a2335adcc73fbc69cccac (2026-04-24)
#                                            github.com/aws/aws-ofi-nccl
#                                            "gin: Size active_put_signal to
#                                            full sequence number space"
#                                            (upstream fix superseding the
#                                            earlier env-tunable ring-size
#                                            workaround). Built with
#                                            --enable-platform-aws against the
#                                            NCCL 2.30.4 pip wheel headers.
#   GDRCopy      v2.5.1                      github.com/NVIDIA/gdrcopy
#   NVSHMEM      nvidia-nvshmem-cu12>=3.3.9  pip wheel. Only libnvshmem_host.so
#                                            is linked at DeepEP build time;
#                                            runtime does not call NVSHMEM
#                                            when DEEP_EP_BACKEND=nccl.
#   DeepEP       146cc356aa00c39ac1590c05775e05b0f031e70c
#                on branch aws-efa-auto-qp-cap-v2 of
#                github.com/dmvevents/DeepEP-1 (fork of deepseek-ai/DeepEP@main
#                post-PR #605 merge b306af0), carrying the three PR #612
#                commits:
#                  fe20874  aws-efa: cap auto-QP at 2
#                  0b78333  aws-efa: EFA fast path in get_rdma_gbs
#                  146cc356  aws-efa: raise kScaleoutUpdateInterval 3 -> 16
#                The patches/ directory in this repo carries the same three
#                commits as reviewable .patch files; see docs/ARCHITECTURE.md
#                for why we clone the pre-patched branch rather than apply
#                them sequentially at image build time.
#   PyTorch      pip index https://download.pytorch.org/whl/cu129 (falls back
#                to cu128 if cu129 not yet indexed for the host arch)
#   NumPy        <2 (torch.distributed.all_gather_object compat with the
#                DeepEP V2 NCCL comm handle exchange)
#
# Reproducibility note: apt packages from ubuntu 24.04 noble are NOT pinned by
# version - they change on every `apt-get update`. The subset installed here
# is intentionally minimal (build toolchain + numa/nl userspace) and the
# binary artifacts that matter for the DeepEP stack are all pinned (EFA
# installer version, aws-ofi-nccl SHA, DeepEP SHA, NCCL pip wheel version).
#
# Smoke test (inside the image):
#   python3 -c "import deep_ep; print(deep_ep.ElasticBuffer)"
#   bash /preflight.sh
#
# Build:
#   docker build -t deepep-v2-efa-base:dev .
#
# =============================================================================

FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Wave 9: cu12 unification. NCCL pin sourced from pins.env at repo root; CI
# passes this via --build-arg. Default fallback so `docker build .` works
# standalone. HISTORICAL: mixing cu12 (torch cu129) with cu13 NCCL caused
# invalid-device-ordinal crashes at first MoE dispatch (Wave 8 evidence).
ARG NVIDIA_NCCL_PIN=nvidia-nccl-cu12>=2.30.4

# -----------------------------------------------------------------------------
# System deps + EFA userspace + aws-ofi-nccl (NGC-aware install path)
# -----------------------------------------------------------------------------
# EFA installer 1.48+ ships a --build-ngc path specifically for NVIDIA DL base
# containers: it installs libfabric1-aws + libfabric-aws-dev + libfabric-aws-bin
# debs plus the NGC-flavored libnccl-ofi-ngc-v3 deb. The NGC plugin is built
# with --disable-nccl-net-library so it is NCCL-version-agnostic.
#
# We deliberately do NOT install rdma-core / libibverbs-dev / ibverbs-providers
# from the noble apt repository. Those would land v50 and block the v61
# rdma-core bundled inside the EFA 1.48 tarball. libfabric1-aws hard-depends
# on ibverbs-providers >= 59 so the apt path cannot satisfy it.
ARG EFA_INSTALLER_VERSION=1.48.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates curl wget tar xz-utils file \
    autoconf automake libtool pkg-config cmake ninja-build patchelf \
    python3-dev python3-pip python3-setuptools python3-wheel \
    libopenmpi-dev openmpi-bin \
    libnuma-dev libnl-3-dev libnl-route-3-dev \
    kmod pciutils \
 && rm -rf /var/lib/apt/lists/*

# The 1.48 installer's NGC path runs plain `dpkg -i`. libfabric1-aws declares
# `Depends: ibverbs-providers (>= 59)` but noble ships 50.0. Manually installing
# the v61 debs from the tarball first (with --auto-deconfigure) upgrades the
# preinstalled v50 ibverbs-providers to v61 and then libfabric1-aws can drop
# in. This is the same sequence the NGC CUDA DL Base 26.03 image uses.
RUN set -eux; \
    mkdir -p /var/build-scratch/efa; \
    cd /var/build-scratch/efa; \
    curl -fsSL -o aws-efa-installer.tar.gz \
      "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"; \
    tar -xzf aws-efa-installer.tar.gz; \
    cd aws-efa-installer; \
    arch="$(uname -m)"; \
    dpkg -i --auto-deconfigure \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibverbs1_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/ibverbs-providers_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibverbs-dev_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/librdmacm1_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/librdmacm-dev_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibumad3_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibumad-dev_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibmad5_*.deb \
        DEBS/UBUNTU2404/"${arch}"/rdma-core/libibnetdisc5_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/libfabric1-aws_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/libfabric-aws-bin_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/libfabric-aws-dev_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/efa-config_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/efa-profile_*.deb; \
    dpkg -i DEBS/UBUNTU2404/"${arch}"/libnccl-ofi-ngc-v3_*.deb; \
    rm -f /opt/amazon/efa/lib/libfabric.a; \
    test -f /opt/amazon/efa/include/rdma/fabric.h; \
    test -f /opt/amazon/efa/lib/libfabric.so.1; \
    test -f /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so; \
    strings /lib/x86_64-linux-gnu/libefa.so.1 | grep '^EFA_1\.4$' > /dev/null \
        || (echo "FATAL: libefa.so.1 missing EFA_1.4 symbol" && exit 1); \
    ldconfig; \
    rm -rf /var/build-scratch/efa

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/opt/amazon/efa/bin:${PATH} \
    PKG_CONFIG_PATH=/opt/amazon/efa/lib/pkgconfig:/opt/amazon/efa/lib64/pkgconfig \
    LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/efa/lib64:/opt/amazon/ofi-nccl/lib:/usr/local/cuda/lib64

# -----------------------------------------------------------------------------
# GDRCopy v2.5.1 userspace (kernel module lives on the host)
# -----------------------------------------------------------------------------
RUN set -eux; \
    git clone --depth=1 --branch v2.5.1 \
      https://github.com/NVIDIA/gdrcopy.git /var/build-scratch/gdrcopy; \
    cd /var/build-scratch/gdrcopy; \
    CUDA=/usr/local/cuda make lib_install; \
    ldconfig; \
    rm -rf /var/build-scratch/gdrcopy

# -----------------------------------------------------------------------------
# PyTorch + NCCL 2.30.4 via pip (DeepEP V2 upstream README recommends this
# over building NCCL from source, so the JIT can find headers automatically)
#
# Pinned to torch==2.9.1 -- the version resolved and proven during our
# sglang overlay build on 2026-04-24 (see bench/logs/sglang-build-
# 20260424T230559Z.log). An `==` pin is required here because DeepEP V2's
# `_C.so` is built against a specific torch C++ ABI; a silent bump to
# torch 3.0 on download.pytorch.org/whl/cu129 would break JIT loads
# across every consumer overlay. Both the cu129 and cu128 indexes carry
# the 2.9.1+cu12x wheel (verified 2026-05-05).
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir --break-system-packages \
      "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu129 \
 || pip install --no-cache-dir --break-system-packages \
      "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128

# numpy<2 is required at runtime by torch.distributed.all_gather_object
# (DeepEP V2's ElasticBuffer.__init__ calls it from get_nccl_comm_handle).
# Without numpy<2 the NCCL comm handle exchange raises "Numpy is not available".
RUN pip install --no-cache-dir --break-system-packages \
      "numpy<2"

# Purge the apt-installed NCCL 2.26.x that the CUDA base image ships. Left in
# place, /lib/x86_64-linux-gnu/libnccl.so.2 (2.26.5) shadows the pip wheel's
# 2.30.4 via ld.so. DeepEP V2 requires >=2.30.4 (Gin backend API).
RUN set -eux; \
    apt-get remove --purge -y libnccl2 libnccl-dev 2>/dev/null || true; \
    apt-mark hold libnccl2 libnccl-dev 2>/dev/null || true; \
    rm -f /lib/x86_64-linux-gnu/libnccl.so* /usr/lib/x86_64-linux-gnu/libnccl.so*; \
    ldconfig

ARG NVIDIA_NCCL_PIN
RUN set -eux; \
    echo "[wave9] NVIDIA_NCCL_PIN=${NVIDIA_NCCL_PIN}"; \
    pip install --no-cache-dir --break-system-packages --no-deps \
      "${NVIDIA_NCCL_PIN}"; \
    NCCL_LIB="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    echo "NCCL_LIB=${NCCL_LIB}"; \
    test -n "${NCCL_LIB}"; \
    ls -la "${NCCL_LIB}"; \
    SO2="$(ls "${NCCL_LIB}"/libnccl.so.* 2>/dev/null | head -1)"; \
    echo "SO2=${SO2}"; \
    test -n "${SO2}"; \
    ln -sf "$(basename "${SO2}")" "${NCCL_LIB}/libnccl.so"; \
    ls -la "${NCCL_LIB}/libnccl.so" "${NCCL_LIB}/libnccl.so.2" || true; \
    echo "${NCCL_LIB}" > /etc/ld.so.conf.d/aa-nvidia-nccl.conf; \
    ldconfig; \
    ldconfig -p | grep 'libnccl.so' | head -3

# NVSHMEM pip wheel for the legacy link path. DeepEP V2's setup.py still links
# against libnvshmem_host.so even when the primary runtime is NCCL Gin
# (upstream setup.py has a TODO: "make NVSHMEM and legacy optional").
# Required >=3.3.9 per DeepEP docs/nvshmem.md on the V2 branch.
# The wheel only ships libnvshmem_host.so.X (versioned), but DeepEP's link
# line uses `-l:libnvshmem_host.so` (unversioned). Create the symlink.
RUN pip install --no-cache-dir --break-system-packages \
      "nvidia-nvshmem-cu12==3.3.9" \
 && NVSHMEM_LIB="$(find /usr/local/lib /usr/lib -path '*/nvidia/nvshmem/lib' -type d 2>/dev/null | head -1)" \
 && test -n "${NVSHMEM_LIB}" && test -d "${NVSHMEM_LIB}" \
 && ls "${NVSHMEM_LIB}" \
 && HOST_SO="$(ls "${NVSHMEM_LIB}"/libnvshmem_host.so.* | head -1)" \
 && ln -sf "${HOST_SO}" "${NVSHMEM_LIB}/libnvshmem_host.so" \
 && echo "${NVSHMEM_LIB}" > /etc/ld.so.conf.d/zz-nvshmem.conf \
 && ldconfig

# -----------------------------------------------------------------------------
# aws-ofi-nccl built from master, pinned to commit 6e504db3403931cde (2026-04-24):
# "gin: Size active_put_signal to full sequence number space". Official upstream
# fix for the 128-slot GIN ring overflow that caused DeepEP V2 dispatch to hang
# with "received count: 0" on AWS EFA once num_allocated_qps >= 5. Supersedes
# our earlier out-of-tree ring-size patch.
#
# Must be built AFTER the nvidia-nccl-cu12 pip install (see NVIDIA_NCCL_PIN)
# so --with-nccl finds matching 2.30.4 headers. The aws-ofi-nccl configure
# script auto-picks up whichever nvidia-nccl-cu* wheel is installed, so the
# cu12 switch in Wave 9 needs no separate change here.
# The plugin's ncclGinPlugin_v11/v12 symbols must
# match the NCCL it is loaded alongside, or DeepEP's railedGinType check
# (see csrc/kernels/backend/nccl.cu in DeepEP V2) fails.
#
# Installed alongside the installer-bundled NGC plugin at /opt/amazon/ofi-nccl.
# The runtime NCCL_NET_PLUGIN env points at /opt/amazon (NGC path) because
# it is built --disable-nccl-net-library and therefore NCCL-version-agnostic;
# /opt/aws-ofi-nccl is retained for overlays that need --with-nccl behaviour.
# -----------------------------------------------------------------------------
ARG AWS_OFI_NCCL_SHA=6e504db3403931cde43a2335adcc73fbc69cccac
RUN set -eux; \
    git clone https://github.com/aws/aws-ofi-nccl.git /var/build-scratch/aws-ofi-nccl; \
    cd /var/build-scratch/aws-ofi-nccl; \
    git checkout "${AWS_OFI_NCCL_SHA}"; \
    ./autogen.sh; \
    NCCL_INCLUDE_DIR="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/include' -type d 2>/dev/null | head -1)"; \
    NCCL_LIB_DIR_OFI="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    test -n "${NCCL_INCLUDE_DIR}" && test -n "${NCCL_LIB_DIR_OFI}"; \
    NCCL_ROOT="$(dirname "${NCCL_INCLUDE_DIR}")"; \
    CPPFLAGS="-I/opt/amazon/efa/include -I/opt/amazon/efa/include/rdma -I/usr/local/cuda/include -I${NCCL_INCLUDE_DIR}" \
    LDFLAGS="-L/opt/amazon/efa/lib -L/opt/amazon/efa/lib64 -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -L${NCCL_LIB_DIR_OFI}" \
    ./configure \
      --prefix=/opt/aws-ofi-nccl \
      --with-libfabric=/opt/amazon/efa \
      --with-cuda=/usr/local/cuda \
      --with-nccl="${NCCL_ROOT}" \
      --enable-platform-aws \
      --disable-tests; \
    make -j"$(nproc)"; \
    make install; \
    nm -D /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -q ncclGinPlugin; \
    echo /opt/aws-ofi-nccl/lib > /etc/ld.so.conf.d/zz-aws-ofi-nccl.conf; \
    ldconfig; \
    rm -rf /var/build-scratch/aws-ofi-nccl

# -----------------------------------------------------------------------------
# DeepEP V2 with PR #612 (auto-QP cap + get_rdma_gbs EFA fast path + dispatch
# kScaleoutUpdateInterval 3 -> 16). See patches/README.md for the three-patch
# rationale and upstream PR link.
#
# We clone the pre-patched branch rather than `git am` the three patches from
# ./patches/ for two reasons: (1) the branch is the exact tree tested in the
# reference 2-node p5en.48xlarge H200 EFA run that produced 740us D+C p50,
# which the patches are extracted from; (2) it avoids any risk of `git am`
# conflicts against a moving upstream base (the patches were generated
# against b306af0 and are byte-for-byte identical with the branch tip).
# When PR #612 merges upstream, flip DEEPEP_FORK/DEEPEP_BRANCH to vanilla and
# drop the ./patches/ directory.
# -----------------------------------------------------------------------------
# Canonical values live in ../pins.env at the repo root. Both the GHA workflow
# (.github/workflows/{build-and-push,test-build}.yml) and ci/buildspec.yml
# source that file and pass the values via --build-arg, which overrides these
# defaults. The hardcoded values below are retained as a legacy-safety
# fallback so `docker build .` still works in an emergency.
ARG DEEPEP_FORK=https://github.com/dmvevents/DeepEP-1.git
ARG DEEPEP_BRANCH=aws-efa-auto-qp-cap-v2
ARG DEEPEP_SHA=146cc356aa00c39ac1590c05775e05b0f031e70c

RUN set -eux; \
    git clone --recurse-submodules "${DEEPEP_FORK}" /opt/DeepEP; \
    cd /opt/DeepEP; \
    git checkout "${DEEPEP_BRANCH}"; \
    # Verify the pinned SHA is at the branch tip for reproducibility.
    test "$(git rev-parse HEAD)" = "${DEEPEP_SHA}" \
      || (echo "FATAL: DeepEP HEAD $(git rev-parse HEAD) != pinned ${DEEPEP_SHA}" && exit 1); \
    export TORCH_CUDA_ARCH_LIST="9.0"; \
    export MAX_JOBS="$(nproc)"; \
    # DeepEP V2 setup.py adds -Wl,-rpath for nccl but not a matching -L, so the
    # linker cannot resolve `-l:libnccl.so`. Inject the path via LIBRARY_PATH.
    NCCL_LIB_DIR="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    test -n "${NCCL_LIB_DIR}"; \
    export LIBRARY_PATH="${NCCL_LIB_DIR}:${LIBRARY_PATH:-}"; \
    pip3 install --no-build-isolation --break-system-packages -e .; \
    python3 -c "import deep_ep; print('DeepEP V2 OK:', deep_ep.ElasticBuffer)"

# -----------------------------------------------------------------------------
# Preflight harness + runtime env
# -----------------------------------------------------------------------------
COPY preflight.sh /preflight.sh
RUN chmod +x /preflight.sh

# Runtime env matches the reference bench_elastic_ep.sh from the V2 recipe.
# NCCL_NET_PLUGIN points at the --build-ngc install path (installer-bundled,
# NCCL-version-agnostic). The PR #612 ring fix in aws-ofi-nccl 6e504db
# removes the need for the old OFI_NCCL_GIN_MAX_REQUESTS knob but it is
# retained as defence in depth for any older plugin loaded in override.
ENV NCCL_NET_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so \
    NCCL_GIN_TYPE=2 \
    NCCL_GIN_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    NCCL_CUMEM_HOST_ENABLE=1 \
    NCCL_NVLS_ENABLE=0 \
    NCCL_IGNORE_DISABLED_P2P=1 \
    FI_PROVIDER=efa \
    FI_EFA_USE_DEVICE_RDMA=1 \
    FI_EFA_ENABLE_SHM_TRANSFER=0 \
    FI_EFA_FORK_SAFE=1 \
    OFI_NCCL_PROTOCOL=RDMA \
    OFI_NCCL_GIN_MAX_REQUESTS=512 \
    DEEP_EP_BACKEND=nccl \
    EP_EFA_MAX_QPS=2 \
    EP_EFA_RDMA_GBS=25.0

WORKDIR /opt

LABEL org.opencontainers.image.title="deepep-v2-efa-base" \
      org.opencontainers.image.description="DeepEP V2 (NCCL Gin) on AWS EFA, SM 9.0a base image for H100/H200" \
      org.opencontainers.image.source="https://github.com/antonai-work/deepep-v2-efa-base" \
      org.opencontainers.image.licenses="Apache-2.0" \
      deepep.branch="aws-efa-auto-qp-cap-v2" \
      deepep.sha="146cc356aa00c39ac1590c05775e05b0f031e70c" \
      aws-ofi-nccl.sha="6e504db3403931cde43a2335adcc73fbc69cccac" \
      efa.installer="1.48.0"
