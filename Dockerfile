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
#                                            Wave 10: we wanted
#                                            nvidia/cuda:13.0-devel-ubuntu24.04
#                                            but it is not yet on Docker Hub
#                                            (verified 2026-05-06). The cu13
#                                            runtime components land via pip
#                                            wheels on top of the cu12.9 devel
#                                            build-OS. Flip this to 13.0-devel
#                                            when NVIDIA publishes it.
#   CUDA         12.9.0                      NVIDIA registry. Provides nvcc
#                                            (build-time) + libcudart.so.12
#                                            (runtime). Wave 10 retry 1:
#                                            torch is cu129 (libcudart.so.12
#                                            compatible) and NCCL is cu13
#                                            (dlopens CUDA runtime so no
#                                            static libcudart.so.13 dep).
#                                            True cu13 runtime unification
#                                            is blocked on NVIDIA publishing
#                                            cu13 devtoolkit wheels.
#   EFA user     1.48.0                      efa-installer.amazonaws.com tarball
#                                            installed via --build-ngc path
#                                            (libfabric1-aws + libnccl-ofi-ngc-v3)
#   NCCL         2.30.4                      pip wheel `nvidia-nccl-cu13>=2.30.4`
#                                            (Wave 10: NCCL wheel flipped back
#                                            to cu13 because cu12 NCCL 2.30.4
#                                            exposes `ncclTeamWorld` (weak
#                                            symbol, verified via `nm -D` on
#                                            the cu12 wheel 2026-05-06) BUT
#                                            does NOT export
#                                            `ncclCommProperties`, which
#                                            DeepEP V2 PR #612's
#                                            csrc/kernels/backend/nccl.cu
#                                            calls. cu13 NCCL exposes both
#                                            symbols. Wave 9c's
#                                            `ImportError: undefined symbol:
#                                            ncclTeamWorld` was a secondary
#                                            artefact of the same GIN ABI
#                                            gap. The wheel is installed
#                                            under torch cu129; Python-level
#                                            ldconfig resolution gives
#                                            libnccl.so.2 -> cu13 wheel, and
#                                            DeepEP's link-time `-l:libnccl.so`
#                                            resolves to the cu13 NCCL. torch
#                                            itself never dlopens libnccl
#                                            directly - torch.distributed.
#                                            ProcessGroupNCCL.init_process
#                                            dlopens whatever libnccl.so.2
#                                            ld.so resolves, so the cu13 NCCL
#                                            carries for torch's NCCL work
#                                            too. Runtime libcudart is cu12
#                                            (from the base image and torch
#                                            cu129 wheel); NCCL's internal
#                                            libcudart.so.13 references are
#                                            resolved through weak linkage.)
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
#                                            Wave 10 retry 1: kept on cu12 to
#                                            match torch cu129's libcudart.so.12
#                                            so the DeepEP `_C.so` is not forced
#                                            to load libcudart.so.13 at import
#                                            time. Only NCCL flips to cu13.
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
#   PyTorch      torch==2.11.0+cu129 from
#                https://download.pytorch.org/whl/cu129 (Wave 10 retry 1).
#                Previously torch==2.9.1+cu129 (Wave 9 cu12 stack). Wave 10
#                originally targeted torch==2.11.0+cu130 but NVIDIA has not
#                yet published cu13 developer toolkit wheels (nvidia-cuda-
#                nvcc-cu13 == 0.0.1 placeholder as of 2026-05-06), and
#                Docker Hub does not yet ship nvidia/cuda:13.0-devel-
#                ubuntu24.04 either. Without a cu13 nvcc, torch
#                cpp_extension's `cuda_ver.major != torch_cuda_version.major`
#                check hard-fails DeepEP's `pip install -e .`.
#                Staying on cu129 torch aligns with the base image nvcc and
#                lets DeepEP build; the NCCL wheel separately flips to cu13
#                (see NCCL entry) to get the GIN ABI symbols DeepEP needs.
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

# Wave 10 retry 1: partial cu13 migration. NCCL pin sourced from pins.env
# at repo root; CI passes this via --build-arg. Default fallback so
# `docker build .` works standalone. RATIONALE: the god-mode consensus
# recommended a full cu13 stack. cu13 NCCL is achievable today via the
# published nvidia-nccl-cu13 wheel. cu13 torch + cu13 nvcc are not: torch
# cu130 wheels exist but NVIDIA has not published cu13 devtoolkit wheels
# (nvidia-cuda-nvcc-cu13 is a 0.0.1 placeholder), so DeepEP's
# CUDAExtension build fails torch's cpp_extension major-version check.
# This commit keeps torch on cu129 (matching the cu12.9-devel base nvcc)
# and flips only NCCL to cu13 to get GIN ABI under a fully-buildable
# toolchain.
ARG NVIDIA_NCCL_PIN=nvidia-nccl-cu13>=2.30.4
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu129

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
# Wave 10 retry 1: pinned to torch==2.11.0+cu129 (latest cu129 wheel on
# https://download.pytorch.org/whl/cu129/torch/, verified 2026-05-06 via
# `pip index versions torch`). cu130 was the initial Wave 10 target but
# NVIDIA has not published cu13 developer-toolkit wheels yet, and the
# torch cpp_extension major-version check hard-fails DeepEP's
# `pip install -e .` when torch is cu130 but nvcc is 12.9.
# An `==` pin is required here because DeepEP V2's `_C.so` is built
# against a specific torch C++ ABI; a silent bump on the cu129 index
# would break JIT loads across every consumer overlay.
# -----------------------------------------------------------------------------
ARG TORCH_INDEX
RUN pip install --no-cache-dir --break-system-packages \
      "torch==2.11.0" --index-url "${TORCH_INDEX}"

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
# --upgrade is required: torch==2.11.0+cu129 transitively installs
# nvidia-nccl-cu12 at a version pinned inside the torch wheel (which may
# be lower than 2.30.4); without --upgrade, pip reports "Requirement
# already satisfied" and leaves the torch-bundled NCCL in place. DeepEP
# V2's csrc/kernels/backend/nccl.cu calls `ncclTeamWorld`,
# `ncclCommQueryProperties`, and `ncclGinBarrierCreateRequirement` which
# only ship in NCCL >= 2.30.4.
#
# When NVIDIA_NCCL_PIN flips to cu13 (Wave 10 retry 1 default), we must
# also uninstall the torch-bundled nvidia-nccl-cu12 first so only one
# `site-packages/nvidia/nccl/` directory exists. Both wheels land at the
# same path; the later install wins, but uninstall-first is clearer and
# avoids any partial-overlap issues.
#
# Version detection is made cu12/cu13-agnostic: extract the wheel name
# from the NVIDIA_NCCL_PIN build arg so the same block works across
# cu12/cu13 flips without hardcoding `pip show nvidia-nccl-cuNN`.
RUN set -eux; \
    echo "[wave10] NVIDIA_NCCL_PIN=${NVIDIA_NCCL_PIN}"; \
    pkg_name="$(echo "${NVIDIA_NCCL_PIN}" | sed -e 's/[<>=!].*//' -e 's/[[:space:]]*//g')"; \
    if [ "${pkg_name}" = "nvidia-nccl-cu13" ]; then \
      echo "[wave10] flipping to cu13 NCCL - uninstalling torch-bundled cu12 NCCL first"; \
      pip uninstall -y --break-system-packages nvidia-nccl-cu12 2>/dev/null || true; \
    fi; \
    pip install --no-cache-dir --break-system-packages --no-deps --upgrade \
      "${NVIDIA_NCCL_PIN}"; \
    echo "[wave10] nccl pip pkg: ${pkg_name}"; \
    pip show "${pkg_name}" | grep -E '^Version:' | awk '{print "[wave10] installed '"${pkg_name}"': " $2}'; \
    installed_ver="$(pip show "${pkg_name}" | awk '/^Version:/ {print $2}')"; \
    python3 -c "import sys; v=sys.argv[1]; assert tuple(map(int, v.split('.'))) >= (2,30,4), 'nccl '+v+' below 2.30.4 floor'; print('[wave10] floor OK')" "${installed_ver}"; \
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
    ldconfig -p | grep 'libnccl.so' | head -3; \
    echo "[wave10] probing GIN ABI symbols in the installed NCCL wheel..."; \
    gin_syms="$(nm -D --defined-only "${SO2}" 2>/dev/null | grep -Eo 'ncclTeamWorld|ncclCommQueryProperties|ncclGinBarrierCreateRequirement' | sort -u || true)"; \
    echo "[wave10] GIN symbols found: ${gin_syms:-NONE}"; \
    for sym in ncclTeamWorld ncclCommQueryProperties ncclGinBarrierCreateRequirement; do \
      echo "${gin_syms}" | grep -q "${sym}" \
        || (echo "[wave10] FATAL: NCCL wheel missing GIN ABI symbol ${sym} (Wave 9c signature)" && exit 1); \
    done; \
    echo "[wave10] GIN ABI OK (all 3 required symbols present)"

# NVSHMEM pip wheel for the legacy link path. DeepEP V2's setup.py still links
# against libnvshmem_host.so even when the primary runtime is NCCL Gin
# (upstream setup.py has a TODO: "make NVSHMEM and legacy optional").
# Required >=3.3.9 per DeepEP docs/nvshmem.md on the V2 branch. Wave 10
# retry 1 keeps NVSHMEM on cu12 (not cu13) to match torch cu129's
# libcudart.so.12 so DeepEP's `_C.so` is not forced to load
# libcudart.so.13 at import time. Only NCCL flips to cu13. The wheel
# only ships libnvshmem_host.so.X (versioned), but DeepEP's link line
# uses `-l:libnvshmem_host.so` (unversioned). Create the symlink.
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
# Must be built AFTER the nvidia-nccl-cu13 pip install (see NVIDIA_NCCL_PIN)
# so --with-nccl finds matching cu13 2.30.4 headers. The aws-ofi-nccl
# configure script auto-picks up whichever nvidia-nccl-cu* wheel is
# installed (by path glob under nvidia/nccl), so the cu12->cu13 flip in
# Wave 10 needs no separate change here.
#
# The plugin's ncclGinPlugin_v11/v12 symbols must match the NCCL it is
# loaded alongside, or DeepEP's railedGinType check (see
# csrc/kernels/backend/nccl.cu in DeepEP V2) fails.
#
# Wave 10: because the build-OS is still cu12.9-devel, libcudart.so.12
# remains the compile target for non-CUDA-side C. We spot-check which
# libcudart the plugin actually links against after build to catch any
# mismatch between intent (cu13) and outcome.
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
    echo "[wave10] aws-ofi-nccl libcudart linkage:"; \
    ldd /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep libcudart || echo "[wave10] (no libcudart linkage - plugin does not call CUDA runtime directly)"; \
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
