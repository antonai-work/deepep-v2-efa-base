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
#                                            Wave 12: per the plan guidance
#                                            we keep the BUILD OS on cu12.9
#                                            even though nvidia/cuda:13.0.0-
#                                            devel-ubuntu24.04 now exists on
#                                            Docker Hub. cu13 RUNTIME libs
#                                            (libcudart.so.13, nvcc 13) land
#                                            via the pip wheel chain:
#                                            torch==2.11.0+cu130 Requires-Dist
#                                            pulls `cuda-toolkit==13.0.2`
#                                            which brings
#                                            nvidia-cuda-runtime==13.0.96 and
#                                            nvidia-cuda-nvcc==13.0.88. The
#                                            Wave 10 comment that "cu13 nvcc
#                                            is blocked" was based on probing
#                                            the legacy `nvidia-cuda-nvcc-cu13`
#                                            package (0.0.1 placeholder); the
#                                            real cu13 nvcc now ships via the
#                                            unified `cuda-toolkit` meta and
#                                            this block installs it correctly.
#   CUDA         12.9.0 (build OS) +         Base image provides the cu12.9
#                13.0 (runtime via pip)      compile toolchain that is kept
#                                            only as a fallback. cu13
#                                            nvcc/libcudart flow in through
#                                            pip's site-packages and are made
#                                            visible via PATH and
#                                            LD_LIBRARY_PATH before the
#                                            DeepEP build step so torch's
#                                            cpp_extension major-version
#                                            check passes (torch.version.cuda
#                                            == '13.0' matches the nvcc 13
#                                            from the wheel).
#   EFA user     1.48.0                      efa-installer.amazonaws.com tarball
#                                            installed via --build-ngc path
#                                            (libfabric1-aws + libnccl-ofi-ngc-v3)
#   NCCL         2.30.4                      pip wheel `nvidia-nccl-cu13>=2.30.4`
#                                            (Wave 10: NCCL flipped to cu13
#                                            for the DeepEP V2 GIN ABI
#                                            symbols: ncclTeamWorld,
#                                            ncclCommQueryProperties,
#                                            ncclGinBarrierCreateRequirement.
#                                            Wave 12: torch now also cu13 so
#                                            every runtime component links
#                                            libcudart.so.13 uniformly - no
#                                            more TLS split across the
#                                            torch/vllm._C/deep_ep boundary
#                                            which was the root cause of
#                                            Wave 11's c10::cuda::SetDevice
#                                            -112 crash at MoE dispatch.)
#                                            apt libnccl2 2.26.x is purged
#   aws-ofi-nccl 206c02c478c6d724af09c3cbca59c06863a0b9c0 (2026-05-06)
#                                            github.com/aws/aws-ofi-nccl
#                                            "gin: Size active_put_signal to
#                                            full sequence number space"
#                                            (upstream fix superseding the
#                                            earlier env-tunable ring-size
#                                            workaround). Built with
#                                            --enable-platform-aws against the
#                                            NCCL 2.30.4 pip wheel headers.
#                                            Wave 16: bumped to 206c02c4 (May 6)
#                                            + baked-in p5.48xlarge topology
#                                            XML (122 nodes, fixes NCCL
#                                            "too many XML nodes (max 256)"
#                                            on HyperPod P5 during auto-discovery).
#   GDRCopy      v2.5.1                      github.com/NVIDIA/gdrcopy
#   NVSHMEM      nvidia-nvshmem-cu13>=3.4.5  pip wheel. Only libnvshmem_host.so
#                                            is linked at DeepEP build time;
#                                            runtime does not call NVSHMEM
#                                            when DEEP_EP_BACKEND=nccl.
#                                            Wave 12: flipped to cu13 to match
#                                            torch cu130's libcudart.so.13
#                                            now that torch is on cu130.
#                                            Keeping cu12 NVSHMEM here would
#                                            reintroduce the libcudart TLS
#                                            split that Wave 12 is explicitly
#                                            closing (torch cu130 brings
#                                            libcudart.so.13 so NVSHMEM must
#                                            link it too). The cu13 wheel
#                                            torch drags in is 3.4.5 per
#                                            torch 2.11.0+cu130 metadata.
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
#   PyTorch      torch==2.11.0+cu130 from
#                https://download.pytorch.org/whl/cu130 (Wave 12).
#                Wave 12 closes Wave 10's partial cu13 migration: the torch
#                cu130 wheel brings the full cu13 runtime chain through its
#                Requires-Dist (`cuda-toolkit==13.0.2` -> nvidia-cuda-runtime
#                ==13.0.96 + nvidia-cuda-nvcc==13.0.88) so the Wave 10 concern
#                about missing cu13 nvcc wheels no longer applies (the legacy
#                `nvidia-cuda-nvcc-cu13` is still a 0.0.1 placeholder, but the
#                new unified `nvidia-cuda-nvcc` package is what torch 2.11
#                installs). DeepEP's `pip install -e .` succeeds because
#                torch.cpp_extension finds the wheel-provided nvcc and
#                torch.version.cuda == '13.0' matches.
#                An `==` pin is required here because DeepEP V2's `_C.so` is
#                built against a specific torch C++ ABI; a silent bump on the
#                cu130 index would break JIT loads across every consumer
#                overlay.
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

# Wave 12: complete cu13 migration. NCCL pin + torch index sourced from
# pins.env at repo root; CI passes both via --build-arg. Default fallbacks
# below let `docker build .` work standalone. Wave 10 flipped NCCL to cu13
# but left torch on cu129; Wave 11 proved the resulting libcudart TLS
# split still crashes vllm._C at MoE dispatch with
# c10::cuda::SetDevice(-112). Wave 12 flips torch to cu130 so the whole
# stack links libcudart.so.13 uniformly. cu13 nvcc now ships via the
# unified `cuda-toolkit==13.0.2` meta package that torch 2.11.0+cu130
# Requires-Dist pulls in (not the legacy nvidia-cuda-nvcc-cu13 placeholder
# Wave 10 probed), so DeepEP's `pip install -e .` builds cleanly.
ARG NVIDIA_NCCL_PIN=nvidia-nccl-cu13>=2.30.4
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130

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

# Wave 12: CUDA_HOME and LD_LIBRARY_PATH are set here as a fallback only;
# the DeepEP build step later re-points CUDA_HOME to the pip-installed cu13
# toolchain for the duration of that RUN. Runtime cu13 libcudart resolution
# comes from ldconfig (see /etc/ld.so.conf.d/aa-nvidia-cuda-runtime-cu13.conf
# written later), so LD_LIBRARY_PATH here intentionally does NOT include
# /usr/local/cuda/lib64 first - that would shadow the cu13 libcudart with
# the base image's cu12 libcudart. We keep /usr/local/cuda/lib64 at the
# end of LD_LIBRARY_PATH for stubs/build-time linkage only.
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/opt/amazon/efa/bin:${PATH} \
    PKG_CONFIG_PATH=/opt/amazon/efa/lib/pkgconfig:/opt/amazon/efa/lib64/pkgconfig \
    LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/efa/lib64:/opt/amazon/ofi-nccl/lib

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
# PyTorch + cu13 runtime chain via pip (Wave 12).
#
# torch==2.11.0+cu130 from https://download.pytorch.org/whl/cu130 is the
# first cu130 stable wheel. Its Requires-Dist pulls cuda-toolkit==13.0.2
# which in turn pulls nvidia-cuda-runtime==13.0.96 (libcudart.so.13) and
# nvidia-cuda-nvcc==13.0.88 (nvcc 13). DeepEP's `pip install -e .` then
# uses the wheel-provided nvcc; torch.cpp_extension's `torch.version.cuda`
# is '13.0' and the nvcc major matches, so the major-version check passes.
#
# An `==` pin is required here because DeepEP V2's `_C.so` is built
# against a specific torch C++ ABI; a silent bump on the cu130 index
# would break JIT loads across every consumer overlay.
# -----------------------------------------------------------------------------
ARG TORCH_INDEX
RUN pip install --no-cache-dir --break-system-packages \
      "torch==2.11.0" --index-url "${TORCH_INDEX}"

# torch 2.11.0+cu130's Requires-Dist pulls
#   cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,
#                cusparse,nvjitlink,nvrtc,nvtx]==13.0.2
# which gives us libcudart.so.13 via `cudart` and libnvrtc via `nvrtc` but
# does NOT include the `nvcc` extra.
#
# Wave 12 empirical gotchas found during CodeBuild iterations:
#
# (1) cuda-toolkit==13.0.2 -> nvcc 13.0.88 has a bug where its ptxas
#     cannot consume the PTX 9.2 that its nvcc emits when compiling for
#     sm_90a (verified 2026-05-06 attempt 1: "ptxas fatal : Unsupported
#     .version 9.2; current version is '9.0'"). cu13.2.x ships nvcc
#     +ptxas that correctly handle PTX 9.2 for sm_90a.
# (2) cccl's cuda_toolkit.h runs a strict nvcc.minor == cudart.minor
#     check and errors out "CUDA compiler and CUDA toolkit headers are
#     incompatible" when the minors differ (attempt 2: nvcc 13.2,
#     cudart 13.0).
#
# Resolution: install the full cu13.2.x set directly (no cuda-toolkit
# metapackage, no auto-downgrades), using --no-deps so torch's own
# Requires-Dist pin on cudart 13.0.96.* doesn't conflict. cudart ABI is
# stable across 13.0<->13.2 minor versions so the 13.2.75 runtime is a
# drop-in replacement for torch's bundled 13.0.96. The set includes:
#   nvcc           (compiler driver, invokes cicc+ptxas+fatbinary)
#   nvvm           (cicc + libdevice, for CUDA-C -> PTX)
#   cuda-crt       (device crt for -rdc=true device link step)
#   cuda-cccl      (thrust/cub headers, DeepEP uses cub::WarpReduce)
#   cuda-runtime   (libcudart.so.13 at 13.2.75, upgrades torch's 13.0.96)
RUN pip install --no-cache-dir --break-system-packages --no-deps \
      "nvidia-cuda-nvcc>=13.2.78,<13.3" \
      "nvidia-nvvm>=13.2.78,<13.3" \
      "nvidia-cuda-crt>=13.2.78,<13.3" \
      "nvidia-cuda-cccl>=13.2.75,<13.3" \
      "nvidia-cuda-runtime>=13.2.75,<13.3"

# Expose the cu13 runtime + nvcc that torch and cuda-toolkit just pulled in.
# site-packages is where pip writes the nvidia-cuda-runtime wheel's
# libcudart.so.13 and the nvidia-cuda-nvcc wheel's nvcc binary. Put them on
# LD_LIBRARY_PATH/PATH and
# register libcudart's dir with ldconfig so downstream compiled .so files
# (DeepEP's _C.so, aws-ofi-nccl's libnccl-net-ofi.so) resolve them at
# link time AND at runtime. This is the key Wave 12 mechanism: without this
# step, /usr/local/cuda/bin/nvcc (cu12.9) shadows the cu13 nvcc and
# torch.cpp_extension's major-version check fails with
# "detected CUDA version (12.9) mismatches version used to compile PyTorch
# (13.0)" when DeepEP builds.
RUN set -eux; \
    CU13_ROOT="$(find /usr/local/lib /usr/lib -type d -path '*/nvidia/cu13' 2>/dev/null | head -1)"; \
    echo "[wave12] cu13 root: ${CU13_ROOT}"; \
    test -n "${CU13_ROOT}" && test -d "${CU13_ROOT}"; \
    test -d "${CU13_ROOT}/lib"; \
    test -d "${CU13_ROOT}/bin"; \
    test -d "${CU13_ROOT}/include"; \
    test -x "${CU13_ROOT}/bin/nvcc"; \
    test -f "${CU13_ROOT}/lib/libcudart.so.13"; \
    test -d "${CU13_ROOT}/include/cccl"; \
    test -d "${CU13_ROOT}/nvvm/bin"; \
    test -x "${CU13_ROOT}/nvvm/bin/cicc"; \
    test -d "${CU13_ROOT}/nvvm/libdevice"; \
    test -f "${CU13_ROOT}/nvvm/libdevice/libdevice.10.bc"; \
    echo "[wave12] cu13 lib dir contents:"; \
    ls -la "${CU13_ROOT}/lib" | head -30; \
    echo "${CU13_ROOT}/lib" > /etc/ld.so.conf.d/aa-nvidia-cuda-cu13.conf; \
    ldconfig; \
    echo "[wave12] ldconfig libcudart entries:"; \
    ldconfig -p | grep -E 'libcudart\.so' | head -5; \
    "${CU13_ROOT}/bin/nvcc" --version; \
    "${CU13_ROOT}/bin/nvcc" --version | grep -Eq 'release 13\.' \
        || (echo "[wave12] FATAL: wheel-provided nvcc is not cu13" && exit 1); \
    # Verify ptxas can consume the PTX emitted by its own nvcc for sm_90a
    # (Wave 12 attempt 1 regression: 13.0.88 nvcc+ptxas disagreed on PTX
    # 9.2 support; 13.2.78 correctly handles it). This is a compile-time
    # sanity check to catch the regression before running the full DeepEP
    # build.
    printf '.version 9.2\n.target sm_90a\n.address_size 64\n.visible .entry foo() { ret; }\n' > /tmp/wave12-ptx-test.ptx; \
    "${CU13_ROOT}/bin/ptxas" -arch=sm_90a /tmp/wave12-ptx-test.ptx -o /tmp/wave12-ptx-test.cubin 2>&1 \
        && echo "[wave12] ptxas handles PTX 9.2 + sm_90a" \
        || (echo "[wave12] FATAL: ptxas cannot assemble PTX 9.2 for sm_90a - upgrade cu13 nvcc" && exit 1); \
    rm -f /tmp/wave12-ptx-test.ptx /tmp/wave12-ptx-test.cubin; \
    # DeepEP setup.py hardcodes `/usr/local/cuda/include/cccl` so make
    # that path resolve to the cu13 cccl headers via symlink. This is
    # the minimum-intrusive way to avoid patching DeepEP's setup.py.
    mkdir -p /usr/local/cuda/include; \
    ln -sfn "${CU13_ROOT}/include/cccl" /usr/local/cuda/include/cccl; \
    ls -la /usr/local/cuda/include/cccl; \
    # Wave 12: the cu13 runtime wheel ships libcudart.so.13 without an
    # unversioned libcudart.so symlink. ld at link time needs
    # `libcudart.so` for `-lcudart`. Create the symlink so DeepEP's
    # final link step resolves. Ditto for libcudadevrt and libcuda stubs
    # if present.
    for versioned in "${CU13_ROOT}/lib"/libcudart.so.*; do \
      [ -f "${versioned}" ] || continue; \
      ln -sfn "$(basename "${versioned}")" "${CU13_ROOT}/lib/libcudart.so"; \
      break; \
    done; \
    ls -la "${CU13_ROOT}/lib/libcudart.so" "${CU13_ROOT}/lib/libcudart.so.13" || true; \
    # The `-lcuda` (libcuda.so driver API stub) also doesn't ship in
    # this wheel - DeepEP passes -lcuda for CUDA Driver API calls in
    # cuda_driver.cu. Use the base image's stub at /usr/local/cuda/
    # lib64/stubs/libcuda.so (shipped by nvidia/cuda:12.9.0-devel-
    # ubuntu24.04). Keep that stub dir on LIBRARY_PATH for DeepEP build.
    test -f /usr/local/cuda/lib64/stubs/libcuda.so || \
      (echo "[wave12] FATAL: no libcuda.so stub in base image" && exit 1); \
    echo "CU13_ROOT=${CU13_ROOT}"                 >  /etc/wave12-cuda13.env; \
    echo "PYTORCH_CUDA_NVCC_DIR=${CU13_ROOT}/bin" >> /etc/wave12-cuda13.env; \
    echo "PYTORCH_CUDA_RUNTIME_DIR=${CU13_ROOT}/lib" >> /etc/wave12-cuda13.env; \
    echo "PYTORCH_CUDA_ROOT=${CU13_ROOT}"         >> /etc/wave12-cuda13.env; \
    echo "PYTORCH_CUDART_ROOT=${CU13_ROOT}"       >> /etc/wave12-cuda13.env; \
    cat /etc/wave12-cuda13.env

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
# --upgrade is required: torch==2.11.0+cu130 transitively installs
# nvidia-nccl-cu13 at a version pinned inside the torch wheel (2.28.9
# per the torch 2.11.0+cu130 METADATA); without --upgrade, pip reports
# "Requirement already satisfied" and leaves the torch-bundled NCCL in
# place. DeepEP V2's csrc/kernels/backend/nccl.cu calls `ncclTeamWorld`,
# `ncclCommQueryProperties`, and `ncclGinBarrierCreateRequirement` which
# only ship in NCCL >= 2.30.4.
#
# Wave 12: torch cu130 already brings nvidia-nccl-cu13 transitively, so
# we only need to force-upgrade the existing cu13 wheel to >=2.30.4.
# The cu12/cu13 flip logic from Wave 10 is retained as defence in depth
# for a future flip back.
#
# Version detection is made cu12/cu13-agnostic: extract the wheel name
# from the NVIDIA_NCCL_PIN build arg so the same block works across
# cu12/cu13 flips without hardcoding `pip show nvidia-nccl-cuNN`.
RUN set -eux; \
    echo "[wave12] NVIDIA_NCCL_PIN=${NVIDIA_NCCL_PIN}"; \
    pkg_name="$(echo "${NVIDIA_NCCL_PIN}" | sed -e 's/[<>=!].*//' -e 's/[[:space:]]*//g')"; \
    if [ "${pkg_name}" = "nvidia-nccl-cu13" ]; then \
      echo "[wave12] flipping to cu13 NCCL - uninstalling any stray cu12 NCCL first"; \
      pip uninstall -y --break-system-packages nvidia-nccl-cu12 2>/dev/null || true; \
    fi; \
    pip install --no-cache-dir --break-system-packages --no-deps --upgrade \
      "${NVIDIA_NCCL_PIN}"; \
    echo "[wave12] nccl pip pkg: ${pkg_name}"; \
    pip show "${pkg_name}" | grep -E '^Version:' | awk '{print "[wave12] installed '"${pkg_name}"': " $2}'; \
    installed_ver="$(pip show "${pkg_name}" | awk '/^Version:/ {print $2}')"; \
    python3 -c "import sys; v=sys.argv[1]; assert tuple(map(int, v.split('.'))) >= (2,30,4), 'nccl '+v+' below 2.30.4 floor'; print('[wave12] floor OK')" "${installed_ver}"; \
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
    echo "[wave12] probing GIN ABI symbols in the installed NCCL wheel..."; \
    gin_syms="$(nm -D --defined-only "${SO2}" 2>/dev/null | grep -Eo 'ncclTeamWorld|ncclCommQueryProperties|ncclGinBarrierCreateRequirement' | sort -u || true)"; \
    echo "[wave12] GIN symbols found: ${gin_syms:-NONE}"; \
    for sym in ncclTeamWorld ncclCommQueryProperties ncclGinBarrierCreateRequirement; do \
      echo "${gin_syms}" | grep -q "${sym}" \
        || (echo "[wave12] FATAL: NCCL wheel missing GIN ABI symbol ${sym} (Wave 9c signature)" && exit 1); \
    done; \
    echo "[wave12] GIN ABI OK (all 3 required symbols present)"; \
    echo "[wave12] NCCL libcudart linkage:"; \
    ldd "${SO2}" | grep -E 'libcudart\.so' \
      || echo "[wave12] (NCCL does not dlopen libcudart directly)"; \
    if ldd "${SO2}" | grep -qE 'libcudart\.so\.12'; then \
      echo "[wave12] FATAL: NCCL wheel still links libcudart.so.12 - expected libcudart.so.13"; \
      exit 1; \
    fi

# NVSHMEM pip wheel for the legacy link path. DeepEP V2's setup.py still links
# against libnvshmem_host.so even when the primary runtime is NCCL Gin
# (upstream setup.py has a TODO: "make NVSHMEM and legacy optional").
# Required >=3.3.9 per DeepEP docs/nvshmem.md on the V2 branch. Wave 12
# flips NVSHMEM to cu13 to match torch cu130's libcudart.so.13. Leaving
# NVSHMEM on cu12 would reintroduce the libcudart TLS split that Wave 12
# is explicitly closing. The cu13 wheel torch pulls transitively is
# 3.4.5 per torch 2.11.0+cu130 METADATA; our explicit install ensures
# the floor is enforced. Uninstall any cu12 stray first (defence in depth
# in case a downstream overlay or prior layer left one behind). The
# wheel ships libnvshmem_host.so.X (versioned), but DeepEP's link line
# uses `-l:libnvshmem_host.so` (unversioned). Create the symlink.
RUN set -eux; \
    echo "[wave12] NVSHMEM flip to cu13 - uninstall any cu12 NVSHMEM first"; \
    pip uninstall -y --break-system-packages nvidia-nvshmem-cu12 2>/dev/null || true; \
    pip install --no-cache-dir --break-system-packages --upgrade --no-deps \
      "nvidia-nvshmem-cu13>=3.4.5"; \
    NVSHMEM_LIB="$(find /usr/local/lib /usr/lib -path '*/nvidia/nvshmem/lib' -type d 2>/dev/null | head -1)"; \
    test -n "${NVSHMEM_LIB}" && test -d "${NVSHMEM_LIB}"; \
    ls "${NVSHMEM_LIB}"; \
    HOST_SO="$(ls "${NVSHMEM_LIB}"/libnvshmem_host.so.* | head -1)"; \
    ln -sf "${HOST_SO}" "${NVSHMEM_LIB}/libnvshmem_host.so"; \
    echo "${NVSHMEM_LIB}" > /etc/ld.so.conf.d/zz-nvshmem.conf; \
    ldconfig; \
    echo "[wave12] NVSHMEM libcudart linkage:"; \
    ldd "${HOST_SO}" | grep -E 'libcudart\.so' \
      || echo "[wave12] (no direct libcudart linkage)"; \
    if ldd "${HOST_SO}" | grep -qE 'libcudart\.so\.12'; then \
      echo "[wave12] FATAL: NVSHMEM wheel still links libcudart.so.12 - expected libcudart.so.13"; \
      exit 1; \
    fi; \
    echo "[wave12] NVSHMEM cu13 OK"

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
ARG AWS_OFI_NCCL_SHA=206c02c478c6d724af09c3cbca59c06863a0b9c0
RUN set -eux; \
    source /etc/wave12-cuda13.env; \
    test -n "${CU13_ROOT}"; \
    git clone https://github.com/aws/aws-ofi-nccl.git /var/build-scratch/aws-ofi-nccl; \
    cd /var/build-scratch/aws-ofi-nccl; \
    git checkout "${AWS_OFI_NCCL_SHA}"; \
 \
    ./autogen.sh; \
    NCCL_INCLUDE_DIR="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/include' -type d 2>/dev/null | head -1)"; \
    NCCL_LIB_DIR_OFI="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    test -n "${NCCL_INCLUDE_DIR}" && test -n "${NCCL_LIB_DIR_OFI}"; \
    NCCL_ROOT="$(dirname "${NCCL_INCLUDE_DIR}")"; \
    # Wave 12: point at cu13 toolchain for headers/libs so any libcudart
    # linkage that slips in ends up linking libcudart.so.13 (and not the
    # cu12.9 from /usr/local/cuda). The plugin normally has no direct
    # libcudart dep but being explicit costs nothing.
    CPPFLAGS="-I/opt/amazon/efa/include -I/opt/amazon/efa/include/rdma -I${CU13_ROOT}/include -I${NCCL_INCLUDE_DIR}" \
    LDFLAGS="-L/opt/amazon/efa/lib -L/opt/amazon/efa/lib64 -L${CU13_ROOT}/lib -L${CU13_ROOT}/lib/stubs -L${NCCL_LIB_DIR_OFI}" \
    ./configure \
      --prefix=/opt/aws-ofi-nccl \
      --with-libfabric=/opt/amazon/efa \
      --with-cuda="${CU13_ROOT}" \
      --with-nccl="${NCCL_ROOT}" \
      --enable-platform-aws \
      --disable-tests; \
    make -j"$(nproc)"; \
    make install; \ \ \
    nm -D /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -q ncclGinPlugin; \
    echo "[wave12] aws-ofi-nccl libcudart linkage:"; \
    ldd /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep libcudart || echo "[wave12] (no libcudart linkage - plugin does not call CUDA runtime directly)"; \
    if ldd /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so | grep -qE 'libcudart\.so\.12'; then \
      echo "[wave12] FATAL: aws-ofi-nccl plugin still links libcudart.so.12 - expected libcudart.so.13 or none"; \
      exit 1; \
    fi; \
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
    # Wave 12: route DeepEP's build through the pip-installed cu13 nvcc so
    # torch.cpp_extension's cuda-major check passes. torch.version.cuda ==
    # '13.0' for torch==2.11.0+cu130; the wheel-provided nvcc is cu13.2.78
    # (13.0.88 has a ptxas bug that breaks sm_90a compilation). The major
    # still matches so torch.cpp_extension's check passes.
    # CUDA_HOME is re-pointed to nvidia/cu13/ (the wheel's unified
    # `bin`/`lib`/`include` root) so the torch.cpp_extension probe finds a
    # cu13 nvcc instead of /usr/local/cuda/bin/nvcc (cu12.9).
    source /etc/wave12-cuda13.env; \
    test -n "${CU13_ROOT}"; \
    echo "[wave12] deepep build using CU13_ROOT=${CU13_ROOT}"; \
    export CUDA_HOME="${CU13_ROOT}"; \
    export PATH="${CU13_ROOT}/bin:${PATH}"; \
    export LD_LIBRARY_PATH="${CU13_ROOT}/lib:${LD_LIBRARY_PATH:-}"; \
    export CPATH="${CU13_ROOT}/include:${CPATH:-}"; \
    # Wave 12: nvcc (13.2.78) and cudart (13.2.75) both at 13.2, matching
    # cccl (13.2.75). cccl's cuda_toolkit.h compatibility check passes
    # because nvcc.minor == cudart.minor == 2. torch 2.11.0+cu130 was
    # compiled against cudart 13.0.96 but cudart ABI is stable across
    # 13.0<->13.2 so the 13.2 runtime is a drop-in replacement. No
    # NVCC_PREPEND_FLAGS override needed.
    # DeepEP V2 setup.py adds -Wl,-rpath for nccl but not a matching -L, so the
    # linker cannot resolve `-l:libnccl.so`. Inject the path via LIBRARY_PATH.
    # Also include the cu12.9 base image stubs dir to resolve `-lcuda`
    # (libcuda.so stub) which the cu13 wheel does not bundle.
    NCCL_LIB_DIR="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    test -n "${NCCL_LIB_DIR}"; \
    export LIBRARY_PATH="${NCCL_LIB_DIR}:${CU13_ROOT}/lib:/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}"; \
    which nvcc && nvcc --version | tail -1; \
    python3 -c "import torch; print('[wave12] torch.version.cuda=', torch.version.cuda, 'torch.__version__=', torch.__version__)"; \
    pip3 install --no-build-isolation --break-system-packages -e .; \
    python3 -c "import deep_ep; print('DeepEP V2 OK:', deep_ep.ElasticBuffer)"; \
    echo "[wave12] DeepEP _C.so libcudart linkage:"; \
    DEEPEP_CSO="$(find /opt/DeepEP -name '_C*.so' 2>/dev/null | head -1)"; \
    echo "[wave12] DEEPEP_CSO=${DEEPEP_CSO}"; \
    if [ -n "${DEEPEP_CSO}" ]; then \
      ldd "${DEEPEP_CSO}" | grep -E 'libcudart\.so' || echo "[wave12] (deepep _C.so does not link libcudart directly)"; \
      if ldd "${DEEPEP_CSO}" | grep -qE 'libcudart\.so\.12'; then \
        echo "[wave12] FATAL: deepep _C.so still links libcudart.so.12 - expected libcudart.so.13"; \
        exit 1; \
      fi; \
    fi

# -----------------------------------------------------------------------------
# Wave 12: runtime LD_LIBRARY_PATH convenience for the cu13 wheel-provided
# libs. ldconfig already registers the wheel's lib dirs via
# /etc/ld.so.conf.d/aa-nvidia-cuda-runtime-cu13.conf, so dynamic resolution
# works at runtime without LD_LIBRARY_PATH. This env var is included only
# for overlays that shell out to binaries ignoring ldconfig (e.g. CI tools
# that set their own LD_LIBRARY_PATH without a base).
# -----------------------------------------------------------------------------
RUN set -eux; \
    source /etc/wave12-cuda13.env; \
    NCCL_LIB_DIR="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    NVSHMEM_LIB_DIR="$(find /usr/local/lib /usr/lib -path '*/nvidia/nvshmem/lib' -type d 2>/dev/null | head -1)"; \
    echo "NCCL_LIB_DIR=${NCCL_LIB_DIR}" >> /etc/wave12-cuda13.env; \
    echo "NVSHMEM_LIB_DIR=${NVSHMEM_LIB_DIR}" >> /etc/wave12-cuda13.env; \
    cat /etc/wave12-cuda13.env

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
      efa.installer="1.48.0" \
      wave.version="12" \
      cuda.runtime="13.0" \
      torch.version="2.11.0+cu130"
