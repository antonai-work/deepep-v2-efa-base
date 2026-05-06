# Changelog

All notable changes to the published container image are documented
here. The source repo follows [semver](https://semver.org/) on the
`Dockerfile` + `patches/` surface; image tags are `v<semver>-sm90a`
where `sm90a` indicates the CUDA compute capability target (H100,
H200).

## v0.2.1-sm90a - 2026-05-06

**Wave 12 completes the cu13 migration that Wave 10 left partial.**

### Fixed
- `pins.env` flips `TORCH_INDEX` from `cu129` to `cu130`. Wave 10
  (v0.2.0-sm90a) shipped NCCL on cu13 while leaving torch on cu129;
  Wave 11 runtime-tested that image and confirmed the libcudart TLS
  split persists at the torch/vllm._C boundary with
  `c10::cuda::SetDevice` receiving error code `-112` at MoE dispatch.
  torch 2.11.0+cu130 is now a stable wheel and pulls `cuda-toolkit==
  13.0.2` transitively, which provides `nvidia-cuda-runtime==13.0.96`
  (libcudart.so.13) and - via the explicit `cuda-toolkit[nvcc,crt,cccl]
  ==13.0.2` install added to the Dockerfile - `nvidia-cuda-nvcc==
  13.0.88` (nvcc 13). DeepEP's `pip install -e .` now finds the cu13
  nvcc and its major-version check passes.
- NVSHMEM flipped from `nvidia-nvshmem-cu12==3.3.9` to
  `nvidia-nvshmem-cu13>=3.4.5` to match torch cu130's libcudart.so.13.
  Leaving NVSHMEM on cu12 would reintroduce the TLS split Wave 12 is
  closing.
- aws-ofi-nccl build now uses the cu13 wheel toolchain (`CU13_ROOT`)
  for `--with-cuda` + `CPPFLAGS`/`LDFLAGS`, and its build step
  hard-fails if the resulting plugin links libcudart.so.12.
- DeepEP `_C.so` build is now hardened: the Dockerfile asserts that
  the produced shared object does not link libcudart.so.12.
- `preflight.sh` check 6 (new) verifies torch.version.cuda is 13.x,
  DeepEP `_C.so` is not cu12-linked, the NCCL wheel is cu13 >=2.30.4,
  and ldconfig surfaces libcudart.so.13. This is the direct Wave 12
  invariant.

### Downstream impact
- All three consumer overlays (`vllm-deepep-v2-efa`,
  `nemo-rl-deepep-v2-efa`, and any future sibling) must bump their
  base `FROM` to `v0.2.1-sm90a`. Wave 13+ lands that bump.

### Preflight
- Expanded from 5/5 to 6/6. `bash /preflight.sh` prints
  `6/6 checks PASS` on success.

## v0.2.0-sm90a - 2026-05-06 (SUPERSEDED by v0.2.1)

Wave 10 partial cu13 migration: NCCL flipped to cu13 but torch left
on cu129. Wave 11 proved the libcudart TLS split still crashes
`c10::cuda::SetDevice` at MoE dispatch. Do not consume.

## v0.1.2-sm90a - 2026-05-06

**Wave 9 cu12 unification - supersedes v0.1.0 and v0.1.1.**

### Fixed
- NCCL pip wheel flipped from `nvidia-nccl-cu13==2.30.4` to
  `nvidia-nccl-cu12>=2.30.4`. Every other runtime component in this
  image and in child images links `libcudart.so.12` (torch cu129 is
  a CUDA 12.9 build, not cu13; DeepEP `_C.so` and vllm `_C.abi3.so`
  are all cu12). Wave 8 evidence: child containers crashed with
  `invalid device ordinal` (-48/-64/-128) at first MoE dispatch
  because `libnccl.so.2` was the only runtime component linking
  `libcudart.so.13`.
- `pins.env` now carries `NVIDIA_NCCL_PIN=nvidia-nccl-cu12>=2.30.4`.
  Both `.github/workflows/{build-and-push,test-build}.yml` and
  `ci/buildspec.yml` source `pins.env` and pass the pin via
  `--build-arg NVIDIA_NCCL_PIN=...`. The Dockerfile retains a
  matching `ARG NVIDIA_NCCL_PIN=nvidia-nccl-cu12>=2.30.4` default.
- `aws-ofi-nccl` build is unchanged: configure auto-finds the
  installed `nvidia-nccl-cu*` headers regardless of cu12/cu13.

### Downstream impact
- All three consumer overlays (`vllm-deepep-v2-efa`,
  `nemo-rl-deepep-v2-efa`, and any future sibling) must bump their
  base FROM to `v0.1.2-sm90a`. Wave 9b and 9c land that bump.

### Preflight
- Unchanged. `bash /preflight.sh` still prints `5/5 checks PASS`.

## v0.1.1-sm90a - 2026-05-06 (SUPERSEDED)

Pins.env extraction (Wave 7d-1 OKR-1). Inherited the same cu13 NCCL
poison as v0.1.0; do not consume.

## v0.1.0-sm90a - 2026-05-05 (SUPERSEDED by v0.1.2)

**First stable release.**

### Image contents
- Base OS: `nvidia/cuda:12.9.0-devel-ubuntu24.04`.
- EFA userspace: 1.48.0 via installer `--build-ngc` path. Installs
  `libfabric1-aws` + the NGC-flavored `libnccl-ofi-ngc-v3` plugin.
- NCCL: 2.30.4 via `nvidia-nccl-cu13` pip wheel; apt `libnccl2` 2.26.x
  is purged to avoid shadowing.
- aws-ofi-nccl: source build at commit
  `6e504db3403931cde43a2335adcc73fbc69cccac` (2026-04-24,
  "gin: Size active_put_signal to full sequence number space").
- GDRCopy: v2.5.1 userspace (kernel module assumed on host).
- NVSHMEM: `>= 3.3.9` via `nvidia-nvshmem-cu12` (link-time only; runtime
  uses NCCL Gin backend).
- PyTorch: cu129 wheel (cu128 fallback).
- NumPy: `< 2`.
- DeepEP V2: `c84dcac613c8df743a6487a312bbc966c745c600`, branch
  `aws-efa-auto-qp-cap-v2` of `dmvevents/DeepEP-1`. Three commits
  ahead of vanilla `deepseek-ai/DeepEP@b306af0` (the merge commit of
  PR #605), carrying PR #612 (3 commits: auto-QP cap, get_rdma_gbs
  EFA fast path, kScaleoutUpdateInterval 3 -> 16).

### Upstream status at publish time
- DeepEP V2 (PR #605): **merged** on `main` as `b306af0` on
  2026-04-29.
- DeepEP PR #612 (AWS EFA optimizations for V2): **still open** at
  publish time. This image freezes the three PR #612 commits as a
  vendor-applied patch set. Once #612 merges, the next image release
  will drop the fork pin and point `DEEPEP_FORK` at
  `deepseek-ai/DeepEP@main`.
- aws-ofi-nccl PR #1206 (runtime-tunable GIN ring): closed
  2026-04-28 as superseded by upstream `6e504db`. The image uses
  `6e504db` directly; the closed PR is mentioned only in historical
  context inside `patches/README.md`.

### Baked runtime env
```
FI_PROVIDER=efa
FI_EFA_USE_DEVICE_RDMA=1
FI_EFA_ENABLE_SHM_TRANSFER=0
FI_EFA_FORK_SAFE=1
NCCL_NET_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so
NCCL_GIN_ENABLE=1
NCCL_GIN_TYPE=2
NCCL_CUMEM_ENABLE=1
NCCL_CUMEM_HOST_ENABLE=1
NCCL_NVLS_ENABLE=0
NCCL_IGNORE_DISABLED_P2P=1
OFI_NCCL_PROTOCOL=RDMA
OFI_NCCL_GIN_MAX_REQUESTS=512
DEEP_EP_BACKEND=nccl
EP_EFA_MAX_QPS=2
EP_EFA_RDMA_GBS=25.0
```

### Preflight contract
`bash /preflight.sh` inside the image must print `5/5 checks PASS`
on the final stdout line. The five checks are documented in
`README.md`. This is enforced on every PR and every `main` push via
the `test-build.yml` GHA workflow.

### Known limitations
- No cuDNN, no FlashAttention, no transformer engine. Engines that
  need those install them in the overlay.
- Target arch is SM 9.0a only; Ampere / Lovelace kernels are not
  compiled. If a child workload needs `sm_80` or `sm_89`, rebuild the
  image with `TORCH_CUDA_ARCH_LIST` broadened and cut a new tag.
- The `libfabric-aws` debs use `dpkg --auto-deconfigure` against the
  CUDA base's `ibverbs-providers`. This is packaging-metadata-only;
  libfabric links against `libibverbs.so.1` which is ABI-stable
  across these versions. A future EFA installer release that drops
  the `>= 59` dependency will simplify this step.
