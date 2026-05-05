# Changelog

All notable changes to the published container image are documented
here. The source repo follows [semver](https://semver.org/) on the
`Dockerfile` + `patches/` surface; image tags are `v<semver>-sm90a`
where `sm90a` indicates the CUDA compute capability target (H100,
H200).

## v0.1.0-sm90a - 2026-05-05

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
