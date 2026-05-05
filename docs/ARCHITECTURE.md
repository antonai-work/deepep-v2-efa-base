# Architecture

## Layer diagram

```
+---------------------------------------------------------------+
|  Child image: engine-deepep-v2 (vLLM, SGLang, Megatron, ...)  |
|  FROM ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.0-sm90a    |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
|  deepep-v2-efa-base  (this repo)                              |
|                                                               |
|  /opt/DeepEP/                        [DeepEP V2 installed -e] |
|  /opt/aws-ofi-nccl/                  [NCCL GIN plugin, 2.30.4 |
|                                       headers, --with-nccl]   |
|  /opt/amazon/ofi-nccl/               [NGC plugin, NCCL-agno.] |
|  /opt/amazon/efa/                    [libfabric + libefa]     |
|  /usr/.../nvidia/nccl/               [NCCL 2.30.4 pip wheel]  |
|  /usr/.../nvidia/nvshmem/            [NVSHMEM host wheel]     |
|  /usr/local/cuda/                    [CUDA 12.9 dev]          |
|  PyTorch cu129 wheel (cu128 fallback)                         |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
|  nvidia/cuda:12.9.0-devel-ubuntu24.04                         |
+---------------------------------------------------------------+
```

## Why each pin is the pin

### DeepEP source selection

Two equally-valid paths exist for building a DeepEP V2 tree with the
three PR #612 changes:

**Path A (chosen here): clone the pre-patched branch.**
We clone `dmvevents/DeepEP-1@aws-efa-auto-qp-cap-v2` at the exact SHA
`c84dcac613c8df743a6487a312bbc966c745c600` and verify `git rev-parse
HEAD` equals the pin. This branch is 3 commits ahead of upstream
`deepseek-ai/DeepEP@b306af0` (the merge commit of PR #605) and carries
exactly the three patches listed under `patches/` - byte-for-byte
identical, generated with `git format-patch` against this tree.

**Path B (documented, not chosen): apply patches/*.patch on vanilla.**
Clone `deepseek-ai/DeepEP@b306af0`, `git am patches/0001-*.patch
patches/0002-*.patch patches/0003-*.patch`, then `pip install -e .`.

We chose Path A for two reasons:

1. **Tested tree identity.** The reference run that validated the stack
   (740 us D+C p50, 2-node p5en.48xlarge H200) used exactly this branch
   tip. Cloning the same SHA means the image builds the same tree that
   produced the validation number. Path B would reconstruct a
   byte-identical tree in normal operation, but exposes an additional
   failure mode (patch context drift if upstream `main` moves) that we
   do not want on the critical-path build.
2. **Build-time simplicity.** `git clone --branch ... && git checkout
   <sha>` is a single verifiable step. `git am` with three patches is
   three steps, each of which can fail on a future upstream rebase in a
   way that would silently produce a different tree.

The `patches/` directory is retained as the human-reviewable diff over
vanilla upstream. If you want to build from vanilla
`deepseek-ai/DeepEP`, follow `patches/README.md`.

When PR #612 merges upstream, this image will flip to
`deepseek-ai/DeepEP@main` at the merge SHA, drop `patches/0001..0003`,
and the `Dockerfile` will clone the vanilla repo directly.

### aws-ofi-nccl pin `6e504db`

Earlier versions of aws-ofi-nccl's GIN plugin size the per-peer request
ring at 128 entries. DeepEP V2's hybrid-mode auto-QP formula
`num_sms * 16 + 1` overruns this ring the moment `num_allocated_qps`
exceeds ~5 QPs. The failure mode is a hard assert ("Next sequence
number is in use") surfaced to CUDA as `CUDA_ERROR_LAUNCH_FAILED` on
the first dispatch, far from the actual cause.

Two independent fixes existed:
- Runtime-tunable ring size via `OFI_NCCL_GIN_MAX_REQUESTS` (our own
  out-of-tree patch, PR #1206, closed 2026-04-28 as superseded).
- Upstream `6e504db` (2026-04-24): size the `active_put_signal` bitset
  to the full sequence-number space. This is collision-free by
  construction and needs no knob.

We pin `6e504db` directly as the canonical fix. The knob is still set
in the image (`OFI_NCCL_GIN_MAX_REQUESTS=512`) as defence in depth for
any overlay that rebuilds the plugin at an older SHA.

### NCCL 2.30.4 via pip, not apt

The `nvidia/cuda:12.9.0-devel-ubuntu24.04` base image ships
`libnccl2 2.26.5-1+cuda12.9` at `/lib/x86_64-linux-gnu/libnccl.so.2`.
DeepEP V2's NCCL Gin backend requires `>= 2.30.4` for the API it binds
against. If the apt NCCL is left in place it shadows the pip wheel via
the default ld.so search order, so we explicitly `apt-get purge` the
apt NCCL, `rm -f` the lingering `.so` symlinks, and install
`nvidia-nccl-cu13>=2.30.4` via pip with an `ldconfig` entry at
`/etc/ld.so.conf.d/aa-nvidia-nccl.conf` that ensures the pip wheel's
NCCL wins the dynamic linker race.

We use the `cu13` wheel (not `cu12`) because that is the one that ships
2.30.4+. CUDA 12.9 runtime is ABI-compatible.

### EFA installer `1.48.0` via `--build-ngc`

EFA installer 1.48+ ships a `--build-ngc` code path specifically for
NVIDIA DL-base containers:

- Skips kernel module installation (the kernel module lives on the
  host).
- Installs `libfabric1-aws` + `libfabric-aws-dev` + `libfabric-aws-bin`
  into `/opt/amazon/efa/`.
- Installs the NGC flavor of the aws-ofi-nccl plugin
  (`libnccl-ofi-ngc-v3`) built with `--disable-nccl-net-library`, i.e.
  it is NCCL-version-agnostic. This is the plugin we point
  `NCCL_NET_PLUGIN` at, not the one we rebuild from source.

We do NOT install apt's `rdma-core`, `libibverbs-dev`, or
`ibverbs-providers`. Noble ships v50 of those, which would block the
v61 rdma-core bundled inside the EFA 1.48.0 tarball - and
`libfabric1-aws` hard-depends on `ibverbs-providers >= 59`. The
Dockerfile runs `dpkg -i --auto-deconfigure` on the tarball's v61 debs
first, which upgrades the preinstalled v50 `ibverbs-providers`, then
`libfabric1-aws` slots in cleanly.

### Why we build aws-ofi-nccl from source even though NGC ships one

We keep both:

- **NGC plugin** (`/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so`):
  installer-provided, NCCL-version-agnostic, pointed at by
  `NCCL_NET_PLUGIN`. This is what actually runs.
- **Source build** (`/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so`):
  built with `--with-nccl=<pip-wheel-root>` so its GIN plugin symbols
  match the exact NCCL 2.30.4 API we linked DeepEP against. Some
  overlays (e.g. ones that rebuild DeepEP with different flags) prefer
  to re-point `NCCL_NET_PLUGIN` at this path. Retained as a matched
  fallback.

## Size and target arch

- Target SM arch: `9.0a` (H100 and H200). The Dockerfile exports
  `TORCH_CUDA_ARCH_LIST=9.0` before `pip install -e .` on DeepEP so
  only SM 9.0 kernels are compiled. This keeps the image smaller and
  avoids build failures on kernels that require features not present
  on earlier architectures.
- Compressed image size target: `< 20 GB`. Achieved on the reference
  build by:
  - `rm -rf /var/build-scratch/*` at the end of every layer's source
    clone (the build-scratch prefix is what the `Dockerfile` uses; the
    top-level scratch directory never escapes a single `RUN` layer)
  - `rm -f /opt/amazon/efa/lib/libfabric.a` (static lib, unused at
    runtime, saves ~50 MB)
  - Using CUDA `devel` (not `cudnn-devel`) - we do not need cuDNN

## Not included (on purpose)

- **No engine.** No vLLM, SGLang, Megatron-LM, NeMo-RL, TRT-LLM, or
  LLM-D. Child images add exactly one engine on top and keep this base
  engine-agnostic.
- **No model weights.** Mount via FSx for Lustre, S3, or bake into the
  child image.
- **No kubectl / AWS CLI.** These belong in the engine overlay or the
  K8s-side entrypoint, not in a library base.
- **No cuDNN.** If an engine needs it, install in the overlay. Saves
  ~1 GB compressed.
