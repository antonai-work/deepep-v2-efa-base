# deepep-v2-efa-base

**DeepEP V2 + AWS EFA base container image.** Everything below `/opt/DeepEP`
in the image is DeepEP V2 (NCCL Gin backend) built against AWS EFA + the
aws-ofi-nccl GIN plugin, pinned at commits that produce a working 2-node
dispatch + combine loop on `p5.48xlarge` (H100) and `p5en.48xlarge` (H200).

**Status:** Release `v0.2.2-sm90a` published to `ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.2-sm90a` (2026-05-06, Wave 16 P5 topology XML fix). Last validated 2026-05-06.

The image is meant to be consumed by downstream inference and training
repos via a single `FROM ghcr.io/antonai-work/deepep-v2-efa-base:<tag>`
directive. Each engine integration (vLLM, SGLang, Megatron-LM, NeMo-RL,
etc.) adds its own runtime on top without re-solving the CUDA + EFA +
NCCL + aws-ofi-nccl + DeepEP build problem.

## Sibling repos (reproducibility triad)

| Repo | Purpose | Status |
|---|---|---|
| [deepep-v2-efa-base](https://github.com/antonai-work/deepep-v2-efa-base) | Base substrate (this repo) | v0.2.2-sm90a released |
| [nemo-rl-deepep-v2-efa](https://github.com/antonai-work/nemo-rl-deepep-v2-efa) | Training stack (Megatron-LM + NeMo-RL) | Dual-path build verified 2026-05-06 |
| [vllm-deepep-v2-efa](https://github.com/antonai-work/vllm-deepep-v2-efa) | Inference stack (vLLM + TRT-LLM) | Dual-path build verified 2026-05-06 |

Together, these three repos provide end-to-end DeepEP V2 MoE reproducibility on AWS EFA, from base substrate through training and inference.

## Upstream PRs

Five PRs filed 2026-04-28 through 2026-05-05, covering the full training + inference stack. All five are independent, EFA-specific, and safe on non-EFA fabrics:

| Upstream repo | PR | Status (2026-05-06) | Applies to |
|---|---|---|---|
| [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | [#612](https://github.com/deepseek-ai/DeepEP/pull/612) | OPEN, mergeable | Base substrate (all frameworks) |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | [#4632](https://github.com/NVIDIA/Megatron-LM/pull/4632) | OPEN, mergeable | Training |
| [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) | [#2410](https://github.com/NVIDIA-NeMo/RL/pull/2410) | OPEN, mergeable | Training |
| [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) | [#2411](https://github.com/NVIDIA-NeMo/RL/pull/2411) | OPEN, mergeable | Training (dep pin bump) |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | [#24443](https://github.com/sgl-project/sglang/pull/24443) | OPEN, mergeable | Inference |

Plus: [vllm-project/vllm#41183](https://github.com/vllm-project/vllm/pull/41183) augmented with EFA traffic evidence via comment (OPEN, actively reviewed).

DeepEP PR #612 is consumed by this repo as `patches/0001-0003`. The framework-specific PRs are consumed by the sibling repos.

## What's inside

| Layer | Version | Source |
|---|---|---|
| Base OS | `ubuntu 24.04 noble` | `nvidia/cuda:12.9.0-devel-ubuntu24.04` |
| CUDA | `13.0.x` runtime (cu13 unified post-Wave 13) | `nvidia-cuda-runtime-cu13` + `nvidia-cuda-*-cu13` wheels |
| EFA userspace | `1.48.0` | `efa-installer.amazonaws.com/aws-efa-installer-1.48.0.tar.gz`, `--build-ngc` path |
| libfabric | `libfabric1-aws` (bundled with EFA 1.48.0) | EFA tarball |
| NCCL | `>= 2.30.4` | pip `nvidia-nccl-cu13>=2.30.4` (Wave 13: cu13-unified to match torch cu130 + DeepEP `_C.so`) |
| aws-ofi-nccl | `6e504db3403931cde43a2335adcc73fbc69cccac` (2026-04-24) | `aws/aws-ofi-nccl@6e504db` |
| GDRCopy | `v2.5.1` | `NVIDIA/gdrcopy@v2.5.1` |
| NVSHMEM | `>= 3.6.0` | pip `nvidia-nvshmem-cu13>=3.6.0` (guarded post-torch to prevent downgrade to 3.4.5) |
| PyTorch | `2.11.0+cu130` | `download.pytorch.org/whl/cu130` |
| NumPy | `< 2` | pip |
| DeepEP V2 | `146cc356aa00c39ac1590c05775e05b0f031e70c` | `dmvevents/DeepEP-1@aws-efa-auto-qp-cap-v2` |

## Consuming the image

```dockerfile
FROM ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.2-sm90a

# Your engine install goes here.
RUN pip install --no-cache-dir --break-system-packages vllm==<pin>
```

Runtime env already baked into the image so child images do not need to
re-export it:

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

See [`docs/USAGE.md`](docs/USAGE.md) for multi-node K8s and `torchrun`
examples.

## Available tags

| Tag | Meaning |
|---|---|
| `v0.2.2-sm90a` | Current stable release (cu13-unified + P5 topology XML baked-in, Wave 16). Targeted SM arch: `9.0a` (H100 + H200). Fixes NCCL "too many XML nodes (max 256)" on HyperPod P5.48xlarge. |
| `v0.2.1-sm90a` | Superseded (cu13-unified, Wave 13 — missing P5 topology XML, caused NCCL 256-node overflow on first 2-node cross-node test on HyperPod). ECR digest: `sha256:5f6d45e42657c3ee3f20db9ca0f01f21c14c96c7538b598787c4b5bb9be5e974`. GHCR digest: `sha256:783fab846df7416d6ba91f0015ab51622edde5c17025b8808e2e3e6a8561953b`. |
| `v0.2.0-sm90a` | Superseded (partial cu13: NCCL flipped but torch still cu129; caused `c10::cuda::SetDevice(-64)` at MoE dispatch — see Wave 8 retrospective). |
| `v0.1.2-sm90a` | Superseded (cu12 + bfded348 DeepEP baseline; NCCL pinned to 2.30.4). |
| `v0.1.0-sm90a` | First public release (cu12-based). Deprecated. |
| `v<X.Y.Z>-sm90a` | Every subsequent release bumps SemVer; `sm90a` is kept as the primary arch suffix. |
| `sha-<short>` | Exact git SHA of the `Dockerfile` + patches that produced the image, for audit / bisect. |

All tags under `ghcr.io/antonai-work/deepep-v2-efa-base`. There is
intentionally **no `latest` tag** - pin to `v0.2.2-sm90a` (or newer) so
downstream images rebuild deterministically.

## Why this repo exists

DeepEP V2 landed on `main` upstream on 2026-04-29 (PR #605). AWS EFA
compatibility for V2 requires three small changes that are still open
as PR #612 on `deepseek-ai/DeepEP`, plus an upstream aws-ofi-nccl fix
that landed after the plugin's last tag. Every downstream engine has to
pin the same combination or fail with `CUDA_ERROR_LAUNCH_FAILED` on the
first dispatch.

Rather than duplicate that solved-problem layer across every engine
integration, this image freezes it once, publishes it to GHCR, and
lets every child repo depend on a single tag.

## What's in `patches/`

Three standalone `git format-patch` files extracted from DeepEP PR #612.
See [`patches/README.md`](patches/README.md) for the full summary and
how to apply them on vanilla upstream.

The `Dockerfile` does not apply them directly - it clones a pre-patched
fork branch at a pinned SHA, for the reasons explained in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Validation

Cross-framework evidence (2-node EFA traffic, NCCL init markers, DeepEP dispatch+combine latencies, loss curves) is documented across the three repos:

| Document | Location | Coverage |
|---|---|---|
| VALIDATION-EVIDENCE.md | Base + both sibling repos | Per-framework E2E proofs (24-token chat completion, 3-step training loss, etc.) |
| EFA-TRAFFIC-EVIDENCE.md | Base + both sibling repos | Hardware-counter proof of MoE traffic over EFA (not NVLink) |
| DEEPEP-BENCHMARKS.md | Base + both sibling repos | Microbenchmark guide (D+C latency, low-latency kernel, output interpretation) |

Base substrate D+C latency: ~740 us p50 on 2-node p5en.48xlarge (H200), ~930 us on p5.48xlarge (H100).

## Continuous Integration

This repository provides two independent build paths:

### GitHub Actions → GHCR (default)
- Publishes to `ghcr.io/antonai-work/deepep-v2-efa-base`
- Public, free, zero configuration
- Triggered automatically on tag push
- Workflow: `.github/workflows/build-and-push.yml`

### AWS CodeBuild → ECR (opt-in)
- Publishes to your private ECR repository
- For AWS-native deployments requiring private caching
- Manual setup required (IAM role, ECR repo, CodeBuild project)
- Setup guide: [`ci/CODEBUILD-SETUP.md`](ci/CODEBUILD-SETUP.md)

Use GitHub Actions unless you need private ECR hosting.

## Preflight

`preflight.sh` ships inside the image at `/preflight.sh`. The GitHub
Actions `test-build.yml` workflow runs it on every PR. Run it locally:

```
docker run --rm ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.1-sm90a \
    bash /preflight.sh
# expected final line: "6/6 checks PASS" (Wave 13 adds cu13 runtime unified check)
```

The five checks:

1. `deep_ep.ElasticBuffer` importable and has the expected class path.
2. `deep_ep.buffers.legacy.Buffer` importable (V1-compat shim retained).
3. `ldconfig -p` sees `libnccl-net-ofi.so`.
4. `/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so` physically exists.
5. DeepEP PR #612 patches landed in the cloned tree
   (`num_allocated_qps` clamp + `EFA fast path` marker).

## Build modes

Two build modes available (GitHub Actions default, CodeBuild opt-in). Both produce identical output and pass the same 5-check preflight gate. See sibling repos for detailed fast-vs-vanilla comparison.

## License

Apache 2.0. See `LICENSE`. DeepEP and aws-ofi-nccl sources carry their
own upstream licenses, unmodified; this repository contributes only the
build glue (`Dockerfile`, `preflight.sh`, CI, patch extracts).

## Repository layout

```
.
|-- Dockerfile                     # The base image recipe (~240 lines)
|-- preflight.sh                   # 5-check validation harness
|-- LICENSE                        # Apache 2.0
|-- .github/workflows/
|   |-- build-and-push.yml         # Tag push -> build + push to GHCR
|   `-- test-build.yml             # PR / main push -> build-only + preflight
|-- ci/
|   |-- buildspec.yml              # AWS CodeBuild specification
|   |-- CODEBUILD-SETUP.md         # CodeBuild setup instructions
|   `-- README.md                  # GHCR vs ECR comparison
|-- patches/
|   |-- 0001-*.patch               # auto-QP cap at 2 on EFA
|   |-- 0002-*.patch               # get_rdma_gbs EFA fast path
|   |-- 0003-*.patch               # kScaleoutUpdateInterval 3 -> 16
|   `-- README.md
`-- docs/
    |-- ARCHITECTURE.md            # What's in the image + why
    |-- USAGE.md                   # FROM examples + runtime env + K8s
    |-- VALIDATION-EVIDENCE.md     # Cross-framework E2E evidence
    |-- EFA-TRAFFIC-EVIDENCE.md    # Hardware-counter EFA proofs
    |-- DEEPEP-BENCHMARKS.md       # Microbenchmark guide
    `-- CHANGELOG.md               # Release notes
```
