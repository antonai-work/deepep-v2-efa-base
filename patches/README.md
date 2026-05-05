# Patches

Three standalone `.patch` files extracted from DeepEP upstream PR #612
("AWS EFA optimizations for V2"). Each patch is a standard
`git format-patch` output and applies with `git am <file.patch>` on a
fresh checkout of `deepseek-ai/DeepEP` at `main@b306af0` (the merge
commit of PR #605 `epv2-release`).

## Upstream PR

| Patch files | Upstream repo | PR |
|---|---|---|
| `0001`, `0002`, `0003` | [`deepseek-ai/DeepEP`](https://github.com/deepseek-ai/DeepEP) | [#612 "AWS EFA optimizations for V2"](https://github.com/deepseek-ai/DeepEP/pull/612) |

PR #612 remained open as of the `v0.1.0-sm90a` tag (see
`docs/CHANGELOG.md` for upstream status at image publish time).
Once #612 merges, the `Dockerfile` pin can be moved to vanilla
`deepseek-ai/DeepEP@main` and these patches dropped.

## Applying patches on vanilla upstream

```
git clone https://github.com/deepseek-ai/DeepEP
cd DeepEP
git checkout b306af0
git am /path/to/patches/0001-aws-efa-cap-auto-QP-at-2-on-EFA-to-avoid-128-slot-GI.patch \
       /path/to/patches/0002-aws-efa-add-EFA-fast-path-in-get_rdma_gbs-to-fix-SM-.patch \
       /path/to/patches/0003-aws-efa-raise-dispatch-kScaleoutUpdateInterval-from-.patch
pip install -e .
```

The `Dockerfile` at the repo root currently takes a shortcut and clones a
pre-patched fork branch rather than applying these files individually
(see `docs/ARCHITECTURE.md` "DeepEP source selection" for the rationale).
The patches in this directory are the audited, canonical representation of
the delta versus vanilla upstream and are kept for review / alternative
build paths.

## Patch summaries

### 0001 - auto-QP cap at 2 on EFA

DeepEP V2's hybrid-mode auto-QP formula `num_sms * 16 + 1` exceeds the
aws-ofi-nccl GIN plugin's 128-slot per-peer request ring, triggering a
hard assert surfaced to CUDA as `CUDA_ERROR_LAUNCH_FAILED`. This patch
detects AWS EFA via `FI_PROVIDER=efa` or `/sys/class/infiniband/rdmap*s0`
and caps auto-sized `num_allocated_qps` and the theoretical QP
recommendation to `EP_EFA_MAX_QPS` (default 2, validated on
p5en.48xlarge H200). Non-EFA behavior unchanged.

Also adds a sysfs fallback to `get_rdma_gbs()` so `/sys/class/infiniband/
*/ports/1/rate` is used when `ibstat` cannot enumerate EFA rdmap HCAs.

### 0002 - EFA fast path in `get_rdma_gbs`

Adds an EFA-specific branch to `get_rdma_gbs()` that returns 25 GB/s
(overridable via `EP_EFA_RDMA_GBS`) when `/sys/class/infiniband/rdmap*s0`
is present. Without this, sysfs reports the 16-rail x 200 Gb/s = 400 GB/s
line rate, which makes `ElasticBuffer.get_theoretical_num_sms()` pick
`num_sms >= 64` on H200 and spends more time on inter-SM notify/forward
synchronization than on the cross-node RDMA itself (24 SMs observed 2x
slower than 4 SMs in the reference QP sweep).

### 0003 - raise dispatch `kScaleoutUpdateInterval` from 3 to 16

On 2-node H200 EFA with the aws-ofi-nccl GIN proxy + libfabric SRD
backend, each cross-node atomic is far more expensive than an
InfiniBand atomic (CPU-relayed proxy hop). Raising the per-channel
scaleout update interval from 3 tokens to 16 cuts the atomic
frequency ~3x. Measured on a reference workload (num_tokens=128, 8
ranks/node, ui=16):

- dispatch kernel mean: 463us -> 279us (-40%)
- D+C kernel mean:      785us -> 615us (-22%)
- EFA TX per run:       6.13 GB (unchanged)

`kNumSlotsPerForwardChunk` tracks `kScaleoutUpdateInterval` so the
forward-warp consume loop stays aligned with the release granularity.
The symmetric combine-side interval (`kNumScaleupUpdateInterval`) was
tested at 16 and showed a regression (+60us); it is kept at 3.
