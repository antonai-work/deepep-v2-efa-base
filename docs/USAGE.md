# Usage

## Pulling the image

```
docker pull ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.0-sm90a
```

GHCR packages are public. No authentication needed for `docker pull`.

## Smoke test the image (single node, no EFA required)

```
docker run --rm ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.0-sm90a \
    bash /preflight.sh
# expected final line: "5/5 checks PASS"
```

```
docker run --rm ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.0-sm90a \
    python3 -c "import deep_ep; print(deep_ep.ElasticBuffer)"
# expected: <class 'deep_ep.buffers.elastic.ElasticBuffer'>
```

## Building a child image

Minimal child Dockerfile that pip-installs vLLM on top:

```dockerfile
FROM ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.0-sm90a

ARG VLLM_VERSION=0.19.1
RUN pip install --no-cache-dir --break-system-packages \
      "vllm==${VLLM_VERSION}"

# (Optional) re-run the base preflight at child-build time to catch
# accidentally-shadowed NCCL or aws-ofi-nccl.
RUN bash /preflight.sh
```

Build:

```
docker build -t vllm-deepep-v2:dev .
```

Runtime env inherited from the base image is sufficient for DeepEP to
pick the NCCL Gin backend and AWS EFA. You typically do not need to
re-export any of the `FI_*`, `NCCL_*`, or `EP_EFA_*` variables.

## Running on a Kubernetes pod with EFA

Minimum requirements on the node:
- Instance type with EFA: `p5.48xlarge` (H100) or `p5en.48xlarge`
  (H200).
- NVIDIA driver on the host (bring-your-own, base image assumes
  driver is injected).
- EFA device plugin + NVIDIA device plugin installed on the cluster.
- Security group allows self-referencing EFA egress (required for
  SRD loopback on AWS EFA).

Pod spec fragment (distilled):

```yaml
spec:
  containers:
  - name: deepep
    image: ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.0-sm90a
    resources:
      limits:
        nvidia.com/gpu: 8
        vpc.amazonaws.com/efa: 32   # 32 EFA devices on p5/p5en.48xlarge
    securityContext:
      capabilities:
        add: ["IPC_LOCK"]
    volumeMounts:
    - name: shm
      mountPath: /dev/shm
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 32Gi
```

## Running a 2-node DeepEP bench

Once two pods of this image are Running and can see each other on the
EFA network:

```bash
# Pod 0 (rank 0):
MASTER_ADDR=<pod0-dns>  \
MASTER_PORT=29500       \
RANK=0                  \
WORLD_SIZE=16           \
NPROC_PER_NODE=8        \
torchrun \
    --nnodes=2 --node-rank=0 \
    --nproc-per-node=8 \
    --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
    /opt/DeepEP/tests/elastic/test_ep.py

# Pod 1 (rank 1):
MASTER_ADDR=<pod0-dns>  \
MASTER_PORT=29500       \
RANK=1                  \
WORLD_SIZE=16           \
NPROC_PER_NODE=8        \
torchrun \
    --nnodes=2 --node-rank=1 \
    --nproc-per-node=8 \
    --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
    /opt/DeepEP/tests/elastic/test_ep.py
```

On `p5en.48xlarge` H200, expected D+C p50 is roughly 740 us cached,
3-4 ms first iteration (for the DeepEP reference config).

## Verifying traffic actually went over EFA (not NVLink)

Capture EFA hardware counters before and after:

```bash
before_tx=$(cat /sys/class/infiniband/*/ports/1/hw_counters/tx_bytes \
            | awk '{s+=$1} END{print s}')
# ... run your workload ...
after_tx=$(cat /sys/class/infiniband/*/ports/1/hw_counters/tx_bytes \
            | awk '{s+=$1} END{print s}')
delta=$((after_tx - before_tx))
echo "EFA TX delta: $(( delta / 1024 / 1024 )) MB"
```

On the DeepEP reference config (128 tokens, 256 experts, topk=8, 2
nodes) expect `>= 1 GB` TX per node. Zero delta means NCCL silently
fell back to NVLink (same-host) or TCP (cross-host); investigate
`NCCL_DEBUG=INFO` output for the plugin-load banner and EFA device
enumeration.

## Downgrading / upgrading tag

To move from `v0.1.0-sm90a` to a later release, edit your child
Dockerfile's `FROM` line and rebuild. There is no `:latest` on
purpose - every consumer must name an exact release so builds
reproduce.
