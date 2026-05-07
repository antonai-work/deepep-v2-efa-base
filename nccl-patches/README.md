# Patched NCCL libraries for deepep-v2-efa-base

One-file NCCL library drop-in builds. Each `.so.2` is produced from upstream
`github.com/NVIDIA/nccl` at a named tag plus ONE code change, baked via
`make -j src.build NVCC_GENCODE='-gencode=arch=compute_90,code=sm_90'`.

## v0.2.5-sm90a: NCCL_TOPO_XML_MAX_NODES 256 -> 2048

### Patch

```
--- src/graph/topo.h (upstream v2.30.4-1, line 192)
-#define NCCL_TOPO_XML_MAX_NODES 256
+#define NCCL_TOPO_XML_MAX_NODES 2048
```

Captured in `0001-topo-xml-max-nodes-2048.patch`.

### Why this bump

Wave 19 / 21 / 22 on p5.48xlarge (H100 x 8, EFA x 32) all hit:

```
graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)
```

during DeepEP V2 `tests/elastic/test_ep.py` cross-node bring-up.

Attempted fixes that did NOT work:
- `NCCL_TOPO_FILE` with a pre-fused 122-node XML per rank: `ncclTopoFuseXmlRecursive` dedup requires ALL attributes to match, and per-rank autogen includes rank-specific attrs (rank, pci_busid fields), so the 8-rank fuse still appends 8 x 122 = 976 nodes.
- `FI_EFA_DEVICE_LIST` with 4 of 32 EFA devices: NCCL's `/sys/class/pci_bus/*` walk sees every NIC PCI BDF regardless of what libfabric filters.

The remaining fix is to raise the cap. 2048 covers 8 ranks x ~250 nodes with headroom.

### Reproducing the patched libnccl.so.2

On a host with CUDA 12.8+ toolkit installed (nvcc must be on PATH):

```bash
./build.sh
```

Or manually:

```bash
git clone https://github.com/NVIDIA/nccl.git /tmp/nccl-build
cd /tmp/nccl-build
git checkout v2.30.4-1
sed -i 's/#define NCCL_TOPO_XML_MAX_NODES 256$/#define NCCL_TOPO_XML_MAX_NODES 2048/' src/graph/topo.h
CUDA_HOME=/usr/local/cuda NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" make -j8 src.build
cp build/lib/libnccl.so.2 <this_dir>/libnccl.so.2
sha256sum libnccl.so.2 > libnccl.so.2.sha256
```

The build takes ~45 minutes on 8 cores (CUDA device code dominates).

### Verifying the installed copy

```bash
# In a running container based on v0.2.5-sm90a:
python3 -c "import torch.distributed as d; print('NCCL OK')"
strings /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 \
  | grep -c 'too many XML nodes'    # expect 1 (the WARN string, not the cap)
```

The GIN ABI symbols (`ncclTeamWorld`, `ncclCommQueryProperties`,
`ncclGinBarrierCreateRequirement`) are verified at image build time by
Dockerfile.v0.2.5's RUN block; a wheel missing those symbols aborts the
build.
