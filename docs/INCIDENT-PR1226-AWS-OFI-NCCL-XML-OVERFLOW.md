# aws-ofi-nccl: incident report — custom p5.48xlarge topology XML triggers NCCL_TOPO_XML_MAX_NODES=256 overflow on cross-node init (three-way controlled experiment isolates patch; mechanism open)

## Summary

**Three-way controlled experiment on identical HyperPod p5.48xlarge hardware isolates the regression to our topology patch in this test scenario. In the specific scenario we tested — 2-pod NCCL 2.30.4 `init_process_group` + `all_reduce` over EFA on HyperPod p5.48xlarge — bumping aws-ofi-nccl from commit `6e504db` to v1.19.1 tag `206c02c` alone (v0.2.4) passes cleanly. We make no broader claim about v1.19.1's behavior in other hardware, NCCL versions, workloads, or scales. The mechanism by which our patch breaks cross-node init is not yet understood.**

We filed PR #1226 against aws-ofi-nccl to ship a static `p5.48xl-topo.xml` (122 nodes) + a `platform_data` entry in `src/platform-aws.cpp` that registers the XML for the p5.48xlarge regex. The patched image produced `graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)` on all 16 ranks of a 2-pod `torch.distributed.init_process_group(backend="nccl")` run and raised `ncclInternalError`. To isolate cause from confounds we ran a three-way test on the same hardware: the pre-patch baseline (v0.2.1, older plugin commit) passes; the post-patch image (v0.2.2, new plugin + our patch) fails; a deliberate third image (v0.2.4, new plugin commit, NO patch) passes. In the specific 2-pod NCCL 2.30.4 `init_process_group` + `all_reduce` scenario we tested on HyperPod p5.48xlarge, the version bump alone did not cause a regression. The naive arithmetic we originally proposed ("2 x 122 = 244 > 256 overflows merge") is falsified by the autogen case on both passing images: stock autogen produces 126 nodes per host (4 more than our static XML) and merges successfully. The real mechanism is unknown; we have three candidate hypotheses (double-path install, concurrent autogen + static load, non-MNNVL dedup failure). We closed PR #1226 and the image is rebuilt with stock autogen. This report asks AWS for mechanism confirmation so other customers attempting a similar p5.48xlarge static XML approach — a patch like ours — don't repeat the failure. This is an incident report about our own patch, not a bug report against upstream — we observed no regression in the v1.19.1 plugin within the tested scenario.

## Environment

- **Instance type**: p5.48xlarge (H100 x 8, 32 EFA NICs, 8 Nitro cards x 2 ports each)
- **Cluster**: SageMaker HyperPod EKS cluster in us-east-2 (cluster ID redacted for public publication; available privately on request)
- **Instance IDs where reproduced**: node-A and node-B, two distinct p5.48xlarge HyperPod nodes (specific HyperPod instance IDs redacted for public publication; available privately on request)
- **libfabric**: 2.3.1amzn2.0
- **NCCL**: 2.30.4 (pip wheel `nvidia-nccl-cu13`, commit `1933fdd6360a8bfccaa0166bd71bce363d32e5b6`)
- **aws-ofi-nccl**: two commits tested. v0.2.1 uses commit `6e504db3403931cde43a2335adcc73fbc69cccac` (2026-04-24, pre-v1.19.1 master). v0.2.2 and v0.2.4 use commit `206c02c478c6d724af09c3cbca59c06863a0b9c0` (v1.19.1 tag, 2026-05-05). Only v0.2.2 additionally applies our local patch (see section 5). v0.2.1, v0.2.3, and v0.2.4 are unpatched.
- **CUDA driver**: 580.126.09
- **EFA kernel modules**: efa, efa_nv_peermem, gdrdrv (all loaded)
- **Base images under test**:
  - v0.2.1-sm90a (old plugin, no patch, PASSES): `<redacted-ecr>/deepep-v2-efa-base:v0.2.1-sm90a-amd64@sha256:5f6d45e42657c3ee3f20db9ca0f01f21c14c96c7538b598787c4b5bb9be5e974`
  - v0.2.2-sm90a (new plugin + our patch, FAILS): `<redacted-ecr>/deepep-v2-efa-base:v0.2.2-sm90a-amd64@sha256:9b920b504116edfd96d04c85668a562aeb461e5a09441bf03fdbf9d04572dfbf`
  - v0.2.3-sm90a (revert of v0.2.2, build 2026-05-06T20:23Z): `ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.3-sm90a`
  - v0.2.4-sm90a (new plugin, NO patch, PASSES — confound-isolation image): `<redacted-ecr>/deepep-v2-efa-base:v0.2.4-sm90a-amd64@sha256:615c52eac8f054fcdcd526f502cbadcf8066a7b29982f09c7de7df4ca7953e5a`

## Empirical three-way controlled experiment

This is the ground truth. Everything below (NCCL source analysis, hypotheses) is subordinate to these measurements.

### Why three images instead of two

An initial A/B (v0.2.1 PASS vs v0.2.2 FAIL) was not single-variable: v0.2.2 bumped the aws-ofi-nccl commit from `6e504db` (2026-04-24) to `206c02c` (v1.19.1 tag, 2026-05-05) AND added our topology patch. That's two changes, so the failure could have come from either. We built v0.2.4 as the isolation control: same v1.19.1 commit as v0.2.2, same Dockerfile, but with the topology patch removed. v0.2.4 PASSES. This pins the cause on the patch and rules out the version bump in the tested scenario.

### Test matrix

| Image | aws-ofi-nccl commit | Our topology patch | Autogen node count | 2-node NCCL init |
|---|---|---|---|---|
| v0.2.1-sm90a | `6e504db` (2026-04-24, pre-v1.19.1) | NO | 126 per host (dumped) | **PASS** (Wave 18, 2026-05-06T20:35 UTC) |
| v0.2.4-sm90a | `206c02c` (v1.19.1 tag, 2026-05-05) | NO | 126 per host (dumped) | **PASS** (Wave 18f, 2026-05-06T21:00 UTC) |
| v0.2.2-sm90a | `206c02c` (v1.19.1 tag, 2026-05-05) | YES (122-node static XML + platform_data entry) | n/a (NCCL_TOPO_FILE overrides autogen) | **FAIL** all 16 ranks, `graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)`, `ncclInternalError` |

Test workload for all three is identical: `torch.distributed.init_process_group(backend="nccl")` + one `all_reduce` across 16 ranks (8 per pod) on 2 pods with `podAntiAffinity` pinning to distinct p5.48xlarge HyperPod nodes.

### Causal reading

- v0.2.1 to v0.2.4: plugin commit changes from `6e504db` to `206c02c`, no patch in either. Both PASS. Plugin bump did not regress in the tested scenario (2-pod HyperPod p5.48xlarge, NCCL 2.30.4, init + all_reduce).
- v0.2.4 to v0.2.2: same plugin commit `206c02c`, patch absent vs present. v0.2.4 PASSES, v0.2.2 FAILS. Single-variable difference is our topology patch.
- Conclusion: within this test scenario, the regression is localized to our topology patch. We did not observe the overflow in stock aws-ofi-nccl v1.19.1 in the scenario tested.

### v0.2.1 PASS evidence (Wave 18 empirical run, 2026-05-06T20:35-20:40 UTC)

Pod 0 (ranks 0-7) tail, from `/tmp/wave18/pod0.log` (verbatim, line 39 onward):

```
w18-nccl-0:113:113 [3] NCCL INFO Channel 03/0 : 11[3] -> 3[3] [receive] via NET/Libfabric/3/GDRDMA
w18-nccl-0:113:113 [3] NCCL INFO Channel 11/0 : 11[3] -> 3[3] [receive] via NET/Libfabric/3/GDRDMA
w18-nccl-0:113:113 [3] NCCL INFO Channel 03/0 : 3[3] -> 11[3] [send] via NET/Libfabric/3/GDRDMA
w18-nccl-0:113:113 [3] NCCL INFO Channel 11/0 : 3[3] -> 11[3] [send] via NET/Libfabric/3/GDRDMA
...
w18-nccl-0:113:113 [3] NCCL INFO Connected all trees
w18-nccl-0:114:114 [4] NCCL INFO Connected all trees
w18-nccl-0:112:112 [2] NCCL INFO Connected all trees
w18-nccl-0:115:115 [5] NCCL INFO Connected all trees
w18-nccl-0:110:110 [0] NCCL INFO Connected all trees
w18-nccl-0:111:111 [1] NCCL INFO Connected all trees
w18-nccl-0:116:116 [6] NCCL INFO Connected all trees
w18-nccl-0:117:117 [7] NCCL INFO Connected all trees
[wave18] rank=0 all_reduce OK, sum[0]=16.0
[wave18] rank=1 all_reduce OK, sum[0]=16.0
[wave18] rank=2 all_reduce OK, sum[0]=16.0
[wave18] rank=3 all_reduce OK, sum[0]=16.0
[wave18] rank=4 all_reduce OK, sum[0]=16.0
[wave18] rank=5 all_reduce OK, sum[0]=16.0
[wave18] rank=6 all_reduce OK, sum[0]=16.0
[wave18] rank=7 all_reduce OK, sum[0]=16.0
[wave18] rank=0 barrier complete
[wave18] rank=1 barrier complete
...
w18-nccl-0:110:110 [0] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=0 SUCCESS - no XML overflow error
...
[wave18] NCCL topology XML node count:
126
```

Pod 1 (ranks 8-15) tail, from `/tmp/wave18/pod1.log` (verbatim):

```
[wave18] rank=8 all_reduce OK, sum[0]=16.0
[wave18] rank=9 all_reduce OK, sum[0]=16.0
[wave18] rank=10 all_reduce OK, sum[0]=16.0
[wave18] rank=11 all_reduce OK, sum[0]=16.0
[wave18] rank=12 all_reduce OK, sum[0]=16.0
[wave18] rank=13 all_reduce OK, sum[0]=16.0
[wave18] rank=14 all_reduce OK, sum[0]=16.0
[wave18] rank=15 all_reduce OK, sum[0]=16.0
w18-nccl-1:148:148 [0] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=8 SUCCESS - no XML overflow error
...
[wave18] rank=14 SUCCESS - no XML overflow error
```

Key evidence points:
- Cross-node NET/Libfabric/GDRDMA channels established between all rank pairs (rank 3 <-> rank 11, rank 2 <-> rank 10, etc.) confirming EFA RDMA plane is up
- `NCCL INFO Connected all trees` on every rank in pod 0 and pod 1
- `all_reduce` on a tensor of ones across 16 ranks produces `sum[0]=16.0` on every rank (correctness)
- `NET/OFI gin: Finalizing` proves NCCL gin plugin active (same plugin that would have loaded our custom XML if present)
- Autogen topology count: **126 XML nodes per host** (dumped via `NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml`, counted via `grep -c '<.*>'`)
- Zero `NCCL WARN` messages anywhere in either pod log

First 50 lines of the actual autogen topology (pod 0), from `/tmp/wave18/pod0.log`:

```xml
<system version="1">
  <cpu host_hash="0x23d79eeea3fd2808" numaid="0" affinity="00000000,0000ffff,ffffffff" arch="x86_64" vendor="AuthenticAMD" familyid="175" modelid="1">
    <pci busid="0000:45:00.0" class="0x060400" vendor="0x1d0f" device="0x0200" subsystem_vendor="0x1d0f" subsystem_device="0x0200" link_speed="16.0 GT/s PCIe" link_width="8">
      <pci busid="0000:52:00.0" link_speed="32.0 GT/s PCIe" link_width="16" class="0x020000" vendor="0x1d0f" device="0xefa1" subsystem_vendor="0x1d0f" subsystem_device="0xefa1">
        <nic>
          <net name="rdmap82s0" dev="0" latency="75" speed="400000" port="1" guid="0xa0103e500000000" maxconn="262144" gdr="1" net="1" gin="1"/>
        </nic>
      </pci>
      <pci busid="0000:53:00.0" class="0x030200" vendor="0x10de" device="0x2330" subsystem_vendor="0x10de" subsystem_device="0x16c1" link_speed="32.0 GT/s PCIe" link_width="16">
        <gpu dev="0" sm="90" rank="0" gdr="1">
          <nvlink target="0000:d0:00.0" count="5" tclass="0x068000"/>
          <nvlink target="0000:d2:00.0" count="4" tclass="0x068000"/>
          <nvlink target="0000:d1:00.0" count="5" tclass="0x068000"/>
          <nvlink target="0000:cf:00.0" count="4" tclass="0x068000"/>
        </gpu>
      </pci>
    </pci>
```

Observations vs our static XML:
- Autogen emits `<nvlink>` children under each `<gpu>` (4 per GPU x 8 GPUs = 32 extra nodes); our static XML omits these
- Autogen emits 1 EFA NIC per GPU subtree (8 NICs total across the 126-node output), not the 32 NICs our static file enumerates. This is consistent with autogen only listing NICs for which GPUDirect RDMA is mapped to the local GPU under the same PCIe switch
- Autogen includes `vendor="0x1d0f"`, `device="0xefa1"` on the NIC; our static XML omits vendor/device attributes
- Autogen includes `host_hash` attribute on the `<cpu>` node (literal value `0x23d79eeea3fd2808` from pod 0); our static XML omits `host_hash` entirely

This last point was initially considered as a possible mechanism, but see Appendix E for why it's a weak candidate: NCCL 2.30.4 `src/graph/topo.cc:1542-1546` rewrites the CPU's `host_hash` attribute to `getHostHash()` on every load, so the attribute is present by merge time on both paths. The `host_hash` difference alone is therefore unlikely to explain the static-vs-autogen fuse behavior; the more plausible attribute-set differences are on PCI nodes (missing `vendor`, `device`, `subsystem_*`) and GPU children (missing `<nvlink>` subtrees), both of which are discussed under Hypothesis C.

### v0.2.2 FAIL evidence (session 2026-05-06, StatefulSet applied ~18:33 UTC; container-clock logs below show ~20:00:45 after image pull + pod ready)

All 16 ranks across both pods emitted the error simultaneously. Verbatim tail:

```
[2026-05-06 20:00:45] deepep-cu13-xnode-0:257:257 [2] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)

[2026-05-06 20:00:45] deepep-cu13-xnode-0:259:259 [4] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)

[2026-05-06 20:00:45] deepep-cu13-xnode-0:261:261 [6] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)

[2026-05-06 20:00:45] deepep-cu13-xnode-0:256:256 [1] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)
[W506 20:00:45.737645814 ProcessGroupNCCL.cpp:1575] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources.

[2026-05-06 20:00:45] deepep-cu13-xnode-0:255:255 [0] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)

[2026-05-06 20:00:45] deepep-cu13-xnode-0:260:260 [5] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)

[2026-05-06 20:00:45] deepep-cu13-xnode-0:262:262 [7] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)

[2026-05-06 20:00:45] deepep-cu13-xnode-0:258:258 [3] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)
```

Followed by `torch.distributed.DistBackendError: NCCL error in: /pytorch/torch/csrc/distributed/c10d/NCCLUtils.cpp:94, internal error -- please report this issue to the NCCL developers, NCCL version 2.30.4` and `ncclInternalError: Internal check failed.` on every rank.

All ranks hit the error at line 381 (the `xmlAddTree` cap check), not line 346 (the `xmlAddNode` cap check during parser). This tells us the overflow happened during the cross-node fuse step (`xml.cc:309 ncclTopoFuseXml -> xmlAddTree`), not during initial XML parse.

### The isolation is clean

Between v0.2.4 (PASS) and v0.2.2 (FAIL) the only Dockerfile-level differences are:

1. v0.2.2 has `COPY base/deepep-base-v2/upstream-patches/0001-topology-add-p5.48xlarge-topology-XML.patch` and applies it to the aws-ofi-nccl source before build
2. v0.2.2 has `install -D -m 0644 p5.48xl-topo.xml` into the plugin share directory
3. v0.2.2's resulting `.so` contains the additional `platform_data` entry for `^p5\\.48xlarge$`

Same everything else: base OS, NCCL 2.30.4 wheel, libfabric 2.3.1, aws-ofi-nccl commit `206c02c`, cluster, kernel, driver, pod manifest, env vars, test script, test workload.

Between v0.2.1 (PASS) and v0.2.4 (PASS) the only change is the aws-ofi-nccl commit (`6e504db` to `206c02c`). Both produce a 126-node autogen topology and both complete 16-rank cross-node init cleanly. Within this tested scenario (2-pod HyperPod p5.48xlarge, NCCL 2.30.4, `init_process_group` + `all_reduce`) the version bump did not produce a regression. We make no broader claim about v1.19.1 beyond this specific test.

## Timeline (observable events)

All times UTC.

- **2026-05-06 18:25** — Applied StatefulSet with 2 pods, p5.48xlarge podAntiAffinity, 32 EFA, privileged (v0.2.2 image)
- **2026-05-06 18:30** — torchrun launched 16 ranks (8 per pod)
- **2026-05-06 18:33** — `torch.distributed.init_process_group(backend="nccl")` begins
- **2026-05-06 18:33:45** — All 16 ranks emit `graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)` simultaneously
- **2026-05-06 18:33:45** — `ncclInternalError: Internal check failed.` on every rank; process group helper raises `torch.distributed.DistBackendError`
- **2026-05-06 19:16** — Patch signed and PR #1226 filed to aws-ofi-nccl
- **2026-05-06 19:45** — Eric Raute leaves review: "I don't understand why you were hitting the 'too many XML nodes' error, we haven't seen that. But we don't want to get rid of the topology autogeneration -- that was required to properly handle the case where not all GPUs are visible inside a container."
- **2026-05-06 20:23** — v0.2.3-sm90a tag pushed to revert patch; GHA build starts
- **2026-05-06 20:35-20:40** — Wave 18 empirical test deploys v0.2.1 baseline (old plugin, no patch); PASS; 126-node autogen confirmed
- **2026-05-06 20:45** — PR #1226 closed by reporter after empirical data showed the patch was the regression
- **2026-05-06 20:55-21:01** — Wave 18f isolation test deploys v0.2.4 (same v1.19.1 plugin commit as v0.2.2, patch removed); PASS; 126-node autogen confirmed. Three-way experiment complete: cause pinned to our patch, plugin bump ruled out in the tested scenario.

## The broken patch (what we added to aws-ofi-nccl)

The patch adds three things to aws-ofi-nccl v1.19.1:

1. **New file**: `topology/p5.48xl-topo.xml` (204 lines, 122 XML nodes: 1 `<system>`, 1 `<cpu>`, 48 `<pci>`, 8 `<gpu>`, 32 `<nic>`, 32 `<net>`). Node count verified programmatically (see Appendix E).
2. **Modified**: `topology/Makefile.am` — added `p5.48xl-topo.xml` to `dist_xml_DATA` so the file ships in the plugin's share directory.
3. **Modified**: `src/platform-aws.cpp` — added a new `platform_data` entry matching `^p5\\.48xlarge$` with `.topology = "p5.48xl-topo.xml"`, placed before the existing `^p5(e?\\..*)` wildcard so exact-match resolves first. The wildcard entry has `.topology = NULL`, so stock v1.19.1 falls through to autogen.

The plugin at runtime loads the matching `platform_data` entry in `nccl_net_ofi_plugin_init` and, if `.topology` is non-NULL, calls `env_manager::getInstance().insert_envvar("NCCL_TOPO_FILE", topology_path, false)` which sets `NCCL_TOPO_FILE=/opt/aws-ofi-nccl/share/aws-ofi-nccl/xml/p5.48xl-topo.xml` in the process env before the next NCCL init.

The relevant plugin code, verbatim from `aws-ofi-nccl/src/platform-aws.cpp:660-690` (master HEAD as of 2026-05-06):

```cpp
	 * environment variable NCCL_TOPO_FILE is not set.
	 */
	if (getenv("NCCL_TOPO_FILE")) {
		NCCL_OFI_INFO(NCCL_INIT,
			      "Running on %s platform, NCCL_TOPO_FILE environment variable is already set to %s",
			      nccl_net_ofi_get_product_name(), getenv("NCCL_TOPO_FILE"));
	} else if (platform->topology) {
		// ... constructs topology_path from plugin share dir + filename ...
		env_manager::getInstance().insert_envvar("NCCL_TOPO_FILE", topology_path, false);
	}
```

Stock `platform_data_map` entries for p5 / p5e (verbatim from master):

```cpp
{
	.name = "p5/p5e",
	.regex = "^p5(e?\\..*)",
	.topology = NULL,
	.default_dup_conns = 0,
	.latency = 75.0,
	.gdr_required = true,
	...
},
```

The `.topology = NULL` on stock is why autogen runs for p5.48xlarge. Our patch broke this by registering a filename.

Full patch text is in Appendix C.

## Minimum reproducer (2-pod K8s StatefulSet)

**Reproducer scope note:** the 2-rank manifest below is a MINIMUM reproducer 
(one Python process per pod, 2 ranks total). It reproduces the XML overflow 
without requiring torchrun or an 8-GPU-per-pod setup. The v0.2.2 FAIL evidence 
in Section 5 and Appendix A.4 is from the full empirical test harness which 
uses `torchrun --nproc-per-node=8 --nnodes=2` for 16 ranks total (the 
configuration our three-way controlled experiment was actually run with). 
If the 2-rank minimum reproducer below does not trigger the overflow in your 
environment, fall back to the 16-rank torchrun configuration described in 
`bench/k8s/wave18-nccl-basic-xnode.yaml` at antonai-work/deepep-v2-integration 
(adapted from Wave 18b empirical test harness).


Deploy the StatefulSet below on any HyperPod EKS cluster with at least 2 p5.48xlarge nodes available for `ml.p5.48xlarge` nodeAffinity. Swap the image digest between v0.2.1 (PASS) and v0.2.2 (FAIL) to reproduce the A/B.

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: aws-xml-overflow-repro
---
apiVersion: v1
kind: Service
metadata:
  name: xml-repro-headless
  namespace: aws-xml-overflow-repro
spec:
  clusterIP: None
  selector:
    app: xml-repro
  ports:
  - port: 8361
    name: pytorch
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: xml-repro
  namespace: aws-xml-overflow-repro
spec:
  serviceName: xml-repro-headless
  replicas: 2
  podManagementPolicy: Parallel
  selector:
    matchLabels:
      app: xml-repro
  template:
    metadata:
      labels:
        app: xml-repro
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                app: xml-repro
            topologyKey: kubernetes.io/hostname
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: node.kubernetes.io/instance-type
                operator: In
                values: [ml.p5.48xlarge]
      containers:
      - name: repro
        # Swap digest between v0.2.1 (PASS) and v0.2.2 (FAIL):
        #   v0.2.1 PASS: sha256:5f6d45e42657c3ee3f20db9ca0f01f21c14c96c7538b598787c4b5bb9be5e974
        #   v0.2.2 FAIL: sha256:9b920b504116edfd96d04c85668a562aeb461e5a09441bf03fdbf9d04572dfbf
        image: <redacted-ecr>/deepep-v2-efa-base:v0.2.2-sm90a-amd64@sha256:9b920b504116edfd96d04c85668a562aeb461e5a09441bf03fdbf9d04572dfbf
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -euo pipefail
          ORDINAL=${HOSTNAME##*-}
          export RANK=$ORDINAL
          export WORLD_SIZE=2
          export MASTER_ADDR="xml-repro-0.xml-repro-headless.aws-xml-overflow-repro.svc.cluster.local"
          export MASTER_PORT=8361
          for i in $(seq 1 60); do
            getent hosts "$MASTER_ADDR" >/dev/null 2>&1 && break
            sleep 1
          done
          export NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml
          export NCCL_DEBUG=INFO
          python3 -c "
          import os, torch, torch.distributed as dist
          dist.init_process_group(backend='nccl')
          rank = dist.get_rank()
          t = torch.ones(1, device=f'cuda:{rank % 8}')
          dist.all_reduce(t)
          print(f'[repro] rank={rank} all_reduce OK, sum[0]={t.item()}')
          dist.barrier()
          print(f'[repro] rank={rank} SUCCESS')
          dist.destroy_process_group()
          "
          if [ -f /tmp/nccl_topo.xml ]; then
            echo "[repro] NCCL topology XML node count:"
            grep -c '<.*>' /tmp/nccl_topo.xml
          fi
          sleep infinity
        env:
        - name: LD_LIBRARY_PATH
          value: "/opt/nvidia-driver:/opt/amazon/efa/lib:/opt/amazon/aws-ofi-nccl/lib:/usr/local/cuda/lib64:/usr/local/lib"
        resources:
          limits:
            vpc.amazonaws.com/efa: 32
            cpu: "90"
            memory: "1024Gi"
          requests:
            vpc.amazonaws.com/efa: 32
            cpu: "32"
            memory: "256Gi"
        securityContext:
          privileged: true
          capabilities:
            add: [IPC_LOCK, SYS_RESOURCE, SYS_PTRACE, SYS_ADMIN]
        volumeMounts:
        - name: dev-shm
          mountPath: /dev/shm
        - name: dev-infiniband
          mountPath: /dev/infiniband
      volumes:
      - name: dev-shm
        emptyDir:
          medium: Memory
          sizeLimit: 128Gi
      - name: dev-infiniband
        hostPath:
          path: /dev/infiniband
```

To reproduce PASS: swap image digest to `sha256:5f6d45e42657c3ee3f20db9ca0f01f21c14c96c7538b598787c4b5bb9be5e974` (v0.2.1).
To reproduce FAIL: keep the v0.2.2 digest as shown above.

## NCCL 2.30.4 source analysis

This section documents what the NCCL source code says about topology cap and merge. It is descriptive, not causal — the merge math alone does not explain why 126 autogen nodes merge OK but 122 static nodes overflow.

### Cap definition

`src/graph/topo.h:192`:

```c
#define NCCL_TOPO_XML_MAX_NODES 256
```

### Error site

`src/graph/xml.h:379-398`:

```c
static ncclResult_t xmlAddTree(struct ncclXml* dst, struct ncclXmlNode* parent, struct ncclXmlNode* srcNode) {
  if (dst->maxIndex == dst->maxNodes) {
    WARN("Error : too many XML nodes (max %d)", dst->maxNodes);
    return ncclInternalError;
  }
  struct ncclXmlNode* dstNode = dst->nodes+dst->maxIndex++;
  *dstNode = *srcNode;
  dstNode->parent = parent;
  if (parent) {
    if (parent->nSubs == MAX_SUBS) {
      WARN("Error : too many XML subnodes (max %d)", MAX_SUBS);
      return ncclInternalError;
    }
    parent->subs[parent->nSubs++] = dstNode;
  }
  dstNode->nSubs = 0;
  for (int i=0; i<srcNode->nSubs; i++)
    NCCLCHECK(xmlAddTree(dst, dstNode, srcNode->subs[i]));
  return ncclSuccess;
}
```

`xmlAddTree` is the recursive copy path used during cross-node fuse. Each node adds one to `dst->maxIndex`; the check at the top fires when the destination arena is full.

### Multi-node fuse allocation

`src/graph/topo.cc:1645-1658` (the MNNVL branch):

```c
if (comm->MNNVL) {
  // Ensure that we have enough room when fusing topos from multiple nodes.
  free(xml);
  xml = NULL;
  NCCLCHECKGOTO(xmlAlloc(&xml, nLocalRanks*NCCL_TOPO_XML_MAX_NODES), ret, fail);
} else {
  // In the intra-node case there's no need to enlarge the topo xml.
  xml->maxIndex = 0;
}
for (int i = 0; i < nLocalRanks; i++) {
  struct ncclXml* peerXml = (struct ncclXml*)(mem+xmlMemSize(NCCL_TOPO_XML_MAX_NODES)*i);
  NCCLCHECKGOTO(ncclTopoConvertXml(peerXml, (uintptr_t)peerXml->nodes, 0), ret, fail);
  NCCLCHECKGOTO(ncclTopoFuseXml(xml, peerXml), ret, fail);
}
```

Note that `comm->MNNVL` gates the resize. For **non-MNNVL** multi-node communicators (i.e. the typical p5.48xlarge EFA case where there is no NVLink between nodes), `xml` is NOT resized — it stays at `NCCL_TOPO_XML_MAX_NODES = 256` capacity while the fuse loop attempts to merge `nLocalRanks` peer topologies (each up to 256 nodes). This is the cap that our v0.2.2 run hit.

### Fuse recursion

`src/graph/xml.cc:309-323`:

```cpp
ncclResult_t ncclTopoFuseXml(struct ncclXml* dst, struct ncclXml* src) {
  struct ncclXmlNode* topNodeDst;
  NCCLCHECK(xmlFindTag(dst, "system", &topNodeDst));

  if (topNodeDst == NULL) {
    xmlAddTree(dst, NULL, src->nodes);
    return ncclSuccess;
  }

  struct ncclXmlNode* topNodeSrc;
  NCCLCHECK(xmlFindTag(src, "system", &topNodeSrc));

  NCCLCHECK(xmlTopoFuseXmlRecursive(dst, topNodeDst, topNodeSrc));

  return ncclSuccess;
}
```

`xmlTopoFuseXmlRecursive` (xml.cc:~270-308) walks the source tree and either calls `xmlAddTree` (append new subtree) when no dedup match is found, or recurses into a matching destination subtree when one is found. The dedup decision is attribute-based. This is the step where "merge math" gets complicated: matching subtrees are dedupe'd (not duplicated), and only mismatches are appended. The exact dedup predicate is what drives whether 2 x 122 = 244 stays below 256 via aggressive dedup, or blows past 256 when dedup fails.

Key open question: what does autogen produce attribute-wise that makes its two-host trees dedupe to <256 total, while our static XML does not?


## Hypotheses for the real mechanism

The cause is our topology patch (pinned by the v0.2.4 isolation image). The mechanism — why this specific patch produces `too many XML nodes (max 256)` during cross-node fuse when autogen at the same per-host count does not — remains unresolved. We have three candidate hypotheses; each is plausible, none is confirmed by this report. We would value AWS guidance to identify which (if any) is correct, or to propose a fourth we have missed.

The v0.2.4 run rules in a useful constraint: plugin-side initialization code paths that execute on both v0.2.4 (PASS) and v0.2.2 (FAIL) cannot be the mechanism, because v0.2.4 exercises the same aws-ofi-nccl `.so` entry points as v0.2.2 minus the platform_data entry for `^p5\.48xlarge$`. The mechanism must be specific to what happens after the plugin sets `NCCL_TOPO_FILE` and NCCL loads our static XML instead of running its own autogen.

### Hypothesis A: double-path XML installation causes duplicate load

The v0.2.2 Dockerfile installs `p5.48xl-topo.xml` to two locations:

1. `/opt/aws-ofi-nccl/share/aws-ofi-nccl/xml/p5.48xl-topo.xml` (via `make install` from autotools `dist_xml_DATA`)
2. `/opt/amazon/ofi-nccl/share/aws-ofi-nccl/xml/p5.48xl-topo.xml` (via `install -D` directive in Dockerfile to match the older path the AWS EFA installer uses)

If the plugin scans both paths for topology files and emits them additively into the XML (or if NCCL processes both files somehow during init), the single-host topology could be 2 x 122 = 244 nodes before any cross-node merge. Cross-node merge would then easily overflow.

**Evidence for**: Double-install is an anomaly unique to v0.2.2; v0.2.1 has neither path populated; v0.2.3 reverts both.
**Evidence against**: We have not inspected the aws-ofi-nccl source to confirm it scans multiple XML paths. The plugin uses `platform->topology` as a filename only, and resolves it against a single share dir determined at build time. Unless there is an env manipulation that adds a second path, this hypothesis requires mechanism we cannot locate in the source.

### Hypothesis B: autogen and static-XML load concurrently

The plugin's behavior when both `NCCL_TOPO_FILE` is set AND NCCL has fallback code to auto-discover PCI topology may be additive rather than replacement. If NCCL loads our static XML first (122 nodes into a 256-capacity arena), then executes its autogen pass on top of that arena (adding ~126 autogen nodes), the single-host total reaches 248. Cross-node merge of two such hosts hits the cap almost immediately.

**Evidence for**: NCCL 2.30.4 `topo.cc:1530-1546` reads `NCCL_TOPO_FILE` then calls `xmlFindTag("cpu", ...)` and rewrites `host_hash`. If NCCL treats a loaded file as a starting point rather than a complete topology, subsequent autodiscovery could add to it.
**Evidence against**: `topo.cc:1534-1537` has `if (xmlTopoFile)` / else branches — the else does autodiscovery from `/var/run/nvidia-topologyd/virtualTopology.xml` as fallback, not additively. Full autodiscovery via `ncclTopoFillGpu` / `ncclTopoFillNet` should not run when `NCCL_TOPO_FILE` is set.

### Hypothesis C: non-MNNVL communicator + attribute-mismatch defeats dedup

On p5.48xlarge there is no NVLink between hosts (`comm->MNNVL == false`), so `topo.cc:1645-1649` does NOT resize the dst xml to `nLocalRanks * 256`. It stays at 256 capacity.

Cross-node fuse (`xml.cc:309 ncclTopoFuseXml` -> `xmlTopoFuseXmlRecursive`) walks source and dest trees and decides per-subtree whether to dedup (match and recurse) or append (add-tree). The dedup predicate uses node attributes. Autogen produces XML with rich attributes (`vendor`, `device`, `subsystem_vendor`, `subsystem_device`, `link_speed`, `link_width`, `host_hash` rewritten to `getHostHash()` at load). Our static XML is sparse on attributes — it omits `vendor`, `device`, `subsystem_vendor`, `subsystem_device` on PCI bridges, and omits `host_hash` on `<cpu>`.

If the dedup predicate matches on attributes that exist in autogen but not in our static XML, then autogen's two-host trees dedup aggressively (PCI bridges with identical attributes collapse into one subtree, only unique NIC/GPU leaves append), staying below 256 total. Our static XML, lacking those attributes, may fall out of dedup and every single subtree appends as new — crossing 256 far before the second host's full tree is processed.

**Evidence for**: Among the three candidate hypotheses we considered, this is the one that most readily accommodates the 122-static-fails vs 126-autogen-passes observation at otherwise near-equal per-host node counts, because merge behavior is attribute-driven and our static XML is attribute-sparse.
**Evidence against**: We have not read `xmlTopoFuseXmlRecursive` in full, and we have not run a controlled test that varies only attribute richness. The exact dedup predicate (which attributes matter, how many must match, and whether missing attributes count as "match" or "miss") is unverified on our side. We present this as a plausible candidate, not a confirmed mechanism.

### What we have ruled out

- **Plain cap arithmetic**: autogen produces 126 nodes per host (4 more than our 122-node static XML), yet autogen passes on both v0.2.1 and v0.2.4. The cap alone does not explain it.
- **NCCL version regression**: same NCCL 2.30.4 wheel in all three images.
- **libfabric / kernel / driver**: identical across all three images.
- **aws-ofi-nccl plugin regression (within tested scenario)**: v0.2.4 uses the same commit `206c02c` as v0.2.2 but passes cleanly in the 2-pod HyperPod p5.48xlarge NCCL 2.30.4 init + all_reduce test. The version bump from `6e504db` to `206c02c` is not the cause of our observed failure. This does not exclude the possibility of other v1.19.1 regressions in scenarios we did not test.
- **Confounded A/B attribution**: the initial v0.2.1 vs v0.2.2 comparison conflated plugin version with patch presence. The v0.2.4 image resolves that confound and pins cause on the patch alone.

## Ask / what AWS can help with

We have a reproducible failure isolated to our own topology patch via a three-way controlled experiment. Within the tested scenario (2-pod HyperPod p5.48xlarge, NCCL 2.30.4, `init_process_group` + `all_reduce`) we observed no regression in the v1.19.1 plugin. What we still do not know is the exact mechanism by which our static XML, loaded via `NCCL_TOPO_FILE`, pushes the cross-node fuse over the 256-node cap when autogen at the same per-host count does not. We would value your team's help identifying which mechanism is at play, primarily so other customers attempting a similar p5.48xlarge static XML approach don't hit the same surprise.

Note on choice of channel: we considered filing an AWS SIM ticket and opted against it, because the three-way experiment shows the upstream v1.19.1 plugin is not regressed in our tested scenario — this is an incident with our own patch, not a bug report. This public writeup is the channel we chose instead. If AWS would prefer that a SIM be filed for the mechanism questions in Appendix G, we are happy to open one on request.

Specifically, any of these would unblock us:

1. **Confirm or refute Hypothesis A**: Does the plugin ever scan two topology paths? Or does the Dockerfile's double-install into `/opt/aws-ofi-nccl/share/...` and `/opt/amazon/ofi-nccl/share/...` only materialize at one path that the plugin reads?
2. **Confirm or refute Hypothesis B**: In NCCL 2.30.4 with `NCCL_TOPO_FILE` set to a valid file, does any autodiscovery code run after the file is loaded (adding to the same XML arena), or is the file considered the complete topology?
3. **Confirm or refute Hypothesis C**: In `xmlTopoFuseXmlRecursive`, what is the dedup predicate? Which attributes are required to match for two subtrees to be considered equivalent? Does a missing attribute count as "match" or "miss"?
4. **Or a fourth hypothesis we missed**: If AWS has internal knowledge of why static XMLs for p5-class instances misbehave in cross-node fuse where autogen succeeds, we'd value that context.

We are also happy to run any specific diagnostic you suggest. Fast-turnaround options available to us:

- Dump autogen topology via `NCCL_TOPO_DUMP_FILE` on both pods (already done for one-side; can do both sides).
- Enable `NCCL_DEBUG=INFO` + `NCCL_DEBUG_SUBSYS=GRAPH` and send the full log from a v0.2.2 FAIL run so the merge trace is visible.
- Rebuild with our static XML stripped down to the autogen's 126-node structure (including `host_hash`, `vendor`, `device` attributes) and re-run, to test whether attribute richness is the variable.
- Rebuild with static XML installed to only one path (removing the double-install) and re-run, to test Hypothesis A.

Our preference is to get a definitive "autogen is sufficient for p5.48xlarge, don't ship a static XML" recommendation from AWS, plus a documented rationale so other customers don't encounter the same regression we introduced. If that's not the right answer, tell us what is.

## How we resolved it

We closed PR #1226 after the three-way controlled experiment isolated our patch as the cause. We tagged v0.2.3-sm90a on 2026-05-06T20:23:22Z as a revert-only build, and v0.2.4-sm90a on 2026-05-06T20:55Z as the confound-isolation image that keeps the plugin bump but drops the patch. v0.2.4 passes cleanly, proving the post-revert path is stable.

- **v0.2.3 GitHub Actions build**: https://github.com/antonai-work/deepep-v2-efa-base/actions/runs/14446677012
- **v0.2.4 GitHub Actions build**: https://github.com/antonai-work/deepep-v2-efa-base/actions/runs/25460504491
- **v0.2.4 ECR digest (now the recommended base)**: `sha256:615c52eac8f054fcdcd526f502cbadcf8066a7b29982f09c7de7df4ca7953e5a`
- **Validation evidence**: Wave 18f 2026-05-06T21:00 UTC; 16 ranks SUCCESS, 126 autogen nodes, zero NCCL warnings in either pod log.

## Evidence files attached

1. **Closed PR (the regression we introduced)**: https://github.com/aws/aws-ofi-nccl/pull/1226
2. **Base image repo (public)**: https://github.com/antonai-work/deepep-v2-efa-base
3. **Integration repo (private)**: antonai-work/deepep-v2-integration — private; referenced paths mirrored here where possible.
4. **This report (permanent link)**: https://github.com/antonai-work/deepep-v2-efa-base/blob/main/docs/INCIDENT-PR1226-AWS-OFI-NCCL-XML-OVERFLOW.md
5. **Wave 18 empirical report**: internal evidence ledger (private repo); verbatim logs quoted inline in this report.
6. **Eric Raute review quote** (PR #1226): "I don't understand why you were hitting the 'too many XML nodes' error, we haven't seen that. But we don't want to get rid of the topology autogeneration -- that was required to properly handle the case where not all GPUs are visible inside a container."

## Contact

- **Reporter**: Anton Alexander (external customer)
- **Email**: <redacted-email>
- **Referrer**: Brian Barrett (AWS, aws-ofi-nccl contributor)

We are available for follow-up testing or to provide additional logs/snapshots from the HyperPod cluster on short notice.

### Identifiers redacted for public publication

The following identifiers were present in our internal notes but are redacted in this public document. AWS engineers who need the full values can request them privately via the Contact addresses above.

- **SageMaker HyperPod EKS cluster ID** (redacted from the Environment section).
- **Two p5.48xlarge HyperPod instance IDs** (referred to in this document as "node-A" and "node-B"). The original values were `hyperpod-i-...` format.

The following identifiers are intentionally NOT redacted because they are already public or are load-bearing for reproducibility:

- AWS account ID redacted in ECR paths; image digests are stable and verifiable against the public GHCR mirror at `ghcr.io/antonai-work/deepep-v2-efa-base`.
- ECR image digests (`sha256:...`) — public, pinned for exact reproduction.
- GitHub Actions build run URLs under `antonai-work/deepep-v2-efa-base` — public.
- Upstream commit SHAs (`6e504db...`, `206c02c...`, `1933fdd...`) — public.
- Verbatim pod log excerpts — scanned for proprietary content; none found.


## Appendix A: Full verbatim pod logs (three-way)

This appendix reproduces full pod logs for all three images: v0.2.1 PASS (old plugin, no patch), v0.2.4 PASS (new plugin, no patch), and v0.2.2 FAIL (new plugin + patch). AWS engineers asked for verbatim text rather than summaries; we include pod tails in full so there is no ambiguity about what NCCL emitted in each case.

### A.1 v0.2.1 PASS — pod 0 (ranks 0-7), final 100 lines

Excerpt from `/tmp/wave18/pod0.log` showing all 8 ranks reaching `all_reduce OK`, `barrier complete`, and `SUCCESS - no XML overflow error`, followed by the `grep -c '<.*>'` output of `126` for the autogen topology dump:

```
w18-nccl-0:112:112 [2] NCCL INFO Channel 15/0 : 2[2] -> 1[1] via P2P/CUMEM
w18-nccl-0:115:115 [5] NCCL INFO Channel 04/0 : 5[5] -> 4[4] via P2P/CUMEM
w18-nccl-0:111:111 [1] NCCL INFO Channel 11/0 : 1[1] -> 0[0] via P2P/CUMEM
w18-nccl-0:116:116 [6] NCCL INFO Channel 05/0 : 6[6] -> 5[5] via P2P/CUMEM
w18-nccl-0:114:114 [4] NCCL INFO Channel 14/0 : 4[4] -> 3[3] via P2P/CUMEM
w18-nccl-0:114:114 [4] NCCL INFO Channel 15/0 : 4[4] -> 3[3] via P2P/CUMEM
w18-nccl-0:117:117 [7] NCCL INFO Channel 08/0 : 7[7] -> 6[6] via P2P/CUMEM
w18-nccl-0:111:111 [1] NCCL INFO Channel 12/0 : 1[1] -> 0[0] via P2P/CUMEM
w18-nccl-0:116:116 [6] NCCL INFO Channel 07/0 : 6[6] -> 5[5] via P2P/CUMEM
w18-nccl-0:115:115 [5] NCCL INFO Channel 06/0 : 5[5] -> 4[4] via P2P/CUMEM
w18-nccl-0:116:116 [6] NCCL INFO Channel 08/0 : 6[6] -> 5[5] via P2P/CUMEM
w18-nccl-0:117:117 [7] NCCL INFO Channel 09/0 : 7[7] -> 6[6] via P2P/CUMEM
w18-nccl-0:111:111 [1] NCCL INFO Channel 13/0 : 1[1] -> 0[0] via P2P/CUMEM
w18-nccl-0:115:115 [5] NCCL INFO Channel 07/0 : 5[5] -> 4[4] via P2P/CUMEM
w18-nccl-0:116:116 [6] NCCL INFO Channel 09/0 : 6[6] -> 5[5] via P2P/CUMEM
w18-nccl-0:117:117 [7] NCCL INFO Channel 10/0 : 7[7] -> 6[6] via P2P/CUMEM
w18-nccl-0:111:111 [1] NCCL INFO Channel 14/0 : 1[1] -> 0[0] via P2P/CUMEM
w18-nccl-0:111:111 [1] NCCL INFO Channel 15/0 : 1[1] -> 0[0] via P2P/CUMEM
w18-nccl-0:116:116 [6] NCCL INFO Channel 10/0 : 6[6] -> 5[5] via P2P/CUMEM
w18-nccl-0:117:117 [7] NCCL INFO Channel 11/0 : 7[7] -> 6[6] via P2P/CUMEM
w18-nccl-0:115:115 [5] NCCL INFO Channel 08/0 : 5[5] -> 4[4] via P2P/CUMEM
w18-nccl-0:116:116 [6] NCCL INFO Channel 11/0 : 6[6] -> 5[5] via P2P/CUMEM
w18-nccl-0:117:117 [7] NCCL INFO Channel 12/0 : 7[7] -> 6[6] via P2P/CUMEM
w18-nccl-0:115:115 [5] NCCL INFO Channel 09/0 : 5[5] -> 4[4] via P2P/CUMEM
w18-nccl-0:116:116 [6] NCCL INFO Channel 12/0 : 6[6] -> 5[5] via P2P/CUMEM
w18-nccl-0:117:117 [7] NCCL INFO Channel 13/0 : 7[7] -> 6[6] via P2P/CUMEM
w18-nccl-0:115:115 [5] NCCL INFO Channel 10/0 : 5[5] -> 4[4] via P2P/CUMEM
w18-nccl-0:116:116 [6] NCCL INFO Channel 13/0 : 6[6] -> 5[5] via P2P/CUMEM
w18-nccl-0:117:117 [7] NCCL INFO Channel 14/0 : 7[7] -> 6[6] via P2P/CUMEM
w18-nccl-0:115:115 [5] NCCL INFO Channel 11/0 : 5[5] -> 4[4] via P2P/CUMEM
w18-nccl-0:116:116 [6] NCCL INFO Channel 15/0 : 6[6] -> 5[5] via P2P/CUMEM
w18-nccl-0:115:115 [5] NCCL INFO Channel 12/0 : 5[5] -> 4[4] via P2P/CUMEM
w18-nccl-0:115:115 [5] NCCL INFO Channel 14/0 : 5[5] -> 4[4] via P2P/CUMEM
w18-nccl-0:115:115 [5] NCCL INFO Channel 15/0 : 5[5] -> 4[4] via P2P/CUMEM
w18-nccl-0:113:113 [3] NCCL INFO Connected all trees
w18-nccl-0:114:114 [4] NCCL INFO Connected all trees
w18-nccl-0:112:112 [2] NCCL INFO Connected all trees
w18-nccl-0:115:115 [5] NCCL INFO Connected all trees
w18-nccl-0:110:110 [0] NCCL INFO Connected all trees
w18-nccl-0:111:111 [1] NCCL INFO Connected all trees
w18-nccl-0:116:116 [6] NCCL INFO Connected all trees
w18-nccl-0:117:117 [7] NCCL INFO Connected all trees
[wave18] rank=0 all_reduce OK, sum[0]=16.0
[wave18] rank=1 all_reduce OK, sum[0]=16.0
[wave18] rank=2 all_reduce OK, sum[0]=16.0
[wave18] rank=3 all_reduce OK, sum[0]=16.0
[wave18] rank=4 all_reduce OK, sum[0]=16.0
[wave18] rank=5 all_reduce OK, sum[0]=16.0
[wave18] rank=6 all_reduce OK, sum[0]=16.0
[wave18] rank=7 all_reduce OK, sum[0]=16.0
/usr/local/lib/python3.12/dist-packages/torch/distributed/distributed_c10d.py:4876: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning.
  warnings.warn(  # warn only once
[wave18] rank=0 barrier complete[wave18] rank=1 barrier complete
[wave18] rank=5 barrier complete
[wave18] rank=3 barrier complete[wave18] rank=7 barrier complete[wave18] rank=4 barrier complete[wave18] rank=2 barrier complete[wave18] rank=6 barrier complete




w18-nccl-0:110:110 [0] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=0 SUCCESS - no XML overflow error
w18-nccl-0:112:112 [2] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=2 SUCCESS - no XML overflow error
w18-nccl-0:117:117 [7] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=7 SUCCESS - no XML overflow error
w18-nccl-0:114:114 [4] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=4 SUCCESS - no XML overflow error
w18-nccl-0:113:113 [3] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=3 SUCCESS - no XML overflow error
w18-nccl-0:115:115 [5] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=5 SUCCESS - no XML overflow error
w18-nccl-0:111:111 [1] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=1 SUCCESS - no XML overflow error
w18-nccl-0:116:116 [6] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=6 SUCCESS - no XML overflow error
+ '[' -f /tmp/nccl_topo.xml ']'
+ echo '[wave18] NCCL topology XML node count:'
[wave18] NCCL topology XML node count:
+ grep -c '<.*>' /tmp/nccl_topo.xml
126
```

### A.2 v0.2.1 PASS — pod 1 (ranks 8-15), final excerpt

```
[wave18] rank=10 all_reduce OK, sum[0]=16.0
[wave18] rank=11 all_reduce OK, sum[0]=16.0
[wave18] rank=12 all_reduce OK, sum[0]=16.0
[wave18] rank=13 all_reduce OK, sum[0]=16.0
[wave18] rank=14 all_reduce OK, sum[0]=16.0
[wave18] rank=15 all_reduce OK, sum[0]=16.0
[wave18] rank=10 barrier complete[wave18] rank=9 barrier complete[wave18] rank=11 barrier complete[wave18] rank=8 barrier complete[wave18] rank=13 barrier complete[wave18] rank=14 barrier complete

[wave18] rank=12 barrier complete[wave18] rank=15 barrier complete




w18-nccl-1:148:148 [0] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=8 SUCCESS - no XML overflow error
w18-nccl-1:150:150 [2] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=10 SUCCESS - no XML overflow error
w18-nccl-1:152:152 [4] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=12 SUCCESS - no XML overflow error
w18-nccl-1:155:155 [7] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=15 SUCCESS - no XML overflow error
w18-nccl-1:153:153 [5] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=13 SUCCESS - no XML overflow error
w18-nccl-1:151:151 [3] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=11 SUCCESS - no XML overflow error
w18-nccl-1:149:149 [1] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=9 SUCCESS - no XML overflow error
w18-nccl-1:154:154 [6] NCCL INFO NET/OFI gin: Finalizing
[wave18] rank=14 SUCCESS - no XML overflow error
+ '[' -f /tmp/nccl_topo.xml ']'
[wave18] no /tmp/nccl_topo.xml generated (expected if NCCL_TOPO_DUMP_FILE not honored)
[wave18] grep for 'too many XML nodes' or 'XML' errors:
+ echo '[wave18] no /tmp/nccl_topo.xml generated (expected if NCCL_TOPO_DUMP_FILE not honored)'
+ echo '[wave18] grep for '\''too many XML nodes'\'' or '\''XML'\'' errors:'
+ grep -i 'too many\|xml.*error\|xml.*fail' /tmp/wave18_nccl_output.log
[wave18] rank=8 SUCCESS - no XML overflow error
[wave18] rank=10 SUCCESS - no XML overflow error
[wave18] rank=12 SUCCESS - no XML overflow error
[wave18] rank=15 SUCCESS - no XML overflow error
[wave18] rank=13 SUCCESS - no XML overflow error
[wave18] rank=11 SUCCESS - no XML overflow error
[wave18] rank=9 SUCCESS - no XML overflow error
[wave18] rank=14 SUCCESS - no XML overflow error
+ echo '[wave18] grep for NCCL WARN:'
[wave18] grep for NCCL WARN:
+ grep 'NCCL WARN' /tmp/wave18_nccl_output.log
+ head -20
+ echo '=== Wave 18 test complete, pod staying alive for log collection ==='
+ sleep infinity
=== Wave 18 test complete, pod staying alive for log collection ===
```

Note: the `grep 'NCCL WARN'` output is empty (no matches) on both pods. This is load-bearing evidence that v0.2.1 produces zero NCCL warnings, not just no overflow warnings.

### A.3 v0.2.4 PASS — pod 0 (ranks 0-7), Wave 18f 2026-05-06T21:00 UTC

v0.2.4 is the confound-isolation image: same aws-ofi-nccl commit `206c02c` as v0.2.2, same Dockerfile, patch removed. It auto-generates 126 XML nodes per host and completes 16-rank cross-node init cleanly.

Final excerpt from `/tmp/wave18f/pod0.log` (lines 1640-1723, verbatim):

```
w18f-nccl-0:115:115 [5] NCCL INFO NET/OFI gin: Finalizing
[wave18f] rank=5 SUCCESS - no XML overflow error
w18f-nccl-0:112:112 [2] NCCL INFO NET/OFI gin: Finalizing
[wave18f] rank=2 SUCCESS - no XML overflow error
w18f-nccl-0:111:111 [1] NCCL INFO NET/OFI gin: Finalizing
[wave18f] rank=1 SUCCESS - no XML overflow error
w18f-nccl-0:116:116 [6] NCCL INFO NET/OFI gin: Finalizing
[wave18f] rank=6 SUCCESS - no XML overflow error
+ '[' -f /tmp/nccl_topo.xml ']'
+ echo '[wave18f] NCCL topology XML node count:'
+ grep -c '<.*>' /tmp/nccl_topo.xml
[wave18f] NCCL topology XML node count:
126
+ echo '[wave18f] first 50 lines of generated topology:'
+ head -50 /tmp/nccl_topo.xml
[wave18f] first 50 lines of generated topology:
<system version="1">
  <cpu host_hash="0xb595a027a7e3fab4" numaid="0" affinity="00000000,0000ffff,ffffffff" arch="x86_64" vendor="AuthenticAMD" familyid="175" modelid="1">
    <pci busid="0000:45:00.0" class="0x060400" vendor="0x1d0f" device="0x0200" subsystem_vendor="0x1d0f" subsystem_device="0x0200" link_speed="16.0 GT/s PCIe" link_width="8">
      <pci busid="0000:52:00.0" link_speed="32.0 GT/s PCIe" link_width="16" class="0x020000" vendor="0x1d0f" device="0xefa1" subsystem_vendor="0x1d0f" subsystem_device="0xefa1">
        <nic>
          <net name="rdmap82s0" dev="0" latency="75" speed="400000" port="1" guid="0xa0103e500000000" maxconn="262144" gdr="1" net="1" gin="1"/>
        </nic>
      </pci>
      <pci busid="0000:53:00.0" class="0x030200" vendor="0x10de" device="0x2330" subsystem_vendor="0x10de" subsystem_device="0x16c1" link_speed="32.0 GT/s PCIe" link_width="16">
```

Node count: **126** (identical to v0.2.1 baseline). The autogen XML structure is identical between v0.2.1 and v0.2.4 within the excerpt visible in the logs — same `<cpu host_hash="...">`, same PCI class/vendor/device attributes, same sparse NIC placement per GPU island.

The v0.2.4 `host_hash` value (`0xb595a027a7e3fab4`) differs from v0.2.1's (`0x23d79eeea3fd2808`) only because they ran on different HyperPod nodes in different test windows. That's expected — `getHostHash()` varies per host per process.

SUCCESS markers at the end of pod 0:

```
[wave18f] rank=7 SUCCESS - no XML overflow error
[wave18f] rank=4 SUCCESS - no XML overflow error
[wave18f] rank=3 SUCCESS - no XML overflow error
[wave18f] rank=0 SUCCESS - no XML overflow error
[wave18f] rank=5 SUCCESS - no XML overflow error
[wave18f] rank=2 SUCCESS - no XML overflow error
[wave18f] rank=1 SUCCESS - no XML overflow error
[wave18f] rank=6 SUCCESS - no XML overflow error
+ echo '[wave18f] grep for NCCL WARN:'
[wave18f] grep for NCCL WARN:
+ grep 'NCCL WARN' /tmp/wave18f_nccl_output.log
+ head -20
+ echo '=== Wave 18f test complete, pod staying alive for log collection ==='
+ sleep infinity
=== Wave 18f test complete, pod staying alive for log collection ===
```

Zero `NCCL WARN` lines in either pod log. Total log size: 1723 (pod 0) + 1647 (pod 1) = 3370 lines, no overflow errors anywhere.

### A.4 v0.2.2 FAIL — pod 0 (previous session, 2026-05-06 18:33 UTC)

Full verbatim tail from the failing v0.2.2 run (all 16 ranks emitted the same error within ~300ms):

```
[2026-05-06 20:00:45] deepep-cu13-xnode-0:257:257 [2] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)

[2026-05-06 20:00:45] deepep-cu13-xnode-0:259:259 [4] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)

[2026-05-06 20:00:45] deepep-cu13-xnode-0:261:261 [6] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)

[2026-05-06 20:00:45] deepep-cu13-xnode-0:256:256 [1] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)
[W506 20:00:45.737645814 ProcessGroupNCCL.cpp:1575] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())

[2026-05-06 20:00:45] deepep-cu13-xnode-0:255:255 [0] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)

[2026-05-06 20:00:45] deepep-cu13-xnode-0:260:260 [5] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)

[2026-05-06 20:00:45] deepep-cu13-xnode-0:262:262 [7] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)

[2026-05-06 20:00:45] deepep-cu13-xnode-0:258:258 [3] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)

[2026-05-06 20:00:45] deepep-cu13-xnode-0:309:309 [0] graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)
[W506 20:00:45.740326676 ProcessGroupNCCL.cpp:1575] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
```

Process group traceback on every rank:

```
Traceback (most recent call last):
  File "/opt/DeepEP/tests/elastic/test_ep.py", line ...
    dist.init_process_group(backend="nccl")
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/c10d_logger.py", ...
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/distributed_c10d.py", ...
torch.distributed.DistBackendError: NCCL error in: /pytorch/torch/csrc/distributed/c10d/NCCLUtils.cpp:94, internal error -- please report this issue to the NCCL developers, NCCL version 2.30.4
ncclInternalError: Internal check failed.
Last error:
Error : too many XML nodes (max 256)
```


## Appendix B: Configuration dump (preflight & env vars from v0.2.1 PASS run)

This appendix snapshots the pod environment so AWS can verify the build-time and runtime variables that went into the PASS case. Reproducing the FAIL case requires only swapping the image digest; everything else (env, manifest, command) is held constant.

### B.1 EFA device enumeration (inside pod, v0.2.1)

```
--- Preflight checks ---
EFA devices: 32
```

(Count of entries under `/dev/infiniband/`.)

### B.2 libfabric provider info

```
provider: efa
    fabric: efa-direct
    domain: rdmap82s0
    version: 2.3
    type: FI_EP_RDM
    protocol: FI_PROTO_EFA
```

(Excerpt from `fi_info -p efa` on p5.48xlarge. Provider is `efa-direct`, which is what aws-ofi-nccl-gin requires.)

### B.3 Plugin topology XML path

On v0.2.1 (unpatched), the plugin share directory is empty of p5-specific XMLs:

```
$ ls -la /opt/amazon/ofi-nccl/share/aws-ofi-nccl/xml/ 2>/dev/null
total 20
drwxr-xr-x 2 root root 4096 May  5 11:14 .
drwxr-xr-x 3 root root 4096 May  5 11:14 ..
-rw-r--r-- 1 root root 2314 Feb 15 12:00 p4d-24xl-topo.xml
-rw-r--r-- 1 root root 2314 Feb 15 12:00 p4de-24xl-topo.xml
```

(No `p5.48xl-topo.xml`; autogen takes over because `platform_data.topology = NULL` for `p5/p5e` in stock v1.19.1.)

On v0.2.2 (patched), the same directory additionally contained our custom file `p5.48xl-topo.xml`. The actual XML file size (from the source-of-truth copy in our repo at `base/deepep-base-v2/assets/p5.48xl-topo.xml`) is **10205 bytes** — we note the exact byte count because we do not have a verbatim `ls -la` capture from the v0.2.2 pod filesystem and prefer to cite the authoritative source rather than reconstruct.

### B.4 Relevant runtime env vars

```
NCCL_DEBUG=INFO
NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml
LD_LIBRARY_PATH=/opt/nvidia-driver:/opt/amazon/efa/lib:/opt/amazon/aws-ofi-nccl/lib:/usr/local/cuda/lib64:/usr/local/lib
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=all
```

On v0.2.1 the plugin does NOT set `NCCL_TOPO_FILE` (because `platform->topology` is NULL for p5).
On v0.2.2 the plugin DOES set `NCCL_TOPO_FILE=/opt/aws-ofi-nccl/share/aws-ofi-nccl/xml/p5.48xl-topo.xml` at init.

### B.5 NCCL version string check

```
$ python3 -c "import torch; print(torch.cuda.nccl.version())"
(2, 30, 4)

$ strings /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 | grep 'NCCL version' | head -1
NCCL version 2.30.4+cuda13.0
```

Identical NCCL wheel on v0.2.1 and v0.2.2. The only binary difference between images is the aws-ofi-nccl `.so` (rebuilt against our patch in v0.2.2) and the presence of the baked XML file.

## Appendix C: Complete patch text

The patch applied to aws-ofi-nccl v1.19.1 in v0.2.2. This is the full git-format patch committed to `base/deepep-base-v2/upstream-patches/0001-topology-add-p5.48xlarge-topology-XML.patch`.

```
From 79565c0e5a384bc35bf8a0e6378e60f88c2b2565 Mon Sep 17 00:00:00 2001
From: Anton Alexander <<redacted-email>>
Date: Wed, 6 May 2026 19:16:33 +0000
Subject: [PATCH] topology: add p5.48xlarge topology XML

This adds a pre-computed NCCL topology XML for p5.48xlarge, so that
the plugin sets NCCL_TOPO_FILE at initialization and NCCL does not
fall back to its PCI tree auto-discovery.

Without this file, NCCL's auto-discovery enumerates every /sys/class/
pci_bus entry and exceeds the hardcoded NCCL_TOPO_XML_MAX_NODES=256
(src/graph/xml.h) on p5.48xlarge hosts where 32 EFA interfaces + 8 H100
GPUs + associated PCIe switches, bridges, and root complexes push the
node count well above the cap.

[NOTE (2026-05-06, post-empirical): The above rationale is WRONG.
 Empirical A/B test on same hardware shows stock autogen produces
 126 nodes, well below 256. This patch is the regression, not the
 fix. Keeping the original commit message here for historical
 accuracy; see docs/INCIDENT-PR1226-AWS-OFI-NCCL-XML-OVERFLOW.md for the
 corrected analysis.]

Result without this patch: ncclInternalError during
init_process_group with 'too many XML nodes (max 256)'. This is
reproducible on any distributed workload that calls
torch.distributed.init_process_group(backend='nccl') on p5.48xlarge,
including recently on AWS HyperPod EKS.

This patch mirrors the existing approach for p4d / p4de / g5.

Topology structure:
- 8 PCIe switches (one per GPU island)
- Each island: 1 H100 + 4 EFA devices (2 Nitro cards x 2 ports)
- CPU at numaid=0 (p5.48xlarge runs AMD Genoa in NPS=1)
- Total XML nodes: 122 (well under 256 cap)

The platform-aws.cpp entry for p5.48xlarge must precede the wildcard
regex entry for 'p5/p5e' so that exact-match resolution beats fallback
regex.

Tested 2026-05-06 on AWS HyperPod p5.48xlarge, 2-node DeepEP V2 cross-
node EFA validation. With this patch + baked-in XML, NCCL init passes
without the 256-node overflow.

[NOTE: "passes" above refers to a SINGLE-NODE test during patch
 development. The cross-node failure was not discovered until the
 patch shipped in v0.2.2-sm90a.]

Signed-off-by: Anton Alexander <<redacted-email>>
---
 topology/p5.48xl-topo.xml | 204 ++++++++++++++++++++++++++++++++++++++
 topology/Makefile.am      |   1 +
 src/platform-aws.cpp      |  16 +++
 3 files changed, 221 insertions(+)
 create mode 100644 topology/p5.48xl-topo.xml
```

The full `topology/p5.48xl-topo.xml` file content (204 lines) is enumerated below. Node count breakdown: 1 `<system>`, 1 `<cpu>`, 48 `<pci>`, 8 `<gpu>`, 32 `<nic>`, 32 `<net>` = 122 total.

```xml
<system version="1">
  <cpu numaid="0" affinity="00000000,00000000,0000ffff,ffffffff" arch="x86_64" vendor="AuthenticAMD" familyid="25" modelid="17">
    <pci busid="0000:00:00.0" class="0x060400" link_speed="16.0 GT/s PCIe" link_width="16">
      <pci busid="0000:4f:00.0" class="0x030200" link_speed="32.0 GT/s PCIe" link_width="16">
        <gpu dev="0" sm="90" rank="0" gdr="1"/>
      </pci>
      <pci busid="0000:50:00.0" class="0x020000" link_speed="16.0 GT/s PCIe" link_width="8">
        <nic>
          <net name="rdmap79s0" dev="0" speed="100000" port="1" latency="0.0" guid="0" maxconn="1024" gdr="1"/>
        </nic>
      </pci>
      <pci busid="0000:51:00.0" class="0x020000" link_speed="16.0 GT/s PCIe" link_width="8">
        <nic>
          <net name="rdmap80s0" dev="1" speed="100000" port="1" latency="0.0" guid="1" maxconn="1024" gdr="1"/>
        </nic>
      </pci>
      <pci busid="0000:52:00.0" class="0x020000" link_speed="16.0 GT/s PCIe" link_width="8">
        <nic>
          <net name="rdmap81s0" dev="2" speed="100000" port="1" latency="0.0" guid="2" maxconn="1024" gdr="1"/>
        </nic>
      </pci>
      <pci busid="0000:53:00.0" class="0x020000" link_speed="16.0 GT/s PCIe" link_width="8">
        <nic>
          <net name="rdmap82s0" dev="3" speed="100000" port="1" latency="0.0" guid="3" maxconn="1024" gdr="1"/>
        </nic>
      </pci>
    </pci>
    <!-- ... 7 more similar blocks for PCIe switches 0000:01 through 0000:07 ... -->
    <!-- Full 204-line text lives at:
         https://github.com/antonai-work/deepep-v2-integration/blob/main/base/deepep-base-v2/upstream-patches/0001-topology-add-p5.48xlarge-topology-XML.patch
    -->
  </cpu>
</system>
```

Attribute profile: every `<pci>` has `busid`, `class`, `link_speed`, `link_width`. No `vendor`, `device`, `subsystem_vendor`, `subsystem_device` attributes. No `host_hash` on `<cpu>`. `<gpu>` has `dev`, `sm`, `rank`, `gdr` only (no `<nvlink>` children). `<net>` has `name`, `dev`, `speed`, `port`, `latency`, `guid`, `maxconn`, `gdr` (no `vendor`, `device`, `net=1`, `gin=1` markers that autogen emits).

Contrast with autogen output, which has all of the above attributes plus `<nvlink>` subtrees under each `<gpu>` (4 nvlink nodes x 8 GPUs = 32 additional subtree nodes).


## Appendix D: Prior successful run audit

Before v0.2.2 shipped, we had multiple successful 2-node p5.48xlarge NCCL/DeepEP runs. Every single one used stock aws-ofi-nccl autogen (no custom XML). This is supporting evidence for the claim that autogen has always been the viable path on this hardware.

| Date (UTC) | Workload | Result | Image | Topology source |
|---|---|---|---|---|
| 2026-04-29 13:10 | DeepEP V2 Megatron stress 10-cycle | p50 = 3.2 ms dispatch+combine, 16 GPU, 2 nodes, exit 0 | v0.2.0-sm90a | stock autogen |
| 2026-04-29 13:17 | Megatron 2-node training, 3 steps | loss 11.80 -> 11.32, 0.816 GB EFA TX, exit 0 | v0.2.0-sm90a | stock autogen |
| 2026-05-05 20:10 | NCCL allreduce bench, 16 GPU 2 nodes | 330 GB/s bus bandwidth at 1 GiB | v0.2.1-sm90a | stock autogen |
| 2026-05-05 22:56 | NCCL allreduce bench, 8 GPU single node | 357 GB/s bus bandwidth | v0.2.1-sm90a | stock autogen |
| 2026-05-06 20:35 | Wave 18 isolation PASS | 16 ranks all_reduce OK, 126 autogen nodes | v0.2.1-sm90a (plugin `6e504db`, no patch) | stock autogen |
| 2026-05-06 18:33 | Wave 16 FAIL (the incident) | "too many XML nodes (max 256)" on all 16 ranks | v0.2.2-sm90a (plugin `206c02c` + patch) | static 122-node XML |
| 2026-05-06 21:00 | Wave 18f isolation PASS | 16 ranks all_reduce OK, 126 autogen nodes | v0.2.4-sm90a (plugin `206c02c`, no patch) | stock autogen |

Every passing row uses stock autogen. The single failing row uses our static XML. Because v0.2.1 and v0.2.4 differ ONLY in aws-ofi-nccl commit (old vs new) and both pass, the version bump is not the cause. Because v0.2.4 and v0.2.2 share the same plugin commit and differ ONLY in presence/absence of our topology patch, the patch is the cause.

Eric Raute's initial review noted "I don't understand why you were hitting the 'too many XML nodes' error, we haven't seen that." The three-way test explains that puzzlement on our side: the error manifests only when our specific patch is applied; it is not reproducible against stock aws-ofi-nccl in the tested scenario. We do not speculate about AWS's internal image configurations.

## Appendix E: NCCL source citations

All line numbers are from NCCL 2.30.4-1 (git tag `v2.30.4-1`, commit `1933fdd6360a8bfccaa0166bd71bce363d32e5b6`). Source tree mirrored from `https://github.com/NVIDIA/nccl.git`.

### E.1 Cap definition

`src/graph/topo.h:192`:

```c
#define NCCL_TOPO_XML_MAX_NODES 256
```

This is the compile-time cap. Our local static XML (122 nodes) is well below this, and autogen (126 nodes) is also below it, so neither triggers an overflow at the parser level (`src/graph/xml.h:344-348 xmlAddNode`).

### E.2 Error sites

`src/graph/xml.h:344-348` (parser site, fired on initial XML read if arena is exhausted):

```c
static ncclResult_t xmlAddNode(struct ncclXml* xml, struct ncclXmlNode* parent, const char* subName, struct ncclXmlNode** sub) {
  if (xml->maxIndex == xml->maxNodes) {
    WARN("Error : too many XML nodes (max %d)", xml->maxNodes);
    return ncclInternalError;
  }
```

`src/graph/xml.h:379-383` (merge site, fired during cross-node fuse):

```c
static ncclResult_t xmlAddTree(struct ncclXml* dst, struct ncclXmlNode* parent, struct ncclXmlNode* srcNode) {
  if (dst->maxIndex == dst->maxNodes) {
    WARN("Error : too many XML nodes (max %d)", dst->maxNodes);
    return ncclInternalError;
  }
```

The v0.2.2 logs show line 381 specifically — this is the `xmlAddTree` check, which only fires during `ncclTopoFuseXml`'s recursive `xmlAddTree` descent. That tells us the initial parse of our 122-node file succeeded (122 < 256), and the overflow happened later when NCCL tried to merge a peer's topology into this arena.

### E.3 Allocation and MNNVL branch

`src/graph/topo.cc:1525-1537` (initial alloc + NCCL_TOPO_FILE load):

```c
  int* localRanks = NULL;
  struct ncclXml* rankXml;
  int localRank = -1, nLocalRanks = 0;
  struct ncclTopoNetInfo netInfo = {0};
  NCCLCHECK(xmlAlloc(&xml, NCCL_TOPO_XML_MAX_NODES));
  const char* xmlTopoFile = ncclGetEnv("NCCL_TOPO_FILE");
  if (xmlTopoFile) {
    INFO(NCCL_ENV, "NCCL_TOPO_FILE set by environment to %s", xmlTopoFile);
    NCCLCHECKGOTO(ncclTopoGetXmlFromFile(xmlTopoFile, xml, 1), ret, fail);
  } else {
    // Try default XML topology location
    NCCLCHECKGOTO(ncclTopoGetXmlFromFile("/var/run/nvidia-topologyd/virtualTopology.xml", xml, 0), ret, fail);
  }
```

Initial `xml` is sized to `NCCL_TOPO_XML_MAX_NODES = 256`.

`src/graph/topo.cc:1645-1658` (multi-node fuse):

```c
  if (comm->MNNVL) {
    // Ensure that we have enough room when fusing topos from multiple nodes.
    free(xml);
    xml = NULL;
    NCCLCHECKGOTO(xmlAlloc(&xml, nLocalRanks*NCCL_TOPO_XML_MAX_NODES), ret, fail);
  } else {
    // In the intra-node case there's no need to enlarge the topo xml.
    xml->maxIndex = 0;
  }
  for (int i = 0; i < nLocalRanks; i++) {
    struct ncclXml* peerXml = (struct ncclXml*)(mem+xmlMemSize(NCCL_TOPO_XML_MAX_NODES)*i);
    NCCLCHECKGOTO(ncclTopoConvertXml(peerXml, (uintptr_t)peerXml->nodes, 0), ret, fail);
    NCCLCHECKGOTO(ncclTopoFuseXml(xml, peerXml), ret, fail);
  }
```

The comment "In the intra-node case there's no need to enlarge the topo xml" is revealing: it confirms that the non-MNNVL branch (our p5.48xlarge case) keeps the dst xml at 256 capacity during fuse. Note: NCCL's "In the intra-node case" phrasing is arguably a slight misnomer — the `else` branch actually fires for any non-MNNVL communicator, which includes multi-node EFA as in our scenario. The behavior is correctly gated on `comm->MNNVL`, but the comment text may mislead readers into thinking the path is only for single-host communicators. The fuse of 2 x (126 autogen) = 252 worst-case fits; 2 x (122 static) = 244 worst-case also fits. But worst-case assumes zero dedup. Actual behavior depends on `xmlTopoFuseXmlRecursive`.

### E.4 Fuse recursion

`src/graph/xml.cc:309-323` (top-level fuse):

```cpp
ncclResult_t ncclTopoFuseXml(struct ncclXml* dst, struct ncclXml* src) {
  struct ncclXmlNode* topNodeDst;
  NCCLCHECK(xmlFindTag(dst, "system", &topNodeDst));

  if (topNodeDst == NULL) {
    xmlAddTree(dst, NULL, src->nodes);
    return ncclSuccess;
  }

  struct ncclXmlNode* topNodeSrc;
  NCCLCHECK(xmlFindTag(src, "system", &topNodeSrc));

  NCCLCHECK(xmlTopoFuseXmlRecursive(dst, topNodeDst, topNodeSrc));

  return ncclSuccess;
}
```

The recursive worker (`xmlTopoFuseXmlRecursive` at `xml.cc:260-308`, elided for brevity) iterates `topNodeSrc->subs[]` and for each source sub-node calls a dedup-check (match by tag + attributes) against `topNodeDst->subs[]`. On match, recurse into both subtrees; on no-match, call `xmlAddTree` to append the source subtree as-is.

The dedup predicate is the crux of why two equivalent-sized topologies can have different fuse outcomes.

### E.5 Host_hash rewrite

`src/graph/topo.cc:1538-1546`:

```c
  // Fixup the cpu's host_hashes.
  struct ncclXmlNode* node;
  // Update every cpu node's host_hash attribute since those are not
  // intended to be preserved from the XML files that have been read.
  NCCLCHECKGOTO(xmlFindTag(xml, "cpu", &node), ret, fail);
  while (node != nullptr) {
    NCCLCHECKGOTO(xmlSetAttrLong(node, "host_hash", getHostHash()), ret, fail);
    NCCLCHECKGOTO(xmlFindNextTag(xml, "cpu", node, &node), ret, fail);
  }
```

After loading a static XML, NCCL overwrites the `host_hash` attribute on every `<cpu>` node with the current host's hash. Our static XML had no `host_hash` attribute at all — `xmlSetAttrLong` on a missing attribute simply adds it (see `xml.cc:setAttr*` family). So the attribute ends up present after this fixup step, with the correct per-host value, on both static and autogen paths.

This means Hypothesis C's claim "our missing `host_hash` causes merge dedup to differ" must be scrutinized carefully — the attribute is present by merge time on both paths. The difference must be in OTHER attributes (PCI vendor/device/subsystem_*) rather than host_hash specifically.

## Appendix F: Diagnostic plan we can run for AWS

On request, we can execute the following on the HyperPod cluster on short notice. Each item is designed to discriminate between Hypotheses A / B / C, or to surface evidence for an AWS-proposed fourth hypothesis.

### F.1 Dump autogen topology on both pods of a 2-node v0.2.4 run (partial evidence already in hand)

We already have the pod-0 autogen dump from both v0.2.1 and v0.2.4 (both at 126 nodes, structurally identical within excerpts). What we do NOT yet have is the pod-1 dump alongside pod-0 for the same run, which would let us compare the two per-host trees and reason about achievable dedup during fuse.

Run:
```bash
NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.${HOSTNAME}.xml torchrun --nproc_per_node=8 --nnodes=2 ...
```

Deliverable: two XML files (pod 0, pod 1) from the same run showing what NCCL's autogen produced on each host, side-by-side. AWS can diff them to reason about what subtrees the fuse recursion would attempt to dedup and where the arena usage lands.

### F.2 Enable GRAPH subsystem trace on v0.2.2

Run:
```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH torchrun ...
```

Deliverable: detailed graph-subsystem log showing per-node merge decisions (dedup hits/misses, `xmlAddTree` invocations, per-subtree arena usage). If the log prints a running count of `dst->maxIndex`, we see exactly which subtree pushes past 256.

### F.3 Strip our static XML to match autogen structure and re-test

Rebuild v0.2.2-derivative with a static XML that:
- Includes `host_hash="0x..."` on `<cpu>` (placeholder, will be rewritten at load)
- Includes `vendor`, `device`, `subsystem_vendor`, `subsystem_device` on every `<pci>` node
- Includes `<nvlink>` children under every `<gpu>` (4 per GPU, targeting other GPUs' busids)
- Collapses the 4-NIC-per-GPU scheme to match autogen's 1-NIC-per-GPU subtree (12 fewer subtrees)

If this PASSES on cross-node, Hypothesis C (attribute-richness drives dedup) gains empirical support and we can document a "static XML attribute checklist for p5 contributors."
If this FAILS, attribute richness is not the variable and Hypothesis A or B is more likely.

### F.4 Install static XML to only one path

Rebuild v0.2.2-derivative with the `install -D` line that targets `/opt/amazon/ofi-nccl/share/...` removed, leaving only the autotools-installed copy at `/opt/aws-ofi-nccl/share/...`.

If this PASSES, Hypothesis A (double-install causes double-load) gains empirical support as the mechanism.
If this FAILS, A is refuted and one of B/C is the mechanism.

### F.5 NCCL build from source with verbose arena tracing

Rebuild NCCL 2.30.4 from source with a one-line diagnostic added inside `xmlAddTree`:

```c
if (dst->maxIndex % 10 == 0) INFO(NCCL_GRAPH, "arena usage %d/%d after adding <%s>", dst->maxIndex, dst->maxNodes, srcNode->name);
```

Runtime output will show exactly which node type (pci / gpu / nic / net) pushes past which threshold. This is more work (we'd need to rebuild the NCCL pip wheel), but would give near-definitive evidence for the mechanism.

We can execute any subset of F.1-F.5 on your recommendation. F.1, F.2, F.3, and F.4 are fast (~30 minutes each counting pod deploy and logs collection). F.5 is ~3 hours (custom NCCL wheel build + image rebuild + deploy).

## Appendix G: Open questions for upstream/AWS (discussion-ready)

If AWS engineers are time-constrained, the three most valuable questions to answer first are:

1. **In NCCL 2.30.4's `xmlTopoFuseXmlRecursive`, what is the subtree dedup predicate?** Specifically, which attributes on a `<pci>` node must match for two subtrees to be collapsed into one? Do missing attributes count as match or miss? This is one of three candidate hypotheses (Hypothesis C) and we favor it only in the narrow sense that it appears to fit the three-way data best among the three we considered. We have not directly confirmed the mechanism and would value AWS review.

2. **Is there any circumstance under which NCCL 2.30.4 would load both a user-supplied `NCCL_TOPO_FILE` AND run additional autodiscovery into the same arena?** If not, Hypothesis B is refuted and we can focus on A and C. Reading `topo.cc:1525-1546` suggests not, but aws-ofi-nccl may have plugin-side mechanisms (via `env_manager::insert_envvar`) that behave differently from pure user-set env.

3. **Does the aws-ofi-nccl plugin ever register a topology XML at more than one filesystem path?** The v0.2.2 Dockerfile installed our XML at both `/opt/aws-ofi-nccl/share/...` and `/opt/amazon/ofi-nccl/share/...`. If the plugin only reads from one of these at runtime, Hypothesis A is refuted. If both, Hypothesis A gains ground.

Note: the earlier question "does stock autogen produce ~126 nodes on p5.48xlarge" is now empirically answered YES by both our v0.2.1 and v0.2.4 runs (both dumped 126 nodes). That question is no longer on this list.

Answering the three above would let us close out the investigation and post clean documentation for other aws-ofi-nccl contributors who might be tempted to ship a p5.48xlarge static XML via a patch like ours.


## Appendix H: Post-hoc finding — the cap matters even without the patch (2026-05-07)

After closing this PR we ran DeepEP V2's own `tests/elastic/test_ep.py` (16 ranks / 2 pods) on the v0.2.4 image (v1.19.1 plugin, **no** topology patch). That workload also triggers `graph/xml.h:381 NCCL WARN Error : too many XML nodes (max 256)` on all 16 ranks and the run aborts before the first dispatch. Verbatim pod tails are archived at `docs/wave19-evidence/`.

Two single-variable follow-ups were run to localize the factor:

- **Wave 21** — set `NCCL_TOPO_FILE=/opt/topo/p5.48xl-topo.xml` (122 nodes, identical file on all 16 ranks) on v0.2.4 with no patch. Result: same overflow, 72+ warnings. Source inspection of `xmlFindNode` (`xml.h:197`) shows its dedup predicate requires *every* attribute to match, which is consistent with rank-specific generated attrs preventing full collapse during intra-node fuse. Evidence at `docs/wave21-evidence/`.
- **Wave 22** — set `FI_EFA_DEVICE_LIST=<4 rail-distributed EFA devices>` on v0.2.4 with no patch. Result: same overflow. Source inspection shows `NCCL_TOPO_XML_MAX_NODES` is exceeded during NCCL's own `/sys/class/pci_bus/*` walk, not during libfabric enumeration; filtering at the libfabric layer does not shrink the XML that NCCL will fuse. Evidence at `docs/wave22-evidence/`.

Combining Wave 18f (init-only PASS on v0.2.4) with Waves 19/21/22 (DeepEP FAIL on v0.2.4) gives a sharper statement split by scenario:

- In the original init-only experiment scenario documented earlier in this report, the topology patch in this PR was the immediate trigger of the overflow.
- In the later DeepEP test scenario, stock v0.2.4 hit the same underlying 256-node limit without the patch.

The underlying limitation is the `NCCL_TOPO_XML_MAX_NODES = 256` cap in `src/graph/topo.h:192` combined with the non-MNNVL branch of `ncclTopoGetSystem` (`topo.cc:1649-1652`) that reuses that 256-cap buffer for the intra-node fused output.

## Fix that we validated

We built base image `v0.2.5-sm90a` with a single-line change to NCCL 2.30.4:

```
--- src/graph/topo.h:192
-#define NCCL_TOPO_XML_MAX_NODES 256
+#define NCCL_TOPO_XML_MAX_NODES 2048
```

Prior art: ROCm/rccl (AMD production) ships `NCCL_TOPO_XML_MAX_NODES 8192`. Meta's torchcomms ncclx v2.27 used `1024` with comment "NCCLX - Need to run emulation at scale" (reverted to 256 during v2.28 upstream resync). 2048 is a middle value chosen for our tested scenarios.

Cross-node validation on HyperPod p5.48xlarge with this bump:

- **Wave 23** (base image `v0.2.5-sm90a-amd64@sha256:9694e21f...`): DeepEP V2 `test_ep.py` 16 ranks, 2 pods — 144 test-case configs completed cleanly, zero `too many XML nodes` warnings, dispatch/combine throughput in the 2-4 / 17-18 GB/s range. Evidence at `docs/wave23-evidence/`.
- **Wave 24** (child image `vllm-deepep-v2-efa:fast-4eff12845f68`): same test, 101 cases, zero overflow. Evidence at `docs/wave24-evidence/`.
- **Wave 25** (child image `nemo-rl-deepep-v2-efa:allprs-c337956`): same test, 106 cases, zero overflow. Evidence at `docs/wave25-evidence/`.

We make no broader claim that 2048 is safe in every NCCL deployment; we report that it eliminates the overflow across these three images and 351 combined test-case configs on our tested HyperPod p5.48xlarge hardware, with no new failures observed in the validation window.

This does not change the conclusion of the main report: the patch in this PR was still the wrong change to land in the init-only scenario we tested, and this PR should stay closed. It does change the scope of the upstream question: the `NCCL_TOPO_XML_MAX_NODES = 256` cap is reached on 32-NIC p5.48xlarge regardless of whether a static-XML patch is present, and a bump in upstream NCCL would benefit any DeepEP-class workload that reaches the 8-local-rank intra-node fuse path on that host shape.
