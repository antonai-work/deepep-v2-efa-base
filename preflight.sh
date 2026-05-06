#!/bin/bash
# preflight.sh - 5-check validation harness for deepep-v2-efa-base.
#
# Runs inside the image at build-test time (GHA `test-build.yml`) and at
# deploy time (child images can invoke as `bash /preflight.sh`).
#
# All five checks must PASS. Final line of stdout is guaranteed to be one of:
#   "5/5 checks PASS"
#   "N/5 checks PASS (see above for failures)" where N < 5
#
# Exit code: 0 on all-pass, 1 otherwise.

set -u
set -o pipefail

PASS=0
FAIL=0

SCRATCH="${XDG_RUNTIME_DIR:-/var/run}"
mkdir -p "${SCRATCH}" 2>/dev/null || SCRATCH="."
LAST_OUT="${SCRATCH}/preflight-last.out"

check() {
    local name="$1"
    shift
    printf '[check] %-48s ... ' "$name"
    if "$@" >"${LAST_OUT}" 2>&1; then
        echo "PASS"
        PASS=$((PASS+1))
    else
        echo "FAIL"
        echo "--- output ---"
        cat "${LAST_OUT}"
        echo "--- end output ---"
        FAIL=$((FAIL+1))
    fi
}

# ---------------------------------------------------------------------------
# 1. DeepEP V2 primary API (ElasticBuffer) import + repr.
# Expected repr: <class 'deep_ep.buffers.elastic.ElasticBuffer'>
# ---------------------------------------------------------------------------
check_deepep_v2_api() {
    python3 -c "
import deep_ep
r = repr(deep_ep.ElasticBuffer)
assert r == \"<class 'deep_ep.buffers.elastic.ElasticBuffer'>\", 'unexpected repr: ' + r
print(r)
"
}
check "DeepEP V2 ElasticBuffer import" check_deepep_v2_api

# ---------------------------------------------------------------------------
# 2. DeepEP V1 legacy Buffer shim still present on the V2 branch
# (api-shim / child overlays rely on it for V1-compat code paths).
# ---------------------------------------------------------------------------
check_deepep_v1_legacy() {
    python3 -c "
from deep_ep.buffers.legacy import Buffer
print(Buffer)
"
}
check "DeepEP V1 legacy Buffer import" check_deepep_v1_legacy

# ---------------------------------------------------------------------------
# 3. aws-ofi-nccl plugin is discoverable by the dynamic linker.
# ---------------------------------------------------------------------------
check_ldconfig_plugin() {
    ldconfig -p | grep -q libnccl-net-ofi.so
}
check "ldconfig sees libnccl-net-ofi.so" check_ldconfig_plugin

# ---------------------------------------------------------------------------
# 4. aws-ofi-nccl plugin binary is physically installed at the expected path.
# ---------------------------------------------------------------------------
check_plugin_file() {
    test -f /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so
}
check "aws-ofi-nccl plugin at /opt/aws-ofi-nccl" check_plugin_file

# ---------------------------------------------------------------------------
# 5. DeepEP PR #612 patches applied: both the auto-QP cap (num_allocated_qps
# clamp) and the EFA fast path marker ("EFA fast path") appear in
# deep_ep/buffers/elastic.py and deep_ep/utils/envs.py respectively.
# ---------------------------------------------------------------------------
check_patches_applied() {
    grep -l 'num_allocated_qps' /opt/DeepEP/deep_ep/buffers/elastic.py >/dev/null \
        && grep -q 'EFA fast path' /opt/DeepEP/deep_ep/utils/envs.py
}
check "DeepEP PR #612 patches applied" check_patches_applied

# ---------------------------------------------------------------------------
# 6. Wave 12 cu13 runtime unification: torch reports cuda 13.0, DeepEP _C.so
# does not link libcudart.so.12, and NCCL wheel is cu13. This is the key
# invariant Wave 12 exists to enforce - if any of these fail, Wave 11's
# c10::cuda::SetDevice(-112) crash at MoE dispatch will recur.
# ---------------------------------------------------------------------------
check_wave12_cu13_unified() {
    python3 -c "
import torch
cv = torch.version.cuda
assert cv is not None and cv.startswith('13.'), 'torch.version.cuda != 13.x: ' + repr(cv)
print('torch.version.cuda=', cv, 'torch.__version__=', torch.__version__)
" || return 1
    # DeepEP _C.so must not link libcudart.so.12.
    cso="$(find /opt/DeepEP -name '_C*.so' 2>/dev/null | head -1)"
    [ -n "${cso}" ] || { echo "FATAL: no DeepEP _C.so found"; return 1; }
    if ldd "${cso}" 2>/dev/null | grep -qE 'libcudart\.so\.12'; then
        echo "FATAL: ${cso} links libcudart.so.12 - Wave 12 cu13 unification broken"
        return 1
    fi
    # NCCL wheel is cu13.
    python3 -c "
import subprocess
out = subprocess.check_output(['pip', 'show', 'nvidia-nccl-cu13'], stderr=subprocess.DEVNULL).decode()
assert 'Version:' in out, 'nvidia-nccl-cu13 not installed'
for line in out.splitlines():
    if line.startswith('Version:'):
        v = line.split(':', 1)[1].strip()
        parts = tuple(map(int, v.split('.')))
        assert parts >= (2, 30, 4), 'nccl ' + v + ' below 2.30.4 floor'
        print('nvidia-nccl-cu13 version=', v)
" || return 1
    # libcudart.so.13 is the one ldconfig surfaces.
    ldconfig -p | grep -qE 'libcudart\.so\.13' || {
        echo "FATAL: ldconfig does not surface libcudart.so.13"
        return 1
    }
    return 0
}
check "Wave 12 cu13 runtime unified" check_wave12_cu13_unified

TOTAL=$((PASS+FAIL))
if [ "$FAIL" -eq 0 ]; then
    echo "${PASS}/${TOTAL} checks PASS"
    exit 0
else
    echo "${PASS}/${TOTAL} checks PASS (see above for failures)"
    exit 1
fi
