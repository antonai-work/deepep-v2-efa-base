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

TOTAL=$((PASS+FAIL))
if [ "$FAIL" -eq 0 ]; then
    echo "${PASS}/${TOTAL} checks PASS"
    exit 0
else
    echo "${PASS}/${TOTAL} checks PASS (see above for failures)"
    exit 1
fi
