# Acceptance Status — deepep-v2-efa-base

Last updated: 2026-05-14
Base image: `ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.5-sm90a`
Topology validated: 2× p5.48xlarge (16 H100), AWS EFA cross-node MoE

## Quick reference

Base substrate: CUDA 12.9 + NCCL 2.30.4 (patched) + aws-ofi-nccl 6e504db + DeepEP V2 + PR #612 patches. Image tag v0.2.5-sm90a is the canonical pin for all 6 child overlays.

## Acceptance matrix

This repo is one of 7 in the [DeepEP V2 acceptance set](https://github.com/orgs/antonai-work/repositories?q=deepep-v2-efa). Cross-framework status:

| Framework | Qwen3-30B-A3B | Moonlight-16B (DSv3 family) | DeepSeek-V2-Lite | GLM-4.5-Air-FP8 |
|---|---|---|---|---|
| **base D+C** | ✅ p50=3.0 ms (10-cycle stress) | — | — | — |
| **vLLM** (PR #41183 native) | ✅ 24 tok DP=16 EP=16 | ✅ 8 tok + EOS | ❌ compile hang | ❌ cutlass FP8 broadcast vs DeepEP |
| **SGLang** 0.5.11 native | ✅ /generate FP8 | ❌ pre_permute gap | ❌ same chain | ❌ SiLU vector divisibility |
| **Megatron-LM** Shape Y | ✅ 3-step loss 11.70→11.32 | N/A (training) | N/A | N/A |
| **NeMo-RL** rollout | ✅ rollout 9.45 s [64,8192] | N/A | N/A | N/A |
| **TRT-LLM** 0.21 (shim) | ✅ 4×512 tok, 4.58 GB EFA | ✅ 24 tok | ❌ DSv2 not in registry | ❌ transformers too old for glm4_moe |
| **Dynamo** scheduler | ✅ routing + sim backend | N/A | N/A | N/A |

**Substrate proven model-agnostic.** Remaining failures are all upstream framework gaps (issue tracker below).

## What this repo specifically validates

- ElasticBuffer D+C round-trip kernel
- aws-ofi-nccl + GIN ring sizing fix
- p5.48xlarge topology XML overflow workaround
- DeepEP V2 PR #612 EFA patches

## Open upstream issues being tracked

| Repo | Issue | Status |
|---|---|---|
| NVIDIA/TensorRT-LLM | [DeepseekV2ForCausalLM registry](../deepep-v2-integration/issues/drafts/trt-llm-001-deepseek-v2-arch.md) | Drafted, ready to file |
| vllm-project/vllm | [BATCHED_VLLM_CUTLASS for compressed-tensors FP8 + DeepEP V2](../deepep-v2-integration/issues/drafts/vllm-002-cutlass-fp8-deepep.md) | Drafted |
| sgl-project/sglang | [pre_permute deepep_normal/triton registration](../deepep-v2-integration/issues/drafts/sglang-003-pre-permute-triton.md) | Drafted |
| sgl-project/sglang | [SiLU kVecSize divisibility](../deepep-v2-integration/issues/drafts/sglang-004-silu-vector-size.md) | Drafted |
| deepseek-ai/DeepEP | [#612 EFA auto-QP cap + EFA RDMA fast path](https://github.com/deepseek-ai/DeepEP/pull/612) | Filed (open, awaiting review 8d) |
| vllm-project/vllm | [#41183 native DeepEP V2 integration](https://github.com/vllm-project/vllm/pull/41183) | Filed (open, conflicting since 2026-05-03) |
| NVIDIA/Megatron-LM | [#4632 V2 dispatcher (Shape Y)](https://github.com/NVIDIA/Megatron-LM/pull/4632) | Filed (draft) |
| NVIDIA-NeMo/RL | [#2410](https://github.com/NVIDIA-NeMo/RL/pull/2410), [#2411](https://github.com/NVIDIA-NeMo/RL/pull/2411) | Filed (draft) |

## Tracking + reproduction

- **Acceptance tracker workspace**: [antonai-work/deepep-v2-integration](https://github.com/antonai-work/deepep-v2-integration) (private)
- **Public report (gist, secret)**: shared internally
- **Per-experiment evidence**: each PASS has `experiments/<framework>_<model>/{README.md, manifest.yaml, pod-logs/, oracle-response.json}`
- **State vector + RCA docs**: `docs/state-vector-2026-05-13/F0[1-4]_*.md` in the workspace repo
- **OKRs**: `issues/OKRS-PER-GAP.md` in the workspace repo

## Performance numbers (from base validation)

- **D+C latency** (2-pod p5.48xlarge): cold 11.3 s, warm 155 ms, steady-state p50 = 3.0 ms (10-cycle stress)
- **Cross-node EFA TX** (TRT-LLM Run #5): 4.58 GB in 29 s with ep=16 spanning 2 pods
- **Megatron training**: 0.816 GB EFA TX delta confirmed during 3-step train

EFA `hw_counters` are NOT exposed under `/sys` on p5.48xlarge HyperPod nodes; cross-node traffic evidence comes from libfabric internal counters + structural facts (ep group spans both pods).
