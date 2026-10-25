#!/usr/bin/env bash
set -e

# Configure these values for the current host.
MODEL_PATH="/data/l00845740/minimax_m3dot1_flash_w8a8_mxfp8_quarot_l32"
LOCAL_IP="197.13.50.1"
NIC_NAME="enp39s0f0"
SERVICE_PORT=31050
DP_RPC_PORT=6884
KV_PORT=30000
MAX_NUM_SEQS=128
MAX_MODEL_LEN=133000
MAX_BATCHED_TOKENS=32768
GPU_MEMORY_UTILIZATION=0.92

export VLLM_USE_V2_MODEL_RUNNER=1
export HCCL_BUFFSIZE=512
export HCCL_IF_IP="$LOCAL_IP"
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$SERVICE_PORT" \
    --data-parallel-size 1 \
    --data-parallel-size-local 1 \
    --data-parallel-address "$LOCAL_IP" \
    --data-parallel-rpc-port "$DP_RPC_PORT" \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --served-model-name minimax-m3.1-flash \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --disable-hybrid-kv-cache-manager \
    --enable-expert-parallel \
    --quantization ascend \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --distributed-executor-backend mp \
    --kv-cache-dtype fp8 \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m3 \
    --language-model-only \
    --safetensors-load-strategy prefetch \
    --enforce-eager \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --additional-config '{
        "enable_cpu_binding": true,
        "multistream_overlap_shared_expert": true,
        "enable_shared_expert_dp": true,
        "enable_flashcomm1": true
    }' \
    --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_producer",
        "kv_port": "'"$KV_PORT"'",
        "kv_connector_extra_config": {
            "use_ascend_direct": true,
            "prefill": {
                "dp_size": 1,
                "tp_size": 4,
                "pp_size": 2
            },
            "decode": {
                "dp_size": 2,
                "tp_size": 4,
                "pp_size": 1
            }
        }
    }'
