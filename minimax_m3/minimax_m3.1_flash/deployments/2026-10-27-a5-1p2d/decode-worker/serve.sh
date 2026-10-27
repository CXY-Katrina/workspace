#!/usr/bin/env bash
set -e

# Configure these values for the current host.
MODEL_PATH="/data/l00845740/minimax_m3dot1_flash_w8a8_mxfp8_quarot_l32"
LOCAL_IP="197.13.45.1"
NIC_NAME="enp39s0f0"
SERVICE_PORT=31160
DP_RPC_PORT=6985
KV_PORT=27900
MAX_NUM_SEQS=256
MAX_MODEL_LEN=133000
MAX_BATCHED_TOKENS=32768
GPU_MEMORY_UTILIZATION=0.92

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export HCCL_BUFFSIZE=1024
export HCCL_IF_IP="$LOCAL_IP"
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$SERVICE_PORT" \
    --data-parallel-size 4 \
    --data-parallel-size-local 2 \
    --data-parallel-start-rank 2 \
    --headless \
    --data-parallel-address "197.13.52.1" \
    --data-parallel-rpc-port "$DP_RPC_PORT" \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
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
    --enable-prefix-caching \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/work/decode-profile-20261027/profiling","torch_profiler_with_stack":false,"ignore_frontend":true}' \
    --seed 1024 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":1024}' \
    --speculative-config '{
        "method":"dspark",
        "num_speculative_tokens":7,
        "enforce_eager":true,
        "kv_cache_dtype":"bfloat16",
        "draft_sample_method":"probabilistic"
    }' \
    --additional-config '{
        "enable_cpu_binding": true,
        "ascend_compilation_config": {
            "fuse_norm_quant": false,
            "fuse_qknorm_rope": true
        },
        "multistream_overlap_shared_expert": true,
        "enable_shared_expert_dp": true,
        "enable_flashcomm1": true
    }' \
    --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_consumer",
        "kv_port": "'"$KV_PORT"'",
        "kv_connector_extra_config": {
            "use_ascend_direct": true,
            "prefill": {
                "dp_size": 1,
                "tp_size": 4,
                "pp_size": 2
            },
            "decode": {
                "dp_size": 4,
                "tp_size": 4,
                "pp_size": 1
            }
        }
    }'
