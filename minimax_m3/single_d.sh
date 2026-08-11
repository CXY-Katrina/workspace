unset ftp_proxy
unset https_proxy
unset http_proxy

export NETWORK_CARD_NAME=enp48s3u1u2
export IP_ADDRESS=141.61.81.153


export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=30000
export HCCL_EXEC_TIMEOUT=60
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=$IP_ADDRESS
export GLOO_SOCKET_IFNAME=$NETWORK_CARD_NAME
export TP_SOCKET_IFNAME=$NETWORK_CARD_NAME
export HCCL_SOCKET_IFNAME=$NETWORK_CARD_NAME

export VLLM_USE_V1=1
export HCCL_BUFFSIZE=2048
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH=/home/liziyu/b061/vllm-ascend/examples/disaggregated_prefill_v1/ranktable.json
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_DETERMINISTIC=true
export VLLM_ASCEND_LLMDD_RPC_PORT=6557
export TASK_QUEUE_ENABLE=1
export exportVLLM_LOGGING_LEVEL="info"
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

#   --speculative-config '{"model":"/workspace/MiniMax-M3-EAGLE3", "method":"eagle3", "num_speculative_tokens":3}' \
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_DISABLE_COMPILE_CACHE=0
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export VLLM_SERVER_DEV_MODE=1


vllm serve /workspace/MiniMax-M3-w8a8-0626  \
  --host 0.0.0.0 \
  --port 30060 \
  --enable-expert-parallel \
  --data-parallel-size 4 \
  --data-parallel-size-local 4 \
  --data-parallel-start-rank 0 \
  --api-server-count 1 \
  --data-parallel-address 141.61.81.153 \
  --data-parallel-rpc-port 5964  \
  --tensor-parallel-size 4 \
  --seed 1024 \
  --served-model-name minimax-m3 \
  --reasoning-parser minimax_m3 \
  --distributed-executor-backend mp \
  --max-model-len 67560 \
  --max-num-batched-tokens 32768 \
  --trust-remote-code \
  --max-num_seqs 64 \
  --gpu-memory-utilization 0.95 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"model":"/workspace/MiniMax-M3-EAGLE3-GQA", "method":"eagle3", "num_speculative_tokens":3}' \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/workspace/hjb/profile/decode_profiling_128k_tp4_dp4_2", "torch_profiler_with_stack": false}' \
  --additional-config '{
    "enable_cpu_binding": true,
    "ascend_compilation_config": {
      "enable_static_kernel": false,
      "fuse_norm_quant": false
    },
    "multistream_overlap_shared_expert": true,
    "weight_nz_mode": 2,
    "enable_shared_expert_dp": true,
    "enable_flashcomm1": true,
    "enable_reduce_sample": false
  }' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_consumer",
  "kv_port": "23010",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 4,
                    "pp_size": 2,
                    "pp_layer_partition": "30,30"
             },
             "decode": {
                    "dp_size": 4,
                    "tp_size": 4
             }
      }
  }' \
  > /workspace/hjb/m3_support/logs/decode_log_w8a8_64k_tp4_dp4.log 2>&1 &