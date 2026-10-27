unset ftp_proxy
unset https_proxy
unset http_proxy
export NETWORK_CARD_NAME=enp194s0f0
export IP_ADDRESS=80.5.9.138

export VLLM_SERVER_DEV_MODE=1
export HCCL_BUFFSIZE=1024
export HCCL_IF_IP=$IP_ADDRESS
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_SOCKET_IFNAME=$NETWORK_CARD_NAME
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export GLOO_SOCKET_IFNAME=$NETWORK_CARD_NAME
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

export VLLM_ASCEND_LLMDD_RPC_PORT=6657
export VLLM_PP_LAYER_PARTITION="30,30"
export VLLM_DISABLE_COMPILE_CACHE=0

env | grep GLOO
env | grep SOCKET
echo $HCCL_IF_IP
vllm serve /mnt/y00976694/MiniMax-M3-w8a8-from-bf16-convert \
  --host 0.0.0.0 \
  --port 30050 \
  --enforce-eager \
  --enable-expert-parallel \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --api-server-count 1 \
  --data-parallel-address 80.5.9.138 \
  --data-parallel-rpc-port 6884  \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4 \
  --seed 1024 \
  --served-model-name minimax-m3 \
  --max-model-len 67560 \
  --max-num-batched-tokens 32768 \
  --long-prefill-token-threshold 2048 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --reasoning-parser minimax_m3 \
  --speculative-config '{"model":"/mnt/y00976694/MiniMax-M3-EAGLE3-GQA/", "method":"eagle3", "num_speculative_tokens":3}' \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/workspace/profill", "torch_profiler_with_stack": false}' \
  --enforce-eager \
  --additional-config '{
    "enable_cpu_binding": true,
    "ascend_compilation_config": {
      "enable_static_kernel": true,
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
  "kv_role": "kv_producer",
  "kv_port": "31000",
  "engine_id": "0",
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
   > ./out.log 2>&1 &