#Prefill
unset ftp_proxy
unset https_proxy
unset http_proxy

export NETWORK_CARD_NAME=enp48s3u1u2
export IP_ADDRESS=141.61.81.142

export VLLM_PP_LAYER_PARTITION="30,30"
export VLLM_SERVER_DEV_MODE=1
export HCCL_BUFFSIZE=2048
export HCCL_IF_IP=$IP_ADDRESS
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_SOCKET_IFNAME=$NETWORK_CARD_NAME
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export GLOO_SOCKET_IFNAME=$NETWORK_CARD_NAME
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve /workspace/MiniMax-M3-w8a8-0626 \
  --host 0.0.0.0 \
  --port $2 \
  --enable-expert-parallel \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size $7 \
  --seed 1024 \
  --served-model-name minimax-m3 \
  --reasoning-parser minimax_m3 \
  --distributed-executor-backend mp \
  --max-model-len 67560 \
  --max-num-batched-tokens 32768 \
  --trust-remote-code \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.95 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"model":"/workspace/MiniMax-M3-EAGLE3-GQA", "method":"eagle3", "num_speculative_tokens":3}' \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/workspace/hjb/profile/prefill_pp_profiling_128k_tp4_pp2_dp2_30_30", "torch_profiler_with_stack": false}' \
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
   > /workspace/hjb/m3_support/logs/prefill_log_w8a8_64k_tp4_pp2_dp2_fc_30_30.log 2>&1 &