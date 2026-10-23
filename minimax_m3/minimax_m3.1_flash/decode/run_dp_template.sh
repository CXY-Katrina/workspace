#!/usr/bin/env bash
set -e
source /work/runtime_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_BUFFSIZE=2048
DP_RANK_ARGS=(--data-parallel-size-local "$DP_SIZE")
mkdir -p /work/logs
exec /work/.venv/bin/vllm serve /home/t00932610/minimax_m3dot1_mxfp8 \
    --host 0.0.0.0 \
    --port "$SERVICE_PORT" \
    --data-parallel-size "$DP_SIZE" \
    "${DP_RANK_ARGS[@]}" \
    --data-parallel-address "$DP_ADDRESS" \
    --data-parallel-rpc-port "$DP_RPC_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --pipeline-parallel-size "$PP_SIZE" \
    --served-model-name minimax-m3 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --max-model-len 133000 \
    --disable-hybrid-kv-cache-manager \
    --enable-expert-parallel \
    --quantization mxfp8 \
    --gpu-memory-utilization 0.92 \
    --distributed-executor-backend mp \
    --kv-cache-dtype fp8 \
    --speculative-config '{"method":"dspark","model":"/work/pr9-8627ba952/draft-prepared","num_speculative_tokens":7,"kv_cache_dtype":"bfloat16","enforce_eager":true}' \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m3 \
    --limit-mm-per-prompt '{"image":1,"video":1}' \
    --safetensors-load-strategy prefetch \
    --seed 1024 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false,"fuse_qknorm_rope":true},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true,"enable_flashcomm1":true}' \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"26900","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":2,"tp_size":4,"pp_size":1},"decode":{"dp_size":2,"tp_size":4,"pp_size":1}}}' > /work/logs/decode-rank${DP_RANK}.log 2>&1
