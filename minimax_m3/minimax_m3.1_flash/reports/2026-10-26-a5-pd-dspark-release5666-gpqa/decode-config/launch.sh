#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HCCL_IF_IP=197.13.51.1
export HCCL_SOCKET_IFNAME=enp39s0f0
export GLOO_SOCKET_IFNAME=enp39s0f0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /workspace
exec bash /workspace/config/serve.sh
