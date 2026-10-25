#!/bin/bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /workspace
exec bash /workspace/nonspec-gpqa2-20261025/config/run_pd_decode.sh
