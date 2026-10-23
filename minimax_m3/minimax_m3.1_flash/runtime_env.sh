#!/usr/bin/env bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [ -f /usr/local/Ascend/cann-9.2.0/share/info/ascendnpu-ir/bin/set_env.sh ]; then
    source /usr/local/Ascend/cann-9.2.0/share/info/ascendnpu-ir/bin/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
fi
unset ftp_proxy https_proxy http_proxy all_proxy FTP_PROXY HTTPS_PROXY HTTP_PROXY ALL_PROXY
export PATH=/work/.venv/bin:/work/bin:$PATH
export PYTHONPATH=/work/src/vllm:/work/src/vllm-ascend:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/python3.12.13/lib/python3.12/site-packages/mooncake:${LD_LIBRARY_PATH:-}
export VLLM_USE_V2_MODEL_RUNNER=1
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export HCCL_IF_IP=$LOCAL_IP
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=enp39s0f0
export GLOO_SOCKET_IFNAME=enp39s0f0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
