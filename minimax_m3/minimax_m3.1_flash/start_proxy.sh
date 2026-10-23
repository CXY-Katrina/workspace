#!/usr/bin/env bash
set -e
export LOCAL_IP=197.13.50.1
source /work/runtime_env.sh
mkdir -p /work/logs
exec /work/.venv/bin/python /work/src/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py --host 0.0.0.0 --port 8009 --prefiller-hosts 197.13.50.1 --prefiller-ports 31050 --decoder-hosts 197.13.51.1 --decoder-ports 31060 > /work/logs/proxy.log 2>&1
