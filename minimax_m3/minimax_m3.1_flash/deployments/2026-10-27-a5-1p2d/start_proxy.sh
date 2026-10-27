#!/usr/bin/env bash
set -eo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
exec python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host 0.0.0.0 \
    --port 8019 \
    --prefiller-hosts 197.13.51.1 \
    --prefiller-ports 31150 \
    --decoder-hosts 197.13.52.1 \
    --decoder-ports 31160 \
    --max-retries 0 \
    --workers 4
