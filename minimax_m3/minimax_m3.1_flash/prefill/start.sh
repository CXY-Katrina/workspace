#!/usr/bin/env bash
set -e
cd /work/prefill
export LOCAL_IP=197.13.50.1
export DP_SIZE=2 TP_SIZE=4 PP_SIZE=1 DP_RANK=0
export DP_ADDRESS=197.13.50.1 DP_RPC_PORT=6884 SERVICE_PORT=31050
exec bash /work/prefill/run_dp_template.sh
