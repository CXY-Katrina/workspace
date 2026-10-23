#!/usr/bin/env bash
set -e
cd /work/decode
export LOCAL_IP=197.13.51.1
export DP_SIZE=2 TP_SIZE=4 PP_SIZE=1 DP_RANK=0
export DP_ADDRESS=197.13.51.1 DP_RPC_PORT=6885 SERVICE_PORT=31060
exec bash /work/decode/run_dp_template.sh
