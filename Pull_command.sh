#Running on Node P
python launch_online_dp.py \
    --dp-size 2 \
    --tp-size 4 \
    --dp-size-local 2 \
    --dp-rank-start 0 \
    --dp-address 141.61.81.142 \
    --dp-rpc-port 6884 \
    --vllm-start-port 30050 \
    --pp-size 2 \
    --pp-layer-partition "30,30"

#Running on Node D
python launch_online_dp.py \
    --dp-size 4 \
    --tp-size 4 \
    --dp-size-local 4 \
    --dp-rank-start 0 \
    --dp-address 141.61.81.153 \
    --dp-rpc-port 5964 \
    --vllm-start-port 30060


#Set up proxy
python load_balance_proxy_server_example.py \
  --port 12347 \
  --host 141.61.81.142 \
  --prefiller-hosts \
    141.61.81.142 \
    141.61.81.142 \
  --prefiller-ports \
    30050 30051 \
  --decoder-hosts \
    141.61.81.153 \
    141.61.81.153 \
    141.61.81.153 \
    141.61.81.153 \
  --decoder-ports \
    30060 30061 30062 30063