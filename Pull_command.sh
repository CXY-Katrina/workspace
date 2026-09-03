#Set up proxy
python load_balance_proxy_server_example.py \
  --port 30088 \
  --host 80.5.9.138 \
  --prefiller-hosts \
    80.5.9.138 \
    80.5.9.138 \
  --prefiller-ports \
    30050 30051 \
  --decoder-hosts \
    80.5.9.136 \
    80.5.9.136 \
    80.5.9.136 \
    80.5.9.136 \
  --decoder-ports \
    30060 30061 30062 30063