set -euo pipefail
export LWS_WORKER_INDEX=${1:?}
export no_proxy=localhost,127.0.0.1,192.168.13.197,192.168.13.198,192.168.13.159,178.27.4.197,178.27.4.198,178.27.4.159
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
mkdir -p /workspace/run03-quarot-20260920/node-$LWS_WORKER_INDEX
cd /workspace/run03-quarot-20260920/node-$LWS_WORKER_INDEX
test -e examples || ln -s /workspace/harness-latest/examples examples
python -u /workspace/services_quarot.py
