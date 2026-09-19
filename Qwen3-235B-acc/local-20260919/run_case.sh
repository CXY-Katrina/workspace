set -euo pipefail
NODE_INDEX=${1:?node index required}
ATTEMPT=${2:-run01}
export LWS_WORKER_INDEX=$NODE_INDEX
export CONFIG_YAML_PATH=/workspace/QWEN3_235B_PD.local.yaml
export BENCHMARK_HOME=/workspace/benchmark
export EXTERNAL_DP_LOG_DIR=/workspace/$ATTEMPT/rank-logs
export LOG_PREFIX=/workspace/$ATTEMPT
export EXTERNAL_DP_MAX_WAIT_SECONDS=14400
export VLLM_ASCEND_REF=e139b7d573d3769fd1407d5027d7d4831f5469f0
export BENCHMARK_JOB_NAME=QWEN3_235B_PD-local
export no_proxy=localhost,127.0.0.1,192.168.13.197,192.168.13.198,192.168.13.159,178.27.4.197,178.27.4.198,178.27.4.159
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
mkdir -p /workspace/$ATTEMPT/node-$NODE_INDEX
cd /workspace/$ATTEMPT/node-$NODE_INDEX
test -e examples || ln -s /vllm-workspace/vllm-ascend/examples examples
python -u - <<'PY'
import sys
sys.path.insert(0,'/vllm-workspace/vllm-ascend')
from tests.e2e.nightly.multi_node.external_dp.scripts.test_external_dp import test_external_dp
test_external_dp()
PY
