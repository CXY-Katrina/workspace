set -euo pipefail
while [ ! -f /workspace/run05-noeplb-quarot-20260920/services.ready ]; do sleep 10; done
export BENCHMARK_HOME=/workspace/benchmark
export LOG_PREFIX=/workspace/run05-noeplb-quarot-20260920
export no_proxy=localhost,127.0.0.1,178.27.4.197,178.27.4.198,178.27.4.159
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
mkdir -p /workspace/run05-noeplb-quarot-20260920/client
cd /workspace/run05-noeplb-quarot-20260920/client
python -u - <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, '/workspace/harness-latest')
from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import ExternalDPConfigLoader
from tools.aisbench import run_aisbench_cases
cfg = ExternalDPConfigLoader.from_yaml('/workspace/quarot-debug/noeplb.yaml')
results = run_aisbench_cases(model=cfg.model, port=1999, aisbench_cases=cfg.benchmark_cases, host_ip='178.27.4.197')
Path('/workspace/run05-noeplb-quarot-20260920/benchmark-results.json').write_text(json.dumps(results, default=str, indent=2))
print('AISBENCH_NOEPLB_FULL_COMPLETED', flush=True)
PY
