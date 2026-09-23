import os
import time
from pathlib import Path

import sys
sys.path.insert(0, '/workspace/harness-latest')
from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import ExternalDPConfigLoader, RankResolver
from tests.e2e.nightly.multi_node.external_dp.scripts.runtime import ExternalDPServerManager, ExternalDPProxyLauncher, wait_ranks_ready

idx = int(os.environ['LWS_WORKER_INDEX'])
root = Path('/workspace/run06-mrv2-noeplb-latest-20260920')
root.mkdir(parents=True, exist_ok=True)
node = root / f'node-{idx}'
node.mkdir(exist_ok=True)
os.chdir(node)
if not Path('examples').exists():
    Path('examples').symlink_to('/workspace/harness-latest/examples')
cfg = ExternalDPConfigLoader.from_yaml('/workspace/payload_config_v1')
ranks = RankResolver(cfg).resolve()
with ExternalDPServerManager(config=cfg, ranks=ranks, current_node_index=idx, log_root=root/'rank-logs'), ExternalDPProxyLauncher(config=cfg, ranks=ranks, current_node_index=idx, log_root=root/'rank-logs') as proxy:
    if idx == 0:
        wait_ranks_ready(ranks, timeout=14400)
        proxy.wait_ready()
        (root/'services.ready').write_text('ready')
        print('MRV2_NOEPLB_SERVICES_READY', flush=True)
    while True:
        time.sleep(30)
