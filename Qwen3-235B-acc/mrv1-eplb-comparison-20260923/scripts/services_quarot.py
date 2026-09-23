import os,sys,time
from pathlib import Path
sys.path.insert(0,'/workspace/harness-latest')
from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import ExternalDPConfigLoader,RankResolver
from tests.e2e.nightly.multi_node.external_dp.scripts.runtime import ExternalDPServerManager,ExternalDPProxyLauncher,wait_ranks_ready
idx=int(os.environ['LWS_WORKER_INDEX'])
cfg=ExternalDPConfigLoader.from_yaml('/workspace/QWEN3_235B_PD.quarot.yaml')
ranks=RankResolver(cfg).resolve()
logs=Path('/workspace/run03-quarot-20260920/rank-logs')
with ExternalDPServerManager(config=cfg,ranks=ranks,current_node_index=idx,log_root=logs), ExternalDPProxyLauncher(config=cfg,ranks=ranks,current_node_index=idx,log_root=logs) as proxy:
 if idx==0:
  wait_ranks_ready(ranks,timeout=14400)
  proxy.wait_ready()
  Path('/workspace/run03-quarot-20260920/services.ready').write_text('ready')
  print('ALL_QUAROT_SERVICES_READY_KEEP_RUNNING',flush=True)
 while True:time.sleep(30)
