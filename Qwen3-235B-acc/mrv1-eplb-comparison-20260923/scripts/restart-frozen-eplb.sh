set -e
case "$(hostname)" in host127) idx=0;; ipb21b04c6*) idx=1;; ipb21b049f*) idx=2;; *) exit 2;; esac
docker stop -t 15 qwen235-acc-20260919
docker start qwen235-acc-20260919
docker exec qwen235-acc-20260919 python -c 'from pathlib import Path; import subprocess; root=Path("/vllm-workspace/vllm-ascend"); paths=["vllm_ascend/eplb/adaptor/vllm_adaptor.py","vllm_ascend/eplb/eplb_updator.py","vllm_ascend/eplb/core/eplb_device_transfer_loader.py"]; [(root/p).write_bytes(subprocess.check_output(["git","show","e139b7d573d3769fd1407d5027d7d4831f5469f0:"+p],cwd=root)) for p in paths]; subprocess.run(["git","diff","--exit-code","--",*paths],cwd=root,check=True); print("All three debug-only files restored to e139")'
mkdir -p /mnt/share/qwen235-acc-20260919/quarot-debug/source-frozen-eplb
docker exec -d -e LWS_WORKER_INDEX=$idx -e DIAG_LABEL=source-frozen-eplb -e DIAG_CONFIG=/workspace/quarot-debug/frozen-eplb.yaml -e no_proxy=localhost,127.0.0.1,178.27.4.197,178.27.4.198,178.27.4.159 qwen235-acc-20260919 bash -c 'unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; python -u /workspace/quarot_debug_services.py > /workspace/quarot-debug/source-frozen-eplb/node-'"$idx"'.log 2>&1'
if [ "$idx" = 0 ]; then
docker exec -d qwen235-acc-20260919 sh -c 'while [ ! -f /workspace/quarot-debug/source-frozen-eplb/services.ready ]; do sleep 5; done; python -u /workspace/quarot_repro.py --label source-frozen-eplb-replay --repeats 3 > /workspace/quarot-debug/source-frozen-eplb-replay.log 2>&1; echo $? > /workspace/quarot-debug/source-frozen-eplb-replay.exit'
fi
