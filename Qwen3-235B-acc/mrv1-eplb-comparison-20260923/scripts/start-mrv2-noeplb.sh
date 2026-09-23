set -e
case "$(hostname)" in host127) idx=0;; ipb21b04c6*) idx=1;; ipb21b049f*) idx=2;; *) exit 2;; esac
docker stop -t 20 qwen235-acc-20260919 >/dev/null || true
docker start qwen235-acc-20260919 >/dev/null
rm -rf /mnt/share/qwen235-acc-20260919/run06-mrv2-noeplb-latest-20260920
mkdir -p /mnt/share/qwen235-acc-20260919/run06-mrv2-noeplb-latest-20260920
docker exec -d -e LWS_WORKER_INDEX=$idx -e no_proxy=localhost,127.0.0.1,178.27.4.197,178.27.4.198,178.27.4.159 qwen235-acc-20260919 bash -lc 'unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; python -u /workspace/quarot_mrv2_noeplb_services.py > /workspace/run06-mrv2-noeplb-latest-20260920/node-'"$idx"'.log 2>&1'
