set -e
case "$(hostname)" in host127) idx=0;; ipb21b04c6*) idx=1;; ipb21b049f*) idx=2;; *) exit 1;; esac
docker exec -d qwen235-acc-20260919 bash -c "bash /workspace/services_quarot.sh $idx > /workspace/run03-quarot-20260920/node-$idx.log 2>&1; echo \$? > /workspace/run03-quarot-20260920/node-$idx.exit"
if [ "$idx" = 0 ]; then
 docker exec -d qwen235-acc-20260919 bash -c 'bash /workspace/benchmark_quarot.sh > /workspace/run03-quarot-20260920/client.log 2>&1; echo $? > /workspace/run03-quarot-20260920/client.exit'
fi
echo STARTED_NODE_$idx
