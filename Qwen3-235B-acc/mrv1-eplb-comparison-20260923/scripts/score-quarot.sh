docker exec -i qwen235-acc-20260919 python - <<'PY'
import json,pathlib
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
from ais_bench.benchmark.datasets.gsm8k import gsm8k_postprocess,gsm8k_dataset_postprocess,Gsm8kEvaluator
p=pathlib.Path('/workspace/run03-quarot-20260920'); f=next((p/'client/outputs').rglob('*.jsonl'))
r=[]
for l in f.read_text().splitlines():
 try:r.append(json.loads(l))
 except json.JSONDecodeError: pass
e=Gsm8kEvaluator(); c=sum(e.is_equal(gsm8k_postprocess(extract_non_reasoning_content(x['prediction'])),gsm8k_dataset_postprocess(x['gold'])) for x in r)
s={'saved':len(r),'correct':c,'wrong':len(r)-c,'partial_accuracy':100*c/len(r)}
(p/'interim-score.json').write_text(json.dumps(s,indent=2)); print(s)
PY
