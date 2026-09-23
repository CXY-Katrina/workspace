docker exec -i qwen235-acc-20260919 python - <<'PY'
import json,pathlib,yaml
p=pathlib.Path('/workspace')
files=sorted((p/'run03-quarot-20260920').glob('weights-*.json'))
m=[json.loads(f.read_text()) for f in files]
assert len(m)==3
assert m[0]==m[1]==m[2], 'WEIGHT MISMATCH'
a=yaml.safe_load((p/'QWEN3_235B_PD.local.yaml').read_text())
b=yaml.safe_load((p/'QWEN3_235B_PD.local.yaml').read_text())
b['model']='/mnt/weight/Qwen3-235B-A22B-w8a8-QuaRot'
b['benchmarks']['acc']['model_path']=b['model']
(p/'QWEN3_235B_PD.quarot.yaml').write_text(yaml.safe_dump(b,sort_keys=False))
print('ALL THREE WEIGHT MANIFESTS IDENTICAL',len(m[0]))
print('CONFIG CHANGES',a['model'],'->',b['model'],a['benchmarks']['acc']['model_path'],'->',b['benchmarks']['acc']['model_path'])
print(json.dumps(b['benchmarks'],indent=2))
PY
