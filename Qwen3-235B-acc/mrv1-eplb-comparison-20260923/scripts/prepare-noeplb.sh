docker exec -i qwen235-acc-20260919 python - <<'PY'
from pathlib import Path
p=Path('/workspace/QWEN3_235B_PD.quarot.yaml');text=p.read_text()
assert text.count('"dynamic_eplb":true')==2
out=Path('/workspace/quarot-debug/noeplb.yaml')
out.write_text(text.replace('"dynamic_eplb":true','"dynamic_eplb":false'))
print('Changed only dynamic_eplb on 2 decode nodes')
PY
