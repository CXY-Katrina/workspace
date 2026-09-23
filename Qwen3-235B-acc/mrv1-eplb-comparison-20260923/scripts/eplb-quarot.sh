python3 - <<'PY'
from pathlib import Path
p=Path('/mnt/share/qwen235-acc-20260919/run03-quarot-20260920')
for f in (p/'rank-logs').rglob('*.log'):
 lines=[x for x in f.read_text(errors='replace').splitlines() if any(k in x.lower() for k in ['eplb','rebalanc','error'])]
 if f.name=='rank-0.log': print(str(f), '\n'.join(lines[-5:])[-1800:])
PY
