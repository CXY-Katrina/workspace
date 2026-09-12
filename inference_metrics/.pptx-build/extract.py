from pathlib import Path
import xml.etree.ElementTree as ET
import json
root=Path(__file__).resolve().parent.parent
slides=[]
for filename,limit in [('single-request.svg',450),('system-window.svg',490)]:
    items=[]
    def walk(el,dx=0,dy=0):
        tag=el.tag.split('}')[-1]
        attrs=dict(el.attrib)
        if tag=='g' and 'translate' in attrs.get('transform',''):
            vals=attrs['transform'].split('(')[1].split(')')[0].split()
            dx+=float(vals[0]); dy+=float(vals[1])
        if tag in ['text','line','rect','circle']:
            if tag=='rect' and attrs.get('width')=='1200':return
            if tag=='text' and float(attrs.get('y',0))>limit:return
            attrs['dx']=dx;attrs['dy']=dy
            items.append({'tag':tag,'attrs':attrs,'text':''.join(el.itertext())})
        for child in el:walk(child,dx,dy)
    walk(ET.parse(root/filename).getroot())
    slides.append(items)
(root/'.pptx-build/diagrams.json').write_text(json.dumps(slides,ensure_ascii=False),encoding='utf-8')
