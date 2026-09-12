import fs from 'node:fs/promises';
import path from 'node:path';
import {Presentation,PresentationFile} from 'file:///C:/Users/Katrina/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const ROOT=path.resolve('inference_metrics');
const diagrams=JSON.parse(await fs.readFile(path.join(ROOT,'.pptx-build/diagrams.json'),'utf8'));
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const font='Microsoft YaHei';
let count=0;
for(let si=0;si<diagrams.length;si++){
  const slide=deck.slides.add();slide.background.fill='#ffffff';
  const ox=40,oy=si===0?95:75,k=1;
  for(const item of diagrams[si]){
    const a=item.attrs,dx=a.dx||0,dy=a.dy||0;
    const n=(key)=>Number(a[key]||0);
    const X=v=>ox+(v+dx)*k,Y=v=>oy+(v+dy)*k;
    const name=`s${si+1}-${item.tag}-${++count}`;
    if(item.tag==='text'){
      const size=n('font-size'),anchor=a['text-anchor']||'start';
      let w=Math.max(90,[...item.text].reduce((sum,ch)=>sum+(/[\u2e80-\uffff]/u.test(ch)?size:size*.60),0)+20);
      let left=X(n('x'))-(anchor==='middle'?w/2:anchor==='end'?w:0);
      left=Math.max(10,left);w=Math.min(w,1270-left);
      const shape=slide.shapes.add({name,geometry:'textbox',position:{left,top:Y(n('y'))-size*1.08,width:w,height:size*1.7},fill:'none',line:{fill:'none',width:0}});
      shape.text=item.text;
      shape.text.style={typeface:font,fontSize:size,bold:n('font-weight')>=600,color:a.fill,alignment:anchor==='middle'?'center':anchor==='end'?'right':'left',autoFit:'none',verticalAlignment:'top',insets:{left:0,right:0,top:0,bottom:0}};
    }else if(item.tag==='rect'){
      slide.shapes.add({name,geometry:'roundRect',position:{left:X(n('x')),top:Y(n('y')),width:n('width'),height:n('height')},borderRadius:n('rx'),fill:a.fill,line:{fill:'none',width:0}});
    }else if(item.tag==='circle'){
      const r=n('r');slide.shapes.add({name,geometry:'ellipse',position:{left:X(n('cx')-r),top:Y(n('cy')-r),width:r*2,height:r*2},fill:a.fill,line:{fill:a.stroke||'none',width:n('stroke-width')}});
    }else if(item.tag==='line'){
      const x1=X(n('x1')),x2=X(n('x2')),y1=Y(n('y1')),y2=Y(n('y2'));
      slide.shapes.add({name,geometry:'line',position:{left:Math.min(x1,x2),top:Math.min(y1,y2),width:Math.max(.01,Math.abs(x2-x1)),height:Math.max(.01,Math.abs(y2-y1))},fill:'none',line:{fill:a.stroke,width:n('stroke-width'),style:a['stroke-dasharray']?'dash':'solid'}});
    }
  }
  slide.speakerNotes.textFrame.setText(si===0?'依据用户提供的图 1 重建。教学示例，非实测数据。客户端 token 到达边界，N=5，一块一个 token。TTFT=500 ms，ITL=50/100/50/100 ms，TPOT=75 ms/token，E2E=820 ms。模型 Decode 可与首包传输重叠，流程按客户端观测简化展开。来源：https://docs.vllm.ai/en/latest/benchmarking/cli/#understanding-the-latency-metrics':'依据用户提供的图 2 重建。教学示例，非实测数据。W=[0,10) 秒，4 个请求，14 个输出 token。QPS=0.4 请求/秒，输出 TPS=1.4 token/秒。t=3.5 秒时在途并发为 3，Decode 请求数为 2。来源：https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/');
}
await(await PresentationFile.exportPptx(deck)).save(path.join(ROOT,'.pptx-build/candidate.pptx'));
for(let i=0;i<2;i++){
  const blob=await deck.export({slide:deck.slides.items[i],format:'png',scale:1.5});
  await fs.writeFile(path.join(ROOT,`.pptx-build/slide-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));
}
console.log('Created two editable slides.');
