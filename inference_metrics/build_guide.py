from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parent
C = {'ink':'#1d3045','muted':'#63768a','line':'#d9e2eb','teal':'#087f8c','purple':'#7352bd','amber':'#b36a0c','blue':'#2865b5','queue':'#dce4ec'}

def txt(x,y,s,size=17,fill='ink',anchor='start',weight=400):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{C.get(fill,fill)}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(s))}</text>'
def line(x1,y1,x2,y2,color='line',width=2,dash=''):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C.get(color,color)}" stroke-width="{width}" stroke-dasharray="{dash}"/>'
def rect(x,y,w,h,color,rx=8,opacity=1):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{C.get(color,color)}" opacity="{opacity}"/>'
def bracket(x1,x2,y,label,color,ident=''):
    return f'<g class="measure" data-metric="{ident}">'+line(x1,y,x2,y,color,3)+line(x1,y-7,x1,y+7,color,3)+line(x2,y-7,x2,y+7,color,3)+txt((x1+x2)/2,y-12,label,18,color,'middle',600)+'</g>'
def svg(content,h,title):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {h}" role="img" aria-label="{title}" style="font-family:Segoe UI,Microsoft YaHei,sans-serif"><title>{title}</title><rect width="1200" height="{h}" fill="#ffffff"/>{content}</svg>'

s=txt(36,40,'一条请求：先等首 token，再看后续输出节奏',25,weight=650)
s+=txt(36,70,'示例：客户端时间戳，单位 ms；N = 5。每次流式输出恰好含 1 个 token。',16,'muted')
s+=txt(36,133,'流程',17,'muted')
for x,w,c,a,b in [(150,100,'queue','接入 / 排队','0–100'),(250,300,'teal','Prefill：读输入，建立 KV','100–400'),(550,100,'queue','首包返回','400–500'),(650,300,'purple','后续 Decode / 调度 / 返回','首 token 之后')]:
    s+=rect(x,94,w,68,c)+txt(x+w/2,122,a,14,'#ffffff' if c in ('teal','purple') else 'ink','middle',600)+txt(x+w/2,145,b,13,'#ffffff' if c in ('teal','purple') else 'muted','middle')
s+=txt(976,125,'收尾',15,'muted')+txt(976,149,'20 ms',14,'muted')
s+=line(150,202,1060,202,'ink')
for x,name,time,y in [(150,'t₀','0',237),(650,'t₁','500',237),(700,'t₂','550',280),(800,'t₃','650',237),(850,'t₄','700',280),(950,'t₅','800',237),(970,'t_end','820',280)]:
    s+=line(x,164,x,y-20,'line',1,'4 4')+f'<circle cx="{x}" cy="202" r="6" fill="{C["ink"]}"/>'+txt(x,y,name,16,'ink','middle',600)+txt(x,y+20,time,15,'muted','middle')
s+=txt(1080,207,'时间 →',16,'muted')
s+=bracket(150,650,326,'TTFT = 500 ms','teal','ttft')
s+=bracket(650,700,326,'50','purple','itl')+bracket(700,800,326,'100','purple','itl')+bracket(800,850,326,'50','purple','itl')+bracket(850,950,326,'100','purple','itl')+txt(976,331,'ITL：逐次间隔',16,'purple')
s+=bracket(650,950,380,'TPOT = 300 ÷ 4 = 75 ms/token','purple','tpot')
s+=bracket(150,970,430,'E2E = 820 ms = TTFT 500 + 后续输出 300 + 收尾 20','blue','e2e')
s+=txt(36,478,'Prefill 通常产生首 token 的 logits，再采样首 token；常规 Decode 利用 KV cache 逐步生成后续 token。',17)
s+=txt(36,505,'上方阶段按客户端观测简化展开；真实服务器的 Decode 可与首包传输重叠。间隔含等待、调度和通信，非纯 GPU 时间。',15,'muted')
single=svg(s,530,'TTFT、ITL、TPOT 与 E2E 的测量范围')

rows=[('A',0,.5,2,[2,3,4],4),('B',1,1.5,3,[3,4,5,6],6),('C',2,2.5,5,[5,6,7,8],8),('D',6,6.5,7,[7,8,9],9)]
x=lambda t:150+90*t
s=txt(36,40,'多条请求：吞吐看整段窗口，并发看某一时刻',25,weight=650)
s+=txt(36,70,'演示窗口 W = [0, 10) 秒；4 个成功请求，14 个输出 token；无失败、无跨窗口请求。',16,'muted')
s+=bracket(x(0),x(10),111,'W = 10 s：QPS 与系统 TPS 共用的墙钟窗口','blue')
for t in range(11):
    s+=line(x(t),140,x(t),407,'line',1,'4 5')+txt(x(t),435,f'{t}s',15,'muted','middle')
for j,(name,start,pref,first,tokens,end) in enumerate(rows):
    y=155+j*64
    s+=txt(50,y+23,'请求 '+name,18,weight=600)
    s+=rect(x(start),y,x(pref)-x(start),32,'queue',4)+rect(x(pref),y,x(first)-x(pref),32,'teal',4)+rect(x(first),y,x(end)-x(first),32,'purple',4)
    for t in tokens:
        s+=f'<circle cx="{x(t)}" cy="{y+16}" r="6" fill="#ffffff" stroke="{C["purple"]}" stroke-width="2"/>'
    s+=txt(x(end)+14,y+22,f'✓ {len(tokens)} tokens',15,'muted')
s+='<g id="cursor" transform="translate(465 0)">'+line(0,133,0,408,'amber',3,'6 4')+txt(10,148,'t = 3.5 s',16,'amber',weight=650)+'</g>'
s+=txt(36,477,'● 白点 = 输出 token（含首 token）    ✓ = 请求完成    灰 = 接入/排队    青 = Prefill 至首 token    紫 = 后续输出',16,'muted')
s+=txt(36,512,'在 t = 3.5 s：A、B 正在输出，C 仍在 Prefill；在途并发 C(t) = 3，但 Decode 中的请求只有 2 条。',17,weight=600)
window=svg(s,540,'QPS、TPS 与并发的统计范围')
for name,data in [('single-request.svg',single),('system-window.svg',window)]:
    (ROOT/name).write_text(data,encoding='utf-8')

html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>推理性能指标｜从 Prefill 到 Decode</title>
<style>
:root{--ink:#1d3045;--muted:#63768a;--teal:#087f8c;--purple:#7352bd;--line:#d9e2eb;--paper:#fff;--bg:#edf2f6}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.75 'Segoe UI','Microsoft YaHei',sans-serif}main{max-width:1280px;margin:auto;padding:36px 32px 64px}header{padding:12px 0 24px}.eyebrow{font-size:13px;letter-spacing:2px;color:var(--teal);font-weight:700}h1{font-size:38px;line-height:1.3;margin:10px 0 14px;letter-spacing:-1px}h2{font-size:25px;margin:0 0 12px}h3{font-size:18px;margin:0 0 8px}p{margin:8px 0 14px}a{color:var(--teal);text-underline-offset:4px}.lead{max-width:990px;color:var(--muted);font-size:18px}nav{display:flex;gap:10px;flex-wrap:wrap;margin-top:22px}nav a,.pill{border:1px solid var(--line);background:var(--paper);border-radius:50px;padding:7px 15px;text-decoration:none;font-size:14px}.panel{background:var(--paper);border:1px solid var(--line);border-radius:18px;padding:26px;margin:22px 0;box-shadow:0 5px 18px #182e4605}.panel svg{display:block;width:100%;height:auto}.section-head{display:flex;align-items:center;justify-content:space-between;gap:15px}.n{color:var(--teal);font-size:14px;font-weight:700;margin-bottom:4px}.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:16px 0}.box{border-radius:12px;background:#f4f7fa;padding:18px}.box strong.value{display:block;font-size:26px;color:var(--teal);line-height:1.4}.box p{font-size:14px;margin:6px 0 0}.formula{font-family:Consolas,'Microsoft YaHei',monospace;font-size:17px;font-weight:600;color:var(--purple)}.callout{border-left:4px solid var(--teal);padding:10px 18px;background:#eff8f8;margin:18px 0}.small{font-size:14px}.controls{display:flex;align-items:center;flex-wrap:wrap;gap:10px;padding:10px 6px}button{font:inherit;cursor:pointer;border:1px solid var(--line);border-radius:8px;background:white;color:var(--ink);padding:6px 16px}button[aria-pressed=true]{background:var(--teal);color:white;border-color:var(--teal)}button:focus-visible,a:focus-visible,input:focus-visible{outline:3px solid #edb442;outline-offset:3px}.measure{transition:opacity .15s}.dim{opacity:.14}input[type=range]{width:min(400px,80vw);accent-color:var(--teal)}output{font-weight:650;color:var(--teal)}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:15px}th{text-align:left;color:var(--muted);font-size:13px;border-bottom:2px solid var(--line);padding:12px 10px}td{border-bottom:1px solid var(--line);padding:14px 10px;vertical-align:top}td:first-child{font-weight:650;white-space:nowrap}.tag{display:inline-block;font-size:12px;border-radius:4px;padding:1px 6px;background:#eaf4f4;color:var(--teal)}.purple{background:#f1edfa;color:var(--purple)}.split{display:grid;grid-template-columns:1fr 1fr;gap:22px}.derive{padding:16px;background:#f5f2fa;border-radius:12px;font-size:17px;text-align:center}.sources{font-size:14px}.sources li{margin:8px 0}footer{color:var(--muted);font-size:13px}@media(max-width:750px){main{padding:20px 14px}h1{font-size:29px}.panel{padding:16px}.grid,.split{grid-template-columns:1fr}.section-head{align-items:flex-start;flex-direction:column}.panel svg{min-width:700px}.diagram-scroll{overflow-x:auto}.lead{font-size:16px}}@media print{body{background:white}main{max-width:none;padding:0}.panel{break-inside:avoid;box-shadow:none}.controls,nav,button{display:none}a{color:inherit}.grid{grid-template-columns:repeat(3,1fr)}}
</style><main>
<header><div class="eyebrow">LLM INFERENCE · 性能指标图解</div><h1>一次请求有多快？<br>整个系统能承载多少？</h1><p class="lead">TTFT 看多久开始输出；ITL 看每次输出是否卡顿；TPOT 看后续输出的平均节奏。TPS 与 QPS 看整个系统的产出，并发看同时有多少请求尚未结束。</p><nav><a href="#single">① 单请求时间线</a><a href="#system">② 多请求统计窗口</a><a href="#stats">③ 采样、推导与统计</a><a href="#pitfalls">④ 口径与易错点</a><a href="prefill-decode.html">打开推理流程交互图 ↗</a></nav></header>
<section class="panel" id="single"><div class="n">01 / 单请求</div><div class="section-head"><h2>TTFT、ITL 与 TPOT，各自量哪一段？</h2><a class="small" href="single-request.svg" download>下载 SVG 图</a></div><p class="muted">Prefill 处理输入 token，建立 KV cache，并为首 token 提供预测；常规自回归 Decode 利用已有 KV，逐步生成后续 token。<a href="https://arxiv.org/abs/2403.02310">[1]</a></p>
<div class="controls" aria-label="高亮测量范围"><span class="small">点击高亮：</span><button data-select="all" aria-pressed="true">全部</button><button data-select="ttft" aria-pressed="false">TTFT</button><button data-select="itl" aria-pressed="false">ITL</button><button data-select="tpot" aria-pressed="false">TPOT</button><button data-select="e2e" aria-pressed="false">E2E</button></div>
<div class="diagram-scroll" id="single-diagram">{{SINGLE}}</div><p id="metric-note" class="callout" aria-live="polite">先确定测量边界，再看公式：t₀ 是发送时间，t₁…tₙ 是输出 token 到达时间，t_end 是请求完成时间。</p>
<div class="table-wrap"><table><thead><tr><th>指标与单位</th><th>含义 / 测量范围</th><th>公式（时间单位统一）</th><th>本例</th></tr></thead><tbody>
<tr><td>TTFT<br><span class="muted small">Time To First Token · ms</span></td><td>从发请求到收到首个输出 token；体现“等多久开始回答”。包括首 token 前的通信、预处理、排队、Prefill、采样与返回。</td><td class="formula">t₁ − t₀</td><td>500 ms</td></tr>
<tr><td>ITL<br><span class="muted small">Inter-Token Latency · ms</span></td><td>相邻两个输出 token 的到达间隔；一条有 N 个 token 的请求，有 N−1 个间隔。用于观察输出抖动、卡顿。</td><td class="formula">ITLᵢ = tᵢ − tᵢ₋₁<br>i = 2…N</td><td>50、100、50、100 ms</td></tr>
<tr><td>TPOT<br><span class="muted small">Time Per Output Token · ms/token</span></td><td>首 token 到末 token 的总时间，摊到后续 N−1 个 token；每条请求得到一个均值。</td><td class="formula">(tₙ − t₁) / (N−1)</td><td>300 / 4 = 75 ms/token</td></tr>
<tr><td>E2E / 请求延迟<br><span class="muted small">End-to-End Latency · ms</span></td><td>从发送请求到请求完成；本例把末 token 后的流结束收尾单独保留。</td><td class="formula">t_end − t₀</td><td>820 ms</td></tr>
</tbody></table></div>
<p class="small muted">本页采用严格的 token 到达边界。vLLM 压测常按 (请求延迟 − TTFT)/(N−1) 计算 TPOT，ITL 则记录流式输出块间隔；若请求延迟结束于末 token，两种 TPOT 公式相同。若工具把末 token 后的收尾也算入延迟，本例会得到 (820−500)/4 = 80 ms/token，而非 75。<a href="https://docs.vllm.ai/en/latest/benchmarking/cli/#understanding-the-latency-metrics">[2]</a></p>
<div class="grid"><div class="box"><h3>首 token 之后的单请求速度</h3><strong class="value">13.33 token/s</strong><p>(N−1)/(tₙ−t₁) = 1000/75。这个特定口径等于 TPOT 的倒数。</p></div><div class="box"><h3>单请求全程输出速度</h3><strong class="value">6.10 token/s</strong><p>N/E2E = 5/0.82。包含等待首 token 与收尾，不能直接用 1/TPOT 代替。</p></div><div class="box"><h3>纯 Prefill 时长</h3><strong class="value">300 ms</strong><p>示意中的 100→400 ms。客户端只看 TTFT 无法独立拆出这个阶段，需服务端埋点。</p></div></div></section>
<section class="panel" id="system"><div class="n">02 / 多请求</div><div class="section-head"><h2>QPS、TPS 看窗口；并发看重叠</h2><a class="small" href="system-window.svg" download>下载 SVG 图</a></div><p class="muted">这里的 TPS 明确指 Tokens Per Second。系统输出 TPS 包含首 token；QPS 按成功完成请求计算。所有速率的分子、分母必须使用同一统计口径。</p><div class="diagram-scroll" id="system-diagram">{{WINDOW}}</div>
<div class="controls"><label for="at-time">移动时间切线：</label><input id="at-time" type="range" min="0" max="10" value="3.5" step="0.1"><output id="concurrency" for="at-time">t = 3.5 s · 在途 3 · Decode 2</output></div>
<div class="grid"><div class="box"><h3>QPS · Queries Per Second</h3><strong class="value">0.4 请求/s</strong><p>窗口内成功完成 4 个请求 ÷ 10 s。这里也叫请求吞吐 / RPS；一个请求输出许多 token，仍只计 1 个请求。</p></div><div class="box"><h3>系统输出 TPS</h3><strong class="value">1.4 token/s</strong><p>窗口内输出 14 个 token ÷ 10 s。分母是整段墙钟时间，不是各请求 Decode 时长之和。</p></div><div class="box"><h3>平均在途并发 C̄</h3><strong class="value">1.8 个请求</strong><p>并发曲线面积 ÷ 10 s = (4+5+6+3)/10；瞬时并发是整数，时间平均并发可以是小数。</p></div></div>
<div class="split"><div><h3>并发 C(t) 到底数什么？</h3><p>本页数“已发送、尚未完成”的请求，包括接入、排队、Prefill、Decode、返回与收尾。图中每条横条是一条请求的在途生命周期；竖线穿过多少条，就是该时刻的在途并发。</p><p>服务端 waiting、running、Decode 中请求数，以及单次迭代 batch size，是不同范围的计数。<b>客户端并发不等于 GPU batch size。</b><a href="https://docs.vllm.ai/en/stable/design/metrics/">[3]</a></p></div><div><h3>设置值与测量值要分开</h3><p><b>并发上限 32</b> 是压测参数；<b>平均实际并发 29.6</b> 才是测量结果。固定并发的闭环压测通常完成一个就补发一个。设定请求到达率的开环压测则按计划发请求；拥塞时到达率可能高于完成 QPS。<a href="https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/">[4]</a></p><p>增加并发可能提高系统吞吐；饱和后可能主要增加排队与延迟，不能假设吞吐线性增长。</p></div></div>
<div class="derive">同一请求集合、同一窗口：输出 TPS = 完成 QPS × 平均输出 token 数<br><span class="small">本例：0.4 × (14/4) = 1.4 token/s</span></div>
<p class="small muted">上式为本页计数定义的代数推导。真实滑动窗口中，输出 token 可能来自尚未完成或窗口前发起的请求，必须处理跨窗口归属；不能把“本窗口产生的 token”直接与“本窗口完成请求的平均长度”混算。</p>
<div class="derive">稳定系统、边界一致：C̄ ≈ 完成 QPS × 平均 E2E（秒）<br><span class="small">Little 定律；本例所有请求完整落入窗口，1.8 = 0.4 × 4.5</span></div><p class="small muted">这是平均量之间的关系，不能代入峰值并发或 P99 延迟。一般情形需稳定流量、完整核算进出与驻留时间；有失败/取消时应统一完成率及延迟样本。本例有限窗口等式直接来自 18 请求·秒 ÷ 10 秒的面积计算。</p></section>
<section class="panel" id="stats"><div class="n">03 / 指标性质</div><h2>“统计指标”与“推导指标”不是互斥分类</h2><p>严格地说，耗时也是两个时间戳相减的结果。工程上更实用的三层划分是：<b>基础观测 → 单次测量 / 派生计算 → 跨样本统计</b>。因此一个指标可以既是推导值，又被进一步统计为 Mean、P50、P95、P99。</p>
<div class="grid"><div class="box"><h3>① 基础观测</h3><p>发送、首 token、后续 token、结束的时间戳；输入/输出 token 数；成功请求计数；瞬时在途数。一般来自客户端采样或服务端埋点。</p></div><div class="box"><h3>② 单次测量与派生</h3><p>时间戳差 → TTFT / ITL / E2E；时长 ÷ token 数 → TPOT；计数 ÷ 窗口 → TPS / QPS；并发面积 ÷ 窗口 → C̄。</p></div><div class="box"><h3>③ 聚合统计</h3><p>对 TTFT、逐间隔 ITL、逐请求 TPOT 等样本计算均值、P50、P95、P99。也可统计多个窗口的 TPS；必须说明窗口长度。</p></div></div>
<div class="table-wrap"><table><thead><tr><th>指标</th><th>从哪里来</th><th>常见统计单位 / 聚合方式</th><th>分类</th></tr></thead><tbody>
<tr><td>TTFT、E2E、排队时长</td><td>对同一边界的起止事件计时</td><td>每请求一个样本 → Mean / P50 / P99</td><td><span class="tag">直接计时</span> → 聚合统计</td></tr>
<tr><td>ITL</td><td>相邻 token / 输出块事件的时间差</td><td>每间隔一个样本 → 间隔分布；长输出贡献更多样本</td><td><span class="tag">直接计时</span> → 聚合统计</td></tr>
<tr><td>TPOT</td><td>首后总耗时 ÷ 后续输出 token 数</td><td>每请求先求均值 → 再跨请求统计</td><td><span class="tag purple">派生的请求内均值</span> → 聚合统计</td></tr>
<tr><td>TPS、QPS</td><td>token / 请求计数 ÷ 墙钟窗口</td><td>每窗口一个速率；可对多窗口进一步统计</td><td><span class="tag purple">派生速率</span>，也是窗口统计量</td></tr>
<tr><td>瞬时并发 C(t)</td><td>某时刻尚未结束的请求数</td><td>可统计最大值或时间平均值</td><td><span class="tag">直接计数 / Gauge</span></td></tr>
<tr><td>平均并发 C̄</td><td>∫ C(t) dt / W</td><td>按持续时间加权；非不规则采样点的简单平均</td><td><span class="tag purple">派生的时间统计量</span></td></tr>
<tr><td>并发上限、目标 QPS</td><td>人为设置的负载参数</td><td>须与实际在途数、实际完成率分列</td><td>配置参数，不是实测性能结果</td></tr>
</tbody></table></div>
<div class="callout"><b>Mean ITL 与 Mean TPOT 为什么可能不同？</b><br>A 请求只有 1 个间隔，耗时 100 ms；B 请求有 9 个间隔，每个 10 ms。按请求平均 TPOT = (100+10)/2 = <b>55 ms/token</b>；把所有间隔放一起平均 ITL = (100+9×10)/10 = <b>19 ms</b>。一个按请求等权，另一个按间隔等权。</div>
<p>在每块恰好一个 token、边界一致时，<span class="formula">TPOTᵣ = mean(ITLᵣ)</span> 只保证<b>同一个请求内部</b>成立。跨请求时，全体 ITL 均值是按 Nᵣ−1 加权的 TPOT 均值；P99 TPOT 是“最慢尾部请求的平均节奏”，P99 ITL 是“尾部输出间隔”，也不能互换。</p><p class="small muted">P99 表示约 99% 的所选样本不超过该值；分位数插值实现可能不同。N=1 时没有首后间隔，TPOT / ITL 应记为不适用并说明剔除方式，而不是强行写成 0。</p></section>
<section class="panel" id="pitfalls"><div class="n">04 / 读报告前统一口径</div><h2>这些区别会改变你的结论</h2><div class="split"><div><h3>流式输出块不一定是一个 token</h3><p>网络缓冲、合并返回、投机解码可能一次给出多个 token。客户端能直接观测的是块到达间隔，无法恢复块内每个 token 的真实生成时间。vLLM benchmark 的 ITL 按块间隔记，GenAI-Perf 的 ITL 可将块间隔除以后一个块的 token 数；名称一样不代表公式一样。<a href="https://docs.vllm.ai/en/latest/benchmarking/cli/#understanding-the-latency-metrics">[2]</a> <a href="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/genai-perf-README.html#metrics">[5]</a></p><h3>客户端延迟不等于服务器计算时间</h3><p>服务端 TTFT 可能从前端收到请求或开始 tokenization 计时；客户端从发送计时。客户端 ITL 也不等于单次 Decode kernel 时长。计时尽量使用同一侧的单调时钟，跨机器时间戳不能直接相减。<a href="https://docs.vllm.ai/en/stable/design/metrics/">[3]</a></p><h3>Prefill / Decode 分离时看事件归属</h3><p>KV 传输、Decode 侧排队若发生在首 token 可见前，会进入 TTFT；若首 token 已返回后再等待移交，会进入后续 ITL / TPOT，尤其首后间隔。阶段归属依实际返回路径判断。</p></div><div><h3>TPS 必须说明是哪个 TPS</h3><p><b>输出 TPS</b> = 输出 token 数/W；<b>输入 TPS</b> = 输入 token 数/W；<b>总 token TPS</b> = (输入+输出)/W。它们也不同于 Prefill 纯计算吞吐或单用户 token/s。缓存命中的输入 token 是否仍计入逻辑输入量，也要注明。</p><h3>完成吞吐不等于发压速率</h3><p>完成 QPS、到达 QPS、目标发压 QPS 分别对应三个量。建议同时列成功数、错误率和超时率；Goodput 可定义为“成功且满足 TTFT / TPOT / E2E 等 SLO 的请求数/W”。计数只纳入成功请求时应明确标注。</p><h3>比较结果要固定工作负载</h3><p>至少说明：模型与精度、硬件与副本数、输入/输出长度分布、并发或到达率、缓存命中、流式方式、工具版本、预热和计时窗口。输出变长，QPS 可能降低，输出 TPS 却不一定降低。</p></div></div>
<div class="callout">最小报告示例：<b>成功完成 QPS + 系统输出 TPS + 实际平均/峰值并发 + TTFT P50/P99 + TPOT P50/P99 + ITL P99 + 平均输出长度 + 错误率</b>，并注明客户端/服务端和 token/输出块口径。</div></section>
<section class="panel sources"><h2>来源与定义边界</h2><p>按 2026-09-12 可访问的一手资料核对。全部数值为教学构造，不代表任何模型或硬件的实际性能。</p><ol><li><a href="https://arxiv.org/abs/2403.02310">Sarathi-Serve 论文</a>：Prefill / Decode 阶段语义。</li><li><a href="https://docs.vllm.ai/en/latest/benchmarking/cli/#understanding-the-latency-metrics">vLLM Benchmark CLI：Understanding the Latency Metrics</a>：客户端 TTFT、ITL、TPOT 定义，逐块与逐 token 差异。</li><li><a href="https://docs.vllm.ai/en/stable/design/metrics/">vLLM Metrics Design</a>：服务端埋点边界、队列和 running / waiting 计数。</li><li><a href="https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/">NVIDIA：LLM Inference Benchmarking — Fundamental Concepts</a>：系统/单用户吞吐、并发、batch size、请求率。</li><li><a href="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/genai-perf-README.html#metrics">NVIDIA GenAI-Perf Metrics</a>：逐块归一化 ITL、token / request throughput。</li></ol><p>三层分类、示例计算和窗口一致性的推导是本说明的分析；没有把它们当成某个框架的统一命名标准。</p></section>
<footer>本页可离线浏览。两张统计范围图可下载为 SVG；流程图提供独立交互浏览。打印本页可保留完整图文。</footer></main>
<script>
const notes={all:'先确定测量边界，再看公式：t₀ 是发送时间，t₁…tₙ 是输出 token 到达时间，t_end 是请求完成时间。',ttft:'TTFT = t₁ − t₀ = 500 ms。从发请求到第一个输出 token 到达；排队和通信也在范围里，不能当成纯 Prefill 时间。',itl:'ITL 分别为 50、100、50、100 ms。保留每个间隔，才看得到卡顿和抖动；本例一块一个 token。',tpot:'TPOT = (800−500)/(5−1) = 75 ms/token。5 个输出 token 只有 4 个首后间隔；首 token 的耗时已计入 TTFT。',e2e:'E2E = 820−0 = 820 ms。本例末 token 后还有 20 ms 收尾；若压测工具在末 token 即结束计时，它会报告 800 ms。'};
document.querySelectorAll('[data-select]').forEach(b=>b.addEventListener('click',()=>{const k=b.dataset.select;document.querySelectorAll('[data-select]').forEach(v=>v.setAttribute('aria-pressed',String(v===b)));document.querySelectorAll('#single-diagram .measure').forEach(g=>g.classList.toggle('dim',k!=='all'&&g.dataset.metric!==k));document.getElementById('metric-note').textContent=notes[k]}));
const rows=[{s:0,f:2,e:4},{s:1,f:3,e:6},{s:2,f:5,e:8},{s:6,f:7,e:9}];document.getElementById('at-time').addEventListener('input',e=>{const t=+e.target.value,active=rows.filter(r=>r.s<=t&&t<r.e).length,dec=rows.filter(r=>r.f<=t&&t<r.e).length;document.getElementById('cursor').setAttribute('transform',`translate(${150+90*t} 0)`);document.querySelector('#cursor text').textContent=`t = ${t.toFixed(1)} s`;document.getElementById('concurrency').textContent=`t = ${t.toFixed(1)} s · 在途 ${active} · Decode ${dec}`});
</script></html>'''
html=html.replace('{{SINGLE}}',single).replace('{{WINDOW}}',window)
html=html.replace('Little 定律；', '<a href="https://ocw.mit.edu/courses/6-02-introduction-to-eecs-ii-digital-communication-systems-fall-2012/resources/lecture-22-sliding-window-analysis-littles-law/">Little 定律（MIT 课程）</a>；')
(ROOT/'index.html').write_text(html,encoding='utf-8')
print('Created index.html and 2 standalone SVG diagrams.')
