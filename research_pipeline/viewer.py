"""Self-contained HTML replay of one or more runs: a node graph where every node shows its inputs (left ports) and
outputs (right ports); click a node for full payloads. No external dependencies."""
from __future__ import annotations

import json
from dataclasses import asdict


def run_json(r):
    return dict(question=r.question, understanding=r.understanding, reach=r.reach, coverage=r.coverage, answer=r.answer, status=r.status, attempts=r.attempts, depth=r.depth, umbrella=r.umbrella,
                composite=r.score.composite, dims=r.score.dims, followups=r.followups, trace=r.trace)


def write_viewer(results, path):
    data = json.dumps([run_json(r) for r in results]).replace("</", "<\\/")
    open(path, "w").write(TEMPLATE.replace("__DATA__", data))
    return path


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Research Pipeline Replay</title>
<style>
:root{--bg:#090b10;--panel:#0f131b;--card:#141a24;--line:#232c3b;--dim:#7d8aa0;--fg:#e6ecf6;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--flow:#ff9f43;
--c0:#2f8f5b;--c1:#5b6bd6;--c2:#c26a8a;--c3:#3b82c4;--c4:#3b82c4;--c5:#d9822b;--c6:#c26a1f;--c7:#b59a1d;--c8:#8b5cf6;--c9:#a855c7;--c10:#14a3a3;--c11:#b59a1d;--c12:#d4558a;--c13:#2f8f5b}
@media (prefers-color-scheme:light){:root:not([data-theme=dark]){--bg:#eef1f6;--panel:#fff;--card:#fff;--line:#d5dce8;--dim:#5b6a80;--fg:#12192a}}
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--fg);font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;display:grid;grid-template-rows:auto auto 1fr 190px;overflow:hidden}
.top{display:flex;gap:18px;align-items:center;padding:8px 14px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.top b{color:var(--fg)}.top span{color:var(--dim)}.top .q{flex:1;min-width:200px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--fg)}
.ctl{display:flex;gap:8px;align-items:center;padding:6px 14px;border-bottom:1px solid var(--line);flex-wrap:wrap}
button,select{background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 10px;font:inherit;cursor:pointer}
button:hover{border-color:var(--dim)}button.on{border-color:var(--ok);color:var(--ok)}
input[type=range]{accent-color:var(--flow);flex:1;min-width:140px}
.tabs{display:flex;gap:6px;flex-wrap:wrap}.tab{padding:3px 9px;border-radius:5px;border:1px solid var(--line);cursor:pointer;color:var(--dim)}
.tab.sel{color:var(--fg);border-color:var(--fg)}.tab.acc{border-left:3px solid var(--ok)}.tab.rev{border-left:3px solid var(--bad)}
.legend{color:var(--dim);margin-left:auto}.legend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin:0 4px 0 10px;vertical-align:middle}
.main{display:grid;grid-template-columns:auto 1fr 360px;min-height:0}.main.nocfg #cfg{display:none}.main.nocfg{grid-template-columns:1fr 360px}
#cv{overflow:auto;position:relative;background:radial-gradient(var(--line) 1px,transparent 1px) 0 0/22px 22px}
#stage{position:relative;transform-origin:0 0}
svg{position:absolute;left:0;top:0;pointer-events:none;overflow:visible}
.lane{position:absolute;top:0;color:var(--dim);font-size:10px;letter-spacing:.12em;text-transform:uppercase;border-bottom:1px solid var(--line);padding:6px 0;width:236px}
.node{position:absolute;width:236px;background:var(--card);border:1px solid var(--line);border-radius:7px;overflow:visible;transition:opacity .35s,box-shadow .35s,border-color .35s;cursor:pointer}
.node.pending{opacity:.16}.node.pending .rows{visibility:hidden}
.node.active{border-color:var(--flow);box-shadow:0 0 0 1px var(--flow),0 0 22px #ff9f4355}
.node.sel{border-color:var(--fg)}
.node.fail{border-color:var(--bad)}.node.fail .hd{background:#a73838!important}
.hd{padding:4px 8px;font-weight:700;color:#fff;border-radius:6px 6px 0 0;display:flex;justify-content:space-between;gap:6px}
.hd small{font-weight:400;opacity:.85}
.sec{padding:3px 8px 0;color:var(--dim);font-size:9px;letter-spacing:.14em}
.r{display:flex;justify-content:space-between;gap:8px;padding:1px 8px;position:relative}
.r .k{color:var(--dim);flex:none;max-width:42%;overflow:hidden;text-overflow:ellipsis}.r .v{text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.r.in:before,.r.out:after{content:"";position:absolute;top:50%;width:7px;height:7px;margin-top:-3.5px;border-radius:50%;background:var(--dim);border:1px solid var(--bg)}
.r.in:before{left:-4px}.r.out:after{right:-4px;background:var(--c)}
.node.active .r.out:after{background:var(--flow)}
.more{color:var(--dim);padding:0 8px 4px;font-size:10px}
.bar{height:4px;background:var(--line);border-radius:2px;margin:3px 8px 0;position:relative}.bar i{display:block;height:100%;border-radius:2px;background:var(--ok)}
.pad{height:6px}
.w{fill:none;stroke:#8fa1bd;stroke-opacity:.35;stroke-width:1.3}.w.live{stroke:var(--flow);stroke-opacity:1;stroke-width:2;stroke-dasharray:6 5;animation:fl .7s linear infinite}
.w.regen{stroke:var(--flow);stroke-dasharray:3 4;stroke-opacity:.9;stroke-width:1.8}.w.hide{display:none}
@keyframes fl{to{stroke-dashoffset:-22}}
.side{border-left:1px solid var(--line);background:var(--panel);overflow:auto;padding:12px;min-height:0}
.side h3{margin:0 0 6px;font-size:13px}.side h4{margin:12px 0 4px;color:var(--dim);font-size:10px;letter-spacing:.14em;font-weight:400}
.kv{display:grid;grid-template-columns:96px 1fr;gap:2px 8px;margin:0}.kv dt{color:var(--dim)}.kv dd{margin:0;white-space:pre-wrap;word-break:break-word}
.chip{display:inline-block;padding:1px 7px;border-radius:9px;border:1px solid var(--line);margin:1px 3px 1px 0}
.big{font-size:22px;font-weight:700}.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.bottom{display:grid;grid-template-columns:1.2fr 1fr 1fr;border-top:1px solid var(--line);min-height:0}
.bp{border-right:1px solid var(--line);padding:8px 12px;overflow:auto;min-height:0}.bp h5{margin:0 0 6px;color:var(--dim);letter-spacing:.14em;font-size:10px;font-weight:400}
.lg div{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--dim)}.lg div.n{color:var(--fg)}.lg .f{color:var(--bad)}.lg .o{color:var(--ok)}
.sb{display:grid;grid-template-columns:150px 1fr 40px;align-items:center;gap:8px;margin:3px 0}.sb .t{height:7px;background:var(--line);border-radius:4px;position:relative}
.sb .t i{position:absolute;left:0;top:0;height:100%;border-radius:4px}.sb .t u{position:absolute;top:-3px;bottom:-3px;width:1px;background:var(--fg);opacity:.5}
.modal{position:fixed;inset:0;background:#000b;z-index:50;display:flex;align-items:center;justify-content:center;padding:14px}.modal[hidden]{display:none}
.mbox{background:var(--panel);border:1px solid var(--line);border-radius:10px;max-width:1080px;width:100%;max-height:94vh;overflow:auto;padding:16px}
.mh{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;font-size:14px}.mgrid{display:grid;grid-template-columns:minmax(300px,1fr) minmax(320px,1.25fr);gap:20px}
.pyr{display:flex;flex-direction:column;align-items:center;gap:4px}.pyr h4{margin:0 0 6px;color:var(--dim);font-size:10px;letter-spacing:.14em;font-weight:400;align-self:flex-start}
.tier{width:var(--w);min-width:230px;border-radius:6px;padding:7px 10px;color:#fff;cursor:pointer;text-align:center;transition:transform .15s}.tier:hover{transform:scale(1.015)}
.tier b{display:block}.tier span{font-size:11px;opacity:.95}.tier ul{margin:6px 0 0;padding-left:16px;text-align:left;font-size:11px;display:none}.tier.open ul{display:block}
.banner{background:#7f1d1d;color:#fff;border-radius:8px;padding:10px 12px;margin-bottom:12px;font-size:12px}
.qrow{margin:9px 0}.qrow .g{display:inline-block;font-size:10px;letter-spacing:.1em;padding:1px 6px;border-radius:4px;background:var(--line);margin-right:6px}.qrow .g.crit{background:var(--bad);color:#fff}
.qrow input{width:100%;margin-top:4px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 8px;font:inherit}
.opts button{margin:3px 4px 0 0;padding:2px 8px;font-size:11px}.mf{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}
.mf .go{background:var(--flow);color:#1a1000;border:0;font-weight:700}.msg{white-space:pre-wrap;color:var(--dim);font-size:11px;border-top:1px solid var(--line);margin-top:12px;padding-top:8px;max-height:140px;overflow:auto}
.libi{border:1px solid var(--line);border-radius:6px;padding:6px 8px;margin:6px 0;background:var(--bg);font-size:11px}.libi .t{font-weight:700;font-size:12px}
.chk{display:flex!important;gap:6px;align-items:center;text-transform:none!important;letter-spacing:0!important;color:var(--fg)!important;font-size:12px!important}.chk input{width:auto!important}
@media(max-width:900px){.mgrid{grid-template-columns:1fr}}
#cfg{width:340px;border-right:1px solid var(--line);background:var(--panel);overflow:auto;padding:0 0 70px;position:relative;min-height:0}
#cfg .cfgh{padding:10px 12px;letter-spacing:.14em;color:var(--dim);font-size:10px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between}
#cfg details{border-bottom:1px solid var(--line);padding:0 12px}#cfg summary{padding:9px 0;cursor:pointer;font-weight:700}
#cfg label{display:block;color:var(--dim);margin:8px 0 3px;font-size:10px;letter-spacing:.08em;text-transform:uppercase}
#cfg textarea,#cfg input[type=text],#cfg input[type=number],#cfg select{width:100%;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:inherit}
#cfg textarea{resize:vertical}#cfg details>*:last-child{margin-bottom:12px}
.prm{display:grid;grid-template-columns:1fr 46px;gap:2px 8px;align-items:center}.prm label{grid-column:1/3}.prm b{text-align:right}
.rrow{border:1px solid var(--line);border-radius:6px;padding:6px 8px;margin:6px 0;background:var(--bg)}.rrow.req{border-color:var(--ok)}.rrow.exc{opacity:.5;border-style:dashed}
.rrow .rt{display:flex;justify-content:space-between;gap:6px;align-items:center}.rrow select{width:92px!important;padding:2px 4px!important}.rrow textarea{margin-top:5px}
.kdoc{display:flex;gap:6px;align-items:center;border:1px solid var(--line);border-radius:6px;padding:4px 6px;margin:4px 0;background:var(--bg)}.kdoc span{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kdoc input{width:52px!important}.kdoc button{padding:0 6px}
.runbar{position:sticky;bottom:0;left:0;right:0;background:var(--panel);border-top:1px solid var(--line);padding:10px 12px;margin-top:-62px;z-index:3}
.runbtn{width:100%;padding:9px;font-weight:700;background:var(--flow);color:#1a1000;border:0;border-radius:7px}.runbtn:disabled{opacity:.5}
#err{color:var(--bad);margin-top:6px;white-space:pre-wrap}.hint{color:var(--dim);font-size:11px;margin:4px 0}
.empty{position:absolute;left:40px;top:60px;color:var(--dim);max-width:520px;font-size:14px;line-height:1.7}
@media(max-width:900px){.main{grid-template-columns:1fr}#cfg{width:auto}.side{display:none}.bottom{grid-template-columns:1fr}body{grid-template-rows:auto auto 1fr 120px}}
</style></head><body>
<div class="top"><b>PIPELINE</b><span class="q" id="q"></span><span>NODES <b id="sn">0</b></span><span>EDGES <b id="se">0</b></span><span>ATTEMPT <b id="sa">1</b></span><span>SCORE <b id="ss">–</b></span><span>T <b id="st">0.00</b></span><span>STEP <b id="sp">0</b></span></div>
<div class="ctl"><div class="tabs" id="tabs"></div>
<button id="play">▶ Play</button><button id="prev">◀</button><button id="next">▶|</button>
<input type="range" id="scrub" min="0" value="0"><select id="speed"><option value="1.6">0.6×</option><option value="1" selected>1×</option><option value="0.5">2×</option><option value="0.2">5×</option></select>
<button id="cfgbtn">⚙ Setup</button><button id="follow" class="on">follow</button><button id="zm">zoom −</button><button id="zp">zoom +</button>
<span class="legend"><i style="background:var(--dim)"></i>input port<i style="background:var(--c1)"></i>output port<i style="background:var(--flow)"></i>live / regenerate loop<i style="background:var(--bad)"></i>failed gate</span></div>
<div class="main" id="main"><aside id="cfg" style="display:none">
<div class="cfgh"><span>SETUP</span><span id="srv"></span></div>
<details open><summary>1 · Prompt</summary>
<label>Question / research prompt</label><textarea id="f-q" rows="4" placeholder="e.g. What are the surgical, rehabilitation and toxicity considerations after severe traumatic brain injury?"></textarea>
<label>Mode</label><select id="f-mode"><option value="ask">Single question</option><option value="loop">Loop: discuss → follow-up questions → discuss</option></select>
<label>Examples</label><select id="f-ex"></select><div style="margin-top:8px"><button id="f-und">① Understand &amp; confirm first</button></div><div class="hint">Language pyramid: words → structure → meaning → intent → context. I reply with what I understood plus when / how / why / where / place / situation questions before any research (clearance 1).</div></details>
<details><summary>2 · Roles</summary>
<label>Global system prompt (applies to every role)</label><textarea id="f-sys" rows="5"></textarea>
<label>Roster — all healthcare (<span id="rcount"></span>)</label><input type="text" id="f-rsearch" placeholder="search any role, e.g. nurse practitioner, pharmacist, dentist…"><select id="f-ros" style="margin-top:6px"></select>
<div class="hint">auto = router decides · required = always on the panel · excluded = never. Role instructions are added to that expert's prompt.</div><div id="roster"></div></details>
<details><summary>3 · Expertise demands</summary>
<label>Demands (injected into every expert + chair prompt)</label><textarea id="f-dem" rows="4" placeholder="e.g. Prefer RCTs and meta-analyses. Address paediatric patients separately. Flag any dosing claims."></textarea>
<label>Primary umbrella</label><select id="f-umb"></select><div id="p-sec3"></div></details>
<details><summary>4 · Parameters</summary><div id="p-sec4"></div>
<label>LLM</label><select id="p-llm"><option value="mock">Offline mock (no API, extractive)</option><option value="anthropic">Anthropic API (needs ANTHROPIC_API_KEY on server)</option></select>
<label class="chk"><input type="checkbox" id="p-clr"> require confirmation (clearance 1 → 2) before research</label><label class="chk"><input type="checkbox" id="p-all"> ALL specialties: every umbrella discusses internally + plenary</label><label class="chk"><input type="checkbox" id="p-reach"> auto-extend reach to the question's demands</label><label class="chk"><input type="checkbox" id="p-lib"> use + train evidence library</label><div id="p-sec4b"></div>
<label>Model</label><input type="text" id="p-model"><label><input type="checkbox" id="p-pubmed" style="width:auto"> also search PubMed (needs network)</label></details>
<details><summary>5 · Knowledge</summary><div class="hint">Evidence the experts may cite. Each doc gets a quality weight (trial/review ≈ 0.9, opinion ≈ 0.3).</div>
<div id="kdocs"></div><label>Add text</label><input type="text" id="k-title" placeholder="title"><textarea id="k-text" rows="4" placeholder="paste abstract / guideline / notes"></textarea>
<div style="display:flex;gap:6px;margin-top:6px"><input type="number" id="k-q" value="0.7" min="0" max="1" step="0.05" style="width:70px"><button id="k-add">+ add</button><label style="margin:0;flex:1"><input type="file" id="k-file" accept=".txt,.md" multiple style="display:none"><span class="chip" style="cursor:pointer;display:block;text-align:center">upload .txt/.md</span></label></div></details>
<details id="libsec"><summary>6 · Evidence library</summary><div class="hint">Score library per specialty and a common library per umbrella. Scores = quality, study design, recency, source and fit, then learned from what the experts cite in answers that pass the gate.</div><label>Library</label><select id="lib-scope"></select><div id="lib-list"></div></details>
<details><summary>7 · Learning</summary><div id="lstats" class="hint"></div>
<label>Training questions (one per line, runs in background)</label><textarea id="t-q" rows="4"></textarea>
<div style="display:flex;gap:6px;margin-top:6px"><button id="t-go">Train in background</button><button id="t-exp">Export data</button></div><div id="t-log" class="hint"></div></details>
<div class="runbar"><button class="runbtn" id="run">▶ Run pipeline</button><div id="err"></div></div></aside><div id="cv"><div id="stage"><svg id="wires"></svg></div></div><aside class="side" id="side"><div style="color:var(--dim)">Run summary, final answer and per-node inputs/outputs appear here after you press Run.</div></aside></div>
<div class="bottom"><div class="bp"><h5>RUN LOG</h5><div class="lg" id="log"></div></div><div class="bp"><h5>SPECIALISTS · CONFIDENCE</h5><div id="spec"></div></div><div class="bp"><h5>SCORE &amp; GATE</h5><div id="sc"></div></div></div>
<script>
let RUNS=__DATA__;
const LANES=["Inputs","Language pyramid","Clearance & reach","Gather + library","Route (multi-term)","Internal specialists","Internal debate","Umbrella lead","Other umbrellas / external","All-umbrella plenary","Chair","Score & filter","Learn & loop","Output"];
const LW=270,NW=236,TOP=34,GAP=18;
let run,ev,cur=0,timer=null,sel=null,follow=true,zoom=1,pos={};
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function sm(v,n=46){if(v==null)return"–";if(typeof v==="number")return Number.isInteger(v)?String(v):v.toFixed(2);if(typeof v==="boolean")return String(v);
 if(Array.isArray(v)){if(!v.length)return"[ ]";const f=typeof v[0]==="object"?(v[0].title||v[0].id||"item"):String(v[0]);return`${v.length} · ${f}`.slice(0,n)}
 if(typeof v==="object"){const k=Object.keys(v);return`{${k.length}} ${k.slice(0,3).join(", ")}`.slice(0,n)}
 const s=String(v).replace(/\s+/g," ");return s.length>n?s.slice(0,n-1)+"…":s||"–"}
function full(v){return typeof v==="string"?v:JSON.stringify(v,null,1).replace(/[{}\[\]",]/g,"").replace(/\n\s*\n/g,"\n").trim()}
function loadRun(i){
 if(!RUNS[i])return;$("stage").querySelector(".empty")?.remove();
 run=RUNS[i];stop();cur=0;sel=null;
 ev=run.trace.map(e=>({...e}));const last=[...ev].reverse().find(e=>e.node.startsWith("score"));
 if(run.status!=="awaiting_confirmation")ev.push({seq:ev.length,t:(ev[ev.length-1]||{t:0}).t,node:"out:answer",label:"Final answer",lane:13,after:[last?last.node:""],inputs:{status:run.status,composite:run.composite,attempts:run.attempts},outputs:{answer:run.answer,umbrella:run.umbrella},status:run.status==="accepted"?"ok":"fail",note:run.status.toUpperCase()});
 document.querySelectorAll(".tab").forEach((t,j)=>t.classList.toggle("sel",j===i));
 $("q").textContent=run.question;$("scrub").max=ev.length-1;layout();render();play()}
function rowsOf(o,cls,max=4){const ks=Object.keys(o);return ks.slice(0,max).map(k=>`<div class="r ${cls}"><span class="k">${esc(k)}</span><span class="v">${esc(sm(o[k]))}</span></div>`).join("")+(ks.length>max?`<div class="more">+${ks.length-max} more ›</div>`:"")}
function hOf(e){return 24+(Object.keys(e.inputs).length?14+Math.min(4,Object.keys(e.inputs).length)*17+(Object.keys(e.inputs).length>4?14:0):0)+(Object.keys(e.outputs).length?14+Math.min(4,Object.keys(e.outputs).length)*17+(Object.keys(e.outputs).length>4?14:0):0)+(e.score!=null?10:0)+8}
function layout(){
 const st=$("stage");st.querySelectorAll(".node,.lane").forEach(n=>n.remove());pos={};
 const byLane={};ev.forEach(e=>(byLane[e.lane]=byLane[e.lane]||[]).push(e));
 const heights=Object.values(byLane).map(a=>a.reduce((s,e)=>s+hOf(e)+GAP,0));const H=Math.max(...heights)+TOP+40;
 LANES.forEach((n,i)=>{const d=document.createElement("div");d.className="lane";d.style.left=24+i*LW+"px";d.textContent=`${i+1} · ${n}`;st.appendChild(d)});
 Object.entries(byLane).forEach(([l,arr])=>{const th=arr.reduce((s,e)=>s+hOf(e)+GAP,-GAP);let y=TOP+(H-TOP-th)/2;
  arr.forEach(e=>{const h=hOf(e),x=24+l*LW;pos[e.node]={x,y,h,e};const d=document.createElement("div");d.className="node pending";d.dataset.n=e.node;d.style.cssText=`left:${x}px;top:${y}px;--c:var(--c${l})`;
   const att=/#(\d)/.exec(e.node);
   d.innerHTML=`<div class="hd" style="background:var(--c${l})"><span>${esc(e.label)}</span><small>${att?"↻ attempt "+att[1]:e.t.toFixed(2)+"s"}</small></div><div class="rows">`+
    (Object.keys(e.inputs).length?`<div class="sec">INPUT</div>${rowsOf(e.inputs,"in")}`:"")+(Object.keys(e.outputs).length?`<div class="sec">OUTPUT</div>${rowsOf(e.outputs,"out")}`:"")+
    (e.score!=null?`<div class="bar"><i style="width:${e.score*100}%;background:${e.status==="ok"?"var(--ok)":"var(--bad)"}"></i></div>`:"")+`<div class="pad"></div></div>`;
   d.onclick=()=>{sel=e.node;render()};st.appendChild(d);y+=h+GAP})});
 const W=24+LANES.length*LW,svg=$("wires");svg.setAttribute("width",W);svg.setAttribute("height",H);svg.innerHTML="";
 st.style.width=W+"px";st.style.height=H+"px";
 const mk=(a,b,cls)=>{const A=pos[a],B=pos[b];if(!A||!B)return;const x1=A.x+NW,y1=A.y+14,x2=B.x,y2=B.y+14,dx=Math.max(40,(x2-x1)/2);
  const p=document.createElementNS("http://www.w3.org/2000/svg","path");p.setAttribute("d",`M${x1},${y1} C${x1+dx},${y1} ${x2-dx},${y2} ${x2},${y2}`);p.setAttribute("class","w hide "+cls);p.dataset.a=a;p.dataset.b=b;svg.appendChild(p)};
 ev.forEach((e,i)=>{e.after.forEach(a=>mk(a,e.node,""));
  if(e.node.startsWith("score")&&e.status==="fail"){const n=ev[i+1];if(n&&!/^(store|followups|out:)/.test(n.node))mk(e.node,n.node,"regen")}});
 $("se").textContent=svg.querySelectorAll("path").length;$("sn").textContent=0;
}
function render(){
 const i=cur,idx=Object.fromEntries(ev.map((e,k)=>[e.node,k]));
 document.querySelectorAll(".node").forEach(n=>{const k=idx[n.dataset.n];n.classList.toggle("pending",k>i);n.classList.toggle("active",k===i);n.classList.toggle("sel",n.dataset.n===sel);
  n.classList.toggle("fail",k<=i&&ev[k].status==="fail"&&!n.dataset.n.startsWith("out"))});
 document.querySelectorAll("path.w").forEach(p=>{const a=idx[p.dataset.a],b=idx[p.dataset.b];p.classList.toggle("hide",!(a<=i&&b<=i));p.classList.toggle("live",b===i&&!p.classList.contains("regen"))});
 $("sn").textContent=Math.min(i+1,ev.length);$("sp").textContent=`${i+1}/${ev.length}`;$("st").textContent=ev[i].t.toFixed(2);$("scrub").value=i;
 const sc=[...ev.slice(0,i+1)].reverse().find(e=>e.node.startsWith("score"));$("sa").textContent=(ev.slice(0,i+1).filter(e=>e.node.startsWith("score")).length)||1;
 $("ss").textContent=sc?sc.outputs.composite:"–";
 // log
 $("log").innerHTML=ev.slice(0,i+1).map((e,k)=>`<div class="${k===i?"n":""} ${e.status==="fail"?"f":""}${e.node==="out:answer"&&e.status==="ok"?"o":""}">${String(e.seq).padStart(2,"0")}  ${e.t.toFixed(2).padStart(5)}s  ${esc(e.label)}${e.note?" · "+esc(e.note):""}</div>`).join("");$("log").scrollTop=1e5;
 // specialists
 const sp=ev.slice(0,i+1).filter(e=>/^(spec|ext|umb):/.test(e.node));
 $("spec").innerHTML=sp.map(e=>`<div class="sb"><span>${esc(e.label.slice(0,22))}</span><div class="t"><i style="width:${e.outputs.confidence*100}%;background:${/^(ext|umb)/.test(e.node)?"var(--c8)":"var(--c5)"}"></i></div><span>${e.outputs.confidence.toFixed(2)}</span></div>`).join("")||'<span style="color:var(--dim)">waiting…</span>';
 // score
 $("sc").innerHTML=sc?Object.entries(sc.outputs.dims).map(([k,v])=>`<div class="sb"><span>${k}</span><div class="t"><i style="width:${v*100}%;background:${v>=.6?"var(--ok)":"var(--bad)"}"></i><u style="left:60%"></u></div><span>${v.toFixed(2)}</span></div>`).join("")+`<div style="margin-top:6px">composite <b class="big ${sc.status==="ok"?"ok":"bad"}">${sc.outputs.composite}</b> <span class="chip">${sc.note}</span></div>`:'<span style="color:var(--dim)">waiting…</span>';
 side();
 if(follow&&pos[ev[i].node]){const p=pos[ev[i].node],c=$("cv");c.scrollTo({left:Math.max(0,(p.x-120)*zoom),top:Math.max(0,(p.y-80)*zoom),behavior:"smooth"})}
}
function side(){const e=sel?ev.find(x=>x.node===sel):null,s=$("side");
 if(!e){s.innerHTML=`<h3>Run summary</h3><dl class="kv"><dt>question</dt><dd>${esc(run.question)}</dd><dt>umbrella</dt><dd>${esc(run.umbrella)}</dd><dt>status</dt><dd class="${run.status==="accepted"?"ok":"bad"}">${run.status}</dd><dt>score</dt><dd><b class="big">${run.composite}</b></dd><dt>attempts</dt><dd>${run.attempts}</dd></dl>
 <h4>UNDERSTOOD AS</h4><div>${esc(run.understanding?run.understanding.restatement.replace(/\*\*/g,""):"–")}</div>${run.reach&&run.reach.reasons.length?`<h4>REACH EXTENDED</h4>${run.reach.reasons.map(r=>`<div>+ ${esc(r)}</div>`).join("")}`:""}${run.coverage?`<h4>SPECIALTY COVERAGE · all ${run.coverage.length} umbrellas</h4>${[...new Set(run.coverage.map(c=>c.category))].map(cat=>`<div style="margin:4px 0"><div style="color:var(--dim);font-size:10px">${esc(cat)}</div>${run.coverage.filter(c=>c.category===cat).map(c=>`<span class="chip" title="${esc(c.name)} · ${c.roles} roles · relevance ${c.relevance}${c.consulted.length?" · "+esc(c.consulted.join(", ")):""}" style="${c.status==="lead"?"background:var(--ok);color:#06240f;border-color:var(--ok)":c.status==="reach"?"background:var(--flow);color:#1a1000;border-color:var(--flow)":c.status==="consulted"?"border-color:var(--ok);color:var(--ok)":"opacity:.55"}">${esc(c.name)}${c.consulted.length?" · "+c.consulted.length:""}</span>`).join("")}</div>`).join("")}<div class="hint">filled green = lead · orange = added by reach · green outline = consulted · faded = available, not needed for this question (turn on “consult ALL specialties” to include them)</div>`:""}<h4>FINAL ANSWER (OUTPUT)</h4><div style="white-space:pre-wrap">${esc(run.answer)}</div><h4>NEXT QUESTIONS (LOOP)</h4>${run.followups.map(f=>`<div>→ ${esc(f)}</div>`).join("")||"–"}<h4>TIP</h4><div style="color:var(--dim)">Click any node to inspect its full inputs and outputs. Orange dashed wire = a failed gate sending the draft back for regeneration.</div>`;return}
 const up=e.after.filter(Boolean).map(a=>`<span class="chip">${esc((ev.find(x=>x.node===a)||{label:a}).label)}</span>`).join(""),dn=ev.filter(x=>x.after.includes(e.node)).map(x=>`<span class="chip">${esc(x.label)}</span>`).join("");
 const kv=o=>`<dl class="kv">${Object.entries(o).map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(full(v))}</dd>`).join("")||"<dd>–</dd>"}</dl>`;
 s.innerHTML=`<h3>${esc(e.label)} <span class="chip ${e.status==="ok"?"ok":"bad"}">${e.status}</span></h3><div style="color:var(--dim)">${LANES[e.lane]} · t=${e.t.toFixed(2)}s ${e.note?"· "+esc(e.note):""}</div><h4>◀ FED BY</h4>${up||"–"}<h4>INPUT</h4>${kv(e.inputs)}<h4>OUTPUT ▶</h4>${kv(e.outputs)}<h4>FEEDS</h4>${dn||"–"}<h4><a href="#" onclick="sel=null;render();return false" style="color:var(--flow)">← run summary</a></h4>`}
function play(){stop();$("play").textContent="⏸ Pause";$("play").classList.add("on");timer=setInterval(()=>{if(cur>=ev.length-1)return stop();cur++;render()},900*parseFloat($("speed").value))}
function stop(){clearInterval(timer);timer=null;$("play").textContent="▶ Play";$("play").classList.remove("on")}
$("play").onclick=()=>timer?stop():(cur>=ev.length-1&&(cur=0),play());
$("prev").onclick=()=>{stop();cur=Math.max(0,cur-1);render()};$("next").onclick=()=>{stop();cur=Math.min(ev.length-1,cur+1);render()};
$("scrub").oninput=e=>{stop();cur=+e.target.value;render()};$("speed").onchange=()=>timer&&play();
$("follow").onclick=e=>{follow=!follow;e.target.classList.toggle("on",follow)};
const z=d=>{zoom=Math.min(1.4,Math.max(.3,zoom+d));$("stage").style.transform=`scale(${zoom})`};$("zm").onclick=()=>z(-.15);$("zp").onclick=()=>z(.15);
function buildTabs(){$("tabs").innerHTML=RUNS.map((r,i)=>`<div class="tab ${r.status==="accepted"?"acc":"rev"}" title="${esc(r.question)}" onclick="loadRun(${i})">Q${i+1}${r.depth?" ↳"+r.depth:""} · ${r.composite}</div>`).join("")}
buildTabs();
if(RUNS.length)loadRun(0);else{$("stage").insertAdjacentHTML("beforeend",'<div class="empty">Set the prompt, roles, expertise demands, parameters and knowledge in <b>⚙ Setup</b>, then press <b>Run pipeline</b>.<br>The graph shows every step: its <b>inputs</b> (left ports) and <b>outputs</b> (right ports), the prompt each expert received, the score gate, and any regeneration loop.</div>')}
/* ================= SETUP PANEL (needs the local server: python -m research_pipeline serve) ================= */
const EXAMPLES=["What are the surgical, rehabilitation and toxicity considerations after severe traumatic brain injury?","How should atrial fibrillation be managed after an ischemic stroke?","What does the evidence say about early mobilisation after stroke?"];
let META=null,C={question:"",mode:"ask",system_prompt:"",demands:"",umbrella:"auto",params:{},state:{},role_prompts:{},knowledge:[],roster:"all",train:""};
const PR=[["threshold","Pass threshold (composite score)",.4,.95,.01],["max_attempts","Max regeneration attempts",1,6,1],["evidence_n","Evidence items per attempt",2,20,1],["loop_depth","Loop depth (follow-up rounds)",0,4,1],["max_questions","Max questions in a loop",1,15,1]];
const PR3=[["top_internal","Internal experts on panel",1,8,1],["top_external","External experts (other umbrellas)",0,5,1],["min_quality","Min evidence quality",0,1,.05]];
function save(){try{localStorage.setItem("rp-cfg",JSON.stringify(C))}catch(e){}}
function slider(id,label,mn,mx,st,box){const d=document.createElement("div");d.className="prm";d.innerHTML=`<label>${label}</label><input type="range" min="${mn}" max="${mx}" step="${st}" id="p-${id}"><b id="v-${id}"></b>`;$(box).appendChild(d);
 const r=d.querySelector("input");r.value=C.params[id]??META.defaults[id];const u=()=>{C.params[id]=+r.value;$("v-"+id).textContent=r.value;save()};r.oninput=u;u()}
function roster(){if(C.roster==="all"&&!($("f-rsearch").value||"").trim()){rosterAll();return}
 const u=META.umbrellas.find(x=>x.id===C.roster)||META.umbrellas[0],q=($("f-rsearch").value||"").toLowerCase().trim();$("roster").innerHTML="";
 const list=q?META.umbrellas.flatMap(x=>x.specialties.filter(s=>(s.name+" "+s.tier+" "+x.name+" "+s.keywords.join(" ")).toLowerCase().includes(q))):u.specialties;
 if(q&&!list.length)$("roster").innerHTML='<div class="hint">No role matches.</div>';
 list.slice(0,60).forEach(s=>{const st=C.state[s.id]||"auto",d=document.createElement("div");d.className="rrow "+(st==="required"?"req":st==="excluded"?"exc":"");
  d.innerHTML=`<div class="rt"><span><b>${esc(s.name)}</b> <span class="chip">${s.tier}</span></span><select><option>auto</option><option>required</option><option>excluded</option></select></div><details class="scp"><summary style="font-size:11px;color:var(--dim);padding:3px 0">scope of practice</summary><div class="hint"><b>Core:</b> ${esc(s.scope)}<br><b>Extended / advanced:</b> ${esc(s.extended)}<br><b>Refer on:</b> ${esc(s.refer)}<br><b>Examination &amp; workup:</b> ${esc(s.workup)}<br><b>Route terms:</b> ${esc((s.terms||[]).join(", "))}</div></details><textarea rows="2" placeholder="Role instruction, e.g. focus on paediatric cases; cite guidelines"></textarea>`;
  const sel=d.querySelector("select"),ta=d.querySelector("textarea");sel.value=st;ta.value=C.role_prompts[s.id]||"";
  sel.onchange=()=>{C.state[s.id]=sel.value;save();roster()};ta.oninput=()=>{C.role_prompts[s.id]=ta.value;save()};$("roster").appendChild(d)})}
function rowEl(s){const st=C.state[s.id]||"auto",d=document.createElement("div");d.className="rrow "+(st==="required"?"req":st==="excluded"?"exc":"");
 d.innerHTML=`<div class="rt"><span><b>${esc(s.name)}</b> <span class="chip">${s.tier}</span></span><select><option>auto</option><option>required</option><option>excluded</option></select></div><details class="scp"><summary style="font-size:11px;color:var(--dim);padding:3px 0">scope of practice</summary><div class="hint"><b>Core:</b> ${esc(s.scope)}<br><b>Extended / advanced:</b> ${esc(s.extended)}<br><b>Refer on:</b> ${esc(s.refer)}<br><b>Examination &amp; workup:</b> ${esc(s.workup)}<br><b>Route terms:</b> ${esc((s.terms||[]).join(", "))}</div></details><textarea rows="2" placeholder="Role instruction"></textarea>`;
 const sel=d.querySelector("select"),ta=d.querySelector("textarea");sel.value=st;ta.value=C.role_prompts[s.id]||"";
 sel.onchange=()=>{C.state[s.id]=sel.value;save();d.className="rrow "+(sel.value==="required"?"req":sel.value==="excluded"?"exc":"")};ta.oninput=()=>{C.role_prompts[s.id]=ta.value;save()};return d}
function rosterAll(){const box=$("roster");box.innerHTML="";
 [...new Set(META.umbrellas.map(u=>u.category))].forEach(cat=>{const h=document.createElement("div");h.className="hint";h.style.cssText="margin-top:10px;font-weight:700;color:var(--fg)";h.textContent=cat;box.appendChild(h);
  META.umbrellas.filter(u=>u.category===cat).forEach(u=>{const d=document.createElement("details");d.innerHTML=`<summary style="cursor:pointer;padding:3px 0">${esc(u.name)} <span class="chip">${u.specialties.length} roles</span></summary><div class="hint">${esc(u.body_areas.join(" · "))}</div>`;let done=false;
   d.ontoggle=()=>{if(d.open&&!done){done=true;u.specialties.forEach(s=>d.appendChild(rowEl(s)))}};box.appendChild(d)})})}
function kdocs(){$("kdocs").innerHTML="";C.knowledge.forEach((k,i)=>{const d=document.createElement("div");d.className="kdoc";d.innerHTML=`<span title="${esc(k.text.slice(0,300))}">${esc(k.title)}</span><input type="number" min="0" max="1" step="0.05" value="${k.quality}"><button>✕</button>`;
 d.querySelector("input").onchange=e=>{k.quality=+e.target.value;save()};d.querySelector("button").onclick=()=>{C.knowledge.splice(i,1);save();kdocs()};$("kdocs").appendChild(d)})}
function addDoc(title,text,q){if(text.trim()){C.knowledge.push({title:title||text.trim().slice(0,50),text:text.trim(),quality:q});save();kdocs()}}

/* ===== clearance modal, reach, library ===== */
const PR4=[["max_reach_extra","Max extra experts from reach",0,6,1],["per_umbrella","Roles per umbrella (all-specialty mode)",1,4,1]];
const WL={1:"Lexical · words & terms",2:"Syntactic · structure",3:"Semantic · meaning",4:"Pragmatic · intent",5:"Contextual · situation"};
const clean=t=>String(t||"").replace(/\*\*/g,"").replace(/\*/g,"");
function pyramid(u){const W={5:44,4:58,3:72,2:86,1:100};
 return `<div class="pyr"><h4>LANGUAGE PYRAMID — how I read you (click a tier)</h4>`+[5,4,3,2,1].map(n=>{const L=u.levels[n];return `<div class="tier" style="--w:${W[n]}%;background:var(--c${n})" onclick="this.classList.toggle('open')"><b>${n} · ${esc(WL[n])}</b><span>${esc(clean(L.summary).slice(0,150))}</span><ul>${L.items.map(i=>`<li>${esc(clean(i))}</li>`).join("")}</ul></div>`}).join("")+`</div>`}
function showClearance(u,answers={}){
 const m=$("clr");m.hidden=false;
 m.innerHTML=`<div class="mbox"><div class="mh"><b>Clearance 1 · confirm my understanding before I research</b><button id="clr-x">✕</button></div>
 ${u.emergent?`<div class="banner">⚠ <b>This may be an emergency.</b> If this is happening now (chest pain, stroke signs, trouble breathing, overdose, thoughts of suicide) call your local emergency number or go to the nearest emergency department. Research output is not a substitute for urgent care.</div>`:""}
 <div class="mgrid">${pyramid(u)}<div><div style="margin-bottom:6px"><b>I understand that…</b></div><div style="font-size:13px">${esc(clean(u.restatement))}</div>
 <div class="hint" style="margin-top:6px">Perspective: ${esc(u.perspective)} · urgency: ${esc(u.urgency)} · clearance level ${u.clearance}${u.ready?" · enough to proceed":" · critical details missing"}</div>
 <div style="margin-top:12px"><b>To get it exactly right, tell me (skip any you don't know):</b></div><div id="qs">${u.questions.length?u.questions.map((q,i)=>`<div class="qrow"><span class="g ${q.critical?"crit":""}">${q.group}</span>${esc(q.text)}<div class="opts">${(q.options||[]).map(o=>`<button data-k="${esc(q.key)}" data-v="${esc(o)}">${esc(o.length>44?o.slice(0,42)+"…":o)}</button>`).join("")}</div><input type="text" data-key="${esc(q.key)}" placeholder="your answer (optional)" value="${esc(answers[q.key]||"")}"></div>`).join(""):'<div class="hint">Nothing critical is missing.</div>'}</div></div></div>
 <div class="msg">${esc(clean(u.message))}</div>
 <div class="mf"><button class="go" id="clr-go">✔ Confirm &amp; run research</button><button id="clr-up">↻ Update understanding with my answers</button><button id="clr-skip">Skip questions &amp; run</button><button id="clr-cancel">Cancel</button></div></div>`;
 const collect=()=>{const a={...answers};m.querySelectorAll("input[data-key]").forEach(i=>{const k=i.dataset.key,t=i.value.trim();if(t)a[k.startsWith("ambiguity:")?"clarify_"+k.slice(10):k]=t});return a};
 m.querySelectorAll(".opts button").forEach(b=>b.onclick=()=>{m.querySelector(`input[data-key="${b.dataset.k}"]`).value=b.dataset.v});
 const close=()=>{m.hidden=true;m.innerHTML=""};
 $("clr-x").onclick=$("clr-cancel").onclick=close;
 $("clr-up").onclick=async()=>{const a=collect();try{const j=await api("/api/understand",{...req(),context:{answers:a}});showClearance(j.understanding,a)}catch(e){$("err").textContent=e.message}};
 $("clr-go").onclick=$("clr-skip").onclick=async e=>{const a=e.target.id==="clr-skip"?{...answers}:collect();close();await execute({answers:a,confirmed:true})};
}
async function execute(context){$("err").textContent="";$("run").disabled=true;$("run").textContent="⏳ running…";
 try{const j=await api("/api/run",{...req(),context});
  if(j.awaiting){RUNS=[{question:C.question,understanding:j.understanding,reach:null,answer:"",status:"awaiting_confirmation",attempts:0,depth:0,umbrella:"",composite:0,dims:{},followups:[],trace:j.trace}];buildTabs();loadRun(0);showClearance(j.understanding,context&&context.answers||{})}
  else{RUNS=j.runs;buildTabs();loadRun(0)}
  stats(j.stats);refreshLib()}catch(e){$("err").textContent=e.message}
 $("run").disabled=false;$("run").textContent="▶ Run pipeline"}
async function refreshLib(){try{const m=await api("/api/meta");const ov=Object.fromEntries(m.library.map(o=>[o.scope,o.items]));
 const cur=$("lib-scope").value;const roleOpts=(META.umbrellas.find(x=>x.id===C.roster)||{specialties:[]}).specialties.map(s=>`<option value="${s.id}">${esc(s.name)}</option>`).join("");
 $("lib-scope").innerHTML=`<optgroup label="Umbrella common libraries (all specialties merged)">`+META.umbrellas.map(u=>`<option value="umb:${u.id}">${esc(u.name)} — ${ov["umb:"+u.id]||0} items</option>`).join("")+`</optgroup><optgroup label="Specialty libraries · roster umbrella: ${esc(C.roster)}">${roleOpts}</optgroup>`;
 if(cur)$("lib-scope").value=cur;loadLib()}catch(e){}}
async function loadLib(){const sc=$("lib-scope").value;if(!sc){$("lib-list").innerHTML="";return}
 try{const r=await api("/api/library?scope="+encodeURIComponent(sc));
  $("lib-list").innerHTML=r.items.length?`<div class="hint">${r.count} item(s) · ${r.total_library} in the whole library</div>`+r.items.map(i=>`<div class="libi"><div class="t">${esc(i.title)}</div><div class="bar"><i style="width:${i.overall*100}%;background:${i.overall>=.7?"var(--ok)":i.overall>=.5?"var(--warn)":"var(--bad)"}"></i></div>
   <div style="margin-top:4px"><span class="chip">score ${i.overall}</span><span class="chip">${esc(i.design)}</span>${i.year?`<span class="chip">${i.year}</span>`:""}<span class="chip">cited ${i.cited}×</span><span class="chip">${esc(i.source.split(":")[0])}</span></div>
   <div style="color:var(--dim)">quality ${i.parts.quality} · design ${i.parts.design} · recency ${i.parts.recency} · source ${i.parts.source} · fit ${i.parts.fit} · learned ${i.learned}</div>
   ${i.by_specialty&&i.by_specialty.length?`<div>${i.by_specialty.map(b=>`<span class="chip">${esc(b.id)} ${b.score}</span>`).join("")}</div>`:""}</div>`).join(""):'<div class="hint">Empty — run the pipeline (or ingest documents) to build this library.</div>'}catch(e){$("lib-list").innerHTML='<div class="hint">'+esc(e.message)+'</div>'}}
function req(){const ids=v=>Object.entries(C.state).filter(([k,x])=>x===v).map(([k])=>k);
 return{question:C.question,mode:C.mode,params:{...C.params,llm:$("p-llm").value,model:$("p-model").value,pubmed:$("p-pubmed").checked,umbrella:C.umbrella,require_clearance:$("p-clr").checked,auto_reach:$("p-reach").checked,consult_all:$("p-all").checked,use_library:$("p-lib").checked},
  roles:{system_prompt:C.system_prompt,role_prompts:Object.fromEntries(Object.entries(C.role_prompts).filter(([k,v])=>v.trim()))},
  expertise:{demands:C.demands,required:ids("required"),excluded:ids("excluded")},knowledge:C.knowledge,questions:C.train.split("\n")}}
async function api(path,body){const r=await fetch(path,body?{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(body)}:{});const j=await r.json();if(!r.ok)throw new Error(j.error||r.statusText);return j}
function stats(st){$("lstats").textContent=`runs ${st.runs} · accepted ${st.accepted} · mean score ${st.mean_score} · mean attempts ${st.mean_attempts}`+(st.experts.length?" · best expert: "+st.experts[0].expert+" ("+st.experts[0].mean+")":"")}
async function initCfg(){
 try{META=await api("/api/meta")}catch(e){$("cfgbtn").style.display="none";return}
 try{Object.assign(C,JSON.parse(localStorage.getItem("rp-cfg")||"{}"))}catch(e){}
 $("cfg").style.display="";$("srv").textContent=META.corpus?"corpus: "+META.corpus.split("/").pop():"";$("srv").title=META.corpus||"";
 if(!C.system_prompt)C.system_prompt=META.defaults.system_prompt;if(!C.question)C.question=EXAMPLES[0];
 $("f-q").value=C.question;$("f-q").oninput=e=>{C.question=e.target.value;save()};$("f-mode").value=C.mode;$("f-mode").onchange=e=>{C.mode=e.target.value;save()};
 $("f-ex").innerHTML='<option value="">— load an example —</option>'+EXAMPLES.map((x,i)=>`<option value="${i}">${esc(x.slice(0,60))}…</option>`).join("");$("f-ex").onchange=e=>{if(e.target.value!==""){C.question=EXAMPLES[e.target.value];$("f-q").value=C.question;save()}};
 $("f-sys").value=C.system_prompt;$("f-sys").oninput=e=>{C.system_prompt=e.target.value;save()};
 $("f-dem").value=C.demands;$("f-dem").oninput=e=>{C.demands=e.target.value;save()};
 const cats=[...new Set(META.umbrellas.map(u=>u.category))];
 $("f-ros").innerHTML=`<option value="all">★ ALL healthcare — browse every umbrella (${META.umbrellas.length})</option>`+cats.map(c=>`<optgroup label="${esc(c)}">`+META.umbrellas.filter(u=>u.category===c).map(u=>`<option value="${u.id}">${esc(u.name)} (${u.specialties.length} roles)</option>`).join("")+"</optgroup>").join("");
 $("rcount").textContent=`${META.umbrellas.length} umbrellas · ${META.umbrellas.reduce((n,u)=>n+u.specialties.length,0)} roles`;$("f-rsearch").oninput=roster;if(!C.roster)C.roster="all";$("f-ros").value=C.roster;$("f-ros").onchange=e=>{C.roster=e.target.value;save();roster();refreshLib()};
 $("f-umb").innerHTML='<option value="auto">auto (router decides from the question)</option>'+cats.map(c=>`<optgroup label="${esc(c)}">`+META.umbrellas.filter(u=>u.category===c).map(u=>`<option value="${u.id}">${esc(u.name)}</option>`).join("")+"</optgroup>").join("");$("f-umb").value=C.umbrella;$("f-umb").onchange=e=>{C.umbrella=e.target.value;save()};
 PR.forEach(p=>slider(p[0],p[1],p[2],p[3],p[4],"p-sec4"));PR3.forEach(p=>slider(p[0],p[1],p[2],p[3],p[4],"p-sec3"));
 PR4.forEach(p=>slider(p[0],p[1],p[2],p[3],p[4],"p-sec4b"));
 $("p-all").checked=!!C.params.consult_all;$("p-all").onchange=()=>{C.params.consult_all=$("p-all").checked;save()};
 $("p-clr").checked=C.params.require_clearance??META.defaults.require_clearance;$("p-reach").checked=C.params.auto_reach??META.defaults.auto_reach;$("p-lib").checked=C.params.use_library??META.defaults.use_library;
 [$("p-clr"),$("p-reach"),$("p-lib")].forEach(c=>c.onchange=()=>{C.params.require_clearance=$("p-clr").checked;C.params.auto_reach=$("p-reach").checked;C.params.use_library=$("p-lib").checked;save()});
 $("lib-scope").onchange=loadLib;$("libsec").ontoggle=()=>{if($("libsec").open)refreshLib()};
 $("f-und").onclick=async()=>{$("err").textContent="";try{const j=await api("/api/understand",{...req(),context:{answers:{}}});showClearance(j.understanding,{})}catch(e){$("err").textContent=e.message}};
 $("p-llm").value=C.params.llm||"mock";$("p-model").value=C.params.model||META.defaults.model;$("p-pubmed").checked=!!C.params.pubmed;
 if(!META.llm_ready)$("p-llm").options[1].text+=" — key NOT set";
 roster();kdocs();stats(META.stats);refreshLib();$("t-q").value=C.train;$("t-q").oninput=e=>{C.train=e.target.value;save()};
 $("k-add").onclick=()=>{addDoc($("k-title").value,$("k-text").value,+$("k-q").value);$("k-title").value=$("k-text").value=""};
 $("k-file").onchange=async e=>{for(const f of e.target.files){const t=await f.text();addDoc(f.name.replace(/\.\w+$/,""),t,+$("k-q").value)}e.target.value=""};
 $("run").onclick=()=>execute({answers:{},confirmed:!$("p-clr").checked});
 $("t-go").onclick=async()=>{try{await api("/api/train",req());poll()}catch(e){$("t-log").textContent=e.message}};
 const poll=async()=>{const s=await api("/api/train");$("t-log").textContent=`${s.running?"training… ":"done "}${s.done}/${s.total}\n`+s.log.slice(-5).map(l=>`${l.status} ${l.score} (${l.attempts}×) ${l.q.slice(0,40)}`).join("\n");$("t-log").style.whiteSpace="pre-wrap";
  if(s.running)setTimeout(poll,1000);else api("/api/meta").then(m=>stats(m.stats))};
 $("t-exp").onclick=async()=>{const x=await api("/api/export",{});[["sft.jsonl",x.sft],["preferences.jsonl",x.prefs]].forEach(([n,t])=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([t]));a.download=n;a.click()});$("t-log").textContent=`exported ${x.n_sft} SFT examples, ${x.n_prefs} preference pairs`};
}
$("cfgbtn").onclick=()=>$("main").classList.toggle("nocfg");
initCfg();
</script><div class="modal" id="clr" hidden></div></body></html>"""
