# -*- coding: utf-8 -*-
"""Customer page for climate history, event logs, and every screen shot."""
from __future__ import annotations

from typing import Any, Callable

from flask import Flask, Response, jsonify, make_response

from ibvs_pad_touch.web.panel_records import PanelRecords

StatusFn = Callable[[], dict[str, Any]]

PAGE_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>精密空调记录</title>
  <style>
    :root{color-scheme:light}
    body{margin:0;background:#f4f6f8;color:#1c2430;font-family:"Microsoft YaHei",Arial,sans-serif}
    header{background:#12324d;color:#fff;padding:22px 28px}
    header h1{margin:0 0 6px;font-size:26px;font-weight:600}
    header p{margin:0;color:#d5e4f2;font-size:14px}
    nav{display:flex;gap:16px;padding:12px 28px;background:#fff;border-bottom:1px solid #d9e1e8}
    nav a{color:#1d5f8a;text-decoration:none;font-size:14px}
    main{max-width:1100px;margin:0 auto;padding:20px 20px 48px}
    section{background:#fff;border:1px solid #e1e7ee;border-radius:10px;padding:16px 18px;margin:0 0 16px}
    h2{margin:0 0 12px;font-size:18px}
    .status{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
    .status div{background:#f7fafc;border-radius:8px;padding:10px 12px}
    .status span{display:block;color:#6b7785;font-size:12px;margin-bottom:4px}
    .big{font-size:32px;font-weight:650;letter-spacing:0;margin:0}
    .sub{color:#5c6b7a;margin:4px 0 0}
    .cards{display:grid;grid-template-columns:1.1fr .9fr;gap:16px}
    .shot{display:flex;gap:14px;align-items:flex-start}
    img{width:220px;max-width:100%;height:auto;background:#111;border-radius:6px;vertical-align:top}
    td img{width:160px}
    pre{white-space:pre-wrap;font-family:"Microsoft YaHei",Consolas,monospace;font-size:14px;line-height:1.55;margin:8px 0 0}
    table{width:100%;border-collapse:collapse;font-size:14px}
    th,td{text-align:left;padding:8px 6px;border-bottom:1px solid #e6edf3;vertical-align:top}
    th{color:#5c6b7a;font-weight:600}
    .empty{color:#6b7785}
    details{margin-top:6px}
    summary{cursor:pointer;color:#1d5f8a}
    @media(max-width:800px){.cards{grid-template-columns:1fr}}
  </style>
</head>
<body>
<header>
  <h1>精密空调记录</h1>
  <p>温湿度现状与事件记录。请在机器人所在的局域网内用笔记本打开本页。</p>
</header>
<nav>
  <a href="#live">任务状态</a>
  <a href="#latest">最新报告</a>
  <a href="#climate">温湿度历史</a>
  <a href="#events">事件记录</a>
  <a href="#screens">每次操作后的屏幕</a>
</nav>
<main>
  <section id="live"><h2>任务状态</h2><div id="liveBody" class="empty">正在读取…</div></section>
  <div class="cards" id="latest">
    <section><h2>最新温湿度</h2><div id="climateNow" class="empty">还没有温湿度报告。</div></section>
    <section><h2>最新事件记录</h2><div id="eventsNow" class="empty">还没有事件记录。</div></section>
  </div>
  <section id="climate"><h2>温湿度历史</h2><div id="climateList"></div></section>
  <section id="events"><h2>事件记录</h2><div id="eventList"></div></section>
  <section id="screens"><h2>每次操作后的屏幕</h2><div id="screenList"></div></section>
  <section id="tasks"><h2>任务记录</h2><div id="taskList"></div></section>
</main>
<script>
function el(tag, text){
  const node=document.createElement(tag);
  if(text!=null) node.textContent=text;
  return node;
}
function shot(item){
  if(!item.image) return el('div');
  const link=document.createElement('a');
  link.href=item.image;
  link.target='_blank';
  const img=document.createElement('img');
  img.src=item.image;
  img.alt=item.page_name||'屏幕';
  link.appendChild(img);
  return link;
}
function reportBlock(item){
  const box=el('div');
  const title=el('strong', (item.captured_text||'')+'  '+(item.label||'')+'  '+(item.page_name||''));
  box.appendChild(title);
  if(item.selection_text) box.appendChild(el('div', '黑块 '+item.selection_text));
  if(item.report_text){
    const pre=el('pre', item.report_text);
    box.appendChild(pre);
  }
  if(item.ocr_text){
    const details=el('details');
    details.appendChild(el('summary', '识别原文'));
    details.appendChild(el('pre', item.ocr_text));
    box.appendChild(details);
  }
  return box;
}
function climateCard(item){
  const wrap=el('div', '');
  wrap.className='shot';
  const report=item.report||{};
  const text=el('div');
  if(report.temperature_text){
    text.appendChild(el('p', report.temperature_text+' ℃'));
    text.lastChild.className='big';
  }
  if(report.humidity_text){
    text.appendChild(el('p', '湿度 '+report.humidity_text+' %'));
    text.lastChild.className='sub';
  }
  text.appendChild(reportBlock(item));
  wrap.appendChild(text);
  wrap.appendChild(shot(item));
  return wrap;
}
function fillLatest(id, item, builder, emptyText){
  const host=document.getElementById(id);
  host.innerHTML='';
  host.className=item?'':'empty';
  host.appendChild(item?builder(item):el('div', emptyText));
}
function table(columns, rows){
  const table=el('table');
  const head=el('tr');
  columns.forEach(function(name){ head.appendChild(el('th', name)); });
  table.appendChild(head);
  rows.forEach(function(cells){
    const tr=el('tr');
    cells.forEach(function(cell){
      const td=el('td');
      if(cell instanceof Node) td.appendChild(cell);
      else td.textContent=cell==null?'':String(cell);
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  return table;
}
function render(data){
  const live=data.live||{};
  const host=document.getElementById('liveBody');
  host.className='status';
  host.innerHTML='';
  const running=!!(live.sequence_run||live.running);
  const index=Number(live.sequence_index);
  const total=Number(live.sequence_total||0);
  const progress=index>=0&&total?('第 '+(index+1)+' 步 / 共 '+total+' 步'):'无';
  const fields=[
    ['组合', running?'进行中':'未在运行'],
    ['进度', progress],
    ['阶段', live.sequence_stage||live.control_stage||'无'],
    ['说明', live.message||'无'],
    ['屏幕判断', live.screen_progress?'开':'关'],
    ['最近一次读屏', live.screen_report||'无']
  ];
  fields.forEach(function(pair){
    const card=el('div');
    card.appendChild(el('span', pair[0]));
    card.appendChild(el('div', pair[1]));
    host.appendChild(card);
  });
  const history=Array.isArray(live.screen_history)?live.screen_history:[];
  if(history.length){
    const card=el('div');
    card.appendChild(el('span', '本次读屏'));
    card.appendChild(el('pre', history.join('\\n')));
    host.appendChild(card);
  }
  const climate=(data.climate||[]);
  const events=(data.events||[]);
  fillLatest('climateNow', climate.length?climate[0]:null, climateCard, '还没有温湿度报告。');
  fillLatest('eventsNow', events.length?events[0]:null, function(item){
    const wrap=el('div');
    wrap.className='shot';
    wrap.appendChild(reportBlock(item));
    wrap.appendChild(shot(item));
    return wrap;
  }, '还没有事件记录。');
  const climateHost=document.getElementById('climateList');
  climateHost.innerHTML='';
  if(!climate.length) climateHost.appendChild(el('p', '还没有温湿度历史。到达温湿度现状页后，这里会留下温度、湿度和当时的画面。'));
  else climateHost.appendChild(table(['时间','温度','湿度','画面'], climate.map(function(item){
    const report=item.report||{};
    return [item.captured_text, report.temperature_text?report.temperature_text+' ℃':'—', report.humidity_text?report.humidity_text+' %':'—', shot(item)];
  })));
  const eventHost=document.getElementById('eventList');
  eventHost.innerHTML='';
  if(!events.length) eventHost.appendChild(el('p', '还没有事件记录。翻到事件记录页后，每一页会单独保存。'));
  else events.forEach(function(item){
    const block=el('div');
    block.className='shot';
    block.appendChild(reportBlock(item));
    block.appendChild(shot(item));
    eventHost.appendChild(block);
  });
  const screenHost=document.getElementById('screenList');
  screenHost.innerHTML='';
  const screens=data.screens||[];
  if(!screens.length){
    screenHost.appendChild(el('p', '还没有屏幕读取。操作页打开「读取屏幕判断进度」后，每次按键前后的画面会记在这里。'));
  }else{
    screens.forEach(function(item){
      const block=el('div');
      block.className='shot';
      block.style.marginBottom='14px';
      block.appendChild(reportBlock(item));
      block.appendChild(shot(item));
      screenHost.appendChild(block);
    });
  }
  const taskHost=document.getElementById('taskList');
  taskHost.innerHTML='';
  const tasks=data.tasks||[];
  if(!tasks.length) taskHost.appendChild(el('p', '还没有任务。'));
  else taskHost.appendChild(table(['开始','结束','状态','说明'], tasks.map(function(item){
    return [item.started_text, item.finished_text||'未结束', item.state_text, item.message||item.stage||''];
  })));
}
async function refresh(){
  try{
    const res=await fetch('/api/overview', {cache:'no-store'});
    const data=await res.json();
    render(data);
  }catch(err){
    const host=document.getElementById('liveBody');
    host.className='empty';
    host.textContent='记录页暂时没有连上。';
  }
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


def build_records_app(records: PanelRecords, status_fn: StatusFn | None = None) -> Flask:
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False

    def live() -> dict[str, Any]:
        if not callable(status_fn):
            return {}
        try:
            return dict(status_fn() or {})
        except Exception:
            return {}

    @app.get("/")
    def index() -> Response:
        response = make_response(PAGE_HTML)
        response.headers["Content-Type"] = "text/html; charset=utf-8"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/overview")
    def overview() -> Response:
        payload = records.overview(live())
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/media/<int:row_id>.jpg")
    def media(row_id: int) -> Response:
        blob = records.image_bytes(row_id)
        if not blob:
            return make_response("no image", 404)
        response = make_response(blob)
        response.headers["Content-Type"] = "image/jpeg"
        response.headers["Cache-Control"] = "public, max-age=86400"
        return response

    return app
