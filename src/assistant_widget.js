/* 关联风险全景 · AI 风控助手（网页组件）
 * 在浏览器里检索：企业名称索引 → 企业风险画像分片；方法知识库 → 最相关段落。
 * 检索结果作为“资料”连同问题发给 Cloudflare Worker，由 Workers AI 流式作答。
 * 用法：<script src="assets/assistant.js" data-base="./" data-auto="1" defer></script>
 *   data-base 指向站点根目录；data-auto="1" 表示宽屏首次访问时自动在右侧停靠展开。
 * 面板两种形态：停靠（贴右侧整高、页面整体左移，拖左边缘调宽）与浮窗（右下角，拖左上角调宽高），
 * 均可一键最小化为右下角按钮；形态、尺寸与开合状态记在本机。
 */
(function () {
  const API = 'https://graph-risk-assistant.gaaiyun-risk-selfcheck.workers.dev/chat';
  const me = document.currentScript;
  const BASE = (me && me.dataset.base) || './';
  const AUTO = !!(me && me.dataset.auto === '1');
  const DATA = BASE + 'assistant/';
  const KEY = 'ra-state-v2';
  const SUGGEST = ['海航控股的风险情况', '什么算“出险”？', '事件乘数怎么计算？', '多跳传导公式是什么？', '预警名单怎么分级？', '报告的核心结论是什么？'];
  const METHOD_WORDS = /怎么|如何|什么|公式|原理|计算|定义|口径|方法|模型|指标|乘数|率比|分级|归因|SHAP|AUC|传导|衰减|标签|出险|违约|结论|数据/;
  const FONT = '-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI","PingFang SC","Microsoft YaHei",sans-serif';

  const css = `
  :root{--ra-dock:0px}
  html.ra-docked body{margin-right:var(--ra-dock)!important}
  html.ra-docked [data-ra-shift]{right:var(--ra-dock)!important}
  .ra-launch{position:fixed;right:24px;bottom:24px;z-index:9998;display:flex;align-items:center;gap:12px;padding:8px 22px 8px 8px;border:0;border-radius:999px;cursor:pointer;text-align:left;
    color:#fff;font:600 15px/1.2 ${FONT};background:#0071E3;box-shadow:0 10px 30px rgba(0,113,227,.35),0 2px 6px rgba(0,0,0,.08);
    transition:transform .25s cubic-bezier(.2,.7,.2,1),box-shadow .25s,opacity .25s}
  .ra-launch:hover{transform:translateY(-2px);box-shadow:0 14px 36px rgba(0,113,227,.45),0 2px 6px rgba(0,0,0,.08)}
  .ra-launch .dot{position:relative;flex:none;width:38px;height:38px;border-radius:50%;background:rgba(255,255,255,.2);display:flex;align-items:center;justify-content:center}
  .ra-launch .dot::after{content:"";position:absolute;inset:-3px;border-radius:50%;border:2px solid rgba(255,255,255,.55);animation:ra-ping 2.4s ease-out infinite}
  .ra-launch small{display:block;font-size:12px;font-weight:400;opacity:.85;margin-top:3px}
  @keyframes ra-ping{0%{transform:scale(.85);opacity:.9}100%{transform:scale(1.45);opacity:0}}
  .ra-launch.hide{opacity:0;pointer-events:none;transform:translateY(12px)}
  .ra-panel{position:fixed;z-index:9999;display:flex;flex-direction:column;color:#1D1D1F;font:14.5px/1.75 ${FONT};
    background:rgba(255,255,255,.95);backdrop-filter:saturate(180%) blur(30px);-webkit-backdrop-filter:saturate(180%) blur(30px);
    opacity:0;pointer-events:none;transition:opacity .3s cubic-bezier(.2,.7,.2,1),transform .35s cubic-bezier(.2,.7,.2,1)}
  .ra-panel.dock{top:0;right:0;bottom:0;width:var(--ra-w,420px);border-left:1px solid #E5E5EA;box-shadow:-12px 0 40px rgba(0,0,0,.06);transform:translateX(24px)}
  .ra-panel.float{right:24px;bottom:24px;width:var(--ra-w,420px);height:var(--ra-h,640px);max-height:calc(100vh - 48px);max-width:calc(100vw - 32px);border-radius:18px;
    border:1px solid rgba(0,0,0,.08);box-shadow:0 24px 80px rgba(0,0,0,.18),0 2px 8px rgba(0,0,0,.06);transform:translateY(16px) scale(.98);transform-origin:bottom right;overflow:hidden}
  .ra-panel.on{opacity:1;pointer-events:auto;transform:none}
  .ra-grip{position:absolute;z-index:2}
  .ra-panel.dock .ra-grip{left:-4px;top:0;bottom:0;width:8px;cursor:ew-resize}
  .ra-panel.float .ra-grip{left:0;top:0;width:20px;height:20px;cursor:nwse-resize}
  .ra-panel.float .ra-grip::before{content:"";position:absolute;left:6px;top:6px;width:8px;height:8px;border-left:2px solid #C7C7CC;border-top:2px solid #C7C7CC;border-radius:2px 0 0 0}
  .ra-panel.dock .ra-grip:hover::after,.ra-panel.dock.drag .ra-grip::after{content:"";position:absolute;left:3px;top:0;bottom:0;width:2px;background:#0071E3}
  .ra-panel.drag{transition:none;user-select:none}
  .ra-head{padding:14px 12px 12px 20px;border-bottom:1px solid #E5E5EA;display:flex;gap:12px;align-items:center}
  .ra-head .ic{flex:none;width:34px;height:34px;border-radius:50%;background:#0071E3;display:flex;align-items:center;justify-content:center}
  .ra-head .tt{min-width:0;flex:1}
  .ra-head b{display:block;font-size:16px;font-weight:600;letter-spacing:-.01em}
  .ra-head span{display:block;font-size:12px;color:#86868B;line-height:1.5;margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .ra-tools{display:flex;gap:2px;flex:none}
  .ra-tools button{width:32px;height:32px;border:0;border-radius:8px;background:transparent;color:#6E6E73;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background .15s,color .15s}
  .ra-tools button:hover{background:#F0F0F2;color:#1D1D1F}
  .ra-log{flex:1;overflow-y:auto;padding:18px 20px;scroll-behavior:smooth}
  .ra-log::-webkit-scrollbar{width:6px}.ra-log::-webkit-scrollbar-thumb{background:#D2D2D7;border-radius:3px}
  .ra-msg{margin:0 0 16px;animation:ra-in .35s cubic-bezier(.2,.7,.2,1)}
  @keyframes ra-in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
  .ra-msg.user{display:flex;justify-content:flex-end}
  .ra-msg.user div{max-width:85%;background:#0071E3;color:#fff;padding:8px 14px;border-radius:18px 18px 4px 18px}
  .ra-msg.bot .bd b{font-weight:600}
  .ra-msg.bot .bd ul{margin:6px 0;padding-left:18px}
  .ra-msg.bot .bd li{margin:3px 0}
  .ra-msg.bot .bd p{margin:6px 0}
  .ra-msg.bot .bd code{background:#F5F5F7;padding:1px 6px;border-radius:5px;font-size:13px}
  .ra-src{margin-top:8px;font-size:12px;color:#86868B}
  .ra-src i{font-style:normal;display:inline-block;background:#F5F5F7;border-radius:999px;padding:0 9px;margin:4px 6px 0 0}
  .ra-intro{font-size:14px;color:#424245}
  .ra-intro h4{margin:0 0 6px;font-size:21px;font-weight:600;letter-spacing:-.01em;color:#1D1D1F}
  .ra-intro ul{margin:10px 0 0;padding:0;list-style:none}
  .ra-intro li{padding:9px 0;border-top:1px solid #F0F0F2;font-size:13.5px;line-height:1.65}
  .ra-intro li b{display:block;color:#1D1D1F;font-weight:600;font-size:14px}
  .ra-chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
  .ra-chips button{border:0;background:#F5F5F7;color:#1D1D1F;border-radius:999px;padding:6px 14px;font:inherit;font-size:13px;cursor:pointer;transition:background .15s,color .15s}
  .ra-chips button:hover{background:#E8F1FD;color:#0071E3}
  .ra-typing span{display:inline-block;width:6px;height:6px;margin-right:4px;border-radius:50%;background:#0071E3;animation:ra-bounce 1s infinite}
  .ra-typing span:nth-child(2){animation-delay:.15s}.ra-typing span:nth-child(3){animation-delay:.3s}
  @keyframes ra-bounce{0%,80%,100%{opacity:.25;transform:translateY(0)}40%{opacity:1;transform:translateY(-4px)}}
  .ra-foot{padding:10px 14px 14px;border-top:1px solid #E5E5EA}
  .ra-box{display:flex;gap:8px;align-items:flex-end;background:#F5F5F7;border:1px solid transparent;border-radius:22px;padding:6px 6px 6px 16px;transition:border-color .2s,background .2s}
  .ra-box:focus-within{border-color:#0071E3;background:#fff}
  .ra-box textarea{flex:1;resize:none;border:0;outline:0;background:transparent;color:#1D1D1F;font:inherit;max-height:120px;min-height:24px;padding:4px 0}
  .ra-box textarea::placeholder{color:#86868B}
  .ra-box button{flex:none;width:32px;height:32px;border:0;border-radius:50%;cursor:pointer;background:#0071E3;color:#fff;display:flex;align-items:center;justify-content:center}
  .ra-box button:disabled{opacity:.35;cursor:default}
  .ra-note{font-size:11.5px;color:#86868B;margin-top:8px;text-align:center}
  @media (max-width:760px){.ra-panel.dock,.ra-panel.float{top:auto;left:12px;right:12px;bottom:12px;width:auto;height:min(78vh,640px);border-radius:18px;border:1px solid rgba(0,0,0,.08)}
    .ra-grip,.ra-mode{display:none!important}.ra-launch{right:16px;bottom:16px}}
  @media (prefers-reduced-motion:reduce){.ra-launch .dot::after,.ra-msg{animation:none}}`;
  const ICON = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/><path d="M8.5 11h.01M12 11h.01M15.5 11h.01"/></svg>';
  const I_FLOAT = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="4" width="18" height="16" rx="3"/><rect x="11" y="11" width="7" height="6" rx="1.5"/></svg>';
  const I_DOCK = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="4" width="18" height="16" rx="3"/><line x1="14" y1="4" x2="14" y2="20"/></svg>';
  const I_MIN = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><line x1="6" y1="12" x2="18" y2="12"/></svg>';

  let kb = null, idx = null, shards = {}, history = [], busy = false;
  const S = {open: null, dock: true, w: 420, h: 640};
  try { Object.assign(S, JSON.parse(localStorage.getItem(KEY) || '{}')); } catch (e) {}
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch (e) {} };
  const narrow = () => innerWidth <= 760;

  const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);
  const launch = document.createElement('button');
  launch.className = 'ra-launch'; launch.type = 'button';
  launch.innerHTML = `<span class="dot">${ICON}</span><span>AI 风控助手<small>问企业风险、方法与公式</small></span>`;
  const panel = document.createElement('section');
  panel.setAttribute('aria-label', 'AI 风控助手');
  panel.innerHTML = `<div class="ra-grip" title="拖动调整大小"></div>
    <div class="ra-head"><div class="ic">${ICON}</div><div class="tt"><b>AI 风控助手</b><span>企业风险 · 监测方法 · 测度口径与公式 · 报告内容</span></div>
      <div class="ra-tools"><button type="button" class="ra-mode"></button><button type="button" class="ra-min" title="最小化" aria-label="最小化">${I_MIN}</button></div></div>
    <div class="ra-log"></div>
    <div class="ra-foot"><div class="ra-box"><textarea rows="1" placeholder="输入公司全称、简称或股票代码，或提问方法与公式"></textarea><button type="button" aria-label="发送"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M6 11l6-6 6 6"/></svg></button></div>
    <div class="ra-note">模型：智谱 GLM-4.7-Flash · 基于本研究的图谱数据与方法文档作答</div></div>`;
  document.body.appendChild(launch); document.body.appendChild(panel);
  const log = panel.querySelector('.ra-log'), ta = panel.querySelector('textarea'), send = panel.querySelector('.ra-box button'), modeBtn = panel.querySelector('.ra-mode');

  function layout() {
    const isOpen = panel.classList.contains('on'), dock = S.dock && !narrow();
    panel.className = 'ra-panel ' + (dock ? 'dock' : 'float') + (isOpen ? ' on' : '');
    panel.style.setProperty('--ra-w', S.w + 'px'); panel.style.setProperty('--ra-h', S.h + 'px');
    modeBtn.innerHTML = S.dock ? I_FLOAT : I_DOCK;
    modeBtn.title = S.dock ? '改为浮窗' : '停靠到右侧'; modeBtn.setAttribute('aria-label', modeBtn.title);
    const docked = isOpen && dock;
    document.documentElement.classList.toggle('ra-docked', docked);
    document.documentElement.style.setProperty('--ra-dock', docked ? S.w + 'px' : '0px');
    setTimeout(() => dispatchEvent(new Event('resize')), 60);
  }
  function intro() {
    const d = document.createElement('div'); d.className = 'ra-msg bot';
    d.innerHTML = `<div class="ra-intro"><h4>有什么想问的？</h4>我检索本研究 1.6 万个主体的风险画像与方法文档，回答四类问题：<ul>
      <li><b>企业风险情况</b>输入全称、简称或股票代码：出险与违约记录、样本外违约概率、公告风险事件、股权质押、票据名单与有风险的关联方</li>
      <li><b>风险监测方法</b>关联事件预警、预警名单、分级复核与风险归因怎么做</li>
      <li><b>测度口径与公式原理</b>违约标签、时间衰减、多跳传导、事件乘数、Mantel-Haenszel 率比、SHAP</li>
      <li><b>研究报告内容</b>数据来源、核心结论与产品用法</li></ul></div><div class="ra-chips"></div>`;
    const chips = d.querySelector('.ra-chips');
    SUGGEST.forEach(q => { const b = document.createElement('button'); b.type = 'button'; b.textContent = q; b.onclick = () => ask(q); chips.appendChild(b); });
    log.appendChild(d);
  }
  function open(q, focus) {
    panel.classList.add('on'); launch.classList.add('hide'); S.open = true; save(); layout();
    if (!log.children.length) intro();
    if (focus !== false) setTimeout(() => ta.focus({preventScroll: true}), 250);
    if (q) ask(q);
  }
  function minimize() { panel.classList.remove('on'); launch.classList.remove('hide'); S.open = false; save(); layout(); }
  launch.onclick = () => open();
  panel.querySelector('.ra-min').onclick = minimize;
  modeBtn.onclick = () => { S.dock = !S.dock; save(); layout(); };
  addEventListener('keydown', e => { if (e.key === 'Escape' && panel.classList.contains('on') && panel.classList.contains('float')) minimize(); });
  // 拖动调整大小：停靠时拖左边缘调宽；浮窗时拖左上角同时调宽高
  panel.querySelector('.ra-grip').addEventListener('pointerdown', e => {
    e.preventDefault(); const x0 = e.clientX, y0 = e.clientY, w0 = S.w, h0 = S.h, dock = panel.classList.contains('dock');
    panel.classList.add('drag');
    const mv = ev => {
      S.w = Math.round(Math.max(340, Math.min(innerWidth * .6, w0 + x0 - ev.clientX)));
      if (!dock) S.h = Math.round(Math.max(380, Math.min(innerHeight - 48, h0 + y0 - ev.clientY)));
      panel.style.setProperty('--ra-w', S.w + 'px'); panel.style.setProperty('--ra-h', S.h + 'px');
      if (dock) document.documentElement.style.setProperty('--ra-dock', S.w + 'px');
    };
    const up = () => { removeEventListener('pointermove', mv); removeEventListener('pointerup', up); panel.classList.remove('drag'); save(); dispatchEvent(new Event('resize')); };
    addEventListener('pointermove', mv); addEventListener('pointerup', up);
  });
  window.RiskAssistant = {open, minimize};
  layout();
  if (S.open === true || (S.open === null && AUTO && innerWidth >= 1280)) open('', false);

  const esc = s => s.replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  function md(s) {
    const lines = esc(s).replace(/\$+/g, '').split('\n'); let out = '', inList = false;
    for (let ln of lines) {
      ln = ln.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`([^`]+)`/g, '<code>$1</code>').replace(/^#{1,4}\s*(.+)$/, '<b>$1</b>');
      const m = ln.match(/^\s*(?:[-*•]|\d+[.、])\s+(.*)$/);
      if (m) { if (!inList) { out += '<ul>'; inList = true; } out += `<li>${m[1]}</li>`; continue; }
      if (inList) { out += '</ul>'; inList = false; }
      if (ln.trim()) out += `<p>${ln}</p>`;
    }
    return out + (inList ? '</ul>' : '');
  }
  const bigrams = s => { s = s.replace(/\s+/g, ''); const o = new Set(); for (let i = 0; i < s.length - 1; i++) o.add(s.slice(i, i + 2)); return o; };
  async function load() {
    if (!kb) kb = await (await fetch(DATA + 'kb.json')).json();
    if (!idx) idx = await (await fetch(DATA + 'names.json')).json();
  }
  async function retrieve(q) {
    await load();
    const codes = new Set(q.match(/\d{6}/g) || []), hits = [];
    for (const [i, full, short, code] of idx.items) {
      let score = 0;
      if (code && codes.has(code)) score = 100;
      else if (full && full.length >= 4 && q.includes(full)) score = 50 + full.length;
      else if (short && short.length >= 2 && q.includes(short)) score = 20 + short.length;
      if (score) hits.push([score, i]);
    }
    hits.sort((a, b) => b[0] - a[0]);
    const parts = [], src = [];
    for (const [, i] of hits.slice(0, 2)) {
      const k = i % idx.shards;
      if (!shards[k]) shards[k] = await (await fetch(`${DATA}p/${k}.json`)).json();
      if (shards[k][i]) { parts.push('## 企业风险画像\n' + shards[k][i]); src.push('企业风险画像'); }
    }
    const qb = bigrams(q);
    const ranked = kb.map(r => { let s = 0; const tb = bigrams(r.title), xb = bigrams(r.text); qb.forEach(g => { if (tb.has(g)) s += 3; if (xb.has(g)) s += 1; }); return [s, r]; })
      .filter(x => x[0] > 0).sort((a, b) => b[0] - a[0]);
    const nKb = parts.length ? (METHOD_WORDS.test(q.replace(/[^一-龥A-Za-z]/g, '')) && hits.length === 0 ? 3 : 1) : 4;
    const tag = hits.length ? '，全样本方法资料，其中数字不是该企业的数值' : '';
    for (const [, r] of ranked.slice(0, nKb)) { parts.push(`## ${r.title}（${r.src}${tag}）\n${r.text}`); src.push(`${r.src}｜${r.title}`); }
    return {context: parts.join('\n\n').slice(0, 14000), src: [...new Set(src)]};
  }
  function bubble(role, html) { const d = document.createElement('div'); d.className = 'ra-msg ' + role; d.innerHTML = role === 'user' ? `<div>${html}</div>` : `<div class="bd">${html}</div>`; log.appendChild(d); log.scrollTop = log.scrollHeight; return d; }
  async function ask(q) {
    q = (q || ta.value).trim(); if (!q || busy) return;
    busy = true; send.disabled = true; ta.value = ''; ta.style.height = '';
    bubble('user', esc(q));
    const bot = bubble('bot', '<span class="ra-typing"><span></span><span></span><span></span></span>'), bd = bot.querySelector('.bd');
    let text = '';
    try {
      const {context, src} = await retrieve(q);
      const res = await fetch(API, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question: q, context, history})});
      if (!res.ok) { let m = '服务暂时不可用，请稍后再试。'; try { m = (await res.json()).error || m; } catch (e) {} throw new Error(m); }
      const reader = res.body.getReader(), dec = new TextDecoder(); let buf = '';
      for (;;) {
        const {value, done} = await reader.read(); if (done) break;
        buf += dec.decode(value, {stream: true}); const lines = buf.split('\n'); buf = lines.pop();
        for (const ln of lines) {
          if (!ln.startsWith('data:')) continue; const p = ln.slice(5).trim(); if (!p || p === '[DONE]') continue;
          try { const j = JSON.parse(p); const c = (j.choices && j.choices[0] && j.choices[0].delta && j.choices[0].delta.content) || j.response || ''; if (c) { text += c; bd.innerHTML = md(text); log.scrollTop = log.scrollHeight; } } catch (e) {}
        }
      }
      if (!text) text = '没有得到回答，请换一种问法再试。';
      bd.innerHTML = md(text) + (src.length ? `<div class="ra-src">参考：${src.map(s => `<i>${esc(s)}</i>`).join('')}</div>` : '');
      history.push({role: 'user', content: q}, {role: 'assistant', content: text}); history = history.slice(-6);
    } catch (e) {
      bd.innerHTML = `<p>${esc(e.message || '网络异常，请稍后再试。')}</p>`;
    } finally { busy = false; send.disabled = false; log.scrollTop = log.scrollHeight; }
  }
  send.onclick = () => ask();
  ta.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(); } });
  ta.addEventListener('input', () => { ta.style.height = 'auto'; ta.style.height = Math.min(120, ta.scrollHeight) + 'px'; });
  document.querySelectorAll('[data-ask]').forEach(b => b.addEventListener('click', e => { e.preventDefault(); open(b.dataset.ask || ''); }));
})();
