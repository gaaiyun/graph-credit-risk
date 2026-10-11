/* 关联风险全景 · AI 风控助手（网页组件）
 * 在浏览器里检索：企业名称索引 → 企业风险画像分片；方法知识库 → 最相关段落。
 * 检索结果作为“资料”连同问题发给 Cloudflare Worker，由 Workers AI 流式作答。
 * 用法：<script src="assets/assistant.js" data-base="./" defer></script>，data-base 指向站点根目录。
 */
(function () {
  const API = 'https://graph-risk-assistant.gaaiyun-risk-selfcheck.workers.dev/chat';
  const me = document.currentScript;
  const BASE = (me && me.dataset.base) || './';
  const DATA = BASE + 'assistant/';
  const SUGGEST = ['海航控股的风险情况', '什么算“出险”？', '事件乘数怎么计算？', '多跳传导公式是什么？', '预警名单怎么分级？', '报告的核心结论是什么？'];
  const METHOD_WORDS = /怎么|如何|什么|公式|原理|计算|定义|口径|方法|模型|指标|乘数|率比|分级|归因|SHAP|AUC|传导|衰减|标签|出险|违约|结论|数据/;

  const css = `
  .ra-launch{position:fixed;right:28px;bottom:28px;z-index:9998;display:flex;align-items:center;gap:10px;height:52px;padding:0 22px 0 16px;border:0;border-radius:999px;cursor:pointer;
    color:#fff;font:600 15px/1 "Manrope","PingFang SC","Microsoft YaHei",sans-serif;letter-spacing:.02em;background:linear-gradient(135deg,#2E6BFF,#6A4DFF);
    box-shadow:0 14px 36px rgba(46,107,255,.45),inset 0 1px 0 rgba(255,255,255,.25);transition:transform .25s cubic-bezier(.2,.7,.2,1),box-shadow .25s}
  .ra-launch:hover{transform:translateY(-2px);box-shadow:0 18px 44px rgba(46,107,255,.55),inset 0 1px 0 rgba(255,255,255,.25)}
  .ra-launch .dot{position:relative;width:26px;height:26px;border-radius:50%;background:rgba(255,255,255,.18);display:flex;align-items:center;justify-content:center}
  .ra-launch .dot::after{content:"";position:absolute;inset:-4px;border-radius:50%;border:2px solid rgba(255,255,255,.45);animation:ra-ping 2.2s ease-out infinite}
  @keyframes ra-ping{0%{transform:scale(.8);opacity:.9}100%{transform:scale(1.5);opacity:0}}
  .ra-launch.hide{opacity:0;pointer-events:none;transform:translateY(12px)}
  .ra-panel{position:fixed;right:28px;bottom:28px;z-index:9999;width:min(440px,calc(100vw - 32px));height:min(680px,calc(100vh - 56px));display:flex;flex-direction:column;
    border-radius:18px;overflow:hidden;color:#E6ECF5;font:14.5px/1.75 "Manrope","PingFang SC","Microsoft YaHei",sans-serif;
    background:rgba(9,16,32,.86);backdrop-filter:blur(22px) saturate(150%);-webkit-backdrop-filter:blur(22px) saturate(150%);
    border:1px solid rgba(255,255,255,.10);box-shadow:0 30px 80px rgba(2,6,23,.55);transform-origin:bottom right;
    opacity:0;transform:translateY(16px) scale(.98);pointer-events:none;transition:opacity .3s cubic-bezier(.2,.7,.2,1),transform .3s cubic-bezier(.2,.7,.2,1)}
  .ra-panel.on{opacity:1;transform:none;pointer-events:auto}
  .ra-panel::before{content:"";position:absolute;left:0;right:0;top:0;height:2px;background:linear-gradient(90deg,#2E6BFF,#7E57FF,#FFB020)}
  .ra-head{padding:18px 20px 14px;border-bottom:1px solid rgba(255,255,255,.08);display:flex;gap:12px;align-items:flex-start}
  .ra-head .ic{flex:none;width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#2E6BFF,#6A4DFF);display:flex;align-items:center;justify-content:center}
  .ra-head b{display:block;font-size:16px;color:#fff;letter-spacing:.02em}
  .ra-head span{display:block;font-size:12.5px;color:#8D9DBB;line-height:1.6;margin-top:2px}
  .ra-head button{margin-left:auto;background:none;border:0;color:#8D9DBB;font-size:22px;line-height:1;cursor:pointer;padding:2px 4px}
  .ra-head button:hover{color:#fff}
  .ra-log{flex:1;overflow-y:auto;padding:18px 20px;scroll-behavior:smooth}
  .ra-log::-webkit-scrollbar{width:6px}.ra-log::-webkit-scrollbar-thumb{background:rgba(255,255,255,.12);border-radius:3px}
  .ra-msg{margin:0 0 16px;animation:ra-in .35s cubic-bezier(.2,.7,.2,1)}
  @keyframes ra-in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
  .ra-msg.user{display:flex;justify-content:flex-end}
  .ra-msg.user div{max-width:85%;background:linear-gradient(135deg,#2E6BFF,#4E5BFF);color:#fff;padding:9px 14px;border-radius:14px 14px 4px 14px}
  .ra-msg.bot .bd{color:#D9E2F1}
  .ra-msg.bot .bd b{color:#fff}
  .ra-msg.bot .bd ul{margin:6px 0;padding-left:18px}
  .ra-msg.bot .bd li{margin:3px 0}
  .ra-msg.bot .bd p{margin:6px 0}
  .ra-msg.bot .bd code{background:rgba(255,255,255,.08);padding:1px 6px;border-radius:4px;font-size:13px}
  .ra-src{margin-top:8px;font-size:12px;color:#6F80A0}
  .ra-src i{font-style:normal;display:inline-block;border:1px solid rgba(255,255,255,.1);border-radius:999px;padding:0 8px;margin:4px 6px 0 0}
  .ra-intro{font-size:13.5px;color:#AFC0DC;margin-bottom:12px}
  .ra-intro b{color:#fff}
  .ra-intro ul{margin:8px 0 0;padding-left:18px}
  .ra-chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
  .ra-chips button{border:1px solid rgba(255,255,255,.14);background:rgba(255,255,255,.04);color:#D9E2F1;border-radius:999px;padding:5px 12px;font:inherit;font-size:13px;cursor:pointer;transition:all .2s}
  .ra-chips button:hover{border-color:#6E9CF5;background:rgba(46,107,255,.16);color:#fff}
  .ra-typing span{display:inline-block;width:6px;height:6px;margin-right:4px;border-radius:50%;background:#6E9CF5;animation:ra-bounce 1s infinite}
  .ra-typing span:nth-child(2){animation-delay:.15s}.ra-typing span:nth-child(3){animation-delay:.3s}
  @keyframes ra-bounce{0%,80%,100%{opacity:.25;transform:translateY(0)}40%{opacity:1;transform:translateY(-4px)}}
  .ra-foot{padding:12px 14px 14px;border-top:1px solid rgba(255,255,255,.08)}
  .ra-box{display:flex;gap:8px;align-items:flex-end;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);border-radius:14px;padding:8px 8px 8px 14px;transition:border-color .2s}
  .ra-box:focus-within{border-color:#6E9CF5}
  .ra-box textarea{flex:1;resize:none;border:0;outline:0;background:transparent;color:#fff;font:inherit;max-height:120px;min-height:24px;padding:2px 0}
  .ra-box textarea::placeholder{color:#6F80A0}
  .ra-box button{flex:none;width:36px;height:36px;border:0;border-radius:10px;cursor:pointer;background:linear-gradient(135deg,#2E6BFF,#6A4DFF);color:#fff;display:flex;align-items:center;justify-content:center}
  .ra-box button:disabled{opacity:.4;cursor:default}
  .ra-note{font-size:11.5px;color:#5E6E8C;margin-top:8px;text-align:center}
  @media (max-width:560px){.ra-panel{right:16px;bottom:16px}.ra-launch{right:16px;bottom:16px}}
  @media (prefers-reduced-motion:reduce){.ra-launch .dot::after,.ra-msg{animation:none}}`;
  const ICON = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/><path d="M8.5 11h.01M12 11h.01M15.5 11h.01"/></svg>';

  let kb = null, idx = null, shards = {}, history = [], busy = false;
  const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);
  const launch = document.createElement('button');
  launch.className = 'ra-launch'; launch.type = 'button'; launch.innerHTML = `<span class="dot">${ICON}</span>AI 风控助手`;
  const panel = document.createElement('section');
  panel.className = 'ra-panel'; panel.setAttribute('aria-label', 'AI 风控助手');
  panel.innerHTML = `<div class="ra-head"><div class="ic">${ICON}</div><div><b>AI 风控助手</b><span>可问企业风险情况、风险监测方法、测度口径与计算公式原理、研究报告内容</span></div><button type="button" aria-label="关闭">×</button></div>
    <div class="ra-log"></div>
    <div class="ra-foot"><div class="ra-box"><textarea rows="1" placeholder="输入公司全称、简称或股票代码，或提问方法与公式"></textarea><button type="button" aria-label="发送"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg></button></div>
    <div class="ra-note">回答基于本研究的图谱数据与方法文档 · Cloudflare Workers AI</div></div>`;
  document.body.appendChild(launch); document.body.appendChild(panel);
  const log = panel.querySelector('.ra-log'), ta = panel.querySelector('textarea'), send = panel.querySelector('.ra-box button');

  function intro() {
    const d = document.createElement('div'); d.className = 'ra-msg bot';
    d.innerHTML = `<div class="ra-intro">你好，我是关联风险全景的 <b>AI 风控助手</b>。我可以回答：<ul><li><b>企业风险情况</b>：出险与违约记录、样本外违约概率、公告风险事件、股权质押、票据名单与有风险的关联方</li><li><b>风险监测方法</b>：关联事件预警、预警名单、分级复核与风险归因</li><li><b>测度口径与公式原理</b>：违约标签、时间衰减、多跳传导、事件乘数、Mantel-Haenszel 率比、SHAP</li><li><b>研究报告内容</b>：数据来源、核心结论与产品用法</li></ul></div><div class="ra-chips"></div>`;
    const chips = d.querySelector('.ra-chips');
    SUGGEST.forEach(q => { const b = document.createElement('button'); b.type = 'button'; b.textContent = q; b.onclick = () => ask(q); chips.appendChild(b); });
    log.appendChild(d);
  }
  function open(q) { panel.classList.add('on'); launch.classList.add('hide'); if (!log.children.length) intro(); setTimeout(() => ta.focus(), 250); if (q) ask(q); }
  function close() { panel.classList.remove('on'); launch.classList.remove('hide'); }
  launch.onclick = () => open();
  panel.querySelector('.ra-head button').onclick = close;
  window.RiskAssistant = {open};

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
