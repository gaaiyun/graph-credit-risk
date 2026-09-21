"""把同一份内容块渲染成 docx（参赛提交）和 HTML（GitHub Pages）。
块类型：h2 / h3 / p / fig / table / callout / formula / svg(仅网页，docx 用替代图) / explorer(仅网页) / pagebreak(仅 docx)"""
import html
import json
import re
import shutil
from pathlib import Path
import pandas as pd
import docx_helpers as dh


def _md_inline(text):
    """**加粗** → <b>；其余转义"""
    t = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)


def _plain(text):
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text)


def to_docx(blocks, path, title, subtitle, abstract, keywords):
    doc = dh.new_doc()
    dh.heading(doc, title, 0)
    p = dh.para(doc, subtitle, size=11, indent=False, align="center", space_after=6)
    dh.para(doc, abstract, size=9.5, indent=False, bold_prefix="摘要：", space_after=2)
    dh.para(doc, keywords, size=9.5, indent=False, bold_prefix="关键词：", space_after=4)
    fig_n = tab_n = 0
    for b in blocks:
        k = b[0]
        if k == "h2":
            dh.heading(doc, f"{b[2]} {b[3]}", 1)
        elif k == "h3":
            dh.heading(doc, b[1], 2)
        elif k == "p":
            para_with_bold(doc, b[1])
        elif k == "formula":
            dh.para(doc, b[1], size=10, indent=False, align="center", space_after=3)
        elif k == "callout":
            para_with_bold(doc, f"**{b[1]}** {b[2]}")
        elif k in ("fig", "svg"):
            if k == "svg" and not b[3]:
                continue
            fig_n += 1
            path_png = b[1] if k == "fig" else b[3]
            dh.figure(doc, path_png, f"图{fig_n} {b[2]}", width_cm=b[4] if k == "fig" and len(b) > 4 and b[4] else 15.5,
                      note=(b[3] if k == "fig" and len(b) > 3 else None))
        elif k == "table":
            tab_n += 1
            df, cap = b[1], b[2]
            note = b[3] if len(b) > 3 else None
            widths = b[5] if len(b) > 5 else None
            if cap.startswith("附表"):
                tab_n -= 1
            dh.table(doc, df, caption=cap if cap.startswith("附表") else f"表{tab_n} {cap}", note=note, col_widths=widths)
        elif k == "pagebreak":
            dh.page_break(doc)
    doc.save(path)


def para_with_bold(doc, text, size=10.5):
    p = doc.add_paragraph()
    p.alignment = dh.WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = dh.Pt(size * 2)
    p.paragraph_format.space_after = dh.Pt(2)
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if not part:
            continue
        r = p.add_run(part)
        if i % 2 == 1:
            dh.set_font(r, cn=dh.HEAD_CN, size=size, bold=True)
        else:
            dh.set_font(r, size=size)
    return p


def to_html(blocks, template_path, out_dir, lede, kpis, footer, data, date):
    out_dir = Path(out_dir)
    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    parts, toc = [], []
    fig_n = tab_n = 0
    for b in blocks:
        k = b[0]
        if k == "h2":
            _, sid, num, text = b
            parts.append(f'<h2 id="{sid}"><span class="n">{num}</span>{html.escape(text)}</h2>')
            toc.append(f'<a href="#{sid}">{num} {html.escape(text)}</a>')
        elif k == "h3":
            sid = f"h{len(toc)}"
            parts.append(f'<h3 id="{sid}">{html.escape(b[1])}</h3>')
            toc.append(f'<a class="l2" href="#{sid}">{html.escape(b[1])}</a>')
        elif k == "p":
            parts.append(f"<p>{_md_inline(b[1])}</p>")
        elif k == "formula":
            parts.append(f'<p style="text-align:center;font-family:var(--mono);font-size:14px">{html.escape(b[1])}</p>')
        elif k == "callout":
            parts.append(f'<div class="callout"><div class="t">{html.escape(b[1])}</div>{_md_inline(b[2])}</div>')
        elif k == "fig":
            fig_n += 1
            src = Path(b[1])
            shutil.copy(src, assets / src.name)
            note = f'<br><span style="color:var(--muted)">{html.escape(b[3])}</span>' if len(b) > 3 and b[3] else ""
            parts.append(f'<figure><div class="card"><img src="assets/{src.name}" alt="{html.escape(b[2])}" loading="lazy"></div>'
                         f'<figcaption><b>图{fig_n}</b>　{html.escape(b[2])}{note}</figcaption></figure>')
        elif k == "svg":
            fig_n += 1
            parts.append(f'<figure><div class="svgchart"><div id="{b[1]}"></div></div><figcaption><b>图{fig_n}</b>　{html.escape(b[2])}</figcaption></figure>')
        elif k == "explorer":
            fig_n += 1
            parts.append('<figure><div class="explorer"><div class="tabs" id="exTabs"></div><div class="ex-body">'
                         '<div class="ex-graph" id="exGraph"></div><div class="ex-side" id="exSide"></div></div></div>'
                         f'<figcaption><b>图{fig_n}</b>　{html.escape(b[1])}</figcaption></figure>')
        elif k == "table":
            tab_n += 1
            df, cap = b[1], b[2]
            note = b[3] if len(b) > 3 and b[3] else ""
            hl = set(b[4]) if len(b) > 4 and b[4] else set()
            th = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
            rows = []
            for i, r in enumerate(df.itertuples(index=False)):
                cls = ' class="hl"' if i in hl else ""
                rows.append(f"<tr{cls}>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>")
            if cap.startswith("附表"):
                tab_n -= 1
            cap_txt = html.escape(cap) if cap.startswith("附表") else f"表{tab_n}　{html.escape(cap)}"
            parts.append(f'<div class="tcap">{cap_txt}</div><div class="tbl"><table><thead><tr>{th}</tr></thead>'
                         f'<tbody>{"".join(rows)}</tbody></table></div>' + (f'<p class="note">{html.escape(note)}</p>' if note else ""))
    tpl = Path(template_path).read_text(encoding="utf-8")
    kp = "".join(f'<div class="kpi reveal" style="animation-delay:{0.3 + i * 0.07:.2f}s"><div class="v">{v}</div><div class="l">{html.escape(l)}</div></div>'
                 for i, (v, l) in enumerate(kpis))
    out = (tpl.replace("{{CONTENT}}", "\n".join(parts)).replace("{{TOC}}", "\n".join(toc)).replace("{{LEDE}}", _md_inline(lede))
              .replace("{{KPIS}}", kp).replace("{{FOOTER}}", footer).replace("{{DATE}}", date)
              .replace("{{DATA_JSON}}", json.dumps(data, ensure_ascii=False)))
    (out_dir / "index.html").write_text(out, encoding="utf-8")
    return out_dir / "index.html"
