"""风控助手的静态数据：方法知识库、企业名称索引与企业风险画像分片，输出到 docs/assistant/。

检索在浏览器端完成：页面先在名称索引里匹配问题中的公司名或代码，再按编号取对应分片里的画像，
连同知识库中最相关的段落一起发给 Cloudflare Worker 调用模型。数据全部取自图谱浏览器已公开的内容。
依赖 24 号 --publish 产出的 docs/graph/data 与 12 号写出的 cache/work/report_kb.json。
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from common import *

ROOT_DIR = Path(__file__).resolve().parents[1]
GX = ROOT_DIR / "docs" / "graph" / "data"
OUTD = ROOT_DIR / "docs" / "assistant"
(OUTD / "p").mkdir(parents=True, exist_ok=True)
SHARDS = 64
REL_CN = {"sup": "供应商", "cus": "客户", "holder": "股东", "invest": "对外投资", "ctrl": "同一控制", "person": "同一关键人员", "affil": "联营/合营",
          "rp_other": "其他关联方", "codef": "共同被告（担保连带）", "debtor": "债务人", "creditor": "债权人"}
ANN_CN = ["债务逾期公告", "债券兑付违约", "被列为失信被执行人", "银行账户被冻结", "违规担保或担保逾期", "被申请破产或重整", "被立案调查", "新被实施 ST",
          "控股股东或实控人失信、破产重整或债务违约", "股东所持股份被司法冻结", "股东质押平仓或被动减持", "债券违约、展期或破产重整公告"]
TYPE_CN = ["上市公司", "非上市企业", "自然人", "金融机构", "政府与国资", "名称不完整"]


def load_js(path, key):
    t = path.read_text(encoding="utf-8")
    tag = f'GXD["{key}"]='
    return json.loads(t[t.index(tag) + len(tag):].rstrip().rstrip(";"))


# ---------------------------------------------------------------- 1. 知识库
kb = []
md = (ROOT_DIR / "assistant" / "methods.md").read_text(encoding="utf-8")
for part in md.split("\n## ")[1:]:
    title, _, body = part.partition("\n")
    kb.append({"title": title.strip(), "text": body.strip(), "src": "方法说明"})
for r in json.load(open(WORK / "report_kb.json", encoding="utf-8")):
    kb.append({"title": r["title"], "text": r["text"], "src": "研究报告"})
for i, r in enumerate(kb):
    r["id"] = i
json.dump(kb, open(OUTD / "kb.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print(f"[知识库] {len(kb)} 段，{sum(len(r['text']) for r in kb):,} 字")

# ---------------------------------------------------------------- 2. 企业画像
nd = load_js(GX / "nodes.js", "nodes")
N, L = nd["n"], nd["listed"]
names, types, REL = nd["names"], nd["types"], nd["rel"]
code, short, ind, prov, board = nd["code"], nd["short"], nd["ind"], nd["prov"], nd["board"]
bill = {int(k): v for k, v in nd["bill"].items()}
reg = {int(k): v for k, v in nd["reg"].items()}
ann = {int(k): v for k, v in nd["ann"].items()}
pl = {int(k): v for k, v in nd["pl"].items()}
years = list(range(2016, 2024))
Y = {}
for t in years:
    d = load_js(GX / f"y{t}.js", f"y{t}")
    fin, trade, y, q4, pd4 = {}, {}, {}, {}, {}
    for k in range(0, len(d["risk"]), 5):
        i = d["risk"][k]
        fin[i], trade[i] = d["risk"][k + 1], d["risk"][k + 2]
    for k in range(0, len(d["firms"]), 6):
        i = d["firms"][k]
        y[i], pd4[i], q4[i] = d["firms"][k + 1], d["firms"][k + 3], d["firms"][k + 5]
    nb = defaultdict(list)
    P = d["pairs"]
    for k in range(0, len(P), 4):
        a, b, m = P[k], P[k + 1], P[k + 2]
        rels = [REL[j] for j in range(len(REL)) if m >> j & 1]
        nb[a].append((b, rels))
        nb[b].append((a, rels))
    Y[t] = {"fin": fin, "trade": trade, "y": y, "q4": q4, "pd4": pd4, "nb": nb}


def label(i):
    return f"{short[i]}（{code[i]}）" if i < L else names[i]


def status(t, i):
    if Y[t]["y"].get(i) == 1:
        return "决策日后 12 个月内违约"
    if Y[t]["fin"].get(i, 0) > 0:
        return "出险（近 36 个月金融债务被诉）"
    if Y[t]["trade"].get(i, 0) > 0:
        return "有经营欠款诉讼"
    return ""


def profile(i):
    kind = int(types[i])
    lines = [f"主体：{names[i]}" + (f"，股票代码 {code[i]}，简称 {short[i]}，申万行业 {ind[i]}" + (f"，注册地 {prov[i]}" if prov[i] else "") + f"，{board[i]}" if i < L else f"，类型 {TYPE_CN[kind]}")]
    if i in reg:
        c, cty, cap, ins, size, est = reg[i]
        bits = [x for x in [f"{c}{cty}", f"注册资本 {cap:,} 万元" if cap else "", f"参保 {ins} 人" if ins is not None else "", f"{size}企业" if size else "",
                            f"成立于 {est} 年" if est else ""] if x]
        lines.append("工商登记：" + "，".join(bits))
    risky_years = [t for t in years if Y[t]["fin"].get(i, 0) > 0]
    def_years = [t for t in years if Y[t]["y"].get(i) == 1]
    present = [t for t in years if i in Y[t]["nb"] or i in Y[t]["y"]]
    if present:
        lines.append(f"出现在图谱的观察年：{present[0]}—{present[-1]}")
    lines.append("出险观察年：" + ("、".join(map(str, risky_years)) if risky_years else "无") + "（决策日前 36 个月内作为被告被提起金融债务诉讼）")
    if i < L:
        lines.append("样本内首次违约：" + ("、".join(f"{t} 年观察期（决策日 {t + 1}-04-30 后 12 个月内）" for t in def_years) if def_years else "无"))
        sc = [(t, Y[t]["pd4"][i], Y[t]["q4"][i]) for t in years if Y[t]["pd4"].get(i, -1) >= 0]
        if sc:
            lines.append("样本外违约概率（含图谱特征）：" + "；".join(f"{t} 年 {p / 100:.2f}%（分位 {q / 10:.1f}）" for t, p, q in sc))
    if i in ann:
        cnt, last = defaultdict(int), {}
        for k, ym in ann[i]:
            cnt[k] += 1
            last[k] = max(last.get(k, ""), ym)
        lines.append("公告风险事件：" + "；".join(f"{ANN_CN[k]} {cnt[k]} 次（最近 {last[k]}）" for k in sorted(cnt, key=lambda k: last[k], reverse=True)))
    if i in pl:
        lines.append("年末股权质押比例：" + "，".join(f"{yy} 年 {v}%" for yy, v in pl[i]))
    if i in bill:
        ov, un, lo, lu = bill[i]
        bits = ([f"承兑人逾期名单 {ov} 期（最近 {lo}）"] if ov else []) + ([f"信用信息未披露名单 {un} 期（最近 {lu}）"] if un else [])
        lines.append("票交所票据名单：" + "；".join(bits))
    if present:
        t = present[-1]
        nb = Y[t]["nb"].get(i, [])
        risky = []
        for j, rels in nb:
            s = status(t, j)
            if s:
                risky.append((0 if "违约" in s else 1 if "出险" in s else 2, label(j), "、".join(REL_CN.get(r, r) for r in rels), s))
        risky.sort()
        lines.append(f"{t} 年图谱中的关联方共 {len(nb)} 个，其中有风险事件的 {len(risky)} 个" + ("：" if risky else "。"))
        for _, nm, rel, s in risky[:10]:
            lines.append(f"  - {nm}（关系：{rel}）：{s}")
    return "\n".join(lines)


keep = []
for i in range(N):
    if i < L or i in bill or i in ann or any(Y[t]["fin"].get(i, 0) > 0 for t in years):
        keep.append(i)
shards = defaultdict(dict)
index = []
for i in keep:
    shards[i % SHARDS][i] = profile(i)
    index.append([i, names[i], short[i] if i < L else "", code[i] if i < L else ""])
for k, v in shards.items():
    json.dump(v, open(OUTD / "p" / f"{k}.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
json.dump({"shards": SHARDS, "items": index}, open(OUTD / "names.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
size = sum(f.stat().st_size for f in (OUTD / "p").glob("*.json"))
print(f"[画像] {len(keep):,} 个主体（上市 {L:,}），分片 {SHARDS} 个共 {size / 1e6:.1f} MB；名称索引 {(OUTD / 'names.json').stat().st_size / 1e6:.2f} MB")
print(shards[[i for i in keep if i < L and code[i] == "600221"][0] % SHARDS][[i for i in keep if i < L and code[i] == "600221"][0]])

# 网页组件
import shutil
shutil.copy(ROOT_DIR / "src" / "assistant_widget.js", ROOT_DIR / "docs" / "assets" / "assistant.js")
print("[组件] docs/assets/assistant.js")
