"""全量关联图谱浏览器（实名）：把 2016—2023 年逐年建模图谱导出成可离线打开的交互页面。

布局：在全部年份关系的并集图上跑一次 ForceAtlas2（tools/layout/layout.mjs），各年份共用坐标，切换年份时节点不跳动。
数据：节点全称、主体类型、行业；逐年的关系（按关系类型位掩码）、节点风险（36 个月衰减）、样本企业的违约结局与样本外违约概率。
输出含实名：本地 report/graph_explorer/（report/ 不入库）；--publish 同步到 docs/graph/ 公开（用户已确认不脱敏）。
用法：python 24_graph_explorer.py [--relayout] [--iter 700] [--zip] [--publish]
"""
import argparse
import json
import shutil
import subprocess
import unicodedata
import zipfile

import numpy as np
import pandas as pd
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--relayout", action="store_true")
ap.add_argument("--iter", type=int, default=700)
ap.add_argument("--out", default=str(ROOT / "report" / "graph_explorer"))
ap.add_argument("--pos", default=str(WORK / "gx_pos.json"), help="布局坐标文件（node layout.mjs 的输出）")
ap.add_argument("--zip", action="store_true", help="另存一份 zip 供提交")
ap.add_argument("--publish", action="store_true", help="同步到 docs/graph/（GitHub Pages，实名公开）")
ap.add_argument("--fa2", default="{}", help="覆盖 ForceAtlas2 参数的 JSON（默认值写在 layout.mjs）")
args = ap.parse_args()

GX = Path(args.out)
LAY = ROOT / "tools" / "layout"
UNION, POS = WORK / "gx_union.json", Path(args.pos)
REL = ["sup", "cus", "holder", "invest", "ctrl", "person", "affil", "rp_other", "codef", "debtor", "creditor"]
BIT = {r: 1 << i for i, r in enumerate(REL)}

E = pd.read_parquet(WORK / "edges.parquet")
P = pd.read_parquet(WORK / "predictions.parquet")[["code", "t", "y", "sw_l1"]]
O = pd.read_parquet(WORK / "oot_rolling.parquet")[["code", "t", "M1 自身", "M4 +结构位置"]]
R = pd.read_parquet(WORK / "node_risk.parquet").rename(columns={"index": "node"})
M = pd.read_parquet(WORK / "firm_master.parquet").set_index("code")

# ---------------------------------------------------------------- 节点表：上市公司在前（按代码），外部主体在后
allid = pd.unique(pd.concat([E.src, E.dst, "C:" + P.code]))
listed = sorted(i for i in allid if i.startswith("C:"))
ents = sorted(i for i in allid if not i.startswith("C:"))
ids = listed + ents
idx = pd.Series(np.arange(len(ids)), index=ids)
L = len(listed)


def ent_type(name):
    if is_person(name):
        return 2
    if is_fin_holder(name):
        return 3
    if is_gov(name):
        return 4
    if is_anonymous(name):
        return 5
    return 1


codes = [i[2:] for i in listed]
short = [unicodedata.normalize("NFKC", str(M.short.get(c) or c)).replace(" ", "") for c in codes]
full = [M.full.get(c) or s for c, s in zip(codes, short)]
ind_code = P.sort_values("t").drop_duplicates("code", keep="last").set_index("code").sw_l1
ind = [SW_L1.get(str(ind_code.get(c, "")), "") for c in codes]
prov = [M.province.get(c) or "" for c in codes]
board = [M.board.get(c) or "" for c in codes]
types = "0" * L + "".join(str(ent_type(i[2:])) for i in ents)
names = full + [i[2:] for i in ents]

# ---------------------------------------------------------------- 并集图布局（ForceAtlas2）
si, di = idx[E.src].values, idx[E.dst].values
k = si < di
up = np.unique(np.stack([si[k], di[k]], 1), axis=0)
need = args.relayout or not POS.exists() or len(json.loads(POS.read_text())) != 2 * len(ids)
if need:
    UNION.write_text(json.dumps({"n": len(ids), "pairs": up.ravel().tolist()}))
    print(f"[layout] {len(ids)} 节点 {len(up)} 对关联，ForceAtlas2 {args.iter} 轮 ...", flush=True)
    subprocess.run(["node", str(LAY / "layout.mjs"), str(UNION), str(POS), str(args.iter), args.fa2], check=True, cwd=LAY)
xy = np.array(json.loads(POS.read_text()), dtype=float).reshape(-1, 2)
xy -= np.median(xy, 0)
xy *= 10000 / np.quantile(np.abs(xy), 0.999)          # 99.9% 分位落在 ±10000，极端离群点不压缩主体
xy = np.clip(np.round(xy), -30000, 30000).astype(int)

# ---------------------------------------------------------------- 输出
(GX / "data").mkdir(parents=True, exist_ok=True)
(GX / "lib").mkdir(exist_ok=True)


def js(path, var, obj):
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    path.write_text(f"window.GXD=window.GXD||{{}};GXD[{json.dumps(var)}]={s};\n", encoding="utf-8")
    return path.stat().st_size


# 票据承兑人名单（票交所，自行抓取）：主体 → 名单类型与最近一期
bill = {}
bp = WORK / "bill_events.parquet"
if bp.exists():
    B = pd.read_parquet(bp)
    b_listed = {norm_name(f): c for c, f in zip(M.index, M.full) if isinstance(f, str)}
    B["node"] = np.where(B.name.map(b_listed).notna(), "C:" + B.name.map(b_listed).astype(str), "E:" + B.name)
    B = B[B.node.isin(idx.index)]
    for node, g in B.groupby("node"):
        ov = g[g.kind == "overdue"]
        un = g[g.kind == "undisclosed"]
        bill[int(idx[node])] = [len(ov), len(un), (ov.ym.max() if len(ov) else ""), (un.ym.max() if len(un) else "")]
    print(f"[bill] 名单主体命中图谱 {len(bill):,}（逾期 {sum(1 for v in bill.values() if v[0]):,}）")

# 工商登记（H 盘工商注册数据匹配结果）：非上市主体的所在地、注册资本、参保人数、规模、成立年份
reg = {}
rp = WORK / "node_registry.parquet"
if rp.exists():
    Rg = pd.read_parquet(rp)
    Rg = Rg[Rg.node.isin(idx.index)]
    for r in Rg.itertuples(index=False):
        reg[int(idx[r.node])] = [str(getattr(r, "所属城市") or ""), str(getattr(r, "所属区县") or ""),
                                 None if pd.isna(r.cap) else round(float(r.cap)), None if pd.isna(r.insured) else int(r.insured),
                                 str(getattr(r, "企业规模") or "").replace("-", ""), None if pd.isna(r.est) else int(r.est.year)]
    print(f"[工商] 图谱中 {len(reg):,} 个非上市主体附工商登记")
# 公告风险事件：上市公司（巨潮股票栏目公告标题 + ST 变动）与发债主体（巨潮债券栏目，含非上市）
ann = {}
ANN_KINDS = ["debt_overdue", "bond_default", "shixin", "acct_frozen", "illegal_guar", "bankruptcy", "csrc_probe", "st_new",
             "sh_distress", "share_frozen", "pledge_liq", "bond_issuer"]
evs = []
if (WORK / "alt_events.parquet").exists():
    Ev = pd.read_parquet(WORK / "alt_events.parquet")
    evs.append(Ev.assign(node="C:" + Ev.code)[["node", "date", "kind"]])
if (WORK / "bond_events.parquet").exists():
    evs.append(pd.read_parquet(WORK / "bond_events.parquet")[["node", "date"]].assign(kind="bond_issuer"))
if evs:
    Ev = pd.concat(evs, ignore_index=True)
    Ev = Ev[Ev.kind.isin(ANN_KINDS) & Ev.node.isin(idx.index)].drop_duplicates().sort_values("date", ascending=False)
    for node, g in Ev.groupby("node"):
        ann[int(idx[node])] = [[ANN_KINDS.index(k), f"{d:%Y-%m}"] for k, d in zip(g.kind, g.date)][:40]
    print(f"[公告] {len(ann):,} 个主体附公告风险事件（其中非上市 {sum(1 for i in ann if i >= L):,}）")
# 股权质押比例（东方财富转载的中国结算数据）：上市公司各年年末
pl = {}
if (WORK / "pledge_panel.parquet").exists():
    Pp = pd.read_parquet(WORK / "pledge_panel.parquet")
    Pp["date"] = pd.to_datetime(Pp.date)
    Pp = Pp[Pp.date.dt.month == 12].assign(node="C:" + Pp.code.astype(str).str.zfill(6))
    Pp = Pp[Pp.node.isin(idx.index) & (Pp.pledge_ratio > 0)].sort_values("date")
    for node, g in Pp.groupby("node"):
        pl[int(idx[node])] = [[int(d.year), round(float(v), 1)] for d, v in zip(g.date, g.pledge_ratio)]
    print(f"[质押] {len(pl):,} 家上市公司附年末股权质押比例")

size = js(GX / "data" / "nodes.js", "nodes", {
    "n": len(ids), "listed": L, "names": names, "types": types, "xy": xy.ravel().tolist(),
    "code": codes, "short": short, "ind": ind, "prov": prov, "board": board, "rel": REL, "bill": bill, "reg": reg, "ann": ann, "pl": pl})
print(f"[nodes] {len(ids)}（上市 {L}）{size / 1e6:.1f} MB")

meta = {}
for t in YEARS:
    e = E[E.t == t]
    a, b = idx[e.src].values, idx[e.dst].values
    k = a < b
    d = pd.DataFrame({"a": a[k], "b": b[k], "m": e.rel.map(BIT).values[k],
                      "weq": np.where(e.rel.isin(["holder", "invest"]).values[k], e.w.values[k], 0.0)})
    g = d.groupby(["a", "b"], sort=True).agg(m=("m", "sum"), weq=("weq", "max")).reset_index()
    pairs = np.stack([g.a, g.b, g.m, np.round(g.weq * 1000).astype(int)], 1).ravel().tolist()

    r = R[(R.t == t) & ((R.s > 0) | (R.dist > 0))]
    r = r[r.node.isin(idx.index)]
    risk = np.stack([idx[r.node].values, *(np.round(r[c].values * 100).astype(int) for c in ["fin", "trade", "viol", "dist"])], 1)

    p = P[P.t == t].merge(O[O.t == t], on=["code", "t"], how="left")
    p["pct1"] = p["M1 自身"].rank(pct=True) * 1000
    p["pct4"] = p["M4 +结构位置"].rank(pct=True) * 1000
    firms = [[int(idx["C:" + c]), int(y), -1 if pd.isna(p1) else round(p1 * 1e4), -1 if pd.isna(p4) else round(p4 * 1e4),
              -1 if pd.isna(q1) else round(q1), -1 if pd.isna(q4) else round(q4)]
             for c, y, p1, p4, q1, q4 in zip(p.code, p.y, p["M1 自身"], p["M4 +结构位置"], p.pct1, p.pct4)]
    size = js(GX / "data" / f"y{t}.js", f"y{t}", {"t": t, "pairs": pairs, "risk": risk.ravel().tolist(),
                                                  "firms": [v for f in firms for v in f]})
    nodes_t = np.union1d(np.union1d(g.a, g.b), [f[0] for f in firms])
    inside = r.node.map(idx).isin(nodes_t)
    meta[t] = {"nodes": int(len(nodes_t)), "pairs": int(len(g)), "firms": len(firms), "defaults": int(p.y.sum()),
               "risky": int(((r.fin > 0) & inside).sum()), "events": int(((r.s > 0) & inside).sum())}
    print(f"[y{t}] 节点 {meta[t]['nodes']:,} 关联 {meta[t]['pairs']:,} 样本企业 {len(firms):,} 违约 {meta[t]['defaults']} "
          f"出险 {meta[t]['risky']:,}  {size / 1e6:.1f} MB")

for f in ["graphology/dist/graphology.umd.min.js", "sigma/dist/sigma.min.js"]:
    shutil.copy2(LAY / "node_modules" / f, GX / "lib" / Path(f).name)
html = (ROOT / "src" / "graph_template.html").read_text(encoding="utf-8")
build = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
cases = json.loads((WORK / "cases_private.json").read_text(encoding="utf-8")) if (WORK / "cases_private.json").exists() else []
hint = [short[codes.index(c["code"])] for c in cases if c["code"] in codes]          # 报告案例企业，只写进本地实名版
html = (html.replace("__META__", json.dumps(meta, ensure_ascii=False)).replace("__YEARS__", json.dumps(YEARS))
        .replace("__BUILD__", build).replace("__HINT__", json.dumps(hint, ensure_ascii=False)))
(GX / "index.html").write_text(html, encoding="utf-8")
print("[done]", GX / "index.html")
if args.zip:                                           # 提交用离线包：解压后双击 graph_explorer/index.html
    z = ROOT / "report" / "关联图谱全景_graph_explorer.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(GX.rglob("*")):
            if f.is_file():
                zf.write(f, Path("graph_explorer") / f.relative_to(GX))
    print("[zip]", z, f"{z.stat().st_size / 1e6:.1f} MB")
if args.publish:                                       # 用户确认公开实名版（2026-09-22）：同步到 GitHub Pages 的 docs/graph/
    shutil.copytree(GX, ROOT / "docs" / "graph", dirs_exist_ok=True)
    print("[publish]", ROOT / "docs" / "graph")
