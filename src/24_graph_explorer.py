"""全量关联图谱浏览器（实名）：把 2016—2023 年逐年建模图谱导出成可离线打开的交互页面。

布局：在全部年份关系的并集图上跑一次 ForceAtlas2（tools/layout/layout.mjs），各年份共用坐标，切换年份时节点不跳动。
数据：节点全称、主体类型、行业；逐年的关系（按关系类型位掩码）、节点风险（36 个月衰减）、样本企业的违约结局与样本外违约概率。
输出含实名，受 CNRDS / CSMAR 许可限制只留在本地：report/graph_explorer/（report/ 不入库）。
用法：python 24_graph_explorer.py [--relayout] [--iter 700] [--zip]
"""
import argparse
import json
import shutil
import subprocess
import unicodedata

import numpy as np
import pandas as pd
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--relayout", action="store_true")
ap.add_argument("--iter", type=int, default=700)
ap.add_argument("--out", default=str(ROOT / "report" / "graph_explorer"))
ap.add_argument("--pos", default=str(WORK / "gx_pos.json"), help="布局坐标文件（node layout.mjs 的输出）")
ap.add_argument("--zip", action="store_true", help="另存一份 zip 供提交")
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


size = js(GX / "data" / "nodes.js", "nodes", {
    "n": len(ids), "listed": L, "names": names, "types": types, "xy": xy.ravel().tolist(),
    "code": codes, "short": short, "ind": ind, "prov": prov, "board": board, "rel": REL})
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
if args.zip:                                           # 提交用：整个文件夹打包，解压后双击 index.html 即可
    z = shutil.make_archive(str(GX.parent / "关联图谱全景_graph_explorer"), "zip", GX.parent, GX.name)
    print("[zip]", z, f"{Path(z).stat().st_size / 1e6:.1f} MB")
