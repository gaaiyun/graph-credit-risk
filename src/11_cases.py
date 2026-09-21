"""案例：测试期违约企业中，图谱信息显著提高风险排序的样本；回溯关联方风险路径（一跳 + 经外部实体的二跳）。
输出：报告用网络图 PNG（上市公司实名、自然人打码）、网页用 JSON（全部脱敏）。"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
FIG = OUT / "fig"
REL_CN = {"sup": "供应商", "cus": "客户", "holder": "股东", "invest": "对外投资", "ctrl": "同一控制", "person": "同一关键人员",
          "affil": "联营/合营", "rp_other": "其他关联方", "codef": "共同被告", "debtor": "债务人", "creditor": "债权人"}
KEY_REL = ["holder", "codef", "debtor", "cus", "sup", "ctrl", "affil", "person", "invest", "rp_other", "creditor"]

P = pd.read_parquet(WORK / "panel.parquet")
pred = pd.read_parquet(WORK / "predictions.parquet")
E = pd.read_parquet(WORK / "edges.parquet")
R = pd.read_parquet(WORK / "node_risk.parquet")
R = R.rename(columns={R.columns[0]: "node"}) if "node" not in R.columns else R
parties = pd.read_parquet(WORK / "lit_parties.parquet")
cases = pd.read_parquet(WORK / "lit_cases.parquet")
master = pd.read_parquet(WORK / "firm_master.parquet").set_index("code")
short = master.short.to_dict()

for c in ["gbm::M1 自身", "gbm::M4 +结构位置"]:
    pred[c + "_pct"] = pred.groupby("t")[c].rank(pct=True)
T = pred[pred.t.isin(TEST_YEARS) & (pred.y == 1)].copy()
T["gain"] = T["gbm::M4 +结构位置_pct"] - T["gbm::M1 自身_pct"]


def label(node, public=False, idx=None):
    kind, name = node[:2], node[2:]
    if kind == "C:":
        return f"上市公司{idx}" if public else f"{short.get(name, name)}({name})"
    if is_person(name):
        return f"自然人{idx}"
    return f"非上市企业{idx}" if public else name


def node_events(node, D):
    lo = D - pd.DateOffset(months=LOOKBACK_M)
    if node.startswith("C:"):
        c = node[2:]
        e = cases[(cases.code == c) & cases.self_def & cases.ctype.isin(["fin", "trade"]) & (cases.date > lo) & (cases.date <= D)]
        ev = [{"date": str(d.date()), "case": n, "amt": (None if pd.isna(a) else float(a))} for d, n, a in zip(e.date, e.Cname, e.amt)]
        p = parties[(parties.name_code == c) & (parties.code != c) & (parties.side == "def") & parties.ctype.isin(["fin", "trade"])
                    & (parties.date > lo) & (parties.date <= D)]
    else:
        ev = []
        p = parties[(parties.name == node[2:]) & (parties.side == "def") & parties.ctype.isin(["fin", "trade"])
                    & (parties.date > lo) & (parties.date <= D)]
    p = p.merge(cases[["case_id", "Cname", "amt"]], on="case_id")
    ev += [{"date": str(d.date()), "case": n, "amt": (None if pd.isna(a) else float(a))} for d, n, a in zip(p.date, p.Cname, p.amt)]
    ev = sorted({(e["date"], e["case"]): e for e in ev}.values(), key=lambda e: e["date"])
    return ev


def build_case(code, t):
    D = decision_date(t)
    rk = R[R.t == t].set_index("node")
    Et = E[E.t == t]
    risk = lambda n: float(rk.fin.get(n, 0) + rk.trade.get(n, 0)) if n in rk.index else 0.0
    focal = "C:" + code
    one = Et[Et.src == focal]
    one = one.groupby("dst").agg(rels=("rel", lambda s: sorted(set(s), key=KEY_REL.index)), w=("w", "max")).reset_index()
    one["risk"] = one.dst.map(risk)
    deg = pd.concat([Et.src, Et.dst]).value_counts()
    # 二跳：经非枢纽外部实体/上市公司到达的有风险节点
    two = Et[Et.src.isin(one.dst) & (Et.dst != focal) & ~Et.dst.isin(one.dst)].copy()
    two = two[two.src.map(deg).fillna(0) <= 100]
    two["risk"] = two.dst.map(risk)
    two = two[two.risk > 0].sort_values("risk", ascending=False).drop_duplicates("dst").head(8)
    via = one[one.dst.isin(two.src)]
    riskies = one[(one.risk > 0) & ~one.dst.isin(via.dst)].sort_values("risk", ascending=False).head(max(0, 9 - len(via)))
    keep1 = pd.concat([via, riskies])
    extra = one[(one.risk == 0) & ~one.dst.isin(keep1.dst)].sort_values("w", ascending=False)
    extra = extra[extra.rels.map(lambda r: r[0] in ("holder", "cus", "sup", "ctrl"))].head(max(0, 12 - len(keep1)))
    keep1 = pd.concat([keep1, extra])
    two = two[two.src.isin(keep1.dst)]
    nodes = [{"id": focal, "ring": 0, "risk": 0.0}] + \
            [{"id": r.dst, "ring": 1, "risk": r.risk, "rels": r.rels} for r in keep1.itertuples()] + \
            [{"id": r.dst, "ring": 2, "risk": r.risk, "via": r.src, "rels": [r.rel]} for r in two.itertuples()]
    edges = [{"s": focal, "t": r.dst, "rel": r.rels[0]} for r in keep1.itertuples()] + \
            [{"s": r.src, "t": r.dst, "rel": r.rel} for r in two.itertuples()]
    for n in nodes:
        n["events"] = node_events(n["id"], D) if n["risk"] > 0 else []
    # 径向布局
    ring1 = [n for n in nodes if n["ring"] == 1]
    ang = {n["id"]: 2 * np.pi * i / max(len(ring1), 1) + 0.3 for i, n in enumerate(ring1)}
    for n in nodes:
        if n["ring"] == 0:
            n["x"], n["y"] = 0.0, 0.0
        elif n["ring"] == 1:
            n["x"], n["y"] = float(np.cos(ang[n["id"]])), float(np.sin(ang[n["id"]]))
    kids = {}
    for n in nodes:
        if n["ring"] == 2:
            kids.setdefault(n["via"], []).append(n)
    for via, ks in kids.items():
        a0 = ang.get(via, 0)
        for j, n in enumerate(ks):
            a = a0 + (j - (len(ks) - 1) / 2) * 0.28
            n["x"], n["y"] = float(1.95 * np.cos(a)), float(1.95 * np.sin(a))
    ev = cases[(cases.code == code) & cases.self_def & (cases.ctype == "fin") & (cases.date > D)].sort_values("date").head(1)
    row = pred[(pred.code == code) & (pred.t == t)].iloc[0]
    return {"code": code, "t": int(t), "decision": str(D.date()), "industry": SW_L1.get(P[(P.code == code) & (P.t == t)].sw_l1.iloc[0], ""),
            "event": {"date": str(ev.date.iloc[0].date()), "case": ev.Cname.iloc[0], "amt": None if pd.isna(ev.amt.iloc[0]) else float(ev.amt.iloc[0])},
            "pct_m1": float(row["gbm::M1 自身_pct"]), "pct_m4": float(row["gbm::M4 +结构位置_pct"]),
            "nodes": nodes, "edges": edges}


def draw(case, path, public=False):
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=200)
    pos = {n["id"]: (n["x"], n["y"]) for n in case["nodes"]}
    for e in case["edges"]:
        (x0, y0), (x1, y1) = pos[e["s"]], pos[e["t"]]
        ax.plot([x0, x1], [y0, y1], color="#cbd2d9", lw=0.9, zorder=1)
        f = 0.5
        ax.text(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, REL_CN.get(e["rel"], e["rel"]), fontsize=5.6, color="#52606d", ha="center",
                va="center", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9), zorder=2)
    cnt = {}
    for n in case["nodes"]:
        k = "C" if n["id"].startswith("C:") else ("P" if is_person(n["id"][2:]) else "E")
        cnt[k] = cnt.get(k, 0) + 1
        n["_lab"] = ("案例企业" if (public and n["ring"] == 0) else label(n["id"], public, cnt[k]))
        x, y = pos[n["id"]]
        colr = "#1f2933" if n["ring"] == 0 else ("#c2410c" if n["risk"] > 0 else "#9aa5b1")
        size = 150 if n["ring"] == 0 else (45 + 30 * min(n["risk"], 3))
        mk = "s" if n["id"].startswith("C:") else "o"
        ax.scatter([x], [y], s=size, color=colr, marker=mk, zorder=3, edgecolor="white", lw=0.8)
        txt = n["_lab"] if len(n["_lab"]) <= 14 else n["_lab"][:13] + "…"
        if n["ring"] == 0:
            ax.text(x, y - 0.16, txt, fontsize=7, ha="center", va="top", color="#1f2933", zorder=4, fontweight="bold")
        else:
            a = np.arctan2(y, x)
            ax.text(x + 0.1 * np.cos(a), y + 0.1 * np.sin(a), txt, fontsize=6, ha="left" if np.cos(a) >= 0 else "right",
                    va="center", color="#1f2933", zorder=4)
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.3, max(xs) + 1.3)
    ax.set_ylim(min(ys) - 0.35, max(ys) + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ttl = f"{'案例企业' if public else label('C:' + case['code'])}｜决策日 {case['decision']}｜首次被诉 {case['event']['date']}（{case['event']['case']}）"
    ax.set_title(ttl, fontsize=7.5, color="#1f2933")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    cand = T.sort_values("gain", ascending=False)
    out_cases = []
    for r in cand.itertuples():
        c = build_case(r.code, r.t)
        risky1 = [n for n in c["nodes"] if n["ring"] >= 1 and n["risk"] > 0]
        if len(risky1) >= 1 and c["pct_m4"] >= 0.85:
            out_cases.append(c)
        if len(out_cases) >= 3:
            break
    for i, c in enumerate(out_cases):
        draw(c, FIG / f"fig7_case{i + 1}.png", public=False)
        print(f"case{i + 1}: {label('C:' + c['code'])} t={c['t']} 行业={c['industry']} M1分位={c['pct_m1']:.3f} M4分位={c['pct_m4']:.3f} "
              f"事件={c['event']}  风险邻居={[(label(n['id'], idx=i), n.get('rels'), n['ring'], len(n['events'])) for i, n in enumerate(c['nodes']) if n['risk'] > 0]}")
    # 网页用：全部脱敏
    pub = []
    sw_now = pd.read_parquet(WORK / "sw_hist.parquet")
    for i, c in enumerate(out_cases):
        cnt, mp = {}, {}
        D = pd.Timestamp(c["decision"])
        ind_at = sw_now[sw_now.start <= D].groupby("code").sw.last().str[:2].map(SW_L1)
        for n in c["nodes"]:
            if n["ring"] == 0:
                mp[n["id"]] = "案例企业"
                continue
            if n["id"].startswith("C:"):
                base = f"{ind_at.get(n['id'][2:], '')}上市公司"
            elif is_person(n["id"][2:]):
                base = "自然人"
            else:
                base = REL_CN.get((n.get("rels") or ["rp_other"])[0], "关联方")
            cnt[base] = cnt.get(base, 0) + 1
            mp[n["id"]] = f"{base}{cnt[base]}"
        pub.append({"title": f"案例{i + 1}：{c['industry']}企业", "t": c["t"], "decision": c["decision"],
                    "event": {"date": c["event"]["date"][:7], "case": c["event"]["case"] if len(c["event"]["case"]) <= 10 else "融资租赁合同纠纷"},
                    "pct_m1": c["pct_m1"], "pct_m4": c["pct_m4"],
                    "nodes": [{"id": mp[n["id"]], "ring": n["ring"], "risk": round(n["risk"], 3), "x": n["x"], "y": n["y"],
                               "listed": n["id"].startswith("C:"), "rels": [REL_CN[x] for x in n.get("rels", [])], "via": mp.get(n.get("via"), ""),
                               "n_events": len(n["events"])} for n in c["nodes"]],
                    "edges": [{"s": mp[e["s"]], "t": mp[e["t"]], "rel": REL_CN[e["rel"]]} for e in c["edges"]]})
    json.dump(pub, open(OUT / "cases_public.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(out_cases, open(WORK / "cases_private.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("cases:", len(out_cases))
