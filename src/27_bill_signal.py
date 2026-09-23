"""票据名单作为新的风险信号：关联方（或企业自己）出现在票交所承兑人名单上，是否预示未来违约。

两类信号分开看：
- overdue   ：承兑人逾期 / 持续逾期名单（商票实际违约，2021-12—2023-07 共 5 期存档）
- undisclosed：信用信息未披露名单（未按规定披露承兑与付款信息，2023-01 起按月，每期约 3 千家）

口径与正文一致：月度面板、标签 y6（未来 6 个月内首次违约）、只用滚动外推测试期；
暴露 = 决策月之前 12 个月内，某个一跳关联方（或企业自己）上过名单。
除总体违约率外，报告两项条件检验：① 按企业分层的 MH 优势比（同一家企业内部比较）；
② 在“没有一跳诉讼信号”的月份里单独看票据信号，回答它是不是已有信号的重复。
"""
import json

import numpy as np
import pandas as pd

from common import *

rng = np.random.default_rng(20260923)
ONE_HOP = ["dy_sup_3m", "dy_cus_3m", "dy_holder_3m", "dy_invest_3m", "dy_ctrl_3m", "dy_person_3m",
           "dy_affil_3m", "dy_rp_other_3m", "dy_codef_3m", "dy_debtor_3m", "dy_all_enf_3m"]
LOOKBACK = pd.DateOffset(months=12)

B = pd.read_parquet(WORK / "bill_events.parquet")
E = pd.read_parquet(WORK / "edges.parquet")
master = pd.read_parquet(WORK / "firm_master.parquet")
dyn = pd.read_parquet(WORK / "dyn_panel.parquet", columns=["code", "t", "k", "S", "y6", "def_date"] + ONE_HOP)
oot = pd.read_parquet(WORK / "dyn_oot.parquet", columns=["code", "t", "k"])
dyn = dyn.merge(oot, on=["code", "t", "k"], how="inner")
dyn["one"] = (dyn[ONE_HOP].to_numpy() > 0).any(1)

# ---------------------------------------------------------------- 名单主体 → 图谱节点
listed_name = master.assign(nm=master.full.map(lambda x: norm_name(x) if isinstance(x, str) else None)).dropna(subset=["nm"])
name2code = dict(zip(listed_name.nm, listed_name.code))
B["node"] = np.where(B.name.map(name2code).notna(), "C:" + B.name.map(name2code).astype(str), "E:" + B.name)

# 一跳关联方（含反向边，图里每条关系都存了两向）
pairs = E[["src", "dst", "rel", "t"]].copy()
pairs = pairs[pairs.src.str.startswith("C:")]
pairs["code"] = pairs.src.str[2:]
link = pairs.merge(B[["node", "date", "kind"]], left_on="dst", right_on="node", how="inner")
own = B.merge(master[["code"]].assign(node="C:" + master.code), on="node", how="inner")[["code", "date", "kind"]]
print(f"[匹配] 关联方命中 {link.node.nunique():,} 个主体、{link.code.nunique():,} 家上市公司；企业自身上榜 {own.code.nunique()} 家")


def flag_months(ev, kinds, label):
    """ev: code/date/kind 事件表 → 每个企业-月是否在过去 12 个月内出现过该类名单。
    关联方事件带观察年 t：企业-月只认当年图谱里的关联方，不用决策日之后才披露的关系。"""
    on = ["code", "t"] if "t" in ev.columns else ["code"]
    e = ev[ev.kind.isin(kinds)][on + ["date"]].drop_duplicates()
    m = dyn[["code", "t", "k", "S"]].merge(e, on=on, how="inner")
    m = m[(m.date <= m.S) & (m.date > m.S - LOOKBACK)]
    key = set(zip(m.code, m.t, m.k))
    dyn[label] = [(c, t, k) in key for c, t, k in zip(dyn.code, dyn.t, dyn.k)]
    return dyn[label].sum()


for label, ev, kinds in [("bill_overdue", link, ["overdue"]), ("bill_undis", link, ["undisclosed"]),
                         ("bill_any", link, ["overdue", "undisclosed"]), ("own_bill", own, ["overdue", "undisclosed"])]:
    n = flag_months(ev, kinds, label)
    print(f"  {label:<14} 暴露企业-月 {n:,}")

own_score = pd.read_parquet(WORK / "dyn_oot.parquet", columns=["code", "t", "k", "D1 自身(含自身时序)"])
dyn = dyn.merge(own_score, on=["code", "t", "k"], how="left")
dyn["q5"] = dyn.groupby("S")["D1 自身(含自身时序)"].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
dyn["stratum"] = dyn.S.dt.strftime("%Y-%m") + "_" + dyn.q5.astype("Int64").astype(str)   # 同月 × 自身评分五分位
COV = dyn[dyn.S >= "2023-01-01"]                                   # 名单开始按月发布之后，口径才完整
base = dyn.y6.mean()


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mh_or(a, b, c, d, n):
    num, den = (a * d / n).sum(), (b * c / n).sum()
    return num / den if den > 0 else np.nan


def strat_or(df, expo, strata):
    """按 strata 分层的 MH 优势比；企业整簇 bootstrap 给区间。"""
    d = df[[expo, "y6", strata, "code"]].dropna()
    g = d.groupby([strata, expo]).y6.agg(["sum", "count"]).unstack(fill_value=0)
    for col in [("sum", True), ("count", True), ("sum", False), ("count", False)]:
        if col not in g:
            g[col] = 0
    a = g[("sum", True)].values.astype(float)
    b = (g[("count", True)] - g[("sum", True)]).values.astype(float)
    c = g[("sum", False)].values.astype(float)
    dd = (g[("count", False)] - g[("sum", False)]).values.astype(float)
    n = a + b + c + dd
    point = mh_or(a, b, c, dd, n)
    codes = d.code.unique()
    bs = []
    for _ in range(400):
        pick = pd.Series(rng.integers(0, len(codes), len(codes))).map(lambda i: codes[i])
        s_ = d[d.code.isin(set(pick))]
        g2 = s_.groupby([strata, expo]).y6.agg(["sum", "count"]).unstack(fill_value=0)
        for col in [("sum", True), ("count", True), ("sum", False), ("count", False)]:
            if col not in g2:
                g2[col] = 0
        a2 = g2[("sum", True)].values.astype(float)
        b2 = (g2[("count", True)] - g2[("sum", True)]).values.astype(float)
        c2 = g2[("sum", False)].values.astype(float)
        d2 = (g2[("count", False)] - g2[("sum", False)]).values.astype(float)
        bs.append(mh_or(a2, b2, c2, d2, a2 + b2 + c2 + d2))
    return {"OR": float(point), "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5)),
            "层数": int(len(a))}


def within_firm(df, expo):
    t = df.groupby(["code", expo]).y6.agg(["sum", "count"]).unstack(fill_value=0)
    for col in [("sum", True), ("count", True), ("sum", False), ("count", False)]:
        if col not in t:
            t[col] = 0
    a = t[("sum", True)].values.astype(float)
    b = (t[("count", True)] - t[("sum", True)]).values.astype(float)
    c = t[("sum", False)].values.astype(float)
    d = (t[("count", False)] - t[("sum", False)]).values.astype(float)
    n = a + b + c + d
    point = mh_or(a, b, c, d, n)
    bs = [mh_or(*[v[i] for v in (a, b, c, d, n)]) for i in (rng.integers(0, len(a), len(a)) for _ in range(800))]
    return {"OR": float(point), "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5)),
            "共变企业数": int((((a + b) > 0) & ((c + d) > 0)).sum())}


rows, res = [], {"基准6个月违约率": float(base), "覆盖期": "2023-01 起月度名单完整"}
for label, cn in [("bill_overdue", "关联方上过逾期/持续逾期名单"), ("bill_overdue", "关联方上过逾期名单（限 2023 年起，与未披露同窗口）"),
                  ("bill_undis", "关联方上过信用信息未披露名单"),
                  ("bill_any", "关联方上过任一票据名单"), ("own_bill", "企业自己上过票据名单")]:
    d = dyn if cn == "关联方上过逾期/持续逾期名单" else COV           # 逾期名单只有 2021-12—2023-07，默认不限窗口
    k, n = int(d.loc[d[label], "y6"].sum()), int(d[label].sum())
    lo, hi = wilson(k, n)
    r0 = d.loc[~d[label], "y6"].mean()
    wf = within_firm(d[["code", label, "y6"]], label)
    sm = strat_or(d, label, "stratum")                              # 同月 × 自身评分五分位
    rows.append({"信号": cn, "暴露企业-月": n, "6个月违约": k, "违约率": k / n if n else np.nan, "CI_lo": lo, "CI_hi": hi,
                 "未暴露违约率": r0, "倍数": (k / n) / r0 if n and r0 else np.nan,
                 "同月同评分MH优势比": sm["OR"], "分层_lo": sm["lo"], "分层_hi": sm["hi"],
                 "企业内MH优势比": wf["OR"], "MH_lo": wf["lo"], "MH_hi": wf["hi"], "共变企业数": wf["共变企业数"]})
    res[label + ("_窗口对齐" if "限 2023" in cn else "")] = {"n": n, "rate": k / n if n else None, "base": float(r0),
                                                             "分层": sm, "within": wf}

# 增量：只看没有一跳诉讼信号的月份
clean = COV[~COV.one]
for label, cn in [("bill_undis", "无一跳诉讼信号时，关联方上未披露名单"), ("bill_any", "无一跳诉讼信号时，关联方上任一票据名单")]:
    k, n = int(clean.loc[clean[label], "y6"].sum()), int(clean[label].sum())
    lo, hi = wilson(k, n)
    r0 = clean.loc[~clean[label], "y6"].mean()
    rows.append({"信号": cn, "暴露企业-月": n, "6个月违约": k, "违约率": k / n if n else np.nan, "CI_lo": lo, "CI_hi": hi,
                 "未暴露违约率": r0, "倍数": (k / n) / r0 if n and r0 else np.nan})
    res["增量_" + label] = {"n": n, "rate": k / n if n else None, "base": float(r0)}

out = pd.DataFrame(rows)
out.to_csv(OUT / "bill_signal.csv", index=False, encoding="utf-8-sig")
print("\n" + out.round(4).to_string(index=False))
res["名单规模"] = {"记录数": int(len(B)), "企业数": int(B.name.nunique()),
                   "未披露期数": int(B[B.kind == "undisclosed"].ym.nunique()), "逾期期数": int(B[B.kind == "overdue"].ym.nunique()),
                   "命中关联方": int(link.node.nunique()), "命中上市公司": int(link.code.nunique()), "自身上榜": int(own.code.nunique())}
json.dump(res, open(OUT / "bill_signal.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("\n[done]", OUT / "bill_signal.json")
