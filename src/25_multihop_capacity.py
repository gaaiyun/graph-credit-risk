"""补充三组证据：

1) 多层级传导的独立价值：违约前 6 个月里，只有多层级关联方（兄弟企业、共同创投的其他被投企业、二跳上下游）
   出事、一跳看不到的违约有多少；按信号层级分组的 6 个月违约率。
2) 预警名单的容量权衡：每月取前 1%/3%/5%/10% 时的名单规模、命中率、覆盖率与提前量。
3) 自身对照（SCCS）：只看既触发又违约的企业，违约是否集中在关联方出事之后——企业层面的固定差异自动抵消。
   暴露期 = 首次触发之后，观测窗 = 决策日到决策日+12 个月；条件泊松似然解出 IRR，按企业整簇 bootstrap 给区间。
另附单次切分与滚动外推的逐年 AUC 对比（说明为什么必须滚动外推）。
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score

from common import *

rng = np.random.default_rng(20260923)
ONE_HOP = ["dy_sup_3m", "dy_cus_3m", "dy_holder_3m", "dy_invest_3m", "dy_ctrl_3m", "dy_person_3m",
           "dy_affil_3m", "dy_rp_other_3m", "dy_codef_3m", "dy_debtor_3m", "dy_all_enf_3m"]
MULTI = ["dy_sib_ctrl_3m", "dy_sib_pe_3m", "dy_sc2hop_3m"]
D4 = "D4 +多层级时序"
D1 = "D1 自身(含自身时序)"


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------- 1. 多层级传导的独立价值
dyn = pd.read_parquet(WORK / "dyn_panel.parquet", columns=["code", "t", "k", "S", "y6", "def_date"] + ONE_HOP + MULTI)
oot = pd.read_parquet(WORK / "dyn_oot.parquet", columns=["code", "t", "k", D1, D4])
dyn = dyn.merge(oot, on=["code", "t", "k"], how="inner")                       # 只看滚动外推测试期 2019—2023
dyn["one"] = (dyn[ONE_HOP].to_numpy() > 0).any(1)
dyn["multi"] = (dyn[MULTI].to_numpy() > 0).any(1)
dyn["grp"] = np.where(dyn.one, "一跳关联方出事", np.where(dyn.multi, "仅多层级关联方出事", "无关联信号"))

rows = []
for g, d in dyn.groupby("grp"):
    k, n = int(d.y6.sum()), len(d)
    lo, hi = wilson(k, n)
    rows.append({"信号层级": g, "企业-月": n, "6个月内违约": k, "违约率": k / n, "CI_lo": lo, "CI_hi": hi})
lvl = pd.DataFrame(rows).sort_values("违约率", ascending=False)
lvl.to_csv(OUT / "signal_levels.csv", index=False, encoding="utf-8-sig")
print(lvl.assign(违约率=lambda d: (d.违约率 * 100).round(2)).to_string(index=False))

# 违约事件视角：违约前 6 个月内，哪一层能看见
ev = dyn[dyn.def_date.notna() & (dyn.def_date > dyn.S) & (dyn.def_date <= dyn.S + pd.DateOffset(months=6))]
g = ev.groupby(["code", "def_date"]).agg(one=("one", "any"), multi=("multi", "any"))
n_ev = len(g)
n_one = int(g.one.sum())
n_multi_only = int((~g.one & g.multi).sum())
res = {"违约事件数": n_ev, "一跳可见": n_one, "仅多层级可见": n_multi_only, "无关联信号": int((~g.one & ~g.multi).sum()),
       "一跳可见占比": n_one / n_ev, "仅多层级可见占比": n_multi_only / n_ev,
       "多层级使可见违约增加": n_multi_only / max(n_one, 1),
       "分层违约率": {r["信号层级"]: {"n": r["企业-月"], "rate": r["违约率"], "lo": r["CI_lo"], "hi": r["CI_hi"]} for _, r in lvl.iterrows()}}
print(f"[多层级] 违约事件 {n_ev}：一跳可见 {n_one}（{n_one / n_ev:.1%}），仅多层级可见 {n_multi_only}（{n_multi_only / n_ev:.1%}）")

# ---------------------------------------------------------------- 2. 预警名单容量权衡
months = dyn.S.nunique()
cap = []
for q in [0.99, 0.97, 0.95, 0.90]:
    for name, col in [("D4 +多层级时序", D4), ("D1 自身", D1)]:
        flag = dyn.groupby("S")[col].rank(pct=True) >= q
        f = dyn[flag]
        e = dyn[dyn.def_date.notna() & (dyn.def_date > dyn.S) & (dyn.def_date <= dyn.S + pd.DateOffset(months=6))].assign(flag=flag)
        grp = e.groupby(["code", "def_date"])
        hit = grp.flag.any()
        first = e[e.flag].groupby(["code", "def_date"]).S.min()
        lead = (first.index.get_level_values(1) - first.values).days / 30.4375
        cap.append({"模型": name, "名单比例": 1 - q, "每月名单家数": len(f) / months, "名单6个月违约率": f.y6.mean(),
                    "覆盖违约比例": hit.mean(), "覆盖违约数": int(hit.sum()), "违约事件数": int(len(hit)),
                    "提前月数中位数": float(np.median(lead)) if len(lead) else np.nan})
cap = pd.DataFrame(cap)
cap["相对基准倍数"] = cap["名单6个月违约率"] / dyn.y6.mean()
cap.to_csv(OUT / "alert_capacity.csv", index=False, encoding="utf-8-sig")
print(cap.round(4).to_string(index=False))

# ---------------------------------------------------------------- 3. 自身对照（SCCS）
P = pd.read_parquet(WORK / "panel.parquet")
E = pd.read_parquet(WORK / "edges.parquet")
parties = pd.read_parquet(WORK / "lit_parties.parquet")
cases = pd.read_parquet(WORK / "lit_cases.parquet")
pev = parties[(parties.side == "def") & parties.ctype.isin(["fin", "trade"])].copy()
pev["node"] = np.where(pev.name_code.notna(), "C:" + pev.name_code.astype(str), "E:" + pev.name)
own = cases[cases.self_def & cases.ctype.isin(["fin", "trade"])][["case_id", "code", "date"]].assign(node=lambda d: "C:" + d.code)
EV = pd.concat([pev[["node", "date", "case_id"]], own[["node", "date", "case_id"]]], ignore_index=True).drop_duplicates()
pc = pd.concat([cases[["case_id", "code"]], parties[parties.name_code.notna()][["case_id", "name_code"]].rename(columns={"name_code": "code"})])
pc_set = set(zip(pc.case_id, pc.code))

recs = []
for t in YEARS:
    D, Dend = decision_date(t), decision_date(t) + pd.DateOffset(months=12)
    Et = E[(E.t == t) & E.rel.isin(["holder", "codef", "debtor"])]
    Et = Et[Et.src.str[2:].isin(P[P.t == t].code)]
    w = EV[(EV.date > D) & (EV.date <= Dend)]
    m = Et[["src", "dst"]].drop_duplicates().merge(w, left_on="dst", right_on="node")
    m["code"] = m.src.str[2:]
    m = m[[(c, k) not in pc_set for c, k in zip(m.case_id, m.code)]]
    trig = m.groupby("code").date.min()                                        # 不要求早于违约：自身对照要完整观测窗
    s = P[(P.t == t) & (P.y == 1)][["code", "event_date"]].copy()
    s["trig"] = s.code.map(trig)
    s = s[s.trig.notna() & s.event_date.notna() & (s.event_date > D) & (s.event_date <= Dend)]
    s["T_pre"] = (s.trig - D).dt.days / 30.4375
    s["T_post"] = (Dend - s.trig).dt.days / 30.4375
    s["post"] = (s.event_date > s.trig).astype(int)
    recs.append(s)
S = pd.concat(recs, ignore_index=True)


def sccs_irr(d):
    """条件泊松似然：每个病例落在暴露期的概率 = ρ·T_post /(ρ·T_post + T_pre)，解 ρ。"""
    n_post = d.post.sum()
    f = lambda r: (r * d.T_post / (r * d.T_post + d.T_pre)).sum() - n_post
    if n_post == 0 or n_post == len(d):
        return np.nan
    return brentq(f, 1e-4, 1e4)


irr = sccs_irr(S)
bs = []
for _ in range(2000):
    b = S.sample(len(S), replace=True, random_state=int(rng.integers(1e9)))
    v = sccs_irr(b)
    if np.isfinite(v):
        bs.append(v)
sccs = {"病例数(既触发又违约)": int(len(S)), "违约发生在触发后": int(S.post.sum()),
        "按暴露时长的期望": float((S.T_post / (S.T_pre + S.T_post)).sum()),
        "IRR": float(irr), "CI_lo": float(np.percentile(bs, 2.5)), "CI_hi": float(np.percentile(bs, 97.5)),
        "暴露期时间占比中位数": float((S.T_post / (S.T_pre + S.T_post)).median())}
print("[自身对照]", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sccs.items()})

# ---------------------------------------------------------------- 3b. 企业内比较（按企业分层的 MH 优势比）
# 月度面板在企业首次违约后终止，所以"违约引发关联方被诉"的级联事件不会进入暴露变量，比 SCCS 干净。
mh = dyn[["code", "one", "y6"]].copy()
tab = mh.groupby(["code", "one"]).y6.agg(["sum", "count"]).unstack(fill_value=0)
a = tab[("sum", True)].values.astype(float)                                    # 暴露且违约
b = (tab[("count", True)] - tab[("sum", True)]).values.astype(float)
c = tab[("sum", False)].values.astype(float)                                   # 未暴露且违约
d = (tab[("count", False)] - tab[("sum", False)]).values.astype(float)
n = a + b + c + d


def mh_or(a, b, c, d, n):
    num, den = (a * d / n).sum(), (b * c / n).sum()
    return num / den if den > 0 else np.nan


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
    bs = [mh_or(*[v[i] for v in (a, b, c, d, n)]) for i in (rng.integers(0, len(a), len(a)) for _ in range(1000))]
    return {"企业数": int(len(a)), "有暴露也有未暴露月份的企业": int((((a + b) > 0) & ((c + d) > 0)).sum()),
            "暴露月违约": int(a.sum()), "未暴露月违约": int(c.sum()), "暴露月数": int((a + b).sum()),
            "MH优势比(按企业分层)": float(point), "CI_lo": float(np.nanpercentile(bs, 2.5)), "CI_hi": float(np.nanpercentile(bs, 97.5))}


within = within_firm(dyn[["code", "one", "y6"]], "one")
within_multi = within_firm(dyn.loc[~dyn.one, ["code", "multi", "y6"]], "multi")   # 只在没有一跳信号的月份里比较多层级
print("[企业内比较·一跳]", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in within.items()})
print("[企业内比较·仅多层级]", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in within_multi.items()})

# ---------------------------------------------------------------- 4. 单次切分 vs 滚动外推
pred = pd.read_parquet(WORK / "predictions.parquet")[["code", "t", "y", "gbm::M1 自身"]]
roll = pd.read_parquet(WORK / "oot_rolling.parquet")[["code", "t", "y", "M1 自身"]]
sv = []
for t in TEST_YEARS + VALID_YEARS + [2019, 2020]:
    a, b = pred[pred.t == t], roll[roll.t == t]
    if not len(b):
        continue
    sv.append({"观察年": t, "单次切分AUC": roc_auc_score(a.y, a["gbm::M1 自身"]), "滚动外推AUC": roc_auc_score(b.y, b["M1 自身"]),
               "单次切分中该年角色": "训练" if t in TRAIN_YEARS else ("验证" if t in VALID_YEARS else "测试")})
sv = pd.DataFrame(sv).sort_values("观察年").drop_duplicates("观察年")
sv.to_csv(OUT / "split_vs_rolling.csv", index=False, encoding="utf-8-sig")
print(sv.round(3).to_string(index=False))

res["测试期月份数"] = int(months)
res["容量"] = cap.to_dict("records")
res["自身对照"] = sccs
res["企业内比较"] = within
res["企业内比较_仅多层级"] = within_multi
res["单次切分对比"] = sv.to_dict("records")
json.dump(res, open(OUT / "multihop_capacity.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("[done]", OUT / "multihop_capacity.json")
