"""应用验证与机制分析：
A. 经营类客户情景：拿掉财务报表（小微/经营类客户常无可信报表），比较 无报表基线 vs 无报表+图谱
B. 稳健性：剔除诉讼衍生边（共同被告/债务人）后图谱增量是否仍在
C. 传导强度分解：分关系类型的风险邻居 → 违约 的 Logit（控制自身财务、年份与行业固定效应，公司聚类标准误）
D. 业务指标：固定通过率下的坏账率、换入换出
E. 预警提前期：违约企业的首个关联方风险事件比自身违约早多少个月
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import importlib
from common import *

M = importlib.import_module("04_model")
P, SETS, te, va = M.P, M.SETS, M.te, M.va
pred = pd.read_parquet(WORK / "predictions.parquet")
out = {}

# ------------------------------------------------ A. 无财务报表情景 + B. 剔除诉讼衍生边
LIT_DERIVED = [c for c in M.HOP1 + M.MULTI if "codef" in c or "debtor" in c or "creditor" in c]
scen = {
    "S0 无报表基线": M.OWN_EV,
    "S1 无报表+图谱": M.OWN_EV + M.HOP1 + M.MULTI + M.STRUCT,
    "S2 无报表+图谱+同业": M.OWN_EV + M.HOP1 + M.MULTI + M.STRUCT + M.COMP,
    "R1 M4剔除诉讼衍生边": [c for c in SETS["M4 +结构位置"] if c not in LIT_DERIVED],
    "R2 M1+同业竞争": SETS["M1 自身"] + M.COMP,
}
yt = P.loc[te.index, "y"].values
rows = []
for name, cols in scen.items():
    p, _, _ = M.fit_lgb(cols)
    pred[f"gbm::{name}"] = p
    r = {"模型": name, "特征数": len(cols), **M.metrics(yt, p[te.index])}
    rows.append(r)
    print(f"{name:16s} test AUC={r['AUC']:.4f} KS={r['KS']:.4f} AP={r['AP']:.4f} cap10={r['Top10%捕获率']:.3f}", flush=True)
rng = np.random.default_rng(1)
pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
pairs = [("S1 无报表+图谱", "S0 无报表基线"), ("S2 无报表+图谱+同业", "S0 无报表基线"),
         ("R1 M4剔除诉讼衍生边", "M1 自身"), ("R2 M1+同业竞争", "M1 自身")]
d = {pr: [] for pr in pairs}
for b in range(2000):
    idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
    for a_, b_ in pairs:
        pa = pred.loc[te.index, f"gbm::{a_}"].values[idx]
        pb = pred.loc[te.index, f"gbm::{b_}"].values[idx]
        d[(a_, b_)].append(roc_auc_score(yt[idx], pa) - roc_auc_score(yt[idx], pb))
sc = pd.DataFrame(rows)
for a_, b_ in pairs:
    x = np.array(d[(a_, b_)])
    sc.loc[sc.模型 == a_, "对照"] = b_
    sc.loc[sc.模型 == a_, "ΔAUC"] = x.mean()
    sc.loc[sc.模型 == a_, "ΔAUC_lo"] = np.percentile(x, 2.5)
    sc.loc[sc.模型 == a_, "ΔAUC_hi"] = np.percentile(x, 97.5)
    sc.loc[sc.模型 == a_, "P(Δ≤0)"] = (x <= 0).mean()
sc.to_csv(OUT / "scenario.csv", index=False, encoding="utf-8-sig")
print(sc.round(4).to_string(index=False))

# ------------------------------------------------ C. 传导强度分解（Logit，公司聚类）
ctrl = ["lev", "roa", "cfo_to_debt", "cash_ratio", "log_ta", "impair_to_rev", "own_trade", "own_other_def", "own_viol", "firm_age", "soe"]
rel_lab = {"sup": "供应商", "cus": "客户", "holder": "股东", "invest": "对外投资", "ctrl": "同一控制", "person": "同一关键人员",
           "affil": "联营/合营/子公司", "rp_other": "其他关联方", "codef": "共同被告(连带)", "debtor": "债务人(应收)"}
D = P.copy()
for r in rel_lab:
    D[f"risky_{r}"] = (D[f"nb_{r}_risky"] > 0).astype(float)
D["peer_fin_rate_pp"] = D.peer_fin_rate * 100
D["rel_rev_growth_c"] = D.rel_rev_growth.clip(-2, 2)
X = D[ctrl].astype(float)
X = X.clip(X.quantile(0.01), X.quantile(0.99), axis=1)
X = X.fillna(X.median())
X = (X - X.mean()) / X.std()
Z = pd.concat([X, D[[f"risky_{r}" for r in rel_lab] + ["peer_fin_rate_pp", "rel_rev_growth_c"]].fillna(0),
               pd.get_dummies(D.t, prefix="yr", drop_first=True, dtype=float),
               pd.get_dummies(D.sw_l1, prefix="ind", drop_first=True, dtype=float)], axis=1)
Z = sm.add_constant(Z)
keep = Z.columns[(Z.std() > 0) | (Z.columns == "const")]
lg = sm.Logit(D.y.values, Z[keep]).fit(disp=0, maxiter=500, cov_type="cluster", cov_kwds={"groups": pd.factorize(D.code)[0]})
coef = []
for r, lab in rel_lab.items():
    c = f"risky_{r}"
    if c in lg.params:
        b, se = lg.params[c], lg.bse[c]
        coef.append({"关系": lab, "key": r, "OR": np.exp(b), "OR_lo": np.exp(b - 1.96 * se), "OR_hi": np.exp(b + 1.96 * se),
                     "p": lg.pvalues[c], "覆盖率": D[c].mean()})
for c, lab in [("peer_fin_rate_pp", "同业违约率(每+1pp)"), ("rel_rev_growth_c", "相对同业营收增速")]:
    b, se = lg.params[c], lg.bse[c]
    coef.append({"关系": lab, "key": c, "OR": np.exp(b), "OR_lo": np.exp(b - 1.96 * se), "OR_hi": np.exp(b + 1.96 * se),
                 "p": lg.pvalues[c], "覆盖率": np.nan})
coef = pd.DataFrame(coef)
coef.to_csv(OUT / "contagion_or.csv", index=False, encoding="utf-8-sig")
print(coef.round(3).to_string(index=False))
print("Logit N=", int(lg.nobs), " pseudo R2=", round(lg.prsquared, 3))

# ------------------------------------------------ D. 业务指标：固定通过率
biz = []
T = pred.loc[te.index]
for q in [0.05, 0.10, 0.20]:
    for name in ["M1 自身", "M4 +结构位置", "S0 无报表基线", "S1 无报表+图谱"]:
        s = T[f"gbm::{name}"].values
        cut = np.quantile(s, 1 - q)
        rej = s >= cut
        biz.append({"拒绝比例": q, "模型": name, "拒绝中坏样本": int(T.y.values[rej].sum()),
                    "通过样本坏账率": T.y.values[~rej].mean(), "坏样本拦截率": T.y.values[rej].sum() / T.y.sum()})
biz = pd.DataFrame(biz)
base_bad = T.y.mean()
biz["坏账率降幅"] = 1 - biz.通过样本坏账率 / base_bad
biz.to_csv(OUT / "business.csv", index=False, encoding="utf-8-sig")
print(f"测试期整体坏账率 {base_bad:.4%}")
print(biz.round(4).to_string(index=False))
swap = {}
for a_, b_ in [("M4 +结构位置", "M1 自身"), ("S1 无报表+图谱", "S0 无报表基线")]:
    sa, sb = T[f"gbm::{a_}"].values, T[f"gbm::{b_}"].values
    ra, rb = sa >= np.quantile(sa, 0.9), sb >= np.quantile(sb, 0.9)
    swap[a_] = {"仅新模型拦截的坏样本": int((ra & ~rb & (T.y.values == 1)).sum()),
                "仅旧模型拦截的坏样本": int((~ra & rb & (T.y.values == 1)).sum()),
                "换入好样本": int((~ra & rb & (T.y.values == 0)).sum())}
print("换入换出（拒绝 10%）:", swap)

# ------------------------------------------------ E. 预警提前期
E = pd.read_parquet(WORK / "edges.parquet")
parties = pd.read_parquet(WORK / "lit_parties.parquet")
cases = pd.read_parquet(WORK / "lit_cases.parquet")
ev_nodes = parties[(parties.side == "def") & parties.ctype.isin(["fin", "trade"])].copy()
ev_nodes = ev_nodes[ev_nodes.name_code.isna() | (ev_nodes.name_code != ev_nodes.code)]
ev_nodes["node"] = np.where(ev_nodes.name_code.notna(), "C:" + ev_nodes.name_code.astype(str), "E:" + ev_nodes.name)
own_ev = cases[cases.self_def & cases.ctype.isin(["fin", "trade"])][["code", "date"]].assign(node=lambda x: "C:" + x.code)
node_ev = pd.concat([ev_nodes[["node", "date", "case_id"]], own_ev[["node", "date"]]], ignore_index=True)
lead = []
dflt = P[P.y == 1]
for r in dflt.itertuples():
    Dd = decision_date(r.t)
    nb = E[(E.t == r.t) & (E.src == "C:" + r.code) & ~E.rel.isin(["codef"])].dst.unique()
    ne = node_ev[node_ev.node.isin(nb) & (node_ev.date <= Dd) & (node_ev.date > Dd - pd.DateOffset(months=LOOKBACK_M))]
    own_w = cases[(cases.code == r.code) & cases.self_def & (cases.date <= Dd) & (cases.date > Dd - pd.DateOffset(months=LOOKBACK_M))]
    lead.append({"code": r.code, "t": r.t, "event_date": r.event_date,
                 "graph_first": ne.date.min() if len(ne) else pd.NaT, "own_first": own_w.date.min() if len(own_w) else pd.NaT,
                 "n_nb_events": len(ne)})
lead = pd.DataFrame(lead)
lead["lead_graph_m"] = (lead.event_date - lead.graph_first).dt.days / 30.4375
lead["lead_own_m"] = (lead.event_date - lead.own_first).dt.days / 30.4375
# 对照：非违约企业中有关联方风险信号的比例（特异性）
nondf = P[P.y == 0]
has_sig = (nondf[[f"nb_{r}_risky" for r in rel_lab if r != "codef"]].sum(1) > 0).mean()
has_sig_d = (dflt[[f"nb_{r}_risky" for r in rel_lab if r != "codef"]].sum(1) > 0).mean()
le = {"违约企业数": len(lead), "违约前已有关联方风险信号占比": float(lead.graph_first.notna().mean()),
      "非违约企业有关联方风险信号占比": float(has_sig), "违约企业(特征口径)有信号占比": float(has_sig_d),
      "关联方信号领先违约 中位数(月)": float(lead.lead_graph_m.median()),
      "自身诉讼信号领先违约 中位数(月)": float(lead.lead_own_m.median()),
      "只有关联方信号、无自身诉讼信号的违约企业占比": float((lead.graph_first.notna() & lead.own_first.isna()).mean())}
print(json.dumps(le, ensure_ascii=False, indent=1))
lead.to_csv(OUT / "lead_time.csv", index=False, encoding="utf-8-sig")
json.dump({"lead": le, "swap": swap}, open(OUT / "analysis_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pred.to_parquet(WORK / "predictions.parquet", index=False)
