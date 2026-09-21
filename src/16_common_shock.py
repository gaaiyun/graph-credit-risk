"""共同冲击稳健性：
A. 线性概率模型，控制行业×年份固定效应（吸收同一行业同一年份的共同冲击），公司聚类标准误；
B. 贷中预警率比按行业×年份分层的 Mantel-Haenszel 估计，按企业聚类 bootstrap 95% CI。"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from common import *

P = pd.read_parquet(WORK / "panel.parquet")
rel_lab = {"sup": "供应商", "cus": "客户", "holder": "股东", "invest": "对外投资", "ctrl": "同一控制", "person": "同一关键人员",
           "affil": "联营/合营/子公司", "rp_other": "其他关联方", "codef": "共同被告", "debtor": "债务人"}
ctrl = ["lev", "roa", "cfo_to_debt", "cash_ratio", "log_ta", "impair_to_rev", "own_trade", "own_other_def", "own_viol", "firm_age", "soe"]
X = P[ctrl].astype(float)
X = X.clip(X.quantile(0.01), X.quantile(0.99), axis=1)
X = X.fillna(X.median())
X = (X - X.mean()) / X.std()
R = pd.DataFrame({f"risky_{r}": (P[f"nb_{r}_risky"] > 0).astype(float) for r in rel_lab})
cell = P.sw_l1.astype(str) + "_" + P.t.astype(str)
# 组内去均值（Frisch-Waugh）吸收行业×年份固定效应，避免 248 个哑变量
Z = pd.concat([R, X], axis=1)
Zd = Z - Z.groupby(cell.values).transform("mean")
yd = P.y - P.y.groupby(cell.values).transform("mean")
ols = sm.OLS(yd.values, Zd.values).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(P.code)[0]})
base = P.y.mean()
rows = []
for i, (r, lab) in enumerate(rel_lab.items()):
    b, se = ols.params[i], ols.bse[i]
    rows.append({"关系": lab, "key": r, "系数(百分点)": 100 * b, "CI_lo": 100 * (b - 1.96 * se), "CI_hi": 100 * (b + 1.96 * se),
                 "相对基准违约率": b / base})
lpm = pd.DataFrame(rows)
lpm.to_csv(OUT / "lpm_indyear.csv", index=False, encoding="utf-8-sig")
print(f"基准违约率 {base:.4f}；行业×年份单元 {cell.nunique()} 个")
print(lpm.round(3).to_string(index=False))

# B. MH 率比
S = pd.read_parquet(WORK / "monitor_detail.parquet").merge(P[["code", "t", "sw_l1"]], on=["code", "t"], how="left")
S["cell"] = S.sw_l1.astype(str) + "_" + S.t.astype(str)


def mh(d):
    g = d.groupby("cell").agg(a=("post_def", "sum"), Pm=("post_m", "sum"), b=("pre_def", "sum"), Qm=("pre_m", "sum"))
    T = g.Pm + g.Qm
    num = (g.a * g.Qm / T).sum()
    den = (g.b * g.Pm / T).sum()
    return num / den if den > 0 else np.nan


est = mh(S)
rng = np.random.default_rng(21)
firms = S.code.unique()
idx = {c: np.where(S.code.values == c)[0] for c in firms}
bs = []
for _ in range(1000):
    pick = rng.choice(firms, len(firms))
    rr = mh(S.iloc[np.concatenate([idx[c] for c in pick])])
    if np.isfinite(rr):
        bs.append(rr)
res = {"MH率比(行业×年份分层)": float(est), "CI_lo": float(np.percentile(bs, 2.5)), "CI_hi": float(np.percentile(bs, 97.5)),
       "分层数": int(S.cell.nunique())}
print(res)
json.dump({"mh_monitor": res}, open(OUT / "common_shock.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
